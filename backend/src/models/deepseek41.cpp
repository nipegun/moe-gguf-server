#include "models.h"

#include "gguf.h"
#include "mgs-impl.h"
#include "mgs-kv-cache-dsv41.h"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cmath>
#include <cstring>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#ifndef _WIN32
#include <fcntl.h>
#include <unistd.h>
#endif

// DeepSeek-V4.1 (Flash)
//
// Shares the building blocks of DeepSeek-V4 (hyper-connection residual stream with Sinkhorn mixing,
// low-rank q, grouped low-rank output, attention sinks, sqrtsoftplus MoE with a SwiGLU clamp) and
// differs in four ways:
//
// 1. The hyper-connection coefficients lag by one sublayer: a sublayer computes the pre mix that the
//    NEXT sublayer collapses the copies with. Layer 0 collapses with a one-hot on the first copy and
//    the last FFN pre mix collapses the copies before the output head, so there is no HC head.
// 2. Only a few "KV source" layers compress KV: ratio 2 is a softmax-gated pooling of 2 tokens without
//    overlap, ratio 1 a plain projection. The layers after a source read its rows. The index keys are
//    derived from the pooled latent before RoPE and only the index key owners write them; the other
//    index sources score their own queries against those keys and the remaining compressed layers
//    reuse the top-k of the most recent index source. The candidate source additionally keeps its
//    best blocks of rows, and the later index sources only choose inside them.
// 3. There is no extra RMS norm on q after wq_b.
// 4. Engram layers add an n-gram hash lookup into the residual stream before the block. The hash
//    runs on the host and the table rows are read from the model file on demand.
//
// Two GGUF layouts declare this architecture. The DwarfStar tools write the keys of the HF config
// (translated by fMgsDeepseek41AdaptMetadata) and an I8 engram table of E4M3 values with E8M0
// scales; the converter of the ggml ecosystem writes the standard keys and an engram table in a
// regular ggml type. Both name the other tensors alike, apart from the engram projections.

static float fDsv41RopeAttnFactor(float pFreqScale, float pExtFactor) {
  if (pExtFactor == 0.0f) {
    return 1.0f;
  }

  // ggml scales YaRN rotations by its magnitude factor, the reference does not: cancel it
  return 1.0f / (1.0f + 0.1f * logf(1.0f / pFreqScale));
}

//
// metadata
//

static double fDsv41GgufNumber(const gguf_context *pMeta, int64_t pId) {
  switch (gguf_get_kv_type(pMeta, pId)) {
  case GGUF_TYPE_UINT8:
    return gguf_get_val_u8(pMeta, pId);
  case GGUF_TYPE_INT8:
    return gguf_get_val_i8(pMeta, pId);
  case GGUF_TYPE_UINT16:
    return gguf_get_val_u16(pMeta, pId);
  case GGUF_TYPE_INT16:
    return gguf_get_val_i16(pMeta, pId);
  case GGUF_TYPE_UINT32:
    return gguf_get_val_u32(pMeta, pId);
  case GGUF_TYPE_INT32:
    return gguf_get_val_i32(pMeta, pId);
  case GGUF_TYPE_UINT64:
    return (double)gguf_get_val_u64(pMeta, pId);
  case GGUF_TYPE_INT64:
    return (double)gguf_get_val_i64(pMeta, pId);
  case GGUF_TYPE_FLOAT32:
    return gguf_get_val_f32(pMeta, pId);
  case GGUF_TYPE_FLOAT64:
    return gguf_get_val_f64(pMeta, pId);
  case GGUF_TYPE_BOOL:
    return gguf_get_val_bool(pMeta, pId) ? 1.0 : 0.0;
  default:
    throw std::runtime_error(fFormat("GGUF key %s is not a number", gguf_get_key(pMeta, pId)));
  }
}

// reads an integer array of any integer element type
static bool fDsv41GgufU64Array(const gguf_context *pMeta, const std::string &pKey, std::vector<uint64_t> &pRes) {
  const int64_t cId = gguf_find_key(pMeta, pKey.c_str());
  if (cId < 0) {
    return false;
  }
  if (gguf_get_kv_type(pMeta, cId) != GGUF_TYPE_ARRAY) {
    pRes.assign(1, (uint64_t)fDsv41GgufNumber(pMeta, cId));
    return true;
  }

  const size_t cN = gguf_get_arr_n(pMeta, cId);
  const void *vData = gguf_get_arr_data(pMeta, cId);

  pRes.resize(cN);
  for (size_t vI = 0; vI < cN; ++vI) {
    switch (gguf_get_arr_type(pMeta, cId)) {
    case GGUF_TYPE_UINT8:
      pRes[vI] = ((const uint8_t *)vData)[vI];
      break;
    case GGUF_TYPE_INT8:
      pRes[vI] = (uint64_t)((const int8_t *)vData)[vI];
      break;
    case GGUF_TYPE_UINT16:
      pRes[vI] = ((const uint16_t *)vData)[vI];
      break;
    case GGUF_TYPE_INT16:
      pRes[vI] = (uint64_t)((const int16_t *)vData)[vI];
      break;
    case GGUF_TYPE_UINT32:
      pRes[vI] = ((const uint32_t *)vData)[vI];
      break;
    case GGUF_TYPE_INT32:
      pRes[vI] = (uint64_t)((const int32_t *)vData)[vI];
      break;
    case GGUF_TYPE_UINT64:
      pRes[vI] = ((const uint64_t *)vData)[vI];
      break;
    case GGUF_TYPE_INT64:
      pRes[vI] = (uint64_t)((const int64_t *)vData)[vI];
      break;
    default:
      throw std::runtime_error(fFormat("GGUF key %s is not an integer array", pKey.c_str()));
    }
  }

  return true;
}

void fMgsDeepseek41AdaptMetadata(struct gguf_context *pMeta) {
  const std::string cArch = "deepseek41.";

  // standard keys present (or no HF config keys to translate): nothing to do
  if (gguf_find_key(pMeta, (cArch + "block_count").c_str()) >= 0 ||
      gguf_find_key(pMeta, (cArch + "hidden_size").c_str()) < 0) {
    return;
  }

  MGS_LOG_INFO("%s: translating the DeepSeek-V4.1 HF config keys to the standard keys\n", __func__);

  const auto cHas = [&](const std::string &pKey) { return gguf_find_key(pMeta, (cArch + pKey).c_str()) >= 0; };

  const auto cNumber = [&](const std::string &pKey) -> double {
    const int64_t cId = gguf_find_key(pMeta, (cArch + pKey).c_str());
    if (cId < 0) {
      throw std::runtime_error(fFormat("DeepSeek-V4.1 metadata lacks %s%s", cArch.c_str(), pKey.c_str()));
    }
    return fDsv41GgufNumber(pMeta, cId);
  };

  const auto cSetU32 = [&](const std::string &pDst, const std::string &pSrc) {
    if (!cHas(pDst) && cHas(pSrc)) {
      gguf_set_val_u32(pMeta, (cArch + pDst).c_str(), (uint32_t)cNumber(pSrc));
    }
  };

  const auto cSetF32 = [&](const std::string &pDst, const std::string &pSrc) {
    if (!cHas(pDst) && cHas(pSrc)) {
      gguf_set_val_f32(pMeta, (cArch + pDst).c_str(), (float)cNumber(pSrc));
    }
  };

  cSetU32("context_length", "max_position_embeddings");
  cSetU32("embedding_length", "hidden_size");
  cSetU32("block_count", "num_hidden_layers");
  cSetU32("attention.head_count", "num_attention_heads");
  cSetU32("attention.head_count_kv", "num_key_value_heads");
  cSetU32("attention.key_length", "head_dim");
  cSetU32("attention.value_length", "head_dim");
  cSetU32("rope.dimension_count", "qk_rope_head_dim");
  cSetF32("rope.freq_base", "rope_theta");
  cSetF32("attention.layer_norm_rms_epsilon", "rms_norm_eps");

  if (cHas("rope_scaling.factor") && !cHas("rope.scaling.type")) {
    gguf_set_val_str(pMeta, (cArch + "rope.scaling.type").c_str(), "yarn");
  }
  cSetF32("rope.scaling.factor", "rope_scaling.factor");
  cSetU32("rope.scaling.original_context_length", "rope_scaling.original_max_position_embeddings");
  cSetF32("rope.scaling.yarn_beta_fast", "rope_scaling.beta_fast");
  cSetF32("rope.scaling.yarn_beta_slow", "rope_scaling.beta_slow");

  cSetU32("expert_count", "n_routed_experts");
  cSetU32("expert_used_count", "num_experts_per_tok");
  cSetU32("expert_shared_count", "n_shared_experts");
  cSetU32("expert_feed_forward_length", "moe_intermediate_size");
  cSetF32("expert_weights_scale", "routed_scaling_factor");
  if (!cHas("expert_weights_norm") && cHas("norm_topk_prob")) {
    gguf_set_val_bool(pMeta, (cArch + "expert_weights_norm").c_str(), cNumber("norm_topk_prob") != 0.0);
  }
  if (!cHas("expert_gating_func") && cHas("scoring_func")) {
    const std::string cScoring = gguf_get_val_str(pMeta, gguf_find_key(pMeta, (cArch + "scoring_func").c_str()));
    uint32_t vGating = cMgsExpertGatingFuncTypeNone;
    if (cScoring == "softmax") {
      vGating = cMgsExpertGatingFuncTypeSoftmax;
    } else if (cScoring == "sigmoid") {
      vGating = cMgsExpertGatingFuncTypeSigmoid;
    } else if (cScoring == "sqrtsoftplus") {
      vGating = cMgsExpertGatingFuncTypeSqrtSoftplus;
    } else {
      throw std::runtime_error("unsupported DeepSeek-V4.1 scoring function: " + cScoring);
    }
    gguf_set_val_u32(pMeta, (cArch + "expert_gating_func").c_str(), vGating);
  }
  cSetF32("swiglu_clamp_exp", "swiglu_limit");
  cSetF32("swiglu_clamp_shexp", "swiglu_limit");

  cSetU32("attention.q_lora_rank", "q_lora_rank");
  cSetU32("attention.sliding_window", "sliding_window");
  cSetU32("attention.indexer.head_count", "index_n_heads");
  cSetU32("attention.indexer.key_length", "index_head_dim");
  cSetU32("attention.indexer.top_k", "index_topk");
  cSetU32("attention.output_group_count", "o_groups");
  cSetU32("attention.output_lora_rank", "o_lora_rank");
  cSetF32("attention.compress_rope_freq_base", "compress_rope_theta");
  cSetU32("attention.candidate_source_layer_id", "candidate_source_layer_id");
  cSetU32("attention.candidate_topk_blocks", "candidate_topk_blocks");
  cSetU32("attention.candidate_block_size", "candidate_block_size");

  cSetU32("hyper_connection.count", "hc_mult");
  cSetU32("hyper_connection.sinkhorn_iterations", "hc_sinkhorn_iters");
  cSetF32("hyper_connection.epsilon", "hc_eps");

  std::vector<uint64_t> lRatios;
  if (!cHas("attention.compress_ratios") && fDsv41GgufU64Array(pMeta, cArch + "compress_ratios", lRatios)) {
    std::vector<uint32_t> lRatios32(lRatios.begin(), lRatios.end());
    gguf_set_arr_data(pMeta, (cArch + "attention.compress_ratios").c_str(), GGUF_TYPE_UINT32, lRatios32.data(),
                      lRatios32.size());
  }

  // the engram keys mostly share their names; derive the shape and the bucket offsets
  std::vector<uint64_t> lLayers;
  std::vector<uint64_t> lMultipliers;
  std::vector<uint64_t> lPrimes;
  if (fDsv41GgufU64Array(pMeta, cArch + "engram.layer_ids", lLayers) && !lLayers.empty() &&
      fDsv41GgufU64Array(pMeta, cArch + "engram.multipliers", lMultipliers) &&
      fDsv41GgufU64Array(pMeta, cArch + "engram.primes", lPrimes)) {
    const size_t cNLayer = lLayers.size();
    const size_t cNGram = lMultipliers.size() / cNLayer;
    const size_t cNCols = lPrimes.size() / cNLayer;
    if (cNGram < 2 || lMultipliers.size() != cNGram * cNLayer || lPrimes.size() != cNCols * cNLayer ||
        cNCols % (cNGram - 1) != 0) {
      throw std::runtime_error("inconsistent DeepSeek-V4.1 engram metadata");
    }

    if (!cHas("engram.max_ngram_size")) {
      gguf_set_val_u32(pMeta, (cArch + "engram.max_ngram_size").c_str(), (uint32_t)cNGram);
    }
    if (!cHas("engram.head_count")) {
      gguf_set_val_u32(pMeta, (cArch + "engram.head_count").c_str(), (uint32_t)(cNCols / (cNGram - 1)));
    }
    if (!cHas("engram.key_length")) {
      const int64_t cId = gguf_find_key(pMeta, (cArch + "engram.encoding").c_str());
      const std::string cEncoding = cId >= 0 ? gguf_get_val_str(pMeta, cId) : "";
      if (cEncoding != "e4m3_e8m0_32_row264") {
        throw std::runtime_error("unsupported DeepSeek-V4.1 engram encoding: " + cEncoding);
      }
      gguf_set_val_u32(pMeta, (cArch + "engram.key_length").c_str(), 256);
    }
    if (!cHas("engram.offsets")) {
      // every (n-gram, head) bucket starts where the previous one of its layer ended
      std::vector<uint64_t> lOffsets(lPrimes.size());
      for (size_t vL = 0; vL < cNLayer; ++vL) {
        uint64_t vSum = 0;
        for (size_t vB = 0; vB < cNCols; ++vB) {
          lOffsets[vL * cNCols + vB] = vSum;
          vSum += lPrimes[vL * cNCols + vB];
        }
      }
      gguf_set_arr_data(pMeta, (cArch + "engram.offsets").c_str(), GGUF_TYPE_UINT64, lOffsets.data(),
                        lOffsets.size());
    }
  }
}

//
// engram tables
//

// E4M3 (no infinities) to float
static float fDsv41E4m3(uint8_t pByte) {
  const int cExponent = (pByte >> 3) & 15;
  const int cMantissa = pByte & 7;
  const float cValue = cExponent ? ldexpf((float)(8 + cMantissa), cExponent - 10) : ldexpf((float)cMantissa, -9);
  return (pByte & 128) ? -cValue : cValue;
}

// value of every (E8M0 scale, E4M3 code) pair, rounded to BF16 as the reference does
static const std::vector<float> &fDsv41EngramLut() {
  static const std::vector<float> lLut = []() {
    std::vector<float> lRes(256 * 256, 0.0f);
    for (uint32_t vScale = 0; vScale < 256; ++vScale) {
      for (uint32_t vCode = 0; vCode < 256; ++vCode) {
        float vValue = 0.0f;
        // NaN codes and the NaN scale read as zero
        if ((vCode & 127u) != 127u && vScale != 255u) {
          vValue = ldexpf(fDsv41E4m3((uint8_t)vCode), (int)vScale - 127);
          uint32_t vBits;
          memcpy(&vBits, &vValue, sizeof(vBits));
          vBits = (vBits + 0x7fffu + ((vBits >> 16) & 1u)) & 0xffff0000u;
          if ((vBits & 0x7f800000u) == 0x7f800000u) {
            vBits = 0;
          }
          memcpy(&vValue, &vBits, sizeof(vValue));
        }
        lRes[vScale * 256 + vCode] = vValue;
      }
    }
    return lRes;
  }();

  return lLut;
}

#ifndef _WIN32
static bool fDsv41PreadAll(int pFd, void *pDst, size_t pSize, uint64_t pOffset) {
  uint8_t *vDst = (uint8_t *)pDst;
  size_t vDone = 0;
  while (vDone < pSize) {
    const ssize_t cN = pread(pFd, vDst + vDone, pSize - vDone, (off_t)(pOffset + vDone));
    if (cN < 0 && errno == EINTR) {
      continue;
    }
    if (cN <= 0) {
      return false;
    }
    vDone += (size_t)cN;
  }
  return true;
}
#endif

static void fDsv41EngramDecodeRow(const mgs_dsv41_engram_table &pTable, const uint8_t *pSrc, float *pDst) {
  if (pTable.vType == GGML_TYPE_I8) {
    const std::vector<float> &lLut = fDsv41EngramLut();
    const uint8_t *vScales = pSrc + pTable.vRowDim;
    for (int64_t vJ = 0; vJ < pTable.vRowDim; ++vJ) {
      pDst[vJ] = lLut[(size_t)vScales[vJ / 32] * 256 + pSrc[vJ]];
    }
    return;
  }

  if (pTable.vType == GGML_TYPE_F32) {
    memcpy(pDst, pSrc, pTable.vRowDim * sizeof(float));
    return;
  }

  ggml_get_type_traits(pTable.vType)->to_float(pSrc, pDst, pTable.vRowDim);
}

mgs_model_deepseek41::~mgs_model_deepseek41() {
#ifndef _WIN32
  for (auto &vTable : lEngramTables) {
    if (vTable.vFd >= 0) {
      close(vTable.vFd);
      vTable.vFd = -1;
    }
  }
#endif
}

int mgs_model_deepseek41::mEngramIndex(int pIl) const {
  for (size_t vE = 0; vE < lEngramLayerIds.size(); ++vE) {
    if (lEngramLayerIds[vE] == (uint32_t)pIl) {
      return (int)vE;
    }
  }
  return -1;
}

void mgs_model_deepseek41::mEngramReadRows(int pE, const uint32_t *pRows, size_t pCount, float *pOut) const {
  const mgs_dsv41_engram_table &vTable = lEngramTables.at(pE);
  const int64_t cDim = vTable.vRowDim;

  if (vTable.vFd < 0) {
    std::fill(pOut, pOut + pCount * cDim, 0.0f);
    return;
  }

#ifdef _WIN32
  GGML_UNUSED(pRows);
  std::fill(pOut, pOut + pCount * cDim, 0.0f);
#else
  // sort the requests by row so that repeated rows are read once and the reads go forward in the file
  std::vector<std::pair<uint32_t, uint32_t>> lRequests(pCount);
  for (size_t vI = 0; vI < pCount; ++vI) {
    lRequests[vI] = {pRows[vI], (uint32_t)vI};
  }
  std::sort(lRequests.begin(), lRequests.end());

  // random reads of a few hundred bytes are latency bound: overlap them with a few readers
  const size_t cNReaders =
      pCount < 256 ? 1 : std::max<size_t>(1, std::min<size_t>(16, std::thread::hardware_concurrency()));

  std::atomic<bool> vFailed(false);

  const auto cReadPart = [&](size_t pPart) {
    const size_t cBegin = pCount * pPart / cNReaders;
    const size_t cEnd = pCount * (pPart + 1) / cNReaders;
    std::vector<uint8_t> lRaw(vTable.vRowBytes);
    const float *vPrevious = nullptr;

    for (size_t vI = cBegin; vI < cEnd; ++vI) {
      float *vDst = pOut + (size_t)lRequests[vI].second * cDim;
      const uint32_t cRow = lRequests[vI].first;

      if (vI > cBegin && cRow == lRequests[vI - 1].first && vPrevious != nullptr) {
        memcpy(vDst, vPrevious, cDim * sizeof(float));
        continue;
      }

      const uint64_t cOffset = vTable.vOffset + (uint64_t)cRow * vTable.vRowBytes;
      if (cRow >= vTable.vNRows || !fDsv41PreadAll(vTable.vFd, lRaw.data(), vTable.vRowBytes, cOffset)) {
        vFailed = true;
        std::fill(vDst, vDst + cDim, 0.0f);
        vPrevious = nullptr;
        continue;
      }

      fDsv41EngramDecodeRow(vTable, lRaw.data(), vDst);
      vPrevious = vDst;
    }
  };

  std::vector<std::thread> lThreads;
  lThreads.reserve(cNReaders > 0 ? cNReaders - 1 : 0);
  for (size_t vPart = 1; vPart < cNReaders; ++vPart) {
    lThreads.emplace_back(cReadPart, vPart);
  }
  cReadPart(0);
  for (auto &vThread : lThreads) {
    vThread.join();
  }

  if (vFailed) {
    MGS_LOG_ERROR("%s: failed to read some rows of engram table %d: %s\n", __func__, pE, strerror(errno));
  }
#endif
}

// A descriptor of its own for a model file: the loader closes its files after loading, may read with
// O_DIRECT and asks for sequential readahead, while the engram rows are small random reads.
static int fDsv41OpenModelFile(const mgs_model_loader &pMl, uint16_t pIdx) {
#ifdef _WIN32
  GGML_UNUSED(pMl);
  GGML_UNUSED(pIdx);
  throw std::runtime_error("the DeepSeek-V4.1 engram tables are only supported on POSIX systems");
#else
  const int cFd = pMl.vFiles.at(pIdx)->mFileId();

  int vFd = -1;
#ifdef __linux__
  const std::string cPath = "/proc/self/fd/" + std::to_string(cFd);
  vFd = open(cPath.c_str(), O_RDONLY | O_CLOEXEC);
#endif
  if (vFd < 0) {
    vFd = dup(cFd);
  }
  if (vFd < 0) {
    throw std::runtime_error(fFormat("failed to open the model file for the engram tables: %s", strerror(errno)));
  }

#ifdef __linux__
  posix_fadvise(vFd, 0, 0, POSIX_FADV_RANDOM);
#endif

  return vFd;
#endif
}

//
// loading
//

void mgs_model_deepseek41::mLoadArchHparams(mgs_model_loader &pMl) {
  vHparams.vNLayerNextn = 0;

  pMl.mGetKey(cLlmKvAttentionLayernormRmsEps, vHparams.vFNormRmsEps);
  pMl.mGetKey(cLlmKvAttentionQLoraRank, vHparams.vNLoraQ);
  pMl.mGetKey(cLlmKvAttentionSlidingWindow, vHparams.vNSwa);

  pMl.mGetKey(cLlmKvExpertFeedForwardLength, vHparams.vNFfExp);
  pMl.mGetKey(cLlmKvExpertSharedCount, vHparams.vNExpertShared);
  pMl.mGetKey(cLlmKvExpertWeightsScale, vHparams.vExpertWeightsScale);
  pMl.mGetKey(cLlmKvExpertWeightsNorm, vHparams.vExpertWeightsNorm);
  pMl.mGetKey(cLlmKvExpertGatingFunc, vHparams.vExpertGatingFunc);
  if (vHparams.vExpertGatingFunc == cMgsExpertGatingFuncTypeNone) {
    throw std::runtime_error("DeepSeek-V4.1 needs an expert gating function");
  }

  pMl.mGetKeyOrArr(cLlmKvSwigluClampExp, vHparams.aSwigluClampExp, vHparams.vNLayerAll);
  if (!pMl.mGetKeyOrArr(cLlmKvSwigluClampShexp, vHparams.aSwigluClampShexp, vHparams.vNLayerAll, false)) {
    vHparams.aSwigluClampShexp = vHparams.aSwigluClampExp;
  }

  pMl.mGetKey(cLlmKvAttentionIndexerHeadCount, vHparams.vIndexerNHead);
  pMl.mGetKey(cLlmKvAttentionIndexerKeyLength, vHparams.vIndexerHeadSize);
  pMl.mGetKey(cLlmKvAttentionIndexerTopK, vHparams.vIndexerTopK);

  pMl.mGetKey(cLlmKvAttentionOutputGroupCount, vHparams.vDsv4OGroupCount);
  pMl.mGetKey(cLlmKvAttentionOutputLoraRank, vHparams.vDsv4OLoraRank);
  pMl.mGetKey(cLlmKvAttentionCompressRopeFreqBase, vHparams.vDsv4CompressRopeBase);
  pMl.mGetKey(cLlmKvHyperConnectionCount, vHparams.vDsv4HcMult);
  pMl.mGetKey(cLlmKvHyperConnectionSinkhornIterations, vHparams.vDsv4HcSinkhornIters);
  pMl.mGetKey(cLlmKvHyperConnectionEpsilon, vHparams.vDsv4HcEps);

  pMl.mGetKey(cLlmKvRopeScalingYarnBetaFast, vHparams.vYarnBetaFast, false);
  pMl.mGetKey(cLlmKvRopeScalingYarnBetaSlow, vHparams.vYarnBetaSlow, false);

  uint32_t vNCompressRatios = 0;
  pMl.mGetArrN(cLlmKvAttentionCompressRatios, vNCompressRatios);
  if (vNCompressRatios < vHparams.mNLayer()) {
    throw std::runtime_error("DeepSeek-V4.1 compress_ratios is shorter than block_count");
  }
  pMl.mGetArr(cLlmKvAttentionCompressRatios, vHparams.aDsv4CompressRatios);

  // optional: the two-level indexer top-k (the default follows the model once the layer roles are known)
  const std::string cArch = pMl.mGetArchName() + ".";
  int32_t vCandidateSource = -1;
  if (pMl.mGetKey(cArch + "attention.candidate_source_layer_id", vCandidateSource, false)) {
    vHparams.vDsv41CandidateSource = vCandidateSource;
    pMl.mGetKey(cArch + "attention.candidate_topk_blocks", vHparams.vDsv41CandidateTopKBlocks);
    pMl.mGetKey(cArch + "attention.candidate_block_size", vHparams.vDsv41CandidateBlockSize);
  } else {
    vHparams.vDsv41CandidateSource = -2; // derived in mLoadArchTensors
  }

  vHparams.vSwaType = cMgsSwaTypeStandard;
  vHparams.mSetSwaPattern(0);

  vHparams.aDsv41KvSource.fill(-1);
  vHparams.aDsv41IndexKeySource.fill(-1);
  vHparams.aDsv41TopKSource.fill(-1);

  // engram
  const gguf_context *vMeta = pMl.vMetadata;

  std::vector<uint64_t> lLayers;
  if (fDsv41GgufU64Array(vMeta, cArch + "engram.layer_ids", lLayers) && !lLayers.empty()) {
    pMl.mGetKey(cArch + "engram.head_count", vHparams.vEngramNHead);
    pMl.mGetKey(cArch + "engram.key_length", vHparams.vEngramKeyLength);
    pMl.mGetKey(cArch + "engram.max_ngram_size", vHparams.vEngramMaxNgramSize);
    pMl.mGetKey(cArch + "engram.pad_id", vEngramPadId);

    if (vHparams.vEngramNHead == 0 || vHparams.vEngramMaxNgramSize < 2 || vHparams.vEngramKeyLength == 0) {
      throw std::runtime_error("DeepSeek-V4.1 engram needs at least one head, a 2-gram and a key length");
    }

    lEngramLayerIds.clear();
    for (const uint64_t cLayer : lLayers) {
      if (cLayer >= vHparams.mNLayer()) {
        throw std::runtime_error(fFormat("engram layer %u is out of range", (uint32_t)cLayer));
      }
      lEngramLayerIds.push_back((uint32_t)cLayer);
    }

    std::vector<uint64_t> lTokenMap;
    if (!fDsv41GgufU64Array(vMeta, cArch + "engram.multipliers", lEngramMultipliers) ||
        !fDsv41GgufU64Array(vMeta, cArch + "engram.primes", lEngramPrimes) ||
        !fDsv41GgufU64Array(vMeta, cArch + "engram.offsets", lEngramOffsets) ||
        !fDsv41GgufU64Array(vMeta, cArch + "engram.token_map", lTokenMap)) {
      throw std::runtime_error("DeepSeek-V4.1 engram hash constants are missing");
    }

    lEngramTokenMap.assign(lTokenMap.begin(), lTokenMap.end());

    // the hash indexes straight into these, so a short array would read past the end
    const size_t cNLayer = lEngramLayerIds.size();
    const size_t cNCols = (size_t)(vHparams.vEngramMaxNgramSize - 1) * vHparams.vEngramNHead;
    if (lEngramMultipliers.size() != cNLayer * vHparams.vEngramMaxNgramSize) {
      throw std::runtime_error("engram multiplier count does not match layers * ngram size");
    }
    if (lEngramPrimes.size() != cNLayer * cNCols || lEngramOffsets.size() != lEngramPrimes.size()) {
      throw std::runtime_error("engram prime or offset count does not match layers * buckets");
    }
    for (const uint64_t cPrime : lEngramPrimes) {
      if (cPrime == 0) {
        throw std::runtime_error("engram prime of zero would divide by zero in the hash");
      }
    }
  }

  vType = cLlmTypeUnknown;
}

void mgs_model_deepseek41::mLoadArchTensors(mgs_model_loader &pMl) {
  MGS_LOAD_LOCALS;

  const int64_t cQLoraRank = vHparams.vNLoraQ;
  const int64_t cNFfExp = vHparams.vNFfExp;
  const int64_t cNExpertShared = vHparams.vNExpertShared;

  const int64_t cNEmbdHead = vHparams.mNEmbdHeadK();
  const int64_t cOGroups = vHparams.vDsv4OGroupCount;
  const int64_t cOLoraRank = vHparams.vDsv4OLoraRank;
  const int64_t cHcMult = vHparams.vDsv4HcMult;
  const int64_t cHcDim = cHcMult * cNEmbd;
  const int64_t cHcMixDim = (2 + cHcMult) * cHcMult;
  const int64_t cNEmbdIndexer = vHparams.vIndexerHeadSize;
  const int64_t cNIndexerHead = vHparams.vIndexerNHead;

  if (!lEngramLayerIds.empty() && (int64_t)lEngramTokenMap.size() < cNVocab) {
    throw std::runtime_error(fFormat("engram token map has %zu entries, too few for %lld tokens",
                                     lEngramTokenMap.size(), (long long)cNVocab));
  }

  vTokEmbd = mCreateTensor(cTn(cLlmTensorTokenEmbd, "weight"), {cNEmbd, cNVocab}, 0);
  vOutputNorm = mCreateTensor(cTn(cLlmTensorOutputNorm, "weight"), {cNEmbd}, 0);
  vOutput = mCreateTensor(cTn(cLlmTensorOutput, "weight"), {cNEmbd, cNVocab}, 0);

  lEngramTables.assign(lEngramLayerIds.size(), mgs_dsv41_engram_table());

  for (int vI = 0; vI < cNLayer; ++vI) {
    auto &vLayer = lLayers[vI];

    vLayer.vAttnNorm = mCreateTensor(cTn(cLlmTensorAttnNorm, "weight", vI), {cNEmbd}, 0);
    vLayer.vAttnSinks = mCreateTensor(cTn(cLlmTensorAttnSinks, "weight", vI), {cNHead}, 0);
    vLayer.vWqA = mCreateTensor(cTn(cLlmTensorAttnQA, "weight", vI), {cNEmbd, cQLoraRank}, 0);
    vLayer.vAttnQANorm = mCreateTensor(cTn(cLlmTensorAttnQANorm, "weight", vI), {cQLoraRank}, 0);
    vLayer.vWqB = mCreateTensor(cTn(cLlmTensorAttnQB, "weight", vI), {cQLoraRank, cNHead * cNEmbdHead}, 0);
    vLayer.vWkv = mCreateTensor(cTn(cLlmTensorAttnKv, "weight", vI), {cNEmbd, cNEmbdHead}, 0);
    vLayer.vAttnKvNorm = mCreateTensor(cTn(cLlmTensorAttnKvNorm, "weight", vI), {cNEmbdHead}, 0);
    // the file lays wo_a out as (n_head * n_embd_head / o_groups, o_lora_rank * o_groups):
    // reshape at load and keep the graph free of it
    vLayer.vWoA = mCreateTensor(cTn(cLlmTensorAttnOutA, "weight", vI),
                                {cNHead * cNEmbdHead / cOGroups, cOLoraRank, cOGroups}, cTensorAllowReshape);
    vLayer.vWoB = mCreateTensor(cTn(cLlmTensorAttnOutB, "weight", vI), {cOGroups * cOLoraRank, cNEmbd}, 0);

    vLayer.vHcAttnFn = mCreateTensor(cTn(cLlmTensorHcAttnFn, "weight", vI), {cHcDim, cHcMixDim}, 0);
    vLayer.vHcAttnBase = mCreateTensor(cTn(cLlmTensorHcAttnBase, "weight", vI), {cHcMixDim}, 0);
    vLayer.vHcAttnScale = mCreateTensor(cTn(cLlmTensorHcAttnScale, "weight", vI), {3}, 0);
    vLayer.vHcFfnFn = mCreateTensor(cTn(cLlmTensorHcFfnFn, "weight", vI), {cHcDim, cHcMixDim}, 0);
    vLayer.vHcFfnBase = mCreateTensor(cTn(cLlmTensorHcFfnBase, "weight", vI), {cHcMixDim}, 0);
    vLayer.vHcFfnScale = mCreateTensor(cTn(cLlmTensorHcFfnScale, "weight", vI), {3}, 0);

    // only the KV source layers carry a compressor, and only those that pool more than one token a gate
    vLayer.vAttnCompWkv =
        mCreateTensor(cTn(cLlmTensorAttnCompressorWkv, "weight", vI), {cNEmbd, cNEmbdHead}, cTensorNotRequired);
    vLayer.vAttnCompWgate =
        mCreateTensor(cTn(cLlmTensorAttnCompressorWgate, "weight", vI), {cNEmbd, cNEmbdHead}, cTensorNotRequired);
    vLayer.vAttnCompNorm =
        mCreateTensor(cTn(cLlmTensorAttnCompressorNorm, "weight", vI), {cNEmbdHead}, cTensorNotRequired);

    // an index source scores queries against shared index keys, which only a key owner derives
    vLayer.vIndexerAttnQB = mCreateTensor(cTn(cLlmTensorIndexerAttnQB, "weight", vI),
                                          {cQLoraRank, cNIndexerHead * cNEmbdIndexer}, cTensorNotRequired);
    vLayer.vIndexerProj =
        mCreateTensor(cTn(cLlmTensorIndexerProj, "weight", vI), {cNEmbd, cNIndexerHead}, cTensorNotRequired);
    vLayer.vIndexerAttnK =
        mCreateTensor(cTn(cLlmTensorIndexerAttnK, "weight", vI), {cNEmbdHead, cNEmbdIndexer}, cTensorNotRequired);
    vLayer.vIndexerKNorm =
        mCreateTensor(cTn(cLlmTensorIndexerKNorm, "weight", vI), {cNEmbdIndexer}, cTensorNotRequired);

    vLayer.vFfnGateInp = mCreateTensor(cTn(cLlmTensorFfnGateInp, "weight", vI), {cNEmbd, cNExpert}, 0);
    vLayer.vFfnExpProbsB = mCreateTensor(cTn(cLlmTensorFfnExpProbsB, "bias", vI), {cNExpert}, 0);
    vLayer.vFfnExpProbsBVl =
        mCreateTensor(cTn(cLlmTensorFfnExpProbsBVl, "bias", vI), {cNExpert}, cTensorNotRequired);
    vLayer.vFfnNorm = mCreateTensor(cTn(cLlmTensorFfnNorm, "weight", vI), {cNEmbd}, 0);

    mCreateTensorGateUpExps(vLayer, vI, cNEmbd, cNFfExp, cNExpert, 0);
    vLayer.vFfnDownExps = mCreateTensor(cTn(cLlmTensorFfnDownExps, "weight", vI), {cNFfExp, cNEmbd, cNExpert}, 0);

    vLayer.vFfnGateShexp =
        mCreateTensor(cTn(cLlmTensorFfnGateShexp, "weight", vI), {cNEmbd, cNFfExp * cNExpertShared}, 0);
    vLayer.vFfnDownShexp =
        mCreateTensor(cTn(cLlmTensorFfnDownShexp, "weight", vI), {cNFfExp * cNExpertShared, cNEmbd}, 0);
    vLayer.vFfnUpShexp = mCreateTensor(cTn(cLlmTensorFfnUpShexp, "weight", vI), {cNEmbd, cNFfExp * cNExpertShared}, 0);

    const int cE = mEngramIndex(vI);
    if (cE >= 0) {
      const int64_t cNCols = (int64_t)(vHparams.vEngramMaxNgramSize - 1) * vHparams.vEngramNHead;
      const int64_t cKeyLength = vHparams.vEngramKeyLength;

      // The table has hundreds of millions of rows and is far too large to load, but each token
      // touches only n_cols of them: they are read from the file on demand.
      const std::string cEmbdName = cTn(cLlmTensorEngramEmbd, "weight", vI).mStr();
      const auto *vWeight = pMl.mGetWeight(cEmbdName.c_str());
      mgs_dsv41_engram_table &vTable = lEngramTables[cE];
      vTable.vRowDim = cKeyLength;
      vTable.vType = GGML_TYPE_F32;

      if (vWeight == nullptr && !pMl.vFiles.empty()) {
        throw std::runtime_error(fFormat("%s is missing", cEmbdName.c_str()));
      }

      if (vWeight != nullptr) {
        const ggml_tensor *vMeta = vWeight->vTensor;
        vTable.vNRows = vMeta->ne[1];
        vTable.vType = vMeta->type;

        if (vMeta->type == GGML_TYPE_I8) {
          // DwarfStar: key_length E4M3 values followed by one E8M0 scale per 32 values
          if (vMeta->ne[0] != cKeyLength + cKeyLength / 32 || cKeyLength % 32 != 0) {
            throw std::runtime_error(fFormat("%s has an unexpected row size", cEmbdName.c_str()));
          }
          vTable.vRowBytes = (size_t)vMeta->ne[0];
        } else {
          if (vMeta->ne[0] != cKeyLength) {
            throw std::runtime_error(fFormat("%s has an unexpected row size", cEmbdName.c_str()));
          }
          if (vMeta->type != GGML_TYPE_F32 && ggml_get_type_traits(vMeta->type)->to_float == nullptr) {
            throw std::runtime_error(fFormat("%s has an unsupported type %s", cEmbdName.c_str(),
                                             ggml_type_name(vMeta->type)));
          }
          vTable.vRowBytes = ggml_row_size(vMeta->type, cKeyLength);
        }

        // a row index is a bucket offset plus a hash, so the last bucket has to end inside the table
        uint64_t vMaxRow = 0;
        for (int64_t vB = 0; vB < cNCols; ++vB) {
          const size_t cK = (size_t)cE * cNCols + vB;
          vMaxRow = std::max(vMaxRow, lEngramOffsets[cK] + lEngramPrimes[cK]);
        }
        if (vMaxRow > vTable.vNRows || vTable.vNRows > std::numeric_limits<uint32_t>::max()) {
          throw std::runtime_error(fFormat("%s has %llu rows, which does not fit the engram buckets (%llu)",
                                           cEmbdName.c_str(), (unsigned long long)vTable.vNRows,
                                           (unsigned long long)vMaxRow));
        }

        mCreateTensor(cTn(cLlmTensorEngramEmbd, "weight", vI), {vMeta->ne[0], vMeta->ne[1]},
                      mgs_model_loader::cTensorExternal);

        if (!pMl.vFiles.empty()) {
          vTable.vOffset = vWeight->vOffs;
          vTable.vFd = fDsv41OpenModelFile(pMl, vWeight->vIdx);
        }
      } else {
        // a model without files (user-provided tensor data) has no table to read: the rows read as zeros
        mCreateTensor(cTn(cLlmTensorEngramEmbd, "weight", vI), {cKeyLength, 1}, mgs_model_loader::cTensorExternal);
      }

      // the projections are named after the reference (engram_wkv/q/k) or after the DwarfStar tools
      vLayer.vEngramWkv = mCreateTensor(cTn(cLlmTensorEngramWkv, "weight", vI),
                                        {cNCols * cKeyLength, cNEmbd * (cHcMult + 1)}, cTensorNotRequired);
      if (vLayer.vEngramWkv == nullptr) {
        vLayer.vEngramWkv =
            mCreateTensor(cTn(cLlmTensorEngramKv, "weight", vI), {cNCols * cKeyLength, cNEmbd * (cHcMult + 1)}, 0);
      }
      vLayer.vEngramQ = mCreateTensor(cTn(cLlmTensorEngramQ, "weight", vI), {cNEmbd, cHcMult}, cTensorNotRequired);
      if (vLayer.vEngramQ == nullptr) {
        vLayer.vEngramQ = mCreateTensor(cTn(cLlmTensorEngramQNorm, "weight", vI), {cNEmbd, cHcMult}, 0);
      }
      vLayer.vEngramK = mCreateTensor(cTn(cLlmTensorEngramK, "weight", vI), {cNEmbd, cHcMult}, cTensorNotRequired);
      if (vLayer.vEngramK == nullptr) {
        vLayer.vEngramK = mCreateTensor(cTn(cLlmTensorEngramKNorm, "weight", vI), {cNEmbd, cHcMult}, 0);
      }
    }
  }

  // Work out which layer publishes what each layer reads: only a KV source carries a compressor, only
  // an index key owner carries indexer.attn_k and only an index source carries indexer.attn_q_b.
  int32_t vLastKvSource = -1;
  int32_t vLastKeyOwner = -1;
  int32_t vLastIndexSource = -1;

  for (int vI = 0; vI < cNLayer; ++vI) {
    const auto &vLayer = lLayers[vI];
    const uint32_t cRatio = vHparams.aDsv4CompressRatios[vI];

    if (cRatio == 0) {
      if (vLayer.vAttnCompWkv || vLayer.vIndexerAttnQB || vLayer.vIndexerAttnK) {
        throw std::runtime_error(fFormat("layer %d has a compressor or an indexer but no compression ratio", vI));
      }
      continue;
    }

    if (vLayer.vAttnCompWkv) {
      if (!vLayer.vAttnCompNorm || (cRatio > 1 && !vLayer.vAttnCompWgate)) {
        throw std::runtime_error(fFormat("layer %d has an incomplete compressor for ratio %u", vI, cRatio));
      }
      vLastKvSource = vI;
    }
    if (vLayer.vIndexerAttnK) {
      // the index keys come from the pooled latent of the same layer
      if (!vLayer.vAttnCompWkv || !vLayer.vIndexerKNorm) {
        throw std::runtime_error(fFormat("layer %d owns index keys but does not compress its KV", vI));
      }
      vLastKeyOwner = vI;
    }
    if (vLayer.vIndexerAttnQB) {
      if (!vLayer.vIndexerProj) {
        throw std::runtime_error(fFormat("layer %d has an indexer query without its weights", vI));
      }
      vLastIndexSource = vI;
    }

    if (vLastKvSource < 0 || vLastKeyOwner < 0 || vLastIndexSource < 0) {
      throw std::runtime_error(fFormat("layer %d reads a compressed stream before any layer publishes one", vI));
    }

    // a reader whose ratio differs from its source would index rows that stand for other positions
    if (vHparams.aDsv4CompressRatios[vLastKvSource] != cRatio ||
        vHparams.aDsv4CompressRatios[vLastIndexSource] != cRatio ||
        (vLayer.vIndexerAttnQB && vHparams.aDsv4CompressRatios[vLastKeyOwner] != cRatio)) {
      throw std::runtime_error(fFormat("layer %d compresses at ratio %u but reads a stream of another ratio", vI,
                                       cRatio));
    }

    vHparams.aDsv41KvSource[vI] = vLastKvSource;
    vHparams.aDsv41IndexKeySource[vI] = vLastKeyOwner;
    vHparams.aDsv41TopKSource[vI] = vLastIndexSource;
  }

  // without metadata, the candidate source is the first index source of the finest ratio, keeping
  // 2048 blocks of 8 rows as the released model does
  if (vHparams.vDsv41CandidateSource == -2) {
    vHparams.vDsv41CandidateSource = -1;

    uint32_t vMinRatio = 0;
    for (int vI = 0; vI < cNLayer; ++vI) {
      const uint32_t cRatio = vHparams.aDsv4CompressRatios[vI];
      if (cRatio != 0 && (vMinRatio == 0 || cRatio < vMinRatio)) {
        vMinRatio = cRatio;
      }
    }
    for (int vI = 0; vI < cNLayer && vMinRatio == 1; ++vI) {
      if (vHparams.aDsv4CompressRatios[vI] == vMinRatio && vHparams.mIsDsv41IndexSource(vI)) {
        vHparams.vDsv41CandidateSource = vI;
        vHparams.vDsv41CandidateTopKBlocks = 2048;
        vHparams.vDsv41CandidateBlockSize = 8;
        break;
      }
    }
  }

  if (vHparams.vDsv41CandidateSource >= 0) {
    const int32_t cSource = vHparams.vDsv41CandidateSource;
    if (cSource >= cNLayer || !vHparams.mIsDsv41IndexSource(cSource) || vHparams.vDsv41CandidateTopKBlocks == 0 ||
        vHparams.vDsv41CandidateBlockSize == 0 || 256 % vHparams.vDsv41CandidateBlockSize != 0) {
      MGS_LOG_WARN("%s: ignoring the invalid indexer candidate settings (source layer %d)\n", __func__, cSource);
      vHparams.vDsv41CandidateSource = -1;
    }
  }

  MGS_LOG_INFO("%s: DeepSeek-V4.1: KV sources:", __func__);
  for (int vI = 0; vI < cNLayer; ++vI) {
    if (vHparams.mIsDsv41KvSource(vI)) {
      MGS_LOG_CONT(" %d(r%u)", vI, vHparams.aDsv4CompressRatios[vI]);
    }
  }
  MGS_LOG_CONT(", index sources:");
  for (int vI = 0; vI < cNLayer; ++vI) {
    if (vHparams.mIsDsv41IndexSource(vI)) {
      MGS_LOG_CONT(" %d", vI);
    }
  }
  MGS_LOG_CONT(", candidate source: %d, engram layers: %zu\n", vHparams.vDsv41CandidateSource,
               lEngramLayerIds.size());
}

std::unique_ptr<llm_graph_context> mgs_model_deepseek41::mBuildArchGraph(const llm_graph_params &pParams) const {
  return std::make_unique<graph>(*this, pParams);
}

//
// graph inputs
//

static void fDsv41SetI64(ggml_tensor *pDst, const std::vector<int64_t> &pSrc) {
  if (!pDst || !pDst->buffer) {
    return;
  }

  GGML_ASSERT(pDst->ne[0] == (int64_t)pSrc.size());
  ggml_backend_tensor_set(pDst, pSrc.data(), 0, pSrc.size() * ggml_element_size(pDst));
}

static void fDsv41SetI32(ggml_tensor *pDst, const std::vector<int32_t> &pSrc) {
  if (!pDst || !pDst->buffer) {
    return;
  }

  GGML_ASSERT(pDst->ne[0] == (int64_t)pSrc.size());
  ggml_backend_tensor_set(pDst, pSrc.data(), 0, pSrc.size() * ggml_element_size(pDst));
}

// fills a [n_cols, n_tokens/n_stream, 1, n_stream] host tensor from a per-token value function
template <typename fn_t> static void fDsv41SetPerToken(ggml_tensor *pDst, uint32_t pNTokens, fn_t pValue) {
  if (!pDst || !pDst->buffer) {
    return;
  }

  GGML_ASSERT(pDst->type == GGML_TYPE_F32 || pDst->type == GGML_TYPE_F16);
  GGML_ASSERT(ggml_backend_buffer_is_host(pDst->buffer));
  GGML_ASSERT(pDst->ne[1] * pDst->ne[3] == (int64_t)pNTokens);

  if (pDst->type == GGML_TYPE_F32) {
    float *vData = (float *)pDst->data;

    for (int64_t vI = 0; vI < (int64_t)pNTokens; ++vI) {
      for (int64_t vJ = 0; vJ < pDst->ne[0]; ++vJ) {
        vData[vI * pDst->ne[0] + vJ] = pValue(vI, vJ);
      }
    }
  } else {
    ggml_fp16_t *vData = (ggml_fp16_t *)pDst->data;

    for (int64_t vI = 0; vI < (int64_t)pNTokens; ++vI) {
      for (int64_t vJ = 0; vJ < pDst->ne[0]; ++vJ) {
        vData[vI * pDst->ne[0] + vJ] = mgs_cast<ggml_fp16_t>(pValue(vI, vJ));
      }
    }
  }
}

static ggml_tensor *fDsv41BuildInput1d(ggml_context *pCtx, ggml_type pType, int64_t pNe0, const std::string &pName) {
  if (pNe0 == 0) {
    return nullptr;
  }

  ggml_tensor *vRes = ggml_new_tensor_1d(pCtx, pType, pNe0);
  ggml_set_input(vRes);
  ggml_set_name(vRes, pName.c_str());

  return vRes;
}

static bool fDsv41CanReuseTensor1d(ggml_tensor *pT, int64_t pNe0) {
  return (pT == nullptr && pNe0 == 0) || (pT != nullptr && pT->ne[0] == pNe0);
}

static bool fDsv41CanReuseMask(ggml_tensor *pT, int64_t pNe0, uint32_t pNTokens, int64_t pNStream) {
  if (pNe0 == 0) {
    return pT == nullptr;
  }

  return pT != nullptr && pT->ne[0] == pNe0 && pT->ne[1] == (int64_t)pNTokens / pNStream && pT->ne[2] == 1 &&
         pT->ne[3] == pNStream;
}

// the inputs of the raw sliding-window cache and of every compression group
class llm_graph_input_dsv41 : public llm_graph_input_i {
public:
  struct comp_input {
    ggml_tensor *vStatePos = nullptr;             // I32 [n_state]
    ggml_tensor *vStatePersistSrcIdxs = nullptr;  // I32 [n_state_persist]
    ggml_tensor *vStatePersistDstIdxs = nullptr;  // I32 [n_state_persist]
    ggml_tensor *vStateRestoreSrcIdxs = nullptr;  // I32 [n_state_restore]
    ggml_tensor *vStateRestoreDstIdxs = nullptr;  // I32 [n_state_restore]
    ggml_tensor *vStateSnapshotSrcIdxs = nullptr; // I32 [n_state_snapshot]
    ggml_tensor *vStateSnapshotDstIdxs = nullptr; // I32 [n_state_snapshot]
    ggml_tensor *vStateReadIdxs = nullptr;        // I32 [ratio*n_state_write]
    ggml_tensor *vStateWriteIdxs = nullptr;       // I64 [n_state_write]
    ggml_tensor *vStateWritePos = nullptr;        // I32 [n_state_write]

    ggml_tensor *vKqMask = nullptr;   // F32/F16 [n_kv, n_batch/n_stream, 1, n_stream], attention
    ggml_tensor *vIdxMask = nullptr;  // F32/F16 [n_kv, n_batch/n_stream, 1, n_stream], indexer
    ggml_tensor *vCandPin = nullptr;  // F32 [n_kv/block, n_batch/n_stream, 1, n_stream], newest block pin

    ggml_tensor *vKRot = nullptr;    // Hadamard rotation of the compressed rows
    ggml_tensor *vIdxKRot = nullptr; // Hadamard rotation of the index keys
  };

  llm_graph_input_dsv41(const mgs_cparams &pCparams, const mgs_hparams &pHparams,
                        const mgs_kv_cache_dsv41_context *pMctx)
      : cCparams(pCparams), cHparams(pHparams), vMctx(pMctx) {}
  ~llm_graph_input_dsv41() = default;

  void mSetInput(const mgs_ubatch *pUbatch) override;

  bool mCanReuse(const llm_graph_params &pParams) override;

  // the graph width of a candidate pin for this group plan, 0 when the candidate pool changes nothing
  int64_t mCandidateBlocks(uint32_t pG, const mgs_kv_cache_dsv41_context::comp_plan &pPlan) const;

  ggml_tensor *vSelfKIdxs = nullptr;  // I64 [n_batch]
  ggml_tensor *vSelfKqMask = nullptr; // F32/F16 [n_kv, n_batch/n_stream, 1, n_stream]
  ggml_tensor *vSelfKRot = nullptr;

  std::vector<comp_input> lComp;

  const mgs_cparams cCparams;
  const mgs_hparams &cHparams;

  const mgs_kv_cache_dsv41_context *vMctx;
};

int64_t llm_graph_input_dsv41::mCandidateBlocks(uint32_t pG,
                                                const mgs_kv_cache_dsv41_context::comp_plan &pPlan) const {
  const int32_t cSource = cHparams.vDsv41CandidateSource;
  if (cSource < 0 || pPlan.vNKv == 0 || cHparams.aDsv4CompressRatios[cSource] != vMctx->mGetRatio(pG)) {
    return 0;
  }

  // below topk_blocks blocks every block is kept, so the pool changes nothing
  const int64_t cBlockSize = cHparams.vDsv41CandidateBlockSize;
  if (pPlan.vNKv <= (int64_t)cHparams.vDsv41CandidateTopKBlocks * cBlockSize || pPlan.vNKv % cBlockSize != 0) {
    return 0;
  }

  return pPlan.vNKv / cBlockSize;
}

void llm_graph_input_dsv41::mSetInput(const mgs_ubatch *pUbatch) {
  const auto *vRaw = vMctx->mGetRaw();
  const uint32_t cNTokens = pUbatch->vNTokens;

  if (vSelfKIdxs && vSelfKIdxs->buffer) {
    vRaw->mSetInputKIdxs(vSelfKIdxs);
  }
  if (vSelfKqMask && vSelfKqMask->buffer) {
    vRaw->mSetInputKqMask(vSelfKqMask, pUbatch, cCparams.vCausalAttn);
  }
  if (vSelfKRot && vSelfKRot->buffer) {
    vRaw->mSetInputKRot(vSelfKRot);
  }

  for (uint32_t vG = 0; vG < lComp.size(); ++vG) {
    const auto &vPlan = vMctx->mGetPlan(vG, *pUbatch);
    const auto &vInp = lComp[vG];

    fDsv41SetI32(vInp.vStatePos, vPlan.lStatePos);
    fDsv41SetI32(vInp.vStatePersistSrcIdxs, vPlan.lStatePersistSrcIdxs);
    fDsv41SetI32(vInp.vStatePersistDstIdxs, vPlan.lStatePersistDstIdxs);
    fDsv41SetI32(vInp.vStateRestoreSrcIdxs, vPlan.lStateRestoreSrcIdxs);
    fDsv41SetI32(vInp.vStateRestoreDstIdxs, vPlan.lStateRestoreDstIdxs);
    fDsv41SetI32(vInp.vStateSnapshotSrcIdxs, vPlan.lStateSnapshotSrcIdxs);
    fDsv41SetI32(vInp.vStateSnapshotDstIdxs, vPlan.lStateSnapshotDstIdxs);
    fDsv41SetI32(vInp.vStateReadIdxs, vPlan.lStateReadIdxs);
    fDsv41SetI64(vInp.vStateWriteIdxs, vPlan.lStateWriteIdxs);
    fDsv41SetI32(vInp.vStateWritePos, vPlan.lStateWritePos);

    // a compressed row is visible once the query has passed the last token of its group
    const auto cVisible = [&](int64_t pI, int64_t pJ) -> float {
      return pJ < vPlan.lNVisible[pI] ? 0.0f : -INFINITY;
    };
    fDsv41SetPerToken(vInp.vKqMask, cNTokens, cVisible);
    fDsv41SetPerToken(vInp.vIdxMask, cNTokens, cVisible);

    // the block that holds the newest visible row is only partly filled: pin it into the pool
    const int64_t cBlockSize = std::max<int64_t>(1, cHparams.vDsv41CandidateBlockSize);
    fDsv41SetPerToken(vInp.vCandPin, cNTokens, [&](int64_t pI, int64_t pJ) -> float {
      const int32_t cNVisible = vPlan.lNVisible[pI];
      return cNVisible > 0 && pJ == (cNVisible - 1) / cBlockSize ? INFINITY : 0.0f;
    });

    if (vInp.vKRot && vInp.vKRot->buffer) {
      vMctx->mGetComp(vG)->mSetInputKRot(vInp.vKRot);
    }
    if (vInp.vIdxKRot && vInp.vIdxKRot->buffer && vMctx->mGetIdx(vG)) {
      vMctx->mGetIdx(vG)->mSetInputKRot(vInp.vIdxKRot);
    }
  }
}

bool llm_graph_input_dsv41::mCanReuse(const llm_graph_params &pParams) {
  vMctx = static_cast<const mgs_kv_cache_dsv41_context *>(pParams.vMctx);

  const auto *vRaw = vMctx->mGetRaw();
  const uint32_t cNTokens = pParams.vUbatch.vNTokens;

  if (lComp.size() != vMctx->mGetNGroup()) {
    return false;
  }

  const int64_t cNStream = lComp.empty() ? 1 : vMctx->mGetPlan(0, pParams.vUbatch).vNStream;

  bool vRes = true;

  if (vSelfKIdxs && vSelfKIdxs->buffer) {
    vRes &= vSelfKIdxs->ne[0] == vRaw->mGetNWrite();
  }
  if (vSelfKqMask && vSelfKqMask->buffer) {
    vRes &= fDsv41CanReuseMask(vSelfKqMask, vRaw->mGetNKv(), cNTokens, cNStream);
  }

  for (uint32_t vG = 0; vG < lComp.size(); ++vG) {
    const auto &vPlan = vMctx->mGetPlan(vG, pParams.vUbatch);
    const auto &vInp = lComp[vG];

    vRes &= vPlan.vNStream == cNStream;
    vRes &= fDsv41CanReuseTensor1d(vInp.vStatePos, vPlan.lStatePos.size());
    vRes &= fDsv41CanReuseTensor1d(vInp.vStatePersistSrcIdxs, vPlan.lStatePersistSrcIdxs.size());
    vRes &= fDsv41CanReuseTensor1d(vInp.vStatePersistDstIdxs, vPlan.lStatePersistDstIdxs.size());
    vRes &= fDsv41CanReuseTensor1d(vInp.vStateRestoreSrcIdxs, vPlan.lStateRestoreSrcIdxs.size());
    vRes &= fDsv41CanReuseTensor1d(vInp.vStateRestoreDstIdxs, vPlan.lStateRestoreDstIdxs.size());
    vRes &= fDsv41CanReuseTensor1d(vInp.vStateSnapshotSrcIdxs, vPlan.lStateSnapshotSrcIdxs.size());
    vRes &= fDsv41CanReuseTensor1d(vInp.vStateSnapshotDstIdxs, vPlan.lStateSnapshotDstIdxs.size());
    vRes &= fDsv41CanReuseTensor1d(vInp.vStateReadIdxs, vPlan.lStateReadIdxs.size());
    vRes &= fDsv41CanReuseTensor1d(vInp.vStateWriteIdxs, vPlan.lStateWriteIdxs.size());
    vRes &= fDsv41CanReuseTensor1d(vInp.vStateWritePos, vPlan.lStateWritePos.size());
    vRes &= fDsv41CanReuseMask(vInp.vKqMask, vPlan.vNKv, cNTokens, cNStream);
    vRes &= fDsv41CanReuseMask(vInp.vIdxMask, vPlan.vNKv, cNTokens, cNStream);
    vRes &= fDsv41CanReuseMask(vInp.vCandPin, mCandidateBlocks(vG, vPlan), cNTokens, cNStream);
  }

  return vRes;
}

llm_graph_input_dsv41 *mgs_model_deepseek41::graph::mBuildInpDsv41() const {
  const auto *vMctxCur = static_cast<const mgs_kv_cache_dsv41_context *>(vMctx);
  const auto *vRawCtx = vMctxCur->mGetRaw();

  GGML_ASSERT(vHparams.vSwaType != cMgsSwaTypeNone && "DeepSeek-V4.1 expects a SWA raw cache");

  auto vInp = std::make_unique<llm_graph_input_dsv41>(vCparams, vHparams, vMctxCur);

  const uint32_t cNGroup = vMctxCur->mGetNGroup();
  const int64_t cNStream = cNGroup == 0 ? 1 : vMctxCur->mGetPlan(0, vUbatch).vNStream;
  GGML_ASSERT(cNTokens % cNStream == 0);

  const ggml_type cAttnMaskType = vCparams.vFlashAttn ? GGML_TYPE_F16 : GGML_TYPE_F32;

  vInp->vSelfKIdxs = vRawCtx->mBuildInputKIdxs(vCtx0, vUbatch);
  vInp->vSelfKqMask = ggml_new_tensor_4d(vCtx0, cAttnMaskType, vRawCtx->mGetNKv(), cNTokens / cNStream, 1, cNStream);
  ggml_set_input(vInp->vSelfKqMask);
  ggml_set_name(vInp->vSelfKqMask, "attn_inp_kq_mask");
  vInp->vSelfKRot = vRawCtx->mBuildInputKRot(vCtx0);

  vInp->lComp.resize(cNGroup);
  for (uint32_t vG = 0; vG < cNGroup; ++vG) {
    const auto &vPlan = vMctxCur->mGetPlan(vG, vUbatch);
    auto &vComp = vInp->lComp[vG];
    const std::string cPrefix = "dsv41_r" + std::to_string(vMctxCur->mGetRatio(vG)) + "_";

    GGML_ASSERT(vPlan.vNStream == cNStream);

    vComp.vStatePos = fDsv41BuildInput1d(vCtx0, GGML_TYPE_I32, vPlan.lStatePos.size(), cPrefix + "state_pos");
    vComp.vStatePersistSrcIdxs = fDsv41BuildInput1d(vCtx0, GGML_TYPE_I32, vPlan.lStatePersistSrcIdxs.size(),
                                                    cPrefix + "state_persist_src_idxs");
    vComp.vStatePersistDstIdxs = fDsv41BuildInput1d(vCtx0, GGML_TYPE_I32, vPlan.lStatePersistDstIdxs.size(),
                                                    cPrefix + "state_persist_dst_idxs");
    vComp.vStateRestoreSrcIdxs = fDsv41BuildInput1d(vCtx0, GGML_TYPE_I32, vPlan.lStateRestoreSrcIdxs.size(),
                                                    cPrefix + "state_restore_src_idxs");
    vComp.vStateRestoreDstIdxs = fDsv41BuildInput1d(vCtx0, GGML_TYPE_I32, vPlan.lStateRestoreDstIdxs.size(),
                                                    cPrefix + "state_restore_dst_idxs");
    vComp.vStateSnapshotSrcIdxs = fDsv41BuildInput1d(vCtx0, GGML_TYPE_I32, vPlan.lStateSnapshotSrcIdxs.size(),
                                                     cPrefix + "state_snapshot_src_idxs");
    vComp.vStateSnapshotDstIdxs = fDsv41BuildInput1d(vCtx0, GGML_TYPE_I32, vPlan.lStateSnapshotDstIdxs.size(),
                                                     cPrefix + "state_snapshot_dst_idxs");
    vComp.vStateReadIdxs =
        fDsv41BuildInput1d(vCtx0, GGML_TYPE_I32, vPlan.lStateReadIdxs.size(), cPrefix + "state_read_idxs");
    vComp.vStateWriteIdxs =
        fDsv41BuildInput1d(vCtx0, GGML_TYPE_I64, vPlan.lStateWriteIdxs.size(), cPrefix + "state_write_idxs");
    vComp.vStateWritePos =
        fDsv41BuildInput1d(vCtx0, GGML_TYPE_I32, vPlan.lStateWritePos.size(), cPrefix + "state_write_pos");

    if (vPlan.vNKv > 0) {
      vComp.vKqMask = ggml_new_tensor_4d(vCtx0, cAttnMaskType, vPlan.vNKv, cNTokens / cNStream, 1, cNStream);
      ggml_set_input(vComp.vKqMask);
      ggml_set_name(vComp.vKqMask, (cPrefix + "kq_mask").c_str());

      // the fused lightning indexer takes an F16 mask
      vComp.vIdxMask = ggml_new_tensor_4d(vCtx0, vCparams.vFusedLid ? GGML_TYPE_F16 : GGML_TYPE_F32, vPlan.vNKv,
                                          cNTokens / cNStream, 1, cNStream);
      ggml_set_input(vComp.vIdxMask);
      ggml_set_name(vComp.vIdxMask, (cPrefix + "idx_mask").c_str());

      const int64_t cNBlocks = vInp->mCandidateBlocks(vG, vPlan);
      if (cNBlocks > 0) {
        vComp.vCandPin = ggml_new_tensor_4d(vCtx0, GGML_TYPE_F32, cNBlocks, cNTokens / cNStream, 1, cNStream);
        ggml_set_input(vComp.vCandPin);
        ggml_set_name(vComp.vCandPin, (cPrefix + "cand_pin").c_str());
      }
    }

    vComp.vKRot = vMctxCur->mGetComp(vG)->mBuildInputKRot(vCtx0);
    if (vMctxCur->mGetIdx(vG)) {
      vComp.vIdxKRot = vMctxCur->mGetIdx(vG)->mBuildInputKRot(vCtx0);
    }
  }

  return (llm_graph_input_dsv41 *)vRes->mAddInput(std::move(vInp));
}

// Engram n-gram hash: every token gathers n_cols rows of the table of each engram layer.
//   rolling_i = (t[0]*m[0]) ^ ... ^ (t[i]*m[i]);  row = rolling_i % prime[i][h] + offset[i][h]
// t[0] is the token itself and t[i] the token i positions back, all mapped to the compressed vocab.
// Look-back stops at the start of the sequence, where the padding id stands in.
class llm_graph_input_dsv41_engram : public llm_graph_input_i {
public:
  llm_graph_input_dsv41_engram(const mgs_model_deepseek41 &pModel, const mgs_kv_cache_dsv41_raw_context *pMctx)
      : cModel(pModel), vMctx(pMctx) {}
  ~llm_graph_input_dsv41_engram() = default;

  void mSetInput(const mgs_ubatch *pUbatch) override;

  bool mCanReuse(const llm_graph_params &pParams) override {
    vMctx = static_cast<const mgs_kv_cache_dsv41_context *>(pParams.vMctx)->mGetRaw();

    bool vRes = true;
    for (ggml_tensor *vEmb : lEmb) {
      vRes &= vEmb->ne[1] == (int64_t)pParams.vUbatch.vNTokens;
    }
    return vRes;
  }

  // per engram layer, the gathered rows: F32 [n_cols * key_length, n_tokens]
  std::vector<ggml_tensor *> lEmb;

  // I32 [hc]: 0 .. hc-1, to dequantize the gate weights with get_rows
  ggml_tensor *vHcIds = nullptr;

  const mgs_model_deepseek41 &cModel;

  // the preceding tokens live in the cells of the raw cache
  const mgs_kv_cache_dsv41_raw_context *vMctx;

  // scratch, reused across mSetInput() calls
  std::vector<mgs_token> lPrev;
  std::vector<uint32_t> lRows;
  std::vector<float> lValues;
};

void llm_graph_input_dsv41_engram::mSetInput(const mgs_ubatch *pUbatch) {
  const auto &vHp = cModel.vHparams;

  const int64_t cNTokens = pUbatch->vNTokens;
  const int64_t cNGram = vHp.vEngramMaxNgramSize;
  const int64_t cNHeads = vHp.vEngramNHead;
  const int64_t cNCols = (cNGram - 1) * cNHeads;
  const int64_t cNPrev = cNGram - 1;
  const int64_t cKeyLength = vHp.vEngramKeyLength;

  if (vHcIds && vHcIds->buffer) {
    std::vector<int32_t> lIds(vHcIds->ne[0]);
    for (size_t vI = 0; vI < lIds.size(); ++vI) {
      lIds[vI] = (int32_t)vI;
    }
    ggml_backend_tensor_set(vHcIds, lIds.data(), 0, ggml_nbytes(vHcIds));
  }

  // an embedding ubatch (an image) has no token ids: its positions read as padding, which also
  // ends the look-back of the tokens after it
  const uint64_t cPad = cModel.vEngramPadId;
  const auto cMap = [&](mgs_token pToken) -> uint64_t {
    if (pToken < 0 || (size_t)pToken >= cModel.lEngramTokenMap.size()) {
      return cPad;
    }
    return cModel.lEngramTokenMap[pToken];
  };

  // predecessors come from the raw cache cells; mApply() already stored this ubatch
  GGML_ASSERT(vMctx != nullptr);
  vMctx->mGetPrevTokens(*pUbatch, (uint32_t)cNPrev, lPrev);

  std::vector<uint64_t> lCtx(cNGram);

  lRows.resize((size_t)cNTokens * cNCols);
  lValues.resize((size_t)cNTokens * cNCols * cKeyLength);

  for (size_t vE = 0; vE < lEmb.size(); ++vE) {
    const uint64_t *vMult = cModel.lEngramMultipliers.data() + vE * cNGram;
    const uint64_t *vPrime = cModel.lEngramPrimes.data() + vE * cNCols;
    const uint64_t *vOffset = cModel.lEngramOffsets.data() + vE * cNCols;

    for (int64_t vI = 0; vI < cNTokens; ++vI) {
      const bool cHasToken = pUbatch->vToken != nullptr;
      lCtx[0] = cHasToken ? cMap(pUbatch->vToken[vI]) : cPad;

      bool vBlocked = !cHasToken;
      for (int64_t vS = 1; vS < cNGram; ++vS) {
        // predecessor vS positions back; lPrev is oldest first, missing entries are MGS_TOKEN_NULL
        const mgs_token cToken = vBlocked ? MGS_TOKEN_NULL : lPrev[vI * cNPrev + (cNPrev - vS)];
        vBlocked = vBlocked || cToken < 0;
        lCtx[vS] = vBlocked ? cPad : cMap(cToken);
      }

      // compressed ids stay far below 2^20 and the multipliers below 2^43, so no product overflows
      uint64_t vRolling = lCtx[0] * vMult[0];
      for (int64_t vS = 1; vS < cNGram; ++vS) {
        vRolling ^= lCtx[vS] * vMult[vS];

        for (int64_t vH = 0; vH < cNHeads; ++vH) {
          const int64_t cB = (vS - 1) * cNHeads + vH;
          lRows[vI * cNCols + cB] = (uint32_t)(vRolling % vPrime[cB] + vOffset[cB]);
        }
      }
    }

    cModel.mEngramReadRows((int)vE, lRows.data(), lRows.size(), lValues.data());

    if (lEmb[vE]->buffer) {
      ggml_backend_tensor_set(lEmb[vE], lValues.data(), 0, lValues.size() * sizeof(float));
    }
  }
}

llm_graph_input_dsv41_engram *mgs_model_deepseek41::graph::mBuildInpEngram(const mgs_model_deepseek41 &pModel) const {
  const auto *vMctxCur = static_cast<const mgs_kv_cache_dsv41_context *>(vMctx);

  auto vInp = std::make_unique<llm_graph_input_dsv41_engram>(pModel, vMctxCur->mGetRaw());

  const int64_t cNCols = (int64_t)(vHparams.vEngramMaxNgramSize - 1) * vHparams.vEngramNHead;
  const int64_t cKeyLength = vHparams.vEngramKeyLength;

  for (size_t vE = 0; vE < pModel.lEngramLayerIds.size(); ++vE) {
    ggml_tensor *vEmb = ggml_new_tensor_2d(vCtx0, GGML_TYPE_F32, cNCols * cKeyLength, cNTokens);
    ggml_set_input(vEmb);
    ggml_set_name(vEmb, ("engram_rows_" + std::to_string(vE)).c_str());
    vInp->lEmb.push_back(vEmb);
  }

  vInp->vHcIds = ggml_new_tensor_1d(vCtx0, GGML_TYPE_I32, vHparams.vDsv4HcMult);
  ggml_set_input(vInp->vHcIds);
  ggml_set_name(vInp->vHcIds, "engram_hc_ids");

  return (llm_graph_input_dsv41_engram *)vRes->mAddInput(std::move(vInp));
}

//
// graph
//

static size_t fDsv41ElemOffset(const ggml_tensor *pT, int64_t pI) { return ggml_row_size(pT->type, pI); }

static ggml_tensor *fDsv41View1d(ggml_context *pCtx, ggml_tensor *pT, int64_t pNe0, int64_t pI0) {
  return ggml_view_1d(pCtx, pT, pNe0, fDsv41ElemOffset(pT, pI0));
}

static ggml_tensor *fDsv41View2d(ggml_context *pCtx, ggml_tensor *pT, int64_t pNe0, int64_t pNe1, int64_t pI0) {
  return ggml_view_2d(pCtx, pT, pNe0, pNe1, pT->nb[1], fDsv41ElemOffset(pT, pI0));
}

// mean over the hyper-connection copies: [n_embd, hc, n_tokens] -> [n_embd, n_tokens]
static ggml_tensor *fDsv41HcMean(ggml_context *pCtx, ggml_tensor *pX) {
  const int64_t cHc = pX->ne[1];

  ggml_tensor *vAcc = ggml_view_2d(pCtx, pX, pX->ne[0], pX->ne[2], pX->nb[2], 0);
  for (int64_t vS = 1; vS < cHc; ++vS) {
    vAcc = ggml_add(pCtx, vAcc, ggml_view_2d(pCtx, pX, pX->ne[0], pX->ne[2], pX->nb[2], vS * pX->nb[1]));
  }
  return ggml_scale(pCtx, vAcc, 1.0f / cHc);
}

static ggml_tensor *fDsv41HcAffine(ggml_context *pCtx, ggml_tensor *pX, ggml_tensor *pScale, ggml_tensor *pBase) {
  pX = ggml_mul(pCtx, pX, pScale);
  pX = ggml_add(pCtx, pX, pBase);
  return pX;
}

ggml_tensor *mgs_model_deepseek41::graph::mBuildHcPre(ggml_tensor *pX, ggml_tensor *pWeights, int pIl) const {
  GGML_ASSERT(pX->ne[0] == cNEmbd);
  GGML_ASSERT(pX->ne[1] == vHparams.vDsv4HcMult);

  const int64_t cHc = vHparams.vDsv4HcMult;
  const int64_t cNt = pX->ne[2];

  if (vCparams.vFusedDsv4HcPre && pIl >= 0) {
    ggml_tensor *vResult = ggml_dsv4_hc_pre(vCtx0, pX, pWeights);
    vRes->mAddFusedNode({cLlmFusedOpDsv4HcPre, vResult, pIl});
    return vResult;
  }

  ggml_tensor *vResult = nullptr;
  for (int64_t vIh = 0; vIh < cHc; ++vIh) {
    ggml_tensor *vXh = ggml_view_2d(vCtx0, pX, cNEmbd, cNt, pX->nb[2], vIh * pX->nb[1]);
    ggml_tensor *vWh = ggml_view_2d(vCtx0, pWeights, 1, cNt, pWeights->nb[1], vIh * pWeights->nb[0]);
    ggml_tensor *vCur = ggml_mul(vCtx0, vXh, vWh);
    vResult = vResult ? ggml_add(vCtx0, vResult, vCur) : vCur;
  }

  return vResult;
}

ggml_tensor *mgs_model_deepseek41::graph::mBuildHcSinkhorn(ggml_tensor *pComb, int pIl) const {
  GGML_UNUSED(pIl);

  // comb is [dst_hc, src_hc, n_tokens]: row softmax over dst, one column normalization, then
  // repeated row/column normalization
  pComb = ggml_soft_max(vCtx0, pComb);

  ggml_tensor *vEps = ggml_new_tensor_1d(vCtx0, GGML_TYPE_F32, 1);
  vEps = ggml_fill(vCtx0, vEps, vHparams.vDsv4HcEps);

  pComb = ggml_add(vCtx0, pComb, vEps);

  auto vNormCols = [&]() {
    ggml_tensor *vCombSrcDst = ggml_cont(vCtx0, ggml_permute(vCtx0, pComb, 1, 0, 2, 3));
    ggml_tensor *vColSum = ggml_sum_rows(vCtx0, vCombSrcDst);
    vColSum = ggml_add(vCtx0, vColSum, vEps);
    vColSum = ggml_permute(vCtx0, vColSum, 1, 0, 2, 3);
    pComb = ggml_div(vCtx0, pComb, vColSum);
  };

  auto vNormRows = [&]() {
    ggml_tensor *vRowSum = ggml_sum_rows(vCtx0, pComb);
    vRowSum = ggml_add(vCtx0, vRowSum, vEps);
    pComb = ggml_div(vCtx0, pComb, vRowSum);
  };

  vNormCols();
  for (uint32_t vI = 1; vI < vHparams.vDsv4HcSinkhornIters; ++vI) {
    vNormRows();
    vNormCols();
  }

  return pComb;
}

void mgs_model_deepseek41::graph::mBuildHcMixes(ggml_tensor *pX, ggml_tensor *pHcFn, ggml_tensor *pHcScale,
                                                ggml_tensor *pHcBase, ggml_tensor **pPre, ggml_tensor **pPost,
                                                ggml_tensor **pComb, int pIl) const {
  const int64_t cHc = vHparams.vDsv4HcMult;
  const int64_t cHcDim = cHc * cNEmbd;
  const int64_t cHcMixDim = (2 + cHc) * cHc;
  const int64_t cNt = pX->ne[2];

  GGML_ASSERT(pHcFn->ne[1] == cHcMixDim);

  // normalized over the whole flattened hc * n_embd stream, one statistic per token
  ggml_tensor *vFlat = ggml_reshape_2d(vCtx0, pX, cHcDim, cNt);
  ggml_tensor *vFlatNorm = ggml_rms_norm(vCtx0, vFlat, cNormRmsEps);
  ggml_tensor *vMixes = ggml_mul_mat(vCtx0, pHcFn, vFlatNorm);
  mCb(vMixes, "hc_mixes", pIl);

  ggml_tensor *vScalePre = fDsv41View1d(vCtx0, pHcScale, 1, 0);
  ggml_tensor *vScalePost = fDsv41View1d(vCtx0, pHcScale, 1, 1);

  ggml_tensor *vBasePre = fDsv41View1d(vCtx0, pHcBase, cHc, 0);
  ggml_tensor *vBasePost = fDsv41View1d(vCtx0, pHcBase, cHc, cHc);

  *pPre = fDsv41View2d(vCtx0, vMixes, cHc, cNt, 0);
  *pPre = fDsv41HcAffine(vCtx0, *pPre, vScalePre, vBasePre);
  *pPre = ggml_sigmoid(vCtx0, *pPre);
  *pPre = ggml_scale_bias(vCtx0, *pPre, 1.0f, vHparams.vDsv4HcEps);
  mCb(*pPre, "hc_pre", pIl);

  *pPost = fDsv41View2d(vCtx0, vMixes, cHc, cNt, cHc);
  *pPost = fDsv41HcAffine(vCtx0, *pPost, vScalePost, vBasePost);
  *pPost = ggml_sigmoid(vCtx0, *pPost);
  *pPost = ggml_scale(vCtx0, *pPost, 2.0f);
  mCb(*pPost, "hc_post", pIl);

  if (vCparams.vFusedDsv4HcComb) {
    *pComb = ggml_dsv4_hc_comb(vCtx0, vMixes, pHcScale, pHcBase, vHparams.vDsv4HcEps,
                               (int32_t)vHparams.vDsv4HcSinkhornIters);
    vRes->mAddFusedNode({cLlmFusedOpDsv4HcComb, *pComb, pIl});
  } else {
    ggml_tensor *vScaleComb = fDsv41View1d(vCtx0, pHcScale, 1, 2);
    ggml_tensor *vBaseComb = fDsv41View1d(vCtx0, pHcBase, cHc * cHc, 2 * cHc);

    *pComb = fDsv41View2d(vCtx0, vMixes, cHc * cHc, cNt, 2 * cHc);
    *pComb = fDsv41HcAffine(vCtx0, *pComb, vScaleComb, vBaseComb);
    *pComb = ggml_reshape_3d(vCtx0, *pComb, cHc, cHc, cNt);
    *pComb = mBuildHcSinkhorn(*pComb, pIl);
  }
  mCb(*pComb, "hc_comb", pIl);
}

ggml_tensor *mgs_model_deepseek41::graph::mBuildHcPost(ggml_tensor *pX, ggml_tensor *pResidual, ggml_tensor *pPost,
                                                       ggml_tensor *pComb, int pIl) const {
  GGML_ASSERT(pX->ne[0] == cNEmbd);
  GGML_ASSERT(pResidual->ne[1] == vHparams.vDsv4HcMult);

  if (vCparams.vFusedDsv4HcPost) {
    ggml_tensor *vResult = ggml_dsv4_hc_post(vCtx0, pX, pResidual, pPost, pComb);
    vRes->mAddFusedNode({cLlmFusedOpDsv4HcPost, vResult, pIl});
    return vResult;
  }

  const int64_t cHc = vHparams.vDsv4HcMult;
  const int64_t cNt = pX->ne[1];

  ggml_tensor *vOut = nullptr;
  for (int64_t vDst = 0; vDst < cHc; ++vDst) {
    ggml_tensor *vPostDst = ggml_view_2d(vCtx0, pPost, 1, cNt, pPost->nb[1], vDst * pPost->nb[0]);
    ggml_tensor *vCur = ggml_mul(vCtx0, pX, vPostDst);

    for (int64_t vSrc = 0; vSrc < cHc; ++vSrc) {
      ggml_tensor *vResSrc = ggml_view_2d(vCtx0, pResidual, cNEmbd, cNt, pResidual->nb[2], vSrc * pResidual->nb[1]);
      ggml_tensor *vCombSrcDst =
          ggml_view_2d(vCtx0, pComb, 1, cNt, pComb->nb[2], vDst * pComb->nb[0] + vSrc * pComb->nb[1]);
      vCur = ggml_add(vCtx0, vCur, ggml_mul(vCtx0, vResSrc, vCombSrcDst));
    }

    vCur = ggml_reshape_3d(vCtx0, vCur, cNEmbd, 1, cNt);
    vOut = vOut ? ggml_concat(vCtx0, vOut, vCur, 1) : vCur;
  }

  return vOut;
}

ggml_tensor *mgs_model_deepseek41::graph::mBuildEngram(const mgs_model &pModel, llm_graph_input_dsv41_engram *pInp,
                                                       ggml_tensor *pX, int pE, int pIl) const {
  const auto &vLayer = pModel.lLayers[pIl];
  const int64_t cHc = vHparams.vDsv4HcMult;
  const int64_t cHcDim = cHc * cNEmbd;
  const int64_t cNt = pX->ne[2];

  // one projection makes a key per hc copy plus one value they all share
  ggml_tensor *vKv = mBuildLoraMm(vLayer.vEngramWkv, pInp->lEmb[pE]);
  mCb(vKv, "engram_kv", pIl);

  ggml_tensor *vKey = ggml_cont(vCtx0, ggml_view_2d(vCtx0, vKv, cHcDim, cNt, vKv->nb[1], 0));
  vKey = ggml_reshape_3d(vCtx0, vKey, cNEmbd, cHc, cNt);
  ggml_tensor *vValue = ggml_cont(vCtx0, ggml_view_2d(vCtx0, vKv, cNEmbd, cNt, vKv->nb[1], cHcDim * vKv->nb[0]));

  // the gate weights reach ggml_mul, which needs float: get_rows dequantizes quantized ones
  const auto cAsF32 = [&](ggml_tensor *pW) {
    if (pW->type == GGML_TYPE_F32) {
      return pW;
    }
    return ggml_get_rows(vCtx0, pW, pInp->vHcIds);
  };

  // the reference keeps the query and key weights apart but only ever uses their product
  ggml_tensor *vW = ggml_mul(vCtx0, cAsF32(vLayer.vEngramQ), cAsF32(vLayer.vEngramK));

  // normalized per (token, hc copy) over n_embd, not jointly over the copies
  ggml_tensor *vH = ggml_rms_norm(vCtx0, pX, cNormRmsEps);
  vKey = ggml_rms_norm(vCtx0, vKey, cNormRmsEps);

  ggml_tensor *vDot = ggml_sum_rows(vCtx0, ggml_mul(vCtx0, ggml_mul(vCtx0, vH, vW), vKey));
  vDot = ggml_scale(vCtx0, vDot, 1.0f / sqrtf((float)cNEmbd));

  // signed square root before the sigmoid, matching the training kernel
  ggml_tensor *vMag = ggml_sqrt(vCtx0, ggml_clamp(vCtx0, ggml_abs(vCtx0, vDot), 1e-6f, INFINITY));
  ggml_tensor *vGate = ggml_sigmoid(vCtx0, ggml_mul(vCtx0, ggml_sgn(vCtx0, vDot), vMag));
  mCb(vGate, "engram_gate", pIl);

  // the value is shared across the copies, only the gate differs
  ggml_tensor *vV = ggml_reshape_3d(vCtx0, vValue, cNEmbd, 1, cNt);
  vV = ggml_repeat_4d(vCtx0, vV, cNEmbd, cHc, cNt, 1);

  return ggml_add(vCtx0, pX, ggml_mul(vCtx0, vV, vGate));
}

ggml_tensor *mgs_model_deepseek41::graph::mBuildRopeTail(ggml_tensor *pX, ggml_tensor *pPos, bool pCompressed,
                                                         bool pBack, int pIl) const {
  GGML_UNUSED(pIl);

  const int64_t cHeadDim = pX->ne[0];
  const int64_t cNRot = vHparams.mNRot();
  const int64_t cNope = cHeadDim - cNRot;

  GGML_ASSERT(cHeadDim >= cNRot);

  // a layer with a compressed stream rotates with the compressor base and YaRN; a window-only layer
  // with the model base and no YaRN
  const float cBase = pCompressed ? vHparams.vDsv4CompressRopeBase : cFreqBase;
  const float cScale = pCompressed ? cFreqScale : 1.0f;
  const float cExt = pCompressed ? cExtFactor : 0.0f;
  const float cAttn = fDsv41RopeAttnFactor(cScale, cExt);
  const float cBf = pCompressed ? cBetaFast : 0.0f;
  const float cBs = pCompressed ? cBetaSlow : 0.0f;
  const int32_t cCtxOrig = pCompressed ? cNCtxOrig : 0;

  ggml_tensor *vNope = ggml_view_3d(vCtx0, pX, cNope, pX->ne[1], pX->ne[2], pX->nb[1], pX->nb[2], 0);
  ggml_tensor *vPe = ggml_view_3d(vCtx0, pX, cNRot, pX->ne[1], pX->ne[2], pX->nb[1], pX->nb[2],
                                  ggml_row_size(pX->type, cNope));

  if (pBack) {
    vPe = ggml_rope_ext_back(vCtx0, vPe, pPos, nullptr, cNRot, cRopeType, cCtxOrig, cBase, cScale, cExt, cAttn, cBf,
                             cBs);
  } else {
    vPe = ggml_rope_ext(vCtx0, vPe, pPos, nullptr, cNRot, cRopeType, cCtxOrig, cBase, cScale, cExt, cAttn, cBf, cBs);
  }

  return ggml_concat(vCtx0, vNope, vPe, 0);
}

ggml_tensor *mgs_model_deepseek41::graph::mBuildPooledLatent(ggml_tensor *pKvState, ggml_tensor *pScoreState,
                                                             ggml_tensor *pStateReadIdxs, int64_t pNBlocks,
                                                             int64_t pRatio, ggml_tensor *pNorm, int pIl) const {
  const int64_t cHeadDim = pKvState->ne[0];

  GGML_ASSERT(pNBlocks > 0);
  GGML_ASSERT(pStateReadIdxs->ne[0] == pRatio * pNBlocks);

  ggml_tensor *vKv = ggml_get_rows(vCtx0, pKvState, pStateReadIdxs);
  vKv = ggml_reshape_3d(vCtx0, vKv, cHeadDim, pRatio, pNBlocks);

  ggml_tensor *vScore = ggml_get_rows(vCtx0, pScoreState, pStateReadIdxs);
  vScore = ggml_reshape_3d(vCtx0, vScore, cHeadDim, pRatio, pNBlocks);

  // softmax-gated pooling over the ratio tokens of every group (a single token at ratio 1)
  ggml_tensor *vValues = ggml_cont(vCtx0, ggml_permute(vCtx0, vKv, 1, 0, 2, 3));
  ggml_tensor *vScores = ggml_cont(vCtx0, ggml_permute(vCtx0, vScore, 1, 0, 2, 3));

  ggml_tensor *vWeights = ggml_soft_max(vCtx0, vScores);
  ggml_tensor *vComp = ggml_mul(vCtx0, vValues, vWeights);
  vComp = ggml_sum_rows(vCtx0, vComp);
  vComp = ggml_cont(vCtx0, ggml_permute(vCtx0, vComp, 1, 0, 2, 3));
  mCb(vComp, "comp_pooled", pIl);

  vComp = mBuildNorm(vComp, pNorm, nullptr, cLlmNormRms, pIl);
  mCb(vComp, "comp_latent", pIl);

  return vComp;
}

ggml_tensor *mgs_model_deepseek41::graph::mBuildTopKMask(ggml_tensor *pKqMask, ggml_tensor *pTopK, const char *pName,
                                                         int pIl) const {
  GGML_ASSERT(pKqMask);
  GGML_ASSERT(pTopK);

  ggml_tensor *vKqMaskAll = ggml_fill(vCtx0, pKqMask, -INFINITY);
  vKqMaskAll = ggml_view_4d(vCtx0, vKqMaskAll, 1, vKqMaskAll->ne[0], vKqMaskAll->ne[1], vKqMaskAll->ne[3],
                            vKqMaskAll->nb[0], vKqMaskAll->nb[1], vKqMaskAll->nb[2], 0);

  ggml_tensor *vTopK3d = ggml_view_4d(vCtx0, pTopK, pTopK->ne[0], pTopK->ne[1], pTopK->ne[3], 1, pTopK->nb[1],
                                      pTopK->nb[2], pTopK->ne[3] * pTopK->nb[3], 0);

  ggml_tensor *vZeros =
      ggml_new_tensor_4d(vCtx0, pKqMask->type, 1, vTopK3d->ne[0], vTopK3d->ne[1], vTopK3d->ne[2]);
  vZeros = ggml_fill(vCtx0, vZeros, 0.0f);

  ggml_tensor *vKqMaskTopK = ggml_set_rows(vCtx0, vKqMaskAll, vZeros, vTopK3d);
  vKqMaskTopK = ggml_view_4d(vCtx0, vKqMaskTopK, vKqMaskTopK->ne[1], vKqMaskTopK->ne[2], 1, vKqMaskTopK->ne[3],
                             vKqMaskTopK->nb[2], vKqMaskTopK->nb[3], vKqMaskTopK->nb[3], 0);

  vKqMaskTopK = ggml_add(vCtx0, vKqMaskTopK, pKqMask);
  mCb(vKqMaskTopK, pName, pIl);

  return vKqMaskTopK;
}

// Scores this layer's queries against the shared index keys and keeps the best compressed rows.
ggml_tensor *mgs_model_deepseek41::graph::mBuildIndexerTopK(const mgs_model &pModel, llm_graph_input_dsv41 *pInp,
                                                            uint32_t pG, ggml_tensor *pQr, ggml_tensor *pCur,
                                                            ggml_tensor *pInpPos, int pIl) {
  const auto &vLayer = pModel.lLayers[pIl];
  const auto &vComp = pInp->lComp[pG];
  const int64_t cNIdxHeadDim = vHparams.vIndexerHeadSize;
  const int64_t cNIdxHead = vHparams.vIndexerNHead;
  const int64_t cNt = pCur->ne[1];

  GGML_ASSERT(vComp.vIdxMask);
  GGML_ASSERT(pInp->vMctx->mGetIdx(pG) != nullptr);

  ggml_tensor *vIdxQ = mBuildLoraMm(vLayer.vIndexerAttnQB, pQr);
  vIdxQ = ggml_reshape_3d(vCtx0, vIdxQ, cNIdxHeadDim, cNIdxHead, cNt);
  vIdxQ = mBuildRopeTail(vIdxQ, pInpPos, true, false, pIl);
  if (vComp.vIdxKRot) {
    vIdxQ = mgs_mul_mat_hadamard(vCtx0, vIdxQ, vComp.vIdxKRot);
  }
  mCb(vIdxQ, "idx_q", pIl);

  // one weight per head, scaled so that the score matches softmax_scale * n_heads**-0.5
  ggml_tensor *vIdxW = mBuildLoraMm(vLayer.vIndexerProj, pCur);
  vIdxW = ggml_scale(vCtx0, vIdxW, 1.0f / sqrtf(float(cNIdxHeadDim * cNIdxHead)));
  mCb(vIdxW, "idx_weights", pIl);

  ggml_tensor *vIdxK = pInp->vMctx->mGetIdx(pG)->mGetK(vCtx0, vHparams.aDsv41IndexKeySource[pIl]);
  const int64_t cNComp = vComp.vIdxMask->ne[0];
  GGML_ASSERT(cNComp > 0);
  GGML_ASSERT(cNComp <= vIdxK->ne[2]);

  vIdxK = ggml_view_4d(vCtx0, vIdxK, vIdxK->ne[0], vIdxK->ne[1], cNComp, vIdxK->ne[3], vIdxK->nb[1], vIdxK->nb[2],
                       vIdxK->nb[3], 0);
  mCb(vIdxK, "idx_k", pIl);

  const int64_t cNStream = vIdxK->ne[3];
  vIdxQ = ggml_view_4d(vCtx0, vIdxQ, vIdxQ->ne[0], vIdxQ->ne[1], vIdxQ->ne[2] / cNStream, cNStream, vIdxQ->nb[1],
                       vIdxQ->nb[2], vIdxQ->nb[3] / cNStream, 0);
  vIdxW = ggml_view_4d(vCtx0, vIdxW, vIdxW->ne[0], vIdxW->ne[1] / cNStream, vIdxW->ne[2], cNStream, vIdxW->nb[1],
                       vIdxW->nb[2] / cNStream, vIdxW->nb[3] / cNStream, 0);

  ggml_tensor *vScore = nullptr;
  if (vCparams.vFusedLid) {
    vScore = ggml_lightning_indexer(vCtx0, vIdxQ, vIdxK, vIdxW, vComp.vIdxMask);
    vRes->mAddFusedNode({cLlmFusedOpLightningIndexer, vScore, pIl});
  } else {
    ggml_tensor *vQ = ggml_permute(vCtx0, vIdxQ, 0, 2, 1, 3);
    ggml_tensor *vK = ggml_permute(vCtx0, vIdxK, 0, 2, 1, 3);

    ggml_tensor *vKq = ggml_mul_mat(vCtx0, vK, vQ);
    vKq = ggml_cont(vCtx0, ggml_permute(vCtx0, vKq, 2, 1, 0, 3));

    vScore = ggml_relu(vCtx0, vKq);
    vScore = ggml_mul(vCtx0, vScore, vIdxW);
    vScore = ggml_sum_rows(vCtx0, vScore);
    vScore = ggml_cont(vCtx0, ggml_permute(vCtx0, vScore, 2, 1, 0, 3));

    ggml_tensor *vMask = vComp.vIdxMask;
    if (vMask->type != vScore->type) {
      vMask = ggml_cast(vCtx0, vMask, vScore->type);
    }
    vScore = ggml_add(vCtx0, vScore, vMask);
  }
  mCb(vScore, "idx_score", pIl);

  // two-level top-k: the candidate source keeps its best blocks of rows (pinning the block of the
  // newest row, which is only partly filled), and the later index sources choose inside them
  const int32_t cCandSource = vHparams.vDsv41CandidateSource;
  if (cCandSource == pIl && vComp.vCandPin) {
    const int64_t cBlockSize = vHparams.vDsv41CandidateBlockSize;
    const int64_t cNBlocks = vComp.vCandPin->ne[0];

    GGML_ASSERT(vScore->ne[0] == cNBlocks * cBlockSize);

    ggml_tensor *vBlocks = ggml_pool_2d(vCtx0, vScore, GGML_OP_POOL_MAX, (int)cBlockSize, 1, (int)cBlockSize, 1,
                                        0.0f, 0.0f);
    vBlocks = ggml_add(vCtx0, vBlocks, vComp.vCandPin);
    mCb(vBlocks, "idx_block_score", pIl);

    const int64_t cNKeep = std::min<int64_t>(cNBlocks, vHparams.vDsv41CandidateTopKBlocks);
    ggml_tensor *vTopBlocks = ggml_cont(vCtx0, ggml_top_k(vCtx0, vBlocks, (int)cNKeep));

    // a zero mask shaped like the pin (the pin itself holds infinities)
    ggml_tensor *vBlockZero = ggml_fill(vCtx0, vComp.vCandPin, 0.0f);
    ggml_tensor *vBlockMask = mBuildTopKMask(vBlockZero, vTopBlocks, "idx_cand_blocks", pIl);
    vBlockMask = ggml_reshape_4d(vCtx0, vBlockMask, 1, cNBlocks, vBlockMask->ne[1], vBlockMask->ne[3]);
    vBlockMask = ggml_repeat_4d(vCtx0, vBlockMask, cBlockSize, cNBlocks, vBlockMask->ne[2], vBlockMask->ne[3]);
    vCandidateMask = ggml_reshape_4d(vCtx0, vBlockMask, cBlockSize * cNBlocks, vBlockMask->ne[2], 1,
                                     vBlockMask->ne[3]);
    mCb(vCandidateMask, "idx_cand_mask", pIl);
  } else if (cCandSource >= 0 && cCandSource < pIl && vCandidateMask &&
             vHparams.aDsv4CompressRatios[cCandSource] == vHparams.aDsv4CompressRatios[pIl]) {
    vScore = ggml_add(vCtx0, vScore, vCandidateMask);
    mCb(vScore, "idx_score_cand", pIl);
  }

  const uint32_t cNTopK = vScore->ne[0] < vHparams.vIndexerTopK ? vScore->ne[0] : vHparams.vIndexerTopK;
  ggml_tensor *vTopK = ggml_cont(vCtx0, ggml_top_k(vCtx0, vScore, cNTopK));
  mCb(vTopK, "idx_top_k", pIl);

  return vTopK;
}

ggml_tensor *mgs_model_deepseek41::graph::mBuildAttention(const mgs_model &pModel, llm_graph_input_dsv41 *pInp,
                                                          ggml_tensor *pCur, ggml_tensor *pInpPos, int pIl) {
  const auto &vLayer = pModel.lLayers[pIl];
  const auto *vMctxCur = pInp->vMctx;
  const auto *vRaw = vMctxCur->mGetRaw();

  const int64_t cNEmbdHead = vHparams.mNEmbdHeadK();
  const int64_t cNGroups = vHparams.vDsv4OGroupCount;
  const int64_t cOLoraRank = vHparams.vDsv4OLoraRank;
  const int64_t cOGroupDim = (cNHead / cNGroups) * cNEmbdHead;
  const int64_t cNt = pCur->ne[1];
  const uint32_t cRatio = vHparams.aDsv4CompressRatios[pIl];
  const bool cCompressed = cRatio != 0;

  GGML_ASSERT(cNEmbdHead == cNEmbdHeadV);
  GGML_ASSERT(cNHead % cNGroups == 0);

  // query: V4.1 normalizes only the low-rank part
  ggml_tensor *vQr = mBuildLoraMm(vLayer.vWqA, pCur);
  vQr = mBuildNorm(vQr, vLayer.vAttnQANorm, nullptr, cLlmNormRms, pIl);
  mCb(vQr, "qr", pIl);

  ggml_tensor *vQ = mBuildLoraMm(vLayer.vWqB, vQr);
  vQ = ggml_reshape_3d(vCtx0, vQ, cNEmbdHead, cNHead, cNt);
  vQ = mBuildRopeTail(vQ, pInpPos, cCompressed, false, pIl);
  mCb(vQ, "q", pIl);

  // the sliding-window KV, which every layer keeps for itself
  ggml_tensor *vKv = mBuildLoraMm(vLayer.vWkv, pCur);
  vKv = mBuildNorm(vKv, vLayer.vAttnKvNorm, nullptr, cLlmNormRms, pIl);
  vKv = ggml_reshape_3d(vCtx0, vKv, cNEmbdHead, 1, cNt);
  vKv = mBuildRopeTail(vKv, pInpPos, cCompressed, false, pIl);
  mCb(vKv, "kv", pIl);

  // the compression group of this layer
  int32_t vG = -1;
  if (cCompressed) {
    for (uint32_t vI = 0; vI < vMctxCur->mGetNGroup(); ++vI) {
      if (vMctxCur->mGetRatio(vI) == cRatio) {
        vG = (int32_t)vI;
      }
    }
    GGML_ASSERT(vG >= 0 && "a compressed layer needs the group of its ratio");
  }

  const llm_graph_input_dsv41::comp_input *vComp = vG >= 0 ? &pInp->lComp[vG] : nullptr;

  // a KV source pools the completed groups of its compressor state into new compressed rows
  if (vComp && vHparams.mIsDsv41KvSource(pIl) && vComp->vStatePos) {
    const mgs_dsv41_comp_state *vState = vMctxCur->mGetState(vG);

    ggml_tensor *vStateKv = mBuildLoraMm(vLayer.vAttnCompWkv, pCur);
    mCb(vStateKv, "comp_state_kv", pIl);

    // at ratio 1 there is nothing to pool and no gate: the softmax over a single token returns 1
    ggml_tensor *vStateScore = vLayer.vAttnCompWgate ? mBuildLoraMm(vLayer.vAttnCompWgate, pCur) : vStateKv;
    mCb(vStateScore, "comp_state_score", pIl);

    ggml_tensor *vRestoredKv = vState->mGetKvAll(vCtx0, pIl);
    ggml_tensor *vRestoredScore = vState->mGetScoreAll(vCtx0, pIl);
    if (vComp->vStateRestoreSrcIdxs && vComp->vStateRestoreDstIdxs) {
      vRestoredKv = vState->mCpyKv(vCtx0, ggml_get_rows(vCtx0, vRestoredKv, vComp->vStateRestoreSrcIdxs),
                                   vComp->vStateRestoreDstIdxs, pIl);
      vRestoredScore = vState->mCpyScore(vCtx0, ggml_get_rows(vCtx0, vRestoredScore, vComp->vStateRestoreSrcIdxs),
                                         vComp->vStateRestoreDstIdxs, pIl);
    }

    ggml_tensor *vBaseKv = fDsv41View2d(vCtx0, vRestoredKv, vRestoredKv->ne[0], vState->mGetNRows(), 0);
    ggml_tensor *vBaseScore = fDsv41View2d(vCtx0, vRestoredScore, vRestoredScore->ne[0], vState->mGetNRows(), 0);

    ggml_tensor *vSourceKv = ggml_concat(vCtx0, vBaseKv, vStateKv, 1);
    ggml_tensor *vSourceScore = ggml_concat(vCtx0, vBaseScore, vStateScore, 1);

    if (vComp->vStateWriteIdxs) {
      const int64_t cNBlocks = vComp->vStateWritePos->ne[0];

      // the index keys come from the latent before RoPE, so build both forms
      ggml_tensor *vLatent =
          mBuildPooledLatent(vSourceKv, vSourceScore, vComp->vStateReadIdxs, cNBlocks, cRatio, vLayer.vAttnCompNorm,
                             pIl);

      if (vHparams.mOwnsDsv41IndexKey(pIl)) {
        const auto *vIdxCtx = vMctxCur->mGetIdx(vG);
        GGML_ASSERT(vIdxCtx != nullptr);

        ggml_tensor *vIdxK = mBuildLoraMm(vLayer.vIndexerAttnK, ggml_reshape_2d(vCtx0, vLatent, cNEmbdHead, cNBlocks));
        vIdxK = mBuildNorm(vIdxK, vLayer.vIndexerKNorm, nullptr, cLlmNormRms, pIl);
        vIdxK = ggml_reshape_3d(vCtx0, vIdxK, vHparams.vIndexerHeadSize, 1, cNBlocks);
        vIdxK = mBuildRopeTail(vIdxK, vComp->vStateWritePos, true, false, pIl);
        if (vComp->vIdxKRot) {
          vIdxK = mgs_mul_mat_hadamard(vCtx0, vIdxK, vComp->vIdxKRot);
        }
        mCb(vIdxK, "idx_k_new", pIl);

        ggml_build_forward_expand(vGf, vIdxCtx->mCpyK(vCtx0, vIdxK, vComp->vStateWriteIdxs, pIl));
      }

      // a row stands for the first token of its group
      ggml_tensor *vRow = mBuildRopeTail(vLatent, vComp->vStateWritePos, true, false, pIl);
      if (vComp->vKRot) {
        vRow = mgs_mul_mat_hadamard(vCtx0, vRow, vComp->vKRot);
      }
      mCb(vRow, "comp_row", pIl);

      ggml_build_forward_expand(vGf, vMctxCur->mGetComp(vG)->mCpyK(vCtx0, vRow, vComp->vStateWriteIdxs, pIl));
    }

    // rollback snapshots of [restored state | this ubatch]
    if (vComp->vStateSnapshotSrcIdxs && vComp->vStateSnapshotDstIdxs) {
      ggml_tensor *vSnapKv = ggml_concat(vCtx0, vRestoredKv, vStateKv, 1);
      ggml_tensor *vSnapScore = ggml_concat(vCtx0, vRestoredScore, vStateScore, 1);

      ggml_build_forward_expand(vGf, vState->mCpyKv(vCtx0, ggml_get_rows(vCtx0, vSnapKv, vComp->vStateSnapshotSrcIdxs),
                                                    vComp->vStateSnapshotDstIdxs, pIl));
      ggml_build_forward_expand(vGf,
                                vState->mCpyScore(vCtx0, ggml_get_rows(vCtx0, vSnapScore, vComp->vStateSnapshotSrcIdxs),
                                                  vComp->vStateSnapshotDstIdxs, pIl));
    }

    // carry whatever did not complete a group into the next ubatch
    ggml_tensor *vPersistKv = ggml_get_rows(vCtx0, vStateKv, vComp->vStatePersistSrcIdxs);
    ggml_tensor *vPersistScore = ggml_get_rows(vCtx0, vStateScore, vComp->vStatePersistSrcIdxs);

    ggml_build_forward_expand(vGf, vState->mCpyKv(vCtx0, vPersistKv, vComp->vStatePersistDstIdxs, pIl));
    ggml_build_forward_expand(vGf, vState->mCpyScore(vCtx0, vPersistScore, vComp->vStatePersistDstIdxs, pIl));
  }

  // an index source picks the rows; the layers after it reuse what it picked
  ggml_tensor *vCompMask = nullptr;
  if (vComp && vComp->vKqMask) {
    if (vHparams.mIsDsv41IndexSource(pIl)) {
      ggml_tensor *vTopK = mBuildIndexerTopK(pModel, pInp, (uint32_t)vG, vQr, pCur, pInpPos, pIl);
      lTopKMask[pIl] = mBuildTopKMask(vComp->vKqMask, vTopK, "comp_top_k_mask", pIl);
    }
    vCompMask = lTopKMask[vHparams.aDsv41TopKSource[pIl]];
    GGML_ASSERT(vCompMask && "a compressed layer reads the top-k of an index source");
  }

  ggml_tensor *vKRot = pInp->vSelfKRot;
  if (vKRot) {
    vQ = mgs_mul_mat_hadamard(vCtx0, vQ, vKRot);
    vKv = mgs_mul_mat_hadamard(vCtx0, vKv, vKRot);
  }

  ggml_build_forward_expand(vGf, vQ);
  ggml_build_forward_expand(vGf, vKv);
  ggml_build_forward_expand(vGf, vRaw->mCpyK(vCtx0, vKv, pInp->vSelfKIdxs, pIl));

  ggml_tensor *vK = vRaw->mGetK(vCtx0, pIl);
  ggml_tensor *vKqMask = pInp->vSelfKqMask;
  mCb(vK, "raw_k", pIl);

  if (vCompMask) {
    ggml_tensor *vCompK = vMctxCur->mGetComp(vG)->mGetK(vCtx0, vHparams.aDsv41KvSource[pIl]);
    const int64_t cNComp = vComp->vKqMask->ne[0];
    GGML_ASSERT(cNComp <= vCompK->ne[2]);

    vCompK = ggml_view_4d(vCtx0, vCompK, vCompK->ne[0], vCompK->ne[1], cNComp, vCompK->ne[3], vCompK->nb[1],
                          vCompK->nb[2], vCompK->nb[3], 0);
    mCb(vCompK, "comp_k", pIl);

    vK = ggml_concat(vCtx0, vK, vCompK, 2);
    vKqMask = ggml_concat(vCtx0, vKqMask, vCompMask, 0);
    mCb(vK, "k_all", pIl);
    mCb(vKqMask, "kq_mask_all", pIl);
  }

  ggml_tensor *vOut = mBuildAttnMha(vQ, vK, vK, nullptr, vKqMask, vLayer.vAttnSinks, nullptr,
                                    1.0f / sqrtf(float(cNEmbdHead)), pIl);
  if (vKRot) {
    vOut = mgs_mul_mat_hadamard(vCtx0, vOut, vKRot);
  }
  mCb(vOut, "attn_mha", pIl);

  // remove the rotation the query carried into attention, then the grouped output projection
  vOut = ggml_reshape_3d(vCtx0, vOut, cNEmbdHead, cNHead, cNt);
  vOut = mBuildRopeTail(vOut, pInpPos, cCompressed, true, pIl);
  mCb(vOut, "attn_derope", pIl);

  vOut = ggml_reshape_3d(vCtx0, vOut, cOGroupDim, cNGroups, cNt);
  vOut = ggml_permute(vCtx0, vOut, 0, 2, 1, 3);
  ggml_tensor *vOa = ggml_mul_mat(vCtx0, vLayer.vWoA, vOut);
  mCb(vOa, "attn_wo_a", pIl);
  vOa = ggml_permute(vCtx0, vOa, 0, 2, 1, 3);
  vOa = ggml_cont_2d(vCtx0, vOa, cOLoraRank * cNGroups, cNt);

  vOut = mBuildLoraMm(vLayer.vWoB, vOa);
  mCb(vOut, "attn_out", pIl);

  return vOut;
}

mgs_model_deepseek41::graph::graph(const mgs_model &pModel, const llm_graph_params &pParams)
    : llm_graph_context(pParams) {
  const auto &vModel41 = static_cast<const mgs_model_deepseek41 &>(pModel);

  lTopKMask.assign(cNLayer, nullptr);

  ggml_tensor *vCur;

  ggml_tensor *vInp = mBuildInpEmbd(pModel.vTokEmbd);
  ggml_tensor *vInpPos = mBuildInpPos();
  ggml_tensor *vInpOutIds = mBuildInpOutIds();
  llm_graph_input_dsv41 *vInpDsv41 = mBuildInpDsv41();
  ggml_build_forward_expand(vGf, vInpDsv41->vSelfKqMask);

  llm_graph_input_dsv41_engram *vInpEngram = vModel41.lEngramLayerIds.empty() ? nullptr : mBuildInpEngram(vModel41);

  const int64_t cHc = vHparams.vDsv4HcMult;
  ggml_tensor *vInpL = ggml_reshape_3d(vCtx0, vInp, cNEmbd, 1, cNTokens);
  vInpL = ggml_repeat_4d(vCtx0, vInpL, cNEmbd, cHc, cNTokens, 1);
  mCb(vInpL, "hc_init", -1);

  // the mix the next attention collapses the copies with; layer 0 has none and takes the first copy
  ggml_tensor *vPreMix = nullptr;

  for (int vIl = 0; vIl < cNLayer; ++vIl) {
    const auto &vLayer = pModel.lLayers[vIl];

    if ((size_t)vIl < vCparams.lEmbeddingsLayerInp.size() && vCparams.lEmbeddingsLayerInp[vIl]) {
      vRes->lTLayerInp[vIl] = fDsv41HcMean(vCtx0, vInpL);
      mCb(vRes->lTLayerInp[vIl], "layer_inp", vIl);
      ggml_build_forward_expand(vGf, vRes->lTLayerInp[vIl]);
    }

    // the engram sits before the block and writes straight into the stream
    const int cE = vModel41.mEngramIndex(vIl);
    if (cE >= 0 && vInpEngram) {
      vInpL = mBuildEngram(pModel, vInpEngram, vInpL, cE, vIl);
      mCb(vInpL, "engram_out", vIl);
    }

    ggml_tensor *vResidual = vInpL;
    ggml_tensor *vAttnPre = nullptr;
    ggml_tensor *vPost = nullptr;
    ggml_tensor *vComb = nullptr;

    // this sublayer's pre mix is for the FFN; the attention collapses with the incoming one
    mBuildHcMixes(vInpL, vLayer.vHcAttnFn, vLayer.vHcAttnScale, vLayer.vHcAttnBase, &vAttnPre, &vPost, &vComb, vIl);

    if (vPreMix) {
      vCur = mBuildHcPre(vInpL, vPreMix, vIl);
    } else {
      vCur = ggml_cont(vCtx0, ggml_view_2d(vCtx0, vInpL, cNEmbd, cNTokens, vInpL->nb[2], 0));
    }
    mCb(vCur, "hc_attn_pre", vIl);

    vCur = mBuildNorm(vCur, vLayer.vAttnNorm, nullptr, cLlmNormRms, vIl);
    mCb(vCur, "attn_norm", vIl);

    vCur = mBuildAttention(pModel, vInpDsv41, vCur, vInpPos, vIl);

    vInpL = mBuildHcPost(vCur, vResidual, vPost, vComb, vIl);
    mCb(vInpL, "hc_attn_post", vIl);

    vResidual = vInpL;

    // the FFN mix is what the next layer's attention collapses with
    mBuildHcMixes(vInpL, vLayer.vHcFfnFn, vLayer.vHcFfnScale, vLayer.vHcFfnBase, &vPreMix, &vPost, &vComb, vIl);

    vCur = mBuildHcPre(vInpL, vAttnPre, vIl);
    mCb(vCur, "hc_ffn_pre", vIl);

    ggml_build_forward_expand(vGf, vResidual);
    ggml_build_forward_expand(vGf, vPost);
    ggml_build_forward_expand(vGf, vComb);

    vCur = mBuildNorm(vCur, vLayer.vFfnNorm, nullptr, cLlmNormRms, vIl);
    mCb(vCur, "ffn_norm", vIl);

    // image embeddings route with their own bias
    ggml_tensor *vExpProbsB = vLayer.vFfnExpProbsB;
    if (vUbatch.vEmbd != nullptr && vLayer.vFfnExpProbsBVl) {
      vExpProbsB = vLayer.vFfnExpProbsBVl;
    }

    ggml_tensor *vMoeOut =
        mBuildMoeFfn(vCur, vLayer.vFfnGateInp, vLayer.vFfnUpExps, vLayer.vFfnGateExps, vLayer.vFfnDownExps, vExpProbsB,
                     cNExpert, vHparams.vNExpertUsed, cLlmFfnSilu, vHparams.vExpertWeightsNorm,
                     vHparams.vExpertWeightsScale, (mgs_expert_gating_func_type)vHparams.vExpertGatingFunc, vIl,
                     nullptr, vLayer.vFfnGateUpExps);
    mCb(vMoeOut, "ffn_moe_out", vIl);

    ggml_tensor *vFfnShexp =
        mBuildFfn(vCur, vLayer.vFfnUpShexp, nullptr, nullptr, vLayer.vFfnGateShexp, nullptr, nullptr,
                  vLayer.vFfnDownShexp, nullptr, nullptr, nullptr, cLlmFfnSilu, cLlmFfnPar, vIl);
    mCb(vFfnShexp, "ffn_shexp", vIl);

    vCur = ggml_add(vCtx0, vMoeOut, vFfnShexp);
    mCb(vCur, "ffn_out", vIl);

    vInpL = mBuildHcPost(vCur, vResidual, vPost, vComb, vIl);
    vInpL = mBuildCvec(vInpL, vIl);
    mCb(vInpL, "l_last", vIl);
  }

  if ((size_t)cNLayer < vCparams.lEmbeddingsLayerInp.size() && vCparams.lEmbeddingsLayerInp[cNLayer]) {
    vRes->lTLayerInp[cNLayer] = fDsv41HcMean(vCtx0, vInpL);
    mCb(vRes->lTLayerInp[cNLayer], "layer_inp", cNLayer);
    ggml_build_forward_expand(vGf, vRes->lTLayerInp[cNLayer]);
  }

  if (vInpOutIds) {
    ggml_tensor *vFlat = ggml_reshape_2d(vCtx0, vInpL, cNEmbd * cHc, cNTokens);
    vInpL = ggml_reshape_3d(vCtx0, ggml_get_rows(vCtx0, vFlat, vInpOutIds), cNEmbd, cHc, cNOutputs);
    vPreMix = ggml_get_rows(vCtx0, vPreMix, vInpOutIds);
  }

  // the last FFN mix is the one nothing has consumed: it collapses the copies for the output head
  vCur = mBuildHcPre(vInpL, vPreMix, -1);
  mCb(vCur, "hc_out", -1);

  vCur = mBuildNorm(vCur, pModel.vOutputNorm, nullptr, cLlmNormRms, -1);
  mCb(vCur, "result_norm", -1);
  vRes->vTEmbd = vCur;

  vCur = ggml_mul_mat(vCtx0, pModel.vOutput, vCur);
  mCb(vCur, "result_output", -1);
  vRes->vTLogits = vCur;

  ggml_build_forward_expand(vGf, vCur);
}
