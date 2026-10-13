#pragma once

#include "mgs.h"

#include <array>
#include <bitset>
#include <cassert>
#include <cmath>

// bump if necessary
#define MGS_MAX_LAYERS 512
#define MGS_MAX_EXPERTS 1024 // Kimi-K3
#define MGS_MAX_PLE_NGRAM 8  // qwen4exp
#define MGS_MAX_PLE_HEADS 64 // qwen4exp

enum mgs_expert_gating_func_type {
  cMgsExpertGatingFuncTypeNone = 0,
  cMgsExpertGatingFuncTypeSoftmax = 1,
  cMgsExpertGatingFuncTypeSigmoid = 2,
  cMgsExpertGatingFuncTypeSoftmaxWeight = 3, // applied to the router weights instead of the logits
  cMgsExpertGatingFuncTypeSqrtSoftplus = 4,
};

enum mgs_swa_type {
  cMgsSwaTypeNone = 0,
  cMgsSwaTypeStandard = 1,
  cMgsSwaTypeChunked = 2,
  cMgsSwaTypeSymmetric = 3,
};

// forward declaration; full definition in mgs-graph.h
enum llm_ffn_op_type : int;

struct mgs_hparams_posnet {
  uint32_t vNEmbd;
  uint32_t vNLayer;
};

struct mgs_hparams_convnext {
  uint32_t vNEmbd;
  uint32_t vNLayer;
};

struct mgs_hparams {
  // note: use the `Impl` suffix to avoid name conflict between members and getters
  //       for example: mNEmbdOut() vs vNEmbdOutImpl

  bool vVocabOnly;
  bool vNoAlloc;
  bool vRopeFinetuned;
  bool vUseParRes;
  bool vSwinNorm;
  bool vNormBeforeResidual = false;
  bool vNormBeforeFc = false;

  uint32_t vNCtxTrain; // context size the model was trained on
  uint32_t vNEmbd;
  uint32_t vNLayerAll;
  uint32_t vNLayerNextn = 0;
  uint32_t vNExpert = 0;
  uint32_t vNExpertUsed = 0;
  uint32_t vNRelAttnBkts = 0;

  // TODO: this needs to be reworked
  int32_t vNLayerKvFromStart = -1; // if non-negative, the first n_layer_kv_from_start layers have KV cache

  // different head size for full_attention and SWA layers
  uint32_t vNEmbdHeadKFull; // dimension of keys (d_k). d_q is assumed to be the same, but there are n_head q heads,
                            // and only n_head_kv k-v heads
  uint32_t vNEmbdHeadVFull; // dimension of values (d_v) aka n_embd_head
  uint32_t vNEmbdHeadKSwa;
  uint32_t vNEmbdHeadVSwa;

  // different RoPE dimensions for full_attention and SWA layers
  uint32_t vNRotFull;
  uint32_t vNRotSwa;

  // note: MLA converts into MQA with larger heads, then decompresses to MHA
  uint32_t vNEmbdHeadKMlaImpl = 0;
  uint32_t vNEmbdHeadVMlaImpl = 0;

  // for WavTokenizer
  struct mgs_hparams_posnet vPosnet;
  struct mgs_hparams_convnext vConvnext;

  uint32_t vNShortconvLCache = 0;

  std::array<uint32_t, MGS_MAX_LAYERS> aNHeadArr;
  std::array<uint32_t, MGS_MAX_LAYERS> aNHeadKvArr;
  std::array<uint32_t, MGS_MAX_LAYERS> aNFfArr;

  uint32_t vNLayerDenseLead = 0;
  uint32_t vNLoraQ = 0;
  uint32_t vNLoraKv = 0;
  uint32_t vNFfExp = 0;
  uint32_t vNFfShexp = 0;
  uint32_t vNFfChexp = 0;
  uint32_t vNExpertShared = 0;
  uint32_t vNNormGroups = 0;
  uint32_t vNExpertGroups = 0;
  uint32_t vNGroupUsed = 0;
  uint32_t vNGroupExperts = 0;

  float vExpertGroupScale = 0.05f;
  float vExpertWeightsScale = 0.0f;
  bool vExpertWeightsNorm = false;
  uint32_t vExpertGatingFunc = cMgsExpertGatingFuncTypeNone;
  uint32_t vMoeEveryNLayers = 0;
  uint32_t vMoeLatentSize = 0;

  float vFNormEps;
  float vFNormRmsEps;
  float vFNormGroupEps;

  float vFAttnLogitSoftcapping = 50.0f;
  float vFRouterLogitSoftcapping = 30.0f;
  float vFFinalLogitSoftcapping = 30.0f;

  // for RWKV
  uint32_t vRescaleEveryNLayers = 0;
  uint32_t vTimeMixExtraDim = 0;
  uint32_t vTimeDecayExtraDim = 0;
  uint32_t vWkvHeadSize = 0;
  uint32_t vTokenShiftCount = 2;
  uint32_t vNLoraDecay = 0;
  uint32_t vNLoraIclr = 0;
  uint32_t vNLoraValueResMix = 0;
  uint32_t vNLoraGate = 0;

  float vRopeAttnFactor = 1.0f;
  float vRopeFreqBaseTrain;
  float vRopeFreqBaseTrainSwa = 10000.0f;
  float vRopeFreqScaleTrain;
  float vRopeFreqScaleTrainSwa = 1.0f;
  float vRopeScalingAlpha = 0.0f; // NTK-aware alpha for XDRoPE

  uint32_t vNCtxOrigYarn;
  float vRopeYarnLogMul = 0.0f;

  float vYarnExtFactor = -1.0f;
  float vYarnAttnFactor = 1.0f;
  float vYarnBetaFast = 32.0f;
  float vYarnBetaSlow = 1.0f;

  std::array<int, 4> aRopeSections;

  // Sliding Window Attention (SWA)
  mgs_swa_type vSwaType = cMgsSwaTypeNone;
  // the size of the sliding window (0 - no SWA)
  uint32_t vNSwa = 0;

  // if is_swa_impl[il] == 1, then layer il is SWA
  // if is_swa_impl[il] == 0, then layer il is dense (i.e. non-SWA)
  // by default, all layers are dense
  // note: using uint32_t type for compatibility reason
  std::array<uint32_t, MGS_MAX_LAYERS> aIsSwaImpl;

  // for hybrid state space models
  std::array<uint32_t, MGS_MAX_LAYERS> aIsRecrImpl;

  // for State Space Models
  uint32_t vSsmDConv = 0;
  uint32_t vSsmDInner = 0;
  uint32_t vSsmDState = 0;
  uint32_t vSsmDtRank = 0;
  uint32_t vSsmNGroup = 0;

  // for Kimi Linear KDA
  uint32_t vNEmbdHeadKda = 0;
  // KDA decay gate lower bound (kimi-k3, glm5-next): -INFINITY disables the bounded sigmoid gate
  float vKdaGateLowerBound = -INFINITY;

  // kimi-k3
  uint32_t vNExpertLatent = 0;     // routed experts hidden size (0 = experts run at n_embd)
  uint32_t vAttnResBlockSize = 0;  // 0 = no cross-layer attention residuals
  float vSituBeta = 1.0f;          // situ activation: beta * tanh(gate / beta) * sigmoid(gate)
  float vSituLinearBeta = 0.0f;    // 0 = no linear-beta transform on the up branch

  bool vSsmDtBCRms = false;

  float vFClampKqv = 0.0f;
  float vFMaxAlibiBias = 0.0f;
  float vFLogitScale = 0.0f;

  // Additional scale factors (Granite/Granite MoE)
  float vFResidualScale = 0.0f;
  float vFEmbeddingScale = 0.0f;
  float vFAttentionScale = 0.0f;

  // grok-2
  float vFAttnOutScale = 0.0f;
  uint32_t vAttnTempLength = 0;

  float vFAttnValueScale = 0.0f;

  bool vCausalAttn = true;
  bool vUseAlibi = false;
  bool vAttnSoftCap = false;
  bool vUseKqNorm = false;

  // for Classifiers
  uint32_t vNClsOut = 1;

  // input embedding dimension (0 = use n_embd)
  uint32_t vNEmbdInpImpl = 0;

  // encoder input embedding dimension (0 = use mNEmbdInp())
  // e.g. the eagle3 encoder fuses target_layers * target_hidden features
  uint32_t vNEmbdInpEncImpl = 0;

  // output embedding dimension (0 = use n_embd)
  uint32_t vNEmbdOutImpl = 0;

  // smallthinker
  uint32_t vNMoeLayerStep = 0;
  uint32_t vNNoRopeLayerStep = 4;
  uint32_t vNAttnTempFloorScale = 0;
  float vFAttnTempScale = 0.0f;
  float vFAttnTempOffset = 0.0f; // offset position index

  // gemma3n altup
  uint32_t vNAltup = 4;    // altup_num_inputs
  uint32_t vIAltupAct = 0; // altup_active_idx
  uint32_t vLaurelRank = 64;
  uint32_t vNEmbdAltup = 256;

  // needed for sentence-transformers dense layers
  uint32_t vDense2FeatIn = 0;  // in_features of the 2_Dense
  uint32_t vDense2FeatOut = 0; // out_features of the 2_Dense
  uint32_t vDense3FeatIn = 0;  // in_features of the 3_Dense
  uint32_t vDense3FeatOut = 0; // out_features of the 3_Dense

  // xIELU
  std::array<float, MGS_MAX_LAYERS> aXieluAlphaN;
  std::array<float, MGS_MAX_LAYERS> aXieluAlphaP;
  std::array<float, MGS_MAX_LAYERS> aXieluBeta;
  std::array<float, MGS_MAX_LAYERS> aXieluEps;

  // DSA (deepseek sparse attention)
  uint32_t vIndexerNHead = 0;
  uint32_t vIndexerHeadSize = 0;
  uint32_t vIndexerTopK = 0;
  uint32_t vIndexerKpool = 0; // k-pool size
  bool vIndexerKpoolSelectTail = true;
  // head-size slots per cached indexer row, the last one holds the pooled key
  uint32_t vIndexerKpoolRow = 3;
  // pools are consecutive cells in sequence order, not runs of consecutive positions
  bool vIndexerKpoolByOrder = false;
  // MSA
  uint32_t vIndexerBlockSize = 0;
  uint32_t vIndexerLocalBlocks = 0;

  // Indexer is "full" (1) or "shared" (0)
  // Shared indexers reuse top-k from previous full layer
  std::array<uint32_t, MGS_MAX_LAYERS> aIsIndexerFullImpl;

  // DeepSeek-V4.1 (the hyper-connection fields are shared with other models)
  uint32_t vDsv4OGroupCount = 0;
  uint32_t vDsv4OLoraRank = 0;
  uint32_t vDsv4HcMult = 0;
  uint32_t vDsv4HcSinkhornIters = 0;
  float vDsv4CompressRopeBase = 0.0f;
  float vDsv4HcEps = 0.0f;
  std::array<uint32_t, MGS_MAX_LAYERS> aDsv4CompressRatios;

  // DeepSeek-V4.1 compresses KV on a few source layers and shares each stream with the layers that
  // follow, so for every layer these hold the layer that publishes what it reads, or -1 when it reads
  // nothing. A layer whose entry is itself is the source.
  std::array<int32_t, MGS_MAX_LAYERS> aDsv41KvSource;
  std::array<int32_t, MGS_MAX_LAYERS> aDsv41IndexKeySource;
  std::array<int32_t, MGS_MAX_LAYERS> aDsv41TopKSource;
  // two-level indexer top-k: the candidate source keeps its best blocks, the later index sources
  // choose only inside them (-1 disables it)
  int32_t vDsv41CandidateSource = -1;
  uint32_t vDsv41CandidateTopKBlocks = 0;
  uint32_t vDsv41CandidateBlockSize = 0;

  bool mIsDsv41KvSource(uint32_t pIl) const { return aDsv41KvSource[pIl] == (int32_t)pIl; }
  bool mOwnsDsv41IndexKey(uint32_t pIl) const { return aDsv41IndexKeySource[pIl] == (int32_t)pIl; }
  bool mIsDsv41IndexSource(uint32_t pIl) const { return aDsv41TopKSource[pIl] == (int32_t)pIl; }

  // DeepSeek-V4.1 engram: n-gram hash lookups added into the residual stream (0 heads = none)
  uint32_t vEngramNHead = 0;
  uint32_t vEngramKeyLength = 0;
  uint32_t vEngramMaxNgramSize = 0;

  // qwen4exp: rank of the hyper-connection mixer (0 = full rank, DeepSeek-V4)
  uint32_t vHcLowRank = 0;

  // qwen4exp per-layer n-gram hash embeddings (PLE)
  uint32_t vPleNgramSize = 0;
  uint32_t vPleHeadsPerNgram = 0;
  uint32_t vPleConvKernel = 0;
  uint32_t vPleNHeads = 0; // (ngram_size - 1) * heads_per_ngram
  uint32_t vPleHeadDim = 0;
  uint32_t vPleEosTokenId = 0;
  // the id the PLE hash stands in at image positions; 0 makes the loader fall back to EOS
  uint32_t vPleImageTokenId = 0;
  // the file lists PLE layer indices, so this is never a per-layer gguf array and can hold one bit per layer
  std::bitset<MGS_MAX_LAYERS> sIsPleImpl;
  // the hash multipliers reach ~2e13 and have to stay 64-bit
  std::array<uint64_t, MGS_MAX_PLE_NGRAM> aPleLayerMultipliers;
  // head offsets and vocab sizes are token-space indices; the gather truncates them to int32 anyway
  std::array<uint32_t, MGS_MAX_PLE_HEADS> aPleHeadOffsets;
  std::array<uint32_t, MGS_MAX_PLE_HEADS> aPleHeadVocabSizes;

  bool mIsPle(uint32_t pIl) const;

  // PLE conv history row width: (kernel - 1) * ngram_size streams of hc * n_embd; 0 without a PLE module
  uint32_t mPleConvState() const;

  // qwen3vl deepstack
  // When parsed from GGUF, this implies the first N layers consume the first
  // N deepstack embeddings. Use deepstack_mapping_arr if you need a more
  // complex mapping. If using deepstack_mapping_arr, also make sure to set
  // n_deepstack_layers to the number of unique deepstack layers so that
  // n_embd_imp is accurate (see granite.cpp).
  // TODO: can be expressed via the `new n_embd_inp_impl` and remove this param
  uint32_t vNDeepstackLayers = 0;

  // deepstack layer array (Granite4 Vision)
  // -1  => no deepstack
  // >=0 => input embedding index for deepstack injection
  std::array<int32_t, MGS_MAX_LAYERS> aDeepstackMappingArr;

  // needed by encoder-decoder models (e.g. T5, FLAN-T5)
  mgs_token vDecStartTokenId = MGS_TOKEN_NULL;
  uint32_t vDecNLayer = 0;

  enum mgs_pooling_type vPoolingType = MGS_POOLING_TYPE_NONE;
  enum mgs_rope_type vRopeType = MGS_ROPE_TYPE_NONE;
  enum mgs_rope_scaling_type vRopeScalingTypeTrain = MGS_ROPE_SCALING_TYPE_NONE;

  // Resolved FFN gated activation flavor for archs that read
  // `<arch>.hidden_activation` from the GGUF (e.g. ModernBert derivatives).
  // Defaults to LLM_FFN_NONE (sentinel = 0); the mapping from the GGUF
  // string to a real op is done at hparam-load time via
  // fLlmFfnOpTypeFromString() in mgs-model.cpp, mirroring how
  // rope_scaling_type_train is handled.
  enum llm_ffn_op_type vLlmFfnOp;

  // Step35: optional per-layer clamps for (Swi)GLU
  std::array<float, MGS_MAX_LAYERS> aSwigluClampExp;   // clamping for expert FFN
  std::array<float, MGS_MAX_LAYERS> aSwigluClampShexp; // shared expert

  // this value n_pattern means that every nth layer is dense (i.e. non-SWA)
  // dense_first means whether the pattern is start with a dense layer
  // note that if n_pattern == 0, all layers are SWA
  //           if n_pattern == 1, all layers are dense
  // example 1: n_pattern = 3, dense_first = false
  //   il == 0: swa
  //   il == 1: swa
  //   il == 2: dense
  //   il == 3: swa
  //   il == 4: swa
  //   il == 5: dense
  //   il == 6: swa
  //   etc ...
  // example 2: n_pattern = 2, dense_first = true
  //   il == 0: dense
  //   il == 1: swa
  //   il == 2: dense
  //   il == 3: swa
  //   etc ...
  void mSetSwaPattern(uint32_t pNPattern, bool pDenseFirst = false);

  // return true if one of the layers is SWA
  bool mIsSwaAny() const;

  bool mIsSwa(uint32_t pIl) const;

  bool mIsIndexerFull(uint32_t pIl) const;

  void mSetRecrPattern(uint32_t pNPattern, bool pDenseFirst = false);

  // whether or not the given layer is recurrent (for hybrid models)
  bool mIsRecr(uint32_t pIl) const;

  uint32_t mNHead(uint32_t pIl = 0) const;

  uint32_t mNHeadKv(uint32_t pIl = 0) const;

  uint32_t mNFf(uint32_t pIl = 0) const;

  uint32_t mNGqa(uint32_t pIl = 0) const;

  uint32_t mNRot(uint32_t pIl = 0) const;

  // dimension of main + auxiliary input embeddings
  uint32_t mNEmbdInp() const;

  // dimension of the encoder input embeddings
  uint32_t mNEmbdInpEnc() const;

  // dimension of output embeddings
  uint32_t mNEmbdOut() const;

  // dimension of key/value embeddings for each head (per layer)
  uint32_t mNEmbdHeadK(uint32_t pIl = 0) const;
  uint32_t mNEmbdHeadV(uint32_t pIl = 0) const;

  // dimension of key embeddings across all k-v heads
  uint32_t mNEmbdKGqa(uint32_t pIl = 0) const;

  // dimension of value embeddings across all k-v heads
  uint32_t mNEmbdVGqa(uint32_t pIl = 0) const;

  // true if any layer has a different n_embd_k_gqa/n_embd_v_gqa
  bool mIsNEmbdKGqaVariable() const;
  bool mIsNEmbdVGqaVariable() const;

  // return the maximum n_embd_k_gqa/n_embd_v_gqa across all layers
  uint32_t mNEmbdKGqaMax() const;
  uint32_t mNEmbdVGqaMax() const;

  // dimension of the rolling state embeddings
  // corresponds to Mamba's conv_states size or RWKV's token_shift states size
  uint32_t mNEmbdR() const;

  // dimension of the recurrent state embeddings
  uint32_t mNEmbdS() const;

  uint32_t mNPosPerEmbd() const;

  // note: currently only support if either all or none of the layers are MLA
  bool mIsMla() const;

  uint32_t mNEmbdHeadKMla() const;
  uint32_t mNEmbdHeadVMla() const;

  bool mHasKv(uint32_t pIl) const;

  // number of effective layers (excludes nextn layers)
  uint32_t mNLayer() const;

  // note that this function uses different SWA parameters from those in the hparams
  // note: inlined on purpose for performance reasons
  // TODO: think of a better place for this function
  // TODO: pack the SWA params in a struct?
  static bool mIsMaskedSwa(uint32_t pNSwa, mgs_swa_type pSwaType, mgs_pos p0, mgs_pos p1) {
    assert(p0 >= 0 && p1 >= 0);

    switch (pSwaType) {
    case cMgsSwaTypeNone: {
    } break;
    case cMgsSwaTypeStandard: {
      if (p1 - p0 >= (int32_t)pNSwa) {
        return true;
      }
    } break;
    case cMgsSwaTypeChunked: {
      const mgs_pos cPosChunkStart = (p1 / pNSwa) * pNSwa;

      if (p0 < cPosChunkStart) {
        return true;
      }
    } break;
    case cMgsSwaTypeSymmetric: {
      const int32_t cHalfNSwa = (int32_t)pNSwa / 2;
      const int32_t cPosDiff = p1 - p0;

      // Mask if outside the symmetric window
      if (cPosDiff < -cHalfNSwa || cPosDiff > cHalfNSwa) {
        return true;
      }
    } break;
    }

    return false;
  }

  bool mUseMrope() const;
};

static_assert(std::is_trivially_copyable<mgs_hparams>::value, "mgs_hparams must be trivially copyable");
