#include "speculative.h"

#include "common.h"
#include "ggml.h"
#include "mgs.h"
#include "log.h"
#include "ngram-cache.h"
#include "ngram-map.h"
#include "ngram-mod.h"
#include "sampling.h"

#include "../src/mgs-ext.h" // staging API: mgs_set_embeddings_nextn / mgs_get_embeddings_nextn_ith (used by MTP)

#include <algorithm>
#include <cassert>
#include <cstring>
#include <iomanip>
#include <map>
#include <cinttypes>

#define SPC_DBG(fmt, ...) LOG_DBG("spec %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define SPC_TRC(fmt, ...) LOG_TRC("spec %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define SPC_INF(fmt, ...) LOG_INF("spec %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define SPC_WRN(fmt, ...) LOG_WRN("spec %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define SPC_ERR(fmt, ...) LOG_ERR("spec %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define SPC_CNT(fmt, ...) LOG_CNT("" fmt, __VA_ARGS__)

#define SPEC_VOCAB_MAX_SIZE_DIFFERENCE 128
#define SPEC_VOCAB_CHECK_START_TOKEN_ID 5

const std::map<std::string, common_speculative_type> dCommonSpeculativeTypeFromNameMap = {
    {"none", cCommonSpeculativeTypeNone},
    {"draft-simple", cCommonSpeculativeTypeDraftSimple},
    {"draft-eagle3", cCommonSpeculativeTypeDraftEagle3},
    {"draft-mtp", cCommonSpeculativeTypeDraftMtp},
    {"draft-dflash", cCommonSpeculativeTypeDraftDflash},
    {"draft-dspark", cCommonSpeculativeTypeDraftDspark},
    {"ngram-simple", cCommonSpeculativeTypeNgramSimple},
    {"ngram-map-k", cCommonSpeculativeTypeNgramMapK},
    {"ngram-map-k4v", cCommonSpeculativeTypeNgramMapK4v},
    {"ngram-mod", cCommonSpeculativeTypeNgramMod},
    {"ngram-cache", cCommonSpeculativeTypeNgramCache}};

static std::string fCommonSpeculativeGetDevicesStr(const std::vector<ggml_backend_dev_t> &pDevices) {
  std::string vResult;
  for (size_t vI = 0; vI < pDevices.size(); vI++) {
    if (pDevices[vI] == nullptr) {
      continue;
    }
    if (!vResult.empty())
      vResult += ", ";
    vResult += ggml_backend_dev_name(pDevices[vI]);
  }
  return vResult.empty() ? "default" : vResult;
}

struct common_speculative_config {
  common_speculative_type vType;
  common_params_speculative vParams;

  common_speculative_config(common_speculative_type pT,
                            const common_params_speculative &pP = common_params_speculative{})
      : vType(pT), vParams(pP) {}
};

static bool fCommonSpeculativeAreCompatible(const mgs_model *pModelTgt, const mgs_model *pModelDft) {
  const mgs_vocab *vVocabTgt = mgs_model_get_vocab(pModelTgt);
  const mgs_vocab *vVocabDft = mgs_model_get_vocab(pModelDft);

  const auto cVocabTypeTgt = mgs_vocab_type(vVocabTgt);
  SPC_DBG("vocab_type tgt: %d\n", cVocabTypeTgt);

  const auto cVocabTypeDft = mgs_vocab_type(vVocabDft);
  SPC_DBG("vocab_type dft: %d\n", cVocabTypeDft);

  if (cVocabTypeTgt != cVocabTypeDft) {
    SPC_WRN("draft model vocab type must match target model to use speculation but "
            "vocab_type_dft = %d while vocab_type_tgt = %d\n",
            cVocabTypeDft, cVocabTypeTgt);
    return false;
  }

  if (mgs_vocab_get_add_bos(vVocabTgt) != mgs_vocab_get_add_bos(vVocabDft) ||
      (mgs_vocab_get_add_bos(vVocabTgt) && mgs_vocab_bos(vVocabTgt) != mgs_vocab_bos(vVocabDft))) {
    SPC_WRN("draft model bos tokens must match target model to use speculation. add: %d - %d, id: %d - %d)\n",
            mgs_vocab_get_add_bos(vVocabTgt), mgs_vocab_get_add_bos(vVocabDft), mgs_vocab_bos(vVocabTgt),
            mgs_vocab_bos(vVocabDft));
    return false;
  }

  if (mgs_vocab_get_add_eos(vVocabTgt) != mgs_vocab_get_add_eos(vVocabDft) ||
      (mgs_vocab_get_add_eos(vVocabTgt) && mgs_vocab_eos(vVocabTgt) != mgs_vocab_eos(vVocabDft))) {
    SPC_WRN("draft model eos tokens must match target model to use speculation. add: %d - %d, id: %d - %d)\n",
            mgs_vocab_get_add_eos(vVocabTgt), mgs_vocab_get_add_eos(vVocabDft), mgs_vocab_eos(vVocabTgt),
            mgs_vocab_eos(vVocabDft));
    return false;
  }

  {
    const int cNVocabTgt = mgs_vocab_n_tokens(vVocabTgt);
    const int cNVocabDft = mgs_vocab_n_tokens(vVocabDft);
    const int cVocabDiff = cNVocabTgt > cNVocabDft ? cNVocabTgt - cNVocabDft : cNVocabDft - cNVocabTgt;

    if (cVocabDiff > SPEC_VOCAB_MAX_SIZE_DIFFERENCE) {
      SPC_DBG("draft model vocab must closely match target model to use speculation but "
              "target vocab size %d does not match draft vocab size %d - difference %d, max allowed %d\n",
              cNVocabTgt, mgs_vocab_n_tokens(vVocabDft), cVocabDiff, SPEC_VOCAB_MAX_SIZE_DIFFERENCE);
      return false;
    }

    for (int vI = SPEC_VOCAB_CHECK_START_TOKEN_ID; vI < std::min(cNVocabTgt, cNVocabDft); ++vI) {
      const char *vTokenTextTgt = mgs_vocab_get_text(vVocabTgt, vI);
      const char *vTokenTextDft = mgs_vocab_get_text(vVocabDft, vI);

      if (std::strcmp(vTokenTextTgt, vTokenTextDft) != 0) {
        SPC_DBG("draft model vocab must match target model to use speculation but "
                "token %d content differs - target '%s', draft '%s'\n",
                vI, fCommonTokenToPiece(vVocabTgt, vI).c_str(), fCommonTokenToPiece(vVocabDft, vI).c_str());
        return false;
      }
    }
  }

  return true;
}

using common_speculative_draft_params_vec = std::vector<common_speculative_draft_params>;

// state of an implementation of speculative decoding
//
// each implementation has a unique type and a state that is implementation-specific
// in a subclass of common_speculative_impl
struct common_speculative_impl {
  const common_speculative_type cType;

  uint32_t vNSeq;

  size_t vNCallBegin = 0;  // number of times this implementation was called for refresh.
  size_t vNCallDraft = 0;  // number of times this implementation was called for generation.
  size_t vNCallAccept = 0; // number of times this implementation was called for accumulation.

  size_t vNGenDrafts = 0; // number of times a draft or part was generated by this implementation.
  size_t vNAccDrafts = 0; // number of times a draft or part was accepted by the target model.
  size_t vNGenTokens = 0; // number of tokens generated by this implementation.
  size_t vNAccTokens = 0; // number of tokens accepted by the target model.

  std::vector<size_t> lNAccTokensPerPos; // number of tokens accepted per draft position.

  // TODO: track performance of most recent calls
  const bool cGenPerf = true; // whether to generate performance stats.

  int64_t vTBeginUs = 0;  // total time spent in refresh of this implementation in microseconds.
  int64_t vTDraftUs = 0;  // total time spent in generating drafts in this implementation in microseconds.
  int64_t vTAcceptUs = 0; // total time spent in accumulation of this implementation in microseconds.

  common_speculative_impl(common_speculative_type pType, uint32_t pNSeq) : cType(pType), vNSeq(pNSeq) {}

  virtual ~common_speculative_impl() = default;

  virtual void begin(mgs_seq_id pSeqId, const mgs_tokens &pPrompt) = 0;

  virtual bool mProcess(const mgs_batch &pBatch) = 0;

  virtual void mDraft(common_speculative_draft_params_vec &pDparams) = 0;

  virtual void mAccept(mgs_seq_id pSeqId, uint16_t pNAccepted, bool pIsOther) = 0;

  // (optional) serialize/restore per-seq internal state (e.g. eagle3's deferred boundary).
  virtual bool mGetState(mgs_seq_id /*seq_id*/, std::vector<uint8_t> & /*data*/) const { return false; }
  virtual void mSetState(mgs_seq_id /*seq_id*/, const std::vector<uint8_t> & /*data*/) {}

  // true if this implementation requires the target context to extract post-norm embeddings
  virtual bool mNeedEmbd() const = 0;

  // true if this implementation requires the target context to extract pre-norm embeddings
  virtual bool mNeedEmbdNextn() const { return false; }
};

struct common_speculative_impl_draft_simple : public common_speculative_impl {
  common_params_speculative_draft vParams;

  mgs_batch vBatch;

  std::vector<common_sampler_ptr> lSmpls;

  common_speculative_impl_draft_simple(const common_params_speculative &pParams, uint32_t pNSeq)
      : common_speculative_impl(cCommonSpeculativeTypeDraftSimple, pNSeq), vParams(pParams.vDraft) {
    auto *vCtxDft = this->vParams.vCtxDft;
    auto *vCtxTgt = this->vParams.vCtxTgt;

    SPC_TRC("%s", "adding speculative implementation 'draft-simple'\n");
    SPC_TRC("- n_max=%d, n_min=%d, p_min=%f\n", this->vParams.vNMax, this->vParams.vNMin, this->vParams.vPMin);
    SPC_TRC("- gpu_layers=%d, cache_k=%s, cache_v=%s, ctx_tgt=%s, ctx_dft=%s, devices=[%s]\n",
            this->vParams.vNGpuLayers, ggml_type_name(this->vParams.vCacheTypeK),
            ggml_type_name(this->vParams.vCacheTypeV), vCtxTgt ? "yes" : "no", vCtxDft ? "yes" : "no",
            fCommonSpeculativeGetDevicesStr(this->vParams.lDevices).c_str());

    vBatch = mgs_batch_init(mgs_n_batch(vCtxDft), 0, 1);

    // TODO: optimize or pass from outside?
    // {
    //     common_params_sampling vParams;
    //     vParams.vNoPerf = false;
    //
    //     vParams.vTopK = 40;
    //     vParams.vTopP = 0.9;
    //
    //     vParams.lSamplers = {
    //         cCommonSamplerTypeTopK,
    //         cCommonSamplerTypeTopP,
    //         cCommonSamplerTypeInfill,
    //     };
    //
    //     result->smpl = fCommonSamplerInit(mgs_get_model(vCtxDft), vParams);
    // }

    lSmpls.resize(pNSeq);
    for (auto &vSmpl : lSmpls) {
      common_params_sampling vParams;
      vParams.vNoPerf = false;
      vParams.vTopK = 10;
      vParams.lSamplers = {
          cCommonSamplerTypeTopK,
      };

      vSmpl.reset(fCommonSamplerInit(mgs_get_model(vCtxDft), vParams));
    }

    const bool cVocabCmpt = fCommonSpeculativeAreCompatible(mgs_get_model(vCtxTgt), mgs_get_model(vCtxDft));
    SPC_DBG("vocab_cmpt = %d\n", cVocabCmpt);

    if (!cVocabCmpt) {
      SPC_ERR("%s", "the target and draft vocabs are not compatible\n");

      throw std::runtime_error("draft model vocab type must match target model to use speculation");
    }

    if (pNSeq != mgs_n_seq_max(vCtxDft)) {
      SPC_ERR("n_seq mismatch: %d != %d\n", pNSeq, mgs_n_seq_max(vCtxDft));

      throw std::runtime_error("the draft model number of sequences is incompatible with the speculative n_seq");
    }
  }

  ~common_speculative_impl_draft_simple() override { mgs_batch_free(vBatch); }

  void begin(mgs_seq_id /*seq_id*/, const mgs_tokens & /*prompt*/) override {
    // noop
  }

  bool mProcess(const mgs_batch &pBatch) override {
    auto *vCtxDft = vParams.vCtxDft;

    mgs_batch vBatchDft = pBatch;
    vBatchDft.logits = nullptr;

    const int cRet = mgs_decode(vCtxDft, vBatchDft);

    if (cRet != 0) {
      SPC_ERR("failed to decode draft batch, ret = %d\n", cRet);

      return false;
    }

    return true;
  }

  void mDraft(common_speculative_draft_params_vec &pDparams) override {
    auto &vCtxDft = vParams.vCtxDft;

    fCommonBatchClear(vBatch);

    // keep track of which sequences are still drafting
    int vNDrafting = 0;
    std::vector<bool> lDrafting(vNSeq);

    for (mgs_seq_id vSeqId = 0; vSeqId < (mgs_seq_id)vNSeq; ++vSeqId) {
      auto &vDp = pDparams[vSeqId];

      if (!vDp.vDrafting) {
        continue;
      }

      vNDrafting++;
      lDrafting[vSeqId] = true;
      fCommonSamplerReset(lSmpls[vSeqId].get());

      fCommonBatchAdd(vBatch, vDp.vIdLast, vDp.vNPast, {vSeqId}, true);
    }

    int vRet = mgs_decode(vCtxDft, vBatch);
    if (vRet != 0) {
      SPC_ERR("mgs_decode returned %d\n", vRet);
      return;
    }

    int vI = 0;

    while (vNDrafting > 0) {
      int vIBatch = 0;

      fCommonBatchClear(vBatch);

      for (mgs_seq_id vSeqId = 0; vSeqId < (mgs_seq_id)vNSeq; ++vSeqId) {
        if (!lDrafting[vSeqId]) {
          continue;
        }

        auto *vSmpl = lSmpls[vSeqId].get();

        fCommonSamplerSample(vSmpl, vCtxDft, vIBatch, true);
        ++vIBatch;

        const auto *vCurP = fCommonSamplerGetCandidates(vSmpl, true);

        for (int vK = 0; vK < std::min(3, (int)vCurP->size); ++vK) {
          SPC_DBG(" - seq_id %d, draft candidate %3d, pos %3d: %6d (%8.3f) '%s'\n", vSeqId, vK, vI, vCurP->data[vK].id,
                  vCurP->data[vK].p, fCommonTokenToPiece(vCtxDft, vCurP->data[vK].id).c_str());
        }

        // add drafted token for each sequence
        const mgs_token cId = vCurP->data[0].id;

        // only collect very high-confidence draft tokens
        if (vCurP->data[0].p < vParams.vPMin) {
          lDrafting[vSeqId] = false;
          vNDrafting--;

          continue;
        }

        fCommonSamplerAccept(vSmpl, cId, true);

        auto &vDp = pDparams.at(vSeqId);
        auto &vResult = *vDp.vResult;

        vResult.push_back(cId);

        if ((vParams.vNMax <= (int)vResult.size()) || (vDp.vNMax > 0 && vDp.vNMax <= (int)vResult.size())) {
          lDrafting[vSeqId] = false;
          vNDrafting--;
          continue;
        }

        fCommonBatchAdd(vBatch, cId, vDp.vNPast + vI + 1, {vSeqId}, true);
      }

      if (vBatch.n_tokens == 0) {
        break;
      }

      // evaluate the drafted tokens on the draft model
      vRet = mgs_decode(vCtxDft, vBatch);
      if (vRet != 0) {
        SPC_ERR("mgs_decode[%d] returned %d\n", vI, vRet);
        break;
      }

      ++vI;
    }

    for (auto &vDp : pDparams) {
      if (!vDp.vDrafting) {
        continue;
      }

      if (vDp.vResult->size() < (size_t)vParams.vNMin) {
        vDp.vResult->clear();
      }
    }
  }

  void mAccept(mgs_seq_id /*seq_id*/, uint16_t /*n_accepted*/, bool /*is_other*/) override {
    // noop
  }

  bool mNeedEmbd() const override { return false; }
};

// EAGLE3 speculative decoding state
//
// Input of draft decoder: (This is different compared to MTP)
//   At "pos P", the decoder takes input pair (t_{P+1}, g_P), with RoPE at P.
//     - t_{P+1} = token at sequence pos P+1 (the *next* token after P)
//     - g_P     = encoder output = projection of target's extracted hidden states at P
//
// Deferred boundary (MTP doesn't have this issue):
//   Within a single mProcess() call with n_tokens, we can only write decoder KV for
//   training pos 0..n_tokens-2. The last training pos (n_tokens-1) needs t_{n_tokens}
//   which lies *outside* this batch — it is the token target will sample next or the first token from next ubatch.
//   So the last training pos of each mProcess() call is *deferred* to whichever next call has
//   the missing token in hand:
//     - multi-ubatch prefill: the next mProcess()'s first token completes the pair
//                              (handled by the per-seq "cross-ubatch bridge")
//     - single-ubatch prefill / after verify: mDraft()'s seed step uses "vDp.vIdLast"
//                              (target's freshest sample) to complete the pair
//
// Per-seq carry-over state:
//   lPendingGLast     [n_embd_dec]  ┐  the deferred boundary's (g, pos). Set by
//   lPendingPosLast   mgs_pos     ┘  mProcess() at end of ubatch (= last row);
//                                       rebased by mAccept() to first-non-accepted pos.
//   lVerifyG          [N × n_embd_dec] snapshot of mProcess()'s encoder output;
//   lVerifyPosFirst   mgs_pos         consumed by mAccept() to recover the right
//   lVerifyGRows      int32_t           lPendingGLast row for any n_accepted value.
//
// Performance is overall good but there is waste in verify cycle:
//   mProcess() runs encoder + decoder on the *full* verify batch including rows for
//   rejected drafts. The KV at those positions is then dropped.
//
// TODO: Not sure if we need optimization for this waste?
// If so we may need hybrid stash:
//      in verify mode, have mProcess() only stash features and let mDraft() seed run
//      encoder+decoder on n_accepted+1 rows).
struct common_speculative_impl_draft_eagle3 : public common_speculative_impl {
  common_params_speculative_draft vParams;
  mgs_batch vBatch;

  std::vector<common_sampler_ptr> lSmpls;

  // backend sampler chain per seq, attached to ctx_dft
  std::vector<mgs_sampler *> lBackendChains;

  int32_t vNEmbdDec = 0;  // draft hidden size
  int32_t vNEmbdEnc = 0;  // target_layer_ids_n * target_hidden_size
  int32_t vNEmbdTgt = 0;  // target model hidden size
  int32_t vNLayerTgt = 0; // target model layer count

  const int32_t *vTargetLayerIds = nullptr; // model_dft's extract layer indices
  uint32_t vTargetLayerIdsN = 0;

  // [per-seq] deferred boundary state
  std::vector<std::vector<float>> lPendingGLast;
  std::vector<mgs_pos> lPendingPosLast;

  // [per-seq] snapshot of the most recent mProcess()'s encoder output
  std::vector<std::vector<float>> lVerifyG; // [n_seq][n_rows * n_embd_dec]
  std::vector<mgs_pos> lVerifyPosFirst;     // [n_seq] — pos of lVerifyG[seq][0]
  std::vector<int32_t> lVerifyGRows;        // [n_seq] — number of rows

  // scratch buffer for concatenated target features [n_tokens, n_embd_enc]
  std::vector<float> lFeaturesBuf;
  std::vector<float> lGEmbdBuf;

  common_speculative_impl_draft_eagle3(const common_params_speculative &pParams, uint32_t pNSeq)
      : common_speculative_impl(cCommonSpeculativeTypeDraftEagle3, pNSeq), vParams(pParams.vDraft) {
    SPC_TRC("%s", "adding speculative implementation 'draft-eagle3'\n");
    SPC_TRC("- n_max=%d, n_min=%d, p_min=%f, backend_sampling=%d\n", pParams.vDraft.vNMax, pParams.vDraft.vNMin,
            pParams.vDraft.vPMin, (int)pParams.vDraft.vBackendSampling);

    auto *vCtxTgt = this->vParams.vCtxTgt;
    auto *vCtxDft = this->vParams.vCtxDft;
    GGML_ASSERT(vCtxTgt && vCtxDft && "EAGLE3 requires ctx_tgt and ctx_dft to be set");

    const mgs_model *vModelDft = mgs_get_model(vCtxDft);
    const mgs_model *vModelTgt = mgs_get_model(vCtxTgt);

    vTargetLayerIds = mgs_model_target_layer_ids(vModelDft);
    vTargetLayerIdsN = mgs_model_target_layer_ids_n(vModelDft);
    if (vTargetLayerIdsN != 3) {
      throw std::runtime_error("draft model is not eagle3 (expected 3 extract layers, got " +
                               std::to_string(vTargetLayerIdsN) + ")");
    }

    vNEmbdTgt = mgs_model_n_embd(vModelTgt);
    vNEmbdDec = mgs_model_n_embd(vModelDft);
    vNEmbdEnc = (int32_t)vTargetLayerIdsN * vNEmbdTgt;
    vNLayerTgt = mgs_model_n_layer(vModelTgt);

    const int32_t cNB = (int32_t)mgs_n_batch(vCtxDft);
    vBatch = mgs_batch_init(/*n_tokens=*/cNB, /*embd=*/vNEmbdDec, /*n_seq_max=*/1);
    // mgs_batch_init allocates only one of token/embd; eagle3 decoder needs both.
    // TODO: fix, how to call without malloc
    vBatch.token = (mgs_token *)malloc(sizeof(mgs_token) * cNB);

    lSmpls.resize(pNSeq);
    for (auto &vS : lSmpls) {
      common_params_sampling vSparams;
      vSparams.vNoPerf = false;
      vSparams.vTopK = 10;
      vSparams.lSamplers = {cCommonSamplerTypeTopK};
      vS.reset(fCommonSamplerInit(mgs_get_model(vCtxDft), vSparams));
    }

    // offload draft sampling to the backend
    lBackendChains.assign(pNSeq, nullptr);
    if (this->vParams.vBackendSampling) {
      for (mgs_seq_id vSeqId = 0; vSeqId < (mgs_seq_id)pNSeq; ++vSeqId) {
        mgs_sampler *vChain = mgs_sampler_chain_init(mgs_sampler_chain_default_params());
        mgs_sampler_chain_add(vChain, mgs_sampler_init_top_k(10));

        if (!mgs_set_sampler(vCtxDft, vSeqId, vChain)) {
          SPC_WRN("backend offload failed for seq_id=%d; using CPU sampler\n", (int)vSeqId);
          mgs_sampler_free(vChain);
          vChain = nullptr;
        }
        lBackendChains[vSeqId] = vChain;
      }
    }

    // turn on extraction of the target layers' hidden states
    for (uint32_t vK = 0; vK < vTargetLayerIdsN; ++vK) {
      if (vTargetLayerIds[vK] < vNLayerTgt) {
        mgs_set_embeddings_layer_inp(vCtxTgt, (uint32_t)vTargetLayerIds[vK], true);
      } else if (vTargetLayerIds[vK] == vNLayerTgt) {
        mgs_set_embeddings_nextn(vCtxTgt, true, /*masked*/ false);
      } else {
        GGML_ABORT("EAGLE3: target layer id %d exceeds target n_layer %d", vTargetLayerIds[vK], vNLayerTgt);
      }
    }

    // turn on extraction of the draft model's pre-norm hidden state
    // (used both for the encoder output g_embd and the decoder pre-norm output).
    mgs_set_embeddings_nextn(vCtxDft, true, /*masked*/ true);

    lPendingGLast.assign(pNSeq, std::vector<float>(vNEmbdDec, 0.0f));
    lPendingPosLast.assign(pNSeq, -1);

    lVerifyG.assign(pNSeq, std::vector<float>());
    lVerifyPosFirst.assign(pNSeq, -1);
    lVerifyGRows.assign(pNSeq, 0);
  }

  ~common_speculative_impl_draft_eagle3() override {
    auto *vCtxDft = this->vParams.vCtxDft;
    for (mgs_seq_id vSeqId = 0; vSeqId < (mgs_seq_id)lBackendChains.size(); ++vSeqId) {
      if (lBackendChains[vSeqId] == nullptr) {
        continue;
      }
      if (vCtxDft) {
        mgs_set_sampler(vCtxDft, vSeqId, nullptr);
      }
      mgs_sampler_free(lBackendChains[vSeqId]);
    }
    lBackendChains.clear();

    if (vBatch.token != nullptr) {
      free(vBatch.token);
      vBatch.token = nullptr;
    }
    mgs_batch_free(vBatch);
  }

  void begin(mgs_seq_id pSeqId, const mgs_tokens &pPrompt) override {
    const int32_t cN = (int32_t)pPrompt.size();
    if (cN <= 0) {
      return;
    }
    // expected state after prefill: ctx_dft has pos 0..N-2 (last position is deferred to
    // mDraft()'s seed step). Warn only if more than one position is missing.
    auto *vCtxDft = this->vParams.vCtxDft;
    const mgs_pos cPosMax = mgs_memory_seq_pos_max(mgs_get_memory(vCtxDft), pSeqId);
    if (cPosMax < cN - 2) {
      SPC_WRN("ctx_dft pos_max=%d < N-2=%d — mProcess() did not run on every prefill ubatch. "
              "Drafts may degrade.\n",
              (int)cPosMax, cN - 2);
    }
  }

  bool mProcess(const mgs_batch &pBatchIn) override {
    if (pBatchIn.n_tokens <= 0) {
      return true;
    }

    if (pBatchIn.token == nullptr || pBatchIn.embd != nullptr) {
      return true;
    }

    const int32_t cNTokens = pBatchIn.n_tokens;

    // i_batch_beg[seq] / i_batch_end[seq]: inclusive batch indices of this seq's
    // first/last token in batch_in. Assumes per-seq tokens are contiguous within
    // the ubatch (server's default ordering).
    std::vector<int32_t> lIBatchBeg(vNSeq, -1);
    std::vector<int32_t> lIBatchEnd(vNSeq, -1);
    for (int vK = 0; vK < cNTokens; ++vK) {
      GGML_ASSERT(pBatchIn.n_seq_id[vK] == 1);
      const mgs_seq_id cSeqId = pBatchIn.seq_id[vK][0];
      if (cSeqId < 0 || cSeqId >= (mgs_seq_id)vNSeq) {
        continue;
      }
      lIBatchEnd[cSeqId] = vK;
      if (lIBatchBeg[cSeqId] < 0) {
        lIBatchBeg[cSeqId] = vK;
      }
    }

    auto *vCtxTgt = this->vParams.vCtxTgt;
    auto *vCtxDft = this->vParams.vCtxDft;

    // Interleave each extract_layer's hidden state into a contiguous buffer of
    // shape [n_tokens, target_layer_ids_n * n_embd_tgt]. Then run EAGLE3 encoder
    // to get one g_embd row per token.
    lFeaturesBuf.resize((size_t)cNTokens * vNEmbdEnc, 0.0f);

    for (uint32_t vK = 0; vK < vTargetLayerIdsN; ++vK) {
      const float *vLayer = vTargetLayerIds[vK] < vNLayerTgt
                                ? mgs_get_embeddings_layer_inp(vCtxTgt, (uint32_t)vTargetLayerIds[vK])
                                : mgs_get_embeddings_nextn(vCtxTgt);
      if (!vLayer) {
        GGML_ABORT("EAGLE3: target layer %d input not extracted.", vTargetLayerIds[vK]);
      }
      for (int32_t vI = 0; vI < cNTokens; ++vI) {
        float *vDst = lFeaturesBuf.data() + (size_t)vI * vNEmbdEnc + vK * (size_t)vNEmbdTgt;
        const float *vSrc = vLayer + (size_t)vI * vNEmbdTgt;
        std::memcpy(vDst, vSrc, (size_t)vNEmbdTgt * sizeof(float));
      }
    }

    lGEmbdBuf.resize((size_t)cNTokens * vNEmbdDec);

    // mgs_encode() requires the full encoder batch to fit in n_ubatch.
    // Allow batch > ubatch: eagle3's per-token encoder can be chunked safely.
    const int32_t cNUbatchDft = (int32_t)mgs_n_ubatch(vCtxDft);
    for (int32_t vI = 0; vI < cNTokens; vI += cNUbatchDft) {
      const int32_t cNChunk = std::min(cNUbatchDft, cNTokens - vI);

      mgs_batch vEncBatch = {
          /*.n_tokens =*/cNChunk,
          /*.token    =*/nullptr,
          /*.embd     =*/lFeaturesBuf.data() + (size_t)vI * vNEmbdEnc,
          /*.pos      =*/nullptr,
          /*.n_seq_id =*/nullptr,
          /*.seq_id   =*/nullptr,
          /*.logits   =*/nullptr,
      };
      const int32_t cRc = mgs_encode(vCtxDft, vEncBatch);
      if (cRc != 0) {
        SPC_ERR("mgs_encode(ctx_dft) failed rc=%d (n_tokens=%d, offset=%d)\n", cRc, (int)cNChunk, (int)vI);
        return false;
      }

      // g_embd has shape [n_chunk, n_embd_dec] in ctx_dft's pre-norm embeddings buffer.
      const float *vGEmbdChunk = mgs_get_embeddings_nextn(vCtxDft);
      GGML_ASSERT(vGEmbdChunk && "EAGLE3 encoder produced no output.");
      std::memcpy(lGEmbdBuf.data() + (size_t)vI * vNEmbdDec, vGEmbdChunk, (size_t)cNChunk * vNEmbdDec * sizeof(float));
    }

    const float *vGEmbd = lGEmbdBuf.data();

    const size_t cRowBytes = (size_t)vNEmbdDec * sizeof(float);

    // EAGLE3 decoder input convention: at memory pos P the input pair is
    // (token[P+1], g_embd[P]). This shifts the token index "left by one" relative to g_embd.
    //
    // Per seq, in order:
    //   (a) cross-ubatch bridge — when applicable, write the previously-deferred
    //       pos using this ubatch's first token + lPendingGLast.
    //   (b) main write loop — for k in [beg, end-1], write (token[k+1], g_embd[k])
    //       at pos[k]. The last training pos (k=end) is left unwritten = new
    //       deferred boundary, completed by the next mProcess() or mDraft() call.
    //   (c) refresh deferred state — stash this ubatch's full g_embd into verify_g,
    //       update lPendingGLast / lPendingPosLast to the last row.
    fCommonBatchClear(vBatch);

    for (mgs_seq_id vSeqId = 0; vSeqId < (mgs_seq_id)vNSeq; ++vSeqId) {
      const int32_t cBeg = lIBatchBeg[vSeqId];
      const int32_t cEnd = lIBatchEnd[vSeqId];
      if (cBeg < 0 || cEnd < 0) {
        continue;
      }

      // cross-ubatch bridge — complete the prior ubatch's deferred boundary.
      // Fires iff all three preconditions hold:
      //   1) lPendingPosLast >= 0
      //   2) lPendingPosLast + 1 == pos[cBeg]
      //   3) lPendingPosLast > cDftPosMax // TODO: is this check needed?
      const mgs_pos cPendingPos = lPendingPosLast[vSeqId];
      if (cPendingPos >= 0 && cPendingPos + 1 == pBatchIn.pos[cBeg]) {
        const mgs_pos cDftPosMax = mgs_memory_seq_pos_max(mgs_get_memory(vCtxDft), vSeqId);
        if (cPendingPos > cDftPosMax) {
          fCommonBatchAdd(vBatch, pBatchIn.token[cBeg], cPendingPos, {vSeqId}, /*logits=*/false);
          std::memcpy(vBatch.embd + (size_t)(vBatch.n_tokens - 1) * vNEmbdDec, lPendingGLast[vSeqId].data(), cRowBytes);
        }
      }

      for (int32_t vK = cBeg; vK < cEnd; ++vK) {
        fCommonBatchAdd(vBatch, pBatchIn.token[vK + 1], pBatchIn.pos[vK], {vSeqId}, /*logits=*/false);
        std::memcpy(vBatch.embd + (size_t)(vBatch.n_tokens - 1) * vNEmbdDec, vGEmbd + (size_t)vK * vNEmbdDec,
                    cRowBytes);
      }

      // refresh deferred state
      const int32_t cNRows = cEnd - cBeg + 1;
      lVerifyPosFirst[vSeqId] = pBatchIn.pos[cBeg];
      lPendingPosLast[vSeqId] = pBatchIn.pos[cEnd];
      lVerifyGRows[vSeqId] = cNRows;
      lVerifyG[vSeqId].resize((size_t)cNRows * vNEmbdDec, 0.0f);
      std::memcpy(lVerifyG[vSeqId].data(), vGEmbd + (size_t)cBeg * vNEmbdDec, cRowBytes * cNRows);
      std::memcpy(lPendingGLast[vSeqId].data(), vGEmbd + (size_t)cEnd * vNEmbdDec, cRowBytes);
    }

    if (vBatch.n_tokens > 0) {
      const int32_t cRc = mgs_decode(vCtxDft, vBatch);
      if (cRc != 0) {
        SPC_ERR("mgs_decode(ctx_dft) failed rc=%d (n_tokens=%d, ubatch_pos[0]=%d)\n", cRc, (int)vBatch.n_tokens,
                (int)pBatchIn.pos[0]);
        return false;
      }
    }

    return true;
  }

  void mDraft(common_speculative_draft_params_vec &pDparams) override {
    auto &vCtxDft = vParams.vCtxDft;

    fCommonBatchClear(vBatch);

    // keep track of which sequences are still drafting
    int vNDrafting = 0;
    std::vector<bool> lDrafting(vNSeq);

    const size_t cRowBytes = (size_t)vNEmbdDec * sizeof(float);

    // Complete the deferred boundary pair (vDp.vIdLast, lPendingGLast) at memory
    // pos lPendingPosLast. vDp.vIdLast is target's freshest sample (= corrected
    // token after verify, or first generated token after prefill), matching the
    // EAGLE3 input convention (token[P+1], g_embd[P]) at pos P.
    for (mgs_seq_id vSeqId = 0; vSeqId < (mgs_seq_id)vNSeq; ++vSeqId) {
      auto &vDp = pDparams[vSeqId];

      if (!vDp.vDrafting) {
        continue;
      }
      if (lPendingPosLast[vSeqId] < 0) {
        continue;
      }

      vNDrafting++;
      lDrafting[vSeqId] = true;
      fCommonSamplerReset(lSmpls[vSeqId].get());

      mgs_memory_seq_rm(mgs_get_memory(vCtxDft), vSeqId, lPendingPosLast[vSeqId], -1);

      fCommonBatchAdd(vBatch, vDp.vIdLast, lPendingPosLast[vSeqId], {vSeqId}, true);
      std::memcpy(vBatch.embd + (size_t)(vBatch.n_tokens - 1) * vNEmbdDec, lPendingGLast[vSeqId].data(), cRowBytes);
    }

    if (vBatch.n_tokens == 0) {
      return;
    }

    int vRet = mgs_decode(vCtxDft, vBatch);
    if (vRet != 0) {
      SPC_ERR("mgs_decode returned %d\n", vRet);
      return;
    }

    int vI = 0;

    while (vNDrafting > 0) {
      int vIBatch = 0;

      fCommonBatchClear(vBatch);

      for (mgs_seq_id vSeqId = 0; vSeqId < (mgs_seq_id)vNSeq; ++vSeqId) {
        if (!lDrafting[vSeqId]) {
          continue;
        }

        auto *vSmpl = lSmpls[vSeqId].get();

        fCommonSamplerSample(vSmpl, vCtxDft, vIBatch, true);
        // pre-norm hidden state of this position becomes g_embd for the next step
        const float *vPrenorm = mgs_get_embeddings_nextn_ith(vCtxDft, vIBatch);
        ++vIBatch;

        const auto *vCurP = fCommonSamplerGetCandidates(vSmpl, true);

        for (int vK = 0; vK < std::min(3, (int)vCurP->size); ++vK) {
          SPC_DBG(" - seq_id %d, draft candidate %3d, pos %3d: %6d (%8.3f) '%s'\n", vSeqId, vK, vI, vCurP->data[vK].id,
                  vCurP->data[vK].p, fCommonTokenToPiece(vCtxDft, vCurP->data[vK].id).c_str());
        }

        const mgs_token cId = vCurP->data[0].id;

        // only collect very high-confidence draft tokens
        // (configurable via --spec-draft-p-min, set to 0.0 to disable early-stop)
        if (vCurP->data[0].p < vParams.vPMin) {
          lDrafting[vSeqId] = false;
          vNDrafting--;

          continue;
        }

        fCommonSamplerAccept(vSmpl, cId, true);

        auto &vDp = pDparams.at(vSeqId);
        auto &vResult = *vDp.vResult;

        vResult.push_back(cId);

        if (vParams.vNMax <= (int)vResult.size()) {
          lDrafting[vSeqId] = false;
          vNDrafting--;
          continue;
        }

        fCommonBatchAdd(vBatch, cId, lPendingPosLast[vSeqId] + (vI + 1), {vSeqId}, true);
        std::memcpy(vBatch.embd + (size_t)(vBatch.n_tokens - 1) * vNEmbdDec, vPrenorm, cRowBytes);
      }

      if (vBatch.n_tokens == 0) {
        break;
      }

      vRet = mgs_decode(vCtxDft, vBatch);
      if (vRet != 0) {
        SPC_ERR("mgs_decode[%d] returned %d\n", vI, vRet);
        break;
      }

      ++vI;
    }

    for (mgs_seq_id vSeqId = 0; vSeqId < (mgs_seq_id)vNSeq; ++vSeqId) {
      auto &vDp = pDparams[vSeqId];
      if (!vDp.vDrafting) {
        continue;
      }

      if (vDp.vResult->size() < (size_t)vParams.vNMin) {
        vDp.vResult->clear();
      }
    }
  }

  void mAccept(mgs_seq_id pSeqId, uint16_t pNAccepted, bool /*is_other*/) override {
    if (pSeqId < 0 || pSeqId >= (mgs_seq_id)vNSeq) {
      return;
    }

    const int32_t cNRows = lVerifyGRows[pSeqId];
    if (cNRows <= 0) {
      return;
    }

    const int32_t cIG = std::min<int32_t>(pNAccepted, cNRows - 1);
    lPendingPosLast[pSeqId] = lVerifyPosFirst[pSeqId] + cIG;
    std::memcpy(lPendingGLast[pSeqId].data(), lVerifyG[pSeqId].data() + (size_t)cIG * vNEmbdDec,
                (size_t)vNEmbdDec * sizeof(float));
  }

  // we only need to stash the deferred boundary's g_embd row for recurrent/hybrid targets:
  // their single-position checkpoints drop it on restore
  bool mNeedBoundaryStash() const {
    const mgs_model *vModelTgt = mgs_get_model(vParams.vCtxTgt);
    return mgs_model_is_recurrent(vModelTgt) || mgs_model_is_hybrid(vModelTgt);
  }

  bool mGetState(mgs_seq_id pSeqId, std::vector<uint8_t> &pData) const override {
    if (!mNeedBoundaryStash()) {
      return false;
    }
    if (pSeqId < 0 || pSeqId >= (mgs_seq_id)vNSeq || lPendingPosLast[pSeqId] < 0) {
      return false;
    }

    const mgs_pos cPos = lPendingPosLast[pSeqId];
    const std::vector<float> &lG = lPendingGLast[pSeqId];

    pData.resize(sizeof(mgs_pos) + lG.size() * sizeof(float));
    std::memcpy(pData.data(), &cPos, sizeof(mgs_pos));
    std::memcpy(pData.data() + sizeof(mgs_pos), lG.data(), lG.size() * sizeof(float));
    return true;
  }

  void mSetState(mgs_seq_id pSeqId, const std::vector<uint8_t> &pData) override {
    if (!mNeedBoundaryStash()) {
      return;
    }
    if (pSeqId < 0 || pSeqId >= (mgs_seq_id)vNSeq) {
      return;
    }
    if (pData.size() != sizeof(mgs_pos) + (size_t)vNEmbdDec * sizeof(float)) {
      return;
    }

    mgs_pos vPos = -1;
    std::memcpy(&vPos, pData.data(), sizeof(mgs_pos));

    lPendingPosLast[pSeqId] = vPos;
    lPendingGLast[pSeqId].resize(vNEmbdDec);
    std::memcpy(lPendingGLast[pSeqId].data(), pData.data() + sizeof(mgs_pos), (size_t)vNEmbdDec * sizeof(float));
  }

  bool mNeedEmbd() const override { return false; }
};

// DFlash: block-diffusion drafting with a draft-side KV cache injection
struct common_speculative_impl_draft_dflash : public common_speculative_impl {
  common_params_speculative_draft vParams;

  mgs_batch vBatch;       // noise tokens
  mgs_batch vBatchInject; // target features for KV cache injection

  std::vector<common_sampler_ptr> lSmpls;

  int32_t vNEmbdDec = 0; // draft hidden size
  int32_t vNEmbdEnc = 0; // target_layer_ids_n * target_hidden_size
  int32_t vNEmbdTgt = 0; // target model hidden size

  int32_t vBlockSize = 0;
  mgs_token vMaskTokenId = 0;

  // draft-dspark: the draft carries a Markov head and uses an anchor-first block layout
  const bool cIsDspark;

  const int32_t *vTargetLayerIds = nullptr; // model_dft's extract layer indices
  uint32_t vTargetLayerIdsN = 0;

  // scratch buffer for concatenated target features [n_tokens, n_embd_enc]
  std::vector<float> lFeaturesBuf;

  common_speculative_impl_draft_dflash(const common_params_speculative &pParams, uint32_t pNSeq,
                                       common_speculative_type pType = cCommonSpeculativeTypeDraftDflash)
      : common_speculative_impl(pType, pNSeq), vParams(pParams.vDraft),
        cIsDspark(pType == cCommonSpeculativeTypeDraftDspark) {
    auto *vCtxTgt = this->vParams.vCtxTgt;
    auto *vCtxDft = this->vParams.vCtxDft;
    GGML_ASSERT(vCtxTgt && vCtxDft && "DFlash requires ctx_tgt and ctx_dft to be set");

    const mgs_model *vModelDft = mgs_get_model(vCtxDft);
    const mgs_model *vModelTgt = mgs_get_model(vCtxTgt);

    vTargetLayerIds = mgs_model_target_layer_ids(vModelDft);
    vTargetLayerIdsN = mgs_model_target_layer_ids_n(vModelDft);
    GGML_ASSERT(vTargetLayerIdsN > 0 && "DFlash model has no target_layer_ids");

    vNEmbdTgt = mgs_model_n_embd(vModelTgt);
    vNEmbdDec = mgs_model_n_embd(vModelDft);
    vNEmbdEnc = (int32_t)vTargetLayerIdsN * vNEmbdTgt;

    // read the trained block size from the dflash.block_size metadata key
    vBlockSize = 16;
    {
      char aBuf[32] = {};
      if (mgs_model_meta_val_str(vModelDft, "dflash.block_size", aBuf, sizeof(aBuf)) >= 0) {
        vBlockSize = std::atoi(aBuf);
      }
    }
    vMaskTokenId = mgs_vocab_mask(mgs_model_get_vocab(vModelDft));

    LOG_INF("%s: adding speculative implementation '%s'\n", __func__, fCommonSpeculativeTypeToStr(pType).c_str());
    LOG_INF("%s: - n_max=%d, n_min=%d, p_min=%.2f\n", __func__, this->vParams.vNMax, this->vParams.vNMin,
            this->vParams.vPMin);
    LOG_INF("%s: - block_size=%d, mask_token_id=%d, n_extract=%u\n", __func__, vBlockSize, vMaskTokenId,
            vTargetLayerIdsN);

    // DFlash input is [id_last, <mask> * (block_size-1)]: in-place denoising yields at most
    // block_size-1 draft tokens, DSpark yield a full block_size draft tokens
    const int32_t cNDraftMax = cIsDspark ? vBlockSize : vBlockSize - 1;
    if (this->vParams.vNMax > cNDraftMax || this->vParams.vNMin > cNDraftMax) {
      LOG_WRN("%s: requested draft size (n_max=%d, n_min=%d) exceeds the trained block size %d -- clamping to %d\n",
              __func__, this->vParams.vNMax, this->vParams.vNMin, vBlockSize, cNDraftMax);
      this->vParams.vNMax = std::min(this->vParams.vNMax, cNDraftMax);
      this->vParams.vNMin = std::min(this->vParams.vNMin, cNDraftMax);
    }

    vBatch = mgs_batch_init(mgs_n_batch(vCtxDft), 0, pNSeq);
    vBatchInject = mgs_batch_init(mgs_n_batch(vCtxDft), vNEmbdDec, pNSeq);

    lSmpls.resize(pNSeq);
    for (auto &vS : lSmpls) {
      common_params_sampling vSparams;
      vSparams.vNoPerf = false;
      vSparams.vTopK = 10;
      vSparams.lSamplers = {cCommonSamplerTypeTopK};
      vS.reset(fCommonSamplerInit(vModelDft, vSparams));
    }

    // turn on extraction of the target layers' input embeddings
    for (uint32_t vK = 0; vK < vTargetLayerIdsN; ++vK) {
      mgs_set_embeddings_layer_inp(vCtxTgt, (uint32_t)vTargetLayerIds[vK], true);
    }

    mgs_set_embeddings_nextn(vCtxDft, true, /*masked*/ true);
    mgs_set_causal_attn(vCtxDft, false); // DFlash needs non-causal attention
  }

  ~common_speculative_impl_draft_dflash() override {
    mgs_batch_free(vBatch);
    mgs_batch_free(vBatchInject);
  }

  void begin(mgs_seq_id pSeqId, const mgs_tokens &pPrompt) override {
    if (pSeqId < 0 || pSeqId >= (mgs_seq_id)vNSeq) {
      return;
    }

    const int32_t cN = (int32_t)pPrompt.size();
    if (cN <= 0) {
      return;
    }

    const mgs_pos cPosMax = mgs_memory_seq_pos_max(mgs_get_memory(vParams.vCtxDft), pSeqId);
    if (cPosMax < cN - 1) {
      LOG_WRN("%s: ctx_dft pos_max=%d < N-1=%d - mProcess() did not run on every prefill ubatch. "
              "Drafts may degrade.\n",
              __func__, (int)cPosMax, cN - 1);
    }
  }

  bool mProcess(const mgs_batch &pBatchIn) override {
    if (pBatchIn.n_tokens <= 0) {
      return true;
    }

    if (pBatchIn.token == nullptr || pBatchIn.embd != nullptr) {
      return true;
    }

    const int32_t cNTokens = pBatchIn.n_tokens;

    // per-seq inclusive batch range (assumes each seq's tokens are contiguous in the batch)
    std::vector<int32_t> lIBatchBeg(vNSeq, -1);
    std::vector<int32_t> lIBatchEnd(vNSeq, -1);
    for (int32_t vK = 0; vK < cNTokens; ++vK) {
      GGML_ASSERT(pBatchIn.n_seq_id[vK] == 1);
      const mgs_seq_id cSeqId = pBatchIn.seq_id[vK][0];
      if (cSeqId < 0 || cSeqId >= (mgs_seq_id)vNSeq) {
        continue;
      }
      lIBatchEnd[cSeqId] = vK;
      if (lIBatchBeg[cSeqId] < 0) {
        lIBatchBeg[cSeqId] = vK;
      }
    }

    auto *vCtxTgt = this->vParams.vCtxTgt;
    auto *vCtxDft = this->vParams.vCtxDft;

    const int32_t cNUbatch = (int32_t)mgs_n_ubatch(vCtxDft);

    for (mgs_seq_id vSeqId = 0; vSeqId < (mgs_seq_id)vNSeq; ++vSeqId) {
      if (lIBatchBeg[vSeqId] < 0) {
        continue;
      }
      const int32_t cNRows = lIBatchEnd[vSeqId] - lIBatchBeg[vSeqId] + 1;

      for (int32_t vOffset = 0; vOffset < cNRows; vOffset += cNUbatch) {
        const int32_t cNChunk = std::min(cNUbatch, cNRows - vOffset);

        // gather this chunk's target features, interleaved by extract layer
        lFeaturesBuf.resize((size_t)cNChunk * vNEmbdEnc);
        for (uint32_t vK = 0; vK < vTargetLayerIdsN; ++vK) {
          const float *vLayer = mgs_get_embeddings_layer_inp(vCtxTgt, (uint32_t)vTargetLayerIds[vK]);
          if (!vLayer) {
            GGML_ABORT("DFlash: target layer %d input not extracted.", vTargetLayerIds[vK]);
          }
          for (int32_t vI = 0; vI < cNChunk; ++vI) {
            float *vDst = lFeaturesBuf.data() + (size_t)vI * vNEmbdEnc + vK * (size_t)vNEmbdTgt;
            const float *vSrc = vLayer + (size_t)(lIBatchBeg[vSeqId] + vOffset + vI) * vNEmbdTgt;
            std::memcpy(vDst, vSrc, (size_t)vNEmbdTgt * sizeof(float));
          }
        }

        // fuse extracted features through DFlash encoder
        mgs_batch vEncBatch = {
            /*.n_tokens =*/cNChunk,
            /*.token    =*/nullptr,
            /*.embd     =*/lFeaturesBuf.data(),
            /*.pos      =*/nullptr,
            /*.n_seq_id =*/nullptr,
            /*.seq_id   =*/nullptr,
            /*.logits   =*/nullptr,
        };

        int32_t vRc = mgs_encode(vCtxDft, vEncBatch);
        if (vRc != 0) {
          LOG_ERR("%s: mgs_encode(ctx_dft) failed rc=%d (n_tokens=%d, offset=%d)\n", __func__, vRc, (int)cNChunk,
                  (int)vOffset);
          return false;
        }

        const float *vInpG = mgs_get_embeddings_nextn(vCtxDft);
        GGML_ASSERT(vInpG && "DFlash encoder produced no output.");

        // inject the DFlash decoder K/V cache at the tokens' target positions
        vBatchInject.n_tokens = cNChunk;
        std::memcpy(vBatchInject.embd, vInpG, (size_t)cNChunk * vNEmbdDec * sizeof(float));

        for (int32_t vI = 0; vI < cNChunk; ++vI) {
          vBatchInject.pos[vI] = pBatchIn.pos[lIBatchBeg[vSeqId] + vOffset + vI];
          vBatchInject.n_seq_id[vI] = 1;
          vBatchInject.seq_id[vI][0] = vSeqId;
          vBatchInject.logits[vI] = false;
        }
        vRc = mgs_decode(vCtxDft, vBatchInject);
        if (vRc != 0) {
          LOG_ERR("%s: mgs_decode(ctx_dft) failed rc=%d (n_tokens=%d, offset=%d)\n", __func__, vRc, (int)cNChunk,
                  (int)vOffset);
          return false;
        }
      }
    }

    return true;
  }

  void mDraft(common_speculative_draft_params_vec &pDparams) override {
    auto &vCtxDft = vParams.vCtxDft;

    fCommonBatchClear(vBatch);

    // build one batch holding every drafting sequence's noise block into a single decode)
    // record where each block starts and its size
    std::vector<int32_t> lIBlockBeg(vNSeq, -1);
    std::vector<int32_t> lNBlock(vNSeq, 0);

    for (mgs_seq_id vSeqId = 0; vSeqId < (mgs_seq_id)vNSeq; ++vSeqId) {
      auto &vDp = pDparams[vSeqId];
      if (!vDp.vDrafting) {
        continue;
      }

      fCommonSamplerReset(lSmpls[vSeqId].get());

      const int32_t cN = (int32_t)vDp.vNPast;

      const int32_t cNDraft = vParams.vNMax;

      const int32_t cNBlockTokens = cNDraft + (cIsDspark ? 0 : 1);
      lIBlockBeg[vSeqId] = vBatch.n_tokens;
      lNBlock[vSeqId] = cNBlockTokens;
      for (int32_t vI = 0; vI < cNBlockTokens; ++vI) {
        fCommonBatchAdd(vBatch, vI == 0 ? vDp.vIdLast : vMaskTokenId, cN + vI, {vSeqId}, true);
      }
    }

    if (vBatch.n_tokens == 0) {
      return;
    }

    // decode all sequence's noise block in a single batch
    int vRet = mgs_decode(vCtxDft, vBatch);
    if (vRet != 0) {
      LOG_WRN("%s: mgs_decode returned %d\n", __func__, vRet);
      return;
    }

    for (mgs_seq_id vSeqId = 0; vSeqId < (mgs_seq_id)vNSeq; ++vSeqId) {
      if (lIBlockBeg[vSeqId] < 0) {
        continue;
      }
      auto &vDp = pDparams[vSeqId];

      const int32_t cBeg = lIBlockBeg[vSeqId];
      const int32_t cNBlockTokens = lNBlock[vSeqId];

      auto *vSmpl = lSmpls[vSeqId].get();

      auto &vResult = *vDp.vResult;

      if (cIsDspark) {
        // DSpark predicts the next token from position 0 and optionally truncates
        // at the first position below the confidence threshold.
        const float *vConf = vParams.vPMin > 0.0f ? mgs_get_embeddings_nextn(vCtxDft) : nullptr;

        for (int32_t vI = 0; vI < cNBlockTokens; ++vI) {
          const int32_t cIdx = cBeg + vI;

          if (vConf && vConf[(size_t)cIdx * vNEmbdDec] < vParams.vPMin) {
            break;
          }

          fCommonSamplerSample(vSmpl, vCtxDft, cIdx, true);

          const auto *vCurP = fCommonSamplerGetCandidates(vSmpl, true);

          for (int vK = 0; vK < std::min(3, (int)vCurP->size); ++vK) {
            LOG_DBG(" - seq_id %d, draft candidate %3d, pos %3d: %6d (%8.3f) '%s'\n", vSeqId, vK, vI,
                    vCurP->data[vK].id, vCurP->data[vK].p, fCommonTokenToPiece(vCtxDft, vCurP->data[vK].id).c_str());
          }

          const mgs_token cId = vCurP->data[0].id;

          fCommonSamplerAccept(vSmpl, cId, true);

          vResult.push_back(cId);
        }
      } else {
        // greedily read the predicted block at this sequence's noise positions 1..cNBlockTokens-1
        for (int32_t vI = 1; vI < cNBlockTokens; ++vI) {
          fCommonSamplerSample(vSmpl, vCtxDft, cBeg + vI, true);

          const auto *vCurP = fCommonSamplerGetCandidates(vSmpl, true);

          for (int vK = 0; vK < std::min(3, (int)vCurP->size); ++vK) {
            LOG_DBG(" - seq_id %d, draft candidate %3d, pos %3d: %6d (%8.3f) '%s'\n", vSeqId, vK, vI - 1,
                    vCurP->data[vK].id, vCurP->data[vK].p, fCommonTokenToPiece(vCtxDft, vCurP->data[vK].id).c_str());
          }

          const mgs_token cId = vCurP->data[0].id;

          if (vCurP->data[0].p < vParams.vPMin) {
            break;
          }

          fCommonSamplerAccept(vSmpl, cId, true);

          vResult.push_back(cId);
        }
      }

      if (vResult.size() < (size_t)vParams.vNMin) {
        vResult.clear();
      }
    }
  }

  void mAccept(mgs_seq_id /*seq_id*/, uint16_t /*n_accepted*/, bool /*is_other*/) override {
    // noop
  }

  bool mNeedEmbd() const override { return false; }
};

struct common_speculative_impl_draft_mtp : public common_speculative_impl {
  common_params_speculative_draft vParams; // reuses the draft-model params slot (ctx_tgt/ctx_dft)

  mgs_batch vBatch;

  std::vector<common_sampler_ptr> lSmpls;

  // backend sampler chain per seq, attached to ctx_dft
  std::vector<mgs_sampler *> lBackendChains;

  int32_t vNEmbd = 0;

  // One MTP draft driver, two modes (set once in the ctor):
  //   chain_heads (step35): n_mtp_layers trained heads, one per draft step.
  //   otherwise (qwen35 / qwen35moe): a single trained MTP head.
  int32_t vNMtpLayers = 1;
  bool vChainHeads = false; // derived in the ctor: n_mtp_layers > 1

  // Per-sequence cross-batch carryover: pair (h_p, x_{p+1}) at MTP pos p+1.
  // The last h-row of one mProcess() call needs the first token of the NEXT
  // call to pair with, so it's stashed here until that next call fires.
  std::vector<std::vector<float>> lPendingH; // [n_seq][n_embd]

  std::vector<int32_t> lIBatchBeg;
  std::vector<int32_t> lIBatchEnd;

  // Hidden rows from the most recent target verification batch, grouped by seq.
  // Row 0 corresponds to the sampled token, row N to the Nth accepted draft token.
  std::vector<std::vector<float>> lVerifyH;
  std::vector<int32_t> lVerifyHRows;

  std::vector<int> lILast;
  std::vector<std::vector<float>> lChainH;

  common_speculative_impl_draft_mtp(const common_params_speculative &pParams, uint32_t pNSeq)
      : common_speculative_impl(cCommonSpeculativeTypeDraftMtp, pNSeq), vParams(pParams.vDraft) {
    auto *vCtxTgt = this->vParams.vCtxTgt;
    auto *vCtxDft = this->vParams.vCtxDft;
    GGML_ASSERT(vCtxTgt && vCtxDft && "MTP requires ctx_tgt and ctx_dft to be set");

    vNEmbd = mgs_model_n_embd_out(mgs_get_model(vCtxDft));
    GGML_ASSERT(vNEmbd == mgs_model_n_embd_out(mgs_get_model(vCtxTgt)) &&
                "MTP input row width must match the target h_nextn width");
    vNMtpLayers = std::max(1, (int)mgs_model_n_layer_nextn(mgs_get_model(vCtxDft)));

    SPC_TRC("%s", "adding speculative implementation 'draft-mtp'\n");
    SPC_TRC("- n_max=%d, n_min=%d, p_min=%.2f, n_embd=%d, backend_sampling=%d\n", this->vParams.vNMax,
            this->vParams.vNMin, this->vParams.vPMin, vNEmbd, (int)this->vParams.vBackendSampling);
    SPC_TRC("- gpu_layers=%d, cache_k=%s, cache_v=%s, ctx_tgt=%s, ctx_dft=%s, devices=[%s]\n",
            this->vParams.vNGpuLayers, ggml_type_name(this->vParams.vCacheTypeK),
            ggml_type_name(this->vParams.vCacheTypeV), vCtxTgt ? "yes" : "no", vCtxDft ? "yes" : "no",
            fCommonSpeculativeGetDevicesStr(this->vParams.lDevices).c_str());

    const int32_t cNB = (int32_t)mgs_n_batch(vCtxDft);
    vBatch = mgs_batch_init(/*n_tokens=*/cNB, /*embd=*/vNEmbd, /*n_seq_max=*/1);
    // mgs_batch_init allocates only one of token/embd; MTP needs both.
    // TODO: fix, how to call without malloc
    vBatch.token = (mgs_token *)malloc(sizeof(mgs_token) * cNB);

    lSmpls.resize(pNSeq);
    for (auto &vS : lSmpls) {
      common_params_sampling vSparams;
      vSparams.vNoPerf = false;
      vSparams.vTopK = 10;
      vSparams.lSamplers = {cCommonSamplerTypeTopK};
      vS.reset(fCommonSamplerInit(mgs_get_model(vCtxDft), vSparams));
    }

    // offload draft sampling to the backend
    lBackendChains.assign(pNSeq, nullptr);
    if (this->vParams.vBackendSampling) {
      for (mgs_seq_id vSeqId = 0; vSeqId < (mgs_seq_id)pNSeq; ++vSeqId) {
        mgs_sampler *vChain = mgs_sampler_chain_init(mgs_sampler_chain_default_params());
        mgs_sampler_chain_add(vChain, mgs_sampler_init_top_k(10));

        if (!mgs_set_sampler(vCtxDft, vSeqId, vChain)) {
          SPC_WRN("backend offload failed for seq_id=%d; using CPU sampler\n", (int)vSeqId);
          mgs_sampler_free(vChain);
          vChain = nullptr;
        }
        lBackendChains[vSeqId] = vChain;
      }
    }

    mgs_set_embeddings_nextn(vCtxTgt, true, /*masked*/ false);
    mgs_set_embeddings_nextn(vCtxDft, true, /*masked*/ true);

    vChainHeads = vNMtpLayers > 1;

    if (vChainHeads) {
      this->vParams.vNMax = std::min(this->vParams.vNMax, vNMtpLayers);

      lChainH.assign(pNSeq, {});
      for (auto &vC : lChainH) {
        vC.reserve((size_t)(this->vParams.vNMax + 1) * vNEmbd);
      }
    }

    lPendingH.assign(pNSeq, std::vector<float>(vNEmbd, 0.0f));

    lILast.assign(pNSeq, -1);
    lIBatchBeg.assign(pNSeq, -1);
    lIBatchEnd.assign(pNSeq, -1);

    lVerifyH.assign(pNSeq, {});
    lVerifyHRows.assign(pNSeq, 0);
  }

  ~common_speculative_impl_draft_mtp() override {
    auto *vCtxDft = this->vParams.vCtxDft;
    for (mgs_seq_id vSeqId = 0; vSeqId < (mgs_seq_id)lBackendChains.size(); ++vSeqId) {
      if (lBackendChains[vSeqId] == nullptr) {
        continue;
      }
      if (vCtxDft) {
        mgs_set_sampler(vCtxDft, vSeqId, nullptr);
      }
      mgs_sampler_free(lBackendChains[vSeqId]);
    }
    lBackendChains.clear();

    if (vBatch.token != nullptr) {
      free(vBatch.token);
      vBatch.token = nullptr;
    }
    mgs_batch_free(vBatch);
  }

  void begin(mgs_seq_id pSeqId, const mgs_tokens &pPrompt) override {
    const int32_t cN = (int32_t)pPrompt.size();
    if (cN <= 0) {
      return;
    }

    auto *vCtxDft = this->vParams.vCtxDft;
    const mgs_pos cPosMax = mgs_memory_seq_pos_max(mgs_get_memory(vCtxDft), pSeqId);

    if (cPosMax < cN - 1) {
      SPC_WRN("ctx_dft pos_max=%d < N-1=%d - "
              "mProcess() hook may not have run on every prefill ubatch "
              "(need_embd / logits=1 on every prompt position?). "
              "Drafts may degrade.\n",
              (int)cPosMax, cN - 1);
    }
  }

  bool mProcess(const mgs_batch &pBatchIn) override {
    if (pBatchIn.n_tokens <= 0) {
      return true;
    }

    // TODO: how to make it work with vision tokens?
    if (pBatchIn.token == nullptr || pBatchIn.embd != nullptr) {
      return true;
    }

    const int32_t cNTokens = pBatchIn.n_tokens;

    // remember the frist and last batch index for each sequence
    std::fill(lIBatchBeg.begin(), lIBatchBeg.end(), -1);
    std::fill(lIBatchEnd.begin(), lIBatchEnd.end(), -1);

    for (int vK = 0; vK < cNTokens; ++vK) {
      for (mgs_seq_id vSeqId = 0; vSeqId < (mgs_seq_id)vNSeq; ++vSeqId) {
        GGML_ASSERT(pBatchIn.n_seq_id[vK] == 1);

        if (pBatchIn.seq_id[vK][0] == vSeqId) {
          lIBatchEnd[vSeqId] = vK;
          if (lIBatchBeg[vSeqId] < 0) {
            lIBatchBeg[vSeqId] = vK;
          }
        }
      }
    }

    auto *vCtxTgt = this->vParams.vCtxTgt;
    auto *vCtxDft = this->vParams.vCtxDft;

    const size_t cRowBytes = (size_t)vNEmbd * sizeof(float);

    // catch-up decode: feed the target hidden states of this batch to the draft head
    fCommonBatchClear(vBatch);

    for (int vK = 0; vK < cNTokens; ++vK) {
      fCommonBatchAdd(vBatch, pBatchIn.token[vK], pBatchIn.pos[vK], {pBatchIn.seq_id[vK][0]}, 0);
    }

    // shift the tgt embeddings to the right by one position
    // assumes that the tokens in the batch are sequential for each sequence
    // i.e. we cannot have seq_id like this: [0, 0, 0, 1, 1, 0, 1, 1]
    //                                                       ^--- this is a problem
    // TODO:this is generally true, but would be nice to assert it
    {
      const float *vHTgt = mgs_get_embeddings_nextn(vCtxTgt);
      std::memcpy(vBatch.embd + (size_t)1 * vNEmbd, vHTgt, cRowBytes * (cNTokens - 1));
    }

    // fill the pending embeddings from a previous run
    auto vSetH = [&](int pIdx, const float *pHRow) {
      std::memcpy(vBatch.embd + (size_t)pIdx * vNEmbd, pHRow, cRowBytes);
    };

    for (mgs_seq_id vSeqId = 0; vSeqId < (mgs_seq_id)vNSeq; ++vSeqId) {
      if (lIBatchBeg[vSeqId] < 0) {
        continue;
      }

      vSetH(lIBatchBeg[vSeqId], lPendingH[vSeqId].data());
    }

    auto *vMemDft = mgs_get_memory(vCtxDft);

    bool vOk = true;
    for (int vHead = 0; vHead < vNMtpLayers; ++vHead) {
      if (vChainHeads) {
        for (mgs_seq_id vSeqId = 0; vSeqId < (mgs_seq_id)vNSeq; ++vSeqId) {
          if (lIBatchBeg[vSeqId] < 0) {
            continue;
          }
          mgs_memory_seq_rm(vMemDft, vSeqId, pBatchIn.pos[lIBatchBeg[vSeqId]], -1);
        }
        mgs_set_nextn_layer_offset(vCtxDft, vHead);
      }

      const int32_t cRc = mgs_decode(vCtxDft, vBatch);
      if (cRc != 0) {
        SPC_ERR("mgs_decode(ctx_dft) head=%d failed rc=%d (pos=%d)\n", vHead, (int)cRc, (int)pBatchIn.pos[0]);
        vOk = false;
        break;
      }
    }

    if (vChainHeads) {
      mgs_set_nextn_layer_offset(vCtxDft, 0); // restore default for non-draft decodes
    }
    if (!vOk) {
      return false;
    }

    for (mgs_seq_id vSeqId = 0; vSeqId < (mgs_seq_id)vNSeq; ++vSeqId) {
      if (lIBatchEnd[vSeqId] < 0) {
        continue;
      }

      const int32_t cNRows = lIBatchEnd[vSeqId] - lIBatchBeg[vSeqId] + 1;
      lVerifyHRows[vSeqId] = cNRows;
      lVerifyH[vSeqId].resize((size_t)cNRows * vNEmbd);

      for (int32_t vI = 0; vI < cNRows; ++vI) {
        const float *vH = mgs_get_embeddings_nextn_ith(vCtxTgt, lIBatchBeg[vSeqId] + vI);
        std::memcpy(lVerifyH[vSeqId].data() + (size_t)vI * vNEmbd, vH, cRowBytes);
      }

      std::memcpy(lPendingH[vSeqId].data(), lVerifyH[vSeqId].data() + (size_t)(cNRows - 1) * vNEmbd, cRowBytes);
    }

    return true;
  }

  void mDraft(common_speculative_draft_params_vec &pDparams) override {
    auto &vCtxDft = vParams.vCtxDft;

    fCommonBatchClear(vBatch);

    // keep track of which sequences are still drafting
    int vNDrafting = 0;
    std::vector<bool> lDrafting(vNSeq);

    const size_t cRowBytes = (size_t)vNEmbd * sizeof(float);

    for (mgs_seq_id vSeqId = 0; vSeqId < (mgs_seq_id)vNSeq; ++vSeqId) {
      auto &vDp = pDparams[vSeqId];

      if (!vDp.vDrafting) {
        continue;
      }

      vNDrafting++;
      lDrafting[vSeqId] = true;
      fCommonSamplerReset(lSmpls[vSeqId].get());

      fCommonBatchAdd(vBatch, vDp.vIdLast, vDp.vNPast, {vSeqId}, true);
      std::memcpy(vBatch.embd + (size_t)(vBatch.n_tokens - 1) * vNEmbd, lPendingH[vSeqId].data(), cRowBytes);

      lILast[vSeqId] = vBatch.n_tokens - 1;

      if (vChainHeads) {
        lChainH[vSeqId].assign(lPendingH[vSeqId].begin(), lPendingH[vSeqId].end());
      }
    }

    int vI = 0;

    while (vNDrafting > 0) {
      // each step decodes under a different head, i.e. a different decoder layer, and
      // KV is per layer. mProcess() filled this layer's KV only for positions < n_past
      // (prompt + accepted prefix) — nothing in the draft region yet. so reset the
      // draft region (the seq_rm lower bound is n_past, leaving the prompt KV intact)
      // and select head i so it rebuilds its own layer's KV there; decoding just the
      // latest token would leave its attention reading cells only another head wrote.
      if (vChainHeads) {
        auto *vMemDft = mgs_get_memory(vCtxDft);
        for (mgs_seq_id vSeqId = 0; vSeqId < (mgs_seq_id)vNSeq; ++vSeqId) {
          if (lDrafting[vSeqId]) {
            mgs_memory_seq_rm(vMemDft, vSeqId, pDparams[vSeqId].vNPast, -1);
          }
        }
        mgs_set_nextn_layer_offset(vCtxDft, vI);
      }

      int vRet = mgs_decode(vCtxDft, vBatch);
      if (vRet != 0) {
        SPC_ERR("mgs_decode[%d] returned %d\n", vI, vRet);
        break;
      }

      // rebuild the batch for the next step: the growing-KV paths re-add only the
      // new token (the KV already holds the prefix), while chained heads re-add the
      // whole prefix at the next head. dropped sequences are simply not re-added.
      fCommonBatchClear(vBatch);

      for (mgs_seq_id vSeqId = 0; vSeqId < (mgs_seq_id)vNSeq; ++vSeqId) {
        if (!lDrafting[vSeqId]) {
          continue;
        }

        auto *vSmpl = lSmpls[vSeqId].get();

        fCommonSamplerSample(vSmpl, vCtxDft, lILast[vSeqId], true);
        const float *vHRow = mgs_get_embeddings_nextn_ith(vCtxDft, lILast[vSeqId]);

        const auto *vCurP = fCommonSamplerGetCandidates(vSmpl, true);

        for (int vK = 0; vK < std::min(3, (int)vCurP->size); ++vK) {
          SPC_DBG(" - seq_id %d, draft candidate %3d, pos %3d: %6d (%8.3f) '%s'\n", vSeqId, vK, vI, vCurP->data[vK].id,
                  vCurP->data[vK].p, fCommonTokenToPiece(vCtxDft, vCurP->data[vK].id).c_str());
        }

        // add drafted token for each sequence
        const mgs_token cId = vCurP->data[0].id;

        // only collect very high-confidence draft tokens
        if (vCurP->data[0].p < vParams.vPMin) {
          lDrafting[vSeqId] = false;
          vNDrafting--;

          continue;
        }

        fCommonSamplerAccept(vSmpl, cId, true);

        auto &vDp = pDparams.at(vSeqId);
        auto &vResult = *vDp.vResult;

        vResult.push_back(cId);

        if (vParams.vNMax <= (int)vResult.size()) {
          lDrafting[vSeqId] = false;
          vNDrafting--;
          continue;
        }

        if (vChainHeads) {
          lChainH[vSeqId].insert(lChainH[vSeqId].end(), vHRow, vHRow + vNEmbd);

          const int cNRows = (int)vResult.size() + 1; // id_last + tokens drafted so far
          for (int vT = 0; vT < cNRows; ++vT) {
            const mgs_token cTok = (vT == 0) ? vDp.vIdLast : vResult[vT - 1];
            fCommonBatchAdd(vBatch, cTok, vDp.vNPast + vT, {vSeqId}, vT == cNRows - 1);
            std::memcpy(vBatch.embd + (size_t)(vBatch.n_tokens - 1) * vNEmbd,
                        lChainH[vSeqId].data() + (size_t)vT * vNEmbd, cRowBytes);
          }
        } else {
          fCommonBatchAdd(vBatch, cId, vDp.vNPast + vI + 1, {vSeqId}, true);
          std::memcpy(vBatch.embd + (size_t)(vBatch.n_tokens - 1) * vNEmbd, vHRow, cRowBytes);
        }

        lILast[vSeqId] = vBatch.n_tokens - 1;
      }

      if (vBatch.n_tokens == 0) {
        break;
      }

      ++vI;
    }

    if (vChainHeads) {
      mgs_set_nextn_layer_offset(vCtxDft, 0); // restore default for non-draft decodes
    }

    for (mgs_seq_id vSeqId = 0; vSeqId < (mgs_seq_id)vNSeq; ++vSeqId) {
      auto &vDp = pDparams[vSeqId];
      if (!vDp.vDrafting) {
        continue;
      }

      if (vDp.vResult->size() < (size_t)vParams.vNMin) {
        vDp.vResult->clear();
      }
    }
  }

  void mAccept(mgs_seq_id pSeqId, uint16_t pNAccepted, bool /*is_other*/) override {
    if (pSeqId < 0 || pSeqId >= (mgs_seq_id)vNSeq) {
      return;
    }

    const int32_t cNRows = lVerifyHRows[pSeqId];
    if (cNRows <= 0) {
      return;
    }

    const int32_t cIH = std::min<int32_t>(pNAccepted, cNRows - 1);
    const size_t cRowBytes = (size_t)vNEmbd * sizeof(float);
    std::memcpy(lPendingH[pSeqId].data(), lVerifyH[pSeqId].data() + (size_t)cIH * vNEmbd, cRowBytes);
  }

  bool mNeedEmbd() const override { return false; }

  bool mNeedEmbdNextn() const override { return true; }
};

// state of self-speculation (simple implementation, not ngram-map)
struct common_speculative_impl_ngram_simple : public common_speculative_impl {
  common_params_speculative_ngram_map vParams;

  // shared across all sequences
  common_ngram_simple_config vConfig;

  common_speculative_impl_ngram_simple(const common_params_speculative &pParams, uint32_t pNSeq,
                                       common_ngram_simple_config pConfig)
      : common_speculative_impl(cCommonSpeculativeTypeNgramSimple, pNSeq), vParams(pParams.vNgramSimple),
        vConfig(pConfig) {
    SPC_TRC("%s", "adding speculative implementation 'ngram-simple'\n");
    SPC_TRC("- size_n=%d, size_m=%d, min_hits=%d\n", this->vParams.vSizeN, this->vParams.vSizeM,
            this->vParams.vMinHits);
  }

  void begin(mgs_seq_id /*seq_id*/, const mgs_tokens & /*prompt*/) override {
    // noop
  }

  bool mProcess(const mgs_batch & /*batch*/) override {
    // TODO: implement
    return true;
  }

  void mDraft(common_speculative_draft_params_vec &pDparams) override {
    assert(pDparams.size() == vNSeq);

    for (mgs_seq_id vSeqId = 0; vSeqId < (mgs_seq_id)vNSeq; ++vSeqId) {
      auto &vDp = pDparams[vSeqId];
      if (!vDp.vDrafting) {
        continue;
      }

      *vDp.vResult = fCommonNgramSimpleDraft(vConfig, *vDp.vPrompt, vDp.vIdLast);
    }
  }

  void mAccept(mgs_seq_id /*seq_id*/, uint16_t /*n_accepted*/, bool /*is_other*/) override {
    // noop
  }

  bool mNeedEmbd() const override { return false; }
};

struct common_speculative_impl_ngram_map_k : public common_speculative_impl {
  // n_seq configs
  std::vector<common_ngram_map> lConfig;

  common_speculative_impl_ngram_map_k(const common_ngram_map &pConfig, uint32_t pNSeq)
      : common_speculative_impl(pConfig.vKeyOnly ? cCommonSpeculativeTypeNgramMapK : cCommonSpeculativeTypeNgramMapK4v,
                                pNSeq) {
    for (uint32_t vI = 0; vI < pNSeq; vI++) {
      this->lConfig.push_back(pConfig);
    }

    SPC_TRC("adding speculative implementation '%s'\n", fCommonSpeculativeTypeToStr(this->cType).c_str());
    SPC_TRC("- size_key=%d, size_value=%d, key_only=%d, min_hits=%d\n", pConfig.vSizeKey, pConfig.vSizeValue,
            pConfig.vKeyOnly, pConfig.vMinHits);
  }

  void begin(mgs_seq_id pSeqId, const mgs_tokens &pPrompt) override {
    GGML_ASSERT(pSeqId < (mgs_seq_id)vNSeq);

    fCommonNgramMapBegin(lConfig[pSeqId], pPrompt);
  }

  bool mProcess(const mgs_batch & /*batch*/) override {
    // TODO: implement
    return true;
  }

  void mDraft(common_speculative_draft_params_vec &pDparams) override {
    assert(pDparams.size() == vNSeq);

    for (mgs_seq_id vSeqId = 0; vSeqId < (mgs_seq_id)vNSeq; ++vSeqId) {
      auto &vDp = pDparams[vSeqId];
      if (!vDp.vDrafting) {
        continue;
      }

      fCommonNgramMapDraft(lConfig[vSeqId], *vDp.vPrompt, vDp.vIdLast, *vDp.vResult);
    }
  }

  void mAccept(mgs_seq_id pSeqId, uint16_t pNAccepted, bool pIsOther) override {
    GGML_ASSERT((pSeqId < (mgs_seq_id)lConfig.size()));

    if (pIsOther) {
      return;
    }

    fCommonNgramMapAccept(lConfig[pSeqId], pNAccepted);
  }

  bool mNeedEmbd() const override { return false; }
};

struct common_speculative_impl_ngram_mod : public common_speculative_impl {
  common_params_speculative_ngram_mod vParams;

  // shared across all sequences
  common_ngram_mod vMod;

  // enable trace logging if MGS_TRACE is set
  const bool cVerbose;

  struct seq_info {
    // the last position in the prompt that was added to the ngram container
    size_t vILast = 0;

    // length of the last drafted n-gram (number of tokens returned by draft)
    size_t vNDraftLast = 0;

    // consecutive accept rounds with low acceptance fraction (< 0.5)
    int vNLow = 0;
  };

  std::vector<seq_info> lSinfos;

  common_speculative_impl_ngram_mod(const common_params_speculative &pParams, uint32_t pNSeq)
      : common_speculative_impl(cCommonSpeculativeTypeNgramMod, pNSeq), vParams(pParams.vNgramMod),
        vMod(pParams.vNgramMod.vNMatch, 4 * 1024 * 1024), cVerbose(std::getenv("MGS_TRACE") != nullptr) {
    static_assert(sizeof(mgs_token) == sizeof(common_ngram_mod::entry_t));

    SPC_TRC("%s", "adding speculative implementation 'ngram-mod'\n");
    SPC_TRC("- n_match=%d, n_max=%d, n_min=%d\n", this->vParams.vNMatch, this->vParams.vNMax, this->vParams.vNMin);
    SPC_TRC("- mod size=%zu (%.3f MB)\n", vMod.size(), (float)(vMod.mSizeBytes()) / 1024 / 1024);

    if (this->vParams.vNMatch < 16) {
      SPC_WRN("ngram_mod n_match=%d is too small - poor quality is possible\n", this->vParams.vNMatch);
    }

    lSinfos.resize(pNSeq);
  }

  void begin(mgs_seq_id pSeqId, const mgs_tokens &pPrompt) override {
    auto &vSinfo = lSinfos[pSeqId];

    vSinfo.vILast = 0;
    vSinfo.vNDraftLast = 0;

    const size_t cN = vMod.mGetN();
    if (pPrompt.size() < cN) {
      return;
    }

    for (size_t vI = 0; vI < pPrompt.size() - cN; ++vI) {
      vMod.mAdd(pPrompt.data() + vI);
    }

    vSinfo.vILast = pPrompt.size() - cN;

    const double cF = (double)vMod.mGetUsed() / (double)vMod.size();
    SPC_TRC("ngram_mod occupancy = %zu/%zu (%.2f)\n", vMod.mGetUsed(), vMod.size(), cF);

    constexpr double cFThold = 0.25;
    if (cF > cFThold) {
      SPC_WRN("ngram_mod occupancy %.2f exceeds threshold (%.2f) - resetting\n", cF, cFThold);

      vMod.mReset();
    }
  }

  void mDraftOne(mgs_seq_id pSeqId, common_speculative_draft_params &pDparams) {
    auto &vSinfo = lSinfos[pSeqId];
    auto &vResult = *pDparams.vResult;

    const auto &vPrompt = *pDparams.vPrompt;

    vSinfo.vNDraftLast = 0;

    const size_t cCurLen = vPrompt.size();
    if (cCurLen < vMod.mGetN()) {
      return;
    }

    const size_t cN = vMod.mGetN();

    // add new ngrams in chunks
    if (vSinfo.vILast + 32 < cCurLen) {
      for (size_t vI = vSinfo.vILast; vI < cCurLen - cN; ++vI) {
        vMod.mAdd(vPrompt.data() + vI);
      }

      vSinfo.vILast = cCurLen - cN;
    }

    vResult.resize(cN + vParams.vNMax);
    for (size_t vI = 0; vI < cN - 1; ++vI) {
      vResult[vI] = vPrompt.at(cCurLen - cN + 1 + vI);
    }
    vResult[cN - 1] = pDparams.vIdLast;

    for (int vI = 0; vI < vParams.vNMax; ++vI) {
      const mgs_token cToken = vMod.mGet(vResult.data() + vI);
      if (cToken == common_ngram_mod::cEmpty) {
        if (vI < vParams.vNMin) {
          vResult.clear();
          return;
        }

        vResult.resize(cN + vI);
        break;
      }
      vResult[cN + vI] = cToken;
    }

    // only return the m tokens that were drafted
    for (size_t vI = 0; cN + vI < vResult.size(); ++vI) {
      vResult[vI] = vResult[cN + vI];
    }
    vResult.resize(vResult.size() - cN);

    // store length of drafted n-gram for later acceptance analysis
    vSinfo.vNDraftLast = vResult.size();
  }

  bool mProcess(const mgs_batch & /*batch*/) override {
    // TODO: implement
    return true;
  }

  void mDraft(common_speculative_draft_params_vec &pDparams) override {
    assert(pDparams.size() == vNSeq);

    for (mgs_seq_id vSeqId = 0; vSeqId < (mgs_seq_id)vNSeq; ++vSeqId) {
      auto &vDp = pDparams[vSeqId];
      if (!vDp.vDrafting) {
        continue;
      }

      mDraftOne(vSeqId, vDp);
    }
  }

  void mAccept(mgs_seq_id pSeqId, uint16_t pNAccepted, bool pIsOther) override {
    if (pIsOther) {
      return;
    }

    auto &vSinfo = lSinfos[pSeqId];

    // compute acceptance fraction if we have a recorded draft length
    if (vSinfo.vNDraftLast > 0) {
      const double cFAcc = (double)pNAccepted / (double)vSinfo.vNDraftLast;
      if (cFAcc < 0.25) {
        vSinfo.vNLow++;
        if (vSinfo.vNLow >= 5) {
          if (cVerbose) {
            SPC_TRC("low acceptance streak (%d) - resetting ngram_mod\n", vSinfo.vNLow);
          }

          vMod.mReset();
          vSinfo.vNLow = 0;
          vSinfo.vILast = 0;
        }
      } else {
        vSinfo.vNLow = 0;
      }
    }
  }

  bool mNeedEmbd() const override { return false; }
};

struct common_speculative_impl_ngram_cache : public common_speculative_impl {
  common_params_speculative_ngram_cache vParams;

  uint16_t vNDraft;

  bool vSaveDynamic;
  bool vSaveStatic;

  struct seq_info {
    size_t vCacheSize = 0; // number of tokens in n-gram cache

    common_ngram_cache vNgramCacheContext;
    common_ngram_cache vNgramCacheDynamic;
    common_ngram_cache vNgramCacheStatic;
  };

  std::vector<seq_info> lSinfos;

  common_speculative_impl_ngram_cache(const common_params_speculative &pParams, uint32_t pNSeq, uint16_t pNDraft,
                                      const std::string &pPathStatic, const std::string &pPathDynamic,
                                      bool pSaveDynamic, bool pSaveStatic)
      : common_speculative_impl(cCommonSpeculativeTypeNgramCache, pNSeq), vParams(pParams.vNgramCache),
        vNDraft(pNDraft), vSaveDynamic(pSaveDynamic), vSaveStatic(pSaveStatic) {
    SPC_TRC("%s", "adding speculative implementation 'ngram-cache'\n");
    SPC_TRC("- n_draft=%d, cache_static=%s, cache_dynamic=%s\n", pNDraft,
            pPathStatic.empty() ? "none" : pPathStatic.c_str(), pPathDynamic.empty() ? "none" : pPathDynamic.c_str());

    lSinfos.resize(pNSeq);

    if (!pPathStatic.empty()) {
      try {
        auto vNgramCacheStatic = fCommonNgramCacheLoad(pPathStatic);

        for (auto &vSinfo : lSinfos) {
          vSinfo.vNgramCacheStatic = vNgramCacheStatic;
        }
      } catch (...) {
        SPC_ERR("failed to open static lookup cache: %s", pPathStatic.c_str());
        GGML_ABORT("Couldn't read static lookup cache");
      }
    }

    if (!pPathDynamic.empty()) {
      try {
        auto vNgramCacheDynamic = fCommonNgramCacheLoad(pPathDynamic);

        for (auto &vSinfo : lSinfos) {
          vSinfo.vNgramCacheDynamic = vNgramCacheDynamic;
        }
      } catch (...) {
        SPC_ERR("failed to open dynamic lookup cache: %s", pPathDynamic.c_str());
        GGML_ABORT("Couldn't read dynamic lookup cache");
      }
    }
  }

  void begin(mgs_seq_id /*seq_id*/, const mgs_tokens & /*prompt*/) override {
    // noop
  }

  void mDraftOne(mgs_seq_id pSeqId, common_speculative_draft_params &pDparams) {
    auto &vSinfo = lSinfos[pSeqId];
    auto &vResult = *pDparams.vResult;

    const auto &vPrompt = *pDparams.vPrompt;

    if (vSinfo.vCacheSize < vPrompt.size() + 1) {
      mgs_tokens vTokensNew;
      vTokensNew.reserve(vPrompt.size() + 1 - vSinfo.vCacheSize);
      for (size_t vJ = vSinfo.vCacheSize; vJ < vPrompt.size(); ++vJ) {
        vTokensNew.push_back(vPrompt[vJ]);
      }
      vTokensNew.push_back(pDparams.vIdLast); // add the last token

      // Update context ngram cache with new dparams.prompt:
      fCommonNgramCacheUpdate(vSinfo.vNgramCacheContext, MGS_NGRAM_MIN, MGS_NGRAM_MAX, vTokensNew, vTokensNew.size(),
                              false);
      vSinfo.vCacheSize = vPrompt.size() + 1;
    }

    mgs_tokens vInp;
    vInp.reserve(vPrompt.size() + 1);
    for (size_t vJ = 0; vJ < vPrompt.size(); ++vJ) {
      vInp.push_back(vPrompt[vJ]);
    }
    vInp.push_back(pDparams.vIdLast);

    vResult.push_back(pDparams.vIdLast);

    fCommonNgramCacheDraft(vInp, vResult, vNDraft, MGS_NGRAM_MIN, MGS_NGRAM_MAX, vSinfo.vNgramCacheContext,
                           vSinfo.vNgramCacheDynamic, vSinfo.vNgramCacheStatic);

    if (vResult.size() > 0) {
      // delete first token in result (which is the id_last token)
      vResult.erase(vResult.begin());
    }
  }

  bool mProcess(const mgs_batch & /*batch*/) override {
    // TODO: implement
    return true;
  }

  void mDraft(common_speculative_draft_params_vec &pDparams) override {
    assert(pDparams.size() == vNSeq);

    for (mgs_seq_id vSeqId = 0; vSeqId < (mgs_seq_id)vNSeq; ++vSeqId) {
      auto &vDp = pDparams[vSeqId];
      if (!vDp.vDrafting) {
        continue;
      }

      mDraftOne(vSeqId, vDp);
    }
  }

  void mAccept(mgs_seq_id /*seq_id*/, uint16_t /*n_accepted*/, bool /*is_other*/) override {
    // noop
  }

  bool mNeedEmbd() const override { return false; }
};

struct common_speculative {
  common_speculative_draft_params_vec vDparams;

  // list of implementations to use and their states
  std::vector<std::unique_ptr<common_speculative_impl>> lImpls;

  // which implementaion was used for a given seq_id
  std::vector<common_speculative_impl *> lImplLast;
};

static common_ngram_map fGetCommonNgramMap(common_speculative_type pType,
                                           const common_params_speculative_ngram_map &pConfig) {
  uint16_t vSizeKey = pConfig.vSizeN;
  uint16_t vSizeValue = pConfig.vSizeM;
  bool vKeyOnly = pType == cCommonSpeculativeTypeNgramMapK;
  uint16_t vMinHits = pConfig.vMinHits;

  return common_ngram_map(vSizeKey, vSizeValue, vKeyOnly, vMinHits);
}

static common_speculative_impl_ngram_cache fCreateStateNgramCache(const common_speculative_config &pConfig,
                                                                  uint32_t pNSeq, const std::string &pPathStatic,
                                                                  const std::string &pPathDynamic) {
  uint16_t vNDraft = 8; // TODO get from config?

  // TODO bool param in common/common.h to set save_static/save_dynamic?
  bool vSaveStatic = false;
  bool vSaveDynamic = false;

  common_speculative_impl_ngram_cache vState(pConfig.vParams, pNSeq, vNDraft, pPathStatic, pPathDynamic, vSaveStatic,
                                             vSaveDynamic);

  return vState;
}

std::string fCommonSpeculativeTypeNameStr(const std::vector<common_speculative_type> &pTypes) {
  std::string vResult;

  for (size_t vI = 0; vI < pTypes.size(); vI++) {
    if (vI > 0) {
      vResult += ",";
    }
    vResult += fCommonSpeculativeTypeToStr(pTypes[vI]);
  }
  return vResult;
}

const char *fCommonSpeculativeAllTypesStr() {
  static std::string vAllTypesStr = []() {
    std::vector<common_speculative_type> lTypes;
    lTypes.reserve(cCommonSpeculativeTypeCount);
    for (int vI = 0; vI < cCommonSpeculativeTypeCount; vI++) {
      lTypes.push_back((common_speculative_type)vI);
    }
    return fCommonSpeculativeTypeNameStr(lTypes);
  }();
  return vAllTypesStr.c_str();
}

std::string fCommonSpeculativeTypeToStr(common_speculative_type pType) {
  switch (pType) {
  case cCommonSpeculativeTypeNone:
    return "none";
  case cCommonSpeculativeTypeDraftSimple:
    return "draft-simple";
  case cCommonSpeculativeTypeDraftEagle3:
    return "draft-eagle3";
  case cCommonSpeculativeTypeDraftMtp:
    return "draft-mtp";
  case cCommonSpeculativeTypeDraftDflash:
    return "draft-dflash";
  case cCommonSpeculativeTypeDraftDspark:
    return "draft-dspark";
  case cCommonSpeculativeTypeNgramSimple:
    return "ngram-simple";
  case cCommonSpeculativeTypeNgramMapK:
    return "ngram-map-k";
  case cCommonSpeculativeTypeNgramMapK4v:
    return "ngram-map-k4v";
  case cCommonSpeculativeTypeNgramMod:
    return "ngram-mod";
  case cCommonSpeculativeTypeNgramCache:
    return "ngram-cache";
  default:
    return "unknown";
  }
}

std::vector<common_speculative_type> fCommonSpeculativeTypesFromNames(const std::vector<std::string> &pNames) {
  std::vector<common_speculative_type> lTypes;
  lTypes.reserve(pNames.size());

  for (const auto &vName : pNames) {
    auto vType = dCommonSpeculativeTypeFromNameMap.find(vName);
    if (vType != dCommonSpeculativeTypeFromNameMap.end()) {
      if (vType->second == cCommonSpeculativeTypeNone) {
        return std::vector<common_speculative_type>{cCommonSpeculativeTypeNone};
      }
      lTypes.push_back(vType->second);
      continue;
    }
    throw std::invalid_argument("unknown speculative type: " + vName);
  }

  return lTypes;
}

common_speculative_type fCommonSpeculativeTypeFromName(const std::string &pName) {
  const auto cIt = dCommonSpeculativeTypeFromNameMap.find(pName);
  if (cIt == dCommonSpeculativeTypeFromNameMap.end()) {
    return cCommonSpeculativeTypeCount;
  }
  return cIt->second;
}

static uint32_t fCommonGetEnabledSpeculativeConfigs(const std::vector<common_speculative_type> &pConfigs) {
  uint32_t vResult = 0;
  for (size_t vI = 0; vI < pConfigs.size(); vI++) {
    vResult |= (1u << pConfigs[vI]);
  }
  return vResult;
}

int32_t fCommonSpeculativeNMax(const common_params_speculative *pSpec) {
  int32_t vNMax = 0;

  for (const auto cType : pSpec->lTypes) {
    switch (cType) {
    case cCommonSpeculativeTypeDraftSimple:
    case cCommonSpeculativeTypeDraftEagle3:
    case cCommonSpeculativeTypeDraftMtp:
    case cCommonSpeculativeTypeDraftDflash:
    case cCommonSpeculativeTypeDraftDspark:
      vNMax = std::max(vNMax, std::max(0, pSpec->vDraft.vNMax));
      break;
    case cCommonSpeculativeTypeNgramSimple:
      vNMax = std::max(vNMax, (int32_t)pSpec->vNgramSimple.vSizeM);
      break;
    case cCommonSpeculativeTypeNgramMapK:
      vNMax = std::max(vNMax, (int32_t)pSpec->vNgramMapK.vSizeM);
      break;
    case cCommonSpeculativeTypeNgramMapK4v:
      vNMax = std::max(vNMax, (int32_t)pSpec->vNgramMapK4v.vSizeM);
      break;
    case cCommonSpeculativeTypeNgramMod:
      vNMax = std::max(vNMax, std::max(0, pSpec->vNgramMod.vNMax));
      break;
    case cCommonSpeculativeTypeNgramCache:
      vNMax = std::max(vNMax, (int32_t)8);
      break;
    case cCommonSpeculativeTypeNone:
    case cCommonSpeculativeTypeCount:
      break;
    }
  }

  return vNMax;
}

common_params fCommonBaseParamsToSpeculative(const common_params &pParams) {
  const bool cHasDraft = pParams.vSpeculative.mHasDft();

  const auto &vParamsSpec = pParams.vSpeculative.vDraft;
  common_params vResult = pParams;

  vResult.vEmbedding = false;
  vResult.vPoolingType = MGS_POOLING_TYPE_UNSPECIFIED;

  if (cHasDraft) {
    vResult.lDevices = vParamsSpec.lDevices;
    vResult.vModel = vParamsSpec.vMparams;
    vResult.vNGpuLayers = vParamsSpec.vNGpuLayers;
    vResult.lTensorBuftOverrides = vParamsSpec.lTensorBuftOverrides;

    if (vParamsSpec.vCpuparams.vNThreads > 0) {
      vResult.vCpuparams.vNThreads = vParamsSpec.vCpuparams.vNThreads;
      vResult.vCpuparamsBatch.vNThreads = vParamsSpec.vCpuparamsBatch.vNThreads;
    }
  }

  vResult.vCacheTypeK = vParamsSpec.vCacheTypeK;
  vResult.vCacheTypeV = vParamsSpec.vCacheTypeV;
  vResult.vNOutputsMax = pParams.vNParallel;

  return vResult;
}

struct common_speculative_init_result::impl {
  impl() = default;
  ~impl() = default;

  // note: the order in which model, context, etc. are declared matters because their destructors will be called
  // bottom-to-top
  mgs_model_ptr vModel;
  mgs_context_ptr vContext;
};

common_speculative_init_result::common_speculative_init_result(common_params &pParams, mgs_model *pModelTgt,
                                                               mgs_context *pCtxTgt, mgs_model_ptr pDraftModel)
    : vPimpl(new impl{}) {
  const bool cHasDraft = pParams.vSpeculative.mHasDft();
  const bool cSpecMtp = std::find(pParams.vSpeculative.lTypes.begin(), pParams.vSpeculative.lTypes.end(),
                                  cCommonSpeculativeTypeDraftMtp) != pParams.vSpeculative.lTypes.end();
  GGML_ASSERT(cHasDraft || cSpecMtp);

  auto vMparams = fCommonModelParamsToMgs(pParams);
  auto vCparams = fCommonContextParamsToMgs(pParams);

  if (cSpecMtp) {
    vCparams.ctx_type = MGS_CONTEXT_TYPE_MTP;
  }

  // note: for small models maybe we can set this to the maximum possible draft from all speculative types
  //       the extra memory for small models is likely negligible?
  vCparams.n_rs_seq = 0;
  vCparams.ctx_other = pCtxTgt;

  std::string vModelPath;
  if (cHasDraft) {
    vModelPath = pParams.vSpeculative.vDraft.vMparams.vPath;
    LOG_INF("%s: loading draft model '%s'\n", __func__, vModelPath.c_str());

    mgs_model *vModelDft =
        pDraftModel ? pDraftModel.release() : mgs_model_load_from_file(pParams.vModel.vPath.c_str(), vMparams);
    if (vModelDft == NULL) {
      LOG_ERR("%s: failed to load draft model, '%s'\n", __func__, vModelPath.c_str());
      return;
    }

    vPimpl->vModel.reset(vModelDft);

    mgs_context *vCtxDft = mgs_init_from_model(vModelDft, vCparams);
    if (vCtxDft == nullptr) {
      LOG_ERR("%s: failed to create MTP context\n", __func__);
      return;
    }

    vPimpl->vContext.reset(vCtxDft);
  } else if (cSpecMtp) {
    vModelPath = pParams.vModel.vPath;

    LOG_INF("%s: creating MTP draft context against the target model '%s'\n", __func__, vModelPath.c_str());

    mgs_context *vCtxDft = mgs_init_from_model(pModelTgt, vCparams);
    if (vCtxDft == nullptr) {
      LOG_ERR("%s: failed to create MTP context\n", __func__);
      return;
    }

    vPimpl->vContext.reset(vCtxDft);
  }
}

common_speculative_init_result::~common_speculative_init_result() = default;

mgs_model *common_speculative_init_result::mModel() { return vPimpl->vModel.get(); }

mgs_context *common_speculative_init_result::mContext() { return vPimpl->vContext.get(); }

common_speculative_init_result_ptr fCommonSpeculativeInitFromParams(common_params &pParams, mgs_model *pModelTgt,
                                                                    mgs_context *pCtxTgt, mgs_model_ptr pDraftModel) {
  return std::make_unique<common_speculative_init_result>(pParams, pModelTgt, pCtxTgt, std::move(pDraftModel));
}

// initialization of the speculative decoding system
//
common_speculative *fCommonSpeculativeInit(common_params_speculative &pParams, uint32_t pNSeq) {
  // Compute the implementations to use based on the config and their order of preference
  std::vector<common_speculative_config> lConfigs = {}; // list of speculative configs to try
  {
    uint32_t vEnabledConfigs = fCommonGetEnabledSpeculativeConfigs(pParams.lTypes);

    bool vHasDraftSimple = (vEnabledConfigs & (1u << cCommonSpeculativeTypeDraftSimple));
    bool vHasDraftEagle3 =
        (vEnabledConfigs & (1u << cCommonSpeculativeTypeDraftEagle3)) && pParams.vDraft.vCtxDft != nullptr;
    bool vHasDraftMtp = (vEnabledConfigs & (1u << cCommonSpeculativeTypeDraftMtp)) && pParams.vDraft.vCtxDft != nullptr;
    bool vHasDraftDflash =
        (vEnabledConfigs & (1u << cCommonSpeculativeTypeDraftDflash)) && pParams.vDraft.vCtxDft != nullptr;
    bool vHasDraftDspark =
        (vEnabledConfigs & (1u << cCommonSpeculativeTypeDraftDspark)) && pParams.vDraft.vCtxDft != nullptr;

    bool vHasNgramCache = (vEnabledConfigs & (1u << cCommonSpeculativeTypeNgramCache));
    bool vHasNgramSimple = (vEnabledConfigs & (1u << cCommonSpeculativeTypeNgramSimple));
    bool vHasNgramMapK = (vEnabledConfigs & (1u << cCommonSpeculativeTypeNgramMapK));
    bool vHasNgramMapK4v = (vEnabledConfigs & (1u << cCommonSpeculativeTypeNgramMapK4v));
    bool vHasNgramMod = (vEnabledConfigs & (1u << cCommonSpeculativeTypeNgramMod));

    // when adding a new type - update here the logic above
    static_assert(cCommonSpeculativeTypeCount == 11);

    // this list here defines the priority of the speculators
    // the one with highest priority are listed first
    if (vHasNgramSimple) {
      // This implementation can guess a lot of tokens without any draft model.
      lConfigs.push_back(common_speculative_config(cCommonSpeculativeTypeNgramSimple, pParams));
    }
    if (vHasNgramMapK) {
      lConfigs.push_back(common_speculative_config(cCommonSpeculativeTypeNgramMapK, pParams));
    }
    if (vHasNgramMapK4v) {
      // This implementation can guess tokens with high acceptance rate but is more expensive.
      lConfigs.push_back(common_speculative_config(cCommonSpeculativeTypeNgramMapK4v, pParams));
    }
    if (vHasNgramMod) {
      lConfigs.push_back(common_speculative_config(cCommonSpeculativeTypeNgramMod, pParams));
    }
    if (vHasNgramCache) {
      lConfigs.push_back(common_speculative_config(cCommonSpeculativeTypeNgramCache, pParams));
    }
    if (vHasDraftSimple) {
      lConfigs.push_back(common_speculative_config(cCommonSpeculativeTypeDraftSimple, pParams));
    }
    if (vHasDraftEagle3) {
      lConfigs.push_back(common_speculative_config(cCommonSpeculativeTypeDraftEagle3, pParams));
    }
    if (vHasDraftMtp) {
      lConfigs.push_back(common_speculative_config(cCommonSpeculativeTypeDraftMtp, pParams));
    }
    if (vHasDraftDflash) {
      lConfigs.push_back(common_speculative_config(cCommonSpeculativeTypeDraftDflash, pParams));
    }
    if (vHasDraftDspark) {
      lConfigs.push_back(common_speculative_config(cCommonSpeculativeTypeDraftDspark, pParams));
    }
  }

  std::vector<std::unique_ptr<common_speculative_impl>> lImpls = {};

  for (const common_speculative_config &vConfig : lConfigs) {
    switch (vConfig.vType) {
    case cCommonSpeculativeTypeNone:
      break;
    case cCommonSpeculativeTypeDraftSimple: {
      lImpls.push_back(std::make_unique<common_speculative_impl_draft_simple>(vConfig.vParams, pNSeq));
      break;
    }
    case cCommonSpeculativeTypeDraftEagle3: {
      lImpls.push_back(std::make_unique<common_speculative_impl_draft_eagle3>(vConfig.vParams, pNSeq));
      break;
    }
    case cCommonSpeculativeTypeDraftMtp: {
      lImpls.push_back(std::make_unique<common_speculative_impl_draft_mtp>(vConfig.vParams, pNSeq));
      break;
    }
    case cCommonSpeculativeTypeDraftDflash: {
      lImpls.push_back(std::make_unique<common_speculative_impl_draft_dflash>(vConfig.vParams, pNSeq));
      break;
    }
    case cCommonSpeculativeTypeDraftDspark: {
      lImpls.push_back(std::make_unique<common_speculative_impl_draft_dflash>(vConfig.vParams, pNSeq,
                                                                              cCommonSpeculativeTypeDraftDspark));
      break;
    }
    case cCommonSpeculativeTypeNgramSimple: {
      common_ngram_map vNgramMap = fGetCommonNgramMap(vConfig.vType, vConfig.vParams.vNgramSimple);

      uint16_t vNgramSizeKey = vNgramMap.vSizeKey;
      uint16_t vMgramSizeValue = vNgramMap.vSizeValue;

      auto vConfigSimple = common_ngram_simple_config{/* .vSizeNgram = */ vNgramSizeKey,
                                                      /* .vSizeMgram = */ vMgramSizeValue};
      auto vState = std::make_unique<common_speculative_impl_ngram_simple>(
          /* .pParams = */ vConfig.vParams,
          /* .pNSeq   = */ pNSeq,
          /* .pConfig = */ vConfigSimple);
      lImpls.push_back(std::move(vState));
      break;
    }
    case cCommonSpeculativeTypeNgramMapK: {
      lImpls.push_back(std::make_unique<common_speculative_impl_ngram_map_k>(
          fGetCommonNgramMap(vConfig.vType, vConfig.vParams.vNgramMapK), pNSeq));
      break;
    }
    case cCommonSpeculativeTypeNgramMapK4v: {
      lImpls.push_back(std::make_unique<common_speculative_impl_ngram_map_k>(
          fGetCommonNgramMap(vConfig.vType, vConfig.vParams.vNgramMapK4v), pNSeq));
      break;
    }
    case cCommonSpeculativeTypeNgramMod: {
      lImpls.push_back(std::make_unique<common_speculative_impl_ngram_mod>(vConfig.vParams, pNSeq));
      break;
    }
    case cCommonSpeculativeTypeNgramCache: {
      auto vState = fCreateStateNgramCache(vConfig, pNSeq, pParams.vNgramCache.vLookupCacheStatic,
                                           pParams.vNgramCache.vLookupCacheDynamic);
      lImpls.push_back(std::make_unique<common_speculative_impl_ngram_cache>(vState));
      break;
    }
    default:
      break;
    }
  }

  if (lImpls.empty()) {
    SPC_TRC("%s", "no implementations specified for speculative decoding\n");
    return nullptr;
  }

  auto *vResult = new common_speculative{/* .vDparams  = */ common_speculative_draft_params_vec(pNSeq),
                                         /* .lImpls    = */ std::move(lImpls),
                                         /* .lImplLast = */ std::vector<common_speculative_impl *>(pNSeq, nullptr)};

  return vResult;
}

void fCommonSpeculativeFree(common_speculative *pSpec) {
  if (pSpec == nullptr) {
    return;
  }

  delete pSpec;
}

common_speculative_draft_params &fCommonSpeculativeGetDraftParams(common_speculative *pSpec, mgs_seq_id pSeqId) {
  GGML_ASSERT(pSpec);
  GGML_ASSERT(pSeqId < (mgs_seq_id)pSpec->vDparams.size());

  return pSpec->vDparams[pSeqId];
}

void fCommonSpeculativeBegin(common_speculative *pSpec, mgs_seq_id pSeqId, const mgs_tokens &pPrompt) {
  if (pSpec == nullptr) {
    return;
  }

  for (auto &vImpl : pSpec->lImpls) {
    common_time_meas vTm(vImpl->vTBeginUs, !vImpl->cGenPerf);
    vImpl->begin(pSeqId, pPrompt);
    vImpl->vNCallBegin++;
  }
}

bool fCommonSpeculativeProcess(common_speculative *pSpec, const mgs_batch &pBatch) {
  bool vResult = true;

  if (pSpec == nullptr) {
    return vResult;
  }

  for (auto &vImpl : pSpec->lImpls) {
    vResult = vResult && vImpl->mProcess(pBatch);
  }

  return vResult;
}

bool fCommonSpeculativeNeedEmbd(common_speculative *pSpec) {
  if (pSpec == nullptr) {
    return false;
  }

  for (auto &vImpl : pSpec->lImpls) {
    if (vImpl->mNeedEmbd()) {
      return true;
    }
  }

  return false;
}

bool fCommonSpeculativeNeedEmbdNextn(common_speculative *pSpec) {
  if (pSpec == nullptr) {
    return false;
  }

  for (auto &vImpl : pSpec->lImpls) {
    if (vImpl->mNeedEmbdNextn()) {
      return true;
    }
  }

  return false;
}

void fCommonSpeculativeDraft(common_speculative *pSpec) {
  if (pSpec == nullptr) {
    return;
  }

  auto &vDparams = pSpec->vDparams;

  {
    int vNDrafting = 0;

    for (auto &vDp : vDparams) {
      GGML_ASSERT(!vDp.vDrafting || vDp.vResult->empty());

      if (vDp.vDrafting) {
        vNDrafting++;
      }
    }

    if (vNDrafting == 0) {
      return;
    }
  }

  for (auto &vImpl : pSpec->lImpls) {
    {
      common_time_meas vTm(vImpl->vTDraftUs, !vImpl->cGenPerf);
      vImpl->mDraft(vDparams);
      vImpl->vNCallDraft++;
    }

    int vNDrafting = 0;

    for (mgs_seq_id vSeqId = 0; vSeqId < (mgs_seq_id)vDparams.size(); ++vSeqId) {
      auto &vDp = vDparams[vSeqId];

      if (!vDp.vDrafting) {
        continue;
      }

      auto &vResult = *vDp.vResult;

      // a new draft has been sampled
      if (vDp.vDrafting && !vResult.empty()) {
        vDp.vDrafting = false;

        if (vDp.vNMax > 0) {
          if (!vResult.empty() && (int)vResult.size() > vDp.vNMax) {
            SPC_DBG("truncating draft to %d tokens\n", vDp.vNMax);
            vResult.resize(vDp.vNMax);
          }
        }

        if (!vResult.empty()) {
          SPC_DBG("called impl %s, hist size = %zu, call_count = %zu, gen = %zu\n",
                  fCommonSpeculativeTypeToStr(vImpl.get()->cType).c_str(), vDp.vPrompt->size(),
                  vImpl.get()->vNCallDraft, vResult.size());

          // remember which implementation was used
          pSpec->lImplLast[vSeqId] = vImpl.get();

          vImpl->vNGenDrafts++;
          vImpl->vNGenTokens += vResult.size();
        }
      }

      if (vDp.vDrafting) {
        vNDrafting++;
      }
    }

    if (vNDrafting == 0) {
      break;
    }
  }

  // these sequences failed to generate a draft
  for (mgs_seq_id vSeqId = 0; vSeqId < (mgs_seq_id)vDparams.size(); ++vSeqId) {
    auto &vDp = vDparams[vSeqId];

    if (vDp.vDrafting) {
      vDp.vDrafting = false;
    }
  }
}

void fCommonSpeculativeAccept(common_speculative *pSpec, mgs_seq_id pSeqId, uint16_t pNAccepted) {
  common_speculative_impl *vImpl = pSpec->lImplLast[pSeqId];

  GGML_ASSERT(vImpl);

  {
    common_time_meas vTm(vImpl->vTAcceptUs, !vImpl->cGenPerf);

    if (vImpl->lNAccTokensPerPos.size() < pNAccepted) {
      vImpl->lNAccTokensPerPos.resize(pNAccepted, 0);
    }

    for (size_t vI = 0; vI < pNAccepted; ++vI) {
      vImpl->lNAccTokensPerPos[vI]++;
    }

    if (pNAccepted > 0) {
      vImpl->vNAccDrafts++;
      vImpl->vNAccTokens += pNAccepted;
    }

    vImpl->mAccept(pSeqId, pNAccepted, false);
    vImpl->vNCallAccept++;
  }

  // accept with the rest of the implementations, using is_other == true
  for (auto &vImplOther : pSpec->lImpls) {
    if (vImplOther.get() != vImpl) {
      vImplOther->mAccept(pSeqId, pNAccepted, true);
    }
  }
}

// TODO: support the case of more than one speculative implementations having a state
bool fCommonSpeculativeGetState(common_speculative *pSpec, mgs_seq_id pSeqId, std::vector<uint8_t> &pData) {
  if (pSpec == nullptr) {
    return false;
  }

  for (auto &vImpl : pSpec->lImpls) {
    if (vImpl->mGetState(pSeqId, pData)) {
      return true;
    }
  }

  return false;
}

void fCommonSpeculativeSetState(common_speculative *pSpec, mgs_seq_id pSeqId, const std::vector<uint8_t> &pData) {
  if (pSpec == nullptr) {
    return;
  }

  for (auto &vImpl : pSpec->lImpls) {
    vImpl->mSetState(pSeqId, pData);
  }
}

void fCommonSpeculativePrintStats(const common_speculative *pSpec) {
  if (pSpec == nullptr) {
    return;
  }

  for (const auto &vImpl : pSpec->lImpls) {
    std::string vStrPerf;
    if (vImpl->cGenPerf) {
      std::ostringstream vOss;
      vOss << std::fixed << std::setprecision(3) << vImpl->vTBeginUs / 1000.0 << ", ";
      vOss << std::fixed << std::setprecision(3) << vImpl->vTDraftUs / 1000.0 << ", ";
      vOss << std::fixed << std::setprecision(3) << vImpl->vTAcceptUs / 1000.0;
      vStrPerf = ", dur(b,g,a) = " + vOss.str() + " ms";
    } else {
      vStrPerf = "";
    }

    std::string vStrStats;
    if (vImpl->vNCallAccept > 0) {
      const double cMean = 1.0 + (double)vImpl->vNAccTokens / (double)vImpl->vNCallAccept;
      std::ostringstream vTmp;
      vTmp << std::fixed << std::setprecision(3);
      for (size_t vI = 0; vI < vImpl->lNAccTokensPerPos.size(); ++vI) {
        if (vI > 0) {
          vTmp << ", ";
        }
        vTmp << (double)vImpl->lNAccTokensPerPos[vI] / (double)vImpl->vNCallAccept;
      }
      std::ostringstream vOss;
      vOss << std::fixed << std::setprecision(2) << cMean;
      vStrStats = ", #mean acc len = " + vOss.str() + ", #acc rate/pos = (" + vTmp.str() + ")";
    }

    SPC_TRC("statistics %16s: #calls(b,g,a) = %4zu %6zu %6zu, #gen drafts = %6zu, #acc drafts = %5zu, #gen tokens = "
            "%6zu, #acc tokens = %5zu%s%s\n",
            fCommonSpeculativeTypeToStr(vImpl->cType).c_str(), vImpl->vNCallBegin, vImpl->vNCallDraft,
            vImpl->vNCallAccept, vImpl->vNGenDrafts, vImpl->vNAccDrafts, vImpl->vNGenTokens, vImpl->vNAccTokens,
            vStrStats.c_str(), vStrPerf.c_str());
  }
}
