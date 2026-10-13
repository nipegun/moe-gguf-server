#pragma once

#include "mgs-memory-hybrid.h"

#include <array>
#include <cstdint>
#include <limits>
#include <memory>
#include <vector>

//
// mgs_memory_hybrid_idx
//

// mgs_memory_hybrid plus a third cache with one indexer key per token, for block-sparse attention (qwen4exp QSA,
// glm5-next k-pool DSA)
// the indexer is a side buffer over the attention cells: same size, padding, streams and slots, so cell j is one
// token in both

class mgs_memory_hybrid_idx : public mgs_memory_hybrid {
public:
  mgs_memory_hybrid_idx(const mgs_model &pModel,
                        /* attn */
                        ggml_type pTypeK, ggml_type pTypeV, bool pVTrans, uint32_t pKvSize, uint32_t pNPad,
                        uint32_t pNSwa, mgs_swa_type pSwaType,
                        /* recurrent */
                        ggml_type pTypeR, ggml_type pTypeS, uint32_t pRsSize,
                        /* common */
                        uint32_t pNSeqMax, uint32_t pNRsSeq, bool pOffload, bool pUnified,
                        /* layer filters */
                        const layer_filter_cb &pFilterAttn, const layer_filter_cb &pFilterRecr,
                        /* the indexer cache exists only if this is given */
                        const layer_filter_cb &pFilterIdx);

  // defined out of line because kpool_layout is incomplete here
  ~mgs_memory_hybrid_idx();

  //
  // mgs_memory_i
  //

  mgs_memory_context_ptr mInitBatch(mgs_batch_allocr &pBalloc, uint32_t pNUbatch, bool pEmbdAll) override;

  mgs_memory_context_ptr mInitFull() override;

  mgs_memory_context_ptr mInitUpdate(mgs_context *pLctx, bool pOptimize) override;

  void mClear(bool pData) override;

  bool mSeqRm(mgs_seq_id pSeqId, mgs_pos p0, mgs_pos p1) override;
  void mSeqCp(mgs_seq_id pSeqIdSrc, mgs_seq_id pSeqIdDst, mgs_pos p0, mgs_pos p1) override;
  void mSeqKeep(mgs_seq_id pSeqId) override;
  void mSeqAdd(mgs_seq_id pSeqId, mgs_pos p0, mgs_pos p1, mgs_pos pShift) override;
  void mSeqDiv(mgs_seq_id pSeqId, mgs_pos p0, mgs_pos p1, int pD) override;

  std::map<ggml_backend_buffer_type_t, size_t> mMemoryBreakdown() const override;

  // state write/load

  void mStateWrite(mgs_io_write_i &pIo, mgs_seq_id pSeqId = -1, mgs_state_seq_flags pFlags = 0) const override;
  void mStateRead(mgs_io_read_i &pIo, mgs_seq_id pSeqId = -1, mgs_state_seq_flags pFlags = 0) override;

  //
  // mgs_memory_hybrid_idx specific API
  //

  mgs_kv_cache *mGetMemIdx() const; // nullptr when the model carries no indexer

  // the model's indexer pool size
  uint32_t mGetKpool() const { return vHparamsIdx.vIndexerKpool; }

  // whether pools are kpool consecutive cells in sequence order (qwen4exp) instead of kpool consecutive positions
  bool mGetKpoolByOrder() const { return vHparamsIdx.vIndexerKpoolByOrder; }

  // which cells of a sequence make up which pool of kpool consecutive positions (or cells, in order mode)
  // it is kept here because it outlives the batch: pools are fixed by the positions relative to the
  // sequence's first one, so a ubatch only ever appends to it. sequence edits drop it, see aMemIdxStale
  struct kpool_layout;

  const kpool_layout &mKpoolLayoutUpdate();
  const kpool_layout &mKpoolLayoutGet() const;

  // the pooled keys persist in the idx cache across batches. a sequence edit can regroup the pools
  // from some position on, which stales every pooled key at or after it. cPosClean means none
  using stale_pos_t = std::array<mgs_pos, MGS_MAX_SEQ>;

  static constexpr mgs_pos cPosClean = std::numeric_limits<mgs_pos>::max();

  static stale_pos_t mStalePosClean() {
    stale_pos_t aRes;
    aRes.fill(cPosClean);
    return aRes;
  }

  const stale_pos_t &mMemIdxStaleGet() const { return aMemIdxStale; }
  void mMemIdxStaleClear() { aMemIdxStale.fill(cPosClean); }

private:
  // forget seq_id (all of it if seq_id < 0) in every cache at once, so a failed restore cannot leave the caches
  // out of step; seq_id < 0 drops the whole context, as the caches themselves do on a failed restore
  void mStateDrop(mgs_seq_id pSeqId);

  // the indexer cache holds one key head per layer, so it needs its own hparams:
  // mgs_kv_cache keeps a reference to what it is given
  mgs_hparams vHparamsIdx;

  const std::unique_ptr<mgs_kv_cache> cMemIdx;

  // unique_ptr because kpool_layout is incomplete here
  std::unique_ptr<kpool_layout> vKpoolLay;

  // seq_id < 0 stales every sequence, p0 < 0 stales the sequence from its first position
  void mMemIdxStaleSet(mgs_seq_id pSeqId, mgs_pos p0);

  // the position an edit at p0 stales the sequence from
  mgs_pos mMemIdxStalePos(mgs_seq_id pSeqId, mgs_pos p0) const;

  stale_pos_t aMemIdxStale = mStalePosClean();
};

class mgs_memory_hybrid_idx_context : public mgs_memory_hybrid_context {
public:
  class kpool_access {
  public:
    ggml_tensor *mGatherKeyGate(ggml_tensor *pIdxs) const;
    ggml_tensor *mScatterPooled(ggml_tensor *pValues, ggml_tensor *pIdxs) const;
    ggml_tensor *mGatherPooled(ggml_tensor *pIdxs) const;

  private:
    friend class mgs_memory_hybrid_idx_context;

    kpool_access(ggml_context *pCtx, ggml_tensor *pK, int64_t pNEmbd);

    ggml_context *vCtx;
    ggml_tensor *vKeyGate;
    ggml_tensor *vPooled;
  };

  using slot_info_vec_t = mgs_kv_cache::slot_info_vec_t;

  // used for errors
  explicit mgs_memory_hybrid_idx_context(mgs_memory_status pStatus);

  // used to create a full-cache context
  explicit mgs_memory_hybrid_idx_context(mgs_memory_hybrid_idx *pMem);

  // used to create an update context
  mgs_memory_hybrid_idx_context(mgs_memory_hybrid_idx *pMem, mgs_context *pLctx, bool pOptimize);

  // used to create a batch processing context from a batch
  mgs_memory_hybrid_idx_context(mgs_memory_hybrid_idx *pMem, slot_info_vec_t pSinfosAttn, slot_info_vec_t pSinfosIdx,
                                std::vector<mgs_ubatch> pUbatches);

  // defined out of line because kpool_state is incomplete here
  ~mgs_memory_hybrid_idx_context();

  //
  // mgs_memory_context_i
  //

  bool mNext() override;
  bool mApply() override;

  //
  // mgs_memory_hybrid_idx_context specific API
  //

  // nullptr with no indexer
  const mgs_kv_cache_context *mGetIdx() const;

  // streams in the current slot info, the `ns` of mGetK/mGetV; 1 if unified
  uint32_t mGetNStream() const;

  // glm5-next and qwen4exp: complete pools of kpool cells per sequence, scored as whole pools
  uint32_t mGetNKpool() const;    // padded pool count, where the last pool is always unused
  uint32_t mGetNKpoolNew() const; // pools to re-pool this ubatch, padded to a stable bound, never below 1
  kpool_access mGetKpoolAccess(ggml_context *pCtx, int32_t pIl, int64_t pNEmbd) const;
  // pSelMask (F32 [n_sel, 1, 1, n_tokens], can be null): 0 for the live selection slots, -inf for the dead ones
  // pNewPoolPos (I32 [4*n_new]): M-RoPE position of each new pool's first member, for pooled keys rotated at
  // pooling time
  void mSetInputKpool(ggml_tensor *pPoolCells, ggml_tensor *pPoolIdxs, ggml_tensor *pPoolMask, ggml_tensor *pTailIdxs,
                      ggml_tensor *pSelMask, ggml_tensor *pNewPoolIdxs, ggml_tensor *pNewPoolRep,
                      const mgs_ubatch *pUbatch, ggml_tensor *pNewPoolPos = nullptr) const;

private:
  mgs_memory_hybrid_idx *vMem = nullptr;

  // streams per ubatch, read from the slot infos before cCtxIdx takes them
  // declared first, so it is initialised while pSinfosIdx is still intact
  const std::vector<uint32_t> lNsUbatch;

  // the indexer cells of each ubatch, kept for pools in cache order (qwen4exp): token s*n + i of ubatch u
  // sits in cell idxs[s][i] of stream strm[s] of lSinfosKpool[u], and several cells can share a position
  const slot_info_vec_t lSinfosKpool;

  // null unless the model has an indexer
  const mgs_memory_context_ptr cCtxIdx;

  // mirrors the base class's ubatch cursor, which is private there
  size_t vICur = 0;

  // which pools of the layout this ubatch must re-pool; the layout itself belongs to the memory
  struct kpool_state;
  kpool_state mKpoolBuildSizes() const;
  void mKpoolBuildState(const mgs_ubatch &pUbatch);
  const kpool_state &mKpoolCur() const;

  // unique_ptr because kpool_state is incomplete here
  std::unique_ptr<kpool_state> vKpoolSt;

  // the ubatch vKpoolSt was built for, guards against reads before apply
  size_t vIKpool = SIZE_MAX;

  // whether this context tracks k-pool states
  bool mKpoolTrack() const;

  // positions each sequence must re-pool from, cleared only after the first ubatch succeeds
  mgs_memory_hybrid_idx::stale_pos_t aMemIdxStaleBatch = mgs_memory_hybrid_idx::mStalePosClean();
};
