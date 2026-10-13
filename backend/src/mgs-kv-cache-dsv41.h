#pragma once

#include "mgs-kv-cache.h"
#include "mgs-kv-cache-iswa.h"

#include <map>
#include <memory>
#include <unordered_map>
#include <vector>

// DeepSeek-V4.1 KV cache
//
// Every layer keeps a raw sliding-window K cache. On top of it, the layers with a compression ratio
// read compressed KV rows that only a few "KV source" layers produce: each source pools `ratio`
// consecutive tokens into one row with a softmax gate (ratio 1 is a plain projection) and the
// layers after it, up to the next source, read the same rows. The same holds for the lightning
// indexer keys, which only the "index key owner" layers write.
//
// The compressed storage is organised in groups, one per distinct compression ratio. A group holds
// a K-only cache with the rows of its source layers, an indexer key cache for its key owners and the
// compressor state that carries an incomplete group of tokens across ubatches. Layers that read a
// source's rows have no storage of their own: the graph addresses them through their source layer.

// Ring of not yet pooled compressor inputs (one plane per stream, plus rollback snapshot planes).
class mgs_dsv41_comp_state {
public:
  using stream_copy_info = mgs_kv_cache::stream_copy_info;

  stream_copy_info vScInfo;

  mgs_dsv41_comp_state(const mgs_model &pModel, bool pOffload, bool pUnified, uint32_t pNSeqMax, uint32_t pRatio,
                       uint32_t pStateSize, uint32_t pNEmbdState, uint32_t pNRsSeq, const char *pName,
                       const mgs_memory_i::layer_filter_cb &pFilter);

  void mClear(mgs_seq_id pSeqId, bool pData);
  void mSeqCp(mgs_seq_id pSeqIdSrc, mgs_seq_id pSeqIdDst);
  void mApplyCopies(const stream_copy_info &pScInfo) const;

  uint32_t mGetRatio() const;
  uint32_t mGetStateSize() const;
  uint32_t mGetNStream() const;
  uint32_t mGetNRsSeq() const;
  uint32_t mGetNRows() const;

  std::map<ggml_backend_buffer_type_t, size_t> mMemoryBreakdown() const;

  void mStateWrite(mgs_io_write_i &pIo, mgs_seq_id pSeqId, mgs_state_seq_flags pFlags,
                   const std::vector<uint32_t> &pRsIdx) const;
  void mStateRead(mgs_io_read_i &pIo, mgs_seq_id pSeqId, mgs_state_seq_flags pFlags);

  ggml_tensor *mGetKvAll(ggml_context *pCtx, int32_t pIl) const;
  ggml_tensor *mGetScoreAll(ggml_context *pCtx, int32_t pIl) const;

  ggml_tensor *mCpyKv(ggml_context *pCtx, ggml_tensor *pCur, ggml_tensor *pIdxs, int32_t pIl) const;
  ggml_tensor *mCpyScore(ggml_context *pCtx, ggml_tensor *pCur, ggml_tensor *pIdxs, int32_t pIl) const;

private:
  struct layer {
    uint32_t vIl;

    ggml_tensor *vKv;
    ggml_tensor *vScore;

    std::vector<ggml_tensor *> lKvStream;
    std::vector<ggml_tensor *> lScoreStream;
  };

  const uint32_t cRatio;
  const uint32_t cStateSize;
  const uint32_t cNEmbdState;
  const uint32_t cNStream;
  const uint32_t cNRsSeq;

  std::vector<std::pair<ggml_context_ptr, ggml_backend_buffer_ptr>> lCtxsBufs;

  std::vector<layer> lLayers;

  std::unordered_map<int32_t, int32_t> dMapLayerIds;

  size_t mTotalSize() const;
};

//
// mgs_kv_cache_dsv41
//

class mgs_kv_cache_dsv41 : public mgs_memory_i {
public:
  // the compressed storage of one compression ratio
  struct comp_group {
    uint32_t vRatio = 0;

    // K-only views of the model hparams for the compressed rows and the indexer keys
    mgs_hparams vHparamsComp;
    mgs_hparams vHparamsIdx;

    std::unique_ptr<mgs_kv_cache> vKvComp;
    std::unique_ptr<mgs_kv_cache> vKvIdx; // nullptr when no layer of this ratio owns indexer keys
    std::unique_ptr<mgs_dsv41_comp_state> vState;

    comp_group(const mgs_hparams &pHparams) : vHparamsComp(pHparams), vHparamsIdx(pHparams) {}
  };

  mgs_kv_cache_dsv41(const mgs_model &pModel, ggml_type pTypeK, ggml_type pTypeV, bool pVTrans, bool pOffload,
                     bool pSwaFull, bool pUnified, uint32_t pKvSize, uint32_t pNSeqMax, uint32_t pNUbatch,
                     uint32_t pNPad, uint32_t pNRsSeq, const layer_filter_cb &pFilter);

  ~mgs_kv_cache_dsv41() = default;

  //
  // mgs_memory_i
  //

  mgs_memory_context_ptr mInitBatch(mgs_batch_allocr &pBalloc, uint32_t pNUbatch, bool pEmbdAll) override;

  mgs_memory_context_ptr mInitFull() override;

  mgs_memory_context_ptr mInitUpdate(mgs_context *pLctx, bool pOptimize) override;

  bool mGetCanShift() const override;

  void mClear(bool pData) override;

  bool mSeqRm(mgs_seq_id pSeqId, mgs_pos p0, mgs_pos p1) override;
  void mSeqCp(mgs_seq_id pSeqIdSrc, mgs_seq_id pSeqIdDst, mgs_pos p0, mgs_pos p1) override;
  void mSeqKeep(mgs_seq_id pSeqId) override;
  void mSeqAdd(mgs_seq_id pSeqId, mgs_pos p0, mgs_pos p1, mgs_pos pShift) override;
  void mSeqDiv(mgs_seq_id pSeqId, mgs_pos p0, mgs_pos p1, int pD) override;

  mgs_pos mSeqPosMin(mgs_seq_id pSeqId) const override;
  mgs_pos mSeqPosMax(mgs_seq_id pSeqId) const override;

  std::map<ggml_backend_buffer_type_t, size_t> mMemoryBreakdown() const override;

  void mStateWrite(mgs_io_write_i &pIo, mgs_seq_id pSeqId = -1, mgs_state_seq_flags pFlags = 0) const override;
  void mStateRead(mgs_io_read_i &pIo, mgs_seq_id pSeqId = -1, mgs_state_seq_flags pFlags = 0) override;

  //
  // mgs_kv_cache_dsv41 specific API
  //

  mgs_kv_cache_iswa *mGetRaw() const;

  uint32_t mGetNGroup() const;
  const comp_group &mGetGroup(uint32_t pG) const;

  uint32_t mGetNRsSeq() const;
  const std::vector<uint32_t> &mGetRsIdx() const;
  void mResetRsIdxForUbatches(const std::vector<mgs_ubatch> &pUbatches);

private:
  mgs_hparams vHparamsRaw;

  const uint32_t cNSeqMax;
  const uint32_t cNRsSeq;

  std::vector<uint32_t> lRsIdx;

  std::unique_ptr<mgs_kv_cache_iswa> vKvRaw;

  // owned through pointers: the caches keep references to the hparams of their group
  std::vector<std::unique_ptr<comp_group>> lGroups;

  void mClearCompressed(mgs_seq_id pSeqId, bool pData);
};

// The raw attention only uses the SWA half of kv_raw. The base half is kept for the generic ISWA
// bookkeeping, but it has no layers to expose here.
class mgs_kv_cache_dsv41_raw_context : public mgs_memory_context_i {
public:
  using slot_info_vec_t = mgs_kv_cache::slot_info_vec_t;

  mgs_kv_cache_dsv41_raw_context(mgs_kv_cache_iswa *pKv);

  mgs_kv_cache_dsv41_raw_context(mgs_kv_cache_iswa *pKv, mgs_context *pLctx, bool pOptimize);

  mgs_kv_cache_dsv41_raw_context(mgs_kv_cache_iswa *pKv, slot_info_vec_t pSinfosBaseWrite,
                                 slot_info_vec_t pSinfosSwaWrite, slot_info_vec_t pSinfosSwaRead,
                                 std::vector<mgs_ubatch> pUbatches, std::vector<mgs_ubatch> pUbatchesWrite);

  bool mNext() override;
  bool mApply() override;

  mgs_memory_status mGetStatus() const override;
  const mgs_ubatch &mGetUbatch() const override;

  uint32_t mGetNKv() const;
  uint32_t mGetNWrite() const;

  ggml_tensor *mGetK(ggml_context *pCtx, int32_t pIl) const;
  ggml_tensor *mCpyK(ggml_context *pCtx, ggml_tensor *pKCur, ggml_tensor *pKIdxs, int32_t pIl) const;

  ggml_tensor *mBuildInputKIdxs(ggml_context *pCtx, const mgs_ubatch &pUbatch) const;
  ggml_tensor *mBuildInputKRot(ggml_context *pCtx) const;

  void mSetInputKIdxs(ggml_tensor *pDst) const;
  void mSetInputKqMask(ggml_tensor *pDst, const mgs_ubatch *pUbatch, bool pCausalAttn) const;
  void mSetInputKRot(ggml_tensor *pDst) const;

  // the tokens that precede every token of the ubatch, for the engram n-gram hash
  void mGetPrevTokens(const mgs_ubatch &pUbatch, uint32_t pN, std::vector<mgs_token> &pRes) const;

private:
  size_t vINext = 0;

  mgs_kv_cache *vKvSwa = nullptr;

  slot_info_vec_t vSinfosWrite;
  slot_info_vec_t vSinfosRead;
  std::vector<mgs_ubatch> lUbatches;
  std::vector<mgs_ubatch> lUbatchesWrite;

  const mgs_memory_context_ptr cCtxBaseMem;
  const mgs_memory_context_ptr cCtxSwaMem;

  uint32_t vNKv = 0;

  const mgs_memory_status cStatus;
};

// Compressed KV rows are graph outputs, not normal token KV writes.
// Keep a small context that exposes K tensors without generic apply() semantics.
class mgs_kv_cache_dsv41_comp_context {
public:
  using slot_info_vec_t = mgs_kv_cache::slot_info_vec_t;

  mgs_kv_cache_dsv41_comp_context(mgs_kv_cache *pKv);

  mgs_kv_cache_dsv41_comp_context(mgs_kv_cache *pKv, slot_info_vec_t pSinfos, std::vector<mgs_ubatch> pUbatches);

  bool mNext();

  uint32_t mGetNKv() const;

  // pIl is the layer that owns the storage (a KV source or an index key owner)
  ggml_tensor *mGetK(ggml_context *pCtx, int32_t pIl) const;
  ggml_tensor *mCpyK(ggml_context *pCtx, ggml_tensor *pKCur, ggml_tensor *pKIdxs, int32_t pIl) const;

  ggml_tensor *mBuildInputKRot(ggml_context *pCtx) const;
  void mSetInputKRot(ggml_tensor *pDst) const;

private:
  mgs_kv_cache *vKv;

  size_t vICur = 0;
  slot_info_vec_t vSinfos;
  std::vector<mgs_ubatch> lUbatches;

  uint32_t vNKv;
};

class mgs_kv_cache_dsv41_context : public mgs_memory_context_i {
public:
  using slot_info_vec_t = mgs_kv_cache::slot_info_vec_t;
  using stream_copy_info = mgs_kv_cache::stream_copy_info;

  struct comp_plan {
    // Per-ubatch recipe for updating compressor state, committing completed
    // compressed rows, and masking the compressed attention source.

    // Group slot of every token, i.e. pos % ratio, for the compressor-state updates.
    std::vector<int32_t> lStatePos;

    // Current-ubatch source row ids and unique persistent-state
    // destination row ids for deterministic ring-state updates.
    std::vector<int32_t> lStatePersistSrcIdxs;
    std::vector<int32_t> lStatePersistDstIdxs;

    // Device-side rollback restore copies snapshot planes back to the
    // current compressor-state plane before the graph reads it.
    std::vector<int32_t> lStateRestoreSrcIdxs;
    std::vector<int32_t> lStateRestoreDstIdxs;

    // Device-side rollback snapshots copy rows from the graph-local
    // [persistent_state | current_ubatch_scratch] tensor into rollback
    // planes after the graph has computed current-token compressor state.
    std::vector<int32_t> lStateSnapshotSrcIdxs;
    std::vector<int32_t> lStateSnapshotDstIdxs;

    // Flattened source row ids used for state-backed commits, ratio per committed row. Source rows
    // index the graph-local [persistent_state | current_ubatch_scratch] tensor.
    std::vector<int32_t> lStateReadIdxs;

    // Final compressed-cache row ids written by state-backed commits.
    // A decode step that completes no row targets a masked scratch row.
    std::vector<int64_t> lStateWriteIdxs;

    // RoPE positions for state-backed commits (the first token of each pooled group).
    std::vector<int32_t> lStateWritePos;

    // Number of completed compressed rows visible for each query token.
    std::vector<int32_t> lNVisible;

    // Number of streams used by the attention graph for this ubatch.
    int64_t vNStream = 1;

    // Graph-width for compressed rows. This can be larger than n_visible
    // so masked padding rows do not force a new graph at every row.
    int64_t vNKv = 0;
  };

  // the per-ratio part of a context
  struct group_context {
    std::vector<comp_plan> lPlans;

    mgs_memory_context_ptr vCtxCompMem;
    mgs_memory_context_ptr vCtxIdxMem;

    std::unique_ptr<mgs_kv_cache_dsv41_comp_context> vCtxComp;
    std::unique_ptr<mgs_kv_cache_dsv41_comp_context> vCtxIdx;

    mgs_dsv41_comp_state *vState = nullptr;

    stream_copy_info vScInfo;

    comp_plan vReservePlan;
  };

  mgs_kv_cache_dsv41_context(mgs_memory_status pStatus);

  mgs_kv_cache_dsv41_context(mgs_kv_cache_dsv41 *pKv);

  mgs_kv_cache_dsv41_context(mgs_kv_cache_dsv41 *pKv, mgs_context *pLctx, bool pOptimize,
                             std::vector<stream_copy_info> pScInfos);

  mgs_kv_cache_dsv41_context(mgs_kv_cache_dsv41 *pKv, slot_info_vec_t pSinfosRawBaseWrite,
                             slot_info_vec_t pSinfosRawSwaWrite, slot_info_vec_t pSinfosRawSwaRead,
                             std::vector<mgs_ubatch> pUbatches, std::vector<mgs_ubatch> pUbatchesRaw);

  virtual ~mgs_kv_cache_dsv41_context();

  //
  // mgs_memory_context_i
  //

  bool mNext() override;
  bool mApply() override;

  mgs_memory_status mGetStatus() const override;
  const mgs_ubatch &mGetUbatch() const override;

  //
  // mgs_kv_cache_dsv41_context specific API
  //

  const mgs_kv_cache_dsv41_raw_context *mGetRaw() const;

  uint32_t mGetNGroup() const;
  uint32_t mGetRatio(uint32_t pG) const;

  const mgs_kv_cache_dsv41_comp_context *mGetComp(uint32_t pG) const;
  const mgs_kv_cache_dsv41_comp_context *mGetIdx(uint32_t pG) const;
  const mgs_dsv41_comp_state *mGetState(uint32_t pG) const;

  // the plan of the current ubatch, or a worst-case plan while reserving the graph
  const comp_plan &mGetPlan(uint32_t pG, const mgs_ubatch &pUbatch) const;

private:
  size_t vINext = 0;

  std::vector<mgs_ubatch> lUbatches;

  const mgs_kv_cache_dsv41 *vKv = nullptr;

  const std::unique_ptr<mgs_kv_cache_dsv41_raw_context> cCtxRaw;

  mutable std::vector<group_context> lGroups;

  bool vReservePlans = false;

  mgs_memory_status vStatus = cMgsMemoryStatusSuccess;
};
