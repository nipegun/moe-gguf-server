#pragma once

#include "mgs-kv-cache.h"

#include <vector>

// mgs_kv_cache_msa

// uses two instances of mgs_kv_cache, one for K/V tensors, and one for the MSA indexer tensors
// both receive identical sequence operations and identical ubatches, so their cell layouts stay in synced.
// the context also exposes per-ubatch pos - cell translation maps populated from mgs_kv_cells via
// mgs_kv_cache::mGetCells(), which the model graph uses to run MSA block selection in position space

class mgs_kv_cache_msa : public mgs_memory_i {
public:
  mgs_kv_cache_msa(const mgs_model &pModel, ggml_type pTypeK, ggml_type pTypeV, bool pVTrans, bool pOffload,
                   bool pUnified, uint32_t pKvSize, uint32_t pNSeqMax, uint32_t pNPad, uint32_t pNSwa,
                   mgs_swa_type pSwaType, const layer_filter_cb &pFilter, const layer_filter_cb &pFilterIdx,
                   const layer_reuse_cb &pReuse);

  ~mgs_kv_cache_msa() = default;

  // mgs_memory_i

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

  // state write/load

  void mStateWrite(mgs_io_write_i &pIo, mgs_seq_id pSeqId = -1, mgs_state_seq_flags pFlags = 0) const override;
  void mStateRead(mgs_io_read_i &pIo, mgs_seq_id pSeqId = -1, mgs_state_seq_flags pFlags = 0) override;

  // mgs_kv_cache_msa specific API

  mgs_kv_cache *mGetBase() const;
  mgs_kv_cache *mGetIdx() const;

  uint32_t mGetNPad() const { return cNPad; }
  uint32_t mGetNSeqMax() const { return cNSeqMax; }
  uint32_t mGetNSwa() const { return cNSwa; }
  mgs_swa_type mGetSwaType() const { return cSwaType; }

private:
  // keep the indexer KV cache hparams instance here as mgs_kv_cache stores only a reference
  mgs_hparams vHparamsIdx;

  const uint32_t cNStream = 1;
  const uint32_t cNSeqMax = 1;
  const uint32_t cNPad = 1;

  const uint32_t cNSwa = 0;
  const mgs_swa_type cSwaType = cMgsSwaTypeNone;

  std::unique_ptr<mgs_kv_cache> vKvBase;
  std::unique_ptr<mgs_kv_cache> vKvIdx;
};

class mgs_kv_cache_msa_context : public mgs_memory_context_i {
public:
  using slot_info_vec_t = mgs_kv_cache::slot_info_vec_t;

  // used for errors
  mgs_kv_cache_msa_context(mgs_memory_status pStatus);

  // used to create a full-cache context
  mgs_kv_cache_msa_context(mgs_kv_cache_msa *pKv);

  // used to create an update context
  mgs_kv_cache_msa_context(mgs_kv_cache_msa *pKv, mgs_context *pLctx, bool pOptimize);

  // used to create a batch processing context from a batch
  mgs_kv_cache_msa_context(mgs_kv_cache_msa *pKv, slot_info_vec_t pSinfosBase, slot_info_vec_t pSinfosIdx,
                           std::vector<mgs_ubatch> pUbatches);

  virtual ~mgs_kv_cache_msa_context();

  // mgs_memory_context_i

  bool mNext() override;
  bool mApply() override;

  mgs_memory_status mGetStatus() const override;
  const mgs_ubatch &mGetUbatch() const override;

  // mgs_kv_cache_msa_context specific API

  const mgs_kv_cache_context *mGetBase() const;
  const mgs_kv_cache_context *mGetIdx() const;

  // max position currently present in the cache plus one, padded MSA blocks are defined over token positions
  // so the block-selection tensors are sized by this value rather than by the number of cells
  uint32_t mGetNPos() const;

  // position <-> cell translation maps, populated from the base cache cells
  // the model graph relates cache contents to token positions only through these per ubatch inputs
  // value for empty or other-sequence cells is 0 so consumers must mask them
  void mSetInputCellPos(ggml_tensor *pDst, const mgs_ubatch *pUbatch, int32_t pDiv) const;
  // positions without a cell map to cell 0, consumers must mask them assumes one sequence per stream
  void mSetInputPosSlot(ggml_tensor *pDst, const mgs_ubatch *pUbatch) const;
  void mSetInputPosMask(ggml_tensor *pDst, const mgs_ubatch *pUbatch) const;

private:
  mgs_kv_cache_msa *vKv;

  // the index of the next ubatch to process
  size_t vINext = 0;

  std::vector<mgs_ubatch> lUbatches;

  const mgs_memory_context_ptr cCtxBase;
  const mgs_memory_context_ptr cCtxIdx;

  const mgs_memory_status cStatus;
};
