#pragma once

#include "mgs-kv-cache.h"

#include <vector>

//
// mgs_kv_cache_dsa
//

// utilizes two instances of mgs_kv_cache:
// - the first instance is for caching key tensors of the model,
// - the second instance is for caching lightning indexer key tensors

class mgs_kv_cache_dsa : public mgs_memory_i {
public:
  mgs_kv_cache_dsa(const mgs_model &pModel, ggml_type pTypeK, ggml_type pTypeV, bool pVTrans, bool pOffload,
                   bool pUnified, uint32_t pKvSize, uint32_t pNSeqMax, uint32_t pNPad, uint32_t pNSwa,
                   mgs_swa_type pSwaType, const layer_filter_cb &pFilterMla, const layer_filter_cb &pFilterLid,
                   const layer_reuse_cb &pReuse);

  ~mgs_kv_cache_dsa() = default;

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

  // state write/load

  void mStateWrite(mgs_io_write_i &pIo, mgs_seq_id pSeqId = -1, mgs_state_seq_flags pFlags = 0) const override;
  void mStateRead(mgs_io_read_i &pIo, mgs_seq_id pSeqId = -1, mgs_state_seq_flags pFlags = 0) override;

  //
  // mgs_kv_cache_dsa specific API
  //

  mgs_kv_cache *mGetMla() const;
  mgs_kv_cache *mGetLid() const;

private:
  // we keep indexer KV cache hparams instance here as mgs_kv_cache stores only reference to it
  mgs_hparams vHparamsLid;
  const uint32_t cNStream = 1;

  std::unique_ptr<mgs_kv_cache> vKvMla;
  std::unique_ptr<mgs_kv_cache> vKvLid;
};

class mgs_kv_cache_dsa_context : public mgs_memory_context_i {
public:
  using slot_info_vec_t = mgs_kv_cache::slot_info_vec_t;

  // used for errors
  mgs_kv_cache_dsa_context(mgs_memory_status pStatus);

  // used to create a full-cache context
  mgs_kv_cache_dsa_context(mgs_kv_cache_dsa *pKv);

  // used to create an update context
  mgs_kv_cache_dsa_context(mgs_kv_cache_dsa *pKv, mgs_context *pLctx, bool pOptimize);

  // used to create a batch processing context from a batch
  mgs_kv_cache_dsa_context(mgs_kv_cache_dsa *pKv, slot_info_vec_t pSinfosBase, slot_info_vec_t pSinfosIk,
                           std::vector<mgs_ubatch> pUbatches);

  virtual ~mgs_kv_cache_dsa_context();

  //
  // mgs_memory_context_i
  //

  bool mNext() override;
  bool mApply() override;

  mgs_memory_status mGetStatus() const override;
  const mgs_ubatch &mGetUbatch() const override;

  //
  // mgs_kv_cache_dsa_context specific API
  //

  const mgs_kv_cache_context *mGetMla() const;
  const mgs_kv_cache_context *mGetLid() const;

private:
  // mgs_kv_cache_dsa * vKv;

  // the index of the next ubatch to process
  size_t vINext = 0;

  std::vector<mgs_ubatch> lUbatches;

  const mgs_memory_context_ptr cCtxMla;
  const mgs_memory_context_ptr cCtxLid;

  const mgs_memory_status cStatus;
};
