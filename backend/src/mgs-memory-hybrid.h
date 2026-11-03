#pragma once

#include "mgs-batch.h"
#include "mgs-graph.h"
#include "mgs-kv-cache.h"
#include "mgs-memory.h"
#include "mgs-memory-recurrent.h"

#include <memory>
#include <vector>

//
// mgs_memory_hybrid
//

// utilizes instances of mgs_memory_recurrent and mgs_kv_cache to
//   support models where each layer may be either attention-based or recurrent

class mgs_memory_hybrid : public mgs_memory_i {
public:
  mgs_memory_hybrid(const mgs_model &pModel,
                    /* attn */
                    ggml_type pTypeK, ggml_type pTypeV, bool pVTrans, uint32_t pKvSize, uint32_t pNPad, uint32_t pNSwa,
                    mgs_swa_type pSwaType,
                    /* recurrent */
                    ggml_type pTypeR, ggml_type pTypeS, uint32_t pRsSize,
                    /* common */
                    uint32_t pNSeqMax, uint32_t pNRsSeq, bool pOffload, bool pUnified,
                    /* layer filters */
                    const layer_filter_cb &pFilterAttn = nullptr, const layer_filter_cb &pFilterRecr = nullptr);

  ~mgs_memory_hybrid() = default;

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
  // mgs_memory_hybrid specific API
  //

  mgs_kv_cache *mGetMemAttn() const;
  mgs_memory_recurrent *mGetMemRecr() const;

private:
  const mgs_hparams &vHparams;

  const std::unique_ptr<mgs_kv_cache> cMemAttn;
  const std::unique_ptr<mgs_memory_recurrent> cMemRecr;
};

class mgs_memory_hybrid_context : public mgs_memory_context_i {
public:
  using slot_info_vec_t = mgs_kv_cache::slot_info_vec_t;

  // init failure
  explicit mgs_memory_hybrid_context(mgs_memory_status pStatus);

  // init full
  explicit mgs_memory_hybrid_context(mgs_memory_hybrid *pMem);

  // init update
  explicit mgs_memory_hybrid_context(mgs_memory_hybrid *pMem, mgs_context *pLctx, bool pOptimize);

  // init success
  mgs_memory_hybrid_context(mgs_memory_hybrid *pMem, slot_info_vec_t pSinfosAttn, std::vector<mgs_ubatch> pUbatches);

  ~mgs_memory_hybrid_context() = default;

  bool mNext() override;
  bool mApply() override;

  mgs_memory_status mGetStatus() const override;
  const mgs_ubatch &mGetUbatch() const override;

  //
  // mgs_memory_hybrid_context
  //

  const mgs_kv_cache_context *mGetAttn() const;
  const mgs_memory_recurrent_context *mGetRecr() const;

private:
  // the index of the next ubatch to process
  size_t vINext = 0;

  std::vector<mgs_ubatch> lUbatches;

  const mgs_memory_context_ptr cCtxAttn;
  const mgs_memory_context_ptr cCtxRecr;

  const mgs_memory_status cStatus;
};
