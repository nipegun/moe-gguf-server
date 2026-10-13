#pragma once

#include "mgs-batch.h"
#include "mgs-graph.h"
#include "mgs-memory.h"

#include <map>
#include <set>
#include <vector>

//
// mgs_memory_recurrent
//

// TODO: extract the cache state used for graph computation into mgs_memory_recurrent_context_i
//       see the implementation of mgs_kv_cache_context_i for an example how to do it
class mgs_memory_recurrent : public mgs_memory_i {
public:
  mgs_memory_recurrent(const mgs_model &pModel, ggml_type pTypeR, ggml_type pTypeS, bool pOffload, uint32_t pMemSize,
                       uint32_t pNSeqMax, uint32_t pNRsSeq, const layer_filter_cb &pFilter);

  ~mgs_memory_recurrent() = default;

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

  mgs_pos mSeqPosMin(mgs_seq_id pSeqId) const override;
  mgs_pos mSeqPosMax(mgs_seq_id pSeqId) const override;

  std::map<ggml_backend_buffer_type_t, size_t> mMemoryBreakdown() const override;

  bool mPrepare(const std::vector<mgs_ubatch> &pUbatches);

  // find a contiguous slot of memory cells and emplace the ubatch there
  bool mFindSlot(const mgs_ubatch &pUbatch);

  bool mGetCanShift() const override;

  // state write/load

  void mStateWrite(mgs_io_write_i &pIo, mgs_seq_id pSeqId = -1, mgs_state_seq_flags pFlags = 0) const override;
  void mStateRead(mgs_io_read_i &pIo, mgs_seq_id pSeqId = -1, mgs_state_seq_flags pFlags = 0) override;

  uint32_t vHead = 0; // the location where the batch will be placed in the cache (see mFindSlot())
  uint32_t vSize = 0; // total number of cells, shared across all sequences
  uint32_t vUsed = 0; // used cells (i.e. at least one seq_id)

  // number of recurrent-state snapshots per seq for rollback; tensors are widened to (1 + n_rs_seq) groups
  uint32_t vNRsSeq = 0;

  // per-seq rollback index
  std::vector<uint32_t> lRsIdx;

  void mSetRsIdx(mgs_seq_id pSeqId, uint32_t pIdx);

  // computed before each graph build
  uint32_t vN = 0;

  // first zero-ed state
  int32_t vRsZ = -1;

  // TODO: optimize for recurrent state needs
  struct mem_cell {
    mgs_pos vPos = -1;
    int32_t vSrc = -1;  // used to know where states should be copied from
    int32_t vSrc0 = -1; // like src, but only used when setting the inputs (allowing to copy once)
    int32_t vTail = -1;

    std::set<mgs_seq_id> sSeqId;

    bool mHasSeqId(const mgs_seq_id &pId) const { return sSeqId.find(pId) != sSeqId.end(); }

    bool mIsEmpty() const { return sSeqId.empty(); }

    bool mIsSameSeq(const mem_cell &pOther) const { return sSeqId == pOther.sSeqId; }
  };

  std::vector<mem_cell> lCells;

  // per layer
  std::vector<ggml_tensor *> lRL;
  std::vector<ggml_tensor *> lSL;
  // a second conv history (qwen4exp PLE) that must stay replicated across devices, so it cannot share the r row
  std::vector<ggml_tensor *> lPL;

private:
  // const mgs_model & vModel;
  const mgs_hparams &vHparams;

  const uint32_t cNSeqMax = 1;

  // ggml contexts for the KV cache along with the allocated backend buffers:
  std::vector<std::pair<ggml_context_ptr, ggml_backend_buffer_ptr>> lCtxsBufs;

  // true if no layers - can happen if the layer filter removes all layers
  bool mIsEmpty() const;

  size_t mTotalSize() const;

  size_t mSizeRBytes() const;
  size_t mSizeSBytes() const;
  size_t mSizePBytes() const;

  void mStateWriteMeta(mgs_io_write_i &pIo, const std::vector<std::pair<uint32_t, uint32_t>> &pCellRanges,
                       mgs_seq_id pSeqId = -1) const;
  void mStateWriteData(mgs_io_write_i &pIo, const std::vector<std::pair<uint32_t, uint32_t>> &pCellRanges) const;

  bool mStateReadMeta(mgs_io_read_i &pIo, uint32_t pCellCount, mgs_seq_id pDestSeqId = -1);
  bool mStateReadData(mgs_io_read_i &pIo, uint32_t pCellCount);
};

class mgs_memory_recurrent_context : public mgs_memory_context_i {
public:
  // used for errors
  mgs_memory_recurrent_context(mgs_memory_status pStatus);

  // used to create a full-cache or update context
  mgs_memory_recurrent_context(mgs_memory_recurrent *pMem);

  // used to create a batch processing context from a batch
  mgs_memory_recurrent_context(mgs_memory_recurrent *pMem, std::vector<mgs_ubatch> pUbatches);

  virtual ~mgs_memory_recurrent_context();

  //
  // mgs_memory_context_i
  //

  bool mNext() override;
  bool mApply() override;

  mgs_memory_status mGetStatus() const override;
  const mgs_ubatch &mGetUbatch() const override;

  //
  // mgs_memory_recurrent_context specific API
  //

  uint32_t mGetNRs() const;
  uint32_t mGetHead() const;
  int32_t mGetRsZ() const;
  uint32_t mGetSize() const;

  ggml_tensor *mGetRL(int32_t pIl) const;
  ggml_tensor *mGetSL(int32_t pIl) const;
  ggml_tensor *mGetPL(int32_t pIl) const;

  int32_t mSCopy(int pI) const;

private:
  const mgs_memory_status cStatus;

  mgs_memory_recurrent *vMem;

  size_t vINext = 0;

  std::vector<mgs_ubatch> lUbatches;

  //
  // data needed for building the compute graph for the current ubatch:
  // TODO: extract all the state like `vHead` and `vN` here
  //

  const bool cIsFull = false;
};
