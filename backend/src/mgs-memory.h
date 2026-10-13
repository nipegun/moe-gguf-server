#pragma once

#include "mgs.h"
#include "mgs-graph.h"

#include <map>
#include <memory>
#include <functional>

struct mgs_ubatch;

class mgs_batch_allocr;

class mgs_io_write_i;
class mgs_io_read_i;

struct mgs_memory_params {
  // kv cache
  ggml_type vTypeK;
  ggml_type vTypeV;

  // use full-size SWA cache
  bool vSwaFull;

  mgs_context_type vCtxType;

  mgs_memory_t vMemOther;
};

enum mgs_memory_status {
  cMgsMemoryStatusSuccess = 0,
  cMgsMemoryStatusNoUpdate,
  cMgsMemoryStatusFailedPrepare,
  cMgsMemoryStatusFailedCompute,
};

// helper function for combining the status of two memory contexts
// useful for implementing hybrid memory types (e.g. iSWA)
mgs_memory_status mgs_memory_status_combine(mgs_memory_status pS0, mgs_memory_status pS1);

// helper function for checking if a memory status indicates a failure
bool mgs_memory_status_is_fail(mgs_memory_status pStatus);

// the interface for managing the memory context during batch processing
// this interface is implemented per memory type. see:
//   - mgs_kv_cache_context
//   - mgs_kv_cache_iswa_context
//   ...
//
// the only method that should mutate the memory and the memory context is mgs_memory_i::apply()
struct mgs_memory_context_i {
  virtual ~mgs_memory_context_i() = default;

  // consume the current ubatch from the context and proceed to the next one
  // return false if we are done
  virtual bool mNext() = 0;

  // apply the memory state for the current ubatch to the memory object
  // return false on failure
  virtual bool mApply() = 0;

  // get the current ubatch
  virtual const mgs_ubatch &mGetUbatch() const = 0;

  // get the status of the memory context - used for error handling and checking if any updates would be applied
  virtual mgs_memory_status mGetStatus() const = 0;
};

using mgs_memory_context_ptr = std::unique_ptr<mgs_memory_context_i>;

// general concept of LLM memory
// the KV cache is a type of LLM memory, but there can be other types
struct mgs_memory_i {
  // this callback is used to filter out layers that should not be included in the cache
  using layer_filter_cb = std::function<bool(int32_t pIl)>;

  // this callback is used to specify which layers should reuse memory from other layers
  // return negative value to indicate that the layer il should not reuse memory
  using layer_reuse_cb = std::function<int32_t(int32_t pIl)>;

  using layer_share_cb = std::function<int32_t(int32_t pIl)>;

  virtual ~mgs_memory_i() = default;

  // split the input batch into a set of ubatches and verify that they can fit into the cache
  // return a context object containing the ubatches and memory state required to process them
  // check the mgs_memory_context_i::mGetStatus() for the result
  virtual mgs_memory_context_ptr mInitBatch(mgs_batch_allocr &pBalloc, uint32_t pNUbatch, bool pEmbdAll) = 0;

  // simulate full cache, used for allocating worst-case compute buffers
  virtual mgs_memory_context_ptr mInitFull() = 0;

  // prepare for any pending memory updates, such as shifts, copies, etc.
  // status == MGS_MEMORY_STATUS_NO_UPDATE if there is nothing to update
  virtual mgs_memory_context_ptr mInitUpdate(mgs_context *pLctx, bool pOptimize) = 0;

  // getters
  virtual bool mGetCanShift() const = 0;

  //
  // ops
  //

  // if data == true, the data buffers will also be cleared together with the metadata
  virtual void mClear(bool pData) = 0;

  virtual bool mSeqRm(mgs_seq_id pSeqId, mgs_pos p0, mgs_pos p1) = 0;
  virtual void mSeqCp(mgs_seq_id pSeqIdSrc, mgs_seq_id pSeqIdDst, mgs_pos p0, mgs_pos p1) = 0;
  virtual void mSeqKeep(mgs_seq_id pSeqId) = 0;
  virtual void mSeqAdd(mgs_seq_id pSeqId, mgs_pos p0, mgs_pos p1, mgs_pos pShift) = 0;
  virtual void mSeqDiv(mgs_seq_id pSeqId, mgs_pos p0, mgs_pos p1, int pD) = 0;

  virtual mgs_pos mSeqPosMin(mgs_seq_id pSeqId) const = 0;
  virtual mgs_pos mSeqPosMax(mgs_seq_id pSeqId) const = 0;

  virtual std::map<ggml_backend_buffer_type_t, size_t> mMemoryBreakdown() const = 0;

  //
  // state write/read
  //

  virtual void mStateWrite(mgs_io_write_i &pIo, mgs_seq_id pSeqId = -1, mgs_state_seq_flags pFlags = 0) const = 0;
  virtual void mStateRead(mgs_io_read_i &pIo, mgs_seq_id pSeqId = -1, mgs_state_seq_flags pFlags = 0) = 0;
};

using mgs_memory_ptr = std::unique_ptr<mgs_memory_i>;
