#include "mgs-memory-hybrid.h"

#include "mgs-impl.h"
#include "mgs-model.h"
#include "mgs-context.h"

//
// mgs_memory_hybrid
//

mgs_memory_hybrid::mgs_memory_hybrid(const mgs_model &pModel,
                                     /* attn */
                                     ggml_type pTypeK, ggml_type pTypeV, bool pVTrans, uint32_t pKvSize, uint32_t pNPad,
                                     uint32_t pNSwa, mgs_swa_type pSwaType,
                                     /* recurrent */
                                     ggml_type pTypeR, ggml_type pTypeS, uint32_t pRsSize,
                                     /* common */
                                     uint32_t pNSeqMax, uint32_t pNRsSeq, bool pOffload, bool pUnified,
                                     /* layer filters */
                                     const layer_filter_cb &pFilterAttn, const layer_filter_cb &pFilterRecr)
    : vHparams(pModel.vHparams),
      cMemAttn(new mgs_kv_cache(
          pModel, pModel.vHparams, pTypeK, pTypeV, pVTrans, pOffload, pUnified, pKvSize, pNSeqMax, pNPad, pNSwa,
          pSwaType, nullptr, pFilterAttn == nullptr ? [&](int32_t pIl) { return !vHparams.mIsRecr(pIl); } : pFilterAttn,
          nullptr, nullptr)),
      cMemRecr(new mgs_memory_recurrent(pModel, pTypeR, pTypeS, pOffload, pRsSize, pNSeqMax, pNRsSeq,
                                        pFilterRecr == nullptr ? [&](int32_t pIl) { return vHparams.mIsRecr(pIl); }
                                                               : pFilterRecr)) {}

mgs_memory_context_ptr mgs_memory_hybrid::mInitBatch(mgs_batch_allocr &pBalloc, uint32_t pNUbatch, bool pEmbdAll) {
  do {
    pBalloc.mSplitReset();

    // follow the recurrent pattern for creating the ubatch splits
    std::vector<mgs_ubatch> lUbatches;

    while (true) {
      mgs_ubatch vUbatch;

      if (pEmbdAll) {
        // if all tokens are output, split by sequence
        vUbatch = pBalloc.mSplitSeq(pNUbatch);
      } else {
        // Use non-sequential split when KV cache is unified (needed for hellaswag/winogrande/multiple-choice)
        const bool cUnified = (cMemAttn->mGetNStream() == 1);

        // [TAG_RECURRENT_ROLLBACK_SPLITS]
        // the trailing (1 + n_rs_seq) tokens of each seq must stay in the same ubatch
        //   so that the rollback snapshots remain valid
        const uint32_t cNRsSeq = cMemRecr->vNRsSeq;

        vUbatch = pBalloc.mSplitEqual(pNUbatch, !cUnified, cNRsSeq > 0 ? cNRsSeq + 1 : 0);
      }

      if (vUbatch.vNTokens == 0) {
        break;
      }

      lUbatches.push_back(std::move(vUbatch)); // NOLINT
    }

    if (pBalloc.mGetNUsed() < pBalloc.mGetNTokens()) {
      // failed to find a suitable split
      break;
    }

    // prepare the recurrent batches first
    if (!cMemRecr->mPrepare(lUbatches)) {
      // TODO: will the recurrent cache be in an undefined context at this point?
      MGS_LOG_ERROR("%s: failed to prepare recurrent ubatches\n", __func__);
      return std::make_unique<mgs_memory_hybrid_context>(cMgsMemoryStatusFailedPrepare);
    }

    // prepare the attention cache
    auto vHeadsAttn = cMemAttn->mPrepare(lUbatches);
    if (vHeadsAttn.empty()) {
      MGS_LOG_ERROR("%s: failed to prepare attention ubatches\n", __func__);
      return std::make_unique<mgs_memory_hybrid_context>(cMgsMemoryStatusFailedPrepare);
    }

    return std::make_unique<mgs_memory_hybrid_context>(this, std::move(vHeadsAttn), std::move(lUbatches));
  } while (false);

  return std::make_unique<mgs_memory_hybrid_context>(cMgsMemoryStatusFailedPrepare);
}

mgs_memory_context_ptr mgs_memory_hybrid::mInitFull() { return std::make_unique<mgs_memory_hybrid_context>(this); }

mgs_memory_context_ptr mgs_memory_hybrid::mInitUpdate(mgs_context *pLctx, bool pOptimize) {
  return std::make_unique<mgs_memory_hybrid_context>(this, pLctx, pOptimize);
}

bool mgs_memory_hybrid::mGetCanShift() const {
  // Shifting is trivially supported for recurrent
  return cMemAttn->mGetCanShift();
}

void mgs_memory_hybrid::mClear(bool pData) {
  cMemAttn->mClear(pData);
  cMemRecr->mClear(pData);
}

bool mgs_memory_hybrid::mSeqRm(mgs_seq_id pSeqId, mgs_pos p0, mgs_pos p1) {
  // Try removing from the recurrent cache first since it may fail. If it does
  // fail, the cache will not have been mutated.
  if (!cMemRecr->mSeqRm(pSeqId, p0, p1)) {
    return false;
  }
  return cMemAttn->mSeqRm(pSeqId, p0, p1);
}

void mgs_memory_hybrid::mSeqCp(mgs_seq_id pSeqIdSrc, mgs_seq_id pSeqIdDst, mgs_pos p0, mgs_pos p1) {
  cMemAttn->mSeqCp(pSeqIdSrc, pSeqIdDst, p0, p1);
  cMemRecr->mSeqCp(pSeqIdSrc, pSeqIdDst, p0, p1);
}

void mgs_memory_hybrid::mSeqKeep(mgs_seq_id pSeqId) {
  cMemAttn->mSeqKeep(pSeqId);
  cMemRecr->mSeqKeep(pSeqId);
}

void mgs_memory_hybrid::mSeqAdd(mgs_seq_id pSeqId, mgs_pos p0, mgs_pos p1, mgs_pos pShift) {
  cMemAttn->mSeqAdd(pSeqId, p0, p1, pShift);
  cMemRecr->mSeqAdd(pSeqId, p0, p1, pShift);
}

void mgs_memory_hybrid::mSeqDiv(mgs_seq_id pSeqId, mgs_pos p0, mgs_pos p1, int pD) {
  cMemAttn->mSeqDiv(pSeqId, p0, p1, pD);
  cMemRecr->mSeqDiv(pSeqId, p0, p1, pD);
}

mgs_pos mgs_memory_hybrid::mSeqPosMin(mgs_seq_id pSeqId) const {
  // the min of the total cache is the max of the two caches' min values
  return std::max(cMemAttn->mSeqPosMin(pSeqId), cMemRecr->mSeqPosMin(pSeqId));
}

mgs_pos mgs_memory_hybrid::mSeqPosMax(mgs_seq_id pSeqId) const {
  // the max of the total cache is the min of the two caches' max values
  return std::min(cMemAttn->mSeqPosMax(pSeqId), cMemRecr->mSeqPosMax(pSeqId));
}

std::map<ggml_backend_buffer_type_t, size_t> mgs_memory_hybrid::mMemoryBreakdown() const {
  std::map<ggml_backend_buffer_type_t, size_t> dMb = cMemAttn->mMemoryBreakdown();
  for (const auto &vBuftSize : cMemRecr->mMemoryBreakdown()) {
    dMb[vBuftSize.first] += vBuftSize.second;
  }
  return dMb;
}

void mgs_memory_hybrid::mStateWrite(mgs_io_write_i &pIo, mgs_seq_id pSeqId, mgs_state_seq_flags pFlags) const {
  if ((pFlags & MGS_STATE_SEQ_FLAGS_PARTIAL_ONLY) == 0) {
    cMemAttn->mStateWrite(pIo, pSeqId, pFlags);
  }
  cMemRecr->mStateWrite(pIo, pSeqId, pFlags);
}

void mgs_memory_hybrid::mStateRead(mgs_io_read_i &pIo, mgs_seq_id pSeqId, mgs_state_seq_flags pFlags) {
  if ((pFlags & MGS_STATE_SEQ_FLAGS_PARTIAL_ONLY) == 0) {
    cMemAttn->mStateRead(pIo, pSeqId, pFlags);
  }
  cMemRecr->mStateRead(pIo, pSeqId, pFlags);
}

mgs_kv_cache *mgs_memory_hybrid::mGetMemAttn() const { return cMemAttn.get(); }

mgs_memory_recurrent *mgs_memory_hybrid::mGetMemRecr() const { return cMemRecr.get(); }

mgs_memory_hybrid_context::mgs_memory_hybrid_context(mgs_memory_status pStatus) : cStatus(pStatus) {}

mgs_memory_hybrid_context::mgs_memory_hybrid_context(mgs_memory_hybrid *pMem)
    : cCtxAttn(pMem->mGetMemAttn()->mInitFull()), cCtxRecr(pMem->mGetMemRecr()->mInitFull()),
      cStatus(mgs_memory_status_combine(cCtxAttn->mGetStatus(), cCtxRecr->mGetStatus())) {}

mgs_memory_hybrid_context::mgs_memory_hybrid_context(mgs_memory_hybrid *pMem, mgs_context *pLctx, bool pOptimize)
    : cCtxAttn(pMem->mGetMemAttn()->mInitUpdate(pLctx, pOptimize)),
      cCtxRecr(pMem->mGetMemRecr()->mInitUpdate(pLctx, pOptimize)),
      cStatus(mgs_memory_status_combine(cCtxAttn->mGetStatus(), cCtxRecr->mGetStatus())) {}

mgs_memory_hybrid_context::mgs_memory_hybrid_context(mgs_memory_hybrid *pMem, slot_info_vec_t pSinfosAttn,
                                                     std::vector<mgs_ubatch> pUbatches)
    : lUbatches(std::move(pUbatches)),
      // note: here we copy the ubatches. not sure if this is ideal
      cCtxAttn(new mgs_kv_cache_context(pMem->mGetMemAttn(), std::move(pSinfosAttn), this->lUbatches)),
      cCtxRecr(new mgs_memory_recurrent_context(pMem->mGetMemRecr(), this->lUbatches)),
      cStatus(mgs_memory_status_combine(cCtxAttn->mGetStatus(), cCtxRecr->mGetStatus())) {}

bool mgs_memory_hybrid_context::mNext() {
  assert(cStatus == cMgsMemoryStatusSuccess);

  cCtxAttn->mNext();
  cCtxRecr->mNext();

  if (++vINext >= lUbatches.size()) {
    return false;
  }

  return true;
}

bool mgs_memory_hybrid_context::mApply() {
  assert(!mgs_memory_status_is_fail(cStatus));

  bool vRes = true;

  vRes = vRes & cCtxAttn->mApply();
  vRes = vRes & cCtxRecr->mApply();

  return vRes;
}

mgs_memory_status mgs_memory_hybrid_context::mGetStatus() const { return cStatus; }

const mgs_ubatch &mgs_memory_hybrid_context::mGetUbatch() const {
  assert(cStatus == cMgsMemoryStatusSuccess);
  return lUbatches[vINext];
}

const mgs_kv_cache_context *mgs_memory_hybrid_context::mGetAttn() const {
  return static_cast<const mgs_kv_cache_context *>(cCtxAttn.get());
}

const mgs_memory_recurrent_context *mgs_memory_hybrid_context::mGetRecr() const {
  return static_cast<const mgs_memory_recurrent_context *>(cCtxRecr.get());
}
