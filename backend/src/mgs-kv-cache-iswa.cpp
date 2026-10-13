#include "mgs-kv-cache-iswa.h"

#include "mgs-impl.h"
#include "mgs-batch.h"
#include "mgs-model.h"

#include <algorithm>
#include <cassert>

//
// mgs_kv_cache_iswa
//

mgs_kv_cache_iswa::mgs_kv_cache_iswa(const mgs_model &pModel, ggml_type pTypeK, ggml_type pTypeV, bool pVTrans,
                                     bool pOffload, bool pSwaFull, bool pUnified, uint32_t pKvSize, uint32_t pNSeqMax,
                                     uint32_t pNUbatch, uint32_t pNPad, mgs_memory_t pMemOther,
                                     const layer_filter_cb &pFilter, const layer_reuse_cb &pReuse,
                                     const layer_share_cb &pShare)
    : mgs_kv_cache_iswa(pModel, pModel.vHparams, pTypeK, pTypeV, pVTrans, pOffload, pSwaFull, pUnified, pKvSize,
                        pNSeqMax, pNUbatch, pNPad, pMemOther, pFilter, pReuse, pShare) {}

mgs_kv_cache_iswa::mgs_kv_cache_iswa(const mgs_model &pModel, const mgs_hparams &pHparams, ggml_type pTypeK,
                                     ggml_type pTypeV, bool pVTrans, bool pOffload, bool pSwaFull, bool pUnified,
                                     uint32_t pKvSize, uint32_t pNSeqMax, uint32_t pNUbatch, uint32_t pNPad,
                                     mgs_memory_t pMemOther, const layer_filter_cb &pFilter,
                                     const layer_reuse_cb &pReuse, const layer_share_cb &pShare)
    : cUnified(pUnified) {

  // chain filters
  const layer_filter_cb cFilterBase = [&](int32_t pIl) {
    if (pFilter && !pFilter(pIl)) {
      return false;
    }

    return !pModel.vHparams.mIsSwa(pIl);
  };

  const layer_filter_cb cFilterSwa = [&](int32_t pIl) {
    if (pFilter && !pFilter(pIl)) {
      return false;
    }

    return pModel.vHparams.mIsSwa(pIl);
  };

  const uint32_t cSizeBase = pKvSize;

  // note: the SWA cache is always padded to 256 for performance
  uint32_t vSizeSwa = GGML_PAD(std::min(cSizeBase, pHparams.vNSwa * (pUnified ? pNSeqMax : 1) + pNUbatch), 256);

  // when using full-size SWA cache, we set the SWA cache size to be equal to the base cache size
  if (pSwaFull) {
    MGS_LOG_WARN("%s: using full-size SWA cache\n", __func__);

    vSizeSwa = cSizeBase;
  }

  MGS_LOG_INFO("%s: creating non-SWA KV cache, size = %u cells\n", __func__, cSizeBase);

  mgs_memory_t vMemOtherBase = nullptr;
  if (pMemOther) {
    vMemOtherBase = static_cast<mgs_kv_cache_iswa *>(pMemOther)->mGetBase();
  }

  mgs_memory_t vMemOtherSwa = nullptr;
  if (pMemOther) {
    vMemOtherSwa = static_cast<mgs_kv_cache_iswa *>(pMemOther)->mGetSwa();
  }

  vKvBase =
      std::make_unique<mgs_kv_cache>(pModel, pHparams, pTypeK, pTypeV, pVTrans, pOffload, pUnified, cSizeBase, pNSeqMax,
                                     pNPad, 0, cMgsSwaTypeNone, vMemOtherBase, cFilterBase, pReuse, pShare);

  MGS_LOG_INFO("%s: creating     SWA KV cache, size = %u cells\n", __func__, vSizeSwa);

  vKvSwa = std::make_unique<mgs_kv_cache>(pModel, pHparams, pTypeK, pTypeV, pVTrans, pOffload, pUnified, vSizeSwa,
                                          pNSeqMax, pNPad, pHparams.vNSwa, pHparams.vSwaType, vMemOtherSwa, cFilterSwa,
                                          pReuse, pShare);
}

void mgs_kv_cache_iswa::mClear(bool pData) {
  vKvBase->mClear(pData);
  vKvSwa->mClear(pData);
}

bool mgs_kv_cache_iswa::mSeqRm(mgs_seq_id pSeqId, mgs_pos p0, mgs_pos p1) {
  bool vRes = true;

  vRes = vRes & vKvBase->mSeqRm(pSeqId, p0, p1);
  vRes = vRes & vKvSwa->mSeqRm(pSeqId, p0, p1);

  return vRes;
}

void mgs_kv_cache_iswa::mSeqCp(mgs_seq_id pSeqIdSrc, mgs_seq_id pSeqIdDst, mgs_pos p0, mgs_pos p1) {
  vKvBase->mSeqCp(pSeqIdSrc, pSeqIdDst, p0, p1);
  vKvSwa->mSeqCp(pSeqIdSrc, pSeqIdDst, p0, p1);
}

void mgs_kv_cache_iswa::mSeqKeep(mgs_seq_id pSeqId) {
  vKvBase->mSeqKeep(pSeqId);
  vKvSwa->mSeqKeep(pSeqId);
}

void mgs_kv_cache_iswa::mSeqAdd(mgs_seq_id pSeqId, mgs_pos p0, mgs_pos p1, mgs_pos pShift) {
  vKvBase->mSeqAdd(pSeqId, p0, p1, pShift);
  vKvSwa->mSeqAdd(pSeqId, p0, p1, pShift);
}

void mgs_kv_cache_iswa::mSeqDiv(mgs_seq_id pSeqId, mgs_pos p0, mgs_pos p1, int pD) {
  vKvBase->mSeqDiv(pSeqId, p0, p1, pD);
  vKvSwa->mSeqDiv(pSeqId, p0, p1, pD);
}

mgs_pos mgs_kv_cache_iswa::mSeqPosMin(mgs_seq_id pSeqId) const {
  // the base cache is a superset of the SWA cache, so we can just check the SWA cache
  return vKvSwa->mSeqPosMin(pSeqId);
}

mgs_pos mgs_kv_cache_iswa::mSeqPosMax(mgs_seq_id pSeqId) const { return vKvSwa->mSeqPosMax(pSeqId); }

std::map<ggml_backend_buffer_type_t, size_t> mgs_kv_cache_iswa::mMemoryBreakdown() const {
  std::map<ggml_backend_buffer_type_t, size_t> dMb = vKvBase->mMemoryBreakdown();
  for (const auto &vBuftSize : vKvSwa->mMemoryBreakdown()) {
    dMb[vBuftSize.first] += vBuftSize.second;
  }
  return dMb;
}

mgs_memory_context_ptr mgs_kv_cache_iswa::mInitBatch(mgs_batch_allocr &pBalloc, uint32_t pNUbatch, bool pEmbdAll) {
  GGML_UNUSED(pEmbdAll);

  // first try simple split
  do {
    if (!cUnified) {
      // requires equal splits, so we skip the simple split
      break;
    }

    pBalloc.mSplitReset();

    std::vector<mgs_ubatch> lUbatches;
    while (true) {
      auto vUbatch = pBalloc.mSplitSimple(pNUbatch);

      if (vUbatch.vNTokens == 0) {
        break;
      }

      lUbatches.push_back(std::move(vUbatch)); // NOLINT
    }

    if (pBalloc.mGetNUsed() < pBalloc.mGetNTokens()) {
      // failed to find a suitable split
      break;
    }

    auto vSinfosBase = vKvBase->mPrepare(lUbatches);
    if (vSinfosBase.empty()) {
      break;
    }

    auto vSinfosSwa = vKvSwa->mPrepare(lUbatches);
    if (vSinfosSwa.empty()) {
      break;
    }

    assert(vSinfosBase.size() == vSinfosSwa.size());

    return std::make_unique<mgs_kv_cache_iswa_context>(this, std::move(vSinfosBase), std::move(vSinfosSwa),
                                                       std::move(lUbatches));
  } while (false);

  // if it fails, try equal split
  do {
    pBalloc.mSplitReset();

    std::vector<mgs_ubatch> lUbatches;
    while (true) {
      auto vUbatch = pBalloc.mSplitEqual(pNUbatch, !cUnified, 0);

      if (vUbatch.vNTokens == 0) {
        break;
      }

      lUbatches.push_back(std::move(vUbatch)); // NOLINT
    }

    if (pBalloc.mGetNUsed() < pBalloc.mGetNTokens()) {
      // failed to find a suitable split
      break;
    }

    auto vSinfosBase = vKvBase->mPrepare(lUbatches);
    if (vSinfosBase.empty()) {
      break;
    }

    auto vSinfosSwa = vKvSwa->mPrepare(lUbatches);
    if (vSinfosSwa.empty()) {
      break;
    }

    assert(vSinfosBase.size() == vSinfosSwa.size());

    return std::make_unique<mgs_kv_cache_iswa_context>(this, std::move(vSinfosBase), std::move(vSinfosSwa),
                                                       std::move(lUbatches));
  } while (false);

  // TODO: if we fail again, we should attempt different splitting strategies
  //       but to do that properly, we first have to refactor the batches to be more flexible

  return std::make_unique<mgs_kv_cache_iswa_context>(cMgsMemoryStatusFailedPrepare);
}

mgs_memory_context_ptr mgs_kv_cache_iswa::mInitFull() { return std::make_unique<mgs_kv_cache_iswa_context>(this); }

mgs_memory_context_ptr mgs_kv_cache_iswa::mInitUpdate(mgs_context *pLctx, bool pOptimize) {
  return std::make_unique<mgs_kv_cache_iswa_context>(this, pLctx, pOptimize);
}

bool mgs_kv_cache_iswa::mGetCanShift() const {
  return vKvBase->mGetCanShift() && vKvSwa->mGetCanShift() && vKvBase->mGetSize() == vKvSwa->mGetSize();
}

void mgs_kv_cache_iswa::mStateWrite(mgs_io_write_i &pIo, mgs_seq_id pSeqId, mgs_state_seq_flags pFlags) const {
  if ((pFlags & MGS_STATE_SEQ_FLAGS_PARTIAL_ONLY) == 0) {
    vKvBase->mStateWrite(pIo, pSeqId, pFlags);
  }

  vKvSwa->mStateWrite(pIo, pSeqId, pFlags);
}

void mgs_kv_cache_iswa::mStateRead(mgs_io_read_i &pIo, mgs_seq_id pSeqId, mgs_state_seq_flags pFlags) {
  if ((pFlags & MGS_STATE_SEQ_FLAGS_PARTIAL_ONLY) == 0) {
    vKvBase->mStateRead(pIo, pSeqId, pFlags);
  }

  vKvSwa->mStateRead(pIo, pSeqId, pFlags);
}

mgs_kv_cache *mgs_kv_cache_iswa::mGetBase() const { return vKvBase.get(); }

mgs_kv_cache *mgs_kv_cache_iswa::mGetSwa() const { return vKvSwa.get(); }

//
// mgs_kv_cache_iswa_context
//

mgs_kv_cache_iswa_context::mgs_kv_cache_iswa_context(mgs_memory_status pStatus) : cStatus(pStatus) {}

mgs_kv_cache_iswa_context::mgs_kv_cache_iswa_context(mgs_kv_cache_iswa *pKv)
    : cCtxBase(pKv->mGetBase()->mInitFull()), cCtxSwa(pKv->mGetSwa()->mInitFull()),
      cStatus(mgs_memory_status_combine(cCtxBase->mGetStatus(), cCtxSwa->mGetStatus())) {}

mgs_kv_cache_iswa_context::mgs_kv_cache_iswa_context(mgs_kv_cache_iswa *pKv, mgs_context *pLctx, bool pOptimize)
    : cCtxBase(pKv->mGetBase()->mInitUpdate(pLctx, pOptimize)), cCtxSwa(pKv->mGetSwa()->mInitUpdate(pLctx, pOptimize)),
      cStatus(mgs_memory_status_combine(cCtxBase->mGetStatus(), cCtxSwa->mGetStatus())) {}

mgs_kv_cache_iswa_context::mgs_kv_cache_iswa_context(mgs_kv_cache_iswa *pKv, slot_info_vec_t pSinfosBase,
                                                     slot_info_vec_t pSinfosSwa, std::vector<mgs_ubatch> pUbatches)
    : lUbatches(std::move(pUbatches)),
      // note: here we copy the ubatches. not sure if this is ideal
      cCtxBase(new mgs_kv_cache_context(pKv->mGetBase(), std::move(pSinfosBase), this->lUbatches)),
      cCtxSwa(new mgs_kv_cache_context(pKv->mGetSwa(), std::move(pSinfosSwa), this->lUbatches)),
      cStatus(mgs_memory_status_combine(cCtxBase->mGetStatus(), cCtxSwa->mGetStatus())) {}

mgs_kv_cache_iswa_context::~mgs_kv_cache_iswa_context() = default;

bool mgs_kv_cache_iswa_context::mNext() {
  assert(cStatus == cMgsMemoryStatusSuccess);

  cCtxBase->mNext();
  cCtxSwa->mNext();

  if (++vINext >= lUbatches.size()) {
    return false;
  }

  return true;
}

bool mgs_kv_cache_iswa_context::mApply() {
  assert(!mgs_memory_status_is_fail(cStatus));

  bool vRes = true;

  vRes = vRes & cCtxBase->mApply();
  vRes = vRes & cCtxSwa->mApply();

  return vRes;
}

mgs_memory_status mgs_kv_cache_iswa_context::mGetStatus() const { return cStatus; }

const mgs_ubatch &mgs_kv_cache_iswa_context::mGetUbatch() const {
  assert(cStatus == cMgsMemoryStatusSuccess);

  return lUbatches[vINext];
}

const mgs_kv_cache_context *mgs_kv_cache_iswa_context::mGetBase() const {
  assert(cStatus == cMgsMemoryStatusSuccess);

  return static_cast<const mgs_kv_cache_context *>(cCtxBase.get());
}

const mgs_kv_cache_context *mgs_kv_cache_iswa_context::mGetSwa() const {
  assert(cStatus == cMgsMemoryStatusSuccess);

  return static_cast<const mgs_kv_cache_context *>(cCtxSwa.get());
}
