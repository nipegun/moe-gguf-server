#include "mgs-kv-cache-dsa.h"

#include "mgs-impl.h"
#include "mgs-batch.h"
#include "mgs-model.h"

#include <algorithm>
#include <cassert>

//
// mgs_kv_cache_dsa
//

mgs_kv_cache_dsa::mgs_kv_cache_dsa(const mgs_model &pModel, ggml_type pTypeK, ggml_type pTypeV, bool pVTrans,
                                   bool pOffload, bool pUnified, uint32_t pKvSize, uint32_t pNSeqMax, uint32_t pNPad,
                                   uint32_t pNSwa, mgs_swa_type pSwaType, const layer_filter_cb &pFilterMla,
                                   const layer_filter_cb &pFilterLid, const layer_reuse_cb &pReuse)
    : vHparamsLid(pModel.vHparams), cNStream(pUnified ? 1 : pNSeqMax) {

  MGS_LOG_INFO("%s: creating main KV cache, size = %u cells\n", __func__, pKvSize);

  vKvMla = std::make_unique<mgs_kv_cache>(pModel, pModel.vHparams, pTypeK, pTypeV, pVTrans, pOffload, pUnified, pKvSize,
                                          pNSeqMax, pNPad, pNSwa, pSwaType, nullptr, pFilterMla, pReuse, nullptr);

  // we use mgs_kv_cache for caching indexer keys
  // by hand-tweaking some hparams we fool it to create
  // indexer key cache tensors with correct dimensions

  // DSA lightning indexer uses MQA with single key head
  std::fill(vHparamsLid.aNHeadKvArr.begin(), vHparamsLid.aNHeadKvArr.end(), 1);
  vHparamsLid.vNEmbdHeadKFull = pModel.vHparams.vIndexerHeadSize;
  vHparamsLid.vRopeType = MGS_ROPE_TYPE_NEOX;

  MGS_LOG_INFO("%s: creating indexer KV cache, size = %u cells\n", __func__, pKvSize);

  vKvLid = std::make_unique<mgs_kv_cache>(pModel, vHparamsLid, pTypeK, pTypeV, pVTrans, pOffload, pUnified, pKvSize,
                                          pNSeqMax, pNPad, pNSwa, pSwaType, nullptr, pFilterLid, pReuse, nullptr);
}

void mgs_kv_cache_dsa::mClear(bool pData) {
  vKvMla->mClear(pData);
  vKvLid->mClear(pData);
}

bool mgs_kv_cache_dsa::mSeqRm(mgs_seq_id pSeqId, mgs_pos p0, mgs_pos p1) {
  bool vRes = true;

  vRes = vRes & vKvMla->mSeqRm(pSeqId, p0, p1);
  vRes = vRes & vKvLid->mSeqRm(pSeqId, p0, p1);

  return vRes;
}

void mgs_kv_cache_dsa::mSeqCp(mgs_seq_id pSeqIdSrc, mgs_seq_id pSeqIdDst, mgs_pos p0, mgs_pos p1) {
  vKvMla->mSeqCp(pSeqIdSrc, pSeqIdDst, p0, p1);
  vKvLid->mSeqCp(pSeqIdSrc, pSeqIdDst, p0, p1);
}

void mgs_kv_cache_dsa::mSeqKeep(mgs_seq_id pSeqId) {
  vKvMla->mSeqKeep(pSeqId);
  vKvLid->mSeqKeep(pSeqId);
}

void mgs_kv_cache_dsa::mSeqAdd(mgs_seq_id pSeqId, mgs_pos p0, mgs_pos p1, mgs_pos pShift) {
  vKvMla->mSeqAdd(pSeqId, p0, p1, pShift);
  vKvLid->mSeqAdd(pSeqId, p0, p1, pShift);
}

void mgs_kv_cache_dsa::mSeqDiv(mgs_seq_id pSeqId, mgs_pos p0, mgs_pos p1, int pD) {
  vKvMla->mSeqDiv(pSeqId, p0, p1, pD);
  vKvLid->mSeqDiv(pSeqId, p0, p1, pD);
}

mgs_pos mgs_kv_cache_dsa::mSeqPosMin(mgs_seq_id pSeqId) const { return vKvMla->mSeqPosMin(pSeqId); }

mgs_pos mgs_kv_cache_dsa::mSeqPosMax(mgs_seq_id pSeqId) const { return vKvMla->mSeqPosMax(pSeqId); }

std::map<ggml_backend_buffer_type_t, size_t> mgs_kv_cache_dsa::mMemoryBreakdown() const {
  std::map<ggml_backend_buffer_type_t, size_t> dMb = vKvMla->mMemoryBreakdown();
  for (const auto &vBuftSize : vKvLid->mMemoryBreakdown()) {
    dMb[vBuftSize.first] += vBuftSize.second;
  }
  return dMb;
}

mgs_memory_context_ptr mgs_kv_cache_dsa::mInitBatch(mgs_batch_allocr &pBalloc, uint32_t pNUbatch, bool pEmbdAll) {
  GGML_UNUSED(pEmbdAll);

  do {
    pBalloc.mSplitReset();

    std::vector<mgs_ubatch> lUbatches;
    while (true) {
      auto vUbatch = cNStream == 1 ? pBalloc.mSplitSimple(pNUbatch) : pBalloc.mSplitEqual(pNUbatch, true, 0);

      if (vUbatch.vNTokens == 0) {
        break;
      }

      lUbatches.push_back(std::move(vUbatch)); // NOLINT
    }

    if (pBalloc.mGetNUsed() < pBalloc.mGetNTokens()) {
      // failed to find a suitable split
      break;
    }

    auto vSinfosMla = vKvMla->mPrepare(lUbatches);
    if (vSinfosMla.empty()) {
      break;
    }

    auto vSinfosLid = vKvLid->mPrepare(lUbatches);
    if (vSinfosLid.empty()) {
      break;
    }

    assert(vSinfosMla.size() == vSinfosLid.size());

    return std::make_unique<mgs_kv_cache_dsa_context>(this, std::move(vSinfosMla), std::move(vSinfosLid),
                                                      std::move(lUbatches));
  } while (false);

  return std::make_unique<mgs_kv_cache_dsa_context>(cMgsMemoryStatusFailedPrepare);
}

mgs_memory_context_ptr mgs_kv_cache_dsa::mInitFull() { return std::make_unique<mgs_kv_cache_dsa_context>(this); }

mgs_memory_context_ptr mgs_kv_cache_dsa::mInitUpdate(mgs_context *pLctx, bool pOptimize) {
  return std::make_unique<mgs_kv_cache_dsa_context>(this, pLctx, pOptimize);
}

bool mgs_kv_cache_dsa::mGetCanShift() const {
  return vKvMla->mGetCanShift() && vKvLid->mGetCanShift() && vKvMla->mGetSize() == vKvLid->mGetSize();
}

void mgs_kv_cache_dsa::mStateWrite(mgs_io_write_i &pIo, mgs_seq_id pSeqId, mgs_state_seq_flags pFlags) const {
  vKvMla->mStateWrite(pIo, pSeqId, pFlags);
  vKvLid->mStateWrite(pIo, pSeqId, pFlags);
}

void mgs_kv_cache_dsa::mStateRead(mgs_io_read_i &pIo, mgs_seq_id pSeqId, mgs_state_seq_flags pFlags) {
  vKvMla->mStateRead(pIo, pSeqId, pFlags);
  vKvLid->mStateRead(pIo, pSeqId, pFlags);
}

mgs_kv_cache *mgs_kv_cache_dsa::mGetMla() const { return vKvMla.get(); }

mgs_kv_cache *mgs_kv_cache_dsa::mGetLid() const { return vKvLid.get(); }

//
// mgs_kv_cache_dsa_context
//

mgs_kv_cache_dsa_context::mgs_kv_cache_dsa_context(mgs_memory_status pStatus) : cStatus(pStatus) {}

mgs_kv_cache_dsa_context::mgs_kv_cache_dsa_context(mgs_kv_cache_dsa *pKv)
    : cCtxMla(pKv->mGetMla()->mInitFull()), cCtxLid(pKv->mGetLid()->mInitFull()),
      cStatus(mgs_memory_status_combine(cCtxMla->mGetStatus(), cCtxLid->mGetStatus())) {}

mgs_kv_cache_dsa_context::mgs_kv_cache_dsa_context(mgs_kv_cache_dsa *pKv, mgs_context *pLctx, bool pOptimize)
    : cCtxMla(pKv->mGetMla()->mInitUpdate(pLctx, pOptimize)), cCtxLid(pKv->mGetLid()->mInitUpdate(pLctx, pOptimize)),
      cStatus(mgs_memory_status_combine(cCtxMla->mGetStatus(), cCtxLid->mGetStatus())) {}

mgs_kv_cache_dsa_context::mgs_kv_cache_dsa_context(mgs_kv_cache_dsa *pKv, slot_info_vec_t pSinfosMla,
                                                   slot_info_vec_t pSinfosLid, std::vector<mgs_ubatch> pUbatches)
    : lUbatches(std::move(pUbatches)),
      // note: here we copy the ubatches. not sure if this is ideal
      cCtxMla(new mgs_kv_cache_context(pKv->mGetMla(), std::move(pSinfosMla), this->lUbatches)),
      cCtxLid(new mgs_kv_cache_context(pKv->mGetLid(), std::move(pSinfosLid), this->lUbatches)),
      cStatus(mgs_memory_status_combine(cCtxMla->mGetStatus(), cCtxLid->mGetStatus())) {}

mgs_kv_cache_dsa_context::~mgs_kv_cache_dsa_context() = default;

bool mgs_kv_cache_dsa_context::mNext() {
  assert(cStatus == cMgsMemoryStatusSuccess);

  cCtxMla->mNext();
  cCtxLid->mNext();

  if (++vINext >= lUbatches.size()) {
    return false;
  }

  return true;
}

bool mgs_kv_cache_dsa_context::mApply() {
  assert(!mgs_memory_status_is_fail(cStatus));

  bool vRes = true;

  vRes = vRes & cCtxMla->mApply();
  vRes = vRes & cCtxLid->mApply();

  return vRes;
}

mgs_memory_status mgs_kv_cache_dsa_context::mGetStatus() const { return cStatus; }

const mgs_ubatch &mgs_kv_cache_dsa_context::mGetUbatch() const {
  assert(cStatus == cMgsMemoryStatusSuccess);

  return lUbatches[vINext];
}

const mgs_kv_cache_context *mgs_kv_cache_dsa_context::mGetMla() const {
  assert(cStatus == cMgsMemoryStatusSuccess);

  return static_cast<const mgs_kv_cache_context *>(cCtxMla.get());
}

const mgs_kv_cache_context *mgs_kv_cache_dsa_context::mGetLid() const {
  assert(cStatus == cMgsMemoryStatusSuccess);

  return static_cast<const mgs_kv_cache_context *>(cCtxLid.get());
}
