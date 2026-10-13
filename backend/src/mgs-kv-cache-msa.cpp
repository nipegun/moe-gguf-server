#include "mgs-kv-cache-msa.h"

#include "mgs-impl.h"
#include "mgs-batch.h"
#include "mgs-model.h"

#include <algorithm>
#include <cassert>
#include <cmath>

// mgs_kv_cache_msa

mgs_kv_cache_msa::mgs_kv_cache_msa(const mgs_model &pModel, ggml_type pTypeK, ggml_type pTypeV, bool pVTrans,
                                   bool pOffload, bool pUnified, uint32_t pKvSize, uint32_t pNSeqMax, uint32_t pNPad,
                                   uint32_t pNSwa, mgs_swa_type pSwaType, const layer_filter_cb &pFilter,
                                   const layer_filter_cb &pFilterIdx, const layer_reuse_cb &pReuse)
    : vHparamsIdx(pModel.vHparams), cNStream(pUnified ? 1 : pNSeqMax), cNSeqMax(pNSeqMax), cNPad(pNPad), cNSwa(pNSwa),
      cSwaType(pSwaType) {

  MGS_LOG_INFO("%s: creating main KV cache, size = %u cells\n", __func__, pKvSize);

  vKvBase =
      std::make_unique<mgs_kv_cache>(pModel, pModel.vHparams, pTypeK, pTypeV, pVTrans, pOffload, pUnified, pKvSize,
                                     pNSeqMax, pNPad, pNSwa, pSwaType, nullptr, pFilter, pReuse, nullptr);

  // the MSA indexer uses a single key head per layer
  std::fill(vHparamsIdx.aNHeadKvArr.begin(), vHparamsIdx.aNHeadKvArr.end(), 1);
  vHparamsIdx.vNEmbdHeadKFull = pModel.vHparams.vIndexerHeadSize;
  // the rope parameters are kept identical to the main cache

  MGS_LOG_INFO("%s: creating indexer KV cache, size = %u cells\n", __func__, pKvSize);

  vKvIdx = std::make_unique<mgs_kv_cache>(pModel, vHparamsIdx, pTypeK, pTypeV, pVTrans, pOffload, pUnified, pKvSize,
                                          pNSeqMax, pNPad, pNSwa, pSwaType, nullptr, pFilterIdx, pReuse, nullptr);
}

void mgs_kv_cache_msa::mClear(bool pData) {
  vKvBase->mClear(pData);
  vKvIdx->mClear(pData);
}

bool mgs_kv_cache_msa::mSeqRm(mgs_seq_id pSeqId, mgs_pos p0, mgs_pos p1) {
  bool vRes = true;

  vRes = vRes & vKvBase->mSeqRm(pSeqId, p0, p1);
  vRes = vRes & vKvIdx->mSeqRm(pSeqId, p0, p1);

  return vRes;
}

void mgs_kv_cache_msa::mSeqCp(mgs_seq_id pSeqIdSrc, mgs_seq_id pSeqIdDst, mgs_pos p0, mgs_pos p1) {
  vKvBase->mSeqCp(pSeqIdSrc, pSeqIdDst, p0, p1);
  vKvIdx->mSeqCp(pSeqIdSrc, pSeqIdDst, p0, p1);
}

void mgs_kv_cache_msa::mSeqKeep(mgs_seq_id pSeqId) {
  vKvBase->mSeqKeep(pSeqId);
  vKvIdx->mSeqKeep(pSeqId);
}

void mgs_kv_cache_msa::mSeqAdd(mgs_seq_id pSeqId, mgs_pos p0, mgs_pos p1, mgs_pos pShift) {
  vKvBase->mSeqAdd(pSeqId, p0, p1, pShift);
  vKvIdx->mSeqAdd(pSeqId, p0, p1, pShift);
}

void mgs_kv_cache_msa::mSeqDiv(mgs_seq_id pSeqId, mgs_pos p0, mgs_pos p1, int pD) {
  vKvBase->mSeqDiv(pSeqId, p0, p1, pD);
  vKvIdx->mSeqDiv(pSeqId, p0, p1, pD);
}

mgs_pos mgs_kv_cache_msa::mSeqPosMin(mgs_seq_id pSeqId) const { return vKvBase->mSeqPosMin(pSeqId); }

mgs_pos mgs_kv_cache_msa::mSeqPosMax(mgs_seq_id pSeqId) const { return vKvBase->mSeqPosMax(pSeqId); }

std::map<ggml_backend_buffer_type_t, size_t> mgs_kv_cache_msa::mMemoryBreakdown() const {
  std::map<ggml_backend_buffer_type_t, size_t> dMb = vKvBase->mMemoryBreakdown();
  for (const auto &vBuftSize : vKvIdx->mMemoryBreakdown()) {
    dMb[vBuftSize.first] += vBuftSize.second;
  }
  return dMb;
}

mgs_memory_context_ptr mgs_kv_cache_msa::mInitBatch(mgs_batch_allocr &pBalloc, uint32_t pNUbatch, bool pEmbdAll) {
  GGML_UNUSED(pEmbdAll);

  do {
    pBalloc.mSplitReset();

    std::vector<mgs_ubatch> lUbatches;
    while (true) {
      auto vUbatch = cNStream == 1 ? pBalloc.mSplitSimple(pNUbatch) : pBalloc.mSplitEqual(pNUbatch, true, 0);

      if (vUbatch.vNTokens == 0) {
        break;
      }

      lUbatches.push_back(std::move(vUbatch));
    }

    if (pBalloc.mGetNUsed() < pBalloc.mGetNTokens()) {
      // failed to find a suitable split
      break;
    }

    auto vSinfosBase = vKvBase->mPrepare(lUbatches);
    if (vSinfosBase.empty()) {
      break;
    }

    auto vSinfosIdx = vKvIdx->mPrepare(lUbatches);
    if (vSinfosIdx.empty()) {
      break;
    }

    assert(vSinfosBase.size() == vSinfosIdx.size());

    return std::make_unique<mgs_kv_cache_msa_context>(this, std::move(vSinfosBase), std::move(vSinfosIdx),
                                                      std::move(lUbatches));
  } while (false);

  return std::make_unique<mgs_kv_cache_msa_context>(cMgsMemoryStatusFailedPrepare);
}

mgs_memory_context_ptr mgs_kv_cache_msa::mInitFull() { return std::make_unique<mgs_kv_cache_msa_context>(this); }

mgs_memory_context_ptr mgs_kv_cache_msa::mInitUpdate(mgs_context *pLctx, bool pOptimize) {
  return std::make_unique<mgs_kv_cache_msa_context>(this, pLctx, pOptimize);
}

bool mgs_kv_cache_msa::mGetCanShift() const {
  return vKvBase->mGetCanShift() && vKvIdx->mGetCanShift() && vKvBase->mGetSize() == vKvIdx->mGetSize();
}

void mgs_kv_cache_msa::mStateWrite(mgs_io_write_i &pIo, mgs_seq_id pSeqId, mgs_state_seq_flags pFlags) const {
  vKvBase->mStateWrite(pIo, pSeqId, pFlags);
  vKvIdx->mStateWrite(pIo, pSeqId, pFlags);
}

void mgs_kv_cache_msa::mStateRead(mgs_io_read_i &pIo, mgs_seq_id pSeqId, mgs_state_seq_flags pFlags) {
  vKvBase->mStateRead(pIo, pSeqId, pFlags);
  vKvIdx->mStateRead(pIo, pSeqId, pFlags);
}

mgs_kv_cache *mgs_kv_cache_msa::mGetBase() const { return vKvBase.get(); }

mgs_kv_cache *mgs_kv_cache_msa::mGetIdx() const { return vKvIdx.get(); }

// mgs_kv_cache_msa_context

mgs_kv_cache_msa_context::mgs_kv_cache_msa_context(mgs_memory_status pStatus) : vKv(nullptr), cStatus(pStatus) {}

mgs_kv_cache_msa_context::mgs_kv_cache_msa_context(mgs_kv_cache_msa *pKv)
    : vKv(pKv), cCtxBase(pKv->mGetBase()->mInitFull()), cCtxIdx(pKv->mGetIdx()->mInitFull()),
      cStatus(mgs_memory_status_combine(cCtxBase->mGetStatus(), cCtxIdx->mGetStatus())) {}

mgs_kv_cache_msa_context::mgs_kv_cache_msa_context(mgs_kv_cache_msa *pKv, mgs_context *pLctx, bool pOptimize)
    : vKv(pKv), cCtxBase(pKv->mGetBase()->mInitUpdate(pLctx, pOptimize)),
      cCtxIdx(pKv->mGetIdx()->mInitUpdate(pLctx, pOptimize)),
      cStatus(mgs_memory_status_combine(cCtxBase->mGetStatus(), cCtxIdx->mGetStatus())) {}

mgs_kv_cache_msa_context::mgs_kv_cache_msa_context(mgs_kv_cache_msa *pKv, slot_info_vec_t pSinfosBase,
                                                   slot_info_vec_t pSinfosIdx, std::vector<mgs_ubatch> pUbatches)
    : vKv(pKv), lUbatches(std::move(pUbatches)),
      // here we copy the ubatches. not sure if this is ideal
      cCtxBase(new mgs_kv_cache_context(pKv->mGetBase(), std::move(pSinfosBase), this->lUbatches)),
      cCtxIdx(new mgs_kv_cache_context(pKv->mGetIdx(), std::move(pSinfosIdx), this->lUbatches)),
      cStatus(mgs_memory_status_combine(cCtxBase->mGetStatus(), cCtxIdx->mGetStatus())) {}

mgs_kv_cache_msa_context::~mgs_kv_cache_msa_context() = default;

bool mgs_kv_cache_msa_context::mNext() {
  assert(cStatus == cMgsMemoryStatusSuccess);

  cCtxBase->mNext();
  cCtxIdx->mNext();

  if (++vINext >= lUbatches.size()) {
    return false;
  }

  return true;
}

bool mgs_kv_cache_msa_context::mApply() {
  assert(!mgs_memory_status_is_fail(cStatus));

  bool vRes = true;

  vRes = vRes & cCtxBase->mApply();
  vRes = vRes & cCtxIdx->mApply();

  return vRes;
}

mgs_memory_status mgs_kv_cache_msa_context::mGetStatus() const { return cStatus; }

const mgs_ubatch &mgs_kv_cache_msa_context::mGetUbatch() const {
  assert(cStatus == cMgsMemoryStatusSuccess);

  return lUbatches[vINext];
}

const mgs_kv_cache_context *mgs_kv_cache_msa_context::mGetBase() const {
  assert(cStatus == cMgsMemoryStatusSuccess);

  return static_cast<const mgs_kv_cache_context *>(cCtxBase.get());
}

const mgs_kv_cache_context *mgs_kv_cache_msa_context::mGetIdx() const {
  assert(cStatus == cMgsMemoryStatusSuccess);

  return static_cast<const mgs_kv_cache_context *>(cCtxIdx.get());
}

uint32_t mgs_kv_cache_msa_context::mGetNPos() const {
  // pad the value so that the graph remains constant across batches and can be reused
  const uint32_t cNPadCur = std::max(vKv->mGetNPad(), 256u);

  mgs_pos vPosMax = -1;

  for (mgs_seq_id vSeqId = 0; vSeqId < (mgs_seq_id)vKv->mGetNSeqMax(); ++vSeqId) {
    vPosMax = std::max(vPosMax, vKv->mSeqPosMax(vSeqId));
  }

  return std::max(cNPadCur, GGML_PAD((uint32_t)(vPosMax + 1), cNPadCur));
}

void mgs_kv_cache_msa_context::mSetInputCellPos(ggml_tensor *pDst, const mgs_ubatch *pUbatch, int32_t pDiv) const {
  GGML_ASSERT(ggml_backend_buffer_is_host(pDst->buffer));
  GGML_ASSERT(pDst->type == GGML_TYPE_I32);
  GGML_ASSERT(pDiv > 0);

  const int64_t cNTokens = pUbatch->vNTokens;
  const int64_t cNKv = pDst->ne[0];
  const int64_t cNStreamUb = pDst->ne[1];

  GGML_ASSERT(cNTokens % cNStreamUb == 0);
  const int64_t cNTps = cNTokens / cNStreamUb;

  int32_t *vData = (int32_t *)pDst->data;

  for (int64_t vS = 0; vS < cNStreamUb; ++vS) {
    const mgs_seq_id cSeqId = pUbatch->vSeqId[vS * cNTps][0];

    const auto &vCells = vKv->mGetBase()->mGetCells(cSeqId);

    for (int64_t vJ = 0; vJ < cNKv; ++vJ) {
      // the value for empty or other-sequence cells is irrelevant as consumers mask them
      vData[vS * cNKv + vJ] =
          vCells.mIsEmpty(vJ) || !vCells.mSeqHas(vJ, cSeqId) ? 0 : (int32_t)(vCells.mPosGet(vJ) / pDiv);
    }
  }
}

void mgs_kv_cache_msa_context::mSetInputPosSlot(ggml_tensor *pDst, const mgs_ubatch *pUbatch) const {
  GGML_ASSERT(ggml_backend_buffer_is_host(pDst->buffer));
  GGML_ASSERT(pDst->type == GGML_TYPE_I32 || pDst->type == GGML_TYPE_F32);

  const int64_t cNTokens = pUbatch->vNTokens;
  const int64_t cNPos = pDst->ne[0];
  const int64_t cNStreamUb = pDst->ne[1];

  GGML_ASSERT(cNTokens % cNStreamUb == 0);
  const int64_t cNTps = cNTokens / cNStreamUb;

  for (int64_t vS = 0; vS < cNStreamUb; ++vS) {
    const mgs_seq_id cSeqId = pUbatch->vSeqId[vS * cNTps][0];

    const auto &vCells = vKv->mGetBase()->mGetCells(cSeqId);

    std::vector<int32_t> lMap(cNPos, 0);

    for (uint32_t vJ = 0; vJ < vCells.size(); ++vJ) {
      if (vCells.mIsEmpty(vJ) || !vCells.mSeqHas(vJ, cSeqId)) {
        continue;
      }

      const mgs_pos cP0 = vCells.mPosGet(vJ);

      if (cP0 < 0 || cP0 >= cNPos) {
        continue;
      }

      lMap[cP0] = (int32_t)vJ;
    }

    if (pDst->type == GGML_TYPE_I32) {
      int32_t *vData = (int32_t *)pDst->data + vS * cNPos;
      std::copy(lMap.begin(), lMap.end(), vData);
    } else {
      float *vData = (float *)pDst->data + vS * cNPos;
      for (int64_t vP = 0; vP < cNPos; ++vP) {
        vData[vP] = (float)lMap[vP];
      }
    }
  }
}

void mgs_kv_cache_msa_context::mSetInputPosMask(ggml_tensor *pDst, const mgs_ubatch *pUbatch) const {
  GGML_ASSERT(ggml_backend_buffer_is_host(pDst->buffer));
  GGML_ASSERT(pDst->type == GGML_TYPE_F32);

  const int64_t cNTokens = pUbatch->vNTokens;
  const int64_t cNPos = pDst->ne[0];

  GGML_ASSERT(pDst->ne[1] == cNTokens);

  const uint32_t cNSwa = vKv->mGetNSwa();
  const mgs_swa_type cSwaType = vKv->mGetSwaType();

  float *vData = (float *)pDst->data;

  std::fill(vData, vData + cNPos * cNTokens, -INFINITY);

  for (int64_t vI = 0; vI < cNTokens; ++vI) {
    const mgs_seq_id cSeqId = pUbatch->vSeqId[vI][0];

    const auto &vCells = vKv->mGetBase()->mGetCells(cSeqId);

    const mgs_pos cP1 = pUbatch->vPos[vI];

    for (uint32_t vJ = 0; vJ < vCells.size(); ++vJ) {
      if (vCells.mIsEmpty(vJ) || !vCells.mSeqHas(vJ, cSeqId)) {
        continue;
      }

      const mgs_pos cP0 = vCells.mPosGet(vJ);

      if (cP0 < 0 || cP0 >= cNPos) {
        continue;
      }

      // causal mask
      if (cP0 > cP1) {
        continue;
      }

      // apply SWA if any
      if (mgs_hparams::mIsMaskedSwa(cNSwa, cSwaType, cP0, cP1)) {
        continue;
      }

      vData[vI * cNPos + cP0] = 0.0f;
    }
  }
}
