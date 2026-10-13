#include "mgs-memory-hybrid-idx.h"

#include "mgs-impl.h"
#include "mgs-batch.h"
#include "mgs-io.h"
#include "mgs-model.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iterator>
#include <stdexcept>
#include <type_traits>

//
// mgs_memory_hybrid_idx
//

mgs_memory_hybrid_idx::mgs_memory_hybrid_idx(const mgs_model &pModel,
                                             /* attn */
                                             ggml_type pTypeK, ggml_type pTypeV, bool pVTrans, uint32_t pKvSize,
                                             uint32_t pNPad, uint32_t pNSwa, mgs_swa_type pSwaType,
                                             /* recurrent */
                                             ggml_type pTypeR, ggml_type pTypeS, uint32_t pRsSize,
                                             /* common */
                                             uint32_t pNSeqMax, uint32_t pNRsSeq, bool pOffload, bool pUnified,
                                             /* layer filters */
                                             const layer_filter_cb &pFilterAttn, const layer_filter_cb &pFilterRecr,
                                             const layer_filter_cb &pFilterIdx)
    : mgs_memory_hybrid(pModel, pTypeK, pTypeV, pVTrans, pKvSize, pNPad, pNSwa, pSwaType, pTypeR, pTypeS, pRsSize,
                        pNSeqMax, pNRsSeq, pOffload, pUnified, pFilterAttn, pFilterRecr),
      vHparamsIdx(pModel.vHparams), cMemIdx(pFilterIdx == nullptr ? nullptr : [&] {
        // MQA with a single key head of indexer_head_size, as mgs_kv_cache_dsa shapes its own
        std::fill(vHparamsIdx.aNHeadKvArr.begin(), vHparamsIdx.aNHeadKvArr.end(), 1);
        // a k-pool indexer caches its per-token rows and the pooled key side by side
        // (glm5-next: key | gate | pooled, qwen4exp: key | pooled)
        const uint32_t cRowSlots = pModel.vHparams.vIndexerKpool > 0 ? pModel.vHparams.vIndexerKpoolRow : 1;
        vHparamsIdx.vNEmbdHeadKFull = pModel.vHparams.vIndexerHeadSize * cRowSlots;

        // the cached indexer keys are raw, rotation happens after pooling at read time, so a
        // K-shift must not rotate them while the stream copies in the same update still apply
        vHparamsIdx.vRopeType = MGS_ROPE_TYPE_NONE;

        // fool mgs_kv_cache into thinking this is a MLA cache, so it won't cache V tensors
        vHparamsIdx.vNEmbdHeadKMlaImpl = pModel.vHparams.vIndexerHeadSize;
        vHparamsIdx.vNEmbdHeadVMlaImpl = pModel.vHparams.vIndexerHeadSize;

        MGS_LOG_INFO("%s: creating indexer KV cache, size = %u cells\n", __func__, pKvSize);

        return new mgs_kv_cache(pModel, vHparamsIdx, pTypeK, pTypeV, pVTrans, pOffload, pUnified, pKvSize, pNSeqMax,
                                pNPad, pNSwa, pSwaType, nullptr, pFilterIdx, nullptr, nullptr, "idx_");
      }()) {}

mgs_memory_context_ptr mgs_memory_hybrid_idx::mInitBatch(mgs_batch_allocr &pBalloc, uint32_t pNUbatch,
                                                         bool pEmbdAll) {
  // note: repeats mgs_memory_hybrid::mInitBatch, as the indexer needs the attention slot infos that the base
  // context hides
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
        // use non-sequential split when KV cache is unified (needed for hellaswag/winogrande/multiple-choice)
        const bool cUnified = (mGetMemAttn()->mGetNStream() == 1);

        // [TAG_RECURRENT_ROLLBACK_SPLITS]
        // the trailing (1 + n_rs_seq) tokens of each seq must stay in the same ubatch
        //   so that the rollback snapshots remain valid
        const uint32_t cNRsSeq = mGetMemRecr()->vNRsSeq;

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
    if (!mGetMemRecr()->mPrepare(lUbatches)) {
      MGS_LOG_ERROR("%s: failed to prepare recurrent ubatches\n", __func__);
      return std::make_unique<mgs_memory_hybrid_idx_context>(cMgsMemoryStatusFailedPrepare);
    }

    // prepare the attention cache
    auto vHeadsAttn = mGetMemAttn()->mPrepare(lUbatches);
    if (vHeadsAttn.empty()) {
      MGS_LOG_ERROR("%s: failed to prepare attention ubatches\n", __func__);
      return std::make_unique<mgs_memory_hybrid_idx_context>(cMgsMemoryStatusFailedPrepare);
    }

    // the indexer uses the attention cache's slot layout; a separate one can drift from it
    mgs_kv_cache::slot_info_vec_t vHeadsIdx;
    if (cMemIdx) {
      vHeadsIdx = vHeadsAttn;
    }

    return std::make_unique<mgs_memory_hybrid_idx_context>(this, std::move(vHeadsAttn), std::move(vHeadsIdx),
                                                           std::move(lUbatches));
  } while (false);

  return std::make_unique<mgs_memory_hybrid_idx_context>(cMgsMemoryStatusFailedPrepare);
}

mgs_memory_context_ptr mgs_memory_hybrid_idx::mInitFull() {
  return std::make_unique<mgs_memory_hybrid_idx_context>(this);
}

mgs_memory_context_ptr mgs_memory_hybrid_idx::mInitUpdate(mgs_context *pLctx, bool pOptimize) {
  return std::make_unique<mgs_memory_hybrid_idx_context>(this, pLctx, pOptimize);
}

void mgs_memory_hybrid_idx::mClear(bool pData) {
  mgs_memory_hybrid::mClear(pData);

  if (cMemIdx) {
    cMemIdx->mClear(pData);
    mMemIdxStaleSet(-1, 0);
  }
}

// a pooled key is only valid while the grouping that produced it holds. grouping is sequence relative,
// so an edit at p0 leaves every pool that ends before p0 alone
void mgs_memory_hybrid_idx::mMemIdxStaleSet(mgs_seq_id pSeqId, mgs_pos p0) {
  p0 = std::max<mgs_pos>(p0, 0);

  if (pSeqId < 0) {
    for (auto &vP : aMemIdxStale) {
      vP = std::min(vP, p0);
    }

    return;
  }

  GGML_ASSERT(pSeqId < (mgs_seq_id)MGS_MAX_SEQ);

  aMemIdxStale[pSeqId] = std::min(aMemIdxStale[pSeqId], p0);
}

// an edit at or below the first position moves pos_min, which regroups the whole sequence
mgs_pos mgs_memory_hybrid_idx::mMemIdxStalePos(mgs_seq_id pSeqId, mgs_pos p0) const {
  if (pSeqId < 0 || p0 <= cMemIdx->mSeqPosMin(pSeqId)) {
    return 0;
  }

  return p0;
}

bool mgs_memory_hybrid_idx::mSeqRm(mgs_seq_id pSeqId, mgs_pos p0, mgs_pos p1) {
  // same order as mgs_memory_hybrid::mSeqRm: the recurrent cache can refuse, so try it first
  if (!mGetMemRecr()->mSeqRm(pSeqId, p0, p1)) {
    return false;
  }

  if (cMemIdx) {
    const mgs_pos cStale = mMemIdxStalePos(pSeqId, p0);
    cMemIdx->mSeqRm(pSeqId, p0, p1);
    mMemIdxStaleSet(pSeqId, cStale);
  }

  return mGetMemAttn()->mSeqRm(pSeqId, p0, p1);
}

void mgs_memory_hybrid_idx::mSeqCp(mgs_seq_id pSeqIdSrc, mgs_seq_id pSeqIdDst, mgs_pos p0, mgs_pos p1) {
  // only whole sequences are copied: the recurrent state ignores the range, and a shared cell holds a single pool
  // grouping
  GGML_ASSERT(p0 <= 0 && p1 < 0 && "partial seq_cp is not supported");

  mgs_memory_hybrid::mSeqCp(pSeqIdSrc, pSeqIdDst, p0, p1);

  if (cMemIdx) {
    cMemIdx->mSeqCp(pSeqIdSrc, pSeqIdDst, p0, p1);
    // a whole sequence copy gives the destination the source's pools, rep rows included: the source keeps its
    // pooled keys, the destination rebuilds its layout and re-pools into the same rows
    mMemIdxStaleSet(pSeqIdDst, 0);
  }
}

void mgs_memory_hybrid_idx::mSeqKeep(mgs_seq_id pSeqId) {
  mgs_memory_hybrid::mSeqKeep(pSeqId);

  if (cMemIdx) {
    cMemIdx->mSeqKeep(pSeqId);
    // every other sequence loses its cells, so their layouts must rebuild
    mMemIdxStaleSet(-1, 0);
  }
}

void mgs_memory_hybrid_idx::mSeqAdd(mgs_seq_id pSeqId, mgs_pos p0, mgs_pos p1, mgs_pos pShift) {
  mgs_memory_hybrid::mSeqAdd(pSeqId, p0, p1, pShift);

  if (cMemIdx) {
    // a negative shift moves the cells below p0, so they regroup as well
    const mgs_pos cStale = mMemIdxStalePos(pSeqId, pShift < 0 ? p0 + pShift : p0);
    cMemIdx->mSeqAdd(pSeqId, p0, p1, pShift);
    mMemIdxStaleSet(pSeqId, cStale);
  }
}

void mgs_memory_hybrid_idx::mSeqDiv(mgs_seq_id pSeqId, mgs_pos p0, mgs_pos p1, int pD) {
  mgs_memory_hybrid::mSeqDiv(pSeqId, p0, p1, pD);

  if (cMemIdx) {
    cMemIdx->mSeqDiv(pSeqId, p0, p1, pD);
    mMemIdxStaleSet(pSeqId, 0);
  }
}

std::map<ggml_backend_buffer_type_t, size_t> mgs_memory_hybrid_idx::mMemoryBreakdown() const {
  std::map<ggml_backend_buffer_type_t, size_t> dMb = mgs_memory_hybrid::mMemoryBreakdown();

  if (cMemIdx) {
    for (const auto &vBuftSize : cMemIdx->mMemoryBreakdown()) {
      dMb[vBuftSize.first] += vBuftSize.second;
    }
  }

  return dMb;
}

void mgs_memory_hybrid_idx::mStateWrite(mgs_io_write_i &pIo, mgs_seq_id pSeqId, mgs_state_seq_flags pFlags) const {
  mgs_memory_hybrid::mStateWrite(pIo, pSeqId, pFlags);

  // [TAG_HYBRID_IDX_STATE] the indexer section goes last, so it is a pure suffix
  // the indexer mirrors the attention cache, so it uses the same PARTIAL_ONLY gate
  if ((pFlags & MGS_STATE_SEQ_FLAGS_PARTIAL_ONLY) == 0) {
    if (cMemIdx) {
      cMemIdx->mStateWrite(pIo, pSeqId, pFlags);
    }
  }
}

void mgs_memory_hybrid_idx::mStateRead(mgs_io_read_i &pIo, mgs_seq_id pSeqId, mgs_state_seq_flags pFlags) {
  // note: repeats mgs_memory_hybrid::mStateRead
  // the indexer needs the attention cache's cells, and a half-failed restore must leave all three caches alike

  // [TAG_HYBRID_IDX_SINFO]
  // the indexer restore adopts the attention cache's layout instead of searching for cells of its own
  // two mFindSlot calls agree only while both caches see the same occupancy, which a restore cannot promise
  mgs_kv_cache::slot_info_vec_t vSinfosAttn;

  try {
    if ((pFlags & MGS_STATE_SEQ_FLAGS_PARTIAL_ONLY) == 0) {
      mGetMemAttn()->mStateReadSinfo(pIo, pSeqId, pFlags, cMemIdx ? &vSinfosAttn : nullptr, nullptr);
    }

    mGetMemRecr()->mStateRead(pIo, pSeqId, pFlags);

    // [TAG_HYBRID_IDX_STATE] must mirror the write order in mStateWrite
    if ((pFlags & MGS_STATE_SEQ_FLAGS_PARTIAL_ONLY) == 0) {
      if (cMemIdx) {
        cMemIdx->mStateReadSinfo(pIo, pSeqId, pFlags, nullptr, &vSinfosAttn);
        // the restore rewrites the cells behind the pool layout's back
        mMemIdxStaleSet(pSeqId, 0);
      }
    }
  } catch (...) {
    // a half-restored context is the one state the indexer cannot fix by itself: attention holds new cells, the
    // indexer old ones. drop what was being restored from all of them, which is a state they do agree on
    mStateDrop(pSeqId);

    throw;
  }
}

void mgs_memory_hybrid_idx::mStateDrop(mgs_seq_id pSeqId) {
  // dropped directly, not via mSeqRm: the recurrent cache may refuse it and then only the other two get cleared
  if (pSeqId < 0) {
    mClear(true);

    return;
  }

  mGetMemAttn()->mStateClear(pSeqId);
  mGetMemRecr()->mSeqRm(pSeqId, -1, -1);

  if (cMemIdx) {
    cMemIdx->mStateClear(pSeqId);
    mMemIdxStaleSet(pSeqId, 0);
  }
}

mgs_kv_cache *mgs_memory_hybrid_idx::mGetMemIdx() const { return cMemIdx.get(); }

//
// mgs_memory_hybrid_idx_context
//

// streams in each ubatch's slot info, matching mGetK/mGetV's `ns`
static std::vector<uint32_t> fMemoryHybridIdxNs(const mgs_kv_cache::slot_info_vec_t &pSinfos) {
  std::vector<uint32_t> lRes;
  lRes.reserve(pSinfos.size());

  for (const auto &vSinfo : pSinfos) {
    lRes.push_back(vSinfo.vS1 - vSinfo.vS0 + 1);
  }

  return lRes;
}

// which cells of a sequence make up which pool, for the whole cache
struct mgs_memory_hybrid_idx::kpool_layout {
  struct seq {
    mgs_pos vPosMin = 0;
    uint32_t vStrm = 0;                                // stream holding this sequence's cells
    std::vector<std::pair<mgs_pos, uint32_t>> lCells; // position and stream local cell pairs, sorted by position
    std::vector<uint32_t> lPools;

    // where the pool scan stopped, so an append resumes instead of starting over
    size_t vJNext = 0;
  };

  std::array<seq, MGS_MAX_SEQ> aSeqs;

  uint32_t vNPoolReal = 0;
};

// which pools of the layout the current ubatch must re-pool, in the layout's pool order
struct mgs_memory_hybrid_idx_context::kpool_state {
  std::vector<uint32_t> lIsNew;
  std::vector<uint32_t> lRepGen; // per global cell, the generation that last marked a pool with that rep
  uint32_t vGeneration = 0;

  uint32_t vNPoolReal = 0;
  uint32_t vNNew = 0;
  uint32_t vNNewG = 1; // graph size of the new pool list, stable across decode steps
};

namespace {

// the last padded pool is always unused
uint32_t fKpoolPad(uint32_t pNPool) { return std::max<uint32_t>(64u, GGML_PAD(pNPool + 1, 64u)); }

// rank of (pos, cell) in a sequence's cells sorted by position then cell, or -1 when absent
// in order mode the rank alone places a token: cells sharing a position (M-RoPE images) have distinct ranks
int64_t fKpoolRank(const std::vector<std::pair<mgs_pos, uint32_t>> &pCells, mgs_pos pPos, uint32_t pCell) {
  auto vIt = std::lower_bound(pCells.begin(), pCells.end(), std::make_pair(pPos, pCell));
  return vIt != pCells.end() && vIt->second == pCell && vIt->first == pPos ? vIt - pCells.begin() : -1;
}

} // namespace

mgs_memory_hybrid_idx::~mgs_memory_hybrid_idx() = default;

const mgs_memory_hybrid_idx::kpool_layout &mgs_memory_hybrid_idx::mKpoolLayoutGet() const {
  GGML_ASSERT(vKpoolLay != nullptr);

  return *vKpoolLay;
}

// pools are fixed by the positions relative to the sequence's first one, so the layout survives a plain
// append. a sequence edit can regroup them, and aMemIdxStale tells us it happened
const mgs_memory_hybrid_idx::kpool_layout &mgs_memory_hybrid_idx::mKpoolLayoutUpdate() {
  GGML_ASSERT(cMemIdx != nullptr);

  if (!vKpoolLay) {
    vKpoolLay = std::make_unique<kpool_layout>();
  }

  auto &vLay = *vKpoolLay;

  const uint32_t cKpool = mGetKpool();
  const uint32_t cNStreamKv = cMemIdx->mGetNStream();
  const bool cUnified = cNStreamKv == 1;

  vLay.vNPoolReal = 0;

  for (mgs_seq_id vS = 0; vS < MGS_MAX_SEQ; ++vS) {
    auto &vSq = vLay.aSeqs[vS];

    // a non unified cache gives each sequence its own stream, with stream local cell indices
    if (!cUnified && vS >= (mgs_seq_id)cNStreamKv) {
      vSq = kpool_layout::seq();
      continue;
    }

    const auto &vCells = cMemIdx->mGetCells(cUnified ? 0 : vS);
    const auto &vSp = vCells.mSeqPosGet(vS);

    vSq.vStrm = cUnified ? 0 : cMemIdx->mGetStream(vS);

    if (aMemIdxStale[vS] == cPosClean && !vSq.lCells.empty() && !vSp.empty() && vSq.vPosMin == vSp.begin()->first) {
      for (auto vIt = vSp.upper_bound(vSq.lCells.back()); vIt != vSp.end(); ++vIt) {
        vSq.lCells.push_back(*vIt);
      }
    }

    // the appended tail accounts for every cell only if nothing before it was dropped, but an edit can
    // regroup a sequence without changing its cell count, so a stale sequence must rebuild regardless
    if (vSq.lCells.size() != vSp.size() || aMemIdxStale[vS] != cPosClean) {
      vSq.lCells.assign(vSp.begin(), vSp.end());
      vSq.lPools.clear();
      vSq.vJNext = 0;
      vSq.vPosMin = vSp.empty() ? 0 : vSp.begin()->first;
    }

    // pools start at the first valid token
    size_t vJ = vSq.vJNext;
    if (vHparamsIdx.vIndexerKpoolByOrder) {
      // consecutive cells in sequence order, whatever their positions
      for (; vJ + cKpool <= vSq.lCells.size(); vJ += cKpool) {
        vSq.lPools.push_back((uint32_t)vJ);
      }
    } else {
      while (vJ + cKpool <= vSq.lCells.size()) {
        const mgs_pos cP0 = vSq.lCells[vJ].first;
        if ((cP0 - vSq.vPosMin) % (mgs_pos)cKpool != 0) {
          ++vJ;
          continue;
        }
        bool vOk = true;
        for (uint32_t vK = 1; vK < cKpool; ++vK) {
          if (vSq.lCells[vJ + vK].first != cP0 + (mgs_pos)vK) {
            vOk = false;
            break;
          }
        }
        if (vOk) {
          vSq.lPools.push_back((uint32_t)vJ);
          vJ += cKpool;
        } else {
          ++vJ;
        }
      }
    }
    vSq.vJNext = vJ;

    vLay.vNPoolReal += (uint32_t)vSq.lPools.size();
  }

  return vLay;
}

mgs_memory_hybrid_idx_context::mgs_memory_hybrid_idx_context(mgs_memory_status pStatus)
    : mgs_memory_hybrid_context(pStatus) {}

mgs_memory_hybrid_idx_context::mgs_memory_hybrid_idx_context(mgs_memory_hybrid_idx *pMem)
    : mgs_memory_hybrid_context(pMem), vMem(pMem),
      // graph reservation walks a full context, and qwen4exp builds the sparse attention only when this is set
      // without it the reserved worst case is the dense graph, so ggml-alloc must grow the buffer on the first decode
      lNsUbatch(pMem->mGetMemIdx() == nullptr ? std::vector<uint32_t>()
                                              : std::vector<uint32_t>{pMem->mGetMemIdx()->mGetNStream()}),
      cCtxIdx(pMem->mGetMemIdx() == nullptr ? nullptr : new mgs_kv_cache_context(pMem->mGetMemIdx())) {
  if (mKpoolTrack()) {
    pMem->mKpoolLayoutUpdate();
    auto vSt = mKpoolBuildSizes();
    const auto *vIdx = pMem->mGetMemIdx();
    const uint64_t cNPoolMax = uint64_t(vIdx->mGetSize() / pMem->mGetKpool()) * vIdx->mGetNSeqMax();
    GGML_ASSERT(cNPoolMax <= UINT32_MAX - 64);
    vSt.vNPoolReal = std::max(vSt.vNPoolReal, uint32_t(cNPoolMax));
    vSt.vNNew = vSt.vNPoolReal;
    vSt.vNNewG = std::max(vSt.vNNew, 1u);
    vKpoolSt = std::make_unique<kpool_state>(std::move(vSt));
    vIKpool = 0;
  }
}

mgs_memory_hybrid_idx_context::mgs_memory_hybrid_idx_context(mgs_memory_hybrid_idx *pMem, mgs_context *pLctx,
                                                             bool pOptimize)
    : mgs_memory_hybrid_context(pMem, pLctx, pOptimize), vMem(pMem),
      // the update applies a pending cross-stream seq_cp, else the copy keeps stale indexer keys
      cCtxIdx(pMem->mGetMemIdx() == nullptr ? nullptr : pMem->mGetMemIdx()->mInitUpdate(pLctx, pOptimize)) {}

mgs_memory_hybrid_idx_context::mgs_memory_hybrid_idx_context(mgs_memory_hybrid_idx *pMem,
                                                             slot_info_vec_t pSinfosAttn, slot_info_vec_t pSinfosIdx,
                                                             std::vector<mgs_ubatch> pUbatches)
    // note: the base copies the ubatches; cCtxIdx gets a copy of its own
    : mgs_memory_hybrid_context(pMem, std::move(pSinfosAttn), pUbatches), vMem(pMem),
      lNsUbatch(fMemoryHybridIdxNs(pSinfosIdx)),
      lSinfosKpool(pMem->mGetMemIdx() != nullptr && pMem->mGetKpool() > 0 && pMem->mGetKpoolByOrder()
                       ? pSinfosIdx
                       : slot_info_vec_t()),
      cCtxIdx(pMem->mGetMemIdx() == nullptr
                  ? nullptr
                  : new mgs_kv_cache_context(pMem->mGetMemIdx(), std::move(pSinfosIdx), pUbatches)) {
  // sequence edits force the touched positions to re-pool
  aMemIdxStaleBatch = pMem->mMemIdxStaleGet();
}

mgs_memory_hybrid_idx_context::~mgs_memory_hybrid_idx_context() = default;

bool mgs_memory_hybrid_idx_context::mNext() {
  // clear only after a successful ubatch
  if (vICur == 0 && vMem != nullptr) {
    vMem->mMemIdxStaleClear();
  }

  if (cCtxIdx) {
    cCtxIdx->mNext();
  }

  ++vICur;

  return mgs_memory_hybrid_context::mNext();
}

bool mgs_memory_hybrid_idx_context::mApply() {
  bool vRes = mgs_memory_hybrid_context::mApply();

  if (cCtxIdx) {
    vRes = vRes & cCtxIdx->mApply();
  }

  // extend the pool layout with this ubatch's cells, then pick what it must re-pool
  if (vRes && mKpoolTrack()) {
    vMem->mKpoolLayoutUpdate();
    if (!vKpoolSt) {
      vKpoolSt = std::make_unique<kpool_state>();
    }
    mKpoolBuildState(mGetUbatch());
    vIKpool = vICur;
  }

  return vRes;
}

bool mgs_memory_hybrid_idx_context::mKpoolTrack() const {
  // derived from vMem instead of being cached
  return vMem != nullptr && vMem->mGetMemIdx() != nullptr && vMem->mGetKpool() > 0 && !lNsUbatch.empty();
}

const mgs_kv_cache_context *mgs_memory_hybrid_idx_context::mGetIdx() const {
  return static_cast<const mgs_kv_cache_context *>(cCtxIdx.get());
}

uint32_t mgs_memory_hybrid_idx_context::mGetNStream() const {
  GGML_ASSERT(vICur < lNsUbatch.size());

  return lNsUbatch[vICur];
}

mgs_memory_hybrid_idx_context::kpool_access::kpool_access(ggml_context *pCtx, ggml_tensor *pK, int64_t pNEmbd)
    : vCtx(pCtx) {
  // rows are the per-token part (glm5-next: key | gate, qwen4exp: key), then the pooled key
  const int64_t cNTok = pK->ne[0] - pNEmbd;
  GGML_ASSERT(cNTok > 0 && cNTok % pNEmbd == 0);

  const int64_t cNCells = pK->ne[1] * pK->ne[2];

  // pool indices can refer to other streams, so these views cover the full storage
  vKeyGate = ggml_view_2d(pCtx, pK, cNTok, cNCells, pK->nb[1], 0);
  vPooled = ggml_view_2d(pCtx, pK, pNEmbd, cNCells, pK->nb[1], ggml_row_size(pK->type, cNTok));
}

ggml_tensor *mgs_memory_hybrid_idx_context::kpool_access::mGatherKeyGate(ggml_tensor *pIdxs) const {
  return ggml_get_rows(vCtx, vKeyGate, pIdxs);
}

ggml_tensor *mgs_memory_hybrid_idx_context::kpool_access::mScatterPooled(ggml_tensor *pValues,
                                                                         ggml_tensor *pIdxs) const {
  return ggml_set_rows(vCtx, vPooled, pValues, pIdxs);
}

ggml_tensor *mgs_memory_hybrid_idx_context::kpool_access::mGatherPooled(ggml_tensor *pIdxs) const {
  return ggml_get_rows(vCtx, vPooled, pIdxs);
}

mgs_memory_hybrid_idx_context::kpool_access mgs_memory_hybrid_idx_context::mGetKpoolAccess(ggml_context *pCtx,
                                                                                           int32_t pIl,
                                                                                           int64_t pNEmbd) const {
  GGML_ASSERT(vMem != nullptr && vMem->mGetMemIdx() != nullptr);

  return kpool_access(pCtx, vMem->mGetMemIdx()->mGetKStorage(pIl), pNEmbd);
}

// k-pool DSA indexer (glm5-next, qwen4exp QSA)

// sizes only, used by the full cache context so mGetNKpool() works during graph reserve
mgs_memory_hybrid_idx_context::kpool_state mgs_memory_hybrid_idx_context::mKpoolBuildSizes() const {
  const auto &vLay = vMem->mKpoolLayoutGet();

  kpool_state vSt;
  vSt.vNPoolReal = vLay.vNPoolReal;

  return vSt;
}

// which pools this ubatch must re-pool
// pool cache lifecycle:
// 1. cpy_k writes each token's key | gate into its idx cache row, pooled slot are zeroed.
// 2. this marks the pools the ubatch touches or completes as new, during decode that's one pool every kpool tokens,
//    zero elsewise.
// 3. the graph pools only the new pools and set_rows each result into the pooled slot of the pool's last member row.
// 4. all pools are gathered in one get_rows via pool_cells, fresh ones just written, older ones from whatever batch
//    last wrote them.
// a seq_* edit regroups the pools from the edited position on, so it stales them and the first ubatch of the next
// batch rebuilds them from the still-valid key | gate rows, rewriting the (possibly different) rep rows.
// orphaned pooled slots are never cleared, a slot is only ever read through pool_cells, which follows the current
// grouping.
void mgs_memory_hybrid_idx_context::mKpoolBuildState(const mgs_ubatch &pUbatch) {
  const auto &vLay = vMem->mKpoolLayoutGet();
  auto &vSt = *vKpoolSt;

  const auto *vIdx = vMem->mGetMemIdx();
  const uint32_t cKvSize = vIdx->mGetSize();
  const uint32_t cKpool = vMem->mGetKpool();

  vSt.vNPoolReal = vLay.vNPoolReal;
  vSt.vNNew = 0;
  if (++vSt.vGeneration == 0) {
    std::fill(vSt.lIsNew.begin(), vSt.lIsNew.end(), 0);
    std::fill(vSt.lRepGen.begin(), vSt.lRepGen.end(), 0);
    vSt.vGeneration = 1;
  }
  vSt.lIsNew.resize(vLay.vNPoolReal, 0);
  vSt.lRepGen.resize((size_t)cKvSize * vIdx->mGetNStream(), 0);

  std::array<uint32_t, MGS_MAX_SEQ> aPoolStart;

  // a pool is marked once per rep: sequences sharing cells (a seq_cp, or tokens decoded for several sequences)
  // share their pools, whose single pooled row they all read through pool_cells, so the scatter rows stay unique
  auto vMark = [&](mgs_seq_id pS, size_t pK) {
    const auto &vSq = vLay.aSeqs[pS];
    const size_t cRep = (size_t)vSq.vStrm * cKvSize + vSq.lCells[vSq.lPools[pK] + cKpool - 1].second;
    if (vSt.lRepGen[cRep] != vSt.vGeneration) {
      vSt.lRepGen[cRep] = vSt.vGeneration;
      vSt.lIsNew[aPoolStart[pS] + pK] = vSt.vGeneration;
      ++vSt.vNNew;
    }
  };

  uint32_t vIp = 0;
  for (mgs_seq_id vS = 0; vS < MGS_MAX_SEQ; ++vS) {
    const auto &vSq = vLay.aSeqs[vS];
    aPoolStart[vS] = vIp;
    vIp += (uint32_t)vSq.lPools.size();

    // a sequence edit invalidates only pools ending after the edited position
    const mgs_pos cStaleFrom = vICur == 0 ? aMemIdxStaleBatch[vS] : mgs_memory_hybrid_idx::cPosClean;
    if (cStaleFrom == mgs_memory_hybrid_idx::cPosClean) {
      continue;
    }

    auto vFirst = std::lower_bound(vSq.lPools.begin(), vSq.lPools.end(), cStaleFrom,
                                   [&](uint32_t pJ, mgs_pos pP) { return vSq.lCells[pJ + cKpool - 1].first < pP; });
    for (auto vIt = vFirst; vIt != vSq.lPools.end(); ++vIt) {
      vMark(vS, vIt - vSq.lPools.begin());
    }
  }
  GGML_ASSERT(vIp == vSt.lIsNew.size());

  // in order mode a token's cell gives its rank, and the rank its pool: positions cannot, as an image shares one
  const bool cByOrder = vMem->mGetKpoolByOrder();
  const auto *vSinfo = cByOrder ? &lSinfosKpool[vICur] : nullptr;
  const uint32_t cNTps = cByOrder ? (uint32_t)vSinfo->size() : 0;

  for (uint32_t vI = 0; vI < pUbatch.vNTokens; ++vI) {
    const mgs_pos cP = pUbatch.vPos[vI];
    for (int32_t vK = 0; vK < pUbatch.vNSeqId[vI]; ++vK) {
      const mgs_seq_id cS = pUbatch.vSeqId[vI][vK];
      const auto &vSq = vLay.aSeqs[cS];
      if (cByOrder) {
        const int64_t cR = fKpoolRank(vSq.lCells, cP, vSinfo->lIdxs[vI / cNTps][vI % cNTps]);
        GGML_ASSERT(cR >= 0);
        if ((size_t)cR / cKpool < vSq.lPools.size()) {
          vMark(cS, (size_t)cR / cKpool);
        }
        continue;
      }
      auto vIt = std::upper_bound(vSq.lPools.begin(), vSq.lPools.end(), cP,
                                  [&](mgs_pos pPos, uint32_t pJ) { return pPos < vSq.lCells[pJ].first; });
      if (vIt == vSq.lPools.begin()) {
        continue;
      }
      --vIt;
      if (cP <= vSq.lCells[*vIt + cKpool - 1].first) {
        vMark(cS, vIt - vSq.lPools.begin());
      }
    }
  }

  // a ubatch touches at most t_s/kpool + 1 pools per sequence, pad to that bound so the graph keeps its shape
  // as the count moves; reserve sizes the list for every pool the cache can hold, so never pad past n_pool_max
  const uint32_t cNPoolMax = cKvSize / cKpool * vIdx->mGetNSeqMax();
  const uint32_t cBound = pUbatch.vNTokens / cKpool + pUbatch.vNSeqsUnq;
  vSt.vNNewG = std::max({vSt.vNNew, 1u, std::min({cBound, fKpoolPad(vSt.vNPoolReal) - 1, cNPoolMax})});
}

const mgs_memory_hybrid_idx_context::kpool_state &mgs_memory_hybrid_idx_context::mKpoolCur() const {
  GGML_ASSERT(vKpoolSt != nullptr && vIKpool == vICur && "k-pool state read before mApply()");

  return *vKpoolSt;
}

uint32_t mgs_memory_hybrid_idx_context::mGetNKpool() const { return fKpoolPad(mKpoolCur().vNPoolReal); }

uint32_t mgs_memory_hybrid_idx_context::mGetNKpoolNew() const { return mKpoolCur().vNNewG; }

void mgs_memory_hybrid_idx_context::mSetInputKpool(ggml_tensor *pPoolCells, ggml_tensor *pPoolIdxs,
                                                   ggml_tensor *pPoolMask, ggml_tensor *pTailIdxs,
                                                   ggml_tensor *pSelMask, ggml_tensor *pNewPoolIdxs,
                                                   ggml_tensor *pNewPoolRep, const mgs_ubatch *pUbatch,
                                                   ggml_tensor *pNewPoolPos) const {
  GGML_ASSERT(vMem != nullptr && vMem->mGetMemIdx() != nullptr);
  GGML_ASSERT(ggml_backend_buffer_is_host(pPoolCells->buffer));
  GGML_ASSERT(ggml_backend_buffer_is_host(pPoolIdxs->buffer));
  GGML_ASSERT(ggml_backend_buffer_is_host(pPoolMask->buffer));
  GGML_ASSERT(ggml_backend_buffer_is_host(pTailIdxs->buffer));

  const uint32_t cKpool = vMem->mGetKpool();
  const uint32_t cNKv = mGetIdx()->mGetNKv();

  const auto &vSt = mKpoolCur();
  const auto &vLay = vMem->mKpoolLayoutGet();

  const uint32_t cNTokens = pUbatch->vNTokens;
  const uint32_t cNPool = (uint32_t)pPoolCells->ne[0];
  const uint32_t cNNew = vSt.vNNew;
  // the graph always pools at least one entry, padded to a stable bound, see mKpoolBuildState
  const uint32_t cNNewG = vSt.vNNewG;

  const bool cByOrder = vMem->mGetKpoolByOrder();

  GGML_ASSERT(cNPool == fKpoolPad(vSt.vNPoolReal));
  GGML_ASSERT(vSt.lIsNew.size() == vSt.vNPoolReal);
  GGML_ASSERT(pPoolMask->ne[0] == (int64_t)cNPool && pPoolMask->ne[1] == (int64_t)cNTokens);
  GGML_ASSERT(pTailIdxs->ne[0] == (int64_t)cKpool - 1 && pTailIdxs->ne[1] == (int64_t)cNTokens);
  GGML_ASSERT(pPoolIdxs->ne[0] == (int64_t)cKpool && pPoolIdxs->ne[1] == (int64_t)cNPool);
  GGML_ASSERT(ggml_backend_buffer_is_host(pNewPoolIdxs->buffer));
  GGML_ASSERT(pNewPoolIdxs->ne[0] == (int64_t)cKpool && pNewPoolIdxs->ne[1] == (int64_t)cNNewG);
  // the graph always scatters the fresh pooled keys back into the cache
  GGML_ASSERT(pNewPoolRep != nullptr && ggml_backend_buffer_is_host(pNewPoolRep->buffer));
  GGML_ASSERT(pNewPoolRep->ne[0] == (int64_t)cNNewG);
  if (pNewPoolPos != nullptr) {
    GGML_ASSERT(ggml_backend_buffer_is_host(pNewPoolPos->buffer));
    GGML_ASSERT(pNewPoolPos->ne[0] == 4 * (int64_t)cNNewG);
  }

  const uint32_t cKvSize = vMem->mGetMemIdx()->mGetSize();
  const uint32_t cNStreamKv = vMem->mGetMemIdx()->mGetNStream();

  auto vGcell = [&](const mgs_memory_hybrid_idx::kpool_layout::seq &pSq, uint32_t pCell) {
    return (int64_t)pSq.vStrm * cKvSize + pCell;
  };

  // sequences present in this ubatch, pools of absent sequences must fall on the scatter sentinel row
  std::vector<uint8_t> lSeqInUb(MGS_MAX_SEQ, 0);
  for (uint32_t vI = 0; vI < cNTokens; ++vI) {
    for (int32_t vK = 0; vK < pUbatch->vNSeqId[vI]; ++vK) {
      lSeqInUb[pUbatch->vSeqId[vI][vK]] = 1;
    }
  }

  // a cell of this ubatch, written before any read, so the padded pools read a finite K row
  int64_t vDummyCell = 0;
  {
    const mgs_seq_id cS = pUbatch->vSeqId[0][0];
    const auto &vSq = vLay.aSeqs[cS];
    auto vIt = std::lower_bound(vSq.lCells.begin(), vSq.lCells.end(), std::make_pair(pUbatch->vPos[0], 0u));
    GGML_ASSERT(vIt != vSq.lCells.end() && vIt->first == pUbatch->vPos[0]);
    vDummyCell = vGcell(vSq, vIt->second);
  }

  // in order mode a token sees the pools and the tail up to its own rank in the sequence, which its cell pins down
  std::vector<int64_t> lRank;
  if (cByOrder) {
    const auto &vSinfo = lSinfosKpool[vICur];
    const uint32_t cNTps = (uint32_t)vSinfo.size();

    lRank.resize(cNTokens);
    for (uint32_t vI = 0; vI < cNTokens; ++vI) {
      lRank[vI] = fKpoolRank(vLay.aSeqs[pUbatch->vSeqId[vI][0]].lCells, pUbatch->vPos[vI],
                             vSinfo.lIdxs[vI / cNTps][vI % cNTps]);
      GGML_ASSERT(lRank[vI] >= 0);
    }
  }

  // padding and absent cells point at the n_kv sentinel row, one past the live cells
  const int32_t cSentinel = (int32_t)cNKv;

  float *vGm = nullptr;
  uint32_t vNSel = 0;
  uint32_t vNTop = 0; // pools per token in the selection
  if (pSelMask != nullptr) {
    GGML_ASSERT(ggml_backend_buffer_is_host(pSelMask->buffer));
    GGML_ASSERT(pSelMask->type == GGML_TYPE_F32);
    GGML_ASSERT(pSelMask->ne[3] == (int64_t)cNTokens && pSelMask->ne[1] == 1 && pSelMask->ne[2] == 1);
    vNSel = (uint32_t)pSelMask->ne[0];
    // the tail slots, when selected, are the n_sel % kpool != 0 remainder
    vNTop = vNSel / cKpool;
    GGML_ASSERT(vNSel % cKpool == 0 || vNSel % cKpool == cKpool - 1);
    vGm = (float *)pSelMask->data;
  }

  // pools are laid out per sequence
  std::vector<uint32_t> lSeqPoolStart(MGS_MAX_SEQ, 0);
  std::vector<mgs_pos> lPoolEnd;
  lPoolEnd.reserve(cNPool);

  int32_t *vPcell = (int32_t *)pPoolCells->data;
  int32_t *vPidx = (int32_t *)pPoolIdxs->data;
  int32_t *vNidx = (int32_t *)pNewPoolIdxs->data;
  int64_t *vNrep = (int64_t *)pNewPoolRep->data;
  int32_t *vNpos = pNewPoolPos != nullptr ? (int32_t *)pNewPoolPos->data : nullptr;

  if (vNpos != nullptr) {
    std::fill(vNpos, vNpos + 4 * cNNewG, 0);
  }

  uint32_t vINew = 0;
  for (mgs_seq_id vS = 0; vS < MGS_MAX_SEQ; ++vS) {
    const auto &vSq = vLay.aSeqs[vS];
    lSeqPoolStart[vS] = (uint32_t)lPoolEnd.size();

    const bool cInert = cNStreamKv > 1 && !lSeqInUb[vS];

    for (size_t vPi = 0; vPi < vSq.lPools.size(); ++vPi) {
      const uint32_t cJ = vSq.lPools[vPi];
      const uint32_t cIp = (uint32_t)lPoolEnd.size();
      GGML_ASSERT(cIp + 1 < cNPool);

      // the pooled key lives in the last member's row
      const uint32_t cRep = vSq.lCells[cJ + cKpool - 1].second;
      vPcell[cIp] = (int32_t)vGcell(vSq, cRep);

      for (uint32_t vK = 0; vK < cKpool; ++vK) {
        vPidx[(size_t)cIp * cKpool + vK] = cInert ? cSentinel : (int32_t)vSq.lCells[cJ + vK].second;
      }

      if (vSt.lIsNew[cIp] == vSt.vGeneration) {
        GGML_ASSERT(vINew < cNNew);
        for (uint32_t vK = 0; vK < cKpool; ++vK) {
          vNidx[(size_t)vINew * cKpool + vK] = (int32_t)vGcell(vSq, vSq.lCells[cJ + vK].second);
        }
        vNrep[vINew] = vGcell(vSq, cRep);
        if (vNpos != nullptr) {
          // a pooled key is rotated to the M-RoPE position of its first member
          const uint32_t cC = vSq.lCells[cJ].second;
          const auto &vE = vMem->mGetMemIdx()->mGetCells(vS).mExtGet(cC);
          vNpos[0 * cNNewG + vINew] = vSq.lCells[cJ].first;
          vNpos[1 * cNNewG + vINew] = vE.vY;
          vNpos[2 * cNNewG + vINew] = vE.vX;
          vNpos[3 * cNNewG + vINew] = vSq.lCells[cJ].first;
        }
        ++vINew;
      }

      lPoolEnd.push_back(vSq.lCells[cJ + cKpool - 1].first);
    }
  }
  GGML_ASSERT(vINew == cNNew);

  // padded entries re-pool cells whose pooled slot is never read: only the reps of complete pools are read
  // each entry takes its own cell, entries sharing one would write it from several threads in the scatter
  if (cNNewG > cNNew) {
    std::vector<int64_t> lReps(vPcell, vPcell + lPoolEnd.size());
    std::sort(lReps.begin(), lReps.end());

    int64_t vPadCell = 0;
    for (uint32_t vI = cNNew; vI < cNNewG; ++vI, ++vPadCell) {
      while (std::binary_search(lReps.begin(), lReps.end(), vPadCell)) {
        ++vPadCell;
      }
      GGML_ASSERT(vPadCell < (int64_t)cKvSize * cNStreamKv);
      for (uint32_t vK = 0; vK < cKpool; ++vK) {
        vNidx[(size_t)vI * cKpool + vK] = (int32_t)vPadCell;
      }
      vNrep[vI] = vPadCell;
    }
  }

  const uint32_t cNPoolReal = (uint32_t)lPoolEnd.size();
  for (uint32_t vIp = cNPoolReal; vIp < cNPool; ++vIp) {
    vPcell[vIp] = (int32_t)vDummyCell; // pool_cells always addresses the K storage
    for (uint32_t vK = 0; vK < cKpool; ++vK) {
      vPidx[(size_t)vIp * cKpool + vK] = cSentinel;
    }
  }

  // a pool is visible when it belongs to the token's sequence and ends at or before it
  auto vFillMask = [&](auto *pData) {
    using T = std::remove_pointer_t<decltype(pData)>;
    const T cKeep = mgs_cast<T>(0.0f);
    const T cDrop = mgs_cast<T>(-INFINITY);

    for (uint32_t vI = 0; vI < cNTokens; ++vI) {
      const mgs_seq_id cS = pUbatch->vSeqId[vI][0];
      const mgs_pos cP = pUbatch->vPos[vI];

      T *vRow = pData + (size_t)vI * cNPool;
      std::fill(vRow, vRow + cNPool, cDrop);

      const uint32_t cP0 = lSeqPoolStart[cS];
      const uint32_t cP1 = cP0 + (uint32_t)vLay.aSeqs[cS].lPools.size();
      const uint32_t cNv =
          cByOrder ? std::min(cP1 - cP0, (uint32_t)((lRank[vI] + 1) / cKpool))
                   : (uint32_t)(std::upper_bound(lPoolEnd.begin() + cP0, lPoolEnd.begin() + cP1, cP) -
                                (lPoolEnd.begin() + cP0));
      std::fill(vRow + cP0, vRow + cP0 + cNv, cKeep);

      // finite visible pools occupy the first min(nv, n_top) ranked slots
      if (vGm != nullptr) {
        const uint32_t cNvc = std::min(cNv, vNTop);
        float *vGrow = vGm + (size_t)vI * vNSel;
        std::fill(vGrow, vGrow + (size_t)cNvc * cKpool, 0.0f);
        std::fill(vGrow + (size_t)cNvc * cKpool, vGrow + (size_t)vNTop * cKpool, -INFINITY);
      }
    }
  };
  if (pPoolMask->type == GGML_TYPE_F16) {
    vFillMask((ggml_fp16_t *)pPoolMask->data);
  } else {
    vFillMask((float *)pPoolMask->data);
  }

  int32_t *vTidx = (int32_t *)pTailIdxs->data;
  for (uint32_t vI = 0; vI < cNTokens; ++vI) {
    const mgs_seq_id cS = pUbatch->vSeqId[vI][0];
    const mgs_pos cP = pUbatch->vPos[vI];
    const auto &vSq = vLay.aSeqs[cS];

    const uint32_t cNTail = cByOrder ? (uint32_t)((lRank[vI] + 1) % cKpool)
                                     : (uint32_t)((cP - vSq.vPosMin + 1) % (mgs_pos)cKpool);

    for (uint32_t vK = 0; vK < cKpool - 1; ++vK) {
      int32_t vCell = cSentinel;
      bool vReal = false;
      if (vK < cNTail && cByOrder) {
        const uint32_t cC = vSq.lCells[lRank[vI] - vK].second;
        vCell = (int32_t)cC;
        vReal = true;
      } else if (vK < cNTail) {
        const mgs_pos cPt = cP - (mgs_pos)vK;
        auto vIt = std::lower_bound(vSq.lCells.begin(), vSq.lCells.end(), std::make_pair(cPt, 0u));
        if (vIt != vSq.lCells.end() && vIt->first == cPt) {
          vCell = (int32_t)vIt->second;
          vReal = true;
        }
      }
      vTidx[(size_t)vI * (cKpool - 1) + vK] = vCell;

      if (vGm != nullptr && vNSel % cKpool != 0) {
        vGm[(size_t)vI * vNSel + (size_t)vNTop * cKpool + vK] = vReal ? 0.0f : -INFINITY;
      }
    }
  }
}
