#include "mgs-memory-recurrent.h"

#include "ggml-backend.h"
#include "mgs-impl.h"
#include "mgs-io.h"
#include "mgs-batch.h"
#include "mgs-model.h"

#include <algorithm>
#include <cassert>
#include <cstring>
#include <limits>
#include <map>
#include <stdexcept>

//
// mgs_memory_recurrent
//

mgs_memory_recurrent::mgs_memory_recurrent(const mgs_model &pModel, ggml_type pTypeR, ggml_type pTypeS, bool pOffload,
                                           uint32_t pMemSize, uint32_t pNSeqMax, uint32_t pNRsSeq,
                                           const layer_filter_cb &pFilter)
    : vHparams(pModel.vHparams), cNSeqMax(pNSeqMax) {
  const int32_t cNLayer = vHparams.mNLayer();

  vHead = 0;
  vSize = pMemSize;
  vUsed = 0;

  this->vNRsSeq = pNRsSeq;
  lRsIdx.assign(pNSeqMax, 0);

  lCells.clear();
  lCells.resize(pMemSize);

  // define a comparator for the buft -> ctx map to ensure that the order is well-defined:
  struct ggml_backend_buft_comparator {
    bool operator()(const ggml_backend_buffer_type_t &pLhs, const ggml_backend_buffer_type_t &pRhs) const {
      return strcmp(ggml_backend_buft_name(pLhs), ggml_backend_buft_name(pRhs)) < 0;
    }
  };
  std::map<ggml_backend_buffer_type_t, ggml_context_ptr, ggml_backend_buft_comparator> dCtxMap;

  // create a context for each buffer type
  auto vCtxForBuft = [&](ggml_backend_buffer_type_t pBuft) -> ggml_context * {
    auto vIt = dCtxMap.find(pBuft);
    if (vIt == dCtxMap.end()) {
      ggml_init_params vParams = {
          // r and s per layer, plus the separate PLE conv row where the model has one
          /*.mem_size   =*/size_t((vHparams.mPleConvState() > 0 ? 3u : 2u) * cNLayer * ggml_tensor_overhead()),
          /*.mem_buffer =*/NULL,
          /*.no_alloc   =*/true,
      };

      ggml_context *vCtx = ggml_init(vParams);
      if (!vCtx) {
        return nullptr;
      }

      dCtxMap.emplace(pBuft, vCtx);

      return vCtx;
    }

    return vIt->second.get();
  };

  lRL.resize(cNLayer);
  lSL.resize(cNLayer);
  lPL.resize(cNLayer);

  for (int vI = 0; vI < cNLayer; vI++) {
    if (pFilter && !pFilter(vI)) {
      MGS_LOG_DEBUG("%s: layer %3d: skipped\n", __func__, vI);
      continue;
    }

    const char *vDevName = "CPU";

    ggml_backend_buffer_type_t vBuft = ggml_backend_cpu_buffer_type();

    if (pOffload) {
      auto *vDev = pModel.mDevLayer(vI);
      vBuft = ggml_backend_dev_buffer_type(vDev);

      vDevName = ggml_backend_dev_name(vDev);
    }

    MGS_LOG_DEBUG("%s, layer %3d: dev = %s\n", __func__, vI, vDevName);

    ggml_context *vCtx = vCtxForBuft(vBuft);
    if (!vCtx) {
      throw std::runtime_error("failed to create ggml context for rs cache");
    }

    const uint32_t cNRows = pMemSize * (1 + pNRsSeq);
    ggml_tensor *vR = ggml_new_tensor_2d(vCtx, pTypeR, vHparams.mNEmbdR(), cNRows);
    ggml_tensor *vS = ggml_new_tensor_2d(vCtx, pTypeS, vHparams.mNEmbdS(), cNRows);
    ggml_format_name(vR, "cache_r_l%d", vI);
    ggml_format_name(vS, "cache_s_l%d", vI);
    lRL[vI] = vR;
    lSL[vI] = vS;

    // the PLE history needs its own row: a tensor split must mirror it while the delta-net conv state next door
    // stays split
    if (vHparams.mPleConvState() > 0 && vHparams.mIsPle(vI)) {
      ggml_tensor *vP = ggml_new_tensor_2d(vCtx, pTypeR, vHparams.mPleConvState(), cNRows);
      ggml_format_name(vP, "cache_ple_r_l%d", vI);
      lPL[vI] = vP;
    }
  }

  // allocate tensors and initialize the buffers to avoid NaNs in the padding
  for (auto &[vBuft, vCtx] : dCtxMap) {
    ggml_backend_buffer_t vBuf = ggml_backend_alloc_ctx_tensors_from_buft(vCtx.get(), vBuft);
    if (!vBuf) {
      throw std::runtime_error("failed to allocate buffer for rs cache");
    }
    ggml_backend_buffer_clear(vBuf, 0);
    MGS_LOG_INFO("%s: %10s RS buffer size = %8.2f MiB\n", __func__, ggml_backend_buffer_name(vBuf),
                 ggml_backend_buffer_get_size(vBuf) / 1024.0 / 1024.0);
    lCtxsBufs.emplace_back(std::move(vCtx), vBuf);
  }

  if (mIsEmpty()) {
    if (vNRsSeq > 0) {
      vNRsSeq = 0;
      MGS_LOG_INFO("%s: disabling rollback snapshots because the memory module is empty\n", __func__);
    }
  }

  {
    const size_t cMemorySizeR = mSizeRBytes();
    const size_t cMemorySizeS = mSizeSBytes();
    const size_t cMemorySizeP = mSizePBytes();

    MGS_LOG_INFO("%s: size = %7.2f MiB (%6u cells, %3d layers, %2u seqs %2u rs_seq), R (%s): %7.2f MiB, "
                 "S (%s): %7.2f MiB, P (%s): %7.2f MiB\n",
                 __func__, (float)(cMemorySizeR + cMemorySizeS + cMemorySizeP) / (1024.0f * 1024.0f), pMemSize,
                 cNLayer, pNSeqMax, vNRsSeq, ggml_type_name(pTypeR), (float)cMemorySizeR / (1024.0f * 1024.0f),
                 ggml_type_name(pTypeS), (float)cMemorySizeS / (1024.0f * 1024.0f), ggml_type_name(pTypeR),
                 (float)cMemorySizeP / (1024.0f * 1024.0f));
  }
}

void mgs_memory_recurrent::mClear(bool pData) {
  for (int32_t vI = 0; vI < (int32_t)vSize; ++vI) {
    lCells[vI].vPos = -1;
    lCells[vI].sSeqId.clear();
    lCells[vI].vSrc = -1;
    lCells[vI].vTail = -1;
  }

  vHead = 0;
  vUsed = 0;

  if (pData) {
    for (auto &[_, vBuf] : lCtxsBufs) {
      ggml_backend_buffer_clear(vBuf.get(), 0);
    }
  }

  std::fill(lRsIdx.begin(), lRsIdx.end(), 0);
}

bool mgs_memory_recurrent::mSeqRm(mgs_seq_id pSeqId, mgs_pos p0, mgs_pos p1) {
  uint32_t vNewHead = vSize;

  if (p0 < 0) {
    p0 = 0;
  }

  if (p1 < 0) {
    p1 = std::numeric_limits<mgs_pos>::max();
  }

  const bool cRmAll = p0 == 0 && p1 == std::numeric_limits<mgs_pos>::max();
  if (cRmAll) {
    if (pSeqId >= 0) {
      mSetRsIdx(pSeqId, 0);
    } else {
      std::fill(lRsIdx.begin(), lRsIdx.end(), 0);
    }
  }

  // models like Mamba or RWKV can't have a state partially erased at the end
  // of the sequence because their state isn't preserved for previous tokens
  if (pSeqId >= (int64_t)vSize) {
    // could be fatal
    return false;
  }
  if (0 <= pSeqId) {
    int32_t &vTailId = lCells[pSeqId].vTail;
    if (vTailId >= 0) {
      auto &vCell = lCells[vTailId];

      // partial rollback via per-token snapshot index (bounded by n_rs_seq)
      if (0 < p0 && p0 <= vCell.vPos && p1 > vCell.vPos) {
        // a cell shared by several sequences cannot move back for only one of them
        if (vCell.sSeqId.size() > 1) {
          return false;
        }
        // the filter kept no layer (e.g. an MTP draft context), so only the position moves back
        if (mIsEmpty()) {
          vCell.vPos = p0 - 1;
          return true;
        }
        const mgs_pos cRollback = vCell.vPos - (p0 - 1);
        if (cRollback >= 1 && cRollback <= (mgs_pos)vNRsSeq) {
          mSetRsIdx(pSeqId, (uint32_t)cRollback);
          vCell.vPos = p0 - 1;
          return true;
        }
        return false;
      }
      // invalidate tails which will be cleared
      if (p0 <= vCell.vPos && vCell.vPos < p1) {
        vTailId = -1;
      }
    }
  } else {
    // seq_id is negative, then the range should include everything or nothing
    if (p0 != p1 && (p0 != 0 || p1 != std::numeric_limits<mgs_pos>::max())) {
      // printf("[DEBUG] inside `mgs_memory_recurrent::mSeqRm`: `seq_id` is negative, so returning false\n");
      return false;
    }
  }

  for (uint32_t vI = 0; vI < vSize; ++vI) {
    if (lCells[vI].vPos >= p0 && lCells[vI].vPos < p1) {
      if (pSeqId < 0) {
        lCells[vI].sSeqId.clear();
      } else if (lCells[vI].mHasSeqId(pSeqId)) {
        lCells[vI].sSeqId.erase(pSeqId);
      } else {
        continue;
      }
      if (lCells[vI].mIsEmpty()) {
        // keep count of the number of used cells
        if (lCells[vI].vPos >= 0) {
          vUsed--;
        }
        lCells[vI].vPos = -1;
        lCells[vI].vSrc = -1;
        if (vNewHead == vSize) {
          vNewHead = vI;
        }
      }
    }
  }

  // If we freed up a slot, set head to it so searching can start there.
  if (vNewHead != vSize && vNewHead < vHead) {
    vHead = vNewHead;
  }

  return true;
}

void mgs_memory_recurrent::mSeqCp(mgs_seq_id pSeqIdSrc, mgs_seq_id pSeqIdDst, mgs_pos p0, mgs_pos p1) {
  if (pSeqIdSrc == pSeqIdDst) {
    return;
  }

  if (p0 < 0) {
    p0 = 0;
  }

  if (p1 < 0) {
    p1 = std::numeric_limits<mgs_pos>::max();
  }

  if ((uint32_t)pSeqIdDst < vSize && (uint32_t)pSeqIdSrc < vSize) {
    auto &vTailSrc = lCells[pSeqIdSrc];
    auto &vTailDst = lCells[pSeqIdDst];
    if (vTailDst.vTail >= 0) {
      // clear destination seq_id if it wasn't empty
      auto &vCellDst = lCells[vTailDst.vTail];

      vCellDst.sSeqId.erase(pSeqIdDst);
      vTailDst.vTail = -1;
      if (vCellDst.sSeqId.empty()) {
        vCellDst.vPos = -1;
        vCellDst.vSrc = -1;
        vUsed -= 1;
      }
    }
    if (vTailSrc.vTail >= 0) {
      auto &vCellSrc = lCells[vTailSrc.vTail];

      vCellSrc.sSeqId.insert(pSeqIdDst);
      vTailDst.vTail = vTailSrc.vTail;
    }
  }
}

void mgs_memory_recurrent::mSeqKeep(mgs_seq_id pSeqId) {
  uint32_t vNewHead = vSize;

  for (uint32_t vI = 0; vI < vSize; ++vI) {
    if ((mgs_seq_id)vI != pSeqId) {
      lCells[vI].vTail = -1;
    }

    if (!lCells[vI].mHasSeqId(pSeqId)) {
      if (lCells[vI].vPos >= 0) {
        vUsed--;
      }

      lCells[vI].vPos = -1;
      lCells[vI].vSrc = -1;
      lCells[vI].sSeqId.clear();

      if (vNewHead == vSize) {
        vNewHead = vI;
      }
    } else {
      lCells[vI].sSeqId.clear();
      lCells[vI].sSeqId.insert(pSeqId);
    }
  }

  // If we freed up a slot, set head to it so searching can start there.
  if (vNewHead != vSize && vNewHead < vHead) {
    vHead = vNewHead;
  }
}

void mgs_memory_recurrent::mSeqAdd(mgs_seq_id pSeqId, mgs_pos p0, mgs_pos p1, mgs_pos pShift) {
  if (pShift == 0) {
    return;
  }

  if (p0 < 0) {
    p0 = 0;
  }

  if (p1 < 0) {
    p1 = std::numeric_limits<mgs_pos>::max();
  }

  // If there is no range then return early to avoid looping over the
  if (p0 == p1) {
    return;
  }

  // for Mamba-like or RWKV models, only the pos needs to be shifted
  if (0 <= pSeqId && pSeqId < (int64_t)vSize) {
    const int32_t cTailId = lCells[pSeqId].vTail;
    if (cTailId >= 0) {
      auto &vCell = lCells[cTailId];
      if (vCell.mHasSeqId(pSeqId) && p0 <= vCell.vPos && vCell.vPos < p1) {
        vCell.vPos += pShift;
      }
    }
  }
}

void mgs_memory_recurrent::mSeqDiv(mgs_seq_id pSeqId, mgs_pos p0, mgs_pos p1, int pD) {
  if (pD == 1) {
    return;
  }

  if (p0 < 0) {
    p0 = 0;
  }

  if (p1 < 0) {
    p1 = std::numeric_limits<mgs_pos>::max();
  }

  // If there is no range then return early to avoid looping over the cache.
  if (p0 == p1) {
    return;
  }

  // for Mamba-like or RWKV models, only the pos needs to be changed
  if (0 <= pSeqId && pSeqId < (int64_t)vSize) {
    const int32_t cTailId = lCells[pSeqId].vTail;
    if (cTailId >= 0) {
      auto &vCell = lCells[cTailId];
      if (vCell.mHasSeqId(pSeqId) && p0 <= vCell.vPos && vCell.vPos < p1) {
        vCell.vPos /= pD;
      }
    }
  }
}

mgs_pos mgs_memory_recurrent::mSeqPosMin(mgs_seq_id pSeqId) const {
  mgs_pos vResult = std::numeric_limits<mgs_pos>::max();

  for (uint32_t vI = 0; vI < vSize; ++vI) {
    if (lCells[vI].mHasSeqId(pSeqId)) {
      vResult = std::min(vResult, lCells[vI].vPos);
    }
  }

  if (vResult == std::numeric_limits<mgs_pos>::max()) {
    vResult = -1;
  }

  return vResult;
}

mgs_pos mgs_memory_recurrent::mSeqPosMax(mgs_seq_id pSeqId) const {
  mgs_pos vResult = -1;

  for (uint32_t vI = 0; vI < vSize; ++vI) {
    if (lCells[vI].mHasSeqId(pSeqId)) {
      vResult = std::max(vResult, lCells[vI].vPos);
    }
  }

  return vResult;
}

void mgs_memory_recurrent::mSetRsIdx(mgs_seq_id pSeqId, uint32_t pIdx) {
  if (pSeqId < 0 || (size_t)pSeqId >= lRsIdx.size()) {
    return;
  }
  lRsIdx[pSeqId] = (pIdx > vNRsSeq) ? vNRsSeq : pIdx;
}

std::map<ggml_backend_buffer_type_t, size_t> mgs_memory_recurrent::mMemoryBreakdown() const {
  std::map<ggml_backend_buffer_type_t, size_t> dRet;
  for (const auto &[_, vBuf] : lCtxsBufs) {
    dRet[ggml_backend_buffer_get_type(vBuf.get())] += ggml_backend_buffer_get_size(vBuf.get());
  }
  return dRet;
}

mgs_memory_context_ptr mgs_memory_recurrent::mInitBatch(mgs_batch_allocr &pBalloc, uint32_t pNUbatch, bool pEmbdAll) {
  do {
    pBalloc.mSplitReset();

    std::vector<mgs_ubatch> lUbatches;
    while (true) {
      mgs_ubatch vUbatch;

      if (pEmbdAll) {
        // if all tokens are output, split by sequence
        vUbatch = pBalloc.mSplitSeq(pNUbatch);
      } else {
        // TODO: non-sequential equal split can be done if using unified KV cache
        //       for simplicity, we always use sequential equal split for now
        // [TAG_RECURRENT_ROLLBACK_SPLITS]
        // the trailing (1 + n_rs_seq) tokens of each seq must stay in the same ubatch
        //   so that the rollback snapshots remain valid
        vUbatch = pBalloc.mSplitEqual(pNUbatch, true, vNRsSeq > 0 ? vNRsSeq + 1 : 0);
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

    if (!mPrepare(lUbatches)) {
      break;
    }

    return std::make_unique<mgs_memory_recurrent_context>(this, std::move(lUbatches));
  } while (false);

  return std::make_unique<mgs_memory_recurrent_context>(cMgsMemoryStatusFailedPrepare);
}

mgs_memory_context_ptr mgs_memory_recurrent::mInitFull() {
  return std::make_unique<mgs_memory_recurrent_context>(this);
}

mgs_memory_context_ptr mgs_memory_recurrent::mInitUpdate(mgs_context *pLctx, bool pOptimize) {
  GGML_UNUSED(pLctx);
  GGML_UNUSED(pOptimize);

  return std::make_unique<mgs_memory_recurrent_context>(cMgsMemoryStatusNoUpdate);
}

bool mgs_memory_recurrent::mPrepare(const std::vector<mgs_ubatch> &pUbatches) {
  // simply remember the full state because it is very small for this type of cache
  // TODO: optimize
  auto vOrgCells = lCells;
  auto vOrgUsed = vUsed;
  auto vOrgHead = vHead;

  bool vSuccess = true;

  for (const auto &vUbatch : pUbatches) {
    if (!mFindSlot(vUbatch)) {
      vSuccess = false;
      break;
    }
  }

  // restore the original state
  lCells = std::move(vOrgCells);
  vUsed = vOrgUsed;
  vHead = vOrgHead;

  return vSuccess;
}

bool mgs_memory_recurrent::mFindSlot(const mgs_ubatch &pUbatch) {
  const uint32_t cNSeqTokens = pUbatch.vNSeqTokens;
  const uint32_t cNSeqs = pUbatch.vNSeqs;

  // if we have enough unused cells before the current head ->
  //   better to start searching from the beginning of the cache, hoping to fill it
  if (vHead > vUsed + 2 * cNSeqs) {
    vHead = 0;
  }

  // For recurrent state architectures (like Mamba or RWKV),
  // each cache cell can store the state for a whole sequence.
  // A slot should be always be contiguous.

  // can only process batches with an equal number of new tokens in each sequence
  GGML_ASSERT(pUbatch.mEqualSeqs());

  int32_t vMin = vSize - 1;
  int32_t vMax = 0;

  // everything should fit if all seq_ids are smaller than the max
  for (uint32_t vS = 0; vS < cNSeqs; ++vS) {
    const uint32_t cI = vS * cNSeqTokens; // first token of sequence set s
    const uint32_t cNSeqId = pUbatch.vNSeqId[cI];

    for (uint32_t vJ = 0; vJ < cNSeqId; ++vJ) {
      const mgs_seq_id cSeqId = pUbatch.vSeqId[cI][vJ];

      if (cSeqId < 0 || (uint32_t)cSeqId >= vSize) {
        // too big seq_id
        // TODO: would it be possible to resize the cache instead?
        MGS_LOG_ERROR("%s: seq_id=%d >= n_seq_max=%u Try using a bigger --parallel value\n", __func__, cSeqId,
                      cNSeqMax);
        return false;
      }
      if (vJ > 0) {
        auto &vSeq = lCells[cSeqId];
        if (vSeq.vTail >= 0) {
          auto &vCell = lCells[vSeq.vTail];
          // clear cells from seq_ids that become shared
          // (should not normally happen, but let's handle it anyway)
          vCell.sSeqId.erase(cSeqId);
          vSeq.vTail = -1;
          if (vCell.sSeqId.empty()) {
            vCell.vPos = -1;
            vCell.vSrc = -1;
            vUsed -= 1;
          }
        }
      }
    }
  }

#ifndef NDEBUG
  {
    std::vector<int32_t> lTailsVerif;
    lTailsVerif.assign(vSize, -1);
    for (uint32_t vI = 0; vI < vSize; ++vI) {
      auto &vCell = lCells[vI];
      for (mgs_seq_id vSeqId : vCell.sSeqId) {
        if (lTailsVerif[vSeqId] != -1) {
          MGS_LOG_ERROR("%s: duplicate tail for seq_id %d in cell %d and %d\n", __func__, vSeqId, vI,
                        lTailsVerif[vSeqId]);
        }
        lTailsVerif[vSeqId] = vI;
      }
    }
    for (uint32_t vI = 0; vI < vSize; ++vI) {
      if (lTailsVerif[vI] != lCells[vI].vTail) {
        MGS_LOG_ERROR("%s: wrong tail for seq_id %d, (%d instead of %d)\n", __func__, vI, lCells[vI].vTail,
                      lTailsVerif[vI]);
      }
    }
  }
#endif

  // find next empty cell
  uint32_t vNextEmptyCell = vHead;

  for (uint32_t vI = 0; vI < vSize; ++vI) {
    if (vNextEmptyCell >= vSize) {
      vNextEmptyCell -= vSize;
    }
    auto &vCell = lCells[vNextEmptyCell];
    if (vCell.mIsEmpty()) {
      break;
    }
    vNextEmptyCell += 1;
  }

  // find usable cell range
  for (uint32_t vS = 0; vS < cNSeqs; ++vS) {
    const uint32_t cI = vS * cNSeqTokens;
    const mgs_seq_id cSeqId = pUbatch.vSeqId[cI][0];
    auto &vSeqMeta = lCells[cSeqId];
    bool vHasCell = false;
    if (vSeqMeta.vTail >= 0) {
      auto &vCell = lCells[vSeqMeta.vTail];
      GGML_ASSERT(vCell.mHasSeqId(cSeqId));
      // does this seq_id "own" the cell?
      if (vCell.sSeqId.size() == 1) {
        vHasCell = true;
      }
    }
    if (!vHasCell) {
      auto &vEmptyCell = lCells[vNextEmptyCell];
      GGML_ASSERT(vEmptyCell.mIsEmpty());
      // copy old tail into the empty cell
      if (vSeqMeta.vTail >= 0) {
        auto &vOrigCell = lCells[vSeqMeta.vTail];
        vEmptyCell.vPos = vOrigCell.vPos;
        vEmptyCell.vSrc = vOrigCell.vSrc;
        vOrigCell.sSeqId.erase(cSeqId);
        vEmptyCell.sSeqId.insert(cSeqId);   // will be overwritten
        GGML_ASSERT(!vOrigCell.mIsEmpty()); // has at least one remaining seq_id
      }
      vSeqMeta.vTail = vNextEmptyCell;
      // find next empty cell
      if (vS + 1 < cNSeqs) {
        for (uint32_t vJ = 0; vJ < vSize; ++vJ) {
          vNextEmptyCell += 1;
          if (vNextEmptyCell >= vSize) {
            vNextEmptyCell -= vSize;
          }
          auto &vCell = lCells[vNextEmptyCell];
          if (vCell.mIsEmpty()) {
            break;
          }
        }
      }
    }
    if (vMin > vSeqMeta.vTail) {
      vMin = vSeqMeta.vTail;
    }
    if (vMax < vSeqMeta.vTail) {
      vMax = vSeqMeta.vTail;
    }
  }

  // gather and re-order
  for (uint32_t vS = 0; vS < cNSeqs; ++vS) {
    const uint32_t cI = vS * cNSeqTokens;
    const int32_t cDstId = vS + vMin;
    const int32_t cSrcId = lCells[pUbatch.vSeqId[cI][0]].vTail;
    if (cDstId != cSrcId) {
      auto &vDstCell = lCells[cDstId];
      auto &vSrcCell = lCells[cSrcId];

      std::swap(vDstCell.vPos, vSrcCell.vPos);
      std::swap(vDstCell.vSrc, vSrcCell.vSrc);
      std::swap(vDstCell.sSeqId, vSrcCell.sSeqId);

      // swap tails
      for (uint32_t vJ = 0; vJ < vSize; ++vJ) {
        int32_t &vTail = lCells[vJ].vTail;
        if (vTail == cSrcId) {
          vTail = cDstId;
        } else if (vTail == cDstId) {
          vTail = cSrcId;
        }
      }
    }
  }

  // update the pos of the used seqs
  for (uint32_t vS = 0; vS < cNSeqs; ++vS) {
    const uint32_t cI = vS * cNSeqTokens;
    const mgs_pos cLastPos = pUbatch.vPos[cI + cNSeqTokens - 1];
    const int32_t cCellId = vS + vMin;
    auto &vCell = lCells[cCellId];

    if (vCell.vPos >= 0 && cLastPos != vCell.vPos + (mgs_pos)cNSeqTokens) {
      // What should happen when the pos backtracks or skips a value?
      // Clearing the state mid-batch would require special-casing which isn't done.
      MGS_LOG_WARN("%s: non-consecutive token position %d after %d for sequence %d with %u new tokens\n", __func__,
                   cLastPos, vCell.vPos, pUbatch.vSeqId[cI][0], cNSeqTokens);
    }
    vCell.vPos = cLastPos;
    vCell.sSeqId.clear();
    for (int32_t vJ = 0; vJ < pUbatch.vNSeqId[cI]; ++vJ) {
      const mgs_seq_id cSeqId = pUbatch.vSeqId[cI][vJ];
      vCell.sSeqId.insert(cSeqId);
      lCells[cSeqId].vTail = cCellId;
    }
  }

  // Find first cell without src refs, to use as the zero-ed state
  {
    // TODO: bake-in src refcounts in the cell metadata
    std::vector<int32_t> lRefcounts(vSize, 0);
    for (size_t vI = 0; vI < vSize; ++vI) {
      const int32_t cSrc = lCells[vI].vSrc;
      if (cSrc >= 0) {
        lRefcounts[cSrc] += 1;
      }
    }

    vRsZ = -1;
    for (int vI = vMin; vI <= vMax; ++vI) {
      if (lRefcounts[vI] == 0) {
        vRsZ = vI;
        break;
      }
    }

    for (int vI = vMin; vI <= vMax; ++vI) {
      if (lCells[vI].vSrc < 0) {
        GGML_ASSERT(vRsZ >= 0);
        lCells[vI].vSrc0 = vRsZ;
      } else {
        // Stage the source ids for all used cells to allow correct seq_* behavior
        // and still make these values available when setting the inputs
        lCells[vI].vSrc0 = lCells[vI].vSrc;
      }
      lCells[vI].vSrc = vI; // avoid moving or clearing twice
    }
  }

  // allow getting the range of used cells, from head to head + n
  vHead = vMin;
  vN = vMax - vMin + 1;
  vUsed = std::count_if(lCells.begin(), lCells.end(), [](const mem_cell &pCell) { return !pCell.mIsEmpty(); });

  // sanity check
  return vN >= cNSeqs;
}

bool mgs_memory_recurrent::mGetCanShift() const {
  // shifting the pos is trivial for recurrent models
  return true;
}

bool mgs_memory_recurrent::mIsEmpty() const { return lCtxsBufs.empty(); }

size_t mgs_memory_recurrent::mTotalSize() const {
  size_t vSize = 0;
  for (const auto &[_, vBuf] : lCtxsBufs) {
    vSize += ggml_backend_buffer_get_size(vBuf.get());
  }

  return vSize;
}

size_t mgs_memory_recurrent::mSizeRBytes() const {
  size_t vSizeRBytes = 0;

  for (const auto &vR : lRL) {
    if (vR != nullptr) {
      vSizeRBytes += ggml_nbytes(vR);
    }
  }

  return vSizeRBytes;
}

size_t mgs_memory_recurrent::mSizePBytes() const {
  size_t vSizePBytes = 0;

  for (const auto &vP : lPL) {
    if (vP != nullptr) {
      vSizePBytes += ggml_nbytes(vP);
    }
  }

  return vSizePBytes;
}

size_t mgs_memory_recurrent::mSizeSBytes() const {
  size_t vSizeSBytes = 0;

  for (const auto &vS : lSL) {
    if (vS != nullptr) {
      vSizeSBytes += ggml_nbytes(vS);
    }
  }

  return vSizeSBytes;
}

void mgs_memory_recurrent::mStateWrite(mgs_io_write_i &pIo, mgs_seq_id pSeqId, mgs_state_seq_flags pFlags) const {
  GGML_UNUSED(pFlags);

  std::vector<std::pair<uint32_t, uint32_t>> lCellRanges;     // ranges, from inclusive, to exclusive
  std::vector<std::pair<uint32_t, uint32_t>> lCellRangesData; // logical source row ranges
  uint32_t vCellCount = 0;

  // Count the number of cells with the specified seq_id
  // Find all the ranges of cells with this seq id (or all, when -1)
  uint32_t vCellRangeBegin = vSize;
  for (uint32_t vI = 0; vI < vSize; ++vI) {
    const auto &vCell = lCells[vI];
    if ((pSeqId == -1 && !vCell.mIsEmpty()) || vCell.mHasSeqId(pSeqId)) {
      ++vCellCount;
      uint32_t vRsIdxCur = 0;

      if (vNRsSeq != 0) {
        if (pSeqId != -1) {
          GGML_ASSERT(pSeqId >= 0 && (size_t)pSeqId < lRsIdx.size());
          vRsIdxCur = lRsIdx[pSeqId];
        } else {
          bool vHasRsIdx = false;
          for (const mgs_seq_id cCellSeqId : vCell.sSeqId) {
            GGML_ASSERT(cCellSeqId >= 0 && (size_t)cCellSeqId < lRsIdx.size());

            const uint32_t cSeqRsIdx = lRsIdx[cCellSeqId];
            if (!vHasRsIdx) {
              vRsIdxCur = cSeqRsIdx;
              vHasRsIdx = true;
            } else if (vRsIdxCur != cSeqRsIdx) {
              GGML_ABORT("cannot write shared recurrent state with different rollback indices");
            }
          }
        }
      }

      const uint32_t cCellId = vRsIdxCur * vSize + (vCell.vSrc >= 0 ? vCell.vSrc : (int32_t)vI);
      if (lCellRangesData.empty() || lCellRangesData.back().second != cCellId) {
        lCellRangesData.emplace_back(cCellId, cCellId + 1);
      } else {
        lCellRangesData.back().second++;
      }

      if (vCellRangeBegin == vSize) {
        vCellRangeBegin = vI;
      }
    } else {
      if (vCellRangeBegin != vSize) {
        lCellRanges.emplace_back(vCellRangeBegin, vI);
        vCellRangeBegin = vSize;
      }
    }
  }
  if (vCellRangeBegin != vSize) {
    lCellRanges.emplace_back(vCellRangeBegin, vSize);
  }

  if ((pFlags & MGS_STATE_SEQ_FLAGS_ON_DEVICE) && lCellRanges.size() > 1) {
    GGML_ABORT("cannot save/load multiple ranges of cells to/from device memory\n");
  }

  // DEBUG CHECK: Sum of cell counts in ranges should equal the total cell count
  uint32_t vCellCountCheck = 0;
  for (const auto &vRange : lCellRanges) {
    vCellCountCheck += vRange.second - vRange.first;
  }
  GGML_ASSERT(vCellCount == vCellCountCheck);

  vCellCountCheck = 0;
  for (const auto &vRange : lCellRangesData) {
    vCellCountCheck += vRange.second - vRange.first;
  }
  GGML_ASSERT(vCellCount == vCellCountCheck);

  pIo.mWrite(&vCellCount, sizeof(vCellCount));

  mStateWriteMeta(pIo, lCellRanges, pSeqId);
  mStateWriteData(pIo, lCellRangesData);
}

void mgs_memory_recurrent::mStateRead(mgs_io_read_i &pIo, mgs_seq_id pSeqId, mgs_state_seq_flags pFlags) {
  GGML_UNUSED(pFlags);

  uint32_t vCellCount;
  pIo.mRead(&vCellCount, sizeof(vCellCount));

  bool vRes = true;

  vRes = vRes && mStateReadMeta(pIo, vCellCount, pSeqId);

  try {
    vRes = vRes && mStateReadData(pIo, vCellCount);
  } catch (...) {
    vRes = false;
  }

  if (!vRes) {
    if (pSeqId == -1) {
      mClear(true);
    } else {
      mSeqRm(pSeqId, -1, -1);
    }
    throw std::runtime_error("failed to restore kv cache");
  }

  if (vNRsSeq != 0) {
    if (pSeqId == -1) {
      std::fill(lRsIdx.begin(), lRsIdx.end(), 0);
    } else {
      mSetRsIdx(pSeqId, 0);
    }
  }
}

void mgs_memory_recurrent::mStateWriteMeta(mgs_io_write_i &pIo,
                                           const std::vector<std::pair<uint32_t, uint32_t>> &pCellRanges,
                                           mgs_seq_id pSeqId) const {
  for (const auto &vRange : pCellRanges) {
    for (uint32_t vI = vRange.first; vI < vRange.second; ++vI) {
      const auto &vCell = lCells[vI];
      const mgs_pos cPos = vCell.vPos;
      const uint32_t cNSeqId = pSeqId == -1 ? vCell.sSeqId.size() : 0;

      pIo.mWrite(&cPos, sizeof(cPos));
      pIo.mWrite(&cNSeqId, sizeof(cNSeqId));

      if (cNSeqId) {
        for (auto vSeqId : vCell.sSeqId) {
          pIo.mWrite(&vSeqId, sizeof(vSeqId));
        }
      }
    }
  }
}

void mgs_memory_recurrent::mStateWriteData(mgs_io_write_i &pIo,
                                           const std::vector<std::pair<uint32_t, uint32_t>> &pCellRanges) const {
  const uint32_t cSTrans = 0;
  const uint32_t cNLayer = vHparams.mNLayer();

  pIo.mWrite(&cSTrans, sizeof(cSTrans));
  pIo.mWrite(&cNLayer, sizeof(cNLayer));

  // Iterate and write all the R tensors first, each row is a cell
  // Get whole range at a time
  for (uint32_t vIl = 0; vIl < cNLayer; ++vIl) {
    // skip null layers (read_data will handle this by checking "r_l" and "s_l" for null)
    if (lRL[vIl] == nullptr)
      continue;

    // Write R tensor type
    const int32_t cRTypeI = (int32_t)lRL[vIl]->type;
    pIo.mWrite(&cRTypeI, sizeof(cRTypeI));

    // Write row size of R tensor
    const uint64_t cRSizeRow = ggml_row_size(lRL[vIl]->type, vHparams.mNEmbdR());
    pIo.mWrite(&cRSizeRow, sizeof(cRSizeRow));

    // Write each logical cell row range. With pending recurrent rollback,
    // the logical current state may live in a rollback snapshot plane.
    for (const auto &vRange : pCellRanges) {
      const size_t cRangeSize = vRange.second - vRange.first;
      const size_t cBufSize = cRangeSize * cRSizeRow;
      pIo.mWriteTensor(lRL[vIl], vRange.first * cRSizeRow, cBufSize);
    }

    // the PLE conv history is a second recurrent row, so it has to travel with the first
    if (lPL[vIl] != nullptr) {
      const uint64_t cPSizeRow = ggml_row_size(lPL[vIl]->type, vHparams.mPleConvState());
      pIo.mWrite(&cPSizeRow, sizeof(cPSizeRow));

      for (const auto &vRange : pCellRanges) {
        const size_t cRangeSize = vRange.second - vRange.first;
        pIo.mWriteTensor(lPL[vIl], vRange.first * cPSizeRow, cRangeSize * cPSizeRow);
      }
    }
  }

  if (!cSTrans) {
    for (uint32_t vIl = 0; vIl < cNLayer; ++vIl) {
      // skip null layers (read_data will handle this by checking "r_l" and "s_l" for null)
      if (lSL[vIl] == nullptr)
        continue;

      // Write S tensor type
      const int32_t cSTypeI = (int32_t)lSL[vIl]->type;
      pIo.mWrite(&cSTypeI, sizeof(cSTypeI));

      // Write row size of S tensor
      const uint64_t cSSizeRow = ggml_row_size(lSL[vIl]->type, vHparams.mNEmbdS());
      pIo.mWrite(&cSSizeRow, sizeof(cSSizeRow));

      // Write each logical cell row range. With pending recurrent rollback,
      // the logical current state may live in a rollback snapshot plane.
      for (const auto &vRange : pCellRanges) {
        const size_t cRangeSize = vRange.second - vRange.first;
        const size_t cBufSize = cRangeSize * cSSizeRow;
        pIo.mWriteTensor(lSL[vIl], vRange.first * cSSizeRow, cBufSize);
      }
    }
  } else {
    // When S tensor is transposed, we also need the element size and get the element ranges from each row
    const uint32_t cMemSize = vSize;
    for (uint32_t vIl = 0; vIl < cNLayer; ++vIl) {
      // skip null layers (read_data will handle this by checking "r_l" and "s_l" for null)
      if (lSL[vIl] == nullptr)
        continue;

      const uint32_t cNEmbdS = vHparams.mNEmbdS();

      // Write S tensor type
      const int32_t cSTypeI = (int32_t)lSL[vIl]->type;
      pIo.mWrite(&cSTypeI, sizeof(cSTypeI));

      // Write element size
      const uint32_t cSSizeEl = ggml_type_size(lSL[vIl]->type);
      pIo.mWrite(&cSSizeEl, sizeof(cSSizeEl));

      // Write GQA embedding size
      pIo.mWrite(&cNEmbdS, sizeof(cNEmbdS));

      // For each row, we get the element values of each logical cell
      for (uint32_t vJ = 0; vJ < cNEmbdS; ++vJ) {
        for (const auto &vRange : pCellRanges) {
          const size_t cRangeSize = vRange.second - vRange.first;
          const size_t cSrcOffset = (vRange.first + vJ * cMemSize) * cSSizeEl;
          const size_t cBufSize = cRangeSize * cSSizeEl;
          pIo.mWriteTensor(lSL[vIl], cSrcOffset, cBufSize);
        }
      }
    }
  }
}

bool mgs_memory_recurrent::mStateReadMeta(mgs_io_read_i &pIo, uint32_t pCellCount, mgs_seq_id pDestSeqId) {
  if (pDestSeqId != -1) {
    // single sequence
    mSeqRm(pDestSeqId, -1, -1);

    if (pCellCount == 0) {
      return true;
    }

    mgs_batch_allocr vBalloc(vHparams.mNPosPerEmbd());

    mgs_ubatch vUbatch = vBalloc.mUbatchReserve(pCellCount, 1);

    for (uint32_t vI = 0; vI < pCellCount; ++vI) {
      mgs_pos vPos;
      uint32_t vNSeqId;

      pIo.mRead(&vPos, sizeof(vPos));
      pIo.mRead(&vNSeqId, sizeof(vNSeqId));

      if (vNSeqId != 0) {
        MGS_LOG_ERROR("%s: invalid seq_id-agnostic kv cell\n", __func__);
        return false;
      }

      vUbatch.vPos[vI] = vPos;
    }
    vUbatch.vNSeqId[0] = 1;
    vUbatch.vSeqId[0] = &pDestSeqId;

    if (!mFindSlot(vUbatch)) {
      MGS_LOG_ERROR("%s: failed to find available cells in kv cache\n", __func__);
      return false;
    }

    // DEBUG CHECK: vHead should be our first cell, vHead + pCellCount - 1 should be our last cell (verify seq_id
    // and pos values) Assume that this is one contiguous block of cells
    GGML_ASSERT(vHead + pCellCount <= vSize);
    GGML_ASSERT(lCells[vHead].vPos == vUbatch.vPos[0]);
    GGML_ASSERT(lCells[vHead + pCellCount - 1].vPos == vUbatch.vPos[pCellCount - 1]);
    GGML_ASSERT(lCells[vHead].mHasSeqId(pDestSeqId));
    GGML_ASSERT(lCells[vHead + pCellCount - 1].mHasSeqId(pDestSeqId));
  } else {
    // whole KV cache restore

    if (pCellCount > vSize) {
      MGS_LOG_ERROR("%s: not enough cells in kv cache\n", __func__);
      return false;
    }

    mClear(true);

    for (uint32_t vI = 0; vI < pCellCount; ++vI) {
      auto &vCell = lCells[vI];

      mgs_pos vPos;
      uint32_t vNSeqId;

      pIo.mRead(&vPos, sizeof(vPos));
      pIo.mRead(&vNSeqId, sizeof(vNSeqId));

      vCell.vPos = vPos;

      for (uint32_t vJ = 0; vJ < vNSeqId; ++vJ) {
        mgs_seq_id vSeqId;
        pIo.mRead(&vSeqId, sizeof(vSeqId));

        if (vSeqId < 0 || (uint32_t)vSeqId >= this->cNSeqMax) {
          MGS_LOG_ERROR("%s: invalid seq_id, %d is out of range [0, %u)\n", __func__, vSeqId, this->cNSeqMax);
          return false;
        }

        vCell.sSeqId.insert(vSeqId);

        int32_t &vTail = lCells[vSeqId].vTail;
        if (vTail != -1) {
          MGS_LOG_ERROR("%s: duplicate tail for seq_id %d in cell %d and %d\n", __func__, vSeqId, vI, vTail);
          return false;
        }
        vTail = vI;
      }
    }

    vHead = 0;
    vUsed = pCellCount;
  }

  for (uint32_t vI = 0; vI < pCellCount; ++vI) {
    uint32_t vCellId = vHead + vI;
    // make sure the recurrent states will keep their restored state
    lCells[vCellId].vSrc = vCellId;
  }

  return true;
}

bool mgs_memory_recurrent::mStateReadData(mgs_io_read_i &pIo, uint32_t pCellCount) {
  uint32_t vSTrans;
  uint32_t vNLayer;
  pIo.mRead(&vSTrans, sizeof(vSTrans));
  pIo.mRead(&vNLayer, sizeof(vNLayer));

  if (vNLayer != vHparams.mNLayer()) {
    MGS_LOG_ERROR("%s: mismatched layer count (%u instead of %u)\n", __func__, vNLayer, vHparams.mNLayer());
    return false;
  }
  if (pCellCount > vSize) {
    MGS_LOG_ERROR("%s: not enough cells in kv cache to restore state (%u > %u)\n", __func__, pCellCount, vSize);
    return false;
  }
  if (false != (bool)vSTrans) {
    MGS_LOG_ERROR("%s: incompatible s transposition\n", __func__);
    return false;
  }

  // For each layer, read the keys for each cell, one row is one cell, read as one contiguous block
  for (uint32_t vIl = 0; vIl < vNLayer; ++vIl) {
    // skip null layers
    if (lRL[vIl] == nullptr)
      continue;

    // Read type of key
    int32_t vRTypeIRef;
    pIo.mRead(&vRTypeIRef, sizeof(vRTypeIRef));
    const int32_t cRTypeI = (int32_t)lRL[vIl]->type;
    if (cRTypeI != vRTypeIRef) {
      MGS_LOG_ERROR("%s: mismatched r type (%d != %d, layer %d)\n", __func__, cRTypeI, vRTypeIRef, vIl);
      return false;
    }

    // Read row size of key
    uint64_t vRSizeRowRef;
    pIo.mRead(&vRSizeRowRef, sizeof(vRSizeRowRef));
    const size_t cRSizeRow = ggml_row_size(lRL[vIl]->type, vHparams.mNEmbdR());
    if (cRSizeRow != vRSizeRowRef) {
      MGS_LOG_ERROR("%s: mismatched r row size (%zu != %zu, layer %d)\n", __func__, cRSizeRow, (size_t)vRSizeRowRef,
                    vIl);
      return false;
    }

    if (pCellCount) {
      // Read and set the keys for the whole cell range
      pIo.mReadTensor(lRL[vIl], vHead * cRSizeRow, pCellCount * cRSizeRow);
    }

    if (lPL[vIl] != nullptr) {
      uint64_t vPSizeRowRef;
      pIo.mRead(&vPSizeRowRef, sizeof(vPSizeRowRef));
      const size_t cPSizeRow = ggml_row_size(lPL[vIl]->type, vHparams.mPleConvState());
      if (cPSizeRow != vPSizeRowRef) {
        MGS_LOG_ERROR("%s: mismatched ple row size (%zu != %zu, layer %d)\n", __func__, cPSizeRow,
                      (size_t)vPSizeRowRef, vIl);
        return false;
      }

      if (pCellCount) {
        pIo.mReadTensor(lPL[vIl], vHead * cPSizeRow, pCellCount * cPSizeRow);
      }
    }
  }

  if (!vSTrans) {
    for (uint32_t vIl = 0; vIl < vNLayer; ++vIl) {
      // skip null layers
      if (lSL[vIl] == nullptr)
        continue;

      // Read type of value
      int32_t vSTypeIRef;
      pIo.mRead(&vSTypeIRef, sizeof(vSTypeIRef));
      const int32_t cSTypeI = (int32_t)lSL[vIl]->type;

      if (cSTypeI != vSTypeIRef) {
        MGS_LOG_ERROR("%s: mismatched s type (%d != %d, layer %d)\n", __func__, cSTypeI, vSTypeIRef, vIl);
        return false;
      }

      // Read row size of value
      uint64_t vSSizeRowRef;
      pIo.mRead(&vSSizeRowRef, sizeof(vSSizeRowRef));
      const size_t cSSizeRow = ggml_row_size(lSL[vIl]->type, vHparams.mNEmbdS());
      if (cSSizeRow != vSSizeRowRef) {
        MGS_LOG_ERROR("%s: mismatched s row size (%zu != %zu, layer %d)\n", __func__, cSSizeRow, (size_t)vSSizeRowRef,
                      vIl);
        return false;
      }

      if (pCellCount) {
        // Read and set the values for the whole cell range
        pIo.mReadTensor(lSL[vIl], vHead * cSSizeRow, pCellCount * cSSizeRow);
      }
    }
  } else {
    // For each layer, read the values for each cell (transposed)
    for (uint32_t vIl = 0; vIl < vNLayer; ++vIl) {
      // skip null layers
      if (lSL[vIl] == nullptr)
        continue;

      const uint32_t cNEmbdS = vHparams.mNEmbdS();

      // Read type of value
      int32_t vSTypeIRef;
      pIo.mRead(&vSTypeIRef, sizeof(vSTypeIRef));
      const int32_t cSTypeI = (int32_t)lSL[vIl]->type;
      if (cSTypeI != vSTypeIRef) {
        MGS_LOG_ERROR("%s: mismatched s type (%d != %d, layer %d)\n", __func__, cSTypeI, vSTypeIRef, vIl);
        return false;
      }

      // Read element size of value
      uint32_t vSSizeElRef;
      pIo.mRead(&vSSizeElRef, sizeof(vSSizeElRef));
      const size_t cSSizeEl = ggml_type_size(lSL[vIl]->type);
      if (cSSizeEl != vSSizeElRef) {
        MGS_LOG_ERROR("%s: mismatched s element size (%zu != %zu, layer %d)\n", __func__, cSSizeEl, (size_t)vSSizeElRef,
                      vIl);
        return false;
      }

      // Read state embedding size
      uint32_t vNEmbdSRef;
      pIo.mRead(&vNEmbdSRef, sizeof(vNEmbdSRef));
      if (cNEmbdS != vNEmbdSRef) {
        MGS_LOG_ERROR("%s: mismatched s embedding size (%u != %u, layer %d)\n", __func__, cNEmbdS, vNEmbdSRef, vIl);
        return false;
      }

      if (pCellCount) {
        // For each row in the transposed matrix, read the values for the whole cell range
        for (uint32_t vJ = 0; vJ < cNEmbdS; ++vJ) {
          const size_t cDstOffset = (vHead + vJ * vSize) * cSSizeEl;
          pIo.mReadTensor(lSL[vIl], cDstOffset, pCellCount * cSSizeEl);
        }
      }
    }
  }

  return true;
}

//
// mgs_memory_recurrent_context
//

mgs_memory_recurrent_context::mgs_memory_recurrent_context(mgs_memory_status pStatus) : cStatus(pStatus) {}

mgs_memory_recurrent_context::mgs_memory_recurrent_context(mgs_memory_recurrent *pMem)
    : cStatus(cMgsMemoryStatusSuccess), vMem(pMem), cIsFull(true) {}

mgs_memory_recurrent_context::mgs_memory_recurrent_context(mgs_memory_recurrent *pMem,
                                                           std::vector<mgs_ubatch> pUbatches)
    : cStatus(cMgsMemoryStatusSuccess), vMem(pMem), lUbatches(std::move(pUbatches)) {}

mgs_memory_recurrent_context::~mgs_memory_recurrent_context() = default;

bool mgs_memory_recurrent_context::mNext() {
  assert(cStatus == cMgsMemoryStatusSuccess);

  if (++vINext >= lUbatches.size()) {
    return false;
  }

  return true;
}

bool mgs_memory_recurrent_context::mApply() {
  assert(!mgs_memory_status_is_fail(cStatus));

  // no ubatches -> this is an update
  if (lUbatches.empty()) {
    // recurrent cache never performs updates
    assert(cStatus == cMgsMemoryStatusNoUpdate);

    return true;
  }

  vMem->mFindSlot(lUbatches[vINext]);

  return true;
}

mgs_memory_status mgs_memory_recurrent_context::mGetStatus() const { return cStatus; }

const mgs_ubatch &mgs_memory_recurrent_context::mGetUbatch() const {
  assert(cStatus == cMgsMemoryStatusSuccess);

  return lUbatches[vINext];
}

uint32_t mgs_memory_recurrent_context::mGetNRs() const { return cIsFull ? vMem->vSize : vMem->vN; }

uint32_t mgs_memory_recurrent_context::mGetHead() const { return cIsFull ? 0 : vMem->vHead; }

int32_t mgs_memory_recurrent_context::mGetRsZ() const { return cIsFull ? 0 : vMem->vRsZ; }

uint32_t mgs_memory_recurrent_context::mGetSize() const { return vMem->vSize; }

ggml_tensor *mgs_memory_recurrent_context::mGetRL(int32_t pIl) const { return vMem->lRL[pIl]; }

ggml_tensor *mgs_memory_recurrent_context::mGetSL(int32_t pIl) const { return vMem->lSL[pIl]; }

ggml_tensor *mgs_memory_recurrent_context::mGetPL(int32_t pIl) const { return vMem->lPL[pIl]; }

int32_t mgs_memory_recurrent_context::mSCopy(int pI) const {
  const uint32_t cCellIdx = pI + vMem->vHead;
  const int32_t cSrc0 = vMem->lCells[cCellIdx].vSrc0;

  if (vMem->vNRsSeq == 0) {
    return cSrc0;
  }

  uint32_t vIdx = 0;
  if (!vMem->lCells[cCellIdx].sSeqId.empty()) {
    const mgs_seq_id cSeq = *vMem->lCells[cCellIdx].sSeqId.begin();
    if (cSeq >= 0 && (size_t)cSeq < vMem->lRsIdx.size()) {
      vIdx = vMem->lRsIdx[cSeq];
      // reset rollback idx
      vMem->lRsIdx[cSeq] = 0;
    }
  }
  return (int32_t)(vIdx * vMem->vSize) + cSrc0;
}
