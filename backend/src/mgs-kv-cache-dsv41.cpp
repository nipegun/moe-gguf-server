#include "mgs-kv-cache-dsv41.h"

#include "ggml-backend.h"
#include "mgs-impl.h"
#include "mgs-batch.h"
#include "mgs-io.h"
#include "mgs-model.h"

#include <algorithm>
#include <cassert>
#include <climits>
#include <cstdlib>
#include <cstring>
#include <map>
#include <sstream>
#include <stdexcept>

static constexpr uint32_t cDsv41StateMagic = 0x31345344; // DS41
static constexpr uint32_t cDsv41StateVersion = 1;
static constexpr uint32_t cDsv41StateModeFull = 0;
static constexpr uint32_t cDsv41StateModePartial = 1;
static constexpr uint32_t cDsv41KCacheStateVer = 1;
static constexpr uint32_t cDsv41CompStateVer = 1;

static uint32_t fDsv41CompSize(uint32_t pKvSize, uint32_t pRatio) {
  return std::max<uint32_t>(1, (pKvSize + pRatio - 1) / pRatio);
}

static void fDsv41ClearTensorStream(ggml_tensor *pTensor, uint32_t pStream) {
  GGML_ASSERT(ggml_is_contiguous(pTensor));
  GGML_ASSERT(pTensor->ne[3] == 1);
  GGML_ASSERT(pStream < (uint32_t)pTensor->ne[2]);

  const size_t cStreamSize = pTensor->nb[2];
  ggml_backend_tensor_memset(pTensor, 0, pStream * cStreamSize, cStreamSize);
}

static uint32_t fDsv41StateNUsedKRows(mgs_pos pPosMax, uint32_t pRatio, uint32_t pKvSize) {
  if (pPosMax < 0) {
    return 0;
  }

  const uint64_t cNRows = ((uint64_t)pPosMax + 1) / pRatio;

  return (uint32_t)std::min<uint64_t>(pKvSize, cNRows);
}

static int64_t fDsv41StreamOffset(uint32_t pNStream, mgs_seq_id pSeqId, uint32_t pSize) {
  if (pNStream <= 1) {
    return 0;
  }
  if (pSeqId < 0 || (uint32_t)pSeqId >= pNStream) {
    throw std::runtime_error("DSV41 sequence id out of stream range");
  }

  return (int64_t)pSeqId * pSize;
}

static bool fDsv41UbatchHasCoupled(const mgs_ubatch &pUbatch) {
  for (uint32_t vI = 0; vI < pUbatch.vNTokens; ++vI) {
    if (pUbatch.vNSeqId[vI] > 1) {
      return true;
    }
  }

  return false;
}

static bool fDsv41TokenHasSeq(const mgs_ubatch &pUbatch, uint32_t pI, mgs_seq_id pSeqId) {
  for (int32_t vS = 0; vS < pUbatch.vNSeqId[pI]; ++vS) {
    if (pUbatch.vSeqId[pI][vS] == pSeqId) {
      return true;
    }
  }

  return false;
}

static mgs_ubatch fDsv41BuildRawWriteUbatch(const mgs_ubatch &pUbatch) {
  if (!fDsv41UbatchHasCoupled(pUbatch)) {
    return pUbatch;
  }
  if (pUbatch.vEmbd) {
    throw std::runtime_error("DSV41 coupled embedding ubatches are not supported");
  }

  std::vector<uint32_t> lCounts(pUbatch.vNSeqsUnq, 0);
  uint32_t vNTokens = 0;
  for (uint32_t vS = 0; vS < pUbatch.vNSeqsUnq; ++vS) {
    const mgs_seq_id cSeqId = pUbatch.vSeqIdUnq[vS];
    for (uint32_t vI = 0; vI < pUbatch.vNTokens; ++vI) {
      if (fDsv41TokenHasSeq(pUbatch, vI, cSeqId)) {
        ++lCounts[vS];
        ++vNTokens;
      }
    }
  }

  if (vNTokens == 0) {
    return pUbatch;
  }

  const uint32_t cNSeqTokens = lCounts[0];
  for (uint32_t vS = 1; vS < lCounts.size(); ++vS) {
    if (lCounts[vS] != cNSeqTokens) {
      throw std::runtime_error("DSV41 coupled raw writes require equal sequence lengths");
    }
  }

  auto vData = std::make_shared<mgs_ubatch::data_t>();
  vData->lPos.resize((size_t)vNTokens * pUbatch.vNPos);
  vData->lNSeqId.reserve(vNTokens);
  vData->lSeqId.reserve(vNTokens);
  vData->lSeqIdData.reserve(vNTokens);
  vData->lSeqIdUnq.assign(pUbatch.vSeqIdUnq, pUbatch.vSeqIdUnq + pUbatch.vNSeqsUnq);
  vData->lSeqIdx.assign(MGS_MAX_SEQ, -1);
  vData->lOutput.assign(vNTokens, 0);
  if (pUbatch.vToken) {
    vData->lToken.reserve(vNTokens);
  }

  for (uint32_t vS = 0; vS < vData->lSeqIdUnq.size(); ++vS) {
    vData->lSeqIdx[vData->lSeqIdUnq[vS]] = vS;
  }

  for (uint32_t vS = 0; vS < pUbatch.vNSeqsUnq; ++vS) {
    const mgs_seq_id cSeqId = pUbatch.vSeqIdUnq[vS];
    for (uint32_t vI = 0; vI < pUbatch.vNTokens; ++vI) {
      if (!fDsv41TokenHasSeq(pUbatch, vI, cSeqId)) {
        continue;
      }

      const uint32_t cDst = vData->lNSeqId.size();
      if (pUbatch.vToken) {
        vData->lToken.push_back(pUbatch.vToken[vI]);
      }
      for (uint32_t vP = 0; vP < pUbatch.vNPos; ++vP) {
        vData->lPos[(size_t)vP * vNTokens + cDst] = pUbatch.vPos[(size_t)vP * pUbatch.vNTokens + vI];
      }
      vData->lNSeqId.push_back(1);
      vData->lSeqIdData.push_back(cSeqId);
    }
  }

  for (uint32_t vI = 0; vI < vNTokens; ++vI) {
    vData->lSeqId.push_back(&vData->lSeqIdData[vI]);
  }

  mgs_ubatch vRes{
      /*.vBEqualSeqs  =*/true,
      /*.vNTokens     =*/vNTokens,
      /*.vNSeqTokens  =*/cNSeqTokens,
      /*.vNSeqs       =*/pUbatch.vNSeqsUnq,
      /*.vNSeqsUnq    =*/pUbatch.vNSeqsUnq,
      /*.vNPos        =*/pUbatch.vNPos,
      /*.vToken       =*/vData->lToken.empty() ? nullptr : vData->lToken.data(),
      /*.vEmbd        =*/nullptr,
      /*.vPos         =*/vData->lPos.data(),
      /*.vNSeqId      =*/vData->lNSeqId.data(),
      /*.vSeqId       =*/vData->lSeqId.data(),
      /*.vSeqIdUnq    =*/vData->lSeqIdUnq.data(),
      /*.vSeqIdx      =*/vData->lSeqIdx.data(),
      /*.vOutput      =*/vData->lOutput.data(),
      /*.vData        =*/vData,
  };

  return vRes;
}

static std::vector<mgs_ubatch> fDsv41BuildRawWriteUbatches(const std::vector<mgs_ubatch> &pUbatches) {
  std::vector<mgs_ubatch> lRes;
  lRes.reserve(pUbatches.size());
  for (const mgs_ubatch &vUbatch : pUbatches) {
    lRes.push_back(fDsv41BuildRawWriteUbatch(vUbatch));
  }
  return lRes;
}

static bool fDsv41BatchHasCoupled(const mgs_batch &pBatch) {
  if (!pBatch.n_seq_id) {
    return false;
  }

  for (int32_t vI = 0; vI < pBatch.n_tokens; ++vI) {
    if (pBatch.n_seq_id[vI] > 1) {
      return true;
    }
  }

  return false;
}

static int64_t fDsv41CompGraphNStream(const mgs_ubatch &pUbatch, uint32_t pNStream) {
  // Coupled sequence sets must stay in one graph stream because their
  // compressed state is shared. Independent per-seq state can fan out.
  if (pNStream <= 1 || pUbatch.vNSeqsUnq <= 1 || fDsv41UbatchHasCoupled(pUbatch)) {
    return 1;
  }

  return pUbatch.vNSeqsUnq;
}

static void fDsv41StateSrcStreamRange(uint32_t pNStream, mgs_seq_id pSeqId, uint32_t &pS0, uint32_t &pNs) {
  if (pSeqId >= 0 && pNStream > 1) {
    if ((uint32_t)pSeqId >= pNStream) {
      throw std::runtime_error("DSV41 state sequence id out of stream range");
    }

    pS0 = (uint32_t)pSeqId;
    pNs = 1;
    return;
  }

  pS0 = 0;
  pNs = pSeqId >= 0 ? 1 : pNStream;
}

static void fDsv41StateDstStreamRange(uint32_t pNStream, mgs_seq_id pSeqId, uint32_t pNs, uint32_t &pS0) {
  if (pSeqId >= 0) {
    if (pNs != 1) {
      throw std::runtime_error("DSV41 sequence state stream count mismatch");
    }
    if (pNStream > 1 && (uint32_t)pSeqId >= pNStream) {
      throw std::runtime_error("DSV41 state sequence id out of stream range");
    }

    pS0 = pNStream > 1 ? (uint32_t)pSeqId : 0;
    return;
  }

  if (pNs != pNStream) {
    throw std::runtime_error("DSV41 full state stream count mismatch");
  }

  pS0 = 0;
}

static void fDsv41StateWriteTensorStreams(mgs_io_write_i &pIo, ggml_tensor *pTensor, uint32_t pTensorRows,
                                          uint32_t pNRows, uint32_t pS0, uint32_t pNs,
                                          const std::vector<uint32_t> *pStreamIds = nullptr) {
  const int32_t cTypeI = (int32_t)pTensor->type;
  const uint64_t cNe0 = pTensor->ne[0];
  const uint64_t cRows = pNRows;
  const uint64_t cRowSize = ggml_row_size(pTensor->type, pTensor->ne[0]);

  if (pNRows > pTensorRows) {
    throw std::runtime_error("DSV41 state tensor row count exceeds storage");
  }

  pIo.mWrite(&cTypeI, sizeof(cTypeI));
  pIo.mWrite(&cNe0, sizeof(cNe0));
  pIo.mWrite(&cRows, sizeof(cRows));
  pIo.mWrite(&cRowSize, sizeof(cRowSize));

  const size_t cStreamStride = (size_t)pTensorRows * cRowSize;
  const size_t cSize = (size_t)pNRows * cRowSize;
  if (cSize == 0) {
    return;
  }

  if (pStreamIds && pStreamIds->size() != pNs) {
    throw std::runtime_error("DSV41 state tensor stream map size mismatch");
  }

  for (uint32_t vS = 0; vS < pNs; ++vS) {
    const uint32_t cStream = pStreamIds ? (*pStreamIds)[vS] : pS0 + vS;
    if ((int64_t)cStream >= pTensor->ne[2]) {
      throw std::runtime_error("DSV41 state tensor stream out of range");
    }
    const size_t cOffset = (size_t)cStream * cStreamStride;
    pIo.mWriteTensor(pTensor, cOffset, cSize);
  }
}

static void fDsv41StateReadTensorStreams(mgs_io_read_i &pIo, ggml_tensor *pTensor, uint32_t pTensorRows,
                                         uint32_t pNRows, uint32_t pS0, uint32_t pNs) {
  int32_t vTypeIRef;
  uint64_t vNe0Ref;
  uint64_t vRowsRef;
  uint64_t vRowSizeRef;

  pIo.mRead(&vTypeIRef, sizeof(vTypeIRef));
  pIo.mRead(&vNe0Ref, sizeof(vNe0Ref));
  pIo.mRead(&vRowsRef, sizeof(vRowsRef));
  pIo.mRead(&vRowSizeRef, sizeof(vRowSizeRef));

  const int32_t cTypeI = (int32_t)pTensor->type;
  const uint64_t cNe0 = pTensor->ne[0];
  const uint64_t cRows = pNRows;
  const uint64_t cRowSize = ggml_row_size(pTensor->type, pTensor->ne[0]);

  if (cTypeI != vTypeIRef || cNe0 != vNe0Ref || cRows != vRowsRef || cRowSize != vRowSizeRef) {
    throw std::runtime_error("DSV41 state tensor metadata mismatch");
  }
  if (pNRows > pTensorRows) {
    throw std::runtime_error("DSV41 state tensor row count exceeds storage");
  }

  const size_t cStreamStride = (size_t)pTensorRows * cRowSize;
  const size_t cSize = (size_t)pNRows * cRowSize;
  if (cSize == 0) {
    return;
  }

  for (uint32_t vS = 0; vS < pNs; ++vS) {
    const size_t cOffset = (size_t)(pS0 + vS) * cStreamStride;
    pIo.mReadTensor(pTensor, cOffset, cSize);
  }
}

static void fDsv41StateWriteKCache(mgs_io_write_i &pIo, const mgs_kv_cache *pKv, mgs_seq_id pSeqId,
                                   uint32_t pNRows) {
  uint32_t vS0;
  uint32_t vNs;
  fDsv41StateSrcStreamRange(pKv->mGetNStream(), pSeqId, vS0, vNs);

  const uint32_t cVersion = cDsv41KCacheStateVer;
  const uint32_t cKvSize = pKv->mGetSize();
  const auto cLayerIds = pKv->mGetLayerIds();
  const uint32_t cNLayer = cLayerIds.size();

  if (pNRows > cKvSize) {
    throw std::runtime_error("DSV41 K-cache state row count exceeds cache size");
  }

  pIo.mWrite(&cVersion, sizeof(cVersion));
  pIo.mWrite(&pNRows, sizeof(pNRows));
  pIo.mWrite(&vNs, sizeof(vNs));
  pIo.mWrite(&cNLayer, sizeof(cNLayer));

  for (uint32_t vIl : cLayerIds) {
    pIo.mWrite(&vIl, sizeof(vIl));
    fDsv41StateWriteTensorStreams(pIo, pKv->mGetKStorage(vIl), cKvSize, pNRows, vS0, vNs);
  }
}

static void fDsv41StateReadKCache(mgs_io_read_i &pIo, mgs_kv_cache *pKv, mgs_seq_id pSeqId) {
  uint32_t vVersion;
  uint32_t vNRowsRef;
  uint32_t vNs;
  uint32_t vNLayerRef;

  pIo.mRead(&vVersion, sizeof(vVersion));
  pIo.mRead(&vNRowsRef, sizeof(vNRowsRef));
  pIo.mRead(&vNs, sizeof(vNs));
  pIo.mRead(&vNLayerRef, sizeof(vNLayerRef));

  if (vVersion != cDsv41KCacheStateVer) {
    throw std::runtime_error("DSV41 K-cache state version mismatch");
  }

  const uint32_t cKvSize = pKv->mGetSize();
  if (vNRowsRef > cKvSize) {
    MGS_LOG_INFO("kv rows ref %d kv %d\n", vNRowsRef, cKvSize);
    throw std::runtime_error("DSV41 K-cache state size mismatch");
  }

  uint32_t vS0;
  fDsv41StateDstStreamRange(pKv->mGetNStream(), pSeqId, vNs, vS0);

  const auto cLayerIds = pKv->mGetLayerIds();
  if (vNLayerRef != cLayerIds.size()) {
    throw std::runtime_error("DSV41 K-cache layer count mismatch");
  }

  for (uint32_t vIl : cLayerIds) {
    uint32_t vIlRef;
    pIo.mRead(&vIlRef, sizeof(vIlRef));
    if (vIlRef != vIl) {
      throw std::runtime_error("DSV41 K-cache layer id mismatch");
    }

    fDsv41StateReadTensorStreams(pIo, pKv->mGetKStorage(vIl), cKvSize, vNRowsRef, vS0, vNs);
  }
}

static std::string fDsv41PlanPositions(const std::vector<int32_t> &pValues) {
  std::ostringstream vSs;
  vSs << "[";
  for (size_t vI = 0; vI < pValues.size(); ++vI) {
    if (vI > 0) {
      vSs << ", ";
    }
    vSs << pValues[vI];
  }
  vSs << "]";
  return vSs.str();
}

// The compressor pools `ratio` consecutive tokens without overlap. A completed group becomes one
// compressed row at index pos / ratio, rotated at the position of its first token.
static mgs_kv_cache_dsv41_context::comp_plan fDsv41BuildCompPlan(const mgs_ubatch &pUbatch, uint32_t pRatio,
                                                                 uint32_t pStateSize, uint32_t pKvSize,
                                                                 uint32_t pNStream, uint32_t pNRsSeq,
                                                                 const std::vector<uint32_t> &pRsIdx) {
  mgs_kv_cache_dsv41_context::comp_plan vPlan;
  vPlan.lNVisible.resize(pUbatch.vNTokens);
  vPlan.vNStream = fDsv41CompGraphNStream(pUbatch, pNStream);

  // pNStream is the persistent cache/state layout; vPlan.vNStream is the
  // graph view for this ubatch and can be a subset of those streams.
  if (pNStream <= 1 && pUbatch.vNSeqsUnq > 1) {
    throw std::runtime_error("DSV41 single compressed stream cannot serve multiple sequences");
  }

  const int64_t cStateRows = (int64_t)pStateSize * pNStream;

  struct persist_row {
    int32_t vDst;
    int32_t vSrc;
    mgs_pos vPos;
  };

  std::vector<persist_row> lPersistRows;

  std::map<std::pair<mgs_seq_id, mgs_pos>, int64_t> dCurrTokenIdxMap;
  std::map<mgs_seq_id, uint32_t> dStateWriteCounts;

  for (uint32_t vI = 0; vI < pUbatch.vNTokens; ++vI) {
    for (int32_t vS = 0; vS < pUbatch.vNSeqId[vI]; ++vS) {
      dCurrTokenIdxMap[std::make_pair(pUbatch.vSeqId[vI][vS], pUbatch.vPos[vI])] = vI;
    }
  }

  const auto cStateSourceIdx = [&](mgs_seq_id pSeqId, mgs_pos pPos) -> int32_t {
    const auto cKey = std::make_pair(pSeqId, pPos);
    if (dCurrTokenIdxMap.find(cKey) != dCurrTokenIdxMap.end()) {
      return (int32_t)(cStateRows + dCurrTokenIdxMap.at(cKey));
    }

    const int64_t cStreamOff = fDsv41StreamOffset(pNStream, pSeqId, pStateSize);
    return (int32_t)(cStreamOff + pPos % pStateSize);
  };

  for (uint32_t vI = 0; vI < pUbatch.vNTokens; ++vI) {
    const mgs_pos cPos = pUbatch.vPos[vI];

    if (cPos < 0) {
      continue;
    }

    vPlan.lStatePos.push_back((int32_t)(cPos % pRatio));

    const int64_t cNVisible = (int64_t)(cPos + 1) / pRatio;
    vPlan.lNVisible[vI] = (int32_t)cNVisible;
    vPlan.vNKv = std::max(vPlan.vNKv, cNVisible);

    for (int32_t vS = 0; vS < pUbatch.vNSeqId[vI]; ++vS) {
      const mgs_seq_id cSeqId = pUbatch.vSeqId[vI][vS];
      const int64_t cStreamOff = fDsv41StreamOffset(pNStream, cSeqId, pStateSize);
      const int32_t cStateIdx = (int32_t)(cStreamOff + cPos % pStateSize);

      const auto cIt = std::find_if(lPersistRows.begin(), lPersistRows.end(),
                                    [cStateIdx](const persist_row &pRow) { return pRow.vDst == cStateIdx; });
      if (cIt == lPersistRows.end()) {
        lPersistRows.push_back({cStateIdx, (int32_t)vI, cPos});
      } else if (cPos > cIt->vPos) {
        cIt->vSrc = (int32_t)vI;
        cIt->vPos = cPos;
      }

      if ((cPos + 1) % pRatio != 0) {
        continue;
      }

      const mgs_pos cSourceStart = cPos + 1 - pRatio;
      const int64_t cCacheOff = fDsv41StreamOffset(pNStream, cSeqId, pKvSize);

      vPlan.lStateWriteIdxs.push_back(cCacheOff + cPos / pRatio);
      vPlan.lStateWritePos.push_back((int32_t)cSourceStart);
      ++dStateWriteCounts[cSeqId];

      for (uint32_t vJ = 0; vJ < pRatio; ++vJ) {
        vPlan.lStateReadIdxs.push_back(cStateSourceIdx(cSeqId, cSourceStart + vJ));
      }
    }
  }

  if (!vPlan.lStatePos.empty()) {
    assert(pKvSize > 0);

    // Pad each stream to the reserve plan's row count, so a decode step that completes no row
    // keeps the graph shape. The dummy row targets the last cache row, which cannot be live
    // while a group is still incomplete.
    const auto cAppendDummyBlock = [&](mgs_seq_id pSeqId, uint32_t pI) {
      const int64_t cCacheOff = fDsv41StreamOffset(pNStream, pSeqId, pKvSize);
      const int32_t cSourceIdx = cStateSourceIdx(pSeqId, pUbatch.vPos[pI]);

      vPlan.lStateWriteIdxs.push_back(cCacheOff + pKvSize - 1);
      vPlan.lStateWritePos.push_back(0);

      for (uint32_t vJ = 0; vJ < pRatio; ++vJ) {
        vPlan.lStateReadIdxs.push_back(cSourceIdx);
      }
    };

    if (fDsv41UbatchHasCoupled(pUbatch)) {
      if (vPlan.lStateWriteIdxs.empty()) {
        uint32_t vI = 0;
        while (vI < pUbatch.vNTokens && pUbatch.vPos[vI] < 0) {
          ++vI;
        }
        assert(vI < pUbatch.vNTokens);
        cAppendDummyBlock(pUbatch.vSeqId[vI][0], vI);
      }
    } else {
      const uint32_t cNBlocks = (std::max<uint32_t>(1, pUbatch.vNSeqTokens) + pRatio - 1) / pRatio;

      for (uint32_t vS = 0; vS < pUbatch.vNSeqsUnq; ++vS) {
        const mgs_seq_id cSeqId = pUbatch.vSeqIdUnq[vS];
        const uint32_t cNWrites = dStateWriteCounts[cSeqId];
        if (cNWrites >= cNBlocks) {
          continue;
        }
        if (cNWrites + 1 != cNBlocks) {
          throw std::runtime_error("DSV41 compressed sequence positions are not contiguous");
        }

        uint32_t vI = 0;
        while (vI < pUbatch.vNTokens && (pUbatch.vPos[vI] < 0 || !fDsv41TokenHasSeq(pUbatch, vI, cSeqId))) {
          ++vI;
        }
        assert(vI < pUbatch.vNTokens);
        cAppendDummyBlock(cSeqId, vI);
      }
    }
  }

  vPlan.vNKv = GGML_PAD(vPlan.vNKv, 256u);

  std::sort(lPersistRows.begin(), lPersistRows.end(),
            [](const persist_row &pA, const persist_row &pB) { return pA.vDst < pB.vDst; });

  for (const persist_row &vRow : lPersistRows) {
    vPlan.lStatePersistSrcIdxs.push_back(vRow.vSrc);
    vPlan.lStatePersistDstIdxs.push_back(vRow.vDst);
  }

  if (pNRsSeq > 0) {
    for (uint32_t vS = 0; vS < pUbatch.vNSeqsUnq; ++vS) {
      const mgs_seq_id cSeqId = pUbatch.vSeqIdUnq[vS];
      if (cSeqId < 0 || (uint32_t)cSeqId >= pNStream) {
        continue;
      }

      const int64_t cStreamOff = fDsv41StreamOffset(pNStream, cSeqId, pStateSize);
      const uint32_t cRollback = (uint32_t)cSeqId < pRsIdx.size() ? pRsIdx[cSeqId] : 0;
      // Keep the restore graph fixed-width when no rollback is pending.
      const int64_t cSrcPlane = cRollback > 0 && cRollback <= pNRsSeq ? (int64_t)cRollback * cStateRows : 0;
      for (uint32_t vR = 0; vR < pStateSize; ++vR) {
        vPlan.lStateRestoreSrcIdxs.push_back((int32_t)(cSrcPlane + cStreamOff + vR));
        vPlan.lStateRestoreDstIdxs.push_back((int32_t)(cStreamOff + vR));
      }

      std::vector<uint32_t> lTokenIdxs;
      lTokenIdxs.reserve(pUbatch.vNTokens);
      for (uint32_t vI = 0; vI < pUbatch.vNTokens; ++vI) {
        if (fDsv41TokenHasSeq(pUbatch, vI, cSeqId)) {
          lTokenIdxs.push_back(vI);
        }
      }
      if (lTokenIdxs.empty()) {
        continue;
      }

      const uint32_t cNSeqTokens = (uint32_t)lTokenIdxs.size();
      const int64_t cScratchOff = (int64_t)cStateRows * (1 + pNRsSeq);
      for (uint32_t vD = 1; vD <= pNRsSeq; ++vD) {
        const int64_t cDstPlane = (int64_t)vD * cStateRows;

        for (uint32_t vR = 0; vR < pStateSize; ++vR) {
          int32_t vSrc;
          if (vD <= cNSeqTokens) {
            const uint32_t cPrefix = cNSeqTokens - vD;
            vSrc = (int32_t)(cStreamOff + vR);

            for (uint32_t vJ = 0; vJ < cPrefix; ++vJ) {
              const uint32_t cITok = lTokenIdxs[vJ];
              if (pUbatch.vPos[cITok] >= 0 && (uint32_t)(pUbatch.vPos[cITok] % pStateSize) == vR) {
                vSrc = (int32_t)(cScratchOff + cITok);
              }
            }
          } else {
            const int64_t cSrcPlane = (int64_t)(vD - cNSeqTokens) * cStateRows;
            vSrc = (int32_t)(cSrcPlane + cStreamOff + vR);
          }

          vPlan.lStateSnapshotSrcIdxs.push_back(vSrc);
          vPlan.lStateSnapshotDstIdxs.push_back((int32_t)(cDstPlane + cStreamOff + vR));
        }
      }
    }
  }

  static const bool cDebug = []() {
    const char *vEnv = getenv("MGS_DSV41_COMPRESS_DEBUG");
    return vEnv && atoi(vEnv) > 0;
  }();

  if (cDebug) {
    MGS_LOG_INFO("%s: ratio=%u, n_tokens=%u, state_persist_dst=%s, state_write_pos=%s\n", __func__, pRatio,
                 pUbatch.vNTokens, fDsv41PlanPositions(vPlan.lStatePersistDstIdxs).c_str(),
                 fDsv41PlanPositions(vPlan.lStateWritePos).c_str());
  }

  return vPlan;
}

static std::vector<mgs_kv_cache_dsv41_context::comp_plan>
fDsv41BuildCompPlans(const std::vector<mgs_ubatch> &pUbatches, uint32_t pRatio, uint32_t pStateSize,
                     uint32_t pKvSize, uint32_t pNStream, uint32_t pNRsSeq, const std::vector<uint32_t> &pRsIdx) {
  std::vector<mgs_kv_cache_dsv41_context::comp_plan> lPlans;
  lPlans.reserve(pUbatches.size());

  for (const mgs_ubatch &vUbatch : pUbatches) {
    lPlans.push_back(fDsv41BuildCompPlan(vUbatch, pRatio, pStateSize, pKvSize, pNStream, pNRsSeq, pRsIdx));
  }

  return lPlans;
}

static mgs_kv_cache::slot_info_vec_t fDsv41BuildCompSinfos(const std::vector<mgs_ubatch> &pUbatches,
                                                           uint32_t pNStream) {
  mgs_kv_cache::slot_info_vec_t vSinfos;
  vSinfos.reserve(pUbatches.size());

  for (const mgs_ubatch &vUbatch : pUbatches) {
    if (pNStream <= 1 && vUbatch.vNSeqsUnq > 1) {
      throw std::runtime_error("DSV41 single compressed stream cannot serve multiple sequences");
    }

    const uint32_t cNs = (uint32_t)fDsv41CompGraphNStream(vUbatch, pNStream);
    mgs_kv_cache::slot_info vSinfo;
    vSinfo.vS0 = pNStream > 1 ? MGS_MAX_SEQ : 0;
    vSinfo.vS1 = 0;
    vSinfo.mResize(cNs);

    for (uint32_t vS = 0; vS < cNs; ++vS) {
      const mgs_seq_id cSeqId = pNStream > 1 ? vUbatch.vSeqIdUnq[vS] : 0;
      const uint32_t cStrm = (uint32_t)fDsv41StreamOffset(pNStream, cSeqId, 1);

      vSinfo.vS0 = std::min(vSinfo.vS0, cStrm);
      vSinfo.vS1 = std::max(vSinfo.vS1, cStrm);
      vSinfo.lStrm[vS] = cStrm;
      vSinfo.lIdxs[vS].resize(1, 0);
    }

    if (pNStream > 1 && vSinfo.vS1 - vSinfo.vS0 + 1 != cNs) {
      throw std::runtime_error("DSV41 compressed streams are not contiguous in ubatch");
    }

    vSinfos.push_back(std::move(vSinfo));
  }

  return vSinfos;
}

static mgs_kv_cache::slot_info_vec_t fDsv41BuildRawReadSinfos(const mgs_kv_cache::slot_info_vec_t &pSinfosWrite,
                                                              const std::vector<mgs_ubatch> &pUbatches) {
  mgs_kv_cache::slot_info_vec_t vSinfos;
  vSinfos.reserve(pUbatches.size());

  for (size_t vI = 0; vI < pUbatches.size(); ++vI) {
    const mgs_ubatch &vUbatch = pUbatches[vI];
    const auto &vSinfoWrite = pSinfosWrite[vI];

    if (!fDsv41UbatchHasCoupled(vUbatch)) {
      vSinfos.push_back(vSinfoWrite);
      continue;
    }

    const mgs_seq_id cSeqId = vUbatch.vSeqId[0][0];
    uint32_t vIStream = 0;
    for (; vIStream < vSinfoWrite.mNStream(); ++vIStream) {
      if (vSinfoWrite.lStrm[vIStream] == cSeqId) {
        break;
      }
    }
    if (vIStream == vSinfoWrite.mNStream()) {
      throw std::runtime_error("DSV41 raw write stream not found for coupled read");
    }

    mgs_kv_cache::slot_info vSinfo;
    vSinfo.vS0 = vSinfoWrite.lStrm[vIStream];
    vSinfo.vS1 = vSinfoWrite.lStrm[vIStream];
    vSinfo.mResize(1);
    vSinfo.lStrm[0] = vSinfoWrite.lStrm[vIStream];
    vSinfo.lIdxs[0] = vSinfoWrite.lIdxs[vIStream];
    vSinfos.push_back(std::move(vSinfo));
  }

  return vSinfos;
}

static mgs_kv_cache_dsv41_context::comp_plan fDsv41BuildReserveCompPlan(const mgs_ubatch &pUbatch, uint32_t pRatio,
                                                                        uint32_t pStateSize, uint32_t pKvSize,
                                                                        uint32_t pNStream, uint32_t pNRsSeq) {
  mgs_kv_cache_dsv41_context::comp_plan vPlan;
  vPlan.lNVisible.resize(pUbatch.vNTokens);
  vPlan.vNStream = fDsv41CompGraphNStream(pUbatch, pNStream);
  vPlan.vNKv = pKvSize;

  if (pUbatch.vNTokens == 0) {
    return vPlan;
  }

  const uint32_t cNSeqs = std::max<uint32_t>(1, pUbatch.vNSeqs);
  const uint32_t cNSeqTokens = std::max<uint32_t>(1, pUbatch.vNSeqTokens);
  const uint64_t cNBlocksU64 = (uint64_t)cNSeqs * ((cNSeqTokens + pRatio - 1) / pRatio);
  const size_t cNBlocks = (size_t)std::max<uint64_t>(1, cNBlocksU64);
  GGML_ASSERT((uint64_t)cNBlocks == std::max<uint64_t>(1, cNBlocksU64));

  const uint64_t cStateRows = (uint64_t)pStateSize * pNStream;
  const size_t cNPersist = (size_t)std::min<uint64_t>(pUbatch.vNTokens, cStateRows);
  const size_t cNRestore = pNRsSeq > 0 ? (size_t)pStateSize * std::max<uint32_t>(1, pUbatch.vNSeqsUnq) : 0;
  const size_t cNSnapshot = (size_t)pNRsSeq * pStateSize * std::max<uint32_t>(1, pUbatch.vNSeqsUnq);

  vPlan.lStatePos.resize(pUbatch.vNTokens);
  vPlan.lStatePersistSrcIdxs.resize(cNPersist);
  vPlan.lStatePersistDstIdxs.resize(cNPersist);
  vPlan.lStateRestoreSrcIdxs.resize(cNRestore);
  vPlan.lStateRestoreDstIdxs.resize(cNRestore);
  vPlan.lStateSnapshotSrcIdxs.resize(cNSnapshot);
  vPlan.lStateSnapshotDstIdxs.resize(cNSnapshot);
  vPlan.lStateReadIdxs.resize(pRatio * cNBlocks);
  vPlan.lStateWriteIdxs.resize(cNBlocks);
  vPlan.lStateWritePos.resize(cNBlocks);

  return vPlan;
}

static void fDsv41MakeKOnly(mgs_hparams &pHparams) {
  // mgs_kv_cache uses hparams.mIsMla() to allocate K-only storage.
  pHparams.vNEmbdHeadKMlaImpl = pHparams.mNEmbdHeadK();
  pHparams.vNEmbdHeadVMlaImpl = pHparams.mNEmbdHeadK();
}

//
// mgs_dsv41_comp_state
//

mgs_dsv41_comp_state::mgs_dsv41_comp_state(const mgs_model &pModel, bool pOffload, bool pUnified, uint32_t pNSeqMax,
                                           uint32_t pRatio, uint32_t pStateSize, uint32_t pNEmbdState,
                                           uint32_t pNRsSeq, const char *pName,
                                           const mgs_memory_i::layer_filter_cb &pFilter)
    : cRatio(pRatio), cStateSize(pStateSize), cNEmbdState(pNEmbdState), cNStream(pUnified ? 1 : pNSeqMax),
      cNRsSeq(pNRsSeq) {
  const mgs_hparams &vHparams = pModel.vHparams;

  struct ggml_backend_buft_comparator {
    bool operator()(const ggml_backend_buffer_type_t &pLhs, const ggml_backend_buffer_type_t &pRhs) const {
      return strcmp(ggml_backend_buft_name(pLhs), ggml_backend_buft_name(pRhs)) < 0;
    }
  };

  std::map<ggml_backend_buffer_type_t, ggml_context_ptr, ggml_backend_buft_comparator> dCtxMap;

  auto vCtxForBuft = [&](ggml_backend_buffer_type_t pBuft) -> ggml_context * {
    auto vIt = dCtxMap.find(pBuft);
    if (vIt == dCtxMap.end()) {
      ggml_init_params vParams = {
          /*.mem_size   =*/size_t(2u * (1 + cNStream) * vHparams.mNLayer() * ggml_tensor_overhead()),
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

  for (uint32_t vIl = 0; vIl < vHparams.mNLayer(); ++vIl) {
    if (pFilter && !pFilter(vIl)) {
      continue;
    }

    const char *vDevName = "CPU";

    ggml_backend_buffer_type_t vBuft = ggml_backend_cpu_buffer_type();

    if (pOffload) {
      auto *vDev = pModel.mDevLayer(vIl);
      vBuft = ggml_backend_dev_buffer_type(vDev);

      vDevName = ggml_backend_dev_name(vDev);
    }

    MGS_LOG_DEBUG("%s: layer %3d: dev = %s\n", __func__, vIl, vDevName);

    ggml_context *vCtx = vCtxForBuft(vBuft);
    if (!vCtx) {
      throw std::runtime_error("failed to create ggml context for DSV41 compressor state");
    }

    const uint32_t cNPlanes = cNStream * (1 + pNRsSeq);
    ggml_tensor *vKv = ggml_new_tensor_3d(vCtx, GGML_TYPE_F32, pNEmbdState, pStateSize, cNPlanes);
    ggml_tensor *vScore = ggml_new_tensor_3d(vCtx, GGML_TYPE_F32, pNEmbdState, pStateSize, cNPlanes);

    ggml_format_name(vKv, "dsv41_%s_state_kv_l%d", pName, vIl);
    ggml_format_name(vScore, "dsv41_%s_state_score_l%d", pName, vIl);

    std::vector<ggml_tensor *> lKvStream;
    std::vector<ggml_tensor *> lScoreStream;

    for (uint32_t vS = 0; vS < cNStream; ++vS) {
      lKvStream.push_back(ggml_view_2d(vCtx, vKv, pNEmbdState, pStateSize, vKv->nb[1], vS * vKv->nb[2]));
      lScoreStream.push_back(ggml_view_2d(vCtx, vScore, pNEmbdState, pStateSize, vScore->nb[1], vS * vScore->nb[2]));
    }

    dMapLayerIds[vIl] = lLayers.size();

    lLayers.push_back({vIl, vKv, vScore, std::move(lKvStream), std::move(lScoreStream)});
  }

  for (auto &[vBuft, vCtx] : dCtxMap) {
    ggml_backend_buffer_t vBuf = ggml_backend_alloc_ctx_tensors_from_buft(vCtx.get(), vBuft);
    if (!vBuf) {
      throw std::runtime_error("failed to allocate buffer for DSV41 compressor state");
    }

    ggml_backend_buffer_clear(vBuf, 0);

    MGS_LOG_INFO("%s: %10s DSV41 %s state buffer size = %8.2f MiB\n", __func__, ggml_backend_buffer_name(vBuf),
                 pName, ggml_backend_buffer_get_size(vBuf) / 1024.0 / 1024.0);

    lCtxsBufs.emplace_back(std::move(vCtx), vBuf);
  }

  MGS_LOG_INFO("%s: %s ratio = %u, state = %u x %u, streams = %u, rs_seq = %u, layers = %zu, size = %7.2f MiB\n",
               __func__, pName, pRatio, pStateSize, pNEmbdState, cNStream, pNRsSeq, lLayers.size(),
               mTotalSize() / 1024.0 / 1024.0);
}

void mgs_dsv41_comp_state::mClear(mgs_seq_id pSeqId, bool pData) {
  if (!pData) {
    return;
  }

  if (pSeqId >= 0) {
    GGML_ASSERT((uint32_t)pSeqId < cNStream);

    for (const auto &vLayer : lLayers) {
      for (uint32_t vD = 0; vD <= cNRsSeq; ++vD) {
        const uint32_t cStream = vD * cNStream + (uint32_t)pSeqId;
        fDsv41ClearTensorStream(vLayer.vKv, cStream);
        fDsv41ClearTensorStream(vLayer.vScore, cStream);
      }
    }
    return;
  }

  for (auto &[_, vBuf] : lCtxsBufs) {
    ggml_backend_buffer_clear(vBuf.get(), 0);
  }
}

void mgs_dsv41_comp_state::mSeqCp(mgs_seq_id pSeqIdSrc, mgs_seq_id pSeqIdDst) {
  GGML_ASSERT(pSeqIdSrc >= 0 && (uint32_t)pSeqIdSrc < cNStream);
  GGML_ASSERT(pSeqIdDst >= 0 && (uint32_t)pSeqIdDst < cNStream);

  if (pSeqIdSrc == pSeqIdDst) {
    return;
  }

  mClear(pSeqIdDst, true);

  vScInfo.lSsrc.push_back((uint32_t)pSeqIdSrc);
  vScInfo.lSdst.push_back((uint32_t)pSeqIdDst);
}

void mgs_dsv41_comp_state::mApplyCopies(const stream_copy_info &pScInfo) const {
  for (size_t vI = 0; vI < pScInfo.lSsrc.size(); ++vI) {
    const uint32_t cSsrc = pScInfo.lSsrc[vI];
    const uint32_t cSdst = pScInfo.lSdst[vI];

    for (const auto &vLayer : lLayers) {
      ggml_backend_tensor_copy(vLayer.lKvStream[cSsrc], vLayer.lKvStream[cSdst]);
      ggml_backend_tensor_copy(vLayer.lScoreStream[cSsrc], vLayer.lScoreStream[cSdst]);
    }
  }
}

uint32_t mgs_dsv41_comp_state::mGetRatio() const { return cRatio; }

uint32_t mgs_dsv41_comp_state::mGetStateSize() const { return cStateSize; }

uint32_t mgs_dsv41_comp_state::mGetNStream() const { return cNStream; }

uint32_t mgs_dsv41_comp_state::mGetNRsSeq() const { return cNRsSeq; }

uint32_t mgs_dsv41_comp_state::mGetNRows() const { return cStateSize * cNStream; }

std::map<ggml_backend_buffer_type_t, size_t> mgs_dsv41_comp_state::mMemoryBreakdown() const {
  std::map<ggml_backend_buffer_type_t, size_t> dRet;
  for (const auto &[_, vBuf] : lCtxsBufs) {
    ggml_backend_buffer_type_t vBuft = ggml_backend_buffer_get_type(vBuf.get());
    dRet[vBuft] += ggml_backend_buffer_get_size(vBuf.get());
  }
  return dRet;
}

void mgs_dsv41_comp_state::mStateWrite(mgs_io_write_i &pIo, mgs_seq_id pSeqId, mgs_state_seq_flags pFlags,
                                       const std::vector<uint32_t> &pRsIdx) const {
  GGML_UNUSED(pFlags);

  uint32_t vS0;
  uint32_t vNs;
  fDsv41StateSrcStreamRange(cNStream, pSeqId, vS0, vNs);

  std::vector<uint32_t> lStreamIds(vNs);
  for (uint32_t vS = 0; vS < vNs; ++vS) {
    const uint32_t cSeq = pSeqId >= 0 ? (uint32_t)pSeqId : vS0 + vS;
    if (cSeq >= pRsIdx.size() || pRsIdx[cSeq] > cNRsSeq) {
      throw std::runtime_error("DSV41 recurrent state rollback index out of range");
    }
    lStreamIds[vS] = pRsIdx[cSeq] * cNStream + vS0 + vS;
  }

  const uint32_t cVersion = cDsv41CompStateVer;
  const uint32_t cNLayer = lLayers.size();

  pIo.mWrite(&cVersion, sizeof(cVersion));
  pIo.mWrite(&cRatio, sizeof(cRatio));
  pIo.mWrite(&cStateSize, sizeof(cStateSize));
  pIo.mWrite(&cNEmbdState, sizeof(cNEmbdState));
  pIo.mWrite(&vNs, sizeof(vNs));
  pIo.mWrite(&cNLayer, sizeof(cNLayer));

  for (const auto &vLayer : lLayers) {
    pIo.mWrite(&vLayer.vIl, sizeof(vLayer.vIl));

    fDsv41StateWriteTensorStreams(pIo, vLayer.vKv, cStateSize, cStateSize, vS0, vNs, &lStreamIds);
    fDsv41StateWriteTensorStreams(pIo, vLayer.vScore, cStateSize, cStateSize, vS0, vNs, &lStreamIds);
  }
}

void mgs_dsv41_comp_state::mStateRead(mgs_io_read_i &pIo, mgs_seq_id pSeqId, mgs_state_seq_flags pFlags) {
  GGML_UNUSED(pFlags);

  uint32_t vVersion;
  uint32_t vRatioRef;
  uint32_t vStateSizeRef;
  uint32_t vNEmbdStateRef;
  uint32_t vNs;
  uint32_t vNLayerRef;

  pIo.mRead(&vVersion, sizeof(vVersion));
  pIo.mRead(&vRatioRef, sizeof(vRatioRef));
  pIo.mRead(&vStateSizeRef, sizeof(vStateSizeRef));
  pIo.mRead(&vNEmbdStateRef, sizeof(vNEmbdStateRef));
  pIo.mRead(&vNs, sizeof(vNs));
  pIo.mRead(&vNLayerRef, sizeof(vNLayerRef));

  if (vVersion != cDsv41CompStateVer) {
    throw std::runtime_error("DSV41 compressor state version mismatch");
  }
  if (vRatioRef != cRatio || vStateSizeRef != cStateSize || vNEmbdStateRef != cNEmbdState) {
    throw std::runtime_error("DSV41 compressor state metadata mismatch");
  }
  if (vNLayerRef != lLayers.size()) {
    throw std::runtime_error("DSV41 compressor state layer count mismatch");
  }

  uint32_t vS0;
  fDsv41StateDstStreamRange(cNStream, pSeqId, vNs, vS0);

  for (const auto &vLayer : lLayers) {
    uint32_t vIlRef;
    pIo.mRead(&vIlRef, sizeof(vIlRef));
    if (vIlRef != vLayer.vIl) {
      throw std::runtime_error("DSV41 compressor state layer id mismatch");
    }

    fDsv41StateReadTensorStreams(pIo, vLayer.vKv, cStateSize, cStateSize, vS0, vNs);
    fDsv41StateReadTensorStreams(pIo, vLayer.vScore, cStateSize, cStateSize, vS0, vNs);
  }
}

ggml_tensor *mgs_dsv41_comp_state::mGetKvAll(ggml_context *pCtx, int32_t pIl) const {
  const int32_t cIds = dMapLayerIds.at(pIl);
  ggml_tensor *vState = lLayers[cIds].vKv;

  return ggml_view_2d(pCtx, vState, vState->ne[0], mGetNRows() * (1 + cNRsSeq), vState->nb[1], 0);
}

ggml_tensor *mgs_dsv41_comp_state::mGetScoreAll(ggml_context *pCtx, int32_t pIl) const {
  const int32_t cIds = dMapLayerIds.at(pIl);
  ggml_tensor *vState = lLayers[cIds].vScore;

  return ggml_view_2d(pCtx, vState, vState->ne[0], mGetNRows() * (1 + cNRsSeq), vState->nb[1], 0);
}

ggml_tensor *mgs_dsv41_comp_state::mCpyKv(ggml_context *pCtx, ggml_tensor *pCur, ggml_tensor *pIdxs,
                                          int32_t pIl) const {
  return ggml_set_rows(pCtx, mGetKvAll(pCtx, pIl), pCur, pIdxs);
}

ggml_tensor *mgs_dsv41_comp_state::mCpyScore(ggml_context *pCtx, ggml_tensor *pCur, ggml_tensor *pIdxs,
                                             int32_t pIl) const {
  return ggml_set_rows(pCtx, mGetScoreAll(pCtx, pIl), pCur, pIdxs);
}

size_t mgs_dsv41_comp_state::mTotalSize() const {
  size_t vSize = 0;

  for (const auto &[_, vBuf] : lCtxsBufs) {
    vSize += ggml_backend_buffer_get_size(vBuf.get());
  }

  return vSize;
}

//
// mgs_kv_cache_dsv41
//

mgs_kv_cache_dsv41::mgs_kv_cache_dsv41(const mgs_model &pModel, ggml_type pTypeK, ggml_type pTypeV, bool pVTrans,
                                       bool pOffload, bool pSwaFull, bool pUnified, uint32_t pKvSize,
                                       uint32_t pNSeqMax, uint32_t pNUbatch, uint32_t pNPad, uint32_t pNRsSeq,
                                       const layer_filter_cb &pFilter)
    : vHparamsRaw(pModel.vHparams), cNSeqMax(pNSeqMax), cNRsSeq(pNRsSeq), lRsIdx(pNSeqMax, 0) {
  const mgs_hparams &vHparams = pModel.vHparams;

  GGML_UNUSED(pUnified);

  // Keep the KV and compressor streams per sequence even when the public KV mode is unified.
  const bool cUnifiedRaw = false;
  const bool cUnifiedCompressed = false;

  vHparamsRaw.vNLayerNextn = 0;

  MGS_LOG_INFO("%s: creating DSV41 raw KV cache\n", __func__);

  fDsv41MakeKOnly(vHparamsRaw);

  vKvRaw = std::make_unique<mgs_kv_cache_iswa>(pModel, vHparamsRaw, pTypeK, pTypeV, pVTrans, pOffload, pSwaFull,
                                               cUnifiedRaw, pKvSize, pNSeqMax, pNUbatch, pNPad, nullptr, pFilter,
                                               nullptr, nullptr);

  // one group per distinct compression ratio, in the order the layers first use them
  std::vector<uint32_t> lRatios;
  for (uint32_t vIl = 0; vIl < vHparams.mNLayer(); ++vIl) {
    const uint32_t cRatio = vHparams.aDsv4CompressRatios[vIl];
    if (cRatio != 0 && std::find(lRatios.begin(), lRatios.end(), cRatio) == lRatios.end()) {
      lRatios.push_back(cRatio);
    }
  }

  for (const uint32_t cRatio : lRatios) {
    auto vGroup = std::make_unique<comp_group>(vHparams);
    vGroup->vRatio = cRatio;

    vGroup->vHparamsComp.vNLayerNextn = 0;
    fDsv41MakeKOnly(vGroup->vHparamsComp);

    vGroup->vHparamsIdx.vNLayerNextn = 0;
    std::fill(vGroup->vHparamsIdx.aNHeadKvArr.begin(), vGroup->vHparamsIdx.aNHeadKvArr.end(), 1);
    vGroup->vHparamsIdx.vNEmbdHeadKFull = vHparams.vIndexerHeadSize;
    vGroup->vHparamsIdx.vNEmbdHeadVFull = vHparams.vIndexerHeadSize;
    vGroup->vHparamsIdx.vNEmbdHeadKSwa = vHparams.vIndexerHeadSize;
    vGroup->vHparamsIdx.vNEmbdHeadVSwa = vHparams.vIndexerHeadSize;
    fDsv41MakeKOnly(vGroup->vHparamsIdx);

    const layer_filter_cb cFilterSource = [&](int32_t pIl) {
      if (pFilter && !pFilter(pIl)) {
        return false;
      }

      return vHparams.aDsv4CompressRatios[pIl] == cRatio && vHparams.mIsDsv41KvSource(pIl);
    };

    const layer_filter_cb cFilterKeyOwner = [&](int32_t pIl) {
      if (pFilter && !pFilter(pIl)) {
        return false;
      }

      return vHparams.aDsv4CompressRatios[pIl] == cRatio && vHparams.mOwnsDsv41IndexKey(pIl);
    };

    bool vHasKeyOwner = false;
    for (uint32_t vIl = 0; vIl < vHparams.mNLayer(); ++vIl) {
      vHasKeyOwner = vHasKeyOwner || cFilterKeyOwner((int32_t)vIl);
    }

    const uint32_t cCompSize = GGML_PAD(fDsv41CompSize(pKvSize, cRatio), 256u);

    MGS_LOG_INFO("%s: creating DSV41 compressed KV cache, ratio = %u, size = %u cells\n", __func__, cRatio,
                 cCompSize);

    vGroup->vKvComp =
        std::make_unique<mgs_kv_cache>(pModel, vGroup->vHparamsComp, pTypeK, pTypeV, pVTrans, pOffload,
                                       cUnifiedCompressed, cCompSize, pNSeqMax, pNPad, 0, cMgsSwaTypeNone, nullptr,
                                       cFilterSource, nullptr, nullptr);

    if (vHasKeyOwner) {
      MGS_LOG_INFO("%s: creating DSV41 lightning-indexer key cache, ratio = %u, size = %u cells\n", __func__,
                   cRatio, cCompSize);

      vGroup->vKvIdx =
          std::make_unique<mgs_kv_cache>(pModel, vGroup->vHparamsIdx, pTypeK, pTypeV, pVTrans, pOffload,
                                         cUnifiedCompressed, cCompSize, pNSeqMax, pNPad, 0, cMgsSwaTypeNone, nullptr,
                                         cFilterKeyOwner, nullptr, nullptr);
    }

    const std::string cName = "ratio" + std::to_string(cRatio);

    vGroup->vState =
        std::make_unique<mgs_dsv41_comp_state>(pModel, pOffload, cUnifiedCompressed, pNSeqMax, cRatio, cRatio,
                                               vHparams.mNEmbdHeadK(), pNRsSeq, cName.c_str(), cFilterSource);

    lGroups.push_back(std::move(vGroup));
  }

  // The attention reads compressed-K / compressor-state rows that the current graph does not
  // necessarily overwrite; uninitialized buffer contents would otherwise leak in. Zero all
  // compressed buffers up front so reads of un-written rows are deterministic.
  mClearCompressed(-1, true);
}

mgs_memory_context_ptr mgs_kv_cache_dsv41::mInitBatch(mgs_batch_allocr &pBalloc, uint32_t pNUbatch, bool pEmbdAll) {
  GGML_UNUSED(pEmbdAll);

  const bool cRawPerSeq = vKvRaw->mGetBase()->mGetNStream() != 1;
  const bool cCompPerSeq = cNSeqMax > 1;
  const bool cHasCoupled = fDsv41BatchHasCoupled(pBalloc.mGetBatch());

  const auto cMakeContext = [&](std::vector<mgs_ubatch> pUbatches) -> mgs_memory_context_ptr {
    auto vUbatchesRaw = fDsv41BuildRawWriteUbatches(pUbatches);

    auto vSinfosRawBaseWrite = vKvRaw->mGetBase()->mPrepare(vUbatchesRaw);
    if (vSinfosRawBaseWrite.empty()) {
      return nullptr;
    }

    auto vSinfosRawSwaWrite = vKvRaw->mGetSwa()->mPrepare(vUbatchesRaw);
    if (vSinfosRawSwaWrite.empty()) {
      return nullptr;
    }

    auto vSinfosRawSwaRead = fDsv41BuildRawReadSinfos(vSinfosRawSwaWrite, pUbatches);

    return std::make_unique<mgs_kv_cache_dsv41_context>(this, std::move(vSinfosRawBaseWrite),
                                                        std::move(vSinfosRawSwaWrite), std::move(vSinfosRawSwaRead),
                                                        std::move(pUbatches), std::move(vUbatchesRaw));
  };

  // Match mgs_kv_cache_iswa splitting when the compressed state does not
  // require per-sequence graph layout.
  do {
    if (cRawPerSeq || cCompPerSeq) {
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
      break;
    }

    if (auto vCtx = cMakeContext(std::move(lUbatches))) {
      return vCtx;
    }
  } while (false);

  // When raw or compressed state is per-sequence, independent sequences can
  // share an equal-length ubatch. Coupled sequence sets still serialize.
  do {
    pBalloc.mSplitReset();

    std::vector<mgs_ubatch> lUbatches;
    while (true) {
      mgs_ubatch vUbatch;
      if (cHasCoupled) {
        vUbatch = pBalloc.mSplitSeq(pNUbatch);
      } else {
        vUbatch = pBalloc.mSplitEqual(pNUbatch, cRawPerSeq || cCompPerSeq, 0);
      }

      if (vUbatch.vNTokens == 0) {
        break;
      }
      lUbatches.push_back(std::move(vUbatch)); // NOLINT
    }

    if (pBalloc.mGetNUsed() < pBalloc.mGetNTokens()) {
      break;
    }

    if (auto vCtx = cMakeContext(std::move(lUbatches))) {
      return vCtx;
    }
  } while (false);

  return std::make_unique<mgs_kv_cache_dsv41_context>(cMgsMemoryStatusFailedPrepare);
}

mgs_memory_context_ptr mgs_kv_cache_dsv41::mInitFull() { return std::make_unique<mgs_kv_cache_dsv41_context>(this); }

mgs_memory_context_ptr mgs_kv_cache_dsv41::mInitUpdate(mgs_context *pLctx, bool pOptimize) {
  std::vector<mgs_kv_cache::stream_copy_info> lScInfos;
  lScInfos.reserve(lGroups.size());
  for (auto &vGroup : lGroups) {
    lScInfos.push_back(std::move(vGroup->vState->vScInfo));
    vGroup->vState->vScInfo = {};
  }

  return std::make_unique<mgs_kv_cache_dsv41_context>(this, pLctx, pOptimize, std::move(lScInfos));
}

bool mgs_kv_cache_dsv41::mGetCanShift() const {
  // Compressed row metadata uses group-derived positions, so shifting is not supported.
  return false;
}

void mgs_kv_cache_dsv41::mClear(bool pData) {
  vKvRaw->mClear(pData);
  mClearCompressed(-1, true); // compressed buffers must never expose stale/uninit rows
}

bool mgs_kv_cache_dsv41::mSeqRm(mgs_seq_id pSeqId, mgs_pos p0, mgs_pos p1) {
  if (p1 >= 0) {
    return false;
  }

  if (p0 > 0) {
    if (pSeqId < 0 || (uint32_t)pSeqId >= cNSeqMax) {
      return false;
    }

    const mgs_pos cPosMax = vKvRaw->mSeqPosMax(pSeqId);
    if (p0 > cPosMax) {
      bool vRes = true;

      vRes = vRes & vKvRaw->mSeqRm(pSeqId, p0, -1);
      for (const auto &vGroup : lGroups) {
        vRes = vRes & vGroup->vKvComp->mSeqRm(pSeqId, p0 / vGroup->vRatio, -1);
        if (vGroup->vKvIdx) {
          vRes = vRes & vGroup->vKvIdx->mSeqRm(pSeqId, p0 / vGroup->vRatio, -1);
        }
      }

      return vRes;
    }

    if (cNRsSeq == 0) {
      return false;
    }

    const mgs_pos cRollback = cPosMax - (p0 - 1);
    if (cRollback < 1 || cRollback > (mgs_pos)cNRsSeq) {
      return false;
    }

    const bool cRes = vKvRaw->mSeqRm(pSeqId, p0, p1);
    if (cRes) {
      lRsIdx[pSeqId] = (uint32_t)cRollback;
    }

    return cRes;
  }

  const bool cRes = vKvRaw->mSeqRm(pSeqId, p0, p1);

  if (cRes) {
    mClearCompressed(pSeqId, true);
  }

  return cRes;
}

void mgs_kv_cache_dsv41::mSeqCp(mgs_seq_id pSeqIdSrc, mgs_seq_id pSeqIdDst, mgs_pos p0, mgs_pos p1) {
  GGML_ASSERT(p0 <= 0 && p1 < 0 && "DSV41 only supports full sequence copies");

  vKvRaw->mSeqCp(pSeqIdSrc, pSeqIdDst, p0, p1);
  for (const auto &vGroup : lGroups) {
    vGroup->vKvComp->mSeqCp(pSeqIdSrc, pSeqIdDst, -1, -1);
    if (vGroup->vKvIdx) {
      vGroup->vKvIdx->mSeqCp(pSeqIdSrc, pSeqIdDst, -1, -1);
    }
    vGroup->vState->mSeqCp(pSeqIdSrc, pSeqIdDst);
  }

  if (pSeqIdSrc != pSeqIdDst) {
    lRsIdx[pSeqIdDst] = 0;
  }
}

void mgs_kv_cache_dsv41::mSeqKeep(mgs_seq_id pSeqId) {
  GGML_ASSERT(pSeqId >= 0 && (uint32_t)pSeqId < cNSeqMax);

  vKvRaw->mSeqKeep(pSeqId);

  for (mgs_seq_id vId = 0; vId < (mgs_seq_id)cNSeqMax; ++vId) {
    if (vId == pSeqId) {
      continue;
    }

    vKvRaw->mSeqRm(vId, -1, -1);
    mClearCompressed(vId, true);
  }
}

void mgs_kv_cache_dsv41::mSeqAdd(mgs_seq_id pSeqId, mgs_pos p0, mgs_pos p1, mgs_pos pShift) {
  vKvRaw->mSeqAdd(pSeqId, p0, p1, pShift);
}

void mgs_kv_cache_dsv41::mSeqDiv(mgs_seq_id pSeqId, mgs_pos p0, mgs_pos p1, int pD) {
  vKvRaw->mSeqDiv(pSeqId, p0, p1, pD);
}

mgs_pos mgs_kv_cache_dsv41::mSeqPosMin(mgs_seq_id pSeqId) const {
  if (pSeqId < 0 || (uint32_t)pSeqId >= cNSeqMax) {
    return -1;
  }

  // The raw SWA cache may contain a wider window, but the compressed state cannot be rolled back
  // within that window. Report only the current boundary so the server uses checkpoints for rollback.
  return vKvRaw->mSeqPosMax(pSeqId);
}

mgs_pos mgs_kv_cache_dsv41::mSeqPosMax(mgs_seq_id pSeqId) const {
  if (pSeqId < 0 || (uint32_t)pSeqId >= cNSeqMax) {
    return -1;
  }

  return vKvRaw->mSeqPosMax(pSeqId);
}

std::map<ggml_backend_buffer_type_t, size_t> mgs_kv_cache_dsv41::mMemoryBreakdown() const {
  std::map<ggml_backend_buffer_type_t, size_t> dMb = vKvRaw->mMemoryBreakdown();

  const auto cAdd = [&dMb](const std::map<ggml_backend_buffer_type_t, size_t> &pOther) {
    for (const auto &vBuftSize : pOther) {
      dMb[vBuftSize.first] += vBuftSize.second;
    }
  };

  for (const auto &vGroup : lGroups) {
    cAdd(vGroup->vKvComp->mMemoryBreakdown());
    if (vGroup->vKvIdx) {
      cAdd(vGroup->vKvIdx->mMemoryBreakdown());
    }
    cAdd(vGroup->vState->mMemoryBreakdown());
  }

  return dMb;
}

void mgs_kv_cache_dsv41::mStateWrite(mgs_io_write_i &pIo, mgs_seq_id pSeqId, mgs_state_seq_flags pFlags) const {
  const bool cPartialOnly = pFlags & MGS_STATE_SEQ_FLAGS_PARTIAL_ONLY;

  const uint32_t cMagic = cDsv41StateMagic;
  const uint32_t cVersion = cDsv41StateVersion;
  const uint32_t cMode = cPartialOnly ? cDsv41StateModePartial : cDsv41StateModeFull;
  const uint32_t cNGroup = lGroups.size();

  pIo.mWrite(&cMagic, sizeof(cMagic));
  pIo.mWrite(&cVersion, sizeof(cVersion));
  pIo.mWrite(&cMode, sizeof(cMode));
  pIo.mWrite(&cNGroup, sizeof(cNGroup));

  vKvRaw->mStateWrite(pIo, pSeqId, pFlags);

  if (!cPartialOnly) {
    const mgs_pos cPosMax = pSeqId >= 0 ? vKvRaw->mSeqPosMax(pSeqId) : -1;

    // FIXME : note that we conflate token positions with rows, which is not true for multi-modal case.
    for (const auto &vGroup : lGroups) {
      const uint32_t cSize = vGroup->vKvComp->mGetSize();
      const uint32_t cNRows = pSeqId >= 0 ? fDsv41StateNUsedKRows(cPosMax, vGroup->vRatio, cSize) : cSize;

      fDsv41StateWriteKCache(pIo, vGroup->vKvComp.get(), pSeqId, cNRows);
      if (vGroup->vKvIdx) {
        fDsv41StateWriteKCache(pIo, vGroup->vKvIdx.get(), pSeqId, cNRows);
      }
    }
  }

  for (const auto &vGroup : lGroups) {
    vGroup->vState->mStateWrite(pIo, pSeqId, pFlags, lRsIdx);
  }
}

void mgs_kv_cache_dsv41::mStateRead(mgs_io_read_i &pIo, mgs_seq_id pSeqId, mgs_state_seq_flags pFlags) {
  uint32_t vMagic;
  uint32_t vVersion;
  uint32_t vMode = cDsv41StateModeFull;
  uint32_t vNGroup = 0;

  pIo.mRead(&vMagic, sizeof(vMagic));
  pIo.mRead(&vVersion, sizeof(vVersion));

  if (vMagic != cDsv41StateMagic) {
    throw std::runtime_error("DSV41 state magic mismatch");
  }
  if (vVersion != cDsv41StateVersion) {
    throw std::runtime_error("DSV41 state version mismatch");
  }

  pIo.mRead(&vMode, sizeof(vMode));
  if (vMode != cDsv41StateModeFull && vMode != cDsv41StateModePartial) {
    throw std::runtime_error("DSV41 state mode mismatch");
  }

  const bool cPartialOnly = vMode == cDsv41StateModePartial;
  if (cPartialOnly != !!(pFlags & MGS_STATE_SEQ_FLAGS_PARTIAL_ONLY)) {
    throw std::runtime_error("DSV41 state flags mismatch");
  }

  pIo.mRead(&vNGroup, sizeof(vNGroup));
  if (vNGroup != lGroups.size()) {
    throw std::runtime_error("DSV41 state compression group count mismatch");
  }

  vKvRaw->mStateRead(pIo, pSeqId, pFlags);

  if (!cPartialOnly) {
    for (const auto &vGroup : lGroups) {
      vGroup->vKvComp->mClear(true);
      fDsv41StateReadKCache(pIo, vGroup->vKvComp.get(), pSeqId);
      if (vGroup->vKvIdx) {
        vGroup->vKvIdx->mClear(true);
        fDsv41StateReadKCache(pIo, vGroup->vKvIdx.get(), pSeqId);
      }
    }
  }

  for (const auto &vGroup : lGroups) {
    vGroup->vState->mStateRead(pIo, pSeqId, pFlags);
  }

  if (pSeqId >= 0) {
    GGML_ASSERT((uint32_t)pSeqId < cNSeqMax);
    lRsIdx[pSeqId] = 0;
  } else {
    std::fill(lRsIdx.begin(), lRsIdx.end(), 0);
  }
}

mgs_kv_cache_iswa *mgs_kv_cache_dsv41::mGetRaw() const { return vKvRaw.get(); }

uint32_t mgs_kv_cache_dsv41::mGetNGroup() const { return (uint32_t)lGroups.size(); }

const mgs_kv_cache_dsv41::comp_group &mgs_kv_cache_dsv41::mGetGroup(uint32_t pG) const { return *lGroups.at(pG); }

uint32_t mgs_kv_cache_dsv41::mGetNRsSeq() const { return cNRsSeq; }

const std::vector<uint32_t> &mgs_kv_cache_dsv41::mGetRsIdx() const { return lRsIdx; }

void mgs_kv_cache_dsv41::mResetRsIdxForUbatches(const std::vector<mgs_ubatch> &pUbatches) {
  if (cNRsSeq == 0) {
    return;
  }

  for (const mgs_ubatch &vUbatch : pUbatches) {
    for (uint32_t vI = 0; vI < vUbatch.vNTokens; ++vI) {
      for (int32_t vS = 0; vS < vUbatch.vNSeqId[vI]; ++vS) {
        const mgs_seq_id cSeqId = vUbatch.vSeqId[vI][vS];
        if (cSeqId >= 0 && (uint32_t)cSeqId < cNSeqMax) {
          lRsIdx[cSeqId] = 0;
        }
      }
    }
  }
}

void mgs_kv_cache_dsv41::mClearCompressed(mgs_seq_id pSeqId, bool pData) {
  const auto cClearSeq = [pSeqId, pData](mgs_kv_cache *pKv) {
    pKv->mSeqRm(pSeqId, -1, -1);

    if (pData) {
      for (uint32_t vIl : pKv->mGetLayerIds()) {
        fDsv41ClearTensorStream(pKv->mGetKStorage(vIl), (uint32_t)pSeqId);
      }
    }
  };

  if (pSeqId >= 0) {
    GGML_ASSERT((uint32_t)pSeqId < cNSeqMax);
  }

  for (const auto &vGroup : lGroups) {
    if (pSeqId < 0) {
      vGroup->vKvComp->mClear(pData);
      if (vGroup->vKvIdx) {
        vGroup->vKvIdx->mClear(pData);
      }
    } else {
      cClearSeq(vGroup->vKvComp.get());
      if (vGroup->vKvIdx) {
        cClearSeq(vGroup->vKvIdx.get());
      }
    }

    vGroup->vState->mClear(pSeqId, pData);
  }

  if (pSeqId >= 0) {
    lRsIdx[pSeqId] = 0;
  } else {
    std::fill(lRsIdx.begin(), lRsIdx.end(), 0);
  }
}

//
// mgs_kv_cache_dsv41_raw_context
//

static mgs_kv_cache::slot_info fDsv41BuildFullSinfo(const mgs_kv_cache *pKv) {
  const uint32_t cNStream = pKv->mGetNStream();

  mgs_kv_cache::slot_info vSinfo;
  vSinfo.vS0 = 0;
  vSinfo.vS1 = cNStream - 1;
  vSinfo.mResize(cNStream);
  for (uint32_t vS = 0; vS < cNStream; ++vS) {
    vSinfo.lStrm[vS] = vS;
    vSinfo.lIdxs[vS].resize(1, 0);
  }

  return vSinfo;
}

mgs_kv_cache_dsv41_raw_context::mgs_kv_cache_dsv41_raw_context(mgs_kv_cache_iswa *pKv)
    : vKvSwa(pKv->mGetSwa()), cCtxBaseMem(nullptr), cCtxSwaMem(nullptr), vNKv(vKvSwa->mGetSize()),
      cStatus(cMgsMemoryStatusSuccess) {
  vSinfosRead.push_back(fDsv41BuildFullSinfo(vKvSwa));
  vSinfosWrite = vSinfosRead;
}

mgs_kv_cache_dsv41_raw_context::mgs_kv_cache_dsv41_raw_context(mgs_kv_cache_iswa *pKv, mgs_context *pLctx,
                                                               bool pOptimize)
    : vKvSwa(pKv->mGetSwa()), cCtxBaseMem(pKv->mGetBase()->mInitUpdate(pLctx, pOptimize)),
      cCtxSwaMem(pKv->mGetSwa()->mInitUpdate(pLctx, pOptimize)), vNKv(vKvSwa->mGetSize()),
      cStatus(mgs_memory_status_combine(cCtxBaseMem->mGetStatus(), cCtxSwaMem->mGetStatus())) {}

mgs_kv_cache_dsv41_raw_context::mgs_kv_cache_dsv41_raw_context(mgs_kv_cache_iswa *pKv,
                                                               slot_info_vec_t pSinfosBaseWrite,
                                                               slot_info_vec_t pSinfosSwaWrite,
                                                               slot_info_vec_t pSinfosSwaRead,
                                                               std::vector<mgs_ubatch> pUbatches,
                                                               std::vector<mgs_ubatch> pUbatchesWrite)
    : vKvSwa(pKv->mGetSwa()), vSinfosWrite(std::move(pSinfosSwaWrite)), vSinfosRead(std::move(pSinfosSwaRead)),
      lUbatches(std::move(pUbatches)), lUbatchesWrite(std::move(pUbatchesWrite)),
      cCtxBaseMem(
          std::make_unique<mgs_kv_cache_context>(pKv->mGetBase(), std::move(pSinfosBaseWrite), this->lUbatchesWrite)),
      cCtxSwaMem(nullptr), vNKv(vKvSwa->mGetSize()), cStatus(cMgsMemoryStatusSuccess) {}

bool mgs_kv_cache_dsv41_raw_context::mNext() {
  if (lUbatches.empty()) {
    return true;
  }

  if (cCtxBaseMem) {
    cCtxBaseMem->mNext();
  }

  if (++vINext >= lUbatches.size()) {
    return false;
  }

  return true;
}

bool mgs_kv_cache_dsv41_raw_context::mApply() {
  bool vRes = true;

  if (cCtxBaseMem) {
    vRes = vRes & cCtxBaseMem->mApply();
  }
  if (cCtxSwaMem) {
    vRes = vRes & cCtxSwaMem->mApply();
  }
  if (!lUbatchesWrite.empty()) {
    vKvSwa->mApplyUbatch(vSinfosWrite[vINext], lUbatchesWrite[vINext]);
    vNKv = vKvSwa->mGetNKv(vSinfosRead[vINext]);
  }

  return vRes;
}

mgs_memory_status mgs_kv_cache_dsv41_raw_context::mGetStatus() const { return cStatus; }

const mgs_ubatch &mgs_kv_cache_dsv41_raw_context::mGetUbatch() const {
  assert(cStatus == cMgsMemoryStatusSuccess);

  return lUbatches[vINext];
}

uint32_t mgs_kv_cache_dsv41_raw_context::mGetNKv() const { return vNKv; }

uint32_t mgs_kv_cache_dsv41_raw_context::mGetNWrite() const {
  if (lUbatchesWrite.empty()) {
    return 0;
  }

  return lUbatchesWrite[vINext].vNTokens;
}

ggml_tensor *mgs_kv_cache_dsv41_raw_context::mGetK(ggml_context *pCtx, int32_t pIl) const {
  return vKvSwa->mGetK(pCtx, pIl, vNKv, vSinfosRead[vINext]);
}

ggml_tensor *mgs_kv_cache_dsv41_raw_context::mCpyK(ggml_context *pCtx, ggml_tensor *pKCur, ggml_tensor *pKIdxs,
                                                   int32_t pIl) const {
  const auto &vSinfo = vSinfosWrite[vINext];

  if (pKCur->ne[2] == pKIdxs->ne[0]) {
    return vKvSwa->mCpyK(pCtx, pKCur, pKIdxs, pIl, vSinfo);
  }

  // k_idxs may be expanded to one block per stream while k_cur is only
  // the token block. Keep zero deps on all copies so each write executes.
  const int64_t cNFanout = (int64_t)vSinfo.size() * vSinfo.mNStream();

  GGML_ASSERT(vSinfo.mNStream() > 1);
  GGML_ASSERT(pKCur->ne[2] == (int64_t)vSinfo.size());
  GGML_ASSERT(pKIdxs->ne[0] == cNFanout);

  ggml_tensor *vRes = nullptr;
  for (uint32_t vS = 0; vS < vSinfo.mNStream(); ++vS) {
    ggml_tensor *vKIdxsS = ggml_view_1d(pCtx, pKIdxs, vSinfo.size(), vS * vSinfo.size() * ggml_element_size(pKIdxs));
    ggml_tensor *vCur = vKvSwa->mCpyK(pCtx, pKCur, vKIdxsS, pIl, vSinfo);
    if (vRes == nullptr) {
      vRes = vCur;
    } else {
      vRes = ggml_add(pCtx, vRes, ggml_sub(pCtx, vCur, vCur));
    }
  }

  return vRes;
}

ggml_tensor *mgs_kv_cache_dsv41_raw_context::mBuildInputKIdxs(ggml_context *pCtx, const mgs_ubatch &pUbatch) const {
  const uint32_t cNTokens = lUbatchesWrite.empty() ? pUbatch.vNTokens : lUbatchesWrite[vINext].vNTokens;

  ggml_tensor *vKIdxs = ggml_new_tensor_1d(pCtx, GGML_TYPE_I64, cNTokens);
  ggml_set_input(vKIdxs);

  return vKIdxs;
}

ggml_tensor *mgs_kv_cache_dsv41_raw_context::mBuildInputKRot(ggml_context *pCtx) const {
  return vKvSwa->mBuildInputKRot(pCtx);
}

void mgs_kv_cache_dsv41_raw_context::mSetInputKIdxs(ggml_tensor *pDst) const {
  vKvSwa->mSetInputKIdxs(pDst, &lUbatchesWrite[vINext], vSinfosWrite[vINext]);
}

void mgs_kv_cache_dsv41_raw_context::mSetInputKqMask(ggml_tensor *pDst, const mgs_ubatch *pUbatch,
                                                     bool pCausalAttn) const {
  vKvSwa->mSetInputKqMask(pDst, pUbatch, pCausalAttn);
}

void mgs_kv_cache_dsv41_raw_context::mSetInputKRot(ggml_tensor *pDst) const { vKvSwa->mSetInputKRot(pDst); }

void mgs_kv_cache_dsv41_raw_context::mGetPrevTokens(const mgs_ubatch &pUbatch, uint32_t pN,
                                                    std::vector<mgs_token> &pRes) const {
  vKvSwa->mGetPrevTokens(pUbatch, pN, pRes);
}

//
// mgs_kv_cache_dsv41_comp_context
//

mgs_kv_cache_dsv41_comp_context::mgs_kv_cache_dsv41_comp_context(mgs_kv_cache *pKv)
    : vKv(pKv), vNKv(pKv->mGetSize()) {
  const uint32_t cNStream = pKv->mGetNStream();

  vSinfos.resize(1);
  vSinfos[0].vS0 = 0;
  vSinfos[0].vS1 = cNStream - 1;
  vSinfos[0].lIdxs.resize(cNStream);
  for (uint32_t vS = 0; vS < cNStream; ++vS) {
    vSinfos[0].lStrm.push_back(vS);
    vSinfos[0].lIdxs[vS].resize(1, 0);
  }
}

mgs_kv_cache_dsv41_comp_context::mgs_kv_cache_dsv41_comp_context(mgs_kv_cache *pKv, slot_info_vec_t pSinfos,
                                                                 std::vector<mgs_ubatch> pUbatches)
    : vKv(pKv), vSinfos(std::move(pSinfos)), lUbatches(std::move(pUbatches)), vNKv(pKv->mGetSize()) {}

bool mgs_kv_cache_dsv41_comp_context::mNext() {
  if (lUbatches.empty()) {
    return true;
  }

  if (++vICur >= lUbatches.size()) {
    return false;
  }

  return true;
}

uint32_t mgs_kv_cache_dsv41_comp_context::mGetNKv() const { return vNKv; }

ggml_tensor *mgs_kv_cache_dsv41_comp_context::mGetK(ggml_context *pCtx, int32_t pIl) const {
  return vKv->mGetK(pCtx, pIl, vNKv, vSinfos[vICur]);
}

ggml_tensor *mgs_kv_cache_dsv41_comp_context::mCpyK(ggml_context *pCtx, ggml_tensor *pKCur, ggml_tensor *pKIdxs,
                                                    int32_t pIl) const {
  return vKv->mCpyK(pCtx, pKCur, pKIdxs, pIl, vSinfos[vICur]);
}

ggml_tensor *mgs_kv_cache_dsv41_comp_context::mBuildInputKRot(ggml_context *pCtx) const {
  return vKv->mBuildInputKRot(pCtx);
}

void mgs_kv_cache_dsv41_comp_context::mSetInputKRot(ggml_tensor *pDst) const { vKv->mSetInputKRot(pDst); }

//
// mgs_kv_cache_dsv41_context
//

mgs_kv_cache_dsv41_context::mgs_kv_cache_dsv41_context(mgs_memory_status pStatus) : vStatus(pStatus) {}

mgs_kv_cache_dsv41_context::mgs_kv_cache_dsv41_context(mgs_kv_cache_dsv41 *pKv)
    : vKv(pKv), cCtxRaw(std::make_unique<mgs_kv_cache_dsv41_raw_context>(pKv->mGetRaw())), vReservePlans(true),
      vStatus(cCtxRaw->mGetStatus()) {
  lGroups.resize(pKv->mGetNGroup());
  for (uint32_t vG = 0; vG < pKv->mGetNGroup(); ++vG) {
    const auto &vGroup = pKv->mGetGroup(vG);
    auto &vCtx = lGroups[vG];

    vCtx.vCtxCompMem = vGroup.vKvComp->mInitFull();
    vStatus = mgs_memory_status_combine(vStatus, vCtx.vCtxCompMem->mGetStatus());
    vCtx.vCtxComp = std::make_unique<mgs_kv_cache_dsv41_comp_context>(vGroup.vKvComp.get());
    if (vGroup.vKvIdx) {
      vCtx.vCtxIdxMem = vGroup.vKvIdx->mInitFull();
      vStatus = mgs_memory_status_combine(vStatus, vCtx.vCtxIdxMem->mGetStatus());
      vCtx.vCtxIdx = std::make_unique<mgs_kv_cache_dsv41_comp_context>(vGroup.vKvIdx.get());
    }
    vCtx.vState = vGroup.vState.get();
  }
}

mgs_kv_cache_dsv41_context::mgs_kv_cache_dsv41_context(mgs_kv_cache_dsv41 *pKv, mgs_context *pLctx, bool pOptimize,
                                                       std::vector<stream_copy_info> pScInfos)
    : vKv(pKv), cCtxRaw(std::make_unique<mgs_kv_cache_dsv41_raw_context>(pKv->mGetRaw(), pLctx, pOptimize)),
      vStatus(cCtxRaw->mGetStatus()) {
  lGroups.resize(pKv->mGetNGroup());

  bool vHasCopies = false;
  for (uint32_t vG = 0; vG < pKv->mGetNGroup(); ++vG) {
    const auto &vGroup = pKv->mGetGroup(vG);
    auto &vCtx = lGroups[vG];

    vCtx.vCtxCompMem = vGroup.vKvComp->mInitUpdate(pLctx, pOptimize);
    vStatus = mgs_memory_status_combine(vStatus, vCtx.vCtxCompMem->mGetStatus());
    if (vGroup.vKvIdx) {
      vCtx.vCtxIdxMem = vGroup.vKvIdx->mInitUpdate(pLctx, pOptimize);
      vStatus = mgs_memory_status_combine(vStatus, vCtx.vCtxIdxMem->mGetStatus());
    }
    vCtx.vState = vGroup.vState.get();
    if (vG < pScInfos.size()) {
      vCtx.vScInfo = std::move(pScInfos[vG]);
    }
    vHasCopies = vHasCopies || !vCtx.vScInfo.empty();
  }

  if (vHasCopies) {
    vStatus = mgs_memory_status_combine(vStatus, cMgsMemoryStatusSuccess);
  }
}

mgs_kv_cache_dsv41_context::mgs_kv_cache_dsv41_context(mgs_kv_cache_dsv41 *pKv, slot_info_vec_t pSinfosRawBaseWrite,
                                                       slot_info_vec_t pSinfosRawSwaWrite,
                                                       slot_info_vec_t pSinfosRawSwaRead,
                                                       std::vector<mgs_ubatch> pUbatches,
                                                       std::vector<mgs_ubatch> pUbatchesRaw)
    : lUbatches(std::move(pUbatches)), vKv(pKv),
      cCtxRaw(std::make_unique<mgs_kv_cache_dsv41_raw_context>(
          pKv->mGetRaw(), std::move(pSinfosRawBaseWrite), std::move(pSinfosRawSwaWrite), std::move(pSinfosRawSwaRead),
          this->lUbatches, std::move(pUbatchesRaw))),
      vStatus(cCtxRaw->mGetStatus()) {
  lGroups.resize(pKv->mGetNGroup());
  for (uint32_t vG = 0; vG < pKv->mGetNGroup(); ++vG) {
    const auto &vGroup = pKv->mGetGroup(vG);
    auto &vCtx = lGroups[vG];

    vCtx.lPlans = fDsv41BuildCompPlans(this->lUbatches, vGroup.vRatio, vGroup.vState->mGetStateSize(),
                                       vGroup.vKvComp->mGetSize(), vGroup.vState->mGetNStream(), pKv->mGetNRsSeq(),
                                       pKv->mGetRsIdx());
    vCtx.vCtxComp = std::make_unique<mgs_kv_cache_dsv41_comp_context>(
        vGroup.vKvComp.get(), fDsv41BuildCompSinfos(this->lUbatches, vGroup.vKvComp->mGetNStream()),
        this->lUbatches);
    if (vGroup.vKvIdx) {
      vCtx.vCtxIdx = std::make_unique<mgs_kv_cache_dsv41_comp_context>(
          vGroup.vKvIdx.get(), fDsv41BuildCompSinfos(this->lUbatches, vGroup.vKvIdx->mGetNStream()),
          this->lUbatches);
    }
    vCtx.vState = vGroup.vState.get();
  }

  pKv->mResetRsIdxForUbatches(this->lUbatches);
}

mgs_kv_cache_dsv41_context::~mgs_kv_cache_dsv41_context() = default;

bool mgs_kv_cache_dsv41_context::mNext() {
  assert(vStatus == cMgsMemoryStatusSuccess);

  cCtxRaw->mNext();
  for (auto &vCtx : lGroups) {
    if (vCtx.vCtxComp) {
      vCtx.vCtxComp->mNext();
    }
    if (vCtx.vCtxIdx) {
      vCtx.vCtxIdx->mNext();
    }
  }

  if (++vINext >= lUbatches.size()) {
    return false;
  }

  return true;
}

bool mgs_kv_cache_dsv41_context::mApply() {
  assert(!mgs_memory_status_is_fail(vStatus));

  bool vRes = true;

  vRes = vRes & cCtxRaw->mApply();

  for (auto &vCtx : lGroups) {
    if (vCtx.vCtxCompMem) {
      vRes = vRes & vCtx.vCtxCompMem->mApply();
    }
    if (vCtx.vCtxIdxMem) {
      vRes = vRes & vCtx.vCtxIdxMem->mApply();
    }
    if (lUbatches.empty()) {
      vCtx.vState->mApplyCopies(vCtx.vScInfo);
    }
  }

  return vRes;
}

mgs_memory_status mgs_kv_cache_dsv41_context::mGetStatus() const { return vStatus; }

const mgs_ubatch &mgs_kv_cache_dsv41_context::mGetUbatch() const {
  assert(vStatus == cMgsMemoryStatusSuccess);

  return lUbatches[vINext];
}

const mgs_kv_cache_dsv41_raw_context *mgs_kv_cache_dsv41_context::mGetRaw() const {
  assert(vStatus == cMgsMemoryStatusSuccess);

  return cCtxRaw.get();
}

uint32_t mgs_kv_cache_dsv41_context::mGetNGroup() const { return (uint32_t)lGroups.size(); }

uint32_t mgs_kv_cache_dsv41_context::mGetRatio(uint32_t pG) const { return lGroups.at(pG).vState->mGetRatio(); }

const mgs_kv_cache_dsv41_comp_context *mgs_kv_cache_dsv41_context::mGetComp(uint32_t pG) const {
  assert(vStatus == cMgsMemoryStatusSuccess);

  return lGroups.at(pG).vCtxComp.get();
}

const mgs_kv_cache_dsv41_comp_context *mgs_kv_cache_dsv41_context::mGetIdx(uint32_t pG) const {
  assert(vStatus == cMgsMemoryStatusSuccess);

  return lGroups.at(pG).vCtxIdx.get();
}

const mgs_dsv41_comp_state *mgs_kv_cache_dsv41_context::mGetState(uint32_t pG) const {
  assert(vStatus == cMgsMemoryStatusSuccess);

  return lGroups.at(pG).vState;
}

const mgs_kv_cache_dsv41_context::comp_plan &mgs_kv_cache_dsv41_context::mGetPlan(uint32_t pG,
                                                                                  const mgs_ubatch &pUbatch) const {
  assert(vStatus == cMgsMemoryStatusSuccess);

  auto &vCtx = lGroups.at(pG);

  if (vReservePlans) {
    const mgs_dsv41_comp_state *vState = vCtx.vState;
    vCtx.vReservePlan = fDsv41BuildReserveCompPlan(pUbatch, vState->mGetRatio(), vState->mGetStateSize(),
                                                   vCtx.vCtxComp->mGetNKv(), vState->mGetNStream(),
                                                   vState->mGetNRsSeq());
    return vCtx.vReservePlan;
  }

  static const comp_plan cEmpty;
  if (vCtx.lPlans.empty()) {
    return cEmpty;
  }

  return vCtx.lPlans[vINext];
}
