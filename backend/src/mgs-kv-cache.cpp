#include "mgs-kv-cache.h"

#include "mgs-impl.h"
#include "mgs-io.h"
#include "mgs-model.h"
#include "mgs-context.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <limits>
#include <map>
#include <stdexcept>
#include <unordered_map>

static bool ggml_is_power_of_2(int pN) { return (pN & (pN - 1)) == 0; }

// orthonormal Walsh-Hadamard rotation matrix
// note: res^2 == I
static void ggml_gen_hadamard(ggml_tensor *pTensor) {
  assert(pTensor->type == GGML_TYPE_F32);

  const int cN = pTensor->ne[0];

  assert(ggml_is_power_of_2(cN));
  assert(pTensor->ne[1] == cN);
  assert(pTensor->ne[2] == 1);
  assert(pTensor->ne[3] == 1);

  std::vector<float> lDataF32;

  float *vData = (float *)pTensor->data;

  if (pTensor->type != GGML_TYPE_F32) {
    lDataF32.resize(cN * cN);
    vData = lDataF32.data();
  }

  vData[0 * cN + 0] = 1.0 / sqrtf(cN);

  for (int vS = 1; vS < cN; vS *= 2) {
    for (int vI = 0; vI < vS; vI++) {
      for (int vJ = 0; vJ < vS; vJ++) {
        const float cVal = vData[vI * cN + vJ];

        vData[(vI + vS) * cN + (vJ)] = cVal;
        vData[(vI)*cN + (vJ + vS)] = cVal;
        vData[(vI + vS) * cN + (vJ + vS)] = -cVal;
      }
    }
  }

  if (pTensor->type != GGML_TYPE_F32) {
    ggml_quantize_chunk(pTensor->type, vData, pTensor->data, 0, 1, cN * cN, nullptr);
  }
}

//
// mgs_kv_cache
//

mgs_kv_cache::mgs_kv_cache(const mgs_model &pModel, const mgs_hparams &pHparams, ggml_type pTypeK, ggml_type pTypeV,
                           bool pVTrans, bool pOffload, bool pUnified, uint32_t pKvSize, uint32_t pNSeqMax,
                           uint32_t pNPad, uint32_t pNSwa, mgs_swa_type pSwaType, mgs_memory_t pMemOther,
                           const layer_filter_cb &pFilter, const layer_reuse_cb &pReuse, const layer_share_cb &pShare,
                           const char *pNameTag)
    : vModel(pModel), vHparams(pHparams), vVTrans(pVTrans), cNSeqMax(pNSeqMax), cNStream(pUnified ? 1 : pNSeqMax),
      cNPad(pNPad), cNSwa(pNSwa), cSwaType(pSwaType), vOther(static_cast<mgs_kv_cache *>(pMemOther)),
      vVCellsImpl(vOther ? vOther->vVCellsImpl : std::make_shared<mgs_kv_cells_vec>()), vVCells(*vVCellsImpl) {

  // shared cells view the source cache's K/V tensors, so the cell count
  // follows the source allocation: a fitted target can be smaller than the
  // draft default and oversized views would overflow the source tensors
  if (vOther) {
    const uint32_t cSizeOther = vOther->mGetSize();
    if (pKvSize != cSizeOther) {
      MGS_LOG_WARN("%s: kv_size = %u overridden to %u to match the shared source cache\n", __func__, pKvSize,
                   cSizeOther);
      pKvSize = cSizeOther;
    }
  }

  GGML_ASSERT(pKvSize % pNPad == 0);

  const uint32_t cNLayer = pHparams.vNLayerAll;

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
          /*.mem_size   =*/size_t(2u * (1 + cNStream) * cNLayer * ggml_tensor_overhead()),
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

  GGML_ASSERT(cNStream == 1 || cNStream == pNSeqMax);

  lVHeads.resize(cNStream);
  for (uint32_t vS = 0; vS < cNStream; ++vS) {
    lVHeads[vS] = 0;
  }

  vVCells.resize(cNStream);
  for (uint32_t vS = 0; vS < cNStream; ++vS) {
    vVCells[vS].mResize(pKvSize);
  }

  // by default, all sequence ids are mapped to the 0th stream
  lSeqToStream.resize(MGS_MAX_SEQ, 0);

  if (cNStream > 1) {
    lSeqToStream.resize(cNStream, 0);
    for (uint32_t vS = 0; vS < cNStream; ++vS) {
      lSeqToStream[vS] = vS;
    }
  }

  // [TAG_V_CACHE_VARIABLE]
  if (pVTrans && pHparams.mIsNEmbdVGqaVariable()) {
    MGS_LOG_WARN(
        "%s: the V embeddings have different sizes across layers and FA is not enabled - padding V cache to %d\n",
        __func__, pHparams.mNEmbdVGqaMax());
  }

  const bool cIsMla = pHparams.mIsMla();

  for (uint32_t vIl = 0; vIl < cNLayer; vIl++) {
    if (!pHparams.mHasKv(vIl)) {
      MGS_LOG_DEBUG("%s: layer %3d: does not have KV cache\n", __func__, vIl);
      continue;
    }

    if (pFilter && !pFilter(vIl)) {
      MGS_LOG_DEBUG("%s: layer %3d: filtered\n", __func__, vIl);
      continue;
    }

    if (pShare && vOther) {
      const int32_t cIlShare = pShare(vIl);

      if (cIlShare >= 0) {
        const auto &vLayerShare = vOther->lLayers[vOther->dMapLayerIds[cIlShare]];

        MGS_LOG_WARN("%s: layer %3d: sharing with layer %d. k = %p, v = %p\n", __func__, vIl, cIlShare,
                     vLayerShare.vK->data, vLayerShare.vV->data);

        dMapLayerIds[vIl] = lLayers.size();

        lLayers.push_back(vLayerShare);
        lLayers.back().vIl = vIl;

        continue;
      }
    }

    if (vNEmbdHeadKAll == 0) {
      vNEmbdHeadKAll = (int32_t)pHparams.mNEmbdHeadK(vIl);
    } else if (vNEmbdHeadKAll > 0 && vNEmbdHeadKAll != (int32_t)pHparams.mNEmbdHeadK(vIl)) {
      vNEmbdHeadKAll = -1;
    }

    if (!cIsMla) {
      if (vNEmbdHeadVAll == 0) {
        vNEmbdHeadVAll = (int32_t)pHparams.mNEmbdHeadV(vIl);
      } else if (vNEmbdHeadVAll > 0 && vNEmbdHeadVAll != (int32_t)pHparams.mNEmbdHeadV(vIl)) {
        vNEmbdHeadVAll = -1;
      }
    }

    // [TAG_V_CACHE_VARIABLE]
    const uint32_t cNEmbdKGqa = pHparams.mNEmbdKGqa(vIl);
    const uint32_t cNEmbdVGqa = !pVTrans ? pHparams.mNEmbdVGqa(vIl) : pHparams.mNEmbdVGqaMax();

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
      throw std::runtime_error("failed to create ggml context for kv cache");
    }

    const bool cHasK = true;
    const bool cHasV = !cIsMla;

    ggml_tensor *vK = cHasK ? ggml_new_tensor_3d(vCtx, pTypeK, cNEmbdKGqa, pKvSize, cNStream) : nullptr;
    ggml_tensor *vV = cHasV ? ggml_new_tensor_3d(vCtx, pTypeV, cNEmbdVGqa, pKvSize, cNStream) : nullptr;

    cHasK &&ggml_format_name(vK, "cache_%sk_l%d", pNameTag, vIl);
    cHasV &&ggml_format_name(vV, "cache_%sv_l%d", pNameTag, vIl);

    std::vector<ggml_tensor *> lKStream;
    std::vector<ggml_tensor *> lVStream;

    for (uint32_t vS = 0; vS < cNStream; ++vS) {
      lKStream.push_back(cHasK ? ggml_view_2d(vCtx, vK, cNEmbdKGqa, pKvSize, vK->nb[1], vS * vK->nb[2]) : nullptr);
      lVStream.push_back(cHasV ? ggml_view_2d(vCtx, vV, cNEmbdVGqa, pKvSize, vV->nb[1], vS * vV->nb[2]) : nullptr);
    }

    dMapLayerIds[vIl] = lLayers.size();

    lLayers.push_back({
        vIl,
        vK,
        vV,
        lKStream,
        lVStream,
    });
  }

  if (pReuse) {
    MGS_LOG_DEBUG("%s: reusing layers:\n", __func__);

    for (uint32_t vIl = 0; vIl < cNLayer; vIl++) {
      const int32_t cIlReuse = pReuse(vIl);

      if (cIlReuse < 0) {
        MGS_LOG_DEBUG("%s: - layer %3d: no reuse\n", __func__, vIl);
        continue;
      }

      if (pFilter && !pFilter(vIl)) {
        MGS_LOG_DEBUG("%s: - layer %3d: filtered\n", __func__, vIl);
        continue;
      }

      GGML_ASSERT(dMapLayerIds.find(cIlReuse) != dMapLayerIds.end());

      dMapLayerIds[vIl] = dMapLayerIds[cIlReuse];

      MGS_LOG_DEBUG("%s: - layer %3d: reuse layer %d, is_swa = %d\n", __func__, vIl, cIlReuse, pHparams.mIsSwa(vIl));
    }
  }

  // allocate tensors and initialize the buffers to avoid NaNs in the padding
  for (auto &[vBuft, vCtx] : dCtxMap) {
    ggml_backend_buffer_t vBuf;
    if (pHparams.vNoAlloc) {
      vBuf = ggml_backend_buft_alloc_buffer(vBuft, /*size =*/0); // dummy buffer
      for (ggml_tensor *vT = ggml_get_first_tensor(vCtx.get()); vT != nullptr;
           vT = ggml_get_next_tensor(vCtx.get(), vT)) {
        vT->buffer = vBuf; // set dummy buffer for KV cache so that the backend scheduler won't try to allocate it
      }
    } else {
      vBuf = ggml_backend_alloc_ctx_tensors_from_buft(vCtx.get(), vBuft); // real buffer
    }
    if (!vBuf) {
      throw std::runtime_error("failed to allocate buffer for kv cache");
    }

    MGS_LOG_INFO("%s: %10s KV buffer size = %8.2f MiB\n", __func__, ggml_backend_buffer_name(vBuf),
                 ggml_backend_buffer_get_size(vBuf) / 1024.0 / 1024.0);

    ggml_backend_buffer_clear(vBuf, 0);
    lCtxsBufs.emplace_back(std::move(vCtx), vBuf);
  }

  {
    const size_t cMemorySizeK = mSizeKBytes();
    const size_t cMemorySizeV = mSizeVBytes();

    MGS_LOG_INFO("%s: size = %7.2f MiB (%6u cells, %3d layers, %2u/%u seqs), K (%s): %7.2f MiB, V (%s): %7.2f MiB\n",
                 __func__, (float)(cMemorySizeK + cMemorySizeV) / (1024.0f * 1024.0f), pKvSize, (int)lLayers.size(),
                 pNSeqMax, cNStream, ggml_type_name(pTypeK), (float)cMemorySizeK / (1024.0f * 1024.0f),
                 ggml_type_name(pTypeV), (float)cMemorySizeV / (1024.0f * 1024.0f));
  }

  // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
  if (vOther) {
    vNEmbdHeadKAll = vOther->vNEmbdHeadKAll;
    vNEmbdHeadVAll = vOther->vNEmbdHeadVAll;

    vAttnRotK = vOther->vAttnRotK;
    vAttnRotV = vOther->vAttnRotV;
  } else {
    const char *vMgsAttnRotDisable = getenv("MGS_ATTN_ROT_DISABLE");
    const bool cAttnRotDisable = vMgsAttnRotDisable ? atoi(vMgsAttnRotDisable) : false;
    if (cAttnRotDisable) {
      MGS_LOG_WARN("%s: attention rotation force disabled (MGS_ATTN_ROT_DISABLE)\n", __func__);
    }

    vAttnRotK = !cAttnRotDisable && vNEmbdHeadKAll > 0 && ggml_is_quantized(pTypeK) && pHparams.mNEmbdHeadK() % 64 == 0;

    // always create Hadamard rotation tensors for DeepSeek lightning indexers
    if ((pModel.vArch == cLlmArchGlmDsa || pModel.vArch == cLlmArchDeepseek41) &&
        pHparams.vNEmbdHeadKFull == pHparams.vIndexerHeadSize) {
      vAttnRotK = true;
    }

    vAttnRotV = !cAttnRotDisable && vNEmbdHeadVAll > 0 && ggml_is_quantized(pTypeV) && pHparams.mNEmbdHeadV() % 64 == 0;
  }

  MGS_LOG_INFO("%s: attn_rot_k = %d, n_embd_head_k_all = %d\n", __func__, vAttnRotK, vNEmbdHeadKAll);
  MGS_LOG_INFO("%s: attn_rot_v = %d, n_embd_head_k_all = %d\n", __func__, vAttnRotV, vNEmbdHeadVAll);

  // pre-compute the haramard matrices and keep them in host memory
  // TODO: in the future, we can make copies in the backend buffers to avoid host -> device transfers
  if (vAttnRotK || vAttnRotV) {
    for (int64_t vN = 64; vN <= std::max(vNEmbdHeadKAll, vNEmbdHeadVAll); vN *= 2) {
      dAttnRotHadamard[vN] = std::vector<float>(vN * vN);

      ggml_init_params vParams = {
          /* .mem_size   = */ 1 * ggml_tensor_overhead(),
          /* .mem_buffer = */ nullptr,
          /* .no_alloc   = */ true,
      };

      ggml_context_ptr vCtx{ggml_init(vParams)};

      ggml_tensor *vTmp = ggml_new_tensor_2d(vCtx.get(), GGML_TYPE_F32, vN, vN);
      vTmp->data = dAttnRotHadamard[vN].data();

      ggml_gen_hadamard(vTmp);
    }
  }

  const char *vMgsKvCacheDebug = getenv("MGS_KV_CACHE_DEBUG");
  vDebug = vMgsKvCacheDebug ? atoi(vMgsKvCacheDebug) : 0;
}

void mgs_kv_cache::mClear(bool pData) {
  for (uint32_t vS = 0; vS < cNStream; ++vS) {
    vVCells[vS].mReset();
    lVHeads[vS] = 0;
  }

  if (pData) {
    for (auto &[_, vBuf] : lCtxsBufs) {
      ggml_backend_buffer_clear(vBuf.get(), 0);
    }
  }
}

bool mgs_kv_cache::mSeqRm(mgs_seq_id pSeqId, mgs_pos p0, mgs_pos p1) {
  // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
  if (vOther) {
    return true;
  }

  GGML_ASSERT(pSeqId == -1 || (pSeqId >= 0 && (size_t)pSeqId < lSeqToStream.size()));

  if (p0 < 0) {
    p0 = 0;
  }

  if (p1 < 0) {
    p1 = std::numeric_limits<mgs_pos>::max();
  }

  if (pSeqId >= 0) {
    auto &vCells = vVCells[lSeqToStream[pSeqId]];
    auto &vHead = lVHeads[lSeqToStream[pSeqId]];

    uint32_t vNewHead = vCells.size();

    for (uint32_t vI = 0; vI < vCells.size(); ++vI) {
      if (!vCells.mPosIn(vI, p0, p1)) {
        continue;
      }

      if (vCells.mSeqHas(vI, pSeqId) && vCells.mSeqRm(vI, pSeqId)) {
        if (vNewHead == vCells.size()) {
          vNewHead = vI;
        }
      }
    }

    // If we freed up a slot, set head to it so searching can start there.
    if (vNewHead != vCells.size() && vNewHead < vHead) {
      vHead = vNewHead;
    }
  } else {
    // match any sequence
    for (uint32_t vS = 0; vS < cNStream; ++vS) {
      auto &vCells = vVCells[vS];
      auto &vHead = lVHeads[vS];

      uint32_t vNewHead = vCells.size();

      for (uint32_t vI = 0; vI < vCells.size(); ++vI) {
        if (!vCells.mPosIn(vI, p0, p1)) {
          continue;
        }

        vCells.mRm(vI);

        if (vNewHead == vCells.size()) {
          vNewHead = vI;
        }
      }

      // If we freed up a slot, set head to it so searching can start there.
      if (vNewHead != vCells.size() && vNewHead < vHead) {
        vHead = vNewHead;
      }
    }
  }

  return true;
}

void mgs_kv_cache::mSeqCp(mgs_seq_id pSeqIdSrc, mgs_seq_id pSeqIdDst, mgs_pos p0, mgs_pos p1) {
  // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
  if (vOther) {
    return;
  }

  GGML_ASSERT(pSeqIdSrc >= 0 && (size_t)pSeqIdSrc < lSeqToStream.size());
  GGML_ASSERT(pSeqIdDst >= 0 && (size_t)pSeqIdDst < lSeqToStream.size());

  const auto cS0 = lSeqToStream[pSeqIdSrc];
  const auto cS1 = lSeqToStream[pSeqIdDst];

  if (cS0 == cS1) {
    // since both sequences are in the same stream, no data copy is necessary
    // we just have to update the cells meta data

    auto &vCells = vVCells[cS0];

    if (pSeqIdSrc == pSeqIdDst) {
      return;
    }

    if (p0 < 0) {
      p0 = 0;
    }

    if (p1 < 0) {
      p1 = std::numeric_limits<mgs_pos>::max();
    }

    for (uint32_t vI = 0; vI < vCells.size(); ++vI) {
      if (!vCells.mPosIn(vI, p0, p1)) {
        continue;
      }

      if (vCells.mSeqHas(vI, pSeqIdSrc)) {
        vCells.mSeqAdd(vI, pSeqIdDst);
      }
    }

    return;
  }

  // cross-stream sequence copies require to copy the actual buffer data

  bool vIsFull = true;

  if (p0 > 0 && p0 + 1 < (int)mGetSize()) {
    vIsFull = false;
  }

  if (p1 > 0 && p1 + 1 < (int)mGetSize()) {
    vIsFull = false;
  }

  GGML_ASSERT(vIsFull && "mSeqCp() is only supported for full KV buffers");

  // enqueue the copy operation - the buffer copy will be performed during the next update
  vScInfo.lSsrc.push_back(cS0);
  vScInfo.lSdst.push_back(cS1);

  vVCells[cS1].mReset();
  for (uint32_t vI = 0; vI < vVCells[cS0].size(); ++vI) {
    if (vVCells[cS0].mSeqHas(vI, pSeqIdSrc)) {
      mgs_pos vPos = vVCells[cS0].mPosGet(vI);
      mgs_pos vShift = vVCells[cS0].mGetShift(vI);

      mgs_kv_cell_ext vExt = vVCells[cS0].mExtGet(vI);

      if (vShift != 0) {
        vPos -= vShift;
        assert(vPos >= 0);
      }

      vVCells[cS1].mPosSet(vI, vPos);
      vVCells[cS1].mSeqAdd(vI, pSeqIdDst);

      if (vShift != 0) {
        vVCells[cS1].mPosAdd(vI, vShift);
      }

      vVCells[cS1].mExtSet(vI, vExt);
    }
  }

  lVHeads[cS1] = lVHeads[cS0];

  // for (uint32_t vS = 0; vS < cNStream; ++vS) {
  //     MGS_LOG_WARN("%s: seq %d: min = %d, max = %d\n", __func__, vS, vVCells[vS].mSeqPosMin(vS),
  //     vVCells[vS].mSeqPosMax(vS));
  // }
}

void mgs_kv_cache::mSeqKeep(mgs_seq_id pSeqId) {
  // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
  if (vOther) {
    return;
  }

  GGML_ASSERT(pSeqId >= 0 && (size_t)pSeqId < lSeqToStream.size());

  auto &vCells = vVCells[lSeqToStream[pSeqId]];
  auto &vHead = lVHeads[lSeqToStream[pSeqId]];

  uint32_t vNewHead = vCells.size();

  for (uint32_t vI = 0; vI < vCells.size(); ++vI) {
    if (vCells.mSeqKeep(vI, pSeqId)) {
      if (vNewHead == vCells.size()) {
        vNewHead = vI;
      }
    }
  }

  // If we freed up a slot, set head to it so searching can start there.
  if (vNewHead != vCells.size() && vNewHead < vHead) {
    vHead = vNewHead;
  }
}

void mgs_kv_cache::mSeqAdd(mgs_seq_id pSeqId, mgs_pos p0, mgs_pos p1, mgs_pos pShift) {
  // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
  if (vOther) {
    return;
  }

  GGML_ASSERT(pSeqId >= 0 && (size_t)pSeqId < lSeqToStream.size());
  GGML_ASSERT(vHparams.mNPosPerEmbd() == 1 && "mSeqAdd() is only supported for mNPosPerEmbd() == 1");

  auto &vCells = vVCells[lSeqToStream[pSeqId]];
  auto &vHead = lVHeads[lSeqToStream[pSeqId]];

  if (pShift == 0) {
    return;
  }

  uint32_t vNewHead = vCells.size();

  if (p0 < 0) {
    p0 = 0;
  }

  if (p1 < 0) {
    p1 = std::numeric_limits<mgs_pos>::max();
  }

  // If there is no range then return early to avoid looping over all cells.
  if (p0 == p1) {
    return;
  }

  for (uint32_t vI = 0; vI < vCells.size(); ++vI) {
    if (!vCells.mPosIn(vI, p0, p1)) {
      continue;
    }

    if (vCells.mSeqHas(vI, pSeqId)) {
      if (vCells.mPosAdd(vI, pShift)) {
        if (vNewHead == vCells.size()) {
          vNewHead = vI;
        }
      }
    }
  }

  // If we freed up a slot, set head to it so searching can start there.
  // Otherwise we just start the next search from the beginning.
  vHead = vNewHead != vCells.size() ? vNewHead : 0;
}

void mgs_kv_cache::mSeqDiv(mgs_seq_id pSeqId, mgs_pos p0, mgs_pos p1, int pD) {
  // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
  if (vOther) {
    return;
  }

  GGML_ASSERT(pSeqId >= 0 && (size_t)pSeqId < lSeqToStream.size());
  GGML_ASSERT(vHparams.mNPosPerEmbd() == 1 && "mSeqDiv() is only supported for mNPosPerEmbd() == 1");

  auto &vCells = vVCells[lSeqToStream[pSeqId]];

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

  for (uint32_t vI = 0; vI < vCells.size(); ++vI) {
    if (!vCells.mPosIn(vI, p0, p1)) {
      continue;
    }

    if (vCells.mSeqHas(vI, pSeqId)) {
      vCells.mPosDiv(vI, pD);
    }
  }
}

mgs_pos mgs_kv_cache::mSeqPosMin(mgs_seq_id pSeqId) const {
  // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
  if (vOther) {
    return vOther->mSeqPosMin(pSeqId);
  }

  GGML_ASSERT(pSeqId >= 0 && (size_t)pSeqId < lSeqToStream.size());

  const auto &vCells = vVCells[lSeqToStream[pSeqId]];

  return vCells.mSeqPosMin(pSeqId);
}

mgs_pos mgs_kv_cache::mSeqPosMax(mgs_seq_id pSeqId) const {
  // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
  if (vOther) {
    return vOther->mSeqPosMax(pSeqId);
  }

  GGML_ASSERT(pSeqId >= 0 && (size_t)pSeqId < lSeqToStream.size());

  const auto &vCells = vVCells[lSeqToStream[pSeqId]];

  return vCells.mSeqPosMax(pSeqId);
}

std::map<ggml_backend_buffer_type_t, size_t> mgs_kv_cache::mMemoryBreakdown() const {
  std::map<ggml_backend_buffer_type_t, size_t> dRet;
  for (const auto &[vCtx, vBuf] : lCtxsBufs) {
    ggml_backend_buffer_type_t vBuft = ggml_backend_buffer_get_type(vBuf.get());

    if (vHparams.vNoAlloc) {
      GGML_ASSERT(ggml_backend_buffer_get_base(vBuf.get()) == nullptr);
      dRet[vBuft] += ggml_backend_alloc_ctx_tensors_from_buft_size(vCtx.get(), vBuft);
    } else {
      // GGML_ASSERT(ggml_backend_buffer_get_base(buf.get()) != nullptr); // multi_buffer does not have a defined base
      dRet[vBuft] += ggml_backend_buffer_get_size(vBuf.get());
    }
  }

  return dRet;
}

mgs_memory_context_ptr mgs_kv_cache::mInitBatch(mgs_batch_allocr &pBalloc, uint32_t pNUbatch, bool pEmbdAll) {
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

    auto vSinfos = mPrepare(lUbatches);
    if (vSinfos.empty()) {
      break;
    }

    return std::make_unique<mgs_kv_cache_context>(this, std::move(vSinfos), std::move(lUbatches));
  } while (false);

  return std::make_unique<mgs_kv_cache_context>(cMgsMemoryStatusFailedPrepare);
}

mgs_memory_context_ptr mgs_kv_cache::mInitFull() { return std::make_unique<mgs_kv_cache_context>(this); }

mgs_memory_context_ptr mgs_kv_cache::mInitUpdate(mgs_context *pLctx, bool pOptimize) {
  GGML_UNUSED(pOptimize);

  bool vDoShift = mGetHasShift();

  return std::make_unique<mgs_kv_cache_context>(this, pLctx, vDoShift, std::move(vScInfo));
}

mgs_kv_cache::slot_info_vec_t mgs_kv_cache::mPrepare(const std::vector<mgs_ubatch> &pUbatches) {
  mgs_kv_cache::slot_info_vec_t vRes;

  struct state_t {
    slot_info vSinfo; // slot info for the ubatch

    std::vector<uint32_t> lVHeadsOld; // old positions of the heads, before placing the ubatch

    std::vector<mgs_kv_cells> lVCells; // copy of the old cells, before placing the ubatch
  };

  // remember the old state of the cells so we can restore it in the end
  std::vector<state_t> lStates;

  bool vSuccess = true;

  for (const auto &vUbatch : pUbatches) {
    // only find a suitable slot for the ubatch. don't modify the cells yet
    const auto cSinfoNew = mFindSlot(vUbatch, false);
    if (cSinfoNew.empty()) {
      vSuccess = false;
      break;
    }

    // remember the position that we found
    vRes.push_back(cSinfoNew);

    // store the old state of the cells in the recovery stack
    {
      state_t vState = {cSinfoNew, lVHeads, {}};

      for (uint32_t vS = 0; vS < cSinfoNew.mNStream(); ++vS) {
        auto &vCells = vVCells[cSinfoNew.lStrm[vS]];

        vState.lVCells.push_back(vCells.mCp(cSinfoNew.lIdxs[vS]));
      }

      lStates.push_back(std::move(vState));
    }

    // now emplace the ubatch
    mApplyUbatch(cSinfoNew, vUbatch);
  }

  GGML_ASSERT(!lStates.empty() || !vSuccess);

  // iterate backwards and restore the cells to their original state
  for (auto vIt = lStates.rbegin(); vIt != lStates.rend(); ++vIt) {
    const auto &vSinfo = vIt->vSinfo;

    for (uint32_t vS = 0; vS < vSinfo.mNStream(); ++vS) {
      auto &vCells = vVCells[vSinfo.lStrm[vS]];
      auto &vHead = lVHeads[vSinfo.lStrm[vS]];

      vCells.mSet(vSinfo.lIdxs[vS], vIt->lVCells[vS]);
      vHead = vIt->lVHeadsOld[vS];
    }
  }

  if (!vSuccess) {
    return {};
  }

  return vRes;
}

bool mgs_kv_cache::mUpdate(mgs_context *pLctx, bool pDoShift, const stream_copy_info &pScInfo) {
  // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
  if (vOther) {
    return true;
  }

  bool vUpdated = false;

  auto *vSched = pLctx->mGetSched();

  if (!pScInfo.empty()) {
    assert(cNStream > 1 && "stream copy should never happen with a single stream");

    mgs_synchronize(pLctx);

    const size_t cNCopy = pScInfo.lSsrc.size();

    for (size_t vI = 0; vI < cNCopy; ++vI) {
      const auto cSsrc = pScInfo.lSsrc[vI];
      const auto cSdst = pScInfo.lSdst[vI];

      assert(cSsrc < cNStream);
      assert(cSdst < cNStream);

      MGS_LOG_DEBUG("%s: copying KV buffer: stream %d to stream %d\n", __func__, cSsrc, cSdst);

      assert(cSsrc != cSdst);

      for (uint32_t vIl = 0; vIl < lLayers.size(); ++vIl) {
        const auto &vLayer = lLayers[vIl];

        ggml_backend_tensor_copy(vLayer.lKStream[cSsrc], vLayer.lKStream[cSdst]);

        if (vLayer.lVStream[cSsrc]) {
          ggml_backend_tensor_copy(vLayer.lVStream[cSsrc], vLayer.lVStream[cSdst]);
        }
      }
    }
  }

  if (pDoShift) {
    if (!mGetCanShift()) {
      GGML_ABORT("The current KV cache / model configuration does not support K-shift");
    }

    MGS_LOG_DEBUG("%s: applying K-shift\n", __func__);

    // apply K-shift if needed
    if (vHparams.vRopeType != MGS_ROPE_TYPE_NONE) {
      ggml_backend_sched_reset(vSched);

      auto *vRes = pLctx->mGetGfResReserve();

      vRes->mReset();

      auto *vGf = mBuildGraphShift(vRes, pLctx);
      if (!ggml_backend_sched_alloc_graph(vSched, vGf)) {
        MGS_LOG_ERROR("%s: failed to allocate compute graph for K-shift\n", __func__);
        return vUpdated;
      }

      vRes->mSetInputs(nullptr);

      if (pLctx->mGraphCompute(vGf, false) != GGML_STATUS_SUCCESS) {
        MGS_LOG_ERROR("%s: failed to compute K-shift\n", __func__);
        return vUpdated;
      }

      vUpdated = true;
    }

    for (uint32_t vS = 0; vS < cNStream; ++vS) {
      auto &vCells = vVCells[vS];

      vCells.mResetShift();
    }
  }

  return vUpdated;
}

mgs_kv_cache::slot_info mgs_kv_cache::mFindSlot(const mgs_ubatch &pUbatch, bool pCont) const {

  if (vDebug > 0) {
    for (uint32_t vS = 0; vS < pUbatch.vNSeqsUnq; ++vS) {
      const auto cSeqId = pUbatch.vSeqIdUnq[vS];
      const auto cStreamId = lSeqToStream[cSeqId];
      const auto &vCells = vVCells[cStreamId];
      const uint32_t cHeadCur = lVHeads[cStreamId];

      MGS_LOG_DEBUG("%s: stream[%d], n = %5d, used = %5d, head = %5d, size = %5d, n_swa = %5d\n", __func__, cStreamId,
                    vCells.mUsedMaxP1(), vCells.mGetUsed(), cHeadCur, mGetSize(), cNSwa);

      if ((vDebug == 2 && cNSwa > 0) || vDebug > 2) {
        std::string vSs;
        for (uint32_t vI = 0; vI < vCells.size(); ++vI) {
          if (vCells.mIsEmpty(vI)) {
            vSs += '.';
          } else {
            assert(vCells.mSeqCount(vI) >= 1);

            if (vCells.mSeqCount(vI) == 1) {
              vSs += std::to_string(vCells.mSeqGet(vI));
            } else {
              vSs += 'M';
            }
          }
          if (vI % 256 == 255) {
            vSs += " *";
            vSs += '\n';
          }
        }
        MGS_LOG_DEBUG("\n%s\n", vSs.c_str());
      }

      if ((vDebug == 2 && cNSwa > 0) || vDebug > 2) {
        std::string vSs;
        for (uint32_t vI = 0; vI < vCells.size(); ++vI) {
          std::string vCur;
          if (vCells.mIsEmpty(vI)) {
            vCur = '.';
          } else {
            vCur = std::to_string(vCells.mPosGet(vI));
          }
          const int cN = vCur.size();
          for (int vJ = 0; vJ < 5 - cN; ++vJ) {
            vCur += ' ';
          }
          vSs += vCur;
          if (vI % 256 == 255) {
            vSs += " *";
          }
          if (vI % 64 == 63) {
            vSs += '\n';
          }
        }
        MGS_LOG_DEBUG("\n%s\n", vSs.c_str());
      }

      for (int vS = 0; vS < MGS_MAX_SEQ; ++vS) {
        if (vCells.mSeqPosMin(vS) < 0) {
          continue;
        }

        MGS_LOG_DEBUG("%s: stream[%d] min[%d] = %5d, max[%d] = %5d\n", __func__, cStreamId, vS, vCells.mSeqPosMin(vS),
                      vS, vCells.mSeqPosMax(vS));
      }
    }
  }

  uint32_t vNTokens = pUbatch.vNTokens;
  uint32_t vNSeqs = 1;

  if (cNStream > 1) {
    GGML_ASSERT(vNTokens % pUbatch.vNSeqsUnq == 0);

    vNSeqs = pUbatch.vNSeqsUnq;
    vNTokens = vNTokens / vNSeqs;
  }

  slot_info vRes = {
      /*.vS0   =*/MGS_MAX_SEQ,
      /*.vS1   =*/0,
      /*.lStrm =*/{},
      /*.lIdxs =*/{},
  };

  vRes.mResize(vNSeqs);

  for (uint32_t vS = 0; vS < vNSeqs; ++vS) {
    const auto cSeqId = pUbatch.vSeqIdUnq[vS];

    if (cNStream > 1) {
      GGML_ASSERT(pUbatch.vNSeqId[vS * vNTokens] == 1);
      GGML_ASSERT(pUbatch.vSeqId[vS * vNTokens][0] == cSeqId);
    }

    vRes.vS0 = std::min<uint32_t>(vRes.vS0, lSeqToStream[cSeqId]);
    vRes.vS1 = std::max<uint32_t>(vRes.vS1, lSeqToStream[cSeqId]);

    vRes.lStrm[vS] = lSeqToStream[cSeqId];
    vRes.lIdxs[vS].reserve(vNTokens);

    const auto &vCells = vVCells[lSeqToStream[cSeqId]];

    uint32_t vHeadCur = lVHeads[lSeqToStream[cSeqId]];

    // if we have enough unused cells before the current head ->
    //   better to start searching from the beginning of the cache, hoping to fill it
    if (vHeadCur > vCells.mGetUsed() + 2 * vNTokens) {
      vHeadCur = 0;
    }

    if (vNTokens > vCells.size()) {
      MGS_LOG_ERROR("%s: n_tokens = %d > size = %u\n", __func__, vNTokens, vCells.size());
      return {};
    }

    uint32_t vNTested = 0;

    // for continuous slots, we test that all tokens in the ubatch fit, starting from the current head
    // for non-continuous slots, we test the tokens one by one
    const uint32_t cNTest = pCont ? vNTokens : 1;

    while (true) {
      if (vHeadCur + cNTest > vCells.size()) {
        vNTested += vCells.size() - vHeadCur;
        vHeadCur = 0;
        continue;
      }

      for (uint32_t vI = 0; vI < cNTest; vI++) {
        const auto cIdx = vHeadCur;

        vHeadCur++;
        vNTested++;

        // const mgs_pos    cPos    = pUbatch.vPos[vI];
        // const mgs_seq_id cSeqId = pUbatch.vSeqId[vI][0];

        // can we use this cell? either:
        //  - the cell is empty
        //  - the cell is occupied only by one sequence:
        //    - (disabled) mask causally, if the sequence is the same as the one we are inserting
        //    - mask SWA, using current max pos for that sequence in the cache
        //                always insert in the cell with minimum pos
        bool vCanUse = vCells.mIsEmpty(cIdx);

        if (!vCanUse && vCells.mSeqCount(cIdx) == 1) {
          const mgs_pos cPosCell = vCells.mPosGet(cIdx);

          // (disabled) causal mask
          // note: it's better to purge any "future" tokens beforehand
          // if (vCells.mSeqHas(cIdx, cSeqId)) {
          //    vCanUse = cPosCell >= cPos;
          //}

          if (!vCanUse) {
            const mgs_seq_id cSeqIdCell = vCells.mSeqGet(cIdx);

            // SWA mask
            if (mgs_hparams::mIsMaskedSwa(cNSwa, cSwaType, cPosCell, vCells.mSeqPosMax(cSeqIdCell) + 1)) {
              vCanUse = true;
            }
          }
        }

        if (vCanUse) {
          vRes.lIdxs[vS].push_back(cIdx);
        } else {
          if (pCont) {
            break;
          }
        }
      }

      if (vRes.lIdxs[vS].size() == vNTokens) {
        break;
      }

      if (pCont) {
        vRes.lIdxs[vS].clear();
      }

      if (vNTested >= vCells.size()) {
        // MGS_LOG_ERROR("%s: failed to find a slot for %d tokens\n", __func__, vNTokens);
        return {};
      }
    }

    // we didn't find a suitable slot - return empty result
    if (vRes.lIdxs[vS].size() < vNTokens) {
      return {};
    }
  }

  assert(vRes.vS1 >= vRes.vS0);

  return vRes;
}

void mgs_kv_cache::mApplyUbatch(const slot_info &pSinfo, const mgs_ubatch &pUbatch) {
  // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
  if (vOther) {
    return;
  }

  // keep track of the max sequence position that we would overwrite with this ubatch
  // for non-SWA cache, this would be always empty
  mgs_seq_id aSeqPosMaxRm[MGS_MAX_SEQ];
  for (uint32_t vS = 0; vS < MGS_MAX_SEQ; ++vS) {
    aSeqPosMaxRm[vS] = -1;
  }

  assert(pUbatch.vNTokens == pSinfo.mNStream() * pSinfo.size());

  for (uint32_t vS = 0; vS < pSinfo.mNStream(); ++vS) {
    for (uint32_t vIi = 0; vIi < pSinfo.size(); ++vIi) {
      const uint32_t cI = vS * pSinfo.size() + vIi;

      auto &vCells = vVCells[pSinfo.lStrm[vS]];

      const auto cIdx = pSinfo.lIdxs[vS][vIi];

      if (!vCells.mIsEmpty(cIdx)) {
        assert(vCells.mSeqCount(cIdx) == 1);

        const mgs_seq_id cSeqId = vCells.mSeqGet(cIdx);
        const mgs_pos cPos = vCells.mPosGet(cIdx);

        aSeqPosMaxRm[cSeqId] = std::max(aSeqPosMaxRm[cSeqId], cPos);

        vCells.mRm(cIdx);
      }

      vCells.mPosSet(cIdx, pUbatch.vPos[cI]);

      if (pUbatch.mIsPos2d() || pUbatch.vToken || vHparams.vPleNHeads > 0) {
        mgs_kv_cell_ext vExt;

        if (pUbatch.mIsPos2d()) {
          vExt.vX = pUbatch.vPos[cI + pUbatch.vNTokens * 2];
          vExt.vY = pUbatch.vPos[cI + pUbatch.vNTokens];
        }

        if (pUbatch.vToken) {
          vExt.vTok = pUbatch.vToken[cI];
        } else if (vHparams.vPleNHeads > 0) {
          // an embd batch (multimodal input) has no token ids: the PLE layers hash the image token in its place
          vExt.vTok = vHparams.vPleImageTokenId != 0 ? (mgs_token)vHparams.vPleImageTokenId
                                                     : (mgs_token)vHparams.vPleEosTokenId;
        }

        vCells.mExtSet(cIdx, vExt);
      }

      for (int32_t vS = 0; vS < pUbatch.vNSeqId[cI]; vS++) {
        vCells.mSeqAdd(cIdx, pUbatch.vSeqId[cI][vS]);
      }
    }
  }

  // note: we want to preserve the invariant that all positions between [pos_min, pos_max] for each sequence
  //       will be present in the cache. so we have to purge any position which is less than those we would overwrite
  for (uint32_t vS = 0; vS < MGS_MAX_SEQ; ++vS) {
    if (aSeqPosMaxRm[vS] == -1) {
      continue;
    }

    GGML_ASSERT(vS < lSeqToStream.size());

    auto &vCells = vVCells[lSeqToStream[vS]];

    if (vCells.mSeqPosMin(vS) <= aSeqPosMaxRm[vS]) {
      MGS_LOG_DEBUG("%s: purging positions [%d, %d] of sequence %d from KV cache\n", __func__, vCells.mSeqPosMin(vS),
                    aSeqPosMaxRm[vS], vS);

      mSeqRm(vS, vCells.mSeqPosMin(vS), aSeqPosMaxRm[vS] + 1);
    }
  }

  // move the head at the end of the slot
  for (uint32_t vS = 0; vS < pSinfo.mNStream(); ++vS) {
    auto &vHead = lVHeads[pSinfo.lStrm[vS]];

    vHead = pSinfo.lIdxs[vS].back() + 1;
  }
}

bool mgs_kv_cache::mGetCanShift() const {
  // Step35 uses per-layer RoPE dims; K-shift assumes a single global n_rot.
  if (vModel.vArch == cLlmArchStep35) {
    return false;
  }
  if (vHparams.mNPosPerEmbd() > 1) {
    return false;
  }
  return true;
}

uint32_t mgs_kv_cache::mGetSize() const {
  const auto &vCells = vVCells[lSeqToStream[0]];

  return vCells.size();
}

uint32_t mgs_kv_cache::mGetNSeqMax() const { return cNSeqMax; }

uint32_t mgs_kv_cache::mGetNStream() const { return cNStream; }

bool mgs_kv_cache::mGetHasShift() const {
  bool vResult = false;

  for (uint32_t vS = 0; vS < cNStream; ++vS) {
    vResult |= vVCells[vS].mGetHasShift();
  }

  return vResult;
}

ggml_type mgs_kv_cache::mTypeK() const { return lLayers[0].vK->type; }

ggml_type mgs_kv_cache::mTypeV() const { return lLayers[0].vV->type; }

std::vector<uint32_t> mgs_kv_cache::mGetLayerIds() const {
  std::vector<uint32_t> lRes;
  lRes.reserve(lLayers.size());

  for (const auto &vLayer : lLayers) {
    lRes.push_back(vLayer.vIl);
  }

  return lRes;
}

ggml_tensor *mgs_kv_cache::mGetKStorage(int32_t pIl) const {
  const int32_t cIkv = dMapLayerIds.at(pIl);

  return lLayers[cIkv].vK;
}

const mgs_kv_cells &mgs_kv_cache::mGetCells(mgs_seq_id pSeqId) const {
  GGML_ASSERT(pSeqId >= 0 && (size_t)pSeqId < lSeqToStream.size());

  return vVCells[lSeqToStream[pSeqId]];
}

uint32_t mgs_kv_cache::mGetStream(mgs_seq_id pSeqId) const {
  GGML_ASSERT(pSeqId >= 0 && (size_t)pSeqId < lSeqToStream.size());

  return lSeqToStream[pSeqId];
}

uint32_t mgs_kv_cache::mGetNKv(const slot_info &pSinfo) const {
  uint32_t vResult = 0;

  // pad the n_kv value so that the graph remains constant across batches and can be reused
  // note: this also helps some backends with performance
  const uint32_t cNPadCur = std::max(cNPad, 256u);

  for (uint32_t vS = 0; vS < pSinfo.mNStream(); ++vS) {
    const auto &vCells = vVCells[pSinfo.lStrm[vS]];

    vResult = std::max(std::min(vCells.size(), std::max(cNPadCur, GGML_PAD(vCells.mUsedMaxP1(), cNPadCur))), vResult);
  }

  return vResult;
}

ggml_tensor *mgs_kv_cache::mGetK(ggml_context *pCtx, int32_t pIl, uint32_t pNKv, const slot_info &pSinfo) const {
  const int32_t cIkv = dMapLayerIds.at(pIl);

  auto *vK = lLayers[cIkv].vK;

  const uint64_t cKvSize = mGetSize();
  const uint64_t cNEmbdKGqa = vK->ne[0];

  assert(cNEmbdKGqa == vHparams.mNEmbdKGqa(pIl));

  const uint32_t cNs = pSinfo.vS1 - pSinfo.vS0 + 1;

  return ggml_view_4d(pCtx, vK, vHparams.mNEmbdHeadK(pIl), vHparams.mNHeadKv(pIl), pNKv, cNs,
                      ggml_row_size(vK->type, vHparams.mNEmbdHeadK(pIl)), ggml_row_size(vK->type, cNEmbdKGqa),
                      ggml_row_size(vK->type, cNEmbdKGqa * cKvSize),
                      ggml_row_size(vK->type, cNEmbdKGqa * cKvSize) * pSinfo.vS0);
}

ggml_tensor *mgs_kv_cache::mGetV(ggml_context *pCtx, int32_t pIl, uint32_t pNKv, const slot_info &pSinfo) const {
  const int32_t cIkv = dMapLayerIds.at(pIl);

  auto *vV = lLayers[cIkv].vV;

  const uint64_t cKvSize = mGetSize();
  const uint64_t cNEmbdVGqa = vV->ne[0];

  // [TAG_V_CACHE_VARIABLE]
  assert(cNEmbdVGqa >= vHparams.mNEmbdVGqa(pIl));

  const uint32_t cNs = pSinfo.vS1 - pSinfo.vS0 + 1;

  if (!vVTrans) {
    // note: v->nb[1] <= v->nb[2]
    return ggml_view_4d(pCtx, vV, vHparams.mNEmbdHeadV(pIl), vHparams.mNHeadKv(pIl), pNKv, cNs,
                        ggml_row_size(vV->type, vHparams.mNEmbdHeadV(pIl)), // v->nb[1]
                        ggml_row_size(vV->type, cNEmbdVGqa),                // v->nb[2]
                        ggml_row_size(vV->type, cNEmbdVGqa * cKvSize),      // v->nb[3]
                        ggml_row_size(vV->type, cNEmbdVGqa * cKvSize) * pSinfo.vS0);
  }

  // note: v->nb[1] > v->nb[2]
  return ggml_view_4d(pCtx, vV, pNKv, vHparams.mNHeadKv(pIl), vHparams.mNEmbdHeadV(pIl), cNs,
                      ggml_row_size(vV->type, cKvSize * vHparams.mNEmbdHeadV(pIl)), // v->nb[1]
                      ggml_row_size(vV->type, cKvSize),                             // v->nb[2]
                      ggml_row_size(vV->type, cKvSize * cNEmbdVGqa),                // v->nb[3]
                      ggml_row_size(vV->type, cKvSize * cNEmbdVGqa) * pSinfo.vS0);
}

ggml_tensor *mgs_kv_cache::mCpyK(ggml_context *pCtx, ggml_tensor *pKCur, ggml_tensor *pKIdxs, int32_t pIl,
                                 const slot_info &pSinfo) const {
  GGML_UNUSED(pSinfo);

  const int32_t cIkv = dMapLayerIds.at(pIl);

  ggml_tensor *vK = lLayers[cIkv].vK;

  const int64_t cNEmbdHead = pKCur->ne[0];
  const int64_t cNHead = pKCur->ne[1];
  const int64_t cNTokens = pKCur->ne[2];

  const int64_t cNEmbdGqa = cNEmbdHead * cNHead;

  // we can merge dims 0 and 1
  // TODO: add ggml helper function for this?
  GGML_ASSERT(ggml_row_size(pKCur->type, cNEmbdHead) == pKCur->nb[1]);

  pKCur = ggml_view_2d(pCtx, pKCur, cNEmbdGqa, cNTokens, pKCur->nb[2], 0);

  const int64_t cNStream = vK->ne[2];

  if (cNStream > 1) {
    const int64_t cKvSize = mGetSize();

    assert(cNEmbdGqa == vK->ne[0]);
    assert(cKvSize == vK->ne[1]);

    // merge the buffer across all streams because the idxs are global
    vK = ggml_reshape_2d(pCtx, vK, cNEmbdGqa, cKvSize * cNStream);
  }

  // store the current K values into the cache
  return ggml_set_rows(pCtx, vK, pKCur, pKIdxs);
}

ggml_tensor *mgs_kv_cache::mCpyV(ggml_context *pCtx, ggml_tensor *pVCur, ggml_tensor *pVIdxs, int32_t pIl,
                                 const slot_info &pSinfo) const {
  GGML_UNUSED(pSinfo);

  const int32_t cIkv = dMapLayerIds.at(pIl);

  auto *vV = lLayers[cIkv].vV;

  const int64_t cNEmbdHead = pVCur->ne[0];
  const int64_t cNHead = pVCur->ne[1];
  const int64_t cNTokens = pVCur->ne[2];

  const int64_t cNEmbdGqa = cNEmbdHead * cNHead;

  // we can merge dims 0 and 1
  GGML_ASSERT(ggml_row_size(pVCur->type, cNEmbdHead) == pVCur->nb[1]);

  const int64_t cNStream = vV->ne[2];

  // take this branch when FA is enabled (the V cache is not transposed)
  if (!vVTrans) {
    pVCur = ggml_view_2d(pCtx, pVCur, cNEmbdGqa, cNTokens, pVCur->nb[2], 0);

    if (cNStream > 1) {
      const int64_t cKvSize = mGetSize();

      assert(cNEmbdGqa == vV->ne[0]);
      assert(cKvSize == vV->ne[1]);

      // merge the buffer across all streams because the idxs are global
      vV = ggml_reshape_2d(pCtx, vV, cNEmbdGqa, cKvSize * cNStream);
    }

    return ggml_set_rows(pCtx, vV, pVCur, pVIdxs);
  }

  if (ggml_row_size(pVCur->type, cNEmbdGqa) == pVCur->nb[2]) {
    // we can merge dims 0, 1 and 2
    pVCur = ggml_reshape_2d(pCtx, pVCur, cNEmbdGqa, cNTokens);
  } else {
    // otherwise -> make a copy to get contiguous data
    pVCur = ggml_cont_2d(pCtx, pVCur, cNEmbdGqa, cNTokens);
  }

  // [TAG_V_CACHE_VARIABLE]
  if (cNEmbdGqa < vV->ne[0]) {
    pVCur = ggml_pad(pCtx, pVCur, vV->ne[0] - cNEmbdGqa, 0, 0, 0);
  }

  // in this branch the v_idxs are constructed in such a way that each row is a single head element
  ggml_tensor *vVView = ggml_reshape_2d(pCtx, vV, 1, ggml_nelements(vV));

  pVCur = ggml_reshape_2d(pCtx, pVCur, 1, ggml_nelements(pVCur));

  return ggml_set_rows(pCtx, vVView, pVCur, pVIdxs);
}

ggml_tensor *mgs_kv_cache::mBuildInputKIdxs(ggml_context *pCtx, const mgs_ubatch &pUbatch) const {
  const uint32_t cNTokens = pUbatch.vNTokens;

  ggml_tensor *vKIdxs = ggml_new_tensor_1d(pCtx, GGML_TYPE_I64, cNTokens);

  ggml_set_input(vKIdxs);

  return vKIdxs;
}

ggml_tensor *mgs_kv_cache::mBuildInputVIdxs(ggml_context *pCtx, const mgs_ubatch &pUbatch) const {
  const uint32_t cNTokens = pUbatch.vNTokens;

  ggml_tensor *vVIdxs;

  if (!vVTrans) {
    vVIdxs = ggml_new_tensor_1d(pCtx, GGML_TYPE_I64, cNTokens);
  } else {
    vVIdxs = ggml_new_tensor_1d(pCtx, GGML_TYPE_I64, cNTokens * vHparams.mNEmbdVGqaMax());
  }

  ggml_set_input(vVIdxs);

  return vVIdxs;
}

ggml_tensor *mgs_kv_cache::mBuildInputKRot(ggml_context *pCtx) const {
  ggml_tensor *vRes = nullptr;

  if (vAttnRotK) {
    int vNrot = 64;

    // TODO: investigate if using the smallest rotation matrix is beneficial also for K (similar as for V)
    do {
      vNrot *= 2;
    } while (vNEmbdHeadKAll % vNrot == 0);
    vNrot /= 2;

    vRes = ggml_new_tensor_2d(pCtx, GGML_TYPE_F32, vNrot, vNrot);
    ggml_set_input(vRes);
    ggml_set_name(vRes, "attn_inp_k_rot");
  }

  return vRes;
}

ggml_tensor *mgs_kv_cache::mBuildInputVRot(ggml_context *pCtx) const {
  ggml_tensor *vRes = nullptr;

  if (vAttnRotV) {
    int vNrot = 64;
    // using smaller rotation matrices for V seems beneficial
    // do {
    //    vNrot *= 2;
    //} while (vHparams.mNEmbdHeadV() % vNrot == 0);
    // vNrot /= 2;

    vRes = ggml_new_tensor_2d(pCtx, GGML_TYPE_F32, vNrot, vNrot);
    ggml_set_input(vRes);
    ggml_set_name(vRes, "attn_inp_v_rot");
  }

  return vRes;
}

void mgs_kv_cache::mSetInputKIdxs(ggml_tensor *pDst, const mgs_ubatch *pUbatch, const slot_info &pSinfo) const {
  const uint32_t cNTokens = pUbatch->vNTokens;
  GGML_ASSERT(cNTokens == (int64_t)pSinfo.size() * pSinfo.mNStream());

  GGML_ASSERT(ggml_backend_buffer_is_host(pDst->buffer));
  int64_t *vData = (int64_t *)pDst->data;

  for (uint32_t vS = 0; vS < pSinfo.mNStream(); ++vS) {
    const int64_t cOffs = pSinfo.lStrm[vS] * mGetSize();

    for (uint32_t vI = 0; vI < pSinfo.size(); ++vI) {
      vData[vS * pSinfo.size() + vI] = cOffs + pSinfo.lIdxs[vS][vI];
    }
  }
}

void mgs_kv_cache::mSetInputVIdxs(ggml_tensor *pDst, const mgs_ubatch *pUbatch, const slot_info &pSinfo) const {
  const uint32_t cNTokens = pUbatch->vNTokens;
  GGML_ASSERT(cNTokens == (int64_t)pSinfo.size() * pSinfo.mNStream());

  GGML_ASSERT(ggml_backend_buffer_is_host(pDst->buffer));
  int64_t *vData = (int64_t *)pDst->data;

  if (!vVTrans) {
    for (uint32_t vS = 0; vS < pSinfo.mNStream(); ++vS) {
      const int64_t cOffs = pSinfo.lStrm[vS] * mGetSize();

      for (uint32_t vI = 0; vI < pSinfo.size(); ++vI) {
        vData[vS * pSinfo.size() + vI] = cOffs + pSinfo.lIdxs[vS][vI];
      }
    }
  } else {
    // note: the V cache is transposed when not using flash attention
    const int64_t cKvSize = mGetSize();

    const int64_t cNEmbdVGqa = vHparams.mNEmbdVGqaMax();

    for (uint32_t vS = 0; vS < pSinfo.mNStream(); ++vS) {
      const int64_t cOffs = pSinfo.lStrm[vS] * cKvSize * cNEmbdVGqa;

      for (uint32_t vI = 0; vI < pSinfo.size(); ++vI) {
        for (uint32_t vJ = 0; vJ < cNEmbdVGqa; ++vJ) {
          vData[vS * pSinfo.size() * cNEmbdVGqa + vI * cNEmbdVGqa + vJ] = cOffs + vJ * cKvSize + pSinfo.lIdxs[vS][vI];
        }
      }
    }
  }
}

void mgs_kv_cache::mSetInputKShift(ggml_tensor *pDst) const {
  GGML_ASSERT(ggml_backend_buffer_is_host(pDst->buffer));

  int32_t *vData = (int32_t *)pDst->data;

  for (uint32_t vS = 0; vS < cNStream; ++vS) {
    const auto &vCells = vVCells[vS];

    for (uint32_t vI = 0; vI < vCells.size(); ++vI) {
      vData[vS * vCells.size() + vI] = vCells.mIsEmpty(vI) ? 0 : vCells.mGetShift(vI);
    }
  }
}

struct args_set_input_kq_mask {
  const mgs_hparams &vHparams;
  const mgs_ubatch *vUbatch;

  const std::vector<mgs_kv_cells> &lVCells;
  const std::vector<uint32_t> &lSeqToStream;

  uint32_t vNSwa;
  mgs_swa_type vSwaType;

  int64_t vNKv;
  int64_t vNStream;
  int64_t vNTps;
};

template <typename T, bool causal, bool swa, bool cIs2d, bool alibi>
static void fSetInputKqMaskImpl(const args_set_input_kq_mask &pArgs, T *pData) {
  // const auto & vHparams = pArgs.vHparams;
  const auto &vUbatch = pArgs.vUbatch;

  const auto &vVCells = pArgs.lVCells;
  const auto &vSeqToStream = pArgs.lSeqToStream;

  const uint32_t cNSwa = pArgs.vNSwa;
  const mgs_swa_type cSwaType = pArgs.vSwaType;

  const int64_t cNKv = pArgs.vNKv;
  const int64_t cNStream = pArgs.vNStream;
  const int64_t cNTps = pArgs.vNTps;

  const T cMaskKeep = mgs_cast<T>(0.0f);
  const T cMaskDrop = mgs_cast<T>(-INFINITY);

  // the min position in the batch for each sequence
  mgs_pos aSeqPosMin[MGS_MAX_SEQ];
  std::fill(aSeqPosMin, aSeqPosMin + MGS_MAX_SEQ, INT32_MAX);

  for (uint32_t vI = 0; vI < vUbatch->vNTokens; ++vI) {
    const mgs_seq_id cSeqId = vUbatch->vSeqId[vI][0];

    aSeqPosMin[cSeqId] = std::min(aSeqPosMin[cSeqId], vUbatch->vPos[vI]);
  }

  for (uint32_t vS = 0; vS < cNStream; ++vS) {
    // bookkeeping of the KQ mask cells that could change for other tokens of the same sequence
    std::unordered_map<mgs_seq_id, uint32_t> dSeqSrct;
    std::unordered_map<mgs_seq_id, std::vector<uint32_t>> dSeqIdxs;

    for (uint32_t vIi = 0; vIi < cNTps; ++vIi) {
      const uint32_t cI = vS * cNTps + vIi;

      const mgs_seq_id cSeqId = vUbatch->vSeqId[cI][0];

      const auto &vCells = vVCells.at(vSeqToStream[cSeqId]);

      mgs_pos vP0 = -1;
      const mgs_pos cP1 = vUbatch->vPos[cI];

      // for M-RoPE
      const mgs_pos cP1X = cIs2d ? vUbatch->vPos[cI + vUbatch->vNTokens * 2] : 0;
      const mgs_pos cP1Y = cIs2d ? vUbatch->vPos[cI + vUbatch->vNTokens] : 0;

      const uint64_t cIdst = cNKv * cI;

      // for tokens of the same sequence, the mask is mostly the same, so we can reuse it
      // the only cells that could change are the ones that are with similar positions as the
      //   ones in the batch (i.e. due to causal masking, SWA, etc.)
      // keep track of those cells and shortcut the loop to save time
      // note: this optimization is not compatible with Alibi position encoding
      bool vPrev = false;

      auto &vIdxs = dSeqIdxs[cSeqId];

      if (!alibi) {
        if (dSeqSrct.find(cSeqId) != dSeqSrct.end()) {
          const uint32_t cSrct = dSeqSrct[cSeqId];

          const uint64_t cIdstPrev = cNKv * cSrct;

          std::copy(pData + cIdstPrev, pData + cIdstPrev + cNKv, pData + cIdst);

          vPrev = true;
        } else {
          vIdxs.clear();
          vIdxs.reserve(vUbatch->vNTokens + cNSwa + 32);

          dSeqSrct[cSeqId] = cI;
        }
      }

      for (uint32_t vJj = 0; vJj < cNKv; ++vJj) {
        uint32_t vJ = vJj;

        // we have an exiting mask for this sequence -> update just seq_idxs
        if (!alibi) {
          if (vPrev) {
            if (vJj >= vIdxs.size()) {
              break;
            }

            vJ = vIdxs[vJj];
          }
        }

        if (vCells.mIsEmpty(vJ)) {
          goto skip;
        }

        // mask the token if not the same sequence
        if (!vCells.mSeqHas(vJ, cSeqId)) {
          goto skip;
        }

        vP0 = vCells.mPosGet(vJ);

        if (!alibi) {
          if (!vPrev) {
            // record all cells for which: p0 >= seq_pos_min[seq_id] - n_swa - 32
            if (vP0 + (int32_t)(cNSwa + 32) >= aSeqPosMin[cSeqId]) {
              vIdxs.push_back(vJ);
            }
          }
        }

        if (causal) {
          // mask future tokens
          if (vP0 > cP1) {
            goto skip;
          }

          // M-RoPE causal mask
          if (cIs2d) {
            if (vP0 == cP1) {
              const auto &vP0Ext = vCells.mExtGet(vJ);

              if (vP0Ext.mIs2dGt(cP1X, cP1Y)) {
                goto skip;
              }
            }
          }
        }

        // apply SWA if any
        if (swa) {
          if (mgs_hparams::mIsMaskedSwa(cNSwa, cSwaType, vP0, cP1)) {
            goto skip;
          }
        }

        if (alibi) {
          pData[cIdst + vJ] = mgs_cast<T>(static_cast<float>(-std::abs(vP0 - cP1)));
        } else {
          pData[cIdst + vJ] = cMaskKeep;
        }

        continue;
      skip:
        pData[cIdst + vJ] = cMaskDrop;
      }
    }
  }
}

template <typename T, bool causal, bool swa, bool cIs2d>
static void fSetInputKqMaskImpl(const args_set_input_kq_mask &pArgs, T *pData) {
  const bool cAlibi = pArgs.vHparams.vUseAlibi;
  if (cAlibi) {
    fSetInputKqMaskImpl<T, causal, swa, cIs2d, true>(pArgs, pData);
  } else {
    fSetInputKqMaskImpl<T, causal, swa, cIs2d, false>(pArgs, pData);
  }
}

template <typename T, bool causal, bool swa>
static void fSetInputKqMaskImpl(const args_set_input_kq_mask &pArgs, T *pData) {
  const bool cIs2d = pArgs.vUbatch->mIsPos2d();
  if (cIs2d) {
    fSetInputKqMaskImpl<T, causal, swa, true>(pArgs, pData);
  } else {
    fSetInputKqMaskImpl<T, causal, swa, false>(pArgs, pData);
  }
}

template <typename T, bool causal> static void fSetInputKqMaskImpl(const args_set_input_kq_mask &pArgs, T *pData) {
  const bool cSwa = pArgs.vSwaType != cMgsSwaTypeNone;
  if (cSwa) {
    fSetInputKqMaskImpl<T, causal, true>(pArgs, pData);
  } else {
    fSetInputKqMaskImpl<T, causal, false>(pArgs, pData);
  }
}

template <typename T> static void fSetInputKqMaskImpl(const args_set_input_kq_mask &pArgs, T *pData, bool pCausalAttn) {
  if (pCausalAttn) {
    fSetInputKqMaskImpl<T, true>(pArgs, pData);
  } else {
    fSetInputKqMaskImpl<T, false>(pArgs, pData);
  }
}

void mgs_kv_cache::mSetInputKqMask(ggml_tensor *pDst, const mgs_ubatch *pUbatch, bool pCausalAttn) const {
  const uint32_t cNTokens = pUbatch->vNTokens;

  GGML_ASSERT(ggml_backend_buffer_is_host(pDst->buffer));

  const int64_t cNKv = pDst->ne[0];
  const int64_t cNStream = pDst->ne[3]; // num streams in the current ubatch

  GGML_ASSERT(cNTokens % cNStream == 0);

  // n_tps == n_tokens_per_stream
  const int64_t cNTps = cNTokens / cNStream;

  // const int64_t cTStart = ggml_time_us();

  const args_set_input_kq_mask cArgs = {
      /*.vHparams         =*/vHparams,
      /*.vUbatch          =*/pUbatch,
      /*.lVCells          =*/vVCells,
      /*.lSeqToStream     =*/lSeqToStream,
      /*.vNSwa            =*/cNSwa,
      /*.vSwaType         =*/cSwaType,
      /*.vNKv             =*/cNKv,
      /*.vNStream         =*/cNStream,
      /*.vNTps            =*/cNTps,
  };

  if (pDst->type == GGML_TYPE_F16) {
    fSetInputKqMaskImpl<ggml_fp16_t>(cArgs, (ggml_fp16_t *)pDst->data, pCausalAttn);
  } else {
    fSetInputKqMaskImpl<float>(cArgs, (float *)pDst->data, pCausalAttn);
  }

  // const int64_t cTEnd = ggml_time_us();

  // MGS_LOG_ERROR("%s: kq mask time: %0.3f ms\n", __func__, (cTEnd - cTStart)/1000.0);
}

void mgs_kv_cache::mSetInputPosBucket(ggml_tensor *pDst, const mgs_ubatch *pUbatch) const {
  const int64_t cNTokens = pUbatch->vNTokens;

  GGML_ASSERT(cNStream == 1 && "TODO: support multiple streams");
  const auto &vCells = vVCells[0];

  GGML_ASSERT(ggml_backend_buffer_is_host(pDst->buffer));
  GGML_ASSERT(!pUbatch->mEqualSeqs()); // TODO: use ubatch->n_seqs instead of failing

  int32_t *vData = (int32_t *)pDst->data;

  const int32_t cNKv = pDst->ne[0];

  for (int vH = 0; vH < 1; ++vH) {
    for (int vI = 0; vI < cNTokens; ++vI) {
      for (int vJ = 0; vJ < cNKv; ++vJ) {
        // the position when the cells is empty is irrelevant - it will be masked out later in the attention
        const mgs_pos cP0 = vCells.mIsEmpty(vJ) ? -1 : vCells.mPosGet(vJ);

        vData[vH * (cNKv * cNTokens) + vI * cNKv + vJ] =
            mgs_relative_position_bucket(cP0, pUbatch->vPos[vI], vHparams.vNRelAttnBkts, false);
      }
    }
  }
}

void mgs_kv_cache::mSetInputKRot(ggml_tensor *pDst) const {
  GGML_ASSERT(ggml_backend_buffer_is_host(pDst->buffer));

  const auto cNRot = pDst->ne[0];
  GGML_ASSERT(dAttnRotHadamard.count(pDst->ne[0]));

  memcpy(pDst->data, dAttnRotHadamard.at(cNRot).data(), ggml_nbytes(pDst));
}

void mgs_kv_cache::mSetInputVRot(ggml_tensor *pDst) const {
  GGML_ASSERT(ggml_backend_buffer_is_host(pDst->buffer));

  const auto cNRot = pDst->ne[0];
  GGML_ASSERT(dAttnRotHadamard.count(pDst->ne[0]));

  memcpy(pDst->data, dAttnRotHadamard.at(cNRot).data(), ggml_nbytes(pDst));
}

bool mgs_kv_cache::mHasCellExt() const {
  // M-RoPE needs the 2D position, the PLE and engram n-gram hashes need the token id
  return vHparams.mNPosPerEmbd() > 1 || vHparams.vPleNHeads > 0 || vHparams.vEngramNHead > 0;
}

void mgs_kv_cache::mGetPrevTokens(const mgs_ubatch &pUbatch, uint32_t pN, std::vector<mgs_token> &pRes) const {
  const uint32_t cNTokens = pUbatch.vNTokens;

  pRes.clear();
  pRes.resize(cNTokens * pN, MGS_TOKEN_NULL);

  if (pN == 0) {
    return;
  }

  // note: mApplyUbatch() has already stored the current ubatch, so the cells cover the tokens
  //       of this very ubatch as well, which is what we want
  // the nearest cell at or before a position also resolves M-RoPE gaps, where multiple tokens
  // share the same temporal pos

  // an embd (multimodal) ubatch can repeat one position for a whole image, so positions
  // do not encode the token order; resolve its predecessors by ubatch order instead
  const bool cByOrder = !pUbatch.vToken;

  std::vector<uint32_t> lOrd; // index among the ubatch tokens of the same seq
  std::unordered_map<mgs_seq_id, std::vector<uint32_t>> dSeqIdx;

  if (cByOrder) {
    lOrd.resize(cNTokens);
    for (uint32_t vI = 0; vI < cNTokens; ++vI) {
      auto &lV = dSeqIdx[pUbatch.vSeqId[vI][0]];
      lOrd[vI] = lV.size();
      lV.push_back(vI);
    }
  }

  for (uint32_t vI = 0; vI < cNTokens; ++vI) {
    // TODO: a token that belongs to more than one sequence has an ambiguous history.
    //       the n-gram architectures have to reject such batches
    const mgs_seq_id cSeqId = pUbatch.vSeqId[vI][0];

    for (uint32_t vJ = 0; vJ < pN; ++vJ) {
      const mgs_pos cD = (mgs_pos)(pN - vJ);

      mgs_pos vP;
      if (cByOrder) {
        const auto &lV = dSeqIdx[cSeqId];
        const int64_t cK = (int64_t)lOrd[vI] - cD;
        // k >= 0: an earlier token of this very ubatch; k < 0: before the chunk
        vP = cK >= 0 ? pUbatch.vPos[lV[cK]] : pUbatch.vPos[lV[0]] + (mgs_pos)cK;
      } else {
        vP = pUbatch.vPos[vI] - cD;
      }

      if (vP < 0) {
        continue;
      }

      pRes[vI * pN + vJ] = vVCells[lSeqToStream[cSeqId]].mSeqPosTokLe(cSeqId, vP);
    }
  }
}

size_t mgs_kv_cache::mTotalSize() const {
  size_t vSize = 0;

  for (const auto &[_, vBuf] : lCtxsBufs) {
    vSize += ggml_backend_buffer_get_size(vBuf.get());
  }

  return vSize;
}

size_t mgs_kv_cache::mSizeKBytes() const {
  size_t vSizeKBytes = 0;

  for (const auto &vLayer : lLayers) {
    vSizeKBytes += ggml_nbytes(vLayer.vK);
  }

  return vSizeKBytes;
}

size_t mgs_kv_cache::mSizeVBytes() const {
  size_t vSizeVBytes = 0;

  for (const auto &vLayer : lLayers) {
    vSizeVBytes += vLayer.vV ? ggml_nbytes(vLayer.vV) : 0;
  }

  return vSizeVBytes;
}

ggml_tensor *mgs_kv_cache::mBuildRopeShift(const mgs_cparams &pCparams, ggml_context *pCtx, ggml_tensor *pCur,
                                           ggml_tensor *pShift, ggml_tensor *pRot, ggml_tensor *pFactors,
                                           float pFreqBase, float pFreqScale, uint32_t pIl) const {
  const auto &vNCtxOrig = pCparams.vNCtxOrigYarn;

  const auto &vYarnExtFactor = pCparams.vYarnExtFactor;
  const auto &vYarnBetaFast = pCparams.vYarnBetaFast;
  const auto &vYarnBetaSlow = pCparams.vYarnBetaSlow;
  const auto &vYarnAttnFactor = pCparams.vYarnAttnFactor;

  const auto &vNRot = vHparams.mNRot(pIl);
  const auto &vRopeType = vHparams.vRopeType == MGS_ROPE_TYPE_MROPE || vHparams.vRopeType == MGS_ROPE_TYPE_IMROPE
                              // @ngxson : this is a workaround
                              // for M-RoPE, we want to rotate the whole vector when doing KV shift
                              // a normal RoPE should work, we just need to use the correct ordering
                              ? MGS_ROPE_TYPE_NEOX
                              : vHparams.vRopeType;
  ggml_tensor *vTmp;

  if (ggml_is_quantized(pCur->type)) {
    // dequantize to f32 -> RoPE -> quantize back
    vTmp = ggml_cast(pCtx, pCur, GGML_TYPE_F32);

    // rotate back
    vTmp = mgs_mul_mat_hadamard(pCtx, vTmp, pRot);

    vTmp = ggml_rope_ext(pCtx, vTmp, pShift, pFactors, vNRot, vRopeType, vNCtxOrig, pFreqBase, pFreqScale,
                         vYarnExtFactor, vYarnAttnFactor, vYarnBetaFast, vYarnBetaSlow);

    // rotate fwd
    vTmp = mgs_mul_mat_hadamard(pCtx, vTmp, pRot);

    vTmp = ggml_cpy(pCtx, vTmp, pCur);
  } else {
    // we rotate only the first n_rot dimensions
    vTmp = ggml_rope_ext_inplace(pCtx, pCur, pShift, pFactors, vNRot, vRopeType, vNCtxOrig, pFreqBase, pFreqScale,
                                 vYarnExtFactor, vYarnAttnFactor, vYarnBetaFast, vYarnBetaSlow);
  }

  return vTmp;
}

class llm_graph_input_k_shift : public llm_graph_input_i {
public:
  llm_graph_input_k_shift(const mgs_kv_cache *pKvSelf) : vKvSelf(pKvSelf) {}
  virtual ~llm_graph_input_k_shift() = default;

  void mSetInput(const mgs_ubatch *pUbatch) override;

  ggml_tensor *vKShift; // I32 [kv_size*n_stream]

  // note: assumes k_rot^2 == I
  ggml_tensor *vKRot = nullptr;

  const mgs_kv_cache *vKvSelf;
};

void llm_graph_input_k_shift::mSetInput(const mgs_ubatch *pUbatch) {
  GGML_UNUSED(pUbatch);

  if (vKShift) {
    vKvSelf->mSetInputKShift(vKShift);
  }

  if (vKRot) {
    vKvSelf->mSetInputKRot(vKRot);
  }
}

ggml_cgraph *mgs_kv_cache::mBuildGraphShift(llm_graph_result *pRes, mgs_context *pLctx) const {
  // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
  GGML_ASSERT(!vOther);

  auto *vCtx = pRes->mGetCtx();
  auto *vGf = pRes->mGetGf();

  auto vInp = std::make_unique<llm_graph_input_k_shift>(this);

  vInp->vKShift = ggml_new_tensor_1d(vCtx, GGML_TYPE_I32, (int64_t)mGetSize() * cNStream);
  ggml_set_input(vInp->vKShift);

  vInp->vKRot = mBuildInputKRot(vCtx);

  const auto &vCparams = pLctx->mGetCparams();

  for (const auto &vLayer : lLayers) {
    const uint32_t cIl = vLayer.vIl;

    const int64_t cNHeadKv = vHparams.mNHeadKv(cIl);
    const int64_t cNEmbdKGqa = vHparams.mNEmbdKGqa(cIl);

    const auto cNRot = vHparams.mNRot(cIl);
    const auto cNEmbdHeadK = vHparams.mNEmbdHeadK(cIl);
    const auto cNEmbdNope = vHparams.vNLoraKv > 0 ? cNEmbdHeadK - cNRot : 0;

    const float cFreqBaseL = vModel.mGetRopeFreqBase(vCparams, cIl);
    const float cFreqScaleL = vModel.mGetRopeFreqScale(vCparams, cIl);

    ggml_tensor *vRopeFactors = vModel.mGetRopeFactors(vCparams, cIl);

    ggml_tensor *vK = ggml_view_3d(
        vCtx, vLayer.vK, cNRot, cNHeadKv, mGetSize() * cNStream, ggml_row_size(vLayer.vK->type, cNEmbdHeadK),
        ggml_row_size(vLayer.vK->type, cNEmbdKGqa), ggml_row_size(vLayer.vK->type, cNEmbdNope));

    ggml_tensor *vCur =
        mBuildRopeShift(vCparams, vCtx, vK, vInp->vKShift, vInp->vKRot, vRopeFactors, cFreqBaseL, cFreqScaleL, cIl);

    ggml_build_forward_expand(vGf, vCur);
  }

  pRes->mAddInput(std::move(vInp));

  return vGf;
}

void mgs_kv_cache::mStateWrite(mgs_io_write_i &pIo, mgs_seq_id pSeqId, mgs_state_seq_flags pFlags) const {
  // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
  if (vOther) {
    return;
  }

  GGML_UNUSED(pFlags);

  pIo.mWrite(&cNStream, sizeof(cNStream));

  for (uint32_t vS = 0; vS < cNStream; ++vS) {
    cell_ranges_t vCr{vS, {}};

    uint32_t vCellCount = 0;

    const auto &vCells = vVCells[vS];

    // Count the number of cells with the specified seq_id
    // Find all the ranges of cells with this seq id (or all, when -1)
    uint32_t vCellRangeBegin = vCells.size();

    for (uint32_t vI = 0; vI < vCells.size(); ++vI) {
      bool vAddCell = true;

      vAddCell = vAddCell && !vCells.mIsEmpty(vI);
      vAddCell = vAddCell && (pSeqId == -1 || vCells.mSeqHas(vI, pSeqId));

      // check the cell is not SWA-masked
      if (vAddCell && pSeqId != -1) {
        const bool cIsMasked =
            mgs_hparams::mIsMaskedSwa(cNSwa, cSwaType, vCells.mPosGet(vI), vCells.mSeqPosMax(pSeqId));

        vAddCell = !cIsMasked;
      }

      if (vAddCell) {
        ++vCellCount;
        if (vCellRangeBegin == vCells.size()) {
          vCellRangeBegin = vI;
        }
      } else {
        if (vCellRangeBegin != vCells.size()) {
          vCr.lData.emplace_back(vCellRangeBegin, vI);
          vCellRangeBegin = vCells.size();
        }
      }
    }

    if (vCellRangeBegin != vCells.size()) {
      vCr.lData.emplace_back(vCellRangeBegin, vCells.size());
    }

    // DEBUG CHECK: Sum of cell counts in ranges should equal the total cell count
    uint32_t vCellCountCheck = 0;
    for (const auto &vRange : vCr.lData) {
      vCellCountCheck += vRange.second - vRange.first;
    }
    GGML_ASSERT(vCellCount == vCellCountCheck);

    pIo.mWrite(&vCellCount, sizeof(vCellCount));

    // skip empty streams
    if (vCellCount == 0) {
      continue;
    }

    mStateWriteMeta(pIo, vCr, pSeqId);
    mStateWriteData(pIo, vCr);
  }
}

void mgs_kv_cache::mStateRead(mgs_io_read_i &pIo, mgs_seq_id pSeqId, mgs_state_seq_flags pFlags) {
  mStateReadSinfo(pIo, pSeqId, pFlags, nullptr, nullptr);
}

void mgs_kv_cache::mStateReadSinfo(mgs_io_read_i &pIo, mgs_seq_id pSeqId, mgs_state_seq_flags pFlags,
                                   slot_info_vec_t *pSinfosOut, const slot_info_vec_t *pSinfosIn) {
  // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
  if (vOther) {
    return;
  }

  GGML_UNUSED(pFlags);

  GGML_ASSERT(pSeqId == -1 || (pSeqId >= 0 && (size_t)pSeqId < lSeqToStream.size()));

  if (pSinfosOut) {
    pSinfosOut->assign(cNStream, slot_info{});
  }

  if (pSinfosIn && pSinfosIn->size() != cNStream) {
    throw std::runtime_error("failed to restore kv cache: mirrored slot layout has the wrong stream count");
  }

  uint32_t vNStreamCur;
  pIo.mRead(&vNStreamCur, sizeof(vNStreamCur));
  if (vNStreamCur != cNStream) {
    throw std::runtime_error("n_stream mismatch");
  }

  // a whole-context restore replaces every stream, so the cache is emptied once here
  // mClear() resets all streams at once, so doing it per stream below would keep only the last one
  if (pSeqId == -1) {
    mClear(true);
  }

  for (uint32_t vS = 0; vS < cNStream; ++vS) {
    uint32_t vCellCount;
    pIo.mRead(&vCellCount, sizeof(vCellCount));

    if (vCellCount == 0) {
      // a mirrored cache must be empty here as well, or the two no longer agree cell for cell
      if (pSinfosIn && !(*pSinfosIn)[vS].empty()) {
        throw std::runtime_error("failed to restore kv cache: mirrored cache holds cells this one does not");
      }
      continue;
    }

    const uint32_t cStrm = pSeqId == -1 ? vS : lSeqToStream[pSeqId];

    slot_info vSinfo;

    bool vRes = true;
    vRes = vRes && mStateReadMeta(pIo, cStrm, vCellCount, vSinfo, pSeqId, pSinfosIn ? &(*pSinfosIn)[vS] : nullptr);

    try {
      vRes = vRes && mStateReadData(pIo, cStrm, vCellCount, vSinfo);
    } catch (...) {
      vRes = false;
    }

    if (!vRes) {
      mStateClear(pSeqId);
      throw std::runtime_error("failed to restore kv cache");
    }

    if (pSinfosOut) {
      (*pSinfosOut)[vS] = vSinfo;
    }
  }
}

void mgs_kv_cache::mStateClear(mgs_seq_id pSeqId) {
  if (pSeqId == -1) {
    mClear(true);
    return;
  }

  GGML_ASSERT(pSeqId >= 0 && (size_t)pSeqId < lSeqToStream.size());

  mSeqRm(pSeqId, -1, -1);
}

void mgs_kv_cache::mStateWriteMeta(mgs_io_write_i &pIo, const cell_ranges_t &pCr, mgs_seq_id pSeqId) const {
  const auto &vCells = vVCells[pCr.vStrm];

  for (const auto &vRange : pCr.lData) {
    for (uint32_t vI = vRange.first; vI < vRange.second; ++vI) {
      std::vector<mgs_seq_id> lSeqIds;

      for (mgs_seq_id vCur = 0; vCur < (int)cNSeqMax; ++vCur) {
        if (vCur == pSeqId || pSeqId == -1) {
          if (vCells.mSeqHas(vI, vCur)) {
            lSeqIds.push_back(vCur);
          }
        }
      }

      const mgs_pos cPos = vCells.mPosGet(vI);
      const uint32_t cNSeqId = lSeqIds.size();

      pIo.mWrite(&cPos, sizeof(cPos));
      pIo.mWrite(&cNSeqId, sizeof(cNSeqId));

      if (mHasCellExt()) {
        const mgs_kv_cell_ext cExt = vCells.mExtGet(vI);
        pIo.mWrite(&cExt, sizeof(cExt));
      }

      for (const auto &vSeqId : lSeqIds) {
        pIo.mWrite(&vSeqId, sizeof(vSeqId));
      }
    }
  }
}

void mgs_kv_cache::mStateWriteData(mgs_io_write_i &pIo, const cell_ranges_t &pCr) const {
  const auto &vCells = vVCells[pCr.vStrm];

  const uint32_t cVTrans = this->vVTrans ? 1 : 0;
  const uint32_t cNLayer = lLayers.size();

  pIo.mWrite(&cVTrans, sizeof(cVTrans));
  pIo.mWrite(&cNLayer, sizeof(cNLayer));

  // Iterate and write all the keys first, each row is a cell
  // Get whole range at a time
  for (const auto &vLayer : lLayers) {
    const uint32_t cIl = vLayer.vIl;

    const uint32_t cNEmbdKGqa = vHparams.mNEmbdKGqa(cIl);

    auto *vK = vLayer.lKStream[pCr.vStrm];

    // Write key type
    const int32_t cKTypeI = (int32_t)vK->type;
    pIo.mWrite(&cKTypeI, sizeof(cKTypeI));

    // Write row size of key
    const uint64_t cKSizeRow = ggml_row_size(vK->type, cNEmbdKGqa);
    pIo.mWrite(&cKSizeRow, sizeof(cKSizeRow));

    // Read each range of cells of k_size length and write out
    for (const auto &vRange : pCr.lData) {
      const size_t cRangeSize = vRange.second - vRange.first;
      const size_t cBufSize = cRangeSize * cKSizeRow;
      pIo.mWriteTensor(vK, vRange.first * cKSizeRow, cBufSize);
    }
  }

  if (!cVTrans) {
    for (const auto &vLayer : lLayers) {
      const uint32_t cIl = vLayer.vIl;

      const uint32_t cNEmbdVGqa = vHparams.mNEmbdVGqa(cIl);

      auto *vV = vLayer.lVStream[pCr.vStrm];
      if (!vV) {
        continue;
      }

      // Write value type
      const int32_t cVTypeI = (int32_t)vV->type;
      pIo.mWrite(&cVTypeI, sizeof(cVTypeI));

      // Write row size of value
      const uint64_t cVSizeRow = ggml_row_size(vV->type, cNEmbdVGqa);
      pIo.mWrite(&cVSizeRow, sizeof(cVSizeRow));

      // Read each range of cells of v_size length and write out
      for (const auto &vRange : pCr.lData) {
        const size_t cRangeSize = vRange.second - vRange.first;
        const size_t cBufSize = cRangeSize * cVSizeRow;
        pIo.mWriteTensor(vV, vRange.first * cVSizeRow, cBufSize);
      }
    }
  } else {
    // When v is transposed, we also need the element size and get the element ranges from each row
    const uint32_t cKvSize = vCells.size();

    for (const auto &vLayer : lLayers) {
      const uint32_t cIl = vLayer.vIl;

      const uint32_t cNEmbdVGqa = vHparams.mNEmbdVGqa(cIl);

      auto *vV = vLayer.lVStream[pCr.vStrm];
      if (!vV) {
        continue;
      }

      // Write value type
      const int32_t cVTypeI = (int32_t)vV->type;
      pIo.mWrite(&cVTypeI, sizeof(cVTypeI));

      // Write element size
      const uint32_t cVSizeEl = ggml_type_size(vV->type);
      pIo.mWrite(&cVSizeEl, sizeof(cVSizeEl));

      // Write GQA embedding size
      pIo.mWrite(&cNEmbdVGqa, sizeof(cNEmbdVGqa));

      // For each row, we get the element values of each cell
      for (uint32_t vJ = 0; vJ < cNEmbdVGqa; ++vJ) {
        // Read each range of cells of v_size_el length and write out
        for (const auto &vRange : pCr.lData) {
          const size_t cRangeSize = vRange.second - vRange.first;
          const size_t cSrcOffset = (vRange.first + vJ * cKvSize) * cVSizeEl;
          const size_t cBufSize = cRangeSize * cVSizeEl;
          pIo.mWriteTensor(vV, cSrcOffset, cBufSize);
        }
      }
    }
  }
}

bool mgs_kv_cache::mStateReadMeta(mgs_io_read_i &pIo, uint32_t pStrm, uint32_t pCellCount, slot_info &pSinfo,
                                  mgs_seq_id pDestSeqId, const slot_info *pSinfoIn) {
  auto &vCells = vVCells[pStrm];
  auto &vHead = lVHeads[pStrm];

  if (pDestSeqId != -1) {
    // single sequence
    mSeqRm(pDestSeqId, -1, -1);

    mgs_batch_allocr vBalloc(vHparams.mNPosPerEmbd());

    mgs_ubatch vUbatch = vBalloc.mUbatchReserve(pCellCount, 1);

    vUbatch.vSeqIdUnq[0] = pDestSeqId;

    // the ext as it was saved, to put back after mApplyUbatch()
    std::vector<mgs_kv_cell_ext> lExts;
    if (mHasCellExt()) {
      lExts.resize(pCellCount);
    }

    for (uint32_t vI = 0; vI < pCellCount; ++vI) {
      mgs_pos vPos;
      uint32_t vNSeqId;

      pIo.mRead(&vPos, sizeof(vPos));
      pIo.mRead(&vNSeqId, sizeof(vNSeqId));

      if (vNSeqId != 1) {
        MGS_LOG_ERROR("%s: invalid seq_id-agnostic kv cell\n", __func__);
        return false;
      }

      if (mHasCellExt()) {
        mgs_kv_cell_ext vExt;
        pIo.mRead(&vExt, sizeof(vExt));

        if (vHparams.mNPosPerEmbd() > 1) {
          vUbatch.vPos[vI + vUbatch.vNTokens] = vExt.vY;
          vUbatch.vPos[vI + vUbatch.vNTokens * 2] = vExt.vX;
        }

        // mApplyUbatch() below restores ext.vTok from the ubatch tokens
        vUbatch.vToken[vI] = vExt.vTok;

        lExts[vI] = vExt;
      }

      // read the sequence id, but directly discard it - we will use dest_seq_id instead
      {
        mgs_seq_id vSeqId;
        pIo.mRead(&vSeqId, sizeof(vSeqId));
      }

      vUbatch.vPos[vI] = vPos;
      vUbatch.vNSeqId[vI] = vNSeqId;
      vUbatch.vSeqId[vI] = &pDestSeqId;
    }

    if (pSinfoIn) {
      // this cache mirrors another one, so it takes that cache's layout instead of searching for its own cells
      if (pSinfoIn->empty() || pSinfoIn->mNStream() != 1 || pSinfoIn->lIdxs[0].size() != pCellCount) {
        MGS_LOG_ERROR("%s: mirrored slot layout holds %d cells, this cache restores %d\n", __func__,
                      pSinfoIn->empty() ? 0 : (int)pSinfoIn->lIdxs[0].size(), pCellCount);
        return false;
      }

      pSinfo = *pSinfoIn;

      // the layout is cell indices, so it means the same in both caches only while their streams line up
      pSinfo.vS0 = pStrm;
      pSinfo.vS1 = pStrm;
      pSinfo.lStrm[0] = pStrm;

      // mSeqRm above freed exactly the cells this sequence held
      // anything else in the way is a cache that had already drifted, which this restore must not hide
      for (uint32_t vI = 0; vI < pCellCount; ++vI) {
        const uint32_t cIdx = pSinfo.lIdxs[0][vI];

        if (cIdx >= vCells.size() || !vCells.mIsEmpty(cIdx)) {
          MGS_LOG_ERROR("%s: cell %u of the mirrored slot layout is not free\n", __func__, cIdx);
          return false;
        }
      }
    } else {
      pSinfo = mFindSlot(vUbatch, false);
      if (pSinfo.empty()) {
        MGS_LOG_ERROR("%s: failed to find %d available cells in kv cache\n", __func__, pCellCount);
        return false;
      }
    }

    // note: mApplyUbatch() rebuilds mgs_kv_cell_ext from the ubatch
    //       only ext.vTok and the M-RoPE 2D position round-trip through it
    mApplyUbatch(pSinfo, vUbatch);

    // mApplyUbatch() takes the 2D position from the ubatch, and that ubatch is built with this
    // cache's own n_pos_per_embd. a cache that does not use M-RoPE itself but mirrors one that
    // does (the qwen4exp QSA indexer) would drop x and y. put the saved ext back instead, which
    // is what the whole-context path below already does.
    for (uint32_t vI = 0; vI < (uint32_t)lExts.size(); ++vI) {
      vCells.mExtSet(pSinfo.lIdxs[0][vI], lExts[vI]);
    }

    MGS_LOG_DEBUG("%s: cell_count = %d, dest_seq_id = %d\n", __func__, pCellCount, pDestSeqId);

    // DEBUG CHECK: verify that all cells were allocated and have correct seq_id and pos values
    GGML_ASSERT(pSinfo.mNStream() == 1);
    GGML_ASSERT(pSinfo.lIdxs[0].size() == pCellCount);
    for (uint32_t vI = 0; vI < pCellCount; ++vI) {
      const uint32_t cIdx = pSinfo.lIdxs[0][vI];
      GGML_ASSERT(vCells.mPosGet(cIdx) == vUbatch.vPos[vI]);
      GGML_ASSERT(vCells.mSeqHas(cIdx, pDestSeqId));
    }
  } else {
    // whole KV cache restore

    if (pCellCount > vCells.size()) {
      MGS_LOG_ERROR("%s: not enough cells in kv cache\n", __func__);
      return false;
    }

    // the cells go in from 0, so a mirrored cache lands on the same ones as long as it restores the same count
    if (pSinfoIn && (pSinfoIn->empty() || pSinfoIn->mNStream() != 1 || pSinfoIn->lIdxs[0].size() != pCellCount)) {
      MGS_LOG_ERROR("%s: mirrored slot layout holds %d cells, this cache restores %d\n", __func__,
                    pSinfoIn->empty() ? 0 : (int)pSinfoIn->lIdxs[0].size(), pCellCount);
      return false;
    }

    for (uint32_t vI = 0; vI < pCellCount; ++vI) {
      mgs_pos vPos;
      uint32_t vNSeqId;

      pIo.mRead(&vPos, sizeof(vPos));
      pIo.mRead(&vNSeqId, sizeof(vNSeqId));

      vCells.mPosSet(vI, vPos);

      if (mHasCellExt()) {
        mgs_kv_cell_ext vExt;
        pIo.mRead(&vExt, sizeof(vExt));
        vCells.mExtSet(vI, vExt);
      }

      for (uint32_t vJ = 0; vJ < vNSeqId; ++vJ) {
        mgs_seq_id vSeqId;
        pIo.mRead(&vSeqId, sizeof(vSeqId));

        if (vSeqId < 0 || (uint32_t)vSeqId >= cNSeqMax) {
          MGS_LOG_ERROR("%s: invalid seq_id, %d is out of range [0, %u)\n", __func__, vSeqId, cNSeqMax);
          return false;
        }

        vCells.mSeqAdd(vI, vSeqId);
      }
    }

    // Create contiguous slot_info for whole cache restore
    pSinfo.vS0 = pStrm;
    pSinfo.vS1 = pStrm;
    pSinfo.mResize(1);
    pSinfo.lStrm[0] = pStrm;
    pSinfo.lIdxs[0].resize(pCellCount);
    for (uint32_t vI = 0; vI < pCellCount; ++vI) {
      pSinfo.lIdxs[0][vI] = vI;
    }

    vHead = 0;
  }

  return true;
}

bool mgs_kv_cache::mStateReadData(mgs_io_read_i &pIo, uint32_t pStrm, uint32_t pCellCount, const slot_info &pSinfo) {
  auto &vCells = vVCells[pStrm];

  uint32_t vVTrans;
  uint32_t vNLayer;

  pIo.mRead(&vVTrans, sizeof(vVTrans));
  pIo.mRead(&vNLayer, sizeof(vNLayer));

  if (vNLayer != lLayers.size()) {
    MGS_LOG_ERROR("%s: mismatched layer count (%u instead of %u)\n", __func__, vNLayer, (uint32_t)lLayers.size());
    return false;
  }

  if (pCellCount > vCells.size()) {
    MGS_LOG_ERROR("%s: not enough cells in kv cache to restore state (%u > %u)\n", __func__, pCellCount, vCells.size());
    return false;
  }

  if (this->vVTrans != (bool)vVTrans) {
    MGS_LOG_ERROR("%s: incompatible V transposition\n", __func__);
    return false;
  }

  // For each layer, read the keys for each cell, one row is one cell, read as one contiguous block
  for (const auto &vLayer : lLayers) {
    const uint32_t cIl = vLayer.vIl;

    const uint32_t cNEmbdKGqa = vHparams.mNEmbdKGqa(cIl);

    auto *vK = vLayer.lKStream[pStrm];

    // Read type of key
    int32_t vKTypeIRef;
    pIo.mRead(&vKTypeIRef, sizeof(vKTypeIRef));
    const int32_t cKTypeI = (int32_t)vK->type;
    if (cKTypeI != vKTypeIRef) {
      MGS_LOG_ERROR("%s: mismatched key type (%d != %d, layer %d)\n", __func__, cKTypeI, vKTypeIRef, cIl);
      return false;
    }

    // Read row size of key
    uint64_t vKSizeRowRef;
    pIo.mRead(&vKSizeRowRef, sizeof(vKSizeRowRef));
    const size_t cKSizeRow = ggml_row_size(vK->type, cNEmbdKGqa);
    if (cKSizeRow != vKSizeRowRef) {
      MGS_LOG_ERROR("%s: mismatched key row size (%zu != %zu, layer %d)\n", __func__, cKSizeRow, (size_t)vKSizeRowRef,
                    cIl);
      return false;
    }

    if (pCellCount) {
      if (pSinfo.mIsContiguous()) {
        // Fast path: contiguous cells, single memcpy
        pIo.mReadTensor(vK, pSinfo.mHead() * cKSizeRow, pCellCount * cKSizeRow);
      } else {
        // Slow path: scatter to non-contiguous positions
        for (uint32_t vI = 0; vI < pCellCount; ++vI) {
          const size_t cDstOffset = pSinfo.lIdxs[0][vI] * cKSizeRow;
          pIo.mReadTensor(vK, cDstOffset, cKSizeRow);
        }
      }
    }
  }

  if (!this->vVTrans) {
    for (const auto &vLayer : lLayers) {
      const uint32_t cIl = vLayer.vIl;

      const uint32_t cNEmbdVGqa = vHparams.mNEmbdVGqa(cIl);

      auto *vV = vLayer.lVStream[pStrm];
      if (!vV) {
        continue;
      }

      // Read type of value
      int32_t vVTypeIRef;
      pIo.mRead(&vVTypeIRef, sizeof(vVTypeIRef));
      const int32_t cVTypeI = (int32_t)vV->type;
      if (cVTypeI != vVTypeIRef) {
        MGS_LOG_ERROR("%s: mismatched value type (%d != %d, layer %d)\n", __func__, cVTypeI, vVTypeIRef, cIl);
        return false;
      }

      // Read row size of value
      uint64_t vVSizeRowRef;
      pIo.mRead(&vVSizeRowRef, sizeof(vVSizeRowRef));
      const size_t cVSizeRow = ggml_row_size(vV->type, cNEmbdVGqa);
      if (cVSizeRow != vVSizeRowRef) {
        MGS_LOG_ERROR("%s: mismatched value row size (%zu != %zu, layer %d)\n", __func__, cVSizeRow,
                      (size_t)vVSizeRowRef, cIl);
        return false;
      }

      if (pCellCount) {
        if (pSinfo.mIsContiguous()) {
          // Fast path: contiguous cells, single memcpy
          pIo.mReadTensor(vV, pSinfo.mHead() * cVSizeRow, pCellCount * cVSizeRow);
        } else {
          // Slow path: scatter to non-contiguous positions
          for (uint32_t vI = 0; vI < pCellCount; ++vI) {
            const size_t cDstOffset = pSinfo.lIdxs[0][vI] * cVSizeRow;
            pIo.mReadTensor(vV, cDstOffset, cVSizeRow);
          }
        }
      }
    }
  } else {
    // For each layer, read the values for each cell (transposed)
    for (const auto &vLayer : lLayers) {
      const uint32_t cIl = vLayer.vIl;

      const uint32_t cNEmbdVGqa = vHparams.mNEmbdVGqa(cIl);

      auto *vV = vLayer.lVStream[pStrm];
      if (!vV) {
        continue;
      }

      // Read type of value
      int32_t vVTypeIRef;
      pIo.mRead(&vVTypeIRef, sizeof(vVTypeIRef));
      const int32_t cVTypeI = (int32_t)vV->type;
      if (cVTypeI != vVTypeIRef) {
        MGS_LOG_ERROR("%s: mismatched value type (%d != %d, layer %d)\n", __func__, cVTypeI, vVTypeIRef, cIl);
        return false;
      }

      // Read element size of value
      uint32_t vVSizeElRef;
      pIo.mRead(&vVSizeElRef, sizeof(vVSizeElRef));
      const size_t cVSizeEl = ggml_type_size(vV->type);
      if (cVSizeEl != vVSizeElRef) {
        MGS_LOG_ERROR("%s: mismatched value element size (%zu != %zu, layer %d)\n", __func__, cVSizeEl,
                      (size_t)vVSizeElRef, cIl);
        return false;
      }

      // Read GQA embedding size
      uint32_t vNEmbdVGqaRef;
      pIo.mRead(&vNEmbdVGqaRef, sizeof(vNEmbdVGqaRef));
      if (cNEmbdVGqa != vNEmbdVGqaRef) {
        MGS_LOG_ERROR("%s: mismatched GQA embedding size (%u != %u, layer %d)\n", __func__, cNEmbdVGqa, vNEmbdVGqaRef,
                      cIl);
        return false;
      }

      if (pCellCount) {
        if (pSinfo.mIsContiguous()) {
          // Fast path: contiguous cells
          const uint32_t cH = pSinfo.mHead();
          for (uint32_t vJ = 0; vJ < cNEmbdVGqa; ++vJ) {
            const size_t cDstOffset = (cH + vJ * vCells.size()) * cVSizeEl;
            pIo.mReadTensor(vV, cDstOffset, pCellCount * cVSizeEl);
          }
        } else {
          // Slow path: scatter to non-contiguous positions
          for (uint32_t vJ = 0; vJ < cNEmbdVGqa; ++vJ) {
            for (uint32_t vI = 0; vI < pCellCount; ++vI) {
              const size_t cDstOffset = (pSinfo.lIdxs[0][vI] + vJ * vCells.size()) * cVSizeEl;
              pIo.mReadTensor(vV, cDstOffset, cVSizeEl);
            }
          }
        }
      }
    }
  }

  return true;
}

//
// mgs_kv_cache_context
//

mgs_kv_cache_context::mgs_kv_cache_context(mgs_memory_status pStatus) : vStatus(pStatus) {}

mgs_kv_cache_context::mgs_kv_cache_context(mgs_kv_cache *pKv) : vStatus(cMgsMemoryStatusSuccess), vKv(pKv) {
  vNKv = pKv->mGetSize();

  const uint32_t cNStream = pKv->mGetNStream();

  // create a dummy slot info - the actual data is irrelevant. we just need to build the graph
  vSinfos.resize(1);
  vSinfos[0].vS0 = 0;
  vSinfos[0].vS1 = cNStream - 1;
  vSinfos[0].lIdxs.resize(cNStream);
  for (uint32_t vS = 0; vS < cNStream; ++vS) {
    vSinfos[0].lStrm.push_back(vS);
    vSinfos[0].lIdxs[vS].resize(1, 0);
  }
}

mgs_kv_cache_context::mgs_kv_cache_context(mgs_kv_cache *pKv, mgs_context *pLctx, bool pDoShift,
                                           stream_copy_info pScInfo)
    : vStatus(cMgsMemoryStatusSuccess), vKv(pKv), vLctx(pLctx), vDoShift(pDoShift), vScInfo(std::move(pScInfo)) {
  if (!pDoShift && this->vScInfo.empty()) {
    vStatus = cMgsMemoryStatusNoUpdate;
  }
}

mgs_kv_cache_context::mgs_kv_cache_context(mgs_kv_cache *pKv, mgs_kv_cache::slot_info_vec_t pSinfos,
                                           std::vector<mgs_ubatch> pUbatches)
    : vStatus(cMgsMemoryStatusSuccess), vKv(pKv), vSinfos(std::move(pSinfos)), lUbatches(std::move(pUbatches)) {}

mgs_kv_cache_context::~mgs_kv_cache_context() = default;

bool mgs_kv_cache_context::mNext() {
  assert(vStatus == cMgsMemoryStatusSuccess);

  if (++vICur >= lUbatches.size()) {
    return false;
  }

  return true;
}

bool mgs_kv_cache_context::mApply() {
  assert(!mgs_memory_status_is_fail(vStatus));

  // no ubatches -> this is a KV cache update
  if (lUbatches.empty()) {
    vKv->mUpdate(vLctx, vDoShift, vScInfo);

    return true;
  }

  vKv->mApplyUbatch(vSinfos[vICur], lUbatches[vICur]);
  vNKv = vKv->mGetNKv(vSinfos[vICur]);

  return true;
}

mgs_memory_status mgs_kv_cache_context::mGetStatus() const { return vStatus; }

const mgs_ubatch &mgs_kv_cache_context::mGetUbatch() const {
  assert(vStatus == cMgsMemoryStatusSuccess);

  return lUbatches[vICur];
}

uint32_t mgs_kv_cache_context::mGetNKv() const { return vNKv; }

ggml_type mgs_kv_cache_context::mTypeK() const { return vKv->mTypeK(); }

ggml_type mgs_kv_cache_context::mTypeV() const { return vKv->mTypeV(); }

ggml_tensor *mgs_kv_cache_context::mGetK(ggml_context *pCtx, int32_t pIl) const {
  return vKv->mGetK(pCtx, pIl, vNKv, vSinfos[vICur]);
}

ggml_tensor *mgs_kv_cache_context::mGetV(ggml_context *pCtx, int32_t pIl) const {
  return vKv->mGetV(pCtx, pIl, vNKv, vSinfos[vICur]);
}

ggml_tensor *mgs_kv_cache_context::mCpyK(ggml_context *pCtx, ggml_tensor *pKCur, ggml_tensor *pKIdxs,
                                         int32_t pIl) const {
  return vKv->mCpyK(pCtx, pKCur, pKIdxs, pIl, vSinfos[vICur]);
}

ggml_tensor *mgs_kv_cache_context::mCpyV(ggml_context *pCtx, ggml_tensor *pVCur, ggml_tensor *pVIdxs,
                                         int32_t pIl) const {
  return vKv->mCpyV(pCtx, pVCur, pVIdxs, pIl, vSinfos[vICur]);
}

ggml_tensor *mgs_kv_cache_context::mBuildInputKIdxs(ggml_context *pCtx, const mgs_ubatch &pUbatch) const {
  return vKv->mBuildInputKIdxs(pCtx, pUbatch);
}

ggml_tensor *mgs_kv_cache_context::mBuildInputVIdxs(ggml_context *pCtx, const mgs_ubatch &pUbatch) const {
  return vKv->mBuildInputVIdxs(pCtx, pUbatch);
}

ggml_tensor *mgs_kv_cache_context::mBuildInputKRot(ggml_context *pCtx) const { return vKv->mBuildInputKRot(pCtx); }

ggml_tensor *mgs_kv_cache_context::mBuildInputVRot(ggml_context *pCtx) const { return vKv->mBuildInputVRot(pCtx); }

void mgs_kv_cache_context::mSetInputKShift(ggml_tensor *pDst) const { vKv->mSetInputKShift(pDst); }

void mgs_kv_cache_context::mSetInputKIdxs(ggml_tensor *pDst, const mgs_ubatch *pUbatch) const {
  vKv->mSetInputKIdxs(pDst, pUbatch, vSinfos[vICur]);
}

void mgs_kv_cache_context::mSetInputVIdxs(ggml_tensor *pDst, const mgs_ubatch *pUbatch) const {
  vKv->mSetInputVIdxs(pDst, pUbatch, vSinfos[vICur]);
}

void mgs_kv_cache_context::mSetInputKqMask(ggml_tensor *pDst, const mgs_ubatch *pUbatch, bool pCausalAttn) const {
  vKv->mSetInputKqMask(pDst, pUbatch, pCausalAttn);
}

void mgs_kv_cache_context::mSetInputPosBucket(ggml_tensor *pDst, const mgs_ubatch *pUbatch) const {
  vKv->mSetInputPosBucket(pDst, pUbatch);
}

void mgs_kv_cache_context::mSetInputKRot(ggml_tensor *pDst) const { vKv->mSetInputKRot(pDst); }

void mgs_kv_cache_context::mSetInputVRot(ggml_tensor *pDst) const { vKv->mSetInputVRot(pDst); }

void mgs_kv_cache_context::mGetPrevTokens(const mgs_ubatch &pUbatch, uint32_t pN,
                                          std::vector<mgs_token> &pRes) const {
  vKv->mGetPrevTokens(pUbatch, pN, pRes);
}
