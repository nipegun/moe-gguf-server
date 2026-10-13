#include "mgs-graph.h"

#include "mgs-impl.h"
#include "mgs-model.h"
#include "mgs-expert-cache.h"
#include "mgs-batch.h"
#include "mgs-cparams.h"

#include "mgs-kv-cache.h"
#include "mgs-kv-cache-iswa.h"
#include "mgs-kv-cache-dsa.h"
#include "mgs-kv-cache-msa.h"
#include "mgs-memory-hybrid.h"
#include "mgs-memory-hybrid-iswa.h"
#include "mgs-memory-recurrent.h"

#include <cassert>
#include <cmath>
#include <cstring>
#include <numeric>
#include <sstream>
#include <string>
#include <unordered_set>

// dedup helpers

static ggml_tensor *fBuildAttnInpKqMask(ggml_context *pCtx, const mgs_kv_cache_context *pMctx,
                                        const mgs_ubatch &pUbatch, const mgs_cparams &pCparams) {
  const auto cNKv = pMctx->mGetNKv();
  const auto cNTokens = pUbatch.vNTokens;
  const auto cNStream = pCparams.vKvUnified ? 1 : pUbatch.vNSeqsUnq;

  // flash attention requires an f16 mask
  const auto cType = pCparams.vFlashAttn ? GGML_TYPE_F16 : GGML_TYPE_F32;

  ggml_tensor *vRes = ggml_new_tensor_4d(pCtx, cType, cNKv, cNTokens / cNStream, 1, cNStream);
  ggml_set_input(vRes);
  ggml_set_name(vRes, "attn_inp_kq_mask");

  return vRes;
}

static bool fCanReuseKqMask(ggml_tensor *pKqMask, const mgs_kv_cache_context *pMctx, const mgs_ubatch &pUbatch,
                            const mgs_cparams &pCparams) {
  const auto cNKv = pMctx->mGetNKv();
  const auto cNTokens = pUbatch.vNTokens;
  const auto cNStream = pCparams.vKvUnified ? 1 : pUbatch.vNSeqsUnq;

  bool vRes = true;

  vRes &= (pKqMask->ne[0] == cNKv);
  vRes &= (pKqMask->ne[1] == cNTokens / cNStream);
  vRes &= (pKqMask->ne[2] == 1);
  vRes &= (pKqMask->ne[3] == cNStream);

  return vRes;
}

// impl

void llm_graph_input_embd::mSetInput(const mgs_ubatch *pUbatch) {
  if (pUbatch->vToken) {
    const int64_t cNTokens = pUbatch->vNTokens;

    ggml_backend_tensor_set(vTokens, pUbatch->vToken, 0, cNTokens * ggml_element_size(vTokens));
  }

  if (pUbatch->vEmbd) {
    GGML_ASSERT(cNEmbd == vEmbd->ne[0]);

    const int64_t cNTokens = pUbatch->vNTokens;

    ggml_backend_tensor_set(vEmbd, pUbatch->vEmbd, 0, cNTokens * cNEmbd * ggml_element_size(vEmbd));
  }
}

bool llm_graph_input_embd::mCanReuse(const llm_graph_params &pParams) {
  bool vRes = true;

  vRes &= (!pParams.vUbatch.vToken) || (vTokens && vTokens->ne[0] == pParams.vUbatch.vNTokens);
  vRes &= (!pParams.vUbatch.vEmbd) || (vEmbd && vEmbd->ne[1] == pParams.vUbatch.vNTokens);

  return vRes;
}

void llm_graph_input_embd_h::mSetInput(const mgs_ubatch *pUbatch) {
  const int64_t cNTokens = pUbatch->vNTokens;

  if (pUbatch->vToken) {
    ggml_backend_tensor_set(vTokens, pUbatch->vToken, 0, cNTokens * ggml_element_size(vTokens));
  } else {
    // note: mtmd embedding input goes through here
    GGML_ASSERT(pUbatch->vEmbd);
    GGML_ASSERT(cNEmbd == vEmbd->ne[0]);

    ggml_backend_tensor_set(vEmbd, pUbatch->vEmbd, 0, cNTokens * cNEmbd * ggml_element_size(vH));
  }

  // TODO: extend mgs_ubatch to differentiate between token embeddings and hidden states
  //       for now, we assume that the hidden state is always provided as an embedding
  if (pUbatch->vEmbd) {
    GGML_ASSERT(cNEmbd == vH->ne[0]);

    ggml_backend_tensor_set(vH, pUbatch->vEmbd, 0, cNTokens * cNEmbd * ggml_element_size(vH));
  }
}

bool llm_graph_input_embd_h::mCanReuse(const llm_graph_params &pParams) {
  bool vRes = true;

  vRes &= (!pParams.vUbatch.vToken) || (vTokens && vTokens->ne[0] == pParams.vUbatch.vNTokens);
  vRes &= (!pParams.vUbatch.vEmbd) || (vEmbd && vEmbd->ne[1] == pParams.vUbatch.vNTokens);
  vRes &= (!pParams.vUbatch.vEmbd) || (vH && vH->ne[1] == pParams.vUbatch.vNTokens);

  return vRes;
}

void llm_graph_input_pos::mSetInput(const mgs_ubatch *pUbatch) {
  if (pUbatch->vPos && vPos) {
    const int64_t cNTokens = pUbatch->vNTokens;

    if (pUbatch->vToken && cNPosPerEmbd == 4) {
      // in case we're using M-RoPE with text tokens, convert the 1D positions to 4D
      // the 3 first dims are the same, and 4th dim is all 0
      std::vector<mgs_pos> lPosData(cNTokens * cNPosPerEmbd);
      // copy the first dimension
      for (int vI = 0; vI < cNTokens; ++vI) {
        lPosData[vI] = pUbatch->vPos[vI];
        lPosData[cNTokens + vI] = pUbatch->vPos[vI];
        lPosData[2 * cNTokens + vI] = pUbatch->vPos[vI];
        lPosData[3 * cNTokens + vI] = 0; // 4th dim is 0
      }
      ggml_backend_tensor_set(vPos, lPosData.data(), 0, lPosData.size() * ggml_element_size(vPos));
    } else {
      ggml_backend_tensor_set(vPos, pUbatch->vPos, 0, cNTokens * cNPosPerEmbd * ggml_element_size(vPos));
    }
  }
}

bool llm_graph_input_pos::mCanReuse(const llm_graph_params &pParams) {
  bool vRes = true;

  vRes &= vPos->ne[0] == pParams.vUbatch.vNTokens * cNPosPerEmbd;

  return vRes;
}

void llm_graph_input_attn_temp::mSetInput(const mgs_ubatch *pUbatch) {
  if (pUbatch->vPos && vAttnScale) {
    const int64_t cNTokens = pUbatch->vNTokens;

    GGML_ASSERT(cFAttnTempScale != 0.0f);
    GGML_ASSERT(cNAttnTempFloorScale != 0);

    std::vector<float> lAttnScaleData(cNTokens, 0.0f);
    for (int vI = 0; vI < cNTokens; ++vI) {
      const float cPos = pUbatch->vPos[vI];
      lAttnScaleData[vI] =
          std::log(std::floor((cPos + cFAttnTempOffset) / cNAttnTempFloorScale) + 1.0) * cFAttnTempScale + 1.0;
    }

    ggml_backend_tensor_set(vAttnScale, lAttnScaleData.data(), 0, cNTokens * ggml_element_size(vAttnScale));
  }
}

void llm_graph_input_pos_bucket::mSetInput(const mgs_ubatch *pUbatch) {
  if (vPosBucket) {
    const int64_t cNTokens = pUbatch->vNTokens;

    GGML_ASSERT(ggml_backend_buffer_is_host(vPosBucket->buffer));
    GGML_ASSERT(!pUbatch->mEqualSeqs()); // TODO: use ubatch->n_seqs instead of failing

    int32_t *vData = (int32_t *)vPosBucket->data;

    for (int vJ = 0; vJ < cNTokens; ++vJ) {
      for (int vI = 0; vI < cNTokens; ++vI) {
        vData[vJ * cNTokens + vI] =
            mgs_relative_position_bucket(pUbatch->vPos[vI], pUbatch->vPos[vJ], cHparams.vNRelAttnBkts, true);
      }
    }
  }
}

void llm_graph_input_pos_bucket_kv::mSetInput(const mgs_ubatch *pUbatch) {
  if (vPosBucket) {
    vMctx->mSetInputPosBucket(vPosBucket, pUbatch);
  }
}

void llm_graph_input_out_ids::mSetInput(const mgs_ubatch *pUbatch) {
  GGML_ASSERT(vOutIds);

  const int64_t cNTokens = pUbatch->vNTokens;

  GGML_ASSERT(ggml_backend_buffer_is_host(vOutIds->buffer));
  int32_t *vData = (int32_t *)vOutIds->data;

  if (cNOutputs == cNTokens) {
    for (int vI = 0; vI < cNTokens; ++vI) {
      vData[vI] = vI;
    }

    return;
  }

  GGML_ASSERT(pUbatch->vOutput);

  int vNOutputs = 0;

  for (int vI = 0; vI < cNTokens; ++vI) {
    if (pUbatch->vOutput[vI]) {
      vData[vNOutputs++] = vI;
    }
  }
}

bool llm_graph_input_out_ids::mCanReuse(const llm_graph_params &pParams) {
  bool vRes = true;

  vRes &= cNOutputs == pParams.vNOutputs;

  return vRes;
}

void llm_graph_input_mean::mSetInput(const mgs_ubatch *pUbatch) {
  if (cCparams.vEmbeddings &&
      (cCparams.vPoolingType == MGS_POOLING_TYPE_MEAN || cCparams.vPoolingType == MGS_POOLING_TYPE_RANK)) {

    const int64_t cNTokens = pUbatch->vNTokens;
    const int64_t cNSeqTokens = pUbatch->vNSeqTokens;
    const int64_t cNSeqsUnq = pUbatch->vNSeqsUnq;

    GGML_ASSERT(vMean);
    GGML_ASSERT(ggml_backend_buffer_is_host(vMean->buffer));

    float *vData = (float *)vMean->data;
    memset(vMean->data, 0, cNTokens * cNSeqsUnq * ggml_element_size(vMean));

    std::vector<uint64_t> lSums(cNSeqsUnq, 0);
    for (int vI = 0; vI < cNTokens; vI += cNSeqTokens) {
      for (int vS = 0; vS < pUbatch->vNSeqId[vI]; ++vS) {
        const mgs_seq_id cSeqId = pUbatch->vSeqId[vI][vS];
        const int32_t cSeqIdx = pUbatch->vSeqIdx[cSeqId];

        lSums[cSeqIdx] += pUbatch->vNSeqTokens;
      }
    }

    std::vector<float> lDiv(cNSeqsUnq, 0.0f);
    for (int vS = 0; vS < cNSeqsUnq; ++vS) {
      const uint64_t cSum = lSums[vS];
      if (cSum > 0) {
        lDiv[vS] = 1.0f / float(cSum);
      }
    }

    for (int vI = 0; vI < cNTokens; vI += cNSeqTokens) {
      for (int vS = 0; vS < pUbatch->vNSeqId[vI]; ++vS) {
        const mgs_seq_id cSeqId = pUbatch->vSeqId[vI][vS];
        const int32_t cSeqIdx = pUbatch->vSeqIdx[cSeqId];

        for (int vJ = 0; vJ < cNSeqTokens; ++vJ) {
          vData[cSeqIdx * cNTokens + vI + vJ] = lDiv[cSeqIdx];
        }
      }
    }
  }
}

void llm_graph_input_cls::mSetInput(const mgs_ubatch *pUbatch) {
  const int64_t cNTokens = pUbatch->vNTokens;
  const int64_t cNSeqsUnq = pUbatch->vNSeqsUnq;

  if (cCparams.vEmbeddings &&
      (cCparams.vPoolingType == MGS_POOLING_TYPE_CLS || cCparams.vPoolingType == MGS_POOLING_TYPE_RANK ||
       cCparams.vPoolingType == MGS_POOLING_TYPE_LAST)) {
    GGML_ASSERT(vCls);
    GGML_ASSERT(ggml_backend_buffer_is_host(vCls->buffer));

    uint32_t *vData = (uint32_t *)vCls->data;
    memset(vCls->data, 0, cNSeqsUnq * ggml_element_size(vCls));

    std::vector<int> lTargetPos(cNSeqsUnq, -1);
    std::vector<int> lTargetRow(cNSeqsUnq, -1);

    const bool cLast =
        (cCparams.vPoolingType == MGS_POOLING_TYPE_LAST ||
         (cCparams.vPoolingType == MGS_POOLING_TYPE_RANK &&
          (cArch == cLlmArchQwen3 || cArch == cLlmArchQwen3vl)) // qwen3 reranking & embedding models use last token
        );

    for (int vI = 0; vI < cNTokens; ++vI) {
      const mgs_pos cPos = pUbatch->vPos[vI];

      for (int vS = 0; vS < pUbatch->vNSeqId[vI]; ++vS) {
        const mgs_seq_id cSeqId = pUbatch->vSeqId[vI][vS];
        const int32_t cSeqIdx = pUbatch->vSeqIdx[cSeqId];

        if ((lTargetPos[cSeqIdx] == -1) || (cLast && cPos >= lTargetPos[cSeqIdx]) ||
            (!cLast && cPos < lTargetPos[cSeqIdx])) {
          lTargetPos[cSeqIdx] = cPos;
          lTargetRow[cSeqIdx] = vI;
        }
      }
    }

    for (int vS = 0; vS < cNSeqsUnq; ++vS) {
      if (lTargetRow[vS] >= 0) {
        vData[vS] = lTargetRow[vS];
      }
    }
  }
}

void llm_graph_input_rs::mSetInput(const mgs_ubatch *pUbatch) {
  GGML_UNUSED(pUbatch);

  const int64_t cNRs = vMctx->mGetNRs();

  if (vSCopy) {
    GGML_ASSERT(ggml_backend_buffer_is_host(vSCopy->buffer));
    int32_t *vData = (int32_t *)vSCopy->data;

    // assuming copy destinations ALWAYS happen ONLY on the cells between head and head+n
    for (uint32_t vI = 0; vI < cNRs; ++vI) {
      vData[vI] = vMctx->mSCopy(vI);
    }
  }
}

bool llm_graph_input_rs::mCanReuse(const llm_graph_params &pParams) {
  const auto *vMctx = static_cast<const mgs_memory_recurrent_context *>(pParams.vMctx);

  this->vMctx = vMctx;

  bool vRes = true;

  vRes &= vSCopy->ne[0] == vMctx->mGetNRs();

  vRes &= vSCopyMain->ne[0] == pParams.vUbatch.vNSeqs;
  vRes &= vSCopyExtra->ne[0] == vMctx->mGetNRs() - pParams.vUbatch.vNSeqs;

  vRes &= vHead == vMctx->mGetHead();
  vRes &= vRsZ == vMctx->mGetRsZ();

  return vRes;
}

void llm_graph_input_cross_embd::mSetInput(const mgs_ubatch *pUbatch) {
  GGML_UNUSED(pUbatch);

  if (vCrossEmbd && !vCross->lVEmbd.empty()) {
    assert(vCrossEmbd->type == GGML_TYPE_F32);

    ggml_backend_tensor_set(vCrossEmbd, vCross->lVEmbd.data(), 0, ggml_nbytes(vCrossEmbd));
  }
}

template <typename T>
static void fPrintMask(const T *pData, int64_t pNTokens, int64_t pNKv, int64_t pNSwa, mgs_swa_type pSwaType) {
  MGS_LOG_DEBUG("%s: === Attention mask ===\n", __func__);
  const char *vSwaTypeStr = "unknown";

  switch (pSwaType) {
  case cMgsSwaTypeNone:
    vSwaTypeStr = "MGS_SWA_TYPE_NONE";
    break;
  case cMgsSwaTypeStandard:
    vSwaTypeStr = "MGS_SWA_TYPE_STANDARD";
    break;
  case cMgsSwaTypeChunked:
    vSwaTypeStr = "MGS_SWA_TYPE_CHUNKED";
    break;
  case cMgsSwaTypeSymmetric:
    vSwaTypeStr = "MGS_SWA_TYPE_SYMMETRIC";
    break;
  };

  MGS_LOG_DEBUG("%s: n_swa : %d, n_kv: %d, swa_type: %s\n", __func__, (int)pNSwa, (int)pNKv, vSwaTypeStr);
  MGS_LOG_DEBUG("%s: '0' = can attend, '∞' = masked\n", __func__);
  MGS_LOG_DEBUG("%s: Rows = query tokens, Columns = key/value tokens\n\n", __func__);

  MGS_LOG_DEBUG("    ");
  for (int vJ = 0; vJ < std::min((int64_t)20, pNKv); ++vJ) {
    MGS_LOG_DEBUG("%2d", vJ);
  }
  MGS_LOG_DEBUG("\n");

  for (int vI = 0; vI < std::min((int64_t)20, pNTokens); ++vI) {
    MGS_LOG_DEBUG(" %2d ", vI);
    for (int vJ = 0; vJ < std::min((int64_t)20, pNKv); ++vJ) {
      float vVal = mgs_cast<float>(pData[vI * pNKv + vJ]);
      if (vVal == -INFINITY) {
        MGS_LOG_DEBUG(" ∞");
      } else {
        MGS_LOG_DEBUG(" 0");
      }
    }
    MGS_LOG_DEBUG("\n");
  }
}

void llm_graph_input_attn_no_cache::mSetInput(const mgs_ubatch *pUbatch) {
  const int64_t cNKv = pUbatch->vNTokens;
  const int64_t cNTokens = pUbatch->vNTokens;

  const auto cFillMask = [&](auto *pData, int64_t pNe, int pNSwa, mgs_swa_type pSwaType) {
    using T = std::remove_reference_t<decltype(*pData)>;
    std::fill(pData, pData + pNe, mgs_cast<T>(-INFINITY));

    for (int vI1 = 0; vI1 < cNTokens; ++vI1) {
      const mgs_seq_id cS1 = pUbatch->vSeqId[vI1][0];
      const mgs_pos cP1 = pUbatch->vPos[vI1];

      const uint64_t cIdst = vI1 * cNKv;

      for (int vI0 = 0; vI0 < cNTokens; ++vI0) {
        const mgs_seq_id cS0 = pUbatch->vSeqId[vI0][0];
        const mgs_pos cP0 = pUbatch->vPos[vI0];

        // mask different sequences
        if (cS0 != cS1) {
          continue;
        }

        // mask future tokens
        if (cCparams.vCausalAttn && cP0 > cP1) {
          continue;
        }

        // apply SWA if any
        if (mgs_hparams::mIsMaskedSwa(pNSwa, pSwaType, cP0, cP1)) {
          continue;
        }

        pData[cIdst + vI0] = mgs_cast<T>(cHparams.vUseAlibi ? -std::abs(cP0 - cP1) : 0.0f);
      }
    }

    if (vDebug) {
      fPrintMask(pData, cNTokens, cNKv, pNSwa, pSwaType);
    }
  };

  GGML_ASSERT(vSelfKqMask);
  GGML_ASSERT(ggml_backend_buffer_is_host(vSelfKqMask->buffer));
  if (vSelfKqMask->type == GGML_TYPE_F16) {
    cFillMask((ggml_fp16_t *)vSelfKqMask->data, ggml_nelements(vSelfKqMask), 0, cMgsSwaTypeNone);
  } else {
    cFillMask((float *)vSelfKqMask->data, ggml_nelements(vSelfKqMask), 0, cMgsSwaTypeNone);
  }

  if (cHparams.vSwaType != cMgsSwaTypeNone) {
    GGML_ASSERT(vSelfKqMaskSwa);
    GGML_ASSERT(ggml_backend_buffer_is_host(vSelfKqMaskSwa->buffer));
    if (vSelfKqMaskSwa->type == GGML_TYPE_F16) {
      cFillMask((ggml_fp16_t *)vSelfKqMaskSwa->data, ggml_nelements(vSelfKqMaskSwa), cHparams.vNSwa, cHparams.vSwaType);
    } else {
      cFillMask((float *)vSelfKqMaskSwa->data, ggml_nelements(vSelfKqMaskSwa), cHparams.vNSwa, cHparams.vSwaType);
    }
  }
}

void llm_graph_input_attn_kv::mSetInput(const mgs_ubatch *pUbatch) {
  vMctx->mSetInputKIdxs(vSelfKIdxs, pUbatch);
  vMctx->mSetInputVIdxs(vSelfVIdxs, pUbatch);

  // the mask is left unallocated when the graph only stores K/V without attending
  // (e.g. DFlash's KV-injection pass)
  if (vSelfKqMask && vSelfKqMask->buffer) {
    vMctx->mSetInputKqMask(vSelfKqMask, pUbatch, cCparams.vCausalAttn);
  }

  if (vSelfKRot && vSelfKRot->buffer) {
    vMctx->mSetInputKRot(vSelfKRot);
  }

  if (vSelfVRot && vSelfVRot->buffer) {
    vMctx->mSetInputVRot(vSelfVRot);
  }
}

bool llm_graph_input_attn_kv::mCanReuse(const llm_graph_params &pParams) {
  const auto *vMctx = static_cast<const mgs_kv_cache_context *>(pParams.vMctx);

  this->vMctx = vMctx;

  bool vRes = true;

  vRes &= vSelfKIdxs->ne[0] == pParams.vUbatch.vNTokens;
  // vRes &= vSelfVIdxs->ne[0] == pParams.vUbatch.vNTokens; // TODO: need to move this to the unified cache and
  // check there

  vRes &= fCanReuseKqMask(vSelfKqMask, vMctx, pParams.vUbatch, pParams.vCparams);

  return vRes;
}

void llm_graph_input_attn_k::mSetInput(const mgs_ubatch *pUbatch) {
  vMctx->mSetInputKIdxs(vSelfKIdxs, pUbatch);

  vMctx->mSetInputKqMask(vSelfKqMask, pUbatch, cCparams.vCausalAttn);
}

bool llm_graph_input_attn_k::mCanReuse(const llm_graph_params &pParams) {
  const auto *vMctx = static_cast<const mgs_kv_cache_context *>(pParams.vMctx);

  this->vMctx = vMctx;

  bool vRes = true;

  vRes &= vSelfKIdxs->ne[0] == pParams.vUbatch.vNTokens;

  vRes &= fCanReuseKqMask(vSelfKqMask, vMctx, pParams.vUbatch, pParams.vCparams);

  return vRes;
}

llm_graph_input_attn_kv_msa::llm_graph_input_attn_kv_msa(const mgs_hparams &pHparams, const mgs_cparams &pCparams,
                                                         const mgs_kv_cache_msa_context *pMctx)
    : llm_graph_input_attn_kv(pHparams, pCparams, pMctx->mGetBase()), vMctxMsa(pMctx) {}

void llm_graph_input_attn_kv_msa::mSetInput(const mgs_ubatch *pUbatch) {
  llm_graph_input_attn_kv::mSetInput(pUbatch);

  if (vSelfKIdxsIdx) {
    vMctxMsa->mGetIdx()->mSetInputKIdxs(vSelfKIdxsIdx, pUbatch);
  }
}

bool llm_graph_input_attn_kv_msa::mCanReuse(const llm_graph_params &pParams) {
  vMctxMsa = static_cast<const mgs_kv_cache_msa_context *>(pParams.vMctx);

  // the parent class operates on the base cache context
  this->vMctx = vMctxMsa->mGetBase();

  bool vRes = true;

  vRes &= vSelfKIdxs->ne[0] == pParams.vUbatch.vNTokens;
  if (vSelfKIdxsIdx) {
    vRes &= vSelfKIdxsIdx->ne[0] == pParams.vUbatch.vNTokens;
  }

  vRes &= fCanReuseKqMask(vSelfKqMask, this->vMctx, pParams.vUbatch, pParams.vCparams);

  return vRes;
}

void llm_graph_input_attn_k_dsa::mSetInput(const mgs_ubatch *pUbatch) {
  vMctx->mGetMla()->mSetInputKIdxs(vSelfKIdxsMla, pUbatch);

  vMctx->mGetMla()->mSetInputKqMask(vSelfKqMaskMla, pUbatch, cCparams.vCausalAttn);

  vMctx->mGetLid()->mSetInputKIdxs(vSelfKIdxsLid, pUbatch);

  vMctx->mGetLid()->mSetInputKqMask(vSelfKqMaskLid, pUbatch, cCparams.vCausalAttn);

  vMctx->mGetLid()->mSetInputKRot(vSelfKRotLid);
}

bool llm_graph_input_attn_k_dsa::mCanReuse(const llm_graph_params &pParams) {
  const auto *vMctx = static_cast<const mgs_kv_cache_dsa_context *>(pParams.vMctx);

  this->vMctx = vMctx;

  bool vRes = true;

  vRes &= vSelfKIdxsMla->ne[0] == pParams.vUbatch.vNTokens;
  vRes &= vSelfKIdxsLid->ne[0] == pParams.vUbatch.vNTokens;

  vRes &= fCanReuseKqMask(vSelfKqMaskMla, vMctx->mGetMla(), pParams.vUbatch, pParams.vCparams);
  vRes &= fCanReuseKqMask(vSelfKqMaskLid, vMctx->mGetLid(), pParams.vUbatch, pParams.vCparams);

  return vRes;
}

void llm_graph_input_attn_kv_iswa::mSetInput(const mgs_ubatch *pUbatch) {
  // base tensors may not be allocated if there are no non-SWA attention layers
  if (vSelfKIdxs && vSelfKIdxs->buffer) {
    vMctx->mGetBase()->mSetInputKIdxs(vSelfKIdxs, pUbatch);
    if (vSelfVIdxs) {
      vMctx->mGetBase()->mSetInputVIdxs(vSelfVIdxs, pUbatch);
    }
  }

  // the kq mask guards on its own buffer: shared cells leave idxs unbacked while the mask stays live
  if (vSelfKqMask && vSelfKqMask->buffer) {
    vMctx->mGetBase()->mSetInputKqMask(vSelfKqMask, pUbatch, cCparams.vCausalAttn);
  }

  // swa tensors may not be allocated if there are no SWA attention layers
  if (vSelfKIdxsSwa && vSelfKIdxsSwa->buffer) {
    vMctx->mGetSwa()->mSetInputKIdxs(vSelfKIdxsSwa, pUbatch);
    if (vSelfVIdxsSwa) {
      vMctx->mGetSwa()->mSetInputVIdxs(vSelfVIdxsSwa, pUbatch);
    }
  }

  if (vSelfKqMaskSwa && vSelfKqMaskSwa->buffer) {
    vMctx->mGetSwa()->mSetInputKqMask(vSelfKqMaskSwa, pUbatch, cCparams.vCausalAttn);
  }

  if (vSelfKRot && vSelfKRot->buffer) {
    vMctx->mGetBase()->mSetInputKRot(vSelfKRot);
  }

  if (vSelfVRot && vSelfVRot->buffer) {
    vMctx->mGetBase()->mSetInputVRot(vSelfVRot);
  }

  if (vSelfKRotSwa && vSelfKRotSwa->buffer) {
    vMctx->mGetSwa()->mSetInputKRot(vSelfKRotSwa);
  }

  if (vSelfVRotSwa && vSelfVRotSwa->buffer) {
    vMctx->mGetSwa()->mSetInputVRot(vSelfVRotSwa);
  }
}

bool llm_graph_input_attn_kv_iswa::mCanReuse(const llm_graph_params &pParams) {
  const auto *vMctx = static_cast<const mgs_kv_cache_iswa_context *>(pParams.vMctx);

  this->vMctx = vMctx;

  bool vRes = true;

  // base tensors may not be allocated if there are no non-SWA attention layers
  if (vSelfKIdxs && vSelfKIdxs->buffer) {
    vRes &= vSelfKIdxs->ne[0] == pParams.vUbatch.vNTokens;
    // vRes &= vSelfVIdxs->ne[0] == pParams.vUbatch.vNTokens; // TODO: need to move this to the unified cache and
    // check there
  }

  if (vSelfKqMask && vSelfKqMask->buffer) {
    vRes &= fCanReuseKqMask(vSelfKqMask, vMctx->mGetBase(), pParams.vUbatch, pParams.vCparams);
  }

  // swa tensors may not be allocated if there are no SWA attention layers
  if (vSelfKIdxsSwa && vSelfKIdxsSwa->buffer) {
    vRes &= vSelfKIdxsSwa->ne[0] == pParams.vUbatch.vNTokens;
    // vRes &= vSelfVIdxsSwa->ne[0] == pParams.vUbatch.vNTokens; // TODO: need to move this to the unified cache
    // and check there
  }

  if (vSelfKqMaskSwa && vSelfKqMaskSwa->buffer) {
    vRes &= fCanReuseKqMask(vSelfKqMaskSwa, vMctx->mGetSwa(), pParams.vUbatch, pParams.vCparams);
  }

  return vRes;
}

void llm_graph_input_attn_k_iswa::mSetInput(const mgs_ubatch *pUbatch) {
  // base tensors may not be allocated if there are no non-SWA attention layers
  if (vSelfKIdxs && vSelfKIdxs->buffer) {
    vMctx->mGetBase()->mSetInputKIdxs(vSelfKIdxs, pUbatch);
  }

  // the kq mask guards on its own buffer: shared cells leave idxs unbacked while the mask stays live
  if (vSelfKqMask && vSelfKqMask->buffer) {
    vMctx->mGetBase()->mSetInputKqMask(vSelfKqMask, pUbatch, cCparams.vCausalAttn);
  }

  // swa tensors may not be allocated if there are no SWA attention layers
  if (vSelfKIdxsSwa && vSelfKIdxsSwa->buffer) {
    vMctx->mGetSwa()->mSetInputKIdxs(vSelfKIdxsSwa, pUbatch);
  }

  if (vSelfKqMaskSwa && vSelfKqMaskSwa->buffer) {
    vMctx->mGetSwa()->mSetInputKqMask(vSelfKqMaskSwa, pUbatch, cCparams.vCausalAttn);
  }

  if (vSelfKRot && vSelfKRot->buffer) {
    vMctx->mGetBase()->mSetInputKRot(vSelfKRot);
  }

  if (vSelfKRotSwa && vSelfKRotSwa->buffer) {
    vMctx->mGetSwa()->mSetInputKRot(vSelfKRotSwa);
  }
}

bool llm_graph_input_attn_k_iswa::mCanReuse(const llm_graph_params &pParams) {
  const auto *vMctx = static_cast<const mgs_kv_cache_iswa_context *>(pParams.vMctx);

  this->vMctx = vMctx;

  bool vRes = true;

  // base tensors may not be allocated if there are no non-SWA attention layers
  if (vSelfKIdxs && vSelfKIdxs->buffer) {
    vRes &= vSelfKIdxs->ne[0] == pParams.vUbatch.vNTokens;
  }

  if (vSelfKqMask && vSelfKqMask->buffer) {
    vRes &= fCanReuseKqMask(vSelfKqMask, vMctx->mGetBase(), pParams.vUbatch, pParams.vCparams);
  }

  // swa tensors may not be allocated if there are no SWA attention layers
  if (vSelfKIdxsSwa && vSelfKIdxsSwa->buffer) {
    vRes &= vSelfKIdxsSwa->ne[0] == pParams.vUbatch.vNTokens;
  }

  if (vSelfKqMaskSwa && vSelfKqMaskSwa->buffer) {
    vRes &= fCanReuseKqMask(vSelfKqMaskSwa, vMctx->mGetSwa(), pParams.vUbatch, pParams.vCparams);
  }

  return vRes;
}

void llm_graph_input_attn_cross::mSetInput(const mgs_ubatch *pUbatch) {
  GGML_ASSERT(vCrossKqMask);

  const int64_t cNEnc = vCrossKqMask->ne[0];
  const int64_t cNTokens = pUbatch->vNTokens;

  GGML_ASSERT(ggml_backend_buffer_is_host(vCrossKqMask->buffer));
  GGML_ASSERT(!pUbatch->mEqualSeqs()); // TODO: use ubatch->n_seqs instead of failing

  const auto cFillMask = [&](auto *pData) {
    using T = std::remove_reference_t<decltype(*pData)>;
    for (int vI = 0; vI < cNTokens; ++vI) {
      GGML_ASSERT(!vCross->sSeqIdsEnc.empty() && "mgs_encode must be called first");
      for (int vJ = 0; vJ < cNEnc; ++vJ) {
        float vF = -INFINITY;

        for (int vS = 0; vS < pUbatch->vNSeqId[vI]; ++vS) {
          const mgs_seq_id cSeqId = pUbatch->vSeqId[vI][vS];

          if (vCross->sSeqIdsEnc[vJ].find(cSeqId) != vCross->sSeqIdsEnc[vJ].end()) {
            vF = 0.0f;
          }
        }

        pData[vI * cNEnc + vJ] = mgs_cast<T>(vF);
      }
    }
  };

  if (vCrossKqMask->type == GGML_TYPE_F16) {
    cFillMask((ggml_fp16_t *)vCrossKqMask->data);
  } else {
    cFillMask((float *)vCrossKqMask->data);
  }
}

void llm_graph_input_mem_hybrid::mSetInput(const mgs_ubatch *pUbatch) {
  vMctx->mGetAttn()->mSetInputKIdxs(vInpAttn->vSelfKIdxs, pUbatch);
  vMctx->mGetAttn()->mSetInputVIdxs(vInpAttn->vSelfVIdxs, pUbatch);

  vMctx->mGetAttn()->mSetInputKqMask(vInpAttn->vSelfKqMask, pUbatch, cCparams.vCausalAttn);

  if (vInpAttn->vSelfKRot) {
    vMctx->mGetAttn()->mSetInputKRot(vInpAttn->vSelfKRot);
  }

  if (vInpAttn->vSelfVRot) {
    vMctx->mGetAttn()->mSetInputVRot(vInpAttn->vSelfVRot);
  }

  const int64_t cNRs = vMctx->mGetRecr()->mGetNRs();

  if (vInpRs->vSCopy) {
    GGML_ASSERT(ggml_backend_buffer_is_host(vInpRs->vSCopy->buffer));
    int32_t *vData = (int32_t *)vInpRs->vSCopy->data;

    // assuming copy destinations ALWAYS happen ONLY on the cells between head and head+n
    for (uint32_t vI = 0; vI < cNRs; ++vI) {
      vData[vI] = vMctx->mGetRecr()->mSCopy(vI);
    }
  }
}

bool llm_graph_input_mem_hybrid::mCanReuse(const llm_graph_params &pParams) {
  const auto *vMctx = static_cast<const mgs_memory_hybrid_context *>(pParams.vMctx);

  this->vMctx = vMctx;

  bool vRes = true;

  vRes &= vInpAttn->vSelfKIdxs->ne[0] == pParams.vUbatch.vNTokens;
  // vRes &= vInpAttn->vSelfVIdxs->ne[0] == pParams.vUbatch.vNTokens; // TODO: need to move this to the unified
  // cache and check there

  vRes &= fCanReuseKqMask(vInpAttn->vSelfKqMask, vMctx->mGetAttn(), pParams.vUbatch, pParams.vCparams);

  vRes &= vInpRs->vSCopy->ne[0] == vMctx->mGetRecr()->mGetNRs();

  vRes &= vInpRs->vSCopyMain->ne[0] == pParams.vUbatch.vNSeqs;
  vRes &= vInpRs->vSCopyExtra->ne[0] == vMctx->mGetRecr()->mGetNRs() - pParams.vUbatch.vNSeqs;

  vRes &= vInpRs->vHead == vMctx->mGetRecr()->mGetHead();
  vRes &= vInpRs->vRsZ == vMctx->mGetRecr()->mGetRsZ();

  return vRes;
}

// TODO: Hybrid input classes are a bit redundant.
// Instead of creating a hybrid input, the graph can simply create 2 separate inputs.
// Refactoring is required in the future.
void llm_graph_input_mem_hybrid_k::mSetInput(const mgs_ubatch *pUbatch) {
  vMctx->mGetAttn()->mSetInputKIdxs(vInpAttn->vSelfKIdxs, pUbatch);

  vMctx->mGetAttn()->mSetInputKqMask(vInpAttn->vSelfKqMask, pUbatch, cCparams.vCausalAttn);

  const int64_t cNRs = vMctx->mGetRecr()->mGetNRs();

  if (vInpRs->vSCopy) {
    GGML_ASSERT(ggml_backend_buffer_is_host(vInpRs->vSCopy->buffer));
    int32_t *vData = (int32_t *)vInpRs->vSCopy->data;

    // assuming copy destinations ALWAYS happen ONLY on the cells between head and head+n
    for (uint32_t vI = 0; vI < cNRs; ++vI) {
      vData[vI] = vMctx->mGetRecr()->mSCopy(vI);
    }
  }
}

bool llm_graph_input_mem_hybrid_k::mCanReuse(const llm_graph_params &pParams) {
  const auto *vMctx = static_cast<const mgs_memory_hybrid_context *>(pParams.vMctx);

  this->vMctx = vMctx;

  bool vRes = true;

  vRes &= vInpAttn->vSelfKIdxs->ne[0] == pParams.vUbatch.vNTokens;

  vRes &= fCanReuseKqMask(vInpAttn->vSelfKqMask, vMctx->mGetAttn(), pParams.vUbatch, pParams.vCparams);

  vRes &= vInpRs->vSCopy->ne[0] == vMctx->mGetRecr()->mGetNRs();

  vRes &= vInpRs->vSCopyMain->ne[0] == pParams.vUbatch.vNSeqs;
  vRes &= vInpRs->vSCopyExtra->ne[0] == vMctx->mGetRecr()->mGetNRs() - pParams.vUbatch.vNSeqs;

  vRes &= vInpRs->vHead == vMctx->mGetRecr()->mGetHead();
  vRes &= vInpRs->vRsZ == vMctx->mGetRecr()->mGetRsZ();

  return vRes;
}

void llm_graph_input_mem_hybrid_iswa::mSetInput(const mgs_ubatch *pUbatch) {
  const auto *vAttnCtx = vMctx->mGetAttn();

  // base tensors may not be allocated if there are no non-SWA attention layers
  if (vInpAttn->vSelfKIdxs && vInpAttn->vSelfKIdxs->buffer) {
    vAttnCtx->mGetBase()->mSetInputKIdxs(vInpAttn->vSelfKIdxs, pUbatch);
    vAttnCtx->mGetBase()->mSetInputVIdxs(vInpAttn->vSelfVIdxs, pUbatch);
  }

  if (vInpAttn->vSelfKqMask && vInpAttn->vSelfKqMask->buffer) {
    vAttnCtx->mGetBase()->mSetInputKqMask(vInpAttn->vSelfKqMask, pUbatch, cCparams.vCausalAttn);
  }

  // swa tensors may not be allocated if there are no SWA attention layers
  if (vInpAttn->vSelfKIdxsSwa && vInpAttn->vSelfKIdxsSwa->buffer) {
    vAttnCtx->mGetSwa()->mSetInputKIdxs(vInpAttn->vSelfKIdxsSwa, pUbatch);
    vAttnCtx->mGetSwa()->mSetInputVIdxs(vInpAttn->vSelfVIdxsSwa, pUbatch);
  }

  if (vInpAttn->vSelfKqMaskSwa && vInpAttn->vSelfKqMaskSwa->buffer) {
    vAttnCtx->mGetSwa()->mSetInputKqMask(vInpAttn->vSelfKqMaskSwa, pUbatch, cCparams.vCausalAttn);
  }

  if (vInpAttn->vSelfKRot) {
    vAttnCtx->mGetBase()->mSetInputKRot(vInpAttn->vSelfKRot);
  }

  if (vInpAttn->vSelfVRot) {
    vAttnCtx->mGetBase()->mSetInputVRot(vInpAttn->vSelfVRot);
  }

  if (vInpAttn->vSelfKRotSwa) {
    vAttnCtx->mGetSwa()->mSetInputKRot(vInpAttn->vSelfKRotSwa);
  }

  if (vInpAttn->vSelfVRotSwa) {
    vAttnCtx->mGetSwa()->mSetInputVRot(vInpAttn->vSelfVRotSwa);
  }

  const int64_t cNRs = vMctx->mGetRecr()->mGetNRs();

  if (vInpRs->vSCopy) {
    GGML_ASSERT(ggml_backend_buffer_is_host(vInpRs->vSCopy->buffer));
    int32_t *vData = (int32_t *)vInpRs->vSCopy->data;

    // assuming copy destinations ALWAYS happen ONLY on the cells between head and head+n
    for (uint32_t vI = 0; vI < cNRs; ++vI) {
      vData[vI] = vMctx->mGetRecr()->mSCopy(vI);
    }
  }
}

bool llm_graph_input_mem_hybrid_iswa::mCanReuse(const llm_graph_params &pParams) {
  const auto *vMctx = static_cast<const mgs_memory_hybrid_iswa_context *>(pParams.vMctx);

  this->vMctx = vMctx;

  bool vRes = true;

  const auto *vAttnCtx = vMctx->mGetAttn();

  // base tensors may not be allocated if there are no non-SWA attention layers
  if (vInpAttn->vSelfKIdxs && vInpAttn->vSelfKIdxs->buffer) {
    vRes &= vInpAttn->vSelfKIdxs->ne[0] == pParams.vUbatch.vNTokens;
    // vRes &= vInpAttn->vSelfVIdxs->ne[0] == pParams.vUbatch.vNTokens; // TODO: need to move this to the unified
    // cache and check there
  }

  vRes &= fCanReuseKqMask(vInpAttn->vSelfKqMask, vAttnCtx->mGetBase(), pParams.vUbatch, pParams.vCparams);

  // swa tensors may not be allocated if there are no SWA attention layers
  if (vInpAttn->vSelfKIdxsSwa && vInpAttn->vSelfKIdxsSwa->buffer) {
    vRes &= vInpAttn->vSelfKIdxsSwa->ne[0] == pParams.vUbatch.vNTokens;
    // vRes &= vInpAttn->vSelfVIdxsSwa->ne[0] == pParams.vUbatch.vNTokens; // TODO: need to move this to the
    // unified cache and check there
  }

  vRes &= fCanReuseKqMask(vInpAttn->vSelfKqMaskSwa, vAttnCtx->mGetSwa(), pParams.vUbatch, pParams.vCparams);

  vRes &= vInpRs->vSCopy->ne[0] == vMctx->mGetRecr()->mGetNRs();

  vRes &= vInpRs->vSCopyMain->ne[0] == pParams.vUbatch.vNSeqs;
  vRes &= vInpRs->vSCopyExtra->ne[0] == vMctx->mGetRecr()->mGetNRs() - pParams.vUbatch.vNSeqs;

  vRes &= vInpRs->vHead == vMctx->mGetRecr()->mGetHead();
  vRes &= vInpRs->vRsZ == vMctx->mGetRecr()->mGetRsZ();

  return vRes;
}

void llm_graph_input_sampling::mSetInput(const mgs_ubatch *pUbatch) {
  // set the inputs only for the active samplers in the current ubatch
  std::unordered_set<mgs_seq_id> sActiveSamplers;
  for (uint32_t vI = 0; vI < pUbatch->vNTokens; vI++) {
    if (pUbatch->vOutput[vI]) {
      mgs_seq_id vSeqId = pUbatch->vSeqId[vI][0];
      sActiveSamplers.insert(vSeqId);
    }
  }

  for (auto vSeqId : sActiveSamplers) {
    if (dSamplers.find(vSeqId) == dSamplers.end()) {
      continue;
    }

    auto &vSampler = dSamplers[vSeqId];

    if (vSampler->iface->backend_set_input) {
      vSampler->iface->backend_set_input(vSampler);
    }
  }
}

bool llm_graph_input_sampling::mCanReuse(const llm_graph_params &pParams) {
  if (dSamplers.size() != pParams.dSamplers.size()) {
    return false;
  }

  for (const auto &[vSeqId, vSampler] : pParams.dSamplers) {
    if (dSamplers[vSeqId] != vSampler) {
      return false;
    }
  }

  return true;
}

//
// llm_graph_result
//

llm_graph_result::llm_graph_result(int64_t pMaxNodes) : vMaxNodes(pMaxNodes) {
  mReset();

  const char *vMgsGraphResultDebug = getenv("MGS_GRAPH_RESULT_DEBUG");
  vDebug = vMgsGraphResultDebug ? atoi(vMgsGraphResultDebug) : 0;
}

int64_t llm_graph_result::mGetMaxNodes() const { return vMaxNodes; }

void llm_graph_result::mReset() {
  vTInpTokens = nullptr;
  vTInpEmbd = nullptr;
  vTLogits = nullptr;
  vTEmbd = nullptr;
  vTEmbdPooled = nullptr;
  vTHNextn = nullptr;

  lTLayerInp.resize(MGS_MAX_LAYERS + 1);
  std::fill(lTLayerInp.begin(), lTLayerInp.end(), nullptr);

  dTSampled.clear();
  dTSampledProbs.clear();
  dTSampledLogits.clear();
  dTCandidates.clear();

  vParams = {};

  lInputs.clear();
  lFusedNodes.clear();

  lBufComputeMeta.resize(ggml_tensor_overhead() * vMaxNodes + ggml_graph_overhead_custom(vMaxNodes, false));

  ggml_init_params vParams = {
      /*.mem_size   =*/lBufComputeMeta.size(),
      /*.mem_buffer =*/lBufComputeMeta.data(),
      /*.no_alloc   =*/true,
  };

  vCtxCompute.reset(ggml_init(vParams));

  vGf = ggml_new_graph_custom(vCtxCompute.get(), vMaxNodes, false);
}

void llm_graph_result::mSetInputs(const mgs_ubatch *pUbatch) {
  for (auto &vInput : lInputs) {
    vInput->mSetInput(pUbatch);
  }
}

void llm_graph_result::mSetOutputs(const llm_graph_params &pParams) {
  if (vTLogits != nullptr) {
    ggml_set_output(vTLogits);
  }
  if (vTEmbd != nullptr) {
    ggml_set_output(vTEmbd);
  }
  if (vTEmbdPooled != nullptr) {
    ggml_set_output(vTEmbdPooled);
  }
  if (vTHNextn != nullptr) {
    ggml_set_output(vTHNextn);
  }
  {
    const auto &vEmbeddingsLayerInp = pParams.vCparams.lEmbeddingsLayerInp;
    for (size_t vIl = 0; vIl < vEmbeddingsLayerInp.size(); ++vIl) {
      if (vEmbeddingsLayerInp[vIl]) {
        GGML_ASSERT(lTLayerInp[vIl] != nullptr && "layer input tensor is null");
        ggml_set_output(lTLayerInp[vIl]);
      }
    }
  }
  for (auto &[vSeqId, vT] : dTSampled) {
    if (vT != nullptr) {
      ggml_set_output(vT);
    }
  }
  for (auto &[vSeqId, vT] : dTSampledProbs) {
    if (vT != nullptr) {
      ggml_set_output(vT);
    }
  }
  for (auto &[vSeqId, vT] : dTSampledLogits) {
    if (vT != nullptr) {
      ggml_set_output(vT);
    }
  }
  for (auto &[vSeqId, vT] : dTCandidates) {
    if (vT != nullptr) {
      ggml_set_output(vT);
    }
  }
}

bool llm_graph_result::mCanReuse(const llm_graph_params &pParams) {
  if (!this->vParams.mAllowReuse(pParams)) {
    if (vDebug > 1) {
      MGS_LOG_DEBUG("%s: cannot reuse graph due to incompatible graph parameters\n", __func__);
    }

    return false;
  }

  if (vDebug > 1) {
    MGS_LOG_DEBUG("%s: checking compatibility of %d inputs:\n", __func__, (int)lInputs.size());
  }

  bool vRes = true;

  for (auto &vInput : lInputs) {
    const bool cCur = vInput->mCanReuse(pParams);

    if (vDebug > 1) {
      MGS_LOG_DEBUG("%s: can_reuse = %d\n", "placeholder", cCur);
    }

    vRes = vRes && cCur;
  }

  if (vDebug > 0) {
    MGS_LOG_DEBUG("%s: can reuse graph = %d\n", __func__, vRes);
  }

  return vRes;
}

llm_graph_input_i *llm_graph_result::mAddInput(llm_graph_input_ptr pInput) {
  lInputs.emplace_back(std::move(pInput));
  return lInputs.back().get();
}

void llm_graph_result::mAddFusedNode(llm_graph_fused_node pResult) { lFusedNodes.push_back(pResult); }

void llm_graph_result::mSetParams(const llm_graph_params &pParams) { this->vParams = pParams; }

//
// llm_graph_context
//

llm_graph_context::llm_graph_context(const llm_graph_params &pParams)
    : cArch(pParams.vArch), vHparams(pParams.vHparams), vCparams(pParams.vCparams), vUbatch(pParams.vUbatch),
      cNEmbd(vHparams.vNEmbd), cNLayer(vHparams.mNLayer()), cNLayerNextn(vHparams.vNLayerNextn),
      cNRot(vHparams.mNRot()), cNCtx(vCparams.vNCtx), cNHead(vHparams.mNHead()), cNHeadKv(vHparams.mNHeadKv()),
      cNEmbdHeadK(vHparams.mNEmbdHeadK()), cNEmbdKGqa(vHparams.mNEmbdKGqa()), cNEmbdHeadV(vHparams.mNEmbdHeadV()),
      cNEmbdVGqa(vHparams.mNEmbdVGqa()), cNExpert(vHparams.vNExpert), cNExpertUsed(vHparams.vNExpertUsed),
      cFreqBase(vCparams.vRopeFreqBase), cFreqScale(vCparams.vRopeFreqScale), cExtFactor(vCparams.vYarnExtFactor),
      cAttnFactor(vCparams.vYarnAttnFactor), cBetaFast(vCparams.vYarnBetaFast), cBetaSlow(vCparams.vYarnBetaSlow),
      cNormEps(vHparams.vFNormEps), cNormRmsEps(vHparams.vFNormRmsEps), cNTokens(vUbatch.vNTokens),
      cNOutputs(pParams.vNOutputs), cNCtxOrig(vCparams.vNCtxOrigYarn), cPoolingType(vCparams.vPoolingType),
      cRopeType(vHparams.vRopeType), vSched(pParams.vSched), vBackendCpu(pParams.vBackendCpu), vCvec(pParams.vCvec),
      vLoras(pParams.vLoras), vMctx(pParams.vMctx), vCross(pParams.vCross), dSamplers(pParams.dSamplers),
      vCbFunc(pParams.vCb), vRes(pParams.vRes), vExpertCache(pParams.vExpertCache), vCtx0(vRes->mGetCtx()),
      vGf(vRes->mGetGf()) {
  vRes->mSetParams(pParams);
}

void llm_graph_context::mCb(ggml_tensor *pCur, const char *pName, int pIl) const {
  if (vCbFunc) {
    vCbFunc(vUbatch, pCur, pName, pIl);
  }
}

ggml_tensor *llm_graph_context::mBuildCvec(ggml_tensor *pCur, int pIl) const {
  return vCvec->mApplyTo(vCtx0, pCur, pIl);
}

ggml_tensor *llm_graph_context::mBuildLoraMm(ggml_tensor *pW, ggml_tensor *pCur, ggml_tensor *pWS) const {
  ggml_tensor *vRes = ggml_mul_mat(vCtx0, pW, pCur);

  if (pWS) {
    vRes = ggml_mul(vCtx0, vRes, pWS);
  }

  for (const auto &vLora : *vLoras) {
    mgs_adapter_lora_weight *vLw = vLora.first->mGetWeight(pW);
    if (vLw == nullptr) {
      continue;
    }

    const float cAdapterScale = vLora.second;
    const float cScale = vLw->mGetScale(vLora.first->vAlpha, cAdapterScale);

    ggml_tensor *vAbCur = ggml_mul_mat(vCtx0, vLw->vB, ggml_mul_mat(vCtx0, vLw->vA, pCur));

    vAbCur = ggml_scale(vCtx0, vAbCur, cScale);
    vRes = ggml_add(vCtx0, vRes, vAbCur);
  }

  return vRes;
}

ggml_tensor *llm_graph_context::mBuildLoraMmId(ggml_tensor *pW,   // ggml_tensor * as
                                               ggml_tensor *pCur, // ggml_tensor * b
                                               ggml_tensor *pIds, ggml_tensor *pWS, ggml_tensor *pIdsS) const {
  ggml_tensor *vRes = ggml_mul_mat_id(vCtx0, pW, pCur, pIds);

  if (pWS) {
    const int64_t cNExpert = pWS->ne[0];
    const int64_t cNTokens = pCur->ne[2];
    ggml_tensor *vS = ggml_reshape_3d(vCtx0, pWS, 1, cNExpert, 1);
    vS = ggml_repeat_4d(vCtx0, vS, 1, cNExpert, cNTokens, 1);
    vS = ggml_get_rows(vCtx0, vS, pIdsS ? pIdsS : pIds);
    vRes = ggml_mul(vCtx0, vRes, vS);
  }
  for (const auto &vLora : *vLoras) {
    mgs_adapter_lora_weight *vLw = vLora.first->mGetWeight(pW);
    if (vLw == nullptr) {
      continue;
    }

    const float cAlpha = vLora.first->vAlpha;
    const float cRank = (float)vLw->vB->ne[0];
    const float cScale = cAlpha ? vLora.second * cAlpha / cRank : vLora.second;

    ggml_tensor *vAbCur = ggml_mul_mat_id(vCtx0, vLw->vB, ggml_mul_mat_id(vCtx0, vLw->vA, pCur, pIds), pIds);

    vAbCur = ggml_scale(vCtx0, vAbCur, cScale);
    vRes = ggml_add(vCtx0, vRes, vAbCur);
  }

  return vRes;
}

ggml_tensor *llm_graph_context::mBuildNorm(ggml_tensor *pCur, ggml_tensor *pMw, ggml_tensor *pMb, llm_norm_type pType,
                                           int pIl) const {
  switch (pType) {
  case cLlmNorm:
    pCur = ggml_norm(vCtx0, pCur, vHparams.vFNormEps);
    break;
  case cLlmNormRms:
    pCur = ggml_rms_norm(vCtx0, pCur, vHparams.vFNormRmsEps);
    break;
  case cLlmNormGroup: {
    pCur = ggml_reshape_3d(vCtx0, pCur, pCur->ne[0], 1, pCur->ne[1]);
    pCur = ggml_group_norm(vCtx0, pCur, vHparams.vNNormGroups, vHparams.vFNormGroupEps);
    pCur = ggml_reshape_2d(vCtx0, pCur, pCur->ne[0], pCur->ne[2]);
  } break;
  }

  if (pMw || pMb) {
    mCb(pCur, "norm", pIl);
  }

  if (pMw) {
    pCur = ggml_mul(vCtx0, pCur, pMw);
    if (pMb) {
      mCb(pCur, "norm_w", pIl);
    }
  }

  if (pMb) {
    pCur = ggml_add(vCtx0, pCur, pMb);
  }

  return pCur;
}

llm_graph_qkv llm_graph_context::mBuildQkv(const mgs_layer &pLayer, ggml_tensor *pCur, int64_t pNEmbdHead,
                                           int64_t pNHead, int64_t pNHeadKv, int pIl) const {
  const int64_t cNEmbdQ = pNEmbdHead * pNHead;
  const int64_t cNEmbdKv = pNEmbdHead * pNHeadKv;

  ggml_tensor *vQcur, *vKcur, *vVcur;

  if (pLayer.vWqkv) {
    // fused QKV path
    ggml_tensor *vQkv = mBuildLoraMm(pLayer.vWqkv, pCur, pLayer.vWqkvS);
    mCb(vQkv, "wqkv", pIl);
    if (pLayer.vWqkvB) {
      vQkv = ggml_add(vCtx0, vQkv, pLayer.vWqkvB);
      mCb(vQkv, "wqkv_b", pIl);
    }
    if (vHparams.vFClampKqv > 0.0f) {
      vQkv = ggml_clamp(vCtx0, vQkv, -vHparams.vFClampKqv, vHparams.vFClampKqv);
      mCb(vQkv, "wqkv_clamped", pIl);
    }
    vQcur =
        ggml_view_3d(vCtx0, vQkv, pNEmbdHead, pNHead, cNTokens, ggml_row_size(vQkv->type, pNEmbdHead), vQkv->nb[1], 0);
    vKcur = ggml_view_3d(vCtx0, vQkv, pNEmbdHead, pNHeadKv, cNTokens, ggml_row_size(vQkv->type, pNEmbdHead),
                         vQkv->nb[1], ggml_row_size(vQkv->type, cNEmbdQ));
    vVcur = ggml_view_3d(vCtx0, vQkv, pNEmbdHead, pNHeadKv, cNTokens, ggml_row_size(vQkv->type, pNEmbdHead),
                         vQkv->nb[1], ggml_row_size(vQkv->type, cNEmbdQ + cNEmbdKv));
  } else {
    // separate Q/K/V path
    vQcur = mBuildLoraMm(pLayer.vWq, pCur, pLayer.vWqS);
    mCb(vQcur, "Qcur", pIl);
    if (pLayer.vWqB) {
      vQcur = ggml_add(vCtx0, vQcur, pLayer.vWqB);
      mCb(vQcur, "Qcur", pIl);
    }
    if (vHparams.vFClampKqv > 0.0f) {
      vQcur = ggml_clamp(vCtx0, vQcur, -vHparams.vFClampKqv, vHparams.vFClampKqv);
      mCb(vQcur, "Qcur_clamped", pIl);
    }
    vKcur = mBuildLoraMm(pLayer.vWk, pCur, pLayer.vWkS);
    mCb(vKcur, "Kcur", pIl);
    if (pLayer.vWkB) {
      vKcur = ggml_add(vCtx0, vKcur, pLayer.vWkB);
      mCb(vKcur, "Kcur", pIl);
    }
    if (vHparams.vFClampKqv > 0.0f) {
      vKcur = ggml_clamp(vCtx0, vKcur, -vHparams.vFClampKqv, vHparams.vFClampKqv);
      mCb(vKcur, "Kcur_clamped", pIl);
    }
    vVcur = mBuildLoraMm(pLayer.vWv, pCur, pLayer.vWvS);
    mCb(vVcur, "Vcur", pIl);
    if (pLayer.vWvB) {
      vVcur = ggml_add(vCtx0, vVcur, pLayer.vWvB);
      mCb(vVcur, "Vcur", pIl);
    }
    if (vHparams.vFClampKqv > 0.0f) {
      vVcur = ggml_clamp(vCtx0, vVcur, -vHparams.vFClampKqv, vHparams.vFClampKqv);
      mCb(vVcur, "Vcur_clamped", pIl);
    }
    vQcur = ggml_reshape_3d(vCtx0, vQcur, pNEmbdHead, pNHead, cNTokens);
    vKcur = ggml_reshape_3d(vCtx0, vKcur, pNEmbdHead, pNHeadKv, cNTokens);
    vVcur = ggml_reshape_3d(vCtx0, vVcur, pNEmbdHead, pNHeadKv, cNTokens);
  }

  mCb(vQcur, "Qcur", pIl);
  mCb(vKcur, "Kcur", pIl);
  mCb(vVcur, "Vcur", pIl);

  return {vQcur, vKcur, vVcur};
}

ggml_tensor *llm_graph_context::mBuildFfn(ggml_tensor *pCur, ggml_tensor *pUp, ggml_tensor *pUpB, ggml_tensor *pUpS,
                                          ggml_tensor *pGate, ggml_tensor *pGateB, ggml_tensor *pGateS,
                                          ggml_tensor *pDown, ggml_tensor *pDownB, ggml_tensor *pDownS,
                                          ggml_tensor *pActScales, llm_ffn_op_type pTypeOp, llm_ffn_gate_type pTypeGate,
                                          int pIl) const {
  // NVFP4 support is currently restricted to
  // 1) LORA absence (*_s would be applied after LORA residual, which is incorrect)
  // 2) bias absense (*_s would be applied after bias addition, which is incorrect)
  // TODO: disambiguate LLM-architectural scales (which use *_s) from NVFP4 scale_2 (which also uses *_s currently)
  auto vHasLora = [this](ggml_tensor *pW) {
    if (!pW) {
      return false;
    }
    for (const auto &vLora : *vLoras) {
      if (vLora.first->mGetWeight(pW) != nullptr) {
        return true;
      }
    }
    return false;
  };

  GGML_ASSERT(!pUpS || !pUpB || !pUp || pUp->type != GGML_TYPE_NVFP4);
  GGML_ASSERT(!pGateS || !pGateB || !pGate || pGate->type != GGML_TYPE_NVFP4);
  GGML_ASSERT(!pDownS || !pDownB || !pDown || pDown->type != GGML_TYPE_NVFP4);
  GGML_ASSERT(!pUpS || !pUp || pUp->type != GGML_TYPE_NVFP4 || !vHasLora(pUp));
  GGML_ASSERT(!pGateS || !pGate || pGate->type != GGML_TYPE_NVFP4 || !vHasLora(pGate));
  GGML_ASSERT(!pDownS || !pDown || pDown->type != GGML_TYPE_NVFP4 || !vHasLora(pDown));

  ggml_tensor *vTmp = pUp ? mBuildLoraMm(pUp, pCur) : pCur;
  mCb(vTmp, "ffn_up", pIl);

  if (pUpB) {
    vTmp = ggml_add(vCtx0, vTmp, pUpB);
    mCb(vTmp, "ffn_up_b", pIl);
  }

  if (pUpS) {
    vTmp = ggml_mul(vCtx0, vTmp, pUpS);
    mCb(vTmp, "ffn_up_s", pIl);
  }

  if (pGate) {
    switch (pTypeGate) {
    case cLlmFfnSeq: {
      pCur = mBuildLoraMm(pGate, vTmp);
      mCb(pCur, "ffn_gate", pIl);
    } break;
    case cLlmFfnPar: {
      pCur = mBuildLoraMm(pGate, pCur);
      mCb(pCur, "ffn_gate", pIl);
    } break;
    }

    if (pGateB) {
      pCur = ggml_add(vCtx0, pCur, pGateB);
      mCb(pCur, "ffn_gate_b", pIl);
    }

    if (pGateS) {
      pCur = ggml_mul(vCtx0, pCur, pGateS);
      mCb(pCur, "ffn_gate_s", pIl);
    }

  } else {
    pCur = vTmp;
  }

  switch (pTypeOp) {
  case cLlmFfnSilu:
    if (pGate && pTypeGate == cLlmFfnPar) {
      if (pIl >= 0) {
        const float cLimit = vHparams.aSwigluClampShexp[pIl];
        constexpr float cEps = 1e-6f;
        if (cLimit > cEps) {
          if (cArch == cLlmArchGlm5Next || cArch == cLlmArchDeepseek41) {
            // fused silu(min(gate, limit)) * clamp(up, -limit, limit)
            pCur = ggml_swiglu_clamp(vCtx0, pCur, vTmp, cLimit);
            mCb(pCur, "ffn_swiglu_limited", pIl);
            pTypeGate = cLlmFfnSeq;
            break;
          }

          vTmp = ggml_clamp(vCtx0, vTmp, -cLimit, cLimit);
          mCb(vTmp, "ffn_up_clamped", pIl);

          ggml_tensor *vGateAct = ggml_silu(vCtx0, pCur);
          mCb(vGateAct, "ffn_silu", pIl);
          vGateAct = ggml_clamp(vCtx0, vGateAct, -INFINITY, cLimit);
          mCb(vGateAct, "ffn_silu_clamped", pIl);
          pCur = ggml_mul(vCtx0, vGateAct, vTmp);
          mCb(pCur, "ffn_swiglu_limited", pIl);
          pTypeGate = cLlmFfnSeq;
          break;
        }
      }

      pCur = ggml_swiglu_split(vCtx0, pCur, vTmp);
      mCb(pCur, "ffn_swiglu", pIl);
      pTypeGate = cLlmFfnSeq;
    } else {
      pCur = ggml_silu(vCtx0, pCur);
      mCb(pCur, "ffn_silu", pIl);
    }
    break;
  case cLlmFfnGelu:
    if (pGate && pTypeGate == cLlmFfnPar) {
      pCur = ggml_geglu_split(vCtx0, pCur, vTmp);
      mCb(pCur, "ffn_geglu", pIl);
      pTypeGate = cLlmFfnSeq;
    } else {
      pCur = ggml_gelu(vCtx0, pCur);
      mCb(pCur, "ffn_gelu", pIl);
      if (pActScales != NULL) {
        pCur = ggml_div(vCtx0, pCur, pActScales);
        mCb(pCur, "ffn_act", pIl);
      }
    }
    break;
  case cLlmFfnRelu:
    if (pGate && pTypeGate == cLlmFfnPar) {
      pCur = ggml_reglu_split(vCtx0, pCur, vTmp);
      mCb(pCur, "ffn_reglu", pIl);
      pTypeGate = cLlmFfnSeq;
    } else {
      pCur = ggml_relu(vCtx0, pCur);
      mCb(pCur, "ffn_relu", pIl);
    }
    break;
  case cLlmFfnReluSqr: {
    pCur = ggml_relu(vCtx0, pCur);
    mCb(pCur, "ffn_relu", pIl);

    pCur = ggml_sqr(vCtx0, pCur);
    mCb(pCur, "ffn_sqr(relu)", pIl);
  } break;
  case cLlmFfnSwiglu: {
    pCur = ggml_swiglu(vCtx0, pCur);
    mCb(pCur, "ffn_swiglu", pIl);
  } break;
  case cLlmFfnSwigluOaiMoe:
    if (pGate && pTypeGate == cLlmFfnPar) {
      // same alpha/limit constants as gpt-oss
      const float cAlpha = 1.702f;
      const float cLimit = 7.0f;
      pCur = ggml_swiglu_oai(vCtx0, pCur, vTmp, cAlpha, cLimit);
      mCb(pCur, "ffn_swiglu_oai", pIl);
      pTypeGate = cLlmFfnSeq;
    } else {
      GGML_ABORT("LLM_FFN_SWIGLU_OAI_MOE requires a parallel gate");
    }
    break;
  case cLlmFfnGeglu: {
    pCur = ggml_geglu(vCtx0, pCur);
    mCb(pCur, "ffn_geglu", pIl);
  } break;
  case cLlmFfnReglu: {
    pCur = ggml_reglu(vCtx0, pCur);
    mCb(pCur, "ffn_reglu", pIl);
  } break;
  default:
    GGML_ABORT("fatal error");
  }

  if (pGate && pTypeGate == cLlmFfnPar) {
    pCur = ggml_mul(vCtx0, pCur, vTmp);
    mCb(pCur, "ffn_gate_par", pIl);
  }

  if (pDown) {
    pCur = mBuildLoraMm(pDown, pCur);
    if (cArch == cLlmArchGlm4 || cArch == cLlmArchGlm4Moe || cArch == cLlmArchJais2) {
      // GLM4, GLM4_MOE, and JAIS2 seem to have numerical issues with half-precision accumulators
      ggml_mul_mat_set_prec(pCur, GGML_PREC_F32);
    }
  }

  if (pDownB) {
    mCb(pCur, "ffn_down", pIl);
  }

  if (pDownB) {
    pCur = ggml_add(vCtx0, pCur, pDownB);
  }

  if (pDownS) {
    pCur = ggml_mul(vCtx0, pCur, pDownS);
    mCb(pCur, "ffn_down_s", pIl);
  }

  return pCur;
}

ggml_tensor *llm_graph_context::mBuildMoeFfn(ggml_tensor *pCur, ggml_tensor *pGateInp, ggml_tensor *pUpExps,
                                             ggml_tensor *pGateExps, ggml_tensor *pDownExps, ggml_tensor *pExpProbsB,
                                             int64_t pNExpert, int64_t pNExpertUsed, llm_ffn_op_type pTypeOp,
                                             bool pNormW, float pWScale, mgs_expert_gating_func_type pGatingOp, int pIl,
                                             ggml_tensor *pProbsIn, ggml_tensor *pGateUpExps, ggml_tensor *pUpExpsS,
                                             ggml_tensor *pGateExpsS, ggml_tensor *pDownExpsS,
                                             ggml_tensor *pSelectedExpertsIn) const {
  return mBuildMoeFfn(pCur, pGateInp, /* gate_inp_b  */ nullptr, pUpExps, /* up_exps_b   */ nullptr, pGateExps,
                      /* gate_exps_b */ nullptr, pDownExps, /* down_exps_b */ nullptr, pExpProbsB, pNExpert,
                      pNExpertUsed, pTypeOp, pNormW, pWScale, pGatingOp, pIl, pProbsIn, pGateUpExps,
                      /* gate_up_exps_b */ nullptr, pUpExpsS, pGateExpsS, pDownExpsS, pSelectedExpertsIn);
}

ggml_tensor *llm_graph_context::mBuildMoeFfn(ggml_tensor *pCur, ggml_tensor *pGateInp, ggml_tensor *pGateInpB,
                                             ggml_tensor *pUpExps, ggml_tensor *pUpExpsB, ggml_tensor *pGateExps,
                                             ggml_tensor *pGateExpsB, ggml_tensor *pDownExps, ggml_tensor *pDownExpsB,
                                             ggml_tensor *pExpProbsB, int64_t pNExpert, int64_t pNExpertUsed,
                                             llm_ffn_op_type pTypeOp, bool pNormW, float pWScale,
                                             mgs_expert_gating_func_type pGatingOp, int pIl, ggml_tensor *pProbsIn,
                                             ggml_tensor *pGateUpExps, ggml_tensor *pGateUpExpsB, ggml_tensor *pUpExpsS,
                                             ggml_tensor *pGateExpsS, ggml_tensor *pDownExpsS,
                                             ggml_tensor *pSelectedExpertsIn) const {
  const int64_t cNEmbd = pCur->ne[0];
  const int64_t cNTokens = pCur->ne[1];

  ggml_tensor *vLogits = nullptr;

  if (pProbsIn == nullptr) {
    vLogits = mBuildLoraMm(pGateInp, pCur); // [n_expert, n_tokens]
    if (pGatingOp == cMgsExpertGatingFuncTypeSqrtSoftplus) {
      ggml_mul_mat_set_prec(vLogits, GGML_PREC_F32);
    }
    mCb(vLogits, "ffn_moe_logits", pIl);
  } else {
    vLogits = pProbsIn;
  }

  if (pGateInpB) {
    vLogits = ggml_add(vCtx0, vLogits, pGateInpB);
    mCb(vLogits, "ffn_moe_logits_biased", pIl);
  }

  ggml_tensor *vProbs = nullptr;
  switch (pGatingOp) {
  case cMgsExpertGatingFuncTypeSoftmax: {
    vProbs = ggml_soft_max(vCtx0, vLogits); // [n_expert, n_tokens]
  } break;
  case cMgsExpertGatingFuncTypeSigmoid: {
    vProbs = ggml_sigmoid(vCtx0, vLogits); // [n_expert, n_tokens]
  } break;
  case cMgsExpertGatingFuncTypeSoftmaxWeight: {
    vProbs = vLogits; // [n_expert, n_tokens]
  } break;
  case cMgsExpertGatingFuncTypeSqrtSoftplus: {
    vProbs = ggml_sqrt(vCtx0, ggml_softplus(vCtx0, vLogits)); // [n_expert, n_tokens]
  } break;
  default:
    GGML_ABORT("fatal error");
  }
  mCb(vProbs, "ffn_moe_probs", pIl);

  // add experts selection bias - introduced in DeepSeek V3
  // leave probs unbiased as it's later used to get expert weights
  ggml_tensor *vSelectionProbs = vProbs;
  if (pExpProbsB != nullptr) {
    vSelectionProbs = ggml_add(vCtx0, vProbs, pExpProbsB);
    mCb(vSelectionProbs, "ffn_moe_probs_biased", pIl);
  }

  if (cArch == cLlmArchGrovemoe) {
    vSelectionProbs = ggml_sigmoid(vCtx0, vLogits); // [n_expert, n_tokens]
    mCb(vSelectionProbs, "ffn_moe_probs_biased", pIl);
  }

  // select top n_group_used expert groups
  // https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/e815299b0bcbac849fa540c768ef21845365c9eb/modeling_deepseek.py#L440-L457
  if (vHparams.vNExpertGroups > 1 && cNTokens > 0) {
    const int64_t cNExpPerGroup = pNExpert / vHparams.vNExpertGroups;

    // organize experts into n_expert_groups
    ggml_tensor *vSelectionGroups = ggml_reshape_3d(vCtx0, vSelectionProbs, cNExpPerGroup, vHparams.vNExpertGroups,
                                                    cNTokens); // [n_exp_per_group, n_expert_groups, n_tokens]

    ggml_tensor *vGroupScores = ggml_argsort_top_k(vCtx0, vSelectionGroups, 2); // [2, n_expert_groups, n_tokens]
    vGroupScores = ggml_get_rows(vCtx0,
                                 ggml_reshape_4d(vCtx0, vSelectionGroups, 1, vSelectionGroups->ne[0],
                                                 vSelectionGroups->ne[1], vSelectionGroups->ne[2]),
                                 vGroupScores); // [1, 2, n_expert_groups, n_tokens]

    // get top n_group_used expert groups
    vGroupScores = ggml_sum_rows(vCtx0, ggml_reshape_3d(vCtx0, vGroupScores, vGroupScores->ne[1], vGroupScores->ne[2],
                                                        vGroupScores->ne[3])); // [1, n_expert_groups, n_tokens]
    vGroupScores =
        ggml_reshape_2d(vCtx0, vGroupScores, vGroupScores->ne[1], vGroupScores->ne[2]); // [n_expert_groups, n_tokens]

    ggml_tensor *vExpertGroups =
        ggml_argsort_top_k(vCtx0, vGroupScores, vHparams.vNGroupUsed); // [n_group_used, n_tokens]
    mCb(vExpertGroups, "ffn_moe_group_topk", pIl);

    // mask out the other groups
    vSelectionProbs =
        ggml_get_rows(vCtx0, vSelectionGroups, vExpertGroups); // [n_exp_per_group, n_group_used, n_tokens]
    vSelectionProbs = ggml_set_rows(vCtx0, ggml_fill(vCtx0, vSelectionGroups, -INFINITY), vSelectionProbs,
                                    vExpertGroups); // [n_exp_per_group, n_expert_groups, n_tokens]
    vSelectionProbs = ggml_reshape_2d(vCtx0, vSelectionProbs, pNExpert, cNTokens); // [n_expert, n_tokens]
    mCb(vSelectionProbs, "ffn_moe_probs_masked", pIl);
  }

  // select experts
  ggml_tensor *vSelectedExperts = pSelectedExpertsIn;
  if (vSelectedExperts == nullptr) {
    vSelectedExperts = ggml_argsort_top_k(vCtx0, vSelectionProbs, pNExpertUsed); // [n_expert_used, n_tokens]
    mCb(vSelectedExperts->src[0], "ffn_moe_argsort", pIl);
  }
  mCb(vSelectedExperts, "ffn_moe_topk", pIl);

  if (cArch == cLlmArchGrovemoe && pNExpert != vHparams.vNExpert) {
    // TODO: Use scalar div instead when/if implemented
    ggml_tensor *vFSel = ggml_cast(vCtx0, vSelectedExperts, GGML_TYPE_F32);
    vSelectedExperts = ggml_cast(vCtx0, ggml_scale(vCtx0, vFSel, 1.0f / float(vHparams.vNGroupExperts)), GGML_TYPE_I32);
    vProbs = ggml_reshape_3d(vCtx0, vProbs, 1, vHparams.vNExpert, cNTokens);
  } else {
    vProbs = ggml_reshape_3d(vCtx0, vProbs, 1, pNExpert, cNTokens);
  }

  ggml_tensor *vWeights = ggml_get_rows(vCtx0, vProbs, vSelectedExperts); // [1, n_expert_used, n_tokens]
  mCb(vWeights, "ffn_moe_weights", pIl);

  if (pGatingOp == cMgsExpertGatingFuncTypeSoftmaxWeight) {
    vWeights = ggml_reshape_2d(vCtx0, vWeights, pNExpertUsed, cNTokens);
    vWeights = ggml_soft_max(vCtx0, vWeights); // [n_expert_used, n_tokens]
    vWeights = ggml_reshape_3d(vCtx0, vWeights, 1, pNExpertUsed, cNTokens);
    mCb(vWeights, "ffn_moe_weights_softmax", pIl);
  }

  if (pNormW) {
    vWeights = ggml_reshape_2d(vCtx0, vWeights, pNExpertUsed, cNTokens);

    ggml_tensor *vWeightsSum = ggml_sum_rows(vCtx0, vWeights); // [1, n_tokens]
    mCb(vWeightsSum, "ffn_moe_weights_sum", pIl);

    // Avoid division by zero, clamp to smallest number representable by F16
    vWeightsSum = ggml_clamp(vCtx0, vWeightsSum, 6.103515625e-5, INFINITY);
    mCb(vWeightsSum, "ffn_moe_weights_sum_clamped", pIl);

    vWeights = ggml_div(vCtx0, vWeights, vWeightsSum); // [n_expert_used, n_tokens]
    mCb(vWeights, "ffn_moe_weights_norm", pIl);

    vWeights = ggml_reshape_3d(vCtx0, vWeights, 1, pNExpertUsed, cNTokens);
  }
  if (pWScale != 0.0f && pWScale != 1.0f) {
    vWeights = ggml_scale(vCtx0, vWeights, pWScale);
    mCb(vWeights, "ffn_moe_weights_scaled", pIl);
  }

  // call early so that topk-moe can be used
  ggml_build_forward_expand(vGf, vWeights);

  pCur = ggml_reshape_3d(vCtx0, pCur, cNEmbd, 1, cNTokens);

  // Expert chain shared by the regular execution and by both branches of the hybrid
  // CPU/device execution: selected_experts picks the experts (or the device cache
  // slots) and selected_experts_s picks their per-expert scales. The parameters keep
  // the names of the upstream implementation so the chain itself is unchanged.
  auto fBuildExperts = [&](ggml_tensor *pCur, ggml_tensor *pSelectedExperts, ggml_tensor *pSelectedExpertsS,
                           ggml_tensor *pUpExps, ggml_tensor *pUpExpsB, ggml_tensor *pUpExpsS, ggml_tensor *pGateExps,
                           ggml_tensor *pGateExpsB, ggml_tensor *pGateExpsS, ggml_tensor *pGateUpExps,
                           ggml_tensor *pGateUpExpsB, ggml_tensor *pDownExps, ggml_tensor *pDownExpsB,
                           ggml_tensor *pDownExpsS) -> ggml_tensor * {
    ggml_tensor *vUp = nullptr;
    ggml_tensor *vExperts = nullptr;

    if (pGateUpExps) {
      // merged gate_up path: one mul_mat_id, then split into gate and up views
      // [n_ff*2, n_expert_used, n_tokens]
      ggml_tensor *vGateUp = mBuildLoraMmId(pGateUpExps, pCur, pSelectedExperts, pUpExpsS, pSelectedExpertsS);
      mCb(vGateUp, "ffn_moe_gate_up", pIl);

      if (pUpExpsS) {
        mCb(vGateUp, "ffn_moe_gate_up_scaled", pIl);
      }

      if (pGateUpExpsB) {
        vGateUp = ggml_add_id(vCtx0, vGateUp, pGateUpExpsB, pSelectedExperts);
        mCb(vGateUp, "ffn_moe_gate_up_biased", pIl);
      }

      const int64_t cNFf = vGateUp->ne[0] / 2;
      pCur = ggml_view_3d(vCtx0, vGateUp, cNFf, vGateUp->ne[1], vGateUp->ne[2], vGateUp->nb[1], vGateUp->nb[2], 0);
      mCb(pCur, "ffn_moe_gate", pIl);
      vUp = ggml_view_3d(vCtx0, vGateUp, cNFf, vGateUp->ne[1], vGateUp->ne[2], vGateUp->nb[1], vGateUp->nb[2],
                         cNFf * vGateUp->nb[0]);
      mCb(vUp, "ffn_moe_up", pIl);
    } else {
      // separate gate and up path
      // [n_ff, n_expert_used, n_tokens]
      vUp = mBuildLoraMmId(pUpExps, pCur, pSelectedExperts, pUpExpsS, pSelectedExpertsS);
      mCb(vUp, "ffn_moe_up", pIl);

      if (pUpExpsS) {
        mCb(vUp, "ffn_moe_up_scaled", pIl);
      }

      if (pUpExpsB) {
        vUp = ggml_add_id(vCtx0, vUp, pUpExpsB, pSelectedExperts);
        mCb(vUp, "ffn_moe_up_biased", pIl);
      }

      if (pGateExps) {
        // [n_ff, n_expert_used, n_tokens]
        pCur = mBuildLoraMmId(pGateExps, pCur, pSelectedExperts, pGateExpsS, pSelectedExpertsS);
        mCb(pCur, "ffn_moe_gate", pIl);
      } else {
        pCur = vUp;
      }

      if (pGateExpsS) {
        mCb(pCur, "ffn_moe_gate_scaled", pIl);
      }

      if (pGateExpsB) {
        pCur = ggml_add_id(vCtx0, pCur, pGateExpsB, pSelectedExperts);
        mCb(pCur, "ffn_moe_gate_biased", pIl);
      }
    }

    const bool cHasGate = pGateExps || pGateUpExps;

    switch (pTypeOp) {
    case cLlmFfnSilu:
      if (pGateExps) {
        if (pIl >= 0) {
          const float cLimit = vHparams.aSwigluClampExp[pIl];
          constexpr float cEps = 1e-6f;
          if (cLimit > cEps) {
            if (cArch == cLlmArchGlm5Next || cArch == cLlmArchDeepseek41) {
              // fused silu(min(gate, limit)) * clamp(up, -limit, limit)
              pCur = ggml_swiglu_clamp(vCtx0, pCur, vUp, cLimit);
              mCb(pCur, "ffn_moe_swiglu_limited", pIl);
              break;
            }

            vUp = ggml_clamp(vCtx0, vUp, -cLimit, cLimit);
            mCb(vUp, "ffn_moe_up_clamped", pIl);

            ggml_tensor *vGateAct = ggml_silu(vCtx0, pCur);
            mCb(vGateAct, "ffn_moe_silu", pIl);
            vGateAct = ggml_clamp(vCtx0, vGateAct, -INFINITY, cLimit);
            mCb(vGateAct, "ffn_moe_silu_clamped", pIl);
            pCur = ggml_mul(vCtx0, vGateAct, vUp);
            mCb(pCur, "ffn_moe_swiglu_limited", pIl);
            break;
          }
        }
      }

      if (cHasGate) {
        pCur = ggml_swiglu_split(vCtx0, pCur, vUp);
        mCb(pCur, "ffn_moe_swiglu", pIl);
      } else {
        pCur = ggml_silu(vCtx0, pCur);
        mCb(pCur, "ffn_moe_silu", pIl);
      }
      break;
    case cLlmFfnGelu:
      if (cHasGate) {
        pCur = ggml_geglu_split(vCtx0, pCur, vUp);
        mCb(pCur, "ffn_moe_geglu", pIl);
      } else {
        pCur = ggml_gelu(vCtx0, pCur);
        mCb(pCur, "ffn_moe_gelu", pIl);
      }
      break;
    case cLlmFfnSwigluOaiMoe: {
      // TODO: move to hparams?
      constexpr float cAlpha = 1.702f;
      constexpr float cLimit = 7.0f;
      pCur = ggml_swiglu_oai(vCtx0, pCur, vUp, cAlpha, cLimit);
      mCb(pCur, "ffn_moe_swiglu_oai", pIl);
    } break;
    case cLlmFfnSitu: {
      // situ(gate, up) = beta*tanh(gate/beta)*sigmoid(gate) * lb*tanh(up/lb)
      GGML_ASSERT(cHasGate);
      const float cBeta = vHparams.vSituBeta;
      const float cLinearBeta = vHparams.vSituLinearBeta;

      ggml_tensor *vAct = ggml_scale(vCtx0, ggml_tanh(vCtx0, ggml_scale(vCtx0, pCur, 1.0f / cBeta)), cBeta);
      vAct = ggml_mul(vCtx0, vAct, ggml_sigmoid(vCtx0, pCur));
      if (cLinearBeta > 0.0f) {
        vUp = ggml_scale(vCtx0, ggml_tanh(vCtx0, ggml_scale(vCtx0, vUp, 1.0f / cLinearBeta)), cLinearBeta);
      }
      pCur = ggml_mul(vCtx0, vAct, vUp);
      mCb(pCur, "ffn_moe_situ", pIl);
    } break;
    case cLlmFfnRelu:
      if (cHasGate) {
        pCur = ggml_reglu_split(vCtx0, pCur, vUp);
        mCb(pCur, "ffn_moe_reglu", pIl);
      } else {
        pCur = ggml_relu(vCtx0, pCur);
        mCb(pCur, "ffn_moe_relu", pIl);
      }
      break;
    case cLlmFfnReluSqr:
      if (cHasGate) {
        // TODO: add support for gated squared relu
        GGML_ABORT("fatal error: gated squared relu not implemented");
      } else {
        pCur = ggml_relu(vCtx0, pCur);
        pCur = ggml_sqr(vCtx0, pCur);
        mCb(pCur, "ffn_moe_relu_sqr", pIl);
      }
      break;
    default:
      GGML_ABORT("fatal error");
    }

    // [n_embd, n_expert_used, n_tokens]
    vExperts = mBuildLoraMmId(pDownExps, pCur, pSelectedExperts, pDownExpsS, pSelectedExpertsS);
    mCb(vExperts, "ffn_moe_down", pIl);

    if (pDownExpsS) {
      mCb(vExperts, "ffn_moe_down_scaled", pIl);
    }

    if (pDownExpsB) {
      vExperts = ggml_add_id(vCtx0, vExperts, pDownExpsB, pSelectedExperts);
      mCb(vExperts, "ffn_moe_down_biased", pIl);
    }
    return vExperts;
  };

  // Hybrid execution for small micro-batches whose experts live in host memory: the
  // experts held by the device cache (plus the misses uploaded during the partition)
  // run on the device while the remaining experts run on the CPU at the same time.
  mgs_expert_cache_layer *vCacheLayer = nullptr;
  if (vExpertCache != nullptr && vSched != nullptr && vBackendCpu != nullptr && pIl >= 0 && cNTokens > 0 &&
      cNTokens < vExpertCache->mMaxTokens()) {
    mgs_expert_tensors vTensors;
    vTensors.aRoles[cMgsExpertRoleUp] = pUpExps;
    vTensors.aRoles[cMgsExpertRoleGate] = pGateExps;
    vTensors.aRoles[cMgsExpertRoleGateUp] = pGateUpExps;
    vTensors.aRoles[cMgsExpertRoleDown] = pDownExps;
    vTensors.aRoles[cMgsExpertRoleUpB] = pUpExpsB;
    vTensors.aRoles[cMgsExpertRoleGateB] = pGateExpsB;
    vTensors.aRoles[cMgsExpertRoleGateUpB] = pGateUpExpsB;
    vTensors.aRoles[cMgsExpertRoleDownB] = pDownExpsB;
    vTensors.aRoles[cMgsExpertRoleUpS] = pUpExpsS;
    vTensors.aRoles[cMgsExpertRoleGateS] = pGateExpsS;
    vTensors.aRoles[cMgsExpertRoleDownS] = pDownExpsS;

    // LoRA adapters on the experts would be missing from the device branch.
    bool vHasExpertLora = false;
    for (const auto &vLora : *vLoras) {
      for (ggml_tensor *vTensor : vTensors.aRoles) {
        vHasExpertLora = vHasExpertLora || (vTensor != nullptr && vLora.first->mGetWeight(vTensor) != nullptr);
      }
    }
    if (!vHasExpertLora) {
      vCacheLayer = vExpertCache->mFindLayer(pIl, vTensors, pNExpertUsed);
    }
  }

  ggml_tensor *vExperts = nullptr;

  if (vCacheLayer == nullptr) {
    vExperts = fBuildExperts(pCur, vSelectedExperts, vSelectedExperts, pUpExps, pUpExpsB, pUpExpsS, pGateExps,
                             pGateExpsB, pGateExpsS, pGateUpExps, pGateUpExpsB, pDownExps, pDownExpsB, pDownExpsS);
  } else {
    ggml_backend_t vDeviceBackend = vExpertCache->mBackend(vCacheLayer);
    const mgs_expert_tensors &vPool = vExpertCache->mPoolTensors(vCacheLayer);

    // Pin every node of a branch to its backend; views follow their source tensors.
    auto fAssignBackend = [&](int pStart, int pEnd, ggml_backend_t pBackend) {
      for (int vIndex = pStart; vIndex < pEnd; ++vIndex) {
        ggml_tensor *vNode = ggml_graph_node(vGf, vIndex);
        if (vNode->op == GGML_OP_VIEW || vNode->op == GGML_OP_RESHAPE || vNode->op == GGML_OP_PERMUTE ||
            vNode->op == GGML_OP_TRANSPOSE || vNode->op == GGML_OP_NONE) {
          continue;
        }
        if (ggml_backend_supports_op(pBackend, vNode)) {
          ggml_backend_sched_set_tensor_backend(vSched, vNode, pBackend);
        }
      }
    };

    // The CPU partition runs first: it copies the selection and the input to the host,
    // then queues the device identifiers and the expert uploads.
    ggml_tensor *vPartition = vExpertCache->mBuildPartition(vCtx0, vCacheLayer, vSelectedExperts, pCur);
    mCb(vPartition, "ffn_moe_hybrid_partition", pIl);
    ggml_build_forward_expand(vGf, vPartition);
    ggml_backend_sched_set_tensor_backend(vSched, vPartition, vBackendCpu);

    // Device branch over the cache slots. It is queued before the CPU branch starts.
    const int vDeviceStart = ggml_graph_n_nodes(vGf);
    ggml_tensor *vDeviceIds = vExpertCache->mBuildDeviceIds(vCtx0, vCacheLayer, pNExpertUsed, cNTokens);
    ggml_tensor *vDeviceExperts = fBuildExperts(
        pCur, vDeviceIds, vDeviceIds, vPool.aRoles[cMgsExpertRoleUp], vPool.aRoles[cMgsExpertRoleUpB],
        vPool.aRoles[cMgsExpertRoleUpS], vPool.aRoles[cMgsExpertRoleGate], vPool.aRoles[cMgsExpertRoleGateB],
        vPool.aRoles[cMgsExpertRoleGateS], vPool.aRoles[cMgsExpertRoleGateUp], vPool.aRoles[cMgsExpertRoleGateUpB],
        vPool.aRoles[cMgsExpertRoleDown], vPool.aRoles[cMgsExpertRoleDownB], vPool.aRoles[cMgsExpertRoleDownS]);
    ggml_build_forward_expand(vGf, vDeviceExperts);
    fAssignBackend(vDeviceStart, ggml_graph_n_nodes(vGf), vDeviceBackend);

    // CPU branch over the host weights: -1 identifiers skip the device experts.
    const int vCpuStart = ggml_graph_n_nodes(vGf);
    ggml_tensor *vCpuIds = ggml_view_2d(vCtx0, vPartition, pNExpertUsed, cNTokens, vPartition->nb[1], 0);
    ggml_tensor *vScaleIds =
        ggml_view_2d(vCtx0, vPartition, pNExpertUsed, cNTokens, vPartition->nb[1], vPartition->nb[2]);
    ggml_tensor *vCpuExperts =
        fBuildExperts(pCur, vCpuIds, vScaleIds, pUpExps, pUpExpsB, pUpExpsS, pGateExps, pGateExpsB, pGateExpsS,
                      pGateUpExps, pGateUpExpsB, pDownExps, pDownExpsB, pDownExpsS);
    ggml_tensor *vCpuTimer = vExpertCache->mBuildCpuTimer(vCtx0, vCacheLayer, vCpuExperts);
    ggml_build_forward_expand(vGf, vCpuTimer);
    fAssignBackend(vCpuStart, ggml_graph_n_nodes(vGf), vBackendCpu);

    // Each branch produced zeros for the experts of the other one.
    vExperts = ggml_add(vCtx0, vDeviceExperts, vCpuExperts);
    mCb(vExperts, "ffn_moe_hybrid_merge", pIl);
    ggml_backend_sched_set_tensor_backend(vSched, vExperts, vDeviceBackend);
  }

  vExperts = ggml_mul(vCtx0, vExperts, vWeights);
  mCb(vExperts, "ffn_moe_weighted", pIl);

  ggml_build_forward_expand(vGf, vExperts);

  ggml_tensor *aCurExperts[MGS_MAX_EXPERTS] = {nullptr};

  assert(pNExpertUsed > 0);

  // order the views before the adds
  for (uint32_t vI = 0; vI < vHparams.vNExpertUsed; ++vI) {
    aCurExperts[vI] = ggml_view_2d(vCtx0, vExperts, cNEmbd, cNTokens, vExperts->nb[2], vI * vExperts->nb[1]);

    ggml_build_forward_expand(vGf, aCurExperts[vI]);
  }

  // aggregate experts
  ggml_tensor *vMoeOut = aCurExperts[0];

  for (uint32_t vI = 1; vI < vHparams.vNExpertUsed; ++vI) {
    vMoeOut = ggml_add(vCtx0, vMoeOut, aCurExperts[vI]);

    ggml_build_forward_expand(vGf, vMoeOut);
  }

  if (vHparams.vNExpertUsed == 1) {
    // avoid returning a non-contiguous tensor
    vMoeOut = ggml_cont(vCtx0, vMoeOut);
  }

  mCb(vMoeOut, "ffn_moe_out", pIl);

  return vMoeOut;
}

// input embeddings with optional lora
ggml_tensor *llm_graph_context::mBuildInpEmbd(ggml_tensor *pTokEmbd) const {
  const int64_t cNEmbdInp = vHparams.mNEmbdInp();
  const int64_t cNEmbd = vHparams.vNEmbd;

  assert(cNEmbdInp >= cNEmbd);

  auto vInp = std::make_unique<llm_graph_input_embd>(cNEmbdInp);

  vInp->vTokens = ggml_new_tensor_1d(vCtx0, GGML_TYPE_I32, vUbatch.vNTokens);
  mCb(vInp->vTokens, "inp_tokens", -1);
  ggml_set_input(vInp->vTokens);
  vRes->vTInpTokens = vInp->vTokens;

  vInp->vEmbd = ggml_new_tensor_2d(vCtx0, GGML_TYPE_F32, cNEmbdInp, vUbatch.vNTokens);
  mCb(vInp->vEmbd, "inp_embd", -1);
  ggml_set_input(vInp->vEmbd);

  // select one of the 2 inputs, based on the batch contents
  std::array<ggml_tensor *, 2> aInps;

  // token embeddings path (ubatch.token != nullptr)
  {
    auto &vCur = aInps[0];

    vCur = ggml_get_rows(vCtx0, pTokEmbd, vInp->vTokens);

    // apply lora for embedding tokens if needed
    for (const auto &vLora : *vLoras) {
      mgs_adapter_lora_weight *vLw = vLora.first->mGetWeight(pTokEmbd);
      if (vLw == nullptr) {
        continue;
      }

      const float cAdapterScale = vLora.second;
      const float cScale = vLw->mGetScale(vLora.first->vAlpha, cAdapterScale);

      ggml_tensor *vInpLDelta = ggml_scale(vCtx0,
                                           ggml_mul_mat(vCtx0, vLw->vB, // non-transposed lora_b
                                                        ggml_get_rows(vCtx0, vLw->vA, vInp->vTokens)),
                                           cScale);

      vCur = ggml_add(vCtx0, vCur, vInpLDelta);
    }

    if (cNEmbdInp != cNEmbd) {
      vCur = ggml_pad(vCtx0, vCur, vHparams.mNEmbdInp() - cNEmbd, 0, 0, 0);
    }
  }

  // vector embeddings path (ubatch.embd != nullptr)
  {
    auto &vCur = aInps[1];

    vCur = vInp->vEmbd;
  }

  assert(ggml_are_same_shape(aInps[0], aInps[1]));
  assert(ggml_are_same_stride(aInps[0], aInps[1]));

  ggml_tensor *vCur = ggml_build_forward_select(vGf, aInps.data(), aInps.size(), vUbatch.vToken ? 0 : 1);

  if (cNEmbdInp != cNEmbd) {
    vCur = ggml_view_2d(vCtx0, vCur, cNEmbd, cNTokens, vCur->nb[1], 0);
  }

  vRes->vTInpEmbd = vCur;

  // For Granite architecture
  // NOTE: For deepstack models, only apply scale to token inputs (ie text-only input).
  //  Raw embeddings are assumed to be multimodal inputs that should not be scaled.
  if (vHparams.vFEmbeddingScale != 0.0f && (vUbatch.vToken || vHparams.vNDeepstackLayers == 0)) {
    if (!ggml_is_contiguous(vCur)) {
      vCur = ggml_cont(vCtx0, vCur);
    }
    vCur = ggml_scale(vCtx0, vCur, vHparams.vFEmbeddingScale);
  }

  mCb(vCur, "embd", -1);

  vRes->mAddInput(std::move(vInp));

  // make sure the produced embeddings are immediately materialized in the ggml graph
  ggml_build_forward_expand(vGf, vCur);

  return vCur;
}

ggml_tensor *llm_graph_context::mBuildInpPos() const {
  auto vInp = std::make_unique<llm_graph_input_pos>(vHparams.mNPosPerEmbd());

  auto &vCur = vInp->vPos;

  vCur = ggml_new_tensor_1d(vCtx0, GGML_TYPE_I32, (int64_t)cNTokens * vHparams.mNPosPerEmbd());
  ggml_set_input(vCur);

  vRes->mAddInput(std::move(vInp));

  return vCur;
}

ggml_tensor *llm_graph_context::mBuildInpAttnScale() const {
  auto vInp = std::make_unique<llm_graph_input_attn_temp>(vHparams.vNAttnTempFloorScale, vHparams.vFAttnTempScale,
                                                          vHparams.vFAttnTempOffset);

  auto &vCur = vInp->vAttnScale;

  // this need to be 1x1xN for broadcasting
  vCur = ggml_new_tensor_3d(vCtx0, GGML_TYPE_F32, 1, 1, cNTokens);
  ggml_set_input(vCur);
  ggml_set_name(vCur, "attn_scale");

  vRes->mAddInput(std::move(vInp));

  return vCur;
}

ggml_tensor *llm_graph_context::mBuildInpOutIds() const {
  // note: when all tokens are output, we could skip this optimization to spare the ggml_get_rows() calls,
  //       but this would make the graph topology depend on the number of output tokens, which can interfere with
  //       features that require constant topology such as pipeline parallelism
  // if (cNOutputs < cNTokens) {
  //    return nullptr;
  //}

  auto vInp = std::make_unique<llm_graph_input_out_ids>(vHparams, vCparams, cNOutputs);

  auto &vCur = vInp->vOutIds;

  vCur = ggml_new_tensor_1d(vCtx0, GGML_TYPE_I32, cNOutputs);
  ggml_set_input(vCur);

  vRes->mAddInput(std::move(vInp));

  return vCur;
}

ggml_tensor *llm_graph_context::mBuildInpMean() const {
  auto vInp = std::make_unique<llm_graph_input_mean>(vCparams);

  auto &vCur = vInp->vMean;

  vCur = ggml_new_tensor_2d(vCtx0, GGML_TYPE_F32, cNTokens, vUbatch.vNSeqsUnq);
  ggml_set_input(vCur);

  vRes->mAddInput(std::move(vInp));

  return vCur;
}

ggml_tensor *llm_graph_context::mBuildInpCls() const {
  auto vInp = std::make_unique<llm_graph_input_cls>(vCparams, cArch);

  auto &vCur = vInp->vCls;

  vCur = ggml_new_tensor_1d(vCtx0, GGML_TYPE_I32, vUbatch.vNSeqsUnq);
  ggml_set_input(vCur);

  vRes->mAddInput(std::move(vInp));

  return vCur;
}

ggml_tensor *llm_graph_context::mBuildInpCrossEmbd() const {
  auto vInp = std::make_unique<llm_graph_input_cross_embd>(vCross);

  auto &vCur = vInp->vCrossEmbd;

  // if we have the output embeddings from the encoder, use them directly
  // TODO: needs more work to be correct, for now just use the tensor shape
  // if (vCross->vTEmbd) {
  //    vCur = ggml_view_tensor(vCtx0, vCross->vTEmbd);

  //    return vCur;
  //}

  const auto cNEmbd = !vCross->lVEmbd.empty() ? vCross->vNEmbd : vHparams.mNEmbdInp();
  const auto cNEnc = !vCross->lVEmbd.empty() ? vCross->vNEnc : vHparams.vNCtxTrain;

  vCur = ggml_new_tensor_2d(vCtx0, GGML_TYPE_F32, cNEmbd, cNEnc);
  ggml_set_input(vCur);

  vRes->mAddInput(std::move(vInp));

  return vCur;
}

ggml_tensor *llm_graph_context::mBuildInpPosBucketEnc() const {
  auto vInp = std::make_unique<llm_graph_input_pos_bucket>(vHparams);

  auto &vCur = vInp->vPosBucket;

  vCur = ggml_new_tensor_2d(vCtx0, GGML_TYPE_I32, cNTokens, cNTokens);
  ggml_set_input(vCur);

  vRes->mAddInput(std::move(vInp));

  return vCur;
}

ggml_tensor *llm_graph_context::mBuildInpPosBucketDec() const {
  const auto *vMctxCur = static_cast<const mgs_kv_cache_context *>(vMctx);

  auto vInp = std::make_unique<llm_graph_input_pos_bucket_kv>(vHparams, vMctxCur);

  const auto cNKv = vMctxCur->mGetNKv();

  auto &vCur = vInp->vPosBucket;

  vCur = ggml_new_tensor_2d(vCtx0, GGML_TYPE_I32, cNKv, cNTokens);
  ggml_set_input(vCur);

  vRes->mAddInput(std::move(vInp));

  return vCur;
}

ggml_tensor *llm_graph_context::mBuildPosBias(ggml_tensor *pPosBucket, ggml_tensor *pAttnRelB) const {
  ggml_tensor *vPosBucket1d = ggml_reshape_1d(vCtx0, pPosBucket, pPosBucket->ne[0] * pPosBucket->ne[1]);
  mCb(vPosBucket1d, "pos_bucket_1d", -1);

  ggml_tensor *vPosBias = ggml_get_rows(vCtx0, pAttnRelB, vPosBucket1d);

  vPosBias = ggml_reshape_3d(vCtx0, vPosBias, vPosBias->ne[0], pPosBucket->ne[0], pPosBucket->ne[1]);
  vPosBias = ggml_permute(vCtx0, vPosBias, 2, 0, 1, 3);
  vPosBias = ggml_cont(vCtx0, vPosBias);

  mCb(vPosBias, "pos_bias", -1);

  return vPosBias;
}

ggml_tensor *llm_graph_context::mBuildAttnMha(ggml_tensor *pQ, ggml_tensor *pK, ggml_tensor *pV, ggml_tensor *pKqB,
                                              ggml_tensor *pKqMask, ggml_tensor *pSinks, ggml_tensor *pVMla,
                                              float pKqScale, int pIl) const {
  const bool cVTrans = pV->nb[1] > pV->nb[2];

  // split the batch into streams if needed
  const auto cNStream = pK->ne[3];

  // the stream dim steps over one stream's worth of dim-2 (tokens): (ne[2]/n_stream) rows of stride nb[2].
  // q->nb[3]/n_stream only equals that for a contiguous q; nope-only MLA (glm5-next) passes a permuted
  // q_absorbed where nb[3] != ne[2]*nb[2], so using nb[3] would read another head's queries for streams s >= 1.
  pQ = ggml_view_4d(vCtx0, pQ, pQ->ne[0], pQ->ne[1], pQ->ne[2] / cNStream, cNStream, pQ->nb[1], pQ->nb[2],
                    pQ->nb[2] * (pQ->ne[2] / cNStream), 0);

  pQ = ggml_permute(vCtx0, pQ, 0, 2, 1, 3);
  pK = ggml_permute(vCtx0, pK, 0, 2, 1, 3);
  pV = ggml_permute(vCtx0, pV, 0, 2, 1, 3);

  ggml_tensor *vCur;

  const bool cUseFlashAttn = vCparams.vFlashAttn && pKqB == nullptr;
  if (cUseFlashAttn) {
    GGML_ASSERT(pKqB == nullptr && "Flash attention does not support KQ bias yet");

    if (cVTrans) {
      pV = ggml_transpose(vCtx0, pV);
    }

    // this can happen when KV cache is not used (e.g. an embedding model with non-causal attn)
    if (pK->type == GGML_TYPE_F32) {
      pK = ggml_cast(vCtx0, pK, GGML_TYPE_F16);
    }

    if (pV->type == GGML_TYPE_F32) {
      pV = ggml_cast(vCtx0, pV, GGML_TYPE_F16);
    }

    vCur = ggml_flash_attn_ext(vCtx0, pQ, pK, pV, pKqMask, pKqScale, vHparams.vFMaxAlibiBias,
                               vHparams.vAttnSoftCap ? vHparams.vFAttnLogitSoftcapping : 0.0f);
    vRes->mAddFusedNode({cLlmFusedOpFlashAttn, vCur, pIl});

    ggml_flash_attn_ext_add_sinks(vCur, pSinks);
    ggml_flash_attn_ext_set_prec(vCur, GGML_PREC_F32);

    if (pVMla) {
#if 0
            // v_mla can be applied as a matrix-vector multiplication with broadcasting across dimension 3 == n_tokens.
            // However, the code is optimized for dimensions 0 and 1 being large, so this is inefficient.
            vCur = ggml_reshape_4d(vCtx0, vCur, pVMla->ne[0], 1, cNHead, cNTokens);
            vCur = ggml_mul_mat(vCtx0, pVMla, vCur);
#else
      // It's preferable to do the calculation as a matrix-matrix multiplication with n_tokens in dimension 1.
      // The permutations are noops and only change how the tensor data is interpreted.
      vCur = ggml_permute(vCtx0, vCur, 0, 2, 1, 3);
      vCur = ggml_mul_mat(vCtx0, pVMla, vCur);
      mCb(vCur, "fattn_mla", pIl);
      vCur = ggml_permute(vCtx0, vCur, 0, 2, 1, 3);
      vCur = ggml_cont(vCtx0, vCur); // Needed because ggml_reshape_2d expects contiguous inputs.
#endif
    }

    vCur = ggml_reshape_2d(vCtx0, vCur, vCur->ne[0] * vCur->ne[1], vCur->ne[2] * vCur->ne[3]);
  } else {
    ggml_tensor *vKq = ggml_mul_mat(vCtx0, pK, pQ);
    mCb(vKq, "kq", pIl);

    // note: this op tends to require high floating point range
    //       while for some models F16 is enough, for others it is not, so we default to F32 here
    ggml_mul_mat_set_prec(vKq, GGML_PREC_F32);

    if (cArch == cLlmArchGrok) {
      // need to do the following:
      // multiply by attn_output_multiplier
      // and then :
      // kq = 30 * tanh(kq / 30)
      // before the softmax below

      vKq = ggml_tanh(vCtx0, ggml_scale(vCtx0, vKq, vHparams.vFAttnOutScale / vHparams.vFAttnLogitSoftcapping));
      mCb(vKq, "kq_tanh", pIl);
      vKq = ggml_scale(vCtx0, vKq, vHparams.vFAttnLogitSoftcapping);
      mCb(vKq, "kq_scaled", pIl);
    }

    if (vHparams.vAttnSoftCap) {
      vKq = ggml_scale(vCtx0, vKq, 1.0f / vHparams.vFAttnLogitSoftcapping);
      mCb(vKq, "kq_scaled_1", pIl);
      vKq = ggml_tanh(vCtx0, vKq);
      mCb(vKq, "kq_tanh", pIl);
      vKq = ggml_scale(vCtx0, vKq, vHparams.vFAttnLogitSoftcapping);
      mCb(vKq, "kq_scaled_2", pIl);
    }

    if (pKqB) {
      vKq = ggml_add(vCtx0, vKq, pKqB);
      mCb(vKq, "kq_plus_kq_b", pIl);
    }

    vKq = ggml_soft_max_ext(vCtx0, vKq, pKqMask, pKqScale, vHparams.vFMaxAlibiBias);
    ggml_soft_max_add_sinks(vKq, pSinks);
    mCb(vKq, "kq_soft_max", pIl);

    if (!cVTrans) {
      // note: avoid this branch
      pV = ggml_cont(vCtx0, ggml_transpose(vCtx0, pV));
      mCb(pV, "v_cont", pIl);
    }

    ggml_tensor *vKqv = ggml_mul_mat(vCtx0, pV, vKq);
    mCb(vKqv, "kqv", pIl);

    // for MLA with the absorption optimization, we need to "decompress" from MQA back to MHA
    if (pVMla) {
      vKqv = ggml_mul_mat(vCtx0, pVMla, vKqv);
      mCb(vKqv, "kqv_mla", pIl);
    }

    vCur = ggml_permute(vCtx0, vKqv, 0, 2, 1, 3);

    // recombine streams
    vCur = ggml_cont_2d(vCtx0, vCur, vCur->ne[0] * vCur->ne[1], vCur->ne[2] * vCur->ne[3]);

    if (!vCparams.vOffloadKqv) {
      // all nodes between the KV store and the attention output are run on the CPU
      ggml_backend_sched_set_tensor_backend(vSched, vCur, vBackendCpu);
    }
  }

  ggml_build_forward_expand(vGf, vCur);

  return vCur;
}

llm_graph_input_attn_no_cache *llm_graph_context::mBuildAttnInpNoCache() const {
  auto vInp = std::make_unique<llm_graph_input_attn_no_cache>(vHparams, vCparams);

  // flash attention requires an f16 mask
  const auto cTypeMask = vCparams.vFlashAttn ? GGML_TYPE_F16 : GGML_TYPE_F32;

  // note: there is no KV cache, so the number of KV values is equal to the number of tokens in the batch
  vInp->vSelfKqMask = ggml_new_tensor_4d(vCtx0, cTypeMask, cNTokens, cNTokens, 1, 1);
  ggml_set_input(vInp->vSelfKqMask);

  vInp->vSelfKqMaskCnv = vInp->vSelfKqMask;

  if (vHparams.vSwaType != cMgsSwaTypeNone) {
    vInp->vSelfKqMaskSwa = ggml_new_tensor_4d(vCtx0, cTypeMask, cNTokens, cNTokens, 1, 1);
    ggml_set_input(vInp->vSelfKqMaskSwa);

    vInp->vSelfKqMaskSwaCnv = vInp->vSelfKqMaskSwa;
  } else {
    vInp->vSelfKqMaskSwa = nullptr;
    vInp->vSelfKqMaskSwaCnv = nullptr;
  }

  return (llm_graph_input_attn_no_cache *)vRes->mAddInput(std::move(vInp));
}

ggml_tensor *llm_graph_context::mBuildAttn(llm_graph_input_attn_no_cache *pInp, ggml_tensor *pWo, ggml_tensor *pWoB,
                                           ggml_tensor *pWoS, ggml_tensor *pQCur, ggml_tensor *pKCur,
                                           ggml_tensor *pVCur, ggml_tensor *pKqB, ggml_tensor *pSinks,
                                           ggml_tensor *pVMla, float pKqScale, int pIl) const {
  GGML_UNUSED(cNTokens);

  // these nodes are added to the graph together so that they are not reordered
  // by doing so, the number of splits in the graph is reduced
  ggml_build_forward_expand(vGf, pQCur);
  ggml_build_forward_expand(vGf, pKCur);
  ggml_build_forward_expand(vGf, pVCur);

  const bool cIsSwa = vHparams.mIsSwa(pIl);

  const auto &vKqMask = cIsSwa ? pInp->mGetKqMaskSwa() : pInp->mGetKqMask();

  // [TAG_NO_CACHE_PAD]
  // TODO: if vUbatch.mEqualSeqs() == true, we can split the three tensors below into vUbatch.vNSeqsUnq streams
  //       but it might not be worth it
  // assert(!vUbatch.mEqualSeqs() || (pKCur->ne[3] == 1 && pKCur->ne[3] == vUbatch.vNSeqsUnq));

  ggml_tensor *vQ = pQCur;
  ggml_tensor *vK = pKCur;
  ggml_tensor *vV = pVCur;

  ggml_tensor *vCur = mBuildAttnMha(vQ, vK, vV, pKqB, vKqMask, pSinks, pVMla, pKqScale, pIl);
  mCb(vCur, "kqv_out", pIl);

  if (pWo) {
    vCur = mBuildLoraMm(pWo, vCur, pWoS);
  }

  if (pWoB) {
    // mCb(vCur, "kqv_wo", pIl);
  }

  if (pWoB) {
    vCur = ggml_add(vCtx0, vCur, pWoB);
  }

  return vCur;
}

static std::unique_ptr<llm_graph_input_attn_kv> fBuildAttnInpKvImpl(ggml_context *pCtx0, const mgs_ubatch &pUbatch,
                                                                    const mgs_hparams &pHparams,
                                                                    const mgs_cparams &pCparams,
                                                                    const mgs_kv_cache_context *pMctxCur) {

  auto vInp = std::make_unique<llm_graph_input_attn_kv>(pHparams, pCparams, pMctxCur);

  {
    GGML_ASSERT(pHparams.vSwaType == cMgsSwaTypeNone && "Use mgs_kv_cache_iswa for SWA");

    vInp->vSelfKIdxs = pMctxCur->mBuildInputKIdxs(pCtx0, pUbatch);
    vInp->vSelfVIdxs = pMctxCur->mBuildInputVIdxs(pCtx0, pUbatch);

    vInp->vSelfKqMask = fBuildAttnInpKqMask(pCtx0, pMctxCur, pUbatch, pCparams);
    vInp->vSelfKqMaskCnv = vInp->vSelfKqMask;
  }

  vInp->vSelfKRot = pMctxCur->mBuildInputKRot(pCtx0);
  vInp->vSelfVRot = pMctxCur->mBuildInputVRot(pCtx0);

  return vInp;
}

llm_graph_input_attn_kv *llm_graph_context::mBuildAttnInpKv() const {
  const auto *vMctxCur = static_cast<const mgs_kv_cache_context *>(vMctx);

  auto vInp = fBuildAttnInpKvImpl(vCtx0, vUbatch, vHparams, vCparams, vMctxCur);

  return (llm_graph_input_attn_kv *)vRes->mAddInput(std::move(vInp));
}

ggml_tensor *llm_graph_context::mBuildAttn(llm_graph_input_attn_kv *pInp, ggml_tensor *pWo, ggml_tensor *pWoB,
                                           ggml_tensor *pWoS, ggml_tensor *pQCur, ggml_tensor *pKCur,
                                           ggml_tensor *pVCur, ggml_tensor *pKqB, ggml_tensor *pSinks,
                                           ggml_tensor *pVMla, // TODO: remove
                                           float pKqScale, int pIl) const {
  GGML_ASSERT(pVMla == nullptr);

  if (pInp->vSelfKRot) {
    pQCur = mgs_mul_mat_hadamard(vCtx0, pQCur, pInp->vSelfKRot);
    pKCur = mgs_mul_mat_hadamard(vCtx0, pKCur, pInp->vSelfKRot);
  }

  if (pInp->vSelfVRot) {
    pVCur = mgs_mul_mat_hadamard(vCtx0, pVCur, pInp->vSelfVRot);
  }

  // these nodes are added to the graph together so that they are not reordered
  // by doing so, the number of splits in the graph is reduced
  // expand k later to enable rope fusion which directly writes into k-v cache
  ggml_build_forward_expand(vGf, pQCur);
  ggml_build_forward_expand(vGf, pVCur);
  ggml_build_forward_expand(vGf, pKCur);

  const auto *vMctxCur = pInp->vMctx;

  // store to KV cache
  {
    const auto &vKIdxs = pInp->mGetKIdxs();
    const auto &vVIdxs = pInp->mGetVIdxs();

    ggml_build_forward_expand(vGf, vMctxCur->mCpyK(vCtx0, pKCur, vKIdxs, pIl));
    ggml_build_forward_expand(vGf, vMctxCur->mCpyV(vCtx0, pVCur, vVIdxs, pIl));
  }

  ggml_tensor *vKqMask = pInp->mGetKqMask();

  ggml_tensor *vQ = pQCur;
  ggml_tensor *vK = vMctxCur->mGetK(vCtx0, pIl);
  ggml_tensor *vV = vMctxCur->mGetV(vCtx0, pIl);

  ggml_tensor *vCur = mBuildAttnMha(vQ, vK, vV, pKqB, vKqMask, pSinks, pVMla, pKqScale, pIl);
  mCb(vCur, "kqv_out", pIl);

  if (pInp->vSelfVRot) {
    vCur = mgs_mul_mat_hadamard(vCtx0, vCur, pInp->vSelfVRot);
  }

  if (pWo) {
    if (cArch == cLlmArchGlm4 || cArch == cLlmArchGlm4Moe || cArch == cLlmArchJais2) {
      // GLM4, GLM4_MOE, and JAIS2 seem to have numerical issues with half-precision accumulators
      vCur = mBuildLoraMm(pWo, vCur);
      ggml_mul_mat_set_prec(vCur, GGML_PREC_F32);
      if (pWoS) {
        vCur = ggml_mul(vCtx0, vCur, pWoS);
      }
    } else {
      vCur = mBuildLoraMm(pWo, vCur, pWoS);
    }
  }

  if (pWoB) {
    vCur = ggml_add(vCtx0, vCur, pWoB);
  }

  return vCur;
}

static std::unique_ptr<llm_graph_input_attn_k> fBuildAttnInpKImpl(ggml_context *pCtx0, const mgs_ubatch &pUbatch,
                                                                  const mgs_hparams &pHparams,
                                                                  const mgs_cparams &pCparams,
                                                                  const mgs_kv_cache_context *pMctxCur) {

  auto vInp = std::make_unique<llm_graph_input_attn_k>(pHparams, pCparams, pMctxCur);

  {
    GGML_ASSERT(pHparams.vSwaType == cMgsSwaTypeNone && "Use mgs_kv_cache_iswa for SWA");

    vInp->vSelfKIdxs = pMctxCur->mBuildInputKIdxs(pCtx0, pUbatch);

    vInp->vSelfKqMask = fBuildAttnInpKqMask(pCtx0, pMctxCur, pUbatch, pCparams);
    vInp->vSelfKqMaskCnv = vInp->vSelfKqMask;
  }

  return vInp;
}

llm_graph_input_attn_k *llm_graph_context::mBuildAttnInpK() const {
  const auto *vMctxCur = static_cast<const mgs_kv_cache_context *>(vMctx);

  auto vInp = fBuildAttnInpKImpl(vCtx0, vUbatch, vHparams, vCparams, vMctxCur);

  return (llm_graph_input_attn_k *)vRes->mAddInput(std::move(vInp));
}

ggml_tensor *llm_graph_context::mBuildAttn(llm_graph_input_attn_k *pInp, ggml_tensor *pWo, ggml_tensor *pWoB,
                                           ggml_tensor *pWoS, ggml_tensor *pQCur, ggml_tensor *pKCur,
                                           ggml_tensor *pVCur, ggml_tensor *pKqB, ggml_tensor *pSinks,
                                           ggml_tensor *pVMla, float pKqScale, int pIl) const {
  // these nodes are added to the graph together so that they are not reordered
  // by doing so, the number of splits in the graph is reduced
  // expand k later to enable rope fusion which directly writes into k-v cache
  ggml_build_forward_expand(vGf, pQCur);
  ggml_build_forward_expand(vGf, pVCur);
  ggml_build_forward_expand(vGf, pKCur);

  const auto *vMctxCur = pInp->vMctx;

  // store to KV cache
  {
    const auto &vKIdxs = pInp->mGetKIdxs();

    ggml_build_forward_expand(vGf, vMctxCur->mCpyK(vCtx0, pKCur, vKIdxs, pIl));
  }

  const auto &vKqMask = pInp->mGetKqMask();

  ggml_tensor *vQ = pQCur;
  ggml_tensor *vK = vMctxCur->mGetK(vCtx0, pIl);
  ggml_tensor *vV =
      ggml_view_4d(vCtx0, vK, pVCur->ne[0], vK->ne[1], vK->ne[2], vK->ne[3], vK->nb[1], vK->nb[2], vK->nb[3], 0);

  ggml_tensor *vCur = mBuildAttnMha(vQ, vK, vV, pKqB, vKqMask, pSinks, pVMla, pKqScale, pIl);
  mCb(vCur, "kqv_out", pIl);

  if (pWo) {
    if (cArch == cLlmArchGlm4 || cArch == cLlmArchGlm4Moe) {
      // GLM4 and GLM4_MOE seem to have numerical issues with half-precision accumulators
      vCur = mBuildLoraMm(pWo, vCur);
      ggml_mul_mat_set_prec(vCur, GGML_PREC_F32);
      if (pWoS) {
        vCur = ggml_mul(vCtx0, vCur, pWoS);
      }
    } else {
      vCur = mBuildLoraMm(pWo, vCur, pWoS);
    }
  }

  if (pWoB) {
    vCur = ggml_add(vCtx0, vCur, pWoB);
  }

  return vCur;
}

ggml_tensor *llm_graph_context::mBuildAttn(llm_graph_input_attn_k_dsa *pInp, ggml_tensor *pWo, ggml_tensor *pWoB,
                                           ggml_tensor *pWoS, ggml_tensor *pQCur, ggml_tensor *pKCur,
                                           ggml_tensor *pVCur, ggml_tensor *pKqB, ggml_tensor *pSinks,
                                           ggml_tensor *pVMla, ggml_tensor *pTopK, float pKqScale, int pIl) const {
  // these nodes are added to the graph together so that they are not reordered
  // by doing so, the number of splits in the graph is reduced
  // expand k later to enable rope fusion which directly writes into k-v cache
  ggml_build_forward_expand(vGf, pQCur);
  ggml_build_forward_expand(vGf, pVCur);
  ggml_build_forward_expand(vGf, pKCur);

  const auto *vMctxCur = pInp->vMctx->mGetMla();

  // store to KV cache
  {
    const auto &vKIdxs = pInp->mGetKIdxsMla();

    ggml_build_forward_expand(vGf, vMctxCur->mCpyK(vCtx0, pKCur, vKIdxs, pIl));
  }

  const auto &vKqMask = pInp->mGetKqMaskMla();

  // prepare new kq mask - starts filled with -INFINITY
  ggml_tensor *vKqMaskAll = ggml_fill(vCtx0, vKqMask, -INFINITY);

  // reshape KQ mask into tensor with rows of size 1:
  // [n_kv, n_batch, 1, n_stream] -> [1, n_kv, n_batch, n_stream]
  vKqMaskAll = ggml_view_4d(vCtx0, vKqMaskAll, 1, vKqMaskAll->ne[0], vKqMaskAll->ne[1], vKqMaskAll->ne[3],
                            vKqMaskAll->nb[0], vKqMaskAll->nb[1], vKqMaskAll->nb[2], 0);

  // reshape top_k indices: [n_top_k, n_batch, 1, n_stream] -> [n_top_k, n_batch, n_stream, 1]
  ggml_tensor *vTopK3d = ggml_view_4d(vCtx0, pTopK, pTopK->ne[0], pTopK->ne[1], pTopK->ne[3], 1, pTopK->nb[1],
                                      pTopK->nb[2], pTopK->ne[3] * pTopK->nb[3], 0);

  // prepare zero-filled tensor with rows of size 1: [1, n_top_k, n_batch, n_stream]
  // this will be our source of zero values for unmasking top k mask elements
  ggml_tensor *vZeros = ggml_new_tensor_4d(vCtx0, GGML_TYPE_F32, 1, vTopK3d->ne[0], vTopK3d->ne[1], vTopK3d->ne[2]);
  vZeros = ggml_fill(vCtx0, vZeros, 0.0f);

  // modify KQ mask by unmasking elements that are in top_k indices
  // ggml_set_rows([1, n_kv, n_batch, n_stream], [1, n_top_k, n_batch, n_stream], [n_top_k, n_batch, n_stream, 1])
  ggml_tensor *vKqMaskTopK = ggml_set_rows(vCtx0, vKqMaskAll, vZeros, vTopK3d);

  // reshape to restore the original shape of KQ mask:
  // [1, n_kv, n_batch, n_stream] -> [n_kv, n_batch, 1, n_stream]
  vKqMaskTopK = ggml_view_4d(vCtx0, vKqMaskTopK, vKqMaskTopK->ne[1], vKqMaskTopK->ne[2], 1, vKqMaskTopK->ne[3],
                             vKqMaskTopK->nb[2], vKqMaskTopK->nb[3], vKqMaskTopK->nb[3], 0);

  // combine with the original kq mask
  vKqMaskTopK = ggml_add(vCtx0, vKqMaskTopK, vKqMask);

  ggml_tensor *vQ = pQCur;
  ggml_tensor *vK = vMctxCur->mGetK(vCtx0, pIl);
  ggml_tensor *vV =
      ggml_view_4d(vCtx0, vK, pVCur->ne[0], vK->ne[1], vK->ne[2], vK->ne[3], vK->nb[1], vK->nb[2], vK->nb[3], 0);

  ggml_tensor *vCur = mBuildAttnMha(vQ, vK, vV, pKqB, vKqMaskTopK, pSinks, pVMla, pKqScale, pIl);
  mCb(vCur, "kqv_out", pIl);

  if (pWo) {
    vCur = mBuildLoraMm(pWo, vCur, pWoS);
  }

  if (pWoB) {
    vCur = ggml_add(vCtx0, vCur, pWoB);
  }

  return vCur;
}

ggml_tensor *llm_graph_context::mBuildAttn(llm_graph_input_attn_kv_iswa *pInp, ggml_tensor *pWo, ggml_tensor *pWoB,
                                           ggml_tensor *pWoS, ggml_tensor *pQCur, ggml_tensor *pKCur,
                                           ggml_tensor *pVCur, ggml_tensor *pKqB, ggml_tensor *pSinks,
                                           ggml_tensor *pVMla, float pKqScale, int pIl) const {
  const bool cIsSwa = vHparams.mIsSwa(pIl);

  auto *vKRot = cIsSwa ? pInp->vSelfKRotSwa : pInp->vSelfKRot;
  auto *vVRot = cIsSwa ? pInp->vSelfVRotSwa : pInp->vSelfVRot;

  if (vKRot) {
    pQCur = mgs_mul_mat_hadamard(vCtx0, pQCur, vKRot);
    if (pKCur) {
      pKCur = mgs_mul_mat_hadamard(vCtx0, pKCur, vKRot);
    }
  }
  if (vVRot) {
    if (pVCur) {
      pVCur = mgs_mul_mat_hadamard(vCtx0, pVCur, vVRot);
    }
  }

  // these nodes are added to the graph together so that they are not reordered
  // by doing so, the number of splits in the graph is reduced
  ggml_build_forward_expand(vGf, pQCur);

  if (pKCur) {
    ggml_build_forward_expand(vGf, pKCur);
  }

  if (pVCur) {
    ggml_build_forward_expand(vGf, pVCur);
  }

  const auto *vMctxIswa = pInp->vMctx;

  const auto *vMctxCur = cIsSwa ? vMctxIswa->mGetSwa() : vMctxIswa->mGetBase();

  // optionally store to KV cache
  if (pKCur) {
    const auto &vKIdxs = cIsSwa ? pInp->mGetKIdxsSwa() : pInp->mGetKIdxs();

    ggml_build_forward_expand(vGf, vMctxCur->mCpyK(vCtx0, pKCur, vKIdxs, pIl));
  }

  if (pVCur) {
    const auto &vVIdxs = cIsSwa ? pInp->mGetVIdxsSwa() : pInp->mGetVIdxs();

    ggml_build_forward_expand(vGf, vMctxCur->mCpyV(vCtx0, pVCur, vVIdxs, pIl));
  }

  const auto &vKqMask = cIsSwa ? pInp->mGetKqMaskSwa() : pInp->mGetKqMask();

  ggml_tensor *vQ = pQCur;
  ggml_tensor *vK = vMctxCur->mGetK(vCtx0, pIl);
  ggml_tensor *vV = vMctxCur->mGetV(vCtx0, pIl);

  ggml_tensor *vCur = mBuildAttnMha(vQ, vK, vV, pKqB, vKqMask, pSinks, pVMla, pKqScale, pIl);
  mCb(vCur, "kqv_out", pIl);

  if (vVRot) {
    vCur = mgs_mul_mat_hadamard(vCtx0, vCur, vVRot);
  }

  if (pWo) {
    vCur = mBuildLoraMm(pWo, vCur, pWoS);
  }

  if (pWoB) {
    // mCb(vCur, "kqv_wo", pIl);
  }

  if (pWoB) {
    vCur = ggml_add(vCtx0, vCur, pWoB);
  }

  return vCur;
}

ggml_tensor *llm_graph_context::mBuildAttn(llm_graph_input_attn_k_iswa *pInp, ggml_tensor *pWo, ggml_tensor *pWoB,
                                           ggml_tensor *pWoS, ggml_tensor *pQCur, ggml_tensor *pKCur,
                                           ggml_tensor *pVCur, ggml_tensor *pKqB, ggml_tensor *pSinks,
                                           ggml_tensor *pVMla, float pKqScale, int pIl) const {
  const bool cIsSwa = vHparams.mIsSwa(pIl);

  GGML_UNUSED(pVCur);

  auto *vKRot = cIsSwa ? pInp->vSelfKRotSwa : pInp->vSelfKRot;

  if (vKRot) {
    pQCur = mgs_mul_mat_hadamard(vCtx0, pQCur, vKRot);
    if (pKCur) {
      pKCur = mgs_mul_mat_hadamard(vCtx0, pKCur, vKRot);
    }
  }

  // these nodes are added to the graph together so that they are not reordered
  // by doing so, the number of splits in the graph is reduced
  ggml_build_forward_expand(vGf, pQCur);

  if (pKCur) {
    ggml_build_forward_expand(vGf, pKCur);
  }

  const auto *vMctxIswa = pInp->vMctx;
  const auto *vMctxCur = cIsSwa ? vMctxIswa->mGetSwa() : vMctxIswa->mGetBase();

  // optionally store to KV cache
  if (pKCur) {
    const auto &vKIdxs = cIsSwa ? pInp->mGetKIdxsSwa() : pInp->mGetKIdxs();

    ggml_build_forward_expand(vGf, vMctxCur->mCpyK(vCtx0, pKCur, vKIdxs, pIl));
  }

  const auto &vKqMask = cIsSwa ? pInp->mGetKqMaskSwa() : pInp->mGetKqMask();

  // MLA-style attention: the cached K is used as V
  ggml_tensor *vQ = pQCur;
  ggml_tensor *vK = vMctxCur->mGetK(vCtx0, pIl);
  ggml_tensor *vV = vK;

  ggml_tensor *vCur = mBuildAttnMha(vQ, vK, vV, pKqB, vKqMask, pSinks, pVMla, pKqScale, pIl);
  mCb(vCur, "kqv_out", pIl);

  if (vKRot) {
    vCur = mgs_mul_mat_hadamard(vCtx0, vCur, vKRot);
  }

  if (pWo) {
    vCur = mBuildLoraMm(pWo, vCur, pWoS);
  }

  if (pWoB) {
    vCur = ggml_add(vCtx0, vCur, pWoB);
  }

  return vCur;
}

llm_graph_input_attn_cross *llm_graph_context::mBuildAttnInpCross() const {
  auto vInp = std::make_unique<llm_graph_input_attn_cross>(vCross);

  const int32_t cNEnc = !vCross->lVEmbd.empty() ? vCross->vNEnc : vHparams.vNCtxTrain;

  // flash attention requires an f16 mask
  const auto cTypeMask = vCparams.vFlashAttn ? GGML_TYPE_F16 : GGML_TYPE_F32;

  vInp->vCrossKqMask = ggml_new_tensor_4d(vCtx0, cTypeMask, cNEnc, cNTokens, 1, 1);
  ggml_set_input(vInp->vCrossKqMask);

  vInp->vCrossKqMaskCnv = vInp->vCrossKqMask;

  return (llm_graph_input_attn_cross *)vRes->mAddInput(std::move(vInp));
}

ggml_tensor *llm_graph_context::mBuildAttn(llm_graph_input_attn_cross *pInp, ggml_tensor *pWo, ggml_tensor *pWoB,
                                           ggml_tensor *pWoS, ggml_tensor *pQCur, ggml_tensor *pKCur,
                                           ggml_tensor *pVCur, ggml_tensor *pKqB, ggml_tensor *pSinks,
                                           ggml_tensor *pVMla, float pKqScale, int pIl) const {
  // these nodes are added to the graph together so that they are not reordered
  // by doing so, the number of splits in the graph is reduced
  ggml_build_forward_expand(vGf, pQCur);
  ggml_build_forward_expand(vGf, pKCur);
  ggml_build_forward_expand(vGf, pVCur);

  const auto &vKqMask = pInp->mGetKqMaskCross();

  ggml_tensor *vQ = pQCur;
  ggml_tensor *vK = pKCur;
  ggml_tensor *vV = pVCur;

  ggml_tensor *vCur = mBuildAttnMha(vQ, vK, vV, pKqB, vKqMask, pSinks, pVMla, pKqScale, pIl);
  mCb(vCur, "kqv_out", pIl);

  if (pWo) {
    vCur = mBuildLoraMm(pWo, vCur, pWoS);
  }

  if (pWoB) {
    // mCb(vCur, "kqv_wo", pIl);
  }

  if (pWoB) {
    vCur = ggml_add(vCtx0, vCur, pWoB);
  }

  return vCur;
}

llm_graph_input_attn_k_dsa *llm_graph_context::mBuildAttnInpKDsa() const {
  const auto *vMctxCur = static_cast<const mgs_kv_cache_dsa_context *>(vMctx);

  auto vInp = std::make_unique<llm_graph_input_attn_k_dsa>(vHparams, vCparams, vMctxCur);

  {
    vInp->vSelfKIdxsMla = vMctxCur->mGetMla()->mBuildInputKIdxs(vCtx0, vUbatch);

    vInp->vSelfKqMaskMla = fBuildAttnInpKqMask(vCtx0, vMctxCur->mGetMla(), vUbatch, vCparams);
    vInp->vSelfKqMaskMlaCnv = vInp->vSelfKqMaskMla;
  }

  {
    vInp->vSelfKIdxsLid = vMctxCur->mGetLid()->mBuildInputKIdxs(vCtx0, vUbatch);

    // ensure that mask type matches fused lightning indexer use (requires f16 mask)
    auto vCparamsCopy = vCparams;
    vCparamsCopy.vFlashAttn = vCparams.vFusedLid;

    vInp->vSelfKqMaskLid = fBuildAttnInpKqMask(vCtx0, vMctxCur->mGetLid(), vUbatch, vCparamsCopy);
    vInp->vSelfKqMaskLidCnv = vInp->vSelfKqMaskLid;

    vInp->vSelfKRotLid = vMctxCur->mGetLid()->mBuildInputKRot(vCtx0);
  }

  return (llm_graph_input_attn_k_dsa *)vRes->mAddInput(std::move(vInp));
}

llm_graph_input_attn_kv_msa *llm_graph_context::mBuildAttnInpKvMsa(bool pMsaEnabled) const {
  const auto *vMctxCur = static_cast<const mgs_kv_cache_msa_context *>(vMctx);

  auto vInp = std::make_unique<llm_graph_input_attn_kv_msa>(vHparams, vCparams, vMctxCur);

  const auto *vMctxBase = vMctxCur->mGetBase();
  const auto *vMctxIdx = vMctxCur->mGetIdx();

  {
    GGML_ASSERT(vHparams.vSwaType == cMgsSwaTypeNone && "Use mgs_kv_cache_iswa for SWA");

    vInp->vSelfKIdxs = vMctxBase->mBuildInputKIdxs(vCtx0, vUbatch);
    vInp->vSelfVIdxs = vMctxBase->mBuildInputVIdxs(vCtx0, vUbatch);

    vInp->vSelfKqMask = fBuildAttnInpKqMask(vCtx0, vMctxBase, vUbatch, vCparams);
    vInp->vSelfKqMaskCnv = vInp->vSelfKqMask;
  }

  vInp->vSelfKRot = vMctxBase->mBuildInputKRot(vCtx0);
  vInp->vSelfVRot = vMctxBase->mBuildInputVRot(vCtx0);

  if (pMsaEnabled) {
    vInp->vSelfKIdxsIdx = vMctxIdx->mBuildInputKIdxs(vCtx0, vUbatch);
  }

  return (llm_graph_input_attn_kv_msa *)vRes->mAddInput(std::move(vInp));
}

// TODO: maybe separate the inner implementation into a separate function
//       like with the non-sliding window equivalent
//       once sliding-window hybrid caches are a thing.
llm_graph_input_attn_kv_iswa *llm_graph_context::mBuildAttnInpKvIswa() const {
  const auto *vMctxCur = static_cast<const mgs_kv_cache_iswa_context *>(vMctx);

  auto vInp = std::make_unique<llm_graph_input_attn_kv_iswa>(vHparams, vCparams, vMctxCur);

  {
    vInp->vSelfKIdxs = vMctxCur->mGetBase()->mBuildInputKIdxs(vCtx0, vUbatch);
    vInp->vSelfVIdxs = vMctxCur->mGetBase()->mBuildInputVIdxs(vCtx0, vUbatch);

    vInp->vSelfKqMask = fBuildAttnInpKqMask(vCtx0, vMctxCur->mGetBase(), vUbatch, vCparams);
    vInp->vSelfKqMaskCnv = vInp->vSelfKqMask;
  }

  {
    GGML_ASSERT(vHparams.vSwaType != cMgsSwaTypeNone && "Use mgs_kv_cache for non-SWA");

    vInp->vSelfKIdxsSwa = vMctxCur->mGetSwa()->mBuildInputKIdxs(vCtx0, vUbatch);
    vInp->vSelfVIdxsSwa = vMctxCur->mGetSwa()->mBuildInputVIdxs(vCtx0, vUbatch);

    vInp->vSelfKqMaskSwa = fBuildAttnInpKqMask(vCtx0, vMctxCur->mGetSwa(), vUbatch, vCparams);
    vInp->vSelfKqMaskSwaCnv = vInp->vSelfKqMaskSwa;
  }

  vInp->vSelfKRot = vMctxCur->mGetBase()->mBuildInputKRot(vCtx0);
  vInp->vSelfVRot = vMctxCur->mGetBase()->mBuildInputVRot(vCtx0);

  vInp->vSelfKRotSwa = vMctxCur->mGetSwa()->mBuildInputKRot(vCtx0);
  vInp->vSelfVRotSwa = vMctxCur->mGetSwa()->mBuildInputVRot(vCtx0);

  return (llm_graph_input_attn_kv_iswa *)vRes->mAddInput(std::move(vInp));
}

llm_graph_input_attn_k_iswa *llm_graph_context::mBuildAttnInpKIswa() const {
  const auto *vMctxCur = static_cast<const mgs_kv_cache_iswa_context *>(vMctx);

  auto vInp = std::make_unique<llm_graph_input_attn_k_iswa>(vHparams, vCparams, vMctxCur);

  {
    vInp->vSelfKIdxs = vMctxCur->mGetBase()->mBuildInputKIdxs(vCtx0, vUbatch);

    vInp->vSelfKqMask = fBuildAttnInpKqMask(vCtx0, vMctxCur->mGetBase(), vUbatch, vCparams);
    vInp->vSelfKqMaskCnv = vInp->vSelfKqMask;
  }

  {
    GGML_ASSERT(vHparams.vSwaType != cMgsSwaTypeNone && "Use mgs_kv_cache for non-SWA");

    vInp->vSelfKIdxsSwa = vMctxCur->mGetSwa()->mBuildInputKIdxs(vCtx0, vUbatch);

    vInp->vSelfKqMaskSwa = fBuildAttnInpKqMask(vCtx0, vMctxCur->mGetSwa(), vUbatch, vCparams);
    vInp->vSelfKqMaskSwaCnv = vInp->vSelfKqMaskSwa;
  }

  vInp->vSelfKRot = vMctxCur->mGetBase()->mBuildInputKRot(vCtx0);

  vInp->vSelfKRotSwa = vMctxCur->mGetSwa()->mBuildInputKRot(vCtx0);

  return (llm_graph_input_attn_k_iswa *)vRes->mAddInput(std::move(vInp));
}

ggml_tensor *llm_graph_context::mBuildRs(ggml_tensor *pS, ggml_tensor *pStateCopyMain, ggml_tensor *pStateCopyExtra,
                                         int32_t pStateSize, int32_t pNSeqs, uint32_t pNRs, uint32_t pRsHead,
                                         uint32_t pRsSize, int32_t pRsZero,
                                         const llm_graph_get_rows_fn &pGetStateRows) const {

  GGML_UNUSED(pRsSize);
  ggml_tensor *vStates = ggml_reshape_2d(vCtx0, pS, pStateSize, pS->ne[1]);

  // Clear a single state which will then be copied to the other cleared states.
  // Note that this is a no-op when the view is zero-sized.
  ggml_tensor *vStateZero =
      ggml_view_1d(vCtx0, vStates, pStateSize * (pRsZero >= 0), pRsZero * vStates->nb[1] * (pRsZero >= 0));
  ggml_build_forward_expand(vGf, ggml_scale_inplace(vCtx0, vStateZero, 0));

  // copy states
  // NOTE: assuming the copy destinations are ALL contained between rs_head and rs_head + n_rs
  // {state_size, rs_size} -> {state_size, n_seqs}
  ggml_tensor *vOutputStates = pGetStateRows(vCtx0, vStates, pStateCopyMain);
  ggml_build_forward_expand(vGf, vOutputStates);

  // copy extra states which won't be changed further (between n_seqs and n_rs)
  ggml_tensor *vStatesExtra = ggml_get_rows(vCtx0, vStates, pStateCopyExtra);
  ggml_build_forward_expand(
      vGf, ggml_cpy(vCtx0, vStatesExtra,
                    ggml_view_2d(vCtx0, pS, pStateSize, (pNRs - pNSeqs), pS->nb[1], (pRsHead + pNSeqs) * pS->nb[1])));

  return vOutputStates;
}

static std::unique_ptr<llm_graph_input_rs> fBuildRsInpImpl(ggml_context *pCtx0, const mgs_ubatch &pUbatch,
                                                           const mgs_memory_recurrent_context *pMctxCur) {

  auto vInp = std::make_unique<llm_graph_input_rs>(pMctxCur);

  const int64_t cNRs = pMctxCur->mGetNRs();
  const int64_t cNSeqs = pUbatch.vNSeqs;

  vInp->vSCopy = ggml_new_tensor_1d(pCtx0, GGML_TYPE_I32, cNRs);
  ggml_set_input(vInp->vSCopy);

  vInp->vSCopyMain = ggml_view_1d(pCtx0, vInp->vSCopy, cNSeqs, 0);
  vInp->vSCopyExtra = ggml_view_1d(pCtx0, vInp->vSCopy, cNRs - cNSeqs, cNSeqs * vInp->vSCopy->nb[0]);

  vInp->vHead = pMctxCur->mGetHead();
  vInp->vRsZ = pMctxCur->mGetRsZ();

  return vInp;
}

llm_graph_input_rs *llm_graph_context::mBuildRsInp() const {
  const auto *vMctxCur = static_cast<const mgs_memory_recurrent_context *>(vMctx);

  auto vInp = fBuildRsInpImpl(vCtx0, vUbatch, vMctxCur);

  return (llm_graph_input_rs *)vRes->mAddInput(std::move(vInp));
}

ggml_tensor *llm_graph_context::mBuildRs(llm_graph_input_rs *pInp, ggml_tensor *pS, int32_t pStateSize, int32_t pNSeqs,
                                         const llm_graph_get_rows_fn &pGetStateRows) const {
  const auto *vKvState = pInp->vMctx;

  return mBuildRs(pS, pInp->vSCopyMain, pInp->vSCopyExtra, pStateSize, pNSeqs, vKvState->mGetNRs(),
                  vKvState->mGetHead(), vKvState->mGetSize(), vKvState->mGetRsZ(), pGetStateRows);
}

ggml_tensor *llm_graph_context::mBuildRwkvTokenShiftLoad(llm_graph_input_rs *pInp, const mgs_ubatch &pUbatch,
                                                         int pIl) const {
  const auto *vMctxCur = static_cast<const mgs_memory_recurrent_context *>(vMctx);

  const auto cTokenShiftCount = vHparams.vTokenShiftCount;

  const int64_t cNSeqs = pUbatch.vNSeqs;

  ggml_tensor *vTokenShiftAll = vMctxCur->mGetRL(pIl);

  ggml_tensor *vTokenShift = mBuildRs(pInp, vTokenShiftAll, vHparams.mNEmbdR(), cNSeqs);

  vTokenShift = ggml_reshape_3d(vCtx0, vTokenShift, vHparams.vNEmbd, cTokenShiftCount, cNSeqs);

  return vTokenShift;
}

ggml_tensor *llm_graph_context::mBuildRwkvTokenShiftStore(ggml_tensor *pTokenShift, const mgs_ubatch &pUbatch,
                                                          int pIl) const {
  const auto *vMctxCur = static_cast<const mgs_memory_recurrent_context *>(vMctx);

  const auto cTokenShiftCount = vHparams.vTokenShiftCount;
  const auto cNEmbd = vHparams.vNEmbd;

  const int64_t cNSeqs = pUbatch.vNSeqs;

  const auto cKvHead = vMctxCur->mGetHead();

  return ggml_cpy(vCtx0, ggml_view_1d(vCtx0, pTokenShift, cNEmbd * cNSeqs * cTokenShiftCount, 0),
                  ggml_view_1d(vCtx0, vMctxCur->mGetRL(pIl), vHparams.mNEmbdR() * cNSeqs,
                               vHparams.mNEmbdR() * cKvHead * ggml_element_size(vMctxCur->mGetRL(pIl))));
}

llm_graph_input_mem_hybrid *llm_graph_context::mBuildInpMemHybrid() const {
  const auto *vMctxCur = static_cast<const mgs_memory_hybrid_context *>(vMctx);

  auto vInpRs = fBuildRsInpImpl(vCtx0, vUbatch, vMctxCur->mGetRecr());
  auto vInpAttn = fBuildAttnInpKvImpl(vCtx0, vUbatch, vHparams, vCparams, vMctxCur->mGetAttn());

  auto vInp = std::make_unique<llm_graph_input_mem_hybrid>(vCparams, std::move(vInpAttn), std::move(vInpRs), vMctxCur);

  return (llm_graph_input_mem_hybrid *)vRes->mAddInput(std::move(vInp));
}

llm_graph_input_mem_hybrid_k *llm_graph_context::mBuildInpMemHybridK() const {
  const auto *vMctxCur = static_cast<const mgs_memory_hybrid_context *>(vMctx);

  auto vInpRs = fBuildRsInpImpl(vCtx0, vUbatch, vMctxCur->mGetRecr());
  auto vInpAttn = fBuildAttnInpKImpl(vCtx0, vUbatch, vHparams, vCparams, vMctxCur->mGetAttn());

  auto vInp =
      std::make_unique<llm_graph_input_mem_hybrid_k>(vCparams, std::move(vInpAttn), std::move(vInpRs), vMctxCur);

  return (llm_graph_input_mem_hybrid_k *)vRes->mAddInput(std::move(vInp));
}

llm_graph_input_mem_hybrid_iswa *llm_graph_context::mBuildInpMemHybridIswa() const {
  const auto *vMctxCur = static_cast<const mgs_memory_hybrid_iswa_context *>(vMctx);

  auto vInpRs = fBuildRsInpImpl(vCtx0, vUbatch, vMctxCur->mGetRecr());

  // build iswa attention input
  const auto *vAttnCtx = vMctxCur->mGetAttn();

  auto vInpAttn = std::make_unique<llm_graph_input_attn_kv_iswa>(vHparams, vCparams, vAttnCtx);

  {
    vInpAttn->vSelfKIdxs = vAttnCtx->mGetBase()->mBuildInputKIdxs(vCtx0, vUbatch);
    vInpAttn->vSelfVIdxs = vAttnCtx->mGetBase()->mBuildInputVIdxs(vCtx0, vUbatch);

    vInpAttn->vSelfKqMask = fBuildAttnInpKqMask(vCtx0, vAttnCtx->mGetBase(), vUbatch, vCparams);
    vInpAttn->vSelfKqMaskCnv = vInpAttn->vSelfKqMask;
  }

  {
    vInpAttn->vSelfKIdxsSwa = vAttnCtx->mGetSwa()->mBuildInputKIdxs(vCtx0, vUbatch);
    vInpAttn->vSelfVIdxsSwa = vAttnCtx->mGetSwa()->mBuildInputVIdxs(vCtx0, vUbatch);

    vInpAttn->vSelfKqMaskSwa = fBuildAttnInpKqMask(vCtx0, vAttnCtx->mGetSwa(), vUbatch, vCparams);
    vInpAttn->vSelfKqMaskSwaCnv = vInpAttn->vSelfKqMaskSwa;
  }

  auto vInp =
      std::make_unique<llm_graph_input_mem_hybrid_iswa>(vCparams, std::move(vInpAttn), std::move(vInpRs), vMctxCur);

  return (llm_graph_input_mem_hybrid_iswa *)vRes->mAddInput(std::move(vInp));
}

void llm_graph_context::mBuildDenseOut(ggml_tensor *pDense2, ggml_tensor *pDense2B, ggml_tensor *pDense3) const {
  if (!vCparams.vEmbeddings || !(pDense2 || pDense2B || pDense3)) {
    return;
  }
  ggml_tensor *vCur = vRes->vTEmbdPooled != nullptr ? vRes->vTEmbdPooled : vRes->vTEmbd;
  GGML_ASSERT(vCur != nullptr && "missing t_embd_pooled/t_embd");

  if (pDense2) {
    vCur = ggml_mul_mat(vCtx0, pDense2, vCur);
  }
  if (pDense2B) {
    vCur = ggml_add(vCtx0, vCur, pDense2B);
  }
  if (pDense3) {
    vCur = ggml_mul_mat(vCtx0, pDense3, vCur);
  }
  mCb(vCur, "result_embd_pooled", -1);
  vRes->vTEmbdPooled = vCur;
  ggml_build_forward_expand(vGf, vCur);
}

void llm_graph_context::mBuildPooling(ggml_tensor *pCls, ggml_tensor *pClsB, ggml_tensor *pClsOut,
                                      ggml_tensor *pClsOutB, ggml_tensor *pClsNorm) const {
  if (!vCparams.vEmbeddings) {
    return;
  }

  ggml_tensor *vInp = vRes->vTEmbd;

  //// find result_norm tensor for input
  // for (int vI = ggml_graph_n_nodes(vGf) - 1; vI >= 0; --vI) {
  //     vInp = ggml_graph_node(vGf, vI);
  //     if (strcmp(vInp->name, "result_norm") == 0 || strcmp(vInp->name, "result_embd") == 0) {
  //         break;
  //     }

  //    vInp = nullptr;
  //}

  GGML_ASSERT(vInp != nullptr && "missing result_norm/result_embd tensor");

  ggml_tensor *vCur;

  switch (cPoolingType) {
  case MGS_POOLING_TYPE_NONE: {
    vCur = vInp;
  } break;
  case MGS_POOLING_TYPE_MEAN: {
    ggml_tensor *vInpMean = mBuildInpMean();
    vCur = ggml_mul_mat(vCtx0, ggml_cont(vCtx0, ggml_transpose(vCtx0, vInp)), vInpMean);
  } break;
  case MGS_POOLING_TYPE_CLS:
  case MGS_POOLING_TYPE_LAST: {
    ggml_tensor *vInpCls = mBuildInpCls();
    vCur = ggml_get_rows(vCtx0, vInp, vInpCls);
  } break;
  case MGS_POOLING_TYPE_RANK: {
    if (cArch == cLlmArchModernBert) {
      // modern bert gte reranker builds mean first then applies prediction head and classifier
      // https://github.com/huggingface/transformers/blob/main/src/transformers/models/modernbert/modular_modernbert.py#L1404-1411
      ggml_tensor *vInpMean = mBuildInpMean();
      vCur = ggml_mul_mat(vCtx0, ggml_cont(vCtx0, ggml_transpose(vCtx0, vInp)), vInpMean);
    } else {
      ggml_tensor *vInpCls = mBuildInpCls();
      vCur = ggml_get_rows(vCtx0, vInp, vInpCls);
    }

    // classification head
    // https://github.com/huggingface/transformers/blob/5af7d41e49bbfc8319f462eb45253dcb3863dfb7/src/transformers/models/roberta/modeling_roberta.py#L1566
    if (pCls) {
      vCur = ggml_mul_mat(vCtx0, pCls, vCur);
      if (pClsB) {
        vCur = ggml_add(vCtx0, vCur, pClsB);
      }
      if (cArch == cLlmArchModernBert) {
        vCur = ggml_gelu(vCtx0, vCur);
      } else {
        vCur = ggml_tanh(vCtx0, vCur);
      }
      if (pClsNorm) {
        // head norm
        vCur = mBuildNorm(vCur, pClsNorm, NULL, cLlmNorm, -1);
      }
    }

    // some models don't have `cls_out`, for example: https://huggingface.co/jinaai/jina-reranker-v1-tiny-en
    // https://huggingface.co/jinaai/jina-reranker-v1-tiny-en/blob/cb5347e43979c3084a890e3f99491952603ae1b7/modeling_bert.py#L884-L896
    // Single layer classification head (direct projection)
    // https://github.com/huggingface/transformers/blob/f4fc42216cd56ab6b68270bf80d811614d8d59e4/src/transformers/models/bert/modeling_bert.py#L1476
    if (pClsOut) {
      vCur = ggml_mul_mat(vCtx0, pClsOut, vCur);
      if (pClsOutB) {
        vCur = ggml_add(vCtx0, vCur, pClsOutB);
      }
    }

    // softmax for qwen3 reranker
    if (cArch == cLlmArchQwen3 || cArch == cLlmArchQwen3vl) {
      vCur = ggml_soft_max(vCtx0, vCur);
    }
  } break;
  default: {
    GGML_ABORT("unknown pooling type");
  }
  }

  mCb(vCur, "result_embd_pooled", -1);
  vRes->vTEmbdPooled = vCur;

  ggml_build_forward_expand(vGf, vCur);
}

void llm_graph_context::mBuildSampling() const {
  if (dSamplers.empty() || !vRes->vTLogits) {
    return;
  }

  std::array<ggml_tensor *, 2> aOuts;
  aOuts[0] = vRes->vTLogits;

  auto vInpSampling = std::make_unique<llm_graph_input_sampling>(dSamplers);
  vRes->mAddInput(std::move(vInpSampling));

  std::map<mgs_seq_id, int32_t> dSeqToLogitRow;
  int32_t vLogitRowIdx = 0;

  for (uint32_t vI = 0; vI < vUbatch.vNTokens; vI++) {
    if (vUbatch.vOutput[vI]) {
      mgs_seq_id vSeqId = vUbatch.vSeqId[vI][0];
      dSeqToLogitRow[vSeqId] = vLogitRowIdx;
      vLogitRowIdx++;
    }
  }

  // vRes->vTLogits will contain logits for all tokens that want the logits calculated (logits=1 or output=1)
  GGML_ASSERT(vRes->vTLogits != nullptr && "missing t_logits tensor");

  // add a dummy row of logits
  // this trick makes the graph static, regardless of which samplers are activated
  // this is important in order to minimize graph reallocations
  ggml_tensor *vLogitsT = ggml_pad(vCtx0, vRes->vTLogits, 0, 1, 0, 0);

  for (const auto &[vSeqId, vSampler] : dSamplers) {
    const auto cIt = dSeqToLogitRow.find(vSeqId);

    // inactive samplers always work on the first row
    const auto cRowIdx = cIt != dSeqToLogitRow.end() ? cIt->second : 0;
    const int cIOut = cIt != dSeqToLogitRow.end() ? 1 : 0;

    ggml_tensor *vLogitsSeq = ggml_view_1d(vCtx0, vLogitsT, vLogitsT->ne[0], cRowIdx * vLogitsT->nb[1]);
    ggml_format_name(vLogitsSeq, "logits_seq_%d", vSeqId);

    struct mgs_sampler_data vData = {
        /*.logits      =*/vLogitsSeq,
        /*.probs       =*/nullptr,
        /*.sampled     =*/nullptr,
        /*.candidates  =*/nullptr,
    };

    assert(vSampler->iface->backend_apply);
    vSampler->iface->backend_apply(vSampler, vCtx0, vGf, &vData);

    if (vData.sampled != nullptr) {
      vRes->dTSampled[vSeqId] = vData.sampled;
      aOuts[1] = vData.sampled;
      ggml_build_forward_select(vGf, aOuts.data(), aOuts.size(), cIOut);
    }

    if (vData.probs != nullptr) {
      vRes->dTSampledProbs[vSeqId] = vData.probs;
      aOuts[1] = vData.probs;
      ggml_build_forward_select(vGf, aOuts.data(), aOuts.size(), cIOut);
    }

    if (vData.logits != nullptr) {
      vRes->dTSampledLogits[vSeqId] = vData.logits;
      aOuts[1] = vData.logits;
      ggml_build_forward_select(vGf, aOuts.data(), aOuts.size(), cIOut);
    }

    if (vData.candidates != nullptr) {
      vRes->dTCandidates[vSeqId] = vData.candidates;
      aOuts[1] = vData.candidates;
      ggml_build_forward_select(vGf, aOuts.data(), aOuts.size(), cIOut);
    }
  }

  // TODO: Call mgs_sampler_accept_ggml after all samplers have been applied.
  /*
  for (const auto & [vSeqId, vSampler] : dSamplers) {
      if (auto vIt = vRes->dTSampled.find(vSeqId); vIt != vRes->dTSampled.end()) {
          ggml_tensor * vSelectedToken = vIt->second;
          if (vSelectedToken != nullptr) {
              mgs_sampler_accept_ggml(vSampler, vCtx0, vGf, vSelectedToken);
          }
      }
  }
  */
}

int32_t mgs_relative_position_bucket(mgs_pos pX, mgs_pos pY, uint64_t pNBuckets, bool pBidirectional) {
  // TODO move to hparams if a T5 variant appears that uses a different value
  const int64_t cMaxDistance = 128;

  if (pBidirectional) {
    pNBuckets >>= 1;
  }

  const int64_t cMaxExact = pNBuckets >> 1;

  int32_t vRelativePosition = pX - pY;
  int32_t vRelativeBucket = 0;

  if (pBidirectional) {
    vRelativeBucket += (vRelativePosition > 0) * pNBuckets;
    vRelativePosition = std::abs(vRelativePosition);
  } else {
    vRelativePosition = -std::min<int32_t>(vRelativePosition, 0);
  }

  int32_t vRelativePositionIfLarge =
      floorf(cMaxExact +
             logf(1.0 * vRelativePosition / cMaxExact) * (pNBuckets - cMaxExact) / log(1.0 * cMaxDistance / cMaxExact));
  vRelativePositionIfLarge = std::min<int32_t>(vRelativePositionIfLarge, pNBuckets - 1);
  vRelativeBucket += (vRelativePosition < cMaxExact ? vRelativePosition : vRelativePositionIfLarge);

  return vRelativeBucket;
}
