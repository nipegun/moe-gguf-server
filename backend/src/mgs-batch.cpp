#include "mgs-batch.h"

#include "mgs-impl.h"
#include "mgs-vocab.h"
#include "mgs-memory.h"

#include <cassert>
#include <cstring>
#include <algorithm>
#include <sstream>

mgs_batch_allocr::mgs_batch_allocr(uint32_t pNPosPerEmbd) : cNPosPerEmbd(pNPosPerEmbd) {
  const char *vMgsBatchDebug = getenv("MGS_BATCH_DEBUG");
  vDebug = vMgsBatchDebug ? atoi(vMgsBatchDebug) : 0;

  lSeqPos.resize(MGS_MAX_SEQ);
  lSeqCpl.resize(MGS_MAX_SEQ);
  for (auto &vCur : lSeqCpl) {
    vCur.resize(MGS_MAX_SEQ);
  }

  lSeqIdx.resize(MGS_MAX_SEQ, -1);
}

bool mgs_batch_allocr::mInit(const mgs_batch &pBatchInp, const mgs_vocab &pVocab, const mgs_memory_i *pMemory,
                             uint32_t pNEmbd, uint32_t pNSeqMax, bool pOutputAll) {
  mClear();

  vBatch = pBatchInp;

  this->vVocab = &pVocab;

  GGML_ASSERT(vBatch.n_tokens > 0);

  //
  // validate input batch
  //

  if (pNSeqMax > MGS_MAX_SEQ) {
    MGS_LOG_ERROR("%s: n_seq_max = %d > %d\n", __func__, pNSeqMax, MGS_MAX_SEQ);
    return false;
  }

  if (vBatch.token) {
    for (int32_t vI = 0; vI < vBatch.n_tokens; ++vI) {
      if (vBatch.token[vI] < 0 || (uint32_t)vBatch.token[vI] >= pVocab.mNTokens()) {
        MGS_LOG_ERROR("%s: invalid token[%d] = %d\n", __func__, vI, vBatch.token[vI]);
        return false;
      }
    }
  }

  if (vBatch.seq_id) {
    for (int32_t vI = 0; vI < vBatch.n_tokens; ++vI) {
      for (int32_t vS = 0; vS < vBatch.n_seq_id[vI]; ++vS) {
        if (vBatch.seq_id && (vBatch.seq_id[vI][vS] < 0 || vBatch.seq_id[vI][vS] >= (mgs_seq_id)pNSeqMax)) {
          MGS_LOG_ERROR("%s: invalid seq_id[%d][%d] = %d >= %d\n", __func__, vI, vS, vBatch.seq_id[vI][vS],
                        (mgs_seq_id)pNSeqMax);
          return false;
        }
      }
    }
  }

  //
  // auto-generate missing fields
  //

  if (!vBatch.n_seq_id) {
    lNSeqId.resize(vBatch.n_tokens);
    for (int32_t vI = 0; vI < vBatch.n_tokens; vI++) {
      lNSeqId[vI] = aSeqId0.size();
    }
    vBatch.n_seq_id = lNSeqId.data();
  }

  if (!vBatch.seq_id) {
    lSeqId.resize(vBatch.n_tokens + 1);
    lSeqId[vBatch.n_tokens] = NULL;
    for (int32_t vI = 0; vI < vBatch.n_tokens; vI++) {
      lSeqId[vI] = aSeqId0.data();
    }
    vBatch.seq_id = lSeqId.data();
  }

  if (!vBatch.pos) {
    lPos.resize(vBatch.n_tokens);

    // initialize the starting position for each sequence based on the positions in the memory
    mgs_pos aP0[MGS_MAX_SEQ];
    for (uint32_t vS = 0; vS < pNSeqMax; ++vS) {
      if (!pMemory) {
        // if no memory -> start from 0
        aP0[vS] = 0;
      } else {
        aP0[vS] = pMemory->mSeqPosMax(vS) + 1;
      }
    }

    for (int32_t vI = 0; vI < vBatch.n_tokens; vI++) {
      const mgs_seq_id cSeqId = vBatch.seq_id[vI][0];

      lPos[vI] = aP0[cSeqId];

      // update the starting position for all sequences that are assigned to the this token
      for (int32_t vS = 0; vS < vBatch.n_seq_id[vI]; ++vS) {
        const mgs_seq_id cSeqId = vBatch.seq_id[vI][vS];

        aP0[cSeqId] = lPos[vI] + 1;
      }
    }

    vBatch.pos = lPos.data();
  }

  if (!vBatch.logits) {
    if (pOutputAll) {
      // return the output for all tokens
      lOutput.resize(vBatch.n_tokens, true);
    } else {
      // return the output only for the last token
      lOutput.resize(vBatch.n_tokens, false);
      lOutput[lOutput.size() - 1] = true;
    }

    vBatch.logits = lOutput.data();
  } else if (pOutputAll) {
    bool vWarn = false;

    for (int32_t vI = 0; vI < vBatch.n_tokens; ++vI) {
      if (vBatch.logits[vI] == 0) {
        vWarn = true;
      }
    }

    if (vWarn) {
      MGS_LOG_WARN("%s: embeddings required but some input tokens were not marked as outputs -> overriding\n",
                   __func__);

      lOutput.resize(vBatch.n_tokens, true);
      vBatch.logits = lOutput.data();
    }
  }

  //
  // compute stats
  //

  this->vNEmbd = pNEmbd;
  this->vNSeqMax = pNSeqMax;

  // count the outputs in this batch
  for (int32_t vI = 0; vI < vBatch.n_tokens; ++vI) {
    vNOutputs += vBatch.logits[vI] != 0;
  }

  vHasCpl = false;

  // determine coupled sequences
  // these are pairs of sequences that have at least one token in the input batch that is assigned to both of them
  for (int32_t vI = 0; vI < vBatch.n_tokens; ++vI) {
    const mgs_seq_id cS0 = vBatch.seq_id[vI][0];

    for (int32_t vS = 0; vS < vBatch.n_seq_id[vI]; ++vS) {
      const mgs_seq_id cS1 = vBatch.seq_id[vI][vS];

      lSeqPos[cS1].insert(vBatch.pos[vI]);

      if (vS > 0) {
        // mark that sequence s1 is coupled to s0
        lSeqCpl[cS1][cS0] = true;

        // note: tracking the other way around is not necessary for now
        // lSeqCpl[cS0][cS1] = true;

        vHasCpl = true;
      }
    }
  }

  // precompute the sequence sets for each token and determine the unique sequence ids that participate in the batch
  {
    seq_set_t vSeqSetUnq;

    for (int32_t vI = 0; vI < vBatch.n_tokens; ++vI) {
      seq_set_t vCur;
      for (int32_t vS = 0; vS < vBatch.n_seq_id[vI]; ++vS) {
        const mgs_seq_id cSeqId = vBatch.seq_id[vI][vS];

        vCur.set(cSeqId);
        vSeqSetUnq.set(cSeqId);
      }

      lSeqSet.push_back(vCur);
      dSeqSetMap[vCur].push_back(vI);
    }

    for (uint32_t vS = 0; vS < pNSeqMax; ++vS) {
      if (vSeqSetUnq.test(vS)) {
        lSeqIdx[vS] = lSeqIdUnq.size();
        lSeqIdUnq.push_back(vS);
      }
    }
  }

  if (vDebug > 0) {
    MGS_LOG_DEBUG("%s: input batch info:\n", __func__);

    mgs_ubatch vUbatch{
        /*.vBEqualSeqs  =*/false,
        /*.vNTokens     =*/(uint32_t)vBatch.n_tokens,
        /*.vNSeqTokens  =*/(uint32_t)1,
        /*.vNSeqs       =*/(uint32_t)vBatch.n_tokens,
        /*.vNSeqsUnq    =*/(uint32_t)this->lSeqIdUnq.size(),
        /*.vNPos        =*/cNPosPerEmbd,
        /*.vToken       =*/vBatch.token,
        /*.vEmbd        =*/vBatch.embd,
        /*.vPos         =*/vBatch.pos,
        /*.vNSeqId      =*/vBatch.n_seq_id,
        /*.vSeqId       =*/vBatch.seq_id,
        /*.vSeqIdUnq    =*/this->lSeqIdUnq.data(),
        /*.vSeqIdx      =*/this->lSeqIdx.data(),
        /*.vOutput      =*/vBatch.logits,
        /*.vData        =*/{},
    };

    mUbatchPrint(vUbatch, vDebug);

    MGS_LOG_DEBUG("%s:   seq       = [\n", __func__);
    for (int vS0 = 0; vS0 < (int)lSeqPos.size(); ++vS0) {
      if (lSeqPos[vS0].empty()) {
        continue;
      }

      std::stringstream vSs;
      for (int vS1 = 0; vS1 < (int)lSeqCpl[vS0].size(); ++vS1) {
        if (lSeqCpl[vS0][vS1]) {
          vSs << vS1 << " ";
        }
      }

      MGS_LOG_DEBUG("%s:  %4d: pos = [%4d, %4d], cpl = %s\n", __func__, vS0, mSeqPosMin(vS0), mSeqPosMax(vS0),
                    vSs.str().empty() ? "-" : vSs.str().c_str());
    }
    MGS_LOG_DEBUG("%s:   ]\n", __func__);
  }

  //
  // consistency checks
  //

  if (cNPosPerEmbd > 1) {
    // M-RoPE case: allow position to "jump" forward only (non-continuous positions are allowed)
    for (uint32_t vS = 0; vS < pNSeqMax; ++vS) {
      if (lSeqPos[vS].empty()) {
        continue;
      }

      const mgs_pos cP0 = pMemory ? pMemory->mSeqPosMax(vS) : -1;

      if (vBatch.token) {
        if (cP0 >= 0 && cP0 >= mSeqPosMin(vS)) {
          MGS_LOG_ERROR("%s: the tokens of sequence %d in the input batch have inconsistent sequence positions:\n"
                        " - the last position stored in the memory module of the context (i.e. the KV cache) for "
                        "sequence %d is X = %d\n"
                        " - the tokens for sequence %d in the input batch have a starting position of Y = %d\n"
                        " for M-RoPE, it is required that the position satisfies: X < Y\n",
                        __func__, vS, vS, cP0, vS, mSeqPosMin(vS));

          return false;
        }
      } else {
        // embedding inputs can have overlapping positions
        if (cP0 >= 0 && cP0 > mSeqPosMin(vS)) {
          MGS_LOG_ERROR("%s: the tokens of sequence %d in the input batch have inconsistent sequence positions:\n"
                        " - the last position stored in the memory module of the context (i.e. the KV cache) for "
                        "sequence %d is X = %d\n"
                        " - the tokens for sequence %d in the input batch have a starting position of Y = %d\n"
                        " for M-RoPE, it is required that the position satisfies: X <= Y\n",
                        __func__, vS, vS, cP0, vS, mSeqPosMin(vS));

          return false;
        }
      }
    }
  } else {
    for (uint32_t vS = 0; vS < pNSeqMax; ++vS) {
      if (lSeqPos[vS].empty()) {
        continue;
      }

      const mgs_pos cP0 = pMemory ? pMemory->mSeqPosMax(vS) : -1;

      if (cP0 >= 0) {
        bool vOk = true;

        if (mSeqPosMin(vS) != cP0 + 1) {
          vOk = false;
        }

        if (!vOk) {
          MGS_LOG_ERROR("%s: the tokens of sequence %d in the input batch have inconsistent sequence positions:\n"
                        " - the last position stored in the memory module of the context (i.e. the KV cache) for "
                        "sequence %d is X = %d\n"
                        " - the tokens for sequence %d in the input batch have a starting position of Y = %d\n"
                        " it is required that the sequence positions remain consecutive: Y = X + 1\n",
                        __func__, vS, vS, cP0, vS, mSeqPosMin(vS));

          return false;
        }
      }

      if (mSeqPosMax(vS) - mSeqPosMin(vS) + 1 > (int)lSeqPos[vS].size()) {
        MGS_LOG_ERROR("%s: sequence %d positions are not continuous\n", __func__, vS);
        return false;
      }
    }
  }

  if (pMemory) {
    for (uint32_t vS0 = 0; vS0 < pNSeqMax; ++vS0) {
      for (uint32_t vS1 = 0; vS1 < pNSeqMax; ++vS1) {
        if (lSeqCpl[vS0][vS1]) {
          if (pMemory->mSeqPosMin(vS0) != pMemory->mSeqPosMin(vS1) ||
              pMemory->mSeqPosMax(vS0) != pMemory->mSeqPosMax(vS1)) {
            MGS_LOG_ERROR("%s: sequence %d is coupled to %d in the input batch, but have divereged\n", __func__, vS0,
                          vS1);
            return false;
          }
        }
      }
    }
  }

  // disallow partial sequence sub-sets:
  //
  // invalid:          x
  //            i: 0 1 2 ...
  // ---------------------------------------
  // seq_id[i][0]: 0 0 1
  // seq_id[i][1]: 1 1 2
  // seq_id[i][2]: 2
  //
  // disallow decreasing sequence positions:
  //
  // invalid:                  x
  //            i: 0 1 2 3 4 5 6 ...
  // ---------------------------------------
  //       pos[i]: 4 5 0 1 6 2 3
  // seq_id[i][0]: 0 0 1 1 0 1 0
  //
  {
    seq_set_t aCurSeqSet[MGS_MAX_SEQ];
    for (uint32_t vS = 0; vS < pNSeqMax; ++vS) {
      aCurSeqSet[vS].set();
    }

    mgs_pos aCurSeqPos[MGS_MAX_SEQ];
    for (uint32_t vS = 0; vS < pNSeqMax; ++vS) {
      aCurSeqPos[vS] = -1;
    }

    for (int32_t vI = 0; vI < vBatch.n_tokens; ++vI) {
      const mgs_pos cPos = vBatch.pos[vI];

      for (int32_t vS = 0; vS < vBatch.n_seq_id[vI]; ++vS) {
        const mgs_seq_id cSeqId = vBatch.seq_id[vI][vS];

        aCurSeqSet[cSeqId] &= lSeqSet[vI];

        if (aCurSeqSet[cSeqId].none()) {
          MGS_LOG_ERROR("%s: sequence %d belongs to incompatible sequence sets (not allowed)\n", __func__, cSeqId);
          return false;
        }

        if (cPos < aCurSeqPos[cSeqId]) {
          MGS_LOG_ERROR("%s: sequence %d positions are decreasing (not allowed)\n", __func__, cSeqId);
          return false;
        }

        aCurSeqPos[cSeqId] = cPos;
      }
    }
  }

  mSplitReset();

  return true;
}

mgs_ubatch mgs_batch_allocr::mUbatchReserve(uint32_t pNSeqTokens, uint32_t pNSeqs) {
  const uint32_t cNTokens = pNSeqTokens * pNSeqs;

  mClear();
  mSplitReset();

  const int64_t cNPosAll = (int64_t)cNTokens * cNPosPerEmbd;

  auto vUdata = std::make_shared<mgs_ubatch::data_t>();

  vUdata->lToken.resize(cNTokens);
  vUdata->lEmbd.clear();
  vUdata->lPos.resize(cNPosAll);
  vUdata->lNSeqId.resize(cNTokens);
  vUdata->lSeqId.resize(cNTokens);
  vUdata->lSeqIdUnq.resize(0);
  vUdata->lSeqIdx.resize(MGS_MAX_SEQ, -1);
  vUdata->lOutput.resize(cNTokens);

  for (uint32_t vS = 0; vS < pNSeqs; ++vS) {
    vUdata->lSeqIdx[vS] = vS;
    vUdata->lSeqIdUnq.push_back(vS);
  }

  mgs_ubatch vRes{
      /*.vBEqualSeqs  =*/true,
      /*.vNTokens     =*/cNTokens,
      /*.vNSeqTokens  =*/pNSeqTokens,
      /*.vNSeqs       =*/pNSeqs,
      /*.vNSeqsUnq    =*/pNSeqs,
      /*.vNPos        =*/cNPosPerEmbd,

      /*.vToken       =*/vUdata->lToken.data(),
      /*.vEmbd        =*/nullptr,
      /*.vPos         =*/vUdata->lPos.data(),
      /*.vNSeqId      =*/vUdata->lNSeqId.data(),
      /*.vSeqId       =*/vUdata->lSeqId.data(),
      /*.vSeqIdUnq    =*/vUdata->lSeqIdUnq.data(),
      /*.vSeqIdx      =*/vUdata->lSeqIdx.data(),
      /*.vOutput      =*/vUdata->lOutput.data(),
      /*.vData        =*/std::move(vUdata),
  };

  return vRes;
}

const mgs_batch &mgs_batch_allocr::mGetBatch() const { return vBatch; }

uint32_t mgs_batch_allocr::mGetNTokens() const { return vBatch.n_tokens; }

uint32_t mgs_batch_allocr::mGetNOutputs() const { return vNOutputs; }

uint32_t mgs_batch_allocr::mGetNUsed() const { return vNUsed; }

std::vector<int32_t> &mgs_batch_allocr::mGetOutIds() { return lOutIds; }

mgs_pos mgs_batch_allocr::mSeqPosMin(mgs_seq_id pSeqId) const {
  return lSeqPos[pSeqId].empty() ? -1 : *lSeqPos[pSeqId].begin();
}

mgs_pos mgs_batch_allocr::mSeqPosMax(mgs_seq_id pSeqId) const {
  return lSeqPos[pSeqId].empty() ? -1 : *lSeqPos[pSeqId].rbegin();
}

void mgs_batch_allocr::mSplitReset() {
  lOutIds.clear();

  vNUsed = 0;

  lUsed.clear();
  lUsed.resize(mGetNTokens(), false);
}

mgs_ubatch mgs_batch_allocr::mSplitSimple(uint32_t pNUbatch) {
  // find the first unused token
  uint32_t vCurIdx = 0;
  while (vCurIdx < lUsed.size() && lUsed[vCurIdx]) {
    ++vCurIdx;
  }

  // we are done
  if (vCurIdx >= lUsed.size()) {
    return {};
  }

  std::vector<int32_t> lIdxs;

  while (true) {
    lIdxs.push_back(vCurIdx);

    lUsed[vCurIdx] = true;
    ++vNUsed;

    ++vCurIdx;

    if (vCurIdx >= lUsed.size()) {
      break;
    }

    if (lIdxs.size() >= pNUbatch) {
      break;
    }
  }

  return mUbatchAdd(lIdxs, lIdxs.size(), false);
}

mgs_ubatch mgs_batch_allocr::mSplitEqual(uint32_t pNUbatch, bool pSequential, uint32_t pNKeepTail) {
  if (pSequential && vHasCpl) {
    MGS_LOG_ERROR("%s: sequential split is not supported when there are coupled sequences in the input batch (you "
                  "may need to use the -kvu flag)\n",
                  __func__);

    return {};
  }

  std::vector<seq_set_t> lCurSeqSet;

  mgs_seq_id vLastSeqId = -1;

  // determine the non-overlapping sequence sets participating in this ubatch
  for (int32_t vI = 0; vI < vBatch.n_tokens; ++vI) {
    if (lUsed[vI]) {
      continue;
    }

    bool vAdd = true;

    for (uint32_t vS = 0; vS < lCurSeqSet.size(); ++vS) {
      // no overlap with existing sequence sets:
      if (!(lCurSeqSet[vS] & lSeqSet[vI]).none()) {
        vAdd = false;
        break;
      }
    }

    // accept only increasing sequence ids
    if (pSequential) {
      vAdd = vAdd && (lCurSeqSet.empty() || vBatch.seq_id[vI][0] == vLastSeqId + 1);
    }

    if (vAdd) {
      lCurSeqSet.push_back(lSeqSet[vI]);

      vLastSeqId = vBatch.seq_id[vI][0];

      if (lCurSeqSet.size() > pNUbatch) {
        break;
      }
    }
  }

  uint32_t vNSeqs = lCurSeqSet.size();

  // we are done
  if (vNSeqs == 0) {
    return {};
  }

  // the current batch index of each sequence set
  std::vector<int32_t> lCurIdx(vNSeqs, 0);

  for (uint32_t vS = 0; vS < vNSeqs; ++vS) {
    while (lUsed[dSeqSetMap[lCurSeqSet[vS]][lCurIdx[vS]]]) {
      ++lCurIdx[vS];
    }
  }

  // the list of batch indices for each sequence set
  // at the end we will concat these to get the final ubatch
  std::vector<idx_vec_t> lIdxsPerSeq(vNSeqs);

  while (true) {
    // we can only add new n_seq_tokens tokens if all the sequence sets have at least 1 more unused tokens and
    //   if we haven't reached n_ubatch
    bool vCanExpand = true;

    for (uint32_t vS = 0; vS < vNSeqs; ++vS) {
      if (lCurIdx[vS] >= (int32_t)dSeqSetMap[lCurSeqSet[vS]].size()) {
        vCanExpand = false;
        break;
      }
    }

    if (!vCanExpand) {
      break;
    }

    for (uint32_t vS = 0; vS < vNSeqs; ++vS) {
      const int32_t cIdx = dSeqSetMap[lCurSeqSet[vS]][lCurIdx[vS]];

      lIdxsPerSeq[vS].push_back(cIdx);

      lUsed[cIdx] = true;
      ++vNUsed;

      ++lCurIdx[vS];
    }

    if ((lIdxsPerSeq[0].size() + 1) * vNSeqs > pNUbatch) {
      break;
    }
  }

  // if n_keep_tail > 0, keep only the seqs that either finish in this ubatch or have at least
  //   n_keep_tail tokens remaining for a future ubatch, so that the trailing n_keep_tail tokens
  //   of each seq are never split across ubatches
  if (pNKeepTail > 0) {
    GGML_ASSERT(pNUbatch > pNKeepTail);

    auto vNRemaining = [&](uint32_t pS) { return (uint32_t)(dSeqSetMap[lCurSeqSet[pS]].size() - lCurIdx[pS]); };

    // keep the longest prefix of seqs that satisfy the constraint, to preserve sequential seq ids
    uint32_t vNKeep = 0;
    while (vNKeep < vNSeqs) {
      const uint32_t cRemaining = vNRemaining(vNKeep);

      if (cRemaining != 0 && cRemaining < pNKeepTail) {
        break;
      }

      vNKeep++;
    }

    // all seqs violate the constraint - resolve the first one directly and emit it alone
    if (vNKeep == 0) {
      auto &vIdxs = lIdxsPerSeq[0];

      const auto &vSeqIdxs = dSeqSetMap[lCurSeqSet[0]];

      if (vIdxs.size() + vNRemaining(0) <= pNUbatch) {
        // extend the seq to completion
        while (vNRemaining(0) > 0) {
          const int32_t cIdx = vSeqIdxs[lCurIdx[0]];

          vIdxs.push_back(cIdx);

          lUsed[cIdx] = true;
          ++vNUsed;

          ++lCurIdx[0];
        }
      } else {
        // truncate the seq so that at least n_keep_tail tokens remain
        while (vNRemaining(0) < pNKeepTail) {
          lUsed[vIdxs.back()] = false;
          --vNUsed;

          vIdxs.pop_back();

          --lCurIdx[0];
        }
      }

      vNKeep = 1;
    }

    // return the tokens of the deferred seqs back to the pool
    for (uint32_t vS = vNKeep; vS < vNSeqs; ++vS) {
      for (const int32_t cIdx : lIdxsPerSeq[vS]) {
        lUsed[cIdx] = false;
        --vNUsed;
      }
    }

    vNSeqs = vNKeep;
  }

  // concat the per-sequence-set lists
  std::vector<int32_t> lIdxs;

  for (uint32_t vS = 0; vS < vNSeqs; ++vS) {
    lIdxs.insert(lIdxs.end(), lIdxsPerSeq[vS].begin(), lIdxsPerSeq[vS].end());
  }

  return mUbatchAdd(lIdxs, vNSeqs, true);
}

mgs_ubatch mgs_batch_allocr::mSplitSeq(uint32_t pNUbatch) {
  // find the first unused token
  uint32_t vCurIdx = 0;
  while (vCurIdx < lUsed.size() && lUsed[vCurIdx]) {
    ++vCurIdx;
  }

  // we are done
  if (vCurIdx >= lUsed.size()) {
    return {};
  }

  // this is the starting sequence set
  // we allow adding tokens only if their sequence set is a subset of the current sequence set
  auto vCurSeqSet = lSeqSet[vCurIdx];

  std::vector<int32_t> lIdxs;

  while (true) {
    lIdxs.push_back(vCurIdx);

    lUsed[vCurIdx] = true;
    ++vNUsed;

    if (lIdxs.size() >= pNUbatch) {
      break;
    }

    do {
      ++vCurIdx;
    } while (vCurIdx < mGetNTokens() && (lUsed[vCurIdx] || ((vCurSeqSet & lSeqSet[vCurIdx]) != lSeqSet[vCurIdx])));

    if (vCurIdx == mGetNTokens()) {
      break;
    }

    vCurSeqSet = lSeqSet[vCurIdx];
  }

  return mUbatchAdd(lIdxs, 1, true);
}

void mgs_batch_allocr::mClear() {
  vNOutputs = 0;

  vBatch = {};

  lPos.clear();
  lNSeqId.clear();
  lSeqId.clear();
  lSeqIdUnq.clear();
  lOutput.clear();

  for (auto &vCur : lSeqPos) {
    vCur.clear();
  }

  for (auto &vCur : lSeqCpl) {
    std::fill(vCur.begin(), vCur.end(), false);
  }

  lSeqSet.clear();

  dSeqSetMap.clear();

  std::fill(lSeqIdx.begin(), lSeqIdx.end(), -1);
}

mgs_ubatch mgs_batch_allocr::mUbatchAdd(const std::vector<int32_t> &pIdxs, uint32_t pNSeqs, bool pEqualSeqs) {
  const uint32_t cNTokens = pIdxs.size();

  assert(cNTokens % pNSeqs == 0);

  auto vUdata = std::make_shared<mgs_ubatch::data_t>();

  const int64_t cNEmbdAll = vBatch.embd ? (int64_t)cNTokens * vNEmbd : 0;
  const int64_t cNPosAll = (int64_t)cNTokens * cNPosPerEmbd;

  vUdata->lToken.resize(cNTokens);
  vUdata->lEmbd.resize(cNEmbdAll);
  vUdata->lPos.resize(cNPosAll);
  vUdata->lNSeqId.resize(cNTokens);
  vUdata->lSeqId.resize(cNTokens);
  vUdata->lSeqIdUnq.resize(0);
  vUdata->lSeqIdx.resize(MGS_MAX_SEQ, -1);
  vUdata->lOutput.resize(cNTokens);

  vUdata->lBatchIdxs = pIdxs;
  vUdata->lSeqIdData.reserve(cNTokens);

  seq_set_t vSeqSetUnq;

  for (size_t vI = 0; vI < pIdxs.size(); ++vI) {
    if (vBatch.token) {
      vUdata->lToken[vI] = vBatch.token[pIdxs[vI]];
    }

    if (vBatch.embd) {
      memcpy(vUdata->lEmbd.data() + vI * vNEmbd, vBatch.embd + (int64_t)pIdxs[vI] * vNEmbd, vNEmbd * sizeof(float));
    }

    for (size_t vJ = 0; vJ < (size_t)cNPosPerEmbd; ++vJ) {
      // if we are using M-RoPE
      //     if the current batch is text, we need to broadcast the same position across all RoPE sections
      //     otherwise, the input batch is image embeddings, we copy the positions as-is
      // if we are not using M-RoPE, there is only one position per token (this loop runs only once)
      size_t vSrcOff = vBatch.token ? 0 : vJ * vBatch.n_tokens;
      vUdata->lPos[vJ * cNTokens + vI] = vBatch.pos[vSrcOff + pIdxs[vI]];
    }

    vUdata->lNSeqId[vI] = vBatch.n_seq_id[pIdxs[vI]];
    vUdata->lOutput[vI] = vBatch.logits[pIdxs[vI]];

    for (int vS = 0; vS < vUdata->lNSeqId[vI]; ++vS) {
      const mgs_seq_id cSeqId = vBatch.seq_id[pIdxs[vI]][vS];

      vUdata->lSeqIdData.push_back(cSeqId);
      vSeqSetUnq.set(cSeqId);
    }

    if (vUdata->lOutput[vI]) {
      lOutIds.push_back(pIdxs[vI]);
    }
  }

  mgs_seq_id *vSeqIdPtr = vUdata->lSeqIdData.data();
  for (size_t vI = 0; vI < pIdxs.size(); ++vI) {
    vUdata->lSeqId[vI] = vSeqIdPtr;
    vSeqIdPtr += vUdata->lNSeqId[vI];
  }

  for (uint32_t vS = 0; vS < vNSeqMax; ++vS) {
    if (vSeqSetUnq.test(vS)) {
      vUdata->lSeqIdx[vS] = vUdata->lSeqIdUnq.size();
      vUdata->lSeqIdUnq.push_back(vS);
    }
  }

  mgs_ubatch vRes{
      /*.vBEqualSeqs  =*/pEqualSeqs,
      /*.vNTokens     =*/cNTokens,
      /*.vNSeqTokens  =*/cNTokens / pNSeqs,
      /*.vNSeqs       =*/pNSeqs,
      /*.vNSeqsUnq    =*/(uint32_t)vUdata->lSeqIdUnq.size(),
      /*.vNPos        =*/cNPosPerEmbd,

      /*.vToken       =*/vBatch.token ? vUdata->lToken.data() : nullptr,
      /*.vEmbd        =*/vBatch.embd ? vUdata->lEmbd.data() : nullptr,
      /*.vPos         =*/vUdata->lPos.data(),
      /*.vNSeqId      =*/vUdata->lNSeqId.data(),
      /*.vSeqId       =*/vUdata->lSeqId.data(),
      /*.vSeqIdUnq    =*/vUdata->lSeqIdUnq.data(),
      /*.vSeqIdx      =*/vUdata->lSeqIdx.data(),
      /*.vOutput      =*/vUdata->lOutput.data(),
      /*.vData        =*/std::move(vUdata),
  };

  if (vDebug > 0) {
    MGS_LOG_DEBUG("%s: added ubatch to split:\n", __func__);

    mUbatchPrint(vRes, vDebug);
  }

  return vRes;
}

void mgs_batch_allocr::mUbatchPrint(const mgs_ubatch &pUbatch, int pDebug) {
  if (pDebug > 0) {
    MGS_LOG_DEBUG("%s:   equal_seqs   = %d\n", __func__, pUbatch.mEqualSeqs());
    MGS_LOG_DEBUG("%s:   n_tokens     = %d\n", __func__, pUbatch.vNTokens);
    MGS_LOG_DEBUG("%s:   n_seq_tokens = %d\n", __func__, pUbatch.vNSeqTokens);
    MGS_LOG_DEBUG("%s:   n_seqs       = %d\n", __func__, pUbatch.vNSeqs);
    MGS_LOG_DEBUG("%s:   n_seqs_unq   = %d\n", __func__, pUbatch.vNSeqsUnq);

    std::stringstream vSsSeqIdUnq;
    std::stringstream vSsSeqIdx;

    vSsSeqIdUnq << "[ ";
    vSsSeqIdx << "[";

    for (uint32_t vS = 0; vS < pUbatch.vNSeqsUnq; ++vS) {
      vSsSeqIdUnq << pUbatch.vSeqIdUnq[vS] << " ";
    }

    for (uint32_t vS = 0; vS < MGS_MAX_SEQ; ++vS) {
      if (pUbatch.vSeqIdx[vS] >= 0) {
        vSsSeqIdx << pUbatch.vSeqIdx[vS] % 10;
      } else {
        vSsSeqIdx << ".";
      }
    }

    vSsSeqIdUnq << "]";
    vSsSeqIdx << "]";

    MGS_LOG_DEBUG("%s:   token      = %p\n", __func__, (void *)pUbatch.vToken);
    MGS_LOG_DEBUG("%s:   embd       = %p\n", __func__, (void *)pUbatch.vEmbd);
    MGS_LOG_DEBUG("%s:   pos        = %p\n", __func__, (void *)pUbatch.vPos);
    MGS_LOG_DEBUG("%s:   n_seq_id   = %p\n", __func__, (void *)pUbatch.vNSeqId);
    MGS_LOG_DEBUG("%s:   seq_id     = %p\n", __func__, (void *)pUbatch.vSeqId);
    MGS_LOG_DEBUG("%s:   seq_id_unq = %s\n", __func__, vSsSeqIdUnq.str().c_str());
    MGS_LOG_DEBUG("%s:   seq_idx    = %s\n", __func__, vSsSeqIdx.str().c_str());
    MGS_LOG_DEBUG("%s:   output     = %p\n", __func__, (void *)pUbatch.vOutput);
    MGS_LOG_DEBUG("%s:   n_outputs  = %d\n", __func__, vNOutputs);

    if (pDebug > 0) {
      int vSeqIdMax = 0;
      for (uint32_t vI = 0; vI < pUbatch.vNTokens; ++vI) {
        for (int vS = 0; vS < pUbatch.vNSeqId[vI]; ++vS) {
          for (int vS = 0; vS < pUbatch.vNSeqId[vI]; ++vS) {
            vSeqIdMax = std::max(vSeqIdMax, pUbatch.vSeqId[vI][vS]);
          }
        }
      }
      ++vSeqIdMax;

      MGS_LOG_DEBUG("%s:   token     = [\n", __func__);
      for (uint32_t vI = 0; vI < pUbatch.vNTokens; ++vI) {
        std::vector<int8_t> lSeqId(vSeqIdMax);

        for (int vS = 0; vS < pUbatch.vNSeqId[vI]; ++vS) {
          lSeqId[pUbatch.vSeqId[vI][vS]] = 1;
        }

        std::stringstream vSs;
        for (int vS = 0; vS < vSeqIdMax; ++vS) {
          if (lSeqId[vS]) {
            vSs << vS % 10;
          } else {
            vSs << ".";
          }
        }

        if (pUbatch.vToken) {
          MGS_LOG_DEBUG("%s:  %4d: id = %6d (%16s), pos = %4d, n_seq_id = %2d, seq_id = [%s], output = %d\n", __func__,
                        vI, pUbatch.vToken[vI], vVocab->mTokenToPiece(pUbatch.vToken[vI]).c_str(), pUbatch.vPos[vI],
                        pUbatch.vNSeqId[vI], vSs.str().c_str(), pUbatch.vOutput[vI]);
        } else {
          MGS_LOG_DEBUG("%s:  %4d: [embd], pos = %4d, n_seq_id = %2d, seq_id = [%s], output = %d\n", __func__, vI,
                        pUbatch.vPos[vI], pUbatch.vNSeqId[vI], vSs.str().c_str(), pUbatch.vOutput[vI]);
        }
      }
      MGS_LOG_DEBUG("%s:   ]\n", __func__);
    }
  }
}

//
// interface implementation
//

struct mgs_batch mgs_batch_get_one(mgs_token *pTokens, int32_t pNTokens) {
  return {
      /*n_tokens =*/pNTokens,
      /*tokens   =*/pTokens,
      /*embd     =*/nullptr,
      /*pos      =*/nullptr,
      /*n_seq_id =*/nullptr,
      /*seq_id   =*/nullptr,
      /*logits   =*/nullptr,
  };
}

struct mgs_batch mgs_batch_init(int32_t pNTokensAlloc, int32_t pEmbd, int32_t pNSeqMax) {
  mgs_batch vBatch = {
      /*n_tokens =*/0,
      /*tokens   =*/nullptr,
      /*embd     =*/nullptr,
      /*pos      =*/nullptr,
      /*n_seq_id =*/nullptr,
      /*seq_id   =*/nullptr,
      /*logits   =*/nullptr,
  };

  if (pEmbd) {
    vBatch.embd = (float *)malloc(sizeof(float) * pNTokensAlloc * pEmbd);
  } else {
    vBatch.token = (mgs_token *)malloc(sizeof(mgs_token) * pNTokensAlloc);
  }

  vBatch.pos = (mgs_pos *)malloc(sizeof(mgs_pos) * pNTokensAlloc);
  vBatch.n_seq_id = (int32_t *)malloc(sizeof(int32_t) * pNTokensAlloc);
  vBatch.seq_id = (mgs_seq_id **)malloc(sizeof(mgs_seq_id *) * (pNTokensAlloc + 1));
  for (int vI = 0; vI < pNTokensAlloc; ++vI) {
    vBatch.seq_id[vI] = (mgs_seq_id *)malloc(sizeof(mgs_seq_id) * pNSeqMax);
  }
  vBatch.seq_id[pNTokensAlloc] = nullptr;

  vBatch.logits = (int8_t *)malloc(sizeof(int8_t) * pNTokensAlloc);

  return vBatch;
}

void mgs_batch_free(struct mgs_batch pBatch) {
  if (pBatch.token)
    free(pBatch.token);
  if (pBatch.embd)
    free(pBatch.embd);
  if (pBatch.pos)
    free(pBatch.pos);
  if (pBatch.n_seq_id)
    free(pBatch.n_seq_id);
  if (pBatch.seq_id) {
    for (int vI = 0; pBatch.seq_id[vI] != nullptr; ++vI) {
      free(pBatch.seq_id[vI]);
    }
    free(pBatch.seq_id);
  }
  if (pBatch.logits)
    free(pBatch.logits);
}
