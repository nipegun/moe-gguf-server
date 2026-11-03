#pragma once

#include "mgs.h"
#include "mgs-cparams.h"

#include <bitset>
#include <cassert>
#include <cstring>
#include <limits>
#include <map>
#include <set>
#include <vector>

struct mgs_kv_cell_ext {
  // 2D spatial positions, typically used for M-RoPE
  mgs_pos vX = 0;
  mgs_pos vY = 0;

  // vTok is MGS_TOKEN_NULL when the cell is produced by embedding input (i.e. multimodal)
  // use case: n-gram embeddings hash
  mgs_token vTok = MGS_TOKEN_NULL;

  // return true if the current 2D spatial position is greater than other
  bool mIs2dGt(mgs_pos pOx, mgs_pos pOy) const { return (vY > pOy) || (vY == pOy && vX > pOx); }

  void mReset() {
    static_assert(std::is_trivially_copyable_v<mgs_kv_cell_ext>);

    *this = mgs_kv_cell_ext{};
  }
};

// meta information about KV cells that can be part of multiple sequences at the same time
// TODO: add unit tests
class mgs_kv_cells {
public:
  void mReset() {
    for (uint32_t vI = 0; vI < lPos.size(); ++vI) {
      lPos[vI] = -1;
      lExt[vI].mReset();
      lShift[vI] = 0;
      lSeq[vI].reset();
    }

    vHasShift = false;

    sUsed.clear();

    for (uint32_t vS = 0; vS < MGS_MAX_SEQ; ++vS) {
      sSeqPos[vS].clear();
    }
  }

  void mResetShift() {
    vHasShift = false;

    for (uint32_t vI = 0; vI < lShift.size(); ++vI) {
      lShift[vI] = 0;
    }
  }

  uint32_t size() const { return lPos.size(); }

  void mResize(uint32_t pN) {
    lPos.resize(pN);
    lExt.resize(pN);
    lShift.resize(pN);
    lSeq.resize(pN);

    mReset();
  }

  bool mIsEmpty(uint32_t pI) const {
    assert(pI < lPos.size());
    assert((lPos[pI] < 0 && lPos[pI] == -1) || lPos[pI] >= 0);

    return lPos[pI] == -1;
  }

  uint32_t mGetUsed() const { return sUsed.size(); }

  // the index of the first cell that is used
  // return 0 if no cells are used
  uint32_t mUsedMin() const { return sUsed.empty() ? 0 : *sUsed.begin(); }

  // the index of the last cell that is used + 1
  // return 0 if no cells are used
  uint32_t mUsedMaxP1() const { return sUsed.empty() ? 0 : *sUsed.rbegin() + 1; }

  bool mGetHasShift() const { return vHasShift; }

  // move cell isrc to idst (used during defrag)
  // void mMv(uint32_t pIsrc, uint32_t pIdst) {
  //    assert(pIsrc < lPos.size());
  //    assert(pIdst < lPos.size());

  //    assert(lPos[pIdst] == -1);
  //    assert(lPos[pIsrc] != -1);

  //    lPos  [pIdst] = lPos  [pIsrc];
  //    lShift[pIdst] = lShift[pIsrc];
  //    lSeq  [pIdst] = lSeq  [pIsrc];

  //    lPos  [pIsrc] = -1;
  //    lShift[pIsrc] =  0;
  //    lSeq  [pIsrc].reset();

  //    sUsed.erase (pIsrc);
  //    sUsed.insert(pIdst);
  //}

  // copy the state of cells [i, i + n) (used for save/restore the state of the cells)
  mgs_kv_cells mCp(uint32_t pI, uint32_t pN) const {
    assert(pI + pN <= lPos.size());

    mgs_kv_cells vRes;

    vRes.mResize(pN);

    for (uint32_t vJ = 0; vJ < pN; ++vJ) {
      const auto cIdx = pI + vJ;

      vRes.lPos[vJ] = lPos[cIdx];
      vRes.lExt[vJ] = lExt[cIdx];
      vRes.lSeq[vJ] = lSeq[cIdx];

      assert(lShift[cIdx] == 0);
    }

    return vRes;
  }

  // copy the state of cells [idxs[0], idxs[1], ..., idxs[idxs.size() - 1])
  mgs_kv_cells mCp(const std::vector<uint32_t> &pIdxs) const {
    mgs_kv_cells vRes;

    vRes.mResize(pIdxs.size());

    for (uint32_t vJ = 0; vJ < pIdxs.size(); ++vJ) {
      const auto cIdx = pIdxs[vJ];

      vRes.lPos[vJ] = lPos[cIdx];
      vRes.lExt[vJ] = lExt[cIdx];
      vRes.lSeq[vJ] = lSeq[cIdx];

      assert(lShift[cIdx] == 0);
    }

    return vRes;
  }

  // set the state of cells [i, i + other.pos.size()) (used for save/restore the state of the cells)
  void mSet(uint32_t pI, const mgs_kv_cells &pOther) {
    assert(pI + pOther.lPos.size() <= lPos.size());

    for (uint32_t vJ = 0; vJ < pOther.lPos.size(); ++vJ) {
      const auto cIdx = pI + vJ;

      if (lPos[cIdx] == -1 && pOther.lPos[vJ] != -1) {
        sUsed.insert(pI + vJ);
      }

      if (lPos[cIdx] != -1 && pOther.lPos[vJ] == -1) {
        sUsed.erase(pI + vJ);
      }

      if (lPos[cIdx] != -1) {
        mSeqPosRm(pI + vJ);
      }

      lPos[cIdx] = pOther.lPos[vJ];
      lExt[cIdx] = pOther.lExt[vJ];
      lSeq[cIdx] = pOther.lSeq[vJ];

      if (lPos[cIdx] != -1) {
        mSeqPosAdd(pI + vJ);
      }

      assert(lShift[cIdx] == 0);
    }
  }

  // set the state of cells [idxs[0], idxs[1], ..., idxs[idxs.size() - 1])
  void mSet(const std::vector<uint32_t> &pIdxs, const mgs_kv_cells &pOther) {
    assert(pIdxs.size() == pOther.lPos.size());

    for (uint32_t vJ = 0; vJ < pOther.lPos.size(); ++vJ) {
      const auto cIdx = pIdxs[vJ];

      if (lPos[cIdx] == -1 && pOther.lPos[vJ] != -1) {
        sUsed.insert(cIdx);
      }

      if (lPos[cIdx] != -1 && pOther.lPos[vJ] == -1) {
        sUsed.erase(cIdx);
      }

      if (lPos[cIdx] != -1) {
        mSeqPosRm(cIdx);
      }

      lPos[cIdx] = pOther.lPos[vJ];
      lExt[cIdx] = pOther.lExt[vJ];
      lSeq[cIdx] = pOther.lSeq[vJ];

      if (lPos[cIdx] != -1) {
        mSeqPosAdd(cIdx);
      }

      assert(lShift[cIdx] == 0);
    }
  }

  // clear a non-empty cell
  void mRm(uint32_t pI) {
    assert(pI < lPos.size());
    assert(lPos[pI] != -1);

    mSeqPosRm(pI);
    lSeq[pI].reset();

    lPos[pI] = -1;
    lExt[pI].mReset();
    lShift[pI] = 0;

    sUsed.erase(pI);
  }

  // note: call only if the cell has seq_id
  // return true if the cell becomes empty
  bool mSeqRm(uint32_t pI, mgs_seq_id pSeqId) {
    assert(pI < lPos.size());
    assert(lSeq[pI].test(pSeqId));
    assert(lPos[pI] != -1);
    assert(pSeqId >= 0);

    lSeq[pI].reset(pSeqId);
    mSeqPosDec(pSeqId, pI);

    if (lSeq[pI].none()) {
      lPos[pI] = -1;
      lExt[pI].mReset();
      lShift[pI] = 0;

      sUsed.erase(pI);

      return true;
    }

    return false;
  }

  // return true if the cell becomes empty (i.e. it did not contain seq_id before the call)
  bool mSeqKeep(uint32_t pI, mgs_seq_id pSeqId) {
    assert(pI < lPos.size());

    if (lSeq[pI].test(pSeqId)) {
      mSeqPosRm(pI);
      lSeq[pI].reset();

      lSeq[pI].set(pSeqId);
      mSeqPosInc(pSeqId, pI);

      return false;
    }

    if (lSeq[pI].any()) {
      mSeqPosRm(pI);
      lSeq[pI].reset();

      lPos[pI] = -1;
      lExt[pI].mReset();
      lShift[pI] = 0;

      sUsed.erase(pI);

      return true;
    }

    assert(lPos[pI] == -1);

    return false;
  }

  // number of different sequences in the cell
  int mSeqCount(uint32_t pI) const {
    assert(pI < lPos.size());
    assert(lPos[pI] != -1);

    return lSeq[pI].count();
  }

  // check if the cell contains seq_id
  bool mSeqHas(uint32_t pI, mgs_seq_id pSeqId) const {
    assert(pI < lPos.size());
    assert(pSeqId >= 0);

    return lSeq[pI].test(pSeqId);
  }

  // the full set of sequences this cell is visible to
  const std::bitset<MGS_MAX_SEQ> &mSeqGetAll(uint32_t pI) const {
    assert(pI < lPos.size());

    return lSeq[pI];
  }

  // the (pos, cell) pairs of sequence seq_id, ordered by position
  const std::set<std::pair<mgs_pos, uint32_t>> &mSeqPosGet(mgs_seq_id pSeqId) const {
    assert(pSeqId >= 0);
    assert(pSeqId < MGS_MAX_SEQ);

    return sSeqPos[pSeqId];
  }

  // the token of the cell of sequence seq_id at the largest position <= p
  // when several cells share that position, the one with the highest index wins
  // return MGS_TOKEN_NULL if the sequence has no cell at or before p
  // note: used by n-gram input embeddings to recover the tokens preceding a ubatch
  mgs_token mSeqPosTokLe(mgs_seq_id pSeqId, mgs_pos pP) const {
    assert(pSeqId >= 0);
    assert(pSeqId < MGS_MAX_SEQ);

    const auto &vSp = sSeqPos[pSeqId];

    auto vIt = vSp.upper_bound({pP, std::numeric_limits<uint32_t>::max()});
    if (vIt == vSp.begin()) {
      return MGS_TOKEN_NULL;
    }

    return lExt[(--vIt)->second].vTok;
  }

  // note: call only if the cell is not empty and the seq_id is not in the cell
  void mSeqAdd(uint32_t pI, mgs_seq_id pSeqId) {
    assert(pI < lPos.size());
    assert(lPos[pI] != -1);
    assert(!lSeq[pI].test(pSeqId));

    lSeq[pI].set(pSeqId);
    mSeqPosInc(pSeqId, pI);
  }

  // return the sequence id of this cell
  // note: call only for cells with exactly one sequence
  mgs_seq_id mSeqGet(uint32_t pI) const {
    assert(lSeq[pI].count() == 1);

    for (int vS = 0; vS < MGS_MAX_SEQ; ++vS) {
      if (lSeq[pI].test(vS)) {
        return vS;
      }
    }

    return -1;
  }

  // the minimum position of sequence seq_id currently present in any of the cells
  // return -1 if the sequence is not present
  mgs_pos mSeqPosMin(mgs_seq_id pSeqId) const {
    assert(pSeqId >= 0);
    assert(pSeqId < MGS_MAX_SEQ);

    if (sSeqPos[pSeqId].empty()) {
      return -1;
    }

    return sSeqPos[pSeqId].begin()->first;
  }

  // the maximum position of sequence seq_id currently present in any of the cells
  // return -1 if the sequence is not present
  mgs_pos mSeqPosMax(mgs_seq_id pSeqId) const {
    assert(pSeqId >= 0);
    assert(pSeqId < MGS_MAX_SEQ);

    if (sSeqPos[pSeqId].empty()) {
      return -1;
    }

    return sSeqPos[pSeqId].rbegin()->first;
  }

  // note: call only if the cell is not empty
  mgs_pos mPosGet(uint32_t pI) const {
    assert(pI < lPos.size());
    assert(lPos[pI] != -1);

    return lPos[pI];
  }

  const mgs_kv_cell_ext &mExtGet(uint32_t pI) const {
    assert(pI < lPos.size());
    assert(lPos[pI] != -1);

    return lExt[pI];
  }

  // note: call only if the cell is not empty
  mgs_pos mGetShift(uint32_t pI) const {
    assert(pI < lPos.size());
    assert(lPos[pI] != -1);

    return lShift[pI];
  }

  // check if a cell is not empty and its position is within [p0, p1)
  bool mPosIn(uint32_t pI, mgs_pos p0, mgs_pos p1) const {
    assert(pI < lPos.size());

    return lPos[pI] >= p0 && lPos[pI] < p1;
  }

  // set the position of an empty cell
  // does not modify "has_shift"
  // note: call only if the cell is empty
  void mPosSet(uint32_t pI, mgs_pos pP) {
    assert(pI < lPos.size());
    assert(lPos[pI] == -1);
    assert(lSeq[pI].none());

    lPos[pI] = pP;

    sUsed.insert(pI);
  }

  void mExtSet(uint32_t pI, mgs_kv_cell_ext pP) {
    assert(pI < lExt.size());
    lExt[pI] = pP;
  }

  // pos[i] = pos[i] + d
  // sets "has_shift" to true
  // note: call only if the cell is not empty
  bool mPosAdd(uint32_t pI, mgs_pos pD) {
    assert(pI < lPos.size());
    assert(lPos[pI] != -1);

    mSeqPosRm(pI);

    lPos[pI] += pD;
    lShift[pI] += pD;

    vHasShift = true;

    if (lPos[pI] < 0) {
      lSeq[pI].reset();
      lPos[pI] = -1;
      lShift[pI] = 0;

      sUsed.erase(pI);

      return true;
    }

    mSeqPosAdd(pI);

    return false;
  }

  // pos[i] = pos[i] / d
  // sets "has_shift" to true
  // note: call only if the cell is not empty
  void mPosDiv(uint32_t pI, int pD) {
    assert(pI < lPos.size());
    assert(lPos[pI] != -1);

    const mgs_pos cPOld = lPos[pI];

    mSeqPosRm(pI);

    lPos[pI] /= pD;
    lShift[pI] += cPOld - lPos[pI];

    mSeqPosAdd(pI);

    vHasShift = true;
  }

private:
  bool vHasShift = false;

  // set of indices of used cells (i.e. pos[i] != -1, allowed to not have any seq_id)
  std::set<uint32_t> sUsed;

  std::vector<mgs_pos> lPos;

  // stores extra info per cell
  std::vector<mgs_kv_cell_ext> lExt;

  // this array accumulates any applied shifts to the pos array since the last mResetShift() call
  // this is used to queue multiple updates to the pos array, which in the end can be applied in one go:
  //
  //   vCells.mPosAdd(vX, vShiftX);
  //   vCells.mPosDiv(vY, vShiftY);
  //   ...
  //
  //   if (vCells.mGetHasShift()) {
  //      for (int vI = 0; vI < vN; ++vI) {
  //          auto vShiftI = vCells.mGetShift(vI);
  //          ...
  //      }
  //      vCells.mResetShift();
  //   }
  //
  std::vector<mgs_pos> lShift;

  using seq_set_t = std::bitset<MGS_MAX_SEQ>;

  // the bitset seq[i] tells us which sequences are currently occupying the i-th cell
  std::vector<seq_set_t> lSeq;

  // the set sSeqPos[s] holds one (pos, cell) pair per cell that carries sequence s, ordered by position
  // this way sSeqPos[s].begin() and sSeqPos[s].rbegin() give us the min/max positions currently in the cache
  // and upper_bound() on a position finds the nearest cell of the sequence in logarithmic time
  //
  // the cell index is part of the key because a position can occur more than once for the same seq:
  //  - during performing a cache reuse via (rm + add)
  //  - some vision models have input embeddings with repeating positions
  //
  std::set<std::pair<mgs_pos, uint32_t>> sSeqPos[MGS_MAX_SEQ];

  // helper functions for updating `sSeqPos`, once cell at a time:

  void mSeqPosDec(mgs_seq_id pS, uint32_t pI) {
    const auto cN = sSeqPos[pS].erase({lPos[pI], pI});
    assert(cN == 1);
    GGML_UNUSED(cN);
  }

  void mSeqPosInc(mgs_seq_id pS, uint32_t pI) { sSeqPos[pS].insert({lPos[pI], pI}); }

  // remove cell i
  void mSeqPosRm(uint32_t pI) {
    for (int vS = 0; vS < MGS_MAX_SEQ; ++vS) {
      if (lSeq[pI].test(vS)) {
        mSeqPosDec(vS, pI);
      }
    }
  }

  // add cell i
  void mSeqPosAdd(uint32_t pI) {
    for (int vS = 0; vS < MGS_MAX_SEQ; ++vS) {
      if (lSeq[pI].test(vS)) {
        mSeqPosInc(vS, pI);
      }
    }
  }
};

using mgs_kv_cells_vec = std::vector<mgs_kv_cells>;
