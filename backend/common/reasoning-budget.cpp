#include "reasoning-budget.h"
#include "common.h"
#include "trie.h"
#include "unicode.h"

#include "log.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

struct vTokenMatcher {
  std::vector<mgs_tokens> lSeqs;
  common_aho_corasick vAc;
  size_t vState = 0;

  vTokenMatcher(const std::vector<mgs_tokens> &pSeqs) : lSeqs(mCollect(pSeqs)), vAc(mBuildTrie(this->lSeqs)) {}

  static std::vector<mgs_tokens> mCollect(const std::vector<mgs_tokens> &pSeqs) {
    std::vector<mgs_tokens> lRes;
    for (const auto &vSeq : pSeqs) {
      if (!vSeq.empty() && std::find(lRes.begin(), lRes.end(), vSeq) == lRes.end()) {
        lRes.push_back(vSeq);
      }
    }
    return lRes;
  }

  static common_trie mBuildTrie(const std::vector<mgs_tokens> &pSeqs) {
    common_trie vT;
    for (const auto &vSeq : pSeqs) {
      vT.mInsert(std::vector<uint32_t>(vSeq.begin(), vSeq.end()));
    }
    return vT;
  }

  // returns the index into seqs of the longest sequence ending at this token, or -1
  int32_t mAdvance(mgs_token pToken) {
    vState = vAc.mNext(vState, (uint32_t)pToken);
    const int32_t cP = vAc.mMatchPattern(vState);
    if (cP >= 0) {
      vState = 0;
    }
    return cP;
  }

  void mReset() { vState = 0; }
};

struct common_reasoning_budget_ctx {
  const mgs_vocab *vVocab;

  vTokenMatcher vStartMatcher;
  vTokenMatcher vEndMatcher;
  mgs_tokens vForcedTokens;

  int32_t vBudget;    // maximum tokens in reasoning block
  int32_t vRemaining; // tokens remaining in budget

  common_reasoning_budget_state vState;

  // for forcing
  size_t vForcePos; // next position in vForcedTokens to force

  int32_t vEndMatch; // index into vEndMatcher.lSeqs of the sequence that transitioned to DONE, -1 if none
};

static const char *fCommonReasoningBudgetName(const struct mgs_sampler * /*smpl*/) { return "reasoning-budget"; }

static void fCommonReasoningBudgetAccept(struct mgs_sampler *pSmpl, mgs_token pToken) {
  auto *vCtx = (common_reasoning_budget_ctx *)pSmpl->ctx;

  switch (vCtx->vState) {
  case cReasoningBudgetIdle: {
    if (vCtx->vStartMatcher.mAdvance(pToken) >= 0) {
      vCtx->vState = cReasoningBudgetCounting;
      vCtx->vRemaining = vCtx->vBudget;
      COM_TRC("activated, budget=%d tokens\n", vCtx->vBudget);

      if (vCtx->vRemaining <= 0) {
        vCtx->vState = cReasoningBudgetForcing;
        vCtx->vForcePos = 0;
        COM_TRC("%s", "budget=0, forcing immediately\n");
      }
    }
    break;
  }
  case cReasoningBudgetCounting:
  case cReasoningBudgetWaitingUtf8: {
    const int32_t cMatch = vCtx->vEndMatcher.mAdvance(pToken);
    if (cMatch >= 0) {
      vCtx->vState = cReasoningBudgetDone;
      vCtx->vEndMatch = cMatch;
      COM_TRC("%s", "deactivated (natural end)\n");
      break;
    }

    bool vUtf8Complete = true;
    if (vCtx->vVocab != nullptr) {
      const std::string cPiece = fCommonTokenToPiece(vCtx->vVocab, pToken, false);
      vUtf8Complete = fCommonUtf8IsComplete(cPiece);
    }

    if (vCtx->vState == cReasoningBudgetWaitingUtf8) {
      if (vUtf8Complete) {
        vCtx->vState = cReasoningBudgetForcing;
        vCtx->vForcePos = 0;
        vCtx->vEndMatcher.mReset();
        COM_TRC("%s", "UTF-8 complete, now forcing end sequence\n");
      }
    } else if (vCtx->vState == cReasoningBudgetCounting) {
      vCtx->vRemaining--;
      if (vCtx->vRemaining <= 0) {
        if (vUtf8Complete) {
          vCtx->vState = cReasoningBudgetForcing;
          vCtx->vForcePos = 0;
          vCtx->vEndMatcher.mReset();
          COM_TRC("%s", "budget exhausted, forcing end sequence\n");
        } else {
          vCtx->vState = cReasoningBudgetWaitingUtf8;
          vCtx->vEndMatcher.mReset();
          COM_TRC("%s", "budget exhausted, waiting for UTF-8 completion\n");
        }
      }
    }
    break;
  }
  case cReasoningBudgetForcing: {
    // track the end sequence within forced_tokens so it is also reported on DONE
    const int32_t cMatch = vCtx->vEndMatcher.mAdvance(pToken);
    vCtx->vForcePos++;
    if (vCtx->vForcePos >= vCtx->vForcedTokens.size()) {
      vCtx->vState = cReasoningBudgetDone;
      vCtx->vEndMatch = cMatch;
      COM_TRC("%s", "forced sequence complete, done\n");
    }
    break;
  }
  case cReasoningBudgetDone:
    // Re-arm on a new start tag: some models emit multiple <think> blocks
    // per response, and each should get a fresh budget window.
    if (vCtx->vStartMatcher.mAdvance(pToken) >= 0) {
      vCtx->vState = cReasoningBudgetCounting;
      vCtx->vRemaining = vCtx->vBudget;
      vCtx->vEndMatcher.mReset();
      vCtx->vEndMatch = -1;
      COM_TRC("re-activated on new start tag, budget=%d tokens\n", vCtx->vBudget);

      if (vCtx->vRemaining <= 0) {
        vCtx->vState = cReasoningBudgetForcing;
        vCtx->vForcePos = 0;
        COM_TRC("%s", "budget=0, forcing immediately\n");
      }
    }
    break;
  }
}

static void fCommonReasoningBudgetApply(struct mgs_sampler *pSmpl, mgs_token_data_array *pCurP) {
  auto *vCtx = (common_reasoning_budget_ctx *)pSmpl->ctx;

  if (vCtx->vState != cReasoningBudgetForcing) {
    // passthrough — don't modify logits
    return;
  }

  if (vCtx->vForcePos >= vCtx->vForcedTokens.size()) {
    return;
  }

  const mgs_token cForced = vCtx->vForcedTokens[vCtx->vForcePos];

  // set all logits to -inf except the forced token
  for (size_t vI = 0; vI < pCurP->size; vI++) {
    if (pCurP->data[vI].id != cForced) {
      pCurP->data[vI].logit = -INFINITY;
    }
  }
}

static void fCommonReasoningBudgetReset(struct mgs_sampler *pSmpl) {
  auto *vCtx = (common_reasoning_budget_ctx *)pSmpl->ctx;
  vCtx->vState = cReasoningBudgetIdle;
  vCtx->vRemaining = vCtx->vBudget;
  vCtx->vStartMatcher.mReset();
  vCtx->vEndMatcher.mReset();
  vCtx->vForcePos = 0;
  vCtx->vEndMatch = -1;
}

static struct mgs_sampler *fCommonReasoningBudgetInitState(const struct mgs_vocab *pVocab,
                                                           const std::vector<mgs_tokens> &pStartSeqs,
                                                           const std::vector<mgs_tokens> &pEndSeqs,
                                                           const mgs_tokens &pForcedTokens, int32_t pBudget,
                                                           common_reasoning_budget_state pInitialState);

static struct mgs_sampler *fCommonReasoningBudgetClone(const struct mgs_sampler *pSmpl);

static void fCommonReasoningBudgetFree(struct mgs_sampler *pSmpl) { delete (common_reasoning_budget_ctx *)pSmpl->ctx; }

static struct mgs_sampler_i vCommonReasoningBudgetI = {
    /* .name              = */ fCommonReasoningBudgetName,
    /* .accept            = */ fCommonReasoningBudgetAccept,
    /* .apply             = */ fCommonReasoningBudgetApply,
    /* .reset             = */ fCommonReasoningBudgetReset,
    /* .clone             = */ fCommonReasoningBudgetClone,
    /* .free              = */ fCommonReasoningBudgetFree,
    /* .backend_init      = */ nullptr,
    /* .backend_accept    = */ nullptr,
    /* .backend_apply     = */ nullptr,
    /* .backend_set_input = */ nullptr,
};

static struct mgs_sampler *fCommonReasoningBudgetClone(const struct mgs_sampler *pSmpl) {
  const auto *vCtx = (const common_reasoning_budget_ctx *)pSmpl->ctx;

  return mgs_sampler_init(
      /* .iface = */ &vCommonReasoningBudgetI,
      /* .ctx   = */ new common_reasoning_budget_ctx(*vCtx));
}

static struct mgs_sampler *fCommonReasoningBudgetInitState(const struct mgs_vocab *pVocab,
                                                           const std::vector<mgs_tokens> &pStartSeqs,
                                                           const std::vector<mgs_tokens> &pEndSeqs,
                                                           const mgs_tokens &pForcedTokens, int32_t pBudget,
                                                           common_reasoning_budget_state pInitialState) {
  // promote COUNTING with budget <= 0 to FORCING
  if (pInitialState == cReasoningBudgetCounting && pBudget <= 0) {
    pInitialState = cReasoningBudgetForcing;
  }

  return mgs_sampler_init(
      /* .iface = */ &vCommonReasoningBudgetI,
      /* .ctx   = */ new common_reasoning_budget_ctx{
          /* .vVocab        = */ pVocab,
          /* .vStartMatcher = */ vTokenMatcher(pStartSeqs),
          /* .vEndMatcher   = */ vTokenMatcher(pEndSeqs),
          /* .vForcedTokens = */ pForcedTokens,
          /* .vBudget       = */ pBudget,
          /* .vRemaining    = */ pBudget,
          /* .vState        = */ pInitialState,
          /* .vForcePos     = */ 0,
          /* .vEndMatch     = */ -1,
      });
}

struct mgs_sampler *fCommonReasoningBudgetInit(const struct mgs_vocab *pVocab,
                                               const std::vector<mgs_tokens> &pStartSeqs,
                                               const std::vector<mgs_tokens> &pEndSeqs, const mgs_tokens &pForcedTokens,
                                               int32_t pBudget, common_reasoning_budget_state pInitialState) {
  return fCommonReasoningBudgetInitState(pVocab, pStartSeqs, pEndSeqs, pForcedTokens, pBudget, pInitialState);
}

common_reasoning_budget_state fCommonReasoningBudgetGetState(const struct mgs_sampler *pSmpl) {
  if (!pSmpl) {
    return cReasoningBudgetIdle;
  }
  return ((const common_reasoning_budget_ctx *)pSmpl->ctx)->vState;
}

const mgs_tokens *fCommonReasoningBudgetGetEndMatch(const struct mgs_sampler *pSmpl) {
  if (!pSmpl) {
    return nullptr;
  }

  const auto *vCtx = (const common_reasoning_budget_ctx *)pSmpl->ctx;
  if (vCtx->vEndMatch < 0) {
    return nullptr;
  }

  return &vCtx->vEndMatcher.lSeqs[vCtx->vEndMatch];
}

bool fCommonReasoningBudgetForce(struct mgs_sampler *pSmpl) {
  if (!pSmpl) {
    return false;
  }

  auto *vCtx = (common_reasoning_budget_ctx *)pSmpl->ctx;

  // only a sampler that is actively counting down the budget may be forced;
  // any other state (idle, already forcing/waiting, or done) is left untouched
  if (vCtx->vState != cReasoningBudgetCounting) {
    return false;
  }

  vCtx->vState = cReasoningBudgetForcing;
  vCtx->vForcePos = 0;
  vCtx->vEndMatcher.mReset();
  COM_TRC("%s", "forced into forcing state (manual transition)\n");

  return true;
}
