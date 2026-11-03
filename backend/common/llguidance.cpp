#include "sampling.h"
#include "log.h"

#ifdef MGS_USE_LLGUIDANCE

#include "llguidance.h"
#include <cmath>

struct mgs_sampler_llg {
  const mgs_vocab *vVocab;
  std::string vGrammarKind;
  std::string vGrammarData;
  LlgTokenizer *vTokenizer;
  LlgMatcher *vGrammar;
};

static LlgMatcher *fMgsSamplerLlgNew(LlgTokenizer *pTokenizer, const char *pGrammarKind, const char *pGrammarData) {
  LlgConstraintInit vCinit;
  llg_constraint_init_set_defaults(&vCinit, pTokenizer);
  const char *vLogLevel = getenv("LLGUIDANCE_LOG_LEVEL");
  if (vLogLevel && *vLogLevel) {
    vCinit.log_stderr_level = atoi(vLogLevel);
  }
  auto vC = llg_new_matcher(&vCinit, pGrammarKind, pGrammarData);
  if (llg_matcher_get_error(vC)) {
    LOG_ERR("llg error: %s\n", llg_matcher_get_error(vC));
    llg_free_matcher(vC);
    return nullptr;
  }

  return vC;
}

static const char *fMgsSamplerLlgName(const mgs_sampler * /*pSmpl*/) { return "llguidance"; }

static void fMgsSamplerLlgAcceptImpl(mgs_sampler *pSmpl, mgs_token pToken) {
  auto *vCtx = (mgs_sampler_llg *)pSmpl->ctx;
  if (vCtx->vGrammar) {
    llg_matcher_consume_token(vCtx->vGrammar, pToken);
  }
}

static void fMgsSamplerLlgApply(mgs_sampler *pSmpl, mgs_token_data_array *pCurP) {
  auto *vCtx = (mgs_sampler_llg *)pSmpl->ctx;
  if (vCtx->vGrammar) {
    const uint32_t *vMask = llg_matcher_get_mask(vCtx->vGrammar);
    if (vMask == nullptr) {
      if (llg_matcher_compute_mask(vCtx->vGrammar) == 0) {
        vMask = llg_matcher_get_mask(vCtx->vGrammar);
      } else {
        LOG_ERR("llg error: %s\n", llg_matcher_get_error(vCtx->vGrammar));
        llg_free_matcher(vCtx->vGrammar);
        vCtx->vGrammar = nullptr;
        return;
      }
    }

    for (size_t vI = 0; vI < pCurP->size; ++vI) {
      auto vToken = pCurP->data[vI].id;
      if ((vMask[vToken / 32] & (1 << (vToken % 32))) == 0) {
        pCurP->data[vI].logit = -INFINITY;
      }
    }
  }
}

static void fMgsSamplerLlgReset(mgs_sampler *pSmpl) {
  auto *vCtx = (mgs_sampler_llg *)pSmpl->ctx;
  if (vCtx->vGrammar) {
    llg_matcher_reset(vCtx->vGrammar);
  }
}

static mgs_sampler *fMgsSamplerLlgClone(const mgs_sampler *pSmpl) {
  const auto *vCtx = (const mgs_sampler_llg *)pSmpl->ctx;

  auto *vResult = mgs_sampler_init_llg(vCtx->vVocab, nullptr, nullptr);

  // copy the state
  {
    auto *vResultCtx = (mgs_sampler_llg *)vResult->ctx;

    if (vCtx->vGrammar) {
      vResultCtx->vGrammarKind = vCtx->vGrammarKind;
      vResultCtx->vGrammarData = vCtx->vGrammarData;
      vResultCtx->vGrammar = llg_clone_matcher(vCtx->vGrammar);
      vResultCtx->vTokenizer = llg_clone_tokenizer(vCtx->vTokenizer);
    }
  }

  return vResult;
}

static void fMgsSamplerLlgFree(mgs_sampler *pSmpl) {
  const auto *vCtx = (mgs_sampler_llg *)pSmpl->ctx;

  if (vCtx->vGrammar) {
    llg_free_matcher(vCtx->vGrammar);
    llg_free_tokenizer(vCtx->vTokenizer);
  }

  delete vCtx;
}

static mgs_sampler_i vMgsSamplerLlgI = {
    /* .name              = */ fMgsSamplerLlgName,
    /* .accept            = */ fMgsSamplerLlgAcceptImpl,
    /* .apply             = */ fMgsSamplerLlgApply,
    /* .reset             = */ fMgsSamplerLlgReset,
    /* .clone             = */ fMgsSamplerLlgClone,
    /* .free              = */ fMgsSamplerLlgFree,
    /* .backend_init      = */ NULL,
    /* .backend_accept    = */ NULL,
    /* .backend_apply     = */ NULL,
    /* .backend_set_input = */ NULL,
};

static size_t fMgsSamplerLlgTokenizeFn(const void *pUserData, const uint8_t *pBytes, size_t pBytesLen,
                                       uint32_t *pOutputTokens, size_t pOutputTokensLen) {
  const mgs_vocab *vVocab = (const mgs_vocab *)pUserData;
  int vR = 0;
  try {
    vR = mgs_tokenize(vVocab, (const char *)pBytes, pBytesLen, (int32_t *)pOutputTokens, pOutputTokensLen, false, true);
  } catch (const std::exception &vE) {
    GGML_ABORT("mgs_tokenize failed: %s\n", vE.what());
  }
  if (vR < 0) {
    return -vR;
  }
  return vR;
}

static LlgTokenizer *fMgsSamplerLlgNewTokenizer(const mgs_vocab *pVocab) {
  // TODO store the tokenizer in the vocab somehow
  static const mgs_vocab *vVocabCache;
  static LlgTokenizer *vTokenizerCache;

  if (vVocabCache == pVocab) {
    return llg_clone_tokenizer(vTokenizerCache);
  }

  auto vTokEos = mgs_vocab_eot(pVocab);
  if (vTokEos == MGS_TOKEN_NULL) {
    vTokEos = mgs_vocab_eos(pVocab);
  }

  size_t vVocabSize = mgs_vocab_n_tokens(pVocab);

  auto vTokenLens = new uint32_t[vVocabSize];
  // we typically have ~7 bytes per token; let's go on the safe side here
  auto vTokenBytesSize = vVocabSize * 16 + 1024 * 1024;
  auto vTokenBytes = new uint8_t[vTokenBytesSize];

  size_t vOffset = 0;
  for (size_t vI = 0; vI < vVocabSize; vI++) {
    size_t vMaxToken = 1024;
    if (vTokenBytesSize - vOffset < vMaxToken) {
      GGML_ABORT("token_bytes buffer too small\n");
    }

    mgs_token vToken = vI;
    auto vDp = (char *)vTokenBytes + vOffset;
    auto vSize = mgs_detokenize(pVocab, &vToken, 1, vDp, vMaxToken, false, false);
    if (vSize < 0) {
      GGML_ABORT("mgs_detokenize failed\n");
    }
    if (vSize == 0) {
      vSize = mgs_detokenize(pVocab, &vToken, 1, vDp + 1, vMaxToken - 1, false, true);
      if (vSize < 0) {
        GGML_ABORT("mgs_detokenize failed\n");
      }
      if (vSize != 0) {
        *vDp = '\xff'; // special token prefix marker
        vSize += 1;
      }
    }

    vTokenLens[vI] = vSize;
    vOffset += vSize;
  }

  LlgTokenizerInit vTinit = {
      /* .vocab_size                         = */ (uint32_t)vVocabSize,
      /* .tok_eos                            = */ (uint32_t)vTokEos,
      /* .token_lens                         = */ vTokenLens,
      /* .token_bytes                        = */ vTokenBytes,
      /* .tokenizer_json                     = */ nullptr,
      /* .tokenize_assumes_string            = */ true,
      /* .tokenize_fn                        = */ fMgsSamplerLlgTokenizeFn,
      /* .use_approximate_greedy_tokenize_fn = */ false,
      /* .tokenize_user_data                 = */ pVocab,
      /* .slices                             = */ nullptr,
  };

  char aErrorBuffer[1024];
  LlgTokenizer *vTokenizer = llg_new_tokenizer(&vTinit, aErrorBuffer, sizeof(aErrorBuffer));

  delete[] vTokenBytes;
  delete[] vTokenLens;

  if (vTokenizer == nullptr) {
    LOG_ERR("llg tokenizer error: %s\n", aErrorBuffer);
    return vTokenizer;
  }

  if (vTokenizerCache) {
    llg_free_tokenizer(vTokenizerCache);
  }
  vVocabCache = pVocab;
  vTokenizerCache = vTokenizer;

  return llg_clone_tokenizer(vTokenizerCache);
}

mgs_sampler *mgs_sampler_init_llg(const mgs_vocab *pVocab, const char *pGrammarKind, const char *pGrammarData) {
  auto *vCtx = new mgs_sampler_llg;

  if (pGrammarKind != nullptr && pGrammarKind[0] != '\0') {
    auto vTokenizer = fMgsSamplerLlgNewTokenizer(pVocab);
    *vCtx = {
        /* .vVocab       = */ pVocab,
        /* .vGrammarKind = */ pGrammarKind,
        /* .vGrammarData = */ pGrammarData,
        /* .vTokenizer   = */ vTokenizer,
        /* .vGrammar     = */ fMgsSamplerLlgNew(vTokenizer, pGrammarKind, pGrammarData),
    };
    if (vCtx->vGrammar) {
      GGML_ASSERT(((size_t)mgs_vocab_n_tokens(pVocab) + 31) / 32 * 4 == llg_matcher_get_mask_byte_size(vCtx->vGrammar));
    }
  } else {
    *vCtx = {
        /* .vVocab       = */ pVocab,
        /* .vGrammarKind = */ {},
        /* .vGrammarData = */ {},
        /* .vTokenizer   = */ nullptr,
        /* .vGrammar     = */ nullptr,
    };
  }

  return mgs_sampler_init(
      /* .iface = */ &vMgsSamplerLlgI,
      /* .ctx   = */ vCtx);
}

#else

mgs_sampler *mgs_sampler_init_llg(const mgs_vocab *, const char *, const char *) {
  LOG_WRN("llguidance (cmake -DMGS_LLGUIDANCE=ON) is not enabled");
  return nullptr;
}

#endif // MGS_USE_LLGUIDANCE
