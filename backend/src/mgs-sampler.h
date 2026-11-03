#pragma once

#include "mgs.h"

#include <vector>

struct mgs_vocab;
struct mgs_grammar;

// sampler chain

struct mgs_sampler_chain {
  mgs_sampler_chain_params vParams;

  // has .backend_init() been called?
  bool vIsInit = false;

  struct info {
    bool vIsBackend;

    mgs_sampler *vPtr;
  };

  std::vector<info> lSamplers;

  // pre-allocated buffer for mgs_sampler_sample to avoid repeated allocations
  std::vector<mgs_token_data> lCur;

  // timing

  mutable int64_t vTSampleUs;

  mutable int32_t vNSample;
};

struct mgs_sampler *mgs_sampler_init_dry_testing(int32_t pContextSize, float pDryMultiplier, float pDryBase,
                                                 int32_t pDryAllowedLength, int32_t pDryPenaltyLastN,
                                                 const std::vector<std::vector<mgs_token>> &pSeqBreakers);
