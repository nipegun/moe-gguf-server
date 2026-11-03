#pragma once

#include "common.h"

#include <cstdint>
#include <string>
#include <vector>

// Number of draft tokens of the MTP integrated in a model, measured on the machine that runs mgs. Whether MTP pays
// off depends on the machine: on an RTX 4060 Ti, Qwen3.8-Flash-Next (most experts read from RAM and disk) generated
// 1.4 tokens/s with MTP and 1.7 without it, while Qwen3.6-35B-A3B generated 40 with MTP and 31 without it. The
// server alternates 0 (no draft) to cSpecTuningMaxDraft draft tokens in blocks of generated tokens and keeps the
// fastest; the result is saved per model, machine and version of mgs in ~/.mgs/tuning/mtp.json, so the next start
// uses it directly, and does not load MTP at all when generation was faster without it.

// Highest number of draft tokens measured.
constexpr int32_t cSpecTuningMaxDraft = 3;

// Key of the results of the model of pParams on this machine with this version of mgs.
std::string fSpecTuningKey(const common_params &pParams);

// Path of the file of saved results, or an empty string when the home folder is unknown.
std::string fSpecTuningPath();

// The number of draft tokens saved for pKey, or -1 when there is none; pSummary describes the measurement.
int32_t fSpecTuningLoad(const std::string &pKey, std::string &pSummary);

// Saves pDraft and the speeds in tokens/s per number of draft tokens (pRates[k]) for pKey.
bool fSpecTuningSave(const std::string &pKey, int32_t pDraft, const std::vector<double> &pRates, std::string &pError);

// Chooses the number of draft tokens while one slot generates. After cWarmupTokens generated tokens that do not
// count, the candidates 0 to pMaxDraft take turns step by step, in an order that is reversed every other round, until
// each one has generated cMinTokens, so that they share the same text and the same state of the page cache. In blocks
// of generated tokens they did not: with 32-token blocks the first answer on a cold page cache gave 2 draft tokens
// 4.6 tokens/s on a Radeon 860M, which generates 7.2 with them, and with 8-token blocks Qwen3.8-Flash-Next on an RTX
// 4060 Ti (experts read from disk) gave them 1.85, where they generate 1.4. Only the time between two steps of the
// same task while no other slot works is counted. Drafting has to beat no draft by cKeepMargin: without drafting the
// MTP context still processes every batch, and its weights and buffers keep device memory that the experts, the
// context and the micro-batch get when MTP is not loaded, none of which the measurement sees.
struct common_spec_tuner {
  static constexpr int64_t cWarmupTokens = 32;
  static constexpr int64_t cMinTokens = 64;
  static constexpr double cKeepMargin = 0.05;

  explicit common_spec_tuner(int32_t pMaxDraft);

  // Credits the time since the previous step of the same task to the number of draft tokens it used and returns the
  // number for the next step. pClean is false when another slot also works in this step; pReplay is true when this
  // step repeats a partly accepted draft, whose cost belongs to the number of draft tokens of the previous step, so
  // the turn does not change on it.
  int32_t mStep(int32_t pSlotId, int32_t pTaskId, int64_t pNDecoded, int64_t pTimeUs, bool pClean, bool pReplay);

  // True once every candidate has generated cMinTokens; mBest() and mRates() hold the result.
  bool mFinished() const { return vBest >= 0; }
  int32_t mBest() const { return vBest; }
  std::vector<double> mRates() const;

private:
  std::vector<int32_t> lOrder;
  size_t vOrderIndex = 0;
  std::vector<int64_t> lTokens;
  std::vector<int64_t> lTimeUs;
  int64_t vWarmupTokens = 0;
  int32_t vBest = -1;

  int32_t vLastSlot = -1;
  int32_t vLastTask = -1;
  int64_t vLastDecoded = 0;
  int64_t vLastTimeUs = 0;
};
