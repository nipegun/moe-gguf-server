#pragma once

#include "mgs.h"

#include "common.h"

#include <cstdint>
#include <vector>

enum common_reasoning_budget_state {
  cReasoningBudgetIdle,        // waiting for start sequence
  cReasoningBudgetCounting,    // counting down tokens
  cReasoningBudgetForcing,     // forcing budget message + end sequence
  cReasoningBudgetWaitingUtf8, // budget exhausted, waiting for UTF-8 completion
  cReasoningBudgetDone,        // passthrough forever
};

// Creates a reasoning budget sampler that limits token generation inside a
// reasoning block (e.g. between <think> and </think>).
//
// State machine: IDLE -> COUNTING -> WAITING_UTF8 -> FORCING -> DONE
//   IDLE:         passthrough, watching for a start sequence
//   COUNTING:     counting down remaining tokens, watching for a natural end sequence
//   WAITING_UTF8: budget exhausted, allowing tokens to complete a UTF-8 sequence
//   FORCING:      forces forced_tokens token-by-token (all other logits -> -inf)
//   DONE:         passthrough forever
//
// Parameters:
//   vocab          - vocabulary (used for UTF-8 boundary detection; can be nullptr)
//   start_seqs     - token sequences, any of which activates counting
//   end_seqs       - token sequences, any of which naturally deactivates
//   forced_tokens  - token sequence forced when budget expires
//   budget         - max tokens allowed in the reasoning block
//   initial_state  - initial state
//
struct mgs_sampler *fCommonReasoningBudgetInit(const struct mgs_vocab *pVocab,
                                               const std::vector<mgs_tokens> &pStartSeqs,
                                               const std::vector<mgs_tokens> &pEndSeqs, const mgs_tokens &pForcedTokens,
                                               int32_t pBudget,
                                               common_reasoning_budget_state pInitialState = cReasoningBudgetIdle);

common_reasoning_budget_state fCommonReasoningBudgetGetState(const struct mgs_sampler *pSmpl);

// The end sequence that transitioned the sampler to DONE, or nullptr if none
// was recorded. Cleared when a new start sequence re-arms the sampler.
const mgs_tokens *fCommonReasoningBudgetGetEndMatch(const struct mgs_sampler *pSmpl);

// Manually transition the reasoning budget sampler into the FORCING state.
// Returns true if the transition occurred.
bool fCommonReasoningBudgetForce(struct mgs_sampler *pSmpl);
