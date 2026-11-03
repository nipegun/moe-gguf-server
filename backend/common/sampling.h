#pragma once

#include "mgs.h"

#include "common.h"

#include <string>
#include <vector>

// common_sampler extends mgs_sampler with additional functionality:
//
//  - grammar support
//  - custom sampler logic based on the parameters
//  - history of the last accepted tokens
//  - performance metrics
//
// This goal is to have a common implementation of the sampling logic shared across the examples.
// For example, depending on the temperature, the sampling chain can be very simple (greedy) or more
// complex (top-k, top-p, etc).
//
// Another example is related to the grammar. In general, the grammar constraints applied on the full
// vocabulary can be very taxing. To improve performance, the grammar can be applied only to the sampled
// token in order to verify if it fits the grammar. And only if the token doesn't fit the grammar, the
// grammar constraints are applied to the full vocabulary and the token is resampled.
//
// The common_sampler also maintains a container with the last accepted tokens. In the future, this can
// be moved into the core mgs engine.
//
// For convenience, the common_sampler also maintains a container with the current candidate tokens.
// This can be used to access the probabilities of the rest of the non-sampled tokens.
//
// TODO: measure grammar performance
//

struct common_sampler;

// mgs_sampler API overloads

// note: can mutate params in some cases
struct common_sampler *fCommonSamplerInit(const struct mgs_model *pModel, struct common_params_sampling &pParams,
                                          int32_t pNCtx = 0);

void fCommonSamplerFree(struct common_sampler *pGsmpl);

// if is_generated is true, the token is accepted by the sampling chain, the reasoning budget sampler, and the grammar
// sampler
void fCommonSamplerAccept(struct common_sampler *pGsmpl, mgs_token pToken, bool pIsGenerated);
void fCommonSamplerReset(struct common_sampler *pGsmpl);
struct common_sampler *fCommonSamplerClone(struct common_sampler *pGsmpl);

// arguments can be nullptr to skip printing
void fCommonPerfPrint(const struct mgs_context *pCtx, const struct common_sampler *pGsmpl);

// get the underlying mgs_sampler_chain
struct mgs_sampler *fCommonSamplerGet(const struct common_sampler *pGsmpl);

// extended sampling implementation:
//
// - set logits
// - apply the configured sampler chain
// - check if the token fits the grammar (if any)
// - if not: resample by first applying the grammar constraints and then sampling again (slower path)
//
// if grammar_first is true, the grammar is applied before the samplers (slower)
// useful in cases where all the resulting candidates (not just the sampled one) must fit the grammar
//
mgs_token fCommonSamplerSample(struct common_sampler *pGsmpl, struct mgs_context *pCtx, int pIdx,
                               bool pGrammarFirst = false);

// generalized version of fCommonSamplerSample
//
// will cross-reference the sampled tokens with a batch of draft tokens and accept those that match
// if the sampler disagrees at some point, we stop and return the accepted tokens up to now
//
//      fCommonSamplerSampleAndAcceptN(pGsmpl, pCtx, { pIdx }, {});
//
// is equivalent to
//
//      fCommonSamplerSample(pGsmpl, pCtx, pIdx);
//      fCommonSamplerAccept(pGsmpl, vToken, true);
//
// requires: pIdxs.size() == pDraft.size() + 1
//
// returns at least 1 token, up to pIdxs.size()
//
std::vector<mgs_token> fCommonSamplerSampleAndAcceptN(struct common_sampler *pGsmpl, struct mgs_context *pCtx,
                                                      const std::vector<int> &pIdxs, const mgs_tokens &pDraft,
                                                      bool pGrammarFirst = false);

// assume idxs == [ 0, 1, 2, ..., pDraft.size() ]
std::vector<mgs_token> fCommonSamplerSampleAndAcceptN(struct common_sampler *pGsmpl, struct mgs_context *pCtx,
                                                      const mgs_tokens &pDraft, bool pGrammarFirst = false);

uint32_t fCommonSamplerGetSeed(const struct common_sampler *pGsmpl);

// force the reasoning budget sampler (if any) to begin forcing its end sequence now.
bool fCommonSamplerReasoningBudgetForce(struct common_sampler *pGsmpl);

// helpers

// access the internal list of current candidate tokens
// if do_sort == true, the candidates are guaranteed to be sorted afterwards (in descending order of probability)
// the .sorted flag of the result indicates whether the returned candidates are sorted
mgs_token_data_array *fCommonSamplerGetCandidates(struct common_sampler *pGsmpl, bool pDoSort);

// get the last accepted token
mgs_token fCommonSamplerLast(const struct common_sampler *pGsmpl);

// print the sampler chain into a string
std::string fCommonSamplerPrint(const struct common_sampler *pGsmpl);

// get a string representation of the last accepted tokens
std::string fCommonSamplerPrevStr(common_sampler *pGsmpl, mgs_context *pCtx, int pN);

char fCommonSamplerTypeToChr(enum common_sampler_type pCnstr);
std::string fCommonSamplerTypeToStr(enum common_sampler_type pCnstr);

std::vector<enum common_sampler_type> fCommonSamplerTypesFromNames(const std::vector<std::string> &pNames);
std::vector<enum common_sampler_type> fCommonSamplerTypesFromChars(const std::string &pChars);

mgs_sampler *mgs_sampler_init_llg(const mgs_vocab *pVocab, const char *pGrammarKind, const char *pGrammarData);

struct common_sampler_deleter {
  void operator()(common_sampler *pS) { fCommonSamplerFree(pS); }
};

typedef std::unique_ptr<common_sampler, common_sampler_deleter> common_sampler_ptr;
