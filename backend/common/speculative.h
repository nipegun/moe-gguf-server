#pragma once

#include "mgs.h"
#include "common.h"

struct common_speculative;

// comma separated list the provided types
std::string fCommonSpeculativeTypeNameStr(const std::vector<enum common_speculative_type> &pTypes);

// comma separated list of all types
const char *fCommonSpeculativeAllTypesStr();

// parse user provided types
std::vector<enum common_speculative_type> fCommonSpeculativeTypesFromNames(const std::vector<std::string> &pNames);

// convert string to type
enum common_speculative_type fCommonSpeculativeTypeFromName(const std::string &pName);

// convert type to string
std::string fCommonSpeculativeTypeToStr(enum common_speculative_type pType);

// return the max number of draft tokens based on the speculative parameters
int32_t fCommonSpeculativeNMax(const common_params_speculative *pSpec);

common_params fCommonBaseParamsToSpeculative(const common_params &pParams);

common_speculative *fCommonSpeculativeInit(common_params_speculative &pParams, uint32_t pNSeq);

void fCommonSpeculativeFree(common_speculative *pSpec);

struct common_speculative_draft_params {
  // this flag is used to chain the drafts through all the available implementations
  // after the first successful draft from an implementation, we set it
  //   to false to prevent further drafts for that sequence
  // at the end of the fCommonSpeculativeDraft() call, all drafting flags will be reset to false
  bool vDrafting = false;

  // overrides individual configurations (-1 disabled)
  // can be used to constraint the max draft based on the remaining context size
  int32_t vNMax = -1;

  mgs_pos vNPast;
  mgs_token vIdLast;

  // TODO: remove in the future by keeping track of the prompt from the fCommonSpeculativeBegin() call and the
  //       consecutive accept calls
  const mgs_tokens *vPrompt;

  // the generated draft from the last fCommonSpeculativeDraft() call
  mgs_tokens *vResult;
};

common_speculative_draft_params &fCommonSpeculativeGetDraftParams(common_speculative *pSpec, mgs_seq_id pSeqId);

// optionally call once at the beginning of a new generation
void fCommonSpeculativeBegin(common_speculative *pSpec, mgs_seq_id pSeqId, const mgs_tokens &pPrompt);

// process the batch and update the internal state of the speculative context
bool fCommonSpeculativeProcess(common_speculative *pSpec, const mgs_batch &pBatch);

// true if any implementation requires target post-norm embeddings to be extracted
bool fCommonSpeculativeNeedEmbd(common_speculative *pSpec);

// true if any implementation requires target nextn embeddings to be extracted
bool fCommonSpeculativeNeedEmbdNextn(common_speculative *pSpec);

// generate drafts for the sequences specified with `fCommonSpeculativeGetDraftParams`
void fCommonSpeculativeDraft(common_speculative *pSpec);

// informs the speculative context that n_accepted tokens were accepted by the target model
void fCommonSpeculativeAccept(common_speculative *pSpec, mgs_seq_id, uint16_t pNAccepted);

// (optional) get/set internal state
bool fCommonSpeculativeGetState(common_speculative *pSpec, mgs_seq_id pSeqId, std::vector<uint8_t> &pData);
void fCommonSpeculativeSetState(common_speculative *pSpec, mgs_seq_id pSeqId, const std::vector<uint8_t> &pData);

// print statistics about the speculative decoding
void fCommonSpeculativePrintStats(const common_speculative *pSpec);

struct common_speculative_deleter {
  void operator()(common_speculative *pS) { fCommonSpeculativeFree(pS); }
};

typedef std::unique_ptr<common_speculative, common_speculative_deleter> common_speculative_ptr;

struct common_speculative_init_result {
  common_speculative_init_result(common_params &pParams, mgs_model *pModelTgt, mgs_context *pCtxTgt,
                                 mgs_model_ptr pDraftModel);
  ~common_speculative_init_result();

  mgs_model *mModel();
  mgs_context *mContext();

private:
  struct impl;
  std::unique_ptr<impl> vPimpl;
};

using common_speculative_init_result_ptr = std::unique_ptr<common_speculative_init_result>;

common_speculative_init_result_ptr fCommonSpeculativeInitFromParams(common_params &pParams, mgs_model *pModelTgt,
                                                                    mgs_context *pCtxTgt, mgs_model_ptr pDraftModel);
