#pragma once

#include "mgs.h"

#include <cstdint>
#include <vector>

#define MGS_MAX_SEQ 256

struct mgs_cparams {
  uint32_t vNCtx;    // context size used during inference
  uint32_t vNCtxSeq; // context for a single sequence
  uint32_t vNBatch;
  uint32_t vNUbatch;
  uint32_t vNSeqMax;
  uint32_t vNRsSeq;       // number of recurrent-state snapshots per seq for rollback
  uint32_t vNOutputsMax;  // max outputs supported by the context
  int32_t vNThreads;      // number of threads to use for generation
  int32_t vNThreadsBatch; // number of threads to use for batch processing

  int32_t vNextnLayerOffset = 0;

  float vRopeFreqBase;
  float vRopeFreqScale;

  uint32_t vNCtxOrigYarn;
  // These hyperparameters are not exposed in GGUF, because all
  // existing YaRN models use the same values for them.
  float vYarnExtFactor;
  float vYarnAttnFactor;
  float vYarnBetaFast;
  float vYarnBetaSlow;

  bool vEmbeddings;
  bool vEmbeddingsNextn;       // also extract the hidden state before the final output norm
  bool vEmbeddingsNextnMasked; // extract for only rows where batch.logits != 0
  bool vCausalAttn;
  bool vOffloadKqv;
  bool vFlashAttn;
  bool vAutoFa;
  bool vFusedGdnAr; // use fused gated delta net (autoregressive)
  bool vFusedGdnCh; // use fused gated delta net (chunked)
  bool vAutoFgdn;
  bool vFusedLid; // use fused lightning indexer
  bool vAutoFlid;
  bool vFusedDsv4HcPre;
  bool vFusedDsv4HcComb;
  bool vFusedDsv4HcPost;
  bool vAutoFhc;
  bool vNoPerf;
  bool vOpOffload;
  bool vKvUnified;
  bool vPipelineParallel;

  std::vector<bool> lEmbeddingsLayerInp; // [n_layer()] extract input embeddings for layer

  enum mgs_context_type vCtxType;
  enum mgs_pooling_type vPoolingType;

  ggml_backend_sched_eval_callback vCbEval;
  void *vCbEvalUserData;

  mgs_context *vCtxOther;
};
