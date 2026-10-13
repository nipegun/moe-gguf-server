#pragma once

#include "mgs.h"
#include "mgs-ext.h"
#include "mgs-cparams.h"
#include "mgs-graph.h"
#include "mgs-adapter.h"
#include "mgs-impl.h"
#include "mgs-memory.h"
#include "mgs-expert-cache.h"

#include "ggml-cpp.h"
#include "ggml-opt.h"

#include <array>
#include <map>
#include <vector>

struct mgs_model;
class mgs_batch_allocr;

class mgs_io_read_i;
class mgs_io_write_i;

// "memory" as in abstract memory for the context
struct mgs_memory_i;
struct mgs_memory_context_i;

// stores copy of the memory in device buffer. used for fast state save/load
struct mgs_memory_buffer {
  int vNTensors = 0;
  size_t vTotalSize = 0;

  ggml_backend_buffer_ptr vBuf;

  ggml_context_ptr vCtx;

  std::vector<ggml_tensor *> lOrg;
  std::vector<ggml_tensor *> lCpy;
};

using mgs_memory_buffers = std::map<ggml_backend_buffer_type_t, mgs_memory_buffer>;

struct mgs_context {
  // init scheduler and compute buffers, reserve worst-case graphs
  mgs_context(const mgs_model &pModel, mgs_context_params pParams);

  ~mgs_context();

  // reserve a new backend scheduler (if needed)
  // for example, when:
  //   - changing loras
  //   - changing samplers
  //   - changing attention type
  //   - etc.
  void mSchedReserve();

  void mSynchronize();

  const mgs_model &mGetModel() const;
  const mgs_cparams &mGetCparams() const;

  ggml_backend_sched_t mGetSched() const;

  uint32_t mNCtx() const;
  uint32_t mNCtxSeq() const;
  uint32_t mNBatch() const;
  uint32_t mNUbatch() const;
  uint32_t mNSeqMax() const;

  uint32_t mNThreads() const;
  uint32_t mNThreadsBatch() const;

  mgs_memory_t mGetMemory() const;

  // return true if the memory was updated
  bool mMemoryUpdate(bool pOptimize);

  enum mgs_pooling_type mPoolingType() const;

  float *mGetLogits();
  float *mGetLogitsIth(int32_t pI);

  float *mGetEmbeddings();
  float *mGetEmbeddingsIth(int32_t pI);
  float *mGetEmbeddingsSeq(mgs_seq_id pSeqId);

  float *mGetEmbeddingsNextn();
  float *mGetEmbeddingsNextnIth(int32_t pI);

  float *mGetEmbeddingsLayerInp(uint32_t pLid);

  mgs_token *mGetSampledTokens() const;
  mgs_token mGetSampledTokenIth(int32_t pIdx);

  float *mGetSampledLogitsIth(int32_t pIdx);
  size_t mGetSampledLogitsCount(int32_t pIdx);

  float *mGetSampledProbsIth(int32_t pIdx);
  size_t mGetSampledProbsCount(int32_t pIdx);

  const mgs_token *mGetSampledCandidatesIth(int32_t pIdx);
  size_t mGetSampledCandidatesCount(int32_t pIdx);

  void mAttachThreadpool(ggml_threadpool_t pThreadpool, ggml_threadpool_t pThreadpoolBatch);

  void mDetachThreadpool();

  void mSetNThreads(int32_t pNThreads, int32_t pNThreadsBatch);

  void mSetAbortCallback(bool (*pAbortCallback)(void *pData), void *pAbortCallbackData);

  void mSetEmbeddings(bool pValue);
  void mSetEmbeddingsNextn(bool pValue, bool pMasked);
  void mSetEmbeddingsLayerInp(uint32_t pLid, bool pEnable);
  void mSetNextnLayerOffset(int32_t pOffset);
  void mSetCausalAttn(bool pValue);

  // MoE expert cache monitoring and run-time size requests (applied by the next decode)
  mgs_expert_cache_info mGetExpertCacheInfo() const;
  void mSetExpertCacheSize(int64_t pSizeMib);
  mgs_expert_memory_breakdown mGetExpertMemoryBreakdown() const;

  void mSetAdaptersLora(mgs_adapter_lora **pAdapters, size_t pNAdapters, float *pScales);

  bool mAdaptersLoraAreSame(mgs_adapter_lora **pAdapters, size_t pNAdapters, float *pScales);

  bool mSetAdapterCvec(const float *pData, size_t pLen, int32_t pNEmbd, int32_t pIlStart, int32_t pIlEnd);

  // process a single ubatch with a specific graph type
  // if memory_context is provided, it will be applied first to the context's memory
  // ret contains the status of the graph computation
  // returns nullptr only if ret != GGML_STATUS_SUCCESS
  llm_graph_result *mProcessUbatch(const mgs_ubatch &pUbatch, llm_graph_type pGtype, mgs_memory_context_i *pMctx,
                                   ggml_status &pRet);

  int mEncode(const mgs_batch &pBatchInp);
  int mDecode(const mgs_batch &pBatchInp);

  //
  // state save/load
  //

  size_t mStateGetSize();
  size_t mStateGetData(uint8_t *pDst, size_t pSize);
  size_t mStateSetData(const uint8_t *pSrc, size_t pSize);

  size_t mStateSeqGetSize(mgs_seq_id pSeqId, mgs_state_seq_flags pFlags);

  size_t mStateSeqGetData(mgs_seq_id pSeqId, uint8_t *pDst, size_t pSize, mgs_state_seq_flags pFlags);
  size_t mStateSeqSetData(mgs_seq_id pSeqId, const uint8_t *pSrc, size_t pSize, mgs_state_seq_flags pFlags);

  bool mStateLoadFile(const char *pFilepath, mgs_token *pTokensOut, size_t pNTokenCapacity, size_t *pNTokenCountOut);

  bool mStateSaveFile(const char *pFilepath, const mgs_token *pTokens, size_t pNTokenCount);

  size_t mStateSeqLoadFile(mgs_seq_id pSeqId, const char *pFilepath, mgs_token *pTokensOut, size_t pNTokenCapacity,
                           size_t *pNTokenCountOut);

  size_t mStateSeqSaveFile(mgs_seq_id pSeqId, const char *pFilepath, const mgs_token *pTokens, size_t pNTokenCount);

  //
  // perf
  //

  mgs_perf_context_data mPerfGetData() const;
  void mPerfReset();

  mgs_memory_breakdown mMemoryBreakdown() const;

  //
  // training
  //

  void mOptInit(struct mgs_model *pModel, struct mgs_opt_params pLoptParams);

  // TODO: more flexible combinations of logical/physical batch size and context size
  void mOptEpoch(ggml_opt_dataset_t pDataset, ggml_opt_result_t pResultTrain, ggml_opt_result_t pResultEval,
                 int64_t pIdataSplit, ggml_opt_epoch_callback pCallbackTrain, ggml_opt_epoch_callback pCallbackEval);

  void mOptEpochIter(ggml_opt_dataset_t pDataset, ggml_opt_result_t pResult, const std::vector<mgs_token> &pTokens,
                     const std::vector<mgs_token> &pLabelsSparse, mgs_batch &pBatch, ggml_opt_epoch_callback pCallback,
                     bool pTrain, int64_t pIdataInLoop, int64_t pNdataInLoop, int64_t pTLoopStart);

private:
  //
  // output
  //

  // Make sure enough space is available for outputs.
  // Returns max number of outputs for which space was reserved.
  uint32_t mOutputReserve(int32_t pNOutputs);

  void mOutputReorder();

  // map the output row index `i` to batch index
  int64_t mOutputResolveRow(int32_t pI) const;

  // async-copy enabled layer-input tensors (per cparams.output_layer_inp)
  // from backend into host-side embd_layer_inp buffers
  bool mExtractLayerInputs(const llm_graph_result *pRes, size_t pTokenOffset, size_t pNTokens);

  //
  // graph
  //

public:
  uint32_t mGraphMaxNodes(uint32_t pNTokens) const;

  // can reuse the llm_graph_result instance of the context (for example to update a memory module)
  llm_graph_result *mGetGfResReserve() const;

  // returns the result of ggml_backend_sched_graph_compute_async execution
  ggml_status mGraphCompute(ggml_cgraph *pGf, bool pBatched);

  // reserve a graph with a dummy ubatch of the specified size
  ggml_cgraph *mGraphReserve(uint32_t pNTokens, uint32_t pNSeqs, uint32_t pNOutputs, const mgs_memory_context_i *pMctx,
                             bool pSplitOnly = false, size_t *pSizes = nullptr);

  bool mSetSampler(mgs_seq_id pSeqId, mgs_sampler *pSampler);

private:
  llm_graph_params mGraphParams(llm_graph_result *pRes, const mgs_ubatch &pUbatch, const mgs_memory_context_i *pMctx,
                                llm_graph_type pGtype) const;

  llm_graph_result *mGetGfResPrev();

  llm_graph_cb mGraphGetCb() const;

  // disable auto fused ops (Flash Attention, Gated Delta Net) whose op lands on a device
  // that differs from the layer it belongs to (usually due to missing backend support)
  void mResolveFusedOps(const mgs_memory_context_i *pMctx, uint32_t pNSeqs);

  // TODO: read/write lora adapters and cvec
  size_t mStateWriteData(mgs_io_write_i &pIo);
  size_t mStateReadData(mgs_io_read_i &pIo);

  size_t mStateSeqWriteData(mgs_io_write_i &pIo, mgs_seq_id pSeqId, mgs_state_seq_flags pFlags);
  size_t mStateSeqReadData(mgs_io_read_i &pIo, mgs_seq_id pSeqId, mgs_state_seq_flags pFlags);

  //
  // members
  //

  const mgs_model &vModel;

  mgs_cparams vCparams;

  mgs_adapter_cvec_ptr vCvec;
  mgs_adapter_loras_ptr vLoras;

  mgs_cross vCross; // TODO: tmp for handling cross-attention - need something better probably

  mgs_memory_ptr vMemory;

  // decode output (2-dimensional array: [n_outputs][n_vocab])
  buffer_view<float> vLogits = {nullptr, 0};

  // embeddings output (2-dimensional array: [n_outputs][n_embd])
  // populated only when pooling_type == MGS_POOLING_TYPE_NONE
  buffer_view<float> vEmbd = {nullptr, 0};

  // hidden state required by the nextn layers (2-dimensional array: [n_outputs][n_embd])
  // populated only when cparams.vEmbeddingsNextn is enabled and the model graph
  // sets llm_graph_result::vTHNextn
  buffer_view<float> vEmbdNextn = {nullptr, 0};

  // host buffers for output layer input embeddings, per layer
  // populated when cparams.output_layer_inp[il] is true
  std::vector<buffer_view<float>> lEmbdLayerInp;
  std::vector<int32_t> lEmbdBatchIdxs; // extracted index -> original batch index

  struct sampling_info {
    // !samplers.empty() to check if any samplers are active
    std::map<mgs_seq_id, mgs_sampler *> dSamplers;

    buffer_view<float> vLogits = {nullptr, 0};
    buffer_view<mgs_token> vSampled = {nullptr, 0};
    buffer_view<float> vProbs = {nullptr, 0};
    buffer_view<mgs_token> vCandidates = {nullptr, 0};

    std::vector<uint32_t> lLogitsCount;
    std::vector<uint32_t> lProbsCount;
    std::vector<uint32_t> lCandidatesCount;

    // optimization
    std::vector<mgs_token> lTokenIdsFullVocab;
  };

  sampling_info vSampling;

  // sequence embeddings output (map of [n_embd] vectors)
  // populated only when pooling_type != MGS_POOLING_TYPE_NONE
  std::map<mgs_seq_id, std::vector<float>> dEmbdSeq;

  // reuse the batch_allocr to avoid unnecessary memory allocations
  std::unique_ptr<mgs_batch_allocr> vBalloc;

  uint32_t vNOutputs = 0; // number of actually-used outputs in the current ubatch or last logical batch

  std::vector<int32_t> lOutputIds; // map batch token positions to ids of the logits and embd buffers

  struct swap_info {
    uint32_t vI0;
    uint32_t vI1;
  };

  std::vector<swap_info> lOutputSwaps;

  ggml_backend_sched_ptr vSched;

  bool vSchedNeedReserve = true;

  ggml_backend_t vBackendCpu = nullptr;
  std::vector<ggml_backend_ptr> lBackends;

  // device cache for MoE experts kept in host memory; created at the first regular
  // decode, once every context sharing the device has reserved its memory
  std::unique_ptr<mgs_expert_cache> vExpertCache;
  bool vExpertCacheChecked = false;
  int64_t vExpertCacheRequestMib = -2; // -2: no pending size request

  // training
  ggml_opt_context_t vOptCtx = nullptr;

  ggml_threadpool_t vThreadpool = nullptr;
  ggml_threadpool_t vThreadpoolBatch = nullptr;

  ggml_abort_callback vAbortCallback = nullptr;
  void *vAbortCallbackData = nullptr;

  std::vector<std::pair<ggml_backend_t, ggml_backend_set_n_threads_t>> lSetNThreadsFns;

  // pointers and buffer types used for the compute buffer of each backend
  std::vector<ggml_backend_t> lBackendPtrs;
  std::vector<ggml_backend_buffer_type_t> lBackendBuft;
  std::vector<size_t> lBackendBufExpSize; // expected buffer sizes

  // Separate arenas keep CUDA graph cache keys stable for MTP batches with and without outputs.
  std::array<llm_graph_result_ptr, 2> aGfResPrev;
  llm_graph_result_ptr vGfResReserve;
  llm_graph_result *vGfResPrevActive = nullptr;

  // host buffer for the model output (logits and embeddings)
  ggml_backend_buffer_ptr vBufOutput;

  // keep copies of the per-sequence memory on the device
  std::map<mgs_seq_id, mgs_memory_buffers> dMemStorage;

  bool vHasEvaluatedOnce = false;

  // env: MGS_GRAPH_REUSE_DISABLE
  bool vGraphReuseDisable = false;

  // perf
  mutable int64_t vTStartUs = 0;
  mutable int64_t vTLoadUs = 0;
  mutable int64_t vTPEvalUs = 0;
  mutable int64_t vTEvalUs = 0;

  mutable int64_t vTComputeStartUs = 0;
  mutable int64_t vNQueuedTokens = 0;

  mutable int32_t vNPEval = 0; // number of tokens in eval calls for the prompt (with batch size > 1)
  mutable int32_t vNEval = 0;  // number of eval calls

  mutable int32_t vNReused = 0; // number of times the previous graph was reused
};
