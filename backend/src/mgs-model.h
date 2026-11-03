#pragma once

#include "mgs.h"
#include "mgs-arch.h"
#include "mgs-graph.h"
#include "mgs-hparams.h"
#include "mgs-memory.h"
#include "mgs-vocab.h"

#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

struct mgs_cparams;
struct mgs_ubatch;
struct mgs_model_loader;

// available models
enum llm_type {
  cLlmTypeUnknown,
  cLlmType31bA3Point5B,
  cLlmType35bA3b, // Qwen3.5
  cLlmTypeA3b,     // Qwen3.8 Flash Next
  cLlmType120bA12b, // Nemotron 3 Super
  cLlmType122bA10b, // Qwen3.5
  cLlmType428bA23b, // Minimax M3
  cLlmType310bA15b, // /MiMo-V2-Flash
  cLlmType320bA18b, // GLM-5.3-Flash
  cLlmType397bA17b, // Qwen3.5
  cLlmType744bA40b, // GLM-5
  cLlmType2Point8tA50b, // Kimi-K3
};

std::string mgs_rope_scaling_type_name(mgs_rope_scaling_type pRopeScalingType);

struct mgs_layer_nextn {
  struct ggml_tensor *vEhProj = nullptr;
  struct ggml_tensor *vEhProjS = nullptr;
  struct ggml_tensor *vEhProjInS = nullptr;
  struct ggml_tensor *vEmbedTokens = nullptr;
  struct ggml_tensor *vEnorm = nullptr;
  struct ggml_tensor *vHnorm = nullptr;
  struct ggml_tensor *vSharedHeadHead = nullptr;
  struct ggml_tensor *vSharedHeadHeadS = nullptr;
  struct ggml_tensor *vSharedHeadHeadInS = nullptr;
  struct ggml_tensor *vSharedHeadNorm = nullptr;

  // qwen4exp: the MTP block collapses its hyper-connection streams with its own mixer
  struct ggml_tensor *vHcHeadNorm = nullptr;
  struct ggml_tensor *vHcHeadDown = nullptr;
  struct ggml_tensor *vHcHeadUp = nullptr;
};

struct mgs_layer {
  // normalization
  struct ggml_tensor *vAttnNorm = nullptr;
  struct ggml_tensor *vAttnQNorm = nullptr;
  struct ggml_tensor *vAttnKNorm = nullptr;
  struct ggml_tensor *vAttnOutNorm = nullptr;
  struct ggml_tensor *vAttnQANorm = nullptr;
  struct ggml_tensor *vAttnKvANorm = nullptr;
  struct ggml_tensor *vAttnPostNorm = nullptr;
  struct ggml_tensor *vSsmNorm = nullptr;
  struct ggml_tensor *vSsmDtNorm = nullptr;
  struct ggml_tensor *vSsmBNorm = nullptr;
  struct ggml_tensor *vSsmCNorm = nullptr;

  // attention
  struct ggml_tensor *vWq = nullptr;
  struct ggml_tensor *vWk = nullptr;
  struct ggml_tensor *vWv = nullptr;
  struct ggml_tensor *vWo = nullptr;
  struct ggml_tensor *vWqkv = nullptr;
  struct ggml_tensor *vWqA = nullptr;
  struct ggml_tensor *vWqB = nullptr;
  struct ggml_tensor *vWkvAMqa = nullptr;
  struct ggml_tensor *vWkv = nullptr;
  struct ggml_tensor *vWkB = nullptr;
  struct ggml_tensor *vWvB = nullptr;
  struct ggml_tensor *vWqkvB = nullptr;
  struct ggml_tensor *vWoA = nullptr;
  struct ggml_tensor *vWoB = nullptr;
  struct ggml_tensor *vWqkvGate = nullptr;

  // normalization
  struct ggml_tensor *vFfnNorm = nullptr;
  struct ggml_tensor *vLayerOutNorm = nullptr;

  // ff
  struct ggml_tensor *vFfnGate = nullptr; // w1
  struct ggml_tensor *vFfnDown = nullptr; // w2
  struct ggml_tensor *vFfnUp = nullptr;   // w3

  // ff MoE
  struct ggml_tensor *vFfnGateInp = nullptr;
  struct ggml_tensor *vFfnGateExps = nullptr;
  struct ggml_tensor *vFfnDownExps = nullptr;
  struct ggml_tensor *vFfnUpExps = nullptr;
  struct ggml_tensor *vFfnGateUpExps = nullptr;
  struct ggml_tensor *vFfnGateExpsB = nullptr;
  struct ggml_tensor *vFfnDownExpsB = nullptr;
  struct ggml_tensor *vFfnUpExpsB = nullptr;
  struct ggml_tensor *vFfnGateUpExpsB = nullptr;

  // ff MoE per-expert scales (NVFP4 per-tensor scale2)
  struct ggml_tensor *vFfnGateExpsS = nullptr;
  struct ggml_tensor *vFfnDownExpsS = nullptr;
  struct ggml_tensor *vFfnUpExpsS = nullptr;

  // ff MoE latent proj
  struct ggml_tensor *vFfnLatentDown = nullptr;
  struct ggml_tensor *vFfnLatentUp = nullptr;

  // ff shared expert (shexp)
  struct ggml_tensor *vFfnGateInpShexp = nullptr;
  struct ggml_tensor *vFfnGateShexp = nullptr;
  struct ggml_tensor *vFfnDownShexp = nullptr;
  struct ggml_tensor *vFfnUpShexp = nullptr;

  // ff adjugate experts (chexps)

  // ff bias
  struct ggml_tensor *vFfnGateB = nullptr;
  struct ggml_tensor *vFfnDownB = nullptr; // b2
  struct ggml_tensor *vFfnUpB = nullptr;   // b3
  struct ggml_tensor *vFfnExpProbsB = nullptr;
  struct ggml_tensor *vFfnExpProbsBVl = nullptr; // routing bias for image tokens (DeepSeek-V4.1)

  // mamba proj
  struct ggml_tensor *vSsmIn = nullptr;
  struct ggml_tensor *vSsmX = nullptr;
  struct ggml_tensor *vSsmDt = nullptr;
  struct ggml_tensor *vSsmOut = nullptr;

  // mamba
  struct ggml_tensor *vSsmConv1d = nullptr;
  struct ggml_tensor *vSsmA = nullptr;
  struct ggml_tensor *vSsmD = nullptr;

  // mamba bias
  struct ggml_tensor *vSsmConv1dB = nullptr;
  struct ggml_tensor *vSsmDtB = nullptr;

  // qwen3next

  // qwen3.5
  struct ggml_tensor *vSsmAlpha = nullptr;

  // long rope factors
  struct ggml_tensor *vRopeLong = nullptr;
  struct ggml_tensor *vRopeShort = nullptr;
  struct ggml_tensor *vRopeFreqs = nullptr;

  // bitnet scale
  struct ggml_tensor *vWqS = nullptr;
  struct ggml_tensor *vWkS = nullptr;
  struct ggml_tensor *vWvS = nullptr;
  struct ggml_tensor *vWoS = nullptr;
  struct ggml_tensor *vWqkvS = nullptr;
  struct ggml_tensor *vWqkvGateS = nullptr;
  struct ggml_tensor *vFfnGateS = nullptr;
  struct ggml_tensor *vFfnUpS = nullptr;
  struct ggml_tensor *vFfnDownS = nullptr;
  struct ggml_tensor *vFfnGateShexpS = nullptr;
  struct ggml_tensor *vFfnUpShexpS = nullptr;
  struct ggml_tensor *vFfnDownShexpS = nullptr;
  struct ggml_tensor *vSsmInS = nullptr;
  struct ggml_tensor *vSsmOutS = nullptr;
  struct ggml_tensor *vSsmAlphaS = nullptr;
  struct ggml_tensor *vSsmBetaS = nullptr;

  // input scales
  struct ggml_tensor *vWqInS = nullptr;
  struct ggml_tensor *vWkInS = nullptr;
  struct ggml_tensor *vWvInS = nullptr;
  struct ggml_tensor *vWoInS = nullptr;
  struct ggml_tensor *vWqkvInS = nullptr;
  struct ggml_tensor *vWqkvGateInS = nullptr;
  struct ggml_tensor *vFfnGateInS = nullptr;
  struct ggml_tensor *vFfnUpInS = nullptr;
  struct ggml_tensor *vFfnDownInS = nullptr;
  struct ggml_tensor *vFfnGateExpsInS = nullptr;
  struct ggml_tensor *vFfnDownExpsInS = nullptr;
  struct ggml_tensor *vFfnUpExpsInS = nullptr;
  struct ggml_tensor *vFfnGateShexpInS = nullptr;
  struct ggml_tensor *vFfnUpShexpInS = nullptr;
  struct ggml_tensor *vFfnDownShexpInS = nullptr;
  struct ggml_tensor *vSsmInInS = nullptr;
  struct ggml_tensor *vSsmOutInS = nullptr;
  struct ggml_tensor *vSsmAlphaInS = nullptr;
  struct ggml_tensor *vSsmBetaInS = nullptr;

  // openai-moe
  struct ggml_tensor *vAttnSinks = nullptr;

  // DeepSeek-V4.1 (the hyper-connection tensors are shared with other models)
  struct ggml_tensor *vAttnKvNorm = nullptr;
  struct ggml_tensor *vHcAttnFn = nullptr;
  struct ggml_tensor *vHcAttnBase = nullptr;
  struct ggml_tensor *vHcAttnScale = nullptr;
  struct ggml_tensor *vHcFfnFn = nullptr;
  struct ggml_tensor *vHcFfnBase = nullptr;
  struct ggml_tensor *vHcFfnScale = nullptr;

  // qwen4exp low-rank hyper-connection mixers, one before the token mixer and one before the MoE
  struct ggml_tensor *vHcAttnNorm = nullptr;
  struct ggml_tensor *vHcAttnDown = nullptr;
  struct ggml_tensor *vHcAttnUp = nullptr;
  struct ggml_tensor *vHcAttnInject = nullptr;
  struct ggml_tensor *vHcFfnNorm = nullptr;
  struct ggml_tensor *vHcFfnDown = nullptr;
  struct ggml_tensor *vHcFfnUp = nullptr;
  struct ggml_tensor *vHcFfnInject = nullptr;

  // qwen4exp per-layer n-gram hash embedding module
  struct ggml_tensor *vPleKey = nullptr;
  struct ggml_tensor *vPleValue = nullptr;
  struct ggml_tensor *vPleNormKey = nullptr;
  struct ggml_tensor *vPleNormQuery = nullptr;
  struct ggml_tensor *vPleNormConv = nullptr;
  struct ggml_tensor *vPleConv1d = nullptr;
  struct ggml_tensor *vAttnCompWkv = nullptr;
  struct ggml_tensor *vAttnCompWgate = nullptr;
  struct ggml_tensor *vAttnCompNorm = nullptr;

  // DeepSeek-V4.1 engram (the n-gram table itself is read from the model file on demand)
  struct ggml_tensor *vEngramWkv = nullptr;
  struct ggml_tensor *vEngramQ = nullptr;
  struct ggml_tensor *vEngramK = nullptr;

  // Kimi Linear KDA (using ssm_ prefix for consistency)
  // Note: ssm_dt_b already exists above (mamba bias), reused for Kimi dt_bias
  struct ggml_tensor *vSsmQConv = nullptr;
  struct ggml_tensor *vSsmKConv = nullptr;
  struct ggml_tensor *vSsmVConv = nullptr;
  struct ggml_tensor *vSsmFA = nullptr;
  struct ggml_tensor *vSsmFB = nullptr;
  struct ggml_tensor *vSsmBeta = nullptr;
  struct ggml_tensor *vSsmGA = nullptr;
  struct ggml_tensor *vSsmGB = nullptr;
  struct ggml_tensor *vSsmONorm = nullptr;

  // kimi-k3
  struct ggml_tensor *vSsmG = nullptr;          // full-rank KDA gate (replaces ssm_g_a/ssm_g_b)
  struct ggml_tensor *vAttnResScore = nullptr;  // fused res_norm*res_proj, pre-attention
  struct ggml_tensor *vFfnResScore = nullptr;   // fused res_norm*res_proj, pre-FFN
  struct ggml_tensor *vFfnRoutedDown = nullptr; // latent MoE: n_embd -> n_expert_latent
  struct ggml_tensor *vFfnRoutedUp = nullptr;   // latent MoE: n_expert_latent -> n_embd
  struct ggml_tensor *vFfnRoutedNorm = nullptr;

  // DSA (deepseek sparse attention)
  struct ggml_tensor *vIndexerKNorm = nullptr;
  struct ggml_tensor *vIndexerKNormB = nullptr;
  struct ggml_tensor *vIndexerProj = nullptr;
  struct ggml_tensor *vIndexerAttnK = nullptr;
  struct ggml_tensor *vIndexerAttnQB = nullptr; // note: for lora a/b, not bias

  // glm5-next k-pool indexer
  struct ggml_tensor *vIndexerKpoolGate = nullptr;
  struct ggml_tensor *vIndexerKpoolApe = nullptr;

  // MSA
  struct ggml_tensor *vIndexQProj = nullptr;
  struct ggml_tensor *vIndexKProj = nullptr;
  struct ggml_tensor *vIndexQNorm = nullptr;
  struct ggml_tensor *vIndexKNorm = nullptr;


  struct mgs_layer_nextn vNextn;
};

struct mgs_device {
  ggml_backend_dev_t vDev;
};

struct mgs_meta_device_get_split_state_userdata {
  size_t vNDevices;
  const struct mgs_model *vModel;
};

struct ggml_backend_meta_split_state mgs_meta_device_get_split_state(const struct ggml_tensor *pTensor,
                                                                     void *pUserdata);

struct mgs_model {
  llm_type vType = cLlmTypeUnknown;
  llm_arch vArch = cLlmArchUnknown;

  std::string vName = "n/a";

  mgs_hparams vHparams = {};
  mgs_vocab vVocab;

  // for classifier models
  std::vector<std::string> lClassifierLabels;

  struct ggml_tensor *vTokEmbd = nullptr;
  struct ggml_tensor *vTypeEmbd = nullptr;
  struct ggml_tensor *vTokNorm = nullptr;
  struct ggml_tensor *vTokNormB = nullptr;

  struct ggml_tensor *vOutputNorm = nullptr;
  struct ggml_tensor *vOutputResScore = nullptr; // kimi-k3: final cross-layer residual mix
  struct ggml_tensor *vOutputNormB = nullptr;
  struct ggml_tensor *vOutput = nullptr;
  struct ggml_tensor *vOutputB = nullptr;

  // NVFP4 per-tensor scale2, input_scale for LM head
  struct ggml_tensor *vOutputS = nullptr;
  struct ggml_tensor *vOutputInS = nullptr;

  // qwen4exp: the final hyper-connection mixer, which is also the output norm
  struct ggml_tensor *vHcHeadNorm = nullptr;
  struct ggml_tensor *vHcHeadDown = nullptr;
  struct ggml_tensor *vHcHeadUp = nullptr;

  // qwen4exp per-layer token embeddings
  struct ggml_tensor *vPerLayerTokEmbd = nullptr;

  std::vector<mgs_layer> lLayers;

  // Dense linear projection applied after pooling (lfm2moe)
  //  For Sentence Transformers models structure see
  //  https://sbert.net/docs/sentence_transformer/usage/custom_models.html#structure-of-sentence-transformer-models
  struct ggml_tensor *vDense2OutLayers = nullptr;
  struct ggml_tensor *vDense2OutLayersB = nullptr;

  // gguf metadata
  std::unordered_map<std::string, std::string> dGgufKv;

  // list of devices used in this model
  std::vector<mgs_device> lDevices;

  // for quantize-stats only
  std::vector<std::pair<std::string, struct ggml_tensor *>> lTensorsByName;

  // for keeping track of associated LoRA adapters
  std::unordered_set<mgs_adapter_lora *> sLoras;

  // statically allocated context for assigning
  struct mgs_meta_device_get_split_state_userdata vGetSplitStateUd;

  int64_t vTLoadUs = 0;
  int64_t vTStartUs = 0;

  explicit mgs_model(const mgs_model_params &pParams);
  virtual ~mgs_model();

  std::string mArchName() const;
  std::string mTypeName() const;

  std::string mDesc() const;

  mgs_ftype mFtype() const;

  size_t size() const; // file size
  size_t mNTensors() const;
  size_t mNDevices() const;
  const float *mTensorSplit() const;

  uint32_t mNGpuLayers() const;
  mgs_split_mode mSplitMode() const;

  std::map<ggml_backend_buffer_type_t, size_t> mMemoryBreakdown() const;

  // total number of parameters in the model
  uint64_t mNElements() const;

  void mPrintInfo() const;

  ggml_backend_dev_t mDevLayer(int pIl) const;
  ggml_backend_dev_t mDevOutput() const;

  ggml_backend_buffer_type_t mSelectBuft(int pIl) const;

  bool mHasTensorOverrides() const;

  const struct ggml_tensor *mGetTensor(const char *pName) const;

  float mGetRopeFreqBase(const mgs_cparams &pCparams, int pIl) const;
  float mGetRopeFreqScale(const mgs_cparams &pCparams, int pIl) const;

  ggml_tensor *mGetRopeFactors(const mgs_cparams &pCparams, int pIl) const;

  mgs_memory_i *mCreateMemory(const mgs_memory_params &pParams, const mgs_cparams &pCparams) const;

  ggml_cgraph *mBuildGraph(const llm_graph_params &pParams) const;

  virtual void mLoadStats(mgs_model_loader &pMl) = 0;
  virtual void mLoadHparams(mgs_model_loader &pMl) = 0;
  virtual void mLoadVocab(mgs_model_loader &pMl) = 0;
  virtual bool mLoadTensors(mgs_model_loader &pMl) = 0; // returns false if cancelled by progress_callback

  // model must define these
  virtual void mLoadArchHparams(mgs_model_loader &pMl) = 0;
  virtual void mLoadArchTensors(mgs_model_loader &pMl) = 0;
  virtual std::unique_ptr<llm_graph_context> mBuildArchGraph(const llm_graph_params &pParams) const = 0;

protected:
  mgs_model_params vParams;

  struct impl;
  std::unique_ptr<impl> vPimpl;
};

mgs_model *mgs_model_create(llm_arch pArch, const mgs_model_params &pParams);
mgs_model *mgs_model_create(mgs_model_loader &pMl, const mgs_model_params &pParams);

// mgs only loads Mixture of Experts models. A separate MTP head loaded as MTP is the exception, because a dense
// prediction head can serve a MoE target; it must really be a head-only file: every block is an MTP block, or the
// file has no trunk.
bool fMgsModelIsMoe(uint32_t pExpertCount, uint32_t pLayerCount, uint32_t pMtpLayerCount, bool pHasTrunk,
                    bool pLoadMtp);

// model must inherit from this
struct mgs_model_base : public mgs_model {
  friend struct mgs_model;

  mgs_model *vModel;
  mgs_model_loader *vMl = nullptr;
  const LLM_TN cTn;

  // mgs_model_loader is not yet defined at this point, so we will set it after construction
  const int cTensorDuplicated;
  const int cTensorNotRequired;
  const int cTensorSkip;
  const int cTensorSkipIfVirtual;
  const int cTensorAllowReshape;

  explicit mgs_model_base(const mgs_model_params &pParams);
  virtual ~mgs_model_base() = default;

  ggml_tensor *mCreateTensor(mgs_model_loader &pMl, const LLM_TN_IMPL &pTn, const std::initializer_list<int64_t> &pNe,
                             int pFlags);

  // convenience overload of create_tensor that doesn't require mgs_model_loader
  ggml_tensor *mCreateTensor(const LLM_TN_IMPL &pTn, const std::initializer_list<int64_t> &pNe, int pFlags);

  // helper: try merged gate_up_exps first, fall back to separate gate and up
  void mCreateTensorGateUpExps(mgs_layer &pLayer, int pBid, int64_t pNEmbd, int64_t pNFf, int64_t pNExpert, int pFlags);

  // helper: try to load merged qkv first, fall back to separate q, k, v
  void mCreateTensorQkv(mgs_layer &pLayer, int pBid, int64_t pNEmbd, int64_t pNEmbdQ, int64_t pNEmbdK, int64_t pNEmbdV,
                        int pFlags);

  void mLoadStats(mgs_model_loader &pMl) override;
  void mLoadHparams(mgs_model_loader &pMl) override;
  void mLoadVocab(mgs_model_loader &pMl) override;
  bool mLoadTensors(mgs_model_loader &pMl) override;

  // model must define these
  void mLoadArchHparams(mgs_model_loader &pMl) override = 0;
  void mLoadArchTensors(mgs_model_loader &pMl) override = 0;
  std::unique_ptr<llm_graph_context> mBuildArchGraph(const llm_graph_params &pParams) const override = 0;
};

const char *fLlmTypeName(llm_type pType);

// convenience macro for loading local variables for mLoadTensors() in mgs_model_base
// note: cast to int64_t since we will use these for the tensor dimensions
#define MGS_LOAD_LOCALS                                                                                                \
  const int cNLayer = vHparams.mNLayer();                                                                              \
  GGML_UNUSED(cNLayer);                                                                                                \
  const int cNLayerAll = vHparams.vNLayerAll;                                                                          \
  GGML_UNUSED(cNLayerAll);                                                                                             \
  const int cNLayerNextn = vHparams.vNLayerNextn;                                                                      \
  GGML_UNUSED(cNLayerNextn);                                                                                           \
  const int64_t cNHead = vHparams.mNHead();                                                                            \
  GGML_UNUSED(cNHead);                                                                                                 \
  const int64_t cNHeadKv = vHparams.mNHeadKv();                                                                        \
  GGML_UNUSED(cNHeadKv);                                                                                               \
  const int64_t cNEmbd = vHparams.vNEmbd;                                                                              \
  GGML_UNUSED(cNEmbd);                                                                                                 \
  const int64_t cNEmbdKGqa = vHparams.mNEmbdKGqa();                                                                    \
  GGML_UNUSED(cNEmbdKGqa);                                                                                             \
  const int64_t cNEmbdVGqa = vHparams.mNEmbdVGqa();                                                                    \
  GGML_UNUSED(cNEmbdVGqa);                                                                                             \
  const int64_t cNEmbdHeadK = vHparams.mNEmbdHeadK();                                                                  \
  GGML_UNUSED(cNEmbdHeadK);                                                                                            \
  const int64_t cNEmbdHeadV = vHparams.mNEmbdHeadV();                                                                  \
  GGML_UNUSED(cNEmbdHeadV);                                                                                            \
  const int64_t cNFf = vHparams.mNFf();                                                                                \
  GGML_UNUSED(cNFf);                                                                                                   \
  const int64_t cNEmbdGqa = cNEmbdVGqa;                                                                                \
  GGML_UNUSED(cNEmbdGqa);                                                                                              \
  const int64_t cNVocab = vVocab.mNTokens();                                                                           \
  GGML_UNUSED(cNVocab);                                                                                                \
  const int64_t cNTokenTypes = vVocab.mNTokenTypes();                                                                  \
  GGML_UNUSED(cNTokenTypes);                                                                                           \
  const int64_t cNRot = vHparams.mNRot();                                                                              \
  GGML_UNUSED(cNRot);                                                                                                  \
  const int64_t cNExpert = vHparams.vNExpert;                                                                          \
  GGML_UNUSED(cNExpert);                                                                                               \
  const int64_t cNExpertUsed = vHparams.vNExpertUsed;                                                                  \
  GGML_UNUSED(cNExpertUsed);                                                                                           \
  const int64_t cNCtxTrain = vHparams.vNCtxTrain;                                                                      \
  GGML_UNUSED(cNCtxTrain);

// For internal test use
// TODO: remove
const std::vector<std::pair<std::string, ggml_tensor *>> &mgs_internal_get_tensor_map(const mgs_model *pModel);
