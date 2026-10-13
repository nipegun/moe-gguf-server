#pragma once

#include "mgs-model.h"
#include "mgs-graph.h"
#include "mgs-model-loader.h"

// note: almost all graphs require at least sqrtf, so include cmath globally
#include <cmath>

// L2 norm along dim 0 that matches the FLA reference: rms_norm with eps/n, rescaled by 1/sqrt(n)
static inline ggml_tensor *fBuildGdnL2Norm(ggml_context *pCtx, ggml_tensor *pX, float pEps) {
  const float cN = pX->ne[0];

  return ggml_scale(pCtx, ggml_rms_norm(pCtx, pX, pEps / cN), 1.0f / sqrtf(cN));
}

//
// base classes
//

struct llm_build_mamba_base : public llm_graph_context {
  llm_build_mamba_base(const llm_graph_params &pParams);

  virtual ~llm_build_mamba_base() = default;

  ggml_tensor *mBuildMambaLayer(llm_graph_input_rs *pInp, ggml_tensor *pCur, const mgs_model &pModel,
                                const mgs_ubatch &pUbatch, int pIl);
  ggml_tensor *mBuildMamba2Layer(llm_graph_input_rs *pInp, ggml_tensor *pCur, const mgs_model &pModel,
                                 const mgs_ubatch &pUbatch, int pIl) const;
};

struct llm_build_delta_net_base : public llm_graph_context {
  llm_build_delta_net_base(const llm_graph_params &pParams);

  virtual ~llm_build_delta_net_base() = default;

  // returns pair of output and new state
  std::pair<ggml_tensor *, ggml_tensor *> mBuildDeltaNetChunking(ggml_tensor *pQ, ggml_tensor *pK, ggml_tensor *pV,
                                                                 ggml_tensor *pG, ggml_tensor *pB, ggml_tensor *pS,
                                                                 int pIl);

  // returns pair of output and new state
  std::pair<ggml_tensor *, ggml_tensor *> mBuildDeltaNetAutoregressive(ggml_tensor *pQ, ggml_tensor *pK,
                                                                       ggml_tensor *pV, ggml_tensor *pG,
                                                                       ggml_tensor *pB, ggml_tensor *pS, int pIl);

  // use the ggml_gated_delta_net fused operator (K=1; state has shape [S_v, S_v, H_v, n_seqs])
  std::pair<ggml_tensor *, ggml_tensor *> mBuildDeltaNetFused(ggml_tensor *pQ, ggml_tensor *pK, ggml_tensor *pV,
                                                              ggml_tensor *pG, ggml_tensor *pB, ggml_tensor *pS,
                                                              int pIl);

  // choose one of two implementations above based on the number of tokens
  std::pair<ggml_tensor *, ggml_tensor *> mBuildDeltaNet(ggml_tensor *pQ, ggml_tensor *pK, ggml_tensor *pV,
                                                         ggml_tensor *pG, ggml_tensor *pB, ggml_tensor *pS, int pIl);

  // read conv state from cache, concat with qkv_mixed, write back (single slot or per-token)
  // qkv_mixed: (qkv_dim, n_seq_tokens, n_seqs); returns conv_input: (kernel_size + n_seq_tokens - 1, channels, n_seqs)
  ggml_tensor *mBuildConvState(llm_graph_input_rs *pInp, ggml_tensor *pConvStatesAll, ggml_tensor *pQkvMixed,
                               int64_t pConvKernelSize, int64_t pConvChannels, int pIl);

  // run delta-net attention and write the new recurrent state(s) back to ssm_states_all
  // s: (head_v_dim, head_v_dim, num_v_heads, n_seqs); returns output: (head_v_dim, num_v_heads, n_seq_tokens, n_seqs)
  ggml_tensor *mBuildRecurrentAttn(llm_graph_input_rs *pInp, ggml_tensor *pSsmStatesAll, ggml_tensor *pQ,
                                   ggml_tensor *pK, ggml_tensor *pV, ggml_tensor *pG, ggml_tensor *pB, ggml_tensor *pS,
                                   int pIl);
};

//
// models
//

struct mgs_model_grok : public mgs_model_base {
  mgs_model_grok(const struct mgs_model_params &pParams) : mgs_model_base(pParams) {}
  void mLoadArchHparams(mgs_model_loader &pMl) override;
  void mLoadArchTensors(mgs_model_loader &pMl) override;

  struct graph : public llm_graph_context {
    graph(const mgs_model &pModel, const llm_graph_params &pParams);
  };

  std::unique_ptr<llm_graph_context> mBuildArchGraph(const llm_graph_params &pParams) const override;
};

struct mgs_model_nomic_bert_moe : public mgs_model_base {
  mgs_model_nomic_bert_moe(const struct mgs_model_params &pParams) : mgs_model_base(pParams) {}
  void mLoadArchHparams(mgs_model_loader &pMl) override;
  void mLoadArchTensors(mgs_model_loader &pMl) override;

  struct graph : public llm_graph_context {
    graph(const mgs_model &pModel, const llm_graph_params &pParams);
  };

  std::unique_ptr<llm_graph_context> mBuildArchGraph(const llm_graph_params &pParams) const override;
};

struct mgs_model_mellum : public mgs_model_base {
  mgs_model_mellum(const struct mgs_model_params &pParams) : mgs_model_base(pParams) {}
  void mLoadArchHparams(mgs_model_loader &pMl) override;
  void mLoadArchTensors(mgs_model_loader &pMl) override;

  template <bool iswa> struct graph : public llm_graph_context {
    graph(const mgs_model &pModel, const llm_graph_params &pParams);
  };

  std::unique_ptr<llm_graph_context> mBuildArchGraph(const llm_graph_params &pParams) const override;
};

struct mgs_model_llada_moe : public mgs_model_base {
  mgs_model_llada_moe(const struct mgs_model_params &pParams) : mgs_model_base(pParams) {}
  void mLoadArchHparams(mgs_model_loader &pMl) override;
  void mLoadArchTensors(mgs_model_loader &pMl) override;

  struct graph : public llm_graph_context {
    graph(const mgs_model &pModel, const llm_graph_params &pParams);
  };

  std::unique_ptr<llm_graph_context> mBuildArchGraph(const llm_graph_params &pParams) const override;
};

struct mgs_model_rnd1 : public mgs_model_base {
  mgs_model_rnd1(const struct mgs_model_params &pParams) : mgs_model_base(pParams) {}
  void mLoadArchHparams(mgs_model_loader &pMl) override;
  void mLoadArchTensors(mgs_model_loader &pMl) override;

  struct graph : public llm_graph_context {
    graph(const mgs_model &pModel, const llm_graph_params &pParams);
  };

  std::unique_ptr<llm_graph_context> mBuildArchGraph(const llm_graph_params &pParams) const override;
};

struct mgs_model_qwen2moe : public mgs_model_base {
  mgs_model_qwen2moe(const struct mgs_model_params &pParams) : mgs_model_base(pParams) {}
  void mLoadArchHparams(mgs_model_loader &pMl) override;
  void mLoadArchTensors(mgs_model_loader &pMl) override;

  struct graph : public llm_graph_context {
    graph(const mgs_model &pModel, const llm_graph_params &pParams);
  };

  std::unique_ptr<llm_graph_context> mBuildArchGraph(const llm_graph_params &pParams) const override;
};

struct mgs_model_qwen3moe : public mgs_model_base {
  mgs_model_qwen3moe(const struct mgs_model_params &pParams) : mgs_model_base(pParams) {}
  void mLoadArchHparams(mgs_model_loader &pMl) override;
  void mLoadArchTensors(mgs_model_loader &pMl) override;

  struct graph : public llm_graph_context {
    graph(const mgs_model &pModel, const llm_graph_params &pParams);
  };

  std::unique_ptr<llm_graph_context> mBuildArchGraph(const llm_graph_params &pParams) const override;
};

struct mgs_model_qwen3vlmoe : public mgs_model_base {
  mgs_model_qwen3vlmoe(const struct mgs_model_params &pParams) : mgs_model_base(pParams) {}
  void mLoadArchHparams(mgs_model_loader &pMl) override;
  void mLoadArchTensors(mgs_model_loader &pMl) override;

  struct graph : public llm_graph_context {
    graph(const mgs_model &pModel, const llm_graph_params &pParams);
  };

  std::unique_ptr<llm_graph_context> mBuildArchGraph(const llm_graph_params &pParams) const override;
};

struct mgs_model_phimoe : public mgs_model_base {
  mgs_model_phimoe(const struct mgs_model_params &pParams) : mgs_model_base(pParams) {}
  void mLoadArchHparams(mgs_model_loader &pMl) override;
  void mLoadArchTensors(mgs_model_loader &pMl) override;

  template <bool iswa> struct graph : public llm_graph_context {
    graph(const mgs_model &pModel, const llm_graph_params &pParams);
  };

  std::unique_ptr<llm_graph_context> mBuildArchGraph(const llm_graph_params &pParams) const override;
};

struct mgs_model_jamba : public mgs_model_base {
  mgs_model_jamba(const struct mgs_model_params &pParams) : mgs_model_base(pParams) {}
  void mLoadArchHparams(mgs_model_loader &pMl) override;
  void mLoadArchTensors(mgs_model_loader &pMl) override;

  struct graph : public llm_build_mamba_base {
    graph(const mgs_model &pModel, const llm_graph_params &pParams);
  };

  std::unique_ptr<llm_graph_context> mBuildArchGraph(const llm_graph_params &pParams) const override;
};

struct mgs_model_cohere2moe : public mgs_model_base {
  mgs_model_cohere2moe(const struct mgs_model_params &pParams) : mgs_model_base(pParams) {}
  void mLoadArchHparams(mgs_model_loader &pMl) override;
  void mLoadArchTensors(mgs_model_loader &pMl) override;

  struct graph : public llm_graph_context {
    graph(const mgs_model &pModel, const llm_graph_params &pParams);
  };

  struct graph_mtp : public llm_graph_context {
    graph_mtp(const mgs_model &pModel, const llm_graph_params &pParams);
  };

  std::unique_ptr<llm_graph_context> mBuildArchGraph(const llm_graph_params &pParams) const override;
};

struct mgs_model_dbrx : public mgs_model_base {
  mgs_model_dbrx(const struct mgs_model_params &pParams) : mgs_model_base(pParams) {}
  void mLoadArchHparams(mgs_model_loader &pMl) override;
  void mLoadArchTensors(mgs_model_loader &pMl) override;

  struct graph : public llm_graph_context {
    graph(const mgs_model &pModel, const llm_graph_params &pParams);
  };

  std::unique_ptr<llm_graph_context> mBuildArchGraph(const llm_graph_params &pParams) const override;
};

struct mgs_model_olmoe : public mgs_model_base {
  mgs_model_olmoe(const struct mgs_model_params &pParams) : mgs_model_base(pParams) {}
  void mLoadArchHparams(mgs_model_loader &pMl) override;
  void mLoadArchTensors(mgs_model_loader &pMl) override;

  struct graph : public llm_graph_context {
    graph(const mgs_model &pModel, const llm_graph_params &pParams);
  };

  std::unique_ptr<llm_graph_context> mBuildArchGraph(const llm_graph_params &pParams) const override;
};

struct mgs_model_arctic : public mgs_model_base {
  mgs_model_arctic(const struct mgs_model_params &pParams) : mgs_model_base(pParams) {}
  void mLoadArchHparams(mgs_model_loader &pMl) override;
  void mLoadArchTensors(mgs_model_loader &pMl) override;

  struct graph : public llm_graph_context {
    graph(const mgs_model &pModel, const llm_graph_params &pParams);
  };

  std::unique_ptr<llm_graph_context> mBuildArchGraph(const llm_graph_params &pParams) const override;
};

class llm_graph_input_dsv41;
class llm_graph_input_dsv41_engram;

// one DeepSeek-V4.1 engram table: an n-gram embedding table that stays in the model file
struct mgs_dsv41_engram_table {
  int vFd = -1;          // a private descriptor of the model file
  uint64_t vOffset = 0;  // file offset of row 0
  uint64_t vNRows = 0;
  size_t vRowBytes = 0;  // bytes of one row in the file
  int64_t vRowDim = 0;   // values per row
  ggml_type vType = GGML_TYPE_COUNT; // GGML_TYPE_I8: E4M3 values followed by E8M0 scales per 32 values
};

// DeepSeek-V4.1: hyper-connections with lagged mixes, shared compressed KV and indexer, engram tables
struct mgs_model_deepseek41 : public mgs_model_base {
  mgs_model_deepseek41(const struct mgs_model_params &pParams) : mgs_model_base(pParams) {}
  ~mgs_model_deepseek41() override;
  void mLoadArchHparams(mgs_model_loader &pMl) override;
  void mLoadArchTensors(mgs_model_loader &pMl) override;

  // engram hash constants: per engram layer, max_ngram multipliers and (max_ngram - 1) * n_head buckets
  std::vector<uint32_t> lEngramLayerIds;
  std::vector<uint32_t> lEngramTokenMap;
  std::vector<uint64_t> lEngramMultipliers;
  std::vector<uint64_t> lEngramPrimes;
  std::vector<uint64_t> lEngramOffsets;
  uint32_t vEngramPadId = 0;
  std::vector<mgs_dsv41_engram_table> lEngramTables;

  // index of a layer among the engram layers, -1 when it has none
  int mEngramIndex(int pIl) const;

  // reads pCount rows of engram layer pE as floats, pOut holds pCount * row_dim values
  void mEngramReadRows(int pE, const uint32_t *pRows, size_t pCount, float *pOut) const;

  struct graph : public llm_graph_context {
    graph(const mgs_model &pModel, const llm_graph_params &pParams);

    llm_graph_input_dsv41 *mBuildInpDsv41() const;
    llm_graph_input_dsv41_engram *mBuildInpEngram(const mgs_model_deepseek41 &pModel) const;

    // the pre / post / comb coefficients of one sublayer (the pre mix is consumed by the next sublayer)
    void mBuildHcMixes(ggml_tensor *pX, ggml_tensor *pHcFn, ggml_tensor *pHcScale, ggml_tensor *pHcBase,
                       ggml_tensor **pPre, ggml_tensor **pPost, ggml_tensor **pComb, int pIl) const;

    // collapse the hc copies with per-copy weights: [n_embd, hc, n_tokens] -> [n_embd, n_tokens]
    ggml_tensor *mBuildHcPre(ggml_tensor *pX, ggml_tensor *pWeights, int pIl) const;

    ggml_tensor *mBuildHcPost(ggml_tensor *pX, ggml_tensor *pResidual, ggml_tensor *pPost, ggml_tensor *pComb,
                              int pIl) const;

    ggml_tensor *mBuildHcSinkhorn(ggml_tensor *pComb, int pIl) const;

    ggml_tensor *mBuildEngram(const mgs_model &pModel, llm_graph_input_dsv41_engram *pInp, ggml_tensor *pX, int pE,
                              int pIl) const;

    // pools the completed groups of the compressor state; returns the latent before RoPE
    ggml_tensor *mBuildPooledLatent(ggml_tensor *pKvState, ggml_tensor *pScoreState, ggml_tensor *pStateReadIdxs,
                                    int64_t pNBlocks, int64_t pRatio, ggml_tensor *pNorm, int pIl) const;

    // RoPE on the trailing n_rot dims of [head_dim, n_head, n_tokens] with the rope of the compressed stream
    ggml_tensor *mBuildRopeTail(ggml_tensor *pX, ggml_tensor *pPos, bool pCompressed, bool pBack, int pIl) const;

    ggml_tensor *mBuildTopKMask(ggml_tensor *pKqMask, ggml_tensor *pTopK, const char *pName, int pIl) const;

    ggml_tensor *mBuildIndexerTopK(const mgs_model &pModel, llm_graph_input_dsv41 *pInp, uint32_t pG, ggml_tensor *pQr,
                                   ggml_tensor *pCur, ggml_tensor *pInpPos, int pIl);

    ggml_tensor *mBuildAttention(const mgs_model &pModel, llm_graph_input_dsv41 *pInp, ggml_tensor *pCur,
                                 ggml_tensor *pInpPos, int pIl);

    // per layer, the top-k mask an index source built for the compressed rows (readers reuse it)
    std::vector<ggml_tensor *> lTopKMask;

    // the candidate blocks the candidate source kept, as a mask over the compressed rows
    ggml_tensor *vCandidateMask = nullptr;
  };

  std::unique_ptr<llm_graph_context> mBuildArchGraph(const llm_graph_params &pParams) const override;
};

struct mgs_model_glm_dsa : public mgs_model_base {
  mgs_model_glm_dsa(const struct mgs_model_params &pParams) : mgs_model_base(pParams) {}
  void mLoadArchHparams(mgs_model_loader &pMl) override;
  void mLoadArchTensors(mgs_model_loader &pMl) override;

  struct graph : public llm_graph_context {
    graph(const mgs_model &pModel, const llm_graph_params &pParams);
  };

  struct graph_mtp : public llm_graph_context {
    graph_mtp(const mgs_model &pModel, const llm_graph_params &pParams);
  };

  std::unique_ptr<llm_graph_context> mBuildArchGraph(const llm_graph_params &pParams) const override;
};

class mgs_memory_hybrid_idx_context;

// GLM-5.3-Flash: hybrid KDA (linear) + nope MLA with a k-pool DSA indexer, mHC residual streams, DeepSeek-style MoE
struct mgs_model_glm5_next : public mgs_model_base {
  mgs_model_glm5_next(const struct mgs_model_params &pParams) : mgs_model_base(pParams) {}
  void mLoadArchHparams(mgs_model_loader &pMl) override;
  void mLoadArchTensors(mgs_model_loader &pMl) override;

  // k-pool indexer inputs on top of the generic hybrid input
  class llm_graph_input_kpool;

  struct graph : public llm_build_delta_net_base {
    graph(const mgs_model &pModel, const llm_graph_params &pParams);

    // build the helpers without the trunk, so graph_mtp can reuse them
    struct no_build {};
    graph(const mgs_model &pModel, const llm_graph_params &pParams, no_build);

    // collapse the hc streams with per-stream weights
    ggml_tensor *mBuildHcPre(ggml_tensor *pX, ggml_tensor *pWeights, int pIl) const;

    // mean over the hyper-connection streams: [n_embd, hc, n_tokens] -> [n_embd, n_tokens]
    ggml_tensor *mBuildHcMean(ggml_tensor *pX) const;

    // returns the collapsed input and fills the post / comb weights
    ggml_tensor *mBuildHcPre(ggml_tensor *pX, ggml_tensor *pHcFn, ggml_tensor *pHcScale, ggml_tensor *pHcBase,
                             ggml_tensor **pPost, ggml_tensor **pComb, int pIl) const;

    ggml_tensor *mBuildHcPost(ggml_tensor *pX, ggml_tensor *pResidual, ggml_tensor *pPost, ggml_tensor *pComb,
                              int pIl) const;

    ggml_tensor *mBuildHcSinkhorn(ggml_tensor *pComb, int pIl) const;

    const mgs_model &vModel;

    llm_graph_input_kpool *mBuildInpKpool(const mgs_memory_hybrid_idx_context *pMctxHyb);

    ggml_tensor *mBuildKdaLayer(ggml_tensor *pCur, const mgs_layer &pLayer, llm_graph_input_rs *pInpRs, int64_t pDConv,
                                int64_t pHeadDim, int64_t pNHeadKda, int64_t pDInner, int64_t pNSeqTokens,
                                int64_t pNSeqs, int pIl);

    ggml_tensor *mBuildKpoolSelect(ggml_tensor *pCur, ggml_tensor *pQr, ggml_tensor *pKqMask, const mgs_layer &pLayer,
                                   const mgs_memory_hybrid_idx_context *pMctxHyb, llm_graph_input_kpool *pInpKpool,
                                   int pIl);

    ggml_tensor *mBuildDsaLayer(ggml_tensor *pCur, const mgs_layer &pLayer,
                                const mgs_memory_hybrid_idx_context *pMctxHyb, llm_graph_input_attn_k *pInpAttn,
                                llm_graph_input_kpool *pInpKpool, ggml_tensor **pPrevSel, int pIl);
  };

  // the NextN / MTP draft head: one DSA block appended after the trunk
  struct graph_mtp : public graph {
    graph_mtp(const mgs_model &pModel, const llm_graph_params &pParams);
  };

  std::unique_ptr<llm_graph_context> mBuildArchGraph(const llm_graph_params &pParams) const override;
};

struct mgs_model_dflash : public mgs_model_base {
  mgs_model_dflash(const struct mgs_model_params &pParams) : mgs_model_base(pParams) {}
  void mLoadArchHparams(mgs_model_loader &pMl) override;
  void mLoadArchTensors(mgs_model_loader &pMl) override;

  template <bool is_enc> struct graph : public llm_graph_context {
    graph(const mgs_model &pModel, const llm_graph_params &pParams);

    ggml_tensor *mBuildInpEmbdEnc() const;
  };

  std::unique_ptr<llm_graph_context> mBuildArchGraph(const llm_graph_params &pParams) const override;
};

struct mgs_model_glm4_moe : public mgs_model_base {
  mgs_model_glm4_moe(const struct mgs_model_params &pParams) : mgs_model_base(pParams) {}
  void mLoadArchHparams(mgs_model_loader &pMl) override;
  void mLoadArchTensors(mgs_model_loader &pMl) override;

  struct graph : public llm_graph_context {
    graph(const mgs_model &pModel, const llm_graph_params &pParams);
  };

  struct graph_mtp : public llm_graph_context {
    graph_mtp(const mgs_model &pModel, const llm_graph_params &pParams);
  };

  std::unique_ptr<llm_graph_context> mBuildArchGraph(const llm_graph_params &pParams) const override;
};

struct mgs_model_nemotron_h_moe : public mgs_model_base {
  mgs_model_nemotron_h_moe(const struct mgs_model_params &pParams) : mgs_model_base(pParams) {}
  void mLoadArchHparams(mgs_model_loader &pMl) override;
  void mLoadArchTensors(mgs_model_loader &pMl) override;

  struct graph : public llm_build_mamba_base {
    graph(const mgs_model &pModel, const llm_graph_params &pParams);
    ggml_tensor *mBuildFfnLayer(ggml_tensor *pCur, const mgs_model &pModel, int pIl);
    ggml_tensor *mBuildAttentionLayer(ggml_tensor *pCur, llm_graph_input_attn_kv *pInpAttn, const mgs_model &pModel,
                                      int64_t pNEmbdHead, int pIl);
  };

  std::unique_ptr<llm_graph_context> mBuildArchGraph(const llm_graph_params &pParams) const override;
};

struct mgs_model_exaone_moe : public mgs_model_base {
  mgs_model_exaone_moe(const struct mgs_model_params &pParams) : mgs_model_base(pParams) {}
  void mLoadArchHparams(mgs_model_loader &pMl) override;
  void mLoadArchTensors(mgs_model_loader &pMl) override;

  struct graph : public llm_graph_context {
    graph(const mgs_model &pModel, const llm_graph_params &pParams);
  };

  std::unique_ptr<llm_graph_context> mBuildArchGraph(const llm_graph_params &pParams) const override;
};

struct mgs_model_granite_moe : public mgs_model_base {
  mgs_model_granite_moe(const struct mgs_model_params &pParams) : mgs_model_base(pParams) {}
  void mLoadArchHparams(mgs_model_loader &pMl) override;
  void mLoadArchTensors(mgs_model_loader &pMl) override;

  struct graph : public llm_graph_context {
    graph(const mgs_model &pModel, const llm_graph_params &pParams);

  private:
    ggml_tensor *mBuildAttentionLayer(ggml_tensor *pCur, ggml_tensor *pInpPos, llm_graph_input_attn_kv *pInpAttn,
                                      const mgs_model &pModel, const int64_t pNEmbdHead, const int pIl);

    ggml_tensor *mBuildLayerFfn(ggml_tensor *pCur, ggml_tensor *pInpSA, const mgs_model &pModel, const int pIl);
  };

  std::unique_ptr<llm_graph_context> mBuildArchGraph(const llm_graph_params &pParams) const override;
};

struct mgs_model_granite_hybrid : public mgs_model_base {
  mgs_model_granite_hybrid(const struct mgs_model_params &pParams) : mgs_model_base(pParams) {}
  void mLoadArchHparams(mgs_model_loader &pMl) override;
  void mLoadArchTensors(mgs_model_loader &pMl) override;

  struct graph : public llm_build_mamba_base {
    graph(const mgs_model &pModel, const llm_graph_params &pParams);
    ggml_tensor *mBuildLayerFfn(ggml_tensor *pCur, ggml_tensor *pInpSA, const mgs_model &pModel, const int pIl);
    ggml_tensor *mBuildAttentionLayer(ggml_tensor *pCur, ggml_tensor *pInpPos, llm_graph_input_attn_kv *pInpAttn,
                                      const mgs_model &pModel, const int64_t pNEmbdHead, const int pIl);
  };

  std::unique_ptr<llm_graph_context> mBuildArchGraph(const llm_graph_params &pParams) const override;
};

struct mgs_model_bailingmoe : public mgs_model_base {
  mgs_model_bailingmoe(const struct mgs_model_params &pParams) : mgs_model_base(pParams) {}
  void mLoadArchHparams(mgs_model_loader &pMl) override;
  void mLoadArchTensors(mgs_model_loader &pMl) override;

  struct graph : public llm_graph_context {
    graph(const mgs_model &pModel, const llm_graph_params &pParams);
  };

  std::unique_ptr<llm_graph_context> mBuildArchGraph(const llm_graph_params &pParams) const override;
};

struct mgs_model_bailingmoe2 : public mgs_model_base {
  mgs_model_bailingmoe2(const struct mgs_model_params &pParams) : mgs_model_base(pParams) {}
  void mLoadArchHparams(mgs_model_loader &pMl) override;
  void mLoadArchTensors(mgs_model_loader &pMl) override;

  struct graph : public llm_graph_context {
    graph(const mgs_model &pModel, const llm_graph_params &pParams);
  };

  std::unique_ptr<llm_graph_context> mBuildArchGraph(const llm_graph_params &pParams) const override;
};

struct mgs_model_dots1 : public mgs_model_base {
  mgs_model_dots1(const struct mgs_model_params &pParams) : mgs_model_base(pParams) {}
  void mLoadArchHparams(mgs_model_loader &pMl) override;
  void mLoadArchTensors(mgs_model_loader &pMl) override;

  struct graph : public llm_graph_context {
    graph(const mgs_model &pModel, const llm_graph_params &pParams);
  };

  std::unique_ptr<llm_graph_context> mBuildArchGraph(const llm_graph_params &pParams) const override;
};

struct mgs_model_afmoe : public mgs_model_base {
  mgs_model_afmoe(const struct mgs_model_params &pParams) : mgs_model_base(pParams) {}
  void mLoadArchHparams(mgs_model_loader &pMl) override;
  void mLoadArchTensors(mgs_model_loader &pMl) override;

  struct graph : public llm_graph_context {
    graph(const mgs_model &pModel, const llm_graph_params &pParams);
  };

  std::unique_ptr<llm_graph_context> mBuildArchGraph(const llm_graph_params &pParams) const override;
};

struct mgs_model_laguna : public mgs_model_base {
  mgs_model_laguna(const struct mgs_model_params &pParams) : mgs_model_base(pParams) {}
  void mLoadArchHparams(mgs_model_loader &pMl) override;
  void mLoadArchTensors(mgs_model_loader &pMl) override;

  struct graph : public llm_graph_context {
    graph(const mgs_model &pModel, const llm_graph_params &pParams);
  };

  std::unique_ptr<llm_graph_context> mBuildArchGraph(const llm_graph_params &pParams) const override;
};

struct mgs_model_ernie4_5_moe : public mgs_model_base {
  mgs_model_ernie4_5_moe(const struct mgs_model_params &pParams) : mgs_model_base(pParams) {}
  void mLoadArchHparams(mgs_model_loader &pMl) override;
  void mLoadArchTensors(mgs_model_loader &pMl) override;

  struct graph : public llm_graph_context {
    graph(const mgs_model &pModel, const llm_graph_params &pParams);
  };

  std::unique_ptr<llm_graph_context> mBuildArchGraph(const llm_graph_params &pParams) const override;
};

struct mgs_model_hunyuan_moe : public mgs_model_base {
  mgs_model_hunyuan_moe(const struct mgs_model_params &pParams) : mgs_model_base(pParams) {}
  void mLoadArchHparams(mgs_model_loader &pMl) override;
  void mLoadArchTensors(mgs_model_loader &pMl) override;

  struct graph : public llm_graph_context {
    graph(const mgs_model &pModel, const llm_graph_params &pParams);
  };

  std::unique_ptr<llm_graph_context> mBuildArchGraph(const llm_graph_params &pParams) const override;
};

struct mgs_model_hy_v3 : public mgs_model_base {
  mgs_model_hy_v3(const struct mgs_model_params &pParams) : mgs_model_base(pParams) {}
  void mLoadArchHparams(mgs_model_loader &pMl) override;
  void mLoadArchTensors(mgs_model_loader &pMl) override;

  struct graph : public llm_graph_context {
    graph(const mgs_model &pModel, const llm_graph_params &pParams);
  };

  struct graph_mtp : public llm_graph_context {
    graph_mtp(const mgs_model &pModel, const llm_graph_params &pParams);
  };

  std::unique_ptr<llm_graph_context> mBuildArchGraph(const llm_graph_params &pParams) const override;
};

struct mgs_model_openai_moe : public mgs_model_base {
  mgs_model_openai_moe(const struct mgs_model_params &pParams) : mgs_model_base(pParams) {}
  void mLoadArchHparams(mgs_model_loader &pMl) override;
  void mLoadArchTensors(mgs_model_loader &pMl) override;

  struct graph : public llm_graph_context {
    graph(const mgs_model &pModel, const llm_graph_params &pParams);
  };

  std::unique_ptr<llm_graph_context> mBuildArchGraph(const llm_graph_params &pParams) const override;
};

struct mgs_model_lfm2moe : public mgs_model_base {
  mgs_model_lfm2moe(const struct mgs_model_params &pParams) : mgs_model_base(pParams) {}
  void mLoadArchHparams(mgs_model_loader &pMl) override;
  void mLoadArchTensors(mgs_model_loader &pMl) override;

  template <bool iswa> struct graph : public llm_graph_context {
    graph(const mgs_model &pModel, const llm_graph_params &pParams);
  };

  std::unique_ptr<llm_graph_context> mBuildArchGraph(const llm_graph_params &pParams) const override;
};

struct mgs_model_smallthinker : public mgs_model_base {
  mgs_model_smallthinker(const struct mgs_model_params &pParams) : mgs_model_base(pParams) {}
  void mLoadArchHparams(mgs_model_loader &pMl) override;
  void mLoadArchTensors(mgs_model_loader &pMl) override;

  template <bool iswa> struct graph : public llm_graph_context {
    graph(const mgs_model &pModel, const llm_graph_params &pParams);
  };

  std::unique_ptr<llm_graph_context> mBuildArchGraph(const llm_graph_params &pParams) const override;
};

struct mgs_model_grovemoe : public mgs_model_base {
  mgs_model_grovemoe(const struct mgs_model_params &pParams) : mgs_model_base(pParams) {}
  void mLoadArchHparams(mgs_model_loader &pMl) override;
  void mLoadArchTensors(mgs_model_loader &pMl) override;

  struct graph : public llm_graph_context {
    graph(const mgs_model &pModel, const llm_graph_params &pParams);
  };

  std::unique_ptr<llm_graph_context> mBuildArchGraph(const llm_graph_params &pParams) const override;
};

struct mgs_model_minimax_m2 : public mgs_model_base {
  mgs_model_minimax_m2(const struct mgs_model_params &pParams) : mgs_model_base(pParams) {}
  void mLoadArchHparams(mgs_model_loader &pMl) override;
  void mLoadArchTensors(mgs_model_loader &pMl) override;

  struct graph : public llm_graph_context {
    graph(const mgs_model &pModel, const llm_graph_params &pParams);
  };

  std::unique_ptr<llm_graph_context> mBuildArchGraph(const llm_graph_params &pParams) const override;
};

struct msa_params {
  int vBlk;
  int vTopkBlocks;
  int vLocal;
};

struct mgs_model_minimax_m3 : public mgs_model_base {
  mgs_model_minimax_m3(const struct mgs_model_params &pParams) : mgs_model_base(pParams) {}
  void mLoadArchHparams(mgs_model_loader &pMl) override;
  void mLoadArchTensors(mgs_model_loader &pMl) override;
  msa_params vMsaP;
  struct graph : public llm_graph_context {
    graph(const mgs_model &pModel, const llm_graph_params &pParams);

    ggml_tensor *mBuildAttnMsaFa(ggml_tensor *pQCur, // [D, HQ, S] f32
                                 ggml_tensor *pK,    // [D, n_keys, 1, C]  C = HKV or HKV*n_stream
                                 ggml_tensor *pV,    // [D, n_keys, 1, C]
                                 ggml_tensor *pMask, // [n_keys, R, 1, C] f16, R = HQ*T/(Gp*C)
                                 int64_t pGp, float pKqScale, int pIl) const;
  };
  std::unique_ptr<llm_graph_context> mBuildArchGraph(const llm_graph_params &pParams) const override;
};

struct mgs_model_qwen3next : public mgs_model_base {
  mgs_model_qwen3next(const struct mgs_model_params &pParams) : mgs_model_base(pParams) {}
  void mLoadArchHparams(mgs_model_loader &pMl) override;
  void mLoadArchTensors(mgs_model_loader &pMl) override;

  struct graph : public llm_build_delta_net_base {
    graph(const mgs_model &pModel, const llm_graph_params &pParams);

  private:
    ggml_tensor *mBuildLayerAttn(llm_graph_input_attn_kv *pInpAttn, ggml_tensor *pCur, ggml_tensor *pInpPos, int pIl);

    ggml_tensor *mBuildLayerAttnLinear(llm_graph_input_rs *pInp, ggml_tensor *pCur, int pIl);

    ggml_tensor *mBuildLayerFfn(ggml_tensor *pCur, int pIl);

    ggml_tensor *mBuildNormGated(ggml_tensor *pInput, ggml_tensor *pWeights, ggml_tensor *pGate, int pLayer);

    // returns pair of qkv, z
    std::pair<ggml_tensor *, ggml_tensor *> mBuildQkvz(ggml_tensor *pInput, int pIl);

    const mgs_model &vModel;
  };

  struct graph_mtp : public llm_graph_context {
    graph_mtp(const mgs_model &pModel, const llm_graph_params &pParams);
  };

  std::unique_ptr<llm_graph_context> mBuildArchGraph(const llm_graph_params &pParams) const override;
};

struct mgs_model_qwen35 : public mgs_model_base {
  mgs_model_qwen35(const struct mgs_model_params &pParams) : mgs_model_base(pParams) {}
  void mLoadArchHparams(mgs_model_loader &pMl) override;
  void mLoadArchTensors(mgs_model_loader &pMl) override;

  struct graph_mtp : public llm_graph_context {
    graph_mtp(const mgs_model &pModel, const llm_graph_params &pParams);
  };

  std::unique_ptr<llm_graph_context> mBuildArchGraph(const llm_graph_params &pParams) const override;
};

struct mgs_model_qwen35moe : public mgs_model_base {
  mgs_model_qwen35moe(const struct mgs_model_params &pParams) : mgs_model_base(pParams) {}
  void mLoadArchHparams(mgs_model_loader &pMl) override;
  void mLoadArchTensors(mgs_model_loader &pMl) override;

  struct graph : public llm_build_delta_net_base {
    graph(const mgs_model &pModel, const llm_graph_params &pParams);

  private:
    ggml_tensor *mBuildLayerAttn(llm_graph_input_attn_kv *pInpAttn, ggml_tensor *pCur, ggml_tensor *pInpPos,
                                 int *pSections, int pIl);

    ggml_tensor *mBuildLayerAttnLinear(llm_graph_input_rs *pInp, ggml_tensor *pCur, int pIl);

    ggml_tensor *mBuildLayerFfn(ggml_tensor *pCur, int pIl);

    ggml_tensor *mBuildNormGated(ggml_tensor *pInput, ggml_tensor *pWeights, ggml_tensor *pGate, int pLayer);

    // returns pair of qkv, z
    std::pair<ggml_tensor *, ggml_tensor *> mBuildQkvz(ggml_tensor *pInput, int pIl);

    const mgs_model &vModel;
  };

  struct graph_mtp : public llm_graph_context {
    graph_mtp(const mgs_model &pModel, const llm_graph_params &pParams);
  };

  std::unique_ptr<llm_graph_context> mBuildArchGraph(const llm_graph_params &pParams) const override;
};

class mgs_memory_hybrid_idx_context;

// Qwen3.8-Flash-Next: hybrid gated delta net + QSA block-sparse attention, low-rank hyper-connections,
// per-layer n-gram hash embeddings and MoE
struct mgs_model_qwen4exp : public mgs_model_base {
  mgs_model_qwen4exp(const struct mgs_model_params &pParams) : mgs_model_base(pParams) {}

  class llm_graph_input_kpool;

  void mLoadArchHparams(mgs_model_loader &pMl) override;
  void mLoadArchTensors(mgs_model_loader &pMl) override;

  // the PLE table stays memory-mapped on the host, so its gathered rows are worth prefetching
  bool mCanPrefetchPle() const;

  struct graph : public llm_build_delta_net_base {
    graph(const mgs_model &pModel, const llm_graph_params &pParams);

  protected:
    // the helpers alone, graph_mtp builds its own body
    struct no_build {};
    graph(const mgs_model &pModel, const llm_graph_params &pParams, no_build)
        : llm_build_delta_net_base(pParams), vModel(pModel) {}

    // true when the last layer must be narrowed to the output rows before the nextn hidden state is captured
    bool mCropBeforeNextn(const ggml_tensor *pInpOutIds) const;

    // true when the nextn hidden state must be narrowed to the output rows after it is captured
    bool mCropAfterNextn(const ggml_tensor *pInpOutIds) const;

    // HC replaces every layer norm: residual is [n_embd, hc, n_tokens]
    ggml_tensor *mBuildHcMix(ggml_tensor *pX, ggml_tensor *pWNorm, ggml_tensor *pWDown, ggml_tensor *pWUp,
                             ggml_tensor *pWInject, ggml_tensor **pInject, int pIl);

    ggml_tensor *mBuildHcCombine(ggml_tensor *pResidual, ggml_tensor *pBlockOut, ggml_tensor *pInject, int pIl);

    ggml_tensor *mBuildLayerAttn(llm_graph_input_attn_kv *pInpAttn, const mgs_memory_hybrid_idx_context *pMctxHyb,
                                 llm_graph_input_kpool *pInpKpool, ggml_tensor *pCur, ggml_tensor *pInpPos,
                                 int *pSections, int pIl);

    // dense self-attention over the cells the QSA mask keeps
    ggml_tensor *mBuildAttnQsa(llm_graph_input_attn_kv *pInp, ggml_tensor *pQCur, ggml_tensor *pKCur,
                               ggml_tensor *pVCur, ggml_tensor *pSel, int64_t pNSel, float pKqScale, int pIl);

    // the QSA layers share one set of k-pool inputs, see mgs_memory_hybrid_idx
    llm_graph_input_kpool *mBuildInpKpool(const mgs_memory_hybrid_idx_context *pMctxHyb);

    // QSA: the additive mask [n_kv, n_tokens] of the top blocks and the tail, kq_mask included
    ggml_tensor *mBuildQsaSel(const mgs_memory_hybrid_idx_context *pMctxHyb, llm_graph_input_kpool *pInpKpool,
                              ggml_tensor *pCur, ggml_tensor *pInpPos, ggml_tensor *pKqMask, int *pSections, int pIl);

    ggml_tensor *mBuildLayerAttnLinear(llm_graph_input_rs *pInp, ggml_tensor *pCur, int pIl);

    ggml_tensor *mBuildLayerFfn(ggml_tensor *pCur, int pIl);

    ggml_tensor *mBuildNormGated(ggml_tensor *pInput, ggml_tensor *pWeights, ggml_tensor *pGate, int pLayer);

    // mBuildRs writes the state tensor in place, so one gather per cache tensor is reused
    std::map<ggml_tensor *, ggml_tensor *> dRsRows;

    // one conv history per cache tensor: delta-net and PLE each have their own
    ggml_tensor *mBuildConvStateAt(llm_graph_input_rs *pInp, ggml_tensor *pConvStatesAll, ggml_tensor *pX,
                                   int64_t pStateCols, int64_t pChannels, int pIl);

    ggml_tensor *mBuildInpPle(const mgs_memory_hybrid_idx_context *pMctxHyb);

    ggml_tensor *mBuildPle(llm_graph_input_rs *pInp, ggml_tensor *pEmb, ggml_tensor *pHidden, int pIl);

    // returns pair of qkv, z
    std::pair<ggml_tensor *, ggml_tensor *> mBuildQkvz(ggml_tensor *pInput, int pIl);

    const mgs_model &vModel;
  };

  // MTP draft head: one QSA block after the trunk, fed by the trunk's hc-wide residual
  struct graph_mtp : public graph {
    graph_mtp(const mgs_model &pModel, const llm_graph_params &pParams);
  };

  std::unique_ptr<llm_graph_context> mBuildArchGraph(const llm_graph_params &pParams) const override;
};

struct mgs_model_mistral3 : public mgs_model_base {
  mgs_model_mistral3(const struct mgs_model_params &pParams) : mgs_model_base(pParams) {}
  void mLoadArchHparams(mgs_model_loader &pMl) override;
  void mLoadArchTensors(mgs_model_loader &pMl) override;

  struct graph : public llm_graph_context {
    graph(const mgs_model &pModel, const llm_graph_params &pParams);
  };

  std::unique_ptr<llm_graph_context> mBuildArchGraph(const llm_graph_params &pParams) const override;
};

struct mgs_model_mimo2 : public mgs_model_base {
  mgs_model_mimo2(const struct mgs_model_params &pParams) : mgs_model_base(pParams) {}
  void mLoadArchHparams(mgs_model_loader &pMl) override;
  void mLoadArchTensors(mgs_model_loader &pMl) override;

  struct graph : public llm_graph_context {
    graph(const mgs_model &pModel, const llm_graph_params &pParams);
  };

  struct graph_mtp : public llm_graph_context {
    graph_mtp(const mgs_model &pModel, const llm_graph_params &pParams);
  };

  std::unique_ptr<llm_graph_context> mBuildArchGraph(const llm_graph_params &pParams) const override;
};

struct mgs_model_kimi_k3 : public mgs_model_base {
  mgs_model_kimi_k3(const struct mgs_model_params &pParams) : mgs_model_base(pParams) {}
  void mLoadArchHparams(mgs_model_loader &pMl) override;
  void mLoadArchTensors(mgs_model_loader &pMl) override;

  struct graph : public llm_build_delta_net_base {
    graph(const mgs_model &pModel, const llm_graph_params &pParams);

    const mgs_model &vModel;

    // cross-layer residual attention: the banked checkpoints, [n_embd, n_ckpt, n_tokens]
    ggml_tensor *vResiStack = nullptr;

    void mResPush(ggml_tensor *pCur, int64_t pNEmbd, int64_t pNTokens);
    ggml_tensor *mResMix(ggml_tensor *pCur, ggml_tensor *pScoreW, int64_t pNTokens, int pIl);

    ggml_tensor *mBuildKdaLayer(ggml_tensor *pCur, const mgs_layer &pLayer, llm_graph_input_rs *pInpRs, int64_t pDConv,
                                int64_t pHeadDim, int64_t pNHeadKda, int64_t pDInner, int64_t pNSeqTokens,
                                int64_t pNSeqs, int pIl);

    ggml_tensor *mBuildMlaLayer(ggml_tensor *pCur, const mgs_layer &pLayer, llm_graph_input_attn_k *pInpAttnK,
                                int64_t pNEmbdHeadKMla, int64_t pKvLoraRank, int64_t pNEmbdHeadQkRope,
                                int64_t pNEmbdHeadQkNope, float pKqScale, int pIl);

    ggml_tensor *mBuildLatentMoe(ggml_tensor *pCur, const mgs_layer &pLayer, int pIl);
  };

  std::unique_ptr<llm_graph_context> mBuildArchGraph(const llm_graph_params &pParams) const override;
};

struct mgs_model_kimi_linear : public mgs_model_base {
  mgs_model_kimi_linear(const struct mgs_model_params &pParams) : mgs_model_base(pParams) {}
  void mLoadArchHparams(mgs_model_loader &pMl) override;
  void mLoadArchTensors(mgs_model_loader &pMl) override;

  struct graph : public llm_build_delta_net_base {
    graph(const mgs_model &pModel, const llm_graph_params &pParams);

    std::pair<ggml_tensor *, ggml_tensor *> mBuildKdaAutoregressive(ggml_tensor *pQ, ggml_tensor *pK, ggml_tensor *pV,
                                                                    ggml_tensor *pGk, ggml_tensor *pBeta,
                                                                    ggml_tensor *pState, int pIl);

    std::pair<ggml_tensor *, ggml_tensor *> mBuildKdaChunking(ggml_tensor *pQ, ggml_tensor *pK, ggml_tensor *pV,
                                                              ggml_tensor *pGk, ggml_tensor *pBeta, ggml_tensor *pState,
                                                              ggml_tensor *pCausalMask, ggml_tensor *pIdentity,
                                                              ggml_tensor *pDiagMask, int pIl);

    const mgs_model &vModel;
  };

  std::unique_ptr<llm_graph_context> mBuildArchGraph(const llm_graph_params &pParams) const override;
};

struct mgs_model_step35 : public mgs_model_base {
  mgs_model_step35(const struct mgs_model_params &pParams) : mgs_model_base(pParams) {}
  void mLoadArchHparams(mgs_model_loader &pMl) override;
  void mLoadArchTensors(mgs_model_loader &pMl) override;

  struct graph : public llm_graph_context {
    graph(const mgs_model &pModel, const llm_graph_params &pParams);
  };

  struct graph_mtp : public llm_graph_context {
    graph_mtp(const mgs_model &pModel, const llm_graph_params &pParams);
  };

  std::unique_ptr<llm_graph_context> mBuildArchGraph(const llm_graph_params &pParams) const override;
};

struct mgs_model_minicpm : public mgs_model_base {
  mgs_model_minicpm(const struct mgs_model_params &pParams) : mgs_model_base(pParams) {}
  void mLoadArchHparams(mgs_model_loader &pMl) override;
  void mLoadArchTensors(mgs_model_loader &pMl) override;

  using graph = mgs_model_granite_moe::graph;

  std::unique_ptr<llm_graph_context> mBuildArchGraph(const llm_graph_params &pParams) const override;
};
