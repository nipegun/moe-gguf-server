#pragma once

#include "mgs-arch.h"
#include "mgs-batch.h"
#include "mgs-hparams.h"
#include "mgs-adapter.h"

#include <cstdint>
#include <vector>
#include <memory>
#include <set>
#include <functional>
#include <map>

struct ggml_cgraph;
struct ggml_context;
struct ggml_tensor;

struct mgs_cparams;
struct mgs_layer;

struct mgs_memory_context_i;

class mgs_kv_cache_context;
class mgs_kv_cache_dsa_context;
class mgs_kv_cache_msa_context;
class mgs_kv_cache_iswa_context;
class mgs_memory_recurrent_context;
class mgs_expert_cache;
class mgs_memory_hybrid_context;
class mgs_memory_hybrid_iswa_context;

// certain models (typically multi-modal) can produce different types of graphs
enum llm_graph_type {
  cLlmGraphTypeDefault,
  cLlmGraphTypeEncoder,
  cLlmGraphTypeDecoder,
  cLlmGraphTypeDecoderMtp,
};

enum llm_fused_op {
  cLlmFusedOpFlashAttn,
  cLlmFusedOpGdnAr,
  cLlmFusedOpGdnCh,
  cLlmFusedOpLightningIndexer,
  cLlmFusedOpDsv4HcPre,
  cLlmFusedOpDsv4HcComb,
  cLlmFusedOpDsv4HcPost,
};

enum llm_ffn_op_type : int {
  cLlmFfnNone = 0, // sentinel: unset; archs must assign before use
  cLlmFfnSilu,
  cLlmFfnGelu,
  cLlmFfnRelu,
  cLlmFfnReluSqr,
  cLlmFfnSwiglu,
  cLlmFfnGeglu,
  cLlmFfnReglu,
  cLlmFfnSwigluOaiMoe,
  cLlmFfnSitu, // kimi-k3
};

enum llm_ffn_gate_type {
  cLlmFfnSeq,
  cLlmFfnPar, // ffn_gate is parallel to ffn_up
};

enum llm_norm_type {
  cLlmNorm,
  cLlmNormRms,
  cLlmNormGroup,
};

// TODO: tmp - need something better to pass the data from the encoder to the decoder
struct mgs_cross {
  // the output embeddings from the encoder as a ggml tensor
  // TODO: this needs more work to be correct, for now copy the embeddings data to host memory
  // ggml_tensor * vTEmbd = nullptr;

  int64_t vNEmbd = 0;
  int64_t vNEnc = 0;

  // embeddings data copied to host memory (tmp)
  std::vector<float> lVEmbd;

  // needed to construct the cross-attention mask in the decoder
  std::vector<std::set<mgs_seq_id>> sSeqIdsEnc;
};

struct llm_graph_params;

//
// llm_graph_input
//

class llm_graph_input_i {
public:
  llm_graph_input_i() {
    const char *vMgsGraphInputDebug = getenv("MGS_GRAPH_INPUT_DEBUG");
    vDebug = vMgsGraphInputDebug ? atoi(vMgsGraphInputDebug) : 0;
  }

  virtual ~llm_graph_input_i() = default;

  virtual void mSetInput(const mgs_ubatch *pUbatch) = 0;

  // return true if the resulting input tensors using the provided graph parameters would be
  //   the same as the previous input tensors that we have currently stored in the object
  virtual bool mCanReuse(const llm_graph_params &pParams) {
    // returning false here by default will prevent from reusing the graph if the check
    //   for the input type has not been implemented yet
    GGML_UNUSED(pParams);
    return false;
  }

protected:
  // env: MGS_GRAPH_INPUT_DEBUG
  int vDebug = 0;
};

using llm_graph_input_ptr = std::unique_ptr<llm_graph_input_i>;

class llm_graph_input_embd : public llm_graph_input_i {
public:
  llm_graph_input_embd(int64_t pNEmbd) : cNEmbd(pNEmbd) {}
  virtual ~llm_graph_input_embd() = default;

  void mSetInput(const mgs_ubatch *pUbatch) override;

  bool mCanReuse(const llm_graph_params &pParams) override;

  ggml_tensor *vTokens = nullptr; // I32 [n_batch]
  ggml_tensor *vEmbd = nullptr;   // F32 [n_embd, n_batch]

  const int64_t cNEmbd = 0;
};

// similar to llm_graph_input_embd but with an additional hidden state input
class llm_graph_input_embd_h : public llm_graph_input_i {
public:
  llm_graph_input_embd_h(int64_t pNEmbd) : cNEmbd(pNEmbd) {}
  virtual ~llm_graph_input_embd_h() = default;

  void mSetInput(const mgs_ubatch *pUbatch) override;

  bool mCanReuse(const llm_graph_params &pParams) override;

  ggml_tensor *vTokens = nullptr; // I32 [n_batch]
  ggml_tensor *vEmbd = nullptr;   // F32 [n_embd, n_batch]
  ggml_tensor *vH = nullptr;      // F32 [n_embd, n_batch]

  const int64_t cNEmbd = 0;
};

class llm_graph_input_pos : public llm_graph_input_i {
public:
  llm_graph_input_pos(uint32_t pNPosPerEmbd) : cNPosPerEmbd(pNPosPerEmbd) {}
  virtual ~llm_graph_input_pos() = default;

  void mSetInput(const mgs_ubatch *pUbatch) override;

  bool mCanReuse(const llm_graph_params &pParams) override;

  ggml_tensor *vPos = nullptr; // I32 [n_batch]

  const uint32_t cNPosPerEmbd = 1;
};

// temperature tuning
class llm_graph_input_attn_temp : public llm_graph_input_i {
public:
  llm_graph_input_attn_temp(uint32_t pNAttnTempFloorScale, float pFAttnTempScale, float pFAttnTempOffset)
      : cNAttnTempFloorScale(pNAttnTempFloorScale), cFAttnTempScale(pFAttnTempScale),
        cFAttnTempOffset(pFAttnTempOffset) {}
  virtual ~llm_graph_input_attn_temp() = default;

  void mSetInput(const mgs_ubatch *pUbatch) override;

  ggml_tensor *vAttnScale = nullptr; // F32 [n_batch]

  const uint32_t cNAttnTempFloorScale;
  const float cFAttnTempScale;
  const float cFAttnTempOffset;
};

class llm_graph_input_pos_bucket : public llm_graph_input_i {
public:
  llm_graph_input_pos_bucket(const mgs_hparams &pHparams) : cHparams(pHparams) {}
  virtual ~llm_graph_input_pos_bucket() = default;

  void mSetInput(const mgs_ubatch *pUbatch) override;

  ggml_tensor *vPosBucket = nullptr; // I32 [n_batch, n_batch]

  const mgs_hparams cHparams;
};

class llm_graph_input_pos_bucket_kv : public llm_graph_input_i {
public:
  llm_graph_input_pos_bucket_kv(const mgs_hparams &pHparams, const mgs_kv_cache_context *pMctx)
      : cHparams(pHparams), vMctx(pMctx) {}
  virtual ~llm_graph_input_pos_bucket_kv() = default;

  void mSetInput(const mgs_ubatch *pUbatch) override;

  ggml_tensor *vPosBucket = nullptr; // I32 [n_kv, n_batch]

  const mgs_hparams cHparams;

  const mgs_kv_cache_context *vMctx;
};

class llm_graph_input_out_ids : public llm_graph_input_i {
public:
  llm_graph_input_out_ids(const mgs_hparams &pHparams, const mgs_cparams &pCparams, uint32_t pNOutputs)
      : cHparams(pHparams), cCparams(pCparams), cNOutputs(pNOutputs) {}
  virtual ~llm_graph_input_out_ids() = default;

  void mSetInput(const mgs_ubatch *pUbatch) override;

  bool mCanReuse(const llm_graph_params &pParams) override;

  ggml_tensor *vOutIds; // I32 [n_outputs]

  const mgs_hparams cHparams;
  const mgs_cparams cCparams;

  const uint32_t cNOutputs;
};

class llm_graph_input_mean : public llm_graph_input_i {
public:
  llm_graph_input_mean(const mgs_cparams &pCparams) : cCparams(pCparams) {}
  virtual ~llm_graph_input_mean() = default;

  void mSetInput(const mgs_ubatch *pUbatch) override;

  ggml_tensor *vMean; // F32 [n_batch, n_batch]

  const mgs_cparams cCparams;
};

class llm_graph_input_cls : public llm_graph_input_i {
public:
  llm_graph_input_cls(const mgs_cparams &pCparams, const llm_arch pArch) : cCparams(pCparams), cArch(pArch) {}
  virtual ~llm_graph_input_cls() = default;

  void mSetInput(const mgs_ubatch *pUbatch) override;

  ggml_tensor *vCls; // I32 [n_batch]

  const mgs_cparams cCparams;
  const llm_arch cArch;
};

class llm_graph_input_rs : public llm_graph_input_i {
public:
  llm_graph_input_rs(const mgs_memory_recurrent_context *pMctx) : vMctx(pMctx) {}
  virtual ~llm_graph_input_rs() = default;

  void mSetInput(const mgs_ubatch *pUbatch) override;

  bool mCanReuse(const llm_graph_params &pParams) override;

  ggml_tensor *vSCopy; // I32 [n_rs]

  // views of s_copy, computed once per graph
  // and shared across layers which use build_rs
  ggml_tensor *vSCopyMain;  // I32 [n_seqs]
  ggml_tensor *vSCopyExtra; // I32 [n_rs - n_seqs]

  const mgs_memory_recurrent_context *vMctx;

  // used in view offsets, need to match for valid graph reuse
  uint32_t vHead;
  int32_t vRsZ;
};

class llm_graph_input_cross_embd : public llm_graph_input_i {
public:
  llm_graph_input_cross_embd(const mgs_cross *pCross) : vCross(pCross) {}
  virtual ~llm_graph_input_cross_embd() = default;

  void mSetInput(const mgs_ubatch *pUbatch) override;

  ggml_tensor *vCrossEmbd; // F32 [n_embd, n_outputs_enc]

  const mgs_cross *vCross;
};

class llm_graph_input_attn_no_cache : public llm_graph_input_i {
public:
  llm_graph_input_attn_no_cache(const mgs_hparams &pHparams, const mgs_cparams &pCparams)
      : cHparams(pHparams), cCparams(pCparams) {}
  ~llm_graph_input_attn_no_cache() = default;

  void mSetInput(const mgs_ubatch *pUbatch) override;

  ggml_tensor *mGetKqMask() const { return vSelfKqMaskCnv; }
  ggml_tensor *mGetKqMaskSwa() const { return vSelfKqMaskSwaCnv; }

  // n_tokens == n_batch
  ggml_tensor *vSelfKqMask = nullptr;       // F32/F16 [n_tokens, n_batch/n_stream, 1, n_stream]
  ggml_tensor *vSelfKqMaskCnv = nullptr;    //         [n_tokens, n_batch/n_stream, 1, n_stream]
  ggml_tensor *vSelfKqMaskSwa = nullptr;    // F32/F16 [n_tokens, n_batch/n_stream, 1, n_stream]
  ggml_tensor *vSelfKqMaskSwaCnv = nullptr; //         [n_tokens, n_batch/n_stream, 1, n_stream]

  const mgs_hparams cHparams;
  const mgs_cparams cCparams;
};

class llm_graph_input_attn_kv : public llm_graph_input_i {
public:
  llm_graph_input_attn_kv(const mgs_hparams &pHparams, const mgs_cparams &pCparams, const mgs_kv_cache_context *pMctx)
      : cHparams(pHparams), cCparams(pCparams), vMctx(pMctx) {}
  ~llm_graph_input_attn_kv() = default;

  void mSetInput(const mgs_ubatch *pUbatch) override;

  bool mCanReuse(const llm_graph_params &pParams) override;

  ggml_tensor *mGetKIdxs() const { return vSelfKIdxs; }
  ggml_tensor *mGetVIdxs() const { return vSelfVIdxs; }

  ggml_tensor *mGetKqMask() const { return vSelfKqMaskCnv; }

  ggml_tensor *vSelfKIdxs = nullptr; // I64 [n_batch]
  ggml_tensor *vSelfVIdxs = nullptr; // I64 [n_batch] or [n_batch*n_embd_v_gqa]

  ggml_tensor *vSelfKqMask = nullptr;    // F32/F16 [n_kv, n_batch/n_stream, 1, n_stream]
  ggml_tensor *vSelfKqMaskCnv = nullptr; //         [n_kv, n_batch/n_stream, 1, n_stream]

  // note: assumes v_rot^2 == I
  ggml_tensor *vSelfKRot = nullptr;
  ggml_tensor *vSelfVRot = nullptr;

  // note: these have to be copies because in order to be able to reuse a graph, its inputs
  //       need to carry these parameters with them. otherwise, they can point to freed
  //       llm_graph_params from a previous batch, causing stack-use-after-return
  const mgs_hparams cHparams;
  const mgs_cparams cCparams;

  const mgs_kv_cache_context *vMctx;
};

// V-less input for the KV cache
class llm_graph_input_attn_k : public llm_graph_input_i {
public:
  llm_graph_input_attn_k(const mgs_hparams &pHparams, const mgs_cparams &pCparams, const mgs_kv_cache_context *pMctx)
      : cHparams(pHparams), cCparams(pCparams), vMctx(pMctx) {}
  ~llm_graph_input_attn_k() = default;

  void mSetInput(const mgs_ubatch *pUbatch) override;

  bool mCanReuse(const llm_graph_params &pParams) override;

  ggml_tensor *mGetKIdxs() const { return vSelfKIdxs; }

  ggml_tensor *mGetKqMask() const { return vSelfKqMaskCnv; }

  ggml_tensor *vSelfKIdxs = nullptr; // I64 [n_batch]

  ggml_tensor *vSelfKqMask = nullptr;    // F32/F16 [n_kv, n_batch/n_stream, 1, n_stream]
  ggml_tensor *vSelfKqMaskCnv = nullptr; //         [n_kv, n_batch/n_stream, 1, n_stream]

  const mgs_hparams cHparams;
  const mgs_cparams cCparams;

  const mgs_kv_cache_context *vMctx;
};

class llm_graph_input_attn_k_dsa : public llm_graph_input_i {
public:
  llm_graph_input_attn_k_dsa(const mgs_hparams &pHparams, const mgs_cparams &pCparams,
                             const mgs_kv_cache_dsa_context *pMctx)
      : cHparams(pHparams), cCparams(pCparams), vMctx(pMctx) {}
  ~llm_graph_input_attn_k_dsa() = default;

  void mSetInput(const mgs_ubatch *pUbatch) override;

  bool mCanReuse(const llm_graph_params &pParams) override;

  ggml_tensor *mGetKIdxsMla() const { return vSelfKIdxsMla; }
  ggml_tensor *mGetKIdxsLid() const { return vSelfKIdxsLid; }

  ggml_tensor *mGetKqMaskMla() const { return vSelfKqMaskMlaCnv; }
  ggml_tensor *mGetKqMaskLid() const { return vSelfKqMaskLid; }

  ggml_tensor *vSelfKIdxsMla = nullptr; // I64 [n_batch]
  ggml_tensor *vSelfKIdxsLid = nullptr; // I64 [n_batch]

  ggml_tensor *vSelfKqMaskMla = nullptr;    // F32/F16 [n_kv, n_batch/n_stream, 1, n_stream]
  ggml_tensor *vSelfKqMaskMlaCnv = nullptr; //         [n_kv, n_batch/n_stream, 1, n_stream]
  ggml_tensor *vSelfKqMaskLid = nullptr;    // F32     [n_kv, n_batch/n_stream, 1, n_stream]
  ggml_tensor *vSelfKqMaskLidCnv = nullptr; //         [n_kv, n_batch/n_stream, 1, n_stream]

  ggml_tensor *vSelfKRotLid = nullptr;

  const mgs_hparams cHparams;
  const mgs_cparams cCparams;

  const mgs_kv_cache_dsa_context *vMctx;
};

// standard K/V attention input against the base cache, plus destination indices for the indexer key cache
class llm_graph_input_attn_kv_msa : public llm_graph_input_attn_kv {
public:
  llm_graph_input_attn_kv_msa(const mgs_hparams &pHparams, const mgs_cparams &pCparams,
                              const mgs_kv_cache_msa_context *pMctx);
  ~llm_graph_input_attn_kv_msa() = default;

  void mSetInput(const mgs_ubatch *pUbatch) override;

  bool mCanReuse(const llm_graph_params &pParams) override;

  ggml_tensor *mGetKIdxsIdx() const { return vSelfKIdxsIdx; }

  ggml_tensor *vSelfKIdxsIdx = nullptr; // I64 [n_batch]

  const mgs_kv_cache_msa_context *vMctxMsa;
};

class llm_graph_input_attn_kv_iswa : public llm_graph_input_i {
public:
  llm_graph_input_attn_kv_iswa(const mgs_hparams &pHparams, const mgs_cparams &pCparams,
                               const mgs_kv_cache_iswa_context *pMctx)
      : cHparams(pHparams), cCparams(pCparams), vMctx(pMctx) {}
  ~llm_graph_input_attn_kv_iswa() = default;

  void mSetInput(const mgs_ubatch *pUbatch) override;

  bool mCanReuse(const llm_graph_params &pParams) override;

  ggml_tensor *mGetKIdxs() const { return vSelfKIdxs; }
  ggml_tensor *mGetVIdxs() const { return vSelfVIdxs; }
  ggml_tensor *mGetKIdxsSwa() const { return vSelfKIdxsSwa; }
  ggml_tensor *mGetVIdxsSwa() const { return vSelfVIdxsSwa; }

  ggml_tensor *mGetKqMask() const { return vSelfKqMaskCnv; }
  ggml_tensor *mGetKqMaskSwa() const { return vSelfKqMaskSwaCnv; }

  ggml_tensor *vSelfKIdxs = nullptr;    // I64 [n_batch]
  ggml_tensor *vSelfVIdxs = nullptr;    // I64 [n_batch] or [n_batch*n_embd_v_gqa]
  ggml_tensor *vSelfKIdxsSwa = nullptr; // I64 [n_batch]
  ggml_tensor *vSelfVIdxsSwa = nullptr; // I64 [n_batch] or [n_batch*n_embd_v_gqa]

  ggml_tensor *vSelfKqMask = nullptr;       // F32/F16 [n_kv, n_batch/n_stream, 1, n_stream]
  ggml_tensor *vSelfKqMaskCnv = nullptr;    //         [n_kv, n_batch/n_stream, 1, n_stream]
  ggml_tensor *vSelfKqMaskSwa = nullptr;    // F32/F16 [n_kv, n_batch/n_stream, 1, n_stream]
  ggml_tensor *vSelfKqMaskSwaCnv = nullptr; //         [n_kv, n_batch/n_stream, 1, n_stream]

  ggml_tensor *vSelfKRot = nullptr;
  ggml_tensor *vSelfVRot = nullptr;

  ggml_tensor *vSelfKRotSwa = nullptr;
  ggml_tensor *vSelfVRotSwa = nullptr;

  const mgs_hparams cHparams;
  const mgs_cparams cCparams;

  const mgs_kv_cache_iswa_context *vMctx;
};

class llm_graph_input_attn_k_iswa : public llm_graph_input_i {
public:
  llm_graph_input_attn_k_iswa(const mgs_hparams &pHparams, const mgs_cparams &pCparams,
                              const mgs_kv_cache_iswa_context *pMctx)
      : cHparams(pHparams), cCparams(pCparams), vMctx(pMctx) {}
  ~llm_graph_input_attn_k_iswa() = default;

  void mSetInput(const mgs_ubatch *pUbatch) override;

  bool mCanReuse(const llm_graph_params &pParams) override;

  ggml_tensor *mGetKIdxs() const { return vSelfKIdxs; }
  ggml_tensor *mGetKIdxsSwa() const { return vSelfKIdxsSwa; }

  ggml_tensor *mGetKqMask() const { return vSelfKqMaskCnv; }
  ggml_tensor *mGetKqMaskSwa() const { return vSelfKqMaskSwaCnv; }

  ggml_tensor *vSelfKIdxs = nullptr;    // I64 [n_batch]
  ggml_tensor *vSelfKIdxsSwa = nullptr; // I64 [n_batch]

  ggml_tensor *vSelfKqMask = nullptr;       // F32/F16 [n_kv, n_batch/n_stream, 1, n_stream]
  ggml_tensor *vSelfKqMaskCnv = nullptr;    //         [n_kv, n_batch/n_stream, 1, n_stream]
  ggml_tensor *vSelfKqMaskSwa = nullptr;    // F32/F16 [n_kv, n_batch/n_stream, 1, n_stream]
  ggml_tensor *vSelfKqMaskSwaCnv = nullptr; //         [n_kv, n_batch/n_stream, 1, n_stream]

  ggml_tensor *vSelfKRot = nullptr;
  ggml_tensor *vSelfKRotSwa = nullptr;

  const mgs_hparams cHparams;
  const mgs_cparams cCparams;

  const mgs_kv_cache_iswa_context *vMctx;
};

class llm_graph_input_attn_cross : public llm_graph_input_i {
public:
  llm_graph_input_attn_cross(const mgs_cross *pCross) : vCross(pCross) {}
  ~llm_graph_input_attn_cross() = default;

  void mSetInput(const mgs_ubatch *pUbatch) override;

  ggml_tensor *mGetKqMaskCross() const { return vCrossKqMaskCnv; }

  ggml_tensor *vCrossKqMask = nullptr;    // F32/F16 [n_outputs_enc, n_batch, 1, 1]
  ggml_tensor *vCrossKqMaskCnv = nullptr; // F32/F16 [n_outputs_enc, n_batch, 1, 1]

  const mgs_cross *vCross = nullptr;
};

class llm_graph_input_mem_hybrid : public llm_graph_input_i {
public:
  llm_graph_input_mem_hybrid(const mgs_cparams &pCparams, std::unique_ptr<llm_graph_input_attn_kv> pInpAttn,
                             std::unique_ptr<llm_graph_input_rs> pInpRs, const mgs_memory_hybrid_context *pMctx)
      : vInpAttn(std::move(pInpAttn)), vInpRs(std::move(pInpRs)), cCparams(pCparams), vMctx(pMctx) {}
  virtual ~llm_graph_input_mem_hybrid() = default;

  void mSetInput(const mgs_ubatch *pUbatch) override;

  bool mCanReuse(const llm_graph_params &pParams) override;

  std::unique_ptr<llm_graph_input_attn_kv> vInpAttn;
  std::unique_ptr<llm_graph_input_rs> vInpRs;

  llm_graph_input_attn_kv *mGetAttn() const { return vInpAttn.get(); }
  llm_graph_input_rs *mGetRecr() const { return vInpRs.get(); }

  const mgs_cparams cCparams;

  const mgs_memory_hybrid_context *vMctx;
};

class llm_graph_input_mem_hybrid_k : public llm_graph_input_i {
public:
  llm_graph_input_mem_hybrid_k(const mgs_cparams &pCparams, std::unique_ptr<llm_graph_input_attn_k> pInpAttn,
                               std::unique_ptr<llm_graph_input_rs> pInpRs, const mgs_memory_hybrid_context *pMctx)
      : vInpAttn(std::move(pInpAttn)), vInpRs(std::move(pInpRs)), cCparams(pCparams), vMctx(pMctx) {}
  virtual ~llm_graph_input_mem_hybrid_k() = default;

  void mSetInput(const mgs_ubatch *pUbatch) override;

  bool mCanReuse(const llm_graph_params &pParams) override;

  std::unique_ptr<llm_graph_input_attn_k> vInpAttn;
  std::unique_ptr<llm_graph_input_rs> vInpRs;

  llm_graph_input_attn_k *mGetAttn() const { return vInpAttn.get(); }
  llm_graph_input_rs *mGetRecr() const { return vInpRs.get(); }

  const mgs_cparams cCparams;

  const mgs_memory_hybrid_context *vMctx;
};

class llm_graph_input_mem_hybrid_iswa : public llm_graph_input_i {
public:
  llm_graph_input_mem_hybrid_iswa(const mgs_cparams &pCparams, std::unique_ptr<llm_graph_input_attn_kv_iswa> pInpAttn,
                                  std::unique_ptr<llm_graph_input_rs> pInpRs,
                                  const mgs_memory_hybrid_iswa_context *pMctx)
      : vInpAttn(std::move(pInpAttn)), vInpRs(std::move(pInpRs)), cCparams(pCparams), vMctx(pMctx) {}
  virtual ~llm_graph_input_mem_hybrid_iswa() = default;

  void mSetInput(const mgs_ubatch *pUbatch) override;

  bool mCanReuse(const llm_graph_params &pParams) override;

  std::unique_ptr<llm_graph_input_attn_kv_iswa> vInpAttn;
  std::unique_ptr<llm_graph_input_rs> vInpRs;

  llm_graph_input_attn_kv_iswa *mGetAttn() const { return vInpAttn.get(); }
  llm_graph_input_rs *mGetRecr() const { return vInpRs.get(); }

  const mgs_cparams cCparams;

  const mgs_memory_hybrid_iswa_context *vMctx;
};

class llm_graph_input_sampling : public llm_graph_input_i {
public:
  llm_graph_input_sampling(std::map<mgs_seq_id, mgs_sampler *> pSamplers) : dSamplers(std::move(pSamplers)) {}
  virtual ~llm_graph_input_sampling() = default;

  void mSetInput(const mgs_ubatch *pUbatch) override;
  bool mCanReuse(const llm_graph_params &pParams) override;

  std::map<mgs_seq_id, mgs_sampler *> dSamplers;
};

//
// llm_graph_result
//

// these objects deliver the result from the graph build process back to the mgs_context
// note that the input tensors created for the graph are referenced here - the goal is to be able to populate their
//   specific data, by calling the mSetInputs() method
// along with the input tensors, the object also provides commonly used outputs tensors, such as logits, embeddings,
// etc.
//   these are used by the mgs_context to extact the relevant data, based on the compute parameters

// callback that allows us to apply custom logic to each tensor (e.g. ggml-alloc, offloading, etc.)
using llm_graph_cb = std::function<void(const mgs_ubatch &pUbatch, ggml_tensor *pCur, const char *pName, int pIl)>;

class llm_graph_result;

struct llm_graph_params {
  llm_arch vArch = cLlmArchUnknown;

  mgs_hparams vHparams;
  mgs_cparams vCparams;

  mgs_ubatch vUbatch; // note: intentionally make a copy

  llm_graph_type vGtype;

  ggml_backend_sched_t vSched;
  ggml_backend_t vBackendCpu;

  const mgs_adapter_cvec *vCvec;
  const mgs_adapter_loras *vLoras;
  const mgs_memory_context_i *vMctx;
  const mgs_cross *vCross;

  std::map<mgs_seq_id, mgs_sampler *> dSamplers;

  static bool mSamplersEqual(const std::map<mgs_seq_id, mgs_sampler *> &pLhs,
                             const std::map<mgs_seq_id, mgs_sampler *> &pRhs) {
    if (pLhs.size() != pRhs.size()) {
      return false;
    }
    for (const auto &[vSeqId, vSampler] : pLhs) {
      auto vIt = pRhs.find(vSeqId);
      if (vIt == pRhs.end() || vIt->second != vSampler) {
        return false;
      }
    }
    return true;
  }

  uint32_t vNOutputs;

  llm_graph_cb vCb;

  llm_graph_result *vRes;

  // device cache for host-resident MoE experts (nullptr when unused)
  mgs_expert_cache *vExpertCache = nullptr;

  // return true if the "other" params would result in a graph with the same topology as with the current params
  //   having the same topology allows us to reuse the graph in some cases
  bool mAllowReuse(const llm_graph_params &pOther) const {
    // first check the ubatch
    bool vCanReuseUbatch = vUbatch.mEqualSeqs() == pOther.vUbatch.mEqualSeqs() &&
                           vUbatch.vNTokens == pOther.vUbatch.vNTokens &&
                           vUbatch.vNSeqTokens == pOther.vUbatch.vNSeqTokens &&
                           vUbatch.vNSeqs == pOther.vUbatch.vNSeqs && vUbatch.vNSeqsUnq == pOther.vUbatch.vNSeqsUnq &&
                           ((!vUbatch.vToken && !pOther.vUbatch.vToken) || (!vUbatch.vEmbd && !pOther.vUbatch.vEmbd) ||
                            (vUbatch.vToken && pOther.vUbatch.vToken && vUbatch.vEmbd && pOther.vUbatch.vEmbd));

    // when we split the batch using "equal_seqs" we have to verify that the participating sequences are the same
    //   the reason is because the set of attention streams would be different for different sequences
    if (vCanReuseUbatch && vUbatch.mEqualSeqs()) {
      if (!vUbatch.vData) {
        // if the old ubatch does not own it's data, then we cannot guarantee that it is still alive, and
        //   therefore we cannot perform the sequence id check. normally should never happen
        vCanReuseUbatch = false;
      } else {
        for (uint32_t vS = 0; vS < vUbatch.vNSeqsUnq; ++vS) {
          vCanReuseUbatch &= vUbatch.vSeqIdUnq[vS] == pOther.vUbatch.vSeqIdUnq[vS];
        }
      }
    }

    if (!vCanReuseUbatch) {
      return false;
    }

    if (vNOutputs != pOther.vNOutputs) {
      return false;
    }

    if (!mSamplersEqual(dSamplers, pOther.dSamplers)) {
      return false;
    }

    if (dSamplers.size() > 0) {
      if (!vUbatch.vData || !pOther.vUbatch.vData) {
        return false;
      }

      // check that the outputs are the same for all samplers
      for (uint32_t vI = 0; vI < vUbatch.vNTokens; ++vI) {
        if (vUbatch.vOutput[vI] != pOther.vUbatch.vOutput[vI] ||
            vUbatch.vSeqId[vI][0] != pOther.vUbatch.vSeqId[vI][0]) {
          return false;
        }
      }
    }

    if (vCparams.vNextnLayerOffset != pOther.vCparams.vNextnLayerOffset) {
      return false;
    }

    return vCparams.vEmbeddings == pOther.vCparams.vEmbeddings &&
           vCparams.vEmbeddingsNextn == pOther.vCparams.vEmbeddingsNextn &&
           vCparams.vEmbeddingsNextnMasked == pOther.vCparams.vEmbeddingsNextnMasked &&
           vCparams.vCausalAttn == pOther.vCparams.vCausalAttn && vArch == pOther.vArch && vGtype == pOther.vGtype &&
           vCvec == pOther.vCvec && vLoras == pOther.vLoras && vCross == pOther.vCross &&
           vExpertCache == pOther.vExpertCache;
  }
};

struct llm_graph_fused_node {
  llm_fused_op vOp;
  ggml_tensor *vTensor;
  int vIl;
};

class llm_graph_result {
public:
  llm_graph_result(int64_t pMaxNodes);

  virtual ~llm_graph_result() = default;

  ggml_tensor *mGetInpTokens() const { return vTInpTokens; }
  ggml_tensor *mGetLogits() const { return vTLogits; }
  ggml_tensor *mGetEmbd() const { return vTEmbd; }
  ggml_tensor *mGetEmbdPooled() const { return vTEmbdPooled; }
  ggml_tensor *mGetHNextn() const { return vTHNextn; }

  ggml_tensor *mGetLayerInp(int pIl) const { return lTLayerInp[pIl]; }

  ggml_cgraph *mGetGf() const { return vGf; }
  ggml_context *mGetCtx() const { return vCtxCompute.get(); }

  int64_t mGetMaxNodes() const;

  void mReset();

  void mSetInputs(const mgs_ubatch *pUbatch);
  void mSetOutputs(const llm_graph_params &pParams);

  // try to update the existing graph result using the new graph parameters in order to reuse it
  // this can only be done if we determine that the resulting graph using the new graph parameters
  //   would be identical to the existing graph. in that case, we simply have to update the memory
  //   contexts of the input tensors of the graph and we can reuse it for another computation
  // return true if the graph was updated and can be reused
  bool mCanReuse(const llm_graph_params &pParams);

  llm_graph_input_i *mAddInput(llm_graph_input_ptr pInput);

  void mAddFusedNode(llm_graph_fused_node pResult);

  const std::vector<llm_graph_fused_node> &mGetFusedNodes() const { return lFusedNodes; }

  void mSetParams(const llm_graph_params &pParams);

  // important graph nodes
  ggml_tensor *vTInpTokens = nullptr;
  ggml_tensor *vTInpEmbd = nullptr; // [n_embd_inp, n_tokens]
  ggml_tensor *vTLogits = nullptr;
  ggml_tensor *vTEmbd = nullptr;
  ggml_tensor *vTEmbdPooled = nullptr;
  ggml_tensor *vTHNextn = nullptr; // [n_embd, n_outputs] hidden state before final output norm

  std::vector<ggml_tensor *> lTLayerInp;

  std::map<mgs_seq_id, ggml_tensor *> dTSampledLogits;
  std::map<mgs_seq_id, ggml_tensor *> dTCandidates;
  std::map<mgs_seq_id, ggml_tensor *> dTSampled;
  std::map<mgs_seq_id, ggml_tensor *> dTSampledProbs;

  std::vector<llm_graph_input_ptr> lInputs;
  std::vector<llm_graph_fused_node> lFusedNodes;

  ggml_context_ptr vCtxCompute;

  // memory buffers used to evaluate the model
  std::vector<uint8_t> lBufComputeMeta;

  ggml_cgraph *vGf;

  int64_t vMaxNodes;

private:
  // keep a copy of the previous graph parameters
  // we will use this to determine whether the graph can be reused by comparing them with the new parameters
  // note: these are updated after constructing the new graph
  llm_graph_params vParams;

  // env: MGS_GRAPH_RESULT_DEBUG
  int vDebug = 0;
};

using llm_graph_result_ptr = std::unique_ptr<llm_graph_result>;

//
// llm_graph_context
//

// used in build_rs to properly order writes and avoid unnecessary copies
using llm_graph_get_rows_fn = std::function<ggml_tensor *(ggml_context *, ggml_tensor *pStates, ggml_tensor *pIds)>;

struct llm_graph_qkv {
  ggml_tensor *vQ; // [n_embd_head, n_head,    n_tokens]
  ggml_tensor *vK; // [n_embd_head, n_head_kv, n_tokens]
  ggml_tensor *vV; // [n_embd_head, n_head_kv, n_tokens]
};

struct llm_graph_context {
  const llm_arch cArch;

  const mgs_hparams &vHparams;
  const mgs_cparams &vCparams;
  const mgs_ubatch &vUbatch;

  const int64_t cNEmbd;
  const int64_t cNLayer;
  const int64_t cNLayerNextn;
  const int64_t cNRot;
  const int64_t cNCtx; // user-specified context size (can be different from n_ctx_train)
  const int64_t cNHead;
  const int64_t cNHeadKv;
  const int64_t cNEmbdHeadK;
  const int64_t cNEmbdKGqa;
  const int64_t cNEmbdHeadV;
  const int64_t cNEmbdVGqa;
  const int64_t cNExpert;
  const int64_t cNExpertUsed;

  const float cFreqBase;
  const float cFreqScale;
  const float cExtFactor;
  const float cAttnFactor;
  const float cBetaFast;
  const float cBetaSlow;
  const float cNormEps;
  const float cNormRmsEps;

  const int64_t cNTokens;
  const int64_t cNOutputs;
  const int32_t cNCtxOrig; // yarn

  const enum mgs_pooling_type cPoolingType;
  const enum mgs_rope_type cRopeType;

  ggml_backend_sched_t vSched;

  ggml_backend_t vBackendCpu; // TODO: needed by build_attn_mha, figure out a way to remove?

  const mgs_adapter_cvec *vCvec;
  const mgs_adapter_loras *vLoras;
  const mgs_memory_context_i *vMctx;
  const mgs_cross *vCross;

  std::map<mgs_seq_id, mgs_sampler *> dSamplers;

  const llm_graph_cb &vCbFunc;

  llm_graph_result *vRes;

  mgs_expert_cache *vExpertCache;

  ggml_context *vCtx0 = nullptr;
  ggml_cgraph *vGf = nullptr;

  llm_graph_context(const llm_graph_params &pParams);
  virtual ~llm_graph_context() = default;

  void mCb(ggml_tensor *pCur, const char *pName, int pIl) const;

  //
  // common
  //

  ggml_tensor *mBuildCvec(ggml_tensor *pCur, int pIl) const;

  // do mat_mul, while optionally apply lora and per-tensor scale
  ggml_tensor *mBuildLoraMm(ggml_tensor *pW, ggml_tensor *pCur, ggml_tensor *pWS = nullptr) const;

  // do mat_mul_id, while optionally apply lora and per-expert scale
  // ids_s selects the per-expert scales when it differs from ids (hybrid expert execution)
  ggml_tensor *mBuildLoraMmId(ggml_tensor *pW,   // ggml_tensor * as
                              ggml_tensor *pCur, // ggml_tensor * b
                              ggml_tensor *pIds, ggml_tensor *pWS = nullptr, ggml_tensor *pIdsS = nullptr) const;

  ggml_tensor *mBuildNorm(ggml_tensor *pCur, ggml_tensor *pMw, ggml_tensor *pMb, llm_norm_type pType, int pIl) const;

  // compute Q, K, V projections with optional bias and reshape
  // supports both fused wqkv and separate wq/wk/wv paths
  llm_graph_qkv mBuildQkv(const mgs_layer &pLayer, ggml_tensor *pCur, int64_t pNEmbdHead, int64_t pNHead,
                          int64_t pNHeadKv, int pIl) const;

  ggml_tensor *mBuildFfn(ggml_tensor *pCur, ggml_tensor *pUp, ggml_tensor *pUpB, ggml_tensor *pUpS, ggml_tensor *pGate,
                         ggml_tensor *pGateB, ggml_tensor *pGateS, ggml_tensor *pDown, ggml_tensor *pDownB,
                         ggml_tensor *pDownS, ggml_tensor *pActScales, llm_ffn_op_type pTypeOp,
                         llm_ffn_gate_type pTypeGate, int pIl) const;

  // build MoE FFN without bias tensors
  ggml_tensor *mBuildMoeFfn(ggml_tensor *pCur, ggml_tensor *pGateInp, ggml_tensor *pUpExps, ggml_tensor *pGateExps,
                            ggml_tensor *pDownExps, ggml_tensor *pExpProbsB, int64_t pNExpert, int64_t pNExpertUsed,
                            llm_ffn_op_type pTypeOp, bool pNormW, float pWScale, mgs_expert_gating_func_type pGatingOp,
                            int pIl, ggml_tensor *pProbsIn = nullptr, ggml_tensor *pGateUpExps = nullptr,
                            ggml_tensor *pUpExpsS = nullptr, ggml_tensor *pGateExpsS = nullptr,
                            ggml_tensor *pDownExpsS = nullptr, ggml_tensor *pSelectedExpertsIn = nullptr) const;

  ggml_tensor *mBuildMoeFfn(ggml_tensor *pCur, ggml_tensor *pGateInp, ggml_tensor *pGateInpB, ggml_tensor *pUpExps,
                            ggml_tensor *pUpExpsB, ggml_tensor *pGateExps, ggml_tensor *pGateExpsB,
                            ggml_tensor *pDownExps, ggml_tensor *pDownExpsB, ggml_tensor *pExpProbsB, int64_t pNExpert,
                            int64_t pNExpertUsed, llm_ffn_op_type pTypeOp, bool pNormW, float pWScale,
                            mgs_expert_gating_func_type pGatingOp, int pIl, ggml_tensor *pProbsIn = nullptr,
                            ggml_tensor *pGateUpExps = nullptr, ggml_tensor *pGateUpExpsB = nullptr,
                            ggml_tensor *pUpExpsS = nullptr, ggml_tensor *pGateExpsS = nullptr,
                            ggml_tensor *pDownExpsS = nullptr, ggml_tensor *pSelectedExpertsIn = nullptr) const;

  //
  // inputs
  //

  ggml_tensor *mBuildInpEmbd(ggml_tensor *pTokEmbd) const;
  ggml_tensor *mBuildInpPos() const;
  ggml_tensor *mBuildInpAttnScale() const;
  ggml_tensor *mBuildInpOutIds() const;
  ggml_tensor *mBuildInpMean() const;
  ggml_tensor *mBuildInpCls() const;

  ggml_tensor *mBuildInpCrossEmbd() const;
  ggml_tensor *mBuildInpPosBucketEnc() const;
  ggml_tensor *mBuildInpPosBucketDec() const;
  ggml_tensor *mBuildPosBias(ggml_tensor *pPosBucket, ggml_tensor *pAttnRelB) const;

  //
  // attention
  //

  ggml_tensor *mBuildAttnMha(ggml_tensor *pQ, // [n_embd_head_q, n_head_q, n_tokens]
                             ggml_tensor *pK, // [n_embd_head_k, n_head_k, n_tokens]
                             ggml_tensor *pV, // [n_embd_head_v, n_head_v, n_tokens] (v_trans = false)
                             ggml_tensor *pKqB, ggml_tensor *pKqMask,
                             ggml_tensor *pSinks, // [n_head_q]
                             ggml_tensor *pVMla,  // [n_embd_head_v_mla, n_embd_head_v, n_head_v]
                             float pKqScale, int pIl) const;

  llm_graph_input_attn_no_cache *mBuildAttnInpNoCache() const;

  ggml_tensor *mBuildAttn(llm_graph_input_attn_no_cache *pInp, ggml_tensor *pWo, ggml_tensor *pWoB, ggml_tensor *pWoS,
                          ggml_tensor *pQCur, // [n_embd_head_q, n_head_q, n_tokens]
                          ggml_tensor *pKCur, // [n_embd_head_k, n_head_k, n_tokens]
                          ggml_tensor *pVCur, // [n_embd_head_v, n_head_v, n_tokens]
                          ggml_tensor *pKqB,
                          ggml_tensor *pSinks, // [n_head_q]
                          ggml_tensor *pVMla,  // [n_embd_head_v_mla, n_embd_head_v, n_head_v]
                          float pKqScale, int pIl) const;

  llm_graph_input_attn_kv *mBuildAttnInpKv() const;

  ggml_tensor *mBuildAttn(llm_graph_input_attn_kv *pInp, ggml_tensor *pWo, ggml_tensor *pWoB, ggml_tensor *pWoS,
                          ggml_tensor *pQCur, // [n_embd_head_q, n_head_q, n_tokens]
                          ggml_tensor *pKCur, // [n_embd_head_k, n_head_k, n_tokens]
                          ggml_tensor *pVCur, // [n_embd_head_v, n_head_v, n_tokens]
                          ggml_tensor *pKqB,
                          ggml_tensor *pSinks, // [n_head_q]
                          ggml_tensor *pVMla,  // [n_embd_head_v_mla, n_embd_head_v, n_head_v] // TODO: remove
                          float pKqScale, int pIl) const;

  llm_graph_input_attn_k *mBuildAttnInpK() const;

  ggml_tensor *mBuildAttn(llm_graph_input_attn_k *pInp, ggml_tensor *pWo, ggml_tensor *pWoB, ggml_tensor *pWoS,
                          ggml_tensor *pQCur, // [n_embd_head_q, n_head_q, n_tokens]
                          ggml_tensor *pKCur, // [n_embd_head_k, n_head_k, n_tokens]
                          ggml_tensor *pVCur, // [n_embd_head_v, n_head_v, n_tokens]
                          ggml_tensor *pKqB,
                          ggml_tensor *pSinks, // [n_head_q]
                          ggml_tensor *pVMla,  // [n_embd_head_v_mla, n_embd_head_v, n_head_v]
                          float pKqScale, int pIl) const;

  llm_graph_input_attn_k_dsa *mBuildAttnInpKDsa() const;

  llm_graph_input_attn_kv_msa *mBuildAttnInpKvMsa(bool pMsaEnabled) const;

  ggml_tensor *mBuildAttn(llm_graph_input_attn_k_dsa *pInp, ggml_tensor *pWo, ggml_tensor *pWoB, ggml_tensor *pWoS,
                          ggml_tensor *pQCur, // [n_embd_head_q, n_head_q, n_tokens]
                          ggml_tensor *pKCur, // [n_embd_head_k, n_head_k, n_tokens]
                          ggml_tensor *pVCur, // [n_embd_head_v, n_head_v, n_tokens]
                          ggml_tensor *pKqB,
                          ggml_tensor *pSinks, // [n_head_q]
                          ggml_tensor *pVMla,  // [n_embd_head_v_mla, n_embd_head_v, n_head_v]
                          ggml_tensor *pTopK,  // [n_indexer_top_k, n_tokens]
                          float pKqScale, int pIl) const;

  llm_graph_input_attn_kv_iswa *mBuildAttnInpKvIswa() const;


  // note: if k_cur or v_cur are not provided, they will not be stored in the memory
  ggml_tensor *mBuildAttn(llm_graph_input_attn_kv_iswa *pInp, ggml_tensor *pWo, ggml_tensor *pWoB, ggml_tensor *pWoS,
                          ggml_tensor *pQCur, // [n_embd_head_q, n_head_q, n_tokens]
                          ggml_tensor *pKCur, // [n_embd_head_k, n_head_k, n_tokens] optional
                          ggml_tensor *pVCur, // [n_embd_head_v, n_head_v, n_tokens] optional
                          ggml_tensor *pKqB,
                          ggml_tensor *pSinks, // [n_head_q]
                          ggml_tensor *pVMla,  // [n_embd_head_v_mla, n_embd_head_v, n_head_v]
                          float pKqScale, int pIl) const;

  llm_graph_input_attn_k_iswa *mBuildAttnInpKIswa() const;

  // note: if k_cur is not provided, it will not be stored in the memory
  // note: the K cache is used as V (MLA-style attention)
  ggml_tensor *mBuildAttn(llm_graph_input_attn_k_iswa *pInp, ggml_tensor *pWo, ggml_tensor *pWoB, ggml_tensor *pWoS,
                          ggml_tensor *pQCur, // [n_embd_head_q, n_head_q, n_tokens]
                          ggml_tensor *pKCur, // [n_embd_head_k, n_head_k, n_tokens] optional
                          ggml_tensor *pVCur, // [n_embd_head_v, n_head_v, n_tokens] optional
                          ggml_tensor *pKqB,
                          ggml_tensor *pSinks, // [n_head_q]
                          ggml_tensor *pVMla,  // [n_embd_head_v_mla, n_embd_head_v, n_head_v]
                          float pKqScale, int pIl) const;

  llm_graph_input_attn_cross *mBuildAttnInpCross() const;

  ggml_tensor *mBuildAttn(llm_graph_input_attn_cross *pInp, ggml_tensor *pWo, ggml_tensor *pWoB, ggml_tensor *pWoS,
                          ggml_tensor *pQCur, // [n_embd_head_q, n_head_q, n_tokens]
                          ggml_tensor *pKCur, // [n_embd_head_k, n_head_k, n_tokens]
                          ggml_tensor *pVCur, // [n_embd_head_v, n_head_v, n_tokens]
                          ggml_tensor *pKqB,
                          ggml_tensor *pSinks, // [n_head_q]
                          ggml_tensor *pVMla,  // [n_embd_head_v_mla, n_embd_head_v, n_head_v]
                          float pKqScale, int pIl) const;

  //
  // recurrent
  //

  // TODO: move this implementation to mgs_memory_recurrent.
  //       this is analogous to mgs_kv_cache::mCpyK / mCpyV
  //       when moving, avoid passing `ggml_cgraph` - only pass `ggml_context`. would likely need to split the
  //         implementation in 2 separate methods. the goal is to avoid calling `ggml_build_forward_expand` in
  //         `mgs_memory_recurrent`
  ggml_tensor *mBuildRs(ggml_tensor *pS, ggml_tensor *pStateCopyMain, ggml_tensor *pStateCopyExtra, int32_t pStateSize,
                        int32_t pNSeqs, uint32_t pNRs, uint32_t pRsHead, uint32_t pRsSize, int32_t pRsZero,
                        const llm_graph_get_rows_fn &pGetStateRows = ggml_get_rows) const;

  llm_graph_input_rs *mBuildRsInp() const;

  ggml_tensor *mBuildRs(llm_graph_input_rs *pInp, ggml_tensor *pS, int32_t pStateSize, int32_t pNSeqs,
                        const llm_graph_get_rows_fn &pGetStateRows = ggml_get_rows) const;

  ggml_tensor *mBuildRwkvTokenShiftLoad(llm_graph_input_rs *pInp, const mgs_ubatch &pUbatch, int pIl) const;

  ggml_tensor *mBuildRwkvTokenShiftStore(ggml_tensor *pTokenShift, const mgs_ubatch &pUbatch, int pIl) const;
  //
  // hybrid
  //

  llm_graph_input_mem_hybrid *mBuildInpMemHybrid() const;
  llm_graph_input_mem_hybrid_k *mBuildInpMemHybridK() const;

  llm_graph_input_mem_hybrid_iswa *mBuildInpMemHybridIswa() const;

  //
  // pooling
  //

  void mBuildPooling(ggml_tensor *pCls, ggml_tensor *pClsB, ggml_tensor *pClsOut, ggml_tensor *pClsOutB,
                     ggml_tensor *pClsNorm) const;

  //
  // sampling (backend sampling)
  //

  void mBuildSampling() const;

  //
  // dense (out)
  //

  void mBuildDenseOut(ggml_tensor *pDense2, ggml_tensor *pDense2B, ggml_tensor *pDense3) const;
};

// TODO: better name
int32_t mgs_relative_position_bucket(mgs_pos pX, mgs_pos pY, uint64_t pNBuckets, bool pBidirectional);
