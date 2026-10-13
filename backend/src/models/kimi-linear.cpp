#include "models.h"
#include "mgs-memory-recurrent.h"

void mgs_model_kimi_linear::mLoadArchHparams(mgs_model_loader &pMl) {
  pMl.mGetKey(cLlmKvAttentionLayernormRmsEps, vHparams.vFNormRmsEps);
  pMl.mGetKey(cLlmKvAttentionKeyLengthMla, vHparams.vNEmbdHeadKMlaImpl);
  pMl.mGetKey(cLlmKvAttentionValueLengthMla, vHparams.vNEmbdHeadVMlaImpl);
  pMl.mGetKey(cLlmKvAttentionKvLoraRank, vHparams.vNLoraKv);
  pMl.mGetKey(cLlmKvSsmConvKernel, vHparams.vSsmDConv);
  pMl.mGetKey(cLlmKvKdaHeadDim, vHparams.vNEmbdHeadKda);

  // MLA qk_rope_head_dim (for reference)
  // qk_rope_head_dim = 64, qk_nope_head_dim = 128, qk_head_dim = 192

  // Mark KDA layers as recurrent using n_head_kv pattern (like Jamba)
  // Set n_head_kv = 0 for KDA layers (recurrent), n_head_kv = n_head for MLA layers (attention)
  for (uint32_t vI = 0; vI < vHparams.mNLayer(); ++vI) {
    vHparams.aIsRecrImpl[vI] = vHparams.mNHeadKv(vI) == 0; // KDA layers are recurrent
  }

  // MoE parameters - Kimi uses moe_intermediate_size = 1024
  pMl.mGetKey(cLlmKvExpertFeedForwardLength, vHparams.vNFfExp);
  pMl.mGetKey(cLlmKvExpertSharedCount, vHparams.vNExpertShared);
  pMl.mGetKey(cLlmKvLeadingDenseBlockCount, vHparams.vNLayerDenseLead, false);
  pMl.mGetKey(cLlmKvExpertWeightsScale, vHparams.vExpertWeightsScale, false);
  pMl.mGetKey(cLlmKvExpertGatingFunc, vHparams.vExpertGatingFunc);

  switch (vHparams.mNLayer()) {
  case 27:
    vType = cLlmType48bA3b;
    break; // Kimi-Linear-48B-A3B
  default:
    vType = cLlmTypeUnknown;
  }
}

void mgs_model_kimi_linear::mLoadArchTensors(mgs_model_loader &) {
  MGS_LOAD_LOCALS;

  vTokEmbd = mCreateTensor(cTn(cLlmTensorTokenEmbd, "weight"), {cNEmbd, cNVocab}, 0);

  // output
  vOutputNorm = mCreateTensor(cTn(cLlmTensorOutputNorm, "weight"), {cNEmbd}, 0);
  vOutput = mCreateTensor(cTn(cLlmTensorOutput, "weight"), {cNEmbd, cNVocab}, 0);

  for (int vI = 0; vI < cNLayer; ++vI) {
    auto &vLayer = lLayers[vI];

    vLayer.vAttnNorm = mCreateTensor(cTn(cLlmTensorAttnNorm, "weight", vI), {cNEmbd}, 0);

    // Check for KDA specific tensors to determine layer type or if it's a mixed model
    // Assuming KDA layer if KDA tensors are present

    // KDA uses head_dim = 128 (from linear_attn_config.head_dim)
    const int64_t cNEmbdHeadKKda = vHparams.vNEmbdHeadKda;
    const int64_t cNEmbdHeadVKda = vHparams.vNEmbdHeadKda;
    const int64_t cSsmDConv = vHparams.vSsmDConv;

    if (vHparams.mIsRecr(vI)) {
      // Conv1d weights: try 4D first, then 3D (quantization may remove trailing 1)
      // 4D: [d_conv, 1, d_inner, 1], 3D: [d_conv, 1, d_inner]
      vLayer.vSsmQConv = mCreateTensor(cTn(cLlmTensorSsmConv1dQ, "weight", vI),
                                       {cSsmDConv, 1, cNEmbdHeadKKda * cNHead, 1}, cTensorNotRequired);
      if (!vLayer.vSsmQConv) {
        vLayer.vSsmQConv =
            mCreateTensor(cTn(cLlmTensorSsmConv1dQ, "weight", vI), {cSsmDConv, 1, cNEmbdHeadKKda * cNHead}, 0);
      }

      // KDA Layer - Conv1d weights may be 3D or 4D
      vLayer.vSsmKConv = mCreateTensor(cTn(cLlmTensorSsmConv1dK, "weight", vI),
                                       {cSsmDConv, 1, cNEmbdHeadKKda * cNHead, 1}, cTensorNotRequired);
      if (!vLayer.vSsmKConv) {
        vLayer.vSsmKConv =
            mCreateTensor(cTn(cLlmTensorSsmConv1dK, "weight", vI), {cSsmDConv, 1, cNEmbdHeadKKda * cNHead}, 0);
      }
      vLayer.vSsmVConv = mCreateTensor(cTn(cLlmTensorSsmConv1dV, "weight", vI),
                                       {cSsmDConv, 1, cNEmbdHeadVKda * cNHead, 1}, cTensorNotRequired);
      if (!vLayer.vSsmVConv) {
        vLayer.vSsmVConv =
            mCreateTensor(cTn(cLlmTensorSsmConv1dV, "weight", vI), {cSsmDConv, 1, cNEmbdHeadVKda * cNHead}, 0);
      }

      // q, k, v projections
      // Python: q_proj, k_proj, v_proj
      mCreateTensorQkv(vLayer, vI, cNEmbd, cNEmbdHeadKKda * cNHead, cNEmbdHeadKKda * cNHead, cNEmbdHeadVKda * cNHead,
                       0);

      // KDA specific projections
      // f_a_proj, f_b_proj
      vLayer.vSsmFA = mCreateTensor(cTn(cLlmTensorSsmFA, "weight", vI), {cNEmbd, cNEmbdHeadKKda}, 0); // head_dim
      vLayer.vSsmFB = mCreateTensor(cTn(cLlmTensorSsmFB, "weight", vI), {cNEmbdHeadKKda, cNEmbdHeadKKda * cNHead},
                                    0); // projection_size

      // b_proj (beta mixing coefficient)
      vLayer.vSsmBeta = mCreateTensor(cTn(cLlmTensorSsmBeta, "weight", vI), {cNEmbd, cNHead}, 0);

      // A_log - Shape in GGUF: [1, num_heads, 1, 1] (4D) or [1, num_heads] (2D after quantization) Note: -exp(A_log) is
      // applied in convert_hf_to_gguf.py
      vLayer.vSsmA = mCreateTensor(cTn(cLlmTensorSsmA, vI), {1, cNHead, 1, 1}, cTensorNotRequired);
      if (!vLayer.vSsmA) {
        vLayer.vSsmA = mCreateTensor(cTn(cLlmTensorSsmA, vI), {1, cNHead}, 0);
      }

      // dt_bias - shape [n_embd_head_k_kda * n_head] = [4096]
      vLayer.vSsmDtB = mCreateTensor(cTn(cLlmTensorSsmDt, "bias", vI), {cNEmbdHeadKKda * cNHead}, 0);

      // g_a_proj, g_b_proj (output gate)
      vLayer.vSsmGA = mCreateTensor(cTn(cLlmTensorSsmGA, "weight", vI), {cNEmbd, cNEmbdHeadKKda}, 0);
      vLayer.vSsmGB = mCreateTensor(cTn(cLlmTensorSsmGB, "weight", vI), {cNEmbdHeadKKda, cNEmbdHeadKKda * cNHead}, 0);

      // o_norm (reusing SSM_NORM)
      vLayer.vSsmONorm = mCreateTensor(cTn(cLlmTensorSsmNorm, "weight", vI), {cNEmbdHeadKKda}, 0); // FusedRMSNormGated

      // o_proj
      vLayer.vWo = mCreateTensor(cTn(cLlmTensorAttnOut, "weight", vI), {cNEmbdHeadVKda * cNHead, cNEmbd}, 0);

    } else {
      // MLA Layer - use MLA-specific head dimensions
      const int64_t cQLoraRank = vHparams.vNLoraQ;
      const int64_t cKvLoraRank = vHparams.vNLoraKv;
      const int64_t cNEmbdHeadKMla = vHparams.mNEmbdHeadKMla();
      const int64_t cNEmbdHeadVMla = vHparams.mNEmbdHeadVMla();

      vLayer.vAttnQANorm = mCreateTensor(cTn(cLlmTensorAttnQANorm, "weight", vI), {cQLoraRank}, cTensorNotRequired);
      vLayer.vAttnKvANorm = mCreateTensor(cTn(cLlmTensorAttnKvANorm, "weight", vI), {cKvLoraRank}, 0);

      if (vLayer.vAttnQANorm) {
        vLayer.vWqA = mCreateTensor(cTn(cLlmTensorAttnQA, "weight", vI), {cNEmbd, cQLoraRank}, 0);
        vLayer.vWqB = mCreateTensor(cTn(cLlmTensorAttnQB, "weight", vI), {cQLoraRank, cNHead * cNEmbdHeadKMla}, 0);
      } else {
        // Kimi MLA without Q compression: wq = [n_embd, n_head * n_embd_head_k_mla]
        vLayer.vWq = mCreateTensor(cTn(cLlmTensorAttnQ, "weight", vI), {cNEmbd, cNHead * cNEmbdHeadKMla}, 0);
      }

      // Kimi: qk_rope_head_dim = 64 (actual RoPE dimension for MLA)
      // Note: hparams.mNRot() may be 72 (from conversion) but actual is 64
      const int64_t cQkRopeHeadDim = vHparams.mNRot(); // From config: qk_rope_head_dim
      vLayer.vWkvAMqa =
          mCreateTensor(cTn(cLlmTensorAttnKvAMqa, "weight", vI), {cNEmbd, cKvLoraRank + cQkRopeHeadDim}, 0);
      vLayer.vWkB =
          mCreateTensor(cTn(cLlmTensorAttnKB, "weight", vI), {cNEmbdHeadKMla - cQkRopeHeadDim, cKvLoraRank, cNHead}, 0);
      vLayer.vWvB = mCreateTensor(cTn(cLlmTensorAttnVB, "weight", vI), {cKvLoraRank, cNEmbdHeadVMla, cNHead}, 0);
      vLayer.vWo = mCreateTensor(cTn(cLlmTensorAttnOut, "weight", vI), {cNHead * cNEmbdHeadVMla, cNEmbd}, 0);
    }

    vLayer.vFfnNorm = mCreateTensor(cTn(cLlmTensorFfnNorm, "weight", vI), {cNEmbd}, 0);

    // MoE intermediate size (different from dense FFN)
    const int64_t cNFfExp = vHparams.vNFfExp;

    // Kimi uses n_layer_dense_lead to determine which layers use dense FFN vs MoE
    // first_k_dense_replace = 1 means layer 0 uses dense FFN, layers 1+ use MoE
    if (vI < (int)vHparams.vNLayerDenseLead) {
      // Dense FFN layer - use normal n_ff
      vLayer.vFfnGate = mCreateTensor(cTn(cLlmTensorFfnGate, "weight", vI), {cNEmbd, cNFf}, 0);
      vLayer.vFfnDown = mCreateTensor(cTn(cLlmTensorFfnDown, "weight", vI), {cNFf, cNEmbd}, 0);
      vLayer.vFfnUp = mCreateTensor(cTn(cLlmTensorFfnUp, "weight", vI), {cNEmbd, cNFf}, 0);
    } else {
      // MoE layer - use n_ff_exp (1024) instead of n_ff (9216)
      vLayer.vFfnGateInp = mCreateTensor(cTn(cLlmTensorFfnGateInp, "weight", vI), {cNEmbd, cNExpert}, 0);
      vLayer.vFfnGateExps = mCreateTensor(cTn(cLlmTensorFfnGateExps, "weight", vI), {cNEmbd, cNFfExp, cNExpert}, 0);
      vLayer.vFfnDownExps = mCreateTensor(cTn(cLlmTensorFfnDownExps, "weight", vI), {cNFfExp, cNEmbd, cNExpert}, 0);
      vLayer.vFfnUpExps = mCreateTensor(cTn(cLlmTensorFfnUpExps, "weight", vI), {cNEmbd, cNFfExp, cNExpert}, 0);

      // Shared experts use moe_intermediate_size * num_shared_experts
      // Kimi: shared_expert_intermediate_size = 1024 * 1 = 1024
      // Tensors are 2D: [n_embd, n_ff_shexp] or [n_ff_shexp, n_embd]
      const int64_t cNFfShexpActual = cNFfExp * (vHparams.vNExpertShared > 0 ? vHparams.vNExpertShared : 1);
      vLayer.vFfnGateShexp =
          mCreateTensor(cTn(cLlmTensorFfnGateShexp, "weight", vI), {cNEmbd, cNFfShexpActual}, cTensorNotRequired);
      vLayer.vFfnDownShexp =
          mCreateTensor(cTn(cLlmTensorFfnDownShexp, "weight", vI), {cNFfShexpActual, cNEmbd}, cTensorNotRequired);
      vLayer.vFfnUpShexp =
          mCreateTensor(cTn(cLlmTensorFfnUpShexp, "weight", vI), {cNEmbd, cNFfShexpActual}, cTensorNotRequired);

      vLayer.vFfnExpProbsB = mCreateTensor(cTn(cLlmTensorFfnExpProbsB, "bias", vI), {cNExpert}, 0);
    }
  }
}

std::unique_ptr<llm_graph_context> mgs_model_kimi_linear::mBuildArchGraph(const llm_graph_params &pParams) const {
  return std::make_unique<graph>(*this, pParams);
}

// Causal Conv1d function for Q,K,V
// When qkv is 0, it is Q, 1 is K, 2 is V
static ggml_tensor *fCausalConv1d(ggml_cgraph *pGf, ggml_context *pCtx0, ggml_tensor *pConvStatesAll,
                                  ggml_tensor *pConvStateAll, int64_t pQkv, ggml_tensor *pX, ggml_tensor *pProjW,
                                  ggml_tensor *pConvW, int64_t pDConv, int64_t pHeadDim, int64_t pNHead,
                                  int64_t pNSeqTokens, int64_t pNSeqs, int64_t pNTokens, int64_t pKvHead) {
  const int64_t cDInner = pHeadDim * pNHead;
  const int64_t cConvStateSize = (pDConv - 1) * cDInner;
  const int64_t cNEmbdRTotal = 3 * cConvStateSize; // Q + K + V

  // conv_state_all is [n_embd_r_total, n_seqs], split into Q, K, V
  // Each conv state is [(d_conv-1) * d_inner] per sequence, need to reshape to [d_conv-1, d_inner, n_seqs]
  // Memory layout: for each seq, Q state is first conv_state_size elements, then K, then V
  // conv_state_all has stride: nb[0] = element_size, nb[1] = n_embd_r_total * element_size
  // View Q conv state: offset 0, size conv_state_size per seq
  // conv_state_all is [n_embd_r_total, n_seqs] with memory layout:
  //   state[i + seq * n_embd_r_total] where i = conv_step + channel * (d_conv-1) + {0, conv_state_size,
  //   2*conv_state_size} for Q/K/V
  // We want [d_conv-1, d_inner, n_seqs] view:
  //   nb1 = (d_conv-1) * element_size (stride between channels)
  //   nb2 = n_embd_r_total * element_size (stride between seqs)
  ggml_tensor *vConvStateX =
      ggml_view_3d(pCtx0, pConvStateAll, pDConv - 1, cDInner, pNSeqs,
                   (pDConv - 1) * ggml_element_size(pConvStateAll), // nb1: stride between channels
                   cNEmbdRTotal * ggml_element_size(pConvStateAll), // nb2: stride between seqs
                   pQkv * cConvStateSize * ggml_element_size(pConvStateAll));

  // Causal Conv1d function for Q,K,V
  // When qkv is 0, it is Q, 1 is K, 2 is V
  // Step 1: Q, K, V projections -> [d_inner, n_tokens]
  ggml_tensor *vXProj = ggml_mul_mat(pCtx0, pProjW, pX);

  // Reshape input: {d_inner, n_tokens} -> {d_inner, n_seq_tokens, n_seqs}
  ggml_tensor *vX3d = ggml_reshape_3d(pCtx0, vXProj, cDInner, pNSeqTokens, pNSeqs);

  // Concat Q conv state and current input: {d_conv-1 + n_seq_tokens, d_inner, n_seqs}
  ggml_tensor *vConvX = ggml_concat(pCtx0, vConvStateX, ggml_transpose(pCtx0, vX3d), 0);

  // Save last (d_conv-1) columns back to Q conv state
  ggml_tensor *vLastConvX = ggml_view_3d(pCtx0, vConvX, pDConv - 1, cDInner, pNSeqs, vConvX->nb[1], vConvX->nb[2],
                                         pNSeqTokens * vConvX->nb[0]);
  ggml_build_forward_expand(
      pGf,
      ggml_cpy(
          pCtx0, vLastConvX,
          ggml_view_3d(
              pCtx0, pConvStatesAll, pDConv - 1, cDInner, pNSeqs,
              (pDConv - 1) * ggml_element_size(pConvStatesAll), // nb1: contiguous within one channel's conv taps
              cNEmbdRTotal * ggml_element_size(pConvStatesAll), // nb2: stride between sequences (skip over K,V states)
              (pKvHead * cNEmbdRTotal + pQkv * cConvStateSize) *
                  ggml_element_size(pConvStatesAll)))); // offset to first seq's Q/K/V state
  // Reshape conv weight: GGUF [d_conv, 1, d_inner, 1] -> ggml_ssm_conv expects [d_conv, d_inner]
  // GGUF stores as [d_conv, 1, d_inner, 1] with memory layout w[conv_step + channel * d_conv]
  // vLLM stores as [d_inner, d_conv] with memory layout w[channel * d_conv + conv_step]
  // ggml_ssm_conv computes: c[conv_step + channel * d_conv]
  // GGUF layout: [d_conv, 1, d_inner] or [d_conv, 1, d_inner, 1] -> reshape to [d_conv, d_inner]
  // Reshape conv weight from [d_conv, 1, d_inner, 1] to [d_conv, d_inner] for ggml_ssm_conv
  ggml_tensor *vConvWeight = ggml_reshape_2d(pCtx0, pConvW, pDConv, cDInner);

  // Apply conv1d
  // ggml_ssm_conv output: {d_inner, n_seq_tokens, n_seqs}
  ggml_tensor *vXcur = ggml_ssm_conv(pCtx0, vConvX, vConvWeight);
  // Reshape to 2D for bias add: {d_inner, n_tokens}
  vXcur = ggml_reshape_2d(pCtx0, vXcur, cDInner, pNTokens);
  vXcur = ggml_silu(pCtx0, vXcur);

  return ggml_reshape_4d(pCtx0, vXcur, pHeadDim, pNHead, pNSeqTokens, pNSeqs);
}

mgs_model_kimi_linear::graph::graph(const mgs_model &pModel, const llm_graph_params &pParams)
    : llm_build_delta_net_base(pParams), vModel(pModel) {
  ggml_tensor *vCur;
  ggml_tensor *vInpL;

  vInpL = mBuildInpEmbd(pModel.vTokEmbd);
  mCb(vInpL, "model.embed_tokens", -1);

  // Note: Kimi MLA does NOT use RoPE (rotary_emb=None in vLLM)
  // So we don't need inp_pos

  auto *vInpK = mBuildInpMemHybridK();
  auto *vInpRs = vInpK->mGetRecr();
  auto *vInpAttnK = vInpK->mGetAttn();

  // Output ids for selecting which tokens to output
  ggml_tensor *vInpOutIds = mBuildInpOutIds();

  // Kimi dimension constants
  const int64_t cNHead = vHparams.mNHead();
  const int64_t cHeadDim = vHparams.vNEmbdHeadKda;
  const int64_t cDConv = vHparams.vSsmDConv;
  const int64_t cDInner = cNHead * cHeadDim; // 32 * 128 = 4096
  const int64_t cNSeqs = vUbatch.vNSeqs;
  const int64_t cNSeqTokens = vUbatch.vNSeqTokens;

  // Verify batch consistency for recurrent layers
  GGML_ASSERT(cNSeqs != 0);
  GGML_ASSERT(vUbatch.mEqualSeqs());
  GGML_ASSERT(vUbatch.vNTokens == cNSeqTokens * cNSeqs);

  // MLA params
  const int64_t cNEmbdHeadKMla = vHparams.mNEmbdHeadKMla();
  const int64_t cKvLoraRank = vHparams.vNLoraKv;
  // qk_rope_head_dim = 64 (from Kimi config) which is hparams.mNRot()
  // Confirmed from tensor shape: wkv_a_mqa [2304, 576] = [n_embd, kv_lora_rank + qk_rope_head_dim]
  const int64_t cNEmbdHeadQkRope = vHparams.mNRot();                  // config.qk_rope_head_dim
  const int64_t cNEmbdHeadQkNope = cNEmbdHeadKMla - cNEmbdHeadQkRope; // 192 - 64 = 128
  // Attention scale for MLA
  const float cKqScaleMla = 1.0f / sqrtf((float)cNEmbdHeadKMla);

  for (int vIl = 0; vIl < cNLayer; ++vIl) {
    const auto &vLayer = pModel.lLayers[vIl];
    ggml_tensor *vInpSA = vInpL;

    // Attention Norm
    vCur = mBuildNorm(vInpL, vLayer.vAttnNorm, NULL, cLlmNormRms, vIl);
    mCb(vCur, "attn_norm", vIl);

    ggml_build_forward_expand(vGf, vCur);

    if (vHparams.mIsRecr(vIl)) {
      // === KDA Layer (Kimi Delta Attention) with Recurrent State ===
      // Reference: vLLM kda.py
      const auto *vMctxCur = vInpRs->vMctx;
      const auto cKvHead = vMctxCur->mGetHead();

      // Get conv states from r_l tensor (Q, K, V each have separate state)
      ggml_tensor *vConvStatesAll = vMctxCur->mGetRL(vIl);
      mCb(vConvStatesAll, "conv_states_all", vIl);
      ggml_tensor *vConvStateAll = mBuildRs(vInpRs, vConvStatesAll, vHparams.mNEmbdR(), cNSeqs);
      ggml_tensor *vQcur =
          fCausalConv1d(vGf, vCtx0, vConvStatesAll, vConvStateAll, 0, vCur, vLayer.vWq, vLayer.vSsmQConv, cDConv,
                        cHeadDim, cNHead, cNSeqTokens, cNSeqs, cNTokens, cKvHead);
      ggml_tensor *vKcur =
          fCausalConv1d(vGf, vCtx0, vConvStatesAll, vConvStateAll, 1, vCur, vLayer.vWk, vLayer.vSsmKConv, cDConv,
                        cHeadDim, cNHead, cNSeqTokens, cNSeqs, cNTokens, cKvHead);
      ggml_tensor *vVcur =
          fCausalConv1d(vGf, vCtx0, vConvStatesAll, vConvStateAll, 2, vCur, vLayer.vWv, vLayer.vSsmVConv, cDConv,
                        cHeadDim, cNHead, cNSeqTokens, cNSeqs, cNTokens, cKvHead);

      // g1 = -exp(A_log) * softplus(f_b(f_a(x)) + dt_bias)
      ggml_tensor *vFA = ggml_mul_mat(vCtx0, vLayer.vSsmFA, vCur);
      ggml_tensor *vG1 = ggml_mul_mat(vCtx0, vLayer.vSsmFB, vFA);
      mCb(vG1, "g1 f_b(f_a(cur))", vIl);
      vG1 = ggml_add(vCtx0, vG1, vLayer.vSsmDtB);
      vG1 = ggml_softplus(vCtx0, vG1);
      vG1 = ggml_reshape_3d(vCtx0, vG1, cHeadDim, cNHead, cNTokens);

      // A_log shape is [1, n_head] or [1, n_head, 1, 1], need to broadcast to [head_dim, n_head, n_tokens]. No need to
      // -exp(a_log) because it was done in convert_hf_to_gguf.py Reshape to [1, n_head, 1] for broadcasting with g1
      // [head_dim, n_head, n_tokens]
      ggml_tensor *vA = ggml_reshape_3d(vCtx0, vLayer.vSsmA, 1, cNHead, 1);
      vG1 = ggml_mul(vCtx0, vG1, vA);
      mCb(vG1, "kda_g1", vIl);

      vG1 = ggml_reshape_4d(vCtx0, vG1, cHeadDim, cNHead, cNSeqTokens, cNSeqs);

      // Compute beta (mixing coefficient)
      ggml_tensor *vBeta = ggml_mul_mat(vCtx0, vLayer.vSsmBeta, vCur);
      vBeta = ggml_reshape_4d(vCtx0, vBeta, 1, cNHead, cNSeqTokens, cNSeqs);
      mCb(vBeta, "kda_beta", vIl);

      vBeta = ggml_sigmoid(vCtx0, vBeta);

      // Reshape for KDA recurrence
      // {n_embd, n_tokens} -> {n_embd, n_seq_tokens, n_seqs}
      vCur = ggml_reshape_3d(vCtx0, vCur, vCur->ne[0], cNSeqTokens, cNSeqs);

      // Get SSM state and compute KDA recurrence using ggml_kda_scan
      ggml_tensor *vSsmStatesAll = vMctxCur->mGetSL(vIl);
      ggml_tensor *vState = mBuildRs(vInpRs, vSsmStatesAll, vHparams.mNEmbdS(), cNSeqs);
      vState = ggml_reshape_4d(vCtx0, vState, cHeadDim, cHeadDim, cNHead, cNSeqs);

      const float cEpsNorm = vHparams.vFNormRmsEps;

      vQcur = ggml_l2_norm(vCtx0, vQcur, cEpsNorm);
      vKcur = ggml_l2_norm(vCtx0, vKcur, cEpsNorm);

      // Choose between build_delta_net_chunking and build_delta_net_recurrent based on n_tokens
      auto vAttnOut = mBuildDeltaNet(vQcur, vKcur, vVcur, vG1, vBeta, vState, vIl);

      ggml_tensor *vOutput = ggml_cont(vCtx0, vAttnOut.first);
      ggml_tensor *vNewState = vAttnOut.second;
      mCb(vOutput, "attn_output", vIl);
      mCb(vNewState, "new_state", vIl);

      // Update the recurrent states
      ggml_build_forward_expand(
          vGf, ggml_cpy(vCtx0, vNewState,
                        ggml_view_1d(vCtx0, vSsmStatesAll, vHparams.mNEmbdS() * cNSeqs,
                                     cKvHead * vHparams.mNEmbdS() * ggml_element_size(vSsmStatesAll))));

      // Output gating g2 = g_b(g_a(x))
      ggml_tensor *vCur2d = ggml_reshape_2d(vCtx0, vCur, vCur->ne[0], cNSeqTokens * cNSeqs);
      ggml_tensor *vGA = ggml_mul_mat(vCtx0, vLayer.vSsmGA, vCur2d);
      ggml_tensor *vG2 = ggml_mul_mat(vCtx0, vLayer.vSsmGB, vGA);
      mCb(vG2, "g2 g_b(g_a(cur_2d))", vIl);
      vG2 = ggml_reshape_3d(vCtx0, vG2, cHeadDim, cNHead, cNSeqTokens * cNSeqs);

      // Apply o_norm with sigmoid gating
      // Note: Kimi model uses sigmoid gating, not SiLU (despite FusedRMSNormGated default being swish)
      // Formula: output = RMSNorm(x) * sigmoid(g)
      ggml_tensor *vAttnOutFinal = ggml_reshape_3d(vCtx0, vOutput, cHeadDim, cNHead, cNSeqTokens * cNSeqs);
      ggml_tensor *vNormed = mBuildNorm(vAttnOutFinal, vLayer.vSsmONorm, nullptr, cLlmNormRms, vIl);
      mCb(vNormed, "kda_normed", vIl);
      ggml_tensor *vGate = ggml_sigmoid(vCtx0, vG2);
      ggml_tensor *vGated = ggml_mul(vCtx0, vNormed, vGate);

      // Output projection
      vGated = ggml_cont_2d(vCtx0, vGated, cDInner, cNTokens);
      vCur = ggml_mul_mat(vCtx0, vLayer.vWo, vGated);
      mCb(vCur, "kda_out", vIl);

    } else {
      // === MLA Layer (Multi-head Latent Attention) without KV Cache ===
      // Reference: vLLM mla.py
      // Step 1: Q projection and reshape
      // vLLM Kimi: q = q_proj(hidden_states), then view as [n_tokens, n_head, qk_head_dim]
      // Note: Kimi MLA does NOT use RoPE (rotary_emb=None in vLLM)
      ggml_tensor *vQcur = ggml_mul_mat(vCtx0, vLayer.vWq, vCur);

      // Step 2: KV compression
      // kv_cmpr_pe = kv_a_proj_with_mqa(hidden_states) -> [kv_lora_rank + qk_rope_head_dim, n_tokens]
      ggml_tensor *vKvCmprPe = ggml_mul_mat(vCtx0, vLayer.vWkvAMqa, vCur);

      // Split: kv_cmpr = kv_lora[:kv_lora_rank], k_pe = kv_lora[kv_lora_rank:]
      ggml_tensor *vKvCmpr = ggml_view_2d(vCtx0, vKvCmprPe, cKvLoraRank, cNTokens,
                                          ggml_row_size(vKvCmprPe->type, cKvLoraRank + cNEmbdHeadQkRope), 0);
      ggml_tensor *vKPe = ggml_view_3d(vCtx0, vKvCmprPe, cNEmbdHeadQkRope, 1, cNTokens,
                                       ggml_row_size(vKvCmprPe->type, cKvLoraRank + cNEmbdHeadQkRope),
                                       ggml_row_size(vKvCmprPe->type, cKvLoraRank + cNEmbdHeadQkRope),
                                       ggml_row_size(vKvCmprPe->type, cKvLoraRank));
      // Note: Kimi MLA does NOT apply RoPE (rotary_emb=None in vLLM)
      // k_pe is used directly without RoPE
      // Normalize kv_c
      vKvCmpr = mBuildNorm(vKvCmpr, vLayer.vAttnKvANorm, nullptr, cLlmNormRms, vIl);

      // extract q_nope
      ggml_tensor *vQNope =
          ggml_view_3d(vCtx0, vQcur, cNEmbdHeadQkNope, cNHead, cNTokens, ggml_row_size(vQcur->type, cNEmbdHeadKMla),
                       ggml_row_size(vQcur->type, cNEmbdHeadKMla) * cNHead, 0);
      mCb(vQNope, "q_nope", vIl);

      // and {n_embd_head_qk_rope, n_head, n_tokens}
      ggml_tensor *vQPe = ggml_view_3d(
          vCtx0, vQcur, cNEmbdHeadQkRope, cNHead, cNTokens, ggml_row_size(vQcur->type, cNEmbdHeadKMla),
          ggml_row_size(vQcur->type, cNEmbdHeadKMla) * cNHead, ggml_row_size(vQcur->type, cNEmbdHeadQkNope));
      mCb(vQPe, "q_pe", vIl);

      // {n_embd_head_qk_nope, n_tokens, n_head}
      vQNope = ggml_permute(vCtx0, vQNope, 0, 2, 1, 3);
      mCb(vQNope, "q_nope_perm", vIl);

      // {n_embd_head_qk_nope, kv_lora_rank, n_head} x {n_embd_head_qk_nope, n_tokens, n_head}
      ggml_tensor *vQNopeAbsorbed = ggml_mul_mat(vCtx0, vLayer.vWkB, vQNope);
      mCb(vQNopeAbsorbed, "q_nope_absorbed", vIl);

      // {kv_lora_rank, n_head, n_tokens}
      vQNopeAbsorbed = ggml_permute(vCtx0, vQNopeAbsorbed, 0, 2, 1, 3);
      mCb(vQNopeAbsorbed, "q_nope_absorbed_perm", vIl);

      // {n_embd_head_qk_rope + kv_lora_rank, n_head, n_tokens}
      // note: rope must go first for in-place context shifting in mBuildRopeShift()
      vQcur = ggml_concat(vCtx0, vQNopeAbsorbed, vQPe, 0);
      mCb(vQcur, "Qcur", vIl);

      vKvCmpr = ggml_reshape_3d(vCtx0, vKvCmpr, cKvLoraRank, 1, cNTokens);
      mCb(vKvCmpr, "kv_cmpr_reshape", vIl);

      // {n_embd_head_qk_rope + kv_lora_rank, 1, n_tokens}
      ggml_tensor *vKcur = ggml_concat(vCtx0, vKvCmpr, vKPe, 0);
      mCb(vKcur, "Kcur", vIl);

      // {kv_lora_rank, 1, n_tokens}
      ggml_tensor *vVcur = vKvCmpr;
      mCb(vVcur, "Vcur", vIl);

      vCur = mBuildAttn(vInpAttnK, vLayer.vWo, NULL, vLayer.vWoS, vQcur, vKcur, vVcur, nullptr, nullptr, vLayer.vWvB,
                        cKqScaleMla, vIl);
      mCb(vCur, "mla_out", vIl);
    }

    // On last layer, select only the output tokens
    if (vIl == cNLayer - 1 && vInpOutIds) {
      vCur = ggml_get_rows(vCtx0, vCur, vInpOutIds);
      vInpSA = ggml_get_rows(vCtx0, vInpSA, vInpOutIds);
    }

    // Residual
    ggml_tensor *vFfnInp = ggml_add(vCtx0, vCur, vInpSA);
    mCb(vFfnInp, "ffn_inp", vIl);

    // FFN Norm
    vCur = mBuildNorm(vFfnInp, vLayer.vFfnNorm, NULL, cLlmNormRms, vIl);
    mCb(vCur, "ffn_norm", vIl);

    if ((uint32_t)vIl < vHparams.vNLayerDenseLead) {
      // Dense FFN layer
      vCur = mBuildFfn(vCur, vLayer.vFfnUp, NULL, NULL, vLayer.vFfnGate, NULL, NULL, vLayer.vFfnDown, NULL, NULL, NULL,
                       cLlmFfnSilu, cLlmFfnPar, vIl);
      mCb(vCur, "ffn_out", vIl);
    } else {
      // MoE layer
      // Kimi uses moe_renormalize=True and routed_scaling_factor (stored as expert_weights_scale) = 2.446
      ggml_tensor *vMoeOut =
          mBuildMoeFfn(vCur, vLayer.vFfnGateInp, vLayer.vFfnUpExps, vLayer.vFfnGateExps, vLayer.vFfnDownExps,
                       vLayer.vFfnExpProbsB, vHparams.vNExpert, vHparams.vNExpertUsed, cLlmFfnSilu, true,
                       vHparams.vExpertWeightsScale, (mgs_expert_gating_func_type)vHparams.vExpertGatingFunc, vIl);
      mCb(vMoeOut, "ffn_moe_out", vIl);

      // Shared expert
      {
        ggml_tensor *vFfnShexp = mBuildFfn(vCur, vLayer.vFfnUpShexp, NULL, NULL, vLayer.vFfnGateShexp, NULL, NULL,
                                           vLayer.vFfnDownShexp, NULL, NULL, NULL, cLlmFfnSilu, cLlmFfnPar, vIl);
        mCb(vFfnShexp, "ffn_shexp", vIl);

        vCur = ggml_add(vCtx0, vMoeOut, vFfnShexp);
        mCb(vCur, "ffn_out", vIl);
      }
    }
    // Residual
    vCur = ggml_add(vCtx0, vCur, vFfnInp);

    vCur = mBuildCvec(vCur, vIl);
    mCb(vCur, "l_out", vIl);

    // input for next layer
    vInpL = vCur;
  }
  vCur = vInpL;

  // Final Norm
  vCur = mBuildNorm(vCur, pModel.vOutputNorm, NULL, cLlmNormRms, -1);

  mCb(vCur, "result_norm", -1);
  vRes->vTEmbd = vCur;

  // Output
  vCur = ggml_mul_mat(vCtx0, pModel.vOutput, vCur);
  mCb(vCur, "result_output", -1);
  vRes->vTLogits = vCur;

  ggml_build_forward_expand(vGf, vCur);
}
