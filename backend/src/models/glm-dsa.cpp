#include "models.h"

#include "mgs-kv-cache-dsa.h"

// https://huggingface.co/zai-org/GLM-5.2/blob/main/config.json#L26
const std::array<uint32_t, MGS_MAX_LAYERS> aGlm5Point2DefaultIndexerTypes = {
    1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1,
    0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0,
};

void mgs_model_glm_dsa::mLoadArchHparams(mgs_model_loader &pMl) {
  pMl.mGetKey(cLlmKvExpertFeedForwardLength, vHparams.vNFfExp);
  pMl.mGetKey(cLlmKvAttentionLayernormRmsEps, vHparams.vFNormRmsEps);
  pMl.mGetKeyOrArr(cLlmKvRopeDimensionSections, vHparams.aRopeSections, 4, false);

  // MoE parameters
  pMl.mGetKey(cLlmKvExpertCount, vHparams.vNExpert);
  pMl.mGetKey(cLlmKvExpertUsedCount, vHparams.vNExpertUsed);
  pMl.mGetKey(cLlmKvExpertSharedCount, vHparams.vNExpertShared);
  pMl.mGetKey(cLlmKvLeadingDenseBlockCount, vHparams.vNLayerDenseLead, false);
  pMl.mGetKey(cLlmKvExpertWeightsScale, vHparams.vExpertWeightsScale, false);
  pMl.mGetKey(cLlmKvExpertWeightsNorm, vHparams.vExpertWeightsNorm, false);

  // deepseek MLA parameters
  pMl.mGetKey(cLlmKvAttentionQLoraRank, vHparams.vNLoraQ);
  pMl.mGetKey(cLlmKvAttentionKvLoraRank, vHparams.vNLoraKv);
  pMl.mGetKey(cLlmKvAttentionKeyLengthMla, vHparams.vNEmbdHeadKMlaImpl);
  pMl.mGetKey(cLlmKvAttentionValueLengthMla, vHparams.vNEmbdHeadVMlaImpl);
  pMl.mGetKey(cLlmKvExpertFeedForwardLength, vHparams.vNFfExp);
  pMl.mGetKey(cLlmKvExpertSharedCount, vHparams.vNExpertShared);

  // DSA parameters
  pMl.mGetKey(cLlmKvAttentionIndexerHeadCount, vHparams.vIndexerNHead);
  pMl.mGetKey(cLlmKvAttentionIndexerKeyLength, vHparams.vIndexerHeadSize);
  pMl.mGetKey(cLlmKvAttentionIndexerTopK, vHparams.vIndexerTopK);

  // Expert gating function
  pMl.mGetKey(cLlmKvExpertGatingFunc, vHparams.vExpertGatingFunc);

  // NextN/MTP parameters
  pMl.mGetKey(cLlmKvNextnPredictLayers, vHparams.vNLayerNextn, false);
  GGML_ASSERT(vHparams.vNLayerNextn < vHparams.vNLayerAll && "n_layer_nextn must be < n_layer_all");

  // GGUFs without indexer_types metadata: GLM 5 and 5.1 (full indexers) and the published GLM 5.2 and 5.3 files
  // (shared indexers, 1M context)
  const bool cIsPre5Point2 = vHparams.vNCtxTrain < 1048576;
  if (cIsPre5Point2) {
    std::fill(vHparams.aIsIndexerFullImpl.begin(), vHparams.aIsIndexerFullImpl.end(), 1);
  } else {
    vHparams.aIsIndexerFullImpl = aGlm5Point2DefaultIndexerTypes;
  }
  pMl.mGetKeyOrArr(cLlmKvAttentionIndexerTypes, vHparams.aIsIndexerFullImpl, vHparams.mNLayer(), false);

  switch (vHparams.mNLayer()) {
  case 78: // GGUF with NextN/MTP metadata: n_layer() excludes the nextn layer
  case 79:
    vType = cLlmType744bA40b;
    break;
  default:
    vType = cLlmTypeUnknown;
  }
}

void mgs_model_glm_dsa::mLoadArchTensors(mgs_model_loader &pMl) {
  MGS_LOAD_LOCALS;
  const int64_t cNExpertShared = vHparams.vNExpertShared;

  // MTP-only: the GGUF carries only the NextN/MTP block(s) (user split target/draft).
  const bool cMtpOnly = (vHparams.vNLayerNextn > 0) && (pMl.mGetWeight("blk.0.attn_norm.weight") == nullptr);
  // Trunk-only: the GGUF declares MTP layers in metadata but the actual MTP
  // tensors live in a separate file (or were stripped at conversion). Mark
  // MTP tensors NOT_REQUIRED so the trunk loads cleanly.
  const std::string cMtpProbe = "blk." + std::to_string(cNLayer) + ".nextn.eh_proj.weight";
  const bool cTrunkOnly = (vHparams.vNLayerNextn > 0) && (pMl.mGetWeight(cMtpProbe.c_str()) == nullptr);
  const int cTrunkFlags = cMtpOnly ? cTensorNotRequired : 0;
  int vMtpFlags = cTrunkOnly ? cTensorNotRequired : 0;

  if (!pMl.vLoadMtp) {
    vMtpFlags |= cTensorSkip;
  }

  // note: these are the actual head sizes you get after "decompression" using wv_b for MLA
  const int64_t cNEmbdHeadKMla = vHparams.mNEmbdHeadKMla();
  const int64_t cNEmbdHeadVMla = vHparams.mNEmbdHeadVMla();

  const int64_t cNEmbdHeadQkRope = vHparams.mNRot();
  const int64_t cNEmbdHeadQkNope = cNEmbdHeadKMla - cNEmbdHeadQkRope;

  const int64_t cQLoraRank = vHparams.vNLoraQ;
  const int64_t cKvLoraRank = vHparams.vNLoraKv;

  const int64_t cNFfExp = vHparams.vNFfExp;

  vTokEmbd = mCreateTensor(cTn(cLlmTensorTokenEmbd, "weight"), {cNEmbd, cNVocab}, 0);

  // output
  vOutputNorm = mCreateTensor(cTn(cLlmTensorOutputNorm, "weight"), {cNEmbd}, 0);
  // try to load output.weight, if not found, use token_embd (tied embeddings)
  vOutput = mCreateTensor(cTn(cLlmTensorOutput, "weight"), {cNEmbd, cNVocab}, cTensorNotRequired);
  if (!vOutput) {
    vOutput = mCreateTensor(cTn(cLlmTensorTokenEmbd, "weight"), {cNEmbd, cNVocab}, cTensorDuplicated);
  }

  for (int vI = 0; vI < cNLayerAll; ++vI) {
    // NextN/MTP layers (i >= n_layer) are full decoder blocks used by the
    // LLM_GRAPH_TYPE_DECODER_MTP draft head; load them like qwen35moe/step35/hy_v3.
    const int cFlags = (vI >= cNLayer) ? vMtpFlags : cTrunkFlags;

    auto &vLayer = lLayers[vI];

    vLayer.vAttnNorm = mCreateTensor(cTn(cLlmTensorAttnNorm, "weight", vI), {cNEmbd}, cFlags);
    vLayer.vAttnQANorm = mCreateTensor(cTn(cLlmTensorAttnQANorm, "weight", vI), {cQLoraRank}, cFlags);
    vLayer.vAttnKvANorm = mCreateTensor(cTn(cLlmTensorAttnKvANorm, "weight", vI), {cKvLoraRank}, cFlags);

    vLayer.vWqA = mCreateTensor(cTn(cLlmTensorAttnQA, "weight", vI), {cNEmbd, cQLoraRank}, cFlags);
    vLayer.vWqB = mCreateTensor(cTn(cLlmTensorAttnQB, "weight", vI), {cQLoraRank, cNHead * cNEmbdHeadKMla}, cFlags);

    vLayer.vWkvAMqa =
        mCreateTensor(cTn(cLlmTensorAttnKvAMqa, "weight", vI), {cNEmbd, cKvLoraRank + cNEmbdHeadQkRope}, cFlags);

    // the converter splits kv_b_proj into the per-head k_b and v_b tensors used by the absorbed MLA path
    vLayer.vWkB = mCreateTensor(cTn(cLlmTensorAttnKB, "weight", vI), {cNEmbdHeadQkNope, cKvLoraRank, cNHead}, cFlags);
    vLayer.vWvB = mCreateTensor(cTn(cLlmTensorAttnVB, "weight", vI), {cKvLoraRank, cNEmbdHeadVMla, cNHead}, cFlags);

    vLayer.vWo = mCreateTensor(cTn(cLlmTensorAttnOut, "weight", vI), {cNHead * cNEmbdHeadVMla, cNEmbd}, cFlags);

    vLayer.vFfnNorm = mCreateTensor(cTn(cLlmTensorFfnNorm, "weight", vI), {cNEmbd}, cFlags);

    // DSA indexer
    vLayer.vIndexerKNorm = mCreateTensor(cTn(cLlmTensorIndexerKNorm, "weight", vI), {vHparams.vIndexerHeadSize},
                                         cFlags | cTensorNotRequired);
    vLayer.vIndexerKNormB = mCreateTensor(cTn(cLlmTensorIndexerKNorm, "bias", vI), {vHparams.vIndexerHeadSize},
                                          cFlags | cTensorNotRequired);
    vLayer.vIndexerProj = mCreateTensor(cTn(cLlmTensorIndexerProj, "weight", vI), {cNEmbd, vHparams.vIndexerNHead},
                                        cFlags | cTensorNotRequired);
    vLayer.vIndexerAttnK = mCreateTensor(cTn(cLlmTensorIndexerAttnK, "weight", vI), {cNEmbd, vHparams.vIndexerHeadSize},
                                         cFlags | cTensorNotRequired);
    vLayer.vIndexerAttnQB =
        mCreateTensor(cTn(cLlmTensorIndexerAttnQB, "weight", vI),
                      {cQLoraRank, vHparams.vIndexerNHead * vHparams.vIndexerHeadSize}, cFlags | cTensorNotRequired);
    if (vI < (int)vHparams.vNLayerDenseLead) {
      vLayer.vFfnGate = mCreateTensor(cTn(cLlmTensorFfnGate, "weight", vI), {cNEmbd, cNFf}, cFlags);
      vLayer.vFfnDown = mCreateTensor(cTn(cLlmTensorFfnDown, "weight", vI), {cNFf, cNEmbd}, cFlags);
      vLayer.vFfnUp = mCreateTensor(cTn(cLlmTensorFfnUp, "weight", vI), {cNEmbd, cNFf}, cFlags);
    } else {
      vLayer.vFfnGateInp = mCreateTensor(cTn(cLlmTensorFfnGateInp, "weight", vI), {cNEmbd, cNExpert}, cFlags);
      vLayer.vFfnExpProbsB = mCreateTensor(cTn(cLlmTensorFfnExpProbsB, "bias", vI), {cNExpert}, cTensorNotRequired);

      if (cNExpert == 0) {
        throw std::runtime_error("n_expert must be > 0");
      }
      if (cNExpertUsed == 0) {
        throw std::runtime_error("n_expert_used must be > 0");
      }

      // MoE branch
      vLayer.vFfnGateExps =
          mCreateTensor(cTn(cLlmTensorFfnGateExps, "weight", vI), {cNEmbd, cNFfExp, cNExpert}, cFlags);
      vLayer.vFfnDownExps =
          mCreateTensor(cTn(cLlmTensorFfnDownExps, "weight", vI), {cNFfExp, cNEmbd, cNExpert}, cFlags);
      vLayer.vFfnUpExps = mCreateTensor(cTn(cLlmTensorFfnUpExps, "weight", vI), {cNEmbd, cNFfExp, cNExpert}, cFlags);

      // Shared expert branch
      vLayer.vFfnGateShexp =
          mCreateTensor(cTn(cLlmTensorFfnGateShexp, "weight", vI), {cNEmbd, cNFfExp * cNExpertShared}, cFlags);
      vLayer.vFfnDownShexp =
          mCreateTensor(cTn(cLlmTensorFfnDownShexp, "weight", vI), {cNFfExp * cNExpertShared, cNEmbd}, cFlags);
      vLayer.vFfnUpShexp =
          mCreateTensor(cTn(cLlmTensorFfnUpShexp, "weight", vI), {cNEmbd, cNFfExp * cNExpertShared}, cFlags);
    }

    // NextN/MTP tensors - the NextN-specific wiring around the extra decoder block
    if (vI >= cNLayer) {
      vLayer.vNextn.vEhProj = mCreateTensor(cTn(cLlmTensorNextnEhProj, "weight", vI), {2 * cNEmbd, cNEmbd}, cFlags);
      vLayer.vNextn.vEnorm = mCreateTensor(cTn(cLlmTensorNextnEnorm, "weight", vI), {cNEmbd}, cFlags);
      vLayer.vNextn.vHnorm = mCreateTensor(cTn(cLlmTensorNextnHnorm, "weight", vI), {cNEmbd}, cFlags);

      // Optional tensors
      vLayer.vNextn.vEmbedTokens =
          mCreateTensor(cTn(cLlmTensorNextnEmbedTokens, "weight", vI), {cNEmbd, cNVocab}, cFlags | cTensorNotRequired);
      vLayer.vNextn.vSharedHeadHead = mCreateTensor(cTn(cLlmTensorNextnSharedHeadHead, "weight", vI), {cNEmbd, cNVocab},
                                                    cFlags | cTensorNotRequired);
      vLayer.vNextn.vSharedHeadNorm =
          mCreateTensor(cTn(cLlmTensorNextnSharedHeadNorm, "weight", vI), {cNEmbd}, cFlags | cTensorNotRequired);
    }
  }
}

std::unique_ptr<llm_graph_context> mgs_model_glm_dsa::mBuildArchGraph(const llm_graph_params &pParams) const {
  if (pParams.vGtype == cLlmGraphTypeDecoderMtp) {
    return std::make_unique<graph_mtp>(*this, pParams);
  }
  return std::make_unique<graph>(*this, pParams);
}

mgs_model_glm_dsa::graph::graph(const mgs_model &pModel, const llm_graph_params &pParams) : llm_graph_context(pParams) {
  // note: these are the actual head sizes you get after "decompression" using wv_b for MLA
  const int64_t cNEmbdHeadK = vHparams.mNEmbdHeadKMla();
  const int64_t cNEmbdHeadV = vHparams.mNEmbdHeadVMla();
  GGML_UNUSED(cNEmbdHeadV);

  const int64_t cNEmbdHeadQkRope = vHparams.mNRot();
  const int64_t cNEmbdHeadQkNope = cNEmbdHeadK - cNEmbdHeadQkRope;

  const int64_t cNIndexerHead = vHparams.vIndexerNHead;
  const int64_t cNEmbdIndexerHead = vHparams.vIndexerHeadSize;
  const int64_t cNEmbdIndexerHeadRope = vHparams.mNRot();
  const int64_t cNEmbdIndexerHeadNope = cNEmbdIndexerHead - cNEmbdIndexerHeadRope;
  const uint32_t cNIndexerTopK = vHparams.vIndexerTopK;

  const uint32_t cKvLoraRank = vHparams.vNLoraKv;

  // We have to pre-scale kq_scale and attn_factor to make the YaRN RoPE work correctly.
  // [TAG_YARN_LOG_MUL_FIX]

  // first cancel the adjustment from mgs_hparams::yarn_attn_factor_adjust to get the original attn_factor
  GGML_ASSERT(cExtFactor >= 0.0f);
  const float cAttnFactorOrg = cAttnFactor * (1.0f + 0.1f * logf(1.0f / cFreqScale));

  // use the original attn_factor to pre-scale the kq_scale
  const float cMscale = cAttnFactorOrg * (1.0f + 0.1f * vHparams.vRopeYarnLogMul * logf(1.0f / cFreqScale));
  const float cKqScale = 1.0f * cMscale * cMscale / sqrtf(float(cNEmbdHeadK));

  ggml_tensor *vCur;
  ggml_tensor *vInpL;

  // {n_embd, n_tokens}
  vInpL = mBuildInpEmbd(pModel.vTokEmbd);

  // inp_pos - contains the positions
  ggml_tensor *vInpPos = mBuildInpPos();

  llm_graph_input_attn_k_dsa *vInpAttnDsa = mBuildAttnInpKDsa();

  ggml_tensor *vInpOutIds = mBuildInpOutIds();

  // Difference vs Deepseek 3.2: shared indexer layers reuse the top_k from the previous full indexer layers
  // See https://huggingface.co/zai-org/GLM-5.2/blob/main/config.json#L30
  ggml_tensor *vPrevTopK = nullptr;
  for (int vIl = 0; vIl < cNLayer; ++vIl) {
    ggml_tensor *vInpSA = vInpL;

    // norm
    vCur = mBuildNorm(vInpL, pModel.lLayers[vIl].vAttnNorm, NULL, cLlmNormRms, vIl);
    mCb(vCur, "attn_norm", vIl);

    // self_attention
    {
      ggml_tensor *vQr = ggml_mul_mat(vCtx0, pModel.lLayers[vIl].vWqA, vCur);
      mCb(vQr, "qr", vIl);

      vQr = mBuildNorm(vQr, pModel.lLayers[vIl].vAttnQANorm, nullptr, cLlmNormRms, vIl);
      mCb(vQr, "qr", vIl);

      ggml_tensor *vTopK = nullptr;

      // lightning indexer
      if (vHparams.mIsIndexerFull(vIl)) {
        // "full" layer
        ggml_tensor *vIndexerQ = ggml_mul_mat(vCtx0, pModel.lLayers[vIl].vIndexerAttnQB, vQr);
        mCb(vIndexerQ, "indexer_q", vIl);

        // split into {n_embd_indexer_head_rope, n_indexer_head, n_tokens}
        ggml_tensor *vIndexerQPe = ggml_view_3d(vCtx0, vIndexerQ, cNEmbdIndexerHeadRope, cNIndexerHead, cNTokens,
                                                ggml_row_size(vIndexerQ->type, cNEmbdIndexerHead),
                                                ggml_row_size(vIndexerQ->type, cNEmbdIndexerHead) * cNIndexerHead, 0);
        mCb(vIndexerQPe, "indexer_q_pe", vIl);

        // and {n_embd_indexer_head_nope, n_indexer_head, n_tokens}
        ggml_tensor *vIndexerQNope = ggml_view_3d(vCtx0, vIndexerQ, cNEmbdIndexerHeadNope, cNIndexerHead, cNTokens,
                                                  ggml_row_size(vIndexerQ->type, cNEmbdIndexerHead),
                                                  ggml_row_size(vIndexerQ->type, cNEmbdIndexerHead) * cNIndexerHead,
                                                  ggml_row_size(vIndexerQ->type, cNEmbdIndexerHeadNope));
        mCb(vIndexerQNope, "indexer_q_nope", vIl);

        vIndexerQPe = ggml_rope_ext(vCtx0, vIndexerQPe, vInpPos, nullptr, cNRot, MGS_ROPE_TYPE_NORM, cNCtxOrig,
                                    cFreqBase, cFreqScale, cExtFactor, cAttnFactor, cBetaFast, cBetaSlow);
        mCb(vIndexerQPe, "indexer_q_pe", vIl);

        // {n_embd_indexer_head_rope + n_embd_indexer_head_nope, n_head, n_tokens}
        vIndexerQ = ggml_concat(vCtx0, vIndexerQPe, vIndexerQNope, 0);
        mCb(vIndexerQ, "indexer_q", vIl);

        ggml_tensor *vIndexerK = ggml_mul_mat(vCtx0, pModel.lLayers[vIl].vIndexerAttnK, vCur);
        mCb(vIndexerK, "indexer_k", vIl);

        vIndexerK =
            mBuildNorm(vIndexerK, pModel.lLayers[vIl].vIndexerKNorm, pModel.lLayers[vIl].vIndexerKNormB, cLlmNorm, vIl);
        mCb(vIndexerK, "indexer_k", vIl);

        // split into {n_embd_indexer_head_rope, 1, n_tokens}
        ggml_tensor *vIndexerKPe = ggml_view_3d(vCtx0, vIndexerK, cNEmbdIndexerHeadRope, 1, cNTokens,
                                                ggml_row_size(vIndexerK->type, cNEmbdIndexerHead),
                                                ggml_row_size(vIndexerK->type, cNEmbdIndexerHead) * 1, 0);
        mCb(vIndexerKPe, "indexer_k_pe", vIl);

        // and {n_embd_indexer_head_nope, 1, n_tokens}
        ggml_tensor *vIndexerKNope = ggml_view_3d(vCtx0, vIndexerK, cNEmbdIndexerHeadNope, 1, cNTokens,
                                                  ggml_row_size(vIndexerK->type, cNEmbdIndexerHead),
                                                  ggml_row_size(vIndexerK->type, cNEmbdIndexerHead) * 1,
                                                  ggml_row_size(vIndexerK->type, cNEmbdIndexerHeadNope));
        mCb(vIndexerKNope, "indexer_k_nope", vIl);

        vIndexerKPe = ggml_rope_ext(vCtx0, vIndexerKPe, vInpPos, nullptr, cNRot, MGS_ROPE_TYPE_NORM, cNCtxOrig,
                                    cFreqBase, cFreqScale, cExtFactor, cAttnFactor, cBetaFast, cBetaSlow);
        mCb(vIndexerKPe, "indexer_k_pe", vIl);

        // {n_embd_indexer_head_rope + n_embd_indexer_head_nope, 1, n_tokens}
        vIndexerK = ggml_concat(vCtx0, vIndexerKPe, vIndexerKNope, 0);
        mCb(vIndexerK, "indexer_k", vIl);

        // perform Hadamard transform on indexer q and k
        vIndexerQ = ggml_mul_mat(vCtx0, vInpAttnDsa->vSelfKRotLid, vIndexerQ);
        mCb(vIndexerQ, "indexer_q", vIl);
        vIndexerK = ggml_mul_mat(vCtx0, vInpAttnDsa->vSelfKRotLid, vIndexerK);
        mCb(vIndexerK, "indexer_k", vIl);

        // store indexer keys to KV cache
        const auto *vMctxLid = vInpAttnDsa->vMctx->mGetLid();
        const auto &vKIdxsLid = vInpAttnDsa->mGetKIdxsLid();
        ggml_build_forward_expand(vGf, vMctxLid->mCpyK(vCtx0, vIndexerK, vKIdxsLid, vIl));

        // prepare indexer weights
        ggml_tensor *vIndexerWeights = ggml_mul_mat(vCtx0, pModel.lLayers[vIl].vIndexerProj, vCur);
        mCb(vIndexerWeights, "indexer_weights", vIl);

        // get cached indexer keys
        vIndexerK = vMctxLid->mGetK(vCtx0, vIl);

        // split the batch into streams if needed
        const auto cNStream = vIndexerK->ne[3];
        vIndexerQ = ggml_view_4d(vCtx0, vIndexerQ, vIndexerQ->ne[0], vIndexerQ->ne[1], vIndexerQ->ne[2] / cNStream,
                                 cNStream, vIndexerQ->nb[1], vIndexerQ->nb[2], vIndexerQ->nb[3] / cNStream, 0);
        vIndexerWeights = ggml_view_4d(
            vCtx0, vIndexerWeights, vIndexerWeights->ne[0], vIndexerWeights->ne[1] / cNStream, vIndexerWeights->ne[2],
            cNStream, vIndexerWeights->nb[1], vIndexerWeights->nb[2] / cNStream, vIndexerWeights->nb[3] / cNStream, 0);

        // pre-scale weights to avoid scaling operations on huge indexer_score tensor
        vIndexerWeights = ggml_scale(vCtx0, vIndexerWeights, 1.0f / sqrtf(float(cNEmbdIndexerHead * cNIndexerHead)));
        mCb(vIndexerWeights, "indexer_weights", vIl);

        ggml_tensor *vIndexerScore = nullptr;
        if (vCparams.vFusedLid) {
          vIndexerScore =
              ggml_lightning_indexer(vCtx0, vIndexerQ, vIndexerK, vIndexerWeights, vInpAttnDsa->mGetKqMaskLid());
          mCb(vIndexerScore, "indexer_score", vIl);
          vRes->mAddFusedNode({cLlmFusedOpLightningIndexer, vIndexerScore, vIl});
        } else {
          // calculate indexer kq
          vIndexerQ = ggml_permute(vCtx0, vIndexerQ, 0, 2, 1, 3);
          mCb(vIndexerQ, "indexer_q", vIl);
          vIndexerK = ggml_permute(vCtx0, vIndexerK, 0, 2, 1, 3);
          mCb(vIndexerK, "indexer_k", vIl);

          ggml_tensor *vIndexerKq = ggml_mul_mat(vCtx0, vIndexerK, vIndexerQ);
          mCb(vIndexerKq, "indexer_kq", vIl);

          // ReLU requires contiguous tensors
          vIndexerKq = ggml_cont(vCtx0, ggml_permute(vCtx0, vIndexerKq, 2, 1, 0, 3));
          mCb(vIndexerKq, "indexer_kq", vIl);

          // apply ReLU
          vIndexerScore = ggml_relu(vCtx0, vIndexerKq);
          mCb(vIndexerScore, "indexer_score", vIl);

          // multiply scores by indexer weights
          vIndexerScore = ggml_mul(vCtx0, vIndexerScore, vIndexerWeights);
          mCb(vIndexerScore, "indexer_score", vIl);

          // sum by q n_indexer_head dimension
          vIndexerScore = ggml_sum_rows(vCtx0, vIndexerScore);
          mCb(vIndexerScore, "indexer_score", vIl);

          // permute result to match KQ mask
          vIndexerScore = ggml_cont(vCtx0, ggml_permute(vCtx0, vIndexerScore, 2, 1, 0, 3));
          mCb(vIndexerScore, "indexer_score", vIl);

          // mask indexer scores
          ggml_tensor *vIndexerKqMask = vInpAttnDsa->mGetKqMaskLid();
          vIndexerScore = ggml_add(vCtx0, vIndexerScore, vIndexerKqMask);
          mCb(vIndexerScore, "indexer_score", vIl);
        }

        // get indices of top k indexer scores
        uint32_t vNTopK = vIndexerScore->ne[0] < cNIndexerTopK ? vIndexerScore->ne[0] : cNIndexerTopK;
        vTopK = ggml_cont(vCtx0, ggml_top_k(vCtx0, vIndexerScore, vNTopK));
        vPrevTopK = vTopK;
        mCb(vTopK, "top_k", vIl);
      } else {
        // "shared" indexer layer - reuse top-k from a previous full layer
        GGML_ASSERT(vPrevTopK != nullptr && "shared indexer layer must follow a previous full indexer layer");
        vTopK = vPrevTopK;
        mCb(vTopK, "top_k", vIl);
      }

      ggml_tensor *vQ = ggml_mul_mat(vCtx0, pModel.lLayers[vIl].vWqB, vQr);
      mCb(vQ, "q", vIl);

      // split into {n_embd_head_qk_nope, n_head, n_tokens}
      ggml_tensor *vQNope =
          ggml_view_3d(vCtx0, vQ, cNEmbdHeadQkNope, cNHead, cNTokens, ggml_row_size(vQ->type, cNEmbdHeadK),
                       ggml_row_size(vQ->type, cNEmbdHeadK) * cNHead, 0);
      mCb(vQNope, "q_nope", vIl);

      // and {n_embd_head_qk_rope, n_head, n_tokens}
      ggml_tensor *vQPe =
          ggml_view_3d(vCtx0, vQ, cNEmbdHeadQkRope, cNHead, cNTokens, ggml_row_size(vQ->type, cNEmbdHeadK),
                       ggml_row_size(vQ->type, cNEmbdHeadK) * cNHead, ggml_row_size(vQ->type, cNEmbdHeadQkNope));
      mCb(vQPe, "q_pe", vIl);

      ggml_tensor *vKvCmprPe = ggml_mul_mat(vCtx0, pModel.lLayers[vIl].vWkvAMqa, vCur);
      mCb(vKvCmprPe, "kv_cmpr_pe", vIl);

      // split into {kv_lora_rank, n_tokens}
      ggml_tensor *vKvCmpr = ggml_view_2d(vCtx0, vKvCmprPe, cKvLoraRank, cNTokens,
                                          ggml_row_size(vKvCmprPe->type, cKvLoraRank + cNEmbdHeadQkRope), 0);
      mCb(vKvCmpr, "kv_cmpr", vIl);

      // and {n_embd_head_qk_rope, 1, n_tokens}
      ggml_tensor *vKPe = ggml_view_3d(vCtx0, vKvCmprPe, cNEmbdHeadQkRope, 1, cNTokens,
                                       ggml_row_size(vKvCmprPe->type, cKvLoraRank + cNEmbdHeadQkRope),
                                       ggml_row_size(vKvCmprPe->type, cKvLoraRank + cNEmbdHeadQkRope),
                                       ggml_row_size(vKvCmprPe->type, cKvLoraRank));
      mCb(vKPe, "k_pe", vIl);

      vQPe = ggml_rope_ext(vCtx0, vQPe, vInpPos, nullptr, cNRot, cRopeType, cNCtxOrig, cFreqBase, cFreqScale,
                           cExtFactor, cAttnFactor, cBetaFast, cBetaSlow);
      mCb(vQPe, "q_pe", vIl);

      vKPe = ggml_rope_ext(vCtx0, vKPe, vInpPos, nullptr, cNRot, cRopeType, cNCtxOrig, cFreqBase, cFreqScale,
                           cExtFactor, cAttnFactor, cBetaFast, cBetaSlow);
      mCb(vKPe, "k_pe", vIl);

      vKvCmpr = mBuildNorm(vKvCmpr, pModel.lLayers[vIl].vAttnKvANorm, nullptr, cLlmNormRms, vIl);
      mCb(vKvCmpr, "kv_cmpr", vIl);

      // MLA attention
      {
        // {n_embd_head_qk_nope, n_tokens, n_head}
        vQNope = ggml_permute(vCtx0, vQNope, 0, 2, 1, 3);
        mCb(vQNope, "q_nope_perm", vIl);

        // {n_embd_head_qk_nope, kv_lora_rank, n_head} x {n_embd_head_qk_nope, n_tokens, n_head}
        ggml_tensor *vQNopeAbsorbed = ggml_mul_mat(vCtx0, pModel.lLayers[vIl].vWkB, vQNope);
        mCb(vQNopeAbsorbed, "q_nope_absorbed", vIl);

        // {kv_lora_rank, n_head, n_tokens}
        vQNopeAbsorbed = ggml_permute(vCtx0, vQNopeAbsorbed, 0, 2, 1, 3);
        mCb(vQNopeAbsorbed, "q_nope_absorbed_perm", vIl);

        // {n_embd_head_qk_rope + kv_lora_rank, n_head, n_tokens}
        // note: rope must go first for in-place context shifting in mBuildRopeShift()
        ggml_tensor *vQcur = ggml_concat(vCtx0, vQNopeAbsorbed, vQPe, 0);
        mCb(vQcur, "Qcur", vIl);

        vKvCmpr = ggml_reshape_3d(vCtx0, vKvCmpr, cKvLoraRank, 1, cNTokens);
        mCb(vKvCmpr, "kv_cmpr_reshape", vIl);

        // {n_embd_head_qk_rope + kv_lora_rank, 1, n_tokens}
        ggml_tensor *vKcur = ggml_concat(vCtx0, vKvCmpr, vKPe, 0);
        mCb(vKcur, "Kcur", vIl);

        // {kv_lora_rank, 1, n_tokens}
        ggml_tensor *vVcur = vKvCmpr;
        mCb(vVcur, "Vcur", vIl);

        // note: MLA with the absorption optimization converts into MQA (ie: GQA with 1 group)
        vCur = mBuildAttn(vInpAttnDsa, pModel.lLayers[vIl].vWo, NULL, pModel.lLayers[vIl].vWoS, vQcur, vKcur, vVcur,
                          nullptr, nullptr, pModel.lLayers[vIl].vWvB, vTopK, cKqScale, vIl);
      }
    }
    // when unmasked nextn embeddings are requested, t_h_nextn must keep all rows,
    // so the early output masking has to be skipped (it is applied after the final norm instead)
    if (vIl == cNLayer - 1 && vInpOutIds && (!vCparams.vEmbeddingsNextn || vCparams.vEmbeddingsNextnMasked)) {
      vCur = ggml_get_rows(vCtx0, vCur, vInpOutIds);
      vInpSA = ggml_get_rows(vCtx0, vInpSA, vInpOutIds);
    }
    ggml_tensor *vFfnInp = ggml_add(vCtx0, vCur, vInpSA);
    mCb(vFfnInp, "ffn_inp", vIl);

    vCur = mBuildNorm(vFfnInp, pModel.lLayers[vIl].vFfnNorm, NULL, cLlmNormRms, vIl);
    mCb(vCur, "ffn_norm", vIl);

    if ((uint32_t)vIl < vHparams.vNLayerDenseLead) {
      vCur = mBuildFfn(vCur, pModel.lLayers[vIl].vFfnUp, NULL, pModel.lLayers[vIl].vFfnUpS,
                       pModel.lLayers[vIl].vFfnGate, NULL, pModel.lLayers[vIl].vFfnGateS, pModel.lLayers[vIl].vFfnDown,
                       NULL, pModel.lLayers[vIl].vFfnDownS, NULL, cLlmFfnSilu, cLlmFfnPar, vIl);
      mCb(vCur, "ffn_out", vIl);
    } else {
      // MoE branch
      ggml_tensor *vMoeOut = mBuildMoeFfn(
          vCur, pModel.lLayers[vIl].vFfnGateInp, pModel.lLayers[vIl].vFfnUpExps, pModel.lLayers[vIl].vFfnGateExps,
          pModel.lLayers[vIl].vFfnDownExps, pModel.lLayers[vIl].vFfnExpProbsB, cNExpert, cNExpertUsed, cLlmFfnSilu,
          vHparams.vExpertWeightsNorm, vHparams.vExpertWeightsScale,
          (mgs_expert_gating_func_type)vHparams.vExpertGatingFunc, vIl, nullptr, pModel.lLayers[vIl].vFfnGateUpExps,
          pModel.lLayers[vIl].vFfnUpExpsS, pModel.lLayers[vIl].vFfnGateExpsS, pModel.lLayers[vIl].vFfnDownExpsS);
      mCb(vMoeOut, "ffn_moe_out", vIl);

      // FFN shared expert
      {
        ggml_tensor *vFfnShexp = mBuildFfn(vCur, pModel.lLayers[vIl].vFfnUpShexp, NULL,
                                           pModel.lLayers[vIl].vFfnUpShexpS, pModel.lLayers[vIl].vFfnGateShexp, NULL,
                                           pModel.lLayers[vIl].vFfnGateShexpS, pModel.lLayers[vIl].vFfnDownShexp, NULL,
                                           pModel.lLayers[vIl].vFfnDownShexpS, NULL, cLlmFfnSilu, cLlmFfnPar, vIl);
        mCb(vFfnShexp, "ffn_shexp", vIl);

        vCur = ggml_add(vCtx0, vMoeOut, vFfnShexp);
        mCb(vCur, "ffn_out", vIl);
      }
    }
    vCur = ggml_add(vCtx0, vCur, vFfnInp);

    vCur = mBuildCvec(vCur, vIl);
    mCb(vCur, "l_out", vIl);

    // input for next layer
    vInpL = vCur;
  }
  vCur = vInpL;

  vCur = mBuildNorm(vCur, pModel.vOutputNorm, NULL, cLlmNormRms, -1);

  // post-norm hidden state feeds the NextN/MTP draft head
  mCb(vCur, "h_nextn", -1);
  vRes->vTHNextn = vCur;

  if (vCparams.vEmbeddingsNextn && !vCparams.vEmbeddingsNextnMasked && vInpOutIds) {
    vCur = ggml_get_rows(vCtx0, vCur, vInpOutIds);
  }

  mCb(vCur, "result_norm", -1);
  vRes->vTEmbd = vCur;

  // lm_head
  vCur = ggml_mul_mat(vCtx0, pModel.vOutput, vCur);

  mCb(vCur, "result_output", -1);
  vRes->vTLogits = vCur;

  ggml_build_forward_expand(vGf, vCur);
}

// LLM_GRAPH_TYPE_DECODER_MTP draft head for GLM-5.2 (GLM_DSA).
// Semantics mirror the deepseek-family NextN/MTP layer:
//   enorm(embed) + hnorm(prev_hidden) -> concat(e, h) -> eh_proj ->
//   full glm_dsa decoder block (dense MLA attention + sigmoid-gated MoE FFN
//   with shared expert, exactly as the trunk graph builds it) ->
//   shared_head_norm (fallback output_norm) -> shared LM head.
// The DSA indexer is not used at runtime (same as the trunk graph).
mgs_model_glm_dsa::graph_mtp::graph_mtp(const mgs_model &pModel, const llm_graph_params &pParams)
    : llm_graph_context(pParams) {
  GGML_ASSERT(vHparams.vNLayerNextn > 0 && "GLM_DSA MTP requires n_layer_nextn > 0");
  GGML_ASSERT(vHparams.vNLayerNextn == 1 && "GLM_DSA MTP currently only supports a single MTP block");

  const int cIl = vHparams.mNLayer() + vCparams.vNextnLayerOffset;
  GGML_ASSERT(vCparams.vNextnLayerOffset >= 0 && vCparams.vNextnLayerOffset < (int)vHparams.vNLayerNextn &&
              "nextn_layer_offset out of range [0, n_layer_nextn)");
  const auto &vLayer = pModel.lLayers[cIl];

  GGML_ASSERT(vLayer.vNextn.vEhProj && "MTP block missing nextn.eh_proj");
  GGML_ASSERT(vLayer.vNextn.vEnorm && "MTP block missing nextn.enorm");
  GGML_ASSERT(vLayer.vNextn.vHnorm && "MTP block missing nextn.hnorm");
  GGML_ASSERT(vLayer.vFfnGateInp && "MTP block missing ffn_gate_inp");

  // note: these are the actual head sizes you get when treating as MHA or after "decompression" using wv_b for MLA
  const int64_t cNEmbdHeadK = vHparams.mNEmbdHeadKMla();

  const int64_t cNEmbdHeadQkRope = vHparams.mNRot();
  const int64_t cNEmbdHeadQkNope = cNEmbdHeadK - cNEmbdHeadQkRope;

  const uint32_t cKvLoraRank = vHparams.vNLoraKv;

  // We have to pre-scale kq_scale and attn_factor to make the YaRN RoPE work correctly.
  // See the trunk graph for the detailed explanation - this must match it EXACTLY.
  GGML_ASSERT(cExtFactor >= 0.0f);
  const float cAttnFactorOrg = cAttnFactor * (1.0f + 0.1f * logf(1.0f / cFreqScale));

  const float cMscale = cAttnFactorOrg * (1.0f + 0.1f * vHparams.vRopeYarnLogMul * logf(1.0f / cFreqScale));
  const float cKqScale = 1.0f * cMscale * cMscale / sqrtf(float(cNEmbdHeadK));

  // TODO: extract in a common llm_graph_context::build_inp_embd_h()
  auto vInp = std::make_unique<llm_graph_input_embd_h>(vHparams.vNEmbd);

  vInp->vTokens = ggml_new_tensor_1d(vCtx0, GGML_TYPE_I32, cNTokens);
  ggml_set_input(vInp->vTokens);

  vInp->vEmbd = ggml_new_tensor_2d(vCtx0, GGML_TYPE_F32, vHparams.mNEmbdInp(), cNTokens);
  ggml_set_input(vInp->vEmbd);

  ggml_tensor *vTokEmbd;
  if (vUbatch.vToken) {
    ggml_tensor *vTokEmbdW = vLayer.vNextn.vEmbedTokens ? vLayer.vNextn.vEmbedTokens : pModel.vTokEmbd;

    vTokEmbd = ggml_get_rows(vCtx0, vTokEmbdW, vInp->vTokens);
  } else {
    vTokEmbd = vInp->vEmbd;
  }
  mCb(vTokEmbd, "mtp_tok_embd", cIl);

  vInp->vH = ggml_new_tensor_2d(vCtx0, GGML_TYPE_F32, vHparams.vNEmbd, cNTokens);
  ggml_set_input(vInp->vH);
  ggml_set_name(vInp->vH, "mtp_h_input");

  ggml_tensor *vHEmbd = vInp->vH;

  vRes->mAddInput(std::move(vInp));

  ggml_tensor *vInpPos = mBuildInpPos();
  ggml_tensor *vInpOutIds = mBuildInpOutIds();

  // MLA with the absorption optimization uses a K-only cache (V is a view of K)
  auto *vInpAttn = mBuildAttnInpK();

  ggml_tensor *vHNorm = mBuildNorm(vHEmbd, vLayer.vNextn.vHnorm, nullptr, cLlmNormRms, cIl);
  mCb(vHNorm, "mtp_hnorm", cIl);

  ggml_tensor *vENorm = mBuildNorm(vTokEmbd, vLayer.vNextn.vEnorm, nullptr, cLlmNormRms, cIl);
  mCb(vENorm, "mtp_enorm", cIl);

  ggml_tensor *vConcat = ggml_concat(vCtx0, vENorm, vHNorm, /*dim=*/0);
  mCb(vConcat, "mtp_concat", cIl);

  ggml_tensor *vCur = mBuildLoraMm(vLayer.vNextn.vEhProj, vConcat, vLayer.vNextn.vEhProjS);
  mCb(vCur, "mtp_eh_proj", cIl);

  ggml_tensor *vInpSA = vCur;

  vCur = mBuildNorm(vCur, vLayer.vAttnNorm, nullptr, cLlmNormRms, cIl);
  mCb(vCur, "mtp_attn_norm", cIl);

  // self-attention: dense MLA, same construction as the trunk graph
  {
    ggml_tensor *vQ = ggml_mul_mat(vCtx0, vLayer.vWqA, vCur);
    mCb(vQ, "mtp_q", cIl);

    vQ = mBuildNorm(vQ, vLayer.vAttnQANorm, nullptr, cLlmNormRms, cIl);
    mCb(vQ, "mtp_q", cIl);

    vQ = ggml_mul_mat(vCtx0, vLayer.vWqB, vQ);
    mCb(vQ, "mtp_q", cIl);

    // split into {n_embd_head_qk_nope, n_head, n_tokens}
    ggml_tensor *vQNope =
        ggml_view_3d(vCtx0, vQ, cNEmbdHeadQkNope, cNHead, cNTokens, ggml_row_size(vQ->type, cNEmbdHeadK),
                     ggml_row_size(vQ->type, cNEmbdHeadK) * cNHead, 0);
    mCb(vQNope, "mtp_q_nope", cIl);

    // and {n_embd_head_qk_rope, n_head, n_tokens}
    ggml_tensor *vQPe =
        ggml_view_3d(vCtx0, vQ, cNEmbdHeadQkRope, cNHead, cNTokens, ggml_row_size(vQ->type, cNEmbdHeadK),
                     ggml_row_size(vQ->type, cNEmbdHeadK) * cNHead, ggml_row_size(vQ->type, cNEmbdHeadQkNope));
    mCb(vQPe, "mtp_q_pe", cIl);

    ggml_tensor *vKvCmprPe = ggml_mul_mat(vCtx0, vLayer.vWkvAMqa, vCur);
    mCb(vKvCmprPe, "mtp_kv_cmpr_pe", cIl);

    // split into {kv_lora_rank, n_tokens}
    ggml_tensor *vKvCmpr = ggml_view_2d(vCtx0, vKvCmprPe, cKvLoraRank, cNTokens,
                                        ggml_row_size(vKvCmprPe->type, cKvLoraRank + cNEmbdHeadQkRope), 0);
    mCb(vKvCmpr, "mtp_kv_cmpr", cIl);

    // and {n_embd_head_qk_rope, 1, n_tokens}
    ggml_tensor *vKPe = ggml_view_3d(
        vCtx0, vKvCmprPe, cNEmbdHeadQkRope, 1, cNTokens, ggml_row_size(vKvCmprPe->type, cKvLoraRank + cNEmbdHeadQkRope),
        ggml_row_size(vKvCmprPe->type, cKvLoraRank + cNEmbdHeadQkRope), ggml_row_size(vKvCmprPe->type, cKvLoraRank));
    mCb(vKPe, "mtp_k_pe", cIl);

    vQPe = ggml_rope_ext(vCtx0, vQPe, vInpPos, nullptr, cNRot, cRopeType, cNCtxOrig, cFreqBase, cFreqScale, cExtFactor,
                         cAttnFactor, cBetaFast, cBetaSlow);
    mCb(vQPe, "mtp_q_pe", cIl);

    vKPe = ggml_rope_ext(vCtx0, vKPe, vInpPos, nullptr, cNRot, cRopeType, cNCtxOrig, cFreqBase, cFreqScale, cExtFactor,
                         cAttnFactor, cBetaFast, cBetaSlow);
    mCb(vKPe, "mtp_k_pe", cIl);

    vKvCmpr = mBuildNorm(vKvCmpr, vLayer.vAttnKvANorm, nullptr, cLlmNormRms, cIl);
    mCb(vKvCmpr, "mtp_kv_cmpr", cIl);

    // {n_embd_head_qk_nope, n_tokens, n_head}
    vQNope = ggml_permute(vCtx0, vQNope, 0, 2, 1, 3);
    mCb(vQNope, "mtp_q_nope_perm", cIl);

    // {n_embd_head_qk_nope, kv_lora_rank, n_head} x {n_embd_head_qk_nope, n_tokens, n_head}
    ggml_tensor *vQNopeAbsorbed = ggml_mul_mat(vCtx0, vLayer.vWkB, vQNope);
    mCb(vQNopeAbsorbed, "mtp_q_nope_absorbed", cIl);

    // {kv_lora_rank, n_head, n_tokens}
    vQNopeAbsorbed = ggml_permute(vCtx0, vQNopeAbsorbed, 0, 2, 1, 3);
    mCb(vQNopeAbsorbed, "mtp_q_nope_absorbed_perm", cIl);

    // {n_embd_head_qk_rope + kv_lora_rank, n_head, n_tokens}
    // note: rope must go first for in-place context shifting in mBuildRopeShift()
    ggml_tensor *vQcur = ggml_concat(vCtx0, vQNopeAbsorbed, vQPe, 0);
    mCb(vQcur, "mtp_Qcur", cIl);

    vKvCmpr = ggml_reshape_3d(vCtx0, vKvCmpr, cKvLoraRank, 1, cNTokens);
    mCb(vKvCmpr, "mtp_kv_cmpr_reshape", cIl);

    // {n_embd_head_qk_rope + kv_lora_rank, 1, n_tokens}
    ggml_tensor *vKcur = ggml_concat(vCtx0, vKvCmpr, vKPe, 0);
    mCb(vKcur, "mtp_Kcur", cIl);

    // {kv_lora_rank, 1, n_tokens}
    ggml_tensor *vVcur = vKvCmpr;
    mCb(vVcur, "mtp_Vcur", cIl);

    // note: MLA with the absorption optimization converts into MQA (ie: GQA with 1 group)
    vCur = mBuildAttn(vInpAttn, vLayer.vWo, NULL, vLayer.vWoS, vQcur, vKcur, vVcur, nullptr, nullptr, vLayer.vWvB,
                      cKqScale, cIl);
    mCb(vCur, "mtp_attn_out", cIl);
  }

  ggml_tensor *vFfnInp = ggml_add(vCtx0, vCur, vInpSA);
  mCb(vFfnInp, "mtp_ffn_inp", cIl);

  vCur = mBuildNorm(vFfnInp, vLayer.vFfnNorm, NULL, cLlmNormRms, cIl);
  mCb(vCur, "mtp_ffn_norm", cIl);

  // MoE FFN with shared expert - same construction as the trunk graph
  ggml_tensor *vMoeOut =
      mBuildMoeFfn(vCur, vLayer.vFfnGateInp, vLayer.vFfnUpExps, vLayer.vFfnGateExps, vLayer.vFfnDownExps,
                   vLayer.vFfnExpProbsB, cNExpert, cNExpertUsed, cLlmFfnSilu, vHparams.vExpertWeightsNorm,
                   vHparams.vExpertWeightsScale, (mgs_expert_gating_func_type)vHparams.vExpertGatingFunc, cIl, nullptr,
                   vLayer.vFfnGateUpExps, vLayer.vFfnUpExpsS, vLayer.vFfnGateExpsS, vLayer.vFfnDownExpsS);
  mCb(vMoeOut, "mtp_ffn_moe_out", cIl);

  // FFN shared expert
  ggml_tensor *vFfnShexp =
      mBuildFfn(vCur, vLayer.vFfnUpShexp, NULL, vLayer.vFfnUpShexpS, vLayer.vFfnGateShexp, NULL, vLayer.vFfnGateShexpS,
                vLayer.vFfnDownShexp, NULL, vLayer.vFfnDownShexpS, NULL, cLlmFfnSilu, cLlmFfnPar, cIl);
  mCb(vFfnShexp, "mtp_ffn_shexp", cIl);

  vCur = ggml_add(vCtx0, vMoeOut, vFfnShexp);
  mCb(vCur, "mtp_ffn_out", cIl);

  vCur = ggml_add(vCtx0, vCur, vFfnInp);
  mCb(vCur, "mtp_post_ffn", cIl);

  // shared_head_norm applied after the decoder block, before the shared LM head.
  // The post-norm hidden state seeds the next MTP step.
  ggml_tensor *vHeadNormW = vLayer.vNextn.vSharedHeadNorm ? vLayer.vNextn.vSharedHeadNorm : pModel.vOutputNorm;
  GGML_ASSERT(vHeadNormW && "GLM_DSA MTP: missing both nextn.shared_head_norm and output_norm");
  vCur = mBuildNorm(vCur, vHeadNormW, nullptr, cLlmNormRms, -1);

  mCb(vCur, "h_nextn", -1);
  vRes->vTHNextn = vCur;

  vCur = ggml_get_rows(vCtx0, vCur, vInpOutIds);
  mCb(vCur, "mtp_shared_head_norm", -1);

  ggml_tensor *vHeadW = vLayer.vNextn.vSharedHeadHead ? vLayer.vNextn.vSharedHeadHead : pModel.vOutput;
  ggml_tensor *vHeadS = vLayer.vNextn.vSharedHeadHead ? vLayer.vNextn.vSharedHeadHeadS : pModel.vOutputS;
  GGML_ASSERT(vHeadW && "GLM_DSA MTP: missing LM head (nextn.shared_head_head or model.output)");
  vCur = mBuildLoraMm(vHeadW, vCur, vHeadS);
  mCb(vCur, "result_output", -1);

  vRes->vTLogits = vCur;
  ggml_build_forward_expand(vGf, vCur);
}
