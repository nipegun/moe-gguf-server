#include "models.h"

ggml_cgraph *clip_graph_qwen3vl::mBuild() {
  GGML_ASSERT(vModel.vPatchBias != nullptr);
  GGML_ASSERT(vModel.vPositionEmbeddings != nullptr);
  GGML_ASSERT(vModel.vClassEmbedding == nullptr);

  const int cBatchSize = 1;
  const int cNPos = cNPatches;
  const int cNumPositionIds = cNPos * 4; // m-rope requires 4 dim per position

  norm_type vNormT = cNormTypeNormal;

  int aMropeSections[4] = {cDHead / 4, cDHead / 4, cDHead / 4, cDHead / 4};

  ggml_tensor *vInp = mBuildInpWithTemporalMerge();

  // spatial merge
  {
    vInp = ggml_permute(vCtx0, vInp, 1, 2, 0, 3); // [w, h, c, b] -> [c, w, h, b]
    vInp = ggml_cont_4d(vCtx0, vInp, cNEmbd * 2, cNPatchesX / 2, cNPatchesY, cBatchSize);
    vInp = ggml_reshape_4d(vCtx0, vInp, cNEmbd * 2, cNPatchesX / 2, 2, cBatchSize * (cNPatchesY / 2));
    vInp = ggml_permute(vCtx0, vInp, 0, 2, 1, 3);
    vInp = ggml_cont_3d(vCtx0, vInp, cNEmbd, cNPatchesX * cNPatchesY, cBatchSize);
  }

  // add patch bias
  if (vModel.vPatchBias != nullptr) {
    vInp = ggml_add(vCtx0, vInp, vModel.vPatchBias);
    mCb(vInp, "patch_bias", -1);
  }

  // calculate absolute position embedding and apply
  ggml_tensor *vLearnedPosEmbd = mResizePositionEmbeddings(GGML_SCALE_MODE_BILINEAR | GGML_SCALE_FLAG_ALIGN_CORNERS);
  vLearnedPosEmbd = ggml_cont_4d(vCtx0, vLearnedPosEmbd, cNEmbd * 2, cNPatchesX / 2, cNPatchesY, cBatchSize);
  vLearnedPosEmbd =
      ggml_reshape_4d(vCtx0, vLearnedPosEmbd, cNEmbd * 2, cNPatchesX / 2, 2, cBatchSize * (cNPatchesY / 2));
  vLearnedPosEmbd = ggml_permute(vCtx0, vLearnedPosEmbd, 0, 2, 1, 3);
  vLearnedPosEmbd = ggml_cont_3d(vCtx0, vLearnedPosEmbd, cNEmbd, cNPatchesX * cNPatchesY, cBatchSize);
  vInp = ggml_add(vCtx0, vInp, vLearnedPosEmbd);
  mCb(vInp, "inp_pos_emb", -1);

  ggml_tensor *vInpL = vInp;

  ggml_tensor *vPositions = ggml_new_tensor_1d(vCtx0, GGML_TYPE_I32, cNumPositionIds);
  ggml_set_name(vPositions, "positions");
  ggml_set_input(vPositions);

  // pre-layernorm
  if (vModel.vPreLnW) {
    vInpL = mBuildNorm(vInpL, vModel.vPreLnW, vModel.vPreLnB, vNormT, cEps, -1);
  }

  // deepstack features (stack along the feature dimension), [n_embd * len(deepstack_layers), n_patches_x * n_patches_y,
  // batch_size]
  ggml_tensor *vDeepstackFeatures = nullptr;
  const int cMergeFactor = vHparams.vNMerge > 0 ? vHparams.vNMerge * vHparams.vNMerge : 4; // default 2x2=4 for qwen3vl

  // loop over layers
  for (int vIl = 0; vIl < cNLayer; vIl++) {
    auto &vLayer = vModel.lLayers[vIl];

    ggml_tensor *vCur = vInpL; // inpL = residual, cur = hidden_states

    // layernorm1
    vCur = mBuildNorm(vCur, vLayer.vLn1W, vLayer.vLn1B, vNormT, cEps, vIl);
    mCb(vCur, "ln1", vIl);

    // self-attention
    {
      vCur = mBuildMm(vLayer.vQkvW, vCur);
      vCur = ggml_add(vCtx0, vCur, vLayer.vQkvB);

      ggml_tensor *vQcur = ggml_view_3d(vCtx0, vCur, cDHead, cNHead, cNPos,
                                        /* nb1    */ ggml_row_size(vCur->type, cDHead),
                                        /* nb2    */ vCur->nb[1],
                                        /* offset */ 0);

      ggml_tensor *vKcur = ggml_view_3d(vCtx0, vCur, cDHead, cNHead, cNPos,
                                        /* nb1    */ ggml_row_size(vCur->type, cDHead),
                                        /* nb2    */ vCur->nb[1],
                                        /* offset */ ggml_row_size(vCur->type, cNEmbd));

      ggml_tensor *vVcur = ggml_view_3d(vCtx0, vCur, cDHead, cNHead, cNPos,
                                        /* nb1    */ ggml_row_size(vCur->type, cDHead),
                                        /* nb2    */ vCur->nb[1],
                                        /* offset */ ggml_row_size(vCur->type, 2 * cNEmbd));

      mCb(vQcur, "Qcur", vIl);
      mCb(vKcur, "Kcur", vIl);
      mCb(vVcur, "Vcur", vIl);

      // apply M-RoPE
      vQcur = ggml_rope_multi(vCtx0, vQcur, vPositions, nullptr, cDHead / 2, aMropeSections, GGML_ROPE_TYPE_VISION,
                              32768, 10000, 1, 0, 1, 32, 1);
      vKcur = ggml_rope_multi(vCtx0, vKcur, vPositions, nullptr, cDHead / 2, aMropeSections, GGML_ROPE_TYPE_VISION,
                              32768, 10000, 1, 0, 1, 32, 1);

      mCb(vQcur, "Qcur_rope", vIl);
      mCb(vKcur, "Kcur_rope", vIl);

      vCur = mBuildAttn(vLayer.vOW, vLayer.vOB, vQcur, vKcur, vVcur, nullptr, vKqScale, vIl);
      mCb(vCur, "attn_out", vIl);
    }

    // re-add the layer input, e.g., residual
    vCur = ggml_add(vCtx0, vCur, vInpL);

    vInpL = vCur; // inpL = residual, cur = hidden_states

    mCb(vCur, "ffn_inp", vIl);

    // layernorm2
    vCur = mBuildNorm(vCur, vLayer.vLn2W, vLayer.vLn2B, vNormT, cEps, vIl);
    mCb(vCur, "ffn_inp_normed", vIl);

    // ffn
    vCur = mBuildFfn(vCur, vLayer.vFfUpW, vLayer.vFfUpB, vLayer.vFfGateW, vLayer.vFfGateB, vLayer.vFfDownW,
                     vLayer.vFfDownB, vHparams.vFfnOp, vIl);

    mCb(vCur, "ffn_out", vIl);

    // residual 2
    vCur = ggml_add(vCtx0, vInpL, vCur);
    mCb(vCur, "layer_out", vIl);

    if (vLayer.mHasDeepstack()) {
      ggml_tensor *vFeat = ggml_reshape_3d(vCtx0, vCur, cNEmbd * cMergeFactor, cNPos / cMergeFactor, cBatchSize);
      vFeat = mBuildNorm(vFeat, vLayer.vDeepstackNormW, vLayer.vDeepstackNormB, vNormT, cEps, vIl);
      vFeat = mBuildFfn(vFeat, vLayer.vDeepstackFc1W, vLayer.vDeepstackFc1B, nullptr, nullptr, vLayer.vDeepstackFc2W,
                        vLayer.vDeepstackFc2B, ffn_op_type::cFfnGelu, vIl);

      if (!vDeepstackFeatures) {
        vDeepstackFeatures = vFeat;
      } else {
        // concat along the feature dimension
        vDeepstackFeatures = ggml_concat(vCtx0, vDeepstackFeatures, vFeat, 0);
      }
    }

    vInpL = vCur;
  }

  // post-layernorm
  if (vModel.vPostLnW) {
    vInpL = mBuildNorm(vInpL, vModel.vPostLnW, vModel.vPostLnB, vNormT, cEps, cNLayer);
  }

  // multimodal projection
  ggml_tensor *vEmbeddings = vInpL;
  vEmbeddings = ggml_reshape_3d(vCtx0, vEmbeddings, cNEmbd * 4, cNPos / 4, cBatchSize);

  vEmbeddings = mBuildFfn(vEmbeddings, vModel.vMm0W, vModel.vMm0B, nullptr, nullptr, vModel.vMm1W, vModel.vMm1B,
                          ffn_op_type::cFfnGelu, -1);

  if (vDeepstackFeatures) {
    vEmbeddings = ggml_concat(vCtx0, vEmbeddings, vDeepstackFeatures, 0);
  } // concat along the feature dimension

  // build the graph
  ggml_build_forward_expand(vGf, vEmbeddings);

  return vGf;
}
