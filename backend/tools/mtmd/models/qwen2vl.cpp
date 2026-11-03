#include "models.h"

ggml_tensor *clip_graph_qwen2vl::mBuildInpWithTemporalMerge() {
  const clip_image_f32 &vImg = mGetImg(0);
  ggml_tensor *vInpRaw = mBuildInpRaw();

  GGML_ASSERT(vImg.mNx() % (cPatchSize * 2) == 0);
  GGML_ASSERT(vImg.mNy() % (cPatchSize * 2) == 0);

  const size_t cNb1 = ggml_row_size(vInpRaw->type, vImg.mNx());
  const size_t cNb2 = ggml_row_size(vInpRaw->type, vImg.mNx() * vImg.mNy());

  if (vNBatch == 1) {
    // still image input
    return ggml_add(vCtx0, ggml_conv_2d(vCtx0, vModel.vPatchEmbeddings0, vInpRaw, cPatchSize, cPatchSize, 0, 0, 1, 1),
                    ggml_conv_2d(vCtx0, vModel.vPatchEmbeddings1, vInpRaw, cPatchSize, cPatchSize, 0, 0, 1, 1));
  } else if (vNBatch == 2) {
    // 2 frames input (video input)
    ggml_tensor *vInp0 = ggml_view_3d(vCtx0, vInpRaw, vImg.mNx(), vImg.mNy(), 3, cNb1, cNb2, 0);
    ggml_tensor *vInp1 = ggml_view_3d(vCtx0, vInpRaw, vImg.mNx(), vImg.mNy(), 3, cNb1, cNb2,
                                      cNb2 * 3); // move to the second frame
    return ggml_add(vCtx0, ggml_conv_2d(vCtx0, vModel.vPatchEmbeddings0, vInp0, cPatchSize, cPatchSize, 0, 0, 1, 1),
                    ggml_conv_2d(vCtx0, vModel.vPatchEmbeddings1, vInp1, cPatchSize, cPatchSize, 0, 0, 1, 1));
  } else {
    GGML_ASSERT(false && "n_batch > 2 is not supported");
  }
}

ggml_cgraph *clip_graph_qwen2vl::mBuild() {
  GGML_ASSERT(vModel.vPatchBias == nullptr);
  GGML_ASSERT(vModel.vClassEmbedding == nullptr);

  const int cBatchSize = 1;
  const bool cUseWindowAttn = vHparams.vNWaPattern > 0;
  const int cNWaPattern = vHparams.vNWaPattern;
  const int cNPos = cNPatches;
  const int cNumPositionIds = cNPos * 4; // m-rope requires 4 dim per position

  norm_type vNormT = vProjType == cProjectorTypeQwen25vl ? cNormTypeRms     // qwen 2.5 vl
                                                         : cNormTypeNormal; // qwen 2 vl

  int aMropeSections[4] = {cDHead / 4, cDHead / 4, cDHead / 4, cDHead / 4};

  ggml_tensor *vInp = mBuildInpWithTemporalMerge();

  // second conv dimension
  {
    vInp = ggml_permute(vCtx0, vInp, 1, 2, 0, 3); // [w, h, c, b] -> [c, w, h, b]
    vInp = ggml_cont_4d(vCtx0, vInp, cNEmbd * 2, cNPatchesX / 2, cNPatchesY, cBatchSize);
    vInp = ggml_reshape_4d(vCtx0, vInp, cNEmbd * 2, cNPatchesX / 2, 2, cBatchSize * (cNPatchesY / 2));
    vInp = ggml_permute(vCtx0, vInp, 0, 2, 1, 3);
    vInp = ggml_cont_3d(vCtx0, vInp, cNEmbd, cNPatchesX * cNPatchesY, cBatchSize);
  }

  ggml_tensor *vInpL = vInp;
  ggml_tensor *vWindowMask = nullptr;
  ggml_tensor *vWindowIdx = nullptr;
  ggml_tensor *vInvWindowIdx = nullptr;

  ggml_tensor *vPositions = ggml_new_tensor_1d(vCtx0, GGML_TYPE_I32, cNumPositionIds);
  ggml_set_name(vPositions, "positions");
  ggml_set_input(vPositions);

  // pre-layernorm
  if (vModel.vPreLnW) {
    vInpL = mBuildNorm(vInpL, vModel.vPreLnW, vModel.vPreLnB, vNormT, cEps, -1);
  }

  if (cUseWindowAttn) {
    // handle window attention inputs
    vInvWindowIdx = ggml_new_tensor_1d(vCtx0, GGML_TYPE_I32, cNPos / 4);
    ggml_set_name(vInvWindowIdx, "inv_window_idx");
    ggml_set_input(vInvWindowIdx);
    // mask for window attention
    vWindowMask = ggml_new_tensor_2d(vCtx0, GGML_TYPE_F32, cNPos, cNPos);
    ggml_set_name(vWindowMask, "window_mask");
    ggml_set_input(vWindowMask);

    // if flash attn is used, we need to pad the mask and cast to f16
    if (cFlashAttnType == cClipFlashAttnTypeEnabled) {
      vWindowMask = ggml_cast(vCtx0, vWindowMask, GGML_TYPE_F16);
    }

    // inpL shape: [n_embd, n_patches_x * n_patches_y, batch_size]
    GGML_ASSERT(cBatchSize == 1);
    vInpL = ggml_reshape_2d(vCtx0, vInpL, cNEmbd * 4, cNPatchesX * cNPatchesY * cBatchSize / 4);
    vInpL = ggml_get_rows(vCtx0, vInpL, vInvWindowIdx);
    vInpL = ggml_reshape_3d(vCtx0, vInpL, cNEmbd, cNPatchesX * cNPatchesY, cBatchSize);
  }

  // loop over layers
  for (int vIl = 0; vIl < cNLayer; vIl++) {
    const auto &vLayer = vModel.lLayers[vIl];
    const bool cFullAttn = cUseWindowAttn ? (vIl + 1) % cNWaPattern == 0 : true;

    ggml_tensor *vCur = vInpL; // inpL = residual, cur = hidden_states

    // layernorm1
    vCur = mBuildNorm(vCur, vLayer.vLn1W, vLayer.vLn1B, vNormT, cEps, vIl);
    mCb(vCur, "ln1", vIl);

    // self-attention
    {
      ggml_tensor *vQcur = ggml_add(vCtx0, mBuildMm(vLayer.vQW, vCur), vLayer.vQB);
      ggml_tensor *vKcur = ggml_add(vCtx0, mBuildMm(vLayer.vKW, vCur), vLayer.vKB);
      ggml_tensor *vVcur = ggml_add(vCtx0, mBuildMm(vLayer.vVW, vCur), vLayer.vVB);

      vQcur = ggml_reshape_3d(vCtx0, vQcur, cDHead, cNHead, cNPatches);
      vKcur = ggml_reshape_3d(vCtx0, vKcur, cDHead, cNHead, cNPatches);
      vVcur = ggml_reshape_3d(vCtx0, vVcur, cDHead, cNHead, cNPatches);

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

      ggml_tensor *vAttnMask = cFullAttn ? nullptr : vWindowMask;

      vCur = mBuildAttn(vLayer.vOW, vLayer.vOB, vQcur, vKcur, vVcur, vAttnMask, vKqScale, vIl);
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

    vInpL = vCur;
  }

  // post-layernorm
  if (vModel.vPostLnW) {
    vInpL = mBuildNorm(vInpL, vModel.vPostLnW, vModel.vPostLnB, vNormT, cEps, cNLayer);
  }

  // multimodal projection
  ggml_tensor *vEmbeddings = vInpL;
  vEmbeddings = ggml_reshape_3d(vCtx0, vEmbeddings, cNEmbd * 4, cNPos / 4, cBatchSize);
  vEmbeddings =
      mBuildFfn(vEmbeddings, vModel.vMm0W, vModel.vMm0B, nullptr, nullptr, vModel.vMm1W, vModel.vMm1B, cFfnGelu, -1);

  if (cUseWindowAttn) {
    vWindowIdx = ggml_new_tensor_1d(vCtx0, GGML_TYPE_I32, cNPos / 4);
    ggml_set_name(vWindowIdx, "window_idx");
    ggml_set_input(vWindowIdx);

    // embeddings shape: [n_embd, n_patches_x * n_patches_y, batch_size]
    GGML_ASSERT(cBatchSize == 1);
    vEmbeddings = ggml_reshape_2d(vCtx0, vEmbeddings, vHparams.vProjectionDim, cNPatchesX * cNPatchesY / 4);
    vEmbeddings = ggml_get_rows(vCtx0, vEmbeddings, vWindowIdx);
    vEmbeddings = ggml_reshape_3d(vCtx0, vEmbeddings, vHparams.vProjectionDim, cNPatchesX * cNPatchesY / 4, cBatchSize);
  }

  // build the graph
  ggml_build_forward_expand(vGf, vEmbeddings);

  return vGf;
}
