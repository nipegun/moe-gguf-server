#include "models.h"

ggml_cgraph *clip_graph_youtuvl::mBuild() {
  GGML_ASSERT(vModel.vClassEmbedding == nullptr);
  const int cBatchSize = 1;
  const bool cUseWindowAttn = !vHparams.vWaLayerIndexes.empty();
  const int cNPos = cNPatches;
  const int cNumPositionIds = cNPos * 4;
  const int cM = 2;
  const int cWp = cNPatchesX;
  const int cHp = cNPatchesY;
  const int cHm = cHp / cM;
  const int cWm = cWp / cM;
  norm_type vNormT = cNormTypeNormal;

  int aMropeSections[4] = {cDHead / 4, cDHead / 4, cDHead / 4, cDHead / 4};

  ggml_tensor *vInp = mBuildInpRaw();

  // change conv3d to linear
  // reshape and permute to get patches, permute from (patch_size, m, Wm, patch_size, m, Hm, C) to (C, patch_size,
  // patch_size, m, m, Wm, Hm)
  {
    vInp = ggml_reshape_4d(vCtx0, vInp, cWm * cM * cPatchSize, cM * cPatchSize, cHm, 3);
    vInp = ggml_permute(vCtx0, vInp, 1, 2, 3, 0);
    vInp = ggml_cont_4d(vCtx0, vInp, cM * cPatchSize * 3, cWm, cM * cPatchSize, cHm);

    vInp = ggml_permute(vCtx0, vInp, 0, 2, 1, 3);
    vInp = ggml_cont_4d(vCtx0, vInp, cM * cPatchSize * 3, cPatchSize, cM, cHm * cWm);

    vInp = ggml_permute(vCtx0, vInp, 1, 0, 2, 3);
    vInp = ggml_cont_4d(vCtx0, vInp, cPatchSize, 3, cPatchSize, cHm * cWm * cM * cM);

    vInp = ggml_permute(vCtx0, vInp, 2, 0, 1, 3);
    vInp = ggml_cont_3d(vCtx0, vInp, 3 * cPatchSize * cPatchSize, cHm * cWm * cM * cM, 1);
  }
  vInp = mBuildMm(vModel.vPatchEmbeddings0, vInp);

  if (vModel.vPatchBias) {
    vInp = ggml_add(vCtx0, vInp, vModel.vPatchBias);
  }

  vInp = ggml_reshape_2d(vCtx0, vInp, cNEmbd, cNPatches);

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
    const bool cFullAttn = cUseWindowAttn ? vHparams.vWaLayerIndexes.count(vIl) > 0 : true;

    ggml_tensor *vCur = vInpL; // inpL = residual, cur = hidden_states

    // layernorm1
    vCur = mBuildNorm(vCur, vLayer.vLn1W, vLayer.vLn1B, vNormT, cEps, vIl);
    // self-attention
    {
      ggml_tensor *vQcur = ggml_add(vCtx0, mBuildMm(vLayer.vQW, vCur), vLayer.vQB);
      ggml_tensor *vKcur = ggml_add(vCtx0, mBuildMm(vLayer.vKW, vCur), vLayer.vKB);
      ggml_tensor *vVcur = ggml_add(vCtx0, mBuildMm(vLayer.vVW, vCur), vLayer.vVB);

      vQcur = ggml_reshape_3d(vCtx0, vQcur, cDHead, cNHead, cNPatches);
      vKcur = ggml_reshape_3d(vCtx0, vKcur, cDHead, cNHead, cNPatches);
      vVcur = ggml_reshape_3d(vCtx0, vVcur, cDHead, cNHead, cNPatches);

      vQcur = ggml_rope_multi(vCtx0, vQcur, vPositions, nullptr, cDHead / 2, aMropeSections, GGML_ROPE_TYPE_VISION,
                              32768, 10000, 1, 0, 1, 32, 1);
      vKcur = ggml_rope_multi(vCtx0, vKcur, vPositions, nullptr, cDHead / 2, aMropeSections, GGML_ROPE_TYPE_VISION,
                              32768, 10000, 1, 0, 1, 32, 1);

      ggml_tensor *vAttnMask = cFullAttn ? nullptr : vWindowMask;

      vCur = mBuildAttn(vLayer.vOW, vLayer.vOB, vQcur, vKcur, vVcur, vAttnMask, vKqScale, vIl);
    }
    // re-add the layer input, e.g., residual
    vCur = ggml_add(vCtx0, vCur, vInpL);

    vInpL = vCur; // inpL = residual, cur = hidden_states

    // layernorm2
    vCur = mBuildNorm(vCur, vLayer.vLn2W, vLayer.vLn2B, vNormT, cEps, vIl);

    // ffn
    vCur = mBuildFfn(vCur, vLayer.vFfUpW, vLayer.vFfUpB, nullptr, nullptr, vLayer.vFfDownW, vLayer.vFfDownB,
                     vHparams.vFfnOp, vIl);

    // residual 2
    vCur = ggml_add(vCtx0, vInpL, vCur);

    vInpL = vCur;
  }

  ggml_tensor *vEmbeddings = vInpL;
  if (cUseWindowAttn) {
    const int cSpatialMergeUnit = 4;
    vWindowIdx = ggml_new_tensor_1d(vCtx0, GGML_TYPE_I32, cNPos / cSpatialMergeUnit);
    ggml_set_name(vWindowIdx, "window_idx");
    ggml_set_input(vWindowIdx);
    GGML_ASSERT(cBatchSize == 1);
    vEmbeddings = ggml_reshape_2d(vCtx0, vEmbeddings, cNEmbd * cSpatialMergeUnit, cNPatches / cSpatialMergeUnit);
    vEmbeddings = ggml_get_rows(vCtx0, vEmbeddings, vWindowIdx);
    vEmbeddings = ggml_reshape_3d(vCtx0, vEmbeddings, cNEmbd, cNPatches, cBatchSize);
    mCb(vEmbeddings, "window_order_restored", -1);
  }

  // post-layernorm (part of Siglip2VisionTransformer, applied after encoder)
  if (vModel.vPostLnW) {
    vEmbeddings = mBuildNorm(vEmbeddings, vModel.vPostLnW, vModel.vPostLnB, vNormT, cEps, cNLayer);
  }

  // Now apply merger (VLPatchMerger):
  // 1. Apply RMS norm (ln_q in VLPatchMerger)
  vEmbeddings = mBuildNorm(vEmbeddings, vModel.vMmInputNormW, nullptr, cNormTypeRms, 1e-6, -1);
  mCb(vEmbeddings, "merger_normed", -1);

  // 2. First reshape for spatial merge (merge 2x2 patches)
  vEmbeddings = ggml_reshape_3d(vCtx0, vEmbeddings, cNEmbd * 4, cNPos / 4, cBatchSize);
  mCb(vEmbeddings, "merger_reshaped", -1);

  vEmbeddings =
      mBuildFfn(vEmbeddings, vModel.vMm0W, vModel.vMm0B, nullptr, nullptr, vModel.vMm1W, vModel.vMm1B, cFfnGelu, -1);
  ggml_build_forward_expand(vGf, vEmbeddings);

  return vGf;
}
