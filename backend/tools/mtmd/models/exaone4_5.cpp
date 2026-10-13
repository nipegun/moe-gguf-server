// similar to qwen2vl, except for GQA attention
#include "models.h"

ggml_cgraph *clip_graph_exaone4_5::mBuild() {
  GGML_ASSERT(vModel.vPatchBias == nullptr);
  GGML_ASSERT(vModel.vClassEmbedding == nullptr);

  const int cBatchSize = 1;
  const bool cUseWindowAttn = vHparams.vNWaPattern > 0;
  const int cNWaPattern = vHparams.vNWaPattern;
  const int cNPos = cNPatches;
  const int cNumPositionIds = cNPos * 4;

  const norm_type cNormT = cNormTypeRms;

  const int64_t cNKvHead = vHparams.vNHeadKv > 0 ? vHparams.vNHeadKv : cNHead;
  GGML_ASSERT(cNHead % cNKvHead == 0);

  int aRopeSections[4] = {cDHead / 4, cDHead / 4, cDHead / 4, cDHead / 4};
  const float cRopeFreqBase = vHparams.vRopeTheta > 0.0f ? vHparams.vRopeTheta : 10000.0f;

  ggml_tensor *vInpRaw = mBuildInpRaw();
  ggml_tensor *vInp = ggml_conv_2d(vCtx0, vModel.vPatchEmbeddings0, vInpRaw, cPatchSize, cPatchSize, 0, 0, 1, 1);

  GGML_ASSERT(mGetImg(0).mNx() % (cPatchSize * 2) == 0);
  GGML_ASSERT(mGetImg(0).mNy() % (cPatchSize * 2) == 0);

  {
    ggml_tensor *vInp1 = ggml_conv_2d(vCtx0, vModel.vPatchEmbeddings1, vInpRaw, cPatchSize, cPatchSize, 0, 0, 1, 1);
    vInp = ggml_add(vCtx0, vInp, vInp1);
    vInp = ggml_permute(vCtx0, vInp, 1, 2, 0, 3);
    vInp = ggml_cont_4d(vCtx0, vInp, cNEmbd * 2, cNPatchesX / 2, cNPatchesY, cBatchSize);
    vInp = ggml_reshape_4d(vCtx0, vInp, cNEmbd * 2, cNPatchesX / 2, 2, cBatchSize * (cNPatchesY / 2));
    vInp = ggml_permute(vCtx0, vInp, 0, 2, 1, 3);
    vInp = ggml_cont_3d(vCtx0, vInp, cNEmbd, cNPatchesX * cNPatchesY, cBatchSize);
  }

  ggml_tensor *vPositions = ggml_new_tensor_1d(vCtx0, GGML_TYPE_I32, cNumPositionIds);
  ggml_set_name(vPositions, "positions");
  ggml_set_input(vPositions);

  ggml_tensor *vWindowMask = nullptr;
  ggml_tensor *vWindowIdx = nullptr;
  ggml_tensor *vInvWindowIdx = nullptr;

  if (cUseWindowAttn) {
    vWindowIdx = ggml_new_tensor_1d(vCtx0, GGML_TYPE_I32, cNPos / 4);
    ggml_set_name(vWindowIdx, "window_idx");
    ggml_set_input(vWindowIdx);

    vInvWindowIdx = ggml_new_tensor_1d(vCtx0, GGML_TYPE_I32, cNPos / 4);
    ggml_set_name(vInvWindowIdx, "inv_window_idx");
    ggml_set_input(vInvWindowIdx);

    vWindowMask = ggml_new_tensor_2d(vCtx0, GGML_TYPE_F32, cNPos, cNPos);
    ggml_set_name(vWindowMask, "window_mask");
    ggml_set_input(vWindowMask);

    if (cFlashAttnType == cClipFlashAttnTypeEnabled) {
      vWindowMask = ggml_cast(vCtx0, vWindowMask, GGML_TYPE_F16);
    }
  }

  ggml_tensor *vInpL = vInp;

  if (cUseWindowAttn) {
    GGML_ASSERT(cBatchSize == 1);
    vInpL = ggml_reshape_2d(vCtx0, vInpL, cNEmbd * 4, cNPatchesX * cNPatchesY * cBatchSize / 4);
    vInpL = ggml_get_rows(vCtx0, vInpL, vInvWindowIdx);
    vInpL = ggml_reshape_3d(vCtx0, vInpL, cNEmbd, cNPatchesX * cNPatchesY, cBatchSize);
  }

  for (int vIl = 0; vIl < cNLayer; vIl++) {
    const auto &vLayer = vModel.lLayers[vIl];
    const bool cFullAttn = cUseWindowAttn ? (vIl + 1) % cNWaPattern == 0 : true;
    ggml_tensor *vCur = vInpL;

    vCur = mBuildNorm(vCur, vLayer.vLn1W, vLayer.vLn1B, cNormT, cEps, vIl);
    mCb(vCur, "ln1", vIl);

    {
      GGML_ASSERT(vLayer.vQkvW != nullptr);
      vCur = mBuildMm(vLayer.vQkvW, vCur);
      if (vLayer.vQkvB) {
        vCur = ggml_add(vCtx0, vCur, vLayer.vQkvB);
      }

      const int64_t cNEmbdKv = cDHead * cNKvHead;
      ggml_tensor *vQcur =
          ggml_view_3d(vCtx0, vCur, cDHead, cNHead, cNPatches, ggml_row_size(vCur->type, cDHead), vCur->nb[1], 0);
      ggml_tensor *vKcur = ggml_view_3d(vCtx0, vCur, cDHead, cNKvHead, cNPatches, ggml_row_size(vCur->type, cDHead),
                                        vCur->nb[1], ggml_row_size(vCur->type, cNEmbd));
      ggml_tensor *vVcur = ggml_view_3d(vCtx0, vCur, cDHead, cNKvHead, cNPatches, ggml_row_size(vCur->type, cDHead),
                                        vCur->nb[1], ggml_row_size(vCur->type, cNEmbd + cNEmbdKv));

      mCb(vQcur, "Qcur", vIl);
      mCb(vKcur, "Kcur", vIl);
      mCb(vVcur, "Vcur", vIl);

      vQcur = ggml_rope_multi(vCtx0, vQcur, vPositions, nullptr, cDHead / 2, aRopeSections, GGML_ROPE_TYPE_VISION,
                              32768, cRopeFreqBase, 1, 0, 1, 32, 1);
      vKcur = ggml_rope_multi(vCtx0, vKcur, vPositions, nullptr, cDHead / 2, aRopeSections, GGML_ROPE_TYPE_VISION,
                              32768, cRopeFreqBase, 1, 0, 1, 32, 1);

      mCb(vQcur, "Qcur_rope", vIl);
      mCb(vKcur, "Kcur_rope", vIl);
      mCb(vVcur, "Vcur", vIl);

      ggml_tensor *vAttnMask = cFullAttn ? nullptr : vWindowMask;
      vCur = mBuildAttn(vLayer.vOW, vLayer.vOB, vQcur, vKcur, vVcur, vAttnMask, vKqScale, vIl);
      mCb(vCur, "attn_out", vIl);
    }

    vCur = ggml_add(vCtx0, vCur, vInpL);
    vInpL = vCur;

    mCb(vCur, "ffn_inp", vIl);

    vCur = mBuildNorm(vCur, vLayer.vLn2W, vLayer.vLn2B, cNormT, cEps, vIl);
    mCb(vCur, "ffn_inp_normed", vIl);

    vCur = mBuildFfn(vCur, vLayer.vFfUpW, vLayer.vFfUpB, vLayer.vFfGateW, vLayer.vFfGateB, vLayer.vFfDownW,
                     vLayer.vFfDownB, vHparams.vFfnOp, vIl);

    mCb(vCur, "ffn_out", vIl);

    vCur = ggml_add(vCtx0, vInpL, vCur);
    mCb(vCur, "layer_out", vIl);

    vInpL = vCur;
  }

  ggml_tensor *vEmbeddings = vInpL;
  vEmbeddings = mBuildNorm(vEmbeddings, vModel.vPostLnW, vModel.vPostLnB, cNormT, cEps, cNLayer);
  vEmbeddings = ggml_reshape_3d(vCtx0, vEmbeddings, cNEmbd * 4, cNPos / 4, cBatchSize);
  vEmbeddings =
      mBuildFfn(vEmbeddings, vModel.vMm0W, vModel.vMm0B, nullptr, nullptr, vModel.vMm1W, vModel.vMm1B, cFfnGelu, -1);

  if (cUseWindowAttn) {
    GGML_ASSERT(cBatchSize == 1);
    vEmbeddings = ggml_reshape_2d(vCtx0, vEmbeddings, vHparams.vProjectionDim, cNPatchesX * cNPatchesY / 4);
    vEmbeddings = ggml_get_rows(vCtx0, vEmbeddings, vWindowIdx);
    vEmbeddings = ggml_reshape_3d(vCtx0, vEmbeddings, vHparams.vProjectionDim, cNPatchesX * cNPatchesY / 4, cBatchSize);
  }

  ggml_build_forward_expand(vGf, vEmbeddings);

  return vGf;
}
