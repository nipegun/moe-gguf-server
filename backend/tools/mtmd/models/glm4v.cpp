#include "models.h"

ggml_cgraph *clip_graph_glm4v::mBuild() {
  GGML_ASSERT(vModel.vPatchBias != nullptr);
  GGML_ASSERT(vModel.vClassEmbedding == nullptr);

  const int cBatchSize = 1;

  norm_type vNormT = cNormTypeRms;

  ggml_tensor *vInpRaw = mBuildInpRaw();
  ggml_tensor *vInp = ggml_conv_2d(vCtx0, vModel.vPatchEmbeddings0, vInpRaw, cPatchSize, cPatchSize, 0, 0, 1, 1);

  int aMropeSections[4] = {cDHead / 4, cDHead / 4, cDHead / 4, cDHead / 4};
  ggml_tensor *vPositions = ggml_new_tensor_1d(vCtx0, GGML_TYPE_I32, cNPatches * 4);
  ggml_set_name(vPositions, "positions");
  ggml_set_input(vPositions);

  GGML_ASSERT(mGetImg(0).mNx() % (cPatchSize * 2) == 0);
  GGML_ASSERT(mGetImg(0).mNy() % (cPatchSize * 2) == 0);

  // second conv dimension
  {
    auto vInp1 = ggml_conv_2d(vCtx0, vModel.vPatchEmbeddings1, vInpRaw, cPatchSize, cPatchSize, 0, 0, 1, 1);
    vInp = ggml_add(vCtx0, vInp, vInp1);

    vInp = ggml_permute(vCtx0, vInp, 1, 2, 0, 3); // [w, h, c, b] -> [c, w, h, b]
    vInp = ggml_cont_4d(vCtx0, vInp, cNEmbd * 2, cNPatchesX / 2, cNPatchesY, cBatchSize);
    vInp = ggml_reshape_4d(vCtx0, vInp, cNEmbd * 2, cNPatchesX / 2, 2, cBatchSize * (cNPatchesY / 2));
    vInp = ggml_permute(vCtx0, vInp, 0, 2, 1, 3);
    vInp = ggml_cont_3d(vCtx0, vInp, cNEmbd, cNPatchesX * cNPatchesY, cBatchSize);
  }

  // add patch bias
  vInp = ggml_add(vCtx0, vInp, vModel.vPatchBias);
  mCb(vInp, "patch_bias", -1);

  // pos-conv norm (absent in GLM-5.3-Flash)
  if (vModel.vNormEmbdW) {
    vInp = mBuildNorm(vInp, vModel.vNormEmbdW, vModel.vNormEmbdB, vNormT, cEps, -1);
  }

  ggml_tensor *vLearnedPosEmbd = nullptr;
  // Note: GLM-OCR does not have learned position embeddings
  if (vModel.vPositionEmbeddings != nullptr) {
    vLearnedPosEmbd = mResizePositionEmbeddings(GGML_SCALE_MODE_BICUBIC);
    vLearnedPosEmbd = ggml_cont_4d(vCtx0, vLearnedPosEmbd, cNEmbd * 2, cNPatchesX / 2, cNPatchesY, cBatchSize);
    vLearnedPosEmbd =
        ggml_reshape_4d(vCtx0, vLearnedPosEmbd, cNEmbd * 2, cNPatchesX / 2, 2, cBatchSize * (cNPatchesY / 2));
    vLearnedPosEmbd = ggml_permute(vCtx0, vLearnedPosEmbd, 0, 2, 1, 3);
    vLearnedPosEmbd = ggml_cont_3d(vCtx0, vLearnedPosEmbd, cNEmbd, cNPatchesX * cNPatchesY, cBatchSize);
    mCb(vLearnedPosEmbd, "learned_pos_embd", -1);
  }

  auto vAddPos = [&](ggml_tensor *pCur, const clip_layer &) {
    return ggml_rope_multi(vCtx0, pCur, vPositions, nullptr, cDHead / 2, aMropeSections, GGML_ROPE_TYPE_VISION, 32768,
                           vHparams.vRopeTheta, 1, 0, 1, 32, 1);
  };

  ggml_tensor *vCur = mBuildVit(vInp, cNPatches, vNormT, vHparams.vFfnOp, vLearnedPosEmbd, vAddPos);

  mCb(vCur, "vit_out", -1);
  // mCb(ggml_sum(vCtx0, vCur), "vit_out_sum", -1);

  // GLM4V projector
  // ref:
  // https://github.com/huggingface/transformers/blob/40dc11cd3eb4126652aa41ef8272525affd4a636/src/transformers/models/glm4v/modeling_glm4v.py#L116-L130

  // patch merger (downsample)
  {
    int vNMerge = vHparams.vNMerge;
    GGML_ASSERT(vNMerge > 0);

    int vNTokenOut = cNPatches / vNMerge / vNMerge;
    vCur = ggml_reshape_4d(vCtx0, vCur, cNEmbd, vNMerge, vNMerge, vNTokenOut);
    vCur = ggml_cont(vCtx0, ggml_permute(vCtx0, vCur, 2, 0, 1, 3)); // [n_merge, n_merge, n_embd, n_token_out]
    vCur = ggml_conv_2d(vCtx0, vModel.vMmPatchMergerW, vCur, vNMerge, vNMerge, 0, 0, 1, 1);
    vCur = ggml_reshape_2d(vCtx0, vCur, vCur->ne[2], vNTokenOut); // [n_embd_out, n_token_out]

    vCur = ggml_add(vCtx0, vCur, vModel.vMmPatchMergerB);
  }

  // FC projector
  {
    vCur = mBuildMm(vModel.vMmFcW, vCur);
    // default LayerNorm (post_projection_norm)
    vCur = mBuildNorm(vCur, vModel.vMmPostNormW, vModel.vMmPostNormB, cNormTypeNormal, 1e-5, -1);
    vCur = ggml_gelu_erf(vCtx0, vCur);
    mCb(vCur, "after_fc_proj", -1);
  }

  // FFN projector
  {
    vCur = mBuildFfn(vCur, vModel.vMmFfnUpW, vModel.vMmFfnUpB, vModel.vMmFfnGateW, vModel.vMmFfnGateB,
                     vModel.vMmFfnDownW, vModel.vMmFfnDownB, vHparams.vFfnOp, -1);
    mCb(vCur, "after_ffn_proj", -1);
    // mCb(ggml_sum(vCtx0, vCur), "merged_sum", -1);
  }

  // build the graph
  ggml_build_forward_expand(vGf, vCur);

  return vGf;
}
