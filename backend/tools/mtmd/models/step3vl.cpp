#include "models.h"

ggml_cgraph *clip_graph_step3vl::mBuild() {
  GGML_ASSERT(vModel.vClassEmbedding == nullptr);
  GGML_ASSERT(vModel.vPatchEmbeddings0 != nullptr);
  GGML_ASSERT(vModel.vPositionEmbeddings != nullptr);

  norm_type vNormT = cNormTypeNormal;

  ggml_tensor *vPosH = ggml_new_tensor_1d(vCtx0, GGML_TYPE_I32, cNPatches);
  ggml_set_name(vPosH, "pos_h");
  ggml_set_input(vPosH);

  ggml_tensor *vPosW = ggml_new_tensor_1d(vCtx0, GGML_TYPE_I32, cNPatches);
  ggml_set_name(vPosW, "pos_w");
  ggml_set_input(vPosW);

  ggml_tensor *vInp = mBuildInp();
  ggml_tensor *vLearnedPosEmbd = mResizePositionEmbeddings();

  auto vAddPos = [&](ggml_tensor *pCur, const clip_layer &) {
    return mBuildRope2d(vCtx0, pCur, vPosW, vPosH, vHparams.vRopeTheta, false);
  };

  auto vAddSpatialBias = [&](ggml_tensor *pCur, ggml_tensor *pBias) {
    if (pBias == nullptr) {
      return pCur;
    }

    const int64_t cWidth = pCur->ne[0];
    const int64_t cHeight = pCur->ne[1];
    const int64_t cChannels = pCur->ne[2];

    pCur = ggml_reshape_2d(vCtx0, pCur, cWidth * cHeight, cChannels);
    pCur = ggml_cont(vCtx0, ggml_transpose(vCtx0, pCur));
    pCur = ggml_add(vCtx0, pCur, pBias);
    pCur = ggml_cont(vCtx0, ggml_transpose(vCtx0, pCur));
    pCur = ggml_reshape_3d(vCtx0, pCur, cWidth, cHeight, cChannels);

    return pCur;
  };

  ggml_tensor *vCur = mBuildVit(vInp, cNPatches, vNormT, vHparams.vFfnOp, vLearnedPosEmbd, vAddPos);
  mCb(vCur, "vit_out", -1);

  // [n_embd, n_patches] -> [w, h, n_embd] for spatial downsampling convolutions.
  vCur = ggml_permute(vCtx0, vCur, 1, 0, 2, 3);
  vCur = ggml_cont_3d(vCtx0, vCur, cNPatchesX, cNPatchesY, cNEmbd);

  // First downsampler: Conv2d(1536 -> 3072, k=3, s=2, p=1)
  vCur = ggml_conv_2d(vCtx0, vModel.vMm0W, vCur, 2, 2, 1, 1, 1, 1);
  vCur = vAddSpatialBias(vCur, vModel.vMm0B);
  mCb(vCur, "downsample_0", -1);

  // Second downsampler: Conv2d(3072 -> 6144, k=3, s=2, p=1)
  vCur = ggml_conv_2d(vCtx0, vModel.vMm1W, vCur, 2, 2, 1, 1, 1, 1);
  vCur = vAddSpatialBias(vCur, vModel.vMm1B);
  mCb(vCur, "downsample_1", -1);

  // [w, h, c] -> [c, w*h]
  {
    const int64_t cW = vCur->ne[0];
    const int64_t cH = vCur->ne[1];
    vCur = ggml_reshape_3d(vCtx0, vCur, cW * cH, vCur->ne[2], vCur->ne[3]);
    vCur = ggml_cont(vCtx0, ggml_permute(vCtx0, vCur, 1, 0, 2, 3));
  }
  mCb(vCur, "downsample_flatten", -1);

  // Final projector: Linear(6144 -> projection_dim)
  vCur = ggml_mul_mat(vCtx0, vModel.vMmModelProj, vCur);
  mCb(vCur, "projector_out", -1);

  ggml_build_forward_expand(vGf, vCur);
  return vGf;
}
