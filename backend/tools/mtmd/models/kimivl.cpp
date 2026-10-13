#include "models.h"

ggml_cgraph *clip_graph_kimivl::mBuild() {
  // 2D input positions
  ggml_tensor *vPosH = ggml_new_tensor_1d(vCtx0, GGML_TYPE_I32, cNPatches);
  ggml_set_name(vPosH, "pos_h");
  ggml_set_input(vPosH);

  ggml_tensor *vPosW = ggml_new_tensor_1d(vCtx0, GGML_TYPE_I32, cNPatches);
  ggml_set_name(vPosW, "pos_w");
  ggml_set_input(vPosW);

  ggml_tensor *vLearnedPosEmbd = mResizePositionEmbeddings();

  // build ViT with 2D position embeddings
  auto vAddPos = [&](ggml_tensor *pCur, const clip_layer &) {
    // first half is X axis and second half is Y axis
    return mBuildRope2d(vCtx0, pCur, vPosW, vPosH, vHparams.vRopeTheta, false);
  };

  ggml_tensor *vInp = mBuildInp();
  ggml_tensor *vCur = mBuildVit(vInp, cNPatches, cNormTypeNormal, vHparams.vFfnOp, vLearnedPosEmbd, vAddPos);

  mCb(vCur, "vit_out", -1);

  {
    // patch_merger
    const int cScaleFactor = vModel.vHparams.vNMerge;
    vCur = mBuildPatchMergePermute(vCur, cScaleFactor);

    // projection norm
    int vProjInpDim = vCur->ne[0];
    vCur = ggml_view_2d(vCtx0, vCur, cNEmbd, vCur->ne[1] * cScaleFactor * cScaleFactor,
                        ggml_row_size(vCur->type, cNEmbd), 0);
    vCur = ggml_norm(vCtx0, vCur, 1e-5); // default nn.LayerNorm
    vCur = ggml_mul(vCtx0, vCur, vModel.vMmInputNormW);
    vCur = ggml_add(vCtx0, vCur, vModel.vMmInputNormB);
    vCur = ggml_view_2d(vCtx0, vCur, vProjInpDim, vCur->ne[1] / cScaleFactor / cScaleFactor,
                        ggml_row_size(vCur->type, vProjInpDim), 0);
    mCb(vCur, "proj_inp_normed", -1);

    // projection mlp
    vCur = mBuildFfn(vCur, vModel.vMm1W, vModel.vMm1B, nullptr, nullptr, vModel.vMm2W, vModel.vMm2B, cFfnGelu, -1);
    mCb(vCur, "proj_out", -1);
  }

  // build the graph
  ggml_build_forward_expand(vGf, vCur);

  return vGf;
}
