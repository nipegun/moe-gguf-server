#include "models.h"
#include <cstring>
#include <cmath>

// note: this is similar to clip_graph::mResizePositionEmbeddings, major difference is having
// the w/h in ne[1] and ne[2] instead of assuming with sqrt. Could try storing the tensor in 2D instead
// with a w*h? Also the permute is a bit different at (2, 1, 0, 3) instead of (2, 0, 1, 3).
ggml_tensor *clip_graph_kimik25::mResizePositionEmbeddings3d(uint32_t pInterpolationMode) {
  ggml_tensor *vPosEmbd = vModel.vPositionEmbeddings;
  const int cHeight = mGetImg(0).mNy() / cPatchSize;
  const int cWidth = mGetImg(0).mNx() / cPatchSize;
  const uint32_t cMode = pInterpolationMode;

  GGML_ASSERT(vPosEmbd);

  const int64_t cStoredC = vPosEmbd->ne[0]; // C = 1152
  const int64_t cOrigW = vPosEmbd->ne[1];   // W = 64
  const int64_t cOrigH = vPosEmbd->ne[2];   // H = 64

  GGML_ASSERT(cStoredC == cNEmbd);

  if (cHeight == (int)cOrigH && cWidth == (int)cOrigW) {
    // No interpolation needed, just flatten to [C, H*W]
    return ggml_cont_2d(vCtx0, vPosEmbd, cNEmbd, cWidth * cHeight);
  }

  vPosEmbd = ggml_permute(vCtx0, vPosEmbd, 2, 1, 0, 3);
  vPosEmbd = ggml_interpolate(vCtx0, vPosEmbd, cHeight, cWidth, cNEmbd, 1, cMode);
  vPosEmbd = ggml_permute(vCtx0, vPosEmbd, 2, 1, 0, 3);
  vPosEmbd = ggml_cont_2d(vCtx0, vPosEmbd, cNEmbd, cWidth * cHeight);
  return vPosEmbd;
}

ggml_cgraph *clip_graph_kimik25::mBuild() {
  ggml_tensor *vPosH = ggml_new_tensor_1d(vCtx0, GGML_TYPE_I32, cNPatches);
  ggml_set_name(vPosH, "pos_h");
  ggml_set_input(vPosH);

  ggml_tensor *vPosW = ggml_new_tensor_1d(vCtx0, GGML_TYPE_I32, cNPatches);
  ggml_set_name(vPosW, "pos_w");
  ggml_set_input(vPosW);

  ggml_tensor *vLearnedPosEmbd = mResizePositionEmbeddings3d(GGML_SCALE_MODE_BICUBIC);

  // Kimi-K2.5 uses interleaved 2D RoPE pattern natively, but
  // Q / K are permuted during conversion to use split format.
  auto vAddPos = [&](ggml_tensor *pCur, const clip_layer &) {
    pCur = mBuildRope2d(vCtx0, pCur, vPosW, vPosH, vHparams.vRopeTheta, false);
    return pCur;
  };

  ggml_tensor *vInp = mBuildInp();

  // I don't know why, but doing this in the build_vit lead to the ggml_add not occurring?
  // Doing it manually here does work.
  vInp = ggml_add(vCtx0, vInp, vLearnedPosEmbd);

  ggml_tensor *vCur = mBuildVit(vInp, cNPatches, cNormTypeNormal, vHparams.vFfnOp, nullptr, vAddPos);

  mCb(vCur, "vit_out", -1);

  {
    // patch_merger
    const int cScaleFactor = vModel.vHparams.vNMerge;
    vCur = mBuildPatchMergePermute(vCur, cScaleFactor);

    // projection norm
    int vProjInpDim = vCur->ne[0];
    int vNMergedPatches = vCur->ne[1];
    vCur = ggml_view_2d(vCtx0, vCur, cNEmbd, vNMergedPatches * cScaleFactor * cScaleFactor,
                        ggml_row_size(vCur->type, cNEmbd), 0);
    vCur = ggml_norm(vCtx0, vCur, vHparams.vEps);
    vCur = ggml_mul(vCtx0, vCur, vModel.vMmInputNormW);
    vCur = ggml_add(vCtx0, vCur, vModel.vMmInputNormB);
    vCur = ggml_view_2d(vCtx0, vCur, vProjInpDim, vNMergedPatches, ggml_row_size(vCur->type, vProjInpDim), 0);
    mCb(vCur, "proj_inp_normed", -1);

    // projection mlp
    vCur = mBuildFfn(vCur, vModel.vMm1W, vModel.vMm1B, nullptr, nullptr, vModel.vMm2W, vModel.vMm2B, cFfnGelu, -1);

    mCb(vCur, "proj_out", -1);
  }

  // build the graph
  ggml_build_forward_expand(vGf, vCur);

  return vGf;
}
