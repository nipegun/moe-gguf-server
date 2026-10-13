#include "models.h"

ggml_cgraph *clip_graph_pixtral::mBuild() {
  const int cNMerge = vHparams.vNMerge;

  // 2D input positions
  ggml_tensor *vPosH = ggml_new_tensor_1d(vCtx0, GGML_TYPE_I32, cNPatches);
  ggml_set_name(vPosH, "pos_h");
  ggml_set_input(vPosH);

  ggml_tensor *vPosW = ggml_new_tensor_1d(vCtx0, GGML_TYPE_I32, cNPatches);
  ggml_set_name(vPosW, "pos_w");
  ggml_set_input(vPosW);

  auto vAddPos = [&](ggml_tensor *pCur, const clip_layer &) {
    return mBuildRope2d(vCtx0, pCur, vPosH, vPosW, vHparams.vRopeTheta, true);
  };

  ggml_tensor *vInp = mBuildInp();
  ggml_tensor *vCur = mBuildVit(vInp, cNPatches, cNormTypeRms, vHparams.vFfnOp,
                                nullptr, // no learned pos embd
                                vAddPos);

  // mistral small 3.1 patch merger
  // ref:
  // https://github.com/huggingface/transformers/blob/7a3e208892c06a5e278144eaf38c8599a42f53e7/src/transformers/models/mistral3/modeling_mistral3.py#L67
  if (vModel.vMmPatchMergerW) {
    GGML_ASSERT(vHparams.vNMerge > 0);

    vCur = ggml_mul(vCtx0, ggml_rms_norm(vCtx0, vCur, cEps), vModel.vMmInputNormW);

    // reshape image tokens to 2D grid
    vCur = ggml_reshape_3d(vCtx0, vCur, cNEmbd, cNPatchesX, cNPatchesY);
    vCur = ggml_permute(vCtx0, vCur, 2, 0, 1, 3); // [x, y, n_embd]
    vCur = ggml_cont(vCtx0, vCur);

    // torch.nn.functional.unfold is just an im2col under the hood
    // we just need a dummy kernel to make it work
    ggml_tensor *vKernel = ggml_view_3d(vCtx0, vCur, cNMerge, cNMerge, vCur->ne[2], 0, 0, 0);
    vCur = ggml_im2col(vCtx0, vKernel, vCur, cNMerge, cNMerge, 0, 0, 1, 1, true, vInp->type);

    // project to n_embd
    vCur = ggml_reshape_2d(vCtx0, vCur, vCur->ne[0], vCur->ne[1] * vCur->ne[2]);
    vCur = mBuildMm(vModel.vMmPatchMergerW, vCur);
  }

  // LlavaMultiModalProjector (always using GELU activation)
  {
    vCur = mBuildFfn(vCur, vModel.vMm1W, vModel.vMm1B, nullptr, nullptr, vModel.vMm2W, vModel.vMm2B, cFfnGelu, -1);
  }

  // arrangement of the [IMG_BREAK] token
  if (vModel.vTokenEmbdImgBreak) {
    // not efficient, but works
    // the trick is to view the embeddings as a 3D tensor with shape [n_embd, n_patches_per_row, n_rows]
    // and then concatenate the [IMG_BREAK] token to the end of each row, aka n_patches_per_row dimension
    // after the concatenation, we have a tensor with shape [n_embd, n_patches_per_row + 1, n_rows]

    const int cPY = cNPatchesY / cNMerge;
    const int cPX = cNPatchesX / cNMerge;
    const int cPTotal = cPX * cPY;
    const int cNEmbdText = vCur->ne[0];
    const int cNTokensOutput = cPTotal + cPY - 1; // one [IMG_BREAK] per row, except the last row

    ggml_tensor *vTmp = ggml_reshape_3d(vCtx0, vCur, cNEmbdText, cPX, cPY);
    ggml_tensor *vTok = ggml_new_tensor_3d(vCtx0, vTmp->type, cNEmbdText, 1, cPY);
    vTok = ggml_scale(vCtx0, vTok, 0.0); // clear the tensor
    vTok = ggml_add(vCtx0, vTok, vModel.vTokenEmbdImgBreak);
    vTmp = ggml_concat(vCtx0, vTmp, vTok, 1);
    vCur = ggml_view_2d(vCtx0, vTmp, cNEmbdText, cNTokensOutput, ggml_row_size(vTmp->type, cNEmbdText), 0);
  }

  // build the graph
  ggml_build_forward_expand(vGf, vCur);

  return vGf;
}
