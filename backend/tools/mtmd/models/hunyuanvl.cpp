#include "models.h"

ggml_cgraph *clip_graph_hunyuanvl::mBuild() {
  const int cMerge = vHparams.vNMerge;
  const int cPw = cNPatchesX;
  const int cPh = cNPatchesY;

  // position embedding: declared as a graph input, filled on CPU
  // by fClipImageBatchEncode (see PROJECTOR_TYPE_HUNYUANVL branch there).
  ggml_tensor *vPosEmbd = ggml_new_tensor_2d(vCtx0, GGML_TYPE_F32, cNEmbd, cPh * cPw);
  ggml_set_name(vPosEmbd, "hunyuanvl_pos_embd");
  ggml_set_input(vPosEmbd);

  ggml_tensor *vInp = mBuildInp();
  ggml_tensor *vCur = mBuildVit(vInp, cNPatches, cNormTypeNormal, vHparams.vFfnOp, vPosEmbd, nullptr);

  // perceiver projector
  vCur = mBuildNorm(vCur, vModel.vMmPreNormW, nullptr, cNormTypeRms, cEps, -1);

  // [C, W*H] -> [W, H, C] for conv2d
  vCur = ggml_reshape_3d(vCtx0, vCur, cNEmbd, cPw, cPh);
  vCur = ggml_permute(vCtx0, vCur, 2, 0, 1, 3);
  vCur = ggml_cont(vCtx0, vCur);

  // Conv2d(1152->2304, k=2, s=2) + GELU + Conv2d(2304->4608, k=1, s=1)
  vCur = ggml_conv_2d(vCtx0, vModel.vMm0W, vCur, cMerge, cMerge, 0, 0, 1, 1);
  if (vModel.vMm0B) {
    vCur = ggml_add(vCtx0, vCur, ggml_reshape_3d(vCtx0, vModel.vMm0B, 1, 1, vModel.vMm0B->ne[0]));
  }
  vCur = ggml_gelu(vCtx0, vCur);
  vCur = ggml_conv_2d(vCtx0, vModel.vMm1W, vCur, 1, 1, 0, 0, 1, 1);
  if (vModel.vMm1B) {
    vCur = ggml_add(vCtx0, vCur, ggml_reshape_3d(vCtx0, vModel.vMm1B, 1, 1, vModel.vMm1B->ne[0]));
  }

  const int cOw = cPw / cMerge;
  const int cOh = cPh / cMerge;
  const int cIdim = (int)vCur->ne[2]; // OC = 4608

  // append newline along W (dim 0)
  ggml_tensor *vNl = ggml_reshape_4d(vCtx0, vModel.vImageNewline, 1, 1, cIdim, 1);
  vNl = ggml_repeat_4d(vCtx0, vNl, 1, cOh, cIdim, 1);
  vCur = ggml_concat(vCtx0, vCur, vNl, 0);

  // [OW+1, OH, OC] -> [OC, (OW+1)*OH]
  vCur = ggml_permute(vCtx0, vCur, 1, 2, 0, 3);
  vCur = ggml_cont_2d(vCtx0, vCur, cIdim, (cOw + 1) * cOh);

  // project to LLM hidden size
  vCur = mBuildMm(vModel.vMmModelProj, vCur);
  if (vModel.vMmModelProjB) {
    vCur = ggml_add(vCtx0, vCur, vModel.vMmModelProjB);
  }

  // wrap with begin/end tokens
  vCur = ggml_concat(vCtx0, ggml_reshape_2d(vCtx0, vModel.vMmImgBegin, vModel.vMmImgBegin->ne[0], 1), vCur, 1);
  vCur = ggml_concat(vCtx0, vCur, ggml_reshape_2d(vCtx0, vModel.vMmImgEnd, vModel.vMmImgEnd->ne[0], 1), 1);

  vCur = mBuildNorm(vCur, vModel.vMmPostNormW, nullptr, cNormTypeRms, cEps, -1);

  ggml_build_forward_expand(vGf, vCur);
  return vGf;
}
