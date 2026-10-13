#include "models.h"

ggml_cgraph *clip_graph_nemotron_v2_vl::mBuild() {
  GGML_ASSERT(vModel.vClassEmbedding != nullptr);
  GGML_ASSERT(vModel.vPositionEmbeddings != nullptr);

  const int cNRegisters = vModel.vClassEmbedding->ne[1];
  const int cNPos = cNPatches + cNRegisters;

  ggml_tensor *vInp = mBuildInp();

  // add position embeddings (pre-downsampled during GGUF conversion for fixed 512x512 input)
  vInp = ggml_add(vCtx0, vInp, vModel.vPositionEmbeddings);
  mCb(vInp, "inp_pos", -1);

  vInp = ggml_concat(vCtx0, vModel.vClassEmbedding, vInp, 1);

  ggml_tensor *vCur = mBuildVit(vInp, cNPos, cNormTypeNormal, vHparams.vFfnOp, nullptr, nullptr);

  vCur = ggml_view_2d(vCtx0, vCur, cNEmbd, cNPatches, ggml_row_size(vCur->type, cNEmbd),
                      cNRegisters * ggml_row_size(vCur->type, cNEmbd));

  vCur = mBuildPatchMergePermute(vCur, vModel.vHparams.vNMerge);

  {
    vCur = mBuildNorm(vCur, vModel.vMm0W, nullptr, cNormTypeRms, 1e-6, -1);
    vCur = mBuildFfn(vCur, vModel.vMm1W, nullptr, nullptr, nullptr, vModel.vMm3W, nullptr, cFfnReluSqr, -1);
  }

  ggml_build_forward_expand(vGf, vCur);

  return vGf;
}
