#include "models.h"

ggml_cgraph *clip_graph_dotsocr::mBuild() {
  const int cNPos = cNPatches;
  const int cNumPositionIds = cNPos * 4; // m-rope requires 4 dim per position

  // note: similar to PaddleOCR
  int aMropeSections[4] = {cDHead / 4, cDHead / 4, cDHead / 4, cDHead / 4};

  ggml_tensor *vPositions = ggml_new_tensor_1d(vCtx0, GGML_TYPE_I32, cNumPositionIds);
  ggml_set_name(vPositions, "positions");
  ggml_set_input(vPositions);

  auto vAddPos = [&](ggml_tensor *pCur, const clip_layer &) {
    return ggml_rope_multi(vCtx0, pCur, vPositions, nullptr, cDHead / 2, aMropeSections, GGML_ROPE_TYPE_VISION, 32768,
                           10000, 1, 0, 1, 32, 1);
  };

  ggml_tensor *vInp = mBuildInp();
  ggml_tensor *vCur = mBuildVit(vInp, cNPatches, cNormTypeRms, vHparams.vFfnOp, nullptr, vAddPos);

  mCb(vCur, "vit_out", -1);

  // dots.ocr patch merger + projector
  {
    GGML_ASSERT(vHparams.vNMerge > 0);
    vCur = mBuildNorm(vCur, vModel.vMmInputNormW, vModel.vMmInputNormB, cNormTypeNormal, 1e-6, -1);
    vCur = mBuildPatchMergePermute(vCur, vHparams.vNMerge);
    mCb(vCur, "after_patch_merger", -1);
    vCur = mBuildFfn(vCur, vModel.vMm0W, vModel.vMm0B, nullptr, nullptr, // no gate
                     vModel.vMm2W, vModel.vMm2B, cFfnGeluErf, -1);       // nn.GELU() defaults to exact erf-based GELU
    mCb(vCur, "after_projector", -1);
  }

  // build the graph
  ggml_build_forward_expand(vGf, vCur);

  return vGf;
}
