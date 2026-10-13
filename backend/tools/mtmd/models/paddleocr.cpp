#include "models.h"

ggml_cgraph *clip_graph_paddleocr::mBuild() {
  const int cNPos = cNPatches;
  const int cNumPositionIds = cNPos * 4; // m-rope requires 4 dim per position

  int aMropeSections[4] = {cDHead / 4, cDHead / 4, cDHead / 4, cDHead / 4};

  ggml_tensor *vPositions = ggml_new_tensor_1d(vCtx0, GGML_TYPE_I32, cNumPositionIds);
  ggml_set_name(vPositions, "positions");
  ggml_set_input(vPositions);

  auto vAddPos = [&](ggml_tensor *pCur, const clip_layer &) {
    return ggml_rope_multi(vCtx0, pCur, vPositions, nullptr, cDHead / 2, aMropeSections, GGML_ROPE_TYPE_VISION, 32768,
                           10000, 1, 0, 1, 32, 1);
  };

  ggml_tensor *vLearnedPosEmbd = mResizePositionEmbeddings();
  ggml_tensor *vInp = mBuildInp();
  ggml_tensor *vCur = mBuildVit(vInp, cNPatches, cNormTypeNormal, vHparams.vFfnOp, vLearnedPosEmbd, vAddPos);

  mCb(vCur, "vit_out", -1);

  {
    // mlp_AR paddleocr projector
    float vProjNormEps = 1e-5;
    vCur = mBuildNorm(vCur, vModel.vMmInputNormW, vModel.vMmInputNormB, cNormTypeNormal, vProjNormEps, -1);

    const int cScaleFactor = vModel.vHparams.vNMerge;
    vCur = mBuildPatchMergePermute(vCur, cScaleFactor);
    vCur =
        mBuildFfn(vCur, vModel.vMm1W, vModel.vMm1B, nullptr, nullptr, vModel.vMm2W, vModel.vMm2B, vHparams.vFfnOp, -1);
    mCb(vCur, "mlp_out", -1);
  }

  // build the graph
  ggml_build_forward_expand(vGf, vCur);

  return vGf;
}
