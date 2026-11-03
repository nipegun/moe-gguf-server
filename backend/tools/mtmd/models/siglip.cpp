#include "models.h"

ggml_cgraph *clip_graph_siglip::mBuild() {
  ggml_tensor *vInp = mBuildInp();

  ggml_tensor *vLearnedPosEmbd = vModel.vPositionEmbeddings;
  if (vProjType == cProjectorTypeLfm2 || vProjType == cProjectorTypePhi4) {
    vLearnedPosEmbd = mResizePositionEmbeddings();
  }

  ggml_tensor *vCur = mBuildVit(vInp, cNPatches, cNormTypeNormal, vHparams.vFfnOp, vLearnedPosEmbd, nullptr);

  if (vProjType == cProjectorTypeIdefics3) {
    // pixel_shuffle
    // https://github.com/huggingface/transformers/blob/0a950e0bbe1ed58d5401a6b547af19f15f0c195e/src/transformers/models/idefics3/modeling_idefics3.py#L578
    const int cScaleFactor = vModel.vHparams.vNMerge;
    vCur = mBuildPatchMergePermute(vCur, cScaleFactor);
    vCur = mBuildMm(vModel.vMmFcW, vCur);

  } else if (vProjType == cProjectorTypeLfm2) {
    // pixel unshuffle block
    const int cScaleFactor = vModel.vHparams.vNMerge;
    vCur = mBuildPatchMergePermute(vCur, cScaleFactor);

    // projection, in LFM2-VL input norm is optional
    if (vModel.vMmInputNormW) {
      vCur = ggml_norm(vCtx0, vCur, 1e-5); // default nn.LayerNorm
      vCur = ggml_mul(vCtx0, vCur, vModel.vMmInputNormW);
    }

    if (vModel.vMmInputNormB) {
      vCur = ggml_add(vCtx0, vCur, vModel.vMmInputNormB);
    }

    vCur = mBuildFfn(vCur, vModel.vMm1W, vModel.vMm1B, nullptr, nullptr, vModel.vMm2W, vModel.vMm2B, cFfnGelu, -1);

  } else if (vProjType == cProjectorTypeJanusPro) {
    vCur =
        mBuildFfn(vCur, vModel.vMm0W, vModel.vMm0B, nullptr, nullptr, vModel.vMm1W, vModel.vMm1B, vHparams.vFfnOp, -1);

  } else if (vProjType == cProjectorTypePhi4) {
    vCur = mBuildFfn(vCur, vModel.vMm0W, vModel.vMm0B, nullptr, nullptr, vModel.vMm2W, vModel.vMm2B, cFfnGelu, -1);

  } else {
    GGML_ABORT("SigLIP: Unsupported projector type");
  }

  // build the graph
  ggml_build_forward_expand(vGf, vCur);

  return vGf;
}
