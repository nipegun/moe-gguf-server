#include "models.h"

ggml_cgraph *clip_graph_internvl::mBuild() {
  GGML_ASSERT(vModel.vClassEmbedding != nullptr);
  GGML_ASSERT(vModel.vPositionEmbeddings != nullptr);

  const int cNPos = cNPatches + 1;
  ggml_tensor *vInp = mBuildInp();

  // add CLS token
  ggml_tensor *vClsRepeated =
      ggml_repeat_4d(vCtx0, vModel.vClassEmbedding, vModel.vClassEmbedding->ne[0], 1, vNBatch, 1);
  vInp = ggml_concat(vCtx0, vInp, vClsRepeated, 1);

  // The larger models use a different ViT, which uses RMS norm instead of layer norm
  norm_type vNormT = (vHparams.vNEmbd == 3200 && vHparams.vNLayer == 45)
                         ? cNormTypeRms     // 6B ViT (Used by InternVL 2.5/3 - 26B, 38B, 78B)
                         : cNormTypeNormal; // 300M ViT (Used by all smaller InternVL models)

  ggml_tensor *vCur = mBuildVit(vInp, cNPos, vNormT, vHparams.vFfnOp, vModel.vPositionEmbeddings, nullptr);

  // remove CLS token
  vCur = ggml_view_3d(vCtx0, vCur, cNEmbd, cNPatches, vNBatch, vCur->nb[1], vCur->nb[2], 0);
  vCur = ggml_cont(vCtx0, vCur);

  // pixel shuffle
  {
    const int cScaleFactor = vModel.vHparams.vNMerge;
    const int cBsz = vNBatch;
    const int cHeight = cNPatchesY;
    const int cWidth = cNPatchesX;
    GGML_ASSERT(cScaleFactor > 0);
    vCur = ggml_reshape_4d(vCtx0, vCur, cNEmbd * cScaleFactor, cHeight / cScaleFactor, cWidth, cBsz);
    vCur = ggml_permute(vCtx0, vCur, 0, 2, 1, 3);
    vCur = ggml_cont_4d(vCtx0, vCur, cNEmbd * cScaleFactor * cScaleFactor, cHeight / cScaleFactor,
                        cWidth / cScaleFactor, cBsz);
    vCur = ggml_permute(vCtx0, vCur, 0, 2, 1, 3);
    // flatten to 2D
    vCur = ggml_cont_3d(vCtx0, vCur, cNEmbd * cScaleFactor * cScaleFactor, vCur->ne[1] * vCur->ne[2], vCur->ne[3]);
  }

  // projector (always using GELU activation)
  {
    // projector LayerNorm uses pytorch's default eps = 1e-5
    // ref:
    // https://huggingface.co/OpenGVLab/InternVL3-8B-Instruct/blob/a34d3e4e129a5856abfd6aa6de79776484caa14e/modeling_internvl_chat.py#L79
    vCur = mBuildNorm(vCur, vModel.vMm0W, vModel.vMm0B, cNormTypeNormal, 1e-5, -1);
    vCur = mBuildFfn(vCur, vModel.vMm1W, vModel.vMm1B, nullptr, nullptr, vModel.vMm3W, vModel.vMm3B, cFfnGelu, -1);
  }

  // build the graph
  ggml_build_forward_expand(vGf, vCur);

  return vGf;
}
