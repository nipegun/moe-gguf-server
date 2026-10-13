#include "models.h"

ggml_tensor *clip_graph_minimax_m3::mApplyRope(ggml_tensor *pX, ggml_tensor *pPosH, ggml_tensor *pPosW) {
  const int64_t cHn = pX->ne[1];
  const int64_t cP = pX->ne[2];
  const size_t cEs = ggml_element_size(pX);
  const int cDh = (int)pX->ne[0];
  const int cAxd = 2 * ((2 * (cDh / 2) / 3) / 2);

  GGML_ASSERT(pX->nb[0] == cEs);
  GGML_ASSERT(3 * cAxd <= cDh);

  const float cTh = vHparams.vRopeTheta;

  // layout of x is [t, h, w, pad]
  // t is unrotated, h and w are rotated, pad is unrotated
  // note: everything from n_dims onward untouched, so w and pad are rotated in one call.
  auto vSl = [&](int pOff, int pN) {
    return ggml_cont(vCtx0, ggml_view_3d(vCtx0, pX, pN, cHn, cP, pX->nb[1], pX->nb[2], (size_t)pOff * cEs));
  };
  ggml_tensor *vT = vSl(0, cAxd);
  ggml_tensor *vH = vSl(cAxd, cAxd);
  ggml_tensor *vW = vSl(2 * cAxd, cDh - 2 * cAxd); // w + pad

  vH = ggml_rope_ext(vCtx0, vH, pPosH, nullptr, cAxd, GGML_ROPE_TYPE_NEOX, 0, cTh, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
  vW = ggml_rope_ext(vCtx0, vW, pPosW, nullptr, cAxd, GGML_ROPE_TYPE_NEOX, 0, cTh, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
  return ggml_concat(vCtx0, ggml_concat(vCtx0, vT, vH, 0), vW, 0);
}

ggml_cgraph *clip_graph_minimax_m3::mBuild() {
  GGML_ASSERT(vModel.vPatchBias == nullptr);
  GGML_ASSERT(vModel.vClassEmbedding == nullptr);
  GGML_ASSERT(vModel.vPatchEmbeddings0 && vModel.vPatchEmbeddings1);
  GGML_ASSERT(vModel.vMm1W && vModel.vMm2W);
  GGML_ASSERT(vModel.vMmMergerFc1W && vModel.vMmMergerFc2W);

  const int cBatchSize = 1;
  const int cNPos = cNPatches;
  const int cMerge = vHparams.vNMerge;

  // patch embedding
  ggml_tensor *vInpRaw = mBuildInpRaw();
  ggml_tensor *vInp =
      ggml_add(vCtx0, ggml_conv_2d(vCtx0, vModel.vPatchEmbeddings0, vInpRaw, cPatchSize, cPatchSize, 0, 0, 1, 1),
               ggml_conv_2d(vCtx0, vModel.vPatchEmbeddings1, vInpRaw, cPatchSize, cPatchSize, 0, 0, 1, 1));

  // spatial merge
  {
    vInp = ggml_permute(vCtx0, vInp, 1, 2, 0, 3);
    vInp = ggml_cont_4d(vCtx0, vInp, cNEmbd * cMerge, cNPatchesX / cMerge, cNPatchesY, cBatchSize);
    vInp =
        ggml_reshape_4d(vCtx0, vInp, cNEmbd * cMerge, cNPatchesX / cMerge, cMerge, cBatchSize * (cNPatchesY / cMerge));
    vInp = ggml_permute(vCtx0, vInp, 0, 2, 1, 3);
    vInp = ggml_cont_3d(vCtx0, vInp, cNEmbd, cNPatchesX * cNPatchesY, cBatchSize);
  }

  // t (time axis) is always 0 for now, so we leave it unrotated
  ggml_tensor *vPosH = ggml_new_tensor_1d(vCtx0, GGML_TYPE_I32, cNPos);
  ggml_set_name(vPosH, "minimax_pos_h");
  ggml_set_input(vPosH);
  ggml_tensor *vPosW = ggml_new_tensor_1d(vCtx0, GGML_TYPE_I32, cNPos);
  ggml_set_name(vPosW, "minimax_pos_w");
  ggml_set_input(vPosW);

  ggml_tensor *vInpL = mBuildVit(vInp, cNPos, cNormTypeNormal, cFfnGeluErf, nullptr,
                                 [&](ggml_tensor *pC, const clip_layer &) { return mApplyRope(pC, vPosH, vPosW); });

  // projector
  ggml_tensor *vEmb = vInpL;
  vEmb = mBuildFfn(vEmb, vModel.vMm1W, vModel.vMm1B, nullptr, nullptr, vModel.vMm2W, vModel.vMm2B, cFfnGeluErf, -1);

  const int64_t cProj = vEmb->ne[0];
  vEmb = ggml_reshape_2d(vCtx0, vEmb, cProj * cMerge * cMerge, cNPos / (cMerge * cMerge));

  vEmb = mBuildFfn(vEmb, vModel.vMmMergerFc1W, vModel.vMmMergerFc1B, nullptr, nullptr, vModel.vMmMergerFc2W,
                   vModel.vMmMergerFc2B, cFfnGeluErf, -1);

  ggml_build_forward_expand(vGf, vEmb);
  return vGf;
}
