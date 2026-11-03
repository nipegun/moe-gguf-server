#include "models.h"

ggml_tensor *clip_graph_mimovl::mBuildMm(ggml_tensor *pW, ggml_tensor *pX) const {
  ggml_tensor *vCur = ggml_mul_mat(vCtx0, pW, pX);
  ggml_mul_mat_set_prec(vCur, GGML_PREC_F32);
  return vCur;
}

// MiMoVL vision tower for MiMo-V2.5 (non-Pro). Qwen2.5-VL-shaped ViT, except:
//   1. GQA in attention (32 Q / 8 KV heads, head_dim 64).
//   2. Per-head attention sinks on every windowed layer. The sinks adjust
//      the softmax denominator (equivalently, a virtual extra K column with V=0),
//      so they decay attention weight without contributing to the output.
//   3. Per-layer window-attention mode in hparams.lWaPatternMode:
//        -1 -> full,  0 -> row-window+sinks,  1 -> col-window+sinks.
//      Col mode transposes the merge-unit grid on entry and restores
//      it on exit. Both patch and rotary orderings are pre-computed
//      host-side.
//   4. 1D banded sliding window (|q-k| > window_size -> -inf) as a
//      single 2D mask broadcast across heads.
//   5. Per-block MLP biases.
ggml_cgraph *clip_graph_mimovl::mBuild() {
  GGML_ASSERT(vModel.vPatchEmbeddings0 != nullptr);
  GGML_ASSERT(vModel.vPatchEmbeddings1 != nullptr);
  GGML_ASSERT(vModel.vClassEmbedding == nullptr);
  GGML_ASSERT(vHparams.vNHeadKv > 0);
  GGML_ASSERT(cNHead % vHparams.vNHeadKv == 0);
  GGML_ASSERT((int)vHparams.lWaPatternMode.size() == cNLayer);

  const int cBatchSize = 1;
  const int cNPos = cNPatches;
  const int cNHeadKv = vHparams.vNHeadKv;
  const int cMerge = vHparams.vNMerge > 0 ? vHparams.vNMerge : 2;
  const int cMergeUnit = cMerge * cMerge;
  const int cNUnits = cNPos / cMergeUnit;
  GGML_ASSERT(cNUnits * cMergeUnit == cNPos);

  // MiMoVL has head_dim=64 with n_embd=1280, so n_embd is NOT n_head*head_dim
  // (the base class's d_head = n_embd/n_head = 40 is wrong here). Derive
  // head_dim from the fused QKV projection: rows = (n_head + 2*n_head_kv)*head_dim.
  GGML_ASSERT(vModel.lLayers[0].vQkvW != nullptr);
  const int cQkvRows = vModel.lLayers[0].vQkvW->ne[1];
  const int cHeadDim = cQkvRows / (cNHead + 2 * cNHeadKv);
  GGML_ASSERT(cHeadDim * (cNHead + 2 * cNHeadKv) == cQkvRows);
  const float cAttnScale = 1.0f / std::sqrt((float)cHeadDim);
  const int cRopeNDims = cHeadDim / 2;
  int aMropeSections[4] = {cRopeNDims / 2, cRopeNDims / 2, 0, 0};

  // Patch embed: Conv3D(kt=2) split into two Conv2D, then interleave-merge
  // along the height axis to match the merge-tile token order.
  ggml_tensor *vInpRaw = mBuildInpRaw();
  ggml_tensor *vInp = ggml_conv_2d(vCtx0, vModel.vPatchEmbeddings0, vInpRaw, cPatchSize, cPatchSize, 0, 0, 1, 1);
  {
    ggml_tensor *vInp1 = ggml_conv_2d(vCtx0, vModel.vPatchEmbeddings1, vInpRaw, cPatchSize, cPatchSize, 0, 0, 1, 1);
    vInp = ggml_add(vCtx0, vInp, vInp1);

    GGML_ASSERT(mGetImg(0).mNx() % (cPatchSize * 2) == 0);
    GGML_ASSERT(mGetImg(0).mNy() % (cPatchSize * 2) == 0);

    vInp = ggml_permute(vCtx0, vInp, 1, 2, 0, 3); // [w,h,c,b] -> [c,w,h,b]
    vInp = ggml_cont_4d(vCtx0, vInp, cNEmbd * 2, cNPatchesX / 2, cNPatchesY, cBatchSize);
    vInp = ggml_reshape_4d(vCtx0, vInp, cNEmbd * 2, cNPatchesX / 2, 2, cBatchSize * (cNPatchesY / 2));
    vInp = ggml_permute(vCtx0, vInp, 0, 2, 1, 3);
    vInp = ggml_cont_3d(vCtx0, vInp, cNEmbd, cNPatchesX * cNPatchesY, cBatchSize);
  }
  mCb(vInp, "patch_embed", -1);

  ggml_tensor *vPositionsRow = ggml_new_tensor_1d(vCtx0, GGML_TYPE_I32, cNPos * 4);
  ggml_set_name(vPositionsRow, "mimovl_positions_row");
  ggml_set_input(vPositionsRow);

  ggml_tensor *vPositionsCol = ggml_new_tensor_1d(vCtx0, GGML_TYPE_I32, cNPos * 4);
  ggml_set_name(vPositionsCol, "mimovl_positions_col");
  ggml_set_input(vPositionsCol);

  // idx_col is the col-major merge-unit permutation. Take it as F32 so we can
  // derive the inverse permutation in-graph via ggml_argsort;
  // ggml_get_rows requires its index tensor to be I32, so cast back as well.
  ggml_tensor *vIdxColF = ggml_new_tensor_1d(vCtx0, GGML_TYPE_F32, cNUnits);
  ggml_set_name(vIdxColF, "mimovl_idx_col");
  ggml_set_input(vIdxColF);
  ggml_tensor *vIdxCol = ggml_cast(vCtx0, vIdxColF, GGML_TYPE_I32);
  ggml_tensor *vIdxColInv = ggml_argsort(vCtx0, vIdxColF, GGML_SORT_ORDER_ASC);

  ggml_tensor *vWindowMask = ggml_new_tensor_2d(vCtx0, GGML_TYPE_F32, cNPos, cNPos);
  ggml_set_name(vWindowMask, "mimovl_window_mask");
  ggml_set_input(vWindowMask);

  ggml_tensor *vWindowMaskAttn =
      (cFlashAttnType == cClipFlashAttnTypeEnabled) ? ggml_cast(vCtx0, vWindowMask, GGML_TYPE_F16) : vWindowMask;

  // Reorder helper: permute patches at merge-unit granularity. The patch
  // sequence is laid out as n_units groups of merge_unit (=4) consecutive
  // patches; the row<->col transpose only permutes whole groups. We keep
  // the per-group (h,w) ordering intact by reshaping to
  // [n_embd*merge_unit, n_units] before ggml_get_rows.
  auto vReorder = [&](ggml_tensor *pX, ggml_tensor *pIdx) {
    ggml_tensor *vY = ggml_reshape_2d(vCtx0, pX, cNEmbd * cMergeUnit, cNUnits);
    vY = ggml_get_rows(vCtx0, vY, pIdx);
    return ggml_reshape_3d(vCtx0, vY, cNEmbd, cNPos, cBatchSize);
  };

  ggml_tensor *vInpL = vInp;
  int vPrevMode = -1;

  for (int vIl = 0; vIl < cNLayer; vIl++) {
    const auto &vLayer = vModel.lLayers[vIl];
    const int cMode = vHparams.lWaPatternMode[vIl];
    const bool cIsFull = (cMode == -1);
    const bool cIsCol = (cMode == 1);

    // Reorder transitions on entry/exit of a col-mode run.
    if (cIsCol && vPrevMode != 1) {
      vInpL = vReorder(vInpL, vIdxCol);
      mCb(vInpL, "reorder_to_col", vIl);
    } else if (!cIsCol && vPrevMode == 1) {
      vInpL = vReorder(vInpL, vIdxColInv);
      mCb(vInpL, "reorder_to_row", vIl);
    }

    ggml_tensor *vCur = vInpL;

    // Pre-attention RMSNorm.
    vCur = mBuildNorm(vCur, vLayer.vLn1W, vLayer.vLn1B, cNormTypeRms, cEps, vIl);
    mCb(vCur, "ln1", vIl);

    // Fused QKV with GQA.
    ggml_tensor *vQkv = mBuildMm(vLayer.vQkvW, vCur);
    vQkv = ggml_add(vCtx0, vQkv, vLayer.vQkvB);

    const size_t cRow = ggml_row_size(vQkv->type, cHeadDim);
    const size_t cOffK = ggml_row_size(vQkv->type, cNHead * cHeadDim);
    const size_t cOffV = ggml_row_size(vQkv->type, (cNHead + cNHeadKv) * cHeadDim);

    ggml_tensor *vQcur = ggml_view_3d(vCtx0, vQkv, cHeadDim, cNHead, cNPos, cRow, vQkv->nb[1], 0);
    ggml_tensor *vKcur = ggml_view_3d(vCtx0, vQkv, cHeadDim, cNHeadKv, cNPos, cRow, vQkv->nb[1], cOffK);
    ggml_tensor *vVcur = ggml_view_3d(vCtx0, vQkv, cHeadDim, cNHeadKv, cNPos, cRow, vQkv->nb[1], cOffV);

    mCb(vQcur, "Qcur", vIl);
    mCb(vKcur, "Kcur", vIl);
    mCb(vVcur, "Vcur", vIl);

    // 2D RoPE
    ggml_tensor *vPos = cIsCol ? vPositionsCol : vPositionsRow;
    vQcur = ggml_rope_multi(vCtx0, vQcur, vPos, nullptr, cRopeNDims, aMropeSections, GGML_ROPE_TYPE_VISION, 32768,
                            10000.0f, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);
    vKcur = ggml_rope_multi(vCtx0, vKcur, vPos, nullptr, cRopeNDims, aMropeSections, GGML_ROPE_TYPE_VISION, 32768,
                            10000.0f, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);
    mCb(vQcur, "Qcur_rope", vIl);
    mCb(vKcur, "Kcur_rope", vIl);

    // Full layers: plain attention. Windowed layers: banded mask and per-head sinks.
    ggml_tensor *vMask = cIsFull ? nullptr : vWindowMaskAttn;
    ggml_tensor *vSinks = cIsFull ? nullptr : vLayer.vAttnSinks;
    if (!cIsFull) {
      GGML_ASSERT(vLayer.vAttnSinks != nullptr);
    }
    ggml_tensor *vAttnOut = mBuildAttn(vLayer.vOW, vLayer.vOB, vQcur, vKcur, vVcur, vMask, cAttnScale, vIl, vSinks);
    mCb(vAttnOut, "attn_out", vIl);

    // Residual 1.
    vCur = ggml_add(vCtx0, vAttnOut, vInpL);
    vInpL = vCur;
    mCb(vCur, "ffn_inp", vIl);

    // Pre-FFN RMSNorm.
    vCur = mBuildNorm(vCur, vLayer.vLn2W, vLayer.vLn2B, cNormTypeRms, cEps, vIl);
    mCb(vCur, "ffn_inp_normed", vIl);

    // SwiGLU MLP with biases
    vCur = mBuildFfn(vCur, vLayer.vFfUpW, vLayer.vFfUpB, vLayer.vFfGateW, vLayer.vFfGateB, vLayer.vFfDownW,
                     vLayer.vFfDownB, vHparams.vFfnOp, vIl);
    mCb(vCur, "ffn_out", vIl);

    // Residual 2.
    vCur = ggml_add(vCtx0, vInpL, vCur);
    mCb(vCur, "layer_out", vIl);

    vInpL = vCur;
    vPrevMode = cMode;
  }

  // If the last block was col-mode, undo the transpose so the merger sees patches in row order.
  if (vPrevMode == 1) {
    vInpL = vReorder(vInpL, vIdxColInv);
    mCb(vInpL, "reorder_to_row_final", -1);
  }

  // Merger: post-LayerNorm
  vInpL = mBuildNorm(vInpL, vModel.vPostLnW, vModel.vPostLnB, cNormTypeNormal, 1e-6f, cNLayer);
  mCb(vInpL, "post_ln", -1);

  // Spatial merge: pack each merge_unit (=4) of patches into a single
  // (n_embd*merge_unit)-wide row, then run the 2-layer MLP.
  ggml_tensor *vEmbeddings = ggml_reshape_3d(vCtx0, vInpL, cNEmbd * cMergeUnit, cNUnits, cBatchSize);
  vEmbeddings = mBuildFfn(vEmbeddings, vModel.vMm0W, nullptr, nullptr, nullptr, vModel.vMm1W, nullptr, cFfnGelu, -1);
  mCb(vEmbeddings, "vit_out", -1);

  ggml_build_forward_expand(vGf, vEmbeddings);
  return vGf;
}
