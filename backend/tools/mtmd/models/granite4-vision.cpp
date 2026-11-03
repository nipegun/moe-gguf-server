#include "models.h"
#include "../clip-impl.h"
#include "../clip-model.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>

/*
 * Granite Vision 4.1 clip graph
 *
 *   Stage 1a: SigLIP vision tower (N layers, post-norm)
 *   Stage 1b: WindowQFormer blocks (deepstack + spatial)
 *   Stage 1c: Concatenate and pack outputs
 *   Stage 1d: Append newline tokens if add_newline is set
 */

// ---------------------------------------------------------------------------
// Member method implementations
// ---------------------------------------------------------------------------

ggml_tensor *clip_graph_granite4_vision::mGather(ggml_tensor *pSrc, const std::string &pName, int pIdxLen) {
  ggml_tensor *vIdx = ggml_new_tensor_1d(vCtx0, GGML_TYPE_I32, pIdxLen);
  ggml_set_name(vIdx, pName.c_str());
  ggml_set_input(vIdx);
  return ggml_get_rows(vCtx0, pSrc, vIdx);
}

ggml_tensor *clip_graph_granite4_vision::mInterpDown(ggml_tensor *pSrc, int pSide, int pNewSide) {
  const int cNEmbd = pSrc->ne[0];
  ggml_tensor *vT = ggml_reshape_4d(vCtx0, pSrc, cNEmbd, pSide, pSide, 1);
  vT = ggml_cont(vCtx0, ggml_permute(vCtx0, vT, 2, 0, 1, 3));
  const int cKernel = pSide / pNewSide;
  vT = ggml_pool_2d(vCtx0, vT, GGML_OP_POOL_AVG, cKernel, cKernel, cKernel, cKernel, 0, 0);
  vT = ggml_cont(vCtx0, ggml_permute(vCtx0, vT, 1, 2, 0, 3));
  return ggml_reshape_2d(vCtx0, vT, cNEmbd, pNewSide * pNewSide);
}

// ---------------------------------------------------------------------------
// build_block - WindowQFormer block implementation
// ---------------------------------------------------------------------------

ggml_tensor *clip_graph_granite4_vision::mBuildBlock(const qf_block &pBlk, ggml_tensor *pH, int pBid,
                                                     int pSpatialOffset, int pImageSide, int pWindowSide,
                                                     int pQuerySide, float pQformerEps) {

  const int cNEmbd = pH->ne[0];
  GGML_ASSERT(pH->ne[1] == pImageSide * pImageSide);
  const int cN = pImageSide / pWindowSide;
  const int cNewSide = cN * pQuerySide;
  const int cNWindows = cN * cN;
  const int cEncLen = pWindowSide * pWindowSide;
  const int cQueryLen = pQuerySide * pQuerySide;

  auto vCbx = [&](ggml_tensor *&pT, const char *pStep) {
    const std::string cName = "g4v_blk" + std::to_string(pBid) + "_" + pStep;
    ggml_set_name(pT, cName.c_str());
  };

  // 1. Top-level LN
  vCbx(pH, "inp");
  ggml_tensor *vX = mBuildNorm(pH, pBlk.vQfProjNormW, pBlk.vQfProjNormB, cNormTypeNormal, cEps, pBid);
  vCbx(vX, "norm");

  // 2. enc = _win(x, image_side, window_side)
  ggml_tensor *vEnc;
  {
    ggml_tensor *vEncFlat = mGather(vX, "g4v_blk" + std::to_string(pBid) + "_win_idx", pImageSide * pImageSide);
    vEnc = ggml_reshape_3d(vCtx0, vEncFlat, cNEmbd, cEncLen, cNWindows);
  }
  vCbx(vEnc, "enc");

  // 3. downsampled = downsampler(x)
  ggml_tensor *vD;
  (void)pSpatialOffset;
  if (pSpatialOffset >= 0) {
    vD = mGather(vX, "g4v_blk" + std::to_string(pBid) + "_spatial_idx", cNewSide * cNewSide);
  } else {
    vD = mInterpDown(vX, pImageSide, cNewSide);
  }
  vCbx(vD, "downsampled");

  // 4. query_embeds = query + _win(d, new_side, query_side)
  ggml_tensor *vQIn;
  {
    ggml_tensor *vDwFlat = mGather(vD, "g4v_blk" + std::to_string(pBid) + "_qwin_idx", cNewSide * cNewSide);
    ggml_tensor *vDw = ggml_reshape_3d(vCtx0, vDwFlat, cNEmbd, cQueryLen, cNWindows);
    vQIn = ggml_add(vCtx0, vDw, pBlk.vQfProjQuery);
  }
  vCbx(vQIn, "query_embeds");

  // 5. encoder_embeds = enc + image_positions → (C, enc_len, n_windows)
  ggml_tensor *vEIn = ggml_add(vCtx0, vEnc, pBlk.vQfProjImgPos);
  vCbx(vEIn, "encoder_embeds");

  // 6. Qformer forward.
  ggml_tensor *vQueryInput = mBuildNorm(vQIn, pBlk.vQfProjPostNormW, pBlk.vQfProjPostNormB, cNormTypeNormal, pQformerEps, pBid);

  // Helper for linear projections with window batching
  auto vLinear = [&](ggml_tensor *pX, ggml_tensor *pW, ggml_tensor *pB) -> ggml_tensor * {
    ggml_tensor *vT = ggml_reshape_2d(vCtx0, pX, pX->ne[0], pX->ne[1] * pX->ne[2]);
    vT = mBuildMm(pW, vT);
    if (pB)
      vT = ggml_add(vCtx0, vT, pB);
    return vT;
  };

  // Get the single QFormer layer
  GGML_ASSERT(pBlk.lQfProjLayers.size() == 1);
  const auto &vPl = pBlk.lQfProjLayers[0];

  // 6a. Self-attention
  ggml_tensor *vSaOut;
  {
    const int cDH = 64;
    const int cNHead = cNEmbd / cDH;
    const int cNq = vQueryInput->ne[1];
    const float cScale = 1.0f / std::sqrt((float)cDH);

    ggml_tensor *vQ = vLinear(vQueryInput, vPl.vQW, vPl.vQB);
    ggml_tensor *vK = vLinear(vQueryInput, vPl.vKW, vPl.vKB);
    ggml_tensor *vV = vLinear(vQueryInput, vPl.vVW, vPl.vVB);

    vQ = ggml_reshape_4d(vCtx0, vQ, cDH, cNHead, cNq, cNWindows);
    vK = ggml_reshape_4d(vCtx0, vK, cDH, cNHead, cNq, cNWindows);
    vV = ggml_reshape_4d(vCtx0, vV, cDH, cNHead, cNq, cNWindows);

    vSaOut = mBuildAttn(vPl.vOW, vPl.vOB, vQ, vK, vV, nullptr, cScale, pBid);
    vSaOut = ggml_reshape_3d(vCtx0, vSaOut, cNEmbd, cNq, cNWindows);

    vSaOut = ggml_add(vCtx0, vSaOut, vQueryInput);
    vSaOut = mBuildNorm(vSaOut, vPl.vLn1W, vPl.vLn1B, cNormTypeNormal, pQformerEps, pBid);
  }
  vCbx(vSaOut, "sa_out");

  // 6b. Cross-attention
  ggml_tensor *vCaOut;
  {
    const int cDH = 64;
    const int cNHead = cNEmbd / cDH;
    const int cNq = vSaOut->ne[1];
    const int cNkv = vEIn->ne[1];
    const float cScale = 1.0f / std::sqrt((float)cDH);

    ggml_tensor *vQ = vLinear(vSaOut, vPl.vCrossAttnQW, vPl.vCrossAttnQB);
    ggml_tensor *vK = vLinear(vEIn, vPl.vCrossAttnKW, vPl.vCrossAttnKB);
    ggml_tensor *vV = vLinear(vEIn, vPl.vCrossAttnVW, vPl.vCrossAttnVB);

    vQ = ggml_reshape_4d(vCtx0, vQ, cDH, cNHead, cNq, cNWindows);
    vK = ggml_reshape_4d(vCtx0, vK, cDH, cNHead, cNkv, cNWindows);
    vV = ggml_reshape_4d(vCtx0, vV, cDH, cNHead, cNkv, cNWindows);

    vCaOut = mBuildAttn(vPl.vCrossAttnOW, vPl.vCrossAttnOB, vQ, vK, vV, nullptr, cScale, pBid);
    vCaOut = ggml_reshape_3d(vCtx0, vCaOut, cNEmbd, cNq, cNWindows);

    vCaOut = ggml_add(vCtx0, vCaOut, vSaOut);
    vCaOut = mBuildNorm(vCaOut, vPl.vCrossAttnNormW, vPl.vCrossAttnNormB, cNormTypeNormal, pQformerEps, pBid);
  }
  vCbx(vCaOut, "ca_out");

  // 6c. FFN
  ggml_tensor *vFfn;
  {
    ggml_tensor *vT = ggml_reshape_2d(vCtx0, vCaOut, cNEmbd, cQueryLen * cNWindows);
    vT = mBuildMm(vPl.vFfUpW, vT);
    if (vPl.vFfUpB)
      vT = ggml_add(vCtx0, vT, vPl.vFfUpB);
    vT = ggml_gelu_erf(vCtx0, vT);
    vT = mBuildMm(vPl.vFfDownW, vT);
    if (vPl.vFfDownB)
      vT = ggml_add(vCtx0, vT, vPl.vFfDownB);
    vT = ggml_reshape_3d(vCtx0, vT, cNEmbd, cQueryLen, cNWindows);
    vFfn = ggml_add(vCtx0, vT, vCaOut);
    vFfn = mBuildNorm(vFfn, vPl.vLn2W, vPl.vLn2B, cNormTypeNormal, pQformerEps, pBid);
  }
  vCbx(vFfn, "qformer_out");

  // 7. _unwin back to raster
  ggml_tensor *vUnwinned;
  {
    ggml_tensor *vFlat = ggml_reshape_2d(vCtx0, vFfn, cNEmbd, cQueryLen * cNWindows);
    vUnwinned = mGather(vFlat, "g4v_blk" + std::to_string(pBid) + "_unwin_idx", cNewSide * cNewSide);
  }
  vCbx(vUnwinned, "unwin");

  // 8. out_linear
  ggml_tensor *vOut = mBuildMm(pBlk.vQfProjLinearW, vUnwinned);
  if (pBlk.vQfProjLinearB)
    vOut = ggml_add(vCtx0, vOut, pBlk.vQfProjLinearB);
  vCbx(vOut, "out");

  return vOut;
}

// ---------------------------------------------------------------------------
// mBuild() - top-level graph
// ---------------------------------------------------------------------------

// Build the K-tiled, base-scaled newline row tensor.
// Shape: (n_mmproj_embd, 1)
ggml_tensor *clip_graph_granite4_vision::mBuildNewlineRow(ggml_context *pCtx0) {
  const int cK = (int)vModel.lQfProjBlocks.size();
  GGML_ASSERT(cK > 0);
  GGML_ASSERT(cNMmprojEmbd % cK == 0);
  const int cProjectionDim = cNMmprojEmbd / cK;
  GGML_ASSERT(vModel.vImageNewline != nullptr);
  GGML_ASSERT(ggml_nelements(vModel.vImageNewline) == cProjectionDim);

  // Build newline_row[k*projection_dim + d] = nl[d] * (k == 0 ? base : 1.0)
  ggml_tensor *vNl = vModel.vImageNewline; // (projection_dim,)
  ggml_tensor *vNlFirst2d = ggml_reshape_2d(pCtx0, vNl, cProjectionDim, 1);
  ggml_tensor *vNlRow2d;
  if (cK == 1) {
    vNlRow2d = vNlFirst2d;
  } else {
    ggml_tensor *vNl2d = ggml_reshape_2d(pCtx0, vNl, cProjectionDim, 1);
    ggml_tensor *vRestTemplate = ggml_new_tensor_2d(pCtx0, GGML_TYPE_F32, cProjectionDim, cK - 1);
    ggml_tensor *vNlRest = ggml_repeat(pCtx0, vNl2d, vRestTemplate);
    vNlRow2d = ggml_concat(pCtx0, vNlFirst2d, vNlRest, 1); // (projection_dim, K)
  }
  vNlRow2d = ggml_cont(pCtx0, vNlRow2d);
  return ggml_reshape_2d(pCtx0, vNlRow2d, cNMmprojEmbd, 1);
}

// Append a single newline row at the end of the tile output.
ggml_tensor *clip_graph_granite4_vision::mAppendRowwiseNewlines(ggml_context *pCtx0, ggml_tensor *pTileOutput) {
  // For the single-tile case, append one newline row at the end.
  // For the multi-tile rowwise case, this will be called per-tile
  // (though currently only the single-tile path uses it).
  ggml_tensor *vNlRow = mBuildNewlineRow(pCtx0);
  return ggml_concat(pCtx0, pTileOutput, vNlRow, 1);
}

ggml_cgraph *clip_graph_granite4_vision::mBuild() {
  GGML_ASSERT(vModel.vPatchEmbeddings0 != nullptr);
  GGML_ASSERT(vModel.vPositionEmbeddings != nullptr);
  GGML_ASSERT(vModel.vClassEmbedding == nullptr);
  GGML_ASSERT(!vModel.lQfProjBlocks.empty());

  // --- Stage 1a: SigLIP encoder producing intermediate hidden states ---
  ggml_tensor *vInp = mBuildInp();
  vInp = ggml_add(vCtx0, vInp, vModel.vPositionEmbeddings);
  mCb(vInp, "pos_embed", -1);

  ggml_tensor *vInpL = vInp;
  std::vector<ggml_tensor *> lLayerOuts(cNLayer, nullptr);

  for (int vIl = 0; vIl < cNLayer; ++vIl) {
    const auto &vLayer = vModel.lLayers[vIl];
    ggml_tensor *vCur = vInpL;

    vCur = mBuildNorm(vCur, vLayer.vLn1W, vLayer.vLn1B, cNormTypeNormal, cEps, vIl);

    // Self-attention
    ggml_tensor *vQcur = mBuildMm(vLayer.vQW, vCur);
    if (vLayer.vQB)
      vQcur = ggml_add(vCtx0, vQcur, vLayer.vQB);
    ggml_tensor *vKcur = mBuildMm(vLayer.vKW, vCur);
    if (vLayer.vKB)
      vKcur = ggml_add(vCtx0, vKcur, vLayer.vKB);
    ggml_tensor *vVcur = mBuildMm(vLayer.vVW, vCur);
    if (vLayer.vVB)
      vVcur = ggml_add(vCtx0, vVcur, vLayer.vVB);

    vQcur = ggml_reshape_3d(vCtx0, vQcur, cDHead, cNHead, cNPatches);
    vKcur = ggml_reshape_3d(vCtx0, vKcur, cDHead, cNHead, cNPatches);
    vVcur = ggml_reshape_3d(vCtx0, vVcur, cDHead, cNHead, cNPatches);

    vCur = mBuildAttn(vLayer.vOW, vLayer.vOB, vQcur, vKcur, vVcur, nullptr, vKqScale, vIl);

    vCur = ggml_add(vCtx0, vCur, vInpL);
    vInpL = vCur;

    vCur = mBuildNorm(vCur, vLayer.vLn2W, vLayer.vLn2B, cNormTypeNormal, cEps, vIl);
    vCur = mBuildFfn(vCur, vLayer.vFfUpW, vLayer.vFfUpB, vLayer.vFfGateW, vLayer.vFfGateB, vLayer.vFfDownW,
                     vLayer.vFfDownB, vHparams.vFfnOp, vIl);
    vCur = ggml_add(vCtx0, vInpL, vCur);
    mCb(vCur, "layer_out", vIl);
    lLayerOuts[vIl] = vCur;
    vInpL = vCur;
  }

  // --- Stage 1b/1c: WindowQFormer blocks ---
  const int cProjectorCount = vHparams.lFeatureLayers.size();
  const float cQformerEps = 1e-12f;

  ggml_tensor *vMmproj = nullptr;
  for (int vBid = 0; vBid < cProjectorCount; ++vBid) {
    const auto &vBlk = vModel.lQfProjBlocks[vBid];

    int vVlayer = vHparams.lFeatureLayers[vBid];
    GGML_ASSERT(vVlayer >= 0 && vVlayer < cNLayer);
    ggml_tensor *vH = lLayerOuts[vVlayer];

    ggml_tensor *vStream = mBuildBlock(vBlk, vH, vBid, vHparams.lProjSpatialOffsets[vBid], cNPatchesX,
                                       vHparams.vDownsampleWindowSide, vHparams.vDownsampleQuerySide, cQformerEps);
    mCb(vStream, (std::string("proj_") + std::to_string(vBid) + std::string("_v_out")).c_str(), vVlayer);
    vMmproj = vMmproj ? ggml_concat(vCtx0, vMmproj, vStream, 0) : vStream;
  }

  // --- Stage 1d: Append newline tokens if add_newline is set ---
  if (cAddNewline) {
    vMmproj = mAppendRowwiseNewlines(vCtx0, vMmproj);
    ggml_set_name(vMmproj, "g4v_mmproj_out_nl");
  } else {
    ggml_set_name(vMmproj, "g4v_mmproj_out");
  }
  ggml_build_forward_expand(vGf, vMmproj);

  return vGf;
}
