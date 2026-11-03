#include "models.h"

ggml_cgraph *clip_graph_minicpmv4_6::mBuild() {
  const bool cIs4x = vHparams.vNMerge == 2;
  const int cNPos = cNPatches;
  const int cHalfH = cNPatchesY / 2;
  const int cHalfW = cNPatchesX / 2;
  const int cNDs = cHalfH * cHalfW;
  const int cNOut = cIs4x ? cNDs : (cHalfH / 2) * (cHalfW / 2);

  auto vAddI32Input = [&](const char *pName, int pN) {
    ggml_tensor *vT = ggml_new_tensor_1d(vCtx0, GGML_TYPE_I32, pN);
    ggml_set_name(vT, pName);
    ggml_set_input(vT);
    return vT;
  };

  // position indices for ViT learned positional embeddings
  ggml_tensor *vPositions = vAddI32Input("positions", cNPos);
  ggml_tensor *vLearnedPosEmbd = ggml_get_rows(vCtx0, vModel.vPositionEmbeddings, vPositions);

  ggml_tensor *vVitMergerWindowIdx = nullptr;
  ggml_tensor *vVitMergerInvWindowIdx = nullptr;
  ggml_tensor *vVitMergerWindowMask = nullptr;
  ggml_tensor *vVitMergerDsIdx0 = nullptr;
  ggml_tensor *vVitMergerDsIdx1 = nullptr;
  ggml_tensor *vVitMergerDsIdx2 = nullptr;
  ggml_tensor *vVitMergerDsIdx3 = nullptr;

  if (!cIs4x) {
    // ViT merger window reorder indices + block-diagonal mask
    // (mask layout follows qwen2vl: -inf except for 4x4 blocks on the diagonal,
    // so each window-major group of 4 tokens only attends to itself)
    vVitMergerWindowIdx = vAddI32Input("vit_merger_window_idx", cNPos);
    vVitMergerInvWindowIdx = vAddI32Input("vit_merger_inv_window_idx", cNPos);
    vVitMergerWindowMask = ggml_new_tensor_2d(vCtx0, GGML_TYPE_F32, cNPos, cNPos);
    ggml_set_name(vVitMergerWindowMask, "vit_merger_window_mask");
    ggml_set_input(vVitMergerWindowMask);
    if (cFlashAttnType == cClipFlashAttnTypeEnabled) {
      vVitMergerWindowMask = ggml_cast(vCtx0, vVitMergerWindowMask, GGML_TYPE_F16);
    }

    // ViT merger 2x2 downsample gather indices
    vVitMergerDsIdx0 = vAddI32Input("vit_merger_ds_idx_0", cNDs);
    vVitMergerDsIdx1 = vAddI32Input("vit_merger_ds_idx_1", cNDs);
    vVitMergerDsIdx2 = vAddI32Input("vit_merger_ds_idx_2", cNDs);
    vVitMergerDsIdx3 = vAddI32Input("vit_merger_ds_idx_3", cNDs);
  }

  // final merger 2x2 downsample gather indices
  ggml_tensor *vMergerDsIdx0 = vAddI32Input("merger_ds_idx_0", cNOut);
  ggml_tensor *vMergerDsIdx1 = vAddI32Input("merger_ds_idx_1", cNOut);
  ggml_tensor *vMergerDsIdx2 = vAddI32Input("merger_ds_idx_2", cNOut);
  ggml_tensor *vMergerDsIdx3 = vAddI32Input("merger_ds_idx_3", cNOut);

  // patch embedding + positional embedding
  ggml_tensor *vInp = mBuildInp();
  vInp = ggml_add(vCtx0, vInp, vLearnedPosEmbd);
  mCb(vInp, "pos_embed", -1);

  ggml_tensor *vInpL = vInp;
  if (vModel.vPreLnW) {
    vInpL = mBuildNorm(vInpL, vModel.vPreLnW, vModel.vPreLnB, cNormTypeNormal, cEps, -1);
    mCb(vInpL, "pre_ln", -1);
  }

  auto vBuildVitLayers = [&](ggml_tensor *pInput, int pIlBegin, int pIlEnd, int64_t pNPosLayer) {
    for (int vIl = pIlBegin; vIl < pIlEnd; vIl++) {
      auto &vLayer = vModel.lLayers[vIl];
      ggml_tensor *vCur = pInput;

      vCur = mBuildNorm(vCur, vLayer.vLn1W, vLayer.vLn1B, cNormTypeNormal, cEps, vIl);
      mCb(vCur, "layer_inp_normed", vIl);

      {
        ggml_tensor *vQcur = mBuildMm(vLayer.vQW, vCur);
        if (vLayer.vQB) {
          vQcur = ggml_add(vCtx0, vQcur, vLayer.vQB);
        }
        ggml_tensor *vKcur = mBuildMm(vLayer.vKW, vCur);
        if (vLayer.vKB) {
          vKcur = ggml_add(vCtx0, vKcur, vLayer.vKB);
        }
        ggml_tensor *vVcur = mBuildMm(vLayer.vVW, vCur);
        if (vLayer.vVB) {
          vVcur = ggml_add(vCtx0, vVcur, vLayer.vVB);
        }

        vQcur = ggml_reshape_3d(vCtx0, vQcur, cDHead, cNHead, pNPosLayer);
        vKcur = ggml_reshape_3d(vCtx0, vKcur, cDHead, cNHead, pNPosLayer);
        vVcur = ggml_reshape_3d(vCtx0, vVcur, cDHead, cNHead, pNPosLayer);
        mCb(vQcur, "Qcur", vIl);
        mCb(vKcur, "Kcur", vIl);
        mCb(vVcur, "Vcur", vIl);

        vCur = mBuildAttn(vLayer.vOW, vLayer.vOB, vQcur, vKcur, vVcur, nullptr, vKqScale, vIl);
        mCb(vCur, "attn_out", vIl);
      }

      if (vLayer.vLs1W) {
        vCur = ggml_mul(vCtx0, vCur, vLayer.vLs1W);
        mCb(vCur, "attn_out_scaled", vIl);
      }
      vCur = ggml_add(vCtx0, vCur, pInput);
      pInput = vCur;
      mCb(vCur, "ffn_inp", vIl);

      vCur = mBuildNorm(vCur, vLayer.vLn2W, vLayer.vLn2B, cNormTypeNormal, cEps, vIl);
      mCb(vCur, "ffn_inp_normed", vIl);

      vCur = mBuildFfn(vCur, vLayer.vFfUpW, vLayer.vFfUpB, vLayer.vFfGateW, vLayer.vFfGateB, vLayer.vFfDownW,
                       vLayer.vFfDownB, vHparams.vFfnOp, vIl);
      mCb(vCur, "ffn_out", vIl);

      if (vLayer.vLs2W) {
        vCur = ggml_mul(vCtx0, vCur, vLayer.vLs2W);
        mCb(vCur, "ffn_out_scaled", vIl);
      }
      pInput = ggml_add(vCtx0, pInput, vCur);
      mCb(pInput, "layer_out", vIl);
    }
    return pInput;
  };

  if (!cIs4x) {
    const int cInsertLid = vHparams.vInsertLayerId;

    vInpL = vBuildVitLayers(vInpL, 0, cInsertLid + 1, cNPos);

    // ViT merger: window self-attention
    // Tokens are reordered to window-major (4 tokens per window are contiguous),
    // and a block-diagonal mask restricts attention to within each window. This
    // mirrors the qwen2vl windowed-attention pattern so mBuildAttn() can pick the
    // flash-attention path when available.
    {
      ggml_tensor *vResidual = vInpL;
      ggml_tensor *vCur = mBuildNorm(vInpL, vModel.vVitMergerLn1W, vModel.vVitMergerLn1B, cNormTypeNormal, cEps, -1);
      mCb(vCur, "vit_merger_attn_inp_normed", -1);

      vCur = ggml_get_rows(vCtx0, vCur, vVitMergerWindowIdx);
      mCb(vCur, "vit_merger_window_reorder", -1);

      ggml_tensor *vQcur = mBuildMm(vModel.vVitMergerAttnQW, vCur);
      if (vModel.vVitMergerAttnQB) {
        vQcur = ggml_add(vCtx0, vQcur, vModel.vVitMergerAttnQB);
      }
      ggml_tensor *vKcur = mBuildMm(vModel.vVitMergerAttnKW, vCur);
      if (vModel.vVitMergerAttnKB) {
        vKcur = ggml_add(vCtx0, vKcur, vModel.vVitMergerAttnKB);
      }
      ggml_tensor *vVcur = mBuildMm(vModel.vVitMergerAttnVW, vCur);
      if (vModel.vVitMergerAttnVB) {
        vVcur = ggml_add(vCtx0, vVcur, vModel.vVitMergerAttnVB);
      }

      vQcur = ggml_reshape_3d(vCtx0, vQcur, cDHead, cNHead, cNPos);
      vKcur = ggml_reshape_3d(vCtx0, vKcur, cDHead, cNHead, cNPos);
      vVcur = ggml_reshape_3d(vCtx0, vVcur, cDHead, cNHead, cNPos);
      mCb(vQcur, "vit_merger_Qcur", -1);
      mCb(vKcur, "vit_merger_Kcur", -1);
      mCb(vVcur, "vit_merger_Vcur", -1);

      vCur = mBuildAttn(vModel.vVitMergerAttnOW, vModel.vVitMergerAttnOB, vQcur, vKcur, vVcur, vVitMergerWindowMask,
                        vKqScale, -1);
      mCb(vCur, "vit_merger_attn_out", -1);

      vCur = ggml_get_rows(vCtx0, vCur, vVitMergerInvWindowIdx);
      vInpL = ggml_add(vCtx0, vCur, vResidual);
      mCb(vInpL, "vit_merger_attn_residual", -1);
    }

    // ViT merger: 2x2 spatial downsample + MLP (4 tokens -> 1)
    {
      ggml_tensor *vP0 = ggml_get_rows(vCtx0, vInpL, vVitMergerDsIdx0);
      ggml_tensor *vP1 = ggml_get_rows(vCtx0, vInpL, vVitMergerDsIdx1);
      ggml_tensor *vP2 = ggml_get_rows(vCtx0, vInpL, vVitMergerDsIdx2);
      ggml_tensor *vP3 = ggml_get_rows(vCtx0, vInpL, vVitMergerDsIdx3);

      ggml_tensor *vMeanRes = ggml_add(vCtx0, vP0, vP1);
      vMeanRes = ggml_add(vCtx0, vMeanRes, vP2);
      vMeanRes = ggml_add(vCtx0, vMeanRes, vP3);
      vMeanRes = ggml_scale(vCtx0, vMeanRes, 0.25f);
      mCb(vMeanRes, "vit_merger_ds_mean_res", -1);

      ggml_tensor *vCat = ggml_concat(vCtx0, vP0, vP1, 0);
      vCat = ggml_concat(vCtx0, vCat, vP2, 0);
      vCat = ggml_concat(vCtx0, vCat, vP3, 0);

      ggml_tensor *vCur = mBuildNorm(vCat, vModel.vVitMergerDsLnW, vModel.vVitMergerDsLnB, cNormTypeNormal, cEps, -1);
      mCb(vCur, "vit_merger_ds_normed", -1);

      // ViTWindowAttentionMerger downsample MLP uses gelu_pytorch_tanh (FFN_GELU)
      vCur = mBuildFfn(vCur, vModel.vVitMergerDsUpW, vModel.vVitMergerDsUpB, nullptr, nullptr, vModel.vVitMergerDsDownW,
                       vModel.vVitMergerDsDownB, cFfnGelu, -1);
      mCb(vCur, "vit_merger_ds_mlp_out", -1);

      vInpL = ggml_add(vCtx0, vCur, vMeanRes);
      mCb(vInpL, "vit_merger_ds_out", -1);
    }

    vInpL = vBuildVitLayers(vInpL, cInsertLid + 1, cNLayer, cNDs);
  } else {
    vInpL = vBuildVitLayers(vInpL, 0, cNLayer, cNPos);
  }

  if (vModel.vPostLnW) {
    vInpL = mBuildNorm(vInpL, vModel.vPostLnW, vModel.vPostLnB, cNormTypeNormal, cEps, -1);
    mCb(vInpL, "post_ln", -1);
  }

  // Final Merger (DownsampleMLP): another 2x2 spatial merge -> projector embedding
  {
    ggml_tensor *vP0 = ggml_get_rows(vCtx0, vInpL, vMergerDsIdx0);
    ggml_tensor *vP1 = ggml_get_rows(vCtx0, vInpL, vMergerDsIdx1);
    ggml_tensor *vP2 = ggml_get_rows(vCtx0, vInpL, vMergerDsIdx2);
    ggml_tensor *vP3 = ggml_get_rows(vCtx0, vInpL, vMergerDsIdx3);

    ggml_tensor *vCat = ggml_concat(vCtx0, vP0, vP1, 0);
    vCat = ggml_concat(vCtx0, vCat, vP2, 0);
    vCat = ggml_concat(vCtx0, vCat, vP3, 0);

    ggml_tensor *vCur = mBuildNorm(vCat, vModel.vMmInputNormW, vModel.vMmInputNormB, cNormTypeNormal, cEps, -1);
    mCb(vCur, "merger_normed", -1);

    // MiniCPMV4_6DownsampleMLP uses nn.GELU() (erf-based, FFN_GELU_ERF)
    vCur = mBuildFfn(vCur, vModel.vMmFfnUpW, vModel.vMmFfnUpB, nullptr, nullptr, vModel.vMmFfnDownW, vModel.vMmFfnDownB,
                     cFfnGeluErf, -1);
    mCb(vCur, "merger_out", -1);

    vInpL = vCur;
  }

  ggml_build_forward_expand(vGf, vInpL);
  return vGf;
}
