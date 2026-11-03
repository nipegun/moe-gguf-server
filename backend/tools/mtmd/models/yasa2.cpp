// ABOUTME: Yasa2 vision encoder graph builder for ConvNeXt-based architecture.
// ABOUTME: Implements patch embedding, ConvNeXt stages with GRN, and adaptive pooling.

#include "models.h"

static ggml_tensor *fAddChannelBias(ggml_context *pCtx0, ggml_tensor *pXWhcb, ggml_tensor *pBC) {
  if (!pBC) {
    return pXWhcb;
  }
  ggml_tensor *vB4 = ggml_reshape_4d(pCtx0, pBC, 1, 1, pBC->ne[0], 1);
  return ggml_add(pCtx0, pXWhcb, vB4);
}

static ggml_tensor *fMulChannelWeight(ggml_context *pCtx0, ggml_tensor *pXWhcb, ggml_tensor *pWC) {
  if (!pWC) {
    return pXWhcb;
  }
  ggml_tensor *vW4 = ggml_reshape_4d(pCtx0, pWC, 1, 1, pWC->ne[0], 1);
  return ggml_mul(pCtx0, pXWhcb, vW4);
}

ggml_tensor *clip_graph_yasa2::mLayerNormChannels(ggml_tensor *pInp, ggml_tensor *pW, ggml_tensor *pB, float pEps) {
  // Match HF ConvNextLayerNorm(channels_first):
  // u = mean_c(x), s = mean_c((x-u)^2), x = (x-u)/sqrt(s+eps)
  // cast back to input dtype before affine.
  ggml_tensor *vCur = ggml_permute(vCtx0, pInp, 2, 1, 0, 3); // [W,H,C,B] -> [C,H,W,B]
  vCur = ggml_cont(vCtx0, vCur);

  ggml_tensor *vU = ggml_mean(vCtx0, vCur);     // [1,H,W,B]
  ggml_tensor *vXm = ggml_sub(vCtx0, vCur, vU); // [C,H,W,B]

  ggml_tensor *vS = ggml_mul(vCtx0, vXm, vXm); // [C,H,W,B]
  vS = ggml_mean(vCtx0, vS);                   // [1,H,W,B]
  vS = ggml_clamp(vCtx0, vS, pEps, 1e30f);     // avoid div-by-zero in no-alloc warmup
  vS = ggml_sqrt(vCtx0, vS);                   // [1,H,W,B]

  ggml_tensor *vXhat = ggml_div(vCtx0, vXm, vS);  // [C,H,W,B]
  vXhat = ggml_permute(vCtx0, vXhat, 2, 1, 0, 3); // [W,H,C,B]
  vXhat = ggml_cont(vCtx0, vXhat);
  vXhat = fMulChannelWeight(vCtx0, vXhat, pW);
  vXhat = fAddChannelBias(vCtx0, vXhat, pB);
  return vXhat;
}

ggml_tensor *clip_graph_yasa2::mConvnextGrn(ggml_tensor *pInp, ggml_tensor *pW, ggml_tensor *pB) {
  // Exact ConvNeXtV2 GRN:
  // Gx = ||x||_2 over spatial dims (W,H), Nx = Gx / (mean_c(Gx) + eps)
  // y  = w * (x * Nx) + b + x
  const int64_t cWdim = pInp->ne[0];
  const int64_t cHdim = pInp->ne[1];
  const int64_t cCdim = pInp->ne[2];
  const int64_t cBdim = pInp->ne[3];

  // Keep GRN math in fp32 for stability; fp16/bf16 accumulation can drift.
  ggml_tensor *vSq = ggml_mul(vCtx0, pInp, pInp);
  ggml_tensor *vSqFlat = ggml_reshape_4d(vCtx0, vSq, cWdim * cHdim, cCdim, 1, cBdim); // [WH,C,1,B]
  ggml_tensor *vGx = ggml_sum_rows(vCtx0, vSqFlat);                                   // [1,C,1,B]
  vGx = ggml_sqrt(vCtx0, vGx);                                                        // [1,C,1,B]

  ggml_tensor *vGxChFirst = ggml_permute(vCtx0, vGx, 1, 0, 2, 3); // [C,1,1,B]
  vGxChFirst = ggml_cont(vCtx0, vGxChFirst);
  ggml_tensor *vGxMean = ggml_mean(vCtx0, vGxChFirst); // [1,1,1,B]

  vGxMean = ggml_clamp(vCtx0, vGxMean, 1e-6f, 1e30f); // approx +eps, warmup-safe
  ggml_tensor *vNx = ggml_div(vCtx0, vGx, vGxMean);   // [1,C,1,B]
  vNx = ggml_permute(vCtx0, vNx, 0, 2, 1, 3);         // [1,1,C,B]
  vNx = ggml_cont(vCtx0, vNx);

  ggml_tensor *vXnx = ggml_mul(vCtx0, pInp, vNx);
  vXnx = fMulChannelWeight(vCtx0, vXnx, pW);
  vXnx = fAddChannelBias(vCtx0, vXnx, pB);
  return ggml_add(vCtx0, pInp, vXnx);
}

ggml_cgraph *clip_graph_yasa2::mBuild() {
  ggml_tensor *vCur = mBuildInpRaw();

  // Patch embedding Conv2d(kernel=4, stride=4)
  vCur = ggml_conv_2d(vCtx0, vModel.vYasaPatchW, vCur, cPatchSize, cPatchSize, 0, 0, 1, 1);
  vCur = fAddChannelBias(vCtx0, vCur, vModel.vYasaPatchB);
  ggml_set_name(vCur, "yasa2_patch_conv_out");
  mCb(vCur, "yasa2_patch_conv_out", -1);
  vCur = mLayerNormChannels(vCur, vModel.vYasaPatchLnW, vModel.vYasaPatchLnB, cEps);
  ggml_set_name(vCur, "yasa2_patch_ln_out");
  mCb(vCur, "yasa2_patch_ln_out", -1);

  // ConvNeXt stages
  for (size_t vS = 0; vS < vModel.lYasaStages.size(); ++vS) {
    const auto &vStage = vModel.lYasaStages[vS];

    if (vStage.vDownConvW) {
      vCur = mLayerNormChannels(vCur, vStage.vDownLnW, vStage.vDownLnB, cEps);
      vCur = ggml_conv_2d(vCtx0, vStage.vDownConvW, vCur, 2, 2, 0, 0, 1, 1);
      vCur = fAddChannelBias(vCtx0, vCur, vStage.vDownConvB);
      ggml_format_name(vCur, "yasa2_stage%zu_down_out", vS);
    }

    for (size_t vBi = 0; vBi < vStage.lBlocks.size(); ++vBi) {
      const auto &vBlk = vStage.lBlocks[vBi];
      ggml_tensor *vRes = vCur;

      ggml_tensor *vX = ggml_conv_2d_dw(vCtx0, vBlk.vDwW, vCur, 1, 1, 3, 3, 1, 1);
      vX = fAddChannelBias(vCtx0, vX, vBlk.vDwB);
      vX = mLayerNormChannels(vX, vBlk.vLnW, vBlk.vLnB, cEps);

      // pwconv1/pwconv2 are HF Linear layers over channels; implement via matmul on tokens.
      const int64_t cW = vX->ne[0];
      const int64_t cH = vX->ne[1];
      const int64_t cB = vX->ne[3];

      ggml_tensor *vTok = ggml_reshape_3d(vCtx0, vX, cW * cH, vX->ne[2], cB); // [T,C,B]
      vTok = ggml_permute(vCtx0, vTok, 1, 0, 2, 3);                           // [C,T,B]
      vTok = ggml_cont(vCtx0, vTok);

      vTok = ggml_mul_mat(vCtx0, vBlk.vPw1W, vTok); // [4C,T,B]
      if (vBlk.vPw1B) {
        ggml_tensor *vB1 = ggml_reshape_3d(vCtx0, vBlk.vPw1B, vBlk.vPw1B->ne[0], 1, 1); // [4C,1,1]
        vTok = ggml_add(vCtx0, vTok, vB1);
      }
      vX = ggml_permute(vCtx0, vTok, 1, 0, 2, 3); // [T,4C,B]
      vX = ggml_cont(vCtx0, vX);
      vX = ggml_reshape_4d(vCtx0, vX, cW, cH, vTok->ne[0], cB); // [W,H,4C,B]
      vX = ggml_gelu_erf(vCtx0, vX);
      vX = mConvnextGrn(vX, vBlk.vGrnW, vBlk.vGrnB);

      vTok = ggml_reshape_3d(vCtx0, vX, cW * cH, vX->ne[2], cB); // [T,4C,B]
      vTok = ggml_permute(vCtx0, vTok, 1, 0, 2, 3);              // [4C,T,B]
      vTok = ggml_cont(vCtx0, vTok);

      vTok = ggml_mul_mat(vCtx0, vBlk.vPw2W, vTok); // [C,T,B]
      if (vBlk.vPw2B) {
        ggml_tensor *vB2 = ggml_reshape_3d(vCtx0, vBlk.vPw2B, vBlk.vPw2B->ne[0], 1, 1); // [C,1,1]
        vTok = ggml_add(vCtx0, vTok, vB2);
      }
      vX = ggml_permute(vCtx0, vTok, 1, 0, 2, 3); // [T,C,B]
      vX = ggml_cont(vCtx0, vX);
      vX = ggml_reshape_4d(vCtx0, vX, cW, cH, vTok->ne[0], cB); // [W,H,C,B]

      vCur = ggml_add(vCtx0, vRes, vX);
      ggml_format_name(vCur, "yasa2_stage%zu_blk%zu_out", vS, vBi);
    }
  }

  // HF path adds vision position embeddings BEFORE adaptive pooling.
  const int64_t cPreW = vCur->ne[0];
  const int64_t cPreH = vCur->ne[1];
  ggml_tensor *vTokensPre = ggml_reshape_3d(vCtx0, vCur, cPreW * cPreH, vCur->ne[2], vCur->ne[3]); // [T,C,B]
  vTokensPre = ggml_permute(vCtx0, vTokensPre, 1, 0, 2, 3);                                        // [C,T,B]
  vTokensPre = ggml_cont(vCtx0, vTokensPre);
  if (vModel.vYasaVisionPosEmbed && vTokensPre->ne[1] == vModel.vYasaVisionPosEmbed->ne[1]) {
    const int64_t cNCh = vModel.vYasaVisionPosEmbed->ne[0];
    const int64_t cNTokens = vModel.vYasaVisionPosEmbed->ne[1];
    ggml_tensor *vPos = ggml_reshape_3d(vCtx0, vModel.vYasaVisionPosEmbed, (int)cNCh, (int)cNTokens, 1);
    vTokensPre = ggml_add(vCtx0, vTokensPre, vPos);
  }
  vCur = ggml_permute(vCtx0, vTokensPre, 1, 0, 2, 3); // [T,C,B]
  vCur = ggml_cont(vCtx0, vCur);
  vCur = ggml_reshape_4d(vCtx0, vCur, cPreW, cPreH, vCur->ne[1], vCur->ne[2]); // [W,H,C,B]

  // AdaptiveAvgPool2d target is 8x8 for real inputs, but warmup can use tiny images.
  const int cPooledW = std::min(8, (int)vCur->ne[0]);
  const int cPooledH = std::min(8, (int)vCur->ne[1]);
  const int cKw = std::max(1, (int)vCur->ne[0] / cPooledW);
  const int cKh = std::max(1, (int)vCur->ne[1] / cPooledH);
  vCur = ggml_pool_2d(vCtx0, vCur, GGML_OP_POOL_AVG, cKw, cKh, cKw, cKh, 0, 0);

  // [W,H,C,B] -> [C,T,B]
  ggml_tensor *vTokens = ggml_reshape_3d(vCtx0, vCur, vCur->ne[0] * vCur->ne[1], vCur->ne[2], vCur->ne[3]);
  vTokens = ggml_permute(vCtx0, vTokens, 1, 0, 2, 3);
  vTokens = ggml_cont(vCtx0, vTokens);
  mCb(vTokens, "yasa2_tokens", -1);

  GGML_ASSERT(vModel.vMm0W && vModel.vMm2W);
  ggml_tensor *vEmbeddings =
      mBuildFfn(vTokens, vModel.vMm0W, vModel.vMm0B, nullptr, nullptr, vModel.vMm2W, vModel.vMm2B, cFfnGeluErf, -1);
  mCb(vEmbeddings, "yasa2_emb", -1);

  ggml_build_forward_expand(vGf, vEmbeddings);
  return vGf;
}
