#include "models.h"

#include <algorithm>

ggml_cgraph *clip_graph_granite_speech::mBuild() {
  const int cNFrames = mGetImg(0).mNx();
  const int cContextSize = vHparams.vAudioChunkSize;
  const int cCtcLayer = cNLayer / 2;
  const int cConvKernel = vHparams.vAudioConvKernelSize;
  const int cConvPad = cConvKernel / 2;

  const int cNumBlocks = (cNFrames + cContextSize - 1) / cContextSize;
  const int cPaddedLen = cNumBlocks * cContextSize;
  const int cRemainder = cNFrames % cContextSize;

  // Calculate projector input dimension based on feature layers
  const int cProjInputDim = cNEmbd * (vHparams.lFeatureLayers.size() + 1);
  const bool cUseFeatureConcat = !vHparams.lFeatureLayers.empty();

  ggml_tensor *vAttnDists = ggml_new_tensor_1d(vCtx0, GGML_TYPE_I32, cContextSize * cContextSize);
  ggml_set_name(vAttnDists, "attn_dists");
  ggml_set_input(vAttnDists);

  ggml_tensor *vAttnMask = nullptr;
  if (cRemainder > 0) {
    vAttnMask = ggml_new_tensor_4d(vCtx0, GGML_TYPE_F32, cContextSize, cContextSize, 1, cNumBlocks);
    ggml_set_name(vAttnMask, "attn_mask");
    ggml_set_input(vAttnMask);
  }

  ggml_tensor *vInp = mBuildInpRaw(1);
  auto *vCur = ggml_cont(vCtx0, ggml_transpose(vCtx0, vInp));
  mCb(vCur, "inp_transposed", -1);

  vCur = mBuildMm(vModel.vInpProjW, vCur);
  vCur = ggml_add(vCtx0, vCur, vModel.vInpProjB);
  mCb(vCur, "inp_linear", -1);

  // Capture layer 0 if requested (after input_linear)
  ggml_tensor *vConcatResult = nullptr;
  if (cUseFeatureConcat) {
    if (std::find(vHparams.lFeatureLayers.begin(), vHparams.lFeatureLayers.end(), 0) != vHparams.lFeatureLayers.end()) {
      vConcatResult = vCur;
      mCb(vConcatResult, "feature_layer_0", -1);
    }
  }

  for (int vIl = 0; vIl < cNLayer; vIl++) {
    const auto &vLayer = vModel.lLayers[vIl];
    auto *vResidual = vCur;

    // ffn1 (half-step)
    {
      auto *vFfn1 = mBuildNorm(vCur, vLayer.vFfNormW, vLayer.vFfNormB, cNormTypeNormal, cEps, vIl);
      mCb(vFfn1, "ffn1_norm", vIl);

      vFfn1 = mBuildFfn(vFfn1, vLayer.vFfUpW, vLayer.vFfUpB, nullptr, nullptr, vLayer.vFfDownW, vLayer.vFfDownB,
                        cFfnSilu, vIl);
      mCb(vFfn1, "ffn1_out", vIl);

      vResidual = ggml_add(vCtx0, vResidual, ggml_scale(vCtx0, vFfn1, 0.5f));
      mCb(vResidual, "ffn1_residual", vIl);
    }

    // build_attn not used here: Shaw RPE needs pos_attn = mul_mat(pos_emb, Q)
    // injected between KQ product and softmax, which build_attn doesn't support
    {
      auto *vNormed = mBuildNorm(vResidual, vLayer.vLn1W, vLayer.vLn1B, cNormTypeNormal, cEps, vIl);
      mCb(vNormed, "attn_norm", vIl);

      if (cNFrames < cPaddedLen) {
        vNormed = ggml_pad(vCtx0, vNormed, 0, cPaddedLen - cNFrames, 0, 0);
      }

      ggml_tensor *vQ = mBuildMm(vLayer.vQW, vNormed);
      ggml_tensor *vK = mBuildMm(vLayer.vKW, vNormed);
      ggml_tensor *vV = mBuildMm(vLayer.vVW, vNormed);

      vQ = ggml_reshape_4d(vCtx0, vQ, cDHead, cNHead, cContextSize, cNumBlocks);
      vK = ggml_reshape_4d(vCtx0, vK, cDHead, cNHead, cContextSize, cNumBlocks);
      vV = ggml_reshape_4d(vCtx0, vV, cDHead, cNHead, cContextSize, cNumBlocks);

      ggml_tensor *vQPerm = ggml_permute(vCtx0, vQ, 0, 2, 1, 3);
      ggml_tensor *vKPerm = ggml_cont(vCtx0, ggml_permute(vCtx0, vK, 0, 2, 1, 3));

      ggml_tensor *vKq = ggml_mul_mat(vCtx0, vKPerm, vQPerm);

      // Shaw RPE: pos_emb ne[2]=1 broadcasts against Q ne[2]=num_blocks in mul_mat
      ggml_tensor *vPosEmb = ggml_get_rows(vCtx0, vLayer.vAttnRelPosEmb, vAttnDists);
      vPosEmb = ggml_reshape_3d(vCtx0, vPosEmb, cDHead, cContextSize, cContextSize);
      vPosEmb = ggml_reshape_4d(vCtx0, vPosEmb, cDHead, cContextSize, 1, cContextSize);

      ggml_tensor *vQShaw = ggml_permute(vCtx0, vQ, 0, 1, 3, 2);
      ggml_tensor *vPosAttn = ggml_mul_mat(vCtx0, vPosEmb, vQShaw);
      vPosAttn = ggml_cont(vCtx0, ggml_permute(vCtx0, vPosAttn, 0, 2, 3, 1));

      ggml_tensor *vScores = ggml_add(vCtx0, vKq, vPosAttn);
      ggml_tensor *vAttnWeights = ggml_soft_max_ext(vCtx0, vScores, vAttnMask, vKqScale, 0.0f);

      ggml_tensor *vVPerm = ggml_cont(vCtx0, ggml_permute(vCtx0, vV, 1, 2, 0, 3));
      ggml_tensor *vAttnOut = ggml_mul_mat(vCtx0, vVPerm, vAttnWeights);

      vAttnOut = ggml_permute(vCtx0, vAttnOut, 0, 2, 1, 3);
      vAttnOut = ggml_cont_2d(vCtx0, vAttnOut, cNEmbd, cPaddedLen);

      if (cNFrames < cPaddedLen) {
        vAttnOut = ggml_view_2d(vCtx0, vAttnOut, cNEmbd, cNFrames, vAttnOut->nb[1], 0);
      }

      vCur = mBuildMm(vLayer.vOW, vAttnOut);
      vCur = ggml_add(vCtx0, vCur, vLayer.vOB);
      mCb(vCur, "attn_out", vIl);
    }

    vResidual = ggml_add(vCtx0, vResidual, vCur);

    // conv module
    {
      vCur = mBuildNorm(vResidual, vLayer.vNormConvW, vLayer.vNormConvB, cNormTypeNormal, cEps, vIl);
      mCb(vCur, "conv_norm", vIl);

      auto *vX = mBuildMm(vLayer.vConvPw1W, vCur);
      vX = ggml_add(vCtx0, vX, vLayer.vConvPw1B);
      mCb(vX, "conv_pw1", vIl);

      // GLU: ggml has no fused op, manual split + sigmoid gate
      {
        int64_t vD = vX->ne[0] / 2;
        ggml_tensor *vGate = ggml_sigmoid(vCtx0, ggml_view_2d(vCtx0, vX, vD, vX->ne[1], vX->nb[1], vD * vX->nb[0]));
        vX = ggml_mul(vCtx0, ggml_view_2d(vCtx0, vX, vD, vX->ne[1], vX->nb[1], 0), vGate);
        vX = ggml_cont(vCtx0, ggml_transpose(vCtx0, vX));
      }
      mCb(vX, "conv_glu", vIl);

      vX = ggml_pad(vCtx0, vX, cConvPad, 0, 0, 0);
      vX = ggml_roll(vCtx0, vX, cConvPad, 0, 0, 0);
      vX = ggml_pad(vCtx0, vX, cConvPad, 0, 0, 0);
      vX = ggml_ssm_conv(vCtx0, vX, vLayer.vConvDwW);
      mCb(vX, "conv_dw", vIl);

      // folded batch norm
      vX = ggml_add(vCtx0, ggml_mul(vCtx0, vX, vLayer.vConvNormW), vLayer.vConvNormB);
      vX = ggml_silu(vCtx0, vX);
      mCb(vX, "conv_bn_silu", vIl);

      vX = mBuildMm(vLayer.vConvPw2W, vX);
      vX = ggml_add(vCtx0, vX, vLayer.vConvPw2B);
      mCb(vX, "conv_pw2", vIl);

      vCur = vX;
    }

    vResidual = ggml_add(vCtx0, vResidual, vCur);

    // ffn2 (half-step)
    {
      auto *vFfn2 = mBuildNorm(vResidual, vLayer.vFfNorm1W, vLayer.vFfNorm1B, cNormTypeNormal, cEps, vIl);
      mCb(vFfn2, "ffn2_norm", vIl);

      vFfn2 = mBuildFfn(vFfn2, vLayer.vFfUp1W, vLayer.vFfUp1B, nullptr, nullptr, vLayer.vFfDown1W, vLayer.vFfDown1B,
                        cFfnSilu, vIl);
      mCb(vFfn2, "ffn2_out", vIl);

      vResidual = ggml_add(vCtx0, vResidual, ggml_scale(vCtx0, vFfn2, 0.5f));
    }

    vCur = mBuildNorm(vResidual, vLayer.vLn2W, vLayer.vLn2B, cNormTypeNormal, cEps, vIl);
    mCb(vCur, "layer_out", vIl);

    // Capture intermediate layer (il + 1) if requested
    if (cUseFeatureConcat) {
      if (vHparams.mIsFeatureLayer(vIl + 1)) {
        if (vConcatResult == nullptr) {
          vConcatResult = vCur;
        } else {
          vConcatResult = ggml_concat(vCtx0, vConcatResult, vCur, 0);
        }
        mCb(vConcatResult, fStringFormat("feature_layer_%d", vIl + 1).c_str(), vIl);
      }
    }

    // CTC branch
    if (vIl + 1 == cCtcLayer) {
      auto *vMid = mBuildMm(vModel.vCtcOutW, vCur);
      vMid = ggml_add(vCtx0, vMid, vModel.vCtcOutB);
      vMid = ggml_soft_max(vCtx0, vMid);
      vMid = mBuildMm(vModel.vCtcOutMidW, vMid);
      vMid = ggml_add(vCtx0, vMid, vModel.vCtcOutMidB);
      vCur = ggml_add(vCtx0, vCur, vMid);
      mCb(vCur, "ctc_branch", vIl);
    }
  }

  // Append final output to concatenated features if using feature concatenation
  if (cUseFeatureConcat && vConcatResult != nullptr) {
    vConcatResult = ggml_concat(vCtx0, vConcatResult, vCur, 0);
    mCb(vConcatResult, "concat_final", -1);
    vCur = vConcatResult;
  }

  mCb(vCur, "encoder_out", -1);

  // QFormer projector
  {
    const int cWindowSize = vHparams.vAudioProjWindowSize;
    const int cNumQueries = cWindowSize / vHparams.vAudioProjDownsampleRate;
    const int cProjNHead = vHparams.vAudioProjHeadCount;
    const int cProjDHead = cNEmbd / cProjNHead;
    const float cProjKqScale = 1.0f / sqrtf((float)cProjDHead);
    const float cProjEps = 1e-12f;
    const int cNblocksProj = (cNFrames + cWindowSize - 1) / cWindowSize;
    const int cPaddedProj = cNblocksProj * cWindowSize;

    if (cNFrames < cPaddedProj) {
      vCur = ggml_pad(vCtx0, vCur, 0, cPaddedProj - cNFrames, 0, 0);
    }

    ggml_tensor *vEncWindows = ggml_reshape_3d(vCtx0, vCur, cProjInputDim, cWindowSize, cNblocksProj);

    ggml_tensor *vQueries = mBuildNorm(vModel.lQfProjBlocks[0].vQfProjQuery, vModel.lQfProjBlocks[0].vQfProjNormW,
                                       vModel.lQfProjBlocks[0].vQfProjNormB, cNormTypeNormal, cProjEps, -1);
    {
      ggml_tensor *vQ3d = ggml_reshape_3d(vCtx0, vQueries, cNEmbd, cNumQueries, 1);
      ggml_tensor *vQShape = ggml_new_tensor_3d(vCtx0, GGML_TYPE_F32, cNEmbd, cNumQueries, cNblocksProj);
      vQueries = ggml_repeat(vCtx0, vQ3d, vQShape);
    }

    for (int vIl = 0; vIl < (int)vModel.lQfProjBlocks[0].lQfProjLayers.size(); vIl++) {
      const auto &vPl = vModel.lQfProjBlocks[0].lQfProjLayers[vIl];

      // self-attention
      {
        ggml_tensor *vQ = ggml_add(vCtx0, mBuildMm(vPl.vQW, vQueries), vPl.vQB);
        ggml_tensor *vK = ggml_add(vCtx0, mBuildMm(vPl.vKW, vQueries), vPl.vKB);
        ggml_tensor *vV = ggml_add(vCtx0, mBuildMm(vPl.vVW, vQueries), vPl.vVB);

        vQ = ggml_reshape_4d(vCtx0, vQ, cProjDHead, cProjNHead, cNumQueries, cNblocksProj);
        vK = ggml_reshape_4d(vCtx0, vK, cProjDHead, cProjNHead, cNumQueries, cNblocksProj);
        vV = ggml_reshape_4d(vCtx0, vV, cProjDHead, cProjNHead, cNumQueries, cNblocksProj);

        ggml_tensor *vSaOut = mBuildAttn(vPl.vOW, vPl.vOB, vQ, vK, vV, nullptr, cProjKqScale, vIl);
        vSaOut = ggml_reshape_3d(vCtx0, vSaOut, cNEmbd, cNumQueries, cNblocksProj);

        vQueries = mBuildNorm(ggml_add(vCtx0, vSaOut, vQueries), vPl.vLn1W, vPl.vLn1B, cNormTypeNormal, cProjEps, vIl);
      }

      // cross-attention
      {
        ggml_tensor *vQ = ggml_add(vCtx0, mBuildMm(vPl.vCrossAttnQW, vQueries), vPl.vCrossAttnQB);
        ggml_tensor *vK = ggml_add(vCtx0, mBuildMm(vPl.vCrossAttnKW, vEncWindows), vPl.vCrossAttnKB);
        ggml_tensor *vV = ggml_add(vCtx0, mBuildMm(vPl.vCrossAttnVW, vEncWindows), vPl.vCrossAttnVB);

        vQ = ggml_reshape_4d(vCtx0, vQ, cProjDHead, cProjNHead, cNumQueries, cNblocksProj);
        vK = ggml_reshape_4d(vCtx0, vK, cProjDHead, cProjNHead, cWindowSize, cNblocksProj);
        vV = ggml_reshape_4d(vCtx0, vV, cProjDHead, cProjNHead, cWindowSize, cNblocksProj);

        ggml_tensor *vCaOut = mBuildAttn(vPl.vCrossAttnOW, vPl.vCrossAttnOB, vQ, vK, vV, nullptr, cProjKqScale, vIl);
        vCaOut = ggml_reshape_3d(vCtx0, vCaOut, cNEmbd, cNumQueries, cNblocksProj);

        vQueries = mBuildNorm(ggml_add(vCtx0, vCaOut, vQueries), vPl.vCrossAttnNormW, vPl.vCrossAttnNormB,
                              cNormTypeNormal, cProjEps, vIl);
      }

      // ffn
      {
        ggml_tensor *vFfnOut =
            mBuildFfn(vQueries, vPl.vFfUpW, vPl.vFfUpB, nullptr, nullptr, vPl.vFfDownW, vPl.vFfDownB, cFfnGelu, vIl);

        vQueries = mBuildNorm(ggml_add(vCtx0, vFfnOut, vQueries), vPl.vLn2W, vPl.vLn2B, cNormTypeNormal, cProjEps, vIl);
      }
    }

    vCur = ggml_reshape_2d(vCtx0, vQueries, cNEmbd, cNumQueries * cNblocksProj);
    vCur =
        ggml_add(vCtx0, mBuildMm(vModel.lQfProjBlocks[0].vQfProjLinearW, vCur), vModel.lQfProjBlocks[0].vQfProjLinearB);
    mCb(vCur, "projector_out", -1);
  }

  ggml_build_forward_expand(vGf, vCur);

  return vGf;
}
