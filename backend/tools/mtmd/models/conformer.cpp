#include "models.h"

ggml_cgraph *clip_graph_conformer::mBuild() {
  const int cNFrames = mGetImg(0).mNx();
  const int cNPos = cNFrames / 2;
  const int cNPosEmbd = (((((cNFrames + 1) / 2) + 1) / 2 + 1) / 2) * 2 - 1;
  GGML_ASSERT(vModel.vPositionEmbeddings->ne[1] >= cNPos);

  ggml_tensor *vPosEmb = ggml_new_tensor_2d(vCtx0, GGML_TYPE_F32, 512, cNPosEmbd);
  ggml_set_name(vPosEmb, "pos_emb");
  ggml_set_input(vPosEmb);
  ggml_build_forward_expand(vGf, vPosEmb);

  ggml_tensor *vInp = mBuildInpRaw(1);

  auto *vCur = ggml_cont(vCtx0, ggml_transpose(vCtx0, vInp));

  // pre encode, conv subsampling
  {
    // layer.0 - conv2d
    vCur = ggml_conv_2d(vCtx0, vModel.aPreEncodeConvXW[0], vCur, 2, 2, 1, 1, 1, 1);
    vCur = ggml_add(vCtx0, vCur, vModel.aPreEncodeConvXB[0]);
    mCb(vCur, "conformer.pre_encode.conv.{}", 0);

    // layer.1 - relu
    vCur = ggml_relu_inplace(vCtx0, vCur);

    // layer.2 conv2d dw
    vCur = ggml_conv_2d_dw_direct(vCtx0, vModel.aPreEncodeConvXW[2], vCur, 2, 2, 1, 1, 1, 1);
    vCur = ggml_add(vCtx0, vCur, vModel.aPreEncodeConvXB[2]);
    mCb(vCur, "conformer.pre_encode.conv.{}", 2);

    // layer.3 conv2d
    vCur = ggml_conv_2d_direct(vCtx0, vModel.aPreEncodeConvXW[3], vCur, 1, 1, 0, 0, 1, 1);
    vCur = ggml_add(vCtx0, vCur, vModel.aPreEncodeConvXB[3]);
    mCb(vCur, "conformer.pre_encode.conv.{}", 3);

    // layer.4 - relu
    vCur = ggml_relu_inplace(vCtx0, vCur);

    // layer.5 conv2d dw
    vCur = ggml_conv_2d_dw_direct(vCtx0, vModel.aPreEncodeConvXW[5], vCur, 2, 2, 1, 1, 1, 1);
    vCur = ggml_add(vCtx0, vCur, vModel.aPreEncodeConvXB[5]);
    mCb(vCur, "conformer.pre_encode.conv.{}", 5);

    // layer.6 conv2d
    vCur = ggml_conv_2d_direct(vCtx0, vModel.aPreEncodeConvXW[6], vCur, 1, 1, 0, 0, 1, 1);
    vCur = ggml_add(vCtx0, vCur, vModel.aPreEncodeConvXB[6]);
    mCb(vCur, "conformer.pre_encode.conv.{}", 6);

    // layer.7 - relu
    vCur = ggml_relu_inplace(vCtx0, vCur);

    // flatten channel and frequency axis
    vCur = ggml_cont(vCtx0, ggml_permute(vCtx0, vCur, 0, 2, 1, 3));
    vCur = ggml_reshape_2d(vCtx0, vCur, vCur->ne[0] * vCur->ne[1], vCur->ne[2]);

    // calculate out
    vCur = mBuildMm(vModel.vPreEncodeOutW, vCur);
    vCur = ggml_add(vCtx0, vCur, vModel.vPreEncodeOutB);
    mCb(vCur, "conformer.pre_encode.out", -1);
  }

  // pos_emb
  mCb(vPosEmb, "pos_emb", -1);

  for (int vIl = 0; vIl < vHparams.vNLayer; vIl++) {
    const auto &vLayer = vModel.lLayers[vIl];

    auto *vResidual = vCur;

    mCb(vCur, "layer.in", vIl);

    // feed_forward1
    vCur = mBuildNorm(vCur, vLayer.vFfNormW, vLayer.vFfNormB, cNormTypeNormal, 1e-5, vIl);
    mCb(vCur, "conformer.layers.{}.norm_feed_forward1", vIl);

    vCur = mBuildFfn(vCur, vLayer.vFfUpW, vLayer.vFfUpB, nullptr, nullptr, vLayer.vFfDownW, vLayer.vFfDownB, cFfnSilu,
                     vIl);
    mCb(vCur, "conformer.layers.{}.feed_forward1.linear2", vIl);

    const auto cFcFactor = 0.5f;
    vResidual = ggml_add(vCtx0, vResidual, ggml_scale(vCtx0, vCur, cFcFactor));

    // self-attention
    {
      vCur = mBuildNorm(vResidual, vLayer.vLn1W, vLayer.vLn1B, cNormTypeNormal, 1e-5, vIl);
      mCb(vCur, "conformer.layers.{}.norm_self_att", vIl);

      ggml_tensor *vQcur = mBuildMm(vLayer.vQW, vCur);
      vQcur = ggml_add(vCtx0, vQcur, vLayer.vQB);
      vQcur = ggml_reshape_3d(vCtx0, vQcur, cDHead, cNHead, vQcur->ne[1]);
      ggml_tensor *vQBiasU = ggml_add(vCtx0, vQcur, vLayer.vPosBiasU);
      vQBiasU = ggml_permute(vCtx0, vQBiasU, 0, 2, 1, 3);
      ggml_tensor *vQBiasV = ggml_add(vCtx0, vQcur, vLayer.vPosBiasV);
      vQBiasV = ggml_permute(vCtx0, vQBiasV, 0, 2, 1, 3);

      // TODO @ngxson : some cont can/should be removed when ggml_mul_mat support these cases
      ggml_tensor *vKcur = mBuildMm(vLayer.vKW, vCur);
      vKcur = ggml_add(vCtx0, vKcur, vLayer.vKB);
      vKcur = ggml_reshape_3d(vCtx0, vKcur, cDHead, cNHead, vKcur->ne[1]);
      vKcur = ggml_cont(vCtx0, ggml_permute(vCtx0, vKcur, 0, 2, 1, 3));

      ggml_tensor *vVcur = mBuildMm(vLayer.vVW, vCur);
      vVcur = ggml_add(vCtx0, vVcur, vLayer.vVB);
      vVcur = ggml_reshape_3d(vCtx0, vVcur, cDHead, cNHead, vVcur->ne[1]);
      vVcur = ggml_cont(vCtx0, ggml_permute(vCtx0, vVcur, 1, 2, 0, 3));

      // build_attn won't fit due to matrix_ac and matrix_bd separation
      ggml_tensor *vMatrixAc = ggml_mul_mat(vCtx0, vQBiasU, vKcur);
      vMatrixAc = ggml_cont(vCtx0, ggml_permute(vCtx0, vMatrixAc, 1, 0, 2, 3));
      mCb(vMatrixAc, "conformer.layers.{}.self_attn.id3", vIl);

      auto *vP = mBuildMm(vLayer.vLinearPosW, vPosEmb);
      mCb(vP, "conformer.layers.{}.self_attn.linear_pos", vIl);
      vP = ggml_reshape_3d(vCtx0, vP, cDHead, cNHead, vP->ne[1]);
      vP = ggml_permute(vCtx0, vP, 0, 2, 1, 3);

      auto *vMatrixBd = ggml_mul_mat(vCtx0, vQBiasV, vP);
      vMatrixBd = ggml_cont(vCtx0, ggml_permute(vCtx0, vMatrixBd, 1, 0, 2, 3));

      // rel shift
      {
        const auto cPosLen = vMatrixBd->ne[0];
        const auto cQLen = vMatrixBd->ne[1];
        const auto cH = vMatrixBd->ne[2];
        vMatrixBd = ggml_pad(vCtx0, vMatrixBd, 1, 0, 0, 0);
        vMatrixBd = ggml_roll(vCtx0, vMatrixBd, 1, 0, 0, 0);
        vMatrixBd = ggml_reshape_3d(vCtx0, vMatrixBd, cQLen, cPosLen + 1, cH);
        vMatrixBd = ggml_view_3d(vCtx0, vMatrixBd, cQLen, cPosLen, cH, vMatrixBd->nb[1], vMatrixBd->nb[2],
                                 vMatrixBd->nb[0] * cQLen);
        vMatrixBd = ggml_cont_3d(vCtx0, vMatrixBd, cPosLen, cQLen, cH);
      }

      vMatrixBd = ggml_view_3d(vCtx0, vMatrixBd, vMatrixAc->ne[0], vMatrixBd->ne[1], vMatrixBd->ne[2], vMatrixBd->nb[1],
                               vMatrixBd->nb[2], 0);
      auto *vScores = ggml_add(vCtx0, vMatrixAc, vMatrixBd);
      vScores = ggml_scale(vCtx0, vScores, 1.0f / std::sqrt(cDHead));
      mCb(vScores, "conformer.layers.{}.self_attn.id0", vIl);

      ggml_tensor *vAttn = ggml_soft_max(vCtx0, vScores);
      ggml_tensor *vX = ggml_mul_mat(vCtx0, vAttn, vVcur);
      vX = ggml_permute(vCtx0, vX, 2, 0, 1, 3);
      vX = ggml_cont_2d(vCtx0, vX, vX->ne[0] * vX->ne[1], vX->ne[2]);

      ggml_tensor *vOut = mBuildMm(vLayer.vOW, vX);
      vOut = ggml_add(vCtx0, vOut, vLayer.vOB);
      mCb(vOut, "conformer.layers.{}.self_attn.linear_out", vIl);

      vCur = vOut;
    }

    vResidual = ggml_add(vCtx0, vResidual, vCur);
    vCur = mBuildNorm(vResidual, vLayer.vNormConvW, vLayer.vNormConvB, cNormTypeNormal, 1e-5, vIl);
    mCb(vCur, "conformer.layers.{}.norm_conv", vIl);

    // conv
    {
      auto *vX = vCur;
      vX = mBuildMm(vLayer.vConvPw1W, vX);
      vX = ggml_add(vCtx0, vX, vLayer.vConvPw1B);
      mCb(vX, "conformer.layers.{}.conv.pointwise_conv1", vIl);

      // ggml_glu doesn't support sigmoid
      // TODO @ngxson : support this ops in ggml
      {
        int64_t vD = vX->ne[0] / 2;
        ggml_tensor *vGate = ggml_sigmoid(vCtx0, ggml_view_2d(vCtx0, vX, vD, vX->ne[1], vX->nb[1], vD * vX->nb[0]));
        vX = ggml_mul(vCtx0, ggml_view_2d(vCtx0, vX, vD, vX->ne[1], vX->nb[1], 0), vGate);
        vX = ggml_cont(vCtx0, ggml_transpose(vCtx0, vX));
      }

      // use ggml_ssm_conv for f32 precision
      vX = ggml_pad(vCtx0, vX, 4, 0, 0, 0);
      vX = ggml_roll(vCtx0, vX, 4, 0, 0, 0);
      vX = ggml_pad(vCtx0, vX, 4, 0, 0, 0);
      vX = ggml_ssm_conv(vCtx0, vX, vLayer.vConvDwW);
      vX = ggml_add(vCtx0, vX, vLayer.vConvDwB);

      vX = ggml_add(vCtx0, ggml_mul(vCtx0, vX, vLayer.vConvNormW), vLayer.vConvNormB);
      vX = ggml_silu(vCtx0, vX);

      // pointwise_conv2
      vX = mBuildMm(vLayer.vConvPw2W, vX);
      vX = ggml_add(vCtx0, vX, vLayer.vConvPw2B);

      vCur = vX;
    }

    vResidual = ggml_add(vCtx0, vResidual, vCur);

    vCur = mBuildNorm(vResidual, vLayer.vFfNorm1W, vLayer.vFfNorm1B, cNormTypeNormal, 1e-5, vIl);
    mCb(vCur, "conformer.layers.{}.norm_feed_forward2", vIl);

    vCur = mBuildFfn(vCur, vLayer.vFfUp1W, vLayer.vFfUp1B, nullptr, nullptr, vLayer.vFfDown1W, vLayer.vFfDown1B,
                     cFfnSilu, vIl); // TODO(tarek): read activation for ffn from hparams
    mCb(vCur, "conformer.layers.{}.feed_forward2.linear2", vIl);

    vResidual = ggml_add(vCtx0, vResidual, ggml_scale(vCtx0, vCur, cFcFactor));
    mCb(vResidual, "conformer.layers.{}.conv.id", vIl);

    vCur = mBuildNorm(vResidual, vLayer.vLn2W, vLayer.vLn2B, cNormTypeNormal, 1e-5, vIl);
    mCb(vCur, "conformer.layers.{}.norm_out", vIl);
  }

  // audio adapter
  vCur = mBuildNorm(vCur, vModel.vMm0W, vModel.vMm0B, cNormTypeNormal, 1e-5, -1);
  mCb(vCur, "audio_adapter.model.{}", 0);
  vCur = mBuildFfn(vCur, vModel.vMm1W, vModel.vMm1B, nullptr, nullptr, vModel.vMm3W, vModel.vMm3B, cFfnGeluErf, -1);

  mCb(vCur, "projected", -1);

  ggml_build_forward_expand(vGf, vCur);

  return vGf;
}
