#include "models.h"

static constexpr int cParakeetLocalAttnThreshold = 8192;
static constexpr int cParakeetLocalAttnWindow = 128;

// conv subsampling + conformer encoder
ggml_cgraph *clip_graph_parakeet::mBuild() {

  // Conv subsampling
  ggml_tensor *vInp = mBuildInpRaw(1);
  vInp = ggml_cont(vCtx0, ggml_transpose(vCtx0, vInp));

  // [freq, time, channels, batch]
  ggml_tensor *vCur = ggml_conv_2d(vCtx0, vModel.aPreEncodeConvXW[0], vInp, 2, 2, 1, 1, 1, 1);
  vCur = ggml_add(vCtx0, vCur, vModel.aPreEncodeConvXB[0]);
  mCb(vCur, "pre_conv_0", -1);

  vCur = ggml_relu(vCtx0, vCur);
  mCb(vCur, "pre_conv_0_relu", -1);

  // [freq, time, channels, batch]
  vCur = ggml_conv_2d_dw_direct(vCtx0, vModel.aPreEncodeConvXW[2], vCur, 2, 2, 1, 1, 1, 1);
  vCur = ggml_add(vCtx0, vCur, vModel.aPreEncodeConvXB[2]);
  mCb(vCur, "pre_conv_2", -1);

  // [freq, time, channels, batch]
  vCur = ggml_conv_2d(vCtx0, vModel.aPreEncodeConvXW[3], vCur, 1, 1, 0, 0, 1, 1);
  vCur = ggml_add(vCtx0, vCur, vModel.aPreEncodeConvXB[3]);
  mCb(vCur, "pre_conv_3", -1);

  vCur = ggml_relu(vCtx0, vCur);
  mCb(vCur, "pre_conv_3_relu", -1);

  // [freq, time, channels, batch]
  vCur = ggml_conv_2d_dw_direct(vCtx0, vModel.aPreEncodeConvXW[5], vCur, 2, 2, 1, 1, 1, 1);
  mCb(vCur, "pre_conv_5_direct", -1);
  vCur = ggml_add(vCtx0, vCur, vModel.aPreEncodeConvXB[5]);
  mCb(vCur, "pre_conv_5", -1);

  // [freq, time, channels, batch]
  vCur = ggml_conv_2d(vCtx0, vModel.aPreEncodeConvXW[6], vCur, 1, 1, 0, 0, 1, 1);
  vCur = ggml_add(vCtx0, vCur, vModel.aPreEncodeConvXB[6]);
  mCb(vCur, "pre_conv_6", -1);

  vCur = ggml_relu(vCtx0, vCur);
  mCb(vCur, "pre_conv_6_relu", -1);

  // [freq, time, chan]
  vCur = ggml_permute(vCtx0, vCur, 0, 2, 1, 3);
  // [freq, chan, time]
  vCur = ggml_cont(vCtx0, vCur);

  const int cNFreq = vCur->ne[0];
  const int cNChan = vCur->ne[1];
  const int cNFrames = vCur->ne[2];

  // [freq, time, chan, batch] -> [(freq * chan), time]
  vCur = ggml_reshape_2d(vCtx0, vCur, cNFreq * cNChan, cNFrames);

  vCur = mBuildMm(vModel.vPreEncodeOutW, vCur);
  vCur = ggml_add(vCtx0, vCur, vModel.vPreEncodeOutB);

  ggml_set_name(vCur, "pre_enc_out");

  // Encoder

  const auto &vHparams = vModel.vHparams;
  const int cNLayer = vHparams.vNLayer;
  const int cNState = vHparams.vNEmbd;
  const float cFcFactor = 0.5f;

  const int cNTime = vCur->ne[1];
  const bool cLocalAttn = cNTime > cParakeetLocalAttnThreshold;
  const int cAttLeft = cLocalAttn ? cParakeetLocalAttnWindow : cNTime - 1;
  const int cAttRight = cLocalAttn ? cParakeetLocalAttnWindow : cNTime - 1;
  const int cWindowSize = cLocalAttn ? cAttLeft + cAttRight + 1 : 2 * cNTime - 1;
  const int cDHalf = cNState / 2;
  const int cMaskDim = cLocalAttn ? cWindowSize : cNTime;

  // mask [key, n_time]
  struct ggml_tensor *vAttnMask = ggml_new_tensor_2d(vCtx0, GGML_TYPE_F32, cMaskDim, cNTime);
  ggml_set_name(vAttnMask, "attn_mask");
  ggml_set_input(vAttnMask);

  struct ggml_tensor *vLocalMask = nullptr;
  if (cLocalAttn) {
    const int cChunk = cAttLeft + cAttRight;
    vLocalMask = ggml_new_tensor_2d(vCtx0, GGML_TYPE_F32, cChunk + cWindowSize - 1, cChunk);
    ggml_set_name(vLocalMask, "local_mask");
    ggml_set_input(vLocalMask);
  }

  struct ggml_tensor *vPosFreqs = ggml_new_tensor_1d(vCtx0, GGML_TYPE_F32, cDHalf);
  ggml_set_name(vPosFreqs, "pos_freqs");
  ggml_set_input(vPosFreqs);

  struct ggml_tensor *vRelPositions = ggml_new_tensor_2d(vCtx0, GGML_TYPE_F32, 1, cWindowSize);
  ggml_set_name(vRelPositions, "rel_positions");
  ggml_set_input(vRelPositions);

  struct ggml_tensor *vFreqs = ggml_repeat_4d(vCtx0, vPosFreqs, cDHalf, cWindowSize, 1, 1);
  struct ggml_tensor *vTheta = ggml_mul(vCtx0, vFreqs, vRelPositions);

  struct ggml_tensor *vSin = ggml_reshape_3d(vCtx0, ggml_sin(vCtx0, vTheta), 1, cDHalf, cWindowSize);
  struct ggml_tensor *vCos = ggml_reshape_3d(vCtx0, ggml_cos(vCtx0, vTheta), 1, cDHalf, cWindowSize);
  struct ggml_tensor *vPosEmb =
      ggml_reshape_2d(vCtx0, ggml_cont(vCtx0, ggml_concat(vCtx0, vSin, vCos, 0)), cNState, cWindowSize);
  ggml_set_name(vPosEmb, "pos_emb");

  for (int vIl = 0; vIl < cNLayer; ++vIl) {
    const auto &vLayer = vModel.lLayers[vIl];
    // FFN1
    {
      struct ggml_tensor *vResidual = vCur;
      ggml_format_name(vCur, "enc_%d_res", vIl);

      // norm
      vCur = ggml_norm(vCtx0, vCur, vHparams.vEps);
      vCur = ggml_add(vCtx0, ggml_mul(vCtx0, vCur, vLayer.vFfNormW), vLayer.vFfNormB);
      ggml_format_name(vCur, "enc_%d_ffn_norm_1", vIl);

      vCur = mBuildFfn(vCur, vLayer.vFfUpW, nullptr, nullptr, nullptr, vLayer.vFfDownW, nullptr, cFfnSilu, vIl);
      ggml_format_name(vCur, "enc_%d_ffn_1", vIl);

      vCur = ggml_add(vCtx0, vResidual, ggml_scale(vCtx0, vCur, cFcFactor));
      ggml_format_name(vCur, "enc_%d_res_ffn", vIl);
    }

    // self attention block using relative positional encoding from model.position_embedding.
    {
      // [feat, time_frames, 1, 1]
      struct ggml_tensor *vResidual = vCur;

      vCur = ggml_norm(vCtx0, vCur, vHparams.vEps);
      vCur = ggml_add(vCtx0, ggml_mul(vCtx0, vCur, vLayer.vLn1W), vLayer.vLn1B);
      ggml_format_name(vCur, "enc_%d_attn_norm", vIl);

      const int cNHead = vHparams.vNHead;
      const int cDHead = cNState / cNHead;

      // [feat, time_frames, 1, 1]
      struct ggml_tensor *vQCur = mBuildMm(vLayer.vQW, vCur);
      struct ggml_tensor *vKCur = mBuildMm(vLayer.vKW, vCur);
      struct ggml_tensor *vVCur = mBuildMm(vLayer.vVW, vCur);

      // [d_head, n_heads, n_time, 1]
      vQCur = ggml_reshape_3d(vCtx0, vQCur, cDHead, cNHead, cNTime);
      vKCur = ggml_reshape_3d(vCtx0, vKCur, cDHead, cNHead, cNTime);
      vVCur = ggml_reshape_3d(vCtx0, vVCur, cDHead, cNHead, cNTime);

      // [n_state, window_size]
      struct ggml_tensor *vPos = mBuildMm(vLayer.vLinearPosW, vPosEmb);
      // [feat, head, window_size, 1]
      vPos = ggml_reshape_3d(vCtx0, vPos, cDHead, cNHead, vPosEmb->ne[1]);
      // [feat, window_size, head, 1]
      vPos = ggml_cont(vCtx0, ggml_permute(vCtx0, vPos, 0, 2, 1, 3));
      ggml_format_name(vPos, "enc_%d_attn_pos", vIl);

      if (cLocalAttn) {
        const int cChunk = cAttLeft + cAttRight;
        const int cNGroup = (cNTime + cChunk - 1) / cChunk;
        const int cNTimePadded = cNGroup * cChunk;
        const int cNKvChunk = cChunk + cWindowSize - 1;
        const int cNKvDense = cNKvChunk * cNGroup;
        const bool cNeedPadding = cNTimePadded > cNTime;

        vQCur = ggml_cont(vCtx0, ggml_permute(vCtx0, vQCur, 0, 2, 1, 3));
        vKCur = ggml_cont(vCtx0, ggml_permute(vCtx0, vKCur, 0, 2, 1, 3));
        vVCur = ggml_cont(vCtx0, ggml_permute(vCtx0, vVCur, 0, 2, 1, 3));

        // content bias
        struct ggml_tensor *vBiasU = ggml_reshape_3d(vCtx0, vLayer.vPosBiasU, cDHead, 1, cNHead);
        struct ggml_tensor *vQU = ggml_add(vCtx0, vQCur, vBiasU);

        // position bias
        struct ggml_tensor *vBiasV = ggml_reshape_3d(vCtx0, vLayer.vPosBiasV, cDHead, 1, cNHead);
        struct ggml_tensor *vQV = ggml_add(vCtx0, vQCur, vBiasV);

        // right pad the time dimension
        struct ggml_tensor *vQUPadded =
            cNeedPadding ? ggml_pad_ext(vCtx0, vQU, 0, 0, 0, cNTimePadded - cNTime, 0, 0, 0, 0) : vQU;
        vQUPadded = ggml_reshape_4d(vCtx0, vQUPadded, cDHead, cChunk, cNGroup, cNHead);

        // pad front and back for the first and last time frames
        struct ggml_tensor *vKPadded = ggml_pad_ext(vCtx0, vKCur, 0, 0, cAttLeft, cAttRight, 0, 0, 0, 0);
        if (cNKvDense > vKPadded->ne[1]) {
          vKPadded = ggml_pad_ext(vCtx0, vKPadded, 0, 0, 0, cNKvDense - vKPadded->ne[1], 0, 0, 0, 0);
        }

        // sliding window view: each group spans n_kv_chunk keys but steps by chunk
        struct ggml_tensor *vKChunk = ggml_view_4d(vCtx0, vKPadded, cDHead, cNKvChunk, cNGroup, cNHead, vKPadded->nb[1],
                                                   (size_t)cChunk * vKPadded->nb[1], vKPadded->nb[2], 0);
        vKChunk = ggml_cont(vCtx0, vKChunk);

        struct ggml_tensor *vContentScores = ggml_mul_mat(vCtx0, vKChunk, vQUPadded);

        // trim the dense output down to window_size scores per query
        vContentScores = ggml_view_4d(vCtx0, vContentScores, cWindowSize, cChunk, cNGroup, cNHead,
                                      (size_t)(cChunk + cWindowSize) * vContentScores->nb[0], vContentScores->nb[2],
                                      vContentScores->nb[3], 0);
        vContentScores = ggml_cont(vCtx0, vContentScores);

        // ungroup: [window_size, n_time_padded, n_head]
        vContentScores = ggml_reshape_3d(vCtx0, vContentScores, cWindowSize, cNTimePadded, cNHead);
        if (cNeedPadding) {
          vContentScores = ggml_view_3d(vCtx0, vContentScores, cWindowSize, cNTime, cNHead, vContentScores->nb[1],
                                        vContentScores->nb[2], 0);
        }

        // Q_v: [d_head, time, head]
        vQV = ggml_cont(vCtx0, ggml_permute(vCtx0, vQV, 0, 2, 1, 3));
        struct ggml_tensor *vRelPosScores = ggml_mul_mat(vCtx0, vPos, vQV);

        struct ggml_tensor *vAttnScores = ggml_add(vCtx0, vContentScores, vRelPosScores);
        vAttnScores = ggml_soft_max_ext(vCtx0, vAttnScores, vAttnMask, 1.0f / std::sqrt(cDHead), 0.0f);
        ggml_format_name(vAttnScores, "enc_%d_attn_probs", vIl);

        // expand probs back to n_kv_chunk width for the V matmul
        struct ggml_tensor *vProbsPadded =
            cNeedPadding ? ggml_pad_ext(vCtx0, vAttnScores, 0, 0, 0, cNTimePadded - cNTime, 0, 0, 0, 0) : vAttnScores;

        vProbsPadded = ggml_reshape_4d(vCtx0, vProbsPadded, cWindowSize, cChunk, cNGroup, cNHead);
        vProbsPadded = ggml_pad_ext(vCtx0, vProbsPadded, 0, cChunk, 0, 0, 0, 0, 0, 0);
        vProbsPadded =
            ggml_view_4d(vCtx0, vProbsPadded, cNKvChunk, cChunk, cNGroup, cNHead,
                         (size_t)cNKvChunk * vProbsPadded->nb[0], vProbsPadded->nb[2], vProbsPadded->nb[3], 0);
        vProbsPadded = ggml_cont(vCtx0, vProbsPadded);
        vProbsPadded = ggml_mul(vCtx0, vProbsPadded, vLocalMask);

        struct ggml_tensor *vVPadded = ggml_pad_ext(vCtx0, vVCur, 0, 0, cAttLeft, cAttRight, 0, 0, 0, 0);
        if (cNKvDense > vVPadded->ne[1]) {
          vVPadded = ggml_pad_ext(vCtx0, vVPadded, 0, 0, 0, cNKvDense - vVPadded->ne[1], 0, 0, 0, 0);
        }
        vVPadded = ggml_cont(vCtx0, ggml_transpose(vCtx0, vVPadded));

        struct ggml_tensor *vVChunk = ggml_view_4d(vCtx0, vVPadded, cNKvChunk, cDHead, cNGroup, cNHead, vVPadded->nb[1],
                                                   (size_t)cChunk * vVPadded->nb[0], vVPadded->nb[2], 0);
        vVChunk = ggml_cont(vCtx0, vVChunk);

        vCur = ggml_mul_mat(vCtx0, vVChunk, vProbsPadded);
        vCur = ggml_reshape_3d(vCtx0, vCur, cDHead, cNTimePadded, cNHead);
        if (cNeedPadding) {
          vCur = ggml_view_3d(vCtx0, vCur, cDHead, cNTime, cNHead, vCur->nb[1], vCur->nb[2], 0);
        }
        vCur = ggml_cont(vCtx0, ggml_permute(vCtx0, vCur, 0, 2, 1, 3));
        vCur = ggml_reshape_2d(vCtx0, vCur, cNState, cNTime);
        vCur = mBuildMm(vLayer.vOW, vCur);
      } else {
        // full attention
        struct ggml_tensor *vQU = ggml_add(vCtx0, vQCur, vLayer.vPosBiasU);
        ggml_format_name(vQU, "enc_%d_attn_q_u", vIl);

        struct ggml_tensor *vKPrep = ggml_permute(vCtx0, vKCur, 0, 2, 1, 3);
        struct ggml_tensor *vQPrep = ggml_permute(vCtx0, vQU, 0, 2, 1, 3);
        struct ggml_tensor *vContentScores = ggml_mul_mat(vCtx0, vKPrep, vQPrep);
        ggml_format_name(vContentScores, "enc_%d_attn_content_scores", vIl);

        struct ggml_tensor *vQV = ggml_add(vCtx0, vQCur, vLayer.vPosBiasV);
        ggml_format_name(vQV, "enc_%d_attn_q_v", vIl);

        vQV = ggml_permute(vCtx0, vQV, 0, 2, 1, 3);
        vQV = ggml_cont(vCtx0, vQV);
        ggml_format_name(vQV, "enc_%d_attn_q_v_perm", vIl);

        struct ggml_tensor *vRelPosScores = ggml_mul_mat(vCtx0, vPos, vQV);
        ggml_format_name(vRelPosScores, "enc_%d_attn_rel_pos", vIl);

        // Relative positional shift
        {
          const auto cPosWindow = vRelPosScores->ne[0];
          const auto cNFrame = vRelPosScores->ne[1];
          const auto cNHead = vRelPosScores->ne[2];

          vRelPosScores = ggml_pad(vCtx0, vRelPosScores, 1, 0, 0, 0);
          vRelPosScores = ggml_roll(vCtx0, vRelPosScores, 1, 0, 0, 0);

          vRelPosScores = ggml_reshape_3d(vCtx0, vRelPosScores, cNFrame, cPosWindow + 1, cNHead);
          vRelPosScores = ggml_cont(vCtx0, vRelPosScores);
          ggml_format_name(vRelPosScores, "enc_%d_attn_rel_pos_reshaped", vIl);

          int vCenter = cPosWindow / 2;
          size_t vOffset = vRelPosScores->nb[0] * (vCenter + 1);

          vRelPosScores = ggml_view_3d(vCtx0, vRelPosScores, cNFrame, cPosWindow, cNHead, (cPosWindow) * 4,
                                       vRelPosScores->nb[2], vOffset);
          vRelPosScores = ggml_cont(vCtx0, vRelPosScores);
          ggml_format_name(vRelPosScores, "enc_%d_attn_rel_pos_shifted", vIl);

          vRelPosScores = ggml_view_3d(vCtx0, vRelPosScores, vContentScores->ne[0], vContentScores->ne[1],
                                       vRelPosScores->ne[2], vRelPosScores->nb[1], vRelPosScores->nb[2], 0);
          vRelPosScores = ggml_cont(vCtx0, vRelPosScores);
          ggml_format_name(vRelPosScores, "enc_%d_attn_rel_pos_shifted_view", vIl);
        }

        struct ggml_tensor *vAttnScores = ggml_add(vCtx0, vContentScores, vRelPosScores);
        ggml_format_name(vAttnScores, "enc_%d_attn_scores", vIl);
        vAttnScores = ggml_scale(vCtx0, vAttnScores, 1.0f / std::sqrt(cDHead));
        vAttnScores = ggml_add(vCtx0, vAttnScores, vAttnMask);
        ggml_format_name(vAttnScores, "enc_%d_attn_scores_scaled", vIl);

        struct ggml_tensor *vProbs = ggml_soft_max(vCtx0, vAttnScores);
        ggml_format_name(vProbs, "enc_%d_attn_probs", vIl);

        vVCur = ggml_cont(vCtx0, ggml_permute(vCtx0, vVCur, 1, 2, 0, 3));
        ggml_format_name(vVCur, "enc_%d_attn_v_cur", vIl);
        vCur = ggml_mul_mat(vCtx0, vProbs, vVCur);
        ggml_format_name(vCur, "enc_%d_attn_inp", vIl);

        vCur = ggml_permute(vCtx0, vCur, 2, 0, 1, 3);
        vCur = ggml_cont_2d(vCtx0, vCur, cNState, cNTime);
        vCur = mBuildMm(vLayer.vOW, vCur);
      }
      ggml_format_name(vCur, "enc_%d_attn_out", vIl);

      vCur = ggml_add(vCtx0, vResidual, vCur);
      ggml_format_name(vCur, "enc_%d_attn_res", vIl);
    }

    // Convolution
    {
      struct ggml_tensor *vResidual = vCur;
      ggml_format_name(vCur, "enc_%d_residual_conv", vIl);

      vCur = ggml_norm(vCtx0, vCur, vHparams.vEps);
      vCur = ggml_add(vCtx0, ggml_mul(vCtx0, vCur, vLayer.vNormConvW), vLayer.vNormConvB);
      ggml_format_name(vCur, "enc_%d_norm_conv", vIl);

      // pointwise 1d convolution:
      vCur = mBuildMm(vLayer.vConvPw1W, vCur);
      ggml_format_name(vCur, "enc_%d_conv_pw1", vIl);

      {
        int64_t vD = vCur->ne[0] / 2;
        struct ggml_tensor *vSignal = ggml_view_2d(vCtx0, vCur, vD, vCur->ne[1], vCur->nb[1], 0);
        struct ggml_tensor *vGate = ggml_view_2d(vCtx0, vCur, vD, vCur->ne[1], vCur->nb[1], vD * vCur->nb[0]);

        vCur = ggml_mul(vCtx0, vSignal, ggml_sigmoid(vCtx0, vGate));
        ggml_format_name(vCur, "enc_%d_conv_glu", vIl);
      }

      vCur = ggml_cont(vCtx0, ggml_transpose(vCtx0, vCur));

      // use ggml_ssm_conv for f32 precision
      const int cDwPad = (vHparams.vAudioConvKernelSize - 1) / 2;
      vCur = ggml_pad(vCtx0, vCur, cDwPad, 0, 0, 0);
      vCur = ggml_roll(vCtx0, vCur, cDwPad, 0, 0, 0);
      vCur = ggml_pad(vCtx0, vCur, cDwPad, 0, 0, 0);
      ggml_format_name(vCur, "enc_%d_conv_dw_pad", vIl);

      vCur = ggml_ssm_conv(vCtx0, vCur, vLayer.vConvDwW);
      ggml_format_name(vCur, "enc_%d_conv_1d_dw", vIl);

      vCur = ggml_sub(vCtx0, vCur, vLayer.vConvNormMean);
      struct ggml_tensor *vStd = ggml_sqrt(vCtx0, vLayer.vConvNormVar);
      vCur = ggml_div(vCtx0, vCur, vStd);
      vCur = ggml_add(vCtx0, ggml_mul(vCtx0, vCur, vLayer.vConvNormW), vLayer.vConvNormB);
      ggml_format_name(vCur, "enc_%d_conv_bn", vIl);

      vCur = ggml_silu(vCtx0, vCur);
      ggml_format_name(vCur, "enc_%d_conv_silu", vIl);

      vCur = mBuildMm(vLayer.vConvPw2W, vCur);
      ggml_format_name(vCur, "enc_%d_conv_pw2", vIl);

      vCur = ggml_add(vCtx0, vResidual, vCur);
      ggml_format_name(vCur, "enc_%d_conv_res", vIl);
    }

    // FFN2
    {
      struct ggml_tensor *vResidual = vCur;
      vCur = ggml_norm(vCtx0, vCur, vHparams.vEps);
      vCur = ggml_add(vCtx0, ggml_mul(vCtx0, vCur, vLayer.vFfNorm1W), vLayer.vFfNorm1B);
      ggml_format_name(vCur, "enc_%d_ffn_norm_2", vIl);

      vCur = mBuildFfn(vCur, vLayer.vFfUp1W, nullptr, nullptr, nullptr, vLayer.vFfDown1W, nullptr, cFfnSilu, vIl);
      vCur = ggml_add(vCtx0, vResidual, ggml_scale(vCtx0, vCur, 0.5));
      ggml_format_name(vCur, "enc_%d_ffn_res", vIl);
    }

    vCur = ggml_norm(vCtx0, vCur, vHparams.vEps);
    vCur = ggml_add(vCtx0, ggml_mul(vCtx0, vCur, vLayer.vLn2W), vLayer.vLn2B);
  }

  mCb(vCur, "encoder_out", -1);

  vCur = ggml_rms_norm(vCtx0, vCur, 1e-6);
  vCur = ggml_mul(vCtx0, vCur, vModel.vMmNormPreW);
  mCb(vCur, "sound_projection.norm", -1);

  vCur = mBuildFfn(vCur, vModel.vMm0W, vModel.vMm0B, nullptr, nullptr, vModel.vMm1W, vModel.vMm1B, cFfnReluSqr, -1);
  mCb(vCur, "projected", -1);

  ggml_build_forward_expand(vGf, vCur);

  return vGf;
}
