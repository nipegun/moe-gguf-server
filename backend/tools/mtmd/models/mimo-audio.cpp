#include "models.h"

ggml_cgraph *clip_graph_mimo_audio::mBuild() {
  ggml_tensor *vInp = mBuildInpRaw(1); // [n_frames, n_mel, 1]

  ggml_tensor *vCur = ggml_conv_1d_ph(vCtx0, vModel.vConv1d1W, vInp, 1, 1);
  vCur = ggml_add(vCtx0, vCur, vModel.vConv1d1B);
  vCur = ggml_gelu_erf(vCtx0, vCur);

  vCur = ggml_conv_1d_ph(vCtx0, vModel.vConv1d2W, vCur, 2, 1);
  vCur = ggml_add(vCtx0, vCur, vModel.vConv1d2B);
  vCur = ggml_gelu_erf(vCtx0, vCur);

  ggml_tensor *vInpL = ggml_cont(vCtx0, ggml_transpose(vCtx0, vCur)); // [n_embd, n_pos]
  const int64_t cNPos = vInpL->ne[1];
  mCb(vInpL, "after_conv1d", -1);

  GGML_ASSERT((int)vHparams.lWaPatternMode.size() == cNLayer);

  ggml_tensor *vInpPos = ggml_new_tensor_1d(vCtx0, GGML_TYPE_I32, cNPos);
  ggml_set_name(vInpPos, "mimo_audio_positions");
  ggml_set_input(vInpPos);

  ggml_tensor *vFullMask = ggml_new_tensor_2d(vCtx0, GGML_TYPE_F32, cNPos, cNPos);
  ggml_set_name(vFullMask, "mimo_audio_full_mask");
  ggml_set_input(vFullMask);

  ggml_tensor *vWindowMask = ggml_new_tensor_2d(vCtx0, GGML_TYPE_F32, cNPos, cNPos);
  ggml_set_name(vWindowMask, "mimo_audio_window_mask");
  ggml_set_input(vWindowMask);

  build_vit_opts vOpts;
  vOpts.lAttnMaskLayers.resize(cNLayer);
  for (int vIl = 0; vIl < cNLayer; vIl++) {
    vOpts.lAttnMaskLayers[vIl] = vHparams.lWaPatternMode[vIl] == -1 ? vFullMask : vWindowMask;
  }
  // the skip connection below must be added before the post-transformer  norm,
  // so build_vit must not apply that norm itself
  vOpts.vSkipPostLn = true;

  // encoder_skip_layer_id=3 (1-indexed) -> capture output of layer index 2
  const int cSkipCaptureIl = 2;
  GGML_ASSERT(cNLayer > cSkipCaptureIl);
  ggml_tensor *vSkipHidden = nullptr;
  vOpts.vCallbackLayerOut = [&](ggml_tensor *pLayerCur, int pIl) {
    if (pIl == cSkipCaptureIl) {
      vSkipHidden = pLayerCur;
    }
  };

  auto vAddPos = [&](ggml_tensor *pX, const clip_layer &) {
    return ggml_rope_ext(vCtx0, pX, vInpPos, nullptr, cDHead, GGML_ROPE_TYPE_NEOX, 0, vHparams.vRopeTheta, 1.0f, 0.0f,
                         1.0f, 0.0f, 0.0f);
  };

  vInpL = mBuildVit(vInpL, cNPos, cNormTypeNormal, vHparams.vFfnOp, nullptr, vAddPos, vOpts);
  vInpL = ggml_reshape_2d(vCtx0, vInpL, cNEmbd, cNPos); // build_vit restores a (size-1) batch dim

  GGML_ASSERT(vSkipHidden != nullptr);
  vInpL = ggml_add(vCtx0, vInpL, vSkipHidden);

  vInpL = mBuildNorm(vInpL, vModel.vPostLnW, vModel.vPostLnB, cNormTypeNormal, cEps, -1);
  mCb(vInpL, "after_transformer", -1);

  // downsample: strided conv (no bias) + gelu + layernorm
  {
    ggml_tensor *vDs = ggml_cont(vCtx0, ggml_transpose(vCtx0, vInpL)); // [n_pos, n_embd]
    vDs = ggml_conv_1d(vCtx0, vModel.vDownsampleConvW, vDs, 2, 0, 1);
    vDs = ggml_gelu_erf(vCtx0, vDs);
    vDs = ggml_cont(vCtx0, ggml_transpose(vCtx0, vDs)); // [n_embd, n_pos/2]
    vDs = mBuildNorm(vDs, vModel.vDownsampleNormW, vModel.vDownsampleNormB, cNormTypeNormal, cEps, -1);
    vInpL = vDs;
  }
  mCb(vInpL, "after_downsample", -1);

  // RVQ quantize: codebook ne=[dim, max_bins, n_q]
  // quantize input vector to codes (type=I32)
  std::vector<ggml_tensor *> lCodes;
  {
    GGML_ASSERT(vModel.vRvqCodebook != nullptr);
    const int64_t cDim = vModel.vRvqCodebook->ne[0];
    GGML_ASSERT(cDim == vInpL->ne[0]);
    GGML_ASSERT((int64_t)vHparams.lRvqCodebookSize.size() == vModel.vRvqCodebook->ne[2]);

    ggml_tensor *vResidual = vInpL; // [dim, n_pos_ds]

    for (size_t vQ = 0; vQ < vHparams.lRvqCodebookSize.size(); vQ++) {
      const int64_t cBins = vHparams.lRvqCodebookSize[vQ];
      ggml_tensor *vCodebookQ = ggml_view_2d(vCtx0, vModel.vRvqCodebook, cDim, cBins, vModel.vRvqCodebook->nb[1],
                                             vQ * vModel.vRvqCodebook->nb[2]);
      vCodebookQ = ggml_cont(vCtx0, vCodebookQ);

      ggml_tensor *vCodebookNorm = ggml_sum_rows(vCtx0, ggml_sqr(vCtx0, vCodebookQ)); // [1, bins]
      vCodebookNorm = ggml_cont(vCtx0, ggml_transpose(vCtx0, vCodebookNorm));         // [bins, 1]

      ggml_tensor *vDot = ggml_mul_mat(vCtx0, vCodebookQ, vResidual); // [bins, n_pos_ds]
      ggml_tensor *vScores = ggml_sub(vCtx0, ggml_scale(vCtx0, vDot, 2.0f), vCodebookNorm);

      ggml_tensor *vIdx = ggml_argmax(vCtx0, vScores); // [n_pos_ds]
      lCodes.push_back(vIdx);

      ggml_tensor *vQuant = ggml_get_rows(vCtx0, vCodebookQ, vIdx); // [dim, n_pos_ds]
      vResidual = ggml_sub(vCtx0, vResidual, vQuant);
      mCb(vIdx, "rvq_code", (int)vQ);
    }
  }

  // convert codes to LLM embeddings
  ggml_tensor *vCodeEmbdSum = nullptr;
  {
    GGML_ASSERT(vModel.vMmACodeEmbd != nullptr);
    const int64_t cDim = vModel.vMmACodeEmbd->ne[0];
    const int64_t cVocab = vModel.vMmACodeEmbd->ne[1];
    GGML_ASSERT((int64_t)lCodes.size() == vModel.vMmACodeEmbd->ne[2]);
    GGML_ASSERT(cDim == vInpL->ne[0]);

    for (size_t vI = 0; vI < lCodes.size(); vI++) {
      ggml_tensor *vTableI = ggml_view_2d(vCtx0, vModel.vMmACodeEmbd, cDim, cVocab, vModel.vMmACodeEmbd->nb[1],
                                          vI * vModel.vMmACodeEmbd->nb[2]);
      vTableI = ggml_cont(vCtx0, vTableI);

      ggml_tensor *vEmbdI = ggml_get_rows(vCtx0, vTableI, lCodes[vI]); // [dim, n_pos_ds]
      vCodeEmbdSum = vCodeEmbdSum ? ggml_add(vCtx0, vCodeEmbdSum, vEmbdI) : vEmbdI;
    }
    mCb(vCodeEmbdSum, "code_embd_sum", -1);
  }

  // input_local_transformer
  // groups of `cGroupSize` consecutive downsampled frames are processed together, attending only within their own
  // group. Implemented as a block-diagonal mask + in-group-repeating positions (rather than a real batch dim) - same
  // technique as the encoder's masks above.

  // note: hand-rolled here instead of mBuildVit() because this is a second, independent layer stack
  // (own layer array/count, RMSNorm instead of LN, SiLU FFN, own RoPE theta)

  ggml_tensor *vProjected;
  {
    const int cGroupSize = vHparams.vAudioLocalGroupSize;
    GGML_ASSERT(cGroupSize > 0);
    const int64_t cNPosDs = vCodeEmbdSum->ne[1];
    const int64_t cNGroups = (cNPosDs + cGroupSize - 1) / cGroupSize;
    const int64_t cNPadded = cNGroups * cGroupSize;

    ggml_tensor *vCurLocal = vCodeEmbdSum;
    if (cNPadded != cNPosDs) {
      vCurLocal = ggml_pad(vCtx0, vCurLocal, 0, (int)(cNPadded - cNPosDs), 0, 0);
    }

    ggml_tensor *vLocalPos = ggml_new_tensor_1d(vCtx0, GGML_TYPE_I32, cNPadded);
    ggml_set_name(vLocalPos, "mimo_audio_local_positions");
    ggml_set_input(vLocalPos);

    ggml_tensor *vLocalMask = ggml_new_tensor_2d(vCtx0, GGML_TYPE_F32, cNPadded, cNPadded);
    ggml_set_name(vLocalMask, "mimo_audio_local_mask");
    ggml_set_input(vLocalMask);

    const float cLocalRopeTheta = 640000.0f; // audio_config.rope_theta (differs from the encoder's)
    auto vApplyLocalRope = [&](ggml_tensor *pX) {
      return ggml_rope_ext(vCtx0, pX, vLocalPos, nullptr, cDHead, GGML_ROPE_TYPE_NEOX, 0, cLocalRopeTheta, 1.0f, 0.0f,
                           1.0f, 0.0f, 0.0f);
    };

    for (int vIl = 0; vIl < vHparams.vAudioLocalNLayer; vIl++) {
      auto &vLayer = vModel.lMmALocalLayers[vIl];

      ggml_tensor *vAttnIn = mBuildNorm(vCurLocal, vLayer.vLn1W, nullptr, cNormTypeRms, cEps, vIl);

      ggml_tensor *vQcur = mBuildMm(vLayer.vQW, vAttnIn);
      if (vLayer.vQB) {
        vQcur = ggml_add(vCtx0, vQcur, vLayer.vQB);
      }
      ggml_tensor *vKcur = mBuildMm(vLayer.vKW, vAttnIn);
      if (vLayer.vKB) {
        vKcur = ggml_add(vCtx0, vKcur, vLayer.vKB);
      }
      ggml_tensor *vVcur = mBuildMm(vLayer.vVW, vAttnIn);
      if (vLayer.vVB) {
        vVcur = ggml_add(vCtx0, vVcur, vLayer.vVB);
      }

      vQcur = ggml_reshape_3d(vCtx0, vQcur, cDHead, cNHead, cNPadded);
      vKcur = ggml_reshape_3d(vCtx0, vKcur, cDHead, cNHead, cNPadded);
      vVcur = ggml_reshape_3d(vCtx0, vVcur, cDHead, cNHead, cNPadded);

      vQcur = vApplyLocalRope(vQcur);
      vKcur = vApplyLocalRope(vKcur);

      ggml_tensor *vAttnOut = mBuildAttn(vLayer.vOW, nullptr, vQcur, vKcur, vVcur, vLocalMask, vKqScale, vIl);
      vCurLocal = ggml_add(vCtx0, vCurLocal, vAttnOut);

      ggml_tensor *vFfnIn = mBuildNorm(vCurLocal, vLayer.vLn2W, nullptr, cNormTypeRms, cEps, vIl);
      ggml_tensor *vFfnOut =
          mBuildFfn(vFfnIn, vLayer.vFfUpW, nullptr, vLayer.vFfGateW, nullptr, vLayer.vFfDownW, nullptr, cFfnSilu, vIl);
      vCurLocal = ggml_add(vCtx0, vCurLocal, vFfnOut);
    }

    vCurLocal = mBuildNorm(vCurLocal, vModel.vMmALocalNormW, nullptr, cNormTypeRms, cEps, -1);
    mCb(vCurLocal, "after_local_transformer", -1);

    // flatten each group of `cGroupSize` frames into one (cGroupSize*cNEmbd)-dim vector
    // (matching AudioProjection's flattened input)
    ggml_tensor *vGrouped = ggml_reshape_2d(vCtx0, vCurLocal, cNEmbd * cGroupSize, cNGroups);

    // AudioProjection: Linear (no bias) -> GELU -> Linear (no bias)
    vProjected = mBuildFfn(vGrouped, vModel.vMm1W, nullptr, nullptr, nullptr, vModel.vMm2W, nullptr, cFfnGeluErf, -1);
    mCb(vProjected, "after_projection", -1);
  }

  ggml_build_forward_expand(vGf, vProjected);
  return vGf;
}
