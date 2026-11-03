#include "models.h"

ggml_cgraph *clip_graph_whisper_enc::mBuild() {
  const int cNFrames = mGetImg(0).mNx();
  const int cNPos = cNFrames / 2;
  GGML_ASSERT(vModel.vPositionEmbeddings->ne[1] >= cNPos);

  ggml_tensor *vInp = mBuildInpRaw(1);

  // conv1d block
  {
    // convolution + gelu
    ggml_tensor *vCur = ggml_conv_1d_ph(vCtx0, vModel.vConv1d1W, vInp, 1, 1);
    vCur = ggml_add(vCtx0, vCur, vModel.vConv1d1B);

    vCur = ggml_gelu_erf(vCtx0, vCur);

    vCur = ggml_conv_1d_ph(vCtx0, vModel.vConv1d2W, vCur, 2, 1);
    vCur = ggml_add(vCtx0, vCur, vModel.vConv1d2B);

    vCur = ggml_gelu_erf(vCtx0, vCur);
    // transpose
    vInp = ggml_cont(vCtx0, ggml_transpose(vCtx0, vCur));
    mCb(vInp, "after_conv1d", -1);
  }

  // sanity check (only check one layer, but it should be the same for all)
  GGML_ASSERT(vModel.lLayers[0].vLn1W && vModel.lLayers[0].vLn1B);
  GGML_ASSERT(vModel.lLayers[0].vLn2W && vModel.lLayers[0].vLn2B);
  GGML_ASSERT(vModel.lLayers[0].vQB);
  GGML_ASSERT(vModel.lLayers[0].vVB);
  GGML_ASSERT(!vModel.lLayers[0].vKB); // no bias for k

  ggml_tensor *vPosEmbdSelected = ggml_view_2d(vCtx0, vModel.vPositionEmbeddings, vModel.vPositionEmbeddings->ne[0],
                                               cNPos, vModel.vPositionEmbeddings->nb[1], 0);
  ggml_tensor *vCur = mBuildVit(vInp, cNPos, cNormTypeNormal, vHparams.vFfnOp, vPosEmbdSelected, nullptr);

  mCb(vCur, "after_transformer", -1);

  if (vModel.mAudioHasStackFrames()) {
    // StackAudioFrames
    // https://huggingface.co/fixie-ai/ultravox-v0_5-llama-3_2-1b/blob/main/ultravox_model.py
    vCur = mBuildStack(vCur, vHparams.vProjStackFactor, cNEmbd);
    mCb(vCur, "after_stacked", -1);
  }

  if (vProjType == cProjectorTypeUltravox) {
    // UltravoxProjector
    // pre-norm
    vCur = ggml_rms_norm(vCtx0, vCur, 1e-6);
    vCur = ggml_mul(vCtx0, vCur, vModel.vMmNormPreW);

    // ffn in
    vCur = mBuildMm(vModel.vMm1W, vCur);

    // swiglu
    // see SwiGLU in ultravox_model.py, the second half passed through is silu, not the first half
    vCur = ggml_swiglu_swapped(vCtx0, vCur);

    // mid-norm
    vCur = ggml_rms_norm(vCtx0, vCur, 1e-6);
    vCur = ggml_mul(vCtx0, vCur, vModel.vMmNormMidW);

    // ffn out
    vCur = mBuildMm(vModel.vMm2W, vCur);

  } else if (vProjType == cProjectorTypeQwen2a) {
    // projector
    vCur = mBuildMm(vModel.vMmFcW, vCur);
    vCur = ggml_add(vCtx0, vCur, vModel.vMmFcB);

  } else if (vProjType == cProjectorTypeVoxtral) {
    // projector
    vCur = mBuildFfn(vCur, vModel.vMm1W, vModel.vMm1B, nullptr, nullptr, vModel.vMm2W, vModel.vMm2B, cFfnGeluErf, -1);

  } else if (vProjType == cProjectorTypeMusicFlamingo) {
    // projector
    vCur = mBuildFfn(vCur, vModel.vMm1W, vModel.vMm1B, nullptr, nullptr, vModel.vMm2W, vModel.vMm2B, cFfnGeluErf, -1);

  } else if (vProjType == cProjectorTypeMeralion) {
    // stack (above) -> ln -> linear0+silu -> GLU -> out
    vCur = ggml_norm(vCtx0, vCur, vHparams.vEps);
    vCur = ggml_mul(vCtx0, vCur, vModel.vMmNormPreW);
    vCur = ggml_add(vCtx0, vCur, vModel.vMmNormPreB);

    vCur = ggml_mul_mat(vCtx0, vModel.vMm0W, vCur);
    vCur = ggml_add(vCtx0, vCur, vModel.vMm0B);
    vCur = ggml_silu(vCtx0, vCur);

    ggml_tensor *vGate = ggml_mul_mat(vCtx0, vModel.vMm1W, vCur);
    vGate = ggml_add(vCtx0, vGate, vModel.vMm1B);
    vGate = ggml_silu(vCtx0, vGate);

    ggml_tensor *vPool = ggml_mul_mat(vCtx0, vModel.vMm2W, vCur);
    vPool = ggml_add(vCtx0, vPool, vModel.vMm2B);

    vCur = ggml_mul(vCtx0, vGate, vPool);

    vCur = ggml_mul_mat(vCtx0, vModel.vMm3W, vCur);
    vCur = ggml_add(vCtx0, vCur, vModel.vMm3B);

  } else if (vProjType == cProjectorTypeGlma) {
    vCur = ggml_norm(vCtx0, vCur, vHparams.vEps);
    vCur = ggml_mul(vCtx0, vCur, vModel.vMmNormPreW);
    vCur = ggml_add(vCtx0, vCur, vModel.vMmNormPreB);
    vCur = mBuildStack(vCur, vHparams.vProjStackFactor, cNEmbd);
    vCur =
        mBuildFfn(vCur, vModel.vMm1W, vModel.vMm1B, nullptr, nullptr, vModel.vMm2W, vModel.vMm2B, vHparams.vFfnOp, 0);
    vCur = ggml_concat(vCtx0, vModel.vMmBoi, vCur, 1);
    vCur = ggml_concat(vCtx0, vCur, vModel.vMmEoi, 1);
  } else {
    GGML_ABORT("%s: unknown projector type", __func__);
  }

  mCb(vCur, "projected", -1);

  ggml_build_forward_expand(vGf, vCur);

  return vGf;
}
