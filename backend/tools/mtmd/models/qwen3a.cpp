#include "models.h"

ggml_cgraph *clip_graph_qwen3a::mBuild() {
  // Ref implementation:
  // https://github.com/QwenLM/Qwen3-ASR/blob/main/qwen_asr/core/transformers_backend/modeling_qwen3_asr.py

  // inp_raw: [n_frames, n_mel, 1]  (nx=n_frames, ny=n_mel)
  ggml_tensor *vInp = mBuildInpRaw(1);

  const int64_t cNFrames = vInp->ne[0]; // total frames, padded to multiple of chunk_size
  const int64_t cNMel = vInp->ne[1];    // 128
  const int64_t cChunkSize = 100;       // n_window * 2 (n_window=50 from model config)
  const int64_t cNChunks = cNFrames / cChunkSize;

  GGML_ASSERT(cNFrames % cChunkSize == 0); // preprocessor should already pad the input
  GGML_ASSERT(vInp->type == GGML_TYPE_F32);

  // View mel spectrogram as batched 100-frame chunks: [chunk_size, n_mel, 1, n_chunks]
  vInp = ggml_view_4d(vCtx0, vInp, cChunkSize, cNMel, 1, cNChunks,
                      cNFrames * (int64_t)sizeof(float),   // nb[1]: stride over mel bins
                      cChunkSize * (int64_t)sizeof(float), // nb[2]: stride for C=1 (unused)
                      cChunkSize * (int64_t)sizeof(float), // nb[3]: stride over chunks
                      0);
  vInp = ggml_cont(vCtx0, vInp);
  mCb(vInp, "inp_chunks", -1);

  // 3 x conv2d + gelu
  {
    // conv output [OW, OH, C_out, n_chunks]
    auto vConvBlock = [&](ggml_tensor *pX, ggml_tensor *pW, ggml_tensor *pB) {
      pX = ggml_conv_2d(vCtx0, pW, pX, 2, 2, 1, 1, 1, 1);
      if (pB) {
        pX = ggml_add(vCtx0, pX, ggml_reshape_4d(vCtx0, pB, 1, 1, pX->ne[2], 1));
      }
      return ggml_gelu_erf(vCtx0, pX);
    };

    vInp = vConvBlock(vInp, vModel.vConv2d1W, vModel.vConv2d1B);
    vInp = vConvBlock(vInp, vModel.vConv2d2W, vModel.vConv2d2B);
    vInp = vConvBlock(vInp, vModel.vConv2d3W, vModel.vConv2d3B);
    // inp: [OW=13, OH=16, OC=480, n_chunks]
    mCb(vInp, "after_conv_blocks", -1);
  }

  // permute [OW=25, OH=16, OC=480, n_chunks] -> [OH=16, OC=480, OW=25, n_chunks]
  // reshape to [OH*OC=7680, OW*n_chunks]
  // feature index h+16*c = c*16+f (matches python code)
  vInp = ggml_cont(vCtx0, ggml_permute(vCtx0, vInp, 2, 0, 1, 3));
  vInp = ggml_reshape_2d(vCtx0, vInp, vInp->ne[0] * vInp->ne[1], vInp->ne[2] * vInp->ne[3]);

  // Project to d_model: [d_model, 25*n_chunks]
  vInp = ggml_mul_mat(vCtx0, vModel.vConvOutW, vInp);
  if (vModel.vConvOutB) {
    vInp = ggml_add(vCtx0, vInp, vModel.vConvOutB);
  }
  mCb(vInp, "after_conv_out", -1);

  const int64_t cNPos = vInp->ne[1]; // 25 * n_chunks

  // Per-chunk positional embeddings: repeat pos[0:13] for each chunk
  // (position indices reset 0..12 per chunk, not sequential across chunks)
  {
    const int64_t cTokensPerChunk = cNPos / cNChunks; // 13
    ggml_tensor *vPosTmp = ggml_view_2d(vCtx0, vModel.vPositionEmbeddings, vModel.vPositionEmbeddings->ne[0],
                                        cTokensPerChunk, vModel.vPositionEmbeddings->nb[1], 0);
    ggml_tensor *vTgt = ggml_new_tensor_2d(vCtx0, GGML_TYPE_F32, vModel.vPositionEmbeddings->ne[0], cNPos);
    vInp = ggml_add(vCtx0, vInp, ggml_repeat(vCtx0, vPosTmp, vTgt));
  }

  ggml_tensor *vCur = mBuildVit(vInp, cNPos, cNormTypeNormal, vHparams.vFfnOp,
                                nullptr, // pos embd already added above
                                nullptr);
  mCb(vCur, "after_transformer", -1);

  // MLP projector
  vCur = mBuildFfn(vCur, vModel.vMm1W, vModel.vMm1B, nullptr, nullptr, vModel.vMm2W, vModel.vMm2B, cFfnGeluErf, -1);
  mCb(vCur, "projected", -1);

  ggml_build_forward_expand(vGf, vCur);
  return vGf;
}
