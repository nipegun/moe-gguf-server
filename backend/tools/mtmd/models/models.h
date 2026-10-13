#pragma once

#include "../clip-graph.h"

/*
 * IMPORTANT: The mtmd module does NOT accept pull requests that are fully or predominantly AI-generated.
 * We encourage human contributors to ensure the quality and reliability of the codebase.
 */

struct clip_graph_siglip : clip_graph {
  clip_graph_siglip(clip_ctx *pCtx, const clip_image_f32 &pImg) : clip_graph(pCtx, pImg) {}
  ggml_cgraph *mBuild() override;
};

struct clip_graph_pixtral : clip_graph {
  clip_graph_pixtral(clip_ctx *pCtx, const clip_image_f32 &pImg) : clip_graph(pCtx, pImg) {}
  ggml_cgraph *mBuild() override;
};

struct clip_graph_qwen2vl : clip_graph {
  clip_graph_qwen2vl(clip_ctx *pCtx, const clip_image_f32 &pImg) : clip_graph(pCtx, pImg) {}
  ggml_cgraph *mBuild() override;
  ggml_tensor *mBuildInpWithTemporalMerge();
};

struct clip_graph_qwen3vl : clip_graph_qwen2vl {
  clip_graph_qwen3vl(clip_ctx *pCtx, const clip_image_f32 &pImg) : clip_graph_qwen2vl(pCtx, pImg) {}
  ggml_cgraph *mBuild() override;
};

struct clip_graph_minimax_m3 : clip_graph {
  clip_graph_minimax_m3(clip_ctx *pCtx, const clip_image_f32 &pImg) : clip_graph(pCtx, pImg) {}
  ggml_cgraph *mBuild() override;
  ggml_tensor *mApplyRope(ggml_tensor *pX, ggml_tensor *pPosH, ggml_tensor *pPosW);
};

struct clip_graph_mimovl : clip_graph {
  clip_graph_mimovl(clip_ctx *pCtx, const clip_image_f32 &pImg) : clip_graph(pCtx, pImg) {}
  ggml_cgraph *mBuild() override;
  // Force F32 mat-mul accumulation to avoid F16 overflow in the FFN down-proj
  // when the mmproj is stored in F16 (the source weights are BF16; downcasting
  // to F16 reduces dynamic range below the SwiGLU output magnitude on the last few layers).
  ggml_tensor *mBuildMm(ggml_tensor *pW, ggml_tensor *pX) const override;
};

struct clip_graph_step3vl : clip_graph {
  clip_graph_step3vl(clip_ctx *pCtx, const clip_image_f32 &pImg) : clip_graph(pCtx, pImg) {}
  ggml_cgraph *mBuild() override;
};

struct clip_graph_youtuvl : clip_graph {
  clip_graph_youtuvl(clip_ctx *pCtx, const clip_image_f32 &pImg) : clip_graph(pCtx, pImg) {}
  ggml_cgraph *mBuild() override;
};

struct clip_graph_yasa2 : clip_graph {
  clip_graph_yasa2(clip_ctx *pCtx, const clip_image_f32 &pImg) : clip_graph(pCtx, pImg) {}
  ggml_cgraph *mBuild() override;

  ggml_tensor *mLayerNormChannels(ggml_tensor *pInp, ggml_tensor *pW, ggml_tensor *pB, float pEps = 1e-6f);
  ggml_tensor *mConvnextGrn(ggml_tensor *pInp, ggml_tensor *pW, ggml_tensor *pB);
};

struct clip_graph_minicpmv4_6 : clip_graph {
  clip_graph_minicpmv4_6(clip_ctx *pCtx, const clip_image_f32 &pImg) : clip_graph(pCtx, pImg) {}
  ggml_cgraph *mBuild() override;
};

struct clip_graph_internvl : clip_graph {
  clip_graph_internvl(clip_ctx *pCtx, const clip_image_f32 &pImg) : clip_graph(pCtx, pImg) {}
  ggml_cgraph *mBuild() override;
  bool mSupportBatch() const override { return true; }
};

struct clip_graph_nemotron_v2_vl : clip_graph {
  clip_graph_nemotron_v2_vl(clip_ctx *pCtx, const clip_image_f32 &pImg) : clip_graph(pCtx, pImg) {}
  ggml_cgraph *mBuild() override;
};

struct clip_graph_kimivl : clip_graph {
  clip_graph_kimivl(clip_ctx *pCtx, const clip_image_f32 &pImg) : clip_graph(pCtx, pImg) {}
  ggml_cgraph *mBuild() override;
};

struct clip_graph_paddleocr : clip_graph {
  clip_graph_paddleocr(clip_ctx *pCtx, const clip_image_f32 &pImg) : clip_graph(pCtx, pImg) {}
  ggml_cgraph *mBuild() override;
};

struct clip_graph_dotsocr : clip_graph {
  clip_graph_dotsocr(clip_ctx *pCtx, const clip_image_f32 &pImg) : clip_graph(pCtx, pImg) {}
  ggml_cgraph *mBuild() override;
};

struct clip_graph_cogvlm : clip_graph {
  clip_graph_cogvlm(clip_ctx *pCtx, const clip_image_f32 &pImg) : clip_graph(pCtx, pImg) {}
  ggml_cgraph *mBuild() override;
};

struct clip_graph_whisper_enc : clip_graph {
  clip_graph_whisper_enc(clip_ctx *pCtx, const clip_image_f32 &pImg) : clip_graph(pCtx, pImg) {}
  ggml_cgraph *mBuild() override;
};

struct clip_graph_conformer : clip_graph {
  clip_graph_conformer(clip_ctx *pCtx, const clip_image_f32 &pImg) : clip_graph(pCtx, pImg) {}
  ggml_cgraph *mBuild() override;
};

struct clip_graph_granite_speech : clip_graph {
  clip_graph_granite_speech(clip_ctx *pCtx, const clip_image_f32 &pImg) : clip_graph(pCtx, pImg) {}
  ggml_cgraph *mBuild() override;
};

struct clip_graph_glm4v : clip_graph {
  clip_graph_glm4v(clip_ctx *pCtx, const clip_image_f32 &pImg) : clip_graph(pCtx, pImg) {}
  ggml_cgraph *mBuild() override;
};

struct clip_graph_hunyuanvl : clip_graph {
  clip_graph_hunyuanvl(clip_ctx *pCtx, const clip_image_f32 &pImg) : clip_graph(pCtx, pImg) {}
  ggml_cgraph *mBuild() override;
};

struct clip_graph_mobilenetv5 : clip_graph {
  clip_graph_mobilenetv5(clip_ctx *pCtx, const clip_image_f32 &pImg) : clip_graph(pCtx, pImg) {}
  ggml_cgraph *mBuild() override;

  ggml_tensor *mRmsNorm2d(ggml_tensor *pInp, ggml_tensor *pWeight, float pEps = 1e-6f);

  ggml_tensor *mPadSame2d(ggml_tensor *pInp, int pKernelH, int pKernelW, int pStrideH, int pStrideW, int pDilationH = 1,
                          int pDilationW = 1);

  ggml_tensor *mBuildEdgeResidual(ggml_tensor *pInp, const mobilenetv5_block &pBlock, int pStride);

  ggml_tensor *mBuildInvertedResidual(ggml_tensor *pInp, const mobilenetv5_block &pBlock, int pStride);

  ggml_tensor *mBuildMobilenetAttn(ggml_tensor *pInp, const mobilenetv5_block &pBlock);
};

struct clip_graph_qwen3a : clip_graph {
  clip_graph_qwen3a(clip_ctx *pCtx, const clip_image_f32 &pImg) : clip_graph(pCtx, pImg) {}
  ggml_cgraph *mBuild() override;
};

struct clip_graph_mimo_audio : clip_graph {
  clip_graph_mimo_audio(clip_ctx *pCtx, const clip_image_f32 &pImg) : clip_graph(pCtx, pImg) {}
  ggml_cgraph *mBuild() override;
};

struct clip_graph_kimik25 : clip_graph {
  clip_graph_kimik25(clip_ctx *pCtx, const clip_image_f32 &pImg) : clip_graph(pCtx, pImg) {}
  ggml_cgraph *mBuild() override;

  ggml_tensor *mResizePositionEmbeddings3d(uint32_t pInterpolationMode);
};

struct clip_graph_parakeet : clip_graph {
  clip_graph_parakeet(clip_ctx *pCtx, const clip_image_f32 &pImg) : clip_graph(pCtx, pImg) {}
  ggml_cgraph *mBuild() override;
};

struct clip_graph_exaone4_5 : clip_graph {
  clip_graph_exaone4_5(clip_ctx *pCtx, const clip_image_f32 &pImg) : clip_graph(pCtx, pImg) {}
  ggml_cgraph *mBuild() override;
};

struct clip_graph_granite4_vision : clip_graph {
  clip_graph_granite4_vision(clip_ctx *pCtx, const clip_image_f32 &pImg)
      : clip_graph(pCtx, pImg), cAddNewline(pImg.vAddNewline) {}

  ggml_cgraph *mBuild() override;

private:
  // The graph is per-tile since only batch-size 1 is supported in clip. As
  // such, this value is set at construct time based on the tile that will be
  // encoded, then used during build to determine how to handle newlines.
  const bool cAddNewline;

  ggml_tensor *mGather(ggml_tensor *pSrc, const std::string &pName, int pIdxLen);
  ggml_tensor *mInterpDown(ggml_tensor *pSrc, int pSide, int pNewSide);
  ggml_tensor *mBuildBlock(const qf_block &pBlk, ggml_tensor *pH, int pBid, int pSpatialOffset, int pImageSide,
                           int pWindowSide, int pQuerySide, float pQformerEps);

  ggml_tensor *mBuildNewlineRow(ggml_context *pCtx0);
  ggml_tensor *mAppendRowwiseNewlines(ggml_context *pCtx0, ggml_tensor *pTileOutput);
};
