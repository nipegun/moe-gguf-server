#pragma once

#include "ggml.h"
#include "clip.h"
#include "clip-impl.h"

#include <algorithm>
#include <array>
#include <vector>
#include <unordered_set>
#include <cstdint>
#include <cmath>

enum ffn_op_type {
  cFfnGelu,
  cFfnGeluErf,
  cFfnSilu,
  cFfnGeluQuick,
  cFfnReluSqr,
};

enum norm_type {
  cNormTypeNormal,
  cNormTypeRms,
};

enum resize_algo {
  cResizeAlgoBilinear, // stretch to target resolution
  cResizeAlgoBicubic,  // center-crop when aspect ratio doesn't match
  cResizeAlgoBicubicPillow,
  cResizeAlgoLanczos,
};

// Padding style for img_tool::resize
//   PAD_NONE    - no padding; direct resize to target dimensions
//   PAD_CEIL    - aspect-preserving pad (default)
//   PAD_NEAREST - aspect-preserving pad with nearest-integer rounding (Pillow byte-parity)
enum pad_style {
  cPadNone,
  cPadCeil,
  cPadNearest,
};

struct clip_hparams {
  int32_t vImageSize = 0;
  int32_t vPatchSize = 0;
  int32_t vNEmbd = 0;
  int32_t vNFf = 0;
  int32_t vProjectionDim = 0;
  int32_t vNHead = 0;
  int32_t vNHeadKv = 0;
  // 0 = derive from n_embd; set when qkv width != n_embd
  int32_t vNEmbdHead = 0;
  int32_t vNLayer = 0;
  int32_t vNMerge = 1; // number of patch merges **per-side**

  // for preprocessor
  int32_t vImageLongestEdge = 0;
  int32_t vImageMinPixels = -1;
  int32_t vImageMaxPixels = -1;
  resize_algo vImageResizeAlgo = cResizeAlgoBicubic;
  pad_style vImageResizePad = cPadCeil; // padding style when resizing
  std::array<uint8_t, 3> aImagePadColor = {0, 0, 0};

  // (preprocessor) for llava-uhd style models
  std::vector<clip_image_size> lImageResCandidates;
  int32_t vPreprocMinTiles = 0;
  int32_t vPreprocMaxTiles = 0;
  resize_algo vImageResizeAlgoRf = cResizeAlgoBicubic;
  resize_algo vImageResizeAlgoOv = cResizeAlgoBilinear;
  pad_style vImagePadRf = cPadCeil;                    // padding style for the refined image (llava-uhd slicing)
  pad_style vImagePadOv = cPadNone;                    // padding style for the overview image (llava-uhd slicing)
  std::array<uint8_t, 3> aImagePadColorRf = {0, 0, 0}; // padding color for refined image
  std::array<uint8_t, 3> aImagePadColorOv = {0, 0, 0}; // padding color for overview image

  float aImageMean[3];
  float aImageStd[3];

  // for models using dynamic image size, we need to have a smaller image size to warmup
  // otherwise, user will get OOM every time they load the model
  int32_t vWarmupImageSize = 0;
  int32_t vWarmupAudioSize = 3000;

  ffn_op_type vFfnOp = cFfnGelu;

  float vEps = 1e-6;
  float vRopeTheta = 0.0;

  // clamped SwiGLU limit (glm5v), 0 disables it
  float vSwigluClamp = 0.0f;
  std::vector<int32_t> lFeatureLayers;
  int32_t vAttnWindowSize = 0;
  int32_t vNWaPattern = 0;
  std::unordered_set<int32_t>
      vWaLayerIndexes; // explicit layer indexes that use full attention (for irregular patterns like YoutuVL)
  std::vector<int32_t> lWaPatternMode; // mimovl: per-layer window-attention mode

  // Granite4 Vision
  std::vector<int32_t> lProjSpatialOffsets;
  int32_t vDownsampleQuerySide;
  int32_t vDownsampleWindowSide;

  // audio
  int32_t vNMelBins = 0;          // whisper preprocessor
  int32_t vProjStackFactor = 0;   // ultravox
  int32_t vSubsamplingFactor = 0; // parakeet

  int32_t vAudioChunkSize = 0;
  int32_t vAudioConvKernelSize = 0;
  int32_t vAudioMaxPosEmb = 0;
  int32_t vAudioProjWindowSize = 0;
  int32_t vAudioProjDownsampleRate = 0;
  int32_t vAudioProjHeadCount = 0;

  // audio-to-mel preprocessor params
  int32_t vAudioChunkLen = -1; // in seconds
  int32_t vAudioSampleRate = -1;
  int32_t vAudioNFft = -1;
  int32_t vAudioWindowLen = -1;
  int32_t vAudioHopLen = -1;

  // parakeet
  std::vector<float> lMelFilters;
  std::vector<float> lWindow;

  // mimo-audio-tokenizer: residual vector quantizer
  int32_t vRvqNumQuantizers = 0;
  std::vector<int32_t> lRvqCodebookSize; // per-quantizer bin count (ragged, e.g. 1024/1024/256/128x17)

  // mimo-v2.5: LLM-side connector (input_local_transformer)
  int32_t vAudioLocalNLayer = 0;
  int32_t vAudioLocalGroupSize = 0;

  // MiniCPM-V 4.6: ViT layer after which the ViT merger is inserted (read from clip.vision.wa_layer_indexes)
  int32_t vInsertLayerId = 0;

  // custom value provided by user, can be undefined if not set
  int32_t vCustomImageMinTokens = -1;
  int32_t vCustomImageMaxTokens = -1;

  void mSetLimitImageTokens(int pNTokensMin, int pNTokensMax) {
    const int cPatchArea = vPatchSize * vPatchSize * vNMerge * vNMerge;
    vImageMinPixels = (vCustomImageMinTokens > 0 ? vCustomImageMinTokens : pNTokensMin) * cPatchArea;
    vImageMaxPixels = (vCustomImageMaxTokens > 0 ? vCustomImageMaxTokens : pNTokensMax) * cPatchArea;
    vWarmupImageSize = static_cast<int>(std::sqrt(vImageMaxPixels));
  }

  // keeps the pixel budget read from the model, only a user-provided token limit overrides it
  void mSetLimitImageTokens() {
    const int cPatchArea = vPatchSize * vPatchSize * vNMerge * vNMerge;
    if (vCustomImageMinTokens > 0) {
      vImageMinPixels = vCustomImageMinTokens * cPatchArea;
    }
    if (vCustomImageMaxTokens > 0) {
      vImageMaxPixels = vCustomImageMaxTokens * cPatchArea;
    }
  }

  void mSetWarmupNTokens(int pNTokens) {
    int vNTokPerSide = static_cast<int>(std::sqrt(pNTokens));
    GGML_ASSERT(vNTokPerSide * vNTokPerSide == pNTokens && "n_tokens must be n*n");
    vWarmupImageSize = vNTokPerSide * vPatchSize * vNMerge;
    // TODO: support warmup size for custom token numbers
  }

  bool mIsFeatureLayer(int32_t pLayer) const {
    return std::find(lFeatureLayers.begin(), lFeatureLayers.end(), pLayer) != lFeatureLayers.end();
  }
};

struct clip_layer {
  // layernorm 1 (or layer input norm, or pre-attention norm)
  ggml_tensor *vLn1W = nullptr;
  ggml_tensor *vLn1B = nullptr;

  // attention
  ggml_tensor *vKW = nullptr;
  ggml_tensor *vKB = nullptr;
  ggml_tensor *vQW = nullptr;
  ggml_tensor *vQB = nullptr;
  ggml_tensor *vVW = nullptr;
  ggml_tensor *vVB = nullptr;
  ggml_tensor *vQkvW = nullptr;
  ggml_tensor *vQkvB = nullptr;

  ggml_tensor *vOW = nullptr;
  ggml_tensor *vOB = nullptr;

  ggml_tensor *vAttnSinks = nullptr;

  ggml_tensor *vKNorm = nullptr;
  ggml_tensor *vQNorm = nullptr;

  ggml_tensor *vAttnPostNormW = nullptr;

  ggml_tensor *vFfUpW = nullptr;
  ggml_tensor *vFfUpB = nullptr;
  ggml_tensor *vFfGateW = nullptr;
  ggml_tensor *vFfGateB = nullptr;
  ggml_tensor *vFfDownW = nullptr;
  ggml_tensor *vFfDownB = nullptr;

  // layernorm 2 (or pre-FFN norm)
  ggml_tensor *vLn2W = nullptr;
  ggml_tensor *vLn2B = nullptr;

  ggml_tensor *vFfPostNormW = nullptr;

  // layer scale (no bias)
  ggml_tensor *vLs1W = nullptr;
  ggml_tensor *vLs2W = nullptr;
  ggml_tensor *vLsOutW = nullptr; // layer output scale

  // qwen3vl deepstack merger
  ggml_tensor *vDeepstackNormW = nullptr;
  ggml_tensor *vDeepstackNormB = nullptr;
  ggml_tensor *vDeepstackFc1W = nullptr;
  ggml_tensor *vDeepstackFc1B = nullptr;
  ggml_tensor *vDeepstackFc2W = nullptr;
  ggml_tensor *vDeepstackFc2B = nullptr;

  // lfm2
  ggml_tensor *vFfNormW = nullptr;
  ggml_tensor *vFfNormB = nullptr;
  ggml_tensor *vFfNorm1W = nullptr;
  ggml_tensor *vFfNorm1B = nullptr;
  ggml_tensor *vFfUp1W = nullptr;
  ggml_tensor *vFfUp1B = nullptr;
  ggml_tensor *vFfDown1W = nullptr;
  ggml_tensor *vFfDown1B = nullptr;
  ggml_tensor *vPosBiasU = nullptr;
  ggml_tensor *vPosBiasV = nullptr;
  ggml_tensor *vNormConvW = nullptr;
  ggml_tensor *vNormConvB = nullptr;
  ggml_tensor *vLinearPosW = nullptr;

  ggml_tensor *vConvNormW = nullptr;
  ggml_tensor *vConvNormB = nullptr;
  ggml_tensor *vConvNormMean = nullptr; // parakeet
  ggml_tensor *vConvNormVar = nullptr;  // parakeet
  ggml_tensor *vConvDwW = nullptr;
  ggml_tensor *vConvDwB = nullptr;
  ggml_tensor *vConvPw1W = nullptr;
  ggml_tensor *vConvPw1B = nullptr;
  ggml_tensor *vConvPw2W = nullptr;
  ggml_tensor *vConvPw2B = nullptr;

  // granite_speech conformer per-layer
  ggml_tensor *vAttnRelPosEmb = nullptr;

  // granite_speech qformer cross-attention
  ggml_tensor *vCrossAttnQW = nullptr;
  ggml_tensor *vCrossAttnQB = nullptr;
  ggml_tensor *vCrossAttnKW = nullptr;
  ggml_tensor *vCrossAttnKB = nullptr;
  ggml_tensor *vCrossAttnVW = nullptr;
  ggml_tensor *vCrossAttnVB = nullptr;
  ggml_tensor *vCrossAttnOW = nullptr;
  ggml_tensor *vCrossAttnOB = nullptr;
  ggml_tensor *vCrossAttnNormW = nullptr;
  ggml_tensor *vCrossAttnNormB = nullptr;

  bool mHasDeepstack() const { return vDeepstackFc1W != nullptr; }
};

// Expanded MobileNetV5 block structure for Gemma3n vision encoder
struct mobilenetv5_block {
  // Stage 0 (Edge Residual)
  ggml_tensor *vS0ConvExpW = nullptr;
  ggml_tensor *vS0Bn1W = nullptr;
  ggml_tensor *vS0ConvPwlW = nullptr;
  ggml_tensor *vS0Bn2W = nullptr;

  // Stage 1+ (Universal Inverted Residual)
  ggml_tensor *vDwStartW = nullptr;
  ggml_tensor *vDwStartBnW = nullptr;

  ggml_tensor *vPwExpW = nullptr;
  ggml_tensor *vPwExpBnW = nullptr;

  ggml_tensor *vDwMidW = nullptr;
  ggml_tensor *vDwMidBnW = nullptr;

  ggml_tensor *vPwProjW = nullptr;
  ggml_tensor *vPwProjBnW = nullptr;

  ggml_tensor *vLayerScaleW = nullptr;

  // Attention (MQA) components
  ggml_tensor *vAttnQW = nullptr;
  ggml_tensor *vAttnKW = nullptr;
  ggml_tensor *vAttnVW = nullptr;
  ggml_tensor *vAttnOW = nullptr;

  // Optional downsampling/norm in attention
  ggml_tensor *vAttnKDwW = nullptr;
  ggml_tensor *vAttnKNormW = nullptr;
  ggml_tensor *vAttnVDwW = nullptr;
  ggml_tensor *vAttnVNormW = nullptr;

  // Block norm (often present in attention blocks)
  ggml_tensor *vAttnNormW = nullptr;
};

struct yasa2_block {
  ggml_tensor *vDwW = nullptr;
  ggml_tensor *vDwB = nullptr;
  ggml_tensor *vLnW = nullptr;
  ggml_tensor *vLnB = nullptr;
  ggml_tensor *vPw1W = nullptr;
  ggml_tensor *vPw1B = nullptr;
  ggml_tensor *vGrnW = nullptr;
  ggml_tensor *vGrnB = nullptr;
  ggml_tensor *vPw2W = nullptr;
  ggml_tensor *vPw2B = nullptr;
};

struct yasa2_stage {
  ggml_tensor *vDownLnW = nullptr;
  ggml_tensor *vDownLnB = nullptr;
  ggml_tensor *vDownConvW = nullptr;
  ggml_tensor *vDownConvB = nullptr;
  std::vector<yasa2_block> lBlocks;
};

// QFormer projector block for models with 1 (or more) QFormer projectors
// Granite Speech, Granite4 Vision
struct qf_block {
  ggml_tensor *vQfProjQuery = nullptr;
  ggml_tensor *vQfProjNormW = nullptr;
  ggml_tensor *vQfProjNormB = nullptr;
  ggml_tensor *vQfProjLinearW = nullptr;
  ggml_tensor *vQfProjLinearB = nullptr;
  ggml_tensor *vQfProjPostNormW = nullptr;
  ggml_tensor *vQfProjPostNormB = nullptr;
  ggml_tensor *vQfProjImgPos = nullptr; // Vision only
  std::vector<clip_layer> lQfProjLayers;
};

struct clip_model {
  clip_modality vModality = cClipModalityVision;
  projector_type vProjType = cProjectorTypeUnknown;
  clip_hparams vHparams;

  // embeddings
  ggml_tensor *vClassEmbedding = nullptr;
  ggml_tensor *vPatchEmbeddings0 = nullptr;
  ggml_tensor *vPatchEmbeddings1 =
      nullptr; // second Conv2D kernel when we decouple Conv3D along temporal dimension (Qwen2VL)
  ggml_tensor *vPatchBias = nullptr;
  ggml_tensor *vPositionEmbeddings = nullptr;
  ggml_tensor *vNormEmbdW = nullptr;
  ggml_tensor *vNormEmbdB = nullptr;

  ggml_tensor *vPreLnW = nullptr;
  ggml_tensor *vPreLnB = nullptr;

  std::vector<clip_layer> lLayers;

  int32_t vNDeepstackLayers = 0; // used by Qwen3-VL, calculated from clip_layer

  ggml_tensor *vPostLnW;
  ggml_tensor *vPostLnB;

  ggml_tensor *vMmFcW;
  ggml_tensor *vMmFcB;
  ggml_tensor *vMmFfnUpW = nullptr;
  ggml_tensor *vMmFfnUpB = nullptr;
  ggml_tensor *vMmFfnGateW = nullptr;
  ggml_tensor *vMmFfnGateB = nullptr;
  ggml_tensor *vMmFfnDownW = nullptr;
  ggml_tensor *vMmFfnDownB = nullptr;
  ggml_tensor *vMmPostNormW = nullptr;
  ggml_tensor *vMmPostNormB = nullptr;

  // LLaVA projection
  ggml_tensor *vMmInputNormW = nullptr;
  ggml_tensor *vMmInputNormB = nullptr;
  ggml_tensor *vMm0W = nullptr;
  ggml_tensor *vMm0B = nullptr;
  ggml_tensor *vMm2W = nullptr;
  ggml_tensor *vMm2B = nullptr;
  ggml_tensor *vMmMergerFc1W = nullptr; // minimax-m3
  ggml_tensor *vMmMergerFc1B = nullptr;
  ggml_tensor *vMmMergerFc2W = nullptr;
  ggml_tensor *vMmMergerFc2B = nullptr;

  ggml_tensor *vImageNewline = nullptr;
  ggml_tensor *vViewSeperator = nullptr;

  // more mm.N projection layers (internvl, nemotron_v2_vl, meralion, lfm2a, ...)
  ggml_tensor *vMm1W = nullptr;
  ggml_tensor *vMm1B = nullptr;
  ggml_tensor *vMm3W = nullptr;
  ggml_tensor *vMm3B = nullptr;

  // mm.model.fc projection (step3vl, cogvlm, hunyuanvl)
  ggml_tensor *vMmModelProj = nullptr;
  ggml_tensor *vMmModelProjB = nullptr;

  // MiniCPM-V 4.6 ViT merger (window self-attention + ViT MLP downsample)
  ggml_tensor *vVitMergerLn1W = nullptr;
  ggml_tensor *vVitMergerLn1B = nullptr;
  ggml_tensor *vVitMergerAttnQW = nullptr;
  ggml_tensor *vVitMergerAttnQB = nullptr;
  ggml_tensor *vVitMergerAttnKW = nullptr;
  ggml_tensor *vVitMergerAttnKB = nullptr;
  ggml_tensor *vVitMergerAttnVW = nullptr;
  ggml_tensor *vVitMergerAttnVB = nullptr;
  ggml_tensor *vVitMergerAttnOW = nullptr;
  ggml_tensor *vVitMergerAttnOB = nullptr;
  ggml_tensor *vVitMergerDsLnW = nullptr;
  ggml_tensor *vVitMergerDsLnB = nullptr;
  ggml_tensor *vVitMergerDsUpW = nullptr;
  ggml_tensor *vVitMergerDsUpB = nullptr;
  ggml_tensor *vVitMergerDsDownW = nullptr;
  ggml_tensor *vVitMergerDsDownB = nullptr;

  // gemma3
  ggml_tensor *vMmInputProjW = nullptr;
  ggml_tensor *vMmSoftEmbNormW = nullptr;

  // mobilenetv5 for gemma3n
  std::vector<mobilenetv5_block> lMobilenetBlocks;
  std::vector<int> lMobilenetStageEnds;
  ggml_tensor *vMobilenetStemConvW = nullptr;
  ggml_tensor *vMobilenetStemConvB = nullptr;
  ggml_tensor *vMobilenetStemNormW = nullptr;
  ggml_tensor *vMmPostProjNormW = nullptr;

  // Multi-Scale Fusion Adapter (MSFA) components
  ggml_tensor *vMsfaConcatConvW = nullptr;
  ggml_tensor *vMsfaConcatNormW = nullptr;
  ggml_tensor *vMsfaFfnExpandW = nullptr;
  ggml_tensor *vMsfaFfnProjectW = nullptr;
  ggml_tensor *vMsfaFfnExpandBn = nullptr;
  ggml_tensor *vMsfaFfnProjectBn = nullptr;

  // yasa2
  ggml_tensor *vYasaPatchW = nullptr;
  ggml_tensor *vYasaPatchB = nullptr;
  ggml_tensor *vYasaPatchLnW = nullptr;
  ggml_tensor *vYasaPatchLnB = nullptr;
  ggml_tensor *vYasaBackboneLnW = nullptr;
  ggml_tensor *vYasaBackboneLnB = nullptr;
  ggml_tensor *vYasaVisionPosEmbed = nullptr;
  std::vector<yasa2_stage> lYasaStages;

  // pixtral, glm4v
  ggml_tensor *vTokenEmbdImgBreak = nullptr;
  ggml_tensor *vMmPatchMergerW = nullptr;
  ggml_tensor *vMmPatchMergerB = nullptr;

  // ultravox / whisper encoder
  ggml_tensor *vConv1d1W = nullptr;
  ggml_tensor *vConv1d1B = nullptr;
  ggml_tensor *vConv1d2W = nullptr;
  ggml_tensor *vConv1d2B = nullptr;
  ggml_tensor *vConvOutW = nullptr;
  ggml_tensor *vConvOutB = nullptr;
  ggml_tensor *vMmNormPreW = nullptr;
  ggml_tensor *vMmNormPreB = nullptr;
  ggml_tensor *vMmNormMidW = nullptr;

  // mimo-audio-tokenizer: post-transformer downsample + RVQ codebook
  ggml_tensor *vDownsampleConvW = nullptr; // no bias
  ggml_tensor *vDownsampleNormW = nullptr;
  ggml_tensor *vDownsampleNormB = nullptr;
  ggml_tensor *vRvqCodebook = nullptr; // merged 3D [n_q, max_bins, dim]

  // mimo-v2.5: text-side RVQ code embedding ("text codebook")
  ggml_tensor *vMmACodeEmbd = nullptr; // merged 3D [n_channels, vocab, dim]

  // mimo-v2.5: LLM-side connector (input_local_transformer, separate from the
  // audio_tokenizer's own encoder `lLayers`)
  std::vector<clip_layer> lMmALocalLayers;
  ggml_tensor *vMmALocalNormW = nullptr;

  // qwen3a
  ggml_tensor *vConv2d1W = nullptr;
  ggml_tensor *vConv2d1B = nullptr;
  ggml_tensor *vConv2d2W = nullptr;
  ggml_tensor *vConv2d2B = nullptr;
  ggml_tensor *vConv2d3W = nullptr;
  ggml_tensor *vConv2d3B = nullptr;

  // cogvlm
  ggml_tensor *vMmPostFcNormW = nullptr;
  ggml_tensor *vMmPostFcNormB = nullptr;
  ggml_tensor *vMmHTo4hW = nullptr;
  ggml_tensor *vMmGateW = nullptr;
  ggml_tensor *vMm4hToHW = nullptr;
  ggml_tensor *vMmBoi = nullptr;
  ggml_tensor *vMmEoi = nullptr;

  // hunyuanvl perceiver
  ggml_tensor *vMmPreNormW = nullptr;
  ggml_tensor *vMmImgBegin = nullptr;
  ggml_tensor *vMmImgEnd = nullptr;

  // lfm2 audio
  std::array<ggml_tensor *, 7> aPreEncodeConvXW = {nullptr};
  std::array<ggml_tensor *, 7> aPreEncodeConvXB = {nullptr};
  ggml_tensor *vPreEncodeOutW = nullptr;
  ggml_tensor *vPreEncodeOutB = nullptr;

  // granite_speech encoder
  ggml_tensor *vInpProjW = nullptr;
  ggml_tensor *vInpProjB = nullptr;
  ggml_tensor *vCtcOutW = nullptr;
  ggml_tensor *vCtcOutB = nullptr;
  ggml_tensor *vCtcOutMidW = nullptr;
  ggml_tensor *vCtcOutMidB = nullptr;
  // qformer projector(s)
  std::vector<qf_block> lQfProjBlocks;

  bool mAudioHasAvgpool() const {
    return vProjType == cProjectorTypeQwen2a || vProjType == cProjectorTypeVoxtral ||
           vProjType == cProjectorTypeMusicFlamingo;
  }

  bool mAudioHasStackFrames() const {
    return vProjType == cProjectorTypeUltravox || vProjType == cProjectorTypeVoxtral ||
           vProjType == cProjectorTypeMeralion;
  }
};

const clip_hparams *fClipGetHparams(const struct clip_ctx *pCtx);
