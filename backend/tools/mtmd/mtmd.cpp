#include "clip.h"
#include "clip-impl.h"
#include "mtmd.h"
#include "mtmd-audio.h"
#include "mtmd-image.h"
#include "debug/mtmd-debug.h"

#include "mgs.h"

// fix problem with std::min and std::max
#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <climits>
#include <vector>

// for still image data, layout is RGBRGBRGB...
// length of data must be nx * ny * 3 bytes
//
// for audio bitmap: nx = sample count, ny = 1, layout is F32 F32 F32 ...
// length of data must be nx * sizeof(float) bytes
struct mtmd_bitmap {
  uint32_t vNx = 0;
  uint32_t vNy = 0;
  std::string vId;       // optional user-defined id, for ex: can be set to image hash, useful for KV cache tracking
  bool vIsAudio = false; // true if the bitmap is audio

  // lazy-loaded bitmap
  mtmd_bitmap_lazy_callback vLazyCallback = nullptr;
  void *vLazyUserData = nullptr;

  mtmd_bitmap(const unsigned char *pData, uint32_t pNx, uint32_t pNy) : vNx(pNx), vNy(pNy), vIsAudio(false) {
    if (pData) {
      size_t vDataSize = (size_t)pNx * pNy * 3;
      this->lData.resize(vDataSize);
      std::memcpy(this->lData.data(), pData, vDataSize);
    }
  }

  mtmd_bitmap(const unsigned char *pData, uint32_t pNSamples) : vNx(pNSamples), vNy(1), vIsAudio(true) {
    if (pData) {
      size_t vDataSize = (size_t)vNx * sizeof(float);
      this->lData.resize(vDataSize);
      std::memcpy(this->lData.data(), pData, vDataSize);
    }
  }

  const std::vector<unsigned char> &mGetRoBuf() const { return lData; }

  bool mIsPlaceholder() const { return lData.empty(); }

  size_t mNBytes() const { return lData.size(); }

  bool mCanMergeWith(const mtmd_bitmap &pOther) const {
    // [QWEN_VIDEO] can (temporal) merge if both are images with same size
    return !vIsAudio && !pOther.vIsAudio && vNx == pOther.vNx && vNy == pOther.vNy;
  }

private:
  std::vector<unsigned char> lData;
};

// position indexing for decoder model
enum mtmd_pos_type {
  cMtmdPosTypeNormal,    // number of positions equals to number of tokens
  cMtmdPosTypeMrope,     // qwen-vl mrope style, each image takes max(t,h,w) position indexes
  cMtmdPosTypeHunyuanvl, // HunyuanVL mrope + BOI/EOI/newline layout with XD-RoPE dim-3
};

struct mtmd_image_tokens {
  uint32_t vNx = 0; // number of tokens in x direction
  uint32_t vNy = 0; // number of tokens in y direction
  mtmd_pos_type vPos = cMtmdPosTypeNormal;
  uint32_t vImageIdx =
      0; // 0-based position of this image among image chunks in the prompt(used by pos == MTMD_POS_TYPE_HUNYUANVL)
  uint32_t vNTemporalMerge = 1; // for qwen-vl style temporal merge
  uint32_t mNTokens() const {
    if (vPos == cMtmdPosTypeHunyuanvl) {
      // [BOI] [row0 tokens + newline] ... [row(ny-1) tokens + newline] [EOI]
      return (vNx + 1) * vNy + 2;
    }
    uint32_t vNz = vBatchF32.lEntries.size();
    if (vNTemporalMerge > 1) {
      // [QWEN_VIDEO] this logic is quite ugly, it's mostly to make qwen-vl temporal merge work, can be improved in the
      // future
      // TODO: simplify this by repeating the last frame until it fits the temporal merge
      if (vNz % vNTemporalMerge != 0) {
        vNz = vNz / vNTemporalMerge + 1;
      } else {
        vNz = vNz / vNTemporalMerge;
      }
    }
    return vNx * vNy * vNz;
  }
  clip_image_f32_batch vBatchF32; // preprocessed image patches
  std::string vId;                // optional user-defined ID, useful for KV cache tracking

  // true if one of entries in batch_f32 is a placeholder
  bool mIsPlaceholder() const {
    for (const auto &vEntry : vBatchF32.lEntries) {
      if (vEntry.mIsPlaceholder()) {
        return true;
      }
    }
    return false;
  }

  bool mCanBatchWith(const mtmd_image_tokens &pOther) {
    return vNx == pOther.vNx && vNy == pOther.vNy && vPos == pOther.vPos;
  }

  mtmd_image_tokens mClone() {
    return mtmd_image_tokens{vNx, vNy, vPos, vImageIdx, vNTemporalMerge, vBatchF32.mClone(), vId};
  }
};
using mtmd_image_tokens_ptr = std::unique_ptr<mtmd_image_tokens>;

struct mtmd_audio_tokens {
  uint32_t vNTokens = 0;          // number of tokens
  clip_image_f32_batch vBatchF32; // preprocessed image patches
  std::string vId;                // optional user-defined ID, useful for KV cache tracking

  // true if one of entries in batch_f32 is a placeholder
  bool mIsPlaceholder() const {
    for (const auto &vEntry : vBatchF32.lEntries) {
      if (vEntry.mIsPlaceholder()) {
        return true;
      }
    }
    return false;
  }

  mtmd_audio_tokens mClone() { return mtmd_audio_tokens{vNTokens, vBatchF32.mClone(), vId}; }
};
using mtmd_audio_tokens_ptr = std::unique_ptr<mtmd_audio_tokens>;

struct mtmd_input_chunk {
  mtmd_input_chunk_type vType;
  std::vector<mgs_token> lTokensText;
  mtmd_image_tokens_ptr vTokensImage;
  mtmd_audio_tokens_ptr vTokensAudio;

  bool mCanBatchWith(const mtmd_input_chunk &pOther) const {
    if (vType != pOther.vType) {
      return false;
    }

    if (vTokensImage && pOther.vTokensImage) {
      return vTokensImage->mCanBatchWith(*pOther.vTokensImage);
    }

    // TODO: allow batching audio chunks of the same size

    return false;
  }

  bool mIsPlaceholder() const {
    if (vType == MTMD_INPUT_CHUNK_TYPE_IMAGE) {
      return vTokensImage && vTokensImage->mIsPlaceholder();
    } else if (vType == MTMD_INPUT_CHUNK_TYPE_AUDIO) {
      return vTokensAudio && vTokensAudio->mIsPlaceholder();
    }
    return false;
  }
};

struct mtmd_input_chunks {
  std::vector<mtmd_input_chunk> lEntries;
};

struct mtmd_batch {
  mtmd_context *vCtx;
  std::vector<const mtmd_input_chunk *> lEntries;
  std::vector<float> lOutputEmbd; // aggregated output embedding for the whole batch
  mtmd_batch(mtmd_context *pCtx) : vCtx(pCtx) {}
  int32_t mNTokens() const {
    int32_t vN = 0;
    for (const auto *vChunk : lEntries) {
      vN += mtmd_input_chunk_get_n_tokens(vChunk);
    }
    return vN;
  }
};

const char *mtmd_default_marker() { return "<__media__>"; }

static clip_flash_attn_type mtmd_get_clip_flash_attn_type(enum mgs_flash_attn_type pFlashAttnType) {
  switch (pFlashAttnType) {
  case MGS_FLASH_ATTN_TYPE_AUTO:
    return cClipFlashAttnTypeAuto;
  case MGS_FLASH_ATTN_TYPE_DISABLED:
    return cClipFlashAttnTypeDisabled;
  case MGS_FLASH_ATTN_TYPE_ENABLED:
    return cClipFlashAttnTypeEnabled;
  }
  return cClipFlashAttnTypeAuto;
}

mtmd_context_params mtmd_context_params_default() {
  mtmd_context_params vParams{
      /* use_gpu           */ true,
      /* print_timings     */ true,
      /* n_threads         */ 4,
      /* media_marker      */ mtmd_default_marker(),
      /* flash_attn_type   */ MGS_FLASH_ATTN_TYPE_AUTO,
      /* warmup            */ true,
      /* image_min_tokens  */ -1,
      /* image_max_tokens  */ -1,
      /* cb_eval           */ nullptr,
      /* cb_eval_user_data */ nullptr,
      /* batch_max_tokens  */ 1024,
      /* progress_callback */ nullptr,
      /* progress_callback_user_data */ nullptr,
  };
  return vParams;
}

struct mtmd_context {
  struct clip_ctx *vCtxV;      // vision
  struct clip_ctx *vCtxA;      // audio
  std::vector<float> lOutEmbd; // image embedding vector

  bool vPrintTimings;
  int vNThreads;
  std::string vMediaMarker;
  const int cNEmbdText = -1;         // -1 means llm context not provided, skip checking this
  const mgs_vocab *vVocab = nullptr; // can be nullptr if text_model is not provided
  mtmd_pos_type vPosType;

  // these are not token, but strings used to mark the beginning and end of image/audio embeddings
  std::string vImgBeg;
  std::string vImgEnd;
  std::string vAudBeg;
  std::string vAudEnd;

  // for llava-uhd style models, we need special tokens in-between slices
  // minicpmv calls them "slices", other models call them "tiles"
  std::vector<mgs_token> lTokOvImgStart;  // overview image
  std::vector<mgs_token> lTokOvImgEnd;    // overview image
  std::vector<mgs_token> lTokSliImgStart; // single slice start
  std::vector<mgs_token> lTokSliImgEnd;   // single slice end
  std::vector<mgs_token> lTokSliImgMid;   // between 2 slices
  std::vector<mgs_token> lTokRowEnd;      // end of row
  bool vTokRowEndTrail = false;
  bool vOvImgFirst = false;

  // string template for slice image delimiters with row/col (idefics3)
  std::string vSliImgStartTmpl;

  std::unique_ptr<mtmd_audio_preprocessor> vAudioPreproc;
  std::unique_ptr<mtmd_image_preprocessor> vImagePreproc;

  // batching
  int32_t vBatchMaxTokens;

  // TODO @ngxson : add timings

  mtmd_context(const char *pMmprojFname, const mgs_model *pTextModel, const mtmd_context_params &pCtxParams,
               bool pNoAlloc = false)
      : vPrintTimings(pCtxParams.print_timings), vNThreads(pCtxParams.n_threads), vMediaMarker(pCtxParams.media_marker),
        cNEmbdText(pTextModel ? mgs_model_n_embd_inp(pTextModel) : -1),
        vVocab(pTextModel ? mgs_model_get_vocab(pTextModel) : nullptr), vBatchMaxTokens(pCtxParams.batch_max_tokens) {
    if (vMediaMarker.empty()) {
      throw std::runtime_error("media_marker must not be empty");
    }

    if (pTextModel) {
      auto vDecoderRopeType = mgs_model_rope_type(pTextModel);
      switch (vDecoderRopeType) {
      case MGS_ROPE_TYPE_NONE:
      case MGS_ROPE_TYPE_NORM:
      case MGS_ROPE_TYPE_NEOX: {
        vPosType = cMtmdPosTypeNormal;
      } break;
      case MGS_ROPE_TYPE_MROPE:
      case MGS_ROPE_TYPE_IMROPE: {
        vPosType = cMtmdPosTypeMrope;
      } break;
      default:
        throw std::runtime_error(fStringFormat("unsupported decoder rope type: %d\n", vDecoderRopeType));
      }
    }

    clip_context_params vCtxClipParams{
        /* use_gpu           */ pCtxParams.use_gpu,
        /* flash_attn_type   */ mtmd_get_clip_flash_attn_type(pCtxParams.flash_attn_type),
        /* image_min_tokens  */ pCtxParams.image_min_tokens,
        /* image_max_tokens  */ pCtxParams.image_max_tokens,
        /* warmup            */ pCtxParams.warmup,
        /* cb_eval           */ pCtxParams.cb_eval,
        /* cb_eval_user_data */ pCtxParams.cb_eval_user_data,
        /* no_alloc          */ pNoAlloc,
        /* progress_callback */ pCtxParams.progress_callback,
        /* progress_callback_user_data */ pCtxParams.progress_callback_user_data,
    };

    auto vRes = fClipInit(pMmprojFname, vCtxClipParams);
    vCtxV = vRes.vCtxV;
    vCtxA = vRes.vCtxA;
    if (!vCtxV && !vCtxA) {
      throw std::runtime_error(fStringFormat("Failed to load CLIP model from %s\n", pMmprojFname));
    }

    // if both vision and audio mmproj are present, we need to validate their n_embd
    if (vCtxV && vCtxA) {
      int vNEmbdV = fClipNMmprojEmbd(vCtxV);
      int vNEmbdA = fClipNMmprojEmbd(vCtxA);
      if (vNEmbdV != vNEmbdA) {
        throw std::runtime_error(fStringFormat(
            "mismatch between vision and audio mmproj (n_embd_v = %d, n_embd_a = %d)\n", vNEmbdV, vNEmbdA));
      }
    }

    // since we already validate n_embd of vision and audio mmproj,
    // we can safely assume that they are the same
    int vNEmbdClip = fClipNMmprojEmbd(vCtxV ? vCtxV : vCtxA);
    if (cNEmbdText > 0 && cNEmbdText != vNEmbdClip) {
      throw std::runtime_error(fStringFormat("mismatch between text model (n_embd = %d) and mmproj (n_embd = %d)\n"
                                             "hint: you may be using wrong mmproj\n",
                                             cNEmbdText, vNEmbdClip));
    }
    if (vCtxV) {
      mInitVision();
    }
    if (vCtxA) {
      mInitAudio();
    }
  }

  void mInitVision() {
    GGML_ASSERT(vCtxV != nullptr);
    vImagePreproc.reset();

    projector_type vProj = fClipGetProjectorType(vCtxV);

    switch (vProj) {
    case cProjectorTypeCogvlm:
    case cProjectorTypeJanusPro: {
      vImagePreproc = std::make_unique<mtmd_image_preprocessor_fixed_size>(vCtxV);
    } break;
    case cProjectorTypeMINICPMV4Point6: {
      // <image> (overview) </image><slice> (slice) </slice><slice> (slice) </slice>\n ...
      lTokOvImgStart = {mLookupToken("<image>")};
      lTokOvImgEnd = {mLookupToken("</image>")};
      lTokSliImgStart = {mLookupToken("<slice>")};
      lTokSliImgEnd = {mLookupToken("</slice>")};
      lTokRowEnd = {mLookupToken("\n")};
      vTokRowEndTrail = false; // no trailing end-of-row token
      vOvImgFirst = true;
      vImagePreproc = std::make_unique<mtmd_image_preprocessor_minicpmv>(vCtxV);
    } break;
    case cProjectorTypeQwen2vl:
    case cProjectorTypeQwen25vl:
    case cProjectorTypeQwen3vl:
    case cProjectorTypeMimovl: {
      // <|vision_start|> ... (image embeddings) ... <|vision_end|>
      vImgBeg = "<|vision_start|>";
      vImgEnd = "<|vision_end|>";
      vImagePreproc = std::make_unique<mtmd_image_preprocessor_dyn_size>(vCtxV);
    } break;
    case cProjectorTypeMinimaxM3: {
      // ]<]start of image[>[ ... (image embeddings) ... ]<]end of image[>[
      vImgBeg = "]<]start of image[>[";
      vImgEnd = "]<]end of image[>[";
      vImagePreproc = std::make_unique<mtmd_image_preprocessor_dyn_size>(vCtxV);
    } break;
    case cProjectorTypeYoutuvl: {
      // <|vision_start|> ... (image embeddings) ... <|vision_end|>
      vImgBeg = "<|vision_start|>";
      vImgEnd = "<|vision_end|>";
      vImagePreproc = std::make_unique<mtmd_image_preprocessor_youtuvl>(vCtxV);
    } break;
    case cProjectorTypeYasa2: {
      vImgBeg = "<image>";
      vImgEnd = "</image>";
      // Currently only supprots single-tile preprocessing: any input is downscaled
      // to one image_size x image_size tile (64 output tokens via 8x8 adaptive avg
      // pool).
      // However, the model itself supports llava-uhd multi-tile tiling for high-res
      // images. This will be implemented in a future PR (dispatch on has_pinpoints
      // to mtmd_image_preprocessor_llava_uhd) and emit image_grid_pinpoints in the
      // conversion script.
      vImagePreproc = std::make_unique<mtmd_image_preprocessor_fixed_size>(vCtxV);
    } break;
    case cProjectorTypeGemma3:
    case cProjectorTypeGemma3nv: {
      // <start_of_image> ... (image embeddings) ... <end_of_image>
      vImgBeg = "<start_of_image>";
      vImgEnd = "<end_of_image>";
      vImagePreproc = std::make_unique<mtmd_image_preprocessor_fixed_size>(vCtxV);
    } break;
    case cProjectorTypeIdefics3: {
      // https://github.com/huggingface/transformers/blob/a42ba80fa520c784c8f11a973ca9034e5f859b79/src/transformers/models/idefics3/processing_idefics3.py#L192-L215
      lTokOvImgStart = {mLookupToken("\n\n"), mLookupToken("<fake_token_around_image>"), mLookupToken("<global-img>")};
      lTokOvImgEnd = {mLookupToken("<fake_token_around_image>")};
      lTokRowEnd = {mLookupToken("\n")};
      vSliImgStartTmpl = "<fake_token_around_image><row_%d_col_%d>";
      vImagePreproc = std::make_unique<mtmd_image_preprocessor_idefics3>(vCtxV);
    } break;
    case cProjectorTypePixtral: {
      // https://github.com/huggingface/transformers/blob/1cd110c6cb6a6237614130c470e9a902dbc1a4bd/docs/source/en/model_doc/pixtral.md
      vImgEnd = "[IMG_END]";
      vImagePreproc = std::make_unique<mtmd_image_preprocessor_dyn_size>(vCtxV);
    } break;
    case cProjectorTypePhi4: {
      // Phi-4 uses media marker insertion only. Keep image boundary text empty.
      vImagePreproc = std::make_unique<mtmd_image_preprocessor_dyn_size>(vCtxV);
    } break;
    case cProjectorTypeStep3vl: {
      // Step3 format:
      //   <patch_start> (patch) <patch_end> [<patch_newline>]
      //   ... (all patch rows)
      //   <im_start> (overview) <im_end>
      lTokOvImgStart = {mLookupToken("<im_start>")};
      lTokOvImgEnd = {mLookupToken("<im_end>")};
      lTokSliImgStart = {mLookupToken("<patch_start>")};
      lTokSliImgEnd = {mLookupToken("<patch_end>")};
      lTokRowEnd = {mLookupToken("<patch_newline>")};
      vTokRowEndTrail = false;
      vOvImgFirst = false; // patches first, overview last
      vImagePreproc = std::make_unique<mtmd_image_preprocessor_step3vl>(vCtxV);
    } break;
    case cProjectorTypeInternvl: {
      // <img> ... (image embeddings) ... </img>
      vImgBeg = "<img>";
      vImgEnd = "</img>";
      vImagePreproc = std::make_unique<mtmd_image_preprocessor_internvl>(vCtxV);
      vOvImgFirst = false;
    } break;
    case cProjectorTypeKimivl: {
      // <|media_start|> ... (image embeddings) ... <|media_end|>
      vImgBeg = "<|media_start|>";
      vImgEnd = "<|media_end|>";
      vImagePreproc = std::make_unique<mtmd_image_preprocessor_dyn_size>(vCtxV);
    } break;
    case cProjectorTypeKimik25: {
      // GLM-5.2-V reuses the Kimi-K2.5 vision encoder and projector, but marks
      // images with its own tokens, so decide based on the text model vocab
      if (mLookupToken("<|begin_of_image|>") != MGS_TOKEN_NULL) {
        // <|begin_of_image|> ... (image embeddings) ... <|end_of_image|>
        vImgBeg = "<|begin_of_image|>";
        vImgEnd = "<|end_of_image|>";
      } else {
        // <|media_begin|> ... (image embeddings) ... <|media_end|>
        vImgBeg = "<|media_begin|>";
        vImgEnd = "<|media_end|>";
      }
      vImagePreproc = std::make_unique<mtmd_image_preprocessor_dyn_size>(vCtxV);
    } break;
    case cProjectorTypeLightonocr: {
      // <|im_start|> ... (image embeddings) ... <|im_end|>
      vImgBeg = "<|im_start|>";
      vImgEnd = "<|im_end|>";
      vImagePreproc = std::make_unique<mtmd_image_preprocessor_longest_edge>(vCtxV);
    } break;
    case cProjectorTypeDotsOcr: {
      // <|img|> ... (image embeddings) ... <|endofimg|>
      vImgBeg = "<|img|>";
      vImgEnd = "<|endofimg|>";
      vImagePreproc = std::make_unique<mtmd_image_preprocessor_dyn_size>(vCtxV);
    } break;
    case cProjectorTypeNemotronV2Vl: {
      vImagePreproc = std::make_unique<mtmd_image_preprocessor_fixed_size>(vCtxV);
    } break;
    case cProjectorTypeLfm2: {
      // multi-tile:
      //   <|image_start|>
      //     <|img_row_1_col_1|> (tile) <|img_row_1_col_2|> (tile) ...
      //     <|img_thumbnail|> (thumbnail)
      //   <|image_end|>
      // single-tile:
      //   <|image_start|> (image) <|image_end|>
      vImgBeg = "<|image_start|>";
      vImgEnd = "<|image_end|>";
      vSliImgStartTmpl = "<|img_row_%d_col_%d|>";
      lTokOvImgStart = {mLookupToken("<|img_thumbnail|>")};
      vOvImgFirst = false;
      vImagePreproc = std::make_unique<mtmd_image_preprocessor_lfm2>(vCtxV);
    } break;
    case cProjectorTypeGlm4v: {
      // <|begin_of_image|> ... (image embeddings) ... <|end_of_image|>
      vImgBeg = "<|begin_of_image|>";
      vImgEnd = "<|end_of_image|>";
      vImagePreproc = std::make_unique<mtmd_image_preprocessor_dyn_size>(vCtxV);
    } break;
    case cProjectorTypeGlm5v: {
      // <|begin_of_image|> ... (image embeddings) ... <|end_of_image|>
      vImgBeg = "<|begin_of_image|>";
      vImgEnd = "<|end_of_image|>";
      vImagePreproc = std::make_unique<mtmd_image_preprocessor_glm5v>(vCtxV);
    } break;
    case cProjectorTypePaddleocr: {
      // <|IMAGE_START|> ... (image embeddings) ... <|IMAGE_END|>
      vImgBeg = "<|IMAGE_START|>";
      vImgEnd = "<|IMAGE_END|>";
      vImagePreproc = std::make_unique<mtmd_image_preprocessor_dyn_size>(vCtxV);
    } break;
    case cProjectorTypeHunyuanvl: {
      // note: these use fullwidth ｜ (U+FF5C) and ▁ (U+2581) to match the tokenizer vocabulary
      vImgBeg = "<｜hy_place▁holder▁no▁100｜>";
      vImgEnd = "<｜hy_place▁holder▁no▁101｜>";
      vImagePreproc = std::make_unique<mtmd_image_preprocessor_dyn_size>(vCtxV);
    } break;
    case cProjectorTypeEXAONE4Point5: {
      // <vision> ... (image embeddings) ... </vision>
      vImgBeg = "<vision>";
      vImgEnd = "</vision>";
      vImagePreproc = std::make_unique<mtmd_image_preprocessor_dyn_size>(vCtxV);
    } break;
    case cProjectorTypeGranite4Vision: {
      vImgBeg = "<image>";
      vImgEnd = "";
      vImagePreproc = std::make_unique<mtmd_image_preprocessor_granite>(vCtxV);
      vOvImgFirst = true;
    } break;
    default:
      throw std::runtime_error(fStringFormat("%s: unexpected vision projector type %d\n", __func__, vProj));
    }

    GGML_ASSERT(vImagePreproc != nullptr);
  }

  void mInitAudio() {
    GGML_ASSERT(vCtxA != nullptr);
    vAudioPreproc.reset();

    projector_type vProj = fClipGetProjectorType(vCtxA);

    LOG_WRN("%s: audio input is in experimental stage and may have reduced quality\n", __func__);

    // set preprocessor
    switch (vProj) {
    case cProjectorTypeQwen2a:
    case cProjectorTypeQwen25o: {
      // <|audio_bos|> ... (embeddings) ... <|audio_eos|>
      vAudBeg = "<|audio_bos|>";
      vAudEnd = "<|audio_eos|>";
      vAudioPreproc = std::make_unique<mtmd_audio_preprocessor_whisper>(vCtxA);
    } break;
    case cProjectorTypeQwen3a: {
      vAudBeg = "<|audio_start|>";
      vAudEnd = "<|audio_end|>";
      vAudioPreproc = std::make_unique<mtmd_audio_preprocessor_qwen3a>(vCtxA);
    } break;
    case cProjectorTypeVoxtral: {
      // [BEGIN_AUDIO] ... (embeddings) ...
      vAudBeg = "[BEGIN_AUDIO]";
      vAudioPreproc = std::make_unique<mtmd_audio_preprocessor_whisper>(vCtxA);
    } break;
    case cProjectorTypeMusicFlamingo: {
      // <sound> ... (embeddings) ...
      vAudBeg = "<sound>";
      vAudioPreproc = std::make_unique<mtmd_audio_preprocessor_whisper>(vCtxA);
    } break;
    case cProjectorTypeUltravox:
    case cProjectorTypeGlma:
    case cProjectorTypeMeralion: {
      vAudioPreproc = std::make_unique<mtmd_audio_preprocessor_whisper>(vCtxA);
    } break;
    case cProjectorTypeLfm2a: {
      vAudioPreproc = std::make_unique<mtmd_audio_preprocessor_conformer>(vCtxA);
    } break;
    case cProjectorTypeGraniteSpeech: {
      vAudioPreproc = std::make_unique<mtmd_audio_preprocessor_granite_speech>(vCtxA);
    } break;
    case cProjectorTypeParakeet: {
      vAudioPreproc = std::make_unique<mtmd_audio_preprocessor_parakeet>(vCtxA);
    } break;
    case cProjectorTypeMimoAudio: {
      vAudBeg = "<|mimo_audio_start|>";
      vAudEnd = "<|mimo_audio_end|>";
      vAudioPreproc = std::make_unique<mtmd_audio_preprocessor_mimo_audio>(vCtxA);
    } break;
    default:
      throw std::runtime_error(fStringFormat("%s: unexpected audio projector type %d\n", __func__, vProj));
    }

    // initialize audio preprocessor
    GGML_ASSERT(vAudioPreproc != nullptr);
    vAudioPreproc->mInitialize();
  }

  // get clip ctx based on chunk type
  clip_ctx *mGetClipCtx(const mtmd_input_chunk *pChunk) const {
    if (pChunk->vType == MTMD_INPUT_CHUNK_TYPE_IMAGE) {
      return vCtxV;
    } else if (pChunk->vType == MTMD_INPUT_CHUNK_TYPE_AUDIO) {
      return vCtxA;
    }
    GGML_ABORT("unknown chunk type");
  }

  projector_type mProjTypeV() const { return vCtxV ? fClipGetProjectorType(vCtxV) : cProjectorTypeUnknown; }

  projector_type mProjTypeA() const { return vCtxA ? fClipGetProjectorType(vCtxA) : cProjectorTypeUnknown; }

  int64_t mNEmbdOut() const {
    if (vCtxV) {
      return fClipNMmprojEmbd(vCtxV);
    } else if (vCtxA) {
      return fClipNMmprojEmbd(vCtxA);
    } else {
      throw std::runtime_error("no CLIP model loaded");
    }
  }

  ~mtmd_context() {
    fClipFree(vCtxA);
    fClipFree(vCtxV);
  }

private:
  mgs_token mLookupToken(const std::string &pTokenText) {
    if (vVocab == nullptr) {
      // TODO @ngxson : this case is currently hit by mtmd_get_memory_usage
      // but we should reconsider this if this case is needed in other places in the future
      return MGS_TOKEN_NULL;
    }
    const int cNVocab = mgs_vocab_n_tokens(vVocab);
    for (int vI = 0; vI < cNVocab; vI++) {
      if (mTokenToPiece(vVocab, vI, true) == pTokenText) {
        return vI;
      }
    }
    return MGS_TOKEN_NULL;
  }

  std::string mTokenToPiece(const mgs_vocab *pVocab, mgs_token pToken, bool pSpecial) {
    if (pVocab == nullptr) {
      throw std::runtime_error("mgs_vocab is not provided");
    }
    std::string vPiece;
    vPiece.resize(vPiece.capacity()); // using string internal cache, 15 bytes + '\n'
    const int cNChars = mgs_token_to_piece(pVocab, pToken, &vPiece[0], vPiece.size(), 0, pSpecial);
    if (cNChars < 0) {
      vPiece.resize(-cNChars);
      int vCheck = mgs_token_to_piece(pVocab, pToken, &vPiece[0], vPiece.size(), 0, pSpecial);
      GGML_ASSERT(vCheck == -cNChars);
    } else {
      vPiece.resize(cNChars);
    }
    return vPiece;
  }
};

mtmd_context *mtmd_init_from_file(const char *pMmprojFname, const struct mgs_model *pTextModel,
                                  const struct mtmd_context_params pCtxParams) {
  try {
    return new mtmd_context(pMmprojFname, pTextModel, pCtxParams);
  } catch (const std::exception &vE) {
    LOG_ERR("%s: error: %s\n", __func__, vE.what());
    return nullptr;
  }
}

void mtmd_free(mtmd_context *pCtx) { delete pCtx; }

struct mtmd_tokenizer {
  mtmd_context *vCtx;

  std::string vInputText; // note: can contain null bytes; do not use c_str()
  bool vAddSpecial;
  bool vParseSpecial;
  const mgs_vocab *vVocab;

  struct part {
    std::string vText;
    const mtmd_bitmap *vBitmap;
  };
  std::vector<part> lParts;
  // these will be freed when mtmd_tokenizer finishes
  std::vector<mtmd::bitmap> lBmFromLazy; // TODO @ngxson : refactor, free bm_from_lazy progressively
  std::vector<const char *> lTextFromLazy;

  mtmd_input_chunks vCur;
  uint32_t vNImagesAdded = 0; // 0-based index assigned to the next image chunk

  ~mtmd_tokenizer() {
    // note: mtmd::bitmap is already RAII
    for (auto &vStr : lTextFromLazy) {
      free((void *)vStr);
    }
  }

  mtmd_tokenizer(mtmd_context *pCtx, const mtmd_input_text *pText, const mtmd_bitmap **pBmps, size_t pNBitmaps)
      : vCtx(pCtx) {
    vAddSpecial = pText->add_special;
    vParseSpecial = pText->parse_special;
    vVocab = pCtx->vVocab;

    vInputText.assign(pText->text, pText->text_len);

    std::vector<const mtmd_bitmap *> lBitmaps(pBmps, pBmps + pNBitmaps);
    auto vPartsStr = mSplitText(vInputText, pCtx->vMediaMarker);
    size_t vIBm = 0;
    for (const auto &vPart : vPartsStr) {
      if (vPart == pCtx->vMediaMarker) {
        if (vIBm >= lBitmaps.size()) {
          throw std::runtime_error(fStringFormat(
              "number of media markers in text (%zu) exceeds number of bitmaps (%zu)", vIBm + 1, lBitmaps.size()));
        }
        lParts.push_back({"", lBitmaps[vIBm++]});
      } else {
        lParts.push_back({std::move(vPart), nullptr});
      }
    }

    size_t vNMarkers = 0;
    for (const auto &vPart : lParts) {
      if (vPart.vBitmap != nullptr) {
        vNMarkers++;
      }
    }
    if (vNMarkers != lBitmaps.size()) {
      throw std::runtime_error(fStringFormat(
          "number of media markers in text (%zu) does not match number of bitmaps (%zu)", vNMarkers, lBitmaps.size()));
    }

    mExpandLazyBitmaps();
  }

  void mExpandLazyBitmaps() {
    std::vector<part> lExpanded;
    lExpanded.reserve(lParts.size());
    for (auto &vP : lParts) {
      if (vP.vBitmap != nullptr && vP.vBitmap->vLazyCallback) {
        LOG_DBG("%s: expanding lazy bitmap\n", __func__);
        for (size_t vI = 0;; vI++) {
          char *vOutStr = nullptr;
          mtmd_bitmap *vOutBm = nullptr;
          int vRes = vP.vBitmap->vLazyCallback(vI, vP.vBitmap->vLazyUserData, &vOutBm, &vOutStr);
          if (vOutBm && vOutStr) {
            throw std::runtime_error(fStringFormat("lazy callback cannot return both bitmap and text"));
          }
          if (vRes == 0) {
            // OK, append the returned chunk; lazy part is not yet added
            if (vOutBm) {
              auto &vPtr = lBmFromLazy.emplace_back(vOutBm); // remember to free it later
              lExpanded.push_back({"", vPtr.ptr.get()});
              LOG_DBG("%s: lazy callback returned bitmap with dimensions %d x %d\n", __func__, vOutBm->vNx,
                      vOutBm->vNy);
            } else if (vOutStr) {
              auto &vPtr = lTextFromLazy.emplace_back(vOutStr); // remember to free it later
              lExpanded.push_back({vPtr, nullptr});
              LOG_DBG("%s: lazy callback returned text: %s\n", __func__, vOutStr);
            }
          } else if (vRes == -1) {
            // EOF: lazy part removes itself (not added to expanded)
            break;
          } else if (vRes == -2) {
            // error
            throw std::runtime_error(fStringFormat("lazy callback returned error"));
          }
        }
      } else {
        lExpanded.push_back(std::move(vP));
      }
    }
    lParts = std::move(lExpanded);
  }

  int32_t mTokenize(mtmd_input_chunks *pOutput) {
    vCur.lEntries.clear();

    // [QWEN_VIDEO] handle frame merging for models that support it (i.e. qwen-vl)
    int vNMergeFrames = 1;
    if (vCtx->vCtxV) {
      vNMergeFrames = fClipModelNTemporalMerge(vCtx->vCtxV);
      GGML_ASSERT(
          vNMergeFrames <= 2 &&
          "we only support merging maximum 2 images for now; open an issue if this model supports merging more");
    }

    // Build merged_bitmaps: each entry is a group of 1 or 2 bitmaps.
    // For consecutive mergeable bitmap parts, merge them and collapse the second part out of this->lParts.
    std::vector<std::vector<const mtmd_bitmap *>> lMergedBitmaps;
    if (vNMergeFrames > 1) {
      for (size_t vI = 0; vI < lParts.size(); ++vI) {
        if (lParts[vI].vBitmap == nullptr) {
          continue;
        }
        if (vI + 1 < lParts.size() && lParts[vI + 1].vBitmap != nullptr) {
          const mtmd_bitmap *vBmA = lParts[vI].vBitmap;
          const mtmd_bitmap *vBmB = lParts[vI + 1].vBitmap;
          if (vBmA->mCanMergeWith(*vBmB)) {
            LOG_DBG("%s: merging 2 frames at part index %zu and %zu\n", __func__, vI, vI + 1);
            lMergedBitmaps.push_back({vBmA, vBmB});
            lParts.erase(lParts.begin() + vI + 1); // collapse the second bitmap part
            continue;
          }
        }
        LOG_DBG("%s: no merging for part index %zu\n", __func__, vI);
        lMergedBitmaps.push_back({lParts[vI].vBitmap});
      }
    } else {
      for (const auto &vP : lParts) {
        if (vP.vBitmap != nullptr) {
          lMergedBitmaps.push_back({vP.vBitmap});
        }
      }
    }

    size_t vIBm = 0;
    for (const auto &vP : lParts) {
      if (vP.vBitmap != nullptr) {
        if (vIBm >= lMergedBitmaps.size()) {
          LOG_ERR("%s: error: number of bitmaps (%zu) does not match number of markers (%zu)\n", __func__,
                  lMergedBitmaps.size(), lParts.size() - 1);
          return 1;
        }
        auto vBmps = lMergedBitmaps[vIBm++];
        int32_t vRes = mAddMedia(vBmps);
        if (vRes != 0) {
          return vRes;
        }
      } else {
        mAddText(vP.vText, vParseSpecial);
      }
    }

    if (vVocab != nullptr) {
      if (vAddSpecial && mgs_vocab_get_add_bos(vVocab)) {
        // if first chunk is text, we add BOS token to first text chunk
        // otherwise, create a new text chunk with BOS token
        if (!vCur.lEntries.empty() && vCur.lEntries[0].vType == MTMD_INPUT_CHUNK_TYPE_TEXT) {
          // add BOS token to the beginning of first text chunk
          vCur.lEntries[0].lTokensText.insert(vCur.lEntries[0].lTokensText.begin(), mgs_vocab_bos(vVocab));
        } else {
          // create a new text chunk with BOS token at the beginning
          mtmd_input_chunk vBosChunk{
              MTMD_INPUT_CHUNK_TYPE_TEXT,
              {mgs_vocab_bos(vVocab)},
              nullptr, // image tokens
              nullptr, // audio tokens
          };
          vCur.lEntries.insert(vCur.lEntries.begin(), std::move(vBosChunk));
        }
      }

      if (vAddSpecial && mgs_vocab_get_add_eos(vVocab)) {
        // if last chunk is text, we add EOS token to it
        mAddText({mgs_vocab_eos(vVocab)});
      }
    }

    if (vIBm != lMergedBitmaps.size()) {
      LOG_ERR("%s: error: number of bitmaps (%zu) does not match number of markers (%zu)\n", __func__,
              lMergedBitmaps.size(), lParts.size() - 1);
      return 1;
    }

    *pOutput = std::move(vCur);

    return 0;
  }

  void mAddText(const std::string &pTxt, bool pParseSpecial) {
    if (vVocab == nullptr) {
      throw std::runtime_error("mgs_vocab is not provided");
    }
    LOG_DBG("%s: %s\n", __func__, pTxt.c_str());
    auto vTokens = mMtmdTokenizeTextInternal(vVocab, pTxt, /* add_special */ false, pParseSpecial);
    mAddText(vTokens);
  }

  void mAddText(const std::vector<mgs_token> &pTokens) {
    if (pTokens.empty()) {
      return;
    }
    // if last entry is also a text chunk, add tokens to it instead of creating new chunk
    if (!vCur.lEntries.empty() && vCur.lEntries.back().vType == MTMD_INPUT_CHUNK_TYPE_TEXT) {
      vCur.lEntries.back().lTokensText.insert(vCur.lEntries.back().lTokensText.end(), pTokens.begin(), pTokens.end());
    } else {
      mtmd_input_chunk vChunk{
          MTMD_INPUT_CHUNK_TYPE_TEXT,
          pTokens,
          nullptr, // image tokens
          nullptr, // audio tokens
      };
      vCur.lEntries.emplace_back(std::move(vChunk));
    }
  }

  int32_t mAddMedia(std::vector<const mtmd_bitmap *> &pBitmaps) {
    GGML_ASSERT(!pBitmaps.empty());

    // note: only one type of media is supported per call, caller should enforce this
    const bool cIsVision = !pBitmaps[0]->vIsAudio;

    if (cIsVision) {
      // handle image

      if (!vCtx->vCtxV) {
        LOG_ERR("%s: error: model does not support vision input\n", __func__);
        return 2;
      }

      if (!vCtx->vImgBeg.empty()) {
        mAddText(vCtx->vImgBeg, true); // add image begin token
      }

      // TODO @ngxson : this is quite hacky because preprocessor only support batch with one single element, that need
      // to be fixed in the future (e.g. by changing the preprocessor interface always take single input)

      mtmd_image_preproc_out vPreprocOut;

      for (const auto *vBmp : pBitmaps) {
        // sanity check
        GGML_ASSERT(!vBmp->vIsAudio);
        GGML_ASSERT(vCtx->vImagePreproc != nullptr);
        if (vBmp->vNx <= 0 || vBmp->vNy <= 0) {
          LOG_ERR("%s: error: invalid bitmap dimensions: nx = %d, ny = %d\n", __func__, vBmp->vNx, vBmp->vNy);
          return 2;
        }

        // convert mtmd_bitmap to clip_image_u8
        clip_image_u8 vImgU8;
        vImgU8.mSetSize({(int)vBmp->vNx, (int)vBmp->vNy}, vBmp->mIsPlaceholder());
        vImgU8.mCpyBuf(vBmp->mGetRoBuf());

        // preprocess image
        mtmd_image_preproc_out vTmpPreprocOut = vCtx->vImagePreproc->mPreprocess(vImgU8);

        // move entries and grid dimensions to the "global" preproc_out
        for (auto &vEntry : vTmpPreprocOut.lEntries) {
          vPreprocOut.lEntries.emplace_back(std::move(vEntry));
        }

        // for llava-uhd style, we need to handle grid too
        // we don't care about overwriting these values for now because the case where bitmaps.size() > 1 is only for
        // frame merging (qwen-vl), not supported by llava-uhd
        if ((vTmpPreprocOut.vGridX > 0 && vTmpPreprocOut.vGridY > 0) || vTmpPreprocOut.mHasOverview()) {
          GGML_ASSERT(pBitmaps.size() == 1);
          vPreprocOut.vGridX = vTmpPreprocOut.vGridX;
          vPreprocOut.vGridY = vTmpPreprocOut.vGridY;
          vPreprocOut.vOverview = std::move(vTmpPreprocOut.vOverview);
        }
      }

      LOG_DBG("%s: preproc_out has %zu entries, grid_x = %d, grid_y = %d, has_overview = %d\n", __func__,
              vPreprocOut.lEntries.size(), vPreprocOut.vGridX, vPreprocOut.vGridY, vPreprocOut.mHasOverview() ? 1 : 0);

      // handle llava-uhd style preprocessing
      // (output either a grid, or overview-only)
      const bool cHasTilingGrid = (vPreprocOut.vGridX > 0 && vPreprocOut.vGridY > 0) || vPreprocOut.mHasOverview();

      if (cHasTilingGrid) {
        // [QWEN_VIDEO] we do not support "frame merging" for llava-uhd style, so no batching for now
        GGML_ASSERT(pBitmaps.size() == 1);

        const int cNCol = vPreprocOut.vGridX;
        const int cNRow = vPreprocOut.vGridY;

        // split batch into chunks of single images
        auto vChunks = mSplitBatchToChunk(std::move(vPreprocOut), pBitmaps[0]->vId);
        GGML_ASSERT(vChunks.size() > 0);

        // NOTE: preproc_out is invalidated after this point, do not use it anymore

        // split_batch_to_chunk must always put the overview image first
        auto vOvChunk = std::move(vChunks.front());
        vChunks.erase(vChunks.begin());

        // add overview image (first)
        if (vCtx->vOvImgFirst) {
          mAddText(vCtx->lTokOvImgStart);
          vCur.lEntries.emplace_back(std::move(vOvChunk));
          mAddText(vCtx->lTokOvImgEnd);
        }

        // add slices (or tiles)
        if (!vChunks.empty()) {
          LOG_DBG("%s: adding %d slices (%d rows x %d cols)\n", __func__, (int)vChunks.size(), cNRow, cNCol);
          GGML_ASSERT((int)vChunks.size() == cNRow * cNCol);
          for (int vY = 0; vY < cNRow; vY++) {
            for (int vX = 0; vX < cNCol; vX++) {
              const bool cIsLastInRow = (vX == cNCol - 1);
              if (!vCtx->lTokSliImgStart.empty()) {
                mAddText(vCtx->lTokSliImgStart);
              } else if (!vCtx->vSliImgStartTmpl.empty()) {
                // If using a template to preceed a slice image
                const size_t cSz = std::snprintf(nullptr, 0, vCtx->vSliImgStartTmpl.c_str(), vY + 1, vX + 1) + 1;
                std::unique_ptr<char[]> vBuf(new char[cSz]);
                std::snprintf(vBuf.get(), cSz, vCtx->vSliImgStartTmpl.c_str(), vY + 1, vX + 1);
                mAddText(std::string(vBuf.get(), vBuf.get() + cSz - 1), true);
              }

              auto &vCurrChunk = vChunks[vY * cNCol + vX];
              auto &vCurrBatch = vCurrChunk.vTokensImage->vBatchF32;
              if (vCurrBatch.lEntries.size() != 1) {
                throw std::runtime_error(fStringFormat("%s: expect 1 image in batch_f32", __func__));
              }

              LOG_DBG("%s: adding slice image at row %d col %d\n", __func__, vY, vX);
              vCur.lEntries.emplace_back(std::move(vCurrChunk));

              mAddText(vCtx->lTokSliImgEnd);
              if (!cIsLastInRow) {
                mAddText(vCtx->lTokSliImgMid);
              }
            }
            if ((vY != cNRow - 1 || vCtx->vTokRowEndTrail)) {
              mAddText(vCtx->lTokRowEnd);
            }
          }
        }

        // add overview image (last)
        if (!vCtx->vOvImgFirst) {
          mAddText(vCtx->lTokOvImgStart);
          vCur.lEntries.emplace_back(std::move(vOvChunk));
          mAddText(vCtx->lTokOvImgEnd);
        }
      } else {

        if (vPreprocOut.lEntries.size() == 0) {
          LOG_ERR("%s: no image tokens produced by preprocessor\n", __func__);
          return 2;
        }

        size_t vNTokens = 0;
        for (auto &vE : vPreprocOut.lEntries) {
          vNTokens += fClipNOutputTokens(vCtx->vCtxV, &vE);
          if (fClipModelNTemporalMerge(vCtx->vCtxV) == 2) {
            // [QWEN_VIDEO] pair input is merged to the same embd, so only count as one image
            break;
          }
        }

        mtmd_image_tokens_ptr vImageTokens(new mtmd_image_tokens);

        // [QWEN_VIDEO] improve this in the future
        vImageTokens->vNTemporalMerge = fClipModelNTemporalMerge(vCtx->vCtxV);

        if (mtmd_decode_use_mrope(vCtx)) {
          // for Qwen2VL, we need this information for M-RoPE decoding positions
          vImageTokens->vNx = fClipNOutputTokensX(vCtx->vCtxV, &vPreprocOut.lEntries[0]);
          vImageTokens->vNy = fClipNOutputTokensY(vCtx->vCtxV, &vPreprocOut.lEntries[0]);
        } else {
          // other models, we only need the total number of tokens
          vImageTokens->vNx = vNTokens;
          vImageTokens->vNy = 1;
        }
        vImageTokens->vPos = vCtx->vPosType;
        // HunyuanVL wraps the image grid with BOI/EOI and adds one newline per row,
        // and uses XD-RoPE (dim-3 = image index). Override the position type so that
        // n_tokens() and mtmd_image_tokens_get_decoder_pos pick the HunyuanVL layout.
        if (vCtx->mProjTypeV() == cProjectorTypeHunyuanvl) {
          vImageTokens->vPos = cMtmdPosTypeHunyuanvl;
          vImageTokens->vImageIdx = vNImagesAdded;
          GGML_ASSERT(vNTokens == (size_t)vImageTokens->mNTokens());
        }

        clip_image_f32_batch vBatchF32;
        vBatchF32.vIsAudio = false;
        vBatchF32.lEntries = std::move(vPreprocOut.lEntries);
        // do NOT use preproc_out from this point on, it's moved

        vImageTokens->vBatchF32 = std::move(vBatchF32);
        vImageTokens->vId = pBitmaps[0]->vId; // optional

        LOG_DBG("image_tokens->nx = %d\n", vImageTokens->vNx);
        LOG_DBG("image_tokens->ny = %d\n", vImageTokens->vNy);
        LOG_DBG("batch_f32 size = %d\n", (int)vImageTokens->vBatchF32.lEntries.size());

        mtmd_input_chunk vChunk{
            MTMD_INPUT_CHUNK_TYPE_IMAGE,
            {}, // text tokens
            std::move(vImageTokens),
            nullptr, // audio tokens
        };
        vCur.lEntries.emplace_back(std::move(vChunk));
      }

      if (!vCtx->vImgEnd.empty()) {
        mAddText(vCtx->vImgEnd, true); // add image end token
      }

      // advance image-chunk counter so the next image gets the next XD-RoPE dim-3 slot
      vNImagesAdded++;

    } else {
      // handle audio

      GGML_ASSERT(pBitmaps.size() == 1); // no batching support for now
      auto &vBitmap = pBitmaps[0];

      if (!vCtx->vCtxA) {
        LOG_ERR("%s: error: model does not support audio input\n", __func__);
        return 2;
      }

      if (vBitmap->vNx == 0) {
        LOG_ERR("%s: error: empty audio data\n", __func__);
        return 2;
      }

      if (!vCtx->vAudBeg.empty()) {
        mAddText(vCtx->vAudBeg, true); // add audio begin token
      }

      // sanity check
      GGML_ASSERT(vCtx->vAudioPreproc != nullptr);

      // preprocess audio
      std::vector<mtmd_audio_mel> lMelSpecChunks;
      {
        std::vector<float> lDummy;
        const float *vSamples = nullptr;
        size_t vNSamples = 0;
        if (vBitmap->mIsPlaceholder()) {
          // TODO @ngxson : skip underlay processing if bitmap is placeholder
          GGML_ASSERT(vBitmap->vNy == 1);

          lDummy.resize(vBitmap->vNx);
          vSamples = lDummy.data();
          vNSamples = lDummy.size();
        } else {
          const auto &vBuf = vBitmap->mGetRoBuf();
          GGML_ASSERT(vBuf.size() > sizeof(float));
          GGML_ASSERT(vBuf.size() % sizeof(float) == 0);

          vSamples = (const float *)vBuf.data();
          vNSamples = vBuf.size() / sizeof(float);
        }
        bool vOk = vCtx->vAudioPreproc->mPreprocess(vSamples, vNSamples, lMelSpecChunks);
        if (!vOk) {
          LOG_ERR("Unable to preprocess audio\n");
          return 2;
        }
      }

      // consider each mel_spec as a separate audio chunk
      // TODO: maybe support batching, but this may come with memory cost
      for (auto &vMelSpec : lMelSpecChunks) {
        const bool cIsPlaceholder = vMelSpec.lData.empty();

        // Validate dimensions fit in clip_image_size (int)
        GGML_ASSERT(vMelSpec.vNLen <= INT32_MAX && vMelSpec.vNLen >= 0);
        GGML_ASSERT(vMelSpec.vNMel <= INT32_MAX && vMelSpec.vNMel >= 0);
        clip_image_f32 vMelF32;
        vMelF32.mSetSize({(int)vMelSpec.vNLen, (int)vMelSpec.vNMel}, cIsPlaceholder, /* is_audio */ true);
        vMelF32.mCpyBuf(vMelSpec.lData);

        size_t vNTokens = fClipNOutputTokens(vCtx->vCtxA, &vMelF32);

        clip_image_f32_batch vBatchF32;
        vBatchF32.vIsAudio = true;
        vBatchF32.lEntries.push_back(std::move(vMelF32));

        mtmd_audio_tokens_ptr vAudioTokens(new mtmd_audio_tokens);
        vAudioTokens->vNTokens = vNTokens;
        vAudioTokens->vBatchF32 = std::move(vBatchF32);
        vAudioTokens->vId = vBitmap->vId; // optional

        LOG_DBG("audio_tokens->n_tokens = %d\n", vAudioTokens->vNTokens);

        mtmd_input_chunk vChunk{
            MTMD_INPUT_CHUNK_TYPE_AUDIO,
            {},      // text tokens
            nullptr, // image tokens
            std::move(vAudioTokens),
        };
        vCur.lEntries.emplace_back(std::move(vChunk));
      }

      if (!vCtx->vAudEnd.empty()) {
        mAddText(vCtx->vAudEnd, true); // add audio end token
      }
    }

    return 0;
  }

  std::vector<mtmd_input_chunk> mSplitBatchToChunk(mtmd_image_preproc_out &&pPreprocOut, const std::string &pId) {
    std::vector<mtmd_input_chunk> lChunks;

    auto vProcessChunk = [&](clip_image_f32 &&pImg) {
      mtmd_image_tokens_ptr vImageTokens(new mtmd_image_tokens);
      vImageTokens->vNx = fClipNOutputTokens(vCtx->vCtxV, &pImg);
      vImageTokens->vNy = 1;
      vImageTokens->vBatchF32.lEntries.push_back(std::move(pImg));
      vImageTokens->vId = pId;

      GGML_ASSERT(vImageTokens->vNx > 0);

      mtmd_input_chunk vChunk{
          MTMD_INPUT_CHUNK_TYPE_IMAGE,
          {}, // text tokens
          std::move(vImageTokens),
          nullptr, // audio tokens
      };
      lChunks.emplace_back(std::move(vChunk));
    };

    // overview image first
    auto &vOverview = pPreprocOut.vOverview;
    if (vOverview.mNx() == 0 || vOverview.mNy() == 0) {
      throw std::runtime_error(
          fStringFormat("%s: invalid overview image for llava-uhd style preprocessing\n", __func__));
    }
    vProcessChunk(std::move(pPreprocOut.vOverview));

    // then, process slices
    for (auto &vEntry : pPreprocOut.lEntries) {
      if (vEntry.mNx() == 0 || vEntry.mNy() == 0) {
        throw std::runtime_error(
            fStringFormat("%s: invalid image slice for llava-uhd style preprocessing\n", __func__));
      }
      vProcessChunk(std::move(vEntry));
    }

    return lChunks;
  }

  // for example: "a <__media__> b <__media__> c" --> "a", "<__media__>", "b", "<__media__>", "c"
  static std::vector<std::string> mSplitText(const std::string &pInput, const std::string &pDelimiter) {
    std::vector<std::string> lResult;
    if (pInput.empty()) {
      return lResult;
    }
    size_t vStart = 0;
    size_t vPos = 0;
    while ((vPos = pInput.find(pDelimiter, vStart)) != std::string::npos) {
      if (vPos > vStart) {
        lResult.push_back(pInput.substr(vStart, vPos - vStart));
      }
      lResult.push_back(pDelimiter);
      vStart = vPos + pDelimiter.length();
    }
    if (vStart < pInput.length()) {
      lResult.push_back(pInput.substr(vStart));
    }
    return lResult;
  }

  // copied from common_tokenize
  static std::vector<mgs_token> mMtmdTokenizeTextInternal(const struct mgs_vocab *pVocab, const std::string &pText,
                                                          bool pAddSpecial, bool pParseSpecial) {
    if (pVocab == nullptr) {
      throw std::runtime_error("mgs_vocab is not provided");
    }
    // upper limit for the number of tokens
    int vNTokens = pText.length() + 2 * pAddSpecial;
    std::vector<mgs_token> lResult(vNTokens);
    vNTokens =
        mgs_tokenize(pVocab, pText.data(), pText.length(), lResult.data(), lResult.size(), pAddSpecial, pParseSpecial);
    if (vNTokens == std::numeric_limits<int32_t>::min()) {
      throw std::runtime_error("Tokenization failed: input text too large, tokenization result exceeds int32_t limit");
    }
    if (vNTokens < 0) {
      lResult.resize(-vNTokens);
      int vCheck = mgs_tokenize(pVocab, pText.data(), pText.length(), lResult.data(), lResult.size(), pAddSpecial,
                                pParseSpecial);
      GGML_ASSERT(vCheck == -vNTokens);
    } else {
      lResult.resize(vNTokens);
    }
    return lResult;
  }
};

int32_t mtmd_tokenize(mtmd_context *pCtx, mtmd_input_chunks *pOutput, const mtmd_input_text *pText,
                      const mtmd_bitmap **pBitmaps, size_t pNBitmaps) {
  try {
    mtmd_tokenizer vTokenizer(pCtx, pText, pBitmaps, pNBitmaps);
    return vTokenizer.mTokenize(pOutput);
  } catch (const std::exception &vE) {
    LOG_ERR("%s: error: %s\n", __func__, vE.what());
    return 2;
  }
}

static int32_t mtmd_encode_impl(mtmd_context *pCtx, const mtmd_image_tokens *pImageTokens,
                                std::vector<float> &pOutEmbd) {
  clip_ctx *vCtxClip = pCtx->vCtxV;
  if (!vCtxClip) {
    LOG_ERR("%s: this API does not support non-vision input, please use mtmd_encode_chunk instead\n", __func__);
    return 1;
  }

  int vNEmbdOut = pCtx->mNEmbdOut();
  auto vNTokensOut = pImageTokens->mNTokens();
  pOutEmbd.resize((size_t)vNEmbdOut * vNTokensOut);

  if (pImageTokens->mIsPlaceholder()) {
    LOG_ERR("%s: image tokens batch is placeholder\n", __func__);
    return 1;
  }

  bool vOk = fClipImageBatchEncode(vCtxClip, pCtx->vNThreads, &pImageTokens->vBatchF32, pOutEmbd);

  return vOk ? 0 : 1;
}

static int32_t mtmd_encode_chunk_impl(mtmd_context *pCtx, const mtmd_input_chunk *pChunk,
                                      std::vector<float> &pOutEmbd) {
  if (pChunk->vType == MTMD_INPUT_CHUNK_TYPE_TEXT) {
    LOG_WRN("mtmd_encode_chunk has no effect for text chunks\n");
    return 0;
  } else if (pChunk->vType == MTMD_INPUT_CHUNK_TYPE_IMAGE) {
    if (!pCtx->vCtxV) {
      LOG_ERR("%s: model does not support vision input\n", __func__);
      return 1;
    }
    if (pChunk->vTokensImage == nullptr) {
      LOG_ERR("%s: image tokens are null\n", __func__);
      return 1;
    }
    if (pChunk->vTokensImage->mIsPlaceholder()) {
      LOG_ERR("%s: image tokens batch is placeholder\n", __func__);
      return 1;
    }
    return mtmd_encode_impl(pCtx, pChunk->vTokensImage.get(), pOutEmbd);
  } else if (pChunk->vType == MTMD_INPUT_CHUNK_TYPE_AUDIO) {
    if (!pCtx->vCtxA) {
      LOG_ERR("%s: model does not support audio input\n", __func__);
      return 1;
    }
    if (pChunk->vTokensAudio == nullptr) {
      LOG_ERR("%s: audio tokens are null\n", __func__);
      return 1;
    }
    if (pChunk->vTokensAudio->mIsPlaceholder()) {
      LOG_ERR("%s: audio tokens batch is placeholder\n", __func__);
      return 1;
    }
    int vNMmprojEmbd = pCtx->mNEmbdOut();
    pOutEmbd.resize((size_t)pChunk->vTokensAudio->vNTokens * vNMmprojEmbd);
    bool vOk = fClipImageBatchEncode(pCtx->vCtxA, pCtx->vNThreads, &pChunk->vTokensAudio->vBatchF32, pOutEmbd);
    return vOk ? 0 : 1;
  }

  LOG_ERR("%s: unknown chunk type %d\n", __func__, (int)pChunk->vType);
  return 1;
}

int32_t mtmd_encode_chunk(mtmd_context *pCtx, const mtmd_input_chunk *pChunk) {
  // this is the non-batching version
  try {
    return mtmd_encode_chunk_impl(pCtx, pChunk, pCtx->lOutEmbd);
  } catch (const std::exception &vE) {
    LOG_ERR("%s: error: %s\n", __func__, vE.what());
    return 1;
  }
}

float *mtmd_get_output_embd(mtmd_context *pCtx) { return pCtx->lOutEmbd.data(); }

mtmd_batch *mtmd_batch_init(mtmd_context *pCtx) { return new mtmd_batch(pCtx); }

void mtmd_batch_free(mtmd_batch *pBatch) {
  if (pBatch) {
    delete pBatch;
  }
}

int32_t mtmd_batch_add_chunk(mtmd_batch *pBatch, const mtmd_input_chunk *pChunk) {
  if (pChunk->vType == MTMD_INPUT_CHUNK_TYPE_TEXT) {
    LOG_ERR("%s: text chunk is not supported in batch\n", __func__);
    return 1;
  }

  auto *vCtx = pBatch->vCtx->mGetClipCtx(pChunk);
  if (!vCtx) {
    LOG_ERR("%s: model does not support input chunk type %d\n", __func__, (int)pChunk->vType);
    return 1;
  }

  if (pBatch->lEntries.empty()) {
    // batch must have at least one chunk
    pBatch->lEntries.push_back(pChunk);
    return 0;
  }

  if (!fClipSupportBatch(vCtx)) {
    // if no batching support, batch can only have one single chunk
    return 2; // "batch too large" error code
  }

  int32_t vNewNTokens = pBatch->mNTokens() + (int32_t)mtmd_input_chunk_get_n_tokens(pChunk);
  if (vNewNTokens > pBatch->vCtx->vBatchMaxTokens) {
    return 2; // "batch too large" error code
  }

  auto &vFirstChunk = pBatch->lEntries[0];
  if (vFirstChunk->mCanBatchWith(*pChunk)) {
    pBatch->lEntries.push_back(pChunk);
    return 0;
  }

  return 3; // "cannot batch" error code
}

static int32_t mtmd_batch_encode_impl(mtmd_batch *pBatch) {
  if (pBatch->lEntries.empty()) {
    LOG_ERR("%s: batch is empty\n", __func__);
    return 1;
  }
  for (const auto *vChunk : pBatch->lEntries) {
    if (vChunk->mIsPlaceholder()) {
      LOG_ERR("%s: chunk is placeholder\n", __func__);
      return 1;
    }
  }

  // represent the whole batch as one single chunk
  mtmd::input_chunk_ptr vBatchChunk(mtmd_input_chunk_copy(pBatch->lEntries[0]));
  if (vBatchChunk->vTokensImage) {
    auto &vB0F32 = vBatchChunk->vTokensImage->vBatchF32;
    // copy all entries from other chunks into the first chunk's batch_f32
    // note: skip first entry because it's already in batch_chunk
    for (size_t vIc = 1; vIc < pBatch->lEntries.size(); vIc++) {
      auto &vChunk = pBatch->lEntries[vIc];
      GGML_ASSERT(vChunk->vTokensImage);
      auto vB1F32 = vChunk->vTokensImage->vBatchF32.mClone();
      for (size_t vI = 0; vI < vB1F32.lEntries.size(); vI++) {
        vB0F32.lEntries.push_back(std::move(vB1F32.lEntries[vI]));
      }
    }
  } else if (vBatchChunk->vTokensAudio) {
    auto &vB0F32 = vBatchChunk->vTokensAudio->vBatchF32;
    // copy all entries from other chunks into the first chunk's batch_f32
    // note: skip first entry because it's already in batch_chunk
    for (size_t vIc = 1; vIc < pBatch->lEntries.size(); vIc++) {
      auto &vChunk = pBatch->lEntries[vIc];
      GGML_ASSERT(vChunk->vTokensAudio);
      auto vB1F32 = vChunk->vTokensAudio->vBatchF32.mClone();
      for (size_t vI = 0; vI < vB1F32.lEntries.size(); vI++) {
        vB0F32.lEntries.push_back(std::move(vB1F32.lEntries[vI]));
      }
    }
  } else {
    LOG_ERR("%s: unsupported chunk type\n", __func__);
    return 1;
  }

  LOG_DBG("%s: encoding batch with %zu entries and total %zu tokens\n", __func__, pBatch->lEntries.size(),
          mtmd_input_chunk_get_n_tokens(vBatchChunk.get()));
  int32_t vRes = mtmd_encode_chunk_impl(pBatch->vCtx, vBatchChunk.get(), pBatch->lOutputEmbd);
  return vRes;
}

int32_t mtmd_batch_encode(mtmd_batch *pBatch) {
  try {
    return mtmd_batch_encode_impl(pBatch);
  } catch (const std::exception &vE) {
    LOG_ERR("%s: error: %s\n", __func__, vE.what());
    return 1;
  }
}

float *mtmd_batch_get_output_embd(mtmd_batch *pBatch, const mtmd_input_chunk *pChunk) {
  if (pBatch->lOutputEmbd.empty()) {
    LOG_ERR("%s: batch has not been encoded yet\n", __func__);
    return nullptr;
  }
  size_t vOffset = 0;
  const size_t cNEmbd = pBatch->vCtx->mNEmbdOut();
  for (const auto *vC : pBatch->lEntries) {
    size_t vOffsetPrev = vOffset;
    size_t vNTokens = mtmd_input_chunk_get_n_tokens(vC);
    vOffset += vNTokens * cNEmbd;
    GGML_ASSERT(vOffsetPrev < pBatch->lOutputEmbd.size());
    GGML_ASSERT(vOffset <= pBatch->lOutputEmbd.size());
    if (vC == pChunk) {
      return &pBatch->lOutputEmbd.data()[vOffsetPrev];
    }
  }
  return nullptr; // not found
}

bool mtmd_decode_use_non_causal(const mtmd_context *pCtx, const mtmd_input_chunk *pChunk) {
  auto vProjType = pCtx->mProjTypeV();
  if (pChunk && pChunk->vType == MTMD_INPUT_CHUNK_TYPE_AUDIO) {
    vProjType = pCtx->mProjTypeA();
  }
  switch (vProjType) {
  case cProjectorTypeGemma3:
    return true;
  default:
    return false;
  }
}

bool mtmd_decode_use_mrope(const mtmd_context *pCtx) { return pCtx->vPosType == cMtmdPosTypeMrope; }

bool mtmd_support_vision(const mtmd_context *pCtx) { return pCtx->vCtxV != nullptr; }

bool mtmd_support_audio(const mtmd_context *pCtx) { return pCtx->vCtxA != nullptr; }

int mtmd_get_audio_sample_rate(const mtmd_context *pCtx) {
  if (!pCtx->vCtxA) {
    return -1;
  }
  return fClipGetHparams(pCtx->vCtxA)->vAudioSampleRate;
}

const char *mtmd_get_marker(const mtmd_context *pCtx) { return pCtx->vMediaMarker.c_str(); }

//
// public API functions
//

// mtmd_bitmap

mtmd_bitmap *mtmd_bitmap_init(uint32_t pNx, uint32_t pNy, const unsigned char *pData) {
  mtmd_bitmap *vBitmap = new mtmd_bitmap(pData, pNx, pNy);
  return vBitmap;
}

mtmd_bitmap *mtmd_bitmap_init_from_audio(size_t pNSamples, const float *pData) {
  mtmd_bitmap *vBitmap = new mtmd_bitmap((const unsigned char *)pData, pNSamples);
  GGML_ASSERT(vBitmap->vIsAudio);
  if (!vBitmap->mIsPlaceholder()) {
    GGML_ASSERT(vBitmap->mGetRoBuf().size() == pNSamples * sizeof(float));
  }
  return vBitmap;
}

uint32_t mtmd_bitmap_get_nx(const mtmd_bitmap *pBitmap) { return pBitmap->vNx; }

uint32_t mtmd_bitmap_get_ny(const mtmd_bitmap *pBitmap) { return pBitmap->vNy; }

const unsigned char *mtmd_bitmap_get_data(const mtmd_bitmap *pBitmap) {
  if (pBitmap->mIsPlaceholder()) {
    return nullptr;
  }
  return pBitmap->mGetRoBuf().data();
}

size_t mtmd_bitmap_get_n_bytes(const mtmd_bitmap *pBitmap) {
  if (pBitmap->mIsPlaceholder()) {
    return 0;
  }
  return pBitmap->mGetRoBuf().size();
}

bool mtmd_bitmap_is_audio(const mtmd_bitmap *pBitmap) { return pBitmap->vIsAudio; }

const char *mtmd_bitmap_get_id(const mtmd_bitmap *pBitmap) { return pBitmap->vId.c_str(); }

void mtmd_bitmap_set_id(mtmd_bitmap *pBitmap, const char *pId) {
  if (pId) {
    pBitmap->vId = std::string(pId);
  } else {
    pBitmap->vId.clear();
  }
}

mtmd_bitmap *mtmd_bitmap_init_lazy(mtmd_context *pCtx, const char *pId, void *pUserData,
                                   mtmd_bitmap_lazy_callback pCallback) {
  GGML_UNUSED(pCtx); // reserved for future use
  mtmd_bitmap *vBitmap = new mtmd_bitmap(nullptr, 0, 0);
  vBitmap->vLazyCallback = pCallback;
  vBitmap->vLazyUserData = pUserData;
  mtmd_bitmap_set_id(vBitmap, pId);
  return vBitmap;
}

void mtmd_bitmap_free(mtmd_bitmap *pBitmap) {
  if (pBitmap) {
    delete pBitmap;
  }
}

// mtmd_input_chunks

mtmd_input_chunks *mtmd_input_chunks_init() { return new mtmd_input_chunks; }

size_t mtmd_input_chunks_size(const mtmd_input_chunks *pChunks) { return pChunks->lEntries.size(); }

const mtmd_input_chunk *mtmd_input_chunks_get(const mtmd_input_chunks *pChunks, size_t pIdx) {
  if (pIdx >= pChunks->lEntries.size()) {
    return nullptr;
  }
  return &pChunks->lEntries[pIdx];
}

void mtmd_input_chunks_free(mtmd_input_chunks *pChunks) {
  if (pChunks) {
    delete pChunks;
  }
}

// mtmd_input_chunk

enum mtmd_input_chunk_type mtmd_input_chunk_get_type(const mtmd_input_chunk *pChunk) { return pChunk->vType; }

const mgs_token *mtmd_input_chunk_get_tokens_text(const mtmd_input_chunk *pChunk, size_t *pNTokensOutput) {
  if (pChunk->vType == MTMD_INPUT_CHUNK_TYPE_TEXT) {
    *pNTokensOutput = pChunk->lTokensText.size();
    return pChunk->lTokensText.data();
  }
  *pNTokensOutput = 0;
  return nullptr;
}

const mtmd_image_tokens *mtmd_input_chunk_get_tokens_image(const mtmd_input_chunk *pChunk) {
  if (pChunk->vType == MTMD_INPUT_CHUNK_TYPE_IMAGE) {
    return pChunk->vTokensImage.get();
  }
  return nullptr;
}

size_t mtmd_input_chunk_get_n_tokens(const mtmd_input_chunk *pChunk) {
  if (pChunk->vType == MTMD_INPUT_CHUNK_TYPE_TEXT) {
    return pChunk->lTokensText.size();
  } else if (pChunk->vType == MTMD_INPUT_CHUNK_TYPE_IMAGE) {
    return mtmd_image_tokens_get_n_tokens(pChunk->vTokensImage.get());
  } else if (pChunk->vType == MTMD_INPUT_CHUNK_TYPE_AUDIO) {
    return pChunk->vTokensAudio->vNTokens;
  } else {
    GGML_ABORT("invalid chunk type");
  }
}

mgs_pos mtmd_input_chunk_get_n_pos(const mtmd_input_chunk *pChunk) {
  if (pChunk->vType == MTMD_INPUT_CHUNK_TYPE_TEXT) {
    return pChunk->lTokensText.size();
  } else if (pChunk->vType == MTMD_INPUT_CHUNK_TYPE_IMAGE) {
    return mtmd_image_tokens_get_n_pos(pChunk->vTokensImage.get());
  } else if (pChunk->vType == MTMD_INPUT_CHUNK_TYPE_AUDIO) {
    return pChunk->vTokensAudio->vNTokens;
  } else {
    GGML_ABORT("invalid chunk type");
  }
}

const char *mtmd_input_chunk_get_id(const mtmd_input_chunk *pChunk) {
  if (pChunk->vType == MTMD_INPUT_CHUNK_TYPE_IMAGE) {
    return pChunk->vTokensImage->vId.c_str();
  } else if (pChunk->vType == MTMD_INPUT_CHUNK_TYPE_AUDIO) {
    return pChunk->vTokensAudio->vId.c_str();
  }
  return nullptr;
}

mtmd_input_chunk *mtmd_input_chunk_copy(const mtmd_input_chunk *pChunk) {
  mtmd_input_chunk *vCopy = new mtmd_input_chunk{
      pChunk->vType,
      pChunk->lTokensText,
      nullptr,
      nullptr,
  };
  if (pChunk->vTokensImage) {
    // copy the image tokens
    vCopy->vTokensImage = mtmd_image_tokens_ptr(new mtmd_image_tokens());
    *vCopy->vTokensImage = pChunk->vTokensImage->mClone();
  }
  if (pChunk->vTokensAudio) {
    // copy the audio tokens
    vCopy->vTokensAudio = mtmd_audio_tokens_ptr(new mtmd_audio_tokens());
    *vCopy->vTokensAudio = pChunk->vTokensAudio->mClone();
  }
  return vCopy;
}

void mtmd_input_chunk_free(mtmd_input_chunk *pChunk) {
  if (pChunk) {
    delete pChunk;
  }
}

// mtmd_image_tokens

size_t mtmd_image_tokens_get_n_tokens(const mtmd_image_tokens *pImageTokens) { return pImageTokens->mNTokens(); }

mtmd_decoder_pos mtmd_image_tokens_get_decoder_pos(const mtmd_image_tokens *pImageTokens, mgs_pos pPos0, size_t pI) {
  mtmd_decoder_pos vPos;
  switch (pImageTokens->vPos) {
  case cMtmdPosTypeMrope: {
    vPos.t = pPos0;
    vPos.x = pPos0 + (pI % pImageTokens->vNx);
    vPos.y = pPos0 + (pI / pImageTokens->vNx);
    vPos.z = 0; // unused for now
  } break;
  case cMtmdPosTypeNormal: {
    vPos.t = pPos0 + pI;
    vPos.x = pPos0 + pI;
    vPos.y = pPos0 + pI;
    vPos.z = pPos0 + pI;
  } break;
  case cMtmdPosTypeHunyuanvl: {
    // HunyuanVL layout: [BOI] [row0 tokens + newline] ... [row(ny-1) tokens + newline] [EOI]
    // Total = 1 + ny*(nx+1) + 1. BOI and EOI use sequential positions in every dim;
    // content and row-newline tokens use (row, col) with XD-RoPE dim-3 = image_idx.
    const uint32_t cNx = pImageTokens->vNx;
    const uint32_t cNTotal = pImageTokens->mNTokens();
    if (pI == 0) {
      // BOI
      vPos.t = pPos0 + pI;
      vPos.x = pPos0 + pI;
      vPos.y = pPos0 + pI;
      vPos.z = pPos0 + pI;
    } else if (pI == cNTotal - 1) {
      // EOI
      vPos.t = pPos0 + pI;
      vPos.x = pPos0 + pI;
      vPos.y = pPos0 + pI;
      vPos.z = pPos0 + pI;
    } else {
      // content token at (row, col), or the trailing newline of a row (col == nx)
      //   section 0 = sequential, section 1 = w(col), section 2 = h(row), section 3 = image_count.
      // set_position_mrope_2d writes .y -> section 1 and .x -> section 2
      const uint32_t cOffset = (uint32_t)pI - 1;
      const uint32_t cRow = cOffset / (cNx + 1);
      const uint32_t cCol = cOffset % (cNx + 1);
      vPos.t = pPos0 + pI;
      vPos.x = cRow;
      vPos.y = cCol;
      vPos.z = pImageTokens->vImageIdx;
    }
  } break;
  default:
    GGML_ABORT("invalid position type");
  }
  return vPos;
}

mgs_pos mtmd_image_tokens_get_n_pos(const mtmd_image_tokens *pImageTokens) {
  switch (pImageTokens->vPos) {
  case cMtmdPosTypeMrope:
    return std::max(pImageTokens->vNx, pImageTokens->vNy);
  case cMtmdPosTypeNormal:
    return pImageTokens->mNTokens();
  case cMtmdPosTypeHunyuanvl:
    // HunyuanVL: the sequential (dim-0) position advances by the full token count
    // (includes BOI/EOI and row newline tokens), not by max(nx, ny)
    return pImageTokens->mNTokens();
  default:
    GGML_ABORT("invalid position type");
  }
}

// test function

mtmd_input_chunks *mtmd_test_create_input_chunks() {
  mtmd_input_chunks *vChunks = mtmd_input_chunks_init();
  if (!vChunks) {
    return nullptr;
  }

  // create a text chunk
  std::vector<mgs_token> lTokensText = {1, 2, 3, 4, 5};
  mtmd_input_chunk vChunkText{
      MTMD_INPUT_CHUNK_TYPE_TEXT,
      std::move(lTokensText),
      nullptr, // image tokens
      nullptr, // audio tokens
  };
  vChunks->lEntries.emplace_back(std::move(vChunkText));

  // create an image chunk
  mtmd_image_tokens_ptr vImageTokens(new mtmd_image_tokens);
  vImageTokens->vNx = 4;
  vImageTokens->vNy = 4;
  vImageTokens->vBatchF32.lEntries.resize(16);
  vImageTokens->vId = "image_1";
  mtmd_input_chunk vChunkImage{
      MTMD_INPUT_CHUNK_TYPE_IMAGE,
      {}, // text tokens
      std::move(vImageTokens),
      nullptr, // audio tokens
  };
  vChunks->lEntries.emplace_back(std::move(vChunkImage));

  return vChunks;
}

void mtmd_log_set(ggml_log_callback pLogCallback, void *pUserData) {
  vGLoggerState.vLogCallback = pLogCallback ? pLogCallback : fClipLogCallbackDefault;
  vGLoggerState.vLogCallbackUserData = pUserData;
}

struct mtmd_caps mtmd_get_cap_from_file(const char *pFname) {
  try {
    auto vTmp = fClipGetCap(pFname);
    mtmd_caps vCap;
    vCap.inp_audio = vTmp.vHasAudio;
    vCap.inp_vision = vTmp.vHasVision;
    return vCap;
  } catch (const std::exception &vE) {
    LOG_ERR("%s: failed to get capabilities from file '%s': %s\n", __func__, pFname, vE.what());
    return mtmd_caps{false, false};
  }
}

//
// Debugging API (NOT intended for public use)
//

static void mtmd_debug_encode_impl(mtmd_context *pCtx, clip_ctx *pCtxClip, clip_image_f32 &pImage) {
  fClipSetDebugOutputEmbeddings(pCtxClip, true);
  int vNMmprojEmbd = fClipNMmprojEmbd(pCtxClip);
  int vNTokens = fClipNOutputTokens(pCtxClip, &pImage);
  std::vector<float> lEmbdOutput(vNTokens * vNMmprojEmbd, 0.0f);
  bool vOk = fClipImageEncode(pCtxClip, pCtx->vNThreads, &pImage, lEmbdOutput);
  if (!vOk) {
    LOG_ERR("%s: failed to encode image\n", __func__);
  }
}

void mtmd_debug_encode_image(mtmd_context *pCtx, const std::vector<std::vector<float>> &pImage) {
  if (!pCtx->vCtxV) {
    LOG_ERR("%s: model does not support vision input\n", __func__);
    return;
  }
  const int cImgSz = (int)pImage.size();
  std::vector<float> lImgBuf;
  lImgBuf.reserve(cImgSz * cImgSz);
  for (const auto &vRow : pImage) {
    lImgBuf.insert(lImgBuf.end(), vRow.begin(), vRow.end());
  }
  clip_image_f32 vInpImage;
  vInpImage.mSetSize({cImgSz, cImgSz}, false, false);
  vInpImage.mCpyBuf(lImgBuf);
  LOG_INF("%s: created input image with nx=%d, ny=%d\n", __func__, cImgSz, cImgSz);
  mtmd_debug_encode_impl(pCtx, pCtx->vCtxV, vInpImage);
}

void mtmd_debug_encode_audio(mtmd_context *pCtx, const std::vector<float> &pInput) {
  if (!pCtx->vCtxA) {
    LOG_ERR("%s: model does not support audio input\n", __func__);
    return;
  }
  int vNMel = fClipGetHparams(pCtx->vCtxA)->vNMelBins;
  const int cAudioNx = (int)pInput.size();
  std::vector<float> lAudioBuf(cAudioNx * vNMel);
  for (int vI = 0; vI < cAudioNx; vI++) {
    for (int vJ = 0; vJ < vNMel; vJ++) {
      lAudioBuf[vJ * cAudioNx + vI] = pInput[vI];
    }
  }
  clip_image_f32 vInpAudio;
  vInpAudio.mSetSize({cAudioNx, vNMel}, false, true);
  vInpAudio.mCpyBuf(lAudioBuf);
  LOG_INF("%s: created input audio with nx=%d, ny=%d\n", __func__, cAudioNx, vNMel);
  mtmd_debug_encode_impl(pCtx, pCtx->vCtxA, vInpAudio);
}

void mtmd_debug_preprocess_image(mtmd_context *pCtx, const std::vector<uint8_t> &pRgbValues, int pNx, int pNy) {
  if (!pCtx->vCtxV) {
    LOG_ERR("%s: model does not support vision input\n", __func__);
    return;
  }
  clip_image_u8 vImgU8;
  vImgU8.mSetSize({pNx, pNy}, false);
  vImgU8.mCpyBuf(pRgbValues);
  GGML_ASSERT(pCtx->vImagePreproc != nullptr);
  mtmd_image_preproc_out vPreprocOut = pCtx->vImagePreproc->mPreprocess(vImgU8);

  clip_image_f32_batch vBatchF32;
  vBatchF32.vIsAudio = false;
  for (auto &vEntry : vPreprocOut.lEntries) {
    vBatchF32.lEntries.push_back(std::move(vEntry));
  }

  LOG_INF("%s: preprocessed image to batch_f32 with %d entries\n", __func__, (int)vBatchF32.lEntries.size());
  for (size_t vI = 0; vI < vBatchF32.lEntries.size(); vI++) {
    LOG_INF("%s: entry %zu has nx=%d, ny=%d\n", __func__, vI, vBatchF32.lEntries[vI].mNx(),
            vBatchF32.lEntries[vI].mNy());
    // TODO: better way to dump entry content?
  }
}

void mtmd_debug_preprocess_audio(mtmd_context *pCtx, const std::vector<float> &pSamples) {
  if (!pCtx->vCtxA) {
    LOG_ERR("%s: model does not support audio input\n", __func__);
    return;
  }
  std::vector<mtmd_audio_mel> lMelSpecChunks;
  bool vOk = pCtx->vAudioPreproc->mPreprocess(pSamples.data(), pSamples.size(), lMelSpecChunks);
  if (!vOk) {
    LOG_ERR("%s: failed to preprocess audio\n", __func__);
    return;
  }
  LOG_INF("%s: preprocessed audio to %zu mel spec chunks\n", __func__, lMelSpecChunks.size());
  for (size_t vI = 0; vI < lMelSpecChunks.size(); vI++) {
    LOG_INF("%s: mel spec chunk %zu has n_len=%d, n_mel=%d\n", __func__, vI, lMelSpecChunks[vI].vNLen,
            lMelSpecChunks[vI].vNMel);

    // dump mel entries: data is stored as [n_mel][n_len] (mel-major)
    const auto &vMel = lMelSpecChunks[vI];
    for (int vM = 0; vM < vMel.vNMel; vM++) {
      for (int vT = 0; vT < vMel.vNLen; vT++) {
        LOG_INF("mel[%zu][m=%d][t=%d] = %f\n", vI, vM, vT, vMel.lData[vM * vMel.vNLen + vT]);
      }
    }
  }
}

static void fStubLogCallback(enum ggml_log_level, const char *, void *) {
  // do nothing
}

std::map<ggml_backend_dev_t, size_t> mtmd_get_memory_usage(const char *pMmprojFname,
                                                           struct mtmd_context_params pCtxParams) {
  mtmd::context_ptr vCtx;
  auto vSavedLogCallback = vGLoggerState.vLogCallback;
  auto vSavedLogUserData = vGLoggerState.vLogCallbackUserData;

  pCtxParams.progress_callback = nullptr;

  try {
    mtmd_log_set(fStubLogCallback, nullptr); // suppress logging
    vCtx.reset(new mtmd_context(pMmprojFname, nullptr, pCtxParams, true));
    mtmd_log_set(vSavedLogCallback, vSavedLogUserData); // restore log callback
    std::map<ggml_backend_dev_t, size_t> dTotalMem;
    auto vMerge = [&](const struct clip_ctx *pC) {
      for (auto &[vDev, vSize] : fClipGetMemUsage(pC)) {
        dTotalMem[vDev] += vSize;
      }
    };
    if (vCtx->vCtxV) {
      vMerge(vCtx->vCtxV);
    }
    if (vCtx->vCtxA) {
      vMerge(vCtx->vCtxA);
    }
    return dTotalMem;
  } catch (const std::exception &vE) {
    mtmd_log_set(vSavedLogCallback, vSavedLogUserData); // restore log callback
    LOG_ERR("%s: error: %s\n", __func__, vE.what());
    return {};
  }
}
