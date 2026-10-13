// fix problem with std::min and std::max
#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

#include "mtmd.h"
#include "mtmd-helper.h"
#include "mgs.h"

#include <algorithm>
#include <cinttypes>
#include <vector>

// #define MTMD_AUDIO_DEBUG

#define MINIAUDIO_IMPLEMENTATION
#ifndef MTMD_AUDIO_DEBUG
#define MA_NO_ENCODING
#endif
#define MA_NO_DEVICE_IO
#define MA_NO_RESOURCE_MANAGER
#define MA_NO_NODE_GRAPH
#define MA_NO_ENGINE
#define MA_NO_GENERATION
#define MA_API static
#include "miniaudio/miniaudio.h"

// stb_image refuses wider or taller images on its own, as a second guard behind cMediaMaxImageSide
#define STBI_MAX_DIMENSIONS (1 << 15)
#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"

#ifdef MTMD_INTERNAL_HEADER
#error "mtmd-helper is a public library outside of mtmd. it must not include internal headers"
#endif

#ifdef MTMD_VIDEO
#include "sheredom/subprocess.h"
#include <thread>
#endif

//
// internal logging functions
//

struct mtmd_helper_logger {
  ggml_log_callback vDefaultCallback = [](ggml_log_level pLevel, const char *pText, void *pUserData) {
    (void)pLevel;
    (void)pUserData;
    fputs(pText, stderr);
    fflush(stderr);
  };

  ggml_log_callback vLogCallback = vDefaultCallback;
  void *vLogCallbackUserData;

  void mLogV(enum ggml_log_level pLevel, const char *pFormat, va_list pArgs) {
    if (pFormat == NULL) {
      return;
    }
    va_list vArgsCopy;
    va_copy(vArgsCopy, pArgs);
    char aBuffer[128];
    int vLen = vsnprintf(aBuffer, 128, pFormat, pArgs);
    if (vLen < 128) {
      vLogCallback(pLevel, aBuffer, vLogCallbackUserData);
    } else {
      char *vBuffer2 = (char *)calloc(vLen + 1, sizeof(char));
      vsnprintf(vBuffer2, vLen + 1, pFormat, vArgsCopy);
      vBuffer2[vLen] = 0;
      vLogCallback(pLevel, vBuffer2, vLogCallbackUserData);
      free(vBuffer2);
    }
    va_end(vArgsCopy);
  }

  void mLog(enum ggml_log_level pLevel, const char *pFormat, ...) {
    va_list vArgs;
    va_start(vArgs, pFormat);
    mLogV(pLevel, pFormat, vArgs);
    va_end(vArgs);
  }
} vGLogger;

#define LOG_DBG(...) vGLogger.mLog(GGML_LOG_LEVEL_DEBUG, __VA_ARGS__)
#define LOG_INF(...) vGLogger.mLog(GGML_LOG_LEVEL_INFO, __VA_ARGS__)
#define LOG_WRN(...) vGLogger.mLog(GGML_LOG_LEVEL_WARN, __VA_ARGS__)
#define LOG_ERR(...) vGLogger.mLog(GGML_LOG_LEVEL_ERROR, __VA_ARGS__)

void mtmd_helper_log_set(ggml_log_callback pLogCallback, void *pUserData) {
  if (pLogCallback == nullptr) {
    pLogCallback = vGLogger.vDefaultCallback;
  }
  vGLogger.vLogCallback = pLogCallback;
  vGLogger.vLogCallbackUserData = pUserData;
  mtmd_log_set(pLogCallback, pUserData);
}

//
// decoding limits
//

// media can come from untrusted clients and a small compressed file can expand to a huge decoded representation, so
// these limits are checked before anything is decoded; they are generous enough for any legitimate input
static constexpr int cMediaMaxImageSide = 1 << 15;                    // pixels per side, as STBI_MAX_DIMENSIONS
static constexpr uint64_t cMediaMaxImagePixels = 100ull * 1000 * 1000; // 100 megapixels, 300 MB as RGB
static constexpr uint64_t cMediaMaxAudioSeconds = 60 * 60;             // one hour at the model's sample rate
static constexpr uint32_t cMediaMaxVideoSide = 8192;                   // pixels per side of a video frame
static constexpr uint64_t cMediaMaxVideoDecodedBytes = 2ull << 30;     // raw RGB frames of one video, 2 GiB

//
// helper functions
//

size_t mtmd_helper_get_n_tokens(const mtmd_input_chunks *pChunks) {
  size_t vNTokens = 0;
  for (size_t vI = 0; vI < mtmd_input_chunks_size(pChunks); vI++) {
    auto vChunk = mtmd_input_chunks_get(pChunks, vI);
    vNTokens += mtmd_input_chunk_get_n_tokens(vChunk);
  }
  return vNTokens;
}

mgs_pos mtmd_helper_get_n_pos(const mtmd_input_chunks *pChunks) {
  mgs_pos vNPos = 0;
  for (size_t vI = 0; vI < mtmd_input_chunks_size(pChunks); vI++) {
    auto vChunk = mtmd_input_chunks_get(pChunks, vI);
    vNPos += mtmd_input_chunk_get_n_pos(vChunk);
  }
  return vNPos;
}

void mtmd_helper_image_get_decoder_pos(const mtmd_image_tokens *pChunks, mgs_pos pPos0, mtmd_decoder_pos *pOutPos) {
  size_t vNTokens = mtmd_image_tokens_get_n_tokens(pChunks);
  for (size_t vI = 0; vI < vNTokens; vI++) {
    pOutPos[vI] = mtmd_image_tokens_get_decoder_pos(pChunks, pPos0, vI);
  }
}

// helper struct to make working with embd batch easier
// note: this will be removed after mgs_batch_ext refactoring
struct decode_embd_batch {
  int vNPosPerEmbd;
  int vNMmprojEmbd;
  std::vector<mgs_pos> lPos;
  std::vector<mgs_pos> lPosView; // used by mrope
  std::vector<int32_t> lNSeqId;
  std::vector<mgs_seq_id> lSeqId0;
  std::vector<mgs_seq_id *> lSeqIds;
  std::vector<int8_t> lLogits;
  mgs_batch vBatch;
  decode_embd_batch(float *pEmbd, int32_t pNTokens, int pNPosPerEmbd, int pNMmprojEmbd)
      : vNPosPerEmbd(pNPosPerEmbd), vNMmprojEmbd(pNMmprojEmbd) {
    GGML_ASSERT(pNTokens > 0 && pNPosPerEmbd > 0 && pNMmprojEmbd > 0);
    lPos.resize(pNTokens * pNPosPerEmbd);
    lNSeqId.resize(pNTokens);
    lSeqIds.resize(pNTokens + 1);
    lLogits.resize(pNTokens);
    lSeqId0.resize(1);
    lSeqIds[pNTokens] = nullptr;
    vBatch = {
        /*n_tokens       =*/pNTokens,
        /*tokens         =*/nullptr,
        /*embd           =*/pEmbd,
        /*pos            =*/lPos.data(),
        /*n_seq_id       =*/lNSeqId.data(),
        /*seq_id         =*/lSeqIds.data(),
        /*logits         =*/lLogits.data(),
    };
  }

  void mSetPositionNormal(mgs_pos pPos0, mgs_seq_id pSeqId) {
    lSeqId0[0] = pSeqId;
    for (int vI = 0; vI < vBatch.n_tokens; vI++) {
      vBatch.pos[vI] = pPos0 + vI;
      vBatch.n_seq_id[vI] = 1;
      vBatch.seq_id[vI] = lSeqId0.data();
      vBatch.logits[vI] = false;
    }
  }

  // M-RoPE for image
  void mSetPositionMrope2d(const std::vector<mtmd_decoder_pos> &pRelPos, mgs_seq_id pSeqId) {
    GGML_ASSERT(vNPosPerEmbd == 4);
    GGML_ASSERT(!pRelPos.empty() && (int32_t)pRelPos.size() == vBatch.n_tokens);
    lSeqId0[0] = pSeqId;
    for (int32_t vI = 0; vI < vBatch.n_tokens; vI++) {
      lPos[vI] = pRelPos[vI].t;
      lPos[vI + vBatch.n_tokens] = pRelPos[vI].y;
      lPos[vI + vBatch.n_tokens * 2] = pRelPos[vI].x;
      lPos[vI + vBatch.n_tokens * 3] = pRelPos[vI].z;
    }
    for (int vI = 0; vI < vBatch.n_tokens; vI++) {
      vBatch.n_seq_id[vI] = 1;
      vBatch.seq_id[vI] = lSeqId0.data();
      vBatch.logits[vI] = false;
    }
  }

  // M-RoPE for audio
  void mSetPositionMrope1d(mgs_pos pPos0, mgs_seq_id pSeqId) {
    GGML_ASSERT(vNPosPerEmbd == 4);
    lSeqId0[0] = pSeqId;
    for (int vI = 0; vI < vBatch.n_tokens; vI++) {
      lPos[vI] = pPos0 + vI;
      lPos[vI + vBatch.n_tokens] = pPos0 + vI;
      lPos[vI + vBatch.n_tokens * 2] = pPos0 + vI;
      lPos[vI + vBatch.n_tokens * 3] = pPos0 + vI;
    }
    for (int vI = 0; vI < vBatch.n_tokens; vI++) {
      vBatch.n_seq_id[vI] = 1;
      vBatch.seq_id[vI] = lSeqId0.data();
      vBatch.logits[vI] = false;
    }
  }

  mgs_batch mGetView(int pOffset, int pNTokens) {
    GGML_ASSERT(pOffset >= 0 && pNTokens > 0 && pOffset + pNTokens <= vBatch.n_tokens);
    mgs_pos *vPosPtr;
    lPosView.clear();
    lPosView.reserve(pNTokens * vNPosPerEmbd);
    if (vNPosPerEmbd > 1) {
      // mrope
      // for example, with layout of src: 1234...1234...1234...1234...
      //       offset 2 will give us dst: 34...34...34...34...
      for (int vI = 0; vI < vNPosPerEmbd; vI++) {
        // assume n_tokens is less than or equal to batch.n_tokens
        // batch.n_tokens is number of **total** tokens
        // n_tokens is number of viewed token
        size_t vSrcIdx = vI * vBatch.n_tokens + pOffset;
        lPosView.insert(lPosView.end(), lPos.data() + vSrcIdx, lPos.data() + vSrcIdx + pNTokens);
      }
      vPosPtr = lPosView.data();
    } else {
      // normal
      vPosPtr = lPos.data() + pOffset;
    }
    return {
        /*n_tokens       =*/pNTokens,
        /*tokens         =*/nullptr,
        /*embd           =*/vBatch.embd + pOffset * vNMmprojEmbd,
        /*pos            =*/vPosPtr,
        /*n_seq_id       =*/vBatch.n_seq_id + pOffset,
        /*seq_id         =*/vBatch.seq_id + pOffset,
        /*logits         =*/vBatch.logits + pOffset,
    };
  }
};

// Helper class to set non-causal attention via RAII
class scope_non_causal {
public:
  scope_non_causal(mgs_context *pContext, bool pEnabled) : vContext(pContext), vEnabled(pEnabled) {
    if (vEnabled) {
      // TODO @ngxson : need to make sure only one image is processed at a time, and n_ubatch must be enough to hold the
      // image
      mgs_set_causal_attn(vContext, false);
    }
  }
  ~scope_non_causal() {
    if (vEnabled) {
      mgs_set_causal_attn(vContext, true);
    }
  }

  scope_non_causal(const scope_non_causal &) = delete;
  scope_non_causal &operator=(const scope_non_causal &) = delete;

private:
  mgs_context *vContext;
  bool vEnabled;
};

// Helper function for decoding an image whose embeddings have already been calculated
int32_t mtmd_helper_decode_image_chunk(mtmd_context *pCtx, struct mgs_context *pLctx, const mtmd_input_chunk *pChunk,
                                       float *pEncodedEmbd, mgs_pos pNPast, mgs_seq_id pSeqId, int32_t pNBatch,
                                       mgs_pos *pNewNPast, mtmd_helper_post_decode_callback pCallback,
                                       void *pUserData) {
  GGML_ASSERT(pNBatch > 0);
  auto vChunkType = mtmd_input_chunk_get_type(pChunk);
  const char *vName = vChunkType == MTMD_INPUT_CHUNK_TYPE_IMAGE ? "image" : "audio";
  if (vChunkType == MTMD_INPUT_CHUNK_TYPE_TEXT) {
    LOG_ERR("failed to decode chunk: input chunk not of image/audio type\n");
    return -1;
  }

  const mgs_model *vModel = mgs_get_model(pLctx);
  int vNMmprojEmbd = mgs_model_n_embd_inp(vModel);
  int vNPosPerEmbd = mtmd_decode_use_mrope(pCtx) ? 4 : 1;

  int32_t vNTokens = mtmd_input_chunk_get_n_tokens(pChunk);
  int32_t vIBatch = 0;
  int32_t vNImgBatches = (vNTokens + pNBatch - 1) / pNBatch;
  decode_embd_batch vBatchEmbd(pEncodedEmbd, vNTokens, vNPosPerEmbd, vNMmprojEmbd);

  if (mtmd_decode_use_mrope(pCtx)) {
    if (vChunkType == MTMD_INPUT_CHUNK_TYPE_IMAGE) {
      const auto cImageTokens = mtmd_input_chunk_get_tokens_image(pChunk);
      if (!cImageTokens) {
        LOG_ERR("failed to decode chunk: image tokens are null\n");
        return -1;
      }
      const auto cNTokens = mtmd_image_tokens_get_n_tokens(cImageTokens);
      std::vector<mtmd_decoder_pos> lRelPos(cNTokens);
      mtmd_helper_image_get_decoder_pos(cImageTokens, pNPast, lRelPos.data());
      vBatchEmbd.mSetPositionMrope2d(lRelPos, pSeqId);
    } else if (vChunkType == MTMD_INPUT_CHUNK_TYPE_AUDIO) {
      vBatchEmbd.mSetPositionMrope1d(pNPast, pSeqId);
    } else {
      GGML_ABORT("invalid chunk type for M-RoPE");
    }
  } else {
    vBatchEmbd.mSetPositionNormal(pNPast, pSeqId);
  }

  const bool cUseNonCausal = mtmd_decode_use_non_causal(pCtx, pChunk);
  const scope_non_causal cNonCausal(pLctx, cUseNonCausal);

  while (vIBatch < vNImgBatches) { // split into batches
    int vPosOffset = vIBatch * pNBatch;
    int vNTokensBatch = std::min(pNBatch, vNTokens - vPosOffset);
    mgs_batch vBatchEmbdView = vBatchEmbd.mGetView(vPosOffset, vNTokensBatch);

    LOG_INF("decoding %s batch %d/%d, n_tokens_batch = %d\n", vName, vIBatch + 1, vNImgBatches, vNTokensBatch);

    int64_t vT1 = ggml_time_ms();
    int32_t vRet = mgs_decode(pLctx, vBatchEmbdView);
    if (vRet != 0) {
      LOG_ERR("failed to decode %s\n", vName);
      return vRet;
    }

    if (pCallback != nullptr) {
      vRet = pCallback(vBatchEmbdView, pUserData);
      if (vRet != 0) {
        LOG_ERR("post-decode callback failed\n");
        return vRet;
      }
    }

    LOG_INF("%s decoded (batch %d/%d) in %" PRId64 " ms\n", vName, vIBatch + 1, vNImgBatches, ggml_time_ms() - vT1);

    vIBatch++;
  }

  pNPast += mtmd_input_chunk_get_n_pos(pChunk);
  *pNewNPast = pNPast;

  return 0;
}

int32_t mtmd_helper_eval_chunk_single(mtmd_context *pCtx, struct mgs_context *pLctx, const mtmd_input_chunk *pChunk,
                                      mgs_pos pNPast, mgs_seq_id pSeqId, int32_t pNBatch, bool pLogitsLast,
                                      mgs_pos *pNewNPast) {
  GGML_ASSERT(pNBatch > 0);
  int32_t vRet;
  mgs_batch vTextBatch = mgs_batch_init(pNBatch, 0, 1);
  auto vChunkType = mtmd_input_chunk_get_type(pChunk);

  if (vChunkType == MTMD_INPUT_CHUNK_TYPE_TEXT) {
    size_t vNTokens;
    const auto cTokens = mtmd_input_chunk_get_tokens_text(pChunk, &vNTokens);
    // LOG_INF("decoding text chunk, n_tokens = %zu\n", vNTokens);
    size_t vI = 0;
    while (vI < vNTokens) {    // split into batches
      vTextBatch.n_tokens = 0; // clear the batch
      for (; vI < vNTokens && vTextBatch.n_tokens < pNBatch; vI++) {
        int32_t vJ = vTextBatch.n_tokens;
        vTextBatch.token[vJ] = cTokens[vI];
        vTextBatch.pos[vJ] = pNPast++;
        vTextBatch.n_seq_id[vJ] = 1;
        vTextBatch.seq_id[vJ][0] = pSeqId;
        vTextBatch.logits[vJ] = false;

        vTextBatch.n_tokens++;
      }
      bool vIsLastToken = (vI == vNTokens);
      if (pLogitsLast && vIsLastToken) {
        vTextBatch.logits[vTextBatch.n_tokens - 1] = true;
      }
      vRet = mgs_decode(pLctx, vTextBatch);
      if (vRet != 0) {
        LOG_ERR("failed to decode text\n");
        mgs_batch_free(vTextBatch);
        return vRet;
      }
      *pNewNPast += vTextBatch.n_tokens;
    }

  } else if (vChunkType == MTMD_INPUT_CHUNK_TYPE_IMAGE || vChunkType == MTMD_INPUT_CHUNK_TYPE_AUDIO) {
    const char *vName = vChunkType == MTMD_INPUT_CHUNK_TYPE_IMAGE ? "image" : "audio";
    int64_t vT0 = ggml_time_ms();

    LOG_INF("encoding %s slice...\n", vName);

    vRet = mtmd_encode_chunk(pCtx, pChunk);
    if (vRet != 0) {
      LOG_ERR("failed to encode %s slice\n", vName);
      mgs_batch_free(vTextBatch);
      return vRet;
    }

    LOG_INF("%s slice encoded in %" PRId64 " ms\n", vName, ggml_time_ms() - vT0);

    float *vEmbd = mtmd_get_output_embd(pCtx);
    vRet = mtmd_helper_decode_image_chunk(pCtx, pLctx, pChunk, vEmbd, pNPast, pSeqId, pNBatch, pNewNPast, nullptr,
                                          nullptr);
    if (vRet != 0) {
      LOG_ERR("failed to decode %s\n", vName);
      mgs_batch_free(vTextBatch);
      return vRet;
    }
  } else {
    GGML_ABORT("chunk type not supported");
  }

  mgs_batch_free(vTextBatch);
  return 0;
}

int32_t mtmd_helper_eval_chunks(mtmd_context *pCtx, struct mgs_context *pLctx, const mtmd_input_chunks *pChunks,
                                mgs_pos pNPast, mgs_seq_id pSeqId, int32_t pNBatch, bool pLogitsLast,
                                mgs_pos *pNewNPast) {
  size_t vNChunks = mtmd_input_chunks_size(pChunks);
  if (vNChunks == 0) {
    LOG_WRN("no chunks to eval\n");
    return 0;
  }

  for (size_t vI = 0; vI < vNChunks; vI++) {
    bool vChunkLogitsLast = (vI == vNChunks - 1) && pLogitsLast;
    auto vChunk = mtmd_input_chunks_get(pChunks, vI);

    int32_t vRes =
        mtmd_helper_eval_chunk_single(pCtx, pLctx, vChunk, pNPast, pSeqId, pNBatch, vChunkLogitsLast, &pNPast);
    if (vRes != 0) {
      LOG_ERR("failed to eval chunk %zu\n", vI);
      return vRes;
    }
    *pNewNPast = pNPast;
  }

  return 0;
}

namespace audio_helpers {

static bool fIsAudioFile(const char *pBuf, size_t pLen) {
  if (pLen < 12) {
    return false;
  }

  // RIFF ref: https://en.wikipedia.org/wiki/Resource_Interchange_File_Format
  // WAV ref: https://www.mmsp.ece.mcgill.ca/Documents/AudioFormats/WAVE/WAVE.html
  bool vIsWav = memcmp(pBuf, "RIFF", 4) == 0 && memcmp(pBuf + 8, "WAVE", 4) == 0;
  bool vIsMp3 = pLen >= 3 && (memcmp(pBuf, "ID3", 3) == 0 ||
                              // Check for MPEG sync word (simplified check)
                              ((unsigned char)pBuf[0] == 0xFF && ((unsigned char)pBuf[1] & 0xE0) == 0xE0));
  bool vIsFlac = memcmp(pBuf, "fLaC", 4) == 0;

  return vIsWav || vIsMp3 || vIsFlac;
}

// returns true if the buffer is a valid audio file
static bool fDecodeAudioFromBuf(const unsigned char *pBufIn, size_t pLen, int pTargetSamplerRate,
                                std::vector<float> &pPcmf32Mono) {
  ma_result vResult;
  const int cChannels = 1;
  ma_decoder_config vDecoderConfig = ma_decoder_config_init(ma_format_f32, cChannels, pTargetSamplerRate);
  ma_decoder vDecoder;

  vResult = ma_decoder_init_memory(pBufIn, pLen, &vDecoderConfig, &vDecoder);
  if (vResult != MA_SUCCESS) {
    return false;
  }

  ma_uint64 vFrameCount;
  ma_uint64 vFramesRead;
  vResult = ma_decoder_get_length_in_pcm_frames(&vDecoder, &vFrameCount);
  if (vResult != MA_SUCCESS) {
    ma_decoder_uninit(&vDecoder);
    return false;
  }

  // the length is reported at the output sample rate, so it bounds the PCM buffer before it is allocated
  const ma_uint64 cMaxFrames = static_cast<ma_uint64>(pTargetSamplerRate) * cMediaMaxAudioSeconds;
  if (vFrameCount > cMaxFrames) {
    LOG_ERR("%s: audio of %" PRIu64 " samples exceeds the decoding limit of %" PRIu64 " seconds\n", __func__,
            static_cast<uint64_t>(vFrameCount), cMediaMaxAudioSeconds);
    ma_decoder_uninit(&vDecoder);
    return false;
  }

  pPcmf32Mono.resize(vFrameCount);
  vResult = ma_decoder_read_pcm_frames(&vDecoder, pPcmf32Mono.data(), vFrameCount, &vFramesRead);
  if (vResult != MA_SUCCESS) {
    ma_decoder_uninit(&vDecoder);
    return false;
  }

#ifdef MTMD_AUDIO_DEBUG
  // save audio to wav file
  ma_encoder_config vConfig = ma_encoder_config_init(ma_encoding_format_wav, ma_format_f32, 1, pTargetSamplerRate);
  ma_encoder vEncoder;
  ma_encoder_init_file("output.wav", &vConfig, &vEncoder);
  ma_encoder_write_pcm_frames(&vEncoder, pPcmf32Mono.data(), pPcmf32Mono.size(), &vFramesRead);
  ma_encoder_uninit(&vEncoder);
#endif

  ma_decoder_uninit(&vDecoder);
  return true;
}

} // namespace audio_helpers

// Computes FNV-1a hash of the data
static std::string fFnvHash(const uint8_t *pData, size_t pLen) {
  const uint64_t cFnvPrime = 0x100000001b3ULL;
  uint64_t vHash = 0xcbf29ce484222325ULL;

  for (size_t vI = 0; vI < pLen; ++vI) {
    vHash ^= pData[vI];
    vHash *= cFnvPrime;
  }
  return std::to_string(vHash);
}

mtmd_helper_bitmap_wrapper mtmd_helper_bitmap_init_from_buf(mtmd_context *pCtx, const unsigned char *pBuf, size_t pLen,
                                                            bool pPlaceholder) {
  // calculate the hash if needed
  std::string vId;
  mtmd_bitmap *vResult = nullptr;

  if (!pPlaceholder) {
    vId = fFnvHash(pBuf, pLen);
  }

  if (audio_helpers::fIsAudioFile((const char *)pBuf, pLen)) {
    std::vector<float> lPcmf32;
    const int cSampleRate = mtmd_get_audio_sample_rate(pCtx);
    if (cSampleRate < 0) {
      LOG_ERR("This model does not support audio input\n");
      return {nullptr, nullptr};
    }
    if (!audio_helpers::fDecodeAudioFromBuf(pBuf, pLen, cSampleRate, lPcmf32)) {
      LOG_ERR("Unable to read WAV audio file from buffer\n");
      return {nullptr, nullptr};
    }
    vResult = mtmd_bitmap_init_from_audio(lPcmf32.size(), pPlaceholder ? nullptr : lPcmf32.data());
    mtmd_bitmap_set_id(vResult, vId.empty() ? nullptr : vId.c_str());
    return {vResult, nullptr};
  }

  // otherwise, we assume it's an image
  if (!vResult) {
    int vNx, vNy, vNc;
    // read the header first: an image over the limits is not decoded here (the video decoder, if it gets the buffer,
    // applies its own frame limits)
    bool vWithinLimits = true;
    if (stbi_info_from_memory(pBuf, (int)pLen, &vNx, &vNy, &vNc) &&
        (vNx <= 0 || vNy <= 0 || vNx > cMediaMaxImageSide || vNy > cMediaMaxImageSide ||
         static_cast<uint64_t>(vNx) * static_cast<uint64_t>(vNy) > cMediaMaxImagePixels)) {
      LOG_ERR("%s: image of %dx%d pixels exceeds the decoding limits (%d pixels per side, %" PRIu64 " pixels)\n",
              __func__, vNx, vNy, cMediaMaxImageSide, cMediaMaxImagePixels);
      vWithinLimits = false;
    }
    auto *vData = vWithinLimits ? stbi_load_from_memory(pBuf, pLen, &vNx, &vNy, &vNc, 3) : nullptr;
    if (vData) {
      vResult = mtmd_bitmap_init(vNx, vNy, pPlaceholder ? nullptr : vData);
      mtmd_bitmap_set_id(vResult, vId.empty() ? nullptr : vId.c_str());
      stbi_image_free(vData);
      return {vResult, nullptr};
    }
    // otherwise, fallthrough to video decoding (if supported)
  }

  // last try: load as video
#ifdef MTMD_VIDEO
  if (!vResult) {
    auto vParams = mtmd_helper_video_init_params_default();
    auto vVideoCtx = mtmd_helper_video_init_from_buf(pCtx, pBuf, pLen, vParams);
    if (!vVideoCtx) {
      LOG_ERR("%s: failed to decode buffer as either image/audio/video\n", __func__);
      return {nullptr, nullptr};
    }
    vResult = mtmd_bitmap_init_lazy(pCtx, vId.empty() ? nullptr : vId.c_str(), vVideoCtx,
                                   [](size_t, void *pUserData, mtmd_bitmap **pOutBitmap, char **pOutText) -> int {
                                     auto *vCtx = static_cast<mtmd_helper_video *>(pUserData);
                                     char *vText = nullptr;
                                     int vRet = mtmd_helper_video_read_next(vCtx, pOutBitmap, &vText);
                                     *pOutText = vText; // heap-allocated by read_next; freed automatically by mtmd
                                     return vRet;
                                   });
    return {vResult, vVideoCtx};
  }
#else
  if (!vResult) {
    LOG_ERR("%s: failed to decode buffer as either image or audio (video support not compiled in)\n", __func__);
    return {nullptr, nullptr};
  }
#endif

  // should not reach here
  return {nullptr, nullptr};
}

mtmd_helper_bitmap_wrapper mtmd_helper_bitmap_init_from_file(mtmd_context *pCtx, const char *pFname,
                                                             bool pPlaceholder) {
#ifdef _WIN32
  int vWlen = MultiByteToWideChar(CP_UTF8, 0, pFname, -1, NULL, 0);
  if (!vWlen) {
    LOG_ERR("Unable to convert filename to UTF-16: %s\n", pFname);
    return {nullptr, nullptr};
  }
  std::vector<wchar_t> lWfname(vWlen);
  vWlen = MultiByteToWideChar(CP_UTF8, 0, pFname, -1, lWfname.data(), vWlen);
  if (!vWlen) {
    LOG_ERR("Unable to convert filename to UTF-16: %s\n", pFname);
    return {nullptr, nullptr};
  }
  FILE *vF = _wfopen(lWfname.data(), L"rb");
#else
  FILE *vF = fopen(pFname, "rb");
#endif
  if (!vF) {
    LOG_ERR("Unable to open file %s: %s\n", pFname, strerror(errno));
    return {nullptr, nullptr};
  }

  std::vector<unsigned char> lBuf;

  fseek(vF, 0, SEEK_END);
  long vFileSize = ftell(vF);
  fseek(vF, 0, SEEK_SET);
  if (vFileSize < 0) {
    LOG_ERR("Failed to get file size of %s\n", pFname);
    fclose(vF);
    return {nullptr, nullptr};
  }
  lBuf.resize(vFileSize);

  size_t vNRead = fread(lBuf.data(), 1, vFileSize, vF);
  fclose(vF);
  if (vNRead != (size_t)vFileSize) {
    LOG_ERR("Failed to read entire file %s", pFname);
    return {nullptr, nullptr};
  }

  return mtmd_helper_bitmap_init_from_buf(pCtx, lBuf.data(), lBuf.size(), pPlaceholder);
}

bool mtmd_helper_support_video(mtmd_context *pCtx) {
#ifdef MTMD_VIDEO
  return mtmd_support_vision(pCtx);
#else
  GGML_UNUSED(pCtx);
  return false;
#endif
}

//
// Video input helpers
//

#ifdef MTMD_VIDEO

struct mtmd_helper_video {
  mtmd_context *vMctx;
  std::string vPath;
  std::vector<uint8_t> lInputBuf; // non-empty when initialized from buffer
  std::string vFfmpegBin;
  std::string vFfprobeBin;
  float vFpsTarget = 0.0f;
  mtmd_helper_video_info vInfo = {};

  // RAII wrapper for managing subprocess
  struct subprocess_handle {
    struct subprocess_s vProc = {};
    bool vAlive = false;
    std::thread vFeeder;

    subprocess_handle() = default;
    subprocess_handle(const subprocess_handle &) = delete;
    subprocess_handle &operator=(const subprocess_handle &) = delete;
    ~subprocess_handle() { mStop(); }

    void mStop() {
      if (vAlive) {
        subprocess_terminate(&vProc);
      }
      // join before destroy: feeder holds a FILE* from subprocess_stdin;
      // subprocess_destroy closes it, so the thread must finish first
      if (vFeeder.joinable()) {
        vFeeder.join();
      }
      if (vAlive) {
        subprocess_destroy(&vProc);
        vAlive = false;
      }
    }

    FILE *mStdoutPipe() { return subprocess_stdout(&vProc); }

    // buf is tied to lifetime of mtmd_helper_video, so it's guaranteed to outlive the feeder thread
    void mStartFeeder(const std::vector<uint8_t> &pBuf) {
      vFeeder = std::thread([this, &pBuf]() {
        FILE *vF = subprocess_stdin(&vProc);
        if (!vF) {
          return;
        }
        fwrite(pBuf.data(), 1, pBuf.size(), vF);
        fclose(vF);
        vProc.stdin_file = nullptr; // prevent double-close in subprocess_destroy
      });
    }
  };

  subprocess_handle vSp;
  int32_t vCurrentFrame = 0;
  int32_t vMaxFrames = 0; // frames that fit in cMediaMaxVideoDecodedBytes, set by mProbe
  bool vOverLimit = false; // the video has more frames than vMaxFrames

  std::string vPromptStart = "Video:";
  int32_t vTimestampIntervalMs = 5000; // emit a timestamp text every N ms (0 = disabled)
  float vNextTimestampMs = 0.0f;       // next elapsed-ms threshold at which to emit

  std::vector<uint8_t> lFrameBuf;
  std::string vPendingText; // text queued to be returned before the next frame
  bool vStartEmitted = false;

  bool mIsBufInput() const { return !lInputBuf.empty(); }

  bool mProbe(float pFpsTargetArg) {
    const char *vInputArg = mIsBufInput() ? "pipe:0" : vPath.c_str();
    std::vector<const char *> lCmd = {
        vFfprobeBin.c_str(),
        "-v",
        "quiet",
        "-show_entries",
        "stream=width,height,r_frame_rate,nb_frames,duration",
        "-select_streams",
        "v:0",
        "-of",
        "default=noprint_wrappers=1",
    };
    if (mIsBufInput()) {
      // buffers come from requests: a playlist or concat file inside them must not make ffprobe open URLs or files
      lCmd.push_back("-protocol_whitelist");
      lCmd.push_back("pipe");
    }
    lCmd.push_back(vInputArg);
    lCmd.push_back(nullptr);

    LOG_DBG("%s: launching:", __func__);
    for (size_t vI = 0; lCmd[vI]; vI++) {
      LOG_DBG(" %s", lCmd[vI]);
    }
    LOG_DBG("\n");

    subprocess_handle vProbeSp;
    if (subprocess_create(lCmd.data(), subprocess_option_search_user_path | subprocess_option_inherit_environment,
                          &vProbeSp.vProc) != 0) {
      LOG_ERR("%s: failed to launch ffprobe\n", __func__);
      return false;
    }
    vProbeSp.vAlive = true;

    if (mIsBufInput()) {
      vProbeSp.mStartFeeder(lInputBuf);
    }

    uint32_t vWidth = 0;
    uint32_t vHeight = 0;
    float vOrigFps = 0.0f;
    float vDuration = -1.0f;
    int32_t vNFramesOrig = -1;
    char aLine[256];
    FILE *vFp = vProbeSp.mStdoutPipe();

    while (fgets(aLine, sizeof(aLine), vFp)) {
      char *vEq = strchr(aLine, '=');
      if (!vEq)
        continue;
      *vEq = '\0';
      const char *vKey = aLine;
      const char *vVal = vEq + 1;
      char *vNl = (char *)strchr(vVal, '\n');
      if (vNl)
        *vNl = '\0';

      if (strcmp(vKey, "width") == 0) {
        vWidth = (uint32_t)atoi(vVal);
      } else if (strcmp(vKey, "height") == 0) {
        vHeight = (uint32_t)atoi(vVal);
      } else if (strcmp(vKey, "r_frame_rate") == 0) {
        vOrigFps = mParseRational(vVal);
      } else if (strcmp(vKey, "nb_frames") == 0 && strcmp(vVal, "N/A") != 0) {
        vNFramesOrig = atoi(vVal);
      } else if (strcmp(vKey, "duration") == 0 && strcmp(vVal, "N/A") != 0) {
        vDuration = (float)atof(vVal);
      }
    }

    vProbeSp.mStop();

    if (vWidth == 0 || vHeight == 0 || vOrigFps <= 0.0f) {
      return false;
    }

    // every decoded frame is kept as raw RGB until the prompt is tokenized: bound the frame size here and the number
    // of frames in mReadNext (8192 x 8192 x 3 fits size_t on every supported target)
    if (vWidth > cMediaMaxVideoSide || vHeight > cMediaMaxVideoSide) {
      LOG_ERR("%s: video of %ux%u pixels exceeds the decoding limit of %u pixels per side\n", __func__, vWidth,
              vHeight, cMediaMaxVideoSide);
      return false;
    }
    const uint64_t cFrameBytes = static_cast<uint64_t>(vWidth) * vHeight * 3;
    vMaxFrames = static_cast<int32_t>(std::min<uint64_t>(cMediaMaxVideoDecodedBytes / cFrameBytes, INT32_MAX));

    if (vDuration < 0.0f && vNFramesOrig > 0) {
      vDuration = (float)vNFramesOrig / vOrigFps;
    }

    vFpsTarget = pFpsTargetArg > 0.0f ? pFpsTargetArg : vOrigFps;
    vInfo.width = vWidth;
    vInfo.height = vHeight;
    vInfo.fps = vFpsTarget;
    LOG_DBG("%s: %ux%u fps=%.2f duration=%.2fs n_frames=%d\n", __func__, vWidth, vHeight, vFpsTarget, vDuration,
            vInfo.n_frames);
    vInfo.n_frames = vDuration > 0.0f ? (int32_t)(vDuration * vFpsTarget + 0.5f) : -1;
    lFrameBuf.resize((size_t)vWidth * vHeight * 3);
    return true;
  }

  bool mStartFfmpeg(float pSeekSeconds) {
    char aSeekBuf[64];
    char aFpsBuf[64];

    std::vector<const char *> lCmd;
    lCmd.push_back(vFfmpegBin.c_str());

    if (!mIsBufInput() && pSeekSeconds > 0.0f) {
      // input-side seek: fast, keyframe-accurate; only valid for seekable file inputs
      snprintf(aSeekBuf, sizeof(aSeekBuf), "%.6f", pSeekSeconds);
      lCmd.push_back("-ss");
      lCmd.push_back(aSeekBuf);
    }

    lCmd.push_back("-nostdin");
    if (mIsBufInput()) {
      // buffers come from requests: a playlist or concat file inside them must not make ffmpeg open URLs or files
      lCmd.push_back("-protocol_whitelist");
      lCmd.push_back("cache,pipe");
    }
    lCmd.push_back("-i");
    // cache:pipe:0 wraps stdin with a seekable in-memory cache, letting ffmpeg seek
    // backwards for container headers (e.g. MP4 moov atom at end of file)
    lCmd.push_back(mIsBufInput() ? "cache:pipe:0" : vPath.c_str());

    if (pSeekSeconds > 0.0f && mIsBufInput()) {
      // output-side seek: frame-accurate but decodes and discards frames up to seek point
      snprintf(aSeekBuf, sizeof(aSeekBuf), "%.6f", pSeekSeconds);
      lCmd.push_back("-ss");
      lCmd.push_back(aSeekBuf);
    }

    if (vFpsTarget > 0.0f) {
      snprintf(aFpsBuf, sizeof(aFpsBuf), "fps=%.6f", vFpsTarget);
      lCmd.push_back("-vf");
      lCmd.push_back(aFpsBuf);
    }

    lCmd.push_back("-f");
    lCmd.push_back("rawvideo");
    lCmd.push_back("-pix_fmt");
    lCmd.push_back("rgb24");
    lCmd.push_back("pipe:1");
    lCmd.push_back("-loglevel");
    lCmd.push_back("error");
    lCmd.push_back(nullptr);

    LOG_DBG("%s: launching:", __func__);
    for (size_t vI = 0; lCmd[vI]; vI++) {
      LOG_DBG(" %s", lCmd[vI]);
    }
    LOG_DBG("\n");

    int vRet =
        subprocess_create(lCmd.data(), subprocess_option_search_user_path | subprocess_option_inherit_environment, &vSp.vProc);

    vSp.vAlive = (vRet == 0);
    LOG_DBG("%s: subprocess_create ret=%d proc_alive=%d\n", __func__, vRet, (int)vSp.vAlive);

    if (vSp.vAlive && mIsBufInput()) {
      LOG_DBG("%s: starting feeder thread for %zu-byte buffer\n", __func__, lInputBuf.size());
      vSp.mStartFeeder(lInputBuf);
    }

    return vSp.vAlive;
  }

  void mStopFfmpeg() { vSp.mStop(); }

  mtmd_bitmap *mReadNextFrame() {
    if (!vSp.vAlive)
      return nullptr;

    FILE *vFp = vSp.mStdoutPipe();
    const size_t cFrameSize = (size_t)vInfo.width * vInfo.height * 3;
    LOG_DBG("%s: reading frame %d, expecting %zu bytes (%ux%u)\n", __func__, vCurrentFrame, cFrameSize, vInfo.width,
            vInfo.height);

    size_t vTotalRead = 0;
    while (vTotalRead < cFrameSize) {
      size_t vN = fread(lFrameBuf.data() + vTotalRead, 1, cFrameSize - vTotalRead, vFp);
      if (vN == 0) {
        // clean EOF only if no bytes read yet; partial frame is an error
        LOG_DBG("%s: fread returned 0 after %zu/%zu bytes (ferror=%d)\n", __func__, vTotalRead, cFrameSize,
                ferror(vFp));
        vSp.vAlive = false;
        return nullptr;
      }
      vTotalRead += vN;
    }

    // the frame was read into the reused buffer; refuse it before it becomes one more retained bitmap
    if (vCurrentFrame >= vMaxFrames) {
      LOG_ERR("%s: video exceeds the decoding limit of %d frames of %ux%u pixels (%" PRIu64 " bytes)\n", __func__,
              vMaxFrames, vInfo.width, vInfo.height, cMediaMaxVideoDecodedBytes);
      vOverLimit = true;
      vSp.mStop();
      return nullptr;
    }

    LOG_DBG("%s: frame %d read OK\n", __func__, vCurrentFrame);
    vCurrentFrame++;
    return mtmd_bitmap_init(vInfo.width, vInfo.height, lFrameBuf.data());
  }

  int32_t mReadNext(mtmd_bitmap **pOutBitmap, char **pOutText) {
    *pOutBitmap = nullptr;
    *pOutText = nullptr;

    if (!vPendingText.empty()) {
      *pOutText = strdup(vPendingText.c_str());
      vPendingText.clear();
      return *pOutText ? 0 : -2;
    }

    LOG_DBG("%s: proc_alive=%d start_emitted=%d current_frame=%d\n", __func__, (int)vSp.vAlive, (int)vStartEmitted,
            vCurrentFrame);

    if (!vSp.vAlive) {
      return (vCurrentFrame == 0 || vOverLimit) ? -2 : -1;
    }

    if (!vStartEmitted) {
      vStartEmitted = true;
      if (!vPromptStart.empty()) {
        *pOutText = strdup(vPromptStart.c_str());
        return *pOutText ? 0 : -2;
      }
    }

    mtmd_bitmap *vFrame = mReadNextFrame();
    if (!vFrame)
      return vOverLimit ? -2 : -1;
    *pOutBitmap = vFrame;

    if (vTimestampIntervalMs > 0) {
      // vCurrentFrame was already incremented by mReadNextFrame(); undo for elapsed calc
      float vElapsedMs = (float)(vCurrentFrame - 1) / vInfo.fps * 1000.0f;
      if (vElapsedMs >= vNextTimestampMs) {
        char aTsBuf[32];
        float vElapsedS = vElapsedMs / 1000.0f;
        int vMinutes = (int)(vElapsedS / 60);
        float vSeconds = vElapsedS - vMinutes * 60.0f;
        snprintf(aTsBuf, sizeof(aTsBuf), "[%dm%.2fs]", vMinutes, vSeconds);
        vPendingText = aTsBuf;
        vNextTimestampMs += (float)vTimestampIntervalMs;
      }
    }

    return 0;
  }

  static float mParseRational(const char *pS) {
    int vNum = 0, vDen = 1;
    if (sscanf(pS, "%d/%d", &vNum, &vDen) == 2 && vDen > 0) {
      return (float)vNum / (float)vDen;
    }
    float vVal;
    if (sscanf(pS, "%f", &vVal) == 1) {
      return vVal;
    }
    return 0.0f;
  }
};
#endif

mtmd_helper_video_init_params mtmd_helper_video_init_params_default() {
  return {
      /* fps_target             */ 4.0f,
      /* ffmpeg_bin_dir         */ nullptr,
      /* timestamp_interval_ms  */ 5000,
  };
}

static std::string fVideoResolveBin(const char *pBinDir, const char *pName) {
  if (!pBinDir || pBinDir[0] == '\0') {
    return pName; // rely on PATH
  }
  std::string vResult = pBinDir;
  char vLast = vResult.back();
  if (vLast != '/' && vLast != '\\') {
#ifdef _WIN32
    vResult += '\\';
#else
    vResult += '/';
#endif
  }
  vResult += pName;
#ifdef _WIN32
  vResult += ".exe";
#endif
  return vResult;
}

mtmd_helper_video *mtmd_helper_video_init(mtmd_context *pMctx, const char *pPath,
                                          mtmd_helper_video_init_params pParams) {
#ifdef MTMD_VIDEO
  auto *vCtx = new mtmd_helper_video();

  vCtx->vMctx = pMctx;
  vCtx->vPath = pPath;
  vCtx->vFfmpegBin = fVideoResolveBin(pParams.ffmpeg_bin_dir, "ffmpeg");
  vCtx->vFfprobeBin = fVideoResolveBin(pParams.ffmpeg_bin_dir, "ffprobe");
  vCtx->vTimestampIntervalMs = pParams.timestamp_interval_ms;

  if (!vCtx->mProbe(pParams.fps_target)) {
    LOG_ERR("%s: ffprobe failed for '%s' (is ffprobe in PATH?)\n", __func__, pPath);
    delete vCtx;
    return nullptr;
  }

  if (!vCtx->mStartFfmpeg(0.0f)) {
    LOG_ERR("%s: failed to start ffmpeg for '%s' (is ffmpeg in PATH?)\n", __func__, pPath);
    delete vCtx;
    return nullptr;
  }

  return vCtx;
#else
  GGML_UNUSED(pMctx);
  GGML_UNUSED(pPath);
  GGML_UNUSED(pParams);
  LOG_ERR("%s: video is not supported in this build (MTMD_VIDEO is set to OFF)\n", __func__);
  return nullptr;
#endif
}

mtmd_helper_video *mtmd_helper_video_init_from_buf(mtmd_context *pMctx, const unsigned char *pBuf, size_t pLen,
                                                   mtmd_helper_video_init_params pParams) {
#ifdef MTMD_VIDEO
  auto *vCtx = new mtmd_helper_video();

  vCtx->vMctx = pMctx;
  vCtx->lInputBuf.assign(pBuf, pBuf + pLen);
  vCtx->vFfmpegBin = fVideoResolveBin(pParams.ffmpeg_bin_dir, "ffmpeg");
  vCtx->vFfprobeBin = fVideoResolveBin(pParams.ffmpeg_bin_dir, "ffprobe");
  vCtx->vTimestampIntervalMs = pParams.timestamp_interval_ms;

  if (!vCtx->mProbe(pParams.fps_target)) {
    LOG_ERR("%s: ffprobe failed on buffer (is ffprobe in PATH?)\n", __func__);
    delete vCtx;
    return nullptr;
  }

  if (!vCtx->mStartFfmpeg(0.0f)) {
    LOG_ERR("%s: failed to start ffmpeg on buffer (is ffmpeg in PATH?)\n", __func__);
    delete vCtx;
    return nullptr;
  }

  return vCtx;
#else
  GGML_UNUSED(pMctx);
  GGML_UNUSED(pBuf);
  GGML_UNUSED(pLen);
  GGML_UNUSED(pParams);
  LOG_ERR("%s: video is not supported in this build (MTMD_VIDEO is set to OFF)\n", __func__);
  return nullptr;
#endif
}

void mtmd_helper_video_free(mtmd_helper_video *pCtx) {
#ifdef MTMD_VIDEO
  if (!pCtx)
    return;
  pCtx->mStopFfmpeg();
  delete pCtx;
#else
  GGML_UNUSED(pCtx);
  LOG_ERR("%s: video is not supported in this build (MTMD_VIDEO is set to OFF)\n", __func__);
#endif
}

mtmd_helper_video_info mtmd_helper_video_get_info(const mtmd_helper_video *pCtx) {
#ifdef MTMD_VIDEO
  return pCtx->vInfo;
#else
  GGML_UNUSED(pCtx);
  GGML_ASSERT(false && "video is not supported in this build (MTMD_VIDEO is set to OFF)");
#endif
}

int32_t mtmd_helper_video_read_next(mtmd_helper_video *pCtx, mtmd_bitmap **pOutBitmap, char **pOutText) {
#ifdef MTMD_VIDEO
  if (!pCtx)
    return -2;
  return pCtx->mReadNext(pOutBitmap, pOutText);
#else
  GGML_UNUSED(pCtx);
  GGML_UNUSED(pOutBitmap);
  GGML_UNUSED(pOutText);
  GGML_ASSERT(false && "video is not supported in this build (MTMD_VIDEO is set to OFF)");
#endif
}
