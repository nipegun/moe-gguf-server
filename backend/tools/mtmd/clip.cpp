#include "clip.h"
#include "clip-impl.h"
#include "clip-model.h"
#include "clip-graph.h"
#include "models/models.h"

#include "ggml.h"
#include "ggml-cpp.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "gguf.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <stdexcept>
#include <unordered_set>
#include <vector>
#include <cinttypes>
#include <limits>
#include <array>
#include <functional>
#include <float.h>

struct clip_logger_state vGLoggerState = {fClipLogCallbackDefault, NULL};

// #define CLIP_DEBUG_FUNCTIONS

#ifdef CLIP_DEBUG_FUNCTIONS
static void fClipImageWriteImageToPpm(const clip_image_u8 &pImg, const std::string &pFilename) {
  std::ofstream vFile(pFilename, std::ios::binary);
  if (!vFile.is_open()) {
    LOG_ERR("Failed to open file for writing: %s\n", pFilename.c_str());
    return;
  }

  // PPM header: P6 format, width, height, and max color value
  const auto cPpmSize = pImg.mGetSize();
  vFile << "P6\n" << cPpmSize.vWidth << " " << cPpmSize.vHeight << "\n255\n";

  // Write pixel data
  const auto &vPpmBuf = pImg.mGetRoBuf();
  for (size_t vI = 0; vI < vPpmBuf.size(); vI += 3) {
    // PPM expects binary data in RGB format, which matches our image buffer
    vFile.write(reinterpret_cast<const char *>(&vPpmBuf[vI]), 3);
  }

  vFile.close();
}

static void fClipImageSaveToBmp(const clip_image_u8 &pImg, const std::string &pFilename) {
  std::ofstream vFile(pFilename, std::ios::binary);
  if (!vFile.is_open()) {
    LOG_ERR("Failed to open file for writing: %s\n", pFilename.c_str());
    return;
  }

  const auto cBmpSize = pImg.mGetSize();
  int vFileSize = 54 + 3 * cBmpSize.vWidth * cBmpSize.vHeight; // File header + info header + pixel data
  int vBytesPerPixel = 3;
  int vWidthInBytes = cBmpSize.vWidth * vBytesPerPixel;
  int vPaddingAmount = (4 - (vWidthInBytes % 4)) % 4;
  int vStride = vWidthInBytes + vPaddingAmount;

  // Bitmap file header
  unsigned char aFileHeader[14] = {
      'B', 'M',       // Signature
      0,   0,   0, 0, // Image file size in bytes
      0,   0,   0, 0, // Reserved
      54,  0,   0, 0  // Start of pixel array
  };

  // Total file size
  vFileSize = 54 + (vStride * cBmpSize.vHeight);
  aFileHeader[2] = (unsigned char)(vFileSize);
  aFileHeader[3] = (unsigned char)(vFileSize >> 8);
  aFileHeader[4] = (unsigned char)(vFileSize >> 16);
  aFileHeader[5] = (unsigned char)(vFileSize >> 24);

  // Bitmap information header (BITMAPINFOHEADER)
  unsigned char aInfoHeader[40] = {
      40, 0, 0, 0, // Size of this header (40 bytes)
      0,  0, 0, 0, // Image width
      0,  0, 0, 0, // Image height
      1,  0,       // Number of color planes
      24, 0,       // Bits per pixel
      0,  0, 0, 0, // No compression
      0,  0, 0, 0, // Image size (can be 0 for no compression)
      0,  0, 0, 0, // X pixels per meter (not specified)
      0,  0, 0, 0, // Y pixels per meter (not specified)
      0,  0, 0, 0, // Total colors (color table not used)
      0,  0, 0, 0  // Important colors (all are important)
  };

  // Width and height in the information header
  aInfoHeader[4] = (unsigned char)(cBmpSize.vWidth);
  aInfoHeader[5] = (unsigned char)(cBmpSize.vWidth >> 8);
  aInfoHeader[6] = (unsigned char)(cBmpSize.vWidth >> 16);
  aInfoHeader[7] = (unsigned char)(cBmpSize.vWidth >> 24);
  aInfoHeader[8] = (unsigned char)(cBmpSize.vHeight);
  aInfoHeader[9] = (unsigned char)(cBmpSize.vHeight >> 8);
  aInfoHeader[10] = (unsigned char)(cBmpSize.vHeight >> 16);
  aInfoHeader[11] = (unsigned char)(cBmpSize.vHeight >> 24);

  // Write file headers
  vFile.write(reinterpret_cast<char *>(aFileHeader), sizeof(aFileHeader));
  vFile.write(reinterpret_cast<char *>(aInfoHeader), sizeof(aInfoHeader));

  // Pixel data
  std::vector<unsigned char> lPadding(3, 0);           // Max padding size to be added to each row
  for (int vY = cBmpSize.vHeight - 1; vY >= 0; --vY) { // BMP files are stored bottom-to-top
    for (int vX = 0; vX < cBmpSize.vWidth; ++vX) {
      // Each pixel
      const auto cPx = pImg.mGetPixel(vX, vY);
      unsigned char aPixel[3] = {cPx[2], // BMP stores pixels in BGR format
                                 cPx[1], cPx[0]};
      vFile.write(reinterpret_cast<char *>(aPixel), 3);
    }
    // Write padding for the row
    vFile.write(reinterpret_cast<char *>(lPadding.data()), vPaddingAmount);
  }

  vFile.close();
}

// debug function to convert f32 to u8
static void fClipImageConvertF32ToU8(const clip_image_f32 &pSrc, clip_image_u8 &pDst) {
  pDst.mSetSize(pSrc.mGetSize(), false);
  const auto &vSrcBuf = pSrc.mGetRoBuf();
  std::vector<uint8_t> lDstBuf(pSrc.mNElements());
  for (size_t vI = 0; vI < pSrc.mNElements(); ++vI) {
    lDstBuf[vI] = static_cast<uint8_t>(std::min(std::max(int(vSrcBuf[vI] * 255.0f), 0), 255));
  }
  pDst.mCpyBuf(lDstBuf);
}
#endif

struct clip_ctx {
  clip_model vModel;

  gguf_context_ptr vCtxGguf;
  ggml_context_ptr vCtxData;

  std::vector<uint8_t> lBufComputeMeta;

  std::vector<ggml_backend_t> lBackendPtrs;
  std::vector<ggml_backend_buffer_type_t> lBackendBuft;

  ggml_backend_t vBackend = nullptr;
  ggml_backend_t vBackendCpu = nullptr;
  ggml_backend_buffer_ptr vBuf;

  int vMaxNodes = 8192;
  ggml_backend_sched_ptr vSched;
  clip_flash_attn_type vFlashAttnType = cClipFlashAttnTypeAuto;
  bool vIsAllocated = false;

  bool vDebugOutputEmbeddings = false;

  // for measuring memory usage
  bool vNoAlloc = false;
  std::map<ggml_backend_dev_t, size_t> dMemUsage;
  std::map<ggml_backend_dev_t, size_t> dMemCompute;

  bool vSupportBatch = false;

  clip_ctx(clip_context_params &pCtxParams) {
    vFlashAttnType = pCtxParams.vFlashAttnType;
    vNoAlloc = pCtxParams.vNoAlloc;
    vBackendCpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (!vBackendCpu) {
      throw std::runtime_error("failed to initialize CPU backend");
    }
    if (pCtxParams.vUseGpu) {
      auto *vBackendName = std::getenv("MTMD_BACKEND_DEVICE");
      if (vBackendName != nullptr) {
        vBackend = ggml_backend_init_by_name(vBackendName, nullptr);
        if (!vBackend) {
          LOG_WRN("%s: Warning: Failed to initialize \"%s\" backend, falling back to default GPU backend\n", __func__,
                  vBackendName);
        }
      }
      if (!vBackend) {
        vBackend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr);
        vBackend = vBackend ? vBackend : ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_IGPU, nullptr);
      }
    }

    if (vBackend) {
      LOG_INF("%s: CLIP using %s backend\n", __func__, ggml_backend_name(vBackend));
      lBackendPtrs.push_back(vBackend);
      lBackendBuft.push_back(ggml_backend_get_default_buffer_type(vBackend));
    } else {
      vBackend = vBackendCpu;
      LOG_INF("%s: CLIP using CPU backend\n", __func__);
    }

    if (pCtxParams.vImageMinTokens > 0) {
      vModel.vHparams.vCustomImageMinTokens = pCtxParams.vImageMinTokens;
    }
    if (pCtxParams.vImageMaxTokens > 0) {
      vModel.vHparams.vCustomImageMaxTokens = pCtxParams.vImageMaxTokens;
    }

    lBackendPtrs.push_back(vBackendCpu);
    lBackendBuft.push_back(ggml_backend_get_default_buffer_type(vBackendCpu));

    vSched.reset(
        ggml_backend_sched_new(lBackendPtrs.data(), lBackendBuft.data(), lBackendPtrs.size(), 8192, false, true));

    if (pCtxParams.vCbEval != nullptr) {
      ggml_backend_sched_set_eval_callback(vSched.get(), pCtxParams.vCbEval, pCtxParams.vCbEvalUserData);
    }

    vDebugOutputEmbeddings = std::getenv("MTMD_DEBUG_EMBEDDINGS") != nullptr;
  }

  ~clip_ctx() {
    ggml_backend_free(vBackend);
    if (vBackend != vBackendCpu) {
      ggml_backend_free(vBackendCpu);
    }
  }
};

//
// clip_graph
//

clip_graph::clip_graph(clip_ctx *pCtx, const clip_image_f32 &pImg)
    : vModel(pCtx->vModel), vHparams(vModel.vHparams), vProjType(vModel.vProjType), cPatchSize(vHparams.vPatchSize),
      cNPatchesX(pImg.mNx() / cPatchSize), cNPatchesY(pImg.mNy() / cPatchSize), cNPatches(cNPatchesX * cNPatchesY),
      cNEmbd(vHparams.vNEmbd), cNHead(vHparams.vNHead), cNHeadKv(vHparams.vNHeadKv),
      cDHead(vHparams.vNEmbdHead > 0 ? vHparams.vNEmbdHead : (cNHead > 0 ? cNEmbd / cNHead : 0)),
      cNLayer(vHparams.vNLayer), cNMmprojEmbd(fClipNMmprojEmbd(pCtx)), cEps(vHparams.vEps),
      vKqScale(cDHead > 0 ? 1.0f / sqrtf((float)cDHead) : 0.0f), cFlashAttnType(pCtx->vFlashAttnType) {
  struct ggml_init_params vParams = {
      /*.mem_size   =*/pCtx->lBufComputeMeta.size(),
      /*.mem_buffer =*/pCtx->lBufComputeMeta.data(),
      /*.no_alloc   =*/true,
  };
  vCtx0Ptr.reset(ggml_init(vParams));
  vCtx0 = vCtx0Ptr.get();
  vGf = ggml_new_graph_custom(vCtx0, pCtx->vMaxNodes, false);
}

ggml_tensor *clip_graph::mBuildMm(ggml_tensor *pW, ggml_tensor *pX) const { return ggml_mul_mat(vCtx0, pW, pX); }

void clip_graph::mCb(ggml_tensor *pCur, const char *pName, int pIl) const {
  if (pIl >= 0) {
    ggml_format_name(pCur, "%s-%d", pName, pIl);
  } else {
    ggml_set_name(pCur, pName);
  }
}

// siglip2 naflex
ggml_tensor *clip_graph::mResizePositionEmbeddings(uint32_t pInterpolationMode) {
  ggml_tensor *vPosEmbd = vModel.vPositionEmbeddings;
  const int cHeight = mGetImg(0).mNy() / cPatchSize;
  const int cWidth = mGetImg(0).mNx() / cPatchSize;
  const uint32_t cMode = pInterpolationMode;
  const int cNPerSide = (int)std::sqrt(vPosEmbd->ne[1]);

  GGML_ASSERT(vPosEmbd);

  if (cHeight == cNPerSide && cWidth == cNPerSide) {
    return vPosEmbd;
  }

  vPosEmbd = ggml_reshape_3d(vCtx0, vPosEmbd, cNEmbd, cNPerSide, cNPerSide); // -> (n_embd, n_per_side, n_per_side)
  vPosEmbd = ggml_permute(vCtx0, vPosEmbd, 2, 0, 1, 3);                      // -> (n_per_side, n_per_side, n_embd)
  vPosEmbd = ggml_interpolate(vCtx0, vPosEmbd, cWidth, cHeight, cNEmbd, 1, cMode); // -> (width, height, n_embd)
  vPosEmbd = ggml_permute(vCtx0, vPosEmbd, 1, 2, 0, 3);                            // -> (n_embd, width, height)
  vPosEmbd = ggml_cont_2d(vCtx0, vPosEmbd, cNEmbd, cWidth * cHeight);              // -> (n_embd, width * height)

  return vPosEmbd;
}

// build vision transformer (ViT) cgraph
// this function should cover most of the models
// if your model has specific features, you should probably duplicate this function
ggml_tensor *clip_graph::mBuildVit(ggml_tensor *pInp, int64_t pNPos, norm_type pNormT, ffn_op_type pFfnT,
                                   ggml_tensor *pLearnedPosEmbd,
                                   std::function<ggml_tensor *(ggml_tensor *, const clip_layer &)> pAddPos,
                                   const build_vit_opts &pOpts) {
  // batch dim: inp is [n_embd, n_pos, B]
  const int64_t cB = pInp->ne[2];

  if (pLearnedPosEmbd) {
    pInp = ggml_add(vCtx0, pInp, pLearnedPosEmbd);
    mCb(pInp, "pos_embed", -1);
  }

  // flatten batch; unflatten again in attention
  pInp = ggml_reshape_2d(vCtx0, pInp, cNEmbd, pNPos * cB);

  ggml_tensor *vInpL = pInp;

  // pre-layernorm
  if (vModel.vPreLnW) {
    vInpL = mBuildNorm(vInpL, vModel.vPreLnW, vModel.vPreLnB, pNormT, cEps, -1);
    mCb(vInpL, "pre_ln", -1);
  }

  // loop over layers
  for (int vIl = 0; vIl < cNLayer; vIl++) {
    auto &vLayer = vModel.lLayers[vIl];
    ggml_tensor *vCur = vInpL; // inpL = residual, cur = hidden_states

    ggml_tensor *vAttnMask = pOpts.vAttnMask;
    if (pOpts.lAttnMaskLayers.size() > (size_t)vIl) {
      vAttnMask = pOpts.lAttnMaskLayers[vIl];
    }

    // layernorm1
    vCur = mBuildNorm(vCur, vLayer.vLn1W, vLayer.vLn1B, pNormT, cEps, vIl);
    mCb(vCur, "layer_inp_normed", vIl);

    // self-attention
    {
      ggml_tensor *vQcur = nullptr;
      ggml_tensor *vKcur = nullptr;
      ggml_tensor *vVcur = nullptr;
      if (vLayer.vQkvW != nullptr) {
        // fused qkv
        vCur = mBuildMm(vLayer.vQkvW, vCur);
        if (vLayer.vQkvB != nullptr) {
          vCur = ggml_add(vCtx0, vCur, vLayer.vQkvB);
        }

        // Q/K/V as [d_head, n_head, n_pos, B], the batch stride is cur->nb[1]*n_pos.
        vQcur = ggml_view_4d(vCtx0, vCur, cDHead, cNHead, pNPos, cB,
                             /* nb1    */ ggml_row_size(vCur->type, cDHead),
                             /* nb2    */ vCur->nb[1],
                             /* nb3    */ vCur->nb[1] * pNPos,
                             /* offset */ 0);

        vKcur = ggml_view_4d(vCtx0, vCur, cDHead, cNHead, pNPos, cB,
                             /* nb1    */ ggml_row_size(vCur->type, cDHead),
                             /* nb2    */ vCur->nb[1],
                             /* nb3    */ vCur->nb[1] * pNPos,
                             /* offset */ ggml_row_size(vCur->type, cNHead * cDHead));

        vVcur = ggml_view_4d(vCtx0, vCur, cDHead, cNHead, pNPos, cB,
                             /* nb1    */ ggml_row_size(vCur->type, cDHead),
                             /* nb2    */ vCur->nb[1],
                             /* nb3    */ vCur->nb[1] * pNPos,
                             /* offset */ ggml_row_size(vCur->type, 2 * cNHead * cDHead));

        if (vLayer.vQNorm) {
          GGML_ASSERT(vLayer.vQNorm->ne[0] == vQcur->ne[0]);
          vQcur = mBuildNorm(vQcur, vLayer.vQNorm, NULL, pNormT, cEps, vIl);
          mCb(vQcur, "Qcur_norm", vIl);
        }

        if (vLayer.vKNorm) {
          GGML_ASSERT(vLayer.vKNorm->ne[0] == vKcur->ne[0]);
          vKcur = mBuildNorm(vKcur, vLayer.vKNorm, NULL, pNormT, cEps, vIl);
          mCb(vKcur, "Kcur_norm", vIl);
        }

      } else {
        // separate q, k, v
        vQcur = mBuildMm(vLayer.vQW, vCur);
        if (vLayer.vQB) {
          vQcur = ggml_add(vCtx0, vQcur, vLayer.vQB);
        }

        vKcur = mBuildMm(vLayer.vKW, vCur);
        if (vLayer.vKB) {
          vKcur = ggml_add(vCtx0, vKcur, vLayer.vKB);
        }

        vVcur = mBuildMm(vLayer.vVW, vCur);
        if (vLayer.vVB) {
          vVcur = ggml_add(vCtx0, vVcur, vLayer.vVB);
        }

        // if true, norm must be applied after reshaping to (d_head, n_head, n_pos)
        bool vNormPerHead = vLayer.vQNorm && vLayer.vQNorm->ne[0] == cDHead;

        if (!vNormPerHead) {
          if (vLayer.vQNorm) {
            vQcur = mBuildNorm(vQcur, vLayer.vQNorm, NULL, pNormT, cEps, vIl);
            mCb(vQcur, "Qcur_norm", vIl);
          }
          if (vLayer.vKNorm) {
            vKcur = mBuildNorm(vKcur, vLayer.vKNorm, NULL, pNormT, cEps, vIl);
            mCb(vKcur, "Kcur_norm", vIl);
          }
        }

        vQcur = ggml_reshape_4d(vCtx0, vQcur, cDHead, cNHead, pNPos, cB);
        vKcur = ggml_reshape_4d(vCtx0, vKcur, cDHead, cNHeadKv, pNPos, cB);
        vVcur = ggml_reshape_4d(vCtx0, vVcur, cDHead, cNHeadKv, pNPos, cB);

        if (vNormPerHead) {
          if (vLayer.vQNorm) {
            vQcur = mBuildNorm(vQcur, vLayer.vQNorm, NULL, pNormT, cEps, vIl);
            mCb(vQcur, "Qcur_norm_per_head", vIl);
          }
          if (vLayer.vKNorm) {
            vKcur = mBuildNorm(vKcur, vLayer.vKNorm, NULL, pNormT, cEps, vIl);
            mCb(vKcur, "Kcur_norm_per_head", vIl);
          }
        }
      }

      mCb(vQcur, "Qcur", vIl);
      mCb(vKcur, "Kcur", vIl);
      mCb(vVcur, "Vcur", vIl);

      if (pAddPos) {
        vQcur = pAddPos(vQcur, vLayer);
        vKcur = pAddPos(vKcur, vLayer);
        mCb(vQcur, "Qcur_pos", vIl);
        mCb(vKcur, "Kcur_pos", vIl);
      }

      // build_attn returns a flat 2D [n_embd, n_pos*B]
      vCur = mBuildAttn(vLayer.vOW, vLayer.vOB, vQcur, vKcur, vVcur, vAttnMask, vKqScale, vIl);
      mCb(vCur, "attn_out", vIl);
    }

    if (vLayer.vLs1W) {
      vCur = ggml_mul(vCtx0, vCur, vLayer.vLs1W);
      mCb(vCur, "attn_out_scaled", vIl);
    }

    if (vLayer.vAttnPostNormW) {
      vCur = mBuildNorm(vCur, vLayer.vAttnPostNormW, nullptr, pNormT, cEps, vIl);
      mCb(vCur, "attn_post_normed", vIl);
    }

    // re-add the layer input, e.g., residual
    vCur = ggml_add(vCtx0, vCur, vInpL);

    vInpL = vCur; // inpL = residual, cur = hidden_states

    if (pOpts.vCallbackLayerOut) {
      pOpts.vCallbackLayerOut(vCur, vIl);
    }

    mCb(vCur, "ffn_inp", vIl);

    // layernorm2 (pre-ffn norm)
    vCur = mBuildNorm(vCur, vLayer.vLn2W, vLayer.vLn2B, pNormT, cEps, vIl);
    mCb(vCur, "ffn_inp_normed", vIl);

    // ffn
    vCur = mBuildFfn(vCur, vLayer.vFfUpW, vLayer.vFfUpB, vLayer.vFfGateW, vLayer.vFfGateB, vLayer.vFfDownW,
                     vLayer.vFfDownB, pFfnT, vIl);

    mCb(vCur, "ffn_out", vIl);

    if (vLayer.vFfPostNormW) {
      vCur = mBuildNorm(vCur, vLayer.vFfPostNormW, nullptr, pNormT, cEps, vIl);
      mCb(vCur, "ffn_post_normed", vIl);
    }

    if (vLayer.vLs2W) {
      vCur = ggml_mul(vCtx0, vCur, vLayer.vLs2W);
      mCb(vCur, "ffn_out_scaled", vIl);
    }

    // residual 2
    vCur = ggml_add(vCtx0, vInpL, vCur);
    mCb(vCur, "layer_out", vIl);

    if (vLayer.vLsOutW) {
      vCur = ggml_mul(vCtx0, vCur, vLayer.vLsOutW);
      mCb(vCur, "layer_out_scaled", vIl);
    }

    vInpL = vCur;
  }

  if (vModel.mAudioHasAvgpool()) {
    ggml_tensor *vCur = vInpL;
    vCur = ggml_transpose(vCtx0, vCur);
    vCur = ggml_cont(vCtx0, vCur);
    vCur = ggml_pool_1d(vCtx0, vCur, GGML_OP_POOL_AVG, 2, 2, 0);
    vCur = ggml_transpose(vCtx0, vCur);
    vCur = ggml_cont(vCtx0, vCur);
    vInpL = vCur;
  }

  // post-layernorm
  if (vModel.vPostLnW && !pOpts.vSkipPostLn) {
    vInpL = mBuildNorm(vInpL, vModel.vPostLnW, vModel.vPostLnB, pNormT, cEps, -1);
  }

  // restore the batch dim
  GGML_ASSERT(vInpL->ne[1] % cB == 0);
  vInpL = ggml_reshape_3d(vCtx0, vInpL, cNEmbd, vInpL->ne[1] / cB, cB);
  return vInpL;
}

// build the input after conv2d (inp_raw --> patches)
// returns tensor with shape [n_embd, n_patches]
ggml_tensor *clip_graph::mBuildInp() {
  ggml_tensor *vInpRaw = mBuildInpRaw();
  ggml_tensor *vInp = ggml_conv_2d(vCtx0, vModel.vPatchEmbeddings0, vInpRaw, cPatchSize, cPatchSize, 0, 0, 1, 1);
  vInp = ggml_reshape_3d(vCtx0, vInp, cNPatches, cNEmbd, vNBatch);
  vInp = ggml_cont(vCtx0, ggml_transpose(vCtx0, vInp));
  if (vModel.vPatchBias) {
    vInp = ggml_add(vCtx0, vInp, vModel.vPatchBias);
    mCb(vInp, "patch_bias", -1);
  }
  return vInp;
}

ggml_tensor *clip_graph::mBuildInpRaw(int pChannels) {
  const clip_image_f32 &vImg = mGetImg(0);
  ggml_tensor *vInpRaw = ggml_new_tensor_4d(vCtx0, GGML_TYPE_F32, vImg.mNx(), vImg.mNy(), pChannels, vNBatch);
  ggml_set_name(vInpRaw, "inp_raw");
  ggml_set_input(vInpRaw);
  return vInpRaw;
}

ggml_tensor *clip_graph::mBuildNorm(ggml_tensor *pCur, ggml_tensor *pMw, ggml_tensor *pMb, norm_type pType,
                                    float pNormEps, int pIl) const {

  pCur = pType == cNormTypeRms ? ggml_rms_norm(vCtx0, pCur, pNormEps) : ggml_norm(vCtx0, pCur, pNormEps);

  if (pMw) {
    pCur = ggml_mul(vCtx0, pCur, pMw);
    mCb(pCur, "norm_w", pIl);
  }

  if (pMb) {
    pCur = ggml_add(vCtx0, pCur, pMb);
    mCb(pCur, "norm_b", pIl);
  }

  return pCur;
}

ggml_tensor *clip_graph::mBuildFfn(ggml_tensor *pCur, ggml_tensor *pUp, ggml_tensor *pUpB, ggml_tensor *pGate,
                                   ggml_tensor *pGateB, ggml_tensor *pDown, ggml_tensor *pDownB, ffn_op_type pTypeOp,
                                   int pIl) const {

  ggml_tensor *vTmp = pUp ? mBuildMm(pUp, pCur) : pCur;
  mCb(vTmp, "ffn_up", pIl);

  if (pUpB) {
    vTmp = ggml_add(vCtx0, vTmp, pUpB);
    mCb(vTmp, "ffn_up_b", pIl);
  }

  if (pGate) {
    pCur = mBuildMm(pGate, pCur);
    mCb(pCur, "ffn_gate", pIl);

    if (pGateB) {
      pCur = ggml_add(vCtx0, pCur, pGateB);
      mCb(pCur, "ffn_gate_b", pIl);
    }
  } else {
    pCur = vTmp;
  }

  // we only support parallel ffn for now
  switch (pTypeOp) {
  case cFfnSilu:
    if (pGate) {
      // clamped SwiGLU (glm5v): silu(min(gate, limit)) * clamp(up, -limit, limit)
      pCur = vHparams.vSwigluClamp > 0.0f ? ggml_swiglu_clamp(vCtx0, pCur, vTmp, vHparams.vSwigluClamp)
                                          : ggml_swiglu_split(vCtx0, pCur, vTmp);
      mCb(pCur, "ffn_swiglu", pIl);
    } else {
      pCur = ggml_silu(vCtx0, pCur);
      mCb(pCur, "ffn_silu", pIl);
    }
    break;
  case cFfnGelu:
    if (pGate) {
      pCur = ggml_geglu_split(vCtx0, pCur, vTmp);
      mCb(pCur, "ffn_geglu", pIl);
    } else {
      pCur = ggml_gelu(vCtx0, pCur);
      mCb(pCur, "ffn_gelu", pIl);
    }
    break;
  case cFfnGeluErf:
    if (pGate) {
      pCur = ggml_geglu_erf_split(vCtx0, pCur, vTmp);
      mCb(pCur, "ffn_geglu_erf", pIl);
    } else {
      pCur = ggml_gelu_erf(vCtx0, pCur);
      mCb(pCur, "ffn_gelu_erf", pIl);
    }
    break;
  case cFfnGeluQuick:
    if (pGate) {
      pCur = ggml_geglu_quick_split(vCtx0, pCur, vTmp);
      mCb(pCur, "ffn_geglu_quick", pIl);
    } else {
      pCur = ggml_gelu_quick(vCtx0, pCur);
      mCb(pCur, "ffn_gelu_quick", pIl);
    }
    break;
  case cFfnReluSqr: {
    pCur = ggml_relu(vCtx0, pCur);
    pCur = ggml_sqr(vCtx0, pCur);
    mCb(pCur, "ffn_relu_sqr", pIl);
  } break;
  }

  if (pDown) {
    pCur = mBuildMm(pDown, pCur);
  }

  if (pDownB) {
    mCb(pCur, "ffn_down", pIl);
  }

  if (pDownB) {
    pCur = ggml_add(vCtx0, pCur, pDownB);
  }

  return pCur;
}

ggml_tensor *clip_graph::mBuildAttn(ggml_tensor *pWo, ggml_tensor *pWoB, ggml_tensor *pQCur, ggml_tensor *pKCur,
                                    ggml_tensor *pVCur, ggml_tensor *pKqMask, float pKqScale, int pIl,
                                    ggml_tensor *pSinks) const {
  // these nodes are added to the graph together so that they are not reordered
  // by doing so, the number of splits in the graph is reduced
  ggml_build_forward_expand(vGf, pQCur);
  ggml_build_forward_expand(vGf, pKCur);
  ggml_build_forward_expand(vGf, pVCur);

  ggml_tensor *vQ = ggml_permute(vCtx0, pQCur, 0, 2, 1, 3);
  // mCb(vQ, "q", pIl);

  ggml_tensor *vK = ggml_permute(vCtx0, pKCur, 0, 2, 1, 3);
  // mCb(vK, "k", pIl);

  ggml_tensor *vCur;

  if (cFlashAttnType == cClipFlashAttnTypeEnabled) {
    ggml_tensor *vV = ggml_permute(vCtx0, pVCur, 0, 2, 1, 3);

    vK = ggml_cast(vCtx0, vK, GGML_TYPE_F16);
    vV = ggml_cast(vCtx0, vV, GGML_TYPE_F16);
    if (pKqMask) {
      pKqMask = ggml_cast(vCtx0, pKqMask, GGML_TYPE_F16);
    }

    vCur = ggml_flash_attn_ext(vCtx0, vQ, vK, vV, pKqMask, pKqScale, 0.0f, 0.0f);
    ggml_flash_attn_ext_set_prec(vCur, GGML_PREC_F32);
    if (pSinks != nullptr) {
      ggml_flash_attn_ext_add_sinks(vCur, pSinks);
    }

    vCur = ggml_reshape_2d(vCtx0, vCur, vCur->ne[0] * vCur->ne[1], vCur->ne[2] * vCur->ne[3]);

  } else {
    ggml_tensor *vV = ggml_permute(vCtx0, pVCur, 1, 2, 0, 3);
    vV = ggml_cont(vCtx0, vV);

    ggml_tensor *vKq = ggml_mul_mat(vCtx0, vK, vQ);
    // F32 may not needed for vision encoders?
    // ggml_mul_mat_set_prec(vKq, GGML_PREC_F32);

    vKq = ggml_soft_max_ext(vCtx0, vKq, pKqMask, pKqScale, 0.0f);
    if (pSinks != nullptr) {
      ggml_soft_max_add_sinks(vKq, pSinks);
    }

    ggml_tensor *vKqv = ggml_mul_mat(vCtx0, vV, vKq);
    vCur = ggml_permute(vCtx0, vKqv, 0, 2, 1, 3);
    vCur = ggml_cont_2d(vCtx0, vCur, vCur->ne[0] * vCur->ne[1], vCur->ne[2] * vCur->ne[3]);
  }

  mCb(vCur, "kqv_out", pIl);

  if (pWo) {
    vCur = mBuildMm(pWo, vCur);
  }

  if (pWoB) {
    vCur = ggml_add(vCtx0, vCur, pWoB);
  }

  return vCur;
}

// implementation of the 2D RoPE without adding a new op in ggml
// this is not efficient (use double the memory), but works on all backends
// TODO: there was a more efficient which relies on ggml_view and ggml_rope_ext_inplace, but the rope inplace does not
// work well with non-contiguous tensors ; we should fix that and revert back to the original implementation
ggml_tensor *clip_graph::mBuildRope2d(ggml_context *pCtx0, ggml_tensor *pCur,
                                      ggml_tensor *pPosA, // first half
                                      ggml_tensor *pPosB, // second half
                                      const float pFreqBase, const bool pInterleaveFreq) {
  const int64_t cNDim = pCur->ne[0];
  const int64_t cNHead = pCur->ne[1];
  const int64_t cNPos = pCur->ne[2];

  // for example, if we have cur tensor of shape (n_dim=8, n_head, n_pos)
  // we will have a list of 4 inv_freq: 1e-0, 1e-1, 1e-2, 1e-3
  // first half of cur will use 1e-0, 1e-2 (even)
  // second half of cur will use 1e-1, 1e-3 (odd)
  // the trick here is to rotate just half of n_dim, so inv_freq will automatically be even
  //  ^ don't ask me why, it's math! -2(2i) / n_dim == -2i / (n_dim/2)
  // then for the second half, we use freq_scale to shift the inv_freq
  //  ^ why? replace (2i) with (2i+1) in the above equation
  const float cFreqScaleOdd = pInterleaveFreq ? std::pow(pFreqBase, (float)-2 / cNDim) : 1.0;

  // first half
  ggml_tensor *vFirst;
  {
    vFirst = ggml_view_3d(pCtx0, pCur, cNDim / 2, cNHead, cNPos, pCur->nb[1], pCur->nb[2], 0);
    vFirst = ggml_rope_ext(pCtx0, vFirst,
                           pPosA,     // positions
                           nullptr,   // freq factors
                           cNDim / 2, // n_dims
                           0, 0, pFreqBase, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
  }

  // second half
  ggml_tensor *vSecond;
  {
    vSecond = ggml_view_3d(pCtx0, pCur, cNDim / 2, cNHead, cNPos, pCur->nb[1], pCur->nb[2],
                           cNDim / 2 * ggml_element_size(pCur));
    vSecond = ggml_rope_ext(pCtx0, vSecond,
                            pPosB,     // positions
                            nullptr,   // freq factors
                            cNDim / 2, // n_dims
                            0, 0, pFreqBase, cFreqScaleOdd, 0.0f, 1.0f, 0.0f, 0.0f);
  }

  pCur = ggml_concat(pCtx0, vFirst, vSecond, 0);
  return pCur;
}

// Generic function to stack frames for audio processing
// Abstracts out the StackAudioFrames logic used by ultravox
ggml_tensor *clip_graph::mBuildStack(ggml_tensor *pCur, int32_t pStackFactor, int32_t pNEmbed) {
  if (pStackFactor <= 1) {
    return pCur;
  }

  int64_t vTotalElements = ggml_nelements(pCur);
  int64_t vStride = pNEmbed * pStackFactor;

  // Calculate padded length
  int64_t vPaddedLen = GGML_PAD(vTotalElements, vStride);
  int64_t vPad = vPaddedLen - vTotalElements;

  if (vPad > 0) {
    // Pad the tensor to make it divisible by stride
    pCur = ggml_view_1d(vCtx0, pCur, vTotalElements, 0);
    pCur = ggml_pad(vCtx0, pCur, vPad, 0, 0, 0);
  }

  // Reshape to [stride, padded_len / stride]
  pCur = ggml_view_2d(vCtx0, pCur, vStride, vPaddedLen / vStride, ggml_row_size(pCur->type, vStride), 0);
  return pCur;
}

// aka pixel_shuffle / pixel_unshuffle / patch_merger (Kimi-VL)
// support dynamic resolution
ggml_tensor *clip_graph::mBuildPatchMergePermute(ggml_tensor *pCur, int pScaleFactor) {
  GGML_ASSERT(pScaleFactor > 1);

  const int cNEmbd = pCur->ne[0];
  int vWidth = mGetImg(0).mNx() / cPatchSize;
  int vHeight = mGetImg(0).mNy() / cPatchSize;

  // pad width and height to factor
  const int64_t cPadWidth = CLIP_ALIGN(vWidth, pScaleFactor) - vWidth;
  const int64_t cPadHeight = CLIP_ALIGN(vHeight, pScaleFactor) - vHeight;
  pCur = ggml_reshape_3d(vCtx0, pCur, cNEmbd, vWidth, vHeight);
  if (cPadWidth || cPadHeight) {
    pCur = ggml_pad(vCtx0, pCur, 0, cPadWidth, cPadHeight, 0);
    vWidth += cPadWidth;
    vHeight += cPadHeight;
  }

  // unshuffle h
  pCur = ggml_reshape_3d(vCtx0, pCur, cNEmbd * pScaleFactor, vWidth / pScaleFactor, vHeight);
  pCur = ggml_permute(vCtx0, pCur, 0, 2, 1, 3);

  // unshuffle w
  pCur = ggml_cont_3d(vCtx0, pCur, cNEmbd * pScaleFactor * pScaleFactor, vHeight / pScaleFactor, vWidth / pScaleFactor);
  pCur = ggml_permute(vCtx0, pCur, 0, 2, 1, 3);

  pCur = ggml_cont_2d(vCtx0, pCur, pCur->ne[0], pCur->ne[1] * pCur->ne[2]);
  mCb(pCur, "pixel_shuffle", -1);

  return pCur;
}

static std::unique_ptr<clip_graph> fClipGetGraphBuilder(clip_ctx *pCtx, const clip_image_f32_batch &pImgs) {
  const clip_image_f32 &vImg = pImgs.lEntries[0];
  std::unique_ptr<clip_graph> vBuilder;

  switch (pCtx->vModel.vProjType) {
  case cProjectorTypeGemma3:
  case cProjectorTypeIdefics3:
  case cProjectorTypeLfm2:
  case cProjectorTypeJanusPro:
  case cProjectorTypePhi4: {
    vBuilder = std::make_unique<clip_graph_siglip>(pCtx, vImg);
  } break;
  case cProjectorTypeGemma3nv: {
    vBuilder = std::make_unique<clip_graph_mobilenetv5>(pCtx, vImg);
  } break;
  case cProjectorTypePixtral:
  case cProjectorTypeLightonocr: {
    vBuilder = std::make_unique<clip_graph_pixtral>(pCtx, vImg);
  } break;
  case cProjectorTypeDotsOcr: {
    vBuilder = std::make_unique<clip_graph_dotsocr>(pCtx, vImg);
  } break;
  case cProjectorTypeQwen2vl:
  case cProjectorTypeQwen25vl: {
    vBuilder = std::make_unique<clip_graph_qwen2vl>(pCtx, vImg);
  } break;
  case cProjectorTypeQwen3vl: {
    vBuilder = std::make_unique<clip_graph_qwen3vl>(pCtx, vImg);
  } break;
  case cProjectorTypeEXAONE4Point5: {
    vBuilder = std::make_unique<clip_graph_exaone4_5>(pCtx, vImg);
  } break;
  case cProjectorTypeMimovl: {
    vBuilder = std::make_unique<clip_graph_mimovl>(pCtx, vImg);
  } break;
  case cProjectorTypeMinimaxM3: {
    vBuilder = std::make_unique<clip_graph_minimax_m3>(pCtx, vImg);
  } break;
  case cProjectorTypeStep3vl: {
    vBuilder = std::make_unique<clip_graph_step3vl>(pCtx, vImg);
  } break;
  case cProjectorTypeMINICPMV4Point6: {
    vBuilder = std::make_unique<clip_graph_minicpmv4_6>(pCtx, vImg);
  } break;
  case cProjectorTypeInternvl: {
    vBuilder = std::make_unique<clip_graph_internvl>(pCtx, vImg);
  } break;
  case cProjectorTypeNemotronV2Vl: {
    vBuilder = std::make_unique<clip_graph_nemotron_v2_vl>(pCtx, vImg);
  } break;
  case cProjectorTypeUltravox:
  case cProjectorTypeVoxtral:
  case cProjectorTypeQwen2a:
  case cProjectorTypeGlma:
  case cProjectorTypeMeralion:
  case cProjectorTypeMusicFlamingo: {
    vBuilder = std::make_unique<clip_graph_whisper_enc>(pCtx, vImg);
  } break;
  case cProjectorTypeKimivl: {
    vBuilder = std::make_unique<clip_graph_kimivl>(pCtx, vImg);
  } break;
  case cProjectorTypePaddleocr: {
    vBuilder = std::make_unique<clip_graph_paddleocr>(pCtx, vImg);
  } break;
  case cProjectorTypeKimik25: {
    vBuilder = std::make_unique<clip_graph_kimik25>(pCtx, vImg);
  } break;
  case cProjectorTypeCogvlm: {
    vBuilder = std::make_unique<clip_graph_cogvlm>(pCtx, vImg);
  } break;
  case cProjectorTypeHunyuanvl: {
    vBuilder = std::make_unique<clip_graph_hunyuanvl>(pCtx, vImg);
  } break;
  case cProjectorTypeLfm2a: {
    vBuilder = std::make_unique<clip_graph_conformer>(pCtx, vImg);
  } break;
  case cProjectorTypeGraniteSpeech: {
    vBuilder = std::make_unique<clip_graph_granite_speech>(pCtx, vImg);
  } break;
  case cProjectorTypeGlm4v:
  case cProjectorTypeGlm5v: {
    vBuilder = std::make_unique<clip_graph_glm4v>(pCtx, vImg);
  } break;
  case cProjectorTypeQwen3a: {
    vBuilder = std::make_unique<clip_graph_qwen3a>(pCtx, vImg);
  } break;
  case cProjectorTypeMimoAudio: {
    vBuilder = std::make_unique<clip_graph_mimo_audio>(pCtx, vImg);
  } break;
  case cProjectorTypeYoutuvl: {
    vBuilder = std::make_unique<clip_graph_youtuvl>(pCtx, vImg);
  } break;
  case cProjectorTypeYasa2: {
    vBuilder = std::make_unique<clip_graph_yasa2>(pCtx, vImg);
  } break;
  case cProjectorTypeParakeet: {
    vBuilder = std::make_unique<clip_graph_parakeet>(pCtx, vImg);
  } break;
  case cProjectorTypeGranite4Vision: {
    vBuilder = std::make_unique<clip_graph_granite4_vision>(pCtx, vImg);
  } break;
  default:
    GGML_ABORT("missing cgraph builder");
  }

  vBuilder->vImgBatch = &pImgs;

  // TODO [QWEN_VIDEO]: improve this in the future
  vBuilder->vNBatch = pImgs.lEntries.size();

  return vBuilder;
}

//
// clip_model_loader
//

struct clip_model_loader {
  ggml_context_ptr vCtxMeta;
  gguf_context_ptr vCtxGguf;

  std::string vFname;

  size_t vModelSize = 0; // in bytes

  bool vHasVision = false;
  bool vHasAudio = false;

  mtmd_progress_callback vProgressCallback = nullptr;
  void *vProgressCallbackUserData = nullptr;

  // TODO @ngxson : we should not pass clip_ctx here, it should be clip_model
  clip_model_loader(const char *pFname, bool pSkipTensors = false, mtmd_progress_callback pProgressCb = nullptr,
                    void *pProgressUserData = nullptr)
      : vFname(pFname), vProgressCallback(pProgressCb), vProgressCallbackUserData(pProgressUserData) {
    struct ggml_context *vMeta = nullptr;

    struct gguf_init_params vParams = {
        /*.no_alloc = */ true,
        /*.ctx      = */ &vMeta,
    };

    vCtxGguf = gguf_context_ptr(gguf_init_from_file(pFname, vParams));
    if (!vCtxGguf.get()) {
      throw std::runtime_error(
          fStringFormat("%s: failed to load CLIP model from %s. Does this file exist?\n", __func__, pFname));
    }

    vCtxMeta.reset(vMeta);

    const int cNTensors = gguf_get_n_tensors(vCtxGguf.get());

    // print gguf info
    {
      std::string vName;
      mGetString(KEY_NAME, vName, false);
      std::string vDescription;
      mGetString(KEY_DESCRIPTION, vDescription, false);
      LOG_INF("%s: model name:   %s\n", __func__, vName.c_str());
      LOG_INF("%s: description:  %s\n", __func__, vDescription.c_str());
      LOG_INF("%s: GGUF version: %d\n", __func__, gguf_get_version(vCtxGguf.get()));
      LOG_INF("%s: alignment:    %zu\n", __func__, gguf_get_alignment(vCtxGguf.get()));
      LOG_INF("%s: n_tensors:    %d\n", __func__, cNTensors);
      LOG_INF("%s: n_kv:         %d\n", __func__, (int)gguf_get_n_kv(vCtxGguf.get()));
      LOG_INF("\n");
    }

    // modalities
    {
      mGetBool(KEY_HAS_VISION_ENC, vHasVision, false);
      mGetBool(KEY_HAS_AUDIO_ENC, vHasAudio, false);

      if (vHasVision) {
        LOG_INF("%s: has vision encoder\n", __func__);
      }
      if (vHasAudio) {
        LOG_INF("%s: has audio encoder\n", __func__);
      }
    }

    // tensors
    if (!pSkipTensors) {
      for (int vI = 0; vI < cNTensors; ++vI) {
        const char *vName = gguf_get_tensor_name(vCtxGguf.get(), vI);
        const size_t cOffset = gguf_get_tensor_offset(vCtxGguf.get(), vI);
        enum ggml_type vType = gguf_get_tensor_type(vCtxGguf.get(), vI);
        ggml_tensor *vCur = ggml_get_tensor(vMeta, vName);
        size_t vTensorSize = ggml_nbytes(vCur);
        vModelSize += vTensorSize;
        LOG_DBG("%s: tensor[%d]: n_dims = %d, name = %s, tensor_size=%zu, offset=%zu, shape:[%" PRIu64 ", %" PRIu64
                ", %" PRIu64 ", %" PRIu64 "], type = %s\n",
                __func__, vI, ggml_n_dims(vCur), vCur->name, vTensorSize, cOffset, vCur->ne[0], vCur->ne[1],
                vCur->ne[2], vCur->ne[3], ggml_type_name(vType));
      }
    }
  }

  void mLoadHparams(clip_model &pModel, clip_modality pModality) {
    auto &vHparams = pModel.vHparams;
    std::string vLogFfnOp; // for logging

    // sanity check
    if (pModality == cClipModalityVision) {
      GGML_ASSERT(vHasVision);
    } else if (pModality == cClipModalityAudio) {
      GGML_ASSERT(vHasAudio);
    }
    pModel.vModality = pModality;

    // projector type
    std::string vProjType;
    {
      // default key
      mGetString(KEY_PROJ_TYPE, vProjType, false);

      // for models with mixed modalities
      if (vProjType.empty()) {
        if (pModality == cClipModalityVision) {
          mGetString(KEY_VISION_PROJ_TYPE, vProjType, false);
        } else if (pModality == cClipModalityAudio) {
          mGetString(KEY_AUDIO_PROJ_TYPE, vProjType, false);
        } else {
          GGML_ABORT("unknown modality");
        }
      }

      if (vProjType.empty()) {
        throw std::runtime_error(fStringFormat("%s: missing projector type (%s, %s or %s); this mmproj file is not "
                                               "supported, convert the model again with a current converter\n",
                                               __func__, KEY_PROJ_TYPE, KEY_VISION_PROJ_TYPE, KEY_AUDIO_PROJ_TYPE));
      }

      pModel.vProjType = fClipProjectorTypeFromString(vProjType);

      if (pModel.vProjType == cProjectorTypeUnknown) {
        throw std::runtime_error(fStringFormat("%s: unknown projector type: %s\n", __func__, vProjType.c_str()));
      }

      // qwen2.5o is written by the converters for both modalities: pick the projector of this modality
      if (pModel.vProjType == cProjectorTypeQwen25o) {
        pModel.vProjType = pModality == cClipModalityVision ? cProjectorTypeQwen25vl : cProjectorTypeQwen2a;
      }
    }

    const bool cIsVision = pModel.vModality == cClipModalityVision;
    const bool cIsAudio = pModel.vModality == cClipModalityAudio;

    // other hparams
    {
      const char *vPrefix = cIsVision ? "vision" : "audio";
      mGetU32(fStringFormat(KEY_N_EMBD, vPrefix), vHparams.vNEmbd);
      mGetU32(fStringFormat(KEY_N_HEAD, vPrefix), vHparams.vNHead);
      mGetU32(fStringFormat(KEY_N_EMBD_HEAD, vPrefix), vHparams.vNEmbdHead, false);
      mGetU32(fStringFormat(KEY_N_FF, vPrefix), vHparams.vNFf);
      mGetU32(fStringFormat(KEY_N_BLOCK, vPrefix), vHparams.vNLayer);
      mGetU32(fStringFormat(KEY_PROJ_DIM, vPrefix), vHparams.vProjectionDim);
      mGetF32(fStringFormat(KEY_LAYER_NORM_EPS, vPrefix), vHparams.vEps);

      // n_head_kv is optional (for GQA), default to n_head
      vHparams.vNHeadKv = vHparams.vNHead;

      if (cIsVision) {
        mGetU32(KEY_IMAGE_SIZE, vHparams.vImageSize);
        mGetU32(KEY_PATCH_SIZE, vHparams.vPatchSize);
      } else if (cIsAudio) {
        mGetU32(KEY_A_NUM_MEL_BINS, vHparams.vNMelBins);
        // some hparams are unused, but still need to set to avoid issues
        vHparams.vImageSize = 0;
        vHparams.vPatchSize = 1;

      } else {
        GGML_ASSERT(false && "unknown modality");
      }

      // for pinpoints, we need to convert it into a list of resolution candidates
      {
        std::vector<int> lPinpoints;
        mGetArrInt(KEY_IMAGE_GRID_PINPOINTS, lPinpoints, false);
        if (lPinpoints.size() % 2 != 0) {
          throw std::runtime_error(fStringFormat(
              "%s: image_grid_pinpoints must have an even number of elements, got %zu\n", __func__, lPinpoints.size()));
        }
        if (!lPinpoints.empty()) {
          for (size_t vI = 0; vI < lPinpoints.size(); vI += 2) {
            vHparams.lImageResCandidates.push_back({
                lPinpoints[vI],
                lPinpoints[vI + 1],
            });
          }
        }
      }

      // default warmup value
      vHparams.vWarmupImageSize = vHparams.vImageSize;

      {
        bool vUseGelu = false;
        bool vUseSilu = false;
        mGetBool(KEY_USE_GELU, vUseGelu, false);
        mGetBool(KEY_USE_SILU, vUseSilu, false);
        if (vUseGelu && vUseSilu) {
          throw std::runtime_error(fStringFormat("%s: both use_gelu and use_silu are set to true\n", __func__));
        }
        if (vUseGelu) {
          vHparams.vFfnOp = cFfnGelu;
          vLogFfnOp = "gelu";
        } else if (vUseSilu) {
          vHparams.vFfnOp = cFfnSilu;
          vLogFfnOp = "silu";
        } else {
          vHparams.vFfnOp = cFfnGeluQuick;
          vLogFfnOp = "gelu_quick";
        }
      }

      if (cIsVision) {
        std::vector<float> lImageMean;
        std::vector<float> lImageStd;
        mGetArrF32(KEY_IMAGE_MEAN, lImageMean, false);
        mGetArrF32(KEY_IMAGE_STD, lImageStd, false);
        if (lImageMean.size() < 3 || lImageStd.size() < 3) {
          throw std::runtime_error(
              fStringFormat("%s: image_mean/image_std arrays must have at least 3 elements, got %zu and %zu\n",
                            __func__, lImageMean.size(), lImageStd.size()));
        }
        for (int vI = 0; vI < 3; ++vI) {
          vHparams.aImageMean[vI] = lImageMean[vI];
          vHparams.aImageStd[vI] = lImageStd[vI];
        }
      }

      // Load the vision/audio feature layer indices if they are explicitly provided
      // NOTE: gguf conversions should standardize the values of the vision feature layer to
      // be non-negative, since we use -1 to mark values as unset here.
      mGetArrInt(fStringFormat(KEY_FEATURE_LAYERS, vPrefix), vHparams.lFeatureLayers, false);

      // model-specific params
      switch (pModel.vProjType) {
      case cProjectorTypeCogvlm: {
        vHparams.aImagePadColor = {122, 116, 104};
      } break;
      case cProjectorTypeMINICPMV4Point6: {
        // MiniCPM-V 4.6 unified merger projector
        // ViT merger 2x2 + final merger 2x2 = 4x spatial merge per dimension
        vHparams.vNMerge = 4;
        mGetU32(KEY_PROJ_SCALE_FACTOR, vHparams.vNMerge, false);
        GGML_ASSERT(vHparams.vNMerge == 2 || vHparams.vNMerge == 4);

        // borrow wa_layer_indexes for vit_merger insertion point
        std::vector<int> lWaLayerIndexesVec;
        mGetArrInt(KEY_WIN_ATTN_LAYER_INDEXES, lWaLayerIndexesVec, false);
        if (!lWaLayerIndexesVec.empty()) {
          vHparams.vInsertLayerId = lWaLayerIndexesVec[0];
        }
      } break;
      case cProjectorTypeInternvl: {
        // use default llava-uhd preprocessing params
        // older version of internvl doesn't have min/max tiles, we need to provide default values for them to avoid
        // issues
        vHparams.vPreprocMinTiles = 1;
        vHparams.vPreprocMaxTiles = 12;
        mGetU32(KEY_PROJ_SCALE_FACTOR, vHparams.vNMerge, false);
        mGetU32(KEY_PREPROC_MIN_TILES, vHparams.vPreprocMinTiles, false);
        mGetU32(KEY_PREPROC_MAX_TILES, vHparams.vPreprocMaxTiles, false);
        GGML_ASSERT(vHparams.vPreprocMinTiles <= vHparams.vPreprocMaxTiles && vHparams.vPreprocMaxTiles < INT32_MAX);
        mSetInternvlDhrResCandidates(pModel);
      } break;
      case cProjectorTypeNemotronV2Vl: {
        mGetU32(KEY_PROJ_SCALE_FACTOR, vHparams.vNMerge, false);
      } break;
      case cProjectorTypeParakeet: {
        mGetU32(KEY_AUDIO_SUBSAMPLING_FACTOR, vHparams.vSubsamplingFactor);
        GGML_ASSERT(vHparams.vSubsamplingFactor == 8 &&
                    "subsampling_factor must match the conv strides in clip_graph_parakeet::mBuild()");
        mGetU32(KEY_A_CONV_KERNEL_SIZE, vHparams.vAudioConvKernelSize);
        GGML_ASSERT(vHparams.vAudioConvKernelSize > 0 && vHparams.vAudioConvKernelSize % 2 == 1 &&
                    "audio_conv_kernel_size must be a positive odd integer");
        vHparams.vAudioChunkLen = 0;
        vHparams.vAudioSampleRate = 16000;
        vHparams.vAudioNFft = 512;
        vHparams.vAudioWindowLen = 400;
        vHparams.vAudioHopLen = 160;
      } break;
      case cProjectorTypeIdefics3: {
        // use default llava-uhd preprocessing params
        mGetU32(KEY_PROJ_SCALE_FACTOR, vHparams.vNMerge, false);
        mGetU32(KEY_PREPROC_IMAGE_SIZE, vHparams.vImageLongestEdge, false);
      } break;
      case cProjectorTypeLfm2: {
        vHparams.vImageResizeAlgo = cResizeAlgoBilinear;
        vHparams.vImageResizeAlgoRf = cResizeAlgoBilinear;
        vHparams.vImageResizeAlgoOv = cResizeAlgoBilinear;
        mGetU32(KEY_PROJ_SCALE_FACTOR, vHparams.vNMerge, false);
        // ref: https://huggingface.co/LiquidAI/LFM2.5-VL-1.6B/blob/main/processor_config.json
        vHparams.mSetLimitImageTokens(64, 256);
      } break;
      case cProjectorTypePhi4: {
        vHparams.vNMerge = 1;
        vHparams.vImageResizeAlgo = cResizeAlgoBilinear;
        mGetU32(KEY_IMAGE_MIN_PIXELS, vHparams.vImageMinPixels);
        mGetU32(KEY_IMAGE_MAX_PIXELS, vHparams.vImageMaxPixels);
        vHparams.mSetWarmupNTokens(16 * 16);
      } break;
      case cProjectorTypePixtral: {
        // ref: https://huggingface.co/mistral-community/pixtral-12b/blob/main/preprocessor_config.json
        // TODO: verify the image_min_tokens
        vHparams.vNMerge = 1; // the original pixtral does not use patch merging
        vHparams.vImageResizeAlgo = cResizeAlgoBilinear;
        vHparams.vRopeTheta = 10000.0f;
        mGetU32(KEY_SPATIAL_MERGE_SIZE, vHparams.vNMerge, false);
        vHparams.mSetLimitImageTokens(8, 1024);
        vHparams.mSetWarmupNTokens(256); // avoid OOM on warmup
      } break;
      case cProjectorTypeLightonocr: {
        vHparams.vNMerge = 1;
        vHparams.vImageResizeAlgo = cResizeAlgoBicubic;
        vHparams.vRopeTheta = 10000.0f;
        mGetU32(KEY_SPATIAL_MERGE_SIZE, vHparams.vNMerge, false);
        vHparams.vImageLongestEdge = vHparams.vImageSize;
        mGetU32(KEY_PREPROC_IMAGE_SIZE, vHparams.vImageLongestEdge, false);
        vHparams.mSetWarmupNTokens(256); // avoid OOM on warmup
      } break;
      case cProjectorTypeDotsOcr: {
        vHparams.vRopeTheta = 10000.0f;
        mGetU32(KEY_PROJ_SCALE_FACTOR, vHparams.vNMerge);
        mGetU32(KEY_IMAGE_MIN_PIXELS, vHparams.vImageMinPixels);
        mGetU32(KEY_IMAGE_MAX_PIXELS, vHparams.vImageMaxPixels);
        vHparams.mSetWarmupNTokens(46 * 46); // avoid OOM on warmup
      } break;
      case cProjectorTypeKimivl: {
        vHparams.vImageResizeAlgo = cResizeAlgoBilinear;
        vHparams.vRopeTheta = 10000.0f;
        mGetU32(KEY_PROJ_SCALE_FACTOR, vHparams.vNMerge, false);
        // TODO: check kimivl preprocessor for exact values
        vHparams.mSetLimitImageTokens(8, 1024);
        vHparams.mSetWarmupNTokens(256); // avoid OOM on warmup
      } break;
      case cProjectorTypeKimik25: {
        vHparams.vImageResizeAlgo = cResizeAlgoBicubic;
        vHparams.vRopeTheta = 10000.0f;
        mGetU32(KEY_PROJ_SCALE_FACTOR, vHparams.vNMerge, false);

        mGetU32(KEY_IMAGE_MIN_PIXELS, vHparams.vImageMinPixels);
        mGetU32(KEY_IMAGE_MAX_PIXELS, vHparams.vImageMaxPixels);
        if (vHparams.vImageMinPixels <= 0 || vHparams.vImageMaxPixels <= 0) {
          throw std::runtime_error(fStringFormat("%s: kimik25: %s and %s must be greater than 0\n", __func__,
                                                 KEY_IMAGE_MIN_PIXELS, KEY_IMAGE_MAX_PIXELS));
        }
        vHparams.vWarmupImageSize = static_cast<int>(std::sqrt(vHparams.vImageMaxPixels));
      } break;
      case cProjectorTypeGemma3: {
        // default value (used by all model sizes in gemma 3 family)
        // number of patches for each **side** is reduced by a factor of 4
        vHparams.vNMerge = 4;
        vHparams.vImageResizeAlgo = cResizeAlgoBilinear;
        // test model (tinygemma3) has a different value, we optionally read it
        mGetU32(KEY_PROJ_SCALE_FACTOR, vHparams.vNMerge, false);
      } break;

      case cProjectorTypeGemma3nv: {
        // Gemma3n uses MobileNetV5 which produces 256 tokens (16x16)
        // Similar configuration to Gemma3
        vHparams.vNMerge = 1; // MobileNetV5 handles resizing internally
        mGetU32(KEY_PROJ_SCALE_FACTOR, vHparams.vNMerge, false);
      } break;
      case cProjectorTypeQwen2vl:
      case cProjectorTypeQwen25vl:
      case cProjectorTypeQwen3vl: {
        vHparams.vNMerge = 2; // default value for Qwen 2 and 2.5
        vHparams.vImageResizeAlgo = cResizeAlgoBilinear;
        mGetU32(KEY_SPATIAL_MERGE_SIZE, vHparams.vNMerge, false);
        mGetU32(KEY_WIN_ATTN_PATTERN, vHparams.vNWaPattern,
                pModel.vProjType == cProjectorTypeQwen25vl); // only 2.5 requires it
        // ref: https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/blob/main/preprocessor_config.json
        vHparams.mSetLimitImageTokens(8, 4096);
        vHparams.mSetWarmupNTokens(46 * 46); // avoid OOM on warmup
        const int cWarnMinPixels =
            1024 * vHparams.vNMerge * vHparams.vNMerge * vHparams.vPatchSize * vHparams.vPatchSize;
        if (vHparams.vImageMinPixels < cWarnMinPixels) {
          LOG_WRN("%s: Qwen-VL models require at minimum 1024 image tokens to function correctly on grounding tasks\n",
                  __func__);
          LOG_WRN("%s: if you encounter problems with accuracy, try adding --image-min-tokens 1024\n\n", __func__);
        }
      } break;
      case cProjectorTypeMinimaxM3: {
        vHparams.vNMerge = 2; // spatial_merge_size
        vHparams.vImageResizeAlgo = cResizeAlgoBicubicPillow;
        vHparams.vImageResizePad = cPadNone;
        mGetU32(KEY_SPATIAL_MERGE_SIZE, vHparams.vNMerge, false);
        vHparams.vRopeTheta = 10000.0f; // vision_config.rope_theta
        // MiniMax-M3: max_pixels 451584 (=672^2) -> 576 merged tokens (image_seq_length)
        vHparams.mSetLimitImageTokens(8, 576);
        vHparams.mSetWarmupNTokens(16 * 16);
      } break;
      case cProjectorTypeMimovl: {
        vHparams.vNMerge = 2; // spatial_merge_size
        vHparams.vImageResizeAlgo = cResizeAlgoBicubicPillow;
        mGetU32(KEY_SPATIAL_MERGE_SIZE, vHparams.vNMerge, false);
        mGetU32(fStringFormat(KEY_N_HEAD_KV, "vision"), vHparams.vNHeadKv);
        // 1D banded sliding-window radius (visual_token_window_size); required
        mGetU32(KEY_ATTN_WINDOW_SIZE, vHparams.vAttnWindowSize);
        std::vector<int> lPat;
        mGetArrInt(KEY_WA_PATTERN_MODE, lPat, true);
        GGML_ASSERT((int)lPat.size() == vHparams.vNLayer && "mimovl wa_pattern_mode length must equal n_layer");
        vHparams.lWaPatternMode.assign(lPat.begin(), lPat.end());
        mGetU32(KEY_IMAGE_MIN_PIXELS, vHparams.vImageMinPixels);
        mGetU32(KEY_IMAGE_MAX_PIXELS, vHparams.vImageMaxPixels);
        vHparams.mSetWarmupNTokens(46 * 46); // avoid OOM on warmup
      } break;
      case cProjectorTypeStep3vl: {
        vHparams.vNMerge = 4; // two stride-2 downsamplers after patching
        mGetU32(KEY_PROJ_SCALE_FACTOR, vHparams.vNMerge, false);
        vHparams.vRopeTheta = 10000.0f;
        mGetU32(KEY_PREPROC_IMAGE_SIZE, vHparams.vImageLongestEdge);
        if (vHparams.vImageLongestEdge <= 0) {
          throw std::runtime_error(
              fStringFormat("%s: step3vl: %s must be greater than 0\n", __func__, KEY_PREPROC_IMAGE_SIZE));
        }
        vHparams.vWarmupImageSize = vHparams.vImageSize;
      } break;
      case cProjectorTypeYoutuvl: {
        vHparams.vNMerge = 2;
        vHparams.vImageResizeAlgo = cResizeAlgoBilinear;
        vHparams.vImageResizePad = cPadNone;
        mGetU32(KEY_SPATIAL_MERGE_SIZE, vHparams.vNMerge, false);
        mGetU32(KEY_ATTN_WINDOW_SIZE, vHparams.vAttnWindowSize, true);
        std::vector<int> lWaLayerIndexesVec;
        mGetArrInt(KEY_WIN_ATTN_LAYER_INDEXES, lWaLayerIndexesVec, true);
        for (auto &vLayer : lWaLayerIndexesVec) {
          vHparams.vWaLayerIndexes.insert(vLayer);
        }
        // support max_height * max_width = 8000 * 8000. 8000/16/2 = 250 image tokens
        vHparams.mSetLimitImageTokens(1, 62500);
        vHparams.mSetWarmupNTokens(16 * 16); // avoid OOM on warmup
      } break;
      case cProjectorTypeYasa2: {
        vHparams.vFfnOp = cFfnGeluErf;
        vLogFfnOp = "gelu_erf";
        vHparams.vImageResizeAlgo = cResizeAlgoBicubic;

        // reka model performs better when using resize_bicubic, which stretches
        // the image to fit fixed square size
        vHparams.vImageResizePad = cPadNone;
      } break;
      case cProjectorTypeGlm4v: {
        vHparams.vRopeTheta = 10000.0f;
        vHparams.vNMerge = 2; // default value for GLM4-V
        vHparams.vImageResizeAlgo = cResizeAlgoBilinear;
        mGetU32(KEY_SPATIAL_MERGE_SIZE, vHparams.vNMerge, false);
        vHparams.mSetLimitImageTokens(8, 4096);
        vHparams.mSetWarmupNTokens(46 * 46); // avoid OOM on warmup
      } break;
      case cProjectorTypeGlm5v: {
        // glm4v tower with clamped SwiGLU, ceil-aligned resize and its own token budget
        vHparams.vRopeTheta = 10000.0f;
        vHparams.vNMerge = 2;
        vHparams.vImageResizeAlgo = cResizeAlgoBicubic;
        mGetU32(KEY_SPATIAL_MERGE_SIZE, vHparams.vNMerge, false);
        mGetF32(KEY_SWIGLU_CLAMP, vHparams.vSwigluClamp, true);
        mGetU32(KEY_IMAGE_MIN_PIXELS, vHparams.vImageMinPixels);
        mGetU32(KEY_IMAGE_MAX_PIXELS, vHparams.vImageMaxPixels);
        vHparams.mSetLimitImageTokens();
        vHparams.mSetWarmupNTokens(46 * 46); // avoid OOM on warmup
      } break;
      case cProjectorTypeUltravox:
      case cProjectorTypeQwen2a:
      case cProjectorTypeQwen3a:
      case cProjectorTypeGlma:
      case cProjectorTypeVoxtral:
      case cProjectorTypeMeralion:
      case cProjectorTypeMusicFlamingo: {
        bool vRequireStack = pModel.vProjType == cProjectorTypeUltravox || pModel.vProjType == cProjectorTypeVoxtral ||
                             pModel.vProjType == cProjectorTypeMeralion || pModel.vProjType == cProjectorTypeGlma;
        mGetU32(KEY_A_PROJ_STACK_FACTOR, vHparams.vProjStackFactor, vRequireStack);
        vHparams.vFfnOp = cFfnGeluErf;
        vLogFfnOp = "gelu_erf"; // temporary solution for logging

        // audio preprocessing params
        vHparams.vAudioChunkLen = 30; // in seconds
        vHparams.vAudioSampleRate = 16000;
        vHparams.vAudioNFft = 400;
        vHparams.vAudioWindowLen = 400;
        vHparams.vAudioHopLen = 160;
      } break;
      case cProjectorTypeMimoAudio: {
        mGetU32(KEY_A_RVQ_NUM_QUANTIZERS, vHparams.vRvqNumQuantizers, false);
        mGetArrInt(KEY_A_RVQ_CODEBOOK_SIZE, vHparams.lRvqCodebookSize, false);
        if (vHparams.vRvqNumQuantizers <= 0) {
          throw std::runtime_error(fStringFormat("%s: mimo_audio: missing %s\n", __func__, KEY_A_RVQ_NUM_QUANTIZERS));
        }
        if ((int)vHparams.lRvqCodebookSize.size() != vHparams.vRvqNumQuantizers) {
          throw std::runtime_error(fStringFormat("%s: mimo_audio: %s length (%zu) must equal %s (%d)\n", __func__,
                                                 KEY_A_RVQ_CODEBOOK_SIZE, vHparams.lRvqCodebookSize.size(),
                                                 KEY_A_RVQ_NUM_QUANTIZERS, vHparams.vRvqNumQuantizers));
        }
        vHparams.vFfnOp = cFfnGeluErf; // PyTorch F.gelu default (approximate="none")
        vHparams.vRopeTheta = 10000.0f;

        // audio preprocessing params (mel spectrogram)
        vHparams.vAudioSampleRate = 24000;
        vHparams.vAudioNFft = 960;
        vHparams.vAudioWindowLen = 960;
        vHparams.vAudioHopLen = 240;

        mGetU32(KEY_A_ATTN_WINDOW_SIZE, vHparams.vAttnWindowSize);
        std::vector<int> lWaPattern;
        mGetArrInt(KEY_A_WA_PATTERN_MODE, lWaPattern, true);
        if ((int)lWaPattern.size() != vHparams.vNLayer) {
          throw std::runtime_error(fStringFormat("%s: mimo_audio: %s length (%zu) must equal n_layer (%d)\n", __func__,
                                                 KEY_A_WA_PATTERN_MODE, lWaPattern.size(), vHparams.vNLayer));
        }
        vHparams.lWaPatternMode.assign(lWaPattern.begin(), lWaPattern.end());

        mGetU32(KEY_A_LOCAL_BLOCK_COUNT, vHparams.vAudioLocalNLayer);
        mGetU32(KEY_A_LOCAL_GROUP_SIZE, vHparams.vAudioLocalGroupSize);
        if (vHparams.vAudioLocalGroupSize <= 0) {
          throw std::runtime_error(fStringFormat("%s: mimo_audio: %s must be > 0\n", __func__, KEY_A_LOCAL_GROUP_SIZE));
        }
      } break;
      case cProjectorTypePaddleocr: {
        vHparams.vNMerge = 2;
        vHparams.vImageResizeAlgo = cResizeAlgoBilinear;
        mGetU32(KEY_IMAGE_MIN_PIXELS, vHparams.vImageMinPixels);
        mGetU32(KEY_IMAGE_MAX_PIXELS, vHparams.vImageMaxPixels);

        vHparams.mSetWarmupNTokens(28 * 28); // avoid OOM on warmup
      } break;
      case cProjectorTypeHunyuanvl: {
        vHparams.vNMerge = 2;
        vHparams.vImageResizeAlgo = cResizeAlgoBicubicPillow;
        vHparams.vImageResizePad = cPadNone;
        vHparams.vFfnOp = cFfnGelu;
        vHparams.mSetLimitImageTokens(256, 16384);
        mGetU32(KEY_SPATIAL_MERGE_SIZE, vHparams.vNMerge, false);
        mGetU32(KEY_IMAGE_MIN_PIXELS, vHparams.vImageMinPixels, false);
        mGetU32(KEY_IMAGE_MAX_PIXELS, vHparams.vImageMaxPixels, false);
        vHparams.mSetWarmupNTokens(32 * 32);
      } break;
      case cProjectorTypeLfm2a: {
        // audio preprocessing params
        vHparams.vAudioChunkLen = 1; // in seconds
        vHparams.vAudioSampleRate = 16000;
        vHparams.vAudioNFft = 512;
        vHparams.vAudioWindowLen = 400;
        vHparams.vAudioHopLen = 160;
      } break;
      case cProjectorTypeEXAONE4Point5: {
        vHparams.vNMerge = 2;
        mGetU32(KEY_SPATIAL_MERGE_SIZE, vHparams.vNMerge, false);
        mGetU32(KEY_WIN_ATTN_PATTERN, vHparams.vNWaPattern, false);
        mGetU32(KEY_IMAGE_MIN_PIXELS, vHparams.vImageMinPixels);
        mGetU32(KEY_IMAGE_MAX_PIXELS, vHparams.vImageMaxPixels);
        vHparams.mSetWarmupNTokens(46 * 46);
        if (vHparams.vRopeTheta <= 0.0f) {
          vHparams.vRopeTheta = 10000.0f;
        }
        mGetU32(fStringFormat(KEY_N_HEAD_KV, "vision"), vHparams.vNHeadKv);
      } break;
      case cProjectorTypeGraniteSpeech: {
        vHparams.vAudioChunkLen = 0;
        vHparams.vAudioSampleRate = 16000;
        vHparams.vAudioNFft = 512;
        vHparams.vAudioWindowLen = 400;
        vHparams.vAudioHopLen = 160;
        mGetU32(KEY_A_CHUNK_SIZE, vHparams.vAudioChunkSize);
        mGetU32(KEY_A_CONV_KERNEL_SIZE, vHparams.vAudioConvKernelSize);
        mGetU32(KEY_A_MAX_POS_EMB, vHparams.vAudioMaxPosEmb);
        mGetU32(KEY_A_PROJ_WINDOW_SIZE, vHparams.vAudioProjWindowSize);
        mGetU32(KEY_A_PROJ_DOWNSAMPLE_RATE, vHparams.vAudioProjDownsampleRate);
        mGetU32(KEY_A_PROJ_HEAD_COUNT, vHparams.vAudioProjHeadCount);
        // NOTE: feature layers loaded above in common path
      } break;
      case cProjectorTypeJanusPro: {
        vHparams.aImagePadColor = {127, 127, 127};
        vHparams.vImageResizeAlgo = cResizeAlgoBilinear;
      } break;
      case cProjectorTypeGranite4Vision: {
        // SigLIP tower.
        vHparams.vImageResizeAlgo = cResizeAlgoBicubicPillow;
        vHparams.vImageResizePad = cPadCeil;

        // NOTE: feature_layers loaded in common path as optional
        mGetArrInt(KEY_PROJ_SPATIAL_OFFSETS, vHparams.lProjSpatialOffsets);
        if (vHparams.lFeatureLayers.size() != vHparams.lProjSpatialOffsets.size()) {
          throw std::runtime_error(fStringFormat("%s: feature_layers.size() %d != proj_spatial_offsets.size() %d",
                                                 vHparams.lFeatureLayers.size(), vHparams.lProjSpatialOffsets.size()));
        }

        mGetU32(KEY_PROJ_SAMPLE_QUERY_SIDE, vHparams.vDownsampleQuerySide);
        mGetU32(KEY_PROJ_SAMPLE_WINDOW_SIDE, vHparams.vDownsampleWindowSide);
        vHparams.vWarmupImageSize = vHparams.vImageSize;
      } break;
      default:
        throw std::runtime_error(fStringFormat("%s: unknown vision projector type %s\n", __func__, vProjType.c_str()));
      }

      // sanity check
      {
        if (vHparams.vImageSize < 0) {
          // note: some models having hparams.vImageSize == 0, which means the image size is dynamic
          throw std::runtime_error(
              fStringFormat("%s: image_size (%d) cannot be negative\n", __func__, vHparams.vImageSize));
        }
        if (vHparams.vImageSize > 65536) {
          throw std::runtime_error(
              fStringFormat("%s: image_size (%d) is too large (max 65536)\n", __func__, vHparams.vImageSize));
        }
        if (vHparams.vPatchSize <= 0 || vHparams.vPatchSize >= 65536) {
          throw std::runtime_error(fStringFormat("%s: patch_size (%d) must be positive and less than 65536\n", __func__,
                                                 vHparams.vPatchSize));
        }
        if (vHparams.vNEmbd <= 0) {
          throw std::runtime_error(
              fStringFormat("%s: n_embd (%d) must be greater than 0\n", __func__, vHparams.vNEmbd));
        }
        if (vHparams.vImageMaxPixels < vHparams.vImageMinPixels) {
          throw std::runtime_error(fStringFormat("%s: image_max_pixels (%d) is less than image_min_pixels (%d)\n",
                                                 __func__, vHparams.vImageMaxPixels, vHparams.vImageMinPixels));
        }
        if (vHparams.vNMerge < 0 || vHparams.vNMerge >= 65536) {
          throw std::runtime_error(fStringFormat("%s: n_merge (%d) must be greater than 0 and less than 65536\n",
                                                 __func__, vHparams.vNMerge));
        }
      }

      LOG_INF("%s: projector:          %s\n", __func__, vProjType.c_str());
      LOG_INF("%s: n_embd:             %d\n", __func__, vHparams.vNEmbd);
      LOG_INF("%s: n_head:             %d\n", __func__, vHparams.vNHead);
      LOG_INF("%s: n_ff:               %d\n", __func__, vHparams.vNFf);
      LOG_INF("%s: n_layer:            %d\n", __func__, vHparams.vNLayer);
      LOG_INF("%s: ffn_op:             %s\n", __func__, vLogFfnOp.c_str());
      LOG_INF("%s: projection_dim:     %d\n", __func__, vHparams.vProjectionDim);
      if (cIsVision) {
        LOG_INF("\n--- vision hparams ---\n");
        LOG_INF("%s: image_size:         %d\n", __func__, vHparams.vImageSize);
        LOG_INF("%s: patch_size:         %d\n", __func__, vHparams.vPatchSize);
        LOG_INF("%s: n_merge:            %d\n", __func__, vHparams.vNMerge);
        LOG_INF("%s: n_wa_pattern: %d\n", __func__, vHparams.vNWaPattern);
        if (!vHparams.vWaLayerIndexes.empty()) {
          LOG_INF("%s: wa_layer_indexes:  ", __func__);
          for (auto &vLayer : vHparams.vWaLayerIndexes) {
            LOG_INF("%d ", vLayer);
          }
          LOG_INF("\n");
        }
        if (vHparams.vImageMinPixels > 0) {
          LOG_INF("%s: image_min_pixels:   %d%s\n", __func__, vHparams.vImageMinPixels,
                  vHparams.vCustomImageMinTokens > 0 ? " (custom value)" : "");
        }
        if (vHparams.vImageMaxPixels > 0) {
          LOG_INF("%s: image_max_pixels:   %d%s\n", __func__, vHparams.vImageMaxPixels,
                  vHparams.vCustomImageMaxTokens > 0 ? " (custom value)" : "");
        }
      } else if (cIsAudio) {
        LOG_INF("\n--- audio hparams ---\n");
        LOG_INF("%s: n_mel_bins:         %d\n", __func__, vHparams.vNMelBins);
        LOG_INF("%s: proj_stack_factor:  %d\n", __func__, vHparams.vProjStackFactor);
        LOG_INF("%s: audio_chunk_len:    %d\n", __func__, vHparams.vAudioChunkLen);
        LOG_INF("%s: audio_sample_rate:  %d\n", __func__, vHparams.vAudioSampleRate);
        LOG_INF("%s: audio_n_fft:        %d\n", __func__, vHparams.vAudioNFft);
        LOG_INF("%s: audio_window_len:   %d\n", __func__, vHparams.vAudioWindowLen);
        LOG_INF("%s: audio_hop_len:      %d\n", __func__, vHparams.vAudioHopLen);

        // Validate audio hparams loaded from GGUF metadata
        if (vHparams.vNMelBins <= 0 || vHparams.vNMelBins > 256) {
          throw std::runtime_error(
              fStringFormat("%s: n_mel_bins (%d) must be in range [1, 256]\n", __func__, vHparams.vNMelBins));
        }
        if (vHparams.vAudioSampleRate <= 0 || vHparams.vAudioNFft <= 0 || vHparams.vAudioHopLen <= 0 ||
            vHparams.vAudioWindowLen <= 0) {
          throw std::runtime_error(fStringFormat(
              "%s: audio hparams invalid: sample_rate=%d n_fft=%d window_len=%d hop_len=%d\n", __func__,
              vHparams.vAudioSampleRate, vHparams.vAudioNFft, vHparams.vAudioWindowLen, vHparams.vAudioHopLen));
        }
      }
      LOG_INF("\n");
      LOG_INF("%s: model size:         %.2f MiB\n", __func__, vModelSize / 1024.0 / 1024.0);
      LOG_INF("%s: metadata size:      %.2f MiB\n", __func__, ggml_get_mem_size(vCtxMeta.get()) / 1024.0 / 1024.0);
    }
  }

  void mLoadTensors(clip_ctx &pCtxClip) {
    auto &vModel = pCtxClip.vModel;
    auto &vHparams = vModel.vHparams;
    std::map<std::string, size_t> dTensorOffset;
    std::vector<ggml_tensor *> lTensorsToLoad;

    auto vFin = fOpenIfstreamBinary(vFname);
    if (!vFin) {
      throw std::runtime_error(fStringFormat("%s: failed to open %s\n", __func__, vFname.c_str()));
    }

    // TODO @ngxson : support both audio and video in the future
    const char *vPrefix = vModel.vModality == cClipModalityAudio ? "a" : "v";

    // get offsets
    for (int64_t vI = 0; vI < gguf_get_n_tensors(vCtxGguf.get()); ++vI) {
      const char *vName = gguf_get_tensor_name(vCtxGguf.get(), vI);
      dTensorOffset[vName] = gguf_get_data_offset(vCtxGguf.get()) + gguf_get_tensor_offset(vCtxGguf.get(), vI);
    }

    // create data context
    struct ggml_init_params vParams = {
        /*.mem_size =*/static_cast<size_t>(gguf_get_n_tensors(vCtxGguf.get()) + 1) * ggml_tensor_overhead(),
        /*.mem_buffer =*/NULL,
        /*.no_alloc =*/true,
    };
    pCtxClip.vCtxData.reset(ggml_init(vParams));
    if (!pCtxClip.vCtxData) {
      throw std::runtime_error(fStringFormat("%s: failed to init ggml context\n", __func__));
    }

    // helper function
    std::unordered_set<std::string> sLoadedTensorNames;
    auto vGetTensor = [&](const std::string &pName, bool pRequired = true) {
      // Each tensor should only be loaded once; duplicates indicate a bug
      if (sLoadedTensorNames.count(pName)) {
        throw std::runtime_error(fStringFormat("%s: tensor already loaded: %s\n", __func__, pName.c_str()));
      }
      ggml_tensor *vCur = ggml_get_tensor(vCtxMeta.get(), pName.c_str());
      if (!vCur && pRequired) {
        throw std::runtime_error(fStringFormat("%s: unable to find tensor %s\n", __func__, pName.c_str()));
      }
      if (vCur) {
        lTensorsToLoad.push_back(vCur);
        ggml_tensor *vDataTensor = ggml_dup_tensor(pCtxClip.vCtxData.get(), vCur);
        ggml_set_name(vDataTensor, vCur->name);
        sLoadedTensorNames.insert(pName);
        vCur = vDataTensor;
        // add to weight memory counter
        pCtxClip.dMemUsage[ggml_backend_get_device(pCtxClip.vBackend)] += ggml_nbytes(vCur);
      }
      return vCur;
    };

    auto vGetVector = [&](const std::string &pName) {
      std::vector<float> lResult;
      auto vIt = dTensorOffset.find(pName);
      if (vIt == dTensorOffset.end()) {
        return lResult;
      }

      const int64_t cIdx = gguf_find_tensor(vCtxGguf.get(), pName.c_str());
      if (cIdx < 0) {
        throw std::runtime_error(fStringFormat("%s: failed to find tensor %s\n", __func__, pName.c_str()));
      }

      if (const auto cType = gguf_get_tensor_type(vCtxGguf.get(), cIdx); cType != GGML_TYPE_F32) {
        throw std::runtime_error(fStringFormat("%s: %s must be %s, was %s\n", __func__, pName.c_str(),
                                               ggml_type_name(GGML_TYPE_F32), ggml_type_name(cType)));
      }

      const size_t cNBytes = gguf_get_tensor_size(vCtxGguf.get(), cIdx);
      if (cNBytes == 0) {
        throw std::runtime_error(fStringFormat("%s: tensor %s is empty\n", __func__, pName.c_str()));
      }

      const size_t cNElems = cNBytes / sizeof(float);
      lResult.resize(cNElems);
      vFin.seekg(vIt->second, std::ios::beg);
      vFin.read(reinterpret_cast<char *>(lResult.data()), cNBytes);
      return lResult;
    };

    vModel.vClassEmbedding = vGetTensor(TN_CLASS_EMBD, false);

    vModel.vPreLnW = vGetTensor(fStringFormat(TN_LN_PRE, vPrefix, "weight"), false);
    vModel.vPreLnB = vGetTensor(fStringFormat(TN_LN_PRE, vPrefix, "bias"), false);

    vModel.vPostLnW = vGetTensor(fStringFormat(TN_LN_POST, vPrefix, "weight"), false);
    vModel.vPostLnB = vGetTensor(fStringFormat(TN_LN_POST, vPrefix, "bias"), false);

    vModel.vPatchBias = vGetTensor(TN_PATCH_BIAS, false);
    vModel.vPatchEmbeddings0 = vGetTensor(TN_PATCH_EMBD, false);
    vModel.vPatchEmbeddings1 = vGetTensor(TN_PATCH_EMBD_1, false);

    vModel.vNormEmbdW = vGetTensor(fStringFormat(TN_NORM_EMBD, "weight"), false);
    vModel.vNormEmbdB = vGetTensor(fStringFormat(TN_NORM_EMBD, "bias"), false);

    vModel.vPositionEmbeddings = vGetTensor(fStringFormat(TN_POS_EMBD, vPrefix), false);

    const bool cHasStandardLayers = (vModel.vProjType != cProjectorTypeGemma3nv);

    // layers
    const int cNLayersToLoad = cHasStandardLayers ? vHparams.vNLayer : 0;
    vModel.lLayers.resize(cNLayersToLoad);
    for (int vIl = 0; vIl < cNLayersToLoad; ++vIl) {
      auto &vLayer = vModel.lLayers[vIl];
      vLayer.vKW = vGetTensor(fStringFormat(TN_ATTN_K, vPrefix, vIl, "weight"), false);
      vLayer.vQW = vGetTensor(fStringFormat(TN_ATTN_Q, vPrefix, vIl, "weight"), false);
      vLayer.vVW = vGetTensor(fStringFormat(TN_ATTN_V, vPrefix, vIl, "weight"), false);
      vLayer.vOW = vGetTensor(fStringFormat(TN_ATTN_OUTPUT, vPrefix, vIl, "weight"));
      vLayer.vQkvW = vGetTensor(fStringFormat(TN_ATTN_QKV, vPrefix, vIl, "weight"), false);
      vLayer.vKNorm = vGetTensor(fStringFormat(TN_ATTN_K_NORM, vPrefix, vIl, "weight"), false);
      vLayer.vQNorm = vGetTensor(fStringFormat(TN_ATTN_Q_NORM, vPrefix, vIl, "weight"), false);
      vLayer.vLn1W = vGetTensor(fStringFormat(TN_LN_1, vPrefix, vIl, "weight"), false);
      vLayer.vLn2W = vGetTensor(fStringFormat(TN_LN_2, vPrefix, vIl, "weight"), false);
      vLayer.vLs1W = vGetTensor(fStringFormat(TN_LS_1, vPrefix, vIl, "weight"), false);                    // no bias
      vLayer.vLs2W = vGetTensor(fStringFormat(TN_LS_2, vPrefix, vIl, "weight"), false);                    // no bias
      vLayer.vLsOutW = vGetTensor(fStringFormat(TN_LS_OUT, vPrefix, vIl, "weight"), false);                // no bias
      vLayer.vAttnPostNormW = vGetTensor(fStringFormat(TN_ATTN_POST_NORM, vPrefix, vIl, "weight"), false); // no bias
      vLayer.vFfPostNormW = vGetTensor(fStringFormat(TN_FFN_POST_NORM, vPrefix, vIl, "weight"), false);    // no bias

      vLayer.vKB = vGetTensor(fStringFormat(TN_ATTN_K, vPrefix, vIl, "bias"), false);
      vLayer.vQB = vGetTensor(fStringFormat(TN_ATTN_Q, vPrefix, vIl, "bias"), false);
      vLayer.vVB = vGetTensor(fStringFormat(TN_ATTN_V, vPrefix, vIl, "bias"), false);
      vLayer.vOB = vGetTensor(fStringFormat(TN_ATTN_OUTPUT, vPrefix, vIl, "bias"), false);
      vLayer.vQkvB = vGetTensor(fStringFormat(TN_ATTN_QKV, vPrefix, vIl, "bias"), false);
      vLayer.vLn1B = vGetTensor(fStringFormat(TN_LN_1, vPrefix, vIl, "bias"), false);
      vLayer.vLn2B = vGetTensor(fStringFormat(TN_LN_2, vPrefix, vIl, "bias"), false);

      // ffn
      vLayer.vFfUpW = vGetTensor(fStringFormat(TN_FFN_UP, vPrefix, vIl, "weight"));
      vLayer.vFfUpB = vGetTensor(fStringFormat(TN_FFN_UP, vPrefix, vIl, "bias"), false);
      vLayer.vFfGateW = vGetTensor(fStringFormat(TN_FFN_GATE, vPrefix, vIl, "weight"), false);
      vLayer.vFfGateB = vGetTensor(fStringFormat(TN_FFN_GATE, vPrefix, vIl, "bias"), false);
      vLayer.vFfDownW = vGetTensor(fStringFormat(TN_FFN_DOWN, vPrefix, vIl, "weight"));
      vLayer.vFfDownB = vGetTensor(fStringFormat(TN_FFN_DOWN, vPrefix, vIl, "bias"), false);

      // mimovl per-head attention sink bias
      vLayer.vAttnSinks = vGetTensor(fStringFormat(TN_ATTN_SINKS, vPrefix, vIl), false);

      // qwen3vl deepstack layer
      vLayer.vDeepstackNormW = vGetTensor(fStringFormat(TN_DEEPSTACK_NORM, vIl, "weight"), false);
      vLayer.vDeepstackNormB = vGetTensor(fStringFormat(TN_DEEPSTACK_NORM, vIl, "bias"), false);
      vLayer.vDeepstackFc1W = vGetTensor(fStringFormat(TN_DEEPSTACK_FC1, vIl, "weight"), false);
      vLayer.vDeepstackFc1B = vGetTensor(fStringFormat(TN_DEEPSTACK_FC1, vIl, "bias"), false);
      vLayer.vDeepstackFc2W = vGetTensor(fStringFormat(TN_DEEPSTACK_FC2, vIl, "weight"), false);
      vLayer.vDeepstackFc2B = vGetTensor(fStringFormat(TN_DEEPSTACK_FC2, vIl, "bias"), false);
      if (vLayer.mHasDeepstack()) {
        vModel.vNDeepstackLayers++;
      }
    }

    switch (vModel.vProjType) {
    case cProjectorTypeMINICPMV4Point6: {
      const bool cMergerRequired = vHparams.vNMerge == 4;
      auto vGetMergerTensor = [&](const std::string &pName, bool pRequired = true) {
        return vGetTensor(pName, cMergerRequired && pRequired);
      };

      // ViT merger: window self-attention
      vModel.vVitMergerLn1W = vGetMergerTensor(fStringFormat(TN_VIT_MERGER_LN1, "weight"));
      vModel.vVitMergerLn1B = vGetMergerTensor(fStringFormat(TN_VIT_MERGER_LN1, "bias"));
      vModel.vVitMergerAttnQW = vGetMergerTensor(fStringFormat(TN_VIT_MERGER_ATTN_Q, "weight"));
      vModel.vVitMergerAttnQB = vGetMergerTensor(fStringFormat(TN_VIT_MERGER_ATTN_Q, "bias"), false);
      vModel.vVitMergerAttnKW = vGetMergerTensor(fStringFormat(TN_VIT_MERGER_ATTN_K, "weight"));
      vModel.vVitMergerAttnKB = vGetMergerTensor(fStringFormat(TN_VIT_MERGER_ATTN_K, "bias"), false);
      vModel.vVitMergerAttnVW = vGetMergerTensor(fStringFormat(TN_VIT_MERGER_ATTN_V, "weight"));
      vModel.vVitMergerAttnVB = vGetMergerTensor(fStringFormat(TN_VIT_MERGER_ATTN_V, "bias"), false);
      vModel.vVitMergerAttnOW = vGetMergerTensor(fStringFormat(TN_VIT_MERGER_ATTN_O, "weight"));
      vModel.vVitMergerAttnOB = vGetMergerTensor(fStringFormat(TN_VIT_MERGER_ATTN_O, "bias"), false);
      // ViT merger: MLP downsample
      vModel.vVitMergerDsLnW = vGetMergerTensor(fStringFormat(TN_VIT_MERGER_DS_LN, "weight"));
      vModel.vVitMergerDsLnB = vGetMergerTensor(fStringFormat(TN_VIT_MERGER_DS_LN, "bias"));
      vModel.vVitMergerDsUpW = vGetMergerTensor(fStringFormat(TN_VIT_MERGER_DS_UP, "weight"));
      vModel.vVitMergerDsUpB = vGetMergerTensor(fStringFormat(TN_VIT_MERGER_DS_UP, "bias"), false);
      vModel.vVitMergerDsDownW = vGetMergerTensor(fStringFormat(TN_VIT_MERGER_DS_DOWN, "weight"));
      vModel.vVitMergerDsDownB = vGetMergerTensor(fStringFormat(TN_VIT_MERGER_DS_DOWN, "bias"), false);
      // Final Merger (DownsampleMLP)
      vModel.vMmInputNormW = vGetTensor(TN_MM_INP_NORM);
      vModel.vMmInputNormB = vGetTensor(TN_MM_INP_NORM_B, false);
      vModel.vMmFfnUpW = vGetTensor(fStringFormat(TN_MM_UP, "weight"));
      vModel.vMmFfnUpB = vGetTensor(fStringFormat(TN_MM_UP, "bias"), false);
      vModel.vMmFfnDownW = vGetTensor(fStringFormat(TN_MM_DOWN, "weight"));
      vModel.vMmFfnDownB = vGetTensor(fStringFormat(TN_MM_DOWN, "bias"), false);
    } break;
    case cProjectorTypeQwen2vl:
    case cProjectorTypeQwen25vl:
    case cProjectorTypeEXAONE4Point5: {
      vModel.vMm0W = vGetTensor(fStringFormat(TN_LLAVA_PROJ, 0, "weight"));
      vModel.vMm0B = vGetTensor(fStringFormat(TN_LLAVA_PROJ, 0, "bias"));
      vModel.vMm1W = vGetTensor(fStringFormat(TN_LLAVA_PROJ, 2, "weight"));
      vModel.vMm1B = vGetTensor(fStringFormat(TN_LLAVA_PROJ, 2, "bias"));
    } break;
    case cProjectorTypeQwen3vl: {
      vModel.vMm0W = vGetTensor(fStringFormat(TN_LLAVA_PROJ, 0, "weight"));
      vModel.vMm0B = vGetTensor(fStringFormat(TN_LLAVA_PROJ, 0, "bias"));
      vModel.vMm1W = vGetTensor(fStringFormat(TN_LLAVA_PROJ, 2, "weight"));
      vModel.vMm1B = vGetTensor(fStringFormat(TN_LLAVA_PROJ, 2, "bias"));
    } break;
    case cProjectorTypeMimovl: {
      vModel.vMm0W = vGetTensor(fStringFormat(TN_LLAVA_PROJ, 0, "weight"));
      vModel.vMm0B = vGetTensor(fStringFormat(TN_LLAVA_PROJ, 0, "bias"), false);
      vModel.vMm1W = vGetTensor(fStringFormat(TN_LLAVA_PROJ, 2, "weight"));
      vModel.vMm1B = vGetTensor(fStringFormat(TN_LLAVA_PROJ, 2, "bias"), false);
    } break;
    case cProjectorTypeMinimaxM3: {
      // per-patch MLP: mm.1 -> gelu -> mm.2
      vModel.vMm1W = vGetTensor(fStringFormat(TN_LLAVA_PROJ, 1, "weight"));
      vModel.vMm1B = vGetTensor(fStringFormat(TN_LLAVA_PROJ, 1, "bias"));
      vModel.vMm2W = vGetTensor(fStringFormat(TN_LLAVA_PROJ, 2, "weight"));
      vModel.vMm2B = vGetTensor(fStringFormat(TN_LLAVA_PROJ, 2, "bias"));
      // 2x2 merge MLP: mm.merge.fc1 -> gelu -> mm.merge.fc2
      vModel.vMmMergerFc1W = vGetTensor(fStringFormat(TN_MM_MERGER_FC1, "weight"));
      vModel.vMmMergerFc1B = vGetTensor(fStringFormat(TN_MM_MERGER_FC1, "bias"));
      vModel.vMmMergerFc2W = vGetTensor(fStringFormat(TN_MM_MERGER_FC2, "weight"));
      vModel.vMmMergerFc2B = vGetTensor(fStringFormat(TN_MM_MERGER_FC2, "bias"));
    } break;
    case cProjectorTypeStep3vl: {
      vModel.vMm0W = vGetTensor(fStringFormat(TN_LLAVA_PROJ, 0, "weight"));
      vModel.vMm0B = vGetTensor(fStringFormat(TN_LLAVA_PROJ, 0, "bias"), false);
      vModel.vMm1W = vGetTensor(fStringFormat(TN_LLAVA_PROJ, 1, "weight"));
      vModel.vMm1B = vGetTensor(fStringFormat(TN_LLAVA_PROJ, 1, "bias"), false);
      vModel.vMmModelProj = vGetTensor(fStringFormat(TN_MM_PROJECTOR, "weight"));
    } break;
    case cProjectorTypeYoutuvl: {
      vModel.vMmInputNormW = vGetTensor(TN_MM_INP_NORM);                    // merger.ln_q (RMS norm)
      vModel.vMm0W = vGetTensor(fStringFormat(TN_LLAVA_PROJ, 0, "weight")); // merger.mlp.0
      vModel.vMm0B = vGetTensor(fStringFormat(TN_LLAVA_PROJ, 0, "bias"));
      vModel.vMm1W = vGetTensor(fStringFormat(TN_LLAVA_PROJ, 2, "weight")); // merger.mlp.2
      vModel.vMm1B = vGetTensor(fStringFormat(TN_LLAVA_PROJ, 2, "bias"));
    } break;
    case cProjectorTypeYasa2: {
      // reuse tensors already loaded by the common section
      // (TN_PATCH_EMBD and TN_PATCH_BIAS have the same tensor names)
      GGML_ASSERT(vModel.vPatchEmbeddings0 && "yasa2 requires v.patch_embd.weight");
      vModel.vYasaPatchW = vModel.vPatchEmbeddings0;
      vModel.vYasaPatchB = vModel.vPatchBias;
      vModel.vYasaPatchLnW = vGetTensor(TN_YASA_PATCH_LN_W, false);
      vModel.vYasaPatchLnB = vGetTensor(TN_YASA_PATCH_LN_B, false);
      vModel.vYasaBackboneLnW = vGetTensor(TN_YASA_BACKBONE_LN_W, false);
      vModel.vYasaBackboneLnB = vGetTensor(TN_YASA_BACKBONE_LN_B, false);
      vModel.vYasaVisionPosEmbed = vGetTensor(TN_YASA_POS_EMBD, false);
      vModel.vMm0W = vGetTensor(fStringFormat(TN_LLAVA_PROJ, 0, "weight"));
      vModel.vMm0B = vGetTensor(fStringFormat(TN_LLAVA_PROJ, 0, "bias"), false);
      vModel.vMm2W = vGetTensor(fStringFormat(TN_LLAVA_PROJ, 2, "weight"));
      vModel.vMm2B = vGetTensor(fStringFormat(TN_LLAVA_PROJ, 2, "bias"), false);

      vModel.lYasaStages.clear();
      for (int vS = 0;; ++vS) {
        yasa2_stage vStage;
        vStage.vDownLnW = vGetTensor(fStringFormat(TN_YASA_STAGE_DOWN_LN, vS, "weight"), false);
        vStage.vDownLnB = vGetTensor(fStringFormat(TN_YASA_STAGE_DOWN_LN, vS, "bias"), false);
        vStage.vDownConvW = vGetTensor(fStringFormat(TN_YASA_STAGE_DOWN_CONV, vS, "weight"), false);
        vStage.vDownConvB = vGetTensor(fStringFormat(TN_YASA_STAGE_DOWN_CONV, vS, "bias"), false);

        for (int vBi = 0;; ++vBi) {
          yasa2_block vBlk;
          vBlk.vDwW = vGetTensor(fStringFormat(TN_YASA_STAGE_BLK, vS, vBi, "dw", "weight"), false);
          if (!vBlk.vDwW) {
            break;
          }
          vBlk.vDwB = vGetTensor(fStringFormat(TN_YASA_STAGE_BLK, vS, vBi, "dw", "bias"), false);
          vBlk.vLnW = vGetTensor(fStringFormat(TN_YASA_STAGE_BLK, vS, vBi, "ln", "weight"), false);
          vBlk.vLnB = vGetTensor(fStringFormat(TN_YASA_STAGE_BLK, vS, vBi, "ln", "bias"), false);
          vBlk.vPw1W = vGetTensor(fStringFormat(TN_YASA_STAGE_BLK, vS, vBi, "pw1", "weight"), false);
          vBlk.vPw1B = vGetTensor(fStringFormat(TN_YASA_STAGE_BLK, vS, vBi, "pw1", "bias"), false);
          vBlk.vGrnW = vGetTensor(fStringFormat(TN_YASA_STAGE_BLK, vS, vBi, "grn", "weight"), false);
          vBlk.vGrnB = vGetTensor(fStringFormat(TN_YASA_STAGE_BLK, vS, vBi, "grn", "bias"), false);
          vBlk.vPw2W = vGetTensor(fStringFormat(TN_YASA_STAGE_BLK, vS, vBi, "pw2", "weight"), false);
          vBlk.vPw2B = vGetTensor(fStringFormat(TN_YASA_STAGE_BLK, vS, vBi, "pw2", "bias"), false);
          vStage.lBlocks.push_back(vBlk);
        }

        if (!vStage.vDownConvW && vStage.lBlocks.empty()) {
          break;
        }
        vModel.lYasaStages.push_back(std::move(vStage));
      }
    } break;
    case cProjectorTypeGlm4v:
    case cProjectorTypeGlm5v: {
      vModel.vMmFcW = vGetTensor(fStringFormat(TN_MM_PROJECTOR, "weight"));
      vModel.vMmFfnUpW = vGetTensor(fStringFormat(TN_MM_UP, "weight"));
      vModel.vMmFfnUpB = vGetTensor(fStringFormat(TN_MM_UP, "bias"), false);
      vModel.vMmFfnGateW = vGetTensor(fStringFormat(TN_MM_GATE, "weight"));
      vModel.vMmFfnGateB = vGetTensor(fStringFormat(TN_MM_GATE, "bias"), false);
      vModel.vMmFfnDownW = vGetTensor(fStringFormat(TN_MM_DOWN, "weight"));
      vModel.vMmFfnDownB = vGetTensor(fStringFormat(TN_MM_DOWN, "bias"), false);
      vModel.vMmPostNormW = vGetTensor(fStringFormat(TN_MM_POST_NORM, "weight"));
      vModel.vMmPostNormB = vGetTensor(fStringFormat(TN_MM_POST_NORM, "bias"), false);
      vModel.vMmPatchMergerW = vGetTensor(fStringFormat(TN_MM_PATCH_MERGER, "weight"));
      vModel.vMmPatchMergerB = vGetTensor(fStringFormat(TN_MM_PATCH_MERGER, "bias"));
    } break;
    case cProjectorTypeGemma3: {
      vModel.vMmInputProjW = vGetTensor(TN_MM_INP_PROJ);
      vModel.vMmSoftEmbNormW = vGetTensor(TN_MM_SOFT_EMB_N);
    } break;
    case cProjectorTypeGemma3nv: {
      vModel.vMobilenetStemConvW = vGetTensor(TN_MNV5_STEM_CONV, false);
      vModel.vMobilenetStemConvB = vGetTensor(TN_MNV5_STEM_BIAS, false);
      vModel.vMobilenetStemNormW = vGetTensor(TN_MNV5_STEM_BN, false);

      vModel.vMsfaFfnExpandW = vGetTensor(TN_MNV5_MSFA_FFN_EXP_W, false);
      vModel.vMsfaFfnExpandBn = vGetTensor(TN_MNV5_MSFA_FFN_EXP_BN, false); // Consume BN if present but likely folded
      vModel.vMsfaFfnProjectW = vGetTensor(TN_MNV5_MSFA_FFN_PROJ_W, false);
      vModel.vMsfaFfnProjectBn = vGetTensor(TN_MNV5_MSFA_FFN_PROJ_BN, false);

      vModel.vMsfaConcatNormW = vGetTensor(TN_MNV5_MSFA_NORM, false);

      // Dynamically load blocks stage by stage
      for (int vStage = 0; vStage < 4; ++vStage) {
        int vBlocksFoundInStage = 0;

        for (int vBlkIdx = 0;; ++vBlkIdx) {
          bool vFoundBlock = false;
          mobilenetv5_block vBlock;

          // 1. Check for Edge Residual (S0)
          vBlock.vS0ConvExpW = vGetTensor(fStringFormat(TN_MNV5_BLK_S0_EXP_W, vStage, vBlkIdx), false);
          if (vBlock.vS0ConvExpW) {
            vFoundBlock = true;
            vBlock.vS0Bn1W = vGetTensor(fStringFormat(TN_MNV5_BLK_S0_BN1_W, vStage, vBlkIdx), false);
            vBlock.vS0ConvPwlW = vGetTensor(fStringFormat(TN_MNV5_BLK_S0_PWL_W, vStage, vBlkIdx), false);
            vBlock.vS0Bn2W = vGetTensor(fStringFormat(TN_MNV5_BLK_S0_BN2_W, vStage, vBlkIdx), false);
          }
          // 2. Check for UIR (Universal Inverted Residual)
          else {
            // Check for dw_start OR pw_exp (some UIR blocks skip dw_start)
            vBlock.vDwStartW = vGetTensor(fStringFormat(TN_MNV5_BLK_DW_START_W, vStage, vBlkIdx), false);
            vBlock.vPwExpW = vGetTensor(fStringFormat(TN_MNV5_BLK_PW_EXP_W, vStage, vBlkIdx), false);

            if (vBlock.vDwStartW || vBlock.vPwExpW) {
              vFoundBlock = true;
              if (vBlock.vDwStartW) {
                vBlock.vDwStartBnW = vGetTensor(fStringFormat(TN_MNV5_BLK_DW_START_BN, vStage, vBlkIdx), false);
              }
              if (vBlock.vPwExpW) {
                vBlock.vPwExpBnW = vGetTensor(fStringFormat(TN_MNV5_BLK_PW_EXP_BN, vStage, vBlkIdx), false);
              }
              vBlock.vDwMidW = vGetTensor(fStringFormat(TN_MNV5_BLK_DW_MID_W, vStage, vBlkIdx), false);
              if (vBlock.vDwMidW) {
                vBlock.vDwMidBnW = vGetTensor(fStringFormat(TN_MNV5_BLK_DW_MID_BN, vStage, vBlkIdx), false);
              }
              vBlock.vPwProjW = vGetTensor(fStringFormat(TN_MNV5_BLK_PW_PROJ_W, vStage, vBlkIdx), false);
              if (vBlock.vPwProjW) {
                vBlock.vPwProjBnW = vGetTensor(fStringFormat(TN_MNV5_BLK_PW_PROJ_BN, vStage, vBlkIdx), false);
              }
              vBlock.vLayerScaleW = vGetTensor(fStringFormat(TN_MNV5_BLK_LAYER_SCALE, vStage, vBlkIdx), false);
            }
          }

          // 3. Check for Attention (MQA)
          // Even if UIR/Edge check failed, this might be a pure attention block
          ggml_tensor *vAttnQCheck = vGetTensor(fStringFormat(TN_MNV5_ATTN_Q_W, vStage, vBlkIdx), false);
          if (vAttnQCheck) {
            vFoundBlock = true;
            vBlock.vAttnQW = vAttnQCheck;
            vBlock.vAttnKW = vGetTensor(fStringFormat(TN_MNV5_ATTN_K_W, vStage, vBlkIdx), false);
            vBlock.vAttnVW = vGetTensor(fStringFormat(TN_MNV5_ATTN_V_W, vStage, vBlkIdx), false);
            vBlock.vAttnOW = vGetTensor(fStringFormat(TN_MNV5_ATTN_O_W, vStage, vBlkIdx), false);
            vBlock.vAttnKDwW = vGetTensor(fStringFormat(TN_MNV5_ATTN_K_DW, vStage, vBlkIdx), false);
            vBlock.vAttnKNormW = vGetTensor(fStringFormat(TN_MNV5_ATTN_K_NORM, vStage, vBlkIdx), false);
            vBlock.vAttnVDwW = vGetTensor(fStringFormat(TN_MNV5_ATTN_V_DW, vStage, vBlkIdx), false);
            vBlock.vAttnVNormW = vGetTensor(fStringFormat(TN_MNV5_ATTN_V_NORM, vStage, vBlkIdx), false);
            vBlock.vAttnNormW = vGetTensor(fStringFormat(TN_MNV5_ATTN_NORM, vStage, vBlkIdx), false);
            // Note: Attention blocks also have layer_scale, load it if not already loaded by UIR check
            if (!vBlock.vLayerScaleW) {
              vBlock.vLayerScaleW = vGetTensor(fStringFormat(TN_MNV5_BLK_LAYER_SCALE, vStage, vBlkIdx), false);
            }
          }

          if (vFoundBlock) {
            vModel.lMobilenetBlocks.push_back(vBlock);
            vBlocksFoundInStage++;
          } else {
            // End of blocks for this stage
            break;
          }
        }

        // Track where this stage ends in the flat vector
        if (vBlocksFoundInStage > 0) {
          vModel.lMobilenetStageEnds.push_back(vModel.lMobilenetBlocks.size() - 1);
          LOG_INF("%s: Stage %d ended at global block index %zu\n", __func__, vStage,
                  vModel.lMobilenetBlocks.size() - 1);
        }
      }
      vModel.vMmInputProjW = vGetTensor(TN_MM_INP_PROJ);
      vModel.vMmSoftEmbNormW = vGetTensor(TN_MM_SOFT_EMB_N);
    } break;
    case cProjectorTypeIdefics3: {
      vModel.vMmFcW = vGetTensor(fStringFormat(TN_MM_PROJECTOR, "weight"));
    } break;
    case cProjectorTypeLfm2: {
      vModel.vMmInputNormW = vGetTensor(TN_MM_INP_NORM, false);
      vModel.vMmInputNormB = vGetTensor(TN_MM_INP_NORM_B, false);
      vModel.vMm1W = vGetTensor(fStringFormat(TN_LLAVA_PROJ, 1, "weight"));
      vModel.vMm1B = vGetTensor(fStringFormat(TN_LLAVA_PROJ, 1, "bias"));
      vModel.vMm2W = vGetTensor(fStringFormat(TN_LLAVA_PROJ, 2, "weight"));
      vModel.vMm2B = vGetTensor(fStringFormat(TN_LLAVA_PROJ, 2, "bias"));
    } break;
    case cProjectorTypeKimivl:
    case cProjectorTypePaddleocr:
    case cProjectorTypeKimik25: {
      vModel.vMmInputNormW = vGetTensor(TN_MM_INP_NORM);
      vModel.vMmInputNormB = vGetTensor(TN_MM_INP_NORM_B);
      vModel.vMm1W = vGetTensor(fStringFormat(TN_LLAVA_PROJ, 1, "weight"));
      vModel.vMm1B = vGetTensor(fStringFormat(TN_LLAVA_PROJ, 1, "bias"));
      vModel.vMm2W = vGetTensor(fStringFormat(TN_LLAVA_PROJ, 2, "weight"));
      vModel.vMm2B = vGetTensor(fStringFormat(TN_LLAVA_PROJ, 2, "bias"));
    } break;
    case cProjectorTypePixtral: {
      vModel.vMm1W = vGetTensor(fStringFormat(TN_LLAVA_PROJ, 1, "weight"));
      vModel.vMm1B = vGetTensor(fStringFormat(TN_LLAVA_PROJ, 1, "bias"), false);
      vModel.vMm2W = vGetTensor(fStringFormat(TN_LLAVA_PROJ, 2, "weight"));
      vModel.vMm2B = vGetTensor(fStringFormat(TN_LLAVA_PROJ, 2, "bias"), false);
      // [IMG_BREAK] token embedding
      vModel.vTokenEmbdImgBreak = vGetTensor(TN_TOK_IMG_BREAK);
      // for mistral small 3.1
      vModel.vMmInputNormW = vGetTensor(TN_MM_INP_NORM, false);
      vModel.vMmPatchMergerW = vGetTensor(fStringFormat(TN_MM_PATCH_MERGER, "weight"), false);
    } break;
    case cProjectorTypeLightonocr: {
      vModel.vMm1W = vGetTensor(fStringFormat(TN_LLAVA_PROJ, 1, "weight"));
      vModel.vMm1B = vGetTensor(fStringFormat(TN_LLAVA_PROJ, 1, "bias"), false);
      vModel.vMm2W = vGetTensor(fStringFormat(TN_LLAVA_PROJ, 2, "weight"));
      vModel.vMm2B = vGetTensor(fStringFormat(TN_LLAVA_PROJ, 2, "bias"), false);
      vModel.vMmInputNormW = vGetTensor(TN_MM_INP_NORM, false);
      vModel.vMmPatchMergerW = vGetTensor(fStringFormat(TN_MM_PATCH_MERGER, "weight"), false);
    } break;
    case cProjectorTypeDotsOcr: {
      vModel.vMm0W = vGetTensor(fStringFormat(TN_LLAVA_PROJ, 0, "weight"));
      vModel.vMm0B = vGetTensor(fStringFormat(TN_LLAVA_PROJ, 0, "bias"));
      vModel.vMm2W = vGetTensor(fStringFormat(TN_LLAVA_PROJ, 2, "weight"));
      vModel.vMm2B = vGetTensor(fStringFormat(TN_LLAVA_PROJ, 2, "bias"));
      vModel.vMmInputNormW = vGetTensor(TN_MM_INP_NORM);
      vModel.vMmInputNormB = vGetTensor(TN_MM_INP_NORM_B);
      // post_trunk_norm: applied after all ViT blocks, before the merger
      vModel.vPostLnW = vGetTensor(fStringFormat(TN_MM_POST_NORM, "weight"));
    } break;
    case cProjectorTypeUltravox: {
      vModel.vConv1d1W = vGetTensor(fStringFormat(TN_CONV1D, 1, "weight"));
      vModel.vConv1d1B = vGetTensor(fStringFormat(TN_CONV1D, 1, "bias"));
      vModel.vConv1d2W = vGetTensor(fStringFormat(TN_CONV1D, 2, "weight"));
      vModel.vConv1d2B = vGetTensor(fStringFormat(TN_CONV1D, 2, "bias"));
      vModel.vMm1W = vGetTensor(fStringFormat(TN_MM_AUDIO_MLP, 1, "weight"));
      vModel.vMm2W = vGetTensor(fStringFormat(TN_MM_AUDIO_MLP, 2, "weight"));
      vModel.vMmNormPreW = vGetTensor(fStringFormat(TN_MM_NORM_PRE, "weight"));
      vModel.vMmNormMidW = vGetTensor(fStringFormat(TN_MM_NORM_MID, "weight"));
    } break;
    case cProjectorTypeMeralion: {
      // Whisper encoder conv layers
      vModel.vConv1d1W = vGetTensor(fStringFormat(TN_CONV1D, 1, "weight"));
      vModel.vConv1d1B = vGetTensor(fStringFormat(TN_CONV1D, 1, "bias"));
      vModel.vConv1d2W = vGetTensor(fStringFormat(TN_CONV1D, 2, "weight"));
      vModel.vConv1d2B = vGetTensor(fStringFormat(TN_CONV1D, 2, "bias"));
      // MERaLiON adaptor: 4 linear layers + ln_pre
      // linear_0 = frame compression (19200->6400) + SiLU
      // linear_1 = gate_proj (6400->6400) for GLU
      // linear_2 = pool_proj (6400->6400) for GLU
      // linear_3 = out_proj  (6400->3584)
      vModel.vMm0W = vGetTensor(fStringFormat(TN_MM_AUDIO_MLP, 0, "weight"));
      vModel.vMm0B = vGetTensor(fStringFormat(TN_MM_AUDIO_MLP, 0, "bias"));
      vModel.vMm1W = vGetTensor(fStringFormat(TN_MM_AUDIO_MLP, 1, "weight"));
      vModel.vMm1B = vGetTensor(fStringFormat(TN_MM_AUDIO_MLP, 1, "bias"));
      vModel.vMm2W = vGetTensor(fStringFormat(TN_MM_AUDIO_MLP, 2, "weight"));
      vModel.vMm2B = vGetTensor(fStringFormat(TN_MM_AUDIO_MLP, 2, "bias"));
      vModel.vMm3W = vGetTensor(fStringFormat(TN_MM_AUDIO_MLP, 3, "weight"));
      vModel.vMm3B = vGetTensor(fStringFormat(TN_MM_AUDIO_MLP, 3, "bias"));
      // ln_speech (LayerNorm before adaptor)
      vModel.vMmNormPreW = vGetTensor(fStringFormat(TN_MM_NORM_PRE, "weight"));
      vModel.vMmNormPreB = vGetTensor(fStringFormat(TN_MM_NORM_PRE, "bias"));
    } break;
    case cProjectorTypeQwen2a: {
      vModel.vConv1d1W = vGetTensor(fStringFormat(TN_CONV1D, 1, "weight"));
      vModel.vConv1d1B = vGetTensor(fStringFormat(TN_CONV1D, 1, "bias"));
      vModel.vConv1d2W = vGetTensor(fStringFormat(TN_CONV1D, 2, "weight"));
      vModel.vConv1d2B = vGetTensor(fStringFormat(TN_CONV1D, 2, "bias"));
      vModel.vMmFcW = vGetTensor(fStringFormat(TN_MM_AUDIO_FC, "weight"));
      vModel.vMmFcB = vGetTensor(fStringFormat(TN_MM_AUDIO_FC, "bias"));
    } break;
    case cProjectorTypeQwen3a: {
      vModel.vConv2d1W = vGetTensor(fStringFormat(TN_CONV2D, 1, "weight"));
      vModel.vConv2d1B = vGetTensor(fStringFormat(TN_CONV2D, 1, "bias"));
      vModel.vConv2d2W = vGetTensor(fStringFormat(TN_CONV2D, 2, "weight"));
      vModel.vConv2d2B = vGetTensor(fStringFormat(TN_CONV2D, 2, "bias"));
      vModel.vConv2d3W = vGetTensor(fStringFormat(TN_CONV2D, 3, "weight"));
      vModel.vConv2d3B = vGetTensor(fStringFormat(TN_CONV2D, 3, "bias"));
      vModel.vConvOutW = vGetTensor(fStringFormat(TN_CONV_OUT, "weight")); // no bias
      vModel.vMm1W = vGetTensor(fStringFormat(TN_MM_AUDIO_MLP, 1, "weight"));
      vModel.vMm1B = vGetTensor(fStringFormat(TN_MM_AUDIO_MLP, 1, "bias"));
      vModel.vMm2W = vGetTensor(fStringFormat(TN_MM_AUDIO_MLP, 2, "weight"));
      vModel.vMm2B = vGetTensor(fStringFormat(TN_MM_AUDIO_MLP, 2, "bias"));
    } break;
    case cProjectorTypeMimoAudio: {
      vModel.vConv1d1W = vGetTensor(fStringFormat(TN_CONV1D, 1, "weight"));
      vModel.vConv1d1B = vGetTensor(fStringFormat(TN_CONV1D, 1, "bias"));
      vModel.vConv1d2W = vGetTensor(fStringFormat(TN_CONV1D, 2, "weight"));
      vModel.vConv1d2B = vGetTensor(fStringFormat(TN_CONV1D, 2, "bias"));
      vModel.vDownsampleConvW = vGetTensor(fStringFormat(TN_A_DOWNSAMPLE_CONV, "weight"));
      vModel.vDownsampleNormW = vGetTensor(fStringFormat(TN_A_DOWNSAMPLE_NORM, "weight"));
      vModel.vDownsampleNormB = vGetTensor(fStringFormat(TN_A_DOWNSAMPLE_NORM, "bias"));
      vModel.vRvqCodebook = vGetTensor(fStringFormat(TN_A_RVQ_CODEBOOK, "weight"), false);
      vModel.vMmACodeEmbd = vGetTensor(fStringFormat(TN_MM_A_CODE_EMBD, "weight"), false);
      if (!vModel.vRvqCodebook || !vModel.vMmACodeEmbd) {
        throw std::runtime_error(
            fStringFormat("%s: mimo_audio: missing %s or %s\n", __func__, TN_A_RVQ_CODEBOOK, TN_MM_A_CODE_EMBD));
      }
      // hparams.lRvqCodebookSize comes from GGUF metadata and is independent of the
      // tensors' actual shapes - bound it so codebook/code_embd views built from it
      // (mimo-audio.cpp) can never read past either tensor's allocated bins/vocab.
      for (int32_t vBins : vHparams.lRvqCodebookSize) {
        if (vBins <= 0 || vBins > vModel.vRvqCodebook->ne[1] || vBins > vModel.vMmACodeEmbd->ne[1]) {
          throw std::runtime_error(
              fStringFormat("%s: mimo_audio: %s entry (%d) out of range for codebook/code_embd tensors\n", __func__,
                            KEY_A_RVQ_CODEBOOK_SIZE, vBins));
        }
      }

      // LLM-side connector: input_local_transformer + projection
      vModel.lMmALocalLayers.resize(vHparams.vAudioLocalNLayer);
      for (int vIl = 0; vIl < vHparams.vAudioLocalNLayer; vIl++) {
        auto &vLayer = vModel.lMmALocalLayers[vIl];
        vLayer.vQW = vGetTensor(fStringFormat(TN_MM_A_LOCAL_ATTN_Q, vIl, "weight"));
        vLayer.vQB = vGetTensor(fStringFormat(TN_MM_A_LOCAL_ATTN_Q, vIl, "bias"));
        vLayer.vKW = vGetTensor(fStringFormat(TN_MM_A_LOCAL_ATTN_K, vIl, "weight"));
        vLayer.vKB = vGetTensor(fStringFormat(TN_MM_A_LOCAL_ATTN_K, vIl, "bias"));
        vLayer.vVW = vGetTensor(fStringFormat(TN_MM_A_LOCAL_ATTN_V, vIl, "weight"));
        vLayer.vVB = vGetTensor(fStringFormat(TN_MM_A_LOCAL_ATTN_V, vIl, "bias"));
        vLayer.vOW = vGetTensor(fStringFormat(TN_MM_A_LOCAL_ATTN_OUT, vIl, "weight"));
        vLayer.vFfGateW = vGetTensor(fStringFormat(TN_MM_A_LOCAL_FFN_GATE, vIl, "weight"));
        vLayer.vFfUpW = vGetTensor(fStringFormat(TN_MM_A_LOCAL_FFN_UP, vIl, "weight"));
        vLayer.vFfDownW = vGetTensor(fStringFormat(TN_MM_A_LOCAL_FFN_DOWN, vIl, "weight"));
        vLayer.vLn1W = vGetTensor(fStringFormat(TN_MM_A_LOCAL_LN1, vIl, "weight"));
        vLayer.vLn2W = vGetTensor(fStringFormat(TN_MM_A_LOCAL_LN2, vIl, "weight"));
      }
      vModel.vMmALocalNormW = vGetTensor(fStringFormat(TN_MM_A_LOCAL_NORM, "weight"));

      vModel.vMm1W = vGetTensor(fStringFormat(TN_MM_AUDIO_MLP, 1, "weight"));
      vModel.vMm2W = vGetTensor(fStringFormat(TN_MM_AUDIO_MLP, 2, "weight"));
    } break;
    case cProjectorTypeVoxtral: {
      vModel.vConv1d1W = vGetTensor(fStringFormat(TN_CONV1D, 1, "weight"));
      vModel.vConv1d1B = vGetTensor(fStringFormat(TN_CONV1D, 1, "bias"));
      vModel.vConv1d2W = vGetTensor(fStringFormat(TN_CONV1D, 2, "weight"));
      vModel.vConv1d2B = vGetTensor(fStringFormat(TN_CONV1D, 2, "bias"));
      vModel.vMm1W = vGetTensor(fStringFormat(TN_MM_AUDIO_MLP, 1, "weight"));
      vModel.vMm2W = vGetTensor(fStringFormat(TN_MM_AUDIO_MLP, 2, "weight"));
    } break;
    case cProjectorTypeMusicFlamingo: {
      vModel.vConv1d1W = vGetTensor(fStringFormat(TN_CONV1D, 1, "weight"));
      vModel.vConv1d1B = vGetTensor(fStringFormat(TN_CONV1D, 1, "bias"));
      vModel.vConv1d2W = vGetTensor(fStringFormat(TN_CONV1D, 2, "weight"));
      vModel.vConv1d2B = vGetTensor(fStringFormat(TN_CONV1D, 2, "bias"));
      vModel.vMm1W = vGetTensor(fStringFormat(TN_MM_AUDIO_MLP, 1, "weight"));
      vModel.vMm1B = vGetTensor(fStringFormat(TN_MM_AUDIO_MLP, 1, "bias"));
      vModel.vMm2W = vGetTensor(fStringFormat(TN_MM_AUDIO_MLP, 2, "weight"));
      vModel.vMm2B = vGetTensor(fStringFormat(TN_MM_AUDIO_MLP, 2, "bias"));
    } break;
    case cProjectorTypeInternvl: {
      vModel.vMm0W = vGetTensor(fStringFormat(TN_MVLM_PROJ_MLP, 0, "weight"));
      vModel.vMm0B = vGetTensor(fStringFormat(TN_MVLM_PROJ_MLP, 0, "bias"));
      vModel.vMm1W = vGetTensor(fStringFormat(TN_MVLM_PROJ_MLP, 1, "weight"));
      vModel.vMm1B = vGetTensor(fStringFormat(TN_MVLM_PROJ_MLP, 1, "bias"));
      vModel.vMm3W = vGetTensor(fStringFormat(TN_MVLM_PROJ_MLP, 3, "weight"));
      vModel.vMm3B = vGetTensor(fStringFormat(TN_MVLM_PROJ_MLP, 3, "bias"));
    } break;
    case cProjectorTypeNemotronV2Vl: {
      vModel.vMm0W = vGetTensor(fStringFormat(TN_MVLM_PROJ_MLP, 0, "weight"));
      vModel.vMm1W = vGetTensor(fStringFormat(TN_MVLM_PROJ_MLP, 1, "weight"));
      vModel.vMm3W = vGetTensor(fStringFormat(TN_MVLM_PROJ_MLP, 3, "weight"));
    } break;
    case cProjectorTypeGlma: {
      vModel.vConv1d1W = vGetTensor(fStringFormat(TN_CONV1D, 1, "weight"));
      vModel.vConv1d1B = vGetTensor(fStringFormat(TN_CONV1D, 1, "bias"));
      vModel.vConv1d2W = vGetTensor(fStringFormat(TN_CONV1D, 2, "weight"));
      vModel.vConv1d2B = vGetTensor(fStringFormat(TN_CONV1D, 2, "bias"));
      vModel.vMm1W = vGetTensor(fStringFormat(TN_MM_AUDIO_MLP, 1, "weight"));
      vModel.vMm1B = vGetTensor(fStringFormat(TN_MM_AUDIO_MLP, 1, "bias"));
      vModel.vMm2W = vGetTensor(fStringFormat(TN_MM_AUDIO_MLP, 2, "weight"));
      vModel.vMm2B = vGetTensor(fStringFormat(TN_MM_AUDIO_MLP, 2, "bias"));
      vModel.vMmNormPreW = vGetTensor(fStringFormat(TN_MM_NORM_PRE, "weight"));
      vModel.vMmNormPreB = vGetTensor(fStringFormat(TN_MM_NORM_PRE, "bias"));
      vModel.vMmBoi = vGetTensor(fStringFormat(TN_TOK_BOI));
      vModel.vMmEoi = vGetTensor(fStringFormat(TN_TOK_EOI));
    } break;
    case cProjectorTypeCogvlm: {
      vModel.vMmModelProj = vGetTensor(fStringFormat(TN_MM_PROJECTOR, "weight"));
      vModel.vMmPostFcNormW = vGetTensor(fStringFormat(TN_MM_POST_FC_NORM, "weight"));
      vModel.vMmPostFcNormB = vGetTensor(fStringFormat(TN_MM_POST_FC_NORM, "bias"));
      vModel.vMmHTo4hW = vGetTensor(fStringFormat(TN_MM_H_TO_4H, "weight"));
      vModel.vMmGateW = vGetTensor(fStringFormat(TN_MM_GATE, "weight"));
      vModel.vMm4hToHW = vGetTensor(fStringFormat(TN_MM_4H_TO_H, "weight"));
      vModel.vMmBoi = vGetTensor(TN_TOK_BOI);
      vModel.vMmEoi = vGetTensor(TN_TOK_EOI);
    } break;
    case cProjectorTypeHunyuanvl: {
      // proj.0 -> mm.0 (conv1), proj.2 -> mm.2 (conv2), mlp -> mm.model.fc (linear)
      vModel.vMm0W = vGetTensor(fStringFormat(TN_LLAVA_PROJ, 0, "weight"));
      vModel.vMm0B = vGetTensor(fStringFormat(TN_LLAVA_PROJ, 0, "bias"));
      vModel.vMm1W = vGetTensor(fStringFormat(TN_LLAVA_PROJ, 2, "weight"));
      vModel.vMm1B = vGetTensor(fStringFormat(TN_LLAVA_PROJ, 2, "bias"));
      vModel.vMmModelProj = vGetTensor(fStringFormat(TN_MM_PROJECTOR, "weight"));
      vModel.vMmModelProjB = vGetTensor(fStringFormat(TN_MM_PROJECTOR, "bias"));
      vModel.vMmPreNormW = vGetTensor(fStringFormat(TN_MM_PRE_NORM, "weight"));
      vModel.vMmPostNormW = vGetTensor(fStringFormat(TN_MM_POST_NORM, "weight"));
      vModel.vMmImgBegin = vGetTensor(TN_TOK_IMG_BEGIN);
      vModel.vMmImgEnd = vGetTensor(TN_TOK_IMG_END);
      vModel.vImageNewline = vGetTensor(TN_IMAGE_NEWLINE);
      vModel.vViewSeperator = vGetTensor(TN_IMAGE_SEPERATOR, false);
    } break;
    case cProjectorTypeJanusPro: {
      vModel.vMm0W = vGetTensor(fStringFormat(TN_LLAVA_PROJ, 0, "weight"));
      vModel.vMm0B = vGetTensor(fStringFormat(TN_LLAVA_PROJ, 0, "bias"));
      vModel.vMm1W = vGetTensor(fStringFormat(TN_LLAVA_PROJ, 1, "weight"));
      vModel.vMm1B = vGetTensor(fStringFormat(TN_LLAVA_PROJ, 1, "bias"));
    } break;
    case cProjectorTypePhi4: {
      vModel.vMm0W = vGetTensor(fStringFormat(TN_LLAVA_PROJ, 0, "weight"));
      vModel.vMm0B = vGetTensor(fStringFormat(TN_LLAVA_PROJ, 0, "bias"));
      vModel.vMm2W = vGetTensor(fStringFormat(TN_LLAVA_PROJ, 2, "weight"));
      vModel.vMm2B = vGetTensor(fStringFormat(TN_LLAVA_PROJ, 2, "bias"));
    } break;
    case cProjectorTypeLfm2a: {
      for (int vI : {0, 2, 3, 5, 6}) {
        vModel.aPreEncodeConvXW[vI] = vGetTensor(fStringFormat(TN_CONV1D, vI, "weight"));
        vModel.aPreEncodeConvXB[vI] = vGetTensor(fStringFormat(TN_CONV1D, vI, "bias"));
      }
      vModel.vPreEncodeOutW = vGetTensor(fStringFormat(TN_PRE_ENCODE_OUT, "weight"));
      vModel.vPreEncodeOutB = vGetTensor(fStringFormat(TN_PRE_ENCODE_OUT, "bias"));

      vModel.vMm0W = vGetTensor(fStringFormat(TN_MM_AUDIO_MLP, 0, "weight"));
      vModel.vMm0B = vGetTensor(fStringFormat(TN_MM_AUDIO_MLP, 0, "bias"));
      vModel.vMm1W = vGetTensor(fStringFormat(TN_MM_AUDIO_MLP, 1, "weight"));
      vModel.vMm1B = vGetTensor(fStringFormat(TN_MM_AUDIO_MLP, 1, "bias"));
      vModel.vMm3W = vGetTensor(fStringFormat(TN_MM_AUDIO_MLP, 3, "weight"));
      vModel.vMm3B = vGetTensor(fStringFormat(TN_MM_AUDIO_MLP, 3, "bias"));

      for (int vIl = 0; vIl < vHparams.vNLayer; ++vIl) {
        auto &vLayer = vModel.lLayers[vIl];

        vLayer.vFfNormW = vGetTensor(fStringFormat(TN_FFN_NORM, vPrefix, vIl, "weight"));
        vLayer.vFfNormB = vGetTensor(fStringFormat(TN_FFN_NORM, vPrefix, vIl, "bias"));
        vLayer.vFfNorm1W = vGetTensor(fStringFormat(TN_FFN_NORM_1, vPrefix, vIl, "weight"));
        vLayer.vFfNorm1B = vGetTensor(fStringFormat(TN_FFN_NORM_1, vPrefix, vIl, "bias"));
        vLayer.vFfUp1W = vGetTensor(fStringFormat(TN_FFN_UP_1, vPrefix, vIl, "weight"));
        vLayer.vFfUp1B = vGetTensor(fStringFormat(TN_FFN_UP_1, vPrefix, vIl, "bias"));
        vLayer.vFfDown1W = vGetTensor(fStringFormat(TN_FFN_DOWN_1, vPrefix, vIl, "weight"));
        vLayer.vFfDown1B = vGetTensor(fStringFormat(TN_FFN_DOWN_1, vPrefix, vIl, "bias"));

        vLayer.vPosBiasU = vGetTensor(fStringFormat(TN_POS_BIAS_U, vPrefix, vIl));
        vLayer.vPosBiasV = vGetTensor(fStringFormat(TN_POS_BIAS_V, vPrefix, vIl));

        vLayer.vNormConvW = vGetTensor(fStringFormat(TN_NORM_CONV, vPrefix, vIl, "weight"));
        vLayer.vNormConvB = vGetTensor(fStringFormat(TN_NORM_CONV, vPrefix, vIl, "bias"));

        vLayer.vLinearPosW = vGetTensor(fStringFormat(TN_LINEAR_POS, vPrefix, vIl, "weight"));

        vLayer.vConvNormW = vGetTensor(fStringFormat(TN_CONV_NORM, vPrefix, vIl, "weight"));
        vLayer.vConvNormB = vGetTensor(fStringFormat(TN_CONV_NORM, vPrefix, vIl, "bias"));
        vLayer.vConvDwW = vGetTensor(fStringFormat(TN_CONV_DW, vPrefix, vIl, "weight"));
        vLayer.vConvDwB = vGetTensor(fStringFormat(TN_CONV_DW, vPrefix, vIl, "bias"));
        vLayer.vConvPw1W = vGetTensor(fStringFormat(TN_CONV_PW1, vPrefix, vIl, "weight"));
        vLayer.vConvPw1B = vGetTensor(fStringFormat(TN_CONV_PW1, vPrefix, vIl, "bias"));
        vLayer.vConvPw2W = vGetTensor(fStringFormat(TN_CONV_PW2, vPrefix, vIl, "weight"));
        vLayer.vConvPw2B = vGetTensor(fStringFormat(TN_CONV_PW2, vPrefix, vIl, "bias"));
      }
    } break;
    case cProjectorTypeParakeet: {

      vHparams.lMelFilters = vGetVector(TN_MEL_FILTERS);
      vHparams.lWindow = vGetVector(TN_WINDOW);

      // Subsampling layers (conv1d)
      for (int vI : {0, 2, 3, 5, 6}) {
        vModel.aPreEncodeConvXW[vI] = vGetTensor(fStringFormat(TN_CONV1D, vI, "weight"));
        vModel.aPreEncodeConvXB[vI] = vGetTensor(fStringFormat(TN_CONV1D, vI, "bias"));
      }
      vModel.vPreEncodeOutW = vGetTensor(fStringFormat(TN_PRE_ENCODE_OUT, "weight"));
      vModel.vPreEncodeOutB = vGetTensor(fStringFormat(TN_PRE_ENCODE_OUT, "bias"));

      // Projection layers
      vModel.vMmNormPreW = vGetTensor(fStringFormat(TN_MM_NORM_PRE, "weight"), false);
      vModel.vMm0W = vGetTensor(fStringFormat(TN_MM_AUDIO_MLP, 1, "weight"), false);
      vModel.vMm1W = vGetTensor(fStringFormat(TN_MM_AUDIO_MLP, 2, "weight"), false);

      // Encoder layers
      for (int vIl = 0; vIl < vHparams.vNLayer; ++vIl) {
        auto &vLayer = vModel.lLayers[vIl];

        // Attention (from shared above)

        // Relative position encoding
        vLayer.vLinearPosW = vGetTensor(fStringFormat(TN_LINEAR_POS, vPrefix, vIl, "weight"));
        vLayer.vPosBiasU = vGetTensor(fStringFormat(TN_POS_BIAS_U, vPrefix, vIl));
        vLayer.vPosBiasV = vGetTensor(fStringFormat(TN_POS_BIAS_V, vPrefix, vIl));

        // Convolution module
        vLayer.vConvPw1W = vGetTensor(fStringFormat(TN_CONV_PW1, vPrefix, vIl, "weight"));
        vLayer.vConvPw1B = vGetTensor(fStringFormat(TN_CONV_PW1, vPrefix, vIl, "bias"), false);
        vLayer.vConvDwW = vGetTensor(fStringFormat(TN_CONV_DW, vPrefix, vIl, "weight"));
        vLayer.vConvDwB = vGetTensor(fStringFormat(TN_CONV_DW, vPrefix, vIl, "bias"), false);
        vLayer.vConvNormW = vGetTensor(fStringFormat(TN_CONV_NORM, vPrefix, vIl, "weight"));
        vLayer.vConvNormB = vGetTensor(fStringFormat(TN_CONV_NORM, vPrefix, vIl, "bias"));
        vLayer.vConvNormMean = vGetTensor(fStringFormat(TN_CONV_NORM_MEAN, vPrefix, vIl));
        vLayer.vConvNormVar = vGetTensor(fStringFormat(TN_CONV_NORM_VAR, vPrefix, vIl));
        vLayer.vConvPw2W = vGetTensor(fStringFormat(TN_CONV_PW2, vPrefix, vIl, "weight"));
        vLayer.vConvPw2B = vGetTensor(fStringFormat(TN_CONV_PW2, vPrefix, vIl, "bias"), false);

        // Feed-forward networks
        vLayer.vFfNormW = vGetTensor(fStringFormat(TN_FFN_NORM, vPrefix, vIl, "weight"));
        vLayer.vFfNormB = vGetTensor(fStringFormat(TN_FFN_NORM, vPrefix, vIl, "bias"));

        vLayer.vFfNorm1W = vGetTensor(fStringFormat(TN_FFN_NORM_1, vPrefix, vIl, "weight"));
        vLayer.vFfNorm1B = vGetTensor(fStringFormat(TN_FFN_NORM_1, vPrefix, vIl, "bias"));
        vLayer.vFfUp1W = vGetTensor(fStringFormat(TN_FFN_UP_1, vPrefix, vIl, "weight"));
        vLayer.vFfUp1B = vGetTensor(fStringFormat(TN_FFN_UP_1, vPrefix, vIl, "bias"), false);
        vLayer.vFfDown1W = vGetTensor(fStringFormat(TN_FFN_DOWN_1, vPrefix, vIl, "weight"));
        vLayer.vFfDown1B = vGetTensor(fStringFormat(TN_FFN_DOWN_1, vPrefix, vIl, "bias"), false);

        // Layer norms
        vLayer.vNormConvW = vGetTensor(fStringFormat(TN_NORM_CONV, vPrefix, vIl, "weight"));
        vLayer.vNormConvB = vGetTensor(fStringFormat(TN_NORM_CONV, vPrefix, vIl, "bias"));
      }
    } break;
    case cProjectorTypeGraniteSpeech: {
      vModel.vInpProjW = vGetTensor(fStringFormat(TN_INP_PROJ, "weight"));
      vModel.vInpProjB = vGetTensor(fStringFormat(TN_INP_PROJ, "bias"));
      vModel.vCtcOutW = vGetTensor(fStringFormat(TN_CTC_OUT, "weight"));
      vModel.vCtcOutB = vGetTensor(fStringFormat(TN_CTC_OUT, "bias"));
      vModel.vCtcOutMidW = vGetTensor(fStringFormat(TN_CTC_OUT_MID, "weight"));
      vModel.vCtcOutMidB = vGetTensor(fStringFormat(TN_CTC_OUT_MID, "bias"));

      // per-layer tensors not loaded by the generic loop above
      for (int vIl = 0; vIl < vHparams.vNLayer; ++vIl) {
        auto &vLayer = vModel.lLayers[vIl];

        vLayer.vAttnRelPosEmb = vGetTensor(fStringFormat(TN_ATTN_REL_POS_EMB, vPrefix, vIl));

        vLayer.vFfNormW = vGetTensor(fStringFormat(TN_FFN_NORM, vPrefix, vIl, "weight"));
        vLayer.vFfNormB = vGetTensor(fStringFormat(TN_FFN_NORM, vPrefix, vIl, "bias"));

        vLayer.vFfNorm1W = vGetTensor(fStringFormat(TN_FFN_NORM_1, vPrefix, vIl, "weight"));
        vLayer.vFfNorm1B = vGetTensor(fStringFormat(TN_FFN_NORM_1, vPrefix, vIl, "bias"));
        vLayer.vFfUp1W = vGetTensor(fStringFormat(TN_FFN_UP_1, vPrefix, vIl, "weight"));
        vLayer.vFfUp1B = vGetTensor(fStringFormat(TN_FFN_UP_1, vPrefix, vIl, "bias"));
        vLayer.vFfDown1W = vGetTensor(fStringFormat(TN_FFN_DOWN_1, vPrefix, vIl, "weight"));
        vLayer.vFfDown1B = vGetTensor(fStringFormat(TN_FFN_DOWN_1, vPrefix, vIl, "bias"));

        vLayer.vNormConvW = vGetTensor(fStringFormat(TN_NORM_CONV, vPrefix, vIl, "weight"));
        vLayer.vNormConvB = vGetTensor(fStringFormat(TN_NORM_CONV, vPrefix, vIl, "bias"));
        vLayer.vConvNormW = vGetTensor(fStringFormat(TN_CONV_NORM, vPrefix, vIl, "weight"));
        vLayer.vConvNormB = vGetTensor(fStringFormat(TN_CONV_NORM, vPrefix, vIl, "bias"));
        vLayer.vConvDwW = vGetTensor(fStringFormat(TN_CONV_DW, vPrefix, vIl, "weight"));
        vLayer.vConvPw1W = vGetTensor(fStringFormat(TN_CONV_PW1, vPrefix, vIl, "weight"));
        vLayer.vConvPw1B = vGetTensor(fStringFormat(TN_CONV_PW1, vPrefix, vIl, "bias"));
        vLayer.vConvPw2W = vGetTensor(fStringFormat(TN_CONV_PW2, vPrefix, vIl, "weight"));
        vLayer.vConvPw2B = vGetTensor(fStringFormat(TN_CONV_PW2, vPrefix, vIl, "bias"));
      }

      vModel.lQfProjBlocks.resize(1);
      auto &vQf = vModel.lQfProjBlocks[0];
      vQf.vQfProjQuery = vGetTensor(fStringFormat(TN_QF_PROJ_QUERY, vPrefix));
      vQf.vQfProjNormW = vGetTensor(fStringFormat(TN_QF_PROJ_NORM, vPrefix, "weight"));
      vQf.vQfProjNormB = vGetTensor(fStringFormat(TN_QF_PROJ_NORM, vPrefix, "bias"));
      vQf.vQfProjLinearW = vGetTensor(fStringFormat(TN_QF_PROJ_LINEAR, vPrefix, "weight"));
      vQf.vQfProjLinearB = vGetTensor(fStringFormat(TN_QF_PROJ_LINEAR, vPrefix, "bias"));

      const int cNProjLayers = 2;
      vQf.lQfProjLayers.resize(cNProjLayers);
      for (int vIl = 0; vIl < cNProjLayers; ++vIl) {
        auto &vPl = vQf.lQfProjLayers[vIl];

        vPl.vQW = vGetTensor(fStringFormat(TN_QF_SELF_ATTN_Q, vPrefix, vIl, "weight"));
        vPl.vQB = vGetTensor(fStringFormat(TN_QF_SELF_ATTN_Q, vPrefix, vIl, "bias"));
        vPl.vKW = vGetTensor(fStringFormat(TN_QF_SELF_ATTN_K, vPrefix, vIl, "weight"));
        vPl.vKB = vGetTensor(fStringFormat(TN_QF_SELF_ATTN_K, vPrefix, vIl, "bias"));
        vPl.vVW = vGetTensor(fStringFormat(TN_QF_SELF_ATTN_V, vPrefix, vIl, "weight"));
        vPl.vVB = vGetTensor(fStringFormat(TN_QF_SELF_ATTN_V, vPrefix, vIl, "bias"));
        vPl.vOW = vGetTensor(fStringFormat(TN_QF_SELF_ATTN_O, vPrefix, vIl, "weight"));
        vPl.vOB = vGetTensor(fStringFormat(TN_QF_SELF_ATTN_O, vPrefix, vIl, "bias"));
        vPl.vLn1W = vGetTensor(fStringFormat(TN_QF_SELF_ATTN_N, vPrefix, vIl, "weight"));
        vPl.vLn1B = vGetTensor(fStringFormat(TN_QF_SELF_ATTN_N, vPrefix, vIl, "bias"));

        vPl.vCrossAttnQW = vGetTensor(fStringFormat(TN_QF_CROSS_ATTN_Q, vPrefix, vIl, "weight"));
        vPl.vCrossAttnQB = vGetTensor(fStringFormat(TN_QF_CROSS_ATTN_Q, vPrefix, vIl, "bias"));
        vPl.vCrossAttnKW = vGetTensor(fStringFormat(TN_QF_CROSS_ATTN_K, vPrefix, vIl, "weight"));
        vPl.vCrossAttnKB = vGetTensor(fStringFormat(TN_QF_CROSS_ATTN_K, vPrefix, vIl, "bias"));
        vPl.vCrossAttnVW = vGetTensor(fStringFormat(TN_QF_CROSS_ATTN_V, vPrefix, vIl, "weight"));
        vPl.vCrossAttnVB = vGetTensor(fStringFormat(TN_QF_CROSS_ATTN_V, vPrefix, vIl, "bias"));
        vPl.vCrossAttnOW = vGetTensor(fStringFormat(TN_QF_CROSS_ATTN_O, vPrefix, vIl, "weight"));
        vPl.vCrossAttnOB = vGetTensor(fStringFormat(TN_QF_CROSS_ATTN_O, vPrefix, vIl, "bias"));
        vPl.vCrossAttnNormW = vGetTensor(fStringFormat(TN_QF_CROSS_ATTN_N, vPrefix, vIl, "weight"));
        vPl.vCrossAttnNormB = vGetTensor(fStringFormat(TN_QF_CROSS_ATTN_N, vPrefix, vIl, "bias"));

        vPl.vFfUpW = vGetTensor(fStringFormat(TN_QF_FFN_UP, vPrefix, vIl, "weight"));
        vPl.vFfUpB = vGetTensor(fStringFormat(TN_QF_FFN_UP, vPrefix, vIl, "bias"));
        vPl.vFfDownW = vGetTensor(fStringFormat(TN_QF_FFN_DOWN, vPrefix, vIl, "weight"));
        vPl.vFfDownB = vGetTensor(fStringFormat(TN_QF_FFN_DOWN, vPrefix, vIl, "bias"));
        vPl.vLn2W = vGetTensor(fStringFormat(TN_QF_FFN_NORM, vPrefix, vIl, "weight"));
        vPl.vLn2B = vGetTensor(fStringFormat(TN_QF_FFN_NORM, vPrefix, vIl, "bias"));
      }
    } break;
    case cProjectorTypeGranite4Vision: {
      // image_newline lives at the top-level.
      vModel.vImageNewline = vGetTensor(TN_IMAGE_NEWLINE);

      // Load separate layerwise and spatial projector tensors
      const auto cProjectorCount = vHparams.lFeatureLayers.size();
      vModel.lQfProjBlocks.resize(cProjectorCount);
      for (size_t vBid = 0; vBid < cProjectorCount; ++vBid) {
        auto &vB = vModel.lQfProjBlocks[vBid];

        // non-layerwise tensors
        vB.vQfProjImgPos = vGetTensor(fStringFormat(TN_MULTI_PROJ_IMG_POS, vBid));
        vB.vQfProjQuery = vGetTensor(fStringFormat(TN_MULTI_PROJ_QUERY, vPrefix, vBid));
        vB.vQfProjLinearW = vGetTensor(fStringFormat(TN_MULTI_PROJ_LINEAR, vPrefix, vBid, "weight"));
        vB.vQfProjLinearB = vGetTensor(fStringFormat(TN_MULTI_PROJ_LINEAR, vPrefix, vBid, "bias"));
        vB.vQfProjNormW = vGetTensor(fStringFormat(TN_MULTI_PROJ_NORM, vPrefix, vBid, "weight"));
        vB.vQfProjNormB = vGetTensor(fStringFormat(TN_MULTI_PROJ_NORM, vPrefix, vBid, "bias"));
        vB.vQfProjPostNormW = vGetTensor(fStringFormat(TN_MULTI_PROJ_POST_NORM, vPrefix, vBid, "weight"));
        vB.vQfProjPostNormB = vGetTensor(fStringFormat(TN_MULTI_PROJ_POST_NORM, vPrefix, vBid, "bias"));

        // laywerwise tensors
        // NOTE: If any model uses multi-layer qformers, this will need to change
        vB.lQfProjLayers.resize(1);
        auto &vPl = vB.lQfProjLayers[0];

        vPl.vQW = vGetTensor(fStringFormat(TN_QF_SELF_ATTN_Q, vPrefix, vBid, "weight"));
        vPl.vQB = vGetTensor(fStringFormat(TN_QF_SELF_ATTN_Q, vPrefix, vBid, "bias"));
        vPl.vKW = vGetTensor(fStringFormat(TN_QF_SELF_ATTN_K, vPrefix, vBid, "weight"));
        vPl.vKB = vGetTensor(fStringFormat(TN_QF_SELF_ATTN_K, vPrefix, vBid, "bias"));
        vPl.vVW = vGetTensor(fStringFormat(TN_QF_SELF_ATTN_V, vPrefix, vBid, "weight"));
        vPl.vVB = vGetTensor(fStringFormat(TN_QF_SELF_ATTN_V, vPrefix, vBid, "bias"));
        vPl.vOW = vGetTensor(fStringFormat(TN_QF_SELF_ATTN_O, vPrefix, vBid, "weight"));
        vPl.vOB = vGetTensor(fStringFormat(TN_QF_SELF_ATTN_O, vPrefix, vBid, "bias"));
        vPl.vLn1W = vGetTensor(fStringFormat(TN_QF_SELF_ATTN_N, vPrefix, vBid, "weight"));
        vPl.vLn1B = vGetTensor(fStringFormat(TN_QF_SELF_ATTN_N, vPrefix, vBid, "bias"));

        vPl.vCrossAttnQW = vGetTensor(fStringFormat(TN_QF_CROSS_ATTN_Q, vPrefix, vBid, "weight"));
        vPl.vCrossAttnQB = vGetTensor(fStringFormat(TN_QF_CROSS_ATTN_Q, vPrefix, vBid, "bias"));
        vPl.vCrossAttnKW = vGetTensor(fStringFormat(TN_QF_CROSS_ATTN_K, vPrefix, vBid, "weight"));
        vPl.vCrossAttnKB = vGetTensor(fStringFormat(TN_QF_CROSS_ATTN_K, vPrefix, vBid, "bias"));
        vPl.vCrossAttnVW = vGetTensor(fStringFormat(TN_QF_CROSS_ATTN_V, vPrefix, vBid, "weight"));
        vPl.vCrossAttnVB = vGetTensor(fStringFormat(TN_QF_CROSS_ATTN_V, vPrefix, vBid, "bias"));
        vPl.vCrossAttnOW = vGetTensor(fStringFormat(TN_QF_CROSS_ATTN_O, vPrefix, vBid, "weight"));
        vPl.vCrossAttnOB = vGetTensor(fStringFormat(TN_QF_CROSS_ATTN_O, vPrefix, vBid, "bias"));
        vPl.vCrossAttnNormW = vGetTensor(fStringFormat(TN_QF_CROSS_ATTN_N, vPrefix, vBid, "weight"));
        vPl.vCrossAttnNormB = vGetTensor(fStringFormat(TN_QF_CROSS_ATTN_N, vPrefix, vBid, "bias"));

        vPl.vFfUpW = vGetTensor(fStringFormat(TN_QF_FFN_UP, vPrefix, vBid, "weight"));
        vPl.vFfUpB = vGetTensor(fStringFormat(TN_QF_FFN_UP, vPrefix, vBid, "bias"));
        vPl.vFfDownW = vGetTensor(fStringFormat(TN_QF_FFN_DOWN, vPrefix, vBid, "weight"));
        vPl.vFfDownB = vGetTensor(fStringFormat(TN_QF_FFN_DOWN, vPrefix, vBid, "bias"));
        vPl.vLn2W = vGetTensor(fStringFormat(TN_QF_FFN_NORM, vPrefix, vBid, "weight"));
        vPl.vLn2B = vGetTensor(fStringFormat(TN_QF_FFN_NORM, vPrefix, vBid, "bias"));
      }

    } break;
    default:
      GGML_ASSERT(false && "unknown projector type");
    }

    // load data
    {
      std::vector<uint8_t> lReadBuf;

      // start loading event
      if (vProgressCallback) {
        vProgressCallback(0.0, vProgressCallbackUserData);
      }

      // compute total tensor data size for progress reporting
      size_t vTotalDataSize = 0;
      for (auto &vT : lTensorsToLoad) {
        vTotalDataSize += ggml_nbytes(vT);
      }

      // alloc memory and offload data
      ggml_backend_buffer_type_t vBuft = ggml_backend_get_default_buffer_type(pCtxClip.vBackend);
      pCtxClip.vBuf.reset(ggml_backend_alloc_ctx_tensors_from_buft(pCtxClip.vCtxData.get(), vBuft));
      ggml_backend_buffer_set_usage(pCtxClip.vBuf.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
      // read the weight from file
      if (!pCtxClip.vNoAlloc) {
        size_t vDataLoaded = 0;
        for (auto &vT : lTensorsToLoad) {
          ggml_tensor *vCur = ggml_get_tensor(pCtxClip.vCtxData.get(), vT->name);
          GGML_ASSERT(vCur && "tensor not found in ctx_data");
          auto vItOff = dTensorOffset.find(vT->name);
          GGML_ASSERT(vItOff != dTensorOffset.end() && "no offset for tensor");
          const size_t cOffset = vItOff->second;
          vFin.seekg(cOffset, std::ios::beg);
          if (!vFin) {
            throw std::runtime_error(fStringFormat("%s: failed to seek for tensor %s\n", __func__, vT->name));
          }
          size_t vNumBytes = ggml_nbytes(vCur);
          if (ggml_backend_buft_is_host(vBuft)) {
            // for the CPU and Metal backend, we can read directly into the tensor
            vFin.read(reinterpret_cast<char *>(vCur->data), vNumBytes);
          } else {
            // read into a temporary buffer first, then copy to device memory
            lReadBuf.resize(vNumBytes);
            vFin.read(reinterpret_cast<char *>(lReadBuf.data()), vNumBytes);
            ggml_backend_tensor_set(vCur, lReadBuf.data(), 0, vNumBytes);
          }
          vDataLoaded += vNumBytes;
          if (vProgressCallback && vTotalDataSize > 0) {
            const float cProgress = (float)vDataLoaded / (float)vTotalDataSize;
            if (!vProgressCallback(cProgress, vProgressCallbackUserData)) {
              throw std::runtime_error(fStringFormat("%s: model loading cancelled by progress_callback\n", __func__));
            }
          }
        }
        LOG_DBG("%s: loaded %zu tensors from %s\n", __func__, lTensorsToLoad.size(), vFname.c_str());
      } else {
        LOG_DBG("%s: no_alloc is set, skipping tensor data loading (%zu tensors)\n", __func__, lTensorsToLoad.size());
      }
      vFin.close();
    }
  }

  struct support_info_op {
    ggml_tensor *vOp;

    // true if the op runs on the accelerated clip_ctx.vBackend
    bool vIsAccel = true;
  };

  struct support_info_graph {
    // whether the clip_ctx.vBackend supports flash attention
    bool vFattn = true;
    ggml_tensor *vFattnOp = nullptr; // for debugging

    std::vector<support_info_op> lOps;
  };

  static clip_image_f32_batch mGetDummyBatch(clip_ctx &pCtxClip) {
    // create a fake batch
    const auto &vHparams = pCtxClip.vModel.vHparams;
    clip_image_f32_batch vBatch;
    clip_image_f32 vImg;
    if (pCtxClip.vModel.vModality == cClipModalityVision) {
      const int cSz = vHparams.vWarmupImageSize;
      vImg.mSetSize({cSz, cSz}, false, false);
      LOG_INF("%s: warmup with image size = %d x %d\n", __func__, cSz, cSz);
    } else {
      if (vHparams.vNMelBins <= 0 || vHparams.vNMelBins > 256) {
        throw std::runtime_error(
            fStringFormat("%s: invalid n_mel_bins (%d), must be in [1, 256]\n", __func__, vHparams.vNMelBins));
      }
      vImg.mSetSize({vHparams.vWarmupAudioSize, vHparams.vNMelBins}, false, false);
      LOG_INF("%s: warmup with audio size = %d\n", __func__, vHparams.vWarmupAudioSize);
    }
    vBatch.lEntries.push_back(vImg);
    return vBatch;
  }

  static void mInitCtx(clip_ctx &pCtxClip) {
    pCtxClip.lBufComputeMeta.resize(pCtxClip.vMaxNodes * ggml_tensor_overhead() + ggml_graph_overhead());

    // check batching support
    auto vBatch = mGetDummyBatch(pCtxClip);
    auto vBuilder = fClipGetGraphBuilder(&pCtxClip, vBatch);
    pCtxClip.vSupportBatch = vBuilder->mSupportBatch();
  }

  static void mWarmup(clip_ctx &pCtxClip) {
    auto vBatch = mGetDummyBatch(pCtxClip);
    mWarmup(pCtxClip, vBatch);
  }

  static void mWarmup(clip_ctx &pCtxClip, const clip_image_f32_batch &pBatch) {
    support_info_graph vInfo;

    if (pCtxClip.vFlashAttnType == cClipFlashAttnTypeAuto) {
      // try to enable flash attention to see if it's supported
      pCtxClip.vFlashAttnType = cClipFlashAttnTypeEnabled;
      vInfo = mReserveComputeMeta(pCtxClip, pBatch);
      if (!vInfo.vFattn && vInfo.vFattnOp) {
        auto vOp = vInfo.vFattnOp;
        LOG_WRN("%s: *****************************************************************\n", __func__);
        LOG_WRN("%s: WARNING: flash attention not supported by %s, memory usage will increase\n", __func__,
                ggml_backend_name(pCtxClip.vBackend));
        LOG_WRN("%s: op params: \n", __func__);
        static auto vPrintShape = [](const char *pFn, const char *pName, ggml_tensor *pT) {
          LOG_WRN("%s:   %s: type = %s, ne = [%d %d %d %d], nb = [%d %d %d %d]\n", pFn, pName, ggml_type_name(pT->type),
                  pT->ne[0], pT->ne[1], pT->ne[2], pT->ne[3], pT->nb[0], pT->nb[1], pT->nb[2], pT->nb[3]);
        };
        vPrintShape(__func__, " dst", vOp);
        vPrintShape(__func__, "src0", vOp->src[0]);
        vPrintShape(__func__, "src1", vOp->src[1]);
        vPrintShape(__func__, "src2", vOp->src[2]);
        LOG_WRN("%s: please report this on github as an issue\n", __func__);
        LOG_WRN("%s: *****************************************************************\n", __func__);
        pCtxClip.vFlashAttnType = cClipFlashAttnTypeDisabled;
        mReserveComputeMeta(pCtxClip, pBatch);
      }
    } else {
      vInfo = mReserveComputeMeta(pCtxClip, pBatch);
      if (!vInfo.vFattn && pCtxClip.vFlashAttnType == cClipFlashAttnTypeEnabled) {
        LOG_WRN("%s: flash attention is not supported by the current backend; falling back to CPU (performance will be "
                "degraded)\n",
                __func__);
      }
    }

    pCtxClip.vIsAllocated = true; // mark buffers as allocated

    LOG_INF("%s: flash attention is %s\n", __func__,
            (pCtxClip.vFlashAttnType == cClipFlashAttnTypeEnabled) ? "enabled" : "disabled");

    // print ops that are not supported by the GPU backend (if there is one)
    if (pCtxClip.vBackend && pCtxClip.vBackend != pCtxClip.vBackendCpu) {
      std::vector<support_info_op> lUnsupportedOps;
      for (const auto &vOp : vInfo.lOps) {
        if (!vOp.vIsAccel) {
          lUnsupportedOps.push_back(vOp);
        }
      }
      if (!lUnsupportedOps.empty()) {
        LOG_WRN("%s: *****************************************************************\n", __func__);
        LOG_WRN("%s: WARNING: the CLIP graph uses unsupported operators by the backend\n", __func__);
        LOG_WRN("%s:          the performance will be suboptimal                      \n", __func__);
        LOG_WRN("%s:          list of unsupported ops (backend=%s):\n", __func__, ggml_backend_name(pCtxClip.vBackend));
        for (const auto &vOp : lUnsupportedOps) {
          LOG_WRN("%s: %16s: type = %s, ne = [%d %d %d %d]\n", __func__, ggml_op_name(vOp.vOp->op),
                  ggml_type_name(vOp.vOp->type), vOp.vOp->ne[0], vOp.vOp->ne[1], vOp.vOp->ne[2], vOp.vOp->ne[3]);
        }
        LOG_WRN("%s: flash attention is %s\n", __func__,
                (pCtxClip.vFlashAttnType == cClipFlashAttnTypeEnabled) ? "enabled" : "disabled");
        LOG_WRN("%s: please report this at https://github.com/nipegun/moe-gguf-server/issues\n", __func__);
        LOG_WRN("%s: *****************************************************************\n", __func__);
      }
    }
  }

  // only initialize backend buffers, but do not allocate them yet
  static support_info_graph mReserveComputeMeta(clip_ctx &pCtxClip, const clip_image_f32_batch &pBatch) {
    ggml_cgraph *vGf = fClipGetGraphBuilder(&pCtxClip, pBatch)->mBuild();
    ggml_backend_sched_reserve(pCtxClip.vSched.get(), vGf);

    pCtxClip.dMemCompute.clear();
    for (size_t vI = 0; vI < pCtxClip.lBackendPtrs.size(); ++vI) {
      ggml_backend_t vBackend = pCtxClip.lBackendPtrs[vI];
      ggml_backend_buffer_type_t vBuft = pCtxClip.lBackendBuft[vI];
      size_t vSize = ggml_backend_sched_get_buffer_size(pCtxClip.vSched.get(), vBackend);
      if (vSize > 1) {
        LOG_INF("%s: %10s compute buffer size = %8.2f MiB\n", __func__, ggml_backend_buft_name(vBuft),
                vSize / 1024.0 / 1024.0);
      }
      pCtxClip.dMemCompute[ggml_backend_get_device(vBackend)] += vSize;
    }

    const int cNSplits = ggml_backend_sched_get_n_splits(pCtxClip.vSched.get());
    const int cNNodes = ggml_graph_n_nodes(vGf);

    LOG_INF("%s: graph splits = %d, nodes = %d\n", __func__, cNSplits, cNNodes);

    support_info_graph vRes{
        /*.vFattn   = */ true,
        /*.vFattnOp = */ nullptr,
        /*.lOps     = */ {},
    };

    // check op support
    for (int vI = 0; vI < ggml_graph_n_nodes(vGf); vI++) {
      ggml_tensor *vNode = ggml_graph_node(vGf, vI);
      vRes.lOps.push_back({vNode, true});
      if (!ggml_backend_supports_op(pCtxClip.vBackend, vNode)) {
        vRes.lOps.back().vIsAccel = false;
        if (vNode->op == GGML_OP_FLASH_ATTN_EXT) {
          vRes.vFattn = false;
          vRes.vFattnOp = vNode;
        }
      }
    }

    return vRes;
  }

  void mGetBool(const std::string &pKey, bool &pOutput, bool pRequired = true) const {
    const int cI = gguf_find_key(vCtxGguf.get(), pKey.c_str());
    if (cI < 0) {
      if (pRequired) {
        throw std::runtime_error("Key not found: " + pKey);
      }
      return;
    }
    pOutput = gguf_get_val_bool(vCtxGguf.get(), cI);
  }

  void mGetI32(const std::string &pKey, int &pOutput, bool pRequired = true) const {
    const int cI = gguf_find_key(vCtxGguf.get(), pKey.c_str());
    if (cI < 0) {
      if (pRequired) {
        throw std::runtime_error("Key not found: " + pKey);
      }
      return;
    }
    pOutput = gguf_get_val_i32(vCtxGguf.get(), cI);
  }

  void mGetU32(const std::string &pKey, int &pOutput, bool pRequired = true) const {
    const int cI = gguf_find_key(vCtxGguf.get(), pKey.c_str());
    if (cI < 0) {
      if (pRequired) {
        throw std::runtime_error("Key not found: " + pKey);
      }
      return;
    }
    const uint32_t cVal = gguf_get_val_u32(vCtxGguf.get(), cI);
    // sanity check
    if (cVal > (uint32_t)INT32_MAX) {
      throw std::runtime_error(
          fStringFormat("%s: value %u for key '%s' exceeds INT32_MAX\n", __func__, cVal, pKey.c_str()));
    }
    pOutput = (int)cVal;
  }

  void mGetF32(const std::string &pKey, float &pOutput, bool pRequired = true) const {
    const int cI = gguf_find_key(vCtxGguf.get(), pKey.c_str());
    if (cI < 0) {
      if (pRequired) {
        throw std::runtime_error("Key not found: " + pKey);
      }
      return;
    }
    pOutput = gguf_get_val_f32(vCtxGguf.get(), cI);
  }

  void mGetArrF32(const std::string &pKey, std::vector<float> &pOutput, bool pRequired = true) const {
    const int cI = gguf_find_key(vCtxGguf.get(), pKey.c_str());
    if (cI < 0) {
      if (pRequired) {
        throw std::runtime_error("Key not found: " + pKey);
      }
      return;
    }
    const auto cType = gguf_get_arr_type(vCtxGguf.get(), cI);
    if (cType != GGUF_TYPE_FLOAT32) {
      throw std::runtime_error(fStringFormat("%s: array '%s' has type %d, expected %d (GGUF_TYPE_FLOAT32)\n", __func__,
                                             pKey.c_str(), cType, GGUF_TYPE_FLOAT32));
    }
    const size_t cN = gguf_get_arr_n(vCtxGguf.get(), cI);
    if (cN > (size_t)std::numeric_limits<int>::max()) {
      throw std::runtime_error(
          fStringFormat("%s: array '%s' is too large (%zu elements)\n", __func__, pKey.c_str(), cN));
    }
    pOutput.resize(cN);
    const float *vValues = (const float *)gguf_get_arr_data(vCtxGguf.get(), cI);
    for (size_t vJ = 0; vJ < cN; ++vJ) {
      pOutput[vJ] = vValues[vJ];
    }
  }

  void mGetString(const std::string &pKey, std::string &pOutput, bool pRequired = true) const {
    const int cI = gguf_find_key(vCtxGguf.get(), pKey.c_str());
    if (cI < 0) {
      if (pRequired) {
        throw std::runtime_error("Key not found: " + pKey);
      }
      return;
    }
    pOutput = std::string(gguf_get_val_str(vCtxGguf.get(), cI));
  }

  void mGetArrInt(const std::string &pKey, std::vector<int> &pOutput, bool pRequired = true) const {
    const int cI = gguf_find_key(vCtxGguf.get(), pKey.c_str());
    if (cI < 0) {
      if (pRequired) {
        throw std::runtime_error("Key not found: " + pKey);
      }
      return;
    }
    const auto cType = gguf_get_arr_type(vCtxGguf.get(), cI);
    if (cType != GGUF_TYPE_INT32) {
      throw std::runtime_error(fStringFormat("%s: array '%s' has type %d, expected %d (GGUF_TYPE_INT32)\n", __func__,
                                             pKey.c_str(), cType, GGUF_TYPE_INT32));
    }
    const size_t cN = gguf_get_arr_n(vCtxGguf.get(), cI);
    if (cN > (size_t)std::numeric_limits<int>::max()) {
      throw std::runtime_error(
          fStringFormat("%s: array '%s' is too large (%zu elements)\n", __func__, pKey.c_str(), cN));
    }
    pOutput.resize(cN);
    const int32_t *vValues = (const int32_t *)gguf_get_arr_data(vCtxGguf.get(), cI);
    for (size_t vJ = 0; vJ < cN; ++vJ) {
      pOutput[vJ] = vValues[vJ];
    }
  }

  static void mSetInternvlDhrResCandidates(clip_model &pModel) {
    auto &vHparams = pModel.vHparams;
    int vMinNum = vHparams.vPreprocMinTiles;
    int vMaxNum = vHparams.vPreprocMaxTiles;
    if (vMinNum < 1) {
      return; // avoid  divide by 0
    }
    for (int vA = vMinNum; vA <= vMaxNum; ++vA) {
      int vBLo = (vMinNum + vA - 1) / vA;
      int vBHi = vMaxNum / vA;
      vBLo = std::max(vBLo, vMinNum);
      vBHi = std::min(vBHi, vMaxNum);
      for (int vB = vBLo; vB <= vBHi; ++vB) {
        vHparams.lImageResCandidates.push_back(clip_image_size{
            vA * vHparams.vImageSize,
            vB * vHparams.vImageSize,
        });
      }
    }
  }
};

struct clip_init_result fClipInit(const char *pFname, struct clip_context_params pCtxParams) {
  clip_ctx *vCtxVision = nullptr;
  clip_ctx *vCtxAudio = nullptr;

  try {
    clip_model_loader vLoader(pFname,
                              /* skip_tensors */ false, pCtxParams.vProgressCallback,
                              pCtxParams.vProgressCallbackUserData);
    bool vSkipAudio = false;

    if (vLoader.vHasVision) {
      vCtxVision = new clip_ctx(pCtxParams);
      vLoader.mLoadHparams(vCtxVision->vModel, cClipModalityVision);
      vLoader.mLoadTensors(*vCtxVision);
      vLoader.mInitCtx(*vCtxVision);
      if (pCtxParams.vWarmup) {
        vLoader.mWarmup(*vCtxVision);
      }

      // TODO: we don't support audio for Gemma 3N, but GGUF contains audio tensors
      // we can remove this check when we implement audio support for Gemma 3N
      vSkipAudio = vCtxVision->vModel.vProjType == cProjectorTypeGemma3nv;
    }

    if (vLoader.vHasAudio && !vSkipAudio) {
      vCtxAudio = new clip_ctx(pCtxParams);
      vLoader.mLoadHparams(vCtxAudio->vModel, cClipModalityAudio);
      vLoader.mLoadTensors(*vCtxAudio);
      vLoader.mInitCtx(*vCtxAudio);
      if (pCtxParams.vWarmup) {
        vLoader.mWarmup(*vCtxAudio);
      }
    }

  } catch (const std::exception &vE) {
    LOG_ERR("%s: failed to load model '%s': %s\n", __func__, pFname, vE.what());

    delete vCtxVision;
    delete vCtxAudio;

    return {nullptr, nullptr};
  }

  return {vCtxVision, vCtxAudio};
}

struct clip_cap fClipGetCap(const char *pFname) {
  clip_cap vRes;
  clip_model_loader vLoader(pFname, /* skip_tensors= */ true);
  vRes.vHasVision = vLoader.vHasVision;
  vRes.vHasAudio = vLoader.vHasAudio;
  return vRes;
}

void fClipFree(clip_ctx *pCtx) {
  if (pCtx == nullptr) {
    return;
  }
  delete pCtx;
}

int fClipNOutputTokensX(const clip_ctx *pCtx, const clip_image_f32 *pImg) {
  const auto &vParams = pCtx->vModel.vHparams;
  const int cNTotal = fClipNOutputTokens(pCtx, pImg);
  const auto &vProj = pCtx->vModel.vProjType;
  switch (vProj) {
  case cProjectorTypeQwen2vl:
  case cProjectorTypeQwen25vl:
  case cProjectorTypeQwen3vl:
  case cProjectorTypeEXAONE4Point5:
  case cProjectorTypeMimovl:
  case cProjectorTypeGlm4v:
  case cProjectorTypeGlm5v:
  case cProjectorTypePaddleocr:
  case cProjectorTypeHunyuanvl:
  case cProjectorTypeYoutuvl:
    return (pImg->mNx() / vParams.vPatchSize) / 2;
  case cProjectorTypeStep3vl:
    return pImg->mNx() / (vParams.vPatchSize * vParams.vNMerge);
  default:
    break;
  }
  return cNTotal;
}

int fClipNOutputTokensY(const clip_ctx *pCtx, const clip_image_f32 *pImg) {
  const auto &vParams = pCtx->vModel.vHparams;
  const auto &vProj = pCtx->vModel.vProjType;
  switch (vProj) {
  case cProjectorTypeQwen2vl:
  case cProjectorTypeQwen25vl:
  case cProjectorTypeQwen3vl:
  case cProjectorTypeEXAONE4Point5:
  case cProjectorTypeMimovl:
  case cProjectorTypeGlm4v:
  case cProjectorTypeGlm5v:
  case cProjectorTypePaddleocr:
  case cProjectorTypeHunyuanvl:
  case cProjectorTypeYoutuvl:
    return (pImg->mNy() / vParams.vPatchSize) / 2;
  case cProjectorTypeStep3vl:
    return pImg->mNy() / (vParams.vPatchSize * vParams.vNMerge);
  default:
    break;
  }
  return 1;
}

int fClipNOutputTokens(const clip_ctx *pCtx, const clip_image_f32 *pImg) {
  const auto &vParams = pCtx->vModel.vHparams;

  // for models with fixed size image, the input image is already pre-processed and resized to square
  int vPatchSize = vParams.vPatchSize;
  int vNPatches = (pImg->mNx() / vPatchSize) * (pImg->mNy() / vPatchSize);

  projector_type vProj = pCtx->vModel.vProjType;

  switch (vProj) {
  case cProjectorTypeJanusPro:
  case cProjectorTypePhi4: {
    // do nothing
  } break;
  case cProjectorTypeYasa2: {
    vNPatches = 64; // adaptive average pooling to 8x8 tokens
  } break;
  case cProjectorTypeMINICPMV4Point6: {
    vNPatches /= vParams.vNMerge * vParams.vNMerge;
  } break;
  case cProjectorTypeQwen2vl:
  case cProjectorTypeQwen25vl:
  case cProjectorTypeQwen3vl:
  case cProjectorTypeEXAONE4Point5:
  case cProjectorTypeMimovl:
  case cProjectorTypeMinimaxM3:
  case cProjectorTypeGlm4v:
  case cProjectorTypeGlm5v:
  case cProjectorTypeYoutuvl: {
    // dynamic size (2 conv, so double patch size)
    int vXPatch = pImg->mNx() / (vParams.vPatchSize * 2);
    int vYPatch = pImg->mNy() / (vParams.vPatchSize * 2);
    vNPatches = vXPatch * vYPatch;
  } break;
  case cProjectorTypeStep3vl: {
    int vXPatch = pImg->mNx() / (vParams.vPatchSize * vParams.vNMerge);
    int vYPatch = pImg->mNy() / (vParams.vPatchSize * vParams.vNMerge);
    vNPatches = vXPatch * vYPatch;
  } break;
  case cProjectorTypeGemma3:
  case cProjectorTypeIdefics3:
  case cProjectorTypeInternvl:
  case cProjectorTypeNemotronV2Vl: {
    // both X and Y are downscaled by the scale factor
    int vScaleFactor = pCtx->vModel.vHparams.vNMerge;
    vNPatches /= (vScaleFactor * vScaleFactor);
  } break;
  case cProjectorTypeGemma3nv: {
    // MobileNetV5 MSFA adapter always outputs fixed 16x16 resolution
    // regardless of input size (see architecture description)
    vNPatches = pCtx->vModel.vHparams.vImageSize / pCtx->vModel.vHparams.vPatchSize;
  } break;
  case cProjectorTypeLfm2:
  case cProjectorTypeKimivl:
  case cProjectorTypeKimik25: {
    // dynamic size
    int vOutPatchSize = vParams.vPatchSize * pCtx->vModel.vHparams.vNMerge;
    int vXPatch = CLIP_ALIGN(pImg->mNx(), vOutPatchSize) / vOutPatchSize;
    int vYPatch = CLIP_ALIGN(pImg->mNy(), vOutPatchSize) / vOutPatchSize;
    vNPatches = vXPatch * vYPatch;
  } break;
  case cProjectorTypePaddleocr:
  case cProjectorTypeDotsOcr: {
    // dynamic size
    int vNMerge = pCtx->vModel.vHparams.vNMerge;
    int vStride = vNMerge * vNMerge;
    vNPatches = CLIP_ALIGN(vNPatches, vStride) / vStride;
  } break;
  case cProjectorTypePixtral:
  case cProjectorTypeLightonocr: {
    // dynamic size
    int vNMerge = pCtx->vModel.vHparams.vNMerge;
    int vNPatchesX = pImg->mNx() / vPatchSize / vNMerge;
    int vNPatchesY = pImg->mNy() / vPatchSize / vNMerge;
    if (pCtx->vModel.vTokenEmbdImgBreak) {
      vNPatches = vNPatchesY * vNPatchesX + vNPatchesY - 1; // + one [IMG_BREAK] per row, except the last row
    } else {
      vNPatches = vNPatchesY * vNPatchesX;
    }
  } break;
  case cProjectorTypeVoxtral:
  case cProjectorTypeUltravox:
  case cProjectorTypeQwen2a:
  case cProjectorTypeMeralion:
  case cProjectorTypeMusicFlamingo: {
    vNPatches = pImg->mNx();

    const int cProjStackFactor = pCtx->vModel.vHparams.vProjStackFactor;
    if (pCtx->vModel.mAudioHasStackFrames()) {
      GGML_ASSERT(cProjStackFactor > 0);
      const int cNLen = CLIP_ALIGN(vNPatches, cProjStackFactor);
      vNPatches = cNLen / cProjStackFactor;
    }

    // whisper downscales input token by half after conv1d
    vNPatches /= 2;

    if (pCtx->vModel.mAudioHasAvgpool()) {
      // divide by 2 because of nn.AvgPool1d(2, stride=2)
      vNPatches /= 2;
    }
  } break;
  case cProjectorTypeQwen3a: {
    // chunk_size=100 frames --> 3x stride-2 conv2d --> 13 tokens per chunk
    const int cChunkSize = 100;
    const int cTokensPerChunk = 13;
    vNPatches = (pImg->mNx() / cChunkSize) * cTokensPerChunk;
  } break;
  case cProjectorTypeGlma: {
    vNPatches = pImg->mNx();
    // whisper downscales input token by half after conv1d
    vNPatches /= 2;
    // reshape by merge_factor
    vNPatches /= pCtx->vModel.vHparams.vProjStackFactor;
    // for BOI and EOI token embeddings
    vNPatches += 2;
  } break;
  case cProjectorTypeCogvlm: {
    vNPatches += 2; // for BOI and EOI token embeddings
  } break;
  case cProjectorTypeHunyuanvl: {
    int vMerge = pCtx->vModel.vHparams.vNMerge;
    int vOw = (pImg->mNx() / vPatchSize) / vMerge;
    int vOh = (pImg->mNy() / vPatchSize) / vMerge;
    vNPatches = (vOw + 1) * vOh + 2;
  } break;
  case cProjectorTypeLfm2a: {
    vNPatches = ((((pImg->mNx() + 1) / 2) + 1) / 2 + 1) / 2;
  } break;
  case cProjectorTypeParakeet: {
    vNPatches = (pImg->mNx() + (vParams.vSubsamplingFactor - 1)) / vParams.vSubsamplingFactor;
  } break;
  case cProjectorTypeMimoAudio: {
    // conv1(s=1) + conv2(s=2) -> RVQ-encoder downsample conv(k=2,s=2)
    int vN = pImg->mNx();
    vN = (vN - 1) / 2 + 1; // conv1 + conv2
    vN = (vN - 2) / 2 + 1; // downsample conv
    const int cGroupSize = vParams.vAudioLocalGroupSize;
    vNPatches = (vN + cGroupSize - 1) / cGroupSize;
  } break;
  case cProjectorTypeGraniteSpeech: {
    const int cWs = pCtx->vModel.vHparams.vAudioProjWindowSize;
    const int cDs = pCtx->vModel.vHparams.vAudioProjDownsampleRate;
    vNPatches = ((pImg->mNx() + cWs - 1) / cWs) * (cWs / cDs);
  } break;
  case cProjectorTypeGranite4Vision: {
    // Per-tile output token count: each projector block outputs
    // query_side^2 tokens per window × n^2 windows.
    // For 384×384 input: n = 24/8 = 3, query_side = 4 → 144.
    const int cWindowSide = pCtx->vModel.vHparams.vDownsampleWindowSide;
    const int cQuerySide = pCtx->vModel.vHparams.vDownsampleQuerySide;
    const int cSide = pImg->mNx() / vParams.vPatchSize;
    const int cN = cSide / cWindowSide;
    vNPatches = (cQuerySide * cN) * (cQuerySide * cN);
    if (pImg->vAddNewline) {
      // For single-tile case: append 1 newline row.
      // For multi-tile rowwise: handled by caller, but here we
      // report the per-tile count including one trailing newline.
      vNPatches += 1;
    }
  } break;
  default:
    GGML_ABORT("unsupported projector type");
  }

  return vNPatches;
}

bool fClipImageEncode(struct clip_ctx *pCtx, int pNThreads, const clip_image_f32 *pImg, std::vector<float> &pOutVec) {
  clip_image_f32_batch vImgs;
  clip_image_f32 vImgCopy = *pImg;
  vImgs.lEntries.push_back(std::move(vImgCopy));

  return fClipImageBatchEncode(pCtx, pNThreads, &vImgs, pOutVec);
}

bool fClipImageBatchEncode(clip_ctx *pCtx, int pNThreads, const clip_image_f32_batch *pImgsCPtr,
                           std::vector<float> &pOutBatchEmbd) {
  const clip_image_f32_batch &vImgs = *pImgsCPtr;
  int vNBatchCur = vImgs.lEntries.size();

  // [QWEN_VIDEO] for video models, the batch dimension is used as temporal dimension for merged frames
  if (!pCtx->vSupportBatch && vNBatchCur > fClipModelNTemporalMerge(pCtx)) {
    LOG_ERR("%s: batch size %d exceeds maximum supported batch/temporal-merge size %d\n", __func__, vNBatchCur,
            fClipModelNTemporalMerge(pCtx));
    return false;
  }

  // if buffers are not allocated, we need to do a warmup run to allocate them
  if (!pCtx->vIsAllocated) {
    clip_model_loader::mWarmup(*pCtx, *pImgsCPtr);
  }

  // build the inference graph
  ggml_backend_sched_reset(pCtx->vSched.get());
  ggml_cgraph *vGf = fClipGetGraphBuilder(pCtx, vImgs)->mBuild();
  ggml_backend_sched_alloc_graph(pCtx->vSched.get(), vGf);

  // set inputs
  const auto &vModel = pCtx->vModel;
  const auto &vHparams = vModel.vHparams;

  const int cImageSizeWidth = vImgs.lEntries[0].mNx();
  const int cImageSizeHeight = vImgs.lEntries[0].mNy();

  const int cPatchSize = vHparams.vPatchSize;
  const int cNumPatches = ((cImageSizeWidth / cPatchSize) * (cImageSizeHeight / cPatchSize));
  const int cNPos = cNumPatches + (vModel.vClassEmbedding ? 1 : 0);
  const int cPosW = cImageSizeWidth / cPatchSize;
  const int cPosH = cImageSizeHeight / cPatchSize;

  auto vGetInpTensor = [&vGf](const char *pName) {
    ggml_tensor *vInp = ggml_graph_get_tensor(vGf, pName);
    if (vInp == nullptr) {
      GGML_ABORT("Failed to get tensor %s", pName);
    }
    if (!(vInp->flags & GGML_TENSOR_FLAG_INPUT)) {
      GGML_ABORT("Tensor %s is not an input tensor", pName);
    }
    return vInp;
  };

  auto vSetInputF32 = [&vGetInpTensor](const char *pName, const std::vector<float> &pValues) {
    ggml_tensor *vCur = vGetInpTensor(pName);
    GGML_ASSERT(vCur->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_nelements(vCur) == (int64_t)pValues.size());
    ggml_backend_tensor_set(vCur, pValues.data(), 0, ggml_nbytes(vCur));
  };

  auto vSetInputI32 = [&vGetInpTensor](const char *pName, std::vector<int32_t> &pValues) {
    ggml_tensor *vCur = vGetInpTensor(pName);
    GGML_ASSERT(vCur->type == GGML_TYPE_I32);
    GGML_ASSERT(ggml_nelements(vCur) == (int64_t)pValues.size());
    ggml_backend_tensor_set(vCur, pValues.data(), 0, ggml_nbytes(vCur));
  };

  // set input pixel values
  if (!vImgs.vIsAudio) {
    size_t vNelem = 0;
    for (const auto &vImg : vImgs.lEntries) {
      vNelem += vImg.mNx() * vImg.mNy() * 3;
    }
    std::vector<float> lInpRaw(vNelem);

    // layout of data (note: the channel dim is unrolled to better visualize the layout):
    //
    // ┌──W──┐
    // │     H │  channel = R
    // ├─────┤ │
    // │     H │  channel = G
    // ├─────┤ │
    // │     H │  channel = B
    // └─────┘ │
    //   ──────┘ x B

    // IMPORTANT: [QWEN_VIDEO] the batch dim is currently used for temporal dim in Qwen-VL models
    // All entries must have the same spatial size (enforced by mCanBatchWith() during merging)
    {
      const int cNx = vImgs.lEntries[0].mNx();
      const int cNy = vImgs.lEntries[0].mNy();
      const int cN = cNx * cNy;

      for (int vB = 0; vB < vNBatchCur; vB++) {
        LOG_DBG("%s: copying image %d/%d to input buffer (nx=%d, ny=%d)\n", __func__, vB + 1, vNBatchCur, cNx, cNy);
        const auto &vBuf = vImgs.lEntries[vB].mGetRoBuf();
        float *vBatchEntry = lInpRaw.data() + vB * (3 * cN);
        for (int vY = 0; vY < cNy; vY++) {
          for (int vX = 0; vX < cNx; vX++) {
            size_t vBaseSrc = 3 * (vY * cNx + vX);
            size_t vBaseDst = vY * cNx + vX;
            vBatchEntry[vBaseDst] = vBuf[vBaseSrc];
            vBatchEntry[1 * cN + vBaseDst] = vBuf[vBaseSrc + 1];
            vBatchEntry[2 * cN + vBaseDst] = vBuf[vBaseSrc + 2];
          }
        }
      }
    }
    vSetInputF32("inp_raw", lInpRaw);

  } else {
    // audio input
    GGML_ASSERT(vImgs.lEntries.size() == 1);

    const auto &vMelInp = vImgs.lEntries[0];
    const auto &vBuf = vMelInp.mGetRoBuf();
    const int cNStep = vMelInp.mNx();
    const int cNMel = vMelInp.mNy();
    GGML_ASSERT((size_t)cNStep * cNMel == vBuf.size());

    vSetInputF32("inp_raw", vBuf);
  }

  // set input per projector
  switch (pCtx->vModel.vProjType) {
  case cProjectorTypeMINICPMV4Point6: {
    const bool cIs4x = vHparams.vNMerge == 2;

    // SigLIP position buckets (70x70 grid of learned position embeddings)
    std::vector<int32_t> lPositions(cPosH * cPosW);
    int aBucketCoordsH[1024];
    int aBucketCoordsW[1024];
    for (int vI = 0; vI < cPosH; vI++) {
      aBucketCoordsH[vI] = std::floor(70.0 * vI / cPosH);
    }
    for (int vI = 0; vI < cPosW; vI++) {
      aBucketCoordsW[vI] = std::floor(70.0 * vI / cPosW);
    }
    for (int vI = 0, vId = 0; vI < cPosH; vI++) {
      for (int vJ = 0; vJ < cPosW; vJ++) {
        lPositions[vId++] = aBucketCoordsH[vI] * 70 + aBucketCoordsW[vJ];
      }
    }
    vSetInputI32("positions", lPositions);

    const int cHalfH = cPosH / 2;
    const int cHalfW = cPosW / 2;

    auto vMakeDsIdx = [](int pOffR, int pOffC, int pDsH, int pDsW, int pStrideW) {
      std::vector<int32_t> lIdx(pDsH * pDsW);
      for (int vI = 0; vI < pDsH; vI++) {
        for (int vJ = 0; vJ < pDsW; vJ++) {
          lIdx[vI * pDsW + vJ] = (2 * vI + pOffR) * pStrideW + (2 * vJ + pOffC);
        }
      }
      return lIdx;
    };

    if (!cIs4x) {
      // window reorder indices for 2x2 windows
      std::vector<int32_t> lWindowIdx(cNPos);
      std::vector<int32_t> lInvWindowIdx(cNPos);
      {
        int vK = 0;
        for (int vWi = 0; vWi < cHalfH; vWi++) {
          for (int vWj = 0; vWj < cHalfW; vWj++) {
            lWindowIdx[vK++] = (2 * vWi) * cPosW + (2 * vWj);
            lWindowIdx[vK++] = (2 * vWi) * cPosW + (2 * vWj + 1);
            lWindowIdx[vK++] = (2 * vWi + 1) * cPosW + (2 * vWj);
            lWindowIdx[vK++] = (2 * vWi + 1) * cPosW + (2 * vWj + 1);
          }
        }
        for (int vI = 0; vI < cNPos; vI++) {
          lInvWindowIdx[lWindowIdx[vI]] = vI;
        }
      }
      vSetInputI32("vit_merger_window_idx", lWindowIdx);
      vSetInputI32("vit_merger_inv_window_idx", lInvWindowIdx);

      // block-diagonal attention mask: tokens in the same 4-token
      // window attend to each other (mask = 0), all other positions
      // are masked out (-inf). matches the window-major reorder above.
      std::vector<float> lWindowMaskData(cNPos * cNPos, std::numeric_limits<float>::lowest());
      for (int vWi = 0; vWi < cNPos / 4; vWi++) {
        for (int vI = 0; vI < 4; vI++) {
          for (int vJ = 0; vJ < 4; vJ++) {
            lWindowMaskData[(vWi * 4 + vI) * cNPos + (vWi * 4 + vJ)] = 0.0f;
          }
        }
      }
      vSetInputF32("vit_merger_window_mask", lWindowMaskData);

      // ViT merger 2x2 downsample indices
      auto vVitMergerDs0 = vMakeDsIdx(0, 0, cHalfH, cHalfW, cPosW);
      auto vVitMergerDs1 = vMakeDsIdx(0, 1, cHalfH, cHalfW, cPosW);
      auto vVitMergerDs2 = vMakeDsIdx(1, 0, cHalfH, cHalfW, cPosW);
      auto vVitMergerDs3 = vMakeDsIdx(1, 1, cHalfH, cHalfW, cPosW);
      vSetInputI32("vit_merger_ds_idx_0", vVitMergerDs0);
      vSetInputI32("vit_merger_ds_idx_1", vVitMergerDs1);
      vSetInputI32("vit_merger_ds_idx_2", vVitMergerDs2);
      vSetInputI32("vit_merger_ds_idx_3", vVitMergerDs3);
    }

    const int cMergerH = cIs4x ? cPosH : cHalfH;
    const int cMergerW = cIs4x ? cPosW : cHalfW;
    auto vMDs0 = vMakeDsIdx(0, 0, cMergerH / 2, cMergerW / 2, cMergerW);
    auto vMDs1 = vMakeDsIdx(0, 1, cMergerH / 2, cMergerW / 2, cMergerW);
    auto vMDs2 = vMakeDsIdx(1, 0, cMergerH / 2, cMergerW / 2, cMergerW);
    auto vMDs3 = vMakeDsIdx(1, 1, cMergerH / 2, cMergerW / 2, cMergerW);
    vSetInputI32("merger_ds_idx_0", vMDs0);
    vSetInputI32("merger_ds_idx_1", vMDs1);
    vSetInputI32("merger_ds_idx_2", vMDs2);
    vSetInputI32("merger_ds_idx_3", vMDs3);
  } break;
  case cProjectorTypeQwen2vl:
  case cProjectorTypeQwen3vl:
  case cProjectorTypeGlm4v:
  case cProjectorTypeGlm5v: {
    const int cMergeRatio = vHparams.vNMerge;
    const int cPw = cImageSizeWidth / cPatchSize;
    const int cPh = cImageSizeHeight / cPatchSize;
    std::vector<int> lPositions(cNPos * 4);
    int vPtr = 0;
    for (int vY = 0; vY < cPh; vY += cMergeRatio) {
      for (int vX = 0; vX < cPw; vX += cMergeRatio) {
        for (int vDy = 0; vDy < 2; vDy++) {
          for (int vDx = 0; vDx < 2; vDx++) {
            lPositions[vPtr] = vY + vDy;
            lPositions[cNumPatches + vPtr] = vX + vDx;
            lPositions[2 * cNumPatches + vPtr] = vY + vDy;
            lPositions[3 * cNumPatches + vPtr] = vX + vDx;
            vPtr++;
          }
        }
      }
    }

    vSetInputI32("positions", lPositions);
  } break;
  case cProjectorTypeStep3vl: {
    std::vector<int32_t> lPosData(cNPos);
    for (int vI = 0; vI < cNPos; vI++) {
      lPosData[vI] = vI / cPosW;
    }
    vSetInputI32("pos_h", lPosData);
    for (int vI = 0; vI < cNPos; vI++) {
      lPosData[vI] = vI % cPosW;
    }
    vSetInputI32("pos_w", lPosData);
  } break;
  case cProjectorTypePaddleocr: {
    const int cMergeRatio = vHparams.vNMerge;
    const int cPw = cImageSizeWidth / cPatchSize;
    const int cPh = cImageSizeHeight / cPatchSize;
    std::vector<int> lPositions(cNPos * 4);
    int vPtr = 0;
    // NOTE: same as Qwen-VL, but x and y are swapped
    for (int vY = 0; vY < cPh; vY += cMergeRatio) {
      for (int vDy = 0; vDy < 2; vDy++) {
        for (int vX = 0; vX < cPw; vX += cMergeRatio) {
          for (int vDx = 0; vDx < 2; vDx++) {
            lPositions[vPtr] = vY + vDy;
            lPositions[cNumPatches + vPtr] = vX + vDx;
            lPositions[2 * cNumPatches + vPtr] = vY + vDy;
            lPositions[3 * cNumPatches + vPtr] = vX + vDx;
            vPtr++;
          }
        }
      }
    }

    vSetInputI32("positions", lPositions);
  } break;
  case cProjectorTypeMinimaxM3: {
    const int cNMerge = vHparams.vNMerge;
    const int cGh = cImageSizeHeight / cPatchSize;
    const int cGw = cImageSizeWidth / cPatchSize;
    std::vector<int32_t> lPosH, lPosW;
    lPosH.reserve(cGh * cGw);
    lPosW.reserve(cGh * cGw);
    for (int vBh = 0; vBh < cGh / cNMerge; vBh++)
      for (int vBw = 0; vBw < cGw / cNMerge; vBw++)
        for (int vMh = 0; vMh < cNMerge; vMh++)
          for (int vMw = 0; vMw < cNMerge; vMw++) {
            lPosH.push_back(vBh * cNMerge + vMh);
            lPosW.push_back(vBw * cNMerge + vMw);
          }
    vSetInputI32("minimax_pos_h", lPosH);
    vSetInputI32("minimax_pos_w", lPosW);
  } break;
  case cProjectorTypeDotsOcr: {
    const int cPw = cImageSizeWidth / cPatchSize;
    const int cPh = cImageSizeHeight / cPatchSize;
    const int cNPos = cPh * cPw;
    std::vector<int> lPositions(cNPos * 4);
    int vPtr = 0;

    // flat layout: [h, w, h, w] for each patch
    // patches are in raster order (matching conv2d output)
    for (int vY = 0; vY < cPh; vY++) {
      for (int vX = 0; vX < cPw; vX++) {
        lPositions[vPtr] = vY;
        lPositions[cNPos + vPtr] = vX;
        lPositions[2 * cNPos + vPtr] = vY;
        lPositions[3 * cNPos + vPtr] = vX;
        vPtr++;
      }
    }

    vSetInputI32("positions", lPositions);
  } break;
  case cProjectorTypeQwen25vl:
  case cProjectorTypeEXAONE4Point5:
  case cProjectorTypeYoutuvl: {
    // pw * ph = number of tokens output by ViT after apply patch merger
    // ipw * ipw = number of vision token been processed inside ViT
    const bool cUseWindowAttn =
        (pCtx->vModel.vProjType == cProjectorTypeQwen25vl || pCtx->vModel.vProjType == cProjectorTypeEXAONE4Point5)
            ? vHparams.vNWaPattern > 0
            : !vHparams.vWaLayerIndexes.empty();
    const int cMergeRatio = 2;
    const int cPw = cImageSizeWidth / cPatchSize / cMergeRatio;
    const int cPh = cImageSizeHeight / cPatchSize / cMergeRatio;
    const int cIpw = cImageSizeWidth / cPatchSize;
    const int cIph = cImageSizeHeight / cPatchSize;

    std::vector<int> lIdx(cPh * cPw);
    std::vector<int> lInvIdx(cPh * cPw);

    if (cUseWindowAttn) {
      const int cAttnWindowSize = vHparams.vAttnWindowSize > 0 ? vHparams.vAttnWindowSize : 112;
      const int cGridWindow = cAttnWindowSize / cPatchSize / cMergeRatio;
      int vDst = 0;
      // [num_vision_tokens, num_vision_tokens] attention mask tensor
      std::vector<float> lMask(pow(cIpw * cIph, 2), std::numeric_limits<float>::lowest());
      int vMaskRow = 0;

      for (int vY = 0; vY < cPh; vY += cGridWindow) {
        for (int vX = 0; vX < cPw; vX += cGridWindow) {
          const int cWinH = std::min(cGridWindow, cPh - vY);
          const int cWinW = std::min(cGridWindow, cPw - vX);
          const int cDst0 = vDst;
          // group all tokens belong to the same window togather (to a continue range)
          for (int vDy = 0; vDy < cWinH; vDy++) {
            for (int vDx = 0; vDx < cWinW; vDx++) {
              const int cSrc = (vY + vDy) * cPw + (vX + vDx);
              GGML_ASSERT(cSrc < (int)lIdx.size());
              GGML_ASSERT(vDst < (int)lInvIdx.size());
              lIdx[cSrc] = vDst;
              lInvIdx[vDst] = cSrc;
              vDst++;
            }
          }

          for (int vR = 0; vR < cWinH * cWinW * cMergeRatio * cMergeRatio; vR++) {
            int vRowOffset = vMaskRow * (cIpw * cIph);
            std::fill(lMask.begin() + vRowOffset + (cDst0 * cMergeRatio * cMergeRatio),
                      lMask.begin() + vRowOffset + (vDst * cMergeRatio * cMergeRatio), 0.0);
            vMaskRow++;
          }
        }
      }

      vSetInputI32("window_idx", lIdx);
      vSetInputI32("inv_window_idx", lInvIdx);
      vSetInputF32("window_mask", lMask);
    } else {
      for (int vI = 0; vI < cPh * cPw; vI++) {
        lIdx[vI] = vI;
      }
    }

    const int cMpow = cMergeRatio * cMergeRatio;
    std::vector<int> lPositions(cNPos * 4);

    int vPtr = 0;
    for (int vY = 0; vY < cIph; vY += cMergeRatio) {
      for (int vX = 0; vX < cIpw; vX += cMergeRatio) {
        for (int vDy = 0; vDy < 2; vDy++) {
          for (int vDx = 0; vDx < 2; vDx++) {
            auto vRemap = lIdx[vPtr / cMpow];
            vRemap = (vRemap * cMpow) + (vPtr % cMpow);

            lPositions[vRemap] = vY + vDy;
            lPositions[cNumPatches + vRemap] = vX + vDx;
            lPositions[2 * cNumPatches + vRemap] = vY + vDy;
            lPositions[3 * cNumPatches + vRemap] = vX + vDx;
            vPtr++;
          }
        }
      }
    }

    vSetInputI32("positions", lPositions);
  } break;
  case cProjectorTypeMimovl: {
    const int cMerge = vHparams.vNMerge;    // 2
    const int cMergeUnit = cMerge * cMerge; // 4
    const int cPatch = vHparams.vPatchSize; // 16
    const int cH = cImageSizeHeight / cPatch;
    const int cW = cImageSizeWidth / cPatch;
    const int cNPosFull = cH * cW;
    const int cLlmH = cH / cMerge;
    const int cLlmW = cW / cMerge;
    const int cNUnits = cLlmH * cLlmW; // n_pos / merge_unit

    // Row-major merge-tile-ordered (h, w) positions
    std::vector<int32_t> lPosHRow(cNPosFull);
    std::vector<int32_t> lPosWRow(cNPosFull);
    {
      int vIdx = 0;
      for (int vTy = 0; vTy < cLlmH; vTy++) {
        for (int vTx = 0; vTx < cLlmW; vTx++) {
          for (int vDy = 0; vDy < cMerge; vDy++) {
            for (int vDx = 0; vDx < cMerge; vDx++) {
              lPosHRow[vIdx] = vTy * cMerge + vDy;
              lPosWRow[vIdx] = vTx * cMerge + vDx;
              vIdx++;
            }
          }
        }
      }
    }

    // Col-major merge-unit permutation
    std::vector<float> lIdxCol(cNUnits);
    for (int vR = 0; vR < cLlmH; vR++) {
      for (int vC = 0; vC < cLlmW; vC++) {
        int vURow = vR * cLlmW + vC;
        int vUCol = vC * cLlmH + vR;
        lIdxCol[vUCol] = (float)vURow;
      }
    }

    // Col-mode positions: permute pos_*_row by idx_col
    std::vector<int32_t> lPosHCol(cNPosFull);
    std::vector<int32_t> lPosWCol(cNPosFull);
    for (int vU = 0; vU < cNUnits; vU++) {
      int vSrc = (int)lIdxCol[vU];
      for (int vK = 0; vK < cMergeUnit; vK++) {
        lPosHCol[vU * cMergeUnit + vK] = lPosHRow[vSrc * cMergeUnit + vK];
        lPosWCol[vU * cMergeUnit + vK] = lPosWRow[vSrc * cMergeUnit + vK];
      }
    }

    // Pack into ggml_rope_multi VISION-mode layout. The non-CPU kernels
    // only read slots 0 and 1, so pack h in slot 0, w in slot 1:
    //   positions[0..n_pos)         = h
    //   positions[n_pos..2*n_pos)   = w
    //   positions[2*n_pos..3*n_pos) = 0
    //   positions[3*n_pos..4*n_pos) = 0
    std::vector<int32_t> lPositionsRow(static_cast<size_t>(cNPosFull) * 4, 0);
    std::vector<int32_t> lPositionsCol(static_cast<size_t>(cNPosFull) * 4, 0);
    for (int vI = 0; vI < cNPosFull; vI++) {
      lPositionsRow[0 * cNPosFull + vI] = lPosHRow[vI];
      lPositionsRow[1 * cNPosFull + vI] = lPosWRow[vI];
      lPositionsCol[0 * cNPosFull + vI] = lPosHCol[vI];
      lPositionsCol[1 * cNPosFull + vI] = lPosWCol[vI];
    }

    // Banded 1D sliding-window mask
    const int cWindow = vHparams.vAttnWindowSize;
    GGML_ASSERT(cWindow > 0);
    std::vector<float> lMask(static_cast<size_t>(cNPosFull) * cNPosFull, std::numeric_limits<float>::lowest());
    for (int vQ = 0; vQ < cNPosFull; vQ++) {
      int vLo = std::max(0, vQ - cWindow);
      int vHi = std::min(cNPosFull - 1, vQ + cWindow);
      for (int vK = vLo; vK <= vHi; vK++) {
        lMask[static_cast<size_t>(vQ) * cNPosFull + vK] = 0.0f;
      }
    }

    vSetInputI32("mimovl_positions_row", lPositionsRow);
    vSetInputI32("mimovl_positions_col", lPositionsCol);
    vSetInputF32("mimovl_idx_col", lIdxCol);
    vSetInputF32("mimovl_window_mask", lMask);
  } break;
  case cProjectorTypePixtral:
  case cProjectorTypeKimivl:
  case cProjectorTypeKimik25:
  case cProjectorTypeLightonocr: {
    // set the 2D positions
    int vNPatchesPerCol = cImageSizeWidth / cPatchSize;
    std::vector<int> lPosData(cNPos);
    // dimension H
    for (int vI = 0; vI < cNPos; vI++) {
      lPosData[vI] = vI / vNPatchesPerCol;
    }
    vSetInputI32("pos_h", lPosData);
    // dimension W
    for (int vI = 0; vI < cNPos; vI++) {
      lPosData[vI] = vI % vNPatchesPerCol;
    }
    vSetInputI32("pos_w", lPosData);
  } break;
  case cProjectorTypeGemma3:
  case cProjectorTypeGemma3nv:
  case cProjectorTypeIdefics3:
  case cProjectorTypeInternvl:
  case cProjectorTypeNemotronV2Vl:
  case cProjectorTypeQwen2a:
  case cProjectorTypeQwen3a:
  case cProjectorTypeGlma:
  case cProjectorTypeUltravox:
  case cProjectorTypeLfm2:
  case cProjectorTypeVoxtral:
  case cProjectorTypeMeralion:
  case cProjectorTypeMusicFlamingo:
  case cProjectorTypeJanusPro:
  case cProjectorTypePhi4:
  case cProjectorTypeCogvlm:
  case cProjectorTypeYasa2: {
    // do nothing
  } break;
  case cProjectorTypeHunyuanvl: {
    // Compute the HunyuanVL 2D position embedding on CPU (with the
    // custom sf=(target+0.1)/n_grid bilinear sampling that the
    // reference implementation uses) and upload it to the graph
    // input declared in clip_graph_hunyuanvl::mBuild().
    GGML_ASSERT(vModel.vPositionEmbeddings != nullptr);
    ggml_tensor *vSrcT = vModel.vPositionEmbeddings;
    const int64_t cNEmbd = vSrcT->ne[0];
    const int64_t cNPos = vSrcT->ne[1]; // = n_grid * n_grid
    const int cNGrid = (int)std::lround(std::sqrt((double)cNPos));
    GGML_ASSERT((int64_t)cNGrid * cNGrid == cNPos);
    const int cOutW = cPosW; // pw
    const int cOutH = cPosH; // ph

    // Pull weight to host.
    std::vector<float> lSrc(cNEmbd * cNPos);
    ggml_backend_tensor_get(vSrcT, lSrc.data(), 0, ggml_nbytes(vSrcT));

    // Output layout matches ggml_new_tensor_2d(F32, n_embd, out_h*out_w):
    //   ne[0] = n_embd (fastest), ne[1] = out_h*out_w
    //   dst[(y*out_w + x) * n_embd + c]
    std::vector<float> lDst((size_t)cNEmbd * cOutH * cOutW);

    const float cSx = (float)(cOutW + 0.1f) / (float)cNGrid;
    const float cSy = (float)(cOutH + 0.1f) / (float)cNGrid;

    for (int vY = 0; vY < cOutH; ++vY) {
      // Match ggml_compute_forward_upscale_f32 pixel-center
      // convention (align_corners=False): src_y = (y+0.5)/sy - 0.5.
      const float cFy = ((float)vY + 0.5f) / cSy - 0.5f;
      int vY0 = (int)std::floor(cFy);
      int vY1 = vY0 + 1;
      vY0 = std::clamp(vY0, 0, cNGrid - 1);
      vY1 = std::clamp(vY1, 0, cNGrid - 1);
      float vWy1 = std::clamp(cFy - (float)vY0, 0.0f, 1.0f);
      const float cWy0 = 1.0f - vWy1;
      for (int vX = 0; vX < cOutW; ++vX) {
        const float cFx = ((float)vX + 0.5f) / cSx - 0.5f;
        int vX0 = (int)std::floor(cFx);
        int vX1 = vX0 + 1;
        vX0 = std::clamp(vX0, 0, cNGrid - 1);
        vX1 = std::clamp(vX1, 0, cNGrid - 1);
        float vWx1 = std::clamp(cFx - (float)vX0, 0.0f, 1.0f);
        const float cWx0 = 1.0f - vWx1;

        const float cW00 = cWy0 * cWx0;
        const float cW01 = cWy0 * vWx1;
        const float cW10 = vWy1 * cWx0;
        const float cW11 = vWy1 * vWx1;

        const float *vS00 = &lSrc[((size_t)vY0 * cNGrid + vX0) * cNEmbd];
        const float *vS01 = &lSrc[((size_t)vY0 * cNGrid + vX1) * cNEmbd];
        const float *vS10 = &lSrc[((size_t)vY1 * cNGrid + vX0) * cNEmbd];
        const float *vS11 = &lSrc[((size_t)vY1 * cNGrid + vX1) * cNEmbd];
        float *vD = &lDst[((size_t)vY * cOutW + vX) * cNEmbd];
        for (int vC = 0; vC < cNEmbd; ++vC) {
          vD[vC] = cW00 * vS00[vC] + cW01 * vS01[vC] + cW10 * vS10[vC] + cW11 * vS11[vC];
        }
      }
    }

    vSetInputF32("hunyuanvl_pos_embd", lDst);
  } break;
  case cProjectorTypeMimoAudio: {
    GGML_ASSERT(vImgs.lEntries.size() == 1);
    const int cNFrames = vImgs.lEntries.front().mNx();
    const int cNPos = (cNFrames - 1) / 2 + 1; // matches conv1(s=1)+conv2(s=2) output length

    std::vector<int32_t> lPositions(cNPos);
    for (int vI = 0; vI < cNPos; vI++) {
      lPositions[vI] = vI;
    }
    vSetInputI32("mimo_audio_positions", lPositions);

    const int cWindow = vHparams.vAttnWindowSize;
    GGML_ASSERT(cWindow > 0);

    const float cNegInf = std::numeric_limits<float>::lowest();
    std::vector<float> lFullMask((size_t)cNPos * cNPos);
    std::vector<float> lWindowMask((size_t)cNPos * cNPos);
    for (int vQ = 0; vQ < cNPos; vQ++) {
      for (int vK = 0; vK < cNPos; vK++) {
        const bool cCausalOk = vK <= vQ;
        lFullMask[(size_t)vQ * cNPos + vK] = cCausalOk ? 0.0f : cNegInf;
        lWindowMask[(size_t)vQ * cNPos + vK] = (cCausalOk && (vQ - vK) <= cWindow) ? 0.0f : cNegInf;
      }
    }
    vSetInputF32("mimo_audio_full_mask", lFullMask);
    vSetInputF32("mimo_audio_window_mask", lWindowMask);

    // input_local_transformer: block-diagonal mask + in-group positions
    {
      const int cNPosDs = (cNPos - 2) / 2 + 1; // matches downsample conv (k=2,s=2,p=0)
      const int cGroupSize = vHparams.vAudioLocalGroupSize;
      GGML_ASSERT(cGroupSize > 0);
      const int cNGroups = (cNPosDs + cGroupSize - 1) / cGroupSize;
      const int cNPadded = cNGroups * cGroupSize;

      std::vector<int32_t> lLocalPositions(cNPadded);
      for (int vI = 0; vI < cNPadded; vI++) {
        lLocalPositions[vI] = vI % cGroupSize;
      }
      vSetInputI32("mimo_audio_local_positions", lLocalPositions);

      std::vector<float> lLocalMask((size_t)cNPadded * cNPadded);
      for (int vQ = 0; vQ < cNPadded; vQ++) {
        for (int vK = 0; vK < cNPadded; vK++) {
          const bool cSameGroup = (vQ / cGroupSize) == (vK / cGroupSize);
          lLocalMask[(size_t)vQ * cNPadded + vK] = cSameGroup ? 0.0f : cNegInf;
        }
      }
      vSetInputF32("mimo_audio_local_mask", lLocalMask);
    }
  } break;
  case cProjectorTypeLfm2a: {
    GGML_ASSERT(vImgs.lEntries.size() == 1);
    const auto cNFrames = fClipNOutputTokens(pCtx, &vImgs.lEntries.front());

    auto vDModel = 512;
    auto vSeqLen = cNFrames * 2 - 1;
    std::vector<float> lPosEmb(vDModel * vSeqLen);
    std::vector<double> lInvFreq(vDModel / 2);
    for (size_t vI = 0; vI < lInvFreq.size(); ++vI) {
      lInvFreq[vI] = std::exp(-(std::log(10000.0) / (float)vDModel) * (2.0f * (float)(vI)));
    }
    for (int64_t vPos = 0; vPos < vSeqLen; ++vPos) {
      for (size_t vI = 0; vI < lInvFreq.size(); ++vI) {
        const float cAng = (cNFrames - vPos - 1) * lInvFreq[vI];
        lPosEmb[vPos * vDModel + 2 * vI + 0] = sinf(cAng); // even
        lPosEmb[vPos * vDModel + 2 * vI + 1] = cosf(cAng); // odd
      }
    }
    vSetInputF32("pos_emb", lPosEmb);
  } break;
  case cProjectorTypeParakeet: {
    GGML_ASSERT(vImgs.lEntries.size() == 1);
    struct ggml_tensor *vAttnMask = ggml_graph_get_tensor(vGf, "attn_mask");
    const int cNQ = vAttnMask->ne[1];
    const int cNK = vAttnMask->ne[0];
    const int cNFrames = vImgs.lEntries.front().mNx();
    const int cNTokensReal = (cNFrames + vHparams.vSubsamplingFactor - 1) / vHparams.vSubsamplingFactor;
    const float cMaskValue = -1e30f;

    std::vector<float> lMaskData(cNQ * cNK);
    if (cNK == cNQ) {
      // full attention: mask keys that are padding
      for (int vQ = 0; vQ < cNQ; ++vQ) {
        for (int vK = 0; vK < cNK; ++vK) {
          lMaskData[vQ * cNK + vK] = (vK >= cNTokensReal) ? cMaskValue : 0.0f;
        }
      }
    } else {
      // local attention: mask keys outside the valid window
      const int cAttLeft = cNK / 2;
      for (int vQ = 0; vQ < cNQ; ++vQ) {
        for (int vK = 0; vK < cNK; ++vK) {
          const int cKey = vQ - cAttLeft + vK;
          lMaskData[vQ * cNK + vK] = (cKey >= 0 && cKey < cNTokensReal) ? 0.0f : cMaskValue;
        }
      }
    }
    vSetInputF32(vAttnMask->name, lMaskData);

    // local attention skew mask: zeroes out the probs that were
    // computed for keys outside the valid sliding window.
    if (struct ggml_tensor *vLocalMask = ggml_graph_get_tensor(vGf, "local_mask")) {
      const int cLmK = vLocalMask->ne[0];
      const int cLmQ = vLocalMask->ne[1];
      const int cWindowSize = cLmK - cLmQ + 1;
      std::vector<float> lLmData(cLmQ * cLmK);
      for (int vQ = 0; vQ < cLmQ; ++vQ) {
        for (int vK = 0; vK < cLmK; ++vK) {
          const int cRel = vK - vQ;
          lLmData[vQ * cLmK + vK] = (cRel >= 0 && cRel < cWindowSize) ? 1.0f : 0.0f;
        }
      }
      vSetInputF32(vLocalMask->name, lLmData);
    }

    // Generate rotation frequencies for relative positional encoding.
    {
      const int cNState = vHparams.vNEmbd;
      const int cDHalf = cNState / 2;
      const float cLog10000 = logf(10000.0f);
      std::vector<float> lFreqs(cDHalf);
      for (int vK = 0; vK < cDHalf; ++vK) {
        lFreqs[vK] = expf(-(float(vK * 2) * cLog10000 / float(cNState)));
      }
      vSetInputF32("pos_freqs", lFreqs);
    }

    // Generate relative positional distance values which scaled by
    // the frequency to produce the angles for sin/cos.
    {
      // window_size is only known after graph construction since it depends on
      // n_time from the conv output, so we read it back from the graph tensor.
      struct ggml_tensor *vRelPos = ggml_graph_get_tensor(vGf, "rel_positions");
      const int cWindowSize = vRelPos->ne[1];
      std::vector<float> lPos(cWindowSize);
      // local attention: window is fixed at [att_left, att_right]
      // full attention: window covers the full sequence, centered
      if (ggml_graph_get_tensor(vGf, "local_mask")) {
        const int cAttLeft = cWindowSize / 2;
        for (int vT = 0; vT < cWindowSize; ++vT) {
          lPos[vT] = float(cAttLeft - vT);
        }
      } else {
        const int cNTime = (cWindowSize + 1) / 2;
        for (int vT = 0; vT < cWindowSize; ++vT) {
          lPos[vT] = float(cNTime - 1 - vT);
        }
      }
      vSetInputF32(vRelPos->name, lPos);
    }
  } break;
  case cProjectorTypeGraniteSpeech: {
    const int cContextSize = pCtx->vModel.vHparams.vAudioChunkSize;
    const int cMaxPosEmb = pCtx->vModel.vHparams.vAudioMaxPosEmb;

    std::vector<int32_t> lDists(cContextSize * cContextSize);
    for (int vI = 0; vI < cContextSize; vI++) {
      for (int vJ = 0; vJ < cContextSize; vJ++) {
        int vD = vI - vJ;
        if (vD < -cContextSize)
          vD = -cContextSize;
        if (vD > cContextSize)
          vD = cContextSize;
        lDists[vI * cContextSize + vJ] = vD + cMaxPosEmb;
      }
    }
    vSetInputI32("attn_dists", lDists);

    const int cNFrames = cImageSizeWidth;
    const int cRemainder = cNFrames % cContextSize;
    if (cRemainder > 0) {
      const int cNumBlocks = (cNFrames + cContextSize - 1) / cContextSize;
      std::vector<float> lMask(cContextSize * cContextSize * cNumBlocks, 0.0f);
      const float cNegInf = -INFINITY;
      const int cLastBlockOffset = (cNumBlocks - 1) * cContextSize * cContextSize;
      for (int vQ = 0; vQ < cContextSize; vQ++) {
        for (int vK = 0; vK < cContextSize; vK++) {
          if (vQ >= cRemainder || vK >= cRemainder) {
            lMask[cLastBlockOffset + vQ * cContextSize + vK] = cNegInf;
          }
        }
      }
      vSetInputF32("attn_mask", lMask);
    }
  } break;
  case cProjectorTypeGranite4Vision: {
    // Granite Vision 4.1 uses precomputed permutation index
    // tensors to express the _win / _unwin / spatial sampling
    // reshapes as ggml_get_rows gathers. The names are set
    // by g4v_gather() in models/granite4-vision.cpp.
    const int cPatchSize = vModel.vHparams.vPatchSize;
    const int cImageSide = vImgs.lEntries.front().mNx() / cPatchSize;
    const int cWindowSide = vHparams.vDownsampleWindowSide;
    const int cQuerySide = vHparams.vDownsampleQuerySide;
    const int cN = cImageSide / cWindowSide;
    const int cNewSide = cN * cQuerySide;

    // Builds the raster→window permutation indices for a
    // (side, side) grid split into (n × n) windows of (win × win)
    // tokens each.  dst[w * win*win + p] = source raster index.
    auto vMakeWinIdx = [](int pSide, int pWin) {
      const int cNn = pSide / pWin;
      std::vector<int32_t> lIdx(static_cast<size_t>(pSide) * pSide);
      for (int vWy = 0; vWy < cNn; ++vWy) {
        for (int vWx = 0; vWx < cNn; ++vWx) {
          for (int vIy = 0; vIy < pWin; ++vIy) {
            for (int vIx = 0; vIx < pWin; ++vIx) {
              const int cW = vWy * cNn + vWx;
              const int cP = vIy * pWin + vIx;
              const int cY = vWy * pWin + vIy;
              const int cX = vWx * pWin + vIx;
              lIdx[static_cast<size_t>(cW) * (pWin * pWin) + cP] = cY * pSide + cX;
            }
          }
        }
      }
      return lIdx;
    };

    auto vMakeUnwinIdx = [&](int pSide, int pWin) {
      const std::vector<int32_t> lFwd = vMakeWinIdx(pSide, pWin);
      std::vector<int32_t> lInv(lFwd.size());
      for (size_t vI = 0; vI < lFwd.size(); ++vI) {
        lInv[lFwd[vI]] = static_cast<int32_t>(vI);
      }
      return lInv;
    };

    auto vMakeSpatialIdx = [](int pSide, int pOffset) {
      const int cOffY = (pOffset >> 1) & 1;
      const int cOffX = pOffset & 1;
      const int cNewS = pSide / 2;
      std::vector<int32_t> lIdx(static_cast<size_t>(cNewS) * cNewS);
      for (int vY = 0; vY < cNewS; ++vY) {
        for (int vX = 0; vX < cNewS; ++vX) {
          lIdx[vY * cNewS + vX] = (vY * 2 + cOffY) * pSide + (vX * 2 + cOffX);
        }
      }
      return lIdx;
    };

    auto vUpload = [&](const std::string &pName, const std::vector<int32_t> &pIdx) {
      ggml_tensor *vT = ggml_graph_get_tensor(vGf, pName.c_str());
      GGML_ASSERT(vT);
      ggml_backend_tensor_set(vT, pIdx.data(), 0, pIdx.size() * sizeof(int32_t));
    };

    // Stage 1b only uses block 0's permutations; future stages
    // will upload all blocks.
    for (size_t vBid = 0; vBid < vHparams.lFeatureLayers.size(); ++vBid) {
      const std::string cPrefix = "g4v_blk" + std::to_string(vBid) + "_";
      vUpload(cPrefix + "win_idx", vMakeWinIdx(cImageSide, cWindowSide));
      vUpload(cPrefix + "qwin_idx", vMakeWinIdx(cNewSide, cQuerySide));
      vUpload(cPrefix + "unwin_idx", vMakeUnwinIdx(cNewSide, cQuerySide));
      const auto cSpatialOffset = vHparams.lProjSpatialOffsets[vBid];
      if (cSpatialOffset >= 0) {
        vUpload(cPrefix + "spatial_idx", vMakeSpatialIdx(cImageSide, cSpatialOffset));
      }
    }
  } break;
  default:
    GGML_ABORT("Unknown projector type");
  }

  // ggml_backend_cpu_set_n_threads(pCtx->vBackendCpu, pNThreads);
  ggml_backend_dev_t vDev = ggml_backend_get_device(pCtx->vBackendCpu);
  ggml_backend_reg_t vReg = vDev ? ggml_backend_dev_backend_reg(vDev) : nullptr;
  if (vReg) {
    auto vGgmlBackendSetNThreadsFn =
        (ggml_backend_set_n_threads_t)ggml_backend_reg_get_proc_address(vReg, "ggml_backend_set_n_threads");
    if (vGgmlBackendSetNThreadsFn) {
      vGgmlBackendSetNThreadsFn(pCtx->vBackendCpu, pNThreads);
    }
  }

  auto vStatus = ggml_backend_sched_graph_compute(pCtx->vSched.get(), vGf);
  if (vStatus != GGML_STATUS_SUCCESS) {
    LOG_ERR("%s: ggml_backend_sched_graph_compute failed with error %d\n", __func__, vStatus);
    return false;
  }

  // the last node is the embedding tensor
  ggml_tensor *vEmbeddings = ggml_graph_node(vGf, -1);

  // sanity check (assuming that all images in batch have the same number of tokens, so we only check the first one)
  const int cNTokensOut = vEmbeddings->ne[1];
  const int cExpectedNTokensOut = fClipNOutputTokens(pCtx, &vImgs.lEntries[0]);
  if (cNTokensOut != cExpectedNTokensOut) {
    LOG_ERR("%s: expected output %d tokens, got %d\n", __func__, cExpectedNTokensOut, cNTokensOut);
    GGML_ABORT("Invalid number of output tokens");
  }

  LOG_DBG("%s: output embedding shape [%d, %d, %d]\n", __func__, (int)vEmbeddings->ne[0], (int)vEmbeddings->ne[1],
          (int)vEmbeddings->ne[2]);

  // copy output to user buffer if provided
  // if output is empty, skip the copy
  if (!pOutBatchEmbd.empty()) {
    if (pOutBatchEmbd.size() != (size_t)ggml_nelements(vEmbeddings)) {
      LOG_ERR("%s: output buffer has %zu elements but expected %zu\n", __func__, pOutBatchEmbd.size(),
              (size_t)ggml_nelements(vEmbeddings));
      GGML_ABORT("Output buffer size mismatch");
    }
    ggml_backend_tensor_get(vEmbeddings, pOutBatchEmbd.data(), 0, ggml_nbytes(vEmbeddings));
  } else {
    LOG_WRN("%s: output buffer is empty, skipping copy\n", __func__);
  }

  // Debug: dump final embeddings if MTMD_DEBUG_EMBEDDINGS is set
  if (pCtx->vDebugOutputEmbeddings) {
    const int64_t cNEmbd = vEmbeddings->ne[0];
    const int64_t cNTokens = vEmbeddings->ne[1];
    std::vector<float> lEmbData(ggml_nelements(vEmbeddings));
    ggml_backend_tensor_get(vEmbeddings, lEmbData.data(), 0, ggml_nbytes(vEmbeddings));

    LOG_INF("\n=== MTMD_DEBUG_EMBEDDINGS ===\n");
    LOG_INF("Shape: [%lld, %lld]\n", (long long)cNEmbd, (long long)cNTokens);

    // Print first few values of first token
    LOG_INF("Token 0 (first 16 values): ");
    for (int vI = 0; vI < std::min((int64_t)16, cNEmbd); vI++) {
      LOG_INF("%.6f ", lEmbData[vI]);
    }
    LOG_INF("\n");

    // Print last few values of first token
    if (cNEmbd > 16) {
      LOG_INF("Token 0 (last 16 values):  ");
      for (int64_t vI = cNEmbd - 16; vI < cNEmbd; vI++) {
        LOG_INF("%.6f ", lEmbData[vI]);
      }
      LOG_INF("\n");
    }

    // Compute and print statistics
    float vSum = 0.0f, vSumSq = 0.0f, vMinVal = lEmbData[0], vMaxVal = lEmbData[0];
    for (size_t vI = 0; vI < lEmbData.size(); vI++) {
      vSum += lEmbData[vI];
      vSumSq += lEmbData[vI] * lEmbData[vI];
      vMinVal = std::min(vMinVal, lEmbData[vI]);
      vMaxVal = std::max(vMaxVal, lEmbData[vI]);
    }
    float vMean = vSum / lEmbData.size();
    float vVariance = (vSumSq / lEmbData.size()) - (vMean * vMean);
    LOG_INF("Stats: mean=%.6f, std=%.6f, min=%.6f, max=%.6f, sum=%.6f\n", vMean, sqrtf(vVariance), vMinVal, vMaxVal,
            vSum);
    LOG_INF("=== END MTMD_DEBUG_EMBEDDINGS ===\n\n");
  }

  return true;
}

int fClipNMmprojEmbd(const struct clip_ctx *pCtx) {
  switch (pCtx->vModel.vProjType) {
  case cProjectorTypePhi4:
  case cProjectorTypePixtral:
  case cProjectorTypeLightonocr:
  case cProjectorTypeDotsOcr:
    return pCtx->vModel.vMm2W->ne[1];
  case cProjectorTypeMINICPMV4Point6:
    return pCtx->vModel.vMmFfnDownW->ne[1];
  case cProjectorTypeMinimaxM3:
    return pCtx->vModel.vMmMergerFc2B->ne[0];
  case cProjectorTypeQwen2vl:
  case cProjectorTypeQwen25vl:
  case cProjectorTypeEXAONE4Point5:
  case cProjectorTypeJanusPro:
  case cProjectorTypeYoutuvl:
    return pCtx->vModel.vMm1B->ne[0];
  case cProjectorTypeQwen3vl:
    // main path + deepstack paths
    return pCtx->vModel.vMm1B->ne[0] * (1 + pCtx->vModel.vNDeepstackLayers);
  case cProjectorTypeMimovl:
    return pCtx->vModel.vMm1W->ne[1];
  case cProjectorTypeStep3vl:
    return pCtx->vModel.vMmModelProj->ne[1];
  case cProjectorTypeGemma3:
  case cProjectorTypeGemma3nv:
    return pCtx->vModel.vMmInputProjW->ne[0];
  case cProjectorTypeIdefics3:
    return pCtx->vModel.vMmFcW->ne[1];
  case cProjectorTypeUltravox:
  case cProjectorTypeVoxtral:
  case cProjectorTypeMusicFlamingo:
    return pCtx->vModel.vMm2W->ne[1];
  case cProjectorTypeMeralion:
    return pCtx->vModel.vMm3W->ne[1]; // out_proj output dim
  case cProjectorTypeInternvl:
  case cProjectorTypeNemotronV2Vl:
    return pCtx->vModel.vMm3W->ne[1];
  case cProjectorTypeQwen2a:
    return pCtx->vModel.vMmFcW->ne[1];
  case cProjectorTypeQwen3a:
    return pCtx->vModel.vMm2W->ne[1];
  case cProjectorTypeGlma:
  case cProjectorTypeLfm2:
  case cProjectorTypeKimivl:
  case cProjectorTypePaddleocr:
  case cProjectorTypeKimik25:
  case cProjectorTypeYasa2:
    return pCtx->vModel.vMm2W->ne[1];
  case cProjectorTypeHunyuanvl:
    return pCtx->vModel.vMmModelProj->ne[1];
  case cProjectorTypeCogvlm:
    return pCtx->vModel.vMm4hToHW->ne[1];
  case cProjectorTypeLfm2a:
    return pCtx->vModel.vPositionEmbeddings->ne[0];
  case cProjectorTypeGraniteSpeech:
    return pCtx->vModel.lQfProjBlocks[0].vQfProjLinearW->ne[1];
  case cProjectorTypeGranite4Vision:
    return pCtx->vModel.lQfProjBlocks.size() * pCtx->vModel.vHparams.vProjectionDim;
  case cProjectorTypeGlm4v:
  case cProjectorTypeGlm5v:
    return pCtx->vModel.vMmFfnDownW->ne[1];
  case cProjectorTypeMimoAudio:
    return pCtx->vModel.vMm2W->ne[1];
  case cProjectorTypeParakeet:
    return pCtx->vModel.vMm1W->ne[1];
  default:
    GGML_ABORT("Unknown projector type");
  }
}

bool fClipHasVisionEncoder(const struct clip_ctx *pCtx) { return pCtx->vModel.vModality == cClipModalityVision; }

bool fClipHasAudioEncoder(const struct clip_ctx *pCtx) { return pCtx->vModel.vModality == cClipModalityAudio; }

bool fClipSupportBatch(const struct clip_ctx *pCtx) { return pCtx->vSupportBatch; }

// TODO @ngxson : this is no longer correct with mtmd_batch API
// this was only meant to be used by qwen-vl-based models, to fuse 2 input images into one (qwen-vl video support)
// this logic should be refactored in near future to distinctly handle "merge frames" and "batching"
int fClipModelNTemporalMerge(const struct clip_ctx *pCtx) {
  switch (pCtx->vModel.vProjType) {
  case cProjectorTypeQwen2vl:
  case cProjectorTypeQwen25vl:
  case cProjectorTypeQwen3vl:
    return 2;
  default:
    return 1;
  }
}

//
// API used internally with mtmd
//

projector_type fClipGetProjectorType(const struct clip_ctx *pCtx) { return pCtx->vModel.vProjType; }

const clip_hparams *fClipGetHparams(const struct clip_ctx *pCtx) { return &pCtx->vModel.vHparams; }

std::map<ggml_backend_dev_t, size_t> fClipGetMemUsage(const struct clip_ctx *pCtx) {
  std::map<ggml_backend_dev_t, size_t> dResult = pCtx->dMemUsage;
  for (auto &[vDev, vSize] : pCtx->dMemCompute) {
    dResult[vDev] += vSize;
  }
  return dResult;
}

//
// API for debugging
//

void fClipSetDebugOutputEmbeddings(clip_ctx *pCtx, bool pEnable) { pCtx->vDebugOutputEmbeddings = pEnable; }
