#pragma once

#include "ggml.h"
#include "ggml-cpp.h"
#include "clip.h"
#include "clip-impl.h"
#include "clip-model.h"

#include <vector>
#include <functional>

#define DEFAULT_INTERPOLATION_MODE (GGML_SCALE_MODE_BILINEAR | GGML_SCALE_FLAG_ANTIALIAS)

struct build_vit_opts {
  ggml_tensor *vAttnMask = nullptr;
  // TODO @ngxson : merge attn_mask and attn_mask_layers into one call
  std::vector<ggml_tensor *> lAttnMaskLayers; // one per layer

  // hook at layer output embeddings
  std::function<void(ggml_tensor *pCur, int pIl)> vCallbackLayerOut = nullptr;

  // whether to skip the automatic post-layernorm (model.vPostLnW) applied at the end
  bool vSkipPostLn = false;
};

struct clip_graph {
  const clip_model &vModel;
  const clip_hparams &vHparams;
  projector_type vProjType;

  // set by fClipGetGraphBuilder() before mBuild() runs; read it with mGetImg()
  const clip_image_f32_batch *vImgBatch = nullptr;

  const int cPatchSize;
  const int cNPatchesX;
  const int cNPatchesY;
  const int cNPatches;
  const int cNEmbd;
  const int cNHead;
  const int cNHeadKv;
  const int cDHead;
  const int cNLayer;
  const int cNMmprojEmbd;
  const float cEps;
  float vKqScale; // TODO: maybe move this to hparams
  const clip_flash_attn_type cFlashAttnType;

  // TODO [QWEN_VIDEO]: improve this in the future
  int vNBatch = 1;

  ggml_context_ptr vCtx0Ptr;
  ggml_context *vCtx0;
  ggml_cgraph *vGf;

  clip_graph(clip_ctx *pCtx, const clip_image_f32 &pImg);

  virtual ~clip_graph() = default;
  virtual ggml_cgraph *mBuild() = 0;

  // wrapper around ggml_mul_mat, allow hooking (e.g. LoRA, clamping) depending on the model
  // tensor w should be the weight matrix, and tensor x should be the input
  virtual ggml_tensor *mBuildMm(ggml_tensor *pW, ggml_tensor *pX) const;
  // TODO: build_mm(w, b, x) to support bias

  virtual bool mSupportBatch() const { return false; }

  //
  // utility functions
  //
  void mCb(ggml_tensor *pCur0, const char *pName, int pIl) const;

  const clip_image_f32 &mGetImg(size_t pIdx) const {
    GGML_ASSERT(vImgBatch);
    GGML_ASSERT(pIdx < vImgBatch->lEntries.size());
    return vImgBatch->lEntries[pIdx];
  }

  // siglip2 naflex
  ggml_tensor *mResizePositionEmbeddings(uint32_t pInterpolationMode = DEFAULT_INTERPOLATION_MODE);

  // build vision transformer (ViT) cgraph
  // this function should cover most of the models
  // if your model has specific features, you should probably duplicate this function
  ggml_tensor *mBuildVit(ggml_tensor *pInp, int64_t pNPos, norm_type pNormT, ffn_op_type pFfnT,
                         ggml_tensor *pLearnedPosEmbd,
                         std::function<ggml_tensor *(ggml_tensor *, const clip_layer &)> pAddPos,
                         const build_vit_opts &pOpts = {});

  // build the input after conv2d (inp_raw --> patches)
  // returns tensor with shape [n_embd, n_patches]
  ggml_tensor *mBuildInp();

  ggml_tensor *mBuildInpRaw(int pChannels = 3);

  ggml_tensor *mBuildNorm(ggml_tensor *pCur, ggml_tensor *pMw, ggml_tensor *pMb, norm_type pType, float pNormEps,
                          int pIl) const;

  ggml_tensor *mBuildFfn(ggml_tensor *pCur, ggml_tensor *pUp, ggml_tensor *pUpB, ggml_tensor *pGate,
                         ggml_tensor *pGateB, ggml_tensor *pDown, ggml_tensor *pDownB, ffn_op_type pTypeOp,
                         int pIl) const;

  ggml_tensor *mBuildAttn(ggml_tensor *pWo, ggml_tensor *pWoB, ggml_tensor *pQCur, ggml_tensor *pKCur,
                          ggml_tensor *pVCur, ggml_tensor *pKqMask, float pKqScale, int pIl,
                          ggml_tensor *pSinks = nullptr) const;

  // implementation of the 2D RoPE without adding a new op in ggml
  // this is not efficient (use double the memory), but works on all backends
  // TODO: there was a more efficient which relies on ggml_view and ggml_rope_ext_inplace, but the rope inplace does not
  // work well with non-contiguous tensors ; we should fix that and revert back to the original implementation
  ggml_tensor *mBuildRope2d(ggml_context *pCtx0, ggml_tensor *pCur,
                            ggml_tensor *pPosA, // first half
                            ggml_tensor *pPosB, // second half
                            const float pFreqBase, const bool pInterleaveFreq);

  // aka pixel_shuffle / pixel_unshuffle / patch_merger (Kimi-VL)
  // support dynamic resolution
  ggml_tensor *mBuildPatchMergePermute(ggml_tensor *pCur, int pScaleFactor);

  // Generic function to stack frames for audio processing
  // Abstracts out the StackAudioFrames logic used by ultravox
  ggml_tensor *mBuildStack(ggml_tensor *pCur, int32_t pStackFactor, int32_t pNEmbed);
};
