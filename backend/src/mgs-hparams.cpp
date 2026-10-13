#include "mgs-hparams.h"

#include "ggml.h"

#include <algorithm>
#include <cassert>

void mgs_hparams::mSetSwaPattern(uint32_t pNPattern, bool pDenseFirst) {
  if (pDenseFirst) {
    for (uint32_t vIl = 0; vIl < mNLayer(); ++vIl) {
      aIsSwaImpl[vIl] = pNPattern == 0 || (vIl % pNPattern != 0);
    }
  } else {
    for (uint32_t vIl = 0; vIl < mNLayer(); ++vIl) {
      aIsSwaImpl[vIl] = pNPattern == 0 || (vIl % pNPattern < (pNPattern - 1));
    }
  }

  for (uint32_t vIl = mNLayer(); vIl < vNLayerAll; ++vIl) {
    aIsSwaImpl[vIl] = false;
  }
}

void mgs_hparams::mSetRecrPattern(uint32_t pNPattern, bool pDenseFirst) {
  if (pDenseFirst) {
    for (uint32_t vIl = 0; vIl < mNLayer(); ++vIl) {
      aIsRecrImpl[vIl] = pNPattern == 0 || (vIl % pNPattern != 0);
    }
  } else {
    for (uint32_t vIl = 0; vIl < mNLayer(); ++vIl) {
      aIsRecrImpl[vIl] = pNPattern == 0 || (vIl % pNPattern < (pNPattern - 1));
    }
  }

  for (uint32_t vIl = mNLayer(); vIl < vNLayerAll; ++vIl) {
    aIsRecrImpl[vIl] = false;
  }
}

bool mgs_hparams::mIsSwaAny() const {
  for (uint32_t vIl = 0; vIl < vNLayerAll; ++vIl) {
    if (aIsSwaImpl[vIl]) {
      return true;
    }
  }

  return false;
}

uint32_t mgs_hparams::mNHead(uint32_t pIl) const {
  if (pIl < vNLayerAll) {
    return aNHeadArr[pIl];
  }

  GGML_ABORT("fatal error");
}

uint32_t mgs_hparams::mNHeadKv(uint32_t pIl) const {
  if (pIl < vNLayerAll) {
    return aNHeadKvArr[pIl];
  }

  GGML_ABORT("fatal error");
}

uint32_t mgs_hparams::mNFf(uint32_t pIl) const {
  if (pIl < vNLayerAll) {
    return aNFfArr[pIl];
  }

  GGML_ABORT("fatal error");
}

uint32_t mgs_hparams::mNGqa(uint32_t pIl) const {
  const uint32_t cNHead = this->mNHead(pIl);
  const uint32_t cNHeadKv = this->mNHeadKv(pIl);

  if (cNHeadKv == 0) {
    return 0;
  }

  return cNHead / cNHeadKv;
}

uint32_t mgs_hparams::mNRot(uint32_t pIl) const {
  if (pIl < vNLayerAll) {
    return mIsSwa(pIl) ? vNRotSwa : vNRotFull;
  }

  GGML_ABORT("fatal error");
}

uint32_t mgs_hparams::mNEmbdInp() const {
  if (vNEmbdInpImpl > 0) {
    return vNEmbdInpImpl;
  }

  uint32_t vNEmbdInp = vNEmbd;

  if (vNDeepstackLayers > 0) {
    vNEmbdInp += vNEmbd * vNDeepstackLayers;
  }

  return vNEmbdInp;
}

uint32_t mgs_hparams::mNEmbdInpEnc() const { return vNEmbdInpEncImpl > 0 ? vNEmbdInpEncImpl : mNEmbdInp(); }

uint32_t mgs_hparams::mNEmbdOut() const { return vNEmbdOutImpl > 0 ? vNEmbdOutImpl : vNEmbd; }

uint32_t mgs_hparams::mNEmbdHeadK(uint32_t pIl) const {
  if (pIl < vNLayerAll) {
    return mIsSwa(pIl) ? vNEmbdHeadKSwa : vNEmbdHeadKFull;
  }

  GGML_ABORT("fatal error");
}

uint32_t mgs_hparams::mNEmbdHeadV(uint32_t pIl) const {
  if (pIl < vNLayerAll) {
    return mIsSwa(pIl) ? vNEmbdHeadVSwa : vNEmbdHeadVFull;
  }

  GGML_ABORT("fatal error");
}

uint32_t mgs_hparams::mNEmbdKGqa(uint32_t pIl) const {
  const uint32_t cNHeadKv = this->mNHeadKv(pIl);

  return mNEmbdHeadK(pIl) * cNHeadKv;
}

uint32_t mgs_hparams::mNEmbdVGqa(uint32_t pIl) const {
  const uint32_t cNHeadKv = this->mNHeadKv(pIl);

  return mNEmbdHeadV(pIl) * cNHeadKv;
}

bool mgs_hparams::mIsNEmbdKGqaVariable() const {
  const uint32_t cVal = mNEmbdKGqa();
  for (uint32_t vIl = 0; vIl < vNLayerAll; ++vIl) {
    if (cVal != mNEmbdKGqa(vIl)) {
      return true;
    }
  }

  return false;
}

bool mgs_hparams::mIsNEmbdVGqaVariable() const {
  const uint32_t cVal = mNEmbdVGqa();
  for (uint32_t vIl = 0; vIl < vNLayerAll; ++vIl) {
    if (cVal != mNEmbdVGqa(vIl)) {
      return true;
    }
  }

  return false;
}

uint32_t mgs_hparams::mNEmbdKGqaMax() const {
  uint32_t vVal = mNEmbdKGqa();
  for (uint32_t vIl = 0; vIl < vNLayerAll; ++vIl) {
    vVal = std::max(vVal, mNEmbdKGqa(vIl));
  }

  return vVal;
}

uint32_t mgs_hparams::mNEmbdVGqaMax() const {
  uint32_t vVal = mNEmbdVGqa();
  for (uint32_t vIl = 0; vIl < vNLayerAll; ++vIl) {
    vVal = std::max(vVal, mNEmbdVGqa(vIl));
  }

  return vVal;
}

uint32_t mgs_hparams::mNEmbdR() const {
  if (vWkvHeadSize != 0) {
    // for RWKV models
    return vTokenShiftCount * vNEmbd;
  }

  if (vNShortconvLCache != 0) {
    // for LFM2 models
    return vNEmbd * (vNShortconvLCache - 1);
  }

  if (vNEmbdHeadKda != 0) {
    // for Kimi KDA layers
    // Conv state for Q, K, V: 3 * (d_conv - 1) * n_head * head_dim
    const uint32_t cDInner = mNHead() * vNEmbdHeadKda; // 32 * 128 = 4096
    return 3 * (vSsmDConv > 0 ? vSsmDConv - 1 : 3) * cDInner;
  }

  // TODO: maybe support other convolution strides than 1
  // NOTE: since the first column of the conv_state is shifted out each time, it's not actually needed
  // Corresponds to Mamba's conv_states size
  return (vSsmDConv > 0 ? vSsmDConv - 1 : 0) * (vSsmDInner + 2 * vSsmNGroup * vSsmDState);
}

uint32_t mgs_hparams::mNEmbdS() const {
  if (vWkvHeadSize != 0) {
    // corresponds to RWKV's wkv_states size
    return vNEmbd * vWkvHeadSize;
  }

  if (vNEmbdHeadKda != 0) {
    // for Kimi KDA layers
    // Full recurrent state: head_dim * head_dim * n_head
    // h tensor shape for delta attention: [head_dim, head_dim, n_head]
    return vNEmbdHeadKda * vNEmbdHeadKda * mNHead(); // 128 * 128 * 32 = 524288
  }

  // corresponds to Mamba's ssm_states size
  return vSsmDState * vSsmDInner;
}

bool mgs_hparams::mIsRecr(uint32_t pIl) const {
  if (pIl < vNLayerAll) {
    return aIsRecrImpl[pIl];
  }

  GGML_ABORT("%s: il (%u) out of bounds (n_layer_all: %u)\n", __func__, pIl, vNLayerAll);
}

uint32_t mgs_hparams::mPleConvState() const {
  if (vPleNHeads == 0 || vPleConvKernel == 0) {
    return 0;
  }

  // dilation equals the n-gram size, matching the reference module
  return (vPleConvKernel - 1) * vPleNgramSize * vDsv4HcMult * vNEmbd;
}

bool mgs_hparams::mIsPle(uint32_t pIl) const {
  if (pIl < vNLayerAll) {
    return sIsPleImpl[pIl];
  }

  GGML_ABORT("%s: il (%u) out of bounds (n_layer_all: %u)\n", __func__, pIl, vNLayerAll);
}

uint32_t mgs_hparams::mNPosPerEmbd() const {
  return vRopeType == MGS_ROPE_TYPE_MROPE || vRopeType == MGS_ROPE_TYPE_IMROPE ? 4 : 1;
}

bool mgs_hparams::mIsSwa(uint32_t pIl) const {
  if (pIl < vNLayerAll) {
    return aIsSwaImpl[pIl];
  }

  GGML_ABORT("%s: il (%u) out of bounds (n_layer_all: %u)\n", __func__, pIl, vNLayerAll);
}

bool mgs_hparams::mIsMla() const {
  assert((vNEmbdHeadKMlaImpl == 0 && vNEmbdHeadVMlaImpl == 0) || (vNEmbdHeadKMlaImpl != 0 && vNEmbdHeadVMlaImpl != 0));

  return vNEmbdHeadKMlaImpl != 0 && vNEmbdHeadVMlaImpl != 0;
}

bool mgs_hparams::mIsIndexerFull(uint32_t pIl) const {
  if (pIl < mNLayer()) {
    return aIsIndexerFullImpl[pIl];
  }

  GGML_ABORT("%s: il (%u) out of bounds (n_layer: %u)\n", __func__, pIl, mNLayer());
}

uint32_t mgs_hparams::mNEmbdHeadKMla() const { return mIsMla() ? vNEmbdHeadKMlaImpl : mNEmbdHeadK(); }

uint32_t mgs_hparams::mNEmbdHeadVMla() const { return mIsMla() ? vNEmbdHeadVMlaImpl : mNEmbdHeadV(); }

bool mgs_hparams::mHasKv(uint32_t pIl) const {
  if (vNLayerKvFromStart >= 0) {
    if (pIl < (uint32_t)vNLayerKvFromStart) {
      return true;
    }

    return false;
  }

  // by default, all layers have kv
  return true;
}

uint32_t mgs_hparams::mNLayer() const { return vNLayerAll - vNLayerNextn; }

bool mgs_hparams::mUseMrope() const { return aRopeSections[0] > 0 && aRopeSections[1] > 0; }
