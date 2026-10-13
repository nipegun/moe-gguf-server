#include "models.h"

// Helpers for MobileNetV5 Blocks
// RMS Norm 2D - normalizes over channels for each spatial position
ggml_tensor *clip_graph_mobilenetv5::mRmsNorm2d(ggml_tensor *pInp, ggml_tensor *pWeight, float pEps) {
  // inp: [W, H, C, B]

  ggml_tensor *vCur = ggml_permute(vCtx0, pInp, 2, 1, 0, 3);
  vCur = ggml_cont(vCtx0, vCur);
  vCur = ggml_rms_norm(vCtx0, vCur, pEps);

  if (pWeight) {
    vCur = ggml_mul(vCtx0, vCur, pWeight);
  }

  vCur = ggml_permute(vCtx0, vCur, 2, 1, 0, 3);
  vCur = ggml_cont(vCtx0, vCur);

  return vCur;
}

// Conv2dSame padding - asymmetric SAME padding like PyTorch/TF
ggml_tensor *clip_graph_mobilenetv5::mPadSame2d(ggml_tensor *pInp, int pKernelH, int pKernelW, int pStrideH,
                                                int pStrideW, int pDilationH, int pDilationW) {
  const int64_t cIh = pInp->ne[1]; // height
  const int64_t cIw = pInp->ne[0]; // width

  // Calculate output size (ceil division)
  const int64_t cOh = (cIh + pStrideH - 1) / pStrideH;
  const int64_t cOw = (cIw + pStrideW - 1) / pStrideW;

  // Calculate padding needed
  const int64_t cPadH = std::max((int64_t)0, (cOh - 1) * pStrideH + (pKernelH - 1) * pDilationH + 1 - cIh);
  const int64_t cPadW = std::max((int64_t)0, (cOw - 1) * pStrideW + (pKernelW - 1) * pDilationW + 1 - cIw);

  // Split padding asymmetrically
  const int cPadHTop = cPadH / 2;
  const int cPadHBottom = cPadH - cPadHTop;
  const int cPadWLeft = cPadW / 2;
  const int cPadWRight = cPadW - cPadWLeft;

  // Apply padding if needed
  // ggml_pad_ext: (ctx, tensor, lp0, rp0, lp1, rp1, lp2, rp2, lp3, rp3)
  // For [W, H, C, B]: p0=width, p1=height, p2=channels, p3=batch
  if (cPadH > 0 || cPadW > 0) {
    pInp = ggml_pad_ext(vCtx0, pInp, cPadWLeft, cPadWRight, // width padding (dim 0)
                        cPadHTop, cPadHBottom,              // height padding (dim 1)
                        0, 0,                               // no channel padding (dim 2)
                        0, 0);                              // no batch padding (dim 3)
  }

  return pInp;
}

// Edge Residual Block (Stage 0)
ggml_tensor *clip_graph_mobilenetv5::mBuildEdgeResidual(ggml_tensor *pInp, const mobilenetv5_block &pBlock,
                                                        int pStride) {
  ggml_tensor *vCur = pInp;

  // 1. Expansion Conv (3x3)
  if (pStride == 2) {
    // Case: Downsampling (Block 0)
    // Replicates Conv2dSame(kernel=3, stride=2)
    vCur = mPadSame2d(vCur, 3, 3, pStride, pStride);
    vCur = ggml_conv_2d_direct(vCtx0, pBlock.vS0ConvExpW, vCur, pStride, pStride, 0, 0, 1, 1);
  } else {
    // Case: Normal 3x3 Block (Block 1, 2)
    // Replicates Conv2d(kernel=3, stride=1, padding=1)
    vCur = ggml_conv_2d_direct(vCtx0, pBlock.vS0ConvExpW, vCur, pStride, pStride, 1, 1, 1, 1);
  }

  // BN + Activation
  if (pBlock.vS0Bn1W)
    vCur = mRmsNorm2d(vCur, pBlock.vS0Bn1W);
  vCur = ggml_gelu(vCtx0, vCur);

  // 2. Pointwise Linear Conv (1x1)
  // 1x1 Convs usually have padding=0 and stride=1
  vCur = ggml_conv_2d_direct(vCtx0, pBlock.vS0ConvPwlW, vCur, 1, 1, 0, 0, 1, 1);
  if (pBlock.vS0Bn2W)
    vCur = mRmsNorm2d(vCur, pBlock.vS0Bn2W);

  // 3. Residual Connection
  // Only apply residual if spatial dimensions and channels match (stride 1)
  if (pStride == 1 && pInp->ne[2] == vCur->ne[2] && pInp->ne[0] == vCur->ne[0]) {
    vCur = ggml_add(vCtx0, vCur, pInp);
  }

  return vCur;
}

// Universal Inverted Residual Block (Stage 1+)
ggml_tensor *clip_graph_mobilenetv5::mBuildInvertedResidual(ggml_tensor *pInp, const mobilenetv5_block &pBlock,
                                                            int pStride) {
  ggml_tensor *vCur = pInp;

  // 1. Depthwise Start (Optional)
  // NOTE: dw_start always has stride=1 (no downsampling here)
  if (pBlock.vDwStartW) {
    int vK = pBlock.vDwStartW->ne[0]; // 3 or 5
    int vP = vK / 2;
    vCur = ggml_conv_2d_dw(vCtx0, pBlock.vDwStartW, vCur, 1, 1, vP, vP, 1, 1);
    if (pBlock.vDwStartBnW)
      vCur = mRmsNorm2d(vCur, pBlock.vDwStartBnW);
  }

  // 2. Pointwise Expansion (1x1)
  if (pBlock.vPwExpW) {
    // Standard 1x1 conv, pad=0, stride=1
    vCur = ggml_conv_2d_direct(vCtx0, pBlock.vPwExpW, vCur, 1, 1, 0, 0, 1, 1);
    if (pBlock.vPwExpBnW)
      vCur = mRmsNorm2d(vCur, pBlock.vPwExpBnW);
    vCur = ggml_gelu(vCtx0, vCur);
  }

  // 3. Depthwise Mid (Optional)
  // NOTE: dw_mid is where downsampling happens (stride=2 for first block of stage)
  if (pBlock.vDwMidW) {
    int vK = pBlock.vDwMidW->ne[0]; // 3 or 5

    if (pStride > 1) {
      // Case: Stride 2 (Downsample) -> Use Asymmetric "Same" Padding
      vCur = mPadSame2d(vCur, vK, vK, pStride, pStride);
      vCur = ggml_conv_2d_dw(vCtx0, pBlock.vDwMidW, vCur, pStride, pStride, 0, 0, 1, 1); // pad=0
    } else {
      // Case: Stride 1 -> Use Standard Symmetric Padding
      int vP = vK / 2;
      vCur = ggml_conv_2d_dw(vCtx0, pBlock.vDwMidW, vCur, pStride, pStride, vP, vP, 1, 1);
    }

    if (pBlock.vDwMidBnW)
      vCur = mRmsNorm2d(vCur, pBlock.vDwMidBnW);
    vCur = ggml_gelu(vCtx0, vCur);
  }

  // 4. Pointwise Projection (1x1)
  if (pBlock.vPwProjW) {
    vCur = ggml_conv_2d_direct(vCtx0, pBlock.vPwProjW, vCur, 1, 1, 0, 0, 1, 1);
    if (pBlock.vPwProjBnW)
      vCur = mRmsNorm2d(vCur, pBlock.vPwProjBnW);
  }

  // Apply Layer Scaling if present
  if (pBlock.vLayerScaleW) {
    vCur = ggml_mul(vCtx0, vCur, pBlock.vLayerScaleW);
  }

  // 5. Residual Connection
  bool vSameSpatial = (pInp->ne[0] == vCur->ne[0]) && (pInp->ne[1] == vCur->ne[1]);
  bool vSameChannel = (pInp->ne[2] == vCur->ne[2]);
  if (vSameSpatial && vSameChannel) {
    vCur = ggml_add(vCtx0, vCur, pInp);
  }

  return vCur;
}

// Attention Block (MQA)
ggml_tensor *clip_graph_mobilenetv5::mBuildMobilenetAttn(ggml_tensor *pInp, const mobilenetv5_block &pBlock) {
  ggml_tensor *vCur = pInp;

  // Norm
  if (pBlock.vAttnNormW) {
    vCur = mRmsNorm2d(vCur, pBlock.vAttnNormW, 1e-6f);
  }

  // 1. Q Calculation
  ggml_tensor *vQ = ggml_conv_2d_direct(vCtx0, pBlock.vAttnQW, vCur, 1, 1, 0, 0, 1, 1);

  // 2. K Calculation (Downsampled)
  // Uses Conv2dSame(640, 640, kernel_size=(3, 3), stride=(2, 2), groups=640)
  ggml_tensor *vKInp = vCur;
  if (pBlock.vAttnKDwW) {
    int vKSize = pBlock.vAttnKDwW->ne[0];                                      // Usually 3
    vKInp = mPadSame2d(vCur, vKSize, vKSize, 2, 2);                            // Apply SAME padding
    vKInp = ggml_conv_2d_dw(vCtx0, pBlock.vAttnKDwW, vKInp, 2, 2, 0, 0, 1, 1); // padding=0
    if (pBlock.vAttnKNormW) {
      vKInp = mRmsNorm2d(vKInp, pBlock.vAttnKNormW, 1e-6f);
    }
  }
  ggml_tensor *vK = ggml_conv_2d_direct(vCtx0, pBlock.vAttnKW, vKInp, 1, 1, 0, 0, 1, 1);

  // 3. V Calculation (Downsampled)
  // Uses Conv2dSame(640, 640, kernel_size=(3, 3), stride=(2, 2), groups=640)
  ggml_tensor *vVInp = vCur;
  if (pBlock.vAttnVDwW) {
    int vVSize = pBlock.vAttnVDwW->ne[0];                                      // Usually 3
    vVInp = mPadSame2d(vCur, vVSize, vVSize, 2, 2);                            // Apply SAME padding
    vVInp = ggml_conv_2d_dw(vCtx0, pBlock.vAttnVDwW, vVInp, 2, 2, 0, 0, 1, 1); // padding=0
    if (pBlock.vAttnVNormW) {
      vVInp = mRmsNorm2d(vVInp, pBlock.vAttnVNormW, 1e-6f);
    }
  }
  ggml_tensor *vV = ggml_conv_2d_direct(vCtx0, pBlock.vAttnVW, vVInp, 1, 1, 0, 0, 1, 1);

  const int cW = vCur->ne[0];
  const int cH = vCur->ne[1];
  const int cB = vCur->ne[3];
  const int cD = vK->ne[2]; // Head dimension
  const int cNHead = vQ->ne[2] / cD;
  const int cN = cW * cH;

  // Process Q: [W, H, D*n_head, B] -> [D, N, n_head, B]
  vQ = ggml_reshape_3d(vCtx0, vQ, cN, cD * cNHead, cB);
  vQ = ggml_reshape_4d(vCtx0, vQ, cN, cD, cNHead, cB);
  vQ = ggml_permute(vCtx0, vQ, 1, 0, 2, 3); // [D, N, n_head, B]
  vQ = ggml_cont(vCtx0, vQ);

  const int cWk = vK->ne[0];
  const int cHk = vK->ne[1];
  const int cM = cWk * cHk;

  // Process K: [Wk, Hk, D, B] -> [D, M, 1, B]
  vK = ggml_reshape_3d(vCtx0, vK, cM, cD, cB);
  vK = ggml_reshape_4d(vCtx0, vK, cM, cD, 1, cB);
  vK = ggml_permute(vCtx0, vK, 1, 0, 2, 3); // [D, M, 1, B]
  vK = ggml_cont(vCtx0, vK);

  // Process V: [Wk, Hk, D, B] -> [M, D, 1, B]
  vV = ggml_reshape_3d(vCtx0, vV, cM, cD, cB);
  vV = ggml_reshape_4d(vCtx0, vV, cM, cD, 1, cB);
  vV = ggml_cont(vCtx0, vV); // [M, D, 1, B]

  // Multi-Query Attention
  float vScale = 1.0f / sqrtf((float)cD);

  // Step 1: Compute Q @ K.T
  ggml_tensor *vScores = ggml_mul_mat(vCtx0, vK, vQ);

  vScores = ggml_scale(vCtx0, vScores, vScale);

  vScores = ggml_soft_max(vCtx0, vScores);

  ggml_tensor *vKqv = ggml_mul_mat(vCtx0, vV, vScores);

  vKqv = ggml_permute(vCtx0, vKqv, 1, 0, 2, 3);
  vKqv = ggml_cont(vCtx0, vKqv);

  vKqv = ggml_reshape_3d(vCtx0, vKqv, cN, cD * cNHead, cB);
  vKqv = ggml_reshape_4d(vCtx0, vKqv, cW, cH, cD * cNHead, cB);
  vKqv = ggml_cont(vCtx0, vKqv);

  // Output projection
  vCur = ggml_conv_2d_direct(vCtx0, pBlock.vAttnOW, vKqv, 1, 1, 0, 0, 1, 1);

  // Residual & Layer Scale
  if (pInp->ne[0] == vCur->ne[0] && pInp->ne[2] == vCur->ne[2]) {
    if (pBlock.vLayerScaleW) {
      vCur = ggml_mul(vCtx0, vCur, pBlock.vLayerScaleW);
    }
    vCur = ggml_add(vCtx0, vCur, pInp);
  }

  return vCur;
}

ggml_cgraph *clip_graph_mobilenetv5::mBuild() {
  ggml_tensor *vInp = mBuildInpRaw();

  // 1. Stem - Conv2dSame(3, 64, kernel_size=(3, 3), stride=(2, 2))
  ggml_tensor *vCur = mPadSame2d(vInp, 3, 3, 2, 2); // Apply SAME padding

  vCur = ggml_conv_2d_direct(vCtx0, vModel.vMobilenetStemConvW, vCur, 2, 2, 0, 0, 1, 1); // padding=0
  if (vModel.vMobilenetStemConvB) {
    vCur = ggml_add(vCtx0, vCur, vModel.vMobilenetStemConvB);
  }
  if (vModel.vMobilenetStemNormW)
    vCur = mRmsNorm2d(vCur, vModel.vMobilenetStemNormW);
  vCur = ggml_gelu(vCtx0, vCur);

  // 2. Blocks
  std::vector<ggml_tensor *> lIntermediateFeatures;
  const int cTotalBlocks = vModel.lMobilenetBlocks.size();

  auto vIsStageStart = [&](int pI) {
    if (pI == 0)
      return true;
    for (int vEndIdx : vModel.lMobilenetStageEnds) {
      if (pI == vEndIdx + 1)
        return true;
    }
    return false;
  };

  auto vIsFusionPoint = [&](int pI) {
    if (vModel.lMobilenetStageEnds.size() >= 4) {
      if (pI == vModel.lMobilenetStageEnds[2])
        return true; // End of Stage 2
      if (pI == vModel.lMobilenetStageEnds[3])
        return true; // End of Stage 3
    } else {
      if (pI == cTotalBlocks - 1)
        return true;
    }
    return false;
  };

  for (int vI = 0; vI < cTotalBlocks; vI++) {
    const auto &vBlock = vModel.lMobilenetBlocks[vI];
    int vStride = vIsStageStart(vI) ? 2 : 1;

    if (vBlock.vS0ConvExpW)
      vCur = mBuildEdgeResidual(vCur, vBlock, vStride);
    else if (vBlock.vAttnQW)
      vCur = mBuildMobilenetAttn(vCur, vBlock);
    else
      vCur = mBuildInvertedResidual(vCur, vBlock, vStride);

    if (vIsFusionPoint(vI)) {

      lIntermediateFeatures.push_back(vCur);
    }
  }

  // 3. Multi-Scale Fusion Adapter (MSFA)
  if (!lIntermediateFeatures.empty()) {

    // A. Reference Resolution: PyTorch implementation uses inputs[0]
    // We assume intermediate_features[0] is the "High Resolution" target.
    // In MobileNet designs, this is typically the feature map with the smallest stride (e.g. 32x32).
    ggml_tensor *vTargetFeat = lIntermediateFeatures[0];
    int vHighResW = vTargetFeat->ne[0];
    int vHighResH = vTargetFeat->ne[1];

    std::vector<ggml_tensor *> lResizedFeats;

    // B. Resize inputs to match inputs[0] (High Resolution)
    for (auto vFeat : lIntermediateFeatures) {
      int vFeatW = vFeat->ne[0];
      int vFeatH = vFeat->ne[1];

      // PyTorch: if feat_size < high_resolution: interpolate
      if (vFeatW < vHighResW || vFeatH < vHighResH) {
        // Calculate scale factor.
        // Note: PyTorch 'nearest' works on arbitrary float scales.
        // ggml_upscale generally takes integer factors or target sizes depending on helper.
        // Assuming standard power-of-2 scaling (e.g. 16 -> 32 means scale=2).
        int vScaleW = vHighResW / vFeatW;
        // int vScaleH = vHighResH / vFeatH;

        // Safety check for non-integer scaling if strictly replicating
        GGML_ASSERT(vHighResW % vFeatW == 0);

        // Upsample (Nearest Neighbor)
        // 2 is the scale factor
        vFeat = ggml_upscale(vCtx0, vFeat, vScaleW, ggml_scale_mode::GGML_SCALE_MODE_NEAREST);
      }
      lResizedFeats.push_back(vFeat);
    }

    // C. Concatenate at High Resolution (Channel Dim = 2 in ggml)
    vCur = lResizedFeats[0];
    for (size_t vK = 1; vK < lResizedFeats.size(); ++vK) {
      vCur = ggml_concat(vCtx0, vCur, lResizedFeats[vK], 2);
    }

    // D. FFN (UniversalInvertedResidual)
    // Structure: Expand Conv -> Norm -> GELU -> Project Conv -> Norm

    // 1. Expansion
    if (vModel.vMsfaFfnExpandW) {
      // 1x1 Conv
      vCur = ggml_conv_2d_direct(vCtx0, vModel.vMsfaFfnExpandW, vCur, 1, 1, 0, 0, 1, 1);

      if (vModel.vMsfaFfnExpandBn) {
        vCur = mRmsNorm2d(vCur, vModel.vMsfaFfnExpandBn);
      }

      vCur = ggml_gelu(vCtx0, vCur);
    }

    // 2. Projection (No DW because kernel_size=0)
    if (vModel.vMsfaFfnProjectW) {
      // 1x1 Conv
      vCur = ggml_conv_2d_direct(vCtx0, vModel.vMsfaFfnProjectW, vCur, 1, 1, 0, 0, 1, 1);

      // UniversalInvertedResidual typically has a norm after projection
      if (vModel.vMsfaFfnProjectBn) {
        vCur = mRmsNorm2d(vCur, vModel.vMsfaFfnProjectBn);
      }
    }

    // E. Final Downsample to Target Resolution (Output Resolution)
    // PyTorch: matches self.output_resolution (e.g. 16x16)
    const int cTargetOutRes = 16;
    int vCurrentW = vCur->ne[0];

    if (vCurrentW > cTargetOutRes) {
      int vS = vCurrentW / cTargetOutRes;

      GGML_ASSERT(vCurrentW % cTargetOutRes == 0);

      // Avg Pool: Kernel=s, Stride=s
      vCur = ggml_pool_2d(vCtx0, vCur, GGML_OP_POOL_AVG, vS, vS, vS, vS, 0, 0);
    }

    // F. Final Norm
    if (vModel.vMsfaConcatNormW) {
      vCur = mRmsNorm2d(vCur, vModel.vMsfaConcatNormW);
    }
  }

  // 4. Gemma 3n Multimodal Projection (Embedder)
  // Input: 'cur' is [Width, Height, Channels, Batch]
  int vW = vCur->ne[0];
  int vH = vCur->ne[1];
  int vC = vCur->ne[2];
  int vB = vCur->ne[3];

  GGML_ASSERT(vC == vHparams.vNEmbd);

  // 1. Permute and Flatten to [Channels, Tokens, Batch]
  // PyTorch expects (Batch, Seq, Hidden), GGML usually processes (Hidden, Seq, Batch)
  vCur = ggml_permute(vCtx0, vCur, 2, 1, 0, 3); // -> [C, H, W, B]
  vCur = ggml_permute(vCtx0, vCur, 0, 2, 1, 3); // -> [C, W, H, B]
  vCur = ggml_cont(vCtx0, vCur);
  vCur = ggml_reshape_3d(vCtx0, vCur, vC, vW * vH, vB);
  vCur = ggml_cont(vCtx0, vCur);

  // 2. FEATURE SCALING
  // PyTorch: vision_outputs *= self.config.vision_config.hidden_size**0.5
  const float cScaleFactor = sqrtf((float)vC);
  vCur = ggml_scale(vCtx0, vCur, cScaleFactor);

  // 3. SOFT EMBEDDING NORM
  // PyTorch: self._norm(x) * self.weight
  // We must normalize regardless, then multiply if weight exists.
  {
    const float cEps = 1e-6f; // Gemma3n uses 1e-6
    vCur = ggml_rms_norm(vCtx0, vCur, cEps);

    if (vModel.vMmSoftEmbNormW) {
      // Weight shape is (2048,) -> Element-wise broadcast multiply
      vCur = ggml_mul(vCtx0, vCur, vModel.vMmSoftEmbNormW);
    }
  }

  // 4. PROJECTION
  // PyTorch: embedding_projection = nn.Linear(vision_hidden, text_hidden, bias=False)
  // Weight stored as [out_features, in_features] = [text_hidden_size, vision_hidden_size]
  if (vModel.vMmInputProjW) {
    vCur = mBuildMm(vModel.vMmInputProjW, vCur);
  }

  // 5. POST PROJECTION NORM
  // PyTorch: embedding_post_projection_norm = Gemma3nRMSNorm(..., with_scale=False)
  // with_scale=False means weight is registered as buffer with value 1.0
  // So output = rms_norm(x) * 1.0 = rms_norm(x), magnitude ~1
  {
    const float cEps = 1e-6f;
    vCur = ggml_rms_norm(vCtx0, vCur, cEps);

    if (vModel.vMmPostProjNormW) {
      // If weight is loaded, multiply (should be ~1.0 anyway)
      vCur = ggml_mul(vCtx0, vCur, vModel.vMmPostProjNormW);
    }
  }

  ggml_build_forward_expand(vGf, vCur);
  return vGf;
}
