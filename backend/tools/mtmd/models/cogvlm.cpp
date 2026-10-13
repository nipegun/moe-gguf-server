#include "models.h"

ggml_cgraph *clip_graph_cogvlm::mBuild() {
  GGML_ASSERT(vModel.vClassEmbedding != nullptr);
  GGML_ASSERT(vModel.vPositionEmbeddings != nullptr);

  const int cNPos = cNPatches + 1; // +1 for [CLS]

  // build input and concatenate class embedding
  ggml_tensor *vInp = mBuildInp();
  vInp = ggml_concat(vCtx0, vInp, vModel.vClassEmbedding, 1);

  vInp = ggml_add(vCtx0, vInp, vModel.vPositionEmbeddings);
  mCb(vInp, "inp_pos", -1);

  ggml_tensor *vInpL = vInp;

  for (int vIl = 0; vIl < cNLayer; vIl++) {
    auto &vLayer = vModel.lLayers[vIl];
    ggml_tensor *vCur = vInpL;

    vCur = mBuildMm(vLayer.vQkvW, vCur);

    vCur = ggml_add(vCtx0, vCur, vLayer.vQkvB);

    ggml_tensor *vQcur = ggml_view_3d(vCtx0, vCur, cDHead, cNHead, cNPos, cDHead * sizeof(float), vCur->nb[1], 0);
    ggml_tensor *vKcur =
        ggml_view_3d(vCtx0, vCur, cDHead, cNHead, cNPos, cDHead * sizeof(float), vCur->nb[1], cNEmbd * sizeof(float));
    ggml_tensor *vVcur = ggml_view_3d(vCtx0, vCur, cDHead, cNHead, cNPos, cDHead * sizeof(float), vCur->nb[1],
                                      2 * cNEmbd * sizeof(float));

    mCb(vQcur, "Qcur", vIl);
    mCb(vKcur, "Kcur", vIl);
    mCb(vVcur, "Vcur", vIl);

    vCur = mBuildAttn(vLayer.vOW, vLayer.vOB, vQcur, vKcur, vVcur, nullptr, vKqScale, vIl);
    mCb(vCur, "attn_out", vIl);

    vCur = mBuildNorm(vCur, vLayer.vLn1W, vLayer.vLn1B, cNormTypeNormal, cEps, vIl);
    mCb(vCur, "attn_post_norm", vIl);

    vCur = ggml_add(vCtx0, vCur, vInpL);
    vInpL = vCur;

    vCur = mBuildFfn(vCur, vLayer.vFfUpW, vLayer.vFfUpB, vLayer.vFfGateW, vLayer.vFfGateB, vLayer.vFfDownW,
                     vLayer.vFfDownB, vHparams.vFfnOp, vIl);

    mCb(vCur, "ffn_out", vIl);

    vCur = mBuildNorm(vCur, vLayer.vLn2W, vLayer.vLn2B, cNormTypeNormal, cEps, vIl);
    mCb(vCur, "ffn_post_norm", vIl);

    vCur = ggml_add(vCtx0, vCur, vInpL);
    mCb(vCur, "layer_out", vIl);
    vInpL = vCur;
  }

  // remove CLS token
  ggml_tensor *vCur = ggml_view_2d(vCtx0, vInpL, cNEmbd, cNPatches, ggml_row_size(vInpL->type, cNEmbd), 0);

  // Multiply with mm_model_proj
  vCur = mBuildMm(vModel.vMmModelProj, vCur);

  // Apply layernorm, weight, bias
  vCur = mBuildNorm(vCur, vModel.vMmPostFcNormW, vModel.vMmPostFcNormB, cNormTypeNormal, 1e-5, -1);

  // Apply GELU
  vCur = ggml_gelu_inplace(vCtx0, vCur);

  // Branch 1: multiply with mm_h_to_4h_w
  ggml_tensor *vHTo4h = mBuildMm(vModel.vMmHTo4hW, vCur);

  // Branch 2: multiply with mm_gate_w
  ggml_tensor *vGate = mBuildMm(vModel.vMmGateW, vCur);

  // Apply silu
  vGate = ggml_swiglu_split(vCtx0, vGate, vHTo4h);

  // Apply mm_4h_to_h_w
  vCur = mBuildMm(vModel.vMm4hToHW, vGate);

  // Concatenate with boi and eoi
  vCur = ggml_concat(vCtx0, vModel.vMmBoi, vCur, 1);
  vCur = ggml_concat(vCtx0, vCur, vModel.vMmEoi, 1);

  // build the graph
  ggml_build_forward_expand(vGf, vCur);

  return vGf;
}
