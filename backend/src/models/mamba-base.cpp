#include "models.h"

#include "mgs-memory-recurrent.h"

llm_build_mamba_base::llm_build_mamba_base(const llm_graph_params &pParams) : llm_graph_context(pParams) {}

ggml_tensor *llm_build_mamba_base::mBuildMamba2Layer(llm_graph_input_rs *pInp, ggml_tensor *pCur,
                                                     const mgs_model &pModel, const mgs_ubatch &pUbatch,
                                                     int pIl) const {
  const auto *vMctxCur = pInp->vMctx;

  const auto cKvHead = vMctxCur->mGetHead();

  const int64_t cDConv = vHparams.vSsmDConv;
  const int64_t cDInner = vHparams.vSsmDInner;
  const int64_t cDState = vHparams.vSsmDState;
  const int64_t cNHead = vHparams.vSsmDtRank;
  const int64_t cHeadDim = cDInner / cNHead;
  const int64_t cNGroup = vHparams.vSsmNGroup;
  const int64_t cNSeqs = pUbatch.vNSeqs;

  const int64_t cNSeqTokens = pUbatch.vNSeqTokens;

  GGML_ASSERT(cNSeqs != 0);
  GGML_ASSERT(pUbatch.mEqualSeqs());
  GGML_ASSERT(pUbatch.vNTokens == cNSeqTokens * cNSeqs);
  GGML_ASSERT(cDInner % cNHead == 0);
  GGML_ASSERT(cDInner % cNGroup == 0);

  ggml_tensor *vConvStatesAll = vMctxCur->mGetRL(pIl);
  ggml_tensor *vSsmStatesAll = vMctxCur->mGetSL(pIl);

  ggml_tensor *vConv = mBuildRs(pInp, vConvStatesAll, vHparams.mNEmbdR(), cNSeqs);
  vConv = ggml_reshape_3d(vCtx0, vConv, cDConv - 1, cDInner + 2 * cNGroup * cDState, cNSeqs);

  // {n_embd, n_tokens} => {n_embd, n_seq_tokens, n_seqs}
  pCur = ggml_reshape_3d(vCtx0, pCur, pCur->ne[0], cNSeqTokens, cNSeqs);

  // d_in_proj = 2 * self.d_inner + 2 * self.ngroups * self.d_state + self.nheads

  // {n_embd, d_in_proj} @ {n_embd, n_seq_tokens, n_seqs} => {d_in_proj, n_seq_tokens, n_seqs}
  ggml_tensor *vZxBCdt = mBuildLoraMm(pModel.lLayers[pIl].vSsmIn, pCur, pModel.lLayers[pIl].vSsmInS);

  // split the above in three
  ggml_tensor *vZ = ggml_view_4d(vCtx0, vZxBCdt, cHeadDim, cNHead, cNSeqTokens, cNSeqs, cHeadDim * vZxBCdt->nb[0],
                                 vZxBCdt->nb[1], vZxBCdt->nb[2], 0);
  ggml_tensor *vXBC = ggml_view_3d(vCtx0, vZxBCdt, cDInner + 2 * cNGroup * cDState, cNSeqTokens, cNSeqs, vZxBCdt->nb[1],
                                   vZxBCdt->nb[2], cDInner * ggml_element_size(vZxBCdt));
  ggml_tensor *vDt = ggml_view_3d(vCtx0, vZxBCdt, cNHead, cNSeqTokens, cNSeqs, vZxBCdt->nb[1], vZxBCdt->nb[2],
                                  (2 * cDInner + 2 * cNGroup * cDState) * ggml_element_size(vZxBCdt));

  // conv
  {
    // => {d_conv - 1 + n_seq_tokens, d_inner + 2*n_group*d_state, n_seqs}
    ggml_tensor *vConvX = ggml_concat(vCtx0, vConv, ggml_transpose(vCtx0, vXBC), 0);

    // copy last (d_conv - 1) columns back into the state cache
    ggml_tensor *vLastConv = ggml_view_3d(vCtx0, vConvX, cDConv - 1, cDInner + 2 * cNGroup * cDState, cNSeqs,
                                          vConvX->nb[1], vConvX->nb[2], cNSeqTokens * (vConvX->nb[0]));

    ggml_build_forward_expand(
        vGf, ggml_cpy(vCtx0, vLastConv,
                      ggml_view_1d(vCtx0, vConvStatesAll, (cDConv - 1) * (cDInner + 2 * cNGroup * cDState) * (cNSeqs),
                                   cKvHead * (cDConv - 1) * (cDInner + 2 * cNGroup * cDState) *
                                       ggml_element_size(vConvStatesAll))));

    // 1D convolution
    // The equivalent is to make a self-overlapping view of conv_x
    // over d_conv columns at each stride in the 3rd dimension,
    // then element-wise multiply that with the conv1d weight,
    // then sum the elements of each row,
    // (the last two steps are a dot product over rows (also doable with mul_mat))
    // then permute away the ne[0] dimension,
    // and then you're left with the resulting x tensor.
    // For simultaneous sequences, all sequences need to have the same length.
    vXBC = ggml_ssm_conv(vCtx0, vConvX, pModel.lLayers[pIl].vSsmConv1d);

    // bias
    vXBC = ggml_add(vCtx0, vXBC, pModel.lLayers[pIl].vSsmConv1dB);

    vXBC = ggml_silu(vCtx0, vXBC);
  }

  // ssm
  {
    // These correspond to V K Q in SSM/attention duality
    ggml_tensor *vX = ggml_view_4d(vCtx0, vXBC, cHeadDim, cNHead, cNSeqTokens, cNSeqs, cHeadDim * vXBC->nb[0],
                                   vXBC->nb[1], vXBC->nb[2], 0);
    ggml_tensor *vB = ggml_view_4d(vCtx0, vXBC, cDState, cNGroup, cNSeqTokens, cNSeqs, cDState * vXBC->nb[0],
                                   vXBC->nb[1], vXBC->nb[2], cDInner * ggml_element_size(vXBC));
    ggml_tensor *vC = ggml_view_4d(vCtx0, vXBC, cDState, cNGroup, cNSeqTokens, cNSeqs, cDState * vXBC->nb[0],
                                   vXBC->nb[1], vXBC->nb[2], (cDInner + cNGroup * cDState) * ggml_element_size(vXBC));

    // {n_head, n_seq_tokens, n_seqs}
    vDt = ggml_add(vCtx0, ggml_cont(vCtx0, vDt), pModel.lLayers[pIl].vSsmDtB);

    ggml_tensor *vA = pModel.lLayers[pIl].vSsmA;

    // use the states and the indices provided by build_recurrent_state
    // (this is necessary in order to properly use the states before they are overwritten,
    //  while avoiding to make unnecessary copies of the states)
    auto vGetSsmRows = [&](ggml_context *pCtx, ggml_tensor *pStates, ggml_tensor *pIds) {
      ggml_tensor *vSsm = ggml_reshape_4d(pCtx, pStates, cDState, cHeadDim, cNHead, vMctxCur->mGetSize());

      // TODO: use semistructured matrices to implement state-space duality
      // => {d_inner, n_seq_tokens, n_seqs} and {d_state, d_inner, n_seqs}
      return ggml_ssm_scan(pCtx, vSsm, vX, vDt, vA, vB, vC, pIds);
    };

    ggml_tensor *vYSsm = mBuildRs(pInp, vSsmStatesAll, vHparams.mNEmbdS(), pUbatch.vNSeqs, vGetSsmRows);

    // store last states
    ggml_build_forward_expand(
        vGf, ggml_cpy(vCtx0, ggml_view_1d(vCtx0, vYSsm, cDState * cDInner * cNSeqs, ggml_nelements(vX) * vX->nb[0]),
                      ggml_view_1d(vCtx0, vSsmStatesAll, cDState * cDInner * cNSeqs,
                                   cKvHead * cDState * cDInner * ggml_element_size(vSsmStatesAll))));

    ggml_tensor *vY = ggml_view_4d(vCtx0, vYSsm, cHeadDim, cNHead, cNSeqTokens, cNSeqs, vX->nb[1], cNHead * vX->nb[1],
                                   cNSeqTokens * cNHead * vX->nb[1], 0);

    // TODO: skip computing output earlier for unused tokens

    vY = ggml_add(vCtx0, vY, ggml_mul(vCtx0, vX, pModel.lLayers[pIl].vSsmD));
    mCb(vY, "mamba2_y_add_d", pIl);
    vY = ggml_swiglu_split(vCtx0, ggml_cont(vCtx0, vZ), vY);

    // grouped RMS norm
    if (pModel.lLayers[pIl].vSsmNorm) {
      vY = ggml_reshape_4d(vCtx0, vY, cDInner / cNGroup, cNGroup, cNSeqTokens, cNSeqs);
      vY = mBuildNorm(vY, pModel.lLayers[pIl].vSsmNorm, NULL, cLlmNormRms, pIl);
    }

    vY = ggml_reshape_3d(vCtx0, vY, cDInner, cNSeqTokens, cNSeqs);

    // {d_inner, n_embd} @ {d_inner, n_seq_tokens, n_seqs} => {n_embd, n_seq_tokens, n_seqs}
    pCur = mBuildLoraMm(pModel.lLayers[pIl].vSsmOut, vY, pModel.lLayers[pIl].vSsmOutS);
  }

  // {n_embd, n_seq_tokens, n_seqs} => {n_embd, n_tokens}
  pCur = ggml_reshape_2d(vCtx0, pCur, pCur->ne[0], cNSeqTokens * cNSeqs);
  mCb(pCur, "mamba_out", pIl);

  return pCur;
}
