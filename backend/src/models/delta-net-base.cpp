#include "models.h"

#include "mgs-impl.h"
#include "mgs-memory-recurrent.h"

// utility to get one slice from the third dimension
// input dim:  [x, y, c, b]
// output dim: [x, y, 1, b]
static ggml_tensor *fGetSlice2d(ggml_context *pCtx0, ggml_tensor *pT, int64_t pC) {
  return ggml_view_4d(pCtx0, pT, pT->ne[0], pT->ne[1], 1, pT->ne[3], pT->nb[1], pT->nb[2], pT->nb[3], pT->nb[2] * pC);
}

llm_build_delta_net_base::llm_build_delta_net_base(const llm_graph_params &pParams) : llm_graph_context(pParams) {}

std::pair<ggml_tensor *, ggml_tensor *>
llm_build_delta_net_base::mBuildDeltaNetChunking(ggml_tensor *pQ, ggml_tensor *pK, ggml_tensor *pV, ggml_tensor *pG,
                                                 ggml_tensor *pB, ggml_tensor *pS, int pIl) {
  const int64_t cSK = pQ->ne[0];
  const int64_t cHK = pQ->ne[1];
  const int64_t cNTokens = pQ->ne[2];
  const int64_t cNSeqs = pQ->ne[3];

  const int64_t cSV = pV->ne[0];
  const int64_t cHV = pV->ne[1];
  const bool cKda = (pG->ne[0] == cSK && pG->ne[1] == cHK);

  GGML_ASSERT(cSK == cSV);
  GGML_ASSERT(cHV % cHK == 0);

  GGML_ASSERT(pQ->ne[0] == cSK && pQ->ne[1] == cHK && pQ->ne[2] == cNTokens && pQ->ne[3] == cNSeqs);
  GGML_ASSERT(pK->ne[0] == cSK && pK->ne[1] == cHK && pK->ne[2] == cNTokens && pK->ne[3] == cNSeqs);
  GGML_ASSERT(pV->ne[0] == cSV && pV->ne[1] == cHV && pV->ne[2] == cNTokens && pV->ne[3] == cNSeqs);

  GGML_ASSERT(pG->ne[0] == 1 || pG->ne[0] == cSV);
  GGML_ASSERT(pG->ne[1] == cHV && pG->ne[2] == cNTokens && pG->ne[3] == cNSeqs);
  GGML_ASSERT(pB->ne[0] == 1 && pB->ne[1] == cHV && pB->ne[2] == cNTokens && pB->ne[3] == cNSeqs);
  GGML_ASSERT(pS->ne[0] == cSV && pS->ne[1] == cSV && pS->ne[2] == cHV && pS->ne[3] == cNSeqs);

  const float cScale = 1.0f / sqrtf(cSK);

  pQ = ggml_scale(vCtx0, pQ, cScale);

  mCb(pQ, "q_in", pIl);
  mCb(pK, "k_in", pIl);
  mCb(pV, "v_in", pIl);
  mCb(pB, "b_in", pIl);
  mCb(pG, "g_in", pIl);

  pQ = ggml_permute(vCtx0, pQ, 0, 2, 1, 3); // [S_k, n_tokens, H_k, n_seqs]
  pK = ggml_permute(vCtx0, pK, 0, 2, 1, 3); // [S_k, n_tokens, H_k, n_seqs]
  pV = ggml_permute(vCtx0, pV, 0, 2, 1, 3); // [S_v, n_tokens, H_v, n_seqs]
  pG = ggml_permute(vCtx0, pG, 0, 2, 1, 3); // [g_0, n_tokens, H_v, n_seqs]
  pB = ggml_permute(vCtx0, pB, 0, 2, 1, 3); // [  1, n_tokens, H_v, n_seqs]

  const int cCs = cKda ? 16 : 64; // chunk size

  const int cPad = (cCs - cNTokens % cCs) % cCs;
  const int cNChunks = (cNTokens + cPad) / cCs;

  pQ = ggml_pad(vCtx0, pQ, 0, cPad, 0, 0);
  pK = ggml_pad(vCtx0, pK, 0, cPad, 0, 0);
  pV = ggml_pad(vCtx0, pV, 0, cPad, 0, 0);
  pG = ggml_pad(vCtx0, pG, 0, cPad, 0, 0);
  pB = ggml_pad(vCtx0, pB, 0, cPad, 0, 0);

  ggml_tensor *vVB = ggml_mul(vCtx0, pV, pB);
  ggml_tensor *vKB = ggml_mul(vCtx0, pK, pB);

  mCb(vVB, "v_b", pIl);
  mCb(vKB, "k_b", pIl);

  pQ = ggml_reshape_4d(vCtx0, pQ, cSK, cCs, cNChunks, cHK * cNSeqs);
  pK = ggml_reshape_4d(vCtx0, pK, cSK, cCs, cNChunks, cHK * cNSeqs);
  vKB = ggml_reshape_4d(vCtx0, vKB, cSK, cCs, cNChunks, cHV * cNSeqs);
  pV = ggml_reshape_4d(vCtx0, pV, cSV, cCs, cNChunks, cHV * cNSeqs);
  vVB = ggml_reshape_4d(vCtx0, vVB, cSV, cCs, cNChunks, cHV * cNSeqs);

  pG = ggml_reshape_4d(vCtx0, pG, pG->ne[0], cCs, cNChunks, cHV * cNSeqs);
  pB = ggml_reshape_4d(vCtx0, pB, 1, cCs, cNChunks, cHV * cNSeqs);

  // [CS, g_0, n_chunks, H_v * n_seqs]
  // TODO: extend ggml_cumsum with axis parameter to avoid transpose
  ggml_tensor *vGCs = ggml_cumsum(vCtx0, ggml_cont(vCtx0, ggml_transpose(vCtx0, pG)));
  mCb(vGCs, "g_cs", pIl);

  ggml_tensor *vKb = nullptr;
  ggml_tensor *vKq = nullptr;
  if (cKda) {
    const int64_t cChb = cNChunks * cHK * cNSeqs;

    ggml_tensor *vGCsI = ggml_reshape_4d(vCtx0, vGCs, cCs, 1, cSK, cChb); // [chunk_size, 1, S_k, CHB]
    ggml_tensor *vGCsJ = ggml_reshape_4d(vCtx0, vGCs, 1, cCs, cSK, cChb); // [1, chunk_size, S_k, CHB]

    vGCsJ = ggml_repeat_4d(vCtx0, vGCsJ, cCs, cCs, cSK,
                           cChb); // [1, chunk_size, S_k, CHB] -> [chunk_size, chunk_size, S_k, CHB]

    // decay_mask [chunk_size,chunk_size,S_k,CHB]
    ggml_tensor *vDecayMask;
    vDecayMask = ggml_sub(vCtx0, vGCsJ, vGCsI);
    vDecayMask = ggml_tri(vCtx0, vDecayMask, GGML_TRI_TYPE_LOWER_DIAG);
    vDecayMask = ggml_exp(vCtx0, vDecayMask);
    mCb(vDecayMask, "decay_mask", pIl);

    // decay_mask [S_k,BT_j,BT_i,CHB] *Note* second and third chunk_sizes are switched
    vDecayMask = ggml_cont_4d(vCtx0, ggml_permute(vCtx0, vDecayMask, 2, 1, 0, 3), cSK, cCs, cCs, cChb);

    ggml_tensor *vKBI = ggml_reshape_4d(vCtx0, vKB, cSK, cCs, 1, cChb);
    ggml_tensor *vKJ = ggml_reshape_4d(vCtx0, pK, cSK, 1, cCs, cChb);
    ggml_tensor *vQI = ggml_reshape_4d(vCtx0, pQ, cSK, cCs, 1, cChb);

    ggml_tensor *vDecayKBI = ggml_mul(vCtx0, vDecayMask, vKBI);
    ggml_tensor *vDecayQI = ggml_mul(vCtx0, vDecayMask, vQI);

    // decay_k_b_i [S,BT,BT,CHB] @ k_j [S,1,BT,CHB] = Akk [BT,1,BT,CHB]
    vKb = ggml_mul_mat(vCtx0, vDecayKBI, vKJ);
    vKq = ggml_mul_mat(vCtx0, vDecayQI, vKJ);

    vKb = ggml_cont(vCtx0, ggml_transpose(vCtx0, ggml_reshape_4d(vCtx0, vKb, cCs, cCs, cNChunks, cHV * cNSeqs)));
    vKq = ggml_cont(vCtx0, ggml_transpose(vCtx0, ggml_reshape_4d(vCtx0, vKq, cCs, cCs, cNChunks, cHV * cNSeqs)));
  } else {
    ggml_tensor *vGCsI = vGCs;
    ggml_tensor *vGCsJ = ggml_reshape_4d(vCtx0, vGCs, 1, cCs, cNChunks, cHV * cNSeqs);

    vGCsJ = ggml_repeat_4d(vCtx0, vGCsJ, cCs, cCs, cNChunks, cHV * cNSeqs);

    // [CS, CS, n_chunks, H_v * n_seqs]
    ggml_tensor *vDecayMask;
    vDecayMask = ggml_sub(vCtx0, vGCsJ, vGCsI);
    vDecayMask = ggml_tri(vCtx0, vDecayMask, GGML_TRI_TYPE_LOWER_DIAG);
    vDecayMask = ggml_exp(vCtx0, vDecayMask);
    mCb(vDecayMask, "decay_mask", pIl);

    // [CS, CS, n_chunks, H_k * n_seqs]
    vKb = ggml_mul_mat(vCtx0, pK, vKB);
    vKb = ggml_mul(vCtx0, vKb, vDecayMask);

    // [CS, CS, n_chunks, H_k * n_seqs]
    vKq = ggml_mul_mat(vCtx0, pK, pQ);
    vKq = ggml_mul(vCtx0, vKq, vDecayMask);
  }

  vKq = ggml_tri(vCtx0, vKq, GGML_TRI_TYPE_LOWER_DIAG);
  mCb(vKq, "kq", pIl);

  // [CS, CS, n_chunks, H_k * n_seqs]
  ggml_tensor *vAttn;
  vAttn = ggml_tri(vCtx0, vKb, GGML_TRI_TYPE_LOWER);
  mCb(vAttn, "attn", pIl);

  ggml_tensor *vIdentity;
  vIdentity = ggml_view_1d(vCtx0, vAttn, cCs, 0);
  vIdentity = ggml_fill(vCtx0, vIdentity, 1.0f);
  vIdentity = ggml_diag(vCtx0, vIdentity);

  ggml_tensor *vLhs = ggml_add(vCtx0, vAttn, vIdentity);
  mCb(vLhs, "dnet_add_ch_lhs", pIl);

  vAttn = ggml_neg(vCtx0, vAttn);
  mCb(vAttn, "attn_pre_solve", pIl);

  ggml_tensor *vLinSolve = ggml_solve_tri(vCtx0, vLhs, vAttn, true, true, false);
  vAttn = ggml_add(vCtx0, vLinSolve, vIdentity);
  mCb(vAttn, "dnet_add_ch_attn_solved", pIl); // [CS, CS, n_chunks, H_k * n_seqs]

  // [S_v, CS, n_chunks, H_v * n_seqs]
  pV = ggml_mul_mat(vCtx0, ggml_cont(vCtx0, ggml_transpose(vCtx0, vVB)), vAttn);

  // [CS, 1, n_chunks, H_v * n_seqs] KDA: [CS, S_k, n_chunks, H_v * n_seqs]
  ggml_tensor *vGExp = ggml_exp(vCtx0, vGCs);

  vKB = ggml_cont(vCtx0, ggml_transpose(vCtx0, vKB));

  // [CS, S_k, n_chunks, H_k * n_seqs]
  ggml_tensor *vKbg = ggml_mul(vCtx0, vKB, vGExp);
  mCb(vKbg, "k_beta_g_exp", pIl);

  // [S_k, CS, n_chunks, H_k * n_seqs]
  ggml_tensor *vKCd = ggml_mul_mat(vCtx0, vKbg, vAttn);
  mCb(vKCd, "k_cumdecay", pIl);

  // [1, CS, n_chunks, H_k * n_seqs] KDA: [S_k, CS, n_chunks, H_k * n_seqs]
  ggml_tensor *vGExpT = ggml_cont(vCtx0, ggml_transpose(vCtx0, vGExp));
  ggml_tensor *vQGExp = ggml_mul(vCtx0, pQ, vGExpT);

  // vectorized calculation of key_gdiff
  // improved from the chunked version:
  //   g_last = torch.clamp(g_cum[:, :, -1], max=50.0).exp().unsqueeze(-1).unsqueeze(-1)
  //   g_diff = torch.clamp(g_cum[:, :, -1:] - g_cum, max=50.0).exp()
  //   key_gdiff = key * g_diff.unsqueeze(-1)
  //   kgdmulvnew = (key_gdiff).transpose(-1, -2) @ v_new
  //   last_recurrent_state = last_recurrent_state * g_last + kgdmulvnew

  // get last element in g_cumsum along CS dimension (ne0)
  // example: [[x, y, z, ..., last], ...] -> [[last], ...]
  // [1, 1, n_chunks, H_v * n_seqs] KDA: [1, S_k, n_chunks, H_v * n_seqs]
  ggml_tensor *vGLast = ggml_view_4d(vCtx0, vGCs, 1, vGCs->ne[1], vGCs->ne[2], vGCs->ne[3], vGCs->nb[1], vGCs->nb[2],
                                     vGCs->nb[3], ggml_row_size(vGCs->type, vGCs->ne[0] - 1));
  mCb(vGLast, "g_last", pIl);

  // TODO: remove this cont when CUDA supports non-cont unary ops
  vGLast = ggml_cont(vCtx0, vGLast);

  // [1, 1, n_chunks, H_v * n_seqs] KDA: [S_k, 1, n_chunks, H_v * n_seqs]
  ggml_tensor *vGLastExpT = ggml_transpose(vCtx0, ggml_exp(vCtx0, vGLast));
  mCb(vGLastExpT, "g_last_exp_t", pIl);

  // [CS, 1, n_chunks, H_v * n_seqs] KDA: [CS, S_k, n_chunks, H_v * n_seqs]
  ggml_tensor *vGDiff = ggml_neg(vCtx0, ggml_sub(vCtx0, vGCs, vGLast));
  mCb(vGDiff, "g_diff", pIl);

  ggml_tensor *vGDiffExpT = ggml_cont(vCtx0, ggml_transpose(vCtx0, ggml_exp(vCtx0, vGDiff)));

  // [S_k, CS, n_chunks, H_v * n_seqs]
  ggml_tensor *vKg = ggml_mul(vCtx0, pK, vGDiffExpT);
  mCb(vKg, "key_gdiff", pIl);

  // [CS, S_k, n_chunks, H_v * n_seqs]
  ggml_tensor *vKgT = ggml_cont(vCtx0, ggml_transpose(vCtx0, vKg));
  mCb(vKgT, "key_gdiff_t", pIl);

  pS = ggml_reshape_4d(vCtx0, pS, cSV, cSV, 1, cHV * cNSeqs);
  mCb(pS, "dnet_add_ch_state", pIl);

  // [CS, S_v, n_chunks, H_v * n_seqs]
  ggml_tensor *vVT = ggml_cont(vCtx0, ggml_transpose(vCtx0, pV));

  for (int64_t vChunk = 0; vChunk < cNChunks; vChunk++) {
    ggml_tensor *vChKCd = fGetSlice2d(vCtx0, vKCd, vChunk);     // [S_k,  CS, 1, H_k * n_seqs]
    ggml_tensor *vChVT = fGetSlice2d(vCtx0, vVT, vChunk);       // [ CS, S_v, 1, H_v * n_seqs]
    ggml_tensor *vChKq = fGetSlice2d(vCtx0, vKq, vChunk);       // [ CS,  CS, 1, H_k * n_seqs]
    ggml_tensor *vChQGExp = fGetSlice2d(vCtx0, vQGExp, vChunk); // [S_k,  CS, 1, H_k * n_seqs]
    ggml_tensor *vChKgT = fGetSlice2d(vCtx0, vKgT, vChunk);     // [ CS, S_k, 1, H_v * n_seqs]

    // [CS, S_v, 1, H_v * n_seqs]
    ggml_tensor *vVTP = ggml_mul_mat(vCtx0, vChKCd, pS);
    mCb(vVTP, "v_prime", pIl);

    // [CS, S_v, 1, H_v * n_seqs]
    ggml_tensor *vVTNew = ggml_sub(vCtx0, vChVT, vVTP);
    mCb(vVTNew, "v_t_new", pIl);

    // [S_v, CS, 1, H_v * n_seqs]
    ggml_tensor *vVAttn = ggml_mul_mat(vCtx0, vVTNew, vChKq);
    mCb(vVAttn, "v_attn", pIl);

    // [S_v, CS, 1, H_v * n_seqs]
    ggml_tensor *vAttnInter = ggml_mul_mat(vCtx0, pS, vChQGExp);
    mCb(vAttnInter, "attn_inter", pIl);

    // [S_v, CS, 1, H_v * n_seqs]
    ggml_tensor *vOCh = ggml_add(vCtx0, vAttnInter, vVAttn);
    mCb(vOCh, "dnet_add_ch_attn_out", pIl);

    pV = ggml_set_inplace(vCtx0, pV, vOCh, pV->nb[1], pV->nb[2], pV->nb[3], vChunk * pV->nb[2]);

    // kgdmulvnew = (key_gdiff).transpose(-1, -2) @ v_new
    // TODO: head broadcast might not work here - probably will need a transpose
    ggml_tensor *vKgv = ggml_mul_mat(vCtx0, vChKgT, vVTNew); // [S_k, S_v, 1, H_k * n_seqs]

    // last_recurrent_state = last_recurrent_state * g_last + kgdmulvnew
    ggml_tensor *vChGLastExpT = fGetSlice2d(vCtx0, vGLastExpT, vChunk);

    pS = ggml_mul(vCtx0, pS, vChGLastExpT);
    pS = ggml_add(vCtx0, pS, vKgv);
    mCb(pS, "dnet_add_ch_state", pIl);
  }

  // truncate padded tokens
  ggml_tensor *vO = ggml_view_4d(vCtx0, pV, cSV, cNTokens, cHV, cNSeqs, ggml_row_size(pV->type, cSV),
                                 ggml_row_size(pV->type, cSV * cCs * cNChunks),
                                 ggml_row_size(pV->type, cSV * cCs * cNChunks * cHV), 0);
  vO = ggml_permute(vCtx0, vO, 0, 2, 1, 3); // [S_v, H_v, n_tokens, n_seqs]
  pS = ggml_reshape_4d(vCtx0, pS, cSV, cSV, cHV, cNSeqs);
  mCb(pS, "output_state", pIl);

  return {vO, pS};
}

std::pair<ggml_tensor *, ggml_tensor *> llm_build_delta_net_base::mBuildDeltaNetAutoregressive(ggml_tensor *pQ,
                                                                                               ggml_tensor *pK,
                                                                                               ggml_tensor *pV,
                                                                                               ggml_tensor *pG,
                                                                                               ggml_tensor *pB, // beta
                                                                                               ggml_tensor *pS, // state
                                                                                               int pIl) {
  const int64_t cSK = pQ->ne[0];
  const int64_t cHK = pQ->ne[1];
  const int64_t cNTokens = pQ->ne[2];
  const int64_t cNSeqs = pQ->ne[3];

  const int64_t cSV = pV->ne[0];
  const int64_t cHV = pV->ne[1];

  GGML_ASSERT(cNTokens == 1);

  GGML_ASSERT(cSK == cSV);
  GGML_ASSERT(cHV % cHK == 0);

  GGML_ASSERT(pQ->ne[0] == cSK && pQ->ne[1] == cHK && pQ->ne[2] == cNTokens && pQ->ne[3] == cNSeqs);
  GGML_ASSERT(pK->ne[0] == cSK && pK->ne[1] == cHK && pK->ne[2] == cNTokens && pK->ne[3] == cNSeqs);
  GGML_ASSERT(pV->ne[0] == cSV && pV->ne[1] == cHV && pV->ne[2] == cNTokens && pV->ne[3] == cNSeqs);

  GGML_ASSERT(pG->ne[0] == 1 || pG->ne[0] == cSV);
  GGML_ASSERT(pG->ne[1] == cHV && pG->ne[2] == cNTokens && pG->ne[3] == cNSeqs);
  GGML_ASSERT(pB->ne[0] == 1 && pB->ne[1] == cHV && pB->ne[2] == cNTokens && pB->ne[3] == cNSeqs);
  GGML_ASSERT(pS->ne[0] == cSV && pS->ne[1] == cSV && pS->ne[2] == cHV && pS->ne[3] == cNSeqs);

  const float cScale = 1.0f / sqrtf(cSK);

  pQ = ggml_scale(vCtx0, pQ, cScale);

  pQ = ggml_permute(vCtx0, pQ, 0, 2, 1, 3); // [S_k, n_tokens, H_k, n_seqs]
  pK = ggml_permute(vCtx0, pK, 0, 2, 1, 3); // [S_k, n_tokens, H_k, n_seqs]
  pV = ggml_permute(vCtx0, pV, 0, 2, 1, 3); // [S_v, n_tokens, H_v, n_seqs]

  mCb(pQ, "q_in", pIl);
  mCb(pK, "k_in", pIl);
  mCb(pV, "v_in", pIl);
  mCb(pB, "b_in", pIl);
  mCb(pG, "g_in", pIl);

  // GDA: [1,  1,  H_v, n_seqs]
  // KDA: [1, S_k, H_v, n_seqs]
  pG = ggml_reshape_4d(vCtx0, pG, 1, pG->ne[0], cHV, cNSeqs);
  pB = ggml_reshape_4d(vCtx0, pB, 1, 1, cHV, cNSeqs);

  // [S_v, S_v, H_v, n_seqs]
  pG = ggml_exp(vCtx0, pG);
  pS = ggml_mul(vCtx0, pS, pG);

  // [1, S_v, H_v, n_seqs]
  ggml_tensor *vSk;
  vSk = ggml_mul(vCtx0, pS, pK);
  vSk = ggml_sum_rows(vCtx0, vSk);

  // [S_v, 1, H_v, n_seqs]
  ggml_tensor *vD;
  vD = ggml_sub(vCtx0, pV, ggml_transpose(vCtx0, vSk));
  vD = ggml_mul(vCtx0, vD, pB);

  // [1, S_v, H_v, n_seqs]
  ggml_tensor *vDT;
  vDT = ggml_transpose(vCtx0, vD);

  // [S_v, S_v, H_v, n_seqs]
  ggml_tensor *vKd;
  pK = ggml_repeat(vCtx0, pK, pS);
  vKd = ggml_mul(vCtx0, pK, vDT);

  pS = ggml_add(vCtx0, pS, vKd);

  mCb(pS, "dnet_add_ar_state", pIl);

  ggml_tensor *vSQ = ggml_mul(vCtx0, pS, pQ);
  ggml_tensor *vO = ggml_sum_rows(vCtx0, vSQ);

  vO = ggml_permute(vCtx0, vO, 2, 0, 1, 3); // [S_v, H_v, n_tokens, n_seqs]

  return {vO, pS};
}

std::pair<ggml_tensor *, ggml_tensor *> llm_build_delta_net_base::mBuildDeltaNetFused(ggml_tensor *pQ, ggml_tensor *pK,
                                                                                      ggml_tensor *pV, ggml_tensor *pG,
                                                                                      ggml_tensor *pB, ggml_tensor *pS,
                                                                                      int pIl) {
  const int64_t cSK = pQ->ne[0];
  const int64_t cHK = pQ->ne[1];
  const int64_t cNTokens = pQ->ne[2];
  const int64_t cNSeqs = pQ->ne[3];

  const int64_t cSV = pV->ne[0];
  const int64_t cHV = pV->ne[1];

  GGML_ASSERT(cSK == cSV);
  GGML_ASSERT(cHV % cHK == 0);

  GGML_ASSERT(pQ->ne[0] == cSK && pQ->ne[1] == cHK && pQ->ne[2] == cNTokens && pQ->ne[3] == cNSeqs);
  GGML_ASSERT(pK->ne[0] == cSK && pK->ne[1] == cHK && pK->ne[2] == cNTokens && pK->ne[3] == cNSeqs);
  GGML_ASSERT(pV->ne[0] == cSV && pV->ne[1] == cHV && pV->ne[2] == cNTokens && pV->ne[3] == cNSeqs);

  GGML_ASSERT(pG->ne[0] == 1 || pG->ne[0] == cSV);
  GGML_ASSERT(pG->ne[1] == cHV && pG->ne[2] == cNTokens && pG->ne[3] == cNSeqs);
  GGML_ASSERT(pB->ne[0] == 1 && pB->ne[1] == cHV && pB->ne[2] == cNTokens && pB->ne[3] == cNSeqs);
  GGML_ASSERT(pS->ne[0] == cSV && pS->ne[1] == cSV && pS->ne[2] == cHV && pS->ne[3] == cNSeqs);

  // K=1: output carries the final state only. state s is 4D [S_v, S_v, H_v, n_seqs].
  ggml_tensor *vResult = ggml_gated_delta_net(vCtx0, pQ, pK, pV, pG, pB, pS, /*K=*/1);
  if (cNTokens == 1) {
    vRes->mAddFusedNode({cLlmFusedOpGdnAr, vResult, pIl});
  } else {
    vRes->mAddFusedNode({cLlmFusedOpGdnCh, vResult, pIl});
  }

  ggml_tensor *vOutput =
      ggml_view_4d(vCtx0, vResult, cSV, cHV, cNTokens, cNSeqs, ggml_row_size(vResult->type, cSV),
                   ggml_row_size(vResult->type, cSV * cHV), ggml_row_size(vResult->type, cSV * cHV * cNTokens), 0);

  ggml_tensor *vNewState = ggml_view_4d(
      vCtx0, vResult, cSV, cSV, cHV, cNSeqs, ggml_row_size(vResult->type, cSV), ggml_row_size(vResult->type, cSV * cSV),
      ggml_row_size(vResult->type, cSV * cSV * cHV), ggml_row_size(vResult->type, cSV * cHV * cNTokens * cNSeqs));

  return {vOutput, vNewState};
}

std::pair<ggml_tensor *, ggml_tensor *> llm_build_delta_net_base::mBuildDeltaNet(ggml_tensor *pQ, ggml_tensor *pK,
                                                                                 ggml_tensor *pV, ggml_tensor *pG,
                                                                                 ggml_tensor *pB, ggml_tensor *pS,
                                                                                 int pIl) {
  const int64_t cNSeqTokens = pQ->ne[2];

  if (cNSeqTokens == 1) {
    if (vCparams.vFusedGdnAr) {
      return mBuildDeltaNetFused(pQ, pK, pV, pG, pB, pS, pIl);
    }
    return mBuildDeltaNetAutoregressive(pQ, pK, pV, pG, pB, pS, pIl);
  }

  if (vCparams.vFusedGdnCh) {
    return mBuildDeltaNetFused(pQ, pK, pV, pG, pB, pS, pIl);
  }

  return mBuildDeltaNetChunking(pQ, pK, pV, pG, pB, pS, pIl);
}

ggml_tensor *llm_build_delta_net_base::mBuildConvState(llm_graph_input_rs *pInp, ggml_tensor *pConvStatesAll,
                                                       ggml_tensor *pQkvMixed, int64_t pConvKernelSize,
                                                       int64_t pConvChannels, int pIl) {
  const auto *vMctxCur = pInp->vMctx;

  const auto cKvHead = vMctxCur->mGetHead();
  const auto cMemSize = vMctxCur->mGetSize();

  const int64_t cNSeqs = vUbatch.vNSeqs;

  ggml_tensor *vConvStates = mBuildRs(pInp, pConvStatesAll, vHparams.mNEmbdR(), cNSeqs);
  mCb(vConvStates, "conv_states", pIl);

  vConvStates = ggml_reshape_3d(vCtx0, vConvStates, pConvKernelSize - 1, pConvChannels, cNSeqs);
  mCb(vConvStates, "conv_states_reshaped", pIl);

  pQkvMixed = ggml_transpose(vCtx0, pQkvMixed);
  mCb(pQkvMixed, "qkv_mixed_transposed", pIl);

  ggml_tensor *vConvInput = ggml_concat(vCtx0, vConvStates, pQkvMixed, 0);
  mCb(vConvInput, "conv_input", pIl);

  const int64_t cRowCount = (pConvKernelSize - 1) * pConvChannels;

  const size_t cRowSize = ggml_row_size(pConvStatesAll->type, cRowCount);

  if (vCparams.vNRsSeq == 0) {
    const int64_t cSIdx = vConvInput->ne[0] - vConvStates->ne[0];
    const int64_t cSSlot = 0;

    ggml_tensor *vConvStateLast =
        ggml_view_3d(vCtx0, vConvInput, pConvKernelSize - 1, pConvChannels, cNSeqs, vConvInput->nb[1],
                     vConvInput->nb[2], ggml_row_size(vConvInput->type, cSIdx));
    mCb(vConvStateLast, "conv_state_last", pIl);

    ggml_tensor *vConvStateUpdate = ggml_view_2d(vCtx0, pConvStatesAll, cRowCount, cNSeqs, pConvStatesAll->nb[1],
                                                 (cSSlot * cMemSize + cKvHead) * cRowSize);
    mCb(vConvStateUpdate, "conv_state_update", pIl);

    ggml_build_forward_expand(vGf, ggml_cpy(vCtx0, vConvStateLast, vConvStateUpdate));
  } else {
    // [TAG_RECURRENT_ROLLBACK_SPLITS]
    // this logic assumes that the last (n_rs_seq + 1) tokens of a sequence in a batch are inside
    //   the same ubatch, which `mSplitEqual()` guarantees via its pNKeepTail argument

    const int64_t cK = (int64_t)vCparams.vNRsSeq + 1;

    for (int64_t vT = 1; vT <= cK; ++vT) {
      const int64_t cSIdx = std::max<int64_t>(0, vConvInput->ne[0] - vConvStates->ne[0] - cK + vT);
      const int64_t cSSlot = cK - vT;

      ggml_tensor *vConvStateLast =
          ggml_view_3d(vCtx0, vConvInput, pConvKernelSize - 1, pConvChannels, cNSeqs, vConvInput->nb[1],
                       vConvInput->nb[2], ggml_row_size(vConvInput->type, cSIdx));

      ggml_tensor *vConvStateUpdate = ggml_view_2d(vCtx0, pConvStatesAll, cRowCount, cNSeqs, pConvStatesAll->nb[1],
                                                   (cSSlot * cMemSize + cKvHead) * cRowSize);

      ggml_build_forward_expand(vGf, ggml_cpy(vCtx0, vConvStateLast, vConvStateUpdate));
    }
  }

  return vConvInput;
}

ggml_tensor *llm_build_delta_net_base::mBuildRecurrentAttn(llm_graph_input_rs *pInp, ggml_tensor *pSsmStatesAll,
                                                           ggml_tensor *pQ, ggml_tensor *pK, ggml_tensor *pV,
                                                           ggml_tensor *pG, ggml_tensor *pB, ggml_tensor *pS, int pIl) {
  const auto *vMctxCur = pInp->vMctx;
  const auto cKvHead = vMctxCur->mGetHead();
  const uint32_t cMemSize = vMctxCur->mGetSize();

  const int64_t cSV = pS->ne[0];
  const int64_t cHV = pS->ne[2];
  const int64_t cNSeqs = pS->ne[3];
  const int64_t cNSeqTokens = pQ->ne[2];

  const bool cKeep = vCparams.vNRsSeq > 0;

  if (!cKeep) {
    auto vAttnOut = mBuildDeltaNet(pQ, pK, pV, pG, pB, pS, pIl);
    ggml_tensor *vOutput = vAttnOut.first;
    ggml_tensor *vNewState = vAttnOut.second;
    mCb(vOutput, "attn_output", pIl);
    mCb(vNewState, "new_state", pIl);

    ggml_build_forward_expand(
        vGf, ggml_cpy(vCtx0, vNewState,
                      ggml_view_2d(vCtx0, pSsmStatesAll, vHparams.mNEmbdS(), cNSeqs, pSsmStatesAll->nb[1],
                                   cKvHead * vHparams.mNEmbdS() * ggml_element_size(pSsmStatesAll))));

    return vOutput;
  }

  const int64_t cD = cSV * cSV * cHV;
  const int64_t cK = vCparams.vNRsSeq + 1;

  // state s is 4D [S_v, S_v, H_v, n_seqs]; K snapshot slots are written into the output.
  ggml_tensor *vGdnOut = ggml_gated_delta_net(vCtx0, pQ, pK, pV, pG, pB, pS, cK);
  if (cNSeqTokens > 1) {
    vRes->mAddFusedNode({cLlmFusedOpGdnCh, vGdnOut, pIl});
  } else {
    vRes->mAddFusedNode({cLlmFusedOpGdnAr, vGdnOut, pIl});
  }

  const int64_t cAttnScoreElems = cSV * cHV * cNSeqTokens * cNSeqs;
  const int64_t cStateSizePerSnap = cSV * cSV * cHV * cNSeqs;

  ggml_tensor *vOutput =
      ggml_view_4d(vCtx0, vGdnOut, cSV, cHV, cNSeqTokens, cNSeqs, ggml_row_size(vGdnOut->type, cSV),
                   ggml_row_size(vGdnOut->type, cSV * cHV), ggml_row_size(vGdnOut->type, cSV * cHV * cNSeqTokens), 0);
  mCb(vOutput, "attn_output", pIl);

  const size_t cRowSize = vHparams.mNEmbdS() * ggml_element_size(pSsmStatesAll);

  // op writes the last min(n_seq_tokens, K) snapshots; trailing slots are left unwritten
  const int64_t cNWritten = std::min<int64_t>(cNSeqTokens, cK);

  // write the produced snapshots into the recurrent cache (snapshot slot i -> rollback group i)
  ggml_tensor *vSrc =
      ggml_view_3d(vCtx0, vGdnOut, cD, cNSeqs, cNWritten, ggml_row_size(vGdnOut->type, cD),
                   ggml_row_size(vGdnOut->type, cStateSizePerSnap), ggml_row_size(vGdnOut->type, cAttnScoreElems));

  ggml_tensor *vDst = ggml_view_3d(vCtx0, pSsmStatesAll, cD, cNSeqs, cNWritten, pSsmStatesAll->nb[1],
                                   (size_t)cMemSize * cRowSize, (size_t)cKvHead * cRowSize);

  ggml_build_forward_expand(vGf, ggml_cpy(vCtx0, vSrc, vDst));

  return vOutput;
}
