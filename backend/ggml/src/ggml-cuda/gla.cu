#include "common.cuh"
#include "gla.cuh"

template <int HEAD_SIZE>
static __global__ void fGatedLinearAttnF32(const int pB, const int pT, const int pC, const int pH, const float pScale,
                                           const float *pK, const float *pV, const float *pR, const float *pTd,
                                           const float *pS, float *pDst) {
  const int cTid = threadIdx.x;
  const int cBid = blockIdx.x;

  const int cHeadSize = HEAD_SIZE;
  const int cBatchI = cBid / pH;
  const int cHeadI = cBid % pH;
  const int cStateSize = pC * cHeadSize;
  const int cNSeqTokens = pT / pB;

  float aState[cHeadSize];
  __shared__ float _k[cHeadSize], _r[cHeadSize], _td[cHeadSize];

#pragma unroll
  for (int vI = 0; vI < cHeadSize; vI++) {
    aState[vI] = pS[cBatchI * cStateSize + cHeadI * cHeadSize * cHeadSize + vI * cHeadSize + cTid];
  }

  for (int vT = cBatchI * cNSeqTokens * pC + cHeadI * cHeadSize + cTid;
       vT < (cBatchI + 1) * cNSeqTokens * pC + cHeadI * cHeadSize + cTid; vT += pC) {
    __syncthreads();
    _k[cTid] = pK[vT];
    _r[cTid] = pR[vT];
    _td[cTid] = pTd[vT];
    __syncthreads();

    const float cV = pV[vT];
    float vY = 0;
    for (int vJ = 0; vJ < cHeadSize; vJ += 4) {
      const float4 &k = (float4 &)(_k[vJ]);
      const float4 &r = (float4 &)(_r[vJ]);
      const float4 &td = (float4 &)(_td[vJ]);
      float4 &vS = (float4 &)(aState[vJ]);
      float4 vKv;

      vKv.x = k.x * cV;
      vKv.y = k.y * cV;
      vKv.z = k.z * cV;
      vKv.w = k.w * cV;

      vS.x = vS.x * td.x + vKv.x;
      vS.y = vS.y * td.y + vKv.y;
      vS.z = vS.z * td.z + vKv.z;
      vS.w = vS.w * td.w + vKv.w;

      vY += r.x * vS.x;
      vY += r.y * vS.y;
      vY += r.z * vS.z;
      vY += r.w * vS.w;
    }
    pDst[vT] = vY * pScale;
  }

#pragma unroll
  for (int vI = 0; vI < cHeadSize; vI++) {
    pDst[pT * pC + cBatchI * cStateSize + cHeadI * cHeadSize * cHeadSize + vI * cHeadSize + cTid] = aState[vI];
  }
}

void ggml_cuda_op_gated_linear_attn(ggml_backend_cuda_context &pCtx, ggml_tensor *pDst) {
  const float *vKD = (const float *)pDst->src[0]->data;
  const float *vVD = (const float *)pDst->src[1]->data;
  const float *vRD = (const float *)pDst->src[2]->data;
  const float *vTdD = (const float *)pDst->src[3]->data;
  const float *vSD = (const float *)pDst->src[4]->data;

  const int64_t cB = pDst->src[4]->ne[1];
  const int64_t cT = pDst->src[0]->ne[2];
  const int64_t cC = pDst->ne[0];
  const int64_t cH = pDst->src[0]->ne[1];

  float vScale;
  memcpy(&vScale, (float *)pDst->op_params, sizeof(float));

  float *vDstD = (float *)pDst->data;

  cudaStream_t vStream = pCtx.stream();

  GGML_ASSERT(pDst->src[4]->type == GGML_TYPE_F32);
  GGML_ASSERT(cC % cH == 0);
  GGML_ASSERT(cC / cH == 64 || cC / cH == 128);

  if (cC / cH == 64) {
    fGatedLinearAttnF32<64><<<cB * cH, cC / cH, 0, vStream>>>(cB, cT, cC, cH, vScale, vKD, vVD, vRD, vTdD, vSD, vDstD);
  } else {
    fGatedLinearAttnF32<128><<<cB * cH, cC / cH, 0, vStream>>>(cB, cT, cC, cH, vScale, vKD, vVD, vRD, vTdD, vSD, vDstD);
  }
}
