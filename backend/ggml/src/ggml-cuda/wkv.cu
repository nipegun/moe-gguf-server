#include "common.cuh"
#include "wkv.cuh"

template <int block_size>
static __global__ void fRwkvWkvF32(const int pB, const int pT, const int pC, const int pH, const float *pK,
                                   const float *pV, const float *pR, const float *pTf, const float *pTd,
                                   const float *pS, float *pDst) {
  const int cTid = threadIdx.x;
  const int cBid = blockIdx.x;

  const int cHeadSize = block_size;
  const int cBatchI = cBid / pH;
  const int cHeadI = cBid % pH;
  const int cStateSize = pC * cHeadSize;
  const int cNSeqTokens = pT / pB;

  float aState[cHeadSize];
  __shared__ float _k[cHeadSize], _r[cHeadSize], _tf[cHeadSize], _td[cHeadSize];

#pragma unroll
  for (int vI = 0; vI < cHeadSize; vI++) {
    aState[vI] = pS[cBatchI * cStateSize + cHeadI * cHeadSize * cHeadSize + vI * cHeadSize + cTid];
  }

  __syncthreads();
  _tf[cTid] = pTf[cHeadI * cHeadSize + cTid];
  __syncthreads();

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
      const float4 &tf = (float4 &)(_tf[vJ]);
      const float4 &td = (float4 &)(_td[vJ]);
      float4 &vS = (float4 &)(aState[vJ]);
      float4 vKv;

      vKv.x = k.x * cV;
      vKv.y = k.y * cV;
      vKv.z = k.z * cV;
      vKv.w = k.w * cV;

      vY += r.x * (tf.x * vKv.x + vS.x);
      vY += r.y * (tf.y * vKv.y + vS.y);
      vY += r.z * (tf.z * vKv.z + vS.z);
      vY += r.w * (tf.w * vKv.w + vS.w);

      vS.x = vS.x * td.x + vKv.x;
      vS.y = vS.y * td.y + vKv.y;
      vS.z = vS.z * td.z + vKv.z;
      vS.w = vS.w * td.w + vKv.w;
    }
    pDst[vT] = vY;
  }

#pragma unroll
  for (int vI = 0; vI < cHeadSize; vI++) {
    pDst[pT * pC + cBatchI * cStateSize + cHeadI * cHeadSize * cHeadSize + vI * cHeadSize + cTid] = aState[vI];
  }
}

template <int block_size>
static __global__ void fRwkvWkv7F32(const int B, const int pT, const int pC, const int pH, const float *pR,
                                    const float *pW, const float *pK, const float *pV, const float *pA, const float *b,
                                    const float *pS, float *pDst) {
  const int cTid = threadIdx.x;
  const int cBid = blockIdx.x;

  const int cHeadSize = block_size;
  const int cBatchI = cBid / pH;
  const int cHeadI = cBid % pH;
  const int cStateSize = pC * cHeadSize;
  const int cNSeqTokens = pT / B;

  float aState[cHeadSize];
  __shared__ float _r[cHeadSize], _w[cHeadSize], _k[cHeadSize], _a[cHeadSize], _b[cHeadSize];

#pragma unroll
  for (int vI = 0; vI < cHeadSize; vI++) {
    aState[vI] = pS[cBatchI * cStateSize + cHeadI * cHeadSize * cHeadSize + cTid * cHeadSize + vI];
  }

  for (int vT = cBatchI * cNSeqTokens * pC + cHeadI * cHeadSize + cTid;
       vT < (cBatchI + 1) * cNSeqTokens * pC + cHeadI * cHeadSize + cTid; vT += pC) {
    __syncthreads();
    _r[cTid] = pR[vT];
    _w[cTid] = pW[vT];
    _k[cTid] = pK[vT];
    _a[cTid] = pA[vT];
    _b[cTid] = b[vT];
    __syncthreads();

    float vSa = 0;
#pragma unroll
    for (int vJ = 0; vJ < cHeadSize; vJ += 4) {
      const float4 &a = (float4 &)(_a[vJ]);
      const float4 &vS = (float4 &)(aState[vJ]);
      vSa += a.x * vS.x;
      vSa += a.y * vS.y;
      vSa += a.z * vS.z;
      vSa += a.w * vS.w;
    }

    const float cV = pV[vT];
    float vY = 0;
    for (int vJ = 0; vJ < cHeadSize; vJ += 4) {
      const float4 &r = (float4 &)(_r[vJ]);
      const float4 &w = (float4 &)(_w[vJ]);
      const float4 &k = (float4 &)(_k[vJ]);
      const float4 &b = (float4 &)(_b[vJ]);
      float4 &vS = (float4 &)(aState[vJ]);
      float4 vKv;

      vKv.x = k.x * cV;
      vKv.y = k.y * cV;
      vKv.z = k.z * cV;
      vKv.w = k.w * cV;

      vS.x = vS.x * w.x + vKv.x + vSa * b.x;
      vS.y = vS.y * w.y + vKv.y + vSa * b.y;
      vS.z = vS.z * w.z + vKv.z + vSa * b.z;
      vS.w = vS.w * w.w + vKv.w + vSa * b.w;

      vY += vS.x * r.x;
      vY += vS.y * r.y;
      vY += vS.z * r.z;
      vY += vS.w * r.w;
    }
    pDst[vT] = vY;
  }

#pragma unroll
  for (int vI = 0; vI < cHeadSize; vI++) {
    pDst[pT * pC + cBatchI * cStateSize + cHeadI * cHeadSize * cHeadSize + cTid * cHeadSize + vI] = aState[vI];
  }
}

void ggml_cuda_op_rwkv_wkv6(ggml_backend_cuda_context &pCtx, ggml_tensor *pDst) {
  const float *vKD = (const float *)pDst->src[0]->data;
  const float *vVD = (const float *)pDst->src[1]->data;
  const float *vRD = (const float *)pDst->src[2]->data;
  const float *vTfD = (const float *)pDst->src[3]->data;
  const float *vTdD = (const float *)pDst->src[4]->data;
  const float *vSD = (const float *)pDst->src[5]->data;

  const int64_t cB = pDst->src[5]->ne[1];
  const int64_t cT = pDst->src[0]->ne[2];
  const int64_t cC = pDst->ne[0];
  const int64_t cH = pDst->src[0]->ne[1];

  float *vDstD = (float *)pDst->data;

  cudaStream_t vStream = pCtx.stream();

  GGML_ASSERT(pDst->src[5]->type == GGML_TYPE_F32);
  GGML_ASSERT(cC % cH == 0);
  GGML_ASSERT(cC / cH == CUDA_WKV_BLOCK_SIZE || cC / cH == CUDA_WKV_BLOCK_SIZE * 2);

  if (cC / cH == CUDA_WKV_BLOCK_SIZE) {
    fRwkvWkvF32<CUDA_WKV_BLOCK_SIZE>
        <<<cB * cH, cC / cH, 0, vStream>>>(cB, cT, cC, cH, vKD, vVD, vRD, vTfD, vTdD, vSD, vDstD);
  } else {
    fRwkvWkvF32<CUDA_WKV_BLOCK_SIZE * 2>
        <<<cB * cH, cC / cH, 0, vStream>>>(cB, cT, cC, cH, vKD, vVD, vRD, vTfD, vTdD, vSD, vDstD);
  }
}

void ggml_cuda_op_rwkv_wkv7(ggml_backend_cuda_context &pCtx, ggml_tensor *pDst) {
  const float *vRD = (const float *)pDst->src[0]->data;
  const float *vWD = (const float *)pDst->src[1]->data;
  const float *vKD = (const float *)pDst->src[2]->data;
  const float *vVD = (const float *)pDst->src[3]->data;
  const float *vAD = (const float *)pDst->src[4]->data;
  const float *vBD = (const float *)pDst->src[5]->data;
  const float *vSD = (const float *)pDst->src[6]->data;

  const int64_t cB = pDst->src[6]->ne[1];
  const int64_t cT = pDst->src[0]->ne[2];
  const int64_t cC = pDst->ne[0];
  const int64_t cH = pDst->src[0]->ne[1];

  float *vDstD = (float *)pDst->data;

  cudaStream_t vStream = pCtx.stream();

  GGML_ASSERT(pDst->src[6]->type == GGML_TYPE_F32);
  GGML_ASSERT(cC % cH == 0);
  GGML_ASSERT(cC / cH == CUDA_WKV_BLOCK_SIZE || cC / cH == CUDA_WKV_BLOCK_SIZE * 2);

  if (cC / cH == CUDA_WKV_BLOCK_SIZE) {
    fRwkvWkv7F32<CUDA_WKV_BLOCK_SIZE>
        <<<cB * cH, cC / cH, 0, vStream>>>(cB, cT, cC, cH, vRD, vWD, vKD, vVD, vAD, vBD, vSD, vDstD);
  } else {
    fRwkvWkv7F32<CUDA_WKV_BLOCK_SIZE * 2>
        <<<cB * cH, cC / cH, 0, vStream>>>(cB, cT, cC, cH, vRD, vWD, vKD, vVD, vAD, vBD, vSD, vDstD);
  }
}
