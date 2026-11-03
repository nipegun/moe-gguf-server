#include "im2col.cuh"

#define MAX_GRIDDIM_Y 65535
#define MAX_GRIDDIM_Z 65535

template <typename T>
static __global__ void fIm2colKernel(const float *pX, T *pDst, int64_t pIc, int64_t pIw, int64_t pIh, int64_t pOh,
                                     int64_t pOw, int64_t pKw, int64_t pKh, int64_t pIcIhIw, int64_t pIhIw,
                                     int64_t pNOh, int64_t pKhKw, int64_t pIcKhKw, int pS0, int pS1, int p0, int p1,
                                     int pD0, int pD1) {
  const int64_t cI = threadIdx.x + blockIdx.x * blockDim.x;
  if (cI >= pIcKhKw) {
    return;
  }

  const int64_t cIic = cI / (pKhKw);
  const int64_t cRem = cI - cIic * pKhKw;
  const int64_t cIkh = cRem / pKw;
  const int64_t cIkw = cRem - cIkh * pKw;

  for (int64_t vIow = blockIdx.y; vIow < pOw; vIow += MAX_GRIDDIM_Y) {
    for (int64_t vIz = blockIdx.z; vIz < pNOh; vIz += MAX_GRIDDIM_Z) {
      const int64_t cIn = vIz / pOh;
      const int64_t cIoh = vIz - cIn * pOh;

      const int64_t cIiw = vIow * pS0 + cIkw * pD0 - p0;
      const int64_t cIih = cIoh * pS1 + cIkh * pD1 - p1;

      const int64_t cOffsetDst = ((cIn * pOh + cIoh) * pOw + vIow) * pIcKhKw + cIic * pKhKw + cIkh * pKw + cIkw;

      if (cIih < 0 || cIih >= pIh || cIiw < 0 || cIiw >= pIw) {
        pDst[cOffsetDst] = 0.0f;
      } else {
        const int64_t cOffsetSrc = cIic * pIcIhIw + cIn * pIhIw;
        pDst[cOffsetDst] = pX[cOffsetSrc + cIih * pIw + cIiw];
      }
    }
  }

  GGML_UNUSED(pIc);
  GGML_UNUSED(pKh);
}

// im2col: [N, IC, IH, IW] => [N, OH, OW, IC*KH*KW]
template <typename T>
static void fIm2colCuda(const float *pX, T *pDst, int64_t pIw, int64_t pIh, int64_t pOw, int64_t pOh, int64_t pKw,
                        int64_t pKh, int64_t pIc, int64_t pN, int64_t pIcIhIw, int64_t pIhIw, int pS0, int pS1, int p0,
                        int p1, int pD0, int pD1, cudaStream_t pStream) {
  const int64_t cIcKhKw = pIc * pKh * pKw;
  const int64_t cNumBlocks = (cIcKhKw + CUDA_IM2COL_BLOCK_SIZE - 1) / CUDA_IM2COL_BLOCK_SIZE;
  const int64_t cNOh = pN * pOh;
  const int64_t cKhKw = pKw * pKh;
  dim3 vBlockNums(cNumBlocks, MIN(pOw, MAX_GRIDDIM_Y), MIN(cNOh, MAX_GRIDDIM_Z));
  fIm2colKernel<<<vBlockNums, MIN(cIcKhKw, CUDA_IM2COL_BLOCK_SIZE), 0, pStream>>>(
      pX, pDst, pIc, pIw, pIh, pOh, pOw, pKw, pKh, pIcIhIw, pIhIw, cNOh, cKhKw, cIcKhKw, pS0, pS1, p0, p1, pD0, pD1);
}

static void fIm2colCudaF16(const float *pX, half *pDst, int64_t pIw, int64_t pIh, int64_t pOw, int64_t pOh, int64_t pKw,
                           int64_t pKh, int64_t pIc, int64_t pN, int64_t pIcIhIw, int64_t pIhIw, int pS0, int pS1,
                           int p0, int p1, int pD0, int pD1, cudaStream_t pStream) {

  fIm2colCuda<half>(pX, pDst, pIw, pIh, pOw, pOh, pKw, pKh, pIc, pN, pIcIhIw, pIhIw, pS0, pS1, p0, p1, pD0, pD1,
                    pStream);
}

static void fIm2colCudaF32(const float *pX, float *pDst, int64_t pIw, int64_t pIh, int64_t pOw, int64_t pOh,
                           int64_t pKw, int64_t pKh, int64_t pIc, int64_t pN, int64_t pIcIhIw, int64_t pIhIw, int pS0,
                           int pS1, int p0, int p1, int pD0, int pD1, cudaStream_t pStream) {

  fIm2colCuda<float>(pX, pDst, pIw, pIh, pOw, pOh, pKw, pKh, pIc, pN, pIcIhIw, pIhIw, pS0, pS1, p0, p1, pD0, pD1,
                     pStream);
}

void ggml_cuda_op_im2col(ggml_backend_cuda_context &pCtx, ggml_tensor *pDst) {
  const ggml_tensor *vSrc0 = pDst->src[0];
  const ggml_tensor *vSrc1 = pDst->src[1];
  const float *vSrc1D = (const float *)vSrc1->data;
  float *vDstD = (float *)pDst->data;
  cudaStream_t vStream = pCtx.stream();

  GGML_ASSERT(vSrc1->type == GGML_TYPE_F32);
  GGML_ASSERT(pDst->type == GGML_TYPE_F16 || pDst->type == GGML_TYPE_F32);

  const int32_t cS0 = ((const int32_t *)(pDst->op_params))[0];
  const int32_t cS1 = ((const int32_t *)(pDst->op_params))[1];
  const int32_t cP0 = ((const int32_t *)(pDst->op_params))[2];
  const int32_t cP1 = ((const int32_t *)(pDst->op_params))[3];
  const int32_t cD0 = ((const int32_t *)(pDst->op_params))[4];
  const int32_t cD1 = ((const int32_t *)(pDst->op_params))[5];

  const bool cIs2D = ((const int32_t *)(pDst->op_params))[6] == 1;

  const int64_t cIc = vSrc1->ne[cIs2D ? 2 : 1];
  const int64_t cIh = cIs2D ? vSrc1->ne[1] : 1;
  const int64_t cIw = vSrc1->ne[0];

  const int64_t cKh = cIs2D ? vSrc0->ne[1] : 1;
  const int64_t cKw = vSrc0->ne[0];

  const int64_t cOh = cIs2D ? pDst->ne[2] : 1;
  const int64_t cOw = pDst->ne[1];

  const int64_t cIcIhIw = vSrc1->nb[cIs2D ? 2 : 1] / 4; // nb is byte offset, src is type float32
  const int64_t cN = vSrc1->ne[cIs2D ? 3 : 2];
  const int64_t cIhIw = vSrc1->nb[cIs2D ? 3 : 2] / 4; // nb is byte offset, src is type float32

  if (pDst->type == GGML_TYPE_F16) {
    fIm2colCudaF16(vSrc1D, (half *)vDstD, cIw, cIh, cOw, cOh, cKw, cKh, cIc, cN, cIcIhIw, cIhIw, cS0, cS1, cP0, cP1,
                   cD0, cD1, vStream);
  } else {
    fIm2colCudaF32(vSrc1D, (float *)vDstD, cIw, cIh, cOw, cOh, cKw, cKh, cIc, cN, cIcIhIw, cIhIw, cS0, cS1, cP0, cP1,
                   cD0, cD1, vStream);
  }
}

// [N*IC, ID, IH, IW] => [N*OD, OH, OW, IC * KD * KH * KW]
template <typename T>
static __global__ void
fIm2col3dKernel(const float *pSrc, T *pDst, int64_t pN, int64_t pIc, int64_t pId, int64_t pIh, int64_t pIw, int64_t pOc,
                int64_t pKd, int64_t pKh, int64_t pKw, int64_t pOd, int64_t pOh, int64_t pOw, int64_t pOhOw,
                int64_t pKdKhKw, int64_t pIdIhIw, int64_t pKhKw, int64_t pIhIw, int64_t pIcIdIhIw, int64_t pIcKdKhKw,
                int64_t pOwKdKhKw, int64_t pOdOhOwIcKdKhKw, int64_t pOhOwIcKdKhKw, int64_t pOwIcKdKhKw, int64_t pNOdOh,
                int64_t pOdOh, int64_t pStrideQ, int64_t pStrideZ, int64_t pStrideY, int64_t pStrideX, int pS0, int pS1,
                int pS2, int p0, int p1, int p2, int pD0, int pD1, int pD2) {
  const int64_t cI = threadIdx.x + blockIdx.x * blockDim.x;
  if (cI >= pIcKdKhKw) {
    return;
  }
  GGML_UNUSED(pN);
  GGML_UNUSED(pOc);
  GGML_UNUSED(pOhOw);
  GGML_UNUSED(pOd);
  GGML_UNUSED(pOw);
  GGML_UNUSED(pKd);
  GGML_UNUSED(pKh);
  GGML_UNUSED(pIdIhIw);
  GGML_UNUSED(pIhIw);
  GGML_UNUSED(pIcIdIhIw);
  GGML_UNUSED(pOwKdKhKw);

  const int64_t cIic = cI / pKdKhKw;
  const int64_t cIkd = (cI - cIic * pKdKhKw) / pKhKw;
  const int64_t cIkh = (cI - cIic * pKdKhKw - cIkd * pKhKw) / pKw;
  const int64_t cIkw = cI % pKw;

  for (int64_t vIow = blockIdx.y; vIow < pOw; vIow += MAX_GRIDDIM_Y) {
    for (int64_t vIz = blockIdx.z; vIz < pNOdOh; vIz += MAX_GRIDDIM_Z) {
      const int64_t cIn = vIz / pOdOh;
      const int64_t cIod = (vIz - cIn * pOdOh) / pOh;
      const int64_t cIoh = vIz % pOh;

      const int64_t cIiw = vIow * pS0 + cIkw * pD0 - p0;
      const int64_t cIih = cIoh * pS1 + cIkh * pD1 - p1;
      const int64_t cIid = cIod * pS2 + cIkd * pD2 - p2;

      const int64_t cOffsetDst = cIn * pOdOhOwIcKdKhKw + cIod * pOhOwIcKdKhKw + cIoh * pOwIcKdKhKw + vIow * pIcKdKhKw +
                                 cIic * pKdKhKw + cIkd * pKhKw + cIkh * pKw + cIkw;

      if (cIih < 0 || cIih >= pIh || cIiw < 0 || cIiw >= pIw || cIid < 0 || cIid >= pId) {
        pDst[cOffsetDst] = 0.0f;
      } else {
        const int64_t cOffsetSrc =
            ((cIn * pIc + cIic) * pStrideQ) + (cIid * pStrideZ) + (cIih * pStrideY) + (cIiw * pStrideX);
        pDst[cOffsetDst] = pSrc[cOffsetSrc];
      }
    }
  }
}

// [N*IC, ID, IH, IW] => [N*OD, OH, OW, IC * KD * KH * KW]
template <typename T>
static void fIm2col3dCuda(const float *pSrc, T *pDst, int64_t pN, int64_t pIc, int64_t pId, int64_t pIh, int64_t pIw,
                          int64_t pOc, int64_t pKd, int64_t pKh, int64_t pKw, int64_t pOd, int64_t pOh, int64_t pOw,
                          int64_t pStrideQ, int64_t pStrideZ, int64_t pStrideY, int64_t pStrideX, int pS0, int pS1,
                          int pS2, int p0, int p1, int p2, int pD0, int pD1, int pD2, cudaStream_t pStream) {
  const int64_t cOhOw = pOh * pOw;
  const int64_t cKdKhKw = pKd * pKh * pKw;
  const int64_t cIdIhIw = pId * pIh * pIw;
  const int64_t cKhKw = pKh * pKw;
  const int64_t cIhIw = pIh * pIw;
  const int64_t cIcKdKhKw = pIc * pKd * pKh * pKw;
  const int64_t cOwKdKhKw = pOw * pKd * pKh * pKw;
  const int64_t cNOdOh = pN * pOd * pOh;
  const int64_t cOdOh = pOd * pOh;
  const int64_t cIcIdIhIw = pIc * pId * pIh * pIw;
  const int64_t cOdOhOwIcKdKhKw = pOd * pOh * pOw * pIc * pKd * pKh * pKw;
  const int64_t cOhOwIcKdKhKw = pOh * pOw * pIc * pKd * pKh * pKw;
  const int64_t cOwIcKdKhKw = pOw * pIc * pKd * pKh * pKw;
  const int64_t cNumBlocks = (cIcKdKhKw + CUDA_IM2COL_BLOCK_SIZE - 1) / CUDA_IM2COL_BLOCK_SIZE;
  dim3 vBlockNums(cNumBlocks, MIN(pOw, MAX_GRIDDIM_Y), MIN(cNOdOh, MAX_GRIDDIM_Z));
  fIm2col3dKernel<<<vBlockNums, MIN(cIcKdKhKw, CUDA_IM2COL_BLOCK_SIZE), 0, pStream>>>(
      pSrc, pDst, pN, pIc, pId, pIh, pIw, pOc, pKd, pKh, pKw, pOd, pOh, pOw, cOhOw, cKdKhKw, cIdIhIw, cKhKw, cIhIw,
      cIcIdIhIw, cIcKdKhKw, cOwKdKhKw, cOdOhOwIcKdKhKw, cOhOwIcKdKhKw, cOwIcKdKhKw, cNOdOh, cOdOh, pStrideQ, pStrideZ,
      pStrideY, pStrideX, pS0, pS1, pS2, p0, p1, p2, pD0, pD1, pD2);
}

static void fIm2col3dCudaF16(const float *pSrc, half *pDst, int64_t pN, int64_t pIc, int64_t pId, int64_t pIh,
                             int64_t pIw, int64_t pOc, int64_t pKd, int64_t pKh, int64_t pKw, int64_t pOd, int64_t pOh,
                             int64_t pOw, int64_t pStrideQ, int64_t pStrideZ, int64_t pStrideY, int64_t pStrideX,
                             int pS0, int pS1, int pS2, int p0, int p1, int p2, int pD0, int pD1, int pD2,
                             cudaStream_t pStream) {

  fIm2col3dCuda<half>(pSrc, pDst, pN, pIc, pId, pIh, pIw, pOc, pKd, pKh, pKw, pOd, pOh, pOw, pStrideQ, pStrideZ,
                      pStrideY, pStrideX, pS0, pS1, pS2, p0, p1, p2, pD0, pD1, pD2, pStream);
}

static void fIm2col3dCudaF32(const float *pSrc, float *pDst, int64_t pN, int64_t pIc, int64_t pId, int64_t pIh,
                             int64_t pIw, int64_t pOc, int64_t pKd, int64_t pKh, int64_t pKw, int64_t pOd, int64_t pOh,
                             int64_t pOw, int64_t pStrideQ, int64_t pStrideZ, int64_t pStrideY, int64_t pStrideX,
                             int pS0, int pS1, int pS2, int p0, int p1, int p2, int pD0, int pD1, int pD2,
                             cudaStream_t pStream) {

  fIm2col3dCuda<float>(pSrc, pDst, pN, pIc, pId, pIh, pIw, pOc, pKd, pKh, pKw, pOd, pOh, pOw, pStrideQ, pStrideZ,
                       pStrideY, pStrideX, pS0, pS1, pS2, p0, p1, p2, pD0, pD1, pD2, pStream);
}

void ggml_cuda_op_im2col_3d(ggml_backend_cuda_context &pCtx, ggml_tensor *dst) {
  const ggml_tensor *src0 = dst->src[0];
  const ggml_tensor *src1 = dst->src[1];
  const float *vSrc1D = (const float *)src1->data;
  float *vDstD = (float *)dst->data;
  cudaStream_t vStream = pCtx.stream();

  GGML_ASSERT(src1->type == GGML_TYPE_F32);
  GGML_ASSERT(dst->type == GGML_TYPE_F16 || dst->type == GGML_TYPE_F32);

  GGML_TENSOR_BINARY_OP_LOCALS

  const int32_t cS0 = ((const int32_t *)(dst->op_params))[0];
  const int32_t cS1 = ((const int32_t *)(dst->op_params))[1];
  const int32_t cS2 = ((const int32_t *)(dst->op_params))[2];
  const int32_t cP0 = ((const int32_t *)(dst->op_params))[3];
  const int32_t cP1 = ((const int32_t *)(dst->op_params))[4];
  const int32_t cP2 = ((const int32_t *)(dst->op_params))[5];
  const int32_t cD0 = ((const int32_t *)(dst->op_params))[6];
  const int32_t cD1 = ((const int32_t *)(dst->op_params))[7];
  const int32_t cD2 = ((const int32_t *)(dst->op_params))[8];
  const int32_t cIc = ((const int32_t *)(dst->op_params))[9];

  const int64_t cN = ne13 / cIc;
  const int64_t cId = ne12;
  const int64_t cIh = ne11;
  const int64_t cIw = ne10;

  const int64_t cOc = ne03 / cIc;
  const int64_t cKd = ne02;
  const int64_t cKh = ne01;
  const int64_t cKw = ne00;

  const int64_t cOd = ne3 / cN;
  const int64_t cOh = ne2;
  const int64_t cOw = ne1;

  const size_t cEs = ggml_element_size(src1);
  const int64_t cStrideX = src1->nb[0] / cEs;
  const int64_t cStrideY = src1->nb[1] / cEs;
  const int64_t cStrideZ = src1->nb[2] / cEs;
  const int64_t cStrideQ = src1->nb[3] / cEs;

  if (dst->type == GGML_TYPE_F16) {
    fIm2col3dCudaF16(vSrc1D, (half *)vDstD, cN, cIc, cId, cIh, cIw, cOc, cKd, cKh, cKw, cOd, cOh, cOw, cStrideQ,
                     cStrideZ, cStrideY, cStrideX, cS0, cS1, cS2, cP0, cP1, cP2, cD0, cD1, cD2, vStream);
  } else {
    fIm2col3dCudaF32(vSrc1D, (float *)vDstD, cN, cIc, cId, cIh, cIw, cOc, cKd, cKh, cKw, cOd, cOh, cOw, cStrideQ,
                     cStrideZ, cStrideY, cStrideX, cS0, cS1, cS2, cP0, cP1, cP2, cD0, cD1, cD2, vStream);
  }
}
