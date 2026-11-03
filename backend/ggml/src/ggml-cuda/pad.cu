#include "pad.cuh"

#include <stdint.h>

__device__ __forceinline__ int64_t fWrapAround(int64_t pCoord, int64_t pSize) {
  // + size ensures negatives are handled properly
  return (pCoord + pSize) % pSize;
}

static __global__ void fPadF32(const float *pSrc, size_t pS00, size_t pS01, size_t pS02, size_t pS03, float *pDst,
                               const int pLp0, const int pRp0, const int pLp1, const int pRp1, const int pLp2,
                               const int pRp2, const int pLp3, const int pRp3, const int pNe0, const int pNe1,
                               const int pNe2, const int pNe3, const bool pCircular) {
  // blockIdx.z: i3*ne2+i2
  // blockIdx.y: i1
  // blockIDx.x: i0 / CUDA_PAD_BLOCK_SIZE
  // gridDim.y:  ne1
  int vI0 = threadIdx.x + blockIdx.x * blockDim.x;
  int vI1 = blockIdx.y;
  int vI2 = blockIdx.z % pNe2;
  int vI3 = blockIdx.z / pNe2;

  if (vI0 >= pNe0 || vI1 >= pNe1 || vI2 >= pNe2 || vI3 >= pNe3) {
    return;
  }

  const int64_t cDstIdx = vI3 * (pNe0 * pNe1 * pNe2) + vI2 * (pNe0 * pNe1) + vI1 * pNe0 + vI0;

  if (!pCircular) {
    if ((vI0 >= pLp0 && vI0 < pNe0 - pRp0) && (vI1 >= pLp1 && vI1 < pNe1 - pRp1) &&
        (vI2 >= pLp2 && vI2 < pNe2 - pRp2) && (vI3 >= pLp3 && vI3 < pNe3 - pRp3)) {
      const int64_t cI00 = vI0 - pLp0;
      const int64_t cI01 = vI1 - pLp1;
      const int64_t cI02 = vI2 - pLp2;
      const int64_t cI03 = vI3 - pLp3;

      const int64_t cSrcIdx = cI03 * pS03 + cI02 * pS02 + cI01 * pS01 + cI00 * pS00;

      pDst[cDstIdx] = pSrc[cSrcIdx];
    } else {
      pDst[cDstIdx] = 0.0f;
    }
  }
  // circular means on a torus, so x and y wrap around
  else {
    const int64_t cNe00 = pNe0 - pLp0 - pRp0;
    const int64_t cNe01 = pNe1 - pLp1 - pRp1;
    const int64_t cNe02 = pNe2 - pLp2 - pRp2;
    const int64_t cNe03 = pNe3 - pLp3 - pRp3;

    const int64_t cI00 = fWrapAround(vI0 - pLp0, cNe00);
    const int64_t cI01 = fWrapAround(vI1 - pLp1, cNe01);
    const int64_t cI02 = fWrapAround(vI2 - pLp2, cNe02);
    const int64_t cI03 = fWrapAround(vI3 - pLp3, cNe03);

    const int64_t cSrcIdx = cI03 * pS03 + cI02 * pS02 + cI01 * pS01 + cI00 * pS00;

    pDst[cDstIdx] = pSrc[cSrcIdx];
  }
}

static void fPadF32Cuda(const float *pSrc, size_t pS00, size_t pS01, size_t pS02, size_t pS03, float *pDst,
                        const int pLp0, const int pRp0, const int pLp1, const int pRp1, const int pLp2, const int pRp2,
                        const int pLp3, const int pRp3, const int pNe0, const int pNe1, const int pNe2, const int pNe3,
                        const bool pCircular, cudaStream_t pStream) {
  int vNumBlocks = (pNe0 + CUDA_PAD_BLOCK_SIZE - 1) / CUDA_PAD_BLOCK_SIZE;
  dim3 vGridDim(vNumBlocks, pNe1, pNe2 * pNe3);
  fPadF32<<<vGridDim, CUDA_PAD_BLOCK_SIZE, 0, pStream>>>(pSrc, pS00, pS01, pS02, pS03, pDst, pLp0, pRp0, pLp1, pRp1,
                                                         pLp2, pRp2, pLp3, pRp3, pNe0, pNe1, pNe2, pNe3, pCircular);
}

void ggml_cuda_op_pad(ggml_backend_cuda_context &pCtx, ggml_tensor *dst) {
  const ggml_tensor *src0 = dst->src[0];
  const float *vSrc0D = (const float *)src0->data;
  float *vDstD = (float *)dst->data;
  cudaStream_t vStream = pCtx.stream();

  GGML_TENSOR_UNARY_OP_LOCALS;

  GGML_ASSERT(src0->type == GGML_TYPE_F32);
  GGML_ASSERT(dst->type == GGML_TYPE_F32);

  const int32_t cLp0 = ((const int32_t *)(dst->op_params))[0];
  const int32_t cRp0 = ((const int32_t *)(dst->op_params))[1];
  const int32_t cLp1 = ((const int32_t *)(dst->op_params))[2];
  const int32_t cRp1 = ((const int32_t *)(dst->op_params))[3];
  const int32_t cLp2 = ((const int32_t *)(dst->op_params))[4];
  const int32_t cRp2 = ((const int32_t *)(dst->op_params))[5];
  const int32_t cLp3 = ((const int32_t *)(dst->op_params))[6];
  const int32_t cRp3 = ((const int32_t *)(dst->op_params))[7];
  const int32_t cCircular = ((const int32_t *)(dst->op_params))[8];

  const size_t cS00 = nb00 / ggml_type_size(src0->type);
  const size_t cS01 = nb01 / ggml_type_size(src0->type);
  const size_t cS02 = nb02 / ggml_type_size(src0->type);
  const size_t cS03 = nb03 / ggml_type_size(src0->type);

  fPadF32Cuda(vSrc0D, cS00, cS01, cS02, cS03, vDstD, cLp0, cRp0, cLp1, cRp1, cLp2, cRp2, cLp3, cRp3, dst->ne[0],
              dst->ne[1], dst->ne[2], dst->ne[3], (bool)cCircular, vStream);
}
