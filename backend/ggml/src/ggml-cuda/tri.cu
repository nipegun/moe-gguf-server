#include "common.cuh"
#include "convert.cuh"
#include "tri.cuh"
#include "ggml.h"

template <typename T, bool prefix_keep, int add_to_split>
static __global__ void fTriKernel(const T *pSrc, T *pDst, const int64_t pNe00, const int64_t pNe01, const int64_t pNe02,
                                  const int64_t pNe03, const int64_t pNb00, const int64_t pNb01, const int64_t pNb02,
                                  const int64_t pNb03, const int64_t pNb0, const int64_t pNb1, const int64_t pNb2,
                                  const int64_t pNb3) {
  const int64_t cI3 = blockIdx.z;
  const int64_t cI2 = blockIdx.y;
  const int64_t cI1 = blockIdx.x;
  const int64_t cSplitPoint = cI1 + add_to_split;

  GGML_UNUSED_VARS(pNb00, pNb0);

  if (cI3 >= pNe03 || cI2 >= pNe02 || cI1 >= pNe01) {
    return;
  }

  const T *vSrcRow = pSrc + cI1 * pNb01 + cI2 * pNb02 + cI3 * pNb03;
  T *vDstRow = pDst + cI1 * pNb1 + cI2 * pNb2 + cI3 * pNb3;

  if constexpr (prefix_keep) {
    for (int64_t i0 = threadIdx.x; i0 < cSplitPoint; i0 += blockDim.x) {
      vDstRow[i0] = vSrcRow[i0];
    }
    for (int64_t i0 = threadIdx.x + cSplitPoint; i0 < pNe00; i0 += blockDim.x) {
      vDstRow[i0] = ggml_cuda_cast<T, float>(0.0f);
    }
  } else {
    for (int64_t vI0 = threadIdx.x; vI0 < cSplitPoint; vI0 += blockDim.x) {
      vDstRow[vI0] = ggml_cuda_cast<T, float>(0.0f);
    }
    for (int64_t vI0 = threadIdx.x + cSplitPoint; vI0 < pNe00; vI0 += blockDim.x) {
      vDstRow[vI0] = vSrcRow[vI0];
    }
  }
}

template <typename T>
static void fTriCuda(const T *pSrc, T *pDst, const int64_t pNe00, const int64_t pNe01, const int64_t pNe02,
                     const int64_t pNe03, const int64_t pNb00, const int64_t pNb01, const int64_t pNb02,
                     const int64_t pNb03, const int64_t pNb0, const int64_t pNb1, const int64_t pNb2,
                     const int64_t pNb3, const ggml_tri_type pTtype, cudaStream_t pStream) {

  dim3 vBlockDims(CUDA_TRI_BLOCK_SIZE, 1, 1);
  dim3 vGridDims(pNe01, pNe02, pNe03);
  const size_t cTypeSize = sizeof(T);

  const int cAddToSplit = (pTtype == GGML_TRI_TYPE_LOWER_DIAG || pTtype == GGML_TRI_TYPE_UPPER) ? 1 : 0;
  const bool cPrefixKeep = (pTtype == GGML_TRI_TYPE_LOWER || pTtype == GGML_TRI_TYPE_LOWER_DIAG);

  if (cPrefixKeep) {
    if (cAddToSplit == 0) {
      fTriKernel<T, true, 0><<<vGridDims, vBlockDims, 0, pStream>>>(
          pSrc, pDst, pNe00, pNe01, pNe02, pNe03, pNb00 / cTypeSize, pNb01 / cTypeSize, pNb02 / cTypeSize,
          pNb03 / cTypeSize, pNb0 / cTypeSize, pNb1 / cTypeSize, pNb2 / cTypeSize, pNb3 / cTypeSize);
    } else { // only 0 and 1 supported
      fTriKernel<T, true, 1><<<vGridDims, vBlockDims, 0, pStream>>>(
          pSrc, pDst, pNe00, pNe01, pNe02, pNe03, pNb00 / cTypeSize, pNb01 / cTypeSize, pNb02 / cTypeSize,
          pNb03 / cTypeSize, pNb0 / cTypeSize, pNb1 / cTypeSize, pNb2 / cTypeSize, pNb3 / cTypeSize);
    }
  } else {
    if (cAddToSplit == 0) {
      fTriKernel<T, false, 0><<<vGridDims, vBlockDims, 0, pStream>>>(
          pSrc, pDst, pNe00, pNe01, pNe02, pNe03, pNb00 / cTypeSize, pNb01 / cTypeSize, pNb02 / cTypeSize,
          pNb03 / cTypeSize, pNb0 / cTypeSize, pNb1 / cTypeSize, pNb2 / cTypeSize, pNb3 / cTypeSize);
    } else {
      fTriKernel<T, false, 1><<<vGridDims, vBlockDims, 0, pStream>>>(
          pSrc, pDst, pNe00, pNe01, pNe02, pNe03, pNb00 / cTypeSize, pNb01 / cTypeSize, pNb02 / cTypeSize,
          pNb03 / cTypeSize, pNb0 / cTypeSize, pNb1 / cTypeSize, pNb2 / cTypeSize, pNb3 / cTypeSize);
    }
  }
}

void ggml_cuda_op_tri(ggml_backend_cuda_context &pCtx, ggml_tensor *pDst) {
  const ggml_tensor *vSrc0 = pDst->src[0];
  cudaStream_t vStream = pCtx.stream();

  const ggml_tri_type cTtype = static_cast<ggml_tri_type>(ggml_get_op_params_i32(pDst, 0));

  GGML_ASSERT(vSrc0->type == pDst->type);

  switch (vSrc0->type) {
  case GGML_TYPE_F32: {
    fTriCuda((const float *)vSrc0->data, (float *)pDst->data, vSrc0->ne[0], vSrc0->ne[1], vSrc0->ne[2], vSrc0->ne[3],
             vSrc0->nb[0], vSrc0->nb[1], vSrc0->nb[2], vSrc0->nb[3], pDst->nb[0], pDst->nb[1], pDst->nb[2], pDst->nb[3],
             cTtype, vStream);
  } break;
  case GGML_TYPE_F16: {
    fTriCuda((const half *)vSrc0->data, (half *)pDst->data, vSrc0->ne[0], vSrc0->ne[1], vSrc0->ne[2], vSrc0->ne[3],
             vSrc0->nb[0], vSrc0->nb[1], vSrc0->nb[2], vSrc0->nb[3], pDst->nb[0], pDst->nb[1], pDst->nb[2], pDst->nb[3],
             cTtype, vStream);
  } break;
  case GGML_TYPE_BF16: {
    fTriCuda((const nv_bfloat16 *)vSrc0->data, (nv_bfloat16 *)pDst->data, vSrc0->ne[0], vSrc0->ne[1], vSrc0->ne[2],
             vSrc0->ne[3], vSrc0->nb[0], vSrc0->nb[1], vSrc0->nb[2], vSrc0->nb[3], pDst->nb[0], pDst->nb[1],
             pDst->nb[2], pDst->nb[3], cTtype, vStream);
  } break;
  default:
    GGML_ABORT("fatal error");
  }
}
