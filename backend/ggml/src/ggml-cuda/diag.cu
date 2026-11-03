#include "convert.cuh"
#include "diag.cuh"
#include "ggml.h"

template <typename T>
static __global__ void fDiagKernel(T *__restrict__ pDst, const T *__restrict__ pSrc, const int64_t pNe0,
                                   const int64_t pNe1, const int64_t pNe2, const int64_t pNe3,
                                   const int64_t pTotalElements) {
  const int64_t cGlobalIdx = blockIdx.x * blockDim.x + threadIdx.x;

  if (cGlobalIdx >= pTotalElements) {
    return;
  }

  const int64_t cI0 = cGlobalIdx % pNe0;
  const int64_t cI1 = (cGlobalIdx / pNe0) % pNe1;
  const int64_t cI2 = (cGlobalIdx / (pNe0 * pNe1)) % pNe2;
  const int64_t cI3 = cGlobalIdx / (pNe0 * pNe1 * pNe2);

  const int64_t cDstIdx = ((cI3 * pNe2 + cI2) * pNe1 + cI1) * pNe0 + cI0;

  if (cI0 == cI1) {
    const int64_t cBatchIdx = cI3 * pNe2 + cI2;
    const int64_t cSrcIdx = cBatchIdx * pNe0 + cI0;
    pDst[cDstIdx] = pSrc[cSrcIdx];
  } else {
    pDst[cDstIdx] = ggml_cuda_cast<T>(0);
  }
  GGML_UNUSED_VARS(pNe3);
}

void ggml_cuda_op_diag(ggml_backend_cuda_context &pCtx, ggml_tensor *pDst) {
  const ggml_tensor *vSrc0 = pDst->src[0];

  void *vDstD = pDst->data;
  const void *vSrc0D = vSrc0->data;

  cudaStream_t vStream = pCtx.stream();

  GGML_ASSERT(ggml_is_contiguous(pDst));
  GGML_ASSERT(ggml_is_contiguous(vSrc0));

  const int64_t cNe00 = vSrc0->ne[0];
  const int64_t cNe01 = vSrc0->ne[1];
  const int64_t cNe02 = vSrc0->ne[2];
  const int64_t cNe03 = vSrc0->ne[3];

  const int64_t cNe0 = pDst->ne[0];
  const int64_t cNe1 = pDst->ne[1];
  const int64_t cNe2 = pDst->ne[2];
  const int64_t cNe3 = pDst->ne[3];

  GGML_ASSERT(cNe00 == cNe0);
  GGML_ASSERT(cNe01 == 1);
  GGML_ASSERT(cNe02 == cNe2);
  GGML_ASSERT(cNe03 == cNe3);

  const int64_t cNElems = ggml_nelements(pDst);
  const int64_t cNumBlocks = (cNElems + CUDA_DIAG_BLOCK_SIZE - 1) / CUDA_DIAG_BLOCK_SIZE;

  switch (pDst->type) {
  case GGML_TYPE_F32:
    fDiagKernel<<<cNumBlocks, CUDA_DIAG_BLOCK_SIZE, 0, vStream>>>((float *)vDstD, (const float *)vSrc0D, cNe0, cNe1,
                                                                  cNe2, cNe3, cNElems);
    break;
  case GGML_TYPE_F16:
    fDiagKernel<<<cNumBlocks, CUDA_DIAG_BLOCK_SIZE, 0, vStream>>>((half *)vDstD, (const half *)vSrc0D, cNe0, cNe1, cNe2,
                                                                  cNe3, cNElems);
    break;
  default:
    GGML_ABORT("unsupported type");
  }
}
