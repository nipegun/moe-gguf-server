#include "common.cuh"
#include "count-equal.cuh"

#include <cstdint>

template <typename T>
static __global__ void fCountEqual(const T *__restrict__ pX, const T *__restrict__ pY, int64_t *__restrict__ pDst,
                                   const int64_t pDk, const int64_t pK) {
  const int64_t cI0 = (int64_t)blockIdx.x * pDk;
  const int64_t cI1 = min(cI0 + pDk, pK);

  int vNequal = 0;

  for (int64_t vI = cI0 + threadIdx.x; vI < cI1; vI += WARP_SIZE) {
    const T cXi = pX[vI];
    const T cYi = pY[vI];
    vNequal += cXi == cYi;
  }

  vNequal = warp_reduce_sum(vNequal);

  if (threadIdx.x != 0) {
    return;
  }

  atomicAdd((int *)pDst, vNequal);
}

void ggml_cuda_count_equal(ggml_backend_cuda_context &pCtx, ggml_tensor *pDst) {
  const ggml_tensor *vSrc0 = pDst->src[0];
  const ggml_tensor *vSrc1 = pDst->src[1];

  GGML_ASSERT(vSrc0->type == vSrc1->type);
  GGML_ASSERT(pDst->type == GGML_TYPE_I64);

  GGML_ASSERT(ggml_are_same_shape(vSrc0, vSrc1));
  GGML_ASSERT(ggml_is_contiguous(vSrc0));
  GGML_ASSERT(ggml_is_contiguous(vSrc1));
  GGML_ASSERT(ggml_is_contiguous(pDst));

  int64_t *vDstD = (int64_t *)pDst->data;

  cudaStream_t vStream = pCtx.stream();
  const int cNsm = ggml_cuda_info().devices[ggml_cuda_get_device()].nsm;

  const int64_t cNe = ggml_nelements(vSrc0);
  GGML_ASSERT(cNe < (1 << 30) && "atomicAdd implementation only supports int");
  const int64_t cDne = GGML_PAD((cNe + 4 * cNsm - 1) / (4 * cNsm), CUDA_COUNT_EQUAL_CHUNK_SIZE);

  CUDA_CHECK(cudaMemsetAsync(vDstD, 0, ggml_nbytes(pDst), vStream));

  const dim3 cBlocksDim(WARP_SIZE, 1, 1);
  const dim3 cBlocksNum(
      std::min((int64_t)4 * cNsm, (cNe + CUDA_COUNT_EQUAL_CHUNK_SIZE - 1) / CUDA_COUNT_EQUAL_CHUNK_SIZE), 1, 1);

  switch (vSrc0->type) {
  case GGML_TYPE_I32: {
    const int *vSrc0D = (const int *)vSrc0->data;
    const int *vSrc1D = (const int *)vSrc1->data;
    fCountEqual<<<cBlocksNum, cBlocksDim, 0, vStream>>>(vSrc0D, vSrc1D, vDstD, cDne, cNe);
  } break;
  default:
    GGML_ASSERT(false);
    break;
  }
}
