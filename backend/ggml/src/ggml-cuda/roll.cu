#include "ggml-cuda/common.cuh"
#include "roll.cuh"

static __forceinline__ __device__ int64_t fWrapIndex(const int64_t pIdx, const int64_t pNe) {
  if (pIdx < 0) {
    return pIdx + pNe;
  }
  if (pIdx >= pNe) {
    return pIdx - pNe;
  }
  return pIdx;
}

static __global__ void fRollF32Cuda(const float *__restrict__ pSrc, float *__restrict__ pDst, const int64_t pNe00,
                                    const int64_t pNe01, const int64_t pNe02, const int64_t pNe03, const int pS0,
                                    const int pS1, const int pS2, const int pS3) {
  const int64_t cIdx = int64_t(blockDim.x) * blockIdx.x + threadIdx.x;
  const int64_t cNElements = pNe00 * pNe01 * pNe02 * pNe03;

  if (cIdx >= cNElements) {
    return;
  }

  const int64_t cI0 = cIdx % pNe00;
  const int64_t cI1 = (cIdx / pNe00) % pNe01;
  const int64_t cI2 = (cIdx / (pNe00 * pNe01)) % pNe02;
  const int64_t cI3 = (cIdx / (pNe00 * pNe01 * pNe02)) % pNe03;

  const int64_t cD0 = fWrapIndex(cI0 - pS0, pNe00);
  const int64_t cD1 = fWrapIndex(cI1 - pS1, pNe01);
  const int64_t cD2 = fWrapIndex(cI2 - pS2, pNe02);
  const int64_t cD3 = fWrapIndex(cI3 - pS3, pNe03);

  pDst[cI3 * (pNe00 * pNe01 * pNe02) + cI2 * (pNe01 * pNe00) + cI1 * pNe00 + cI0] =
      pSrc[cD3 * (pNe00 * pNe01 * pNe02) + cD2 * (pNe01 * pNe00) + cD1 * pNe00 + cD0];
}

void ggml_cuda_op_roll(ggml_backend_cuda_context &pCtx, ggml_tensor *dst) {
  int vS0 = dst->op_params[0];
  int vS1 = dst->op_params[1];
  int vS2 = dst->op_params[2];
  int vS3 = dst->op_params[3];

  const ggml_tensor *src0 = dst->src[0];
  const float *vSrc0D = (const float *)dst->src[0]->data;
  float *vDstD = (float *)dst->data;

  GGML_TENSOR_UNARY_OP_LOCALS;

  GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
  GGML_ASSERT(ggml_are_same_shape(dst->src[0], dst));

  cudaStream_t vStream = pCtx.stream();

  int64_t vSz = (ne00 * ne01 * ne02 * ne03);
  int64_t vNumBlocks = (vSz + CUDA_ROLL_BLOCK_SIZE - 1) / CUDA_ROLL_BLOCK_SIZE;

  fRollF32Cuda<<<vNumBlocks, CUDA_ROLL_BLOCK_SIZE, 0, vStream>>>(vSrc0D, vDstD, ne00, ne01, ne02, ne03, vS0, vS1, vS2,
                                                                 vS3);
}
