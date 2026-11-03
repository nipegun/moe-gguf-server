#include "common.cuh"
#include "ggml.h"
#include "solve_tri.cuh"

#define MAX_N_FAST 64
#define MAX_K_FAST 32

static __global__ void fGetBatchPointers(const float *pA, float *pX, const float **pAPtrs, float **pXPtrs,
                                         int64_t pNe02, int64_t pTotalBatches, size_t pS02, size_t pS03, size_t pS2,
                                         size_t pS3) {
  const int cIdx = blockIdx.x * blockDim.x + threadIdx.x;
  if (cIdx >= pTotalBatches) {
    return;
  }

  const int64_t cI3 = cIdx / pNe02;
  const int64_t cI2 = cIdx % pNe02;

  pAPtrs[cIdx] = pA + cI3 * pS03 + cI2 * pS02;
  pXPtrs[cIdx] = pX + cI3 * pS3 + cI2 * pS2;
}

static void fSolveTriF32Cublas(ggml_backend_cuda_context &pCtx, const float *pA, const float *pB, float *pX, int pN,
                               int pK, int64_t pNe02, int64_t pNe03, size_t pS02, size_t pS03, size_t pS12, size_t pS13,
                               size_t pS2, size_t pS3, cudaStream_t pStream) {
  const float cAlpha = 1.0f;
  const int64_t cTotalBatches = pNe02 * pNe03;
  if (cTotalBatches == 0) {
    return;
  }

  // Bulk copy B -> X (contiguous tensors)
  if (pX != pB) {
    const int64_t cTotalElementsBx = pN * pK * cTotalBatches;
    CUDA_CHECK(cudaMemcpyAsync(pX, pB, cTotalElementsBx * sizeof(float), cudaMemcpyDeviceToDevice, pStream));
  }

  const int cId = ggml_cuda_get_device();

  ggml_cuda_pool_alloc<const float *> vAPtrsAlloc(pCtx.pool(cId), cTotalBatches);
  ggml_cuda_pool_alloc<float *> vXPtrsAlloc(pCtx.pool(cId), cTotalBatches);

  const float **vAPtrsDev = vAPtrsAlloc.get();
  float **vXPtrsDev = vXPtrsAlloc.get();

  fGetBatchPointers<<<(cTotalBatches + 255) / 256, 256, 0, pStream>>>(pA, pX, vAPtrsDev, vXPtrsDev, pNe02,
                                                                      cTotalBatches, pS02, pS03, pS2, pS3);

  CUBLAS_CHECK(cublasSetStream(pCtx.cublas_handle(cId), pStream));

  // Yes, this is necessary, without this we get RMSE errors
  CUBLAS_CHECK(cublasSetMathMode(pCtx.cublas_handle(cId), CUBLAS_DEFAULT_MATH));
  CUBLAS_CHECK(cublasStrsmBatched(pCtx.cublas_handle(cId), CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N,
                                  CUBLAS_DIAG_NON_UNIT, pK, pN, &cAlpha, vAPtrsDev, pN, vXPtrsDev, pK, cTotalBatches));

  // revert to standard mode from common.cuh
  CUBLAS_CHECK(cublasSetMathMode(pCtx.cublas_handle(cId), CUBLAS_TF32_TENSOR_OP_MATH));

  GGML_UNUSED_VARS(pS12, pS13);
}

// ======================
// Fast Kernel (n <= 64, k <= 32) - Warp-based parallel reduction
// ======================
// When ncols_template == 0 the bounds for the loops in this function are not
// known and can't be unrolled. As we want to keep pragma unroll for all other
// cases we suppress the clang transformation warning here.
#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wpass-failed"
#endif // __clang__
template <int n_template, int k_template>
static __global__ void fSolveTriF32Fast(const float *__restrict__ pA, const float *__restrict__ pB,
                                        float *__restrict__ pX, const uint3 pNe02, const size_t pNb02,
                                        const size_t pNb03, const size_t pNb12, const size_t pNb13, const size_t pNb2,
                                        const size_t pNb3, const int pNArg, const int pKArg) {
  const int cN = n_template == 0 ? pNArg : n_template;
  const int cK = k_template == 0 ? pKArg : k_template;

  const int cBatchIdx = blockIdx.x;
  const int cLane = threadIdx.x;
  const int cColIdx = threadIdx.y;

  if (cColIdx >= cK) {
    return;
  }

  const uint2 cI02I03 = fast_div_modulo(cBatchIdx, pNe02);
  const int64_t cI02 = cI02I03.y;
  const int64_t cI03 = cI02I03.x;

  const float *const cABatch = (const float *)(pA + cI02 * pNb02 + cI03 * pNb03);
  const float *const cBBatch = (const float *)(pB + cI02 * pNb12 + cI03 * pNb13);
  float *vXBatch = (float *)(pX + cI02 * pNb2 + cI03 * pNb3);

  __shared__ float vSA[MAX_N_FAST * MAX_N_FAST];

  const int cOffset = threadIdx.x + threadIdx.y * blockDim.x;

#pragma unroll
  for (int vI = 0; vI < cN * cN; vI += cK * WARP_SIZE) {
    const int cI0 = vI + cOffset;
    if (cI0 < cN * cN) {
      vSA[cI0] = cABatch[cI0];
    }
  }

  __syncthreads();

  float vXLow = (cLane < cN) ? cBBatch[cLane * cK + cColIdx] : 0.0f;
  float vXHigh = (WARP_SIZE + cLane < cN) ? cBBatch[(WARP_SIZE + cLane) * cK + cColIdx] : 0.0f;

  const int cHalf = WARP_SIZE;
  const int cNrowsLow = (cN < cHalf) ? cN : cHalf;

#pragma unroll
  for (int vRow = 0; vRow < cNrowsLow; ++vRow) {
    float vSum = 0.0f;
    if (cLane < vRow) {
      vSum += vSA[vRow * cN + cLane] * vXLow;
    }
    vSum = warp_reduce_sum(vSum);

    if (cLane == vRow) {
      vXLow = (vXLow - vSum) / vSA[vRow * cN + vRow];
    }
  }

#pragma unroll
  for (int vRow = cHalf; vRow < cN; ++vRow) {
    float vSum = vSA[vRow * cN + cLane] * vXLow;
    const int cJ = cHalf + cLane;
    if (cJ < vRow) {
      vSum += vSA[vRow * cN + cJ] * vXHigh;
    }
    vSum = warp_reduce_sum(vSum);

    if (cLane == vRow - cHalf) {
      vXHigh = (vXHigh - vSum) / vSA[vRow * cN + vRow];
    }
  }

#pragma unroll
  for (int vRr = 0; vRr < 2; ++vRr) {
    const int cRow = vRr * WARP_SIZE + cLane;
    if (cRow < cN) {
      const float cVal = (cRow < cHalf) ? vXLow : vXHigh;
      vXBatch[cRow * cK + cColIdx] = cVal;
    }
  }
}
#ifdef __clang__
#pragma clang diagnostic pop
#endif // __clang__

static void fSolveTriF32Cuda(const float *pA, const float *pB, float *pX, int pN, int pK, int64_t pNe02, int64_t pNe03,
                             size_t pNb02, size_t pNb03, size_t pNb12, size_t pNb13, size_t pNb2, size_t pNb3,
                             cudaStream_t pStream) {
  const uint3 cNe02Fd = init_fastdiv_values((uint32_t)pNe02);
  dim3 vThreads(WARP_SIZE, pK);
  dim3 vGrid(pNe02 * pNe03);
  if (pN == 64) {
    switch (pK) {
    case 32:
      fSolveTriF32Fast<64, 32>
          <<<vGrid, vThreads, 0, pStream>>>(pA, pB, pX, cNe02Fd, pNb02, pNb03, pNb12, pNb13, pNb2, pNb3, 0, 0);
      break;
    case 16:
      fSolveTriF32Fast<64, 16>
          <<<vGrid, vThreads, 0, pStream>>>(pA, pB, pX, cNe02Fd, pNb02, pNb03, pNb12, pNb13, pNb2, pNb3, 0, 0);
      break;
    case 14:
      fSolveTriF32Fast<64, 14>
          <<<vGrid, vThreads, 0, pStream>>>(pA, pB, pX, cNe02Fd, pNb02, pNb03, pNb12, pNb13, pNb2, pNb3, 0, 0);
      break;
    case 12:
      fSolveTriF32Fast<64, 12>
          <<<vGrid, vThreads, 0, pStream>>>(pA, pB, pX, cNe02Fd, pNb02, pNb03, pNb12, pNb13, pNb2, pNb3, 0, 0);
      break;
    case 10:
      fSolveTriF32Fast<64, 10>
          <<<vGrid, vThreads, 0, pStream>>>(pA, pB, pX, cNe02Fd, pNb02, pNb03, pNb12, pNb13, pNb2, pNb3, 0, 0);
      break;
    case 8:
      fSolveTriF32Fast<64, 8>
          <<<vGrid, vThreads, 0, pStream>>>(pA, pB, pX, cNe02Fd, pNb02, pNb03, pNb12, pNb13, pNb2, pNb3, 0, 0);
      break;
    case 6:
      fSolveTriF32Fast<64, 6>
          <<<vGrid, vThreads, 0, pStream>>>(pA, pB, pX, cNe02Fd, pNb02, pNb03, pNb12, pNb13, pNb2, pNb3, 0, 0);
      break;
    case 4:
      fSolveTriF32Fast<64, 4>
          <<<vGrid, vThreads, 0, pStream>>>(pA, pB, pX, cNe02Fd, pNb02, pNb03, pNb12, pNb13, pNb2, pNb3, 0, 0);
      break;
    case 2:
      fSolveTriF32Fast<64, 2>
          <<<vGrid, vThreads, 0, pStream>>>(pA, pB, pX, cNe02Fd, pNb02, pNb03, pNb12, pNb13, pNb2, pNb3, 0, 0);
      break;
    case 1:
      fSolveTriF32Fast<64, 1>
          <<<vGrid, vThreads, 0, pStream>>>(pA, pB, pX, cNe02Fd, pNb02, pNb03, pNb12, pNb13, pNb2, pNb3, 0, 0);
      break;
    default:
      fSolveTriF32Fast<0, 0>
          <<<vGrid, vThreads, 0, pStream>>>(pA, pB, pX, cNe02Fd, pNb02, pNb03, pNb12, pNb13, pNb2, pNb3, pN, pK);
    }
  } else { // run general case
    fSolveTriF32Fast<0, 0>
        <<<vGrid, vThreads, 0, pStream>>>(pA, pB, pX, cNe02Fd, pNb02, pNb03, pNb12, pNb13, pNb2, pNb3, pN, pK);
  }
}

void ggml_cuda_op_solve_tri(ggml_backend_cuda_context &pCtx, ggml_tensor *pDst) {
  const ggml_tensor *vSrc0 = pDst->src[0]; // A (n×n, lower triangular)
  const ggml_tensor *vSrc1 = pDst->src[1]; // B (n×k)

  ggml_is_contiguous(vSrc0);
  ggml_is_contiguous(vSrc1);

  const int64_t cN = vSrc0->ne[0];
  const int64_t cK = vSrc1->ne[0];
  const int64_t cNe02 = vSrc0->ne[2];
  const int64_t cNe03 = vSrc0->ne[3];

  if (cN <= MAX_N_FAST && cK <= MAX_K_FAST) {
    fSolveTriF32Cuda((const float *)vSrc0->data, (const float *)vSrc1->data, (float *)pDst->data, cN, cK, vSrc0->ne[2],
                     vSrc0->ne[3], vSrc0->nb[2] / sizeof(float), vSrc0->nb[3] / sizeof(float),
                     vSrc1->nb[2] / sizeof(float), vSrc1->nb[3] / sizeof(float), pDst->nb[2] / sizeof(float),
                     pDst->nb[3] / sizeof(float), pCtx.stream());
  } else {
    fSolveTriF32Cublas(pCtx, (const float *)vSrc0->data, (const float *)vSrc1->data, (float *)pDst->data, cN, cK, cNe02,
                       cNe03, vSrc0->nb[2] / sizeof(float), vSrc0->nb[3] / sizeof(float), vSrc1->nb[2] / sizeof(float),
                       vSrc1->nb[3] / sizeof(float), pDst->nb[2] / sizeof(float), pDst->nb[3] / sizeof(float),
                       pCtx.stream());
  }
}
