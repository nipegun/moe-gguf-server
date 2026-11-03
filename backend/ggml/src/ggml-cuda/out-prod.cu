#include "out-prod.cuh"

#include <cstdint>

static __global__ void fKComputeOutProdPtrs(const float *pSrc0D, const float *pSrc1D, float *pDstD,
                                            const float **pPtrsA, const float **pPtrsB, float **pPtrsC,
                                            const int64_t pNe2, const int64_t pNe3, const int64_t pDps2,
                                            const int64_t pDps3, const size_t pS02, const size_t pS03,
                                            const size_t pS12, const size_t pS13, const size_t pS2, const size_t pS3) {
  const int64_t cI2 = blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t cI3 = blockIdx.y * blockDim.y + threadIdx.y;

  if (cI2 >= pNe2 || cI3 >= pNe3) {
    return;
  }

  const int64_t cIdx = cI3 * pNe2 + cI2;

  pPtrsA[cIdx] = pSrc0D + (cI3 / pDps3) * pS03 + (cI2 / pDps2) * pS02;
  pPtrsB[cIdx] = pSrc1D + cI3 * pS13 + cI2 * pS12;
  pPtrsC[cIdx] = pDstD + cI3 * pS3 + cI2 * pS2;
}

void ggml_cuda_out_prod(ggml_backend_cuda_context &pCtx, ggml_tensor *dst) {
  const ggml_tensor *src0 = dst->src[0];
  const ggml_tensor *src1 = dst->src[1];

  GGML_TENSOR_BINARY_OP_LOCALS

  GGML_ASSERT(src0->type == GGML_TYPE_F32);
  GGML_ASSERT(src1->type == GGML_TYPE_F32);
  GGML_ASSERT(dst->type == GGML_TYPE_F32);

  GGML_ASSERT(ne01 == ne11);
  GGML_ASSERT(ne0 == ne00);
  GGML_ASSERT(ne1 == ne10);

  GGML_ASSERT(ne2 % src0->ne[2] == 0);
  GGML_ASSERT(ne3 % src0->ne[3] == 0);

  GGML_ASSERT(ne2 == src1->ne[2]);
  GGML_ASSERT(ne3 == src1->ne[3]);

  const float *vSrc0D = (const float *)src0->data;
  const float *vSrc1D = (const float *)src1->data;
  float *vDstD = (float *)dst->data;

  cudaStream_t vStream = pCtx.stream();
  cublasHandle_t vHandle = pCtx.cublas_handle();

  const float cAlpha = 1.0f;
  const float cBeta = 0.0f;

  CUBLAS_CHECK(cublasSetStream(vHandle, vStream));

  const int64_t cLda = nb01 / sizeof(float);
  const int64_t cLdc = nb1 / sizeof(float);

  const bool cSrc1T = ggml_is_transposed(src1);
  const cublasOperation_t cSrc1CublasOp = cSrc1T ? CUBLAS_OP_N : CUBLAS_OP_T;
  const int64_t cLdb = (cSrc1T ? nb10 : nb11) / sizeof(float);
  GGML_ASSERT((cSrc1T ? nb11 : nb10) == sizeof(float));

  // data strides in dimensions 2/3
  const size_t cS02 = nb02 / sizeof(float);
  const size_t cS03 = nb03 / sizeof(float);
  const size_t cS12 = nb12 / sizeof(float);
  const size_t cS13 = nb13 / sizeof(float);
  const size_t cS2 = nb2 / sizeof(float);
  const size_t cS3 = nb3 / sizeof(float);

  // dps == dst per src0, used for group query attention
  const int64_t cDps2 = ne2 / ne02;
  const int64_t cDps3 = ne3 / ne03;

  if (cDps2 == 1 && ne2 > 1) {
    // src0 has uniform stride s02 along dim 2; batch the inner loop with a strided GEMM
    GGML_ASSERT(ne2 <= std::numeric_limits<int>::max());
    const int cBatchCount = (int)ne2;
    for (int64_t vI3 = 0; vI3 < ne3; ++vI3) {
      CUBLAS_CHECK(cublasSgemmStridedBatched(vHandle, CUBLAS_OP_N, cSrc1CublasOp, ne0, ne1, ne01, &cAlpha,
                                             vSrc0D + (vI3 / cDps3) * cS03, cLda, cS02, vSrc1D + vI3 * cS13, cLdb, cS12,
                                             &cBeta, vDstD + vI3 * cS3, cLdc, cS2, cBatchCount));
    }
  } else if (ne2 > 1 || ne3 > 1) {
    // dps2 > 1 (src0 broadcast along dim 2 with non-uniform stride) or multiple GEMMs
    // along dim 3: compute per-GEMM pointers on the device and use a single batched GEMM.
    GGML_ASSERT(ne3 > 0);
    GGML_ASSERT(ne2 <= (int64_t)std::numeric_limits<int>::max() / ne3);
    const int cBatchCount = (int)(ne2 * ne3);

    ggml_cuda_pool_alloc<const float *> vPtrsA(pCtx.pool(), cBatchCount);
    ggml_cuda_pool_alloc<const float *> vPtrsB(pCtx.pool(), cBatchCount);
    ggml_cuda_pool_alloc<float *> vPtrsC(pCtx.pool(), cBatchCount);

    const dim3 cBlockDims(16, 16);
    const dim3 cGridDims((ne2 + cBlockDims.x - 1) / cBlockDims.x, (ne3 + cBlockDims.y - 1) / cBlockDims.y);
    fKComputeOutProdPtrs<<<cGridDims, cBlockDims, 0, vStream>>>(vSrc0D, vSrc1D, vDstD, vPtrsA.get(), vPtrsB.get(),
                                                                vPtrsC.get(), ne2, ne3, cDps2, cDps3, cS02, cS03, cS12,
                                                                cS13, cS2, cS3);
    CUDA_CHECK(cudaGetLastError());

    CUBLAS_CHECK(cublasSgemmBatched(vHandle, CUBLAS_OP_N, cSrc1CublasOp, ne0, ne1, ne01, &cAlpha, vPtrsA.get(), cLda,
                                    vPtrsB.get(), cLdb, &cBeta, vPtrsC.get(), cLdc, cBatchCount));
  } else {
    // ne2 == 1 && ne3 == 1: single GEMM
    CUBLAS_CHECK(cublasSgemm(vHandle, CUBLAS_OP_N, cSrc1CublasOp, ne0, ne1, ne01, &cAlpha, vSrc0D, cLda, vSrc1D, cLdb,
                             &cBeta, vDstD, cLdc));
  }
}
