#include "getrows.cuh"
#include "dequantize.cuh"
#include "convert.cuh"

template <int qk, int qr, dequantize_kernel_t dequantize_kernel, typename dst_t>
static __global__ void
fKGetRows(const void *__restrict__ pSrc0, const int32_t *__restrict__ pSrc1, dst_t *__restrict__ pDst,
          const int64_t pNe00, /*const int64_t ne01, const int64_t ne02, const int64_t ne03,*/
          /*const int64_t ne10,*/ const int64_t pNe11, const uint3 pNe12Fdv, /*const int64_t ne13,*/
          /*const size_t s0,*/ const size_t pS1, const size_t pS2, const size_t pS3,
          /*const size_t nb00,*/ const size_t pNb01, const size_t pNb02, const size_t pNb03, const size_t pS10,
          const size_t pS11, const size_t pS12 /*, const size_t s13*/) {

  ggml_cuda_pdl_sync();
  for (int64_t vZ = blockIdx.z; vZ < pNe11 * (int64_t)pNe12Fdv.z; vZ += gridDim.z) {
    for (int64_t vI00 = 2 * (blockIdx.y * blockDim.x + threadIdx.x); vI00 < pNe00; vI00 += gridDim.y * blockDim.x) {
      // The x and y dimensions of the grid are swapped because the maximum allowed grid size for x is higher.
      const int cI10 = blockIdx.x;
      const uint2 cDm = fast_div_modulo((uint32_t)vZ, pNe12Fdv);
      const int cI11 = cDm.x;
      const int cI12 = cDm.y;

      const int cI01 = pSrc1[cI10 * pS10 + cI11 * pS11 + cI12 * pS12];

      dst_t *vDstRow = pDst + cI10 * pS1 + cI11 * pS2 + cI12 * pS3;
      const void *vSrc0Row = (const char *)pSrc0 + cI01 * pNb01 + cI11 * pNb02 + cI12 * pNb03;

      const int cIb = vI00 / qk;          // block index
      const int cIqs = (vI00 % qk) / qr;  // quant index
      const int cIybs = vI00 - vI00 % qk; // dst block start index
      const int cYOffset = qr == 1 ? 1 : qk / 2;

      // dequantize
      float2 v_;
      dequantize_kernel(vSrc0Row, cIb, cIqs, v_);

      vDstRow[cIybs + cIqs + 0] = ggml_cuda_cast<dst_t>(v_.x);
      vDstRow[cIybs + cIqs + cYOffset] = ggml_cuda_cast<dst_t>(v_.y);
    }
  }
}

template <typename dst_t, dequantize_kq_t<dst_t> dequantize_kq>
static __global__ void
fKGetRowsKq(const void *__restrict__ pSrc0, const int32_t *__restrict__ pSrc1, dst_t *__restrict__ pDst,
            const int64_t pNe00, /*const int64_t ne01, const int64_t ne02, const int64_t ne03,*/
            /*const int64_t ne10,*/ const int64_t pNe11, const uint3 pNe12Fdv, /*const int64_t ne13,*/
            /*const size_t s0,*/ const size_t pS1, const size_t pS2, const size_t pS3,
            /*const size_t nb00,*/ const size_t pNb01, const size_t pNb02, const size_t pNb03, const size_t pS10,
            const size_t pS11, const size_t pS12 /*, const size_t s13*/) {

  ggml_cuda_pdl_sync();
  const int64_t cNsb = pNe00 / QK_K; // super-blocks per row
  for (int64_t vZ = blockIdx.z; vZ < pNe11 * (int64_t)pNe12Fdv.z; vZ += gridDim.z) {
    // The x and y dimensions of the grid are swapped because the maximum allowed grid size for x is higher.
    const int cI10 = blockIdx.x;
    const uint2 cDm = fast_div_modulo((uint32_t)vZ, pNe12Fdv);
    const int cI11 = cDm.x;
    const int cI12 = cDm.y;

    const int cI01 = pSrc1[cI10 * pS10 + cI11 * pS11 + cI12 * pS12];

    dst_t *vDstRow = pDst + cI10 * pS1 + cI11 * pS2 + cI12 * pS3;
    const void *vSrc0Row = (const char *)pSrc0 + cI01 * pNb01 + cI11 * pNb02 + cI12 * pNb03;

    for (int64_t vIb = blockIdx.y; vIb < cNsb; vIb += gridDim.y) {
      dequantize_kq(vSrc0Row, vIb, vDstRow + vIb * QK_K, threadIdx.x);
    }
  }
}

template <typename src0_t, typename dst_t>
static __global__ void
fKGetRowsFloat(const src0_t *pSrc0Ptr, const int32_t *pSrc1Ptr, dst_t *pDstPtr,
               const int64_t pNe00, /*const int64_t ne01, const int64_t ne02, const int64_t ne03,*/
               /*const int64_t ne10,*/ const int64_t pNe11, const uint3 pNe12Fdv, /*const int64_t ne13,*/
               /*const size_t s0,*/ const size_t pS1, const size_t pS2, const size_t pS3,
               /*const size_t nb00,*/ const size_t pNb01, const size_t pNb02, const size_t pNb03, const size_t pS10,
               const size_t pS11, const size_t pS12 /*, const size_t s13*/) {

  ggml_cuda_pdl_lc();
  const src0_t *GGML_CUDA_RESTRICT vSrc0 = pSrc0Ptr;
  const int32_t *GGML_CUDA_RESTRICT vSrc1 = pSrc1Ptr;
  dst_t *GGML_CUDA_RESTRICT vDst = pDstPtr;
  ggml_cuda_pdl_sync();
  for (int64_t vZ = blockIdx.z; vZ < pNe11 * (int64_t)pNe12Fdv.z; vZ += gridDim.z) {
    // The x and y dimensions of the grid are swapped because the maximum allowed grid size for x is higher.
    const int cI10 = blockIdx.x;
    const uint2 cDm = fast_div_modulo((uint32_t)vZ, pNe12Fdv);
    const int cI11 = cDm.x;
    const int cI12 = cDm.y;

    const int cI01 = vSrc1[cI10 * pS10 + cI11 * pS11 + cI12 * pS12];

    dst_t *GGML_CUDA_RESTRICT vDstRow = vDst + cI10 * pS1 + cI11 * pS2 + cI12 * pS3;
    const src0_t *GGML_CUDA_RESTRICT vSrc0Row =
        (const src0_t *)((const char *)vSrc0 + cI01 * pNb01 + cI11 * pNb02 + cI12 * pNb03);

    for (int64_t vI00 = blockIdx.y * blockDim.x + threadIdx.x; vI00 < pNe00; vI00 += gridDim.y * blockDim.x) {
      vDstRow[vI00] = ggml_cuda_cast<dst_t>(vSrc0Row[vI00]);
    }
  }
}

template <typename dst_t>
static __global__ void fKGetRowsFloatVec(const dst_t *pSrc0Ptr, const int32_t *pSrc1Ptr, dst_t *pDstPtr,
                                         const int64_t pNe00v, const int64_t pNe11, const uint3 pNe12Fdv,
                                         const size_t pS1, const size_t pS2, const size_t pS3, const size_t pNb01,
                                         const size_t pNb02, const size_t pNb03, const size_t pS10, const size_t pS11,
                                         const size_t pS12) {

  ggml_cuda_pdl_lc();
  ggml_cuda_pdl_sync();
  for (int64_t vZ = blockIdx.z; vZ < pNe11 * (int64_t)pNe12Fdv.z; vZ += gridDim.z) {
    const int cI10 = blockIdx.x;
    const uint2 cDm = fast_div_modulo((uint32_t)vZ, pNe12Fdv);
    const int cI11 = cDm.x;
    const int cI12 = cDm.y;

    const int cI01 = pSrc1Ptr[cI10 * pS10 + cI11 * pS11 + cI12 * pS12];

    int4 *GGML_CUDA_RESTRICT vDstRow = (int4 *)(pDstPtr + cI10 * pS1 + cI11 * pS2 + cI12 * pS3);
    const int4 *GGML_CUDA_RESTRICT vSrc0Row =
        (const int4 *)((const char *)pSrc0Ptr + cI01 * pNb01 + cI11 * pNb02 + cI12 * pNb03);

    for (int64_t vI = blockIdx.y * blockDim.x + threadIdx.x; vI < pNe00v; vI += gridDim.y * blockDim.x) {
      vDstRow[vI] = vSrc0Row[vI];
    }
  }
}

template <typename grad_t, typename dst_t>
static __global__ void fKGetRowsBackFloat(const grad_t *__restrict__ pGrad, const int32_t *__restrict__ pRows,
                                          dst_t *__restrict__ pDst, const int64_t pNcols, const int64_t pNrowsGrad,
                                          const int64_t pNrowsDst) {
  const int cCol = blockIdx.x * blockDim.x + threadIdx.x;

  if (cCol >= pNcols) {
    return;
  }

  ggml_cuda_pdl_sync();

  // grid.y is clamped to the CUDA grid limit, so stride over the destination rows
  for (int64_t vDstRow = blockIdx.y; vDstRow < pNrowsDst; vDstRow += gridDim.y) {
    float vSum = 0.0f;

    for (int64_t vI = 0; vI < pNrowsGrad; ++vI) {
      if (pRows[vI] != vDstRow) {
        continue;
      }
      vSum += pGrad[vI * pNcols + cCol];
    }

    pDst[vDstRow * pNcols + cCol] = vSum;
  }
}

template <int qk, int qr, dequantize_kernel_t dq, typename dst_t>
static void fGetRowsCudaQ(const void *pSrc0D, const int32_t *pSrc1D, dst_t *pDstD, const int64_t pNe00,
                          const size_t pNb01, const size_t pNb02, const size_t pNb03, const int64_t pNe10,
                          const int64_t pNe11, const int64_t pNe12, const size_t pNb10, const size_t pNb11,
                          const size_t pNb12, const size_t pNb1, const size_t pNb2, const size_t pNb3,
                          cudaStream_t pStream) {
  const dim3 cBlockDims(CUDA_GET_ROWS_BLOCK_SIZE, 1, 1);
  const int cBlockNumY = (pNe00 + 2 * CUDA_GET_ROWS_BLOCK_SIZE - 1) / (2 * CUDA_GET_ROWS_BLOCK_SIZE);
  const dim3 cBlockNums(pNe10, MIN(cBlockNumY, UINT16_MAX), MIN(pNe11 * pNe12, UINT16_MAX));

  // strides in elements
  // const size_t s0 = nb0 / sizeof(dst_t);
  const size_t cS1 = pNb1 / sizeof(dst_t);
  const size_t cS2 = pNb2 / sizeof(dst_t);
  const size_t cS3 = pNb3 / sizeof(dst_t);

  const size_t cS10 = pNb10 / sizeof(int32_t);
  const size_t cS11 = pNb11 / sizeof(int32_t);
  const size_t cS12 = pNb12 / sizeof(int32_t);
  // const size_t s13 = nb13 / sizeof(int32_t);

  GGML_ASSERT(pNe00 % 2 == 0);

  GGML_ASSERT(pNe12 > 0);
  GGML_ASSERT(pNe11 <= std::numeric_limits<uint32_t>::max() / pNe12);
  const uint3 cNe12Fdv = init_fastdiv_values(pNe12);

  fKGetRows<qk, qr, dq><<<cBlockNums, cBlockDims, 0, pStream>>>(pSrc0D, pSrc1D, pDstD, pNe00, /*ne01, ne02, ne03,*/
                                                                /*ne10,*/ pNe11, cNe12Fdv,    /*ne13,*/
                                                                /* s0,*/ cS1, cS2, cS3,
                                                                /* nb00,*/ pNb01, pNb02, pNb03, cS10, cS11,
                                                                cS12 /*, s13*/);
}

template <int block_dim, typename dst_t, dequantize_kq_t<dst_t> dequantize_kq>
static void fGetRowsCudaKq(const void *pSrc0D, const int32_t *pSrc1D, dst_t *pDstD, const int64_t pNe00,
                           const size_t pNb01, const size_t pNb02, const size_t pNb03, const int64_t pNe10,
                           const int64_t pNe11, const int64_t pNe12, const size_t pNb10, const size_t pNb11,
                           const size_t pNb12, const size_t pNb1, const size_t pNb2, const size_t pNb3,
                           cudaStream_t pStream) {
  GGML_ASSERT(pNe00 % QK_K == 0);
  const int64_t cNsb = pNe00 / QK_K;

  const dim3 cBlockDims(block_dim, 1, 1);
  const dim3 cBlockNums(pNe10, MIN(cNsb, UINT16_MAX), MIN(pNe11 * pNe12, UINT16_MAX));

  // strides in elements
  // const size_t s0 = nb0 / sizeof(dst_t);
  const size_t cS1 = pNb1 / sizeof(dst_t);
  const size_t cS2 = pNb2 / sizeof(dst_t);
  const size_t cS3 = pNb3 / sizeof(dst_t);

  const size_t cS10 = pNb10 / sizeof(int32_t);
  const size_t cS11 = pNb11 / sizeof(int32_t);
  const size_t cS12 = pNb12 / sizeof(int32_t);
  // const size_t s13 = nb13 / sizeof(int32_t);

  GGML_ASSERT(pNe12 > 0);
  GGML_ASSERT(pNe11 <= std::numeric_limits<uint32_t>::max() / pNe12);
  const uint3 cNe12Fdv = init_fastdiv_values(pNe12);

  fKGetRowsKq<dst_t, dequantize_kq>
      <<<cBlockNums, cBlockDims, 0, pStream>>>(pSrc0D, pSrc1D, pDstD, pNe00, /*ne01, ne02, ne03,*/
                                               /*ne10,*/ pNe11, cNe12Fdv,    /*ne13,*/
                                               /* s0,*/ cS1, cS2, cS3,
                                               /* nb00,*/ pNb01, pNb02, pNb03, cS10, cS11, cS12 /*, s13*/);
}

template <typename src0_t, typename dst_t>
static void fGetRowsCudaFloat(const src0_t *pSrc0D, const int32_t *pSrc1D, dst_t *pDstD, const int64_t pNe00,
                              const size_t pNb01, const size_t pNb02, const size_t pNb03, const int64_t pNe10,
                              const int64_t pNe11, const int64_t pNe12, const size_t pNb10, const size_t pNb11,
                              const size_t pNb12, const size_t pNb1, const size_t pNb2, const size_t pNb3,
                              cudaStream_t pStream) {
  const dim3 cBlockDims(CUDA_GET_ROWS_BLOCK_SIZE, 1, 1);

  // strides in elements
  // const size_t s0 = nb0 / sizeof(dst_t);
  const size_t cS1 = pNb1 / sizeof(dst_t);
  const size_t cS2 = pNb2 / sizeof(dst_t);
  const size_t cS3 = pNb3 / sizeof(dst_t);

  const size_t cS10 = pNb10 / sizeof(int32_t);
  const size_t cS11 = pNb11 / sizeof(int32_t);
  const size_t cS12 = pNb12 / sizeof(int32_t);
  // const size_t s13 = nb13 / sizeof(int32_t);

  GGML_ASSERT(pNe12 > 0);
  GGML_ASSERT(pNe11 <= std::numeric_limits<uint32_t>::max() / pNe12);
  const uint3 cNe12Fdv = init_fastdiv_values(pNe12);

  if constexpr (std::is_same<src0_t, dst_t>::value) {
    constexpr int cVec = 16 / sizeof(dst_t);
    const int64_t cNe00v = pNe00 / cVec;
    const int64_t cVecBlockNumY = (cNe00v + CUDA_GET_ROWS_BLOCK_SIZE - 1) / CUDA_GET_ROWS_BLOCK_SIZE;
    const bool cEnoughBlocks = cVecBlockNumY * pNe10 * pNe11 * pNe12 >= 128;
    const bool cCanVec = cVec > 1 && cEnoughBlocks && (pNe00 % cVec == 0) && (pNb01 % 16 == 0) && (pNb02 % 16 == 0) &&
                         (pNb03 % 16 == 0) && (pNb1 % 16 == 0) && (pNb2 % 16 == 0) && (pNb3 % 16 == 0) &&
                         (((uintptr_t)pSrc0D) % 16 == 0) && (((uintptr_t)pDstD) % 16 == 0);

    if (cCanVec) {
      const int cBlockNumY = cVecBlockNumY;
      const dim3 cBlockNums(pNe10, MIN(cBlockNumY, UINT16_MAX), MIN(pNe11 * pNe12, UINT16_MAX));
      const ggml_cuda_kernel_launch_params cLaunchParams =
          ggml_cuda_kernel_launch_params{cBlockNums, cBlockDims, 0, pStream};
      ggml_cuda_kernel_launch(fKGetRowsFloatVec<dst_t>, cLaunchParams, (const dst_t *)pSrc0D, pSrc1D, pDstD, cNe00v,
                              pNe11, cNe12Fdv, cS1, cS2, cS3, pNb01, pNb02, pNb03, cS10, cS11, cS12);
      return;
    }
  }

  const int cBlockNumY = (pNe00 + CUDA_GET_ROWS_BLOCK_SIZE - 1) / CUDA_GET_ROWS_BLOCK_SIZE;
  const dim3 cBlockNums(pNe10, MIN(cBlockNumY, UINT16_MAX), MIN(pNe11 * pNe12, UINT16_MAX));

  const ggml_cuda_kernel_launch_params cLaunchParams =
      ggml_cuda_kernel_launch_params{cBlockNums, cBlockDims, 0, pStream};
  ggml_cuda_kernel_launch(fKGetRowsFloat<src0_t, dst_t>, cLaunchParams, pSrc0D, pSrc1D, pDstD,
                          pNe00,                     /*ne01, ne02, ne03,*/
                          /*ne10,*/ pNe11, cNe12Fdv, /*ne13,*/
                          /* s0,*/ cS1, cS2, cS3,
                          /* nb00,*/ pNb01, pNb02, pNb03, cS10, cS11, cS12 /*, s13*/);
}

template <typename dst_t>
static void fGgmlCudaGetRowsSwitchSrc0Type(const void *pSrc0D, const ggml_type pSrc0Type, const int32_t *pSrc1D,
                                           dst_t *pDstD, const int64_t pNe00, const size_t pNb01, const size_t pNb02,
                                           const size_t pNb03, const int64_t pNe10, const int64_t pNe11,
                                           const int64_t pNe12, const size_t pNb10, const size_t pNb11,
                                           const size_t pNb12, const size_t pNb1, const size_t pNb2, const size_t pNb3,
                                           cudaStream_t pStream) {
  switch (pSrc0Type) {
  case GGML_TYPE_F16:
    fGetRowsCudaFloat((const half *)pSrc0D, pSrc1D, pDstD, pNe00, pNb01, pNb02, pNb03, pNe10, pNe11, pNe12, pNb10,
                      pNb11, pNb12, pNb1, pNb2, pNb3, pStream);
    break;
  case GGML_TYPE_F32:
    fGetRowsCudaFloat((const float *)pSrc0D, pSrc1D, pDstD, pNe00, pNb01, pNb02, pNb03, pNe10, pNe11, pNe12, pNb10,
                      pNb11, pNb12, pNb1, pNb2, pNb3, pStream);
    break;
  case GGML_TYPE_I32:
    fGetRowsCudaFloat((const int32_t *)pSrc0D, pSrc1D, pDstD, pNe00, pNb01, pNb02, pNb03, pNe10, pNe11, pNe12, pNb10,
                      pNb11, pNb12, pNb1, pNb2, pNb3, pStream);
    break;
  case GGML_TYPE_BF16:
    fGetRowsCudaFloat((const nv_bfloat16 *)pSrc0D, pSrc1D, pDstD, pNe00, pNb01, pNb02, pNb03, pNe10, pNe11, pNe12,
                      pNb10, pNb11, pNb12, pNb1, pNb2, pNb3, pStream);
    break;
  case GGML_TYPE_Q1_0:
    fGetRowsCudaQ<QK1_0, QR1_0, dequantize_q1_0>(pSrc0D, pSrc1D, pDstD, pNe00, pNb01, pNb02, pNb03, pNe10, pNe11, pNe12,
                                                 pNb10, pNb11, pNb12, pNb1, pNb2, pNb3, pStream);
    break;
  case GGML_TYPE_Q2_0:
    fGetRowsCudaQ<QK2_0, QR2_0, dequantize_q2_0>(pSrc0D, pSrc1D, pDstD, pNe00, pNb01, pNb02, pNb03, pNe10, pNe11, pNe12,
                                                 pNb10, pNb11, pNb12, pNb1, pNb2, pNb3, pStream);
    break;
  case GGML_TYPE_Q4_0:
    fGetRowsCudaQ<QK4_0, QR4_0, dequantize_q4_0>(pSrc0D, pSrc1D, pDstD, pNe00, pNb01, pNb02, pNb03, pNe10, pNe11, pNe12,
                                                 pNb10, pNb11, pNb12, pNb1, pNb2, pNb3, pStream);
    break;
  case GGML_TYPE_Q4_1:
    fGetRowsCudaQ<QK4_1, QR4_1, dequantize_q4_1>(pSrc0D, pSrc1D, pDstD, pNe00, pNb01, pNb02, pNb03, pNe10, pNe11, pNe12,
                                                 pNb10, pNb11, pNb12, pNb1, pNb2, pNb3, pStream);
    break;
  case GGML_TYPE_Q5_0:
    fGetRowsCudaQ<QK5_0, QR5_0, dequantize_q5_0>(pSrc0D, pSrc1D, pDstD, pNe00, pNb01, pNb02, pNb03, pNe10, pNe11, pNe12,
                                                 pNb10, pNb11, pNb12, pNb1, pNb2, pNb3, pStream);
    break;
  case GGML_TYPE_Q5_1:
    fGetRowsCudaQ<QK5_1, QR5_1, dequantize_q5_1>(pSrc0D, pSrc1D, pDstD, pNe00, pNb01, pNb02, pNb03, pNe10, pNe11, pNe12,
                                                 pNb10, pNb11, pNb12, pNb1, pNb2, pNb3, pStream);
    break;
  case GGML_TYPE_Q8_0:
    fGetRowsCudaQ<QK8_0, QR8_0, dequantize_q8_0>(pSrc0D, pSrc1D, pDstD, pNe00, pNb01, pNb02, pNb03, pNe10, pNe11, pNe12,
                                                 pNb10, pNb11, pNb12, pNb1, pNb2, pNb3, pStream);
    break;
  case GGML_TYPE_Q2_K:
    fGetRowsCudaKq<64, dst_t, dequantize_q2_K<dst_t>>(pSrc0D, pSrc1D, pDstD, pNe00, pNb01, pNb02, pNb03, pNe10, pNe11,
                                                      pNe12, pNb10, pNb11, pNb12, pNb1, pNb2, pNb3, pStream);
    break;
  case GGML_TYPE_Q3_K:
    fGetRowsCudaKq<64, dst_t, dequantize_q3_K<dst_t>>(pSrc0D, pSrc1D, pDstD, pNe00, pNb01, pNb02, pNb03, pNe10, pNe11,
                                                      pNe12, pNb10, pNb11, pNb12, pNb1, pNb2, pNb3, pStream);
    break;
  case GGML_TYPE_Q4_K:
    fGetRowsCudaKq<32, dst_t, dequantize_q4_K<dst_t>>(pSrc0D, pSrc1D, pDstD, pNe00, pNb01, pNb02, pNb03, pNe10, pNe11,
                                                      pNe12, pNb10, pNb11, pNb12, pNb1, pNb2, pNb3, pStream);
    break;
  case GGML_TYPE_Q5_K:
    fGetRowsCudaKq<64, dst_t, dequantize_q5_K<dst_t>>(pSrc0D, pSrc1D, pDstD, pNe00, pNb01, pNb02, pNb03, pNe10, pNe11,
                                                      pNe12, pNb10, pNb11, pNb12, pNb1, pNb2, pNb3, pStream);
    break;
  case GGML_TYPE_Q6_K:
    fGetRowsCudaKq<64, dst_t, dequantize_q6_K<dst_t>>(pSrc0D, pSrc1D, pDstD, pNe00, pNb01, pNb02, pNb03, pNe10, pNe11,
                                                      pNe12, pNb10, pNb11, pNb12, pNb1, pNb2, pNb3, pStream);
    break;
  case GGML_TYPE_IQ2_XXS:
    fGetRowsCudaKq<32, dst_t, dequantize_iq2_xxs<dst_t>>(pSrc0D, pSrc1D, pDstD, pNe00, pNb01, pNb02, pNb03, pNe10,
                                                         pNe11, pNe12, pNb10, pNb11, pNb12, pNb1, pNb2, pNb3, pStream);
    break;
  case GGML_TYPE_IQ2_XS:
    fGetRowsCudaKq<32, dst_t, dequantize_iq2_xs<dst_t>>(pSrc0D, pSrc1D, pDstD, pNe00, pNb01, pNb02, pNb03, pNe10, pNe11,
                                                        pNe12, pNb10, pNb11, pNb12, pNb1, pNb2, pNb3, pStream);
    break;
  case GGML_TYPE_IQ2_S:
    fGetRowsCudaKq<32, dst_t, dequantize_iq2_s<dst_t>>(pSrc0D, pSrc1D, pDstD, pNe00, pNb01, pNb02, pNb03, pNe10, pNe11,
                                                       pNe12, pNb10, pNb11, pNb12, pNb1, pNb2, pNb3, pStream);
    break;
  case GGML_TYPE_IQ3_XXS:
    fGetRowsCudaKq<32, dst_t, dequantize_iq3_xxs<dst_t>>(pSrc0D, pSrc1D, pDstD, pNe00, pNb01, pNb02, pNb03, pNe10,
                                                         pNe11, pNe12, pNb10, pNb11, pNb12, pNb1, pNb2, pNb3, pStream);
    break;
  case GGML_TYPE_IQ3_S:
    fGetRowsCudaKq<32, dst_t, dequantize_iq3_s<dst_t>>(pSrc0D, pSrc1D, pDstD, pNe00, pNb01, pNb02, pNb03, pNe10, pNe11,
                                                       pNe12, pNb10, pNb11, pNb12, pNb1, pNb2, pNb3, pStream);
    break;
  case GGML_TYPE_IQ1_S:
    fGetRowsCudaKq<32, dst_t, dequantize_iq1_s<dst_t>>(pSrc0D, pSrc1D, pDstD, pNe00, pNb01, pNb02, pNb03, pNe10, pNe11,
                                                       pNe12, pNb10, pNb11, pNb12, pNb1, pNb2, pNb3, pStream);
    break;
  case GGML_TYPE_IQ1_M:
    fGetRowsCudaKq<32, dst_t, dequantize_iq1_m<dst_t>>(pSrc0D, pSrc1D, pDstD, pNe00, pNb01, pNb02, pNb03, pNe10, pNe11,
                                                       pNe12, pNb10, pNb11, pNb12, pNb1, pNb2, pNb3, pStream);
    break;
  case GGML_TYPE_IQ4_NL:
    fGetRowsCudaKq<32, dst_t, dequantize_iq4_nl<dst_t>>(pSrc0D, pSrc1D, pDstD, pNe00, pNb01, pNb02, pNb03, pNe10, pNe11,
                                                        pNe12, pNb10, pNb11, pNb12, pNb1, pNb2, pNb3, pStream);
    break;
  case GGML_TYPE_IQ4_XS:
    fGetRowsCudaKq<32, dst_t, dequantize_iq4_xs<dst_t>>(pSrc0D, pSrc1D, pDstD, pNe00, pNb01, pNb02, pNb03, pNe10, pNe11,
                                                        pNe12, pNb10, pNb11, pNb12, pNb1, pNb2, pNb3, pStream);
    break;
  case GGML_TYPE_MXFP4:
    fGetRowsCudaKq<32, dst_t, dequantize_mxfp4<dst_t>>(pSrc0D, pSrc1D, pDstD, pNe00, pNb01, pNb02, pNb03, pNe10, pNe11,
                                                       pNe12, pNb10, pNb11, pNb12, pNb1, pNb2, pNb3, pStream);
    break;
  default:
    GGML_ABORT("%s: unsupported src0 type: %s\n", __func__, ggml_type_name(pSrc0Type));
    break;
  }
}

void get_rows_cuda(const void *pSrc0D, ggml_type pSrc0Type, const int32_t *pSrc1D, void *pDstD, ggml_type pDstType,
                   int64_t pNe00, size_t pNb01, size_t pNb02, size_t pNb03, int64_t pNe10, int64_t pNe11, int64_t pNe12,
                   size_t pNb10, size_t pNb11, size_t pNb12, size_t pNb1, size_t pNb2, size_t pNb3,
                   cudaStream_t pStream) {
  switch (pDstType) {
  case GGML_TYPE_F32:
    fGgmlCudaGetRowsSwitchSrc0Type(pSrc0D, pSrc0Type, pSrc1D, (float *)pDstD, pNe00, pNb01, pNb02, pNb03, pNe10, pNe11,
                                   pNe12, pNb10, pNb11, pNb12, pNb1, pNb2, pNb3, pStream);
    break;
  case GGML_TYPE_I32:
    fGgmlCudaGetRowsSwitchSrc0Type(pSrc0D, pSrc0Type, pSrc1D, (int32_t *)pDstD, pNe00, pNb01, pNb02, pNb03, pNe10,
                                   pNe11, pNe12, pNb10, pNb11, pNb12, pNb1, pNb2, pNb3, pStream);
    break;
  case GGML_TYPE_F16:
    fGgmlCudaGetRowsSwitchSrc0Type(pSrc0D, pSrc0Type, pSrc1D, (half *)pDstD, pNe00, pNb01, pNb02, pNb03, pNe10, pNe11,
                                   pNe12, pNb10, pNb11, pNb12, pNb1, pNb2, pNb3, pStream);
    break;
  case GGML_TYPE_BF16:
    fGgmlCudaGetRowsSwitchSrc0Type(pSrc0D, pSrc0Type, pSrc1D, (nv_bfloat16 *)pDstD, pNe00, pNb01, pNb02, pNb03, pNe10,
                                   pNe11, pNe12, pNb10, pNb11, pNb12, pNb1, pNb2, pNb3, pStream);
    break;
  default:
    GGML_ABORT("%s: unsupported dst type: %s\n", __func__, ggml_type_name(pDstType));
    break;
  }
}

void ggml_cuda_op_get_rows(ggml_backend_cuda_context &pCtx, ggml_tensor *dst) {
  const ggml_tensor *src0 = dst->src[0];
  const ggml_tensor *src1 = dst->src[1];

  cudaStream_t vStream = pCtx.stream();

  GGML_TENSOR_BINARY_OP_LOCALS

  GGML_ASSERT(src1->type == GGML_TYPE_I32);
  GGML_ASSERT(ne13 == 1);

  GGML_ASSERT(src0->nb[0] == ggml_type_size(src0->type));
  GGML_ASSERT(src1->nb[0] == ggml_type_size(src1->type));
  GGML_ASSERT(dst->nb[0] == ggml_type_size(dst->type));

  get_rows_cuda(src0->data, src0->type, (const int32_t *)src1->data, dst->data, dst->type, ne00, nb01, nb02, nb03, ne10,
                ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, vStream);
}

void ggml_cuda_op_get_rows_back(ggml_backend_cuda_context &pCtx, ggml_tensor *dst) {
  const ggml_tensor *src0 = dst->src[0]; // gradients of forward pass output
  const ggml_tensor *src1 = dst->src[1]; // src1 in forward pass

  GGML_TENSOR_BINARY_OP_LOCALS

  const float *vSrc0D = (const float *)src0->data;
  const int32_t *vSrc1D = (const int32_t *)src1->data;
  float *vDstD = (float *)dst->data;

  cudaStream_t vStream = pCtx.stream();

  GGML_ASSERT(src0->type == GGML_TYPE_F32);
  GGML_ASSERT(src1->type == GGML_TYPE_I32);
  GGML_ASSERT(dst->type == GGML_TYPE_F32);

  GGML_ASSERT(ggml_is_contiguous(src0));
  GGML_ASSERT(ggml_is_contiguous(src1));
  GGML_ASSERT(ggml_is_contiguous(dst));

  GGML_ASSERT(ne02 * ne03 == 1);
  GGML_ASSERT(ne12 * ne13 == 1);
  GGML_ASSERT(ne2 * ne3 == 1);

  const dim3 cBlockDims(CUDA_GET_ROWS_BACK_BLOCK_SIZE, 1, 1);
  const int cBlockNumX = (ne00 + CUDA_GET_ROWS_BACK_BLOCK_SIZE - 1) / CUDA_GET_ROWS_BACK_BLOCK_SIZE;
  const dim3 cBlockNums(cBlockNumX, MIN(ne1, (int64_t)UINT16_MAX), 1);

  fKGetRowsBackFloat<<<cBlockNums, cBlockDims, 0, vStream>>>(vSrc0D, vSrc1D, vDstD, ne00, ne10, ne1);
}
