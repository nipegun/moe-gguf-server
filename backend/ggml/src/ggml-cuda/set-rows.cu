#include "set-rows.cuh"
#include "cpy-utils.cuh"

typedef void (*set_rows_kernel_t)(const char *pSrc, char *pDst);

// Generic quantized set_rows kernel template
template <typename idx_t, typename block_type, int qk, void (*quantize_func)(const float *, block_type *)>
static __global__ void
fKSetRowsQuant(const float *__restrict__ pSrc0, const idx_t *__restrict__ pSrc1, block_type *__restrict__ pDst,
               const int64_t pNeTotal, const int64_t pNe10, const int64_t pNe11, const int64_t pNe12,
               const int64_t pNe13, const int64_t pS01, const int64_t pS02, const int64_t pS03, const int64_t pS10,
               const int64_t pS11, const int64_t pS12, const int64_t pS1, const int64_t pS2, const int64_t pS3,
               const uint3 pNe00, const uint3 pNe01, const uint3 pNe02, const uint3 pNe11Fd, const uint3 pNe12Fd) {
  const int64_t cI = int64_t(blockDim.x) * blockIdx.x + threadIdx.x;

  if (cI >= pNeTotal) {
    return;
  }

  const int64_t cIBase = cI * qk;
  uint32_t vTmp = (uint32_t)cIBase;
  uint2 vDivMod;

  vDivMod = fast_div_modulo(vTmp, pNe00);
  const int64_t cI00 = vDivMod.y;
  vTmp = vDivMod.x;

  vDivMod = fast_div_modulo(vTmp, pNe01);
  const int64_t cI01 = vDivMod.y;
  vTmp = vDivMod.x;

  vDivMod = fast_div_modulo(vTmp, pNe02);
  const int64_t cI02 = vDivMod.y;
  const int64_t cI03 = vDivMod.x;

  const int64_t cI12 = fastmodulo((uint32_t)cI03, pNe12Fd);
  const int64_t cI11 = fastmodulo((uint32_t)cI02, pNe11Fd);
  const int64_t cI10 = cI01;

  ggml_cuda_pdl_sync();
  const int64_t cDstRow = *(pSrc1 + cI10 * pS10 + cI11 * pS11 + cI12 * pS12);

  const float *vSrc0Row = pSrc0 + cI01 * pS01 + cI02 * pS02 + cI03 * pS03;
  block_type *vDstRowPtr = pDst + (cDstRow * pS1 + cI02 * pS2 + cI03 * pS3) / sizeof(block_type);

  const float *vSrcBlock = vSrc0Row + cI00;
  block_type *vDstBlock = vDstRowPtr + cI00 / qk;

  quantize_func(vSrcBlock, vDstBlock);

  GGML_UNUSED(pNe10);
  GGML_UNUSED(pNe11);
  GGML_UNUSED(pNe12);
  GGML_UNUSED(pNe13);
}

// Template dispatch function for quantized set_rows
template <typename idx_t, typename block_type, int qk, void (*quantize_func)(const float *, block_type *)>
static void fSetRowsCudaQuant(const float *pSrc0D, const idx_t *pSrc1D, block_type *pDstD, const int64_t pNe00,
                              const int64_t pNe01, const int64_t pNe02, const int64_t pNe03, const int64_t pNe10,
                              const int64_t pNe11, const int64_t pNe12, const int64_t pNe13, const size_t pNb01,
                              const size_t pNb02, const size_t pNb03, const size_t pNb10, const size_t pNb11,
                              const size_t pNb12, const size_t pNb1, const size_t pNb2, const size_t pNb3,
                              cudaStream_t pStream) {

  GGML_ASSERT(pNe00 % qk == 0);
  const int64_t cNeTotal = (pNe00 * pNe01 * pNe02 * pNe03) / qk;
  const int cNumBlocks = (cNeTotal + CUDA_SET_ROWS_BLOCK_SIZE - 1) / CUDA_SET_ROWS_BLOCK_SIZE;
  const dim3 cBlockSize(CUDA_SET_ROWS_BLOCK_SIZE);
  const dim3 cGridSize(cNumBlocks);

  const int64_t cS01 = pNb01 / sizeof(float);
  const int64_t cS02 = pNb02 / sizeof(float);
  const int64_t cS03 = pNb03 / sizeof(float);
  const int64_t cS10 = pNb10 / sizeof(idx_t);
  const int64_t cS11 = pNb11 / sizeof(idx_t);
  const int64_t cS12 = pNb12 / sizeof(idx_t);
  const int64_t cS1 = pNb1;
  const int64_t cS2 = pNb2;
  const int64_t cS3 = pNb3;

  if (cNeTotal > 0 && pNe00 > 0 && pNe01 > 0 && pNe02 > 0 && pNe11 > 0 && pNe12 > 0) {
    const uint3 cNe00Fd = init_fastdiv_values((uint32_t)pNe00);
    const uint3 cNe01Fd = init_fastdiv_values((uint32_t)pNe01);
    const uint3 cNe02Fd = init_fastdiv_values((uint32_t)pNe02);
    const uint3 cNe11Fd = init_fastdiv_values((uint32_t)pNe11);
    const uint3 cNe12Fd = init_fastdiv_values((uint32_t)pNe12);

    fKSetRowsQuant<idx_t, block_type, qk, quantize_func><<<cGridSize, cBlockSize, 0, pStream>>>(
        pSrc0D, pSrc1D, pDstD, cNeTotal, pNe10, pNe11, pNe12, pNe13, cS01, cS02, cS03, cS10, cS11, cS12, cS1, cS2, cS3,
        cNe00Fd, cNe01Fd, cNe02Fd, cNe11Fd, cNe12Fd);
  }
}

template <typename src_t, typename idx_t, typename dst_t>
static __global__ void fKSetRows(const src_t *pSrc0Ptr, const idx_t *pSrc1Ptr, dst_t *pDstPtr, const int64_t pNeTotal,
                                 const int64_t pNe10, const int64_t pNe11, const int64_t pNe12, const int64_t pNe13,
                                 const int64_t pS01, const int64_t pS02, const int64_t pS03, const int64_t pS10,
                                 const int64_t pS11, const int64_t pS12, const int64_t pS1, const int64_t pS2,
                                 const int64_t pS3, const uint3 pNe00, const uint3 pNe01, const uint3 pNe02,
                                 const uint3 pNe11Fd, const uint3 pNe12Fd) {
  const src_t *GGML_CUDA_RESTRICT vSrc0 = pSrc0Ptr;
  const idx_t *GGML_CUDA_RESTRICT vSrc1 = pSrc1Ptr;
  dst_t *GGML_CUDA_RESTRICT vDst = pDstPtr;
  const int64_t cI = int64_t(blockDim.x) * blockIdx.x + threadIdx.x;

  if (cI >= pNeTotal) {
    return;
  }

  uint32_t vTmp = (uint32_t)cI;
  uint2 vDivMod;

  vDivMod = fast_div_modulo(vTmp, pNe00);
  const int64_t cI00 = vDivMod.y;
  vTmp = vDivMod.x;

  vDivMod = fast_div_modulo(vTmp, pNe01);
  const int64_t cI01 = vDivMod.y;
  vTmp = vDivMod.x;

  vDivMod = fast_div_modulo(vTmp, pNe02);
  const int64_t cI02 = vDivMod.y;
  const int64_t cI03 = vDivMod.x;

  const int64_t cI12 = fastmodulo((uint32_t)cI03, pNe12Fd);
  const int64_t cI11 = fastmodulo((uint32_t)cI02, pNe11Fd);
  const int64_t cI10 = cI01;

  ggml_cuda_pdl_sync();
  const int64_t cDstRow = *(vSrc1 + cI10 * pS10 + cI11 * pS11 + cI12 * pS12);
  ggml_cuda_pdl_lc();

  const src_t *vSrc0Row = vSrc0 + cI01 * pS01 + cI02 * pS02 + cI03 * pS03;
  dst_t *vDstRowPtr = vDst + cDstRow * pS1 + cI02 * pS2 + cI03 * pS3;

  vDstRowPtr[cI00] = ggml_cuda_cast<dst_t>(vSrc0Row[cI00]);

  GGML_UNUSED(pNe10);
  GGML_UNUSED(pNe11);
  GGML_UNUSED(pNe12);
  GGML_UNUSED(pNe13);
}

template <typename src_t, typename idx_t, typename dst_t>
static void set_rows_cuda(const src_t *pSrc0D, const idx_t *pSrc1D, dst_t *pDstD, const int64_t pNe00,
                          const int64_t pNe01, const int64_t pNe02, const int64_t pNe03, const int64_t pNe10,
                          const int64_t pNe11, const int64_t pNe12, const int64_t pNe13, const size_t pNb01,
                          const size_t pNb02, const size_t pNb03, const size_t pNb10, const size_t pNb11,
                          const size_t pNb12, const size_t pNb1, const size_t pNb2, const size_t pNb3,
                          cudaStream_t pStream) {

  const int64_t cNeTotal = pNe00 * pNe01 * pNe02 * pNe03;
  const int cNumBlocks = (cNeTotal + CUDA_SET_ROWS_BLOCK_SIZE - 1) / CUDA_SET_ROWS_BLOCK_SIZE;
  const dim3 cBlockSize(CUDA_SET_ROWS_BLOCK_SIZE);
  const dim3 cGridSize(cNumBlocks);

  const int64_t cS01 = pNb01 / sizeof(src_t);
  const int64_t cS02 = pNb02 / sizeof(src_t);
  const int64_t cS03 = pNb03 / sizeof(src_t);
  const int64_t cS10 = pNb10 / sizeof(idx_t);
  const int64_t cS11 = pNb11 / sizeof(idx_t);
  const int64_t cS12 = pNb12 / sizeof(idx_t);
  const int64_t cS1 = pNb1 / sizeof(dst_t);
  const int64_t cS2 = pNb2 / sizeof(dst_t);
  const int64_t cS3 = pNb3 / sizeof(dst_t);

  if (cNeTotal > 0 && pNe00 > 0 && pNe01 > 0 && pNe02 > 0 && pNe11 > 0 && pNe12 > 0) {
    const uint3 cNe00Fd = init_fastdiv_values((uint32_t)pNe00);
    const uint3 cNe01Fd = init_fastdiv_values((uint32_t)pNe01);
    const uint3 cNe02Fd = init_fastdiv_values((uint32_t)pNe02);
    const uint3 cNe11Fd = init_fastdiv_values((uint32_t)pNe11);
    const uint3 cNe12Fd = init_fastdiv_values((uint32_t)pNe12);

    const ggml_cuda_kernel_launch_params cLaunchParams =
        ggml_cuda_kernel_launch_params(cGridSize, cBlockSize, 0, pStream);
    ggml_cuda_kernel_launch(fKSetRows<src_t, idx_t, dst_t>, cLaunchParams, pSrc0D, pSrc1D, pDstD, cNeTotal, pNe10,
                            pNe11, pNe12, pNe13, cS01, cS02, cS03, cS10, cS11, cS12, cS1, cS2, cS3, cNe00Fd, cNe01Fd,
                            cNe02Fd, cNe11Fd, cNe12Fd);
  }
}

template <typename src_t, typename idx_t>
static void set_rows_cuda(ggml_backend_cuda_context &pCtx, const ggml_tensor *src0, const ggml_tensor *src1,
                          ggml_tensor *dst) {
  const src_t *vSrc0D = (const src_t *)src0->data;
  const idx_t *vSrc1D = (const idx_t *)src1->data;

  GGML_TENSOR_BINARY_OP_LOCALS

  cudaStream_t vStream = pCtx.stream();

  if (dst->type == GGML_TYPE_F32) {
    set_rows_cuda(vSrc0D, vSrc1D, (float *)dst->data, ne00, ne01, ne02, ne03, ne10, ne11, ne12, ne13, nb01, nb02, nb03,
                  nb10, nb11, nb12, nb1, nb2, nb3, vStream);
  } else if (dst->type == GGML_TYPE_F16) {
    set_rows_cuda(vSrc0D, vSrc1D, (half *)dst->data, ne00, ne01, ne02, ne03, ne10, ne11, ne12, ne13, nb01, nb02, nb03,
                  nb10, nb11, nb12, nb1, nb2, nb3, vStream);
  } else if (dst->type == GGML_TYPE_BF16) {
    set_rows_cuda(vSrc0D, vSrc1D, (nv_bfloat16 *)dst->data, ne00, ne01, ne02, ne03, ne10, ne11, ne12, ne13, nb01, nb02,
                  nb03, nb10, nb11, nb12, nb1, nb2, nb3, vStream);
  } else if (dst->type == GGML_TYPE_Q4_0) {
    fSetRowsCudaQuant<idx_t, block_q4_0, QK4_0, quantize_f32_q4_0_block>(
        vSrc0D, vSrc1D, (block_q4_0 *)dst->data, ne00, ne01, ne02, ne03, ne10, ne11, ne12, ne13, nb01, nb02, nb03, nb10,
        nb11, nb12, nb1, nb2, nb3, vStream);
  } else if (dst->type == GGML_TYPE_Q4_1) {
    fSetRowsCudaQuant<idx_t, block_q4_1, QK4_1, quantize_f32_q4_1_block>(
        vSrc0D, vSrc1D, (block_q4_1 *)dst->data, ne00, ne01, ne02, ne03, ne10, ne11, ne12, ne13, nb01, nb02, nb03, nb10,
        nb11, nb12, nb1, nb2, nb3, vStream);
  } else if (dst->type == GGML_TYPE_Q5_0) {
    fSetRowsCudaQuant<idx_t, block_q5_0, QK5_0, quantize_f32_q5_0_block>(
        vSrc0D, vSrc1D, (block_q5_0 *)dst->data, ne00, ne01, ne02, ne03, ne10, ne11, ne12, ne13, nb01, nb02, nb03, nb10,
        nb11, nb12, nb1, nb2, nb3, vStream);
  } else if (dst->type == GGML_TYPE_Q5_1) {
    fSetRowsCudaQuant<idx_t, block_q5_1, QK5_1, quantize_f32_q5_1_block>(
        vSrc0D, vSrc1D, (block_q5_1 *)dst->data, ne00, ne01, ne02, ne03, ne10, ne11, ne12, ne13, nb01, nb02, nb03, nb10,
        nb11, nb12, nb1, nb2, nb3, vStream);
  } else if (dst->type == GGML_TYPE_Q8_0) {
    fSetRowsCudaQuant<idx_t, block_q8_0, QK8_0, quantize_f32_q8_0_block>(
        vSrc0D, vSrc1D, (block_q8_0 *)dst->data, ne00, ne01, ne02, ne03, ne10, ne11, ne12, ne13, nb01, nb02, nb03, nb10,
        nb11, nb12, nb1, nb2, nb3, vStream);
  } else if (dst->type == GGML_TYPE_IQ4_NL) {
    fSetRowsCudaQuant<idx_t, block_iq4_nl, QK4_NL, quantize_f32_iq4_nl_block>(
        vSrc0D, vSrc1D, (block_iq4_nl *)dst->data, ne00, ne01, ne02, ne03, ne10, ne11, ne12, ne13, nb01, nb02, nb03,
        nb10, nb11, nb12, nb1, nb2, nb3, vStream);
  } else {
    GGML_ABORT("unsupported type %s", ggml_type_name(dst->type));
  }
}

template <>
void set_rows_cuda<half, int32_t>(ggml_backend_cuda_context &pCtx, const ggml_tensor *src0, const ggml_tensor *src1,
                                  ggml_tensor *dst) {
  const half *vSrc0D = (const half *)src0->data;
  const int32_t *vSrc1D = (const int32_t *)src1->data;

  GGML_TENSOR_BINARY_OP_LOCALS

  cudaStream_t vStream = pCtx.stream();

  if (dst->type == GGML_TYPE_F16) {
    set_rows_cuda(vSrc0D, vSrc1D, (half *)dst->data, ne00, ne01, ne02, ne03, ne10, ne11, ne12, ne13, nb01, nb02, nb03,
                  nb10, nb11, nb12, nb1, nb2, nb3, vStream);
  } else {
    GGML_ABORT("unsupported type %s", ggml_type_name(dst->type));
  }
}

template <>
void set_rows_cuda<half, int64_t>(ggml_backend_cuda_context &pCtx, const ggml_tensor *src0, const ggml_tensor *src1,
                                  ggml_tensor *dst) {
  const half *vSrc0D = (const half *)src0->data;
  const int64_t *vSrc1D = (const int64_t *)src1->data;

  GGML_TENSOR_BINARY_OP_LOCALS

  cudaStream_t vStream = pCtx.stream();

  if (dst->type == GGML_TYPE_F16) {
    set_rows_cuda(vSrc0D, vSrc1D, (half *)dst->data, ne00, ne01, ne02, ne03, ne10, ne11, ne12, ne13, nb01, nb02, nb03,
                  nb10, nb11, nb12, nb1, nb2, nb3, vStream);
  } else {
    GGML_ABORT("unsupported type %s", ggml_type_name(dst->type));
  }
}

void ggml_cuda_op_set_rows(ggml_backend_cuda_context &pCtx, ggml_tensor *pDst) {
  const ggml_tensor *vSrc0 = pDst->src[0];
  const ggml_tensor *vSrc1 = pDst->src[1];

  GGML_ASSERT(vSrc0->type == GGML_TYPE_F32 || (vSrc0->type == GGML_TYPE_F16 && pDst->type == GGML_TYPE_F16));
  GGML_ASSERT(vSrc1->type == GGML_TYPE_I64 || vSrc1->type == GGML_TYPE_I32);

  if (vSrc0->type == GGML_TYPE_F32) {
    if (vSrc1->type == GGML_TYPE_I64) {
      set_rows_cuda<float, int64_t>(pCtx, vSrc0, vSrc1, pDst);
    } else {
      set_rows_cuda<float, int32_t>(pCtx, vSrc0, vSrc1, pDst);
    }
  } else if (vSrc0->type == GGML_TYPE_F16) {
    if (vSrc1->type == GGML_TYPE_I64) {
      set_rows_cuda<half, int64_t>(pCtx, vSrc0, vSrc1, pDst);
    } else {
      set_rows_cuda<half, int32_t>(pCtx, vSrc0, vSrc1, pDst);
    }
  } else {
    GGML_ABORT("unsupported type %s", ggml_type_name(vSrc0->type));
  }
}
