#include "common.cuh"
#include "mmq.cuh"
#include "quantize.cuh"
#include "mmid.cuh"

#include <cstdint>

static void fGgmlCudaMulMatQSwitchType(ggml_backend_cuda_context &pCtx, const mmq_args &pArgs, cudaStream_t pStream) {
  switch (pArgs.type_x) {
  case GGML_TYPE_Q1_0:
    mul_mat_q_case<GGML_TYPE_Q1_0>(pCtx, pArgs, pStream);
    break;
  case GGML_TYPE_Q2_0:
    mul_mat_q_case<GGML_TYPE_Q2_0>(pCtx, pArgs, pStream);
    break;
  case GGML_TYPE_Q4_0:
    mul_mat_q_case<GGML_TYPE_Q4_0>(pCtx, pArgs, pStream);
    break;
  case GGML_TYPE_Q4_1:
    mul_mat_q_case<GGML_TYPE_Q4_1>(pCtx, pArgs, pStream);
    break;
  case GGML_TYPE_Q5_0:
    mul_mat_q_case<GGML_TYPE_Q5_0>(pCtx, pArgs, pStream);
    break;
  case GGML_TYPE_Q5_1:
    mul_mat_q_case<GGML_TYPE_Q5_1>(pCtx, pArgs, pStream);
    break;
  case GGML_TYPE_Q8_0:
    mul_mat_q_case<GGML_TYPE_Q8_0>(pCtx, pArgs, pStream);
    break;
    // -----------------------------------------------------------------------
  case GGML_TYPE_Q2_K:
    mul_mat_q_case<GGML_TYPE_Q2_K>(pCtx, pArgs, pStream);
    break;
  case GGML_TYPE_Q3_K:
    mul_mat_q_case<GGML_TYPE_Q3_K>(pCtx, pArgs, pStream);
    break;
  case GGML_TYPE_Q4_K:
    mul_mat_q_case<GGML_TYPE_Q4_K>(pCtx, pArgs, pStream);
    break;
  case GGML_TYPE_Q5_K:
    mul_mat_q_case<GGML_TYPE_Q5_K>(pCtx, pArgs, pStream);
    break;
  case GGML_TYPE_Q6_K:
    mul_mat_q_case<GGML_TYPE_Q6_K>(pCtx, pArgs, pStream);
    break;
    // -----------------------------------------------------------------------
  case GGML_TYPE_IQ1_S:
    mul_mat_q_case<GGML_TYPE_IQ1_S>(pCtx, pArgs, pStream);
    break;
  case GGML_TYPE_IQ2_XXS:
    mul_mat_q_case<GGML_TYPE_IQ2_XXS>(pCtx, pArgs, pStream);
    break;
  case GGML_TYPE_IQ2_XS:
    mul_mat_q_case<GGML_TYPE_IQ2_XS>(pCtx, pArgs, pStream);
    break;
  case GGML_TYPE_IQ2_S:
    mul_mat_q_case<GGML_TYPE_IQ2_S>(pCtx, pArgs, pStream);
    break;
  case GGML_TYPE_IQ3_XXS:
    mul_mat_q_case<GGML_TYPE_IQ3_XXS>(pCtx, pArgs, pStream);
    break;
  case GGML_TYPE_IQ3_S:
    mul_mat_q_case<GGML_TYPE_IQ3_S>(pCtx, pArgs, pStream);
    break;
  case GGML_TYPE_IQ4_XS:
    mul_mat_q_case<GGML_TYPE_IQ4_XS>(pCtx, pArgs, pStream);
    break;
  case GGML_TYPE_IQ4_NL:
    mul_mat_q_case<GGML_TYPE_IQ4_NL>(pCtx, pArgs, pStream);
    break;
    // -----------------------------------------------------------------------
  case GGML_TYPE_MXFP4:
    mul_mat_q_case<GGML_TYPE_MXFP4>(pCtx, pArgs, pStream);
    break;
  case GGML_TYPE_NVFP4:
    mul_mat_q_case<GGML_TYPE_NVFP4>(pCtx, pArgs, pStream);
    break;
  default:
    GGML_ABORT("fatal error");
    break;
  }
}

void ggml_cuda_mul_mat_q(ggml_backend_cuda_context &pCtx, const ggml_tensor *src0, const ggml_tensor *src1,
                         const ggml_tensor *pIds, ggml_tensor *dst) {
  GGML_ASSERT(src1->type == GGML_TYPE_F32);
  GGML_ASSERT(dst->type == GGML_TYPE_F32);
  GGML_ASSERT(!pIds || pIds->type == GGML_TYPE_I32); // Optional, used for batched GGML_MUL_MAT_ID.

  GGML_TENSOR_BINARY_OP_LOCALS;

  cudaStream_t vStream = pCtx.stream();
  const int cCc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;

  const size_t cTsSrc0 = ggml_type_size(src0->type);
  const size_t cTsSrc1 = ggml_type_size(src1->type);
  const size_t cTsDst = ggml_type_size(dst->type);

  GGML_ASSERT(nb00 == cTsSrc0);
  GGML_ASSERT(nb10 == cTsSrc1);
  GGML_ASSERT(nb0 == cTsDst);
  GGML_ASSERT(!pIds || pIds->nb[0] == ggml_type_size(pIds->type));

  const char *vSrc0D = (const char *)src0->data;
  const float *vSrc1D = (const float *)src1->data;
  float *vDstD = (float *)dst->data;

  // If src0 is a temporary compute buffer, clear any potential padding.
  if (ggml_backend_buffer_get_usage(src0->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE) {
    const size_t cSizeData = ggml_nbytes(src0);
    const size_t cSizeAlloc = ggml_backend_buffer_get_alloc_size(src0->buffer, src0);
    if (cSizeAlloc > cSizeData) {
      GGML_ASSERT(ggml_is_contiguously_allocated(src0));
      GGML_ASSERT(!src0->view_src);
      CUDA_CHECK(cudaMemsetAsync((char *)src0->data + cSizeData, 0, cSizeAlloc - cSizeData, vStream));
    }
  }

  const int64_t cNe10Padded = GGML_PAD(ne10, MATRIX_ROW_PADDING);

  const int64_t cS01 = src0->nb[1] / cTsSrc0;
  const int64_t cS1 = dst->nb[1] / cTsDst;
  const int64_t cS02 = src0->nb[2] / cTsSrc0;
  const int64_t cS2 = dst->nb[2] / cTsDst;
  const int64_t cS03 = src0->nb[3] / cTsSrc0;
  const int64_t cS3 = dst->nb[3] / cTsDst;

  const bool cFallback = ne01 % 128 != 0;

  const bool cUseNativeFp4 =
      blackwell_mma_available(cCc) && (src0->type == GGML_TYPE_MXFP4 || src0->type == GGML_TYPE_NVFP4);
  const size_t cYBlockSize = cUseNativeFp4 ? sizeof(block_fp4_mmq) : sizeof(block_q8_1_mmq);
  const size_t cYValuesPerBlock = cUseNativeFp4 ? QK_FP4_MMQ : QK8_1_MMQ;

  if (!pIds) {
    const size_t cNbytesSrc1Q81 = ne13 * ne12 * ne11 * cNe10Padded * cYBlockSize / cYValuesPerBlock +
                                  ggml_cuda_mmq_get_J_max(src0->type, cFallback, cCc, ne11) * sizeof(block_q8_1_mmq);
    ggml_cuda_pool_alloc<char> vSrc1Q81(pCtx.pool(), cNbytesSrc1Q81);
    ggml_cuda_pool_alloc<float> vSrc1Scale(pCtx.pool());
    if (src0->type == GGML_TYPE_NVFP4 && cUseNativeFp4) {
      vSrc1Scale.alloc(ne13 * ne12 * ne11);
    }

    {
      const int64_t cS11 = src1->nb[1] / cTsSrc1;
      const int64_t cS12 = src1->nb[2] / cTsSrc1;
      const int64_t cS13 = src1->nb[3] / cTsSrc1;
      if (cUseNativeFp4) {
        static constexpr size_t cAlignFloat8 = 32;
        const bool cUseAlignedFloat8 = ggml_cuda_is_aligned(src1, cAlignFloat8);
        static_assert(sizeof(block_fp4_mmq) == 4 * sizeof(block_q8_1));
        quantize_mmq_fp4_cuda(vSrc1D, nullptr, vSrc1Q81.get(), vSrc1Scale.ptr, src0->type, cUseAlignedFloat8, ne10,
                              cS11, cS12, cS13, cNe10Padded, ne11, ne12, ne13, vStream);

      } else {
        quantize_mmq_q8_1_cuda(vSrc1D, nullptr, vSrc1Q81.get(), src0->type, ne10, cS11, cS12, cS13, cNe10Padded, ne11,
                               ne12, ne13, vStream);
      }
      CUDA_CHECK(cudaGetLastError());
    }

    // Stride depends on quantization format
    const int64_t cS12 = cUseNativeFp4 ? ne11 * cNe10Padded * sizeof(block_fp4_mmq) / (QK_FP4_MMQ * sizeof(int))
                                       : ne11 * cNe10Padded * sizeof(block_q8_1) / (QK8_1 * sizeof(int));
    const int64_t cS13 = ne12 * cS12;

    const mmq_args cArgs = {vSrc0D,
                            src0->type,
                            (const int *)vSrc1Q81.ptr,
                            nullptr,
                            nullptr,
                            vDstD,
                            src0->type == GGML_TYPE_NVFP4 && cUseNativeFp4 ? vSrc1Scale.ptr : nullptr,
                            ne00,
                            ne01,
                            ne1,
                            cS01,
                            ne11,
                            cS1,
                            ne02,
                            ne12,
                            cS02,
                            cS12,
                            cS2,
                            ne03,
                            ne13,
                            cS03,
                            cS13,
                            cS3,
                            ne1,
                            ne1};
    fGgmlCudaMulMatQSwitchType(pCtx, cArgs, vStream);
    return;
  }

  GGML_ASSERT(ne13 == 1);
  GGML_ASSERT(nb12 % nb11 == 0);
  GGML_ASSERT(nb2 % nb1 == 0);

  const int64_t cNExpertUsed = pIds->ne[0];
  const int64_t cNeGetRows = ne12 * cNExpertUsed;
  GGML_ASSERT(ne1 == cNExpertUsed);

  ggml_cuda_pool_alloc<int32_t> vIdsSrc1(pCtx.pool(), cNeGetRows);
  ggml_cuda_pool_alloc<int32_t> vIdsDst(pCtx.pool(), cNeGetRows);
  ggml_cuda_pool_alloc<int32_t> vExpertBounds(pCtx.pool(), ne02 + 1);

  // gate/up activations are broadcast across experts (ne11 == 1): quantize each token once and
  // scatter to its slots. ids_src1 then holds the inverse map (token slot -> compact row).
  const bool cDedupBcast = ne11 == 1 && cNExpertUsed > 1;

  {
    GGML_ASSERT(pIds->nb[0] == ggml_element_size(pIds));
    const int cSi1 = pIds->nb[1] / ggml_element_size(pIds);
    const int cSis1 = nb12 / nb11;

    ggml_cuda_launch_mm_ids_helper((const int32_t *)pIds->data, vIdsSrc1.get(), vIdsDst.get(), vExpertBounds.get(),
                                   ne02, ne12, cNExpertUsed, ne11, cSi1, cSis1, /*write_inverse =*/cDedupBcast,
                                   vStream);
    CUDA_CHECK(cudaGetLastError());
  }

  const size_t cNbytesSrc1Q81 = ne12 * cNExpertUsed * cNe10Padded * cYBlockSize / cYValuesPerBlock +
                                ggml_cuda_mmq_get_J_max(src0->type, cFallback, cCc, ne11) * sizeof(block_q8_1_mmq);
  ggml_cuda_pool_alloc<char> vSrc1Q81(pCtx.pool(), cNbytesSrc1Q81);
  ggml_cuda_pool_alloc<float> vSrc1Scale(pCtx.pool());
  if (src0->type == GGML_TYPE_NVFP4 && cUseNativeFp4) {
    vSrc1Scale.alloc(ne12 * cNExpertUsed);
  }

  const int64_t cNe11Flat = ne12 * cNExpertUsed;
  const int64_t cNe12Flat = 1;
  const int64_t cNe13Flat = 1;

  {
    const int64_t cS11 = src1->nb[1] / cTsSrc1;
    const int64_t cS12 = src1->nb[2] / cTsSrc1;
    const int64_t cS13 = src1->nb[3] / cTsSrc1;

    if (cUseNativeFp4) {
      static constexpr size_t cAlignFloat8 = 32;
      const bool cUseAlignedFloat8 = ggml_cuda_is_aligned(src1, cAlignFloat8);
      if (cDedupBcast) {
        quantize_scatter_mmq_fp4_cuda(vSrc1D, vIdsSrc1.get(), vSrc1Q81.get(), vSrc1Scale.ptr, src0->type,
                                      cUseAlignedFloat8, ne10,
                                      /*stride_token=*/cS12, cNe10Padded, ne12, cNe11Flat, cNExpertUsed, vStream);
      } else {
        quantize_mmq_fp4_cuda(vSrc1D, vIdsSrc1.get(), vSrc1Q81.get(), vSrc1Scale.ptr, src0->type, cUseAlignedFloat8,
                              ne10, cS11, cS12, cS13, cNe10Padded, cNe11Flat, cNe12Flat, cNe13Flat, vStream);
      }
    } else if (cDedupBcast) {
      quantize_scatter_mmq_q8_1_cuda(vSrc1D, vIdsSrc1.get(), vSrc1Q81.get(), src0->type, ne10,
                                     /*stride_token=*/cS12, cNe10Padded, ne12, cNe11Flat, cNExpertUsed, vStream);
    } else {
      quantize_mmq_q8_1_cuda(vSrc1D, vIdsSrc1.get(), vSrc1Q81.get(), src0->type, ne10, cS11, cS12, cS13, cNe10Padded,
                             cNe11Flat, cNe12Flat, cNe13Flat, vStream);
    }
    CUDA_CHECK(cudaGetLastError());
  }

  static_assert(QK_FP4_MMQ == 8 * QK_MXFP4, "QK_FP4_MMQ needs to be 8 * QK_MXFP4");
  const int64_t cS12 = cUseNativeFp4 ? ne11 * cNe10Padded * sizeof(block_fp4_mmq) / (QK_FP4_MMQ * sizeof(int))
                                     : ne11 * cNe10Padded * sizeof(block_q8_1) / (QK8_1 * sizeof(int));
  const int64_t cS13 = ne12 * cS12;

  // Each expert only sees ne12*n_expert_used/ne02 tokens on average.
  // On RDNA3 and RDNA4 it is faster to pick the tile size against this value instead of ne12.
  int64_t vNcolsOpt = ne12;
  if (GGML_CUDA_CC_IS_RDNA3(cCc) || GGML_CUDA_CC_IS_RDNA4(cCc)) {
    vNcolsOpt = (ne12 * cNExpertUsed + ne02 - 1) / ne02;
  }

  // Note that ne02 is used instead of ne12 because the number of y channels determines the z dimension of the CUDA
  // grid.
  const mmq_args cArgs = {vSrc0D,
                          src0->type,
                          (const int *)vSrc1Q81.get(),
                          vIdsDst.get(),
                          vExpertBounds.get(),
                          vDstD,
                          vSrc1Scale.ptr,
                          ne00,
                          ne01,
                          cNeGetRows,
                          cS01,
                          cNeGetRows,
                          cS1,
                          ne02,
                          ne02,
                          cS02,
                          cS12,
                          cS2,
                          ne03,
                          ne13,
                          cS03,
                          cS13,
                          cS3,
                          ne12,
                          vNcolsOpt};

  fGgmlCudaMulMatQSwitchType(pCtx, cArgs, vStream);
}

bool ggml_cuda_should_use_mmq(enum ggml_type pType, int pCc, int64_t pNe11, int64_t pNExperts) {
#ifdef GGML_CUDA_FORCE_CUBLAS
  return false;
#endif // GGML_CUDA_FORCE_CUBLAS

  bool vMmqSupported;

  switch (pType) {
  case GGML_TYPE_Q1_0:
  case GGML_TYPE_Q2_0:
  case GGML_TYPE_Q4_0:
  case GGML_TYPE_Q4_1:
  case GGML_TYPE_Q5_0:
  case GGML_TYPE_Q5_1:
  case GGML_TYPE_Q8_0:
    // -------------------------------------------------
  case GGML_TYPE_Q2_K:
  case GGML_TYPE_Q3_K:
  case GGML_TYPE_Q4_K:
  case GGML_TYPE_Q5_K:
  case GGML_TYPE_Q6_K:
    // -------------------------------------------------
  case GGML_TYPE_IQ1_S:
  case GGML_TYPE_IQ2_XXS:
  case GGML_TYPE_IQ2_XS:
  case GGML_TYPE_IQ2_S:
  case GGML_TYPE_IQ3_XXS:
  case GGML_TYPE_IQ3_S:
  case GGML_TYPE_IQ4_XS:
  case GGML_TYPE_IQ4_NL:
    // -------------------------------------------------
  case GGML_TYPE_MXFP4:
  case GGML_TYPE_NVFP4:
    vMmqSupported = true;
    break;
  default:
    vMmqSupported = false;
    break;
  }

  if (!vMmqSupported) {
    return false;
  }

  // MMQ tiles require at least 48 KiB per-block shared memory; fall back to BLAS otherwise.
  {
    const int cId = ggml_cuda_get_device();
    const size_t cSmpbo = ggml_cuda_info().devices[cId].smpbo;
    if (cSmpbo < 48 * 1024) {
      return false;
    }
  }

  if (turing_mma_available(pCc)) {
    return true;
  }

  if (ggml_cuda_highest_compiled_arch(pCc) < GGML_CUDA_CC_DP4A) {
    // for MoE, mmq is faster even without native dp4a
    // TODO: check if cards older than pascal might benefit from this as well
    return pCc >= GGML_CUDA_CC_PASCAL && pNExperts > 0;
  }

#ifdef GGML_CUDA_FORCE_MMQ
  return true;
#endif // GGML_CUDA_FORCE_MMQ

  if (GGML_CUDA_CC_IS_NVIDIA(pCc)) {
    return !fp16_mma_hardware_available(pCc) || pNe11 < MMQ_DP4A_MAX_BATCH_SIZE;
  }

  if (amd_mfma_available(pCc)) {
    // As of ROCM 7.0 rocblas/tensile performs very poorly on CDNA3 and hipblaslt (via ROCBLAS_USE_HIPBLASLT)
    // performs better but is currently suffering from a crash on this architecture.
    // TODO: Revisit when hipblaslt is fixed on CDNA3
    if (GGML_CUDA_CC_IS_CDNA3(pCc)) {
      return true;
    }
    if (pNExperts > 64 || pNe11 <= 128) {
      return true;
    }
    if (pType == GGML_TYPE_Q4_0 || pType == GGML_TYPE_Q4_1 || pType == GGML_TYPE_Q5_0 || pType == GGML_TYPE_Q5_1) {
      return true;
    }
    if (pNe11 <= 256 && (pType == GGML_TYPE_Q4_K || pType == GGML_TYPE_Q5_K)) {
      return true;
    }
    return false;
  }

  if (amd_wmma_available(pCc)) {
    if (GGML_CUDA_CC_IS_RDNA3(pCc)) {
      // High expert counts are almost always better on MMQ due to
      //     the synchronization overhead in the cuBLAS/hipBLAS path.
      if (pNExperts >= 64) {
        return true;
      }

      // For some quantization types MMQ can have lower peak TOPS than hipBLAS
      //     so it's only faster for sufficiently small batch sizes:
      switch (pType) {
      case GGML_TYPE_Q2_K:
        return pNe11 <= 128;
      case GGML_TYPE_Q6_K:
        return pNe11 <= (GGML_CUDA_CC_IS_RDNA3_0(pCc) ? 128 : 256);
      case GGML_TYPE_IQ2_XS:
      case GGML_TYPE_IQ2_S:
        return GGML_CUDA_CC_IS_RDNA3_5(pCc) || pNe11 <= 128;
      default:
        return true;
      }
    }

    // For RDNA4 MMQ is consistently faster than dequantization + hipBLAS.
    return true;
  }

  // gfx900 (Vega 10) lacks native dp4a, loses to dequant + hipBLAS
  // for dense matrices; keep MMQ only for MoE, where the
  // hipBLAS path is much slower.
  if (pCc == GGML_CUDA_CC_VEGA) {
    return pNExperts > 0;
  }

  return (!GGML_CUDA_CC_IS_CDNA(pCc)) || pNe11 < MMQ_DP4A_MAX_BATCH_SIZE;
}
