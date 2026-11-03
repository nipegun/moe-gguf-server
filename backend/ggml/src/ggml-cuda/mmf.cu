#include "ggml.h"
#include "mmf.cuh"
#include "mmid.cuh"

static __forceinline__ int fMmfGetRowsPerBlock(const int pCc) {
  if (GGML_CUDA_CC_IS_CDNA(pCc)) {
    return MMF_ROWS_PER_BLOCK_CDNA;
  } else {
    return MMF_ROWS_PER_BLOCK;
  }
}

void ggml_cuda_mul_mat_f(ggml_backend_cuda_context &pCtx, const ggml_tensor *src0, const ggml_tensor *src1,
                         const ggml_tensor *pIds, ggml_tensor *dst) {
  GGML_ASSERT(src1->type == GGML_TYPE_F32);
  GGML_ASSERT(!pIds || pIds->type == GGML_TYPE_I32);
  GGML_ASSERT(dst->type == GGML_TYPE_F32);

  GGML_TENSOR_BINARY_OP_LOCALS;

  const size_t cTsSrc0 = ggml_type_size(src0->type);
  const size_t cTsSrc1 = ggml_type_size(src1->type);
  const size_t cTsDst = ggml_type_size(dst->type);

  GGML_ASSERT(ne13 == ne3);

  GGML_ASSERT(nb00 == cTsSrc0);
  GGML_ASSERT(nb10 == cTsSrc1);
  GGML_ASSERT(!pIds || pIds->nb[0] == ggml_type_size(pIds->type));
  GGML_ASSERT(nb0 == cTsDst);

  const float *vSrc1D = (const float *)src1->data;
  const int32_t *vIdsD = pIds ? (const int32_t *)pIds->data : nullptr;
  float *vDstD = (float *)dst->data;

  const int64_t cS01 = src0->nb[1] / cTsSrc0;
  const int64_t cS11 = src1->nb[1] / cTsSrc1;
  const int64_t cS1 = dst->nb[1] / cTsDst;
  const int64_t cS02 = src0->nb[2] / cTsSrc0;
  const int64_t cS12 = src1->nb[2] / cTsSrc1;
  const int64_t cS2 = dst->nb[2] / cTsDst;
  const int64_t cS03 = src0->nb[3] / cTsSrc0;
  const int64_t cS13 = src1->nb[3] / cTsSrc1;
  const int64_t cS3 = dst->nb[3] / cTsDst;

  const int64_t cIdsS0 = pIds ? pIds->nb[0] / ggml_type_size(pIds->type) : 0;
  const int64_t cIdsS1 = pIds ? pIds->nb[1] / ggml_type_size(pIds->type) : 0;

  mmf_ids_data vIdsInfo{};
  mmf_ids_data *vIdsInfoPtr = nullptr;
  ggml_cuda_pool_alloc<int32_t> vIdsSrcCompactDev;
  ggml_cuda_pool_alloc<int32_t> vIdsDstCompactDev;
  ggml_cuda_pool_alloc<int32_t> vExpertBoundsDev;

  // For MUL_MAT_ID the memory layout is different than for MUL_MAT:
  const int64_t cNcolsDst = pIds ? ne2 : ne1;
  const int64_t cNchannelsDst = pIds ? ne1 : ne2;

  const int64_t cStrideColDst = pIds ? cS2 : cS1;
  const int64_t cStrideColY = pIds ? cS12 : cS11;
  const int64_t cStrideChannelDst = pIds ? cS1 : cS2;

  int64_t vStrideChannelY = pIds ? cS11 : cS12;
  int64_t vNchannelsY = pIds ? ne11 : ne12;

  // mul_mat_id: handle broadcast
  if (pIds && vNchannelsY == 1) {
    vStrideChannelY = 0;
    vNchannelsY = pIds->ne[0];
  }

  if (pIds && cNcolsDst > 16) {
    const int64_t cNExpertUsed = pIds->ne[0];
    const int64_t cNExperts = ne02;
    const int64_t cNTokens = ne12;
    const int64_t cNeGetRows = cNTokens * cNExpertUsed;

    vIdsSrcCompactDev.alloc(pCtx.pool(), cNeGetRows);
    vIdsDstCompactDev.alloc(pCtx.pool(), cNeGetRows);
    vExpertBoundsDev.alloc(pCtx.pool(), cNExperts + 1);

    const int cSi1 = static_cast<int>(cIdsS1);
    const int cSis1 = static_cast<int>(src1->nb[2] / src1->nb[1]);

    GGML_ASSERT(cSis1 > 0);

    ggml_cuda_launch_mm_ids_helper(vIdsD, vIdsSrcCompactDev.get(), vIdsDstCompactDev.get(), vExpertBoundsDev.get(),
                                   static_cast<int>(cNExperts), static_cast<int>(cNTokens),
                                   static_cast<int>(cNExpertUsed), static_cast<int>(ne11), cSi1, cSis1,
                                   /*write_inverse =*/false, pCtx.stream());
    CUDA_CHECK(cudaGetLastError());

    vIdsInfo.ids_src_compact = vIdsSrcCompactDev.get();
    vIdsInfo.ids_dst_compact = vIdsDstCompactDev.get();
    vIdsInfo.expert_bounds_dev = vExpertBoundsDev.get();
    vIdsInfo.n_experts = static_cast<int>(cNExperts);
    vIdsInfo.sis1 = cSis1;
    vIdsInfoPtr = &vIdsInfo;
  }

  const int cDevice = ggml_cuda_get_device();
  const int cCc = ggml_cuda_info().devices[cDevice].cc;
  const int cRowsPerBlock = fMmfGetRowsPerBlock(cCc);

  switch (src0->type) {
  case GGML_TYPE_F32: {
    const float *vSrc0D = (const float *)src0->data;
    constexpr int cValsPerT = 1;
    mul_mat_f_switch_rows_per_block<float>(
        cRowsPerBlock, vSrc0D, vSrc1D, vIdsD, vDstD, ne00 / cValsPerT, ne01, cNcolsDst, cS01 / cValsPerT,
        cStrideColY / cValsPerT, cStrideColDst, cIdsS0, cIdsS1, ne02, vNchannelsY, cNchannelsDst, cS02 / cValsPerT,
        vStrideChannelY, cStrideChannelDst, ne03, ne3, cS03 / cValsPerT, cS13, cS3, pCtx.stream(), vIdsInfoPtr);
  } break;
  case GGML_TYPE_F16: {
    const half2 *vSrc0D = (const half2 *)src0->data;
    constexpr int cValsPerT = 2;
    mul_mat_f_switch_rows_per_block<half2>(
        cRowsPerBlock, vSrc0D, vSrc1D, vIdsD, vDstD, ne00 / cValsPerT, ne01, cNcolsDst, cS01 / cValsPerT,
        cStrideColY / cValsPerT, cStrideColDst, cIdsS0, cIdsS1, ne02, vNchannelsY, cNchannelsDst, cS02 / cValsPerT,
        vStrideChannelY, cStrideChannelDst, ne03, ne3, cS03 / cValsPerT, cS13, cS3, pCtx.stream(), vIdsInfoPtr);
  } break;
  case GGML_TYPE_BF16: {
    const nv_bfloat162 *vSrc0D = (const nv_bfloat162 *)src0->data;
    constexpr int cValsPerT = 2;
    mul_mat_f_switch_rows_per_block<nv_bfloat162>(
        cRowsPerBlock, vSrc0D, vSrc1D, vIdsD, vDstD, ne00 / cValsPerT, ne01, cNcolsDst, cS01 / cValsPerT,
        cStrideColY / cValsPerT, cStrideColDst, cIdsS0, cIdsS1, ne02, vNchannelsY, cNchannelsDst, cS02 / cValsPerT,
        vStrideChannelY, cStrideChannelDst, ne03, ne3, cS03 / cValsPerT, cS13, cS3, pCtx.stream(), vIdsInfoPtr);
  } break;
  default:
    GGML_ABORT("unsupported type: %s", ggml_type_name(src0->type));
  }
}

bool ggml_cuda_should_use_mmf(enum ggml_type pType, int pCc, int pWarpSize, const int64_t *pSrc0Ne,
                              const size_t *pSrc0Nb, const int pSrc1Ncols, bool pMulMatId) {
  if (ggml_is_quantized(pType)) {
    return false;
  }

  const size_t cTs = ggml_type_size(pType);
  if (pSrc0Ne[0] % (pWarpSize * (4 / cTs)) != 0) {
    return false;
  }

  if (pSrc0Nb[0] != cTs) {
    return false;
  }

  // Pointers not aligned to the size of half2/nv_bfloat162/float2 would result in a crash:
  for (size_t vI = 1; vI < GGML_MAX_DIMS; ++vI) {
    if (pSrc0Nb[vI] % (2 * cTs) != 0) {
      return false;
    }
  }
  if (pSrc0Ne[1] % fMmfGetRowsPerBlock(pCc) != 0) {
    return false;
  }

  if (GGML_CUDA_CC_IS_CDNA3(pCc) && pType == GGML_TYPE_BF16) {
    return false;
  }

  if (pMulMatId) {
    if (pSrc0Ne[1] <= 1024 && pSrc1Ncols > 512) {
      return false;
    } else if (pSrc0Ne[1] > 1024 && pSrc1Ncols > 128) {
      return false;
    }
  } else {
    if (GGML_CUDA_CC_IS_RDNA3_0(pCc) && pSrc1Ncols > 8) {
      return false;
    } else if (GGML_CUDA_CC_IS_CDNA2(pCc) && (pType == GGML_TYPE_F16 || pType == GGML_TYPE_BF16)) {
      // TODO: truse CDNA2 as CDNA1, tune the perf when CDNA2 is available.
      return false;
    } else if (GGML_CUDA_CC_IS_CDNA1(pCc) && (pType == GGML_TYPE_F16 || pType == GGML_TYPE_BF16)) {
      return false;
    } else if (pSrc1Ncols > 16) {
      return false;
    }
  }

  switch (pType) {
  case GGML_TYPE_F32:
    return ampere_mma_available(pCc) || amd_mfma_available(pCc);
  case GGML_TYPE_F16:
    return volta_mma_available(pCc) || turing_mma_available(pCc) || amd_wmma_available(pCc) || amd_mfma_available(pCc);
  case GGML_TYPE_BF16:
    return ampere_mma_available(pCc) || amd_wmma_available(pCc) || amd_mfma_available(pCc);
  default:
    return false;
  }
}
