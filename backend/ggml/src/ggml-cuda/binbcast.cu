#include "binbcast.cuh"
#include <cstdint>
#include <utility>

template <typename T, size_t> using type_for_index = T;

static __device__ __forceinline__ float op_repeat(const float pA, const float pB) {
  return pB;
  GGML_UNUSED(pA);
}

static __device__ __forceinline__ float op_add(const float pA, const float pB) { return pA + pB; }

static __device__ __forceinline__ float op_sub(const float pA, const float pB) { return pA - pB; }

static __device__ __forceinline__ float op_mul(const float pA, const float pB) { return pA * pB; }

static __device__ __forceinline__ float op_div(const float pA, const float pB) { return pA / pB; }

template <float (*bin_op)(const float, const float), typename src0_t, typename src1_t, typename dst_t,
          typename... src1_ptrs>
static __global__ void fKBinBcast(const src0_t *pSrc0, const src1_t *pSrc1, dst_t *pDst, const uint32_t pNe0,
                                  const uint32_t pNe1, const uint32_t pNe2, const uint3 pNe3, const uint3 pNe10,
                                  const uint3 pNe11, const uint3 pNe12, const uint3 pNe13,
                                  /*const uint32_t         s0,*/
                                  const uint32_t pS1, const uint32_t pS2, const uint32_t pS3, const uint32_t pS00,
                                  const uint32_t pS01, const uint32_t pS02, const uint32_t pS03, const uint32_t pS10,
                                  const uint32_t pS11, const uint32_t pS12, const uint32_t pS13, src1_ptrs... pSrc1s) {
  ggml_cuda_pdl_lc();
  const uint32_t cI0s = blockDim.x * blockIdx.x + threadIdx.x;
  const uint32_t cI1 = (blockDim.y * blockIdx.y + threadIdx.y);
  const uint32_t cI2 = fastdiv((blockDim.z * blockIdx.z + threadIdx.z), pNe3);
  const uint32_t cI3 = (blockDim.z * blockIdx.z + threadIdx.z) - (cI2 * pNe3.z);

  if (cI0s >= pNe0 || cI1 >= pNe1 || cI2 >= pNe2 || cI3 >= pNe3.z) {
    return;
  }

  const uint32_t cI11 = fastmodulo(cI1, pNe11);
  const uint32_t cI12 = fastmodulo(cI2, pNe12);
  const uint32_t cI13 = fastmodulo(cI3, pNe13);

  const size_t cISrc0 = size_t(cI3) * pS03 + size_t(cI2) * pS02 + size_t(cI1) * pS01;
  const size_t cISrc1 = size_t(cI13) * pS13 + size_t(cI12) * pS12 + size_t(cI11) * pS11;
  const size_t cIDst = size_t(cI3) * pS3 + size_t(cI2) * pS2 + size_t(cI1) * pS1;

  const src0_t *vSrc0Row = pSrc0 ? (pSrc0 + cISrc0) : nullptr;
  dst_t *vDstRow = pDst + cIDst;

  const uint32_t cS0 = blockDim.x * gridDim.x;

  ggml_cuda_pdl_sync();
  for (uint32_t vI0 = cI0s; vI0 < pNe0; vI0 += cS0) {
    const uint32_t cI10 = fastmodulo(vI0, pNe10);

    float vResult = vSrc0Row ? (float)vSrc0Row[size_t(vI0) * pS00] : 0.0f;
    if constexpr (sizeof...(src1_ptrs) > 0) {
      vResult = (..., (vResult = bin_op(vResult, (float)pSrc1s[cISrc1 + size_t(cI10) * pS10])));
    } else {
      vResult = bin_op(vResult, (float)pSrc1[cISrc1 + size_t(cI10) * pS10]);
    }

    vDstRow[vI0] = (dst_t)vResult;

    // protect i0 from overflow
    if (pNe0 - vI0 <= cS0) {
      break;
    }
  }
}

template <float (*bin_op)(const float, const float), typename src0_t, typename src1_t, typename dst_t,
          typename... src1_ptrs>
static __global__ void
fKBinBcastUnravel(const src0_t *pSrc0, const src1_t *pSrc1, dst_t *pDst, const uint3 pNe0, const uint3 pNe1,
                  const uint3 pNe2, const uint32_t pNe3, const uint3 pProd012, const uint3 pProd01, const uint3 pNe10,
                  const uint3 pNe11, const uint3 pNe12, const uint3 pNe13,
                  /*const int              s0,*/
                  const uint32_t pS1, const uint32_t pS2, const uint32_t pS3, const uint32_t pS00, const uint32_t pS01,
                  const uint32_t pS02, const uint32_t pS03, const uint32_t pS10, const uint32_t pS11,
                  const uint32_t pS12, const uint32_t pS13, src1_ptrs... pSrc1s) {
  const uint32_t cI = blockDim.x * blockIdx.x + threadIdx.x;

  const uint32_t cI3 = fastdiv(cI, pProd012);
  const uint32_t cI2 = fastdiv(cI - cI3 * pProd012.z, pProd01);
  const uint32_t cI1 = fastdiv(cI - cI3 * pProd012.z - cI2 * pProd01.z, pNe0);
  const uint32_t cI0 = cI - cI3 * pProd012.z - cI2 * pProd01.z - cI1 * pNe0.z;

  if (cI0 >= pNe0.z || cI1 >= pNe1.z || cI2 >= pNe2.z || cI3 >= pNe3) {
    return;
  }

  const uint32_t cI11 = fastmodulo(cI1, pNe11);
  const uint32_t cI12 = fastmodulo(cI2, pNe12);
  const uint32_t cI13 = fastmodulo(cI3, pNe13);

  const size_t cISrc0 = size_t(cI3) * pS03 + size_t(cI2) * pS02 + size_t(cI1) * pS01;
  const size_t cISrc1 = size_t(cI13) * pS13 + size_t(cI12) * pS12 + size_t(cI11) * pS11;
  const size_t cIDst = size_t(cI3) * pS3 + size_t(cI2) * pS2 + size_t(cI1) * pS1;

  const src0_t *vSrc0Row = pSrc0 ? (pSrc0 + cISrc0) : nullptr;
  dst_t *vDstRow = pDst + cIDst;

  const uint32_t cI10 = fastmodulo(cI0, pNe10);

  ggml_cuda_pdl_sync();
  float vResult = vSrc0Row ? (float)vSrc0Row[size_t(cI0) * pS00] : 0.0f;
  if constexpr (sizeof...(src1_ptrs) > 0) {
    vResult = (..., (vResult = bin_op(vResult, (float)pSrc1s[cISrc1 + size_t(cI10) * pS10])));
  } else {
    vResult = bin_op(vResult, (float)pSrc1[cISrc1 + size_t(cI10) * pS10]);
  }

  vDstRow[cI0] = (dst_t)vResult;
}

template <float (*bin_op)(const float, const float), typename src0_t, typename src1_t, typename dst_t, size_t... I>
static void fLaunchBinBcastPack(const ggml_tensor *src0, const ggml_tensor *src1, ggml_tensor *dst,
                                const src0_t *pSrc0Dd, const src1_t *pSrc1Dd, dst_t *pDstDd, cudaStream_t pStream,
                                std::index_sequence<I...>) {
  GGML_TENSOR_BINARY_OP_LOCALS

  int vNr0 = ne10 / ne0;
  int vNr1 = ne11 / ne1;
  int vNr2 = ne12 / ne2;
  int vNr3 = ne13 / ne3;

  int aNr[4] = {vNr0, vNr1, vNr2, vNr3};

  int64_t aCne[] = {ne0, ne1, ne2, ne3};
  int64_t aCne0[] = {ne00, ne01, ne02, ne03};
  int64_t vCne1[] = {ne10, ne11, ne12, ne13};

  size_t aCnb[] = {nb0, nb1, nb2, nb3};
  size_t aCnb0[] = {nb00, nb01, nb02, nb03};
  size_t aCnb1[] = {nb10, nb11, nb12, nb13};

  auto vCollapse = [](int64_t pCne[]) {
    pCne[0] *= pCne[1];
    pCne[1] = pCne[2];
    pCne[2] = pCne[3];
    pCne[3] = 1;
  };

  auto vCollapseNb = [](size_t pCnb[], const int64_t pCne[]) {
    pCnb[1] *= pCne[1];
    pCnb[2] *= pCne[2];
    pCnb[3] *= pCne[3];
  };

  if (ggml_is_contiguous(src0) && ggml_is_contiguous(src1) && !ggml_is_permuted(src0) && !ggml_is_permuted(src1)) {
    for (int vI = 0; vI < 4; vI++) {
      if (aNr[vI] != 1) {
        break;
      }
      if (vI > 0) {
        vCollapseNb(aCnb, aCne);
        vCollapseNb(aCnb0, aCne0);
        vCollapseNb(aCnb1, vCne1);
        vCollapse(aCne);
        vCollapse(aCne0);
        vCollapse(vCne1);
      }
    }
  }

  {
    int64_t vNe0 = aCne[0];
    int64_t vNe1 = aCne[1];
    int64_t vNe2 = aCne[2];
    int64_t vNe3 = aCne[3];

    // int64_t ne00 = cne0[0]; GGML_UNUSED(ne00);
    // int64_t ne01 = cne0[1]; GGML_UNUSED(ne01);
    // int64_t ne02 = cne0[2]; GGML_UNUSED(ne02);
    // int64_t ne03 = cne0[3]; GGML_UNUSED(ne03);

    size_t vNb0 = aCnb[0];
    size_t vNb1 = aCnb[1];
    size_t vNb2 = aCnb[2];
    size_t vNb3 = aCnb[3];

    size_t vNb00 = aCnb0[0];
    size_t vNb01 = aCnb0[1];
    size_t vNb02 = aCnb0[2];
    size_t vNb03 = aCnb0[3];

    size_t vNb10 = aCnb1[0];
    size_t vNb11 = aCnb1[1];
    size_t vNb12 = aCnb1[2];
    size_t vNb13 = aCnb1[3];

    // size_t s0 = nb0 / sizeof(dst_t);
    size_t vS1 = vNb1 / sizeof(dst_t);
    size_t vS2 = vNb2 / sizeof(dst_t);
    size_t vS3 = vNb3 / sizeof(dst_t);

    size_t vS10 = vNb10 / sizeof(src1_t);
    size_t vS11 = vNb11 / sizeof(src1_t);
    size_t vS12 = vNb12 / sizeof(src1_t);
    size_t vS13 = vNb13 / sizeof(src1_t);

    size_t vS00 = vNb00 / sizeof(src0_t);
    size_t vS01 = vNb01 / sizeof(src0_t);
    size_t vS02 = vNb02 / sizeof(src0_t);
    size_t vS03 = vNb03 / sizeof(src0_t);

    GGML_ASSERT(vNe0 <= std::numeric_limits<uint32_t>::max());
    GGML_ASSERT(vNe1 <= std::numeric_limits<uint32_t>::max());
    GGML_ASSERT(vNe2 <= std::numeric_limits<uint32_t>::max());
    GGML_ASSERT(vNe3 <= std::numeric_limits<uint32_t>::max());

    // GGML_ASSERT(s0  <= std::numeric_limits<uint32_t>::max());
    GGML_ASSERT(vS1 <= std::numeric_limits<uint32_t>::max());
    GGML_ASSERT(vS2 <= std::numeric_limits<uint32_t>::max());
    GGML_ASSERT(vS3 <= std::numeric_limits<uint32_t>::max());

    GGML_ASSERT(vS00 <= std::numeric_limits<uint32_t>::max());
    GGML_ASSERT(vS01 <= std::numeric_limits<uint32_t>::max());
    GGML_ASSERT(vS02 <= std::numeric_limits<uint32_t>::max());
    GGML_ASSERT(vS03 <= std::numeric_limits<uint32_t>::max());

    GGML_ASSERT(vS10 <= std::numeric_limits<uint32_t>::max());
    GGML_ASSERT(vS11 <= std::numeric_limits<uint32_t>::max());
    GGML_ASSERT(vS12 <= std::numeric_limits<uint32_t>::max());
    GGML_ASSERT(vS13 <= std::numeric_limits<uint32_t>::max());

    GGML_ASSERT(vCne1[0] <= std::numeric_limits<uint32_t>::max());
    GGML_ASSERT(vCne1[1] <= std::numeric_limits<uint32_t>::max());
    GGML_ASSERT(vCne1[2] <= std::numeric_limits<uint32_t>::max());
    GGML_ASSERT(vCne1[3] <= std::numeric_limits<uint32_t>::max());

    GGML_ASSERT(vNb0 % sizeof(dst_t) == 0);
    GGML_ASSERT(vNb1 % sizeof(dst_t) == 0);
    GGML_ASSERT(vNb2 % sizeof(dst_t) == 0);
    GGML_ASSERT(vNb3 % sizeof(dst_t) == 0);

    GGML_ASSERT(vNb00 % sizeof(src0_t) == 0);
    GGML_ASSERT(vNb01 % sizeof(src0_t) == 0);
    GGML_ASSERT(vNb02 % sizeof(src0_t) == 0);
    GGML_ASSERT(vNb03 % sizeof(src0_t) == 0);

    GGML_ASSERT(vNb10 % sizeof(src1_t) == 0);
    GGML_ASSERT(vNb11 % sizeof(src1_t) == 0);
    GGML_ASSERT(vNb12 % sizeof(src1_t) == 0);
    GGML_ASSERT(vNb13 % sizeof(src1_t) == 0);

    GGML_ASSERT(vNe2 * vNe3 <= std::numeric_limits<unsigned int>::max());

    const int cBlockSize = 128;

    int64_t vHne0 = std::max(vNe0 / 2LL, 1LL);

    dim3 vBlockDims;
    vBlockDims.x = std::min<unsigned int>(vHne0, cBlockSize);
    vBlockDims.y = std::min<unsigned int>(vNe1, cBlockSize / vBlockDims.x);
    vBlockDims.z = std::min(std::min<unsigned int>(vNe2 * vNe3, cBlockSize / vBlockDims.x / vBlockDims.y), 64U);

    dim3 vBlockNums((vHne0 + vBlockDims.x - 1) / vBlockDims.x, (vNe1 + vBlockDims.y - 1) / vBlockDims.y,
                    (vNe2 * vNe3 + vBlockDims.z - 1) / vBlockDims.z);

    const uint3 cNe10 = init_fastdiv_values((uint32_t)vCne1[0]);
    const uint3 cNe11 = init_fastdiv_values((uint32_t)vCne1[1]);
    const uint3 cNe12 = init_fastdiv_values((uint32_t)vCne1[2]);
    const uint3 cNe13 = init_fastdiv_values((uint32_t)vCne1[3]);

    if (vBlockNums.z > 65535 || vBlockNums.y > 65535) {
      int64_t vBlockNum = (vNe0 * vNe1 * vNe2 * vNe3 + cBlockSize - 1) / cBlockSize;

      GGML_ASSERT(vBlockNum <= std::numeric_limits<uint32_t>::max());
      GGML_ASSERT(vBlockNum * cBlockSize <= std::numeric_limits<uint32_t>::max());
      GGML_ASSERT(vNe0 * vNe1 <= std::numeric_limits<uint32_t>::max());
      GGML_ASSERT(vNe0 * vNe1 * vNe2 <= std::numeric_limits<uint32_t>::max());

      const uint3 cProd012 = init_fastdiv_values((uint32_t)(vNe0 * vNe1 * vNe2));
      const uint3 cProd01 = init_fastdiv_values((uint32_t)(vNe0 * vNe1));
      const uint3 cNe0Fastdiv = init_fastdiv_values((uint32_t)vNe0);
      const uint3 cNe1Fastdiv = init_fastdiv_values((uint32_t)vNe1);
      const uint3 cNe2Fastdiv = init_fastdiv_values((uint32_t)vNe2);

      {
        const ggml_cuda_kernel_launch_params cLaunchParams =
            ggml_cuda_kernel_launch_params((dim3)vBlockNum, cBlockSize, 0, pStream);
        ggml_cuda_kernel_launch(fKBinBcastUnravel<bin_op, src0_t, src1_t, dst_t, type_for_index<const src1_t *, I>...>,
                                cLaunchParams, pSrc0Dd, pSrc1Dd, pDstDd, cNe0Fastdiv, cNe1Fastdiv, cNe2Fastdiv, vNe3,
                                cProd012, cProd01, cNe10, cNe11, cNe12, cNe13,
                                /*s0,*/ vS1, vS2, vS3, vS00, vS01, vS02, vS03, vS10, vS11, vS12, vS13,
                                (const src1_t *)dst->src[I + 1]->data...);
      }
    } else {
      GGML_ASSERT(int64_t(vBlockNums.x) * vBlockDims.x <= std::numeric_limits<uint32_t>::max());
      GGML_ASSERT(int64_t(vBlockNums.y) * vBlockDims.y <= std::numeric_limits<uint32_t>::max());
      GGML_ASSERT(int64_t(vBlockNums.z) * vBlockDims.z <= std::numeric_limits<uint32_t>::max());

      const uint3 cNe3Fastdiv = init_fastdiv_values((uint32_t)vNe3);
      {
        const ggml_cuda_kernel_launch_params cLaunchParams =
            ggml_cuda_kernel_launch_params(vBlockNums, vBlockDims, 0, pStream);
        ggml_cuda_kernel_launch(fKBinBcast<bin_op, src0_t, src1_t, dst_t, type_for_index<const src1_t *, I>...>,
                                cLaunchParams, pSrc0Dd, pSrc1Dd, pDstDd, vNe0, vNe1, vNe2, cNe3Fastdiv, cNe10, cNe11,
                                cNe12, cNe13,
                                /*s0,*/ vS1, vS2, vS3, vS00, vS01, vS02, vS03, vS10, vS11, vS12, vS13,
                                (const src1_t *)dst->src[I + 1]->data...);
      }
    }
  }
}

template <typename T>
static __global__ void fKRepeatBack(const T *__restrict__ pSrc, T *__restrict__ pDst, const int64_t pNe00,
                                    const int64_t pNe01, const int64_t pNe02, const int64_t pNe03, const size_t pS00,
                                    const size_t pS01, const size_t pS02, const size_t pS03, const int64_t pNe0,
                                    const int64_t pNe1, const int64_t pNe2, const int64_t pNe3) {

  const int64_t cTid0 = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t cTid1 = int64_t(blockIdx.y) * blockDim.y + threadIdx.y;
  const int64_t cTid23 = int64_t(blockIdx.z) * blockDim.z + threadIdx.z;
  const int64_t cTid2 = cTid23 % pNe2;
  const int64_t cTid3 = cTid23 / pNe2;

  if (cTid0 >= pNe0) {
    return;
  }

  T vSum = 0;
  ggml_cuda_pdl_sync();
  for (int64_t vI3 = cTid3; vI3 < pNe03; vI3 += pNe3) {
    for (int64_t vI2 = cTid2; vI2 < pNe02; vI2 += pNe2) {
      for (int64_t vI1 = cTid1; vI1 < pNe01; vI1 += pNe1) {
        for (int64_t vI0 = cTid0; vI0 < pNe00; vI0 += pNe0) {
          vSum += pSrc[vI3 * pS03 + vI2 * pS02 + vI1 * pS01 + vI0 * pS00];
        }
      }
    }
  }
  pDst[cTid3 * pNe2 * pNe1 * pNe0 + cTid2 * pNe1 * pNe0 + cTid1 * pNe0 + cTid0] = vSum;
}

template <float (*bin_op)(const float, const float), int n_fuse = 1> struct bin_bcast_cuda {
  template <typename src0_t, typename src1_t, typename dst_t>
  void operator()(const struct ggml_tensor *pSrc0, const struct ggml_tensor *pSrc1, struct ggml_tensor *pDst,
                  const src0_t *pSrc0Dd, const src1_t *pSrc1Dd, dst_t *pDstDd, cudaStream_t pStream) {
    fLaunchBinBcastPack<bin_op, src0_t, src1_t, dst_t>(pSrc0, pSrc1, pDst, pSrc0Dd, pSrc1Dd, pDstDd, pStream,
                                                       std::make_index_sequence<n_fuse>{});
  }
};

template <typename T>
static void fRepeatBackCuda(const T *pSrc, T *pDst, const int64_t pNe00, const int64_t pNe01, const int64_t pNe02,
                            const int64_t pNe03, const size_t pS00, const size_t pS01, const size_t pS02,
                            const size_t pS03, const int64_t pNe0, const int64_t pNe1, const int64_t pNe2,
                            const int64_t pNe3, cudaStream_t pStream) {

  const dim3 cBlockDims(WARP_SIZE, 1, 1);
  const dim3 cBlockNums((pNe0 + WARP_SIZE - 1) / WARP_SIZE, pNe1, pNe2 * pNe3);
  fKRepeatBack<T><<<cBlockNums, cBlockDims, 0, pStream>>>(pSrc, pDst, pNe00, pNe01, pNe02, pNe03, pS00, pS01, pS02,
                                                          pS03, pNe0, pNe1, pNe2, pNe3);
}

template <class op>
static void fGgmlCudaOpBinBcast(const ggml_tensor *pSrc0, const ggml_tensor *pSrc1, ggml_tensor *pDst,
                                const void *pSrc0Dd, const void *pSrc1Dd, void *pDstDd, cudaStream_t pStream) {

  GGML_ASSERT(pSrc1->type == GGML_TYPE_F32 || pSrc1->type == GGML_TYPE_F16);

  if (pSrc0->type == GGML_TYPE_F32 && pDst->type == GGML_TYPE_F32) {
    op()(pSrc0, pSrc1, pDst, (const float *)pSrc0Dd, (const float *)pSrc1Dd, (float *)pDstDd, pStream);
  } else if (pSrc0->type == GGML_TYPE_F16 && pSrc1->type == GGML_TYPE_F16 && pDst->type == GGML_TYPE_F16) {
    op()(pSrc0, pSrc1, pDst, (const half *)pSrc0Dd, (const half *)pSrc1Dd, (half *)pDstDd, pStream);
  } else if (pSrc0->type == GGML_TYPE_F16 && pSrc1->type == GGML_TYPE_F32 && pDst->type == GGML_TYPE_F16) {
    op()(pSrc0, pSrc1, pDst, (const half *)pSrc0Dd, (const float *)pSrc1Dd, (half *)pDstDd, pStream);
  } else if (pSrc0->type == GGML_TYPE_F16 && pDst->type == GGML_TYPE_F32) {
    op()(pSrc0, pSrc1, pDst, (const half *)pSrc0Dd, (const float *)pSrc1Dd, (float *)pDstDd, pStream);
  } else {
    fprintf(stderr, "%s: unsupported types: dst: %s, src0: %s, src1: %s\n", __func__, ggml_type_name(pDst->type),
            ggml_type_name(pSrc0->type), ggml_type_name(pSrc1->type));
    GGML_ABORT("fatal error");
  }
}

void ggml_cuda_op_repeat(ggml_backend_cuda_context &pCtx, ggml_tensor *pDst) {
  fGgmlCudaOpBinBcast<bin_bcast_cuda<op_repeat, 0>>(pDst, pDst->src[0], pDst, nullptr, pDst->src[0]->data, pDst->data,
                                                    pCtx.stream());
}

void ggml_cuda_op_add(ggml_backend_cuda_context &pCtx, ggml_tensor *pDst) {
  fGgmlCudaOpBinBcast<bin_bcast_cuda<op_add>>(pDst->src[0], pDst->src[1], pDst, pDst->src[0]->data, pDst->src[1]->data,
                                              pDst->data, pCtx.stream());
}

void ggml_cuda_op_sub(ggml_backend_cuda_context &pCtx, ggml_tensor *pDst) {
  fGgmlCudaOpBinBcast<bin_bcast_cuda<op_sub>>(pDst->src[0], pDst->src[1], pDst, pDst->src[0]->data, pDst->src[1]->data,
                                              pDst->data, pCtx.stream());
}

void ggml_cuda_op_mul(ggml_backend_cuda_context &pCtx, ggml_tensor *pDst) {
  fGgmlCudaOpBinBcast<bin_bcast_cuda<op_mul>>(pDst->src[0], pDst->src[1], pDst, pDst->src[0]->data, pDst->src[1]->data,
                                              pDst->data, pCtx.stream());
}

void ggml_cuda_op_div(ggml_backend_cuda_context &pCtx, ggml_tensor *pDst) {
  fGgmlCudaOpBinBcast<bin_bcast_cuda<op_div>>(pDst->src[0], pDst->src[1], pDst, pDst->src[0]->data, pDst->src[1]->data,
                                              pDst->data, pCtx.stream());
}

template <float (*op)(const float, const float), int n_fuse>
static void fGgmlCudaOpFusedBinbcastImpl(ggml_backend_cuda_context &pCtx, ggml_tensor *pDst) {
  cudaStream_t vStream = pCtx.stream();

  const ggml_tensor *vSrc0 = pDst->src[0];
  const ggml_tensor *vSrc1 = pDst->src[1];

  if (vSrc0->type == GGML_TYPE_F32 && pDst->type == GGML_TYPE_F32) {
    fLaunchBinBcastPack<op, float, float, float>(vSrc0, vSrc1, pDst, (const float *)vSrc0->data,
                                                 (const float *)vSrc1->data, (float *)pDst->data, vStream,
                                                 std::make_index_sequence<n_fuse>{});
  } else if (vSrc0->type == GGML_TYPE_F16 && vSrc1->type == GGML_TYPE_F16 && pDst->type == GGML_TYPE_F16) {
    fLaunchBinBcastPack<op, half, half, half>(vSrc0, vSrc1, pDst, (const half *)vSrc0->data, (const half *)vSrc1->data,
                                              (half *)pDst->data, vStream, std::make_index_sequence<n_fuse>{});
  } else if (vSrc0->type == GGML_TYPE_F16 && vSrc1->type == GGML_TYPE_F32 && pDst->type == GGML_TYPE_F16) {
    fLaunchBinBcastPack<op, half, float, half>(vSrc0, vSrc1, pDst, (const half *)vSrc0->data,
                                               (const float *)vSrc1->data, (half *)pDst->data, vStream,
                                               std::make_index_sequence<n_fuse>{});
  } else if (vSrc0->type == GGML_TYPE_F16 && pDst->type == GGML_TYPE_F32) {
    fLaunchBinBcastPack<op, half, float, float>(vSrc0, vSrc1, pDst, (const half *)vSrc0->data,
                                                (const float *)vSrc1->data, (float *)pDst->data, vStream,
                                                std::make_index_sequence<n_fuse>{});
  } else {
    fprintf(stderr, "%s: unsupported types for fusion: dst: %s, src0: %s, src1: %s\n", __func__,
            ggml_type_name(pDst->type), ggml_type_name(vSrc0->type), ggml_type_name(vSrc1->type));
    GGML_ABORT("fatal error");
  }
}

void ggml_cuda_op_fused_add(ggml_backend_cuda_context &pCtx, ggml_tensor *pDst, int pNFuse) {
  GGML_ASSERT(2 <= pNFuse && pNFuse <= 8);

  switch (pNFuse) {
  case 2:
    fGgmlCudaOpFusedBinbcastImpl<op_add, 2>(pCtx, pDst);
    break;
  case 3:
    fGgmlCudaOpFusedBinbcastImpl<op_add, 3>(pCtx, pDst);
    break;
  case 4:
    fGgmlCudaOpFusedBinbcastImpl<op_add, 4>(pCtx, pDst);
    break;
  case 5:
    fGgmlCudaOpFusedBinbcastImpl<op_add, 5>(pCtx, pDst);
    break;
  case 6:
    fGgmlCudaOpFusedBinbcastImpl<op_add, 6>(pCtx, pDst);
    break;
  case 7:
    fGgmlCudaOpFusedBinbcastImpl<op_add, 7>(pCtx, pDst);
    break;
  case 8:
    fGgmlCudaOpFusedBinbcastImpl<op_add, 8>(pCtx, pDst);
    break;
  default:
    GGML_ASSERT(false && "Unsupported n_fuse value");
  }
}

void ggml_cuda_op_fused_mul(ggml_backend_cuda_context &pCtx, ggml_tensor *pDst, int pNFuse) {
  GGML_ASSERT(2 <= pNFuse && pNFuse <= 8);

  switch (pNFuse) {
  case 2:
    fGgmlCudaOpFusedBinbcastImpl<op_mul, 2>(pCtx, pDst);
    break;
  case 3:
    fGgmlCudaOpFusedBinbcastImpl<op_mul, 3>(pCtx, pDst);
    break;
  case 4:
    fGgmlCudaOpFusedBinbcastImpl<op_mul, 4>(pCtx, pDst);
    break;
  case 5:
    fGgmlCudaOpFusedBinbcastImpl<op_mul, 5>(pCtx, pDst);
    break;
  case 6:
    fGgmlCudaOpFusedBinbcastImpl<op_mul, 6>(pCtx, pDst);
    break;
  case 7:
    fGgmlCudaOpFusedBinbcastImpl<op_mul, 7>(pCtx, pDst);
    break;
  case 8:
    fGgmlCudaOpFusedBinbcastImpl<op_mul, 8>(pCtx, pDst);
    break;
  default:
    GGML_ASSERT(false && "Unsupported n_fuse value");
  }
}

void ggml_cuda_op_repeat_back(ggml_backend_cuda_context &pCtx, ggml_tensor *dst) {
  const ggml_tensor *src0 = dst->src[0];

  GGML_ASSERT(src0->type == dst->type);
  GGML_ASSERT(ggml_is_contiguous(dst));
  GGML_ASSERT(ggml_can_repeat(dst, src0));

  cudaStream_t vStream = pCtx.stream();

  GGML_TENSOR_UNARY_OP_LOCALS;

  GGML_ASSERT(ne2 * ne3 <= (1 << 15));

  const size_t cTs = ggml_type_size(src0->type);
  const size_t cS00 = nb00 / cTs;
  const size_t cS01 = nb01 / cTs;
  const size_t cS02 = nb02 / cTs;
  const size_t cS03 = nb03 / cTs;

  switch (dst->type) {
  case GGML_TYPE_F32: {
    const float *vSrc0D = (const float *)src0->data;
    float *vDstD = (float *)dst->data;
    fRepeatBackCuda(vSrc0D, vDstD, ne00, ne01, ne02, ne03, cS00, cS01, cS02, cS03, ne0, ne1, ne2, ne3, vStream);
  } break;
  default: {
    GGML_ASSERT(false);
  } break;
  }
}
