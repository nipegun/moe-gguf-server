#include "col2im-1d.cuh"
#include "convert.cuh"

// col2im_1d: scatter-add GEMM columns to 1D signal (gather approach)
// columns: [K*OC, T_in]  ->  output: [T_out, OC]
// Supports F32, F16, BF16 data with F32 accumulator.

template <typename T>
static __global__ void fCol2im1dKernel(const T *__restrict__ pCol, T *__restrict__ pDst, const int pTIn,
                                       const uint3 pTOutFd, const int pOc, const int pK, const int pKOc, const int pS0,
                                       const int p0, const int pTotal) {

  const int cIdx = threadIdx.x + blockIdx.x * blockDim.x;
  if (cIdx >= pTotal)
    return;

  // dst layout: [T_out, OC], ne[0]=T_out fastest
  const uint2 cQr = fast_div_modulo((uint32_t)cIdx, pTOutFd); // qr.x = idx / T_out, qr.y = idx % T_out
  const int cOc = (int)cQr.x;
  const int cTOut = (int)cQr.y;
  const int cTAbs = cTOut + p0; // absolute position in uncropped signal

  // Gather: find all (t_in, k) where t_in*s + k == t_abs, 0 <= k < K
  int vTInMin = (cTAbs - pK + pS0) / pS0; // ceil((t_abs - K + 1) / s)
  if (vTInMin < 0)
    vTInMin = 0;
  int vTInMax = cTAbs / pS0;
  if (vTInMax >= pTIn)
    vTInMax = pTIn - 1;

  float vSum = 0.0f;
  for (int vTIn = vTInMin; vTIn <= vTInMax; vTIn++) {
    const int cK = cTAbs - vTIn * pS0;
    // col layout: [K*OC, T_in], column index = oc * K + k
    vSum += ggml_cuda_cast<float>(pCol[(cOc * pK + cK) + vTIn * pKOc]);
  }

  pDst[cIdx] = ggml_cuda_cast<T>(vSum);
}

void ggml_cuda_op_col2im_1d(ggml_backend_cuda_context &pCtx, ggml_tensor *pDst) {
  const ggml_tensor *vSrc0 = pDst->src[0];
  cudaStream_t vStream = pCtx.stream();

  GGML_ASSERT(ggml_is_contiguous(vSrc0));

  const int32_t cS0 = ((const int32_t *)(pDst->op_params))[0];
  const int32_t cOc = ((const int32_t *)(pDst->op_params))[1];
  const int32_t cP0 = ((const int32_t *)(pDst->op_params))[2];

  const int cKOc = (int)vSrc0->ne[0];
  const int cTIn = (int)vSrc0->ne[1];
  const int cK = cKOc / cOc;
  const int cTOut = (int)pDst->ne[0];

  const uint3 cTOutFd = init_fastdiv_values((uint32_t)cTOut);

  const int cTotal = cTOut * cOc;
  const int cBlockSize = 256;
  const int cNumBlocks = (cTotal + cBlockSize - 1) / cBlockSize;

  switch (vSrc0->type) {
  case GGML_TYPE_F32: {
    fCol2im1dKernel<<<cNumBlocks, cBlockSize, 0, vStream>>>((const float *)vSrc0->data, (float *)pDst->data, cTIn,
                                                            cTOutFd, cOc, cK, cKOc, cS0, cP0, cTotal);
  } break;
  case GGML_TYPE_F16: {
    fCol2im1dKernel<<<cNumBlocks, cBlockSize, 0, vStream>>>((const half *)vSrc0->data, (half *)pDst->data, cTIn,
                                                            cTOutFd, cOc, cK, cKOc, cS0, cP0, cTotal);
  } break;
  case GGML_TYPE_BF16: {
    fCol2im1dKernel<<<cNumBlocks, cBlockSize, 0, vStream>>>((const nv_bfloat16 *)vSrc0->data, (nv_bfloat16 *)pDst->data,
                                                            cTIn, cTOutFd, cOc, cK, cKOc, cS0, cP0, cTotal);
  } break;
  default:
    GGML_ABORT("col2im_1d: unsupported type");
  }
}
