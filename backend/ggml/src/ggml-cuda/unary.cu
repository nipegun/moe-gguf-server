#include "unary.cuh"
#include "convert.cuh"

static __device__ __forceinline__ float op_abs(float pX) { return fabsf(pX); }

static __device__ __forceinline__ float op_sgn(float pX) { return (pX > 0.f ? 1.f : ((pX < 0.f ? -1.f : 0.f))); }

static __device__ __forceinline__ float op_neg(float pX) { return -pX; }

static __device__ __forceinline__ float op_step(float pX) { return pX > 0.0f; }

static __device__ __forceinline__ float op_gelu(float pX) { return ggml_cuda_op_gelu_single(pX); }

static __device__ __forceinline__ float op_gelu_erf(float pX) {
  const float cSqrt2Inv = 0.70710678118654752440084436210484f;

  return 0.5f * pX * (1.0f + erff(pX * cSqrt2Inv));
}

static __device__ __forceinline__ float fOpGeluQuick(float pX) {
  const float cGeluQuickCoef = -1.702f;

  return pX * (1.0f / (1.0f + expf(cGeluQuickCoef * pX)));
}

static __device__ __forceinline__ float op_silu(float pX) { return ggml_cuda_op_silu_single(pX); }

static __device__ __forceinline__ float op_tanh(float pX) { return tanhf(pX); }

static __device__ __forceinline__ float op_relu(float pX) { return fmaxf(pX, 0); }

static __device__ __forceinline__ float op_sigmoid(float pX) { return 1.0f / (1.0f + expf(-pX)); }

static __device__ __forceinline__ float op_hardsigmoid(float pX) {
  return fminf(1.0f, fmaxf(0.0f, (pX + 3.0f) / 6.0f));
}

static __device__ __forceinline__ float op_hardswish(float pX) {
  return pX * fminf(1.0f, fmaxf(0.0f, (pX + 3.0f) / 6.0f));
}

static __device__ __forceinline__ float op_exp(float pX) { return expf(pX); }

static __device__ __forceinline__ float op_sqr(float pX) { return pX * pX; }

static __device__ __forceinline__ float op_relu_sqr(float pX) {
  const float cR = fmaxf(pX, 0.0f);
  return cR * cR;
}

static __device__ __forceinline__ float op_sqrt(float pX) { return sqrtf(pX); }

static __device__ __forceinline__ float op_sin(float pX) { return sinf(pX); }

static __device__ __forceinline__ float op_cos(float pX) { return cosf(pX); }

static __device__ __forceinline__ float op_log(float pX) { return logf(pX); }

static __device__ __forceinline__ float op_expm1(float pX) { return expm1f(pX); }

static __device__ __forceinline__ float op_softplus(float pX) { return (pX > 20.0f) ? pX : logf(1.0f + expf(pX)); }

static __device__ __forceinline__ float op_elu(float pX) { return (pX > 0.f) ? pX : expm1f(pX); }

static __device__ __forceinline__ float op_floor(float pX) { return floorf(pX); }

static __device__ __forceinline__ float op_ceil(float pX) { return ceilf(pX); }

static __device__ __forceinline__ float op_round(float pX) { return round(pX); }

static __device__ __forceinline__ float op_trunc(float pX) { return trunc(pX); }

template <float (*op)(float), typename T> static __global__ void fUnaryOpKernel(const T *pX, T *pDst, const int pK) {
  ggml_cuda_pdl_lc();
  const int cI = blockDim.x * blockIdx.x + threadIdx.x;

  if (cI >= pK) {
    return;
  }

  ggml_cuda_pdl_sync();
  pDst[cI] = (T)op((float)pX[cI]);
}

template <float (*op)(float), typename T>
static void fUnaryCuda(const T *pX, T *pDst, const int pK, cudaStream_t pStream) {
  const int cNumBlocks = (pK + CUDA_NEG_BLOCK_SIZE - 1) / CUDA_NEG_BLOCK_SIZE;
  const ggml_cuda_kernel_launch_params cLaunchParams =
      ggml_cuda_kernel_launch_params((dim3)cNumBlocks, CUDA_NEG_BLOCK_SIZE, 0, pStream);
  ggml_cuda_kernel_launch(fUnaryOpKernel<op, T>, cLaunchParams, pX, pDst, pK);
}

template <float (*op)(float)> void fGgmlCudaOpUnary(ggml_backend_cuda_context &pCtx, ggml_tensor *pDst) {
  const ggml_tensor *vSrc0 = pDst->src[0];
  const void *vSrc0D = vSrc0->data;
  void *vDstD = pDst->data;
  cudaStream_t vStream = pCtx.stream();

  GGML_ASSERT(ggml_is_contiguous(vSrc0));

  GGML_ASSERT(vSrc0->type == GGML_TYPE_F32 || vSrc0->type == GGML_TYPE_F16);
  GGML_ASSERT(pDst->type == GGML_TYPE_F32 || pDst->type == GGML_TYPE_F16);
  GGML_ASSERT(vSrc0->type == pDst->type);

  if (vSrc0->type == GGML_TYPE_F16) {
    fUnaryCuda<op>((const half *)vSrc0D, (half *)vDstD, ggml_nelements(vSrc0), vStream);
  } else {
    fUnaryCuda<op>((const float *)vSrc0D, (float *)vDstD, ggml_nelements(vSrc0), vStream);
  }
}

void ggml_cuda_op_abs(ggml_backend_cuda_context &pCtx, ggml_tensor *pDst) { fGgmlCudaOpUnary<op_abs>(pCtx, pDst); }

void ggml_cuda_op_sgn(ggml_backend_cuda_context &pCtx, ggml_tensor *pDst) { fGgmlCudaOpUnary<op_sgn>(pCtx, pDst); }

void ggml_cuda_op_neg(ggml_backend_cuda_context &pCtx, ggml_tensor *pDst) { fGgmlCudaOpUnary<op_neg>(pCtx, pDst); }

void ggml_cuda_op_step(ggml_backend_cuda_context &pCtx, ggml_tensor *pDst) { fGgmlCudaOpUnary<op_step>(pCtx, pDst); }

void ggml_cuda_op_gelu(ggml_backend_cuda_context &pCtx, ggml_tensor *pDst) { fGgmlCudaOpUnary<op_gelu>(pCtx, pDst); }

void ggml_cuda_op_gelu_erf(ggml_backend_cuda_context &pCtx, ggml_tensor *pDst) {
  fGgmlCudaOpUnary<op_gelu_erf>(pCtx, pDst);
}

void ggml_cuda_op_gelu_quick(ggml_backend_cuda_context &pCtx, ggml_tensor *pDst) {
  fGgmlCudaOpUnary<fOpGeluQuick>(pCtx, pDst);
}

void ggml_cuda_op_silu(ggml_backend_cuda_context &pCtx, ggml_tensor *pDst) { fGgmlCudaOpUnary<op_silu>(pCtx, pDst); }

void ggml_cuda_op_tanh(ggml_backend_cuda_context &pCtx, ggml_tensor *pDst) { fGgmlCudaOpUnary<op_tanh>(pCtx, pDst); }

void ggml_cuda_op_relu(ggml_backend_cuda_context &pCtx, ggml_tensor *pDst) { fGgmlCudaOpUnary<op_relu>(pCtx, pDst); }

void ggml_cuda_op_sigmoid(ggml_backend_cuda_context &pCtx, ggml_tensor *pDst) {
  fGgmlCudaOpUnary<op_sigmoid>(pCtx, pDst);
}

void ggml_cuda_op_hardsigmoid(ggml_backend_cuda_context &pCtx, ggml_tensor *pDst) {
  fGgmlCudaOpUnary<op_hardsigmoid>(pCtx, pDst);
}

void ggml_cuda_op_hardswish(ggml_backend_cuda_context &pCtx, ggml_tensor *pDst) {
  fGgmlCudaOpUnary<op_hardswish>(pCtx, pDst);
}

void ggml_cuda_op_exp(ggml_backend_cuda_context &pCtx, ggml_tensor *pDst) { fGgmlCudaOpUnary<op_exp>(pCtx, pDst); }

void ggml_cuda_op_sqr(ggml_backend_cuda_context &pCtx, ggml_tensor *pDst) { fGgmlCudaOpUnary<op_sqr>(pCtx, pDst); }

void ggml_cuda_op_sqrt(ggml_backend_cuda_context &pCtx, ggml_tensor *pDst) { fGgmlCudaOpUnary<op_sqrt>(pCtx, pDst); }

void ggml_cuda_op_sin(ggml_backend_cuda_context &pCtx, ggml_tensor *pDst) { fGgmlCudaOpUnary<op_sin>(pCtx, pDst); }

void ggml_cuda_op_cos(ggml_backend_cuda_context &pCtx, ggml_tensor *pDst) { fGgmlCudaOpUnary<op_cos>(pCtx, pDst); }

void ggml_cuda_op_log(ggml_backend_cuda_context &pCtx, ggml_tensor *pDst) { fGgmlCudaOpUnary<op_log>(pCtx, pDst); }

void ggml_cuda_op_elu(ggml_backend_cuda_context &pCtx, ggml_tensor *pDst) { fGgmlCudaOpUnary<op_elu>(pCtx, pDst); }

void ggml_cuda_op_floor(ggml_backend_cuda_context &pCtx, ggml_tensor *pDst) { fGgmlCudaOpUnary<op_floor>(pCtx, pDst); }

void ggml_cuda_op_ceil(ggml_backend_cuda_context &pCtx, ggml_tensor *pDst) { fGgmlCudaOpUnary<op_ceil>(pCtx, pDst); }

void ggml_cuda_op_round(ggml_backend_cuda_context &pCtx, ggml_tensor *pDst) { fGgmlCudaOpUnary<op_round>(pCtx, pDst); }

void ggml_cuda_op_trunc(ggml_backend_cuda_context &pCtx, ggml_tensor *pDst) { fGgmlCudaOpUnary<op_trunc>(pCtx, pDst); }

void ggml_cuda_op_expm1(ggml_backend_cuda_context &pCtx, ggml_tensor *pDst) { fGgmlCudaOpUnary<op_expm1>(pCtx, pDst); }

void ggml_cuda_op_softplus(ggml_backend_cuda_context &pCtx, ggml_tensor *pDst) {
  fGgmlCudaOpUnary<op_softplus>(pCtx, pDst);
}
/* gated ops */

template <float (*op)(float), typename T>
static __global__ void fUnaryGatedOpKernel(const T *pX, const T *pG, T *pDst, const int64_t pK, const int64_t pN,
                                           const int64_t pO0, const int64_t pO1) {
  ggml_cuda_pdl_lc();
  const int64_t cI = int64_t(blockDim.x) * blockIdx.x + threadIdx.x;

  if (cI >= pK) {
    return;
  }

  // perform base op and multiply with gate (either offset in same tensor or a separate one)
  const int64_t cJ0 = (cI / pN) * pO0 + (cI % pN);
  const int64_t cJ1 = pO0 == pO1 ? cJ0 : (cI / pN) * pO1 + (cI % pN);

  ggml_cuda_pdl_sync();
  pDst[cI] = (T)(op((float)pX[cJ0]) * (float)pG[cJ1]);
}

template <float (*op)(float), typename T>
static void fUnaryGatedCuda(const T *pX, const T *pG, T *pDst, const int64_t pK, const int64_t pN, const int64_t pO0,
                            const int64_t pO1, cudaStream_t pStream) {
  const int64_t cNumBlocks = (pK + CUDA_GLU_BLOCK_SIZE - 1) / CUDA_GLU_BLOCK_SIZE;
  const ggml_cuda_kernel_launch_params cLaunchParams =
      ggml_cuda_kernel_launch_params((dim3)cNumBlocks, CUDA_GLU_BLOCK_SIZE, 0, pStream);
  ggml_cuda_kernel_launch(fUnaryGatedOpKernel<op, T>, cLaunchParams, pX, pG, pDst, pK, pN, pO0, pO1);
}

template <float (*op)(float)> void fGgmlCudaOpUnaryGated(ggml_backend_cuda_context &pCtx, ggml_tensor *pDst) {
  const ggml_tensor *vSrc0 = pDst->src[0];
  const ggml_tensor *vSrc1 = pDst->src[1];
  void *vSrc0D = vSrc0->data;
  void *vSrc1D = vSrc1 ? vSrc1->data : vSrc0->data;
  const int64_t cSrc0O = vSrc0->nb[1];
  const int64_t cSrc1O = vSrc1 ? vSrc1->nb[1] : vSrc0->nb[1];
  void *vDstD = pDst->data;
  const int64_t cNc = vSrc1 ? vSrc0->ne[0] : vSrc0->ne[0] / 2;
  cudaStream_t vStream = pCtx.stream();

  GGML_ASSERT(ggml_is_contiguous_1(vSrc0));
  GGML_ASSERT(vSrc0->nb[0] == ggml_element_size(vSrc0));
  GGML_ASSERT(ggml_is_contiguous(pDst));

  GGML_ASSERT(vSrc0->type == GGML_TYPE_F32 || vSrc0->type == GGML_TYPE_F16);
  GGML_ASSERT(pDst->type == GGML_TYPE_F32 || pDst->type == GGML_TYPE_F16);
  GGML_ASSERT(vSrc0->type == pDst->type);
  GGML_ASSERT(pDst->ne[0] == cNc);
  GGML_ASSERT(ggml_nrows(pDst) == ggml_nrows(vSrc0));

  if (vSrc1) {
    GGML_ASSERT(ggml_is_contiguous_1(vSrc1));
    GGML_ASSERT(vSrc1->nb[0] == ggml_element_size(vSrc1));
    GGML_ASSERT(vSrc1->ne[0] == cNc);
    GGML_ASSERT(vSrc0->type == vSrc1->type);
  }

  const int32_t cSwapped = ((const int32_t *)pDst->op_params)[1];

  if (vSrc0->type == GGML_TYPE_F16) {
    half *vSrc0P = (half *)vSrc0D;
    half *vSrc1P = (half *)vSrc1D;

    if (!vSrc1) {
      vSrc0P += cSwapped ? cNc : 0;
      vSrc1P += cSwapped ? 0 : cNc;
    }

    fUnaryGatedCuda<op>(vSrc0P, vSrc1P, (half *)vDstD, ggml_nelements(pDst), cNc, cSrc0O / sizeof(half),
                        cSrc1O / sizeof(half), vStream);
  } else {
    float *vSrc0P = (float *)vSrc0D;
    float *vSrc1P = (float *)vSrc1D;

    if (!vSrc1) {
      vSrc0P += cSwapped ? cNc : 0;
      vSrc1P += cSwapped ? 0 : cNc;
    }

    fUnaryGatedCuda<op>(vSrc0P, vSrc1P, (float *)vDstD, ggml_nelements(pDst), cNc, cSrc0O / sizeof(float),
                        cSrc1O / sizeof(float), vStream);
  }
}

void ggml_cuda_op_reglu(ggml_backend_cuda_context &pCtx, ggml_tensor *pDst) {
  fGgmlCudaOpUnaryGated<op_relu>(pCtx, pDst);
}

void ggml_cuda_op_geglu(ggml_backend_cuda_context &pCtx, ggml_tensor *pDst) {
  fGgmlCudaOpUnaryGated<op_gelu>(pCtx, pDst);
}

void ggml_cuda_op_swiglu(ggml_backend_cuda_context &pCtx, ggml_tensor *pDst) {
  fGgmlCudaOpUnaryGated<op_silu>(pCtx, pDst);
}

void ggml_cuda_op_geglu_erf(ggml_backend_cuda_context &pCtx, ggml_tensor *pDst) {
  fGgmlCudaOpUnaryGated<op_gelu_erf>(pCtx, pDst);
}

void ggml_cuda_op_geglu_quick(ggml_backend_cuda_context &pCtx, ggml_tensor *pDst) {
  fGgmlCudaOpUnaryGated<fOpGeluQuick>(pCtx, pDst);
}

// swiglu_oai

template <typename T>
static __global__ void fSwigluOaiKernel(const T *pX, const T *pG, T *pDst, const int64_t pK, const int64_t pN,
                                        const int64_t pO0, const int64_t pO1, float pAlpha, float pLimit) {
  const int64_t cI = int64_t(blockDim.x) * blockIdx.x + threadIdx.x;

  if (cI >= pK) {
    return;
  }

  // perform base op and multiply with gate (either offset in same tensor or a separate one)
  const int64_t cJ0 = (cI / pN) * pO0 + (cI % pN);
  const int64_t cJ1 = pO0 == pO1 ? cJ0 : (cI / pN) * pO1 + (cI % pN);

  float vXi = pX[cJ0];
  float vGi = pG[cJ1];

  pDst[cI] = ggml_cuda_op_swiglu_oai_single(vXi, vGi, pAlpha, pLimit);
}

template <typename T>
static void fSwigluOaiCuda(const T *pX, const T *pG, T *pDst, const int64_t pK, const int64_t pN, const int64_t pO0,
                           const int64_t pO1, const float pAlpha, const float pLimit, cudaStream_t pStream) {
  const int64_t cNumBlocks = (pK + CUDA_GLU_BLOCK_SIZE - 1) / CUDA_GLU_BLOCK_SIZE;
  fSwigluOaiKernel<<<cNumBlocks, CUDA_GLU_BLOCK_SIZE, 0, pStream>>>(pX, pG, pDst, pK, pN, pO0, pO1, pAlpha, pLimit);
}

void ggml_cuda_op_swiglu_oai(ggml_backend_cuda_context &pCtx, ggml_tensor *pDst) {
  const ggml_tensor *vSrc0 = pDst->src[0];
  const ggml_tensor *vSrc1 = pDst->src[1];
  void *vSrc0D = vSrc0->data;
  void *vSrc1D = vSrc1 ? vSrc1->data : vSrc0->data;
  const int64_t cSrc0O = vSrc0->nb[1];
  const int64_t cSrc1O = vSrc1 ? vSrc1->nb[1] : vSrc0->nb[1];
  void *vDstD = pDst->data;
  const int64_t cNc = vSrc1 ? vSrc0->ne[0] : vSrc0->ne[0] / 2;
  cudaStream_t vStream = pCtx.stream();

  GGML_ASSERT(ggml_is_contiguous_1(vSrc0));
  GGML_ASSERT(vSrc0->nb[0] == ggml_element_size(vSrc0));
  GGML_ASSERT(ggml_is_contiguous(pDst));

  GGML_ASSERT(vSrc0->type == GGML_TYPE_F32);
  GGML_ASSERT(pDst->type == GGML_TYPE_F32);
  GGML_ASSERT(vSrc0->type == pDst->type);
  GGML_ASSERT(pDst->ne[0] == cNc);
  GGML_ASSERT(ggml_nrows(pDst) == ggml_nrows(vSrc0));

  if (vSrc1) {
    GGML_ASSERT(ggml_is_contiguous_1(vSrc1));
    GGML_ASSERT(vSrc1->nb[0] == ggml_element_size(vSrc1));
    GGML_ASSERT(vSrc1->ne[0] == cNc);
    GGML_ASSERT(vSrc0->type == vSrc1->type);
  }

  // const int32_t swapped = ((const int32_t *) dst->op_params)[1];
  const int32_t cSwapped = ggml_get_op_params_i32(pDst, 1);
  const float cAlpha = ggml_get_op_params_f32(pDst, 2);
  const float cLimit = ggml_get_op_params_f32(pDst, 3);

  float *vSrc0P = (float *)vSrc0D;
  float *vSrc1P = (float *)vSrc1D;

  if (!vSrc1) {
    vSrc0P += cSwapped ? cNc : 0;
    vSrc1P += cSwapped ? 0 : cNc;
  }

  fSwigluOaiCuda(vSrc0P, vSrc1P, (float *)vDstD, ggml_nelements(pDst), cNc, cSrc0O / sizeof(float),
                 cSrc1O / sizeof(float), cAlpha, cLimit, vStream);
}

// swiglu_clamp

template <typename T>
static __global__ void fSwigluClampKernel(const T *pGate, const T *pUp, T *pDst, const int64_t pK, const int64_t pN,
                                          const int64_t pO0, const int64_t pO1, float pLimit) {
  const int64_t cI = int64_t(blockDim.x) * blockIdx.x + threadIdx.x;

  if (cI >= pK) {
    return;
  }

  const int64_t cJ0 = (cI / pN) * pO0 + (cI % pN);
  const int64_t cJ1 = pO0 == pO1 ? cJ0 : (cI / pN) * pO1 + (cI % pN);

  pDst[cI] = (T)ggml_cuda_op_swiglu_clamp_single((float)pGate[cJ0], (float)pUp[cJ1], pLimit);
}

template <typename T>
static void fSwigluClampCuda(const T *pGate, const T *pUp, T *pDst, const int64_t pK, const int64_t pN,
                             const int64_t pO0, const int64_t pO1, const float pLimit, cudaStream_t pStream) {
  const int64_t cNumBlocks = (pK + CUDA_GLU_BLOCK_SIZE - 1) / CUDA_GLU_BLOCK_SIZE;
  fSwigluClampKernel<<<cNumBlocks, CUDA_GLU_BLOCK_SIZE, 0, pStream>>>(pGate, pUp, pDst, pK, pN, pO0, pO1, pLimit);
}

void ggml_cuda_op_swiglu_clamp(ggml_backend_cuda_context &pCtx, ggml_tensor *pDst) {
  const ggml_tensor *vSrc0 = pDst->src[0];
  const ggml_tensor *vSrc1 = pDst->src[1];
  void *vSrc0D = vSrc0->data;
  void *vSrc1D = vSrc1 ? vSrc1->data : vSrc0->data;
  const int64_t cSrc0O = vSrc0->nb[1];
  const int64_t cSrc1O = vSrc1 ? vSrc1->nb[1] : vSrc0->nb[1];
  void *vDstD = pDst->data;
  const int64_t cNc = vSrc1 ? vSrc0->ne[0] : vSrc0->ne[0] / 2;
  cudaStream_t vStream = pCtx.stream();

  GGML_ASSERT(ggml_is_contiguous_1(vSrc0));
  GGML_ASSERT(vSrc0->nb[0] == ggml_element_size(vSrc0));
  GGML_ASSERT(ggml_is_contiguous(pDst));

  GGML_ASSERT(vSrc0->type == GGML_TYPE_F32 || vSrc0->type == GGML_TYPE_F16);
  GGML_ASSERT(vSrc0->type == pDst->type);
  GGML_ASSERT(pDst->ne[0] == cNc);
  GGML_ASSERT(ggml_nrows(pDst) == ggml_nrows(vSrc0));

  if (vSrc1) {
    GGML_ASSERT(ggml_is_contiguous_1(vSrc1));
    GGML_ASSERT(vSrc1->nb[0] == ggml_element_size(vSrc1));
    GGML_ASSERT(vSrc1->ne[0] == cNc);
    GGML_ASSERT(vSrc0->type == vSrc1->type);
  }

  const int32_t cSwapped = ggml_get_op_params_i32(pDst, 1);
  const float cLimit = ggml_get_op_params_f32(pDst, 3);

  if (vSrc0->type == GGML_TYPE_F16) {
    half *vSrc0P = (half *)vSrc0D;
    half *vSrc1P = (half *)vSrc1D;

    if (!vSrc1) {
      vSrc0P += cSwapped ? cNc : 0;
      vSrc1P += cSwapped ? 0 : cNc;
    }

    fSwigluClampCuda(vSrc0P, vSrc1P, (half *)vDstD, ggml_nelements(pDst), cNc, cSrc0O / sizeof(half),
                     cSrc1O / sizeof(half), cLimit, vStream);
  } else {
    float *vSrc0P = (float *)vSrc0D;
    float *vSrc1P = (float *)vSrc1D;

    if (!vSrc1) {
      vSrc0P += cSwapped ? cNc : 0;
      vSrc1P += cSwapped ? 0 : cNc;
    }

    fSwigluClampCuda(vSrc0P, vSrc1P, (float *)vDstD, ggml_nelements(pDst), cNc, cSrc0O / sizeof(float),
                     cSrc1O / sizeof(float), cLimit, vStream);
  }
}

/* CUDA kernel + launcher for xIELU */

template <typename T>
static __global__ void fXieluKernel(const T *pX, T *pDst, const int pK, float pAlphaN, float pAlphaP, float pBeta,
                                    float pEps) {
  const int cI = blockDim.x * blockIdx.x + threadIdx.x;

  if (cI >= pK) {
    return;
  }

  const float cXi = ggml_cuda_cast<float>(pX[cI]);

  const float cGatePos = (cXi > 0.0f);
  const float cYPos = pAlphaP * cXi * cXi + pBeta * cXi;
  const float cMinVEps = fminf(cXi, pEps);
  const float cYNeg = (expm1f(cMinVEps) - cXi) * pAlphaN + pBeta * cXi;
  const float cOut = cGatePos * cYPos + (1.0f - cGatePos) * cYNeg;

  pDst[cI] = ggml_cuda_cast<T>(cOut);
}

template <typename T>
static void fXieluCuda(const T *pX, T *pDst, const int pK, float pAlphaN, float pAlphaP, float pBeta, float pEps,
                       cudaStream_t pStream) {
  const int cNumBlocks = (pK + CUDA_XIELU_BLOCK_SIZE) / CUDA_XIELU_BLOCK_SIZE;
  fXieluKernel<<<cNumBlocks, CUDA_XIELU_BLOCK_SIZE, 0, pStream>>>(pX, pDst, pK, pAlphaN, pAlphaP, pBeta, pEps);
}

void ggml_cuda_op_xielu(ggml_backend_cuda_context &pCtx, ggml_tensor *pDst) {
  const ggml_tensor *vSrc0 = pDst->src[0];
  const void *vSrc0D = vSrc0->data;
  void *vDstD = pDst->data;
  cudaStream_t vStream = pCtx.stream();

  GGML_ASSERT(ggml_is_contiguous(vSrc0));

  GGML_ASSERT(vSrc0->type == GGML_TYPE_F32 || vSrc0->type == GGML_TYPE_F16);
  GGML_ASSERT(pDst->type == GGML_TYPE_F32 || pDst->type == GGML_TYPE_F16);
  GGML_ASSERT(vSrc0->type == pDst->type);

  const float cAlphaN = ggml_get_op_params_f32(pDst, 1);
  const float cAlphaP = ggml_get_op_params_f32(pDst, 2);
  const float cBeta = ggml_get_op_params_f32(pDst, 3);
  const float cEps = ggml_get_op_params_f32(pDst, 4);

  if (vSrc0->type == GGML_TYPE_F16) {
    fXieluCuda((const half *)vSrc0D, (half *)vDstD, ggml_nelements(vSrc0), cAlphaN, cAlphaP, cBeta, cEps, vStream);
  } else {
    fXieluCuda((const float *)vSrc0D, (float *)vDstD, ggml_nelements(vSrc0), cAlphaN, cAlphaP, cBeta, cEps, vStream);
  }
}

/* silu_back */

static __device__ __forceinline__ float fOpSiluBack(float pGrad, float pX) {
  const float cS = 1.0f / (1.0f + expf(-pX));
  return pGrad * cS * (1.0f + pX * (1.0f - cS));
}

template <class T> static __global__ void fSiluBackKernel(const T *pGrad, const T *pXf, T *pDst, const int pK) {
  const int cI = blockDim.x * blockIdx.x + threadIdx.x;

  if (cI >= pK) {
    return;
  }

  pDst[cI] = (T)fOpSiluBack((float)pGrad[cI], (float)pXf[cI]);
}

template <class T> static void fSiluBackCuda(const T *pGrad, const T *pX, T *pDst, const int pK, cudaStream_t pStream) {
  const int cNumBlocks = (pK + CUDA_SILU_BACK_BLOCK_SIZE - 1) / CUDA_SILU_BLOCK_SIZE;
  fSiluBackKernel<<<cNumBlocks, CUDA_SILU_BACK_BLOCK_SIZE, 0, pStream>>>(pGrad, pX, pDst, pK);
}

void ggml_cuda_op_silu_back(ggml_backend_cuda_context &pCtx, ggml_tensor *pDst) {
  const ggml_tensor *vSrc0 = pDst->src[0]; // input from forward pass
  const ggml_tensor *vSrc1 = pDst->src[1]; // grads of forward pass output

  const float *vSrc0D = (const float *)vSrc0->data;
  const float *vSrc1D = (const float *)vSrc1->data;
  float *vDstD = (float *)pDst->data;

  cudaStream_t vStream = pCtx.stream();

  GGML_ASSERT(ggml_is_contiguous(vSrc0));

  GGML_ASSERT(vSrc0->type == GGML_TYPE_F32 || vSrc0->type == GGML_TYPE_F16);
  GGML_ASSERT(pDst->type == GGML_TYPE_F32 || pDst->type == GGML_TYPE_F16);
  GGML_ASSERT(vSrc0->type == pDst->type);

  if (vSrc0->type == GGML_TYPE_F16) {
    fSiluBackCuda((const half *)vSrc0D, (const half *)vSrc1D, (half *)vDstD, ggml_nelements(vSrc0), vStream);
  } else {
    fSiluBackCuda((const float *)vSrc0D, (const float *)vSrc1D, (float *)vDstD, ggml_nelements(vSrc0), vStream);
  }
}

/* leaky relu */

static __device__ __forceinline__ float fOpLeakyRelu(float pX, const float pNegativeSlope) {
  return fmaxf(pX, 0) + fminf(pX, 0.0f) * pNegativeSlope;
}

template <class T>
static __global__ void fLeakyReluKernel(const T *pX, T *pDst, const int pK, const float pNegativeSlope) {
  const int cI = blockDim.x * blockIdx.x + threadIdx.x;

  if (cI >= pK) {
    return;
  }

  pDst[cI] = (T)fOpLeakyRelu((float)pX[cI], pNegativeSlope);
}

template <class T>
static void fLeakyReluCuda(const T *pX, T *pDst, const int pK, const float pNegativeSlope, cudaStream_t pStream) {
  const int cNumBlocks = (pK + CUDA_RELU_BLOCK_SIZE - 1) / CUDA_RELU_BLOCK_SIZE;
  fLeakyReluKernel<<<cNumBlocks, CUDA_RELU_BLOCK_SIZE, 0, pStream>>>(pX, pDst, pK, pNegativeSlope);
}

void ggml_cuda_op_leaky_relu(ggml_backend_cuda_context &pCtx, ggml_tensor *pDst) {
  const ggml_tensor *vSrc0 = pDst->src[0];
  const void *vSrc0D = vSrc0->data;
  void *vDstD = pDst->data;
  cudaStream_t vStream = pCtx.stream();

  GGML_ASSERT(ggml_is_contiguous(vSrc0));

  GGML_ASSERT(vSrc0->type == GGML_TYPE_F32 || vSrc0->type == GGML_TYPE_F16);
  GGML_ASSERT(pDst->type == GGML_TYPE_F32 || pDst->type == GGML_TYPE_F16);
  GGML_ASSERT(vSrc0->type == pDst->type);

  float vNegativeSlope;
  memcpy(&vNegativeSlope, pDst->op_params, sizeof(float));

  if (vSrc0->type == GGML_TYPE_F16) {
    fLeakyReluCuda((const half *)vSrc0D, (half *)vDstD, ggml_nelements(vSrc0), vNegativeSlope, vStream);
  } else {
    fLeakyReluCuda((const float *)vSrc0D, (float *)vDstD, ggml_nelements(vSrc0), vNegativeSlope, vStream);
  }
}

/* fused unary + mul */

template <float (*op)(float)>
static void fGgmlCudaOpUnaryMulImpl(ggml_backend_cuda_context &pCtx, ggml_tensor *pUnaryNode, ggml_tensor *pMulNode) {
  // unary_node: UNARY op applied to unary_node->src[0]
  // mul_node:   MUL(a, b) where one of a/b is unary_node
  // Output goes to mul_node->data

  const ggml_tensor *vUnarySrc = pUnaryNode->src[0]; // input to the unary op
  const ggml_tensor *vOtherSrc = (pMulNode->src[0] == pUnaryNode) ? pMulNode->src[1] : pMulNode->src[0];

  GGML_ASSERT(ggml_is_contiguous_1(vUnarySrc));
  GGML_ASSERT(vUnarySrc->nb[0] == ggml_element_size(vUnarySrc));
  GGML_ASSERT(ggml_is_contiguous_1(vOtherSrc));
  GGML_ASSERT(vOtherSrc->nb[0] == ggml_element_size(vOtherSrc));
  GGML_ASSERT(ggml_are_same_shape(vUnarySrc, vOtherSrc));

  GGML_ASSERT(vUnarySrc->type == GGML_TYPE_F32 || vUnarySrc->type == GGML_TYPE_F16);
  GGML_ASSERT(vUnarySrc->type == vOtherSrc->type);
  GGML_ASSERT(vUnarySrc->type == pMulNode->type);

  cudaStream_t vStream = pCtx.stream();

  const int64_t cK = ggml_nelements(pMulNode);
  const int64_t cNc = vUnarySrc->ne[0];
  const int64_t cUnaryStride = vUnarySrc->nb[1];
  const int64_t cOtherStride = vOtherSrc->nb[1];

  if (vUnarySrc->type == GGML_TYPE_F16) {
    fUnaryGatedCuda<op>((const half *)vUnarySrc->data, (const half *)vOtherSrc->data, (half *)pMulNode->data, cK, cNc,
                        cUnaryStride / sizeof(half), cOtherStride / sizeof(half), vStream);
  } else {
    fUnaryGatedCuda<op>((const float *)vUnarySrc->data, (const float *)vOtherSrc->data, (float *)pMulNode->data, cK,
                        cNc, cUnaryStride / sizeof(float), cOtherStride / sizeof(float), vStream);
  }
}

void ggml_cuda_op_unary_mul(ggml_backend_cuda_context &pCtx, ggml_tensor *pUnaryNode, ggml_tensor *pMulNode) {
  switch (ggml_get_unary_op(pUnaryNode)) {
  case GGML_UNARY_OP_SILU:
    fGgmlCudaOpUnaryMulImpl<op_silu>(pCtx, pUnaryNode, pMulNode);
    break;
  case GGML_UNARY_OP_SIGMOID:
    fGgmlCudaOpUnaryMulImpl<op_sigmoid>(pCtx, pUnaryNode, pMulNode);
    break;
  case GGML_UNARY_OP_SOFTPLUS:
    fGgmlCudaOpUnaryMulImpl<op_softplus>(pCtx, pUnaryNode, pMulNode);
    break;
  default:
    GGML_ABORT("Unsupported unary op for fused unary+mul");
  }
}

/* fused relu + sqr */

void ggml_cuda_op_relu_sqr(ggml_backend_cuda_context &pCtx, ggml_tensor *pReluNode, ggml_tensor *pSqrNode) {
  const ggml_tensor *vSrc = pReluNode->src[0];
  cudaStream_t vStream = pCtx.stream();

  GGML_ASSERT(ggml_is_contiguous(vSrc));
  GGML_ASSERT(vSrc->type == GGML_TYPE_F32 || vSrc->type == GGML_TYPE_F16);
  GGML_ASSERT(vSrc->type == pSqrNode->type);

  const int cK = ggml_nelements(vSrc);
  if (vSrc->type == GGML_TYPE_F16) {
    fUnaryCuda<op_relu_sqr>((const half *)vSrc->data, (half *)pSqrNode->data, cK, vStream);
  } else {
    fUnaryCuda<op_relu_sqr>((const float *)vSrc->data, (float *)pSqrNode->data, cK, vStream);
  }
}
