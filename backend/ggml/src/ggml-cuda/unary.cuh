#pragma once
#include "common.cuh"

#define CUDA_NEG_BLOCK_SIZE 256
#define CUDA_STEP_BLOCK_SIZE 256
#define CUDA_GELU_BLOCK_SIZE 256
#define CUDA_SILU_BLOCK_SIZE 256
#define CUDA_SILU_BACK_BLOCK_SIZE 256
#define CUDA_TANH_BLOCK_SIZE 256
#define CUDA_RELU_BLOCK_SIZE 256
#define CUDA_SIGMOID_BLOCK_SIZE 256
#define CUDA_HARDSIGMOID_BLOCK_SIZE 256
#define CUDA_EXP_BLOCK_SIZE 256
#define CUDA_HARDSWISH_BLOCK_SIZE 256
#define CUDA_SQR_BLOCK_SIZE 256
#define CUDA_SQRT_BLOCK_SIZE 256
#define CUDA_SIN_BLOCK_SIZE 256
#define CUDA_COS_BLOCK_SIZE 256
#define CUDA_GLU_BLOCK_SIZE 256
#define CUDA_XIELU_BLOCK_SIZE 256

void ggml_cuda_op_abs(ggml_backend_cuda_context &pCtx, ggml_tensor *pDst);

void ggml_cuda_op_sgn(ggml_backend_cuda_context &pCtx, ggml_tensor *pDst);

void ggml_cuda_op_neg(ggml_backend_cuda_context &pCtx, ggml_tensor *pDst);

void ggml_cuda_op_step(ggml_backend_cuda_context &pCtx, ggml_tensor *pDst);

void ggml_cuda_op_gelu(ggml_backend_cuda_context &pCtx, ggml_tensor *pDst);

void ggml_cuda_op_silu(ggml_backend_cuda_context &pCtx, ggml_tensor *pDst);

void ggml_cuda_op_silu_back(ggml_backend_cuda_context &pCtx, ggml_tensor *pDst);

void ggml_cuda_op_gelu_erf(ggml_backend_cuda_context &pCtx, ggml_tensor *pDst);

void ggml_cuda_op_gelu_quick(ggml_backend_cuda_context &pCtx, ggml_tensor *pDst);

void ggml_cuda_op_tanh(ggml_backend_cuda_context &pCtx, ggml_tensor *pDst);

void ggml_cuda_op_relu(ggml_backend_cuda_context &pCtx, ggml_tensor *pDst);

void ggml_cuda_op_sigmoid(ggml_backend_cuda_context &pCtx, ggml_tensor *pDst);

void ggml_cuda_op_hardsigmoid(ggml_backend_cuda_context &pCtx, ggml_tensor *pDst);

void ggml_cuda_op_exp(ggml_backend_cuda_context &pCtx, ggml_tensor *pDst);

void ggml_cuda_op_hardswish(ggml_backend_cuda_context &pCtx, ggml_tensor *pDst);

void ggml_cuda_op_leaky_relu(ggml_backend_cuda_context &pCtx, ggml_tensor *pDst);

void ggml_cuda_op_sqr(ggml_backend_cuda_context &pCtx, ggml_tensor *pDst);

void ggml_cuda_op_sqrt(ggml_backend_cuda_context &pCtx, ggml_tensor *pDst);

void ggml_cuda_op_sin(ggml_backend_cuda_context &pCtx, ggml_tensor *pDst);

void ggml_cuda_op_cos(ggml_backend_cuda_context &pCtx, ggml_tensor *pDst);

void ggml_cuda_op_log(ggml_backend_cuda_context &pCtx, ggml_tensor *pDst);

void ggml_cuda_op_expm1(ggml_backend_cuda_context &pCtx, ggml_tensor *pDst);

void ggml_cuda_op_softplus(ggml_backend_cuda_context &pCtx, ggml_tensor *pDst);

void ggml_cuda_op_elu(ggml_backend_cuda_context &pCtx, ggml_tensor *pDst);

void ggml_cuda_op_floor(ggml_backend_cuda_context &pCtx, ggml_tensor *pDst);

void ggml_cuda_op_ceil(ggml_backend_cuda_context &pCtx, ggml_tensor *pDst);

void ggml_cuda_op_round(ggml_backend_cuda_context &pCtx, ggml_tensor *pDst);

void ggml_cuda_op_trunc(ggml_backend_cuda_context &pCtx, ggml_tensor *pDst);

void ggml_cuda_op_reglu(ggml_backend_cuda_context &pCtx, ggml_tensor *pDst);

void ggml_cuda_op_geglu(ggml_backend_cuda_context &pCtx, ggml_tensor *pDst);

void ggml_cuda_op_swiglu(ggml_backend_cuda_context &pCtx, ggml_tensor *pDst);

void ggml_cuda_op_swiglu_oai(ggml_backend_cuda_context &pCtx, ggml_tensor *pDst);

void ggml_cuda_op_swiglu_clamp(ggml_backend_cuda_context &pCtx, ggml_tensor *pDst);

void ggml_cuda_op_geglu_erf(ggml_backend_cuda_context &pCtx, ggml_tensor *pDst);

void ggml_cuda_op_geglu_quick(ggml_backend_cuda_context &pCtx, ggml_tensor *pDst);

void ggml_cuda_op_xielu(ggml_backend_cuda_context &pCtx, ggml_tensor *pDst);

void ggml_cuda_op_unary_mul(ggml_backend_cuda_context &pCtx, ggml_tensor *pUnaryNode, ggml_tensor *pMulNode);

void ggml_cuda_op_relu_sqr(ggml_backend_cuda_context &pCtx, ggml_tensor *pReluNode, ggml_tensor *pSqrNode);

__device__ __forceinline__ float ggml_cuda_op_silu_single(float pX) { return pX / (1.0f + expf(-pX)); }

__device__ __forceinline__ float ggml_cuda_op_gelu_single(float pX) {
  const float cGeluCoefA = 0.044715f;
  const float cSqrt2OverPi = 0.79788456080286535587989211986876f;

  return 0.5f * pX * (1.0f + tanhf(cSqrt2OverPi * pX * (1.0f + cGeluCoefA * pX * pX)));
}

__device__ __forceinline__ float ggml_cuda_op_swiglu_oai_single(float pX, float pG, float pAlpha = 1.702f,
                                                                float pLimit = 7.0f) {
  pX = fminf(pX, pLimit);
  pG = fmaxf(fminf(pG, pLimit), -pLimit);

  float vOutGlu = pX / (1.0f + expf(-pX * pAlpha));
  vOutGlu = vOutGlu * (1.0f + pG);
  return vOutGlu;
}

__device__ __forceinline__ float ggml_cuda_op_swiglu_clamp_single(float pGate, float pUp, float pLimit) {
  pGate = fminf(pGate, pLimit);
  pUp = fmaxf(fminf(pUp, pLimit), -pLimit);

  return ggml_cuda_op_silu_single(pGate) * pUp;
}
