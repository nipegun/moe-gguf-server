#include "snake.cuh"
#include "convert.cuh"

// Fused Snake activation: y = x + sin^2(a * x) * inv_b
// x: [T, C] (T contiguous), a: [1, C], inv_b: [1, C]
// Supports F32, F16, BF16 data with F32 compute.

template <typename T>
static __global__ void fSnakeKernel(const T *__restrict__ pX, const float *__restrict__ pA,
                                    const float *__restrict__ pInvB, T *__restrict__ pDst, const int pTotal,
                                    const uint3 pTLenFastdiv) {
  const int cIdx = blockIdx.x * blockDim.x + threadIdx.x;
  if (cIdx >= pTotal)
    return;

  const int c_ = (int)fastdiv((uint32_t)cIdx, pTLenFastdiv);

  const float cXi = ggml_cuda_cast<float>(pX[cIdx]);
  const float cS = sinf(pA[c_] * cXi);
  pDst[cIdx] = ggml_cuda_cast<T>(cXi + cS * cS * pInvB[c_]);
}

// Internal launcher with explicit x/a/inv_b/dst tensors.
// Shared by the public op (reads dst->src) and the fusion path (explicit args).
static void fLaunchSnake(ggml_backend_cuda_context &pCtx, const ggml_tensor *pX, const ggml_tensor *pA,
                         const ggml_tensor *pInvB, ggml_tensor *pDst) {
  const float *vAD = (const float *)pA->data;
  const float *vInvBD = (const float *)pInvB->data;

  const int cT = (int)pX->ne[0];
  const int cC = (int)pX->ne[1];
  const int cTotal = cT * cC;
  const uint3 cTLenFastdiv = init_fastdiv_values((uint64_t)cT);

  const int cBlockSize = 256;
  const int cGridSize = (cTotal + cBlockSize - 1) / cBlockSize;

  cudaStream_t vStream = pCtx.stream();

  switch (pX->type) {
  case GGML_TYPE_F32: {
    fSnakeKernel<<<cGridSize, cBlockSize, 0, vStream>>>((const float *)pX->data, vAD, vInvBD, (float *)pDst->data,
                                                        cTotal, cTLenFastdiv);
  } break;
  case GGML_TYPE_F16: {
    fSnakeKernel<<<cGridSize, cBlockSize, 0, vStream>>>((const half *)pX->data, vAD, vInvBD, (half *)pDst->data, cTotal,
                                                        cTLenFastdiv);
  } break;
  case GGML_TYPE_BF16: {
    fSnakeKernel<<<cGridSize, cBlockSize, 0, vStream>>>((const nv_bfloat16 *)pX->data, vAD, vInvBD,
                                                        (nv_bfloat16 *)pDst->data, cTotal, cTLenFastdiv);
  } break;
  default:
    GGML_ABORT("snake: unsupported type");
  }
}

// Fusion entry: caller supplies x/a/inv_b explicitly from the matched
// mul -> sin -> sqr -> mul -> add pattern. The dst is the trailing add output.
void ggml_cuda_op_snake_fused(ggml_backend_cuda_context &pCtx, const ggml_tensor *pX, const ggml_tensor *pA,
                              const ggml_tensor *pInvB, ggml_tensor *pDst) {
  fLaunchSnake(pCtx, pX, pA, pInvB, pDst);
}
