#include "pool2d.cuh"

template <typename Ti, typename To>
static __global__ void fPool2dNchwKernel(const int pIh, const int pIw, const int pOh, const int pOw, const int pKh,
                                         const int pKw, const int pSh, const int pSw, const int pPh, const int pPw,
                                         const int pParallelElements, const Ti *pSrc, To *pDst,
                                         const enum ggml_op_pool pOp) {
  int vIdx = threadIdx.x + blockIdx.x * blockDim.x;
  if (vIdx >= pParallelElements) {
    return;
  }

  const int cIHw = pIh * pIw;
  const int cOHw = pOh * pOw;
  const int cNc = vIdx / cOHw;
  const int cCurOh = vIdx % cOHw / pOw;
  const int cCurOw = vIdx % cOHw % pOw;
  const Ti *vIPtr = pSrc + cNc * cIHw;
  To *vOPtr = pDst + cNc * cOHw;
  const int cStartH = cCurOh * pSh - pPh;
  const int cBh = max(0, cStartH);
  const int cEh = min(pIh, cStartH + pKh);
  const int cStartW = cCurOw * pSw - pPw;
  const int cBw = max(0, cStartW);
  const int cEw = min(pIw, cStartW + pKw);
  const To cScale = 1. / (pKh * pKw);
  To vRes = 0;

  switch (pOp) {
  case GGML_OP_POOL_AVG:
    vRes = 0;
    break;
  case GGML_OP_POOL_MAX:
    vRes = -FLT_MAX;
    break;
  default:
    assert(false);
  }

  for (int vI = cBh; vI < cEh; vI += 1) {
    for (int vJ = cBw; vJ < cEw; vJ += 1) {
#if __CUDA_ARCH__ >= 350
      Ti vCur = __ldg(vIPtr + vI * pIw + vJ);
#else
      Ti vCur = vIPtr[vI * pIw + vJ];
#endif
      switch (pOp) {
      case GGML_OP_POOL_AVG:
        vRes += vCur * cScale;
        break;
      case GGML_OP_POOL_MAX:
        vRes = max(vRes, (To)vCur);
        break;
      default:
        assert(false);
      }
    }
  }
  vOPtr[cCurOh * pOw + cCurOw] = vRes;
}

static void fPool2dNchwKernelF32F32Cuda(const int pIh, const int pIw, const int pOh, const int pOw, const int pKh,
                                        const int pKw, const int pSh, const int pSw, const int pPh, const int pPw,
                                        const int pParallelElements, const float *pSrc, float *pDst,
                                        const enum ggml_op_pool pOp, cudaStream_t pStream) {

  const int cNumBlocks = (pParallelElements + CUDA_POOL2D_BLOCK_SIZE - 1) / CUDA_POOL2D_BLOCK_SIZE;
  dim3 vBlockNums(cNumBlocks);
  fPool2dNchwKernel<<<vBlockNums, CUDA_POOL2D_BLOCK_SIZE, 0, pStream>>>(pIh, pIw, pOh, pOw, pKh, pKw, pSh, pSw, pPh,
                                                                        pPw, pParallelElements, pSrc, pDst, pOp);
}

void ggml_cuda_op_pool2d(ggml_backend_cuda_context &pCtx, ggml_tensor *pDst) {
  const ggml_tensor *vSrc0 = pDst->src[0];
  const float *vSrc0D = (const float *)vSrc0->data;
  float *vDstD = (float *)pDst->data;
  cudaStream_t vStream = pCtx.stream();

  GGML_ASSERT(vSrc0->type == GGML_TYPE_F32);
  GGML_ASSERT(pDst->type == GGML_TYPE_F32);

  const int32_t *vOpts = (const int32_t *)pDst->op_params;
  enum ggml_op_pool vOp = static_cast<ggml_op_pool>(vOpts[0]);
  const int cK0 = vOpts[1];
  const int cK1 = vOpts[2];
  const int cS0 = vOpts[3];
  const int cS1 = vOpts[4];
  const int cP0 = vOpts[5];
  const int cP1 = vOpts[6];

  const int64_t cIh = vSrc0->ne[1];
  const int64_t cIw = vSrc0->ne[0];

  const int64_t cN = pDst->ne[3];
  const int64_t cOc = pDst->ne[2];
  const int64_t cOh = pDst->ne[1];
  const int64_t cOw = pDst->ne[0];

  const int cParallelElements = cN * cOc * cOh * cOw;

  fPool2dNchwKernelF32F32Cuda(cIh, cIw, cOh, cOw, cK1, cK0, cS1, cS0, cP1, cP0, cParallelElements, vSrc0D, vDstD, vOp,
                              vStream);
}
