#include "conv2d.cuh"
#include "convert.cuh"

struct conv_params {
  const int64_t cIw, cIh;
  const int64_t cOw, cOh;
  const int64_t cKw, cKh;
  const int64_t cStX, cStY;
  const int64_t cPdX, cPdY;
  const int64_t cDlX, cDlY;
  const int64_t cIc, cOc;
  const int64_t cB;
  const int64_t cTotal;
};

struct kernel_bounds {
  int64_t vYMin, vYMax;
  int64_t vXMin, vXMax;
};

__device__ __forceinline__ int64_t fMax64(int64_t pA, int64_t pB) { return (pA > pB) ? pA : pB; }

__device__ __forceinline__ int64_t fMin64(int64_t pA, int64_t pB) { return (pA < pB) ? pA : pB; }

__device__ __forceinline__ kernel_bounds fCalculateKernelBounds(int64_t pOutX, int64_t pOutY, const conv_params &pP) {
  kernel_bounds vBounds;
  vBounds.vYMin = fMax64(0, (pP.cPdY - pOutY * pP.cStY + pP.cDlY - 1) / pP.cDlY);
  vBounds.vYMax = fMin64(pP.cKh, (pP.cIh + pP.cPdY - pOutY * pP.cStY + pP.cDlY - 1) / pP.cDlY);
  vBounds.vXMin = fMax64(0, (pP.cPdX - pOutX * pP.cStX + pP.cDlX - 1) / pP.cDlX);
  vBounds.vXMax = fMin64(pP.cKw, (pP.cIw + pP.cPdX - pOutX * pP.cStX + pP.cDlX - 1) / pP.cDlX);
  return vBounds;
}

__device__ __forceinline__ int fCalculateInputCoord(int64_t pOutCoord, int64_t pKernCoord, int64_t pStride,
                                                    int64_t pDilation, int64_t pPadding) {
  return pOutCoord * pStride + pKernCoord * pDilation - pPadding;
}

struct whcn_layout {
  __device__ static int64_t mInputIndex(int64_t pN, int64_t pC, int64_t pY, int64_t pX, const conv_params &pP) {
    return pN * (pP.cIc * pP.cIw * pP.cIh) + pC * pP.cIw * pP.cIh + pY * pP.cIw + pX;
  }

  __device__ static int64_t mKernelIndex(int64_t pCOut, int64_t pCIn, int64_t pKy, int64_t pKx, const conv_params &pP) {
    return pCOut * (pP.cIc * pP.cKh * pP.cKw) + pCIn * (pP.cKh * pP.cKw) + pKy * pP.cKw + pKx;
  }

  __device__ static int64_t mOutputIndex(int64_t pN, int64_t pC, int64_t pY, int64_t pX, const conv_params &pP) {
    return pN * (pP.cOc * pP.cOw * pP.cOh) + pC * pP.cOw * pP.cOh + pY * pP.cOw + pX;
  }

  __device__ static void mUnpackIndices(int64_t pGlobalIdx, const conv_params &pP, int64_t &pN, int64_t &pC,
                                        int64_t &pOutY, int64_t &pOutX) {
    pOutX = pGlobalIdx % pP.cOw;
    pOutY = (pGlobalIdx / pP.cOw) % pP.cOh;
    pC = (pGlobalIdx / (pP.cOw * pP.cOh)) % pP.cOc;
    pN = pGlobalIdx / (pP.cOw * pP.cOh * pP.cOc);
  }
};

template <typename T, typename Layout>
static __global__ void fConv2dKernel(const float *__restrict__ pInput, const T *__restrict__ pKernel,
                                     float *__restrict__ pOutput, const conv_params pP) {
  const int64_t cGlobalIdx = blockIdx.x * blockDim.x + threadIdx.x;

  if (cGlobalIdx >= pP.cTotal) {
    return;
  }

  int64_t vN, vCOut, vOutY, vOutX;
  Layout::mUnpackIndices(cGlobalIdx, pP, vN, vCOut, vOutY, vOutX);

  float vAcc = 0.0f;

  for (int64_t vCIn = 0; vCIn < pP.cIc; ++vCIn) {
    kernel_bounds vBounds = fCalculateKernelBounds(vOutX, vOutY, pP);

    for (int64_t vKy = vBounds.vYMin; vKy < vBounds.vYMax; ++vKy) {
      const int64_t cInY = fCalculateInputCoord(vOutY, vKy, pP.cStY, pP.cDlY, pP.cPdY);

      for (int64_t vKx = vBounds.vXMin; vKx < vBounds.vXMax; ++vKx) {
        const int64_t cInX = fCalculateInputCoord(vOutX, vKx, pP.cStX, pP.cDlX, pP.cPdX);

        const float cInputVal = pInput[Layout::mInputIndex(vN, vCIn, cInY, cInX, pP)];
        const T cKernelVal = pKernel[Layout::mKernelIndex(vCOut, vCIn, vKy, vKx, pP)];
        vAcc += (cInputVal * ggml_cuda_cast<float>(cKernelVal));
      }
    }
  }

  // [N, OC, OH, OW]
  pOutput[Layout::mOutputIndex(vN, vCOut, vOutY, vOutX, pP)] = vAcc;
}

template <typename T>
static void fConv2dCuda(const float *pXD, const T *pKD, float *pYD, const conv_params pP, cudaStream_t pSt) {
  const int cBlocks = (pP.cTotal + CUDA_CONV2D_BLOCK_SIZE - 1) / CUDA_CONV2D_BLOCK_SIZE;
  fConv2dKernel<T, whcn_layout><<<cBlocks, CUDA_CONV2D_BLOCK_SIZE, 0, pSt>>>(pXD, pKD, pYD, pP);
}

static void fConv2dCudaF16(const float *pXD, const half *pKD, float *pYD, const conv_params pP, cudaStream_t pSt) {
  fConv2dCuda<half>(pXD, pKD, pYD, pP, pSt);
}

static void fConv2dCudaF32(const float *pXD, const float *pKD, float *pYD, const conv_params pP, cudaStream_t pSt) {
  fConv2dCuda<float>(pXD, pKD, pYD, pP, pSt);
}

void ggml_cuda_op_conv2d(ggml_backend_cuda_context &pCtx, ggml_tensor *pDst) {
  const ggml_tensor *vKernel = pDst->src[0];
  const ggml_tensor *vInput = pDst->src[1];
  float *vKD = (float *)vKernel->data;
  const float *vXD = (const float *)vInput->data;
  float *vYD = (float *)pDst->data;

  GGML_ASSERT(ggml_is_contiguous(vInput));
  GGML_ASSERT(ggml_is_contiguous(vKernel));
  GGML_ASSERT(vKernel->type == GGML_TYPE_F16 || vKernel->type == GGML_TYPE_F32);

  // same number of input channels
  GGML_ASSERT(vInput->ne[2] == vKernel->ne[2]);

  cudaStream_t vSt = pCtx.stream();

  const int32_t *vP = (const int32_t *)pDst->op_params;
  const int cStX = vP[0]; // stride_x
  const int cStY = vP[1]; // stride_y
  const int cPdX = vP[2]; // padding_x
  const int cPdY = vP[3]; // padding_y
  const int cDlX = vP[4]; // dilation_x
  const int cDlY = vP[5]; // dilation_y

  // No cwhn
  GGML_ASSERT(vP[6] == false);

  const int cIw = vInput->ne[0];  // input_w
  const int cIh = vInput->ne[1];  // input_h
  const int cOw = pDst->ne[0];    // output_w
  const int cOh = pDst->ne[1];    // output_h
  const int cKw = vKernel->ne[0]; // kernel_w
  const int cKh = vKernel->ne[1]; // kernel_h
  const int cIc = vInput->ne[2];  // input_channels
  const int cOc = vKernel->ne[3]; // ouptut_chanles
  const int cB = vInput->ne[3];   // n_batches

  const int64_t cTotal = cB * cOc * cOh * cOw;
  conv_params vParams = {cIw, cIh, cOw, cOh, cKw, cKh, cStX, cStY, cPdX, cPdY, cDlX, cDlY, cIc, cOc, cB, cTotal};

  if (vKernel->type == GGML_TYPE_F16) {
    fConv2dCudaF16(vXD, (half *)vKD, vYD, vParams, vSt);
  } else {
    fConv2dCudaF32(vXD, vKD, vYD, vParams, vSt);
  }
}
