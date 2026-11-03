#include "conv2d-dw.cuh"

struct conv_params {
  int vInW, vInH;
  int vOutW, vOutH;
  int vKernelW, vKernelH;
  int vStrideX, vStrideY;
  int vPaddingX, vPaddingY;
  int vDilationX, vDilationY;
  int vChannels, vBatches;
};

struct kernel_bounds {
  int vYMin, vYMax;
  int vXMin, vXMax;
};

__device__ __forceinline__ kernel_bounds fCalculateKernelBounds(int pOutX, int pOutY, const conv_params &pParams) {
  kernel_bounds vBounds;
  vBounds.vYMin = max(0, (pParams.vPaddingY - pOutY * pParams.vStrideY + pParams.vDilationY - 1) / pParams.vDilationY);
  vBounds.vYMax =
      min(pParams.vKernelH,
          (pParams.vInH + pParams.vPaddingY - pOutY * pParams.vStrideY + pParams.vDilationY - 1) / pParams.vDilationY);
  vBounds.vXMin = max(0, (pParams.vPaddingX - pOutX * pParams.vStrideX + pParams.vDilationX - 1) / pParams.vDilationX);
  vBounds.vXMax =
      min(pParams.vKernelW,
          (pParams.vInW + pParams.vPaddingX - pOutX * pParams.vStrideX + pParams.vDilationX - 1) / pParams.vDilationX);
  return vBounds;
}

__device__ __forceinline__ int fCalculateInputCoord(int pOutCoord, int pKernCoord, int pStride, int pDilation,
                                                    int pPadding) {
  return pOutCoord * pStride + pKernCoord * pDilation - pPadding;
}

struct whcn_layout {
  __device__ static int input_index(int pN, int pC, int pY, int pX, const conv_params &pParams) {
    return pN * (pParams.vChannels * pParams.vInW * pParams.vInH) + pC * pParams.vInW * pParams.vInH +
           pY * pParams.vInW + pX;
  }

  __device__ static int kernel_index(int pC, int pKy, int pKx, const conv_params &pParams) {
    return pC * pParams.vKernelH * pParams.vKernelW + pKy * pParams.vKernelW + pKx;
  }

  __device__ static int output_index(int pN, int pC, int pY, int pX, const conv_params &pParams) {
    return pN * (pParams.vChannels * pParams.vOutW * pParams.vOutH) + pC * pParams.vOutW * pParams.vOutH +
           pY * pParams.vOutW + pX;
  }

  __device__ static void unpack_indices(int pGlobalIdx, const conv_params &pParams, int &pN, int &pC, int &pOutY,
                                        int &pOutX) {
    pOutX = pGlobalIdx % pParams.vOutW;
    pOutY = (pGlobalIdx / pParams.vOutW) % pParams.vOutH;
    pC = (pGlobalIdx / (pParams.vOutW * pParams.vOutH)) % pParams.vChannels;
    pN = pGlobalIdx / (pParams.vOutW * pParams.vOutH * pParams.vChannels);
  }
};

struct cwhn_layout {
  __device__ static int input_index(int pN, int pC, int pY, int pX, const conv_params &pParams) {
    return pN * (pParams.vChannels * pParams.vInW * pParams.vInH) + (pY * pParams.vInW + pX) * pParams.vChannels + pC;
  }

  __device__ static int kernel_index(int pC, int pKy, int pKx, const conv_params &pParams) {
    return (pKy * pParams.vKernelW + pKx) * pParams.vChannels + pC;
  }

  __device__ static int output_index(int pN, int pC, int pY, int pX, const conv_params &pParams) {
    return pN * (pParams.vChannels * pParams.vOutW * pParams.vOutH) + pY * (pParams.vOutW * pParams.vChannels) +
           pX * pParams.vChannels + pC;
  }

  __device__ static void unpack_indices(int pGlobalIdx, const conv_params &pParams, int &pN, int &pC, int &pOutY,
                                        int &pOutX) {
    pC = pGlobalIdx % pParams.vChannels;
    pOutX = (pGlobalIdx / pParams.vChannels) % pParams.vOutW;
    pOutY = (pGlobalIdx / (pParams.vChannels * pParams.vOutW)) % pParams.vOutH;
    pN = pGlobalIdx / (pParams.vChannels * pParams.vOutW * pParams.vOutH);
  }
};

template <typename T, typename Layout>
__global__ void fConv2dDwKernel(const T *__restrict__ pInput, const T *__restrict__ pKernel, T *__restrict__ pOutput,
                                const int pInW, const int pInH, const int pOutW, const int pOutH, const int pKernelW,
                                const int pKernelH, const int pStrideX, const int pStrideY, const int pPaddingX,
                                const int pPaddingY, const int pDilationX, const int pDilationY, const int pChannels,
                                const int pBatches) {
  const int cGlobalIdx = blockIdx.x * blockDim.x + threadIdx.x;
  const int cTotalElements = pBatches * pChannels * pOutH * pOutW;

  if (cGlobalIdx >= cTotalElements) {
    return;
  }

  conv_params vParams = {pInW,     pInH,      pOutW,     pOutH,      pKernelW,   pKernelH,  pStrideX,
                         pStrideY, pPaddingX, pPaddingY, pDilationX, pDilationY, pChannels, pBatches};

  int vBatchIdx, vChannelIdx, vOutYIdx, vOutXIdx;
  Layout::unpack_indices(cGlobalIdx, vParams, vBatchIdx, vChannelIdx, vOutYIdx, vOutXIdx);

  T vAccumulator = 0;
  kernel_bounds vBounds = fCalculateKernelBounds(vOutXIdx, vOutYIdx, vParams);

  for (int vKernY = vBounds.vYMin; vKernY < vBounds.vYMax; ++vKernY) {
    int vInYIdx = fCalculateInputCoord(vOutYIdx, vKernY, vParams.vStrideY, vParams.vDilationY, vParams.vPaddingY);

    for (int vKernX = vBounds.vXMin; vKernX < vBounds.vXMax; ++vKernX) {
      int vInXIdx = fCalculateInputCoord(vOutXIdx, vKernX, vParams.vStrideX, vParams.vDilationX, vParams.vPaddingX);

      const T cInputVal = pInput[Layout::input_index(vBatchIdx, vChannelIdx, vInYIdx, vInXIdx, vParams)];
      const T cKernelVal = pKernel[Layout::kernel_index(vChannelIdx, vKernY, vKernX, vParams)];

      vAccumulator += cInputVal * cKernelVal;
    }
  }

  pOutput[Layout::output_index(vBatchIdx, vChannelIdx, vOutYIdx, vOutXIdx, vParams)] = vAccumulator;
}

void ggml_cuda_op_conv2d_dw(ggml_backend_cuda_context &pCtx, ggml_tensor *pDst) {
  const ggml_tensor *vKernel = pDst->src[0];
  const ggml_tensor *vInput = pDst->src[1];

  GGML_ASSERT(vKernel->type == GGML_TYPE_F32 && vInput->type == GGML_TYPE_F32 && pDst->type == GGML_TYPE_F32);
  const float *vWD = (const float *)vKernel->data;
  const float *vXD = (const float *)vInput->data;
  float *vYD = (float *)pDst->data;

  const int32_t *vP = (const int32_t *)pDst->op_params;
  const int cStrideX = vP[0];
  const int cStrideY = vP[1];
  const int cPaddingX = vP[2];
  const int cPaddingY = vP[3];
  const int cDilationX = vP[4];
  const int cDilationY = vP[5];

  const int cInW = vInput->ne[0];
  const int cInH = vInput->ne[1];
  const int cKernelW = vKernel->ne[0];
  const int cKernelH = vKernel->ne[1];
  const int cOutW = pDst->ne[0];
  const int cOutH = pDst->ne[1];
  const int cChannels = pDst->ne[2];
  const int cBatches = pDst->ne[3];

  cudaStream_t vSt = pCtx.stream();

  const int cTotal = cBatches * cChannels * cOutH * cOutW;
  const int cBlocks = (cTotal + CUDA_CONV2D_DW_BLOCK_SIZE - 1) / CUDA_CONV2D_DW_BLOCK_SIZE;

  if (ggml_is_contiguous(vInput)) {
    fConv2dDwKernel<float, whcn_layout><<<cBlocks, CUDA_CONV2D_DW_BLOCK_SIZE, 0, vSt>>>(
        vXD, vWD, vYD, cInW, cInH, cOutW, cOutH, cKernelW, cKernelH, cStrideX, cStrideY, cPaddingX, cPaddingY,
        cDilationX, cDilationY, cChannels, cBatches);
  } else if (ggml_is_contiguous_channels(vInput)) {
    fConv2dDwKernel<float, cwhn_layout><<<cBlocks, CUDA_CONV2D_DW_BLOCK_SIZE, 0, vSt>>>(
        vXD, vWD, vYD, cInW, cInH, cOutW, cOutH, cKernelW, cKernelH, cStrideX, cStrideY, cPaddingX, cPaddingY,
        cDilationX, cDilationY, cChannels, cBatches);
  } else {
    GGML_ABORT("Unsupported memory layout for conv_2d_dw");
  }
}
