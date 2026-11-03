#include "conv2d-transpose.cuh"
#include "convert.cuh"

template <typename kernel_t>
static __global__ void fConv2dTransposeKernel(const float *__restrict__ pInput, const kernel_t *__restrict__ pKernel,
                                              float *__restrict__ pOutput, const int pInW, const int pInH,
                                              const int pOutW, const int pOutH, const int pKernelW, const int pKernelH,
                                              const int pStride, const int pCIn, const int pCOut, const int pBatches) {
  const int cGlobalIdx = blockIdx.x * blockDim.x + threadIdx.x;

  const int cTotalElements = pOutW * pOutH * pCOut * pBatches;

  if (cGlobalIdx >= cTotalElements) {
    return;
  }

  const int cOutXIdx = cGlobalIdx % pOutW;
  const int cOutYIdx = (cGlobalIdx / pOutW) % pOutH;
  const int cCIdx = (cGlobalIdx / (pOutW * pOutH)) % pCOut;
  const int cNIdx = cGlobalIdx / (pOutW * pOutH * pCOut);

  float vAccumulator = 0;
  // For each output idx, find the inputs that contribute to it by checking stride alignment and bounds

  for (int vCInIdx = 0; vCInIdx < pCIn; vCInIdx++) {
    for (int vKh = 0; vKh < pKernelH; ++vKh) {
      int vInY = cOutYIdx - vKh;
      if (vInY < 0 || vInY % pStride) {
        continue;
      }
      vInY /= pStride;
      if (vInY >= pInH) {
        continue;
      }

      for (int vKw = 0; vKw < pKernelW; ++vKw) {
        int vInX = cOutXIdx - vKw;
        if (vInX < 0 || vInX % pStride) {
          continue;
        }
        vInX /= pStride;
        if (vInX >= pInW) {
          continue;
        }

        const int cInputIdx = (pInW * pInH * pCIn) * cNIdx + (pInW * pInH) * vCInIdx + (pInW)*vInY + vInX;
        const int cKernelIdx =
            (pKernelH * pKernelW * pCOut) * vCInIdx + (pKernelH * pKernelW) * cCIdx + (pKernelW)*vKh + vKw;

        float vInputVal = pInput[cInputIdx];
        kernel_t vKernVal = pKernel[cKernelIdx];

        vAccumulator += vInputVal * ggml_cuda_cast<float>(vKernVal);
      }
    }
  }

  pOutput[(pOutW * pOutH * pCOut) * cNIdx + (pOutW * pOutH) * cCIdx + (pOutW)*cOutYIdx + cOutXIdx] = vAccumulator;
}

// input is (W, H, C_in, N), Kernel is (W, H, C_out, C_in)
void ggml_cuda_conv_2d_transpose_p0(ggml_backend_cuda_context &pCtx, ggml_tensor *pDst) {
  const ggml_tensor *vKernel = pDst->src[0];
  const ggml_tensor *vInput = pDst->src[1];

  GGML_ASSERT(vKernel->type == GGML_TYPE_F16 || vKernel->type == GGML_TYPE_F32);
  GGML_ASSERT(vInput->type == GGML_TYPE_F32 && pDst->type == GGML_TYPE_F32);

  const float *vInputData = (const float *)vInput->data;
  float *vOutputData = (float *)pDst->data;
  const void *vKernelData = vKernel->data;

  const int cInputW = vInput->ne[0];
  const int cInputH = vInput->ne[1];
  const int cOutputW = pDst->ne[0];
  const int cOutputH = pDst->ne[1];
  const int cChannelsIn = vInput->ne[2];
  const int cChannelsOut = vKernel->ne[2];
  const int cKernelW = vKernel->ne[0];
  const int cKernelH = vKernel->ne[1];
  const int cStride = pDst->op_params[0];
  const int cBatches = vInput->ne[3];

  GGML_ASSERT(cChannelsIn == vKernel->ne[3]);
  GGML_ASSERT(cStride > 0);

  cudaStream_t vSt = pCtx.stream();

  GGML_ASSERT(ggml_is_contiguous(vInput));
  GGML_ASSERT(ggml_is_contiguous(vKernel));
  GGML_ASSERT(ggml_is_contiguous(pDst));

  const int cTotal = cOutputW * cOutputH * cChannelsOut * cBatches;
  const int cBlocks = (cTotal + CUDA_CONV2D_TRANSPOSE_BLOCK_SIZE - 1) / CUDA_CONV2D_TRANSPOSE_BLOCK_SIZE;

  if (vKernel->type == GGML_TYPE_F16) {
    fConv2dTransposeKernel<half><<<cBlocks, CUDA_CONV2D_TRANSPOSE_BLOCK_SIZE, 0, vSt>>>(
        vInputData, (const half *)vKernelData, vOutputData, cInputW, cInputH, cOutputW, cOutputH, cKernelW, cKernelH,
        cStride, cChannelsIn, cChannelsOut, cBatches);

  } else {
    fConv2dTransposeKernel<float><<<cBlocks, CUDA_CONV2D_TRANSPOSE_BLOCK_SIZE, 0, vSt>>>(
        vInputData, (const float *)vKernelData, vOutputData, cInputW, cInputH, cOutputW, cOutputH, cKernelW, cKernelH,
        cStride, cChannelsIn, cChannelsOut, cBatches);
  }
}
