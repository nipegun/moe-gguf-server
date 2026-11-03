#include "concat.cuh"

#include <stdint.h>

// contiguous kernels
template <typename T, int dim>
static __global__ void __launch_bounds__(CUDA_CONCAT_BLOCK_SIZE)
    fConcatCont(const T *pX, const T *pY, T *pDst, int64_t pNe00, int64_t pNe01, int64_t pNe02, int64_t pNe0,
                int64_t pNe1, int64_t pNe2) {
  static_assert(dim >= 0 && dim <= 2, "dim must be in [0, 2]");

  const int64_t cN = pNe0 * pNe1 * pNe2;

  ggml_cuda_pdl_sync();
  for (int64_t vI = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; vI < cN; vI += (int64_t)blockDim.x * gridDim.x) {
    if constexpr (dim == 0) {
      const int64_t row = vI / pNe0;
      const int64_t i0 = vI - row * pNe0;

      if (i0 < pNe00) {
        pDst[vI] = pX[row * pNe00 + i0];
      } else {
        pDst[vI] = pY[row * (pNe0 - pNe00) + (i0 - pNe00)];
      }
    } else if constexpr (dim == 1) {
      const int64_t dst_plane = pNe0 * pNe1;
      const int64_t src0_plane = pNe0 * pNe01;
      const int64_t src1_plane = dst_plane - src0_plane;
      const int64_t i2 = vI / dst_plane;
      const int64_t i01 = vI - i2 * dst_plane;

      if (i01 < src0_plane) {
        pDst[vI] = pX[i2 * src0_plane + i01];
      } else {
        pDst[vI] = pY[i2 * src1_plane + (i01 - src0_plane)];
      }
    } else {
      const int64_t cSrc0Size = pNe0 * pNe1 * pNe02;

      if (vI < cSrc0Size) {
        pDst[vI] = pX[vI];
      } else {
        pDst[vI] = pY[vI - cSrc0Size];
      }
    }
  }
}

template <typename T>
static void fConcatContCuda(const T *pX, const T *pY, T *pDst, int64_t pNe00, int64_t pNe01, int64_t pNe02,
                            int64_t pNe0, int64_t pNe1, int64_t pNe2, int pDim, cudaStream_t pStream) {
  const int64_t cN = pNe0 * pNe1 * pNe2;
  const int cNumBlocks = (cN + CUDA_CONCAT_BLOCK_SIZE - 1) / CUDA_CONCAT_BLOCK_SIZE;

  if (pDim == 0) {
    const ggml_cuda_kernel_launch_params cLaunchParams =
        ggml_cuda_kernel_launch_params(cNumBlocks, CUDA_CONCAT_BLOCK_SIZE, 0, pStream);
    ggml_cuda_kernel_launch(fConcatCont<T, 0>, cLaunchParams, pX, pY, pDst, pNe00, pNe01, pNe02, pNe0, pNe1, pNe2);
    return;
  }
  if (pDim == 1) {
    fConcatCont<T, 1>
        <<<cNumBlocks, CUDA_CONCAT_BLOCK_SIZE, 0, pStream>>>(pX, pY, pDst, pNe00, pNe01, pNe02, pNe0, pNe1, pNe2);
    return;
  }
  fConcatCont<T, 2>
      <<<cNumBlocks, CUDA_CONCAT_BLOCK_SIZE, 0, pStream>>>(pX, pY, pDst, pNe00, pNe01, pNe02, pNe0, pNe1, pNe2);
}

// non-contiguous kernel (slow)
template <typename T, int dim>
static __global__ void __launch_bounds__(CUDA_CONCAT_BLOCK_SIZE)
    concat_non_cont(const char *pSrc0, const char *pSrc1, char *pDst, int64_t pNe00, int64_t pNe01, int64_t pNe02,
                    int64_t pNe03, uint64_t pNb00, uint64_t pNb01, uint64_t pNb02, uint64_t pNb03, int64_t /*ne10*/,
                    int64_t /*ne11*/, int64_t /*ne12*/, int64_t /*ne13*/, uint64_t pNb10, uint64_t pNb11,
                    uint64_t pNb12, uint64_t pNb13, int64_t pNe0, int64_t /*ne1*/, int64_t /*ne2*/, int64_t /*ne3*/,
                    uint64_t pNb0, uint64_t pNb1, uint64_t pNb2, uint64_t pNb3) {
  static_assert(dim >= 0 && dim <= 3, "dim must be in [0, 3]");

  const int64_t cI3 = blockIdx.z;
  const int64_t cI2 = blockIdx.y;
  const int64_t cI1 = blockIdx.x;

  const T *vX;

  for (int64_t vI0 = threadIdx.x; vI0 < pNe0; vI0 += blockDim.x) {
    if (vI0 < pNe00 && cI1 < pNe01 && cI2 < pNe02 && cI3 < pNe03) {
      vX = (const T *)(pSrc0 + cI3 * pNb03 + cI2 * pNb02 + cI1 * pNb01 + vI0 * pNb00);
    } else {
      if constexpr (dim == 0) {
        vX = (const T *)(pSrc1 + cI3 * pNb13 + cI2 * pNb12 + cI1 * pNb11 + (vI0 - pNe00) * pNb10);
      } else if constexpr (dim == 1) {
        vX = (const T *)(pSrc1 + cI3 * pNb13 + cI2 * pNb12 + (cI1 - pNe01) * pNb11 + vI0 * pNb10);
      } else if constexpr (dim == 2) {
        vX = (const T *)(pSrc1 + cI3 * pNb13 + (cI2 - pNe02) * pNb12 + cI1 * pNb11 + vI0 * pNb10);
      } else if constexpr (dim == 3) {
        vX = (const T *)(pSrc1 + (cI3 - pNe03) * pNb13 + cI2 * pNb12 + cI1 * pNb11 + vI0 * pNb10);
      }
    }

    T *vY = (T *)(pDst + cI3 * pNb3 + cI2 * pNb2 + cI1 * pNb1 + vI0 * pNb0);

    *vY = *vX;
  }
}

template <typename T>
static void fConcatCuda(const ggml_tensor *pSrc0, const ggml_tensor *pSrc1, ggml_tensor *pDst, int pDim,
                        cudaStream_t pStream) {
  if (pDim != 3 && ggml_is_contiguous_to_3(pSrc0) && ggml_is_contiguous_to_3(pSrc1)) {
    const T *vSrc0D = (const T *)pSrc0->data;
    const T *vSrc1D = (const T *)pSrc1->data;
    T *vDstD = (T *)pDst->data;

    for (int64_t vI3 = 0; vI3 < pDst->ne[3]; vI3++) {
      fConcatContCuda(vSrc0D + vI3 * (pSrc0->nb[3] / sizeof(T)), vSrc1D + vI3 * (pSrc1->nb[3] / sizeof(T)),
                      vDstD + vI3 * (pDst->nb[3] / sizeof(T)), ggml_row_size(pSrc0->type, pSrc0->ne[0]) / sizeof(T),
                      pSrc0->ne[1], pSrc0->ne[2], ggml_row_size(pDst->type, pDst->ne[0]) / sizeof(T), pDst->ne[1],
                      pDst->ne[2], pDim, pStream);
    }
  } else if (pDim == 3 && ggml_is_contiguous(pSrc0) && ggml_is_contiguous(pSrc1)) {
    const size_t cSize0 = ggml_nbytes(pSrc0);
    const size_t cSize1 = ggml_nbytes(pSrc1);

    CUDA_CHECK(cudaMemcpyAsync((char *)pDst->data, pSrc0->data, cSize0, cudaMemcpyDeviceToDevice, pStream));
    CUDA_CHECK(cudaMemcpyAsync((char *)pDst->data + cSize0, pSrc1->data, cSize1, cudaMemcpyDeviceToDevice, pStream));
  } else {
    GGML_ASSERT(!ggml_is_quantized(pSrc0->type));

    dim3 vGridDim(pDst->ne[1], pDst->ne[2], pDst->ne[3]);
    auto vLaunchKernel = [&](auto pDim) {
      concat_non_cont<T, pDim><<<vGridDim, CUDA_CONCAT_BLOCK_SIZE, 0, pStream>>>(
          (const char *)pSrc0->data, (const char *)pSrc1->data, (char *)pDst->data, pSrc0->ne[0], pSrc0->ne[1],
          pSrc0->ne[2], pSrc0->ne[3], pSrc0->nb[0], pSrc0->nb[1], pSrc0->nb[2], pSrc0->nb[3], pSrc1->ne[0],
          pSrc1->ne[1], pSrc1->ne[2], pSrc1->ne[3], pSrc1->nb[0], pSrc1->nb[1], pSrc1->nb[2], pSrc1->nb[3], pDst->ne[0],
          pDst->ne[1], pDst->ne[2], pDst->ne[3], pDst->nb[0], pDst->nb[1], pDst->nb[2], pDst->nb[3]);
    };
    switch (pDim) {
    case 0:
      vLaunchKernel(std::integral_constant<int, 0>{});
      break;
    case 1:
      vLaunchKernel(std::integral_constant<int, 1>{});
      break;
    case 2:
      vLaunchKernel(std::integral_constant<int, 2>{});
      break;
    case 3:
      vLaunchKernel(std::integral_constant<int, 3>{});
      break;
    default:
      GGML_ABORT("Invalid dim: %d", pDim);
      break;
    }
  }
}

void ggml_cuda_op_concat(ggml_backend_cuda_context &pCtx, ggml_tensor *pDst) {
  const ggml_tensor *vSrc0 = pDst->src[0];
  const ggml_tensor *vSrc1 = pDst->src[1];

  cudaStream_t vStream = pCtx.stream();

  const int32_t cDim = ((int32_t *)pDst->op_params)[0];

  GGML_ASSERT(vSrc0->type == vSrc1->type);
  GGML_ASSERT(pDst->type == vSrc0->type);

  if (ggml_is_quantized(vSrc0->type)) {
    if (cDim == 3) {
      GGML_ASSERT(ggml_is_contiguous(vSrc0));
      GGML_ASSERT(ggml_is_contiguous(vSrc1));
    } else {
      GGML_ASSERT(ggml_is_contiguous_to_3(vSrc0));
      GGML_ASSERT(ggml_is_contiguous_to_3(vSrc1));
    }
    GGML_ASSERT(vSrc0->ne[0] % ggml_blck_size(vSrc0->type) == 0);
    GGML_ASSERT(vSrc1->ne[0] % ggml_blck_size(vSrc1->type) == 0);

    // if first 3 dimensions are contiguous and ne[0] is multiple of the block size we can concat both tensors as byte
    // tensors
    fConcatCuda<uint8_t>(vSrc0, vSrc1, pDst, cDim, vStream);
  } else {
    GGML_ASSERT(ggml_blck_size(vSrc0->type) == 1);

    switch (ggml_type_size(vSrc0->type)) {
    case 1:
      fConcatCuda<uint8_t>(vSrc0, vSrc1, pDst, cDim, vStream);
      break;
    case 2:
      fConcatCuda<uint16_t>(vSrc0, vSrc1, pDst, cDim, vStream);
      break;
    case 4:
      fConcatCuda<uint32_t>(vSrc0, vSrc1, pDst, cDim, vStream);
      break;
    case 8:
      fConcatCuda<uint64_t>(vSrc0, vSrc1, pDst, cDim, vStream);
      break;
    default:
      GGML_ABORT("Unsupported type size: %zu", ggml_type_size(vSrc0->type));
      break;
    }
  }
}
