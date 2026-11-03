#include "cpy.cuh"
#include "dequantize.cuh"
#include "cpy-utils.cuh"

typedef void (*cpy_kernel_t)(const char *pCx, char *pCdst);

const int CUDA_CPY_TILE_DIM_2D = 32; // 2D tile dimension for transposed blocks
const int CUDA_CPY_BLOCK_NM = 8;     // block size of 3rd dimension if available
const int CUDA_CPY_BLOCK_ROWS = 8;   // block dimension for marching through rows

template <cpy_kernel_t cpy_1>
static __global__ void fCpyScalar(const char *pCx, char *pCdst, const int64_t pNe, const int64_t pNe00,
                                  const int64_t pNe01, const int64_t pNe02, const int64_t pNb00, const int64_t pNb01,
                                  const int64_t pNb02, const int64_t pNb03, const int64_t pNe10, const int64_t pNe11,
                                  const int64_t pNe12, const int64_t pNb10, const int64_t pNb11, const int64_t pNb12,
                                  const int64_t pNb13) {
  ggml_cuda_pdl_lc();
  const int64_t cI = (int64_t)blockDim.x * blockIdx.x + threadIdx.x;

  if (cI >= pNe) {
    return;
  }

  // determine indices i03/i13, i02/i12, i01/i11, i00/i10 as a function of index i of flattened tensor
  // then combine those indices with the corresponding byte offsets to get the total offsets
  const int64_t cI03 = cI / (pNe00 * pNe01 * pNe02);
  const int64_t cI02 = (cI - cI03 * pNe00 * pNe01 * pNe02) / (pNe00 * pNe01);
  const int64_t cI01 = (cI - cI03 * pNe00 * pNe01 * pNe02 - cI02 * pNe01 * pNe00) / pNe00;
  const int64_t cI00 = cI - cI03 * pNe00 * pNe01 * pNe02 - cI02 * pNe01 * pNe00 - cI01 * pNe00;
  const int64_t cXOffset = cI00 * pNb00 + cI01 * pNb01 + cI02 * pNb02 + cI03 * pNb03;

  const int64_t cI13 = cI / (pNe10 * pNe11 * pNe12);
  const int64_t cI12 = (cI - cI13 * pNe10 * pNe11 * pNe12) / (pNe10 * pNe11);
  const int64_t cI11 = (cI - cI13 * pNe10 * pNe11 * pNe12 - cI12 * pNe10 * pNe11) / pNe10;
  const int64_t cI10 = cI - cI13 * pNe10 * pNe11 * pNe12 - cI12 * pNe10 * pNe11 - cI11 * pNe10;
  const int64_t cDstOffset = cI10 * pNb10 + cI11 * pNb11 + cI12 * pNb12 + cI13 * pNb13;

  ggml_cuda_pdl_sync();
  cpy_1(pCx + cXOffset, pCdst + cDstOffset);
}

template <typename T>
static __global__ void
fCpyScalarTranspose(const char *pCx, char *pCdst, const int64_t pNe, const int64_t pNe00, const int64_t pNe01,
                    const int64_t pNe02, const int64_t pNb00, const int64_t pNb01, const int64_t pNb02,
                    const int64_t pNb03, const int64_t pNe10, const int64_t pNe11, const int64_t pNe12,
                    const int64_t pNb10, const int64_t pNb11, const int64_t pNb12, const int64_t pNb13) {

  const T *vSrc = reinterpret_cast<const T *>(pCx);
  T *vDst = reinterpret_cast<T *>(pCdst);

  const int64_t cNmat = pNe / (pNe00 * pNe01);
  const int64_t cN = pNe00 * pNe01;

  const int64_t cX = (int64_t)blockIdx.x * CUDA_CPY_TILE_DIM_2D + threadIdx.x;
  const int64_t cY = (int64_t)blockIdx.y * CUDA_CPY_TILE_DIM_2D + threadIdx.y;
  const int64_t cTx = (int64_t)blockIdx.y * CUDA_CPY_TILE_DIM_2D + threadIdx.x; // transpose block offset
  const int64_t cTy = (int64_t)blockIdx.x * CUDA_CPY_TILE_DIM_2D + threadIdx.y;

  __shared__ float vTile[2][CUDA_CPY_TILE_DIM_2D][CUDA_CPY_TILE_DIM_2D + 1];
  int vCurTileBuf = 0;

  ggml_cuda_pdl_sync();
#pragma unroll
  for (int vI = 0; vI < CUDA_CPY_BLOCK_NM; ++vI) {

    const unsigned int cImat = blockIdx.z * CUDA_CPY_BLOCK_NM + vI;
    if (cImat >= cNmat)
      break;

#pragma unroll
    for (int vJ = 0; vJ < CUDA_CPY_TILE_DIM_2D; vJ += CUDA_CPY_BLOCK_ROWS) {
      if (cX < pNe01 && cY + vJ < pNe00) {
        const int cRow = threadIdx.y + vJ;
        const int cCol = threadIdx.x * sizeof(float) / sizeof(T);
        T *vTile2 = reinterpret_cast<T *>(vTile[vCurTileBuf][cRow]);
        vTile2[cCol] = vSrc[cImat * cN + (cY + vJ) * pNe01 + cX];
      }
    }

    __syncthreads();

#pragma unroll
    for (int vJ = 0; vJ < CUDA_CPY_TILE_DIM_2D; vJ += CUDA_CPY_BLOCK_ROWS) {
      if (cTy + vJ < pNe01 && cTx < pNe00) {
        const int cCol = (threadIdx.y + vJ) * sizeof(float) / sizeof(T);
        const T *vTile2 = reinterpret_cast<const T *>(vTile[vCurTileBuf][threadIdx.x]);
        vDst[cImat * cN + (cTy + vJ) * pNe00 + cTx] = vTile2[cCol];
      }
    }

    vCurTileBuf = (vCurTileBuf + 1) % 2;
  }

  GGML_UNUSED_VARS(pNe02, pNb00, pNb01, pNb02, pNb03, pNe10, pNe11, pNe12, pNb10, pNb11, pNb12, pNb13);
}

static __device__ void fCpyBlckQ80F32(const char *pCxi, char *pCdsti) {
  float *vCdstf = (float *)(pCdsti);

#pragma unroll
  for (int vJ = 0; vJ < QK8_0; vJ += 2) {
    float2 vDq;
    dequantize_q8_0(pCxi, 0, vJ, vDq);
    *(vCdstf + vJ) = vDq.x;
    *(vCdstf + vJ + 1) = vDq.y;
  }
}

template <dequantize_kernel_t dequant, int qk> static __device__ void cpy_blck_q_f32(const char *pCxi, char *pCdsti) {
  float *vCdstf = (float *)(pCdsti);

#pragma unroll
  for (int vJ = 0; vJ < qk / 2; vJ++) {
    float2 vDq;
    dequant(pCxi, 0, vJ, vDq);
    *(vCdstf + vJ) = vDq.x;
    *(vCdstf + vJ + qk / 2) = vDq.y;
  }
}

template <cpy_kernel_t cpy_blck, int qk>
static __global__ void fCpyF32Q(const char *pCx, char *pCdst, const int64_t pNe, const int64_t pNe00,
                                const int64_t pNe01, const int64_t pNe02, const int64_t pNb00, const int64_t pNb01,
                                const int64_t pNb02, const int64_t pNb03, const int64_t pNe10, const int64_t pNe11,
                                const int64_t pNe12, const int64_t pNb10, const int64_t pNb11, const int64_t pNb12,
                                const int64_t pNb13) {
  const int64_t cI = ((int64_t)blockDim.x * blockIdx.x + threadIdx.x) * qk;

  if (cI >= pNe) {
    return;
  }

  const int64_t cI03 = cI / (pNe00 * pNe01 * pNe02);
  const int64_t cI02 = (cI - cI03 * pNe00 * pNe01 * pNe02) / (pNe00 * pNe01);
  const int64_t cI01 = (cI - cI03 * pNe00 * pNe01 * pNe02 - cI02 * pNe01 * pNe00) / pNe00;
  const int64_t cI00 = cI - cI03 * pNe00 * pNe01 * pNe02 - cI02 * pNe01 * pNe00 - cI01 * pNe00;
  const int64_t cXOffset = cI00 * pNb00 + cI01 * pNb01 + cI02 * pNb02 + cI03 * pNb03;

  const int64_t cI13 = cI / (pNe10 * pNe11 * pNe12);
  const int64_t cI12 = (cI - cI13 * pNe10 * pNe11 * pNe12) / (pNe10 * pNe11);
  const int64_t cI11 = (cI - cI13 * pNe10 * pNe11 * pNe12 - cI12 * pNe10 * pNe11) / pNe10;
  const int64_t cI10 = cI - cI13 * pNe10 * pNe11 * pNe12 - cI12 * pNe10 * pNe11 - cI11 * pNe10;
  const int64_t cDstOffset = (cI10 / qk) * pNb10 + cI11 * pNb11 + cI12 * pNb12 + cI13 * pNb13;

  ggml_cuda_pdl_sync();
  cpy_blck(pCx + cXOffset, pCdst + cDstOffset);
}

template <cpy_kernel_t cpy_blck, int qk>
static __global__ void fCpyQF32(const char *pCx, char *pCdst, const int64_t pNe, const int64_t pNe00,
                                const int64_t pNe01, const int64_t pNe02, const int64_t pNb00, const int64_t pNb01,
                                const int64_t pNb02, const int64_t pNb03, const int64_t pNe10, const int64_t pNe11,
                                const int64_t pNe12, const int64_t pNb10, const int64_t pNb11, const int64_t pNb12,
                                const int64_t pNb13) {
  const int64_t cI = ((int64_t)blockDim.x * blockIdx.x + threadIdx.x) * qk;

  if (cI >= pNe) {
    return;
  }

  const int64_t cI03 = cI / (pNe00 * pNe01 * pNe02);
  const int64_t cI02 = (cI - cI03 * pNe00 * pNe01 * pNe02) / (pNe00 * pNe01);
  const int64_t cI01 = (cI - cI03 * pNe00 * pNe01 * pNe02 - cI02 * pNe01 * pNe00) / pNe00;
  const int64_t cI00 = cI - cI03 * pNe00 * pNe01 * pNe02 - cI02 * pNe01 * pNe00 - cI01 * pNe00;
  const int64_t cXOffset = (cI00 / qk) * pNb00 + cI01 * pNb01 + cI02 * pNb02 + cI03 * pNb03;

  const int64_t cI13 = cI / (pNe10 * pNe11 * pNe12);
  const int64_t cI12 = (cI - cI13 * pNe10 * pNe11 * pNe12) / (pNe10 * pNe11);
  const int64_t cI11 = (cI - cI13 * pNe10 * pNe11 * pNe12 - cI12 * pNe10 * pNe11) / pNe10;
  const int64_t cI10 = cI - cI13 * pNe10 * pNe11 * pNe12 - cI12 * pNe10 * pNe11 - cI11 * pNe10;
  const int64_t cDstOffset = cI10 * pNb10 + cI11 * pNb11 + cI12 * pNb12 + cI13 * pNb13;

  ggml_cuda_pdl_sync();
  cpy_blck(pCx + cXOffset, pCdst + cDstOffset);
}

template <typename src_t, typename dst_t>
static __global__ void fCpyScalarContiguous(const char *pCx, char *pCdst, const int64_t pNe) {
  const int64_t cI = (int64_t)blockDim.x * blockIdx.x + threadIdx.x;

  if (cI >= pNe) {
    return;
  }

  const src_t *vX = (const src_t *)pCx;
  dst_t *vDst = (dst_t *)pCdst;

  ggml_cuda_pdl_sync();
  vDst[cI] = ggml_cuda_cast<dst_t>(vX[cI]);
}

template <typename src_t, typename dst_t>
static void fGgmlCpyScalarContiguousCuda(const char *pCx, char *pCdst, const int64_t pNe, cudaStream_t pStream) {

  const int64_t cNumBlocks = (pNe + CUDA_CPY_BLOCK_SIZE - 1) / CUDA_CPY_BLOCK_SIZE;
  GGML_ASSERT(cNumBlocks <= INT_MAX);
  const ggml_cuda_kernel_launch_params cLaunchParams =
      ggml_cuda_kernel_launch_params((dim3)cNumBlocks, CUDA_CPY_BLOCK_SIZE, 0, pStream);
  ggml_cuda_kernel_launch(fCpyScalarContiguous<src_t, dst_t>, cLaunchParams, pCx, pCdst, pNe);
}

template <typename src_t, typename dst_t, bool transposed = false>
static void fGgmlCpyScalarCuda(const char *pCx, char *pCdst, const int64_t pNe, const int64_t pNe00,
                               const int64_t pNe01, const int64_t pNe02, const int64_t pNb00, const int64_t pNb01,
                               const int64_t pNb02, const int64_t pNb03, const int64_t pNe10, const int64_t pNe11,
                               const int64_t pNe12, const int64_t pNb10, const int64_t pNb11, const int64_t pNb12,
                               const int64_t pNb13, cudaStream_t pStream) {

  const auto cLaunchScalarGeneric = [&]() {
    const int64_t cNumBlocks = (pNe + CUDA_CPY_BLOCK_SIZE - 1) / CUDA_CPY_BLOCK_SIZE;
    GGML_ASSERT(cNumBlocks <= INT_MAX);
    const ggml_cuda_kernel_launch_params cLaunchParams =
        ggml_cuda_kernel_launch_params((dim3)cNumBlocks, CUDA_CPY_BLOCK_SIZE, 0, pStream);
    ggml_cuda_kernel_launch(fCpyScalar<cpy_1_scalar<src_t, dst_t>>, cLaunchParams, pCx, pCdst, pNe, pNe00, pNe01, pNe02,
                            pNb00, pNb01, pNb02, pNb03, pNe10, pNe11, pNe12, pNb10, pNb11, pNb12, pNb13);
  };

  if (transposed) {
    GGML_ASSERT(pNe == pNe00 * pNe01 * pNe02); // ne[3] is 1 assumed
    int64_t vNe00n, vNe01n, vNe02n;
    if (pNb00 <= pNb02) { // most likely safe to handle nb00 = nb02 case here
      vNe00n = pNe00;
      vNe01n = pNe01;
      vNe02n = pNe02;
    } else {
      vNe00n = pNe00;
      vNe01n = pNe01 * pNe02;
      vNe02n = 1;
    }

    int64_t vGridX = (vNe01n + CUDA_CPY_TILE_DIM_2D - 1) / CUDA_CPY_TILE_DIM_2D;
    int64_t vGridY = (vNe00n + CUDA_CPY_TILE_DIM_2D - 1) / CUDA_CPY_TILE_DIM_2D;
    int64_t vGridZ = (pNe / (vNe01n * vNe00n) + CUDA_CPY_BLOCK_NM - 1) / CUDA_CPY_BLOCK_NM;
    GGML_ASSERT(vGridX <= INT_MAX);
    if (vGridY > USHRT_MAX || vGridZ > USHRT_MAX) {
      cLaunchScalarGeneric();
    } else {
      dim3 vDimGrid(vGridX, vGridY, vGridZ);
      dim3 vDimBlock(CUDA_CPY_TILE_DIM_2D, CUDA_CPY_BLOCK_ROWS, 1);
      const ggml_cuda_kernel_launch_params cLaunchParams =
          ggml_cuda_kernel_launch_params(vDimGrid, vDimBlock, 0, pStream);
      ggml_cuda_kernel_launch(fCpyScalarTranspose<dst_t>, cLaunchParams, pCx, pCdst, pNe, vNe00n, vNe01n, vNe02n, pNb00,
                              pNb01, pNb02, pNb03, pNe10, pNe11, pNe12, pNb10, pNb11, pNb12, pNb13);
    }
  } else {
    cLaunchScalarGeneric();
  }
}

static void fGgmlCpyF32Q80Cuda(const char *pCx, char *pCdst, const int64_t pNe, const int64_t pNe00,
                               const int64_t pNe01, const int64_t pNe02, const int64_t pNb00, const int64_t pNb01,
                               const int64_t pNb02, const int64_t pNb03, const int64_t pNe10, const int64_t pNe11,
                               const int64_t pNe12, const int64_t pNb10, const int64_t pNb11, const int64_t pNb12,
                               const int64_t pNb13, cudaStream_t pStream) {

  GGML_ASSERT(pNe % QK8_0 == 0);
  const int64_t cNumBlocks = pNe / QK8_0;
  GGML_ASSERT(cNumBlocks <= INT_MAX);
  fCpyF32Q<cpy_blck_f32_q8_0, QK8_0><<<cNumBlocks, 1, 0, pStream>>>(pCx, pCdst, pNe, pNe00, pNe01, pNe02, pNb00, pNb01,
                                                                    pNb02, pNb03, pNe10, pNe11, pNe12, pNb10, pNb11,
                                                                    pNb12, pNb13);
}

static void fGgmlCpyQ80F32Cuda(const char *pCx, char *pCdst, const int64_t pNe, const int64_t pNe00,
                               const int64_t pNe01, const int64_t pNe02, const int64_t pNb00, const int64_t pNb01,
                               const int64_t pNb02, const int64_t pNb03, const int64_t pNe10, const int64_t pNe11,
                               const int64_t pNe12, const int64_t pNb10, const int64_t pNb11, const int64_t pNb12,
                               const int64_t pNb13, cudaStream_t pStream) {

  const int64_t cNumBlocks = pNe;
  GGML_ASSERT(cNumBlocks <= INT_MAX);
  fCpyQF32<fCpyBlckQ80F32, QK8_0><<<cNumBlocks, 1, 0, pStream>>>(pCx, pCdst, pNe, pNe00, pNe01, pNe02, pNb00, pNb01,
                                                                 pNb02, pNb03, pNe10, pNe11, pNe12, pNb10, pNb11, pNb12,
                                                                 pNb13);
}

static void fGgmlCpyF32Q40Cuda(const char *pCx, char *pCdst, const int64_t pNe, const int64_t pNe00,
                               const int64_t pNe01, const int64_t pNe02, const int64_t pNb00, const int64_t pNb01,
                               const int64_t pNb02, const int64_t pNb03, const int64_t pNe10, const int64_t pNe11,
                               const int64_t pNe12, const int64_t pNb10, const int64_t pNb11, const int64_t pNb12,
                               const int64_t pNb13, cudaStream_t pStream) {

  GGML_ASSERT(pNe % QK4_0 == 0);
  const int64_t cNumBlocks = pNe / QK4_0;
  GGML_ASSERT(cNumBlocks <= INT_MAX);
  fCpyF32Q<cpy_blck_f32_q4_0, QK4_0><<<cNumBlocks, 1, 0, pStream>>>(pCx, pCdst, pNe, pNe00, pNe01, pNe02, pNb00, pNb01,
                                                                    pNb02, pNb03, pNe10, pNe11, pNe12, pNb10, pNb11,
                                                                    pNb12, pNb13);
}

static void fGgmlCpyQ40F32Cuda(const char *pCx, char *pCdst, const int64_t pNe, const int64_t pNe00,
                               const int64_t pNe01, const int64_t pNe02, const int64_t pNb00, const int64_t pNb01,
                               const int64_t pNb02, const int64_t pNb03, const int64_t pNe10, const int64_t pNe11,
                               const int64_t pNe12, const int64_t pNb10, const int64_t pNb11, const int64_t pNb12,
                               const int64_t pNb13, cudaStream_t pStream) {
  const int64_t cNumBlocks = pNe;
  GGML_ASSERT(cNumBlocks <= INT_MAX);
  fCpyQF32<cpy_blck_q_f32<dequantize_q4_0, QK4_0>, QK4_0>
      <<<cNumBlocks, 1, 0, pStream>>>(pCx, pCdst, pNe, pNe00, pNe01, pNe02, pNb00, pNb01, pNb02, pNb03, pNe10, pNe11,
                                      pNe12, pNb10, pNb11, pNb12, pNb13);
}

static void fGgmlCpyF32Q41Cuda(const char *pCx, char *pCdst, const int64_t pNe, const int64_t pNe00,
                               const int64_t pNe01, const int64_t pNe02, const int64_t pNb00, const int64_t pNb01,
                               const int64_t pNb02, const int64_t pNb03, const int64_t pNe10, const int64_t pNe11,
                               const int64_t pNe12, const int64_t pNb10, const int64_t pNb11, const int64_t pNb12,
                               const int64_t pNb13, cudaStream_t pStream) {

  GGML_ASSERT(pNe % QK4_1 == 0);
  const int64_t cNumBlocks = pNe / QK4_1;
  GGML_ASSERT(cNumBlocks <= INT_MAX);
  fCpyF32Q<cpy_blck_f32_q4_1, QK4_1><<<cNumBlocks, 1, 0, pStream>>>(pCx, pCdst, pNe, pNe00, pNe01, pNe02, pNb00, pNb01,
                                                                    pNb02, pNb03, pNe10, pNe11, pNe12, pNb10, pNb11,
                                                                    pNb12, pNb13);
}

static void fGgmlCpyQ41F32Cuda(const char *pCx, char *pCdst, const int64_t pNe, const int64_t pNe00,
                               const int64_t pNe01, const int64_t pNe02, const int64_t pNb00, const int64_t pNb01,
                               const int64_t pNb02, const int64_t pNb03, const int64_t pNe10, const int64_t pNe11,
                               const int64_t pNe12, const int64_t pNb10, const int64_t pNb11, const int64_t pNb12,
                               const int64_t pNb13, cudaStream_t pStream) {
  const int64_t cNumBlocks = pNe;
  GGML_ASSERT(cNumBlocks <= INT_MAX);
  fCpyQF32<cpy_blck_q_f32<dequantize_q4_1, QK4_1>, QK4_1>
      <<<cNumBlocks, 1, 0, pStream>>>(pCx, pCdst, pNe, pNe00, pNe01, pNe02, pNb00, pNb01, pNb02, pNb03, pNe10, pNe11,
                                      pNe12, pNb10, pNb11, pNb12, pNb13);
}

static void fGgmlCpyF32Q50Cuda(const char *pCx, char *pCdst, const int64_t pNe, const int64_t pNe00,
                               const int64_t pNe01, const int64_t pNe02, const int64_t pNb00, const int64_t pNb01,
                               const int64_t pNb02, const int64_t pNb03, const int64_t pNe10, const int64_t pNe11,
                               const int64_t pNe12, const int64_t pNb10, const int64_t pNb11, const int64_t pNb12,
                               const int64_t pNb13, cudaStream_t pStream) {

  GGML_ASSERT(pNe % QK5_0 == 0);
  const int64_t cNumBlocks = pNe / QK5_0;
  GGML_ASSERT(cNumBlocks <= INT_MAX);
  fCpyF32Q<cpy_blck_f32_q5_0, QK5_0><<<cNumBlocks, 1, 0, pStream>>>(pCx, pCdst, pNe, pNe00, pNe01, pNe02, pNb00, pNb01,
                                                                    pNb02, pNb03, pNe10, pNe11, pNe12, pNb10, pNb11,
                                                                    pNb12, pNb13);
}

static void fGgmlCpyQ50F32Cuda(const char *pCx, char *pCdst, const int64_t pNe, const int64_t pNe00,
                               const int64_t pNe01, const int64_t pNe02, const int64_t pNb00, const int64_t pNb01,
                               const int64_t pNb02, const int64_t pNb03, const int64_t pNe10, const int64_t pNe11,
                               const int64_t pNe12, const int64_t pNb10, const int64_t pNb11, const int64_t pNb12,
                               const int64_t pNb13, cudaStream_t pStream) {
  const int64_t cNumBlocks = pNe;
  GGML_ASSERT(cNumBlocks <= INT_MAX);
  fCpyQF32<cpy_blck_q_f32<dequantize_q5_0, QK5_0>, QK5_0>
      <<<cNumBlocks, 1, 0, pStream>>>(pCx, pCdst, pNe, pNe00, pNe01, pNe02, pNb00, pNb01, pNb02, pNb03, pNe10, pNe11,
                                      pNe12, pNb10, pNb11, pNb12, pNb13);
}

static void fGgmlCpyF32Q51Cuda(const char *pCx, char *pCdst, const int64_t pNe, const int64_t pNe00,
                               const int64_t pNe01, const int64_t pNe02, const int64_t pNb00, const int64_t pNb01,
                               const int64_t pNb02, const int64_t pNb03, const int64_t pNe10, const int64_t pNe11,
                               const int64_t pNe12, const int64_t pNb10, const int64_t pNb11, const int64_t pNb12,
                               const int64_t pNb13, cudaStream_t pStream) {

  GGML_ASSERT(pNe % QK5_1 == 0);
  const int64_t cNumBlocks = pNe / QK5_1;
  GGML_ASSERT(cNumBlocks <= INT_MAX);
  fCpyF32Q<cpy_blck_f32_q5_1, QK5_1><<<cNumBlocks, 1, 0, pStream>>>(pCx, pCdst, pNe, pNe00, pNe01, pNe02, pNb00, pNb01,
                                                                    pNb02, pNb03, pNe10, pNe11, pNe12, pNb10, pNb11,
                                                                    pNb12, pNb13);
}

static void fGgmlCpyQ51F32Cuda(const char *pCx, char *pCdst, const int64_t pNe, const int64_t pNe00,
                               const int64_t pNe01, const int64_t pNe02, const int64_t pNb00, const int64_t pNb01,
                               const int64_t pNb02, const int64_t pNb03, const int64_t pNe10, const int64_t pNe11,
                               const int64_t pNe12, const int64_t pNb10, const int64_t pNb11, const int64_t pNb12,
                               const int64_t pNb13, cudaStream_t pStream) {
  const int64_t cNumBlocks = pNe;
  GGML_ASSERT(cNumBlocks <= INT_MAX);
  fCpyQF32<cpy_blck_q_f32<dequantize_q5_1, QK5_1>, QK5_1>
      <<<cNumBlocks, 1, 0, pStream>>>(pCx, pCdst, pNe, pNe00, pNe01, pNe02, pNb00, pNb01, pNb02, pNb03, pNe10, pNe11,
                                      pNe12, pNb10, pNb11, pNb12, pNb13);
}

static void fGgmlCpyF32Iq4NlCuda(const char *pCx, char *pCdst, const int64_t pNe, const int64_t pNe00,
                                 const int64_t pNe01, const int64_t pNe02, const int64_t pNb00, const int64_t pNb01,
                                 const int64_t pNb02, const int64_t pNb03, const int64_t pNe10, const int64_t pNe11,
                                 const int64_t pNe12, const int64_t pNb10, const int64_t pNb11, const int64_t pNb12,
                                 const int64_t pNb13, cudaStream_t pStream) {

  GGML_ASSERT(pNe % QK4_NL == 0);
  const int64_t cNumBlocks = pNe / QK4_NL;
  GGML_ASSERT(cNumBlocks <= INT_MAX);
  fCpyF32Q<cpy_blck_f32_iq4_nl, QK4_NL><<<cNumBlocks, 1, 0, pStream>>>(pCx, pCdst, pNe, pNe00, pNe01, pNe02, pNb00,
                                                                       pNb01, pNb02, pNb03, pNe10, pNe11, pNe12, pNb10,
                                                                       pNb11, pNb12, pNb13);
}

// check if a same-type copy reduces to a 2D strided copy (height rows of width
// contiguous bytes), so it can use cudaMemcpy2DAsync instead of the scalar kernel
static bool fGgmlCudaCpyAsMemcpy2d(const ggml_tensor *pSrc0, const ggml_tensor *pSrc1, size_t &pWidth, size_t &pHeight,
                                   size_t &pSpitch, size_t &pDpitch) {
  // require matching shape: a reshaped copy maps elements by flat order, which the
  // prefix walk below does not handle
  if (pSrc0->type != pSrc1->type || !ggml_are_same_shape(pSrc0, pSrc1)) {
    return false;
  }

  // grow the contiguous prefix block shared by both tensors
  size_t vBlockNb = ggml_element_size(pSrc0);
  int vD = 0;
  for (; vD < GGML_MAX_DIMS; ++vD) {
    if (pSrc0->nb[vD] != vBlockNb || pSrc1->nb[vD] != vBlockNb) {
      break;
    }
    vBlockNb *= pSrc0->ne[vD];
  }

  // d == 0: nothing contiguous; d == GGML_MAX_DIMS: fully contiguous (handled by memcpy)
  if (vD == 0 || vD == GGML_MAX_DIMS) {
    return false;
  }

  // dim d carries the rows; everything above it must be a single element
  for (int vI = vD + 1; vI < GGML_MAX_DIMS; ++vI) {
    if (pSrc0->ne[vI] != 1) {
      return false;
    }
  }

  pWidth = vBlockNb;
  pHeight = pSrc0->ne[vD];
  pSpitch = pSrc0->nb[vD];
  pDpitch = pSrc1->nb[vD];

  return pSpitch >= pWidth && pDpitch >= pWidth;
}

void ggml_cuda_cpy(ggml_backend_cuda_context &pCtx, const ggml_tensor *pSrc0, ggml_tensor *pSrc1) {
  const int64_t cNe = ggml_nelements(pSrc0);
  GGML_ASSERT(cNe == ggml_nelements(pSrc1));

  const int64_t cNe00 = pSrc0->ne[0];
  const int64_t cNe01 = pSrc0->ne[1];
  const int64_t cNe02 = pSrc0->ne[2];

  // GGML_ASSERT(src0->ne[3] == 1);

  const int64_t cNb00 = pSrc0->nb[0];
  const int64_t cNb01 = pSrc0->nb[1];
  const int64_t cNb02 = pSrc0->nb[2];
  const int64_t cNb03 = pSrc0->nb[3];

  const int64_t cNe10 = pSrc1->ne[0];
  const int64_t cNe11 = pSrc1->ne[1];
  const int64_t cNe12 = pSrc1->ne[2];

  // GGML_ASSERT(src1->ne[3] == 1);

  const int64_t cNb10 = pSrc1->nb[0];
  const int64_t cNb11 = pSrc1->nb[1];
  const int64_t cNb12 = pSrc1->nb[2];
  const int64_t cNb13 = pSrc1->nb[3];

  cudaStream_t vMainStream = pCtx.stream();

  char *vSrc0Ddc = (char *)pSrc0->data;
  char *vSrc1Ddc = (char *)pSrc1->data;

  const bool cContiguousSrcs = ggml_is_contiguous(pSrc0) && ggml_is_contiguous(pSrc1);
  const bool cCanBeTransposed = cNb01 == (int64_t)ggml_element_size(pSrc0) && pSrc0->ne[3] == 1 &&
                                cNb02 == cNe00 * cNe01 * (int64_t)ggml_element_size(pSrc0);

  size_t vMcWidth = 0, vMcHeight = 0, vMcSpitch = 0, vMcDpitch = 0;

  if (pSrc0->type == pSrc1->type && cContiguousSrcs) {
    GGML_ASSERT(ggml_nbytes(pSrc0) == ggml_nbytes(pSrc1));
    {
      CUDA_CHECK(cudaMemcpyAsync(vSrc1Ddc, vSrc0Ddc, ggml_nbytes(pSrc0), cudaMemcpyDeviceToDevice, vMainStream));
    }
  } else if (fGgmlCudaCpyAsMemcpy2d(pSrc0, pSrc1, vMcWidth, vMcHeight, vMcSpitch, vMcDpitch)) {
    CUDA_CHECK(cudaMemcpy2DAsync(vSrc1Ddc, vMcDpitch, vSrc0Ddc, vMcSpitch, vMcWidth, vMcHeight,
                                 cudaMemcpyDeviceToDevice, vMainStream));
  } else if (pSrc0->type == GGML_TYPE_F32 && pSrc1->type == GGML_TYPE_F32) {
    if (cCanBeTransposed) {
      fGgmlCpyScalarCuda<float, float, true>(vSrc0Ddc, vSrc1Ddc, cNe, cNe00, cNe01, cNe02, cNb00, cNb01, cNb02, cNb03,
                                             cNe10, cNe11, cNe12, cNb10, cNb11, cNb12, cNb13, vMainStream);
    } else {
      fGgmlCpyScalarCuda<float, float>(vSrc0Ddc, vSrc1Ddc, cNe, cNe00, cNe01, cNe02, cNb00, cNb01, cNb02, cNb03, cNe10,
                                       cNe11, cNe12, cNb10, cNb11, cNb12, cNb13, vMainStream);
    }
  } else if (pSrc0->type == GGML_TYPE_F32 && pSrc1->type == GGML_TYPE_BF16) {
    if (cContiguousSrcs) {
      fGgmlCpyScalarContiguousCuda<float, nv_bfloat16>(vSrc0Ddc, vSrc1Ddc, cNe, vMainStream);
    } else {
      fGgmlCpyScalarCuda<float, nv_bfloat16>(vSrc0Ddc, vSrc1Ddc, cNe, cNe00, cNe01, cNe02, cNb00, cNb01, cNb02, cNb03,
                                             cNe10, cNe11, cNe12, cNb10, cNb11, cNb12, cNb13, vMainStream);
    }
  } else if (pSrc0->type == GGML_TYPE_F32 && pSrc1->type == GGML_TYPE_F16) {
    if (cContiguousSrcs) {
      fGgmlCpyScalarContiguousCuda<float, half>(vSrc0Ddc, vSrc1Ddc, cNe, vMainStream);
    } else {
      fGgmlCpyScalarCuda<float, half>(vSrc0Ddc, vSrc1Ddc, cNe, cNe00, cNe01, cNe02, cNb00, cNb01, cNb02, cNb03, cNe10,
                                      cNe11, cNe12, cNb10, cNb11, cNb12, cNb13, vMainStream);
    }
  } else if (pSrc0->type == GGML_TYPE_F32 && pSrc1->type == GGML_TYPE_Q8_0) {
    fGgmlCpyF32Q80Cuda(vSrc0Ddc, vSrc1Ddc, cNe, cNe00, cNe01, cNe02, cNb00, cNb01, cNb02, cNb03, cNe10, cNe11, cNe12,
                       cNb10, cNb11, cNb12, cNb13, vMainStream);
  } else if (pSrc0->type == GGML_TYPE_Q8_0 && pSrc1->type == GGML_TYPE_F32) {
    fGgmlCpyQ80F32Cuda(vSrc0Ddc, vSrc1Ddc, cNe, cNe00, cNe01, cNe02, cNb00, cNb01, cNb02, cNb03, cNe10, cNe11, cNe12,
                       cNb10, cNb11, cNb12, cNb13, vMainStream);
  } else if (pSrc0->type == GGML_TYPE_F32 && pSrc1->type == GGML_TYPE_Q4_0) {
    fGgmlCpyF32Q40Cuda(vSrc0Ddc, vSrc1Ddc, cNe, cNe00, cNe01, cNe02, cNb00, cNb01, cNb02, cNb03, cNe10, cNe11, cNe12,
                       cNb10, cNb11, cNb12, cNb13, vMainStream);
  } else if (pSrc0->type == GGML_TYPE_Q4_0 && pSrc1->type == GGML_TYPE_F32) {
    fGgmlCpyQ40F32Cuda(vSrc0Ddc, vSrc1Ddc, cNe, cNe00, cNe01, cNe02, cNb00, cNb01, cNb02, cNb03, cNe10, cNe11, cNe12,
                       cNb10, cNb11, cNb12, cNb13, vMainStream);
  } else if (pSrc0->type == GGML_TYPE_F32 && pSrc1->type == GGML_TYPE_Q4_1) {
    fGgmlCpyF32Q41Cuda(vSrc0Ddc, vSrc1Ddc, cNe, cNe00, cNe01, cNe02, cNb00, cNb01, cNb02, cNb03, cNe10, cNe11, cNe12,
                       cNb10, cNb11, cNb12, cNb13, vMainStream);
  } else if (pSrc0->type == GGML_TYPE_Q4_1 && pSrc1->type == GGML_TYPE_F32) {
    fGgmlCpyQ41F32Cuda(vSrc0Ddc, vSrc1Ddc, cNe, cNe00, cNe01, cNe02, cNb00, cNb01, cNb02, cNb03, cNe10, cNe11, cNe12,
                       cNb10, cNb11, cNb12, cNb13, vMainStream);
  } else if (pSrc0->type == GGML_TYPE_F32 && pSrc1->type == GGML_TYPE_Q5_0) {
    fGgmlCpyF32Q50Cuda(vSrc0Ddc, vSrc1Ddc, cNe, cNe00, cNe01, cNe02, cNb00, cNb01, cNb02, cNb03, cNe10, cNe11, cNe12,
                       cNb10, cNb11, cNb12, cNb13, vMainStream);
  } else if (pSrc0->type == GGML_TYPE_Q5_0 && pSrc1->type == GGML_TYPE_F32) {
    fGgmlCpyQ50F32Cuda(vSrc0Ddc, vSrc1Ddc, cNe, cNe00, cNe01, cNe02, cNb00, cNb01, cNb02, cNb03, cNe10, cNe11, cNe12,
                       cNb10, cNb11, cNb12, cNb13, vMainStream);
  } else if (pSrc0->type == GGML_TYPE_F32 && pSrc1->type == GGML_TYPE_IQ4_NL) {
    fGgmlCpyF32Iq4NlCuda(vSrc0Ddc, vSrc1Ddc, cNe, cNe00, cNe01, cNe02, cNb00, cNb01, cNb02, cNb03, cNe10, cNe11, cNe12,
                         cNb10, cNb11, cNb12, cNb13, vMainStream);
  } else if (pSrc0->type == GGML_TYPE_F32 && pSrc1->type == GGML_TYPE_Q5_1) {
    fGgmlCpyF32Q51Cuda(vSrc0Ddc, vSrc1Ddc, cNe, cNe00, cNe01, cNe02, cNb00, cNb01, cNb02, cNb03, cNe10, cNe11, cNe12,
                       cNb10, cNb11, cNb12, cNb13, vMainStream);
  } else if (pSrc0->type == GGML_TYPE_Q5_1 && pSrc1->type == GGML_TYPE_F32) {
    fGgmlCpyQ51F32Cuda(vSrc0Ddc, vSrc1Ddc, cNe, cNe00, cNe01, cNe02, cNb00, cNb01, cNb02, cNb03, cNe10, cNe11, cNe12,
                       cNb10, cNb11, cNb12, cNb13, vMainStream);
  } else if (pSrc0->type == GGML_TYPE_F16 && pSrc1->type == GGML_TYPE_F16) {
    if (cCanBeTransposed) {
      fGgmlCpyScalarCuda<half, half, true>(vSrc0Ddc, vSrc1Ddc, cNe, cNe00, cNe01, cNe02, cNb00, cNb01, cNb02, cNb03,
                                           cNe10, cNe11, cNe12, cNb10, cNb11, cNb12, cNb13, vMainStream);
    } else {
      fGgmlCpyScalarCuda<half, half>(vSrc0Ddc, vSrc1Ddc, cNe, cNe00, cNe01, cNe02, cNb00, cNb01, cNb02, cNb03, cNe10,
                                     cNe11, cNe12, cNb10, cNb11, cNb12, cNb13, vMainStream);
    }
  } else if (pSrc0->type == GGML_TYPE_F16 && pSrc1->type == GGML_TYPE_BF16) {
    if (cContiguousSrcs) {
      fGgmlCpyScalarContiguousCuda<half, nv_bfloat16>(vSrc0Ddc, vSrc1Ddc, cNe, vMainStream);
    } else {
      fGgmlCpyScalarCuda<half, nv_bfloat16>(vSrc0Ddc, vSrc1Ddc, cNe, cNe00, cNe01, cNe02, cNb00, cNb01, cNb02, cNb03,
                                            cNe10, cNe11, cNe12, cNb10, cNb11, cNb12, cNb13, vMainStream);
    }
  } else if (pSrc0->type == GGML_TYPE_F16 && pSrc1->type == GGML_TYPE_F32) {
    if (cContiguousSrcs) {
      fGgmlCpyScalarContiguousCuda<half, float>(vSrc0Ddc, vSrc1Ddc, cNe, vMainStream);
    } else {
      fGgmlCpyScalarCuda<half, float>(vSrc0Ddc, vSrc1Ddc, cNe, cNe00, cNe01, cNe02, cNb00, cNb01, cNb02, cNb03, cNe10,
                                      cNe11, cNe12, cNb10, cNb11, cNb12, cNb13, vMainStream);
    }
  } else if (pSrc0->type == GGML_TYPE_BF16 && pSrc1->type == GGML_TYPE_BF16) {
    if (cCanBeTransposed) {
      fGgmlCpyScalarCuda<nv_bfloat16, nv_bfloat16, true>(vSrc0Ddc, vSrc1Ddc, cNe, cNe00, cNe01, cNe02, cNb00, cNb01,
                                                         cNb02, cNb03, cNe10, cNe11, cNe12, cNb10, cNb11, cNb12, cNb13,
                                                         vMainStream);
    } else {
      fGgmlCpyScalarCuda<nv_bfloat16, nv_bfloat16>(vSrc0Ddc, vSrc1Ddc, cNe, cNe00, cNe01, cNe02, cNb00, cNb01, cNb02,
                                                   cNb03, cNe10, cNe11, cNe12, cNb10, cNb11, cNb12, cNb13, vMainStream);
    }
  } else if (pSrc0->type == GGML_TYPE_BF16 && pSrc1->type == GGML_TYPE_F16) {
    if (cContiguousSrcs) {
      fGgmlCpyScalarContiguousCuda<nv_bfloat16, half>(vSrc0Ddc, vSrc1Ddc, cNe, vMainStream);
    } else {
      fGgmlCpyScalarCuda<nv_bfloat16, half>(vSrc0Ddc, vSrc1Ddc, cNe, cNe00, cNe01, cNe02, cNb00, cNb01, cNb02, cNb03,
                                            cNe10, cNe11, cNe12, cNb10, cNb11, cNb12, cNb13, vMainStream);
    }
  } else if (pSrc0->type == GGML_TYPE_BF16 && pSrc1->type == GGML_TYPE_F32) {
    if (cContiguousSrcs) {
      fGgmlCpyScalarContiguousCuda<nv_bfloat16, float>(vSrc0Ddc, vSrc1Ddc, cNe, vMainStream);
    } else {
      fGgmlCpyScalarCuda<nv_bfloat16, float>(vSrc0Ddc, vSrc1Ddc, cNe, cNe00, cNe01, cNe02, cNb00, cNb01, cNb02, cNb03,
                                             cNe10, cNe11, cNe12, cNb10, cNb11, cNb12, cNb13, vMainStream);
    }
  } else if (pSrc0->type == GGML_TYPE_I32 && pSrc1->type == GGML_TYPE_I32) {
    if (cCanBeTransposed) {
      fGgmlCpyScalarCuda<int32_t, int32_t, true>(vSrc0Ddc, vSrc1Ddc, cNe, cNe00, cNe01, cNe02, cNb00, cNb01, cNb02,
                                                 cNb03, cNe10, cNe11, cNe12, cNb10, cNb11, cNb12, cNb13, vMainStream);
    } else {
      fGgmlCpyScalarCuda<int32_t, int32_t>(vSrc0Ddc, vSrc1Ddc, cNe, cNe00, cNe01, cNe02, cNb00, cNb01, cNb02, cNb03,
                                           cNe10, cNe11, cNe12, cNb10, cNb11, cNb12, cNb13, vMainStream);
    }
  } else if (pSrc0->type == GGML_TYPE_F32 && pSrc1->type == GGML_TYPE_I32) {
    if (cContiguousSrcs) {
      fGgmlCpyScalarContiguousCuda<float, int32_t>(vSrc0Ddc, vSrc1Ddc, cNe, vMainStream);
    } else {
      fGgmlCpyScalarCuda<float, int32_t>(vSrc0Ddc, vSrc1Ddc, cNe, cNe00, cNe01, cNe02, cNb00, cNb01, cNb02, cNb03,
                                         cNe10, cNe11, cNe12, cNb10, cNb11, cNb12, cNb13, vMainStream);
    }
  } else if (pSrc0->type == GGML_TYPE_I32 && pSrc1->type == GGML_TYPE_F32) {
    if (cContiguousSrcs) {
      fGgmlCpyScalarContiguousCuda<int32_t, float>(vSrc0Ddc, vSrc1Ddc, cNe, vMainStream);
    } else {
      fGgmlCpyScalarCuda<int32_t, float>(vSrc0Ddc, vSrc1Ddc, cNe, cNe00, cNe01, cNe02, cNb00, cNb01, cNb02, cNb03,
                                         cNe10, cNe11, cNe12, cNb10, cNb11, cNb12, cNb13, vMainStream);
    }
  } else {
    GGML_ABORT("%s: unsupported type combination (%s to %s)\n", __func__, ggml_type_name(pSrc0->type),
               ggml_type_name(pSrc1->type));
  }
}

void ggml_cuda_dup(ggml_backend_cuda_context &pCtx, ggml_tensor *pDst) {
  const ggml_tensor *vSrc0 = pDst->src[0];
  ggml_cuda_cpy(pCtx, vSrc0, pDst);
}
