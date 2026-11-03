#pragma once

#include "vecdotq.cuh"
#include "mma.cuh"

using namespace ggml_cuda_mma;

#include "mmq.cuh"

template <ggml_type type, int J, bool fallback>
static __device__ __forceinline__ void ggml_cuda_mmq_vec_dot_q4_0_q8_1_dp4a(const int *__restrict__ pX,
                                                                            const int *__restrict__ pY,
                                                                            float *__restrict__ pSum, const int pK00) {
  constexpr int cWarpSize = ggml_cuda_get_physical_warp_size();
  constexpr int cNwarps = ggml_cuda_mmq_get_nthreads(type, J, fallback) / cWarpSize;
  constexpr int I = ggml_cuda_mmq_get_I(type, J, fallback);

  constexpr tile_x_sizes cTxs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_Q4_0, I);
  const int *vXQs = (const int *)pX;
  const float *vXDf = (const float *)vXQs + cTxs.qs;
  const int *vYQs = (const int *)pY + 4;
  const half2 *vYDs = (const half2 *)pY;

  // #pragma unroll
  for (int vK01 = 0; vK01 < MMQ_TILE_NE_K; vK01 += QR4_0 * VDR_Q4_0_Q8_1_MMQ) {
    const int cK0 = pK00 + vK01;

#pragma unroll
    for (int vJ0 = 0; vJ0 < J; vJ0 += cNwarps) {
      const int cJ = vJ0 + threadIdx.y;

#pragma unroll
      for (int vI0 = 0; vI0 < I; vI0 += cWarpSize) {
        const int i = vI0 + threadIdx.x;
        const int cKyqs = QI8_1 * ((vK01 / 2) / (QI8_1 / 2)) + (vK01 / 2) % (QI8_1 / 2);

        int vU[2 * VDR_Q4_0_Q8_1_MMQ];

        constexpr int cMaxCpy = ggml_cuda_get_max_cpy_bytes();
        constexpr int cMcpyInt = cMaxCpy / sizeof(int);
        static_assert(VDR_Q4_0_Q8_1_MMQ == 4, "bad VDR_Q4_0_Q8_1_MMQ");

        int vTmp0[4], vTmp1[4];

#pragma unroll
        for (int vL0 = 0; vL0 < 4 / cMcpyInt; ++vL0) {
          ggml_cuda_memcpy_1<cMaxCpy>(vTmp0 + vL0 * cMcpyInt, &vYQs[cJ * MMQ_TILE_Y_K + cKyqs + vL0 * cMcpyInt]);
          ggml_cuda_memcpy_1<cMaxCpy>(vTmp1 + vL0 * cMcpyInt,
                                      &vYQs[cJ * MMQ_TILE_Y_K + cKyqs + QI4_0 + vL0 * cMcpyInt]);
        }

        vU[0] = vTmp0[0];
        vU[2] = vTmp0[1];
        vU[4] = vTmp0[2];
        vU[6] = vTmp0[3];
        vU[1] = vTmp1[0];
        vU[3] = vTmp1[1];
        vU[5] = vTmp1[2];
        vU[7] = vTmp1[3];

        pSum[vJ0 / cNwarps * I / cWarpSize + vI0 / cWarpSize] += vec_dot_q4_0_q8_1_impl<VDR_Q4_0_Q8_1_MMQ>(
            &vXQs[i * (MMQ_TILE_NE_K + 1) + cK0 / QR4_0], vU,
            vXDf[i * (MMQ_TILE_NE_K / QI4_0) + i / QI4_0 + cK0 / (QR4_0 * QI4_0)],
            vYDs[cJ * MMQ_TILE_Y_K + vK01 / QI8_1]);
      }
    }
  }
}

template <ggml_type type, int J, bool fallback>
static __device__ __forceinline__ void ggml_cuda_mmq_vec_dot_q4_1_q8_1_dp4a(const int *__restrict__ pX,
                                                                            const int *__restrict__ pY,
                                                                            float *__restrict__ pSum, const int pK00) {
  constexpr int cWarpSize = ggml_cuda_get_physical_warp_size();
  constexpr int cNwarps = ggml_cuda_mmq_get_nthreads(type, J, fallback) / cWarpSize;
  constexpr int I = ggml_cuda_mmq_get_I(type, J, fallback);

  constexpr tile_x_sizes cTxs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_Q4_1, I);
  const int *vXQs = (const int *)pX;
  const half2 *vXDm = (const half2 *)vXQs + cTxs.qs;
  const int *vYQs = (const int *)pY + 4;
  const half2 *vYDs = (const half2 *)pY;

  // #pragma unroll
  for (int vK01 = 0; vK01 < MMQ_TILE_NE_K; vK01 += QR4_1 * VDR_Q4_1_Q8_1_MMQ) {
    const int cK0 = pK00 + vK01;

#pragma unroll
    for (int vJ0 = 0; vJ0 < J; vJ0 += cNwarps) {
      const int cJ = vJ0 + threadIdx.y;

#pragma unroll
      for (int vI0 = 0; vI0 < I; vI0 += cWarpSize) {
        const int i = vI0 + threadIdx.x;
        const int cKyqs = QI8_1 * ((vK01 / 2) / (QI8_1 / 2)) + (vK01 / 2) % (QI8_1 / 2);

        int vU[2 * VDR_Q4_1_Q8_1_MMQ];

        constexpr int cMaxCpy = ggml_cuda_get_max_cpy_bytes();
        constexpr int cMcpyInt = cMaxCpy / sizeof(int);
        static_assert(VDR_Q4_0_Q8_1_MMQ == 4, "bad VDR_Q4_0_Q8_1_MMQ");

        int vTmp0[4], vTmp1[4];

#pragma unroll
        for (int vL0 = 0; vL0 < 4 / cMcpyInt; ++vL0) {
          ggml_cuda_memcpy_1<cMaxCpy>(vTmp0 + vL0 * cMcpyInt, &vYQs[cJ * MMQ_TILE_Y_K + cKyqs + vL0 * cMcpyInt]);
          ggml_cuda_memcpy_1<cMaxCpy>(vTmp1 + vL0 * cMcpyInt,
                                      &vYQs[cJ * MMQ_TILE_Y_K + cKyqs + QI4_1 + vL0 * cMcpyInt]);
        }

        vU[0] = vTmp0[0];
        vU[2] = vTmp0[1];
        vU[4] = vTmp0[2];
        vU[6] = vTmp0[3];
        vU[1] = vTmp1[0];
        vU[3] = vTmp1[1];
        vU[5] = vTmp1[2];
        vU[7] = vTmp1[3];

        pSum[vJ0 / cNwarps * I / cWarpSize + vI0 / cWarpSize] += vec_dot_q4_1_q8_1_impl<VDR_Q4_1_Q8_1_MMQ>(
            &vXQs[i * (MMQ_TILE_NE_K + 1) + cK0 / QR4_1], vU,
            vXDm[i * (MMQ_TILE_NE_K / QI4_1) + i / QI4_1 + cK0 / (QR4_1 * QI4_1)],
            vYDs[cJ * MMQ_TILE_Y_K + vK01 / QI8_1]);
      }
    }
  }
}

template <ggml_type type, int J, bool fallback>
static __device__ __forceinline__ void ggml_cuda_mmq_vec_dot_q8_0_q8_1_dp4a(const int *__restrict__ pX,
                                                                            const int *__restrict__ pY,
                                                                            float *__restrict__ pSum, const int pK00) {
  constexpr int cWarpSize = ggml_cuda_get_physical_warp_size();
  constexpr int cNwarps = ggml_cuda_mmq_get_nthreads(type, J, fallback) / cWarpSize;
  constexpr int I = ggml_cuda_mmq_get_I(type, J, fallback);

  constexpr tile_x_sizes cTxs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_Q8_0, I);
  const int *vXQs = (const int *)pX;
  const float *vXDf = (const float *)vXQs + cTxs.qs;
  const int *vYQs = (const int *)pY + 4;
  const float *vYDf = (const float *)pY;

  // #pragma unroll
  for (int vK01 = 0; vK01 < MMQ_TILE_NE_K; vK01 += VDR_Q8_0_Q8_1_MMQ) {
    const int cK0 = pK00 + vK01;

#pragma unroll
    for (int vJ0 = 0; vJ0 < J; vJ0 += cNwarps) {
      const int cJ = vJ0 + threadIdx.y;

#pragma unroll
      for (int vI0 = 0; vI0 < I; vI0 += cWarpSize) {
        const int i = vI0 + threadIdx.x;

        pSum[vJ0 / cNwarps * I / cWarpSize + vI0 / cWarpSize] += vec_dot_q8_0_q8_1_impl<float, VDR_Q8_0_Q8_1_MMQ>(
            &vXQs[i * (2 * MMQ_TILE_NE_K + 1) + cK0], &vYQs[cJ * MMQ_TILE_Y_K + cK0 % MMQ_TILE_NE_K],
            vXDf[i * (2 * MMQ_TILE_NE_K / QI8_0) + i / (QI8_0 / 2) + cK0 / QI8_0],
            vYDf[cJ * MMQ_TILE_Y_K + (cK0 / QI8_1) % (MMQ_TILE_NE_K / QI8_1)]);
      }
    }
  }
}

template <ggml_type type, int J, bool fallback, mmq_q8_1_ds_layout ds_layout>
static __device__ __forceinline__ void ggml_cuda_mmq_vec_dot_q8_0_q8_1_mma(const int *__restrict__ pX,
                                                                           const int *__restrict__ pY,
                                                                           float *__restrict__ pSum, const int pK00) {
#if defined(AMD_MFMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
  constexpr data_layout input_layout = get_input_data_layout();
  typedef tile<16, 8, int, input_layout> tile_A;
  typedef tile<16, 8, int, input_layout> tile_B;
  typedef tile<16, 16, int, DATA_LAYOUT_J_MAJOR> tile_C;

  constexpr int I = ggml_cuda_mmq_get_I(type, J, fallback);
  constexpr int cSramStride = ggml_cuda_mmq_get_sram_stride(type, J, fallback);
  constexpr int cRowsPerWarp = ggml_cuda_mmq_get_rows_per_warp(type, J, fallback);
  constexpr int cNtx = cRowsPerWarp / tile_C::I; // Number of x minitiles per warp.

  pY += (threadIdx.y % cNtx) * (tile_C::J * MMQ_TILE_Y_K);

  const int *vXQs = (const int *)pX;
  const float *vXDf = (const float *)vXQs + 2 * MMQ_TILE_NE_K;
  const int *vYQs = (const int *)pY + 4;
  const float *vYDf = (const float *)pY;
  const half2 *vYDs = (const half2 *)pY;

  const int cI0 = (threadIdx.y / cNtx) * cRowsPerWarp;

  for (int vK01 = 0; vK01 < MMQ_TILE_NE_K; vK01 += QI8_0) {
    const int cK0 = pK00 + vK01;

    tile_A vA[cNtx];
#pragma unroll
    for (int vN = 0; vN < cNtx; ++vN) {
      load_ldmatrix(vA[vN], vXQs + (cI0 + vN * tile_A::I) * cSramStride + cK0, cSramStride);
    }

#pragma unroll
    for (int vJ0 = 0; vJ0 < J; vJ0 += cNtx * tile_C::J) {
      tile_B vB;
      load_ldmatrix(vB, vYQs + vJ0 * MMQ_TILE_Y_K + vK01, MMQ_TILE_Y_K);

      float vDB;
      const int cJ = vJ0 + tile_C::get_j(0);
      if (ds_layout == MMQ_Q8_1_DS_LAYOUT_D4) {
        vDB = vYDf[cJ * MMQ_TILE_Y_K + vK01 / QI8_1];
      } else {
        vDB = __low2float(vYDs[cJ * MMQ_TILE_Y_K + vK01 / QI8_1]);
      }

#pragma unroll
      for (int vN = 0; vN < cNtx; ++vN) {
        tile_C vC;
        mma(vC, vA[vN], vB);

#pragma unroll
        for (int vL = 0; vL < tile_C::ne; ++vL) {
          const int i = cI0 + vN * tile_A::I + tile_C::get_i(vL);
          const float vDA = vXDf[i * cSramStride + cK0 / QI8_0];
          pSum[(vJ0 / tile_C::J + vN) * tile_C::ne + vL] += vC.x[vL] * vDA * vDB;
        }
      }
    }
  }
#else
  typedef tile<16, 8, int> tile_A;
  typedef tile<8, 8, int> tile_B;
  typedef tile<16, 8, int> tile_C;

  constexpr int I = ggml_cuda_mmq_get_I(type, J, fallback);
  constexpr int cSramStride = ggml_cuda_mmq_get_sram_stride(type, J, fallback);
  constexpr int cRowsPerWarp = ggml_cuda_mmq_get_rows_per_warp(type, J, fallback);
  constexpr int cNtx = cRowsPerWarp / tile_C::I; // Number of x minitiles per warp.

  pY += (threadIdx.y % cNtx) * (tile_C::J * MMQ_TILE_Y_K);

  const int *vXQs = (const int *)pX;
  const float *vXDf = (const float *)vXQs + 2 * MMQ_TILE_NE_K;
  const int *vYQs = (const int *)pY + 4;
  const float *vYDf = (const float *)pY;
  const half2 *vYDs = (const half2 *)pY;

  tile_A vA[cNtx][MMQ_TILE_NE_K / QI8_0];
  float vDA[cNtx][tile_C::ne / 2][MMQ_TILE_NE_K / QI8_0];

  const int cI0 = (threadIdx.y / cNtx) * cRowsPerWarp;

#pragma unroll
  for (int vN = 0; vN < cNtx; ++vN) {
#pragma unroll
    for (int vK01 = 0; vK01 < MMQ_TILE_NE_K; vK01 += QI8_0) {
      const int cK0 = pK00 + vK01;

      load_ldmatrix(vA[vN][vK01 / QI8_0], vXQs + (cI0 + vN * tile_A::I) * cSramStride + cK0, cSramStride);
    }

#pragma unroll
    for (int vL = 0; vL < tile_C::ne / 2; ++vL) {
      const int i = cI0 + vN * tile_A::I + tile_C::get_i(2 * vL);

#pragma unroll
      for (int vK01 = 0; vK01 < MMQ_TILE_NE_K; vK01 += QI8_0) {
        const int cK0 = pK00 + vK01;

        vDA[vN][vL][vK01 / QI8_0] = vXDf[i * cSramStride + cK0 / QI8_0];
      }
    }
  }

#pragma unroll
  for (int vJ0 = 0; vJ0 < J; vJ0 += cNtx * tile_C::J) {
#pragma unroll
    for (int vK01 = 0; vK01 < MMQ_TILE_NE_K; vK01 += QI8_0) {
      tile_B vB;
      float vDB[tile_C::ne / 2];

      load_generic(vB, vYQs + vJ0 * MMQ_TILE_Y_K + vK01, MMQ_TILE_Y_K); // faster than load_ldmatrix

#pragma unroll
      for (int vL = 0; vL < tile_C::ne / 2; ++vL) {
        const int cJ = vJ0 + tile_C::get_j(vL);

        if (ds_layout == MMQ_Q8_1_DS_LAYOUT_D4) {
          vDB[vL] = vYDf[cJ * MMQ_TILE_Y_K + vK01 / QI8_1];
        } else {
          vDB[vL] = __low2float(vYDs[cJ * MMQ_TILE_Y_K + vK01 / QI8_1]);
        }
      }

#pragma unroll
      for (int vN = 0; vN < cNtx; ++vN) {
        tile_C vC;
        mma(vC, vA[vN][vK01 / QI8_0], vB);

#pragma unroll
        for (int vL = 0; vL < tile_C::ne; ++vL) {
          pSum[(vJ0 / tile_C::J + vN) * tile_C::ne + vL] += vC.x[vL] * vDA[vN][vL / 2][vK01 / QI8_0] * vDB[vL % 2];
        }
      }
    }
  }
#endif // defined(AMD_MFMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
}

template <ggml_type type, int J, bool fallback>
static __device__ __forceinline__ void ggml_cuda_mmq_vec_dot_q8_1_q8_1_dp4a(const int *__restrict__ pX,
                                                                            const int *__restrict__ pY,
                                                                            float *__restrict__ pSum, const int pK00) {
  constexpr int cWarpSize = ggml_cuda_get_physical_warp_size();
  constexpr int cNwarps = ggml_cuda_mmq_get_nthreads(type, J, fallback) / cWarpSize;
  constexpr int I = ggml_cuda_mmq_get_I(type, J, fallback);

  constexpr tile_x_sizes cTxs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_Q5_1, I);
  const int *vXQs = (const int *)pX;
  const half2 *vXDm = (const half2 *)vXQs + cTxs.qs;
  const int *vYQs = (const int *)pY + 4;
  const half2 *vYDs = (const half2 *)pY;

  // #pragma unroll
  for (int vK01 = 0; vK01 < MMQ_TILE_NE_K; vK01 += VDR_Q8_0_Q8_1_MMQ) {
    const int cK0 = pK00 + vK01;

#pragma unroll
    for (int vJ0 = 0; vJ0 < J; vJ0 += cNwarps) {
      const int cJ = vJ0 + threadIdx.y;

#pragma unroll
      for (int vI0 = 0; vI0 < I; vI0 += cWarpSize) {
        const int i = vI0 + threadIdx.x;

        pSum[vJ0 / cNwarps * I / cWarpSize + vI0 / cWarpSize] += vec_dot_q8_1_q8_1_impl<QR5_1 * VDR_Q5_1_Q8_1_MMQ>(
            &vXQs[i * (2 * MMQ_TILE_NE_K + 1) + cK0], &vYQs[cJ * MMQ_TILE_Y_K + vK01],
            vXDm[i * (MMQ_TILE_NE_K / QI5_1) + i / QI5_1 + cK0 / QI8_1], vYDs[cJ * MMQ_TILE_Y_K + vK01 / QI8_1]);
      }
    }
  }
}

template <ggml_type type, int J, bool fallback>
static __device__ __forceinline__ void ggml_cuda_mmq_vec_dot_q8_1_q8_1_mma(const int *__restrict__ pX,
                                                                           const int *__restrict__ pY,
                                                                           float *__restrict__ pSum, const int pK00) {
#if defined(AMD_MFMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
  constexpr data_layout input_layout = get_input_data_layout();
  typedef tile<16, 8, int, input_layout> tile_A;
  typedef tile<16, 8, int, input_layout> tile_B;
  typedef tile<16, 16, int, DATA_LAYOUT_J_MAJOR> tile_C;

  constexpr int I = ggml_cuda_mmq_get_I(type, J, fallback);
  constexpr int cSramStride = ggml_cuda_mmq_get_sram_stride(type, J, fallback);
  constexpr int cRowsPerWarp = ggml_cuda_mmq_get_rows_per_warp(type, J, fallback);
  constexpr int cNtx = cRowsPerWarp / tile_C::I; // Number of x minitiles per warp.

  pY += (threadIdx.y % cNtx) * (tile_C::J * MMQ_TILE_Y_K);

  const int *vXQs = (const int *)pX;
  const half2 *vXDm = (const half2 *)vXQs + 2 * MMQ_TILE_NE_K;
  const int *vYQs = (const int *)pY + 4;
  const half2 *vYDm = (const half2 *)pY;

  const int cI0 = (threadIdx.y / cNtx) * cRowsPerWarp;

  for (int vK01 = 0; vK01 < MMQ_TILE_NE_K; vK01 += QI8_1) {
    const int cK0 = pK00 + vK01;

    tile_A vA[cNtx];
#pragma unroll
    for (int vN = 0; vN < cNtx; ++vN) {
      load_ldmatrix(vA[vN], vXQs + (cI0 + vN * tile_A::I) * cSramStride + cK0, cSramStride);
    }

#pragma unroll
    for (int vJ0 = 0; vJ0 < J; vJ0 += cNtx * tile_C::J) {
      tile_B vB;
      load_ldmatrix(vB, vYQs + vJ0 * MMQ_TILE_Y_K + vK01, MMQ_TILE_Y_K);

      const int cJ = vJ0 + tile_C::get_j(0);
      const float2 vDsB = __half22float2(vYDm[cJ * MMQ_TILE_Y_K + vK01 / QI8_1]);

#pragma unroll
      for (int vN = 0; vN < cNtx; ++vN) {
        tile_C vC;
        mma(vC, vA[vN], vB);

#pragma unroll
        for (int vL = 0; vL < tile_C::ne; ++vL) {
          const int i = cI0 + vN * tile_A::I + tile_C::get_i(vL);
          float2 vDmA = __half22float2(vXDm[i * cSramStride + cK0 / QI8_1]);
          pSum[(vJ0 / tile_C::J + vN) * tile_C::ne + vL] += vDmA.x * vDsB.x * vC.x[vL];
          pSum[(vJ0 / tile_C::J + vN) * tile_C::ne + vL] += vDmA.y * vDsB.y;
        }
      }
    }
  }
#else
  typedef tile<16, 8, int> tile_A;
  typedef tile<8, 8, int> tile_B;
  typedef tile<16, 8, int> tile_C;

  constexpr int I = ggml_cuda_mmq_get_I(type, J, fallback);
  constexpr int cSramStride = ggml_cuda_mmq_get_sram_stride(type, J, fallback);
  constexpr int cRowsPerWarp = ggml_cuda_mmq_get_rows_per_warp(type, J, fallback);
  constexpr int cNtx = cRowsPerWarp / tile_C::I; // Number of x minitiles per warp.

  pY += (threadIdx.y % cNtx) * (tile_C::J * MMQ_TILE_Y_K);

  const int *vXQs = (const int *)pX;
  const half2 *vXDm = (const half2 *)vXQs + 2 * MMQ_TILE_NE_K;
  const int *vYQs = (const int *)pY + 4;
  const half2 *vYDm = (const half2 *)pY;

  tile_A vA[cNtx][MMQ_TILE_NE_K / QI8_1];
  float2 vDmA[cNtx][tile_C::ne / 2][MMQ_TILE_NE_K / QI8_1];

  const int cI0 = (threadIdx.y / cNtx) * cRowsPerWarp;

#pragma unroll
  for (int vN = 0; vN < cNtx; ++vN) {
#pragma unroll
    for (int vK01 = 0; vK01 < MMQ_TILE_NE_K; vK01 += QI8_1) {
      const int cK0 = pK00 + vK01;

      load_ldmatrix(vA[vN][vK01 / QI8_1], vXQs + (cI0 + vN * tile_A::I) * cSramStride + cK0, cSramStride);
    }

#pragma unroll
    for (int vL = 0; vL < tile_C::ne / 2; ++vL) {
      const int i = cI0 + vN * tile_A::I + tile_C::get_i(2 * vL);

#pragma unroll
      for (int vK01 = 0; vK01 < MMQ_TILE_NE_K; vK01 += QI8_1) {
        const int cK0 = pK00 + vK01;

        vDmA[vN][vL][vK01 / QI8_1] = __half22float2(vXDm[i * cSramStride + cK0 / QI8_1]);
      }
    }
  }

#pragma unroll
  for (int vJ0 = 0; vJ0 < J; vJ0 += cNtx * tile_C::J) {
#pragma unroll
    for (int vK01 = 0; vK01 < MMQ_TILE_NE_K; vK01 += QI8_1) {
      tile_B vB;
      float2 vDsB[tile_C::ne / 2];

      load_generic(vB, vYQs + vJ0 * MMQ_TILE_Y_K + vK01, MMQ_TILE_Y_K); // faster than load_ldmatrix

#pragma unroll
      for (int vL = 0; vL < tile_C::ne / 2; ++vL) {
        const int cJ = vJ0 + tile_C::get_j(vL);

        vDsB[vL] = __half22float2(vYDm[cJ * MMQ_TILE_Y_K + vK01 / QI8_1]);
      }

#pragma unroll
      for (int vN = 0; vN < cNtx; ++vN) {
        tile_C vC;
        mma(vC, vA[vN][vK01 / QI8_1], vB);

#pragma unroll
        for (int vL = 0; vL < tile_C::ne; ++vL) {
          pSum[(vJ0 / tile_C::J + vN) * tile_C::ne + vL] +=
              vDmA[vN][vL / 2][vK01 / QI8_1].x * vDsB[vL % 2].x * vC.x[vL];
          pSum[(vJ0 / tile_C::J + vN) * tile_C::ne + vL] += vDmA[vN][vL / 2][vK01 / QI8_1].y * vDsB[vL % 2].y;
        }
      }
    }
  }
#endif // defined(AMD_MFMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
}

// Used for NVFP4, Q3_K, IQ2_S, and IQ2_XS
template <ggml_type type, int J, bool fallback>
static __device__ __forceinline__ void
ggml_cuda_mmq_vec_dot_q8_0_16_q8_1_dp4a(const int *__restrict__ pX, const int *__restrict__ pY,
                                        float *__restrict__ pSum, const int pK00) {
  constexpr int cWarpSize = ggml_cuda_get_physical_warp_size();
  constexpr int cNwarps = ggml_cuda_mmq_get_nthreads(type, J, fallback) / cWarpSize;
  constexpr int I = ggml_cuda_mmq_get_I(type, J, fallback);

  constexpr tile_x_sizes cTxs = mmq_get_dp4a_tile_x_sizes(type, I);
  const int *vXQs = (const int *)pX;
  const float *vXDf = (const float *)vXQs + cTxs.qs;
  const int *vYQs = (const int *)pY + 4;
  const float *vYDf = (const float *)pY;

  // #pragma unroll
  for (int vK01 = 0; vK01 < MMQ_TILE_NE_K; vK01 += QI8_0) {
    const int cK0 = pK00 + vK01;

#pragma unroll
    for (int vJ0 = 0; vJ0 < J; vJ0 += cNwarps) {
      const int cJ = vJ0 + threadIdx.y;

#pragma unroll
      for (int vI0 = 0; vI0 < I; vI0 += cWarpSize) {
        const int i = vI0 + threadIdx.x;

        pSum[vJ0 / cNwarps * I / cWarpSize + vI0 / cWarpSize] += vec_dot_q8_0_16_q8_1_impl<QI8_0>(
            &vXQs[i * (2 * MMQ_TILE_NE_K + 1) + cK0], &vYQs[cJ * MMQ_TILE_Y_K + vK01],
            &vXDf[i * (2 * MMQ_TILE_NE_K * 2 / QI8_0) + i / (QI8_0 / 4) + cK0 / (QI8_0 / 2)],
            vYDf[cJ * MMQ_TILE_Y_K + vK01 / QI8_1]);
      }
    }
  }
}

// Used for Q3_K, IQ2_S, and IQ2_XS:
template <ggml_type type, int J, bool fallback>
static __device__ __forceinline__ void
ggml_cuda_mmq_vec_dot_q8_0_16_q8_1_mma(const int *__restrict__ pX, const int *__restrict__ pY, float *__restrict__ pSum,
                                       const int pK00) {
#if defined(AMD_MFMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
  constexpr data_layout input_layout = get_input_data_layout();
  typedef tile<16, 4, int, input_layout> tile_A;
  typedef tile<16, 4, int, input_layout> tile_B;
  typedef tile<16, 16, int, DATA_LAYOUT_J_MAJOR> tile_C;

  constexpr int I = ggml_cuda_mmq_get_I(type, J, fallback);
  constexpr int cSramStride = ggml_cuda_mmq_get_sram_stride(type, J, fallback);
  constexpr int cRowsPerWarp = ggml_cuda_mmq_get_rows_per_warp(type, J, fallback);
  constexpr int cNtx = cRowsPerWarp / tile_C::I; // Number of x minitiles per warp.

  pY += (threadIdx.y % cNtx) * (tile_C::J * MMQ_TILE_Y_K);

  const int *vXQs = (const int *)pX;
  const float *vXDf = (const float *)vXQs + MMQ_TILE_NE_K * 2;
  const int *vYQs = (const int *)pY + 4;
  const float *vYDf = (const float *)pY;

  const int cI0 = (threadIdx.y / cNtx) * cRowsPerWarp;

  for (int vK01 = 0; vK01 < MMQ_TILE_NE_K; vK01 += 4) {
    const int cK0 = pK00 + vK01;

    tile_A vA[cNtx];
#pragma unroll
    for (int vN = 0; vN < cNtx; ++vN) {
      load_ldmatrix(vA[vN], vXQs + (cI0 + vN * tile_A::I) * cSramStride + cK0, cSramStride);
    }

#pragma unroll
    for (int vJ0 = 0; vJ0 < J; vJ0 += cNtx * tile_C::J) {
      tile_B vB;
      load_ldmatrix(vB, vYQs + vJ0 * MMQ_TILE_Y_K + vK01, MMQ_TILE_Y_K);

      const int cJ = vJ0 + tile_C::get_j(0);
      const float vDB = vYDf[cJ * MMQ_TILE_Y_K + vK01 / QI8_1];

#pragma unroll
      for (int vN = 0; vN < cNtx; ++vN) {
        tile_C vC;
        mma(vC, vA[vN], vB);

#pragma unroll
        for (int vL = 0; vL < tile_C::ne; ++vL) {
          const int i = cI0 + vN * tile_C::I + tile_C::get_i(vL);
          pSum[(vJ0 / tile_C::J + vN) * tile_C::ne + vL] += vC.x[vL] * vXDf[i * cSramStride + cK0 / 4] * vDB;
        }
      }
    }
  }
#elif defined(TURING_MMA_AVAILABLE)

  typedef tile<16, 4, int> tile_A;
  typedef tile<16, 8, int> tile_A_8;
  typedef tile<8, 4, int> tile_B;
  typedef tile<16, 8, int> tile_C;

  constexpr int I = ggml_cuda_mmq_get_I(type, J, fallback);
  constexpr int cSramStride = ggml_cuda_mmq_get_sram_stride(type, J, fallback);
  constexpr int cRowsPerWarp = ggml_cuda_mmq_get_rows_per_warp(type, J, fallback);
  constexpr int cNtx = cRowsPerWarp / tile_C::I; // Number of x minitiles per warp.

  pY += (threadIdx.y % cNtx) * (tile_C::J * MMQ_TILE_Y_K);

  const int *vXQs = (const int *)pX;
  const float *vXDf = (const float *)vXQs + MMQ_TILE_NE_K * 2;
  const int *vYQs = (const int *)pY + 4;
  const float *vYDf = (const float *)pY;

  const int cI0 = (threadIdx.y / cNtx) * (cNtx * tile_A::I);

  tile_A vA[cNtx][8];
  float vDA[cNtx][tile_C::ne / 2][8];

#pragma unroll
  for (int vN = 0; vN < cNtx; ++vN) {
#pragma unroll
    for (int vK01 = 0; vK01 < MMQ_TILE_NE_K; vK01 += 8) {
      const int cK0 = pK00 + vK01;

      load_ldmatrix(((tile_A_8 *)vA[vN])[vK01 / 8], vXQs + (cI0 + vN * tile_A::I) * cSramStride + cK0, cSramStride);
    }

#pragma unroll
    for (int vL = 0; vL < tile_C::ne / 2; ++vL) {
      const int i = cI0 + vN * tile_C::I + tile_C::get_i(2 * vL);

#pragma unroll
      for (int vK01 = 0; vK01 < MMQ_TILE_NE_K; vK01 += 4) {
        const int cK0 = pK00 + vK01;

        vDA[vN][vL][vK01 / 4] = vXDf[i * cSramStride + cK0 / 4];
      }
    }
  }

#pragma unroll
  for (int vJ0 = 0; vJ0 < J; vJ0 += cNtx * tile_C::J) {
#pragma unroll
    for (int vK01 = 0; vK01 < MMQ_TILE_NE_K; vK01 += QR3_K * VDR_Q3_K_Q8_1_MMQ) {
      tile_B vB[2];
      float vDB[tile_C::ne / 2];

      // Here load_generic is faster than load_ldmatrix.
      load_generic(vB[0], vYQs + vJ0 * MMQ_TILE_Y_K + (vK01 + 0), MMQ_TILE_Y_K);
      load_generic(vB[1], vYQs + vJ0 * MMQ_TILE_Y_K + (vK01 + tile_B::J), MMQ_TILE_Y_K);

#pragma unroll
      for (int vL = 0; vL < tile_C::ne / 2; ++vL) {
        const int cJ = vJ0 + tile_C::get_j(vL);

        vDB[vL] = vYDf[cJ * MMQ_TILE_Y_K + vK01 / QI8_1];
      }

#pragma unroll
      for (int vN = 0; vN < cNtx; ++vN) {
        tile_C vC[2];
        mma(vC[0], vA[vN][vK01 / 4 + 0], vB[0]);
        mma(vC[1], vA[vN][vK01 / 4 + 1], vB[1]);

#pragma unroll
        for (int vL = 0; vL < tile_C::ne; ++vL) {
          pSum[(vJ0 / tile_C::J + vN) * tile_C::ne + vL] +=
              vDB[vL % 2] * (vC[0].x[vL] * vDA[vN][vL / 2][vK01 / 4 + 0] + vC[1].x[vL] * vDA[vN][vL / 2][vK01 / 4 + 1]);
        }
      }
    }
  }
#else
  GGML_UNUSED_VARS(pX, pY, pSum, pK00);
  NO_DEVICE_CODE;
#endif // AMD_MFMA_AVAILABLE || AMD_WMMA_AVAILABLE
}

template <ggml_type type, int J, bool fallback>
static __device__ __forceinline__ void ggml_cuda_mmq_vec_dot_q2_K_q8_1_dp4a(const int *__restrict__ pX,
                                                                            const int *__restrict__ pY,
                                                                            float *__restrict__ pSum, const int pK00) {
  constexpr int cWarpSize = ggml_cuda_get_physical_warp_size();
  constexpr int cNwarps = ggml_cuda_mmq_get_nthreads(type, J, fallback) / cWarpSize;
  constexpr int I = ggml_cuda_mmq_get_I(type, J, fallback);

  constexpr tile_x_sizes cTxs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_Q2_K, I);
  const int *vXQs = (const int *)pX;
  const half2 *vXDm = (const half2 *)vXQs + cTxs.qs;
  const int *vYQs = (const int *)pY + 4;
  const half2 *vYDs = (const half2 *)pY;

  float2 vYDf[J / cNwarps];
#pragma unroll
  for (int vJ0 = 0; vJ0 < J; vJ0 += cNwarps) {
    const int cJ = vJ0 + threadIdx.y;

    vYDf[vJ0 / cNwarps] = __half22float2(vYDs[cJ * MMQ_TILE_Y_K]);
  }

#pragma unroll
  for (int vK01 = 0; vK01 < MMQ_TILE_NE_K / 2; vK01 += QR2_K * VDR_Q2_K_Q8_1_MMQ) {
    const int cK0 = pK00 + vK01;

#pragma unroll
    for (int vJ0 = 0; vJ0 < J; vJ0 += cNwarps) {
      const int cJ = vJ0 + threadIdx.y;

#pragma unroll
      for (int vI0 = 0; vI0 < I; vI0 += cWarpSize) {
        const int i = vI0 + threadIdx.x;

        constexpr int cNs = 2;
        pSum[vJ0 / cNwarps * I / cWarpSize + vI0 / cWarpSize] +=
            vec_dot_q2_K_q8_1_impl_mmq<cNs>(&vXQs[i * (2 * MMQ_TILE_NE_K + 1) + cK0], &vYQs[cJ * MMQ_TILE_Y_K + vK01],
                                            &vXDm[i * (MMQ_TILE_NE_K + 1) + cK0 / 4],
                                            vK01 < MMQ_TILE_NE_K / 2 ? vYDf[vJ0 / cNwarps].x : vYDf[vJ0 / cNwarps].y,
                                            &vYDs[cJ * MMQ_TILE_Y_K + (1 + vK01 / QI8_1)]);
      }
    }
  }

  // Some compilers fail to unroll the loop over k01 if there is a conditional statement for ns in the inner loop.
  // As a workaround 2 separate loops are used instead.
#pragma unroll
  for (int vK01 = MMQ_TILE_NE_K / 2; vK01 < MMQ_TILE_NE_K; vK01 += QR2_K * VDR_Q2_K_Q8_1_MMQ) {
    const int cK0 = pK00 + vK01;

#pragma unroll
    for (int vJ0 = 0; vJ0 < J; vJ0 += cNwarps) {
      const int cJ = vJ0 + threadIdx.y;

#pragma unroll
      for (int vI0 = 0; vI0 < I; vI0 += cWarpSize) {
        const int i = vI0 + threadIdx.x;

        constexpr int cNs = 1;
        pSum[vJ0 / cNwarps * I / cWarpSize + vI0 / cWarpSize] +=
            vec_dot_q2_K_q8_1_impl_mmq<cNs>(&vXQs[i * (2 * MMQ_TILE_NE_K + 1) + cK0], &vYQs[cJ * MMQ_TILE_Y_K + vK01],
                                            &vXDm[i * (MMQ_TILE_NE_K + 1) + cK0 / 4],
                                            vK01 < MMQ_TILE_NE_K / 2 ? vYDf[vJ0 / cNwarps].x : vYDf[vJ0 / cNwarps].y,
                                            &vYDs[cJ * MMQ_TILE_Y_K + (1 + vK01 / QI8_1)]);
      }
    }
  }
}

template <ggml_type type, int J, bool fallback>
static __device__ __forceinline__ void ggml_cuda_mmq_vec_dot_q2_K_q8_1_mma(const int *__restrict__ pX,
                                                                           const int *__restrict__ pY,
                                                                           float *__restrict__ pSum, const int pK00) {
#if defined(AMD_MFMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
  constexpr data_layout input_layout = get_input_data_layout();
  typedef tile<16, 4, int, input_layout> tile_A;
  typedef tile<16, 4, int, input_layout> tile_B;
  typedef tile<16, 16, int, DATA_LAYOUT_J_MAJOR> tile_C;

  constexpr int I = ggml_cuda_mmq_get_I(type, J, fallback);
  constexpr int cSramStride = ggml_cuda_mmq_get_sram_stride(type, J, fallback);
  constexpr int cRowsPerWarp = ggml_cuda_mmq_get_rows_per_warp(type, J, fallback);
  constexpr int cNtx = cRowsPerWarp / tile_C::I; // Number of x minitiles per warp.

  pY += (threadIdx.y % cNtx) * (tile_C::J * MMQ_TILE_Y_K);

  const int *vXQs = (const int *)pX;
  const half2 *vXDm = (const half2 *)vXQs + MMQ_TILE_NE_K * 2;
  const int *vYQs = (const int *)pY + 4;
  const half2 *vYDs = (const half2 *)pY;

  const int cI0 = (threadIdx.y / cNtx) * cRowsPerWarp;

  for (int vK01 = 0; vK01 < MMQ_TILE_NE_K; vK01 += 4) {
    const int cK0 = pK00 + vK01;

    tile_A vA[cNtx];
#pragma unroll
    for (int vN = 0; vN < cNtx; ++vN) {
      load_ldmatrix(vA[vN], vXQs + (cI0 + vN * tile_A::I) * cSramStride + cK0, cSramStride);
    }

#pragma unroll
    for (int vJ0 = 0; vJ0 < J; vJ0 += cNtx * tile_C::J) {
      tile_B vB;
      load_ldmatrix(vB, vYQs + vJ0 * MMQ_TILE_Y_K + vK01, MMQ_TILE_Y_K);

      const int cJ = vJ0 + tile_C::get_j(0);
      const float vDB = (vK01 < MMQ_TILE_NE_K / 2) ? __half22float2(vYDs[cJ * MMQ_TILE_Y_K]).x
                                                   : __half22float2(vYDs[cJ * MMQ_TILE_Y_K]).y;
      const float vSB = (vK01 >= MMQ_TILE_NE_K * 3 / 4)
                            ? 0
                            : (((vK01 / 4) % 2) ? __half22float2(vYDs[cJ * MMQ_TILE_Y_K + (1 + vK01 / QI8_1)]).y
                                                : __half22float2(vYDs[cJ * MMQ_TILE_Y_K + (1 + vK01 / QI8_1)]).x);

      tile_C vCm;
      if (vK01 >= MMQ_TILE_NE_K * 3 / 4) {
        tile_A vA1;
#pragma unroll
        for (int vL = 0; vL < tile_A::ne; ++vL) {
          vA1.x[vL] = 0x01010101;
        }
        mma(vCm, vA1, vB);
      }

#pragma unroll
      for (int vN = 0; vN < cNtx; ++vN) {
        tile_C vCd;
        mma(vCd, vA[vN], vB);

#pragma unroll
        for (int vL = 0; vL < tile_C::ne; ++vL) {
          const int i = cI0 + vN * tile_C::I + tile_C::get_i(vL);
          const float2 cDm = __half22float2(vXDm[i * cSramStride + cK0 / 4]);
          float vTmp = vCd.x[vL] * cDm.x;
          if (vK01 >= MMQ_TILE_NE_K * 3 / 4) {
            vTmp -= vCm.x[vL] * cDm.y;
          }
          pSum[(vJ0 / tile_C::J + vN) * tile_C::ne + vL] += vTmp * vDB;
          pSum[(vJ0 / tile_C::J + vN) * tile_C::ne + vL] -= cDm.y * vSB;
        }
      }
    }
  }
#elif defined(TURING_MMA_AVAILABLE)

  typedef tile<16, 4, int> tile_A;
  typedef tile<16, 8, int> tile_A_8;
  typedef tile<8, 4, int> tile_B;
  typedef tile<16, 8, int> tile_C;

  constexpr int I = ggml_cuda_mmq_get_I(type, J, fallback);
  constexpr int cSramStride = ggml_cuda_mmq_get_sram_stride(type, J, fallback);
  constexpr int cRowsPerWarp = ggml_cuda_mmq_get_rows_per_warp(type, J, fallback);
  constexpr int cNtx = cRowsPerWarp / tile_C::I; // Number of x minitiles per warp.

  pY += (threadIdx.y % cNtx) * (tile_C::J * MMQ_TILE_Y_K);

  const int *vXQs = (const int *)pX;
  const half2 *vXDm = (const half2 *)vXQs + MMQ_TILE_NE_K * 2;
  const int *vYQs = (const int *)pY + 4;
  const half2 *vYDs = (const half2 *)pY;

  const int cI0 = (threadIdx.y / cNtx) * (cNtx * tile_A::I);

  tile_A vA[cNtx][8];
  float vDA[cNtx][tile_C::ne / 2][8];
  float vMA[cNtx][tile_C::ne / 2][8];

#pragma unroll
  for (int vN = 0; vN < cNtx; ++vN) {
#pragma unroll
    for (int vK01 = 0; vK01 < MMQ_TILE_NE_K; vK01 += QI8_1) {
      const int cK0 = pK00 + vK01;

      load_ldmatrix(((tile_A_8 *)vA[vN])[vK01 / QI8_1], vXQs + (cI0 + vN * tile_A::I) * cSramStride + cK0, cSramStride);
    }
  }

#pragma unroll
  for (int vN = 0; vN < cNtx; ++vN) {
#pragma unroll
    for (int vL = 0; vL < tile_C::ne / 2; ++vL) {
      const int i = cI0 + vN * tile_C::I + tile_C::get_i(2 * vL);

#pragma unroll
      for (int vK01 = 0; vK01 < MMQ_TILE_NE_K; vK01 += QI8_1 / 2) {
        const int cK0 = pK00 + vK01;

        const float2 cDm = __half22float2(vXDm[i * cSramStride + cK0 / (QI8_1 / 2)]);

        vDA[vN][vL][vK01 / (QI8_1 / 2)] = cDm.x;
        vMA[vN][vL][vK01 / (QI8_1 / 2)] = cDm.y;
      }
    }
  }

#pragma unroll
  for (int vJ0 = 0; vJ0 < J; vJ0 += cNtx * tile_C::J) {
    float2 vDB[tile_C::ne / 2];

#pragma unroll
    for (int vL = 0; vL < tile_C::ne / 2; ++vL) {
      const int cJ = vJ0 + tile_C::get_j(vL);

      vDB[vL] = __half22float2(vYDs[cJ * MMQ_TILE_Y_K]);
    }

#pragma unroll
    for (int vK01 = 0; vK01 < MMQ_TILE_NE_K; vK01 += QI8_1) {
      tile_B vB[2];

      // Here load_generic is faster than load_ldmatrix.
      load_generic(vB[0], vYQs + vJ0 * MMQ_TILE_Y_K + (vK01 + 0), MMQ_TILE_Y_K);
      load_generic(vB[1], vYQs + vJ0 * MMQ_TILE_Y_K + (vK01 + tile_B::J), MMQ_TILE_Y_K);

      tile_C vCm[2];
      if (vK01 >= MMQ_TILE_NE_K * 3 / 4) {
        tile_A vA1;
        vA1.x[0] = 0x01010101;
        vA1.x[1] = 0x01010101;
        mma(vCm[0], vA1, vB[0]);
        mma(vCm[1], vA1, vB[1]);
      }

#pragma unroll
      for (int vN = 0; vN < cNtx; ++vN) {
        tile_C vCd[2];

        mma(vCd[0], vA[vN][vK01 / 4 + 0], vB[0]);
        mma(vCd[1], vA[vN][vK01 / 4 + 1], vB[1]);

#pragma unroll
        for (int vL = 0; vL < tile_C::ne; ++vL) {
          float vTmp = vCd[0].x[vL] * vDA[vN][vL / 2][vK01 / 4 + 0] + vCd[1].x[vL] * vDA[vN][vL / 2][vK01 / 4 + 1];
          if (vK01 >= MMQ_TILE_NE_K * 3 / 4) {
            vTmp -= vCm[0].x[vL] * vMA[vN][vL / 2][vK01 / 4 + 0] + vCm[1].x[vL] * vMA[vN][vL / 2][vK01 / 4 + 1];
          }
          pSum[(vJ0 / tile_C::J + vN) * tile_C::ne + vL] +=
              vTmp * (vK01 < MMQ_TILE_NE_K / 2 ? vDB[vL % 2].x : vDB[vL % 2].y);
        }
      }
    }

#pragma unroll
    for (int vK01 = 0; vK01 < MMQ_TILE_NE_K * 3 / 4; vK01 += QI8_1) {
      float2 vSB[tile_C::ne / 2];

#pragma unroll
      for (int vL = 0; vL < tile_C::ne / 2; ++vL) {
        const int cJ = vJ0 + tile_C::get_j(vL);

        vSB[vL] = __half22float2(vYDs[cJ * MMQ_TILE_Y_K + (1 + vK01 / QI8_1)]);
      }

#pragma unroll
      for (int vN = 0; vN < cNtx; ++vN) {
#pragma unroll
        for (int vL = 0; vL < tile_C::ne; ++vL) {
          pSum[(vJ0 / tile_C::J + vN) * tile_C::ne + vL] -= vMA[vN][vL / 2][vK01 / 4 + 0] * vSB[vL % 2].x;
          pSum[(vJ0 / tile_C::J + vN) * tile_C::ne + vL] -= vMA[vN][vL / 2][vK01 / 4 + 1] * vSB[vL % 2].y;
        }
      }
    }
  }
#else
  GGML_UNUSED_VARS(pX, pY, pSum, pK00);
  NO_DEVICE_CODE;
#endif // AMD_MFMA_AVAILABLE || AMD_WMMA_AVAILABLE
}

template <ggml_type type, int J, bool fallback>
static __device__ __forceinline__ void ggml_cuda_mmq_vec_dot_q3_K_q8_1_dp4a(const int *__restrict__ pX,
                                                                            const int *__restrict__ pY,
                                                                            float *__restrict__ pSum, const int pK00) {
  constexpr int cWarpSize = ggml_cuda_get_physical_warp_size();
  constexpr int cNwarps = ggml_cuda_mmq_get_nthreads(type, J, fallback) / cWarpSize;
  constexpr int I = ggml_cuda_mmq_get_I(type, J, fallback);

  constexpr tile_x_sizes cTxs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_Q3_K, I);
  const int *vXQs = (const int *)pX;
  const float *vXDf = (const float *)vXQs + cTxs.qs;
  const int *vXSc = (const int *)vXDf + cTxs.dm;
  const int *vYQs = (const int *)pY + 4;
  const float *vYDf = (const float *)pY;

  // #pragma unroll
  for (int vK01 = 0; vK01 < MMQ_TILE_NE_K; vK01 += QR3_K * VDR_Q3_K_Q8_1_MMQ) {
    const int cK0 = pK00 + vK01;

#pragma unroll
    for (int vJ0 = 0; vJ0 < J; vJ0 += cNwarps) {
      const int cJ = vJ0 + threadIdx.y;

#pragma unroll
      for (int vI0 = 0; vI0 < I; vI0 += cWarpSize) {
        const int i = vI0 + threadIdx.x;

        const int8_t *vScales = ((const int8_t *)(vXSc + i * (MMQ_TILE_NE_K / 8) + i / 8)) + cK0 / 4;

        pSum[vJ0 / cNwarps * I / cWarpSize + vI0 / cWarpSize] +=
            vec_dot_q3_K_q8_1_impl_mmq(&vXQs[i * (2 * MMQ_TILE_NE_K + 1) + cK0], &vYQs[cJ * MMQ_TILE_Y_K + vK01],
                                       vScales, vXDf[i], vYDf[cJ * MMQ_TILE_Y_K + vK01 / QI8_1]);
      }
    }
  }
}

template <ggml_type type, int J, bool fallback>
static __device__ __forceinline__ void ggml_cuda_mmq_vec_dot_q4_K_q8_1_dp4a(const int *__restrict__ pX,
                                                                            const int *__restrict__ pY,
                                                                            float *__restrict__ pSum, const int pK00) {
  constexpr int cWarpSize = ggml_cuda_get_physical_warp_size();
  constexpr int cNwarps = ggml_cuda_mmq_get_nthreads(type, J, fallback) / cWarpSize;
  constexpr int I = ggml_cuda_mmq_get_I(type, J, fallback);

  constexpr tile_x_sizes cTxs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_Q4_K, I);
  const int *vXQs = (const int *)pX;
  const half2 *vXDm = (const half2 *)vXQs + cTxs.qs;
  const int *vXSc = (const int *)vXDm + cTxs.dm;
  const int *vYQs = (const int *)pY + 4;
  const half2 *vYDs = (const half2 *)pY;

  // #pragma unroll
  for (int vK01 = 0; vK01 < MMQ_TILE_NE_K; vK01 += QR4_K * VDR_Q4_K_Q8_1_MMQ) {
    const int cK0 = pK00 + vK01;

#pragma unroll
    for (int vJ0 = 0; vJ0 < J; vJ0 += cNwarps) {
      const int cJ = vJ0 + threadIdx.y;

#pragma unroll
      for (int vI0 = 0; vI0 < I; vI0 += cWarpSize) {
        const int i = vI0 + threadIdx.x;

        const uint8_t *vSc = (const uint8_t *)&vXSc[i * (MMQ_TILE_NE_K / 8) + i / 8 + cK0 / 32] + 2 * (vK01 / 16);

        pSum[vJ0 / cNwarps * I / cWarpSize + vI0 / cWarpSize] +=
            vec_dot_q4_K_q8_1_impl_mmq(&vXQs[i * (MMQ_TILE_NE_K + 1) + cK0 / 2], &vYQs[cJ * MMQ_TILE_Y_K + vK01], vSc,
                                       vSc + 8, vXDm[i], &vYDs[cJ * MMQ_TILE_Y_K + vK01 / QI8_1]);
      }
    }
  }
}

template <ggml_type type, int J, bool fallback>
static __device__ __forceinline__ void ggml_cuda_mmq_vec_dot_q5_K_q8_1_dp4a(const int *__restrict__ pX,
                                                                            const int *__restrict__ pY,
                                                                            float *__restrict__ pSum, const int pK00) {
  constexpr int cWarpSize = ggml_cuda_get_physical_warp_size();
  constexpr int cNwarps = ggml_cuda_mmq_get_nthreads(type, J, fallback) / cWarpSize;
  constexpr int I = ggml_cuda_mmq_get_I(type, J, fallback);

  constexpr tile_x_sizes cTxs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_Q5_K, I);
  const int *vXQs = (const int *)pX;
  const half2 *vXDm = (const half2 *)vXQs + cTxs.qs;
  const int *vXSc = (const int *)vXDm + cTxs.dm;
  const int *vYQs = (const int *)pY + 4;
  const half2 *vYDs = (const half2 *)pY;

  // #pragma unroll
  for (int vK01 = 0; vK01 < MMQ_TILE_NE_K; vK01 += QR5_K * VDR_Q5_K_Q8_1_MMQ) {
    const int cK0 = pK00 + vK01;

#pragma unroll
    for (int vJ0 = 0; vJ0 < J; vJ0 += cNwarps) {
      const int cJ = vJ0 + threadIdx.y;

#pragma unroll
      for (int vI0 = 0; vI0 < I; vI0 += cWarpSize) {
        const int i = vI0 + threadIdx.x;

        const uint8_t *vSc = ((const uint8_t *)&vXSc[i * (MMQ_TILE_NE_K / 8) + i / 8 + pK00 / 32]) + 2 * (vK01 / 16);

        pSum[vJ0 / cNwarps * I / cWarpSize + vI0 / cWarpSize] +=
            vec_dot_q5_K_q8_1_impl_mmq(&vXQs[i * (QR5_K * MMQ_TILE_NE_K + 1) + cK0], &vYQs[cJ * MMQ_TILE_Y_K + vK01],
                                       vSc, vSc + 8, vXDm[i], &vYDs[cJ * MMQ_TILE_Y_K + vK01 / QI8_1]);
      }
    }
  }
}

template <ggml_type type, int J, bool fallback>
static __device__ __forceinline__ void ggml_cuda_mmq_vec_dot_q6_K_q8_1_dp4a(const int *__restrict__ pX,
                                                                            const int *__restrict__ pY,
                                                                            float *__restrict__ pSum, const int pK00) {
  constexpr int cWarpSize = ggml_cuda_get_physical_warp_size();
  constexpr int cNwarps = ggml_cuda_mmq_get_nthreads(type, J, fallback) / cWarpSize;
  constexpr int I = ggml_cuda_mmq_get_I(type, J, fallback);

  constexpr tile_x_sizes cTxs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_Q6_K, I);
  const int *vXQs = (const int *)pX;
  const float *vXDf = (const float *)vXQs + cTxs.qs;
  const int *vXSc = (const int *)vXDf + cTxs.dm;
  const int *vYQs = (const int *)pY + 4;
  const float *vYDf = (const float *)pY;

  // #pragma unroll
  for (int vK01 = 0; vK01 < MMQ_TILE_NE_K; vK01 += QR6_K * VDR_Q6_K_Q8_1_MMQ) {
    const int cK0 = pK00 + vK01;

#pragma unroll
    for (int vJ0 = 0; vJ0 < J; vJ0 += cNwarps) {
      const int cJ = vJ0 + threadIdx.y;

#pragma unroll
      for (int vI0 = 0; vI0 < I; vI0 += cWarpSize) {
        const int i = vI0 + threadIdx.x;

        const int8_t *vSc = ((const int8_t *)&vXSc[i * (MMQ_TILE_NE_K / 8) + i / 8 + cK0 / 16]);

        pSum[vJ0 / cNwarps * I / cWarpSize + vI0 / cWarpSize] += vec_dot_q6_K_q8_1_impl_mmq(
            &vXQs[i * (QR6_K * MMQ_TILE_NE_K + 1) + cK0], &vYQs[cJ * MMQ_TILE_Y_K + vK01], vSc,
            vXDf[i * (MMQ_TILE_NE_K / QI6_K) + i / QI6_K], &vYDf[cJ * MMQ_TILE_Y_K + vK01 / QI8_1]);
      }
    }
  }
}

template <ggml_type type, int J, bool fallback>
static __device__ __forceinline__ void ggml_cuda_mmq_vec_dot_q6_K_q8_1_mma(const int *__restrict__ pX,
                                                                           const int *__restrict__ pY,
                                                                           float *__restrict__ pSum, const int pK00) {
#if defined(AMD_MFMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
  constexpr data_layout input_layout = get_input_data_layout();
  typedef tile<16, 4, int, input_layout> tile_A;
  typedef tile<16, 4, int, input_layout> tile_B;
  typedef tile<16, 16, int, DATA_LAYOUT_J_MAJOR> tile_C;

  constexpr int I = ggml_cuda_mmq_get_I(type, J, fallback);
  constexpr int cSramStride = ggml_cuda_mmq_get_sram_stride(type, J, fallback);
  constexpr int cRowsPerWarp = ggml_cuda_mmq_get_rows_per_warp(type, J, fallback);
  constexpr int cNtx = cRowsPerWarp / tile_C::I; // Number of x minitiles per warp.

  pY += (threadIdx.y % cNtx) * (tile_C::J * MMQ_TILE_Y_K);

  const int *vXQs = (const int *)pX;
  const float *vXDf = (const float *)vXQs + MMQ_TILE_NE_K * 2;
  const int *vXSc = (const int *)vXDf + MMQ_TILE_NE_K / QI6_K;
  const int *vYQs = (const int *)pY + 4;
  const float *vYDf = (const float *)pY;

  const int cI0 = (threadIdx.y / cNtx) * cRowsPerWarp;

  for (int vK01 = 0; vK01 < MMQ_TILE_NE_K; vK01 += 4) {
    const int cK0 = pK00 + vK01;

    tile_A vA[cNtx];
#pragma unroll
    for (int vN = 0; vN < cNtx; ++vN) {
      load_ldmatrix(vA[vN], vXQs + (cI0 + vN * tile_A::I) * cSramStride + cK0, cSramStride);
    }

#pragma unroll
    for (int vJ0 = 0; vJ0 < J; vJ0 += cNtx * tile_C::J) {
      tile_B vB;
      load_ldmatrix(vB, vYQs + vJ0 * MMQ_TILE_Y_K + vK01, MMQ_TILE_Y_K);

      const int cJ = vJ0 + tile_C::get_j(0);
      const float vDB = vYDf[cJ * MMQ_TILE_Y_K + vK01 / QI8_1];

#pragma unroll
      for (int vN = 0; vN < cNtx; ++vN) {
        tile_C vC;
        mma(vC, vA[vN], vB);

#pragma unroll
        for (int vL = 0; vL < tile_C::ne; ++vL) {
          const int i = cI0 + vN * tile_C::I + tile_C::get_i(vL);
          const int8_t *vSc = (const int8_t *)(vXSc + i * cSramStride + pK00 / 16);
          pSum[(vJ0 / tile_C::J + vN) * tile_C::ne + vL] += vC.x[vL] * vSc[vK01 / 4] * vXDf[i * cSramStride] * vDB;
        }
      }
    }
  }
#elif defined(TURING_MMA_AVAILABLE)

  typedef tile<16, 4, int> tile_A;
  typedef tile<8, 4, int> tile_B;
  typedef tile<16, 8, int> tile_C;

  constexpr int I = ggml_cuda_mmq_get_I(type, J, fallback);
  constexpr int cSramStride = ggml_cuda_mmq_get_sram_stride(type, J, fallback);
  constexpr int cRowsPerWarp = ggml_cuda_mmq_get_rows_per_warp(type, J, fallback);
  constexpr int cNtx = cRowsPerWarp / tile_C::I; // Number of x minitiles per warp.

  pY += (threadIdx.y % cNtx) * (tile_C::J * MMQ_TILE_Y_K);

  const int *vXQs = (const int *)pX;
  const float *vXDf = (const float *)vXQs + MMQ_TILE_NE_K * 2;
  const int *vXSc = (const int *)vXDf + MMQ_TILE_NE_K / QI6_K;
  const int *vYQs = (const int *)pY + 4;
  const float *vYDf = (const float *)pY;

  const int cI0 = (threadIdx.y / cNtx) * (cNtx * tile_A::I);

  tile_A vA[cNtx][8];
  int vScA[cNtx][tile_C::ne / 2][8];
  float vDA[cNtx][tile_C::ne / 2];

#pragma unroll
  for (int vN = 0; vN < cNtx; ++vN) {
#pragma unroll
    for (int vK01 = 0; vK01 < MMQ_TILE_NE_K; vK01 += 8) {
      const int cK0 = pK00 + vK01;

      load_ldmatrix(vA[vN][vK01 / 4 + 0], vXQs + (cI0 + vN * tile_A::I) * cSramStride + (cK0 + 0), cSramStride);
      load_ldmatrix(vA[vN][vK01 / 4 + 1], vXQs + (cI0 + vN * tile_A::I) * cSramStride + (cK0 + tile_A::J), cSramStride);
    }

#pragma unroll
    for (int vK01 = 0; vK01 < MMQ_TILE_NE_K; vK01 += 16) {
      const int cK0 = pK00 + vK01;

#pragma unroll
      for (int vL = 0; vL < tile_C::ne / 2; ++vL) {
        const int i = cI0 + vN * tile_C::I + tile_C::get_i(2 * vL);

        const int cScPacked = vXSc[i * cSramStride + cK0 / 16];
        const int8_t *vSc = (const int8_t *)&cScPacked;

#pragma unroll
        for (int vKsc = 0; vKsc < sizeof(int); ++vKsc) {
          vScA[vN][vL][vK01 / 4 + vKsc] = vSc[vKsc];
        }
      }
    }

#pragma unroll
    for (int vL = 0; vL < tile_C::ne / 2; ++vL) {
      const int i = cI0 + vN * tile_C::I + tile_C::get_i(2 * vL);

      vDA[vN][vL] = vXDf[i * cSramStride];
    }
  }

#pragma unroll
  for (int vJ0 = 0; vJ0 < J; vJ0 += cNtx * tile_C::J) {
    float vTmp[cNtx][tile_C::ne] = {{0.0f}};

#pragma unroll
    for (int vK01 = 0; vK01 < MMQ_TILE_NE_K; vK01 += 8) {
      tile_B vB[2];
      float vDB[tile_C::ne / 2];

      // Here load_generic is faster than load_ldmatrix.
      load_generic(vB[0], vYQs + vJ0 * MMQ_TILE_Y_K + 0 + vK01, MMQ_TILE_Y_K);
      load_generic(vB[1], vYQs + vJ0 * MMQ_TILE_Y_K + tile_B::J + vK01, MMQ_TILE_Y_K);

#pragma unroll
      for (int vL = 0; vL < tile_C::ne / 2; ++vL) {
        const int cJ = vJ0 + tile_C::get_j(vL);

        vDB[vL] = vYDf[cJ * MMQ_TILE_Y_K + vK01 / QI8_1];
      }

#pragma unroll
      for (int vN = 0; vN < cNtx; ++vN) {
        tile_C vC[2];
        mma(vC[0], vA[vN][vK01 / 4 + 0], vB[0]);
        mma(vC[1], vA[vN][vK01 / 4 + 1], vB[1]);

#pragma unroll
        for (int vL = 0; vL < tile_C::ne; ++vL) {
          vTmp[vN][vL] +=
              (vC[0].x[vL] * vScA[vN][vL / 2][vK01 / 4 + 0] + vC[1].x[vL] * vScA[vN][vL / 2][vK01 / 4 + 1]) *
              vDB[vL % 2];
        }
      }
    }

#pragma unroll
    for (int vN = 0; vN < cNtx; ++vN) {
#pragma unroll
      for (int vL = 0; vL < tile_C::ne; ++vL) {
        pSum[(vJ0 / tile_C::J + vN) * tile_C::ne + vL] += vTmp[vN][vL] * vDA[vN][vL / 2];
      }
    }
  }
#else
  GGML_UNUSED_VARS(pX, pY, pSum, pK00);
  NO_DEVICE_CODE;
#endif // AMD_MFMA_AVAILABLE || AMD_WMMA_AVAILABLE
}

// ---------------------------------------------------------------------------------------------

// Shared MMA kernel for MXFP4 and NVFP4 on Blackwell.
// Both quantizations encode values as e2m1 (FP4) and produce one uint32 scale per
// m16n8k64 MMA call; only the PTX kind (scale_vec::2X ue8m0 vs scale_vec::4X ue4m3)
// and the per-type stride constant differ.
template <ggml_type type, int J, bool fallback>
static __device__ __forceinline__ void ggml_cuda_mmq_vec_dot_fp4_fp4_mma(const int *__restrict__ pX,
                                                                         const int *__restrict__ pY,
                                                                         float *__restrict__ pSum, const int pK00) {

  typedef tile<16, 8, int> tile_A;
  typedef tile<8, 8, int> tile_B;
  typedef tile<16, 8, float> tile_C;

  constexpr int cI = ggml_cuda_mmq_get_I(type, J, fallback);
  constexpr int cSramStride = ggml_cuda_mmq_get_sram_stride(type, J, fallback);
  constexpr int cRowsPerWarp = ggml_cuda_mmq_get_rows_per_warp(type, J, fallback);
  constexpr int cNtx = cRowsPerWarp / tile_C::I;
  constexpr int cNfrags = MMQ_TILE_NE_K / tile_A::J;

  pY += (threadIdx.y % cNtx) * (tile_C::J * MMQ_TILE_Y_K);

  const int *vXQs = (const int *)pX;
  const uint32_t *vXSc = (const uint32_t *)(vXQs + 2 * MMQ_TILE_NE_K);
  const int *vYQs = (const int *)pY + 4;
  const uint32_t *vYSc = (const uint32_t *)pY;

  // 2 threads per quad supply the packed scale register to the block_scale MMA,
  // see https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-block-scaling
  const int cTidxA = threadIdx.x / 4 + (threadIdx.x % 2) * 8;
  const int cTidxB = threadIdx.x / 4;
  const int cI0 = (threadIdx.y / cNtx) * cRowsPerWarp;

  tile_A vA[cNtx][cNfrags];
  uint32_t vScaleA[cNtx][cNfrags];

#pragma unroll
  for (int vN = 0; vN < cNtx; ++vN) {
#pragma unroll
    for (int vFrag = 0; vFrag < cNfrags; ++vFrag) {
      const int cK0 = pK00 + vFrag * tile_A::J;
      load_ldmatrix(vA[vN][vFrag], vXQs + (cI0 + vN * tile_A::I) * cSramStride + cK0, cSramStride);
      vScaleA[vN][vFrag] = vXSc[(cI0 + vN * tile_A::I + cTidxA) * cSramStride + cK0 / tile_A::J];
    }
  }

#pragma unroll
  for (int vJ0 = 0; vJ0 < J; vJ0 += cNtx * tile_C::J) {
    tile_B vB[cNfrags];
    uint32_t vScaleB[cNfrags];

#pragma unroll
    for (int vFrag = 0; vFrag < cNfrags; ++vFrag) {
      const int cK0 = vFrag * tile_B::J;
      load_generic(vB[vFrag], vYQs + vJ0 * MMQ_TILE_Y_K + cK0, MMQ_TILE_Y_K);
      vScaleB[vFrag] = vYSc[(vJ0 + cTidxB) * MMQ_TILE_Y_K + vFrag];
    }

#pragma unroll
    for (int vN = 0; vN < cNtx; ++vN) {
#pragma unroll
      for (int vFrag = 0; vFrag < cNfrags; ++vFrag) {
        tile_C vC = {};
        mma_block_scaled_fp4<type>(vC, vA[vN][vFrag], vB[vFrag], vScaleA[vN][vFrag], vScaleB[vFrag]);
#pragma unroll
        for (int vL = 0; vL < tile_C::ne; ++vL) {
          pSum[(vJ0 / tile_C::J + vN) * tile_C::ne + vL] += vC.x[vL];
        }
      }
    }
  }
}
