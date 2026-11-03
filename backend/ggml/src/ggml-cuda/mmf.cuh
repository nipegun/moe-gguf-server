#pragma once

#include "mma.cuh"
#include "common.cuh"
#include "convert.cuh"

using namespace ggml_cuda_mma;

#define MMF_ROWS_PER_BLOCK 32
#define MMF_ROWS_PER_BLOCK_CDNA 64

static __forceinline__ int64_t mmf_get_max_block_size(int pCc) {
  if (GGML_CUDA_CC_IS_CDNA(pCc)) {
    return 512;
  } else {
    return 256;
  }
}

static __forceinline__ int mmf_get_padding(int pCc) {
  if (GGML_CUDA_CC_IS_CDNA(pCc)) {
    return 2;
  } else {
    return 4;
  }
}

static constexpr __device__ int mmf_get_padding() {
#if defined(AMD_MFMA_AVAILABLE)
  return 2;
#else
  return 4;
#endif // defined(AMD_MFMA_AVAILABLE)
}

struct mmf_ids_data {
  const int32_t *ids_src_compact = nullptr;
  const int32_t *ids_dst_compact = nullptr;
  const int32_t *expert_bounds_dev = nullptr;
  int n_experts = 0;
  int sis1 = 0;
};

void ggml_cuda_mul_mat_f(ggml_backend_cuda_context &pCtx, const ggml_tensor *pSrc0, const ggml_tensor *pSrc1,
                         const ggml_tensor *pIds, ggml_tensor *pDst);

bool ggml_cuda_should_use_mmf(enum ggml_type pType, int pCc, int pWarpSize, const int64_t *pScr0Ne,
                              const size_t *pSrc0Nb, const int pSrc1Ncols, bool pMulMatId);

template <typename T, int rows_per_block, int cols_per_block, int nwarps, bool has_ids>
__launch_bounds__(ggml_cuda_get_physical_warp_size() * nwarps, 1) static __global__
    void mul_mat_f(const T *__restrict__ pX, const float *__restrict__ pY, const int32_t *__restrict__ pIds,
                   float *__restrict__ pDst, const int pNcols, const int pNcolsDstTotal, const int pNchannelsDst,
                   const int pStrideRow, const int pStrideColY, const int pStrideColDst, const int pStrideColId,
                   const int pStrideRowId, const int pChannelRatio, const int pStrideChannelX,
                   const int pStrideChannelY, const int pStrideChannelDst, const int pSampleRatio,
                   const int pStrideSampleX, const int pStrideSampleY, const int pStrideSampleDst) {
// TODO: handle this in a consistent and simpler way after AMD MFMA support has been added
#if defined(VOLTA_MMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE) ||                    \
    defined(AMD_MFMA_AVAILABLE)
#if defined(AMD_WMMA_AVAILABLE)
  if constexpr (!(std::is_same_v<T, half2> || std::is_same_v<T, nv_bfloat162>) ||
                rows_per_block != MMF_ROWS_PER_BLOCK) {
    NO_DEVICE_CODE;
  } else {
    typedef tile<16, 8, T, get_input_data_layout()> tile_A;
    typedef tile<16, 8, T, get_input_data_layout()> tile_B;
    typedef tile<16, 16, float, DATA_LAYOUT_J_MAJOR> tile_C;
#elif defined(AMD_MFMA_AVAILABLE)
  if constexpr (rows_per_block != MMF_ROWS_PER_BLOCK_CDNA) {
    NO_DEVICE_CODE;
  } else {
    typedef tile<16, 8, T, DATA_LAYOUT_I_MAJOR> tile_A;
    typedef tile<16, 8, T, DATA_LAYOUT_I_MAJOR> tile_B;
    typedef tile<16, 16, float, DATA_LAYOUT_J_MAJOR> tile_C;
#else
#ifdef VOLTA_MMA_AVAILABLE
  if constexpr (!std::is_same_v<T, half2> || rows_per_block != MMF_ROWS_PER_BLOCK) {
    NO_DEVICE_CODE;
  } else {
    typedef tile<32, 4, T, DATA_LAYOUT_I_MAJOR> tile_A;
    typedef tile<8, 4, T, DATA_LAYOUT_I_MAJOR_MIRRORED> tile_B;
    typedef tile<32, 8, float, DATA_LAYOUT_I_MAJOR> tile_C;
#else
  if constexpr (rows_per_block != MMF_ROWS_PER_BLOCK) {
    NO_DEVICE_CODE;
  } else {
    typedef tile<16, 8, T> tile_A;
    typedef tile<8, 8, T> tile_B;
    typedef tile<16, 8, float> tile_C;
#endif // VOLTA_MMA_AVAILABLE
#endif // defined(AMD_WMMA_AVAILABLE)
    if constexpr (!tile_A::supported() || !tile_B::supported() || !tile_C::supported()) {
      NO_DEVICE_CODE;
      return;
    }

    constexpr int cWarpSize = ggml_cuda_get_physical_warp_size();
    constexpr int cTileKPadded = cWarpSize + mmf_get_padding();
    constexpr int cNtA = rows_per_block / tile_A::I;
    constexpr int cNtB = (cols_per_block + tile_B::I - 1) / tile_B::I;

    const int cRow0 = blockIdx.x * rows_per_block;

    int vExpertIdx = 0;
    [[maybe_unused]] int vColBase = 0;

    const int cChannelDst = has_ids ? 0 : blockIdx.y;

    if constexpr (has_ids) {
      // experts + tiles of ncols_dst are packed in the y dimension
      int vColTiles = (pNcolsDstTotal + cols_per_block - 1) / cols_per_block;
      const int cNchannelsX = gridDim.y / vColTiles;
      const int cTileIdx = blockIdx.y / cNchannelsX;
      vExpertIdx = blockIdx.y - cTileIdx * cNchannelsX;
      vColBase = cTileIdx * cols_per_block;
    }

    const int cChannelX = has_ids ? vExpertIdx : (cChannelDst / pChannelRatio);
    const int cChannelY = cChannelDst;
    const int cSampleDst = blockIdx.z;
    const int cSampleX = cSampleDst / pSampleRatio;
    const int cSampleY = cSampleDst;

    pX += int64_t(cSampleX) * pStrideSampleX + cChannelX * pStrideChannelX + cRow0 * pStrideRow;
    pY += int64_t(cSampleY) * pStrideSampleY + (has_ids ? 0 : cChannelY * pStrideChannelY);
    pDst += int64_t(cSampleDst) * pStrideSampleDst + (has_ids ? 0 : cChannelDst * pStrideChannelDst);

    if constexpr (has_ids) {
      constexpr int cYStrideScale = std::is_same_v<T, float> ? 1 : 2;
      const int64_t cColOffset = vColBase;
      pY += cColOffset * pStrideColY * cYStrideScale;
      pDst += cColOffset * pStrideColDst;
      pIds += cColOffset * pStrideRowId;
    }

    [[maybe_unused]] const float2 *vY2 = (const float2 *)pY;

    extern __shared__ char vDataMmv[];

    char *vShmemBase = vDataMmv;
    [[maybe_unused]] int *vSlotMap = (int *)vShmemBase;
    char *vComputeBase = has_ids ? (vShmemBase + GGML_PAD(cols_per_block, 16) * sizeof(int)) : vShmemBase;

    tile_C vC[cNtA][cNtB];

    T *vTileXy = (T *)vComputeBase + threadIdx.y * (tile_A::I * cTileKPadded);

    if constexpr (has_ids) {
      int vFound = 0;

      for (int vJ0 = 0; vJ0 < cols_per_block; vJ0 += nwarps) {
        const int cJ = vJ0 + threadIdx.y;

        if (threadIdx.x == 0) {
          vSlotMap[cJ] = -1;
        }
        ggml_cuda_syncwarp();

        if (vColBase + cJ >= pNcolsDstTotal) {
          continue;
        }

        const int32_t *__restrict__ vIdRow = pIds + cJ * pStrideRowId;

        for (int vK = threadIdx.x; vK < pNchannelsDst; vK += cWarpSize) {
          int vMatch = vIdRow[vK * pStrideColId] == vExpertIdx;

          if (vMatch) {
            vSlotMap[cJ] = vK;
            vFound = 1;
            break;
          }
        }
      }

      if (!__syncthreads_or(vFound)) {
        return;
      }
    }

    for (int vCol = threadIdx.y * cWarpSize + threadIdx.x; vCol < pNcols; vCol += nwarps * cWarpSize) {
      tile_A vA[cNtA][cWarpSize / tile_A::J];
#pragma unroll
      for (int vItA = 0; vItA < cNtA; ++vItA) {
        ggml_cuda_syncwarp();
#pragma unroll
        for (int vI = 0; vI < tile_A::I; ++vI) {
          vTileXy[vI * cTileKPadded + threadIdx.x] = pX[(vItA * tile_A::I + vI) * pStrideRow + vCol];
        }
        ggml_cuda_syncwarp();
#pragma unroll
        for (int vK0 = 0; vK0 < cWarpSize; vK0 += tile_A::J) {
          load_ldmatrix(vA[vItA][vK0 / tile_A::J], vTileXy + vK0, cTileKPadded);
        }
      }

#pragma unroll
      for (int vItB = 0; vItB < cNtB; ++vItB) {
        ggml_cuda_syncwarp();
        if constexpr (std::is_same_v<T, float>) {
#pragma unroll
          for (int vJ0 = 0; vJ0 < tile_B::I; ++vJ0) {
            const int cJ = vJ0 + vItB * tile_B::I;

            if constexpr (!has_ids) {
              vTileXy[vJ0 * cTileKPadded + threadIdx.x] = cJ < cols_per_block ? pY[cJ * pStrideColY + vCol] : 0.0f;
            } else {
              const bool cValid = cJ < cols_per_block && (vColBase + cJ) < pNcolsDstTotal && vSlotMap[cJ] >= 0;
              vTileXy[vJ0 * cTileKPadded + threadIdx.x] =
                  cValid ? pY[vSlotMap[cJ] * pStrideChannelY + cJ * pStrideColY + vCol] : 0.0f;
            }
          }
        } else if constexpr (std::is_same_v<T, half2> || std::is_same_v<T, nv_bfloat162>) {
#pragma unroll
          for (int vJ0 = 0; vJ0 < tile_B::I; ++vJ0) {
            const int cJ = vJ0 + vItB * tile_B::I;

            if constexpr (!has_ids) {
              const float2 cTmp = cJ < cols_per_block ? vY2[cJ * pStrideColY + vCol] : make_float2(0.0f, 0.0f);
              vTileXy[vJ0 * cTileKPadded + threadIdx.x] = ggml_cuda_cast<T>(cTmp);
            } else {
              const bool cValid = cJ < cols_per_block && (vColBase + cJ) < pNcolsDstTotal && vSlotMap[cJ] >= 0;
              float2 vTmp = cValid
                                ? *(const float2 *)&pY[vSlotMap[cJ] * pStrideChannelY + 2 * (cJ * pStrideColY + vCol)]
                                : make_float2(0.0f, 0.0f);
              vTileXy[vJ0 * cTileKPadded + threadIdx.x] = ggml_cuda_cast<T>(vTmp);
            }
          }
        } else {
          static_assert(std::is_same_v<T, void>, "unsupported type");
        }
        ggml_cuda_syncwarp();
#pragma unroll
        for (int vK0 = 0; vK0 < cWarpSize; vK0 += tile_B::J) {
          tile_B vB;
          load_ldmatrix(vB, vTileXy + vK0, cTileKPadded);
#pragma unroll
          for (int vItA = 0; vItA < cNtA; ++vItA) {
            mma(vC[vItA][vItB], vA[vItA][vK0 / tile_B::J], vB);
          }
        }
      }
    }

    float *vBufIw = (float *)vComputeBase;
    constexpr int cKiw = nwarps * rows_per_block + mmf_get_padding();

    if (nwarps > 1) {
      __syncthreads();
    } else {
      ggml_cuda_syncwarp();
    }
#pragma unroll
    for (int vItB = 0; vItB < cNtB; ++vItB) {
#pragma unroll
      for (int vItA = 0; vItA < cNtA; ++vItA) {
#pragma unroll
        for (int vL = 0; vL < tile_C::ne; ++vL) {
          const int cI = threadIdx.y * rows_per_block + vItA * tile_C::I + tile_C::get_i(vL);
          const int cJ = vItB * tile_C::J + tile_C::get_j(vL);
          vBufIw[cJ * cKiw + cI] = vC[vItA][vItB].x[vL];
        }
      }
    }

    if (nwarps > 1) {
      __syncthreads();
    } else {
      ggml_cuda_syncwarp();
    }

#pragma unroll
    for (int vJ0 = 0; vJ0 < cols_per_block; vJ0 += nwarps) {
      const int cJ = vJ0 + threadIdx.y;

      if (vJ0 + nwarps > cols_per_block && cJ >= cols_per_block) {
        return;
      }

      float vSum[rows_per_block / cWarpSize] = {0.0f};
      static_assert((rows_per_block % cWarpSize) == 0, "rows_per_block must be a multiple of warp_size.");
#pragma unroll
      for (int vI0 = 0; vI0 < nwarps * rows_per_block; vI0 += rows_per_block) {
#pragma unroll
        for (int vI1 = 0; vI1 < sizeof(vSum) / sizeof(vSum[0]); ++vI1) {
          const int cI = vI0 + vI1 * cWarpSize + threadIdx.x;

          vSum[vI1] += vBufIw[cJ * cKiw + cI];
        }
      }

      if constexpr (!has_ids) {
#pragma unroll
        for (int vI0 = 0; vI0 < sizeof(vSum) / sizeof(vSum[0]); ++vI0) {
          pDst[cJ * pStrideColDst + cRow0 + vI0 * cWarpSize + threadIdx.x] = vSum[vI0];
        }
      } else {
        const int cSlot = (cJ < cols_per_block) ? vSlotMap[cJ] : -1;
        if (cSlot >= 0 && (vColBase + cJ) < pNcolsDstTotal) {
#pragma unroll
          for (int vI0 = 0; vI0 < sizeof(vSum) / sizeof(vSum[0]); ++vI0) {
            pDst[cSlot * pStrideChannelDst + cJ * pStrideColDst + cRow0 + vI0 * cWarpSize + threadIdx.x] = vSum[vI0];
          }
        }
      }
    }
  }
#else
  GGML_UNUSED_VARS(pX, pY, pIds, pDst, pNcols, pNcolsDstTotal, pNchannelsDst, pStrideRow, pStrideColY, pStrideColDst,
                   pStrideColId, pStrideRowId, pChannelRatio, pStrideChannelX, pStrideChannelY, pStrideChannelDst,
                   pSampleRatio, pStrideSampleX, pStrideSampleY, pStrideSampleDst);
  NO_DEVICE_CODE;
#endif // defined(VOLTA_MMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE) ||
       // defined(AMD_MFMA_AVAILABLE)
}

// This kernel is for larger batch sizes of mul_mat_id
template <typename T, int rows_per_block, int cols_per_block, int nwarps>
__launch_bounds__(ggml_cuda_get_physical_warp_size() * nwarps, 1) static __global__
    void mul_mat_f_ids(const T *__restrict__ pX, const float *__restrict__ pY,
                       const int32_t *__restrict__ pIdsSrcCompact, const int32_t *__restrict__ pIdsDstCompact,
                       const int32_t *__restrict__ pExpertBounds, float *__restrict__ pDst, const int pNcols,
                       const int pNcolsDstTotal, const int pNchannelsDst, const int pStrideRow, const int pStrideColY,
                       const int pStrideColDst, const int pChannelRatio, const int pStrideChannelX,
                       const int pStrideChannelY, const int pStrideChannelDst, const int pSampleRatio,
                       const int pStrideSampleX, const int pStrideSampleY, const int pStrideSampleDst,
                       const uint3 pSis1Fd, const uint3 pNchFd) {
// TODO: handle this in a consistent and simpler way after AMD MFMA support has been added
#if defined(VOLTA_MMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE) ||                    \
    defined(AMD_MFMA_AVAILABLE)
#if defined(AMD_WMMA_AVAILABLE)
  if constexpr (!(std::is_same_v<T, half2> || std::is_same_v<T, nv_bfloat162>) ||
                rows_per_block != MMF_ROWS_PER_BLOCK) {
    NO_DEVICE_CODE;
  } else {
    typedef tile<16, 8, T, get_input_data_layout()> tile_A;
    typedef tile<16, 8, T, get_input_data_layout()> tile_B;
    typedef tile<16, 16, float, DATA_LAYOUT_J_MAJOR> tile_C;
#elif defined(AMD_MFMA_AVAILABLE)
  if constexpr (rows_per_block != MMF_ROWS_PER_BLOCK_CDNA) {
    NO_DEVICE_CODE;
  } else {
    typedef tile<16, 8, T, DATA_LAYOUT_I_MAJOR> tile_A;
    typedef tile<16, 8, T, DATA_LAYOUT_I_MAJOR> tile_B;
    typedef tile<16, 16, float, DATA_LAYOUT_J_MAJOR> tile_C;
#else
#ifdef VOLTA_MMA_AVAILABLE
  if constexpr (!std::is_same_v<T, half2> || rows_per_block != MMF_ROWS_PER_BLOCK) {
    NO_DEVICE_CODE;
  } else {
    typedef tile<32, 4, T, DATA_LAYOUT_I_MAJOR> tile_A;
    typedef tile<8, 4, T, DATA_LAYOUT_I_MAJOR_MIRRORED> tile_B;
    typedef tile<32, 8, float, DATA_LAYOUT_I_MAJOR> tile_C;
#else
  if constexpr (rows_per_block != MMF_ROWS_PER_BLOCK) {
    NO_DEVICE_CODE;
  } else {
    typedef tile<16, 8, T> tile_A;
    typedef tile<8, 8, T> tile_B;
    typedef tile<16, 8, float> tile_C;
#endif // VOLTA_MMA_AVAILABLE
#endif // defined(AMD_WMMA_AVAILABLE)
    if constexpr (!tile_A::supported() || !tile_B::supported() || !tile_C::supported()) {
      NO_DEVICE_CODE;
      return;
    }

    constexpr int cWarpSize = ggml_cuda_get_physical_warp_size();
    constexpr int cTileKPadded = cWarpSize + mmf_get_padding();
    constexpr int cNtA = rows_per_block / tile_A::I;
    constexpr int cNtB = (cols_per_block + tile_B::I - 1) / tile_B::I;

    const int cRow0 = blockIdx.x * rows_per_block;

    const int cExpertIdx = blockIdx.y;
    const int cExpertStart = pExpertBounds[cExpertIdx];
    const int cExpertEnd = pExpertBounds[cExpertIdx + 1];
    const int cNcolsExpert = cExpertEnd - cExpertStart;

    const int cTilesForExpert = (cNcolsExpert + cols_per_block - 1) / cols_per_block;
    const int cTileIdx = blockIdx.z;
    if (cTileIdx >= cTilesForExpert) {
      return;
    }

    const int cColBase = cTileIdx * cols_per_block;

    GGML_UNUSED(pChannelRatio);

    const int cChannelX = cExpertIdx;
    const int cSampleDst = 0;
    const int cSampleX = cSampleDst / pSampleRatio;
    const int cSampleY = cSampleDst;

    pX += int64_t(cSampleX) * pStrideSampleX + cChannelX * pStrideChannelX + cRow0 * pStrideRow;
    pY += int64_t(cSampleY) * pStrideSampleY;
    pDst += int64_t(cSampleDst) * pStrideSampleDst;

    const int32_t *vIdsSrcExpert = pIdsSrcCompact + cExpertStart;
    const int32_t *vIdsDstExpert = pIdsDstCompact + cExpertStart;

    extern __shared__ char vDataMmv[];
    char *vComputeBase = vDataMmv;

    // const float2 * y2 = (const float2 *) y;

    tile_C vC[cNtA][cNtB];

    T *vTileXy = (T *)vComputeBase + threadIdx.y * (tile_A::I * cTileKPadded);

    for (int vCol = threadIdx.y * cWarpSize + threadIdx.x; vCol < pNcols; vCol += nwarps * cWarpSize) {
      tile_A vA[cNtA][cWarpSize / tile_A::J];
#pragma unroll
      for (int vItA = 0; vItA < cNtA; ++vItA) {
        ggml_cuda_syncwarp();
#pragma unroll
        for (int vI = 0; vI < tile_A::I; ++vI) {
          vTileXy[vI * cTileKPadded + threadIdx.x] = pX[(vItA * tile_A::I + vI) * pStrideRow + vCol];
        }
        ggml_cuda_syncwarp();
#pragma unroll
        for (int vK0 = 0; vK0 < cWarpSize; vK0 += tile_A::J) {
          load_ldmatrix(vA[vItA][vK0 / tile_A::J], vTileXy + vK0, cTileKPadded);
        }
      }

      if constexpr (std::is_same_v<T, float>) {
        float vValsBuf[2][tile_B::I];
        auto vGatherTile = [&](int pTileIdxLocal, float *pVals) {
#pragma unroll
          for (int vJ0 = 0; vJ0 < tile_B::I; ++vJ0) {
            const int cJ = vJ0 + pTileIdxLocal * tile_B::I;
            const int cGlobalJ = cColBase + cJ;
            float vVal = 0.0f;
            if (cJ < cols_per_block && cGlobalJ < cNcolsExpert) {
              const int cSrcEntry = vIdsSrcExpert[cGlobalJ];
              const uint2 cQrm = fast_div_modulo((uint32_t)cSrcEntry, pSis1Fd);
              const int cToken = (int)cQrm.x;
              const int cChannel = (int)cQrm.y;
              if (cToken < pNcolsDstTotal) {
                vVal = pY[cChannel * pStrideChannelY + cToken * pStrideColY + vCol];
              }
            }
            pVals[vJ0] = vVal;
          }
        };

        vGatherTile(0, vValsBuf[0]);

        int vCurrBuf = 0;
        int vNextBuf = 1;
#pragma unroll
        for (int vItB = 0; vItB < cNtB; ++vItB) {
          ggml_cuda_syncwarp();
#pragma unroll
          for (int vJ0 = 0; vJ0 < tile_B::I; ++vJ0) {
            vTileXy[vJ0 * cTileKPadded + threadIdx.x] = vValsBuf[vCurrBuf][vJ0];
          }

          if (vItB + 1 < cNtB) {
            vGatherTile(vItB + 1, vValsBuf[vNextBuf]);
          }

          ggml_cuda_syncwarp();
#pragma unroll
          for (int vK0 = 0; vK0 < cWarpSize; vK0 += tile_B::J) {
            tile_B vB;
            load_ldmatrix(vB, vTileXy + vK0, cTileKPadded);
#pragma unroll
            for (int vItA = 0; vItA < cNtA; ++vItA) {
              mma(vC[vItA][vItB], vA[vItA][vK0 / tile_B::J], vB);
            }
          }

          if (vItB + 1 < cNtB) {
            vCurrBuf ^= 1;
            vNextBuf ^= 1;
          }
        }
      } else if constexpr (std::is_same_v<T, half2> || std::is_same_v<T, nv_bfloat162>) {
        float2 vValsBuf[2][tile_B::I];
        auto vGatherTile = [&](int pTileIdxLocal, float2 *pVals) {
#pragma unroll
          for (int vJ0 = 0; vJ0 < tile_B::I; ++vJ0) {
            const int cJ = vJ0 + pTileIdxLocal * tile_B::I;
            const int cGlobalJ = cColBase + cJ;
            float2 vTmp = make_float2(0.0f, 0.0f);
            if (cJ < cols_per_block && cGlobalJ < cNcolsExpert) {
              const int cSrcEntry = vIdsSrcExpert[cGlobalJ];
              const uint2 cQrm = fast_div_modulo((uint32_t)cSrcEntry, pSis1Fd);
              const int cToken = (int)cQrm.x;
              const int cChannel = (int)cQrm.y;
              if (cToken < pNcolsDstTotal) {
                vTmp = *(const float2 *)&pY[cChannel * pStrideChannelY + 2 * (cToken * pStrideColY + vCol)];
              }
            }
            pVals[vJ0] = vTmp;
          }
        };

        if (cNtB > 0) {
          vGatherTile(0, vValsBuf[0]);
        }

        int vCurrBuf = 0;
        int vNextBuf = 1;
#pragma unroll
        for (int vItB = 0; vItB < cNtB; ++vItB) {
          ggml_cuda_syncwarp();
#pragma unroll
          for (int vJ0 = 0; vJ0 < tile_B::I; ++vJ0) {
            const float2 cTmp = vValsBuf[vCurrBuf][vJ0];
            vTileXy[vJ0 * cTileKPadded + threadIdx.x] = ggml_cuda_cast<T>(cTmp);
          }

          if (vItB + 1 < cNtB) {
            vGatherTile(vItB + 1, vValsBuf[vNextBuf]);
          }

          ggml_cuda_syncwarp();
#pragma unroll
          for (int vK0 = 0; vK0 < cWarpSize; vK0 += tile_B::J) {
            tile_B vB;
            load_ldmatrix(vB, vTileXy + vK0, cTileKPadded);
#pragma unroll
            for (int vItA = 0; vItA < cNtA; ++vItA) {
              mma(vC[vItA][vItB], vA[vItA][vK0 / tile_B::J], vB);
            }
          }

          if (vItB + 1 < cNtB) {
            vCurrBuf ^= 1;
            vNextBuf ^= 1;
          }
        }
      } else {
        static_assert(std::is_same_v<T, void>, "unsupported type");
      }
    }

    float *vBufIw = (float *)vComputeBase;
    constexpr int cKiw = nwarps * rows_per_block + mmf_get_padding();

    if (nwarps > 1) {
      __syncthreads();
    } else {
      ggml_cuda_syncwarp();
    }
#pragma unroll
    for (int vItB = 0; vItB < cNtB; ++vItB) {
#pragma unroll
      for (int vItA = 0; vItA < cNtA; ++vItA) {
#pragma unroll
        for (int vL = 0; vL < tile_C::ne; ++vL) {
          const int cI = threadIdx.y * rows_per_block + vItA * tile_C::I + tile_C::get_i(vL);
          const int cJ = vItB * tile_C::J + tile_C::get_j(vL);
          vBufIw[cJ * cKiw + cI] = vC[vItA][vItB].x[vL];
        }
      }
    }

    if (nwarps > 1) {
      __syncthreads();
    } else {
      ggml_cuda_syncwarp();
    }

#pragma unroll
    for (int vJ0 = 0; vJ0 < cols_per_block; vJ0 += nwarps) {
      const int cJ = vJ0 + threadIdx.y;

      if (vJ0 + nwarps > cols_per_block && cJ >= cols_per_block) {
        return;
      }

      float vSum[rows_per_block / cWarpSize] = {0.0f};
      static_assert((rows_per_block % cWarpSize) == 0, "rows_per_block must be a multiple of warp_size.");
#pragma unroll
      for (int vI0 = 0; vI0 < nwarps * rows_per_block; vI0 += rows_per_block) {
#pragma unroll
        for (int vI1 = 0; vI1 < sizeof(vSum) / sizeof(vSum[0]); ++vI1) {
          const int cI = vI0 + vI1 * cWarpSize + threadIdx.x;

          vSum[vI1] += vBufIw[cJ * cKiw + cI];
        }
      }

      const int cGlobalJ = cColBase + cJ;
      if (cJ < cols_per_block && cGlobalJ < cNcolsExpert && pNchannelsDst > 0) {
        const int cDstEntry = vIdsDstExpert[cGlobalJ];
        const uint2 cQrm = fast_div_modulo((uint32_t)cDstEntry, pNchFd);
        const int cToken = (int)cQrm.x;
        if (cToken < pNcolsDstTotal) {
          const int cSlot = (int)cQrm.y;
#pragma unroll
          for (int vI0 = 0; vI0 < sizeof(vSum) / sizeof(vSum[0]); ++vI0) {
            pDst[cSlot * pStrideChannelDst + cToken * pStrideColDst + cRow0 + vI0 * cWarpSize + threadIdx.x] =
                vSum[vI0];
          }
        }
      }
    }
  }
#else
  GGML_UNUSED_VARS(pX, pY, pIdsSrcCompact, pIdsDstCompact, pExpertBounds, pDst, pNcols, pNcolsDstTotal, pNchannelsDst,
                   pStrideRow, pStrideColY, pStrideColDst, pChannelRatio, pStrideChannelX, pStrideChannelY,
                   pStrideChannelDst, pSampleRatio, pStrideSampleX, pStrideSampleY, pStrideSampleDst, pSis1Fd, pNchFd);
  NO_DEVICE_CODE;
#endif // defined(VOLTA_MMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE) ||
       // defined(AMD_MFMA_AVAILABLE)
}

template <typename T, int rows_per_block, int cols_per_block, int nwarps>
static inline void mul_mat_f_switch_ids(
    const T *pX, const float *pY, const int32_t *pIds, float *pDst, const int64_t pNcolsX, const int64_t pNcolsDst,
    const int64_t pNchannelsDst, const int64_t pStrideRow, const int64_t pStrideColY, const int64_t pStrideColDst,
    const int64_t pStrideColId, const int64_t pStrideRowId, const int64_t pChannelRatio, const int64_t pStrideChannelX,
    const int64_t pStrideChannelY, const int64_t pStrideChannelDst, const int64_t pSampleRatio,
    const int64_t pStrideSampleX, const int64_t pStrideSampleY, const int64_t pStrideSampleDst, const dim3 &pBlockNums,
    const dim3 &pBlockDims, const int pNbytesSharedTotal, cudaStream_t pStream, const mmf_ids_data *pIdsData) {
  const bool cHasIdsData = pIdsData && pIdsData->ids_src_compact;

  // Use the compact-ids kernel only for larger tiles; for small ncols_dst (< 16)
  // we prefer the normal mul_mat_f path with has_ids=true.
  if (cHasIdsData && pNcolsDst > 16) {
    const int cMaxTiles = (int)((pNcolsDst + cols_per_block - 1) / cols_per_block);
    if (cMaxTiles == 0) {
      return;
    }
    dim3 vBlockNumsIds(pBlockNums.x, pIdsData->n_experts, cMaxTiles);

    const uint3 cSis1Fd = pIdsData->sis1 > 0 ? init_fastdiv_values((uint32_t)pIdsData->sis1) : make_uint3(0, 0, 1);
    const uint3 cNchFd = init_fastdiv_values((uint32_t)pNchannelsDst);

    mul_mat_f_ids<T, rows_per_block, cols_per_block, nwarps>
        <<<vBlockNumsIds, pBlockDims, pNbytesSharedTotal, pStream>>>(
            pX, pY, pIdsData->ids_src_compact, pIdsData->ids_dst_compact, pIdsData->expert_bounds_dev, pDst, pNcolsX,
            pNcolsDst, pNchannelsDst, pStrideRow, pStrideColY, pStrideColDst, pChannelRatio, pStrideChannelX,
            pStrideChannelY, pStrideChannelDst, pSampleRatio, pStrideSampleX, pStrideSampleY, pStrideSampleDst, cSis1Fd,
            cNchFd);
  } else if (pIds) {
    const int64_t cColTiles = (pNcolsDst + cols_per_block - 1) / cols_per_block;
    dim3 vBlockNumsIds = pBlockNums;
    vBlockNumsIds.y *= cColTiles;

    mul_mat_f<T, rows_per_block, cols_per_block, nwarps, true>
        <<<vBlockNumsIds, pBlockDims, pNbytesSharedTotal, pStream>>>(
            pX, pY, pIds, pDst, pNcolsX, pNcolsDst, pNchannelsDst, pStrideRow, pStrideColY, pStrideColDst, pStrideColId,
            pStrideRowId, pChannelRatio, pStrideChannelX, pStrideChannelY, pStrideChannelDst, pSampleRatio,
            pStrideSampleX, pStrideSampleY, pStrideSampleDst);
  } else {
    mul_mat_f<T, rows_per_block, cols_per_block, nwarps, false>
        <<<pBlockNums, pBlockDims, pNbytesSharedTotal, pStream>>>(
            pX, pY, pIds, pDst, pNcolsX, cols_per_block, pNchannelsDst, pStrideRow, pStrideColY, pStrideColDst,
            pStrideColId, pStrideRowId, pChannelRatio, pStrideChannelX, pStrideChannelY, pStrideChannelDst,
            pSampleRatio, pStrideSampleX, pStrideSampleY, pStrideSampleDst);
  }
}

template <typename T, int rows_per_block, int cols_per_block>
void mul_mat_f_cuda(const T *pX, const float *pY, const int32_t *pIds, float *pDst, const int64_t pNcolsX,
                    const int64_t pNrowsX, const int64_t pNcolsDst, const int64_t pStrideRow, const int64_t pStrideColY,
                    const int64_t pStrideColDst, const int64_t pStrideColId, const int64_t pStrideRowId,
                    const int64_t pNchannelsX, const int64_t pNchannelsY, const int64_t pNchannelsDst,
                    const int64_t pStrideChannelX, const int64_t pStrideChannelY, const int64_t pStrideChannelDst,
                    const int64_t pNsamplesX, const int64_t pNsamplesDst, const int64_t pStrideSampleX,
                    const int64_t pStrideSampleY, const int64_t pStrideSampleDst, cudaStream_t pStream,
                    const mmf_ids_data *pIdsData) {
  typedef tile<16, 8, T> tile_A_16;
  typedef tile<32, 8, T> tile_A_32;
  typedef tile<16, 8, T> tile_B_16;
  typedef tile<8, 8, T> tile_B_8;

  GGML_ASSERT(pNcolsX % 2 == 0);
  GGML_ASSERT(pStrideRow % 2 == 0);
  GGML_ASSERT(pStrideColY % 2 == 0);
  GGML_ASSERT(pIds || pNchannelsDst % pNchannelsX == 0);
  GGML_ASSERT(pNsamplesDst % pNsamplesX == 0);
  const int64_t cChannelRatio = pNchannelsDst / pNchannelsX;
  const int64_t cSampleRatio = pNsamplesDst / pNsamplesX;

  const int cDevice = ggml_cuda_get_device();
  const int cCc = ggml_cuda_info().devices[cDevice].cc;
  const int cWarpSize = ggml_cuda_info().devices[cDevice].warp_size;

  int64_t vNwarpsBest = 1;
  int64_t vNiterBest = (pNcolsX + cWarpSize * 2 - 1) / (cWarpSize * 2);
  int64_t vMaxBlockSize = mmf_get_max_block_size(cCc);
  for (int64_t vNwarps = 2; vNwarps <= vMaxBlockSize / cWarpSize; vNwarps++) {
    const int64_t cNiter = (pNcolsX + vNwarps * cWarpSize * 2 - 1) / (vNwarps * cWarpSize * 2);
    if (cNiter < vNiterBest) {
      vNiterBest = cNiter;
      vNwarpsBest = vNwarps;
    }
  }

  const int cNbytesSharedIter =
      vNwarpsBest * (volta_mma_available(cCc) ? tile_A_32::I : tile_A_16::I) * (cWarpSize + mmf_get_padding(cCc)) * 4;
  const int cNbytesColsPerBlockPad = (amd_wmma_available(cCc) || amd_mfma_available(cCc)) ? tile_B_16::I : tile_B_8::I;
  const int cNbytesSharedCombine =
      GGML_PAD(cols_per_block, cNbytesColsPerBlockPad) * (vNwarpsBest * rows_per_block + mmf_get_padding(cCc)) * 4;
  const int cNbytesShared = std::max(cNbytesSharedIter, cNbytesSharedCombine);
  const int cNbytesSlotmap = pIds ? GGML_PAD(cols_per_block, 16) * sizeof(int) : 0;
  const int cNbytesSharedTotal = cNbytesShared + cNbytesSlotmap;
  const int64_t cGridY = pIds ? pNchannelsX : pNchannelsDst;

  const dim3 cBlockNums(pNrowsX / rows_per_block, cGridY, pNsamplesDst);
  const dim3 cBlockDims(cWarpSize, vNwarpsBest, 1);

  switch (vNwarpsBest) {
  case 1: {
    mul_mat_f_switch_ids<T, rows_per_block, cols_per_block, 1>(
        pX, pY, pIds, pDst, pNcolsX, pNcolsDst, pNchannelsDst, pStrideRow, pStrideColY, pStrideColDst, pStrideColId,
        pStrideRowId, cChannelRatio, pStrideChannelX, pStrideChannelY, pStrideChannelDst, cSampleRatio, pStrideSampleX,
        pStrideSampleY, pStrideSampleDst, cBlockNums, cBlockDims, cNbytesSharedTotal, pStream, pIdsData);
  } break;
  case 2: {
    mul_mat_f_switch_ids<T, rows_per_block, cols_per_block, 2>(
        pX, pY, pIds, pDst, pNcolsX, pNcolsDst, pNchannelsDst, pStrideRow, pStrideColY, pStrideColDst, pStrideColId,
        pStrideRowId, cChannelRatio, pStrideChannelX, pStrideChannelY, pStrideChannelDst, cSampleRatio, pStrideSampleX,
        pStrideSampleY, pStrideSampleDst, cBlockNums, cBlockDims, cNbytesSharedTotal, pStream, pIdsData);
  } break;
  case 3: {
    mul_mat_f_switch_ids<T, rows_per_block, cols_per_block, 3>(
        pX, pY, pIds, pDst, pNcolsX, pNcolsDst, pNchannelsDst, pStrideRow, pStrideColY, pStrideColDst, pStrideColId,
        pStrideRowId, cChannelRatio, pStrideChannelX, pStrideChannelY, pStrideChannelDst, cSampleRatio, pStrideSampleX,
        pStrideSampleY, pStrideSampleDst, cBlockNums, cBlockDims, cNbytesSharedTotal, pStream, pIdsData);
  } break;
  case 4: {
    mul_mat_f_switch_ids<T, rows_per_block, cols_per_block, 4>(
        pX, pY, pIds, pDst, pNcolsX, pNcolsDst, pNchannelsDst, pStrideRow, pStrideColY, pStrideColDst, pStrideColId,
        pStrideRowId, cChannelRatio, pStrideChannelX, pStrideChannelY, pStrideChannelDst, cSampleRatio, pStrideSampleX,
        pStrideSampleY, pStrideSampleDst, cBlockNums, cBlockDims, cNbytesSharedTotal, pStream, pIdsData);
  } break;
  case 5: {
    mul_mat_f_switch_ids<T, rows_per_block, cols_per_block, 5>(
        pX, pY, pIds, pDst, pNcolsX, pNcolsDst, pNchannelsDst, pStrideRow, pStrideColY, pStrideColDst, pStrideColId,
        pStrideRowId, cChannelRatio, pStrideChannelX, pStrideChannelY, pStrideChannelDst, cSampleRatio, pStrideSampleX,
        pStrideSampleY, pStrideSampleDst, cBlockNums, cBlockDims, cNbytesSharedTotal, pStream, pIdsData);
  } break;
  case 6: {
    mul_mat_f_switch_ids<T, rows_per_block, cols_per_block, 6>(
        pX, pY, pIds, pDst, pNcolsX, pNcolsDst, pNchannelsDst, pStrideRow, pStrideColY, pStrideColDst, pStrideColId,
        pStrideRowId, cChannelRatio, pStrideChannelX, pStrideChannelY, pStrideChannelDst, cSampleRatio, pStrideSampleX,
        pStrideSampleY, pStrideSampleDst, cBlockNums, cBlockDims, cNbytesSharedTotal, pStream, pIdsData);
  } break;
  case 7: {
    mul_mat_f_switch_ids<T, rows_per_block, cols_per_block, 7>(
        pX, pY, pIds, pDst, pNcolsX, pNcolsDst, pNchannelsDst, pStrideRow, pStrideColY, pStrideColDst, pStrideColId,
        pStrideRowId, cChannelRatio, pStrideChannelX, pStrideChannelY, pStrideChannelDst, cSampleRatio, pStrideSampleX,
        pStrideSampleY, pStrideSampleDst, cBlockNums, cBlockDims, cNbytesSharedTotal, pStream, pIdsData);
  } break;
  case 8: {
    mul_mat_f_switch_ids<T, rows_per_block, cols_per_block, 8>(
        pX, pY, pIds, pDst, pNcolsX, pNcolsDst, pNchannelsDst, pStrideRow, pStrideColY, pStrideColDst, pStrideColId,
        pStrideRowId, cChannelRatio, pStrideChannelX, pStrideChannelY, pStrideChannelDst, cSampleRatio, pStrideSampleX,
        pStrideSampleY, pStrideSampleDst, cBlockNums, cBlockDims, cNbytesSharedTotal, pStream, pIdsData);
  } break;
  default: {
    GGML_ABORT("fatal error");
  } break;
  }

  GGML_UNUSED_VARS(pNchannelsY);
}

template <typename T, int rows_per_block>
static void mul_mat_f_switch_cols_per_block(
    const T *pX, const float *pY, const int32_t *pIds, float *pDst, const int64_t pNcolsX, const int64_t pNrowsX,
    const int64_t pNcolsDst, const int64_t pStrideRow, const int64_t pStrideColY, const int64_t pStrideColDst,
    const int64_t pStrideColId, const int pStrideRowId, const int64_t pNchannelsX, const int64_t pNchannelsY,
    const int64_t pNchannelsDst, const int64_t pStrideChannelX, const int64_t pStrideChannelY,
    const int64_t pStrideChannelDst, const int64_t pNsamplesX, const int64_t pNsamplesDst, const int64_t pStrideSampleX,
    const int64_t pStrideSampleY, const int64_t pStrideSampleDst, cudaStream_t pStream, const mmf_ids_data *pIdsData) {

  const int cNcolsCase = (pIds && pNcolsDst > 16) ? 16 : pNcolsDst;

  GGML_ASSERT(pIds || pNcolsDst <= 16);

  switch (cNcolsCase) {
  case 1: {
    mul_mat_f_cuda<T, rows_per_block, 1>(
        pX, pY, pIds, pDst, pNcolsX, pNrowsX, pNcolsDst, pStrideRow, pStrideColY, pStrideColDst, pStrideColId,
        pStrideRowId, pNchannelsX, pNchannelsY, pNchannelsDst, pStrideChannelX, pStrideChannelY, pStrideChannelDst,
        pNsamplesX, pNsamplesDst, pStrideSampleX, pStrideSampleY, pStrideSampleDst, pStream, pIdsData);
  } break;
  case 2: {
    mul_mat_f_cuda<T, rows_per_block, 2>(
        pX, pY, pIds, pDst, pNcolsX, pNrowsX, pNcolsDst, pStrideRow, pStrideColY, pStrideColDst, pStrideColId,
        pStrideRowId, pNchannelsX, pNchannelsY, pNchannelsDst, pStrideChannelX, pStrideChannelY, pStrideChannelDst,
        pNsamplesX, pNsamplesDst, pStrideSampleX, pStrideSampleY, pStrideSampleDst, pStream, pIdsData);
  } break;
  case 3: {
    mul_mat_f_cuda<T, rows_per_block, 3>(
        pX, pY, pIds, pDst, pNcolsX, pNrowsX, pNcolsDst, pStrideRow, pStrideColY, pStrideColDst, pStrideColId,
        pStrideRowId, pNchannelsX, pNchannelsY, pNchannelsDst, pStrideChannelX, pStrideChannelY, pStrideChannelDst,
        pNsamplesX, pNsamplesDst, pStrideSampleX, pStrideSampleY, pStrideSampleDst, pStream, pIdsData);
  } break;
  case 4: {
    mul_mat_f_cuda<T, rows_per_block, 4>(
        pX, pY, pIds, pDst, pNcolsX, pNrowsX, pNcolsDst, pStrideRow, pStrideColY, pStrideColDst, pStrideColId,
        pStrideRowId, pNchannelsX, pNchannelsY, pNchannelsDst, pStrideChannelX, pStrideChannelY, pStrideChannelDst,
        pNsamplesX, pNsamplesDst, pStrideSampleX, pStrideSampleY, pStrideSampleDst, pStream, pIdsData);
  } break;
  case 5: {
    mul_mat_f_cuda<T, rows_per_block, 5>(
        pX, pY, pIds, pDst, pNcolsX, pNrowsX, pNcolsDst, pStrideRow, pStrideColY, pStrideColDst, pStrideColId,
        pStrideRowId, pNchannelsX, pNchannelsY, pNchannelsDst, pStrideChannelX, pStrideChannelY, pStrideChannelDst,
        pNsamplesX, pNsamplesDst, pStrideSampleX, pStrideSampleY, pStrideSampleDst, pStream, pIdsData);
  } break;
  case 6: {
    mul_mat_f_cuda<T, rows_per_block, 6>(
        pX, pY, pIds, pDst, pNcolsX, pNrowsX, pNcolsDst, pStrideRow, pStrideColY, pStrideColDst, pStrideColId,
        pStrideRowId, pNchannelsX, pNchannelsY, pNchannelsDst, pStrideChannelX, pStrideChannelY, pStrideChannelDst,
        pNsamplesX, pNsamplesDst, pStrideSampleX, pStrideSampleY, pStrideSampleDst, pStream, pIdsData);
  } break;
  case 7: {
    mul_mat_f_cuda<T, rows_per_block, 7>(
        pX, pY, pIds, pDst, pNcolsX, pNrowsX, pNcolsDst, pStrideRow, pStrideColY, pStrideColDst, pStrideColId,
        pStrideRowId, pNchannelsX, pNchannelsY, pNchannelsDst, pStrideChannelX, pStrideChannelY, pStrideChannelDst,
        pNsamplesX, pNsamplesDst, pStrideSampleX, pStrideSampleY, pStrideSampleDst, pStream, pIdsData);
  } break;
  case 8: {
    mul_mat_f_cuda<T, rows_per_block, 8>(
        pX, pY, pIds, pDst, pNcolsX, pNrowsX, pNcolsDst, pStrideRow, pStrideColY, pStrideColDst, pStrideColId,
        pStrideRowId, pNchannelsX, pNchannelsY, pNchannelsDst, pStrideChannelX, pStrideChannelY, pStrideChannelDst,
        pNsamplesX, pNsamplesDst, pStrideSampleX, pStrideSampleY, pStrideSampleDst, pStream, pIdsData);
  } break;
  case 9: {
    mul_mat_f_cuda<T, rows_per_block, 9>(
        pX, pY, pIds, pDst, pNcolsX, pNrowsX, pNcolsDst, pStrideRow, pStrideColY, pStrideColDst, pStrideColId,
        pStrideRowId, pNchannelsX, pNchannelsY, pNchannelsDst, pStrideChannelX, pStrideChannelY, pStrideChannelDst,
        pNsamplesX, pNsamplesDst, pStrideSampleX, pStrideSampleY, pStrideSampleDst, pStream, pIdsData);
  } break;
  case 10: {
    mul_mat_f_cuda<T, rows_per_block, 10>(
        pX, pY, pIds, pDst, pNcolsX, pNrowsX, pNcolsDst, pStrideRow, pStrideColY, pStrideColDst, pStrideColId,
        pStrideRowId, pNchannelsX, pNchannelsY, pNchannelsDst, pStrideChannelX, pStrideChannelY, pStrideChannelDst,
        pNsamplesX, pNsamplesDst, pStrideSampleX, pStrideSampleY, pStrideSampleDst, pStream, pIdsData);
  } break;
  case 11: {
    mul_mat_f_cuda<T, rows_per_block, 11>(
        pX, pY, pIds, pDst, pNcolsX, pNrowsX, pNcolsDst, pStrideRow, pStrideColY, pStrideColDst, pStrideColId,
        pStrideRowId, pNchannelsX, pNchannelsY, pNchannelsDst, pStrideChannelX, pStrideChannelY, pStrideChannelDst,
        pNsamplesX, pNsamplesDst, pStrideSampleX, pStrideSampleY, pStrideSampleDst, pStream, pIdsData);
  } break;
  case 12: {
    mul_mat_f_cuda<T, rows_per_block, 12>(
        pX, pY, pIds, pDst, pNcolsX, pNrowsX, pNcolsDst, pStrideRow, pStrideColY, pStrideColDst, pStrideColId,
        pStrideRowId, pNchannelsX, pNchannelsY, pNchannelsDst, pStrideChannelX, pStrideChannelY, pStrideChannelDst,
        pNsamplesX, pNsamplesDst, pStrideSampleX, pStrideSampleY, pStrideSampleDst, pStream, pIdsData);
  } break;
  case 13: {
    mul_mat_f_cuda<T, rows_per_block, 13>(
        pX, pY, pIds, pDst, pNcolsX, pNrowsX, pNcolsDst, pStrideRow, pStrideColY, pStrideColDst, pStrideColId,
        pStrideRowId, pNchannelsX, pNchannelsY, pNchannelsDst, pStrideChannelX, pStrideChannelY, pStrideChannelDst,
        pNsamplesX, pNsamplesDst, pStrideSampleX, pStrideSampleY, pStrideSampleDst, pStream, pIdsData);
  } break;
  case 14: {
    mul_mat_f_cuda<T, rows_per_block, 14>(
        pX, pY, pIds, pDst, pNcolsX, pNrowsX, pNcolsDst, pStrideRow, pStrideColY, pStrideColDst, pStrideColId,
        pStrideRowId, pNchannelsX, pNchannelsY, pNchannelsDst, pStrideChannelX, pStrideChannelY, pStrideChannelDst,
        pNsamplesX, pNsamplesDst, pStrideSampleX, pStrideSampleY, pStrideSampleDst, pStream, pIdsData);
  } break;
  case 15: {
    mul_mat_f_cuda<T, rows_per_block, 15>(
        pX, pY, pIds, pDst, pNcolsX, pNrowsX, pNcolsDst, pStrideRow, pStrideColY, pStrideColDst, pStrideColId,
        pStrideRowId, pNchannelsX, pNchannelsY, pNchannelsDst, pStrideChannelX, pStrideChannelY, pStrideChannelDst,
        pNsamplesX, pNsamplesDst, pStrideSampleX, pStrideSampleY, pStrideSampleDst, pStream, pIdsData);
  } break;
  case 16: {
    mul_mat_f_cuda<T, rows_per_block, 16>(
        pX, pY, pIds, pDst, pNcolsX, pNrowsX, pNcolsDst, pStrideRow, pStrideColY, pStrideColDst, pStrideColId,
        pStrideRowId, pNchannelsX, pNchannelsY, pNchannelsDst, pStrideChannelX, pStrideChannelY, pStrideChannelDst,
        pNsamplesX, pNsamplesDst, pStrideSampleX, pStrideSampleY, pStrideSampleDst, pStream, pIdsData);
  } break;
  default: {
    GGML_ABORT("fatal error");
  } break;
  }
}

template <typename T>
static void mul_mat_f_switch_rows_per_block(
    const int pRowsPerBlock, const T *pX, const float *pY, const int32_t *pIds, float *pDst, const int64_t pNcolsX,
    const int64_t pNrowsX, const int64_t pNcolsDst, const int64_t pStrideRow, const int64_t pStrideColY,
    const int64_t pStrideColDst, const int64_t pStrideColId, const int pStrideRowId, const int64_t pNchannelsX,
    const int64_t pNchannelsY, const int64_t pNchannelsDst, const int64_t pStrideChannelX,
    const int64_t pStrideChannelY, const int64_t pStrideChannelDst, const int64_t pNsamplesX,
    const int64_t pNsamplesDst, const int64_t pStrideSampleX, const int64_t pStrideSampleY,
    const int64_t pStrideSampleDst, cudaStream_t pStream, const mmf_ids_data *pIdsData) {
  switch (pRowsPerBlock) {
  case MMF_ROWS_PER_BLOCK: {
    mul_mat_f_switch_cols_per_block<T, MMF_ROWS_PER_BLOCK>(
        pX, pY, pIds, pDst, pNcolsX, pNrowsX, pNcolsDst, pStrideRow, pStrideColY, pStrideColDst, pStrideColId,
        pStrideRowId, pNchannelsX, pNchannelsY, pNchannelsDst, pStrideChannelX, pStrideChannelY, pStrideChannelDst,
        pNsamplesX, pNsamplesDst, pStrideSampleX, pStrideSampleY, pStrideSampleDst, pStream, pIdsData);
  } break;
  case MMF_ROWS_PER_BLOCK_CDNA: {
    mul_mat_f_switch_cols_per_block<T, MMF_ROWS_PER_BLOCK_CDNA>(
        pX, pY, pIds, pDst, pNcolsX, pNrowsX, pNcolsDst, pStrideRow, pStrideColY, pStrideColDst, pStrideColId,
        pStrideRowId, pNchannelsX, pNchannelsY, pNchannelsDst, pStrideChannelX, pStrideChannelY, pStrideChannelDst,
        pNsamplesX, pNsamplesDst, pStrideSampleX, pStrideSampleY, pStrideSampleDst, pStream, pIdsData);
  } break;
  default:
    GGML_ABORT("unsupported rows_per_block: %i", pRowsPerBlock);
  }
}

#define DECL_MMF_CASE_HELPER(T, nrows_dst, ncols_dst)                                                                  \
  template void mul_mat_f_cuda<T, nrows_dst, ncols_dst>(                                                               \
      const T *x, const float *y, const int32_t *ids, float *dst, const int64_t ncols_x, const int64_t nrows_x,        \
      int64_t ncols_dst_total, const int64_t stride_row, const int64_t stride_col_y, const int64_t stride_col_dst,     \
      const int64_t stride_col_id, const int64_t stride_row_id, const int64_t nchannels_x, const int64_t nchannels_y,  \
      const int64_t nchannels_dst, const int64_t stride_channel_x, const int64_t stride_channel_y,                     \
      const int64_t stride_channel_dst, const int64_t nsamples_x, const int64_t nsamples_dst,                          \
      const int64_t stride_sample_x, const int64_t stride_sample_y, const int64_t stride_sample_dst,                   \
      cudaStream_t stream, const mmf_ids_data *ids_data);

#define DECL_MMF_CASE_EXTERN(ncols_dst)                                                                                                                            \
  extern DECL_MMF_CASE_HELPER(float, MMF_ROWS_PER_BLOCK, ncols_dst) extern DECL_MMF_CASE_HELPER(half2, MMF_ROWS_PER_BLOCK, ncols_dst) extern DECL_MMF_CASE_HELPER( \
      nv_bfloat162, MMF_ROWS_PER_BLOCK,                                                                                                                            \
      ncols_dst) extern DECL_MMF_CASE_HELPER(float, MMF_ROWS_PER_BLOCK_CDNA,                                                                                       \
                                             ncols_dst) extern DECL_MMF_CASE_HELPER(half2, MMF_ROWS_PER_BLOCK_CDNA,                                                \
                                                                                    ncols_dst) extern DECL_MMF_CASE_HELPER(nv_bfloat162,                           \
                                                                                                                           MMF_ROWS_PER_BLOCK_CDNA,                \
                                                                                                                           ncols_dst)

#define DECL_MMF_CASE(ncols_dst)                                                                                       \
  DECL_MMF_CASE_HELPER(float, MMF_ROWS_PER_BLOCK, ncols_dst)                                                           \
  DECL_MMF_CASE_HELPER(half2, MMF_ROWS_PER_BLOCK, ncols_dst)                                                           \
  DECL_MMF_CASE_HELPER(nv_bfloat162, MMF_ROWS_PER_BLOCK, ncols_dst)                                                    \
  DECL_MMF_CASE_HELPER(float, MMF_ROWS_PER_BLOCK_CDNA, ncols_dst)                                                      \
  DECL_MMF_CASE_HELPER(half2, MMF_ROWS_PER_BLOCK_CDNA, ncols_dst)                                                      \
  DECL_MMF_CASE_HELPER(nv_bfloat162, MMF_ROWS_PER_BLOCK_CDNA, ncols_dst)

DECL_MMF_CASE_EXTERN(1);
DECL_MMF_CASE_EXTERN(2);
DECL_MMF_CASE_EXTERN(3);
DECL_MMF_CASE_EXTERN(4);
DECL_MMF_CASE_EXTERN(5);
DECL_MMF_CASE_EXTERN(6);
DECL_MMF_CASE_EXTERN(7);
DECL_MMF_CASE_EXTERN(8);
DECL_MMF_CASE_EXTERN(9);
DECL_MMF_CASE_EXTERN(10);
DECL_MMF_CASE_EXTERN(11);
DECL_MMF_CASE_EXTERN(12);
DECL_MMF_CASE_EXTERN(13);
DECL_MMF_CASE_EXTERN(14);
DECL_MMF_CASE_EXTERN(15);
DECL_MMF_CASE_EXTERN(16);
