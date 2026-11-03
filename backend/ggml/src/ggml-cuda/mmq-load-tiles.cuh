#pragma once

#include "vecdotq.cuh"

#include "mmq.cuh"

template <ggml_type type, int J, bool fallback>
static __device__ __forceinline__ void ggml_cuda_mmq_load_tiles_q1_0(const char *__restrict__ pX,
                                                                     int *__restrict__ pXTile, const int pKbx0,
                                                                     const int pIMax, const int pStride) {
  constexpr int cWarpSize = ggml_cuda_get_physical_warp_size();
  constexpr int cNwarps = ggml_cuda_mmq_get_nthreads(type, J, fallback) / cWarpSize;
  constexpr int cI = ggml_cuda_mmq_get_I(type, J, fallback);
  constexpr int cSramStride = ggml_cuda_mmq_get_sram_stride(type, J, fallback);

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
  int *vXQs = (int *)pXTile;
  float *vXDf = (float *)(vXQs + 2 * MMQ_TILE_NE_K);
#else
  constexpr tile_x_sizes cTxs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_Q8_0, cI);
  int *vXQs = (int *)pXTile;
  float *vXDf = (float *)(vXQs + cTxs.qs);
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)

  constexpr int cBlocksPerIter = MMQ_ITER_K / QK1_0;
  constexpr int cThreadsPerRow = cBlocksPerIter * QI1_0;
  constexpr int cNrows = cWarpSize / cThreadsPerRow;
  constexpr int cScaleEntriesPerBlock = QK1_0 / QK8_1;
  constexpr int cScaleEntriesPerRow = cBlocksPerIter * cScaleEntriesPerBlock;

  const int cTxi = threadIdx.x % cThreadsPerRow;
  const int cKbx = cTxi / QI1_0;
  const int cKqsx = cTxi % QI1_0;

#pragma unroll
  for (int vI0 = 0; vI0 < cI; vI0 += cNrows * cNwarps) {
    int vI = vI0 + threadIdx.y * cNrows + threadIdx.x / cThreadsPerRow;

    if (fallback) {
      vI = min(vI, pIMax);
    }

    const block_q1_0 *vBxi = (const block_q1_0 *)pX + pKbx0 + vI * pStride + cKbx;
    const int16_t *vQxi = (const int16_t *)vBxi->qs + cKqsx * 2;

    const int cDstOffset = cKbx * (cScaleEntriesPerBlock * QI8_0) + cKqsx * QI8_0;
#pragma unroll
    for (int vJ = 0; vJ < 2; ++vJ) {
      const int cQ = vQxi[vJ];

      // unpack crumbs into nibble indices
      const int cN0 = __byte_perm(0x11100100, 0x11100100, cQ >> 0); // [0, 1, 4, 5] [ 8,  9, 12, 13]
      const int cN1 = __byte_perm(0x11100100, 0x11100100, cQ >> 2); // [2, 3, 6, 7] [10, 11, 14, 15]
      // unpack nibbles into byte values
      const int cS0 = __byte_perm(0x01FF, 0x01FF, cN0 >> 0);
      const int cS1 = __byte_perm(0x01FF, 0x01FF, cN1 >> 0);
      const int cS2 = __byte_perm(0x01FF, 0x01FF, cN0 >> 16);
      const int cS3 = __byte_perm(0x01FF, 0x01FF, cN1 >> 16);
      // unshuffle values
      const int cV0 = __byte_perm(cS0, cS1, 0x5410);
      const int cV1 = __byte_perm(cS0, cS1, 0x7632);
      const int cV2 = __byte_perm(cS2, cS3, 0x5410);
      const int cV3 = __byte_perm(cS2, cS3, 0x7632);

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
      vXQs[vI * cSramStride + cDstOffset + vJ * 4 + 0] = cV0;
      vXQs[vI * cSramStride + cDstOffset + vJ * 4 + 1] = cV1;
      vXQs[vI * cSramStride + cDstOffset + vJ * 4 + 2] = cV2;
      vXQs[vI * cSramStride + cDstOffset + vJ * 4 + 3] = cV3;
#else
      vXQs[vI * (2 * MMQ_TILE_NE_K + 1) + cDstOffset + vJ * 4 + 0] = cV0;
      vXQs[vI * (2 * MMQ_TILE_NE_K + 1) + cDstOffset + vJ * 4 + 1] = cV1;
      vXQs[vI * (2 * MMQ_TILE_NE_K + 1) + cDstOffset + vJ * 4 + 2] = cV2;
      vXQs[vI * (2 * MMQ_TILE_NE_K + 1) + cDstOffset + vJ * 4 + 3] = cV3;
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    }
  }

  const int cKsx = threadIdx.x % cScaleEntriesPerRow;
  const int cScaleBlock = cKsx / cScaleEntriesPerBlock;

#pragma unroll
  for (int vI0 = 0; vI0 < cI; vI0 += cNwarps) {
    int vI = vI0 + threadIdx.y;

    if (fallback) {
      vI = min(vI, pIMax);
    }

    const block_q1_0 *vBxi = (const block_q1_0 *)pX + pKbx0 + vI * pStride + cScaleBlock;

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    vXDf[vI * cSramStride + cKsx] = vBxi->d;
#else
    vXDf[vI * (2 * MMQ_TILE_NE_K / QI8_0) + vI / (QI8_0 / 2) + cKsx] = vBxi->d;
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
  }
}

template <ggml_type type, int J, bool fallback>
static __device__ __forceinline__ void ggml_cuda_mmq_load_tiles_q2_0(const char *__restrict__ pX,
                                                                     int *__restrict__ pXTile, const int pKbx0,
                                                                     const int pIMax, const int pStride) {
  constexpr int cWarpSize = ggml_cuda_get_physical_warp_size();
  constexpr int cNwarps = ggml_cuda_mmq_get_nthreads(type, J, fallback) / cWarpSize;
  constexpr int cI = ggml_cuda_mmq_get_I(type, J, fallback);
  constexpr int cSramStride = ggml_cuda_mmq_get_sram_stride(type, J, fallback);

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
  int *vXQs = (int *)pXTile;
  float *vXDf = (float *)(vXQs + 2 * MMQ_TILE_NE_K);
#else
  constexpr tile_x_sizes cTxs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_Q8_0, cI);
  int *vXQs = (int *)pXTile;
  float *vXDf = (float *)(vXQs + cTxs.qs);
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)

  constexpr int cBlocksPerIter = MMQ_ITER_K / QK2_0;
  constexpr int cThreadsPerRow = cBlocksPerIter * QI2_0;
  constexpr int cNrows = cWarpSize / cThreadsPerRow;
  constexpr int cScaleEntriesPerBlock = QK2_0 / QK8_1;
  constexpr int cScaleEntriesPerRow = cBlocksPerIter * cScaleEntriesPerBlock;

  const int cTxi = threadIdx.x % cThreadsPerRow;
  const int cKbx = cTxi / QI2_0;
  const int cKqsx = cTxi % QI2_0;

#pragma unroll
  for (int vI0 = 0; vI0 < cI; vI0 += cNrows * cNwarps) {
    int vI = vI0 + threadIdx.y * cNrows + threadIdx.x / cThreadsPerRow;

    if (fallback) {
      vI = min(vI, pIMax);
    }

    const block_q2_0 *vBxi = (const block_q2_0 *)pX + pKbx0 + vI * pStride + cKbx;
    const int16_t *vQxi = (const int16_t *)vBxi->qs + cKqsx * 4;

    const int cDstOffset = cKbx * (cScaleEntriesPerBlock * QI8_0) + cKqsx * QI8_0;

#pragma unroll
    for (int vJ = 0; vJ < 4; ++vJ) {
      const int cQ = vQxi[vJ];

      // unpack even and odd crumbs into byte values
      const int cQe = __byte_perm(0x020100FF, 0x020100FF, cQ >> 0);
      const int cQo = __byte_perm(0x020100FF, 0x020100FF, cQ >> 2);
      // unshuffle values
      const int cQx = __byte_perm(cQe, cQo, 0x5140);
      const int cQy = __byte_perm(cQe, cQo, 0x7362);

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
      vXQs[vI * cSramStride + cDstOffset + vJ * 2 + 0] = cQx;
      vXQs[vI * cSramStride + cDstOffset + vJ * 2 + 1] = cQy;
#else
      vXQs[vI * (2 * MMQ_TILE_NE_K + 1) + cDstOffset + vJ * 2 + 0] = cQx;
      vXQs[vI * (2 * MMQ_TILE_NE_K + 1) + cDstOffset + vJ * 2 + 1] = cQy;
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    }
  }

  const int cKsx = threadIdx.x % cScaleEntriesPerRow;
  const int cScaleBlock = cKsx / cScaleEntriesPerBlock;

#pragma unroll
  for (int vI0 = 0; vI0 < cI; vI0 += cNwarps) {
    int vI = vI0 + threadIdx.y;

    if (fallback) {
      vI = min(vI, pIMax);
    }

    const block_q2_0 *vBxi = (const block_q2_0 *)pX + pKbx0 + vI * pStride + cScaleBlock;

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    vXDf[vI * cSramStride + cKsx] = vBxi->d;
#else
    vXDf[vI * (2 * MMQ_TILE_NE_K / QI8_0) + vI / (QI8_0 / 2) + cKsx] = vBxi->d;
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
  }
}

template <ggml_type type, int J, bool fallback>
static __device__ __forceinline__ void ggml_cuda_mmq_load_tiles_q4_0(const char *__restrict__ pX,
                                                                     int *__restrict__ pXTile, const int pKbx0,
                                                                     const int pIMax, const int pStride) {
  constexpr int cWarpSize = ggml_cuda_get_physical_warp_size();
  constexpr int cNwarps = ggml_cuda_mmq_get_nthreads(type, J, fallback) / cWarpSize;
  constexpr int cI = ggml_cuda_mmq_get_I(type, J, fallback);
  constexpr int cSramStride = ggml_cuda_mmq_get_sram_stride(type, J, fallback);

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
  int *vXQs = (int *)pXTile;
  float *vXDf = (float *)(vXQs + 2 * MMQ_TILE_NE_K);
#else
  constexpr tile_x_sizes cTxs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_Q4_0, cI);
  int *vXQs = (int *)pXTile;
  float *vXDf = (float *)(vXQs + cTxs.qs);
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)

  constexpr int cThreadsPerRow = MMQ_ITER_K / (4 * QR4_0);
  constexpr int cNrows = cWarpSize / cThreadsPerRow;
  const int cTxi = cWarpSize > cThreadsPerRow ? threadIdx.x % cThreadsPerRow : threadIdx.x;
  const int cKbx = cTxi / QI4_0;
  const int cKqsx = cTxi % QI4_0;

#pragma unroll
  for (int vI0 = 0; vI0 < cI; vI0 += cNrows * cNwarps) {
    int vI = vI0 + (cNrows == 1 ? threadIdx.y : threadIdx.y * cNrows + threadIdx.x / cThreadsPerRow);

    if (fallback) {
      vI = min(vI, pIMax);
    }

    const block_q4_0 *vBxi = (const block_q4_0 *)pX + pKbx0 + vI * pStride + cKbx;
    const int cQs0 = get_int_b2(vBxi->qs, cKqsx);

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    vXQs[vI * cSramStride + cKbx * (2 * QI4_0) + cKqsx + 0] = __vsubss4((cQs0 >> 0) & 0x0F0F0F0F, 0x08080808);
    vXQs[vI * cSramStride + cKbx * (2 * QI4_0) + cKqsx + QI4_0] = __vsubss4((cQs0 >> 4) & 0x0F0F0F0F, 0x08080808);
#else
    vXQs[vI * (MMQ_TILE_NE_K + 1) + cTxi] = cQs0;
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE)
  }

  constexpr int cBlocksPerTileXRow = MMQ_TILE_NE_K / QI4_0;
  constexpr int cRowsPerWarp = cWarpSize / cBlocksPerTileXRow;
  const int cKbxd = threadIdx.x % cBlocksPerTileXRow;

#pragma unroll
  for (int vI0 = 0; vI0 < cI; vI0 += cNwarps * cRowsPerWarp) {
    int vI = vI0 + threadIdx.y * cRowsPerWarp + threadIdx.x / cBlocksPerTileXRow;

    if (fallback) {
      vI = min(vI, pIMax);
    }

    const block_q4_0 *vBxi = (const block_q4_0 *)pX + pKbx0 + vI * pStride + cKbxd;

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    vXDf[vI * cSramStride + cKbxd] = vBxi->d;
#else
    vXDf[vI * (MMQ_TILE_NE_K / QI4_0) + vI / QI4_0 + cKbxd] = vBxi->d;
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
  }
}

template <ggml_type type, int J, bool fallback>
static __device__ __forceinline__ void ggml_cuda_mmq_load_tiles_q4_1(const char *__restrict__ pX,
                                                                     int *__restrict__ pXTile, const int pKbx0,
                                                                     const int pIMax, const int pStride) {
  constexpr int cWarpSize = ggml_cuda_get_physical_warp_size();
  constexpr int cNwarps = ggml_cuda_mmq_get_nthreads(type, J, fallback) / cWarpSize;
  constexpr int cI = ggml_cuda_mmq_get_I(type, J, fallback);
  constexpr int cSramStride = ggml_cuda_mmq_get_sram_stride(type, J, fallback);

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
  int *vXQs = (int *)pXTile;
  half2 *vXDm = (half2 *)(vXQs + 2 * MMQ_TILE_NE_K);
#else
  constexpr tile_x_sizes cTxs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_Q4_1, cI);
  int *vXQs = (int *)pXTile;
  half2 *vXDm = (half2 *)(vXQs + cTxs.qs);
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE)  || defined(AMD_WMMA_AVAILABLE)

  constexpr int cThreadsPerRow = MMQ_ITER_K / (4 * QR4_1);
  constexpr int cNrows = cWarpSize / cThreadsPerRow;
  const int cTxi = cWarpSize > cThreadsPerRow ? threadIdx.x % cThreadsPerRow : threadIdx.x;
  const int cKbx = cTxi / QI4_1;
  const int cKqsx = cTxi % QI4_1;

#pragma unroll
  for (int vI0 = 0; vI0 < cI; vI0 += cNrows * cNwarps) {
    int vI = vI0 + (cNrows == 1 ? threadIdx.y : threadIdx.y * cNrows + threadIdx.x / cThreadsPerRow);

    if (fallback) {
      vI = min(vI, pIMax);
    }

    const block_q4_1 *vBxi = (const block_q4_1 *)pX + pKbx0 + vI * pStride + cKbx;
    const int cQs0 = get_int_b4(vBxi->qs, cKqsx);

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    vXQs[vI * cSramStride + cKbx * (2 * QI4_1) + cKqsx + 0] = (cQs0 >> 0) & 0x0F0F0F0F;
    vXQs[vI * cSramStride + cKbx * (2 * QI4_1) + cKqsx + QI4_1] = (cQs0 >> 4) & 0x0F0F0F0F;
#else
    vXQs[vI * (MMQ_TILE_NE_K + 1) + cTxi] = cQs0;
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
  }

  constexpr int cBlocksPerTileXRow = MMQ_TILE_NE_K / QI4_1;
  constexpr int cRowsPerWarp = cWarpSize / cBlocksPerTileXRow;
  const int cKbxd = threadIdx.x % cBlocksPerTileXRow;

#pragma unroll
  for (int vI0 = 0; vI0 < cI; vI0 += cNwarps * cRowsPerWarp) {
    int vI = vI0 + threadIdx.y * cRowsPerWarp + threadIdx.x / cBlocksPerTileXRow;

    if (fallback) {
      vI = min(vI, pIMax);
    }

    const block_q4_1 *vBxi = (const block_q4_1 *)pX + pKbx0 + vI * pStride + cKbxd;

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    vXDm[vI * cSramStride + cKbxd] = vBxi->dm;
#else
    vXDm[vI * (MMQ_TILE_NE_K / QI4_1) + vI / QI4_1 + cKbxd] = vBxi->dm;
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
  }
}

template <ggml_type type, int J, bool fallback>
static __device__ __forceinline__ void ggml_cuda_mmq_load_tiles_q5_0(const char *__restrict__ pX,
                                                                     int *__restrict__ pXTile, const int pKbx0,
                                                                     const int pIMax, const int pStride) {
  constexpr int cWarpSize = ggml_cuda_get_physical_warp_size();
  constexpr int cNwarps = ggml_cuda_mmq_get_nthreads(type, J, fallback) / cWarpSize;
  constexpr int cI = ggml_cuda_mmq_get_I(type, J, fallback);
  constexpr int cSramStride = ggml_cuda_mmq_get_sram_stride(type, J, fallback);

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
  int *vXQs = (int *)pXTile;
  float *vXDf = (float *)(vXQs + MMQ_TILE_NE_K * 2);
#else
  constexpr tile_x_sizes cTxs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_Q5_0, cI);
  int *vXQs = (int *)pXTile;
  float *vXDf = (float *)(vXQs + cTxs.qs);
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)

  constexpr int cThreadsPerRow = MMQ_ITER_K / (4 * QR5_0);
  constexpr int cNrows = cWarpSize / cThreadsPerRow;
  const int cTxi = cWarpSize > cThreadsPerRow ? threadIdx.x % cThreadsPerRow : threadIdx.x;
  const int cKbx = cTxi / QI5_0;
  const int cKqsx = cTxi % QI5_0;

#pragma unroll
  for (int vI0 = 0; vI0 < cI; vI0 += cNrows * cNwarps) {
    int vI = vI0 + (cNrows == 1 ? threadIdx.y : threadIdx.y * cNrows + threadIdx.x / cThreadsPerRow);

    if (fallback) {
      vI = min(vI, pIMax);
    }

    const block_q5_0 *vBxi = (const block_q5_0 *)pX + pKbx0 + vI * pStride + cKbx;

    const int cQl = get_int_b2(vBxi->qs, cKqsx);
    const int cQh = get_int_b2(vBxi->qh, 0) >> (4 * cKqsx);

    int vQs0 = (cQl >> 0) & 0x0F0F0F0F;
    vQs0 |= (cQh << 4) & 0x00000010;    // 0 ->  4
    vQs0 |= (cQh << 11) & 0x00001000;   // 1 -> 12
    vQs0 |= (cQh << 18) & 0x00100000;   // 2 -> 20
    vQs0 |= (cQh << 25) & 0x10000000;   // 3 -> 28
    vQs0 = __vsubss4(vQs0, 0x10101010); // subtract 16

    int vQs1 = (cQl >> 4) & 0x0F0F0F0F;
    vQs1 |= (cQh >> 12) & 0x00000010;   // 16 ->  4
    vQs1 |= (cQh >> 5) & 0x00001000;    // 17 -> 12
    vQs1 |= (cQh << 2) & 0x00100000;    // 18 -> 20
    vQs1 |= (cQh << 9) & 0x10000000;    // 19 -> 28
    vQs1 = __vsubss4(vQs1, 0x10101010); // subtract 16

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    vXQs[vI * cSramStride + cKbx * (2 * QI5_0) + cKqsx + 0] = vQs0;
    vXQs[vI * cSramStride + cKbx * (2 * QI5_0) + cKqsx + QI5_0] = vQs1;
#else
    vXQs[vI * (2 * MMQ_TILE_NE_K + 1) + cKbx * (2 * QI5_0) + cKqsx + 0] = vQs0;
    vXQs[vI * (2 * MMQ_TILE_NE_K + 1) + cKbx * (2 * QI5_0) + cKqsx + QI5_0] = vQs1;
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
  }

  constexpr int cBlocksPerTileXRow = MMQ_TILE_NE_K / QI5_0;
  constexpr int cRowsPerWarp = cWarpSize / cBlocksPerTileXRow;
  const int cKbxd = threadIdx.x % cBlocksPerTileXRow;

#pragma unroll
  for (int vI0 = 0; vI0 < cI; vI0 += cNwarps * cRowsPerWarp) {
    int vI = vI0 + threadIdx.y * cRowsPerWarp + threadIdx.x / cBlocksPerTileXRow;

    if (fallback) {
      vI = min(vI, pIMax);
    }

    const block_q5_0 *vBxi = (const block_q5_0 *)pX + pKbx0 + vI * pStride + cKbxd;

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    vXDf[vI * cSramStride + cKbxd] = vBxi->d;
#else
    vXDf[vI * (MMQ_TILE_NE_K / QI5_0) + vI / QI5_0 + cKbxd] = vBxi->d;
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE)  || defined(AMD_WMMA_AVAILABLE)
  }
}

template <ggml_type type, int J, bool fallback>
static __device__ __forceinline__ void ggml_cuda_mmq_load_tiles_q5_1(const char *__restrict__ pX,
                                                                     int *__restrict__ pXTile, const int pKbx0,
                                                                     const int pIMax, const int pStride) {
  constexpr int cWarpSize = ggml_cuda_get_physical_warp_size();
  constexpr int cNwarps = ggml_cuda_mmq_get_nthreads(type, J, fallback) / cWarpSize;
  constexpr int cI = ggml_cuda_mmq_get_I(type, J, fallback);
  constexpr int cSramStride = ggml_cuda_mmq_get_sram_stride(type, J, fallback);

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
  int *vXQs = (int *)pXTile;
  half2 *vXDm = (half2 *)(vXQs + 2 * MMQ_TILE_NE_K);
#else
  constexpr tile_x_sizes cTxs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_Q5_1, cI);
  int *vXQs = (int *)pXTile;
  half2 *vXDm = (half2 *)(vXQs + cTxs.qs);
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)

  constexpr int cThreadsPerRow = MMQ_ITER_K / (4 * QR5_1);
  constexpr int cNrows = cWarpSize / cThreadsPerRow;
  const int cTxi = cWarpSize > cThreadsPerRow ? threadIdx.x % cThreadsPerRow : threadIdx.x;
  const int cKbx = cTxi / QI5_1;
  const int cKqsx = cTxi % QI5_1;

#pragma unroll
  for (int vI0 = 0; vI0 < cI; vI0 += cNrows * cNwarps) {
    int vI = vI0 + (cNrows == 1 ? threadIdx.y : threadIdx.y * cNrows + threadIdx.x / cThreadsPerRow);

    if (fallback) {
      vI = min(vI, pIMax);
    }

    const block_q5_1 *vBxi = (const block_q5_1 *)pX + pKbx0 + vI * pStride + cKbx;

    const int cQl = get_int_b4(vBxi->qs, cKqsx);
    const int cQh = get_int_b4(vBxi->qh, 0) >> (4 * cKqsx);

    int vQs0 = (cQl >> 0) & 0x0F0F0F0F;
    vQs0 |= (cQh << 4) & 0x00000010;  // 0 ->  4
    vQs0 |= (cQh << 11) & 0x00001000; // 1 -> 12
    vQs0 |= (cQh << 18) & 0x00100000; // 2 -> 20
    vQs0 |= (cQh << 25) & 0x10000000; // 3 -> 28

    int vQs1 = (cQl >> 4) & 0x0F0F0F0F;
    vQs1 |= (cQh >> 12) & 0x00000010; // 16 ->  4
    vQs1 |= (cQh >> 5) & 0x00001000;  // 17 -> 12
    vQs1 |= (cQh << 2) & 0x00100000;  // 18 -> 20
    vQs1 |= (cQh << 9) & 0x10000000;  // 19 -> 28

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    vXQs[vI * cSramStride + cKbx * (2 * QI5_1) + cKqsx + 0] = vQs0;
    vXQs[vI * cSramStride + cKbx * (2 * QI5_1) + cKqsx + QI5_1] = vQs1;
#else
    vXQs[vI * (2 * MMQ_TILE_NE_K + 1) + cKbx * (2 * QI5_1) + cKqsx + 0] = vQs0;
    vXQs[vI * (2 * MMQ_TILE_NE_K + 1) + cKbx * (2 * QI5_1) + cKqsx + QI5_1] = vQs1;
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
  }

  constexpr int cBlocksPerTileXRow = MMQ_TILE_NE_K / QI5_1;
  constexpr int cRowsPerWarp = cWarpSize / cBlocksPerTileXRow;
  const int cKbxd = threadIdx.x % cBlocksPerTileXRow;

#pragma unroll
  for (int vI0 = 0; vI0 < cI; vI0 += cNwarps * cRowsPerWarp) {
    int vI = vI0 + threadIdx.y * cRowsPerWarp + threadIdx.x / cBlocksPerTileXRow;

    if (fallback) {
      vI = min(vI, pIMax);
    }

    const block_q5_1 *vBxi = (const block_q5_1 *)pX + pKbx0 + vI * pStride + cKbxd;

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    vXDm[vI * cSramStride + cKbxd] = vBxi->dm;
#else
    vXDm[vI * (MMQ_TILE_NE_K / QI5_1) + vI / QI5_1 + cKbxd] = vBxi->dm;
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
  }
}

template <ggml_type type, int J, bool fallback>
static __device__ __forceinline__ void ggml_cuda_mmq_load_tiles_q8_0(const char *__restrict__ pX,
                                                                     int *__restrict__ pXTile, const int pKbx0,
                                                                     const int pIMax, const int pStride) {
  constexpr int cWarpSize = ggml_cuda_get_physical_warp_size();
  constexpr int cNwarps = ggml_cuda_mmq_get_nthreads(type, J, fallback) / cWarpSize;
  constexpr int cI = ggml_cuda_mmq_get_I(type, J, fallback);
  constexpr int cSramStride = ggml_cuda_mmq_get_sram_stride(type, J, fallback);

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
  int *vXQs = (int *)pXTile;
  float *vXDf = (float *)(pXTile + 2 * MMQ_TILE_NE_K);
#else
  constexpr tile_x_sizes cTxs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_Q8_0, cI);
  int *vXQs = (int *)pXTile;
  float *vXDf = (float *)(vXQs + cTxs.qs);
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)

  // MMQ_ITER_K / (4 * QR8_0) == 64 required. but NV has only 32 threads per warp
  constexpr int cThreadsPerRow = 32;
  constexpr int cNrows = cWarpSize / cThreadsPerRow;
  const int cTxi = cWarpSize > cThreadsPerRow ? threadIdx.x % cThreadsPerRow : threadIdx.x;
  const int cKbx = cTxi / QI8_0;
  const int cKqsx = cTxi % QI8_0;

#pragma unroll
  for (int vI0 = 0; vI0 < cI; vI0 += cNrows * cNwarps) {
    int vI = vI0 + (cNrows == 1 ? threadIdx.y : threadIdx.y * cNrows + threadIdx.x / cThreadsPerRow);

    if (fallback) {
      vI = min(vI, pIMax);
    }

    const block_q8_0 *vBxi = (const block_q8_0 *)pX + pKbx0 + vI * pStride + cKbx;

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    vXQs[vI * cSramStride + 0 + cTxi] = get_int_b2(vBxi[0].qs, cKqsx);
    vXQs[vI * cSramStride + MMQ_TILE_NE_K + cTxi] = get_int_b2(vBxi[MMQ_TILE_NE_K / QI8_0].qs, cKqsx);
#else
    vXQs[vI * (2 * MMQ_TILE_NE_K + 1) + 0 + cTxi] = get_int_b2(vBxi[0].qs, cKqsx);
    vXQs[vI * (2 * MMQ_TILE_NE_K + 1) + MMQ_TILE_NE_K + cTxi] = get_int_b2(vBxi[MMQ_TILE_NE_K / QI8_0].qs, cKqsx);
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
  }

  constexpr int cBlocksPerTileXRow = 2 * MMQ_TILE_NE_K / QI8_0;
  constexpr int cRowsPerWarp = cWarpSize / cBlocksPerTileXRow;
  const int cKbxd = threadIdx.x % cBlocksPerTileXRow;

#pragma unroll
  for (int vI0 = 0; vI0 < cI; vI0 += cNwarps * cRowsPerWarp) {
    int vI = vI0 + threadIdx.y * cRowsPerWarp + threadIdx.x / cBlocksPerTileXRow;

    if (fallback) {
      vI = min(vI, pIMax);
    }

    const block_q8_0 *vBxi = (const block_q8_0 *)pX + pKbx0 + vI * pStride + cKbxd;

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    vXDf[vI * cSramStride + cKbxd] = vBxi->d;
#else
    vXDf[vI * (2 * MMQ_TILE_NE_K / QI8_0) + vI / (QI8_0 / 2) + cKbxd] = vBxi->d;
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
  }
}

// ---------------------------------------------------------------------------------------------

template <ggml_type type, int J, bool fallback>
static __device__ __forceinline__ void ggml_cuda_mmq_load_tiles_q2_K(const char *__restrict__ pX,
                                                                     int *__restrict__ pXTile, const int pKbx0,
                                                                     const int pIMax, const int pStride) {
  constexpr int cWarpSize = ggml_cuda_get_physical_warp_size();
  constexpr int cNwarps = ggml_cuda_mmq_get_nthreads(type, J, fallback) / cWarpSize;
  constexpr int cI = ggml_cuda_mmq_get_I(type, J, fallback);
  constexpr int cSramStride = ggml_cuda_mmq_get_sram_stride(type, J, fallback);

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
  int *vXQs = (int *)pXTile;
  half2 *vXDm = (half2 *)(vXQs + 2 * MMQ_TILE_NE_K);
#else
  constexpr tile_x_sizes cTxs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_Q2_K, cI);
  int *vXQs = (int *)pXTile;
  half2 *vXDm = (half2 *)(vXQs + cTxs.qs);
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)

  constexpr int cThreadsPerRow = MMQ_ITER_K / (4 * QR2_K);
  constexpr int cNrows = ggml_cuda_get_physical_warp_size() / cThreadsPerRow;
  const int cKqsx = threadIdx.x % cThreadsPerRow;

#pragma unroll
  for (int vI0 = 0; vI0 < cI; vI0 += cNrows * cNwarps) {
    int vI = vI0 + threadIdx.y * cNrows + threadIdx.x / cThreadsPerRow;

    if (fallback) {
      vI = min(vI, pIMax);
    }

    const block_q2_K *vBxi = (const block_q2_K *)pX + pKbx0 + vI * pStride;

    const int cXQl0 = get_int_b2(vBxi->qs, cKqsx);

#pragma unroll
    for (int vL = 0; vL < QR2_K; ++vL) {
      const int cK = (cKqsx / 8) * 32 + vL * 8 + cKqsx % 8;

      const int cXQsK = (cXQl0 >> (2 * vL)) & 0x03030303;

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
      vXQs[vI * cSramStride + cK] = cXQsK;
#else
      vXQs[vI * (2 * MMQ_TILE_NE_K + 1) + cK] = cXQsK;
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    }

    const int cScM = vBxi->scales[cKqsx];
#ifdef FAST_FP16_AVAILABLE
    const half2 cXDmIk = __hmul2(vBxi->dm, make_half2(cScM & 0x0F, cScM >> 4));
#else
    const float2 cBxiDmf = __half22float2(vBxi->dm);
    const half2 cXDmIk = make_half2(cBxiDmf.x * (cScM & 0x0F), cBxiDmf.y * (cScM >> 4));
#endif // FAST_FP16_AVAILABLE

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    vXDm[vI * cSramStride + cKqsx] = cXDmIk;
#else
    vXDm[vI * (MMQ_TILE_NE_K + 1) + cKqsx] = cXDmIk;
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
  }
}

template <ggml_type type, int J, bool fallback>
static __device__ __forceinline__ void ggml_cuda_mmq_load_tiles_q3_K(const char *__restrict__ pX,
                                                                     int *__restrict__ pXTile, const int pKbx0,
                                                                     const int pIMax, const int pStride) {
  constexpr int cWarpSize = ggml_cuda_get_physical_warp_size();
  constexpr int cNwarps = ggml_cuda_mmq_get_nthreads(type, J, fallback) / cWarpSize;
  constexpr int cI = ggml_cuda_mmq_get_I(type, J, fallback);
  constexpr int cSramStride = ggml_cuda_mmq_get_sram_stride(type, J, fallback);

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
  int *vXQs = (int *)pXTile;
  float *vXDf = (float *)(vXQs + MMQ_TILE_NE_K * 2);
#else
  constexpr tile_x_sizes cTxs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_Q3_K, cI);
  int *vXQs = (int *)pXTile;
  float *vXDf = (float *)(vXQs + cTxs.qs);
  int *vXSc = (int *)(vXDf + cTxs.dm);
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE)

  constexpr int cThreadsPerRow = MMQ_ITER_K / (4 * QR3_K);
  constexpr int cNrows = cWarpSize / cThreadsPerRow;
  const int cKqsx = threadIdx.x % cThreadsPerRow;

#pragma unroll
  for (int vI0 = 0; vI0 < cI; vI0 += cNrows * cNwarps) {
    int vI = vI0 + threadIdx.y * cNrows + threadIdx.x / cThreadsPerRow;

    if (fallback) {
      vI = min(vI, pIMax);
    }

    const block_q3_K *vBxi = (const block_q3_K *)pX + pKbx0 + vI * pStride;

    const int cXQl0 = get_int_b2(vBxi->qs, cKqsx);
    const int cXQh0 = get_int_b2(vBxi->hmask, cKqsx % (QI3_K / 2)) >> (4 * (cKqsx / (QI3_K / 2)));

#pragma unroll
    for (int vL = 0; vL < QR3_K; ++vL) {
      const int cK = (cKqsx / 8) * 32 + vL * 8 + cKqsx % 8;

      const int cXQlK = (cXQl0 >> (2 * vL)) & 0x03030303;
      const int cXQhK = ((cXQh0 >> vL) << 2) & 0x04040404;

      const int cXQsK = __vsubss4(cXQlK | cXQhK, 0x04040404);

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
      vXQs[vI * cSramStride + cK] = cXQsK;
#else
      vXQs[vI * (2 * MMQ_TILE_NE_K + 1) + cK] = cXQsK;
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    }
  }

  constexpr int cRowsPerWarp = cWarpSize / 4;
#pragma unroll
  for (int vI0 = 0; vI0 < cI; vI0 += cNwarps * cRowsPerWarp) {
    int vI = vI0 + threadIdx.y * cRowsPerWarp + threadIdx.x / 4;

    if (fallback) {
      vI = min(vI, pIMax);
    }

    const block_q3_K *vBxi = (const block_q3_K *)pX + pKbx0 + vI * pStride;

    const int cKsc = threadIdx.x % 4;

    const int cKscLow = cKsc % (QI3_K / 8);
    const int cShiftLow = 4 * (cKsc / (QI3_K / 8));
    const int cScLow = (get_int_b2(vBxi->scales, cKscLow) >> cShiftLow) & 0x0F0F0F0F;

    const int cKscHigh = QI3_K / 8;
    const int cShiftHigh = 2 * cKsc;
    const int cScHigh = ((get_int_b2(vBxi->scales, cKscHigh) >> cShiftHigh) << 4) & 0x30303030;

    const int cSc = __vsubss4(cScLow | cScHigh, 0x20202020);

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    const int8_t *vSc8 = (const int8_t *)&cSc;
    const float cD = vBxi->d;

#pragma unroll
    for (int vL = 0; vL < int(sizeof(int)); ++vL) {
      vXDf[vI * cSramStride + sizeof(int) * cKsc + vL] = cD * vSc8[vL];
    }
#else
    vXSc[vI * (MMQ_TILE_NE_K / 8) + vI / 8 + cKsc] = cSc;
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
  }

#if !(defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE))
#pragma unroll
  for (int vI0 = 0; vI0 < cI; vI0 += cNwarps * cWarpSize) {
    int vI = (vI0 + threadIdx.y * cWarpSize + threadIdx.x) % cI;

    if (fallback) {
      vI = min(vI, pIMax);
    }

    const block_q3_K *vBxi = (const block_q3_K *)pX + pKbx0 + vI * pStride;

    vXDf[vI] = vBxi->d;
  }
#endif // !(defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE)) || defined(AMD_WMMA_AVAILABLE)
}

static __device__ __forceinline__ int unpack_scales_q45_K(const int *pScales, const int pKsc) {
  // scale arrangement after the following two lines:
  //   - ksc == 0: sc0, sc1, sc2, sc3
  //   - ksc == 1: sc4, sc5, sc6, sc7
  //   - ksc == 2:  m0,  m1,  m2,  m3
  //   - ksc == 3:  m4,  m5,  m6,  m7
  return ((pScales[(pKsc % 2) + (pKsc != 0)] >> (4 * (pKsc & (pKsc / 2)))) & 0x0F0F0F0F) | // lower 4 bits
         ((pScales[pKsc / 2] >> (2 * (pKsc % 2))) & 0x30303030);                           // upper 2 bits
}

template <ggml_type type, int J, bool fallback>
static __device__ __forceinline__ void ggml_cuda_mmq_load_tiles_q4_K(const char *__restrict__ pX,
                                                                     int *__restrict__ pXTile, const int pKbx0,
                                                                     const int pIMax, const int pStride) {
  constexpr int cWarpSize = ggml_cuda_get_physical_warp_size();
  constexpr int cNwarps = ggml_cuda_mmq_get_nthreads(type, J, fallback) / cWarpSize;
  constexpr int cI = ggml_cuda_mmq_get_I(type, J, fallback);
  constexpr int cSramStride = ggml_cuda_mmq_get_sram_stride(type, J, fallback);

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
  int *vXQs = (int *)pXTile;
  half2 *vXDm = (half2 *)(vXQs + 2 * MMQ_TILE_NE_K);
#else
  constexpr tile_x_sizes cTxs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_Q4_K, cI);
  int *vXQs = (int *)pXTile;
  half2 *vXDm = (half2 *)(vXQs + cTxs.qs);
  int *vXSc = (int *)(vXDm + cTxs.dm);
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)

  constexpr int cThreadsPerRow = MMQ_ITER_K / (4 * QR4_K);
  constexpr int cNrows = cWarpSize / cThreadsPerRow;
  const int cTxi = cWarpSize > cThreadsPerRow ? threadIdx.x % cThreadsPerRow : threadIdx.x;

#pragma unroll
  for (int vI0 = 0; vI0 < cI; vI0 += cNrows * cNwarps) {
    int vI = vI0 + (cNrows == 1 ? threadIdx.y : threadIdx.y * cNrows + threadIdx.x / cThreadsPerRow);

    if (fallback) {
      vI = min(vI, pIMax);
    }

    const block_q4_K *vBxi = (const block_q4_K *)pX + pKbx0 + vI * pStride;
    const int cQs0 = get_int_b4(vBxi->qs, cTxi);

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    vXQs[vI * cSramStride + 16 * (cTxi / 8) + cTxi % 8 + 0] = (cQs0 >> 0) & 0x0F0F0F0F;
    vXQs[vI * cSramStride + 16 * (cTxi / 8) + cTxi % 8 + 8] = (cQs0 >> 4) & 0x0F0F0F0F;
#else
    vXQs[vI * (MMQ_TILE_NE_K + 1) + cTxi] = cQs0;
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
  }

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
  constexpr int cRowsPerWarp = cWarpSize / 2;
#pragma unroll
  for (int vI0 = 0; vI0 < cI; vI0 += cNwarps * cRowsPerWarp) {
#if defined(AMD_MFMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    // Need if on AMD instead of % because warp_size == 64
    // This causes double work and throughput loss (MI300X)
    // H100 loses about 100 t/s with 'if' condition over '%'
    int vI = vI0 + threadIdx.y * cRowsPerWarp + threadIdx.x / 2;
    if (vI < cI) {
#else
    int vI = (vI0 + threadIdx.y * cRowsPerWarp + threadIdx.x / 2) % cI;
    {
#endif // defined(AMD_MFMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
      if (fallback) {
        vI = min(vI, pIMax);
      }

      const block_q4_K *vBxi = (const block_q4_K *)pX + pKbx0 + vI * pStride;

      const int *vScales = (const int *)vBxi->scales;
      const int cKsc = threadIdx.x % 2;

      const int cSc32 = unpack_scales_q45_K(vScales, cKsc + 0);
      const int cM32 = unpack_scales_q45_K(vScales, cKsc + 2);

      const uint8_t *vSc8 = (const uint8_t *)&cSc32;
      const uint8_t *vM8 = (const uint8_t *)&cM32;

      const half2 cDm = vBxi->dm * make_half2(1.0f, -1.0f);

#pragma unroll
      for (int vL = 0; vL < sizeof(int); ++vL) {
        vXDm[vI * cSramStride + sizeof(int) * cKsc + vL] = cDm * make_half2(vSc8[vL], vM8[vL]);
      }
    }
  }
#else
#pragma unroll
  for (int vI0 = 0; vI0 < cI; vI0 += cNwarps * cWarpSize) {
    int vI = (vI0 + threadIdx.y * cWarpSize + threadIdx.x) % cI;

    if (fallback) {
      vI = min(vI, pIMax);
    }

    const block_q4_K *vBxi = (const block_q4_K *)pX + pKbx0 + vI * pStride;

    vXDm[vI] = vBxi->dm;
  }
  constexpr int cRowsPerWarp = cWarpSize / 4;
#pragma unroll
  for (int vI0 = 0; vI0 < cI; vI0 += cNwarps * cRowsPerWarp) {
    int vI = (vI0 + threadIdx.y * cRowsPerWarp + threadIdx.x / (MMQ_TILE_NE_K / 8)) % cI;

    if (fallback) {
      vI = min(vI, pIMax);
    }

    const block_q4_K *vBxi =
        (const block_q4_K *)pX + pKbx0 + vI * pStride + (threadIdx.x % (MMQ_TILE_NE_K / 8)) / (QI4_K / 8);

    const int *vScales = (const int *)vBxi->scales;

    const int cKsc = threadIdx.x % (MMQ_TILE_NE_K / 8);
    const int cScales8 = unpack_scales_q45_K(vScales, cKsc);

    vXSc[vI * (MMQ_TILE_NE_K / 8) + vI / 8 + cKsc] = cScales8;
  }
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
}

template <ggml_type type, int J, bool fallback>
static __device__ __forceinline__ void ggml_cuda_mmq_load_tiles_q5_K(const char *__restrict__ pX,
                                                                     int *__restrict__ pXTile, const int pKbx0,
                                                                     const int pIMax, const int pStride) {
  constexpr int cWarpSize = ggml_cuda_get_physical_warp_size();
  constexpr int cNwarps = ggml_cuda_mmq_get_nthreads(type, J, fallback) / cWarpSize;
  constexpr int cI = ggml_cuda_mmq_get_I(type, J, fallback);
  constexpr int cSramStride = ggml_cuda_mmq_get_sram_stride(type, J, fallback);

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
  int *vXQs = (int *)pXTile;
  half2 *vXDm = (half2 *)(vXQs + MMQ_TILE_NE_K * 2);
#else
  constexpr tile_x_sizes cTxs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_Q5_K, cI);
  int *vXQs = (int *)pXTile;
  half2 *vXDm = (half2 *)(vXQs + cTxs.qs);
  int *vXSc = (int *)(vXDm + cTxs.dm);
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE)

  constexpr int cThreadsPerRow = MMQ_ITER_K / (4 * QR5_K);
  constexpr int cNrows = cWarpSize / cThreadsPerRow;
  const int cTxi = cWarpSize > cThreadsPerRow ? threadIdx.x % cThreadsPerRow : threadIdx.x;

#pragma unroll
  for (int vI0 = 0; vI0 < cI; vI0 += cNrows * cNwarps) {
    int vI = vI0 + (cNrows == 1 ? threadIdx.y : threadIdx.y * cNrows + threadIdx.x / cThreadsPerRow);

    if (fallback) {
      vI = min(vI, pIMax);
    }

    const block_q5_K *vBxi = (const block_q5_K *)pX + pKbx0 + vI * pStride;
    const int cKy = QR5_K * cTxi;

    const int cQl = get_int_b4(vBxi->qs, cTxi);
    const int cQl0 = (cQl >> 0) & 0x0F0F0F0F;
    const int cQl1 = (cQl >> 4) & 0x0F0F0F0F;

    const int cQh = get_int_b4(vBxi->qh, cTxi % (QI5_K / 4));
    const int cQh0 = ((cQh >> (2 * (cTxi / (QI5_K / 4)) + 0)) << 4) & 0x10101010;
    const int cQh1 = ((cQh >> (2 * (cTxi / (QI5_K / 4)) + 1)) << 4) & 0x10101010;

    const int cKq0 = cKy - cKy % (QI5_K / 2) + cTxi % (QI5_K / 4) + 0;
    const int cKq1 = cKy - cKy % (QI5_K / 2) + cTxi % (QI5_K / 4) + QI5_K / 4;

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    vXQs[vI * cSramStride + cKq0] = cQl0 | cQh0;
    vXQs[vI * cSramStride + cKq1] = cQl1 | cQh1;
#else
    vXQs[vI * (2 * MMQ_TILE_NE_K + 1) + cKq0] = cQl0 | cQh0;
    vXQs[vI * (2 * MMQ_TILE_NE_K + 1) + cKq1] = cQl1 | cQh1;
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
  }

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
  constexpr int cRowsPerWarp = cWarpSize / 2;
#pragma unroll
  for (int vI0 = 0; vI0 < cI; vI0 += cNwarps * cRowsPerWarp) {
#if defined(AMD_MFMA_AVAILABLE)
    // Need if on AMD instead of % because warp_size == 64
    // This causes double work and throughput loss (MI300X)
    // H100 loses about 100 t/s with 'if' condition over '%'
    int vI = vI0 + threadIdx.y * cRowsPerWarp + threadIdx.x / 2;
    if (vI < cI) {
#else
    int vI = (vI0 + threadIdx.y * cRowsPerWarp + threadIdx.x / 2) % cI;
    {
#endif // defined(AMD_MFMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
      if (fallback) {
        vI = min(vI, pIMax);
      }

      const block_q5_K *vBxi = (const block_q5_K *)pX + pKbx0 + vI * pStride;

      const int *vScales = (const int *)vBxi->scales;
      const int cKsc = threadIdx.x % 2;

      const int cSc32 = unpack_scales_q45_K(vScales, cKsc + 0);
      const int cM32 = unpack_scales_q45_K(vScales, cKsc + 2);

      const uint8_t *vSc8 = (const uint8_t *)&cSc32;
      const uint8_t *vM8 = (const uint8_t *)&cM32;

      const half2 cDm = vBxi->dm * make_half2(1.0f, -1.0f);

#pragma unroll
      for (int vL = 0; vL < int(sizeof(int)); ++vL) {
        vXDm[vI * cSramStride + sizeof(int) * cKsc + vL] = cDm * make_half2(vSc8[vL], vM8[vL]);
      }
    }
  }
#else
#pragma unroll
  for (int vI0 = 0; vI0 < cI; vI0 += cNwarps * cWarpSize) {
    int vI = (vI0 + threadIdx.y * cWarpSize + threadIdx.x) % cI;

    if (fallback) {
      vI = min(vI, pIMax);
    }

    const block_q5_K *vBxi = (const block_q5_K *)pX + pKbx0 + vI * pStride;

    vXDm[vI] = vBxi->dm;
  }

  constexpr int cRowsPerWarp = cWarpSize / 4;
#pragma unroll
  for (int vI0 = 0; vI0 < cI; vI0 += cNwarps * cRowsPerWarp) {
    int vI = (vI0 + threadIdx.y * cRowsPerWarp + threadIdx.x / (MMQ_TILE_NE_K / 8)) % cI;

    if (fallback) {
      vI = min(vI, pIMax);
    }

    const block_q5_K *vBxi = (const block_q5_K *)pX + pKbx0 + vI * pStride;

    const int *vScales = (const int *)vBxi->scales;

    const int cKsc = threadIdx.x % (MMQ_TILE_NE_K / 8);
    const int cScales8 = unpack_scales_q45_K(vScales, cKsc);

    vXSc[vI * (MMQ_TILE_NE_K / 8) + vI / 8 + cKsc] = cScales8;
  }
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
}

template <ggml_type type, int J, bool fallback>
static __device__ __forceinline__ void ggml_cuda_mmq_load_tiles_q6_K(const char *__restrict__ pX,
                                                                     int *__restrict__ pXTile, const int pKbx0,
                                                                     const int pIMax, const int pStride) {
  constexpr int cWarpSize = ggml_cuda_get_physical_warp_size();
  constexpr int cNwarps = ggml_cuda_mmq_get_nthreads(type, J, fallback) / cWarpSize;
  constexpr int cI = ggml_cuda_mmq_get_I(type, J, fallback);
  constexpr int cSramStride = ggml_cuda_mmq_get_sram_stride(type, J, fallback);

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
  int *vXQs = (int *)pXTile;
  float *vXDf = (float *)(vXQs + MMQ_TILE_NE_K * 2);
  int *vXSc = (int *)(vXDf + MMQ_TILE_NE_K / QI6_K);
#else
  constexpr tile_x_sizes cTxs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_Q6_K, cI);
  int *vXQs = (int *)pXTile;
  float *vXDf = (float *)(vXQs + cTxs.qs);
  int *vXSc = (int *)(vXDf + cTxs.dm);
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)

  constexpr int cThreadsPerRow = MMQ_ITER_K / (4 * QR6_K);
  constexpr int cNrows = cWarpSize / cThreadsPerRow;
  const int cTxi = cWarpSize > cThreadsPerRow ? threadIdx.x % cThreadsPerRow : threadIdx.x;

#pragma unroll
  for (int vI0 = 0; vI0 < cI; vI0 += cNrows * cNwarps) {
    int vI = vI0 + (cNrows == 1 ? threadIdx.y : threadIdx.y * cNrows + threadIdx.x / cThreadsPerRow);

    if (fallback) {
      vI = min(vI, pIMax);
    }

    const block_q6_K *vBxi = (const block_q6_K *)pX + pKbx0 + vI * pStride;

    const int cQl = get_int_b2(vBxi->ql, cTxi);
    const int cQl0 = (cQl >> 0) & 0x0F0F0F0F;
    const int cQl1 = (cQl >> 4) & 0x0F0F0F0F;

    const int cQh = get_int_b2(vBxi->qh, (QI6_K / 4) * (cTxi / (QI6_K / 2)) + cTxi % (QI6_K / 4));
    const int cQh0 = ((cQh >> ((cTxi & 0x08) >> 2)) << 4) & 0x30303030;
    const int cQh1 = (cQh >> ((cTxi & 0x08) >> 2)) & 0x30303030;

    const int cKq0 = 2 * cTxi - cTxi % (QI6_K / 2) + 0;
    const int cKq1 = 2 * cTxi - cTxi % (QI6_K / 2) + QI6_K / 2;

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    vXQs[vI * cSramStride + cKq0] = __vsubss4(cQl0 | cQh0, 0x20202020);
    vXQs[vI * cSramStride + cKq1] = __vsubss4(cQl1 | cQh1, 0x20202020);
#else
    vXQs[vI * (2 * MMQ_TILE_NE_K + 1) + cKq0] = __vsubss4(cQl0 | cQh0, 0x20202020);
    vXQs[vI * (2 * MMQ_TILE_NE_K + 1) + cKq1] = __vsubss4(cQl1 | cQh1, 0x20202020);
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
  }

#pragma unroll
  for (int vI0 = 0; vI0 < cI; vI0 += cNwarps * cWarpSize) {
    int vI = (vI0 + threadIdx.y * cWarpSize + threadIdx.x) % cI;

    if (fallback) {
      vI = min(vI, pIMax);
    }

    const block_q6_K *vBxi = (const block_q6_K *)pX + pKbx0 + vI * pStride;

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    vXDf[vI * cSramStride] = vBxi->d;
#else
    vXDf[vI * (MMQ_TILE_NE_K / QI6_K) + vI / QI6_K] = vBxi->d;
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
  }

  constexpr int cRowsPerWarp = cWarpSize / 4;
#pragma unroll
  for (int vI0 = 0; vI0 < cI; vI0 += cNwarps * cRowsPerWarp) {
    int vI = (vI0 + threadIdx.y * cRowsPerWarp + threadIdx.x / (MMQ_TILE_NE_K / 8)) % cI;

    if (fallback) {
      vI = min(vI, pIMax);
    }

    const block_q6_K *vBxi = (const block_q6_K *)pX + pKbx0 + vI * pStride + (threadIdx.x % (MMQ_TILE_NE_K / 8)) / 4;

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    vXSc[vI * cSramStride + threadIdx.x % 4] = get_int_b2(vBxi->scales, threadIdx.x % (MMQ_TILE_NE_K / 8));
#else
    vXSc[vI * (MMQ_TILE_NE_K / 8) + vI / 8 + threadIdx.x % (MMQ_TILE_NE_K / 8)] =
        get_int_b2(vBxi->scales, threadIdx.x % (QI6_K / 8));
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
  }
}

// ---------------------------------------------------------------------------------------------

template <ggml_type type, int J, bool fallback>
static __device__ __forceinline__ void ggml_cuda_mmq_load_tiles_iq1_s(const char *__restrict__ pX,
                                                                      int *__restrict__ pXTile, const int pKbx0,
                                                                      const int pIMax, const int pStride) {
  constexpr int cWarpSize = ggml_cuda_get_physical_warp_size();
  constexpr int cNwarps = ggml_cuda_mmq_get_nthreads(type, J, fallback) / cWarpSize;
  constexpr int cI = ggml_cuda_mmq_get_I(type, J, fallback);
  constexpr int cSramStride = ggml_cuda_mmq_get_sram_stride(type, J, fallback);

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
  int *vXQs = (int *)pXTile;
  half2 *vXDs = (half2 *)(vXQs + MMQ_TILE_NE_K * 2);
#else
  constexpr tile_x_sizes cTxs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_IQ3_S, cI);
  int *vXQs = (int *)pXTile;
  half2 *vXDs = (half2 *)(vXQs + cTxs.qs);
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)

  constexpr int cThreadsPerRow = MMQ_ITER_K / (4 * QR1_S);
  constexpr int cNrows = cWarpSize / cThreadsPerRow;
  const int cKqsx = threadIdx.x % cThreadsPerRow;

#pragma unroll
  for (int vI0 = 0; vI0 < cI; vI0 += cNwarps * cNrows) {
    int vI = vI0 + threadIdx.y * cNrows + threadIdx.x / cThreadsPerRow;

    if (fallback) {
      vI = min(vI, pIMax);
    }

    const block_iq1_s *vBxi = (const block_iq1_s *)pX + pKbx0 + vI * pStride;

    const int cQsPacked = get_int_b2(vBxi->qs, cKqsx);
    const uint8_t *vQs = (const uint8_t *)&cQsPacked;

    const int cQh = vBxi->qh[cKqsx];

#pragma unroll
    for (int vL = 0; vL < QR1_S / 2; ++vL) {
      const int cGrid = iq1s_grid_gpu[vQs[vL] | (((cQh >> (3 * vL)) & 0x07) << 8)];

      const int cGrid0 = (cGrid >> 0) & 0x0F0F0F0F;
      const int cGrid1 = (cGrid >> 4) & 0x0F0F0F0F;

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
      vXQs[vI * cSramStride + 8 * cKqsx + (2 * vL + 0)] = cGrid0;
      vXQs[vI * cSramStride + 8 * cKqsx + (2 * vL + 1)] = cGrid1;
#else
      vXQs[vI * (2 * MMQ_TILE_NE_K + 1) + 8 * cKqsx + (2 * vL + 0)] = cGrid0;
      vXQs[vI * (2 * MMQ_TILE_NE_K + 1) + 8 * cKqsx + (2 * vL + 1)] = cGrid1;
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    }

    const float cD1q = __half2float(vBxi->d) * (((cQh >> 11) & 0x0E) + 1);
    const float cDelta = -1.0f + IQ1S_DELTA - (cQh & 0x8000) * (2.0f * IQ1S_DELTA / 0x8000);

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    vXDs[vI * cSramStride + cKqsx] = make_half2(cD1q, cD1q * cDelta);
#else
    vXDs[vI * (MMQ_TILE_NE_K / 4) + vI / 4 + cKqsx] = make_half2(cD1q, cD1q * cDelta);
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
  }
}

template <ggml_type type, int J, bool fallback>
static __device__ __forceinline__ void ggml_cuda_mmq_load_tiles_iq2_xxs(const char *__restrict__ pX,
                                                                        int *__restrict__ pXTile, const int pKbx0,
                                                                        const int pIMax, const int pStride) {
  constexpr int cWarpSize = ggml_cuda_get_physical_warp_size();
  constexpr int cNwarps = ggml_cuda_mmq_get_nthreads(type, J, fallback) / cWarpSize;
  constexpr int cI = ggml_cuda_mmq_get_I(type, J, fallback);
  constexpr int cSramStride = ggml_cuda_mmq_get_sram_stride(type, J, fallback);

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
  int *vXQs = (int *)pXTile;
  float *vXDf = (float *)(vXQs + MMQ_TILE_NE_K * 2);
#else
  constexpr tile_x_sizes cTxs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_IQ2_XXS, cI);
  int *vXQs = (int *)pXTile;
  float *vXDf = (float *)(vXQs + cTxs.qs);
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)

  constexpr int cThreadsPerRow = (MMQ_ITER_K / (4 * QR2_XXS)) / 2;
  constexpr int cNrows = cWarpSize / cThreadsPerRow;
  const int cKqsx = cWarpSize > cThreadsPerRow ? threadIdx.x % cThreadsPerRow : threadIdx.x;

#pragma unroll
  for (int vI0 = 0; vI0 < cI; vI0 += cNwarps * cNrows) {
    int vI = vI0 + threadIdx.y * cNrows + threadIdx.x / cThreadsPerRow;

    if (fallback) {
      vI = min(vI, pIMax);
    }

    const block_iq2_xxs *vBxi = (const block_iq2_xxs *)pX + pKbx0 + vI * pStride;

    const int cQ2 = get_int_b2(vBxi->qs, 2 * cKqsx + 0);
    const uint8_t *vAux8 = (const uint8_t *)&cQ2;
    const uint32_t cAux32 = get_int_b2(vBxi->qs, 2 * cKqsx + 1);

#pragma unroll
    for (int vL = 0; vL < QR2_XXS; ++vL) {
      const uint2 cGridPos = ((const uint2 *)iq2xxs_grid)[vAux8[vL]];
      const uint32_t cSigns = unpack_ksigns(cAux32 >> (7 * vL));

      const int cSigns0 = __vcmpne4(cSigns & 0x08040201, 0);
      const int cGrid0 = __vsub4(cGridPos.x ^ cSigns0, cSigns0);

      const int cSigns1 = __vcmpne4(cSigns & 0x80402010, 0);
      const int cGrid1 = __vsub4(cGridPos.y ^ cSigns1, cSigns1);

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
      vXQs[vI * cSramStride + 8 * cKqsx + (2 * vL + 0)] = cGrid0;
      vXQs[vI * cSramStride + 8 * cKqsx + (2 * vL + 1)] = cGrid1;
#else
      vXQs[vI * (2 * MMQ_TILE_NE_K + 1) + 8 * cKqsx + (2 * vL + 0)] = cGrid0;
      vXQs[vI * (2 * MMQ_TILE_NE_K + 1) + 8 * cKqsx + (2 * vL + 1)] = cGrid1;
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    }

    const int cLs = cAux32 >> 27 | 1; // (scale * 2 + 1)
    const float cD = vBxi->d;
#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    vXDf[vI * cSramStride + cKqsx] = cD * cLs / 8; // (d * scale + d / 2) / 4
#else
    vXDf[vI * (MMQ_TILE_NE_K / 4) + vI / 4 + cKqsx] = cD * cLs / 8; // (d * scale + d / 2) / 4
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE)  || defined(AMD_WMMA_AVAILABLE)
  }
}

template <ggml_type type, int J, bool fallback>
static __device__ __forceinline__ void ggml_cuda_mmq_load_tiles_iq2_xs(const char *__restrict__ pX,
                                                                       int *__restrict__ pXTile, const int pKbx0,
                                                                       const int pIMax, const int pStride) {
  constexpr int cWarpSize = ggml_cuda_get_physical_warp_size();
  constexpr int cNwarps = ggml_cuda_mmq_get_nthreads(type, J, fallback) / cWarpSize;
  constexpr int cI = ggml_cuda_mmq_get_I(type, J, fallback);
  constexpr int cSramStride = ggml_cuda_mmq_get_sram_stride(type, J, fallback);

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
  int *vXQs = (int *)pXTile;
  float *vXDf = (float *)(vXQs + MMQ_TILE_NE_K * 2);
#else
  constexpr tile_x_sizes cTxs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_IQ2_XS, cI);
  int *vXQs = (int *)pXTile;
  float *vXDf = (float *)(vXQs + cTxs.qs);
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)

  constexpr int cThreadsPerRow = (MMQ_ITER_K / (4 * QR2_XS)) / 2;
  constexpr int cNrows = cWarpSize / cThreadsPerRow;
  const int cKqsx = threadIdx.x % cThreadsPerRow;

#pragma unroll
  for (int vI0 = 0; vI0 < cI; vI0 += cNwarps * cNrows) {
    int vI = vI0 + threadIdx.y * cNrows + threadIdx.x / cThreadsPerRow;

    if (fallback) {
      vI = min(vI, pIMax);
    }

    const block_iq2_xs *vBxi = (const block_iq2_xs *)pX + pKbx0 + vI * pStride;

    const int2 cQ2Packed = make_int2(get_int_b2(vBxi->qs, 2 * cKqsx + 0), get_int_b2(vBxi->qs, 2 * cKqsx + 1));
    const uint16_t *vQ2 = (const uint16_t *)&cQ2Packed;

#pragma unroll
    for (int vL = 0; vL < QR2_XS; ++vL) {
      const uint2 cGridPos = ((const uint2 *)iq2xs_grid)[vQ2[vL] & 0x1FF];
      const uint32_t cSigns = unpack_ksigns(vQ2[vL] >> 9);

      const int cSigns0 = __vcmpne4(cSigns & 0x08040201, 0);
      const int cGridL = __vsub4(cGridPos.x ^ cSigns0, cSigns0);

      const int cSigns1 = __vcmpne4(cSigns & 0x80402010, 0);
      const int cGridH = __vsub4(cGridPos.y ^ cSigns1, cSigns1);

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
      vXQs[vI * cSramStride + 8 * cKqsx + (2 * vL + 0)] = cGridL;
      vXQs[vI * cSramStride + 8 * cKqsx + (2 * vL + 1)] = cGridH;
#else
      vXQs[vI * (2 * MMQ_TILE_NE_K + 1) + 8 * cKqsx + (2 * vL + 0)] = cGridL;
      vXQs[vI * (2 * MMQ_TILE_NE_K + 1) + 8 * cKqsx + (2 * vL + 1)] = cGridH;
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    }

    const int cLs = vBxi->scales[cKqsx];
    const float cD = vBxi->d;
#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    vXDf[vI * cSramStride + 2 * cKqsx + 0] = ((cLs & 0x0F) * cD + cD / 2) / 4;
    vXDf[vI * cSramStride + 2 * cKqsx + 1] = ((cLs >> 4) * cD + cD / 2) / 4;
#else
    vXDf[vI * (2 * MMQ_TILE_NE_K * 2 / QI8_0) + vI / (QI8_0 / 4) + 2 * cKqsx + 0] = ((cLs & 0x0F) * cD + cD / 2) / 4;
    vXDf[vI * (2 * MMQ_TILE_NE_K * 2 / QI8_0) + vI / (QI8_0 / 4) + 2 * cKqsx + 1] = ((cLs >> 4) * cD + cD / 2) / 4;
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
  }
}

template <ggml_type type, int J, bool fallback>
static __device__ __forceinline__ void ggml_cuda_mmq_load_tiles_iq2_s(const char *__restrict__ pX,
                                                                      int *__restrict__ pXTile, const int pKbx0,
                                                                      const int pIMax, const int pStride) {
  constexpr int cWarpSize = ggml_cuda_get_physical_warp_size();
  constexpr int cNwarps = ggml_cuda_mmq_get_nthreads(type, J, fallback) / cWarpSize;
  constexpr int cI = ggml_cuda_mmq_get_I(type, J, fallback);
  constexpr int cSramStride = ggml_cuda_mmq_get_sram_stride(type, J, fallback);

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
  int *vXQs = (int *)pXTile;
  float *vXDf = (float *)(vXQs + MMQ_TILE_NE_K * 2);
#else
  constexpr tile_x_sizes cTxs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_IQ2_S, cI);
  int *vXQs = (int *)pXTile;
  float *vXDf = (float *)(vXQs + cTxs.qs);
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
  constexpr int cThreadsPerRow = (MMQ_ITER_K / (4 * QR2_S)) / 2;
  constexpr int cNrows = cWarpSize / cThreadsPerRow;
  const int cKqsx = threadIdx.x % cThreadsPerRow;

#pragma unroll
  for (int vI0 = 0; vI0 < cI; vI0 += cNwarps * cNrows) {
    int vI = vI0 + threadIdx.y * cNrows + threadIdx.x / cThreadsPerRow;

    if (fallback) {
      vI = min(vI, pIMax);
    }

    const block_iq2_s *vBxi = (const block_iq2_s *)pX + pKbx0 + vI * pStride;

    const int cQsPacked = get_int_b2(vBxi->qs, cKqsx);
    const uint8_t *vQs = (const uint8_t *)&cQsPacked;

    const int cQh = vBxi->qh[cKqsx];

    const int cSignsPacked32 = get_int_b2(vBxi->qs, QK_K / 32 + cKqsx);
    const uint8_t *vSignsPacked8 = (const uint8_t *)&cSignsPacked32;

#pragma unroll
    for (int vL = 0; vL < QR2_S; ++vL) {
      const int *vGridPos = (const int *)(iq2s_grid + (vQs[vL] | ((cQh << (8 - 2 * vL)) & 0x300)));

      const int cSigns0 = __vcmpne4(((vSignsPacked8[vL] & 0x03) << 7) | ((vSignsPacked8[vL] & 0x0C) << 21), 0x00000000);
      const int cSigns1 = __vcmpne4(((vSignsPacked8[vL] & 0x30) << 3) | ((vSignsPacked8[vL] & 0xC0) << 17), 0x00000000);

      const int cGridL = __vsub4(vGridPos[0] ^ cSigns0, cSigns0);
      const int cGridH = __vsub4(vGridPos[1] ^ cSigns1, cSigns1);

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
      vXQs[vI * cSramStride + 8 * cKqsx + (2 * vL + 0)] = cGridL;
      vXQs[vI * cSramStride + 8 * cKqsx + (2 * vL + 1)] = cGridH;
#else
      vXQs[vI * (2 * MMQ_TILE_NE_K + 1) + 8 * cKqsx + (2 * vL + 0)] = cGridL;
      vXQs[vI * (2 * MMQ_TILE_NE_K + 1) + 8 * cKqsx + (2 * vL + 1)] = cGridH;
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    }

    const int cLs = vBxi->scales[cKqsx];
    const float cD = vBxi->d;
#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    vXDf[vI * cSramStride + 2 * cKqsx + 0] = ((cLs & 0x0F) * cD + cD / 2) / 4;
    vXDf[vI * cSramStride + 2 * cKqsx + 1] = ((cLs >> 4) * cD + cD / 2) / 4;
#else
    vXDf[vI * (2 * MMQ_TILE_NE_K * 2 / QI8_0) + vI / (QI8_0 / 4) + 2 * cKqsx + 0] = ((cLs & 0x0F) * cD + cD / 2) / 4;
    vXDf[vI * (2 * MMQ_TILE_NE_K * 2 / QI8_0) + vI / (QI8_0 / 4) + 2 * cKqsx + 1] = ((cLs >> 4) * cD + cD / 2) / 4;
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
  }
}

template <ggml_type type, int J, bool fallback>
static __device__ __forceinline__ void ggml_cuda_mmq_load_tiles_iq3_xxs(const char *__restrict__ pX,
                                                                        int *__restrict__ pXTile, const int pKbx0,
                                                                        const int pIMax, const int pStride) {
  constexpr int cWarpSize = ggml_cuda_get_physical_warp_size();
  constexpr int cNwarps = ggml_cuda_mmq_get_nthreads(type, J, fallback) / cWarpSize;
  constexpr int cI = ggml_cuda_mmq_get_I(type, J, fallback);
  constexpr int cSramStride = ggml_cuda_mmq_get_sram_stride(type, J, fallback);

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
  int *vXQs = (int *)pXTile;
  float *vXDf = (float *)(vXQs + MMQ_TILE_NE_K * 2);
#else
  constexpr tile_x_sizes cTxs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_IQ3_XXS, cI);
  int *vXQs = (int *)pXTile;
  float *vXDf = (float *)(vXQs + cTxs.qs);
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)

  constexpr int cThreadsPerRow = (MMQ_ITER_K / (4 * QR3_XXS)) / 2;
  constexpr int cNrows = cWarpSize / cThreadsPerRow;
  const int cKqsx = threadIdx.x % cThreadsPerRow;

#pragma unroll
  for (int vI0 = 0; vI0 < cI; vI0 += cNwarps * cNrows) {
    int vI = vI0 + threadIdx.y * cNrows + threadIdx.x / cThreadsPerRow;

    if (fallback) {
      vI = min(vI, pIMax);
    }

    const block_iq3_xxs *vBxi = (const block_iq3_xxs *)pX + pKbx0 + vI * pStride;

    const int2 cQ3Packed = make_int2(get_int_b2(vBxi->qs, 2 * cKqsx + 0), get_int_b2(vBxi->qs, 2 * cKqsx + 1));
    const uint8_t *vQ3 = (const uint8_t *)&cQ3Packed;
    const uint32_t cAux32 = get_int_b2(vBxi->qs, QK_K / 16 + cKqsx);

#pragma unroll
    for (int vL = 0; vL < QR3_XXS; ++vL) {
      const int2 cGridPos = make_int2(iq3xxs_grid[vQ3[2 * vL + 0]], iq3xxs_grid[vQ3[2 * vL + 1]]);
      const uint32_t cSigns = unpack_ksigns(cAux32 >> (7 * vL));

      const int cSigns0 = __vcmpne4(cSigns & 0x08040201, 0);
      const int cGridL = __vsub4(cGridPos.x ^ cSigns0, cSigns0);

      const int cSigns1 = __vcmpne4(cSigns & 0x80402010, 0);
      const int cGridH = __vsub4(cGridPos.y ^ cSigns1, cSigns1);

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
      vXQs[vI * cSramStride + 8 * cKqsx + (2 * vL + 0)] = cGridL;
      vXQs[vI * cSramStride + 8 * cKqsx + (2 * vL + 1)] = cGridH;
#else
      vXQs[vI * (2 * MMQ_TILE_NE_K + 1) + 8 * cKqsx + (2 * vL + 0)] = cGridL;
      vXQs[vI * (2 * MMQ_TILE_NE_K + 1) + 8 * cKqsx + (2 * vL + 1)] = cGridH;
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    }

    const int cLs = cAux32 >> 28;
    const float cD = vBxi->d;
#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    vXDf[vI * cSramStride + cKqsx] = (cLs * cD + cD / 2) / 2;
#else
    vXDf[vI * (MMQ_TILE_NE_K / 4) + vI / 4 + cKqsx] = (cLs * cD + cD / 2) / 2;
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
  }
}

template <ggml_type type, int J, bool fallback>
static __device__ __forceinline__ void ggml_cuda_mmq_load_tiles_iq3_s(const char *__restrict__ pX,
                                                                      int *__restrict__ pXTile, const int pKbx0,
                                                                      const int pIMax, const int pStride) {
  constexpr int cWarpSize = ggml_cuda_get_physical_warp_size();
  constexpr int cNwarps = ggml_cuda_mmq_get_nthreads(type, J, fallback) / cWarpSize;
  constexpr int cI = ggml_cuda_mmq_get_I(type, J, fallback);
  constexpr int cSramStride = ggml_cuda_mmq_get_sram_stride(type, J, fallback);

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
  int *vXQs = (int *)pXTile;
  float *vXDf = (float *)(vXQs + MMQ_TILE_NE_K * 2);
#else
  constexpr tile_x_sizes cTxs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_IQ3_S, cI);
  int *vXQs = (int *)pXTile;
  float *vXDf = (float *)(vXQs + cTxs.qs);
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)

  constexpr int cThreadsPerRow = (MMQ_ITER_K / (4 * QR3_S)) / 2;
  constexpr int cNrows = cWarpSize / cThreadsPerRow;
  const int cKqsx = threadIdx.x % cThreadsPerRow;

#pragma unroll
  for (int vI0 = 0; vI0 < cI; vI0 += cNwarps * cNrows) {
    int vI = vI0 + threadIdx.y * cNrows + threadIdx.x / cThreadsPerRow;

    if (fallback) {
      vI = min(vI, pIMax);
    }

    const block_iq3_s *vBxi = (const block_iq3_s *)pX + pKbx0 + vI * pStride;

    const int2 cQsPacked = make_int2(get_int_b2(vBxi->qs, 2 * cKqsx + 0), get_int_b2(vBxi->qs, 2 * cKqsx + 1));
    const uint8_t *vQs = (const uint8_t *)&cQsPacked;

    const int cQh = vBxi->qh[cKqsx];

    const int cSignsPacked32 = get_int_b2(vBxi->signs, cKqsx);
    const uint8_t *vSignsPacked8 = (const uint8_t *)&cSignsPacked32;

#pragma unroll
    for (int vL = 0; vL < QR3_S; ++vL) {
      const int2 cGridPos = make_int2(iq3s_grid[vQs[2 * vL + 0] | ((cQh << (8 - 2 * vL)) & 0x100)],
                                      iq3s_grid[vQs[2 * vL + 1] | ((cQh << (7 - 2 * vL)) & 0x100)]);

      const int cSigns0 = __vcmpne4(((vSignsPacked8[vL] & 0x03) << 7) | ((vSignsPacked8[vL] & 0x0C) << 21), 0x00000000);
      const int cSigns1 = __vcmpne4(((vSignsPacked8[vL] & 0x30) << 3) | ((vSignsPacked8[vL] & 0xC0) << 17), 0x00000000);

      const int cGridL = __vsub4(cGridPos.x ^ cSigns0, cSigns0);
      const int cGridH = __vsub4(cGridPos.y ^ cSigns1, cSigns1);

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
      vXQs[vI * cSramStride + 8 * cKqsx + (2 * vL + 0)] = cGridL;
      vXQs[vI * cSramStride + 8 * cKqsx + (2 * vL + 1)] = cGridH;
#else
      vXQs[vI * (2 * MMQ_TILE_NE_K + 1) + 8 * cKqsx + (2 * vL + 0)] = cGridL;
      vXQs[vI * (2 * MMQ_TILE_NE_K + 1) + 8 * cKqsx + (2 * vL + 1)] = cGridH;
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    }

    const int cLs = 1 + 2 * ((vBxi->scales[cKqsx / 2] >> (((2 * cKqsx) << 1) & 0x04)) & 0x0F);
    const float cD = vBxi->d;
#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    vXDf[vI * cSramStride + cKqsx] = cLs * cD;
#else
    vXDf[vI * (MMQ_TILE_NE_K / 4) + vI / 4 + cKqsx] = cLs * cD;
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
  }
}

template <ggml_type type, int J, bool fallback>
static __device__ __forceinline__ void ggml_cuda_mmq_load_tiles_iq4_xs(const char *__restrict__ pX,
                                                                       int *__restrict__ pXTile, const int pKbx0,
                                                                       const int pIMax, const int pStride) {
  constexpr int cWarpSize = ggml_cuda_get_physical_warp_size();
  constexpr int cNwarps = ggml_cuda_mmq_get_nthreads(type, J, fallback) / cWarpSize;
  constexpr int cI = ggml_cuda_mmq_get_I(type, J, fallback);
  constexpr int cSramStride = ggml_cuda_mmq_get_sram_stride(type, J, fallback);

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
  int *vXQs = (int *)pXTile;
  float *vXDf = (float *)(vXQs + MMQ_TILE_NE_K * 2);
#else
  constexpr tile_x_sizes cTxs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_IQ4_XS, cI);
  int *vXQs = (int *)pXTile;
  float *vXDf = (float *)(vXQs + cTxs.qs);
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)

  constexpr int cThreadsPerRow = MMQ_ITER_K / (4 * QR4_XS);
  constexpr int cNrows = cWarpSize / cThreadsPerRow;
  const int cKqsx = threadIdx.x % cThreadsPerRow;

#pragma unroll
  for (int vI0 = 0; vI0 < cI; vI0 += cNrows * cNwarps) {
    int vI = vI0 + (cNrows == 1 ? threadIdx.y : threadIdx.y * cNrows + threadIdx.x / cThreadsPerRow);

    if (fallback) {
      vI = min(vI, pIMax);
    }

    const block_iq4_xs *vBxi = (const block_iq4_xs *)pX + pKbx0 + vI * pStride;

    const int cAuxQ4 = get_int_b4(vBxi->qs, cKqsx);
    const int2 cV = get_int_from_table_16(cAuxQ4, kvalues_iq4nl);
    const int cK0 = 8 * (cKqsx / 4) + cKqsx % 4;

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    vXQs[vI * cSramStride + cK0 + 0] = cV.x;
    vXQs[vI * cSramStride + cK0 + 4] = cV.y;
#else
    vXQs[vI * (2 * MMQ_TILE_NE_K + 1) + cK0 + 0] = cV.x;
    vXQs[vI * (2 * MMQ_TILE_NE_K + 1) + cK0 + 4] = cV.y;
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
  }

  constexpr int cRowsPerWarp = cWarpSize / 8;
#pragma unroll
  for (int vI0 = 0; vI0 < cI; vI0 += cNwarps * cRowsPerWarp) {
    int vI = vI0 + threadIdx.y * cRowsPerWarp + threadIdx.x / (MMQ_TILE_NE_K / 4);

    if (fallback) {
      vI = min(vI, pIMax);
    }

    const block_iq4_xs *vBxi = (const block_iq4_xs *)pX + pKbx0 + vI * pStride;

    const float cD = __half2float(vBxi->d);

    const int cLs = ((vBxi->scales_l[(threadIdx.x % 8) / 2] >> (4 * (threadIdx.x % 2))) & 0x0F) |
                    (((vBxi->scales_h >> (2 * (threadIdx.x % 8))) & 0x03) << 4);

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    vXDf[vI * cSramStride + threadIdx.x % 8] = cD * (cLs - 32);
#else
    vXDf[vI * (MMQ_TILE_NE_K / 4) + vI / 4 + threadIdx.x % 8] = cD * (cLs - 32);
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
  }
}

template <ggml_type type, int J, bool fallback>
static __device__ __forceinline__ void ggml_cuda_mmq_load_tiles_iq4_nl(const char *__restrict__ pX,
                                                                       int *__restrict__ pXTile, const int pKbx0,
                                                                       const int pIMax, const int pStride) {
  constexpr int cWarpSize = ggml_cuda_get_physical_warp_size();
  constexpr int cNwarps = ggml_cuda_mmq_get_nthreads(type, J, fallback) / cWarpSize;
  constexpr int cI = ggml_cuda_mmq_get_I(type, J, fallback);
  constexpr int cSramStride = ggml_cuda_mmq_get_sram_stride(type, J, fallback);

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
  int *vXQs = (int *)pXTile;
  float *vXDf = (float *)(vXQs + MMQ_TILE_NE_K * 2);
#else
  constexpr tile_x_sizes cTxs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_IQ4_NL, cI);
  int *vXQs = (int *)pXTile;
  float *vXDf = (float *)(vXQs + cTxs.qs);
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)

  constexpr int cThreadsPerRow = MMQ_ITER_K / (4 * QR4_NL);
  constexpr int cNrows = cWarpSize / cThreadsPerRow;
  const int cTxi = cWarpSize > cThreadsPerRow ? threadIdx.x % cThreadsPerRow : threadIdx.x;
  const int cKbx = cTxi / QI4_NL;
  const int cKqsx = cTxi % QI4_NL;

#pragma unroll
  for (int vI0 = 0; vI0 < cI; vI0 += cNrows * cNwarps) {
    int vI = vI0 + (cNrows == 1 ? threadIdx.y : threadIdx.y * cNrows + threadIdx.x / cThreadsPerRow);

    if (fallback) {
      vI = min(vI, pIMax);
    }

    const block_iq4_nl *vBxi = (const block_iq4_nl *)pX + pKbx0 + vI * pStride + cKbx;

    const int cAuxQ4 = get_int_b2(vBxi->qs, cKqsx);
    const int2 cV = get_int_from_table_16(cAuxQ4, kvalues_iq4nl);
    const int cK0 = cKbx * (2 * QI4_NL) + cKqsx;

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    vXQs[vI * cSramStride + cK0 + 0] = cV.x;
    vXQs[vI * cSramStride + cK0 + QI4_NL] = cV.y;
#else
    vXQs[vI * (2 * MMQ_TILE_NE_K + 1) + cK0 + 0] = cV.x;
    vXQs[vI * (2 * MMQ_TILE_NE_K + 1) + cK0 + QI4_NL] = cV.y;
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
  }

  constexpr int cBlocksPerTileXRow = MMQ_TILE_NE_K / QI4_NL;
  constexpr int cRowsPerWarp = cWarpSize / cBlocksPerTileXRow;
  const int cKbxd = threadIdx.x % cBlocksPerTileXRow;

#pragma unroll
  for (int vI0 = 0; vI0 < cI; vI0 += cNwarps * cRowsPerWarp) {
    int vI = vI0 + threadIdx.y * cRowsPerWarp + threadIdx.x / cBlocksPerTileXRow;

    if (fallback) {
      vI = min(vI, pIMax);
    }

    const block_iq4_nl *vBxi = (const block_iq4_nl *)pX + pKbx0 + vI * pStride + cKbxd;

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    vXDf[vI * cSramStride + cKbxd] = __half2float(vBxi->d);
#else
    vXDf[vI * (MMQ_TILE_NE_K / QI4_NL) + vI / QI4_NL + cKbxd] = __half2float(vBxi->d);
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
  }
}

// ---------------------------------------------------------------------------------------------

template <ggml_type type, int J, bool fallback>
static __device__ __forceinline__ void ggml_cuda_mmq_load_tiles_mxfp4(const char *__restrict__ pX,
                                                                      int *__restrict__ pXTile, const int pKbx0,
                                                                      const int pIMax, const int pStride) {
  constexpr int cWarpSize = ggml_cuda_get_physical_warp_size();
  constexpr int cNwarps = ggml_cuda_mmq_get_nthreads(type, J, fallback) / cWarpSize;
  constexpr int cI = ggml_cuda_mmq_get_I(type, J, fallback);
  constexpr int cSramStride = ggml_cuda_mmq_get_sram_stride(type, J, fallback);

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
  int *vXQs = (int *)pXTile;
  float *vXDf = (float *)(vXQs + MMQ_TILE_NE_K * 2);
#else
  constexpr tile_x_sizes cTxs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_MXFP4, cI);
  int *vXQs = (int *)pXTile;
  float *vXDf = (float *)(vXQs + cTxs.qs);
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)

  constexpr int cThreadsPerRow = MMQ_ITER_K / (4 * QR_MXFP4);
  constexpr int cNrows = cWarpSize / cThreadsPerRow;
  const int cTxi = cWarpSize > cThreadsPerRow ? threadIdx.x % cThreadsPerRow : threadIdx.x;
  const int cKbx = cTxi / QI_MXFP4;
  const int cKqsx = cTxi % QI_MXFP4;

#pragma unroll
  for (int vI0 = 0; vI0 < cI; vI0 += cNrows * cNwarps) {
    int vI = vI0 + (cNrows == 1 ? threadIdx.y : threadIdx.y * cNrows + threadIdx.x / cThreadsPerRow);

    if (fallback) {
      vI = min(vI, pIMax);
    }

    const block_mxfp4 *vBxi = (const block_mxfp4 *)pX + pKbx0 + vI * pStride + cKbx;

    const int cAuxQ4 = get_int_b1(vBxi->qs, cKqsx);
    const int2 cV = get_int_from_table_16(cAuxQ4, kvalues_mxfp4);
    const int cK0 = cKbx * (2 * QI_MXFP4) + cKqsx;

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    vXQs[vI * cSramStride + cK0 + 0] = cV.x;
    vXQs[vI * cSramStride + cK0 + QI_MXFP4] = cV.y;
#else
    vXQs[vI * (2 * MMQ_TILE_NE_K + 1) + cK0 + 0] = cV.x;
    vXQs[vI * (2 * MMQ_TILE_NE_K + 1) + cK0 + QI_MXFP4] = cV.y;
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE)  || defined(AMD_WMMA_AVAILABLE)
  }

  constexpr int cBlocksPerTileXRow = MMQ_TILE_NE_K / QI_MXFP4;
  constexpr int cRowsPerWarp = cWarpSize / cBlocksPerTileXRow;
  const int cKbxd = threadIdx.x % cBlocksPerTileXRow;

#pragma unroll
  for (int vI0 = 0; vI0 < cI; vI0 += cNwarps * cRowsPerWarp) {
    int vI = vI0 + threadIdx.y * cRowsPerWarp + threadIdx.x / cBlocksPerTileXRow;

    if (fallback) {
      vI = min(vI, pIMax);
    }

    const block_mxfp4 *vBxi = (const block_mxfp4 *)pX + pKbx0 + vI * pStride + cKbxd;

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    vXDf[vI * cSramStride + cKbxd] = ggml_cuda_e8m0_to_fp32(vBxi->e) * 0.5f;
#else
    vXDf[vI * (MMQ_TILE_NE_K / QI_MXFP4) + vI / QI_MXFP4 + cKbxd] = ggml_cuda_e8m0_to_fp32(vBxi->e) * 0.5f;
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
  }
}

template <ggml_type type, int J, bool fallback>
static __device__ __forceinline__ void ggml_cuda_mmq_load_tiles_mxfp4_fp4(const char *__restrict__ pX,
                                                                          int *__restrict__ pXTile, const int pKbx0,
                                                                          const int pIMax, const int pStride) {
  constexpr int cWarpSize = ggml_cuda_get_physical_warp_size();
  constexpr int cNwarps = ggml_cuda_mmq_get_nthreads(type, J, fallback) / cWarpSize;
  constexpr int cI = ggml_cuda_mmq_get_I(type, J, fallback);
  constexpr int cSramStride = ggml_cuda_mmq_get_sram_stride(type, J, fallback);

  int *vXQs = (int *)pXTile;
  uint32_t *vXSc = (uint32_t *)(vXQs + 2 * MMQ_TILE_NE_K);

  const int cTxi = threadIdx.x;

  constexpr int cIterK = ggml_cuda_mmq_get_K_vram(type, J, fallback);

  constexpr int cThreadsPerRow = cIterK / QK_MXFP4; // each thread processes 1 block
  constexpr int cRowsPerWarp = cWarpSize / cThreadsPerRow;
  const int cKbx = cTxi % cThreadsPerRow;
  const int cRowInWarp = cTxi / cThreadsPerRow;

#pragma unroll
  for (int vI0 = 0; vI0 < cI; vI0 += cRowsPerWarp * cNwarps) {
    int vI = vI0 + threadIdx.y * cRowsPerWarp + cRowInWarp;

    if constexpr (fallback) {
      vI = min(vI, pIMax);
    }

    const block_mxfp4 *vBxi = (const block_mxfp4 *)pX + pKbx0 + vI * pStride + cKbx;

    // quantize_mxfp4_mmq permutes nibbles to match the quantized format
    const int cK0 = cKbx * 4;
    memcpy(vXQs + vI * cSramStride + cK0, vBxi->qs, 16);

    // Load E8M0 scales: pack 2 consecutive scales into one uint32
    if (cKbx % 2 == 0) {
      uint32_t vE = vBxi->e;
      vE |= ((vBxi + 1)->e << 8);
      vXSc[vI * cSramStride + cKbx / 2] = vE;
    }
  }
}

template <ggml_type type, int J, bool fallback>
static __device__ __forceinline__ void ggml_cuda_mmq_load_tiles_nvfp4(const char *__restrict__ pX,
                                                                      int *__restrict__ pXTile, const int pKb0,
                                                                      const int pIMax, const int pStride) {
  constexpr int cWarpSize = ggml_cuda_get_physical_warp_size();
  constexpr int cNwarps = ggml_cuda_mmq_get_nthreads(type, J, fallback) / cWarpSize;
  constexpr int cI = ggml_cuda_mmq_get_I(type, J, fallback);
  constexpr int cSramStride = ggml_cuda_mmq_get_sram_stride(type, J, fallback);

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
  int *vXQs = (int *)pXTile;
  float *vXDf = (float *)(vXQs + MMQ_TILE_NE_K * 2);
#else
  constexpr tile_x_sizes cTxs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_NVFP4, cI);
  int *vXQs = (int *)pXTile;
  float *vXDf = (float *)(vXQs + cTxs.qs);
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)

  constexpr int cThreadsPerRow = MMQ_ITER_K / QK_NVFP4;
  constexpr int cRowsPerWarp = cWarpSize / cThreadsPerRow;
  const int cKbx = threadIdx.x % cThreadsPerRow;
  const int cRowInWarp = threadIdx.x / cThreadsPerRow;

#pragma unroll
  for (int vI0 = 0; vI0 < cI; vI0 += cRowsPerWarp * cNwarps) {
    int vI = vI0 + threadIdx.y * cRowsPerWarp + cRowInWarp;

    if constexpr (fallback) {
      vI = min(vI, pIMax);
    }

    const block_nvfp4 *vBxi = (const block_nvfp4 *)pX + pKb0 + vI * pStride + cKbx;
    const uint32_t *__restrict__ vSrcQs = reinterpret_cast<const uint32_t *>(vBxi->qs);
    const int cKqs = 16 * cKbx;
    const int cKsc = 4 * cKbx;

#pragma unroll
    for (int vSub = 0; vSub < QK_NVFP4 / QK_NVFP4_SUB; ++vSub) {
      const int2 cQ0 = get_int_from_table_16(vSrcQs[2 * vSub + 0], kvalues_mxfp4);
      const int2 cQ1 = get_int_from_table_16(vSrcQs[2 * vSub + 1], kvalues_mxfp4);

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
      vXQs[vI * cSramStride + cKqs + 4 * vSub + 0] = cQ0.x;
      vXQs[vI * cSramStride + cKqs + 4 * vSub + 1] = cQ1.x;
      vXQs[vI * cSramStride + cKqs + 4 * vSub + 2] = cQ0.y;
      vXQs[vI * cSramStride + cKqs + 4 * vSub + 3] = cQ1.y;
      vXDf[vI * cSramStride + cKsc + vSub] = ggml_cuda_ue4m3_to_fp32(vBxi->d[vSub]);
#else
      vXQs[vI * (2 * MMQ_TILE_NE_K + 1) + cKqs + 4 * vSub + 0] = cQ0.x;
      vXQs[vI * (2 * MMQ_TILE_NE_K + 1) + cKqs + 4 * vSub + 1] = cQ1.x;
      vXQs[vI * (2 * MMQ_TILE_NE_K + 1) + cKqs + 4 * vSub + 2] = cQ0.y;
      vXQs[vI * (2 * MMQ_TILE_NE_K + 1) + cKqs + 4 * vSub + 3] = cQ1.y;
      vXDf[vI * (2 * MMQ_TILE_NE_K * 2 / QI_NVFP4) + vI / (QK_NVFP4_SUB / QI_NVFP4) + cKsc + vSub] =
          ggml_cuda_ue4m3_to_fp32(vBxi->d[vSub]);
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    }
  }
}

template <ggml_type type, int J, bool fallback>
static __device__ __forceinline__ void ggml_cuda_mmq_load_tiles_nvfp4_nvfp4(const char *__restrict__ pX,
                                                                            int *__restrict__ pXTile, const int pKbx0,
                                                                            const int pIMax, const int pStride) {
  constexpr int cWarpSize = ggml_cuda_get_physical_warp_size();
  constexpr int cNwarps = ggml_cuda_mmq_get_nthreads(type, J, fallback) / cWarpSize;
  constexpr int cI = ggml_cuda_mmq_get_I(type, J, fallback);
  constexpr int cIterK = ggml_cuda_mmq_get_K_vram(type, J, fallback);
  constexpr int cThreadsPerRow = cIterK / QK_NVFP4; // each thread processes 1 block
  constexpr int cRowsPerWarp = cWarpSize / cThreadsPerRow;
  constexpr int cSramStride = ggml_cuda_mmq_get_sram_stride(type, J, fallback);

  uint32_t *vXU32 = (uint32_t *)pXTile;

  const int cTxi = threadIdx.x;
  const int cKbx = cTxi % cThreadsPerRow;
  const int cRowInWarp = cTxi / cThreadsPerRow;

  const block_nvfp4 *vBxiBase = (const block_nvfp4 *)pX + pKbx0 + cKbx;
  uint32_t *vXU32Scale = vXU32 + 64 + cKbx;

#pragma unroll
  for (int vI0 = 0; vI0 < cI; vI0 += cRowsPerWarp * cNwarps) {
    int vI = vI0 + threadIdx.y * cRowsPerWarp + cRowInWarp;

    if constexpr (fallback) {
      vI = min(vI, pIMax);
    }

    const block_nvfp4 *vBxi = vBxiBase + vI * pStride;

    const uint32_t *vSrcQs = reinterpret_cast<const uint32_t *>(vBxi->qs);

#pragma unroll
    for (int vSub = 0; vSub < QK_NVFP4 / QK_NVFP4_SUB; ++vSub) {
      vXU32[vI * cSramStride + 8 * cKbx + 2 * vSub + 0] = vSrcQs[2 * vSub + 0];
      vXU32[vI * cSramStride + 8 * cKbx + 2 * vSub + 1] = vSrcQs[2 * vSub + 1];
    }

    vXU32Scale[vI * cSramStride] = get_int_b4(vBxi->d, 0);
  }
}
