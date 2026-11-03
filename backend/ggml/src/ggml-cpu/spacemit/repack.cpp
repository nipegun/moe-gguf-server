#define GGML_COMMON_IMPL_CPP
#define GGML_COMMON_DECL_CPP

#include "repack.h"

#include "ggml-common.h"
#include "ggml-cpu.h"
#include "ggml-impl.h"
#include "ime_kernels.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>

// clang-format off
#if defined(__riscv)

#if !defined(__riscv_v) || !defined(__riscv_v_intrinsic)
#error "riscv v extension or v_intrinsic not enabled"
#else
#include <riscv_vector.h>
#endif

#if !defined(__riscv_zfh)
#error "riscv zfh extension not enabled"
#endif

#else
#error "riscv not enabled in this build"
#endif

#if defined(__GNUC__)
#pragma GCC diagnostic ignored "-Wcast-qual"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#endif

// clang-format on

template <int K> constexpr int fQk0() {
  if constexpr (K == 4) {
    return QK4_0;
  }
  if constexpr (K == 8) {
    return QK8_0;
  }
  return -1;
}

template <int K, int N> struct block {
  ggml_half vD[N];                      // deltas for N qK_0 blocks
  uint8_t aQs[(fQk0<K>() * N * K) / 8]; // quants for N qK_0 blocks
};

template <int K, int N> struct block_with_zp {
  ggml_half vD[N];                      // deltas for N qK_1 blocks
  uint8_t aZp[N];                       // zero points for N qK_1 blocks
  uint8_t aQs[(fQk0<K>() * N * K) / 8]; // quants for N qK_1 blocks
};

// control size
static_assert(sizeof(block<4, 16>) == 16 * sizeof(ggml_half) + QK4_0 * 8, "wrong block<4,16> size/padding");
static_assert(sizeof(block_with_zp<4, 16>) == 16 * sizeof(ggml_half) + QK4_0 * 8 + 16 * sizeof(uint8_t),
              "wrong block_with_zp<4,16> size/padding");

static_assert(sizeof(block<8, 16>) == 16 * sizeof(ggml_half) + QK4_0 * 16, "wrong block<8,16> size/padding");

static_assert(sizeof(block<4, 32>) == 32 * sizeof(ggml_half) + QK4_0 * 16, "wrong block<4,32> size/padding");
static_assert(sizeof(block_with_zp<4, 32>) == 32 * sizeof(ggml_half) + QK4_0 * 16 + 32 * sizeof(uint8_t),
              "wrong block_with_zp<4,32> size/padding");

using block_q4_0x16 = block<4, 16>;
using block_q4_1x16 = block_with_zp<4, 16>;
using block_q8_0x16 = block<8, 16>;

using block_q4_0x32 = block<4, 32>;
using block_q4_1x32 = block_with_zp<4, 32>;
using block_q8_0x32 = block<8, 32>;

struct block_q4_0x32x256 {
  block_q4_0x32 aBlocks[8]; // [f16 * 32 | i4 * 32 * 32] * 8
};

struct block_q4_1x32x256 {
  block_q4_0x32 aBlocks[8];
  uint8_t aZps[32 * 8];
};

static block_q4_0x16 fMakeBlockQ40x16(block_q4_0 *pIn, unsigned int pBlckSizeInterleave) {
  block_q4_0x16 vOut;
  GGML_ASSERT(QK4_0 / pBlckSizeInterleave == 2);

  for (int vI = 0; vI < 16; vI++) {
    vOut.vD[vI] = pIn[vI].d;
  }

  for (int vI = 0; vI < 16; vI++) {
    // [0, 15], in.d & 0x0F
    for (int vJ = 0; vJ < QK4_0 / 4; vJ++) {
      // src [b0 b16] ......... [b8 b24] ......... [b15 b31]
      // dst [b0 b8] ......... [b7 b15]
      vOut.aQs[vI * QK4_0 / 4 + vJ] = (pIn[vI].qs[vJ] & 0x0F) | ((pIn[vI].qs[vJ + QK4_0 / 4] & 0x0F) << 4);
    }
  }

  for (int vI = 0; vI < 16; vI++) {
    // [16, 31], in.d & 0xF0
    for (int vJ = 0; vJ < QK4_0 / 4; vJ++) {
      // src [b0 b16] ......... [b8 b24] ......... [b15 b31]
      // dst [b16 b24] ......... [b23 b31]
      vOut.aQs[4 * QK4_0 + vI * QK4_0 / 4 + vJ] = ((pIn[vI].qs[vJ] & 0xF0) >> 4) | (pIn[vI].qs[vJ + QK4_0 / 4] & 0xF0);
    }
  }

  return vOut;
}

static block_q4_1x16 fMakeBlockQ41x16(block_q4_1 *pIn, unsigned int pBlckSizeInterleave) {
  block_q4_1x16 vOut;
  GGML_ASSERT(QK4_1 / pBlckSizeInterleave == 2);

  for (int vI = 0; vI < 16; vI++) {
    float vD = GGML_FP16_TO_FP32(pIn[vI].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.d);
    float vM = GGML_FP16_TO_FP32(pIn[vI].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.m);
    float vMid = -std::nearbyintf(vM / vD);
    vMid = std::min(15.0f, std::max(0.0f, vMid));
    vOut.vD[vI] = GGML_FP32_TO_FP16(vD);
    vOut.aZp[vI] = static_cast<uint8_t>(vMid);
  }

  for (int vI = 0; vI < 16; vI++) {
    // [0, 15], in.d & 0x0F
    for (int vJ = 0; vJ < QK4_1 / 4; vJ++) {
      // src [b0 b16] ......... [b8 b24] ......... [b15 b31]
      // dst [b0 b8] ......... [b7 b15]
      vOut.aQs[vI * QK4_1 / 4 + vJ] = (pIn[vI].qs[vJ] & 0x0F) | ((pIn[vI].qs[vJ + QK4_1 / 4] & 0x0F) << 4);
    }
  }

  for (int vI = 0; vI < 16; vI++) {
    // [16, 31], in.d & 0xF0
    for (int vJ = 0; vJ < QK4_1 / 4; vJ++) {
      // src [b0 b16] ......... [b8 b24] ......... [b15 b31]
      // dst [b16 b24] ......... [b23 b31]
      vOut.aQs[4 * QK4_1 + vI * QK4_1 / 4 + vJ] = ((pIn[vI].qs[vJ] & 0xF0) >> 4) | (pIn[vI].qs[vJ + QK4_1 / 4] & 0xF0);
    }
  }

  return vOut;
}

static int fRepackQ40ToQ4016Bl(ggml_tensor *pT, int pInterleaveBlock, const void *GGML_RESTRICT pData,
                               size_t pDataSize) {
  GGML_ASSERT(pT->type == GGML_TYPE_Q4_0);
  GGML_ASSERT(pInterleaveBlock == 16);

  constexpr int cNrowsInterleaved = 16;

  block_q4_0x16 *vDst = (block_q4_0x16 *)pT->data;
  const block_q4_0 *vSrc = (const block_q4_0 *)pData;
  block_q4_0 vDstTmp[16];
  int vNrow = ggml_nrows(pT);
  int vNblocks = pT->ne[0] / QK4_0;

  GGML_ASSERT(pDataSize == vNrow * vNblocks * sizeof(block_q4_0));

  if (pT->ne[1] % cNrowsInterleaved != 0 || pT->ne[0] % QK4_0 != 0) {
    return -1;
  }

  for (int vB = 0; vB < vNrow; vB += cNrowsInterleaved) {
    for (int64_t vX = 0; vX < vNblocks; vX++) {
      for (int vI = 0; vI < cNrowsInterleaved; vI++) {
        vDstTmp[vI] = vSrc[vX + vI * vNblocks];
      }
      *vDst++ = fMakeBlockQ40x16(vDstTmp, pInterleaveBlock);
    }
    vSrc += cNrowsInterleaved * vNblocks;
  }
  return 0;

  GGML_UNUSED(pDataSize);
}

static int fRepackQ41ToQ4116Bl(ggml_tensor *pT, int pInterleaveBlock, const void *GGML_RESTRICT pData,
                               size_t pDataSize) {
  GGML_ASSERT(pT->type == GGML_TYPE_Q4_1);
  GGML_ASSERT(pInterleaveBlock == 16);

  constexpr int cNrowsInterleaved = 16;

  block_q4_1x16 *vDst = (block_q4_1x16 *)pT->data;
  const block_q4_1 *vSrc = (const block_q4_1 *)pData;
  block_q4_1 vDstTmp[16];
  int vNrow = ggml_nrows(pT);
  int vNblocks = pT->ne[0] / QK4_1;

  GGML_ASSERT(pDataSize == vNrow * vNblocks * sizeof(block_q4_1));

  if (pT->ne[1] % cNrowsInterleaved != 0 || pT->ne[0] % QK4_1 != 0) {
    return -1;
  }

  for (int vB = 0; vB < vNrow; vB += cNrowsInterleaved) {
    for (int64_t vX = 0; vX < vNblocks; vX++) {
      for (int vI = 0; vI < cNrowsInterleaved; vI++) {
        vDstTmp[vI] = vSrc[vX + vI * vNblocks];
      }
      *vDst++ = fMakeBlockQ41x16(vDstTmp, pInterleaveBlock);
    }
    vSrc += cNrowsInterleaved * vNblocks;
  }
  return 0;

  GGML_UNUSED(pDataSize);
}

static inline void fGetScaleMinK4(int pJ, const uint8_t *GGML_RESTRICT pQ, uint8_t *GGML_RESTRICT pD,
                                  uint8_t *GGML_RESTRICT pM) {
  if (pJ < 4) {
    *pD = pQ[pJ] & 63;
    *pM = pQ[pJ + 4] & 63;
  } else {
    *pD = (pQ[pJ + 4] & 0xF) | ((pQ[pJ - 4] >> 6) << 4);
    *pM = (pQ[pJ + 4] >> 4) | ((pQ[pJ - 0] >> 6) << 4);
  }
}

static int fRepackQ4KToQ4116Bl(ggml_tensor *pT, int pInterleaveBlock, const void *GGML_RESTRICT pData,
                               size_t pDataSize) {
  GGML_ASSERT(pT->type == GGML_TYPE_Q4_K);
  GGML_ASSERT(pInterleaveBlock == 16);
  GGML_ASSERT(QK_K / QK4_1 == 8);

  constexpr int cNrowsInterleaved = 16;

  block_q4_1x16 *vDst = (block_q4_1x16 *)pT->data;
  const block_q4_K *vSrc = (const block_q4_K *)pData;
  block_q4_1 vDstTmp[16];
  int vNrow = ggml_nrows(pT);
  int vNblocks = pT->ne[0] / QK_K;

  if (pT->ne[1] % cNrowsInterleaved != 0 || pT->ne[0] % QK_K != 0) {
    return -1;
  }

  for (int vB = 0; vB < vNrow; vB += cNrowsInterleaved) {
    for (int64_t vX = 0; vX < vNblocks; vX++) {
      for (int vJ = 0; vJ < 8; vJ++) {
        for (int vI = 0; vI < cNrowsInterleaved; vI++) {
          uint8_t vSc, vM;
          const float cD = GGML_FP16_TO_FP32(vSrc[vX + vI * vNblocks].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.d);
          const float cMin = GGML_FP16_TO_FP32(vSrc[vX + vI * vNblocks].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.dmin);
          fGetScaleMinK4(vJ, vSrc[vX + vI * vNblocks].scales, &vSc, &vM);
          const float cD1 = cD * vSc;
          const float cM1 = cMin * vM;

          vDstTmp[vI].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.d = GGML_FP32_TO_FP16(cD1);
          vDstTmp[vI].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.m = GGML_FP32_TO_FP16(-cM1);
          // src -> [b0, b32] [b1, b33] ... [b31, b63]
          // dst -> [b0, b16] [b1, b17] ... [b15, b31] [b32, b48] [b33, b49] ... [b47, b63]
          const uint8_t *vQ = vSrc[vX + vI * vNblocks].qs + (vJ / 2) * QK4_1;
          if (vJ % 2 == 0) {
            for (int vIi = 0; vIi < 16; vIi++) {
              vDstTmp[vI].qs[vIi] = (vQ[vIi] & 0x0F) | ((vQ[vIi + 16] & 0x0F) << 4);
            }
          } else {
            for (int vIi = 0; vIi < 16; vIi++) {
              vDstTmp[vI].qs[vIi] = ((vQ[vIi] & 0xF0) >> 4) | (vQ[vIi + 16] & 0xF0);
            }
          }
        }
        *vDst++ = fMakeBlockQ41x16(vDstTmp, pInterleaveBlock);
      }
    }
    vSrc += cNrowsInterleaved * vNblocks;
  }
  return 0;

  GGML_UNUSED(pDataSize);
}

static block_q4_0x32 fMakeBlockQ40x32(block_q4_0 *pIn, unsigned int pBlckSizeInterleave) {
  block_q4_0x32 vOut;
  assert(QK4_0 / pBlckSizeInterleave == 1);
  GGML_UNUSED(pBlckSizeInterleave);

  for (int vI = 0; vI < 32; vI++) {
    vOut.vD[vI] = pIn[vI].d;
  }

  for (int vI = 0; vI < 32; vI++) {
    // [0, 15], in.d & 0x0F
    for (int vJ = 0; vJ < QK4_0 / 4; vJ++) {
      // src [b0 b16] ......... [b8 b24] ......... [b15 b31]
      // dst [b0 b1] .........  [b14 b15]
      vOut.aQs[vI * QK4_0 / 2 + vJ] = (pIn[vI].qs[vJ * 2] & 0x0F) | ((pIn[vI].qs[vJ * 2 + 1] & 0x0F) << 4);
    }
  }

  for (int vI = 0; vI < 32; vI++) {
    // [16, 31], in.d & 0xF0
    for (int vJ = 0; vJ < QK4_0 / 4; vJ++) {
      // src [b0 b16] ......... [b8 b24] ......... [b15 b31]
      // dst [b16 b17] ......... [b30 b31]
      vOut.aQs[vI * QK4_0 / 2 + QK4_0 / 4 + vJ] = ((pIn[vI].qs[vJ * 2] & 0xF0) >> 4) | (pIn[vI].qs[vJ * 2 + 1] & 0xF0);
    }
  }

  return vOut;
}

static block_q4_1x32 fMakeBlockQ41x32(block_q4_1 *pIn, unsigned int pBlckSizeInterleave) {
  block_q4_1x32 vOut;
  GGML_ASSERT(QK4_1 / pBlckSizeInterleave == 1);
  GGML_UNUSED(pBlckSizeInterleave);

  for (int vI = 0; vI < 32; vI++) {
    float vD = GGML_FP16_TO_FP32(pIn[vI].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.d);
    float vM = GGML_FP16_TO_FP32(pIn[vI].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.m);
    float vMid = -std::nearbyintf(vM / vD);
    vMid = std::min(15.0f, std::max(0.0f, vMid));
    vOut.vD[vI] = GGML_FP32_TO_FP16(vD);
    vOut.aZp[vI] = static_cast<uint8_t>(vMid);
  }

  for (int vI = 0; vI < 32; vI++) {
    // [0, 15], in.d & 0x0F
    for (int vJ = 0; vJ < QK4_1 / 4; vJ++) {
      // src [b0 b16] ......... [b8 b24] ......... [b15 b31]
      // dst [b0 b1] ......... [b14 b15]
      vOut.aQs[vI * QK4_1 / 2 + vJ] = (pIn[vI].qs[vJ * 2] & 0x0F) | ((pIn[vI].qs[vJ * 2 + 1] & 0x0F) << 4);
    }
  }

  for (int vI = 0; vI < 32; vI++) {
    // [16, 31], in.d & 0xF0
    for (int vJ = 0; vJ < QK4_1 / 4; vJ++) {
      // src [b0 b16] ......... [b8 b24] ......... [b15 b31]
      // dst [b16 b24] ......... [b23 b31]
      vOut.aQs[vI * QK4_1 / 2 + QK4_1 / 4 + vJ] = ((pIn[vI].qs[vJ * 2] & 0xF0) >> 4) | (pIn[vI].qs[vJ * 2 + 1] & 0xF0);
    }
  }

  return vOut;
}

static block_q8_0x32 fMakeBlockQ80x32(block_q8_0 *pIn, unsigned int pBlckSizeInterleave) {
  block_q8_0x32 vOut;
  GGML_ASSERT(QK8_0 / pBlckSizeInterleave == 1);
  GGML_UNUSED(pBlckSizeInterleave);

  for (int vI = 0; vI < 32; vI++) {
    vOut.vD[vI] = pIn[vI].d;
  }

  for (int vI = 0; vI < 32; vI++) {
    memcpy(vOut.aQs + vI * QK8_0, pIn[vI].qs, QK8_0);
  }

  return vOut;
}

static int fRepackQ2KToQ2K32Bl(ggml_tensor *pT, int pInterleaveBlock, const void *GGML_RESTRICT pData,
                               size_t pDataSize) {
  GGML_ASSERT(pT->type == GGML_TYPE_Q2_K);
  GGML_ASSERT(pInterleaveBlock == 32);
  GGML_ASSERT(QK_K == 256);

  constexpr int cNrowsInterleaved = 32;

  const block_q2_K *vSrc = (const block_q2_K *)pData;

  auto *vDst = (spacemit_kernels::nrow_block_q2_k<32> *)pT->data;

  int vNrow = ggml_nrows(pT);
  int vNblocks = pT->ne[0] / QK_K;

  GGML_ASSERT(pDataSize == vNrow * vNblocks * sizeof(block_q2_K));

  if (pT->ne[1] % cNrowsInterleaved != 0 || pT->ne[0] % QK_K != 0) {
    return -1;
  }

  uint8_t aQsAux[256] = {0};
  for (int vB = 0; vB < vNrow; vB += cNrowsInterleaved) {
    for (int64_t vX = 0; vX < vNblocks; vX++) {
      for (int vI = 0; vI < cNrowsInterleaved; vI++) {
        const block_q2_K *vSrcBlock = &vSrc[(vB + vI) * vNblocks + vX];

        // scale for [16, N]
        for (int vJ = 0; vJ < 16; vJ++) {
          auto vZpAux = (vDst->scales[vJ * cNrowsInterleaved + vI]) & 0xF0;

          vDst->scales[vJ * cNrowsInterleaved + vI] = (vSrcBlock->scales[vJ] & 0x0F) | vZpAux;
        }

        // zp for [N, 16]
        for (int vJ = 0; vJ < 16; vJ++) {
          auto vScaleAux = (vDst->scales[16 * vI + vJ]) & 0x0F;

          vDst->scales[16 * vI + vJ] = (vSrcBlock->scales[vJ] & 0xF0) | vScaleAux;
        }

        for (int vK = 0; vK < 4; vK++) {
          for (int vJ = 0; vJ < 32; vJ++) {
            aQsAux[vK * 32 + vJ] = (vSrcBlock->qs[vJ] >> (2 * vK)) & 0x03;
          }
        }

        for (int vK = 0; vK < 4; vK++) {
          for (int vJ = 0; vJ < 32; vJ++) {
            aQsAux[vK * 32 + vJ + 128] = (vSrcBlock->qs[vJ + 32] >> (2 * vK)) & 0x03;
          }
        }

        // from nrows_interleaved * [2 * 32byte]
        // to 4 * [nrows_interleaved * 16byte]
        for (int vK = 0; vK < 4; vK++) {
          for (int vJ = 0; vJ < 16; vJ++) {
            uint8_t vQs0 = aQsAux[vJ + vK * 64];
            uint8_t vQs16 = aQsAux[vJ + 16 + vK * 64];
            uint8_t vQs32 = aQsAux[vJ + 32 + vK * 64];
            uint8_t vQs48 = aQsAux[vJ + 48 + vK * 64];

            vDst->qs[(vK * cNrowsInterleaved + vI) * 16 + vJ] =
                (vQs0 & 0x03) | ((vQs16 & 0x03) << 2) | ((vQs32 & 0x03) << 4) | ((vQs48 & 0x03) << 6);
          }
        }

        vDst->scales16[vI] = vSrcBlock->GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.d;
        vDst->zeros16[vI] = vSrcBlock->GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.dmin;
      }
      vDst++;
    }
  }

  return 0;
}

static int fRepackQ3KToQ3K32Bl(ggml_tensor *pT, int pInterleaveBlock, const void *GGML_RESTRICT pData,
                               size_t pDataSize) {
  GGML_ASSERT(pT->type == GGML_TYPE_Q3_K);
  GGML_ASSERT(pInterleaveBlock == 32);
  GGML_ASSERT(QK_K == 256);

  constexpr int cNrowsInterleaved = 32;

  const uint32_t cKmask1 = 0x03030303;
  const uint32_t cKmask2 = 0x0f0f0f0f;

  const block_q3_K *vSrc = (const block_q3_K *)pData;

  auto *vDst = (spacemit_kernels::nrow_block_q3_k<32> *)pT->data;

  int vNrow = ggml_nrows(pT);
  int vNblocks = pT->ne[0] / QK_K;

  GGML_ASSERT(pDataSize == vNrow * vNblocks * sizeof(block_q3_K));

  if (pT->ne[1] % cNrowsInterleaved != 0 || pT->ne[0] % QK_K != 0) {
    return -1;
  }

  uint32_t aBScaleAux[4] = {0};
  uint8_t aQsAux[256] = {0};

  for (int vB = 0; vB < vNrow; vB += cNrowsInterleaved) {
    for (int64_t vX = 0; vX < vNblocks; vX++) {
      for (int vI = 0; vI < cNrowsInterleaved; vI++) {
        const block_q3_K *vSrcBlock = &vSrc[(vB + vI) * vNblocks + vX];

        uint32_t *vAuxs = aBScaleAux;
        int8_t *vScale = (int8_t *)vAuxs;
        memcpy(vAuxs, vSrcBlock->scales, 12);

        uint32_t vTmp = vAuxs[2];
        vAuxs[2] = ((vAuxs[0] >> 4) & cKmask2) | (((vTmp >> 4) & cKmask1) << 4);
        vAuxs[3] = ((vAuxs[1] >> 4) & cKmask2) | (((vTmp >> 6) & cKmask1) << 4);
        vAuxs[0] = (vAuxs[0] & cKmask2) | (((vTmp >> 0) & cKmask1) << 4);
        vAuxs[1] = (vAuxs[1] & cKmask2) | (((vTmp >> 2) & cKmask1) << 4);

        for (int vJ = 0; vJ < 16; vJ++) {
          vDst->scales[vJ * cNrowsInterleaved + vI] = vScale[vJ] - 32;
        }

        for (int vK = 0; vK < 4; vK++) {
          for (int vJ = 0; vJ < 32; vJ++) {
            aQsAux[vK * 32 + vJ] = (vSrcBlock->qs[vJ] >> (2 * vK)) & 0x03;
          }
        }

        for (int vK = 0; vK < 4; vK++) {
          for (int vJ = 0; vJ < 32; vJ++) {
            aQsAux[vK * 32 + vJ + 128] = (vSrcBlock->qs[vJ + 32] >> (2 * vK)) & 0x03;
          }
        }

        // from nrows_interleaved * [2 * 32byte]
        // to 4 * [nrows_interleaved * 16byte]
        for (int vK = 0; vK < 4; vK++) {
          for (int vJ = 0; vJ < 16; vJ++) {
            uint8_t vQs0 = aQsAux[vJ + vK * 64];
            uint8_t vQs16 = aQsAux[vJ + 16 + vK * 64];
            uint8_t vQs32 = aQsAux[vJ + 32 + vK * 64];
            uint8_t vQs48 = aQsAux[vJ + 48 + vK * 64];

            vDst->qs[(vK * cNrowsInterleaved + vI) * 16 + vJ] =
                (vQs0 & 0x03) | ((vQs16 & 0x03) << 2) | ((vQs32 & 0x03) << 4) | ((vQs48 & 0x03) << 6);
          }
        }

        // memcpy(dst->hmask + i * 32, src_block->hmask, 32);

        // from nrows_interleaved * [32byte]
        // to 16 * [nrows_interleaved * uint16_t]
        uint16_t *vDstMask = ((uint16_t *)vDst->hmask) + vI;
        for (int vJ = 0; vJ < 16; vJ++, vDstMask += cNrowsInterleaved) {
          uint8_t vBShift = vJ / 2;
          uint8_t *vBMaskCol = (uint8_t *)(vSrcBlock->hmask + (vJ % 2) * 16);
          // b0 - b15
          uint16_t vMskOut0 = 0;

          for (int vK = 0; vK < 8; vK++) {
            vMskOut0 |= (uint16_t)((vBMaskCol[vK] >> vBShift) & 0x01) << vK;
          }
          for (int vK = 8; vK < 16; vK++) {
            vMskOut0 |= (uint16_t)((vBMaskCol[vK] >> vBShift) & 0x01) << vK;
          }

          vDstMask[0] = vMskOut0;
        }

        vDst->scales16[vI] = vSrcBlock->d;
      }

      vDst++;
    }
  }

  return 0;
}

static int fRepackQ40ToQ4032BlRef(ggml_tensor *pT, int pInterleaveBlock, const void *GGML_RESTRICT pData,
                                  size_t pDataSize) {
  GGML_ASSERT(pT->type == GGML_TYPE_Q4_0);
  GGML_ASSERT(pInterleaveBlock == 32); // unused

  constexpr int cNrowsInterleaved = 32;

  block_q4_0x32 *vDst = (block_q4_0x32 *)pT->data;
  const block_q4_0 *vSrc = (const block_q4_0 *)pData;
  block_q4_0 vDstTmp[32];
  int vNrow = ggml_nrows(pT);
  int vNblocks = pT->ne[0] / QK4_0;

  GGML_ASSERT(pDataSize == vNrow * vNblocks * sizeof(block_q4_0));

  if (pT->ne[1] % cNrowsInterleaved != 0 || pT->ne[0] % QK4_0 != 0) {
    return -1;
  }

  for (int vB = 0; vB < vNrow; vB += cNrowsInterleaved) {
    for (int64_t vX = 0; vX < vNblocks; vX++) {
      for (int vI = 0; vI < cNrowsInterleaved; vI++) {
        vDstTmp[vI] = vSrc[vX + vI * vNblocks];
      }
      *vDst++ = fMakeBlockQ40x32(vDstTmp, pInterleaveBlock);
    }
    vSrc += cNrowsInterleaved * vNblocks;
  }
  return 0;

  GGML_UNUSED(pDataSize);
}

static int fRepackQ40ToQ4025632BlRef(ggml_tensor *pT, int pInterleaveBlock, const void *GGML_RESTRICT pData,
                                     size_t pDataSize) {
  GGML_ASSERT(pT->type == GGML_TYPE_Q4_0);
  GGML_ASSERT(pInterleaveBlock == 32); // unused

  constexpr int cNrowsInterleaved = 32;

  block_q4_0x32x256 *vDst = (block_q4_0x32x256 *)pT->data;
  const block_q4_0 *vSrc = (const block_q4_0 *)pData;
  block_q4_0 vDstTmp[32];
  int vNrow = ggml_nrows(pT);
  int vNblocks = pT->ne[0] / QK4_0;

  GGML_ASSERT(pDataSize == vNrow * vNblocks * sizeof(block_q4_0));
  GGML_ASSERT(vNblocks % 8 == 0); // for 256-block interleaving
  if (pT->ne[1] % cNrowsInterleaved != 0 || pT->ne[0] % QK4_0 != 0) {
    return -1;
  }

  for (int vB = 0; vB < vNrow; vB += cNrowsInterleaved) {
    for (int64_t vX = 0; vX < vNblocks; vX += 8) {
      for (int vJ = 0; vJ < 8; vJ++) {
        for (int vI = 0; vI < cNrowsInterleaved; vI++) {
          vDstTmp[vI] = vSrc[vX + vJ + vI * vNblocks];
        }
        vDst->aBlocks[vJ] = fMakeBlockQ40x32(vDstTmp, pInterleaveBlock);
      }
      vDst++;
    }
    vSrc += cNrowsInterleaved * vNblocks;
  }
  return 0;

  GGML_UNUSED(pDataSize);
}

static int fRepackQ40ToQ4125632BlRef(ggml_tensor *pT, int pInterleaveBlock, const void *GGML_RESTRICT pData,
                                     size_t pDataSize) {
  GGML_ASSERT(pT->type == GGML_TYPE_Q4_1);
  GGML_ASSERT(pInterleaveBlock == 32); // unused

  constexpr int cNrowsInterleaved = 32;

  block_q4_1x32x256 *vDst = (block_q4_1x32x256 *)pT->data;
  const block_q4_1 *vSrc = (const block_q4_1 *)pData;
  block_q4_1 vDstTmp[32];
  int vNrow = ggml_nrows(pT);
  int vNblocks = pT->ne[0] / QK4_0;

  GGML_ASSERT(pDataSize == vNrow * vNblocks * sizeof(block_q4_1));
  GGML_ASSERT(vNblocks % 8 == 0); // for 256-block interleaving
  if (pT->ne[1] % cNrowsInterleaved != 0 || pT->ne[0] % QK4_0 != 0) {
    return -1;
  }

  for (int vB = 0; vB < vNrow; vB += cNrowsInterleaved) {
    for (int64_t vX = 0; vX < vNblocks; vX += 8) {
      for (int vJ = 0; vJ < 8; vJ++) {
        for (int vI = 0; vI < cNrowsInterleaved; vI++) {
          vDstTmp[vI] = vSrc[vX + vJ + vI * vNblocks];
        }

        block_q4_0x32 *vDstBlock = &vDst->aBlocks[vJ];
        uint8_t *vDstZp = vDst->aZps + vJ * cNrowsInterleaved;

        for (int vI = 0; vI < cNrowsInterleaved; vI++) {
          float vD = GGML_FP16_TO_FP32(vDstTmp[vI].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.d);
          float vM = GGML_FP16_TO_FP32(vDstTmp[vI].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.m);
          float vMid = -std::nearbyintf(vM / vD);
          vMid = std::min(15.0f, std::max(0.0f, vMid));

          vDstBlock->vD[vI] = GGML_FP32_TO_FP16(vD);
          vDstZp[vI] = static_cast<uint8_t>(vMid);
        }

        for (int vI = 0; vI < cNrowsInterleaved; vI++) {
          for (int vK = 0; vK < QK4_1 / 4; vK++) {
            vDstBlock->aQs[vI * QK4_1 / 2 + vK] =
                (vDstTmp[vI].qs[vK * 2] & 0x0F) | ((vDstTmp[vI].qs[vK * 2 + 1] & 0x0F) << 4);
          }
        }

        for (int vI = 0; vI < cNrowsInterleaved; vI++) {
          for (int vK = 0; vK < QK4_1 / 4; vK++) {
            vDstBlock->aQs[vI * QK4_1 / 2 + QK4_1 / 4 + vK] =
                ((vDstTmp[vI].qs[vK * 2] & 0xF0) >> 4) | (vDstTmp[vI].qs[vK * 2 + 1] & 0xF0);
          }
        }
      }
      vDst++;
    }
    vSrc += cNrowsInterleaved * vNblocks;
  }
  return 0;

  GGML_UNUSED(pDataSize);
}

// RVV optimized version of repack_q4_0_to_q4_0_32_bl
// Eliminates the intermediate dst_tmp buffer and vectorizes nibble repack.
static int fRepackQ40ToQ4032Bl(ggml_tensor *pT, int pInterleaveBlock, const void *GGML_RESTRICT pData,
                               size_t pDataSize) {
  GGML_ASSERT(pT->type == GGML_TYPE_Q4_0);
  GGML_ASSERT(pInterleaveBlock == 32);

  constexpr int cNrowsInterleaved = 32;
  constexpr int cQsBytes = QK4_0 / 2; // 16

  block_q4_0x32 *vDst = (block_q4_0x32 *)pT->data;
  const block_q4_0 *vSrc = (const block_q4_0 *)pData;
  int vNrow = ggml_nrows(pT);
  int vNblocks = pT->ne[0] / QK4_0;

  GGML_ASSERT(pDataSize == vNrow * vNblocks * sizeof(block_q4_0));

  if (pT->ne[1] % cNrowsInterleaved != 0 || pT->ne[0] % QK4_0 != 0) {
    return -1;
  }

  const ptrdiff_t cRowStride = (ptrdiff_t)vNblocks * sizeof(block_q4_0);

  for (int vB = 0; vB < vNrow; vB += cNrowsInterleaved) {
    for (int64_t vX = 0; vX < vNblocks; vX++) {
      const block_q4_0 *vColSrc = vSrc + vX;

      // --- 1) Gather 32 scale values (ggml_half d) with stride load ---
      // d is at offset 0 of each block_q4_0, stride between rows = row_stride
      {
        const uint8_t *vDBase = (const uint8_t *)&vColSrc->d;
        ggml_half *vDDst = vDst->vD;
        size_t vRemaining = 32;
        size_t vOffset = 0;
        while (vRemaining > 0) {
          size_t vVl = __riscv_vsetvl_e16m1(vRemaining);
          vuint16m1_t vVd = __riscv_vlse16_v_u16m1((const uint16_t *)(vDBase + vOffset * cRowStride), cRowStride, vVl);
          __riscv_vse16_v_u16m1((uint16_t *)(vDDst + vOffset), vVd, vVl);
          vOffset += vVl;
          vRemaining -= vVl;
        }
      }

      // --- 2) Nibble repack qs for each of the 32 rows ---
      // For each row i:
      //   src qs[16]: [b0|b16] [b1|b17] ... [b15|b31]  (lo nibble = b_j, hi nibble = b_{j+16})
      //   dst qs low  8B: (qs[2j] & 0x0F) | ((qs[2j+1] & 0x0F) << 4)  for j=0..7
      //   dst qs high 8B: ((qs[2j] >> 4))  | (qs[2j+1] & 0xF0)         for j=0..7
      {
        const size_t cVl8 = __riscv_vsetvl_e8m1(8);
        for (int vI = 0; vI < 32; vI++) {
          const uint8_t *vSq = vColSrc[vI * vNblocks].qs;
          uint8_t *vDq = vDst->aQs + vI * cQsBytes;

          // stride-2 load to separate even/odd bytes
          vuint8m1_t vVEven = __riscv_vlse8_v_u8m1(vSq, 2, cVl8);    // qs[0], qs[2], ..., qs[14]
          vuint8m1_t vVOdd = __riscv_vlse8_v_u8m1(vSq + 1, 2, cVl8); // qs[1], qs[3], ..., qs[15]

          // low nibble part: (even & 0x0F) | ((odd & 0x0F) << 4)
          vuint8m1_t vVEvenLo = __riscv_vand_vx_u8m1(vVEven, 0x0F, cVl8);
          vuint8m1_t vVOddLo = __riscv_vand_vx_u8m1(vVOdd, 0x0F, cVl8);
          vuint8m1_t vVLo = __riscv_vor_vv_u8m1(vVEvenLo, __riscv_vsll_vx_u8m1(vVOddLo, 4, cVl8), cVl8);

          // high nibble part: (even >> 4) | (odd & 0xF0)
          vuint8m1_t vVEvenHi = __riscv_vsrl_vx_u8m1(vVEven, 4, cVl8);
          vuint8m1_t vVOddHi = __riscv_vand_vx_u8m1(vVOdd, 0xF0, cVl8);
          vuint8m1_t vVHi = __riscv_vor_vv_u8m1(vVEvenHi, vVOddHi, cVl8);

          __riscv_vse8_v_u8m1(vDq, vVLo, cVl8);
          __riscv_vse8_v_u8m1(vDq + 8, vVHi, cVl8);
        }
      }

      vDst++;
    }
    vSrc += cNrowsInterleaved * vNblocks;
  }
  return 0;

  GGML_UNUSED(pDataSize);
}

static int fRepackQ41ToQ4132BlRef(ggml_tensor *pT, int pInterleaveBlock, const void *GGML_RESTRICT pData,
                                  size_t pDataSize) {
  GGML_ASSERT(pT->type == GGML_TYPE_Q4_1);
  GGML_ASSERT(pInterleaveBlock == 32); // unused

  constexpr int cNrowsInterleaved = 32;

  block_q4_1x32 *vDst = (block_q4_1x32 *)pT->data;
  const block_q4_1 *vSrc = (const block_q4_1 *)pData;
  block_q4_1 vDstTmp[32];
  int vNrow = ggml_nrows(pT);
  int vNblocks = pT->ne[0] / QK4_1;

  GGML_ASSERT(pDataSize == vNrow * vNblocks * sizeof(block_q4_1));

  if (pT->ne[1] % cNrowsInterleaved != 0 || pT->ne[0] % QK4_1 != 0) {
    return -1;
  }

  for (int vB = 0; vB < vNrow; vB += cNrowsInterleaved) {
    for (int64_t vX = 0; vX < vNblocks; vX++) {
      for (int vI = 0; vI < cNrowsInterleaved; vI++) {
        vDstTmp[vI] = vSrc[vX + vI * vNblocks];
      }
      *vDst++ = fMakeBlockQ41x32(vDstTmp, pInterleaveBlock);
    }
    vSrc += cNrowsInterleaved * vNblocks;
  }
  return 0;

  GGML_UNUSED(pDataSize);
}

// RVV optimized version of repack_q4_1_to_q4_1_32_bl
// Eliminates the intermediate dst_tmp buffer and vectorizes nibble repack + zp computation.
static int fRepackQ41ToQ4132Bl(ggml_tensor *pT, int pInterleaveBlock, const void *GGML_RESTRICT pData,
                               size_t pDataSize) {
  GGML_ASSERT(pT->type == GGML_TYPE_Q4_1);
  GGML_ASSERT(pInterleaveBlock == 32);

  constexpr int cNrowsInterleaved = 32;
  constexpr int cQsBytes = QK4_1 / 2; // 16

  block_q4_1x32 *vDst = (block_q4_1x32 *)pT->data;
  const block_q4_1 *vSrc = (const block_q4_1 *)pData;
  int vNrow = ggml_nrows(pT);
  int vNblocks = pT->ne[0] / QK4_1;

  GGML_ASSERT(pDataSize == vNrow * vNblocks * sizeof(block_q4_1));

  if (pT->ne[1] % cNrowsInterleaved != 0 || pT->ne[0] % QK4_1 != 0) {
    return -1;
  }

  const ptrdiff_t cRowStride = (ptrdiff_t)vNblocks * sizeof(block_q4_1);

  for (int vB = 0; vB < vNrow; vB += cNrowsInterleaved) {
    for (int64_t vX = 0; vX < vNblocks; vX++) {
      const block_q4_1 *vColSrc = vSrc + vX;

      // --- 1) Gather d and m, compute zp = clamp(nearbyint(-m/d), 0, 15) ---
      // block_q4_1 layout: [d(f16), m(f16), qs[16]]
      // d is at byte offset 0, m is at byte offset 2 from each block start
      {
        const uint8_t *vDmBase = (const uint8_t *)&vColSrc->GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.d;
        ggml_half *vDDst = vDst->vD;
        uint8_t *vZpDst = vDst->aZp;
        size_t vRemaining = 32;
        size_t vOffset = 0;
        while (vRemaining > 0) {
          size_t vVl = __riscv_vsetvl_e16m1(vRemaining);

          // stride load d (f16) from each row
          vuint16m1_t vVdRaw =
              __riscv_vlse16_v_u16m1((const uint16_t *)(vDmBase + vOffset * cRowStride), cRowStride, vVl);
          __riscv_vse16_v_u16m1((uint16_t *)(vDDst + vOffset), vVdRaw, vVl);

          // stride load m (f16) from each row (offset +2 bytes from d)
          vuint16m1_t vVmRaw =
              __riscv_vlse16_v_u16m1((const uint16_t *)(vDmBase + 2 + vOffset * cRowStride), cRowStride, vVl);

          // convert to f32 for zp computation: zp = nearbyint(-m / d)
          vfloat16m1_t vVdF16 = __riscv_vreinterpret_v_u16m1_f16m1(vVdRaw);
          vfloat16m1_t vVmF16 = __riscv_vreinterpret_v_u16m1_f16m1(vVmRaw);

          // -m / d in f16 directly (SpaceMIT X60 supports f16 arithmetic)
          vfloat16m1_t vVNegM = __riscv_vfneg_v_f16m1(vVmF16, vVl);
          vfloat16m1_t vVRatio = __riscv_vfdiv_vv_f16m1(vVNegM, vVdF16, vVl);

          // Convert to f32 for nearbyint, then clamp
          vfloat32m2_t vVRatioF32 = __riscv_vfwcvt_f_f_v_f32m2(vVRatio, vVl);

          // Use integer rounding: convert f32 -> int (rounds to nearest)
          vint32m2_t vVZpI32 = __riscv_vfcvt_x_f_v_i32m2(vVRatioF32, vVl);

          // clamp to [0, 15]
          vVZpI32 = __riscv_vmax_vx_i32m2(vVZpI32, 0, vVl);
          vVZpI32 = __riscv_vmin_vx_i32m2(vVZpI32, 15, vVl);

          // narrow i32 -> u8
          vint16m1_t vVZpI16 = __riscv_vncvt_x_x_w_i16m1(vVZpI32, vVl);
          vint8mf2_t vVZpI8 = __riscv_vncvt_x_x_w_i8mf2(vVZpI16, vVl);
          vuint8mf2_t vVZpU8 = __riscv_vreinterpret_v_i8mf2_u8mf2(vVZpI8);
          __riscv_vse8_v_u8mf2(vZpDst + vOffset, vVZpU8, vVl);

          vOffset += vVl;
          vRemaining -= vVl;
        }
      }

      // --- 2) Nibble repack qs for each of the 32 rows ---
      {
        const size_t cVl8 = __riscv_vsetvl_e8m1(8);
        for (int vI = 0; vI < 32; vI++) {
          const uint8_t *vSq = vColSrc[vI * vNblocks].qs;
          uint8_t *vDq = vDst->aQs + vI * cQsBytes;

          // stride-2 load to separate even/odd bytes
          vuint8m1_t vVEven = __riscv_vlse8_v_u8m1(vSq, 2, cVl8);
          vuint8m1_t vVOdd = __riscv_vlse8_v_u8m1(vSq + 1, 2, cVl8);

          // low nibble part: (even & 0x0F) | ((odd & 0x0F) << 4)
          vuint8m1_t vVEvenLo = __riscv_vand_vx_u8m1(vVEven, 0x0F, cVl8);
          vuint8m1_t vVOddLo = __riscv_vand_vx_u8m1(vVOdd, 0x0F, cVl8);
          vuint8m1_t vVLo = __riscv_vor_vv_u8m1(vVEvenLo, __riscv_vsll_vx_u8m1(vVOddLo, 4, cVl8), cVl8);

          // high nibble part: (even >> 4) | (odd & 0xF0)
          vuint8m1_t vVEvenHi = __riscv_vsrl_vx_u8m1(vVEven, 4, cVl8);
          vuint8m1_t vVOddHi = __riscv_vand_vx_u8m1(vVOdd, 0xF0, cVl8);
          vuint8m1_t vVHi = __riscv_vor_vv_u8m1(vVEvenHi, vVOddHi, cVl8);

          __riscv_vse8_v_u8m1(vDq, vVLo, cVl8);
          __riscv_vse8_v_u8m1(vDq + 8, vVHi, cVl8);
        }
      }

      vDst++;
    }
    vSrc += cNrowsInterleaved * vNblocks;
  }
  return 0;

  GGML_UNUSED(pDataSize);
}

static int fRepackQ4KToQ4132Bl(ggml_tensor *pT, int pInterleaveBlock, const void *GGML_RESTRICT pData,
                               size_t pDataSize) {
  GGML_ASSERT(pT->type == GGML_TYPE_Q4_K);
  GGML_ASSERT(pInterleaveBlock == 32);
  GGML_ASSERT(QK_K / QK4_1 == 8);

  constexpr int cNrowsInterleaved = 32;

  block_q4_1x32 *vDst = (block_q4_1x32 *)pT->data;
  const block_q4_K *vSrc = (const block_q4_K *)pData;
  block_q4_1 vDstTmp[32];
  int vNrow = ggml_nrows(pT);
  int vNblocks = pT->ne[0] / QK_K;

  if (pT->ne[1] % cNrowsInterleaved != 0 || pT->ne[0] % QK_K != 0) {
    return -1;
  }

  for (int vB = 0; vB < vNrow; vB += cNrowsInterleaved) {
    for (int64_t vX = 0; vX < vNblocks; vX++) {
      for (int vJ = 0; vJ < 8; vJ++) {
        for (int vI = 0; vI < cNrowsInterleaved; vI++) {
          uint8_t vSc, vM;
          const float cD = GGML_FP16_TO_FP32(vSrc[vX + vI * vNblocks].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.d);
          const float cMin = GGML_FP16_TO_FP32(vSrc[vX + vI * vNblocks].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.dmin);
          fGetScaleMinK4(vJ, vSrc[vX + vI * vNblocks].scales, &vSc, &vM);
          const float cD1 = cD * vSc;
          const float cM1 = cMin * vM;

          vDstTmp[vI].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.d = GGML_FP32_TO_FP16(cD1);
          vDstTmp[vI].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.m = GGML_FP32_TO_FP16(-cM1);
          // src -> [b0, b32] [b1, b33] ... [b31, b63]
          // dst -> [b0, b16] [b1, b17] ... [b15, b31] [b32, b48] [b33, b49] ... [b47, b63]
          const uint8_t *vQ = vSrc[vX + vI * vNblocks].qs + (vJ / 2) * QK4_1;
          if (vJ % 2 == 0) {
            for (int vIi = 0; vIi < 16; vIi++) {
              vDstTmp[vI].qs[vIi] = (vQ[vIi] & 0x0F) | ((vQ[vIi + 16] & 0x0F) << 4);
            }
          } else {
            for (int vIi = 0; vIi < 16; vIi++) {
              vDstTmp[vI].qs[vIi] = ((vQ[vIi] & 0xF0) >> 4) | (vQ[vIi + 16] & 0xF0);
            }
          }
        }
        *vDst++ = fMakeBlockQ41x32(vDstTmp, pInterleaveBlock);
      }
    }
    vSrc += cNrowsInterleaved * vNblocks;
  }
  return 0;

  GGML_UNUSED(pDataSize);
}

static int fRepackQ6KToQ8032BlRef(ggml_tensor *pT, int pInterleaveBlock, const void *GGML_RESTRICT pData,
                                  size_t pDataSize) {
  GGML_ASSERT(pT->type == GGML_TYPE_Q6_K);
  GGML_ASSERT(pInterleaveBlock == 32);
  GGML_ASSERT(QK_K / QK4_1 == 8);

  constexpr int cNrowsInterleaved = 32;

  block_q8_0x32 *vDst = (block_q8_0x32 *)pT->data;
  const block_q6_K *vSrc = (const block_q6_K *)pData;
  block_q8_0 vDstTmp[32];
  int8_t aAux8[QK4_1];
  int vNrow = ggml_nrows(pT);
  int vNblocks = pT->ne[0] / QK_K;

  if (pT->ne[0] % QK_K != 0) {
    return -1;
  }

  for (int vB = 0; vB < vNrow; vB += cNrowsInterleaved) {
    int64_t vNrowReal = std::min((int64_t)vNrow - vB, (int64_t)cNrowsInterleaved);
    for (int64_t vX = 0; vX < vNblocks; vX++) {
      for (int vBi = 0; vBi < 8; vBi++) {
        int vI = 0;
        for (; vI < vNrowReal; vI++) {
          const uint8_t *vQ4 = vSrc[vX + vI * vNblocks].ql;
          const uint8_t *vQh = vSrc[vX + vI * vNblocks].qh;
          const int8_t *vScales = vSrc[vX + vI * vNblocks].scales;
          float vD = GGML_FP16_TO_FP32(vSrc[vX + vI * vNblocks].d);

          vQ4 += 64 * (vBi / 4);
          vQh += 32 * (vBi / 4);
          int8_t *GGML_RESTRICT vA = aAux8;

          int8_t vBiIdx = vBi % 4;

          if (vBiIdx == 0) {
            for (int vL = 0; vL < 32; ++vL) {
              vA[vL] = (int8_t)((vQ4[vL] & 0xF) | (((vQh[vL] >> 0) & 3) << 4)) - 32;
            }
          } else if (vBiIdx == 1) {
            for (int vL = 0; vL < 32; ++vL) {
              vA[vL] = (int8_t)((vQ4[vL + 32] & 0xF) | (((vQh[vL] >> 2) & 3) << 4)) - 32;
            }
          } else if (vBiIdx == 2) {
            for (int vL = 0; vL < 32; ++vL) {
              vA[vL] = (int8_t)((vQ4[vL + 0] >> 4) | (((vQh[vL] >> 4) & 3) << 4)) - 32;
            }
          } else if (vBiIdx == 3) {
            for (int vL = 0; vL < 32; ++vL) {
              vA[vL] = (int8_t)((vQ4[vL + 32] >> 4) | (((vQh[vL] >> 6) & 3) << 4)) - 32;
            }
          }
          vA = aAux8;

          float vAMaxAbs = 0.0f;
          float vScale0 = vScales[vBi * 2 + 0] * vD;
          float vScale1 = vScales[vBi * 2 + 1] * vD;
          for (int vL = 0; vL < 16; ++vL) {
            vAMaxAbs = std::max(vAMaxAbs, std::abs(vA[vL] * vScale0));
          }

          for (int vL = 16; vL < 32; ++vL) {
            vAMaxAbs = std::max(vAMaxAbs, std::abs(vA[vL] * vScale1));
          }

          float vReflectScale = vAMaxAbs / ((1 << 7) - 1);
          float vReflectScale0 = vScale0 / vReflectScale;
          float vReflectScale1 = vScale1 / vReflectScale;

          for (int vL = 0; vL < 16; ++vL) {
            float vATemp = std::clamp(std::nearbyintf(vA[vL] * vReflectScale0), -128.0f, 127.0f);
            vA[vL] = (int8_t)(vATemp);
          }

          for (int vL = 16; vL < 32; ++vL) {
            float vATemp = std::clamp(std::nearbyintf(vA[vL] * vReflectScale1), -128.0f, 127.0f);
            vA[vL] = (int8_t)(vATemp);
          }

          vDstTmp[vI].d = GGML_FP32_TO_FP16(vReflectScale);

          memcpy(vDstTmp[vI].qs, vA, 32 * sizeof(int8_t));
        }

        for (; vI < cNrowsInterleaved; vI++) {
          memset(&vDstTmp[vI], 0, sizeof(block_q8_0));
        }

        *vDst++ = fMakeBlockQ80x32(vDstTmp, pInterleaveBlock);
      }
    }
    vSrc += cNrowsInterleaved * vNblocks;
  }
  return 0;

  GGML_UNUSED(pDataSize);
}

// RVV optimized version of repack_q6_k_to_q8_0_32_bl
// Vectorizes the Q6_K dequant -> requant pipeline using RVV intrinsics.
// For each sub-block (bi), dequant 32 Q6_K values to int6 -> apply two sub-block scales ->
// find max abs -> compute reflect_scale -> requant to int8 -> gather d with stride load.
static int fRepackQ6KToQ8032Bl(ggml_tensor *pT, int pInterleaveBlock, const void *GGML_RESTRICT pData,
                               size_t pDataSize) {
  GGML_ASSERT(pT->type == GGML_TYPE_Q6_K);
  GGML_ASSERT(pInterleaveBlock == 32);
  GGML_ASSERT(QK_K / QK4_1 == 8);

  constexpr int cNrowsInterleaved = 32;

  block_q8_0x32 *vDst = (block_q8_0x32 *)pT->data;
  const block_q6_K *vSrc = (const block_q6_K *)pData;
  int vNrow = ggml_nrows(pT);
  int vNblocks = pT->ne[0] / QK_K;

  if (pT->ne[1] % cNrowsInterleaved != 0 || pT->ne[0] % QK_K != 0) {
    return -1;
  }

  const ptrdiff_t cRowStride = (ptrdiff_t)vNblocks * sizeof(block_q6_K);

  for (int vB = 0; vB < vNrow; vB += cNrowsInterleaved) {
    for (int64_t vX = 0; vX < vNblocks; vX++) {
      for (int vBi = 0; vBi < 8; vBi++) {
        // --- 1) Gather 32 d values with stride load ---
        // We need to compute reflect_scale per row first, so gather d later.
        // Process each row: dequant Q6_K sub-block -> requant to Q8_0
        for (int vI = 0; vI < cNrowsInterleaved; vI++) {
          const block_q6_K *vSrcBlk = &vSrc[vX + vI * vNblocks];
          const uint8_t *vQ4 = vSrcBlk->ql + 64 * (vBi / 4);
          const uint8_t *vQh = vSrcBlk->qh + 32 * (vBi / 4);
          const int8_t *vScales = vSrcBlk->scales;
          float vD = GGML_FP16_TO_FP32(vSrcBlk->d);

          int8_t vBiIdx = vBi % 4;

          // --- Dequant 32 Q6_K values to int6 (range [-32, 31]) using RVV ---
          // vl = 32 for e8m2 (VLEN=256) or loop for smaller VLEN
          const size_t cVl16 = __riscv_vsetvl_e8m1(16);

          vint8m1_t vVaLo, vVaHi; // 16 elements each

          if (vBiIdx == 0) {
            // a[l] = (q4[l] & 0xF) | (((qh[l] >> 0) & 3) << 4) - 32
            vuint8m1_t vVq4Lo = __riscv_vle8_v_u8m1(vQ4, cVl16);
            vuint8m1_t vVq4Hi = __riscv_vle8_v_u8m1(vQ4 + 16, cVl16);
            vuint8m1_t vVqhLo = __riscv_vle8_v_u8m1(vQh, cVl16);
            vuint8m1_t vVqhHi = __riscv_vle8_v_u8m1(vQh + 16, cVl16);

            vuint8m1_t vVlo4Lo = __riscv_vand_vx_u8m1(vVq4Lo, 0x0F, cVl16);
            vuint8m1_t vVlo4Hi = __riscv_vand_vx_u8m1(vVq4Hi, 0x0F, cVl16);
            vuint8m1_t vVhLo = __riscv_vsll_vx_u8m1(__riscv_vand_vx_u8m1(vVqhLo, 0x03, cVl16), 4, cVl16);
            vuint8m1_t vVhHi = __riscv_vsll_vx_u8m1(__riscv_vand_vx_u8m1(vVqhHi, 0x03, cVl16), 4, cVl16);

            vuint8m1_t vVcombLo = __riscv_vor_vv_u8m1(vVlo4Lo, vVhLo, cVl16);
            vuint8m1_t vVcombHi = __riscv_vor_vv_u8m1(vVlo4Hi, vVhHi, cVl16);

            vVaLo = __riscv_vsub_vx_i8m1(__riscv_vreinterpret_v_u8m1_i8m1(vVcombLo), 32, cVl16);
            vVaHi = __riscv_vsub_vx_i8m1(__riscv_vreinterpret_v_u8m1_i8m1(vVcombHi), 32, cVl16);
          } else if (vBiIdx == 1) {
            // a[l] = (q4[l+32] & 0xF) | (((qh[l] >> 2) & 3) << 4) - 32
            vuint8m1_t vVq4Lo = __riscv_vle8_v_u8m1(vQ4 + 32, cVl16);
            vuint8m1_t vVq4Hi = __riscv_vle8_v_u8m1(vQ4 + 48, cVl16);
            vuint8m1_t vVqhLo = __riscv_vle8_v_u8m1(vQh, cVl16);
            vuint8m1_t vVqhHi = __riscv_vle8_v_u8m1(vQh + 16, cVl16);

            vuint8m1_t vVlo4Lo = __riscv_vand_vx_u8m1(vVq4Lo, 0x0F, cVl16);
            vuint8m1_t vVlo4Hi = __riscv_vand_vx_u8m1(vVq4Hi, 0x0F, cVl16);
            vuint8m1_t vVhLo = __riscv_vsll_vx_u8m1(
                __riscv_vand_vx_u8m1(__riscv_vsrl_vx_u8m1(vVqhLo, 2, cVl16), 0x03, cVl16), 4, cVl16);
            vuint8m1_t vVhHi = __riscv_vsll_vx_u8m1(
                __riscv_vand_vx_u8m1(__riscv_vsrl_vx_u8m1(vVqhHi, 2, cVl16), 0x03, cVl16), 4, cVl16);

            vuint8m1_t vVcombLo = __riscv_vor_vv_u8m1(vVlo4Lo, vVhLo, cVl16);
            vuint8m1_t vVcombHi = __riscv_vor_vv_u8m1(vVlo4Hi, vVhHi, cVl16);

            vVaLo = __riscv_vsub_vx_i8m1(__riscv_vreinterpret_v_u8m1_i8m1(vVcombLo), 32, cVl16);
            vVaHi = __riscv_vsub_vx_i8m1(__riscv_vreinterpret_v_u8m1_i8m1(vVcombHi), 32, cVl16);
          } else if (vBiIdx == 2) {
            // a[l] = (q4[l] >> 4) | (((qh[l] >> 4) & 3) << 4) - 32
            vuint8m1_t vVq4Lo = __riscv_vle8_v_u8m1(vQ4, cVl16);
            vuint8m1_t vVq4Hi = __riscv_vle8_v_u8m1(vQ4 + 16, cVl16);
            vuint8m1_t vVqhLo = __riscv_vle8_v_u8m1(vQh, cVl16);
            vuint8m1_t vVqhHi = __riscv_vle8_v_u8m1(vQh + 16, cVl16);

            vuint8m1_t vVhi4Lo = __riscv_vsrl_vx_u8m1(vVq4Lo, 4, cVl16);
            vuint8m1_t vVhi4Hi = __riscv_vsrl_vx_u8m1(vVq4Hi, 4, cVl16);
            vuint8m1_t vVhLo = __riscv_vsll_vx_u8m1(
                __riscv_vand_vx_u8m1(__riscv_vsrl_vx_u8m1(vVqhLo, 4, cVl16), 0x03, cVl16), 4, cVl16);
            vuint8m1_t vVhHi = __riscv_vsll_vx_u8m1(
                __riscv_vand_vx_u8m1(__riscv_vsrl_vx_u8m1(vVqhHi, 4, cVl16), 0x03, cVl16), 4, cVl16);

            vuint8m1_t vVcombLo = __riscv_vor_vv_u8m1(vVhi4Lo, vVhLo, cVl16);
            vuint8m1_t vVcombHi = __riscv_vor_vv_u8m1(vVhi4Hi, vVhHi, cVl16);

            vVaLo = __riscv_vsub_vx_i8m1(__riscv_vreinterpret_v_u8m1_i8m1(vVcombLo), 32, cVl16);
            vVaHi = __riscv_vsub_vx_i8m1(__riscv_vreinterpret_v_u8m1_i8m1(vVcombHi), 32, cVl16);
          } else { // bi_idx == 3
            // a[l] = (q4[l+32] >> 4) | (((qh[l] >> 6) & 3) << 4) - 32
            vuint8m1_t vVq4Lo = __riscv_vle8_v_u8m1(vQ4 + 32, cVl16);
            vuint8m1_t vVq4Hi = __riscv_vle8_v_u8m1(vQ4 + 48, cVl16);
            vuint8m1_t vVqhLo = __riscv_vle8_v_u8m1(vQh, cVl16);
            vuint8m1_t vVqhHi = __riscv_vle8_v_u8m1(vQh + 16, cVl16);

            vuint8m1_t vVhi4Lo = __riscv_vsrl_vx_u8m1(vVq4Lo, 4, cVl16);
            vuint8m1_t vVhi4Hi = __riscv_vsrl_vx_u8m1(vVq4Hi, 4, cVl16);
            vuint8m1_t vVhLo = __riscv_vsll_vx_u8m1(
                __riscv_vand_vx_u8m1(__riscv_vsrl_vx_u8m1(vVqhLo, 6, cVl16), 0x03, cVl16), 4, cVl16);
            vuint8m1_t vVhHi = __riscv_vsll_vx_u8m1(
                __riscv_vand_vx_u8m1(__riscv_vsrl_vx_u8m1(vVqhHi, 6, cVl16), 0x03, cVl16), 4, cVl16);

            vuint8m1_t vVcombLo = __riscv_vor_vv_u8m1(vVhi4Lo, vVhLo, cVl16);
            vuint8m1_t vVcombHi = __riscv_vor_vv_u8m1(vVhi4Hi, vVhHi, cVl16);

            vVaLo = __riscv_vsub_vx_i8m1(__riscv_vreinterpret_v_u8m1_i8m1(vVcombLo), 32, cVl16);
            vVaHi = __riscv_vsub_vx_i8m1(__riscv_vreinterpret_v_u8m1_i8m1(vVcombHi), 32, cVl16);
          }

          // --- Widen to i16 for scaled abs computation ---
          float vScale0 = vScales[vBi * 2 + 0] * vD;
          float vScale1 = vScales[vBi * 2 + 1] * vD;

          // Widen i8 -> i16 -> f32 for abs*scale computation
          vint16m2_t vVaLoW = __riscv_vsext_vf2_i16m2(vVaLo, cVl16);
          vint16m2_t vVaHiW = __riscv_vsext_vf2_i16m2(vVaHi, cVl16);

          // Compute |a[l] * scale_0| for lo half, |a[l] * scale_1| for hi half
          vfloat32m4_t vVfLo = __riscv_vfcvt_f_x_v_f32m4(__riscv_vsext_vf2_i32m4(vVaLoW, cVl16), cVl16);
          vfloat32m4_t vVfHi = __riscv_vfcvt_f_x_v_f32m4(__riscv_vsext_vf2_i32m4(vVaHiW, cVl16), cVl16);

          vfloat32m4_t vVabsLo = __riscv_vfabs_v_f32m4(__riscv_vfmul_vf_f32m4(vVfLo, vScale0, cVl16), cVl16);
          vfloat32m4_t vVabsHi = __riscv_vfabs_v_f32m4(__riscv_vfmul_vf_f32m4(vVfHi, vScale1, cVl16), cVl16);

          // Find max abs across both halves
          vfloat32m4_t vVabsMax = __riscv_vfmax_vv_f32m4(vVabsLo, vVabsHi, cVl16);

          // Reduce to scalar max
          vfloat32m1_t vVzero = __riscv_vfmv_v_f_f32m1(0.0f, 1);
          vfloat32m1_t vVmaxRed = __riscv_vfredmax_vs_f32m4_f32m1(vVabsMax, vVzero, cVl16);
          float vAMaxAbs = __riscv_vfmv_f_s_f32m1_f32(vVmaxRed);

          float vReflectScale = vAMaxAbs / 127.0f;
          float vReflectScale0 = vScale0 / vReflectScale;
          float vReflectScale1 = vScale1 / vReflectScale;

          // --- Requant: a[l] = clamp(nearbyint(a[l] * reflect_scale_x), -128, 127) ---
          vfloat32m4_t vVscaledLo = __riscv_vfmul_vf_f32m4(vVfLo, vReflectScale0, cVl16);
          vfloat32m4_t vVscaledHi = __riscv_vfmul_vf_f32m4(vVfHi, vReflectScale1, cVl16);

          // fcvt.x rounds to nearest (using current rounding mode)
          vint32m4_t vViLo = __riscv_vfcvt_x_f_v_i32m4(vVscaledLo, cVl16);
          vint32m4_t vViHi = __riscv_vfcvt_x_f_v_i32m4(vVscaledHi, cVl16);

          // Clamp to [-128, 127]
          vViLo = __riscv_vmax_vx_i32m4(vViLo, -128, cVl16);
          vViLo = __riscv_vmin_vx_i32m4(vViLo, 127, cVl16);
          vViHi = __riscv_vmax_vx_i32m4(vViHi, -128, cVl16);
          vViHi = __riscv_vmin_vx_i32m4(vViHi, 127, cVl16);

          // Narrow i32 -> i16 -> i8
          vint16m2_t vVi16Lo = __riscv_vncvt_x_x_w_i16m2(vViLo, cVl16);
          vint16m2_t vVi16Hi = __riscv_vncvt_x_x_w_i16m2(vViHi, cVl16);
          vint8m1_t vVi8Lo = __riscv_vncvt_x_x_w_i8m1(vVi16Lo, cVl16);
          vint8m1_t vVi8Hi = __riscv_vncvt_x_x_w_i8m1(vVi16Hi, cVl16);

          // Store d and qs directly into dst block
          vDst->vD[vI] = GGML_FP32_TO_FP16(vReflectScale);
          int8_t *vDq = (int8_t *)vDst->aQs + vI * QK8_0;
          __riscv_vse8_v_i8m1(vDq, vVi8Lo, cVl16);
          __riscv_vse8_v_i8m1(vDq + 16, vVi8Hi, cVl16);
        }
        vDst++;
      }
    }
    vSrc += cNrowsInterleaved * vNblocks;
  }
  return 0;

  GGML_UNUSED(pDataSize);
}

static int fRepackQ80ToQ8032BlRef(ggml_tensor *pT, int pInterleaveBlock, const void *GGML_RESTRICT pData,
                                  size_t pDataSize) {
  GGML_ASSERT(pT->type == GGML_TYPE_Q8_0);
  GGML_ASSERT(pInterleaveBlock == 32); // unused

  constexpr int cNrowsInterleaved = 32;

  block_q8_0x32 *vDst = (block_q8_0x32 *)pT->data;
  const block_q8_0 *vSrc = (const block_q8_0 *)pData;
  block_q8_0 vDstTmp[32];
  int vNrow = ggml_nrows(pT);
  int vNblocks = pT->ne[0] / QK8_0;

  GGML_ASSERT(pDataSize == vNrow * vNblocks * sizeof(block_q8_0));

  if (pT->ne[0] % QK8_0 != 0) {
    return -1;
  }

  for (int vB = 0; vB < vNrow; vB += cNrowsInterleaved) {
    int64_t vNrowsReal = std::min((int64_t)vNrow - vB, (int64_t)cNrowsInterleaved);
    for (int64_t vX = 0; vX < vNblocks; vX++) {
      int vI = 0;
      for (; vI < vNrowsReal; vI++) {
        vDstTmp[vI] = vSrc[vX + vI * vNblocks];
      }
      for (; vI < cNrowsInterleaved; vI++) {
        memset(&vDstTmp[vI], 0, sizeof(block_q8_0));
      }
      *vDst++ = fMakeBlockQ80x32(vDstTmp, pInterleaveBlock);
    }
    vSrc += cNrowsInterleaved * vNblocks;
  }
  return 0;

  GGML_UNUSED(pDataSize);
}

// RVV optimized version of repack_q8_0_to_q8_0_32_bl
// Eliminates the intermediate dst_tmp buffer and vectorizes scale gather + qs copy.
static int fRepackQ80ToQ8032Bl(ggml_tensor *pT, int pInterleaveBlock, const void *GGML_RESTRICT pData,
                               size_t pDataSize) {
  GGML_ASSERT(pT->type == GGML_TYPE_Q8_0);
  GGML_ASSERT(pInterleaveBlock == 32);

  constexpr int cNrowsInterleaved = 32;

  block_q8_0x32 *vDst = (block_q8_0x32 *)pT->data;
  const block_q8_0 *vSrc = (const block_q8_0 *)pData;
  int vNrow = ggml_nrows(pT);
  int vNblocks = pT->ne[0] / QK8_0;

  GGML_ASSERT(pDataSize == vNrow * vNblocks * sizeof(block_q8_0));

  if (pT->ne[1] % cNrowsInterleaved != 0 || pT->ne[0] % QK8_0 != 0) {
    return -1;
  }

  const ptrdiff_t cRowStride = (ptrdiff_t)vNblocks * sizeof(block_q8_0);

  for (int vB = 0; vB < vNrow; vB += cNrowsInterleaved) {
    for (int64_t vX = 0; vX < vNblocks; vX++) {
      const block_q8_0 *vColSrc = vSrc + vX;

      // --- 1) Gather 32 scale values (ggml_half d) with stride load ---
      {
        const uint8_t *vDBase = (const uint8_t *)&vColSrc->d;
        ggml_half *vDDst = vDst->vD;
        size_t vRemaining = 32;
        size_t vOffset = 0;
        while (vRemaining > 0) {
          size_t vVl = __riscv_vsetvl_e16m1(vRemaining);
          vuint16m1_t vVd = __riscv_vlse16_v_u16m1((const uint16_t *)(vDBase + vOffset * cRowStride), cRowStride, vVl);
          __riscv_vse16_v_u16m1((uint16_t *)(vDDst + vOffset), vVd, vVl);
          vOffset += vVl;
          vRemaining -= vVl;
        }
      }

      // --- 2) Copy qs for each of the 32 rows (32 bytes per row) ---
      {
        for (int vI = 0; vI < 32; vI++) {
          const int8_t *vSq = vColSrc[vI * vNblocks].qs;
          int8_t *vDq = (int8_t *)vDst->aQs + vI * QK8_0;

          size_t vLen = QK8_0;
          size_t vIdx = 0;
          while (vLen > 0) {
            size_t vVl = __riscv_vsetvl_e8m2(vLen);
            vint8m2_t vVs = __riscv_vle8_v_i8m2(vSq + vIdx, vVl);
            __riscv_vse8_v_i8m2(vDq + vIdx, vVs, vVl);
            vIdx += vVl;
            vLen -= vVl;
          }
        }
      }

      vDst++;
    }
    vSrc += cNrowsInterleaved * vNblocks;
  }
  return 0;

  GGML_UNUSED(pDataSize);
}

static void fConvertMxfp4To5bit(const block_mxfp4 &pSrc, spacemit_kernels::nrow_block_mxfp4<1> &pDst) {
  pDst.e[0] = pSrc.e;

  // Decode all 32 mxfp4 values to signed integers via kvalues_mxfp4
  int8_t aVals[32];
  for (int vJ = 0; vJ < QK_MXFP4 / 2; vJ++) {
    aVals[vJ] = kvalues_mxfp4[pSrc.qs[vJ] & 0xF];
    aVals[vJ + QK_MXFP4 / 2] = kvalues_mxfp4[pSrc.qs[vJ] >> 4];
  }

  // vals [b0, b1, b2, b3, ..., b30, b31]
  // Pack abs into qs with reorder: [b0,b1]..[b14,b15]..[b30,b31]
  for (int vJ = 0; vJ < QK_MXFP4 / 2; vJ++) {
    uint8_t vLo0 = static_cast<uint8_t>(std::abs(aVals[vJ * 2]));
    uint8_t vLo1 = static_cast<uint8_t>(std::abs(aVals[vJ * 2 + 1]));
    pDst.qs[vJ] = (vLo0 & 0x0F) | ((vLo1 & 0x0F) << 4);
  }

  // Pack sign bits into qh[4] (32 bits total, 1 bit per weight)
  // reorder: [0,1,2,...,15,16,17,...,31] after the qs reorder above
  uint32_t vSignBits = 0;
  for (int vJ = 0; vJ < 32; vJ++) {
    if (aVals[vJ] < 0) {
      vSignBits |= (1u << vJ);
    }
  }
  memcpy(pDst.qh, &vSignBits, 4);
}

static spacemit_kernels::nrow_block_mxfp4<32> fMakeBlockMxfp4x32(spacemit_kernels::nrow_block_mxfp4<1> *pIn,
                                                                 unsigned int pBlckSizeInterleave) {
  spacemit_kernels::nrow_block_mxfp4<32> vOut;
  GGML_ASSERT(QK_MXFP4 / pBlckSizeInterleave == 1);
  GGML_UNUSED(pBlckSizeInterleave);

  for (int vI = 0; vI < 32; vI++) {
    vOut.e[vI] = pIn[vI].e[0];
  }

  // qs: copy per-row 16 bytes
  for (int vI = 0; vI < 32; vI++) {
    memcpy(vOut.qs + vI * 16, pIn[vI].qs, 16);
  }

  // qh: copy per-row 4 bytes
  for (int vI = 0; vI < 32; vI++) {
    memcpy(vOut.qh + vI * 4, pIn[vI].qh, 4);
  }

  return vOut;
}

static int fRepackMxfp4ToMxfp432Bl(ggml_tensor *pT, int pInterleaveBlock, const void *GGML_RESTRICT pData,
                                   size_t pDataSize) {
  GGML_ASSERT(pT->type == GGML_TYPE_MXFP4);
  GGML_ASSERT(pInterleaveBlock == 32);

  constexpr int cNrowsInterleaved = 32;

  spacemit_kernels::nrow_block_mxfp4<32> *vDst = (spacemit_kernels::nrow_block_mxfp4<32> *)pT->data;
  const block_mxfp4 *vSrc = (const block_mxfp4 *)pData;
  spacemit_kernels::nrow_block_mxfp4<1> vDstTmp[32];
  int vNrow = ggml_nrows(pT);
  int vNblocks = pT->ne[0] / QK_MXFP4;

  GGML_ASSERT(pDataSize == vNrow * vNblocks * sizeof(block_mxfp4));

  if (pT->ne[1] % cNrowsInterleaved != 0 || pT->ne[0] % QK_MXFP4 != 0) {
    return -1;
  }

  for (int vB = 0; vB < vNrow; vB += cNrowsInterleaved) {
    for (int64_t vX = 0; vX < vNblocks; vX++) {
      for (int vI = 0; vI < cNrowsInterleaved; vI++) {
        fConvertMxfp4To5bit(vSrc[vX + vI * vNblocks], vDstTmp[vI]);
      }
      *vDst++ = fMakeBlockMxfp4x32(vDstTmp, pInterleaveBlock);
    }
    vSrc += cNrowsInterleaved * vNblocks;
  }
  return 0;
}

static spacemit_kernels::nrow_block_q5_1<32> fMakeBlockQ51x32(spacemit_kernels::nrow_block_q5_1<1> *pIn,
                                                              unsigned int pBlckSizeInterleave) {
  spacemit_kernels::nrow_block_q5_1<32> vOut;
  GGML_ASSERT(QK5_1 / pBlckSizeInterleave == 1);
  GGML_UNUSED(pBlckSizeInterleave);

  for (int vI = 0; vI < 32; vI++) {
    vOut.scales16[vI] = pIn[vI].scales16[0];
    vOut.zp[vI] = pIn[vI].zp[0];
  }

  // qs: low 4 bits, reorder from [b0,b16],[b1,b17]... to [b0,b1]...[b14,b15] and [b16,b17]...[b30,b31]
  for (int vI = 0; vI < 32; vI++) {
    // low half [0..15]
    for (int vJ = 0; vJ < QK5_1 / 4; vJ++) {
      vOut.qs[vI * QK5_1 / 2 + vJ] = (pIn[vI].qs[vJ * 2] & 0x0F) | ((pIn[vI].qs[vJ * 2 + 1] & 0x0F) << 4);
    }
    // high half [16..31]
    for (int vJ = 0; vJ < QK5_1 / 4; vJ++) {
      vOut.qs[vI * QK5_1 / 2 + QK5_1 / 4 + vJ] = ((pIn[vI].qs[vJ * 2] & 0xF0) >> 4) | (pIn[vI].qs[vJ * 2 + 1] & 0xF0);
    }
  }

  // qh: 5th bit, copy directly
  for (int vI = 0; vI < 32; vI++) {
    for (int vJ = 0; vJ < 4; vJ++) {
      vOut.qh[vI * 4 + vJ] = pIn[vI].qh[vJ];
    }
  }

  return vOut;
}

static spacemit_kernels::nrow_block_q5_0<32> fMakeBlockQ50x32(spacemit_kernels::nrow_block_q5_0<1> *pIn,
                                                              unsigned int pBlckSizeInterleave) {
  spacemit_kernels::nrow_block_q5_0<32> vOut;
  GGML_ASSERT(QK5_0 / pBlckSizeInterleave == 1);
  GGML_UNUSED(pBlckSizeInterleave);

  for (int vI = 0; vI < 32; vI++) {
    vOut.scales16[vI] = pIn[vI].scales16[0];
  }

  // qs: low 4 bits, reorder from [b0,b16],[b1,b17]... to [b0,b1]...[b14,b15] and [b16,b17]...[b30,b31]
  for (int vI = 0; vI < 32; vI++) {
    // low half [0..15]
    for (int vJ = 0; vJ < QK5_0 / 4; vJ++) {
      vOut.qs[vI * QK5_0 / 2 + vJ] = (pIn[vI].qs[vJ * 2] & 0x0F) | ((pIn[vI].qs[vJ * 2 + 1] & 0x0F) << 4);
    }
    // high half [16..31]
    for (int vJ = 0; vJ < QK5_0 / 4; vJ++) {
      vOut.qs[vI * QK5_0 / 2 + QK5_0 / 4 + vJ] = ((pIn[vI].qs[vJ * 2] & 0xF0) >> 4) | (pIn[vI].qs[vJ * 2 + 1] & 0xF0);
    }
  }

  // qh: 5th bit, copy directly
  for (int vI = 0; vI < 32; vI++) {
    for (int vJ = 0; vJ < 4; vJ++) {
      vOut.qh[vI * 4 + vJ] = pIn[vI].qh[vJ];
    }
  }

  return vOut;
}

static int fRepackQ50ToQ5032Bl(ggml_tensor *pT, int pInterleaveBlock, const void *GGML_RESTRICT pData,
                               size_t pDataSize) {
  GGML_ASSERT(pT->type == GGML_TYPE_Q5_0);
  GGML_ASSERT(pInterleaveBlock == 32); // unused

  constexpr int cNrowsInterleaved = 32;

  spacemit_kernels::nrow_block_q5_0<32> *vDst = (spacemit_kernels::nrow_block_q5_0<32> *)pT->data;
  const block_q5_0 *vSrc = (const block_q5_0 *)pData;
  spacemit_kernels::nrow_block_q5_0<1> vDstTmp[32];
  int vNrow = ggml_nrows(pT);
  int vNblocks = pT->ne[0] / QK5_0;

  GGML_ASSERT(pDataSize == vNrow * vNblocks * sizeof(block_q5_0));

  if (pT->ne[1] % cNrowsInterleaved != 0 || pT->ne[0] % QK5_0 != 0) {
    return -1;
  }

  for (int vB = 0; vB < vNrow; vB += cNrowsInterleaved) {
    for (int64_t vX = 0; vX < vNblocks; vX++) {
      for (int vI = 0; vI < cNrowsInterleaved; vI++) {
        const block_q5_0 &vS = vSrc[vX + vI * vNblocks];

        vDstTmp[vI].scales16[0] = vS.d;
        memcpy(vDstTmp[vI].qs, vS.qs, sizeof(vDstTmp[vI].qs));
        memcpy(vDstTmp[vI].qh, vS.qh, sizeof(vDstTmp[vI].qh));
      }
      *vDst++ = fMakeBlockQ50x32(vDstTmp, pInterleaveBlock);
    }
    vSrc += cNrowsInterleaved * vNblocks;
  }
  return 0;
}

static int fRepackQ51ToQ5132Bl(ggml_tensor *pT, int pInterleaveBlock, const void *GGML_RESTRICT pData,
                               size_t pDataSize) {
  GGML_ASSERT(pT->type == GGML_TYPE_Q5_1);
  GGML_ASSERT(pInterleaveBlock == 32); // unused

  constexpr int cNrowsInterleaved = 32;

  spacemit_kernels::nrow_block_q5_1<32> *vDst = (spacemit_kernels::nrow_block_q5_1<32> *)pT->data;
  const block_q5_1 *vSrc = (const block_q5_1 *)pData;
  spacemit_kernels::nrow_block_q5_1<1> vDstTmp[32];
  int vNrow = ggml_nrows(pT);
  int vNblocks = pT->ne[0] / QK5_1;

  GGML_ASSERT(pDataSize == vNrow * vNblocks * sizeof(block_q5_1));

  if (pT->ne[1] % cNrowsInterleaved != 0 || pT->ne[0] % QK5_1 != 0) {
    return -1;
  }

  for (int vB = 0; vB < vNrow; vB += cNrowsInterleaved) {
    for (int64_t vX = 0; vX < vNblocks; vX++) {
      for (int vI = 0; vI < cNrowsInterleaved; vI++) {
        const block_q5_1 &vS = vSrc[vX + vI * vNblocks];

        float vD = GGML_FP16_TO_FP32(vS.GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.d);
        float vM = GGML_FP16_TO_FP32(vS.GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.m);

        if (vD == 0.0f) {
          vDstTmp[vI].scales16[0] = GGML_FP32_TO_FP16(std::fabs(vM));
          vDstTmp[vI].zp[0] = vM < 0.0f ? 1 : 0;
          memset(vDstTmp[vI].qh, 0, sizeof(vDstTmp[vI].qh));
          memset(vDstTmp[vI].qs, vM > 0.0f ? 0x11 : 0x00, sizeof(vDstTmp[vI].qs));
          continue;
        }

        float vMid = std::nearbyintf(-vM / vD);
        vMid = std::min(31.0f, std::max(0.0f, vMid));

        vDstTmp[vI].scales16[0] = GGML_FP32_TO_FP16(vD);
        vDstTmp[vI].zp[0] = static_cast<uint8_t>(vMid);

        // qs: copy low 4 bits directly (same nibble packing)
        memcpy(vDstTmp[vI].qs, vS.qs, QK5_1 / 2);

        // qh: copy 5th bit directly
        memcpy(vDstTmp[vI].qh, vS.qh, 4);
      }
      *vDst++ = fMakeBlockQ51x32(vDstTmp, pInterleaveBlock);
    }
    vSrc += cNrowsInterleaved * vNblocks;
  }
  return 0;
}

static int fRepackQ5KToQ5132Bl(ggml_tensor *pT, int pInterleaveBlock, const void *GGML_RESTRICT pData,
                               size_t pDataSize) {
  GGML_ASSERT(pT->type == GGML_TYPE_Q5_K);
  GGML_ASSERT(pInterleaveBlock == 32);
  GGML_ASSERT(QK_K / QK5_1 == 8);

  constexpr int cNrowsInterleaved = 32;

  spacemit_kernels::nrow_block_q5_1<32> *vDst = (spacemit_kernels::nrow_block_q5_1<32> *)pT->data;
  const block_q5_K *vSrc = (const block_q5_K *)pData;
  spacemit_kernels::nrow_block_q5_1<1> vDstTmp[32];
  int vNrow = ggml_nrows(pT);
  int vNblocks = pT->ne[0] / QK_K;

  if (pT->ne[1] % cNrowsInterleaved != 0 || pT->ne[0] % QK_K != 0) {
    return -1;
  }

  for (int vB = 0; vB < vNrow; vB += cNrowsInterleaved) {
    for (int64_t vX = 0; vX < vNblocks; vX++) {
      for (int vJ = 0; vJ < 8; vJ++) {
        for (int vI = 0; vI < cNrowsInterleaved; vI++) {
          uint8_t vSc, vM;
          const float cD = GGML_FP16_TO_FP32(vSrc[vX + vI * vNblocks].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.d);
          const float cMin = GGML_FP16_TO_FP32(vSrc[vX + vI * vNblocks].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.dmin);
          fGetScaleMinK4(vJ, vSrc[vX + vI * vNblocks].scales, &vSc, &vM);

          float vD1 = cD * vSc;
          float vM1 = cMin * vM;

          float vMid = std::nearbyintf(vM1 / vD1);
          vMid = std::min(31.0f, std::max(0.0f, vMid));
          vDstTmp[vI].scales16[0] = GGML_FP32_TO_FP16(vD1);
          vDstTmp[vI].zp[0] = static_cast<uint8_t>(vMid);

          // src -> [b0, b32] [b1, b33] ... [b31, b63]
          // dst -> [b0, b16] [b1, b17] ... [b15, b31] [b32, b48] [b33, b49] ... [b47, b63]
          const uint8_t *vQ = vSrc[vX + vI * vNblocks].qs + (vJ / 2) * QK5_1;
          if (vJ % 2 == 0) {
            for (int vIi = 0; vIi < 16; vIi++) {
              vDstTmp[vI].qs[vIi] = (vQ[vIi] & 0x0F) | ((vQ[vIi + 16] & 0x0F) << 4);
            }
          } else {
            for (int vIi = 0; vIi < 16; vIi++) {
              vDstTmp[vI].qs[vIi] = ((vQ[vIi] & 0xF0) >> 4) | (vQ[vIi + 16] & 0xF0);
            }
          }

          // Extract the 5th bit (qh) for this sub-block
          // block_q5_K.qh[32]: for sub-block j, the 5th bit is at bit position j in qh[l]
          // qs was reordered: dst_qs maps to src weights [0,16,1,17,...,15,31]
          // So qh must follow the same reorder to stay aligned with qs
          // dst qh[4] = 32 bits for 32 weights in the reordered layout:
          //   byte 0: weights 0..7   (from src_qh[0..7])
          //   byte 1: weights 8..15  (from src_qh[8..15])
          //   byte 2: weights 16..23 (from src_qh[16..23])
          //   byte 3: weights 24..31 (from src_qh[24..31])
          const uint8_t *vSrcQh = vSrc[vX + vI * vNblocks].qh;
          for (int vBi = 0; vBi < 4; vBi++) {
            uint8_t vQhByte = 0;
            for (int vK = 0; vK < 8; vK++) {
              int vSrcIdx = vBi * 8 + vK;
              vQhByte |= ((vSrcQh[vSrcIdx] >> vJ) & 1) << vK;
            }
            vDstTmp[vI].qh[vBi] = vQhByte;
          }
        }
        *vDst++ = fMakeBlockQ51x32(vDstTmp, pInterleaveBlock);
      }
    }
    vSrc += cNrowsInterleaved * vNblocks;
  }
  return 0;
}

namespace ggml::cpu::riscv64_spacemit {

template <typename BLOC_TYPE, int64_t INTER_SIZE, int64_t NB_COLS> int repack(ggml_tensor *, const void *, size_t);

template <> int repack<block_q4_0, 32, 16>(ggml_tensor *pT, const void *pData, size_t pDataSize) {
  return fRepackQ40ToQ4016Bl(pT, 16, pData, pDataSize);
}

template <> int repack<block_q4_1, 32, 16>(ggml_tensor *pT, const void *pData, size_t pDataSize) {
  return fRepackQ41ToQ4116Bl(pT, 16, pData, pDataSize);
}

template <> int repack<block_q4_K, 32, 16>(ggml_tensor *pT, const void *pData, size_t pDataSize) {
  return fRepackQ4KToQ4116Bl(pT, 16, pData, pDataSize);
}

template <> int repack<block_q2_K, 256, 32>(ggml_tensor *pT, const void *pData, size_t pDataSize) {
  return fRepackQ2KToQ2K32Bl(pT, 32, pData, pDataSize);
}

template <> int repack<block_q3_K, 256, 32>(ggml_tensor *pT, const void *pData, size_t pDataSize) {
  return fRepackQ3KToQ3K32Bl(pT, 32, pData, pDataSize);
}

template <> int repack<block_q4_0, 32, 32>(ggml_tensor *pT, const void *pData, size_t pDataSize) {
#if 0
    return fRepackQ40ToQ4032BlRef(pT, 32, pData, pDataSize);
#else
  return fRepackQ40ToQ4032Bl(pT, 32, pData, pDataSize);
#endif
}

template <> int repack<block_q4_0, 256, 32>(ggml_tensor *pT, const void *pData, size_t pDataSize) {
#if 1
  return fRepackQ40ToQ4025632BlRef(pT, 32, pData, pDataSize);
#else
  // return repack_q4_0_to_q4_0_256_32_bl(t, 32, data, data_size);
#endif
}

template <> int repack<block_q4_1, 32, 32>(ggml_tensor *pT, const void *pData, size_t pDataSize) {
#if 0
    return fRepackQ41ToQ4132BlRef(pT, 32, pData, pDataSize);
#else
  return fRepackQ41ToQ4132Bl(pT, 32, pData, pDataSize);
#endif
}

template <> int repack<block_q4_1, 256, 32>(ggml_tensor *pT, const void *pData, size_t pDataSize) {
#if 1
  return fRepackQ40ToQ4125632BlRef(pT, 32, pData, pDataSize);
#else
  return repack_q4_1_to_q4_1_256_32_bl(pT, 32, pData, pDataSize);
#endif
}

template <> int repack<block_q4_K, 32, 32>(ggml_tensor *pT, const void *pData, size_t pDataSize) {
  return fRepackQ4KToQ4132Bl(pT, 32, pData, pDataSize);
}

template <> int repack<block_q6_K, 32, 32>(ggml_tensor *pT, const void *pData, size_t pDataSize) {
#if 1
  return fRepackQ6KToQ8032BlRef(pT, 32, pData, pDataSize);
#else
  return fRepackQ6KToQ8032Bl(pT, 32, pData, pDataSize);
#endif
}

template <> int repack<block_q8_0, 32, 32>(ggml_tensor *pT, const void *pData, size_t pDataSize) {
#if 1
  return fRepackQ80ToQ8032BlRef(pT, 32, pData, pDataSize);
#else
  return fRepackQ80ToQ8032Bl(pT, 32, pData, pDataSize);
#endif
}

template <> int repack<block_mxfp4, 32, 32>(ggml_tensor *pT, const void *pData, size_t pDataSize) {
  return fRepackMxfp4ToMxfp432Bl(pT, 32, pData, pDataSize);
}

template <> int repack<block_q5_0, 32, 32>(ggml_tensor *pT, const void *pData, size_t pDataSize) {
  return fRepackQ50ToQ5032Bl(pT, 32, pData, pDataSize);
}

template <> int repack<block_q5_1, 32, 32>(ggml_tensor *pT, const void *pData, size_t pDataSize) {
  return fRepackQ51ToQ5132Bl(pT, 32, pData, pDataSize);
}

template <> int repack<block_q5_K, 32, 32>(ggml_tensor *pT, const void *pData, size_t pDataSize) {
  return fRepackQ5KToQ5132Bl(pT, 32, pData, pDataSize);
}

} // namespace ggml::cpu::riscv64_spacemit
