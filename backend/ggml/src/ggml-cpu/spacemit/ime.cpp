#define GGML_COMMON_IMPL_CPP
#define GGML_COMMON_DECL_CPP

#include "ime.h"

#include "binary-ops.h"
#include "common.h"
#include "ggml-backend-impl.h"
#include "ggml-common.h"
#include "ggml-cpu.h"
#include "ime_env.h"
#include "ime_kernels.h"
#include "ops.h"
#include "repack.h"
#include "rvv_kernels.h"
#include "spine_mem_pool.h"
#include "traits.h"
#include "vec.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cerrno>
#include <cmath>
#include <cstdio> // for GGML_ASSERT
#include <stdexcept>
#include <thread>
// clang-format off
#if defined(__riscv)

#if !defined(__riscv_v) || !defined(__riscv_v_intrinsic)
#error "riscv v extension or v_intrinsic not enabled"
#else
#include <riscv_vector.h>
#endif

#if !defined(__riscv_zfh) || !defined(__riscv_zvfh)
#error "riscv zfh extension not enabled, GGML_RV_ZFH and GGML_RV_ZVFH must be defined to 1"
#endif

#if !defined(__riscv_zba)
#error "riscv zba extension not enabled, GGML_RV_ZBA must be defined to 1"
#endif

#if defined(RISCV64_SPACEMIT_IME1) || defined(RISCV64_SPACEMIT_IME2)
#else
#error "RISCV64_SPACEMIT_IME1 or RISCV64_SPACEMIT_IME2 not defined"
#endif

#else

#error "riscv not enabled in this build"

#endif

#if defined(__GNUC__)
#pragma GCC diagnostic ignored "-Woverlength-strings"
#pragma GCC diagnostic ignored "-Wcast-qual"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#endif

// clang-format on

extern "C" {
extern void ggml_threadpool_chunk_set(struct ggml_threadpool *pTp, int pValue);
extern int ggml_threadpool_chunk_add(struct ggml_threadpool *pTp, int pValue);
}

namespace ggml::cpu::riscv64_spacemit {

struct TLSContext {
  int vCpuId{-1};
  cpu_set_t vCpuset;
  void *vTcmBuffer{nullptr};
  size_t vTcmBufferSize{0};
};

thread_local TLSContext tls_context;

template <typename BLOC_TYPE, int64_t INTER_SIZE, int64_t NB_COLS> constexpr size_t fGetRepackedBlockTypeSize() {
  if constexpr (std::is_same_v<BLOC_TYPE, block_q6_K> || std::is_same_v<BLOC_TYPE, block_q8_0>) {
    return sizeof(block_q8_0);
  } else if constexpr (std::is_same_v<BLOC_TYPE, block_q4_0>) {
    return sizeof(block_q4_0) * INTER_SIZE / QK4_0;
  } else if constexpr (std::is_same_v<BLOC_TYPE, block_q4_1> || std::is_same_v<BLOC_TYPE, block_q4_K>) {
    return (sizeof(block_q4_0) + sizeof(uint8_t)) * INTER_SIZE / QK4_1;
  } else if constexpr (std::is_same_v<BLOC_TYPE, block_q2_K>) {
    return sizeof(spacemit_kernels::nrow_block_q2_k<1>);
  } else if constexpr (std::is_same_v<BLOC_TYPE, block_q3_K>) {
    return sizeof(spacemit_kernels::nrow_block_q3_k<1>);
  } else if constexpr (std::is_same_v<BLOC_TYPE, block_mxfp4>) {
    return sizeof(spacemit_kernels::nrow_block_mxfp4<1>);
  } else if constexpr (std::is_same_v<BLOC_TYPE, block_q5_1> || std::is_same_v<BLOC_TYPE, block_q5_K>) {
    return sizeof(spacemit_kernels::nrow_block_q5_1<1>);
  } else if constexpr (std::is_same_v<BLOC_TYPE, block_q5_0>) {
    return sizeof(spacemit_kernels::nrow_block_q5_0<1>);
  } else {
    assert(false);
    return 0;
  }
}

template <typename BLOC_TYPE> constexpr bool fBlockTypeHasZp() {
  if constexpr (std::is_same_v<BLOC_TYPE, block_q6_K> || std::is_same_v<BLOC_TYPE, block_q8_0> ||
                std::is_same_v<BLOC_TYPE, block_q3_K> || std::is_same_v<BLOC_TYPE, block_q4_0> ||
                std::is_same_v<BLOC_TYPE, block_mxfp4> || std::is_same_v<BLOC_TYPE, block_q5_0>) {
    return false;
  } else if constexpr (std::is_same_v<BLOC_TYPE, block_q4_1> || std::is_same_v<BLOC_TYPE, block_q4_K> ||
                       std::is_same_v<BLOC_TYPE, block_q2_K> || std::is_same_v<BLOC_TYPE, block_q5_1> ||
                       std::is_same_v<BLOC_TYPE, block_q5_K>) {
    return true;
  } else {
    assert(false);
    return false;
  }
}

class tensor_traits_base : public ggml::cpu::tensor_traits {
public:
  virtual int mRepack(ggml_tensor *pT, const void *pData, size_t pDataSize) = 0;
};

template <typename BLOC_TYPE, int64_t INTER_SIZE, int64_t NB_COLS> class tensor_traits : public tensor_traits_base {
  bool work_size(int /* n_threads */, const ggml_tensor *pOp, size_t &pSize) override {
    switch (pOp->op) {
    case GGML_OP_MUL_MAT: {
      int64_t vSrc1Nelements = ggml_nelements(pOp->src[1]);

      if constexpr (std::is_same_v<BLOC_TYPE, block_q2_K> || std::is_same_v<BLOC_TYPE, block_q3_K>) {
        pSize = spacemit_kernels::div_round_up(vSrc1Nelements, QK_K) * spacemit_kernels::q8k_blk_size(QK_K);
      } else if constexpr (INTER_SIZE == QK4_0) {
        pSize = spacemit_kernels::div_round_up(vSrc1Nelements, QK4_0) * spacemit_kernels::q8_blk_size(QK4_0, true);
      } else if constexpr (INTER_SIZE == 256) {
        pSize = spacemit_kernels::div_round_up(vSrc1Nelements, 256) * spacemit_kernels::q8_hp_blk_size(256, true, true);
      } else {
        GGML_ABORT("unsupported block type");
      }

      pSize = GGML_PAD(pSize, sizeof(int64_t));

      return true;
    }
    case GGML_OP_MUL_MAT_ID: {
      int64_t vSrc1Nelements = ggml_nelements(pOp->src[1]);

      if constexpr (std::is_same_v<BLOC_TYPE, block_q2_K> || std::is_same_v<BLOC_TYPE, block_q3_K>) {
        pSize = spacemit_kernels::div_round_up(vSrc1Nelements, QK_K) * spacemit_kernels::q8k_blk_size(QK_K);
      } else if constexpr (INTER_SIZE == QK4_0) {
        pSize = spacemit_kernels::div_round_up(vSrc1Nelements, QK4_0) * spacemit_kernels::q8_blk_size(QK4_0, true);
      } else if constexpr (INTER_SIZE == 256) {
        pSize = spacemit_kernels::div_round_up(vSrc1Nelements, 256) * spacemit_kernels::q8_hp_blk_size(256, true, true);
      } else {
        GGML_ABORT("unsupported block type");
      }

      pSize = GGML_PAD(pSize, sizeof(int64_t));

      const int64_t cNe02 = pOp->src[0]->ne[2]; // n_as, n_expert
      const int64_t cNe12 = pOp->src[1]->ne[2]; // n_tokens

      const size_t cSizeofMmidRowMapping = sizeof(int64_t);
      pSize += cSizeofMmidRowMapping * cNe02 * (cNe12 + 1) + (cNe02 + 1) * sizeof(int64_t);

      pSize = GGML_PAD(pSize, sizeof(int64_t));

      return true;
    }
    default:
      // GGML_ABORT("fatal error");
      break;
    }
    return false;
  }

  bool compute_forward(ggml_compute_params *pParams, ggml_tensor *pOp) override {
    switch (pOp->op) {
    case GGML_OP_MUL_MAT:
      switch (pOp->src[0]->type) {
      case GGML_TYPE_Q2_K:
      case GGML_TYPE_Q3_K:
      case GGML_TYPE_Q4_0:
      case GGML_TYPE_Q4_1:
      case GGML_TYPE_Q4_K:
      case GGML_TYPE_Q6_K:
      case GGML_TYPE_Q8_0:
      case GGML_TYPE_Q5_1:
      case GGML_TYPE_Q5_K:
        // case GGML_TYPE_MXFP4:
        mForwardMulMat(pParams, pOp);
        return true;
      default:
        // GGML_ABORT("fatal error: unsupported type for src0 in MUL_MAT");
        return false;
      }
      break;
    case GGML_OP_MUL_MAT_ID:
      switch (pOp->src[0]->type) {
      case GGML_TYPE_Q2_K:
      case GGML_TYPE_Q3_K:
      case GGML_TYPE_Q4_0:
      case GGML_TYPE_Q4_1:
      case GGML_TYPE_Q4_K:
      case GGML_TYPE_Q6_K:
      case GGML_TYPE_Q8_0:
      case GGML_TYPE_Q5_1:
      case GGML_TYPE_Q5_K:
        // case GGML_TYPE_MXFP4:
        mForwardMulMatId(pParams, pOp);
        return true;
      default:
        // GGML_ABORT("fatal error: unsupported type for src0 in MUL_MAT_ID");
        return false;
      }
      break;
    default:
      // GGML_ABORT("fatal error");
      break;
    }
    return false;
  }

  void mForwardMulMat(ggml_compute_params *pParams, ggml_tensor *pOp) {
    constexpr size_t cABlkLen = INTER_SIZE;
    constexpr size_t cBBlkLen = INTER_SIZE;

    const ggml_tensor *src0 = pOp->src[0];
    const ggml_tensor *src1 = pOp->src[1];
    ggml_tensor *dst = pOp;

    GGML_TENSOR_BINARY_OP_LOCALS

    int vIth = pParams->ith;
    int vNth = pParams->nth;

    [[maybe_unused]] const enum ggml_type cType = src0->type;

    void *vWData = (void *)src0->data;
    const float *vFeature = (const float *)src1->data;
    float *vOutput = (float *)dst->data;

    const int64_t cGemmM = ne11 * ne12 * ne13;
    const int64_t cGemmK = ne10;
    const int64_t cGemmN = ne01;

    spacemit_kernels::quantize_a_row_def vQuantizeARowI8;
    spacemit_kernels::quantize_a_row_def vQuantizeA4rowI8;
    spacemit_kernels::gemm_kernel_quantize_def vGemmKernel;
    bool vSetKernelImpl = false;

    int64_t vBlockStrideA = spacemit_kernels::q8_blk_size(cABlkLen);

#if defined(RISCV64_SPACEMIT_IME2)
    if (!vSetKernelImpl && (global_spine_env_info.use_ime2)) {
      vQuantizeARowI8 = spacemit_kernels::rvv::quantize_a_row_i8;
      vQuantizeA4rowI8 = spacemit_kernels::rvv::quantize_a_4row_i8;
      vBlockStrideA = spacemit_kernels::q8_blk_size(cABlkLen, true);

      if constexpr (std::is_same_v<BLOC_TYPE, block_q6_K> || std::is_same_v<BLOC_TYPE, block_q8_0>) {
        vGemmKernel = spacemit_kernels::ime2::gemm_kernel_i8i8;
        vSetKernelImpl = true;
      } else if constexpr (std::is_same_v<BLOC_TYPE, block_q4_0> || std::is_same_v<BLOC_TYPE, block_q4_1> ||
                           std::is_same_v<BLOC_TYPE, block_q4_K>) {
        if constexpr (INTER_SIZE == 256) {
          vGemmKernel = spacemit_kernels::ime2::gemm_kernel_i8i4_hp;
          vQuantizeARowI8 = spacemit_kernels::rvv::quantize_a_row_i8_hp;
          vQuantizeA4rowI8 = spacemit_kernels::rvv::quantize_a_4row_i8_hp;
          vBlockStrideA = spacemit_kernels::q8_hp_blk_size(cABlkLen, true, true);
          vSetKernelImpl = true;
        } else {
          vGemmKernel = spacemit_kernels::ime2::gemm_kernel_i8i4;
          vQuantizeARowI8 = spacemit_kernels::rvv::quantize_a_row_i8;
          vQuantizeA4rowI8 = spacemit_kernels::rvv::quantize_a_4row_i8;
          vBlockStrideA = spacemit_kernels::q8_blk_size(cABlkLen, true);
          vSetKernelImpl = true;
        }
      } else if constexpr (std::is_same_v<BLOC_TYPE, block_q2_K>) {
        vQuantizeARowI8 = spacemit_kernels::rvv::quantize_a_row_i8k;
        vQuantizeA4rowI8 = spacemit_kernels::rvv::quantize_a_4row_i8k;
        vBlockStrideA = spacemit_kernels::q8k_blk_size(cABlkLen);

        vGemmKernel = spacemit_kernels::ime2::gemm_kernel_i8i2k;
        vSetKernelImpl = true;
      } else if constexpr (std::is_same_v<BLOC_TYPE, block_q3_K>) {
        vQuantizeARowI8 = spacemit_kernels::rvv::quantize_a_row_i8k;
        vQuantizeA4rowI8 = spacemit_kernels::rvv::quantize_a_4row_i8k;
        vBlockStrideA = spacemit_kernels::q8k_blk_size(cABlkLen);

        vGemmKernel = spacemit_kernels::ime2::gemm_kernel_i8i3k;
        vSetKernelImpl = true;
      } else if constexpr (std::is_same_v<BLOC_TYPE, block_mxfp4>) {
        vGemmKernel = spacemit_kernels::ime2::gemm_kernel_i8mxfp4;
        vSetKernelImpl = true;
      } else if constexpr (std::is_same_v<BLOC_TYPE, block_q5_1> || std::is_same_v<BLOC_TYPE, block_q5_K> ||
                           std::is_same_v<BLOC_TYPE, block_q5_0>) {
        vGemmKernel = spacemit_kernels::ime2::gemm_kernel_i8i5;
        vSetKernelImpl = true;
      }
    }
#endif

#if defined(RISCV64_SPACEMIT_IME1)
    if (!vSetKernelImpl && (global_spine_env_info.use_ime1)) {
      vQuantizeARowI8 = spacemit_kernels::ime1::quantize_a_row_i8;
      vQuantizeA4rowI8 = spacemit_kernels::ime1::quantize_a_4row_i8;

      if constexpr (std::is_same_v<BLOC_TYPE, block_q4_0> || std::is_same_v<BLOC_TYPE, block_q4_1> ||
                    std::is_same_v<BLOC_TYPE, block_q4_K>) {
        vGemmKernel = spacemit_kernels::ime1::gemm_kernel_i8i4;
        vSetKernelImpl = true;
      }
    }
#endif
    if (!vSetKernelImpl) {
      GGML_ABORT("no kernel implementation found for the block type");
    }

    const int64_t cAKBlks = spacemit_kernels::div_round_up(cGemmK, cABlkLen);
    const int64_t cBKBlks = spacemit_kernels::div_round_up(cGemmK, cBBlkLen);

    const int64_t cRowStrideA = cAKBlks * vBlockStrideA;
    const int64_t cGemmWorkspaceSize = GGML_PAD(cGemmM * cRowStrideA, alignof(int64_t));

    if (vIth == 0 && pParams->wsize < cGemmWorkspaceSize) {
      GGML_ABORT("wsize less than gemm_workspace_size");
    }

    uintptr_t vWsPtr = reinterpret_cast<uintptr_t>(pParams->wdata);

    void *vTcmBuffer = ggml::cpu::riscv64_spacemit::tls_context.vTcmBuffer;
    const int64_t cTcmBufferSize = ggml::cpu::riscv64_spacemit::tls_context.vTcmBufferSize;

    auto *vQuantABuffer = reinterpret_cast<uint8_t *>(vWsPtr);

    constexpr int64_t cRowAlign = 4;
    const int64_t cRowBlks = spacemit_kernels::div_round_up(cGemmM, cRowAlign);

    const int64_t cRowStrideB = cBKBlks * fGetRepackedBlockTypeSize<BLOC_TYPE, INTER_SIZE, NB_COLS>();
    const int64_t cPerMbRowsWsize = cRowAlign * cRowStrideA;
    const int64_t cPerNbColsWsize = NB_COLS * cRowStrideB;

    const int64_t cBarrierIdx = static_cast<int64_t>(vIth / 2);

    GGML_ASSERT(global_spine_env_info.init_barrier != nullptr);
    GGML_ASSERT(cBarrierIdx < spine_init_barrier_count);
    spine_barrier_t *vCurBarrier = &global_spine_env_info.init_barrier[cBarrierIdx];

    if (cGemmM == 1) {
      int vTaskPerThread = spacemit_kernels::div_round_up(cAKBlks, vNth);
      int vABlkStart = vIth * vTaskPerThread;
      int vABlkEnd = std::min(vABlkStart + vTaskPerThread, (int)cAKBlks);
      if (vABlkStart < vABlkEnd) {
        vQuantizeARowI8(cABlkLen, vFeature + vABlkStart * cABlkLen, (vABlkEnd - vABlkStart) * cABlkLen,
                        vQuantABuffer + vABlkStart * vBlockStrideA);
      }
    } else {
      int vTaskPerThread = spacemit_kernels::div_round_up(cRowBlks, vNth);
      int vMRowBlkStart = vIth * vTaskPerThread;
      int vMRowBlkEnd = std::min(vMRowBlkStart + vTaskPerThread, (int)cRowBlks);
      for (int vMRowBlk = vMRowBlkStart; vMRowBlk < vMRowBlkEnd; vMRowBlk++) {
        int vMIdx = vMRowBlk * cRowAlign;
        int vRowsTobeHandled = (cGemmM - vMIdx) > cRowAlign ? cRowAlign : (cGemmM - vMIdx);

        if (vRowsTobeHandled == cRowAlign && vQuantizeA4rowI8 != nullptr) {
          const float *vARowPtr = vFeature + vMIdx * cGemmK;
          auto *vQuantARowPtr = vQuantABuffer + vMIdx * cRowStrideA;
          vQuantizeA4rowI8(cABlkLen, vARowPtr, cGemmK, vQuantARowPtr);
        } else {
          while (vRowsTobeHandled) {
            const float *vARowPtr = vFeature + vMIdx * cGemmK;
            auto *vQuantARowPtr = vQuantABuffer + vMIdx * cRowStrideA;
            vQuantizeARowI8(cABlkLen, vARowPtr, cGemmK, vQuantARowPtr);
            vRowsTobeHandled -= 1;
            vMIdx += 1;
          }
        }
      }
    }

    ggml_barrier(pParams->threadpool);

    const int64_t cGemmMStride = cGemmN / cGemmM > 64 ? cGemmM : 16;
    const int64_t cGemmMBlocked = spacemit_kernels::div_round_up(cGemmM, cGemmMStride);
    const int64_t cMaxGemmNStride = spacemit_kernels::div_round_up(cGemmN * cGemmMBlocked, vNth);

    int64_t vGemmNStride = cGemmN;
    if (cMaxGemmNStride < cGemmN) {
      vGemmNStride = std::min(vGemmNStride, spacemit_kernels::div_round_up(cMaxGemmNStride, NB_COLS) * NB_COLS);
    }

    if (vGemmNStride == cGemmN && vTcmBuffer != nullptr && cPerMbRowsWsize <= cTcmBufferSize) {
      for (int64_t vMStart = vIth * cRowAlign; vMStart < cGemmM; vMStart += cRowAlign * vNth) {
        uint8_t *vBCol = reinterpret_cast<uint8_t *>(vWData);
        uint8_t *vBColZp = fBlockTypeHasZp<BLOC_TYPE>() ? vBCol : nullptr;

        int64_t vMRowReal = std::min(cGemmM - vMStart, cRowAlign);

        spacemit_kernels::rvv::memcpy1d(vTcmBuffer, vQuantABuffer + vMStart * cRowStrideA, vMRowReal * cRowStrideA);

        int64_t vNBlkReal = 0;
        for (int64_t vNi = 0; vNi < cGemmN; vNi += vNBlkReal, vBCol += vNBlkReal * cRowStrideB) {
          vNBlkReal = std::min(cGemmN - vNi, (int64_t)NB_COLS);

          uint8_t *vARowPtr = (uint8_t *)vTcmBuffer;
          float *vCBlk = vOutput + vMStart * cGemmN + vNi;

          int32_t vRowsRemaining = vMRowReal;

          while (vRowsRemaining > 0) {
            auto vRowsHandled =
                vGemmKernel(cBBlkLen, vARowPtr, vBCol, vBColZp, vCBlk, vRowsRemaining, vNBlkReal, cBKBlks, cGemmN);

            vCBlk += vRowsHandled * cGemmN;
            vARowPtr += vRowsHandled * cRowStrideA;

            vRowsRemaining -= vRowsHandled;
          }
        }
      }
    } else if (vTcmBuffer != nullptr && cPerNbColsWsize <= cTcmBufferSize) {
      uint8_t *vARow = vQuantABuffer;
      uint8_t *vBCol = reinterpret_cast<uint8_t *>(vTcmBuffer);
      if ((cGemmWorkspaceSize + cPerNbColsWsize) <= cTcmBufferSize) {
        vARow = (uint8_t *)vTcmBuffer;
        vBCol = reinterpret_cast<uint8_t *>(vTcmBuffer) + cGemmWorkspaceSize;
      }
      uint8_t *vBColZp = fBlockTypeHasZp<BLOC_TYPE>() ? vBCol : nullptr;

      int64_t vNi = vIth * NB_COLS;
      int64_t vNbReal = std::min(cGemmN - vNi, NB_COLS);

      if (vIth % 2 == 0 && vNbReal > 0) {
        spacemit_kernels::rvv::memcpy1d(vBCol, reinterpret_cast<uint8_t *>(vWData) + vNi * cRowStrideB,
                                        vNbReal * cRowStrideB);
        if (vARow != vQuantABuffer) {
          spacemit_kernels::rvv::memcpy1d(vARow, vQuantABuffer, cGemmWorkspaceSize);
        }
      }

      spine_barrier_wait(vCurBarrier);

      if (vIth % 2 != 0 && vNbReal > 0) {
        if (vARow != vQuantABuffer) {
          spacemit_kernels::rvv::memcpy1d(vARow, vQuantABuffer, cGemmWorkspaceSize);
        }
        spacemit_kernels::rvv::memcpy1d(vBCol, reinterpret_cast<uint8_t *>(vWData) + vNi * cRowStrideB,
                                        vNbReal * cRowStrideB);
      }

      for (; vNi < cGemmN; vNi += NB_COLS * vNth) {
        int64_t vRowsRemaining = cGemmM;
        float *vCBlk = vOutput + vNi;
        auto *vARowCur = vARow;

        if (vIth % 2 != 0) {
          spine_barrier_wait(vCurBarrier);
        }

        while (vRowsRemaining > 0) {
          auto vRowsHandled =
              vGemmKernel(cBBlkLen, vARowCur, vBCol, vBColZp, vCBlk, vRowsRemaining, vNbReal, cBKBlks, cGemmN);

          vCBlk += vRowsHandled * cGemmN;
          vARowCur += vRowsHandled * cRowStrideA;

          vRowsRemaining -= vRowsHandled;
        }

        if (vIth % 2 == 0) {
          spine_barrier_wait(vCurBarrier);
        }

        const int64_t cNextNi = vNi + NB_COLS * vNth;
        if (cNextNi < cGemmN) {
          vNbReal = std::min(cGemmN - cNextNi, NB_COLS);
          spacemit_kernels::rvv::memcpy1d(vBCol, reinterpret_cast<uint8_t *>(vWData) + cNextNi * cRowStrideB,
                                          vNbReal * cRowStrideB);
        }
      }
    } else {
      const int64_t cTaskCountM = spacemit_kernels::div_round_up(cGemmM, cGemmMStride);
      const int64_t cTaskCountN = spacemit_kernels::div_round_up(cGemmN, vGemmNStride);

      int64_t vTaskCount = cTaskCountM * cTaskCountN;
      int64_t vTaskPerThread = (vTaskCount + vNth - 1) / vNth;
      int64_t vStart = vIth * vTaskPerThread;
      int64_t vEnd = std::min((vIth + 1) * vTaskPerThread, vTaskCount);
      for (int64_t vComputeIdx = vStart; vComputeIdx < vEnd; vComputeIdx++) {
        const auto cTidN = vComputeIdx / cTaskCountM;
        const auto cTidM = vComputeIdx % cTaskCountM;

        const int64_t cMStart = cTidM * cGemmMStride;
        const int64_t cMCount = std::min(cGemmM - cMStart, (int64_t)cGemmMStride);

        const int64_t cNStart = cTidN * vGemmNStride;
        const int64_t cNCount = std::min(cGemmN - cNStart, (int64_t)vGemmNStride);

        const int64_t cNBlk = cMCount == 1 ? cNCount : NB_COLS;

        uint8_t *vBCol = reinterpret_cast<uint8_t *>(vWData) + cNStart * cRowStrideB;
        uint8_t *vBColZp = fBlockTypeHasZp<BLOC_TYPE>() ? vBCol : nullptr;

        int64_t vNBlkReal = 0;
        for (int64_t vNi = 0; vNi < cNCount; vNi += vNBlkReal, vBCol += vNBlkReal * cRowStrideB) {
          vNBlkReal = std::min(cNCount - vNi, cNBlk);

          uint8_t *vARow = vQuantABuffer + cMStart * cRowStrideA;

          float *vCBlk = vOutput + cMStart * cGemmN + cNStart + vNi;

          int64_t vRowsRemaining = cMCount;

          uint8_t *vBColCur = vBCol;
          uint8_t *vBColZpCur = vBColZp;

          while (vRowsRemaining > 0) {
            auto vRowsHandled =
                vGemmKernel(cBBlkLen, vARow, vBColCur, vBColZpCur, vCBlk, vRowsRemaining, vNBlkReal, cBKBlks, cGemmN);

            vCBlk += vRowsHandled * cGemmN;
            vARow += vRowsHandled * cRowStrideA;

            vRowsRemaining -= vRowsHandled;
          }
        }
      }
    }
  }

  void mForwardMulMatId(ggml_compute_params *pParams, ggml_tensor *pOp) {
    constexpr size_t cABlkLen = INTER_SIZE;
    constexpr size_t cBBlkLen = INTER_SIZE;

    const ggml_tensor *src0 = pOp->src[0];
    const ggml_tensor *src1 = pOp->src[1];
    const ggml_tensor *vIds = pOp->src[2];
    ggml_tensor *dst = pOp;

    GGML_TENSOR_BINARY_OP_LOCALS

    int vIth = pParams->ith;
    int vNth = pParams->nth;

    // row groups
    const int cNIds = vIds->ne[0]; // n_expert_used
    const int cNAs = ne02;         // n_expert

    struct mmid_row_mapping {
      int32_t i1;
      int32_t vI2;
    };

    spacemit_kernels::quantize_a_row_def vQuantizeARowI8;
    spacemit_kernels::gemm_kernel_quantize_def vGemmKernel;
    spacemit_kernels::moe_gemm_kernel_quantize_def vMoeGemmKernelM2;
    bool vSetKernelImpl = false;
    size_t vBlockStrideA = spacemit_kernels::q8_blk_size(QK4_0);

#if defined(RISCV64_SPACEMIT_IME2)
    if (!vSetKernelImpl && (global_spine_env_info.use_ime2)) {
      vQuantizeARowI8 = spacemit_kernels::rvv::quantize_a_row_i8;
      vBlockStrideA = spacemit_kernels::q8_blk_size(QK4_0, true);

      if constexpr (std::is_same_v<BLOC_TYPE, block_q6_K> || std::is_same_v<BLOC_TYPE, block_q8_0>) {
        vGemmKernel = spacemit_kernels::ime2::gemm_kernel_i8i8;
        vSetKernelImpl = true;
      } else if constexpr (std::is_same_v<BLOC_TYPE, block_q4_0> || std::is_same_v<BLOC_TYPE, block_q4_1> ||
                           std::is_same_v<BLOC_TYPE, block_q4_K>) {
        if constexpr (INTER_SIZE == 256) {
          vGemmKernel = spacemit_kernels::ime2::gemm_kernel_i8i4_hp;
          vQuantizeARowI8 = spacemit_kernels::rvv::quantize_a_row_i8_hp;
          vBlockStrideA = spacemit_kernels::q8_hp_blk_size(cABlkLen, true, true);
          vSetKernelImpl = true;
        } else {
          vGemmKernel = spacemit_kernels::ime2::gemm_kernel_i8i4;
          vMoeGemmKernelM2 = spacemit_kernels::ime2::moe_m2_gemm_kernel_i8i4;
          vQuantizeARowI8 = spacemit_kernels::rvv::quantize_a_row_i8;
          vBlockStrideA = spacemit_kernels::q8_blk_size(cABlkLen, true);
          vSetKernelImpl = true;
        }
      } else if constexpr (std::is_same_v<BLOC_TYPE, block_q2_K>) {
        vQuantizeARowI8 = spacemit_kernels::rvv::quantize_a_row_i8k;
        vBlockStrideA = spacemit_kernels::q8k_blk_size(cABlkLen);
        vGemmKernel = spacemit_kernels::ime2::gemm_kernel_i8i2k;
        vSetKernelImpl = true;
      } else if constexpr (std::is_same_v<BLOC_TYPE, block_q3_K>) {
        vQuantizeARowI8 = spacemit_kernels::rvv::quantize_a_row_i8k;
        vBlockStrideA = spacemit_kernels::q8k_blk_size(cABlkLen);
        vGemmKernel = spacemit_kernels::ime2::gemm_kernel_i8i3k;
        vSetKernelImpl = true;
      } else if constexpr (std::is_same_v<BLOC_TYPE, block_mxfp4>) {
        vGemmKernel = spacemit_kernels::ime2::gemm_kernel_i8mxfp4;
        vMoeGemmKernelM2 = spacemit_kernels::ime2::moe_m2_gemm_kernel_i8mxfp4;
        vSetKernelImpl = true;
      } else if constexpr (std::is_same_v<BLOC_TYPE, block_q5_1> || std::is_same_v<BLOC_TYPE, block_q5_K> ||
                           std::is_same_v<BLOC_TYPE, block_q5_0>) {
        vGemmKernel = spacemit_kernels::ime2::gemm_kernel_i8i5;
        vMoeGemmKernelM2 = spacemit_kernels::ime2::moe_m2_gemm_kernel_i8i5;
        vSetKernelImpl = true;
      }
    }
#endif

#if defined(RISCV64_SPACEMIT_IME1)
    if (!vSetKernelImpl && (global_spine_env_info.use_ime1)) {
      vQuantizeARowI8 = spacemit_kernels::ime1::quantize_a_row_i8;

      if constexpr (std::is_same_v<BLOC_TYPE, block_q4_0> || std::is_same_v<BLOC_TYPE, block_q4_1> ||
                    std::is_same_v<BLOC_TYPE, block_q4_K>) {
        vGemmKernel = spacemit_kernels::ime1::gemm_kernel_i8i4;
        vSetKernelImpl = true;
      }
    }
#endif
    if (!vSetKernelImpl) {
      GGML_ABORT("no kernel implementation found for the block type");
    }

    const size_t cAKBlks = spacemit_kernels::div_round_up(ne10, cABlkLen);
    const size_t cBKBlks = spacemit_kernels::div_round_up(ne10, cBBlkLen);

    const size_t cNbw1 = cAKBlks * vBlockStrideA;
    const size_t cNbw2 = ne11 * cNbw1;
    const size_t cNbw3 = cNbw2 * ne12;
    const size_t cGemmWorkspaceSize = GGML_PAD(cNbw3, alignof(int64_t));

    const uintptr_t cWsPtr = reinterpret_cast<uintptr_t>(pParams->wdata);
    auto *vQuantABuffer = reinterpret_cast<uint8_t *>(cWsPtr);

    if (ne11 == 1) {
      for (int64_t vIi = vIth; vIi < ne12 * cAKBlks; vIi += vNth) {
        int64_t vI12 = vIi / cAKBlks;
        int64_t vAkBlkId = vIi % cAKBlks;
        vQuantizeARowI8(cABlkLen, (float *)((char *)src1->data + vI12 * nb12) + vAkBlkId * cABlkLen, cABlkLen,
                        vQuantABuffer + vI12 * cNbw2 + vAkBlkId * vBlockStrideA);
      }
    } else {
      for (int64_t vIi = vIth; vIi < ne12 * ne11; vIi += vNth) {
        int64_t vI12 = vIi / ne11;
        int64_t vI11 = vIi % ne11;
        vQuantizeARowI8(cABlkLen, (float *)((char *)src1->data + vI12 * nb12 + vI11 * nb11), ne10,
                        vQuantABuffer + vI12 * cNbw2 + vI11 * cNbw1);
      }
    }

#define MMID_MATRIX_ROW(row_id, i1) matrix_rows[(row_id) * ne12 + (i1)]

    int64_t *vMatrixRowCounts = (int64_t *)(cWsPtr + cGemmWorkspaceSize);
    int32_t *vValidEpCount = (int32_t *)(vMatrixRowCounts + cNAs);
    int32_t *vValidActCount = (int32_t *)(vValidEpCount + 1);
    int64_t *vValidMatrixRowCounts = (int64_t *)(vValidActCount + 1);
    mmid_row_mapping *matrix_rows = (mmid_row_mapping *)(vValidMatrixRowCounts + cNAs);

    if (vIth == 0) {
      // initialize matrix_row_counts
      memset(vMatrixRowCounts, 0, cNAs * sizeof(int64_t));

      // group rows by src0 matrix
      for (int32_t vIid1 = 0; vIid1 < vIds->ne[1]; ++vIid1) {
        for (int32_t vId = 0; vId < cNIds; ++vId) {
          const int32_t cI02 = *(const int32_t *)((const char *)vIds->data + vIid1 * vIds->nb[1] + vId * vIds->nb[0]);

          GGML_ASSERT(cI02 >= 0 && cI02 < cNAs);

          MMID_MATRIX_ROW(cI02, vMatrixRowCounts[cI02]) = {vId, vIid1};
          vMatrixRowCounts[cI02] += 1;
        }
      }

      int32_t vValidEpCountT = 0;
      int32_t vValidActCountT = 0;
      for (int vCurA = 0; vCurA < cNAs; ++vCurA) {
        const int64_t cCne1 = vMatrixRowCounts[vCurA];
        if (cCne1 == 0) {
          continue;
        }
        vValidMatrixRowCounts[vValidEpCountT] = vCurA;
        vValidActCountT += cCne1;
        vValidEpCountT += 1;
      }
      vValidEpCount[0] = vValidEpCountT;
      vValidActCount[0] = vValidActCountT;
    }

    const int64_t cBarrierIdx = static_cast<int64_t>(vIth / 2);

    GGML_ASSERT(global_spine_env_info.init_barrier != nullptr);
    GGML_ASSERT(cBarrierIdx < spine_init_barrier_count);
    spine_barrier_t *vCurBarrier = &global_spine_env_info.init_barrier[cBarrierIdx];

    ggml_barrier(pParams->threadpool);

    const size_t cRowStrideB = cBKBlks * fGetRepackedBlockTypeSize<BLOC_TYPE, INTER_SIZE, NB_COLS>();
    const size_t cExpertBStride = ne01 * cRowStrideB;
    const size_t cPerNbColsWsize = NB_COLS * cRowStrideB;

    std::array<const uint8_t *, 2> aSrcWorkspaces;
    std::array<float *, 2> aDstWorkspaces;

    auto *vTcmBuffer = ggml::cpu::riscv64_spacemit::tls_context.vTcmBuffer;
    const auto cTcmBufferSize = ggml::cpu::riscv64_spacemit::tls_context.vTcmBufferSize;

    const auto cValidEpCountT = vValidEpCount[0];
    const auto cValidActCountT = vValidActCount[0];

    int vNthEs = 1;
    int vNthN = vNth;

    int vIthEs = vIth % vNthEs;
    int vIthN = (vIth / vNthEs) % vNthN;

    if (cValidEpCountT % vNth == 0 && vTcmBuffer != nullptr && cValidEpCountT == cNAs && cValidActCountT == cNAs &&
        cPerNbColsWsize <= cTcmBufferSize) {
      for (int64_t vValidId = vIth; vValidId < cValidEpCountT; vValidId += vNth) {
        const int64_t cCurA = vValidMatrixRowCounts[vValidId];

        auto *vSrc0Cur = (uint8_t *)src0->data + cCurA * cExpertBStride;

        mmid_row_mapping row_mapping = MMID_MATRIX_ROW(cCurA, 0);
        const int cId = row_mapping.i1;
        const int64_t cI11 = cId % ne11;
        const int64_t cI12 = row_mapping.vI2;
        const int64_t i1 = cId;
        const int64_t cI2 = cI12;

        auto *vSrc1Col = vQuantABuffer + (cI11 * cNbw1 + cI12 * cNbw2);
        float *vCBlk = (float *)((char *)dst->data + (i1 * nb1 + cI2 * nb2));

        uint8_t *vARow = vSrc1Col;
        uint8_t *vBCol = reinterpret_cast<uint8_t *>(vTcmBuffer);
        if ((cNbw1 + cPerNbColsWsize) <= cTcmBufferSize) {
          vARow = (uint8_t *)vTcmBuffer;
          vBCol = reinterpret_cast<uint8_t *>(vTcmBuffer) + cNbw1;
        }
        uint8_t *vBColZp = fBlockTypeHasZp<BLOC_TYPE>() ? vBCol : nullptr;

        if (vIth % 2 == 0) {
          spacemit_kernels::rvv::memcpy1d(vBCol, reinterpret_cast<uint8_t *>(vSrc0Cur), cPerNbColsWsize);

          if (vARow != vSrc1Col) {
            spacemit_kernels::rvv::memcpy1d(vARow, vSrc1Col, cNbw1);
          }
        }

        spine_barrier_wait(vCurBarrier);

        if (vIth % 2 != 0) {
          if (vARow != vSrc1Col) {
            spacemit_kernels::rvv::memcpy1d(vARow, vSrc1Col, cNbw1);
          }

          spacemit_kernels::rvv::memcpy1d(vBCol, reinterpret_cast<uint8_t *>(vSrc0Cur), cPerNbColsWsize);
        }

        int64_t vNbReal = std::min(ne01, NB_COLS);
        for (int64_t vNi = 0; vNi < ne01; vNi += NB_COLS) {
          if (vIth % 2 != 0) {
            spine_barrier_wait(vCurBarrier);
          }

          vGemmKernel(cBBlkLen, vARow, vBCol, vBColZp, vCBlk + vNi, 1, vNbReal, cBKBlks, ne01);

          if (vIth % 2 == 0) {
            spine_barrier_wait(vCurBarrier);
          }

          const int64_t cNextNi = vNi + NB_COLS;
          if (cNextNi < ne01) {
            vNbReal = std::min(ne01 - cNextNi, NB_COLS);
            spacemit_kernels::rvv::memcpy1d(vBCol, reinterpret_cast<uint8_t *>(vSrc0Cur) + cNextNi * cRowStrideB,
                                            cPerNbColsWsize);
          }
        }
      }
    } else {
      for (int64_t vValidId = vIthEs; vValidId < cValidEpCountT; vValidId += vNthEs) {
        const int64_t cCurA = vValidMatrixRowCounts[vValidId];
        const int64_t cCne1 = vMatrixRowCounts[cCurA];

        int64_t vSrc1CurStart = 0;
        int64_t vSrc1CurEnd = cCne1;

        int64_t vSrc0CurStart = (vIthN * ne01) / vNthN;
        int64_t vSrc0CurEnd = MIN(((vIthN + 1) * ne01) / vNthN, ne01);

        if (vSrc1CurStart >= vSrc1CurEnd || vSrc0CurStart >= vSrc0CurEnd) {
          continue;
        }

        vSrc0CurStart = (vSrc0CurStart % NB_COLS) ? vSrc0CurStart + NB_COLS - (vSrc0CurStart % NB_COLS) : vSrc0CurStart;
        vSrc0CurEnd = (vSrc0CurEnd % NB_COLS) ? vSrc0CurEnd + NB_COLS - (vSrc0CurEnd % NB_COLS) : vSrc0CurEnd;

        auto *vSrc0Cur = (uint8_t *)src0->data + cCurA * cExpertBStride + vSrc0CurStart * cRowStrideB;
        uint8_t *vBColZp = fBlockTypeHasZp<BLOC_TYPE>() ? vSrc0Cur : nullptr;

        size_t vExtraTcmBufferSize = cTcmBufferSize;
        void *vExtraTcmBuffer = vTcmBuffer;
        if (vTcmBuffer != nullptr && (vSrc1CurEnd - vSrc1CurStart) >= 4 &&
            (vSrc0CurEnd - vSrc0CurStart) * cRowStrideB <= cTcmBufferSize) {
          spacemit_kernels::rvv::memcpy1d(vTcmBuffer, vSrc0Cur, (vSrc0CurEnd - vSrc0CurStart) * cRowStrideB);
          vSrc0Cur = reinterpret_cast<uint8_t *>(vTcmBuffer);
          vBColZp = fBlockTypeHasZp<BLOC_TYPE>() ? vSrc0Cur : nullptr;
          vExtraTcmBufferSize -= (vSrc0CurEnd - vSrc0CurStart) * cRowStrideB;
          vExtraTcmBuffer = reinterpret_cast<void *>(reinterpret_cast<uintptr_t>(vTcmBuffer) +
                                                     (vSrc0CurEnd - vSrc0CurStart) * cRowStrideB);
        }

        int vIr1 = vSrc1CurStart;

        if (vExtraTcmBufferSize >= cNbw1 && vExtraTcmBuffer != nullptr) {
          int64_t vQuantATileSize = vExtraTcmBufferSize / cNbw1;
          do {
            vQuantATileSize = MIN(vQuantATileSize, vSrc1CurEnd - vIr1);

            uint8_t *vQuantATileBuffer = reinterpret_cast<uint8_t *>(vExtraTcmBuffer);

            int vIir1 = vIr1;
            for (; vIir1 < (vIr1 + vQuantATileSize); ++vIir1) {
              mmid_row_mapping row_mapping = MMID_MATRIX_ROW(cCurA, vIir1);

              const int cId = row_mapping.i1; // selected expert index

              const int64_t cI11 = cId % ne11;
              const int64_t cI12 = row_mapping.vI2; // row index in src1

              auto *vSrc1Col = vQuantABuffer + (cI11 * cNbw1 + cI12 * cNbw2);
              spacemit_kernels::rvv::memcpy1d(vQuantATileBuffer, vSrc1Col, cNbw1);
              vQuantATileBuffer = vQuantATileBuffer + cNbw1;
            }

            vQuantATileBuffer = reinterpret_cast<uint8_t *>(vExtraTcmBuffer);
            vIir1 = vIr1;

            if (vMoeGemmKernelM2 != nullptr) {
              for (; vIir1 < (vIr1 + vQuantATileSize - 1); vIir1 += 2, vQuantATileBuffer += 2 * cNbw1) {
                mmid_row_mapping row_mapping_0 = MMID_MATRIX_ROW(cCurA, vIir1);
                mmid_row_mapping vRowMapping1 = MMID_MATRIX_ROW(cCurA, vIir1 + 1);

                aSrcWorkspaces[0] = vQuantATileBuffer;
                aSrcWorkspaces[1] = vQuantATileBuffer + cNbw1;

                aDstWorkspaces[0] =
                    (float *)((char *)dst->data + (row_mapping_0.i1 * nb1 + row_mapping_0.vI2 * nb2)) + vSrc0CurStart;
                aDstWorkspaces[1] =
                    (float *)((char *)dst->data + ((vRowMapping1.i1) * nb1 + (vRowMapping1.vI2) * nb2)) + vSrc0CurStart;
                vMoeGemmKernelM2(cBBlkLen, aSrcWorkspaces.data(), vSrc0Cur, vBColZp, aDstWorkspaces.data(), 1,
                                 vSrc0CurEnd - vSrc0CurStart, cBKBlks, ne01);
              }
            }

            for (; vIir1 < (vIr1 + vQuantATileSize); vIir1++, vQuantATileBuffer += cNbw1) {
              mmid_row_mapping row_mapping_0 = MMID_MATRIX_ROW(cCurA, vIir1);

              vGemmKernel(cBBlkLen, vQuantATileBuffer, vSrc0Cur, vBColZp,
                          (float *)((char *)dst->data + (row_mapping_0.i1 * nb1 + row_mapping_0.vI2 * nb2)) +
                              vSrc0CurStart,
                          1, vSrc0CurEnd - vSrc0CurStart, cBKBlks, ne01);
            }

            vIr1 += vQuantATileSize;
          } while (vIr1 < vSrc1CurEnd);
        } else {
          if (vMoeGemmKernelM2 != nullptr) {
            for (; vIr1 < vSrc1CurEnd - 1; vIr1 += 2) {
              for (int vIir1 = 0; vIir1 < 2; ++vIir1) {
                mmid_row_mapping row_mapping = MMID_MATRIX_ROW(cCurA, vIr1 + vIir1);

                const int cId = row_mapping.i1; // selected expert index

                const int64_t cI11 = cId % ne11;
                const int64_t cI12 = row_mapping.vI2; // row index in src1

                const int64_t i1 = cId;   // selected expert index
                const int64_t cI2 = cI12; // row

                aSrcWorkspaces[vIir1] = vQuantABuffer + (cI11 * cNbw1 + cI12 * cNbw2);

                aDstWorkspaces[vIir1] = (float *)((char *)dst->data + (i1 * nb1 + cI2 * nb2)) + vSrc0CurStart;
              }

              vMoeGemmKernelM2(cBBlkLen, aSrcWorkspaces.data(), vSrc0Cur, vBColZp, aDstWorkspaces.data(), 1,
                               vSrc0CurEnd - vSrc0CurStart, cBKBlks, ne01);
            }
          }

          for (; vIr1 < vSrc1CurEnd; vIr1++) {
            mmid_row_mapping row_mapping = MMID_MATRIX_ROW(cCurA, vIr1);

            const int cId = row_mapping.i1; // selected expert index

            const int64_t cI11 = cId % ne11;
            const int64_t cI12 = row_mapping.vI2; // row index in src1

            const int64_t i1 = cId;   // selected expert index
            const int64_t cI2 = cI12; // row

            auto *vSrc1Col = vQuantABuffer + (cI11 * cNbw1 + cI12 * cNbw2);

            vGemmKernel(cBBlkLen, vSrc1Col, vSrc0Cur, vBColZp,
                        (float *)((char *)dst->data + (i1 * nb1 + cI2 * nb2)) + vSrc0CurStart, 1,
                        vSrc0CurEnd - vSrc0CurStart, cBKBlks, ne01);
          }
        }
      }
    }
#undef MMID_MATRIX_ROW
  }

  int mRepack(ggml_tensor *pT, const void *pData, size_t pDataSize) override {
    GGML_LOG_DEBUG("%s: repack tensor %s with %s_%dx%d\n", __func__, pT->name, ggml_type_name(pT->type), (int)NB_COLS,
                   (int)INTER_SIZE);
    return ggml::cpu::riscv64_spacemit::repack<BLOC_TYPE, INTER_SIZE, NB_COLS>(pT, pData, pDataSize);
  }
};

class tensor_traits_common : public tensor_traits_base {
  bool work_size(int pNThreads, const ggml_tensor *pOp, size_t &pSize) override {
    switch (pOp->op) {
    case GGML_OP_FLASH_ATTN_EXT: {
      const int cNTasks = pNThreads;
      const int64_t cNeq2 = pOp->src[0]->ne[2]; // number of query heads
      const int64_t cDk = pOp->src[1]->ne[0];
      const int64_t cDv = pOp->src[2]->ne[0]; // DV

      // Tiled flash attention scratch (tile sizes defined in common.h)
      // Per-thread: Q_q + KQ + mask + VKQ32 + V32 + K_f32 + padding
      size_t vPrefill = sizeof(float) *
                        (GGML_FA_TILE_Q * cDk + 2 * GGML_FA_TILE_Q * GGML_FA_TILE_KV + GGML_FA_TILE_Q * cDv +
                         GGML_FA_TILE_KV * cDv + GGML_FA_TILE_KV * cDk) *
                        cNTasks;

      // Decode path: n_kv_chunks = n_tasks (one chunk per thread)
      // Per-thread: VKQ accmulator (DV), partial M, partial S + intra-thread scratch for V, Q and VKQ
      size_t vNChunks = cNTasks;
      size_t vDecode = sizeof(float) * (cNeq2 * vNChunks * (2 + cDv) + cNTasks * (cDk + 2 * cDv));

      pSize = MAX(vPrefill, vDecode);
    }
      return true;
    default:
      break;
    }
    return false;
  }

  bool compute_forward(ggml_compute_params *pParams, ggml_tensor *pOp) override {
    switch (pOp->op) {
    case GGML_OP_NORM:
      switch (pOp->src[0]->type) {
      case GGML_TYPE_F32:
        spacemit_kernels::rvv::forward_norm_f32(pParams, pOp);
        return true;
      default:
        GGML_ABORT("fatal error");
      }
    case GGML_OP_RMS_NORM:
      switch (pOp->src[0]->type) {
      case GGML_TYPE_F32:
        spacemit_kernels::rvv::forward_rms_norm_f32(pParams, pOp);
        return true;
      default:
        GGML_ABORT("fatal error");
      }
    case GGML_OP_ADD:
      switch (pOp->src[0]->type) {
      case GGML_TYPE_F32:
        spacemit_kernels::rvv::forward_binary<GGML_OP_ADD, float>(pParams, pOp);
        return true;
      case GGML_TYPE_F16:
        spacemit_kernels::rvv::forward_binary<GGML_OP_ADD, _Float16>(pParams, pOp);
        return true;
      default:
        ggml_compute_forward_add(pParams, pOp);
        return true;
      }
    case GGML_OP_SUB:
      switch (pOp->src[0]->type) {
      case GGML_TYPE_F32:
        spacemit_kernels::rvv::forward_binary<GGML_OP_SUB, float>(pParams, pOp);
        return true;
      case GGML_TYPE_F16:
        spacemit_kernels::rvv::forward_binary<GGML_OP_SUB, _Float16>(pParams, pOp);
        return true;
      default:
        ggml_compute_forward_sub(pParams, pOp);
        return true;
      }
    case GGML_OP_MUL:
      switch (pOp->src[0]->type) {
      case GGML_TYPE_F32:
        spacemit_kernels::rvv::forward_binary<GGML_OP_MUL, float>(pParams, pOp);
        return true;
      case GGML_TYPE_F16:
        spacemit_kernels::rvv::forward_binary<GGML_OP_MUL, _Float16>(pParams, pOp);
        return true;
      default:
        ggml_compute_forward_mul(pParams, pOp);
        return true;
      }
    case GGML_OP_DIV:
      switch (pOp->src[0]->type) {
      case GGML_TYPE_F32:
        spacemit_kernels::rvv::forward_binary<GGML_OP_DIV, float>(pParams, pOp);
        return true;
      case GGML_TYPE_F16:
        spacemit_kernels::rvv::forward_binary<GGML_OP_DIV, _Float16>(pParams, pOp);
        return true;
      default:
        ggml_compute_forward_div(pParams, pOp);
        return true;
      }
    case GGML_OP_FLASH_ATTN_EXT:
      mForwardFlashAttnExtF16(pParams, pOp);
      return true;
    case GGML_OP_CONT: {
      const ggml_tensor *vSrc0 = pOp->src[0];
      if (pOp->type == vSrc0->type && pOp->nb[0] != vSrc0->nb[0] && pOp->nb[0] == vSrc0->nb[1] &&
          pOp->ne[3] * pOp->ne[2] * pOp->nb[2] == vSrc0->ne[3] * vSrc0->ne[2] * vSrc0->nb[2]) {
        spacemit_kernels::rvv::forward_cont_with_permute(pParams, pOp);
      } else {
        ggml_compute_forward_cont(pParams, pOp);
      }
      return true;
    }
    case GGML_OP_CPY: {
      const ggml_tensor *vSrc0 = pOp->src[0];
      if (pOp->type == vSrc0->type && pOp->nb[0] == vSrc0->nb[1] && vSrc0->nb[0] != vSrc0->nb[1] &&
          ggml_nelements(vSrc0) == ggml_nelements(pOp)) {
        spacemit_kernels::rvv::forward_cpy_with_permute(pParams, pOp);
      } else {
        ggml_compute_forward_cpy(pParams, pOp);
      }
      return true;
    }
    case GGML_OP_REPEAT: {
      const bool cRowsEqual = ggml_nrows(pOp->src[0]) == ggml_nrows(pOp);
      const bool cBroadcastOrEqual = pOp->src[0]->ne[0] == 1 || pOp->src[0]->ne[0] == pOp->ne[0];

      if (cRowsEqual && cBroadcastOrEqual) {
        switch (pOp->src[0]->type) {
        case GGML_TYPE_F32:
          spacemit_kernels::rvv::forward_repeat_nrows<int32_t>(pParams, pOp);
          return true;
        case GGML_TYPE_F16:
          spacemit_kernels::rvv::forward_repeat_nrows<int16_t>(pParams, pOp);
          return true;
        default:
          break;
        }
      }

      if (pOp->src[0]->ne[1] == 1 && pOp->src[0]->ne[0] == pOp->ne[0]) {
        switch (pOp->src[0]->type) {
        case GGML_TYPE_F32:
          spacemit_kernels::rvv::forward_repeat_dim1<int32_t>(pParams, pOp);
          return true;
        case GGML_TYPE_F16:
          spacemit_kernels::rvv::forward_repeat_dim1<int16_t>(pParams, pOp);
          return true;
        default:
          break;
        }
      }

      ggml_compute_forward_repeat(pParams, pOp);
    }
      return true;
    case GGML_OP_SUM_ROWS: {
      if (pOp->src[0]->type == GGML_TYPE_F32 && pOp->type == GGML_TYPE_F32) {
        spacemit_kernels::rvv::forward_sum_rows<float>(pParams, pOp);
      } else {
        ggml_compute_forward_sum_rows(pParams, pOp);
      }
    }
      return true;
    case GGML_OP_GET_ROWS: {
      if (pOp->src[0]->type == pOp->type) {
        switch (pOp->src[0]->type) {
        case GGML_TYPE_F32:
          spacemit_kernels::rvv::forward_get_rows<int32_t>(pParams, pOp);
          return true;
        case GGML_TYPE_F16:
          spacemit_kernels::rvv::forward_get_rows<int16_t>(pParams, pOp);
          return true;
        default:
          break;
        }
      }

      ggml_compute_forward_get_rows(pParams, pOp);
    }
      return true;
    case GGML_OP_CONCAT: {
      const int32_t cDim = ggml_get_op_params_i32(pOp, 0);
      if (cDim == 0 && pOp->type == pOp->src[0]->type) {
        switch (pOp->src[0]->type) {
        case GGML_TYPE_F32:
          spacemit_kernels::rvv::forward_concat<int32_t>(pParams, pOp);
          return true;
        case GGML_TYPE_F16:
          spacemit_kernels::rvv::forward_concat<int16_t>(pParams, pOp);
          return true;
        default:
          break;
        }
      }

      ggml_compute_forward_concat(pParams, pOp);
    }
      return true;
    // TODO For GGML_OP_GATED_DELTA_NET
    // case GGML_OP_GATED_DELTA_NET:
    //     return true;
    default:
      break;
    }
    return false;
  }

  void mForwardFlashAttnExtF16(const ggml_compute_params *pParams, ggml_tensor *pDst) {
    const ggml_tensor *vQ = pDst->src[0];
    const ggml_tensor *vK = pDst->src[1];
    const ggml_tensor *v_ = pDst->src[2];

    GGML_TENSOR_LOCALS(int64_t, neq, vQ, ne)
    GGML_TENSOR_LOCALS(size_t, nbq, vQ, nb)
    GGML_TENSOR_LOCALS(int64_t, nek, vK, ne)
    GGML_TENSOR_LOCALS(size_t, nbk, vK, nb)
    GGML_TENSOR_LOCALS(int64_t, nev, v_, ne)
    GGML_TENSOR_LOCALS(size_t, nbv, v_, nb)
    GGML_TENSOR_LOCALS(int64_t, ne, pDst, ne)
    GGML_TENSOR_LOCALS(size_t, nb, pDst, nb)

    const int64_t cDk = nek0;
    const int64_t cDv = nev0;

    const bool cSupportedPrec = (pDst->op_params[3] == GGML_PREC_F32 || pDst->op_params[3] == GGML_PREC_DEFAULT);
    const bool cSupportedTypes = (vQ->type == GGML_TYPE_F32 && vK->type == GGML_TYPE_F16 && v_->type == GGML_TYPE_F16);
    const bool cSupportedShape = (cDk > 0 && cDk <= 128 && cDv > 0 && cDv <= 128);
    const bool cSupportedVlen = (__riscv_vlenb() == 128);

    if (!(cSupportedPrec && cSupportedTypes && cSupportedShape && cSupportedVlen)) {
      ggml_compute_forward_flash_attn_ext(pParams, pDst);
      return;
    }

    // total rows in q
    const int64_t cNr = neq1 * neq2 * neq3;

    // rows per thread
    const int cIth = pParams->ith;
    const int cNth = pParams->nth;

    static constexpr int64_t cQTileSz = ggml_fa_tile_config::Q;
    const bool cUseTiled = !pParams->use_ref && (neq1 >= cQTileSz);

    // 4x chunks per thread
    // int     nth_scaled = nth * 4;
    // int64_t chunk_size = (nr + nth_scaled - 1) / nth_scaled;
    // int64_t nchunk     = (nr + chunk_size - 1) / chunk_size;

    // if (nth == 1 || nchunk < nth) {
    //     nchunk = nth;
    // }

    int64_t vNchunk = cNth;

    if (cIth == 0) {
      // Every thread starts at ith, so the first unprocessed chunk is nth.  This save a bit of coordination right at
      // the start.
      ggml_threadpool_chunk_set(pParams->threadpool, cNth);
    }

    ggml_barrier(pParams->threadpool);

    // The number of elements in each chunk
    const int64_t cDr = (cNr + vNchunk - 1) / vNchunk;

    // The first chunk comes from our thread_id, the rest will get auto-assigned.
    int vCurrentChunk = cIth;

    while (vCurrentChunk < vNchunk) {
      const int64_t cIr0 = cDr * vCurrentChunk;
      const int64_t cIr1 = MIN(cIr0 + cDr, cNr);

      if (cUseTiled) {
        spacemit_kernels::rvv::forward_flash_attn_ext_f16_tiled_vlen1024_vf16(
            pParams, pDst, cIr0, cIr1, ggml::cpu::riscv64_spacemit::tls_context.vTcmBuffer,
            ggml::cpu::riscv64_spacemit::tls_context.vTcmBufferSize);
      } else {
        spacemit_kernels::rvv::forward_flash_attn_ext_f16_one_chunk_vlen1024_vf16(
            pParams, pDst, cIr0, cIr1, ggml::cpu::riscv64_spacemit::tls_context.vTcmBuffer,
            ggml::cpu::riscv64_spacemit::tls_context.vTcmBufferSize);
      }

      vCurrentChunk = ggml_threadpool_chunk_add(pParams->threadpool, 1);
    }
  }

  int mRepack(ggml_tensor *pT, const void *pData, size_t pDataSize) override {
    memcpy(pT->data, pData, pDataSize);
    return 0;
  }
};

// Impl By IME1
static const tensor_traits<block_q4_0, 32, 16> q4_0_16x32_q8_0;
static const tensor_traits<block_q4_1, 32, 16> q4_1_16x32_q8_0;
static const tensor_traits<block_q4_K, 32, 16> q4_k_16x32_q8_0;
// Impl By IME2
static const tensor_traits<block_q2_K, 256, 32> q2_k_32x256_q8_0;
static const tensor_traits<block_q3_K, 256, 32> q3_k_32x256_q8_0;
static const tensor_traits<block_q4_0, 32, 32> q4_0_32x32_q8_0;
static const tensor_traits<block_q4_1, 32, 32> q4_1_32x32_q8_0;
static const tensor_traits<block_q4_0, 256, 32> q4_0_32x256_q8_0;
static const tensor_traits<block_q4_1, 256, 32> q4_1_32x256_q8_0;
static const tensor_traits<block_q4_K, 32, 32> q4_k_32x32_q8_0;
static const tensor_traits<block_q6_K, 32, 32> q6_k_32x32_q8_0;
static const tensor_traits<block_q8_0, 32, 32> q8_0_32x32_q8_0;
static const tensor_traits<block_mxfp4, 32, 32> mxfp4_32x32_q8_0;
static const tensor_traits<block_q5_K, 32, 32> q5_k_32x32_q8_0;
static const tensor_traits<block_q5_1, 32, 32> q5_1_32x32_q8_0;
static const tensor_traits<block_q5_0, 32, 32> q5_0_32x32_q8_0;
// Impl By RVV
static const tensor_traits_common rvv_impl;

} // namespace ggml::cpu::riscv64_spacemit

static const ggml::cpu::tensor_traits *fGgmlRiscv64SpacemitGetOptimalRepackType(const ggml_tensor *pCur) {
  switch (pCur->type) {
  case GGML_TYPE_Q2_K: {
#if defined(RISCV64_SPACEMIT_IME2)
    if (pCur->ne[1] % 32 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
      return &ggml::cpu::riscv64_spacemit::q2_k_32x256_q8_0;
    }
#endif
  } break;
  case GGML_TYPE_Q3_K: {
#if defined(RISCV64_SPACEMIT_IME2)
    if (pCur->ne[1] % 32 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
      return &ggml::cpu::riscv64_spacemit::q3_k_32x256_q8_0;
    }
#endif
  } break;
  case GGML_TYPE_Q4_0: {
#if defined(RISCV64_SPACEMIT_IME2)
    if (pCur->ne[1] % 32 == 0 && pCur->ne[0] % 256 == 0 &&
        (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
      return &ggml::cpu::riscv64_spacemit::q4_0_32x256_q8_0;
    }

    if (pCur->ne[1] % 32 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
      return &ggml::cpu::riscv64_spacemit::q4_0_32x32_q8_0;
    }
#endif

#if defined(RISCV64_SPACEMIT_IME1)
    if (pCur->ne[1] % 16 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime1)) {
      return &ggml::cpu::riscv64_spacemit::q4_0_16x32_q8_0;
    }
#endif
  } break;
  case GGML_TYPE_Q4_1: {
#if defined(RISCV64_SPACEMIT_IME2)
    // TODO
    // if (cur->ne[1] % 32 == 0 && cur->ne[0] % 256 == 0 &&
    //     (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
    //     return &ggml::cpu::riscv64_spacemit::q4_1_32x256_q8_0;
    // }

    if (pCur->ne[1] % 32 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
      return &ggml::cpu::riscv64_spacemit::q4_1_32x32_q8_0;
    }
#endif

#if defined(RISCV64_SPACEMIT_IME1)
    if (pCur->ne[1] % 16 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime1)) {
      return &ggml::cpu::riscv64_spacemit::q4_1_16x32_q8_0;
    }
#endif
  } break;
  case GGML_TYPE_Q4_K: {
#if defined(RISCV64_SPACEMIT_IME2)
    if (pCur->ne[1] % 32 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
      return &ggml::cpu::riscv64_spacemit::q4_k_32x32_q8_0;
    }
#endif

#if defined(RISCV64_SPACEMIT_IME1)
    if (pCur->ne[1] % 16 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime1)) {
      return &ggml::cpu::riscv64_spacemit::q4_k_16x32_q8_0;
    }
#endif
  } break;
  case GGML_TYPE_Q6_K: {
#if defined(RISCV64_SPACEMIT_IME2)
    if ((ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
      return &ggml::cpu::riscv64_spacemit::q6_k_32x32_q8_0;
    }
#endif
  } break;
  case GGML_TYPE_Q8_0: {
#if defined(RISCV64_SPACEMIT_IME2)
    if ((ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
      return &ggml::cpu::riscv64_spacemit::q8_0_32x32_q8_0;
    }
#endif
  } break;
  case GGML_TYPE_MXFP4: {
#if defined(RISCV64_SPACEMIT_IME2)
    // TODO
    // if (cur->ne[1] % 32 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
    //     return &ggml::cpu::riscv64_spacemit::mxfp4_32x32_q8_0;
    // }
#endif
  } break;
  case GGML_TYPE_Q5_K: {
#if defined(RISCV64_SPACEMIT_IME2)
    if (pCur->ne[1] % 32 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
      return &ggml::cpu::riscv64_spacemit::q5_k_32x32_q8_0;
    }
#endif
  } break;
  case GGML_TYPE_Q5_1: {
#if defined(RISCV64_SPACEMIT_IME2)
    if (pCur->ne[1] % 32 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
      return &ggml::cpu::riscv64_spacemit::q5_1_32x32_q8_0;
    }
#endif
  } break;
  case GGML_TYPE_Q5_0: {
#if defined(RISCV64_SPACEMIT_IME2)
    if (pCur->ne[1] % 32 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
      return &ggml::cpu::riscv64_spacemit::q5_0_32x32_q8_0;
    }
#endif
  } break;
  default:
    break;
  }

  return nullptr;
}

static enum ggml_status fGgmlBackendRiscv64SpacemitBufferInitTensor(ggml_backend_buffer_t pBuffer,
                                                                    ggml_tensor *pTensor) {
  pTensor->extra = (void *)const_cast<ggml::cpu::tensor_traits *>(fGgmlRiscv64SpacemitGetOptimalRepackType(pTensor));

  GGML_UNUSED(pBuffer);

  return GGML_STATUS_SUCCESS;
}

static void fGgmlBackendRiscv64SpacemitBufferFreeBuffer(ggml_backend_buffer_t pBuffer) {
  GGML_ASSERT(pBuffer);

  void *vBase = pBuffer->context;
  if (vBase == nullptr) {
    return;
  }

  ggml::cpu::riscv64_spacemit::spine_mem_pool_free(vBase);
}

static void *fGgmlBackendRiscv64SpacemitBufferGetBase(ggml_backend_buffer_t pBuffer) {
  GGML_ASSERT(pBuffer);

  void *vBase = pBuffer->context;
  GGML_ASSERT(vBase != nullptr);
  return vBase;
}

static void fGgmlBackendRiscv64SpacemitBufferMemsetTensor(ggml_backend_buffer_t pBuffer, ggml_tensor *pTensor,
                                                          uint8_t pValue, size_t pOffset, size_t pSize) {
  GGML_ASSERT(pTensor);
  memset((char *)pTensor->data + pOffset, pValue, pSize);

  GGML_UNUSED(pBuffer);
}

static void fGgmlBackendRiscv64SpacemitBufferClear(ggml_backend_buffer_t pBuffer, uint8_t pValue) {
  GGML_ASSERT(pBuffer);

  void *vBase = pBuffer->context;
  GGML_ASSERT(vBase != nullptr);
  memset(vBase, pValue, pBuffer->size);
}

static void fGgmlBackendRiscv64SpacemitBufferSetTensor(ggml_backend_buffer_t pBuffer, ggml_tensor *pTensor,
                                                       const void *pData, size_t pOffset, size_t pSize) {
  GGML_ASSERT(pOffset == 0);
  GGML_ASSERT(pSize == ggml_nbytes(pTensor));

  auto vTensorTraits = (ggml::cpu::riscv64_spacemit::tensor_traits_base *)pTensor->extra;
  if (vTensorTraits) {
    auto vOk = vTensorTraits->mRepack(pTensor, pData, pSize);
    GGML_ASSERT(vOk == 0);
  }

  GGML_UNUSED(pBuffer);
}

static const ggml_backend_buffer_i ggml_backend_riscv64_spacemit_buffer_i = {
    /* .free_buffer     = */ fGgmlBackendRiscv64SpacemitBufferFreeBuffer,
    /* .get_base        = */ fGgmlBackendRiscv64SpacemitBufferGetBase,
    /* .init_tensor     = */ fGgmlBackendRiscv64SpacemitBufferInitTensor,
    /* .memset_tensor   = */ fGgmlBackendRiscv64SpacemitBufferMemsetTensor,
    /* .set_tensor      = */ fGgmlBackendRiscv64SpacemitBufferSetTensor,
    /* .get_tensor      = */ nullptr,
    /* .set_tensor_2d   = */ nullptr,
    /* .get_tensor_2d   = */ nullptr,
    /* .cpy_tensor      = */ nullptr,
    /* .clear           = */ fGgmlBackendRiscv64SpacemitBufferClear,
    /* .reset           = */ nullptr,
};

static const char *fGgmlBackendCpuRiscv64SpacemitBufferTypeGetName(ggml_backend_buffer_type_t pBuft) {
  return "CPU_RISCV64_SPACEMIT";

  GGML_UNUSED(pBuft);
}

static ggml_backend_buffer_t fGgmlBackendCpuRiscv64SpacemitBufferTypeAllocBuffer(ggml_backend_buffer_type_t pBuft,
                                                                                 size_t pSize) {
  void *vBase = ggml::cpu::riscv64_spacemit::spine_mem_pool_alloc(pSize, 64);
  if (vBase == nullptr) {
    return nullptr;
  }

  return ggml_backend_buffer_init(pBuft, ggml_backend_riscv64_spacemit_buffer_i, vBase, pSize);
}

static size_t fGgmlBackendCpuRiscv64SpacemitBufferTypeGetAlignment(ggml_backend_buffer_type_t pBuft) {
  return 64;

  GGML_UNUSED(pBuft);
}

static size_t fGgmlBackendCpuRiscv64SpacemitNbytes(ggml_backend_buffer_type_t pBuft, const ggml_tensor *pTensor) {
  for (int vI = 0; vI < GGML_MAX_DIMS; ++vI) {
    if (pTensor->ne[vI] <= 0) {
      return 0;
    }
  }

  GGML_UNUSED(pBuft);

  const auto fPlainNbytes = [&]() {
    size_t vTotal = ggml_type_size(pTensor->type);
    for (int vI = 0; vI < GGML_MAX_DIMS; ++vI) {
      vTotal += (pTensor->ne[vI] - 1) * pTensor->nb[vI];
    }
    return vTotal;
  };

  const size_t cBlckSize = ggml_blck_size(pTensor->type);
  if (cBlckSize == 1) {
    return fPlainNbytes();
  }

  const size_t cRowNbytes = pTensor->ne[0] * pTensor->nb[0] / cBlckSize;

  const auto fAddStridedNbytes = [&](size_t pTotal, size_t pSrcBlockSize, size_t pDstBlockSize) {
    for (int vI = 1; vI < GGML_MAX_DIMS; ++vI) {
      pTotal += (pTensor->ne[vI] - 1) * (pTensor->nb[vI] / pSrcBlockSize) * pDstBlockSize;
    }
    return pTotal;
  };

  const auto fRemapBlockNbytes = [&](size_t pSrcBlockSize, size_t pDstBlockSize, int64_t pPaddedRows = 0) {
    GGML_ASSERT(cRowNbytes % pSrcBlockSize == 0);

    size_t vTotal = fAddStridedNbytes((cRowNbytes / pSrcBlockSize) * pDstBlockSize, pSrcBlockSize, pDstBlockSize);

    if (pPaddedRows > 0 && pTensor->ne[1] % pPaddedRows != 0) {
      vTotal += (pPaddedRows - pTensor->ne[1] % pPaddedRows) * (pTensor->nb[1] / pSrcBlockSize) * pDstBlockSize;
    }

    return vTotal;
  };

  size_t vNbytes = cRowNbytes;
  switch (pTensor->type) {
  case GGML_TYPE_Q4_K:
    vNbytes = fRemapBlockNbytes(sizeof(block_q4_K), sizeof(block_q4_1) * 8);
    break;
  case GGML_TYPE_Q6_K:
    vNbytes = fRemapBlockNbytes(sizeof(block_q6_K), sizeof(block_q8_0) * 8, 32);
    break;
  case GGML_TYPE_Q8_0:
    vNbytes = fRemapBlockNbytes(sizeof(block_q8_0), sizeof(block_q8_0), 32);
    break;
  case GGML_TYPE_Q2_K:
    vNbytes = fRemapBlockNbytes(sizeof(block_q2_K), sizeof(spacemit_kernels::nrow_block_q2_k<1>));
    break;
  case GGML_TYPE_Q3_K:
    vNbytes = fRemapBlockNbytes(sizeof(block_q3_K), sizeof(spacemit_kernels::nrow_block_q3_k<1>));
    break;
  case GGML_TYPE_MXFP4:
    vNbytes = fRemapBlockNbytes(sizeof(block_mxfp4), sizeof(spacemit_kernels::nrow_block_mxfp4<1>));
    break;
  case GGML_TYPE_Q5_K:
    vNbytes = fRemapBlockNbytes(sizeof(block_q5_K), sizeof(spacemit_kernels::nrow_block_q5_1<1>) * 8);
    break;
  case GGML_TYPE_Q5_1:
    vNbytes = fRemapBlockNbytes(sizeof(block_q5_1), sizeof(spacemit_kernels::nrow_block_q5_1<1>));
    break;
  case GGML_TYPE_Q5_0:
    vNbytes = fRemapBlockNbytes(sizeof(block_q5_0), sizeof(spacemit_kernels::nrow_block_q5_0<1>));
    break;
  default:
    vNbytes = fAddStridedNbytes(cRowNbytes, 1, 1);
    break;
  }

  return vNbytes;
}

namespace ggml::cpu::riscv64_spacemit {

class extra_buffer_type : ggml::cpu::extra_buffer_type {
  bool supports_op(ggml_backend_dev_t, const ggml_tensor *pOp) override {
    switch (pOp->op) {
    case GGML_OP_MUL_MAT:
      if (pOp->src[0]->buffer && (ggml_n_dims(pOp->src[0]) == 2) &&
          pOp->src[0]->buffer->buft == ggml_backend_cpu_riscv64_spacemit_buffer_type() &&
          fGgmlRiscv64SpacemitGetOptimalRepackType(pOp->src[0])) {
        if (pOp->src[1]->buffer && !ggml_backend_buft_is_host(pOp->src[1]->buffer->buft)) {
          return false;
        }
        if (pOp->src[1]->type == GGML_TYPE_F32) {
          return true;
        }
      }
      break;
    case GGML_OP_MUL_MAT_ID:
      if (pOp->src[0]->buffer && (ggml_n_dims(pOp->src[0]) == 3) &&
          pOp->src[0]->buffer->buft == ggml_backend_cpu_riscv64_spacemit_buffer_type() &&
          fGgmlRiscv64SpacemitGetOptimalRepackType(pOp->src[0])) {
        if (pOp->src[1]->buffer && !ggml_backend_buft_is_host(pOp->src[1]->buffer->buft)) {
          return false;
        }
        if (pOp->src[1]->type == GGML_TYPE_F32) {
          return true;
        }
      }
      break;
    default:
      // GGML_ABORT("fatal error");
      break;
    }
    return false;
  }

  ggml::cpu::tensor_traits *get_tensor_traits(const ggml_tensor *pOp) override {
    switch (pOp->op) {
    case GGML_OP_MUL_MAT:
    case GGML_OP_MUL_MAT_ID:
      if (pOp->src[0]->buffer && pOp->src[0]->buffer->buft == ggml_backend_cpu_riscv64_spacemit_buffer_type()) {
        return (ggml::cpu::tensor_traits *)pOp->src[0]->extra;
      }
      break;
    case GGML_OP_NORM:
    case GGML_OP_RMS_NORM:
    case GGML_OP_ADD:
    case GGML_OP_SUB:
    case GGML_OP_MUL:
    case GGML_OP_DIV:
    case GGML_OP_FLASH_ATTN_EXT:
    case GGML_OP_CONT:
    case GGML_OP_CPY:
    case GGML_OP_REPEAT:
    case GGML_OP_SUM_ROWS:
    case GGML_OP_GET_ROWS:
    case GGML_OP_CONCAT:
      // case GGML_OP_GATED_DELTA_NET:
      return (ggml::cpu::tensor_traits *)(&ggml::cpu::riscv64_spacemit::rvv_impl);
    default:
      // GGML_ABORT("fatal error");
      break;
    }

    return nullptr;
  }
};

} // namespace ggml::cpu::riscv64_spacemit

ggml_backend_buffer_type_t ggml_backend_cpu_riscv64_spacemit_buffer_type(void) {
  static ggml_backend_buffer_type vGgmlBackendCpuBufferTypeRiscv64Spacemit = {
      /* .iface    = */
      {
          /* .get_name         = */ fGgmlBackendCpuRiscv64SpacemitBufferTypeGetName,
          /* .alloc_buffer     = */ fGgmlBackendCpuRiscv64SpacemitBufferTypeAllocBuffer,
          /* .get_alignment    = */ fGgmlBackendCpuRiscv64SpacemitBufferTypeGetAlignment,
          /* .get_max_size     = */ nullptr,
          /* .get_alloc_size   = */ fGgmlBackendCpuRiscv64SpacemitNbytes,
          /* .is_host          = */ nullptr,
      },
      /* .device  = */
      ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0),
      /* .context = */
      new ggml::cpu::riscv64_spacemit::extra_buffer_type(),
  };

  return &vGgmlBackendCpuBufferTypeRiscv64Spacemit;
}

extern "C" {
static int fBindAiThread() {
  int vFd, vBytes;
  char aStr[32];

  vFd = open("/proc/set_ai_thread", O_WRONLY);
  if (vFd < 0) {
    GGML_LOG_ERROR("try open /proc/set_ai_thread failed\n");
    return -1;
  }

  snprintf(aStr, 16, "%d", 0);
  vBytes = write(vFd, aStr, strlen(aStr));
  if (vBytes < 0) {
    GGML_LOG_ERROR("try write /proc/set_ai_thread failed\n");
    close(vFd);
    return -1;
  }

  close(vFd);
  return 0;
}

void ggml_backend_cpu_riscv64_spacemit_set_numa_thread_affinity(int pThreadN) {
  int vCpuId = sched_getcpu();
  if (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2 &&
      !((1 << vCpuId) & ggml::cpu::riscv64_spacemit::global_spine_env_info.cpu_mask)) {
    GGML_PRINT_DEBUG("bind_ai_thread for thread %d, pid %d\n", pThreadN, getpid());
    fBindAiThread();
  }

  if (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_tcm &&
      ggml::cpu::riscv64_spacemit::tls_context.vCpuId == -1) {
    CPU_ZERO(&(ggml::cpu::riscv64_spacemit::tls_context.vCpuset));
    pthread_t vMainThread = pthread_self();
    const auto &vPerferCoreIds = ggml::cpu::riscv64_spacemit::global_spine_env_info.perfer_core_ids;
    if (pThreadN < 0 || static_cast<size_t>(pThreadN) >= vPerferCoreIds.size()) {
      GGML_ABORT("thread_n %d exceeds perfer_core_ids size %zu\n", pThreadN, vPerferCoreIds.size());
    }
    auto vPerferCpuId = vPerferCoreIds[static_cast<size_t>(pThreadN)];
    CPU_SET(vPerferCpuId, &(ggml::cpu::riscv64_spacemit::tls_context.vCpuset));
    int vS =
        pthread_setaffinity_np(vMainThread, sizeof(cpu_set_t), &(ggml::cpu::riscv64_spacemit::tls_context.vCpuset));
    if (vS != 0) {
      GGML_ABORT("set thread affinity error for thread_n %d, cpu_id %d\n", pThreadN, vPerferCpuId);
    }

    int vAiCpuId = vPerferCpuId - ggml::cpu::riscv64_spacemit::global_spine_env_info.aicpu_id_offset;
    ggml::cpu::riscv64_spacemit::tls_context.vCpuId = vAiCpuId;
    ggml::cpu::riscv64_spacemit::tls_context.vTcmBuffer =
        ggml::cpu::riscv64_spacemit::spine_mem_pool_tcm_mem_get(vAiCpuId);
    ggml::cpu::riscv64_spacemit::tls_context.vTcmBufferSize =
        ggml::cpu::riscv64_spacemit::global_spine_env_info.tcm_blk_size;
  }

  if (ggml::cpu::riscv64_spacemit::tls_context.vTcmBuffer != nullptr) {
    void *vRt =
        ggml::cpu::riscv64_spacemit::spine_mem_pool_tcm_mem_wait(ggml::cpu::riscv64_spacemit::tls_context.vCpuId);
    if (vRt == nullptr) {
      GGML_ABORT("wait tcm buffer failed for cpu_id: %d", ggml::cpu::riscv64_spacemit::tls_context.vCpuId);
    }
  }
}

void ggml_backend_cpu_riscv64_spacemit_clear_numa_thread_affinity_threaded(int pThreadN) {
  if (ggml::cpu::riscv64_spacemit::tls_context.vTcmBuffer != nullptr) {
    auto vRt =
        ggml::cpu::riscv64_spacemit::spine_mem_pool_tcm_mem_release(ggml::cpu::riscv64_spacemit::tls_context.vCpuId);
    if (vRt != 0) {
      GGML_ABORT("release tcm buffer failed for cpu_id: %d", ggml::cpu::riscv64_spacemit::tls_context.vCpuId);
    }
  }
}
}
