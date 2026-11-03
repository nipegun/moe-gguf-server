#include "ggml.h"
#include "common.cuh"
#include "unary.cuh"
#include "mmvf.cuh"
#include "convert.cuh"

template <typename T, typename type_acc, int ncols_dst, int block_size, bool has_fusion = false,
          bool is_multi_token_id = false>
static __global__ void
fMulMatVecF(const T *pXPtr, const float *pYPtr, const int32_t *pIdsPtr, const ggml_cuda_mm_fusion_args_device pFusion,
            float *pDstPtr, const int pNcols2, const uint3 pNchannelsY, const int pStrideRow, const int pStrideColY2,
            const int pStrideColDst, const uint3 pChannelRatio, const int pStrideChannelX, const int pStrideChannelY,
            const int pStrideChannelDst, const uint3 pSampleRatio, const int pStrideSampleX, const int pStrideSampleY,
            const int pStrideSampleDst, const int pIdsStride) {
  const T *GGML_CUDA_RESTRICT vX = pXPtr;
  const float *GGML_CUDA_RESTRICT vY = pYPtr;
  const int32_t *GGML_CUDA_RESTRICT vIds = pIdsPtr;
  float *GGML_CUDA_RESTRICT vDst = pDstPtr;
  const int cRow = blockIdx.x;
  // for MUL_MAT_ID - blockIdx.y = n_expert_used, blockIdx.z = ncols_dst (tokens)
  const int cChannelDst = blockIdx.y;
  const int cTid = threadIdx.x;

  int vTokenIdx;
  int vChannelX;
  int vChannelY;
  int vSampleDst;

  ggml_cuda_pdl_sync();
  if constexpr (is_multi_token_id) {
    // Multi-token MUL_MAT_ID path, adding these in the normal path causes a perf regression for n_tokens=1 case
    vTokenIdx = blockIdx.z;
    vChannelX = vIds[cChannelDst + vTokenIdx * pIdsStride];
    vChannelY = fastmodulo(cChannelDst, pNchannelsY);
    vSampleDst = 0;
  } else {
    vTokenIdx = vIds ? blockIdx.z : 0;
    vChannelX = vIds ? vIds[blockIdx.y + vTokenIdx * pIdsStride] : fastdiv((uint32_t)cChannelDst, pChannelRatio);
    vChannelY = vIds ? fastmodulo(blockIdx.y, pNchannelsY) : cChannelDst;
    vSampleDst = vIds ? 0 : blockIdx.z;
  }

  const int cSampleX = fastdiv((uint32_t)vSampleDst, pSampleRatio);
  const int cSampleY = vSampleDst;

  constexpr int cWarpSize = ggml_cuda_get_physical_warp_size();

  vX += int64_t(cSampleX) * pStrideSampleX + vChannelX * pStrideChannelX + cRow * pStrideRow;
  vY += int64_t(cSampleY) * pStrideSampleY + vChannelY * pStrideChannelY;
  vDst += int64_t(vSampleDst) * pStrideSampleDst + cChannelDst * pStrideChannelDst;
  if constexpr (is_multi_token_id) {
    vY += vTokenIdx * pStrideColY2 * 2;
    vDst += vTokenIdx * pStrideColDst;
  }

  bool vUseGate = false;
  bool vUseBias = false;
  bool vUseGateBias = false;
  ggml_glu_op vGluOp = ggml_glu_op::GGML_GLU_OP_SWIGLU;
  float vGluLimit = 0.0f;
  const T *vGateX = nullptr;
  const float *vXBias = nullptr;
  const float *vGateBias = nullptr;

  if constexpr (has_fusion) {
    vUseGate = pFusion.gate != nullptr;
    vUseBias = pFusion.x_bias != nullptr;
    vUseGateBias = pFusion.gate_bias != nullptr;
    vGluOp = pFusion.glu_op;
    vGluLimit = pFusion.glu_limit;

    if (vUseGate) {
      vGateX = static_cast<const T *>(pFusion.gate);
    }
    if (vUseBias) {
      vXBias = static_cast<const float *>(pFusion.x_bias);
    }
    if (vUseGateBias) {
      vGateBias = static_cast<const float *>(pFusion.gate_bias);
      vUseGateBias = vUseGate;
    } else {
      vUseGateBias = false;
    }
  }

  if (vUseGate) {
    vGateX += int64_t(cSampleX) * pStrideSampleX + vChannelX * pStrideChannelX + cRow * pStrideRow;
  }

  if constexpr (has_fusion) {
    const int channel_bias = vIds ? vChannelX : cChannelDst;
    if (vUseBias) {
      vXBias += int64_t(vSampleDst) * pStrideSampleDst + channel_bias * pStrideChannelDst;
    }
    if (vUseGateBias) {
      vGateBias += int64_t(vSampleDst) * pStrideSampleDst + channel_bias * pStrideChannelDst;
    }
  }

  const float2 *vY2 = (const float2 *)vY;

  extern __shared__ char aDataMmv[];
  float *vBufIw = (float *)aDataMmv;
  [[maybe_unused]] float *vBufIwGate = nullptr;
  if constexpr (has_fusion) {
    vBufIwGate = (float *)(aDataMmv + cWarpSize * sizeof(float));
  }

  if (block_size > cWarpSize) {
    if (cTid < cWarpSize) {
      vBufIw[cTid] = 0.0f;
      if constexpr (has_fusion) {
        if (vUseGate) {
          vBufIwGate[cTid] = 0.0f;
        }
      }
    }
    __syncthreads();
  }

  float vSumf[ncols_dst] = {0.0f};
  float vSumfGate[ncols_dst];
  if constexpr (has_fusion) {
#pragma unroll
    for (int j = 0; j < ncols_dst; ++j) {
      vSumfGate[j] = 0.0f;
    }
  }

  if constexpr (std::is_same_v<T, float>) {
    const float2 *x2 = (const float2 *)vX;
    [[maybe_unused]] const float2 *gate_x2 = nullptr;
    if constexpr (has_fusion) {
      if (vUseGate) {
        gate_x2 = (const float2 *)vGateX;
      }
    }

    for (int col2 = cTid; col2 < pNcols2; col2 += block_size) {
      const float2 tmpx = x2[col2];
      float2 tmpx_gate = make_float2(0.0f, 0.0f);
      if constexpr (has_fusion) {
        if (vUseGate) {
          tmpx_gate = gate_x2[col2];
        }
      }

#pragma unroll
      for (int j = 0; j < ncols_dst; ++j) {
        const float2 tmpy = vY2[j * pStrideColY2 + col2];
        ggml_cuda_mad(vSumf[j], tmpx.x, tmpy.x);
        ggml_cuda_mad(vSumf[j], tmpx.y, tmpy.y);

        if constexpr (has_fusion) {
          if (vUseGate) {
            ggml_cuda_mad(vSumfGate[j], tmpx_gate.x, tmpy.x);
            ggml_cuda_mad(vSumfGate[j], tmpx_gate.y, tmpy.y);
          }
        }
      }
    }
  } else if constexpr (std::is_same_v<T, half>) {
    const half2 *x2 = (const half2 *)vX;
    [[maybe_unused]] const half2 *gate_x2 = nullptr;
    if constexpr (has_fusion) {
      if (vUseGate) {
        gate_x2 = (const half2 *)vGateX;
      }
    }

    if (std::is_same_v<type_acc, float>) {
      for (int col2 = cTid; col2 < pNcols2; col2 += block_size) {
        const float2 tmpx = __half22float2(x2[col2]);
        float2 tmpx_gate = make_float2(0.0f, 0.0f);
        if constexpr (has_fusion) {
          if (vUseGate) {
            tmpx_gate = __half22float2(gate_x2[col2]);
          }
        }
#pragma unroll
        for (int j = 0; j < ncols_dst; ++j) {
          const float2 tmpy = vY2[j * pStrideColY2 + col2];
          ggml_cuda_mad(vSumf[j], tmpx.x, tmpy.x);
          ggml_cuda_mad(vSumf[j], tmpx.y, tmpy.y);

          if constexpr (has_fusion) {
            if (vUseGate) {
              ggml_cuda_mad(vSumfGate[j], tmpx_gate.x, tmpy.x);
              ggml_cuda_mad(vSumfGate[j], tmpx_gate.y, tmpy.y);
            }
          }
        }
      }
    } else {
#ifdef FP16_AVAILABLE
      half2 sumh2[ncols_dst] = {{0.0f, 0.0f}};
      half2 sumh2_gate[ncols_dst] = {{0.0f, 0.0f}};

      for (int col2 = cTid; col2 < pNcols2; col2 += block_size) {
        const half2 tmpx = x2[col2];
        half2 tmpx_gate = make_half2(0.0f, 0.0f);
        if constexpr (has_fusion) {
          if (vUseGate) {
            tmpx_gate = gate_x2[col2];
          }
        }
#pragma unroll
        for (int j = 0; j < ncols_dst; ++j) {
          const float2 tmpy = vY2[j * pStrideColY2 + col2];
          sumh2[j] += tmpx * make_half2(tmpy.x, tmpy.y);

          if constexpr (has_fusion) {
            if (vUseGate) {
              sumh2_gate[j] += tmpx_gate * make_half2(tmpy.x, tmpy.y);
            }
          }
        }
      }

#pragma unroll
      for (int j = 0; j < ncols_dst; ++j) {
        vSumf[j] = __low2float(sumh2[j]) + __high2float(sumh2[j]);
      }

      if constexpr (has_fusion) {
        if (vUseGate) {
#pragma unroll
          for (int j = 0; j < ncols_dst; ++j) {
            vSumfGate[j] = __low2float(sumh2_gate[j]) + __high2float(sumh2_gate[j]);
          }
        }
      }
#else
      NO_DEVICE_CODE;
#endif // FP16_AVAILABLE
    }
  } else if constexpr (std::is_same_v<T, nv_bfloat16>) {
// TODO: add support for ggml_cuda_mad for hip_bfloat162
#if defined(GGML_USE_HIP)
    const int *x2 = (const int *)vX;
    const int *gate_x2 = nullptr;
    if constexpr (has_fusion) {
      if (vUseGate) {
        gate_x2 = (const int *)vGateX;
      }
    }
    for (int col2 = cTid; col2 < pNcols2; col2 += block_size) {
      const int tmpx = x2[col2];
      int tmpx_gate = 0;
      if constexpr (has_fusion) {
        if (vUseGate) {
          tmpx_gate = gate_x2[col2];
        }
      }
#pragma unroll
      for (int j = 0; j < ncols_dst; ++j) {
        const float2 tmpy = vY2[j * pStrideColY2 + col2];
        const float tmpx0 = ggml_cuda_cast<float>(reinterpret_cast<const nv_bfloat16 *>(&tmpx)[0]);
        const float tmpx1 = ggml_cuda_cast<float>(reinterpret_cast<const nv_bfloat16 *>(&tmpx)[1]);
        ggml_cuda_mad(vSumf[j], tmpx0, tmpy.x);
        ggml_cuda_mad(vSumf[j], tmpx1, tmpy.y);

        if constexpr (has_fusion) {
          if (vUseGate) {
            const float tmpx0_gate = ggml_cuda_cast<float>(reinterpret_cast<const nv_bfloat16 *>(&tmpx_gate)[0]);
            const float tmpx1_gate = ggml_cuda_cast<float>(reinterpret_cast<const nv_bfloat16 *>(&tmpx_gate)[1]);
            ggml_cuda_mad(vSumfGate[j], tmpx0_gate, tmpy.x);
            ggml_cuda_mad(vSumfGate[j], tmpx1_gate, tmpy.y);
          }
        }
      }
    }
#else
    const nv_bfloat162 *x2 = (const nv_bfloat162 *)vX;
    [[maybe_unused]] const nv_bfloat162 *gate_x2 = nullptr;
    if constexpr (has_fusion) {
      if (vUseGate) {
        gate_x2 = (const nv_bfloat162 *)vGateX;
      }
    }
    for (int col2 = cTid; col2 < pNcols2; col2 += block_size) {
      const nv_bfloat162 tmpx = x2[col2];
      [[maybe_unused]] nv_bfloat162 tmpx_gate;
      if constexpr (has_fusion) {
        if (vUseGate) {
          tmpx_gate = gate_x2[col2];
        }
      }
#pragma unroll
      for (int j = 0; j < ncols_dst; ++j) {
        const float2 tmpy = vY2[j * pStrideColY2 + col2];
        ggml_cuda_mad(vSumf[j], tmpx.x, tmpy.x);
        ggml_cuda_mad(vSumf[j], tmpx.y, tmpy.y);

        if constexpr (has_fusion) {
          if (vUseGate) {
            ggml_cuda_mad(vSumfGate[j], tmpx_gate.x, tmpy.x);
            ggml_cuda_mad(vSumfGate[j], tmpx_gate.y, tmpy.y);
          }
        }
      }
    }
#endif
  } else {
    static_assert(std::is_same_v<T, void>, "unsupported type");
  }

  ggml_cuda_pdl_lc();
#pragma unroll
  for (int j = 0; j < ncols_dst; ++j) {
    vSumf[j] = warp_reduce_sum<cWarpSize>(vSumf[j]);

    if constexpr (has_fusion) {
      if (vUseGate) {
        vSumfGate[j] = warp_reduce_sum<cWarpSize>(vSumfGate[j]);
      }
    }

    if (block_size > cWarpSize) {
      vBufIw[cTid / cWarpSize] = vSumf[j];
      if constexpr (has_fusion) {
        if (vUseGate) {
          vBufIwGate[cTid / cWarpSize] = vSumfGate[j];
        }
      }
      __syncthreads();
      if (cTid < cWarpSize) {
        vSumf[j] = vBufIw[cTid];
        vSumf[j] = warp_reduce_sum<cWarpSize>(vSumf[j]);
        if constexpr (has_fusion) {
          if (vUseGate) {
            vSumfGate[j] = vBufIwGate[cTid];
            vSumfGate[j] = warp_reduce_sum<cWarpSize>(vSumfGate[j]);
          }
        }
      }

      if (j < ncols_dst) {
        __syncthreads();
      }
    }
  }

  if (cTid >= ncols_dst) {
    return;
  }

  float vValue = vSumf[cTid];

  if constexpr (has_fusion) {
    if (vUseBias) {
      vValue += vXBias[cTid * pStrideColDst + cRow];
    }

    if (vUseGate) {
      float gate_value = vSumfGate[cTid];
      if (vUseGateBias) {
        gate_value += vGateBias[cTid * pStrideColDst + cRow];
      }
      switch (vGluOp) {
      case GGML_GLU_OP_SWIGLU:
        vValue *= ggml_cuda_op_silu_single(gate_value);
        break;
      case GGML_GLU_OP_GEGLU:
        vValue *= ggml_cuda_op_gelu_single(gate_value);
        break;
      case GGML_GLU_OP_SWIGLU_OAI: {
        vValue = ggml_cuda_op_swiglu_oai_single(gate_value, vValue);
        break;
      }
      case GGML_GLU_OP_SWIGLU_CLAMP:
        vValue = ggml_cuda_op_swiglu_clamp_single(gate_value, vValue, vGluLimit);
        break;
      default:
        break;
      }
    }
  }

  vDst[cTid * pStrideColDst + cRow] = vValue;

  if constexpr (!has_fusion) {
    GGML_UNUSED_VARS(vUseGate, vUseBias, vUseGateBias, vGluOp, vGluLimit, vGateX, vXBias, vGateBias, vSumfGate);
  }
}

template <typename T, typename type_acc, int ncols_dst, int block_size, bool is_multi_token_id = false>
static void fMulMatVecFSwitchFusion(const T *pX, const float *pY, const int32_t *pIds,
                                    const ggml_cuda_mm_fusion_args_device pFusion, float *pDst, const int64_t pNcols,
                                    const uint3 pNchannelsY, const int64_t pStrideRow, const int64_t pStrideColY,
                                    const int64_t pStrideColDst, const uint3 pChannelRatio, const int pStrideChannelX,
                                    const int pStrideChannelY, const int pStrideChannelDst, const uint3 pSampleRatio,
                                    const int pStrideSampleX, const int pStrideSampleY, const int pStrideSampleDst,
                                    const dim3 &pBlockDims, const dim3 &pBlockNums, const int pNbytesShared,
                                    const int pIdsStride, const cudaStream_t pStream) {

  const ggml_cuda_kernel_launch_params cLaunchParams = {pBlockNums, pBlockDims, pNbytesShared, pStream};

  const bool cHasFusion = pFusion.gate != nullptr || pFusion.x_bias != nullptr || pFusion.gate_bias != nullptr;
  if constexpr (ncols_dst == 1) {
    if (cHasFusion) {
      ggml_cuda_kernel_launch(fMulMatVecF<T, type_acc, ncols_dst, block_size, true, is_multi_token_id>, cLaunchParams,
                              pX, pY, pIds, pFusion, pDst, pNcols, pNchannelsY, pStrideRow, pStrideColY, pStrideColDst,
                              pChannelRatio, pStrideChannelX, pStrideChannelY, pStrideChannelDst, pSampleRatio,
                              pStrideSampleX, pStrideSampleY, pStrideSampleDst, pIdsStride);
      return;
    }
  }

  GGML_ASSERT(!cHasFusion && "fusion only supported for ncols_dst=1");

  ggml_cuda_kernel_launch(fMulMatVecF<T, type_acc, ncols_dst, block_size, false, is_multi_token_id>, cLaunchParams, pX,
                          pY, pIds, pFusion, pDst, pNcols, pNchannelsY, pStrideRow, pStrideColY, pStrideColDst,
                          pChannelRatio, pStrideChannelX, pStrideChannelY, pStrideChannelDst, pSampleRatio,
                          pStrideSampleX, pStrideSampleY, pStrideSampleDst, pIdsStride);
}

template <typename T, typename type_acc, int ncols_dst, bool is_multi_token_id = false>
void fLaunchMulMatVecFCuda(const T *pX, const float *pY, const int32_t *pIds,
                           const ggml_cuda_mm_fusion_args_device pFusion, float *pDst, const int64_t pNcols,
                           const int64_t pNrows, const int64_t pStrideRow, const int64_t pStrideColY,
                           const int64_t pStrideColDst, const int64_t pNchannelsX, const int64_t pNchannelsY,
                           const int64_t pNchannelsDst, const int64_t pStrideChannelX, const int64_t pStrideChannelY,
                           const int64_t pStrideChannelDst, const int64_t pNsamplesX, const int64_t pNsamplesDst,
                           const int64_t pStrideSampleX, const int64_t pStrideSampleY, const int64_t pStrideSampleDst,
                           const int64_t pNsamplesOrNtokens, const int64_t pIdsStride, cudaStream_t pStream) {
  GGML_ASSERT(pNcols % 2 == 0);
  GGML_ASSERT(pStrideRow % 2 == 0);
  GGML_ASSERT(pStrideColY % 2 == 0);
  GGML_ASSERT(pIds || pNchannelsDst % pNchannelsX == 0);
  GGML_ASSERT(pNsamplesDst % pNsamplesX == 0);
  const uint3 cNchannelsYFd = pIds ? init_fastdiv_values(pNchannelsY) : make_uint3(0, 0, 0);
  const uint3 cChannelRatioFd = pIds ? make_uint3(0, 0, 0) : init_fastdiv_values(pNchannelsDst / pNchannelsX);
  const uint3 cSampleRatioFd = init_fastdiv_values(pNsamplesDst / pNsamplesX);

  const int cDevice = ggml_cuda_get_device();
  const int cWarpSize = ggml_cuda_info().devices[cDevice].warp_size;

  int64_t vBlockSizeBest = cWarpSize;
  int64_t vNiterBest = (pNcols + 2 * cWarpSize - 1) / (2 * cWarpSize);
  int64_t vMaxBlockSize = 256;
  if (ggml_cuda_info().devices[cDevice].cc > GGML_CUDA_CC_OFFSET_AMD &&
      ggml_cuda_info().devices[cDevice].cc < GGML_CUDA_CC_RDNA1) {
    vMaxBlockSize = 128;
  }
  for (int64_t vBlockSize = 2 * cWarpSize; vBlockSize <= vMaxBlockSize; vBlockSize += cWarpSize) {
    const int64_t cNiter = (pNcols + 2 * vBlockSize - 1) / (2 * vBlockSize);
    if (cNiter < vNiterBest) {
      vNiterBest = cNiter;
      vBlockSizeBest = vBlockSize;
    }
  }

  const bool cHasFusion = pFusion.gate != nullptr || pFusion.x_bias != nullptr || pFusion.gate_bias != nullptr;

  const int cNbytesShared = cWarpSize * sizeof(float) + (cHasFusion ? cWarpSize * sizeof(float) : 0);
  const dim3 cBlockNums(pNrows, pNchannelsDst, pNsamplesOrNtokens);
  const dim3 cBlockDims(vBlockSizeBest, 1, 1);
  switch (vBlockSizeBest) {
  case 32: {
    fMulMatVecFSwitchFusion<T, type_acc, ncols_dst, 32, is_multi_token_id>(
        pX, pY, pIds, pFusion, pDst, pNcols / 2, cNchannelsYFd, pStrideRow, pStrideColY / 2, pStrideColDst,
        cChannelRatioFd, pStrideChannelX, pStrideChannelY, pStrideChannelDst, cSampleRatioFd, pStrideSampleX,
        pStrideSampleY, pStrideSampleDst, cBlockDims, cBlockNums, cNbytesShared, pIdsStride, pStream);
  } break;
  case 64: {
    fMulMatVecFSwitchFusion<T, type_acc, ncols_dst, 64, is_multi_token_id>(
        pX, pY, pIds, pFusion, pDst, pNcols / 2, cNchannelsYFd, pStrideRow, pStrideColY / 2, pStrideColDst,
        cChannelRatioFd, pStrideChannelX, pStrideChannelY, pStrideChannelDst, cSampleRatioFd, pStrideSampleX,
        pStrideSampleY, pStrideSampleDst, cBlockDims, cBlockNums, cNbytesShared, pIdsStride, pStream);
  } break;
  case 96: {
    fMulMatVecFSwitchFusion<T, type_acc, ncols_dst, 96, is_multi_token_id>(
        pX, pY, pIds, pFusion, pDst, pNcols / 2, cNchannelsYFd, pStrideRow, pStrideColY / 2, pStrideColDst,
        cChannelRatioFd, pStrideChannelX, pStrideChannelY, pStrideChannelDst, cSampleRatioFd, pStrideSampleX,
        pStrideSampleY, pStrideSampleDst, cBlockDims, cBlockNums, cNbytesShared, pIdsStride, pStream);
  } break;
  case 128: {
    fMulMatVecFSwitchFusion<T, type_acc, ncols_dst, 128, is_multi_token_id>(
        pX, pY, pIds, pFusion, pDst, pNcols / 2, cNchannelsYFd, pStrideRow, pStrideColY / 2, pStrideColDst,
        cChannelRatioFd, pStrideChannelX, pStrideChannelY, pStrideChannelDst, cSampleRatioFd, pStrideSampleX,
        pStrideSampleY, pStrideSampleDst, cBlockDims, cBlockNums, cNbytesShared, pIdsStride, pStream);
  } break;
  case 160: {
    fMulMatVecFSwitchFusion<T, type_acc, ncols_dst, 160, is_multi_token_id>(
        pX, pY, pIds, pFusion, pDst, pNcols / 2, cNchannelsYFd, pStrideRow, pStrideColY / 2, pStrideColDst,
        cChannelRatioFd, pStrideChannelX, pStrideChannelY, pStrideChannelDst, cSampleRatioFd, pStrideSampleX,
        pStrideSampleY, pStrideSampleDst, cBlockDims, cBlockNums, cNbytesShared, pIdsStride, pStream);
  } break;
  case 192: {
    fMulMatVecFSwitchFusion<T, type_acc, ncols_dst, 192, is_multi_token_id>(
        pX, pY, pIds, pFusion, pDst, pNcols / 2, cNchannelsYFd, pStrideRow, pStrideColY / 2, pStrideColDst,
        cChannelRatioFd, pStrideChannelX, pStrideChannelY, pStrideChannelDst, cSampleRatioFd, pStrideSampleX,
        pStrideSampleY, pStrideSampleDst, cBlockDims, cBlockNums, cNbytesShared, pIdsStride, pStream);
  } break;
  case 224: {
    fMulMatVecFSwitchFusion<T, type_acc, ncols_dst, 224, is_multi_token_id>(
        pX, pY, pIds, pFusion, pDst, pNcols / 2, cNchannelsYFd, pStrideRow, pStrideColY / 2, pStrideColDst,
        cChannelRatioFd, pStrideChannelX, pStrideChannelY, pStrideChannelDst, cSampleRatioFd, pStrideSampleX,
        pStrideSampleY, pStrideSampleDst, cBlockDims, cBlockNums, cNbytesShared, pIdsStride, pStream);
  } break;
  case 256: {
    fMulMatVecFSwitchFusion<T, type_acc, ncols_dst, 256, is_multi_token_id>(
        pX, pY, pIds, pFusion, pDst, pNcols / 2, cNchannelsYFd, pStrideRow, pStrideColY / 2, pStrideColDst,
        cChannelRatioFd, pStrideChannelX, pStrideChannelY, pStrideChannelDst, cSampleRatioFd, pStrideSampleX,
        pStrideSampleY, pStrideSampleDst, cBlockDims, cBlockNums, cNbytesShared, pIdsStride, pStream);
  } break;
  default: {
    GGML_ABORT("fatal error");
  } break;
  }
}

template <typename T, typename type_acc>
static void fMulMatVecFCudaSwitchNcolsDst(
    const T *pX, const float *pY, const int32_t *pIds, const ggml_cuda_mm_fusion_args_device pFusion, float *pDst,
    const int64_t pNcols, const int64_t pNrows, const int64_t pNcolsDst, const int64_t pStrideRow,
    const int64_t pStrideColY, const int64_t pStrideColDst, const int64_t pNchannelsX, const int64_t pNchannelsY,
    const int64_t pNchannelsDst, const int64_t pStrideChannelX, const int64_t pStrideChannelY,
    const int64_t pStrideChannelDst, const int64_t pNsamplesX, const int64_t pNsamplesDst, const int64_t pStrideSampleX,
    const int64_t pStrideSampleY, const int64_t pStrideSampleDst, const int64_t pIdsStride, cudaStream_t pStream) {

  const bool cHasIds = pIds != nullptr;

  if (cHasIds && pNcolsDst > 1) {
    // Multi-token MUL_MAT_ID path only - single-token goes through regular path below
    constexpr int cCNcolsDst = 1;
    fLaunchMulMatVecFCuda<T, type_acc, cCNcolsDst, true>(
        pX, pY, pIds, pFusion, pDst, pNcols, pNrows, pStrideRow, pStrideColY, pStrideColDst, pNchannelsX, pNchannelsY,
        pNchannelsDst, pStrideChannelX, pStrideChannelY, pStrideChannelDst, pNsamplesX, pNsamplesDst, pStrideSampleX,
        pStrideSampleY, pStrideSampleDst, pNcolsDst, pIdsStride, pStream);
    return;
  }

  if (cHasIds) {
    // Single-token MUL_MAT_ID path
    constexpr int cCNcolsDst = 1;
    fLaunchMulMatVecFCuda<T, type_acc, cCNcolsDst>(
        pX, pY, pIds, pFusion, pDst, pNcols, pNrows, pStrideRow, pStrideColY, pStrideColDst, pNchannelsX, pNchannelsY,
        pNchannelsDst, pStrideChannelX, pStrideChannelY, pStrideChannelDst, pNsamplesX, pNsamplesDst, pStrideSampleX,
        pStrideSampleY, pStrideSampleDst, pNcolsDst, pIdsStride, pStream);
    return;
  }

  switch (pNcolsDst) {
  case 1:
    fLaunchMulMatVecFCuda<T, type_acc, 1>(pX, pY, pIds, pFusion, pDst, pNcols, pNrows, pStrideRow, pStrideColY,
                                          pStrideColDst, pNchannelsX, pNchannelsY, pNchannelsDst, pStrideChannelX,
                                          pStrideChannelY, pStrideChannelDst, pNsamplesX, pNsamplesDst, pStrideSampleX,
                                          pStrideSampleY, pStrideSampleDst, pNsamplesDst, pIdsStride, pStream);
    break;
  case 2:
    fLaunchMulMatVecFCuda<T, type_acc, 2>(pX, pY, pIds, pFusion, pDst, pNcols, pNrows, pStrideRow, pStrideColY,
                                          pStrideColDst, pNchannelsX, pNchannelsY, pNchannelsDst, pStrideChannelX,
                                          pStrideChannelY, pStrideChannelDst, pNsamplesX, pNsamplesDst, pStrideSampleX,
                                          pStrideSampleY, pStrideSampleDst, pNsamplesDst, pIdsStride, pStream);
    break;
  case 3:
    fLaunchMulMatVecFCuda<T, type_acc, 3>(pX, pY, pIds, pFusion, pDst, pNcols, pNrows, pStrideRow, pStrideColY,
                                          pStrideColDst, pNchannelsX, pNchannelsY, pNchannelsDst, pStrideChannelX,
                                          pStrideChannelY, pStrideChannelDst, pNsamplesX, pNsamplesDst, pStrideSampleX,
                                          pStrideSampleY, pStrideSampleDst, pNsamplesDst, pIdsStride, pStream);
    break;
  case 4:
    fLaunchMulMatVecFCuda<T, type_acc, 4>(pX, pY, pIds, pFusion, pDst, pNcols, pNrows, pStrideRow, pStrideColY,
                                          pStrideColDst, pNchannelsX, pNchannelsY, pNchannelsDst, pStrideChannelX,
                                          pStrideChannelY, pStrideChannelDst, pNsamplesX, pNsamplesDst, pStrideSampleX,
                                          pStrideSampleY, pStrideSampleDst, pNsamplesDst, pIdsStride, pStream);
    break;
  case 5:
    fLaunchMulMatVecFCuda<T, type_acc, 5>(pX, pY, pIds, pFusion, pDst, pNcols, pNrows, pStrideRow, pStrideColY,
                                          pStrideColDst, pNchannelsX, pNchannelsY, pNchannelsDst, pStrideChannelX,
                                          pStrideChannelY, pStrideChannelDst, pNsamplesX, pNsamplesDst, pStrideSampleX,
                                          pStrideSampleY, pStrideSampleDst, pNsamplesDst, pIdsStride, pStream);
    break;
  case 6:
    fLaunchMulMatVecFCuda<T, type_acc, 6>(pX, pY, pIds, pFusion, pDst, pNcols, pNrows, pStrideRow, pStrideColY,
                                          pStrideColDst, pNchannelsX, pNchannelsY, pNchannelsDst, pStrideChannelX,
                                          pStrideChannelY, pStrideChannelDst, pNsamplesX, pNsamplesDst, pStrideSampleX,
                                          pStrideSampleY, pStrideSampleDst, pNsamplesDst, pIdsStride, pStream);
    break;
  case 7:
    fLaunchMulMatVecFCuda<T, type_acc, 7>(pX, pY, pIds, pFusion, pDst, pNcols, pNrows, pStrideRow, pStrideColY,
                                          pStrideColDst, pNchannelsX, pNchannelsY, pNchannelsDst, pStrideChannelX,
                                          pStrideChannelY, pStrideChannelDst, pNsamplesX, pNsamplesDst, pStrideSampleX,
                                          pStrideSampleY, pStrideSampleDst, pNsamplesDst, pIdsStride, pStream);
    break;
  case 8:
    fLaunchMulMatVecFCuda<T, type_acc, 8>(pX, pY, pIds, pFusion, pDst, pNcols, pNrows, pStrideRow, pStrideColY,
                                          pStrideColDst, pNchannelsX, pNchannelsY, pNchannelsDst, pStrideChannelX,
                                          pStrideChannelY, pStrideChannelDst, pNsamplesX, pNsamplesDst, pStrideSampleX,
                                          pStrideSampleY, pStrideSampleDst, pNsamplesDst, pIdsStride, pStream);
    break;
  default:
    GGML_ABORT("fatal error");
    break;
  }
}

template <typename T>
static void
fMulMatVecFCuda(const T *pX, const float *pY, const int32_t *pIds, const ggml_cuda_mm_fusion_args_device pFusion,
                float *pDst, const int64_t pNcols, const int64_t pNrows, const int64_t pNcolsDst,
                const int64_t pStrideRow, const int64_t pStrideColY, const int pStrideColDst, const int64_t pNchannelsX,
                const int64_t pNchannelsY, const int64_t pNchannelsDst, const int64_t pStrideChannelX,
                const int64_t pStrideChannelY, const int64_t pStrideChannelDst, const int64_t pNsamplesX,
                const int64_t pNsamplesDst, const int64_t pStrideSampleX, const int64_t pStrideSampleY,
                const int64_t pStrideSampleDst, const int64_t pIdsStride, enum ggml_prec pPrec, cudaStream_t pStream) {

  if constexpr (std::is_same_v<T, half>) {
    if (pPrec == GGML_PREC_DEFAULT) {
      fMulMatVecFCudaSwitchNcolsDst<T, half>(
          pX, pY, pIds, pFusion, pDst, pNcols, pNrows, pNcolsDst, pStrideRow, pStrideColY, pStrideColDst, pNchannelsX,
          pNchannelsY, pNchannelsDst, pStrideChannelX, pStrideChannelY, pStrideChannelDst, pNsamplesX, pNsamplesDst,
          pStrideSampleX, pStrideSampleY, pStrideSampleDst, pIdsStride, pStream);
      return;
    }
  }
  fMulMatVecFCudaSwitchNcolsDst<T, float>(pX, pY, pIds, pFusion, pDst, pNcols, pNrows, pNcolsDst, pStrideRow,
                                          pStrideColY, pStrideColDst, pNchannelsX, pNchannelsY, pNchannelsDst,
                                          pStrideChannelX, pStrideChannelY, pStrideChannelDst, pNsamplesX, pNsamplesDst,
                                          pStrideSampleX, pStrideSampleY, pStrideSampleDst, pIdsStride, pStream);
}

void ggml_cuda_mul_mat_vec_f(ggml_backend_cuda_context &pCtx, const ggml_tensor *src0, const ggml_tensor *src1,
                             const ggml_tensor *pIds, ggml_tensor *dst, const ggml_cuda_mm_fusion_args_host *pFusion) {
  GGML_ASSERT(src1->type == GGML_TYPE_F32);
  GGML_ASSERT(!pIds || pIds->type == GGML_TYPE_I32);
  GGML_ASSERT(dst->type == GGML_TYPE_F32);

  GGML_TENSOR_BINARY_OP_LOCALS;

  const size_t cTsSrc0 = ggml_type_size(src0->type);
  const size_t cTsSrc1 = ggml_type_size(src1->type);
  const size_t cTsDst = ggml_type_size(dst->type);

  GGML_ASSERT(!pIds || ne12 <= MMVF_MAX_BATCH_SIZE);
  GGML_ASSERT(ne13 == ne3);

  GGML_ASSERT(nb00 == cTsSrc0);
  GGML_ASSERT(nb10 == cTsSrc1);
  GGML_ASSERT(!pIds || pIds->nb[0] == ggml_type_size(pIds->type));
  GGML_ASSERT(nb0 == cTsDst);

  const int cCc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
  const enum ggml_prec cPrec = fast_fp16_available(cCc) ? ggml_prec(dst->op_params[0]) : GGML_PREC_F32;

  const float *vSrc1D = (const float *)src1->data;
  const int32_t *vIdsD = pIds ? (const int32_t *)pIds->data : nullptr;
  float *vDstD = (float *)dst->data;

  ggml_cuda_mm_fusion_args_device vFusionLocal{};

  if (pFusion) {
    GGML_ASSERT(!pIds || dst->ne[2] == 1);
    GGML_ASSERT(pIds || dst->ne[1] == 1);
    if (pFusion->x_bias) {
      GGML_ASSERT(pFusion->x_bias->type == GGML_TYPE_F32);
      GGML_ASSERT(pFusion->x_bias->ne[0] == dst->ne[0]);
      GGML_ASSERT(!pIds || pFusion->x_bias->ne[1] == src0->ne[2]);
      vFusionLocal.x_bias = pFusion->x_bias->data;
    }
    if (pFusion->gate) {
      GGML_ASSERT(pFusion->gate->type == src0->type && ggml_are_same_stride(pFusion->gate, src0));
      vFusionLocal.gate = pFusion->gate->data;
    }
    if (pFusion->gate_bias) {
      GGML_ASSERT(pFusion->gate_bias->type == GGML_TYPE_F32);
      GGML_ASSERT(pFusion->gate_bias->ne[0] == dst->ne[0]);
      GGML_ASSERT(!pIds || pFusion->gate_bias->ne[1] == src0->ne[2]);
      vFusionLocal.gate_bias = pFusion->gate_bias->data;
    }
    vFusionLocal.glu_op = pFusion->glu_op;
    vFusionLocal.glu_limit = pFusion->glu_limit;
  }

  const int64_t cS01 = src0->nb[1] / cTsSrc0;
  const int64_t cS11 = src1->nb[1] / cTsSrc1;
  const int64_t cS1 = dst->nb[1] / cTsDst;
  const int64_t cS02 = src0->nb[2] / cTsSrc0;
  const int64_t cS12 = src1->nb[2] / cTsSrc1;
  const int64_t cS2 = dst->nb[2] / cTsDst;
  const int64_t cS03 = src0->nb[3] / cTsSrc0;
  const int64_t cS13 = src1->nb[3] / cTsSrc1;
  const int64_t cS3 = dst->nb[3] / cTsDst;

  // For MUL_MAT_ID the memory layout is different than for MUL_MAT:
  const int64_t cNcolsDst = pIds ? ne2 : ne1;
  const int64_t cNchannelsY = pIds ? ne11 : ne12;
  const int64_t cNchannelsDst = pIds ? ne1 : ne2;
  const int64_t cStrideColDst = pIds ? cS2 : cS1;
  const int64_t cStrideColY = pIds ? cS12 : cS11;
  const int64_t cStrideChannelDst = pIds ? cS1 : cS2;
  const int64_t cStrideChannelY = pIds ? cS11 : cS12;

  const int64_t cIdsStride = pIds ? pIds->nb[1] / ggml_type_size(pIds->type) : 0;

  switch (src0->type) {
  case GGML_TYPE_F32: {
    const float *vSrc0D = (const float *)src0->data;
    fMulMatVecFCuda(vSrc0D, vSrc1D, vIdsD, vFusionLocal, vDstD, ne00, ne01, cNcolsDst, cS01, cStrideColY, cStrideColDst,
                    ne02, cNchannelsY, cNchannelsDst, cS02, cStrideChannelY, cStrideChannelDst, ne03, ne3, cS03, cS13,
                    cS3, cIdsStride, cPrec, pCtx.stream());
  } break;
  case GGML_TYPE_F16: {
    const half *vSrc0D = (const half *)src0->data;
    fMulMatVecFCuda(vSrc0D, vSrc1D, vIdsD, vFusionLocal, vDstD, ne00, ne01, cNcolsDst, cS01, cStrideColY, cStrideColDst,
                    ne02, cNchannelsY, cNchannelsDst, cS02, cStrideChannelY, cStrideChannelDst, ne03, ne3, cS03, cS13,
                    cS3, cIdsStride, cPrec, pCtx.stream());
  } break;
  case GGML_TYPE_BF16: {
    const nv_bfloat16 *vSrc0D = (const nv_bfloat16 *)src0->data;
    fMulMatVecFCuda(vSrc0D, vSrc1D, vIdsD, vFusionLocal, vDstD, ne00, ne01, cNcolsDst, cS01, cStrideColY, cStrideColDst,
                    ne02, cNchannelsY, cNchannelsDst, cS02, cStrideChannelY, cStrideChannelDst, ne03, ne3, cS03, cS13,
                    cS3, cIdsStride, cPrec, pCtx.stream());
  } break;
  default:
    GGML_ABORT("unsupported type: %s", ggml_type_name(src0->type));
  }
}

void ggml_cuda_op_mul_mat_vec_f(ggml_backend_cuda_context &pCtx, const ggml_tensor *pSrc0, const ggml_tensor *pSrc1,
                                ggml_tensor *pDst, const char *pSrc0DdI, const float *pSrc1DdfI, const char *pSrc1DdqI,
                                float *pDstDdI, const int64_t pRowLow, const int64_t pRowHigh, const int64_t pSrc1Ncols,
                                const int64_t pSrc1PaddedRowSize, cudaStream_t pStream) {

  GGML_ASSERT(pSrc1->type == GGML_TYPE_F32);
  GGML_ASSERT(pDst->type == GGML_TYPE_F32);

  const int64_t cNe00 = pSrc0->ne[0];
  const int64_t cNe10 = pSrc1->ne[0];
  const int64_t cNe0 = pDst->ne[0];
  const int64_t cRowDiff = pRowHigh - pRowLow;

  const int cId = ggml_cuda_get_device();
  const int cCc = ggml_cuda_info().devices[cId].cc;
  const enum ggml_prec cPrec = fast_fp16_available(cCc) ? ggml_prec(pDst->op_params[0]) : GGML_PREC_F32;

  // ggml_cuda_op provides single, contiguous matrices
  const int64_t cStrideRow = cNe00;
  const int64_t cStrideColY = cNe10;
  const int64_t cStrideColDst = cId == pCtx.device ? cNe0 : cRowDiff; // main device has larger memory buffer
  const int64_t cNchannelsX = 1;
  const int64_t cNchannelsY = 1;
  const int64_t cNchannelsDst = 1;
  const int64_t cStrideChannelX = 0;
  const int64_t cStrideChannelY = 0;
  const int64_t cStrideChannelDst = 0;
  const int64_t cNsamplesX = 1;
  const int64_t cNsamplesDst = 1;
  const int64_t cStrideSampleX = 0;
  const int64_t cStrideSampleY = 0;
  const int64_t cStrideSampleDst = 0;

  ggml_cuda_mm_fusion_args_device vEmpty{};
  switch (pSrc0->type) {
  case GGML_TYPE_F32: {
    const float *vSrc0D = (const float *)pSrc0DdI;
    fMulMatVecFCuda(vSrc0D, pSrc1DdfI, nullptr, vEmpty, pDstDdI, cNe00, cRowDiff, pSrc1Ncols, cStrideRow, cStrideColY,
                    cStrideColDst, cNchannelsX, cNchannelsY, cNchannelsDst, cStrideChannelX, cStrideChannelY,
                    cStrideChannelDst, cNsamplesX, cNsamplesDst, cStrideSampleX, cStrideSampleY, cStrideSampleDst, 0,
                    cPrec, pStream);
  } break;
  case GGML_TYPE_F16: {
    const half *vSrc0D = (const half *)pSrc0DdI;
    fMulMatVecFCuda(vSrc0D, pSrc1DdfI, nullptr, vEmpty, pDstDdI, cNe00, cRowDiff, pSrc1Ncols, cStrideRow, cStrideColY,
                    cStrideColDst, cNchannelsX, cNchannelsY, cNchannelsDst, cStrideChannelX, cStrideChannelY,
                    cStrideChannelDst, cNsamplesX, cNsamplesDst, cStrideSampleX, cStrideSampleY, cStrideSampleDst, 0,
                    cPrec, pStream);
  } break;
  case GGML_TYPE_BF16: {
    const nv_bfloat16 *vSrc0D = (const nv_bfloat16 *)pSrc0DdI;
    fMulMatVecFCuda(vSrc0D, pSrc1DdfI, nullptr, vEmpty, pDstDdI, cNe00, cRowDiff, pSrc1Ncols, cStrideRow, cStrideColY,
                    cStrideColDst, cNchannelsX, cNchannelsY, cNchannelsDst, cStrideChannelX, cStrideChannelY,
                    cStrideChannelDst, cNsamplesX, cNsamplesDst, cStrideSampleX, cStrideSampleY, cStrideSampleDst, 0,
                    cPrec, pStream);
  } break;
  default:
    GGML_ABORT("unsupported type: %s", ggml_type_name(pSrc0->type));
  }

  GGML_UNUSED_VARS(pCtx, pSrc1, pDst, pSrc1DdqI, pSrc1Ncols, pSrc1PaddedRowSize);
}

bool ggml_cuda_should_use_mmvf(enum ggml_type pType, int pCc, const int64_t *pSrc0Ne, const size_t *pSrc0Nb,
                               int64_t pNe11) {
  if (pSrc0Ne[0] % 2 != 0) {
    return false;
  }

  const size_t cTs = ggml_type_size(pType);
  if (pSrc0Nb[0] != cTs) {
    return false;
  }

  // Pointers not aligned to the size of half2/nv_bfloat162/float2 would result in a crash:
  for (size_t vI = 1; vI < GGML_MAX_DIMS; ++vI) {
    if (pSrc0Nb[vI] % (2 * cTs) != 0) {
      return false;
    }
  }

  switch (pType) {
  case GGML_TYPE_F32:
    if (GGML_CUDA_CC_IS_NVIDIA(pCc)) {
      if (ampere_mma_available(pCc)) {
        return pNe11 <= 3;
      }
      if (pCc >= GGML_CUDA_CC_TURING) {
        return pNe11 <= 4;
      }
      return pNe11 <= 3;
    } else if (GGML_CUDA_CC_IS_AMD(pCc)) {
      if (fp32_mma_hardware_available(pCc)) {
        return pNe11 <= 3;
      }
      return pNe11 <= 8;
    }
    return pNe11 <= 8;
  case GGML_TYPE_F16:
    if (GGML_CUDA_CC_IS_NVIDIA(pCc)) {
      const bool cSrc0Small = (pSrc0Ne[1] <= 512 || pSrc0Ne[2] * pSrc0Ne[3] == 1);
      if (ampere_mma_available(pCc)) {
        return cSrc0Small && pNe11 == 1;
      }
      if (pCc >= GGML_CUDA_CC_ADA_LOVELACE) {
        return cSrc0Small && pNe11 <= 4;
      }
      if (fp16_mma_hardware_available(pCc)) {
        return cSrc0Small && pNe11 <= 3;
      }
      return pNe11 <= 8;
    } else if (GGML_CUDA_CC_IS_AMD(pCc)) {
      if (fp16_mma_hardware_available(pCc)) {
        if (GGML_CUDA_CC_IS_RDNA3(pCc)) {
          return pNe11 <= 3;
        }
        if (GGML_CUDA_CC_IS_RDNA4(pCc)) {
          return pNe11 <= 5;
        }
        return pNe11 <= 2;
      }
      return pNe11 <= 8;
    }
    return pNe11 <= 8;
  case GGML_TYPE_BF16:
    if (GGML_CUDA_CC_IS_NVIDIA(pCc)) {
      const bool cSrc0Small = (pSrc0Ne[1] <= 512 || pSrc0Ne[2] * pSrc0Ne[3] == 1);
      if (ampere_mma_available(pCc)) {
        return cSrc0Small && pNe11 == 1;
      }
      if (pCc >= GGML_CUDA_CC_ADA_LOVELACE) {
        return cSrc0Small && pNe11 <= 4;
      }
      if (bf16_mma_hardware_available(pCc)) {
        return cSrc0Small && pNe11 <= 3;
      }
      return pNe11 <= 8;
    } else if (GGML_CUDA_CC_IS_AMD(pCc)) {
      if (bf16_mma_hardware_available(pCc)) {
        return pNe11 <= 3;
      }
      return pNe11 <= 8;
    }
    return pNe11 <= 8;
  default:
    return false;
  }
}
