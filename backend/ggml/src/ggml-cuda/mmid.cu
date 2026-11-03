#include "common.cuh"
#include "mmid.cuh"

// To reduce shared memory use, store "it" and "iex_used" with 22/10 bits each.
struct mm_ids_helper_store {
  uint32_t vData;

  __device__ mm_ids_helper_store(const uint32_t pIt, const uint32_t pIexUsed) {
    vData = (pIt & 0x003FFFFF) | (pIexUsed << 22);
  }

  __device__ uint32_t mIt() const { return vData & 0x003FFFFF; }

  __device__ uint32_t mIexUsed() const { return vData >> 22; }
};
static_assert(sizeof(mm_ids_helper_store) == 4, "unexpected size for mm_ids_helper_store");

// the generic path passes 0, which needs no padding since it never groups lanes by token
template <int n> struct mm_ids_pow2 {
  static constexpr int value = 2 * mm_ids_pow2<(n + 1) / 2>::value;
};
template <> struct mm_ids_pow2<1> {
  static constexpr int value = 1;
};
template <> struct mm_ids_pow2<0> {
  static constexpr int value = 1;
};

// Helper function for mul_mat_id, converts ids to a more convenient format.
// ids_src1 describes how to permute the flattened column indices of src1 in order to get a compact src1 tensor sorted
// by expert. ids_dst describes the same mapping but for the dst tensor. The upper and lower bounds for the ith expert
// in the compact src1 tensor are stored in expert_bounds[i:i+1].
template <int n_expert_used_template>
__launch_bounds__(ggml_cuda_get_physical_warp_size(), 1) static __global__
    void fMmIdsHelper(const int32_t *__restrict__ pIds, int32_t *__restrict__ pIdsSrc1, int32_t *__restrict__ pIdsDst,
                      int32_t *__restrict__ pExpertBounds, const int pNTokens, const int pNExpertUsedVar,
                      const int pNchannelsY, const int pSi1, const int pSis1, const bool pWriteInverse) {
  constexpr int cWarpSize = ggml_cuda_get_physical_warp_size();
  const int cNExpertUsed = n_expert_used_template == 0 ? pNExpertUsedVar : n_expert_used_template;
  const int cExpert = blockIdx.x;

  // token slots per warp lane group, padded to a power of 2 so a warp divides evenly
  constexpr int cNeuPadded = mm_ids_pow2<n_expert_used_template>::value;

  extern __shared__ char aDataMmIdsHelper[];
  mm_ids_helper_store *vStore = (mm_ids_helper_store *)aDataMmIdsHelper;

  int vNexPrev = 0;   // Number of columns for experts with a lower index.
  int vItCompact = 0; // Running index for the compact slice of this expert.

  if constexpr (n_expert_used_template == 0) {
    // Generic implementation:
    for (int vIt = 0; vIt < pNTokens; ++vIt) {
      int vIexUsed = -1; // The index at which the expert is used, if any.
      for (int vIex = threadIdx.x; vIex < cNExpertUsed; vIex += cWarpSize) {
        const int cExpertUsed = pIds[vIt * pSi1 + vIex];
        vNexPrev += cExpertUsed < cExpert;
        if (cExpertUsed == cExpert) {
          vIexUsed = vIex;
        }
      }

      if (vIexUsed != -1) {
        vStore[vItCompact] = mm_ids_helper_store(vIt, vIexUsed);
      }

      if (warp_reduce_any<cWarpSize>(vIexUsed != -1)) {
        vItCompact++;
      }
    }
  } else {
    // Implementation optimized for specific numbers of experts used:
    // a warp holds a whole number of token slots, so the slot count is padded to a power of 2
    static_assert(cNeuPadded <= cWarpSize && cWarpSize % cNeuPadded == 0, "bad n_expert_used");
    for (int it0 = 0; it0 < pNTokens; it0 += cWarpSize / cNeuPadded) {
      const int it = it0 + threadIdx.x / cNeuPadded;

      const int iex = threadIdx.x % cNeuPadded; // The index at which the expert is used, if any.
      const int expert_used =
          (cNeuPadded == cNExpertUsed || iex < cNExpertUsed) && it < pNTokens ? pIds[it * pSi1 + iex] : INT_MAX;
      const int iex_used = expert_used == cExpert ? iex : -1;
      vNexPrev += expert_used < cExpert;

      // Whether the threads at this token position have used the expert:
      const int it_compact_add_self = warp_reduce_any<cNeuPadded>(iex_used != -1);

      // Do a scan over threads at lower token positions in warp to get the correct index for writing data:
      int it_compact_add_lower = 0;
#pragma unroll
      for (int offset = cNeuPadded; offset < cWarpSize; offset += cNeuPadded) {
        const int tmp = __shfl_up_sync(0xFFFFFFFF, it_compact_add_self, offset, cWarpSize);
        if (threadIdx.x >= static_cast<unsigned int>(offset)) {
          it_compact_add_lower += tmp;
        }
      }

      if (iex_used != -1) {
        vStore[vItCompact + it_compact_add_lower] = mm_ids_helper_store(it, iex_used);
      }

      // The thread with the highest index in the warp always has the sum over the whole warp, use it to increment all
      // threads:
      vItCompact += __shfl_sync(0xFFFFFFFF, it_compact_add_lower + it_compact_add_self, cWarpSize - 1, cWarpSize);
    }
  }
  vNexPrev = warp_reduce_sum<cWarpSize>(vNexPrev);
  ggml_cuda_syncwarp();

  for (int vItc = threadIdx.x; vItc < vItCompact; vItc += cWarpSize) {
    const mm_ids_helper_store cStoreIt = vStore[vItc];
    const int cIt = cStoreIt.mIt();
    const int cIexUsed = cStoreIt.mIexUsed();
    pIdsDst[vNexPrev + vItc] = cIt * cNExpertUsed + cIexUsed;
    // ids_src1 holds the forward map, or the inverse map (token slot -> compact row) for quant dedup
    if (pWriteInverse) {
      pIdsSrc1[cIt * cNExpertUsed + cIexUsed] = vNexPrev + vItc;
    } else {
      pIdsSrc1[vNexPrev + vItc] = cIt * pSis1 + cIexUsed % pNchannelsY;
    }
  }

  if (threadIdx.x != 0) {
    return;
  }

  pExpertBounds[cExpert] = vNexPrev;

  if (cExpert < static_cast<int>(gridDim.x) - 1) {
    return;
  }

  pExpertBounds[gridDim.x] = vNexPrev + vItCompact;
}

template <int n_expert_used_template>
static void fLaunchMmIdsHelper(const int32_t *__restrict__ pIds, int32_t *__restrict__ pIdsSrc1,
                               int32_t *__restrict__ pIdsDst, int32_t *__restrict__ pExpertBounds, const int pNExperts,
                               const int pNTokens, const int pNExpertUsedVar, const int pNchannelsY, const int pSi1,
                               const int pSis1, const bool pWriteInverse, cudaStream_t pStream) {
  GGML_ASSERT(pNTokens < (1 << 22) && "too few bits in mm_ids_helper_store");
  GGML_ASSERT(pNExpertUsedVar < (1 << 10) && "too few bits in mm_ids_helper_store");

  const int cId = ggml_cuda_get_device();
  const int cWarpSize = ggml_cuda_info().devices[cId].warp_size;
  const size_t cSmpbo = ggml_cuda_info().devices[cId].smpbo;
  CUDA_SET_SHARED_MEMORY_LIMIT(fMmIdsHelper<n_expert_used_template>, cSmpbo);

  const dim3 cNumBlocks(pNExperts, 1, 1);
  const dim3 cBlockSize(cWarpSize, 1, 1);
  const size_t cNbytesShared = pNTokens * sizeof(mm_ids_helper_store);
  GGML_ASSERT(cNbytesShared <= cSmpbo);
  fMmIdsHelper<n_expert_used_template><<<cNumBlocks, cBlockSize, cNbytesShared, pStream>>>(
      pIds, pIdsSrc1, pIdsDst, pExpertBounds, pNTokens, pNExpertUsedVar, pNchannelsY, pSi1, pSis1, pWriteInverse);
}

void ggml_cuda_launch_mm_ids_helper(const int32_t *__restrict__ pIds, int32_t *__restrict__ pIdsSrc1,
                                    int32_t *__restrict__ pIdsDst, int32_t *__restrict__ pExpertBounds,
                                    const int pNExperts, const int pNTokens, const int pNExpertUsed,
                                    const int pNchannelsY, const int pSi1, const int pSis1, const bool pWriteInverse,
                                    cudaStream_t pStream) {
  switch (pNExpertUsed) {
  case 2:
    fLaunchMmIdsHelper<2>(pIds, pIdsSrc1, pIdsDst, pExpertBounds, pNExperts, pNTokens, pNExpertUsed, pNchannelsY, pSi1,
                          pSis1, pWriteInverse, pStream);
    break;
  case 4:
    fLaunchMmIdsHelper<4>(pIds, pIdsSrc1, pIdsDst, pExpertBounds, pNExperts, pNTokens, pNExpertUsed, pNchannelsY, pSi1,
                          pSis1, pWriteInverse, pStream);
    break;
  case 6:
    fLaunchMmIdsHelper<6>(pIds, pIdsSrc1, pIdsDst, pExpertBounds, pNExperts, pNTokens, pNExpertUsed, pNchannelsY, pSi1,
                          pSis1, pWriteInverse, pStream);
    break;
  case 8:
    fLaunchMmIdsHelper<8>(pIds, pIdsSrc1, pIdsDst, pExpertBounds, pNExperts, pNTokens, pNExpertUsed, pNchannelsY, pSi1,
                          pSis1, pWriteInverse, pStream);
    break;
  case 10:
    fLaunchMmIdsHelper<10>(pIds, pIdsSrc1, pIdsDst, pExpertBounds, pNExperts, pNTokens, pNExpertUsed, pNchannelsY, pSi1,
                           pSis1, pWriteInverse, pStream);
    break;
  case 16:
    fLaunchMmIdsHelper<16>(pIds, pIdsSrc1, pIdsDst, pExpertBounds, pNExperts, pNTokens, pNExpertUsed, pNchannelsY, pSi1,
                           pSis1, pWriteInverse, pStream);
    break;
  case 32:
    fLaunchMmIdsHelper<32>(pIds, pIdsSrc1, pIdsDst, pExpertBounds, pNExperts, pNTokens, pNExpertUsed, pNchannelsY, pSi1,
                           pSis1, pWriteInverse, pStream);
    break;
  default:
    fLaunchMmIdsHelper<0>(pIds, pIdsSrc1, pIdsDst, pExpertBounds, pNExperts, pNTokens, pNExpertUsed, pNchannelsY, pSi1,
                          pSis1, pWriteInverse, pStream);
    break;
  }
}
