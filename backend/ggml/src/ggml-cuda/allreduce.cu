#include "allreduce.cuh"

#if !defined(GGML_USE_HIP)

#include "convert.cuh"
#include "ggml-impl.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <limits>

// ---------------------------------------------------------------------------
// CUDA AllReduce for tensor-parallel inference across two GPUs.
//
// Provides an in-place sum reduction over matching tensors on two CUDA
// devices in the same process.  Used by the tensor-split path alongside
// NCCL; targets setups without NVLink, where data is exchanged between the
// GPUs by staging it through pinned host memory over PCIe.
//
// Two reduction strategies are selected per call by tensor size:
//
//   * Chunked kernel path (small reductions): a single CUDA kernel both
//     stages data through pinned host memory and performs the local sum.
//     Cross-GPU synchronization happens *inside the kernel* (busy-wait on
//     a host-memory flag), which keeps launch overhead low for the
//     latency-sensitive token-generation case.
//
//   * Copy-engine path (large reductions): the transfer is split into
//     D2H + H2D cudaMemcpyAsync chunks driven by the GPU's copy engine,
//     followed by a small device-side add kernel.  Cross-GPU
//     synchronization happens *outside the kernel*, via CUDA events
//     between streams.  This keeps the compute engine free while large
//     transfers are in flight, which matters for prefill-sized tensors.
//     Reductions larger than the per-call inner cap are processed by an
//     outer chunker that issues sequential inner calls.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Cross-GPU signal mechanism
//
// One int per (slot, rank) pair in pinned host memory.  Each AR call writes a
// strictly increasing token (= the AR call number) into its own arrival int.
// The peer spins until its read of the other's arrival int equals the token
// it expects for this call -- a mismatch means the peer hasn't arrived yet.
// Tokens never repeat over realistic call rates (32-bit int wraps in tens of
// days at thousands of ARs/sec), so arrival ints don't need to be reset
// between calls; we initialize once at pipeline init and let the values
// accumulate.
//
// There is exactly one writer (the owning GPU) and one reader (the peer), so
// we don't need atomics.  A volatile store paired with __threadfence_system()
// provides the release ordering that makes the D2H writes visible system-wide
// before the arrival token is observed.
//
// atomicAdd_system() requires hostNativeAtomicSupported, which is unavailable
// on PCIe-attached consumer GPUs without NVLink, so the volatile path is the
// portable choice.
// ---------------------------------------------------------------------------

static __device__ __forceinline__ void fGgmlCudaArSignalSet(int *p_, int pToken) { *(volatile int *)p_ = pToken; }
static __device__ __forceinline__ int fGgmlCudaArSignalGet(const int *p_) { return *(const volatile int *)p_; }

// Byte spacing between adjacent arrival ints.  64 bytes (one cache line)
// ensures each GPU/block's arrival slot lives on its own line, preventing
// false-sharing stalls on the polling GPU.
static constexpr size_t GGML_CUDA_AR_ARRIVAL_STRIDE = 64;

// Number of blocks the chunked kernel launches with.  Each block stripes a
// disjoint slice of the data and synchronizes through its own arrival-token
// slot so multiple SMs can pump PCIe stores in parallel.
static constexpr int GGML_CUDA_AR_KERNEL_BLOCKS = 8;

// ---------------------------------------------------------------------------
// Chunked kernel AllReduce -- 2 GPUs, supports float, half, and bfloat16.
//
// Both GPUs run this kernel simultaneously on independent streams.  sendbuf
// and recvbuf live in T_dst (the caller's tensor type); host_mine / host_other
// carry data in T_wire (the on-wire type, possibly narrower than T_dst -- e.g.
// T_dst=F32 with T_wire=BF16 halves the bytes pushed across PCIe).  When
// T_dst == T_wire the casts below are no-ops.
//
// Each GPU runs three phases:
//
//   Phase 1 (all threads): cast sendbuf (T_dst) -> T_wire and store as
//                          single-instruction-width vectors into host_mine.
//                          __threadfence_system() commits these writes to host
//                          memory.
//   Phase 2 (thread 0):    write token to arrival_mine; spin until
//                          arrival_other == token.
//   Phase 3 (all threads): read T_wire vectors from host_other, cast
//                          each element to T_dst, and sum with the local
//                          sendbuf value (also rounded through T_wire so that
//                          both GPUs truncate identically -- this guarantees
//                          bit-equivalent results across the two devices).
//
// Multi-block: blocks stripe vectors across (gridDim.x * blockDim.x) global
// threads to keep multiple SMs issuing PCIe stores in parallel.  Each block
// has its own arrival-token slot (offset by blockIdx.x * ARRIVAL_STRIDE);
// thread 0 of each block signals/spins on that slot independently of other
// blocks.  Tail elements (the leftover < ELEMS_PER_VEC at the end) are
// handled only by block 0 to avoid cross-block writes to the same slots.
// ---------------------------------------------------------------------------
template <typename T_dst, typename T_wire>
static __global__ void ggml_cuda_ar_kernel(const T_dst *pSendbuf, T_dst *pRecvbuf, T_wire *__restrict__ pHostMine,
                                           const T_wire *__restrict__ pHostOther, int pCount, int *pArrivalMine,
                                           int *pArrivalOther, int pToken) {

  // Vector unit for the wire type, sized to the arch's widest single-instruction
  // copy (16 B on Volta+).  Each phase-1 iter writes one vector to host memory;
  // each phase-3 iter reads one and produces ELEMS_PER_VEC sums.
  constexpr int cElemsPerVec = ggml_cuda_get_max_cpy_bytes() / sizeof(T_wire);
  constexpr int cArrivalInts = (int)(GGML_CUDA_AR_ARRIVAL_STRIDE / sizeof(int));

  const int cTid = threadIdx.x;
  const int cNt = blockDim.x;
  const int cBid = blockIdx.x;
  const int cGtid = cBid * cNt + cTid;
  const int cGnt = gridDim.x * cNt;
  const int cCountVec = pCount / cElemsPerVec;
  const int cTail = cCountVec * cElemsPerVec;

  // Phase 1: cast sendbuf (T_dst) -> host_mine (T_wire) and store as vectors.
  {
    for (int vI = cGtid; vI < cCountVec; vI += cGnt) {
      const int cOff = vI * cElemsPerVec;
      T_wire aWire[cElemsPerVec];
#pragma unroll
      for (int vK = 0; vK < cElemsPerVec; ++vK) {
        aWire[vK] = ggml_cuda_cast<T_wire>(pSendbuf[cOff + vK]);
      }
      ggml_cuda_memcpy_1<sizeof(aWire)>(&pHostMine[cOff], aWire);
    }
    if (cBid == 0 && cTid < pCount - cTail) {
      pHostMine[cTail + cTid] = ggml_cuda_cast<T_wire>(pSendbuf[cTail + cTid]);
    }
  }

  // Commit this block's host writes before signalling.
  __threadfence_system();
  __syncthreads();

  // Phase 2: thread 0 of each block signals on its own arrival slot, then
  // spins for the matching slot from peer.  Per-block tokens mean blocks
  // proceed independently -- no inter-block barrier needed.
  if (cTid == 0) {
    int *vMySlot = pArrivalMine + cBid * cArrivalInts;
    const int *vOtherSlot = pArrivalOther + cBid * cArrivalInts;

    fGgmlCudaArSignalSet(vMySlot, pToken);
    __threadfence_system(); // make our signal visible system-wide

    while (fGgmlCudaArSignalGet(vOtherSlot) != pToken) {
#if __CUDA_ARCH__ >= GGML_CUDA_CC_VOLTA
      __nanosleep(100);
#else
      NO_DEVICE_CODE;
#endif // __CUDA_ARCH__ >= GGML_CUDA_CC_VOLTA
    }
  }

  __syncthreads();

  // Acquire peer's host_other writes (this block's stripe of them).
  __threadfence_system();

  // Phase 3: read peer's T_wire vector, cast both sides through T_wire for
  // bit-equivalence, sum in T_dst precision, and write back to recvbuf.
  {
    for (int vI = cGtid; vI < cCountVec; vI += cGnt) {
      const int cOff = vI * cElemsPerVec;
      T_wire aWire[cElemsPerVec];
      ggml_cuda_memcpy_1<sizeof(aWire)>(aWire, &pHostOther[cOff]);
#pragma unroll
      for (int vK = 0; vK < cElemsPerVec; ++vK) {
        const T_wire cDLow = ggml_cuda_cast<T_wire>(pSendbuf[cOff + vK]);
        pRecvbuf[cOff + vK] = ggml_cuda_cast<T_dst>(ggml_cuda_cast<float>(cDLow) + ggml_cuda_cast<float>(aWire[vK]));
      }
    }
    if (cBid == 0 && cTid < pCount - cTail) {
      const T_wire cDLow = ggml_cuda_cast<T_wire>(pSendbuf[cTail + cTid]);
      pRecvbuf[cTail + cTid] =
          ggml_cuda_cast<T_dst>(ggml_cuda_cast<float>(cDLow) + ggml_cuda_cast<float>(pHostOther[cTail + cTid]));
    }
  }
}

// Combined load-convert-add kernel.  The peer's contribution arrives as T_src
// (which may be a lower-precision type than T_dst when the BF16 round-trip is
// active).  For bit-equivalence between the two GPUs, dst is first rounded
// through T_src's precision via ggml_cuda_cast -- peer already truncated its
// own value the same way before sending -- so both sides perform identical
// arithmetic.  When T_dst == T_src the round-trip cast is a no-op.
template <typename T_dst, typename T_src>
static __global__ void fGgmlCudaArAddKernel(T_dst *__restrict__ pDst, const T_src *__restrict__ pSrc, int pCount) {
  const int cTid = blockIdx.x * blockDim.x + threadIdx.x;
  const int cNt = gridDim.x * blockDim.x;
  for (int vI = cTid; vI < pCount; vI += cNt) {
    const T_src cDLow = ggml_cuda_cast<T_src>(pDst[vI]);
    pDst[vI] = ggml_cuda_cast<T_dst>(ggml_cuda_cast<float>(cDLow) + ggml_cuda_cast<float>(pSrc[vI]));
  }
}

// ---------------------------------------------------------------------------
// Pipeline structure
// ---------------------------------------------------------------------------

// Number of slots in the event / arrival ring.  Two slots is sufficient:
// lockstep guarantees the two GPUs are at most one AR (or chunk) apart, so
// slot[N%2] is always safe to reuse -- peer has already consumed slot[N%2]
// from AR N-2 by the time we get to AR N.  acquire_slot's
// cudaEventSynchronize on ev.ker for both devices makes that consumption
// explicit before we overwrite host_buf[slot] for the new AR.
static constexpr int GGML_CUDA_AR_POOL_SIZE = 2;

// Maximum chunk size (bytes per GPU) handled by one chunked kernel launch.
// Larger tensors are reduced by issuing multiple chunked launches.
static constexpr size_t GGML_CUDA_AR_MAX_BYTES = 1024 * 1024; // 1 MB

// Copy-engine path: largest tensor accepted on this path; sets host_large /
// dev_tmp allocation size.
static constexpr size_t GGML_CUDA_AR_COPY_MAX_BYTES = 32 * 1024 * 1024; // 32 MB

// AR wire size at which the copy-engine path takes over from the chunked-
// kernel path.  Override via GGML_CUDA_AR_COPY_THRESHOLD.
static constexpr size_t GGML_CUDA_AR_COPY_THRESHOLD_DEFAULT = 1024 * 1024; // 1 MB
// Per-call CE chunk-size heuristic: chunk_bytes = clamp(nbytes / 4, MIN, MAX).
// The /4 keeps ~4 chunks in flight at any moment (good D2H/H2D overlap with
// the peer); the clamps cover the cases where nbytes/4 is too small (per-
// memcpy fixed cost dominates) or too large (chunk-level pipelining stalls).
// Env var GGML_CUDA_AR_COPY_CHUNK_BYTES can override with a fixed value.
static constexpr size_t GGML_CUDA_AR_COPY_CHUNK_BYTES_HEURISTIC_MIN = 512 * 1024;      // 512 KB
static constexpr size_t GGML_CUDA_AR_COPY_CHUNK_BYTES_HEURISTIC_MAX = 2 * 1024 * 1024; // 2 MB
// Absolute floor that an env-var override is allowed to set; this caps the
// per-slot copy-event array.  256 KB -> up to 128 chunks per 32 MB tensor.
static constexpr size_t GGML_CUDA_AR_COPY_CHUNK_BYTES_MIN = 256 * 1024;
static constexpr int GGML_CUDA_AR_COPY_MAX_CHUNKS = static_cast<int>(
    (GGML_CUDA_AR_COPY_MAX_BYTES + GGML_CUDA_AR_COPY_CHUNK_BYTES_MIN - 1) / GGML_CUDA_AR_COPY_CHUNK_BYTES_MIN);

struct ggml_cuda_ar_event_slot {
  cudaEvent_t vApp = nullptr;                          // upstream computation complete
  cudaEvent_t vCpy[GGML_CUDA_AR_COPY_MAX_CHUNKS] = {}; // copy-engine D2H chunks complete
  cudaEvent_t vH2d = nullptr; // copy-engine H2Ds complete (handoff AR stream -> compute stream)
  cudaEvent_t vKer = nullptr; // AllReduce kernel complete
};

// Mapped pinned host allocation: cudaHostAlloc + cudaHostGetDevicePointer
// in one place, with the host handle preserved for cudaFreeHost.  Used where
// the CPU never touches the buffer -- only the device reads/writes via the
// mapped device pointer.  Required on systems where cudaDevAttrCanUseHost-
// PointerForRegisteredMem is 0 and the host pointer can't be used as a
// device pointer.
struct ggml_cuda_ar_host_mapping {
  uint8_t *vHost = nullptr; // cudaFreeHost handle; also the H-side ptr for cudaMemcpyAsync
  uint8_t *dev = nullptr;   // device-side pointer for kernels / cudaMemset

  cudaError_t mAlloc(size_t pBytes) {
    cudaError_t vRc =
        cudaHostAlloc(reinterpret_cast<void **>(&vHost), pBytes, cudaHostAllocPortable | cudaHostAllocMapped);
    if (vRc != cudaSuccess) {
      vHost = nullptr;
      return vRc;
    }
    vRc = cudaHostGetDevicePointer(reinterpret_cast<void **>(&dev), vHost, 0);
    if (vRc != cudaSuccess) {
      cudaFreeHost(vHost);
      vHost = nullptr;
      dev = nullptr;
    }
    return vRc;
  }

  void mFree() {
    if (vHost) {
      cudaFreeHost(vHost);
      vHost = nullptr;
      dev = nullptr;
    }
  }
};

struct ggml_cuda_ar_pipeline {
  int vNDevices;
  int vDevices[GGML_CUDA_MAX_DEVICES];
  size_t buf_bytes;  // bytes per device in host_buf[]
  size_t vCopyBytes; // bytes per device in host_large[] / dev_tmp[]
  size_t vCopyThreshold;
  size_t vCopyChunkBytes;
  size_t vBf16Threshold; // tensors >= this size (bytes) are reduced via FP32->BF16 round-trip; 0 disables
  uint64_t vCallCount;

  // Per-device resources.
  ggml_cuda_ar_host_mapping host_buf[GGML_CUDA_MAX_DEVICES];   // pinned staging (chunked kernel)
  ggml_cuda_ar_host_mapping vHostLarge[GGML_CUDA_MAX_DEVICES]; // pinned staging (copy-engine)
  char *vDevTmp[GGML_CUDA_MAX_DEVICES];                        // device scratch for copy-engine path
  cudaStream_t vStreams[GGML_CUDA_MAX_DEVICES];                // non-blocking
  ggml_cuda_ar_event_slot vEvPool[GGML_CUDA_MAX_DEVICES][GGML_CUDA_AR_POOL_SIZE];

  // Copy-engine: per-device "I finished reading my peer's host_large"
  // event.  Indexed by RECORDER device.  Recorded same-device on streams[i]
  // after stage 2's last H2D from host_large[peer].  Waited cross-device
  // by peer's stage-1 stream before the next AR overwrites host_large[peer].
  cudaEvent_t vHostLargeReadDone[GGML_CUDA_MAX_DEVICES];
  bool vHostLargeReadDoneValid;

  // Copy-engine: per-device "my add_kernel is done with dev_tmp" event.
  // Recorded on the compute stream after each add_kernel; the AR stream
  // waits on it before the next copy_impl's H2D overwrites dev_tmp.  Lets us
  // single-buffer dev_tmp despite add_kernel running on a separate stream.
  cudaEvent_t vDevTmpKernelDone[GGML_CUDA_MAX_DEVICES];
  bool vDevTmpKernelDoneValid;

  // Arrival ring: ARRIVAL_STRIDE bytes between adjacent ints.  Mapped pinned
  // memory; CPU never reads/writes -- only the kernel and cudaMemset.
  // Use ggml_cuda_ar_arrival_ptr() to index.
  ggml_cuda_ar_host_mapping vArrival;
};

// Base pointer for the (slot, rank) per-block token block.  The kernel adds
// blockIdx.x * (ARRIVAL_STRIDE/sizeof(int)) internally to land on its own slot.
static int *ggml_cuda_ar_arrival_ptr(const ggml_cuda_ar_pipeline *p_, int pSlot, int pRank) {
  const size_t cOffset =
      ((size_t)pSlot * p_->vNDevices + pRank) * GGML_CUDA_AR_KERNEL_BLOCKS * GGML_CUDA_AR_ARRIVAL_STRIDE;
  return reinterpret_cast<int *>(p_->vArrival.dev + cOffset);
}

static uint64_t fGgmlCudaArEnvU64(const char *pName, uint64_t pDefaultValue) {
  const char *vValue = getenv(pName);
  if (vValue == nullptr || vValue[0] == '\0') {
    return pDefaultValue;
  }

  char *vEnd = nullptr;
  const unsigned long long cParsed = strtoull(vValue, &vEnd, 10);
  return vEnd != vValue ? (uint64_t)cParsed : pDefaultValue;
}

struct ggml_cuda_ar_slot_info {
  int vSlot;
  int vToken;
};

static ggml_cuda_ar_slot_info fGgmlCudaArAcquireSlot(ggml_cuda_ar_pipeline *p_) {
  const int cSlot = static_cast<int>(p_->vCallCount % GGML_CUDA_AR_POOL_SIZE);
  const bool cPoolLapped = p_->vCallCount >= GGML_CUDA_AR_POOL_SIZE;
  p_->vCallCount++;

  if (cPoolLapped) {
    for (int vI = 0; vI < p_->vNDevices; ++vI) {
      ggml_cuda_set_device(p_->vDevices[vI]);
      CUDA_CHECK(cudaEventSynchronize(p_->vEvPool[vI][cSlot].vKer));
    }
  }

  return {cSlot, (int)p_->vCallCount};
}

// Per-AR copy-engine chunk size: env-var override if set, else heuristic
// (clamp(nbytes/4, HEURISTIC_MIN, HEURISTIC_MAX)).
static size_t fGgmlCudaArChunkBytes(const ggml_cuda_ar_pipeline *p_, size_t pNbytes) {
  if (p_->vCopyChunkBytes > 0) {
    return p_->vCopyChunkBytes;
  }
  return std::min(GGML_CUDA_AR_COPY_CHUNK_BYTES_HEURISTIC_MAX,
                  std::max(GGML_CUDA_AR_COPY_CHUNK_BYTES_HEURISTIC_MIN, pNbytes / 4));
}

static void fGgmlCudaArWaitForCompute(ggml_cuda_ar_pipeline *p_, ggml_backend_cuda_context *pCudaCtx, int pRank,
                                      int pSlot) {
  ggml_cuda_ar_event_slot &vEv = p_->vEvPool[pRank][pSlot];
  CUDA_CHECK(cudaEventRecord(vEv.vApp, pCudaCtx->stream()));
  CUDA_CHECK(cudaStreamWaitEvent(p_->vStreams[pRank], vEv.vApp));
}

// ---------------------------------------------------------------------------
// Init / free
// ---------------------------------------------------------------------------

ggml_cuda_ar_pipeline *ggml_cuda_ar_pipeline_init(const int *pDevices, size_t pNDevices) {

  if (pNDevices != 2) {
    GGML_LOG_DEBUG("%s: internal AllReduce only supports n_devices=2 (got %zu); "
                   "falling back\n",
                   __func__, pNDevices);
    return nullptr;
  }

  // The chunked kernel uses __nanosleep, which is sm70+ (Volta+).
  for (size_t vI = 0; vI < pNDevices; ++vI) {
    const int cCc = ggml_cuda_info().devices[pDevices[vI]].cc;
    if (cCc < GGML_CUDA_CC_VOLTA) {
      GGML_LOG_DEBUG("%s: internal AllReduce requires compute capability >= %d "
                     "(device %d has cc=%d); falling back\n",
                     __func__, GGML_CUDA_CC_VOLTA, pDevices[vI], cCc);
      return nullptr;
    }
  }

  auto *vP = new ggml_cuda_ar_pipeline{};
  vP->vNDevices = pNDevices;
  vP->vCopyBytes = GGML_CUDA_AR_COPY_MAX_BYTES;
  vP->vCopyThreshold = fGgmlCudaArEnvU64("GGML_CUDA_AR_COPY_THRESHOLD", GGML_CUDA_AR_COPY_THRESHOLD_DEFAULT);
  // 0 = use the per-call heuristic (default).  Non-zero env value forces a
  // fixed chunk size for diagnostics, with a floor at COPY_CHUNK_BYTES_MIN.
  vP->vCopyChunkBytes = fGgmlCudaArEnvU64("GGML_CUDA_AR_COPY_CHUNK_BYTES", 0);
  if (vP->vCopyChunkBytes > 0 && vP->vCopyChunkBytes < GGML_CUDA_AR_COPY_CHUNK_BYTES_MIN) {
    GGML_LOG_WARN("%s: GGML_CUDA_AR_COPY_CHUNK_BYTES=%zu below minimum %zu; clamping\n", __func__, vP->vCopyChunkBytes,
                  GGML_CUDA_AR_COPY_CHUNK_BYTES_MIN);
    vP->vCopyChunkBytes = GGML_CUDA_AR_COPY_CHUNK_BYTES_MIN;
  }
  // Default 1: BF16 round-trip is always on for F32 inputs (any non-zero
  // ne).  Set GGML_CUDA_AR_BF16_THRESHOLD=0 to disable, or to a larger
  // byte threshold to opt out for small tensors.
  vP->vBf16Threshold = fGgmlCudaArEnvU64("GGML_CUDA_AR_BF16_THRESHOLD", 1);
  for (size_t vI = 0; vI < pNDevices; ++vI) {
    vP->vDevices[vI] = pDevices[vI];
  }

  // Per-device streams and event pools.
  for (size_t vI = 0; vI < pNDevices; ++vI) {
    ggml_cuda_set_device(vP->vDevices[vI]);

    cudaStream_t vStream = nullptr;
    if (cudaStreamCreateWithFlags(&vStream, cudaStreamNonBlocking) != cudaSuccess) {
      GGML_LOG_ERROR("%s: cudaStreamCreateWithFlags failed for device %d\n", __func__, vP->vDevices[vI]);
      ggml_cuda_ar_pipeline_free(vP);
      return nullptr;
    }
    vP->vStreams[vI] = vStream;

    for (int vS = 0; vS < GGML_CUDA_AR_POOL_SIZE; ++vS) {
      bool vOk = cudaEventCreateWithFlags(&vP->vEvPool[vI][vS].vApp, cudaEventDisableTiming) == cudaSuccess &&
                 cudaEventCreateWithFlags(&vP->vEvPool[vI][vS].vH2d, cudaEventDisableTiming) == cudaSuccess &&
                 cudaEventCreateWithFlags(&vP->vEvPool[vI][vS].vKer, cudaEventDisableTiming) == cudaSuccess;
      for (int vC = 0; vOk && vC < GGML_CUDA_AR_COPY_MAX_CHUNKS; ++vC) {
        vOk = cudaEventCreateWithFlags(&vP->vEvPool[vI][vS].vCpy[vC], cudaEventDisableTiming) == cudaSuccess;
      }
      if (!vOk) {
        GGML_LOG_ERROR("%s: cudaEventCreate failed for device %d slot %d\n", __func__, vP->vDevices[vI], vS);
        ggml_cuda_ar_pipeline_free(vP);
        return nullptr;
      }
    }

    if (cudaEventCreateWithFlags(&vP->vHostLargeReadDone[vI], cudaEventDisableTiming) != cudaSuccess) {
      GGML_LOG_ERROR("%s: cudaEventCreate for host_large_read_done failed for device %d\n", __func__, vP->vDevices[vI]);
      ggml_cuda_ar_pipeline_free(vP);
      return nullptr;
    }
    if (cudaEventCreateWithFlags(&vP->vDevTmpKernelDone[vI], cudaEventDisableTiming) != cudaSuccess) {
      GGML_LOG_ERROR("%s: cudaEventCreate for dev_tmp_kernel_done failed for device %d\n", __func__, vP->vDevices[vI]);
      ggml_cuda_ar_pipeline_free(vP);
      return nullptr;
    }
  }

  // Arrival ring: cache-line padded so each GPU's int is on its own line.
  const size_t cArrivalBytes =
      (size_t)GGML_CUDA_AR_POOL_SIZE * pNDevices * GGML_CUDA_AR_KERNEL_BLOCKS * GGML_CUDA_AR_ARRIVAL_STRIDE;
  if (vP->vArrival.mAlloc(cArrivalBytes) != cudaSuccess) {
    GGML_LOG_ERROR("%s: alloc for arrival ring failed (%zu bytes)\n", __func__, cArrivalBytes);
    ggml_cuda_ar_pipeline_free(vP);
    return nullptr;
  }
  ggml_cuda_set_device(vP->vDevices[0]);
  if (cudaMemset(vP->vArrival.dev, 0, cArrivalBytes) != cudaSuccess) {
    GGML_LOG_ERROR("%s: cudaMemset for arrival ring failed (%zu bytes)\n", __func__, cArrivalBytes);
    ggml_cuda_ar_pipeline_free(vP);
    return nullptr;
  }

  // Per-device pinned staging buffers -- POOL_SIZE-deep ring so the chunked-
  // kernel can write the next slot's data while the peer is still reading
  // the previous slot's. Indexed by (slot * buf_bytes) at the call site.
  vP->buf_bytes = GGML_CUDA_AR_MAX_BYTES;
  const size_t cHostBufTotal = (size_t)GGML_CUDA_AR_POOL_SIZE * vP->buf_bytes;
  for (size_t vI = 0; vI < pNDevices; ++vI) {
    if (vP->host_buf[vI].mAlloc(cHostBufTotal) != cudaSuccess) {
      GGML_LOG_ERROR("%s: alloc for staging failed (%zu bytes)\n", __func__, cHostBufTotal);
      ggml_cuda_ar_pipeline_free(vP);
      return nullptr;
    }
  }

  // Copy-engine path: pinned host staging + device scratch, sized for the
  // largest tensor we accept on this path (GGML_CUDA_AR_COPY_MAX_BYTES).
  // dev_tmp is single-buffered; cross-AR safety is enforced by an explicit
  // cross-stream wait in copy_impl on the prior AR's add_kernel-done event.
  for (size_t vI = 0; vI < pNDevices; ++vI) {
    ggml_cuda_set_device(vP->vDevices[vI]);
    if (vP->vHostLarge[vI].mAlloc(vP->vCopyBytes) != cudaSuccess) {
      GGML_LOG_ERROR("%s: alloc for large staging failed (%zu bytes)\n", __func__, vP->vCopyBytes);
      ggml_cuda_ar_pipeline_free(vP);
      return nullptr;
    }
    if (cudaMalloc(reinterpret_cast<void **>(&vP->vDevTmp[vI]), vP->vCopyBytes) != cudaSuccess) {
      GGML_LOG_ERROR("%s: cudaMalloc for copy scratch failed (%zu bytes) on device %d\n", __func__, vP->vCopyBytes,
                     vP->vDevices[vI]);
      ggml_cuda_ar_pipeline_free(vP);
      return nullptr;
    }
  }

  GGML_LOG_INFO("%s: initialized AllReduce pipeline: %zu GPUs, "
                "%zu KB chunked kernel staging + %zu MB copy-engine staging per GPU\n",
                __func__, pNDevices, vP->buf_bytes >> 10, vP->vCopyBytes >> 20);

  return vP;
}

void ggml_cuda_ar_pipeline_free(ggml_cuda_ar_pipeline *p_) {
  if (!p_) {
    return;
  }

  // Drain all in-flight kernels before tearing down resources.
  for (int vI = 0; vI < p_->vNDevices; ++vI) {
    if (p_->vStreams[vI]) {
      ggml_cuda_set_device(p_->vDevices[vI]);
      cudaStreamSynchronize(p_->vStreams[vI]);
    }
  }

  for (int vI = 0; vI < p_->vNDevices; ++vI) {
    p_->host_buf[vI].mFree();
    p_->vHostLarge[vI].mFree();
    if (p_->vDevTmp[vI]) {
      ggml_cuda_set_device(p_->vDevices[vI]);
      cudaFree(p_->vDevTmp[vI]);
    }
    ggml_cuda_set_device(p_->vDevices[vI]);
    for (int vS = 0; vS < GGML_CUDA_AR_POOL_SIZE; ++vS) {
      if (p_->vEvPool[vI][vS].vApp) {
        cudaEventDestroy(p_->vEvPool[vI][vS].vApp);
      }
      for (int vC = 0; vC < GGML_CUDA_AR_COPY_MAX_CHUNKS; ++vC) {
        if (p_->vEvPool[vI][vS].vCpy[vC]) {
          cudaEventDestroy(p_->vEvPool[vI][vS].vCpy[vC]);
        }
      }
      if (p_->vEvPool[vI][vS].vH2d) {
        cudaEventDestroy(p_->vEvPool[vI][vS].vH2d);
      }
      if (p_->vEvPool[vI][vS].vKer) {
        cudaEventDestroy(p_->vEvPool[vI][vS].vKer);
      }
    }
    if (p_->vHostLargeReadDone[vI]) {
      ggml_cuda_set_device(p_->vDevices[vI]);
      cudaEventDestroy(p_->vHostLargeReadDone[vI]);
    }
    if (p_->vDevTmpKernelDone[vI]) {
      ggml_cuda_set_device(p_->vDevices[vI]);
      cudaEventDestroy(p_->vDevTmpKernelDone[vI]);
    }
    if (p_->vStreams[vI]) {
      ggml_cuda_set_device(p_->vDevices[vI]);
      cudaStreamDestroy(p_->vStreams[vI]);
    }
  }
  p_->vArrival.mFree();
  delete p_;
}

// ---------------------------------------------------------------------------
// Dispatch
// ---------------------------------------------------------------------------

// Asymmetric copy_impl: data sent over PCIe in T_src precision (one element of
// nbytes per ne element); accumulated locally into a T_dst buffer.  When
// T_src == T_dst this is the original homogeneous reduction.  When they differ
// (e.g. BF16 wire / F32 accumulator) the add kernel rounds dst through T_src
// for bit-equivalence between GPUs and we skip the otherwise-needed
// post-conversion entirely.
template <typename T_src, typename T_dst>
static bool fGgmlCudaArAllreduceCopyImpl(ggml_cuda_ar_pipeline *p_, ggml_backend_t *pBackends,
                                         T_src *const pSrcBuf[GGML_CUDA_MAX_DEVICES],
                                         T_dst *const pDstBuf[GGML_CUDA_MAX_DEVICES],
                                         const bool pCompute[GGML_CUDA_MAX_DEVICES], int64_t pNe, size_t pNbytes) {
  GGML_ASSERT(p_->vNDevices == 2);
  GGML_ASSERT(pNbytes <= p_->vCopyBytes);
  GGML_ASSERT(pNe <= std::numeric_limits<int>::max());

  const size_t cChunkBytes = fGgmlCudaArChunkBytes(p_, pNbytes);
  GGML_ASSERT(cChunkBytes > 0);

  const int cSlot = fGgmlCudaArAcquireSlot(p_).vSlot;
  const size_t cCopyChunks = (pNbytes + cChunkBytes - 1) / cChunkBytes;
  GGML_ASSERT(cCopyChunks <= GGML_CUDA_AR_COPY_MAX_CHUNKS);

  ggml_backend_cuda_context *vCudaCtx[2] = {};

  // Stage 1: both GPUs copy their local contribution to pinned host memory.
  for (int vI = 0; vI < 2; ++vI) {
    ggml_cuda_set_device(p_->vDevices[vI]);
    vCudaCtx[vI] = static_cast<ggml_backend_cuda_context *>(pBackends[vI]->context);
    GGML_ASSERT(vCudaCtx[vI]->device == p_->vDevices[vI]);

    fGgmlCudaArWaitForCompute(p_, vCudaCtx[vI], vI, cSlot);

    // Wait for peer's H2D from our host_large[i] (recorded in the
    // previous AR's stage 2) to complete before we overwrite host_large[i].
    // host_large_read_done[peer] = peer finished reading host_large[i].
    // No-op on the first AR -- no prior record exists.
    if (p_->vHostLargeReadDoneValid) {
      const int cPeer = 1 - vI;
      CUDA_CHECK(cudaStreamWaitEvent(p_->vStreams[vI], p_->vHostLargeReadDone[cPeer]));
    }

    if (!pCompute[vI]) {
      CUDA_CHECK(cudaMemsetAsync(pSrcBuf[vI], 0, pNbytes, p_->vStreams[vI]));
    }

    for (size_t vC = 0; vC < cCopyChunks; ++vC) {
      const size_t cOffset = vC * cChunkBytes;
      const size_t cThisBytes = (pNbytes - cOffset) < cChunkBytes ? (pNbytes - cOffset) : cChunkBytes;

      CUDA_CHECK(cudaMemcpyAsync(p_->vHostLarge[vI].vHost + cOffset, reinterpret_cast<char *>(pSrcBuf[vI]) + cOffset,
                                 cThisBytes, cudaMemcpyDeviceToHost, p_->vStreams[vI]));
      CUDA_CHECK(cudaEventRecord(p_->vEvPool[vI][cSlot].vCpy[vC], p_->vStreams[vI]));
    }
  }

  // Stage 2: each GPU waits for each peer D2H chunk, pulls that chunk back to
  // local device scratch (dev_tmp), then performs one device-local add over
  // the assembled peer tensor.  The H2Ds run on the AR stream (copy engine)
  // and the add_kernel runs on the caller's compute stream, so the AR stream
  // stays pure-copy and avoids an in-stream copy->compute engine switch every
  // AR.  dev_tmp is single-buffered: the AR stream waits cross-stream on the
  // prior AR's add_kernel-done event before overwriting it.
  for (int vI = 0; vI < 2; ++vI) {
    const int cPeer = 1 - vI;
    ggml_cuda_set_device(p_->vDevices[vI]);

    // Wait for the previous AR's add_kernel (on the compute stream) to
    // finish reading dev_tmp before our H2D overwrites it.  No-op on the
    // first copy_impl call.
    if (p_->vDevTmpKernelDoneValid) {
      CUDA_CHECK(cudaStreamWaitEvent(p_->vStreams[vI], p_->vDevTmpKernelDone[vI]));
    }

    for (size_t vC = 0; vC < cCopyChunks; ++vC) {
      const size_t cOffset = vC * cChunkBytes;
      const size_t cThisBytes = (pNbytes - cOffset) < cChunkBytes ? (pNbytes - cOffset) : cChunkBytes;

      CUDA_CHECK(cudaStreamWaitEvent(p_->vStreams[vI], p_->vEvPool[cPeer][cSlot].vCpy[vC]));
      CUDA_CHECK(cudaMemcpyAsync(p_->vDevTmp[vI] + cOffset, p_->vHostLarge[cPeer].vHost + cOffset, cThisBytes,
                                 cudaMemcpyHostToDevice, p_->vStreams[vI]));
    }

    // Mark our reads of host_large[peer] complete so peer's next AR can
    // safely overwrite it.
    CUDA_CHECK(cudaEventRecord(p_->vHostLargeReadDone[vI], p_->vStreams[vI]));

    // Hand off from AR stream (copy engine) to compute stream: compute
    // stream waits for all H2Ds to finish, then runs the add_kernel.
    CUDA_CHECK(cudaEventRecord(p_->vEvPool[vI][cSlot].vH2d, p_->vStreams[vI]));
    CUDA_CHECK(cudaStreamWaitEvent(vCudaCtx[vI]->stream(), p_->vEvPool[vI][cSlot].vH2d));

    const int cBlockSize = 256;
    int vNBlocks = (int)((pNe + cBlockSize - 1) / cBlockSize);
    if (vNBlocks > 1024) {
      vNBlocks = 1024;
    }
    fGgmlCudaArAddKernel<T_dst, T_src><<<vNBlocks, cBlockSize, 0, vCudaCtx[vI]->stream()>>>(
        pDstBuf[vI], reinterpret_cast<const T_src *>(p_->vDevTmp[vI]), (int)pNe);
    CUDA_CHECK(cudaGetLastError());

    // Record dev_tmp-released on the compute stream so the next copy_impl
    // can wait for the kernel to finish before overwriting dev_tmp.  Also
    // record AR-done as ev.ker for acquire_slot's pool-wraparound sync.
    CUDA_CHECK(cudaEventRecord(p_->vDevTmpKernelDone[vI], vCudaCtx[vI]->stream()));
    CUDA_CHECK(cudaEventRecord(p_->vEvPool[vI][cSlot].vKer, vCudaCtx[vI]->stream()));
  }
  p_->vHostLargeReadDoneValid = true;
  p_->vDevTmpKernelDoneValid = true;

  return true;
}

// Outer-level chunker: copy_impl handles up to copy_bytes per call (limited by
// the host_large / dev_tmp allocation size).  When the full AR exceeds that,
// slice the tensor into copy_bytes-sized pieces and call copy_impl repeatedly.
// Each slice goes through its own stage 1 -> stage 2 cycle and acquires its own
// slot, so cross-AR fences and pool wraparound work the same way as for any
// other sequence of small ARs.
template <typename T_src, typename T_dst>
static bool fGgmlCudaArAllreduceCopyOuter(ggml_cuda_ar_pipeline *p_, ggml_backend_t *pBackends,
                                          T_src *const pSrcBuf[GGML_CUDA_MAX_DEVICES],
                                          T_dst *const pDstBuf[GGML_CUDA_MAX_DEVICES],
                                          const bool pCompute[GGML_CUDA_MAX_DEVICES], int64_t pNe) {
  const int64_t cOuterMaxElems = (int64_t)(p_->vCopyBytes / sizeof(T_src));
  GGML_ASSERT(cOuterMaxElems > 0);

  bool vOk = true;
  for (int64_t vOuterStart = 0; vOuterStart < pNe && vOk; vOuterStart += cOuterMaxElems) {
    const int64_t cOuterNe = std::min(cOuterMaxElems, pNe - vOuterStart);
    const size_t cOuterNbytes = (size_t)cOuterNe * sizeof(T_src);

    T_src *aSrc[GGML_CUDA_MAX_DEVICES] = {};
    T_dst *aDst[GGML_CUDA_MAX_DEVICES] = {};
    for (int vI = 0; vI < p_->vNDevices; ++vI) {
      aSrc[vI] = pSrcBuf[vI] + vOuterStart;
      aDst[vI] = pDstBuf[vI] + vOuterStart;
    }
    vOk = fGgmlCudaArAllreduceCopyImpl<T_src, T_dst>(p_, pBackends, aSrc, aDst, pCompute, cOuterNe, cOuterNbytes);
  }
  return vOk;
}

bool ggml_cuda_ar_allreduce(ggml_cuda_ar_pipeline *p, ggml_backend_t *pBackends, ggml_tensor **pTensors) {
  GGML_ASSERT(p != nullptr);

  const int cN = p->vNDevices;
  GGML_ASSERT(cN == 2);

  const ggml_type cInputType = pTensors[0]->type;
  GGML_ASSERT(cInputType == GGML_TYPE_F32 || cInputType == GGML_TYPE_F16 || cInputType == GGML_TYPE_BF16);

  const int64_t cNe = ggml_nelements(pTensors[0]);
  GGML_ASSERT(cNe > 0);

  const size_t cInputNbytes = ggml_nbytes(pTensors[0]);

  // BF16 round-trip: F32 inputs >= bf16_threshold are converted to BF16 for
  // the reduction (chunked or copy-engine), halving on-wire bytes. Matches
  // NCCL's behaviour. The pre-conversion zeroes inactive shards so the
  // inner paths see them as already-prepared compute tensors.
  const bool cUseBf16 = cInputType == GGML_TYPE_F32 && p->vBf16Threshold > 0 && cInputNbytes >= p->vBf16Threshold;

  const ggml_type cKernelType = cUseBf16 ? GGML_TYPE_BF16 : cInputType;
  const size_t cTypeSize = ggml_type_size(cKernelType);
  GGML_ASSERT(p->buf_bytes >= cTypeSize);
  const size_t cNbytes = (size_t)cNe * cTypeSize;

  bool aComputeFlag[GGML_CUDA_MAX_DEVICES] = {};
  for (int vI = 0; vI < cN; ++vI) {
    aComputeFlag[vI] = (pTensors[vI]->flags & GGML_TENSOR_FLAG_COMPUTE) != 0;
  }

  // Decide between copy-engine and chunked kernel paths based on the working
  // type's actual byte count.  No upper bound: copy_outer slices reductions
  // larger than copy_bytes into copy_bytes-sized pieces.
  const bool cUseCopyEngine = p->vCopyThreshold > 0 && cNbytes >= p->vCopyThreshold;

  // BF16 inactive-shard zeroing: when use_bf16 is on, the combined kernel
  // (chunked kernel path) and the combined add kernel (copy_engine path)
  // both accumulate into the F32 tensor data directly, so an inactive
  // shard's accumulator must start at zero.
  if (cUseBf16) {
    for (int vI = 0; vI < cN; ++vI) {
      if (!aComputeFlag[vI]) {
        auto *vCudaCtx = static_cast<ggml_backend_cuda_context *>(pBackends[vI]->context);
        GGML_ASSERT(vCudaCtx->device == p->vDevices[vI]);
        ggml_cuda_set_device(p->vDevices[vI]);
        CUDA_CHECK(cudaMemsetAsync(pTensors[vI]->data, 0, (size_t)cNe * sizeof(float), vCudaCtx->stream()));
      }
    }
  }

  // Pre-convert F32 -> BF16 into bf16_tmp ONLY for the copy_engine + use_bf16
  // path; the chunked kernel path's combined kernel does the conversion
  // inline as it writes to host_buf.
  ggml_cuda_pool_alloc<nv_bfloat16> vBf16Tmp[GGML_CUDA_MAX_DEVICES];
  void *aCopySrcPtr[GGML_CUDA_MAX_DEVICES] = {};

  if (cUseCopyEngine && cUseBf16) {
    to_bf16_cuda_t vToBf16 = ggml_get_to_bf16_cuda(GGML_TYPE_F32);
    for (int vI = 0; vI < cN; ++vI) {
      auto *vCudaCtx = static_cast<ggml_backend_cuda_context *>(pBackends[vI]->context);
      GGML_ASSERT(vCudaCtx->device == p->vDevices[vI]);
      vBf16Tmp[vI].pool = &vCudaCtx->pool();
      vBf16Tmp[vI].alloc(cNe);
      ggml_cuda_set_device(p->vDevices[vI]);
      if (aComputeFlag[vI]) {
        vToBf16(pTensors[vI]->data, vBf16Tmp[vI].get(), cNe, vCudaCtx->stream());
        CUDA_CHECK(cudaGetLastError());
      } else {
        CUDA_CHECK(cudaMemsetAsync(vBf16Tmp[vI].get(), 0, cNbytes, vCudaCtx->stream()));
      }
      aCopySrcPtr[vI] = vBf16Tmp[vI].get();
    }
  }

  bool vOk = true;
  if (cUseCopyEngine) {
    // After up-front BF16 conversion, the tmp buffers already hold the
    // (possibly zeroed-for-inactive) data, so the inner path can treat
    // every shard as compute.
    bool aInnerCompute[GGML_CUDA_MAX_DEVICES];
    for (int vI = 0; vI < cN; ++vI) {
      aInnerCompute[vI] = cUseBf16 ? true : aComputeFlag[vI];
    }

    // Dispatch into copy_impl with explicit src/dst types.  When use_bf16
    // is on, the wire type is BF16 (src = bf16_tmp) and the accumulator
    // is F32 (dst = tensors[i]->data); the combined add kernel rounds dst
    // through BF16 for bit-equivalence and writes F32 directly, so no
    // post-conversion is needed.  Otherwise src == dst (same native type).
    if (cUseBf16) {
      GGML_ASSERT(cKernelType == GGML_TYPE_BF16);
      nv_bfloat16 *aSrc[GGML_CUDA_MAX_DEVICES] = {};
      float *aDst[GGML_CUDA_MAX_DEVICES] = {};
      for (int vI = 0; vI < cN; ++vI) {
        aSrc[vI] = static_cast<nv_bfloat16 *>(aCopySrcPtr[vI]);
        aDst[vI] = static_cast<float *>(pTensors[vI]->data);
      }
      vOk = fGgmlCudaArAllreduceCopyOuter<nv_bfloat16, float>(p, pBackends, aSrc, aDst, aInnerCompute, cNe);
    } else {
      switch (cKernelType) {
      case GGML_TYPE_F32: {
        float *aBuf[GGML_CUDA_MAX_DEVICES] = {};
        for (int vI = 0; vI < cN; ++vI) {
          aBuf[vI] = static_cast<float *>(pTensors[vI]->data);
        }
        vOk = fGgmlCudaArAllreduceCopyOuter<float, float>(p, pBackends, aBuf, aBuf, aInnerCompute, cNe);
        break;
      }
      case GGML_TYPE_BF16: {
        nv_bfloat16 *aBuf[GGML_CUDA_MAX_DEVICES] = {};
        for (int vI = 0; vI < cN; ++vI) {
          aBuf[vI] = static_cast<nv_bfloat16 *>(pTensors[vI]->data);
        }
        vOk = fGgmlCudaArAllreduceCopyOuter<nv_bfloat16, nv_bfloat16>(p, pBackends, aBuf, aBuf, aInnerCompute, cNe);
        break;
      }
      case GGML_TYPE_F16: {
        half *aBuf[GGML_CUDA_MAX_DEVICES] = {};
        for (int vI = 0; vI < cN; ++vI) {
          aBuf[vI] = static_cast<half *>(pTensors[vI]->data);
        }
        vOk = fGgmlCudaArAllreduceCopyOuter<half, half>(p, pBackends, aBuf, aBuf, aInnerCompute, cNe);
        break;
      }
      default:
        GGML_ASSERT(false);
      }
    }
  } else {
    // host_buf carries T_wire-typed data; max_chunk_elems is the count that
    // fits in one host_buf at the wire size.
    const size_t cMaxChunkElems = p->buf_bytes / cTypeSize;
    const size_t cInputTypeSize = ggml_type_size(cInputType);

    // Chunked kernel path runs entirely on the caller's compute stream:
    // since AR is a barrier here, same-stream ordering subsumes any
    // cross-stream event handshake that the copy-engine path needs, and
    // skips the cross-stream scheduling overhead that was hurting the
    // small-tensor (tg) latency on the AR-stream variant.  Only ev.ker is
    // still recorded at end-of-AR for acquire_slot's pool-wraparound check.
    for (int64_t vChunkStart = 0; vChunkStart < cNe; vChunkStart += (int64_t)cMaxChunkElems) {
      const size_t cRemainingElems = (size_t)(cNe - vChunkStart);
      const size_t chunk_elems = cRemainingElems < cMaxChunkElems ? cRemainingElems : cMaxChunkElems;
      const size_t cChunkDstBytes = chunk_elems * cInputTypeSize;

      const auto [slot, token] = fGgmlCudaArAcquireSlot(p);
      const bool cLastChunk = vChunkStart + (int64_t)chunk_elems == cNe;

      for (int i = 0; i < cN; ++i) {
        const int peer = 1 - i; // valid for n == 2 only
        ggml_cuda_set_device(p->vDevices[i]);
        auto *vCudaCtx = static_cast<ggml_backend_cuda_context *>(pBackends[i]->context);
        GGML_ASSERT(vCudaCtx->device == p->vDevices[i]);
        cudaStream_t stream = vCudaCtx->stream();

        char *data = static_cast<char *>(pTensors[i]->data) + vChunkStart * (int64_t)cInputTypeSize;

        // Match NCCL/meta-backend semantics: inactive shards contribute
        // zeros.  On the BF16 path the F32 tensor data was already
        // zeroed up-front (above), so per-chunk zeroing isn't needed.
        if (!aComputeFlag[i] && !cUseBf16) {
          CUDA_CHECK(cudaMemsetAsync(data, 0, cChunkDstBytes, stream));
        }

#define LAUNCH_AR_KERNEL(T_dst, T_wire)                                                                                \
  ggml_cuda_ar_kernel<T_dst, T_wire><<<dim3(GGML_CUDA_AR_KERNEL_BLOCKS), dim3(256), 0, stream>>>(                      \
      reinterpret_cast<const T_dst *>(data), reinterpret_cast<T_dst *>(data),                                          \
      reinterpret_cast<T_wire *>(p->host_buf[i].dev + (size_t)slot * p->buf_bytes),                                    \
      reinterpret_cast<const T_wire *>(p->host_buf[peer].dev + (size_t)slot * p->buf_bytes),                           \
      static_cast<int>(chunk_elems), ggml_cuda_ar_arrival_ptr(p, slot, i), ggml_cuda_ar_arrival_ptr(p, slot, peer),    \
      token)

        if (cUseBf16) {
          GGML_ASSERT(cInputType == GGML_TYPE_F32);
          LAUNCH_AR_KERNEL(float, nv_bfloat16);
        } else {
          switch (cInputType) {
          case GGML_TYPE_F32:
            LAUNCH_AR_KERNEL(float, float);
            break;
          case GGML_TYPE_F16:
            LAUNCH_AR_KERNEL(half, half);
            break;
          case GGML_TYPE_BF16:
            LAUNCH_AR_KERNEL(nv_bfloat16, nv_bfloat16);
            break;
          default:
            GGML_ASSERT(false);
          }
        }

#undef LAUNCH_AR_KERNEL
        CUDA_CHECK(cudaGetLastError());

        if (cLastChunk) {
          CUDA_CHECK(cudaEventRecord(p->vEvPool[i][slot].vKer, stream));
        }
      }
    }
  }

  return vOk;
}

#else // defined(GGML_USE_HIP)

// HIP lacks the host-mapped pinned-memory APIs (cudaHostAllocPortable
// / cudaHostAllocMapped / cudaHostGetDevicePointer) and __nanosleep that this
// implementation relies on, so the internal AllReduce is a CUDA-only feature.
// The dispatcher in ggml-cuda.cu treats a nullptr pipeline as "init failed"
// and silently falls back to the meta backend's generic AllReduce.
ggml_cuda_ar_pipeline *ggml_cuda_ar_pipeline_init(const int *, size_t) { return nullptr; }
void ggml_cuda_ar_pipeline_free(ggml_cuda_ar_pipeline *) {}
bool ggml_cuda_ar_allreduce(ggml_cuda_ar_pipeline *, ggml_backend_t *, ggml_tensor **) { return false; }

#endif // !defined(GGML_USE_HIP)
