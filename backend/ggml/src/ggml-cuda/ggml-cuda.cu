#include "ggml-cuda.h"
#include "ggml-impl.h"
#include "ggml-backend-impl.h"

#include "ggml-cuda/allreduce.cuh"
#include "ggml-cuda/common.cuh"
#include "ggml-cuda/acc.cuh"
#include "ggml-cuda/add-id.cuh"
#include "ggml-cuda/arange.cuh"
#include "ggml-cuda/amax-pow2.cuh"
#include "ggml-cuda/argmax.cuh"
#include "ggml-cuda/argsort.cuh"
#include "ggml-cuda/binbcast.cuh"
#include "ggml-cuda/clamp.cuh"
#include "ggml-cuda/col2im-1d.cuh"
#include "ggml-cuda/concat.cuh"
#include "ggml-cuda/conv-transpose-1d.cuh"
#include "ggml-cuda/conv2d.cuh"
#include "ggml-cuda/conv2d-dw.cuh"
#include "ggml-cuda/conv2d-transpose.cuh"
#include "ggml-cuda/convert.cuh"
#include "ggml-cuda/count-equal.cuh"
#include "ggml-cuda/cpy.cuh"
#include "ggml-cuda/cross-entropy-loss.cuh"
#include "ggml-cuda/cumsum.cuh"
#include "ggml-cuda/diagmask.cuh"
#include "ggml-cuda/diag.cuh"
#include "ggml-cuda/fattn.cuh"
#include "ggml-cuda/fwht.cuh"
#include "ggml-cuda/getrows.cuh"
#include "ggml-cuda/im2col.cuh"
#include "ggml-cuda/mmf.cuh"
#include "ggml-cuda/mmq.cuh"
#include "ggml-cuda/mmvf.cuh"
#include "ggml-cuda/mmvq.cuh"
#include "ggml-cuda/norm.cuh"
#include "ggml-cuda/opt-step-adamw.cuh"
#include "ggml-cuda/opt-step-sgd.cuh"
#include "ggml-cuda/out-prod.cuh"
#include "ggml-cuda/pad.cuh"
#include "ggml-cuda/pool2d.cuh"
#include "ggml-cuda/quantize.cuh"
#include "ggml-cuda/rope.cuh"
#include "ggml-cuda/roll.cuh"
#include "ggml-cuda/scale.cuh"
#include "ggml-cuda/snake.cuh"
#include "ggml-cuda/softcap.cuh"
#include "ggml-cuda/softmax.cuh"
#include "ggml-cuda/ssm-conv.cuh"
#include "ggml-cuda/ssm-scan.cuh"
#include "ggml-cuda/sum.cuh"
#include "ggml-cuda/sumrows.cuh"
#include "ggml-cuda/top-k.cuh"
#include "ggml-cuda/mean.cuh"
#include "ggml-cuda/tsembd.cuh"
#include "ggml-cuda/topk-moe.cuh"
#include "ggml-cuda/unary.cuh"
#include "ggml-cuda/upscale.cuh"
#include "ggml-cuda/wkv.cuh"
#include "ggml-cuda/gla.cuh"
#include "ggml-cuda/gated_delta_net.cuh"
#include "ggml-cuda/dsv4-hc.cuh"
#include "ggml-cuda/set.cuh"
#include "ggml-cuda/set-rows.cuh"
#include "ggml-cuda/pad_reflect_1d.cuh"
#include "ggml-cuda/solve_tri.cuh"
#include "ggml-cuda/tri.cuh"
#include "ggml-cuda/cumsum.cuh"
#include "ggml-cuda/fill.cuh"
#include "ggml-cuda/lightning-indexer.cuh"
#include "ggml.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <charconv>
#include <cinttypes>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cfloat>
#include <initializer_list>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

static_assert(sizeof(half) == sizeof(ggml_fp16_t), "wrong fp16 size");

#define GGML_LOG_WARN_ONCE(str)                                                                                        \
  {                                                                                                                    \
    static std::once_flag warn_flag;                                                                                   \
    std::call_once(warn_flag, []() { GGML_LOG_WARN(str); });                                                           \
  }

[[noreturn]]
void ggml_cuda_error(const char *pStmt, const char *pFunc, const char *pFile, int pLine, const char *pMsg) {
  int vId = -1; // in case cudaGetDevice fails
  (void)cudaGetDevice(&vId);

  GGML_LOG_ERROR(GGML_CUDA_NAME " error: %s\n", pMsg);
  GGML_LOG_ERROR("  current device: %d, in function %s at %s:%d\n", vId, pFunc, pFile, pLine);
  GGML_LOG_ERROR("  %s\n", pStmt);
  // abort with GGML_ABORT to get a stack trace
  GGML_ABORT(GGML_CUDA_NAME " error");
}

// map a (possibly virtual) device id to the physical CUDA device that backs it
static int fGgmlCudaGetPhysicalDevice(int pDevice) {
  const ggml_cuda_device_info &vInfo = ggml_cuda_info();
  GGML_ASSERT(pDevice >= 0 && pDevice < vInfo.device_count);
  return vInfo.devices[pDevice].physical_device;
}

// this is faster on Windows
// probably because the Windows CUDA libraries forget to make this check before invoking the drivers
void ggml_cuda_set_device(int pDevice) {
  // translate the (possibly virtual) device id to the physical CUDA device that backs it
  const int cPhysicalDevice = fGgmlCudaGetPhysicalDevice(pDevice);

  int vCurrentDevice;
  CUDA_CHECK(cudaGetDevice(&vCurrentDevice));

  if (cPhysicalDevice == vCurrentDevice) {
    return;
  }

  CUDA_CHECK(cudaSetDevice(cPhysicalDevice));
}

int ggml_cuda_get_device() {
  int vId;
  CUDA_CHECK(cudaGetDevice(&vId));
  return vId;
}

static cudaError_t fGgmlCudaDeviceMalloc(void **ptr, size_t size, int device) {
  ggml_cuda_set_device(device);
  cudaError_t vErr;
  if (getenv("GGML_CUDA_ENABLE_UNIFIED_MEMORY") != nullptr) {
    vErr = cudaMallocManaged(ptr, size);
#if defined(GGML_USE_HIP)
    if (vErr == hipSuccess) {
      // hipMemAdviseSetCoarseGrain is an optional performance hint;
      // ignore errors (e.g. hipErrorInvalidValue on some APU/iGPU configs).
      (void)cudaMemAdvise(*ptr, size, hipMemAdviseSetCoarseGrain, device);
      (void)hipGetLastError(); // clear any error
    }

    // fall back to cudaMalloc if not supported (e.g. on Windows)
    if (vErr == hipErrorNotSupported) {
      static bool warned_unsupported = false;
      if (!warned_unsupported) {
        GGML_LOG_WARN("hipMallocManaged unsupported, falling back to hipMalloc.\n");
        warned_unsupported = true;
      }

      vErr = cudaMalloc(ptr, size);
    }
#endif // defined(GGML_USE_HIP)
  } else {
    vErr = cudaMalloc(ptr, size);
  }
  return vErr;
}

#if defined(GGML_USE_HIP)
static int ggml_cuda_parse_id(char devName[]) {
  // A list of possible Target IDs can be found under the rocclr/clr repo in device.cpp
  // these values are not stable so this is susceptible to breakage
  // https://github.com/ROCm/clr/blob/amd-staging/rocclr/device/device.cpp
  int archMajor = 0x0;
  int archMinor = 0x0;
  int archNum = GGML_CUDA_CC_OFFSET_AMD;
  int archLen = strlen(devName);
  char archName[archLen + 1];

  // strip leading 'gfx' while copying into our buffer
  if (archLen > 3) {
    strcpy(archName, &devName[3]);
    archLen -= 3;
  }

  // trim trailing :xnack- or :sramecc- statuses
  archLen = strcspn(archName, ":");
  archName[archLen] = '\0';

  // tease out the version information
  if (archLen > 8) {
    // versions labeled generic use '-' as delimiter
    // strip the trailing "-generic" then iterate through what remains
    if ((strstr(archName, "-generic"))) {
      archName[archLen - 8] = '\0';
      char *pch;
      if ((pch = strtok(archName, "-"))) {
        archMajor = (int)strtoul(pch, 0, 16);
        if ((pch = strtok(NULL, "-"))) {
          archMinor = 0x10 * (int)strtoul(pch, 0, 16);
        }
      }
    }
  } else if (archLen >= 3) {
    // last two digits should be the minor * 0x10 + stepping
    archMinor = (int)strtoul(&archName[archLen - 2], 0, 16);
    archName[archLen - 2] = '\0';

    // only the major version remains
    archMajor = (int)strtoul(archName, 0, 16);
  }
  archNum += archMajor * 0x100;
  archNum += archMinor;
  return archNum;
}
#endif // defined(GGML_USE_HIP)

static ggml_cuda_device_info fGgmlCudaInit() {
  ggml_cuda_device_info vInfo = {};

  cudaError_t vErr = cudaGetDeviceCount(&vInfo.physical_device_count);
  if (vErr != cudaSuccess) {
    GGML_LOG_ERROR("%s: failed to initialize " GGML_CUDA_NAME ": %s\n", __func__, cudaGetErrorString(vErr));
    return vInfo;
  }

  GGML_ASSERT(vInfo.physical_device_count <= GGML_CUDA_MAX_DEVICES);

  // by default expose exactly the physical devices; GGML_CUDA_DEVICES can request a different
  // number of (virtual) devices to emulate multi-GPU systems on a machine with fewer GPUs
  vInfo.device_count = vInfo.physical_device_count;

  const char *vDevicesEnv = getenv("GGML_CUDA_DEVICES");
  if (vDevicesEnv != nullptr && vInfo.physical_device_count > 0) {
    const int cRequested = atoi(vDevicesEnv);
    if (cRequested > 0) {
      vInfo.device_count = cRequested;
    } else {
      GGML_LOG_WARN("%s: ignoring invalid GGML_CUDA_DEVICES=\"%s\"\n", __func__, vDevicesEnv);
    }
  }

  if (vInfo.device_count > GGML_CUDA_MAX_DEVICES) {
    GGML_LOG_WARN("%s: requested %d devices, clamping to GGML_CUDA_MAX_DEVICES=%d\n", __func__, vInfo.device_count,
                  GGML_CUDA_MAX_DEVICES);
    vInfo.device_count = GGML_CUDA_MAX_DEVICES;
  }

  // map each (virtual) device to a backing physical device (round-robin), assign each its index
  // among the (virtual) devices sharing that physical GPU, and store the per-physical share count
  int vPhysicalShareCount[GGML_CUDA_MAX_DEVICES] = {};
  GGML_ASSERT(vInfo.device_count == 0 || vInfo.physical_device_count > 0);
  for (int vId = 0; vId < vInfo.device_count; ++vId) {
    vInfo.devices[vId].physical_device = vId % vInfo.physical_device_count;
    vInfo.devices[vId].virtual_index = vPhysicalShareCount[vInfo.devices[vId].physical_device]++;
  }

  int64_t vTotalVram = 0;
  for (int vId = 0; vId < vInfo.physical_device_count; ++vId) {
    cudaDeviceProp vProp;
    CUDA_CHECK(cudaGetDeviceProperties(&vProp, vId));
    vTotalVram += vProp.totalGlobalMem;
  }
  GGML_LOG_INFO("%s: found %d " GGML_CUDA_NAME " devices (Total VRAM: %zu MiB):\n", __func__,
                vInfo.physical_device_count, (size_t)(vTotalVram / (1024 * 1024)));
  if (vInfo.device_count != vInfo.physical_device_count) {
    GGML_LOG_INFO("%s: emulating %d virtual device(s) on %d physical device(s) (GGML_CUDA_DEVICES)\n", __func__,
                  vInfo.device_count, vInfo.physical_device_count);
  }
  vTotalVram = 0;

  std::vector<std::pair<int, std::string>> vTuringDevicesWithoutMma;
  for (int vId = 0; vId < vInfo.device_count; ++vId) {
    const int cPhysicalId = vInfo.devices[vId].physical_device;

    int vDeviceVmm = 0;

#if defined(GGML_USE_VMM)
    CUdevice vDevice;
    CU_CHECK(cuDeviceGet(&vDevice, cPhysicalId));
    CU_CHECK(cuDeviceGetAttribute(&vDeviceVmm, CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED, vDevice));

    if (vDeviceVmm) {
      CUmemAllocationProp vAllocProp = {};
      vAllocProp.type = CU_MEM_ALLOCATION_TYPE_PINNED;
      vAllocProp.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
      vAllocProp.location.id = cPhysicalId;
      CU_CHECK(cuMemGetAllocationGranularity(&vInfo.devices[vId].vmm_granularity, &vAllocProp,
                                             CU_MEM_ALLOC_GRANULARITY_RECOMMENDED));
    }
#endif // defined(GGML_USE_VMM)
    vInfo.devices[vId].vmm = !!vDeviceVmm;

    cudaDeviceProp vProp;
    CUDA_CHECK(cudaGetDeviceProperties(&vProp, cPhysicalId));

    // a virtual device owns only a share of its physical GPU's memory; report that share so the
    // logged per-device VRAM sums to the physical total above.
    GGML_ASSERT(vPhysicalShareCount[cPhysicalId] > 0);
    vInfo.devices[vId].physical_share_count = vPhysicalShareCount[cPhysicalId];
    const size_t cDeviceVram = vProp.totalGlobalMem / vInfo.devices[vId].physical_share_count;
    const size_t cDeviceVramMib = cDeviceVram / (1024 * 1024);

    vInfo.default_tensor_split[vId] = vTotalVram;
    vTotalVram += cDeviceVram;
#if defined(GGML_USE_HIP)
    vInfo.devices[vId].integrated = vProp.integrated;
#else
    vInfo.devices[vId].integrated = false; // Temporarily disabled due to issues with corrupted output (e.g. #15034)
#endif
    vInfo.devices[vId].nsm = vProp.multiProcessorCount;
    vInfo.devices[vId].smpb = vProp.sharedMemPerBlock;
    vInfo.devices[vId].warp_size = vProp.warpSize;

    int vSupportsCoopLaunch = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&vSupportsCoopLaunch, cudaDevAttrCooperativeLaunch, cPhysicalId));
    vInfo.devices[vId].supports_cooperative_launch = !!vSupportsCoopLaunch;

#if defined(GGML_USE_HIP)
    vInfo.devices[vId].smpbo = vProp.sharedMemPerBlock;

    vInfo.devices[vId].cc = ggml_cuda_parse_id(vProp.gcnArchName);
    if ((vInfo.devices[vId].cc & 0xff00) == 0x0) {
      GGML_LOG_WARN("invalid architecture ID received for device %d %s: %s  cc %d.%d\n", vId, vProp.name,
                    vProp.gcnArchName, vProp.major, vProp.minor);

      // Fallback to prop.major and prop.minor
      if (vProp.major > 0) {
        vInfo.devices[vId].cc = GGML_CUDA_CC_OFFSET_AMD + vProp.major * 0x100;
        vInfo.devices[vId].cc += vProp.minor * 0x10;
      }
    }
    GGML_LOG_INFO("  Device %d: %s, %s (0x%x), VMM: %s, Wave Size: %d, VRAM: %zu MiB\n", vId, vProp.name,
                  vProp.gcnArchName, vInfo.devices[vId].cc & 0xffff, vDeviceVmm ? "yes" : "no", vProp.warpSize,
                  cDeviceVramMib);
#else
    vInfo.devices[vId].smpbo = vProp.sharedMemPerBlockOptin;
    vInfo.devices[vId].cc = 100 * vProp.major + 10 * vProp.minor;
    GGML_LOG_INFO("  Device %d: %s, compute capability %d.%d, VMM: %s, VRAM: %zu MiB\n", vId, vProp.name, vProp.major,
                  vProp.minor, vDeviceVmm ? "yes" : "no", cDeviceVramMib);
    std::string vDeviceName(vProp.name);
    if (vDeviceName == "NVIDIA GeForce MX450") {
      vTuringDevicesWithoutMma.push_back({vId, vDeviceName});
    } else if (vDeviceName == "NVIDIA GeForce MX550") {
      vTuringDevicesWithoutMma.push_back({vId, vDeviceName});
    } else if (vDeviceName.substr(0, 21) == "NVIDIA GeForce GTX 16") {
      vTuringDevicesWithoutMma.push_back({vId, vDeviceName});
    }

    // Temporary performance fix:
    // Setting device scheduling strategy for iGPUs with cc121 to "spinning" to avoid delays in cuda synchronize calls.
    // TODO: Check for future drivers the default scheduling strategy and
    // remove this call again when cudaDeviceScheduleSpin is default.
    if (vProp.major == 12 && vProp.minor == 1) {
      CUDA_CHECK(cudaSetDevice(cPhysicalId));
      CUDA_CHECK(cudaSetDeviceFlags(cudaDeviceScheduleSpin));
    }

#endif // defined(GGML_USE_HIP)
  }

  if (ggml_cuda_highest_compiled_arch(GGML_CUDA_CC_TURING) >= GGML_CUDA_CC_TURING &&
      !vTuringDevicesWithoutMma.empty()) {
    GGML_LOG_INFO("The following devices will have suboptimal performance due to a lack of tensor cores:\n");
    for (size_t vDevicePos = 0; vDevicePos < vTuringDevicesWithoutMma.size(); vDevicePos++) {
      GGML_LOG_INFO("  Device %d: %s\n", vTuringDevicesWithoutMma[vDevicePos].first,
                    vTuringDevicesWithoutMma[vDevicePos].second.c_str());
    }
    GGML_LOG_INFO("Consider compiling with CMAKE_CUDA_ARCHITECTURES=61-virtual;80-virtual and DGGML_CUDA_FORCE_MMQ to "
                  "force the use of the Pascal code for Turing.\n");
  }

  for (int vId = 0; vId < vInfo.device_count; ++vId) {
    vInfo.default_tensor_split[vId] /= vTotalVram;
  }

  // configure logging to stdout
  // CUBLAS_CHECK(cublasLoggerConfigure(1, 1, 0, nullptr));

  if (getenv("GGML_CUDA_P2P") != nullptr) {
    for (int vId = 0; vId < vInfo.physical_device_count; ++vId) {
      CUDA_CHECK(cudaSetDevice(vId));
      for (int vIdOther = 0; vIdOther < vInfo.physical_device_count; ++vIdOther) {
        if (vId == vIdOther) {
          continue;
        }
        int vCanAccessPeer;
        CUDA_CHECK(cudaDeviceCanAccessPeer(&vCanAccessPeer, vId, vIdOther));
        if (vCanAccessPeer) {
          CUDA_CHECK(cudaDeviceEnablePeerAccess(vIdOther, 0));
        }
      }
    }
  }

  return vInfo;
}

const ggml_cuda_device_info &ggml_cuda_info() {
  static ggml_cuda_device_info vInfo = fGgmlCudaInit();
  return vInfo;
}

// #define DEBUG_CUDA_MALLOC

// buffer pool for cuda (legacy)
struct ggml_cuda_pool_leg : public ggml_cuda_pool {
  static const int MAX_BUFFERS = 256;

  int device;
  struct ggml_cuda_buffer {
    void *ptr = nullptr;
    size_t size = 0;
  };

  ggml_cuda_buffer aBufferPool[MAX_BUFFERS] = {};
  size_t pool_size = 0;

  explicit ggml_cuda_pool_leg(int pDevice) : device(pDevice) {}

  ~ggml_cuda_pool_leg() {
    mClearPool();
    GGML_ASSERT(pool_size == 0);
  }

  void mClearPool() {
    ggml_cuda_set_device(device);
    for (int vI = 0; vI < MAX_BUFFERS; ++vI) {
      ggml_cuda_buffer &vB = aBufferPool[vI];
      if (vB.ptr != nullptr) {
        CUDA_CHECK(cudaFree(vB.ptr));
        pool_size -= vB.size;
        vB.ptr = nullptr;
        vB.size = 0;
      }
    }
  }

  void *alloc(size_t size, size_t *pActualSize) override {
#ifdef DEBUG_CUDA_MALLOC
    int nnz = 0;
    size_t max_size = 0;
#endif
    size_t vBestDiff = 1ull << 36;
    int vIbest = -1;
    for (int vI = 0; vI < MAX_BUFFERS; ++vI) {
      ggml_cuda_buffer &vB = aBufferPool[vI];
      if (vB.ptr != nullptr) {
#ifdef DEBUG_CUDA_MALLOC
        ++nnz;
        if (vB.size > max_size)
          max_size = vB.size;
#endif
        if (vB.size >= size) {
          size_t vDiff = vB.size - size;
          if (vDiff < vBestDiff) {
            vBestDiff = vDiff;
            vIbest = vI;
            if (!vBestDiff) {
              void *vPtr = vB.ptr;
              *pActualSize = vB.size;
              vB.ptr = nullptr;
              vB.size = 0;
              return vPtr;
            }
          }
        }
      }
    }
    if (vIbest >= 0) {
      ggml_cuda_buffer &vB = aBufferPool[vIbest];
      void *vPtr = vB.ptr;
      *pActualSize = vB.size;
      vB.ptr = nullptr;
      vB.size = 0;
      return vPtr;
    }
    void *vPtr;
    size_t vLookAheadSize = (size_t)(1.05 * size);
    vLookAheadSize = 256 * ((vLookAheadSize + 255) / 256);
    ggml_cuda_set_device(device);
    cudaError_t vErr = fGgmlCudaDeviceMalloc(&vPtr, vLookAheadSize, device);
    if (vErr == cudaErrorMemoryAllocation) {
      (void)cudaGetLastError();
      const size_t cCachedBytes = pool_size;
      GGML_LOG_DEBUG(GGML_CUDA_NAME
                     " pool[%d]: alloc of %.2f MiB failed, flushing %.2f MiB of cached buffers and retrying\n",
                     device, vLookAheadSize / 1024.0 / 1024.0, cCachedBytes / 1024.0 / 1024.0);
      CUDA_CHECK(cudaDeviceSynchronize());
      mClearPool();
      vErr = fGgmlCudaDeviceMalloc(&vPtr, vLookAheadSize, device);
      if (vErr == cudaSuccess) {
        GGML_LOG_DEBUG(GGML_CUDA_NAME " pool[%d]: retry succeeded\n", device);
      }
    }
    CUDA_CHECK(vErr);
    *pActualSize = vLookAheadSize;
    pool_size += vLookAheadSize;
#ifdef DEBUG_CUDA_MALLOC
    GGML_LOG_INFO("%s[%d]: %d buffers, max_size = %u MB, pool_size = %u MB, requested %u MB\n", __func__, device, nnz,
                  (uint32_t)(max_size / 1024 / 1024), (uint32_t)(pool_size / 1024 / 1024),
                  (uint32_t)(size / 1024 / 1024));
#endif
    return vPtr;
  }

  void free(void *pPtr, size_t pSize) override {
    for (int vI = 0; vI < MAX_BUFFERS; ++vI) {
      ggml_cuda_buffer &vB = aBufferPool[vI];
      if (vB.ptr == nullptr) {
        vB.ptr = pPtr;
        vB.size = pSize;
        return;
      }
    }
    GGML_LOG_DEBUG(GGML_CUDA_NAME " buffer pool full, increase MAX_CUDA_BUFFERS\n");
    ggml_cuda_set_device(device);
    CUDA_CHECK(cudaFree(pPtr));
    pool_size -= pSize;
  }
};

// pool with virtual memory
#if defined(GGML_USE_VMM)
struct ggml_cuda_pool_vmm : public ggml_cuda_pool {
  static const size_t CUDA_POOL_VMM_MAX_SIZE = 1ull << 35; // 32 GB

  int device;
  int vPhysicalDevice;
  CUdeviceptr vPoolAddr = 0;
  size_t vPoolUsed = 0;
  size_t pool_size = 0;
  size_t vGranularity;
#if defined(GGML_USE_HIP)
  std::vector<std::pair<CUdeviceptr, size_t>> mappings;
#endif

  explicit ggml_cuda_pool_vmm(int pDevice)
      : device(pDevice), vPhysicalDevice(fGgmlCudaGetPhysicalDevice(pDevice)),
        vGranularity(ggml_cuda_info().devices[pDevice].vmm_granularity) {}

  ~ggml_cuda_pool_vmm() {
    if (vPoolAddr != 0) {
#if defined(GGML_USE_HIP)
      // Workaround for https://github.com/ROCm/ROCR-Runtime/issues/285
      for (std::pair<CUdeviceptr, size_t> &mapping : mappings) {
        CU_CHECK(cuMemUnmap(mapping.first, mapping.second));
      }
#else
      CU_CHECK(cuMemUnmap(vPoolAddr, pool_size));
#endif
      CU_CHECK(cuMemAddressFree(vPoolAddr, CUDA_POOL_VMM_MAX_SIZE));
    }
  }

  void *alloc(size_t size, size_t *pActualSize) override {
    // round up the allocation size to the alignment to ensure that all allocations are aligned for all data types
    const size_t cAlignment = 128;
    size = cAlignment * ((size + cAlignment - 1) / cAlignment);

    size_t vAvail = pool_size - vPoolUsed;

    if (size > vAvail) {
      // round up to the next multiple of the granularity
      size_t vReserveSize = size - vAvail;
      vReserveSize = vGranularity * ((vReserveSize + vGranularity - 1) / vGranularity);

      GGML_ASSERT(pool_size + vReserveSize <= CUDA_POOL_VMM_MAX_SIZE);

      // allocate more physical memory
      CUmemAllocationProp vProp = {};
      vProp.type = CU_MEM_ALLOCATION_TYPE_PINNED;
      vProp.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
      vProp.location.id = vPhysicalDevice;
      CUmemGenericAllocationHandle vHandle;
      CU_CHECK(cuMemCreate(&vHandle, vReserveSize, &vProp, 0));

      // reserve virtual address space (if not already reserved)
      if (vPoolAddr == 0) {
        CU_CHECK(cuMemAddressReserve(&vPoolAddr, CUDA_POOL_VMM_MAX_SIZE, 0, 0, 0));
      }

      // map at the end of the pool
      CUdeviceptr vStartPtr = (CUdeviceptr)((char *)(vPoolAddr) + pool_size);
      CU_CHECK(cuMemMap(vStartPtr, vReserveSize, 0, vHandle, 0));
#if defined(GGML_USE_HIP)
      mappings.push_back({vStartPtr, vReserveSize});
#endif

      // the memory allocation handle is no longer needed after mapping
      CU_CHECK(cuMemRelease(vHandle));

      // VMM Bug fix for P2P access if GGML_CUDA_P2P is set, or if NCCL build
      bool vUsePeerAccess = getenv("GGML_CUDA_P2P") != nullptr;
#if defined(GGML_USE_NCCL)
      vUsePeerAccess = true;
#endif // defined(GGML_USE_NCCL)

      if (vUsePeerAccess) {
        // NCCL implicitly enables peer access (cudaDeviceEnablePeerAccess), and
        // GGML_CUDA_P2P enables it explicitly. Unlike cudaMalloc buffers, VMM
        // allocations do not become peer-accessible from that alone, so access
        // must be granted explicitly here. With virtual devices, grant access
        // on the backing *physical* devices (deduplicated, since several
        // virtual devices can map to the same physical GPU).
        std::vector<CUmemAccessDesc> vAccessDescs;
        bool aPhysicalSeen[GGML_CUDA_MAX_DEVICES] = {};
        const int cDeviceCount = ggml_cuda_info().device_count;
        for (int vId = 0; vId < cDeviceCount; ++vId) {
          const int cIdPhysical = fGgmlCudaGetPhysicalDevice(vId);
          if (cIdPhysical != vPhysicalDevice) {
            int vCanAccessPeer = 0;
            CUDA_CHECK(cudaDeviceCanAccessPeer(&vCanAccessPeer, cIdPhysical, vPhysicalDevice));
            if (!vCanAccessPeer) {
              continue;
            }
          }
          if (aPhysicalSeen[cIdPhysical]) {
            continue;
          }
          aPhysicalSeen[cIdPhysical] = true;
          CUmemAccessDesc vAccess = {};
          vAccess.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
          vAccess.location.id = cIdPhysical;
          vAccess.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
          vAccessDescs.push_back(vAccess);
        }
        CU_CHECK(cuMemSetAccess(vStartPtr, vReserveSize, vAccessDescs.data(), vAccessDescs.size()));
      } else {
        // set access for non P2P
        CUmemAccessDesc vAccess = {};
        vAccess.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        vAccess.location.id = vPhysicalDevice;
        vAccess.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        CU_CHECK(cuMemSetAccess(vStartPtr, vReserveSize, &vAccess, 1));
      }

      // add to the pool
      pool_size += vReserveSize;

      // printf("cuda pool[%d]: size increased to %llu MB (reserved %llu MB)\n",
      //        device, (unsigned long long) (pool_size/1024/1024),
      //        (unsigned long long) (reserve_size/1024/1024));
    }

    GGML_ASSERT(vPoolAddr != 0);

    void *ptr = (void *)((CUdeviceptr)((char *)(vPoolAddr) + vPoolUsed));
    *pActualSize = size;
    vPoolUsed += size;

#ifdef DEBUG_CUDA_MALLOC
    printf("cuda pool[%d]: allocated %llu bytes at %llx\n", device, (unsigned long long)size, ptr);
#endif

    return ptr;
  }

  void free(void *ptr, size_t size) override {
#ifdef DEBUG_CUDA_MALLOC
    printf("cuda pool[%d]: freed %llu bytes at %llx\n", device, (unsigned long long)size, ptr);
#endif

    vPoolUsed -= size;

    // all deallocations must be in reverse order of the allocations
    GGML_ASSERT(ptr == (void *)((char *)(vPoolAddr) + vPoolUsed));
  }
};
#endif // defined(GGML_USE_VMM)

std::unique_ptr<ggml_cuda_pool> ggml_backend_cuda_context::new_pool_for_device(int pDevice,
                                                                               [[maybe_unused]] int pStreamNo) {
#if defined(GGML_USE_VMM)
  if (ggml_cuda_info().devices[pDevice].vmm) {
    return std::unique_ptr<ggml_cuda_pool>(new ggml_cuda_pool_vmm(pDevice));
  }
#endif // defined(GGML_USE_VMM)
  return std::unique_ptr<ggml_cuda_pool>(new ggml_cuda_pool_leg(pDevice));
}

// destroying a cuBLAS handle while a graph is being captured in a different thread can result in a CUDA error
// this lock is used to ensure that no cuBLAS handle is destroyed while a graph is being captured

static std::mutex ggml_cuda_lock;
static std::condition_variable ggml_cuda_lock_cv;
static std::atomic<int> ggml_cuda_lock_counter;

ggml_backend_cuda_context::~ggml_backend_cuda_context() {
  std::unique_lock<std::mutex> vLock(ggml_cuda_lock);
  ggml_cuda_lock_cv.wait(vLock, [] { return ggml_cuda_lock_counter.load(std::memory_order_relaxed) == 0; });

  if (copy_event != nullptr) {
    CUDA_CHECK(cudaEventDestroy(copy_event));
  }
  for (int vI = 0; vI < GGML_CUDA_MAX_DEVICES; ++vI) {
    for (int vJ = 0; vJ < GGML_CUDA_MAX_STREAMS; ++vJ) {
      if (streams[vI][vJ] != nullptr) {
        CUDA_CHECK(cudaStreamDestroy(streams[vI][vJ]));
      }
    }
    if (cublas_handles[vI] != nullptr) {
      CUBLAS_CHECK(cublasDestroy(cublas_handles[vI]));
    }
  }
}

// cuda buffer

struct ggml_backend_cuda_buffer_context {
  int device;
  void *vDevPtr = nullptr;
  std::string name;

  ggml_backend_cuda_buffer_context(int pDevice, void *pDevPtr)
      : device(pDevice), vDevPtr(pDevPtr), name(GGML_CUDA_NAME + std::to_string(pDevice)) {}

  ~ggml_backend_cuda_buffer_context() { CUDA_CHECK(cudaFree(vDevPtr)); }
};

static void fGgmlBackendCudaBufferFreeBuffer(ggml_backend_buffer_t pBuffer) {
  ggml_backend_cuda_buffer_context *vCtx = (ggml_backend_cuda_buffer_context *)pBuffer->context;
  delete vCtx;
}

static bool fGgmlBackendBufferIsCuda(ggml_backend_buffer_t pBuffer) {
  return pBuffer->iface.free_buffer == fGgmlBackendCudaBufferFreeBuffer;
}

static void *fGgmlBackendCudaBufferGetBase(ggml_backend_buffer_t pBuffer) {
  ggml_backend_cuda_buffer_context *vCtx = (ggml_backend_cuda_buffer_context *)pBuffer->context;
  return vCtx->vDevPtr;
}

static enum ggml_status fGgmlBackendCudaBufferInitTensor(ggml_backend_buffer_t pBuffer, ggml_tensor *pTensor) {
  ggml_backend_cuda_buffer_context *vCtx = (ggml_backend_cuda_buffer_context *)pBuffer->context;

  if (pTensor->view_src != NULL) {
    assert(pTensor->view_src->buffer->buft == pBuffer->buft);
    return GGML_STATUS_SUCCESS;
  }

  if (ggml_is_quantized(pTensor->type) && pTensor->view_src == nullptr &&
      ggml_backend_buffer_get_usage(pBuffer) != GGML_BACKEND_BUFFER_USAGE_COMPUTE) {
    // initialize padding to 0 to avoid possible NaN values
    const size_t cOriginalSize = ggml_nbytes(pTensor);
    const size_t cPaddedSize = ggml_backend_buft_get_alloc_size(pBuffer->buft, pTensor);

    if (cPaddedSize > cOriginalSize) {
      ggml_cuda_set_device(vCtx->device);
      CUDA_CHECK(cudaMemset((char *)pTensor->data + cOriginalSize, 0, cPaddedSize - cOriginalSize));
    }
  }
  return GGML_STATUS_SUCCESS;
}

static void fGgmlBackendCudaBufferMemsetTensor(ggml_backend_buffer_t pBuffer, ggml_tensor *pTensor, uint8_t pValue,
                                               size_t pOffset, size_t pSize) {
  ggml_backend_cuda_buffer_context *vCtx = (ggml_backend_cuda_buffer_context *)pBuffer->context;

  ggml_cuda_set_device(vCtx->device);
  CUDA_CHECK(cudaMemsetAsync((char *)pTensor->data + pOffset, pValue, pSize, cudaStreamPerThread));
  CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
}

static void fGgmlBackendCudaBufferSetTensor(ggml_backend_buffer_t pBuffer, ggml_tensor *pTensor, const void *pData,
                                            size_t pOffset, size_t pSize) {
  ggml_backend_cuda_buffer_context *vCtx = (ggml_backend_cuda_buffer_context *)pBuffer->context;

  ggml_cuda_set_device(vCtx->device);
  CUDA_CHECK(
      cudaMemcpyAsync((char *)pTensor->data + pOffset, pData, pSize, cudaMemcpyHostToDevice, cudaStreamPerThread));
  CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
}

static void fGgmlBackendCudaBufferGetTensor(ggml_backend_buffer_t pBuffer, const ggml_tensor *pTensor, void *pData,
                                            size_t pOffset, size_t pSize) {
  ggml_backend_cuda_buffer_context *vCtx = (ggml_backend_cuda_buffer_context *)pBuffer->context;

  ggml_cuda_set_device(vCtx->device);
  CUDA_CHECK(cudaMemcpyAsync(pData, (const char *)pTensor->data + pOffset, pSize, cudaMemcpyDeviceToHost,
                             cudaStreamPerThread));
  CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
}

static void fGgmlBackendCudaBufferSetTensor2d(ggml_backend_buffer_t pBuffer, struct ggml_tensor *pTensor,
                                              const void *pData, size_t pOffset, size_t pSize, size_t pNCopies,
                                              size_t pStrideTensor, size_t pStrideData) {
  ggml_backend_cuda_buffer_context *vCtx = (ggml_backend_cuda_buffer_context *)pBuffer->context;

  ggml_cuda_set_device(vCtx->device);
  CUDA_CHECK(cudaMemcpy2DAsync((char *)pTensor->data + pOffset, pStrideTensor, pData, pStrideData, pSize, pNCopies,
                               cudaMemcpyHostToDevice, cudaStreamPerThread));
  CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
}

static void fGgmlBackendCudaBufferGetTensor2d(ggml_backend_buffer_t pBuffer, const struct ggml_tensor *pTensor,
                                              void *pData, size_t pOffset, size_t pSize, size_t pNCopies,
                                              size_t pStrideTensor, size_t pStrideData) {
  ggml_backend_cuda_buffer_context *vCtx = (ggml_backend_cuda_buffer_context *)pBuffer->context;

  ggml_cuda_set_device(vCtx->device);
  CUDA_CHECK(cudaMemcpy2DAsync(pData, pStrideData, (const char *)pTensor->data + pOffset, pStrideTensor, pSize,
                               pNCopies, cudaMemcpyDeviceToHost, cudaStreamPerThread));
  CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
}

static bool fGgmlBackendCudaBufferCpyTensor(ggml_backend_buffer_t pBuffer, const ggml_tensor *pSrc, ggml_tensor *pDst) {
  if (fGgmlBackendBufferIsCuda(pSrc->buffer)) {
    ggml_backend_cuda_buffer_context *vSrcCtx = (ggml_backend_cuda_buffer_context *)pSrc->buffer->context;
    ggml_backend_cuda_buffer_context *vDstCtx = (ggml_backend_cuda_buffer_context *)pDst->buffer->context;
    // compare the backing physical devices: distinct virtual devices may share one physical GPU,
    // in which case a same-device copy (not a peer copy) is required
    const int cSrcPhysical = fGgmlCudaGetPhysicalDevice(vSrcCtx->device);
    const int cDstPhysical = fGgmlCudaGetPhysicalDevice(vDstCtx->device);
    if (cSrcPhysical == cDstPhysical) {
      CUDA_CHECK(
          cudaMemcpyAsync(pDst->data, pSrc->data, ggml_nbytes(pSrc), cudaMemcpyDeviceToDevice, cudaStreamPerThread));
    } else {
#ifdef GGML_CUDA_NO_PEER_COPY
      return false;
#else
      CUDA_CHECK(cudaMemcpyPeerAsync(pDst->data, cDstPhysical, pSrc->data, cSrcPhysical, ggml_nbytes(pSrc),
                                     cudaStreamPerThread));
#endif
    }
    CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
    return true;
  }
  return false;

  GGML_UNUSED(pBuffer);
}

static void fGgmlBackendCudaBufferClear(ggml_backend_buffer_t pBuffer, uint8_t pValue) {
  ggml_backend_cuda_buffer_context *vCtx = (ggml_backend_cuda_buffer_context *)pBuffer->context;

  ggml_cuda_set_device(vCtx->device);
  CUDA_CHECK(cudaMemsetAsync(vCtx->vDevPtr, pValue, pBuffer->size, cudaStreamPerThread));
  CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
}

static const ggml_backend_buffer_i ggml_backend_cuda_buffer_interface = {
    /* .free_buffer     = */ fGgmlBackendCudaBufferFreeBuffer,
    /* .get_base        = */ fGgmlBackendCudaBufferGetBase,
    /* .init_tensor     = */ fGgmlBackendCudaBufferInitTensor,
    /* .memset_tensor   = */ fGgmlBackendCudaBufferMemsetTensor,
    /* .set_tensor      = */ fGgmlBackendCudaBufferSetTensor,
    /* .get_tensor      = */ fGgmlBackendCudaBufferGetTensor,
    /* .set_tensor_2d   = */ fGgmlBackendCudaBufferSetTensor2d,
    /* .get_tensor_2d   = */ fGgmlBackendCudaBufferGetTensor2d,
    /* .cpy_tensor      = */ fGgmlBackendCudaBufferCpyTensor,
    /* .clear           = */ fGgmlBackendCudaBufferClear,
    /* .reset           = */ NULL,
};

// cuda buffer type
struct ggml_backend_cuda_buffer_type_context {
  int device;
  std::string name;
};

static const char *fGgmlBackendCudaBufferTypeGetName(ggml_backend_buffer_type_t pBuft) {
  ggml_backend_cuda_buffer_type_context *vCtx = (ggml_backend_cuda_buffer_type_context *)pBuft->context;

  return vCtx->name.c_str();
}

static bool fGgmlBackendBuftIsCuda(ggml_backend_buffer_type_t pBuft) {
  return pBuft->iface.get_name == fGgmlBackendCudaBufferTypeGetName;
}

static ggml_backend_buffer_t fGgmlBackendCudaBufferTypeAllocBuffer(ggml_backend_buffer_type_t pBuft, size_t pSize) {
  ggml_backend_cuda_buffer_type_context *vBuftCtx = (ggml_backend_cuda_buffer_type_context *)pBuft->context;

  ggml_cuda_set_device(vBuftCtx->device);

  void *vDevPtr;
  cudaError_t vErr = fGgmlCudaDeviceMalloc(&vDevPtr, pSize, vBuftCtx->device);
  if (vErr != cudaSuccess) {
    // clear the error
    (void)cudaGetLastError();
    GGML_LOG_ERROR("%s: allocating %.2f MiB on device %d: cudaMalloc failed: %s\n", __func__, pSize / 1024.0 / 1024.0,
                   vBuftCtx->device, cudaGetErrorString(vErr));
    return nullptr;
  }

  ggml_backend_cuda_buffer_context *vCtx = new ggml_backend_cuda_buffer_context(vBuftCtx->device, vDevPtr);

  return ggml_backend_buffer_init(pBuft, ggml_backend_cuda_buffer_interface, vCtx, pSize);
}

static size_t fGgmlBackendCudaBufferTypeGetAlignment(ggml_backend_buffer_type_t pBuft) {
  return 128;

  GGML_UNUSED(pBuft);
}

static size_t fGgmlBackendCudaBufferTypeGetAllocSize(ggml_backend_buffer_type_t pBuft, const ggml_tensor *pTensor) {
  ggml_backend_cuda_buffer_type_context *vBuftCtx = (ggml_backend_cuda_buffer_type_context *)pBuft->context;

  size_t vSize = pTensor->op == GGML_OP_FLASH_ATTN_EXT
                     ? ggml_cuda_flash_attn_ext_get_alloc_size(vBuftCtx->device, pTensor)
                     : ggml_nbytes(pTensor);
  int64_t vNe0 = pTensor->ne[0];

  if (ggml_is_quantized(pTensor->type)) {
    if (vNe0 % MATRIX_ROW_PADDING != 0) {
      GGML_ASSERT(pTensor->nb[0] == ggml_element_size(pTensor));
      vSize += ggml_row_size(pTensor->type, MATRIX_ROW_PADDING - vNe0 % MATRIX_ROW_PADDING);
    }
  }

  return vSize;
}

static const ggml_backend_buffer_type_i ggml_backend_cuda_buffer_type_interface = {
    /* .get_name         = */ fGgmlBackendCudaBufferTypeGetName,
    /* .alloc_buffer     = */ fGgmlBackendCudaBufferTypeAllocBuffer,
    /* .get_alignment    = */ fGgmlBackendCudaBufferTypeGetAlignment,
    /* .get_max_size     = */ NULL, // defaults to SIZE_MAX
    /* .get_alloc_size   = */ fGgmlBackendCudaBufferTypeGetAllocSize,
    /* .is_host          = */ NULL,
};

ggml_backend_buffer_type_t ggml_backend_cuda_buffer_type(int pDevice) {
  static std::mutex vMutex;
  std::lock_guard<std::mutex> vLock(vMutex);

  if (pDevice >= ggml_backend_cuda_get_device_count()) {
    return nullptr;
  }

  static ggml_backend_buffer_type aGgmlBackendCudaBufferTypes[GGML_CUDA_MAX_DEVICES];

  static bool vGgmlBackendCudaBufferTypeInitialized = false;

  if (!vGgmlBackendCudaBufferTypeInitialized) {
    for (int vI = 0; vI < ggml_backend_cuda_get_device_count(); vI++) {
      aGgmlBackendCudaBufferTypes[vI] = {
          /* .iface    = */ ggml_backend_cuda_buffer_type_interface,
          /* .device   = */ ggml_backend_reg_dev_get(ggml_backend_cuda_reg(), vI),
          /* .context  = */ new ggml_backend_cuda_buffer_type_context{vI, GGML_CUDA_NAME + std::to_string(vI)},
      };
    }
    vGgmlBackendCudaBufferTypeInitialized = true;
  }

  return &aGgmlBackendCudaBufferTypes[pDevice];
}

// Communication context for multi-GPU AllReduce during tensor parallelism.
//
// Created once per meta backend instance.  Resources for the selected mode
// (NCCL communicators or the internal AllReduce pipeline) are initialised
// eagerly during comm_init so any init failure surfaces at startup rather
// than mid-run.
struct ggml_backend_cuda_comm_context {
  using try_allreduce_fn = bool (*)(ggml_backend_cuda_comm_context *, struct ggml_tensor **);

  std::vector<ggml_backend_t> lBackends;
  std::vector<int> lDevIds;

  // Set by the init chain (comm_init_{nccl, internal, none}) to one of
  // try_allreduce_{nccl, internal, butterfly}.  nccl needs `comms`,
  // internal needs `ar_pipeline`, butterfly needs nothing.  Per-call
  // failures return false; the meta backend's generic implementation then
  // handles that call.
  try_allreduce_fn vTryAllreduce = nullptr;

  ggml_cuda_ar_pipeline *vArPipeline = nullptr;

#ifdef GGML_USE_NCCL
  std::vector<ncclComm_t> vComms;
#endif // GGML_USE_NCCL

  ~ggml_backend_cuda_comm_context() {
#ifdef GGML_USE_NCCL
    for (ncclComm_t vComm : vComms) {
      NCCL_CHECK(ncclCommDestroy(vComm));
    }
#endif // GGML_USE_NCCL
    ggml_cuda_ar_pipeline_free(vArPipeline);
  }
};

#ifdef GGML_USE_NCCL
// AllReduce via NCCL. Reduces as FP32 for small tensors and BF16 for large
// tensors (bandwidth-bound), then converts back to FP32.
static bool fGgmlBackendCudaCommAllreduceNccl(ggml_backend_cuda_comm_context *pCommCtx, struct ggml_tensor **pTensors) {
  const int64_t cNe = ggml_nelements(pTensors[0]);
  // FIXME the input of llm_graph_context::build_in_out_ids can produce a tensor with 0 elements if n_outputs == 0
  // This then causes a crash in this function
  if (cNe == 0) {
    return true;
  }

  const size_t cNBackends = pCommCtx->lBackends.size();

  for (size_t vI = 0; vI < cNBackends; ++vI) {
    GGML_ASSERT(pTensors[vI] != nullptr);
    GGML_ASSERT(ggml_nelements(pTensors[vI]) == cNe);
    GGML_ASSERT(ggml_is_contiguously_allocated(pTensors[vI]));
  }

  // For small tensors, simply reduce them as FP32.
  // The following heuristic for how "small" a tensor should be is based on RTX 4090s connected via 16x PCIe 4.0.
  if ((cNBackends <= 2 && cNe < 32768) || (cNBackends == 3 && cNe < 131072) || (cNBackends >= 4 && cNe < 262144)) {
    for (size_t vI = 0; vI < cNBackends; ++vI) {
      if ((pTensors[vI]->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
        ggml_backend_cuda_context *vCudaCtx = (ggml_backend_cuda_context *)pCommCtx->lBackends[vI]->context;
        ggml_cuda_set_device(vCudaCtx->device);
        CUDA_CHECK(cudaMemsetAsync(pTensors[vI]->data, 0, ggml_nbytes(pTensors[vI]), vCudaCtx->stream()));
      }
    }
    NCCL_CHECK(ncclGroupStart());
    for (size_t vI = 0; vI < cNBackends; ++vI) {
      ggml_backend_cuda_context *vCudaCtx = (ggml_backend_cuda_context *)pCommCtx->lBackends[vI]->context;
      NCCL_CHECK(ncclAllReduce(pTensors[vI]->data, pTensors[vI]->data, cNe, ncclFloat, ncclSum, pCommCtx->vComms[vI],
                               vCudaCtx->stream()));
    }
    NCCL_CHECK(ncclGroupEnd());
    return true;
  }

  // For large tensors it's faster to compress them to BF16 for the reduction:
  to_bf16_cuda_t vToBf16 = ggml_get_to_bf16_cuda(GGML_TYPE_F32);
  to_fp32_cuda_t vToFp32 = ggml_get_to_fp32_cuda(GGML_TYPE_BF16);

  ggml_cuda_pool_alloc<nv_bfloat16> vTmp[GGML_CUDA_MAX_DEVICES];
  for (size_t vI = 0; vI < cNBackends; ++vI) {
    ggml_backend_cuda_context *vCudaCtx = (ggml_backend_cuda_context *)pCommCtx->lBackends[vI]->context;
    vTmp[vI].pool = &vCudaCtx->pool();
    vTmp[vI].alloc(cNe);

    ggml_cuda_set_device(vCudaCtx->device);
    if (pTensors[vI]->flags & GGML_TENSOR_FLAG_COMPUTE) {
      vToBf16(pTensors[vI]->data, vTmp[vI].get(), cNe, vCudaCtx->stream());
    } else {
      CUDA_CHECK(cudaMemsetAsync(vTmp[vI].get(), 0, cNe * sizeof(nv_bfloat16), vCudaCtx->stream()));
    }
    CUDA_CHECK(cudaGetLastError());
  }

  NCCL_CHECK(ncclGroupStart());
  for (size_t vI = 0; vI < cNBackends; ++vI) {
    ggml_backend_cuda_context *vCudaCtx = (ggml_backend_cuda_context *)pCommCtx->lBackends[vI]->context;
    NCCL_CHECK(ncclAllReduce(vTmp[vI].get(), vTmp[vI].get(), cNe, ncclBfloat16, ncclSum, pCommCtx->vComms[vI],
                             vCudaCtx->stream()));
  }
  NCCL_CHECK(ncclGroupEnd());

  for (size_t vI = 0; vI < cNBackends; ++vI) {
    ggml_backend_cuda_context *vCudaCtx = (ggml_backend_cuda_context *)pCommCtx->lBackends[vI]->context;

    ggml_cuda_set_device(vCudaCtx->device);
    vToFp32(vTmp[vI].get(), (float *)pTensors[vI]->data, cNe, vCudaCtx->stream());
    CUDA_CHECK(cudaGetLastError());
  }

  return true;
}
#endif // GGML_USE_NCCL

// Run the internal AR pipeline.  Returns false on unsupported / failed input
// -- the caller decides whether to abort (env-forced) or fall back silently.
static bool fGgmlBackendCudaCommAllreduceInternal(ggml_backend_cuda_comm_context *pCommCtx,
                                                  struct ggml_tensor **pTensors) {
  GGML_ASSERT(pCommCtx->vArPipeline != nullptr);

  const size_t cNBackends = pCommCtx->lBackends.size();
  GGML_ASSERT(cNBackends == 2);
  GGML_ASSERT(pTensors[0] != nullptr);

  const int64_t cNe = ggml_nelements(pTensors[0]);
  const ggml_type cType = pTensors[0]->type;

  if (cType != GGML_TYPE_F32 && cType != GGML_TYPE_F16 && cType != GGML_TYPE_BF16) {
    GGML_LOG_DEBUG("%s: internal unsupported: type=%d\n", __func__, (int)cType);
    return false;
  }

  if (cNe == 0) {
    return true;
  }

  for (size_t vI = 0; vI < cNBackends; ++vI) {
    if (pTensors[vI] == nullptr) {
      GGML_LOG_ERROR("%s: internal failed: tensor[%zu] is null\n", __func__, vI);
      return false;
    }
    if (ggml_nelements(pTensors[vI]) != cNe || pTensors[vI]->type != cType) {
      GGML_LOG_ERROR("%s: internal failed: tensor[%zu] ne=%" PRId64 " type=%d expected ne=%" PRId64 " type=%d\n",
                     __func__, vI, ggml_nelements(pTensors[vI]), (int)pTensors[vI]->type, cNe, (int)cType);
      return false;
    }
    if (!ggml_is_contiguously_allocated(pTensors[vI])) {
      GGML_LOG_DEBUG("%s: internal unsupported: tensor[%zu] is not contiguously allocated: ne=%" PRId64
                     " nbytes=%zu packed=%zu type=%d\n",
                     __func__, vI, cNe, ggml_nbytes(pTensors[vI]),
                     (size_t)cNe * ggml_type_size(cType) / ggml_blck_size(cType), (int)cType);
      return false;
    }
    if (((uintptr_t)pTensors[vI]->data & 0xF) != 0) {
      GGML_LOG_DEBUG("%s: internal unsupported: tensor[%zu] data pointer is not 16-byte aligned: %p type=%d ne=%" PRId64
                     "\n",
                     __func__, vI, pTensors[vI]->data, (int)cType, cNe);
      return false;
    }
    GGML_ASSERT((ggml_nbytes(pTensors[vI]) & 0xF) == 0);
  }

  return ggml_cuda_ar_allreduce(pCommCtx->vArPipeline, pCommCtx->lBackends.data(), pTensors);
}

// ---------------------------------------------------------------------------
// Per-call dispatch -- three variants, one per backend.  Each is set as
// comm_ctx->try_allreduce by the matching init step.  Per-call failure
// returns false; the meta backend's generic implementation handles that call.
// ---------------------------------------------------------------------------

#ifdef GGML_USE_NCCL
static bool fGgmlBackendCudaCommTryAllreduceNccl(ggml_backend_cuda_comm_context *pCommCtx,
                                                 struct ggml_tensor **pTensors) {
  return fGgmlBackendCudaCommAllreduceNccl(pCommCtx, pTensors);
}
#endif // GGML_USE_NCCL

static bool fGgmlBackendCudaCommTryAllreduceInternal(ggml_backend_cuda_comm_context *pCommCtx,
                                                     struct ggml_tensor **pTensors) {
  return fGgmlBackendCudaCommAllreduceInternal(pCommCtx, pTensors);
}

static bool fGgmlBackendCudaCommTryAllreduceButterfly(ggml_backend_cuda_comm_context *, struct ggml_tensor **) {
  return false;
}

static void fGgmlBackendCudaCommFree(void *pCommCtxV) {
  if (pCommCtxV == nullptr) {
    return;
  }
  delete static_cast<ggml_backend_cuda_comm_context *>(pCommCtxV);
}

// ---------------------------------------------------------------------------
// Init -- chained nccl -> internal -> none.  Each step tries to bring up its
// resource; on failure it warns and recurses into the next step.
// ---------------------------------------------------------------------------
static void fGgmlBackendCudaCommInitNone(ggml_backend_cuda_comm_context *pRet) {
  pRet->vTryAllreduce = fGgmlBackendCudaCommTryAllreduceButterfly;
}

static void fGgmlBackendCudaCommInitInternal(ggml_backend_cuda_comm_context *pRet) {
  pRet->vArPipeline = ggml_cuda_ar_pipeline_init(pRet->lDevIds.data(), pRet->lDevIds.size());
  if (pRet->vArPipeline) {
    pRet->vTryAllreduce = fGgmlBackendCudaCommTryAllreduceInternal;
    return;
  }

  // Clear sticky CUDA error from the failed init.
  (void)cudaGetLastError();
  GGML_LOG_WARN("internal AllReduce init failed (n_devices != 2?); "
                "falling back to meta-backend butterfly\n");
  fGgmlBackendCudaCommInitNone(pRet);
}

static void fGgmlBackendCudaCommInitNccl(ggml_backend_cuda_comm_context *pRet) {
#ifdef GGML_USE_NCCL
  // Disabling NCCL path when CUDA virtual devices are in use since NCCL requires one distinct physical GPU per rank.
  const ggml_cuda_device_info &vInfo = ggml_cuda_info();
  if (vInfo.device_count > vInfo.physical_device_count) {
    GGML_LOG_WARN("NCCL disabled: virtual devices in use; "
                  "falling back to internal AllReduce\n");
    fGgmlBackendCudaCommInitInternal(pRet);
    return;
  }

  const size_t cN = pRet->lDevIds.size();
  pRet->vComms.resize(cN);
  ncclResult_t vRc = ncclCommInitAll(pRet->vComms.data(), (int)cN, pRet->lDevIds.data());
  if (vRc == ncclSuccess) {
    pRet->vTryAllreduce = fGgmlBackendCudaCommTryAllreduceNccl;
    return;
  }

  pRet->vComms.clear();
  GGML_LOG_WARN("NCCL init failed (%s); falling back to internal AllReduce\n", ncclGetErrorString(vRc));
#else // GGML_USE_NCCL
#ifndef GGML_USE_HIP
  GGML_LOG_WARN("NCCL not compiled in; falling back to internal AllReduce.  "
                "Recompile with -DGGML_CUDA_NCCL=ON for best multi-GPU performance.\n");
#endif // !GGML_USE_HIP
#endif // GGML_USE_NCCL

  fGgmlBackendCudaCommInitInternal(pRet);
}

// Top-level init.  Picks one of the three init paths based on
// GGML_CUDA_ALLREDUCE (or the platform default) and lets the chain handle
// any fallback.  Unrecognised env values warn and fall through to the
// platform default.
static void *fGgmlBackendCudaCommInit(ggml_backend_t *pBackends, size_t pNBackends) {
  for (size_t vI = 0; vI < pNBackends; vI++) {
    if (!ggml_backend_is_cuda(pBackends[vI])) {
      return nullptr;
    }
  }

  auto *vRet = new ggml_backend_cuda_comm_context;
  vRet->lBackends.assign(pBackends, pBackends + pNBackends);
  vRet->lDevIds.reserve(pNBackends);
  for (size_t vI = 0; vI < pNBackends; vI++) {
    vRet->lDevIds.push_back(static_cast<ggml_backend_cuda_context *>(pBackends[vI]->context)->device);
  }

  const char *vEnv = getenv("GGML_CUDA_ALLREDUCE");
  if (!vEnv) {
    // Platform default: Linux uses NCCL, otherwise (generally Windows) internal
#if defined(__linux__)
    fGgmlBackendCudaCommInitNccl(vRet);
#else
    fGgmlBackendCudaCommInitInternal(vRet);
#endif // defined(__linux__)
  } else {
    std::string vEnvStr(vEnv);
    if (vEnvStr == "nccl") {
      fGgmlBackendCudaCommInitNccl(vRet);
    } else if (vEnvStr == "internal") {
      fGgmlBackendCudaCommInitInternal(vRet);
    } else if (vEnvStr == "none") {
      fGgmlBackendCudaCommInitNone(vRet);
    } else {
      GGML_LOG_WARN("unknown GGML_CUDA_ALLREDUCE value: %s\n", vEnv);
      fGgmlBackendCudaCommInitNone(vRet);
    }
  }

  return vRet;
}

// Top-level dispatch -- calls the function pointer chosen by comm_init.
// Returns false to let the meta-backend's butterfly run.
static bool fGgmlBackendCudaCommAllreduceTensor(void *pCommCtxV, struct ggml_tensor **pTensors) {
  if (pCommCtxV == nullptr) {
    return false;
  }
  auto *vCommCtx = static_cast<ggml_backend_cuda_comm_context *>(pCommCtxV);
  return vCommCtx->vTryAllreduce(vCommCtx, pTensors);
}

// host buffer type

static const char *fGgmlBackendCudaHostBufferTypeName(ggml_backend_buffer_type_t pBuft) {
  return GGML_CUDA_NAME "_Host";

  GGML_UNUSED(pBuft);
}

static bool fGgmlBackendBuftIsCudaHost(ggml_backend_buffer_type_t pBuft) {
  return pBuft->iface.get_name == fGgmlBackendCudaHostBufferTypeName;
}

static void fGgmlBackendCudaHostBufferFreeBuffer(ggml_backend_buffer_t pBuffer) {
  CUDA_CHECK(cudaFreeHost(pBuffer->context));
}

static void *fGgmlCudaHostMalloc(size_t pSize) {
  if (getenv("GGML_CUDA_NO_PINNED") != nullptr) {
    return nullptr;
  }

  void *vPtr = nullptr;
  cudaError_t vErr = cudaMallocHost((void **)&vPtr, pSize);
  if (vErr != cudaSuccess) {
    // clear the error
    (void)cudaGetLastError();
    GGML_LOG_DEBUG("%s: failed to allocate %.2f MiB of pinned memory: %s\n", __func__, pSize / 1024.0 / 1024.0,
                   cudaGetErrorString(vErr));
    return nullptr;
  }

  return vPtr;
}

static ggml_backend_buffer_t fGgmlBackendCudaHostBufferTypeAllocBuffer(ggml_backend_buffer_type_t pBuft, size_t pSize) {
  void *vPtr = fGgmlCudaHostMalloc(pSize);

  if (vPtr == nullptr) {
    // fallback to cpu buffer
    return ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), pSize);
  }

  ggml_backend_buffer_t vBuffer = ggml_backend_cpu_buffer_from_ptr(vPtr, pSize);
  vBuffer->buft = pBuft;
  vBuffer->iface.free_buffer = fGgmlBackendCudaHostBufferFreeBuffer;

  return vBuffer;
}

ggml_backend_buffer_type_t ggml_backend_cuda_host_buffer_type() {
  static struct ggml_backend_buffer_type vGgmlBackendCudaBufferTypeHost = {
      /* .iface    = */ {
          /* .get_name         = */ fGgmlBackendCudaHostBufferTypeName,
          /* .alloc_buffer     = */ fGgmlBackendCudaHostBufferTypeAllocBuffer,
          /* .get_alignment    = */ ggml_backend_cpu_buffer_type()->iface.get_alignment,
          /* .get_max_size     = */ NULL, // defaults to SIZE_MAX
          /* .get_alloc_size   = */ ggml_backend_cpu_buffer_type()->iface.get_alloc_size,
          /* .is_host          = */ ggml_backend_cpu_buffer_type()->iface.is_host,
      },
      /* .device   = */ ggml_backend_reg_dev_get(ggml_backend_cuda_reg(), 0),
      /* .context  = */ nullptr,
  };

  return &vGgmlBackendCudaBufferTypeHost;
}

// static bool ggml_backend_buffer_is_cuda_host(ggml_backend_buffer_t buffer) {
//     return buffer->buft->iface.get_name == ggml_backend_cuda_host_buffer_type_name;
// }

/// kernels

typedef void (*ggml_cuda_op_mul_mat_t)(ggml_backend_cuda_context &pCtx, const ggml_tensor *pSrc0,
                                       const ggml_tensor *pSrc1, ggml_tensor *pDst, const char *pSrc0DdI,
                                       const float *pSrc1DdfI, const char *pSrc1DdqI, float *pDstDdI,
                                       const int64_t pRowLow, const int64_t pRowHigh, const int64_t pSrc1Ncols,
                                       const int64_t pSrc1PaddedRowSize, cudaStream_t pStream);

static __global__ void fKComputeBatchedPtrs(const void *pSrc0AsF16, const void *pSrc1AsF16, char *pDst,
                                            const void **pPtrsSrc, void **pPtrsDst, int64_t pNe12, int64_t pNe13,
                                            int64_t pNe23, size_t pNb02, size_t pNb03, size_t pNb12, size_t pNb13,
                                            size_t pNbd2, size_t pNbd3, int64_t pR2, int64_t pR3) {
  const int64_t cI13 = blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t cI12 = blockIdx.y * blockDim.y + threadIdx.y;

  if (cI13 >= pNe13 || cI12 >= pNe12) {
    return;
  }

  const int64_t cI03 = cI13 / pR3;
  const int64_t cI02 = cI12 / pR2;

  pPtrsSrc[0 * pNe23 + cI12 + cI13 * pNe12] = (const char *)pSrc0AsF16 + cI02 * pNb02 + cI03 * pNb03;
  pPtrsSrc[1 * pNe23 + cI12 + cI13 * pNe12] = (const char *)pSrc1AsF16 + cI12 * pNb12 + cI13 * pNb13;
  pPtrsDst[0 * pNe23 + cI12 + cI13 * pNe12] = (char *)pDst + cI12 * pNbd2 + cI13 * pNbd3;
}

// Type traits for mapping ggml types to CUDA/cuBLAS types
template <ggml_type T> struct batched_mul_mat_traits;

template <> struct batched_mul_mat_traits<GGML_TYPE_F32> {
  using cuda_type = float;
  static inline const cublasComputeType_t compute_type = CUBLAS_COMPUTE_32F;
  static inline const cudaDataType_t data_type = CUDA_R_32F;
  static inline const ggml_type ggml_type_val = GGML_TYPE_F32;
  static inline const float alpha = 1.0f;
  static inline const float beta = 0.0f;
  static inline const void *get_alpha() {
    static const float cVal = alpha;
    return &cVal;
  }
  static inline const void *get_beta() {
    static const float cVal = beta;
    return &cVal;
  }
  static inline auto convert(ggml_type pSrcType) { return ggml_get_to_fp32_cuda(pSrcType); }
  static inline auto convert_nc(ggml_type pSrcType) { return ggml_get_to_fp32_nc_cuda(pSrcType); }
};

template <> struct batched_mul_mat_traits<GGML_TYPE_BF16> {
  using cuda_type = nv_bfloat16;
  static inline const cublasComputeType_t compute_type = CUBLAS_COMPUTE_32F;
  static inline const cudaDataType_t data_type = CUDA_R_16BF;
  static inline const ggml_type ggml_type_val = GGML_TYPE_BF16;
  static inline const float alpha = 1.0f;
  static inline const float beta = 0.0f;
  static inline const void *get_alpha() {
    static const float cVal = alpha;
    return &cVal;
  }
  static inline const void *get_beta() {
    static const float cVal = beta;
    return &cVal;
  }
  static inline auto convert(ggml_type pSrcType) { return ggml_get_to_bf16_cuda(pSrcType); }
  static inline auto convert_nc(ggml_type pSrcType) { return ggml_get_to_bf16_nc_cuda(pSrcType); }
};

template <> struct batched_mul_mat_traits<GGML_TYPE_F16> {
  using cuda_type = half;
  static inline const cublasComputeType_t compute_type = CUBLAS_COMPUTE_16F;
  static inline const cudaDataType_t data_type = CUDA_R_16F;
  static inline const ggml_type ggml_type_val = GGML_TYPE_F16;
  static inline const half alpha = 1.0;
  static inline const half beta = 0.0;
  static inline const void *get_alpha() {
    static const half cVal = alpha;
    return &cVal;
  }
  static inline const void *get_beta() {
    static const half cVal = beta;
    return &cVal;
  }
  static inline auto convert(ggml_type pSrcType) { return ggml_get_to_fp16_cuda(pSrcType); }
  static inline auto convert_nc(ggml_type pSrcType) { return ggml_get_to_fp16_nc_cuda(pSrcType); }
};

template <ggml_type compute_type>
static void fGgmlCudaMulMatCublasImpl(ggml_backend_cuda_context &pCtx, const ggml_tensor *src0, const ggml_tensor *src1,
                                      ggml_tensor *dst) {
  using traits = batched_mul_mat_traits<compute_type>;
  using cuda_t = typename traits::cuda_type;

  GGML_ASSERT(ggml_is_contiguous(dst));

  // Byte offsets and tensor dimensions are currently used in an inconsistent way for dst.
  // As long as dst is contiguous this does not matter though.

  GGML_TENSOR_BINARY_OP_LOCALS

  const int64_t cNeDst = ggml_nelements(dst);
  cudaStream_t vMainStream = pCtx.stream();
  CUBLAS_CHECK(cublasSetStream(pCtx.cublas_handle(), vMainStream));

  const size_t cSrc0Ts = ggml_type_size(src0->type);
  GGML_ASSERT(nb00 == cSrc0Ts);
  int64_t vS01 = nb01 / cSrc0Ts;
  int64_t vS02 = nb02 / cSrc0Ts;
  int64_t vS03 = nb03 / cSrc0Ts;

  const size_t cSrc1Ts = ggml_type_size(src1->type);
  GGML_ASSERT(nb10 == cSrc1Ts);
  int64_t vS11 = nb11 / cSrc1Ts;
  int64_t vS12 = nb12 / cSrc1Ts;
  int64_t vS13 = nb13 / cSrc1Ts;

  float *vDstDdf = (float *)dst->data;

  const cuda_t *vSrc0Ptr = nullptr;
  const cuda_t *vSrc1Ptr = nullptr;

  ggml_cuda_pool_alloc<cuda_t> vSrc0Alloc(pCtx.pool());
  ggml_cuda_pool_alloc<cuda_t> vSrc1Alloc(pCtx.pool());

  bool vIsSrc0Cont2 = ggml_is_contiguous_2(src0);
  bool vIsSrc1Cont2 = ggml_is_contiguous_2(src1);

  if (src0->type == compute_type) {
    vSrc0Ptr = (const cuda_t *)src0->data;
  } else {
    vSrc0Alloc.alloc(ggml_nelements(src0));

    if (ggml_is_contiguously_allocated(src0)) {
      const auto cConvertFunc = traits::convert(src0->type);
      GGML_ASSERT(cConvertFunc != nullptr);
      cConvertFunc(src0->data, vSrc0Alloc.get(), ggml_nelements(src0), vMainStream);
      const size_t cSrc0Bs = ggml_blck_size(src0->type);
      vS01 *= cSrc0Bs;
      vS02 *= cSrc0Bs;
      vS03 *= cSrc0Bs;
    } else {
      const auto cConvertFunc = traits::convert_nc(src0->type);
      GGML_ASSERT(cConvertFunc != nullptr);
      cConvertFunc(src0->data, vSrc0Alloc.get(), ne00, ne01, ne02, ne03, vS01, vS02, vS03, vMainStream);
      vS01 = ne00;
      vS02 = ne01 * vS01;
      vS03 = ne02 * vS02;
      vIsSrc0Cont2 = true;
    }
    vSrc0Ptr = vSrc0Alloc.get();
  }

  if (src1->type == compute_type) {
    vSrc1Ptr = (const cuda_t *)src1->data;
  } else {
    vSrc1Alloc.alloc(ggml_nelements(src1));

    if (ggml_is_contiguously_allocated(src1)) {
      const auto cConvertFunc = traits::convert(src1->type);
      GGML_ASSERT(cConvertFunc != nullptr);
      cConvertFunc(src1->data, vSrc1Alloc.get(), ggml_nelements(src1), vMainStream);
      const size_t cSrc1Bs = ggml_blck_size(src1->type);
      vS11 *= cSrc1Bs;
      vS12 *= cSrc1Bs;
      vS13 *= cSrc1Bs;
    } else {
      const auto cConvertFunc = traits::convert_nc(src1->type);
      GGML_ASSERT(cConvertFunc != nullptr);
      cConvertFunc(src1->data, vSrc1Alloc.get(), ne10, ne11, ne12, ne13, vS11, vS12, vS13, vMainStream);
      vS11 = ne10;
      vS12 = ne11 * vS11;
      vS13 = ne12 * vS12;
      vIsSrc1Cont2 = true;
    }
    vSrc1Ptr = vSrc1Alloc.get();
  }

  ggml_cuda_pool_alloc<cuda_t> vDstTemp(pCtx.pool());
  char *vDstPtr;
  size_t vNbd2 = dst->nb[2];
  size_t vNbd3 = dst->nb[3];

  cublasComputeType_t vCuComputeType = traits::compute_type;
  cudaDataType_t vCuDataType = traits::data_type;
  cudaDataType_t vCuDataTypeA = traits::data_type;
  cudaDataType_t vCuDataTypeB = traits::data_type;
  const void *vAlpha = traits::get_alpha();
  const void *vBeta = traits::get_beta();

  const int cCc = ggml_cuda_info().devices[pCtx.device].cc;
  bool vPreferF32Output = false;
  if (compute_type == GGML_TYPE_F16) {
    vPreferF32Output = cCc == GGML_CUDA_CC_VOLTA || GGML_CUDA_CC_IS_RDNA4(cCc) || GGML_CUDA_CC_IS_CDNA(cCc);
  } else if (compute_type == GGML_TYPE_BF16) {
    vPreferF32Output = !GGML_CUDA_CC_IS_RDNA3(cCc) && !GGML_CUDA_CC_IS_CDNA(cCc);
  }

  if (vPreferF32Output) {
    vDstPtr = (char *)vDstDdf;
    vCuComputeType = batched_mul_mat_traits<GGML_TYPE_F32>::compute_type;
    vCuDataType = batched_mul_mat_traits<GGML_TYPE_F32>::data_type;
    vAlpha = batched_mul_mat_traits<GGML_TYPE_F32>::get_alpha();
    vBeta = batched_mul_mat_traits<GGML_TYPE_F32>::get_beta();
  } else {
    if constexpr (compute_type == GGML_TYPE_F32) {
      vDstPtr = (char *)vDstDdf; // Direct F32 output
    } else {
      vDstPtr = (char *)vDstTemp.alloc(cNeDst);
      vNbd2 /= sizeof(float) / sizeof(cuda_t);
      vNbd3 /= sizeof(float) / sizeof(cuda_t);
    }
  }

  GGML_ASSERT(ne12 % ne02 == 0);
  GGML_ASSERT(ne13 % ne03 == 0);

  // broadcast factors
  const int64_t cR2 = ne12 / ne02;
  const int64_t cR3 = ne13 / ne03;

  // Theoretically cublasGemmStridedBatchedEx would always work, even for a single matrix.
  // However, for some old NVIDIA and AMD GPUs the strided/Ex GEMM is much slower,
  //     probably because the internal kernel selection logic is suboptimal.
  if (compute_type == GGML_TYPE_F32 && ne12 == 1 && ne13 == 1) {
    CUBLAS_CHECK(cublasSgemm(pCtx.cublas_handle(), CUBLAS_OP_T, CUBLAS_OP_N, ne01, ne11, ne10, (const float *)vAlpha,
                             (const float *)vSrc0Ptr, vS01, (const float *)vSrc1Ptr, vS11, (const float *)vBeta,
                             (float *)vDstPtr, ne0));
  } else if (ne12 == 1 && ne13 == 1) {
    CUBLAS_CHECK(cublasGemmEx(pCtx.cublas_handle(), CUBLAS_OP_T, CUBLAS_OP_N, ne01, ne11, ne10, vAlpha, vSrc0Ptr,
                              vCuDataTypeA, vS01, vSrc1Ptr, vCuDataTypeB, vS11, vBeta, vDstPtr, vCuDataType, ne0,
                              vCuComputeType, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  } else if (cR2 == 1 && cR3 == 1 && vIsSrc0Cont2 && vIsSrc1Cont2) {
    // with a [0, 2, 1, 3] perm. and ne02==1 the matrix strides need to be determined from dim 3:
    const int64_t cSma = ne02 == 1 ? vS03 : vS02;
    const int64_t cSmb = ne12 == 1 ? vS13 : vS12;

    // there is no broadcast and src0, src1 are contiguous across dims 2, 3
    // use cublasGemmStridedBatchedEx
    CUBLAS_CHECK(cublasGemmStridedBatchedEx(pCtx.cublas_handle(), CUBLAS_OP_T, CUBLAS_OP_N, ne01, ne11, ne10, vAlpha,
                                            vSrc0Ptr, vCuDataTypeA, vS01, cSma,          // strideA
                                            vSrc1Ptr, vCuDataTypeB, vS11, cSmb,          // strideB
                                            vBeta, vDstPtr, vCuDataType, ne0, ne1 * ne0, // strideC
                                            ne12 * ne13, vCuComputeType, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  } else {
    // use cublasGemmBatchedEx
    const int64_t cNe23 = ne12 * ne13;

    ggml_cuda_pool_alloc<const void *> vPtrsSrc(pCtx.pool(), 2 * cNe23);
    ggml_cuda_pool_alloc<void *> vPtrsDst(pCtx.pool(), 1 * cNe23);

    const size_t cSrcTypeSize = sizeof(cuda_t);

    const int cThreadsX = 16;
    const int cThreadsY = 16;
    const dim3 cBlockDims(cThreadsX, cThreadsY);

    const dim3 cGridDims((ne13 + cThreadsX - 1) / cThreadsX, (ne12 + cThreadsY - 1) / cThreadsY);
    fKComputeBatchedPtrs<<<cGridDims, cBlockDims, 0, vMainStream>>>(
        vSrc0Ptr, vSrc1Ptr, vDstPtr, vPtrsSrc.get(), vPtrsDst.get(), ne12, ne13, cNe23, vS02 * cSrcTypeSize,
        vS03 * cSrcTypeSize, vS12 * cSrcTypeSize, vS13 * cSrcTypeSize, vNbd2, vNbd3, cR2, cR3);

    CUDA_CHECK(cudaGetLastError());

    CUBLAS_CHECK(cublasGemmBatchedEx(pCtx.cublas_handle(), CUBLAS_OP_T, CUBLAS_OP_N, ne01, ne11, ne10, vAlpha,
                                     (const void **)(vPtrsSrc.get() + 0 * cNe23), vCuDataTypeA, vS01,
                                     (const void **)(vPtrsSrc.get() + 1 * cNe23), vCuDataTypeB, vS11, vBeta,
                                     (void **)(vPtrsDst.get() + 0 * cNe23), vCuDataType, ne0, cNe23, vCuComputeType,
                                     CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  }

  // Convert output back to F32 if needed
  if (vCuDataType != CUDA_R_32F) {
    const to_fp32_cuda_t cToFp32Cuda = ggml_get_to_fp32_cuda(traits::ggml_type_val);
    cToFp32Cuda(vDstTemp.get(), vDstDdf, cNeDst, vMainStream);
  }
}

static void fGgmlCudaMulMatCublas(ggml_backend_cuda_context &pCtx, const ggml_tensor *pSrc0, const ggml_tensor *pSrc1,
                                  ggml_tensor *pDst) {
  ggml_type vComputeType = pSrc0->type;
  if (ggml_is_quantized(vComputeType)) {
    vComputeType =
        fast_fp16_hardware_available(ggml_cuda_info().devices[pCtx.device].cc) ? GGML_TYPE_F16 : GGML_TYPE_F32;
  } else if (vComputeType == GGML_TYPE_F16 && !fast_fp16_hardware_available(ggml_cuda_info().devices[pCtx.device].cc)) {
    vComputeType = GGML_TYPE_F32;
  }
  if (pDst->op_params[0] == GGML_PREC_F32) {
    vComputeType = GGML_TYPE_F32;
  }

  const char *vEnvC = getenv("GGML_CUDA_CUBLAS_COMPUTE_TYPE");
  if (vEnvC != nullptr) {
    std::string vEnvCpp = vEnvC;
    for (char &vC : vEnvCpp) {
      vC = std::tolower(vC);
    }
    if (vEnvCpp == "f32" || vEnvCpp == "fp32") {
      vComputeType = GGML_TYPE_F32;
    } else if (vEnvCpp == "f16" || vEnvCpp == "fp16") {
      vComputeType = GGML_TYPE_F16;
    } else if (vEnvCpp == "bf16") {
      vComputeType = GGML_TYPE_BF16;
    } else if (vEnvCpp != "auto") {
      GGML_LOG_WARN("%s: unknown value for GGML_CUDA_CUBLAS_COMPUTE_TYPE: %s", __func__, vEnvCpp.c_str());
    }
  }

  switch (vComputeType) {
  case GGML_TYPE_F32:
    fGgmlCudaMulMatCublasImpl<GGML_TYPE_F32>(pCtx, pSrc0, pSrc1, pDst);
    break;
  case GGML_TYPE_BF16:
    fGgmlCudaMulMatCublasImpl<GGML_TYPE_BF16>(pCtx, pSrc0, pSrc1, pDst);
    break;
  case GGML_TYPE_F16:
    fGgmlCudaMulMatCublasImpl<GGML_TYPE_F16>(pCtx, pSrc0, pSrc1, pDst);
    break;
  default:
    GGML_ABORT("fatal error");
  }
}

static bool fGgmlCudaShouldFuseMulMat(const ggml_tensor *pFfnUp, const ggml_tensor *pFfnGate, const ggml_tensor *pGlu,
                                      const ggml_tensor *pFfnUpBias = nullptr,
                                      const ggml_tensor *pFfnGateBias = nullptr,
                                      const ggml_tensor *pFfnUpScale = nullptr,
                                      const ggml_tensor *pFfnGateScale = nullptr) {
  const bool cHasBias = pFfnUpBias != nullptr || pFfnGateBias != nullptr;
  const bool cHasScale = pFfnUpScale != nullptr || pFfnGateScale != nullptr;

  if (cHasBias && (!pFfnUpBias || !pFfnGateBias)) {
    return false;
  }
  if (cHasScale && (!pFfnUpScale || !pFfnGateScale)) {
    return false;
  }

  const bool cIsMulMat = pFfnUp->op == GGML_OP_MUL_MAT && pFfnGate->op == GGML_OP_MUL_MAT && pGlu->op == GGML_OP_GLU;
  const bool cIsMulMatId =
      pFfnUp->op == GGML_OP_MUL_MAT_ID && pFfnGate->op == GGML_OP_MUL_MAT_ID && pGlu->op == GGML_OP_GLU;

  GGML_ASSERT(pFfnUp && pFfnGate && pGlu);

  if (!cIsMulMat && !cIsMulMatId) {
    return false;
  }

  const ggml_op cExpectedBiasOp = cIsMulMat ? GGML_OP_ADD : GGML_OP_ADD_ID;
  const ggml_tensor *vFfnUpBiasSrc = cHasScale ? pFfnUpScale : pFfnUp;
  const ggml_tensor *vFfnGateBiasSrc = cHasScale ? pFfnGateScale : pFfnGate;
  const ggml_tensor *vFfnUpOut = cHasBias ? pFfnUpBias : vFfnUpBiasSrc;
  const ggml_tensor *vFfnGateOut = cHasBias ? pFfnGateBias : vFfnGateBiasSrc;

  if (pGlu->src[0] != vFfnGateOut || pGlu->src[1] != vFfnUpOut) {
    return false;
  }

  if (cHasScale) {
    if (pFfnUpScale->op != GGML_OP_MUL || pFfnGateScale->op != GGML_OP_MUL) {
      return false;
    }
    const bool cUpHasMm = pFfnUpScale->src[0] == pFfnUp || pFfnUpScale->src[1] == pFfnUp;
    const bool cGateHasMm = pFfnGateScale->src[0] == pFfnGate || pFfnGateScale->src[1] == pFfnGate;
    if (!cUpHasMm || !cGateHasMm) {
      return false;
    }
  }

  if (cHasBias) {
    if (pFfnUpBias->op != cExpectedBiasOp || pFfnGateBias->op != cExpectedBiasOp) {
      return false;
    }

    if (cExpectedBiasOp == GGML_OP_ADD) {
      const bool cUpHasMul = pFfnUpBias->src[0] == vFfnUpBiasSrc || pFfnUpBias->src[1] == vFfnUpBiasSrc;
      const bool cGateHasMul = pFfnGateBias->src[0] == vFfnGateBiasSrc || pFfnGateBias->src[1] == vFfnGateBiasSrc;
      if (!cUpHasMul || !cGateHasMul) {
        return false;
      }
    } else { // GGML_OP_ADD_ID
      if (pFfnUpBias->src[0] != vFfnUpBiasSrc || pFfnGateBias->src[0] != vFfnGateBiasSrc) {
        return false;
      }
      if (pFfnUpBias->src[2] != pFfnUp->src[2] || pFfnGateBias->src[2] != pFfnGate->src[2]) {
        return false;
      }
    }
  }

  if (pFfnUp->src[0]->type != pFfnGate->src[0]->type || !ggml_are_same_shape(pFfnUp->src[0], pFfnGate->src[0]) ||
      !ggml_are_same_stride(pFfnUp->src[0], pFfnGate->src[0])) {
    return false;
  }

  if (pFfnUp->src[1] != pFfnGate->src[1]) {
    return false;
  }

  if (cIsMulMatId && pFfnUp->src[2] != pFfnGate->src[2]) {
    return false;
  }

  static constexpr std::array<ggml_glu_op, 4> aValidGluOps = {GGML_GLU_OP_SWIGLU, GGML_GLU_OP_GEGLU,
                                                              GGML_GLU_OP_SWIGLU_OAI, GGML_GLU_OP_SWIGLU_CLAMP};

  if (std::find(aValidGluOps.begin(), aValidGluOps.end(), ggml_get_glu_op(pGlu)) == aValidGluOps.end()) {
    return false;
  }

  if (const bool cSwapped = ggml_get_op_params_i32(pGlu, 1); cSwapped) {
    return false;
  }

  return true;
}

static bool fGgmlCudaShouldFuseMulMatVecF(const ggml_tensor *pTensor) {
  ggml_tensor *vSrc0 = pTensor->src[0];
  ggml_tensor *vSrc1 = pTensor->src[1];
  const ggml_tensor *vDst = pTensor;

  const bool cIsMulMatId = pTensor->op == GGML_OP_MUL_MAT_ID;

  bool vUseMulMatVecF =
      (vSrc0->type == GGML_TYPE_F32 || vSrc0->type == GGML_TYPE_F16 || vSrc0->type == GGML_TYPE_BF16) &&
      vSrc1->type == GGML_TYPE_F32 && vDst->type == GGML_TYPE_F32;

  const int cCc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
  vUseMulMatVecF = vUseMulMatVecF && ggml_cuda_should_use_mmvf(vSrc0->type, cCc, vSrc0->ne, vSrc0->nb,
                                                               cIsMulMatId ? vSrc1->ne[2] : vSrc1->ne[1]);

  // we only support fusion for ncols_dst = 1
  if (pTensor->op == GGML_OP_MUL_MAT && vDst->ne[1] != 1) {
    return false;
  }

  if (pTensor->op == GGML_OP_MUL_MAT_ID && vDst->ne[2] != 1) {
    return false;
  }

  return vUseMulMatVecF;
}

static bool fGgmlCudaShouldFuseMulMatVecQ(const ggml_tensor *pTensor) {
  ggml_tensor *vSrc0 = pTensor->src[0];
  ggml_tensor *vSrc1 = pTensor->src[1];
  const ggml_tensor *vDst = pTensor;

  const bool cBadPaddingClear = ggml_backend_buffer_get_usage(vSrc0->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE &&
                                ggml_nbytes(vSrc0) != ggml_backend_buffer_get_alloc_size(vSrc0->buffer, vSrc0) &&
                                vSrc0->view_src;

  bool vUseMulMatVecQ = ggml_is_quantized(vSrc0->type) && !cBadPaddingClear && vSrc1->type == GGML_TYPE_F32 &&
                        vDst->type == GGML_TYPE_F32 && vSrc1->ne[1] <= MMVQ_MAX_BATCH_SIZE;

  // fusion is not universally faster on Pascal
  const int cCc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
  if (cCc <= GGML_CUDA_CC_PASCAL) {
    return false;
  }
  // we only support fusion for ncols_dst = 1
  if (pTensor->op == GGML_OP_MUL_MAT && vDst->ne[1] != 1) {
    return false;
  }

  if (pTensor->op == GGML_OP_MUL_MAT_ID && vDst->ne[2] > get_mmvq_mmid_max_batch(vSrc0->type, cCc)) {
    return false;
  }

  return vUseMulMatVecQ;
}

static void fGgmlCudaMulMat(ggml_backend_cuda_context &pCtx, const ggml_tensor *src0, const ggml_tensor *src1,
                            ggml_tensor *dst) {
  GGML_TENSOR_BINARY_OP_LOCALS

  const int32_t cHint = ggml_get_op_params_i32(dst, 1);
  if (cHint == GGML_HINT_SRC0_IS_HADAMARD && ggml_cuda_op_fwht(pCtx, src1, dst)) {
    return;
  }

  // If src0 is a temporary compute buffer it may have some padding that needs to be cleared for mul_mat_vec_q or
  // mul_mat_q. But if src0 is also a view of another tensor then this cannot be done safely because it may overwrite
  // valid tensor data. Therefore, in such cases use cuBLAS.
  const bool cBadPaddingClear = ggml_backend_buffer_get_usage(src0->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE &&
                                ggml_nbytes(src0) != ggml_backend_buffer_get_alloc_size(src0->buffer, src0) &&
                                src0->view_src;
  if (cBadPaddingClear || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
    fGgmlCudaMulMatCublas(pCtx, src0, src1, dst);
    return;
  }

  const int cCc = ggml_cuda_info().devices[pCtx.device].cc;
  const int cWarpSize = ggml_cuda_info().devices[pCtx.device].warp_size;

  if (ggml_cuda_should_use_mmvf(src0->type, cCc, src0->ne, src0->nb, ne11)) {
    // The custom F16 vector kernel can be used over batched cuBLAS GEMM.
    // But this is only faster for GPUs without tensor cores or with a thin src0 matrix (particularly KQV in attention)
    ggml_cuda_mul_mat_vec_f(pCtx, src0, src1, nullptr, dst);
    return;
  }
  // A transposed vector can still use MMVQ (i.e. ne01 == 1)
  if (ne01 == 1 && ne11 > MMVF_MAX_BATCH_SIZE && ne2 == 1 && ne3 == 1 && src0->type == GGML_TYPE_F32 &&
      ggml_is_contiguous(src0) && ggml_is_contiguous(src1) && ggml_is_contiguous(dst) &&
      ggml_cuda_should_use_mmvf(src1->type, cCc, src1->ne, src1->nb, /*ne11 =*/1)) {
    ggml_tensor vDstVec = *dst;
    vDstVec.ne[0] = ne11;
    vDstVec.ne[1] = 1;
    vDstVec.nb[1] = vDstVec.nb[0] * ne11;
    vDstVec.nb[2] = vDstVec.nb[1];
    vDstVec.nb[3] = vDstVec.nb[1];
    ggml_cuda_mul_mat_vec_f(pCtx, src1, src0, nullptr, &vDstVec);
    return;
  }
  if (ggml_cuda_should_use_mmf(src0->type, cCc, cWarpSize, src0->ne, src0->nb, ne11, /*mul_mat_id =*/false)) {
    ggml_cuda_mul_mat_f(pCtx, src0, src1, nullptr, dst);
    return;
  }
  if (ggml_cuda_should_use_mmvq(src0->type, cCc, ne11)) {
    ggml_cuda_mul_mat_vec_q(pCtx, src0, src1, nullptr, dst);
    return;
  }
  if (ggml_cuda_should_use_mmq(src0->type, cCc, ne11, /*n_experts =*/0)) {
    ggml_cuda_mul_mat_q(pCtx, src0, src1, nullptr, dst);
    return;
  }
  fGgmlCudaMulMatCublas(pCtx, src0, src1, dst);
}

// returns true when ggml_cuda_mul_mat_id takes the fallback path that requires stream synchronization
// [TAG_MUL_MAT_ID_CUDA_GRAPHS]
static bool fGgmlCudaMulMatIdNeedsSync(const ggml_tensor *pDst, const int pCc) {
  const ggml_tensor *vSrc0 = pDst->src[0];
  const ggml_tensor *vSrc1 = pDst->src[1];

  if (vSrc1->type != GGML_TYPE_F32 || pDst->type != GGML_TYPE_F32) {
    return true;
  }

  if (pDst->ne[2] <= MMVQ_MAX_BATCH_SIZE) {
    if (ggml_is_quantized(vSrc0->type)) {
      if (pDst->ne[2] <= get_mmvq_mmid_max_batch(vSrc0->type, pCc)) {
        return false;
      }
    } else if (GGML_CUDA_CC_IS_AMD(pCc)) {
      return false;
    }
  }

  if (ggml_cuda_should_use_mmq(vSrc0->type, pCc, vSrc1->ne[2], /*n_experts=*/vSrc0->ne[2])) {
    return false;
  }

  if (ggml_cuda_should_use_mmf(vSrc0->type, pCc, WARP_SIZE, vSrc0->ne, vSrc0->nb, vSrc1->ne[2], /*mul_mat_id=*/true)) {
    return false;
  }

  return true;
}

static void fGgmlCudaMulMatId(ggml_backend_cuda_context &pCtx, ggml_tensor *dst) {
  const ggml_tensor *src0 = dst->src[0];
  const ggml_tensor *src1 = dst->src[1];
  const ggml_tensor *vIds = dst->src[2];

  GGML_ASSERT(src1->type == GGML_TYPE_F32);
  GGML_ASSERT(dst->type == GGML_TYPE_F32);

  GGML_TENSOR_BINARY_OP_LOCALS

  const int cCc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;

  // [TAG_MUL_MAT_ID_CUDA_GRAPHS]
  if (src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
    static_assert(MMVQ_MAX_BATCH_SIZE == MMVF_MAX_BATCH_SIZE);
    if (ne2 <= MMVQ_MAX_BATCH_SIZE) {
      if (ggml_is_quantized(src0->type)) {
        const int cMmvqMmidMax = get_mmvq_mmid_max_batch(src0->type, cCc);
        if (ne2 <= cMmvqMmidMax) {
          ggml_cuda_mul_mat_vec_q(pCtx, src0, src1, vIds, dst);
          return;
        }
      } else {
        if (GGML_CUDA_CC_IS_AMD(cCc)) {
          ggml_cuda_mul_mat_vec_f(pCtx, src0, src1, vIds, dst);
          return;
        }
      }
    }

    if (ggml_cuda_should_use_mmq(src0->type, cCc, ne12, /*n_experts=*/ne02)) {
      ggml_cuda_mul_mat_q(pCtx, src0, src1, vIds, dst);
      return;
    }

    if (ggml_cuda_should_use_mmf(src0->type, cCc, WARP_SIZE, src0->ne, src0->nb, src1->ne[2], /*mul_mat_id=*/true)) {
      ggml_cuda_mul_mat_f(pCtx, src0, src1, vIds, dst);
      return;
    }
  }

  // note: this path should not be reached when recording CUDA graphs, because it requires stream synchronization
  GGML_ASSERT(fGgmlCudaMulMatIdNeedsSync(dst, cCc));
  cudaStream_t vStream = pCtx.stream();

  GGML_ASSERT(nb12 % nb11 == 0);
  GGML_ASSERT(nb2 % nb1 == 0);

  const ggml_type cTypeSrc1Sorted =
      (src0->type == GGML_TYPE_F16 && !fast_fp16_hardware_available(cCc)) || ggml_is_quantized(src0->type)
          ? GGML_TYPE_F32
          : src0->type;
  const ggml_type cTypeDstSorted = GGML_TYPE_F32;
  const size_t cTsSrc1Sorted = ggml_type_size(cTypeSrc1Sorted);
  const size_t cTsDstSorted = ggml_type_size(cTypeDstSorted);

  const int64_t cNExpertUsed = vIds->ne[0];
  const int64_t cNeGetRows = ne12 * cNExpertUsed;

  std::vector<int32_t> vIdsToSortedHost;
  vIdsToSortedHost.reserve(2 * cNeGetRows);
  std::vector<int32_t> lIdsFromSortedHost(cNeGetRows);

  ggml_cuda_pool_alloc<int32_t> vIdsBufDev(pCtx.pool(), 2 * cNeGetRows);

  std::vector<int32_t> lTokensPerExpert(ne02);

  ggml_cuda_pool_alloc<char> vSrc1Sorted(pCtx.pool(), ne12 * cNExpertUsed * ne10 * cTsSrc1Sorted);
  ggml_cuda_pool_alloc<char> vDstSorted(pCtx.pool(), ne2 * cNExpertUsed * ne0 * cTsDstSorted);

  std::vector<char> vIdsHost(ggml_nbytes(vIds));
  CUDA_CHECK(cudaMemcpyAsync(vIdsHost.data(), vIds->data, ggml_nbytes(vIds), cudaMemcpyDeviceToHost, vStream));
  CUDA_CHECK(cudaStreamSynchronize(vStream));

  for (int64_t vI02 = 0; vI02 < ne02; ++vI02) {   // expert matrices
    for (int64_t vI12 = 0; vI12 < ne12; ++vI12) { // tokens
      for (int64_t vIex = 0; vIex < cNExpertUsed; ++vIex) {
        const int32_t cExpertToUse = *(const int32_t *)(vIdsHost.data() + vI12 * vIds->nb[1] + vIex * vIds->nb[0]);
        assert(cExpertToUse >= 0 && cExpertToUse < ne02);
        if (cExpertToUse == vI02) {
          lIdsFromSortedHost[vI12 * cNExpertUsed + vIex] = vIdsToSortedHost.size();
          vIdsToSortedHost.push_back(vI12 * ne11 + vIex % ne11);
          lTokensPerExpert[vI02]++;
          break;
        }
      }
    }
  }
  GGML_ASSERT(vIdsToSortedHost.size() == size_t(cNeGetRows));

  vIdsToSortedHost.insert(vIdsToSortedHost.end(), lIdsFromSortedHost.begin(), lIdsFromSortedHost.end());

  CUDA_CHECK(cudaMemcpyAsync(vIdsBufDev.ptr, vIdsToSortedHost.data(), 2 * cNeGetRows * sizeof(int32_t),
                             cudaMemcpyHostToDevice, vStream));
  CUDA_CHECK(cudaStreamSynchronize(vStream));

  const int32_t *vIdsToSorted = vIdsBufDev.ptr + 0 * cNeGetRows;
  const int32_t *vIdsFromSorted = vIdsBufDev.ptr + 1 * cNeGetRows;

  get_rows_cuda(src1->data, src1->type, vIdsToSorted, vSrc1Sorted.ptr, cTypeSrc1Sorted, ne10, nb11, nb12, nb13,
                cNeGetRows, 1, 1, sizeof(int32_t), cNeGetRows * sizeof(int32_t), cNeGetRows * sizeof(int32_t),
                ne10 * cTsSrc1Sorted, cNeGetRows * ne10 * cTsSrc1Sorted, cNeGetRows * ne10 * cTsSrc1Sorted, vStream);
  CUDA_CHECK(cudaGetLastError());

  char *vSrc1DataCur = (char *)vSrc1Sorted.ptr;
  char *vDstDataCur = (char *)vDstSorted.ptr;
  for (int64_t vI02 = 0; vI02 < ne02; ++vI02) {
    if (lTokensPerExpert[vI02] == 0) {
      continue;
    }

    ggml_tensor vSrc0Slice = *src0;
    vSrc0Slice.ne[2] = 1;
    vSrc0Slice.nb[3] = vSrc0Slice.nb[2];
    vSrc0Slice.op = GGML_OP_VIEW;
    vSrc0Slice.view_src = dst->src[0]; // non-const pointer to src0
    vSrc0Slice.data = (char *)src0->data + vI02 * nb02;

    ggml_tensor vSrc1Slice;
    memset(&vSrc1Slice, 0, sizeof(vSrc1Slice));
    vSrc1Slice.buffer = src1->buffer;
    vSrc1Slice.type = cTypeSrc1Sorted;
    vSrc1Slice.ne[0] = ne10;
    vSrc1Slice.ne[1] = lTokensPerExpert[vI02];
    vSrc1Slice.ne[2] = 1;
    vSrc1Slice.ne[3] = 1;
    vSrc1Slice.nb[0] = cTsSrc1Sorted;
    vSrc1Slice.nb[1] = vSrc1Slice.ne[0] * vSrc1Slice.nb[0];
    vSrc1Slice.nb[2] = vSrc1Slice.ne[1] * vSrc1Slice.nb[1];
    vSrc1Slice.nb[3] = vSrc1Slice.ne[2] * vSrc1Slice.nb[2];
    vSrc1Slice.data = vSrc1DataCur;

    ggml_tensor vDstSlice;
    memset(&vDstSlice, 0, sizeof(vDstSlice));
    vDstSlice.buffer = dst->buffer;
    vDstSlice.type = cTypeDstSorted;
    vDstSlice.ne[0] = ne0;
    vDstSlice.ne[1] = lTokensPerExpert[vI02];
    vDstSlice.ne[2] = 1;
    vDstSlice.ne[3] = 1;
    vDstSlice.nb[0] = cTsDstSorted;
    vDstSlice.nb[1] = vDstSlice.ne[0] * vDstSlice.nb[0];
    vDstSlice.nb[2] = vDstSlice.ne[1] * vDstSlice.nb[1];
    vDstSlice.nb[3] = vDstSlice.ne[2] * vDstSlice.nb[2];
    vDstSlice.data = vDstDataCur;

    fGgmlCudaMulMat(pCtx, &vSrc0Slice, &vSrc1Slice, &vDstSlice);
    CUDA_CHECK(cudaGetLastError());

    vSrc1DataCur += vSrc1Slice.nb[2];
    vDstDataCur += vDstSlice.nb[2];
  }

  get_rows_cuda(vDstSorted.ptr, cTypeDstSorted, vIdsFromSorted, dst->data, dst->type, ne0, ne0 * cTsDstSorted,
                cNeGetRows * ne0 * cTsDstSorted, cNeGetRows * ne0 * cTsDstSorted, cNeGetRows, 1, 1, sizeof(int32_t),
                cNeGetRows * sizeof(int32_t), cNeGetRows * sizeof(int32_t), nb1, nb2, nb3, vStream);
}

static bool fGgmlCudaComputeForward(ggml_backend_cuda_context &pCtx, struct ggml_tensor *pDst) {
  switch (pDst->op) {
  case GGML_OP_ARGMAX:
    ggml_cuda_argmax(pCtx, pDst);
    break;
  case GGML_OP_AMAX_POW2:
    ggml_cuda_op_amax_pow2(pCtx, pDst);
    break;
  case GGML_OP_COUNT_EQUAL:
    ggml_cuda_count_equal(pCtx, pDst);
    break;
  case GGML_OP_REPEAT:
    ggml_cuda_op_repeat(pCtx, pDst);
    break;
  case GGML_OP_REPEAT_BACK:
    ggml_cuda_op_repeat_back(pCtx, pDst);
    break;
  case GGML_OP_GET_ROWS:
    ggml_cuda_op_get_rows(pCtx, pDst);
    break;
  case GGML_OP_GET_ROWS_BACK:
    ggml_cuda_op_get_rows_back(pCtx, pDst);
    break;
  case GGML_OP_SET_ROWS:
    ggml_cuda_op_set_rows(pCtx, pDst);
    break;
  case GGML_OP_SET:
    ggml_cuda_op_set(pCtx, pDst);
    break;
  case GGML_OP_DUP:
    ggml_cuda_dup(pCtx, pDst);
    break;
  case GGML_OP_CPY:
    ggml_cuda_cpy(pCtx, pDst->src[0], pDst->src[1]);
    break;
  case GGML_OP_CONT:
    ggml_cuda_dup(pCtx, pDst);
    break;
  case GGML_OP_ADD:
  case GGML_OP_ADD1: // TODO: more efficient implementation
    ggml_cuda_op_add(pCtx, pDst);
    break;
  case GGML_OP_ADD_ID:
    ggml_cuda_op_add_id(pCtx, pDst);
    break;
  case GGML_OP_SUB:
    ggml_cuda_op_sub(pCtx, pDst);
    break;
  case GGML_OP_ACC:
    ggml_cuda_op_acc(pCtx, pDst);
    break;
  case GGML_OP_MUL:
    ggml_cuda_op_mul(pCtx, pDst);
    break;
  case GGML_OP_DIV:
    ggml_cuda_op_div(pCtx, pDst);
    break;
  case GGML_OP_UNARY:
    switch (ggml_get_unary_op(pDst)) {
    case GGML_UNARY_OP_ABS:
      ggml_cuda_op_abs(pCtx, pDst);
      break;
    case GGML_UNARY_OP_SGN:
      ggml_cuda_op_sgn(pCtx, pDst);
      break;
    case GGML_UNARY_OP_NEG:
      ggml_cuda_op_neg(pCtx, pDst);
      break;
    case GGML_UNARY_OP_STEP:
      ggml_cuda_op_step(pCtx, pDst);
      break;
    case GGML_UNARY_OP_GELU:
      ggml_cuda_op_gelu(pCtx, pDst);
      break;
    case GGML_UNARY_OP_SILU:
      ggml_cuda_op_silu(pCtx, pDst);
      break;
    case GGML_UNARY_OP_GELU_ERF:
      ggml_cuda_op_gelu_erf(pCtx, pDst);
      break;
    case GGML_UNARY_OP_GELU_QUICK:
      ggml_cuda_op_gelu_quick(pCtx, pDst);
      break;
    case GGML_UNARY_OP_TANH:
      ggml_cuda_op_tanh(pCtx, pDst);
      break;
    case GGML_UNARY_OP_RELU:
      ggml_cuda_op_relu(pCtx, pDst);
      break;
    case GGML_UNARY_OP_SIGMOID:
      ggml_cuda_op_sigmoid(pCtx, pDst);
      break;
    case GGML_UNARY_OP_HARDSIGMOID:
      ggml_cuda_op_hardsigmoid(pCtx, pDst);
      break;
    case GGML_UNARY_OP_HARDSWISH:
      ggml_cuda_op_hardswish(pCtx, pDst);
      break;
    case GGML_UNARY_OP_EXP:
      ggml_cuda_op_exp(pCtx, pDst);
      break;
    case GGML_UNARY_OP_ELU:
      ggml_cuda_op_elu(pCtx, pDst);
      break;
    case GGML_UNARY_OP_XIELU:
      ggml_cuda_op_xielu(pCtx, pDst);
      break;
    case GGML_UNARY_OP_FLOOR:
      ggml_cuda_op_floor(pCtx, pDst);
      break;
    case GGML_UNARY_OP_CEIL:
      ggml_cuda_op_ceil(pCtx, pDst);
      break;
    case GGML_UNARY_OP_ROUND:
      ggml_cuda_op_round(pCtx, pDst);
      break;
    case GGML_UNARY_OP_TRUNC:
      ggml_cuda_op_trunc(pCtx, pDst);
      break;
    case GGML_UNARY_OP_EXPM1:
      ggml_cuda_op_expm1(pCtx, pDst);
      break;
    case GGML_UNARY_OP_SOFTPLUS:
      ggml_cuda_op_softplus(pCtx, pDst);
      break;
    default:
      return false;
    }
    break;
  case GGML_OP_GLU:
    switch (ggml_get_glu_op(pDst)) {
    case GGML_GLU_OP_REGLU:
      ggml_cuda_op_reglu(pCtx, pDst);
      break;
    case GGML_GLU_OP_GEGLU:
      ggml_cuda_op_geglu(pCtx, pDst);
      break;
    case GGML_GLU_OP_SWIGLU:
      ggml_cuda_op_swiglu(pCtx, pDst);
      break;
    case GGML_GLU_OP_SWIGLU_OAI:
      ggml_cuda_op_swiglu_oai(pCtx, pDst);
      break;
    case GGML_GLU_OP_GEGLU_ERF:
      ggml_cuda_op_geglu_erf(pCtx, pDst);
      break;
    case GGML_GLU_OP_GEGLU_QUICK:
      ggml_cuda_op_geglu_quick(pCtx, pDst);
      break;
    case GGML_GLU_OP_SWIGLU_CLAMP:
      ggml_cuda_op_swiglu_clamp(pCtx, pDst);
      break;
    default:
      return false;
    }
    break;
  case GGML_OP_NORM:
    ggml_cuda_op_norm(pCtx, pDst);
    break;
  case GGML_OP_GROUP_NORM:
    ggml_cuda_op_group_norm(pCtx, pDst);
    break;
  case GGML_OP_L2_NORM:
    ggml_cuda_op_l2_norm(pCtx, pDst);
    break;
  case GGML_OP_CONCAT:
    ggml_cuda_op_concat(pCtx, pDst);
    break;
  case GGML_OP_UPSCALE:
    ggml_cuda_op_upscale(pCtx, pDst);
    break;
  case GGML_OP_PAD:
    ggml_cuda_op_pad(pCtx, pDst);
    break;
  case GGML_OP_PAD_REFLECT_1D:
    ggml_cuda_op_pad_reflect_1d(pCtx, pDst);
    break;
  case GGML_OP_ARANGE:
    ggml_cuda_op_arange(pCtx, pDst);
    break;
  case GGML_OP_TIMESTEP_EMBEDDING:
    ggml_cuda_op_timestep_embedding(pCtx, pDst);
    break;
  case GGML_OP_LEAKY_RELU:
    ggml_cuda_op_leaky_relu(pCtx, pDst);
    break;
  case GGML_OP_SILU_BACK:
    ggml_cuda_op_silu_back(pCtx, pDst);
    break;
  case GGML_OP_RMS_NORM:
    ggml_cuda_op_rms_norm(pCtx, pDst);
    break;
  case GGML_OP_RMS_NORM_BACK:
    ggml_cuda_op_rms_norm_back(pCtx, pDst);
    break;
  case GGML_OP_MUL_MAT:
    fGgmlCudaMulMat(pCtx, pDst->src[0], pDst->src[1], pDst);
    break;
  case GGML_OP_MUL_MAT_ID:
    fGgmlCudaMulMatId(pCtx, pDst);
    break;
  case GGML_OP_OUT_PROD:
    ggml_cuda_out_prod(pCtx, pDst);
    break;
  case GGML_OP_SCALE:
    ggml_cuda_op_scale(pCtx, pDst);
    break;
  case GGML_OP_SQR:
    ggml_cuda_op_sqr(pCtx, pDst);
    break;
  case GGML_OP_SQRT:
    ggml_cuda_op_sqrt(pCtx, pDst);
    break;
  case GGML_OP_SIN:
    ggml_cuda_op_sin(pCtx, pDst);
    break;
  case GGML_OP_COS:
    ggml_cuda_op_cos(pCtx, pDst);
    break;
  case GGML_OP_CLAMP:
    ggml_cuda_op_clamp(pCtx, pDst);
    break;
  case GGML_OP_LOG:
    ggml_cuda_op_log(pCtx, pDst);
    break;
  case GGML_OP_NONE:
  case GGML_OP_RESHAPE:
  case GGML_OP_VIEW:
  case GGML_OP_PERMUTE:
  case GGML_OP_TRANSPOSE:
    break;
  case GGML_OP_DIAG:
    ggml_cuda_op_diag(pCtx, pDst);
    break;
  case GGML_OP_DIAG_MASK_INF:
    ggml_cuda_op_diag_mask_inf(pCtx, pDst);
    break;
  case GGML_OP_SOFT_MAX:
    ggml_cuda_op_soft_max(pCtx, pDst);
    break;
  case GGML_OP_SOFT_MAX_BACK:
    ggml_cuda_op_soft_max_back(pCtx, pDst);
    break;
  case GGML_OP_ROPE:
    ggml_cuda_op_rope(pCtx, pDst);
    break;
  case GGML_OP_ROPE_BACK:
    ggml_cuda_op_rope_back(pCtx, pDst);
    break;
  case GGML_OP_ROLL:
    ggml_cuda_op_roll(pCtx, pDst);
    break;
  case GGML_OP_IM2COL:
    ggml_cuda_op_im2col(pCtx, pDst);
    break;
  case GGML_OP_IM2COL_3D:
    ggml_cuda_op_im2col_3d(pCtx, pDst);
    break;
  case GGML_OP_CONV_2D:
    ggml_cuda_op_conv2d(pCtx, pDst);
    break;
  case GGML_OP_CONV_2D_DW:
    ggml_cuda_op_conv2d_dw(pCtx, pDst);
    break;
  case GGML_OP_CONV_TRANSPOSE_2D:
    ggml_cuda_conv_2d_transpose_p0(pCtx, pDst);
    break;
  case GGML_OP_CONV_TRANSPOSE_1D:
    ggml_cuda_op_conv_transpose_1d(pCtx, pDst);
    break;
  case GGML_OP_COL2IM_1D:
    ggml_cuda_op_col2im_1d(pCtx, pDst);
    break;
  case GGML_OP_POOL_2D:
    ggml_cuda_op_pool2d(pCtx, pDst);
    break;
  case GGML_OP_SUM:
    ggml_cuda_op_sum(pCtx, pDst);
    break;
  case GGML_OP_CUMSUM:
    ggml_cuda_op_cumsum(pCtx, pDst);
    break;
  case GGML_OP_SUM_ROWS:
    ggml_cuda_op_sum_rows(pCtx, pDst);
    break;
  case GGML_OP_MEAN:
    ggml_cuda_op_mean(pCtx, pDst);
    break;
  case GGML_OP_SSM_CONV:
    ggml_cuda_op_ssm_conv(pCtx, pDst);
    break;
  case GGML_OP_SSM_SCAN:
    ggml_cuda_op_ssm_scan(pCtx, pDst);
    break;
  case GGML_OP_TOP_K:
    ggml_cuda_op_top_k(pCtx, pDst);
    break;
  case GGML_OP_ARGSORT:
    ggml_cuda_op_argsort(pCtx, pDst);
    break;
  case GGML_OP_FLASH_ATTN_EXT:
    ggml_cuda_flash_attn_ext(pCtx, pDst);
    break;
  case GGML_OP_CROSS_ENTROPY_LOSS:
    ggml_cuda_cross_entropy_loss(pCtx, pDst);
    break;
  case GGML_OP_TRI:
    ggml_cuda_op_tri(pCtx, pDst);
    break;
  case GGML_OP_RWKV_WKV6:
    ggml_cuda_op_rwkv_wkv6(pCtx, pDst);
    break;
  case GGML_OP_GATED_LINEAR_ATTN:
    ggml_cuda_op_gated_linear_attn(pCtx, pDst);
    break;
  case GGML_OP_GATED_DELTA_NET:
    ggml_cuda_op_gated_delta_net(pCtx, pDst);
    break;
  case GGML_OP_DSV4_HC_COMB:
    ggml_cuda_op_dsv4_hc_comb(pCtx, pDst);
    break;
  case GGML_OP_DSV4_HC_PRE:
    ggml_cuda_op_dsv4_hc_pre(pCtx, pDst);
    break;
  case GGML_OP_DSV4_HC_POST:
    ggml_cuda_op_dsv4_hc_post(pCtx, pDst);
    break;
  case GGML_OP_RWKV_WKV7:
    ggml_cuda_op_rwkv_wkv7(pCtx, pDst);
    break;
  case GGML_OP_CROSS_ENTROPY_LOSS_BACK:
    ggml_cuda_cross_entropy_loss_back(pCtx, pDst);
    break;
  case GGML_OP_OPT_STEP_ADAMW:
    ggml_cuda_opt_step_adamw(pCtx, pDst);
    break;
  case GGML_OP_OPT_STEP_SGD:
    ggml_cuda_opt_step_sgd(pCtx, pDst);
    break;
  case GGML_OP_SOLVE_TRI:
    ggml_cuda_op_solve_tri(pCtx, pDst);
    break;
  case GGML_OP_FILL:
    ggml_cuda_op_fill(pCtx, pDst);
    break;
  case GGML_OP_LIGHTNING_INDEXER:
    ggml_cuda_lightning_indexer(pCtx, pDst);
    break;
  default:
    return false;
  }

  cudaError_t vErr = cudaGetLastError();
  if (vErr != cudaSuccess) {
    GGML_LOG_ERROR("%s: %s failed\n", __func__, ggml_op_desc(pDst));
    CUDA_CHECK(vErr);
  }

  return true;
}

////////////////////////////////////////////////////////////////////////////////

// backend

static const char *fGgmlBackendCudaGetName(ggml_backend_t pBackend) {
  ggml_backend_cuda_context *vCudaCtx = (ggml_backend_cuda_context *)pBackend->context;

  return vCudaCtx->name.c_str();
}

static void fGgmlBackendCudaFree(ggml_backend_t pBackend) {
  ggml_backend_cuda_context *vCudaCtx = (ggml_backend_cuda_context *)pBackend->context;

  delete vCudaCtx;
  delete pBackend;
}

static void fGgmlBackendCudaSetTensorAsync(ggml_backend_t pBackend, ggml_tensor *pTensor, const void *pData,
                                           size_t pOffset, size_t pSize) {
  ggml_backend_cuda_context *vCudaCtx = (ggml_backend_cuda_context *)pBackend->context;
  ggml_backend_buffer_t vBuf = pTensor->view_src ? pTensor->view_src->buffer : pTensor->buffer;

  GGML_ASSERT(vBuf->buft == ggml_backend_cuda_buffer_type(vCudaCtx->device) && "unsupported buffer type");

  CUDA_CHECK(
      cudaMemcpyAsync((char *)pTensor->data + pOffset, pData, pSize, cudaMemcpyHostToDevice, vCudaCtx->stream()));
}

static void fGgmlBackendCudaGetTensorAsync(ggml_backend_t pBackend, const ggml_tensor *pTensor, void *pData,
                                           size_t pOffset, size_t pSize) {
  ggml_backend_cuda_context *vCudaCtx = (ggml_backend_cuda_context *)pBackend->context;
  ggml_backend_buffer_t vBuf = pTensor->view_src ? pTensor->view_src->buffer : pTensor->buffer;

  GGML_ASSERT(vBuf->buft == ggml_backend_cuda_buffer_type(vCudaCtx->device) && "unsupported buffer type");

  CUDA_CHECK(
      cudaMemcpyAsync(pData, (const char *)pTensor->data + pOffset, pSize, cudaMemcpyDeviceToHost, vCudaCtx->stream()));
}

static void fGgmlBackendCudaSetTensor2dAsync(ggml_backend_t pBackend, struct ggml_tensor *pTensor, const void *pData,
                                             size_t pOffset, size_t pSize, size_t pNCopies, size_t pStrideTensor,
                                             size_t pStrideData) {
  ggml_backend_cuda_context *vCudaCtx = (ggml_backend_cuda_context *)pBackend->context;
  ggml_backend_buffer_t vBuf = pTensor->view_src ? pTensor->view_src->buffer : pTensor->buffer;

  GGML_ASSERT(vBuf->buft == ggml_backend_cuda_buffer_type(vCudaCtx->device) && "unsupported buffer type");

  CUDA_CHECK(cudaMemcpy2DAsync((char *)pTensor->data + pOffset, pStrideTensor, pData, pStrideData, pSize, pNCopies,
                               cudaMemcpyHostToDevice, vCudaCtx->stream()));
}

static void fGgmlBackendCudaGetTensor2dAsync(ggml_backend_t pBackend, const struct ggml_tensor *pTensor, void *pData,
                                             size_t pOffset, size_t pSize, size_t pNCopies, size_t pStrideTensor,
                                             size_t pStrideData) {
  ggml_backend_cuda_context *vCudaCtx = (ggml_backend_cuda_context *)pBackend->context;
  ggml_backend_buffer_t vBuf = pTensor->view_src ? pTensor->view_src->buffer : pTensor->buffer;

  GGML_ASSERT(vBuf->buft == ggml_backend_cuda_buffer_type(vCudaCtx->device) && "unsupported buffer type");

  CUDA_CHECK(cudaMemcpy2DAsync(pData, pStrideData, (const char *)pTensor->data + pOffset, pStrideTensor, pSize,
                               pNCopies, cudaMemcpyDeviceToHost, vCudaCtx->stream()));
}

static bool fGgmlBackendCudaCpyTensorAsync(ggml_backend_t pBackendSrc, ggml_backend_t pBackendDst,
                                           const ggml_tensor *pSrc, ggml_tensor *pDst) {
  ggml_backend_buffer_t vBufSrc = pSrc->view_src ? pSrc->view_src->buffer : pSrc->buffer;
  ggml_backend_buffer_t vBufDst = pDst->view_src ? pDst->view_src->buffer : pDst->buffer;

  if (!ggml_backend_is_cuda(pBackendSrc) || !ggml_backend_is_cuda(pBackendDst)) {
    return false;
  }

  if (!fGgmlBackendBufferIsCuda(vBufSrc) || !fGgmlBackendBufferIsCuda(vBufDst)) {
    return false;
  }

  // device -> device copy
  ggml_backend_cuda_context *vCudaCtxSrc = (ggml_backend_cuda_context *)pBackendSrc->context;
  ggml_backend_cuda_context *vCudaCtxDst = (ggml_backend_cuda_context *)pBackendDst->context;

  ggml_backend_cuda_buffer_context *vBufCtxSrc = (ggml_backend_cuda_buffer_context *)vBufSrc->context;
  ggml_backend_cuda_buffer_context *vBufCtxDst = (ggml_backend_cuda_buffer_context *)vBufDst->context;

  if (vCudaCtxSrc->device != vBufCtxSrc->device || vCudaCtxDst->device != vBufCtxDst->device) {
#ifndef NDEBUG
    GGML_LOG_DEBUG("%s: backend and buffer devices do not match\n", __func__);
#endif // NDEBUG
    return false;
  }

  if (pBackendSrc != pBackendDst) {
    // copy on src stream
    // compare the backing physical devices: distinct virtual devices may share one physical GPU,
    // in which case a same-device copy (not a peer copy) is required
    const int cSrcPhysical = fGgmlCudaGetPhysicalDevice(vCudaCtxSrc->device);
    const int cDstPhysical = fGgmlCudaGetPhysicalDevice(vCudaCtxDst->device);
    if (cSrcPhysical == cDstPhysical) {
      CUDA_CHECK(
          cudaMemcpyAsync(pDst->data, pSrc->data, ggml_nbytes(pDst), cudaMemcpyDeviceToDevice, vCudaCtxSrc->stream()));
    } else {
#ifdef GGML_CUDA_NO_PEER_COPY
      return false;
#else
      CUDA_CHECK(cudaMemcpyPeerAsync(pDst->data, cDstPhysical, pSrc->data, cSrcPhysical, ggml_nbytes(pDst),
                                     vCudaCtxSrc->stream()));
#endif // GGML_CUDA_NO_PEER_COPY
    }

    // record event on src stream after the copy
    if (!vCudaCtxSrc->copy_event) {
      ggml_cuda_set_device(vCudaCtxSrc->device);
      CUDA_CHECK(cudaEventCreateWithFlags(&vCudaCtxSrc->copy_event, cudaEventDisableTiming));
    }

    CUDA_CHECK(cudaEventRecord(vCudaCtxSrc->copy_event, vCudaCtxSrc->stream()));

    // wait on dst stream for the copy to complete
    CUDA_CHECK(cudaStreamWaitEvent(vCudaCtxDst->stream(), vCudaCtxSrc->copy_event, 0));
  } else {
    // src and dst are on the same backend
    CUDA_CHECK(
        cudaMemcpyAsync(pDst->data, pSrc->data, ggml_nbytes(pDst), cudaMemcpyDeviceToDevice, vCudaCtxSrc->stream()));
  }
  return true;
}

static void fGgmlBackendCudaSynchronize(ggml_backend_t pBackend) {
  ggml_backend_cuda_context *vCudaCtx = (ggml_backend_cuda_context *)pBackend->context;

  CUDA_CHECK(cudaStreamSynchronize(vCudaCtx->stream()));

  GGML_UNUSED(pBackend);
}

static bool fGgmlCudaIsViewOrNoop(const ggml_tensor *pT) {
  return ggml_is_empty(pT) || pT->op == GGML_OP_RESHAPE || pT->op == GGML_OP_TRANSPOSE || pT->op == GGML_OP_VIEW ||
         pT->op == GGML_OP_PERMUTE || pT->op == GGML_OP_NONE;
}

#ifdef USE_CUDA_GRAPH
static bool fGgmlCudaGraphCheckCompability(ggml_cgraph *pCgraph) {

  bool vUseCudaGraph = true;
  // Loop over nodes in GGML graph to obtain info needed for CUDA graph

  for (int vI = 0; vI < pCgraph->n_nodes; vI++) {
    ggml_tensor *vNode = pCgraph->nodes[vI];

    if (fGgmlCudaIsViewOrNoop(vNode)) {
      continue;
    }

    // [TAG_MUL_MAT_ID_CUDA_GRAPHS]
    if (vNode->op == GGML_OP_MUL_MAT_ID) {
      const int cCc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
      if (fGgmlCudaMulMatIdNeedsSync(vNode, cCc)) {
        // the mul_mat_id fallback path synchronizes the stream, so we cannot use CUDA graphs
        vUseCudaGraph = false;
#ifndef NDEBUG
        GGML_LOG_DEBUG("%s: disabling CUDA graphs due to unsupported node type\n", __func__);
#endif
      }
    }

    if (!vUseCudaGraph) {
      break;
    }
  }

  return vUseCudaGraph;
}

static const void *fGgmlCudaGraphGetKey(ggml_cgraph *pCgraph) { return pCgraph->nodes[0]; }

static bool fGgmlCudaGraphUpdateRequired(ggml_backend_cuda_context *pCudaCtx, ggml_cgraph *pCgraph) {
  bool vRes = false;

  const void *vGraphKey = fGgmlCudaGraphGetKey(pCgraph);
  ggml_cuda_graph *vGraph = pCudaCtx->cuda_graph(vGraphKey);

  if (pCgraph->uid != 0 && pCgraph->uid == vGraph->uid) {
    GGML_LOG_DEBUG("CUDA Graph id %zu reused\n", pCgraph->uid);
    GGML_ASSERT((int)vGraph->node_props.size() == pCgraph->n_nodes);
    return false;
  }

  vGraph->uid = pCgraph->uid;

  // Check if the graph size has changed
  if ((int)vGraph->node_props.size() != pCgraph->n_nodes) {
    vRes = true;
    vGraph->node_props.resize(pCgraph->n_nodes);
  }

  for (int vI = 0; vI < pCgraph->n_nodes; vI++) {
    ggml_cuda_graph::node_properties vProp = {};
    memcpy(&vProp.node, pCgraph->nodes[vI], sizeof(ggml_tensor));

    for (int vJ = 0; vJ < GGML_MAX_SRC; ++vJ) {
      if (pCgraph->nodes[vI]->src[vJ]) {
        vProp.node_src_data_ptrs[vJ] = pCgraph->nodes[vI]->src[vJ]->data;
        memcpy(vProp.node_src_ne[vJ], pCgraph->nodes[vI]->src[vJ]->ne, sizeof(vProp.node_src_ne[vJ]));
        memcpy(vProp.node_src_nb[vJ], pCgraph->nodes[vI]->src[vJ]->nb, sizeof(vProp.node_src_nb[vJ]));
      }
    }

    if (vRes || memcmp(&vGraph->node_props[vI], &vProp, sizeof(vProp)) != 0) {
      vGraph->node_props[vI] = vProp;
      vRes = true;
    }
  }

  return vRes;
}

static void fGgmlCudaGraphUpdateExecutable(ggml_backend_cuda_context *pCudaCtx, const void *pGraphKey) {
  ggml_cuda_graph *vGraph = pCudaCtx->cuda_graph(pGraphKey);

#if CUDART_VERSION >= 12000
  cudaGraphExecUpdateResultInfo vResultInfo;
  cudaError_t vStat = cudaGraphExecUpdate(vGraph->instance, vGraph->graph, &vResultInfo);
#else
  cudaGraphNode_t errorNode;
  cudaGraphExecUpdateResult vResultInfo;
  cudaError_t vStat = cudaGraphExecUpdate(vGraph->instance, vGraph->graph, &errorNode, &vResultInfo);
#endif // CUDART_VERSION >= 12000

  if (vStat == cudaErrorGraphExecUpdateFailure) {
#ifndef NDEBUG
    GGML_LOG_DEBUG("%s: CUDA graph update failed\n", __func__);
#endif

    // The pre-existing graph exec cannot be updated due to violated constraints
    // so instead clear error and re-instantiate
    (void)cudaGetLastError();
    CUDA_CHECK(cudaGraphExecDestroy(vGraph->instance));
    vGraph->instance = nullptr;
    CUDA_CHECK(cudaGraphInstantiate(&vGraph->instance, vGraph->graph, NULL, NULL, 0));
  } else {
    GGML_ASSERT(vStat == cudaSuccess);
  }
}
#endif // USE_CUDA_GRAPH

static bool fGgmlCudaShouldFuseRopeSetRows(const ggml_tensor *pRope, const ggml_tensor *pView,
                                           const ggml_tensor *pSetRows) {

  if (pRope->op != GGML_OP_ROPE || pView->op != GGML_OP_VIEW || pSetRows->op != GGML_OP_SET_ROWS) {
    return false;
  }
  // ne3 not tested
  if (pRope->src[0]->ne[3] != 1) {
    return false;
  }

  if (pSetRows->type != GGML_TYPE_F32 && pSetRows->type != GGML_TYPE_F16) {
    return false;
  }

  if (pSetRows->src[1]->type != GGML_TYPE_I64) {
    return false;
  }

  // The view should flatten two dims of rope into one dim
  if (!ggml_is_contiguous(pView) || pView->ne[0] != pRope->ne[0] * pRope->ne[1]) {
    return false;
  }

  // Only norm/neox shaders have the fusion code
  const int cMode = ((const int32_t *)pRope->op_params)[2];
  if (cMode != GGML_ROPE_TYPE_NORMAL && cMode != GGML_ROPE_TYPE_NEOX) {
    return false;
  }

  return true;
}

// match gated_delta_net + the strided cpy that scatters its state snapshots into the cache
// (slot i -> rollback group i, slot 0 newest), so the kernel can write them and skip the cpy.
static int fGgmlCudaTryGdnCacheFusion(const ggml_cgraph *pCgraph, int pNodeIdx,
                                      ggml_cuda_gated_delta_net_fused_cache &pFusedStateCpy) {
  const ggml_tensor *vGdn = pCgraph->nodes[pNodeIdx];
  // the kernel skips the snapshot tail, so the gdn output must not be a graph output
  if (vGdn->op != GGML_OP_GATED_DELTA_NET || vGdn->type != GGML_TYPE_F32 || (vGdn->flags & GGML_TENSOR_FLAG_OUTPUT)) {
    return 0;
  }

  const ggml_tensor *vSrcV = vGdn->src[2];
  const int64_t cSV = vSrcV->ne[0];
  const int64_t cH = vSrcV->ne[1];
  const int64_t cNTokens = vSrcV->ne[2];
  const int64_t cNSeqs = vSrcV->ne[3];
  const int64_t cD = cSV * cSV * cH;
  const int64_t cK = ggml_get_op_params_i32(vGdn, 0);        // snapshot slot count
  const int64_t cNWritten = std::min<int64_t>(cNTokens, cK); // newest n_written slots are written

  // snapshot tail starts right after the attention scores
  const size_t cTailOff = ggml_row_size(GGML_TYPE_F32, cSV * cH * cNTokens * cNSeqs);

  // snapshot cpy is the first real node after the gdn (skip views/no-ops)
  const ggml_tensor *vCpy = nullptr;
  int vSkip = 0;
  for (int vJ = pNodeIdx + 1; vJ < pCgraph->n_nodes && vCpy == nullptr; ++vJ) {
    const ggml_tensor *vN = pCgraph->nodes[vJ];
    if (fGgmlCudaIsViewOrNoop(vN)) {
      continue;
    }
    if (vN->op != GGML_OP_CPY || (vN->flags & GGML_TENSOR_FLAG_OUTPUT)) {
      return 0;
    }
    vCpy = vN;
    vSkip = vJ - pNodeIdx;
  }
  if (vCpy == nullptr) {
    return 0;
  }

  const ggml_tensor *vSrc = vCpy->src[0]; // view of the gdn snapshot tail
  const ggml_tensor *vDst = vCpy->src[1]; // cache view the kernel writes to

  // src must be this gdn's snapshot tail (contiguous, at the tail offset)
  if (vSrc->op != GGML_OP_VIEW || vSrc->view_src != vGdn || vSrc->view_offs != cTailOff || !ggml_is_contiguous(vSrc)) {
    return 0;
  }

  // dst is the [D, n_seqs, n_written] cache view; require nb[1] == D (the per-seq stride the kernel
  // assumes). ggml_cpy pins src to the same element count.
  const std::array<int64_t, GGML_MAX_DIMS> aExpectedNe = {cD, cNSeqs, cNWritten, 1};
  if (vDst->op != GGML_OP_VIEW || vDst->type != GGML_TYPE_F32 || vDst->data == nullptr ||
      !std::equal(aExpectedNe.begin(), aExpectedNe.end(), vDst->ne) || vDst->nb[0] != ggml_type_size(GGML_TYPE_F32) ||
      vDst->nb[1] != (size_t)ggml_row_size(GGML_TYPE_F32, cD)) {
    return 0;
  }

  pFusedStateCpy.data = (float *)vDst->data; // rollback group 0 (newest)
  pFusedStateCpy.slot_stride = cK > 1 ? (int64_t)(vDst->nb[2] / sizeof(float)) : 0;
  return vSkip;
}

static bool fGgmlCudaTopkMoeFusion(const struct ggml_cgraph *pCgraph, int pNodeIdx, ggml_cuda_topk_moe_args &pArgs) {
  pArgs.sigmoid = false;
  pArgs.sqrt_softplus = false;
  pArgs.softmax = false;
  pArgs.delayed_softmax = false;
  pArgs.prob_bias = false;
  pArgs.norm = false;

  const int cNNodes = pCgraph->n_nodes;
  ggml_tensor **vNodes = pCgraph->nodes;

  if (vNodes[pNodeIdx]->op == GGML_OP_SOFT_MAX) {
    pArgs.softmax = true;
  }

  if (vNodes[pNodeIdx]->op == GGML_OP_UNARY) {
    const ggml_unary_op cUnaryOp = ggml_get_unary_op(vNodes[pNodeIdx]);
    if (cUnaryOp == GGML_UNARY_OP_SIGMOID) {
      pArgs.sigmoid = true;
    } else if (cUnaryOp == GGML_UNARY_OP_SOFTPLUS && pNodeIdx + 1 < cNNodes &&
               vNodes[pNodeIdx + 1]->op == GGML_OP_SQRT && vNodes[pNodeIdx + 1]->src[0] == vNodes[pNodeIdx]) {
      // sqrt(softplus(x)) scoring (DeepSeek-V4)
      pArgs.sqrt_softplus = true;
      pNodeIdx++;
    } else {
      return false;
    }
  }

  if (vNodes[pNodeIdx]->op == GGML_OP_ARGSORT) {
    pArgs.delayed_softmax = true;
  }

  pNodeIdx++;

  if (pArgs.sigmoid || pArgs.sqrt_softplus || pArgs.softmax) {
    // SOFTMAX -> RESHAPE
    if (pNodeIdx >= cNNodes || vNodes[pNodeIdx]->op != GGML_OP_RESHAPE ||
        vNodes[pNodeIdx]->src[0] != vNodes[pNodeIdx - 1]) {
      return false;
    }
    ggml_tensor *vProbsReshaped = vNodes[pNodeIdx];
    pNodeIdx++;

    if (pNodeIdx >= cNNodes) {
      return false;
    }

    // src of bias add is the unreshaped probs (-2 instead of -1)
    if (vNodes[pNodeIdx]->op == GGML_OP_ADD && vNodes[pNodeIdx]->src[0] == vNodes[pNodeIdx - 2]) {
      pArgs.prob_bias = true;
      pNodeIdx++;
    }
    // RESHAPE/ADD -> ARGSORT
    if (pNodeIdx >= cNNodes || vNodes[pNodeIdx]->op != GGML_OP_ARGSORT) {
      return false;
    }

    if (pArgs.prob_bias && vNodes[pNodeIdx]->src[0] != vNodes[pNodeIdx - 1]) {
      return false;
    } else if (!pArgs.prob_bias && vNodes[pNodeIdx]->src[0] != vNodes[pNodeIdx - 2]) {
      return false;
    }

    pNodeIdx++;

    // ARGSORT-> VIEW
    if (pNodeIdx >= cNNodes || vNodes[pNodeIdx]->op != GGML_OP_VIEW ||
        vNodes[pNodeIdx]->src[0] != vNodes[pNodeIdx - 1]) {
      return false;
    }
    pNodeIdx++;

    if (pNodeIdx >= cNNodes || vNodes[pNodeIdx]->op != GGML_OP_GET_ROWS) {
      return false;
    }

    // GET_ROWS
    if (vNodes[pNodeIdx]->src[0] != vProbsReshaped || vNodes[pNodeIdx]->src[1] != vNodes[pNodeIdx - 1]) {
      return false;
    }
    pNodeIdx++;
  } else if (pArgs.delayed_softmax) {
    if (pNodeIdx - 2 < 0) {
      return false;
    }
    ggml_tensor *vProbsReshaped = vNodes[pNodeIdx - 2];

    // VIEW->ARGSORT
    if (pNodeIdx >= cNNodes || vNodes[pNodeIdx]->op != GGML_OP_VIEW ||
        vNodes[pNodeIdx]->src[0] != vNodes[pNodeIdx - 1]) {
      return false;
    }
    pNodeIdx++;

    // GET_ROWS
    if (pNodeIdx >= cNNodes || vNodes[pNodeIdx]->src[1] != vNodes[pNodeIdx - 1] ||
        vNodes[pNodeIdx]->src[0] != vProbsReshaped) {
      return false;
    }
    pNodeIdx++;

    static const std::vector<ggml_op> lRemainingOps = {GGML_OP_RESHAPE, GGML_OP_SOFT_MAX, GGML_OP_RESHAPE};

    for (const ggml_op cOp : lRemainingOps) {
      if (pNodeIdx >= cNNodes || vNodes[pNodeIdx]->op != cOp || vNodes[pNodeIdx]->src[0] != vNodes[pNodeIdx - 1]) {
        return false;
      }
      pNodeIdx++;
    }
  }

  // At this point we can check for norm + scale. Everything is now at least valid till the norm
  if (pNodeIdx >= cNNodes) {
    return true;
  }

  if (vNodes[pNodeIdx]->op == GGML_OP_RESHAPE) {
    // check RESHAPE->SUM_ROWS->CLAMP->DIV->RESHAPE
    static const std::vector<ggml_op> lNormOps = {GGML_OP_RESHAPE, GGML_OP_SUM_ROWS, GGML_OP_CLAMP};

    pArgs.norm = true;
    for (const ggml_op cOp : lNormOps) {
      if (vNodes[pNodeIdx]->op == cOp && vNodes[pNodeIdx]->src[0] == vNodes[pNodeIdx - 1]) {
        pNodeIdx++;
      } else {
        pArgs.norm = false;
        return true;
      }
    }

    // DIV <- CLAMP, RESHAPE
    if (vNodes[pNodeIdx]->op != GGML_OP_DIV || vNodes[pNodeIdx]->src[1] != vNodes[pNodeIdx - 1] ||
        vNodes[pNodeIdx]->src[0] != vNodes[pNodeIdx - 3]) {
      pArgs.norm = false;
      return true;
    }
    pNodeIdx++;

    if (vNodes[pNodeIdx]->op != GGML_OP_RESHAPE || vNodes[pNodeIdx]->src[0] != vNodes[pNodeIdx - 1]) {
      pArgs.norm = false;
      return true;
    }

    pNodeIdx++;
  }

  if (vNodes[pNodeIdx]->op == GGML_OP_SCALE && vNodes[pNodeIdx]->src[0] == vNodes[pNodeIdx - 1]) {
    pArgs.scale = true;
  }

  return true;
}

// returns whether the write (out) nodes overwrite the read nodes in operation
static bool fGgmlCudaCheckFusionMemoryRanges(const ggml_cgraph *pCgraph, const int pNodeIdx, const int pNodeCount,
                                             const int *pOutNodes, const int pOutCount, const bool pIsTopkMoe = false) {
  auto fNodesOverlap = [&](const ggml_tensor *pA, const ggml_tensor *pB) {
    const int64_t cAStart = (int64_t)pA->data;
    const int64_t cAEnd = cAStart + ggml_backend_buft_get_alloc_size(pA->buffer->buft, pA);

    const int64_t cBStart = (int64_t)pB->data;
    const int64_t cBEnd = cBStart + ggml_backend_buft_get_alloc_size(pB->buffer->buft, pB);

    if ((cBStart <= cAStart && cAStart < cBEnd) || (cAStart <= cBStart && cBStart < cAEnd)) {
      return true;
    }

    return false;
  };

  bool vIsOk = true;
  // one block reads all logits before it writes, so logits may alias the out nodes
  const ggml_tensor *vLogitsMayAlias = nullptr;
  if (pIsTopkMoe && ggml_nrows(pCgraph->nodes[pNodeIdx]) <= TOPK_MOE_ROWS_PER_BLOCK) {
    vLogitsMayAlias = pCgraph->nodes[pNodeIdx]->src[0];
  }

  for (int vI = 0; vI < pOutCount; ++vI) {
    const ggml_tensor *vDst = pCgraph->nodes[pOutNodes[vI]];

    for (int vJ = pNodeIdx; vJ < pNodeIdx + pNodeCount; ++vJ) {
      // Loop over all srcs of all nodes in the fusion. If the src overlaps
      // the destination and the src is not an intermediate node that's being
      // elided, then disable fusion.

      for (int vSrcIdx = 0; vSrcIdx < GGML_MAX_SRC; ++vSrcIdx) {
        const ggml_tensor *vSrc = pCgraph->nodes[vJ]->src[vSrcIdx];

        if (!vSrc || vSrc->op == GGML_OP_NONE || vSrc == vLogitsMayAlias) {
          continue;
        }

        if (fNodesOverlap(vDst, vSrc)) {
          bool vFound = false;

          for (int vK = pNodeIdx; vK < vJ; ++vK) {
            if (pCgraph->nodes[vK] == vSrc) {
              vFound = true;
              break;
            }
          }

          if (!vFound) {
            vIsOk = false;
            break;
          }
        }
      }
    }
  }

  return vIsOk;
}

static bool fGgmlCudaCanFuse(const struct ggml_cgraph *pCgraph, int pNodeIdx, std::initializer_list<enum ggml_op> pOps,
                             std::initializer_list<enum ggml_unary_op> pUnaryOps) {
#ifndef NDEBUG
  const size_t cNumUnary = std::count(pOps.begin(), pOps.end(), GGML_OP_UNARY);
  GGML_ASSERT(pUnaryOps.size() == cNumUnary);
#endif

  const auto fIsEqual = [](const std::initializer_list<enum ggml_op> &pList1,
                           const std::initializer_list<enum ggml_op> &pList2) {
    return std::equal(pList1.begin(), pList1.end(), pList2.begin(), pList2.end());
  };

  std::initializer_list<enum ggml_op> vMulMatBiasGluOps = {GGML_OP_MUL_MAT, GGML_OP_ADD, GGML_OP_MUL_MAT, GGML_OP_ADD,
                                                           GGML_OP_GLU};
  std::initializer_list<enum ggml_op> vMulMatIdBiasGluOps = {GGML_OP_MUL_MAT_ID, GGML_OP_ADD_ID, GGML_OP_MUL_MAT_ID,
                                                             GGML_OP_ADD_ID, GGML_OP_GLU};

  std::initializer_list<enum ggml_op> vMulMatIdGluOps = {GGML_OP_MUL_MAT_ID, GGML_OP_MUL_MAT_ID, GGML_OP_GLU};
  std::initializer_list<enum ggml_op> vMulMatGluOps = {GGML_OP_MUL_MAT, GGML_OP_MUL_MAT, GGML_OP_GLU};

  if ((fIsEqual(vMulMatBiasGluOps, pOps) || fIsEqual(vMulMatIdBiasGluOps, pOps)) &&
      ggml_can_fuse_subgraph(pCgraph, pNodeIdx, pOps, {pNodeIdx + 4})) {
    const ggml_tensor *vFfnGate = pCgraph->nodes[pNodeIdx];
    const ggml_tensor *vFfnGateBias = pCgraph->nodes[pNodeIdx + 1];
    const ggml_tensor *vFfnUp = pCgraph->nodes[pNodeIdx + 2];
    const ggml_tensor *vFfnUpBias = pCgraph->nodes[pNodeIdx + 3];
    const ggml_tensor *vGlu = pCgraph->nodes[pNodeIdx + 4];

    if (fGgmlCudaShouldFuseMulMat(vFfnUp, vFfnGate, vGlu, vFfnUpBias, vFfnGateBias)) {
      int aOutNodes[] = {pNodeIdx + 4};
      return fGgmlCudaCheckFusionMemoryRanges(pCgraph, pNodeIdx, (int)pOps.size(), aOutNodes, 1);
    }
  }

  if ((fIsEqual(vMulMatIdGluOps, pOps) || fIsEqual(vMulMatGluOps, pOps)) &&
      ggml_can_fuse_subgraph(pCgraph, pNodeIdx, pOps, {pNodeIdx + 2})) {
    const ggml_tensor *vFfnGate = pCgraph->nodes[pNodeIdx];
    const ggml_tensor *vFfnUp = pCgraph->nodes[pNodeIdx + 1];
    const ggml_tensor *vGlu = pCgraph->nodes[pNodeIdx + 2];

    if (fGgmlCudaShouldFuseMulMat(vFfnUp, vFfnGate, vGlu)) {
      int aOutNodes[] = {pNodeIdx + 2};
      return fGgmlCudaCheckFusionMemoryRanges(pCgraph, pNodeIdx, (int)pOps.size(), aOutNodes, 1);
    }
  }

  std::initializer_list<enum ggml_op> vRopeSetRowsOps = {GGML_OP_ROPE, GGML_OP_VIEW, GGML_OP_SET_ROWS};

  if (fIsEqual(vRopeSetRowsOps, pOps) && ggml_can_fuse_subgraph(pCgraph, pNodeIdx, pOps, {pNodeIdx + 2})) {
    const ggml_tensor *vRope = pCgraph->nodes[pNodeIdx];
    const ggml_tensor *vView = pCgraph->nodes[pNodeIdx + 1];
    const ggml_tensor *vSetRows = pCgraph->nodes[pNodeIdx + 2];

    if (fGgmlCudaShouldFuseRopeSetRows(vRope, vView, vSetRows)) {
      return true;
    }
  }

  if (!ggml_can_fuse(pCgraph, pNodeIdx, pOps)) {
    return false;
  }

  if ((pOps.size() == 2 || pOps.size() == 3) && pOps.begin()[0] == GGML_OP_RMS_NORM && pOps.begin()[1] == GGML_OP_MUL) {
    const ggml_tensor *vRmsNorm = pCgraph->nodes[pNodeIdx];
    const ggml_tensor *vMul = pCgraph->nodes[pNodeIdx + 1];
    const ggml_tensor *vAdd = nullptr;

    if (pOps.size() == 3 && pOps.begin()[2] == GGML_OP_ADD) {
      vAdd = pCgraph->nodes[pNodeIdx + 2];
    }

    GGML_ASSERT(vRmsNorm->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(vRmsNorm->type == GGML_TYPE_F32);

    // rms norm only supports F32
    if (vMul->src[0]->type != GGML_TYPE_F32 || vMul->src[1]->type != GGML_TYPE_F32 || vMul->type != GGML_TYPE_F32) {
      return false;
    }

    if (vAdd &&
        (vAdd->src[0]->type != GGML_TYPE_F32 || vAdd->src[1]->type != GGML_TYPE_F32 || vAdd->type != GGML_TYPE_F32)) {
      return false;
    }

    // if rms norm is the B operand, then we don't handle broadcast
    if (vRmsNorm == vMul->src[1] && !ggml_are_same_shape(vMul->src[0], vRmsNorm)) {
      return false;
    }

    // rms_norm kernel assumes contiguous rows
    if (!ggml_is_contiguous_rows(vMul->src[0]) || !ggml_is_contiguous_rows(vMul->src[1])) {
      return false;
    }

    if (vAdd && (!ggml_is_contiguous(vAdd->src[0]) || !ggml_is_contiguous_rows(vAdd->src[1]))) {
      return false;
    }

    return true;
  }

  if (pOps.size() == 2 && pOps.begin()[0] == GGML_OP_SSM_CONV && pOps.begin()[1] == GGML_OP_UNARY &&
      pUnaryOps.size() == 1 && pUnaryOps.begin()[0] == GGML_UNARY_OP_SILU) {
    const ggml_tensor *vSsmConv = pCgraph->nodes[pNodeIdx];
    const ggml_tensor *vSilu = pCgraph->nodes[pNodeIdx + 1];
    if (ggml_get_unary_op(vSilu) != pUnaryOps.begin()[0]) {
      return false;
    }

    if (vSsmConv->type != GGML_TYPE_F32 || vSilu->type != GGML_TYPE_F32) {
      return false;
    }

    return true;
  }

  if (pOps.size() == 3 && pOps.begin()[0] == GGML_OP_SSM_CONV && pOps.begin()[1] == GGML_OP_ADD &&
      pOps.begin()[2] == GGML_OP_UNARY && pUnaryOps.size() == 1 && pUnaryOps.begin()[0] == GGML_UNARY_OP_SILU) {
    const ggml_tensor *vSsmConv = pCgraph->nodes[pNodeIdx];
    const ggml_tensor *vAdd = pCgraph->nodes[pNodeIdx + 1];
    const ggml_tensor *vSilu = pCgraph->nodes[pNodeIdx + 2];
    if (ggml_get_unary_op(vSilu) != pUnaryOps.begin()[0]) {
      return false;
    }

    if (vSsmConv->type != GGML_TYPE_F32 || vAdd->type != GGML_TYPE_F32 || vSilu->type != GGML_TYPE_F32) {
      return false;
    }

    // ADD must consume ssm_conv's output and broadcast a 1-D channel-wise bias.
    const ggml_tensor *vBias = (vAdd->src[0] == vSsmConv) ? vAdd->src[1] : vAdd->src[0];
    if (vBias->type != GGML_TYPE_F32 || !ggml_is_contiguous(vBias)) {
      return false;
    }
    if (ggml_nelements(vBias) != vSsmConv->ne[0] || vBias->ne[0] != vSsmConv->ne[0]) {
      return false;
    }

    return true;
  }

  if (pOps.size() == 2 && pOps.begin()[0] == GGML_OP_UNARY && pOps.begin()[1] == GGML_OP_MUL && pUnaryOps.size() == 1 &&
      (pUnaryOps.begin()[0] == GGML_UNARY_OP_SILU || pUnaryOps.begin()[0] == GGML_UNARY_OP_SIGMOID ||
       pUnaryOps.begin()[0] == GGML_UNARY_OP_SOFTPLUS)) {
    const ggml_tensor *vUnary = pCgraph->nodes[pNodeIdx];
    const ggml_tensor *vMul = pCgraph->nodes[pNodeIdx + 1];

    if (ggml_get_unary_op(vUnary) != pUnaryOps.begin()[0]) {
      return false;
    }

    if (vUnary->type != GGML_TYPE_F32 && vUnary->type != GGML_TYPE_F16) {
      return false;
    }

    if (vUnary->type != vMul->type) {
      return false;
    }

    const ggml_tensor *vOther = (vMul->src[0] == vUnary) ? vMul->src[1] : vMul->src[0];
    if (vOther->type != vUnary->type) {
      return false;
    }
    if (!ggml_is_contiguous_1(vOther) || !ggml_is_contiguous_1(vUnary->src[0]) ||
        !ggml_are_same_shape(vOther, vUnary)) {
      return false;
    }

    return true;
  }

  if (pOps.size() == 2 && pOps.begin()[0] == GGML_OP_UNARY && pOps.begin()[1] == GGML_OP_SQR && pUnaryOps.size() == 1 &&
      pUnaryOps.begin()[0] == GGML_UNARY_OP_RELU) {
    const ggml_tensor *vUnary = pCgraph->nodes[pNodeIdx];
    const ggml_tensor *vSqr = pCgraph->nodes[pNodeIdx + 1];

    if (ggml_get_unary_op(vUnary) != GGML_UNARY_OP_RELU) {
      return false;
    }

    if (vUnary->type != GGML_TYPE_F32 && vUnary->type != GGML_TYPE_F16) {
      return false;
    }

    if (vUnary->type != vSqr->type) {
      return false;
    }

    if (!ggml_is_contiguous(vUnary->src[0])) {
      return false;
    }

    return true;
  }

  if (pOps.size() == 3 && pOps.begin()[0] == GGML_OP_SCALE && pOps.begin()[1] == GGML_OP_UNARY &&
      pOps.begin()[2] == GGML_OP_SCALE && pUnaryOps.size() == 1 && pUnaryOps.begin()[0] == GGML_UNARY_OP_TANH) {
    const ggml_tensor *vScale = pCgraph->nodes[pNodeIdx];
    const ggml_tensor *vTanh = pCgraph->nodes[pNodeIdx + 1];
    const ggml_tensor *vScale2 = pCgraph->nodes[pNodeIdx + 2];

    GGML_ASSERT(vScale->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(vScale->type == GGML_TYPE_F32);

    if (ggml_get_unary_op(vTanh) != GGML_UNARY_OP_TANH) {
      return false;
    }

    // Check for bias
    if (ggml_get_op_params_f32(vScale, 1) != 0.0f || ggml_get_op_params_f32(vScale2, 1) != 0.0f) {
      return false;
    }

    return true;
  }

  return false;
}

// try and fuse nodes and return the number of nodes to skip
static int fGgmlCudaTryFuse(ggml_backend_cuda_context *pCudaCtx, ggml_cgraph *pCgraph, int pI) {

  static bool vDisableFusion =
      getenv("GGML_CUDA_DISABLE_FUSION") != nullptr && std::atoi(getenv("GGML_CUDA_DISABLE_FUSION"));
  if (vDisableFusion) {
    return 0;
  }

  ggml_tensor *vNode = pCgraph->nodes[pI];

  // gated_delta_net -> cpy: scatter recurrent-state snapshots into the cache
  if (vNode->op == GGML_OP_GATED_DELTA_NET) {
    ggml_cuda_gated_delta_net_fused_cache vFusedStateCpy;
    const int cNodesToSkip = fGgmlCudaTryGdnCacheFusion(pCgraph, pI, vFusedStateCpy);
    if (cNodesToSkip > 0) {
#ifdef GGML_CUDA_DEBUG
      GGML_LOG_INFO("%s: fused gated_delta_net snapshot copies for %s (skipped %d nodes)\n", __func__, vNode->name,
                    cNodesToSkip);
#endif
      ggml_cuda_op_gated_delta_net_fused_cache(*pCudaCtx, vNode, vFusedStateCpy);
      return cNodesToSkip;
    }
  }

  // topk-moe
  if (pCgraph->nodes[pI]->op == GGML_OP_UNARY || pCgraph->nodes[pI]->op == GGML_OP_SOFT_MAX ||
      pCgraph->nodes[pI]->op == GGML_OP_ARGSORT) {
    ggml_cuda_topk_moe_args vArgs;
    const bool cCanFuse = fGgmlCudaTopkMoeFusion(pCgraph, pI, vArgs);
    std::vector<ggml_op> lOps;

    if (cCanFuse) {
      const ggml_tensor *vLogits = vNode->src[0];
      ggml_tensor *vWeights = nullptr;
      ggml_tensor *vIds = nullptr;
      const ggml_tensor *vBias = nullptr;
      const ggml_tensor *vClamp = nullptr;
      const ggml_tensor *vScale = nullptr;

      if (!vArgs.delayed_softmax) {
        int aOutNodes[2]; // nodes which can't be elided

        if (vArgs.sigmoid) {
          lOps.insert(lOps.end(), {GGML_OP_UNARY});
        } else if (vArgs.sqrt_softplus) {
          lOps.insert(lOps.end(), {GGML_OP_UNARY, GGML_OP_SQRT});
        } else {
          lOps.insert(lOps.end(), {GGML_OP_SOFT_MAX});
        }
        const int cIProbs = pI + (int)lOps.size() - 1; // last node of the gating activation

        if (vArgs.prob_bias) {
          vBias = pCgraph->nodes[cIProbs + 2]->src[1];
          lOps.insert(lOps.end(), {GGML_OP_RESHAPE, GGML_OP_ADD, GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS});
          aOutNodes[0] = cIProbs + 4;
        } else {
          lOps.insert(lOps.end(), {GGML_OP_RESHAPE, GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS});
          aOutNodes[0] = cIProbs + 3;
        }
        vIds = pCgraph->nodes[aOutNodes[0]];

        if (vArgs.norm) {
          lOps.insert(lOps.end(), {GGML_OP_RESHAPE, GGML_OP_SUM_ROWS, GGML_OP_CLAMP, GGML_OP_DIV, GGML_OP_RESHAPE});
          vClamp = pCgraph->nodes[pI + lOps.size() - 3];
        }
        if (vArgs.scale) {
          lOps.insert(lOps.end(), {GGML_OP_SCALE});
          vScale = pCgraph->nodes[pI + lOps.size() - 1];
        }

        vWeights = pCgraph->nodes[pI + lOps.size() - 1];
        aOutNodes[1] = pI + lOps.size() - 1;

        if (ggml_can_fuse_subgraph(pCgraph, pI, lOps.size(), lOps.data(), aOutNodes, 2) &&
            ggml_cuda_should_use_topk_moe(vNode, vLogits, vWeights, vIds) &&
            fGgmlCudaCheckFusionMemoryRanges(pCgraph, pI, lOps.size(), aOutNodes, 2, /*is_topk_moe=*/true)) {
          ggml_cuda_op_topk_moe(*pCudaCtx, vLogits, vWeights, vIds, vClamp, vScale, vBias, vArgs);
          return lOps.size() - 1;
        }
      } else if (!vArgs.norm && !vArgs.prob_bias) {
        // special case gpt-oss, no norm, no bias.
        lOps.insert(lOps.end(), {GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS, GGML_OP_RESHAPE, GGML_OP_SOFT_MAX,
                                 GGML_OP_RESHAPE});
        vWeights = pCgraph->nodes[pI + 5];
        vIds = pCgraph->nodes[pI + 1];
        const ggml_tensor *vSoftmax = pCgraph->nodes[pI + 4];

        int aOutNodes[2] = {pI + 1, pI + 5};
        if (ggml_can_fuse_subgraph(pCgraph, pI, lOps.size(), lOps.data(), aOutNodes, 2) &&
            ggml_cuda_should_use_topk_moe(vSoftmax, vLogits, vWeights, vIds) &&
            fGgmlCudaCheckFusionMemoryRanges(pCgraph, pI, lOps.size(), aOutNodes, 2, /*is_topk_moe=*/true)) {
          ggml_cuda_op_topk_moe(*pCudaCtx, vLogits, vWeights, vIds, vClamp, vScale, vBias, vArgs);
          return lOps.size() - 1;
        }
      }
    }
  }

  // RoPE + view + set-rows
  if (fGgmlCudaCanFuse(pCgraph, pI, {GGML_OP_ROPE, GGML_OP_VIEW, GGML_OP_SET_ROWS}, {})) {
    ggml_tensor *vRope = pCgraph->nodes[pI];
    ggml_tensor *vSetRows = pCgraph->nodes[pI + 2];

    ggml_cuda_op_rope_fused(*pCudaCtx, vRope, vSetRows);
    return 2;
  }

  // Snake activation: y = x + sin(a*x)^2 * inv_b
  // Naive 5-op decomposition emitted by frontends: mul -> sin -> sqr -> mul -> add
  if (ggml_can_fuse_subgraph(pCgraph, pI, {GGML_OP_MUL, GGML_OP_SIN, GGML_OP_SQR, GGML_OP_MUL, GGML_OP_ADD},
                             {pI + 4})) {
    const ggml_tensor *vMul0 = pCgraph->nodes[pI];
    const ggml_tensor *vSqr = pCgraph->nodes[pI + 2];
    const ggml_tensor *vMul1 = pCgraph->nodes[pI + 3];
    ggml_tensor *vAdd = pCgraph->nodes[pI + 4];

    // x carries the full activation shape, a is the broadcast operand
    const ggml_tensor *vX = ggml_are_same_shape(vMul0, vMul0->src[0]) ? vMul0->src[0] : vMul0->src[1];
    const ggml_tensor *vA = (vX == vMul0->src[0]) ? vMul0->src[1] : vMul0->src[0];

    // mul1 reads sqr and inv_b in either operand order
    const ggml_tensor *vInvB = (vMul1->src[0] == vSqr) ? vMul1->src[1] : vMul1->src[0];

    // closure check: the trailing add must read the same x as the leading mul
    const ggml_tensor *vXInAdd = (vAdd->src[0] == vMul1) ? vAdd->src[1] : vAdd->src[0];

    // Kernel iterates over total = T * C, so x and add must be 2D and
    // a / inv_b must collapse to [1, C, 1, 1]. Higher dims are not handled.
    const bool cDimOk = (vX->ne[2] == 1 && vX->ne[3] == 1) && (vAdd->ne[2] == 1 && vAdd->ne[3] == 1) &&
                        (vA->ne[2] == 1 && vA->ne[3] == 1);
    const bool cShapeOk = ggml_are_same_shape(vA, vInvB) && vA->ne[0] == 1 && vA->ne[1] == vX->ne[1];

    // x is in the supported whitelist and every chain intermediate shares
    // x's type. launch_snake reads a and inv_b as const float *, so they
    // stay F32.
    const ggml_tensor *vSin1 = pCgraph->nodes[pI + 1];
    const bool cTypesOk = (vX->type == GGML_TYPE_F32 || vX->type == GGML_TYPE_F16 || vX->type == GGML_TYPE_BF16) &&
                          (vA->type == GGML_TYPE_F32) && (vInvB->type == GGML_TYPE_F32) && (vMul0->type == vX->type) &&
                          (vSin1->type == vX->type) && (vSqr->type == vX->type) && (vMul1->type == vX->type) &&
                          (vAdd->type == vX->type);

    // kernel reads x[idx] and a[c] / inv_b[c] linearly, so every operand is contiguous
    const bool cContigOk =
        ggml_is_contiguous(vX) && ggml_is_contiguous(vAdd) && ggml_is_contiguous(vA) && ggml_is_contiguous(vInvB);

    if (cTypesOk && cShapeOk && cDimOk && cContigOk && vXInAdd == vX) {
      ggml_cuda_op_snake_fused(*pCudaCtx, vX, vA, vInvB, vAdd);
      return 4;
    }
  }

  // multi-(add or mul)
  if (vNode->op == GGML_OP_ADD || vNode->op == GGML_OP_MUL) {
    int vNFuse = 0;
    ggml_op aOps[8];
    std::fill(aOps, aOps + 8, vNode->op);

    for (; vNFuse <= 6; ++vNFuse) {
      if (!ggml_can_fuse(pCgraph, pI + vNFuse, aOps + vNFuse, 2)) {
        break;
      }
      if (pCgraph->nodes[pI + vNFuse] != pCgraph->nodes[pI + vNFuse + 1]->src[0]) {
        break;
      }
      if (!ggml_are_same_layout(pCgraph->nodes[pI + vNFuse]->src[1], pCgraph->nodes[pI + vNFuse + 1]->src[1])) {
        break;
      }
    }

    vNFuse++;

    if (vNFuse > 1) {
      ggml_tensor vFusedNode;
      memcpy(&vFusedNode, vNode, sizeof(ggml_tensor));
      for (int vJ = 0; vJ < vNFuse - 1; ++vJ) {
        vFusedNode.src[vJ + 2] = pCgraph->nodes[pI + vJ + 1]->src[1];
      }
      vFusedNode.data = pCgraph->nodes[pI + vNFuse - 1]->data;
      if (vNode->op == GGML_OP_ADD) {
        ggml_cuda_op_fused_add(*pCudaCtx, &vFusedNode, vNFuse);
      } else {
        ggml_cuda_op_fused_mul(*pCudaCtx, &vFusedNode, vNFuse);
      }
      return vNFuse - 1;
    }
  }

  bool vFusedMulMatVec = false;
  int vFusedNodeCount = 0;

  auto fGetMulMatScale = [](const ggml_tensor *pScaleNode, const ggml_tensor *pMmNode) -> const ggml_tensor * {
    const bool cScaleLhsMm = pScaleNode->src[0] == pMmNode;
    const bool cScaleRhsMm = pScaleNode->src[1] == pMmNode;
    if (!cScaleLhsMm && !cScaleRhsMm) {
      return nullptr;
    }

    const ggml_tensor *vScale = cScaleLhsMm ? pScaleNode->src[1] : pScaleNode->src[0];
    if (pMmNode->src[0]->type != GGML_TYPE_NVFP4 || pScaleNode->type != GGML_TYPE_F32 ||
        vScale->type != GGML_TYPE_F32 || !ggml_is_contiguous(vScale) || ggml_nelements(vScale) != 1 ||
        !ggml_are_same_shape(pScaleNode, pMmNode)) {
      return nullptr;
    }

    return vScale;
  };

  auto fGetMulMatIdScale = [](const ggml_tensor *pReshape, const ggml_tensor *pRepeat, const ggml_tensor *pGetrows,
                              const ggml_tensor *pScaleNode, const ggml_tensor *pMmNode) -> const ggml_tensor * {
    if (pRepeat->src[0] != pReshape || pGetrows->src[0] != pRepeat || pGetrows->src[1] != pMmNode->src[2]) {
      return nullptr;
    }
    if (!((pScaleNode->src[0] == pMmNode && pScaleNode->src[1] == pGetrows) ||
          (pScaleNode->src[0] == pGetrows && pScaleNode->src[1] == pMmNode))) {
      return nullptr;
    }

    const ggml_tensor *vScale = pReshape->src[0];
    if (pMmNode->src[0]->type != GGML_TYPE_NVFP4 || pScaleNode->type != GGML_TYPE_F32 ||
        vScale->type != GGML_TYPE_F32 || !ggml_is_contiguous(vScale) ||
        ggml_nelements(vScale) != pMmNode->src[0]->ne[2] || !ggml_are_same_shape(pScaleNode, pMmNode)) {
      return nullptr;
    }

    return vScale;
  };

  auto fGetBiasTensor = [](const ggml_tensor *pBiasNode, const ggml_tensor *pMulNode,
                           ggml_op pOpBias) -> const ggml_tensor * {
    if (pOpBias == GGML_OP_ADD) {
      if (pBiasNode->src[0] == pMulNode) {
        return pBiasNode->src[1];
      }
      if (pBiasNode->src[1] == pMulNode) {
        return pBiasNode->src[0];
      }
      return nullptr;
    }
    GGML_ASSERT(pOpBias == GGML_OP_ADD_ID);
    GGML_ASSERT(pBiasNode->src[0] == pMulNode);
    return pBiasNode->src[1];
  };

  // gate + glu + up, with optional scale/bias on both lanes.
  for (ggml_op vOp : {GGML_OP_MUL_MAT, GGML_OP_MUL_MAT_ID}) {
    const ggml_op cBiasOp = vOp == GGML_OP_MUL_MAT ? GGML_OP_ADD : GGML_OP_ADD_ID;

    if (vOp == GGML_OP_MUL_MAT) {
      for (const bool cWithBias : {false, true}) {
        const int cGateIdx = pI;
        const int cGateScaleIdx = pI + 1;
        const int cGateBiasIdx = cWithBias ? pI + 2 : -1;
        const int cUpIdx = cWithBias ? pI + 3 : pI + 2;
        const int cUpScaleIdx = cUpIdx + 1;
        const int cUpBiasIdx = cWithBias ? cUpIdx + 2 : -1;
        const int cGluIdx = cWithBias ? cUpIdx + 3 : cUpIdx + 2;

        const int aOutNodes[] = {cGluIdx};
        ggml_op aOps[7];
        if (cWithBias) {
          aOps[0] = vOp;
          aOps[1] = GGML_OP_MUL;
          aOps[2] = cBiasOp;
          aOps[3] = vOp;
          aOps[4] = GGML_OP_MUL;
          aOps[5] = cBiasOp;
          aOps[6] = GGML_OP_GLU;
        } else {
          aOps[0] = vOp;
          aOps[1] = GGML_OP_MUL;
          aOps[2] = vOp;
          aOps[3] = GGML_OP_MUL;
          aOps[4] = GGML_OP_GLU;
        }
        const int cNOps = cWithBias ? 7 : 5;

        if (!ggml_can_fuse_subgraph(pCgraph, pI, cNOps, aOps, aOutNodes, 1) ||
            !fGgmlCudaCheckFusionMemoryRanges(pCgraph, pI, cNOps, aOutNodes, 1)) {
          continue;
        }

        ggml_tensor *vGateN = pCgraph->nodes[cGateIdx];
        ggml_tensor *vGateScaleN = pCgraph->nodes[cGateScaleIdx];
        ggml_tensor *vGateOutN = cWithBias ? pCgraph->nodes[cGateBiasIdx] : vGateScaleN;
        ggml_tensor *vUpN = pCgraph->nodes[cUpIdx];
        ggml_tensor *vUpScaleN = pCgraph->nodes[cUpScaleIdx];
        ggml_tensor *vUpOutN = cWithBias ? pCgraph->nodes[cUpBiasIdx] : vUpScaleN;
        const ggml_tensor *vGlu = pCgraph->nodes[cGluIdx];

        if (!fGgmlCudaShouldFuseMulMat(vUpN, vGateN, vGlu, cWithBias ? vUpOutN : nullptr,
                                       cWithBias ? vGateOutN : nullptr, vUpScaleN, vGateScaleN)) {
          continue;
        }

        const ggml_tensor *vGateScale = fGetMulMatScale(vGateScaleN, vGateN);
        const ggml_tensor *vUpScale = fGetMulMatScale(vUpScaleN, vUpN);
        if (!vGateScale || !vUpScale) {
          continue;
        }

        const ggml_tensor *vUpBias = cWithBias ? fGetBiasTensor(vUpOutN, vUpScaleN, cBiasOp) : nullptr;
        const ggml_tensor *vGateBias = cWithBias ? fGetBiasTensor(vGateOutN, vGateScaleN, cBiasOp) : nullptr;
        if (cWithBias && (!ggml_are_same_shape(vGateOutN->src[0], vGateOutN->src[1]) ||
                          !ggml_are_same_shape(vUpOutN->src[0], vUpOutN->src[1]))) {
          continue;
        }

        const ggml_tensor *vSrc0 = vUpN->src[0];
        const ggml_tensor *vSrc1 = vUpN->src[1];
        const ggml_tensor *vIds = vUpN->src[2];

        ggml_cuda_mm_fusion_args_host vFusionData{};
        vFusionData.gate = vGateN->src[0];
        vFusionData.x_bias = vUpBias;
        vFusionData.gate_bias = vGateBias;
        vFusionData.x_scale = vUpScale;
        vFusionData.gate_scale = vGateScale;
        vFusionData.glu_op = ggml_get_glu_op(vGlu);
        vFusionData.glu_limit = ggml_get_op_params_f32(vGlu, 3);

        if (fGgmlCudaShouldFuseMulMatVecQ(vUpN)) {
          ggml_cuda_mul_mat_vec_q(*pCudaCtx, vSrc0, vSrc1, vIds, pCgraph->nodes[cGluIdx], &vFusionData);
          vFusedMulMatVec = true;
          vFusedNodeCount = cNOps;
          break;
        }
      }

      if (vFusedMulMatVec) {
        break;
      }
    } else {
      for (const bool cWithBias : {false, true}) {
        const int cGateIdx = pI;
        const int cGateScaleIdx = pI + 4;
        const int cGateBiasIdx = cWithBias ? pI + 5 : -1;
        const int cUpIdx = cWithBias ? pI + 6 : pI + 5;
        const int cUpScaleIdx = cUpIdx + 4;
        const int cUpBiasIdx = cWithBias ? cUpIdx + 5 : -1;
        const int cGluIdx = cWithBias ? cUpIdx + 6 : cUpIdx + 5;

        const int aOutNodes[] = {cGluIdx};
        ggml_op aOps[13];
        if (cWithBias) {
          aOps[0] = vOp;
          aOps[1] = GGML_OP_RESHAPE;
          aOps[2] = GGML_OP_REPEAT;
          aOps[3] = GGML_OP_GET_ROWS;
          aOps[4] = GGML_OP_MUL;
          aOps[5] = cBiasOp;
          aOps[6] = vOp;
          aOps[7] = GGML_OP_RESHAPE;
          aOps[8] = GGML_OP_REPEAT;
          aOps[9] = GGML_OP_GET_ROWS;
          aOps[10] = GGML_OP_MUL;
          aOps[11] = cBiasOp;
          aOps[12] = GGML_OP_GLU;
        } else {
          aOps[0] = vOp;
          aOps[1] = GGML_OP_RESHAPE;
          aOps[2] = GGML_OP_REPEAT;
          aOps[3] = GGML_OP_GET_ROWS;
          aOps[4] = GGML_OP_MUL;
          aOps[5] = vOp;
          aOps[6] = GGML_OP_RESHAPE;
          aOps[7] = GGML_OP_REPEAT;
          aOps[8] = GGML_OP_GET_ROWS;
          aOps[9] = GGML_OP_MUL;
          aOps[10] = GGML_OP_GLU;
        }
        const int cNOps = cWithBias ? 13 : 11;

        if (!ggml_can_fuse_subgraph(pCgraph, pI, cNOps, aOps, aOutNodes, 1) ||
            !fGgmlCudaCheckFusionMemoryRanges(pCgraph, pI, cNOps, aOutNodes, 1)) {
          continue;
        }

        ggml_tensor *vGateN = pCgraph->nodes[cGateIdx];
        ggml_tensor *vGateScaleN = pCgraph->nodes[cGateScaleIdx];
        ggml_tensor *vGateOutN = cWithBias ? pCgraph->nodes[cGateBiasIdx] : vGateScaleN;
        ggml_tensor *vUpN = pCgraph->nodes[cUpIdx];
        ggml_tensor *vUpScaleN = pCgraph->nodes[cUpScaleIdx];
        ggml_tensor *vUpOutN = cWithBias ? pCgraph->nodes[cUpBiasIdx] : vUpScaleN;
        const ggml_tensor *vGlu = pCgraph->nodes[cGluIdx];

        if (!fGgmlCudaShouldFuseMulMat(vUpN, vGateN, vGlu, cWithBias ? vUpOutN : nullptr,
                                       cWithBias ? vGateOutN : nullptr, vUpScaleN, vGateScaleN)) {
          continue;
        }

        const ggml_tensor *vGateScale = fGetMulMatIdScale(pCgraph->nodes[cGateIdx + 1], pCgraph->nodes[cGateIdx + 2],
                                                          pCgraph->nodes[cGateIdx + 3], vGateScaleN, vGateN);
        const ggml_tensor *vUpScale = fGetMulMatIdScale(pCgraph->nodes[cUpIdx + 1], pCgraph->nodes[cUpIdx + 2],
                                                        pCgraph->nodes[cUpIdx + 3], vUpScaleN, vUpN);
        if (!vGateScale || !vUpScale) {
          continue;
        }

        const ggml_tensor *vUpBias = cWithBias ? fGetBiasTensor(vUpOutN, vUpScaleN, cBiasOp) : nullptr;
        const ggml_tensor *vGateBias = cWithBias ? fGetBiasTensor(vGateOutN, vGateScaleN, cBiasOp) : nullptr;

        const ggml_tensor *vSrc0 = vUpN->src[0];
        const ggml_tensor *vSrc1 = vUpN->src[1];
        const ggml_tensor *vIds = vUpN->src[2];

        ggml_cuda_mm_fusion_args_host vFusionData{};
        vFusionData.gate = vGateN->src[0];
        vFusionData.x_bias = vUpBias;
        vFusionData.gate_bias = vGateBias;
        vFusionData.x_scale = vUpScale;
        vFusionData.gate_scale = vGateScale;
        vFusionData.glu_op = ggml_get_glu_op(vGlu);
        vFusionData.glu_limit = ggml_get_op_params_f32(vGlu, 3);

        if (fGgmlCudaShouldFuseMulMatVecQ(vUpN)) {
          ggml_cuda_mul_mat_vec_q(*pCudaCtx, vSrc0, vSrc1, vIds, pCgraph->nodes[cGluIdx], &vFusionData);
          vFusedMulMatVec = true;
          vFusedNodeCount = cNOps;
          break;
        }
      }

      if (vFusedMulMatVec) {
        break;
      }
    }

    if (fGgmlCudaCanFuse(pCgraph, pI, {vOp, cBiasOp, vOp, cBiasOp, GGML_OP_GLU}, {})) {
      ggml_tensor *vGlu = pCgraph->nodes[pI + 4];
      ggml_tensor *vGateBiasN = vGlu->src[0];
      ggml_tensor *vUpBiasN = vGlu->src[1];

      // we don't assume the order for {gate, up}. Instead infer it from the bias tensor
      ggml_tensor *vGateN = nullptr;
      ggml_tensor *vUpN = nullptr;

      if (vGateBiasN->src[0] == pCgraph->nodes[pI] || vGateBiasN->src[1] == pCgraph->nodes[pI]) {
        vGateN = pCgraph->nodes[pI];
        vUpN = pCgraph->nodes[pI + 2];
      } else if (vGateBiasN->src[0] == pCgraph->nodes[pI + 2] || vGateBiasN->src[1] == pCgraph->nodes[pI + 2]) {
        vGateN = pCgraph->nodes[pI + 2];
        vUpN = pCgraph->nodes[pI];
      } else {
        continue;
      }

      const ggml_tensor *vUpBiasTensor = fGetBiasTensor(vUpBiasN, vUpN, cBiasOp);
      const ggml_tensor *vGateBiasTensor = fGetBiasTensor(vGateBiasN, vGateN, cBiasOp);

      if (!vUpBiasTensor || !vGateBiasTensor) {
        continue;
      }

      // we don't support repeating adds
      if (cBiasOp == GGML_OP_ADD && (!ggml_are_same_shape(vGateBiasN->src[0], vGateBiasN->src[1]) ||
                                     !ggml_are_same_shape(vUpBiasN->src[0], vUpBiasN->src[1]))) {
        continue;
      }

      const ggml_tensor *vSrc0 = vUpN->src[0];
      const ggml_tensor *vSrc1 = vUpN->src[1];
      const ggml_tensor *vIds = vUpN->src[2];

      if (fGgmlCudaShouldFuseMulMatVecF(vUpN)) {
        ggml_cuda_mm_fusion_args_host vFusionData{};
        vFusionData.gate = vGateN->src[0];
        vFusionData.x_bias = vUpBiasTensor;
        vFusionData.gate_bias = vGateBiasTensor;
        vFusionData.glu_op = ggml_get_glu_op(vGlu);
        vFusionData.glu_limit = ggml_get_op_params_f32(vGlu, 3);

        ggml_cuda_mul_mat_vec_f(*pCudaCtx, vSrc0, vSrc1, vIds, vGlu, &vFusionData);
        vFusedMulMatVec = true;
        vFusedNodeCount = 5;
        break;
      }

      if (fGgmlCudaShouldFuseMulMatVecQ(vUpN)) {
        ggml_cuda_mm_fusion_args_host vFusionData{};
        vFusionData.gate = vGateN->src[0];
        vFusionData.x_bias = vUpBiasTensor;
        vFusionData.gate_bias = vGateBiasTensor;
        vFusionData.glu_op = ggml_get_glu_op(vGlu);
        vFusionData.glu_limit = ggml_get_op_params_f32(vGlu, 3);

        ggml_cuda_mul_mat_vec_q(*pCudaCtx, vSrc0, vSrc1, vIds, vGlu, &vFusionData);
        vFusedMulMatVec = true;
        vFusedNodeCount = 5;
        break;
      }
    } else if (fGgmlCudaCanFuse(pCgraph, pI, {vOp, vOp, GGML_OP_GLU}, {})) {
      ggml_tensor *vGlu = pCgraph->nodes[pI + 2];
      ggml_tensor *vGate = vGlu->src[0];
      ggml_tensor *vUp = vGlu->src[1];

      bool vOk = (vGate == pCgraph->nodes[pI] && vUp == pCgraph->nodes[pI + 1]) ||
                 (vGate == pCgraph->nodes[pI + 1] && vUp == pCgraph->nodes[pI]);

      if (!vOk) {
        continue;
      }

      const ggml_tensor *vSrc0 = vUp->src[0];
      const ggml_tensor *vSrc1 = vUp->src[1];
      const ggml_tensor *vIds = vUp->src[2];

      if (fGgmlCudaShouldFuseMulMatVecF(vUp)) {
        ggml_cuda_mm_fusion_args_host vFusionData{};
        vFusionData.gate = vGate->src[0];
        vFusionData.glu_op = ggml_get_glu_op(vGlu);
        vFusionData.glu_limit = ggml_get_op_params_f32(vGlu, 3);

        ggml_cuda_mul_mat_vec_f(*pCudaCtx, vSrc0, vSrc1, vIds, vGlu, &vFusionData);
        vFusedMulMatVec = true;
        vFusedNodeCount = 3;
        break;
      }

      if (fGgmlCudaShouldFuseMulMatVecQ(vUp)) {
        ggml_cuda_mm_fusion_args_host vFusionData{};
        vFusionData.gate = vGate->src[0];
        vFusionData.glu_op = ggml_get_glu_op(vGlu);
        vFusionData.glu_limit = ggml_get_op_params_f32(vGlu, 3);

        ggml_cuda_mul_mat_vec_q(*pCudaCtx, vSrc0, vSrc1, vIds, vGlu, &vFusionData);
        vFusedMulMatVec = true;
        vFusedNodeCount = 3;
        break;
      }
    }
  }

  if (vFusedMulMatVec) {
    return vFusedNodeCount - 1;
  }

  vFusedMulMatVec = false;
  vFusedNodeCount = 0;

  // mul_mat + scale + optional bias
  for (ggml_op vOp : {GGML_OP_MUL_MAT, GGML_OP_MUL_MAT_ID}) {
    const ggml_op cBiasOp = vOp == GGML_OP_MUL_MAT ? GGML_OP_ADD : GGML_OP_ADD_ID;

    for (const bool cWithBias : {false, true}) {
      const int cNOps = vOp == GGML_OP_MUL_MAT ? (cWithBias ? 3 : 2) : (cWithBias ? 6 : 5);
      const int aOutNodes[] = {pI + cNOps - 1};
      ggml_op aOps[6];
      if (vOp == GGML_OP_MUL_MAT) {
        if (cWithBias) {
          aOps[0] = vOp;
          aOps[1] = GGML_OP_MUL;
          aOps[2] = cBiasOp;
        } else {
          aOps[0] = vOp;
          aOps[1] = GGML_OP_MUL;
        }
      } else {
        if (cWithBias) {
          aOps[0] = vOp;
          aOps[1] = GGML_OP_RESHAPE;
          aOps[2] = GGML_OP_REPEAT;
          aOps[3] = GGML_OP_GET_ROWS;
          aOps[4] = GGML_OP_MUL;
          aOps[5] = cBiasOp;
        } else {
          aOps[0] = vOp;
          aOps[1] = GGML_OP_RESHAPE;
          aOps[2] = GGML_OP_REPEAT;
          aOps[3] = GGML_OP_GET_ROWS;
          aOps[4] = GGML_OP_MUL;
        }
      }

      if (!ggml_can_fuse_subgraph(pCgraph, pI, cNOps, aOps, aOutNodes, 1) ||
          !fGgmlCudaCheckFusionMemoryRanges(pCgraph, pI, cNOps, aOutNodes, 1)) {
        continue;
      }

      ggml_tensor *vMmNode = pCgraph->nodes[pI];
      ggml_tensor *vScaleNode = vOp == GGML_OP_MUL_MAT ? pCgraph->nodes[pI + 1] : pCgraph->nodes[pI + 4];
      ggml_tensor *vOutNode = cWithBias ? pCgraph->nodes[pI + cNOps - 1] : vScaleNode;

      const ggml_tensor *vScale = nullptr;
      if (vOp == GGML_OP_MUL_MAT) {
        vScale = fGetMulMatScale(vScaleNode, vMmNode);
      } else {
        vScale = fGetMulMatIdScale(pCgraph->nodes[pI + 1], pCgraph->nodes[pI + 2], pCgraph->nodes[pI + 3], vScaleNode,
                                   vMmNode);
      }
      if (!vScale) {
        continue;
      }

      const ggml_tensor *vBias = cWithBias ? fGetBiasTensor(vOutNode, vScaleNode, cBiasOp) : nullptr;
      if (cWithBias && !vBias) {
        continue;
      }
      if (cWithBias && cBiasOp == GGML_OP_ADD && !ggml_are_same_shape(vOutNode->src[0], vOutNode->src[1])) {
        continue;
      }
      if (cWithBias && cBiasOp == GGML_OP_ADD_ID && vOutNode->src[2] != vMmNode->src[2]) {
        continue;
      }

      const ggml_tensor *vSrc0 = vMmNode->src[0];
      const ggml_tensor *vSrc1 = vMmNode->src[1];
      const ggml_tensor *vIds = vMmNode->src[2];

      ggml_cuda_mm_fusion_args_host vFusionData{};
      vFusionData.x_bias = vBias;
      vFusionData.x_scale = vScale;

      if (fGgmlCudaShouldFuseMulMatVecQ(vMmNode)) {
        ggml_cuda_mul_mat_vec_q(*pCudaCtx, vSrc0, vSrc1, vIds, vOutNode, &vFusionData);
        vFusedMulMatVec = true;
        vFusedNodeCount = cNOps;
        break;
      }
    }
    if (vFusedMulMatVec) {
      break;
    }
  }

  if (vFusedMulMatVec) {
    return vFusedNodeCount - 1;
  }

  // mul_mat + add
  for (ggml_op vOp : {GGML_OP_MUL_MAT, GGML_OP_MUL_MAT_ID}) {
    const ggml_op cBiasOp = vOp == GGML_OP_MUL_MAT ? GGML_OP_ADD : GGML_OP_ADD_ID;

    if (!ggml_can_fuse(pCgraph, pI, {vOp, cBiasOp})) {
      continue;
    }

    ggml_tensor *vMmNode = pCgraph->nodes[pI];
    ggml_tensor *vBiasNode = pCgraph->nodes[pI + 1];

    ggml_tensor *vBiasTensor = nullptr;
    if (cBiasOp == GGML_OP_ADD) {
      if (vBiasNode->src[0] == vMmNode) {
        vBiasTensor = vBiasNode->src[1];
      } else if (vBiasNode->src[1] == vMmNode) {
        vBiasTensor = vBiasNode->src[0];
      } else {
        continue;
      }
    } else {
      if (vBiasNode->src[0] != vMmNode) {
        continue;
      }
      vBiasTensor = vBiasNode->src[1];
    }

    const ggml_tensor *vSrc0 = vMmNode->src[0];
    const ggml_tensor *vSrc1 = vMmNode->src[1];
    const ggml_tensor *vIds = vMmNode->src[2];

    if (cBiasOp == GGML_OP_ADD_ID && vBiasNode->src[2] != vIds) {
      continue;
    }

    if (cBiasOp == GGML_OP_ADD && !ggml_are_same_shape(vBiasNode->src[0], vBiasNode->src[1])) {
      continue;
    }

    ggml_cuda_mm_fusion_args_host vFusionData{};
    vFusionData.x_bias = vBiasTensor;

    if (fGgmlCudaShouldFuseMulMatVecF(vMmNode)) {
      ggml_cuda_mul_mat_vec_f(*pCudaCtx, vSrc0, vSrc1, vIds, vBiasNode, &vFusionData);
      vFusedMulMatVec = true;
      vFusedNodeCount = 2;
      break;
    }

    if (fGgmlCudaShouldFuseMulMatVecQ(vMmNode)) {
      ggml_cuda_mul_mat_vec_q(*pCudaCtx, vSrc0, vSrc1, vIds, vBiasNode, &vFusionData);
      vFusedMulMatVec = true;
      vFusedNodeCount = 2;
      break;
    }
  }

  if (vFusedMulMatVec) {
    return vFusedNodeCount - 1;
  }

  if (fGgmlCudaCanFuse(pCgraph, pI, {GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ADD}, {})) {
    ggml_cuda_op_rms_norm_fused_add(*pCudaCtx, vNode, pCgraph->nodes[pI + 1], pCgraph->nodes[pI + 2]);
    return 2;
  }

  if (fGgmlCudaCanFuse(pCgraph, pI, {GGML_OP_RMS_NORM, GGML_OP_MUL}, {})) {
    ggml_cuda_op_rms_norm_fused(*pCudaCtx, vNode, pCgraph->nodes[pI + 1]);
    return 1;
  }

  if (fGgmlCudaCanFuse(pCgraph, pI, {GGML_OP_SSM_CONV, GGML_OP_ADD, GGML_OP_UNARY}, {GGML_UNARY_OP_SILU})) {
    ggml_cuda_op_ssm_conv(*pCudaCtx, vNode, pCgraph->nodes[pI + 1], pCgraph->nodes[pI + 2]);
    return 2;
  }

  if (fGgmlCudaCanFuse(pCgraph, pI, {GGML_OP_SSM_CONV, GGML_OP_UNARY}, {GGML_UNARY_OP_SILU})) {
    ggml_cuda_op_ssm_conv(*pCudaCtx, vNode, /*bias_add_node=*/nullptr, pCgraph->nodes[pI + 1]);
    return 1;
  }

  if (fGgmlCudaCanFuse(pCgraph, pI, {GGML_OP_UNARY, GGML_OP_MUL}, {GGML_UNARY_OP_SILU}) ||
      fGgmlCudaCanFuse(pCgraph, pI, {GGML_OP_UNARY, GGML_OP_MUL}, {GGML_UNARY_OP_SIGMOID}) ||
      fGgmlCudaCanFuse(pCgraph, pI, {GGML_OP_UNARY, GGML_OP_MUL}, {GGML_UNARY_OP_SOFTPLUS})) {
    ggml_cuda_op_unary_mul(*pCudaCtx, vNode, pCgraph->nodes[pI + 1]);
    return 1;
  }

  if (fGgmlCudaCanFuse(pCgraph, pI, {GGML_OP_UNARY, GGML_OP_SQR}, {GGML_UNARY_OP_RELU})) {
    ggml_cuda_op_relu_sqr(*pCudaCtx, vNode, pCgraph->nodes[pI + 1]);
    return 1;
  }

  if (fGgmlCudaCanFuse(pCgraph, pI, {GGML_OP_SCALE, GGML_OP_UNARY, GGML_OP_SCALE}, {GGML_UNARY_OP_TANH})) {
    ggml_cuda_op_softcap(*pCudaCtx, pCgraph->nodes[pI + 2], vNode);
    return 2;
  }

  return 0;
}

static void fGgmlCudaGraphEvaluateAndCapture(ggml_backend_cuda_context *pCudaCtx, ggml_cgraph *pCgraph,
                                             const bool pUseCudaGraph, const bool pCudaGraphUpdateRequired,
                                             const void *pGraphKey) {
  bool vGraphEvaluatedOrCaptured = false;

  // flag used to determine whether it is an integrated_gpu
  const bool cIntegrated = ggml_cuda_info().devices[pCudaCtx->device].integrated;

  ggml_cuda_stream_context &vStreamCtx = pCudaCtx->stream_context();
  bool vIsConcurrentEventActive = false;
  ggml_cuda_concurrent_event *vConcurrentEvent = nullptr;
  bool vShouldLaunchConcurrentEvents = false;

  const auto fTryLaunchConcurrentEvent = [&](const ggml_tensor *node) {
    if (vStreamCtx.concurrent_events.find(node) != vStreamCtx.concurrent_events.end()) {
      vConcurrentEvent = &vStreamCtx.concurrent_events[node];

      vIsConcurrentEventActive = true;

      GGML_LOG_DEBUG("Launching %d streams at %s\n", vConcurrentEvent->n_streams, node->name);

      cudaStream_t vMainStream = pCudaCtx->stream(); // this should be stream 0
      GGML_ASSERT(pCudaCtx->curr_stream_no == 0);
      CUDA_CHECK(cudaEventRecord(vConcurrentEvent->fork_event, vMainStream));

      for (int vI = 1; vI <= vConcurrentEvent->n_streams; ++vI) {
        cudaStream_t vStream = pCudaCtx->stream(pCudaCtx->device, vI);
        CUDA_CHECK(cudaStreamWaitEvent(vStream, vConcurrentEvent->fork_event));
      }
    }
  };

  while (!vGraphEvaluatedOrCaptured) {
    // Only perform the graph execution if CUDA graphs are not enabled, or we are capturing the graph.
    // With the use of CUDA graphs, the execution will be performed by the graph launch.
    if (!pUseCudaGraph || pCudaGraphUpdateRequired) {
      [[maybe_unused]] int vPrevI = 0;

      if (vStreamCtx.concurrent_events.size() > 0) {
        vShouldLaunchConcurrentEvents = true;
        for (const auto &[tensor, event] : vStreamCtx.concurrent_events) {
          vShouldLaunchConcurrentEvents = vShouldLaunchConcurrentEvents && event.is_valid();
        }
      }

      if (vShouldLaunchConcurrentEvents) {
        // Restore original node order within each concurrent region to enable fusion within streams

        std::unordered_map<const ggml_tensor *, int> dNodeToIdx;
        dNodeToIdx.reserve(pCgraph->n_nodes);
        for (int vI = 0; vI < pCgraph->n_nodes; ++vI) {
          dNodeToIdx[pCgraph->nodes[vI]] = vI;
        }

        for (auto &[fork_node, event] : vStreamCtx.concurrent_events) {
          // Find positions of all nodes from this event in the current graph
          std::vector<int> lPositions;
          lPositions.reserve(event.original_order.size());

          bool vAllFound = true;
          for (const ggml_tensor *vOrigNode : event.original_order) {
            auto vIt = dNodeToIdx.find(vOrigNode);
            if (vIt != dNodeToIdx.end()) {
              lPositions.push_back(vIt->second);
            } else {
              vAllFound = false;
              break;
            }
          }

          if (!vAllFound || lPositions.size() != event.original_order.size()) {
            continue;
          }

          // Sort positions to get contiguous range
          std::vector<int> lSortedPositions = lPositions;
          std::sort(lSortedPositions.begin(), lSortedPositions.end());

          bool vIsContiguous = true;
          for (size_t vI = 1; vI < lSortedPositions.size(); ++vI) {
            if (lSortedPositions[vI] != lSortedPositions[vI - 1] + 1) {
              vIsContiguous = false;
              break;
            }
          }

          if (!vIsContiguous) {
            continue;
          }

          // Restore original order at the sorted positions
          int vStartPos = lSortedPositions[0];
          for (size_t vI = 0; vI < event.original_order.size(); ++vI) {
            pCgraph->nodes[vStartPos + vI] = const_cast<ggml_tensor *>(event.original_order[vI]);
          }
        }
      } else {
        vStreamCtx.concurrent_events.clear();
      }

      for (int vI = 0; vI < pCgraph->n_nodes; vI++) {
        ggml_tensor *node = pCgraph->nodes[vI];
        if (vIsConcurrentEventActive) {
          GGML_ASSERT(vConcurrentEvent);

          if (node == vConcurrentEvent->join_node) {
            pCudaCtx->curr_stream_no = 0;
            for (int vI = 1; vI <= vConcurrentEvent->n_streams; ++vI) {
              // Wait on join events of forked streams in the main stream
              CUDA_CHECK(
                  cudaEventRecord(vConcurrentEvent->join_events[vI - 1], pCudaCtx->stream(pCudaCtx->device, vI)));
              CUDA_CHECK(cudaStreamWaitEvent(pCudaCtx->stream(), vConcurrentEvent->join_events[vI - 1]));
            }

            vIsConcurrentEventActive = false;
            vConcurrentEvent = nullptr;
          } else {
            GGML_ASSERT(vConcurrentEvent->stream_mapping.find(node) != vConcurrentEvent->stream_mapping.end());
            pCudaCtx->curr_stream_no = vConcurrentEvent->stream_mapping[node];
            GGML_LOG_DEBUG("Setting stream no to %d for node %s\n", pCudaCtx->curr_stream_no, node->name);
          }
        } else if (vI - vPrevI > 1) {
          // the previous node was fused
          const ggml_tensor *vPrevNode = pCgraph->nodes[vI - 1];
          fTryLaunchConcurrentEvent(vPrevNode);

          if (vIsConcurrentEventActive) {
            pCudaCtx->curr_stream_no = vConcurrentEvent->stream_mapping[node];
            GGML_LOG_DEBUG("Setting stream no to %d for node %s\n", pCudaCtx->curr_stream_no, node->name);
          }
        }

        vPrevI = vI;

        if (fGgmlCudaIsViewOrNoop(node)) {
          continue;
        }

        if ((node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
          continue;
        }

        int vNodesToSkip = fGgmlCudaTryFuse(pCudaCtx, pCgraph, vI);

        if (vNodesToSkip != 0) {
#ifdef GGML_CUDA_DEBUG
          const int last_fused = vI + vNodesToSkip;
          GGML_LOG_INFO("nodes_fused: %d, first: %s (%s), last: %s (%s)\n", vNodesToSkip + 1, ggml_op_name(node->op),
                        node->name, ggml_op_name(pCgraph->nodes[last_fused]->op), pCgraph->nodes[last_fused]->name);
#endif
          vI += vNodesToSkip;
          continue;
        }
#ifndef NDEBUG
        assert(node->buffer->buft == ggml_backend_cuda_buffer_type(pCudaCtx->device));
        for (int vJ = 0; vJ < GGML_MAX_SRC; vJ++) {
          if (node->src[vJ] != nullptr) {
            assert(node->src[vJ]->buffer);
            assert(node->src[vJ]->buffer->buft == ggml_backend_cuda_buffer_type(pCudaCtx->device) ||
                   (cIntegrated && fGgmlBackendBuftIsCudaHost(node->src[vJ]->buffer->buft)));
          }
        }
#else
        GGML_UNUSED(cIntegrated);
#endif // NDEBUG

        bool vOk = fGgmlCudaComputeForward(*pCudaCtx, node);
        if (!vOk) {
          GGML_LOG_ERROR("%s: op not supported %s (%s)\n", __func__, node->name, ggml_op_name(node->op));
        }
        GGML_ASSERT(vOk);

        if (!vIsConcurrentEventActive) {
          fTryLaunchConcurrentEvent(node);
        }
      }
    }

#ifdef USE_CUDA_GRAPH
    ggml_cuda_graph *vGraph = pCudaCtx->cuda_graph(pGraphKey);
    if (pUseCudaGraph && pCudaGraphUpdateRequired) { // End CUDA graph capture
      if (vGraph->graph != nullptr) {
        CUDA_CHECK(cudaGraphDestroy(vGraph->graph));
        vGraph->graph = nullptr;
      }

      CUDA_CHECK(cudaStreamEndCapture(pCudaCtx->stream(), &vGraph->graph));
      vGraphEvaluatedOrCaptured = true; // CUDA graph has been captured

      std::lock_guard<std::mutex> vLock(ggml_cuda_lock);
      if (ggml_cuda_lock_counter.fetch_sub(1, std::memory_order_relaxed) == 1) {
        ggml_cuda_lock_cv.notify_all();
      }
    } else {
      vGraphEvaluatedOrCaptured = true; // ggml graph has been directly evaluated
    }
  }

  if (pUseCudaGraph) {
    ggml_cuda_graph *vGraph = pCudaCtx->cuda_graph(pGraphKey);
    if (vGraph->instance == nullptr) { // Create executable graph from captured graph.
      CUDA_CHECK(cudaGraphInstantiate(&vGraph->instance, vGraph->graph, NULL, NULL, 0));
    }
    if (pCudaGraphUpdateRequired) { // Update graph executable
      fGgmlCudaGraphUpdateExecutable(pCudaCtx, pGraphKey);
    }
    // Launch graph
    CUDA_CHECK(cudaGraphLaunch(vGraph->instance, pCudaCtx->stream()));
#else
    GGML_UNUSED(pGraphKey);
    vGraphEvaluatedOrCaptured = true;
#endif // USE_CUDA_GRAPH
  }
}

#ifdef USE_CUDA_GRAPH
static bool fGgmlCudaGraphSetEnabled(ggml_backend_cuda_context *pCudaCtx, const void *pGraphKey) {
  ggml_cuda_graph *vGraph = pCudaCtx->cuda_graph(pGraphKey);

  if (vGraph->graph == nullptr) {
    if (ggml_cuda_info().devices[pCudaCtx->device].cc < GGML_CUDA_CC_VOLTA) {
      if (!vGraph->disable_due_to_gpu_arch) {
        GGML_LOG_DEBUG("%s: disabling CUDA graphs due to GPU architecture\n", __func__);
      }
      vGraph->disable_due_to_gpu_arch = true;
    }
  }

  return vGraph->is_enabled();
}
#endif // USE_CUDA_GRAPH

static enum ggml_status fGgmlBackendCudaGraphCompute(ggml_backend_t pBackend, ggml_cgraph *pCgraph) {
  ggml_backend_cuda_context *vCudaCtx = (ggml_backend_cuda_context *)pBackend->context;

  ggml_cuda_set_device(vCudaCtx->device);

  bool vUseCudaGraph = false;
  bool vCudaGraphUpdateRequired = false;
  const void *vGraphKey = nullptr;

#ifdef USE_CUDA_GRAPH
  vGraphKey = fGgmlCudaGraphGetKey(pCgraph);

  fGgmlCudaGraphSetEnabled(vCudaCtx, vGraphKey);

  ggml_cuda_graph *vGraph = vCudaCtx->cuda_graph(vGraphKey);
  if (vGraph->is_enabled()) {
    const bool cGraphCompatible = fGgmlCudaGraphCheckCompability(pCgraph);
    if (cGraphCompatible) {
      const bool cPropertiesChanged = fGgmlCudaGraphUpdateRequired(vCudaCtx, pCgraph);

      if (!vGraph->warmup_complete) {
        // Warmup: need at least 2 calls with no property change on the 2nd call
        if (!cPropertiesChanged) {
          vGraph->warmup_complete = true;
          GGML_LOG_DEBUG("%s: CUDA graph warmup complete\n", __func__);
          vUseCudaGraph = true;
          vCudaGraphUpdateRequired = true;
        }
        // else: properties changed or first call - execute directly (use_cuda_graph stays false)
      } else {
        // Post-warmup: normal CUDA graph operation
        if (cPropertiesChanged) {
          // Properties changed - reset warmup, execute directly until stable again
          vGraph->warmup_complete = false;
          GGML_LOG_DEBUG("%s: CUDA graph warmup reset\n", __func__);
        } else {
          vUseCudaGraph = true;
          vCudaGraphUpdateRequired = vGraph->instance == nullptr;
        }
      }
    }
  }
#endif // USE_CUDA_GRAPH

  if (vUseCudaGraph && vCudaGraphUpdateRequired) {
    // Start CUDA graph capture
    {
      std::lock_guard<std::mutex> vLock(ggml_cuda_lock);
      ggml_cuda_lock_counter.fetch_add(1, std::memory_order_relaxed);
    }

    CUDA_CHECK(cudaStreamBeginCapture(vCudaCtx->stream(), cudaStreamCaptureModeRelaxed));
  }

  fGgmlCudaGraphEvaluateAndCapture(vCudaCtx, pCgraph, vUseCudaGraph, vCudaGraphUpdateRequired, vGraphKey);

  return GGML_STATUS_SUCCESS;
}

static void fGgmlBackendCudaEventRecord(ggml_backend_t pBackend, ggml_backend_event_t pEvent) {
  ggml_backend_cuda_context *vCudaCtx = (ggml_backend_cuda_context *)pBackend->context;

  CUDA_CHECK(cudaEventRecord((cudaEvent_t)pEvent->context, vCudaCtx->stream()));
}

static void fGgmlBackendCudaEventWait(ggml_backend_t pBackend, ggml_backend_event_t pEvent) {
  ggml_backend_cuda_context *vCudaCtx = (ggml_backend_cuda_context *)pBackend->context;

  if (ggml_backend_is_cuda(pBackend)) {
    CUDA_CHECK(cudaStreamWaitEvent(vCudaCtx->stream(), (cudaEvent_t)pEvent->context, 0));
  } else {
#if 0
        // untested
        auto wait_fn = [](void * user_data) {
            ggml_backend_event_t pEvent = (ggml_backend_event_t)user_data;
            ggml_backend_event_synchronize(pEvent);
        };

        CUDA_CHECK(cudaLaunchHostFunc(vCudaCtx->stream(), wait_fn, pEvent));
#endif
    GGML_ABORT("fatal error");
  }
}

static void fGgmlBackendCudaGraphOptimize(ggml_backend_t pBackend, ggml_cgraph *pCgraph,
                                          ggml_backend_graph_optimize_params *pParams) {
  GGML_UNUSED(pParams);
  ggml_backend_cuda_context *vCudaCtx = (ggml_backend_cuda_context *)pBackend->context;

#ifdef USE_CUDA_GRAPH
  const void *vGraphKey = fGgmlCudaGraphGetKey(pCgraph);
  const bool cUseCudaGraph = fGgmlCudaGraphSetEnabled(vCudaCtx, vGraphKey);
#else
  const bool cUseCudaGraph = false;
  GGML_UNUSED(vCudaCtx);
  GGML_UNUSED(pCgraph);
#endif

  static bool vEnableGraphOptimization = [] {
    const char *vEnv = getenv("GGML_CUDA_GRAPH_OPT");
    return vEnv != nullptr && atoi(vEnv) == 1;
  }();

  if (!vEnableGraphOptimization) {
    return;
  }

  ggml_cuda_stream_context &vStreamContext = vCudaCtx->stream_context();
  vStreamContext.reset();

  if (!cUseCudaGraph || ggml_backend_cuda_get_device_count() != 1) {
    return;
  }

  // number of out-degrees for a particular node
  std::unordered_map<const ggml_tensor *, int> dFanOut;
  // reverse mapping of node to index in the cgraph
  std::unordered_map<const ggml_tensor *, int> dNodeIndices;

  const auto &fIsNoop = [](const ggml_tensor *pNode) -> bool {
    return ggml_is_empty(pNode) || pNode->op == GGML_OP_NONE || pNode->op == GGML_OP_RESHAPE ||
           pNode->op == GGML_OP_TRANSPOSE || pNode->op == GGML_OP_VIEW || pNode->op == GGML_OP_PERMUTE;
  };

  const auto &fDependsOn = [](const ggml_tensor *pDst, const ggml_tensor *pSrc) -> bool {
    for (uint32_t vS = 0; vS < GGML_MAX_SRC; ++vS) {
      if (pDst->src[vS] == pSrc) {
        return true;
      }
    }
    // implicit dependency if they view the same tensor
    const ggml_tensor *vDst2 = pDst->view_src ? pDst->view_src : pDst;
    const ggml_tensor *vSrc2 = pSrc->view_src ? pSrc->view_src : pSrc;
    if (vDst2 == vSrc2) {
      return true;
    }
    return false;
  };

  for (int vNodeIdx = 0; vNodeIdx < pCgraph->n_nodes; vNodeIdx++) {
    const ggml_tensor *vNode = pCgraph->nodes[vNodeIdx];
    dNodeIndices[vNode] = vNodeIdx;

    if (fIsNoop(vNode)) {
      continue;
    }
    for (int vSrcIdx = 0; vSrcIdx < GGML_MAX_SRC; ++vSrcIdx) {
      const ggml_tensor *vSrc = pCgraph->nodes[vNodeIdx]->src[vSrcIdx];
      // TODO: check why nrows > 1 fails
      if (vNode && !fIsNoop(vNode) && ggml_nrows(vNode) <= 1) {
        dFanOut[vSrc] += 1;
      }
    }
  }

  // Target Q, K, V for concurrency
  // this is a more general way to find nodes which can be candidates for concurrency (although it has not been tested
  // for anything else):
  // 1. find fan-out (fork) nodes where the same input is used at least N times (in QKV, it would be "attn-norm")
  // 2. find the join node, where 2 or more of the outputs are required (in QKV, this would "KQ" or "flash-attn")
  // 3. account for all branches from the fork to the join
  // 4. To extend lifetimes of the tensors, we interleave the branches (see below for more details)
  // 5. save the original cgraph and restore it in graph_compute, to enable fusion within streams

  const int cMinFanOut = 3;
  const int cMaxFanOut = 3;

  // store {fork_idx, join_idx}
  std::vector<std::pair<int, int>> lConcurrentNodeRanges;

  for (const auto &[root_node, count] : dFanOut) {
    if (count >= cMinFanOut && count <= cMaxFanOut) {
      const int cRootNodeIdx = dNodeIndices[root_node];

      // only optimize for attn_norm
      // TODO: make this more generic
      if (!strstr(root_node->name, "attn_norm")) {
        continue;
      }

      bool vIsPartOfEvent = false;
      for (const auto &[start, end] : lConcurrentNodeRanges) {
        if (cRootNodeIdx >= start && cRootNodeIdx <= end) {
          vIsPartOfEvent = true;
        }
      }

      if (vIsPartOfEvent) {
        continue;
      }

      std::vector<std::vector<const ggml_tensor *>> vNodesPerBranch;
      for (int vI = cRootNodeIdx + 1; vI < pCgraph->n_nodes; ++vI) {
        const ggml_tensor *vNode = pCgraph->nodes[vI];
        if (!fIsNoop(vNode) && fDependsOn(vNode, root_node)) {
          vNodesPerBranch.push_back({vNode});
        }
      }

      GGML_ASSERT(vNodesPerBranch.size() == (size_t)count);

      // find the join point
      const ggml_tensor *vJoinNode = nullptr;

      const auto &fBelongsToBranch = [&](const ggml_tensor *pNode,
                                         const std::vector<const ggml_tensor *> &pBranch) -> bool {
        for (const ggml_tensor *vN : pBranch) {
          if (fDependsOn(pNode, vN)) {
            return true;
          }
        }
        return false;
      };

      for (int vI = cRootNodeIdx + 1; vI < pCgraph->n_nodes; ++vI) {
        const ggml_tensor *vCurrNode = pCgraph->nodes[vI];

        int vNumJoins = 0;
        for (size_t vBranchIdx = 0; vBranchIdx < vNodesPerBranch.size(); vBranchIdx++) {
          if (fBelongsToBranch(vCurrNode, vNodesPerBranch[vBranchIdx])) {
            vNumJoins++;
          }
        }

        if (vNumJoins >= 2) {
          vJoinNode = vCurrNode;
          break;
        }

        bool vFoundBranch = false;
        for (size_t vBranchIdx = 0; vBranchIdx < vNodesPerBranch.size(); vBranchIdx++) {
          std::vector<const ggml_tensor *> &lBranchVec = vNodesPerBranch[vBranchIdx];
          if (fBelongsToBranch(vCurrNode, lBranchVec)) {
            // continue accumulating
            if (std::find(lBranchVec.begin(), lBranchVec.end(), vCurrNode) == lBranchVec.end()) {
              lBranchVec.push_back(vCurrNode);
            }
            vFoundBranch = true;
          }
        }

        if (!vFoundBranch && fIsNoop(vCurrNode)) {
          // we can put it in any branch because it will be ignored
          vNodesPerBranch[0].push_back({vCurrNode});
        }
      }

      if (vJoinNode) {
        // Create ggml_cuda_concurrent_event
        ggml_cuda_concurrent_event vConcurrentEvent(vNodesPerBranch.size());
        vConcurrentEvent.join_node = vJoinNode;

        for (size_t vBranchIdx = 0; vBranchIdx < vNodesPerBranch.size(); vBranchIdx++) {
          for (const ggml_tensor *vN : vNodesPerBranch[vBranchIdx]) {
            vConcurrentEvent.stream_mapping[vN] = vBranchIdx + 1;
          }
        }

        int vForkNodeIdx = dNodeIndices[root_node];
        int vJoinNodeIdx = dNodeIndices[vJoinNode];

        int vCurrentBranchIdx = 0;
        int vCurrentNodeIdx = vForkNodeIdx + 1;
        const int cNBranches = vNodesPerBranch.size();

        int vTotalBranchNodes = 0;
        for (std::vector<const ggml_tensor *> lBranchNodes : vNodesPerBranch) {
          vTotalBranchNodes += lBranchNodes.size();
        }

        // there are other nodes in the middle which are unaccounted for
        // usually (cpy) nodes, then ignore this fork
        if (vJoinNodeIdx - vForkNodeIdx - 1 != vTotalBranchNodes) {
          GGML_LOG_DEBUG("Skipping %s because the number of nodes in the middle is not equal to the total number of "
                         "branch nodes %d != %d\n",
                         root_node->name, vJoinNodeIdx - vForkNodeIdx - 1, vTotalBranchNodes);
          continue;
        }

        // Save the original order of nodes in this region before interleaving
        // This is used later to restore grouping for fusion within streams
        vConcurrentEvent.original_order.reserve(vTotalBranchNodes);
        for (int vI = vForkNodeIdx + 1; vI < vJoinNodeIdx; ++vI) {
          vConcurrentEvent.original_order.push_back(pCgraph->nodes[vI]);
        }

        std::unordered_map<const ggml_tensor *, ggml_cuda_concurrent_event> &vConcurrentEvents =
            vCudaCtx->stream_context().concurrent_events;
        GGML_ASSERT(vConcurrentEvents.find(root_node) == vConcurrentEvents.end());
        vConcurrentEvents.emplace(root_node, std::move(vConcurrentEvent));
        GGML_LOG_DEBUG("Adding stream at node %s %p\n", root_node->name, root_node);
        lConcurrentNodeRanges.emplace_back(vForkNodeIdx, vJoinNodeIdx);

        // interleave tensors to extend lifetimes so that ggml graph doesn't recycle them
        // example transformation:
        // [attn-norm, QMul, QNorm, QRope, KMul, KNorm, KRope, VMul, attn] ->
        // [attn-norm, QMul, KMul, VMul, QNorm, VNorm, QRope, KRope, attn]
        while (vCurrentNodeIdx < vJoinNodeIdx) {
          std::vector<const ggml_tensor *> &lBranchNodes = vNodesPerBranch[vCurrentBranchIdx];

          bool vHasNode = false;
          for (std::vector<const ggml_tensor *> lBranchNode : vNodesPerBranch) {
            vHasNode |= lBranchNode.size() > 0;
          }

          GGML_ASSERT(vHasNode);

          if (lBranchNodes.empty()) {
            vCurrentBranchIdx = (vCurrentBranchIdx + 1) % cNBranches;
            continue;
          }

          pCgraph->nodes[vCurrentNodeIdx] = const_cast<ggml_tensor *>(lBranchNodes.front());
          vCurrentNodeIdx++;
          lBranchNodes.erase(lBranchNodes.begin());

          // append all empty nodes
          while (!lBranchNodes.empty() && fIsNoop(lBranchNodes.front())) {
            pCgraph->nodes[vCurrentNodeIdx] = const_cast<ggml_tensor *>(lBranchNodes.front());
            vCurrentNodeIdx++;
            lBranchNodes.erase(lBranchNodes.begin());
          }

          vCurrentBranchIdx = (vCurrentBranchIdx + 1) % cNBranches;
        }
      }
    }
  }
}

static const ggml_backend_i ggml_backend_cuda_interface = {
    /* .get_name                = */ fGgmlBackendCudaGetName,
    /* .free                    = */ fGgmlBackendCudaFree,
    /* .set_tensor_async        = */ fGgmlBackendCudaSetTensorAsync,
    /* .get_tensor_async        = */ fGgmlBackendCudaGetTensorAsync,
    /* .set_tensor_2d_async     = */ fGgmlBackendCudaSetTensor2dAsync,
    /* .get_tensor_2d_async     = */ fGgmlBackendCudaGetTensor2dAsync,
    /* .cpy_tensor_async        = */ fGgmlBackendCudaCpyTensorAsync,
    /* .synchronize             = */ fGgmlBackendCudaSynchronize,
    /* .graph_plan_create       = */ NULL,
    /* .graph_plan_free         = */ NULL,
    /* .graph_plan_update       = */ NULL,
    /* .graph_plan_compute      = */ NULL,
    /* .graph_compute           = */ fGgmlBackendCudaGraphCompute,
    /* .event_record            = */ fGgmlBackendCudaEventRecord,
    /* .event_wait              = */ fGgmlBackendCudaEventWait,
    /* .graph_optimize          = */ fGgmlBackendCudaGraphOptimize,
};

static ggml_guid_t fGgmlBackendCudaGuid() {
  static ggml_guid aGuid = {0x2c, 0xdd, 0xe8, 0x1c, 0x65, 0xb3, 0x65, 0x73,
                            0x6a, 0x12, 0x88, 0x61, 0x1c, 0xc9, 0xdc, 0x25};
  return &aGuid;
}

bool ggml_backend_is_cuda(ggml_backend_t pBackend) {
  return pBackend != NULL && ggml_guid_matches(pBackend->guid, fGgmlBackendCudaGuid());
}

int ggml_backend_cuda_get_device_count() { return ggml_cuda_info().device_count; }

static std::string fGgmlCudaDeviceDescription(int pDevice) {
  cudaDeviceProp vProp;
  CUDA_CHECK(cudaGetDeviceProperties(&vProp, fGgmlCudaGetPhysicalDevice(pDevice)));

  const ggml_cuda_device_info &vInfo = ggml_cuda_info();
  std::string vDescription = vProp.name;
  if (vInfo.device_count > vInfo.physical_device_count) {
    vDescription += " (physical device " + std::to_string(vInfo.devices[pDevice].physical_device) +
                    ", virtual device " + std::to_string(vInfo.devices[pDevice].virtual_index) + ")";
  }
  return vDescription;
}

void ggml_backend_cuda_get_device_description(int pDevice, char *pDescription, size_t pDescriptionSize) {
  snprintf(pDescription, pDescriptionSize, "%s", fGgmlCudaDeviceDescription(pDevice).c_str());
}

static int fGgmlCudaPhysicalDeviceShareCount(int pDevice) {
  const ggml_cuda_device_info &vInfo = ggml_cuda_info();
  GGML_ASSERT(pDevice >= 0 && pDevice < vInfo.device_count);
  return vInfo.devices[pDevice].physical_share_count;
}

void ggml_backend_cuda_get_device_memory(int pDevice, size_t *pFree, size_t *pTotal) {
  ggml_cuda_set_device(pDevice);

  CUDA_CHECK(cudaMemGetInfo(pFree, pTotal));

  // virtual devices sharing one physical GPU share its memory pool; split it between them
  const int cShareCount = fGgmlCudaPhysicalDeviceShareCount(pDevice);
  *pFree /= cShareCount;
  *pTotal /= cShareCount;
}

static bool fCudaHostRegistrationEnabled() {
  const char *vCValue = getenv("GGML_CUDA_REGISTER_HOST");
  return vCValue == nullptr || strcmp(vCValue, "0") != 0;
}

bool ggml_backend_cuda_register_host_buffer(void *pBuffer, size_t size) {
  // This server specializes in MoE: registration remains enabled unless
  // explicitly disabled for diagnostics.
  if (!fCudaHostRegistrationEnabled()) {
    return false;
  }

#if CUDART_VERSION >= 11010 || defined(GGML_USE_HIP)
  cudaError_t vErr = cudaHostRegister(pBuffer, size, cudaHostRegisterPortable | cudaHostRegisterReadOnly);
  if (vErr != cudaSuccess) {
    // clear the error
    (void)cudaGetLastError();

    GGML_LOG_WARN("%s: could not pin %.2f MiB of mmap memory: %s\n", __func__, size / 1024.0 / 1024.0,
                  cudaGetErrorString(vErr));
    return false;
  }
  return true;
#else
  GGML_UNUSED(pBuffer);
  GGML_UNUSED(size);
  return false;
#endif // CUDART_VERSION >= 11010
}

void ggml_backend_cuda_unregister_host_buffer(void *pBuffer) {
  cudaError_t vErr = cudaHostUnregister(pBuffer);
  if (vErr != cudaSuccess) {
    // clear the error
    (void)cudaGetLastError();
  }
}

// backend device

struct ggml_backend_cuda_device_context {
  int device;
  std::string name;
  std::string vDescription;
  std::string vPciBusId;
  int vOpOffloadMinBatchSize;
};

static const char *fGgmlBackendCudaDeviceGetName(ggml_backend_dev_t pDev) {
  ggml_backend_cuda_device_context *vCtx = (ggml_backend_cuda_device_context *)pDev->context;
  return vCtx->name.c_str();
}

static const char *fGgmlBackendCudaDeviceGetDescription(ggml_backend_dev_t pDev) {
  ggml_backend_cuda_device_context *vCtx = (ggml_backend_cuda_device_context *)pDev->context;
  return vCtx->vDescription.c_str();
}

#if defined(__linux__)
// Helper function to get available memory from /proc/meminfo for UMA systems
static bool fGgmlBackendCudaGetAvailableUmaMemory(long *pAvailableMemoryKb, long *pFreeSwapKb) {
  FILE *vMeminfoFile = nullptr;
  // 2KB buffer for reading /proc/meminfo since it does not report size info, should be enough
  const size_t cBufferSize = 2048;
  auto vFileBuffer = std::make_unique<char[]>(cBufferSize);
  size_t vBytesRead = 0;
  long vHugeTlbTotalPages = -1;
  long vHugeTlbFreePages = -1;
  long vHugeTlbPageSize = -1;

  if (pAvailableMemoryKb == nullptr || pFreeSwapKb == nullptr) {
    return false;
  }

  vMeminfoFile = fopen("/proc/meminfo", "r");
  if (vMeminfoFile == nullptr) {
    GGML_LOG_ERROR("%s: failed to open /proc/meminfo\n", __func__);
    return false;
  }

  // Read file into buffer
  vBytesRead = fread(vFileBuffer.get(), 1, cBufferSize - 1, vMeminfoFile);
  fclose(vMeminfoFile);

  if (vBytesRead == 0) {
    GGML_LOG_ERROR("%s: failed to read from /proc/meminfo\n", __func__);
    return false;
  }
  vFileBuffer[vBytesRead] = '\0';

  *pAvailableMemoryKb = -1;
  *pFreeSwapKb = -1;

  // Parse the file buffer line by line
  char *vLine = vFileBuffer.get();
  char *vLineNext;
  while (vLine < vFileBuffer.get() + vBytesRead) {
    // Find the end of the current line
    vLineNext = strchr(vLine, '\n');
    if (vLineNext != nullptr) {
      *vLineNext = '\0';
      vLineNext++;
    } else {
      vLineNext = vFileBuffer.get() + vBytesRead;
    }

    long vValue;
    if (sscanf(vLine, "MemAvailable: %ld kB", &vValue) == 1) {
      *pAvailableMemoryKb = vValue;
    } else if (sscanf(vLine, "SwapFree: %ld kB", &vValue) == 1) {
      *pFreeSwapKb = vValue;
    } else if (sscanf(vLine, "HugePages_Total: %ld", &vValue) == 1) {
      vHugeTlbTotalPages = vValue;
    } else if (sscanf(vLine, "HugePages_Free: %ld", &vValue) == 1) {
      vHugeTlbFreePages = vValue;
    } else if (sscanf(vLine, "Hugepagesize: %ld kB", &vValue) == 1) {
      vHugeTlbPageSize = vValue;
    }

    vLine = vLineNext;
  }

  if (vHugeTlbTotalPages != 0 && vHugeTlbTotalPages != -1) {
    *pAvailableMemoryKb = vHugeTlbFreePages * vHugeTlbPageSize;

    // Hugetlbfs pages are not swappable.
    *pFreeSwapKb = 0;
  }

  GGML_LOG_DEBUG("%s: final available_memory_kb: %ld\n", __func__, *pAvailableMemoryKb);
  return true;
}
#endif // defined(__linux__)

static void fGgmlBackendCudaDeviceGetMemory(ggml_backend_dev_t pDev, size_t *pFree, size_t *pTotal) {
  ggml_backend_cuda_device_context *vCtx = (ggml_backend_cuda_device_context *)pDev->context;
  ggml_cuda_set_device(vCtx->device);
  cudaError_t vErr = cudaMemGetInfo(pFree, pTotal);
  if (vErr != cudaSuccess) {
    (void)cudaGetLastError();
    GGML_LOG_WARN("%s: cudaMemGetInfo failed (%s), returning 0/0\n", __func__, cudaGetErrorString(vErr));
    *pFree = 0;
    *pTotal = 0;
    return;
  }

#if defined(__linux__)
  // Check if this is a UMA (Unified Memory Architecture) system
  cudaDeviceProp vProp;
  CUDA_CHECK(cudaGetDeviceProperties(&vProp, fGgmlCudaGetPhysicalDevice(vCtx->device)));

  // Check if UMA is explicitly enabled via environment variable
  bool vUmaEnv = getenv("GGML_CUDA_ENABLE_UNIFIED_MEMORY") != nullptr;
  bool vIsUma = vProp.integrated > 0 || vUmaEnv;

  if (vIsUma) {
    // For UMA systems (like DGX Spark), use system memory info
    long vAvailableMemoryKb = 0;
    long vFreeSwapKb = 0;

    if (fGgmlBackendCudaGetAvailableUmaMemory(&vAvailableMemoryKb, &vFreeSwapKb) && vAvailableMemoryKb > 0) {
      *pFree = (size_t)vAvailableMemoryKb * 1024;
    } else {
      GGML_LOG_ERROR("%s: /proc/meminfo reading failed, using cudaMemGetInfo\n", __func__);
    }
  }
#endif // defined(__linux__)

  // virtual devices sharing one physical GPU share its memory pool; split it between them
  const int cShareCount = fGgmlCudaPhysicalDeviceShareCount(vCtx->device);
  *pFree /= cShareCount;
  *pTotal /= cShareCount;
}

static enum ggml_backend_dev_type fGgmlBackendCudaDeviceGetType(ggml_backend_dev_t pDev) {
  ggml_backend_cuda_device_context *vCtx = (ggml_backend_cuda_device_context *)pDev->context;

  cudaDeviceProp vProp;
  CUDA_CHECK(cudaGetDeviceProperties(&vProp, fGgmlCudaGetPhysicalDevice(vCtx->device)));

  return vProp.integrated ? GGML_BACKEND_DEVICE_TYPE_IGPU : GGML_BACKEND_DEVICE_TYPE_GPU;
}

static void fGgmlBackendCudaDeviceGetProps(ggml_backend_dev_t pDev, ggml_backend_dev_props *pProps) {
  ggml_backend_cuda_device_context *vCtx = (ggml_backend_cuda_device_context *)pDev->context;

  pProps->name = fGgmlBackendCudaDeviceGetName(pDev);
  pProps->description = fGgmlBackendCudaDeviceGetDescription(pDev);
  pProps->type = fGgmlBackendCudaDeviceGetType(pDev);
  pProps->device_id = vCtx->vPciBusId.empty() ? nullptr : vCtx->vPciBusId.c_str();
  fGgmlBackendCudaDeviceGetMemory(pDev, &pProps->memory_free, &pProps->memory_total);

  bool vHostBuffer = getenv("GGML_CUDA_NO_PINNED") == nullptr;
#ifdef GGML_CUDA_NO_PEER_COPY
  bool vEvents = false;
#else
  bool vEvents = true;
#endif

  pProps->caps = {
      /* .async                 = */ true,
      /* .host_buffer           = */ vHostBuffer,
      /* .buffer_from_host_ptr  = */ false,
      /* .events                = */ vEvents,
  };
}

static ggml_backend_t fGgmlBackendCudaDeviceInitBackend(ggml_backend_dev_t pDev, const char *pParams) {
  GGML_UNUSED(pParams);
  ggml_backend_cuda_device_context *vCtx = (ggml_backend_cuda_device_context *)pDev->context;
  return ggml_backend_cuda_init(vCtx->device);
}

static ggml_backend_buffer_type_t fGgmlBackendCudaDeviceGetBufferType(ggml_backend_dev_t pDev) {
  ggml_backend_cuda_device_context *vCtx = (ggml_backend_cuda_device_context *)pDev->context;
  return ggml_backend_cuda_buffer_type(vCtx->device);
}

static ggml_backend_buffer_type_t fGgmlBackendCudaDeviceGetHostBufferType(ggml_backend_dev_t pDev) {
  GGML_UNUSED(pDev);
  return ggml_backend_cuda_host_buffer_type();
}

// TODO: move these functions here
static bool fGgmlBackendCudaDeviceSupportsOp(ggml_backend_dev_t pDev, const ggml_tensor *pOp) {
  ggml_backend_cuda_device_context *vDevCtx = (ggml_backend_cuda_device_context *)pDev->context;

  // check if all the sources are allocated on this device
  for (int vI = 0; vI < GGML_MAX_SRC; vI++) {
    if (pOp->src[vI] && pOp->src[vI]->buffer && fGgmlBackendBuftIsCuda(pOp->src[vI]->buffer->buft)) {
      ggml_backend_cuda_buffer_type_context *vBuftCtx =
          (ggml_backend_cuda_buffer_type_context *)pOp->src[vI]->buffer->buft->context;
      if (vBuftCtx->device != vDevCtx->device) {
        return false;
      }
    }
  }

  switch (pOp->op) {
  case GGML_OP_UNARY:
    switch (ggml_get_unary_op(pOp)) {
    case GGML_UNARY_OP_ABS:
    case GGML_UNARY_OP_SGN:
    case GGML_UNARY_OP_NEG:
    case GGML_UNARY_OP_STEP:
    case GGML_UNARY_OP_GELU:
    case GGML_UNARY_OP_SILU:
    case GGML_UNARY_OP_RELU:
    case GGML_UNARY_OP_SIGMOID:
    case GGML_UNARY_OP_HARDSIGMOID:
    case GGML_UNARY_OP_HARDSWISH:
    case GGML_UNARY_OP_GELU_ERF:
    case GGML_UNARY_OP_GELU_QUICK:
    case GGML_UNARY_OP_TANH:
    case GGML_UNARY_OP_EXP:
    case GGML_UNARY_OP_EXPM1:
    case GGML_UNARY_OP_SOFTPLUS:
    case GGML_UNARY_OP_ELU:
    case GGML_UNARY_OP_XIELU:
    case GGML_UNARY_OP_FLOOR:
    case GGML_UNARY_OP_CEIL:
    case GGML_UNARY_OP_ROUND:
    case GGML_UNARY_OP_TRUNC:
      // TODO: should become:
      // return ggml_is_contiguous_rows(op->src[0]);
      return ggml_is_contiguous(pOp->src[0]);
    default:
      return false;
    }
    break;
  case GGML_OP_GLU:
    switch (ggml_get_glu_op(pOp)) {
    case GGML_GLU_OP_REGLU:
    case GGML_GLU_OP_GEGLU:
    case GGML_GLU_OP_SWIGLU:
    case GGML_GLU_OP_SWIGLU_OAI:
    case GGML_GLU_OP_GEGLU_ERF:
    case GGML_GLU_OP_GEGLU_QUICK:
    case GGML_GLU_OP_SWIGLU_CLAMP:
      return ggml_is_contiguous_1(pOp->src[0]);
    default:
      return false;
    }
    break;
  case GGML_OP_MUL_MAT:
  case GGML_OP_MUL_MAT_ID: {
    struct ggml_tensor *vA = pOp->src[0];
    struct ggml_tensor *vB = pOp->src[1];
    if (vA->nb[0] != ggml_element_size(vA) || vB->nb[0] != ggml_element_size(vB)) {
      return false; // TODO this could in principle be implemented though currently there is no use case.
    }
    if (vB->type == GGML_TYPE_F16 && vA->type != GGML_TYPE_F16) {
      return false;
    }
    switch (vA->type) {
    case GGML_TYPE_F32:
    case GGML_TYPE_F16:
    case GGML_TYPE_Q1_0:
    case GGML_TYPE_Q2_0:
    case GGML_TYPE_Q4_0:
    case GGML_TYPE_Q4_1:
    case GGML_TYPE_Q5_0:
    case GGML_TYPE_Q5_1:
    case GGML_TYPE_Q8_0:
    case GGML_TYPE_MXFP4:
    case GGML_TYPE_NVFP4:
    case GGML_TYPE_Q2_K:
    case GGML_TYPE_Q3_K:
    case GGML_TYPE_Q4_K:
    case GGML_TYPE_Q5_K:
    case GGML_TYPE_Q6_K:
    case GGML_TYPE_Q8_K:
    case GGML_TYPE_IQ1_M:
    case GGML_TYPE_IQ1_S:
    case GGML_TYPE_IQ2_S:
    case GGML_TYPE_IQ2_XS:
    case GGML_TYPE_IQ2_XXS:
    case GGML_TYPE_IQ3_S:
    case GGML_TYPE_IQ3_XXS:
    case GGML_TYPE_IQ4_NL:
    case GGML_TYPE_IQ4_XS:
    case GGML_TYPE_BF16:
      return true;
    default:
      return false;
    }
  } break;
  case GGML_OP_OUT_PROD:
    return pOp->type == GGML_TYPE_F32 && pOp->src[0]->type == GGML_TYPE_F32 && pOp->src[1]->type == GGML_TYPE_F32;
  case GGML_OP_GET_ROWS: {
    switch (pOp->src[0]->type) {
    case GGML_TYPE_F16:
    case GGML_TYPE_F32:
    case GGML_TYPE_BF16:
    case GGML_TYPE_I32:
    case GGML_TYPE_Q1_0:
    case GGML_TYPE_Q2_0:
    case GGML_TYPE_Q4_0:
    case GGML_TYPE_Q4_1:
    case GGML_TYPE_Q5_0:
    case GGML_TYPE_Q5_1:
    case GGML_TYPE_Q8_0:
    case GGML_TYPE_Q2_K:
    case GGML_TYPE_Q3_K:
    case GGML_TYPE_Q4_K:
    case GGML_TYPE_Q5_K:
    case GGML_TYPE_Q6_K:
    case GGML_TYPE_IQ2_XXS:
    case GGML_TYPE_IQ2_XS:
    case GGML_TYPE_IQ2_S:
    case GGML_TYPE_IQ3_XXS:
    case GGML_TYPE_IQ3_S:
    case GGML_TYPE_IQ1_S:
    case GGML_TYPE_IQ1_M:
    case GGML_TYPE_IQ4_XS:
      return true;
    case GGML_TYPE_IQ4_NL:
    case GGML_TYPE_MXFP4:
      // 32-value sub-blocks, the row size does not guarantee
      // the QK_K super-blocks the get_rows kernel iterates on
      return pOp->src[0]->ne[0] % QK_K == 0;
    default:
      return false;
    }
  } break;
  case GGML_OP_GET_ROWS_BACK: {
    return pOp->type == GGML_TYPE_F32 && pOp->src[0]->type == GGML_TYPE_F32 && pOp->ne[2] == 1 && pOp->ne[3] == 1;
  } break;
  case GGML_OP_SET_ROWS: {
    return (((pOp->type == GGML_TYPE_F32 || pOp->type == GGML_TYPE_F16 || pOp->type == GGML_TYPE_BF16 ||
              pOp->type == GGML_TYPE_Q4_0 || pOp->type == GGML_TYPE_Q4_1 || pOp->type == GGML_TYPE_Q5_0 ||
              pOp->type == GGML_TYPE_Q5_1 || pOp->type == GGML_TYPE_Q8_0 || pOp->type == GGML_TYPE_IQ4_NL) &&
             pOp->src[0]->type == GGML_TYPE_F32) ||
            (pOp->type == GGML_TYPE_F16 && pOp->src[0]->type == GGML_TYPE_F16)) &&
           (pOp->src[1]->type == GGML_TYPE_I64 || pOp->src[1]->type == GGML_TYPE_I32);
  } break;
  case GGML_OP_SET: {
    const ggml_type cT = pOp->type;
    return (cT == GGML_TYPE_F32 || cT == GGML_TYPE_I32) && cT == pOp->src[0]->type && cT == pOp->src[1]->type;
  } break;
  case GGML_OP_CPY: {
    ggml_type vSrc0Type = pOp->src[0]->type;
    ggml_type vSrc1Type = pOp->src[1]->type;
    if ((vSrc0Type == GGML_TYPE_F32 || vSrc0Type == GGML_TYPE_BF16 || vSrc0Type == GGML_TYPE_F16) &&
        (vSrc1Type == GGML_TYPE_F32 || vSrc1Type == GGML_TYPE_BF16 || vSrc1Type == GGML_TYPE_F16)) {
      return true;
    }
    if (vSrc0Type == GGML_TYPE_F32 && vSrc1Type == GGML_TYPE_Q8_0) {
      return true;
    }
    if (vSrc0Type == GGML_TYPE_Q8_0 && vSrc1Type == GGML_TYPE_F32) {
      return true;
    }
    if (vSrc0Type == GGML_TYPE_F32 && vSrc1Type == GGML_TYPE_Q4_0) {
      return true;
    }
    if (vSrc0Type == GGML_TYPE_Q4_0 && vSrc1Type == GGML_TYPE_F32) {
      return true;
    }
    if (vSrc0Type == GGML_TYPE_F32 && vSrc1Type == GGML_TYPE_Q4_1) {
      return true;
    }
    if (vSrc0Type == GGML_TYPE_Q4_1 && vSrc1Type == GGML_TYPE_F32) {
      return true;
    }
    if (vSrc0Type == GGML_TYPE_F32 && vSrc1Type == GGML_TYPE_Q5_0) {
      return true;
    }
    if (vSrc0Type == GGML_TYPE_Q5_0 && vSrc1Type == GGML_TYPE_F32) {
      return true;
    }
    if (vSrc0Type == GGML_TYPE_F32 && vSrc1Type == GGML_TYPE_Q5_1) {
      return true;
    }
    if (vSrc0Type == GGML_TYPE_Q5_1 && vSrc1Type == GGML_TYPE_F32) {
      return true;
    }
    if (vSrc0Type == GGML_TYPE_F32 && vSrc1Type == GGML_TYPE_IQ4_NL) {
      return true;
    }
    if (vSrc0Type == GGML_TYPE_F32 && vSrc1Type == GGML_TYPE_I32) {
      return true;
    }
    if (vSrc0Type == GGML_TYPE_I32 && vSrc1Type == GGML_TYPE_F32) {
      return true;
    }
    if (vSrc0Type == GGML_TYPE_I32 && vSrc1Type == GGML_TYPE_I32) {
      return true;
    }
    if (vSrc0Type == vSrc1Type && ggml_is_contiguous(pOp->src[0]) && ggml_is_contiguous(pOp->src[1])) {
      return true;
    }
    return false;
  } break;
  case GGML_OP_DUP: {
    ggml_type vSrc0Type = pOp->src[0]->type;
    return vSrc0Type != GGML_TYPE_I32 && vSrc0Type != GGML_TYPE_I16;
  } break;
  case GGML_OP_ARGMAX:
  case GGML_OP_COUNT_EQUAL: {
    return true;
  } break;
  case GGML_OP_AMAX_POW2:
    return pOp->src[0]->type == GGML_TYPE_F32 && pOp->src[0]->nb[0] == sizeof(float);
  case GGML_OP_REPEAT: {
    // the CUDA REPEAT path only implements F32/F16; other types assert at runtime
    ggml_type vSrc0Type = pOp->src[0]->type;
    return vSrc0Type == GGML_TYPE_F32 || vSrc0Type == GGML_TYPE_F16;
  } break;
  case GGML_OP_REPEAT_BACK:
    return pOp->type == GGML_TYPE_F32 && (pOp->src[0]->ne[2] * pOp->src[0]->ne[3]) <= (1 << 15);
  case GGML_OP_CONCAT: {
    ggml_type vSrc0Type = pOp->src[0]->type;
    ggml_type vSrc1Type = pOp->src[1]->type;
    const int32_t cDim = pOp->op_params[0];
    return vSrc0Type == vSrc1Type && vSrc0Type == pOp->type &&
           ((ggml_is_quantized(vSrc0Type) &&
             ((cDim == 3 && ggml_is_contiguous(pOp->src[0]) && ggml_is_contiguous(pOp->src[1])) ||
              (cDim != 3 && ggml_is_contiguous_to_3(pOp->src[0]) && ggml_is_contiguous_to_3(pOp->src[1]))) &&
             pOp->src[0]->ne[0] % ggml_blck_size(vSrc0Type) == 0 &&
             pOp->src[1]->ne[0] % ggml_blck_size(vSrc0Type) == 0) ||
            (!ggml_is_quantized(vSrc0Type) && ggml_blck_size(vSrc0Type) == 1 &&
             (ggml_type_size(vSrc0Type) == 1 || ggml_type_size(vSrc0Type) == 2 || ggml_type_size(vSrc0Type) == 4 ||
              ggml_type_size(vSrc0Type) == 8)));
  } break;
  case GGML_OP_CONV_TRANSPOSE_1D: {
    ggml_type vSrc0Type = pOp->src[0]->type;
    ggml_type vSrc1Type = pOp->src[1]->type;
    if (vSrc0Type == GGML_TYPE_F32 && vSrc1Type == GGML_TYPE_F32) {
      return true;
    }
    return false;
  } break;
  case GGML_OP_COL2IM_1D: {
    ggml_type vSrc0Type = pOp->src[0]->type;
    return (vSrc0Type == GGML_TYPE_F32 || vSrc0Type == GGML_TYPE_F16 || vSrc0Type == GGML_TYPE_BF16) &&
           pOp->type == vSrc0Type && ggml_is_contiguous(pOp->src[0]) && ggml_is_contiguous(pOp);
  } break;
  case GGML_OP_SILU_BACK:
    return ggml_is_contiguous(pOp->src[0]) && pOp->src[0]->type == GGML_TYPE_F32;
    break;
  case GGML_OP_NORM:
  case GGML_OP_RMS_NORM:
  case GGML_OP_L2_NORM:
    return ggml_is_contiguous_rows(pOp->src[0]);
  case GGML_OP_RMS_NORM_BACK:
    return ggml_is_contiguous(pOp->src[0]);
    break;
  case GGML_OP_NONE:
  case GGML_OP_RESHAPE:
  case GGML_OP_VIEW:
  case GGML_OP_PERMUTE:
  case GGML_OP_TRANSPOSE:
  case GGML_OP_ADD_ID:
  case GGML_OP_ADD1:
  case GGML_OP_SCALE:
  case GGML_OP_SQR:
  case GGML_OP_SQRT:
  case GGML_OP_SIN:
  case GGML_OP_COS:
  case GGML_OP_CLAMP:
  case GGML_OP_LOG:
    return true;
  case GGML_OP_ADD:
  case GGML_OP_SUB:
  case GGML_OP_MUL:
  case GGML_OP_DIV:
    return (pOp->src[0]->type == GGML_TYPE_F32 || pOp->src[0]->type == GGML_TYPE_F16) &&
           (pOp->src[1]->type == GGML_TYPE_F32 || pOp->src[1]->type == GGML_TYPE_F16) &&
           (pOp->type == GGML_TYPE_F32 || pOp->type == GGML_TYPE_F16);
  case GGML_OP_SSM_SCAN: {
    if (pOp->src[3]->ne[0] == 1) {
      // Mamba2
      // (kernel only supports (d_state == 128 || d_state == 256) && d_head % 16 == 0)
      return (pOp->src[0]->ne[0] == 128 || pOp->src[0]->ne[0] == 256) && pOp->src[0]->ne[1] % 16 == 0;
    } else {
      // Mamba
      // (kernel only supports d_state == 16, d_head == 1, n_head % 128 == 0, n_group == 1)
      return pOp->src[0]->ne[0] == 16 && pOp->src[0]->ne[1] == 1 && pOp->src[0]->ne[2] % 128 == 0 &&
             pOp->src[4]->ne[1] == 1;
    }
  }
  case GGML_OP_SSM_CONV: {
    // assumes d_inner % threads == 0
    return pOp->src[0]->ne[1] % 128 == 0;
  }
  case GGML_OP_CONT:
    return true;
  case GGML_OP_DIAG_MASK_INF:
    return true;
  case GGML_OP_SOFT_MAX:
    return true;
  case GGML_OP_SOFT_MAX_BACK: {
    float vMaxBias = 0.0f;
    memcpy(&vMaxBias, (const float *)pOp->op_params + 1, sizeof(float));
    return vMaxBias == 0.0f;
  }
  case GGML_OP_ROLL:
    if (pOp->src[0]->type == GGML_TYPE_F32) {
      return true;
    }
    return false;
  case GGML_OP_ROPE:
  case GGML_OP_ROPE_BACK: {
    return pOp->src[0]->nb[0] == ggml_type_size(pOp->src[0]->type) && ggml_is_contiguous_2(pOp->src[0]);
  }
  case GGML_OP_IM2COL:
  case GGML_OP_IM2COL_3D:
  case GGML_OP_CONV_2D:
    return (ggml_is_contiguous(pOp->src[0]) && ggml_is_contiguous(pOp->src[1]));
  case GGML_OP_CONV_2D_DW:
    return pOp->src[0]->type == GGML_TYPE_F32;
  case GGML_OP_CONV_TRANSPOSE_2D:
  case GGML_OP_POOL_2D:
    return true;
  case GGML_OP_ACC:
    // TODO: extend support like so:
    // return ggml_is_contiguous_rows(op->src[0]) && ggml_is_contiguous_rows(op->src[1]);
    return ggml_is_contiguous(pOp->src[0]) && ggml_is_contiguous(pOp->src[1]);
  case GGML_OP_SUM:
    return ggml_is_contiguous_rows(pOp->src[0]);
  case GGML_OP_TOP_K:
  case GGML_OP_ARGSORT:
#ifndef GGML_CUDA_USE_CUB
    return pOp->src[0]->ne[0] <= 1024;
#else
    return true;
#endif
  case GGML_OP_SUM_ROWS:
  case GGML_OP_MEAN:
  case GGML_OP_GROUP_NORM:
    return ggml_is_contiguous(pOp->src[0]);
  case GGML_OP_PAD:
    return true;
  case GGML_OP_UPSCALE:
  case GGML_OP_PAD_REFLECT_1D:
  case GGML_OP_ARANGE:
  case GGML_OP_TIMESTEP_EMBEDDING:
  case GGML_OP_LEAKY_RELU:
  case GGML_OP_RWKV_WKV6:
  case GGML_OP_GATED_LINEAR_ATTN:
  case GGML_OP_RWKV_WKV7:
  case GGML_OP_GATED_DELTA_NET:
    return true;
  case GGML_OP_DSV4_HC_COMB:
    return pOp->src[0]->type == GGML_TYPE_F32 && pOp->src[1]->type == GGML_TYPE_F32 &&
           pOp->src[2]->type == GGML_TYPE_F32 && pOp->type == GGML_TYPE_F32;
  case GGML_OP_DSV4_HC_PRE:
    return pOp->src[0]->type == GGML_TYPE_F32 && pOp->src[1]->type == GGML_TYPE_F32 && pOp->type == GGML_TYPE_F32;
  case GGML_OP_DSV4_HC_POST:
    return pOp->src[0]->type == GGML_TYPE_F32 && pOp->src[1]->type == GGML_TYPE_F32 &&
           pOp->src[2]->type == GGML_TYPE_F32 && (pOp->src[3] == nullptr || pOp->src[3]->type == GGML_TYPE_F32) &&
           pOp->type == GGML_TYPE_F32;
  case GGML_OP_FLASH_ATTN_EXT:
    return ggml_cuda_flash_attn_ext_supported(vDevCtx->device, pOp);
  case GGML_OP_CROSS_ENTROPY_LOSS:
  case GGML_OP_CROSS_ENTROPY_LOSS_BACK:
  case GGML_OP_OPT_STEP_ADAMW:
  case GGML_OP_OPT_STEP_SGD:
  case GGML_OP_FILL:
  case GGML_OP_CUMSUM:
  case GGML_OP_TRI:
  case GGML_OP_DIAG:
  case GGML_OP_SOLVE_TRI:
    return true;
  case GGML_OP_LIGHTNING_INDEXER:
    return ggml_cuda_lightning_indexer_supported(vDevCtx->device, pOp);

  default:
    return false;
  }
}

static bool fGgmlBackendCudaDeviceSupportsBuft(ggml_backend_dev_t pDev, ggml_backend_buffer_type_t pBuft) {
  ggml_backend_cuda_device_context *vDevCtx = (ggml_backend_cuda_device_context *)pDev->context;
  const bool cIntegrated = ggml_cuda_info().devices[vDevCtx->device].integrated;
  return (fGgmlBackendBuftIsCuda(pBuft) && pBuft->device == pDev) || (cIntegrated && fGgmlBackendBuftIsCudaHost(pBuft));
}

static int64_t fGetOpBatchSize(const ggml_tensor *pOp) {
  switch (pOp->op) {
  case GGML_OP_GET_ROWS:
    return 0;
  case GGML_OP_MUL_MAT:
    return pOp->ne[1];
  case GGML_OP_MUL_MAT_ID:
  case GGML_OP_ROPE:
  case GGML_OP_ROPE_BACK:
    return pOp->ne[2];
  default:
    return ggml_nrows(pOp);
  }
}

static bool fGgmlBackendCudaDeviceOffloadOp(ggml_backend_dev_t pDev, const ggml_tensor *pOp) {
  ggml_backend_cuda_device_context *vDevCtx = (ggml_backend_cuda_device_context *)pDev->context;

  return fGetOpBatchSize(pOp) >= vDevCtx->vOpOffloadMinBatchSize;
}

static ggml_backend_event_t fGgmlBackendCudaDeviceEventNew(ggml_backend_dev_t pDev) {
#ifdef GGML_CUDA_NO_PEER_COPY
  return nullptr;
#else
  ggml_backend_cuda_device_context *vDevCtx = (ggml_backend_cuda_device_context *)pDev->context;

  ggml_cuda_set_device(vDevCtx->device);

  cudaEvent_t vEvent;
  CUDA_CHECK(cudaEventCreateWithFlags(&vEvent, cudaEventDisableTiming));

  return new ggml_backend_event{
      /* .device  = */ pDev,
      /* .context = */ vEvent,
  };
#endif
}

static void fGgmlBackendCudaDeviceEventFree(ggml_backend_dev_t pDev, ggml_backend_event_t pEvent) {
  GGML_UNUSED(pDev);

  CUDA_CHECK(cudaEventDestroy((cudaEvent_t)pEvent->context));
  delete pEvent;
}

static void fGgmlBackendCudaDeviceEventSynchronize(ggml_backend_dev_t pDev, ggml_backend_event_t pEvent) {
  GGML_UNUSED(pDev);
  CUDA_CHECK(cudaEventSynchronize((cudaEvent_t)pEvent->context));
}

static const ggml_backend_device_i ggml_backend_cuda_device_interface = {
    /* .get_name                = */ fGgmlBackendCudaDeviceGetName,
    /* .get_description         = */ fGgmlBackendCudaDeviceGetDescription,
    /* .get_memory              = */ fGgmlBackendCudaDeviceGetMemory,
    /* .get_type                = */ fGgmlBackendCudaDeviceGetType,
    /* .get_props               = */ fGgmlBackendCudaDeviceGetProps,
    /* .init_backend            = */ fGgmlBackendCudaDeviceInitBackend,
    /* .get_buffer_type         = */ fGgmlBackendCudaDeviceGetBufferType,
    /* .get_host_buffer_type    = */ fGgmlBackendCudaDeviceGetHostBufferType,
    /* .buffer_from_host_ptr    = */ NULL,
    /* .supports_op             = */ fGgmlBackendCudaDeviceSupportsOp,
    /* .supports_buft           = */ fGgmlBackendCudaDeviceSupportsBuft,
    /* .offload_op              = */ fGgmlBackendCudaDeviceOffloadOp,
    /* .event_new               = */ fGgmlBackendCudaDeviceEventNew,
    /* .event_free              = */ fGgmlBackendCudaDeviceEventFree,
    /* .event_synchronize       = */ fGgmlBackendCudaDeviceEventSynchronize,
};

// backend reg

struct ggml_backend_cuda_reg_context {
  std::vector<ggml_backend_dev_t> devices;
};

static const char *fGgmlBackendCudaRegGetName(ggml_backend_reg_t pReg) {
  GGML_UNUSED(pReg);
  return GGML_CUDA_NAME;
}

static size_t fGgmlBackendCudaRegGetDeviceCount(ggml_backend_reg_t pReg) {
  ggml_backend_cuda_reg_context *vCtx = (ggml_backend_cuda_reg_context *)pReg->context;
  return vCtx->devices.size();
}

static ggml_backend_dev_t fGgmlBackendCudaRegGetDevice(ggml_backend_reg_t pReg, size_t pIndex) {
  ggml_backend_cuda_reg_context *vCtx = (ggml_backend_cuda_reg_context *)pReg->context;
  GGML_ASSERT(pIndex < vCtx->devices.size());
  return vCtx->devices[pIndex];
}

static ggml_backend_feature *fGgmlBackendCudaGetFeatures(ggml_backend_reg_t pReg) {
  static std::vector<ggml_backend_feature> vFeatures = []() {
    std::vector<ggml_backend_feature> vFeatures;
#define _STRINGIFY(...) #__VA_ARGS__
#define STRINGIFY(...) _STRINGIFY(__VA_ARGS__)

#ifdef __CUDA_ARCH_LIST__
    vFeatures.push_back({"ARCHS", STRINGIFY(__CUDA_ARCH_LIST__)});
#endif

#ifdef GGML_CUDA_FORCE_MMQ
    vFeatures.push_back({"FORCE_MMQ", "1"});
#endif

#ifdef GGML_CUDA_FORCE_CUBLAS
    vFeatures.push_back({"FORCE_CUBLAS", "1"});
#endif

#ifndef GGML_USE_VMM
    vFeatures.push_back({"NO_VMM", "1"});
#endif

#ifdef GGML_CUDA_NO_PEER_COPY
    vFeatures.push_back({"NO_PEER_COPY", "1"});
#endif

#ifdef GGML_CUDA_USE_GRAPHS
    vFeatures.push_back({"USE_GRAPHS", "1"});
#endif

#ifdef GGML_CUDA_FA_QUANTS
    vFeatures.push_back({"FA_QUANTS", GGML_CUDA_FA_QUANTS});
#endif

    {
      const auto &vInfo = ggml_cuda_info();
      for (int vId = 0; vId < vInfo.device_count; ++vId) {
        if (blackwell_mma_available(vInfo.devices[vId].cc)) {
          vFeatures.push_back({"BLACKWELL_NATIVE_FP4", "1"});
          break;
        }
      }
    }

#undef _STRINGIFY
#undef STRINGIFY

    vFeatures.push_back({nullptr, nullptr});

    return vFeatures;
  }();

  return vFeatures.data();

  GGML_UNUSED(pReg);
}

static void *fGgmlBackendCudaRegGetProcAddress(ggml_backend_reg_t pReg, const char *pName) {
  GGML_UNUSED(pReg);
  if (strcmp(pName, "ggml_backend_comm_init") == 0) {
    return (void *)fGgmlBackendCudaCommInit;
  }
  if (strcmp(pName, "ggml_backend_comm_free") == 0) {
    return (void *)fGgmlBackendCudaCommFree;
  }
  if (strcmp(pName, "ggml_backend_comm_allreduce_tensor") == 0) {
    return (void *)fGgmlBackendCudaCommAllreduceTensor;
  }
  if (strcmp(pName, "ggml_backend_register_host_buffer") == 0) {
    return (void *)ggml_backend_cuda_register_host_buffer;
  }
  if (strcmp(pName, "ggml_backend_unregister_host_buffer") == 0) {
    return (void *)ggml_backend_cuda_unregister_host_buffer;
  }
  if (strcmp(pName, "ggml_backend_get_features") == 0) {
    return (void *)fGgmlBackendCudaGetFeatures;
  }
  return nullptr;
}

static const ggml_backend_reg_i ggml_backend_cuda_reg_interface = {
    /* .get_name          = */ fGgmlBackendCudaRegGetName,
    /* .get_device_count  = */ fGgmlBackendCudaRegGetDeviceCount,
    /* .get_device        = */ fGgmlBackendCudaRegGetDevice,
    /* .get_proc_address  = */ fGgmlBackendCudaRegGetProcAddress,
};

// backend registry
ggml_backend_reg_t ggml_backend_cuda_reg() {
  static ggml_backend_reg vReg;
  static bool vInitialized = false;

  {
    static std::mutex vMutex;
    std::lock_guard<std::mutex> vLock(vMutex);
    if (!vInitialized) {
      ggml_backend_cuda_reg_context *vCtx = new ggml_backend_cuda_reg_context;
      const int cMinBatchSize = getenv("GGML_OP_OFFLOAD_MIN_BATCH") ? atoi(getenv("GGML_OP_OFFLOAD_MIN_BATCH")) : 32;

      const ggml_cuda_device_info &vInfo = ggml_cuda_info();
      const bool cVirtualDevices = vInfo.device_count > vInfo.physical_device_count;

      for (int vI = 0; vI < vInfo.device_count; vI++) {
        const int cPhysicalId = vInfo.devices[vI].physical_device;

        ggml_backend_cuda_device_context *vDevCtx = new ggml_backend_cuda_device_context;
        vDevCtx->device = vI;
        vDevCtx->name = GGML_CUDA_NAME + std::to_string(vI);
        vDevCtx->vDescription = fGgmlCudaDeviceDescription(vI);

        char vPciBusId[32] = {};
        CUDA_CHECK(cudaDeviceGetPCIBusId(vPciBusId, sizeof(vPciBusId), cPhysicalId));
        vDevCtx->vPciBusId = vPciBusId;
        if (cVirtualDevices) {
          // make the pci bus id unique for virtual devices
          vDevCtx->vPciBusId += "-v" + std::to_string(vI);
        }
        for (char &vC : vDevCtx->vPciBusId) {
          vC = std::tolower(vC);
        }
        vDevCtx->vOpOffloadMinBatchSize = cMinBatchSize;

        ggml_backend_dev_t vDev = new ggml_backend_device{/* .iface   = */ ggml_backend_cuda_device_interface,
                                                          /* .reg     = */ &vReg,
                                                          /* .context = */ vDevCtx};
        vCtx->devices.push_back(vDev);
      }

      vReg = ggml_backend_reg{/* .api_version = */ GGML_BACKEND_API_VERSION,
                              /* .iface       = */ ggml_backend_cuda_reg_interface,
                              /* .context     = */ vCtx};
    }

    vInitialized = true;
  }

  return &vReg;
}

ggml_backend_t ggml_backend_cuda_init(int pDevice) {
  if (pDevice < 0 || pDevice >= ggml_backend_cuda_get_device_count()) {
    GGML_LOG_ERROR("%s: invalid device %d\n", __func__, pDevice);
    return nullptr;
  }

  ggml_backend_cuda_context *vCtx = new ggml_backend_cuda_context(pDevice);
  if (vCtx == nullptr) {
    GGML_LOG_ERROR("%s: failed to allocate context\n", __func__);
    return nullptr;
  }

  ggml_backend_t vCudaBackend = new ggml_backend{
      /* .guid    = */ fGgmlBackendCudaGuid(),
      /* .iface   = */ ggml_backend_cuda_interface,
      /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_cuda_reg(), pDevice),
      /* .context = */ vCtx,
  };

  return vCudaBackend;
}

GGML_BACKEND_DL_IMPL(ggml_backend_cuda_reg)
