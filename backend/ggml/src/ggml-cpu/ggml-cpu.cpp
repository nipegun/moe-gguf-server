#include "ggml-backend.h"
#include "ggml-backend-impl.h"
#include "ggml-cpu.h"
#include "repack.h"
#include "traits.h"
#include "ggml-impl.h"
#include "amx/amx.h"

#include <cctype>
#include <string>
#include <vector>

#ifdef GGML_USE_CPU_HBM
#include "hbm.h"
#endif

#ifdef GGML_USE_CPU_KLEIDIAI
#include "kleidiai/kleidiai.h"
#endif

#ifdef GGML_USE_CPU_RISCV64_SPACEMIT
#include "spacemit/ime.h"
#endif

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <unistd.h>
#endif

#if defined(__APPLE__)
#include <sys/sysctl.h>
#include <sys/types.h>
#endif

// ggml-backend interface

std::vector<ggml_backend_buffer_type_t> &ggml_backend_cpu_get_extra_buffer_types() {
  static std::vector<ggml_backend_buffer_type_t> vBufts = []() {
    std::vector<ggml_backend_buffer_type_t> vBufts;

#if defined(__AMX_INT8__) && defined(__AVX512VNNI__)
    if (ggml_backend_amx_buffer_type()) {
      vBufts.push_back(ggml_backend_amx_buffer_type());
    }
#endif

#ifdef GGML_USE_CPU_RISCV64_SPACEMIT
    if (ggml_backend_cpu_riscv64_spacemit_buffer_type()) {
      vBufts.push_back(ggml_backend_cpu_riscv64_spacemit_buffer_type());
    }
#endif

#ifdef GGML_USE_CPU_KLEIDIAI
    if (ggml_backend_cpu_kleidiai_buffer_type()) {
      vBufts.push_back(ggml_backend_cpu_kleidiai_buffer_type());
    }
#endif

#ifdef GGML_USE_CPU_REPACK
    if (ggml_backend_cpu_repack_buffer_type()) {
      vBufts.push_back(ggml_backend_cpu_repack_buffer_type());
    }
#endif

    return vBufts;
  }();

  return vBufts;
}

static ggml_backend_buffer_type_t *fGgmlBackendCpuDeviceGetExtraBuffersType(ggml_backend_dev_t pDevice) {
  static std::vector<ggml_backend_buffer_type_t> lExtraBufts = [] {
    std::vector<ggml_backend_buffer_type_t> vBufts = ggml_backend_cpu_get_extra_buffer_types();
    vBufts.push_back(nullptr);
    return vBufts;
  }();

  return lExtraBufts.data();

  GGML_UNUSED(pDevice);
}

static bool fGgmlBackendCpuIsExtraBufferType(ggml_backend_buffer_type_t pBuft) {
  for (auto *vExtra : ggml_backend_cpu_get_extra_buffer_types()) {
    if (vExtra == pBuft) {
      return true;
    }
  }
  return false;
}

// CPU backend - backend (stream)

struct ggml_backend_cpu_context {
  int vNThreads;
  ggml_threadpool_t vThreadpool;

  uint8_t *vWorkData;
  size_t vWorkSize;

  ggml_abort_callback vAbortCallback;
  void *vAbortCallbackData;

  bool vUseRef; // use reference implementation
};

static const char *fGgmlBackendCpuGetName(ggml_backend_t pBackend) {
  return "CPU";

  GGML_UNUSED(pBackend);
}

static void fGgmlBackendCpuFree(ggml_backend_t pBackend) {
  struct ggml_backend_cpu_context *vCpuCtx = (struct ggml_backend_cpu_context *)pBackend->context;
  delete[] vCpuCtx->vWorkData;
  delete vCpuCtx;
  delete pBackend;
}

struct ggml_backend_plan_cpu {
  struct ggml_cplan vCplan;
  struct ggml_cgraph vCgraph;
};

static ggml_backend_graph_plan_t fGgmlBackendCpuGraphPlanCreate(ggml_backend_t pBackend,
                                                                const struct ggml_cgraph *pCgraph) {
  struct ggml_backend_cpu_context *vCpuCtx = (struct ggml_backend_cpu_context *)pBackend->context;

  struct ggml_backend_plan_cpu *vCpuPlan = new ggml_backend_plan_cpu;

  vCpuPlan->vCplan = ggml_graph_plan(pCgraph, vCpuCtx->vNThreads, vCpuCtx->vThreadpool);
  vCpuPlan->vCgraph = *pCgraph; // FIXME: deep copy

  if (vCpuPlan->vCplan.work_size > 0) {
    vCpuPlan->vCplan.work_data = new uint8_t[vCpuPlan->vCplan.work_size];
    if (vCpuPlan->vCplan.work_data == NULL) {
      delete vCpuPlan;
      return NULL;
    }
  }

  vCpuPlan->vCplan.abort_callback = vCpuCtx->vAbortCallback;
  vCpuPlan->vCplan.abort_callback_data = vCpuCtx->vAbortCallbackData;
  vCpuPlan->vCplan.use_ref = vCpuCtx->vUseRef;

  return vCpuPlan;
}

static void fGgmlBackendCpuGraphPlanFree(ggml_backend_t pBackend, ggml_backend_graph_plan_t pPlan) {
  struct ggml_backend_plan_cpu *vCpuPlan = (struct ggml_backend_plan_cpu *)pPlan;

  delete[] vCpuPlan->vCplan.work_data;
  delete vCpuPlan;

  GGML_UNUSED(pBackend);
}

static enum ggml_status fGgmlBackendCpuGraphPlanCompute(ggml_backend_t pBackend, ggml_backend_graph_plan_t pPlan) {
  struct ggml_backend_plan_cpu *vCpuPlan = (struct ggml_backend_plan_cpu *)pPlan;

  return ggml_graph_compute(&vCpuPlan->vCgraph, &vCpuPlan->vCplan);

  GGML_UNUSED(pBackend);
}

static enum ggml_status fGgmlBackendCpuGraphCompute(ggml_backend_t pBackend, struct ggml_cgraph *pCgraph) {
  struct ggml_backend_cpu_context *vCpuCtx = (struct ggml_backend_cpu_context *)pBackend->context;

  struct ggml_cplan vCplan = ggml_graph_plan(pCgraph, vCpuCtx->vNThreads, vCpuCtx->vThreadpool);

  if (vCpuCtx->vWorkSize < vCplan.work_size) {
    delete[] vCpuCtx->vWorkData;
    vCpuCtx->vWorkData = new uint8_t[vCplan.work_size];
    if (vCpuCtx->vWorkData == NULL) {
      vCpuCtx->vWorkSize = 0;
      return GGML_STATUS_ALLOC_FAILED;
    }
    vCpuCtx->vWorkSize = vCplan.work_size;
  }
  vCplan.work_data = (uint8_t *)vCpuCtx->vWorkData;

  vCplan.abort_callback = vCpuCtx->vAbortCallback;
  vCplan.abort_callback_data = vCpuCtx->vAbortCallbackData;
  vCplan.use_ref = vCpuCtx->vUseRef;

  return ggml_graph_compute(pCgraph, &vCplan);
}

static const struct ggml_backend_i ggml_backend_cpu_i = {
    /* .get_name                = */ fGgmlBackendCpuGetName,
    /* .free                    = */ fGgmlBackendCpuFree,
    /* .set_tensor_async        = */ NULL,
    /* .get_tensor_async        = */ NULL,
    /* .set_tensor_2d_async     = */ NULL,
    /* .get_tensor_2d_async     = */ NULL,
    /* .cpy_tensor_async        = */ NULL,
    /* .synchronize             = */ NULL,
    /* .graph_plan_create       = */ fGgmlBackendCpuGraphPlanCreate,
    /* .graph_plan_free         = */ fGgmlBackendCpuGraphPlanFree,
    /* .graph_plan_update       = */ NULL,
    /* .graph_plan_compute      = */ fGgmlBackendCpuGraphPlanCompute,
    /* .graph_compute           = */ fGgmlBackendCpuGraphCompute,
    /* .event_record            = */ NULL,
    /* .event_wait              = */ NULL,
    /* .graph_optimize          = */ NULL,
};

static ggml_guid_t fGgmlBackendCpuGuid(void) {
  static ggml_guid aGuid = {0xaa, 0x67, 0xc7, 0x43, 0x96, 0xe6, 0xa3, 0x8a,
                            0xe3, 0xaf, 0xea, 0x92, 0x36, 0xbc, 0xfc, 0x89};
  return &aGuid;
}

ggml_backend_t ggml_backend_cpu_init(void) {
  // initialize CPU backend now to avoid slowing the first graph computation
  ggml_cpu_init();

  struct ggml_backend_cpu_context *vCtx = new ggml_backend_cpu_context;
  if (vCtx == NULL) {
    return NULL;
  }

  vCtx->vNThreads = GGML_DEFAULT_N_THREADS;
  vCtx->vThreadpool = NULL;
  vCtx->vWorkData = NULL;
  vCtx->vWorkSize = 0;
  vCtx->vAbortCallback = NULL;
  vCtx->vAbortCallbackData = NULL;
  vCtx->vUseRef = false;

  ggml_backend_t vCpuBackend = new ggml_backend{
      /* .guid    = */ fGgmlBackendCpuGuid(),
      /* .iface   = */ ggml_backend_cpu_i,
      /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0),
      /* .context = */ vCtx,
  };

  if (vCpuBackend == NULL) {
    delete vCtx;
    return NULL;
  }

  return vCpuBackend;
}

bool ggml_backend_is_cpu(ggml_backend_t pBackend) {
  return pBackend != NULL && ggml_guid_matches(pBackend->guid, fGgmlBackendCpuGuid());
}

void ggml_backend_cpu_set_n_threads(ggml_backend_t pBackendCpu, int pNThreads) {
  GGML_ASSERT(ggml_backend_is_cpu(pBackendCpu));

  struct ggml_backend_cpu_context *vCtx = (struct ggml_backend_cpu_context *)pBackendCpu->context;
  vCtx->vNThreads = pNThreads;
}

void ggml_backend_cpu_set_threadpool(ggml_backend_t pBackendCpu, ggml_threadpool_t pThreadpool) {
  GGML_ASSERT(ggml_backend_is_cpu(pBackendCpu));

  struct ggml_backend_cpu_context *vCtx = (struct ggml_backend_cpu_context *)pBackendCpu->context;

  if (vCtx->vThreadpool && vCtx->vThreadpool != pThreadpool) {
    // already had a different threadpool, pause/suspend it before switching
    ggml_threadpool_pause(vCtx->vThreadpool);
  }
  vCtx->vThreadpool = pThreadpool;
}

void ggml_backend_cpu_set_abort_callback(ggml_backend_t pBackendCpu, ggml_abort_callback pAbortCallback,
                                         void *pAbortCallbackData) {
  GGML_ASSERT(ggml_backend_is_cpu(pBackendCpu));

  struct ggml_backend_cpu_context *vCtx = (struct ggml_backend_cpu_context *)pBackendCpu->context;
  vCtx->vAbortCallback = pAbortCallback;
  vCtx->vAbortCallbackData = pAbortCallbackData;
}

void ggml_backend_cpu_set_use_ref(ggml_backend_t pBackendCpu, bool pUseRef) {
  GGML_ASSERT(ggml_backend_is_cpu(pBackendCpu));

  struct ggml_backend_cpu_context *vCtx = (struct ggml_backend_cpu_context *)pBackendCpu->context;
  vCtx->vUseRef = pUseRef;
}

// CPU backend - device

struct ggml_backend_cpu_device_context {
  std::string description = "CPU";

  ggml_backend_cpu_device_context() {
#ifdef __APPLE__
    size_t len = 0;
    if (!sysctlbyname("machdep.cpu.brand_string", NULL, &len, NULL, 0)) {
      description.resize(len);
      sysctlbyname("machdep.cpu.brand_string", &description[0], &len, NULL, 0); // NOLINT
    }
#elif defined(__linux__)
    FILE *vF = fopen("/proc/cpuinfo", "r");
    if (vF) {
      char aBuf[1024];
      while (fgets(aBuf, sizeof(aBuf), vF)) {
        if (strncmp(aBuf, "model name", 10) == 0) {
          char *vP = strchr(aBuf, ':');
          if (vP) {
            vP++;
            while (std::isspace(*vP)) {
              vP++;
            }
            while (std::isspace(vP[strlen(vP) - 1])) {
              vP[strlen(vP) - 1] = '\0';
            }
            description = vP;
            break;
          }
        }
      }
      fclose(vF);
    }
#elif defined(_WIN32)
    HKEY hKey;
    if (RegOpenKeyEx(HKEY_LOCAL_MACHINE, TEXT("HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0"), 0, KEY_READ,
                     &hKey) == ERROR_SUCCESS) {
      DWORD cpu_brand_size = 0;
      if (RegQueryValueExA(hKey, "ProcessorNameString", NULL, NULL, NULL, &cpu_brand_size) == ERROR_SUCCESS) {
        description.resize(cpu_brand_size);
        if (RegQueryValueExA(hKey, "ProcessorNameString", NULL, NULL,
                             (LPBYTE)&description[0], // NOLINT
                             &cpu_brand_size) == ERROR_SUCCESS) {
          if (description.find('\0') != std::string::npos) {
            description.resize(description.find('\0'));
          }
        }
      }
      RegCloseKey(hKey);
    }
#endif
  }
};

static const char *fGgmlBackendCpuDeviceGetName(ggml_backend_dev_t pDev) {
  return "CPU";

  GGML_UNUSED(pDev);
}

static const char *fGgmlBackendCpuDeviceGetDescription(ggml_backend_dev_t pDev) {
  struct ggml_backend_cpu_device_context *vCtx = (struct ggml_backend_cpu_device_context *)pDev->context;

  return vCtx->description.c_str();
}

static void fGgmlBackendCpuDeviceGetMemory(ggml_backend_dev_t pDev, size_t *pFree, size_t *pTotal) {
#ifdef _WIN32
  MEMORYSTATUSEX status;
  status.dwLength = sizeof(status);
  GlobalMemoryStatusEx(&status);
  *pTotal = status.ullTotalPhys;
  *pFree = status.ullAvailPhys;
#else
  long vPages = sysconf(_SC_PHYS_PAGES);
  long vPageSize = sysconf(_SC_PAGE_SIZE);
  *pTotal = vPages * vPageSize;

  // "free" system memory is ill-defined, for practical purposes assume that all of it is free:
  *pFree = *pTotal;
#endif // _WIN32

  GGML_UNUSED(pDev);
}

static enum ggml_backend_dev_type fGgmlBackendCpuDeviceGetType(ggml_backend_dev_t pDev) {
  return GGML_BACKEND_DEVICE_TYPE_CPU;

  GGML_UNUSED(pDev);
}

static void fGgmlBackendCpuDeviceGetProps(ggml_backend_dev_t pDev, struct ggml_backend_dev_props *pProps) {
  pProps->name = fGgmlBackendCpuDeviceGetName(pDev);
  pProps->description = fGgmlBackendCpuDeviceGetDescription(pDev);
  pProps->type = fGgmlBackendCpuDeviceGetType(pDev);
  fGgmlBackendCpuDeviceGetMemory(pDev, &pProps->memory_free, &pProps->memory_total);
  pProps->caps = {
      /* .async                 = */ false,
      /* .host_buffer           = */ false,
      /* .buffer_from_host_ptr  = */ true,
      /* .events                = */ false,
  };
}

static ggml_backend_t fGgmlBackendCpuDeviceInitBackend(ggml_backend_dev_t pDev, const char *pParams) {
  return ggml_backend_cpu_init();

  GGML_UNUSED(pDev);
  GGML_UNUSED(pParams);
}

static ggml_backend_buffer_type_t fGgmlBackendCpuDeviceGetBufferType(ggml_backend_dev_t pDev) {
  return ggml_backend_cpu_buffer_type();

  GGML_UNUSED(pDev);
}

static ggml_backend_buffer_t fGgmlBackendCpuDeviceBufferFromHostPtr(ggml_backend_dev_t pDev, void *pPtr, size_t pSize,
                                                                    size_t pMaxTensorSize) {
  return ggml_backend_cpu_buffer_from_ptr(pPtr, pSize);

  GGML_UNUSED(pDev);
  GGML_UNUSED(pMaxTensorSize);
}

static bool fGgmlBackendCpuDeviceSupportsOp(ggml_backend_dev_t pDev, const struct ggml_tensor *pOp) {
  const struct ggml_tensor *vSrc0 = pOp->src[0];
  const struct ggml_tensor *vSrc1 = pOp->src[1];

  if (pOp->op == GGML_OP_NONE || pOp->op == GGML_OP_RESHAPE || pOp->op == GGML_OP_VIEW || pOp->op == GGML_OP_PERMUTE ||
      pOp->op == GGML_OP_TRANSPOSE) {
    return true;
  }

  // check extra buffer types
  // note: only the first sources are checked for extra buffer types to reduce overhead, increase if necessary
  for (int vI = 0; vI < 4; vI++) {
    if (pOp->src[vI] && pOp->src[vI]->buffer && fGgmlBackendCpuIsExtraBufferType(pOp->src[vI]->buffer->buft)) {
      auto *vBufExtra = (ggml::cpu::extra_buffer_type *)pOp->src[vI]->buffer->buft->context;
      return vBufExtra->supports_op(pDev, pOp);
    }
  }

  switch (pOp->op) {
  case GGML_OP_CPY:
  case GGML_OP_SET_ROWS:
    return pOp->type != GGML_TYPE_IQ3_XXS && pOp->type != GGML_TYPE_IQ3_S && pOp->type != GGML_TYPE_IQ2_XXS &&
           pOp->type != GGML_TYPE_IQ2_XS && pOp->type != GGML_TYPE_IQ2_S && pOp->type != GGML_TYPE_IQ1_S &&
           pOp->type != GGML_TYPE_IQ1_M; // missing type_traits.from_float
  case GGML_OP_MUL_MAT:
    return vSrc1->type == GGML_TYPE_F32 || vSrc1->type == ggml_get_type_traits_cpu(vSrc0->type)->vec_dot_type;
  case GGML_OP_SOFT_MAX_BACK: {
    if (pOp->src[0]->type != GGML_TYPE_F32 || pOp->src[1]->type != GGML_TYPE_F32) {
      return false;
    }
    float vMaxBias = 0.0f;

    memcpy(&vMaxBias, (const float *)pOp->op_params + 1, sizeof(float));

    return vMaxBias == 0.0f;
  }
  case GGML_OP_IM2COL_BACK:
    return vSrc0->type == GGML_TYPE_F32 && (vSrc1->type == GGML_TYPE_F32 || vSrc1->type == GGML_TYPE_F16);
  case GGML_OP_GET_ROWS_BACK:
    return vSrc0->type == GGML_TYPE_F32 || vSrc0->type == GGML_TYPE_F16;
  case GGML_OP_OUT_PROD:
    return (vSrc0->type == GGML_TYPE_F32 || ((vSrc0->type == GGML_TYPE_F16 || ggml_is_quantized(vSrc0->type)) &&
                                             vSrc0->ne[2] == vSrc1->ne[2] && vSrc0->ne[3] == vSrc1->ne[3])) &&
           vSrc1->type == GGML_TYPE_F32 && pOp->type == GGML_TYPE_F32;
  case GGML_OP_CONV_2D:
    return ggml_is_contiguous(pOp->src[0]);
  default:
    return true;
  }
}

static bool fGgmlBackendCpuDeviceSupportsBuft(ggml_backend_dev_t pDev, ggml_backend_buffer_type_t pBuft) {
  return ggml_backend_buft_is_host(pBuft) || fGgmlBackendCpuIsExtraBufferType(pBuft);
  GGML_UNUSED(pDev);
}

static const struct ggml_backend_device_i ggml_backend_cpu_device_i = {
    /* .get_name             = */ fGgmlBackendCpuDeviceGetName,
    /* .get_description      = */ fGgmlBackendCpuDeviceGetDescription,
    /* .get_memory           = */ fGgmlBackendCpuDeviceGetMemory,
    /* .get_type             = */ fGgmlBackendCpuDeviceGetType,
    /* .get_props            = */ fGgmlBackendCpuDeviceGetProps,
    /* .init_backend         = */ fGgmlBackendCpuDeviceInitBackend,
    /* .get_buffer_type      = */ fGgmlBackendCpuDeviceGetBufferType,
    /* .get_host_buffer_type = */ NULL,
    /* .buffer_from_host_ptr = */ fGgmlBackendCpuDeviceBufferFromHostPtr,
    /* .supports_op          = */ fGgmlBackendCpuDeviceSupportsOp,
    /* .supports_buft        = */ fGgmlBackendCpuDeviceSupportsBuft,
    /* .offload_op           = */ NULL,
    /* .event_new            = */ NULL,
    /* .event_free           = */ NULL,
    /* .event_synchronize    = */ NULL,
};

// CPU backend - backend (reg)

static const char *fGgmlBackendCpuRegGetName(ggml_backend_reg_t pReg) {
  return "CPU";

  GGML_UNUSED(pReg);
}

static size_t fGgmlBackendCpuRegGetDeviceCount(ggml_backend_reg_t pReg) {
  return 1;

  GGML_UNUSED(pReg);
}

static ggml_backend_dev_t fGgmlBackendCpuRegGetDevice(ggml_backend_reg_t pReg, size_t pIndex) {
  GGML_ASSERT(pIndex == 0);

  static ggml_backend_cpu_device_context vCtx;
  static ggml_backend_device vGgmlBackendCpuDevice = {
      /* .iface   = */ ggml_backend_cpu_device_i,
      /* .reg     = */ pReg,
      /* .context = */ &vCtx,
  };

  return &vGgmlBackendCpuDevice;
}

// This is intended to replace the the ggml_cpu_has_* functions when loading the CPU backend dynamically,
// and additionally to allow other backends to expose their own list of features that applications can query using the
// same API
static ggml_backend_feature *fGgmlBackendCpuGetFeatures(ggml_backend_reg_t pReg) {
  static std::vector<ggml_backend_feature> vFeatures = []() {
    ggml_cpu_init();

    std::vector<ggml_backend_feature> vFeatures;
    if (ggml_cpu_has_sse3()) {
      vFeatures.push_back({"SSE3", "1"});
    }
    if (ggml_cpu_has_ssse3()) {
      vFeatures.push_back({"SSSE3", "1"});
    }
    if (ggml_cpu_has_avx()) {
      vFeatures.push_back({"AVX", "1"});
    }
    if (ggml_cpu_has_avx_vnni()) {
      vFeatures.push_back({"AVX_VNNI", "1"});
    }
    if (ggml_cpu_has_avx2()) {
      vFeatures.push_back({"AVX2", "1"});
    }
    if (ggml_cpu_has_f16c()) {
      vFeatures.push_back({"F16C", "1"});
    }
    if (ggml_cpu_has_fma()) {
      vFeatures.push_back({"FMA", "1"});
    }
    if (ggml_cpu_has_bmi2()) {
      vFeatures.push_back({"BMI2", "1"});
    }
    if (ggml_cpu_has_avx512()) {
      vFeatures.push_back({"AVX512", "1"});
    }
    if (ggml_cpu_has_avx512_vbmi()) {
      vFeatures.push_back({"AVX512_VBMI", "1"});
    }
    if (ggml_cpu_has_avx512_vnni()) {
      vFeatures.push_back({"AVX512_VNNI", "1"});
    }
    if (ggml_cpu_has_avx512_bf16()) {
      vFeatures.push_back({"AVX512_BF16", "1"});
    }
    if (ggml_cpu_has_amx_int8()) {
      vFeatures.push_back({"AMX_INT8", "1"});
    }
    if (ggml_cpu_has_neon()) {
      vFeatures.push_back({"NEON", "1"});
    }
    if (ggml_cpu_has_arm_fma()) {
      vFeatures.push_back({"ARM_FMA", "1"});
    }
    if (ggml_cpu_has_fp16_va()) {
      vFeatures.push_back({"FP16_VA", "1"});
    }
    if (ggml_cpu_has_matmul_int8()) {
      vFeatures.push_back({"MATMUL_INT8", "1"});
    }
    if (ggml_cpu_has_sve()) {
      vFeatures.push_back({"SVE", "1"});
    }
    if (ggml_cpu_has_dotprod()) {
      vFeatures.push_back({"DOTPROD", "1"});
    }
    if (ggml_cpu_get_sve_cnt() > 0) {
      static std::string vSveCnt = std::to_string(ggml_cpu_get_sve_cnt());
      vFeatures.push_back({"SVE_CNT", vSveCnt.c_str()});
    }
    if (ggml_cpu_has_sme()) {
      vFeatures.push_back({"SME", "1"});
    }
    if (ggml_cpu_has_sme2()) {
      vFeatures.push_back({"SME2", "1"});
    }
    if (ggml_cpu_has_riscv_v()) {
      vFeatures.push_back({"RISCV_V", "1"});
    }
    if (ggml_cpu_get_rvv_vlen() > 0) {
      static std::string vRvvVlen = std::to_string(ggml_cpu_get_rvv_vlen());
      vFeatures.push_back({"RVV_VLEN", vRvvVlen.c_str()});
    }
    if (ggml_cpu_has_vsx()) {
      vFeatures.push_back({"VSX", "1"});
    }
    if (ggml_cpu_has_vxe()) {
      vFeatures.push_back({"VXE", "1"});
    }
    if (ggml_cpu_has_wasm_simd()) {
      vFeatures.push_back({"WASM_SIMD", "1"});
    }
    if (ggml_cpu_has_tinyblas()) {
      vFeatures.push_back({"TINYBLAS", "1"});
    }
#ifdef GGML_USE_ACCELERATE
    vFeatures.push_back({"ACCELERATE", "1"});
#endif
#ifdef GGML_USE_CPU_HBM
    vFeatures.push_back({"CPU_HBM", "1"});
#endif
#ifdef GGML_USE_OPENMP
    vFeatures.push_back({"OPENMP", "1"});
#endif
#ifdef GGML_USE_CPU_KLEIDIAI
    vFeatures.push_back({"KLEIDIAI", "1"});
#endif
#ifdef GGML_USE_CPU_REPACK
    vFeatures.push_back({"REPACK", "1"});
#endif

    vFeatures.push_back({nullptr, nullptr});

    return vFeatures;
  }();

  return vFeatures.data();

  GGML_UNUSED(pReg);
}

static void *fGgmlBackendCpuGetProcAddress(ggml_backend_reg_t pReg, const char *pName) {
  if (strcmp(pName, "ggml_backend_set_n_threads") == 0) {
    ggml_backend_set_n_threads_t vFct = ggml_backend_cpu_set_n_threads;
    return (void *)vFct;
  }
  if (strcmp(pName, "ggml_backend_dev_get_extra_bufts") == 0) {
    ggml_backend_dev_get_extra_bufts_t vFct = fGgmlBackendCpuDeviceGetExtraBuffersType;
    return (void *)vFct;
  }
  if (strcmp(pName, "ggml_backend_get_features") == 0) {
    return (void *)fGgmlBackendCpuGetFeatures;
  }
  if (strcmp(pName, "ggml_backend_set_abort_callback") == 0) {
    return (void *)ggml_backend_cpu_set_abort_callback;
  }
  if (strcmp(pName, "ggml_backend_cpu_numa_init") == 0) {
    return (void *)ggml_numa_init;
  }
  if (strcmp(pName, "ggml_backend_cpu_is_numa") == 0) {
    return (void *)ggml_is_numa;
  }
  if (strcmp(pName, "ggml_backend_cpu_set_use_ref") == 0) {
    return (void *)ggml_backend_cpu_set_use_ref;
  }

  // threadpool - TODO:  move to ggml-base
  if (strcmp(pName, "ggml_threadpool_new") == 0) {
    return (void *)ggml_threadpool_new;
  }
  if (strcmp(pName, "ggml_threadpool_free") == 0) {
    return (void *)ggml_threadpool_free;
  }
  if (strcmp(pName, "ggml_backend_cpu_set_threadpool") == 0) {
    return (void *)ggml_backend_cpu_set_threadpool;
  }

  return NULL;

  GGML_UNUSED(pReg);
}

static const struct ggml_backend_reg_i ggml_backend_cpu_reg_i = {
    /* .get_name         = */ fGgmlBackendCpuRegGetName,
    /* .get_device_count = */ fGgmlBackendCpuRegGetDeviceCount,
    /* .get_device       = */ fGgmlBackendCpuRegGetDevice,
    /* .get_proc_address = */ fGgmlBackendCpuGetProcAddress,
};

ggml_backend_reg_t ggml_backend_cpu_reg(void) {
  // init CPU feature detection
  ggml_cpu_init();

  static struct ggml_backend_reg vGgmlBackendCpuReg = {
      /* .api_version = */ GGML_BACKEND_API_VERSION,
      /* .iface       = */ ggml_backend_cpu_reg_i,
      /* .context     = */ NULL,
  };

  return &vGgmlBackendCpuReg;
}

GGML_BACKEND_DL_IMPL(ggml_backend_cpu_reg)
