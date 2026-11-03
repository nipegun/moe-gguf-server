#include "ggml-backend-impl.h"

#if defined(__aarch64__)

#if defined(__linux__)
#include <sys/auxv.h>
#elif defined(__APPLE__)
#include <sys/sysctl.h>
#endif

#if !defined(HWCAP2_SVE2)
#define HWCAP2_SVE2 (1 << 1)
#endif

#if !defined(HWCAP2_I8MM)
#define HWCAP2_I8MM (1 << 13)
#endif

#if !defined(HWCAP2_SME)
#define HWCAP2_SME (1 << 23)
#endif

struct aarch64_features {
  // has_neon not needed, aarch64 has NEON guaranteed
  bool has_dotprod = false;
  bool vHasFp16Va = false;
  bool vHasSve = false;
  bool vHasSve2 = false;
  bool has_i8mm = false;
  bool has_sme = false;
  bool has_sme2 = false;

  aarch64_features() {
#if defined(__linux__)
    uint32_t vHwcap = getauxval(AT_HWCAP);
    uint32_t vHwcap2 = getauxval(AT_HWCAP2);

    has_dotprod = !!(vHwcap & HWCAP_ASIMDDP);
    vHasFp16Va = !!(vHwcap & HWCAP_FPHP);
    vHasSve = !!(vHwcap & HWCAP_SVE);
    vHasSve2 = !!(vHwcap2 & HWCAP2_SVE2);
    has_i8mm = !!(vHwcap2 & HWCAP2_I8MM);
    has_sme = !!(vHwcap2 & HWCAP2_SME);
#elif defined(__APPLE__)
    int oldp = 0;
    size_t size = sizeof(oldp);

    if (sysctlbyname("hw.optional.arm.FEAT_DotProd", &oldp, &size, NULL, 0) == 0) {
      has_dotprod = static_cast<bool>(oldp);
    }

    if (sysctlbyname("hw.optional.arm.FEAT_I8MM", &oldp, &size, NULL, 0) == 0) {
      has_i8mm = static_cast<bool>(oldp);
    }

    if (sysctlbyname("hw.optional.arm.FEAT_SME", &oldp, &size, NULL, 0) == 0) {
      has_sme = static_cast<bool>(oldp);
    }

    if (sysctlbyname("hw.optional.arm.FEAT_SME2", &oldp, &size, NULL, 0) == 0) {
      has_sme2 = static_cast<bool>(oldp);
    }

    // Apple apparently does not implement SVE yet
#endif
  }
};

static int fGgmlBackendCpuAarch64Score() {
  int vScore = 1;
  aarch64_features vAf;

#ifdef GGML_USE_DOTPROD
  if (!vAf.has_dotprod) {
    return 0;
  }
  vScore += 1 << 1;
#endif
#ifdef GGML_USE_FP16_VECTOR_ARITHMETIC
  if (!vAf.vHasFp16Va) {
    return 0;
  }
  vScore += 1 << 2;
#endif
#ifdef GGML_USE_SVE
  if (!vAf.vHasSve) {
    return 0;
  }
  vScore += 1 << 3;
#endif
#ifdef GGML_USE_MATMUL_INT8
  if (!vAf.has_i8mm) {
    return 0;
  }
  vScore += 1 << 4;
#endif
#ifdef GGML_USE_SVE2
  if (!vAf.vHasSve2) {
    return 0;
  }
  vScore += 1 << 5;
#endif
#ifdef GGML_USE_SME
  if (!vAf.has_sme) {
    return 0;
  }
  vScore += 1 << 6;
#endif

  return vScore;
}

GGML_BACKEND_DL_SCORE_IMPL(fGgmlBackendCpuAarch64Score)

#endif // defined(__aarch64__)
