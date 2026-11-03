#include "ggml-backend-impl.h"

#if defined(__s390x__)
#include <sys/auxv.h>

// find hwcap bits in asm/elf.h
#ifndef HWCAP_VXRS_EXT2
#define HWCAP_VXRS_EXT2 (1 << 15)
#endif

#ifndef HWCAP_NNPA
#define HWCAP_NNPA (1 << 20)
#endif

struct s390x_features {
  bool vHasVxe2 = false;
  bool vHasNnpa = false;

  s390x_features() {
    uint32_t vHwcap = getauxval(AT_HWCAP);
    // NOTE: use hwcap2 with DFLT for z17 and later
    // uint32_t hwcap2 = getauxval(AT_HWCAP2);

    vHasVxe2 = !!(vHwcap & HWCAP_VXRS_EXT2);
    vHasNnpa = !!(vHwcap & HWCAP_NNPA);
  }
};

static int fGgmlBackendCpuS390xScore() {
  int vScore = 1;
  s390x_features vSf;

// IBM z15 / LinuxONE 3
#ifdef GGML_USE_VXE2
  if (!vSf.vHasVxe2) {
    return 0;
  }
  vScore += 1 << 1;
#endif

// IBM z16 / LinuxONE 4 and z17 / LinuxONE 5
#ifdef GGML_USE_NNPA
  if (!vSf.vHasNnpa) {
    return 0;
  }
  vScore += 1 << 2;
#endif

  return vScore;
}

GGML_BACKEND_DL_SCORE_IMPL(fGgmlBackendCpuS390xScore)

#endif // __s390x__
