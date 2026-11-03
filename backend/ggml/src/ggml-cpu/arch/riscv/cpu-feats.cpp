#include "ggml-backend-impl.h"

#if defined(__riscv) && __riscv_xlen == 64
#include <asm/hwprobe.h>
#include <asm/unistd.h>
#include <unistd.h>

struct riscv64_features {
  bool vHasRvv = false;

  riscv64_features() {
    struct riscv_hwprobe vProbe;
    vProbe.key = RISCV_HWPROBE_KEY_IMA_EXT_0;
    vProbe.value = 0;

    int vRet = syscall(__NR_riscv_hwprobe, &vProbe, 1, 0, NULL, 0);

    if (0 == vRet) {
      vHasRvv = !!(vProbe.value & RISCV_HWPROBE_IMA_V);
    }
  }
};

static int fGgmlBackendCpuRiscv64Score() {
  int vScore = 1;
  riscv64_features vRf;

#ifdef GGML_USE_RVV
  if (!vRf.vHasRvv) {
    return 0;
  }
  vScore += 1 << 1;
#endif

  return vScore;
}

GGML_BACKEND_DL_SCORE_IMPL(fGgmlBackendCpuRiscv64Score)

#endif // __riscv && __riscv_xlen == 64
