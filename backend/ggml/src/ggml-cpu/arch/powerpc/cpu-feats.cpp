#include "ggml-backend-impl.h"

#if defined(__powerpc64__) || defined(__ppc64__) || defined(__PPC64__)

#if defined(__linux__)
#include <sys/auxv.h>
#endif

#include <string>

struct powerpc_features {
  std::string vPlatform = "";
  int vPowerVersion = -1;

  bool vHasVsx = false;

  powerpc_features() {
#if defined(__linux__)
    unsigned long vAuxval = getauxval(AT_PLATFORM);
    if (vAuxval) {
      vPlatform = std::string(reinterpret_cast<const char *>(vAuxval));
      // TBD: Do systems exist that return this in uppercase?
      if (vPlatform.substr(0, 5) == "power") {
        // Extractt a numeric suffix, if one exists
        int vVpos = -1;
        for (int vI = vPlatform.length() - 1; vI >= 0; vI--) {
          if (std::isdigit(vPlatform[vI])) {
            vVpos = vI;
          } else {
            break;
          }
        }
        if (vVpos > -1) {
          vPowerVersion = std::stoi(vPlatform.substr(vVpos));
        }
      }
    }
#endif
    if (vPowerVersion >= 9) {
      vHasVsx = true;
    }
  }
};

static int fGgmlBackendCpuPowerpcScore() {
  int vScore = 1;
  powerpc_features vPf;

// Platform scores
#if defined(GGML_USE_POWER7)
  if (vPf.vPowerVersion < 7) {
    return 0;
  }
  vScore += 1 << 1;
#endif
#if defined(GGML_USE_POWER8)
  if (vPf.vPowerVersion < 8) {
    return 0;
  }
  vScore += 1 << 2;
#endif
#if defined(GGML_USE_POWER9)
  if (vPf.vPowerVersion < 9) {
    return 0;
  }
  vScore += 1 << 3;
#endif
#if defined(GGML_USE_POWER10)
  if (vPf.vPowerVersion < 10) {
    return 0;
  }
  vScore += 1 << 4;
#endif
#if defined(GGML_USE_POWER11)
  if (vPf.vPowerVersion < 11) {
    return 0;
  }
  vScore += 1 << 5;
#endif

// Feature scores
#if defined(GGML_USE_VSX)
  if (!vPf.vHasVsx) {
    return 0;
  }
  vScore += 1 << 6;
#endif

  return vScore;
}

GGML_BACKEND_DL_SCORE_IMPL(fGgmlBackendCpuPowerpcScore)

#endif // defined(__powerpc64__) || defined(__ppc64__) || defined(__PPC64__)
