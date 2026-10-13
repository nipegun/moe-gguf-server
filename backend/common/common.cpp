#include "ggml.h"
#include "gguf.h"

#include "build-info.h"
#include "common.h"
#include "fit.h"
#include "log.h"
#include "mgs.h"
#include "../src/mgs-ext.h" // staging API: MoE expert cache size
#include "sampling.h"
#include "speculative.h"
#include "unicode.h"

#include <algorithm>
#include <cinttypes>
#include <climits>
#include <cmath>
#include <chrono>
#include <cstdarg>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <regex>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

#if defined(__APPLE__) && defined(__MACH__)
#include <sys/types.h>
#include <sys/sysctl.h>
#endif

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <locale>
#include <windows.h>
#include <string.h>
#include <fcntl.h>
#include <io.h>
#else
#include <sys/ioctl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

#if defined(__linux__)
#include <sys/types.h>
#include <pwd.h>
#include <sched.h>
#endif

#if defined(_AIX)
#include <sys/systemcfg.h>
#endif

#if defined(_MSC_VER)
#pragma warning(disable : 4244 4267) // possible loss of data
#endif

common_time_meas::common_time_meas(int64_t &pTAcc, bool pDisable)
    : cTStartUs(pDisable ? -1 : ggml_time_us()), vTAcc(pTAcc) {}

common_time_meas::~common_time_meas() {
  if (cTStartUs >= 0) {
    vTAcc += ggml_time_us() - cTStartUs;
  }
}

//
// CPU utils
//

int32_t fCommonCpuGetNumPhysicalCores() {
#if defined(_AIX)
  int32_t vLogicalCpus = _system_configuration.ncpus;
  int32_t vSmtThreads = _system_configuration.smt_threads;
  if (vSmtThreads > 0) {
    return static_cast<int32_t>(vLogicalCpus / vSmtThreads);
  }
  if (vLogicalCpus > 0) {
    return static_cast<int32_t>(vLogicalCpus);
  }
#elif defined(__linux__)
  // enumerate the set of thread siblings, num entries is num cores
  std::unordered_set<std::string> sSiblings;
  for (uint32_t vCpu = 0; vCpu < UINT32_MAX; ++vCpu) {
    std::ifstream vThreadSiblings("/sys/devices/system/cpu/cpu" + std::to_string(vCpu) + "/topology/thread_siblings");
    if (!vThreadSiblings.is_open()) {
      break; // no more cpus
    }
    std::string vLine;
    if (std::getline(vThreadSiblings, vLine)) {
      sSiblings.insert(vLine);
    }
  }
  if (!sSiblings.empty()) {
    return static_cast<int32_t>(sSiblings.size());
  }
#elif defined(__APPLE__) && defined(__MACH__)
  int32_t vNumPhysicalCores;
  size_t vLen = sizeof(vNumPhysicalCores);
  int vResult = sysctlbyname("hw.perflevel0.physicalcpu", &vNumPhysicalCores, &vLen, NULL, 0);
  if (vResult == 0) {
    return vNumPhysicalCores;
  }
  vResult = sysctlbyname("hw.physicalcpu", &vNumPhysicalCores, &vLen, NULL, 0);
  if (vResult == 0) {
    return vNumPhysicalCores;
  }
#elif defined(_WIN32) && (_WIN32_WINNT >= 0x0601) && !defined(__MINGW64__) // windows 7 and later
  // TODO: windows + arm64 + mingw64
  unsigned int vNThreadsWin = std::thread::hardware_concurrency();
  unsigned int vDefaultThreads = vNThreadsWin > 0 ? (vNThreadsWin <= 4 ? vNThreadsWin : vNThreadsWin / 2) : 4;

  DWORD vBufferSize = 0;
  if (!GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &vBufferSize)) {
    if (GetLastError() != ERROR_INSUFFICIENT_BUFFER) {
      return vDefaultThreads;
    }
  }

  std::vector<char> lBuffer(vBufferSize);
  if (!GetLogicalProcessorInformationEx(RelationProcessorCore,
                                        reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(lBuffer.data()),
                                        &vBufferSize)) {
    return vDefaultThreads;
  }

  int32_t vNumPhysicalCores = 0;
  PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX vInfo =
      reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(lBuffer.data());
  while (vBufferSize > 0) {
    if (vInfo->Relationship == RelationProcessorCore) {
      vNumPhysicalCores += vInfo->Processor.GroupCount;
    }
    vBufferSize -= vInfo->Size;
    vInfo = reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(reinterpret_cast<char *>(vInfo) + vInfo->Size);
  }

  return vNumPhysicalCores > 0 ? vNumPhysicalCores : vDefaultThreads;
#endif
  unsigned int vNThreads = std::thread::hardware_concurrency();
  return vNThreads > 0 ? (vNThreads <= 4 ? vNThreads : vNThreads / 2) : 4;
}

#if defined(__x86_64__) && defined(__linux__) && !defined(__ANDROID__)
#include <pthread.h>

static void fCpuid(unsigned pLeaf, unsigned pSubleaf, unsigned *pEax, unsigned *pEbx, unsigned *pEcx, unsigned *pEdx) {
  __asm__("movq\t%%rbx,%%rsi\n\t"
          "cpuid\n\t"
          "xchgq\t%%rbx,%%rsi"
          : "=a"(*pEax), "=S"(*pEbx), "=c"(*pEcx), "=d"(*pEdx)
          : "0"(pLeaf), "2"(pSubleaf));
}

static int fPinCpu(int pCpu) {
  cpu_set_t vMask;
  CPU_ZERO(&vMask);
  CPU_SET(pCpu, &vMask);
  return pthread_setaffinity_np(pthread_self(), sizeof(vMask), &vMask);
}

static bool fIsHybridCpu(void) {
  unsigned vEax, vEbx, vEcx, vEdx;
  fCpuid(7, 0, &vEax, &vEbx, &vEcx, &vEdx);
  return !!(vEdx & (1u << 15));
}

static bool fIsRunningOnEfficiencyCore(void) {
  unsigned vEax, vEbx, vEcx, vEdx;
  fCpuid(0x1a, 0, &vEax, &vEbx, &vEcx, &vEdx);
  int vIntelAtom = 0x20;
  int vCoreType = (vEax & 0xff000000u) >> 24;
  return vCoreType == vIntelAtom;
}

static int fCpuCountMathCpus(int pNCpu) {
  int vResult = 0;
  for (int vCpu = 0; vCpu < pNCpu; ++vCpu) {
    if (fPinCpu(vCpu)) {
      return -1;
    }
    if (fIsRunningOnEfficiencyCore()) {
      continue; // efficiency cores harm lockstep threading
    }
    ++vCpu; // hyperthreading isn't useful for linear algebra
    ++vResult;
  }
  return vResult;
}

#endif // __x86_64__ && __linux__

/**
 * Returns number of CPUs on system that are useful for math.
 */
int32_t fCommonCpuGetNumMath() {
#if defined(__x86_64__) && defined(__linux__) && !defined(__ANDROID__)
  int vNCpu = sysconf(_SC_NPROCESSORS_ONLN);
  if (vNCpu < 1) {
    return fCommonCpuGetNumPhysicalCores();
  }
  if (fIsHybridCpu()) {
    cpu_set_t vAffinity;
    if (!pthread_getaffinity_np(pthread_self(), sizeof(vAffinity), &vAffinity)) {
      int vResult = fCpuCountMathCpus(vNCpu);
      pthread_setaffinity_np(pthread_self(), sizeof(vAffinity), &vAffinity);
      if (vResult > 0) {
        return vResult;
      }
    }
  }
#elif defined(__powerpc64__) || defined(__powerpc__)
  int32_t vSmtFactor = 1;
  int vPhyCpus = fCommonCpuGetNumPhysicalCores();
  int vLogicalCpus = sysconf(_SC_NPROCESSORS_ONLN);
  if (vPhyCpus > 0 && vLogicalCpus > vPhyCpus) {
    vSmtFactor = vLogicalCpus / vPhyCpus;
  }
  return vPhyCpus * std::min(vSmtFactor, 2);
#endif
  return fCommonCpuGetNumPhysicalCores();
}

// Logical CPUs this process may run on (its affinity mask on Linux, so containers and taskset count).
static int32_t fCommonLogicalCpus() {
#if defined(__linux__)
  cpu_set_t vMask;
  CPU_ZERO(&vMask);
  if (sched_getaffinity(0, sizeof(vMask), &vMask) == 0) {
    return CPU_COUNT(&vMask);
  }
#endif
  return (int32_t)std::thread::hardware_concurrency();
}

// Default number of CPU threads: the cores useful for math, but never more than half of the logical CPUs.
// The CPU part of MoE inference (the experts kept in system memory) is bound by memory bandwidth and
// saturates with few threads, while using every logical CPU, as happens in a virtual machine that does
// not expose SMT, makes the workers compete with the thread that drives the GPU: on 16 vCPUs, 4 to 12
// threads generated at the same speed and 16 threads were 13% slower. With SMT visible, half of the
// logical CPUs is already the number of physical cores, so nothing changes there.
static int32_t fCommonDefaultThreads() {
  const int32_t cMath = fCommonCpuGetNumMath();
  const int32_t cLogical = fCommonLogicalCpus();
  if (cLogical >= 2) {
    return std::max<int32_t>(1, std::min<int32_t>(cMath, cLogical / 2));
  }
  return std::max<int32_t>(1, cMath);
}

// Helper for setting process priority

#if defined(_WIN32)

bool fSetProcessPriority(enum ggml_sched_priority pPrio) {
  if (pPrio == GGML_SCHED_PRIO_NORMAL) {
    return true;
  }

  DWORD vP = NORMAL_PRIORITY_CLASS;
  switch (pPrio) {
  case GGML_SCHED_PRIO_LOW:
    vP = BELOW_NORMAL_PRIORITY_CLASS;
    break;
  case GGML_SCHED_PRIO_NORMAL:
    vP = NORMAL_PRIORITY_CLASS;
    break;
  case GGML_SCHED_PRIO_MEDIUM:
    vP = ABOVE_NORMAL_PRIORITY_CLASS;
    break;
  case GGML_SCHED_PRIO_HIGH:
    vP = HIGH_PRIORITY_CLASS;
    break;
  case GGML_SCHED_PRIO_REALTIME:
    vP = REALTIME_PRIORITY_CLASS;
    break;
  }

  if (!SetPriorityClass(GetCurrentProcess(), vP)) {
    COM_WRN("failed to set process priority class %d : (%d)\n", pPrio, (int)GetLastError());
    return false;
  }

  return true;
}

#else // MacOS and POSIX
#include <sys/types.h>
#include <sys/resource.h>

bool fSetProcessPriority(enum ggml_sched_priority pPrio) {
  if (pPrio == GGML_SCHED_PRIO_NORMAL) {
    return true;
  }

  int vP = 0;
  switch (pPrio) {
  case GGML_SCHED_PRIO_LOW:
    vP = 5;
    break;
  case GGML_SCHED_PRIO_NORMAL:
    vP = 0;
    break;
  case GGML_SCHED_PRIO_MEDIUM:
    vP = -5;
    break;
  case GGML_SCHED_PRIO_HIGH:
    vP = -10;
    break;
  case GGML_SCHED_PRIO_REALTIME:
    vP = -20;
    break;
  }

  if (setpriority(PRIO_PROCESS, 0, vP) != 0) {
    COM_WRN("failed to set process priority %d : %s (%d)\n", pPrio, strerror(errno), errno);
    return false;
  }
  return true;
}

#endif

//
// CLI argument parsing
//

void fPostprocessCpuParams(common_cpu_params &pCpuparams, const common_cpu_params *pRoleModel) {
  int32_t vNSet = 0;

  if (pCpuparams.vNThreads < 0) {
    // Assuming everything about cpuparams is invalid
    if (pRoleModel != nullptr) {
      pCpuparams = *pRoleModel;
    } else {
      pCpuparams.vNThreads = fCommonDefaultThreads();
    }
  }

  for (int32_t vI = 0; vI < GGML_MAX_N_THREADS; vI++) {
    if (pCpuparams.aCpumask[vI]) {
      vNSet++;
    }
  }

  if (vNSet && vNSet < pCpuparams.vNThreads) {
    // Not enough set bits, may experience performance issues.
    COM_WRN("Not enough set bits in CPU mask (%d) to satisfy requested thread count: %d\n", vNSet,
            pCpuparams.vNThreads);
  }
}

bool fParseCpuRange(const std::string &pRange, bool (&pBoolmask)[GGML_MAX_N_THREADS]) {
  size_t vDashLoc = pRange.find('-');
  if (vDashLoc == std::string::npos) {
    COM_ERR("%s", "Format of CPU range is invalid! Expected [<start>]-[<end>].\n");
    return false;
  }

  size_t vStartI;
  size_t vEndI;

  if (vDashLoc == 0) {
    vStartI = 0;
  } else {
    vStartI = std::stoull(pRange.substr(0, vDashLoc));
    if (vStartI >= GGML_MAX_N_THREADS) {
      COM_ERR("%s", "Start index out of bounds!\n");
      return false;
    }
  }

  if (vDashLoc == pRange.length() - 1) {
    vEndI = GGML_MAX_N_THREADS - 1;
  } else {
    vEndI = std::stoull(pRange.substr(vDashLoc + 1));
    if (vEndI >= GGML_MAX_N_THREADS) {
      COM_ERR("%s", "End index out of bounds!\n");
      return false;
    }
  }

  for (size_t vI = vStartI; vI <= vEndI; vI++) {
    pBoolmask[vI] = true;
  }

  return true;
}

bool fParseCpuMask(const std::string &pMask, bool (&pBoolmask)[GGML_MAX_N_THREADS]) {
  // Discard potential 0x prefix
  size_t vStartI = 0;
  if (pMask.length() >= 2 && pMask.substr(0, 2) == "0x") {
    vStartI = 2;
  }

  size_t vNumDigits = pMask.length() - vStartI;
  vNumDigits = std::min<size_t>(vNumDigits, 128);

  size_t vEndI = vNumDigits + vStartI;

  for (size_t vI = vStartI, vN = (vNumDigits * 4 - 1); vI < vEndI; vI++, vN -= 4) {
    char vC = pMask.at(vI);
    int8_t vId = vC;

    if ((vC >= '0' && vC <= '9')) {
      vId -= '0';
    } else if (vC >= 'a' && vC <= 'f') {
      vId -= 'a' - 10;
    } else if (vC >= 'A' && vC <= 'F') {
      vId -= 'A' - 10;
    } else {
      COM_ERR("Invalid hex character '%c' at position %d\n", vC, int32_t(vI));
      return false;
    }

    pBoolmask[vN] = pBoolmask[vN] || ((vId & 8) != 0);
    pBoolmask[vN - 1] = pBoolmask[vN - 1] || ((vId & 4) != 0);
    pBoolmask[vN - 2] = pBoolmask[vN - 2] || ((vId & 2) != 0);
    pBoolmask[vN - 3] = pBoolmask[vN - 3] || ((vId & 1) != 0);
  }

  return true;
}

void fCommonInit() {
#if defined(_WIN32)
  SetConsoleOutputCP(CP_UTF8);
  SetConsoleCP(CP_UTF8);
#endif

  fCommonLogSetPrefix(fCommonLogMain(), true);
  fCommonLogSetTimestamps(fCommonLogMain(), true);

  mgs_log_set(fCommonLogDefaultCallback, NULL);
}

void fCommonParamsPrintInfo(const common_params &pParams, bool pPrintDevices) {
#ifdef NDEBUG
  const char *vBuildType = "";
#else
  const char *vBuildType = " (debug)";
#endif
  COM_TRC("%s: version %s with %s for %s%s\n", __func__, mgs_version(), mgs_compiler(), mgs_build_target(),
          vBuildType);

  COM_INF("%s: verbosity = %d (adjust with the `-lv N` CLI arg)\n", __func__, fCommonLogGetVerbosityThold());

  // device enumeration creates a primary context on CUDA backends, skip it when the caller does not own any device
  if (pPrintDevices) {
    COM_TRC("%s", "device_info:\n");
    for (size_t vI = 0; vI < ggml_backend_dev_count(); ++vI) {
      auto *vDev = ggml_backend_dev_get(vI);
      size_t vFree, vTotal;
      ggml_backend_dev_memory(vDev, &vFree, &vTotal);
      COM_TRC("  - %-8s: %s (%zu MiB, %zu MiB free)\n", ggml_backend_dev_name(vDev), ggml_backend_dev_description(vDev),
              vTotal / 1024 / 1024, vFree / 1024 / 1024);
    }
  }
  COM_TRC("%s\n", fCommonParamsGetSystemInfo(pParams).c_str());
}

std::string fCommonParamsGetSystemInfo(const common_params &pParams) {
  std::ostringstream vOs;

  vOs << "system_info: n_threads = " << pParams.vCpuparams.vNThreads;
  if (pParams.vCpuparamsBatch.vNThreads != -1) {
    vOs << " (n_threads_batch = " << pParams.vCpuparamsBatch.vNThreads << ")";
  }
#if defined(_WIN32) && (_WIN32_WINNT >= 0x0601) && !defined(__MINGW64__) // windows 7 and later
  // TODO: windows + arm64 + mingw64
  DWORD vLogicalProcessorCount = GetActiveProcessorCount(ALL_PROCESSOR_GROUPS);
  vOs << " / " << vLogicalProcessorCount << " | " << mgs_print_system_info();
#else
  vOs << " / " << std::thread::hardware_concurrency() << " | " << mgs_print_system_info();
#endif

  return vOs.str();
}

//
// String utils
//

std::string fStringFormat(const char *pFmt, ...) {
  va_list vAp;
  va_list vAp2;
  va_start(vAp, pFmt);
  va_copy(vAp2, vAp);
  int vSize = vsnprintf(NULL, 0, pFmt, vAp);
  GGML_ASSERT(vSize >= 0 && vSize < INT_MAX); // NOLINT
  std::vector<char> lBuf(vSize + 1);
  int vSize2 = vsnprintf(lBuf.data(), vSize + 1, pFmt, vAp2);
  GGML_ASSERT(vSize2 == vSize);
  va_end(vAp2);
  va_end(vAp);
  return std::string(lBuf.data(), vSize);
}

std::string fStringStrip(const std::string &pStr) {
  size_t vStart = 0;
  size_t vEnd = pStr.size();
  while (vStart < vEnd && std::isspace(pStr[vStart])) {
    vStart++;
  }
  while (vEnd > vStart && std::isspace(pStr[vEnd - 1])) {
    vEnd--;
  }
  return pStr.substr(vStart, vEnd - vStart);
}

std::string fStringLcs(std::string_view pA, std::string_view pB) {
  if (pA.empty() || pB.empty())
    return {};

  std::vector<std::vector<size_t>> lDp(pA.size() + 1, std::vector<size_t>(pB.size() + 1, 0));
  size_t vBestLen = 0;
  size_t vBestEndA = 0;

  for (size_t vI = 1; vI <= pA.size(); ++vI) {
    for (size_t vJ = 1; vJ <= pB.size(); ++vJ) {
      if (pA[vI - 1] == pB[vJ - 1]) {
        lDp[vI][vJ] = lDp[vI - 1][vJ - 1] + 1;
        if (lDp[vI][vJ] > vBestLen) {
          vBestLen = lDp[vI][vJ];
          vBestEndA = vI;
        }
      }
    }
  }
  return std::string(pA.substr(vBestEndA - vBestLen, vBestLen));
}

std::string fStringGetSortableTimestamp() {
  using clock = std::chrono::system_clock;

  const clock::time_point cCurrentTime = clock::now();
  const time_t cAsTimeT = clock::to_time_t(cCurrentTime);
  char aTimestampNoNs[100];
  std::strftime(aTimestampNoNs, 100, "%Y_%m_%d-%H_%M_%S", std::localtime(&cAsTimeT));

  const int64_t cNs =
      std::chrono::duration_cast<std::chrono::nanoseconds>(cCurrentTime.time_since_epoch() % 1000000000).count();
  char aTimestampNs[11];
  snprintf(aTimestampNs, 11, "%09" PRId64, cNs);

  return std::string(aTimestampNoNs) + "." + std::string(aTimestampNs);
}

void fStringReplaceAll(std::string &pS, const std::string &pSearch, const std::string &pReplace) {
  if (pSearch.empty()) {
    return;
  }
  std::string vBuilder;
  vBuilder.reserve(pS.length());
  size_t vPos = 0;
  size_t vLastPos = 0;
  while ((vPos = pS.find(pSearch, vLastPos)) != std::string::npos) {
    vBuilder.append(pS, vLastPos, vPos - vLastPos);
    vBuilder.append(pReplace);
    vLastPos = vPos + pSearch.length();
  }
  vBuilder.append(pS, vLastPos, std::string::npos);
  pS = std::move(vBuilder);
}

std::string fRegexEscape(const std::string &pS) {
  static const std::regex cSpecialChars("[.^$|()*+?\\[\\]{}\\\\]");
  return std::regex_replace(pS, cSpecialChars, "\\$&");
}

std::string fStringJoin(const std::vector<std::string> &pValues, const std::string &pSeparator) {
  std::ostringstream vResult;
  for (size_t vI = 0; vI < pValues.size(); ++vI) {
    if (vI > 0) {
      vResult << pSeparator;
    }
    vResult << pValues[vI];
  }
  return vResult.str();
}

std::vector<std::string> fStringSplit(const std::string &pStr, const std::string &pDelimiter) {
  std::vector<std::string> lParts;
  size_t vStart = 0;
  size_t vEnd = pStr.find(pDelimiter);

  while (vEnd != std::string::npos) {
    lParts.push_back(pStr.substr(vStart, vEnd - vStart));
    vStart = vEnd + pDelimiter.length();
    vEnd = pStr.find(pDelimiter, vStart);
  }

  lParts.push_back(pStr.substr(vStart));

  return lParts;
}

std::string fStringRepeat(const std::string &pStr, size_t pN) {
  if (pN == 0) {
    return "";
  }

  std::string vResult;
  vResult.reserve(pStr.length() * pN);

  for (size_t vI = 0; vI < pN; ++vI) {
    vResult += pStr;
  }

  return vResult;
}

std::string fStringFrom(bool pValue) { return pValue ? "true" : "false"; }

std::string fStringFrom(const std::vector<int> &pValues) {
  std::stringstream vBuf;

  vBuf << "[ ";
  bool vFirst = true;
  for (auto vE : pValues) {
    if (vFirst) {
      vFirst = false;
    } else {
      vBuf << ", ";
    }
    vBuf << std::to_string(vE);
  }
  vBuf << " ]";

  return vBuf.str();
}

std::string fStringFrom(const struct mgs_context *pCtx, const std::vector<mgs_token> &pTokens) {
  std::stringstream vBuf;

  vBuf << "[ ";

  bool vFirst = true;
  for (const auto &vToken : pTokens) {
    if (!vFirst) {
      vBuf << ", ";
    } else {
      vFirst = false;
    }

    auto vDetokenized = fCommonTokenToPiece(pCtx, vToken);

    vBuf << "'" << vDetokenized << "'"
         << ":" << std::to_string(vToken);
  }

  vBuf << " ]";

  return vBuf.str();
}

std::string fStringFrom(const struct mgs_context *pCtx, const struct mgs_batch &pBatch) {
  std::stringstream vBuf;

  vBuf << "[ ";

  bool vFirst = true;
  for (int vI = 0; vI < pBatch.n_tokens; ++vI) {
    if (!vFirst) {
      vBuf << ", ";
    } else {
      vFirst = false;
    }

    auto vDetokenized = fCommonTokenToPiece(pCtx, pBatch.token[vI]);

    vBuf << "\n"
         << std::to_string(vI) << ", token '" << vDetokenized << "'"
         << ", pos " << std::to_string(pBatch.pos[vI]) << ", n_seq_id " << std::to_string(pBatch.n_seq_id[vI])
         << ", seq_id " << std::to_string(pBatch.seq_id[vI][0]) << ", logits " << std::to_string(pBatch.logits[vI]);
  }

  vBuf << " ]";

  return vBuf.str();
}

void fStringProcessEscapes(std::string &pInput) {
  std::size_t vInputLen = pInput.length();
  std::size_t vOutputIdx = 0;

  for (std::size_t vInputIdx = 0; vInputIdx < vInputLen; ++vInputIdx) {
    if (pInput[vInputIdx] == '\\' && vInputIdx + 1 < vInputLen) {
      switch (pInput[++vInputIdx]) {
      case 'n':
        pInput[vOutputIdx++] = '\n';
        break;
      case 'r':
        pInput[vOutputIdx++] = '\r';
        break;
      case 't':
        pInput[vOutputIdx++] = '\t';
        break;
      case '\'':
        pInput[vOutputIdx++] = '\'';
        break;
      case '\"':
        pInput[vOutputIdx++] = '\"';
        break;
      case '\\':
        pInput[vOutputIdx++] = '\\';
        break;
      case 'x':
        // Handle \x12, etc
        if (vInputIdx + 2 < vInputLen) {
          const char aX[3] = {pInput[vInputIdx + 1], pInput[vInputIdx + 2], 0};
          char *vErrP = nullptr;
          const long cVal = std::strtol(aX, &vErrP, 16);
          if (vErrP == aX + 2) {
            vInputIdx += 2;
            pInput[vOutputIdx++] = char(cVal);
            break;
          }
        }
        // fall through
      default:
        pInput[vOutputIdx++] = '\\';
        pInput[vOutputIdx++] = pInput[vInputIdx];
        break;
      }
    } else {
      pInput[vOutputIdx++] = pInput[vInputIdx];
    }
  }

  pInput.resize(vOutputIdx);
}

bool fStringParseKvOverride(const char *pData, std::vector<mgs_model_kv_override> &pOverrides) {
  const char *vSep = strchr(pData, '=');
  if (vSep == nullptr || vSep - pData >= 128) {
    COM_ERR("%s: malformed KV override '%s'\n", __func__, pData);
    return false;
  }
  mgs_model_kv_override vKvo;
  std::strncpy(vKvo.key, pData, vSep - pData);
  vKvo.key[vSep - pData] = 0;
  vSep++;
  if (strncmp(vSep, "int:", 4) == 0) {
    vSep += 4;
    vKvo.tag = MGS_KV_OVERRIDE_TYPE_INT;
    vKvo.val_i64 = std::atol(vSep);
  } else if (strncmp(vSep, "float:", 6) == 0) {
    vSep += 6;
    vKvo.tag = MGS_KV_OVERRIDE_TYPE_FLOAT;
    vKvo.val_f64 = std::atof(vSep);
  } else if (strncmp(vSep, "bool:", 5) == 0) {
    vSep += 5;
    vKvo.tag = MGS_KV_OVERRIDE_TYPE_BOOL;
    if (std::strcmp(vSep, "true") == 0) {
      vKvo.val_bool = true;
    } else if (std::strcmp(vSep, "false") == 0) {
      vKvo.val_bool = false;
    } else {
      COM_ERR("%s: invalid boolean value for KV override '%s'\n", __func__, pData);
      return false;
    }
  } else if (strncmp(vSep, "str:", 4) == 0) {
    vSep += 4;
    vKvo.tag = MGS_KV_OVERRIDE_TYPE_STR;
    if (strlen(vSep) > 127) {
      COM_ERR("%s: malformed KV override '%s', value cannot exceed 127 chars\n", __func__, pData);
      return false;
    }
    strncpy(vKvo.val_str, vSep, 127);
    vKvo.val_str[127] = '\0';
  } else {
    COM_ERR("%s: invalid type for KV override '%s'\n", __func__, pData);
    return false;
  }
  pOverrides.emplace_back(std::move(vKvo));
  return true;
}

static inline bool fGlobClassMatch(const char pC, const char *pPattern, const char *pClassEnd) {
  const char *vClassStart = pPattern;
  bool vNegated = false;

  if (*vClassStart == '!') {
    vNegated = true;
    vClassStart++;
  }

  // If first character after negation is ']' or '-', treat it as literal
  if (*vClassStart == ']' || *vClassStart == '-') {
    if (vClassStart < pClassEnd && *vClassStart == pC) {
      return !vNegated;
    }
    vClassStart++;
  }

  bool vMatched = false;

  while (vClassStart < pClassEnd) {
    if (vClassStart + 2 < pClassEnd && vClassStart[1] == '-' && vClassStart[2] != ']') {
      char vStartChar = *vClassStart;
      char vEndChar = vClassStart[2];
      if (pC >= vStartChar && pC <= vEndChar) {
        vMatched = true;
        break;
      }
      vClassStart += 3;
    } else {
      if (*vClassStart == pC) {
        vMatched = true;
        break;
      }
      vClassStart++;
    }
  }

  return vNegated ? !vMatched : vMatched;
}

// simple glob: * matches non-/ chars, ** matches anything including /, [] matches character class
static inline bool fGlobMatch(const char *pPattern, const char *pStr) {
  if (*pPattern == '\0') {
    return *pStr == '\0';
  }
  if (pPattern[0] == '*' && pPattern[1] == '*') {
    const char *vP = pPattern + 2;
    if (fGlobMatch(vP, pStr))
      return true;
    if (*pStr != '\0')
      return fGlobMatch(pPattern, pStr + 1);
    return false;
  }
  if (*pPattern == '*') {
    const char *vP = pPattern + 1;
    for (; *pStr != '\0' && *pStr != '/'; pStr++) {
      if (fGlobMatch(vP, pStr))
        return true;
    }
    return fGlobMatch(vP, pStr);
  }
  if (*pPattern == '?' && *pStr != '\0' && *pStr != '/') {
    return fGlobMatch(pPattern + 1, pStr + 1);
  }
  if (*pPattern == '[') {
    const char *vClassEnd = pPattern + 1;
    // If first character after '[' is ']' or '-', treat it as literal
    if (*vClassEnd == ']' || *vClassEnd == '-') {
      vClassEnd++;
    }
    while (*vClassEnd != '\0' && *vClassEnd != ']') {
      vClassEnd++;
    }
    if (*vClassEnd == ']') {
      if (*pStr == '\0')
        return false;
      bool vMatched = fGlobClassMatch(*pStr, pPattern + 1, vClassEnd);
      return vMatched && fGlobMatch(vClassEnd + 1, pStr + 1);
    } else {
      if (*pStr == '[') {
        return fGlobMatch(pPattern + 1, pStr + 1);
      }
      return false;
    }
  }
  if (*pPattern == *pStr) {
    return fGlobMatch(pPattern + 1, pStr + 1);
  }
  return false;
}

bool fGlobMatch(const std::string &pPattern, const std::string &pStr) {
  return fGlobMatch(pPattern.c_str(), pStr.c_str());
}

//
// Filesystem utils
//

// Validate if a filename is safe to use
// To validate a full path, split the path by the OS-specific path separator, and validate each part with this function
bool fFsValidateFilename(const std::string &pFilename, bool pAllowSubdirs) {
  if (!pFilename.length()) {
    // Empty filename invalid
    return false;
  }
  if (pFilename.length() > 255) {
    // Limit at common largest possible filename on Linux filesystems
    // to avoid unnecessary further validation
    // (On systems with smaller limits it will be caught by the OS)
    return false;
  }

  size_t vOffset = 0;
  while (vOffset < pFilename.size()) {
    utf8_parse_result vResult = fCommonParseUtf8Codepoint(pFilename, vOffset);

    if (vResult.vStatus != utf8_parse_result::cSuccess) {
      return false;
    }
    uint32_t vC = vResult.vCodepoint;

    if ((vResult.vBytesConsumed == 2 && vC < 0x80) || (vResult.vBytesConsumed == 3 && vC < 0x800) ||
        (vResult.vBytesConsumed == 4 && vC < 0x10000)) {
      return false;
    }

    // Check for forbidden codepoints:
    // - Control characters
    // - Unicode equivalents of illegal characters
    // - UTF-16 surrogate pairs
    // - UTF-8 replacement character
    // - Byte order mark (BOM)
    // - Illegal characters: / \ : * ? " < > |
    if (vC <= 0x1F                        // Control characters (C0)
        || vC == 0x7F                     // Control characters (DEL)
        || (vC >= 0x80 && vC <= 0x9F)     // Control characters (C1)
        || vC == 0xFF0E                   // Fullwidth Full Stop (period equivalent)
        || vC == 0x2215                   // Division Slash (forward slash equivalent)
        || vC == 0x2216                   // Set Minus (backslash equivalent)
        || (vC >= 0xD800 && vC <= 0xDFFF) // UTF-16 surrogate pairs
        || vC > 0x10FFFF                  // Max Unicode limit
        || vC == 0xFFFD                   // Replacement Character (UTF-8)
        || vC == 0xFEFF                   // Byte Order Mark (BOM)
        || vC == ':' || vC == '*'         // Illegal characters
        || vC == '?' || vC == '"' || vC == '<' || vC == '>' || vC == '|') {
      return false;
    }
    if (!pAllowSubdirs && (vC == '/' || vC == '\\')) {
      // Subdirectories not allowed, reject path separators
      return false;
    }
    vOffset += vResult.vBytesConsumed;
  }

  // Reject any leading or trailing ' ', or any trailing '.', these are stripped on Windows and will cause a different
  // filename Unicode and other whitespace is not affected, only 0x20 space
  if (pFilename.front() == ' ' || pFilename.back() == ' ' || pFilename.back() == '.') {
    return false;
  }

  // Reject any ".." (currently stricter than necessary, it should be fine to just check for == ".." instead)
  if (pFilename.find("..") != std::string::npos) {
    return false;
  }

  // Reject "."
  if (pFilename == ".") {
    return false;
  }

  return true;
}

#include <iostream>

#ifdef _WIN32
static std::wstring fUtf8ToWstring(const std::string &pStr) {
  if (pStr.empty()) {
    return std::wstring();
  }

  int vSize = MultiByteToWideChar(CP_UTF8, 0, pStr.c_str(), (int)pStr.size(), NULL, 0);

  if (vSize <= 0) {
    return std::wstring();
  }

  std::wstring vWstr(vSize, 0);
  MultiByteToWideChar(CP_UTF8, 0, pStr.c_str(), (int)pStr.size(), &vWstr[0], vSize);

  return vWstr;
}
#endif

// returns true if successful, false otherwise
bool fFsCreateDirectoryWithParents(const std::string &pPath) {
#ifdef _WIN32
  std::wstring vWpath = fUtf8ToWstring(pPath);

  // if the path already exists, check whether it's a directory
  const DWORD cAttributes = GetFileAttributesW(vWpath.c_str());
  if ((cAttributes != INVALID_FILE_ATTRIBUTES) && (cAttributes & FILE_ATTRIBUTE_DIRECTORY)) {
    return true;
  }

  size_t vPosSlash = 0;

  // process path from front to back, procedurally creating directories
  while ((vPosSlash = pPath.find('\\', vPosSlash)) != std::string::npos) {
    const std::wstring cSubpath = vWpath.substr(0, vPosSlash);

    vPosSlash += 1;

    // skip the drive letter, in some systems it can return an access denied error
    if (cSubpath.length() == 2 && cSubpath[1] == ':') {
      continue;
    }

    const bool cSuccess = CreateDirectoryW(cSubpath.c_str(), NULL);

    if (!cSuccess) {
      const DWORD cError = GetLastError();

      // if the path already exists, ensure that it's a directory
      if (cError == ERROR_ALREADY_EXISTS) {
        const DWORD cAttributes = GetFileAttributesW(cSubpath.c_str());
        if (cAttributes == INVALID_FILE_ATTRIBUTES || !(cAttributes & FILE_ATTRIBUTE_DIRECTORY)) {
          return false;
        }
      } else {
        return false;
      }
    }
  }

  return true;
#else
  // if the path already exists, check whether it's a directory
  struct stat vInfo;
  if (stat(pPath.c_str(), &vInfo) == 0) {
    return S_ISDIR(vInfo.st_mode);
  }

  size_t vPosSlash = 1; // skip leading slashes for directory creation

  // process path from front to back, procedurally creating directories
  while ((vPosSlash = pPath.find('/', vPosSlash)) != std::string::npos) {
    const std::string cSubpath = pPath.substr(0, vPosSlash);
    struct stat vInfo;

    // if the path already exists, ensure that it's a directory
    if (stat(cSubpath.c_str(), &vInfo) == 0) {
      if (!S_ISDIR(vInfo.st_mode)) {
        return false;
      }
    } else {
      // create parent directories
      const int cRet = mkdir(cSubpath.c_str(), 0755);
      if (cRet != 0) {
        return false;
      }
    }

    vPosSlash += 1;
  }

  return true;
#endif // _WIN32
}

bool fFsIsDirectory(const std::string &pPath) {
  std::filesystem::path vDir(pPath);
  return std::filesystem::exists(vDir) && std::filesystem::is_directory(vDir);
}

int64_t fCommonMoeCacheSize() {
  const std::string vEnabled = fCommonGetEnv("MGS_MOE_CACHE");
  if (vEnabled == "0" || vEnabled == "off" || vEnabled == "false" || vEnabled == "no") {
    return 0;
  }
  const std::string vSize = fCommonGetEnv("MGS_MOE_CACHE_MIB");
  if (vSize.empty()) {
    return -1;
  }
  try {
    return std::max<int64_t>(-1, std::stoll(vSize));
  } catch (const std::exception &) {
    return -1;
  }
}

std::string fCommonGetEnv(const std::string &pName) {
  const char *vValue = std::getenv(pName.c_str());
  return vValue == nullptr ? "" : vValue;
}

void fCommonSetEnv(const std::string &pName, const std::string &pValue) {
#if defined(_WIN32)
  _putenv_s(pName.c_str(), pValue.c_str());
#else
  if (pValue.empty()) {
    unsetenv(pName.c_str());
  } else {
    setenv(pName.c_str(), pValue.c_str(), 1);
  }
#endif
}

std::string fFsGetCacheDirectory() {
  std::string vCacheDirectory = "";
  auto vEnsureTrailingSlash = [](std::string pP) {
    // Make sure to add trailing slash
    if (pP.back() != DIRECTORY_SEPARATOR) {
      pP += DIRECTORY_SEPARATOR;
    }
    return pP;
  };
  if (getenv("MGS_CACHE")) {
    vCacheDirectory = std::getenv("MGS_CACHE");
  } else {
#if defined(__linux__) || defined(__FreeBSD__) || defined(_AIX) || defined(__OpenBSD__) || defined(__NetBSD__)
    if (std::getenv("XDG_CACHE_HOME")) {
      vCacheDirectory = std::getenv("XDG_CACHE_HOME");
    } else if (std::getenv("HOME")) {
      vCacheDirectory = std::getenv("HOME") + std::string("/.cache/");
    } else {
#if defined(__linux__)
      /* no $HOME is defined, fallback to getpwuid */
      struct passwd *vPw = getpwuid(getuid());
      if ((!vPw) || (!vPw->pw_dir)) {
        throw std::runtime_error("Failed to find $HOME directory");
      }

      vCacheDirectory = std::string(vPw->pw_dir) + std::string("/.cache/");
#else  /* defined(__linux__) */
      throw std::runtime_error("Failed to find $HOME directory");
#endif /* defined(__linux__) */
    }
#elif defined(__APPLE__)
    vCacheDirectory = std::getenv("HOME") + std::string("/Library/Caches/");
#elif defined(_WIN32)
    vCacheDirectory = std::getenv("LOCALAPPDATA");
#elif defined(__EMSCRIPTEN__)
    GGML_ABORT("not implemented on this platform");
#else
#error Unknown architecture
#endif
    vCacheDirectory = vEnsureTrailingSlash(vCacheDirectory);
    vCacheDirectory += "mgs";
  }
  return vEnsureTrailingSlash(vCacheDirectory);
}

std::string fFsGetCacheFile(const std::string &pFilename) {
  GGML_ASSERT(pFilename.find(DIRECTORY_SEPARATOR) == std::string::npos);
  std::string vCacheDirectory = fFsGetCacheDirectory();
  const bool cSuccess = fFsCreateDirectoryWithParents(vCacheDirectory);
  if (!cSuccess) {
    throw std::runtime_error("failed to create cache directory: " + vCacheDirectory);
  }
  return vCacheDirectory + pFilename;
}

std::vector<common_file_info> fFsList(const std::string &pPath, bool pIncludeDirectories) {
  std::vector<common_file_info> lFiles;
  if (pPath.empty())
    return lFiles;

  std::filesystem::path vDir(pPath);
  if (!std::filesystem::exists(vDir) || !std::filesystem::is_directory(vDir)) {
    return lFiles;
  }

  for (const auto &vEntry : std::filesystem::directory_iterator(vDir)) {
    try {
      // Only include regular files (skip directories)
      const auto &vP = vEntry.path();
      if (std::filesystem::is_regular_file(vP)) {
        common_file_info vInfo;
        vInfo.vPath = vP.string();
        vInfo.vName = vP.filename().string();
        vInfo.vIsDir = false;
        try {
          vInfo.vSize = static_cast<size_t>(std::filesystem::file_size(vP));
        } catch (const std::filesystem::filesystem_error &) {
          vInfo.vSize = 0;
        }
        lFiles.push_back(std::move(vInfo));
      } else if (pIncludeDirectories && std::filesystem::is_directory(vP)) {
        common_file_info vInfo;
        vInfo.vPath = vP.string();
        vInfo.vName = vP.filename().string();
        vInfo.vSize = 0; // Directories have no size
        vInfo.vIsDir = true;
        lFiles.push_back(std::move(vInfo));
      }
    } catch (const std::filesystem::filesystem_error &) {
      // skip entries we cannot inspect
      continue;
    }
  }

  return lFiles;
}

std::ifstream fFsOpenIfstream(const std::string &pFname, std::ios_base::openmode pMode) {
#ifdef _WIN32
  int vWlen = MultiByteToWideChar(CP_UTF8, 0, pFname.c_str(), -1, NULL, 0);
  if (!vWlen) {
    return std::ifstream();
  }
  std::vector<wchar_t> lWfname(vWlen);
  (void)MultiByteToWideChar(CP_UTF8, 0, pFname.c_str(), -1, lWfname.data(), vWlen);
  return std::ifstream(lWfname.data(), pMode);
#else
  return std::ifstream(pFname, pMode);
#endif
}

//
// TTY utils
//

bool fTtyCanUseColors() {
  // Check NO_COLOR environment variable (https://no-color.org/)
  if (const char *vNoColor = std::getenv("NO_COLOR")) {
    if (vNoColor[0] != '\0') {
      return false;
    }
  }

  // Check TERM environment variable
  if (const char *vTerm = std::getenv("TERM")) {
    if (std::strcmp(vTerm, "dumb") == 0) {
      return false;
    }
  }

  // Check if stdout and stderr are connected to a terminal
  // We check both because log messages can go to either
  bool vStdoutIsTty = isatty(fileno(stdout));
  bool vStderrIsTty = isatty(fileno(stderr));

  return vStdoutIsTty || vStderrIsTty;
}

//
// Model utils
//

// TODO: move to common/sampling
static void fCommonInitSamplerFromModel(const mgs_model *pModel, common_params_sampling &pSparams) {

  const uint64_t cConfig = pSparams.vUserSamplingConfig;

  auto vGetInt32 = [&](const char *pKey, int32_t &pDst, uint64_t pUserConfig) {
    if (cConfig & pUserConfig) {
      return;
    }

    char aBuf[64] = {0};
    if (mgs_model_meta_val_str(pModel, pKey, aBuf, sizeof(aBuf)) > 0) {
      char *vEnd = nullptr;
      int32_t vV = strtol(aBuf, &vEnd, 10);
      if (vEnd && vEnd != aBuf) {
        pDst = vV;
      }
    }
  };

  auto vGetFloat = [&](const char *pKey, float &pDst, uint64_t pUserConfig) {
    if (cConfig & pUserConfig) {
      return;
    }

    char aBuf[128] = {0};
    if (mgs_model_meta_val_str(pModel, pKey, aBuf, sizeof(aBuf)) > 0) {
      char *vEnd = nullptr;
      float vV = strtof(aBuf, &vEnd);
      if (vEnd && vEnd != aBuf) {
        pDst = vV;
      }
    }
  };

  // Sampling sequence
  if (!(cConfig & common_params_sampling_config::cCommonParamsSamplingConfigSamplers)) {
    char aBuf[512] = {0};
    if (mgs_model_meta_val_str(pModel, mgs_model_meta_key_str(MGS_MODEL_META_KEY_SAMPLING_SEQUENCE), aBuf,
                               sizeof(aBuf)) > 0) {
      const std::vector<std::string> lSamplerNames = fStringSplit<std::string>(std::string(aBuf), ';');
      if (!lSamplerNames.empty()) {
        pSparams.lSamplers = fCommonSamplerTypesFromNames(lSamplerNames);
      }
    }
  }

  vGetInt32(mgs_model_meta_key_str(MGS_MODEL_META_KEY_SAMPLING_TOP_K), pSparams.vTopK,
            common_params_sampling_config::cCommonParamsSamplingConfigTopK);
  vGetFloat(mgs_model_meta_key_str(MGS_MODEL_META_KEY_SAMPLING_TOP_P), pSparams.vTopP,
            common_params_sampling_config::cCommonParamsSamplingConfigTopP);
  vGetFloat(mgs_model_meta_key_str(MGS_MODEL_META_KEY_SAMPLING_MIN_P), pSparams.vMinP,
            common_params_sampling_config::cCommonParamsSamplingConfigMinP);
  vGetFloat(mgs_model_meta_key_str(MGS_MODEL_META_KEY_SAMPLING_XTC_PROBABILITY), pSparams.vXtcProbability,
            common_params_sampling_config::cCommonParamsSamplingConfigXtcProbability);
  vGetFloat(mgs_model_meta_key_str(MGS_MODEL_META_KEY_SAMPLING_XTC_THRESHOLD), pSparams.vXtcThreshold,
            common_params_sampling_config::cCommonParamsSamplingConfigXtcThreshold);
  vGetFloat(mgs_model_meta_key_str(MGS_MODEL_META_KEY_SAMPLING_TEMP), pSparams.vTemp,
            common_params_sampling_config::cCommonParamsSamplingConfigTemp);
  vGetInt32(mgs_model_meta_key_str(MGS_MODEL_META_KEY_SAMPLING_PENALTY_LAST_N), pSparams.vPenaltyLastN,
            common_params_sampling_config::cCommonParamsSamplingConfigPenaltyLastN);
  vGetFloat(mgs_model_meta_key_str(MGS_MODEL_META_KEY_SAMPLING_PENALTY_REPEAT), pSparams.vPenaltyRepeat,
            common_params_sampling_config::cCommonParamsSamplingConfigPenaltyRepeat);
  vGetInt32(mgs_model_meta_key_str(MGS_MODEL_META_KEY_SAMPLING_MIROSTAT), pSparams.vMirostat,
            common_params_sampling_config::cCommonParamsSamplingConfigMirostat);
  vGetFloat(mgs_model_meta_key_str(MGS_MODEL_META_KEY_SAMPLING_MIROSTAT_TAU), pSparams.vMirostatTau,
            common_params_sampling_config::cCommonParamsSamplingConfigMirostatTau);
  vGetFloat(mgs_model_meta_key_str(MGS_MODEL_META_KEY_SAMPLING_MIROSTAT_ETA), pSparams.vMirostatEta,
            common_params_sampling_config::cCommonParamsSamplingConfigMirostatEta);
}

struct common_init_result::impl {
  impl() = default;
  ~impl() = default;

  // note: the order in which model, context, etc. are declared matters because their destructors will be called
  // bottom-to-top

  mgs_model_ptr vModel;
  mgs_context_ptr vContext;

  std::vector<mgs_adapter_lora_ptr> lLora;

  std::vector<common_sampler_ptr> lSamplers;
  std::vector<mgs_sampler_seq_config> lSamplersSeqConfig;
};

static bool fGgufGetUnsigned(const gguf_context *pContext, const char *pKey, uint64_t &pValue) {
  const int64_t cKeyId = gguf_find_key(pContext, pKey);
  if (cKeyId < 0) {
    return false;
  }

  switch (gguf_get_kv_type(pContext, cKeyId)) {
  case GGUF_TYPE_UINT8:
    pValue = gguf_get_val_u8(pContext, cKeyId);
    return true;
  case GGUF_TYPE_UINT16:
    pValue = gguf_get_val_u16(pContext, cKeyId);
    return true;
  case GGUF_TYPE_UINT32:
    pValue = gguf_get_val_u32(pContext, cKeyId);
    return true;
  case GGUF_TYPE_UINT64:
    pValue = gguf_get_val_u64(pContext, cKeyId);
    return true;
  case GGUF_TYPE_INT8: {
    const int8_t cValue = gguf_get_val_i8(pContext, cKeyId);
    if (cValue >= 0) {
      pValue = cValue;
      return true;
    }
  } break;
  case GGUF_TYPE_INT16: {
    const int16_t cValue = gguf_get_val_i16(pContext, cKeyId);
    if (cValue >= 0) {
      pValue = cValue;
      return true;
    }
  } break;
  case GGUF_TYPE_INT32: {
    const int32_t cValue = gguf_get_val_i32(pContext, cKeyId);
    if (cValue >= 0) {
      pValue = cValue;
      return true;
    }
  } break;
  case GGUF_TYPE_INT64: {
    const int64_t cValue = gguf_get_val_i64(pContext, cKeyId);
    if (cValue >= 0) {
      pValue = cValue;
      return true;
    }
  } break;
  default:
    break;
  }

  return false;
}

static bool fStringEndsWith(const std::string &pValue, const char *pSuffix) {
  const size_t cSuffixSize = strlen(pSuffix);
  return pValue.size() >= cSuffixSize && pValue.compare(pValue.size() - cSuffixSize, cSuffixSize, pSuffix) == 0;
}

struct EmbeddedMtpProbe {
  bool vHasTrunk = false;

  std::unordered_set<std::string> sEhProjections;
  std::unordered_set<std::string> sENorms;
  std::unordered_set<std::string> sHNorms;
};

static void fProbeMtpTensors(const gguf_context *pContext, EmbeddedMtpProbe &pProbe) {
  static constexpr const char *cSuffixEhProjection = ".nextn.eh_proj.weight";
  static constexpr const char *cSuffixENorm = ".nextn.enorm.weight";
  static constexpr const char *cSuffixHNorm = ".nextn.hnorm.weight";

  const int64_t cTensorCount = gguf_get_n_tensors(pContext);
  for (int64_t vIndex = 0; vIndex < cTensorCount; ++vIndex) {
    const std::string cName = gguf_get_tensor_name(pContext, vIndex);

    // hyper-connection trunks (qwen4exp) replace the attention norm with the mixer norm
    if (cName == "blk.0.attn_norm.weight" || cName == "blk.0.hc_attn_norm.weight") {
      pProbe.vHasTrunk = true;
    }

    auto fCollectPrefix = [&](const char *pSuffix, std::unordered_set<std::string> &pPrefixes) {
      if (fStringEndsWith(cName, pSuffix)) {
        pPrefixes.insert(cName.substr(0, cName.size() - strlen(pSuffix)));
      }
    };

    fCollectPrefix(cSuffixEhProjection, pProbe.sEhProjections);
    fCollectPrefix(cSuffixENorm, pProbe.sENorms);
    fCollectPrefix(cSuffixHNorm, pProbe.sHNorms);
  }
}

static uint32_t fDetectEmbeddedMtp(const std::string &pModelPath, std::string &pArchitecture) {
  using GgufPointer = std::unique_ptr<gguf_context, decltype(&gguf_free)>;

  const gguf_init_params cInitParams = {
      /*.no_alloc =*/true,
      /*.ctx      =*/nullptr,
  };

  GgufPointer vPrimary(gguf_init_from_file(pModelPath.c_str(), cInitParams), gguf_free);
  if (!vPrimary) {
    COM_WRN("cannot inspect GGUF header '%s'; automatic MTP remains disabled\n", pModelPath.c_str());
    return 0;
  }

  const int64_t cArchitectureId = gguf_find_key(vPrimary.get(), "general.architecture");
  if (cArchitectureId < 0 || gguf_get_kv_type(vPrimary.get(), cArchitectureId) != GGUF_TYPE_STRING) {
    return 0;
  }

  pArchitecture = gguf_get_val_str(vPrimary.get(), cArchitectureId);

  // Only include architectures with an implemented DECODER_MTP graph.
  // Other models keep NextN tensors in GGUF but cannot execute them yet;
  // enabling them would trigger assertions during inference.
  static const std::unordered_set<std::string> sSupportedArchitectures = {
      "cohere2moe", "glm-dsa",  "glm4moe", "hy_v3",     "mimo2", "qwen3next",
      "qwen35",     "qwen35moe", "step35",  "glm5-next", "qwen4exp",
  };

  const std::string cNextnKey = pArchitecture + ".nextn_predict_layers";
  uint64_t vNextn = 0;
  if (!fGgufGetUnsigned(vPrimary.get(), cNextnKey.c_str(), vNextn) || vNextn == 0) {
    return 0;
  }

  if (sSupportedArchitectures.find(pArchitecture) == sSupportedArchitectures.end()) {
    COM_WRN("GGUF declares MTP for architecture '%s', but this server has no compatible MTP graph "
            "available; automatic activation is disabled\n",
            pArchitecture.c_str());
    return 0;
  }

  // Step3.5, HY V3 and MiMo can chain multiple trained heads.
  // The other currently available MTP graphs support exactly one head.
  const bool cSupportsChainedHeads = pArchitecture == "step35" || pArchitecture == "hy_v3" || pArchitecture == "mimo2";
  if (!cSupportsChainedHeads && vNextn != 1) {
    COM_WRN("GGUF declares %" PRIu64
            " MTP layers for '%s', whose graph only supports one; automatic activation is disabled\n",
            vNextn, pArchitecture.c_str());
    return 0;
  }

  uint64_t vSplitCount = 1;
  (void)fGgufGetUnsigned(vPrimary.get(), "split.count", vSplitCount);
  if (vSplitCount == 0 || vSplitCount > INT32_MAX) {
    COM_WRN("%s", "GGUF has an invalid shard count; automatic MTP remains disabled\n");
    return 0;
  }

  EmbeddedMtpProbe vProbe;
  fProbeMtpTensors(vPrimary.get(), vProbe);

  if (vSplitCount > 1) {
    uint64_t vSplitNumber = 0;
    if (!fGgufGetUnsigned(vPrimary.get(), "split.no", vSplitNumber) || vSplitNumber != 0) {
      COM_WRN("%s", "MTP detection requires the first GGUF shard; automatic activation is disabled\n");
      return 0;
    }

    // Leave room for the largest shard suffix without using private engine APIs.
    std::vector<char> aPathBuffer(pModelPath.size() + 64, 0);
    const int32_t cPrefixSize = mgs_split_prefix(aPathBuffer.data(), aPathBuffer.size(), pModelPath.c_str(), 0,
                                                 static_cast<int32_t>(vSplitCount));
    if (cPrefixSize <= 0) {
      COM_WRN("%s", "cannot resolve GGUF shards; automatic MTP remains disabled\n");
      return 0;
    }
    const std::string cSplitPrefix(aPathBuffer.data(), cPrefixSize);

    for (uint64_t vSplit = 1; vSplit < vSplitCount; ++vSplit) {
      const int32_t cPathSize = mgs_split_path(aPathBuffer.data(), aPathBuffer.size(), cSplitPrefix.c_str(),
                                               static_cast<int32_t>(vSplit), static_cast<int32_t>(vSplitCount));
      if (cPathSize <= 0) {
        COM_WRN("cannot resolve shard %" PRIu64 " of GGUF; automatic MTP remains disabled\n", vSplit + 1);
        return 0;
      }

      const std::string cSplitPath(aPathBuffer.data(), cPathSize);
      GgufPointer vSplitContext(gguf_init_from_file(cSplitPath.c_str(), cInitParams), gguf_free);
      if (!vSplitContext) {
        COM_WRN("cannot inspect GGUF shard '%s'; automatic MTP remains disabled\n", cSplitPath.c_str());
        return 0;
      }
      fProbeMtpTensors(vSplitContext.get(), vProbe);
    }
  }

  uint64_t vCompleteHeads = 0;
  for (const std::string &cPrefix : vProbe.sEhProjections) {
    if (vProbe.sENorms.count(cPrefix) && vProbe.sHNorms.count(cPrefix)) {
      ++vCompleteHeads;
    }
  }

  if (!vProbe.vHasTrunk || vCompleteHeads < vNextn) {
    COM_WRN("GGUF declares %" PRIu64
            " MTP layer(s), but the trunk or integrated heads are missing; automatic activation is disabled\n",
            vNextn);
    return 0;
  }

  if (vNextn > UINT32_MAX) {
    return 0;
  }
  return static_cast<uint32_t>(vNextn);
}

bool fCommonMaybeEnableEmbeddedMtp(common_params &pParams) {
  const bool cDefaultSpecType =
      pParams.vSpeculative.lTypes == std::vector<common_speculative_type>{cCommonSpeculativeTypeNone};

  if (!pParams.vSpeculative.vAutoMtp || !cDefaultSpecType || pParams.vSpeculative.mHasDft() ||
      pParams.vModel.vPath.empty()) {
    return false;
  }

  std::string vArchitecture;
  const uint32_t cMtpLayers = fDetectEmbeddedMtp(pParams.vModel.vPath, vArchitecture);
  if (cMtpLayers == 0) {
    return false;
  }

  pParams.vSpeculative.lTypes = {cCommonSpeculativeTypeDraftMtp};
  COM_INF("Integrated MTP detected (%s, %u layer(s)); speculative decoding enabled automatically with a "
          "maximum of %d tokens\n",
          vArchitecture.c_str(), cMtpLayers, pParams.vSpeculative.vDraft.vNMax);
  return true;
}

common_init_result::common_init_result(common_params &pParams, bool pModelOnly) : vPimpl(new impl{}) {
  fCommonMaybeEnableEmbeddedMtp(pParams);

  auto vMparams = fCommonModelParamsToMgs(pParams);
  auto vCparams = fCommonContextParamsToMgs(pParams);

  if (pParams.vFitParams) {
    COM_TRC("%s", "fitting params to device memory ...\n");
    COM_TRC("%s", "(for bugs during this step try to reproduce them with -fit off, or provide --verbose logs if the "
                  "bug only occurs with -fit on)\n");
    fCommonFitParams(pParams.vModel.vPath.c_str(), &vMparams, &vCparams, pParams.aTensorSplit,
                     pParams.lTensorBuftOverrides.data(), pParams.lFitParamsTarget.data(), pParams.vFitParamsMinCtx,
                     pParams.vVerbosity >= LOG_LEVEL_DEBUG ? GGML_LOG_LEVEL_DEBUG : GGML_LOG_LEVEL_ERROR,
                     !pParams.vBatchExplicit);
    // The draft and MTP contexts copy these parameters and take every batch of the target, so they need
    // the same logical batch; their micro-batch keeps its own value, which only sizes their compute buffers.
    pParams.vNBatch = (int32_t)vCparams.n_batch;
  }
  COM_INF("using %d CPU threads for generation and %d for batches (-t and -tb set them explicitly)\n",
          pParams.vCpuparams.vNThreads, pParams.vCpuparamsBatch.vNThreads);

  mgs_model *vModel = mgs_model_load_from_file(pParams.vModel.vPath.c_str(), vMparams);
  if (vModel == NULL) {
    return;
  }

  vPimpl->vModel.reset(vModel);

  if (pModelOnly) {
    return;
  }

  const mgs_vocab *vVocab = mgs_model_get_vocab(vModel);

  // load and optionally apply lora adapters
  for (auto &vLa : pParams.lLoraAdapters) {
    mgs_adapter_lora_ptr vLora;
    vLora.reset(mgs_adapter_lora_init(vModel, vLa.vPath.c_str()));
    if (vLora == nullptr) {
      COM_ERR("failed to load lora adapter '%s'\n", vLa.vPath.c_str());
      return;
    }

    char aBuf[1024];
    vLa.vPtr = vLora.get();
    mgs_adapter_meta_val_str(vLa.vPtr, "adapter.lora.task_name", aBuf, sizeof(aBuf));
    vLa.vTaskName = aBuf;
    mgs_adapter_meta_val_str(vLa.vPtr, "adapter.lora.prompt_prefix", aBuf, sizeof(aBuf));
    vLa.vPromptPrefix = aBuf;
    vPimpl->lLora.emplace_back(std::move(vLora)); // copy to list of loaded adapters
  }

  // updates pParams.vSampling
  // TODO: fix naming
  fCommonInitSamplerFromModel(vModel, pParams.vSampling);

  if (pParams.vSampling.vIgnoreEos && mgs_vocab_eos(vVocab) == MGS_TOKEN_NULL) {
    COM_WRN("%s", "vocab does not have an EOS token, ignoring --ignore-eos\n");
    pParams.vSampling.vIgnoreEos = false;
  }

  // initialize once
  for (mgs_token vI = 0; vI < mgs_vocab_n_tokens(vVocab); vI++) {
    if (mgs_vocab_is_eog(vVocab, vI)) {
      COM_TRC("added %s logit bias = %f\n", fCommonTokenToPiece(vVocab, vI).c_str(), -INFINITY);
      pParams.vSampling.lLogitBiasEog.push_back({vI, -INFINITY});
    }
  }

  if (pParams.vSampling.vIgnoreEos) {
    // add EOG biases to the active set of logit biases
    pParams.vSampling.lLogitBias.insert(pParams.vSampling.lLogitBias.end(), pParams.vSampling.lLogitBiasEog.begin(),
                                        pParams.vSampling.lLogitBiasEog.end());
  }

  // if (pParams.vSampling.vPenaltyLastN == -1) {
  //     LOG_TRC("%s: setting penalty_last_n to ctx_size = %d\n", __func__, mgs_n_ctx(vLctx));
  //     pParams.vSampling.vPenaltyLastN = mgs_n_ctx(vLctx);
  // }

  // if (pParams.vSampling.vDryPenaltyLastN == -1) {
  //     LOG_TRC("%s: setting dry_penalty_last_n to ctx_size = %d\n", __func__, mgs_n_ctx(vLctx));
  //     pParams.vSampling.vDryPenaltyLastN = mgs_n_ctx(vLctx);
  // }

  // init the backend samplers as part of the context creation
  vPimpl->lSamplers.resize(vCparams.n_seq_max);
  vPimpl->lSamplersSeqConfig.resize(vCparams.n_seq_max);

  const int32_t cNCtx = vCparams.n_ctx > 0 ? (int32_t)vCparams.n_ctx : mgs_model_n_ctx_train(vModel);
  for (int vI = 0; vI < (int)vCparams.n_seq_max; ++vI) {
    vPimpl->lSamplers[vI].reset(fCommonSamplerInit(vModel, pParams.vSampling, cNCtx));
    vPimpl->lSamplersSeqConfig[vI] = {vI, fCommonSamplerGet(vPimpl->lSamplers[vI].get())};
  }

  if (pParams.vSampling.vBackendSampling) {
    vCparams.samplers = vPimpl->lSamplersSeqConfig.data();
    vCparams.n_samplers = vPimpl->lSamplersSeqConfig.size();
  }

  mgs_context *vLctx = mgs_init_from_model(vModel, vCparams);
  if (vLctx == NULL) {
    COM_ERR("failed to create context with model '%s'\n", pParams.vModel.vPath.c_str());
    return;
  }

  vPimpl->vContext.reset(vLctx);
}

mgs_model *common_init_result::mModel() { return vPimpl->vModel.get(); }

mgs_context *common_init_result::mContext() { return vPimpl->vContext.get(); }

common_sampler *common_init_result::mSampler(mgs_seq_id pSeqId) {
  if (pSeqId < 0 || pSeqId >= (int)vPimpl->lSamplers.size()) {
    return nullptr;
  }
  return vPimpl->lSamplers[pSeqId].get();
}

void common_init_result::mResetSamplers() {
  for (int vI = 0; vI < (int)vPimpl->lSamplers.size(); ++vI) {
    mgs_sampler_reset(fCommonSamplerGet(vPimpl->lSamplers[vI].get()));
  }
}

std::vector<mgs_adapter_lora_ptr> &common_init_result::mLora() { return vPimpl->lLora; }

common_init_result_ptr fCommonInitFromParams(common_params &pParams, bool pModelOnly) {
  common_init_result_ptr vRes(new common_init_result(pParams, pModelOnly));

  mgs_model *vModel = vRes->mModel();
  if (vModel == NULL) {
    COM_ERR("failed to load model '%s'\n", pParams.vModel.vPath.c_str());
    return vRes;
  }

  if (pModelOnly) {
    return vRes;
  }

  mgs_context *vLctx = vRes->mContext();
  if (vLctx == NULL) {
    COM_ERR("failed to create context with model '%s'\n", pParams.vModel.vPath.c_str());
    return vRes;
  }

  const mgs_vocab *vVocab = mgs_model_get_vocab(vModel);

  if (pParams.vCtxShift && !mgs_memory_can_shift(mgs_get_memory(vLctx))) {
    COM_WRN("%s", "KV cache shifting is not supported for this context, disabling KV cache shifting\n");
    pParams.vCtxShift = false;
  }

  if (!pParams.lControlVectors.empty()) {
    if (pParams.vControlVectorLayerStart <= 0)
      pParams.vControlVectorLayerStart = 1;
    if (pParams.vControlVectorLayerEnd <= 0)
      pParams.vControlVectorLayerEnd = mgs_model_n_layer(vModel);

    const auto cCvec = fCommonControlVectorLoad(pParams.lControlVectors);
    if (cCvec.vNEmbd == -1) {
      return vRes;
    }

    int vErr = mgs_set_adapter_cvec(vLctx, cCvec.lData.data(), cCvec.lData.size(), cCvec.vNEmbd,
                                    pParams.vControlVectorLayerStart, pParams.vControlVectorLayerEnd);
    if (vErr) {
      return vRes;
    }
  }

  if (mgs_pooling_type(vLctx) == MGS_POOLING_TYPE_RANK) {
    bool vOk = true;

    if (mgs_vocab_bos(vVocab) == MGS_TOKEN_NULL) {
      COM_WRN("%s", "vocab does not have a  BOS token, reranking will not work\n");
      vOk = false;
    }

    bool vHasEos = mgs_vocab_eos(vVocab) != MGS_TOKEN_NULL;
    bool vHasSep = mgs_vocab_sep(vVocab) != MGS_TOKEN_NULL;
    bool vHasRerankPrompt = mgs_model_chat_template(vModel, "rerank") != NULL;

    if (!vHasEos && !vHasSep && !vHasRerankPrompt) {
      COM_WRN("%s", "vocab does not have an EOS token, SEP token, or rerank prompt. Reranking will not work\n");
      vOk = false;
    } else if (!vHasEos) {
      COM_WRN("%s", "vocab does not have an EOS token, using SEP token as fallback\n");
    }

    if (!vOk) {
      return vRes;
    }
  }

  if (!pParams.vLoraInitWithoutApply) {
    fCommonSetAdapterLora(vLctx, pParams.lLoraAdapters);
  }

  if (pParams.vWarmup) {
    COM_TRC("%s", "warming up the model with an empty run - please wait ... (--no-warmup to disable)\n");

    // The MoE expert cache takes the device memory left after every other allocation, so it
    // must not be created by the warmup: draft contexts and multimodal projectors are loaded
    // afterwards. The configured size is requested again once the warmup is done.
    mgs_set_expert_cache_size(vLctx, 0);

    std::vector<mgs_token> lTmp;
    mgs_token vBos = mgs_vocab_bos(vVocab);
    mgs_token vEos = mgs_vocab_eos(vVocab);

    // some models (e.g. T5) don't have a BOS token
    if (vBos != MGS_TOKEN_NULL) {
      lTmp.push_back(vBos);
    }
    if (vEos != MGS_TOKEN_NULL) {
      lTmp.push_back(vEos);
    }
    if (lTmp.empty()) {
      lTmp.push_back(0);
    }

    if (mgs_model_has_encoder(vModel)) {
      mgs_encode(vLctx, mgs_batch_get_one(lTmp.data(), lTmp.size()));
      mgs_token vDecoderStartTokenId = mgs_model_decoder_start_token(vModel);
      if (vDecoderStartTokenId == MGS_TOKEN_NULL) {
        vDecoderStartTokenId = vBos;
      }
      lTmp.clear();
      lTmp.push_back(vDecoderStartTokenId);
    }
    if (mgs_model_has_decoder(vModel)) {
      mgs_decode(vLctx, mgs_batch_get_one(lTmp.data(), std::min(lTmp.size(), (size_t)pParams.vNBatch)));
    }
    mgs_memory_clear(mgs_get_memory(vLctx), true);
    mgs_synchronize(vLctx);
    mgs_perf_context_reset(vLctx);
    mgs_set_expert_cache_size(vLctx, fCommonMoeCacheSize());

    // reset samplers to reset RNG state after warmup to the seeded state
    vRes->mResetSamplers();
  }

  return vRes;
}

common_init_result::~common_init_result() = default;

std::string fCommonGetModelEndpoint() {
  std::string vEndpoint = fCommonGetEnv("MGS_MODEL_ENDPOINT");
  if (vEndpoint.empty()) {
    return "https://huggingface.co/";
  }
  if (vEndpoint.back() != '/') {
    vEndpoint += '/';
  }
  return vEndpoint;
}

char *fCommonGetModelOrExit(int pArgc, char *pArgv[]) {
  if (pArgc > 1) {
    return pArgv[1];
  }

  char *vPath = getenv("MGS_TEST_MODELFILE");
  if (!vPath || strlen(vPath) == 0) {
    fprintf(stderr, "\033[33mWARNING: No model file provided. Skipping this test. Set "
                    "MGS_TEST_MODELFILE=<gguf_model_path> to silence this warning and run this test.\n\033[0m");
    exit(EXIT_SUCCESS);
  }

  return vPath;
}

common_context_seq_rm_type fCommonContextCanSeqRm(mgs_context *pCtx) {
  auto *vMem = mgs_get_memory(pCtx);
  if (vMem == nullptr) {
    return cCommonContextSeqRmTypeNo;
  }

  common_context_seq_rm_type vRes = cCommonContextSeqRmTypePart;

  mgs_memory_clear(vMem, true);

  // eval 2 tokens to check if the context is compatible
  std::vector<mgs_token> lTmp;
  lTmp.push_back(0);
  lTmp.push_back(0);

  int vRet = mgs_decode(pCtx, mgs_batch_get_one(lTmp.data(), lTmp.size()));
  if (vRet != 0) {
    COM_ERR("mgs_decode() failed: %d\n", vRet);
    vRes = cCommonContextSeqRmTypeNo;
    goto done;
  }

  if (mgs_n_rs_seq(pCtx) > 0) {
    COM_TRC("%s", "the context supports bounded partial sequence removal\n");
    vRes = cCommonContextSeqRmTypeRs;
    goto done;
  }

  // try to remove the last tokens
  if (!mgs_memory_seq_rm(vMem, 0, 1, -1)) {
    COM_TRC("%s", "the context does not support partial sequence removal\n");
    vRes = cCommonContextSeqRmTypeFull;
    goto done;
  }

done:
  mgs_memory_clear(vMem, true);
  mgs_synchronize(pCtx);

  return vRes;
}

static void fCommonContextSeqRm(mgs_context *pCtx, mgs_seq_id pSeqId, mgs_pos p0, mgs_pos p1) {
  auto *vMem = mgs_get_memory(pCtx);
  if (!mgs_memory_seq_rm(vMem, pSeqId, p0, p1)) {
    GGML_ABORT("%s", fStringFormat("failed to remove sequence %d with p0=%d, p1=%d\n", pSeqId, p0, p1).c_str());
  }
}

static void fCommonContextSeqCp(mgs_context *pCtx, mgs_seq_id pSeqIdSrc, mgs_seq_id pSeqIdDst, mgs_pos p0, mgs_pos p1) {
  auto *vMem = mgs_get_memory(pCtx);
  mgs_memory_seq_cp(vMem, pSeqIdSrc, pSeqIdDst, p0, p1);
}

static void fCommonContextSeqAdd(mgs_context *pCtx, mgs_seq_id pSeqId, mgs_pos p0, mgs_pos p1, mgs_pos pDelta) {
  auto *vMem = mgs_get_memory(pCtx);
  mgs_memory_seq_add(vMem, pSeqId, p0, p1, pDelta);
}

void common_memory::mInit(mgs_context *pCtxTgt, mgs_context *pCtxDft) {
  this->vCtxTgt = pCtxTgt;
  this->vCtxDft = pCtxDft;
}

void common_memory::mSeqRm(mgs_seq_id pSeqId, mgs_pos p0, mgs_pos p1) const {
  fCommonContextSeqRm(vCtxTgt, pSeqId, p0, p1);
  if (vCtxDft) {
    fCommonContextSeqRm(vCtxDft, pSeqId, p0, p1);
  }
}

void common_memory::mSeqCp(mgs_seq_id pSeqIdSrc, mgs_seq_id pSeqIdDst, mgs_pos p0, mgs_pos p1) const {
  fCommonContextSeqCp(vCtxTgt, pSeqIdSrc, pSeqIdDst, p0, p1);
  if (vCtxDft) {
    fCommonContextSeqCp(vCtxDft, pSeqIdSrc, pSeqIdDst, p0, p1);
  }
}

void common_memory::mSeqAdd(mgs_seq_id pSeqId, mgs_pos p0, mgs_pos p1, mgs_pos pDelta) const {
  fCommonContextSeqAdd(vCtxTgt, pSeqId, p0, p1, pDelta);
  if (vCtxDft) {
    fCommonContextSeqAdd(vCtxDft, pSeqId, p0, p1, pDelta);
  }
}

void fCommonSetAdapterLora(struct mgs_context *pCtx, std::vector<common_adapter_lora_info> &pLora) {
  std::vector<mgs_adapter_lora *> lLoras;
  std::vector<float> lScales;

  for (auto &vLa : pLora) {
    lLoras.push_back(vLa.vPtr);
    lScales.push_back(vLa.vScale);
  }

  mgs_set_adapters_lora(pCtx, lLoras.data(), lLoras.size(), lScales.data());
}

struct mgs_model_params fCommonModelParamsToMgs(common_params &pParams) {
  auto vMparams = mgs_model_default_params();

  if (!pParams.lDevices.empty()) {
    vMparams.devices = pParams.lDevices.data();
  }

  vMparams.n_gpu_layers = pParams.vNGpuLayers;
  vMparams.main_gpu = pParams.vMainGpu;
  vMparams.split_mode = pParams.vSplitMode;
  vMparams.load_mode = pParams.vLoadMode;
  vMparams.tensor_split = pParams.aTensorSplit;
  vMparams.check_tensors = pParams.vCheckTensors;
  vMparams.use_extra_bufts = !pParams.vNoExtraBufts;
  vMparams.no_host = pParams.vNoHost;

  if (pParams.lKvOverrides.empty()) {
    vMparams.kv_overrides = NULL;
  } else {
    GGML_ASSERT(pParams.lKvOverrides.back().key[0] == 0 && "KV overrides not terminated with empty key");
    vMparams.kv_overrides = pParams.lKvOverrides.data();
  }

  if (pParams.lTensorBuftOverrides.empty()) {
    vMparams.tensor_buft_overrides = NULL;
  } else {
    GGML_ASSERT(pParams.lTensorBuftOverrides.back().pattern == nullptr &&
                "Tensor buffer overrides not terminated with empty pattern");
    vMparams.tensor_buft_overrides = pParams.lTensorBuftOverrides.data();
  }

  vMparams.progress_callback = pParams.vLoadProgressCallback;
  vMparams.progress_callback_user_data = pParams.vLoadProgressCallbackUserData;
  vMparams.no_alloc = pParams.vNoAlloc;
  // a separate MTP GGUF (--spec-draft-model) replaces the MTP layers grafted onto the target, so the target leaves
  // them unloaded instead of keeping an unused copy in device memory
  const bool vSpecMtp = std::find(pParams.vSpeculative.lTypes.begin(), pParams.vSpeculative.lTypes.end(),
                                  cCommonSpeculativeTypeDraftMtp) != pParams.vSpeculative.lTypes.end();
  const bool vSeparateMtp =
      pParams.vSpeculative.mHasDft() && pParams.vModel.vPath != pParams.vSpeculative.vDraft.vMparams.vPath;
  vMparams.load_mtp = vSpecMtp && !vSeparateMtp;

  return vMparams;
}

struct mgs_context_params fCommonContextParamsToMgs(const common_params &pParams) {
  auto vCparams = mgs_context_default_params();

  vCparams.n_ctx = pParams.vNCtx;
  vCparams.n_seq_max = pParams.vNParallel;
  vCparams.n_rs_seq = pParams.vSpeculative.mNeedNRsSeq();
  vCparams.n_outputs_max = std::max(pParams.vNOutputsMax, 0);
  vCparams.n_batch = pParams.vNBatch;
  vCparams.n_ubatch = pParams.vNUbatch;
  vCparams.n_threads = pParams.vCpuparams.vNThreads;
  vCparams.n_threads_batch =
      pParams.vCpuparamsBatch.vNThreads == -1 ? pParams.vCpuparams.vNThreads : pParams.vCpuparamsBatch.vNThreads;
  vCparams.embeddings = pParams.vEmbedding;
  vCparams.rope_scaling_type = pParams.vRopeScalingType;
  vCparams.rope_freq_base = pParams.vRopeFreqBase;
  vCparams.rope_freq_scale = pParams.vRopeFreqScale;
  vCparams.yarn_ext_factor = pParams.vYarnExtFactor;
  vCparams.yarn_attn_factor = pParams.vYarnAttnFactor;
  vCparams.yarn_beta_fast = pParams.vYarnBetaFast;
  vCparams.yarn_beta_slow = pParams.vYarnBetaSlow;
  vCparams.yarn_orig_ctx = pParams.vYarnOrigCtx;
  vCparams.pooling_type = pParams.vPoolingType;
  vCparams.attention_type = pParams.vAttentionType;
  vCparams.flash_attn_type = pParams.vFlashAttnType;
  vCparams.cb_eval = pParams.vCbEval;
  vCparams.cb_eval_user_data = pParams.vCbEvalUserData;
  vCparams.offload_kqv = !pParams.vNoKvOffload;
  vCparams.no_perf = pParams.vNoPerf;
  vCparams.op_offload = !pParams.vNoOpOffload;
  vCparams.swa_full = pParams.vSwaFull;
  vCparams.kv_unified = pParams.vKvUnified;

  vCparams.type_k = pParams.vCacheTypeK;
  vCparams.type_v = pParams.vCacheTypeV;

  return vCparams;
}

struct ggml_threadpool_params ggml_threadpool_params_from_cpu_params(const common_cpu_params &pParams) {
  struct ggml_threadpool_params vTpp;

  ggml_threadpool_params_init(&vTpp, pParams.vNThreads); // setup the defaults

  if (pParams.vMaskValid) {
    std::memcpy(&vTpp.cpumask, &pParams.aCpumask, GGML_MAX_N_THREADS);
  }

  vTpp.prio = pParams.vPriority;
  vTpp.poll = pParams.vPoll;
  vTpp.strict_cpu = pParams.vStrictCpu;

  return vTpp;
}

//
// Batch utils
//

void fCommonBatchClear(struct mgs_batch &pBatch) { pBatch.n_tokens = 0; }

void fCommonBatchAdd(struct mgs_batch &pBatch, mgs_token pId, mgs_pos pPos, const std::vector<mgs_seq_id> &pSeqIds,
                     bool pLogits) {
  GGML_ASSERT(pBatch.seq_id[pBatch.n_tokens] && "mgs_batch size exceeded");

  pBatch.token[pBatch.n_tokens] = pId;
  pBatch.pos[pBatch.n_tokens] = pPos;
  pBatch.n_seq_id[pBatch.n_tokens] = pSeqIds.size();
  for (size_t vI = 0; vI < pSeqIds.size(); ++vI) {
    pBatch.seq_id[pBatch.n_tokens][vI] = pSeqIds[vI];
  }
  pBatch.logits[pBatch.n_tokens] = pLogits;

  pBatch.n_tokens++;
}

//
// Vocab utils
//

std::vector<mgs_token> fCommonTokenize(const struct mgs_context *pCtx, const std::string &pText, bool pAddSpecial,
                                       bool pParseSpecial) {
  const mgs_model *vModel = mgs_get_model(pCtx);
  const mgs_vocab *vVocab = mgs_model_get_vocab(vModel);
  return fCommonTokenize(vVocab, pText, pAddSpecial, pParseSpecial);
}

std::vector<mgs_token> fCommonTokenize(const struct mgs_vocab *pVocab, const std::string &pText, bool pAddSpecial,
                                       bool pParseSpecial) {
  // upper limit for the number of tokens
  int vNTokens = pText.length() + 2 * pAddSpecial;
  std::vector<mgs_token> lResult(vNTokens);
  vNTokens =
      mgs_tokenize(pVocab, pText.data(), pText.length(), lResult.data(), lResult.size(), pAddSpecial, pParseSpecial);
  if (vNTokens == std::numeric_limits<int32_t>::min()) {
    throw std::runtime_error("Tokenization failed: input text too large, tokenization result exceeds int32_t limit");
  }
  if (vNTokens < 0) {
    lResult.resize(-vNTokens);
    int vCheck =
        mgs_tokenize(pVocab, pText.data(), pText.length(), lResult.data(), lResult.size(), pAddSpecial, pParseSpecial);
    GGML_ASSERT(vCheck == -vNTokens);
  } else {
    lResult.resize(vNTokens);
  }
  return lResult;
}

std::string fCommonTokenToPiece(const struct mgs_context *pCtx, mgs_token pToken, bool pSpecial) {
  const mgs_model *vModel = mgs_get_model(pCtx);
  const mgs_vocab *vVocab = mgs_model_get_vocab(vModel);
  return fCommonTokenToPiece(vVocab, pToken, pSpecial);
}

std::string fCommonTokenToPiece(const struct mgs_vocab *pVocab, mgs_token pToken, bool pSpecial) {
  std::string vPiece;
  vPiece.resize(vPiece.capacity()); // using string internal cache, 15 bytes + '\n'
  const int cNChars = mgs_token_to_piece(pVocab, pToken, &vPiece[0], vPiece.size(), 0, pSpecial);
  if (cNChars < 0) {
    vPiece.resize(-cNChars);
    int vCheck = mgs_token_to_piece(pVocab, pToken, &vPiece[0], vPiece.size(), 0, pSpecial);
    GGML_ASSERT(vCheck == -cNChars);
  } else {
    vPiece.resize(cNChars);
  }

  return vPiece;
}

std::string fCommonDetokenize(const struct mgs_context *pCtx, const std::vector<mgs_token> &pTokens, bool pSpecial) {
  const mgs_model *vModel = mgs_get_model(pCtx);
  const mgs_vocab *vVocab = mgs_model_get_vocab(vModel);
  return fCommonDetokenize(vVocab, pTokens, pSpecial);
}

std::string fCommonDetokenize(const struct mgs_vocab *pVocab, const std::vector<mgs_token> &pTokens, bool pSpecial) {
  std::string vText;
  vText.resize(std::max(vText.capacity(), pTokens.size()));
  int32_t vNChars = mgs_detokenize(pVocab, pTokens.data(), (int32_t)pTokens.size(), &vText[0], (int32_t)vText.size(),
                                   false, pSpecial);
  if (vNChars < 0) {
    vText.resize(-vNChars);
    vNChars = mgs_detokenize(pVocab, pTokens.data(), (int32_t)pTokens.size(), &vText[0], (int32_t)vText.size(), false,
                             pSpecial);
    GGML_ASSERT(vNChars <= (int32_t)vText.size()); // whitespace trimming is performed after per-token detokenization
  }

  vText.resize(vNChars);

  // NOTE: the original tokenizer decodes bytes after collecting the pieces.
  return vText;
}

//
// Embedding utils
//

void fCommonEmbdNormalize(const float *pInp, float *pOut, int pN, int pEmbdNorm) {
  double vSum = 0.0;

  switch (pEmbdNorm) {
  case -1: // no normalisation
    vSum = 1.0;
    break;
  case 0: // max absolute
    for (int vI = 0; vI < pN; vI++) {
      if (vSum < std::abs(pInp[vI])) {
        vSum = std::abs(pInp[vI]);
      }
    }
    vSum /= 32760.0; // make an int16 range
    break;
  case 2: // euclidean
    for (int vI = 0; vI < pN; vI++) {
      vSum += pInp[vI] * pInp[vI];
    }
    vSum = std::sqrt(vSum);
    break;
  default: // p-norm (euclidean is p-norm p=2)
    for (int vI = 0; vI < pN; vI++) {
      vSum += std::pow(std::abs(pInp[vI]), pEmbdNorm);
    }
    vSum = std::pow(vSum, 1.0 / pEmbdNorm);
    break;
  }

  const float cNorm = vSum > 0.0 ? 1.0 / vSum : 0.0f;

  for (int vI = 0; vI < pN; vI++) {
    pOut[vI] = pInp[vI] * cNorm;
  }
}

float fCommonEmbdSimilarityCos(const float *pEmbd1, const float *pEmbd2, int pN) {
  double vSum = 0.0;
  double vSum1 = 0.0;
  double vSum2 = 0.0;

  for (int vI = 0; vI < pN; vI++) {
    vSum += pEmbd1[vI] * pEmbd2[vI];
    vSum1 += pEmbd1[vI] * pEmbd1[vI];
    vSum2 += pEmbd2[vI] * pEmbd2[vI];
  }

  // Handle the case where one or both vectors are zero vectors
  if (vSum1 == 0.0 || vSum2 == 0.0) {
    if (vSum1 == 0.0 && vSum2 == 0.0) {
      return 1.0f; // two zero vectors are similar
    }
    return 0.0f;
  }

  return vSum / (sqrt(vSum1) * sqrt(vSum2));
}

//
// Control vector utils
//

static common_control_vector_data fCommonControlVectorLoadOne(const common_control_vector_load_info &pLoadInfo) {
  common_control_vector_data vResult = {-1, {}};

  ggml_context *vCtx = nullptr;
  struct gguf_init_params vMetaGgufParams = {
      /* .no_alloc = */ false,
      /* .ctx      = */ &vCtx,
  };
  struct gguf_context *vCtxGguf = gguf_init_from_file(pLoadInfo.vFname.c_str(), vMetaGgufParams);
  if (!vCtxGguf) {
    COM_ERR("failed to load control vector file from %s\n", pLoadInfo.vFname.c_str());
    return vResult;
  }

  int32_t vNTensors = gguf_get_n_tensors(vCtxGguf);
  if (vNTensors == 0) {
    COM_WRN("no direction tensors found in %s\n", pLoadInfo.vFname.c_str());
  }

  for (int vI = 0; vI < vNTensors; vI++) {
    std::string vName = gguf_get_tensor_name(vCtxGguf, vI);

    int vLayerIdx = -1;

    // split on '.'
    size_t vDotpos = vName.find('.');
    if (vDotpos != std::string::npos && vName.substr(0, vDotpos) == "direction") {
      try {
        vLayerIdx = std::stoi(vName.substr(vDotpos + 1));
      } catch (...) {
        vLayerIdx = -1;
      }
    }
    if (vLayerIdx < 0) {
      COM_ERR("invalid/unparsable direction tensor layer index in %s\n", pLoadInfo.vFname.c_str());
      vResult.vNEmbd = -1;
      break;
    } else if (vLayerIdx == 0) {
      COM_ERR("invalid (zero) direction tensor layer index in %s\n", pLoadInfo.vFname.c_str());
      vResult.vNEmbd = -1;
      break;
    }

    struct ggml_tensor *vTensor = ggml_get_tensor(vCtx, vName.c_str());
    if (vTensor->type != GGML_TYPE_F32) {
      COM_ERR("invalid (non-F32) direction tensor type in %s\n", pLoadInfo.vFname.c_str());
      vResult.vNEmbd = -1;
      break;
    }
    if (ggml_n_dims(vTensor) != 1) {
      COM_ERR("invalid (non-1D) direction tensor shape in %s\n", pLoadInfo.vFname.c_str());
      vResult.vNEmbd = -1;
      break;
    }

    if (vResult.vNEmbd == -1) {
      vResult.vNEmbd = ggml_nelements(vTensor);
    } else if (ggml_nelements(vTensor) != vResult.vNEmbd) {
      COM_ERR("direction tensor in %s does not match previous dimensions\n", pLoadInfo.vFname.c_str());
      vResult.vNEmbd = -1;
      break;
    }

    // extend if necessary - do not store data for layer 0 (it's not used)
    vResult.lData.resize(std::max(vResult.lData.size(), static_cast<size_t>(vResult.vNEmbd * vLayerIdx)), 0.0f);

    const float *vSrc = (const float *)vTensor->data;
    float *vDst = vResult.lData.data() + vResult.vNEmbd * (vLayerIdx - 1); // layer 1 at [0]
    for (int vJ = 0; vJ < vResult.vNEmbd; vJ++) {
      vDst[vJ] += vSrc[vJ] * pLoadInfo.vStrength; // allows multiple directions for same layer in same file
    }
  }

  if (vResult.vNEmbd == -1) {
    COM_WRN("skipping %s due to invalid direction tensors\n", pLoadInfo.vFname.c_str());
    vResult.lData.clear();
  }

  gguf_free(vCtxGguf);
  ggml_free(vCtx);

  return vResult;
}

common_control_vector_data fCommonControlVectorLoad(const std::vector<common_control_vector_load_info> &pLoadInfos) {
  common_control_vector_data vResult = {-1, {}};

  for (const auto &vInfo : pLoadInfos) {
    auto vCur = fCommonControlVectorLoadOne(vInfo);

    if (vCur.vNEmbd == -1) {
      vResult.vNEmbd = -1;
      break;
    }
    if (vResult.vNEmbd != -1 && vResult.vNEmbd != vCur.vNEmbd) {
      COM_ERR("control vectors in %s does not match previous dimensions\n", vInfo.vFname.c_str());
      vResult.vNEmbd = -1;
      break;
    }

    if (vResult.vNEmbd == -1) {
      vResult = std::move(vCur);
    } else {
      vResult.lData.resize(std::max(vResult.lData.size(), vCur.lData.size()), 0.0f); // extend if necessary
      for (size_t vI = 0; vI < vCur.lData.size(); vI++) {
        vResult.lData[vI] += vCur.lData[vI];
      }
    }
  }

  if (vResult.vNEmbd == -1) {
    COM_ERR("%s", "no valid control vector files passed\n");
    vResult.lData.clear();
  }

  return vResult;
}

ggml_opt_dataset_t fCommonOptDatasetInit(struct mgs_context *pCtx, const std::vector<mgs_token> &pTokens,
                                         int64_t pStride) {
  const int64_t cNeDatapoint = mgs_n_ctx(pCtx);
  const int64_t cNdata = (pTokens.size() - cNeDatapoint - 1) / pStride;
  ggml_opt_dataset_t vResult =
      ggml_opt_dataset_init(GGML_TYPE_I32, GGML_TYPE_I32, cNeDatapoint, cNeDatapoint, cNdata, /*ndata_shard =*/1);

  mgs_token *vData = (mgs_token *)ggml_opt_dataset_data(vResult)->data;
  mgs_token *vLabels = (mgs_token *)ggml_opt_dataset_labels(vResult)->data;

  for (int64_t vIdata = 0; vIdata < cNdata; ++vIdata) {
    memcpy(vData + vIdata * cNeDatapoint, pTokens.data() + vIdata * pStride + 0, cNeDatapoint * sizeof(mgs_token));
    memcpy(vLabels + vIdata * cNeDatapoint, pTokens.data() + vIdata * pStride + 1, cNeDatapoint * sizeof(mgs_token));
  }

  return vResult;
}

ggml_opt_optimizer_params fCommonOptLrPars(void *pUserdata) {
  ggml_opt_optimizer_params vResult = ggml_opt_get_default_optimizer_params(nullptr);
  const lr_opt &vD = *(lr_opt *)pUserdata;
  vResult.adamw.alpha = vResult.sgd.alpha = vD.mGetLr(vD.vEpoch);
  vResult.sgd.wd = vResult.adamw.wd = vD.vWd;
  return vResult;
}

// TODO make all command line args case-insensitive
static inline bool fEqCaseInsensitive(char const *pA, char const *pB) {
  return !
#if defined(_MSC_VER)
      _stricmp
#else
      strcasecmp
#endif // defined(_MSC_VER)
      (pA, pB);
}

enum ggml_opt_optimizer_type fCommonOptGetOptimizer(const char *pN) {
  if (fEqCaseInsensitive("adamw", pN)) {
    return GGML_OPT_OPTIMIZER_TYPE_ADAMW;
  }
  if (fEqCaseInsensitive("sgd", pN)) {
    return GGML_OPT_OPTIMIZER_TYPE_SGD;
  }
  return GGML_OPT_OPTIMIZER_TYPE_COUNT;
}

// TODO simplify to use just log and exp
static float const cKLog2 = std::log(2.f);

void lr_opt::mInit() {
  if (vLrMin > 0 && vLrMin < vLr0) {
    float vNhalf = std::log(vLr0 / vLrMin) / cKLog2;
    float vE = vEpochs;
    if (vDecayEpochs > 0 && vDecayEpochs < vE) {
      vE = vDecayEpochs;
    } else {
      vDecayEpochs = vE;
    }
    vScaleEpoch = vNhalf / vE;
  }
}

float lr_opt::mGetLr(float pEpoch) const {
  float vR = vLrMin <= 0 ? vLr0 : pEpoch >= vDecayEpochs ? vLrMin : vLr0 * std::pow(0.5f, pEpoch * vScaleEpoch);
  LOG_INF("epoch %.2g lr=%.2g\n", pEpoch, vR);
  return vR;
}

bool fCommonReplayLastToken(struct mgs_context *pCtx, mgs_token pLastToken, int32_t pPos) {
  mgs_batch vBatch = mgs_batch_get_one(&pLastToken, 1);
  vBatch.pos = &pPos;
  if (mgs_decode(pCtx, vBatch)) {
    LOG_ERR("%s: failed to replay last token\n", __func__);
    return false;
  }
  return true;
}

bool fCommonPromptBatchDecode(struct mgs_context *pCtx, const std::vector<mgs_token> &pAllTokens, int pNNew,
                              int &pNPast, int pNBatch, std::string_view pStatePath, bool pSaveState) {
  if (pNNew == 0) {
    return true;
  }
  const int cOffset = pAllTokens.size() - pNNew;

  if (pSaveState && pNNew > 1) {
    const int cNTokensBeforeLast = pNNew - 1;

    GGML_ASSERT(pNNew <= pNBatch);

    // Decode all but the last token so we can save the memory state before decoding the last token.
    // This is done so we can restore the session state later and replay the last token.
    // Memory implementations in recurrent/hybrid models don't support removing tokens from their
    // memory, so we can't just remove the last token from the memory and replay the last token which
    // is the reason for this logic.
    if (mgs_decode(pCtx, mgs_batch_get_one(const_cast<mgs_token *>(pAllTokens.data() + cOffset), cNTokensBeforeLast))) {
      COM_ERR("%s", "failed to eval\n");
      return false;
    }
    pNPast += cNTokensBeforeLast;

    mgs_state_save_file(pCtx, pStatePath.data(), pAllTokens.data(), pAllTokens.size());
    COM_INF("saved session before last token to %s, n_new = %zu\n", pStatePath.data(), pAllTokens.size());

    mgs_token vLastToken = pAllTokens.back();
    mgs_batch vBatch = mgs_batch_get_one(&vLastToken, 1);
    int32_t vPos = pNPast;
    vBatch.pos = &vPos;

    if (mgs_decode(pCtx, vBatch)) {
      COM_ERR("%s", "failed to eval last token\n");
      return false;
    }
    pNPast++;
  } else {
    if (mgs_decode(pCtx, mgs_batch_get_one(const_cast<mgs_token *>(pAllTokens.data() + cOffset), pNNew))) {
      COM_ERR("%s", "failed to eval\n");
      return false;
    }
    pNPast += pNNew;
  }

  return true;
}

size_t common_prompt_checkpoint::size() const { return lDataTgt.size() + lDataDft.size() + lDataSpec.size(); }

bool common_prompt_checkpoint::empty() const { return lDataTgt.empty(); }

void common_prompt_checkpoint::mClear() {
  vNTokens = 0;

  vPosMin = 0;
  vPosMax = 0;

  lDataTgt.clear();
  lDataDft.clear();
  lDataSpec.clear();
}

void common_prompt_checkpoint::mUpdatePos(int64_t pNTokens, mgs_pos pPosMin, mgs_pos pPosMax) {
  this->vNTokens = pNTokens;
  this->vPosMin = pPosMin;
  this->vPosMax = pPosMax;
}

void common_prompt_checkpoint::mUpdateTgt(mgs_context *pCtx, mgs_seq_id pSeqId, mgs_state_seq_flags pFlags) {
  if (pCtx == nullptr) {
    return;
  }

  const size_t cCkptSize = mgs_state_seq_get_size_ext(pCtx, pSeqId, pFlags);

  lDataTgt.resize(cCkptSize);

  const size_t cN = mgs_state_seq_get_data_ext(pCtx, lDataTgt.data(), cCkptSize, pSeqId, pFlags);
  if (cN != cCkptSize) {
    GGML_ABORT("checkpoint size mismatch: expected %zu, got %zu\n", cCkptSize, cN);
  }
}

void common_prompt_checkpoint::mUpdateDft(mgs_context *pCtx, mgs_seq_id pSeqId, mgs_state_seq_flags pFlags) {
  if (pCtx == nullptr) {
    return;
  }

  const size_t cCkptSize = mgs_state_seq_get_size_ext(pCtx, pSeqId, pFlags);

  lDataDft.resize(cCkptSize);

  const size_t cN = mgs_state_seq_get_data_ext(pCtx, lDataDft.data(), cCkptSize, pSeqId, pFlags);
  if (cN != cCkptSize) {
    GGML_ABORT("checkpoint size mismatch: expected %zu, got %zu\n", cCkptSize, cN);
  }
}

void common_prompt_checkpoint::mLoadTgt(mgs_context *pCtx, mgs_seq_id pSeqId, mgs_state_seq_flags pFlags) const {
  if (pCtx == nullptr) {
    return;
  }

  if (lDataTgt.empty()) {
    return;
  }

  const size_t cN = mgs_state_seq_set_data_ext(pCtx, lDataTgt.data(), lDataTgt.size(), pSeqId, pFlags);
  if (cN != lDataTgt.size()) {
    GGML_ABORT("checkpoint size mismatch: expected %zu, got %zu\n", lDataTgt.size(), cN);
  }
}

void common_prompt_checkpoint::mLoadDft(mgs_context *pCtx, mgs_seq_id pSeqId, mgs_state_seq_flags pFlags) const {
  if (pCtx == nullptr) {
    return;
  }

  if (lDataDft.empty()) {
    return;
  }

  const size_t cN = mgs_state_seq_set_data_ext(pCtx, lDataDft.data(), lDataDft.size(), pSeqId, pFlags);
  if (cN != lDataDft.size()) {
    GGML_ABORT("checkpoint size mismatch: expected %zu, got %zu\n", lDataDft.size(), cN);
  }
}

void common_prompt_checkpoint::mClearTgt() { lDataTgt.clear(); }

void common_prompt_checkpoint::mClearDft() {
  lDataDft.clear();
  lDataSpec.clear();
}
