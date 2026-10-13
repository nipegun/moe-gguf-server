#include "server-monitor.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <map>
#include <mutex>
#include <set>
#include <sstream>
#include <string>
#include <thread>

#if defined(__linux__)
#include <dlfcn.h>
#endif

const char *const aMonitorSegments[] = {
    "mtp_weights", "expert_weights", "dense_weights", "draft_weights", "kv_cache", "draft_kv_cache",
    "compute",     "expert_cache",   "prefetch",      "prompt_cache",  "other",
};
const size_t cMonitorSegmentCount = sizeof(aMonitorSegments) / sizeof(aMonitorSegments[0]);

// Concurrent pollers (status bars of several browsers, the Monitor page) share one sample.
static constexpr int64_t cMonitorReuseMs = 750;

static int64_t fMonitorNowMs() {
  return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

static double fRoundPercent(double pValue) { return std::round(pValue * 10.0) / 10.0; }

//
// CPU
//

struct monitor_cpu_times {
  int vId = -1; // logical CPU number of the "cpuN" line; -1 for the aggregate "cpu" line
  uint64_t vIdle = 0;
  uint64_t vTotal = 0;
};

// Aggregate line first, then one entry per online logical CPU (offline CPUs have no line, so the numbers may skip).
static bool fReadCpuTimes(std::vector<monitor_cpu_times> &pLTimes) {
  pLTimes.clear();
  try {
    std::ifstream vFile("/proc/stat");
    std::string vLine;
    while (std::getline(vFile, vLine)) {
      if (vLine.rfind("cpu", 0) != 0) {
        break; // the cpu lines come first
      }
      std::istringstream vStream(vLine);
      std::string vName;
      // user nice system idle iowait irq softirq steal (guest time is already part of user and nice)
      uint64_t aValues[8] = {};
      vStream >> vName;
      for (uint64_t &vValue : aValues) {
        if (!(vStream >> vValue)) {
          vValue = 0;
          break;
        }
      }
      monitor_cpu_times vTimes;
      if (vName.size() > 3) {
        try {
          vTimes.vId = std::stoi(vName.substr(3));
        } catch (const std::exception &) {
          continue;
        }
      }
      vTimes.vIdle = aValues[3] + aValues[4];
      for (uint64_t vValue : aValues) {
        vTimes.vTotal += vValue;
      }
      pLTimes.push_back(vTimes);
    }
  } catch (const std::exception &) {
    pLTimes.clear();
  }
  return !pLTimes.empty();
}

static double fCpuPercent(const monitor_cpu_times &pBefore, const monitor_cpu_times &pAfter) {
  if (pAfter.vTotal <= pBefore.vTotal) {
    return 0.0;
  }
  const double vTotal = (double)(pAfter.vTotal - pBefore.vTotal);
  const double vIdle = pAfter.vIdle >= pBefore.vIdle ? (double)(pAfter.vIdle - pBefore.vIdle) : 0.0;
  return std::clamp(100.0 * (1.0 - vIdle / vTotal), 0.0, 100.0);
}

static std::string fCpuModel() {
  static const std::string vModel = [] {
    try {
      std::ifstream vFile("/proc/cpuinfo");
      std::string vLine;
      while (std::getline(vFile, vLine)) {
        // x86 uses "model name"; some ARM kernels only provide "Hardware" or "Processor"
        if (vLine.rfind("model name", 0) == 0 || vLine.rfind("Hardware", 0) == 0 || vLine.rfind("Processor", 0) == 0) {
          const size_t vColon = vLine.find(':');
          if (vColon != std::string::npos) {
            std::string vValue = vLine.substr(vColon + 1);
            vValue.erase(0, vValue.find_first_not_of(" \t"));
            if (!vValue.empty()) {
              return vValue;
            }
          }
        }
      }
    } catch (const std::exception &) {
    }
    return std::string();
  }();
  return vModel;
}

// "model name" of every logical CPU in /proc/cpuinfo. x86 lists it in every processor block, so each processor
// (socket) gets its own; other architectures may only have the global line of fCpuModel.
static const std::map<int, std::string> &fCpuModels() {
  static const std::map<int, std::string> dModels = [] {
    std::map<int, std::string> dResult;
    try {
      std::ifstream vFile("/proc/cpuinfo");
      std::string vLine;
      int vProcessor = -1;
      while (std::getline(vFile, vLine)) {
        const size_t vColon = vLine.find(':');
        if (vColon == std::string::npos) {
          continue;
        }
        std::string vKey = vLine.substr(0, vColon);
        vKey.erase(vKey.find_last_not_of(" \t") + 1);
        std::string vValue = vLine.substr(vColon + 1);
        vValue.erase(0, vValue.find_first_not_of(" \t"));
        if (vKey == "processor") {
          try {
            vProcessor = std::stoi(vValue);
          } catch (const std::exception &) {
            vProcessor = -1;
          }
        } else if (vKey == "model name" && vProcessor >= 0 && !vValue.empty()) {
          dResult[vProcessor] = vValue;
        }
      }
    } catch (const std::exception &) {
    }
    return dResult;
  }();
  return dModels;
}

// Physical processor (socket) of a logical CPU. Systems without topology, or that report -1, have a single one.
static int fCpuPackage(int pCpu) {
  try {
    std::ifstream vFile("/sys/devices/system/cpu/cpu" + std::to_string(pCpu) + "/topology/physical_package_id");
    int vPackage = 0;
    if (vFile >> vPackage && vPackage >= 0) {
      return vPackage;
    }
  } catch (const std::exception &) {
  }
  return 0;
}

struct monitor_cpu_processor {
  std::string vModel;
  monitor_cpu_times vBefore;
  monitor_cpu_times vAfter;
  json aCores = json::array();
};

static json fEmptyCpuJson() { return json{{"usage", nullptr}, {"logical_cpus", 0}, {"processors", json::array()}}; }

// Usage since the previous sample, global and for every physical processor (socket) with its logical CPUs. A missing
// or stale baseline (no poller for a while) is replaced by a fresh 250 ms interval, so the value never averages
// minutes of history. Logical CPUs are matched by number, so one that goes offline or online between two samples
// does not shift the others.
static json fCpuJson() {
  static std::mutex vMutex;
  static std::vector<monitor_cpu_times> lLPrevious;
  static int64_t vPreviousMs = 0;
  static json dLast;
  static int64_t vLastMs = 0;
  constexpr int64_t cStaleMs = 10000;

  std::lock_guard<std::mutex> vLock(vMutex);
  int64_t vNow = fMonitorNowMs();
  if (vLastMs != 0 && vNow - vLastMs < cMonitorReuseMs) {
    return dLast;
  }
  if (lLPrevious.empty() || vNow - vPreviousMs > cStaleMs) {
    if (!fReadCpuTimes(lLPrevious)) {
      return fEmptyCpuJson();
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(250));
    vNow = fMonitorNowMs();
  }
  std::vector<monitor_cpu_times> lLCurrent;
  if (!fReadCpuTimes(lLCurrent)) {
    return fEmptyCpuJson();
  }
  std::map<int, monitor_cpu_times> dBefore;
  for (const monitor_cpu_times &vTimes : lLPrevious) {
    dBefore[vTimes.vId] = vTimes;
  }

  // processors in package order, each with its logical CPUs in number order
  std::map<int, monitor_cpu_processor> dProcessors;
  const std::map<int, std::string> &dModels = fCpuModels();
  for (const monitor_cpu_times &vAfter : lLCurrent) {
    if (vAfter.vId < 0) {
      continue;
    }
    const auto vFound = dBefore.find(vAfter.vId);
    // a CPU that just came online starts from this sample and counts as idle until the next one
    const monitor_cpu_times vBefore = vFound != dBefore.end() ? vFound->second : vAfter;
    monitor_cpu_processor &dProcessor = dProcessors[fCpuPackage(vAfter.vId)];
    const auto vModel = dModels.find(vAfter.vId);
    if (dProcessor.vModel.empty() && vModel != dModels.end()) {
      dProcessor.vModel = vModel->second;
    }
    dProcessor.vBefore.vIdle += vBefore.vIdle;
    dProcessor.vBefore.vTotal += vBefore.vTotal;
    dProcessor.vAfter.vIdle += vAfter.vIdle;
    dProcessor.vAfter.vTotal += vAfter.vTotal;
    dProcessor.aCores.push_back({{"id", vAfter.vId}, {"usage", fRoundPercent(fCpuPercent(vBefore, vAfter))}});
  }
  json aProcessors = json::array();
  for (auto &[vPackage, dProcessor] : dProcessors) {
    aProcessors.push_back({
        {"id", vPackage},
        {"model", dProcessor.vModel.empty() ? fCpuModel() : dProcessor.vModel},
        {"usage", fRoundPercent(fCpuPercent(dProcessor.vBefore, dProcessor.vAfter))},
        {"logical_cpus", dProcessor.aCores.size()},
        {"cores", std::move(dProcessor.aCores)},
    });
  }

  const auto vAggregate = dBefore.find(-1);
  dLast = json{
      {"usage", vAggregate != dBefore.end() && lLCurrent[0].vId < 0
                    ? json(fRoundPercent(fCpuPercent(vAggregate->second, lLCurrent[0])))
                    : json(nullptr)},
      {"logical_cpus", lLCurrent.size() - 1},
      {"processors", aProcessors},
  };
  lLPrevious = std::move(lLCurrent);
  vPreviousMs = vNow;
  vLastMs = vNow;
  return dLast;
}

static json fLoadAverage() {
  try {
    std::ifstream vFile("/proc/loadavg");
    double aValues[3] = {};
    if (vFile >> aValues[0] >> aValues[1] >> aValues[2]) {
      return json::array({aValues[0], aValues[1], aValues[2]});
    }
  } catch (const std::exception &) {
  }
  return nullptr;
}

//
// RAM
//

// Values of /proc/meminfo or /proc/self/status lines, in bytes (the files use kB).
static std::map<std::string, uint64_t> fReadKilobyteFile(const char *pPath) {
  std::map<std::string, uint64_t> dValues;
  try {
    std::ifstream vFile(pPath);
    std::string vLine;
    while (std::getline(vFile, vLine)) {
      const size_t vColon = vLine.find(':');
      if (vColon == std::string::npos || vLine.find(" kB") == std::string::npos) {
        continue;
      }
      uint64_t vValue = 0;
      std::istringstream vStream(vLine.substr(vColon + 1));
      if (vStream >> vValue) {
        dValues[vLine.substr(0, vColon)] = vValue * 1024;
      }
    }
  } catch (const std::exception &) {
    dValues.clear();
  }
  return dValues;
}

static json fRamJson() {
  const auto dInfo = fReadKilobyteFile("/proc/meminfo");
  const auto fValue = [&dInfo](const char *pKey) -> uint64_t {
    const auto vEntry = dInfo.find(pKey);
    return vEntry == dInfo.end() ? 0 : vEntry->second;
  };
  const uint64_t vTotal = fValue("MemTotal");
  const uint64_t vAvailable = std::min(fValue("MemAvailable"), vTotal);
  const uint64_t vSwapTotal = fValue("SwapTotal");
  const uint64_t vSwapFree = std::min(fValue("SwapFree"), vSwapTotal);
  return json{
      {"total_bytes", vTotal},
      {"available_bytes", vAvailable},
      {"used_bytes", vTotal - vAvailable},
      {"swap_total_bytes", vSwapTotal},
      {"swap_used_bytes", vSwapTotal - vSwapFree},
  };
}

size_t fMonitorProcessRss() {
  const auto dStatus = fReadKilobyteFile("/proc/self/status");
  const auto vEntry = dStatus.find("VmRSS");
  return vEntry == dStatus.end() ? 0 : (size_t)vEntry->second;
}

//
// Kernel residency of the GPU memory
//

// Value of an fdinfo memory line ("2552 KiB"), in bytes.
static uint64_t fFdinfoBytes(const std::string &pValue) {
  std::istringstream vStream(pValue);
  uint64_t vNumber = 0;
  std::string vUnit;
  if (!(vStream >> vNumber)) {
    return 0;
  }
  vStream >> vUnit;
  if (vUnit == "KiB") {
    return vNumber * 1024;
  }
  if (vUnit == "MiB") {
    return vNumber * 1024 * 1024;
  }
  if (vUnit == "GiB") {
    return vNumber * 1024 * 1024 * 1024;
  }
  return vNumber;
}

json fMonitorDrmResidency() {
  json dResidency = json::object();
  try {
    std::error_code vError;
    std::set<std::string> sClients; // several descriptors can share one DRM client
    std::filesystem::directory_iterator vIterator("/proc/self/fdinfo", vError);
    for (; !vError && vIterator != std::filesystem::directory_iterator(); vIterator.increment(vError)) {
      std::ifstream vFile(vIterator->path());
      std::map<std::string, std::string> dFields;
      std::string vLine;
      while (std::getline(vFile, vLine)) {
        const size_t vColon = vLine.find(':');
        if (vColon == std::string::npos) {
          continue;
        }
        std::string vValue = vLine.substr(vColon + 1);
        vValue.erase(0, vValue.find_first_not_of(" \t"));
        dFields[vLine.substr(0, vColon)] = vValue;
      }
      if (!dFields.count("drm-driver") || !dFields.count("drm-pdev")) {
        continue;
      }
      const std::string vKey = fMonitorDeviceKey(dFields["drm-pdev"]);
      if (!sClients.insert(vKey + "/" + dFields["drm-client-id"]).second) {
        continue;
      }
      // newer kernels publish drm-resident-*, older ones only the deprecated drm-memory-*
      const auto fRegion = [&dFields](const char *pRegion) -> uint64_t {
        const std::string vResident = std::string("drm-resident-") + pRegion;
        const std::string vMemory = std::string("drm-memory-") + pRegion;
        if (dFields.count(vResident)) {
          return fFdinfoBytes(dFields[vResident]);
        }
        return dFields.count(vMemory) ? fFdinfoBytes(dFields[vMemory]) : 0;
      };
      json &dDevice = dResidency[vKey];
      if (!dDevice.is_object()) {
        dDevice = json{{"vram_bytes", 0}, {"gtt_bytes", 0}};
      }
      dDevice["vram_bytes"] = dDevice["vram_bytes"].get<uint64_t>() + fRegion("vram");
      dDevice["gtt_bytes"] = dDevice["gtt_bytes"].get<uint64_t>() + fRegion("gtt");
      if (dFields.count("amd-evicted-vram")) {
        dDevice["evicted_vram_bytes"] =
            dDevice.value("evicted_vram_bytes", (uint64_t)0) + fFdinfoBytes(dFields["amd-evicted-vram"]);
      }
    }
  } catch (const std::exception &) {
    // without fdinfo the monitor simply shows no residency
  }
  return dResidency;
}

//
// GPUs
//

// Minimal NVML declarations (nvml.h), resolved with dlopen so the build needs no NVML headers
// and machines without the NVIDIA driver simply report no NVIDIA GPU.
typedef int nvml_return_t; // nvmlReturn_t, 0 is NVML_SUCCESS
typedef struct nvml_device_st *nvml_device_t;

struct nvml_utilization_t {
  unsigned int vGpu;
  unsigned int vMemory;
};

struct nvml_memory_t {
  unsigned long long vTotal;
  unsigned long long vFree;
  unsigned long long vUsed;
};

struct nvml_pci_info_t {
  char aBusIdLegacy[16];
  unsigned int vDomain;
  unsigned int vBus;
  unsigned int vDevice;
  unsigned int vPciDeviceId;
  unsigned int vPciSubSystemId;
  char aBusId[32];
};

struct monitor_nvml {
  bool vReady = false;
  nvml_return_t (*fDeviceGetCount)(unsigned int *) = nullptr;
  nvml_return_t (*fDeviceGetHandleByIndex)(unsigned int, nvml_device_t *) = nullptr;
  nvml_return_t (*fDeviceGetName)(nvml_device_t, char *, unsigned int) = nullptr;
  nvml_return_t (*fDeviceGetPciInfo)(nvml_device_t, nvml_pci_info_t *) = nullptr;
  nvml_return_t (*fDeviceGetUtilizationRates)(nvml_device_t, nvml_utilization_t *) = nullptr;
  nvml_return_t (*fDeviceGetMemoryInfo)(nvml_device_t, nvml_memory_t *) = nullptr;
  nvml_return_t (*fDeviceGetTemperature)(nvml_device_t, int, unsigned int *) = nullptr;
  nvml_return_t (*fDeviceGetPowerUsage)(nvml_device_t, unsigned int *) = nullptr;
  nvml_return_t (*fDeviceGetEnforcedPowerLimit)(nvml_device_t, unsigned int *) = nullptr;
};

// Opened once for the life of the process; the driver library is never bundled (it must match the
// kernel module), so the system copy is used, like libcuda.
static const monitor_nvml &fNvml() {
  static const monitor_nvml vNvml = [] {
    monitor_nvml vApi;
#if defined(__linux__)
    void *pLibrary = dlopen("libnvidia-ml.so.1", RTLD_NOW | RTLD_LOCAL);
    if (pLibrary == nullptr) {
      return vApi;
    }
    const auto fSymbol = [pLibrary](const char *pName) { return dlsym(pLibrary, pName); };
    auto fInit = (nvml_return_t (*)())fSymbol("nvmlInit_v2");
    vApi.fDeviceGetCount = (decltype(vApi.fDeviceGetCount))fSymbol("nvmlDeviceGetCount_v2");
    vApi.fDeviceGetHandleByIndex = (decltype(vApi.fDeviceGetHandleByIndex))fSymbol("nvmlDeviceGetHandleByIndex_v2");
    vApi.fDeviceGetName = (decltype(vApi.fDeviceGetName))fSymbol("nvmlDeviceGetName");
    vApi.fDeviceGetPciInfo = (decltype(vApi.fDeviceGetPciInfo))fSymbol("nvmlDeviceGetPciInfo_v3");
    vApi.fDeviceGetUtilizationRates =
        (decltype(vApi.fDeviceGetUtilizationRates))fSymbol("nvmlDeviceGetUtilizationRates");
    vApi.fDeviceGetMemoryInfo = (decltype(vApi.fDeviceGetMemoryInfo))fSymbol("nvmlDeviceGetMemoryInfo");
    vApi.fDeviceGetTemperature = (decltype(vApi.fDeviceGetTemperature))fSymbol("nvmlDeviceGetTemperature");
    vApi.fDeviceGetPowerUsage = (decltype(vApi.fDeviceGetPowerUsage))fSymbol("nvmlDeviceGetPowerUsage");
    vApi.fDeviceGetEnforcedPowerLimit =
        (decltype(vApi.fDeviceGetEnforcedPowerLimit))fSymbol("nvmlDeviceGetEnforcedPowerLimit");
    if (fInit == nullptr || vApi.fDeviceGetCount == nullptr || vApi.fDeviceGetHandleByIndex == nullptr ||
        fInit() != 0) {
      return monitor_nvml();
    }
    vApi.vReady = true;
#endif
    return vApi;
  }();
  return vNvml;
}

static void fAddNvmlGpus(json &pAGpus) {
  const monitor_nvml &vNvml = fNvml();
  unsigned int vCount = 0;
  if (!vNvml.vReady || vNvml.fDeviceGetCount(&vCount) != 0) {
    return;
  }
  for (unsigned int vIndex = 0; vIndex < vCount; ++vIndex) {
    nvml_device_t vDevice = nullptr;
    if (vNvml.fDeviceGetHandleByIndex(vIndex, &vDevice) != 0) {
      continue;
    }
    json dGpu = {
        {"id", "nvml:" + std::to_string(vIndex)},
        {"name", "NVIDIA GPU"},
        {"source", "nvml"},
        {"usage", nullptr},
        {"memory_usage", nullptr},
        {"temperature_c", nullptr},
        {"power_w", nullptr},
        {"power_limit_w", nullptr},
        {"total_bytes", 0},
        {"used_bytes", 0},
    };
    char aName[96] = {};
    if (vNvml.fDeviceGetName && vNvml.fDeviceGetName(vDevice, aName, sizeof(aName)) == 0 && aName[0] != '\0') {
      dGpu["name"] = aName;
    }
    nvml_pci_info_t vPci = {};
    if (vNvml.fDeviceGetPciInfo && vNvml.fDeviceGetPciInfo(vDevice, &vPci) == 0) {
      // same "domain:bus:device.function" form as the device_id of the GGML backends
      char aId[48] = {};
      snprintf(aId, sizeof(aId), "%04x:%02x:%02x.0", vPci.vDomain, vPci.vBus, vPci.vDevice);
      dGpu["id"] = aId;
    }
    nvml_utilization_t vUtilization = {};
    if (vNvml.fDeviceGetUtilizationRates && vNvml.fDeviceGetUtilizationRates(vDevice, &vUtilization) == 0) {
      dGpu["usage"] = std::min(vUtilization.vGpu, 100u);
      dGpu["memory_usage"] = std::min(vUtilization.vMemory, 100u);
    }
    nvml_memory_t vMemory = {};
    if (vNvml.fDeviceGetMemoryInfo && vNvml.fDeviceGetMemoryInfo(vDevice, &vMemory) == 0) {
      dGpu["total_bytes"] = (uint64_t)vMemory.vTotal;
      dGpu["used_bytes"] = (uint64_t)std::min(vMemory.vUsed, vMemory.vTotal);
    }
    unsigned int vValue = 0;
    if (vNvml.fDeviceGetTemperature &&
        vNvml.fDeviceGetTemperature(vDevice, 0 /* NVML_TEMPERATURE_GPU */, &vValue) == 0) {
      dGpu["temperature_c"] = vValue;
    }
    if (vNvml.fDeviceGetPowerUsage && vNvml.fDeviceGetPowerUsage(vDevice, &vValue) == 0) {
      dGpu["power_w"] = std::round(vValue / 100.0) / 10.0; // milliwatts
    }
    if (vNvml.fDeviceGetEnforcedPowerLimit && vNvml.fDeviceGetEnforcedPowerLimit(vDevice, &vValue) == 0) {
      dGpu["power_limit_w"] = std::round(vValue / 100.0) / 10.0;
    }
    pAGpus.push_back(std::move(dGpu));
  }
}

static bool fReadSysfsNumber(const std::filesystem::path &pPath, long long &pValue) {
  try {
    std::ifstream vFile(pPath);
    return (bool)(vFile >> pValue);
  } catch (const std::exception &) {
    return false;
  }
}

static std::string fLowerCase(std::string pValue) {
  std::transform(pValue.begin(), pValue.end(), pValue.begin(), [](unsigned char pC) { return (char)std::tolower(pC); });
  return pValue;
}

// DRM drivers of GPUs. Display-only adapters (the BMC of a server, firmware or virtual framebuffers) are left out.
static const std::set<std::string> sGpuDrivers = {"amdgpu", "i915", "xe", "nouveau", "nvidia"};

// Every DRM card of a GPU driver that NVML did not already list, so each GPU gets its own card even when no model
// uses it. Only amdgpu publishes gpu_busy_percent and its VRAM; the others (Intel, NVIDIA without NVML) take the
// name and the memory from their inference backend once a model uses them.
static void fAddSysfsGpus(json &pAGpus) {
  try {
    std::error_code vError;
    std::filesystem::directory_iterator vIterator("/sys/class/drm", vError);
    for (; !vError && vIterator != std::filesystem::directory_iterator(); vIterator.increment(vError)) {
      const std::string vName = vIterator->path().filename().string();
      if (vName.rfind("card", 0) != 0 || vName.find('-') != std::string::npos) {
        continue; // connectors such as card0-DP-1
      }
      const std::filesystem::path vDevice = vIterator->path() / "device";
      std::error_code vLinkError;
      const std::string vDriver = std::filesystem::read_symlink(vDevice / "driver", vLinkError).filename().string();
      if (vLinkError || sGpuDrivers.count(vDriver) == 0) {
        continue;
      }
      long long vBusy = 0;
      const bool vHasBusy = fReadSysfsNumber(vDevice / "gpu_busy_percent", vBusy);
      std::string vId = vName;
      {
        std::ifstream vFile(vDevice / "uevent");
        std::string vLine;
        while (std::getline(vFile, vLine)) {
          if (vLine.rfind("PCI_SLOT_NAME=", 0) == 0) {
            vId = fLowerCase(vLine.substr(14));
          }
        }
      }
      const bool vListed = std::any_of(pAGpus.begin(), pAGpus.end(),
                                       [&vId](const json &pDGpu) { return pDGpu.value("id", std::string()) == vId; });
      if (vListed) {
        continue;
      }
      long long vTotal = 0;
      long long vUsed = 0;
      fReadSysfsNumber(vDevice / "mem_info_vram_total", vTotal);
      fReadSysfsNumber(vDevice / "mem_info_vram_used", vUsed);
      json dGpu = {
          {"id", vId},
          {"name", "GPU " + vName},
          {"source", "sysfs"},
          {"usage", vHasBusy ? json(std::clamp(vBusy, 0LL, 100LL)) : json(nullptr)},
          {"memory_usage", nullptr},
          {"temperature_c", nullptr},
          {"power_w", nullptr},
          {"power_limit_w", nullptr},
          {"total_bytes", (uint64_t)std::max(vTotal, 0LL)},
          {"used_bytes", (uint64_t)std::clamp(vUsed, 0LL, std::max(vTotal, 0LL))},
      };
      std::filesystem::directory_iterator vHwmon(vDevice / "hwmon", vError);
      for (; !vError && vHwmon != std::filesystem::directory_iterator(); vHwmon.increment(vError)) {
        long long vMilliDegrees = 0;
        if (fReadSysfsNumber(vHwmon->path() / "temp1_input", vMilliDegrees)) {
          dGpu["temperature_c"] = vMilliDegrees / 1000;
          break;
        }
      }
      vError.clear();
      pAGpus.push_back(std::move(dGpu));
    }
  } catch (const std::exception &) {
    // sysfs layouts vary; GPUs that cannot be read are left out
  }
}

static json fGpusJson() {
  static std::mutex vMutex;
  static json aLast = json::array();
  static int64_t vLastMs = 0;

  std::lock_guard<std::mutex> vLock(vMutex);
  const int64_t vNow = fMonitorNowMs();
  if (vLastMs != 0 && vNow - vLastMs < cMonitorReuseMs) {
    return aLast;
  }
  json aGpus = json::array();
  fAddNvmlGpus(aGpus);
  fAddSysfsGpus(aGpus);
  aLast = aGpus;
  vLastMs = vNow;
  return aGpus;
}

//
// Reports
//

json fMonitorUsageJson() {
  const json dCpu = fCpuJson();
  // physical processors without their logical CPUs, so the status bar can label them cpu-1, cpu-2...
  json aProcessors = json::array();
  for (const json &dProcessor : dCpu["processors"]) {
    aProcessors.push_back({{"id", dProcessor["id"]}, {"model", dProcessor["model"]}, {"usage", dProcessor["usage"]}});
  }
  json aGpus = json::array();
  for (const json &dGpu : fGpusJson()) {
    aGpus.push_back({{"id", dGpu["id"]}, {"name", dGpu["name"]}, {"usage", dGpu["usage"]}});
  }
  return json{{"cpu", {{"usage", dCpu["usage"]}, {"processors", aProcessors}}}, {"gpus", aGpus}};
}

// Lower case and without the "-vN" suffix that the CUDA backend adds to virtual devices of one physical GPU.
std::string fMonitorDeviceKey(const std::string &pId) {
  std::string vKey = fLowerCase(pId);
  const size_t vSuffix = vKey.find("-v");
  return vSuffix == std::string::npos ? vKey : vKey.substr(0, vSuffix);
}

static json fEmptySegments() {
  json dSegments = json::object();
  for (size_t vIndex = 0; vIndex < cMonitorSegmentCount; ++vIndex) {
    dSegments[aMonitorSegments[vIndex]] = 0;
  }
  return dSegments;
}

static void fAddSegments(json &pDTarget, const json &pDSource) {
  if (!pDSource.is_object()) {
    return;
  }
  for (const auto &[vKey, vValue] : pDSource.items()) {
    if (pDTarget.contains(vKey) && vValue.is_number_integer() && vValue.get<int64_t>() > 0) {
      pDTarget[vKey] = pDTarget[vKey].get<uint64_t>() + vValue.get<uint64_t>();
    }
  }
}

// "other" is the used memory that no segment explains (other processes, driver contexts,
// projector...); the rest of the total is free.
static void fFinishSegments(json &pDTarget) {
  json &dSegments = pDTarget["segments"];
  uint64_t vKnown = 0;
  for (const auto &[vKey, vValue] : dSegments.items()) {
    if (vKey != "other") {
      vKnown += vValue.get<uint64_t>();
    }
  }
  const uint64_t vTotal = pDTarget.value("total_bytes", (uint64_t)0);
  const uint64_t vUsed = pDTarget.value("used_bytes", (uint64_t)0);
  dSegments["other"] = vUsed > vKnown ? vUsed - vKnown : 0;
  const uint64_t vOccupied = std::max(vUsed, vKnown);
  pDTarget["free_bytes"] = vTotal > vOccupied ? vTotal - vOccupied : 0;
}

json fMonitorReport(const std::vector<json> &pLSnapshots) {
  json dCpu = fCpuJson();
  dCpu["load_average"] = fLoadAverage();

  json dRam = fRamJson();
  dRam["segments"] = fEmptySegments();

  json aGpus = fGpusJson();
  for (json &dGpu : aGpus) {
    dGpu["backends"] = json::array();
    dGpu["segments"] = fEmptySegments();
  }

  json aModels = json::array();
  for (const json &dSnapshot : pLSnapshots) {
    json dModel = {
        {"name", dSnapshot.value("model", std::string())},
        {"state", dSnapshot.value("state", std::string("unknown"))},
    };
    if (dSnapshot.contains("rss_bytes")) {
      dModel["rss_bytes"] = dSnapshot["rss_bytes"];
    }
    if (dSnapshot.contains("expert_cache")) {
      dModel["expert_cache"] = dSnapshot["expert_cache"];
    }
    aModels.push_back(std::move(dModel));
    if (dSnapshot.value("state", std::string()) != "ready") {
      continue;
    }

    for (const json &dDevice : dSnapshot.value("devices", json::array())) {
      const std::string vKey = fMonitorDeviceKey(dDevice.value("id", std::string()));
      json *pGpu = nullptr;
      for (json &dGpu : aGpus) {
        if (fMonitorDeviceKey(dGpu.value("id", std::string())) == vKey) {
          pGpu = &dGpu;
          break;
        }
      }
      if (pGpu == nullptr) {
        // no NVML/sysfs reading for this device: use what its backend reported
        const uint64_t vTotal = dDevice.value("total_bytes", (uint64_t)0);
        const uint64_t vFree = std::min(dDevice.value("free_bytes", (uint64_t)0), vTotal);
        aGpus.push_back({
            {"id", dDevice.value("id", std::string())},
            {"name", dDevice.value("description", dDevice.value("name", std::string()))},
            {"source", "backend"},
            {"usage", nullptr},
            {"memory_usage", nullptr},
            {"temperature_c", nullptr},
            {"power_w", nullptr},
            {"power_limit_w", nullptr},
            {"total_bytes", vTotal},
            {"used_bytes", vTotal - vFree},
            {"backends", json::array()},
            {"segments", fEmptySegments()},
        });
        pGpu = &aGpus.back();
      }
      // sysfs has no marketing name for most GPUs ("GPU card0"); the backend description has it
      if ((*pGpu).value("source", std::string()) == "sysfs" && !dDevice.value("description", std::string()).empty()) {
        (*pGpu)["name"] = dDevice.value("description", std::string());
      }
      // nor the memory of the GPUs that are not amdgpu
      if ((*pGpu).value("total_bytes", (uint64_t)0) == 0 && dDevice.value("total_bytes", (uint64_t)0) > 0) {
        const uint64_t vTotal = dDevice.value("total_bytes", (uint64_t)0);
        const uint64_t vFree = std::min(dDevice.value("free_bytes", (uint64_t)0), vTotal);
        (*pGpu)["total_bytes"] = vTotal;
        (*pGpu)["used_bytes"] = vTotal - vFree;
      }
      const std::string vBackend = dDevice.value("name", std::string());
      json &aBackends = (*pGpu)["backends"];
      if (!vBackend.empty() && std::find(aBackends.begin(), aBackends.end(), json(vBackend)) == aBackends.end()) {
        aBackends.push_back(vBackend);
      }
      fAddSegments((*pGpu)["segments"], dDevice.value("segments", json::object()));
      // kernel residency of what the model processes allocated on this GPU (summed over router children)
      if (dDevice.contains("residency") && dDevice["residency"].is_object()) {
        const json &dResidency = dDevice["residency"];
        json &dKernel = (*pGpu)["kernel"];
        if (!dKernel.is_object()) {
          dKernel = json{{"allocated_bytes", 0}, {"vram_bytes", 0}, {"gtt_bytes", 0}, {"evicted_vram_bytes", 0}};
        }
        uint64_t vAllocated = 0;
        const json dDeviceSegments = dDevice.value("segments", json::object());
        for (const auto &[vKey, vValue] : dDeviceSegments.items()) {
          vAllocated += vValue.is_number_unsigned() ? vValue.get<uint64_t>() : 0;
        }
        dKernel["allocated_bytes"] = dKernel["allocated_bytes"].get<uint64_t>() + vAllocated;
        dKernel["vram_bytes"] = dKernel["vram_bytes"].get<uint64_t>() + dResidency.value("vram_bytes", (uint64_t)0);
        dKernel["gtt_bytes"] = dKernel["gtt_bytes"].get<uint64_t>() + dResidency.value("gtt_bytes", (uint64_t)0);
        dKernel["evicted_vram_bytes"] =
            dKernel["evicted_vram_bytes"].get<uint64_t>() + dResidency.value("evicted_vram_bytes", (uint64_t)0);
      }
    }
    if (dSnapshot.contains("host")) {
      fAddSegments(dRam["segments"], dSnapshot["host"].value("segments", json::object()));
    }
  }

  for (json &dGpu : aGpus) {
    fFinishSegments(dGpu);
    // memory the models allocated on the GPU that the kernel does not keep in VRAM
    if (dGpu.contains("kernel")) {
      json &dKernel = dGpu["kernel"];
      const uint64_t vAllocated = dKernel["allocated_bytes"].get<uint64_t>();
      const uint64_t vVram = dKernel["vram_bytes"].get<uint64_t>();
      dKernel["outside_vram_bytes"] =
          std::max(dKernel["evicted_vram_bytes"].get<uint64_t>(), vAllocated > vVram ? vAllocated - vVram : 0);
    }
  }
  fFinishSegments(dRam);

  json aSegments = json::array();
  for (size_t vIndex = 0; vIndex < cMonitorSegmentCount; ++vIndex) {
    aSegments.push_back(aMonitorSegments[vIndex]);
  }

  return json{
      {"timestamp_ms",
       std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch())
           .count()},
      {"cpu", dCpu},
      {"ram", dRam},
      {"gpus", aGpus},
      {"models", aModels},
      {"segments", aSegments},
  };
}
