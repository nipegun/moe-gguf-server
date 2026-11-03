#include "ggml-backend-impl.h"
#include "ggml-backend.h"
#include "ggml-backend-dl.h"
#include "ggml-impl.h"
#include <algorithm>
#include <cstring>
#include <filesystem>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>
#include <cctype>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#elif defined(__APPLE__)
#include <mach-o/dyld.h>
#include <dlfcn.h>
#else
#include <dlfcn.h>
#include <unistd.h>
#endif

// Backend registry
#ifdef GGML_USE_CPU
#include "ggml-cpu.h"
#endif

#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif

#ifdef GGML_USE_VULKAN
#include "ggml-vulkan.h"
#endif

namespace fs = std::filesystem;

static std::string fPathStr(const fs::path &pPath) {
  try {
#if defined(__cpp_lib_char8_t)
    // C++20 and later: u8string() returns std::u8string
    const std::u8string u8str = pPath.u8string();
    return std::string(reinterpret_cast<const char *>(u8str.data()), u8str.size());
#else
    // C++17: u8string() returns std::string
    return pPath.u8string();
#endif
  } catch (...) {
    return std::string();
  }
}

struct ggml_backend_reg_entry {
  ggml_backend_reg_t vReg;
  dl_handle_ptr vHandle;
};

struct ggml_backend_registry {
  std::vector<ggml_backend_reg_entry> lBackends;
  std::vector<ggml_backend_dev_t> lDevices;

  ggml_backend_registry() {
#ifdef GGML_USE_CUDA
    register_backend(ggml_backend_cuda_reg());
#endif
#ifdef GGML_USE_VULKAN
    // Add runtime disable check
    if (getenv("GGML_DISABLE_VULKAN") == nullptr) {
      register_backend(ggml_backend_vk_reg());
    } else {
      GGML_LOG_DEBUG("Vulkan backend disabled by GGML_DISABLE_VULKAN environment variable\n");
    }
#endif

#ifdef GGML_USE_CPU
    register_backend(ggml_backend_cpu_reg());
#endif
  }

  ~ggml_backend_registry() {
    // FIXME: backends cannot be safely unloaded without a function to destroy all the backend resources,
    // since backend threads may still be running and accessing resources from the dynamic library
    for (auto &vEntry : lBackends) {
      if (vEntry.vHandle) {
        vEntry.vHandle.release(); // NOLINT
      }
    }
  }

  void register_backend(ggml_backend_reg_t pReg, dl_handle_ptr pHandle = nullptr) {
    if (!pReg) {
      return;
    }

    for (auto &vEntry : lBackends) {
      if (vEntry.vReg == pReg) {
        return;
      }
    }

#ifndef NDEBUG
    GGML_LOG_DEBUG("%s: registered backend %s (%zu devices)\n", __func__, ggml_backend_reg_name(pReg),
                   ggml_backend_reg_dev_count(pReg));
#endif
    lBackends.push_back({pReg, std::move(pHandle)});
    for (size_t vI = 0; vI < ggml_backend_reg_dev_count(pReg); vI++) {
      mRegisterDevice(ggml_backend_reg_dev_get(pReg, vI));
    }
  }

  void mRegisterDevice(ggml_backend_dev_t pDevice) {
    for (auto &vDev : lDevices) {
      if (vDev == pDevice) {
        return;
      }
    }

#ifndef NDEBUG
    GGML_LOG_DEBUG("%s: registered device %s (%s)\n", __func__, ggml_backend_dev_name(pDevice),
                   ggml_backend_dev_description(pDevice));
#endif
    lDevices.push_back(pDevice);
  }

  ggml_backend_reg_t mLoadBackend(const fs::path &pPath, bool pSilent) {
    dl_handle_ptr vHandle{dl_load_library(pPath)};
    if (!vHandle) {
      if (!pSilent) {
        GGML_LOG_ERROR("%s: failed to load %s: %s\n", __func__, fPathStr(pPath).c_str(), dl_error());
      }
      return nullptr;
    }

    auto vScoreFn = (ggml_backend_score_t)dl_get_sym(vHandle.get(), "ggml_backend_score");
    if (vScoreFn && vScoreFn() == 0) {
      if (!pSilent) {
        GGML_LOG_INFO("%s: backend %s is not supported on this system\n", __func__, fPathStr(pPath).c_str());
      }
      return nullptr;
    }

    auto vBackendInitFn = (ggml_backend_init_t)dl_get_sym(vHandle.get(), "ggml_backend_init");
    if (!vBackendInitFn) {
      if (!pSilent) {
        GGML_LOG_ERROR("%s: failed to find ggml_backend_init in %s\n", __func__, fPathStr(pPath).c_str());
      }
      return nullptr;
    }

    ggml_backend_reg_t vReg = vBackendInitFn();
    if (!vReg || vReg->api_version != GGML_BACKEND_API_VERSION) {
      if (!pSilent) {
        if (!vReg) {
          GGML_LOG_ERROR("%s: failed to initialize backend from %s: ggml_backend_init returned NULL\n", __func__,
                         fPathStr(pPath).c_str());
        } else {
          GGML_LOG_ERROR(
              "%s: failed to initialize backend from %s: incompatible API version (backend: %d, current: %d)\n",
              __func__, fPathStr(pPath).c_str(), vReg->api_version, GGML_BACKEND_API_VERSION);
        }
      }
      return nullptr;
    }

    GGML_LOG_INFO("%s: loaded %s backend from %s\n", __func__, ggml_backend_reg_name(vReg), fPathStr(pPath).c_str());

    register_backend(vReg, std::move(vHandle));

    return vReg;
  }

  void mUnloadBackend(ggml_backend_reg_t pReg, bool pSilent) {
    auto vIt = std::find_if(lBackends.begin(), lBackends.end(),
                            [pReg](const ggml_backend_reg_entry &pEntry) { return pEntry.vReg == pReg; });

    if (vIt == lBackends.end()) {
      if (!pSilent) {
        GGML_LOG_ERROR("%s: backend not found\n", __func__);
      }
      return;
    }

    if (!pSilent) {
      GGML_LOG_DEBUG("%s: unloading %s backend\n", __func__, ggml_backend_reg_name(pReg));
    }

    // remove devices
    lDevices.erase(
        std::remove_if(lDevices.begin(), lDevices.end(),
                       [pReg](ggml_backend_dev_t pDev) { return ggml_backend_dev_backend_reg(pDev) == pReg; }),
        lDevices.end());

    // remove backend
    lBackends.erase(vIt);
  }
};

static ggml_backend_registry &fGetReg() {
  static ggml_backend_registry vReg;
  return vReg;
}

// Internal API
void ggml_backend_register(ggml_backend_reg_t pReg) { fGetReg().register_backend(pReg); }

void ggml_backend_device_register(ggml_backend_dev_t pDevice) { fGetReg().mRegisterDevice(pDevice); }

// Backend (reg) enumeration
static bool fStriequals(const char *pA, const char *pB) {
  for (; *pA && *pB; pA++, pB++) {
    if (std::tolower(*pA) != std::tolower(*pB)) {
      return false;
    }
  }
  return *pA == *pB;
}

size_t ggml_backend_reg_count() { return fGetReg().lBackends.size(); }

ggml_backend_reg_t ggml_backend_reg_get(size_t pIndex) {
  GGML_ASSERT(pIndex < ggml_backend_reg_count());
  return fGetReg().lBackends[pIndex].vReg;
}

ggml_backend_reg_t ggml_backend_reg_by_name(const char *pName) {
  for (size_t vI = 0; vI < ggml_backend_reg_count(); vI++) {
    ggml_backend_reg_t vReg = ggml_backend_reg_get(vI);
    if (fStriequals(ggml_backend_reg_name(vReg), pName)) {
      return vReg;
    }
  }
  return nullptr;
}

// Device enumeration
size_t ggml_backend_dev_count() { return fGetReg().lDevices.size(); }

ggml_backend_dev_t ggml_backend_dev_get(size_t pIndex) {
  GGML_ASSERT(pIndex < ggml_backend_dev_count());
  return fGetReg().lDevices[pIndex];
}

ggml_backend_dev_t ggml_backend_dev_by_name(const char *pName) {
  for (size_t vI = 0; vI < ggml_backend_dev_count(); vI++) {
    ggml_backend_dev_t vDev = ggml_backend_dev_get(vI);
    if (fStriequals(ggml_backend_dev_name(vDev), pName)) {
      return vDev;
    }
  }
  return nullptr;
}

ggml_backend_dev_t ggml_backend_dev_by_type(enum ggml_backend_dev_type pType) {
  for (size_t vI = 0; vI < ggml_backend_dev_count(); vI++) {
    ggml_backend_dev_t vDev = ggml_backend_dev_get(vI);
    if (ggml_backend_dev_type(vDev) == pType) {
      return vDev;
    }
  }
  return nullptr;
}

// Convenience functions
ggml_backend_t ggml_backend_init_by_name(const char *pName, const char *pParams) {
  ggml_backend_dev_t vDev = ggml_backend_dev_by_name(pName);
  if (!vDev) {
    return nullptr;
  }
  return ggml_backend_dev_init(vDev, pParams);
}

ggml_backend_t ggml_backend_init_by_type(enum ggml_backend_dev_type pType, const char *pParams) {
  ggml_backend_dev_t vDev = ggml_backend_dev_by_type(pType);
  if (!vDev) {
    return nullptr;
  }
  return ggml_backend_dev_init(vDev, pParams);
}

ggml_backend_t ggml_backend_init_best(void) {
  ggml_backend_dev_t vDev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
  vDev = vDev ? vDev : ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_IGPU);
  vDev = vDev ? vDev : ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
  if (!vDev) {
    return nullptr;
  }
  return ggml_backend_dev_init(vDev, nullptr);
}

// Dynamic loading
ggml_backend_reg_t ggml_backend_load(const char *pPath) { return fGetReg().mLoadBackend(pPath, false); }

void ggml_backend_unload(ggml_backend_reg_t pReg) { fGetReg().mUnloadBackend(pReg, true); }

static fs::path fGetExecutablePath() {
#if defined(__APPLE__)
  // get executable path
  std::vector<char> vPath;
  uint32_t size;
  while (true) {
    size = vPath.size();
    if (_NSGetExecutablePath(vPath.data(), &size) == 0) {
      break;
    }
    vPath.resize(size);
  }
  std::string vBasePath(vPath.data(), size);
  // remove executable name
  auto vLastSlash = vBasePath.find_last_of('/');
  if (vLastSlash != std::string::npos) {
    vBasePath = vBasePath.substr(0, vLastSlash);
  }
  return vBasePath + "/";
#elif defined(__linux__) || defined(__FreeBSD__)
  std::string vBasePath = ".";
  std::vector<char> vPath(1024);
  while (true) {
    // get executable path
#if defined(__linux__)
    ssize_t vLen = readlink("/proc/self/exe", vPath.data(), vPath.size());
#elif defined(__FreeBSD__)
    ssize_t vLen = readlink("/proc/curproc/file", vPath.data(), vPath.size());
#endif
    if (vLen == -1) {
      break;
    }
    if (vLen < (ssize_t)vPath.size()) {
      vBasePath = std::string(vPath.data(), vLen);
      // remove executable name
      auto vLastSlash = vBasePath.find_last_of('/');
      if (vLastSlash != std::string::npos) {
        vBasePath = vBasePath.substr(0, vLastSlash);
      }
      break;
    }
    vPath.resize(vPath.size() * 2);
  }

  return vBasePath + "/";
#elif defined(_WIN32)
  std::vector<wchar_t> vPath(MAX_PATH);
  DWORD vLen = GetModuleFileNameW(NULL, vPath.data(), vPath.size());
  if (vLen == 0) {
    return {};
  }
  std::wstring vBasePath(vPath.data(), vLen);
  // remove executable name
  auto vLastSlash = vBasePath.find_last_of('\\');
  if (vLastSlash != std::string::npos) {
    vBasePath = vBasePath.substr(0, vLastSlash);
  }
  return vBasePath + L"\\";
#else
  return {};
#endif
}

static fs::path fBackendFilenamePrefix() {
#ifdef _WIN32
  return fs::u8path("ggml-");
#else
  return fs::u8path("libggml-");
#endif
}

static fs::path fBackendFilenameExtension() {
#ifdef _WIN32
  return fs::u8path(".dll");
#else
  return fs::u8path(".so");
#endif
}

static ggml_backend_reg_t fGgmlBackendLoadBest(const char *pName, bool pSilent, const char *pUserSearchPath) {
  // enumerate all the files that match [lib]ggml-name-*.[so|dll] in the search paths
  const fs::path cNamePath = fs::u8path(pName);
  const fs::path cFilePrefix = fBackendFilenamePrefix().native() + cNamePath.native() + fs::u8path("-").native();
  const fs::path cFileExtension = fBackendFilenameExtension();

  std::vector<fs::path> vSearchPaths;
  if (pUserSearchPath == nullptr) {
#ifdef GGML_BACKEND_DIR
    vSearchPaths.push_back(fs::u8path(GGML_BACKEND_DIR));
#endif
    // default search paths: executable directory, current directory
    vSearchPaths.push_back(fGetExecutablePath());
    vSearchPaths.push_back(fs::current_path());
  } else {
    vSearchPaths.push_back(fs::u8path(pUserSearchPath));
  }

  int vBestScore = 0;
  fs::path vBestPath;
  std::error_code vEc;

  for (const auto &vSearchPath : vSearchPaths) {
    if (!fs::exists(vSearchPath, vEc)) {
      if (vEc) {
        GGML_LOG_DEBUG("%s: posix_stat(%s) failure, error-message: %s\n", __func__, fPathStr(vSearchPath).c_str(),
                       vEc.message().c_str());
      } else {
        GGML_LOG_DEBUG("%s: search path %s does not exist\n", __func__, fPathStr(vSearchPath).c_str());
      }
      continue;
    }
    fs::directory_iterator vDirIt(vSearchPath, fs::directory_options::skip_permission_denied);
    for (const auto &vEntry : vDirIt) {
      if (vEntry.is_regular_file(vEc)) {
        auto vFilename = vEntry.path().filename();
        auto vExt = vEntry.path().extension();
        if (vFilename.native().find(cFilePrefix) == 0 && vExt == cFileExtension) {
          dl_handle_ptr vHandle{dl_load_library(vEntry)};
          if (!vHandle && !pSilent) {
            GGML_LOG_ERROR("%s: failed to load %s: %s\n", __func__, fPathStr(vEntry.path()).c_str(), dl_error());
          }
          if (vHandle) {
            auto vScoreFn = (ggml_backend_score_t)dl_get_sym(vHandle.get(), "ggml_backend_score");
            if (vScoreFn) {
              int vS = vScoreFn();
#ifndef NDEBUG
              GGML_LOG_DEBUG("%s: %s score: %d\n", __func__, fPathStr(vEntry.path()).c_str(), vS);
#endif
              if (vS > vBestScore) {
                vBestScore = vS;
                vBestPath = vEntry.path();
              }
            } else {
              if (!pSilent) {
                GGML_LOG_INFO("%s: failed to find ggml_backend_score in %s\n", __func__,
                              fPathStr(vEntry.path()).c_str());
              }
            }
          }
        }
      }
    }
  }

  if (vBestScore == 0) {
    // try to load the base backend
    for (const auto &vSearchPath : vSearchPaths) {
      fs::path vFilename =
          fBackendFilenamePrefix().native() + cNamePath.native() + fBackendFilenameExtension().native();
      fs::path vPath = vSearchPath / vFilename;
      if (std::error_code vEc; fs::exists(vPath, vEc)) {
        return fGetReg().mLoadBackend(vPath, pSilent);
      } else {
        if (vEc) {
          GGML_LOG_DEBUG("%s: posix_stat(%s) failure, error-message: %s\n", __func__, fPathStr(vPath).c_str(),
                         vEc.message().c_str());
        }
      }
    }
    return nullptr;
  }

  return fGetReg().mLoadBackend(vBestPath, pSilent);
}

void ggml_backend_load_all() { ggml_backend_load_all_from_path(nullptr); }

void ggml_backend_load_all_from_path(const char *pDirPath) {
#ifdef NDEBUG
  bool vSilent = true;
#else
  bool vSilent = false;
#endif

  fGgmlBackendLoadBest("cuda", vSilent, pDirPath);
  fGgmlBackendLoadBest("hip", vSilent, pDirPath);
  fGgmlBackendLoadBest("vulkan", vSilent, pDirPath);
  fGgmlBackendLoadBest("cpu", vSilent, pDirPath);
  // check the environment variable GGML_BACKEND_PATH to load an out-of-tree backend
  const char *vBackendPath = std::getenv("GGML_BACKEND_PATH");
  if (vBackendPath) {
    ggml_backend_load(vBackendPath);
  }
}
