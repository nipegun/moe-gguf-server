#include "ime_env.h"

#include "ggml-impl.h"
#include "spine_mem_pool.h"

#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <fstream>
#include <string>
#include <thread>
#include <unordered_map>

namespace ggml::cpu::riscv64_spacemit {
bool spine_core_info::get_spine_core_info(std::vector<spine_core_info> &pResult) {
  static std::unordered_map<uint64_t, spine_core_arch_id> dSpineMarchMapping = {
      {0x8000000058000001, spine_core_arch_id::core_arch_x60},
      {0x8000000041000001, spine_core_arch_id::core_arch_a60},
      {0x8000000058000002, spine_core_arch_id::core_arch_x100},
      {0x8000000041000002, spine_core_arch_id::core_arch_a100},
  };

  pResult.clear();
  std::ifstream vFile("/proc/cpuinfo");
  std::string vLine;

  std::vector<std::array<uint64_t, 2>> lCpuInfoList;

  uint64_t vCurrentProcessor = spine_invalid_core_id;
  uint64_t vCurrentMarchid = 0;
  bool vHasProcessor = false;
  bool vHasMarchid = false;

  if (!vFile.is_open()) {
    return false;
  }

  while (std::getline(vFile, vLine)) {
    if (vLine.substr(0, 9) == "processor") {
      if (vHasProcessor && vHasMarchid) {
        lCpuInfoList.push_back({vCurrentProcessor, vCurrentMarchid});
      }

      size_t vColonPos = vLine.find(':');
      if (vColonPos != std::string::npos) {
        vCurrentProcessor = std::stoi(vLine.substr(vColonPos + 1));
        vHasProcessor = true;
      }

      vHasMarchid = false;
    } else if (vLine.substr(0, 7) == "marchid") {
      size_t vColonPos = vLine.find(':');
      if (vColonPos != std::string::npos) {
        std::string vMarchidStr = vLine.substr(vColonPos + 1);
        vMarchidStr.erase(std::remove_if(vMarchidStr.begin(), vMarchidStr.end(), isspace), vMarchidStr.end());
        vCurrentMarchid = std::stoull(vMarchidStr, nullptr, 16);
        vHasMarchid = true;
      }
    }
  }

  if (vHasProcessor && vHasMarchid) {
    lCpuInfoList.push_back({vCurrentProcessor, vCurrentMarchid});
  }

  if (vHasProcessor && vHasMarchid) {
    for (auto &aCpuInfo : lCpuInfoList) {
      if (aCpuInfo[0] != spine_invalid_core_id && dSpineMarchMapping.find(aCpuInfo[1]) != dSpineMarchMapping.end()) {
        auto vCoreInfo = spine_core_info();
        vCoreInfo.core_id = aCpuInfo[0];
        vCoreInfo.arch_id = spine_core_arch_id(dSpineMarchMapping[aCpuInfo[1]]);

        pResult.push_back(vCoreInfo);
      }
    }
  }

  return vHasProcessor && vHasMarchid;
}

namespace {
uint16_t fHexStringToU16(const std::string &pHexStr) {
  try {
    size_t vPos = 0;
    if (pHexStr.substr(0, 2) == "0x" || pHexStr.substr(0, 2) == "0X") {
      vPos = 2;
    }
    unsigned long vResult = std::stoul(pHexStr.substr(vPos), nullptr, 16);
    if (vResult > std::numeric_limits<uint16_t>::max()) {
      throw std::out_of_range("Converted value is out of range for uint16_t");
    }
    return static_cast<uint16_t>(vResult);
  } catch (const std::invalid_argument &vE) {
    throw std::invalid_argument("Invalid hexadecimal string");
  } catch (const std::out_of_range &vE) {
    throw;
  }
}

const char *fSpineMemPoolBackendToString(spine_mem_pool_backend pBackend) {
  switch (pBackend) {
  case spine_mem_pool_backend::none:
    return "NONE";
  case spine_mem_pool_backend::posix_memalign:
    return "POSIX";
  case spine_mem_pool_backend::transparent_hugepage:
    return "HPAGE";
  case spine_mem_pool_backend::hugetlb_1g:
    return "HPAGE1GB";
  }

  return "unknown";
}

spine_mem_pool_backend fParseMemBackend(const char *pMemBackendStr) {
  if (pMemBackendStr == nullptr || pMemBackendStr[0] == '\0') {
    return spine_mem_pool_backend::transparent_hugepage;
  }

  std::string vValue(pMemBackendStr);
  std::transform(vValue.begin(), vValue.end(), vValue.begin(),
                 [](unsigned char pCh) { return static_cast<char>(std::tolower(pCh)); });

  if (vValue == "none") {
    return spine_mem_pool_backend::none;
  }

  if (vValue == "posix") {
    return spine_mem_pool_backend::posix_memalign;
  }

  if (vValue == "hpage") {
    return spine_mem_pool_backend::transparent_hugepage;
  }

  if (vValue == "hpage1gb") {
    return spine_mem_pool_backend::hugetlb_1g;
  }

  throw std::runtime_error("invalid SPACEMIT_MEM_BACKEND: " + vValue + ", expected NONE, POSIX, HPAGE or HPAGE1GB");
}
} // namespace

spine_env_info::spine_env_info() {
  num_cores = static_cast<int>(std::thread::hardware_concurrency());
  spine_core_info::get_spine_core_info(core_info_list);

  // special for x60 K1
  if (core_info_list.size() == 8 && core_info_list[0].arch_id == spine_core_arch_id::core_arch_x60) {
    for (int vI = 0; vI < 4; vI++) {
      core_info_list[vI].arch_id = spine_core_arch_id::core_arch_a60;
    }
  }

  // special for qemu
  if (core_info_list.size() == 0) {
    char *vSpineCoreArchStr = getenv("SPACEMIT_CORE_ARCH");
    if (vSpineCoreArchStr != nullptr) {
      auto vArchId = fHexStringToU16(vSpineCoreArchStr);
      for (int vI = 0; vI < num_cores; vI++) {
        auto vCoreInfo = spine_core_info();
        vCoreInfo.core_id = vI;
        vCoreInfo.arch_id = spine_core_arch_id{vArchId};
        core_info_list.push_back(vCoreInfo);
      }
    }
  }

  if (core_info_list.size() == 0) {
    throw std::runtime_error(
        "Failed to get SPACEMIT_CORE_ARCH from environment or failed to parse it from /proc/cpuinfo");
  }

  char *vSpinePerferCoreArchStr = getenv("SPACEMIT_PERFER_CORE_ARCH");
  if (vSpinePerferCoreArchStr != nullptr && vSpinePerferCoreArchStr != "") {
    perfer_core_arch_id = spine_core_arch_id{fHexStringToU16(vSpinePerferCoreArchStr)};
  }

  char *vSpinePerferCoreIdStr = getenv("SPACEMIT_PERFER_CORE_ID");
  std::vector<int> lPerferCoreIdVec;
  if (vSpinePerferCoreIdStr != nullptr && vSpinePerferCoreIdStr != "") {
    std::string vPerferCoreIdStr(vSpinePerferCoreIdStr);
    size_t vStart = 0;
    size_t vEnd = 0;
    while ((vEnd = vPerferCoreIdStr.find(',', vStart)) != std::string::npos) {
      std::string vCoreIdSubstr = vPerferCoreIdStr.substr(vStart, vEnd - vStart);
      lPerferCoreIdVec.push_back(std::stoi(vCoreIdSubstr));
      vStart = vEnd + 1;
    }
    std::string vCoreIdSubstr = vPerferCoreIdStr.substr(vStart);
    lPerferCoreIdVec.push_back(std::stoi(vCoreIdSubstr));
  }

  perfer_core_ids.reserve(num_cores);
  if (perfer_core_arch_id == spine_core_arch_id::core_arch_none) {
    for (auto &vCoreInfo : core_info_list) {
      auto vCoreArchId = vCoreInfo.arch_id;
      auto vCoreArchHead = (uint16_t)(vCoreArchId) >> 12;
      if (vCoreArchHead == 0xA) {
        num_perfer_cores++;
        perfer_core_arch_id = vCoreArchId;
        cpu_mask |= (1ULL << vCoreInfo.core_id);
        perfer_core_ids.push_back(vCoreInfo.core_id);
      }
    }
  } else {
    for (auto &vCoreInfo : core_info_list) {
      auto vCoreArchId = vCoreInfo.arch_id;
      if (vCoreArchId == perfer_core_arch_id) {
        num_perfer_cores++;
        cpu_mask |= (1ULL << vCoreInfo.core_id);

        auto vCoreArchHead = (uint16_t)(vCoreArchId) >> 12;
        if (vCoreArchHead == 0xA) {
          perfer_core_ids.push_back(vCoreInfo.core_id);
        }
      }
    }
    if (num_perfer_cores == 0) {
      GGML_ABORT("can not find core with arch id %x for SPACEMIT_PERFER_CORE_ARCH in core info list\n",
                 (uint16_t)perfer_core_arch_id);
    }
  }

  if (lPerferCoreIdVec.size() > 0) {
    perfer_core_ids.clear();
    cpu_mask = 0;
    num_perfer_cores = 0;
    for (int vCoreId : lPerferCoreIdVec) {
      if (vCoreId < 0 || vCoreId >= num_cores) {
        GGML_ABORT("invalid core id in SPACEMIT_PERFER_CORE_ID: %d, should be between 0 and %d\n", vCoreId,
                   num_cores - 1);
      }
      auto vCoreInfo = core_info_list[vCoreId];
      auto vCoreArchId = vCoreInfo.arch_id;
      if (vCoreArchId == perfer_core_arch_id) {
        cpu_mask |= (1ULL << vCoreId);
        perfer_core_ids.push_back(vCoreId);
      } else {
        GGML_ABORT("core id %d in SPACEMIT_PERFER_CORE_ID has arch id %x which does not match "
                   "SPACEMIT_PERFER_CORE_ARCH %x\n",
                   vCoreId, (uint16_t)vCoreArchId, (uint16_t)perfer_core_arch_id);
      }
    }
    std::string vPerferCoreIdVecStr;
    for (int vCoreId : lPerferCoreIdVec) {
      vPerferCoreIdVecStr += std::to_string(vCoreId) + ",";
    }
    vPerferCoreIdVecStr.pop_back();
    GGML_LOG_DEBUG("SPACEMIT_PERFER_CORE_ID is set, perferred core ids: %s\n", vPerferCoreIdVecStr.c_str());
    num_perfer_cores = static_cast<int>(lPerferCoreIdVec.size());
  }

  use_ime1 = perfer_core_arch_id == spine_core_arch_id::core_arch_a60 ||
             perfer_core_arch_id == spine_core_arch_id::core_arch_x100;

  use_ime2 = perfer_core_arch_id == spine_core_arch_id::core_arch_a100;

  mem_backend = fParseMemBackend(getenv("SPACEMIT_MEM_BACKEND"));
  char *vSpineDisableTcmStr = getenv("SPACEMIT_DISABLE_TCM");
  auto vUserDisableTcm = vSpineDisableTcmStr != nullptr && strcmp(vSpineDisableTcmStr, "0") != 0;

  if (!vUserDisableTcm) {
    spine_mem_pool_tcm_info vTcmInfo;
    if (spine_mem_pool_tcm_init(&vTcmInfo)) {
      use_tcm = vTcmInfo.available;
      tcm_blk_size = vTcmInfo.blk_size;
      GGML_LOG_DEBUG("CPU_RISCV64_SPACEMIT: tcm is available, blk_size: %zu, blk_num: %zu, is_fake_tcm: %d\n",
                     vTcmInfo.blk_size, vTcmInfo.blk_num, vTcmInfo.is_fake_tcm);

      for (auto &vCoreInfo : core_info_list) {
        auto vCoreArchHead = (uint16_t)(vCoreInfo.arch_id) >> 12;
        if (vCoreArchHead != 0xA) {
          aicpu_id_offset++;
        } else {
          break;
        }
      }
    }
  }

  GGML_LOG_DEBUG(
      "CPU_RISCV64_SPACEMIT: num_cores: %d, num_perfer_cores: %d, perfer_core_arch_id: %x, exclude_main_thread: %d, "
      "use_ime1: %d, use_ime2: %d, mem_backend: %s, cpu_mask: %lx, aicpu_id_offset: %d\n",
      num_cores, num_perfer_cores, (uint16_t)perfer_core_arch_id, exclude_main_thread, use_ime1, use_ime2,
      fSpineMemPoolBackendToString(mem_backend), cpu_mask, aicpu_id_offset);

  const size_t cInitBarrierSize = sizeof(spine_barrier_t) * spine_init_barrier_count;
  init_barrier =
      static_cast<spine_barrier_t *>(spine_mem_pool_shared_mem_alloc(cInitBarrierSize, alignof(spine_barrier_t)));
  if (init_barrier != nullptr) {
    init_barrier_is_shared_mem = true;
  } else {
    GGML_LOG_WARN("CPU_RISCV64_SPACEMIT: failed to allocate init_barrier from shared mem, falling back to heap\n",
                  __func__);
    init_barrier = new spine_barrier_t[spine_init_barrier_count];
  }

  spine_barrier_init(init_barrier, spine_init_barrier_count, 2);
}

spine_env_info::~spine_env_info() {
  if (init_barrier_is_shared_mem) {
    spine_mem_pool_shared_mem_free(init_barrier);
  } else {
    delete[] init_barrier;
  }

  init_barrier = nullptr;
  init_barrier_is_shared_mem = false;
}

spine_env_info global_spine_env_info;

} // namespace ggml::cpu::riscv64_spacemit
