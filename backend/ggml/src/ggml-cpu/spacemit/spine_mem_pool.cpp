#include "spine_mem_pool.h"

#include "common.h"
#include "ime_env.h"
#include "spine_tcm.h"

#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace ggml::cpu::riscv64_spacemit {
namespace {

constexpr size_t SPINE_MEM_POOL_CHUNK_SIZE = 512ull * 1024ull * 1024ull;
constexpr size_t SPINE_SHARE_MEM_POOL_CHUNK_SIZE = 512ull * 1024ull;
constexpr size_t SPINE_MEM_POOL_1G_REGION_SIZE = 1ull << 30;
constexpr uint64_t HUGETLB_1G_FLAG_REQUIRE_PUD = 1ull << 0;
constexpr char SPINE_MEM_POOL_HUGETLB_1G_DEV[] = "/dev/hugetlb_1g";
constexpr char SPINE_MEM_POOL_TCM_SYNC_MEM_DEV[] = "/dev/tcm_sync_mem";

struct hugetlb_1g_region {
  uint64_t vSize{0};
  uint64_t vDmaAddr{0};
  uint64_t vFlags{0};
  uint64_t vReserved{0};
};

#define HUGETLB_1G_IOC_MAGIC 'M'
#define HUGETLB_1G_IOC_ALLOC _IOWR(HUGETLB_1G_IOC_MAGIC, 0x00, struct hugetlb_1g_region)
#define HUGETLB_1G_IOC_FREE _IO(HUGETLB_1G_IOC_MAGIC, 0x01)

struct free_block {
  size_t vOffset{0};
  size_t vSize{0};
};

struct pool_chunk {
  uint8_t *vBase{nullptr};
  size_t vSize{0};
  int vFd{-1};
  std::vector<free_block> lFreeBlocks;
};

struct pool_allocation {
  void *vChunkBase{nullptr};
  size_t vChunkSize{0};
  void *vBase{nullptr};
  size_t vSize{0};
};

bool fIsPowerOfTwo(size_t pValue) { return pValue != 0 && (pValue & (pValue - 1)) == 0; }

bool fAlignUp(size_t pValue, size_t pAlignment, size_t *pAlignedValue) {
  if (pAlignedValue == nullptr || pAlignment == 0) {
    return false;
  }

  const size_t cRemainder = pValue % pAlignment;
  if (cRemainder == 0) {
    *pAlignedValue = pValue;
    return true;
  }

  const size_t cPadding = pAlignment - cRemainder;
  if (pValue > std::numeric_limits<size_t>::max() - cPadding) {
    return false;
  }

  *pAlignedValue = pValue + cPadding;
  return true;
}

bool fAlignUpUintptr(uintptr_t pValue, size_t pAlignment, uintptr_t *pAlignedValue) {
  if (pAlignedValue == nullptr || pAlignment == 0) {
    return false;
  }

  const uintptr_t cRemainder = pValue % pAlignment;
  if (cRemainder == 0) {
    *pAlignedValue = pValue;
    return true;
  }

  const uintptr_t cPadding = pAlignment - cRemainder;
  if (pValue > std::numeric_limits<uintptr_t>::max() - cPadding) {
    return false;
  }

  *pAlignedValue = pValue + cPadding;
  return true;
}

class spine_mem_pool_manager {
public:
  explicit spine_mem_pool_manager(size_t pDefaultChunkSize) : vDefaultChunkSize(pDefaultChunkSize) {}

  virtual ~spine_mem_pool_manager() = default;

  void *mAlloc(size_t pSize, size_t pAlignment) {
    if (pSize == 0 || !fIsPowerOfTwo(pAlignment)) {
      return nullptr;
    }

    size_t vAlignedSize = 0;
    if (!fAlignUp(pSize, pAlignment, &vAlignedSize)) {
      GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: align_up failed for size %zu alignment %zu\n", __func__, pSize,
                     pAlignment);
      return nullptr;
    }

    pool_allocation vAllocation;

    std::lock_guard<std::mutex> vLock(vMutex);

    if (!mTryAllocLocked(vAlignedSize, pAlignment, &vAllocation)) {
      if (!mAddChunkLocked(vAlignedSize, pAlignment)) {
        return nullptr;
      }

      if (!mTryAllocLocked(vAlignedSize, pAlignment, &vAllocation)) {
        GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: allocation retry failed for size %zu alignment %zu\n", __func__,
                       vAlignedSize, pAlignment);
        return nullptr;
      }
    }

    try {
      const auto [allocation_it, inserted] = dAllocations.emplace(vAllocation.vBase, vAllocation);
      if (!inserted) {
        GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: duplicate allocation key %p\n", __func__, vAllocation.vBase);
        mRollbackAllocationLocked(vAllocation);
        return nullptr;
      }
    } catch (const std::bad_alloc &) {
      mRollbackAllocationLocked(vAllocation);
      throw;
    }

    return vAllocation.vBase;
  }

  void mFree(void *pBase) {
    if (pBase == nullptr) {
      return;
    }

    std::lock_guard<std::mutex> vLock(vMutex);

    auto vAllocationIt = dAllocations.find(pBase);
    if (vAllocationIt == dAllocations.end()) {
      GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: unknown allocation %p\n", __func__, pBase);
      return;
    }

    pool_allocation vAllocation = vAllocationIt->second;
    dAllocations.erase(vAllocationIt);

    auto vChunkIt = mFindChunkLocked(vAllocation);
    if (vChunkIt == lChunks.end()) {
      GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: unknown chunk for allocation %p size %zu\n", __func__,
                     vAllocation.vBase, vAllocation.vSize);
      return;
    }

    auto *vChunkBase = vChunkIt->vBase;
    auto *vAllocBase = static_cast<uint8_t *>(vAllocation.vBase);
    if (vAllocBase < vChunkBase || vAllocBase >= vChunkBase + vChunkIt->vSize) {
      GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: allocation %p out of chunk range %p..%p\n", __func__, vAllocation.vBase,
                     vChunkBase, vChunkBase + vChunkIt->vSize);
      return;
    }

    const size_t cOffset = static_cast<size_t>(vAllocBase - vChunkBase);
    if (cOffset > vChunkIt->vSize || vAllocation.vSize > vChunkIt->vSize - cOffset) {
      GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: allocation %p size %zu exceeds chunk size %zu\n", __func__,
                     vAllocation.vBase, vAllocation.vSize, vChunkIt->vSize);
      return;
    }

    mInsertFreeBlockLocked(*vChunkIt, {cOffset, vAllocation.vSize});
    mMaybeReleaseEmptyChunkLocked(vChunkIt);
  }

protected:
  void mReleaseChunks() {
    std::lock_guard<std::mutex> vLock(vMutex);

    dAllocations.clear();
    for (auto &vChunk : lChunks) {
      mDeallocChunk(&vChunk);
    }
    lChunks.clear();
  }

  size_t mDefaultChunkSize() const { return vDefaultChunkSize; }

  static void mClearChunk(pool_chunk *pChunk) {
    pChunk->vBase = nullptr;
    pChunk->vSize = 0;
    pChunk->vFd = -1;
    pChunk->lFreeBlocks.clear();
  }

  virtual bool mAllocChunk(size_t pMinSize, size_t pAlignment, void *pHintAddr, pool_chunk *pChunk) = 0;
  virtual void mDeallocChunk(pool_chunk *pChunk) = 0;

private:
  struct alloc_candidate {
    size_t vChunkIndex{0};
    size_t vBlockIndex{0};
    size_t vAlignedOffset{0};
    uintptr_t vAddress{std::numeric_limits<uintptr_t>::max()};
    bool vValid{false};
  };

  std::vector<pool_chunk>::iterator mFindChunkLocked(const pool_allocation &pAllocation) {
    return std::find_if(lChunks.begin(), lChunks.end(), [&](const pool_chunk &pChunk) {
      return pChunk.vBase == pAllocation.vChunkBase && pChunk.vSize == pAllocation.vChunkSize;
    });
  }

  bool mAddChunkLocked(size_t pMinSize, size_t pAlignment) {
    pool_chunk vChunk;
    const size_t cChunkRequest = vDefaultChunkSize == 0 ? pMinSize : std::max(pMinSize, vDefaultChunkSize);
    void *vHintAddr = nullptr;

    for (const auto &vExistingChunk : lChunks) {
      auto *vChunkEnd = vExistingChunk.vBase + vExistingChunk.vSize;
      if (vHintAddr == nullptr || vChunkEnd > vHintAddr) {
        vHintAddr = vChunkEnd;
      }
    }

    if (!mAllocChunk(cChunkRequest, pAlignment, vHintAddr, &vChunk)) {
      return false;
    }

    if (vChunk.vBase == nullptr || vChunk.vSize < pMinSize) {
      GGML_LOG_ERROR(
          "CPU_RISCV64_SPACEMIT: %s: invalid chunk returned for request size %zu, chunk_base=%p chunk_size=%zu\n",
          __func__, pMinSize, vChunk.vBase, vChunk.vSize);
      mDeallocChunk(&vChunk);
      return false;
    }

    try {
      vChunk.lFreeBlocks.push_back({0, vChunk.vSize});
      lChunks.push_back(std::move(vChunk));
    } catch (const std::bad_alloc &) {
      mDeallocChunk(&vChunk);
      throw;
    }

    return true;
  }

  void mRollbackAllocationLocked(const pool_allocation &pAllocation) {
    auto vChunkIt = mFindChunkLocked(pAllocation);
    if (vChunkIt == lChunks.end()) {
      GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: failed to rollback allocation %p, owning chunk not found\n", __func__,
                     pAllocation.vBase);
      return;
    }

    auto *vChunkBase = vChunkIt->vBase;
    auto *vAllocBase = static_cast<uint8_t *>(pAllocation.vBase);
    if (vAllocBase < vChunkBase || vAllocBase >= vChunkBase + vChunkIt->vSize) {
      GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: failed to rollback allocation %p, chunk range is invalid\n", __func__,
                     pAllocation.vBase);
      return;
    }

    const size_t cOffset = static_cast<size_t>(vAllocBase - vChunkBase);
    if (cOffset > vChunkIt->vSize || pAllocation.vSize > vChunkIt->vSize - cOffset) {
      GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: failed to rollback allocation %p size %zu\n", __func__,
                     pAllocation.vBase, pAllocation.vSize);
      return;
    }

    mInsertFreeBlockLocked(*vChunkIt, {cOffset, pAllocation.vSize});
    mMaybeReleaseEmptyChunkLocked(vChunkIt);
  }

  bool mTryAllocLocked(size_t pSize, size_t pAlignment, pool_allocation *pAllocation) {
    alloc_candidate vBest;

    for (size_t vChunkIndex = 0; vChunkIndex < lChunks.size(); ++vChunkIndex) {
      const auto &vChunk = lChunks[vChunkIndex];
      for (size_t vBlockIndex = 0; vBlockIndex < vChunk.lFreeBlocks.size(); ++vBlockIndex) {
        const auto &vBlock = vChunk.lFreeBlocks[vBlockIndex];

        uintptr_t vAlignedAddr = 0;
        const auto cBlockAddr = reinterpret_cast<uintptr_t>(vChunk.vBase + vBlock.vOffset);
        if (!fAlignUpUintptr(cBlockAddr, pAlignment, &vAlignedAddr)) {
          continue;
        }

        if (vAlignedAddr < cBlockAddr) {
          continue;
        }

        const size_t cAlignedOffset = vBlock.vOffset + static_cast<size_t>(vAlignedAddr - cBlockAddr);
        const size_t cPadding = cAlignedOffset - vBlock.vOffset;
        if (cPadding > vBlock.vSize || pSize > vBlock.vSize - cPadding) {
          continue;
        }

        if (!vBest.vValid || vAlignedAddr < vBest.vAddress) {
          vBest.vChunkIndex = vChunkIndex;
          vBest.vBlockIndex = vBlockIndex;
          vBest.vAlignedOffset = cAlignedOffset;
          vBest.vAddress = vAlignedAddr;
          vBest.vValid = true;
        }
      }
    }

    if (!vBest.vValid) {
      return false;
    }

    auto &vChunk = lChunks[vBest.vChunkIndex];
    const free_block cBlock = vChunk.lFreeBlocks[vBest.vBlockIndex];
    const size_t cPadding = vBest.vAlignedOffset - cBlock.vOffset;
    const size_t cAllocEnd = vBest.vAlignedOffset + pSize;
    const size_t cBlockEnd = cBlock.vOffset + cBlock.vSize;

    vChunk.lFreeBlocks.erase(vChunk.lFreeBlocks.begin() + vBest.vBlockIndex);
    auto vInsertIt = vChunk.lFreeBlocks.begin() + vBest.vBlockIndex;
    if (cPadding != 0) {
      vInsertIt = vChunk.lFreeBlocks.insert(vInsertIt, {cBlock.vOffset, cPadding});
      ++vInsertIt;
    }
    if (cAllocEnd < cBlockEnd) {
      vChunk.lFreeBlocks.insert(vInsertIt, {cAllocEnd, cBlockEnd - cAllocEnd});
    }

    pAllocation->vChunkBase = vChunk.vBase;
    pAllocation->vChunkSize = vChunk.vSize;
    pAllocation->vBase = vChunk.vBase + vBest.vAlignedOffset;
    pAllocation->vSize = pSize;
    return true;
  }

  void mMaybeReleaseEmptyChunkLocked(std::vector<pool_chunk>::iterator pChunkIt) {
    if (pChunkIt->lFreeBlocks.size() != 1) {
      return;
    }

    const auto &vBlock = pChunkIt->lFreeBlocks.front();
    if (vBlock.vOffset != 0 || vBlock.vSize != pChunkIt->vSize) {
      return;
    }

    mDeallocChunk(&*pChunkIt);
    lChunks.erase(pChunkIt);
  }

  void mInsertFreeBlockLocked(pool_chunk &pChunk, free_block pBlock) {
    auto vIt = pChunk.lFreeBlocks.begin();
    while (vIt != pChunk.lFreeBlocks.end() && vIt->vOffset < pBlock.vOffset) {
      ++vIt;
    }

    if (vIt != pChunk.lFreeBlocks.begin()) {
      const auto &vPrev = *(vIt - 1);
      if (vPrev.vOffset + vPrev.vSize > pBlock.vOffset) {
        GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: overlapping free block at offset %zu size %zu\n", __func__,
                       pBlock.vOffset, pBlock.vSize);
        return;
      }
    }

    if (vIt != pChunk.lFreeBlocks.end() && pBlock.vOffset + pBlock.vSize > vIt->vOffset) {
      GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: overlapping next free block at offset %zu size %zu\n", __func__,
                     pBlock.vOffset, pBlock.vSize);
      return;
    }

    vIt = pChunk.lFreeBlocks.insert(vIt, pBlock);

    if (vIt != pChunk.lFreeBlocks.begin()) {
      auto vPrev = vIt - 1;
      if (vPrev->vOffset + vPrev->vSize == vIt->vOffset) {
        vIt->vOffset = vPrev->vOffset;
        vIt->vSize += vPrev->vSize;
        vIt = pChunk.lFreeBlocks.erase(vPrev);
      }
    }

    if (vIt + 1 != pChunk.lFreeBlocks.end() && vIt->vOffset + vIt->vSize == (vIt + 1)->vOffset) {
      vIt->vSize += (vIt + 1)->vSize;
      pChunk.lFreeBlocks.erase(vIt + 1);
    }
  }

  std::mutex vMutex;
  std::vector<pool_chunk> lChunks;
  std::unordered_map<void *, pool_allocation> dAllocations;
  size_t vDefaultChunkSize{0};
};

class spine_mem_pool_posix final : public spine_mem_pool_manager {
public:
  spine_mem_pool_posix() : spine_mem_pool_manager(0) {}

  ~spine_mem_pool_posix() override { mReleaseChunks(); }

private:
  bool mAllocChunk(size_t pMinSize, size_t pAlignment, void *pHintAddr, pool_chunk *pChunk) override {
    (void)pHintAddr;

    const size_t cAllocAlignment = std::max(pAlignment, sizeof(void *));
    void *vBase = nullptr;
    const int cRc = posix_memalign(&vBase, cAllocAlignment, pMinSize);
    if (cRc != 0) {
      GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: posix_memalign failed for size %zu alignment %zu, rc=%d\n", __func__,
                     pMinSize, cAllocAlignment, cRc);
      return false;
    }

    pChunk->vBase = static_cast<uint8_t *>(vBase);
    pChunk->vSize = pMinSize;
    pChunk->vFd = -1;
    return true;
  }

  void mDeallocChunk(pool_chunk *pChunk) override {
    std::free(pChunk->vBase);
    mClearChunk(pChunk);
  }
};

class spine_mem_pool_transparent_hugepage final : public spine_mem_pool_manager {
public:
  spine_mem_pool_transparent_hugepage() : spine_mem_pool_manager(SPINE_MEM_POOL_CHUNK_SIZE) {}

  ~spine_mem_pool_transparent_hugepage() override { mReleaseChunks(); }

private:
  bool mAllocChunk(size_t pMinSize, size_t pAlignment, void *pHintAddr, pool_chunk *pChunk) override {
    (void)pAlignment;

    size_t vChunkSize = 0;
    if (!fAlignUp(pMinSize, mDefaultChunkSize(), &vChunkSize)) {
      GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: failed to round chunk size for %zu\n", __func__, pMinSize);
      return false;
    }

    void *vMapAddr = mmap(pHintAddr, vChunkSize, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (vMapAddr == MAP_FAILED) {
      GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: mmap failed for chunk size %zu, errno=%d\n", __func__, vChunkSize,
                     errno);
      return false;
    }

    if (madvise(vMapAddr, vChunkSize, MADV_HUGEPAGE) != 0) {
      GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: madvise(MADV_HUGEPAGE) failed for chunk size %zu, errno=%d\n", __func__,
                     vChunkSize, errno);
      munmap(vMapAddr, vChunkSize);
      return false;
    }

    pChunk->vBase = static_cast<uint8_t *>(vMapAddr);
    pChunk->vSize = vChunkSize;
    pChunk->vFd = -1;
    return true;
  }

  void mDeallocChunk(pool_chunk *pChunk) override {
    if (pChunk->vBase != nullptr && pChunk->vSize != 0 && munmap(pChunk->vBase, pChunk->vSize) != 0) {
      GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: munmap failed for chunk %p size %zu, errno=%d\n", __func__,
                     pChunk->vBase, pChunk->vSize, errno);
    }

    mClearChunk(pChunk);
  }
};

class spine_mem_pool_hugetlb_1g final : public spine_mem_pool_manager {
public:
  spine_mem_pool_hugetlb_1g() : spine_mem_pool_manager(SPINE_MEM_POOL_1G_REGION_SIZE) {}

  ~spine_mem_pool_hugetlb_1g() override { mReleaseChunks(); }

private:
  bool mAllocChunk(size_t pMinSize, size_t pAlignment, void *pHintAddr, pool_chunk *pChunk) override {
    (void)pAlignment;
    (void)pHintAddr;

    size_t vRegionSize = 0;
    if (!fAlignUp(pMinSize, SPINE_MEM_POOL_1G_REGION_SIZE, &vRegionSize)) {
      GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: failed to round hugetlb_1g size for %zu\n", __func__, pMinSize);
      return false;
    }

    const int cFd = open(SPINE_MEM_POOL_HUGETLB_1G_DEV, O_RDWR);
    if (cFd < 0) {
      GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: open(%s) failed, errno=%d\n", __func__, SPINE_MEM_POOL_HUGETLB_1G_DEV,
                     errno);
      return false;
    }

    hugetlb_1g_region vRegion;
    vRegion.vSize = vRegionSize;
    vRegion.vFlags = HUGETLB_1G_FLAG_REQUIRE_PUD;
    if (ioctl(cFd, HUGETLB_1G_IOC_ALLOC, &vRegion) < 0) {
      GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: HUGETLB_1G_IOC_ALLOC failed for size %zu, errno=%d\n", __func__,
                     vRegionSize, errno);
      close(cFd);
      return false;
    }

    void *vMapAddr = mmap(nullptr, vRegion.vSize, PROT_READ | PROT_WRITE, MAP_SHARED, cFd, 0);
    if (vMapAddr == MAP_FAILED) {
      GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: mmap failed for hugetlb_1g size %llu, errno=%d\n", __func__,
                     static_cast<unsigned long long>(vRegion.vSize), errno);
      ioctl(cFd, HUGETLB_1G_IOC_FREE);
      close(cFd);
      return false;
    }

    pChunk->vBase = static_cast<uint8_t *>(vMapAddr);
    pChunk->vSize = vRegion.vSize;
    pChunk->vFd = cFd;
    return true;
  }

  void mDeallocChunk(pool_chunk *pChunk) override {
    if (pChunk->vBase != nullptr && pChunk->vSize != 0 && munmap(pChunk->vBase, pChunk->vSize) != 0) {
      GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: munmap failed for hugetlb_1g chunk %p size %zu, errno=%d\n", __func__,
                     pChunk->vBase, pChunk->vSize, errno);
    }

    if (pChunk->vFd >= 0) {
      if (ioctl(pChunk->vFd, HUGETLB_1G_IOC_FREE) < 0) {
        GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: HUGETLB_1G_IOC_FREE failed for chunk %p, errno=%d\n", __func__,
                       pChunk->vBase, errno);
      }

      close(pChunk->vFd);
    }

    mClearChunk(pChunk);
  }
};

class spine_mem_pool_shared_mem final : public spine_mem_pool_manager {
public:
  spine_mem_pool_shared_mem() : spine_mem_pool_manager(SPINE_SHARE_MEM_POOL_CHUNK_SIZE) {}

  ~spine_mem_pool_shared_mem() override { mReleaseChunks(); }

private:
  bool mAllocChunk(size_t pMinSize, size_t pAlignment, void *pHintAddr, pool_chunk *pChunk) override {
    (void)pAlignment;

    if (pHintAddr != nullptr) {
      GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: shared_mem does not support multiple active chunks\n", __func__);
      return false;
    }

    if (pMinSize > mDefaultChunkSize()) {
      GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: shared_mem request %zu exceeds chunk size %zu\n", __func__, pMinSize,
                     mDefaultChunkSize());
      return false;
    }

    const int cFd = open(SPINE_MEM_POOL_TCM_SYNC_MEM_DEV, O_RDWR | O_SYNC);
    if (cFd < 0) {
      GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: open(%s) failed, errno=%d\n", __func__, SPINE_MEM_POOL_TCM_SYNC_MEM_DEV,
                     errno);
      return false;
    }

    void *vMapAddr = mmap(nullptr, mDefaultChunkSize(), PROT_READ | PROT_WRITE, MAP_SHARED, cFd, 0);
    if (vMapAddr == MAP_FAILED) {
      GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: mmap failed for %s size %zu, errno=%d\n", __func__,
                     SPINE_MEM_POOL_TCM_SYNC_MEM_DEV, mDefaultChunkSize(), errno);
      close(cFd);
      return false;
    }

    pChunk->vBase = static_cast<uint8_t *>(vMapAddr);
    pChunk->vSize = mDefaultChunkSize();
    pChunk->vFd = cFd;
    return true;
  }

  void mDeallocChunk(pool_chunk *pChunk) override {
    if (pChunk->vBase != nullptr && pChunk->vSize != 0 && munmap(pChunk->vBase, pChunk->vSize) != 0) {
      GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: munmap failed for shared_mem chunk %p size %zu, errno=%d\n", __func__,
                     pChunk->vBase, pChunk->vSize, errno);
    }

    if (pChunk->vFd >= 0) {
      close(pChunk->vFd);
    }

    mClearChunk(pChunk);
  }
};

spine_mem_pool_manager &fGetSpineMemPoolManager() {
  static std::once_flag vPoolOnce;
  static std::unique_ptr<spine_mem_pool_manager> vSelectedPool;
  static spine_mem_pool_backend vSelectedBackend = spine_mem_pool_backend::none;

  spine_mem_pool_backend vBackend = global_spine_env_info.mem_backend;
  if (vBackend == spine_mem_pool_backend::none) {
    vBackend = spine_mem_pool_backend::transparent_hugepage;
  }

  std::call_once(vPoolOnce, [&]() {
    vSelectedBackend = vBackend;

    switch (vSelectedBackend) {
    case spine_mem_pool_backend::posix_memalign:
      vSelectedPool = std::make_unique<spine_mem_pool_posix>();
      break;
    case spine_mem_pool_backend::transparent_hugepage:
      vSelectedPool = std::make_unique<spine_mem_pool_transparent_hugepage>();
      break;
    case spine_mem_pool_backend::hugetlb_1g:
      vSelectedPool = std::make_unique<spine_mem_pool_hugetlb_1g>();
      break;
    case spine_mem_pool_backend::none:
      vSelectedBackend = spine_mem_pool_backend::transparent_hugepage;
      vSelectedPool = std::make_unique<spine_mem_pool_transparent_hugepage>();
      break;
    }
  });

  if (vBackend != vSelectedBackend) {
    GGML_LOG_ERROR(
        "CPU_RISCV64_SPACEMIT: %s: mem pool backend is process-global and mutually exclusive, requested=%d but "
        "selected=%d\n",
        __func__, static_cast<int>(vBackend), static_cast<int>(vSelectedBackend));
  }

  if (vSelectedPool) {
    return *vSelectedPool;
  }

  throw std::bad_alloc();
}

spine_mem_pool_manager &fGetSpineMemPoolSharedMemManager() {
  static std::once_flag vSharedMemPoolOnce;
  static std::unique_ptr<spine_mem_pool_shared_mem> vSharedMemPool;

  std::call_once(vSharedMemPoolOnce, [&]() { vSharedMemPool = std::make_unique<spine_mem_pool_shared_mem>(); });

  if (vSharedMemPool) {
    return *vSharedMemPool;
  }

  throw std::bad_alloc();
}

} // namespace

bool spine_mem_pool_tcm_init(spine_mem_pool_tcm_info *pInfo) noexcept {
  if (pInfo == nullptr) {
    return false;
  }

  *pInfo = {};

  if (spine_tcm_open_handle(NULL) != 0 || !spine_tcm_is_available()) {
    return false;
  }

  spine_tcm_mem_info_t vMemInfo;
  if (spine_tcm_mem_info(&vMemInfo) != 0) {
    return false;
  }

  pInfo->available = true;
  pInfo->blk_size = vMemInfo.blk_size;
  pInfo->blk_num = vMemInfo.blk_num;
  pInfo->is_fake_tcm = vMemInfo.is_fake_tcm != 0;
  return true;
}

void *spine_mem_pool_tcm_mem_get(int pCpuId) noexcept { return spine_tcm_mem_get(pCpuId); }

void *spine_mem_pool_tcm_mem_wait(int pCpuId) noexcept { return spine_tcm_mem_try_wait(pCpuId, 1000 * 1000); }

int spine_mem_pool_tcm_mem_release(int pCpuId) noexcept { return spine_tcm_mem_release(pCpuId); }

void *spine_mem_pool_alloc(size_t pSize, size_t pAlignment) noexcept {
  try {
    return fGetSpineMemPoolManager().mAlloc(pSize, pAlignment);
  } catch (const std::bad_alloc &) {
    GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: bad_alloc while allocating size %zu\n", __func__, pSize);
    return nullptr;
  }
}

void *spine_mem_pool_shared_mem_alloc(size_t pSize, size_t pAlignment) noexcept {
  try {
    return fGetSpineMemPoolSharedMemManager().mAlloc(pSize, pAlignment);
  } catch (const std::bad_alloc &) {
    GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: bad_alloc while allocating shared memory size %zu\n", __func__, pSize);
    return nullptr;
  }
}

void spine_mem_pool_free(void *pBase) noexcept {
  try {
    fGetSpineMemPoolManager().mFree(pBase);
  } catch (const std::bad_alloc &) {
    GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: bad_alloc while freeing allocation %p\n", __func__, pBase);
  }
}

void spine_mem_pool_shared_mem_free(void *pBase) noexcept {
  try {
    fGetSpineMemPoolSharedMemManager().mFree(pBase);
  } catch (const std::bad_alloc &) {
    GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: bad_alloc while freeing shared allocation %p\n", __func__, pBase);
  }
}

} // namespace ggml::cpu::riscv64_spacemit

extern "C" {
void *fGgmlBackendCpuRiscv64SpacemitAllocShared(size_t pSize, size_t pAlignment) {
  void *vResult = ggml::cpu::riscv64_spacemit::spine_mem_pool_shared_mem_alloc(pSize, pAlignment);
  if (vResult == nullptr) {
    GGML_LOG_ERROR("CPU_RISCV64_SPACEMIT: %s: failed to allocate shared memory size %zu alignment %zu\n", __func__,
                   pSize, pAlignment);
  }
  return vResult;
}

void fGgmlBackendCpuRiscv64SpacemitFreeShared(void *pPtr) {
  ggml::cpu::riscv64_spacemit::spine_mem_pool_shared_mem_free(pPtr);
}
}
