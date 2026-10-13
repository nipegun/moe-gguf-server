#pragma once

#include <cstdint>
#include <memory>
#include <vector>
#include <cstdio>

// staging buffer size for direct I/O reads, 64MB works well for NVMe drives
#define MGS_DIRECT_IO_BUFFER_SIZE (64 * 1024 * 1024)

struct mgs_file;
struct mgs_mmap;
struct mgs_mlock;

using mgs_files = std::vector<std::unique_ptr<mgs_file>>;
using mgs_mmaps = std::vector<std::unique_ptr<mgs_mmap>>;
using mgs_mlocks = std::vector<std::unique_ptr<mgs_mlock>>;

struct mgs_file {
  mgs_file(const char *pFname, const char *pMode, bool pUseDirectIo = false);
  mgs_file(FILE *pFile);
  ~mgs_file();

  size_t mTell() const;
  size_t size() const;

  int mFileId() const; // fileno overload

  void mSeek(size_t pOffset, int pWhence) const;

  void mReadRaw(void *pPtr, size_t pLen);
  void mReadRawUnsafe(void *pPtr, size_t pLen);
  void mReadAlignedChunk(void *pDest, size_t pSize);
  uint32_t mReadU32();

  void mWriteRaw(const void *pPtr, size_t pLen) const;
  void mWriteU32(uint32_t pVal) const;

  size_t mReadAlignment() const;
  bool mHasDirectIo() const;

private:
  struct impl;
  std::unique_ptr<impl> vPimpl;
};

struct mgs_mmap {
  mgs_mmap(const mgs_mmap &) = delete;
  mgs_mmap(struct mgs_file *pFile, size_t pPrefetch = (size_t)-1, bool pNuma = false);
  ~mgs_mmap();

  size_t size() const;
  void *mAddr() const;

  void mUnmapFragment(size_t pFirst, size_t pLast);

  // Pin the pages backing [first, last) through the backend allocator to
  // accelerate host transfers. Release them before destroying the mapping.
  // Return the registered byte count, or zero when registration is unavailable.
  size_t mRegisterHost(size_t pFirst, size_t pLast, bool (*pRegister)(void *, size_t), void (*pUnregister)(void *));

  static const bool cSupported;

private:
  struct impl;
  std::unique_ptr<impl> vPimpl;

  void *vHostRegisteredAddress = nullptr;
  void (*vHostUnregister)(void *) = nullptr;
};

struct mgs_mlock {
  mgs_mlock();
  ~mgs_mlock();

  void mInit(void *pPtr);
  void mGrowTo(size_t pTargetSize);

  static const bool cSupported;

private:
  struct impl;
  std::unique_ptr<impl> vPimpl;
};

size_t mgs_path_max();

struct mgs_memory_range {
  const void *vAddr;
  size_t vSize;
};

using mgs_memory_ranges = std::vector<mgs_memory_range>;

// prefetch the host pages covering these memory ranges
void fMgsPrefetch(mgs_memory_ranges pMr);
