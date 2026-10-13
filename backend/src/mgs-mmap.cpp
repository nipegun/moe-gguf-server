#include "mgs-mmap.h"

#include "mgs-impl.h"

#include "ggml.h"

#include <cstring>
#include <climits>
#include <stdexcept>
#include <cerrno>
#include <algorithm>

#ifdef __has_include
#if __has_include(<unistd.h>)
#include <unistd.h>
#include <fcntl.h>
#include <sys/stat.h>
#if defined(_POSIX_MAPPED_FILES)
#include <sys/mman.h>
#endif
#if defined(_POSIX_MEMLOCK_RANGE)
#include <sys/resource.h>
#endif
#endif
#endif

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#ifndef PATH_MAX
#define PATH_MAX MAX_PATH
#endif
#include <io.h>
#endif

#if defined(__APPLE__)
#include <TargetConditionals.h>
#endif

#ifdef _WIN32
#define mgs_mmap_ftell _ftelli64
#define mgs_mmap_fseek _fseeki64
#else
#define mgs_mmap_ftell ftello
#define mgs_mmap_fseek fseeko
#endif

// TODO: consider moving to mgs-impl.h if needed in more places
#if defined(_WIN32)
static std::string fMgsFormatWinErr(DWORD pErr) {
  LPSTR vBuf;
  size_t vSize =
      FormatMessageA(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS, NULL,
                     pErr, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), (LPSTR)&vBuf, 0, NULL);
  if (!vSize) {
    return "FormatMessageA failed";
  }
  std::string vRet(vBuf, vSize);
  LocalFree(vBuf);
  return vRet;
}
#endif

// mgs_file

struct mgs_file::impl {
#if defined(_WIN32)
  HANDLE vFpWin32;
  std::string mGetErrorMessageWin32(DWORD pErrorCode) const {
    std::string vRet;
    LPSTR vLpMsgBuf = NULL;
    DWORD vBufLen =
        FormatMessageA(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
                       NULL, pErrorCode, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), (LPSTR)&vLpMsgBuf, 0, NULL);
    if (!vBufLen) {
      vRet = fFormat("Win32 error code: %lx", pErrorCode);
    } else {
      vRet = vLpMsgBuf;
      LocalFree(vLpMsgBuf);
    }

    return vRet;
  }

  impl(const char *pFname, const char *pMode, [[maybe_unused]] const bool pUseDirectIo = false) {
    vFp = ggml_fopen(pFname, pMode);
    if (vFp == NULL) {
      throw std::runtime_error(fFormat("failed to open %s: %s", pFname, strerror(errno)));
    }
    vFpWin32 = (HANDLE)_get_osfhandle(_fileno(vFp));
    mSeek(0, SEEK_END);
    vSize = mTell();
    mSeek(0, SEEK_SET);
  }

  impl(FILE *pFile) : vOwnsFp(false) {
    vFp = pFile;
    vFpWin32 = (HANDLE)_get_osfhandle(_fileno(vFp));
    mSeek(0, SEEK_END);
    vSize = mTell();
    mSeek(0, SEEK_SET);
  }

  size_t mTell() const {
    LARGE_INTEGER vLi;
    vLi.QuadPart = 0;
    BOOL vRet = SetFilePointerEx(vFpWin32, vLi, &vLi, FILE_CURRENT);
    if (!vRet) {
      throw std::runtime_error(fFormat("read error: %s", mGetErrorMessageWin32(GetLastError()).c_str()));
    }

    return vLi.QuadPart;
  }

  void mSeek(size_t pOffset, int pWhence) const {
    static_assert(SEEK_SET == FILE_BEGIN, "SEEK_SET != FILE_BEGIN");
    static_assert(SEEK_CUR == FILE_CURRENT, "SEEK_CUR != FILE_CURRENT");
    static_assert(SEEK_END == FILE_END, "SEEK_END != FILE_END");

    LARGE_INTEGER vLi;
    vLi.QuadPart = pOffset;
    BOOL vRet = SetFilePointerEx(vFpWin32, vLi, NULL, pWhence);
    if (!vRet) {
      throw std::runtime_error(fFormat("read error: %s", mGetErrorMessageWin32(GetLastError()).c_str()));
    }
  }

  void mReadRaw(void *pPtr, size_t pLen) {
    size_t vBytesRead = 0;
    while (vBytesRead < pLen) {
      size_t vChunkSize = std::min<size_t>(pLen - vBytesRead, 64 * 1024 * 1024);
      DWORD vChunkRead = 0;
      BOOL vResult = ReadFile(vFpWin32, reinterpret_cast<char *>(pPtr) + vBytesRead, vChunkSize, &vChunkRead, NULL);
      if (!vResult) {
        throw std::runtime_error(fFormat("read error: %s", mGetErrorMessageWin32(GetLastError()).c_str()));
      }
      if (vChunkRead < vChunkSize || vChunkRead == 0) {
        throw std::runtime_error("unexpectedly reached end of file");
      }

      vBytesRead += vChunkRead;
    }
  }

  uint32_t mReadU32() {
    uint32_t vVal;
    mReadRaw(&vVal, sizeof(vVal));
    return vVal;
  }

  void mWriteRaw(const void *pPtr, size_t pLen) const {
    size_t vBytesWritten = 0;
    while (vBytesWritten < pLen) {
      size_t vChunkSize = std::min<size_t>(pLen - vBytesWritten, 64 * 1024 * 1024);
      DWORD vChunkWritten = 0;
      BOOL vResult =
          WriteFile(vFpWin32, reinterpret_cast<char const *>(pPtr) + vBytesWritten, vChunkSize, &vChunkWritten, NULL);
      if (!vResult) {
        throw std::runtime_error(fFormat("write error: %s", mGetErrorMessageWin32(GetLastError()).c_str()));
      }
      if (vChunkWritten < vChunkSize || vChunkWritten == 0) {
        throw std::runtime_error("unexpectedly failed to write bytes");
      }

      vBytesWritten += vChunkWritten;
    }
  }

  void mWriteU32(uint32_t pVal) const { mWriteRaw(&pVal, sizeof(pVal)); }

  bool mHasDirectIo() const { return true; }

  ~impl() {
    if (vFp && vOwnsFp) {
      std::fclose(vFp);
    }
  }
#else
  impl(const char *pFname, const char *pMode, [[maybe_unused]] const bool pUseDirectIo = false) : vFname(pFname) {
#ifdef __linux__
    // Try unbuffered I/O for read only
    if (pUseDirectIo && std::strcmp(pMode, "rb") == 0) {
      if (mInitFd()) {
        return;
      }
      MGS_LOG_WARN("Failed to open file '%s' with error: %s. Falling back to buffered I/O", pFname, strerror(errno));
    }
#endif
    mInitFp(pMode);
  }

#ifdef __linux__
  bool mInitFd() {
    vFd = open(vFname.c_str(), O_RDONLY | O_DIRECT);

    if (vFd != -1) {
      struct stat vFileStats{};
      fstat(vFd, &vFileStats);

      vSize = vFileStats.st_size;
      vAlignment = vFileStats.st_blksize;

      off_t vRet = lseek(vFd, 0, SEEK_SET);
      if (vRet == -1) {
        throw std::runtime_error(fFormat("seek error: %s", strerror(errno)));
      }
      return true;
    }
    return false;
  }
#endif

  void mInitFp(const char *pMode) {
    vFp = ggml_fopen(vFname.c_str(), pMode);
    if (vFp == NULL) {
      throw std::runtime_error(fFormat("failed to open %s: %s", vFname.c_str(), strerror(errno)));
    }
    mSeek(0, SEEK_END);
    vSize = mTell();
    mSeek(0, SEEK_SET);
  }

  impl(FILE *pFile) : vFname("(file*)"), vOwnsFp(false) {
    vFp = pFile;
    mSeek(0, SEEK_END);
    vSize = mTell();
    mSeek(0, SEEK_SET);
  }

  size_t mTell() const {
    if (vFd == -1) {
      off_t vRet = mgs_mmap_ftell(vFp);
      if (vRet == -1) {
        throw std::runtime_error(fFormat("ftell error: %s", strerror(errno)));
      }

      return (size_t)vRet;
    }

    off_t vPos = lseek(vFd, 0, SEEK_CUR);
    if (vPos == -1) {
      throw std::runtime_error(fFormat("lseek error: %s", strerror(errno)));
    }
    return (size_t)vPos;
  }

  void mSeek(size_t pOffset, int pWhence) const {
    off_t vRet = 0;
    if (vFd == -1) {
      vRet = mgs_mmap_fseek(vFp, pOffset, pWhence);
    } else {
      vRet = lseek(vFd, pOffset, pWhence);
    }
    if (vRet == -1) {
      throw std::runtime_error(fFormat("seek error: %s", strerror(errno)));
    }
  }

  void mReadRawUnsafe(void *pPtr, size_t pLen) {
    if (pLen == 0) {
      return;
    }
    errno = 0;
    if (vFd == -1) {
      const size_t cCurrOff = mTell();
      const size_t cToRead = std::min(pLen, vSize - cCurrOff);

      std::size_t vRet = std::fread(pPtr, cToRead, 1, vFp);
      if (ferror(vFp)) {
        throw std::runtime_error(fFormat("read error: %s", strerror(errno)));
      }
      if (cToRead > 0 && vRet != 1) {
        throw std::runtime_error("unexpectedly reached end of file");
      }
    } else {
      size_t vBytesRead = 0;
      while (vBytesRead < pLen) {
        const size_t cToRead = pLen - vBytesRead;
        ssize_t vRet = ::read(vFd, reinterpret_cast<char *>(pPtr) + vBytesRead, cToRead);

        if (vRet == -1) {
          if (errno == EINTR) {
            continue; // Interrupted by signal, retry
          }
          // Fallback to std::fread in case the DMA controller cannot access the buffer
          if (errno == EFAULT || errno == EINVAL) {
            MGS_LOG_WARN("%s: Falling back to buffered IO due to %s\n", __func__, strerror(errno));
            auto vCurrOff = mTell();
            close(vFd);
            vFd = -1;
            vAlignment = 1;
            mInitFp("rb");
            mSeek(vCurrOff, SEEK_SET);
            mReadRawUnsafe(pPtr, pLen);
            return;
          }
          throw std::runtime_error(fFormat("read error: %s", strerror(errno)));
        }
        if (vRet == 0) {
          // EOF: allow if this read was only pulling alignment padding past file end
          off_t vPos = lseek(vFd, 0, SEEK_CUR);
          if (vPos != -1 && (size_t)vPos == vSize) {
            std::memset(reinterpret_cast<char *>(pPtr) + vBytesRead, 0, pLen - vBytesRead);
            return;
          }
          throw std::runtime_error("unexpectedly reached end of file");
        }

        vBytesRead += (size_t)vRet;
      }
    }
  }

  void mReadAlignedChunk(void *pDest, size_t pSize) {
    size_t vOffset = mTell();
    off_t vAlignedOffset = vOffset & ~(vAlignment - 1);
    off_t vOffsetFromAlignment = vOffset - vAlignedOffset;
    size_t vBytesToRead = (vOffsetFromAlignment + pSize + vAlignment - 1) & ~(vAlignment - 1);

    // stage through a bounded buffer, so that a large tensor is not held in memory twice while it loads
    const size_t cBufferSize = std::min<size_t>(vBytesToRead, MGS_DIRECT_IO_BUFFER_SIZE);

    void *vRawBuffer = nullptr;
    int vRet = posix_memalign(&vRawBuffer, vAlignment, cBufferSize);
    if (vRet != 0) {
      throw std::runtime_error(fFormat("posix_memalign failed with error %d", vRet));
    }

    struct aligned_buffer_deleter {
      void operator()(void *pP) const { free(pP); }
    };
    std::unique_ptr<void, aligned_buffer_deleter> vBuffer(vRawBuffer);

    mSeek(vAlignedOffset, SEEK_SET);

    size_t vSkip = vOffsetFromAlignment;
    size_t vCopied = 0;
    for (size_t vDone = 0; vDone < vBytesToRead;) {
      const size_t cN = std::min(cBufferSize, vBytesToRead - vDone);
      mReadRawUnsafe(vBuffer.get(), cN);

      const size_t cCount = std::min(cN - vSkip, pSize - vCopied);
      memcpy(reinterpret_cast<char *>(pDest) + vCopied, reinterpret_cast<char *>(vBuffer.get()) + vSkip, cCount);

      vCopied += cCount;
      vSkip = 0;
      vDone += cN;
    }
  }

  void mReadRaw(void *pPtr, size_t pLen) {
    if (mHasDirectIo()) {
      mReadAlignedChunk(pPtr, pLen);
    } else {
      mReadRawUnsafe(pPtr, pLen);
    }
  }

  uint32_t mReadU32() {
    uint32_t vRet;
    mReadRaw(&vRet, sizeof(vRet));
    return vRet;
  }

  void mWriteRaw(const void *pPtr, size_t pLen) const {
    if (pLen == 0) {
      return;
    }
    errno = 0;
    size_t vRet = std::fwrite(pPtr, pLen, 1, vFp);
    if (vRet != 1) {
      throw std::runtime_error(fFormat("write error: %s", strerror(errno)));
    }
  }

  void mWriteU32(uint32_t pVal) const { mWriteRaw(&pVal, sizeof(pVal)); }

  bool mHasDirectIo() const { return vFd != -1 && vAlignment > 1; }

  ~impl() {
    if (vFd != -1) {
      close(vFd);
    } else if (vOwnsFp) {
      std::fclose(vFp);
    }
  }
  int vFd = -1;
  std::string vFname;
#endif

  size_t mReadAlignment() const { return vAlignment; }

  size_t vAlignment = 1;

  FILE *vFp{};
  size_t vSize{};
  bool vOwnsFp = true;
};

mgs_file::mgs_file(const char *pFname, const char *pMode, const bool pUseDirectIo)
    : vPimpl(std::make_unique<impl>(pFname, pMode, pUseDirectIo)) {}

mgs_file::mgs_file(FILE *pFile) : vPimpl(std::make_unique<impl>(pFile)) {}

mgs_file::~mgs_file() = default;

size_t mgs_file::mTell() const { return vPimpl->mTell(); }
size_t mgs_file::size() const { return vPimpl->vSize; }

size_t mgs_file::mReadAlignment() const { return vPimpl->mReadAlignment(); }
bool mgs_file::mHasDirectIo() const { return vPimpl->mHasDirectIo(); }

int mgs_file::mFileId() const {
#ifdef _WIN32
  return _fileno(vPimpl->vFp);
#else
  if (vPimpl->vFd != -1) {
    return vPimpl->vFd;
  }
#if defined(fileno)
  return fileno(vPimpl->vFp);
#else
  return ::fileno(vPimpl->vFp);
#endif
#endif
}

void mgs_file::mSeek(size_t pOffset, int pWhence) const { vPimpl->mSeek(pOffset, pWhence); }
void mgs_file::mReadRaw(void *pPtr, size_t pLen) { vPimpl->mReadRaw(pPtr, pLen); }
#ifdef _WIN32
void mgs_file::mReadRawUnsafe(void *pPtr, size_t pLen) { vPimpl->mReadRaw(pPtr, pLen); }
#else
void mgs_file::mReadRawUnsafe(void *pPtr, size_t pLen) { vPimpl->mReadRawUnsafe(pPtr, pLen); }
#endif

uint32_t mgs_file::mReadU32() { return vPimpl->mReadU32(); }

void mgs_file::mWriteRaw(const void *pPtr, size_t pLen) const { vPimpl->mWriteRaw(pPtr, pLen); }
void mgs_file::mWriteU32(uint32_t pVal) const { vPimpl->mWriteU32(pVal); }

// mgs_mmap

struct mgs_mmap::impl {
#ifdef _POSIX_MAPPED_FILES
  std::vector<std::pair<size_t, size_t>> lMappedFragments;

  impl(struct mgs_file *pFile, size_t pPrefetch, bool pNuma) {
    vSize = pFile->size();
    int vFd = pFile->mFileId();
    int vFlags = MAP_SHARED;
    if (pNuma) {
      pPrefetch = 0;
    }
#ifdef __linux__
    if (posix_fadvise(vFd, 0, 0, POSIX_FADV_SEQUENTIAL)) {
      MGS_LOG_WARN("warning: posix_fadvise(.., POSIX_FADV_SEQUENTIAL) failed: %s\n", strerror(errno));
    }
    // populate only when the whole file is prefetched; a partial prefetch only advises the kernel below
    if (pPrefetch >= pFile->size()) {
      vFlags |= MAP_POPULATE;
    }
#endif
    vAddr = mmap(NULL, pFile->size(), PROT_READ, vFlags, vFd, 0);
    if (vAddr == MAP_FAILED) {
      throw std::runtime_error(fFormat("mmap failed: %s", strerror(errno)));
    }

    if (pPrefetch > 0) {
      if (posix_madvise(vAddr, std::min(pFile->size(), pPrefetch), POSIX_MADV_WILLNEED)) {
        MGS_LOG_WARN("warning: posix_madvise(.., POSIX_MADV_WILLNEED) failed: %s\n", strerror(errno));
      }
    }
    if (pNuma) {
      if (posix_madvise(vAddr, pFile->size(), POSIX_MADV_RANDOM)) {
        MGS_LOG_WARN("warning: posix_madvise(.., POSIX_MADV_RANDOM) failed: %s\n", strerror(errno));
      }
    }

    lMappedFragments.emplace_back(0, pFile->size());
  }

  static void mAlignRange(size_t *pFirst, size_t *pLast, size_t pPageSize) {
    size_t vOffsetInPage = *pFirst & (pPageSize - 1);
    size_t vOffsetToPage = vOffsetInPage == 0 ? 0 : pPageSize - vOffsetInPage;
    *pFirst += vOffsetToPage;

    *pLast = *pLast & ~(pPageSize - 1);

    if (*pLast <= *pFirst) {
      *pLast = *pFirst;
    }
  }

  void mUnmapFragment(size_t pFirst, size_t pLast) {
    int vPageSize = sysconf(_SC_PAGESIZE);
    mAlignRange(&pFirst, &pLast, vPageSize);
    size_t vLen = pLast - pFirst;

    if (vLen == 0) {
      return;
    }

    GGML_ASSERT(pFirst % vPageSize == 0);
    GGML_ASSERT(pLast % vPageSize == 0);
    GGML_ASSERT(pLast > pFirst);

    void *vNextPageStart = (uint8_t *)vAddr + pFirst;

    if (munmap(vNextPageStart, vLen)) {
      MGS_LOG_WARN("warning: munmap failed: %s\n", strerror(errno));
    }

    std::vector<std::pair<size_t, size_t>> lNewMappedFragments;
    for (const auto &vFrag : lMappedFragments) {
      if (vFrag.first < pFirst && vFrag.second > pLast) {
        lNewMappedFragments.emplace_back(vFrag.first, pFirst);
        lNewMappedFragments.emplace_back(pLast, vFrag.second);
      } else if (vFrag.first < pFirst && vFrag.second > pFirst) {
        lNewMappedFragments.emplace_back(vFrag.first, pFirst);
      } else if (vFrag.first < pLast && vFrag.second > pLast) {
        lNewMappedFragments.emplace_back(pLast, vFrag.second);
      } else if (vFrag.first >= pFirst && vFrag.second <= pLast) {
      } else {
        lNewMappedFragments.push_back(vFrag);
      }
    }
    lMappedFragments = std::move(lNewMappedFragments);
  }

  ~impl() {
    for (const auto &vFrag : lMappedFragments) {
      if (munmap((char *)vAddr + vFrag.first, vFrag.second - vFrag.first)) {
        MGS_LOG_WARN("warning: munmap failed: %s\n", strerror(errno));
      }
    }
  }
#elif defined(_WIN32)
  HANDLE vHMapping = nullptr;

  impl(struct mgs_file *pFile, size_t pPrefetch, bool pNuma) {
    GGML_UNUSED(pNuma);

    vSize = pFile->size();

    HANDLE vHFile = (HANDLE)_get_osfhandle(pFile->mFileId());

    vHMapping = CreateFileMappingA(vHFile, NULL, PAGE_READONLY, 0, 0, NULL);

    if (vHMapping == NULL) {
      DWORD vError = GetLastError();
      throw std::runtime_error(fFormat("CreateFileMappingA failed: %s", fMgsFormatWinErr(vError).c_str()));
    }

    vAddr = MapViewOfFile(vHMapping, FILE_MAP_READ, 0, 0, 0);
    DWORD vError = GetLastError();

    if (vAddr == NULL) {
      CloseHandle(vHMapping);
      throw std::runtime_error(fFormat("MapViewOfFile failed: %s", fMgsFormatWinErr(vError).c_str()));
    }

    if (pPrefetch > 0) {
#if _WIN32_WINNT >= 0x602
      BOOL(WINAPI * vPrefetchVirtualMemory)(HANDLE, ULONG_PTR, PWIN32_MEMORY_RANGE_ENTRY, ULONG);
      HMODULE vHKernel32 = GetModuleHandleW(L"kernel32.dll");

      vPrefetchVirtualMemory =
          (decltype(vPrefetchVirtualMemory))(void *)GetProcAddress(vHKernel32, "PrefetchVirtualMemory");

      if (vPrefetchVirtualMemory) {
        WIN32_MEMORY_RANGE_ENTRY vRange;
        vRange.VirtualAddress = vAddr;
        vRange.NumberOfBytes = (SIZE_T)std::min(vSize, pPrefetch);
        if (!vPrefetchVirtualMemory(GetCurrentProcess(), 1, &vRange, 0)) {
          MGS_LOG_WARN("warning: PrefetchVirtualMemory failed: %s\n", fMgsFormatWinErr(GetLastError()).c_str());
        }
      }
#else
      MGS_LOG_DEBUG("skipping PrefetchVirtualMemory because _WIN32_WINNT < 0x602\n");
#endif
    }
  }

  void mUnmapFragment(size_t pFirst, size_t pLast) {
    GGML_UNUSED(pFirst);
    GGML_UNUSED(pLast);
  }

  ~impl() {
    if (vHMapping) {
      if (vAddr) {
        if (!UnmapViewOfFile(vAddr)) {
          MGS_LOG_WARN("warning: UnmapViewOfFile failed: %s\n", fMgsFormatWinErr(GetLastError()).c_str());
        }
      }
      if (!CloseHandle(vHMapping)) {
        MGS_LOG_WARN("warning: CloseHandle failed: %s\n", fMgsFormatWinErr(GetLastError()).c_str());
      }
    }
  }
#else
  impl(struct mgs_file *pFile, size_t pPrefetch, bool pNuma) {
    GGML_UNUSED(pFile);
    GGML_UNUSED(pPrefetch);
    GGML_UNUSED(pNuma);

    throw std::runtime_error("mmap not supported");
  }

  void mUnmapFragment(size_t pFirst, size_t pLast) {
    GGML_UNUSED(pFirst);
    GGML_UNUSED(pLast);

    throw std::runtime_error("mmap not supported");
  }
#endif

  void *vAddr;
  size_t vSize;
};

mgs_mmap::mgs_mmap(struct mgs_file *pFile, size_t pPrefetch, bool pNuma)
    : vPimpl(std::make_unique<impl>(pFile, pPrefetch, pNuma)) {}
mgs_mmap::~mgs_mmap() {
  // Release the pages before the implementation destructor unmaps them.
  if (vHostRegisteredAddress && vHostUnregister) {
    vHostUnregister(vHostRegisteredAddress);
  }
}

size_t mgs_mmap::size() const { return vPimpl->vSize; }
void *mgs_mmap::mAddr() const { return vPimpl->vAddr; }

void mgs_mmap::mUnmapFragment(size_t pFirst, size_t pLast) { vPimpl->mUnmapFragment(pFirst, pLast); }

size_t mgs_mmap::mRegisterHost(size_t pFirst, size_t pLast, bool (*pRegister)(void *, size_t),
                               void (*pUnregister)(void *)) {
#ifdef _POSIX_MAPPED_FILES
  if (vHostRegisteredAddress || !pRegister || !pUnregister || pLast <= pFirst) {
    return 0;
  }

  // Expand the interval to the page boundaries retained by mUnmapFragment().
  const size_t cPageSize = sysconf(_SC_PAGESIZE);
  pFirst = pFirst & ~(cPageSize - 1);
  pLast = (pLast + cPageSize - 1) & ~(cPageSize - 1);

  void *vRegisteredAddress = (uint8_t *)vPimpl->vAddr + pFirst;
  if (!pRegister(vRegisteredAddress, pLast - pFirst)) {
    return 0;
  }

  vHostRegisteredAddress = vRegisteredAddress;
  vHostUnregister = pUnregister;
  return pLast - pFirst;
#else
  GGML_UNUSED(pFirst);
  GGML_UNUSED(pLast);
  GGML_UNUSED(pRegister);
  GGML_UNUSED(pUnregister);
  return 0;
#endif
}

#if defined(_POSIX_MEMLOCK_RANGE) || defined(_WIN32)
const bool mgs_mmap::cSupported = true;
#else
const bool mgs_mmap::cSupported = false;
#endif

// mgs_mlock

struct mgs_mlock::impl {
#ifdef _POSIX_MEMLOCK_RANGE
  static size_t mLockGranularity() { return (size_t)sysconf(_SC_PAGESIZE); }

  bool mRawLock(const void *pAddr, size_t pSize) const {
    if (!mlock(pAddr, pSize)) {
      return true;
    }

#ifdef __APPLE__
#define MLOCK_SUGGESTION                                                                                               \
  "Try increasing the sysctl values 'vm.user_wire_limit' and 'vm.global_user_wire_limit' and/or "                      \
  "decreasing 'vm.global_no_user_wire_amount'.  Also try increasing RLIMIT_MEMLOCK (ulimit -l).\n"
#else
#define MLOCK_SUGGESTION "Try increasing RLIMIT_MEMLOCK ('ulimit -l' as root).\n"
#endif

    char *vErrmsg = std::strerror(errno);
    bool vSuggest = (errno == ENOMEM);
#if defined(TARGET_OS_VISION) || defined(TARGET_OS_TV) || defined(_AIX) || defined(__HAIKU__)
    // visionOS/tvOS/Haiku don't support RLIMIT_MEMLOCK
    // Skip resource limit checks on these platforms
    vSuggest = false;
#else
    struct rlimit vLockLimit;
    if (vSuggest && getrlimit(RLIMIT_MEMLOCK, &vLockLimit)) {
      vSuggest = false;
    }
    if (vSuggest && ((uint64_t)vLockLimit.rlim_max > (uint64_t)vLockLimit.rlim_cur + pSize)) {
      vSuggest = false;
    }
#endif

    MGS_LOG_WARN("warning: failed to mlock %zu-byte buffer (after previously locking %zu bytes): %s\n%s", pSize,
                 this->vSize, vErrmsg, vSuggest ? MLOCK_SUGGESTION : "");
    return false;
  }

  static void mRawUnlock(void *pAddr, size_t pSize) {
    if (munlock(pAddr, pSize)) {
      MGS_LOG_WARN("warning: failed to munlock buffer: %s\n", std::strerror(errno));
    }
  }
#elif defined(_WIN32)
  static size_t mLockGranularity() {
    SYSTEM_INFO vSi;
    GetSystemInfo(&vSi);
    return (size_t)vSi.dwPageSize;
  }

  bool mRawLock(void *pPtr, size_t pLen) const {
    for (int vTries = 1;; vTries++) {
      if (VirtualLock(pPtr, pLen)) {
        return true;
      }
      if (vTries == 2) {
        MGS_LOG_WARN("warning: failed to VirtualLock %zu-byte buffer (after previously locking %zu bytes): %s\n", pLen,
                     vSize, fMgsFormatWinErr(GetLastError()).c_str());
        return false;
      }

      SIZE_T vMinWsSize, vMaxWsSize;
      if (!GetProcessWorkingSetSize(GetCurrentProcess(), &vMinWsSize, &vMaxWsSize)) {
        MGS_LOG_WARN("warning: GetProcessWorkingSetSize failed: %s\n", fMgsFormatWinErr(GetLastError()).c_str());
        return false;
      }
      size_t vIncrement = pLen + 1048576;
      vMinWsSize += vIncrement;
      vMaxWsSize += vIncrement;
      if (!SetProcessWorkingSetSize(GetCurrentProcess(), vMinWsSize, vMaxWsSize)) {
        MGS_LOG_WARN("warning: SetProcessWorkingSetSize failed: %s\n", fMgsFormatWinErr(GetLastError()).c_str());
        return false;
      }
    }
  }

  static void mRawUnlock(void *pPtr, size_t pLen) {
    if (!VirtualUnlock(pPtr, pLen)) {
      MGS_LOG_WARN("warning: failed to VirtualUnlock buffer: %s\n", fMgsFormatWinErr(GetLastError()).c_str());
    }
  }
#else
  static size_t mLockGranularity() { return (size_t)65536; }

  bool mRawLock(const void *pAddr, size_t pLen) const {
    MGS_LOG_WARN("warning: mlock not supported on this system\n");
    return false;
  }

  static void mRawUnlock(const void *pAddr, size_t pLen) {}
#endif

  impl() : vAddr(NULL), vSize(0), vFailedAlready(false) {}

  void mInit(void *pPtr) {
    GGML_ASSERT(vAddr == NULL && vSize == 0);
    vAddr = pPtr;
  }

  void mGrowTo(size_t pTargetSize) {
    GGML_ASSERT(vAddr);
    if (vFailedAlready) {
      return;
    }
    size_t vGranularity = mLockGranularity();
    pTargetSize = (pTargetSize + vGranularity - 1) & ~(vGranularity - 1);
    if (pTargetSize > vSize) {
      if (mRawLock((uint8_t *)vAddr + vSize, pTargetSize - vSize)) {
        vSize = pTargetSize;
      } else {
        vFailedAlready = true;
      }
    }
  }

  void *vAddr;
  size_t vSize;

  bool vFailedAlready;
};

mgs_mlock::mgs_mlock() : vPimpl(std::make_unique<impl>()) {}
mgs_mlock::~mgs_mlock() = default;

void mgs_mlock::mInit(void *pPtr) { vPimpl->mInit(pPtr); }
void mgs_mlock::mGrowTo(size_t pTargetSize) { vPimpl->mGrowTo(pTargetSize); }

#if defined(_POSIX_MEMLOCK_RANGE) || defined(_WIN32)
const bool mgs_mlock::cSupported = true;
#else
const bool mgs_mlock::cSupported = false;
#endif

size_t mgs_path_max() { return PATH_MAX; }

void fMgsPrefetch(mgs_memory_ranges pMr) {
#if defined(__linux__) || (defined(_WIN32) && _WIN32_WINNT >= 0x602)
  if (pMr.empty()) {
    return;
  }

#if defined(_WIN32)
  using prefetch_virtual_memory_t = BOOL(WINAPI *)(HANDLE, ULONG_PTR, PWIN32_MEMORY_RANGE_ENTRY, ULONG);
  static const auto cPrefetchVirtualMemory = (prefetch_virtual_memory_t)(void *)GetProcAddress(
      GetModuleHandleW(L"kernel32.dll"), "PrefetchVirtualMemory");
  if (!cPrefetchVirtualMemory) {
    return;
  }

  static const long cPageSize = [] {
    SYSTEM_INFO vInfo;
    GetSystemInfo(&vInfo);
    return (long)vInfo.dwPageSize;
  }();
#else
  static const long cPageSize = sysconf(_SC_PAGESIZE);
#endif
  if (cPageSize <= 0) {
    return;
  }

  const size_t cPage = (size_t)cPageSize;
  std::sort(pMr.begin(), pMr.end(), [](const mgs_memory_range &pA, const mgs_memory_range &pB) {
    return (uintptr_t)pA.vAddr < (uintptr_t)pB.vAddr;
  });

  uintptr_t vBegin = 0;
  uintptr_t vEnd = 0;
#if defined(_WIN32)
  // collect the ranges and prefetch them in one call, so the reads can be issued concurrently
  std::vector<WIN32_MEMORY_RANGE_ENTRY> lEntries;
  auto vPrefetch = [&]() {
    lEntries.push_back({(PVOID)vBegin, (SIZE_T)(vEnd - vBegin)});
    return true;
  };
#else
  auto vPrefetch = [&]() {
    if (madvise((void *)vBegin, vEnd - vBegin, MADV_WILLNEED) != 0) {
      MGS_LOG_WARN("fMgsPrefetch: madvise(MADV_WILLNEED) failed: %s\n", strerror(errno));
      return false;
    }
    return true;
  };
#endif
  for (const auto &vRange : pMr) {
    if (!vRange.vAddr || vRange.vSize == 0) {
      continue;
    }
    const uintptr_t cPointer = (uintptr_t)vRange.vAddr;
    const uintptr_t cFirst = cPointer / cPage * cPage;
    const uintptr_t cLast = (cPointer + vRange.vSize + cPage - 1) / cPage * cPage;
    if (vEnd && cFirst > vEnd) {
      if (!vPrefetch()) {
        return;
      }
      vEnd = 0;
    }
    if (!vEnd) {
      vBegin = cFirst;
    }
    vEnd = std::max(vEnd, cLast);
  }
  if (vEnd) {
    vPrefetch();
  }
#if defined(_WIN32)
  if (!lEntries.empty() &&
      !cPrefetchVirtualMemory(GetCurrentProcess(), (ULONG_PTR)lEntries.size(), lEntries.data(), 0)) {
    MGS_LOG_WARN("fMgsPrefetch: PrefetchVirtualMemory failed: %s\n", fMgsFormatWinErr(GetLastError()).c_str());
  }
#endif
#else
  GGML_UNUSED(pMr);
#endif
}
