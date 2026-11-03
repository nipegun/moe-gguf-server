#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-impl.h"
#include "gguf.h"

#include <cinttypes>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <new>
#include <stdexcept>
#include <string>
#include <vector>

#define GGUF_MAX_STRING_LENGTH (1024 * 1024 * 1024)
#define GGUF_MAX_ARRAY_ELEMENTS (1024 * 1024 * 1024)

#ifdef _WIN32
#define gguf_ftell _ftelli64
#define gguf_fseek _fseeki64
#else
#define gguf_ftell ftello
#define gguf_fseek fseeko
#endif

template <typename T> struct type_to_gguf_type;

template <> struct type_to_gguf_type<uint8_t> {
  static constexpr enum gguf_type value = GGUF_TYPE_UINT8;
};

template <> struct type_to_gguf_type<int8_t> {
  static constexpr enum gguf_type value = GGUF_TYPE_INT8;
};

template <> struct type_to_gguf_type<uint16_t> {
  static constexpr enum gguf_type value = GGUF_TYPE_UINT16;
};

template <> struct type_to_gguf_type<int16_t> {
  static constexpr enum gguf_type value = GGUF_TYPE_INT16;
};

template <> struct type_to_gguf_type<uint32_t> {
  static constexpr enum gguf_type value = GGUF_TYPE_UINT32;
};

template <> struct type_to_gguf_type<int32_t> {
  static constexpr enum gguf_type value = GGUF_TYPE_INT32;
};

template <> struct type_to_gguf_type<float> {
  static constexpr enum gguf_type value = GGUF_TYPE_FLOAT32;
};

template <> struct type_to_gguf_type<bool> {
  static constexpr enum gguf_type value = GGUF_TYPE_BOOL;
};

template <> struct type_to_gguf_type<std::string> {
  static constexpr enum gguf_type value = GGUF_TYPE_STRING;
};

template <> struct type_to_gguf_type<uint64_t> {
  static constexpr enum gguf_type value = GGUF_TYPE_UINT64;
};

template <> struct type_to_gguf_type<int64_t> {
  static constexpr enum gguf_type value = GGUF_TYPE_INT64;
};

template <> struct type_to_gguf_type<double> {
  static constexpr enum gguf_type value = GGUF_TYPE_FLOAT64;
};

static const std::map<gguf_type, size_t> GGUF_TYPE_SIZE = {
    {GGUF_TYPE_UINT8, sizeof(uint8_t)},
    {GGUF_TYPE_INT8, sizeof(int8_t)},
    {GGUF_TYPE_UINT16, sizeof(uint16_t)},
    {GGUF_TYPE_INT16, sizeof(int16_t)},
    {GGUF_TYPE_UINT32, sizeof(uint32_t)},
    {GGUF_TYPE_INT32, sizeof(int32_t)},
    {GGUF_TYPE_FLOAT32, sizeof(float)},
    {GGUF_TYPE_BOOL, sizeof(int8_t)},
    {GGUF_TYPE_STRING, 0}, // undefined
    {GGUF_TYPE_ARRAY, 0},  // undefined
    {GGUF_TYPE_UINT64, sizeof(uint64_t)},
    {GGUF_TYPE_INT64, sizeof(int64_t)},
    {GGUF_TYPE_FLOAT64, sizeof(double)},
};
static_assert(GGUF_TYPE_COUNT == 13, "GGUF_TYPE_COUNT != 13");

static const std::map<gguf_type, const char *> GGUF_TYPE_NAME = {
    {GGUF_TYPE_UINT8, "u8"},    {GGUF_TYPE_INT8, "i8"},   {GGUF_TYPE_UINT16, "u16"},  {GGUF_TYPE_INT16, "i16"},
    {GGUF_TYPE_UINT32, "u32"},  {GGUF_TYPE_INT32, "i32"}, {GGUF_TYPE_FLOAT32, "f32"}, {GGUF_TYPE_BOOL, "bool"},
    {GGUF_TYPE_STRING, "str"},  {GGUF_TYPE_ARRAY, "arr"}, {GGUF_TYPE_UINT64, "u64"},  {GGUF_TYPE_INT64, "i64"},
    {GGUF_TYPE_FLOAT64, "f64"},
};
static_assert(GGUF_TYPE_COUNT == 13, "GGUF_TYPE_COUNT != 13");

size_t gguf_type_size(enum gguf_type pType) {
  auto vIt = GGUF_TYPE_SIZE.find(pType);
  return vIt == GGUF_TYPE_SIZE.end() ? 0 : vIt->second;
}

struct gguf_kv {
  std::string vKey;

  bool vIsArray;
  enum gguf_type vType;

  std::vector<int8_t> data;
  std::vector<std::string> vDataString;

  template <typename T>
  gguf_kv(const std::string &pKey, const T value) : vKey(pKey), vIsArray(false), vType(type_to_gguf_type<T>::value) {
    GGML_ASSERT(!pKey.empty());
    data.resize(sizeof(T));
    memcpy(data.data(), &value, sizeof(T));
  }

  template <typename T>
  gguf_kv(const std::string &pKey, const std::vector<T> &value)
      : vKey(pKey), vIsArray(true), vType(type_to_gguf_type<T>::value) {
    GGML_ASSERT(!pKey.empty());
    data.resize(value.size() * sizeof(T));
    for (size_t vI = 0; vI < value.size(); ++vI) {
      const T cTmp = value[vI];
      memcpy(data.data() + vI * sizeof(T), &cTmp, sizeof(T));
    }
  }

  gguf_kv(const std::string &pKey, const std::string &pValue) : vKey(pKey), vIsArray(false), vType(GGUF_TYPE_STRING) {
    GGML_ASSERT(!pKey.empty());
    vDataString.push_back(pValue);
  }

  gguf_kv(const std::string &pKey, const std::vector<std::string> &pValue)
      : vKey(pKey), vIsArray(true), vType(GGUF_TYPE_STRING) {
    GGML_ASSERT(!pKey.empty());
    vDataString = pValue;
  }

  const std::string &mGetKey() const { return vKey; }

  const enum gguf_type &mGetType() const { return vType; }

  size_t mGetNe() const {
    if (vType == GGUF_TYPE_STRING) {
      const size_t cNe = vDataString.size();
      GGML_ASSERT(vIsArray || cNe == 1);
      return cNe;
    }
    const size_t cTypeSize = gguf_type_size(vType);
    GGML_ASSERT(data.size() % cTypeSize == 0);
    const size_t cNe = data.size() / cTypeSize;
    GGML_ASSERT(vIsArray || cNe == 1);
    return cNe;
  }

  template <typename T> const T &mGetVal(const size_t pI = 0) const {
    GGML_ASSERT(type_to_gguf_type<T>::value == vType);
    if constexpr (std::is_same<T, std::string>::value) {
      GGML_ASSERT(vDataString.size() >= pI + 1);
      return vDataString[pI];
    }
    const size_t cTypeSize = gguf_type_size(vType);
    GGML_ASSERT(data.size() % cTypeSize == 0);
    GGML_ASSERT(data.size() >= (pI + 1) * cTypeSize);
    return reinterpret_cast<const T *>(data.data())[pI];
  }

  void mCast(const enum gguf_type pNewType) {
    const size_t cNewTypeSize = gguf_type_size(pNewType);
    GGML_ASSERT(data.size() % cNewTypeSize == 0);
    vType = pNewType;
  }
};

struct gguf_tensor_info {
  struct ggml_tensor vT; // for holding the equivalent info
  uint64_t vOffset;      // offset from start of `data`, must be a multiple of `ALIGNMENT`
};

struct gguf_context {
  uint32_t vVersion = GGUF_VERSION;

  std::vector<struct gguf_kv> kv;
  std::vector<struct gguf_tensor_info> vInfo;

  size_t vAlignment = GGUF_DEFAULT_ALIGNMENT;
  size_t vOffset = 0; // offset of `data` from beginning of file
  size_t size = 0;    // size of `data` in bytes

  void *data = nullptr;
};

struct gguf_reader {
  gguf_reader(gguf_reader_callback_t pCallback, void *pUserdata, size_t pMaxChunkRead, uint64_t pDataOffset = 0,
              uint64_t pNbytesRemain = 0)
      : vCallback(pCallback), vUserdata(pUserdata), vMaxChunkRead(pMaxChunkRead), vDataOffset(pDataOffset),
        vNbytesRemain(pNbytesRemain) {
    GGML_ASSERT(pMaxChunkRead > 0);
  }

  // helper for remaining bytes in a file
  static uint64_t mFileRemain(FILE *pFile) {
    const int64_t cCur = gguf_ftell(pFile);
    if (cCur < 0) {
      return 0;
    }
    if (gguf_fseek(pFile, 0, SEEK_END) != 0) {
      gguf_fseek(pFile, cCur, SEEK_SET);

      return 0;
    }
    const int64_t cEnd = gguf_ftell(pFile);
    if (cEnd < 0) {
      gguf_fseek(pFile, cCur, SEEK_SET);

      return 0;
    }
    gguf_fseek(pFile, cCur, SEEK_SET);
    return static_cast<uint64_t>(cEnd - cCur);
  }

  template <typename T> bool read(T &pDst) const {
    const size_t cSize = sizeof(pDst);
    if (cSize > vNbytesRemain) {
      return false;
    }
    return mReadRaw(&pDst, cSize) == cSize;
  }

  template <typename T> bool read(std::vector<T> &pDst, const size_t pN) const {
    if (pN > GGUF_MAX_ARRAY_ELEMENTS) {
      return false;
    }
    if constexpr (std::is_same<T, std::string>::value) {
      // strings are prefixed with their length, so we need to account for that
      if (pN > SIZE_MAX / sizeof(uint64_t)) {
        return false;
      }
      if (vNbytesRemain < pN * sizeof(uint64_t)) {
        return false;
      }
    } else {
      if (pN > SIZE_MAX / sizeof(T)) {
        return false;
      }
      if (vNbytesRemain < pN * sizeof(T)) {
        return false;
      }
    }
    pDst.resize(pN);
    for (size_t vI = 0; vI < pDst.size(); ++vI) {
      if constexpr (std::is_same<T, bool>::value) {
        bool vTmp;
        if (!read(vTmp)) {
          return false;
        }
        pDst[vI] = vTmp;
      } else {
        if (!read(pDst[vI])) {
          return false;
        }
      }
    }
    return true;
  }

  bool read(bool &pDst) const {
    int8_t vTmp = -1;
    if (!read(vTmp)) {
      return false;
    }
    pDst = vTmp != 0;
    return true;
  }

  bool read(enum ggml_type &pDst) const {
    int32_t vTmp = -1;
    if (!read(vTmp)) {
      return false;
    }
    pDst = ggml_type(vTmp);
    return true;
  }

  bool read(enum gguf_type &pDst) const {
    int32_t vTmp = -1;
    if (!read(vTmp)) {
      return false;
    }
    pDst = gguf_type(vTmp);
    return true;
  }

  bool read(std::string &pDst) const {
    uint64_t vSize = 0;
    if (!read(vSize)) {
      return false;
    }
    if (vSize > GGUF_MAX_STRING_LENGTH) {
      GGML_LOG_ERROR("%s: string length %" PRIu64 " exceeds maximum %" PRIu64 "\n", __func__, vSize,
                     (uint64_t)GGUF_MAX_STRING_LENGTH);
      return false;
    }
    if (vSize > vNbytesRemain) {
      GGML_LOG_ERROR("%s: string length %" PRIu64 " exceeds remaining file size %" PRIu64 " bytes\n", __func__, vSize,
                     vNbytesRemain);
      return false;
    }
    pDst.resize(static_cast<size_t>(vSize));
    return mReadRaw(pDst.data(), static_cast<size_t>(vSize)) == vSize;
  }

  bool read(void *pDst, const size_t pSize) const {
    if (pSize > vNbytesRemain) {
      return false;
    }
    return mReadRaw(pDst, pSize) == pSize;
  }

  uint64_t mTell() const { return vDataOffset; }

  bool mSeek(uint64_t pAbsoluteOffset) const {
    const uint64_t cEndOffset = uint64_t(vDataOffset) + vNbytesRemain;
    if (pAbsoluteOffset > cEndOffset) {
      return false;
    }

    vDataOffset = pAbsoluteOffset;
    vNbytesRemain = cEndOffset - pAbsoluteOffset;

    return true;
  }

private:
  size_t mReadRaw(void *pDst, size_t pSize) const {
    if (vCallback == nullptr || pSize == 0) {
      return 0;
    }

    uint8_t *vData = static_cast<uint8_t *>(pDst);
    size_t vTotalNread = 0;
    bool vReachedEof = false;

    while (vTotalNread < pSize) {
      const size_t cChunkSize = std::min(vMaxChunkRead, pSize - vTotalNread);
      if (vDataOffset + vTotalNread < vDataOffset) {
        break;
      }
      const size_t cNread =
          vCallback(vUserdata, static_cast<void *>(vData + vTotalNread), vDataOffset + vTotalNread, cChunkSize);
      vTotalNread += cNread;
      if (cNread != cChunkSize) {
        vReachedEof = true;
        break;
      }
    }

    vDataOffset += vTotalNread;
    GGML_ASSERT(vTotalNread <= vNbytesRemain);
    vNbytesRemain -= vTotalNread;

    if (vReachedEof) {
      vNbytesRemain = 0;
    }

    return vTotalNread;
  }

  gguf_reader_callback_t vCallback = nullptr;
  void *vUserdata = nullptr;
  size_t vMaxChunkRead = 0;
  mutable uint64_t vDataOffset = 0;
  mutable uint64_t vNbytesRemain = 0;
};

struct gguf_context *gguf_init_empty(void) { return new gguf_context; }

template <typename T>
bool fGgufReadEmplaceHelper(const struct gguf_reader &pGr, std::vector<struct gguf_kv> &kv, const std::string &pKey,
                            const bool pIsArray, const size_t pN) {
  if (pIsArray) {
    std::vector<T> vValue;
    try {
      if (!pGr.read(vValue, pN)) {
        return false;
      }
    } catch (std::length_error &) {
      GGML_LOG_ERROR("%s: encountered length_error while reading value for key '%s'\n", __func__, pKey.c_str());
      return false;
    } catch (std::bad_alloc &) {
      GGML_LOG_ERROR("%s: encountered bad_alloc error while reading value for key '%s'\n", __func__, pKey.c_str());
      return false;
    }
    kv.emplace_back(pKey, vValue);
  } else {
    T vValue;
    if (!pGr.read(vValue)) {
      return false;
    }
    kv.emplace_back(pKey, vValue);
  }
  return true;
}

static struct gguf_context *fGgufInitFromReader(const struct gguf_reader &pGr, struct gguf_init_params pParams) {
  struct gguf_context *vCtx = new gguf_context;

  bool vOk = true;

  // file magic
  {
    std::vector<char> lMagic;
    vOk = vOk && pGr.read(lMagic, 4);

    if (!vOk) {
      GGML_LOG_ERROR("%s: failed to read magic\n", __func__);
      gguf_free(vCtx);
      return nullptr;
    }

    for (uint32_t vI = 0; vI < lMagic.size(); vI++) {
      if (lMagic[vI] != GGUF_MAGIC[vI]) {
        char vC0 = isprint(lMagic[0]) ? lMagic[0] : '?';
        char vC1 = isprint(lMagic[1]) ? lMagic[1] : '?';
        char vC2 = isprint(lMagic[2]) ? lMagic[2] : '?';
        char vC3 = isprint(lMagic[3]) ? lMagic[3] : '?';
        GGML_LOG_ERROR("%s: invalid magic characters: '%c%c%c%c', expected 'GGUF'\n", __func__, vC0, vC1, vC2, vC3);
        gguf_free(vCtx);
        return nullptr;
      }
    }
  }

  // header
  int64_t vNKv = 0;
  int64_t vNTensors = 0;

  if (vOk && pGr.read(vCtx->vVersion)) {
    if (vOk && vCtx->vVersion == 0) {
      GGML_LOG_ERROR("%s: bad GGUF version: %" PRIu32 "\n", __func__, vCtx->vVersion);
      vOk = false;
    }

    /*
     * bit layout is different when reading non-native endian models.
     * assuming that the GGUF version is 3, the non-native endian model
     * would read it as 0x30000000. we can use the AND operation against
     * the last 4 hexadecimal digits to check if the model is the same
     * endianness as the host system.
     */
    if (vOk && (vCtx->vVersion & 0x0000FFFF) == 0x00000000) {
      GGML_LOG_ERROR("%s: failed to load model: this GGUF file version %" PRIu32
                     " is extremely large, is there a mismatch between the host and model endianness?\n",
                     __func__, vCtx->vVersion);
      vOk = false;
    }

    if (vOk && vCtx->vVersion < GGUF_VERSION) {
      GGML_LOG_ERROR("%s: GGUF version %" PRIu32 " is no longer supported, only version %d is: "
                     "convert the model again\n",
                     __func__, vCtx->vVersion, GGUF_VERSION);
      vOk = false;
    }
    if (vOk && vCtx->vVersion > GGUF_VERSION) {
      GGML_LOG_ERROR("%s: this GGUF file is version %" PRIu32 " but this software only supports up to version %d\n",
                     __func__, vCtx->vVersion, GGUF_VERSION);
      vOk = false;
    }
  } else {
    vOk = false;
  }

  if (vOk && pGr.read(vNTensors)) {
    static_assert(sizeof(size_t) <= 8 && sizeof(gguf_tensor_info) >= 2, "int64_t insufficient for indexing");
    if (vNTensors < 0 || vNTensors > int64_t(SIZE_MAX / sizeof(gguf_tensor_info))) {
      GGML_LOG_ERROR("%s: number of tensors is %" PRIi64 " but must be in [0, %zu]\n", __func__, vNTensors,
                     SIZE_MAX / sizeof(gguf_tensor_info));
      vOk = false;
    }
  } else {
    vOk = false;
  }

  if (vOk && pGr.read(vNKv)) {
    static_assert(sizeof(size_t) <= 8 && sizeof(gguf_tensor_info) >= 2, "int64_t insufficient for indexing");
    if (vNKv < 0 || vNKv > int64_t(SIZE_MAX / sizeof(gguf_kv))) {
      GGML_LOG_ERROR("%s: number of key value pairs is %" PRIi64 " but must be in [0, %zu]\n", __func__, vNKv,
                     SIZE_MAX / sizeof(gguf_kv));
      vOk = false;
    }
  } else {
    vOk = false;
  }

  if (!vOk) {
    GGML_LOG_ERROR("%s: failed to read header\n", __func__);
    gguf_free(vCtx);
    return nullptr;
  }

  // KV pairs
  {
    for (int64_t vI = 0; vOk && vI < vNKv; ++vI) {
      std::string vKey;
      gguf_type vType = gguf_type(-1);
      bool vIsArray = false;
      uint64_t vN = 1;

      try {
        vOk = vOk && pGr.read(vKey);
      } catch (std::length_error &) {
        GGML_LOG_ERROR("%s: encountered length_error while reading key %" PRIi64 "\n", __func__, vI);
        vOk = false;
      } catch (std::bad_alloc &) {
        GGML_LOG_ERROR("%s: encountered bad_alloc error while reading key %" PRIi64 "\n", __func__, vI);
        vOk = false;
      }
      if (vOk && vKey.empty()) {
        GGML_LOG_ERROR("%s: key %" PRIi64 " is empty\n", __func__, vI);
        vOk = false;
      }
      for (size_t vJ = 0; vOk && vJ < vCtx->kv.size(); ++vJ) {
        if (vKey == vCtx->kv[vJ].vKey) {
          GGML_LOG_ERROR("%s: duplicate key '%s' for tensors %zu and %" PRIi64 " \n", __func__, vKey.c_str(), vJ, vI);
          vOk = false;
        }
      }
      if (!vOk) {
        break;
      }

      vOk = vOk && pGr.read(vType);
      if (vType == GGUF_TYPE_ARRAY) {
        vIsArray = true;
        vOk = vOk && pGr.read(vType);
        vOk = vOk && pGr.read(vN);
      }
      if (!vOk) {
        break;
      }

      switch (vType) {
      case GGUF_TYPE_UINT8:
        vOk = vOk && fGgufReadEmplaceHelper<uint8_t>(pGr, vCtx->kv, vKey, vIsArray, vN);
        break;
      case GGUF_TYPE_INT8:
        vOk = vOk && fGgufReadEmplaceHelper<int8_t>(pGr, vCtx->kv, vKey, vIsArray, vN);
        break;
      case GGUF_TYPE_UINT16:
        vOk = vOk && fGgufReadEmplaceHelper<uint16_t>(pGr, vCtx->kv, vKey, vIsArray, vN);
        break;
      case GGUF_TYPE_INT16:
        vOk = vOk && fGgufReadEmplaceHelper<int16_t>(pGr, vCtx->kv, vKey, vIsArray, vN);
        break;
      case GGUF_TYPE_UINT32:
        vOk = vOk && fGgufReadEmplaceHelper<uint32_t>(pGr, vCtx->kv, vKey, vIsArray, vN);
        break;
      case GGUF_TYPE_INT32:
        vOk = vOk && fGgufReadEmplaceHelper<int32_t>(pGr, vCtx->kv, vKey, vIsArray, vN);
        break;
      case GGUF_TYPE_FLOAT32:
        vOk = vOk && fGgufReadEmplaceHelper<float>(pGr, vCtx->kv, vKey, vIsArray, vN);
        break;
      case GGUF_TYPE_BOOL:
        vOk = vOk && fGgufReadEmplaceHelper<bool>(pGr, vCtx->kv, vKey, vIsArray, vN);
        break;
      case GGUF_TYPE_STRING:
        vOk = vOk && fGgufReadEmplaceHelper<std::string>(pGr, vCtx->kv, vKey, vIsArray, vN);
        break;
      case GGUF_TYPE_UINT64:
        vOk = vOk && fGgufReadEmplaceHelper<uint64_t>(pGr, vCtx->kv, vKey, vIsArray, vN);
        break;
      case GGUF_TYPE_INT64:
        vOk = vOk && fGgufReadEmplaceHelper<int64_t>(pGr, vCtx->kv, vKey, vIsArray, vN);
        break;
      case GGUF_TYPE_FLOAT64:
        vOk = vOk && fGgufReadEmplaceHelper<double>(pGr, vCtx->kv, vKey, vIsArray, vN);
        break;
      case GGUF_TYPE_ARRAY:
      default: {
        GGML_LOG_ERROR("%s: key '%s' has invalid GGUF type %d\n", __func__, vKey.c_str(), vType);
        vOk = false;
      } break;
      }
    }

    if (!vOk) {
      GGML_LOG_ERROR("%s: failed to read key-value pairs\n", __func__);
      gguf_free(vCtx);
      return nullptr;
    }
    GGML_ASSERT(int64_t(vCtx->kv.size()) == vNKv);

    const int cAlignmentIdx = gguf_find_key(vCtx, GGUF_KEY_GENERAL_ALIGNMENT);
    vCtx->vAlignment = cAlignmentIdx == -1 ? GGUF_DEFAULT_ALIGNMENT : gguf_get_val_u32(vCtx, cAlignmentIdx);

    if (vCtx->vAlignment == 0 || (vCtx->vAlignment & (vCtx->vAlignment - 1)) != 0) {
      GGML_LOG_ERROR("%s: alignment %zu is not a power of 2\n", __func__, vCtx->vAlignment);
      gguf_free(vCtx);
      return nullptr;
    }
  }

  // read the tensor info
  for (int64_t vI = 0; vOk && vI < vNTensors; ++vI) {
    struct gguf_tensor_info vInfo;

    // tensor name
    {
      std::string vName;
      try {
        vOk = vOk && pGr.read(vName);
      } catch (std::length_error &) {
        GGML_LOG_ERROR("%s: encountered length_error while reading tensor name %" PRIi64 "\n", __func__, vI);
        vOk = false;
      } catch (std::bad_alloc &) {
        GGML_LOG_ERROR("%s: encountered bad_alloc error while reading tensor name %" PRIi64 "\n", __func__, vI);
        vOk = false;
      }
      if (vName.length() >= GGML_MAX_NAME) {
        GGML_LOG_ERROR("%s: tensor name %" PRIi64 " is too long: %zu >= %d\n", __func__, vI, vName.length(),
                       GGML_MAX_NAME);
        vOk = false;
        break;
      }
      ggml_set_name(&vInfo.vT, vName.c_str());

      // make sure there are no duplicate tensor names
      for (int64_t vJ = 0; vOk && vJ < vI; ++vJ) {
        if (strcmp(vInfo.vT.name, vCtx->vInfo[vJ].vT.name) == 0) {
          GGML_LOG_ERROR("%s: duplicate tensor name '%s' for tensors %" PRIi64 " and %" PRIi64 "\n", __func__,
                         vInfo.vT.name, vJ, vI);
          vOk = false;
          break;
        }
      }
    }
    if (!vOk) {
      break;
    }

    // tensor shape
    {
      uint32_t vNDims = 0;
      vOk = vOk && pGr.read(vNDims);
      if (vNDims > GGML_MAX_DIMS) {
        GGML_LOG_ERROR("%s: tensor '%s' has invalid number of dimensions: %" PRIu32 " > %" PRIu32 "\n", __func__,
                       vInfo.vT.name, vNDims, GGML_MAX_DIMS);
        vOk = false;
        break;
      }
      for (uint32_t vJ = 0; vOk && vJ < GGML_MAX_DIMS; ++vJ) {
        vInfo.vT.ne[vJ] = 1;
        if (vJ < vNDims) {
          vOk = vOk && pGr.read(vInfo.vT.ne[vJ]);
        }

        // check that all ne are non-negative
        if (vInfo.vT.ne[vJ] < 0) {
          GGML_LOG_ERROR("%s: tensor '%s' dimension %" PRIu32 " has invalid number of elements: %" PRIi64 " < 0\n",
                         __func__, vInfo.vT.name, vJ, vInfo.vT.ne[vJ]);
          vOk = false;
          break;
        }
      }

      // check that the total number of elements is representable
      // a zero dimension is valid (an empty tensor), so it is skipped instead of being used as a divisor;
      // the product of the remaining dimensions is still checked (with no zero dimension this is the same
      // check as multiplying them all)
      if (vOk) {
        int64_t vNeNonzero = vInfo.vT.ne[0] != 0 ? vInfo.vT.ne[0] : 1;
        for (uint32_t vJ = 1; vJ < GGML_MAX_DIMS; ++vJ) {
          if (vInfo.vT.ne[vJ] == 0) {
            continue;
          }
          if (INT64_MAX / vInfo.vT.ne[vJ] <= vNeNonzero) {
            vOk = false;
            break;
          }
          vNeNonzero *= vInfo.vT.ne[vJ];
        }
        if (!vOk) {
          GGML_LOG_ERROR("%s: total number of elements in tensor '%s' with shape "
                         "(%" PRIi64 ", %" PRIi64 ", %" PRIi64 ", %" PRIi64 ") is >= %" PRIi64 "\n",
                         __func__, vInfo.vT.name, vInfo.vT.ne[0], vInfo.vT.ne[1], vInfo.vT.ne[2], vInfo.vT.ne[3],
                         INT64_MAX);
          break;
        }
      }
    }
    if (!vOk) {
      break;
    }

    // tensor type
    {
      vOk = vOk && pGr.read(vInfo.vT.type);

      // check that tensor type is within defined range
      if (vInfo.vT.type < 0 || vInfo.vT.type >= GGML_TYPE_COUNT) {
        GGML_LOG_ERROR("%s: tensor '%s' has invalid ggml type %d. should be in [0, %d)\n", __func__, vInfo.vT.name,
                       vInfo.vT.type, GGML_TYPE_COUNT);
        vOk = false;
        break;
      }
      const size_t cTypeSize = ggml_type_size(vInfo.vT.type);
      const int64_t cBlckSize = ggml_blck_size(vInfo.vT.type);

      // check that row size is divisible by block size
      if (cBlckSize == 0 || vInfo.vT.ne[0] % cBlckSize != 0) {
        GGML_LOG_ERROR("%s: tensor '%s' of type %d (%s) has %" PRId64 " elements per row, "
                       "not a multiple of block size (%" PRId64 ")\n",
                       __func__, vInfo.vT.name, (int)vInfo.vT.type, ggml_type_name(vInfo.vT.type), vInfo.vT.ne[0],
                       cBlckSize);
        vOk = false;
        break;
      }

      // check that the size of the tensor in bytes is representable
      // zero dimensions are left out of the product: an empty tensor has no bytes, but the strides computed
      // below still multiply its other dimensions (the product was checked above, so it cannot overflow)
      int64_t vNeNonzero = 1;
      for (int vJ = 0; vJ < GGML_MAX_DIMS; ++vJ) {
        if (vInfo.vT.ne[vJ] != 0) {
          vNeNonzero *= vInfo.vT.ne[vJ];
        }
      }
      if (vOk && uint64_t(vNeNonzero / ggml_blck_size(vInfo.vT.type)) > SIZE_MAX / ggml_type_size(vInfo.vT.type)) {
        GGML_LOG_ERROR("%s: tensor '%s' with shape (%" PRIi64 ", %" PRIi64 ", %" PRIi64 ", %" PRIi64
                       ") has a size in bytes > %zu\n",
                       __func__, vInfo.vT.name, vInfo.vT.ne[0], vInfo.vT.ne[1], vInfo.vT.ne[2], vInfo.vT.ne[3],
                       SIZE_MAX);
        vOk = false;
        break;
      }

      // calculate byte offsets given the tensor shape and type
      vInfo.vT.nb[0] = cTypeSize;
      vInfo.vT.nb[1] = vInfo.vT.nb[0] * (vInfo.vT.ne[0] / cBlckSize);
      for (int vJ = 2; vJ < GGML_MAX_DIMS; ++vJ) {
        vInfo.vT.nb[vJ] = vInfo.vT.nb[vJ - 1] * vInfo.vT.ne[vJ - 1];
      }
    }
    if (!vOk) {
      break;
    }

    // tensor data offset within buffer
    vOk = vOk && pGr.read(vInfo.vOffset);

    vCtx->vInfo.push_back(vInfo);
  }

  if (!vOk) {
    GGML_LOG_ERROR("%s: failed to read tensor info\n", __func__);
    gguf_free(vCtx);
    return nullptr;
  }
  GGML_ASSERT(int64_t(vCtx->vInfo.size()) == vNTensors);

  // we require the data section to be aligned, so take into account any padding
  if (vNTensors > 0 && !pGr.mSeek(GGML_PAD(pGr.mTell(), vCtx->vAlignment))) {
    GGML_LOG_ERROR("%s: failed to seek to beginning of data section\n", __func__);
    gguf_free(vCtx);
    return nullptr;
  }

  // store the current file offset - this is where the data section starts
  vCtx->vOffset = pGr.mTell();

  // compute the total size of the data section, taking into account the alignment
  {
    vCtx->size = 0;
    for (size_t vI = 0; vI < vCtx->vInfo.size(); ++vI) {
      const gguf_tensor_info &vTi = vCtx->vInfo[vI];
      if (vTi.vOffset != vCtx->size) {
        GGML_LOG_ERROR("%s: tensor '%s' has offset %" PRIu64 ", expected %zu\n", __func__, vTi.vT.name, vTi.vOffset,
                       vCtx->size);
        GGML_LOG_ERROR("%s: failed to read tensor data\n", __func__);
        gguf_free(vCtx);
        return nullptr;
      }
      size_t vPaddedSize = GGML_PAD(ggml_nbytes(&vTi.vT), vCtx->vAlignment);
      if (SIZE_MAX - vCtx->size < vPaddedSize) {
        GGML_LOG_ERROR("%s: tensor '%s' size overflow, cannot accumulate size %zu + %zu\n", __func__, vTi.vT.name,
                       vCtx->size, vPaddedSize);
        gguf_free(vCtx);
        return nullptr;
      }
      vCtx->size += vPaddedSize;
    }
  }

  // load the tensor data only if requested
  if (pParams.ctx != nullptr) {
    // if the provided gguf_context is no_alloc, then we create "empty" tensors and do not read the binary blob
    // otherwise, we load the binary blob into the created ggml_context as well, and point the "data" members of
    //   the ggml_tensor structs to the appropriate locations in the binary blob

    // compute the exact size needed for the new ggml_context
    size_t vMemSize = 0;
    if (pParams.no_alloc) {
      if (vNTensors != 0 && SIZE_MAX / vNTensors < ggml_tensor_overhead()) {
        GGML_LOG_ERROR("%s: memory size overflow while allocating ggml context\n", __func__);
        gguf_free(vCtx);
        return nullptr;
      }

      const size_t cOverhead = vNTensors * ggml_tensor_overhead();

      vMemSize = cOverhead;
    } else {
      if ((vNTensors + 1) != 0 && SIZE_MAX / (vNTensors + 1) < ggml_tensor_overhead()) {
        GGML_LOG_ERROR("%s: memory size overflow while allocating ggml context\n", __func__);
        gguf_free(vCtx);
        return nullptr;
      }

      const size_t cOverhead = (vNTensors + 1) * ggml_tensor_overhead();

      if (SIZE_MAX - cOverhead < vCtx->size) {
        GGML_LOG_ERROR("%s: memory size overflow while allocating ggml context\n", __func__);
        gguf_free(vCtx);
        return nullptr;
      }

      vMemSize = cOverhead + vCtx->size;
    }

    struct ggml_init_params vPdata = {
        /*mem_size   =*/vMemSize,
        /*mem_buffer =*/nullptr,
        /*no_alloc   =*/pParams.no_alloc,
    };

    *pParams.ctx = ggml_init(vPdata);
    if (*pParams.ctx == nullptr) {
      GGML_LOG_ERROR("%s: failed to initialize ggml context for storing tensors\n", __func__);
      gguf_free(vCtx);
      return nullptr;
    }

    struct ggml_context *vCtxData = *pParams.ctx;

    struct ggml_tensor *vData = nullptr;

    if (!pParams.no_alloc) {
      vData = ggml_new_tensor_1d(vCtxData, GGML_TYPE_I8, vCtx->size);

      vOk = vOk && vData != nullptr;

      if (vOk) {
        ggml_set_name(vData, "GGUF tensor data binary blob");
      }

      // read the binary blob with the tensor data
      vOk = vOk && pGr.read(vData->data, vCtx->size);

      if (!vOk) {
        GGML_LOG_ERROR("%s: failed to read tensor data binary blob\n", __func__);
        ggml_free(vCtxData);
        *pParams.ctx = nullptr;
        gguf_free(vCtx);
        return nullptr;
      }

      vCtx->data = vData->data;
    }

    ggml_set_no_alloc(vCtxData, true);

    // create the tensors
    for (size_t vI = 0; vI < vCtx->vInfo.size(); ++vI) {
      const struct gguf_tensor_info &vInfo = vCtx->vInfo[vI];

      struct ggml_tensor *vCur = ggml_new_tensor(vCtxData, vInfo.vT.type, GGML_MAX_DIMS, vInfo.vT.ne);

      vOk = vOk && vCur != nullptr;

      if (!vOk) {
        break;
      }

      ggml_set_name(vCur, vInfo.vT.name);

      // point the data member to the appropriate location in the binary blob using the tensor info
      if (!pParams.no_alloc) {
        vCur->data = (char *)vData->data + vInfo.vOffset;
      }
    }

    if (!vOk) {
      GGML_LOG_ERROR("%s: failed to create tensors\n", __func__);
      ggml_free(vCtxData);
      *pParams.ctx = nullptr;
      gguf_free(vCtx);
      return nullptr;
    }

    ggml_set_no_alloc(vCtxData, pParams.no_alloc);
  }

  return vCtx;
}

struct gguf_context *gguf_init_from_callback(gguf_reader_callback_t pCallback, void *pUserdata, size_t pMaxChunkRead,
                                             uint64_t pMaxExpectedSize, struct gguf_init_params pParams) {
  if (pCallback == nullptr) {
    return nullptr;
  }

  const struct gguf_reader cGr(pCallback, pUserdata, pMaxChunkRead == 0 ? SIZE_MAX : pMaxChunkRead, 0,
                               pMaxExpectedSize);
  return fGgufInitFromReader(cGr, pParams);
}

struct gguf_file_reader {
  FILE *vFile;
  uint64_t vOffset;
};

static size_t fGgufFileReaderCallback(void *pUserdata, void *pOutput, uint64_t pOffset, size_t pLen) {
  GGML_ASSERT(pLen > 0);

  gguf_file_reader &vReader = *static_cast<gguf_file_reader *>(pUserdata);

  if (vReader.vOffset != pOffset) {
    if (pOffset > INT64_MAX || gguf_fseek(vReader.vFile, static_cast<int64_t>(pOffset), SEEK_SET) != 0) {
      return 0;
    }

    vReader.vOffset = pOffset;
  }

  const size_t cNread = fread(static_cast<uint8_t *>(pOutput), 1, pLen, vReader.vFile);
  vReader.vOffset += cNread;
  return cNread;
}

struct gguf_context *gguf_init_from_file_ptr(FILE *pFile, struct gguf_init_params pParams) {
  if (!pFile) {
    return nullptr;
  }

  const int64_t cCur = gguf_ftell(pFile);
  if (cCur < 0) {
    return nullptr;
  }

  gguf_file_reader vReader = {
      /*.file   = */ pFile,
      /*.offset = */ static_cast<uint64_t>(cCur),
  };
  const struct gguf_reader cGr(fGgufFileReaderCallback, &vReader, SIZE_MAX, vReader.vOffset,
                               gguf_reader::mFileRemain(pFile));
  return fGgufInitFromReader(cGr, pParams);
}

struct gguf_buffer_reader {
  const uint8_t *data;
  size_t size;
};

static size_t fGgufBufferReaderCallback(void *pUserdata, void *pOutput, uint64_t pOffset, size_t pLen) {
  GGML_ASSERT(pLen > 0);

  const gguf_buffer_reader &vReader = *static_cast<gguf_buffer_reader *>(pUserdata);

  if (pOffset > vReader.size || pLen > vReader.size - pOffset) {
    return 0;
  }

  const size_t cDataOffset = static_cast<size_t>(pOffset);
  const size_t cNread = std::min(pLen, vReader.size - cDataOffset);
  memcpy(static_cast<uint8_t *>(pOutput), vReader.data + cDataOffset, cNread);
  return cNread;
}

struct gguf_context *gguf_init_from_buffer(const void *pData, size_t pSize, struct gguf_init_params pParams) {
  if (pData == nullptr || pSize == 0) {
    return nullptr;
  }

  gguf_buffer_reader vReader = {
      /*.data = */ static_cast<const uint8_t *>(pData),
      /*.size = */ pSize,
  };
  const struct gguf_reader cGr(fGgufBufferReaderCallback, &vReader, SIZE_MAX, 0, pSize);
  return fGgufInitFromReader(cGr, pParams);
}

struct gguf_context *gguf_init_from_file(const char *pFname, struct gguf_init_params pParams) {
  FILE *vFile = ggml_fopen(pFname, "rb");

  if (!vFile) {
    GGML_LOG_ERROR("%s: failed to open GGUF file '%s' (%s)\n", __func__, pFname, strerror(errno));
    return nullptr;
  }

  struct gguf_context *vResult = gguf_init_from_file_ptr(vFile, pParams);
  fclose(vFile);
  return vResult;
}

void gguf_free(struct gguf_context *pCtx) {
  if (pCtx == nullptr) {
    return;
  }
  delete pCtx;
}

const char *gguf_type_name(enum gguf_type pType) {
  auto vIt = GGUF_TYPE_NAME.find(pType);
  return vIt == GGUF_TYPE_NAME.end() ? nullptr : vIt->second;
}

uint32_t gguf_get_version(const struct gguf_context *pCtx) { return pCtx->vVersion; }

size_t gguf_get_alignment(const struct gguf_context *pCtx) { return pCtx->vAlignment; }

size_t gguf_get_data_offset(const struct gguf_context *pCtx) { return pCtx->vOffset; }

int64_t gguf_get_n_kv(const struct gguf_context *pCtx) { return pCtx->kv.size(); }

int64_t gguf_find_key(const struct gguf_context *pCtx, const char *pKey) {
  // return -1 if key not found
  int64_t vKeyfound = -1;

  const int64_t cNKv = gguf_get_n_kv(pCtx);

  for (int64_t vI = 0; vI < cNKv; ++vI) {
    if (strcmp(pKey, gguf_get_key(pCtx, vI)) == 0) {
      vKeyfound = vI;
      break;
    }
  }

  return vKeyfound;
}

const char *gguf_get_key(const struct gguf_context *pCtx, int64_t pKeyId) {
  GGML_ASSERT(pKeyId >= 0 && pKeyId < gguf_get_n_kv(pCtx));
  return pCtx->kv[pKeyId].mGetKey().c_str();
}

enum gguf_type gguf_get_kv_type(const struct gguf_context *pCtx, int64_t pKeyId) {
  GGML_ASSERT(pKeyId >= 0 && pKeyId < gguf_get_n_kv(pCtx));
  return pCtx->kv[pKeyId].vIsArray ? GGUF_TYPE_ARRAY : pCtx->kv[pKeyId].mGetType();
}

enum gguf_type gguf_get_arr_type(const struct gguf_context *pCtx, int64_t pKeyId) {
  GGML_ASSERT(pKeyId >= 0 && pKeyId < gguf_get_n_kv(pCtx));
  GGML_ASSERT(pCtx->kv[pKeyId].vIsArray);
  return pCtx->kv[pKeyId].mGetType();
}

const void *gguf_get_arr_data(const struct gguf_context *pCtx, int64_t pKeyId) {
  GGML_ASSERT(pKeyId >= 0 && pKeyId < gguf_get_n_kv(pCtx));
  GGML_ASSERT(pCtx->kv[pKeyId].mGetType() != GGUF_TYPE_STRING);
  return pCtx->kv[pKeyId].data.data();
}

const char *gguf_get_arr_str(const struct gguf_context *pCtx, int64_t pKeyId, size_t pI) {
  GGML_ASSERT(pKeyId >= 0 && pKeyId < gguf_get_n_kv(pCtx));
  GGML_ASSERT(pCtx->kv[pKeyId].mGetType() == GGUF_TYPE_STRING);
  return pCtx->kv[pKeyId].vDataString[pI].c_str();
}

size_t gguf_get_arr_n(const struct gguf_context *pCtx, int64_t pKeyId) {
  GGML_ASSERT(pKeyId >= 0 && pKeyId < gguf_get_n_kv(pCtx));

  if (pCtx->kv[pKeyId].vType == GGUF_TYPE_STRING) {
    return pCtx->kv[pKeyId].vDataString.size();
  }

  const size_t cTypeSize = gguf_type_size(pCtx->kv[pKeyId].vType);
  GGML_ASSERT(pCtx->kv[pKeyId].data.size() % cTypeSize == 0);
  return pCtx->kv[pKeyId].data.size() / cTypeSize;
}

uint8_t gguf_get_val_u8(const struct gguf_context *pCtx, int64_t pKeyId) {
  GGML_ASSERT(pKeyId >= 0 && pKeyId < gguf_get_n_kv(pCtx));
  GGML_ASSERT(pCtx->kv[pKeyId].mGetNe() == 1);
  return pCtx->kv[pKeyId].mGetVal<uint8_t>();
}

int8_t gguf_get_val_i8(const struct gguf_context *pCtx, int64_t pKeyId) {
  GGML_ASSERT(pKeyId >= 0 && pKeyId < gguf_get_n_kv(pCtx));
  GGML_ASSERT(pCtx->kv[pKeyId].mGetNe() == 1);
  return pCtx->kv[pKeyId].mGetVal<int8_t>();
}

uint16_t gguf_get_val_u16(const struct gguf_context *pCtx, int64_t pKeyId) {
  GGML_ASSERT(pKeyId >= 0 && pKeyId < gguf_get_n_kv(pCtx));
  GGML_ASSERT(pCtx->kv[pKeyId].mGetNe() == 1);
  return pCtx->kv[pKeyId].mGetVal<uint16_t>();
}

int16_t gguf_get_val_i16(const struct gguf_context *pCtx, int64_t pKeyId) {
  GGML_ASSERT(pKeyId >= 0 && pKeyId < gguf_get_n_kv(pCtx));
  GGML_ASSERT(pCtx->kv[pKeyId].mGetNe() == 1);
  return pCtx->kv[pKeyId].mGetVal<int16_t>();
}

uint32_t gguf_get_val_u32(const struct gguf_context *pCtx, int64_t pKeyId) {
  GGML_ASSERT(pKeyId >= 0 && pKeyId < gguf_get_n_kv(pCtx));
  GGML_ASSERT(pCtx->kv[pKeyId].mGetNe() == 1);
  return pCtx->kv[pKeyId].mGetVal<uint32_t>();
}

int32_t gguf_get_val_i32(const struct gguf_context *pCtx, int64_t pKeyId) {
  GGML_ASSERT(pKeyId >= 0 && pKeyId < gguf_get_n_kv(pCtx));
  GGML_ASSERT(pCtx->kv[pKeyId].mGetNe() == 1);
  return pCtx->kv[pKeyId].mGetVal<int32_t>();
}

float gguf_get_val_f32(const struct gguf_context *pCtx, int64_t pKeyId) {
  GGML_ASSERT(pKeyId >= 0 && pKeyId < gguf_get_n_kv(pCtx));
  GGML_ASSERT(pCtx->kv[pKeyId].mGetNe() == 1);
  return pCtx->kv[pKeyId].mGetVal<float>();
}

uint64_t gguf_get_val_u64(const struct gguf_context *pCtx, int64_t pKeyId) {
  GGML_ASSERT(pKeyId >= 0 && pKeyId < gguf_get_n_kv(pCtx));
  GGML_ASSERT(pCtx->kv[pKeyId].mGetNe() == 1);
  return pCtx->kv[pKeyId].mGetVal<uint64_t>();
}

int64_t gguf_get_val_i64(const struct gguf_context *pCtx, int64_t pKeyId) {
  GGML_ASSERT(pKeyId >= 0 && pKeyId < gguf_get_n_kv(pCtx));
  GGML_ASSERT(pCtx->kv[pKeyId].mGetNe() == 1);
  return pCtx->kv[pKeyId].mGetVal<int64_t>();
}

double gguf_get_val_f64(const struct gguf_context *pCtx, int64_t pKeyId) {
  GGML_ASSERT(pKeyId >= 0 && pKeyId < gguf_get_n_kv(pCtx));
  GGML_ASSERT(pCtx->kv[pKeyId].mGetNe() == 1);
  return pCtx->kv[pKeyId].mGetVal<double>();
}

bool gguf_get_val_bool(const struct gguf_context *pCtx, int64_t pKeyId) {
  GGML_ASSERT(pKeyId >= 0 && pKeyId < gguf_get_n_kv(pCtx));
  GGML_ASSERT(pCtx->kv[pKeyId].mGetNe() == 1);
  return pCtx->kv[pKeyId].mGetVal<bool>();
}

const char *gguf_get_val_str(const struct gguf_context *pCtx, int64_t pKeyId) {
  GGML_ASSERT(pKeyId >= 0 && pKeyId < gguf_get_n_kv(pCtx));
  GGML_ASSERT(pCtx->kv[pKeyId].mGetNe() == 1);
  return pCtx->kv[pKeyId].mGetVal<std::string>().c_str();
}

const void *gguf_get_val_data(const struct gguf_context *pCtx, int64_t pKeyId) {
  GGML_ASSERT(pKeyId >= 0 && pKeyId < gguf_get_n_kv(pCtx));
  GGML_ASSERT(pCtx->kv[pKeyId].mGetNe() == 1);
  GGML_ASSERT(pCtx->kv[pKeyId].mGetType() != GGUF_TYPE_STRING);
  return pCtx->kv[pKeyId].data.data();
}

int64_t gguf_get_n_tensors(const struct gguf_context *pCtx) { return pCtx->vInfo.size(); }

int64_t gguf_find_tensor(const struct gguf_context *pCtx, const char *pName) {
  // return -1 if tensor not found
  int64_t vTensorId = -1;

  const int64_t cNTensors = gguf_get_n_tensors(pCtx);

  for (int64_t vI = 0; vI < cNTensors; ++vI) {
    if (strcmp(pName, gguf_get_tensor_name(pCtx, vI)) == 0) {
      vTensorId = vI;
      break;
    }
  }

  return vTensorId;
}

size_t gguf_get_tensor_offset(const struct gguf_context *pCtx, int64_t pTensorId) {
  GGML_ASSERT(pTensorId >= 0 && pTensorId < gguf_get_n_tensors(pCtx));
  return pCtx->vInfo[pTensorId].vOffset;
}

const char *gguf_get_tensor_name(const struct gguf_context *pCtx, int64_t pTensorId) {
  GGML_ASSERT(pTensorId >= 0 && pTensorId < gguf_get_n_tensors(pCtx));
  return pCtx->vInfo[pTensorId].vT.name;
}

const int64_t *gguf_get_tensor_ne(const struct gguf_context *pCtx, int64_t pTensorId) {
  GGML_ASSERT(pTensorId >= 0 && pTensorId < gguf_get_n_tensors(pCtx));
  return pCtx->vInfo[pTensorId].vT.ne;
}

enum ggml_type gguf_get_tensor_type(const struct gguf_context *pCtx, int64_t pTensorId) {
  GGML_ASSERT(pTensorId >= 0 && pTensorId < gguf_get_n_tensors(pCtx));
  return pCtx->vInfo[pTensorId].vT.type;
}

size_t gguf_get_tensor_size(const struct gguf_context *pCtx, int64_t pTensorId) {
  GGML_ASSERT(pTensorId >= 0 && pTensorId < gguf_get_n_tensors(pCtx));
  return ggml_nbytes(&pCtx->vInfo[pTensorId].vT);
}

int64_t gguf_remove_key(struct gguf_context *pCtx, const char *pKey) {
  const int64_t cKeyId = gguf_find_key(pCtx, pKey);
  if (cKeyId >= 0) {
    pCtx->kv.erase(pCtx->kv.begin() + cKeyId);
  }
  return cKeyId;
}

template <typename T> static void fGgufCheckReservedKeys(const std::string &pKey, const T pVal) {
  if (pKey == GGUF_KEY_GENERAL_ALIGNMENT) {
    if constexpr (std::is_same<T, uint32_t>::value) {
      GGML_ASSERT(pVal > 0 && (pVal & (pVal - 1)) == 0 && GGUF_KEY_GENERAL_ALIGNMENT " must be power of 2");
    } else {
      GGML_UNUSED(pVal);
      GGML_ABORT(GGUF_KEY_GENERAL_ALIGNMENT " must be type u32");
    }
  }
}

void gguf_set_val_u8(struct gguf_context *pCtx, const char *pKey, uint8_t pVal) {
  fGgufCheckReservedKeys(pKey, pVal);
  gguf_remove_key(pCtx, pKey);
  pCtx->kv.emplace_back(pKey, pVal);
}

void gguf_set_val_i8(struct gguf_context *pCtx, const char *pKey, int8_t pVal) {
  fGgufCheckReservedKeys(pKey, pVal);
  gguf_remove_key(pCtx, pKey);
  pCtx->kv.emplace_back(pKey, pVal);
}

void gguf_set_val_u16(struct gguf_context *pCtx, const char *pKey, uint16_t pVal) {
  fGgufCheckReservedKeys(pKey, pVal);
  gguf_remove_key(pCtx, pKey);
  pCtx->kv.emplace_back(pKey, pVal);
}

void gguf_set_val_i16(struct gguf_context *pCtx, const char *pKey, int16_t pVal) {
  fGgufCheckReservedKeys(pKey, pVal);
  gguf_remove_key(pCtx, pKey);
  pCtx->kv.emplace_back(pKey, pVal);
}

void gguf_set_val_u32(struct gguf_context *pCtx, const char *pKey, uint32_t pVal) {
  fGgufCheckReservedKeys(pKey, pVal);
  gguf_remove_key(pCtx, pKey);
  pCtx->kv.emplace_back(pKey, pVal);
}

void gguf_set_val_i32(struct gguf_context *pCtx, const char *pKey, int32_t pVal) {
  fGgufCheckReservedKeys(pKey, pVal);
  gguf_remove_key(pCtx, pKey);
  pCtx->kv.emplace_back(pKey, pVal);
}

void gguf_set_val_f32(struct gguf_context *pCtx, const char *pKey, float pVal) {
  fGgufCheckReservedKeys(pKey, pVal);
  gguf_remove_key(pCtx, pKey);
  pCtx->kv.emplace_back(pKey, pVal);
}

void gguf_set_val_u64(struct gguf_context *pCtx, const char *pKey, uint64_t pVal) {
  fGgufCheckReservedKeys(pKey, pVal);
  gguf_remove_key(pCtx, pKey);
  pCtx->kv.emplace_back(pKey, pVal);
}

void gguf_set_val_i64(struct gguf_context *pCtx, const char *pKey, int64_t pVal) {
  fGgufCheckReservedKeys(pKey, pVal);
  gguf_remove_key(pCtx, pKey);
  pCtx->kv.emplace_back(pKey, pVal);
}

void gguf_set_val_f64(struct gguf_context *pCtx, const char *pKey, double pVal) {
  fGgufCheckReservedKeys(pKey, pVal);
  gguf_remove_key(pCtx, pKey);
  pCtx->kv.emplace_back(pKey, pVal);
}

void gguf_set_val_bool(struct gguf_context *pCtx, const char *pKey, bool pVal) {
  fGgufCheckReservedKeys(pKey, pVal);
  gguf_remove_key(pCtx, pKey);
  pCtx->kv.emplace_back(pKey, pVal);
}

void gguf_set_val_str(struct gguf_context *pCtx, const char *pKey, const char *pVal) {
  fGgufCheckReservedKeys(pKey, pVal);
  gguf_remove_key(pCtx, pKey);
  pCtx->kv.emplace_back(pKey, std::string(pVal));
}

void gguf_set_arr_data(struct gguf_context *pCtx, const char *pKey, enum gguf_type pType, const void *pData,
                       size_t pN) {
  fGgufCheckReservedKeys(pKey, pData);
  gguf_remove_key(pCtx, pKey);

  const size_t cNbytes = pN * gguf_type_size(pType);
  std::vector<int8_t> lTmp(cNbytes);
  if (!lTmp.empty()) {
    memcpy(lTmp.data(), pData, cNbytes);
  }
  pCtx->kv.emplace_back(pKey, lTmp);
  pCtx->kv.back().mCast(pType);
}

void gguf_set_arr_str(struct gguf_context *pCtx, const char *pKey, const char **pData, size_t pN) {
  fGgufCheckReservedKeys(pKey, pData);
  gguf_remove_key(pCtx, pKey);

  std::vector<std::string> lTmp(pN);
  for (size_t vI = 0; vI < pN; ++vI) {
    lTmp[vI] = pData[vI];
  }
  pCtx->kv.emplace_back(pKey, lTmp);
}

// set or add KV pairs from another context
void gguf_set_kv(struct gguf_context *pCtx, const struct gguf_context *pSrc) {
  const int64_t cNKv = gguf_get_n_kv(pSrc);
  for (int64_t vI = 0; vI < cNKv; ++vI) {
    const struct gguf_kv &vKv = pSrc->kv[vI];

    if (!vKv.vIsArray) {
      switch (vKv.mGetType()) {
      case GGUF_TYPE_UINT8:
        gguf_set_val_u8(pCtx, vKv.mGetKey().c_str(), vKv.mGetVal<uint8_t>());
        break;
      case GGUF_TYPE_INT8:
        gguf_set_val_i8(pCtx, vKv.mGetKey().c_str(), vKv.mGetVal<int8_t>());
        break;
      case GGUF_TYPE_UINT16:
        gguf_set_val_u16(pCtx, vKv.mGetKey().c_str(), vKv.mGetVal<uint16_t>());
        break;
      case GGUF_TYPE_INT16:
        gguf_set_val_i16(pCtx, vKv.mGetKey().c_str(), vKv.mGetVal<int16_t>());
        break;
      case GGUF_TYPE_UINT32:
        gguf_set_val_u32(pCtx, vKv.mGetKey().c_str(), vKv.mGetVal<uint32_t>());
        break;
      case GGUF_TYPE_INT32:
        gguf_set_val_i32(pCtx, vKv.mGetKey().c_str(), vKv.mGetVal<int32_t>());
        break;
      case GGUF_TYPE_FLOAT32:
        gguf_set_val_f32(pCtx, vKv.mGetKey().c_str(), vKv.mGetVal<float>());
        break;
      case GGUF_TYPE_UINT64:
        gguf_set_val_u64(pCtx, vKv.mGetKey().c_str(), vKv.mGetVal<uint64_t>());
        break;
      case GGUF_TYPE_INT64:
        gguf_set_val_i64(pCtx, vKv.mGetKey().c_str(), vKv.mGetVal<int64_t>());
        break;
      case GGUF_TYPE_FLOAT64:
        gguf_set_val_f64(pCtx, vKv.mGetKey().c_str(), vKv.mGetVal<double>());
        break;
      case GGUF_TYPE_BOOL:
        gguf_set_val_bool(pCtx, vKv.mGetKey().c_str(), vKv.mGetVal<bool>());
        break;
      case GGUF_TYPE_STRING:
        gguf_set_val_str(pCtx, vKv.mGetKey().c_str(), vKv.mGetVal<std::string>().c_str());
        break;
      case GGUF_TYPE_ARRAY:
      default:
        GGML_ABORT("invalid type");
      }
      continue;
    }

    const size_t cNe = vKv.mGetNe();

    switch (vKv.mGetType()) {
    case GGUF_TYPE_UINT8:
    case GGUF_TYPE_INT8:
    case GGUF_TYPE_UINT16:
    case GGUF_TYPE_INT16:
    case GGUF_TYPE_UINT32:
    case GGUF_TYPE_INT32:
    case GGUF_TYPE_FLOAT32:
    case GGUF_TYPE_UINT64:
    case GGUF_TYPE_INT64:
    case GGUF_TYPE_FLOAT64:
    case GGUF_TYPE_BOOL: {
      gguf_set_arr_data(pCtx, vKv.mGetKey().c_str(), vKv.mGetType(), vKv.data.data(), cNe);
    } break;
    case GGUF_TYPE_STRING: {
      std::vector<const char *> lTmp(cNe);
      for (size_t vJ = 0; vJ < cNe; ++vJ) {
        lTmp[vJ] = vKv.vDataString[vJ].c_str();
      }
      gguf_set_arr_str(pCtx, vKv.mGetKey().c_str(), lTmp.data(), cNe);
    } break;
    case GGUF_TYPE_ARRAY:
    default:
      GGML_ABORT("invalid type");
    }
  }
}

void gguf_add_tensor(struct gguf_context *pCtx, const struct ggml_tensor *pTensor) {
  GGML_ASSERT(pTensor);
  if (gguf_find_tensor(pCtx, pTensor->name) != -1) {
    GGML_ABORT("duplicate tensor name: %s", pTensor->name);
  }

  struct gguf_tensor_info vTi;
  vTi.vT = *pTensor;
  vTi.vOffset = pCtx->vInfo.empty()
                    ? 0
                    : pCtx->vInfo.back().vOffset + GGML_PAD(ggml_nbytes(&pCtx->vInfo.back().vT), pCtx->vAlignment);
  pCtx->vInfo.push_back(vTi);
}

void gguf_set_tensor_type(struct gguf_context *pCtx, const char *pName, enum ggml_type pType) {
  const int64_t cTensorId = gguf_find_tensor(pCtx, pName);
  if (cTensorId < 0) {
    GGML_ABORT("tensor not found: %s", pName);
  }
  struct ggml_tensor *vTensor = &pCtx->vInfo[cTensorId].vT;
  const size_t cTypeSize = ggml_type_size(pType);
  const int64_t cBlckSize = ggml_blck_size(pType);

  vTensor->type = pType;
  GGML_ASSERT(vTensor->ne[0] % cBlckSize == 0 && "tensor row size not divisible by block size of new type");

  vTensor->nb[0] = cTypeSize;
  vTensor->nb[1] = vTensor->nb[0] * (vTensor->ne[0] / cBlckSize);
  for (int vI = 2; vI < GGML_MAX_DIMS; vI++) {
    vTensor->nb[vI] = vTensor->nb[vI - 1] * vTensor->ne[vI - 1];
  }

  // update offsets
  const int64_t cNTensors = gguf_get_n_tensors(pCtx);
  for (int64_t vI = cTensorId + 1; vI < cNTensors; ++vI) {
    pCtx->vInfo[vI].vOffset =
        pCtx->vInfo[vI - 1].vOffset + GGML_PAD(ggml_nbytes(&pCtx->vInfo[vI - 1].vT), pCtx->vAlignment);
  }
}

void gguf_set_tensor_data(struct gguf_context *pCtx, const char *pName, const void *pData) {
  const int64_t cTensorId = gguf_find_tensor(pCtx, pName);
  if (cTensorId < 0) {
    GGML_ABORT("tensor not found: %s", pName);
  }

  pCtx->vInfo[cTensorId].vT.data = (void *)(uintptr_t)pData; // double cast suppresses warning about casting away const
}

struct gguf_writer_base {
  size_t vWrittenBytes{0u};

  virtual ~gguf_writer_base(void) = default;

  // we bet on devirtualization
  virtual void write(int8_t pVal) = 0;
  virtual void write(const std::vector<int8_t> &pVal) = 0;
  virtual void mWriteTensorData(const struct gguf_tensor_info &pInfo, size_t pOffsetData, size_t pAlignment) = 0;

  template <typename T> void write(const T &pVal) {
    for (size_t vI = 0; vI < sizeof(pVal); ++vI) {
      write(reinterpret_cast<const int8_t *>(&pVal)[vI]);
    }
  }

  void write(const bool &pVal) {
    const int8_t cVal8 = pVal ? 1 : 0;
    write(cVal8);
  }

  void write(const std::string &pVal) {
    {
      const uint64_t cN = pVal.length();
      write(cN);
    }
    for (size_t vI = 0; vI < pVal.length(); ++vI) {
      write((pVal.data())[vI]);
    }
  }

  void write(const char *pVal) { write(std::string(pVal)); }

  void write(const enum ggml_type &pVal) { write(int32_t(pVal)); }

  void write(const enum gguf_type &pVal) { write(int32_t(pVal)); }

  void write(const struct gguf_kv &pKv) {
    const uint64_t cNe = pKv.mGetNe();

    write(pKv.mGetKey());

    if (pKv.vIsArray) {
      write(GGUF_TYPE_ARRAY);
      write(pKv.mGetType());
      write(cNe);
    } else {
      write(pKv.mGetType());
    }

    switch (pKv.mGetType()) {
    case GGUF_TYPE_UINT8:
    case GGUF_TYPE_INT8:
    case GGUF_TYPE_UINT16:
    case GGUF_TYPE_INT16:
    case GGUF_TYPE_UINT32:
    case GGUF_TYPE_INT32:
    case GGUF_TYPE_FLOAT32:
    case GGUF_TYPE_UINT64:
    case GGUF_TYPE_INT64:
    case GGUF_TYPE_FLOAT64: {
      write(pKv.data);
    } break;
    case GGUF_TYPE_BOOL: {
      for (size_t vI = 0; vI < cNe; ++vI) {
        write(pKv.mGetVal<bool>(vI));
      }
    } break;
    case GGUF_TYPE_STRING: {
      for (size_t vI = 0; vI < cNe; ++vI) {
        write(pKv.mGetVal<std::string>(vI));
      }
    } break;
    case GGUF_TYPE_ARRAY:
    default:
      GGML_ABORT("invalid type");
    }
  }

  void mWriteTensorMeta(const struct gguf_tensor_info &pInfo) {
    write(pInfo.vT.name);

    const uint32_t cNDims = ggml_n_dims(&pInfo.vT);
    write(cNDims);

    for (uint32_t vJ = 0; vJ < cNDims; ++vJ) {
      write(pInfo.vT.ne[vJ]);
    }
    write(pInfo.vT.type);
    write(pInfo.vOffset);
  }

  void mPad(const size_t pAlignment) {
    while (vWrittenBytes % pAlignment != 0) {
      const int8_t cZero = 0;
      write(cZero);
    }
  }
};

// vector buffer based writer
struct gguf_writer_buf final : public gguf_writer_base {
  std::vector<int8_t> &vBuf;

  gguf_writer_buf(std::vector<int8_t> &pBuf) : vBuf(pBuf) {}

  using gguf_writer_base::write;

  void write(const int8_t pVal) override {
    vBuf.push_back(pVal);
    vWrittenBytes++;
  }

  void write(const std::vector<int8_t> &pVal) override {
    vBuf.insert(vBuf.end(), pVal.begin(), pVal.end());
    vWrittenBytes += pVal.size();
  }

  void mWriteTensorData(const struct gguf_tensor_info &pInfo, const size_t pOffsetData,
                        const size_t pAlignment) override {
    GGML_ASSERT(vBuf.size() - pOffsetData == pInfo.vOffset);

    GGML_ASSERT(ggml_is_contiguous(&pInfo.vT));
    const size_t cOffset = vBuf.size();
    const size_t cNbytes = ggml_nbytes(&pInfo.vT);

    vBuf.resize(cOffset + cNbytes);
    if (pInfo.vT.buffer) {
      ggml_backend_tensor_get(&pInfo.vT, vBuf.data() + cOffset, 0, cNbytes);
    } else {
      GGML_ASSERT(pInfo.vT.data);
      memcpy(vBuf.data() + cOffset, pInfo.vT.data, cNbytes);
    }
    vWrittenBytes += cNbytes;

    mPad(pAlignment);
  }
};

// file based writer
struct gguf_writer_file final : public gguf_writer_base {
  FILE *vFile;

  gguf_writer_file(FILE *pFile) : vFile(pFile) {}

  using gguf_writer_base::write;

  void write(const int8_t pVal) override {
    const auto cRealVal = static_cast<uint8_t>(pVal);
    const auto cRet = fputc(cRealVal, vFile);
    vWrittenBytes++;
    if (cRet != cRealVal) {
      throw std::runtime_error("unexpected fputc result '" + std::to_string(cRet) + "' instead of '" +
                               std::to_string((int)cRealVal) + "'");
    }
  }

  void write(const std::vector<int8_t> &pVal) override {
    const auto cRet = fwrite(pVal.data(), 1, pVal.size(), vFile);
    vWrittenBytes += pVal.size();
    if (cRet != pVal.size()) {
      throw std::runtime_error("unexpected fwrite number of bytes written, '" + std::to_string(cRet) +
                               "' instead of '" + std::to_string(pVal.size()) + "'");
    }
  }

  void mWriteTensorData(const struct gguf_tensor_info &pInfo, const size_t pOffsetData,
                        const size_t pAlignment) override {
    GGML_ASSERT(vWrittenBytes - pOffsetData == pInfo.vOffset);

    GGML_ASSERT(ggml_is_contiguous(&pInfo.vT));
    const size_t cNbytes = ggml_nbytes(&pInfo.vT);

    std::vector<int8_t> vBuf(cNbytes);
    if (pInfo.vT.buffer) {
      ggml_backend_tensor_get(&pInfo.vT, vBuf.data(), 0, cNbytes);
    } else {
      GGML_ASSERT(pInfo.vT.data);
      memcpy(vBuf.data(), pInfo.vT.data, cNbytes);
    }
    write(vBuf);

    mPad(pAlignment);
  }
};

template <typename writer_t> static void fGgufWriteOut(const struct gguf_context *pCtx, writer_t &pGw, bool pOnlyMeta) {
  const int64_t cNKv = gguf_get_n_kv(pCtx);
  const int64_t cNTensors = gguf_get_n_tensors(pCtx);

  // write header
  pGw.write(GGUF_MAGIC[0]);
  pGw.write(GGUF_MAGIC[1]);
  pGw.write(GGUF_MAGIC[2]);
  pGw.write(GGUF_MAGIC[3]);
  pGw.write(pCtx->vVersion);
  pGw.write(cNTensors);
  pGw.write(cNKv);

  // write key-value pairs
  for (int64_t vI = 0; vI < cNKv; ++vI) {
    pGw.write(pCtx->kv[vI]);
  }

  // write tensor info
  for (int64_t vI = 0; vI < cNTensors; ++vI) {
    pGw.mWriteTensorMeta(pCtx->vInfo[vI]);
  }

  // we require the data section to be aligned
  pGw.mPad(pCtx->vAlignment);

  if (pOnlyMeta) {
    return;
  }

  const size_t cOffsetData = pGw.vWrittenBytes;

  // write tensor data
  for (int64_t vI = 0; vI < cNTensors; ++vI) {
    pGw.mWriteTensorData(pCtx->vInfo[vI], cOffsetData, pCtx->vAlignment);
  }
}

void gguf_write_to_buf(const struct gguf_context *pCtx, std::vector<int8_t> &pBuf, bool pOnlyMeta) {
  gguf_writer_buf vGw(pBuf);
  fGgufWriteOut(pCtx, vGw, pOnlyMeta);
}

bool gguf_write_to_file_ptr(const struct gguf_context *pCtx, FILE *pFile, bool pOnlyMeta) {
  GGML_ASSERT(pFile);

  try {
    gguf_writer_file vGw(pFile);
    fGgufWriteOut(pCtx, vGw, pOnlyMeta);
  } catch (const std::runtime_error &vEx) {
    GGML_LOG_ERROR("%s: failed to write GGUF data: %s\n", __func__, vEx.what());
    return false;
  }
  return true;
}

bool gguf_write_to_file(const struct gguf_context *pCtx, const char *pFname, bool pOnlyMeta) {
  FILE *vFile = ggml_fopen(pFname, "wb");

  if (!vFile) {
    GGML_LOG_ERROR("%s: failed to open file '%s' for writing GGUF data\n", __func__, pFname);
    return false;
  }

  const bool cSuccess = gguf_write_to_file_ptr(pCtx, vFile, pOnlyMeta);
  if (!cSuccess) {
    GGML_LOG_ERROR("%s: failed to write GGUF data into '%s'\n", __func__, pFname);
  }

  fclose(vFile);
  return cSuccess;
}

size_t gguf_get_meta_size(const struct gguf_context *pCtx) {
  // only return size
  std::vector<int8_t> vBuf;
  gguf_write_to_buf(pCtx, vBuf, /*only_meta =*/true);
  return vBuf.size();
}

void gguf_get_meta_data(const struct gguf_context *pCtx, void *pData) {
  std::vector<int8_t> vBuf;
  gguf_write_to_buf(pCtx, vBuf, /*only_meta =*/true);
  memcpy(pData, vBuf.data(), vBuf.size());
}
