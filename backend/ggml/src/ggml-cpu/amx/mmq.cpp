#if defined(__GNUC__)
#pragma GCC diagnostic ignored "-Wpedantic"
#pragma GCC diagnostic ignored "-Wunused-local-typedefs"
#endif

#include "amx.h"
#include "mmq.h"
#include "ggml-impl.h"
#include "ggml-cpu-impl.h"
#include "simd-mappings.h"
#include "quants.h"
#include "ggml-quants.h"
#include <algorithm>
#include <type_traits>

#if defined(__gnu_linux__)
#include <sys/syscall.h>
#include <unistd.h>
#endif

#if (defined(_WIN32) || defined(_WIN64))
#define RESTRICT __restrict
#else
#define RESTRICT __restrict__
#endif

#if (defined(_WIN32) || defined(_WIN64))
#define ALWAYS_INLINE __forceinline
#elif __has_attribute(always_inline) || defined(__GNUC__)
#define ALWAYS_INLINE __attribute__((__always_inline__)) inline
#else
#define ALWAYS_INLINE inline
#endif

#if defined(__AMX_INT8__) && defined(__AVX512VNNI__)

namespace {

// Forced unrolling
template <int n> struct Unroll {
  template <typename Func, typename... Args> ALWAYS_INLINE void operator()(const Func &pF, Args... pArgs) const {
    Unroll<n - 1>{}(pF, pArgs...);
    pF(std::integral_constant<int, n - 1>{}, pArgs...);
  }
};

template <> struct Unroll<1> {
  template <typename Func, typename... Args> ALWAYS_INLINE void operator()(const Func &pF, Args... pArgs) const {
    pF(std::integral_constant<int, 0>{}, pArgs...);
  }
};

// type traits
template <typename T> struct PackedTypes {};
template <> struct PackedTypes<block_q4_0> {
  using type = int8_t;
};
template <> struct PackedTypes<block_q4_1> {
  using type = uint8_t;
};
template <> struct PackedTypes<block_q8_0> {
  using type = int8_t;
};
template <typename T> using packed_B_type = typename PackedTypes<T>::type;

template <typename T> struct do_compensate : std::integral_constant<bool, std::is_same<T, block_q8_0>::value> {};

template <typename T>
struct do_unpack
    : std::integral_constant<bool, std::is_same<T, block_q4_0>::value || std::is_same<T, block_q4_1>::value> {};

template <typename T>
struct is_type_qkk
    : std::integral_constant<bool, std::is_same<T, block_q4_K>::value || std::is_same<T, block_q5_K>::value ||
                                       std::is_same<T, block_q6_K>::value || std::is_same<T, block_iq4_xs>::value> {};

#define GGML_DISPATCH_FLOATING_TYPES(TYPE, ...)                                                                        \
  [&] {                                                                                                                \
    switch (TYPE) {                                                                                                    \
    case GGML_TYPE_F16: {                                                                                              \
      using type = ggml_fp16_t;                                                                                        \
      constexpr int blck_size = 16;                                                                                    \
      return __VA_ARGS__();                                                                                            \
    }                                                                                                                  \
    case GGML_TYPE_BF16: {                                                                                             \
      using type = ggml_bf16_t;                                                                                        \
      constexpr int blck_size = 32;                                                                                    \
      return __VA_ARGS__();                                                                                            \
    }                                                                                                                  \
    default:                                                                                                           \
      fprintf(stderr, "Unsupported floating data type\n");                                                             \
    }                                                                                                                  \
  }()

#define GGML_DISPATCH_QTYPES(QT, ...)                                                                                  \
  [&] {                                                                                                                \
    switch (QT) {                                                                                                      \
    case GGML_TYPE_Q4_0: {                                                                                             \
      using type = block_q4_0;                                                                                         \
      using vec_dot_type = block_q8_0;                                                                                 \
      constexpr int blck_size = QK4_0;                                                                                 \
      return __VA_ARGS__();                                                                                            \
    }                                                                                                                  \
    case GGML_TYPE_Q4_1: {                                                                                             \
      using type = block_q4_1;                                                                                         \
      using vec_dot_type = block_q8_1;                                                                                 \
      constexpr int blck_size = QK4_1;                                                                                 \
      return __VA_ARGS__();                                                                                            \
    }                                                                                                                  \
    case GGML_TYPE_Q8_0: {                                                                                             \
      using type = block_q8_0;                                                                                         \
      using vec_dot_type = block_q8_0;                                                                                 \
      constexpr int blck_size = QK8_0;                                                                                 \
      return __VA_ARGS__();                                                                                            \
    }                                                                                                                  \
    case GGML_TYPE_Q4_K: {                                                                                             \
      using type = block_q4_K;                                                                                         \
      using vec_dot_type = block_q8_K;                                                                                 \
      constexpr int blck_size = QK_K;                                                                                  \
      return __VA_ARGS__();                                                                                            \
    }                                                                                                                  \
    case GGML_TYPE_Q5_K: {                                                                                             \
      using type = block_q5_K;                                                                                         \
      using vec_dot_type = block_q8_K;                                                                                 \
      constexpr int blck_size = QK_K;                                                                                  \
      return __VA_ARGS__();                                                                                            \
    }                                                                                                                  \
    case GGML_TYPE_Q6_K: {                                                                                             \
      using type = block_q6_K;                                                                                         \
      using vec_dot_type = block_q8_K;                                                                                 \
      constexpr int blck_size = QK_K;                                                                                  \
      return __VA_ARGS__();                                                                                            \
    }                                                                                                                  \
    case GGML_TYPE_IQ4_XS: {                                                                                           \
      using type = block_iq4_xs;                                                                                       \
      using vec_dot_type = block_q8_K;                                                                                 \
      constexpr int blck_size = QK_K;                                                                                  \
      return __VA_ARGS__();                                                                                            \
    }                                                                                                                  \
    default:                                                                                                           \
      fprintf(stderr, "Unsupported quantized data type: %d\n", int(TYPE));                                             \
    }                                                                                                                  \
  }()

#define GGML_DISPATCH_BOOL(BOOL_V, BOOL_NAME, ...)                                                                     \
  [&] {                                                                                                                \
    if (BOOL_V) {                                                                                                      \
      constexpr bool BOOL_NAME = true;                                                                                 \
      return __VA_ARGS__();                                                                                            \
    } else {                                                                                                           \
      constexpr bool BOOL_NAME = false;                                                                                \
      return __VA_ARGS__();                                                                                            \
    }                                                                                                                  \
  }()

// define amx tile config data structure
struct tile_config_t {
  uint8_t vPaletteId = 0;
  uint8_t vStartRow = 0;
  uint8_t aReserved0[14] = {0};
  uint16_t aColsb[16] = {0};
  uint8_t aRows[16] = {0};
};

// Notes: amx tile config
//
// Typically, TMUL calculates A and B of size 16 x 64 containing INT8 values,
// and accumulate the result to a 16 x 16 matrix C containing INT32 values,
//
// As many GGUF quantized types as `block_size` of 32, so a 16-16-32 config is used
// instead of the normally used 16-16-64 config.
//
//    Block A: {16, 32}, dtype = int8_t
//    Block B: {16, 32}, dtype = uint8_t/int8_t
//    Block C: {16, 16}, dtype = int32_t
//
// Block B needs to be prepacked to vnni format before feeding into  TMUL:
//    packed_B: from {n, k} to {k/vnni_blk, n, vnni_blck}, viewed in 2d, we get {8, 64}
//
// Therefore, we get tileconfig:
//             A    B    C
//    rows    16    8   16
//    colsb   32   64   16
//
// For tile distribution, follow a 2-2-4 pattern, e.g. A used TMM2-TMM3, B used TMM0-TMM1,
// C used TMM4-TMM7:
//            B TMM0  B TMM1
//    A TMM2  C TMM4  C TMM6
//    A TMM3  C TMM5  C TMM7
//
// Each `amx` kernel handles 4 blocks at a time: 2MB * 2NB, when m < 2 * BLOCK_M, unpack A
// will be needed.
//
// Here another commonly used pattern 1-3-3 is skipped, as it is mostly used when m <=16;
// and the single batch gemm (m=1) has a special fast path with `avx512-vnni`.
//
// ref: https://www.intel.com/content/www/us/en/developer/articles/code-sample/
//    advanced-matrix-extensions-intrinsics-functions.html
//

inline void fGgmlTileConfigInit(void) {
  static thread_local bool vDone = false;

  if (vDone) {
    return;
  }

  alignas(64) tile_config_t vTc = {};
  vTc.vPaletteId = 1;
  vTc.vStartRow = 0;
  vTc.aRows[0] = 8;
  vTc.aColsb[0] = 64;
  vTc.aRows[1] = 8;
  vTc.aColsb[1] = 64;
  vTc.aRows[2] = 16;
  vTc.aColsb[2] = 32;
  vTc.aRows[3] = 16;
  vTc.aColsb[3] = 32;
  vTc.aRows[4] = 16;
  vTc.aColsb[4] = 64;
  vTc.aRows[5] = 16;
  vTc.aColsb[5] = 64;
  vTc.aRows[6] = 16;
  vTc.aColsb[6] = 64;
  vTc.aRows[7] = 16;
  vTc.aColsb[7] = 64;

  _tile_loadconfig(&vTc);
  vDone = true;
}

// we need an extra 16 * 4B (TILE_N * int32_t) for each NB/KB block for compensation.
// See the notes `s8s8 igemm compensation in avx512-vnni` for detail.
template <typename TB> int fGetTileSize() {
  int vTileSize = TILE_N * sizeof(TB);
  if (do_compensate<TB>::value) {
    vTileSize += TILE_N * sizeof(int32_t);
  }
  if (std::is_same<TB, block_q4_K>::value || std::is_same<TB, block_q5_K>::value) {
    vTileSize += TILE_N * 4;
  }
  if (std::is_same<TB, block_iq4_xs>::value) {
    vTileSize += TILE_N * 2;
  }
  return vTileSize;
}

template <typename TB, int BLOCK_K> int fGetRowSize(int pK) {
  int vKb = pK / BLOCK_K;
  int vRowSize = vKb * sizeof(TB);
  if (do_compensate<TB>::value) {
    vRowSize += vKb * sizeof(int32_t);
  }
  if (std::is_same<TB, block_q4_K>::value || std::is_same<TB, block_q5_K>::value) {
    vRowSize += vKb * 4;
  }
  if (std::is_same<TB, block_iq4_xs>::value) {
    vRowSize += vKb * 2;
  }
  return vRowSize;
}

// transpose utils
#define SHUFFLE_EPI32(a, b, mask)                                                                                      \
  _mm256_castps_si256(_mm256_shuffle_ps(_mm256_castsi256_ps(a), _mm256_castsi256_ps(b), mask))
inline void fTranspose8x832bit(__m256i *pV, __m256i *pV1) {
  // unpacking and 32-bit elements
  pV1[0] = _mm256_unpacklo_epi32(pV[0], pV[1]);
  pV1[1] = _mm256_unpackhi_epi32(pV[0], pV[1]);
  pV1[2] = _mm256_unpacklo_epi32(pV[2], pV[3]);
  pV1[3] = _mm256_unpackhi_epi32(pV[2], pV[3]);
  pV1[4] = _mm256_unpacklo_epi32(pV[4], pV[5]);
  pV1[5] = _mm256_unpackhi_epi32(pV[4], pV[5]);
  pV1[6] = _mm256_unpacklo_epi32(pV[6], pV[7]);
  pV1[7] = _mm256_unpackhi_epi32(pV[6], pV[7]);

  // shuffling the 32-bit elements
  pV[0] = SHUFFLE_EPI32(pV1[0], pV1[2], 0x44);
  pV[1] = SHUFFLE_EPI32(pV1[0], pV1[2], 0xee);
  pV[2] = SHUFFLE_EPI32(pV1[4], pV1[6], 0x44);
  pV[3] = SHUFFLE_EPI32(pV1[4], pV1[6], 0xee);
  pV[4] = SHUFFLE_EPI32(pV1[1], pV1[3], 0x44);
  pV[5] = SHUFFLE_EPI32(pV1[1], pV1[3], 0xee);
  pV[6] = SHUFFLE_EPI32(pV1[5], pV1[7], 0x44);
  pV[7] = SHUFFLE_EPI32(pV1[5], pV1[7], 0xee);

  // shuffling 128-bit elements
  pV1[0] = _mm256_permute2f128_si256(pV[2], pV[0], 0x02);
  pV1[1] = _mm256_permute2f128_si256(pV[3], pV[1], 0x02);
  pV1[2] = _mm256_permute2f128_si256(pV[6], pV[4], 0x02);
  pV1[3] = _mm256_permute2f128_si256(pV[7], pV[5], 0x02);
  pV1[4] = _mm256_permute2f128_si256(pV[2], pV[0], 0x13);
  pV1[5] = _mm256_permute2f128_si256(pV[3], pV[1], 0x13);
  pV1[6] = _mm256_permute2f128_si256(pV[6], pV[4], 0x13);
  pV1[7] = _mm256_permute2f128_si256(pV[7], pV[5], 0x13);
}

inline void fTranspose16x432bit(__m512i *pR, __m512i *pD) {

  static const __m512i cIndex1 =
      _mm512_set_epi32(0x0f, 0x0b, 0x07, 0x03, 0x0e, 0x0a, 0x06, 0x02, 0x0d, 0x09, 0x05, 0x01, 0x0c, 0x08, 0x04, 0x00);

  pD[0] = _mm512_permutexvar_epi32(cIndex1, pR[0]);
  pD[1] = _mm512_permutexvar_epi32(cIndex1, pR[1]);
  pD[2] = _mm512_permutexvar_epi32(cIndex1, pR[2]);
  pD[3] = _mm512_permutexvar_epi32(cIndex1, pR[3]);

  pR[0] = _mm512_shuffle_i32x4(pD[0], pD[1], 0x44);
  pR[1] = _mm512_shuffle_i32x4(pD[0], pD[1], 0xee);
  pR[2] = _mm512_shuffle_i32x4(pD[2], pD[3], 0x44);
  pR[3] = _mm512_shuffle_i32x4(pD[2], pD[3], 0xee);

  pD[0] = _mm512_shuffle_i32x4(pR[0], pR[2], 0x88);
  pD[1] = _mm512_shuffle_i32x4(pR[0], pR[2], 0xdd);
  pD[2] = _mm512_shuffle_i32x4(pR[1], pR[3], 0x88);
  pD[3] = _mm512_shuffle_i32x4(pR[1], pR[3], 0xdd);
}

inline void fTranspose16x1632bit(__m512i *pV) {
  __m512i v1[16];
  v1[0] = _mm512_unpacklo_epi32(pV[0], pV[1]);
  v1[1] = _mm512_unpackhi_epi32(pV[0], pV[1]);
  v1[2] = _mm512_unpacklo_epi32(pV[2], pV[3]);
  v1[3] = _mm512_unpackhi_epi32(pV[2], pV[3]);
  v1[4] = _mm512_unpacklo_epi32(pV[4], pV[5]);
  v1[5] = _mm512_unpackhi_epi32(pV[4], pV[5]);
  v1[6] = _mm512_unpacklo_epi32(pV[6], pV[7]);
  v1[7] = _mm512_unpackhi_epi32(pV[6], pV[7]);
  v1[8] = _mm512_unpacklo_epi32(pV[8], pV[9]);
  v1[9] = _mm512_unpackhi_epi32(pV[8], pV[9]);
  v1[10] = _mm512_unpacklo_epi32(pV[10], pV[11]);
  v1[11] = _mm512_unpackhi_epi32(pV[10], pV[11]);
  v1[12] = _mm512_unpacklo_epi32(pV[12], pV[13]);
  v1[13] = _mm512_unpackhi_epi32(pV[12], pV[13]);
  v1[14] = _mm512_unpacklo_epi32(pV[14], pV[15]);
  v1[15] = _mm512_unpackhi_epi32(pV[14], pV[15]);

  pV[0] = _mm512_unpacklo_epi64(v1[0], v1[2]);
  pV[1] = _mm512_unpackhi_epi64(v1[0], v1[2]);
  pV[2] = _mm512_unpacklo_epi64(v1[1], v1[3]);
  pV[3] = _mm512_unpackhi_epi64(v1[1], v1[3]);
  pV[4] = _mm512_unpacklo_epi64(v1[4], v1[6]);
  pV[5] = _mm512_unpackhi_epi64(v1[4], v1[6]);
  pV[6] = _mm512_unpacklo_epi64(v1[5], v1[7]);
  pV[7] = _mm512_unpackhi_epi64(v1[5], v1[7]);
  pV[8] = _mm512_unpacklo_epi64(v1[8], v1[10]);
  pV[9] = _mm512_unpackhi_epi64(v1[8], v1[10]);
  pV[10] = _mm512_unpacklo_epi64(v1[9], v1[11]);
  pV[11] = _mm512_unpackhi_epi64(v1[9], v1[11]);
  pV[12] = _mm512_unpacklo_epi64(v1[12], v1[14]);
  pV[13] = _mm512_unpackhi_epi64(v1[12], v1[14]);
  pV[14] = _mm512_unpacklo_epi64(v1[13], v1[15]);
  pV[15] = _mm512_unpackhi_epi64(v1[13], v1[15]);

  v1[0] = _mm512_shuffle_i32x4(pV[0], pV[4], 0x88);
  v1[1] = _mm512_shuffle_i32x4(pV[1], pV[5], 0x88);
  v1[2] = _mm512_shuffle_i32x4(pV[2], pV[6], 0x88);
  v1[3] = _mm512_shuffle_i32x4(pV[3], pV[7], 0x88);
  v1[4] = _mm512_shuffle_i32x4(pV[0], pV[4], 0xdd);
  v1[5] = _mm512_shuffle_i32x4(pV[1], pV[5], 0xdd);
  v1[6] = _mm512_shuffle_i32x4(pV[2], pV[6], 0xdd);
  v1[7] = _mm512_shuffle_i32x4(pV[3], pV[7], 0xdd);
  v1[8] = _mm512_shuffle_i32x4(pV[8], pV[12], 0x88);
  v1[9] = _mm512_shuffle_i32x4(pV[9], pV[13], 0x88);
  v1[10] = _mm512_shuffle_i32x4(pV[10], pV[14], 0x88);
  v1[11] = _mm512_shuffle_i32x4(pV[11], pV[15], 0x88);
  v1[12] = _mm512_shuffle_i32x4(pV[8], pV[12], 0xdd);
  v1[13] = _mm512_shuffle_i32x4(pV[9], pV[13], 0xdd);
  v1[14] = _mm512_shuffle_i32x4(pV[10], pV[14], 0xdd);
  v1[15] = _mm512_shuffle_i32x4(pV[11], pV[15], 0xdd);

  pV[0] = _mm512_shuffle_i32x4(v1[0], v1[8], 0x88);
  pV[1] = _mm512_shuffle_i32x4(v1[1], v1[9], 0x88);
  pV[2] = _mm512_shuffle_i32x4(v1[2], v1[10], 0x88);
  pV[3] = _mm512_shuffle_i32x4(v1[3], v1[11], 0x88);
  pV[4] = _mm512_shuffle_i32x4(v1[4], v1[12], 0x88);
  pV[5] = _mm512_shuffle_i32x4(v1[5], v1[13], 0x88);
  pV[6] = _mm512_shuffle_i32x4(v1[6], v1[14], 0x88);
  pV[7] = _mm512_shuffle_i32x4(v1[7], v1[15], 0x88);
  pV[8] = _mm512_shuffle_i32x4(v1[0], v1[8], 0xdd);
  pV[9] = _mm512_shuffle_i32x4(v1[1], v1[9], 0xdd);
  pV[10] = _mm512_shuffle_i32x4(v1[2], v1[10], 0xdd);
  pV[11] = _mm512_shuffle_i32x4(v1[3], v1[11], 0xdd);
  pV[12] = _mm512_shuffle_i32x4(v1[4], v1[12], 0xdd);
  pV[13] = _mm512_shuffle_i32x4(v1[5], v1[13], 0xdd);
  pV[14] = _mm512_shuffle_i32x4(v1[6], v1[14], 0xdd);
  pV[15] = _mm512_shuffle_i32x4(v1[7], v1[15], 0xdd);
}

void fQuantizeRowQ8KVnni(const float *RESTRICT pX, void *RESTRICT pVy, int64_t pK) {
  assert(pK % QK_K == 0);
  const int cKb = pK / QK_K;
  constexpr int cKVecs = QK_K / 16;

  block_q8_K *vY = reinterpret_cast<block_q8_K *>(pVy);

  // hold 16 float vecs from x
  __m512 v_[cKVecs];

  // hold the quants vecs
  __m512i aVq[cKVecs / 4];

  // hold the packed quants vecs
  __m512i aVqPacked[cKVecs / 4];

  const __m512 cSignBit = _mm512_set1_ps(-0.f);

  for (int vI = 0; vI < cKb; ++vI) {
    // Compute max(abs(e)) for the block
    __m512 vVamax = _mm512_set1_ps(0.f);
    for (int vJ = 0; vJ < cKVecs; ++vJ) {
      v_[vJ] = _mm512_loadu_ps(pX);
      pX += 16;
      vVamax = _mm512_max_ps(vVamax, _mm512_andnot_ps(cSignBit, v_[vJ]));
    }
    const float cAmax = _mm512_reduce_max_ps(vVamax);

    // Quantize these floats
    const float cIscale = 127.f / cAmax;
    vY[vI].d = GGML_CPU_FP32_TO_FP16(1 / cIscale);
    const float cId = (cAmax != 0.0f) ? cIscale : 0.f;
    const __m512 cVscale = _mm512_set1_ps(cId);

    // Apply multiplier and round to nearest integer
    for (int vJ = 0; vJ < cKVecs; ++vJ) {
      v_[vJ] = _mm512_mul_ps(v_[vJ], cVscale);
      v_[vJ] = _mm512_roundscale_ps(v_[vJ], (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
    }

    // Pack to epi8 vecs
    for (int vJ = 0; vJ < cKVecs / 4; ++vJ) {
      __m128i vQ80 = _mm512_cvtepi32_epi8(_mm512_cvtps_epi32(v_[vJ * 4 + 0]));
      __m128i vQ81 = _mm512_cvtepi32_epi8(_mm512_cvtps_epi32(v_[vJ * 4 + 1]));
      __m128i vQ82 = _mm512_cvtepi32_epi8(_mm512_cvtps_epi32(v_[vJ * 4 + 2]));
      __m128i vQ83 = _mm512_cvtepi32_epi8(_mm512_cvtps_epi32(v_[vJ * 4 + 3]));

      __m256i vQ801 = _mm256_insertf128_si256(_mm256_castsi128_si256(vQ80), (vQ81), 1);
      __m256i vQ823 = _mm256_insertf128_si256(_mm256_castsi128_si256(vQ82), (vQ83), 1);

      aVq[vJ] = _mm512_inserti32x8(_mm512_castsi256_si512(vQ801), vQ823, 1);
      _mm512_storeu_si512((__m512i *)(vY[vI].qs + vJ * 64), aVq[vJ]);
    }

    // Compute the bsums with vnni
    fTranspose16x432bit(aVq, aVqPacked);

    const __m512i cOne = _mm512_set1_epi8(1);
    __m512i vSum = _mm512_setzero_si512();
    for (int vK = 0; vK < 4; ++vK) {
      vSum = _mm512_dpbusd_epi32(vSum, cOne, aVqPacked[vK]);
    }
    _mm256_storeu_si256((__m256i *)(vY[vI].bsums), _mm512_cvtepi32_epi16(vSum));
  }
}

// quantize A from float to `vec_dot_type`
template <typename T> inline void from_float(const float *pX, char *pVy, int64_t pK);

template <> inline void from_float<block_q8_0>(const float *pX, char *pVy, int64_t pK) {
  quantize_row_q8_0(pX, (block_q8_0 *)pVy, pK);
}

template <> inline void from_float<block_q8_1>(const float *pX, char *pVy, int64_t pK) {
  quantize_row_q8_1(pX, (block_q8_1 *)pVy, pK);
}

template <> inline void from_float<block_q8_K>(const float *pX, char *pVy, int64_t pK) {
#if 1
  // TODO: this is reference impl!
  quantize_row_q8_K_ref(pX, (block_q8_K *)pVy, pK);
#else
  fQuantizeRowQ8KVnni(pX, pVy, pK);
#endif
}

// load A from memory to array when nrows can not fill in whole tile
void unpack_A(int8_t *RESTRICT pTile, const block_q8_0 *RESTRICT pA, int pLda, int pNr) {
  assert(pNr != TILE_M);
  for (int vM = 0; vM < pNr; ++vM) {
    const __m256i cV = _mm256_loadu_si256((const __m256i *)(pA[vM * pLda].qs));
    _mm256_storeu_si256((__m256i *)(pTile + vM * TILE_K), cV);
  }
}

void unpack_A(int8_t *RESTRICT pTile, const block_q8_1 *RESTRICT pA, int pLda, int pNr) {
  assert(pNr != TILE_M);
  for (int vM = 0; vM < pNr; ++vM) {
    const __m256i cV = _mm256_loadu_si256((const __m256i *)(pA[vM * pLda].qs));
    _mm256_storeu_si256((__m256i *)(pTile + vM * TILE_K), cV);
  }
}

template <typename TB> void unpack_A(int8_t *RESTRICT pTile, const block_q8_K *RESTRICT pA, int pLda, int pK, int pNr) {
  assert(pNr <= TILE_M);
  for (int vM = 0; vM < pNr; ++vM) {
    const __m256i cV = _mm256_loadu_si256((const __m256i *)(pA[vM * pLda].qs + pK * 32));
    _mm256_storeu_si256((__m256i *)(pTile + vM * TILE_K), cV);
  }
}

template <>
void unpack_A<block_q6_K>(int8_t *RESTRICT pTile, const block_q8_K *RESTRICT pA, int pLda, int pK, int pNr) {
  assert(pNr <= TILE_M);
  // zero padding k from 16 to 32, so that we don't have to re-config amx
  const __m128i cZero = _mm_setzero_si128();
  for (int vM = 0; vM < pNr; ++vM) {
    const __m128i cV = _mm_loadu_si128((const __m128i *)(pA[vM * pLda].qs + pK * 16));
    const __m256i cR = _mm256_insertf128_si256(_mm256_castsi128_si256(cV), cZero, 1);
    _mm256_storeu_si256((__m256i *)(pTile + vM * TILE_K), cR);
  }
}

#define MM256_SET_M128I(a, b) _mm256_insertf128_si256(_mm256_castsi128_si256(b), (a), 1)
inline __m256i fBytesFromNibbles32(const uint8_t *pRsi) {
  const __m128i cTmp = _mm_loadu_si128((const __m128i *)pRsi);
  const __m256i cBytes = MM256_SET_M128I(_mm_srli_epi16(cTmp, 4), cTmp);
  const __m256i cLowMask = _mm256_set1_epi8(0xF);
  return _mm256_and_si256(cLowMask, cBytes);
}

// used for block_q4_K
inline __m512i fBytesFromNibbles64(const uint8_t *pRsi) {
  const __m256i cTmp = _mm256_loadu_si256((const __m256i *)pRsi);
  const __m256i cLowMask = _mm256_set1_epi8(0xF);
  const __m256i cQ4l = _mm256_and_si256(cTmp, cLowMask);
  const __m256i cQ4h = _mm256_and_si256(_mm256_srli_epi16(cTmp, 4), cLowMask);
  return _mm512_inserti32x8(_mm512_castsi256_si512(cQ4l), cQ4h, 1);
}

// used for block_q5_K
inline __m512i fBytesFromNibbles64(const uint8_t *pQs, const uint8_t *pQh, int pK) {
  const __m256i cLowMask = _mm256_set1_epi8(0xF);
  __m256i vHmask = _mm256_set1_epi8(1);
  vHmask = _mm256_slli_epi16(vHmask, pK);

  const __m256i cQ5bits = _mm256_loadu_si256((const __m256i *)pQs);
  const __m256i cHbits = _mm256_loadu_si256((const __m256i *)pQh);

  const __m256i cQ5l0 = _mm256_and_si256(cQ5bits, cLowMask);
  const __m256i cQ5h0 = _mm256_slli_epi16(_mm256_srli_epi16(_mm256_and_si256(cHbits, vHmask), pK + 0), 4);
  const __m256i cQ50 = _mm256_add_epi8(cQ5l0, cQ5h0);
  vHmask = _mm256_slli_epi16(vHmask, 1);

  const __m256i cQ5l1 = _mm256_and_si256(_mm256_srli_epi16(cQ5bits, 4), cLowMask);
  const __m256i cQ5h1 = _mm256_slli_epi16(_mm256_srli_epi16(_mm256_and_si256(cHbits, vHmask), pK + 1), 4);
  const __m256i cQ51 = _mm256_add_epi8(cQ5l1, cQ5h1);

  return _mm512_inserti32x8(_mm512_castsi256_si512(cQ50), cQ51, 1);
}

// used for block_q6_K
inline void fBytesFromNibbles128(__m512i &pR0, __m512i &pR1, const uint8_t *pQs, const uint8_t *pQh) {
  const __m256i cM4 = _mm256_set1_epi8(0xF);
  const __m256i cM2 = _mm256_set1_epi8(0x3);

  const __m256i cQ6bits1 = _mm256_loadu_si256((const __m256i *)pQs);
  const __m256i cQ6bits2 = _mm256_loadu_si256((const __m256i *)(pQs + 32));
  const __m256i cQ6bitsH = _mm256_loadu_si256((const __m256i *)pQh);

  const __m256i cQ6h0 = _mm256_slli_epi16(_mm256_and_si256(cQ6bitsH, cM2), 4);
  const __m256i cQ6h1 = _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(cQ6bitsH, 2), cM2), 4);
  const __m256i cQ6h2 = _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(cQ6bitsH, 4), cM2), 4);
  const __m256i cQ6h3 = _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(cQ6bitsH, 6), cM2), 4);

  const __m256i cQ60 = _mm256_or_si256(_mm256_and_si256(cQ6bits1, cM4), cQ6h0);
  const __m256i cQ61 = _mm256_or_si256(_mm256_and_si256(cQ6bits2, cM4), cQ6h1);
  const __m256i cQ62 = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(cQ6bits1, 4), cM4), cQ6h2);
  const __m256i cQ63 = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(cQ6bits2, 4), cM4), cQ6h3);

  pR0 = _mm512_inserti32x8(_mm512_castsi256_si512(cQ60), cQ61, 1);
  pR1 = _mm512_inserti32x8(_mm512_castsi256_si512(cQ62), cQ63, 1);
}

inline __m512i fPackNibbles(__m512i pR0, __m512i pR1) { return _mm512_or_si512(pR0, _mm512_slli_epi16(pR1, 4)); }

template <typename TB> inline void pack_qs(void *RESTRICT pPackedB, const TB *RESTRICT pB, int pKb) {
  int8_t aTmp[8 * 64];
  __m256i v_[8], v2[8];
  for (int vN = 0; vN < 8; ++vN) {
    v_[vN] = fBytesFromNibbles32(pB[vN * pKb].qs);
  }
  fTranspose8x832bit(v_, v2);
  for (int vN = 0; vN < 8; ++vN) {
    _mm256_storeu_si256((__m256i *)(aTmp + vN * 64), v2[vN]);
  }
  for (int vN = 0; vN < 8; ++vN) {
    v_[vN] = fBytesFromNibbles32(pB[(vN + 8) * pKb].qs);
  }
  fTranspose8x832bit(v_, v2);
  for (int vN = 0; vN < 8; ++vN) {
    _mm256_storeu_si256((__m256i *)(aTmp + vN * 64 + 32), v2[vN]);
  }

  // pack again with 128 to fully utilize vector length
  for (int vN = 0; vN < 8; vN += 2) {
    __m512i vR0 = _mm512_loadu_si512((const __m512i *)(aTmp + vN * 64));
    __m512i vR1 = _mm512_loadu_si512((const __m512i *)(aTmp + vN * 64 + 64));
    __m512i vR1r0 = fPackNibbles(vR0, vR1);
    _mm512_storeu_si512((__m512i *)((char *)pPackedB + vN * 32), vR1r0);
  }
}

template <> inline void pack_qs<block_q8_0>(void *RESTRICT pPackedB, const block_q8_0 *RESTRICT pB, int pKb) {
  __m256i v_[8], v2[8];
  for (int vN = 0; vN < 8; ++vN) {
    v_[vN] = _mm256_loadu_si256((const __m256i *)(pB[vN * pKb].qs));
  }
  fTranspose8x832bit(v_, v2);
  for (int vN = 0; vN < 8; ++vN) {
    _mm256_storeu_si256((__m256i *)((char *)pPackedB + vN * 64), v2[vN]);
  }
  for (int vN = 0; vN < 8; ++vN) {
    v_[vN] = _mm256_loadu_si256((const __m256i *)(pB[(vN + 8) * pKb].qs));
  }
  fTranspose8x832bit(v_, v2);
  for (int vN = 0; vN < 8; ++vN) {
    _mm256_storeu_si256((__m256i *)((char *)pPackedB + vN * 64 + 32), v2[vN]);
  }
}

template <> inline void pack_qs<block_q4_K>(void *RESTRICT pPackedB, const block_q4_K *RESTRICT pB, int pKb) {
  __m512i v_[16];
  // QK_K 256 with 8 groups, handle 2 groups at a time
  char *vPb = (char *)pPackedB;
  for (int vK = 0; vK < QK_K / 64; ++vK) {
    // pack 2 groups { n, g,  k} to {g, k/4, 4n}
    //          e.g. {16, 2, 32} to {2,   8, 64}
    for (int vN = 0; vN < TILE_N; ++vN) {
      v_[vN] = fBytesFromNibbles64(pB[vN * pKb].qs + vK * 32);
    }

    fTranspose16x1632bit(v_);

    // pack again with 128 to fully utilize vector length
    for (int vN = 0; vN < TILE_N; vN += 2) {
      _mm512_storeu_si512((__m512i *)vPb, fPackNibbles(v_[vN], v_[vN + 1]));
      vPb += 64;
    }
  }
}

template <> inline void pack_qs<block_q5_K>(void *RESTRICT pPackedB, const block_q5_K *RESTRICT pB, int pKb) {
  __m512i v_[16];
  const __m512i cLowMask = _mm512_set1_epi8(0xF);
  // QK_K 256 with 8 groups, handle 2 groups at a time
  char *vPb = (char *)pPackedB;
  char *vPh = (char *)pPackedB + (QK_K / 2) * TILE_N;
  for (int vK = 0; vK < QK_K / 64; ++vK) {
    // pack 2 groups { n, g,  k} to {g, k/4, 4n}
    //          e.g. {16, 2, 32} to {2,   8, 64}
    for (int vN = 0; vN < TILE_N; ++vN) {
      v_[vN] = fBytesFromNibbles64(pB[vN * pKb].qs + vK * 32, pB[vN * pKb].qh, /* group */ 2 * vK);
    }

    fTranspose16x1632bit(v_);

    // 1. pack lower 4bits with 2 groups
    for (int vN = 0; vN < TILE_N; vN += 2) {
      // get lower 4 bits
      const __m512i cR0 = _mm512_and_si512(v_[vN], cLowMask);
      const __m512i cR1 = _mm512_and_si512(v_[vN + 1], cLowMask);
      _mm512_storeu_si512((__m512i *)vPb, fPackNibbles(cR0, cR1));
      vPb += 64;
    }

    // 2. pack higher 1bit with 2 groups
    const __m512i cHmask = _mm512_set1_epi8(0x10);
    for (int vG = 0; vG < 2; ++vG) {
      __m512i vHbits = _mm512_setzero_si512();
      vHbits = _mm512_add_epi8(vHbits, _mm512_srli_epi16(_mm512_and_si512(v_[vG * 8 + 0], cHmask), 4));
      vHbits = _mm512_add_epi8(vHbits, _mm512_srli_epi16(_mm512_and_si512(v_[vG * 8 + 1], cHmask), 3));
      vHbits = _mm512_add_epi8(vHbits, _mm512_srli_epi16(_mm512_and_si512(v_[vG * 8 + 2], cHmask), 2));
      vHbits = _mm512_add_epi8(vHbits, _mm512_srli_epi16(_mm512_and_si512(v_[vG * 8 + 3], cHmask), 1));
      vHbits = _mm512_add_epi8(vHbits, _mm512_and_si512(v_[vG * 8 + 4], cHmask));
      vHbits = _mm512_add_epi8(vHbits, _mm512_slli_epi16(_mm512_and_si512(v_[vG * 8 + 5], cHmask), 1));
      vHbits = _mm512_add_epi8(vHbits, _mm512_slli_epi16(_mm512_and_si512(v_[vG * 8 + 6], cHmask), 2));
      vHbits = _mm512_add_epi8(vHbits, _mm512_slli_epi16(_mm512_and_si512(v_[vG * 8 + 7], cHmask), 3));
      _mm512_storeu_si512((__m512i *)vPh, vHbits);
      vPh += 64;
    }
  }
}

template <> inline void pack_qs<block_q6_K>(void *RESTRICT pPackedB, const block_q6_K *RESTRICT pB, int pKb) {
  __m512i v_[32];
  const __m512i cLowMask = _mm512_set1_epi8(0xF);
  // QK_K 256 with 8 groups, handle 4 groups at a time
  char *vPb = (char *)pPackedB;
  char *vPh = (char *)pPackedB + (QK_K / 2) * TILE_N;
  for (int vK = 0; vK < QK_K / 128; ++vK) {
    for (int vN = 0; vN < TILE_N; ++vN) {
      fBytesFromNibbles128(v_[vN], v_[vN + 16], pB[vN * pKb].ql + vK * 64, pB[vN * pKb].qh + vK * 32);
    }

    // top half: group 0,1 or 4,5; bottom half: group 2,3 or 6,7
    fTranspose16x1632bit(v_);
    fTranspose16x1632bit(v_ + 16);

    // 1. pack lower 4bits with 4 groups
    for (int vN = 0; vN < 32; vN += 2) {
      const __m512i cR0 = _mm512_and_si512(v_[vN], cLowMask);
      const __m512i cR1 = _mm512_and_si512(v_[vN + 1], cLowMask);
      _mm512_storeu_si512((__m512i *)vPb, fPackNibbles(cR0, cR1));
      vPb += 64;
    }

    // 2. pack higher 2bit with 4 groups
    const __m512i cHmask = _mm512_set1_epi8(0x30);
    for (int vG = 0; vG < 8; ++vG) {
      __m512i vHbits = _mm512_setzero_si512();
      vHbits = _mm512_add_epi8(vHbits, _mm512_srli_epi16(_mm512_and_si512(v_[vG * 4 + 0], cHmask), 4));
      vHbits = _mm512_add_epi8(vHbits, _mm512_srli_epi16(_mm512_and_si512(v_[vG * 4 + 1], cHmask), 2));
      vHbits = _mm512_add_epi8(vHbits, _mm512_and_si512(v_[vG * 4 + 2], cHmask));
      vHbits = _mm512_add_epi8(vHbits, _mm512_slli_epi16(_mm512_and_si512(v_[vG * 4 + 3], cHmask), 2));
      _mm512_storeu_si512((__m512i *)vPh, vHbits);
      vPh += 64;
    }
  }
}

template <> inline void pack_qs<block_iq4_xs>(void *RESTRICT pPackedB, const block_iq4_xs *RESTRICT pB, int pKb) {
  __m512i v_[16];
  char *vPb = (char *)pPackedB;
  for (int vK = 0; vK < QK_K / 64; ++vK) {
    for (int vN = 0; vN < TILE_N; ++vN) {
      __m256i vR0 = fBytesFromNibbles32(pB[vN * pKb].qs + vK * 32 + 0);
      __m256i vR1 = fBytesFromNibbles32(pB[vN * pKb].qs + vK * 32 + 16);
      v_[vN] = _mm512_inserti32x8(_mm512_castsi256_si512(vR0), vR1, 1);
    }

    fTranspose16x1632bit(v_);

    // pack again with 128 to fully utilize vector length
    for (int vN = 0; vN < TILE_N; vN += 2) {
      _mm512_storeu_si512((__m512i *)vPb, fPackNibbles(v_[vN], v_[vN + 1]));
      vPb += 64;
    }
  }
}

// pack B to vnni formats in 4bits or 8 bits
void pack_B(void *RESTRICT pPackedB, const block_q4_0 *RESTRICT pB, int pKb) {
  pack_qs(pPackedB, pB, pKb);
  ggml_half *vD0 = reinterpret_cast<ggml_half *>((char *)pPackedB + TILE_N * TILE_K / 2);
  for (int vN = 0; vN < TILE_N; ++vN) {
    vD0[vN] = pB[vN * pKb].d;
  }
}

void pack_B(void *RESTRICT pPackedB, const block_q4_1 *RESTRICT pB, int pKb) {
  pack_qs(pPackedB, pB, pKb);
  ggml_half *vD0 = reinterpret_cast<ggml_half *>((char *)pPackedB + TILE_N * TILE_K / 2);
  ggml_half *vM0 = vD0 + TILE_N;
  for (int vN = 0; vN < TILE_N; ++vN) {
    vD0[vN] = pB[vN * pKb].d;
    vM0[vN] = pB[vN * pKb].m;
  }
}

inline void fS8s8Compensation(void *RESTRICT pPackedB) {
  // packed_B layout:
  //   quants {TILE_N, TILEK}  int8_t
  //   d0     {TILE_N}      ggml_half
  //   comp   {TILE_N}        int32_t
  const int cOffset = TILE_N * TILE_K + TILE_N * sizeof(ggml_half);
  __m512i vVcomp = _mm512_setzero_si512();
  const __m512i cOff = _mm512_set1_epi8(static_cast<char>(0x80));
  for (int vK = 0; vK < 8; ++vK) {
    __m512i vVb = _mm512_loadu_si512((const __m512i *)((const char *)pPackedB + vK * 64));
    vVcomp = _mm512_dpbusd_epi32(vVcomp, cOff, vVb);
  }
  _mm512_storeu_si512((__m512i *)((char *)(pPackedB) + cOffset), vVcomp);
}

void pack_B(void *RESTRICT pPackedB, const block_q8_0 *RESTRICT pB, int pKb) {
  pack_qs(pPackedB, pB, pKb);
  ggml_half *vD0 = reinterpret_cast<ggml_half *>((char *)pPackedB + TILE_N * TILE_K);
  for (int vN = 0; vN < TILE_N; ++vN) {
    vD0[vN] = pB[vN * pKb].d;
  }
  fS8s8Compensation(pPackedB);
}

// convert 8 * {min, scale} from int6 to int8
inline void fUnpackMinsAndScales(const uint8_t *pScales, uint32_t *pUtmp) {
  const uint32_t cKmask1 = 0x3f3f3f3f;
  const uint32_t cKmask2 = 0x0f0f0f0f;
  const uint32_t cKmask3 = 0x03030303;

  memcpy(pUtmp, pScales, 12);
  pUtmp[3] = ((pUtmp[2] >> 4) & cKmask2) | (((pUtmp[1] >> 6) & cKmask3) << 4);
  const uint32_t cUaux = pUtmp[1] & cKmask1;
  pUtmp[1] = (pUtmp[2] & cKmask2) | (((pUtmp[0] >> 6) & cKmask3) << 4);
  pUtmp[2] = cUaux;
  pUtmp[0] &= cKmask1;
}

// packed_B layout:
//   quants {8, TILE_N, 16}  uint8
//   scales {8, TILE_N}      uint8
//   mins   {8, TILE_N}      uint8
//   d      {TILE_N}     ggml_half
//   dmin   {TILE_N}     ggml_half
void pack_B(void *RESTRICT pPackedB, const block_q4_K *RESTRICT pB, int pKb) {
  pack_qs(pPackedB, pB, pKb);

  uint8_t *vScales = reinterpret_cast<uint8_t *>((char *)pPackedB + (QK_K / 2) * TILE_N);
  uint8_t *vMins = vScales + 8 * TILE_N;
  ggml_half *vD = reinterpret_cast<ggml_half *>(vMins + 8 * TILE_N);
  ggml_half *vDmin = vD + TILE_N;

  union {
    uint32_t aU32[4];
    uint8_t aU8[16];
  } vS;

  for (int vN = 0; vN < TILE_N; ++vN) {
    fUnpackMinsAndScales(pB[vN * pKb].scales, vS.aU32);
    for (int vK = 0; vK < 8; ++vK) {
      vScales[vK * TILE_N + vN] = vS.aU8[vK];
      vMins[(vK >> 1) * TILE_N * 2 + vN * 2 + (vK & 0x1)] = vS.aU8[vK + 8];
    }
    vD[vN] = pB[vN * pKb].d;
    vDmin[vN] = pB[vN * pKb].dmin;
  }
}

// packed_B layout:
//   quants {8, TILE_N, 16}  uint8
//   qh     {8, TILE_N,  4}  uint8
//   scales {8, TILE_N}      uint8
//   mins   {8, TILE_N}      uint8
//   d      {TILE_N}     ggml_half
//   dmin   {TILE_N}     ggml_half
void pack_B(void *RESTRICT pPackedB, const block_q5_K *RESTRICT pB, int pKb) {
  pack_qs(pPackedB, pB, pKb);

  uint8_t *vScales = reinterpret_cast<uint8_t *>((char *)pPackedB + (QK_K / 2) * TILE_N + (QK_K / 8) * TILE_N);
  uint8_t *vMins = vScales + 8 * TILE_N;
  ggml_half *vD = reinterpret_cast<ggml_half *>(vMins + 8 * TILE_N);
  ggml_half *vDmin = vD + TILE_N;

  union {
    uint32_t aU32[4];
    uint8_t aU8[16];
  } vS;

  for (int vN = 0; vN < TILE_N; ++vN) {
    fUnpackMinsAndScales(pB[vN * pKb].scales, vS.aU32);
    for (int vK = 0; vK < 8; ++vK) {
      vScales[vK * TILE_N + vN] = vS.aU8[vK];
      vMins[(vK >> 1) * TILE_N * 2 + vN * 2 + (vK & 0x1)] = vS.aU8[vK + 8];
    }
    vD[vN] = pB[vN * pKb].d;
    vDmin[vN] = pB[vN * pKb].dmin;
  }
}

// packed_B layout:
//   quants {16, TILE_N, 8}  uint8
//   qh     {16, TILE_N, 4}  uint8
//   scales {16, TILE_N}      uint8
//   d      {TILE_N}     ggml_half
void pack_B(void *RESTRICT pPackedB, const block_q6_K *RESTRICT pB, int pKb) {
  pack_qs(pPackedB, pB, pKb);

  uint8_t *vScales = reinterpret_cast<uint8_t *>((char *)pPackedB + (QK_K / 2) * TILE_N + (QK_K / 4) * TILE_N);
  ggml_half *vD = reinterpret_cast<ggml_half *>(vScales + 16 * TILE_N);
  for (int vN = 0; vN < TILE_N; ++vN) {
    const int8_t *vPs = pB[vN * pKb].scales;
    for (int vK = 0; vK < 16; ++vK) {
      vScales[vK * TILE_N + vN] = vPs[vK];
    }
    vD[vN] = pB[vN * pKb].d;
  }
}

// packed_B layout:
//   quants {8, TILE_N, 16}  uint8
//   scales {8, TILE_N}       int8
//   d      {TILE_N}     ggml_half
void pack_B(void *RESTRICT pPackedB, const block_iq4_xs *RESTRICT pB, int pKb) {
  pack_qs(pPackedB, pB, pKb);

  int8_t *vScales = reinterpret_cast<int8_t *>((char *)pPackedB + (QK_K / 2) * TILE_N);
  ggml_half *vD = reinterpret_cast<ggml_half *>(vScales + 8 * TILE_N);

  // pack the scales
  for (int vN = 0; vN < TILE_N; ++vN) {
    uint16_t vSh = pB[vN * pKb].scales_h;
    for (int vK = 0; vK < 8; vK += 2) {
      const int16_t cLs1 = ((pB[vN * pKb].scales_l[vK / 2] & 0xf) | ((vSh << 4) & 0x30)) - 32;
      const int16_t cLs2 = ((pB[vN * pKb].scales_l[vK / 2] >> 4) | ((vSh << 2) & 0x30)) - 32;
      vScales[(vK + 0) * TILE_N + vN] = cLs1;
      vScales[(vK + 1) * TILE_N + vN] = cLs2;
      vSh >>= 4;
    }
    vD[vN] = pB[vN * pKb].d;
  }
}

template <typename TB, typename packed_B_t = packed_B_type<TB>>
void unpack_B(packed_B_t *RESTRICT pTile, const void *RESTRICT pPackedB) {
  GGML_UNUSED(pTile);
  GGML_UNUSED(pPackedB);
}

template <> void unpack_B<block_q4_0>(int8_t *RESTRICT pTile, const void *RESTRICT pPackedB) {
  const __m512i cOff = _mm512_set1_epi8(8);
  const __m512i cLowMask = _mm512_set1_epi8(0xF);
  for (int vN = 0; vN < 8; vN += 2) {
    __m512i vBytes = _mm512_loadu_si512((const __m512i *)((const char *)pPackedB + vN * 32));
    const __m512i cR0 = _mm512_sub_epi8(_mm512_and_si512(vBytes, cLowMask), cOff);
    const __m512i cR1 = _mm512_sub_epi8(_mm512_and_si512(_mm512_srli_epi16(vBytes, 4), cLowMask), cOff);
    _mm512_storeu_si512((__m512i *)(pTile + vN * 64 + 0), cR0);
    _mm512_storeu_si512((__m512i *)(pTile + vN * 64 + 64), cR1);
  }
}

template <> void unpack_B<block_q4_1>(uint8_t *RESTRICT pTile, const void *RESTRICT pPackedB) {
  const __m512i cLowMask = _mm512_set1_epi8(0xF);
  for (int vN = 0; vN < 8; vN += 2) {
    __m512i vBytes = _mm512_loadu_si512((const __m512i *)((const char *)pPackedB + vN * 32));
    const __m512i cR0 = _mm512_and_si512(vBytes, cLowMask);
    const __m512i cR1 = _mm512_and_si512(_mm512_srli_epi16(vBytes, 4), cLowMask);
    _mm512_storeu_si512((__m512i *)(pTile + vN * 64 + 0), cR0);
    _mm512_storeu_si512((__m512i *)(pTile + vN * 64 + 64), cR1);
  }
}

// packed_B_t for QKK is int8_t
template <typename TB> void unpack_B(int8_t *RESTRICT pTile, const void *RESTRICT pPackedB, int pK) {
  const int cPackedBGroupSize = QK_K / 2 * TILE_N / 8;
  const char *vPackedBGroup = (const char *)pPackedB + pK * cPackedBGroupSize;
  const __m512i cLowMask = _mm512_set1_epi8(0xF);
  for (int vN = 0; vN < 8; vN += 2) {
    __m512i vBytes = _mm512_loadu_si512(vPackedBGroup + vN * 32);
    const __m512i cR0 = _mm512_and_si512(vBytes, cLowMask);
    const __m512i cR1 = _mm512_and_si512(_mm512_srli_epi16(vBytes, 4), cLowMask);
    _mm512_storeu_si512((__m512i *)(pTile + vN * 64 + 0), cR0);
    _mm512_storeu_si512((__m512i *)(pTile + vN * 64 + 64), cR1);
  }
}

template <> void unpack_B<block_q5_K>(int8_t *RESTRICT pTile, const void *RESTRICT pPackedB, int pK) {
  // lower 4bits, stride 256 bytes
  const int cPackedL4GroupSize = QK_K / 2 * TILE_N / 8;
  const char *vPb = (const char *)pPackedB + pK * cPackedL4GroupSize;

  // higher 1bit, stride 64 bytes
  const int cPackedH1GroupSize = QK_K / 8 * TILE_N / 8;
  const char *vPh = (const char *)pPackedB + (QK_K / 2) * TILE_N + pK * cPackedH1GroupSize;
  const __m512i cHbits = _mm512_loadu_si512(vPh);

  const __m512i cLowMask = _mm512_set1_epi8(0xF);
  __m512i vHmask0 = _mm512_set1_epi8(0x1);
  __m512i vHmask1 = _mm512_set1_epi8(0x2);

  for (int vN = 0; vN < 8; vN += 2) {
    __m512i vBytes = _mm512_loadu_si512(vPb + vN * 32);
    __m512i vR0 = _mm512_and_si512(vBytes, cLowMask);
    __m512i vR1 = _mm512_and_si512(_mm512_srli_epi16(vBytes, 4), cLowMask);
    __m512i vH0 = _mm512_slli_epi16(_mm512_srli_epi16(_mm512_and_si512(cHbits, vHmask0), vN), 4);
    __m512i vH1 = _mm512_slli_epi16(_mm512_srli_epi16(_mm512_and_si512(cHbits, vHmask1), vN + 1), 4);

    vHmask0 = _mm512_slli_epi16(vHmask0, 2);
    vHmask1 = _mm512_slli_epi16(vHmask1, 2);
    vR0 = _mm512_add_epi8(vR0, vH0);
    vR1 = _mm512_add_epi8(vR1, vH1);
    _mm512_storeu_si512((__m512i *)(pTile + vN * 64 + 0), vR0);
    _mm512_storeu_si512((__m512i *)(pTile + vN * 64 + 64), vR1);
  }
}

template <> void unpack_B<block_q6_K>(int8_t *RESTRICT pTile, const void *RESTRICT pPackedB, int pK) {
  // lower 4bits, stride 128 bytes
  const int cPackedL4GroupSize = QK_K / 2 * TILE_N / 16;
  const char *vPb = (const char *)pPackedB + pK * cPackedL4GroupSize;

  // higher 2bits, stride 64 bytes
  const int cPackedH2GroupSize = QK_K / 4 * TILE_N / 16;
  const char *vPh = (const char *)pPackedB + (QK_K / 2) * TILE_N + pK * cPackedH2GroupSize;
  const __m512i cHbits = _mm512_loadu_si512(vPh);

  const __m512i cOff = _mm512_set1_epi8(32);
  const __m512i cLowMask = _mm512_set1_epi8(0xF);
  __m512i vHmask0 = _mm512_set1_epi8(0x3); // 0011
  __m512i vHmask1 = _mm512_set1_epi8(0xC); // 1100

  // notes: skip zero padding from row4 to row7 as we have done so in `unpack_A`
  __m512i vBytes = _mm512_loadu_si512(vPb);
  __m512i vR0 = _mm512_and_si512(vBytes, cLowMask);
  __m512i vR1 = _mm512_and_si512(_mm512_srli_epi16(vBytes, 4), cLowMask);
  __m512i vH0 = _mm512_slli_epi16(_mm512_and_si512(cHbits, vHmask0), 4);
  __m512i vH1 = _mm512_slli_epi16(_mm512_and_si512(cHbits, vHmask1), 2);
  _mm512_storeu_si512((__m512i *)(pTile + 0), _mm512_sub_epi8(_mm512_add_epi8(vR0, vH0), cOff));
  _mm512_storeu_si512((__m512i *)(pTile + 64), _mm512_sub_epi8(_mm512_add_epi8(vR1, vH1), cOff));

  vHmask0 = _mm512_slli_epi16(vHmask0, 4);
  vHmask1 = _mm512_slli_epi16(vHmask1, 4);

  vBytes = _mm512_loadu_si512(vPb + 64);
  vR0 = _mm512_and_si512(vBytes, cLowMask);
  vR1 = _mm512_and_si512(_mm512_srli_epi16(vBytes, 4), cLowMask);
  vH0 = _mm512_and_si512(cHbits, vHmask0);
  vH1 = _mm512_srli_epi16(_mm512_and_si512(cHbits, vHmask1), 2);
  _mm512_storeu_si512((__m512i *)(pTile + 128), _mm512_sub_epi8(_mm512_add_epi8(vR0, vH0), cOff));
  _mm512_storeu_si512((__m512i *)(pTile + 192), _mm512_sub_epi8(_mm512_add_epi8(vR1, vH1), cOff));
}

template <> void unpack_B<block_iq4_xs>(int8_t *RESTRICT pTile, const void *RESTRICT pPackedB, int pK) {
  static const __m512i cValues128 = _mm512_set_epi8(
      113, 89, 69, 53, 38, 25, 13, 1, -10, -22, -35, -49, -65, -83, -104, -127, 113, 89, 69, 53, 38, 25, 13, 1, -10,
      -22, -35, -49, -65, -83, -104, -127, 113, 89, 69, 53, 38, 25, 13, 1, -10, -22, -35, -49, -65, -83, -104, -127,
      113, 89, 69, 53, 38, 25, 13, 1, -10, -22, -35, -49, -65, -83, -104, -127);

  const int cPackedBGroupSize = QK_K / 2 * TILE_N / 8;
  const char *vPb = (const char *)pPackedB + pK * cPackedBGroupSize;
  const __m512i cLowMask = _mm512_set1_epi8(0xF);

  for (int vN = 0; vN < 8; vN += 2) {
    __m512i vBytes = _mm512_loadu_si512(vPb + vN * 32);
    const __m512i cR0 = _mm512_shuffle_epi8(cValues128, _mm512_and_si512(vBytes, cLowMask));
    const __m512i cR1 = _mm512_shuffle_epi8(cValues128, _mm512_and_si512(_mm512_srli_epi16(vBytes, 4), cLowMask));
    _mm512_storeu_si512((__m512i *)(pTile + vN * 64 + 0), cR0);
    _mm512_storeu_si512((__m512i *)(pTile + vN * 64 + 64), cR1);
  }
}

template <typename TA, typename TB, bool is_acc> struct acc_C {};

template <bool is_acc> struct acc_C<block_q8_0, block_q4_0, is_acc> {
  static void apply(float *RESTRICT pC, int pLdc, const int32_t *RESTRICT pTile, const block_q8_0 *pA, int pLda,
                    const void *pPackedB, int pNr) {
    const int cOffset = TILE_N * TILE_K / 2;
    const __m512 cVd0 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)((const char *)pPackedB + cOffset)));

    for (int vM = 0; vM < pNr; ++vM) {
      const __m512 cVd1 = _mm512_set1_ps(GGML_CPU_FP16_TO_FP32(pA[vM * pLda].d));
      const __m512 cVtile = _mm512_cvtepi32_ps(_mm512_loadu_si512(pTile + vM * TILE_N));

      __m512 vVsum;
      if (is_acc) {
        vVsum = _mm512_loadu_ps(pC + vM * pLdc);
      } else {
        vVsum = _mm512_set1_ps(0.f);
      }
      vVsum = _mm512_fmadd_ps(cVtile, _mm512_mul_ps(cVd0, cVd1), vVsum);
      _mm512_storeu_ps(pC + vM * pLdc, vVsum);
    }
  }
};

template <bool is_acc> struct acc_C<block_q8_1, block_q4_1, is_acc> {
  static void apply(float *RESTRICT pC, int pLdc, const int32_t *RESTRICT pTile, const block_q8_1 *pA, int pLda,
                    const void *pPackedB, int pNr) {
    const int cOffset = TILE_N * TILE_K / 2;
    const __m512 cVd0 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)((const char *)pPackedB + cOffset)));
    const __m512 cVm0 = _mm512_cvtph_ps(
        _mm256_loadu_si256((const __m256i *)((const char *)pPackedB + cOffset + TILE_N * sizeof(ggml_half))));

    for (int vM = 0; vM < pNr; ++vM) {
      const __m512 cVd1 = _mm512_set1_ps(GGML_CPU_FP16_TO_FP32(pA[vM * pLda].d));
      const __m512 cVs1 = _mm512_set1_ps(GGML_CPU_FP16_TO_FP32(pA[vM * pLda].s));
      const __m512 cVtile = _mm512_cvtepi32_ps(_mm512_loadu_si512(pTile + vM * TILE_N));

      __m512 vVsum;
      if (is_acc) {
        vVsum = _mm512_loadu_ps(pC + vM * pLdc);
      } else {
        vVsum = _mm512_set1_ps(0.f);
      }
      vVsum = _mm512_fmadd_ps(cVtile, _mm512_mul_ps(cVd0, cVd1), vVsum);
      vVsum = _mm512_fmadd_ps(cVm0, cVs1, vVsum);
      _mm512_storeu_ps(pC + vM * pLdc, vVsum);
    }
  }
};

template <bool is_acc> struct acc_C<block_q8_0, block_q8_0, is_acc> {
  static void apply(float *RESTRICT pC, int pLdc, const int32_t *RESTRICT pTile, const block_q8_0 *pA, int pLda,
                    const void *pPackedB, int pNr) {
    const int cOffset = TILE_N * TILE_K;
    const __m512 cVd0 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)((const char *)pPackedB + cOffset)));

    for (int vM = 0; vM < pNr; ++vM) {
      const __m512 cVd1 = _mm512_set1_ps(GGML_CPU_FP16_TO_FP32(pA[vM * pLda].d));
      const __m512 cVtile = _mm512_cvtepi32_ps(_mm512_loadu_si512(pTile + vM * TILE_N));

      __m512 vVsum;
      if (is_acc) {
        vVsum = _mm512_loadu_ps(pC + vM * pLdc);
      } else {
        vVsum = _mm512_set1_ps(0.f);
      }
      vVsum = _mm512_fmadd_ps(cVtile, _mm512_mul_ps(cVd0, cVd1), vVsum);
      _mm512_storeu_ps(pC + vM * pLdc, vVsum);
    }
  }
};

template <bool is_acc> struct acc_C<block_q8_K, block_q4_K, is_acc> {
  static void apply(float *RESTRICT pC, int pLdc, const int32_t *RESTRICT pTile, const block_q8_K *pA, int pLda,
                    const void *pPackedB, int pNr) {
    const uint8_t *vScales = reinterpret_cast<const uint8_t *>((const char *)pPackedB + (QK_K / 2) * TILE_N);
    const uint8_t *vMins = vScales + 8 * TILE_N;
    const ggml_half *vD0 = reinterpret_cast<const ggml_half *>(vMins + 8 * TILE_N);
    const ggml_half *vDmin = vD0 + TILE_N;

    const __m512 cVd0 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)vD0));
    const __m512 cVdmin = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)vDmin));

    for (int vM = 0; vM < pNr; ++vM) {
      const float cD1 = pA[vM * pLda].d;
      const __m512 cVd = _mm512_mul_ps(_mm512_set1_ps(cD1), cVd0);
      const __m512 cVdm = _mm512_mul_ps(_mm512_set1_ps(-cD1), cVdmin);
      const __m512 cVtile = _mm512_cvtepi32_ps(_mm512_loadu_si512(pTile + vM * TILE_N));

      __m512 vVsum;
      if (is_acc) {
        vVsum = _mm512_loadu_ps(pC + vM * pLdc);
      } else {
        vVsum = _mm512_set1_ps(0.f);
      }

      const __m256i cQ8sums = _mm256_loadu_si256((const __m256i *)pA[vM * pLda].bsums);
      const __m128i cQ8s = _mm_hadd_epi16(_mm256_extracti128_si256(cQ8sums, 0), _mm256_extracti128_si256(cQ8sums, 1));

      __m512i vAccM = _mm512_setzero_si512();
      for (int vK = 0; vK < 4; ++vK) {
        __m512i vVmask = _mm512_set1_epi32(vK);
        __m512i vVa = _mm512_permutexvar_epi32(vVmask, _mm512_castsi128_si512(cQ8s));
        __m512i vVb = _mm512_cvtepi8_epi16(_mm256_loadu_si256((const __m256i *)(vMins + vK * 32)));
        vAccM = _mm512_dpwssds_epi32(vAccM, vVa, vVb);
      }

      vVsum = _mm512_fmadd_ps(cVtile, cVd, vVsum);
      vVsum = _mm512_fmadd_ps(_mm512_cvtepi32_ps(vAccM), cVdm, vVsum);
      _mm512_storeu_ps(pC + vM * pLdc, vVsum);
    }
  }
};

template <bool is_acc> struct acc_C<block_q8_K, block_q5_K, is_acc> {
  static void apply(float *RESTRICT pC, int pLdc, const int32_t *RESTRICT pTile, const block_q8_K *pA, int pLda,
                    const void *pPackedB, int pNr) {
    const uint8_t *vScales =
        reinterpret_cast<const uint8_t *>((const char *)pPackedB + (QK_K / 2) * TILE_N + (QK_K / 8) * TILE_N);
    const uint8_t *vMins = vScales + 8 * TILE_N;
    const ggml_half *vD0 = reinterpret_cast<const ggml_half *>(vMins + 8 * TILE_N);
    const ggml_half *vDmin = vD0 + TILE_N;

    const __m512 cVd0 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)vD0));
    const __m512 cVdmin = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)vDmin));

    for (int vM = 0; vM < pNr; ++vM) {
      const float cD1 = pA[vM * pLda].d;
      const __m512 cVd = _mm512_mul_ps(_mm512_set1_ps(cD1), cVd0);
      const __m512 cVdm = _mm512_mul_ps(_mm512_set1_ps(-cD1), cVdmin);
      const __m512 cVtile = _mm512_cvtepi32_ps(_mm512_loadu_si512(pTile + vM * TILE_N));

      __m512 vVsum;
      if (is_acc) {
        vVsum = _mm512_loadu_ps(pC + vM * pLdc);
      } else {
        vVsum = _mm512_set1_ps(0.f);
      }

      const __m256i cQ8sums = _mm256_loadu_si256((const __m256i *)pA[vM * pLda].bsums);
      const __m128i cQ8s = _mm_hadd_epi16(_mm256_extracti128_si256(cQ8sums, 0), _mm256_extracti128_si256(cQ8sums, 1));

      __m512i vAccM = _mm512_setzero_si512();
      for (int vK = 0; vK < 4; ++vK) {
        __m512i vVmask = _mm512_set1_epi32(vK);
        __m512i vVa = _mm512_permutexvar_epi32(vVmask, _mm512_castsi128_si512(cQ8s));
        __m512i vVb = _mm512_cvtepi8_epi16(_mm256_loadu_si256((const __m256i *)(vMins + vK * 32)));
        vAccM = _mm512_dpwssds_epi32(vAccM, vVa, vVb);
      }

      vVsum = _mm512_fmadd_ps(cVtile, cVd, vVsum);
      vVsum = _mm512_fmadd_ps(_mm512_cvtepi32_ps(vAccM), cVdm, vVsum);
      _mm512_storeu_ps(pC + vM * pLdc, vVsum);
    }
  }
};

template <bool is_acc> struct acc_C<block_q8_K, block_q6_K, is_acc> {
  static void apply(float *RESTRICT pC, int pLdc, const int32_t *RESTRICT pTile, const block_q8_K *pA, int pLda,
                    const void *pPackedB, int pNr) {
    const uint8_t *vScales =
        reinterpret_cast<const uint8_t *>((const char *)pPackedB + (QK_K / 2) * TILE_N + (QK_K / 4) * TILE_N);
    const ggml_half *vD0 = reinterpret_cast<const ggml_half *>(vScales + 16 * TILE_N);

    const __m512 cVd0 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)vD0));

    for (int vM = 0; vM < pNr; ++vM) {
      const float cD1 = pA[vM * pLda].d;
      const __m512 cVd = _mm512_mul_ps(_mm512_set1_ps(cD1), cVd0);
      const __m512 cVtile = _mm512_cvtepi32_ps(_mm512_loadu_si512(pTile + vM * TILE_N));

      __m512 vVsum;
      if (is_acc) {
        vVsum = _mm512_loadu_ps(pC + vM * pLdc);
      } else {
        vVsum = _mm512_set1_ps(0.f);
      }

      vVsum = _mm512_fmadd_ps(cVtile, cVd, vVsum);
      _mm512_storeu_ps(pC + vM * pLdc, vVsum);
    }
  }
};

template <bool is_acc> struct acc_C<block_q8_K, block_iq4_xs, is_acc> {
  static void apply(float *RESTRICT pC, int pLdc, const int32_t *RESTRICT pTile, const block_q8_K *pA, int pLda,
                    const void *pPackedB, int pNr) {
    const int8_t *vScales = reinterpret_cast<const int8_t *>((const char *)pPackedB + (QK_K / 2) * TILE_N);
    const ggml_half *vD0 = reinterpret_cast<const ggml_half *>(vScales + 8 * TILE_N);

    const __m512 cVd0 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)vD0));

    for (int vM = 0; vM < pNr; ++vM) {
      const float cD1 = pA[vM * pLda].d;
      const __m512 cVd = _mm512_mul_ps(_mm512_set1_ps(cD1), cVd0);
      const __m512 cVtile = _mm512_cvtepi32_ps(_mm512_loadu_si512(pTile + vM * TILE_N));

      __m512 vVsum;
      if (is_acc) {
        vVsum = _mm512_loadu_ps(pC + vM * pLdc);
      } else {
        vVsum = _mm512_set1_ps(0.f);
      }

      vVsum = _mm512_fmadd_ps(cVtile, cVd, vVsum);
      _mm512_storeu_ps(pC + vM * pLdc, vVsum);
    }
  }
};

template <typename TB> constexpr int get_quants_size();
template <> constexpr int get_quants_size<block_q4_K>() { return (QK_K / 2) * TILE_N; }
template <> constexpr int get_quants_size<block_q5_K>() { return (QK_K / 2) * TILE_N + (QK_K / 8) * TILE_N; }
template <> constexpr int get_quants_size<block_q6_K>() { return (QK_K / 2) * TILE_N + (QK_K / 4) * TILE_N; }
template <> constexpr int get_quants_size<block_iq4_xs>() { return (QK_K / 2) * TILE_N; }

// used for QKK format
template <typename TB, bool is_acc, typename std::enable_if<is_type_qkk<TB>::value, int>::type = 0>
inline void fScaleC(const int32_t *RESTRICT pTile, int32_t *RESTRICT pSumi, const void *pPackedB, int pK, int pNr) {
  const uint8_t *vScales = reinterpret_cast<const uint8_t *>((const char *)pPackedB + get_quants_size<TB>());
  const __m512i cVscale = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i *)(vScales + pK * TILE_N)));

  for (int vM = 0; vM < pNr; ++vM) {
    __m512i vVsumi;
    if (is_acc) {
      vVsumi = _mm512_loadu_si512(pSumi + vM * TILE_N);
    } else {
      vVsumi = _mm512_setzero_si512();
    }
    __m512i vVtile = _mm512_loadu_si512(pTile + vM * TILE_N);
    vVsumi = _mm512_add_epi32(vVsumi, _mm512_mullo_epi32(vVtile, cVscale));
    _mm512_storeu_si512((__m512i *)(pSumi + vM * TILE_N), vVsumi);
  }
}

template <typename TA, typename TB, typename TC, int BLOCK_M, int BLOCK_N, int BLOCK_K> struct tinygemm_kernel_avx {
  static void apply(int pK, const TA *RESTRICT pA, const TB *RESTRICT pB, TC *RESTRICT pC, int pLdc) {
    GGML_UNUSED(pK);
    GGML_UNUSED(pA);
    GGML_UNUSED(pB);
    GGML_UNUSED(pC);
    GGML_UNUSED(pLdc);
  }
};

template <int BLOCK_M, int BLOCK_N, int BLOCK_K>
struct tinygemm_kernel_avx<float, ggml_fp16_t, float, BLOCK_M, BLOCK_N, BLOCK_K> {
  static void apply(int K, const float *RESTRICT pA, const ggml_fp16_t *RESTRICT pB, float *RESTRICT pC, int pLdc) {
    constexpr int cRows = BLOCK_M;
    constexpr int cCols = BLOCK_N;
    assert(BLOCK_K == 16);

    __m512 vVa;
    __m512 aVb[cCols];
    __m512 aVc[cRows * cCols];

    auto vLoadc = [&](auto pIdx) { aVc[pIdx] = _mm512_setzero_ps(); };
    Unroll<cRows * cCols>{}(vLoadc);

    auto vCompute = [&](auto pIdx, auto k) {
      constexpr int cRow = pIdx / cCols;
      constexpr int cCol = pIdx % cCols;

      if constexpr (cCol == 0) {
        vVa = _mm512_loadu_ps(pA + cRow * K + k);
      }
      if constexpr (cRow == 0) {
        aVb[cCol] = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)(pB + cCol * K + k)));
      }
      aVc[pIdx] = _mm512_fmadd_ps(vVa, aVb[cCol], aVc[pIdx]);
    };

    for (int vK = 0; vK < K; vK += 16) {
      Unroll<cRows * cCols>{}(vCompute, vK);
    }

    auto vStorec = [&](auto pIdx) {
      constexpr int cRow = pIdx / cCols;
      constexpr int cCol = pIdx % cCols;
      pC[cRow * pLdc + cCol] = _mm512_reduce_add_ps(aVc[pIdx]);
    };
    Unroll<cRows * cCols>{}(vStorec);
  }
};

#define LAUNCH_TINYGEMM_KERNEL_AVX(MB_SIZE, NB_SIZE)                                                                   \
  tinygemm_kernel_avx<float, type, float, MB_SIZE, NB_SIZE, blck_size>::apply(                                         \
      cK, (const float *)vSrc1->data + vSrc1Offset + vMbStart * cK,                                                    \
      (const type *)vSrc0->data + vSrc0Offset + vNbStart * cK,                                                         \
      (float *)pDst->data + vDstOffset + vMbStart * cLdc + vNbStart, cLdc)

// re-organize in the format {NB, KB, TILE_SIZE}:
#define PACKED_INDEX(n, k, KB, tile_size) (n * KB + k) * tile_size

template <typename TB, int BLOCK_K>
void fConvertBPackedFormat(void *RESTRICT pPackedB, const TB *RESTRICT pB, int pN, int pK) {
  const int cNb = pN / TILE_N;
  const int cKb = pK / BLOCK_K;
  const int cTileSize = fGetTileSize<TB>();

  // parallel on NB should be enough
  parallel_for(cNb, [&](int pBegin, int pEnd) {
    for (int vN = pBegin; vN < pEnd; ++vN) {
      for (int vK = 0; vK < cKb; ++vK) {
        int vN0 = vN * TILE_N;
        pack_B((char *)pPackedB + PACKED_INDEX(vN, vK, cKb, cTileSize), &pB[vN0 * cKb + vK], cKb);
      }
    }
  });
}

template <typename TA, typename TB, typename TC, int BLOCK_M, int BLOCK_N, int BLOCK_K> struct tinygemm_kernel_vnni {};

template <int BLOCK_M, int BLOCK_N, int BLOCK_K>
struct tinygemm_kernel_vnni<block_q8_0, block_q4_0, float, BLOCK_M, BLOCK_N, BLOCK_K> {
  static void apply(int pKb, const void *RESTRICT _A, const void *RESTRICT _B, float *RESTRICT pC, int pLdc) {

    constexpr int cCols = BLOCK_N / 16;
    const int cTileSize = TILE_N * sizeof(block_q4_0);

    const block_q8_0 *RESTRICT vA = static_cast<const block_q8_0 *>(_A);
    const char *RESTRICT vB = static_cast<const char *>(_B);

    __m512i aVa[8];
    __m512 aVc[cCols];
    __m512 vVd1;

    // sum of offsets, shared across COLS
    //
    // avx512-vnni does not have `_mm512_dpbssd_epi32`,
    // need to transform ss to us:
    //   a * (b - 8) is equivalent to b * a - 8 * a
    //   s    u   u                   u   s   u   s
    //
    __m512i vVcomp;

    const __m512i cOff = _mm512_set1_epi8(8);
    const __m512i cLowMask = _mm512_set1_epi8(0xF);

    auto vLoadc = [&](auto pCol) { aVc[pCol] = _mm512_setzero_ps(); };
    Unroll<cCols>{}(vLoadc);

    auto vCompute = [&](auto pCol, auto pI) {
      // load a and compute compensation
      if constexpr (pCol == 0) {
        const int32_t *vAPtr = reinterpret_cast<const int32_t *>(vA[0 * pKb + pI].qs);
        vVcomp = _mm512_setzero_si512();
        for (int vK = 0; vK < 8; ++vK) {
          aVa[vK] = _mm512_set1_epi32(vAPtr[vK]);
          vVcomp = _mm512_dpbusd_epi32(vVcomp, cOff, aVa[vK]);
        }
        vVd1 = _mm512_set1_ps(GGML_CPU_FP16_TO_FP32(vA[0 * pKb + pI].d));
      }

      // load b
      __m512i vVsum = _mm512_setzero_si512();
      const char *vBPtr = vB + PACKED_INDEX(pCol, pI, pKb, cTileSize);
      for (int vK = 0; vK < 8; vK += 2) {
        __m512i vBytes = _mm512_loadu_si512((const __m512i *)(vBPtr + vK * 32));
        __m512i vVb0 = _mm512_and_si512(vBytes, cLowMask);
        vVsum = _mm512_dpbusd_epi32(vVsum, vVb0, aVa[vK + 0]);
        __m512i vVb1 = _mm512_and_si512(_mm512_srli_epi16(vBytes, 4), cLowMask);
        vVsum = _mm512_dpbusd_epi32(vVsum, vVb1, aVa[vK + 1]);
      }
      const int cOffset = TILE_N * TILE_K / 2;
      const __m512 cVd0 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)(vBPtr + cOffset)));
      vVsum = _mm512_sub_epi32(vVsum, vVcomp);

      aVc[pCol] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(vVsum), _mm512_mul_ps(cVd0, vVd1), aVc[pCol]);
    };

    for (int vI = 0; vI < pKb; ++vI) {
      Unroll<cCols>{}(vCompute, vI);
    }

    // store to C
    auto vStorec = [&](auto pCol) { _mm512_storeu_ps((__m512i *)(pC + 0 * pLdc + pCol * 16), aVc[pCol]); };
    Unroll<cCols>{}(vStorec);
  }
};

template <int BLOCK_N, int BLOCK_K> struct tinygemm_kernel_vnni<block_q8_1, block_q4_1, float, 1, BLOCK_N, BLOCK_K> {
  static void apply(int pKb, const void *RESTRICT _A, const void *RESTRICT _B, float *RESTRICT pC, int pLdc) {

    constexpr int cCols = BLOCK_N / 16;
    const int cTileSize = TILE_N * sizeof(block_q4_1);

    const block_q8_1 *RESTRICT vA = static_cast<const block_q8_1 *>(_A);
    const char *RESTRICT vB = static_cast<const char *>(_B);

    __m512i aVa[8];
    __m512i aVb[8];
    __m512 aVc[cCols];
    __m512 vVd1, vVs1;

    const __m512i cLowMask = _mm512_set1_epi8(0xF);

    auto vLoadc = [&](auto pCol) { aVc[pCol] = _mm512_setzero_ps(); };
    Unroll<cCols>{}(vLoadc);

    auto vCompute = [&](auto pCol, auto pI) {
      // load a
      if constexpr (pCol == 0) {
        const int32_t *vAPtr = reinterpret_cast<const int32_t *>(vA[0 * pKb + pI].qs);
        for (int vK = 0; vK < 8; ++vK) {
          aVa[vK] = _mm512_set1_epi32(vAPtr[vK]);
        }
        vVd1 = _mm512_set1_ps(GGML_CPU_FP16_TO_FP32(vA[0 * pKb + pI].d));
        vVs1 = _mm512_set1_ps(GGML_CPU_FP16_TO_FP32(vA[0 * pKb + pI].s));
      }

      // load b
      const char *vBPtr = vB + PACKED_INDEX(pCol, pI, pKb, cTileSize);
      for (int vK = 0; vK < 8; vK += 2) {
        __m512i vBytes = _mm512_loadu_si512((const __m512i *)(vBPtr + vK * 32));
        aVb[vK + 0] = _mm512_and_si512(vBytes, cLowMask);
        aVb[vK + 1] = _mm512_and_si512(_mm512_srli_epi16(vBytes, 4), cLowMask);
      }
      const int cOffset = TILE_N * TILE_K / 2;
      const __m512 cVd0 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)(vBPtr + cOffset)));
      const __m512 cVm0 =
          _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)(vBPtr + cOffset + TILE_N * sizeof(ggml_half))));

      __m512i vVsum = _mm512_setzero_si512();
      for (int vK = 0; vK < 8; ++vK) {
        vVsum = _mm512_dpbusd_epi32(vVsum, aVb[vK], aVa[vK]);
      }

      aVc[pCol] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(vVsum), _mm512_mul_ps(cVd0, vVd1), aVc[pCol]);
      aVc[pCol] = _mm512_fmadd_ps(cVm0, vVs1, aVc[pCol]);
    };

    for (int vI = 0; vI < pKb; ++vI) {
      Unroll<cCols>{}(vCompute, vI);
    }

    // store to C
    auto vStorec = [&](auto pCol) { _mm512_storeu_ps((__m512i *)(pC + 0 * pLdc + pCol * 16), aVc[pCol]); };
    Unroll<cCols>{}(vStorec);
  }
};

template <int BLOCK_M, int BLOCK_N, int BLOCK_K>
struct tinygemm_kernel_vnni<block_q8_0, block_q8_0, float, BLOCK_M, BLOCK_N, BLOCK_K> {
  static void apply(int pKb, const void *RESTRICT _A, const void *RESTRICT _B, float *RESTRICT pC, int pLdc) {

    constexpr int cCols = BLOCK_N / 16;
    const int cTileSize = TILE_N * sizeof(block_q8_0) + TILE_N * sizeof(int32_t);

    const block_q8_0 *RESTRICT vA = static_cast<const block_q8_0 *>(_A);
    const char *RESTRICT vB = static_cast<const char *>(_B);

    __m512i aVa[8];
    __m512i aVb[8];
    __m512 aVc[cCols];
    __m512 vVd1;

    // Notes: s8s8 igemm compensation in avx512-vnni
    // change s8s8 to u8s8 with compensate
    //   a * b = (a + 128) * b - 128 * b
    //   s   s       u       s    u    s
    //
    // (128 * b is pre-computed when packing B to vnni formats)
    //
    const __m512i cOff = _mm512_set1_epi8(static_cast<char>(0x80));

    auto vLoadc = [&](auto pCol) { aVc[pCol] = _mm512_setzero_ps(); };
    Unroll<cCols>{}(vLoadc);

    auto vCompute = [&](auto pCol, auto pI) {
      // load a and add offset 128
      if constexpr (pCol == 0) {
        const int32_t *vAPtr = reinterpret_cast<const int32_t *>(vA[0 * pKb + pI].qs);
        for (int vK = 0; vK < 8; ++vK) {
          aVa[vK] = _mm512_set1_epi32(vAPtr[vK]);
          aVa[vK] = _mm512_add_epi8(aVa[vK], cOff);
        }
        vVd1 = _mm512_set1_ps(GGML_CPU_FP16_TO_FP32(vA[0 * pKb + pI].d));
      }

      // load b
      const char *vBPtr = vB + PACKED_INDEX(pCol, pI, pKb, cTileSize);
      for (int vK = 0; vK < 8; ++vK) {
        aVb[vK] = _mm512_loadu_si512((const __m512i *)(vBPtr + vK * 64));
      }
      const int cOffset = TILE_N * TILE_K;
      const __m512 cVd0 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)(vBPtr + cOffset)));
      const int cOffset2 = TILE_N * TILE_K + TILE_N * sizeof(ggml_half);
      const __m512i cVcomp = _mm512_loadu_si512((const __m512i *)(vBPtr + cOffset2));

      __m512i vVsum = _mm512_setzero_si512();
      for (int vK = 0; vK < 8; ++vK) {
        vVsum = _mm512_dpbusd_epi32(vVsum, aVa[vK], aVb[vK]);
      }
      vVsum = _mm512_sub_epi32(vVsum, cVcomp);

      aVc[pCol] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(vVsum), _mm512_mul_ps(cVd0, vVd1), aVc[pCol]);
    };

    for (int vI = 0; vI < pKb; ++vI) {
      Unroll<cCols>{}(vCompute, vI);
    }

    // store to C
    auto vStorec = [&](auto pCol) { _mm512_storeu_ps((__m512i *)(pC + 0 * pLdc + pCol * 16), aVc[pCol]); };
    Unroll<cCols>{}(vStorec);
  }
};

template <int BLOCK_M, int BLOCK_N, int BLOCK_K>
struct tinygemm_kernel_vnni<block_q8_K, block_q4_K, float, BLOCK_M, BLOCK_N, BLOCK_K> {
  static void apply(int pKb, const void *RESTRICT _A, const void *RESTRICT _B, float *RESTRICT pC, int pLdc) {

    constexpr int cCols = BLOCK_N / 16;
    const int cTileSize = TILE_N * sizeof(block_q4_K) + TILE_N * 4;

    const block_q8_K *RESTRICT vA = static_cast<const block_q8_K *>(_A);
    const char *RESTRICT vB = static_cast<const char *>(_B);

    // a.qs:   8 groups, 32 bytes each group (m256i)
    __m512i aVa[8];
    // a.bsum: 8 groups,  2 bytes each group (m128i)
    __m512i vVaBsum;
    __m512 aVc[cCols];
    __m512 vVd1;

    // packed_B:
    const int cOffsetScales = (QK_K / 2) * TILE_N;
    const int cOffsetMins = (QK_K / 2) * TILE_N + 8 * TILE_N;
    const int cOffsetD0 = (QK_K / 2) * TILE_N + 16 * TILE_N;
    const int cOffsetDmin = (QK_K / 2) * TILE_N + 16 * TILE_N + TILE_N * sizeof(ggml_half);

    const __m512i cLowMask = _mm512_set1_epi8(0xF);

    auto vLoadc = [&](auto pCol) { aVc[pCol] = _mm512_setzero_ps(); };
    Unroll<cCols>{}(vLoadc);

    // Notes: vnni formats in QK_K
    //   a) quants vnni format
    //     int8  {k/4, n, 4}, viewed as 2d {k/4, 4n}, k = 32
    //     from {16, 32} to {8, 64}
    //
    //   b) min vnni format
    //     int16 {k/2, n, 2}, viewed as 2d {k/2, 2n}, k = 8
    //     from {16,  8} to {4, 32}
    //
    auto vCompute = [&](auto pCol, auto pI) {
      // load a
      if constexpr (pCol == 0) {
        for (int vKGroup = 0; vKGroup < QK_K / 32; ++vKGroup) {
          aVa[vKGroup] =
              _mm512_castsi256_si512(_mm256_loadu_si256((const __m256i *)(vA[0 * pKb + pI].qs + vKGroup * 32)));
        }
        const __m256i cQ8sums = _mm256_loadu_si256((const __m256i *)vA[0 * pKb + pI].bsums);
        const __m128i cQ8s = _mm_hadd_epi16(_mm256_extracti128_si256(cQ8sums, 0), _mm256_extracti128_si256(cQ8sums, 1));
        vVaBsum = _mm512_castsi128_si512(cQ8s);
        vVd1 = _mm512_set1_ps(vA[0 * pKb + pI].d);
      }

      // step 1: accumultate the quants
      __m512i vAcc = _mm512_setzero_si512();
      const char *vBPtr = vB + PACKED_INDEX(pCol, pI, pKb, cTileSize);
      const char *vBQs = vBPtr;
      for (int vKGroup = 0; vKGroup < QK_K / 32; ++vKGroup) {
        __m512i vVsum = _mm512_setzero_si512();
        for (int vK = 0; vK < 8; vK += 2) {
          __m512i vVa0 = _mm512_permutexvar_epi32(_mm512_set1_epi32(vK + 0), aVa[vKGroup]);
          __m512i vVa1 = _mm512_permutexvar_epi32(_mm512_set1_epi32(vK + 1), aVa[vKGroup]);

          __m512i vBytes = _mm512_loadu_si512((const __m512i *)vBQs);
          __m512i vVb0 = _mm512_and_si512(vBytes, cLowMask);
          vVsum = _mm512_dpbusd_epi32(vVsum, vVb0, vVa0);
          __m512i vVb1 = _mm512_and_si512(_mm512_srli_epi16(vBytes, 4), cLowMask);
          vVsum = _mm512_dpbusd_epi32(vVsum, vVb1, vVa1);

          vBQs += 64;
        }
        // vacc += scale * (q8 @ q4)
        const __m512i cVscale =
            _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i *)(vBPtr + cOffsetScales + vKGroup * TILE_N)));
        vAcc = _mm512_add_epi32(vAcc, _mm512_mullo_epi32(vVsum, cVscale));
      }
      const __m512 cVd0 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)(vBPtr + cOffsetD0)));
      aVc[pCol] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(vAcc), _mm512_mul_ps(cVd0, vVd1), aVc[pCol]);

      // step 2: accumulate the mins
      __m512i vAccM = _mm512_setzero_si512();
      for (int vK = 0; vK < 4; ++vK) {
        __m512i vVmask = _mm512_set1_epi32(vK);
        __m512i vVa = _mm512_permutexvar_epi32(vVmask, vVaBsum);
        __m512i vVb = _mm512_cvtepi8_epi16(_mm256_loadu_si256((const __m256i *)(vBPtr + cOffsetMins + vK * 32)));
        vAccM = _mm512_dpwssds_epi32(vAccM, vVa, vVb);
      }
      const __m512 cVdmin = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)(vBPtr + cOffsetDmin)));
      aVc[pCol] = _mm512_fnmadd_ps(_mm512_cvtepi32_ps(vAccM), _mm512_mul_ps(cVdmin, vVd1), aVc[pCol]);
    };

    for (int vI = 0; vI < pKb; ++vI) {
      Unroll<cCols>{}(vCompute, vI);
    }

    // store to C
    auto vStorec = [&](auto pCol) { _mm512_storeu_ps((__m512i *)(pC + 0 * pLdc + pCol * 16), aVc[pCol]); };
    Unroll<cCols>{}(vStorec);
  }
};

template <int BLOCK_M, int BLOCK_N, int BLOCK_K>
struct tinygemm_kernel_vnni<block_q8_K, block_q5_K, float, BLOCK_M, BLOCK_N, BLOCK_K> {
  static void apply(int pKb, const void *RESTRICT _A, const void *RESTRICT _B, float *RESTRICT pC, int pLdc) {

    constexpr int cCols = BLOCK_N / 16;
    const int cTileSize = TILE_N * sizeof(block_q5_K) + TILE_N * 4;

    const block_q8_K *RESTRICT vA = static_cast<const block_q8_K *>(_A);
    const char *RESTRICT vB = static_cast<const char *>(_B);

    // a.qs:   8 groups, 32 bytes each group (m256i)
    __m512i aVa[8];
    // a.bsum: 8 groups,  2 bytes each group (m128i)
    __m512i vVaBsum;
    __m512 aVc[cCols];
    __m512 vVd1;

    // packed_B:
    const int cOffsetQh = (QK_K / 2) * TILE_N;
    const int cOffsetScales = (QK_K / 2) * TILE_N + (QK_K / 8) * TILE_N;
    const int cOffsetMins = (QK_K / 2) * TILE_N + (QK_K / 8) * TILE_N + 8 * TILE_N;
    const int cOffsetD0 = (QK_K / 2) * TILE_N + (QK_K / 8) * TILE_N + 16 * TILE_N;
    const int cOffsetDmin = (QK_K / 2) * TILE_N + (QK_K / 8) * TILE_N + 16 * TILE_N + TILE_N * sizeof(ggml_half);

    const __m512i cLowMask = _mm512_set1_epi8(0xF);

    auto vLoadc = [&](auto pCol) { aVc[pCol] = _mm512_setzero_ps(); };
    Unroll<cCols>{}(vLoadc);

    // Q5_K and Q4_K shares the same vnni formats, refer to notes above.
    auto vCompute = [&](auto pCol, auto pI) {
      // load a
      if constexpr (pCol == 0) {
        for (int vKGroup = 0; vKGroup < QK_K / 32; ++vKGroup) {
          aVa[vKGroup] =
              _mm512_castsi256_si512(_mm256_loadu_si256((const __m256i *)(vA[0 * pKb + pI].qs + vKGroup * 32)));
        }
        const __m256i cQ8sums = _mm256_loadu_si256((const __m256i *)vA[0 * pKb + pI].bsums);
        const __m128i cQ8s = _mm_hadd_epi16(_mm256_extracti128_si256(cQ8sums, 0), _mm256_extracti128_si256(cQ8sums, 1));
        vVaBsum = _mm512_castsi128_si512(cQ8s);
        vVd1 = _mm512_set1_ps(vA[0 * pKb + pI].d);
      }

      // step 1: accumultate the quants
      __m512i vAcc = _mm512_setzero_si512();
      const char *vBPtr = vB + PACKED_INDEX(pCol, pI, pKb, cTileSize);
      const char *vBQs = vBPtr;
      const char *vBQh = vBPtr + cOffsetQh;
      for (int vKGroup = 0; vKGroup < QK_K / 32; ++vKGroup) {
        __m512i vVsum = _mm512_setzero_si512();
        __m512i vHmask0 = _mm512_set1_epi8(0x1);
        __m512i vHmask1 = _mm512_set1_epi8(0x2);
        __m512i vHbits = _mm512_loadu_si512((const __m512i *)(vBQh + vKGroup * 64));
        for (int vK = 0; vK < 8; vK += 2) {
          __m512i vVa0 = _mm512_permutexvar_epi32(_mm512_set1_epi32(vK + 0), aVa[vKGroup]);
          __m512i vVa1 = _mm512_permutexvar_epi32(_mm512_set1_epi32(vK + 1), aVa[vKGroup]);

          __m512i vBytes = _mm512_loadu_si512((const __m512i *)vBQs);
          __m512i vVb0 = _mm512_and_si512(vBytes, cLowMask);
          __m512i vVb1 = _mm512_and_si512(_mm512_srli_epi16(vBytes, 4), cLowMask);

          __m512i vVh0 = _mm512_slli_epi16(_mm512_srli_epi16(_mm512_and_si512(vHbits, vHmask0), vK), 4);
          __m512i vVh1 = _mm512_slli_epi16(_mm512_srli_epi16(_mm512_and_si512(vHbits, vHmask1), vK + 1), 4);

          vHmask0 = _mm512_slli_epi16(vHmask0, 2);
          vHmask1 = _mm512_slli_epi16(vHmask1, 2);
          vVb0 = _mm512_add_epi8(vVb0, vVh0);
          vVb1 = _mm512_add_epi8(vVb1, vVh1);

          vVsum = _mm512_dpbusd_epi32(vVsum, vVb0, vVa0);
          vVsum = _mm512_dpbusd_epi32(vVsum, vVb1, vVa1);

          vBQs += 64;
        }
        // vacc += scale * (q8 @ q5)
        const __m512i cVscale =
            _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i *)(vBPtr + cOffsetScales + vKGroup * TILE_N)));
        vAcc = _mm512_add_epi32(vAcc, _mm512_mullo_epi32(vVsum, cVscale));
      }
      const __m512 cVd0 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)(vBPtr + cOffsetD0)));
      aVc[pCol] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(vAcc), _mm512_mul_ps(cVd0, vVd1), aVc[pCol]);

      // step 2: accumulate the mins
      __m512i vAccM = _mm512_setzero_si512();
      for (int vK = 0; vK < 4; ++vK) {
        __m512i vVmask = _mm512_set1_epi32(vK);
        __m512i vVa = _mm512_permutexvar_epi32(vVmask, vVaBsum);
        __m512i vVb = _mm512_cvtepi8_epi16(_mm256_loadu_si256((const __m256i *)(vBPtr + cOffsetMins + vK * 32)));
        vAccM = _mm512_dpwssds_epi32(vAccM, vVa, vVb);
      }
      const __m512 cVdmin = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)(vBPtr + cOffsetDmin)));
      aVc[pCol] = _mm512_fnmadd_ps(_mm512_cvtepi32_ps(vAccM), _mm512_mul_ps(cVdmin, vVd1), aVc[pCol]);
    };

    for (int vI = 0; vI < pKb; ++vI) {
      Unroll<cCols>{}(vCompute, vI);
    }

    // store to C
    auto vStorec = [&](auto pCol) { _mm512_storeu_ps((__m512i *)(pC + 0 * pLdc + pCol * 16), aVc[pCol]); };
    Unroll<cCols>{}(vStorec);
  }
};

template <int BLOCK_M, int BLOCK_N, int BLOCK_K>
struct tinygemm_kernel_vnni<block_q8_K, block_q6_K, float, BLOCK_M, BLOCK_N, BLOCK_K> {
  static void apply(int pKb, const void *RESTRICT _A, const void *RESTRICT _B, float *RESTRICT pC, int pLdc) {

    constexpr int cCols = BLOCK_N / 16;
    const int cTileSize = TILE_N * sizeof(block_q6_K);

    const block_q8_K *RESTRICT vA = static_cast<const block_q8_K *>(_A);
    const char *RESTRICT vB = static_cast<const char *>(_B);

    // load the 256 bytes from A to 4 avx512 vectors
    __m512i aVa[4];
    __m512 aVc[cCols];
    __m512 vVd1;

    // packed_B:
    const int cOffsetQh = (QK_K / 2) * TILE_N;
    const int cOffsetScales = (QK_K / 2) * TILE_N + (QK_K / 4) * TILE_N;
    const int cOffsetD0 = (QK_K / 2) * TILE_N + (QK_K / 4) * TILE_N + 16 * TILE_N;

    // compensation
    __m512i vVcomp;

    const __m512i cM32s = _mm512_set1_epi32(32);
    const __m512i cLowMask = _mm512_set1_epi8(0xF);

    auto vLoadc = [&](auto pCol) { aVc[pCol] = _mm512_setzero_ps(); };
    Unroll<cCols>{}(vLoadc);

    auto vCompute = [&](auto pCol, auto pI) {
      if constexpr (pCol == 0) {
        // load a
        aVa[0] = _mm512_loadu_si512((const __m512i *)(vA[0 * pKb + pI].qs + 0));
        aVa[1] = _mm512_loadu_si512((const __m512i *)(vA[0 * pKb + pI].qs + 64));
        aVa[2] = _mm512_loadu_si512((const __m512i *)(vA[0 * pKb + pI].qs + 128));
        aVa[3] = _mm512_loadu_si512((const __m512i *)(vA[0 * pKb + pI].qs + 192));

        const __m256i cQ8sums = _mm256_loadu_si256((const __m256i *)vA[0 * pKb + pI].bsums);
        vVcomp = _mm512_mullo_epi32(_mm512_cvtepi16_epi32(cQ8sums), cM32s);
        vVd1 = _mm512_set1_ps(vA[0 * pKb + pI].d);
      }

      // accmulate the quants
      __m512i vAcc = _mm512_setzero_si512();
      const char *vBPtr = vB + PACKED_INDEX(pCol, pI, pKb, cTileSize);
      const char *vBQs = vBPtr;
      const char *vBQh = vBPtr + cOffsetQh;
      int vMask = 0;
      for (int vKGroup = 0; vKGroup < QK_K / 16; ++vKGroup) {
        int vR = vKGroup >> 2;
        __m512i vVa0 = _mm512_permutexvar_epi32(_mm512_set1_epi32(vMask++), aVa[vR]);
        __m512i vVa1 = _mm512_permutexvar_epi32(_mm512_set1_epi32(vMask++), aVa[vR]);

        __m512i vVsum = _mm512_setzero_si512();
        __m512i vHmask = _mm512_set1_epi8(0x3);

        __m512i vBytes = _mm512_loadu_si512(vBQs);
        __m512i vHbits = _mm512_loadu_si512(vBQh);
        __m512i vVb0 = _mm512_and_si512(vBytes, cLowMask);
        __m512i vVb1 = _mm512_and_si512(_mm512_srli_epi16(vBytes, 4), cLowMask);
        __m512i vVh0 = _mm512_slli_epi16(_mm512_and_si512(vHbits, vHmask), 4);
        __m512i vVh1 = _mm512_slli_epi16(_mm512_and_si512(vHbits, _mm512_slli_epi16(vHmask, 2)), 2);

        vVb0 = _mm512_add_epi8(vVb0, vVh0);
        vVb1 = _mm512_add_epi8(vVb1, vVh1);
        vVsum = _mm512_dpbusd_epi32(vVsum, vVb0, vVa0);
        vVsum = _mm512_dpbusd_epi32(vVsum, vVb1, vVa1);
        vBQs += 64;

        vVa0 = _mm512_permutexvar_epi32(_mm512_set1_epi32(vMask++), aVa[vR]);
        vVa1 = _mm512_permutexvar_epi32(_mm512_set1_epi32(vMask++), aVa[vR]);

        vBytes = _mm512_loadu_si512(vBQs);
        vVb0 = _mm512_and_si512(vBytes, cLowMask);
        vVb1 = _mm512_and_si512(_mm512_srli_epi16(vBytes, 4), cLowMask);
        vVh0 = _mm512_and_si512(vHbits, _mm512_slli_epi16(vHmask, 4));
        vVh1 = _mm512_srli_epi16(_mm512_and_si512(vHbits, _mm512_slli_epi16(vHmask, 6)), 2);
        vVb0 = _mm512_add_epi8(vVb0, vVh0);
        vVb1 = _mm512_add_epi8(vVb1, vVh1);
        vVsum = _mm512_dpbusd_epi32(vVsum, vVb0, vVa0);
        vVsum = _mm512_dpbusd_epi32(vVsum, vVb1, vVa1);
        vBQs += 64;
        vBQh += 64;

        // B * A - 32 * A
        __m512i vVmask = _mm512_set1_epi32(vKGroup);
        vVsum = _mm512_sub_epi32(vVsum, _mm512_permutexvar_epi32(vVmask, vVcomp));

        // vacc += scale * (q8 @ q6)
        const __m512i cVscale =
            _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i *)(vBPtr + cOffsetScales + vKGroup * TILE_N)));
        vAcc = _mm512_add_epi32(vAcc, _mm512_mullo_epi32(vVsum, cVscale));
      }
      const __m512 cVd0 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)(vBPtr + cOffsetD0)));
      aVc[pCol] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(vAcc), _mm512_mul_ps(cVd0, vVd1), aVc[pCol]);
    };

    for (int vI = 0; vI < pKb; ++vI) {
      Unroll<cCols>{}(vCompute, vI);
    }

    // store to C
    auto vStorec = [&](int pCol) { _mm512_storeu_ps((__m512i *)(pC + 0 * pLdc + pCol * 16), aVc[pCol]); };
    Unroll<cCols>{}(vStorec);
  }
};

template <int BLOCK_M, int BLOCK_N, int BLOCK_K>
struct tinygemm_kernel_vnni<block_q8_K, block_iq4_xs, float, BLOCK_M, BLOCK_N, BLOCK_K> {
  static void apply(int pKb, const void *RESTRICT _A, const void *RESTRICT _B, float *RESTRICT pC, int pLdc) {

    constexpr int cCols = BLOCK_N / 16;
    const int cTileSize = TILE_N * sizeof(block_iq4_xs) + TILE_N * 2;

    const block_q8_K *RESTRICT vA = static_cast<const block_q8_K *>(_A);
    const char *RESTRICT vB = static_cast<const char *>(_B);

    // load the 256 bytes from A to 4 avx512 vectors
    __m512i aVa[4];
    __m512 aVc[cCols];
    __m512 vVd1;

    // packed_B:
    const int cOffsetScales = (QK_K / 2) * TILE_N;
    const int cOffsetD0 = (QK_K / 2) * TILE_N + 8 * TILE_N;

    // compensation
    __m512i vVcomp;

    const __m256i cM128s = _mm256_set1_epi16(128);
    const __m512i cLowMask = _mm512_set1_epi8(0xF);

    const __m512i cValues128 = _mm512_set_epi8(
        113, 89, 69, 53, 38, 25, 13, 1, -10, -22, -35, -49, -65, -83, -104, -127, 113, 89, 69, 53, 38, 25, 13, 1, -10,
        -22, -35, -49, -65, -83, -104, -127, 113, 89, 69, 53, 38, 25, 13, 1, -10, -22, -35, -49, -65, -83, -104, -127,
        113, 89, 69, 53, 38, 25, 13, 1, -10, -22, -35, -49, -65, -83, -104, -127);
    const __m512i cOff = _mm512_set1_epi8(static_cast<char>(0x80));
    const __m512i cValues256 = _mm512_add_epi8(cValues128, cOff);

    auto vLoadc = [&](auto pCol) { aVc[pCol] = _mm512_setzero_ps(); };
    Unroll<cCols>{}(vLoadc);

    auto vCompute = [&](auto pCol, auto pI) {
      if constexpr (pCol == 0) {
        // load a
        aVa[0] = _mm512_loadu_si512((const __m512i *)(vA[0 * pKb + pI].qs + 0));
        aVa[1] = _mm512_loadu_si512((const __m512i *)(vA[0 * pKb + pI].qs + 64));
        aVa[2] = _mm512_loadu_si512((const __m512i *)(vA[0 * pKb + pI].qs + 128));
        aVa[3] = _mm512_loadu_si512((const __m512i *)(vA[0 * pKb + pI].qs + 192));

        // compensation: 128 * A
        const __m256i cQ8sums = _mm256_loadu_si256((const __m256i *)vA[0 * pKb + pI].bsums);
        vVcomp = _mm512_castsi256_si512(_mm256_madd_epi16(cQ8sums, cM128s));
        vVd1 = _mm512_set1_ps(vA[0 * pKb + pI].d);
      }

      // accmulate the quants
      __m512i vAcc = _mm512_setzero_si512();
      const char *vBPtr = vB + PACKED_INDEX(pCol, pI, pKb, cTileSize);
      const char *vBQs = vBPtr;
      int vMask = 0;
      for (int vKGroup = 0; vKGroup < QK_K / 32; ++vKGroup) {
        int vR = vKGroup >> 1;
        __m512i vVmask = _mm512_set1_epi32(vKGroup);
        __m512i vVsum = _mm512_setzero_si512();
        for (int vK = 0; vK < 8; vK += 2) {
          __m512i vVa0 = _mm512_permutexvar_epi32(_mm512_set1_epi32(vMask++), aVa[vR]);
          __m512i vVa1 = _mm512_permutexvar_epi32(_mm512_set1_epi32(vMask++), aVa[vR]);

          __m512i vBytes = _mm512_loadu_si512(vBQs);
          __m512i vVb0 = _mm512_shuffle_epi8(cValues256, _mm512_and_si512(vBytes, cLowMask));
          __m512i vVb1 = _mm512_shuffle_epi8(cValues256, _mm512_and_si512(_mm512_srli_epi16(vBytes, 4), cLowMask));

          vVsum = _mm512_dpbusd_epi32(vVsum, vVb0, vVa0);
          vVsum = _mm512_dpbusd_epi32(vVsum, vVb1, vVa1);
          vBQs += 64;
        }
        // (B + 128) * A - 128 * A
        vVsum = _mm512_sub_epi32(vVsum, _mm512_permutexvar_epi32(vVmask, vVcomp));

        // vacc += scale * (q8 @ q4)
        const __m512i cVscale =
            _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i *)(vBPtr + cOffsetScales + vKGroup * TILE_N)));
        vAcc = _mm512_add_epi32(vAcc, _mm512_mullo_epi32(vVsum, cVscale));
      }
      const __m512 cVd0 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)(vBPtr + cOffsetD0)));
      aVc[pCol] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(vAcc), _mm512_mul_ps(cVd0, vVd1), aVc[pCol]);
    };

    for (int vI = 0; vI < pKb; ++vI) {
      Unroll<cCols>{}(vCompute, vI);
    }

    // store to C
    auto vStorec = [&](auto pCol) { _mm512_storeu_ps((__m512i *)(pC + 0 * pLdc + pCol * 16), aVc[pCol]); };
    Unroll<cCols>{}(vStorec);
  }
};

#define LAUNCH_TINYGEMM_KERNEL_VNNI(NB_SIZE)                                                                           \
  tinygemm_kernel_vnni<vec_dot_type, type, float, 1, NB_SIZE, blck_size>::apply(                                       \
      cKb, vWdataBatch, (const char *)vSrc0->data + vSrc0Offset + PACKED_INDEX(vNb * cKTilesN, 0, cKb, cTileSize),     \
      (float *)pDst->data + vDstOffset + vNbStart, cLdc)

template <typename TA, typename TB, typename TC, int BLOCK_K,
          typename std::enable_if<!is_type_qkk<TB>::value, int>::type = 0>
void tinygemm_kernel_amx(int pM, int pN, int pKb, const void *RESTRICT _A, const void *RESTRICT _B, TC *RESTRICT pC,
                         int pLdc) {
  using packed_B_t = packed_B_type<TB>;
  const int cTileSize = fGetTileSize<TB>();
  const bool cNeedUnpack = do_unpack<TB>::value;

  GGML_ASSERT(pM <= 2 * TILE_M && pN == 2 * TILE_N);
  const TA *RESTRICT vA = static_cast<const TA *>(_A);
  const char *RESTRICT vB = static_cast<const char *>(_B);

  const int cM0 = std::min(pM, TILE_M);
  const int cM1 = std::max(pM - TILE_M, 0);
  const int cLda = pKb * sizeof(TA);
  // const int ldb = KB * sizeof(TB);

  alignas(64) static thread_local packed_B_t vTile0[TILE_N * TILE_K];
  alignas(64) static thread_local packed_B_t vTile1[TILE_N * TILE_K];
  alignas(64) static thread_local int8_t vTile23[TILE_M * TILE_K];

  alignas(64) static thread_local int32_t aTileC0[TILE_M * TILE_N * 4];
  alignas(64) static thread_local int32_t aTileC1[TILE_M * TILE_N * 4];

  // double buffering C to interleave avx512 and amx
  int32_t *vCCur = aTileC0;
  int32_t *vCPre = aTileC1;

  auto vTile4 = [&](int32_t *pBase) { return pBase; };
  auto vTile5 = [&](int32_t *pBase) { return pBase + TILE_M * TILE_N; };
  auto vTile6 = [&](int32_t *pBase) { return pBase + 2 * TILE_M * TILE_N; };
  auto vTile7 = [&](int32_t *pBase) { return pBase + 3 * TILE_M * TILE_N; };

  if (pM == 2 * TILE_M) {
    // i = 0
    const char *vBBlk0 = vB + PACKED_INDEX(0, 0, pKb, cTileSize);
    const char *vBBlk1 = vB + PACKED_INDEX(1, 0, pKb, cTileSize);
    if (cNeedUnpack) {
      unpack_B<TB>(vTile0, vBBlk0);
      _tile_loadd(TMM0, vTile0, TILE_N * VNNI_BLK);
    } else {
      _tile_loadd(TMM0, vBBlk0, TILE_N * VNNI_BLK);
    }

    _tile_zero(TMM4);
    _tile_loadd(TMM2, vA[0].qs, cLda);
    _tile_dpbssd(TMM4, TMM2, TMM0);
    _tile_stored(TMM4, vTile4(vCPre), TILE_N * sizeof(int32_t));

    _tile_zero(TMM5);
    _tile_loadd(TMM3, vA[TILE_M * pKb + 0].qs, cLda);
    _tile_dpbssd(TMM5, TMM3, TMM0);
    _tile_stored(TMM5, vTile5(vCPre), TILE_N * sizeof(int32_t));

    if (cNeedUnpack) {
      unpack_B<TB>(vTile1, vBBlk1);
      _tile_loadd(TMM1, vTile1, TILE_N * VNNI_BLK);
    } else {
      _tile_loadd(TMM1, vBBlk1, TILE_N * VNNI_BLK);
    }

    _tile_zero(TMM6);
    _tile_dpbssd(TMM6, TMM2, TMM1);
    _tile_stored(TMM6, vTile6(vCPre), TILE_N * sizeof(int32_t));

    _tile_zero(TMM7);
    _tile_dpbssd(TMM7, TMM3, TMM1);
    _tile_stored(TMM7, vTile7(vCPre), TILE_N * sizeof(int32_t));

    for (int vI = 1; vI < pKb; ++vI) {
      // index of previous iter
      const int cIi = vI - 1;
      const char *vBBlk0 = vB + PACKED_INDEX(0, vI, pKb, cTileSize);
      const char *vBBlk1 = vB + PACKED_INDEX(1, vI, pKb, cTileSize);
      GGML_DISPATCH_BOOL(cIi > 0, cIsAcc, [&] {
        if (cNeedUnpack) {
          unpack_B<TB>(vTile0, vBBlk0);
          _tile_loadd(TMM0, vTile0, TILE_N * VNNI_BLK);
        } else {
          _tile_loadd(TMM0, vBBlk0, TILE_N * VNNI_BLK);
        }
        _tile_zero(TMM4);
        _tile_loadd(TMM2, vA[vI].qs, cLda);
        acc_C<TA, TB, cIsAcc>::apply(pC, pLdc, vTile4(vCPre), &vA[cIi], pKb, vB + PACKED_INDEX(0, cIi, pKb, cTileSize),
                                     TILE_M);

        _tile_dpbssd(TMM4, TMM2, TMM0);
        _tile_stored(TMM4, vTile4(vCCur), TILE_N * sizeof(int32_t));

        _tile_zero(TMM5);
        _tile_loadd(TMM3, vA[TILE_M * pKb + vI].qs, cLda);
        acc_C<TA, TB, cIsAcc>::apply(pC + TILE_M * pLdc, pLdc, vTile5(vCPre), &vA[TILE_M * pKb + cIi], pKb,
                                     vB + PACKED_INDEX(0, cIi, pKb, cTileSize), TILE_M);

        _tile_dpbssd(TMM5, TMM3, TMM0);
        _tile_stored(TMM5, vTile5(vCCur), TILE_N * sizeof(int32_t));

        if (cNeedUnpack) {
          unpack_B<TB>(vTile1, vBBlk1);
          _tile_loadd(TMM1, vTile1, TILE_N * VNNI_BLK);
        } else {
          _tile_loadd(TMM1, vBBlk1, TILE_N * VNNI_BLK);
        }
        _tile_zero(TMM6);
        acc_C<TA, TB, cIsAcc>::apply(pC + TILE_N, pLdc, vTile6(vCPre), &vA[cIi], pKb,
                                     vB + PACKED_INDEX(1, cIi, pKb, cTileSize), TILE_M);

        _tile_dpbssd(TMM6, TMM2, TMM1);
        _tile_stored(TMM6, vTile6(vCCur), TILE_N * sizeof(int32_t));

        _tile_zero(TMM7);
        acc_C<TA, TB, cIsAcc>::apply(pC + TILE_M * pLdc + TILE_N, pLdc, vTile7(vCPre), &vA[TILE_M * pKb + cIi], pKb,
                                     vB + PACKED_INDEX(1, cIi, pKb, cTileSize), TILE_M);

        _tile_dpbssd(TMM7, TMM3, TMM1);
        _tile_stored(TMM7, vTile7(vCCur), TILE_N * sizeof(int32_t));

        std::swap(vCCur, vCPre);
      });
    }
    // final accumulation
    {
      int vIi = pKb - 1;
      acc_C<TA, TB, true>::apply(pC, pLdc, vTile4(vCPre), &vA[vIi], pKb, vB + PACKED_INDEX(0, vIi, pKb, cTileSize),
                                 TILE_M);
      acc_C<TA, TB, true>::apply(pC + TILE_M * pLdc, pLdc, vTile5(vCPre), &vA[TILE_M * pKb + vIi], pKb,
                                 vB + PACKED_INDEX(0, vIi, pKb, cTileSize), TILE_M);
      acc_C<TA, TB, true>::apply(pC + TILE_N, pLdc, vTile6(vCPre), &vA[vIi], pKb,
                                 vB + PACKED_INDEX(1, vIi, pKb, cTileSize), TILE_M);
      acc_C<TA, TB, true>::apply(pC + TILE_M * pLdc + TILE_N, pLdc, vTile7(vCPre), &vA[TILE_M * pKb + vIi], pKb,
                                 vB + PACKED_INDEX(1, vIi, pKb, cTileSize), TILE_M);
    }
  } else {
    for (int vI = 0; vI < pKb; ++vI) {
      _tile_zero(TMM4);
      _tile_zero(TMM6);
      if (cM1 != 0) {
        _tile_zero(TMM5);
        _tile_zero(TMM7);
      }

      const char *vBBlk0 = vB + PACKED_INDEX(0, vI, pKb, cTileSize);
      const char *vBBlk1 = vB + PACKED_INDEX(1, vI, pKb, cTileSize);
      if (cNeedUnpack) {
        unpack_B<TB>(vTile0, vBBlk0);
        _tile_loadd(TMM0, vTile0, TILE_N * VNNI_BLK);
      } else {
        _tile_loadd(TMM0, vBBlk0, TILE_N * VNNI_BLK);
      }

      if (cNeedUnpack) {
        unpack_B<TB>(vTile1, vBBlk1);
        _tile_loadd(TMM1, vTile1, TILE_N * VNNI_BLK);
      } else {
        _tile_loadd(TMM1, vBBlk1, TILE_N * VNNI_BLK);
      }

      if (cM0 == TILE_M) {
        _tile_loadd(TMM2, vA[vI].qs, cLda);
      } else {
        unpack_A(vTile23, &vA[vI], pKb, cM0);
        _tile_loadd(TMM2, vTile23, TILE_K);
      }

      _tile_dpbssd(TMM4, TMM2, TMM0);
      _tile_dpbssd(TMM6, TMM2, TMM1);

      _tile_stored(TMM4, vTile4(vCCur), TILE_N * sizeof(int32_t));
      _tile_stored(TMM6, vTile6(vCCur), TILE_N * sizeof(int32_t));

      GGML_DISPATCH_BOOL(vI > 0, cIsAcc, [&] {
        acc_C<TA, TB, cIsAcc>::apply(pC, pLdc, vTile4(vCCur), &vA[vI], pKb, vB + PACKED_INDEX(0, vI, pKb, cTileSize),
                                     cM0);
        acc_C<TA, TB, cIsAcc>::apply(pC + TILE_N, pLdc, vTile6(vCCur), &vA[vI], pKb,
                                     vB + PACKED_INDEX(1, vI, pKb, cTileSize), cM0);
      });

      if (cM1 != 0) {
        unpack_A(vTile23, &vA[TILE_M * pKb + vI], pKb, cM1);
        _tile_loadd(TMM3, vTile23, TILE_K);

        _tile_dpbssd(TMM5, TMM3, TMM0);
        _tile_dpbssd(TMM7, TMM3, TMM1);
        _tile_stored(TMM5, vTile5(vCCur), TILE_N * sizeof(int32_t));
        _tile_stored(TMM7, vTile7(vCCur), TILE_N * sizeof(int32_t));
        GGML_DISPATCH_BOOL(vI > 0, cIsAcc, [&] {
          acc_C<TA, TB, cIsAcc>::apply(pC + TILE_M * pLdc, pLdc, vTile5(vCCur), &vA[TILE_M * pKb + vI], pKb,
                                       vB + PACKED_INDEX(0, vI, pKb, cTileSize), cM1);
          acc_C<TA, TB, cIsAcc>::apply(pC + TILE_M * pLdc + TILE_N, pLdc, vTile7(vCCur), &vA[TILE_M * pKb + vI], pKb,
                                       vB + PACKED_INDEX(1, vI, pKb, cTileSize), cM1);
        });
      }
    }
  }
  return;
}

template <typename TA, typename TB, typename TC, int BLOCK_K,
          typename std::enable_if<is_type_qkk<TB>::value, int>::type = 0>
void tinygemm_kernel_amx(int pM, int pN, int pKb, const void *RESTRICT _A, const void *RESTRICT _B, float *RESTRICT pC,
                         int pLdc) {
  static_assert(std::is_same<TA, block_q8_K>::value);
  const int cTileSize = fGetTileSize<TB>();

  GGML_ASSERT(pM <= 2 * TILE_M && pN == 2 * TILE_N);
  const TA *RESTRICT vA = static_cast<const TA *>(_A);
  const char *RESTRICT vB = static_cast<const char *>(_B);

  const int cM0 = std::min(pM, TILE_M);
  const int cM1 = std::max(pM - TILE_M, 0);
  // const int lda = KB * sizeof(TA);

  alignas(64) static thread_local int8_t vTile0[TILE_N * TILE_K];
  alignas(64) static thread_local int8_t vTile1[TILE_N * TILE_K];
  alignas(64) static thread_local int8_t vTile23[TILE_M * TILE_K];

  // mat mul result for each group
  alignas(64) static thread_local int32_t vTile4[TILE_M * TILE_N];
  alignas(64) static thread_local int32_t vTile5[TILE_M * TILE_N];
  alignas(64) static thread_local int32_t vTile6[TILE_M * TILE_N];
  alignas(64) static thread_local int32_t vTile7[TILE_M * TILE_N];

  // sum of each QK_K block, contains 8 groups, int32
  alignas(64) static thread_local int32_t vSumi4[TILE_M * TILE_N];
  alignas(64) static thread_local int32_t vSumi5[TILE_M * TILE_N];
  alignas(64) static thread_local int32_t vSumi6[TILE_M * TILE_N];
  alignas(64) static thread_local int32_t vSumi7[TILE_M * TILE_N];

  const int cKGroupSize = std::is_same<TB, block_q6_K>::value ? 16 : 32;
  for (int vI = 0; vI < pKb; ++vI) {
    // step 1: accumulate the quants across 8 groups, each group with 32
    for (int vK = 0; vK < QK_K / cKGroupSize; ++vK) {
      GGML_DISPATCH_BOOL(vK > 0, cIsAcc, [&] {
        _tile_zero(TMM4);
        _tile_zero(TMM6);

        unpack_B<TB>(vTile0, vB + PACKED_INDEX(0, vI, pKb, cTileSize), vK);
        _tile_loadd(TMM0, vTile0, TILE_N * VNNI_BLK);

        unpack_B<TB>(vTile1, vB + PACKED_INDEX(1, vI, pKb, cTileSize), vK);
        _tile_loadd(TMM1, vTile1, TILE_N * VNNI_BLK);

        unpack_A<TB>(vTile23, &vA[vI], pKb, vK, cM0);
        _tile_loadd(TMM2, vTile23, TILE_K);

        _tile_dpbssd(TMM4, TMM2, TMM0);
        _tile_dpbssd(TMM6, TMM2, TMM1);

        _tile_stored(TMM4, vTile4, TILE_N * sizeof(int32_t));
        _tile_stored(TMM6, vTile6, TILE_N * sizeof(int32_t));

        fScaleC<TB, cIsAcc>(vTile4, vSumi4, vB + PACKED_INDEX(0, vI, pKb, cTileSize), vK, cM0);
        fScaleC<TB, cIsAcc>(vTile6, vSumi6, vB + PACKED_INDEX(1, vI, pKb, cTileSize), vK, cM0);

        if (cM1 != 0) {
          _tile_zero(TMM5);
          _tile_zero(TMM7);

          unpack_A<TB>(vTile23, &vA[TILE_M * pKb + vI], pKb, vK, cM1);
          _tile_loadd(TMM3, vTile23, TILE_K);

          _tile_dpbssd(TMM5, TMM3, TMM0);
          _tile_dpbssd(TMM7, TMM3, TMM1);

          _tile_stored(TMM5, vTile5, TILE_N * sizeof(int32_t));
          _tile_stored(TMM7, vTile7, TILE_N * sizeof(int32_t));

          fScaleC<TB, cIsAcc>(vTile5, vSumi5, vB + PACKED_INDEX(0, vI, pKb, cTileSize), vK, cM1);
          fScaleC<TB, cIsAcc>(vTile7, vSumi7, vB + PACKED_INDEX(1, vI, pKb, cTileSize), vK, cM1);
        }
      });
    }

    // step 2: accmulate the mins
    GGML_DISPATCH_BOOL(vI > 0, cIsAcc, [&] {
      acc_C<TA, TB, cIsAcc>::apply(pC, pLdc, vSumi4, &vA[vI], pKb, vB + PACKED_INDEX(0, vI, pKb, cTileSize), cM0);
      acc_C<TA, TB, cIsAcc>::apply(pC + TILE_N, pLdc, vSumi6, &vA[vI], pKb, vB + PACKED_INDEX(1, vI, pKb, cTileSize),
                                   cM0);
      if (cM1 != 0) {
        acc_C<TA, TB, cIsAcc>::apply(pC + TILE_M * pLdc, pLdc, vSumi5, &vA[TILE_M * pKb + vI], pKb,
                                     vB + PACKED_INDEX(0, vI, pKb, cTileSize), cM1);
        acc_C<TA, TB, cIsAcc>::apply(pC + TILE_M * pLdc + TILE_N, pLdc, vSumi7, &vA[TILE_M * pKb + vI], pKb,
                                     vB + PACKED_INDEX(1, vI, pKb, cTileSize), cM1);
      }
    });
  }
  return;
}

} // anonymous namespace

// get the packed tensor size for quantized weights
size_t ggml_backend_amx_get_alloc_size(const struct ggml_tensor *pTensor) {
  const enum ggml_type TYPE = pTensor->type;

  const int cK = pTensor->ne[0]; // ne0: in_features
  const int cN = pTensor->ne[1]; // ne1: out_features

  auto fGetTensorSize = [&] {
    size_t vRowSizeB{0};
    GGML_DISPATCH_QTYPES(TYPE, [&] { vRowSizeB = fGetRowSize<type, blck_size>(cK); });
    return cN * vRowSizeB;
  };

  if (qtype_has_amx_kernels(TYPE)) {
    return fGetTensorSize();
  } else {
    // for f16, bf16 we don't do packing
    return ggml_nbytes(pTensor);
  }
}

// pack weight to vnni format
void ggml_backend_amx_convert_weight(struct ggml_tensor *pTensor, const void *pData, size_t pOffset, size_t pSize) {
  GGML_ASSERT(pOffset == 0 && pSize == ggml_nbytes(pTensor)); // only full tensor conversion is supported for now

  const enum ggml_type TYPE = pTensor->type;

  const int cK = pTensor->ne[0]; // ne0: in_features
  const int cN = pTensor->ne[1]; // ne1: out_features

  GGML_DISPATCH_QTYPES(TYPE, [&] {
    fConvertBPackedFormat<type, blck_size>((void *)((char *)pTensor->data + pOffset), (const type *)pData, cN, cK);
  });
}

// ne2 is passed explicitly to help compiler optimize repeated calls
inline int64_t fGgmlBatchOffset(const ggml_tensor *pT, int64_t pBatchIdx, int64_t pNe2) {
  const int64_t cI2 = pBatchIdx % pNe2;
  const int64_t cI3 = pBatchIdx / pNe2;
  return cI3 * pT->nb[3] + cI2 * pT->nb[2];
}

size_t ggml_backend_amx_desired_wsize(const struct ggml_tensor *pDst) {
  struct ggml_tensor *vSrc0 = pDst->src[0];

  const enum ggml_type TYPE = vSrc0->type;

  const bool cIsFloatingType = TYPE == GGML_TYPE_F16;
  if (cIsFloatingType) {
    return 0;
  }

  const int cM = pDst->ne[1];
  const int cK = vSrc0->ne[0];
  const int64_t cNBatch = pDst->ne[2] * pDst->ne[3];

  size_t vDesiredWsize = 0;

  GGML_DISPATCH_QTYPES(TYPE, [&] {
    const size_t cRowSizeA = cK / blck_size * sizeof(vec_dot_type);
    vDesiredWsize = cNBatch * cM * cRowSizeA;
  });

  return vDesiredWsize;
}

// NB: mixed dtype gemm with Advanced Matrix Extensions (Intel AMX)
//
// src0: weight in shape of {N, K}, quantized
// src1: input  in shape of {M, K}, float32
// dst:  output in shape of {M, N}, float32
//
// the function performs: dst = src1 @ src0.T for each batch
//
void ggml_backend_amx_mul_mat(const ggml_compute_params *pParams, struct ggml_tensor *pDst) {
  struct ggml_tensor *vSrc0 = pDst->src[0];
  struct ggml_tensor *vSrc1 = pDst->src[1];

  const enum ggml_type TYPE = vSrc0->type;

  // f16 only has avx512 kernels for now,
  // amx kernels will be added once 6th gen xeon is released.
  const bool cIsFloatingType = TYPE == GGML_TYPE_F16;

  const int cM = pDst->ne[1];
  const int cN = pDst->ne[0];
  const int cK = vSrc0->ne[0];
  const int cLdc = pDst->nb[1] / pDst->nb[0];

  const int64_t cNe2 = pDst->ne[2];
  const int64_t cNBatch = cNe2 * pDst->ne[3];

  if (cIsFloatingType) {
    constexpr int cBlockM = 4;
    constexpr int cBlockN = 6;
    const int cMb = div_up(cM, cBlockM);
    const int cNb = div_up(cN, cBlockN);

    parallel_for_ggml(pParams, cNBatch * cMb * cNb, [&](int pBegin, int pEnd) {
      GGML_DISPATCH_FLOATING_TYPES(TYPE, [&] {
        for (int vI = pBegin; vI < pEnd; ++vI) {
          int vBatchIdx = vI / (cMb * cNb);
          int vRemaining = vI % (cMb * cNb);
          int vMb = vRemaining / cNb;
          int vNb = vRemaining % cNb;

          int64_t vSrc0Offset = fGgmlBatchOffset(vSrc0, vBatchIdx, cNe2);
          int64_t vSrc1Offset = fGgmlBatchOffset(vSrc1, vBatchIdx, cNe2);
          int64_t vDstOffset = fGgmlBatchOffset(pDst, vBatchIdx, cNe2);

          int vMbStart = vMb * cBlockM;
          int vMbSize = std::min(cBlockM, cM - vMbStart);
          int vNbStart = vNb * cBlockN;
          int vNbSize = std::min(cBlockN, cN - vNbStart);

          switch (vMbSize << 4 | vNbSize) {
          case 0x12:
            LAUNCH_TINYGEMM_KERNEL_AVX(1, 2);
            break;
          case 0x14:
            LAUNCH_TINYGEMM_KERNEL_AVX(1, 4);
            break;
          case 0x16:
            LAUNCH_TINYGEMM_KERNEL_AVX(1, 6);
            break;
          case 0x22:
            LAUNCH_TINYGEMM_KERNEL_AVX(2, 2);
            break;
          case 0x24:
            LAUNCH_TINYGEMM_KERNEL_AVX(2, 4);
            break;
          case 0x26:
            LAUNCH_TINYGEMM_KERNEL_AVX(2, 6);
            break;
          case 0x32:
            LAUNCH_TINYGEMM_KERNEL_AVX(3, 2);
            break;
          case 0x34:
            LAUNCH_TINYGEMM_KERNEL_AVX(3, 4);
            break;
          case 0x36:
            LAUNCH_TINYGEMM_KERNEL_AVX(3, 6);
            break;
          case 0x42:
            LAUNCH_TINYGEMM_KERNEL_AVX(4, 2);
            break;
          case 0x44:
            LAUNCH_TINYGEMM_KERNEL_AVX(4, 4);
            break;
          case 0x46:
            LAUNCH_TINYGEMM_KERNEL_AVX(4, 6);
            break;
          default:
            fprintf(stderr, "Unexpected block size!\n");
          }
        }
      });
    });
    return;
  }

  // pointer to work space, used convert A from float to quantized type
  void *vWdata = pParams->wdata;

  // TODO: performance improvement: merge quant A
  // if (params->ith == 0) {
  GGML_DISPATCH_QTYPES(TYPE, [&] {
    const size_t cRowSizeA = cK / blck_size * sizeof(vec_dot_type);
    const size_t cDesiredWsize = cNBatch * cM * cRowSizeA;
    if (pParams->wsize < cDesiredWsize) {
      GGML_ABORT("insufficient work space size");
    }

    // Q4_0, Q4_1, Q8_0 handles 1 TILE_K per blck_size
    // Q4_K, Q5_K, Q6_K, IQ4_XS handles 8 TILE_K per blck_size
    GGML_ASSERT(TILE_K == blck_size || TILE_K * 8 == blck_size);

    parallel_for_ggml(pParams, cNBatch * cM, [&](int pBegin, int pEnd) {
      for (int vIdx = pBegin; vIdx < pEnd; ++vIdx) {
        int vBatchIdx = vIdx / cM;
        int vM = vIdx % cM;
        int64_t vSrc1Offset = fGgmlBatchOffset(vSrc1, vBatchIdx, cNe2);
        const float *vAData = (const float *)((const char *)vSrc1->data + vSrc1Offset);
        char *vWdataBatch = (char *)vWdata + vBatchIdx * cM * cRowSizeA;
        from_float<vec_dot_type>(vAData + vM * cK, vWdataBatch + vM * cRowSizeA, cK);
      }
    });
  });
  // }

  ggml_barrier(pParams->threadpool);

  if (cM == 1) {
    // MB = 1 and handle 8 tiles in each block
    constexpr int cKTilesN = 4;
    constexpr int cBlockN = TILE_N * cKTilesN;
    const int cNb = div_up(cN, cBlockN);

    parallel_for_ggml(pParams, cNBatch * cNb, [&](int pBegin, int pEnd) {
      GGML_DISPATCH_QTYPES(TYPE, [&] {
        const int cKb = cK / blck_size;
        const int cTileSize = fGetTileSize<type>();
        const int cRowSizeA = cKb * sizeof(vec_dot_type);
        for (int vI = pBegin; vI < pEnd; ++vI) {
          int vBatchIdx = vI / cNb;
          int vNb = vI % cNb;

          int64_t vSrc0Offset = fGgmlBatchOffset(vSrc0, vBatchIdx, cNe2);
          int64_t vDstOffset = fGgmlBatchOffset(pDst, vBatchIdx, cNe2);
          const char *vWdataBatch = (const char *)vWdata + vBatchIdx * cRowSizeA;

          int vNbStart = vNb * cBlockN;
          int vNbSize = std::min(cBlockN, cN - vNbStart); // 32, 64, 96

          switch (vNbSize) {
          // case 160: LAUNCH_TINYGEMM_KERNEL_VNNI(160); break;
          case 128:
            LAUNCH_TINYGEMM_KERNEL_VNNI(128);
            break;
          case 96:
            LAUNCH_TINYGEMM_KERNEL_VNNI(96);
            break;
          case 64:
            LAUNCH_TINYGEMM_KERNEL_VNNI(64);
            break;
          case 32:
            LAUNCH_TINYGEMM_KERNEL_VNNI(32);
            break;
          default:
            fprintf(stderr, "Unexpected n block size!\n");
          }
        }
      });
    });
    return;
  }

  // handle 4 tiles at a tile
  constexpr int cBlockM = TILE_M * 2;
  constexpr int cBlockN = TILE_N * 2;
  const int cMb = div_up(cM, cBlockM);
  const int cNb = div_up(cN, cBlockN);

  parallel_for_ggml(pParams, cNBatch * cMb * cNb, [&](int pBegin, int pEnd) {
    // init tile config for each thread
    fGgmlTileConfigInit();

    GGML_DISPATCH_QTYPES(TYPE, [&] {
      const int cKb = cK / blck_size;
      const int cTileSize = fGetTileSize<type>();
      const int cRowSizeA = cKb * sizeof(vec_dot_type);

      for (int vI = pBegin; vI < pEnd; ++vI) {
        int vBatchIdx = vI / (cMb * cNb);
        int vRemaining = vI % (cMb * cNb);
        int vMb = vRemaining / cNb;
        int vNb = vRemaining % cNb;

        int64_t vSrc0Offset = fGgmlBatchOffset(vSrc0, vBatchIdx, cNe2);
        int64_t vDstOffset = fGgmlBatchOffset(pDst, vBatchIdx, cNe2);
        const char *vWdataBatch = (const char *)vWdata + vBatchIdx * cM * cRowSizeA;

        int vMbStart = vMb * cBlockM;
        int vMbSize = std::min(cBlockM, cM - vMbStart);
        int vNbStart = vNb * cBlockN;
        int vNbSize = cBlockN;

        tinygemm_kernel_amx<vec_dot_type, type, float, blck_size>(
            vMbSize, vNbSize, cKb, vWdataBatch + vMbStart * cRowSizeA,
            (const char *)vSrc0->data + vSrc0Offset + PACKED_INDEX(vNb * 2, 0, cKb, cTileSize),
            (float *)pDst->data + vDstOffset + vMbStart * cN + vNbStart, cLdc);
      }
    });
  });
}

#endif // if defined(__AMX_INT8__) && defined(__AVX512VNNI__)
