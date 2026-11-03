#pragma once

// GGML internal header

#include "ggml.h"
#include "gguf.h"

#include <assert.h>
#include <math.h>
#include <stdlib.h> // load `stdlib.h` before other headers to work around MinGW bug: https://sourceforge.net/p/mingw-w64/bugs/192/
#include <stdbool.h>
#include <stdint.h>
#include <string.h>

#ifdef __ARM_FEATURE_SVE
#include <arm_sve.h>
#endif // __ARM_FEATURE_SVE

#if defined(__ARM_NEON) && !defined(__CUDACC__)
// if YCM cannot find <arm_neon.h>, make a symbolic link to it, for example:
//
//   $ ln -sfn /Library/Developer/CommandLineTools/usr/lib/clang/13.1.6/include/arm_neon.h ./src/
//
#include <arm_neon.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

void ggml_print_backtrace(void);

uint64_t ggml_graph_next_uid(void);

#ifndef MIN
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#endif

#ifndef MAX
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#endif

// required for mmap as gguf only guarantees 32-byte alignment
#define TENSOR_ALIGNMENT 32

// static_assert should be a #define, but if it's not,
// fall back to the _Static_assert C11 keyword.
// if C99 - static_assert is noop
// ref: https://stackoverflow.com/a/53923785/4039976
#ifndef __cplusplus
#ifndef static_assert
#if defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 201100L)
#define static_assert(cond, msg) _Static_assert(cond, msg)
#else
#define static_assert(cond, msg) struct global_scope_noop_trick
#endif
#endif
#endif

static inline int ggml_up32(int pN) { return (pN + 31) & ~31; }

// static inline int ggml_up64(int n) {
//     return (n + 63) & ~63;
// }

static inline int ggml_up(int pN, int pM) {
  // assert m is a power of 2
  GGML_ASSERT((pM & (pM - 1)) == 0);
  return (pN + pM - 1) & ~(pM - 1);
}

// TODO: move to ggml.h? (won't be able to inline)
static bool ggml_are_same_layout(const struct ggml_tensor *pA, const struct ggml_tensor *pB) {
  if (pA->type != pB->type) {
    return false;
  }
  for (int vI = 0; vI < GGML_MAX_DIMS; vI++) {
    if (pA->ne[vI] != pB->ne[vI]) {
      return false;
    }
    if (pA->nb[vI] != pB->nb[vI]) {
      return false;
    }
  }
  return true;
}

static bool ggml_op_is_empty(enum ggml_op pOp) {
  switch (pOp) {
  case GGML_OP_NONE:
  case GGML_OP_RESHAPE:
  case GGML_OP_TRANSPOSE:
  case GGML_OP_VIEW:
  case GGML_OP_PERMUTE:
    return true;
  default:
    return false;
  }
}

static inline bool ggml_impl_is_view(const struct ggml_tensor *pT) { return pT->view_src != NULL; }

static inline float ggml_compute_softplus_f32(float pInput) {
  return (pInput > 20.0f) ? pInput : logf(1 + expf(pInput));
}
//
// logging
//

GGML_ATTRIBUTE_FORMAT(2, 3)
GGML_API void ggml_log_internal(enum ggml_log_level pLevel, const char *pFormat, ...);
GGML_API void ggml_log_callback_default(enum ggml_log_level pLevel, const char *pText, void *pUserData);

#define GGML_LOG(...) ggml_log_internal(GGML_LOG_LEVEL_NONE, __VA_ARGS__)
#define GGML_LOG_INFO(...) ggml_log_internal(GGML_LOG_LEVEL_INFO, __VA_ARGS__)
#define GGML_LOG_WARN(...) ggml_log_internal(GGML_LOG_LEVEL_WARN, __VA_ARGS__)
#define GGML_LOG_ERROR(...) ggml_log_internal(GGML_LOG_LEVEL_ERROR, __VA_ARGS__)
#define GGML_LOG_DEBUG(...) ggml_log_internal(GGML_LOG_LEVEL_DEBUG, __VA_ARGS__)
#define GGML_LOG_CONT(...) ggml_log_internal(GGML_LOG_LEVEL_CONT, __VA_ARGS__)

#define GGML_DEBUG 0

#if (GGML_DEBUG >= 1)
#define GGML_PRINT_DEBUG(...) GGML_LOG_DEBUG(__VA_ARGS__)
#else
#define GGML_PRINT_DEBUG(...)
#endif

#if (GGML_DEBUG >= 5)
#define GGML_PRINT_DEBUG_5(...) GGML_LOG_DEBUG(__VA_ARGS__)
#else
#define GGML_PRINT_DEBUG_5(...)
#endif

#if (GGML_DEBUG >= 10)
#define GGML_PRINT_DEBUG_10(...) GGML_LOG_DEBUG(__VA_ARGS__)
#else
#define GGML_PRINT_DEBUG_10(...)
#endif

// tensor params

static void ggml_set_op_params(struct ggml_tensor *pTensor, const void *pParams, size_t pParamsSize) {
  GGML_ASSERT(pTensor != NULL); // silence -Warray-bounds warnings
  assert(pParamsSize <= GGML_MAX_OP_PARAMS);
  memcpy(pTensor->op_params, pParams, pParamsSize);
}

static int32_t ggml_get_op_params_i32(const struct ggml_tensor *pTensor, uint32_t pI) {
  assert(pI < GGML_MAX_OP_PARAMS / sizeof(int32_t));
  return ((const int32_t *)(pTensor->op_params))[pI];
}

static float ggml_get_op_params_f32(const struct ggml_tensor *pTensor, uint32_t pI) {
  assert(pI < GGML_MAX_OP_PARAMS / sizeof(float));
  return ((const float *)(pTensor->op_params))[pI];
}

static void ggml_set_op_params_i32(struct ggml_tensor *pTensor, uint32_t pI, int32_t pValue) {
  assert(pI < GGML_MAX_OP_PARAMS / sizeof(int32_t));
  ((int32_t *)(pTensor->op_params))[pI] = pValue;
}

static void ggml_set_op_params_f32(struct ggml_tensor *pTensor, uint32_t pI, float pValue) {
  assert(pI < GGML_MAX_OP_PARAMS / sizeof(float));
  ((float *)(pTensor->op_params))[pI] = pValue;
}

// GGML_OP_AMAX_POW2: smallest power of two e >= 1 with pAmax / e <= pThreshold, capped at 2^127, that is
// 2^clamp(ceil(log2(pAmax / pThreshold)), 0, 127). It is taken from the bits of the ratio, so the CUDA and Vulkan
// kernels, which do the same, give the same exact value. A row of zeros or NaN gives 1.
static inline float fGgmlAmaxPow2F32(float pAmax, float pThreshold) {
  const float cRatio = pAmax / pThreshold;
  if (!(cRatio > 1.0f)) {
    return 1.0f;
  }
  uint32_t vBits;
  memcpy(&vBits, &cRatio, sizeof(vBits));
  int32_t vExp = (int32_t)((vBits >> 23) & 0xFF) - 127;
  if ((vBits & 0x7FFFFF) != 0) {
    vExp += 1;
  }
  if (vExp > 127) {
    vExp = 127;
  }
  const uint32_t cResultBits = (uint32_t)(vExp + 127) << 23;
  float vResult;
  memcpy(&vResult, &cResultBits, sizeof(vResult));
  return vResult;
}

struct ggml_map_custom1_op_params {
  ggml_custom1_op_t fun;
  int n_tasks;
  void *userdata;
};

struct ggml_map_custom2_op_params {
  ggml_custom2_op_t fun;
  int n_tasks;
  void *userdata;
};

struct ggml_map_custom3_op_params {
  ggml_custom3_op_t fun;
  int n_tasks;
  void *userdata;
};

struct ggml_custom_op_params {
  ggml_custom_op_t fun;
  int n_tasks;
  void *userdata;
};

// bitset

typedef uint32_t ggml_bitset_t;

static_assert(sizeof(ggml_bitset_t) == 4, "bitset_t constants must be updated");
#define BITSET_SHR 5 // log2(sizeof(ggml_bitset_t)*8)
#define BITSET_MASK (sizeof(ggml_bitset_t) * 8 - 1)

static size_t ggml_bitset_size(size_t pN) { return (pN + BITSET_MASK) >> BITSET_SHR; }

static inline bool ggml_bitset_get(const ggml_bitset_t *pBitset, size_t pI) {
  return !!(pBitset[pI >> BITSET_SHR] & (1u << (pI & BITSET_MASK)));
}

static inline void ggml_bitset_set(ggml_bitset_t *pBitset, size_t pI) {
  pBitset[pI >> BITSET_SHR] |= (1u << (pI & BITSET_MASK));
}

static inline void ggml_bitset_clear(ggml_bitset_t *pBitset, size_t pI) {
  pBitset[pI >> BITSET_SHR] &= ~(1u << (pI & BITSET_MASK));
}

// hash set

#define GGML_HASHSET_FULL ((size_t)-1)
#define GGML_HASHSET_ALREADY_EXISTS ((size_t)-2)

struct ggml_hash_set {
  size_t size;
  ggml_bitset_t *used;       // whether or not the keys are in use i.e. set
  struct ggml_tensor **keys; // actual tensors in the set, keys[i] is only defined if ggml_bitset_get(used, i)
};

struct ggml_hash_set ggml_hash_set_new(size_t pSize);
void ggml_hash_set_free(struct ggml_hash_set *pHashSet);

// returns the minimum size for a hash set that can hold min_sz elements
size_t ggml_hash_size(size_t pMinSz);

// remove all elements from the hash set
void ggml_hash_set_reset(struct ggml_hash_set *pHashSet);

// returns true if key is in the hash set
static bool ggml_hash_contains(const struct ggml_hash_set *pHashSet, struct ggml_tensor *pKey);

// returns GGML_HASHSET_FULL if table is full, otherwise the current index of the key or where it should be inserted
static size_t ggml_hash_find(const struct ggml_hash_set *pHashSet, const struct ggml_tensor *pKey);

// returns GGML_HASHSET_ALREADY_EXISTS if key already exists, index otherwise, asserts if table is full
static size_t ggml_hash_insert(struct ggml_hash_set *pHashSet, struct ggml_tensor *pKey);

// return index, asserts if table is full
static size_t ggml_hash_find_or_insert(struct ggml_hash_set *pHashSet, struct ggml_tensor *pKey);

// hash function for ggml_tensor
static inline size_t ggml_hash(const struct ggml_tensor *p_) {
  // the last 4 bits are always zero due to alignment
  return (size_t)(uintptr_t)p_ >> 4;
}

static size_t ggml_hash_find(const struct ggml_hash_set *pHashSet, const struct ggml_tensor *pKey) {
  size_t vH = ggml_hash(pKey) % pHashSet->size;

  // linear probing
  size_t vI = vH;
  while (ggml_bitset_get(pHashSet->used, vI) && pHashSet->keys[vI] != pKey) {
    vI = (vI + 1) % pHashSet->size;
    if (vI == vH) {
      // visited all hash table entries -> not found
      return GGML_HASHSET_FULL;
    }
  }
  return vI;
}

static bool ggml_hash_contains(const struct ggml_hash_set *pHashSet, struct ggml_tensor *pKey) {
  size_t vI = ggml_hash_find(pHashSet, pKey);
  return vI != GGML_HASHSET_FULL && ggml_bitset_get(pHashSet->used, vI);
}

static size_t ggml_hash_insert(struct ggml_hash_set *pHashSet, struct ggml_tensor *pKey) {
  size_t vH = ggml_hash(pKey) % pHashSet->size;

  // linear probing
  size_t vI = vH;
  do {
    if (!ggml_bitset_get(pHashSet->used, vI)) {
      ggml_bitset_set(pHashSet->used, vI);
      pHashSet->keys[vI] = pKey;
      return vI;
    }
    if (pHashSet->keys[vI] == pKey) {
      return GGML_HASHSET_ALREADY_EXISTS;
    }
    vI = (vI + 1) % pHashSet->size;
  } while (vI != vH);

  // visited all hash table entries -> not found
  GGML_ABORT("fatal error");
}

static size_t ggml_hash_find_or_insert(struct ggml_hash_set *pHashSet, struct ggml_tensor *pKey) {
  size_t vH = ggml_hash(pKey) % pHashSet->size;

  // linear probing
  size_t vI = vH;
  do {
    if (!ggml_bitset_get(pHashSet->used, vI)) {
      ggml_bitset_set(pHashSet->used, vI);
      pHashSet->keys[vI] = pKey;
      return vI;
    }
    if (pHashSet->keys[vI] == pKey) {
      return vI;
    }
    vI = (vI + 1) % pHashSet->size;
  } while (vI != vH);

  // visited all hash table entries -> not found
  GGML_ABORT("fatal error");
}

// computation graph

enum ggml_cgraph_eval_order {
  GGML_CGRAPH_EVAL_ORDER_LEFT_TO_RIGHT = 0,
  GGML_CGRAPH_EVAL_ORDER_RIGHT_TO_LEFT,
  GGML_CGRAPH_EVAL_ORDER_COUNT
};

struct ggml_cgraph {
  int size;    // maximum number of nodes/leafs/grads/grad_accs
  int n_nodes; // number of nodes currently in use
  int n_leafs; // number of leafs currently in use

  struct ggml_tensor **nodes;     // tensors with data that can change if the graph is evaluated
  struct ggml_tensor **grads;     // the outputs of these tensors are the gradients of the nodes
  struct ggml_tensor **grad_accs; // accumulators for node gradients
  struct ggml_tensor **leafs;     // tensors with constant data
  int32_t *use_counts;            // number of uses of each tensor, indexed by hash table slot

  struct ggml_hash_set visited_hash_set;

  enum ggml_cgraph_eval_order order;

  // an optional identifier that can be utilized to recognize same graphs if two non-zero values match
  // a value of 0 means it is not set and should be ignored
  uint64_t uid;
};

// returns a slice of cgraph with nodes [i0, i1)
// the slice does not have leafs or gradients
// if you need the gradients, get them from the original graph
struct ggml_cgraph ggml_graph_view(struct ggml_cgraph *pCgraph, int pI0, int pI1);

// ggml-alloc.c: true if the operation can reuse memory from its sources
GGML_API bool ggml_op_can_inplace(enum ggml_op pOp);

// Memory allocation

GGML_API void *ggml_aligned_malloc(size_t pSize);
GGML_API void ggml_aligned_free(void *pPtr, size_t pSize);

// FP16 <-> FP32
// ref: https://github.com/Maratyszcza/FP16

static inline float fp32_from_bits(uint32_t pW) {
  union {
    uint32_t as_bits;
    float as_value;
  } vFp32;
  vFp32.as_bits = pW;
  return vFp32.as_value;
}

static inline uint32_t fp32_to_bits(float pF) {
  union {
    float as_value;
    uint32_t as_bits;
  } vFp32;
  vFp32.as_value = pF;
  return vFp32.as_bits;
}

static inline float ggml_compute_fp16_to_fp32(ggml_fp16_t pH) {
  const uint32_t cW = (uint32_t)pH << 16;
  const uint32_t cSign = cW & UINT32_C(0x80000000);
  const uint32_t cTwoW = cW + cW;

  const uint32_t cExpOffset = UINT32_C(0xE0) << 23;
#if (defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 199901L) || defined(__GNUC__) && !defined(__STRICT_ANSI__)) &&  \
    (!defined(__cplusplus) || __cplusplus >= 201703L)
  const float cExpScale = 0x1.0p-112f;
#else
  const float cExpScale = fp32_from_bits(UINT32_C(0x7800000));
#endif
  const float cNormalizedValue = fp32_from_bits((cTwoW >> 4) + cExpOffset) * cExpScale;

  const uint32_t cMagicMask = UINT32_C(126) << 23;
  const float cMagicBias = 0.5f;
  const float cDenormalizedValue = fp32_from_bits((cTwoW >> 17) | cMagicMask) - cMagicBias;

  const uint32_t cDenormalizedCutoff = UINT32_C(1) << 27;
  const uint32_t cResult =
      cSign | (cTwoW < cDenormalizedCutoff ? fp32_to_bits(cDenormalizedValue) : fp32_to_bits(cNormalizedValue));
  return fp32_from_bits(cResult);
}

static inline ggml_fp16_t ggml_compute_fp32_to_fp16(float f) {
#if (defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 199901L) || defined(__GNUC__) && !defined(__STRICT_ANSI__)) &&  \
    (!defined(__cplusplus) || __cplusplus >= 201703L)
  const float cScaleToInf = 0x1.0p+112f;
  const float cScaleToZero = 0x1.0p-110f;
#else
  const float cScaleToInf = fp32_from_bits(UINT32_C(0x77800000));
  const float cScaleToZero = fp32_from_bits(UINT32_C(0x08800000));
#endif
  float vBase = (fabsf(f) * cScaleToInf) * cScaleToZero;

  const uint32_t cW = fp32_to_bits(f);
  const uint32_t cShl1W = cW + cW;
  const uint32_t cSign = cW & UINT32_C(0x80000000);
  uint32_t vBias = cShl1W & UINT32_C(0xFF000000);
  if (vBias < UINT32_C(0x71000000)) {
    vBias = UINT32_C(0x71000000);
  }

  vBase = fp32_from_bits((vBias >> 1) + UINT32_C(0x07800000)) + vBase;
  const uint32_t cBits = fp32_to_bits(vBase);
  const uint32_t cExpBits = (cBits >> 13) & UINT32_C(0x00007C00);
  const uint32_t cMantissaBits = cBits & UINT32_C(0x00000FFF);
  const uint32_t cNonsign = cExpBits + cMantissaBits;
  return (cSign >> 16) | (cShl1W > UINT32_C(0xFF000000) ? UINT16_C(0x7E00) : cNonsign);
}

#define GGML_COMPUTE_FP16_TO_FP32(x) ggml_compute_fp16_to_fp32(x)
#define GGML_COMPUTE_FP32_TO_FP16(x) ggml_compute_fp32_to_fp16(x)

#define GGML_FP16_TO_FP32(x) GGML_COMPUTE_FP16_TO_FP32(x)
#define GGML_FP32_TO_FP16(x) GGML_COMPUTE_FP32_TO_FP16(x)

static inline float ggml_e8m0_to_fp32(uint8_t pX) {
  uint32_t vBits; // Stores the raw bit representation of the float

  // Handle special case for minimum exponent (denormalized float)
  if (pX == 0) {
    // Bit pattern for 2^(-127):
    // - Sign bit: 0 (positive)
    // - Exponent: 0 (denormalized number)
    // - Mantissa: 0x400000 (0.5 in fractional form)
    // Value = 0.5 * 2^(-126) = 2^(-127)
    vBits = 0x00400000;
  }
  // note: disabled as we don't need to handle NaNs
  //// Handle special case for NaN (all bits set)
  // else if (x == 0xFF) {
  //     // Standard quiet NaN pattern:
  //     // - Sign bit: 0
  //     // - Exponent: all 1s (0xFF)
  //     // - Mantissa: 0x400000 (quiet NaN flag)
  //     bits = 0x7FC00000;
  // }
  //  Normalized values (most common case)
  else {
    // Construct normalized float by shifting exponent into position:
    // - Exponent field: 8 bits (positions 30-23)
    // - Mantissa: 0 (implicit leading 1)
    // Value = 2^(x - 127)
    vBits = (uint32_t)pX << 23;
  }

  float vResult; // Final float value
                 // Safely reinterpret bit pattern as float without type-punning issues
  memcpy(&vResult, &vBits, sizeof(float));
  return vResult;
}

// Equal to ggml_e8m0_to_fp32/2
// Useful with MXFP4 quantization since the E0M2 values are doubled
static inline float ggml_e8m0_to_fp32_half(uint8_t pX) {
  uint32_t vBits;

  // For x < 2: use precomputed denormal patterns
  if (pX < 2) {
    // 0x00200000 = 2^(-128), 0x00400000 = 2^(-127)
    vBits = 0x00200000 << pX;
  }
  // For x >= 2: normalized exponent adjustment
  else {
    // 0.5 * 2^(x-127) = 2^(x-128) = normalized with exponent (x-1)
    vBits = (uint32_t)(pX - 1) << 23;
  }
  // Note: NaNs are not handled here

  float vResult;
  memcpy(&vResult, &vBits, sizeof(float));
  return vResult;
}

#define GGML_E8M0_TO_FP32(x) ggml_e8m0_to_fp32(x)
#define GGML_E8M0_TO_FP32_HALF(x) ggml_e8m0_to_fp32_half(x)

// UE4M3: unsigned, 4 exp bits (bias=7), 3 mantissa bits
// Returns value * 0.5 to match kvalues_mxfp4 convention (kvalues = 2 * E2M1_float)
static inline float ggml_ue4m3_to_fp32(uint8_t pX) {
  if (pX == 0 || pX == 0x7F) {
    return 0.0f;
  }
  int vExp = (pX >> 3) & 0xF;
  int vMan = pX & 0x7;
  float vRaw;
  if (vExp == 0) {
    vRaw = ldexpf((float)vMan, -9);
  } else {
    vRaw = ldexpf(1.0f + (float)vMan / 8.0f, vExp - 7);
  }
  return vRaw * 0.5f;
}

static inline uint8_t ggml_fp32_to_ue4m3(float pX) {
  if (!(pX > 0.0f)) {
    return 0;
  }
  if (pX > 448.0f) {
    pX = 448.0f;
  }
  uint32_t vBits;
  memcpy(&vBits, &pX, 4);
  int vFp32Exp = ((vBits >> 23) & 0xFF) - 127;
  int vFp32Man = (vBits >> 20) & 0x7;
  int vUe4m3Exp = vFp32Exp + 7;
  if (vUe4m3Exp <= 0) {
    // subnormal: value = man * 2^-9, man = round(x * 2^9)
    int vMan = (int)(pX * 512.0f + 0.5f);
    if (vMan > 7) {
      vMan = 7;
    }
    if (vMan < 1) {
      return 0;
    }
    return (uint8_t)vMan;
  }
  if (vUe4m3Exp >= 15) {
    return 0x7E;
  }
  int vRoundBit = (vBits >> 19) & 1;
  int vUe4m3Man = vFp32Man + vRoundBit;
  if (vUe4m3Man > 7) {
    vUe4m3Man = 0;
    vUe4m3Exp++;
    if (vUe4m3Exp >= 15) {
      return 0x7E;
    }
  }
  return (uint8_t)((vUe4m3Exp << 3) | vUe4m3Man);
}

/**
 * Converts brain16 to float32.
 *
 * The bfloat16 floating point format has the following structure:
 *
 *       ┌sign
 *       │
 *       │   ┌exponent
 *       │   │
 *       │   │      ┌mantissa
 *       │   │      │
 *       │┌──┴───┐┌─┴───┐
 *     0b0000000000000000 brain16
 *
 * Since bf16 has the same number of exponent bits as a 32bit float,
 * encoding and decoding numbers becomes relatively straightforward.
 *
 *       ┌sign
 *       │
 *       │   ┌exponent
 *       │   │
 *       │   │      ┌mantissa
 *       │   │      │
 *       │┌──┴───┐┌─┴───────────────────┐
 *     0b00000000000000000000000000000000 IEEE binary32
 *
 * For comparison, the standard fp16 format has fewer exponent bits.
 *
 *       ┌sign
 *       │
 *       │  ┌exponent
 *       │  │
 *       │  │    ┌mantissa
 *       │  │    │
 *       │┌─┴─┐┌─┴──────┐
 *     0b0000000000000000 IEEE binary16
 *
 * @see IEEE 754-2008
 */
static inline float ggml_compute_bf16_to_fp32(ggml_bf16_t pH) {
  union {
    float f;
    uint32_t i;
  } vU;
  vU.i = (uint32_t)pH.bits << 16;
  return vU.f;
}

/**
 * Converts float32 to brain16.
 *
 * This is binary identical with Google Brain float conversion.
 * Floats shall round to nearest even, and NANs shall be quiet.
 * Subnormals aren't flushed to zero, except perhaps when used.
 * This code should vectorize nicely if using modern compilers.
 */
static inline ggml_bf16_t ggml_compute_fp32_to_bf16(float pS) {
  ggml_bf16_t vH;
  union {
    float f;
    uint32_t i;
  } vU;
  vU.f = pS;
  if ((vU.i & 0x7fffffff) > 0x7f800000) { /* nan */
    vH.bits = (vU.i >> 16) | 64;          /* force to quiet */
    return vH;
  }
  vH.bits = (vU.i + (0x7fff + ((vU.i >> 16) & 1))) >> 16;
  return vH;
}

#define GGML_FP32_TO_BF16(x) ggml_compute_fp32_to_bf16(x)
#define GGML_BF16_TO_FP32(x) ggml_compute_bf16_to_fp32(x)

static inline int32_t ggml_node_get_use_count(const struct ggml_cgraph *pCgraph, int pNodeIdx) {
  const struct ggml_tensor *vNode = pCgraph->nodes[pNodeIdx];

  size_t vHashPos = ggml_hash_find(&pCgraph->visited_hash_set, vNode);
  if (!ggml_bitset_get(pCgraph->visited_hash_set.used, vHashPos)) {
    return 0;
  }
  return pCgraph->use_counts[vHashPos];
}

// return true if the node's results are only used by N other nodes
// and can be fused into their calculations.
static inline bool ggml_node_has_n_uses(const struct ggml_cgraph *pCgraph, int pNodeIdx, int32_t pNUses) {
  const struct ggml_tensor *vNode = pCgraph->nodes[pNodeIdx];

  // check the use count against how many we're replacing
  if (ggml_node_get_use_count(pCgraph, pNodeIdx) != pNUses) {
    return false;
  }

  // if node is a view, some other node might be using the intermediate result
  // via the view source.
  if (vNode->view_src) {
    return false;
  }

  // If the user requested output for the node, can't fuse
  if (vNode->flags & GGML_TENSOR_FLAG_OUTPUT) {
    return false;
  }

  return true;
}

// Returns true if nodes with indices { node_idxs } are the sequence of ggml_ops in ops[]
// and are fusable. Nodes are considered fusable according to this function if:
// - all nodes except the last have only one use and are not views/outputs (see ggml_node_has_N_uses).
// - all nodes except the last are a src of the following node.
// - all nodes are the same shape.
// TODO: Consider allowing GGML_OP_NONE nodes in between
static inline bool ggml_can_fuse_ext(const struct ggml_cgraph *pCgraph, const int *pNodeIdxs, const enum ggml_op *pOps,
                                     int pNumOps) {
  for (int vI = 0; vI < pNumOps; ++vI) {
    if (pNodeIdxs[vI] >= pCgraph->n_nodes) {
      return false;
    }

    struct ggml_tensor *vNode = pCgraph->nodes[pNodeIdxs[vI]];
    if (vNode->op != pOps[vI]) {
      return false;
    }
    if ((vNode->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
      return false;
    }
    if (vI < pNumOps - 1 && !ggml_node_has_n_uses(pCgraph, pNodeIdxs[vI], 1)) {
      return false;
    }
    if (vI > 0) {
      struct ggml_tensor *vPrev = pCgraph->nodes[pNodeIdxs[vI - 1]];
      if (vNode->src[0] != vPrev && vNode->src[1] != vPrev) {
        return false;
      }
      if (!ggml_are_same_shape(vNode, vPrev)) {
        return false;
      }
    }
  }
  return true;
}

// same as above, for sequential indices starting at node_idx
static inline bool ggml_can_fuse(const struct ggml_cgraph *pCgraph, int pNodeIdx, const enum ggml_op *pOps,
                                 int pNumOps) {
  assert(pNumOps < 32);

  if (pNodeIdx + pNumOps > pCgraph->n_nodes) {
    return false;
  }

  int vIdxs[32];
  for (int vI = 0; vI < pNumOps; ++vI) {
    vIdxs[vI] = pNodeIdx + vI;
  }

  return ggml_can_fuse_ext(pCgraph, vIdxs, pOps, pNumOps);
}

GGML_API bool ggml_can_fuse_subgraph_ext(const struct ggml_cgraph *pCgraph, const int *pNodeIdxs, int pCount,
                                         const enum ggml_op *pOps, const int *pOutputs, int pNumOutputs);

// Returns true if the subgraph formed by {node_idxs} can be fused
// checks whethers all nodes which are not part of outputs can be elided
// by checking if their num_uses are confined to the subgraph
static inline bool ggml_can_fuse_subgraph(const struct ggml_cgraph *pCgraph, int pNodeIdx, int pCount,
                                          const enum ggml_op *pOps, const int *pOutputs, int pNumOutputs) {
  GGML_ASSERT(pCount < 32);
  if (pNodeIdx + pCount > pCgraph->n_nodes) {
    return false;
  }

  int vIdxs[32];

  for (int vI = 0; vI < pCount; ++vI) {
    vIdxs[vI] = pNodeIdx + vI;
  }

  return ggml_can_fuse_subgraph_ext(pCgraph, vIdxs, pCount, pOps, pOutputs, pNumOutputs);
}

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
#include <array>
#include <initializer_list>
#include <vector>

// nicer C++ syntax for ggml_can_fuse
inline bool ggml_can_fuse(const struct ggml_cgraph *cgraph, int node_idx, std::initializer_list<enum ggml_op> ops) {
  return ggml_can_fuse(cgraph, node_idx, ops.begin(), (int)ops.size());
}

inline bool ggml_can_fuse_subgraph(const struct ggml_cgraph *cgraph, int start_idx,
                                   std::initializer_list<enum ggml_op> ops, std::initializer_list<int> outputs = {}) {
  return ggml_can_fuse_subgraph(cgraph, start_idx, ops.size(), ops.begin(), outputs.begin(), outputs.size());
}

// Return true if the edges in the graph match expectations.
inline bool ggml_check_edges(const struct ggml_cgraph *cgraph, int start_idx,
                             std::initializer_list<std::array<int, 3>> edges) {
  for (const auto &edge : edges) {
    int dst_node = edge[0];
    int src_idx = edge[1];
    int src_node = edge[2];
    if (cgraph->nodes[start_idx + dst_node]->src[src_idx] != cgraph->nodes[start_idx + src_node]) {
      return false;
    }
  }
  return true;
}

// expose GGUF internals for test code
GGML_API size_t gguf_type_size(enum gguf_type type);
GGML_API void gguf_write_to_buf(const struct gguf_context *ctx, std::vector<int8_t> &buf, bool only_meta);
#endif // __cplusplus
