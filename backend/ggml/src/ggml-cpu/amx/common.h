#pragma once

#include "ggml.h"
#include "ggml-cpu-impl.h"

#include <algorithm>
#include <memory>
#include <type_traits>

#if defined(GGML_USE_OPENMP)
#include <omp.h>
#else
#include <thread>
#endif

#define TILE_M 16
#define TILE_N 16
#define TILE_K 32
#define VNNI_BLK 4

#define AMX_BLK_SIZE 32

#define TMM0 0
#define TMM1 1
#define TMM2 2
#define TMM3 3
#define TMM4 4
#define TMM5 5
#define TMM6 6
#define TMM7 7

// parallel routines
template <typename T, typename std::enable_if<std::is_integral<T>::value, int>::type = 0> inline T div_up(T pX, T pY) {
  return (pX + pY - 1) / pY;
}

template <typename T> inline void balance211(T pN, T pNth, T pIth, T &pNStart, T &pNEnd) {
#if 0
    // onednn partition pattern
    T& vNMy = pNEnd;
    if (pNth <= 1 || pN == 0) {
        pNStart = 0;
        vNMy = pN;
    } else {
        T n1 = div_up(pN, pNth);
        T n2 = n1 - 1;
        T T1 = pN - n2 * pNth;
        vNMy = pIth < T1 ? n1 : n2;
        pNStart = pIth <= T1 ? pIth*n1 : T1 * n1 + (pIth - T1) * n2;
    }
    pNEnd += pNStart;
#else
  // pytorch aten partition pattern
  T vNMy = div_up(pN, pNth);
  pNStart = pIth * vNMy;
  pNEnd = std::min(pNStart + vNMy, pN);
#endif
}

template <typename func_t> inline void parallel_for(int pN, const func_t &pF) {
  if (pN <= 0) {
    return;
  }
#if defined(GGML_USE_OPENMP)
#pragma omp parallel
  {
    int vNth = omp_get_num_threads();
    int vIth = omp_get_thread_num();
    int vTbegin, vTend;
    balance211(pN, vNth, vIth, vTbegin, vTend);
    pF(vTbegin, vTend);
  }
#else
  int vNth = std::thread::hardware_concurrency();
  if (vNth <= 1) {
    pF(0, pN);
    return;
  }
  if (vNth > pN) {
    vNth = pN;
  }
  std::vector<std::thread> threads;
  threads.reserve(vNth);
  for (int vIth = 0; vIth < vNth; ++vIth) {
    threads.emplace_back([&pF, pN, vIth, vNth] {
      int vTbegin, vTend;
      balance211(pN, vNth, vIth, vTbegin, vTend);
      pF(vTbegin, vTend);
    });
  }
  for (auto &t : threads) {
    t.join();
  }
#endif
}

template <typename func_t> inline void parallel_for_ggml(const ggml_compute_params *pParams, int pN, const func_t &pF) {
  int vTbegin, vTend;
  balance211(pN, pParams->nth, pParams->ith, vTbegin, vTend);
  pF(vTbegin, vTend);
}

// quantized types that have AMX support
inline bool qtype_has_amx_kernels(const enum ggml_type pType) {
  // TODO: fix padding for vnni format
  return (pType == GGML_TYPE_Q4_0) || (pType == GGML_TYPE_Q4_1) || (pType == GGML_TYPE_Q8_0) ||
         (pType == GGML_TYPE_Q4_K) || (pType == GGML_TYPE_Q5_K) || (pType == GGML_TYPE_Q6_K) ||
         (pType == GGML_TYPE_IQ4_XS);
}
