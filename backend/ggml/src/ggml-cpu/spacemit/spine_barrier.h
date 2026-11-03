#pragma once

#include <atomic>
#include <cstdint>

#define SPINE_CACHE_LINE 64
#define SPINE_CACHE_ALIGN __attribute__((aligned(SPINE_CACHE_LINE)))

struct spine_barrier_t {
  SPINE_CACHE_ALIGN std::atomic<int64_t> pending_;
  SPINE_CACHE_ALIGN std::atomic<int64_t> rounds_;
  SPINE_CACHE_ALIGN int64_t total_;
};

inline void spine_barrier_wait(spine_barrier_t *pB) {
  auto vCurRound = pB->rounds_.load(std::memory_order_acquire);
  auto vCnt = --pB->pending_;
  if (vCnt == 0) {
    pB->pending_.store(pB->total_);
    pB->rounds_.store(vCurRound + 1);
  } else {
    while (vCurRound == pB->rounds_.load(std::memory_order_relaxed)) {
      __asm__ volatile("pause " ::: "memory");
    }
  }
}

inline void spine_barrier_init(spine_barrier_t *pB, int pNumBarriers, uint64_t pThreadCount) {
  for (int vI = 0; vI < pNumBarriers; vI++) {
    pB[vI].total_ = pThreadCount;
    pB[vI].pending_.store(pThreadCount);
    pB[vI].rounds_.store(0);
  }
}
