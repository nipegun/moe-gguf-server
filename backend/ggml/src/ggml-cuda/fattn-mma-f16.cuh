#include "common.cuh"
#include "cp-async.cuh"
#include "mma.cuh"
#include "fattn-common.cuh"

using namespace ggml_cuda_mma;

// Config options for the MMA kernel.
// Should not affect results, only speed/register pressure/shared memory use.
struct fattn_mma_config {
  int nthreads;       // Number of threads per CUDA block.
  int occupancy;      // Targeted occupancy for the MMA kernel.
  int nbatch_fa;      // Number of KV rows per softmax rescaling of KQ rowsums and VKQ accumulators.
  int nbatch_K2;      // Number of K half2 values in direction of DKQ to load in parallel.
  int nbatch_V2;      // Number of V half2 values in direction of DV to load in parallel.
  int nbatch_combine; // Number of VKQ half2 values in direction of DV to combine in parallel.
  int nstages_target; // Number of pipeline stages to use ideally, 1 == always load data synchronously, 2 == preload
                      // data if there is hardware support.
  bool Q_in_reg;      // Whether the Q values should be kept permanently in registers.

  constexpr __host__ __device__ fattn_mma_config(int pNthreads, int pOccupancy, int pNbatchFa, int pNbatchK2,
                                                 int pNbatchV2, int pNbatchCombine, int pNstagesTarget, bool pQInReg)
      : nthreads(pNthreads), occupancy(pOccupancy), nbatch_fa(pNbatchFa), nbatch_K2(pNbatchK2), nbatch_V2(pNbatchV2),
        nbatch_combine(pNbatchCombine), nstages_target(pNstagesTarget), Q_in_reg(pQInReg) {}
};

#define GGML_CUDA_FATTN_MMA_CONFIG_CASE(DKQ_, DV_, ncols_, nthreads_, occupancy_, nbatch_fa_, nbatch_K2_, nbatch_V2_,  \
                                        nbatch_combine_, nstages_target_, Q_in_reg_)                                   \
  if (DKQ == (DKQ_) && DV == (DV_) && ncols == (ncols_)) {                                                             \
    static_assert((nthreads_) % 32 == 0 && (nthreads_) <= 512, "bad nthreads");                                        \
    static_assert((occupancy_) <= 8, "bad occupancy");                                                                 \
    static_assert((nbatch_fa_) % 32 == 0 && (nbatch_fa_) <= 256, "bad nbatch_fa");                                     \
    static_assert((nbatch_K2_) % 4 == 0 && (nbatch_K2_) <= 512, "bad nbatch_K2");                                      \
    static_assert((nbatch_V2_) % 4 == 0 && (nbatch_V2_) <= 256, "bad nbatch_V2");                                      \
    static_assert((nbatch_combine_) % 4 == 0 && (nbatch_combine_) <= 128, "bad nbatch_combine");                       \
    static_assert((nstages_target_) >= 1 && (nstages_target_) <= 2, "bad nstages_target");                             \
    return fattn_mma_config{(nthreads_),  (occupancy_),      (nbatch_fa_),      (nbatch_K2_),                          \
                            (nbatch_V2_), (nbatch_combine_), (nstages_target_), (Q_in_reg_)};                          \
  }

static constexpr __host__ __device__ fattn_mma_config ggml_cuda_fattn_mma_get_config_ampere(const int DKQ, const int DV,
                                                                                            const int ncols) {
  GGML_CUDA_FATTN_MMA_CONFIG_CASE(64, 64, 8, 128, 2, 128, 32, 32, 32, 2, true);
  GGML_CUDA_FATTN_MMA_CONFIG_CASE(64, 64, 16, 128, 2, 64, 32, 32, 32, 2, true);
  GGML_CUDA_FATTN_MMA_CONFIG_CASE(64, 64, 32, 128, 2, 64, 32, 32, 32, 2, true);
  GGML_CUDA_FATTN_MMA_CONFIG_CASE(64, 64, 64, 128, 2, 64, 32, 32, 32, 2, true);

  GGML_CUDA_FATTN_MMA_CONFIG_CASE(80, 80, 8, 128, 2, 128, 40, 40, 40, 2, true);
  GGML_CUDA_FATTN_MMA_CONFIG_CASE(80, 80, 16, 128, 2, 64, 40, 40, 40, 2, true);
  GGML_CUDA_FATTN_MMA_CONFIG_CASE(80, 80, 32, 128, 2, 64, 40, 40, 40, 2, true);
  GGML_CUDA_FATTN_MMA_CONFIG_CASE(80, 80, 64, 128, 2, 64, 40, 40, 40, 2, true);

  GGML_CUDA_FATTN_MMA_CONFIG_CASE(96, 96, 8, 128, 2, 128, 48, 48, 48, 2, true);
  GGML_CUDA_FATTN_MMA_CONFIG_CASE(96, 96, 16, 128, 2, 64, 48, 48, 48, 2, true);
  GGML_CUDA_FATTN_MMA_CONFIG_CASE(96, 96, 32, 128, 2, 64, 48, 48, 48, 2, true);
  GGML_CUDA_FATTN_MMA_CONFIG_CASE(96, 96, 64, 128, 2, 64, 48, 48, 48, 2, true);

  GGML_CUDA_FATTN_MMA_CONFIG_CASE(112, 112, 8, 128, 2, 128, 56, 56, 56, 2, true);
  GGML_CUDA_FATTN_MMA_CONFIG_CASE(112, 112, 16, 128, 2, 64, 56, 56, 56, 2, true);
  GGML_CUDA_FATTN_MMA_CONFIG_CASE(112, 112, 32, 128, 2, 64, 56, 56, 56, 2, true);
  GGML_CUDA_FATTN_MMA_CONFIG_CASE(112, 112, 64, 128, 2, 64, 56, 56, 56, 2, true);

  GGML_CUDA_FATTN_MMA_CONFIG_CASE(128, 128, 8, 128, 2, 128, 64, 64, 64, 2, true);
  GGML_CUDA_FATTN_MMA_CONFIG_CASE(128, 128, 16, 128, 2, 64, 64, 64, 64, 2, true);
  GGML_CUDA_FATTN_MMA_CONFIG_CASE(128, 128, 32, 128, 2, 64, 64, 64, 64, 2, true);
  GGML_CUDA_FATTN_MMA_CONFIG_CASE(128, 128, 64, 128, 2, 64, 64, 64, 64, 2, true);

  GGML_CUDA_FATTN_MMA_CONFIG_CASE(192, 128, 8, 64, 4, 64, 96, 64, 64, 2, true);
  GGML_CUDA_FATTN_MMA_CONFIG_CASE(192, 128, 16, 64, 4, 32, 96, 64, 64, 2, true);
  GGML_CUDA_FATTN_MMA_CONFIG_CASE(192, 128, 32, 128, 2, 32, 96, 64, 64, 2, true);
  GGML_CUDA_FATTN_MMA_CONFIG_CASE(192, 128, 64, 128, 2, 32, 96, 64, 64, 2, true);

  GGML_CUDA_FATTN_MMA_CONFIG_CASE(256, 256, 8, 64, 4, 64, 128, 128, 128, 2, true);
  GGML_CUDA_FATTN_MMA_CONFIG_CASE(256, 256, 16, 64, 4, 32, 128, 128, 128, 2, true);
  GGML_CUDA_FATTN_MMA_CONFIG_CASE(256, 256, 32, 128, 2, 32, 128, 128, 128, 2, true);
  GGML_CUDA_FATTN_MMA_CONFIG_CASE(256, 256, 64, 128, 2, 32, 128, 128, 128, 2, true);

  GGML_CUDA_FATTN_MMA_CONFIG_CASE(320, 256, 32, 128, 2, 32, 128, 128, 128, 1, false);
  GGML_CUDA_FATTN_MMA_CONFIG_CASE(320, 256, 64, 256, 1, 32, 128, 128, 128, 1, false);

  GGML_CUDA_FATTN_MMA_CONFIG_CASE(512, 512, 8, 64, 4, 32, 256, 256, 128, 1, false);
  GGML_CUDA_FATTN_MMA_CONFIG_CASE(512, 512, 16, 64, 4, 32, 256, 256, 128, 1, false);
  GGML_CUDA_FATTN_MMA_CONFIG_CASE(512, 512, 32, 128, 2, 32, 128, 128, 128, 1, false);
  GGML_CUDA_FATTN_MMA_CONFIG_CASE(512, 512, 64, 256, 1, 32, 128, 128, 128, 1, false);

  GGML_CUDA_FATTN_MMA_CONFIG_CASE(576, 512, 8, 64, 4, 32, 288, 256, 128, 1, false);
  GGML_CUDA_FATTN_MMA_CONFIG_CASE(576, 512, 16, 64, 4, 32, 288, 256, 128, 1, false);
  GGML_CUDA_FATTN_MMA_CONFIG_CASE(576, 512, 32, 128, 2, 32, 160, 128, 128, 1, false);
  GGML_CUDA_FATTN_MMA_CONFIG_CASE(576, 512, 64, 256, 1, 32, 160, 128, 128, 1, false);

  return fattn_mma_config(32, 1, 0, 0, 0, 0, 0, false);
}

static constexpr __host__ __device__ fattn_mma_config ggml_cuda_fattn_mma_get_config_turing(const int DKQ, const int DV,
                                                                                            const int ncols) {
  GGML_CUDA_FATTN_MMA_CONFIG_CASE(256, 256, 8, 128, 2, 64, 128, 128, 128, 2, true);
  GGML_CUDA_FATTN_MMA_CONFIG_CASE(256, 256, 16, 128, 2, 64, 128, 128, 128, 2, true);
  GGML_CUDA_FATTN_MMA_CONFIG_CASE(256, 256, 32, 128, 2, 64, 128, 128, 64, 2, true);
  GGML_CUDA_FATTN_MMA_CONFIG_CASE(256, 256, 64, 128, 2, 64, 128, 128, 64, 2, true);

  GGML_CUDA_FATTN_MMA_CONFIG_CASE(320, 256, 32, 128, 2, 32, 128, 128, 128, 1, false);
  GGML_CUDA_FATTN_MMA_CONFIG_CASE(320, 256, 64, 256, 1, 32, 128, 128, 128, 1, false);

  GGML_CUDA_FATTN_MMA_CONFIG_CASE(512, 512, 8, 64, 4, 32, 96, 64, 128, 1, false);
  GGML_CUDA_FATTN_MMA_CONFIG_CASE(512, 512, 16, 64, 4, 32, 96, 64, 128, 1, false);
  GGML_CUDA_FATTN_MMA_CONFIG_CASE(512, 512, 32, 128, 2, 32, 128, 128, 128, 1, false);
  GGML_CUDA_FATTN_MMA_CONFIG_CASE(512, 512, 64, 256, 1, 32, 128, 128, 128, 1, false);

  GGML_CUDA_FATTN_MMA_CONFIG_CASE(576, 512, 8, 64, 4, 32, 96, 64, 128, 1, false);
  GGML_CUDA_FATTN_MMA_CONFIG_CASE(576, 512, 16, 64, 4, 32, 96, 64, 128, 1, false);
  GGML_CUDA_FATTN_MMA_CONFIG_CASE(576, 512, 32, 128, 2, 32, 160, 128, 128, 1, false);
  GGML_CUDA_FATTN_MMA_CONFIG_CASE(576, 512, 64, 256, 1, 32, 160, 128, 128, 1, false);

  return ggml_cuda_fattn_mma_get_config_ampere(DKQ, DV, ncols);
}

static constexpr __host__ __device__ fattn_mma_config ggml_cuda_fattn_mma_get_config_volta(const int DKQ, const int DV,
                                                                                           const int ncols) {
  GGML_CUDA_FATTN_MMA_CONFIG_CASE(512, 512, 8, 64, 4, 32, 256, 256, 64, 1, false);
  GGML_CUDA_FATTN_MMA_CONFIG_CASE(512, 512, 16, 64, 4, 32, 256, 256, 64, 1, false);
  GGML_CUDA_FATTN_MMA_CONFIG_CASE(512, 512, 32, 128, 2, 32, 128, 128, 64, 1, false);
  GGML_CUDA_FATTN_MMA_CONFIG_CASE(512, 512, 64, 256, 1, 32, 128, 128, 64, 1, false);

  GGML_CUDA_FATTN_MMA_CONFIG_CASE(576, 512, 8, 64, 4, 32, 288, 256, 64, 1, false);
  GGML_CUDA_FATTN_MMA_CONFIG_CASE(576, 512, 16, 64, 4, 32, 288, 256, 64, 1, false);
  GGML_CUDA_FATTN_MMA_CONFIG_CASE(576, 512, 32, 128, 2, 32, 160, 128, 64, 1, false);
  GGML_CUDA_FATTN_MMA_CONFIG_CASE(576, 512, 64, 256, 1, 32, 160, 128, 64, 1, false);

  // TODO tune specifically for Volta
  return ggml_cuda_fattn_mma_get_config_ampere(DKQ, DV, ncols);
}

static constexpr __host__ __device__ fattn_mma_config ggml_cuda_fattn_mma_get_config_rdna(const int DKQ, const int DV,
                                                                                          const int ncols) {
  GGML_CUDA_FATTN_MMA_CONFIG_CASE(64, 64, 8, 128, 2, 64, 32, 32, 32, 1, true);
  GGML_CUDA_FATTN_MMA_CONFIG_CASE(64, 64, 16, 128, 2, 64, 32, 32, 32, 1, true);
  GGML_CUDA_FATTN_MMA_CONFIG_CASE(64, 64, 32, 128, 2, 64, 32, 32, 32, 1, true);
  GGML_CUDA_FATTN_MMA_CONFIG_CASE(64, 64, 64, 128, 2, 64, 32, 32, 32, 1, true);

  GGML_CUDA_FATTN_MMA_CONFIG_CASE(80, 80, 8, 64, 2, 32, 40, 40, 40, 1, true);
  GGML_CUDA_FATTN_MMA_CONFIG_CASE(80, 80, 16, 64, 2, 32, 40, 40, 40, 1, true);
  GGML_CUDA_FATTN_MMA_CONFIG_CASE(80, 80, 32, 128, 2, 64, 40, 40, 40, 1, true);
  GGML_CUDA_FATTN_MMA_CONFIG_CASE(80, 80, 64, 128, 2, 64, 40, 40, 40, 1, true);

  GGML_CUDA_FATTN_MMA_CONFIG_CASE(96, 96, 8, 64, 2, 32, 48, 48, 48, 1, true);
  GGML_CUDA_FATTN_MMA_CONFIG_CASE(96, 96, 16, 64, 2, 32, 48, 48, 48, 1, true);
  GGML_CUDA_FATTN_MMA_CONFIG_CASE(96, 96, 32, 128, 2, 64, 48, 48, 48, 1, true);
  GGML_CUDA_FATTN_MMA_CONFIG_CASE(96, 96, 64, 128, 2, 64, 48, 48, 48, 1, true);

  GGML_CUDA_FATTN_MMA_CONFIG_CASE(112, 112, 8, 64, 2, 32, 56, 56, 56, 1, true);
  GGML_CUDA_FATTN_MMA_CONFIG_CASE(112, 112, 16, 64, 2, 32, 56, 56, 56, 1, true);
  GGML_CUDA_FATTN_MMA_CONFIG_CASE(112, 112, 32, 128, 2, 64, 56, 56, 56, 1, true);
  GGML_CUDA_FATTN_MMA_CONFIG_CASE(112, 112, 64, 128, 2, 64, 56, 56, 56, 1, true);

  GGML_CUDA_FATTN_MMA_CONFIG_CASE(128, 128, 8, 64, 2, 32, 64, 64, 64, 1, true);
  GGML_CUDA_FATTN_MMA_CONFIG_CASE(128, 128, 16, 64, 2, 32, 64, 64, 64, 1, true);
  GGML_CUDA_FATTN_MMA_CONFIG_CASE(128, 128, 32, 128, 2, 64, 64, 64, 64, 1, true);
  GGML_CUDA_FATTN_MMA_CONFIG_CASE(128, 128, 64, 128, 2, 64, 64, 64, 64, 1, true);

  GGML_CUDA_FATTN_MMA_CONFIG_CASE(192, 128, 8, 64, 2, 32, 96, 64, 64, 1, true);
  GGML_CUDA_FATTN_MMA_CONFIG_CASE(192, 128, 16, 64, 2, 32, 96, 64, 64, 1, true);
  GGML_CUDA_FATTN_MMA_CONFIG_CASE(192, 128, 32, 128, 2, 64, 96, 64, 64, 1, true);
  GGML_CUDA_FATTN_MMA_CONFIG_CASE(192, 128, 64, 128, 2, 64, 96, 64, 64, 1, true);

  GGML_CUDA_FATTN_MMA_CONFIG_CASE(256, 256, 8, 64, 2, 32, 128, 128, 128, 1, true);
  GGML_CUDA_FATTN_MMA_CONFIG_CASE(256, 256, 16, 64, 2, 32, 128, 128, 128, 1, true);
  GGML_CUDA_FATTN_MMA_CONFIG_CASE(256, 256, 32, 128, 2, 64, 128, 128, 64, 1, true);
  GGML_CUDA_FATTN_MMA_CONFIG_CASE(256, 256, 64, 128, 2, 64, 128, 128, 64, 1, true);

  GGML_CUDA_FATTN_MMA_CONFIG_CASE(320, 256, 32, 128, 2, 32, 160, 128, 128, 1, true);
  GGML_CUDA_FATTN_MMA_CONFIG_CASE(320, 256, 64, 128, 2, 32, 160, 128, 128, 1, true);

  GGML_CUDA_FATTN_MMA_CONFIG_CASE(512, 512, 8, 128, 3, 64, 96, 64, 128, 1, true);
  GGML_CUDA_FATTN_MMA_CONFIG_CASE(512, 512, 16, 128, 3, 64, 96, 64, 128, 1, true);
  GGML_CUDA_FATTN_MMA_CONFIG_CASE(512, 512, 32, 128, 2, 32, 128, 128, 128, 1, true);
  GGML_CUDA_FATTN_MMA_CONFIG_CASE(512, 512, 64, 128, 2, 32, 128, 128, 128, 1, true);

  GGML_CUDA_FATTN_MMA_CONFIG_CASE(576, 512, 8, 128, 3, 64, 96, 64, 128, 1, true);
  GGML_CUDA_FATTN_MMA_CONFIG_CASE(576, 512, 16, 128, 3, 64, 96, 64, 128, 1, true);
  GGML_CUDA_FATTN_MMA_CONFIG_CASE(576, 512, 32, 128, 2, 32, 160, 128, 128, 1, true);
  GGML_CUDA_FATTN_MMA_CONFIG_CASE(576, 512, 64, 128, 2, 32, 160, 128, 128, 1, true);

  return fattn_mma_config(32, 1, 0, 0, 0, 0, 0, false);
}

static constexpr __host__ __device__ fattn_mma_config ggml_cuda_fattn_mma_get_config_cdna(const int DKQ, const int DV,
                                                                                          const int ncols) {
  GGML_CUDA_FATTN_MMA_CONFIG_CASE(64, 64, 8, 128, 1, 64, 32, 32, 32, 1, true);
  GGML_CUDA_FATTN_MMA_CONFIG_CASE(64, 64, 16, 256, 2, 64, 32, 32, 32, 1, true);
  GGML_CUDA_FATTN_MMA_CONFIG_CASE(64, 64, 32, 256, 2, 64, 32, 32, 32, 1, true);
  GGML_CUDA_FATTN_MMA_CONFIG_CASE(64, 64, 64, 256, 4, 64, 32, 32, 32, 1, true);

  GGML_CUDA_FATTN_MMA_CONFIG_CASE(80, 80, 8, 256, 2, 64, 40, 40, 40, 1, true);
  GGML_CUDA_FATTN_MMA_CONFIG_CASE(80, 80, 16, 256, 2, 64, 40, 40, 40, 1, true);
  GGML_CUDA_FATTN_MMA_CONFIG_CASE(80, 80, 32, 256, 2, 64, 40, 40, 40, 1, true);
  GGML_CUDA_FATTN_MMA_CONFIG_CASE(80, 80, 64, 256, 2, 64, 40, 40, 40, 1, true);

  GGML_CUDA_FATTN_MMA_CONFIG_CASE(96, 96, 8, 256, 2, 64, 48, 48, 48, 1, true);
  GGML_CUDA_FATTN_MMA_CONFIG_CASE(96, 96, 16, 256, 2, 64, 48, 48, 48, 1, true);
  GGML_CUDA_FATTN_MMA_CONFIG_CASE(96, 96, 32, 256, 2, 64, 48, 48, 48, 1, true);
  GGML_CUDA_FATTN_MMA_CONFIG_CASE(96, 96, 64, 256, 2, 64, 48, 48, 48, 1, true);

  GGML_CUDA_FATTN_MMA_CONFIG_CASE(112, 112, 8, 256, 2, 64, 56, 56, 56, 1, true);
  GGML_CUDA_FATTN_MMA_CONFIG_CASE(112, 112, 16, 256, 2, 64, 56, 56, 56, 1, true);
  GGML_CUDA_FATTN_MMA_CONFIG_CASE(112, 112, 32, 256, 2, 64, 56, 56, 56, 1, true);
  GGML_CUDA_FATTN_MMA_CONFIG_CASE(112, 112, 64, 256, 2, 64, 56, 56, 56, 1, true);

  GGML_CUDA_FATTN_MMA_CONFIG_CASE(128, 128, 8, 256, 2, 64, 64, 64, 64, 1, true);
  GGML_CUDA_FATTN_MMA_CONFIG_CASE(128, 128, 16, 256, 2, 64, 64, 64, 64, 1, true);
  GGML_CUDA_FATTN_MMA_CONFIG_CASE(128, 128, 32, 256, 2, 64, 64, 64, 64, 1, true);
  GGML_CUDA_FATTN_MMA_CONFIG_CASE(128, 128, 64, 256, 2, 64, 64, 64, 64, 1, true);

  GGML_CUDA_FATTN_MMA_CONFIG_CASE(192, 128, 8, 256, 1, 64, 64, 64, 64, 1, true);
  GGML_CUDA_FATTN_MMA_CONFIG_CASE(192, 128, 16, 256, 1, 64, 64, 64, 64, 1, true);
  GGML_CUDA_FATTN_MMA_CONFIG_CASE(192, 128, 32, 256, 1, 64, 64, 64, 64, 1, true);
  GGML_CUDA_FATTN_MMA_CONFIG_CASE(192, 128, 64, 512, 1, 64, 64, 64, 64, 1, true);

  GGML_CUDA_FATTN_MMA_CONFIG_CASE(256, 256, 8, 256, 1, 64, 128, 128, 128, 1, true);
  GGML_CUDA_FATTN_MMA_CONFIG_CASE(256, 256, 16, 256, 1, 64, 128, 128, 128, 1, true);
  GGML_CUDA_FATTN_MMA_CONFIG_CASE(256, 256, 32, 256, 1, 64, 128, 128, 128, 1, true);
  GGML_CUDA_FATTN_MMA_CONFIG_CASE(256, 256, 64, 512, 1, 64, 128, 128, 64, 1, true);

  GGML_CUDA_FATTN_MMA_CONFIG_CASE(320, 256, 32, 256, 1, 64, 160, 128, 128, 1, true);
  GGML_CUDA_FATTN_MMA_CONFIG_CASE(320, 256, 64, 256, 1, 64, 160, 128, 128, 1, true);

  GGML_CUDA_FATTN_MMA_CONFIG_CASE(512, 512, 8, 256, 1, 64, 128, 128, 128, 1, true);
  GGML_CUDA_FATTN_MMA_CONFIG_CASE(512, 512, 16, 256, 1, 64, 128, 128, 128, 1, true);
  GGML_CUDA_FATTN_MMA_CONFIG_CASE(512, 512, 32, 256, 1, 64, 128, 128, 128, 1, true);
  GGML_CUDA_FATTN_MMA_CONFIG_CASE(512, 512, 64, 256, 1, 64, 128, 128, 128, 1, true);

  GGML_CUDA_FATTN_MMA_CONFIG_CASE(576, 512, 8, 256, 1, 64, 128, 128, 128, 1, true);
  GGML_CUDA_FATTN_MMA_CONFIG_CASE(576, 512, 16, 256, 1, 64, 128, 128, 128, 1, true);
  GGML_CUDA_FATTN_MMA_CONFIG_CASE(576, 512, 32, 256, 1, 64, 160, 128, 128, 1, true);
  GGML_CUDA_FATTN_MMA_CONFIG_CASE(576, 512, 64, 256, 1, 64, 160, 128, 128, 1, true);

  return fattn_mma_config(32, 1, 0, 0, 0, 0, 0, false);
}

static __host__ fattn_mma_config ggml_cuda_fattn_mma_get_config(const int pDkq, const int pDv, const int pNcols,
                                                                const int pCc) {
  if (ampere_mma_available(pCc)) {
    return ggml_cuda_fattn_mma_get_config_ampere(pDkq, pDv, pNcols);
  }
  if (turing_mma_available(pCc)) {
    return ggml_cuda_fattn_mma_get_config_turing(pDkq, pDv, pNcols);
  }
  if (amd_mfma_available(pCc)) {
    return ggml_cuda_fattn_mma_get_config_cdna(pDkq, pDv, pNcols);
  }
  if (amd_wmma_available(pCc)) {
    return ggml_cuda_fattn_mma_get_config_rdna(pDkq, pDv, pNcols);
  }
  GGML_ASSERT(volta_mma_available(pCc));
  return ggml_cuda_fattn_mma_get_config_volta(pDkq, pDv, pNcols);
}

static constexpr __device__ fattn_mma_config ggml_cuda_fattn_mma_get_config(const int pDkq, const int pDv,
                                                                            const int pNcols) {
#if defined(AMPERE_MMA_AVAILABLE)
  return ggml_cuda_fattn_mma_get_config_ampere(pDkq, pDv, pNcols);
#elif defined(TURING_MMA_AVAILABLE)
  return ggml_cuda_fattn_mma_get_config_turing(pDkq, pDv, pNcols);
#elif defined(AMD_MFMA_AVAILABLE)
  return ggml_cuda_fattn_mma_get_config_cdna(pDkq, pDv, pNcols);
#elif defined(VOLTA_MMA_AVAILABLE)
  return ggml_cuda_fattn_mma_get_config_volta(pDkq, pDv, pNcols);
#elif defined(AMD_WMMA_AVAILABLE)
  return ggml_cuda_fattn_mma_get_config_rdna(pDkq, pDv, pNcols);
#else
  GGML_UNUSED_VARS(pDkq, pDv, pNcols);
  return fattn_mma_config(32, 1, 0, 0, 0, 0, 0, false);
#endif // defined(AMPERE_MMA_AVAILABLE)
}

static __host__ int ggml_cuda_fattn_mma_get_nthreads(const int pDkq, const int pDv, const int pNcols, const int pCc) {
  return ggml_cuda_fattn_mma_get_config(pDkq, pDv, pNcols, pCc).nthreads;
}

static constexpr __device__ int ggml_cuda_fattn_mma_get_nthreads(const int pDkq, const int pDv, const int pNcols) {
  return ggml_cuda_fattn_mma_get_config(pDkq, pDv, pNcols).nthreads;
}

static __host__ int ggml_cuda_fattn_mma_get_occupancy(const int pDkq, const int pDv, const int pNcols, const int pCc) {
  return ggml_cuda_fattn_mma_get_config(pDkq, pDv, pNcols, pCc).occupancy;
}

static constexpr __device__ int ggml_cuda_fattn_mma_get_occupancy(const int pDkq, const int pDv, const int pNcols) {
  return ggml_cuda_fattn_mma_get_config(pDkq, pDv, pNcols).occupancy;
}

static __host__ int ggml_cuda_fattn_mma_get_nbatch_fa(const int pDkq, const int pDv, const int pNcols, const int pCc) {
  return ggml_cuda_fattn_mma_get_config(pDkq, pDv, pNcols, pCc).nbatch_fa;
}

static constexpr __device__ int ggml_cuda_fattn_mma_get_nbatch_fa(const int pDkq, const int pDv, const int pNcols) {
  return ggml_cuda_fattn_mma_get_config(pDkq, pDv, pNcols).nbatch_fa;
}

static __host__ int ggml_cuda_fattn_mma_get_nbatch_K2(const int pDkq, const int pDv, const int pNcols, const int pCc) {
  return ggml_cuda_fattn_mma_get_config(pDkq, pDv, pNcols, pCc).nbatch_K2;
}

static constexpr __device__ int ggml_cuda_fattn_mma_get_nbatch_K2(const int pDkq, const int pDv, const int pNcols) {
  return ggml_cuda_fattn_mma_get_config(pDkq, pDv, pNcols).nbatch_K2;
}

static __host__ int ggml_cuda_fattn_mma_get_nbatch_V2(const int pDkq, const int pDv, const int pNcols, const int pCc) {
  return ggml_cuda_fattn_mma_get_config(pDkq, pDv, pNcols, pCc).nbatch_V2;
}

static constexpr __device__ int ggml_cuda_fattn_mma_get_nbatch_V2(const int pDkq, const int pDv, const int pNcols) {
  return ggml_cuda_fattn_mma_get_config(pDkq, pDv, pNcols).nbatch_V2;
}

static __host__ int ggml_cuda_fattn_mma_get_nbatch_combine(const int pDkq, const int pDv, const int pNcols,
                                                           const int pCc) {
  return ggml_cuda_fattn_mma_get_config(pDkq, pDv, pNcols, pCc).nbatch_combine;
}

static constexpr __device__ int ggml_cuda_fattn_mma_get_nbatch_combine(const int pDkq, const int pDv,
                                                                       const int pNcols) {
  return ggml_cuda_fattn_mma_get_config(pDkq, pDv, pNcols).nbatch_combine;
}

static __host__ int ggml_cuda_fattn_mma_get_nstages_target(const int pDkq, const int pDv, const int pNcols,
                                                           const int pCc) {
  return ggml_cuda_fattn_mma_get_config(pDkq, pDv, pNcols, pCc).nstages_target;
}

static constexpr __device__ int ggml_cuda_fattn_mma_get_nstages_target(const int pDkq, const int pDv,
                                                                       const int pNcols) {
  return ggml_cuda_fattn_mma_get_config(pDkq, pDv, pNcols).nstages_target;
}

static __host__ bool ggml_cuda_fattn_mma_get_Q_in_reg(const int pDkq, const int pDv, const int pNcols, const int pCc) {
  return ggml_cuda_fattn_mma_get_config(pDkq, pDv, pNcols, pCc).Q_in_reg;
}

static constexpr __device__ bool ggml_cuda_fattn_mma_get_Q_in_reg(const int pDkq, const int pDv, const int pNcols) {
  return ggml_cuda_fattn_mma_get_config(pDkq, pDv, pNcols).Q_in_reg;
}

static constexpr __device__ int get_cols_per_thread() {
#if defined(AMD_WMMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE)
  return 1; // AMD has a single column per thread.
#else
  return 2; // This is specifically KQ columns, Volta only has a single VKQ column.
#endif // defined(AMD_WMMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE)
}

static __host__ int get_cols_per_warp(const int pCc) {
  if (turing_mma_available(pCc) || amd_wmma_available(pCc) || amd_mfma_available(pCc)) {
    return 16;
  } else {
    // Volta
    return 32;
  }
}

// ------------------------------------------------------------------------------------------------------------------

static __host__ int ggml_cuda_fattn_mma_get_nstages(const int pDkq, const int pDv, const int pNcols1, const int pNcols2,
                                                    const int pCc) {
  return cp_async_available(pCc) && pNcols2 >= 2
             ? ggml_cuda_fattn_mma_get_nstages_target(pDkq, pDv, pNcols1 * pNcols2, pCc)
             : 0;
}

static constexpr __device__ int ggml_cuda_fattn_mma_get_nstages(const int pDkq, const int pDv, const int pNcols1,
                                                                const int pNcols2) {
#ifdef CP_ASYNC_AVAILABLE
  return pNcols2 >= 2 ? ggml_cuda_fattn_mma_get_nstages_target(pDkq, pDv, pNcols1 * pNcols2) : 0;
#else
  GGML_UNUSED_VARS(pDkq, pDv, pNcols1, pNcols2);
  return 0;
#endif // CP_ASYNC_AVAILABLE
}

// ------------------------------------------------------------------------------------------------------------------

template <int stride_tile, int nwarps, int nbatch_fa, bool use_cp_async, bool oob_check>
static __device__ __forceinline__ void flash_attn_ext_f16_load_tile(const half2 *const __restrict__ pKv,
                                                                    half2 *const __restrict__ pTileKv, const int pD2,
                                                                    const int pStrideKv, const int pISup) {
  constexpr int cWarpSize = ggml_cuda_get_physical_warp_size();
  // K/V data is loaded with decreasing granularity for D for better memory bandwidth.
  // The minimum granularity is 16 bytes.
  constexpr int cH2PerChunk = 16 / sizeof(half2);
  const int cChunksPerRow = pD2 / cH2PerChunk;
  if constexpr (use_cp_async) {
    static_assert(cWarpSize == 32, "bad warp_size");
    static_assert(!oob_check, "OOB check not compatible with cp_async");
    constexpr int cPreload = 64;

    const unsigned int cTileKv32 = ggml_cuda_cvta_generic_to_shared(pTileKv);

    auto vLoad = [&] __device__(auto pN) {
      const int cStrideK = cWarpSize >> pN;
      const int cK0Start = cStrideK == cWarpSize ? 0 : cChunksPerRow - cChunksPerRow % (2 * cStrideK);
      const int cK0Stop = cChunksPerRow - cChunksPerRow % (1 * cStrideK);
      const int cStrideI = cWarpSize / cStrideK;

      if (cK0Start == cK0Stop) {
        return;
      }

#pragma unroll
      for (int vI0 = 0; vI0 < nbatch_fa; vI0 += nwarps * cStrideI) {
        const int cI = vI0 + threadIdx.y * cStrideI + (cStrideK == cWarpSize ? 0 : threadIdx.x / cStrideK);

        if (vI0 + nwarps * cStrideI > nbatch_fa && cI >= nbatch_fa) {
          break;
        }

#pragma unroll
        for (int vK0 = cK0Start; vK0 < cK0Stop; vK0 += cStrideK) {
          const int cK = vK0 + (cStrideK == cWarpSize ? threadIdx.x : threadIdx.x % cStrideK);

          cp_async_cg_16<cPreload>(cTileKv32 + cI * (stride_tile * sizeof(half2)) + cK * 16,
                                   pKv + cI * pStrideKv + cK * cH2PerChunk);
        }
      }
    };
    // 1: max 32*16=512 bytes, 256 half
    // 2: max 16*16=256 bytes, 128 half
    // 3: max  8*16=128 bytes,  64 half
    // 4: max  4*16= 64 bytes,  32 half
    // 5: max  2*16= 32 bytes,  16 half
    // 6: max  1*16= 16 bytes,   8 half
    ggml_cuda_unroll<6>{}(vLoad);
  } else {
    const half2 cZero[4] = {{0.0f, 0.0f}, {0.0f, 0.0f}, {0.0f, 0.0f}, {0.0f, 0.0f}};
    auto load = [&] __device__(const int n) {
      const int stride_k = 32 >> n;
      const int k0_start = stride_k == 32 ? 0 : cChunksPerRow - cChunksPerRow % (2 * stride_k);
      const int k0_stop = cChunksPerRow - cChunksPerRow % (1 * stride_k);
      const int stride_i = cWarpSize / stride_k;

      if (k0_start == k0_stop) {
        return;
      }

#pragma unroll
      for (int i0 = 0; i0 < nbatch_fa; i0 += nwarps * stride_i) {
        const int i = i0 + threadIdx.y * stride_i + (stride_k == cWarpSize ? 0 : threadIdx.x / stride_k);

        if (i0 + nwarps * stride_i > nbatch_fa && i >= nbatch_fa) {
          break;
        }

#pragma unroll
        for (int k0 = k0_start; k0 < k0_stop; k0 += stride_k) {
          const int k = k0 + (stride_k == cWarpSize ? threadIdx.x : threadIdx.x % stride_k);

          ggml_cuda_memcpy_1<16>(pTileKv + i * stride_tile + k * 4,
                                 !oob_check || i < pISup ? pKv + i * pStrideKv + k * cH2PerChunk : cZero);
        }
      }
    };
    // 1: max 32*16=512 bytes, 256 half
    // 2: max 16*16=256 bytes, 128 half
    // 3: max  8*16=128 bytes,  64 half
    // 4: max  4*16= 64 bytes,  32 half
    // 5: max  2*16= 32 bytes,  16 half
    // 6: max  1*16= 16 bytes,   8 half
    ggml_cuda_unroll<6>{}(load);
  }
}

template <int ncols1, int nwarps, int nbatch_fa, bool use_cp_async, bool oob_check>
static __device__ __forceinline__ void
flash_attn_ext_f16_load_mask(const half *const __restrict__ pMaskH, half *const __restrict__ pTileMask,
                             const int pStrideMask, const int pISup, const int pJ0, const uint3 pNe01) {
  constexpr int cWarpSize = ggml_cuda_get_physical_warp_size();
  if constexpr (use_cp_async) {
    static_assert(nbatch_fa <= 8 * cWarpSize && nbatch_fa % 8 == 0, "bad nbatch_fa");
    static_assert(!oob_check, "OOB check incompatible with cp_async");
    constexpr int cPreload = nbatch_fa >= 32 ? nbatch_fa * sizeof(half) : 64;
    constexpr int cColsPerWarp = 8 * cWarpSize / nbatch_fa;
    constexpr int cStrideJ = nwarps * cColsPerWarp;

    const unsigned int cTileMask32 = ggml_cuda_cvta_generic_to_shared(pTileMask);

#pragma unroll
    for (int vJ1 = 0; vJ1 < ncols1; vJ1 += cStrideJ) {
      const int cJSram = vJ1 + threadIdx.y * cColsPerWarp + threadIdx.x / (cWarpSize / cColsPerWarp);
      const int cJVram = fastmodulo(pJ0 + cJSram, pNe01);

      if (vJ1 + cStrideJ > ncols1 && cJSram >= ncols1) {
        break;
      }

      const int cI = 8 * (threadIdx.x % (nbatch_fa / 8));

      cp_async_cg_16<cPreload>(cTileMask32 + cJSram * (nbatch_fa * sizeof(half) + 16) + cI * sizeof(half),
                               pMaskH + int64_t(cJVram) * pStrideMask + cI);
    }
  } else if constexpr (oob_check) {
#pragma unroll
    for (int j1 = 0; j1 < ncols1; j1 += nwarps) {
      const int j_sram = j1 + threadIdx.y;
      const int j_vram = fastmodulo(pJ0 + j_sram, pNe01);

      if (j1 + nwarps > ncols1 && j_sram >= ncols1) {
        break;
      }

#pragma unroll
      for (int vI0 = 0; vI0 < nbatch_fa; vI0 += cWarpSize) {
        const int i = vI0 + threadIdx.x;

        pTileMask[j_sram * (nbatch_fa + 8) + i] = i < pISup ? pMaskH[int64_t(j_vram) * pStrideMask + i] : half(0.0f);
      }
    }
  } else if constexpr (nbatch_fa < 2 * cWarpSize) {
    constexpr int cols_per_warp = 2 * cWarpSize / nbatch_fa;
    constexpr int stride_j = nwarps * cols_per_warp;
#pragma unroll
    for (int j1 = 0; j1 < ncols1; j1 += stride_j) {
      const int j_sram = j1 + threadIdx.y * cols_per_warp + threadIdx.x / (cWarpSize / cols_per_warp);
      const int j_vram = fastmodulo(pJ0 + j_sram, pNe01);

      if (j1 + stride_j > ncols1 && j_sram >= ncols1) {
        break;
      }

      const int i = threadIdx.x % (cWarpSize / cols_per_warp);

      ggml_cuda_memcpy_1<sizeof(half2)>(pTileMask + j_sram * (nbatch_fa + 8) + 2 * i,
                                        pMaskH + int64_t(j_vram) * pStrideMask + 2 * i);
    }
  } else {
#pragma unroll
    for (int j1 = 0; j1 < ncols1; j1 += nwarps) {
      const int j_sram = j1 + threadIdx.y;
      const int j_vram = fastmodulo(pJ0 + j_sram, pNe01);

      if (j1 + nwarps > ncols1 && j_sram >= ncols1) {
        break;
      }

#pragma unroll
      for (int vI0 = 0; vI0 < nbatch_fa; vI0 += 2 * cWarpSize) {
        const int i = vI0 + 2 * threadIdx.x;

        ggml_cuda_memcpy_1<sizeof(half2)>(pTileMask + j_sram * (nbatch_fa + 8) + i,
                                          pMaskH + int64_t(j_vram) * pStrideMask + i);
      }
    }
  }
}

template <int DKQ, int DV, int ncols1, int ncols2, int nwarps, bool use_logit_softcap, bool V_is_K_view,
          bool needs_fixup, bool is_fixup, bool last_iter, bool oob_check, typename T_A_KQ, typename T_B_KQ,
          typename T_C_KQ, typename T_A_VKQ, typename T_B_VKQ, typename T_C_VKQ>
static __device__ __forceinline__ void flash_attn_ext_f16_iter(
    const float2 *const __restrict__ pQF2, const half2 *const __restrict__ pKH2, const half2 *const __restrict__ pVH2,
    const half *const __restrict__ pMaskH, float2 *const __restrict__ pDstk, float2 *const __restrict__ pDstkFixup,
    const float pScale, const float pSlope, const float pLogitSoftcap, const uint3 pNe01, const int pNe02,
    const int pStrideK, const int pStrideV, const int pStrideMask, half2 *const __restrict__ pTileQ,
    half2 *const __restrict__ pTileK, half2 *const __restrict__ pTileV, half *const __restrict__ pTileMask,
    T_B_KQ *const __restrict__ pQB, T_C_VKQ *const __restrict__ pVkqC, float *const __restrict__ pKqMax,
    float *const __restrict__ pKqRowsum, const int pJt, const int pKb0, const int pKVkqSup) {
#if defined(VOLTA_MMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE) ||                    \
    defined(AMD_MFMA_AVAILABLE)
  constexpr int cWarpSize = ggml_cuda_get_physical_warp_size();
  constexpr int cNcols = ncols1 * ncols2;
  constexpr int cColsPerWarp = T_B_KQ::I;
  constexpr int cColsPerThread = get_cols_per_thread();
  constexpr int cNp =
      cColsPerWarp > cNcols ? nwarps : nwarps * cColsPerWarp / cNcols; // Number of parallel CUDA warps per Q column.
  constexpr int cNbatchFa = ggml_cuda_fattn_mma_get_nbatch_fa(DKQ, DV, cNcols);
  constexpr int cNbatchK2 = ggml_cuda_fattn_mma_get_nbatch_K2(DKQ, DV, cNcols);
  constexpr int cNbatchV2 = ggml_cuda_fattn_mma_get_nbatch_V2(DKQ, DV, cNcols);
  constexpr bool cQInReg = ggml_cuda_fattn_mma_get_Q_in_reg(DKQ, DV, cNcols);
  constexpr int cNstages = ggml_cuda_fattn_mma_get_nstages(DKQ, DV, ncols1, ncols2);

  constexpr int cStrideTileK = cNbatchK2 + 4;

  constexpr int cStrideTileV = V_is_K_view ? cStrideTileK : cNbatchV2 + 4;

  const int cKVkq0 = pKb0 * cNbatchFa;
#if defined(TURING_MMA_AVAILABLE)
  T_C_KQ vKqC[cNbatchFa / (cNp * (cColsPerWarp == 8 ? T_C_KQ::I : T_C_KQ::J))];
#elif defined(AMD_WMMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE)
  T_C_KQ vKqC[cNbatchFa / (cNp * T_C_KQ::J)];
#else  // Volta
  T_C_KQ vKqC[cNbatchFa / (cNp * T_C_KQ::J)];
#endif // defined(TURING_MMA_AVAILABLE)

  if constexpr (cNstages > 1) {
    static_assert(!oob_check, "OOB check incompatible with multi-stage pipeline");
    static_assert(!V_is_K_view, "K data reuse not implemented multi-stage loading");
    static_assert(cNbatchK2 == DKQ / 2, "batching not implemented for multi stage loading");
    constexpr bool use_cp_async = true;
    cp_async_wait_all();
    __syncthreads();
    flash_attn_ext_f16_load_tile<cStrideTileV, nwarps, cNbatchFa, use_cp_async, oob_check>(
        pVH2 + int64_t(cKVkq0) * pStrideV, pTileV, cNbatchV2, pStrideV, pKVkqSup);
  } else {
    constexpr bool cUseCpAsync = cNstages == 1;
    if (ncols2 > 1 || pMaskH) {
      flash_attn_ext_f16_load_mask<ncols1, nwarps, cNbatchFa, cUseCpAsync, oob_check>(
          pMaskH + cKVkq0, pTileMask, pStrideMask, pKVkqSup, pJt * ncols1, pNe01);
    }
  }

  // For MLA K and V have the same data.
  // Therefore, iterate over K in reverse and later re-use the data if possible.
#pragma unroll
  for (int vK0Start = (DKQ / 2 - 1) - (DKQ / 2 - 1) % cNbatchK2; vK0Start >= 0; vK0Start -= cNbatchK2) {
    const int cK0Stop = vK0Start + cNbatchK2 < DKQ / 2 ? vK0Start + cNbatchK2 : DKQ / 2;

    if constexpr (cNstages <= 1) {
      const int cK0Diff = cK0Stop - vK0Start;
      constexpr bool cUseCpAsync = cNstages == 1;
      flash_attn_ext_f16_load_tile<cStrideTileK, nwarps, cNbatchFa, cUseCpAsync, oob_check>(
          pKH2 + int64_t(cKVkq0) * pStrideK + vK0Start, pTileK, cK0Diff, pStrideK, pKVkqSup);
      if (cUseCpAsync) {
        cp_async_wait_all();
      }
      __syncthreads();
    }

    // Calculate tile of KQ:
    if constexpr (cQInReg) {
#pragma unroll
      for (int i_KQ_00 = 0; i_KQ_00 < cNbatchFa; i_KQ_00 += cNp * T_A_KQ::I) {
        const int i_KQ_0 = i_KQ_00 + (threadIdx.y % cNp) * T_A_KQ::I;
#pragma unroll
        for (int k_KQ_0 = vK0Start; k_KQ_0 < cK0Stop; k_KQ_0 += T_A_KQ::J) {
          T_A_KQ K_A;
          load_ldmatrix(K_A, pTileK + i_KQ_0 * cStrideTileK + (k_KQ_0 - vK0Start), cStrideTileK);
          if constexpr (cColsPerWarp == 8) {
            mma(vKqC[i_KQ_00 / (cNp * T_A_KQ::I)], K_A, pQB[k_KQ_0 / T_A_KQ::J]);
          } else {
            // Wide version of KQ_C is column-major
#if defined(AMD_WMMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE)
            // AMD matrix C is column-major.
            mma(vKqC[i_KQ_00 / (cNp * T_A_KQ::I)], K_A, pQB[k_KQ_0 / T_A_KQ::J]);
#else
            // swap A and B for CUDA.
            mma(vKqC[i_KQ_00 / (cNp * T_A_KQ::I)], pQB[k_KQ_0 / T_A_KQ::J], K_A);
#endif // defined(AMD_WMMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE)
          }
        }
      }
    } else {
      constexpr int cStrideTileQ = DKQ / 2 + 4;
#pragma unroll
      for (int k_KQ_0 = vK0Start; k_KQ_0 < cK0Stop; k_KQ_0 += T_A_KQ::J) {
        load_ldmatrix(pQB[0], pTileQ + (threadIdx.y / cNp) * (T_B_KQ::I * cStrideTileQ) + k_KQ_0, cStrideTileQ);

#pragma unroll
        for (int i_KQ_00 = 0; i_KQ_00 < cNbatchFa; i_KQ_00 += cNp * T_A_KQ::I) {
          const int cIKq0 = i_KQ_00 + (threadIdx.y % cNp) * T_A_KQ::I;

          T_A_KQ K_A;
          load_ldmatrix(K_A, pTileK + cIKq0 * cStrideTileK + (k_KQ_0 - vK0Start), cStrideTileK);

          if constexpr (cColsPerWarp == 8) {
            mma(vKqC[i_KQ_00 / (cNp * T_A_KQ::I)], K_A, pQB[0]);
          } else {
            // Wide version of KQ_C is column-major
#if defined(AMD_WMMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE)
            // AMD matrix C is column-major.
            mma(vKqC[i_KQ_00 / (cNp * T_A_KQ::I)], K_A, pQB[0]);
#else
            // swap A and B for CUDA.
            mma(vKqC[i_KQ_00 / (cNp * T_A_KQ::I)], pQB[0], K_A);
#endif // defined(AMD_WMMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE)
          }
        }
      }
    }

    if constexpr (cNstages <= 1) {
      __syncthreads(); // Only needed if tile_K == tile_V.
    }
  }

  if (use_logit_softcap) {
    constexpr int cStride = cColsPerWarp == 8 ? cNp * T_C_KQ::I : cNp * T_C_KQ::J;
    static_assert(cNbatchFa % cStride == 0, "bad loop size");
#pragma unroll
    for (int i = 0; i < cNbatchFa / cStride; ++i) {
#pragma unroll
      for (int l = 0; l < T_C_KQ::ne; ++l) {
        vKqC[i].x[l] = pLogitSoftcap * tanhf(vKqC[i].x[l]);
      }
    }
  }

  float vKqMaxNew[cColsPerThread];
#pragma unroll
  for (int vCol = 0; vCol < cColsPerThread; ++vCol) {
    vKqMaxNew[vCol] = pKqMax[vCol];
  }
  float vKqRowsumAdd[cColsPerThread] = {0.0f};

  if constexpr (cColsPerWarp == 8) {
    if (ncols2 > 1 || pMaskH) {
#pragma unroll
      for (int i00 = 0; i00 < cNbatchFa; i00 += cNp * T_C_KQ::I) {
        const int i0 = i00 + (threadIdx.y % cNp) * T_C_KQ::I;
#pragma unroll
        for (int l = 0; l < T_C_KQ::ne; ++l) {
          const int i = i0 + T_C_KQ::get_i(l);
          const int j = ((threadIdx.y / cNp) * T_C_KQ::J + T_C_KQ::get_j(l)) / ncols2;

          vKqC[i00 / (cNp * T_C_KQ::I)].x[l] += pSlope * __half2float(pTileMask[j * (cNbatchFa + 8) + i]);
        }
      }
    }

    // Calculate softmax for each KQ column using the current max. value.
    // The divisor is stored in KQ_rowsum and will be applied at the end.
    static_assert(cNbatchFa % (cNp * T_C_KQ::I) == 0, "bad loop size");
#pragma unroll
    for (int k0 = 0; k0 < cNbatchFa; k0 += cNp * T_C_KQ::I) {
#pragma unroll
      for (int l = 0; l < T_C_KQ::ne; ++l) {
        if (!oob_check || k0 + (threadIdx.y % cNp) * T_C_KQ::I + T_C_KQ::get_i(l) < pKVkqSup) {
#if defined(AMD_WMMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE)
          constexpr int KQ_idx = 0;
#else
          // Turing + Volta:
          const int KQ_idx = l % 2;
#endif // defined(AMD_WMMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE)
          vKqMaxNew[KQ_idx] = fmaxf(vKqMaxNew[KQ_idx], vKqC[k0 / (cNp * T_C_KQ::I)].x[l] + FATTN_KQ_MAX_OFFSET);
        }
      }
    }

    // Values per KQ column are spread across 8 threads:
#pragma unroll
    for (int vCol = 0; vCol < cColsPerThread; ++vCol) {
#pragma unroll
      for (int vOffset = 16; vOffset >= 4; vOffset >>= 1) {
        vKqMaxNew[vCol] = fmaxf(vKqMaxNew[vCol], __shfl_xor_sync(0xFFFFFFFF, vKqMaxNew[vCol], vOffset, cWarpSize));
      }
    }

    static_assert(cNbatchFa % (cNp * T_C_KQ::I) == 0, "bad loop size");
#pragma unroll
    for (int k0 = 0; k0 < cNbatchFa; k0 += cNp * T_C_KQ::I) {
#pragma unroll
      for (int l = 0; l < T_C_KQ::ne; ++l) {
        if (!oob_check || k0 + (threadIdx.y % cNp) * T_C_KQ::I + T_C_KQ::get_i(l) < pKVkqSup) {
#if defined(AMD_WMMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE)
          constexpr int KQ_idx = 0;
#else
          // Turing + Volta:
          const int KQ_idx = l % 2;
#endif // defined(AMD_WMMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE)
          vKqC[k0 / (cNp * T_C_KQ::I)].x[l] = expf(vKqC[k0 / (cNp * T_C_KQ::I)].x[l] - vKqMaxNew[KQ_idx]);
          vKqRowsumAdd[KQ_idx] += vKqC[k0 / (cNp * T_C_KQ::I)].x[l];
        } else {
          vKqC[k0 / (cNp * T_C_KQ::I)].x[l] = 0.0f;
        }
      }
    }
  } else { // not Turing mma or T_B_KQ::I > 8
    if (ncols2 > 1 || pMaskH) {
#pragma unroll
      for (int i00 = 0; i00 < cNbatchFa; i00 += cNp * T_C_KQ::J) {
        const int i0 = i00 + (threadIdx.y % cNp) * T_C_KQ::J;

        // The mask is stored as 16 bit half values, loading them as 32 bit half2 values is preferred in terms of speed.
        // However, this is not possible for RDNA3 where 2 consecutive l indices are not consecutive in the mask memory
        // layout.
#ifdef RDNA3
#pragma unroll
        for (int l = 0; l < T_C_KQ::ne; ++l) {
          const int i = i0 + T_C_KQ::get_j(l);
          const int j = ((threadIdx.y / cNp) * cColsPerWarp + T_C_KQ::get_i(l)) / ncols2;

          vKqC[i00 / (cNp * T_C_KQ::J)].x[l] += __half2float(pTileMask[j * (cNbatchFa + 8) + i]);
        }
#else
#pragma unroll
        for (int l0 = 0; l0 < T_C_KQ::ne; l0 += 2) {
          const int i = (i0 + T_C_KQ::get_j(l0)) / 2;
          const int j = ((threadIdx.y / cNp) * cColsPerWarp + T_C_KQ::get_i(l0)) / ncols2;

          const float2 tmp = __half22float2(((const half2 *)pTileMask)[j * (cNbatchFa / 2 + 4) + i]);
          vKqC[i00 / (cNp * T_C_KQ::J)].x[l0 + 0] += pSlope * tmp.x;
          vKqC[i00 / (cNp * T_C_KQ::J)].x[l0 + 1] += pSlope * tmp.y;
        }
#endif // RDNA3
      }
    }

    // Calculate softmax for each KQ column using the current max. value.
    // The divisor is stored in KQ_rowsum and will be applied at the end.
    static_assert(cNbatchFa % (cNp * T_C_KQ::J) == 0, "bad loop size");
#pragma unroll
    for (int k0 = 0; k0 < cNbatchFa; k0 += cNp * T_C_KQ::J) {
#pragma unroll
      for (int l = 0; l < T_C_KQ::ne; ++l) {
        if (!oob_check || k0 + (threadIdx.y % cNp) * T_C_KQ::J + T_C_KQ::get_j(l) < pKVkqSup) {
#if defined(AMD_WMMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE)
          constexpr int KQ_idx = 0;
#else
          // Turing + Volta:
          const int KQ_idx = (l / 2) % 2;
#endif // defined(AMD_WMMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE)
          vKqMaxNew[KQ_idx] = fmaxf(vKqMaxNew[KQ_idx], vKqC[(k0 / (cNp * T_C_KQ::J))].x[l] + FATTN_KQ_MAX_OFFSET);
        }
      }
    }

#pragma unroll
    for (int col = 0; col < cColsPerThread; ++col) {
#if defined(TURING_MMA_AVAILABLE)
      // Values per KQ column are spread across 4 threads:
      constexpr int offset_first = 2;
      constexpr int offset_last = 1;
#elif defined(AMD_MFMA_AVAILABLE)
      // MFMA: 4 threads per Q column (threadIdx.x % 16 == col, spaced by 16).
      constexpr int offset_first = 32;
      constexpr int offset_last = 16;
#elif defined(AMD_WMMA_AVAILABLE)
      // Values per KQ column are spread across 2 threads:
      constexpr int offset_first = 16;
      constexpr int offset_last = 16;
#else  // Volta
       // Values per KQ column are spread across 2 threads:
      constexpr int offset_first = 2;
      constexpr int offset_last = 2;
#endif // defined(TURING_MMA_AVAILABLE)
#pragma unroll
      for (int offset = offset_first; offset >= offset_last; offset >>= 1) {
        vKqMaxNew[col] = fmaxf(vKqMaxNew[col], __shfl_xor_sync(0xFFFFFFFF, vKqMaxNew[col], offset, cWarpSize));
      }
    }

    static_assert(cNbatchFa % (cNp * T_C_KQ::J) == 0, "bad loop size");
#pragma unroll
    for (int k0 = 0; k0 < cNbatchFa; k0 += cNp * T_C_KQ::J) {
#pragma unroll
      for (int l = 0; l < T_C_KQ::ne; ++l) {
        if (!oob_check || k0 + (threadIdx.y % cNp) * T_C_KQ::J + T_C_KQ::get_j(l) < pKVkqSup) {
#if defined(AMD_WMMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE)
          constexpr int KQ_idx = 0;
#else
          // Turing + Volta:
          const int KQ_idx = (l / 2) % 2;
#endif // defined(AMD_WMMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE)
          vKqC[(k0 / (cNp * T_C_KQ::J))].x[l] = expf(vKqC[(k0 / (cNp * T_C_KQ::J))].x[l] - vKqMaxNew[KQ_idx]);
          vKqRowsumAdd[KQ_idx] += vKqC[(k0 / (cNp * T_C_KQ::J))].x[l];
        } else {
          vKqC[(k0 / (cNp * T_C_KQ::J))].x[l] = 0.0f;
        }
      }
    }
  }

  {
    float vKqMaxScale[cColsPerThread];
#pragma unroll
    for (int vCol = 0; vCol < cColsPerThread; ++vCol) {
      const float cKqMaxDiff = pKqMax[vCol] - vKqMaxNew[vCol];
      vKqMaxScale[vCol] = expf(cKqMaxDiff);
      pKqMax[vCol] = vKqMaxNew[vCol];

      *((uint32_t *)&vKqMaxScale[vCol]) *= cKqMaxDiff >= SOFTMAX_FTZ_THRESHOLD;

      // Scale previous KQ_rowsum to account for a potential increase in KQ_max:
      pKqRowsum[vCol] = vKqMaxScale[vCol] * pKqRowsum[vCol] + vKqRowsumAdd[vCol];
    }

#if defined(TURING_MMA_AVAILABLE)
    if constexpr (cColsPerWarp == 8) {
      const half2 KQ_max_scale_h2 = make_half2(vKqMaxScale[0], vKqMaxScale[cColsPerThread - 1]);
#pragma unroll
      for (int i = 0; i < DV / T_C_VKQ::I; ++i) {
#pragma unroll
        for (int l = 0; l < T_C_VKQ::ne; ++l) {
          pVkqC[i].x[l] *= KQ_max_scale_h2;
        }
      }
    } else {
#pragma unroll
      for (int col = 0; col < cColsPerThread; ++col) {
        const half2 KQ_max_scale_h2 = make_half2(vKqMaxScale[col], vKqMaxScale[col]);
#pragma unroll
        for (int i = 0; i < (DV / 2) / T_C_VKQ::J; ++i) {
#pragma unroll
          for (int l0 = 0; l0 < T_C_VKQ::ne; l0 += 2) {
            pVkqC[i].x[l0 + col] *= KQ_max_scale_h2;
          }
        }
      }
    }
#elif defined(AMD_WMMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE)
    if constexpr (std::is_same_v<decltype(T_C_VKQ::x), half2[T_C_VKQ::ne]>) {
      const half2 KQ_max_scale_h2 = make_half2(vKqMaxScale[0], vKqMaxScale[0]);
#pragma unroll
      for (int i = 0; i < (DV / 2) / T_C_VKQ::J; ++i) {
#pragma unroll
        for (int l = 0; l < T_C_VKQ::ne; ++l) {
          pVkqC[i].x[l] *= KQ_max_scale_h2;
        }
      }
    } else {
      static_assert(std::is_same_v<decltype(T_C_VKQ::x), float[T_C_VKQ::ne]>, "bad VKQ type");
#pragma unroll
      for (int i = 0; i < DV / T_C_VKQ::J; ++i) {
#pragma unroll
        for (int l = 0; l < T_C_VKQ::ne; ++l) {
          pVkqC[i].x[l] *= vKqMaxScale[0];
        }
      }
    }
#else // Volta
    const half2 KQ_max_scale_h2 = make_half2(vKqMaxScale[(threadIdx.x / 2) % 2], vKqMaxScale[(threadIdx.x / 2) % 2]);
#pragma unroll
    for (int i = 0; i < (DV / 2) / T_C_VKQ::J; ++i) {
#pragma unroll
      for (int l = 0; l < T_C_VKQ::ne; ++l) {
        pVkqC[i].x[l] *= KQ_max_scale_h2;
      }
    }
#endif // defined(TURING_MMA_AVAILABLE)
  }

  // Convert KQ C tiles into B tiles for VKQ calculation:
  T_B_VKQ vB[cNbatchFa / (cNp * 2 * T_B_VKQ::J)];
  static_assert(cNbatchFa % (cNp * 2 * T_B_VKQ::J) == 0, "bad loop size");
  if constexpr (cColsPerWarp == 8) {
#pragma unroll
    for (int vK = 0; vK < cNbatchFa / (cNp * 2 * T_B_VKQ::J); ++vK) {
      vB[vK] = get_transposed(get_half2(vKqC[vK]));
    }
  } else {
    for (int k = 0; k < cNbatchFa / (cNp * 2 * T_B_VKQ::J); ++k) {
      vB[k] = get_half2(vKqC[k]);
    }
  }

  if constexpr (cNstages > 1) {
    static_assert(!V_is_K_view, "K data reuse not implemented multi-stage loading");
    // Preload K tile for next iteration:
    constexpr bool use_cp_async = true;
    cp_async_wait_all();
    __syncthreads();
    if (!last_iter) {
      if (ncols2 > 1 || pMaskH) {
        flash_attn_ext_f16_load_mask<ncols1, nwarps, cNbatchFa, use_cp_async, oob_check>(
            pMaskH + cKVkq0 + cNbatchFa, pTileMask, pStrideMask, pKVkqSup, pJt * ncols1, pNe01);
      }
      flash_attn_ext_f16_load_tile<cStrideTileK, nwarps, cNbatchFa, use_cp_async, oob_check>(
          pKH2 + int64_t(cKVkq0 + cNbatchFa) * pStrideK, pTileK, cNbatchK2, pStrideK, pKVkqSup);
    }
  }

  // Calculate VKQ tile, need to use logical rather than physical elements for i0 due to transposition of V:
#pragma unroll
  for (int vI0Start = 0; vI0Start < DV; vI0Start += 2 * cNbatchV2) {
    static_assert(DV % (2 * cNbatchV2) == 0, "bad loop size");
    const int cI0Stop = vI0Start + 2 * cNbatchV2;

    if constexpr (cNstages <= 1) {
      const int cI0Diff = cI0Stop - vI0Start;
      if (!V_is_K_view || cI0Stop > 2 * cNbatchK2) {
        constexpr bool cUseCpAsync = cNstages == 1;
        flash_attn_ext_f16_load_tile<cStrideTileV, nwarps, cNbatchFa, cUseCpAsync, oob_check>(
            pVH2 + int64_t(cKVkq0) * pStrideV + vI0Start / 2, pTileV, cI0Diff / 2, pStrideV, pKVkqSup);
        if (cUseCpAsync) {
          cp_async_wait_all();
        }
        __syncthreads();
      }
    }
    const half2 *vTileVI = !V_is_K_view || cI0Stop > 2 * cNbatchK2 ? pTileV : pTileV + vI0Start / 2;

#if defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE)
#pragma unroll
    for (int vIVkq0 = vI0Start; vIVkq0 < cI0Stop; vIVkq0 += T_A_VKQ::I) {
      static_assert((cNbatchFa / 2) % (cNp * T_A_VKQ::J) == 0, "bad loop size");
#pragma unroll
      for (int vK00 = 0; vK00 < cNbatchFa / 2; vK00 += cNp * T_A_VKQ::J) {
        const int k0 = vK00 + (threadIdx.y % cNp) * T_A_VKQ::J;

        T_A_VKQ vA; // Transposed in SRAM but not in registers, gets transposed on load.
        load_ldmatrix_trans(vA, vTileVI + 2 * k0 * cStrideTileV + (vIVkq0 - vI0Start) / 2, cStrideTileV);
        if constexpr (T_B_KQ::I == 8) {
          mma(pVkqC[vIVkq0 / T_A_VKQ::I], vA, vB[vK00 / (cNp * T_A_VKQ::J)]);
        } else {
          // Wide version of VKQ_C is column-major.
#if defined(AMD_WMMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE)
          // AMD matrix C is column-major.
          mma(pVkqC[vIVkq0 / T_A_VKQ::I], vA, vB[vK00 / (cNp * T_A_VKQ::J)]);
#else
          // swap A and B for CUDA.
          mma(pVkqC[vIVkq0 / T_A_VKQ::I], vB[vK00 / (cNp * T_A_VKQ::J)], vA);
#endif // defined(AMD_WMMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE)
        }
      }
    }
#else // Volta
    constexpr int i0_stride = 2 * T_C_VKQ::J;
#pragma unroll
    for (int vIVkq0 = vI0Start; vIVkq0 < cI0Stop; vIVkq0 += i0_stride) {
      static_assert(cNbatchFa % (cNp * T_A_VKQ::I) == 0, "bad loop size");
      static_assert(2 * T_B_VKQ::J == T_A_VKQ::I, "bad tile sizes");
#pragma unroll
      for (int vK00 = 0; vK00 < cNbatchFa; vK00 += cNp * T_A_VKQ::I) {
        const int k0 = vK00 + (threadIdx.y % cNp) * T_A_VKQ::I;

        T_A_VKQ vA; // Transposed in both SRAM and registers, load normally.
        load_ldmatrix(vA, vTileVI + k0 * cStrideTileV + (vIVkq0 - vI0Start) / 2, cStrideTileV);
        mma(pVkqC[vIVkq0 / i0_stride], vB[vK00 / (cNp * T_A_VKQ::I)], vA);
      }
    }
#endif // defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE)

    if constexpr (cNstages <= 1) {
      __syncthreads(); // Only needed if tile_K == tile_V.
    }
  }
#else
  GGML_UNUSED_VARS(pQF2, pKH2, pVH2, pMaskH, pDstk, pDstkFixup, pScale, pSlope, pLogitSoftcap, pNe01, pNe02, pStrideK,
                   pStrideV, pStrideMask, pTileQ, pTileK, pTileV, pTileMask, pQB, pVkqC, pKqMax, pKqRowsum, pKb0);
  NO_DEVICE_CODE;
#endif // defined(VOLTA_MMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE) ||
       // defined(AMD_MFMA_AVAILABLE)
}

#if defined(TURING_MMA_AVAILABLE)
template <int DV, int ncols> struct mma_tile_sizes {
  using T_A_KQ = tile<16, 8, half2>;  // row-major
  using T_B_KQ = tile<16, 8, half2>;  // column-major
  using T_C_KQ = tile<16, 16, float>; // column-major
  using T_A_VKQ = tile<16, 8, half2>; // row-major
  using T_B_VKQ = tile<16, 8, half2>; // column-major
  using T_C_VKQ = tile<16, 8, half2>; // column-major
};
template <int DV> struct mma_tile_sizes<DV, 8> {
  using T_A_KQ = tile<16, 8, half2>;  // row-major
  using T_B_KQ = tile<8, 8, half2>;   // column-major
  using T_C_KQ = tile<16, 8, float>;  // row-major
  using T_A_VKQ = tile<16, 8, half2>; // row-major
  using T_B_VKQ = tile<8, 8, half2>;  // column-major
  using T_C_VKQ = tile<16, 4, half2>; // row-major
};
#elif defined(AMD_WMMA_AVAILABLE)
#ifdef RDNA3
template <int DV, int ncols> struct mma_tile_sizes {
  using T_A_KQ = tile<16, 8, half2, DATA_LAYOUT_I_MAJOR_MIRRORED>;  // row-major
  using T_B_KQ = tile<16, 8, half2, DATA_LAYOUT_I_MAJOR_MIRRORED>;  // column-major
  using T_C_KQ = tile<16, 16, float, DATA_LAYOUT_I_MAJOR>;          // column-major
  using T_A_VKQ = tile<32, 8, half2, DATA_LAYOUT_I_MAJOR_MIRRORED>; // row-major
  using T_B_VKQ = tile<16, 8, half2, DATA_LAYOUT_I_MAJOR_MIRRORED>; // column-major
  using T_C_VKQ = tile<16, 16, half2, DATA_LAYOUT_I_MAJOR>;         // column-major
};
template <int ncols> struct mma_tile_sizes<80, ncols> {
  using T_A_KQ = tile<16, 8, half2, DATA_LAYOUT_I_MAJOR_MIRRORED>;  // row-major
  using T_B_KQ = tile<16, 8, half2, DATA_LAYOUT_I_MAJOR_MIRRORED>;  // column-major
  using T_C_KQ = tile<16, 16, float, DATA_LAYOUT_I_MAJOR>;          // column-major
  using T_A_VKQ = tile<16, 8, half2, DATA_LAYOUT_I_MAJOR_MIRRORED>; // row-major
  using T_B_VKQ = tile<16, 8, half2, DATA_LAYOUT_I_MAJOR_MIRRORED>; // column-major
  using T_C_VKQ = tile<16, 16, float, DATA_LAYOUT_I_MAJOR>;         // column-major
};
template <int ncols> struct mma_tile_sizes<112, ncols> {
  using T_A_KQ = tile<16, 8, half2, DATA_LAYOUT_I_MAJOR_MIRRORED>;  // row-major
  using T_B_KQ = tile<16, 8, half2, DATA_LAYOUT_I_MAJOR_MIRRORED>;  // column-major
  using T_C_KQ = tile<16, 16, float, DATA_LAYOUT_I_MAJOR>;          // column-major
  using T_A_VKQ = tile<16, 8, half2, DATA_LAYOUT_I_MAJOR_MIRRORED>; // row-major
  using T_B_VKQ = tile<16, 8, half2, DATA_LAYOUT_I_MAJOR_MIRRORED>; // column-major
  using T_C_VKQ = tile<16, 16, float, DATA_LAYOUT_I_MAJOR>;         // column-major
};
#else
template <int DV, int ncols> struct mma_tile_sizes {
  using T_A_KQ = tile<16, 8, half2, DATA_LAYOUT_I_MAJOR>;             // row-major
  using T_B_KQ = tile<16, 8, half2, DATA_LAYOUT_I_MAJOR>;             // column-major
  using T_C_KQ = tile<16, 16, float, DATA_LAYOUT_I_MAJOR>;            // column-major
  using T_A_VKQ = tile<32, 8, half2, DATA_LAYOUT_I_MAJOR>;            // row-major
  using T_B_VKQ = tile<16, 8, half2, DATA_LAYOUT_I_MAJOR>;            // column-major
  using T_C_VKQ = tile<16, 16, half2, DATA_LAYOUT_I_MAJOR_SCRAMBLED>; // column-major
};
template <int ncols> struct mma_tile_sizes<80, ncols> {
  using T_A_KQ = tile<16, 8, half2>;  // row-major
  using T_B_KQ = tile<16, 8, half2>;  // column-major
  using T_C_KQ = tile<16, 16, float>; // column-major
  using T_A_VKQ = tile<16, 8, half2>; // row-major
  using T_B_VKQ = tile<16, 8, half2>; // column-major
  using T_C_VKQ = tile<16, 8, half2>; // column-major
};
template <int ncols> struct mma_tile_sizes<112, ncols> {
  using T_A_KQ = tile<16, 8, half2>;  // row-major
  using T_B_KQ = tile<16, 8, half2>;  // column-major
  using T_C_KQ = tile<16, 16, float>; // column-major
  using T_A_VKQ = tile<16, 8, half2>; // row-major
  using T_B_VKQ = tile<16, 8, half2>; // column-major
  using T_C_VKQ = tile<16, 8, half2>; // column-major
};
#endif // RDNA3
#elif defined(AMD_MFMA_AVAILABLE)
template <int DV, int ncols> struct mma_tile_sizes {
  using T_A_KQ = tile<16, 8, half2>;  // row-major
  using T_B_KQ = tile<16, 8, half2>;  // column-major
  using T_C_KQ = tile<16, 16, float>; // column-major
  using T_A_VKQ = tile<16, 8, half2>; // row-major
  using T_B_VKQ = tile<16, 8, half2>; // column-major
  using T_C_VKQ = tile<16, 8, half2>; // column-major
};
#else  // Volta
template <int DV, int ncols> struct mma_tile_sizes {
  using T_A_KQ = tile<8, 4, half2, DATA_LAYOUT_I_MAJOR_MIRRORED>;  // row-major
  using T_B_KQ = tile<32, 4, half2, DATA_LAYOUT_I_MAJOR>;          // column-major
  using T_C_KQ = tile<32, 8, float, DATA_LAYOUT_I_MAJOR>;          // column-major
  using T_A_VKQ = tile<8, 4, half2, DATA_LAYOUT_J_MAJOR_MIRRORED>; // column-major
  using T_B_VKQ = tile<32, 4, half2, DATA_LAYOUT_I_MAJOR>;         // column-major
  using T_C_VKQ = tile<32, 4, half2, DATA_LAYOUT_I_MAJOR>;         // column-major
};
#endif // defined(TURING_MMA_AVAILABLE)

template <int DKQ, int DV, int ncols1, int ncols2, int nwarps, bool use_logit_softcap, bool V_is_K_view,
          bool needs_fixup, bool is_fixup>
static __device__ __forceinline__ void flash_attn_ext_f16_process_tile(
    const float2 *const __restrict__ pQF2, const half2 *const __restrict__ pKH2, const half2 *const __restrict__ pVH2,
    const half *const __restrict__ pMaskH, const float *const __restrict__ pSinksF, float2 *const __restrict__ pDstk,
    float2 *const __restrict__ pDstkFixup, const float pScale, const float pSlope, const float pLogitSoftcap,
    const uint3 pNe01, const int pNe02, const int pGqaRatio, const int pNe11, const int pStrideQ1, const int pStrideQ2,
    const int pStrideK, const int pStrideV, const int pStrideMask, const int pJt, const int pZtGqa, const int pKb0Start,
    const int pKb0Stop) {
#if defined(VOLTA_MMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE) ||                    \
    defined(AMD_MFMA_AVAILABLE)
  // In this kernel Q, K, V are matrices while i, j, k are matrix indices.

  constexpr int cWarpSize = ggml_cuda_get_physical_warp_size();
  constexpr int cNcols = ncols1 * ncols2;
  using T_A_KQ = typename mma_tile_sizes<DV, cNcols>::T_A_KQ;
  using T_B_KQ = typename mma_tile_sizes<DV, cNcols>::T_B_KQ;
  using T_C_KQ = typename mma_tile_sizes<DV, cNcols>::T_C_KQ;
  using T_A_VKQ = typename mma_tile_sizes<DV, cNcols>::T_A_VKQ;
  using T_B_VKQ = typename mma_tile_sizes<DV, cNcols>::T_B_VKQ;
  using T_C_VKQ = typename mma_tile_sizes<DV, cNcols>::T_C_VKQ;

  constexpr int cColsPerWarp = T_B_KQ::I;
  constexpr int cColsPerThread = get_cols_per_thread();
  constexpr int cNp =
      cColsPerWarp > cNcols ? nwarps : nwarps * cColsPerWarp / cNcols; // Number of parallel CUDA warps per Q column.
  constexpr int cNbatchFa = ggml_cuda_fattn_mma_get_nbatch_fa(DKQ, DV, cNcols);
  constexpr int cNbatchK2 = ggml_cuda_fattn_mma_get_nbatch_K2(DKQ, DV, cNcols);
  constexpr int cNbatchV2 = ggml_cuda_fattn_mma_get_nbatch_V2(DKQ, DV, cNcols);
  constexpr int cNbatchCombine = ggml_cuda_fattn_mma_get_nbatch_combine(DKQ, DV, cNcols);
  constexpr bool cQInReg = ggml_cuda_fattn_mma_get_Q_in_reg(DKQ, DV, cNcols);
  constexpr int cNstages = ggml_cuda_fattn_mma_get_nstages(DKQ, DV, ncols1, ncols2);

  if (cColsPerWarp > cNcols) {
    NO_DEVICE_CODE;
    return;
  }

  static_assert(nwarps * (cColsPerWarp / ncols2) % ncols1 == 0, "bad nwarps");

  constexpr int cStrideTileQ = DKQ / 2 + 4;
  constexpr int cStrideTileK = cNbatchK2 + 4;

  constexpr int cStrideTileV = V_is_K_view ? cStrideTileK : cNbatchV2 + 4;
  constexpr int cStrideTileKvMax = cStrideTileK > cStrideTileV ? cStrideTileK : cStrideTileV;

  extern __shared__ half2 vTileQ[];
  half2 *vTileK = cQInReg ? vTileQ : vTileQ + cNcols * cStrideTileQ;
  half2 *vTileV = cNstages > 1 ? vTileK + cNbatchFa * cStrideTileK : vTileK;
  half *vTileMask = (half *)(cNstages > 1 ? vTileV + cNbatchFa * cStrideTileV : vTileV + cNbatchFa * cStrideTileKvMax);

  T_B_KQ vQB[(cQInReg ? DKQ / (2 * T_B_KQ::J) : 1)];
#if defined(TURING_MMA_AVAILABLE)
  T_C_VKQ vVkqC[cColsPerWarp == 8 ? DV / T_C_VKQ::I : DV / (2 * T_C_VKQ::J)];
#elif defined(AMD_WMMA_AVAILABLE) && defined(RDNA3)
  T_C_VKQ vVkqC[DV % 32 != 0 ? DV / T_C_VKQ::J : DV / (2 * T_C_VKQ::J)];
#elif defined(AMD_WMMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE)
  T_C_VKQ vVkqC[DV / (2 * T_C_VKQ::J)];
#else  // Volta
  T_C_VKQ vVkqC[DV / (2 * T_C_VKQ::J)];
#endif // defined(TURING_MMA_AVAILABLE)

  float vKqRowsum[cColsPerThread] = {0.0f};
  float vKqMax[cColsPerThread];
#pragma unroll
  for (int col = 0; col < cColsPerThread; ++col) {
    vKqMax[col] = -FLT_MAX / 2.0f;
  }

  // Load Q data into tile_Q, either temporarily or permanently.
  // Q in registers is faster, but register pressure is the biggest bottleneck.
  // The loading is done with decreasing granularity for D for better memory bandwidth.
  const half2 cScaleH2 = make_half2(pScale, pScale);
#pragma unroll
  for (int vStrideK : {cWarpSize, cWarpSize / 2, cWarpSize / 4, cWarpSize / 8}) {
    const int cK0Start = vStrideK == cWarpSize ? 0 : DKQ / 2 - (DKQ / 2) % (2 * vStrideK);
    const int cK0Stop = DKQ / 2 - (DKQ / 2) % (1 * vStrideK);
    const int cStrideJc = cWarpSize / vStrideK;

    if (cK0Start == cK0Stop) {
      continue;
    }

#pragma unroll
    for (int vJc0 = 0; vJc0 < cNcols; vJc0 += nwarps * cStrideJc) {
      const int cJc = vJc0 + threadIdx.y * cStrideJc + (vStrideK == cWarpSize ? 0 : threadIdx.x / vStrideK);

      if (vJc0 + nwarps * cStrideJc > cNcols && cJc >= cNcols) {
        break;
      }

      const int cJ = cJc / ncols2;
      const int c_ = cJc % ncols2;

      if ((ncols1 == 1 || pJt * ncols1 + cJ < int(pNe01.z)) && (ncols2 == 1 || pZtGqa * ncols2 + c_ < pGqaRatio)) {
#pragma unroll
        for (int vK0 = cK0Start; vK0 < cK0Stop; vK0 += vStrideK) {
          const int cK = vK0 + (vStrideK == cWarpSize ? threadIdx.x : threadIdx.x % vStrideK);

          const float2 cTmp = pQF2[(pJt * ncols1 + cJ) * pStrideQ1 + c_ * pStrideQ2 + cK];
          vTileQ[cJc * cStrideTileQ + cK] = cScaleH2 * make_half2(cTmp.x, cTmp.y);
        }
      } else {
#pragma unroll
        for (int vK0 = cK0Start; vK0 < cK0Stop; vK0 += vStrideK) {
          const int cK = vK0 + (vStrideK == cWarpSize ? threadIdx.x : threadIdx.x % vStrideK);

          vTileQ[cJc * cStrideTileQ + cK] = make_half2(0.0f, 0.0f);
        }
      }
    }
  }

  __syncthreads();

  if (cQInReg) {
    const int cJ0 = (threadIdx.y / cNp) * cColsPerWarp;

#pragma unroll
    for (int vK0 = 0; vK0 < DKQ / 2; vK0 += T_B_KQ::J) {
      load_ldmatrix(vQB[vK0 / T_B_KQ::J], vTileQ + cJ0 * cStrideTileQ + vK0, cStrideTileQ);
    }
  }

  __syncthreads();

  int vKb0 = pKb0Start;

  // Preload mask and K data for first iteration when using cp_async with multiple stages:
  if constexpr (cNstages > 1) {
    static_assert(cNbatchK2 == DKQ / 2, "batching not implemented for multi-stage pipeline");
    constexpr bool use_cp_async = true;
    constexpr bool oob_check = false;
    constexpr int k_VKQ_sup = cNbatchFa;
    if (ncols2 > 1 || pMaskH) {
      flash_attn_ext_f16_load_mask<ncols1, nwarps, cNbatchFa, use_cp_async, oob_check>(
          pMaskH + vKb0 * cNbatchFa, vTileMask, pStrideMask, k_VKQ_sup, pJt * ncols1, pNe01);
    }
    flash_attn_ext_f16_load_tile<cStrideTileK, nwarps, cNbatchFa, use_cp_async, oob_check>(
        pKH2 + int64_t(vKb0) * cNbatchFa * pStrideK, vTileK, cNbatchK2, pStrideK, k_VKQ_sup);
  }

  // kb0_start is always < kb0_stop so the last iter can be executed unconditionally.
  if constexpr (ncols2 == 1) {
    constexpr bool oob_check = true;
    for (; vKb0 < pKb0Stop - 1; ++vKb0) {
      constexpr bool last_iter = false;
      constexpr int k_VKQ_sup = cNbatchFa;
      flash_attn_ext_f16_iter<DKQ, DV, ncols1, ncols2, nwarps, use_logit_softcap, V_is_K_view, needs_fixup, is_fixup,
                              last_iter, oob_check, T_A_KQ, T_B_KQ, T_C_KQ, T_A_VKQ, T_B_VKQ, T_C_VKQ>(
          pQF2, pKH2, pVH2, pMaskH, pDstk, pDstkFixup, pScale, pSlope, pLogitSoftcap, pNe01, pNe02, pStrideK, pStrideV,
          pStrideMask, vTileQ, vTileK, vTileV, vTileMask, vQB, vVkqC, vKqMax, vKqRowsum, pJt, vKb0, k_VKQ_sup);
    }
    constexpr bool last_iter = true;
    const int k_VKQ_sup = pNe11 - vKb0 * cNbatchFa;
    flash_attn_ext_f16_iter<DKQ, DV, ncols1, ncols2, nwarps, use_logit_softcap, V_is_K_view, needs_fixup, is_fixup,
                            last_iter, oob_check, T_A_KQ, T_B_KQ, T_C_KQ, T_A_VKQ, T_B_VKQ, T_C_VKQ>(
        pQF2, pKH2, pVH2, pMaskH, pDstk, pDstkFixup, pScale, pSlope, pLogitSoftcap, pNe01, pNe02, pStrideK, pStrideV,
        pStrideMask, vTileQ, vTileK, vTileV, vTileMask, vQB, vVkqC, vKqMax, vKqRowsum, pJt, vKb0, k_VKQ_sup);
  } else {
    constexpr bool cOobCheck = false;
    for (; vKb0 < pKb0Stop - 1; ++vKb0) {
      constexpr bool cLastIter = false;
      constexpr int cKVkqSup = cNbatchFa;
      flash_attn_ext_f16_iter<DKQ, DV, ncols1, ncols2, nwarps, use_logit_softcap, V_is_K_view, needs_fixup, is_fixup,
                              cLastIter, cOobCheck, T_A_KQ, T_B_KQ, T_C_KQ, T_A_VKQ, T_B_VKQ, T_C_VKQ>(
          pQF2, pKH2, pVH2, pMaskH, pDstk, pDstkFixup, pScale, pSlope, pLogitSoftcap, pNe01, pNe02, pStrideK, pStrideV,
          pStrideMask, vTileQ, vTileK, vTileV, vTileMask, vQB, vVkqC, vKqMax, vKqRowsum, pJt, vKb0, cKVkqSup);
    }
    constexpr bool cLastIter = true;
    constexpr int cKVkqSup = cNbatchFa;
    flash_attn_ext_f16_iter<DKQ, DV, ncols1, ncols2, nwarps, use_logit_softcap, V_is_K_view, needs_fixup, is_fixup,
                            cLastIter, cOobCheck, T_A_KQ, T_B_KQ, T_C_KQ, T_A_VKQ, T_B_VKQ, T_C_VKQ>(
        pQF2, pKH2, pVH2, pMaskH, pDstk, pDstkFixup, pScale, pSlope, pLogitSoftcap, pNe01, pNe02, pStrideK, pStrideV,
        pStrideMask, vTileQ, vTileK, vTileV, vTileMask, vQB, vVkqC, vKqMax, vKqRowsum, pJt, vKb0, cKVkqSup);
  }

  // With multi-stage loading there is no __syncthreads at the end of the iter,
  //     there can be a race condition on shared memory access for combining/writing back results.
  if constexpr (cNstages > 1 && nwarps * cColsPerWarp > cNbatchFa) {
    __syncthreads();
  }

  // Finally, sum up partial KQ rowsums.
  {
#if defined(TURING_MMA_AVAILABLE)
    // The partial sums are spread across 8/4 threads.
    constexpr int cOffsetFirst = cColsPerWarp == 8 ? 16 : 2;
    constexpr int cOffsetLast = cColsPerWarp == 8 ? 4 : 1;
#elif defined(AMD_MFMA_AVAILABLE)
    // The partial sums are spread across 4 threads (wavefront64, 16 cols).
    constexpr int cOffsetFirst = 32;
    constexpr int cOffsetLast = 16;
#elif defined(AMD_WMMA_AVAILABLE)
    // The partial sums are spread across 2 threads.
    constexpr int cOffsetFirst = 16;
    constexpr int cOffsetLast = 16;
#else  // Volta
       // The partial sums are spread across 2 threads.
    constexpr int cOffsetFirst = 2;
    constexpr int cOffsetLast = 2;
#endif // defined(TURING_MMA_AVAILABLE)
#pragma unroll
    for (int col = 0; col < cColsPerThread; ++col) {
#pragma unroll
      for (int vOffset = cOffsetFirst; vOffset >= cOffsetLast; vOffset >>= 1) {
        vKqRowsum[col] += __shfl_xor_sync(0xFFFFFFFF, vKqRowsum[col], vOffset, cWarpSize);
      }
    }
  }

  // If attention sinks are used, potentially re-scale if KQ_max is small.
  // Also add the sink as a value to KQ_rowsum, this is done after synchronization of KQ_rowsum
  //     so it's being done unconditionally for every thread.
  if (!is_fixup && (cNp == 1 || threadIdx.y % cNp == 0) && pSinksF) {
    float vKqMaxScale[cColsPerThread];
#pragma unroll
    for (int col = 0; col < cColsPerThread; ++col) {
      const int cJc =
          (threadIdx.y / cNp) * cColsPerWarp + (cColsPerWarp == 8 ? T_C_KQ::get_j(col) : T_C_KQ::get_i(2 * col));
      const float cSink = pSinksF[cJc % ncols2];

      const float cKqMaxNew = fmaxf(vKqMax[col], cSink);
      const float cKqMaxDiff = vKqMax[col] - cKqMaxNew;
      vKqMaxScale[col] = expf(cKqMaxDiff);
      vKqMax[col] = cKqMaxNew;

      *((uint32_t *)&vKqMaxScale[col]) *= cKqMaxDiff >= SOFTMAX_FTZ_THRESHOLD;

      const float cKqMaxAdd = expf(cSink - cKqMaxNew);
      vKqRowsum[col] = vKqMaxScale[col] * vKqRowsum[col] + cKqMaxAdd;
    }

#if defined(TURING_MMA_AVAILABLE)
    if constexpr (cColsPerWarp == 8) {
      const half2 KQ_max_scale_h2 = make_half2(vKqMaxScale[0], vKqMaxScale[cColsPerThread - 1]);
#pragma unroll
      for (int i = 0; i < DV / T_C_VKQ::I; ++i) {
#pragma unroll
        for (int l = 0; l < T_C_VKQ::ne; ++l) {
          vVkqC[i].x[l] *= KQ_max_scale_h2;
        }
      }
    } else {
#pragma unroll
      for (int col = 0; col < cColsPerThread; ++col) {
        const half2 KQ_max_scale_h2 = make_half2(vKqMaxScale[col], vKqMaxScale[col]);
#pragma unroll
        for (int i = 0; i < (DV / 2) / T_C_VKQ::J; ++i) {
#pragma unroll
          for (int l0 = 0; l0 < T_C_VKQ::ne; l0 += 2) {
            vVkqC[i].x[l0 + col] *= KQ_max_scale_h2;
          }
        }
      }
    }
#elif defined(AMD_WMMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE)
    if constexpr (std::is_same_v<decltype(T_C_VKQ::x), half2[T_C_VKQ::ne]>) {
      const half2 KQ_max_scale_h2 = make_half2(vKqMaxScale[0], vKqMaxScale[0]);
#pragma unroll
      for (int i = 0; i < (DV / 2) / T_C_VKQ::J; ++i) {
#pragma unroll
        for (int l = 0; l < T_C_VKQ::ne; ++l) {
          vVkqC[i].x[l] *= KQ_max_scale_h2;
        }
      }
    } else {
      static_assert(std::is_same_v<decltype(T_C_VKQ::x), float[T_C_VKQ::ne]>, "bad VKQ type");
#pragma unroll
      for (int i = 0; i < DV / T_C_VKQ::J; ++i) {
#pragma unroll
        for (int l = 0; l < T_C_VKQ::ne; ++l) {
          vVkqC[i].x[l] *= vKqMaxScale[0];
        }
      }
    }
#else // Volta
    const int col = (threadIdx.x / 2) % 2;
    const half2 KQ_max_scale_h2 = make_half2(vKqMaxScale[col], vKqMaxScale[col]);
#pragma unroll
    for (int i = 0; i < (DV / 2) / T_C_VKQ::J; ++i) {
#pragma unroll
      for (int l = 0; l < T_C_VKQ::ne; ++l) {
        vVkqC[i].x[l] *= KQ_max_scale_h2;
      }
    }
#endif // defined(TURING_MMA_AVAILABLE)
  }

  // Combine VKQ accumulator values if np > 1.
  // It's also faster to do small writes to shared memory, then large write to VRAM than to do small writes to VRAM.
  // So also write VKQ accumulators to shared memory in column-major format if np == 1.

  constexpr int cTileStride = cNbatchCombine + 4;
  static_assert((DV / 2) % cNbatchCombine == 0, "bad nbatch_combine");

  if constexpr (cColsPerWarp == 8) {
    const int cJcCwmo = (threadIdx.x % (2 * T_C_VKQ::J)) / T_C_VKQ::J; // jc combine write meta offset
    const int jc_cwm = threadIdx.y * (2 * T_C_VKQ::J) + 2 * T_C_VKQ::get_j(-1) + cJcCwmo; // jc combine write meta
    const float2 KQ_cmr = make_float2(vKqMax[cJcCwmo], vKqRowsum[cJcCwmo]);               // KQ combine max rowsum

    if (((!needs_fixup && !is_fixup) || cNp > 1) && threadIdx.x < 2 * T_C_VKQ::J) {
      // Use the 16 bytes of padding in each row to store the meta data: KQ max, KQ rowsum, KQ max scale.
      ((float2 *)vTileQ)[jc_cwm * (cTileStride / 2) + cNbatchCombine / 2] = KQ_cmr;
    }

    __syncthreads();

    if (cNp == 1) {
      // No combination is needed, the meta data can be directly written from registers to VRAM.
      if (needs_fixup && threadIdx.x < T_B_KQ::I) {
        float2 *vDstkFixupMeta = pDstkFixup + blockIdx.x * cNcols;
        vDstkFixupMeta[jc_cwm] = KQ_cmr;
      }
      if (is_fixup && threadIdx.x < T_B_KQ::I) {
        float2 *vDstkFixupMeta = pDstkFixup + (gridDim.x + blockIdx.x) * cNcols;
        vDstkFixupMeta[jc_cwm] = KQ_cmr;
      }
    }
  } else {
    // jc_cwm = jc combine write meta
    // KQ_cmr = KQ combine max rowsum
    // Use the 16 bytes of padding in each Q column to store the meta data: KQ max, KQ rowsum, KQ max scale.
#if defined(TURING_MMA_AVAILABLE)
    const int jc_cwm = threadIdx.y * cColsPerWarp + T_C_VKQ::get_i(threadIdx.x % 4);
    const float2 KQ_cmr = make_float2(vKqMax[threadIdx.x % cColsPerThread], vKqRowsum[threadIdx.x % cColsPerThread]);
    const bool thread_should_write = threadIdx.x % 4 < cColsPerThread;
#elif defined(AMD_WMMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE)
    const int jc_cwm = threadIdx.y * cColsPerWarp + T_C_VKQ::get_i(0);
    const float2 KQ_cmr = make_float2(vKqMax[0], vKqRowsum[0]);
    const bool thread_should_write = threadIdx.x / 16 < cColsPerThread;
#else  // Volta
    const int jc_cwm = threadIdx.y * cColsPerWarp + T_C_KQ::get_i(threadIdx.x & 2);
    const float2 KQ_cmr = make_float2(vKqMax[(threadIdx.x & 2) / 2], vKqRowsum[(threadIdx.x & 2) / 2]);
    const bool thread_should_write = T_C_KQ::J == 8 || T_C_KQ::get_j(threadIdx.x & 2) < 8;
#endif // defined(TURING_MMA_AVAILABLE)

    if (((!needs_fixup && !is_fixup) || cNp > 1) && thread_should_write) {
      ((float2 *)vTileQ)[jc_cwm * (cTileStride / 2) + cNbatchCombine / 2] = KQ_cmr;
    }

    __syncthreads();

    if (cNp == 1) {
      // No combination is needed, the meta data can be directly written from registers to VRAM.
      if (needs_fixup && thread_should_write) {
        float2 *dstk_fixup_meta = pDstkFixup + blockIdx.x * cNcols;
        dstk_fixup_meta[jc_cwm] = KQ_cmr;
      }
      if (is_fixup && thread_should_write) {
        float2 *dstk_fixup_meta = pDstkFixup + (gridDim.x + blockIdx.x) * cNcols;
        dstk_fixup_meta[jc_cwm] = KQ_cmr;
      }
    }
  }

  if (cNp > 1 && threadIdx.y % cNp == 0) {
    // Combine the meta data for parallel warps via shared memory.
    // Warps with threadIdx.y % np != 0 must NOT return early.
    // All threads must return simultaneously to avoid race conditions with work on the next tile.

    constexpr int cNmeta = cNp * cColsPerWarp >= cWarpSize ? cNp * cColsPerWarp / cWarpSize : 1;

    const int cJcMeta = threadIdx.y * cColsPerWarp +
                        (cNp * cColsPerWarp < cWarpSize ? threadIdx.x % (cNp * cColsPerWarp) : threadIdx.x);
    float2 *const cMetaPtr = ((float2 *)vTileQ) + cJcMeta * (cTileStride / 2) + cNbatchCombine / 2;
    float2 vMeta[cNmeta];
#pragma unroll
    for (int vImeta = 0; vImeta < cNmeta; ++vImeta) {
      vMeta[vImeta] = cMetaPtr[vImeta * cWarpSize * cTileStride / 2];
    }

    float vKqCmn = vMeta[0].x; // KQ combine max new, max between all parallel warps.
#pragma unroll
    for (int vImeta = 1; vImeta < cNmeta; ++vImeta) {
      vKqCmn = fmaxf(vKqCmn, vMeta[vImeta].x);
    }
#pragma unroll
    for (int vOffset = cNp * cColsPerWarp / 2; vOffset >= cColsPerWarp; vOffset >>= 1) {
      if (vOffset < cWarpSize) {
        vKqCmn = fmaxf(vKqCmn, __shfl_xor_sync(0xFFFFFFFF, vKqCmn, vOffset, cWarpSize));
      }
    }

    float vKqCms[cNmeta]; // KQ combine max scale per warp.
#pragma unroll
    for (int vImeta = 0; vImeta < cNmeta; ++vImeta) {
      vKqCms[vImeta] = expf(vMeta[vImeta].x - vKqCmn);
    }

    float vKqCrs = vKqCms[0] * vMeta[0].y; // KQ combine rowsum, scaled sum of all parallel warps.
#pragma unroll
    for (int vImeta = 1; vImeta < cNmeta; ++vImeta) {
      vKqCrs += vKqCms[vImeta] * vMeta[vImeta].y;
    }
#pragma unroll
    for (int vOffset = cNp * cColsPerWarp / 2; vOffset >= cColsPerWarp; vOffset >>= 1) {
      if (vOffset < cWarpSize) {
        vKqCrs += __shfl_xor_sync(0xFFFFFFFF, vKqCrs, vOffset, cWarpSize);
      }
    }

    __syncthreads();

    // Write back combined meta data:
#pragma unroll
    for (int vImeta = 0; vImeta < cNmeta; ++vImeta) {
      if (cNp * cColsPerWarp >= cWarpSize || threadIdx.x < cNp * cColsPerWarp) {
        // Combined KQ max scale + rowsum.
        cMetaPtr[vImeta * cWarpSize * cTileStride / 2] = make_float2(vKqCms[vImeta], vKqCrs);
      }
    }

    // Combined KQ max + rowsum.
    static_assert(cColsPerWarp <= cWarpSize);
    if (needs_fixup && (cColsPerWarp == cWarpSize || threadIdx.x < cColsPerWarp)) {
      float2 *vDstkFixupMeta = pDstkFixup + blockIdx.x * cNcols;
      vDstkFixupMeta[(threadIdx.y / cNp) * cColsPerWarp + threadIdx.x] = make_float2(vKqCmn, vKqCrs);
    }
    if (is_fixup && (cColsPerWarp == cWarpSize || threadIdx.x < cColsPerWarp)) {
      float2 *vDstkFixupMeta = pDstkFixup + (gridDim.x + blockIdx.x) * cNcols;
      vDstkFixupMeta[(threadIdx.y / cNp) * cColsPerWarp + threadIdx.x] = make_float2(vKqCmn, vKqCrs);
    }
  } else if (cNp > 1) {
    // Warps with threadIdx.y % np == 0 execute a __syncthreads() in the if branch.
    // Therefore, all other warps also need to execute a __syncthreads().
    // Otherwise the points at which warps synchronize with each other would become misaligned.
    __syncthreads();
  }

#pragma unroll
  for (int vK00 = 0; vK00 < DV / 2; vK00 += cNbatchCombine) {
    if constexpr (cColsPerWarp == 8) {
      static_assert(std::is_same_v<decltype(T_C_VKQ::x), half2[T_C_VKQ::ne]>, "bad VKQ type");
      const int cJcCwd = threadIdx.y * T_B_KQ::I + T_B_KQ::get_i(-1); // jc combine write data
#pragma unroll
      for (int vK1 = 0; vK1 < cNbatchCombine; vK1 += T_B_KQ::J) {
        const T_B_KQ cB = get_transposed(
            vVkqC[(vK00 + vK1) / T_B_KQ::J]); // Conversion of C to B matrix puts it in column-major format.

#pragma unroll
        for (int l = 0; l < T_B_KQ::ne; ++l) {
          const int cK = vK1 + T_B_KQ::get_j(l);

          vTileQ[cJcCwd * cTileStride + cK] = cB.x[l];
        }
      }
    } else {
      const int j0 = threadIdx.y * cColsPerWarp;
      if constexpr (std::is_same_v<decltype(T_C_VKQ::x), half2[T_C_VKQ::ne]>) {
        if constexpr (T_C_VKQ::dl == DATA_LAYOUT_I_MAJOR) {
#pragma unroll
          for (int k1 = 0; k1 < cNbatchCombine; k1 += T_C_VKQ::J) {
#pragma unroll
            for (int l = 0; l < T_C_VKQ::ne; ++l) {
              const int j = j0 + T_C_VKQ::get_i(l);
              const int k = k1 + T_C_VKQ::get_j(l);

              vTileQ[j * cTileStride + k] = vVkqC[(vK00 + k1) / T_C_VKQ::J].x[l];
            }
          }
        } else {
          static_assert(T_C_VKQ::dl == DATA_LAYOUT_I_MAJOR_SCRAMBLED, "bad T_C_VKQ data layout");
          using T_C_VKQ_us = tile<T_C_VKQ::I, T_C_VKQ::J, half2, DATA_LAYOUT_I_MAJOR>; // us == unscrambled
#pragma unroll
          for (int k1 = 0; k1 < cNbatchCombine; k1 += T_C_VKQ::J) {
            const T_C_VKQ_us VKQ_C_us = unscramble(vVkqC[(vK00 + k1) / T_C_VKQ::J]);
#pragma unroll
            for (int l = 0; l < T_C_VKQ_us::ne; ++l) {
              const int j = j0 + T_C_VKQ_us::get_i(l);
              const int k = k1 + T_C_VKQ_us::get_j(l);

              vTileQ[j * cTileStride + k] = VKQ_C_us.x[l];
            }
          }
        }
      } else {
        static_assert(std::is_same_v<decltype(T_C_VKQ::x), float[T_C_VKQ::ne]>, "bad VKQ type");
        half *tile_Q_h = (half *)vTileQ;
#pragma unroll
        for (int k1 = 0; k1 < cNbatchCombine; k1 += T_C_VKQ::J / 2) {
#pragma unroll
          for (int l = 0; l < T_C_VKQ::ne; ++l) {
            const int j = j0 + T_C_VKQ::get_i(l);
            const int k = 2 * k1 + T_C_VKQ::get_j(l);

            tile_Q_h[j * (2 * cTileStride) + k] = vVkqC[(vK00 + k1) / (T_C_VKQ::J / 2)].x[l];
          }
        }
      }
    }

    __syncthreads();

    if (cNp == 1 || threadIdx.y % cNp == 0) {
      // The first 2*2*gridDim.x*ncols floats in dstk_fixup are for storing max. values and row sums.
      // The values after that are for the partial results of the individual blocks.
      float2 *vDstkFixupData = pDstkFixup + gridDim.x * (2 * cNcols) + blockIdx.x * (cNcols * (DV / 2));

#pragma unroll
      for (int vStrideK : {cWarpSize, cWarpSize / 2, cWarpSize / 4, cWarpSize / 8}) {
        const int cK0Start = vStrideK == cWarpSize ? 0 : cNbatchCombine - cNbatchCombine % (2 * vStrideK);
        const int cK0Stop = cNbatchCombine - cNbatchCombine % (1 * vStrideK);
        const int cStrideJc = cWarpSize / vStrideK;

        if (cK0Start == cK0Stop) {
          continue;
        }

#pragma unroll
        for (int vJc0Dst = 0; vJc0Dst < cNcols; vJc0Dst += (nwarps / cNp) * cStrideJc) {
          const int cJcDst =
              vJc0Dst + (threadIdx.y / cNp) * cStrideJc + (vStrideK == cWarpSize ? 0 : threadIdx.x / vStrideK);

          if (vJc0Dst + (nwarps / cNp) * cStrideJc > cNcols && cJcDst >= cNcols) {
            break;
          }

          const int cJcTileK = (cJcDst / cColsPerWarp) * (cNp * cColsPerWarp) + cJcDst % cColsPerWarp;

          const int cJDst = cJcDst / ncols2;
          const int cCDst = cJcDst % ncols2;

          if (!is_fixup && ((ncols1 > 1 && pJt * ncols1 + cJDst >= int(pNe01.z)) ||
                            (ncols2 > 1 && pZtGqa * ncols2 + cCDst >= pGqaRatio))) {
            continue;
          }

          const float *vMetaJ = (const float *)vTileQ + cJcTileK * cTileStride + cNbatchCombine;
#pragma unroll
          for (int vK0 = cK0Start; vK0 < cK0Stop; vK0 += vStrideK) {
            const int cK = vK0 + (vStrideK == cWarpSize ? threadIdx.x : threadIdx.x % vStrideK);

            float2 vDstkVal = make_float2(0.0f, 0.0f);
#pragma unroll
            for (int vIp = 0; vIp < cNp; ++vIp) {
              const float cKqCrs = cNp == 1 ? 1.0f : vMetaJ[vIp * cColsPerWarp * cTileStride + 0];
              const float2 cDstkValAdd = __half22float2(vTileQ[(cJcTileK + vIp * cColsPerWarp) * cTileStride + cK]);
              vDstkVal.x += cDstkValAdd.x * cKqCrs;
              vDstkVal.y += cDstkValAdd.y * cKqCrs;
            }

            if (!needs_fixup && !is_fixup) {
              const float cKqRowsumJ = vMetaJ[1];
              vDstkVal.x /= cKqRowsumJ;
              vDstkVal.y /= cKqRowsumJ;
            }

            if (is_fixup) {
              vDstkFixupData[cJcDst * (DV / 2) + vK00 + cK] = vDstkVal;
            } else {
              pDstk[((pJt * ncols1 + cJDst) * pNe02 + cCDst) * (DV / 2) + vK00 + cK] = vDstkVal;
            }
          }
        }
      }
    }
    if (cNp > 1) {
      __syncthreads();
    }
  }
#else
  GGML_UNUSED_VARS(pQF2, pKH2, pVH2, pMaskH, pSinksF, pDstk, pDstkFixup, pScale, pSlope, pLogitSoftcap, pNe01, pNe02,
                   pGqaRatio, pStrideQ1, pStrideQ2, pStrideK, pStrideV, pStrideMask, pJt, pKb0Start, pKb0Stop);
  NO_DEVICE_CODE;
#endif // defined(VOLTA_MMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE) ||
       // defined(AMD_MFMA_AVAILABLE)
}

template <int DKQ, int DV, int ncols1, int ncols2, bool use_logit_softcap, bool V_is_K_view>
__launch_bounds__(ggml_cuda_fattn_mma_get_nthreads(DKQ, DV, ncols1 *ncols2),
                  ggml_cuda_fattn_mma_get_occupancy(DKQ, DV, ncols1 *ncols2)) static __global__
    void flash_attn_ext_f16(const char *pQPtr, const char *pKPtr, const char *pVPtr, const char *pMaskPtr,
                            const char *pSinksPtr, const int *pKvMaxPtr, float *pDstPtr, float2 *pDstMetaPtr,
                            const float pScale, const float pMaxBias, const float pM0, const float pM1,
                            const uint32_t pNHeadLog2, const float pLogitSoftcap, const int32_t pNe00,
                            const uint3 pNe01, const int32_t pNe02, const int32_t pNe03, const int32_t pNb01,
                            const int32_t pNb02, const int32_t pNb03, const int32_t pNe10, const int32_t pNe11,
                            const int32_t pNe12, const int32_t pNe13, const int32_t pNb11, const int32_t pNb12,
                            const int64_t pNb13, const int32_t pNb21, const int32_t pNb22, const int64_t pNb23,
                            const int32_t pNe31, const int32_t pNe32, const int32_t pNe33, const int32_t pNb31,
                            const int32_t pNb32, const int64_t pNb33) {
  ggml_cuda_pdl_sync(); // TODO optimize placement
#if defined(FLASH_ATTN_AVAILABLE) && (defined(VOLTA_MMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) ||                 \
                                      defined(AMD_WMMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE))
  const char *GGML_CUDA_RESTRICT vQ = pQPtr;
  const char *GGML_CUDA_RESTRICT vK = pKPtr;
  const char *GGML_CUDA_RESTRICT vV = pVPtr;
  const char *GGML_CUDA_RESTRICT vMask = pMaskPtr;
  const char *GGML_CUDA_RESTRICT vSinks = pSinksPtr;
  const int *GGML_CUDA_RESTRICT vKvMax = pKvMaxPtr;
  float *GGML_CUDA_RESTRICT vDst = pDstPtr;
  float2 *GGML_CUDA_RESTRICT vDstMeta = pDstMetaPtr;

  // Skip unused kernel variants for faster compilation:
  if (use_logit_softcap && !(DKQ == 128 || DKQ == 256 || DKQ == 512)) {
    NO_DEVICE_CODE;
    return;
  }
  if (DKQ == 192 && ncols2 != 8 && ncols2 != 16) {
    NO_DEVICE_CODE;
    return;
  }
#ifdef VOLTA_MMA_AVAILABLE
  if (ncols1 * ncols2 < 32) {
    NO_DEVICE_CODE;
    return;
  }
#endif // VOLTA_MMA_AVAILABLE

#if __CUDA_ARCH__ == GGML_CUDA_CC_TURING
  if (ncols1 * ncols2 > 32) {
    NO_DEVICE_CODE;
    return;
  }
#endif // __CUDA_ARCH__ == GGML_CUDA_CC_TURING

#if defined(AMD_WMMA_AVAILABLE)
  if (ncols1 * ncols2 < 16 || ncols2 == 1 || DKQ > 128) {
    NO_DEVICE_CODE;
    return;
  }
#endif // defined(AMD_WMMA_AVAILABLE)

#if defined(AMD_MFMA_AVAILABLE)
  if (ncols1 * ncols2 < 16 || DKQ > 256) {
    NO_DEVICE_CODE;
    return;
  }
#endif // defined(AMD_MFMA_AVAILABLE)

  constexpr int cWarpSize = ggml_cuda_get_physical_warp_size();
  constexpr int cNcols = ncols1 * ncols2;
  constexpr int cNbatchFa = ggml_cuda_fattn_mma_get_nbatch_fa(DKQ, DV, cNcols);
  constexpr int cNthreads = ggml_cuda_fattn_mma_get_nthreads(DKQ, DV, cNcols);
  constexpr int cNwarps = cNthreads / cWarpSize;

  const int cGqaRatio = pNe02 / pNe12; // With grouped query attention there are > 1 Q matrices per K, V matrix.

  const int cStrideQ1 = pNb01 / sizeof(float2);
  const int cStrideQ2 = pNb02 / sizeof(float2);
  const int cStrideK = pNb11 / sizeof(half2);
  const int cStrideMask = pNb31 / sizeof(half);

  const int cStrideV = V_is_K_view ? cStrideK : pNb21 / sizeof(half2);

  const int cIterK = (pNe11 + (cNbatchFa - 1)) / cNbatchFa;
  const int cIterJ = (pNe01.z + (ncols1 - 1)) / ncols1;
  const int cIterZGqa = (cGqaRatio + (ncols2 - 1)) / ncols2;

  // kbc == k block continuous, current index in continuous ijk space.
  int vKbc = int64_t(blockIdx.x + 0) * (cIterK * cIterJ * cIterZGqa * pNe12 * pNe03) / gridDim.x;
  const int cKbcStop = int64_t(blockIdx.x + 1) * (cIterK * cIterJ * cIterZGqa * pNe12 * pNe03) / gridDim.x;

  // If the seams of 2 CUDA blocks fall within an output tile their results need to be combined.
  // For this we need to track both the block that starts the tile (needs_fixup) and the block that finishes the tile
  // (is_fixup). In the most general case >2 seams can fall into the same tile.

  // kb0 == k start index when in the output tile.
  int vKb0Start = vKbc % cIterK;
  int vKb0Stop = min(cIterK, vKb0Start + cKbcStop - vKbc);

  while (vKbc < cKbcStop && vKb0Stop == cIterK) {
    // z_KV == K/V head index, zt_gqa = Q head start index per K/V head, jt = token position start index
    const int cSequence = vKbc / (cIterK * cIterJ * cIterZGqa * pNe12);
    const int cZKv = (vKbc - cIterK * cIterJ * cIterZGqa * pNe12 * cSequence) / (cIterK * cIterJ * cIterZGqa);
    const int cZtGqa = (vKbc - cIterK * cIterJ * cIterZGqa * pNe12 * cSequence - cIterK * cIterJ * cIterZGqa * cZKv) /
                       (cIterK * cIterJ);
    const int cJt = (vKbc - cIterK * cIterJ * cIterZGqa * pNe12 * cSequence - cIterK * cIterJ * cIterZGqa * cZKv -
                     cIterK * cIterJ * cZtGqa) /
                    cIterK;

    const int cZtQ = cZKv * cGqaRatio + cZtGqa * ncols2; // Global Q head start index.

    const float2 *vQF2 = (const float2 *)(vQ + pNb03 * cSequence + pNb02 * cZtQ);
    const half2 *vKH2 = (const half2 *)(vK + pNb13 * cSequence + pNb12 * cZKv);
    const half *vMaskH = ncols2 == 1 && !vMask ? nullptr : (const half *)(vMask + pNb33 * (cSequence % pNe33));
    float2 *vDstk = ((float2 *)vDst) + (cSequence * pNe01.z * pNe02 + cZtQ) * (DV / 2);

    const half2 *vVH2 = V_is_K_view ? vKH2 : (const half2 *)(vV + pNb23 * cSequence + pNb22 * cZKv);
    const float *vSinksF = vSinks ? (const float *)vSinks + cZtQ : nullptr;

    const float cSlope = ncols2 == 1 ? get_alibi_slope(pMaxBias, cZtQ, pNHeadLog2, pM0, pM1) : 1.0f;

    if (vKvMax) {
      vKb0Stop = min(vKb0Stop, vKvMax[cSequence * cIterJ + cJt] / cNbatchFa);
    }
    constexpr bool cIsFixup =
        false; // All but (potentially) the last iterations write their data to dst rather than the fixup buffer.
    if (vKb0Start == 0) {
      constexpr bool cNeedsFixup = false; // CUDA block is working on an entire tile.
      flash_attn_ext_f16_process_tile<DKQ, DV, ncols1, ncols2, cNwarps, use_logit_softcap, V_is_K_view, cNeedsFixup,
                                      cIsFixup>(vQF2, vKH2, vVH2, vMaskH, vSinksF, vDstk, vDstMeta, pScale, cSlope,
                                                pLogitSoftcap, pNe01, pNe02, cGqaRatio, pNe11, cStrideQ1, cStrideQ2,
                                                cStrideK, cStrideV, cStrideMask, cJt, cZtGqa, vKb0Start, vKb0Stop);
    } else {
      constexpr bool cNeedsFixup = true; // CUDA block is missing the beginning of a tile.
      flash_attn_ext_f16_process_tile<DKQ, DV, ncols1, ncols2, cNwarps, use_logit_softcap, V_is_K_view, cNeedsFixup,
                                      cIsFixup>(vQF2, vKH2, vVH2, vMaskH, vSinksF, vDstk, vDstMeta, pScale, cSlope,
                                                pLogitSoftcap, pNe01, pNe02, cGqaRatio, pNe11, cStrideQ1, cStrideQ2,
                                                cStrideK, cStrideV, cStrideMask, cJt, cZtGqa, vKb0Start, vKb0Stop);
    }

    vKbc += cIterK;
    vKbc -= vKbc % cIterK;

    vKb0Start = 0;
    vKb0Stop = min(cIterK, cKbcStop - vKbc);
  }

  if (vKbc >= cKbcStop) {
    return;
  }

  // z_KV == K/V head index, zt_gqa = Q head start index per K/V head, jt = token position start index.
  const int cSequence = vKbc / (cIterK * cIterJ * cIterZGqa * pNe12);
  const int cZKv = (vKbc - cIterK * cIterJ * cIterZGqa * pNe12 * cSequence) / (cIterK * cIterJ * cIterZGqa);
  const int cZtGqa =
      (vKbc - cIterK * cIterJ * cIterZGqa * pNe12 * cSequence - cIterK * cIterJ * cIterZGqa * cZKv) / (cIterK * cIterJ);
  const int cJt = (vKbc - cIterK * cIterJ * cIterZGqa * pNe12 * cSequence - cIterK * cIterJ * cIterZGqa * cZKv -
                   cIterK * cIterJ * cZtGqa) /
                  cIterK;

  const int cZtQ = cZKv * cGqaRatio + cZtGqa * ncols2; // Global Q head start index.

  const float2 *vQF2 = (const float2 *)(vQ + pNb03 * cSequence + pNb02 * cZtQ);
  const half2 *vKH2 = (const half2 *)(vK + pNb13 * cSequence + pNb12 * cZKv);
  const half *vMaskH = ncols2 == 1 && !vMask ? nullptr : (const half *)(vMask + pNb33 * (cSequence % pNe33));
  float2 *vDstk = ((float2 *)vDst) + (cSequence * pNe01.z * pNe02 + cZtQ) * (DV / 2);

  const half2 *vVH2 = V_is_K_view ? vKH2 : (const half2 *)(vV + pNb23 * cSequence + pNb22 * cZKv);
  const float *vSinksF = vSinks ? (const float *)vSinks + cZtQ : nullptr;

  const float cSlope = ncols2 == 1 ? get_alibi_slope(pMaxBias, cZtQ, pNHeadLog2, pM0, pM1) : 1.0f;

  if (vKvMax) {
    vKb0Stop = min(vKb0Stop, vKvMax[cSequence * cIterJ + cJt] / cNbatchFa);
  }

  constexpr bool cIsFixup = true; // Last index writes its data to fixup buffer to avoid data races with other blocks.
  constexpr bool cNeedsFixup = false;
  flash_attn_ext_f16_process_tile<DKQ, DV, ncols1, ncols2, cNwarps, use_logit_softcap, V_is_K_view, cNeedsFixup,
                                  cIsFixup>(vQF2, vKH2, vVH2, vMaskH, vSinksF, vDstk, vDstMeta, pScale, cSlope,
                                            pLogitSoftcap, pNe01, pNe02, cGqaRatio, pNe11, cStrideQ1, cStrideQ2,
                                            cStrideK, cStrideV, cStrideMask, cJt, cZtGqa, vKb0Start, vKb0Stop);
#else
  GGML_UNUSED_VARS(pQPtr, pKPtr, pVPtr, pMaskPtr, pSinksPtr, pKvMaxPtr, pDstPtr, pDstMetaPtr, pScale, pMaxBias, pM0,
                   pM1, pNHeadLog2, pLogitSoftcap, pNe00, pNe01, pNe02, pNe03, pNb01, pNb02, pNb03, pNe10, pNe11, pNe12,
                   pNe13, pNb11, pNb12, pNb13, pNb21, pNb22, pNb23, pNe31, pNe32, pNe33, pNb31, pNb32, pNb33);
  NO_DEVICE_CODE;
#endif // defined(FLASH_ATTN_AVAILABLE) && (defined(VOLTA_MMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) ||
       // defined(AMD_WMMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE))
}

template <int DKQ, int DV, int ncols1, int ncols2>
void ggml_cuda_flash_attn_ext_mma_f16_case(ggml_backend_cuda_context &pCtx, ggml_tensor *pDst) {
  const ggml_tensor *vKqv = pDst;
  const int cId = ggml_cuda_get_device();
  const int cCc = ggml_cuda_info().devices[cId].cc;

  constexpr int cNcols = ncols1 * ncols2;

  const int cNthreads = ggml_cuda_fattn_mma_get_nthreads(DKQ, DV, cNcols, cCc);
  const int cNbatchFa = ggml_cuda_fattn_mma_get_nbatch_fa(DKQ, DV, cNcols, cCc);
  const int cNbatchK2 = ggml_cuda_fattn_mma_get_nbatch_K2(DKQ, DV, cNcols, cCc);
  const int cNbatchV2 = ggml_cuda_fattn_mma_get_nbatch_V2(DKQ, DV, cNcols, cCc);
  const int cNbatchCombine = ggml_cuda_fattn_mma_get_nbatch_combine(DKQ, DV, cNcols, cCc);
  const bool cQInReg = ggml_cuda_fattn_mma_get_Q_in_reg(DKQ, DV, cNcols, cCc);
  const int cNstages = ggml_cuda_fattn_mma_get_nstages(DKQ, DV, ncols1, ncols2, cCc);

  const int cColsPerWarp = std::min(cNcols, get_cols_per_warp(cCc));
  const int cWarpSizeHost = ggml_cuda_info().devices[pCtx.device].warp_size;
  const int cNwarps = cNthreads / cWarpSizeHost;

  constexpr bool cVIsKView = DKQ == 576; // Guaranteed by the kernel selection logic in fattn.cu

  const size_t cNbytesSharedKv1stage = cNbatchFa * std::max(cNbatchK2 + 4, cNbatchV2 + 4) * sizeof(half2);
  const size_t cNbytesSharedKv2stage = cNbatchFa * (cNbatchK2 + 4 + cNbatchV2 + 4) * sizeof(half2);
  const size_t cNbytesSharedQ = cNcols * (DKQ / 2 + 4) * sizeof(half2);
  const size_t cNbytesSharedMask = ncols1 * (cNbatchFa / 2 + 4) * sizeof(half2);
  const size_t cNbytesSharedCombine = cNwarps * cColsPerWarp * (cNbatchCombine + 4) * sizeof(half2);

  const size_t cNbytesSharedKv = cNstages <= 1 ? cNbytesSharedKv1stage : cNbytesSharedKv2stage;

  const size_t cNbytesSharedTotal =
      std::max(cNbytesSharedCombine, cQInReg ? std::max(cNbytesSharedQ, cNbytesSharedKv + cNbytesSharedMask)
                                             : cNbytesSharedQ + cNbytesSharedKv + cNbytesSharedMask);

  float vLogitSoftcap;
  memcpy(&vLogitSoftcap, (const float *)vKqv->op_params + 2, sizeof(float));

#if defined(GGML_USE_HIP)
  using fattn_kernel_ptr_t = const void *;
#else
  using fattn_kernel_ptr_t = fattn_kernel_t;
#endif // defined(GGML_USE_HIP)
  fattn_kernel_t vFattnKernel;
  if (vLogitSoftcap == 0.0f) {
    constexpr bool cUseLogitSoftcap = false;
    vFattnKernel = flash_attn_ext_f16<DKQ, DV, ncols1, ncols2, cUseLogitSoftcap, cVIsKView>;

    static bool vSharedMemoryLimitRaised[GGML_CUDA_MAX_DEVICES] = {false};
    if (!vSharedMemoryLimitRaised[cId]) {
      CUDA_CHECK(cudaFuncSetAttribute(reinterpret_cast<fattn_kernel_ptr_t>(vFattnKernel),
                                      cudaFuncAttributeMaxDynamicSharedMemorySize, cNbytesSharedTotal));
      vSharedMemoryLimitRaised[cId] = true;
    }
  } else {
    constexpr bool cUseLogitSoftcap = true;
    vFattnKernel = flash_attn_ext_f16<DKQ, DV, ncols1, ncols2, cUseLogitSoftcap, cVIsKView>;

    static bool vSharedMemoryLimitRaised[GGML_CUDA_MAX_DEVICES] = {false};
    if (!vSharedMemoryLimitRaised[cId]) {
      CUDA_CHECK(cudaFuncSetAttribute(reinterpret_cast<fattn_kernel_ptr_t>(vFattnKernel),
                                      cudaFuncAttributeMaxDynamicSharedMemorySize, cNbytesSharedTotal));
      vSharedMemoryLimitRaised[cId] = true;
    }
  }

  launch_fattn<DV, ncols1, ncols2>(pCtx, pDst, vFattnKernel, cNwarps, cNbytesSharedTotal, cNbatchFa, true, true, true,
                                   cWarpSizeHost);
}

#define DECL_FATTN_MMA_F16_CASE(DKQ, DV, ncols1, ncols2)                                                               \
  template void ggml_cuda_flash_attn_ext_mma_f16_case<DKQ, DV, ncols1, ncols2>(ggml_backend_cuda_context & ctx,        \
                                                                               ggml_tensor * dst)

#define DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2(DKQ, DV, ncols)                                                             \
  extern DECL_FATTN_MMA_F16_CASE(DKQ, DV, (ncols) / 1, 1);                                                             \
  extern DECL_FATTN_MMA_F16_CASE(DKQ, DV, (ncols) / 2, 2);                                                             \
  extern DECL_FATTN_MMA_F16_CASE(DKQ, DV, (ncols) / 4, 4);                                                             \
  extern DECL_FATTN_MMA_F16_CASE(DKQ, DV, (ncols) / 8, 8);                                                             \
  extern DECL_FATTN_MMA_F16_CASE(DKQ, DV, (ncols) / 16, 16);

DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2(64, 64, 8)
DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2(80, 80, 8)
DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2(96, 96, 8)
DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2(112, 112, 8)
DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2(128, 128, 8)
DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2(256, 256, 8)

DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2(64, 64, 16)
DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2(80, 80, 16)
DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2(96, 96, 16)
DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2(112, 112, 16)
DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2(128, 128, 16)
DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2(256, 256, 16)

DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2(64, 64, 32)
DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2(80, 80, 32)
DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2(96, 96, 32)
DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2(112, 112, 32)
DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2(128, 128, 32)
DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2(256, 256, 32)

DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2(64, 64, 64)
DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2(80, 80, 64)
DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2(96, 96, 64)
DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2(112, 112, 64)
DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2(128, 128, 64)
DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2(256, 256, 64)

extern DECL_FATTN_MMA_F16_CASE(512, 512, 4, 2);
extern DECL_FATTN_MMA_F16_CASE(512, 512, 8, 2);
extern DECL_FATTN_MMA_F16_CASE(512, 512, 16, 2);
extern DECL_FATTN_MMA_F16_CASE(512, 512, 32, 2);
extern DECL_FATTN_MMA_F16_CASE(512, 512, 2, 4);
extern DECL_FATTN_MMA_F16_CASE(512, 512, 4, 4);
extern DECL_FATTN_MMA_F16_CASE(512, 512, 8, 4);
extern DECL_FATTN_MMA_F16_CASE(512, 512, 16, 4);
extern DECL_FATTN_MMA_F16_CASE(512, 512, 1, 8);
extern DECL_FATTN_MMA_F16_CASE(512, 512, 2, 8);
extern DECL_FATTN_MMA_F16_CASE(512, 512, 4, 8);
extern DECL_FATTN_MMA_F16_CASE(512, 512, 8, 8);

// The number of viable configurations for Deepseek is very limited:
extern DECL_FATTN_MMA_F16_CASE(576, 512, 1, 16);
extern DECL_FATTN_MMA_F16_CASE(576, 512, 2, 16);
extern DECL_FATTN_MMA_F16_CASE(576, 512, 4, 16);

// Mistral Small 4 (DKQ=320, DV=256), GQA=32-only build:
extern DECL_FATTN_MMA_F16_CASE(320, 256, 1, 32);
extern DECL_FATTN_MMA_F16_CASE(320, 256, 2, 32);

// For GLM 4.7 Flash
extern DECL_FATTN_MMA_F16_CASE(576, 512, 4, 4);
extern DECL_FATTN_MMA_F16_CASE(576, 512, 8, 4);
extern DECL_FATTN_MMA_F16_CASE(576, 512, 16, 4);
extern DECL_FATTN_MMA_F16_CASE(576, 512, 1, 32);
extern DECL_FATTN_MMA_F16_CASE(576, 512, 2, 32);
