#include "common.cuh"
#include "fattn-common.cuh"

// nbatch_fa == number of KQ rows to process per iteration
// nbatch_K == number of K columns to load in parallel for KQ calculation

// TODO optimize kernel parameters for FP16 NVIDIA (P100)
// TODO optimize kernel parameters for head sizes 40, 72, 80, 96, 112

// The ROCm compiler cannot handle templating in __launch_bounds__.
// As a workaround, define a macro to package the kernel parameters as uint32_t:
#define GGML_CUDA_FATTN_TILE_CONFIG_CASE(DKQ_, DV_, ncols_, nthreads, occupancy, nbatch_fa, nbatch_K)                  \
  if (DKQ == (DKQ_) && DV == (DV_) && ncols == (ncols_)) {                                                             \
    static_assert((nthreads) <= 512, "bad nthreads");                                                                  \
    static_assert((occupancy) <= 8, "bad occupancy");                                                                  \
    static_assert((nbatch_fa) <= 256, "bad nbatch_fa");                                                                \
    static_assert((nbatch_K) <= 256, "bad nbatch_K");                                                                  \
    return ((nthreads) << 0) | ((occupancy) << 10) | ((nbatch_fa) << 14) | ((nbatch_K) << 23);                         \
  }

static constexpr __host__ __device__ uint32_t ggml_cuda_fattn_tile_get_config_nvidia_fp16(const int DKQ, const int DV,
                                                                                          const int ncols) {
  GGML_CUDA_FATTN_TILE_CONFIG_CASE(40, 40, 2, 64, 2, 64, 40)
  GGML_CUDA_FATTN_TILE_CONFIG_CASE(40, 40, 4, 128, 2, 64, 40)
  GGML_CUDA_FATTN_TILE_CONFIG_CASE(40, 40, 8, 256, 2, 64, 40)
  GGML_CUDA_FATTN_TILE_CONFIG_CASE(40, 40, 16, 256, 2, 64, 40)
  GGML_CUDA_FATTN_TILE_CONFIG_CASE(40, 40, 32, 256, 2, 64, 40)

  GGML_CUDA_FATTN_TILE_CONFIG_CASE(64, 64, 2, 64, 2, 64, 64)
  GGML_CUDA_FATTN_TILE_CONFIG_CASE(64, 64, 4, 128, 2, 64, 64)
  GGML_CUDA_FATTN_TILE_CONFIG_CASE(64, 64, 8, 256, 2, 64, 64)
  GGML_CUDA_FATTN_TILE_CONFIG_CASE(64, 64, 16, 256, 2, 64, 64)
  GGML_CUDA_FATTN_TILE_CONFIG_CASE(64, 64, 32, 256, 2, 64, 64)

  GGML_CUDA_FATTN_TILE_CONFIG_CASE(72, 72, 2, 64, 2, 64, 72)
  GGML_CUDA_FATTN_TILE_CONFIG_CASE(72, 72, 4, 128, 2, 64, 72)
  GGML_CUDA_FATTN_TILE_CONFIG_CASE(72, 72, 8, 256, 2, 64, 72)
  GGML_CUDA_FATTN_TILE_CONFIG_CASE(72, 72, 16, 256, 2, 64, 72)
  GGML_CUDA_FATTN_TILE_CONFIG_CASE(72, 72, 32, 256, 2, 64, 72)

  GGML_CUDA_FATTN_TILE_CONFIG_CASE(80, 80, 2, 64, 2, 64, 40)
  GGML_CUDA_FATTN_TILE_CONFIG_CASE(80, 80, 4, 128, 2, 64, 40)
  GGML_CUDA_FATTN_TILE_CONFIG_CASE(80, 80, 8, 256, 2, 64, 40)
  GGML_CUDA_FATTN_TILE_CONFIG_CASE(80, 80, 16, 256, 2, 64, 40)
  GGML_CUDA_FATTN_TILE_CONFIG_CASE(80, 80, 32, 256, 2, 64, 40)

  GGML_CUDA_FATTN_TILE_CONFIG_CASE(96, 96, 2, 64, 2, 64, 48)
  GGML_CUDA_FATTN_TILE_CONFIG_CASE(96, 96, 4, 128, 2, 64, 48)
  GGML_CUDA_FATTN_TILE_CONFIG_CASE(96, 96, 8, 256, 2, 64, 48)
  GGML_CUDA_FATTN_TILE_CONFIG_CASE(96, 96, 16, 256, 2, 64, 48)
  GGML_CUDA_FATTN_TILE_CONFIG_CASE(96, 96, 32, 256, 2, 64, 48)

  GGML_CUDA_FATTN_TILE_CONFIG_CASE(112, 112, 2, 64, 2, 64, 56)
  GGML_CUDA_FATTN_TILE_CONFIG_CASE(112, 112, 4, 128, 2, 64, 56)
  GGML_CUDA_FATTN_TILE_CONFIG_CASE(112, 112, 8, 256, 2, 64, 56)
  GGML_CUDA_FATTN_TILE_CONFIG_CASE(112, 112, 16, 256, 2, 64, 56)
  GGML_CUDA_FATTN_TILE_CONFIG_CASE(112, 112, 32, 256, 2, 64, 56)

  GGML_CUDA_FATTN_TILE_CONFIG_CASE(128, 128, 2, 64, 2, 64, 64)
  GGML_CUDA_FATTN_TILE_CONFIG_CASE(128, 128, 4, 128, 2, 64, 64)
  GGML_CUDA_FATTN_TILE_CONFIG_CASE(128, 128, 8, 256, 2, 64, 64)
  GGML_CUDA_FATTN_TILE_CONFIG_CASE(128, 128, 16, 256, 2, 64, 64)
  GGML_CUDA_FATTN_TILE_CONFIG_CASE(128, 128, 32, 256, 2, 64, 64)

  GGML_CUDA_FATTN_TILE_CONFIG_CASE(192, 128, 2, 64, 2, 64, 64)
  GGML_CUDA_FATTN_TILE_CONFIG_CASE(192, 128, 4, 128, 2, 64, 64)
  GGML_CUDA_FATTN_TILE_CONFIG_CASE(192, 128, 8, 256, 2, 64, 64)
  GGML_CUDA_FATTN_TILE_CONFIG_CASE(192, 128, 16, 256, 2, 64, 64)
  GGML_CUDA_FATTN_TILE_CONFIG_CASE(192, 128, 32, 256, 2, 64, 64)

  GGML_CUDA_FATTN_TILE_CONFIG_CASE(256, 256, 2, 64, 2, 64, 64)
  GGML_CUDA_FATTN_TILE_CONFIG_CASE(256, 256, 4, 128, 2, 64, 64)
  GGML_CUDA_FATTN_TILE_CONFIG_CASE(256, 256, 8, 256, 2, 64, 64)
  GGML_CUDA_FATTN_TILE_CONFIG_CASE(256, 256, 16, 256, 2, 64, 64)
  GGML_CUDA_FATTN_TILE_CONFIG_CASE(256, 256, 32, 256, 2, 64, 64)

  GGML_CUDA_FATTN_TILE_CONFIG_CASE(320, 256, 16, 256, 2, 64, 64)

  GGML_CUDA_FATTN_TILE_CONFIG_CASE(512, 512, 2, 64, 2, 64, 64)
  GGML_CUDA_FATTN_TILE_CONFIG_CASE(512, 512, 4, 128, 2, 64, 64)
  GGML_CUDA_FATTN_TILE_CONFIG_CASE(512, 512, 8, 256, 2, 64, 64)
  GGML_CUDA_FATTN_TILE_CONFIG_CASE(512, 512, 16, 256, 2, 64, 64)

  GGML_CUDA_FATTN_TILE_CONFIG_CASE(576, 512, 4, 128, 2, 64, 64)
  GGML_CUDA_FATTN_TILE_CONFIG_CASE(576, 512, 8, 256, 2, 64, 64)
  GGML_CUDA_FATTN_TILE_CONFIG_CASE(576, 512, 16, 256, 2, 64, 64)

  return 0;
}

static constexpr __host__ __device__ uint32_t ggml_cuda_fattn_tile_get_config_nvidia_fp32(const int DKQ, const int DV,
                                                                                          const int ncols) {
  GGML_CUDA_FATTN_TILE_CONFIG_CASE(40, 40, 2, 64, 2, 32, 40)
  GGML_CUDA_FATTN_TILE_CONFIG_CASE(40, 40, 4, 128, 2, 32, 40)
  GGML_CUDA_FATTN_TILE_CONFIG_CASE(40, 40, 8, 256, 2, 32, 40)
  GGML_CUDA_FATTN_TILE_CONFIG_CASE(40, 40, 16, 256, 2, 32, 40)
  GGML_CUDA_FATTN_TILE_CONFIG_CASE(40, 40, 32, 256, 2, 32, 40)

  GGML_CUDA_FATTN_TILE_CONFIG_CASE(64, 64, 2, 128, 3, 64, 64)
  GGML_CUDA_FATTN_TILE_CONFIG_CASE(64, 64, 4, 128, 3, 32, 64)
  GGML_CUDA_FATTN_TILE_CONFIG_CASE(64, 64, 8, 128, 3, 32, 64)
  GGML_CUDA_FATTN_TILE_CONFIG_CASE(64, 64, 16, 128, 3, 64, 64)
  GGML_CUDA_FATTN_TILE_CONFIG_CASE(64, 64, 32, 256, 2, 64, 64)

  GGML_CUDA_FATTN_TILE_CONFIG_CASE(72, 72, 2, 64, 2, 32, 72)
  GGML_CUDA_FATTN_TILE_CONFIG_CASE(72, 72, 4, 128, 2, 32, 72)
  GGML_CUDA_FATTN_TILE_CONFIG_CASE(72, 72, 8, 256, 2, 32, 72)
  GGML_CUDA_FATTN_TILE_CONFIG_CASE(72, 72, 16, 256, 2, 32, 72)
  GGML_CUDA_FATTN_TILE_CONFIG_CASE(72, 72, 32, 256, 2, 32, 72)

  GGML_CUDA_FATTN_TILE_CONFIG_CASE(80, 80, 2, 64, 2, 32, 40)
  GGML_CUDA_FATTN_TILE_CONFIG_CASE(80, 80, 4, 128, 2, 32, 40)
  GGML_CUDA_FATTN_TILE_CONFIG_CASE(80, 80, 8, 256, 2, 32, 40)
  GGML_CUDA_FATTN_TILE_CONFIG_CASE(80, 80, 16, 256, 2, 32, 40)
  GGML_CUDA_FATTN_TILE_CONFIG_CASE(80, 80, 32, 256, 2, 32, 40)

  GGML_CUDA_FATTN_TILE_CONFIG_CASE(96, 96, 2, 64, 2, 32, 48)
  GGML_CUDA_FATTN_TILE_CONFIG_CASE(96, 96, 4, 128, 2, 32, 48)
  GGML_CUDA_FATTN_TILE_CONFIG_CASE(96, 96, 8, 256, 2, 32, 48)
  GGML_CUDA_FATTN_TILE_CONFIG_CASE(96, 96, 16, 256, 2, 32, 48)
  GGML_CUDA_FATTN_TILE_CONFIG_CASE(96, 96, 32, 256, 2, 32, 48)

  GGML_CUDA_FATTN_TILE_CONFIG_CASE(112, 112, 2, 64, 2, 32, 56)
  GGML_CUDA_FATTN_TILE_CONFIG_CASE(112, 112, 4, 128, 2, 32, 56)
  GGML_CUDA_FATTN_TILE_CONFIG_CASE(112, 112, 8, 256, 2, 32, 56)
  GGML_CUDA_FATTN_TILE_CONFIG_CASE(112, 112, 16, 256, 2, 32, 56)
  GGML_CUDA_FATTN_TILE_CONFIG_CASE(112, 112, 32, 256, 2, 32, 56)

  GGML_CUDA_FATTN_TILE_CONFIG_CASE(128, 128, 2, 128, 3, 64, 64)
  GGML_CUDA_FATTN_TILE_CONFIG_CASE(128, 128, 4, 128, 3, 32, 128)
  GGML_CUDA_FATTN_TILE_CONFIG_CASE(128, 128, 8, 128, 3, 64, 128)
  GGML_CUDA_FATTN_TILE_CONFIG_CASE(128, 128, 16, 128, 3, 32, 128)
  GGML_CUDA_FATTN_TILE_CONFIG_CASE(128, 128, 32, 256, 2, 64, 64)

  GGML_CUDA_FATTN_TILE_CONFIG_CASE(192, 128, 2, 128, 3, 64, 64)
  GGML_CUDA_FATTN_TILE_CONFIG_CASE(192, 128, 4, 128, 3, 32, 64)
  GGML_CUDA_FATTN_TILE_CONFIG_CASE(192, 128, 8, 256, 2, 32, 64)
  GGML_CUDA_FATTN_TILE_CONFIG_CASE(192, 128, 16, 256, 2, 32, 64)
  GGML_CUDA_FATTN_TILE_CONFIG_CASE(192, 128, 32, 256, 2, 32, 64)

  GGML_CUDA_FATTN_TILE_CONFIG_CASE(256, 256, 2, 128, 3, 64, 64)
  GGML_CUDA_FATTN_TILE_CONFIG_CASE(256, 256, 4, 128, 3, 32, 64)
  GGML_CUDA_FATTN_TILE_CONFIG_CASE(256, 256, 8, 256, 2, 32, 256)
  GGML_CUDA_FATTN_TILE_CONFIG_CASE(256, 256, 16, 256, 2, 32, 128)
  GGML_CUDA_FATTN_TILE_CONFIG_CASE(256, 256, 32, 256, 2, 32, 64)

  GGML_CUDA_FATTN_TILE_CONFIG_CASE(320, 256, 16, 256, 2, 32, 64)

  GGML_CUDA_FATTN_TILE_CONFIG_CASE(512, 512, 2, 64, 2, 32, 64)
  GGML_CUDA_FATTN_TILE_CONFIG_CASE(512, 512, 4, 128, 2, 32, 64)
  GGML_CUDA_FATTN_TILE_CONFIG_CASE(512, 512, 8, 256, 2, 32, 64)
  GGML_CUDA_FATTN_TILE_CONFIG_CASE(512, 512, 16, 256, 2, 32, 64)

  GGML_CUDA_FATTN_TILE_CONFIG_CASE(576, 512, 4, 128, 2, 32, 64)
  GGML_CUDA_FATTN_TILE_CONFIG_CASE(576, 512, 8, 256, 2, 32, 64)
  GGML_CUDA_FATTN_TILE_CONFIG_CASE(576, 512, 16, 256, 2, 32, 64)

  return 0;
}

static constexpr __host__ __device__ uint32_t ggml_cuda_fattn_tile_get_config_amd(const int DKQ, const int DV,
                                                                                  const int ncols) {
  GGML_CUDA_FATTN_TILE_CONFIG_CASE(40, 40, 2, 64, 2, 32, 40)
  GGML_CUDA_FATTN_TILE_CONFIG_CASE(40, 40, 4, 128, 2, 32, 40)
  GGML_CUDA_FATTN_TILE_CONFIG_CASE(40, 40, 8, 256, 2, 32, 40)
  GGML_CUDA_FATTN_TILE_CONFIG_CASE(40, 40, 16, 256, 2, 32, 40)
  GGML_CUDA_FATTN_TILE_CONFIG_CASE(40, 40, 32, 256, 2, 32, 40)
  GGML_CUDA_FATTN_TILE_CONFIG_CASE(40, 40, 64, 256, 2, 32, 40)

  GGML_CUDA_FATTN_TILE_CONFIG_CASE(64, 64, 2, 64, 3, 32, 64)
  GGML_CUDA_FATTN_TILE_CONFIG_CASE(64, 64, 4, 128, 3, 64, 64)
  GGML_CUDA_FATTN_TILE_CONFIG_CASE(64, 64, 8, 128, 2, 32, 64)
  GGML_CUDA_FATTN_TILE_CONFIG_CASE(64, 64, 16, 256, 2, 128, 64)
  GGML_CUDA_FATTN_TILE_CONFIG_CASE(64, 64, 32, 256, 2, 64, 64)
  GGML_CUDA_FATTN_TILE_CONFIG_CASE(64, 64, 64, 256, 2, 64, 64)

  GGML_CUDA_FATTN_TILE_CONFIG_CASE(72, 72, 2, 64, 2, 32, 72)
  GGML_CUDA_FATTN_TILE_CONFIG_CASE(72, 72, 4, 128, 2, 32, 72)
  GGML_CUDA_FATTN_TILE_CONFIG_CASE(72, 72, 8, 256, 2, 32, 72)
  GGML_CUDA_FATTN_TILE_CONFIG_CASE(72, 72, 16, 256, 2, 32, 72)
  GGML_CUDA_FATTN_TILE_CONFIG_CASE(72, 72, 32, 256, 2, 32, 72)
  GGML_CUDA_FATTN_TILE_CONFIG_CASE(72, 72, 64, 256, 2, 32, 72)

  GGML_CUDA_FATTN_TILE_CONFIG_CASE(80, 80, 2, 64, 2, 32, 40)
  GGML_CUDA_FATTN_TILE_CONFIG_CASE(80, 80, 4, 128, 2, 32, 40)
  GGML_CUDA_FATTN_TILE_CONFIG_CASE(80, 80, 8, 256, 2, 32, 40)
  GGML_CUDA_FATTN_TILE_CONFIG_CASE(80, 80, 16, 256, 2, 32, 40)
  GGML_CUDA_FATTN_TILE_CONFIG_CASE(80, 80, 32, 256, 2, 32, 40)
  GGML_CUDA_FATTN_TILE_CONFIG_CASE(80, 80, 64, 256, 2, 32, 40)

  GGML_CUDA_FATTN_TILE_CONFIG_CASE(96, 96, 2, 64, 2, 32, 48)
  GGML_CUDA_FATTN_TILE_CONFIG_CASE(96, 96, 4, 128, 2, 32, 48)
  GGML_CUDA_FATTN_TILE_CONFIG_CASE(96, 96, 8, 256, 2, 32, 48)
  GGML_CUDA_FATTN_TILE_CONFIG_CASE(96, 96, 16, 256, 2, 32, 48)
  GGML_CUDA_FATTN_TILE_CONFIG_CASE(96, 96, 32, 256, 2, 32, 48)
  GGML_CUDA_FATTN_TILE_CONFIG_CASE(96, 96, 64, 256, 2, 32, 48)

  GGML_CUDA_FATTN_TILE_CONFIG_CASE(112, 112, 2, 64, 2, 32, 56)
  GGML_CUDA_FATTN_TILE_CONFIG_CASE(112, 112, 4, 128, 2, 32, 56)
  GGML_CUDA_FATTN_TILE_CONFIG_CASE(112, 112, 8, 256, 2, 32, 56)
  GGML_CUDA_FATTN_TILE_CONFIG_CASE(112, 112, 16, 256, 2, 32, 56)
  GGML_CUDA_FATTN_TILE_CONFIG_CASE(112, 112, 32, 256, 2, 32, 56)
  GGML_CUDA_FATTN_TILE_CONFIG_CASE(112, 112, 64, 256, 2, 32, 56)

  GGML_CUDA_FATTN_TILE_CONFIG_CASE(128, 128, 2, 256, 2, 128, 64)
  GGML_CUDA_FATTN_TILE_CONFIG_CASE(128, 128, 4, 128, 2, 64, 128)
  GGML_CUDA_FATTN_TILE_CONFIG_CASE(128, 128, 8, 256, 2, 64, 128)
  GGML_CUDA_FATTN_TILE_CONFIG_CASE(128, 128, 16, 256, 2, 64, 128)
  GGML_CUDA_FATTN_TILE_CONFIG_CASE(128, 128, 32, 256, 2, 64, 64)
  GGML_CUDA_FATTN_TILE_CONFIG_CASE(128, 128, 64, 256, 2, 64, 32)

  GGML_CUDA_FATTN_TILE_CONFIG_CASE(192, 128, 2, 256, 2, 128, 64)
  GGML_CUDA_FATTN_TILE_CONFIG_CASE(192, 128, 4, 256, 2, 64, 64)
  GGML_CUDA_FATTN_TILE_CONFIG_CASE(192, 128, 8, 256, 2, 64, 64)
  GGML_CUDA_FATTN_TILE_CONFIG_CASE(192, 128, 16, 256, 2, 32, 64)
  GGML_CUDA_FATTN_TILE_CONFIG_CASE(192, 128, 32, 256, 2, 32, 64)

  GGML_CUDA_FATTN_TILE_CONFIG_CASE(256, 256, 2, 256, 2, 128, 64)
  GGML_CUDA_FATTN_TILE_CONFIG_CASE(256, 256, 4, 256, 2, 64, 128)
  GGML_CUDA_FATTN_TILE_CONFIG_CASE(256, 256, 8, 256, 2, 64, 128)
  GGML_CUDA_FATTN_TILE_CONFIG_CASE(256, 256, 16, 256, 2, 32, 128)
  GGML_CUDA_FATTN_TILE_CONFIG_CASE(256, 256, 32, 256, 2, 32, 128)

  GGML_CUDA_FATTN_TILE_CONFIG_CASE(320, 256, 32, 512, 1, 128, 64)

  GGML_CUDA_FATTN_TILE_CONFIG_CASE(512, 512, 2, 64, 2, 64, 64)
  GGML_CUDA_FATTN_TILE_CONFIG_CASE(512, 512, 4, 128, 2, 64, 64)
  GGML_CUDA_FATTN_TILE_CONFIG_CASE(512, 512, 8, 256, 2, 64, 64)
  GGML_CUDA_FATTN_TILE_CONFIG_CASE(512, 512, 16, 256, 2, 64, 64)
  GGML_CUDA_FATTN_TILE_CONFIG_CASE(512, 512, 32, 512, 1, 128, 64)

  GGML_CUDA_FATTN_TILE_CONFIG_CASE(576, 512, 4, 128, 2, 64, 64)
  GGML_CUDA_FATTN_TILE_CONFIG_CASE(576, 512, 8, 256, 2, 64, 64)
  GGML_CUDA_FATTN_TILE_CONFIG_CASE(576, 512, 16, 256, 2, 64, 64)
  GGML_CUDA_FATTN_TILE_CONFIG_CASE(576, 512, 32, 512, 1, 128, 64)

  return 0;
}

static constexpr __host__ __device__ uint32_t ggml_cuda_fattn_tile_get_config_amd_rdna(const int DKQ, const int DV,
                                                                                       const int ncols) {
  GGML_CUDA_FATTN_TILE_CONFIG_CASE(40, 40, 2, 64, 2, 32, 40)
  GGML_CUDA_FATTN_TILE_CONFIG_CASE(40, 40, 4, 128, 2, 32, 40)
  GGML_CUDA_FATTN_TILE_CONFIG_CASE(40, 40, 8, 256, 2, 32, 40)
  GGML_CUDA_FATTN_TILE_CONFIG_CASE(40, 40, 16, 256, 2, 32, 40)
  GGML_CUDA_FATTN_TILE_CONFIG_CASE(40, 40, 32, 256, 2, 32, 40)
  GGML_CUDA_FATTN_TILE_CONFIG_CASE(40, 40, 64, 256, 2, 32, 40)

  GGML_CUDA_FATTN_TILE_CONFIG_CASE(64, 64, 2, 64, 8, 32, 64)
  GGML_CUDA_FATTN_TILE_CONFIG_CASE(64, 64, 4, 64, 8, 32, 64)
  GGML_CUDA_FATTN_TILE_CONFIG_CASE(64, 64, 8, 128, 5, 128, 64)
  GGML_CUDA_FATTN_TILE_CONFIG_CASE(64, 64, 16, 128, 5, 128, 64)
  GGML_CUDA_FATTN_TILE_CONFIG_CASE(64, 64, 32, 128, 4, 64, 64)
  GGML_CUDA_FATTN_TILE_CONFIG_CASE(64, 64, 64, 128, 5, 64, 64)

  GGML_CUDA_FATTN_TILE_CONFIG_CASE(72, 72, 2, 64, 2, 32, 72)
  GGML_CUDA_FATTN_TILE_CONFIG_CASE(72, 72, 4, 128, 2, 32, 72)
  GGML_CUDA_FATTN_TILE_CONFIG_CASE(72, 72, 8, 256, 2, 32, 72)
  GGML_CUDA_FATTN_TILE_CONFIG_CASE(72, 72, 16, 256, 2, 32, 72)
  GGML_CUDA_FATTN_TILE_CONFIG_CASE(72, 72, 32, 256, 2, 32, 72)
  GGML_CUDA_FATTN_TILE_CONFIG_CASE(72, 72, 64, 256, 2, 32, 72)

  GGML_CUDA_FATTN_TILE_CONFIG_CASE(80, 80, 2, 64, 2, 32, 40)
  GGML_CUDA_FATTN_TILE_CONFIG_CASE(80, 80, 4, 128, 2, 32, 40)
  GGML_CUDA_FATTN_TILE_CONFIG_CASE(80, 80, 8, 256, 2, 32, 40)
  GGML_CUDA_FATTN_TILE_CONFIG_CASE(80, 80, 16, 256, 2, 32, 40)
  GGML_CUDA_FATTN_TILE_CONFIG_CASE(80, 80, 32, 256, 2, 32, 40)
  GGML_CUDA_FATTN_TILE_CONFIG_CASE(80, 80, 64, 256, 2, 32, 40)

  GGML_CUDA_FATTN_TILE_CONFIG_CASE(96, 96, 2, 64, 2, 32, 48)
  GGML_CUDA_FATTN_TILE_CONFIG_CASE(96, 96, 4, 128, 2, 32, 48)
  GGML_CUDA_FATTN_TILE_CONFIG_CASE(96, 96, 8, 256, 2, 32, 48)
  GGML_CUDA_FATTN_TILE_CONFIG_CASE(96, 96, 16, 256, 2, 32, 48)
  GGML_CUDA_FATTN_TILE_CONFIG_CASE(96, 96, 32, 256, 2, 32, 48)
  GGML_CUDA_FATTN_TILE_CONFIG_CASE(96, 96, 64, 256, 2, 32, 48)

  GGML_CUDA_FATTN_TILE_CONFIG_CASE(112, 112, 2, 64, 2, 32, 56)
  GGML_CUDA_FATTN_TILE_CONFIG_CASE(112, 112, 4, 128, 2, 32, 56)
  GGML_CUDA_FATTN_TILE_CONFIG_CASE(112, 112, 8, 256, 2, 32, 56)
  GGML_CUDA_FATTN_TILE_CONFIG_CASE(112, 112, 16, 256, 2, 32, 56)
  GGML_CUDA_FATTN_TILE_CONFIG_CASE(112, 112, 32, 256, 2, 32, 56)
  GGML_CUDA_FATTN_TILE_CONFIG_CASE(112, 112, 64, 256, 2, 32, 56)

  GGML_CUDA_FATTN_TILE_CONFIG_CASE(128, 128, 2, 64, 8, 32, 64)
  GGML_CUDA_FATTN_TILE_CONFIG_CASE(128, 128, 4, 128, 8, 64, 64)
  GGML_CUDA_FATTN_TILE_CONFIG_CASE(128, 128, 8, 128, 8, 64, 64)
  GGML_CUDA_FATTN_TILE_CONFIG_CASE(128, 128, 16, 256, 3, 128, 128)
  GGML_CUDA_FATTN_TILE_CONFIG_CASE(128, 128, 32, 256, 3, 128, 64)
  GGML_CUDA_FATTN_TILE_CONFIG_CASE(128, 128, 64, 256, 3, 64, 64)

  GGML_CUDA_FATTN_TILE_CONFIG_CASE(192, 128, 2, 64, 8, 32, 64)
  GGML_CUDA_FATTN_TILE_CONFIG_CASE(192, 128, 4, 128, 6, 32, 64)
  GGML_CUDA_FATTN_TILE_CONFIG_CASE(192, 128, 8, 128, 6, 32, 64)
  GGML_CUDA_FATTN_TILE_CONFIG_CASE(192, 128, 16, 256, 5, 32, 64)
  GGML_CUDA_FATTN_TILE_CONFIG_CASE(192, 128, 32, 256, 3, 64, 64)

  GGML_CUDA_FATTN_TILE_CONFIG_CASE(256, 256, 2, 64, 8, 32, 64)
  GGML_CUDA_FATTN_TILE_CONFIG_CASE(256, 256, 4, 128, 6, 32, 256)
  GGML_CUDA_FATTN_TILE_CONFIG_CASE(256, 256, 8, 128, 6, 32, 256)
  GGML_CUDA_FATTN_TILE_CONFIG_CASE(256, 256, 16, 256, 5, 32, 256)
  GGML_CUDA_FATTN_TILE_CONFIG_CASE(256, 256, 32, 256, 3, 64, 128)

  GGML_CUDA_FATTN_TILE_CONFIG_CASE(320, 256, 32, 256, 2, 128, 64)

  GGML_CUDA_FATTN_TILE_CONFIG_CASE(512, 512, 2, 64, 2, 64, 64)
  GGML_CUDA_FATTN_TILE_CONFIG_CASE(512, 512, 4, 128, 2, 64, 64)
  GGML_CUDA_FATTN_TILE_CONFIG_CASE(512, 512, 8, 256, 2, 64, 64)
  GGML_CUDA_FATTN_TILE_CONFIG_CASE(512, 512, 16, 256, 4, 64, 64)
  GGML_CUDA_FATTN_TILE_CONFIG_CASE(512, 512, 32, 256, 2, 128, 64)

  GGML_CUDA_FATTN_TILE_CONFIG_CASE(576, 512, 4, 128, 2, 64, 64)
  GGML_CUDA_FATTN_TILE_CONFIG_CASE(576, 512, 8, 256, 2, 64, 64)
  GGML_CUDA_FATTN_TILE_CONFIG_CASE(576, 512, 16, 256, 4, 64, 64)
  GGML_CUDA_FATTN_TILE_CONFIG_CASE(576, 512, 32, 256, 2, 128, 64)

  return 0;
}

static __host__ uint32_t ggml_cuda_fattn_tile_get_config(const int pDkq, const int pDv, const int pNcols,
                                                         const int pCc) {
  if (GGML_CUDA_CC_IS_AMD(pCc)) {
    if (GGML_CUDA_CC_IS_RDNA(pCc)) {
      return ggml_cuda_fattn_tile_get_config_amd_rdna(pDkq, pDv, pNcols);
    }
    return ggml_cuda_fattn_tile_get_config_amd(pDkq, pDv, pNcols);
  }
  if (fast_fp16_available(pCc)) {
    return ggml_cuda_fattn_tile_get_config_nvidia_fp16(pDkq, pDv, pNcols);
  }
  return ggml_cuda_fattn_tile_get_config_nvidia_fp32(pDkq, pDv, pNcols);
}

static constexpr __device__ uint32_t ggml_cuda_fattn_tile_get_config(const int pDkq, const int pDv, const int pNcols) {
#ifdef GGML_USE_HIP
#ifdef RDNA
  return ggml_cuda_fattn_tile_get_config_amd_rdna(pDkq, pDv, pNcols);
#else
  return ggml_cuda_fattn_tile_get_config_amd(pDkq, pDv, pNcols);
#endif // RDNA
#else
#ifdef FAST_FP16_AVAILABLE
  return ggml_cuda_fattn_tile_get_config_nvidia_fp16(pDkq, pDv, pNcols);
#else
  return ggml_cuda_fattn_tile_get_config_nvidia_fp32(pDkq, pDv, pNcols);
#endif // FAST_FP16_AVAILABLE
#endif // GGML_USE_HIP
}

static __host__ int ggml_cuda_fattn_tile_get_nthreads(const int pDkq, const int pDv, const int pNcols, const int pCc) {
  return (ggml_cuda_fattn_tile_get_config(pDkq, pDv, pNcols, pCc) >> 0) & ((1 << 10) - 1);
}

static constexpr __device__ int ggml_cuda_fattn_tile_get_nthreads(const int pDkq, const int pDv, const int pNcols) {
  return (ggml_cuda_fattn_tile_get_config(pDkq, pDv, pNcols) >> 0) & ((1 << 10) - 1);
}

static __host__ int ggml_cuda_fattn_tile_get_occupancy(const int pDkq, const int pDv, const int pNcols, const int pCc) {
  return (ggml_cuda_fattn_tile_get_config(pDkq, pDv, pNcols, pCc) >> 10) & ((1 << 4) - 1);
}

static constexpr __device__ int ggml_cuda_fattn_tile_get_occupancy(const int pDkq, const int pDv, const int pNcols) {
  return (ggml_cuda_fattn_tile_get_config(pDkq, pDv, pNcols) >> 10) & ((1 << 4) - 1);
}

static __host__ int ggml_cuda_fattn_tile_get_nbatch_fa(const int pDkq, const int pDv, const int pNcols, const int pCc) {
  return (ggml_cuda_fattn_tile_get_config(pDkq, pDv, pNcols, pCc) >> 14) & ((1 << 9) - 1);
}

static constexpr __device__ int ggml_cuda_fattn_tile_get_nbatch_fa(const int pDkq, const int pDv, const int pNcols) {
  return (ggml_cuda_fattn_tile_get_config(pDkq, pDv, pNcols) >> 14) & ((1 << 9) - 1);
}

static __host__ int ggml_cuda_fattn_tile_get_nbatch_K(const int pDkq, const int pDv, const int pNcols, const int pCc) {
  return (ggml_cuda_fattn_tile_get_config(pDkq, pDv, pNcols, pCc) >> 23) & ((1 << 9) - 1);
}

static constexpr __device__ int ggml_cuda_fattn_tile_get_nbatch_K(const int pDkq, const int pDv, const int pNcols) {
  return (ggml_cuda_fattn_tile_get_config(pDkq, pDv, pNcols) >> 23) & ((1 << 9) - 1);
}

// TODO: deduplicate with mma-f16
template <int warp_size, int nwarps, int I, int J, int J_padding, bool oob_check>
static __device__ __forceinline__ void flash_attn_tile_load_tile(const half2 *const __restrict__ pKv,
                                                                 half2 *const __restrict__ pTileKv, const int pStrideKv,
                                                                 const int pISup) {
  constexpr int cCpyNb = ggml_cuda_get_max_cpy_bytes();
  constexpr int cCpyNe = cCpyNb / 4;

  auto vLoad = [&] __device__(const int pN) {
    const int cStrideJ = warp_size >> pN;

    if (cStrideJ == 0) {
      return;
    }

    const int cJ0Start = cStrideJ == warp_size ? 0 : ((J / 2) / cCpyNe) - ((J / 2) / cCpyNe) % (2 * cStrideJ);
    const int cJ0Stop = ((J / 2) / cCpyNe) - ((J / 2) / cCpyNe) % (1 * cStrideJ);
    const int cStrideI = warp_size / cStrideJ;

    if (cJ0Start == cJ0Stop) {
      return;
    }

#pragma unroll
    for (int vI0 = 0; vI0 < I; vI0 += nwarps * cStrideI) {
      const int cI = vI0 + threadIdx.y * cStrideI + (cStrideJ == warp_size ? 0 : threadIdx.x / cStrideJ);

      if (vI0 + nwarps * cStrideI <= I || cI < I) {
#pragma unroll
        for (int vJ0 = cJ0Start; vJ0 < cJ0Stop; vJ0 += cStrideJ) {
          const int cJ = vJ0 * cCpyNe + (cStrideJ == warp_size ? threadIdx.x : threadIdx.x % cStrideJ) * cCpyNe;

          const __align__(16) half2 cZero[cCpyNe] = {{0.0f, 0.0f}};
          ggml_cuda_memcpy_1<cCpyNb>(pTileKv + cI * (J / 2 + J_padding) + cJ,
                                     !oob_check || cI < pISup ? pKv + cI * pStrideKv + cJ : cZero);
        }
      }
    }
  };
  // 1: max 64*16=512 bytes, 512 half
  // 2: max 32*16=512 bytes, 256 half
  // 3: max 16*16=256 bytes, 128 half
  // 4: max  8*16=128 bytes,  64 half
  // 5: max  4*16= 64 bytes,  32 half
  // 6: max  2*16= 32 bytes,  16 half
  // 7: max  1*16= 16 bytes,   8 half
  static_assert(J % 8 == 0, "bad J");
  static_assert((J / 2) % cCpyNe == 0, "bad J");
  ggml_cuda_unroll<7>{}(vLoad);
}

template <int warp_size, int nwarps, int I, int J, int J_padding, bool oob_check>
static __device__ __forceinline__ void flash_attn_tile_load_tile(const half2 *const __restrict__ pKv,
                                                                 float *const __restrict__ pTileKv, const int pStrideKv,
                                                                 const int pISup) {
  constexpr int cCpyNb = ggml_cuda_get_max_cpy_bytes();
  constexpr int cCpyNe = cCpyNb / 4;

  auto vLoad = [&] __device__(const int pN) {
    const int cStrideJ = warp_size >> pN;

    if (cStrideJ == 0) {
      return;
    }

    const int cJ0Start = cStrideJ == warp_size ? 0 : (J / cCpyNe) - (J / cCpyNe) % (2 * cStrideJ);
    const int cJ0Stop = (J / cCpyNe) - (J / cCpyNe) % (1 * cStrideJ);
    const int cStrideI = warp_size / cStrideJ;

    if (cJ0Start == cJ0Stop) {
      return;
    }

#pragma unroll
    for (int vI0 = 0; vI0 < I; vI0 += nwarps * cStrideI) {
      const int cI = vI0 + threadIdx.y * cStrideI + (cStrideJ == warp_size ? 0 : threadIdx.x / cStrideJ);

      if (vI0 + nwarps * cStrideI <= I || cI < I) {
#pragma unroll
        for (int vJ0 = cJ0Start; vJ0 < cJ0Stop; vJ0 += cStrideJ) {
          const int cJ =
              vJ0 * (cCpyNe / 2) + (cStrideJ == warp_size ? threadIdx.x : threadIdx.x % cStrideJ) * (cCpyNe / 2);

          const half2 cZero[cCpyNe / 2] = {{0.0f, 0.0f}};
          __align__(16) half2 vTmpH2[cCpyNe / 2];
          ggml_cuda_memcpy_1<sizeof(vTmpH2)>(vTmpH2, !oob_check || cI < pISup ? pKv + cI * pStrideKv + cJ : cZero);

          __align__(16) float2 vTmpF2[cCpyNe / 2];
#pragma unroll
          for (int vL = 0; vL < cCpyNe / 2; ++vL) {
            vTmpF2[vL] = __half22float2(vTmpH2[vL]);
          }
          ggml_cuda_memcpy_1<sizeof(vTmpF2)>(pTileKv + cI * (J + J_padding) + 2 * cJ, vTmpF2);
        }
      }
    }
  };
  // 1: max 32*16=512 bytes, 128 float
  // 2: max 16*16=256 bytes,  64 float
  // 3: max  8*16=128 bytes,  32 float
  // 4: max  4*16= 64 bytes,  16 float
  // 5: max  2*16= 32 bytes,   8 float
  static_assert(J % 8 == 0, "bad J");
  static_assert(J % cCpyNe == 0, "bad J");
  ggml_cuda_unroll<5>{}(vLoad);
}

// Function that performs a single iteration in for the KQ matrix multiplication:
template <int warp_size, int nwarps, int ncols1, int ncols2, int DKQ, int nbatch_fa, int nbatch_K,
          bool use_logit_softcap, bool oob_check, typename T_vec_dot>
static __device__ __forceinline__ void
flash_attn_tile_iter_KQ(T_vec_dot *const pQTmp, const half2 *const __restrict__ pKH2, T_vec_dot *const pKvTmp,
                        const int pStrideK2, const int pKVkq0, const int pKVkqSup, const int pKKq0, float *pKqAcc) {
  constexpr int cCpyNb = ggml_cuda_get_max_cpy_bytes();
  constexpr int cCpyNe = cCpyNb / 4;

  constexpr int cNcols = ncols1 * ncols2;
  constexpr int cCpw = cNcols > nwarps ? cNcols / nwarps : 1; // Q columns per warp
  constexpr int cNp = nwarps > cNcols ? nwarps / cNcols : 1;  // number of parallel warps per Q column

  flash_attn_tile_load_tile<warp_size, nwarps, nbatch_fa, nbatch_K, cCpyNe, oob_check>(
      pKH2 + int64_t(pKVkq0) * pStrideK2 + pKKq0 / 2, pKvTmp, pStrideK2, pKVkqSup);
  __syncthreads();

#ifdef FAST_FP16_AVAILABLE
  static_assert((nbatch_K / 2) % cCpyNe == 0, "bad nbatch_K");
#pragma unroll
  for (int vKKq1 = 0; vKKq1 < nbatch_K / 2; vKKq1 += cCpyNe) {
    __align__(16) half2 vKK[nbatch_fa / (cNp * warp_size)][cCpyNe];
    __align__(16) half2 vQK[cCpw][cCpyNe];
#else
  static_assert(nbatch_K % cCpyNe == 0, "bad nbatch_K");
#pragma unroll
  for (int vKKq1 = 0; vKKq1 < nbatch_K; vKKq1 += cCpyNe) {
    __align__(16) float vKK[nbatch_fa / (cNp * warp_size)][cCpyNe];
    __align__(16) float vQK[cCpw][cCpyNe];
#endif // FAST_FP16_AVAILABLE

#pragma unroll
    for (int vIKq0 = 0; vIKq0 < nbatch_fa; vIKq0 += cNp * warp_size) {
      const int cIKq = vIKq0 + (threadIdx.y % cNp) * warp_size + threadIdx.x;

#ifdef FAST_FP16_AVAILABLE
      ggml_cuda_memcpy_1<cCpyNb>(&vKK[vIKq0 / (cNp * warp_size)], &pKvTmp[cIKq * (nbatch_K / 2 + cCpyNe) + vKKq1]);
#else
      ggml_cuda_memcpy_1<cCpyNb>(&vKK[vIKq0 / (cNp * warp_size)], &pKvTmp[cIKq * (nbatch_K + cCpyNe) + vKKq1]);
#endif // FAST_FP16_AVAILABLE
    }
#pragma unroll
    for (int vJc0 = 0; vJc0 < cCpw; ++vJc0) {
      const int cJc = vJc0 + (threadIdx.y / cNp) * cCpw;

#ifdef FAST_FP16_AVAILABLE
      ggml_cuda_memcpy_1<cCpyNb>(&vQK[vJc0], &pQTmp[cJc * (DKQ / 2) + pKKq0 / 2 + vKKq1]);
#else
      ggml_cuda_memcpy_1<cCpyNb>(&vQK[vJc0], &pQTmp[cJc * DKQ + pKKq0 + vKKq1]);
#endif // FAST_FP16_AVAILABLE
    }

#pragma unroll
    for (int vIKq0 = 0; vIKq0 < nbatch_fa; vIKq0 += cNp * warp_size) {
#pragma unroll
      for (int vJc0 = 0; vJc0 < cCpw; ++vJc0) {
#pragma unroll
        for (int vK = 0; vK < cCpyNe; ++vK) {
          ggml_cuda_mad(pKqAcc[vIKq0 / (cNp * warp_size) * cCpw + vJc0], vKK[vIKq0 / (cNp * warp_size)][vK],
                        vQK[vJc0][vK]);
        }
      }
    }
  }

  if (pKKq0 + nbatch_K < DKQ) {
    __syncthreads(); // Sync not needed on last iteration.
  }
}

// Function that performs a single iteration of the main loop over up to nbatch_fa tokens.
template <int warp_size, int nwarps, int ncols1, int ncols2, int DKQ, int DV, int nbatch_fa, int nbatch_K,
          bool use_logit_softcap, bool oob_check, typename T_vec_dot, typename T_KQ, typename T_acc>
static __device__ __forceinline__ void
flash_attn_tile_iter(T_vec_dot *const pQTmp, const half2 *const __restrict__ pKH2, const half2 *const __restrict__ pVH2,
                     const half *const __restrict__ pMask, const uint3 pNe01, const float pLogitSoftcap,
                     const float pSlope, T_KQ *const pKq, T_vec_dot *const pKvTmp, const int pStrideK2,
                     const int pStrideV2, const int pStrideMask, float *const pKqMax, float *const pKqSum,
                     T_acc *const pVkq, const int pKVkq0, const int pKVkqMax, const int pColQ0) {
  constexpr int cCpyNb = ggml_cuda_get_max_cpy_bytes();
  constexpr int cCpyNe = cCpyNb / 4;

  constexpr int cNcols = ncols1 * ncols2;
  constexpr int cCpw = cNcols > nwarps ? cNcols / nwarps : 1; // Q columns per warp
  constexpr int cNp = nwarps > cNcols ? nwarps / cNcols : 1;  // number of parallel warps per Q column

  constexpr int cDVp = (DV + 2 * warp_size - 1) & ~(2 * warp_size - 1); // DV padded to multiple of 2*warp_size.

  // KQ_cs == KQ chunk size, number of KQ values in j direction to store as one contiguous chunk in memory.
  // KQ is originally 2D but uses a Z-shaped 3D memory pattern like KQ[ncols/KQ_cs][DVp][KQ_cs].
#ifdef FAST_FP16_AVAILABLE
  constexpr int cKqCs = cCpw < 2 * cCpyNe ? cCpw : 2 * cCpyNe;
#else
  constexpr int cKqCs = cCpw < 1 * cCpyNe ? cCpw : 1 * cCpyNe;
#endif // FAST_FP16_AVAILABLE
  static_assert(cCpw % cKqCs == 0, "bad KQ_cs");
  const int cKVkqSup = pKVkqMax - pKVkq0; // k supremum, only smaller k values have valid KV data

  float vKqMaxNew[cCpw];
#pragma unroll
  for (int vJc0 = 0; vJc0 < cCpw; ++vJc0) {
    vKqMaxNew[vJc0] = pKqMax[vJc0];
  }

  float vKqAcc[nbatch_fa / (cNp * warp_size) * cCpw] = {0.0f}; // Accumulators for KQ matrix multiplication.

  // KQ = K @ Q matrix multiplication:
  constexpr int cNbatchKLast = DKQ % nbatch_K;
#pragma unroll
  for (int vKKq0 = 0; vKKq0 < DKQ - cNbatchKLast; vKKq0 += nbatch_K) {
    flash_attn_tile_iter_KQ<warp_size, nwarps, ncols1, ncols2, DKQ, nbatch_fa, nbatch_K, use_logit_softcap, oob_check>(
        pQTmp, pKH2, pKvTmp, pStrideK2, pKVkq0, cKVkqSup, vKKq0, vKqAcc);
  }
  if (cNbatchKLast > 0) {
    constexpr int cKKq0 = DKQ - cNbatchKLast;
    flash_attn_tile_iter_KQ<warp_size, nwarps, ncols1, ncols2, DKQ, nbatch_fa, cNbatchKLast, use_logit_softcap,
                            oob_check>(pQTmp, pKH2, pKvTmp, pStrideK2, pKVkq0, cKVkqSup, cKKq0, vKqAcc);
  }

  // Apply logit softcap + mask, update KQ_max:
#pragma unroll
  for (int vJc0 = 0; vJc0 < cCpw; ++vJc0) {
    const int cJ = fastmodulo(pColQ0 + (vJc0 + (threadIdx.y / cNp) * cCpw) / ncols2, pNe01);

#pragma unroll
    for (int vIKq0 = 0; vIKq0 < nbatch_fa; vIKq0 += cNp * warp_size) {
      const int cIKq = vIKq0 + (threadIdx.y % cNp) * warp_size + threadIdx.x;

#if defined(FAST_FP16_AVAILABLE) && !defined(V_DOT2_F32_F16_AVAILABLE)
      // Without the v_dot2_f32_f16 instruction there is a higher risk of numerical overflow in the KQ calculation.
      // Therefore, scale down Q values and apply the inverse scale the FP32 KQ values afterwards again.
      vKqAcc[vIKq0 / (cNp * warp_size) * cCpw + vJc0] *= 4.0f;
#endif // defined(FAST_FP16_AVAILABLE) && !defined(V_DOT2_F32_F16_AVAILABLE)

      if (use_logit_softcap) {
        vKqAcc[(vIKq0 / (cNp * warp_size)) * cCpw + vJc0] =
            pLogitSoftcap * tanhf(vKqAcc[(vIKq0 / (cNp * warp_size)) * cCpw + vJc0]);
      }

      if (!oob_check || cIKq < cKVkqSup) {
        vKqAcc[(vIKq0 / (cNp * warp_size)) * cCpw + vJc0] +=
            (ncols2 > 1 || pMask) ? pSlope * __half2float(pMask[cJ * pStrideMask + pKVkq0 + cIKq]) : 0.0f;

        vKqMaxNew[vJc0] =
            fmaxf(vKqMaxNew[vJc0], vKqAcc[(vIKq0 / (cNp * warp_size)) * cCpw + vJc0] + FATTN_KQ_MAX_OFFSET);
      }
    }

    vKqMaxNew[vJc0] = warp_reduce_max<warp_size>(vKqMaxNew[vJc0]);
  }

  if constexpr (cNp == 1) {
    __syncthreads();
  } else {
    static_assert(cCpw == 1, "bad cpw");
    __shared__ float vKqMaxNewShared[nwarps];
    if (threadIdx.x == 0) {
      vKqMaxNewShared[threadIdx.y] = vKqMaxNew[0];
    }
    __syncthreads();
    vKqMaxNew[0] = vKqMaxNewShared[(threadIdx.y & ~(cNp - 1)) + threadIdx.x % cNp];
    vKqMaxNew[0] = warp_reduce_max<cNp>(vKqMaxNew[0]);
  }

  // Calculate KQ softmax, write to shared KQ buffer, re-scale VKQ accumulators:
#pragma unroll
  for (int vJc0 = 0; vJc0 < cCpw; vJc0 += cKqCs) {
#ifdef FAST_FP16_AVAILABLE
    __align__(16) half vTmp[nbatch_fa / (cNp * warp_size)][cKqCs];
#else
    __align__(16) float vTmp[nbatch_fa / (cNp * warp_size)][cKqCs];
#endif // FAST_FP16_AVAILABLE

#pragma unroll
    for (int vJc1 = 0; vJc1 < cKqCs; ++vJc1) {
      const int cJc = vJc0 + vJc1;

      const float cKqMaxScale = expf(pKqMax[cJc] - vKqMaxNew[cJc]);
      pKqMax[cJc] = vKqMaxNew[cJc];

      float vKqSumAdd = 0.0f;
#pragma unroll
      for (int vI0 = 0; vI0 < nbatch_fa; vI0 += cNp * warp_size) {
        const float cVal =
            !oob_check || vI0 + (threadIdx.y % cNp) * warp_size + threadIdx.x < static_cast<uint32_t>(cKVkqSup)
                ? expf(vKqAcc[(vI0 / (cNp * warp_size)) * cCpw + cJc] - pKqMax[cJc])
                : 0.0f;
        vKqSumAdd += cVal;
        vTmp[vI0 / (cNp * warp_size)][vJc1] = cVal;
      }
      pKqSum[cJc] = pKqSum[cJc] * cKqMaxScale + vKqSumAdd;

#ifdef FAST_FP16_AVAILABLE
      const half2 cKqMaxScaleH2 = make_half2(cKqMaxScale, cKqMaxScale);
#pragma unroll
      for (int vI0 = 0; vI0 < cDVp / 2; vI0 += warp_size) {
        pVkq[cJc * ((cDVp / 2) / warp_size) + vI0 / warp_size] *= cKqMaxScaleH2;
      }
#else
#pragma unroll
      for (int vI0 = 0; vI0 < cDVp / 2; vI0 += warp_size) {
        pVkq[cJc * ((cDVp / 2) / warp_size) + vI0 / warp_size].x *= cKqMaxScale;
        pVkq[cJc * ((cDVp / 2) / warp_size) + vI0 / warp_size].y *= cKqMaxScale;
      }
#endif // FAST_FP16_AVAILABLE
    }

#pragma unroll
    for (int vI0 = 0; vI0 < nbatch_fa; vI0 += cNp * warp_size) {
      const int cI = vI0 + (threadIdx.y % cNp) * warp_size + threadIdx.x;

      ggml_cuda_memcpy_1<sizeof(vTmp[0])>(
          pKq + (vJc0 / cKqCs + (threadIdx.y / cNp) * (cCpw / cKqCs)) * (nbatch_fa * cKqCs) + cI * cKqCs,
          vTmp[vI0 / (cNp * warp_size)]);
    }
  }

  // VKQ = V @ KQ matrix multiplication:
  static_assert(DV <= DKQ, "bad DV");
  static_assert(DV % nbatch_K == 0 || (nbatch_K % 3 == 0 && DV % (nbatch_K * 2 / 3) == 0), "bad nbatch_K");
  constexpr int cNbatchV = (DV % nbatch_K == 0 ? nbatch_K : nbatch_K * 2 / 3) * nbatch_fa /
                           DV; // Number of V columns that fit in SRAM for K.
  static_assert(nbatch_fa % cNbatchV == 0, "bad nbatch_V");
  static_assert(cNbatchV % cNp == 0, "bad nbatch_V");
#pragma unroll
  for (int vK0 = 0; vK0 < nbatch_fa; vK0 += cNbatchV) {
    flash_attn_tile_load_tile<warp_size, nwarps, cNbatchV, DV, 0, oob_check>(pVH2 + int64_t(pKVkq0 + vK0) * pStrideV2,
                                                                             pKvTmp, pStrideV2, cKVkqSup - vK0);
    __syncthreads();

#ifdef FAST_FP16_AVAILABLE
#pragma unroll
    for (int vK1 = 0; vK1 < cNbatchV; vK1 += cNp) {
      __align__(16) half2 vVK[(cDVp / 2) / warp_size];
      __align__(16) half2 vKqK[cCpw];

      constexpr int cCpyNeD = cCpyNe / 2 < (cDVp / 2) / warp_size ? cCpyNe / 2 : (cDVp / 2) / warp_size;
#pragma unroll
      for (int vI0 = 0; vI0 < cDVp / 2; vI0 += warp_size * cCpyNeD) {
        ggml_cuda_memcpy_1<cCpyNeD * 4>(&vVK[vI0 / warp_size],
                                        &pKvTmp[(vK1 + threadIdx.y % cNp) * (DV / 2) + vI0 + threadIdx.x * cCpyNeD]);
      }
#pragma unroll
      for (int vJcVkq0 = 0; vJcVkq0 < cCpw; vJcVkq0 += cKqCs) {
        const int cJcKq = vJcVkq0 / cKqCs + (threadIdx.y / cNp) * (cCpw / cKqCs);

        __align__(16) half vTmp[cKqCs];
        ggml_cuda_memcpy_1<cKqCs * sizeof(half)>(&vTmp, pKq + cJcKq * (nbatch_fa * cKqCs) +
                                                            (vK0 + vK1 + threadIdx.y % cNp) * cKqCs);
#pragma unroll
        for (int vJcVkq1 = 0; vJcVkq1 < cKqCs; ++vJcVkq1) {
          vKqK[vJcVkq0 + vJcVkq1] = __half2half2(vTmp[vJcVkq1]);
        }
      }

#pragma unroll
      for (int vI0 = 0; vI0 < cDVp / 2; vI0 += warp_size) {
#pragma unroll
        for (int vJcVkq0 = 0; vJcVkq0 < cCpw; ++vJcVkq0) {
          pVkq[vJcVkq0 * ((cDVp / 2) / warp_size) + vI0 / warp_size] += vVK[vI0 / warp_size] * vKqK[vJcVkq0];
        }
      }
    }
#else
#pragma unroll
    for (int vK1 = 0; vK1 < cNbatchV; vK1 += cNp) {
      __align__(16) float2 vVK[(cDVp / 2) / warp_size];
      __align__(16) float vKqK[cCpw];

      constexpr int cCpyNeD = cCpyNe < cDVp / warp_size ? cCpyNe : cDVp / warp_size;
#pragma unroll
      for (int vI0 = 0; vI0 < cDVp; vI0 += warp_size * cCpyNeD) {
        ggml_cuda_memcpy_1<cCpyNeD * 4>(&vVK[vI0 / (2 * warp_size)],
                                        &pKvTmp[(vK1 + threadIdx.y % cNp) * DV + vI0 + threadIdx.x * cCpyNeD]);
      }
#pragma unroll
      for (int vJcVkq0 = 0; vJcVkq0 < cCpw; vJcVkq0 += cKqCs) {
        const int cJcKq = vJcVkq0 / cKqCs + (threadIdx.y / cNp) * (cCpw / cKqCs);

        ggml_cuda_memcpy_1<cKqCs * sizeof(float)>(&vKqK[vJcVkq0], pKq + cJcKq * (nbatch_fa * cKqCs) +
                                                                      (vK0 + vK1 + threadIdx.y % cNp) * cKqCs);
      }

#pragma unroll
      for (int vI0 = 0; vI0 < cDVp / 2; vI0 += warp_size) {
#pragma unroll
        for (int vJcVkq0 = 0; vJcVkq0 < cCpw; ++vJcVkq0) {
          pVkq[vJcVkq0 * ((cDVp / 2) / warp_size) + vI0 / warp_size].x += vVK[vI0 / warp_size].x * vKqK[vJcVkq0];
          pVkq[vJcVkq0 * ((cDVp / 2) / warp_size) + vI0 / warp_size].y += vVK[vI0 / warp_size].y * vKqK[vJcVkq0];
        }
      }
    }
#endif // FAST_FP16_AVAILABLE

    __syncthreads();
  }
}

template <int DKQ, int DV, int ncols1, int ncols2, bool use_logit_softcap> // D == head size
__launch_bounds__(ggml_cuda_fattn_tile_get_nthreads(DKQ, DV, ncols1 *ncols2),
                  ggml_cuda_fattn_tile_get_occupancy(DKQ, DV, ncols1 *ncols2)) static __global__
    void flash_attn_tile(const char *pQPtr, const char *pKPtr, const char *pVPtr, const char *pMaskPtr,
                         const char *pSinksPtr, const int *pKvMaxPtr, float *pDstPtr, float2 *pDstMetaPtr,
                         const float scale, const float pMaxBias, const float pM0, const float pM1,
                         const uint32_t pNHeadLog2, const float pLogitSoftcap, const int32_t pNe00, const uint3 pNe01,
                         const int32_t pNe02, const int32_t pNe03, const int32_t pNb01, const int32_t pNb02,
                         const int32_t pNb03, const int32_t pNe10, const int32_t pNe11, const int32_t pNe12,
                         const int32_t pNe13, const int32_t pNb11, const int32_t pNb12, const int64_t pNb13,
                         const int32_t pNb21, const int32_t pNb22, const int64_t pNb23, const int32_t pNe31,
                         const int32_t pNe32, const int32_t pNe33, const int32_t pNb31, const int32_t pNb32,
                         const int64_t pNb33) {
#ifdef FLASH_ATTN_AVAILABLE
  const char *GGML_CUDA_RESTRICT vQ = pQPtr;
  const char *GGML_CUDA_RESTRICT vK = pKPtr;
  const char *GGML_CUDA_RESTRICT vV = pVPtr;
  const char *GGML_CUDA_RESTRICT vMask = pMaskPtr;
  const char *GGML_CUDA_RESTRICT vSinks = pSinksPtr;
  const int *GGML_CUDA_RESTRICT vKvMax = pKvMaxPtr;
  float *GGML_CUDA_RESTRICT vDst = pDstPtr;
  float2 *GGML_CUDA_RESTRICT vDstMeta = pDstMetaPtr;

  // Skip unused kernel variants for faster compilation:

  if ((use_logit_softcap && !(DV == 128 || DV == 256 || DV == 512))) {
    GGML_UNUSED_VARS(vQ, vK, vV, vMask, vSinks, vKvMax, vDst, vDstMeta, scale, pMaxBias, pM0, pM1, pNHeadLog2,
                     pLogitSoftcap, pNe00, pNe01, pNe02, pNe03, pNb01, pNb02, pNb03, pNe10, pNe11, pNe12, pNe13, pNb11,
                     pNb12, pNb13, pNb21, pNb22, pNb23, pNe31, pNe32, pNe33, pNb31, pNb32, pNb33);
    NO_DEVICE_CODE;
    return;
  }

  static_assert(ggml_cuda_fattn_tile_get_config(DKQ, DV, ncols1 * ncols2) != 0, "kernel config not defined");

  constexpr int cNcols = ncols1 * ncols2;
  constexpr int cWarpSize = 32;
  constexpr int cNwarps = ggml_cuda_fattn_tile_get_nthreads(DKQ, DV, ncols1 * ncols2) / cWarpSize;
  constexpr int cNbatchFa = ggml_cuda_fattn_tile_get_nbatch_fa(DKQ, DV, ncols1 * ncols2);
  constexpr int cNbatchK = ggml_cuda_fattn_tile_get_nbatch_K(DKQ, DV, ncols1 * ncols2);

  // In this kernel Q, K, V are matrices while i, j, k are matrix indices.

  const int cColQ0 = blockIdx.x * ncols1; // Index of the first Q column for this CUDA block to work on.

  const int cSequence = blockIdx.z / (pNe02 / ncols2);
  const int cHead0 = blockIdx.z * ncols2 - cSequence * pNe02; // == blockIdx.z % (ne02/ncols2)
  const int cGqaRatio = pNe02 / pNe12; // With grouped query attention there are > 1 Q matrices per K, V matrix.
  const float *vQF = (const float *)(vQ + pNb03 * cSequence + pNb02 * cHead0);
  const half2 *vKH2 = (const half2 *)(vK + pNb13 * cSequence + pNb12 * (cHead0 / cGqaRatio));
  const half2 *vVH2 = (const half2 *)(vV + pNb23 * cSequence + pNb22 * (cHead0 / cGqaRatio)); // K and V have same shape

  const half *vMaskh = vMask ? (const half *)(vMask + pNb33 * (cSequence % pNe33)) : nullptr;

  const int cStrideK2 = pNb11 / sizeof(half2);
  const int cStrideV2 = pNb21 / sizeof(half2);
  const int cStrideMask = pNb31 / sizeof(half);

  const float cSlope = ncols2 == 1 ? get_alibi_slope(pMaxBias, cHead0, pNHeadLog2, pM0, pM1) : 1.0f;

  constexpr int cCpyNb = ggml_cuda_get_max_cpy_bytes();
  constexpr int cCpyNe = cCpyNb / 4;

  constexpr int cCpw = cNcols > cNwarps ? cNcols / cNwarps : 1; // Q columns per warp.
  constexpr int cNp = cNwarps > cNcols ? cNwarps / cNcols : 1;  // Number of parallel warps per Q column.
  static_assert(cCpw == 1 || cNp == 1, "bad cpw / np");
  static_assert(cNbatchFa % (cNp * cWarpSize) == 0, "nbatch_fa % (np*warp_size) != 0");

  constexpr int cDkQp = (DKQ + 2 * cWarpSize - 1) & ~(2 * cWarpSize - 1); // DKQ padded to multiple of 2*warp_size.
  constexpr int cDVp = (DV + 2 * cWarpSize - 1) & ~(2 * cWarpSize - 1);   // DV  padded to multiple of 2*warp_size.

  // Q_tmp == SRAM buffer to hold Q data for the entire lifetime of the kernel.
  // KV_tmp == SRAM buffer to hold fragments of K/V data while iterating over ne11.
  //     KV_tmp is padded to avoid memory conflicts for K (cpy_ne) and OOB accesses for V (DVp-DV).
  // KQ == SRAM buffer to hold KQ fragments between KQ and VKQ matrix multiplications.
  // VKQ == Accumulators in registers for the final VKQ result.
#ifdef FAST_FP16_AVAILABLE
  __shared__ half2 vQTmp[cNcols * DKQ / 2];
  __shared__ half2 vKvTmp[cNbatchFa * (cNbatchK / 2 + cCpyNe) + cDVp - DV];
  __shared__ half vKq[cNcols * cNbatchFa];
  __align__(16) half2 vVkq[cCpw * ((cDVp / 2) / cWarpSize)] = {{0.0f, 0.0f}};
#else
  __shared__ float vQTmp[cNcols * DKQ];
  __shared__ float vKvTmp[cNbatchFa * (cNbatchK + cCpyNe) + cDVp - DV];
  __shared__ float vKq[cNcols * cNbatchFa];
  __align__(16) float2 vVkq[cCpw * ((cDVp / 2) / cWarpSize)] = {{0.0f, 0.0f}};
#endif // FAST_FP16_AVAILABLE

  float vKqMax[cCpw];
#pragma unroll
  for (int vJ0 = 0; vJ0 < cNcols; vJ0 += cNwarps) {
    vKqMax[vJ0 / cNwarps] = -FLT_MAX / 2.0f;
  }
  float vKqSum[cCpw] = {0.0f};

  ggml_cuda_pdl_sync();

  // Load Q data, convert to FP16 if fast:
#pragma unroll
  for (int vJc0 = 0; vJc0 < cCpw; ++vJc0) {
    const int cJc = vJc0 + (threadIdx.y / cNp) * cCpw;

    const int cJ = cJc / ncols2;
    const int c_ = cJc % ncols2;

    constexpr int cCpyNeD = cCpyNe < cDkQp / cWarpSize ? cCpyNe : cDkQp / cWarpSize;

#pragma unroll
    for (int vI0 = 0; vI0 < cDkQp; vI0 += cNp * cWarpSize * cCpyNeD) {
      if (vI0 + cNp * cWarpSize * cCpyNeD <= DKQ ||
          vI0 + (threadIdx.y % cNp) * (cWarpSize * cCpyNeD) + threadIdx.x * cCpyNeD < DKQ) {
        __align__(16) float vTmpF[cCpyNeD] = {0.0f};
        ggml_cuda_memcpy_1<sizeof(vTmpF)>(
            vTmpF, &vQF[c_ * (pNb02 / sizeof(float)) + fastmodulo(cColQ0 + cJ, pNe01) * (pNb01 / sizeof(float)) + vI0 +
                        (threadIdx.y % cNp) * (cWarpSize * cCpyNeD) + threadIdx.x * cCpyNeD]);

#pragma unroll
        for (int vI1 = 0; vI1 < cCpyNeD; ++vI1) {
          vTmpF[vI1] *= scale;
        }

#ifdef FAST_FP16_AVAILABLE
        __align__(16) half2 vTmpH2[cCpyNeD / 2];
#pragma unroll
        for (int vI1 = 0; vI1 < cCpyNeD; vI1 += 2) {
          vTmpH2[vI1 / 2] = make_half2(vTmpF[vI1 + 0], vTmpF[vI1 + 1]);
#if defined(FAST_FP16_AVAILABLE) && !defined(V_DOT2_F32_F16_AVAILABLE)
          // Without the v_dot2_f32_f16 instruction there is a higher risk of numerical overflow in the KQ calculation.
          // Therefore, scale down Q values and apply the inverse scale the FP32 KQ values afterwards again.
          vTmpH2[vI1 / 2] *= make_half2(0.25f, 0.25f);
#endif // defined(FAST_FP16_AVAILABLE) && !defined(V_DOT2_F32_F16_AVAILABLE)
        }
        ggml_cuda_memcpy_1<sizeof(vTmpH2)>(
            &vQTmp[cJc * (DKQ / 2) + vI0 / 2 + (threadIdx.y % cNp) * (cWarpSize * cCpyNeD / 2) +
                   threadIdx.x * (cCpyNeD / 2)],
            vTmpH2);
#else
        ggml_cuda_memcpy_1<sizeof(vTmpF)>(
            &vQTmp[cJc * DKQ + vI0 + (threadIdx.y % cNp) * (cWarpSize * cCpyNeD) + threadIdx.x * cCpyNeD], vTmpF);
#endif // FAST_FP16_AVAILABLE
      }
    }
  }

  __syncthreads();

  // Main loop over KV cache:
  const int cKVkqMax = vKvMax ? vKvMax[cSequence * gridDim.x + blockIdx.x] : pNe11;
  if (ncols2 == 1) {
    // Branch with out-of-bounds checks.
    int vKVkq0 = blockIdx.y * cNbatchFa;
    while (vKVkq0 < cKVkqMax - cNbatchFa) {
      constexpr bool cOobCheck = false;
      flash_attn_tile_iter<cWarpSize, cNwarps, ncols1, ncols2, DKQ, DV, cNbatchFa, cNbatchK, use_logit_softcap,
                           cOobCheck>(vQTmp, vKH2, vVH2, vMaskh, pNe01, pLogitSoftcap, cSlope, vKq, vKvTmp, cStrideK2,
                                      cStrideV2, cStrideMask, vKqMax, vKqSum, vVkq, vKVkq0, cKVkqMax, cColQ0);
      vKVkq0 += gridDim.y * cNbatchFa;
    }
    if (vKVkq0 < cKVkqMax) {
      constexpr bool cOobCheck = true;
      flash_attn_tile_iter<cWarpSize, cNwarps, ncols1, ncols2, DKQ, DV, cNbatchFa, cNbatchK, use_logit_softcap,
                           cOobCheck>(vQTmp, vKH2, vVH2, vMaskh, pNe01, pLogitSoftcap, cSlope, vKq, vKvTmp, cStrideK2,
                                      cStrideV2, cStrideMask, vKqMax, vKqSum, vVkq, vKVkq0, cKVkqMax, cColQ0);
    }
  } else {
    // Branch without out-of-bounds checks.
    for (int vKVkq0 = blockIdx.y * cNbatchFa; vKVkq0 < cKVkqMax; vKVkq0 += gridDim.y * cNbatchFa) {
      constexpr bool cOobCheck = false;
      flash_attn_tile_iter<cWarpSize, cNwarps, ncols1, ncols2, DKQ, DV, cNbatchFa, cNbatchK, use_logit_softcap,
                           cOobCheck>(vQTmp, vKH2, vVH2, vMaskh, pNe01, pLogitSoftcap, cSlope, vKq, vKvTmp, cStrideK2,
                                      cStrideV2, cStrideMask, vKqMax, vKqSum, vVkq, vKVkq0, cKVkqMax, cColQ0);
    }
  }

#pragma unroll
  for (int vJc0 = 0; vJc0 < cCpw; ++vJc0) {
    vKqSum[vJc0] = warp_reduce_sum<cWarpSize>(vKqSum[vJc0]);
  }

  if constexpr (cNp > 1) {
    static_assert(cCpw == 1, "bad cpw");
    static_assert(cNbatchFa * cNbatchK >= cNwarps * cDVp, "KV_tmp too small");

#ifdef FAST_FP16_AVAILABLE
    half2 *vVkqCombine = (half2 *)vKvTmp;
#else
    float *vVkqCombine = (float *)vKvTmp;
#endif // FAST_FP16_AVAILABLE
    float *vKqSumCombine = (float *)vQTmp;

    if (threadIdx.y % cNp != 0) {
#ifdef FAST_FP16_AVAILABLE
      constexpr int cCpyNeD = cCpyNe < (cDVp / 2) / cWarpSize ? cCpyNe : (cDVp / 2) / cWarpSize;
#pragma unroll
      for (int vI0 = 0; vI0 < cDVp / 2; vI0 += cWarpSize * cCpyNeD) {
        ggml_cuda_memcpy_1<cCpyNeD * 4>(&vVkqCombine[threadIdx.y * (cDVp / 2) + vI0 + threadIdx.x * cCpyNeD],
                                        &vVkq[vI0 / cWarpSize]);
      }
#else
      constexpr int cCpyNeD = cCpyNe < cDVp / cWarpSize ? cCpyNe : cDVp / cWarpSize;
#pragma unroll
      for (int vI0 = 0; vI0 < cDVp; vI0 += cWarpSize * cCpyNeD) {
        ggml_cuda_memcpy_1<cCpyNeD * 4>(&vVkqCombine[threadIdx.y * cDVp + vI0 + threadIdx.x * cCpyNeD],
                                        ((const float *)vVkq) + vI0 / cWarpSize);
      }
#endif // FAST_FP16_AVAILABLE

      if (threadIdx.x == 0) {
        vKqSumCombine[threadIdx.y] = vKqSum[0];
      }

      return;
    }

    __syncthreads();

#pragma unroll
    for (int vIp = 1; vIp < cNp; ++vIp) {
#ifdef FAST_FP16_AVAILABLE
      constexpr int cCpyNeD = cCpyNe < (cDVp / 2) / cWarpSize ? cCpyNe : (cDVp / 2) / cWarpSize;
#pragma unroll
      for (int vI0 = 0; vI0 < cDVp / 2; vI0 += cWarpSize * cCpyNeD) {
        __align__(16) half2 vTmp[cCpyNeD];
        ggml_cuda_memcpy_1<cCpyNeD * 4>(vTmp,
                                        &vVkqCombine[(threadIdx.y + vIp) * (cDVp / 2) + vI0 + threadIdx.x * cCpyNeD]);
#pragma unroll
        for (int vI1 = 0; vI1 < cCpyNeD; ++vI1) {
          vVkq[vI0 / cWarpSize + vI1] += vTmp[vI1];
        }
      }
#else
      constexpr int cCpyNeD = cCpyNe < cDVp / cWarpSize ? cCpyNe : cDVp / cWarpSize;
#pragma unroll
      for (int vI0 = 0; vI0 < cDVp; vI0 += cWarpSize * cCpyNeD) {
        __align__(16) float vTmp[cCpyNeD];
        ggml_cuda_memcpy_1<cCpyNeD * 4>(vTmp, &vVkqCombine[(threadIdx.y + vIp) * cDVp + vI0 + threadIdx.x * cCpyNeD]);
#pragma unroll
        for (int vI1 = 0; vI1 < cCpyNeD; ++vI1) {
          ((float *)vVkq)[vI0 / cWarpSize + vI1] += vTmp[vI1];
        }
      }
#endif // FAST_FP16_AVAILABLE

      vKqSum[0] += vKqSumCombine[threadIdx.y + vIp];
    }
  }

  // Attention sink: adjust KQ max and sum only for the first of all parallel blocks:
  if (vSinks && blockIdx.y == 0) {
#pragma unroll
    for (int vJc0 = 0; vJc0 < cCpw; ++vJc0) {
      const int cJc = vJc0 + (threadIdx.y / cNp) * cCpw;
      const float cSink = ((const float *)vSinks)[cHead0 + cJc % ncols2];

      float vKqMaxNewJ = fmaxf(vKqMax[vJc0], cSink);
      const float cKqMaxScale = expf(vKqMax[vJc0] - vKqMaxNewJ);
      vKqMax[vJc0] = vKqMaxNewJ;

      const float cVal = expf(cSink - vKqMax[vJc0]);
      vKqSum[vJc0] = vKqSum[vJc0] * cKqMaxScale + cVal;

#ifdef FAST_FP16_AVAILABLE
      const half2 cKqMaxScaleH2 = make_half2(cKqMaxScale, cKqMaxScale);
#pragma unroll
      for (int vI0 = 0; vI0 < cDVp / 2; vI0 += cWarpSize) {
        vVkq[vJc0 * ((cDVp / 2) / cWarpSize) + vI0 / cWarpSize] *= cKqMaxScaleH2;
      }
#else
#pragma unroll
      for (int vI0 = 0; vI0 < cDVp / 2; vI0 += cWarpSize) {
        vVkq[vJc0 * ((cDVp / 2) / cWarpSize) + vI0 / cWarpSize].x *= cKqMaxScale;
        vVkq[vJc0 * ((cDVp / 2) / cWarpSize) + vI0 / cWarpSize].y *= cKqMaxScale;
      }
#endif // FAST_FP16_AVAILABLE
    }
  }

  // Write back results:
#pragma unroll
  for (int vJc0 = 0; vJc0 < cCpw; ++vJc0) {
    const int cJc = vJc0 + (threadIdx.y / cNp) * cCpw;

    const int cJ = cJc / ncols2;
    const int c_ = cJc % ncols2;

    if (ncols1 > 1 && cColQ0 + cJ >= int(pNe01.z)) {
      return;
    }

    const float scale = gridDim.y == 1 ? 1.0f / vKqSum[vJc0] : 1.0f;

    const int cJDstUnrolled = ((cSequence * int(pNe01.z) + cColQ0 + cJ) * pNe02 + cHead0 + c_) * gridDim.y + blockIdx.y;

#ifdef FAST_FP16_AVAILABLE
    constexpr int cCpyNeD = cCpyNe / 2 < (cDVp / 2) / cWarpSize ? cCpyNe / 2 : (cDVp / 2) / cWarpSize;
#pragma unroll
    for (int vI0 = 0; vI0 < cDVp / 2; vI0 += cWarpSize * cCpyNeD) {
      __align__(16) float2 vTmp[cCpyNeD];
#pragma unroll
      for (int vI1 = 0; vI1 < cCpyNeD; ++vI1) {
        vTmp[vI1] = __half22float2(vVkq[vJc0 * ((cDVp / 2) / cWarpSize) + vI0 / cWarpSize + vI1]);
        vTmp[vI1].x *= scale;
        vTmp[vI1].y *= scale;
      }
      if (vI0 + cWarpSize * cCpyNeD <= DV / 2 || vI0 + threadIdx.x * cCpyNeD < DV / 2) {
        ggml_cuda_memcpy_1<sizeof(vTmp)>(&vDst[cJDstUnrolled * DV + 2 * vI0 + threadIdx.x * (2 * cCpyNeD)], vTmp);
      }
    }
#else
    constexpr int cCpyNeD = cCpyNe < cDVp / cWarpSize ? cCpyNe : cDVp / cWarpSize;
#pragma unroll
    for (int vI0 = 0; vI0 < cDVp; vI0 += cWarpSize * cCpyNeD) {
      if (vI0 + cWarpSize * cCpyNeD <= DV || vI0 + threadIdx.x * cCpyNeD < DV) {
#pragma unroll
        for (int vI1 = 0; vI1 < cCpyNeD / 2; ++vI1) {
          vVkq[vJc0 * ((cDVp / 2) / cWarpSize) + vI0 / (2 * cWarpSize) + vI1].x *= scale;
          vVkq[vJc0 * ((cDVp / 2) / cWarpSize) + vI0 / (2 * cWarpSize) + vI1].y *= scale;
        }
        ggml_cuda_memcpy_1<cCpyNeD * 4>(&vDst[cJDstUnrolled * DV + vI0 + threadIdx.x * cCpyNeD],
                                        &vVkq[vJc0 * ((cDVp / 2) / cWarpSize) + vI0 / (2 * cWarpSize)]);
      }
    }
#endif // FAST_FP16_AVAILABLE

    if (gridDim.y != 1 && threadIdx.x == 0) {
      vDstMeta[cJDstUnrolled] = make_float2(vKqMax[vJc0], vKqSum[vJc0]);
    }
  }
#else
  GGML_UNUSED_VARS(pQPtr, pKPtr, pVPtr, pMaskPtr, pSinksPtr, pKvMaxPtr, pDstPtr, pDstMetaPtr, scale, pMaxBias, pM0, pM1,
                   pNHeadLog2, pLogitSoftcap, pNe00, pNe01, pNe02, pNe03, pNb01, pNb02, pNb03, pNe10, pNe11, pNe12,
                   pNe13, pNb11, pNb12, pNb13, pNb21, pNb22, pNb23, pNe31, pNe32, pNe33, pNb31, pNb32, pNb33);
  NO_DEVICE_CODE;
#endif // FLASH_ATTN_AVAILABLE
}

template <int DKQ, int DV, int ncols2, bool use_logit_softcap>
static void launch_fattn_tile_switch_ncols1(ggml_backend_cuda_context &pCtx, ggml_tensor *pDst) {
  const ggml_tensor *vQ = pDst->src[0];

  const int cId = ggml_cuda_get_device();
  const int cCc = ggml_cuda_info().devices[cId].cc;
  const int cWarpSize = 32;

  constexpr size_t cNbytesShared = 0;

#ifdef GGML_USE_HIP
  if constexpr (DKQ <= 128) {
    if (vQ->ne[1] > 32 / ncols2) {
      constexpr int cColsPerBlock = 64;
      const int cNwarps = ggml_cuda_fattn_tile_get_nthreads(DKQ, DV, cColsPerBlock, cCc) / cWarpSize;
      const int cNbatchFa = ggml_cuda_fattn_tile_get_nbatch_fa(DKQ, DV, cColsPerBlock, cCc);
      fattn_kernel_t vFattnKernel = flash_attn_tile<DKQ, DV, cColsPerBlock / ncols2, ncols2, use_logit_softcap>;
      launch_fattn<DV, cColsPerBlock / ncols2, ncols2>(pCtx, pDst, vFattnKernel, cNwarps, cNbytesShared, cNbatchFa,
                                                       true, true, false, cWarpSize);
      return;
    }
  }
#endif // GGML_USE_HIP

#ifndef GGML_USE_HIP
  if constexpr (DKQ <= 256)
#endif // GGML_USE_HIP
  {
    if (vQ->ne[1] > 16 / ncols2) {
      constexpr int cColsPerBlock = 32;
      const int cNwarps = ggml_cuda_fattn_tile_get_nthreads(DKQ, DV, cColsPerBlock, cCc) / cWarpSize;
      const int cNbatchFa = ggml_cuda_fattn_tile_get_nbatch_fa(DKQ, DV, cColsPerBlock, cCc);
      fattn_kernel_t vFattnKernel = flash_attn_tile<DKQ, DV, cColsPerBlock / ncols2, ncols2, use_logit_softcap>;
      launch_fattn<DV, cColsPerBlock / ncols2, ncols2>(pCtx, pDst, vFattnKernel, cNwarps, cNbytesShared, cNbatchFa,
                                                       true, true, false, cWarpSize);
      return;
    }
  }

  if constexpr (ncols2 <= 16) {
    if (vQ->ne[1] > 8 / ncols2) {
      constexpr int cColsPerBlock = 16;
      const int cNwarps = ggml_cuda_fattn_tile_get_nthreads(DKQ, DV, cColsPerBlock, cCc) / cWarpSize;
      const int cNbatchFa = ggml_cuda_fattn_tile_get_nbatch_fa(DKQ, DV, cColsPerBlock, cCc);
      fattn_kernel_t vFattnKernel = flash_attn_tile<DKQ, DV, cColsPerBlock / ncols2, ncols2, use_logit_softcap>;
      launch_fattn<DV, cColsPerBlock / ncols2, ncols2>(pCtx, pDst, vFattnKernel, cNwarps, cNbytesShared, cNbatchFa,
                                                       true, true, false, cWarpSize);
      return;
    }
  }

  if constexpr (ncols2 <= 8) {
    if (vQ->ne[1] > 4 / ncols2) {
      constexpr int cColsPerBlock = 8;
      const int cNwarps = ggml_cuda_fattn_tile_get_nthreads(DKQ, DV, cColsPerBlock, cCc) / cWarpSize;
      const int cNbatchFa = ggml_cuda_fattn_tile_get_nbatch_fa(DKQ, DV, cColsPerBlock, cCc);
      fattn_kernel_t vFattnKernel = flash_attn_tile<DKQ, DV, cColsPerBlock / ncols2, ncols2, use_logit_softcap>;
      launch_fattn<DV, cColsPerBlock / ncols2, ncols2>(pCtx, pDst, vFattnKernel, cNwarps, cNbytesShared, cNbatchFa,
                                                       true, true, false, cWarpSize);
      return;
    }
  }

  if constexpr (ncols2 <= 4) {
    if (vQ->ne[1] > 2 / ncols2) {
      constexpr int cColsPerBlock = 4;
      const int cNwarps = ggml_cuda_fattn_tile_get_nthreads(DKQ, DV, cColsPerBlock, cCc) / cWarpSize;
      const int cNbatchFa = ggml_cuda_fattn_tile_get_nbatch_fa(DKQ, DV, cColsPerBlock, cCc);
      fattn_kernel_t vFattnKernel = flash_attn_tile<DKQ, DV, cColsPerBlock / ncols2, ncols2, use_logit_softcap>;
      launch_fattn<DV, cColsPerBlock / ncols2, ncols2>(pCtx, pDst, vFattnKernel, cNwarps, cNbytesShared, cNbatchFa,
                                                       true, true, false, cWarpSize);
      return;
    }
  }

  if constexpr (ncols2 <= 2) {
    constexpr int cColsPerBlock = 2;
    const int cNwarps = ggml_cuda_fattn_tile_get_nthreads(DKQ, DV, cColsPerBlock, cCc) / cWarpSize;
    const int cNbatchFa = ggml_cuda_fattn_tile_get_nbatch_fa(DKQ, DV, cColsPerBlock, cCc);
    fattn_kernel_t vFattnKernel = flash_attn_tile<DKQ, DV, cColsPerBlock / ncols2, ncols2, use_logit_softcap>;
    launch_fattn<DV, cColsPerBlock / ncols2, ncols2>(pCtx, pDst, vFattnKernel, cNwarps, cNbytesShared, cNbatchFa, true,
                                                     true, false, cWarpSize);
    return;
  }

  GGML_ABORT("fatal error");
}

template <int DKQ, int DV, bool use_logit_softcap>
static void launch_fattn_tile_switch_ncols2(ggml_backend_cuda_context &pCtx, ggml_tensor *pDst) {
  const ggml_tensor *vKqv = pDst;
  const ggml_tensor *vQ = pDst->src[0];
  const ggml_tensor *vK = pDst->src[1];
  const ggml_tensor *vMask = pDst->src[3];

  float vMaxBias = 0.0f;
  memcpy(&vMaxBias, (const float *)vKqv->op_params + 1, sizeof(float));

  GGML_ASSERT(vQ->ne[2] % vK->ne[2] == 0);
  const int cGqaRatio = vQ->ne[2] / vK->ne[2];

  // On NVIDIA (Pascal and older) the GQA optimizations seem to be detrimental in some cases.
  // However, for DKQ == 576, DV == 512 only the kernel variant with GQA optimizations is implemented.
  const bool cNvidia = GGML_CUDA_CC_IS_NVIDIA(ggml_cuda_info().devices[ggml_cuda_get_device()].cc);
  const int cGqaLimit = cNvidia && cGqaRatio <= 4 && DV <= 256 ? 16 : INT_MAX;
  const bool cUseGqaOpt = vMask && vMaxBias == 0.0f && vQ->ne[1] <= cGqaLimit && vK->ne[1] % FATTN_KQ_STRIDE == 0;

  if constexpr (DKQ == 320) {
    // This branch is only used for Mistral Small 4 which has a GQA ratio of 32.
    // On AMD, simply use that GQA ratio with 32 columns / block since we always have enough SRAM.
    // On NVIDIA however, the tile kernel is only used for GPUs that can't use the mma kernel (Pascal and older).
    // Therefore, use a GQA ratio of 16 with 16 columns / block to stay below 48 kiB of SRAM / block.
#ifdef GGML_USE_HIP
    if (cUseGqaOpt && cGqaRatio % 32 == 0) {
      launch_fattn_tile_switch_ncols1<DKQ, DV, 32, use_logit_softcap>(pCtx, pDst);
      return;
    }
#else
    if (cUseGqaOpt && cGqaRatio % 16 == 0) {
      launch_fattn_tile_switch_ncols1<DKQ, DV, 16, use_logit_softcap>(pCtx, pDst);
      return;
    }
#endif // GGML_USE_HIP
    GGML_ABORT("flash-attn tile (320/256): expected GQA ratio multiple of 32");
  }

  if constexpr (DKQ == 576) {
    if (cUseGqaOpt && cGqaRatio % 16 == 0) {
      launch_fattn_tile_switch_ncols1<DKQ, DV, 16, use_logit_softcap>(pCtx, pDst);
      return;
    }
    if (cUseGqaOpt && cGqaRatio % 4 == 0) {
      launch_fattn_tile_switch_ncols1<DKQ, DV, 4, use_logit_softcap>(pCtx, pDst);
      return;
    }
  }

  if constexpr (DKQ == 192) {
    // MiMo-V2.5 / V2.5-Pro / V2-Flash: gqa_ratio is 8 (SWA) or 16 (full attn)
    if (cUseGqaOpt && cGqaRatio % 16 == 0) {
      launch_fattn_tile_switch_ncols1<DKQ, DV, 16, use_logit_softcap>(pCtx, pDst);
      return;
    }
    if (cUseGqaOpt && cGqaRatio % 8 == 0) {
      launch_fattn_tile_switch_ncols1<DKQ, DV, 8, use_logit_softcap>(pCtx, pDst);
      return;
    }
    GGML_ABORT("flash-attn tile (192/128): expected GQA ratio multiple of 8");
  }

  if constexpr (DKQ <= 512 && DKQ != 320 && DKQ != 192) {
    if (cUseGqaOpt && cGqaRatio % 8 == 0) {
      launch_fattn_tile_switch_ncols1<DKQ, DV, 8, use_logit_softcap>(pCtx, pDst);
      return;
    }

    if (cUseGqaOpt && cGqaRatio % 4 == 0) {
      launch_fattn_tile_switch_ncols1<DKQ, DV, 4, use_logit_softcap>(pCtx, pDst);
      return;
    }

    if (cUseGqaOpt && cGqaRatio % 2 == 0) {
      launch_fattn_tile_switch_ncols1<DKQ, DV, 2, use_logit_softcap>(pCtx, pDst);
      return;
    }

    if constexpr (DV <= 256) {
      launch_fattn_tile_switch_ncols1<DKQ, DV, 1, use_logit_softcap>(pCtx, pDst);
      return;
    }
  }
  GGML_ABORT("fatal error");
}

template <int DKQ, int DV> void ggml_cuda_flash_attn_ext_tile_case(ggml_backend_cuda_context &pCtx, ggml_tensor *pDst) {
  const ggml_tensor *vKqv = pDst;

  float vLogitSoftcap;
  memcpy(&vLogitSoftcap, (const float *)vKqv->op_params + 2, sizeof(float));

  if (vLogitSoftcap == 0.0f) {
    constexpr bool cUseLogitSoftcap = false;
    launch_fattn_tile_switch_ncols2<DKQ, DV, cUseLogitSoftcap>(pCtx, pDst);
  } else {
    constexpr bool cUseLogitSoftcap = true;
    launch_fattn_tile_switch_ncols2<DKQ, DV, cUseLogitSoftcap>(pCtx, pDst);
  }
}

void ggml_cuda_flash_attn_ext_tile(ggml_backend_cuda_context &pCtx, ggml_tensor *pDst);

#define DECL_FATTN_TILE_CASE(DKQ, DV)                                                                                  \
  template void ggml_cuda_flash_attn_ext_tile_case<DKQ, DV>(ggml_backend_cuda_context & ctx, ggml_tensor * dst)

extern DECL_FATTN_TILE_CASE(40, 40);
extern DECL_FATTN_TILE_CASE(64, 64);
extern DECL_FATTN_TILE_CASE(72, 72);
extern DECL_FATTN_TILE_CASE(80, 80);
extern DECL_FATTN_TILE_CASE(96, 96);
extern DECL_FATTN_TILE_CASE(112, 112);
extern DECL_FATTN_TILE_CASE(128, 128);
extern DECL_FATTN_TILE_CASE(192, 128);
extern DECL_FATTN_TILE_CASE(256, 256);
extern DECL_FATTN_TILE_CASE(320, 256);
extern DECL_FATTN_TILE_CASE(512, 512);
extern DECL_FATTN_TILE_CASE(576, 512);
