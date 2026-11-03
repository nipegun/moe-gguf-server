#include "amax-pow2.cuh"

// rows handled by each block, one warp per row
static constexpr int cAmaxPow2RowsPerBlock = 4;

// Same value as fGgmlAmaxPow2F32 on the CPU: 2^clamp(ceil(log2(pAmax / pThreshold)), 0, 127) from the bits of the
// ratio.
static __device__ __forceinline__ float fAmaxPow2(const float pAmax, const float pThreshold) {
  const float cRatio = pAmax / pThreshold;
  if (!(cRatio > 1.0f)) {
    return 1.0f;
  }
  const uint32_t cBits = __float_as_uint(cRatio);
  int32_t vExp = (int32_t)((cBits >> 23) & 0xFF) - 127;
  if ((cBits & 0x7FFFFF) != 0) {
    vExp += 1;
  }
  vExp = min(vExp, 127);
  return __uint_as_float((uint32_t)(vExp + 127) << 23);
}

static __global__ void fAmaxPow2F32(const float *__restrict__ pX, float *__restrict__ pDst, const int64_t pNe00,
                                    const int64_t pNe01, const int64_t pNe02, const int64_t pNRows, const int64_t pS01,
                                    const int64_t pS02, const int64_t pS03, const float pThreshold) {
  const int64_t cRow = (int64_t)blockIdx.x * cAmaxPow2RowsPerBlock + threadIdx.y;
  if (cRow >= pNRows) {
    return;
  }

  const int64_t cI03 = cRow / (pNe02 * pNe01);
  const int64_t cI02 = (cRow - cI03 * pNe02 * pNe01) / pNe01;
  const int64_t cI01 = cRow - cI03 * pNe02 * pNe01 - cI02 * pNe01;

  const float *vRow = pX + cI01 * pS01 + cI02 * pS02 + cI03 * pS03;

  float vAmax = 0.0f;
  for (int64_t vCol = threadIdx.x; vCol < pNe00; vCol += WARP_SIZE) {
    vAmax = fmaxf(vAmax, fabsf(vRow[vCol]));
  }
  vAmax = warp_reduce_max(vAmax);

  if (threadIdx.x == 0) {
    pDst[cRow] = fAmaxPow2(vAmax, pThreshold);
  }
}

void ggml_cuda_op_amax_pow2(ggml_backend_cuda_context &pCtx, ggml_tensor *pDst) {
  const ggml_tensor *vSrc0 = pDst->src[0];

  GGML_ASSERT(vSrc0->type == GGML_TYPE_F32);
  GGML_ASSERT(pDst->type == GGML_TYPE_F32);
  GGML_ASSERT(vSrc0->nb[0] == sizeof(float));
  GGML_ASSERT(ggml_is_contiguous(pDst));

  const float cThreshold = ggml_get_op_params_f32(pDst, 0);

  const int64_t cNRows = ggml_nrows(vSrc0);
  if (cNRows == 0) {
    return;
  }

  const size_t cTs = sizeof(float);
  const int64_t cS01 = vSrc0->nb[1] / cTs;
  const int64_t cS02 = vSrc0->nb[2] / cTs;
  const int64_t cS03 = vSrc0->nb[3] / cTs;

  const dim3 cBlockDims(WARP_SIZE, cAmaxPow2RowsPerBlock, 1);
  const dim3 cBlockNums((cNRows + cAmaxPow2RowsPerBlock - 1) / cAmaxPow2RowsPerBlock, 1, 1);

  fAmaxPow2F32<<<cBlockNums, cBlockDims, 0, pCtx.stream()>>>(
      (const float *)vSrc0->data, (float *)pDst->data, vSrc0->ne[0], vSrc0->ne[1], vSrc0->ne[2], cNRows, cS01, cS02,
      cS03, cThreshold);
}
