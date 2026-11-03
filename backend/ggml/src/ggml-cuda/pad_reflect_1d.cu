#include "pad_reflect_1d.cuh"

static __global__ __launch_bounds__(CUDA_PAD_REFLECT_1D_BLOCK_SIZE, 1) void fPadReflect1dKernelF32(
    const void *__restrict__ pSrc0, void *__restrict__ pDst, const int64_t pNe0, const int64_t pNe00, const uint3 pNe01,
    const int64_t pNe02, const int64_t pNe03, const int64_t pNb00, const int64_t pNb01, const int64_t pNb02,
    const int64_t pNb03, const int64_t pNb0, const int64_t pNb1, const int64_t pNb2, const int64_t pNb3, const int p0,
    const int p1) {
  const int64_t cI3 = blockIdx.z;
  const int64_t cI2 = blockIdx.y;

  const uint2 cDivModPacked = fast_div_modulo(blockIdx.x, pNe01);
  const int64_t cTile1 = cDivModPacked.y; // i1
  const int64_t cTile0 = cDivModPacked.x; // nth i0 tile
  const int64_t cI1 = cTile1;
  const int64_t cI0 = threadIdx.x + cTile0 * blockDim.x;

  // ne01.z is original value of unpacked ne01 (see init_fastdiv_values in common.cuh)
  if (cI0 >= pNe0 || cI1 >= pNe01.z || cI2 >= pNe02 || cI3 >= pNe03) {
    return;
  }

  const char *vSrc0Ptr = (const char *)pSrc0 + cI3 * pNb03 + cI2 * pNb02 + cI1 * pNb01;
  char *vDstPtr = (char *)pDst + cI3 * pNb3 + cI2 * pNb2 + cI1 * pNb1;

  const int64_t cRelI0 = cI0 - p0; // relative i0 in src0
  int64_t vSrcIdx;

  if (cRelI0 < 0) {
    // Left padding - reflect
    vSrcIdx = -cRelI0;
  } else if (cRelI0 < pNe00) {
    // Middle - copy
    vSrcIdx = cRelI0;
  } else {
    // Right padding - reflect
    vSrcIdx = 2 * pNe00 - 2 - cRelI0;
  }
  const float cValue = *(const float *)(vSrc0Ptr + vSrcIdx * pNb00);
  *(float *)(vDstPtr + cI0 * pNb0) = cValue;

  GGML_UNUSED(p1);
}

void ggml_cuda_op_pad_reflect_1d(ggml_backend_cuda_context &pCtx, ggml_tensor *pDst) {
  const ggml_tensor *vSrc0 = pDst->src[0];
  cudaStream_t vStream = pCtx.stream();

  GGML_ASSERT(vSrc0->type == GGML_TYPE_F32);
  GGML_ASSERT(pDst->type == GGML_TYPE_F32);

  const int32_t *vOpts = (const int32_t *)pDst->op_params;
  const int cP0 = vOpts[0];
  const int cP1 = vOpts[1];

  const int64_t cNe00 = vSrc0->ne[0];
  const int64_t cNe01 = vSrc0->ne[1];
  const uint3 cNe01Packed = init_fastdiv_values(cNe01);
  const int64_t cNe02 = vSrc0->ne[2];
  const int64_t cNe03 = vSrc0->ne[3];

  const int64_t cNe0 = pDst->ne[0];

  // sanity: padded length matches
  GGML_ASSERT(cNe0 == cNe00 + cP0 + cP1);

  constexpr int64_t cBx = CUDA_PAD_REFLECT_1D_BLOCK_SIZE; // threads per block (x)
  const int64_t cTiles0 = (cNe0 + cBx - 1) / cBx;         // number of tiles along i0
  // grid.x covers i1 and all tiles of i0: [ne01 * tiles0]
  // grid.y covers i2: [ne02]
  // grid.z covers i3: [ne03]
  const dim3 cGridDims((unsigned)(cNe01 * cTiles0), (unsigned)cNe02, (unsigned)cNe03);
  const dim3 cBlockDims((unsigned)cBx, 1, 1);

  fPadReflect1dKernelF32<<<cGridDims, cBlockDims, 0, vStream>>>(
      vSrc0->data, pDst->data, cNe0, cNe00, cNe01Packed, cNe02, cNe03, vSrc0->nb[0], vSrc0->nb[1], vSrc0->nb[2],
      vSrc0->nb[3], pDst->nb[0], pDst->nb[1], pDst->nb[2], pDst->nb[3], cP0, cP1);
}
