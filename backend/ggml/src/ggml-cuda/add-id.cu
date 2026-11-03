#include "add-id.cuh"

static __global__ void fAddIdKernel(const float *pSrc0, const float *pSrc1, const int32_t *pSrc2, float *pDst,
                                    int64_t pNe0, int64_t pNe1, size_t pNb01, size_t pNb02, size_t pNb11,
                                    size_t pNb21) {

  const int64_t cI1 = blockIdx.x;
  const int64_t cI2 = blockIdx.y;

  const int cI11 = *(const int32_t *)((const char *)pSrc2 + cI1 * sizeof(int32_t) + cI2 * pNb21);

  const size_t cNb1 = pNe0 * sizeof(float);
  const size_t cNb2 = pNe1 * cNb1;

  float *vDstRow = (float *)((char *)pDst + cI1 * cNb1 + cI2 * cNb2);
  const float *vSrc0Row = (const float *)((const char *)pSrc0 + cI1 * pNb01 + cI2 * pNb02);
  const float *vSrc1Row = (const float *)((const char *)pSrc1 + cI11 * pNb11);

  for (int64_t vI0 = threadIdx.x; vI0 < pNe0; vI0 += blockDim.x) {
    vDstRow[vI0] = vSrc0Row[vI0] + vSrc1Row[vI0];
  }
}

void ggml_cuda_op_add_id(ggml_backend_cuda_context &pCtx, ggml_tensor *dst) {
  const ggml_tensor *src0 = dst->src[0];
  const ggml_tensor *src1 = dst->src[1];
  const ggml_tensor *src2 = dst->src[2];

  GGML_TENSOR_TERNARY_OP_LOCALS

  GGML_ASSERT(dst->type == GGML_TYPE_F32);
  GGML_ASSERT(src0->type == GGML_TYPE_F32);
  GGML_ASSERT(src1->type == GGML_TYPE_F32);
  GGML_ASSERT(src2->type == GGML_TYPE_I32);

  GGML_ASSERT(nb00 == sizeof(float));
  GGML_ASSERT(nb10 == sizeof(float));
  GGML_ASSERT(nb20 == sizeof(int32_t));

  const float *vSrc0D = (const float *)src0->data;
  const float *vSrc1D = (const float *)src1->data;
  const int32_t *vSrc2D = (const int32_t *)src2->data;
  float *vDstD = (float *)dst->data;

  int vThreads = std::min((int)ne00, 768); // cols
  dim3 vBlocks(ne01, ne02);                // n_experts_used, n_tokens
  fAddIdKernel<<<vBlocks, vThreads, 0, pCtx.stream()>>>(vSrc0D, vSrc1D, vSrc2D, vDstD, ne0, ne1, nb01, nb02, nb11,
                                                        nb21);
}
