#include "set.cuh"
#include "cpy.cuh"

void ggml_cuda_op_set(ggml_backend_cuda_context &pCtx, ggml_tensor *pDst) {
  const ggml_tensor *vSrc0 = pDst->src[0];
  const ggml_tensor *vSrc1 = pDst->src[1];

  GGML_ASSERT((vSrc0->type == GGML_TYPE_F32 || vSrc0->type == GGML_TYPE_I32));
  GGML_ASSERT(vSrc1->type == vSrc0->type);
  GGML_ASSERT(pDst->type == vSrc0->type);

  GGML_ASSERT(ggml_is_contiguous(pDst));
  GGML_ASSERT(ggml_is_contiguous(vSrc0));
  GGML_ASSERT(ggml_is_contiguous(vSrc1));

  const size_t cNb1 = ((int32_t *)pDst->op_params)[0];
  const size_t cNb2 = ((int32_t *)pDst->op_params)[1];
  const size_t cNb3 = ((int32_t *)pDst->op_params)[2];
  const size_t cOffset = ((int32_t *)pDst->op_params)[3];
  const bool cInplace = (bool)((int32_t *)pDst->op_params)[4];

  if (!cInplace) {
    ggml_cuda_cpy(pCtx, vSrc0, pDst);
  }

  ggml_tensor vDstView = *pDst;
  vDstView.data = (void *)((char *)pDst->data + cOffset);
  vDstView.ne[0] = vSrc1->ne[0];
  vDstView.ne[1] = vSrc1->ne[1];
  vDstView.ne[2] = vSrc1->ne[2];
  vDstView.ne[3] = vSrc1->ne[3];

  vDstView.nb[0] = ggml_element_size(pDst);
  vDstView.nb[1] = cNb1;
  vDstView.nb[2] = cNb2;
  vDstView.nb[3] = cNb3;

  ggml_cuda_cpy(pCtx, vSrc1, &vDstView);
}
