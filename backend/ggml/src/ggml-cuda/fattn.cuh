#include "common.cuh"

void ggml_cuda_flash_attn_ext(ggml_backend_cuda_context &pCtx, ggml_tensor *pDst);

bool ggml_cuda_flash_attn_ext_supported(int pDevice, const ggml_tensor *pDst);

size_t ggml_cuda_flash_attn_ext_get_alloc_size(int pDevice, const ggml_tensor *pDst);
