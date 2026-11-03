#include "common.cuh"

void ggml_cuda_lightning_indexer(ggml_backend_cuda_context &pCtx, ggml_tensor *pDst);
bool ggml_cuda_lightning_indexer_supported(int pDevice, const ggml_tensor *pDst);
