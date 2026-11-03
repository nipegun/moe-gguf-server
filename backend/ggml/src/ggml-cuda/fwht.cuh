#include "common.cuh"

// Returns whether the Fast Walsh-Hadamard transform could be used.
bool ggml_cuda_op_fwht(ggml_backend_cuda_context &pCtx, const ggml_tensor *pSrc, ggml_tensor *pDst);
