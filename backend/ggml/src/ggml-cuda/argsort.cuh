#include "common.cuh"

void ggml_cuda_op_argsort(ggml_backend_cuda_context &pCtx, ggml_tensor *pDst);

#ifdef GGML_CUDA_USE_CUB
int argsort_f32_i32_cuda_cub_chunk_nrows(const size_t pNb01, const int64_t pNrows);
void argsort_f32_i32_cuda_cub(ggml_cuda_pool &pPool, const float *pX, int *pDst, const int pNcols, const int pNrows,
                              ggml_sort_order pOrder, cudaStream_t pStream);
#endif // GGML_CUDA_USE_CUB
void argsort_f32_i32_cuda_bitonic(const float *pX, int *pDst, const int pNcols, const int pNrows,
                                  ggml_sort_order pOrder, cudaStream_t pStream);
