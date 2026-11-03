#pragma once
#include "common.h"

size_t ggml_backend_amx_desired_wsize(const struct ggml_tensor *pDst);

size_t ggml_backend_amx_get_alloc_size(const struct ggml_tensor *pTensor);

void ggml_backend_amx_convert_weight(struct ggml_tensor *pTensor, const void *pData, size_t pOffset, size_t pSize);

void ggml_backend_amx_mul_mat(const struct ggml_compute_params *pParams, struct ggml_tensor *pDst);
