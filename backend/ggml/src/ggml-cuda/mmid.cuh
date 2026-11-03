#pragma once

void ggml_cuda_launch_mm_ids_helper(const int32_t *pIds, int32_t *pIdsSrc1, int32_t *pIdsDst, int32_t *pExpertBounds,
                                    int pNExperts, int pNTokens, int pNExpertUsed, int pNchannelsY, int pSi1, int pSis1,
                                    bool pWriteInverse, cudaStream_t pStream);
