#pragma once

#include "mtmd.h"

#include <vector>

// INTERNAL HEADER FOR DEBUGGING PURPOSES ONLY
// NOT INTENDED FOR PUBLIC USE
// Do not raise issues related to this debugging API

// encode take the pre-processed f32 values, print the intermidiate values via cb_eval callback
MTMD_API void mtmd_debug_encode_image(mtmd_context *pCtx, const std::vector<std::vector<float>> &pImage);
MTMD_API void mtmd_debug_encode_audio(mtmd_context *pCtx,
                                      const std::vector<float> &pInput); // will be broadcasted to fit n_mel

// preprocess take the raw input values
MTMD_API void mtmd_debug_preprocess_image(mtmd_context *pCtx, const std::vector<uint8_t> &pRgbValues, int pNx, int pNy);
MTMD_API void mtmd_debug_preprocess_audio(mtmd_context *pCtx, const std::vector<float> &pPcmSamples);
