#pragma once

#include "ggml.h"
#include "mgs.h"

#include <vector>

enum common_params_fit_status {
  cCommonParamsFitStatusSuccess = 0, // found allocations that are projected to fit
  cCommonParamsFitStatusFailure = 1, // could not find allocations that are projected to fit
  cCommonParamsFitStatusError = 2, // a hard error occurred, e.g. because no model could be found at the specified path
};

// fits mparams and cparams to free device memory (assumes system memory is unlimited)
//   - returns true if the parameters could be successfully modified to fit device memory
//   - this function is NOT thread safe because it modifies the global mgs logger state
//   - only parameters that have the same value as in mgs_default_model_params are modified
//     with the exception of the context size which is modified if and only if equal to 0
//     and of the batch sizes, which are raised when pAutoBatch is set and the MoE experts stay in system memory
common_params_fit_status fCommonFitParams(
    const char *pPathModel, mgs_model_params *pMparams, mgs_context_params *pCparams,
    float *pTensorSplit, // writable buffer for tensor split, needs at least mgs_max_devices elements
    mgs_model_tensor_buft_override
        *pTensorBuftOverrides, // writable buffer for overrides, needs at least mgs_max_tensor_buft_overrides elements
    size_t *pMargins,          // margins of memory to leave per device in bytes
    uint32_t pNCtxMin,         // minimum context size to set when trying to reduce memory use
    ggml_log_level pLogLevel,  // minimum log level to print during fitting, lower levels go to debug log
    bool pAutoBatch = false);  // the batch sizes were not given, so they may be raised

// print estimated memory to stdout
void fCommonFitPrint(const char *pPathModel, mgs_model_params *pMparams, mgs_context_params *pCparams);

void fCommonMemoryBreakdownPrint(const mgs_context *pCtx);

struct common_device_memory_data {
  int64_t vTotal;
  int64_t vFree;
  size_t vModel;
  size_t vContext;
  size_t vCompute;
};

using common_device_memory_data_vec = std::vector<common_device_memory_data>;

// Load a model + context with no_alloc and return the per-device memory breakdown.
common_device_memory_data_vec fCommonGetDeviceMemoryData(const char *pPathModel, const mgs_model_params *pMparams,
                                                         const mgs_context_params *pCparams,
                                                         std::vector<ggml_backend_dev_t> &pDevs, uint32_t &pHpNgl,
                                                         uint32_t &pHpNCtxTrain, uint32_t &pHpNExpert,
                                                         ggml_log_level pLogLevel);
