#pragma once

#include <memory>
#include <string>
#include <vector>

// common debug functions and structs

struct common_params;

// Intended to use as callback for ggml_backend_sched_eval_callback
// prints tensors that are processed in the computation graph
// by default prints all tensors, but can be configured by creating a `common_debug_cb_user_data` instance with
// non-empty pFilterPatterns.
// `common_debug_cb_user_data` contains `vAbortOnNan` flag that determines whether an error should be thrown whenever a
// NaN is encountered in a tensor (useful for stopping debug sessions on first erroneous tensor) The callback data will
// be passed as the third parameter (pUserData)
bool fCommonDebugCbEval(struct ggml_tensor *pT, bool pAsk, void *pUserData);

struct common_debug_cb_user_data {
  struct impl;
  std::unique_ptr<impl> vPimpl;

  common_debug_cb_user_data();
  ~common_debug_cb_user_data();

  common_debug_cb_user_data(const common_debug_cb_user_data &) = delete;
  common_debug_cb_user_data &operator=(const common_debug_cb_user_data &) = delete;

  common_debug_cb_user_data(common_params &pParams, const std::vector<std::string> &pFilterPatterns,
                            bool pAbortOnNan = false);
};
