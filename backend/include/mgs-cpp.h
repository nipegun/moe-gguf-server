#pragma once

#ifndef __cplusplus
#error "This header is for C++ only"
#endif

#include <memory>

#include "mgs.h"

struct mgs_model_deleter {
  void operator()(mgs_model *model) { mgs_model_free(model); }
};

struct mgs_context_deleter {
  void operator()(mgs_context *context) { mgs_free(context); }
};

struct mgs_sampler_deleter {
  void operator()(mgs_sampler *sampler) { mgs_sampler_free(sampler); }
};

struct mgs_adapter_lora_deleter {
  void operator()(mgs_adapter_lora *adapter) { mgs_adapter_lora_free(adapter); }
};

typedef std::unique_ptr<mgs_model, mgs_model_deleter> mgs_model_ptr;
typedef std::unique_ptr<mgs_context, mgs_context_deleter> mgs_context_ptr;
typedef std::unique_ptr<mgs_sampler, mgs_sampler_deleter> mgs_sampler_ptr;
typedef std::unique_ptr<mgs_adapter_lora, mgs_adapter_lora_deleter> mgs_adapter_lora_ptr;
