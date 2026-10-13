#pragma once

// this is a staging header for new mgs API
// breaking changes and C++ are allowed. everything here should be considered WIP
// try as much as possible to not include this header in the rest of the codebase

#include "mgs.h"

#include <cstdint>
#include <map>

// Reserve a new compute graph. It is valid until the next call to mgs_graph_reserve.
MGS_API struct ggml_cgraph *mgs_graph_reserve(struct mgs_context *ctx, uint32_t n_tokens, uint32_t n_seqs,
                                                  uint32_t n_outputs);

// Get the default ggml_type for a given ftype.
MGS_API ggml_type mgs_ftype_get_default_type(mgs_ftype ftype);

struct quantize_state_impl;

MGS_API quantize_state_impl *mgs_quant_init(const mgs_model *model, const mgs_model_quantize_params *params);

MGS_API void mgs_quant_free(quantize_state_impl *qs);

// Descriptor for constructing a mock model for quantization testing.
struct mgs_quant_model_desc {
  const char *architecture;
  uint32_t n_embd;
  uint32_t n_ff;
  uint32_t n_layer;
  uint32_t n_head;
  uint32_t n_head_kv;
  uint32_t n_expert;
  uint32_t n_embd_head_k;
  uint32_t n_embd_head_v;
};

// Create a mock model from a metadata descriptor (for testing).
// The returned model must be freed with mgs_model_free().
MGS_API mgs_model *mgs_quant_model_from_metadata(const mgs_quant_model_desc *desc);

// Returns true if this tensor should be quantized (based on name, dims, params).
MGS_API bool mgs_quant_tensor_allows_quantization(const quantize_state_impl *qs, const ggml_tensor *tensor);

// Compute quantization type assignments for a list of tensors.
// All tensors should be quantizable (use mgs_quant_tensor_allows_quantization to filter).
// result_types: caller-allocated array of n_tensors elements, filled with assigned types.
MGS_API void mgs_quant_compute_types(quantize_state_impl *qs, mgs_ftype ftype, ggml_tensor **tensors,
                                         ggml_type *result_types, size_t n_tensors);

//
// device memory querying
//

// "memory" as in physical memory for a buffer type, in bytes
struct mgs_memory_breakdown_data {
  size_t model = 0;   // memory allocated for the model
  size_t context = 0; // memory allocated for the context
  size_t compute = 0; // memory allocated for temporary compute buffers

  size_t total() const { return model + context + compute; }
};

struct mgs_device_memory_data {
  int64_t total;
  int64_t free;
  mgs_memory_breakdown_data mb;
};

// TODO: convert to C-style data structure
using mgs_memory_breakdown = std::map<ggml_backend_buffer_type_t, mgs_memory_breakdown_data>;

MGS_API int32_t mgs_model_n_expert(const struct mgs_model *model);
MGS_API int32_t mgs_model_n_devices(const struct mgs_model *model);

MGS_API ggml_backend_dev_t mgs_model_get_device(const struct mgs_model *model, int i);

MGS_API mgs_memory_breakdown mgs_get_memory_breakdown(const struct mgs_context *ctx);

// Set whether the context outputs nextn embeddings or not
// If masked == true,  output the embeddings only for the tokens with batch.logits != 0
// If masked == false, output the embeddings for all tokens in the batch regardless of batch.logits
MGS_API void mgs_set_embeddings_nextn(struct mgs_context *ctx, bool value, bool masked);

// Select which appended NextN block the DECODER_MTP graph runs (offset past
// the trunk: il = n_layer() + offset). Used by the speculative NextN driver to
// chain multiple trained NextN heads. Default 0 (first head).
MGS_API void mgs_set_nextn_layer_offset(struct mgs_context *ctx, int32_t offset);

// mirrors:
// MGS_API float * mgs_get_embeddings(struct mgs_context * ctx);
MGS_API float *mgs_get_embeddings_nextn(struct mgs_context *ctx);

// MGS_API float * mgs_get_embeddings_ith(struct mgs_context * ctx, int32_t i);
MGS_API float *mgs_get_embeddings_nextn_ith(struct mgs_context *ctx, int32_t i);

// Set whether the context outputs the input embeddings of a specific layer
MGS_API void mgs_set_embeddings_layer_inp(struct mgs_context *ctx, uint32_t lid, bool value);

// mirrors:
// MGS_API float * mgs_get_embeddings(struct mgs_context * ctx);
MGS_API float *mgs_get_embeddings_layer_inp(struct mgs_context *ctx, uint32_t lid);

MGS_API mgs_context *mgs_get_ctx_other(struct mgs_context *ctx);

//
// model/context data extraction
//

// returns pointer to the target-model layer indices
MGS_API const int32_t *mgs_model_target_layer_ids(const struct mgs_model *model);
// returns the number of extracted layers from target model
MGS_API uint32_t mgs_model_target_layer_ids_n(const struct mgs_model *model);

//
// MoE expert cache
//

// State of the device cache that holds the most recently used MoE experts kept in host memory.
struct mgs_expert_cache_info {
  bool active = false;               // a cache is allocated
  size_t size = 0;                   // device memory of the cache, in bytes
  int32_t n_layer = 0;               // layers served by the cache
  int64_t n_slot = 0;                // experts the cache can hold
  int64_t n_expert = 0;              // experts of the cached layers
  uint64_t n_step = 0;               // decoding steps executed with the hybrid path
  uint64_t n_hit = 0;                // selected experts found in the cache
  uint64_t n_miss = 0;               // selected experts missing from the cache
  uint64_t n_upload = 0;             // missing experts uploaded into the cache
  uint64_t n_cpu_expert = 0;         // missing experts computed on the CPU
  uint64_t prefetch_cache_bytes = 0; // expert prefetch bytes copied from the cache
  uint64_t prefetch_host_bytes = 0;  // expert prefetch bytes uploaded from host memory
  double upload_bandwidth = 0.0;     // measured upload bandwidth, bytes per second
  double cpu_bandwidth = 0.0;        // estimated CPU expert bandwidth, bytes per second
};

MGS_API mgs_expert_cache_info mgs_get_expert_cache_info(const struct mgs_context *ctx);

// Request a new expert cache size in MiB: -1 restores the automatic size and 0 disables the cache.
// The next decode rebuilds the cache, empty, between two micro-batches.
MGS_API void mgs_set_expert_cache_size(struct mgs_context *ctx, int64_t size_mib);

//
// memory monitoring
//

// Part of the weights of a model in one buffer type (both are included in mgs_memory_breakdown_data::model).
struct mgs_model_weight_split {
  size_t mtp = 0;     // weights of the MTP (nextn) layers
  size_t experts = 0; // routed expert tensors (*exps.*) outside the MTP layers
};

using mgs_model_weight_breakdown = std::map<ggml_backend_buffer_type_t, mgs_model_weight_split>;

// Walks every tensor of the model: call it once after loading, the result does not change.
MGS_API mgs_model_weight_breakdown mgs_model_get_weight_breakdown(const struct mgs_model *model);

// Memory of the MoE expert machinery of a context in one buffer type (not part of mgs_get_memory_breakdown).
struct mgs_expert_memory_data {
  size_t cache = 0;    // expert cache: pools, slot identifiers and pinned staging
  size_t prefetch = 0; // expert prefetch slots of the scheduler
};

using mgs_expert_memory_breakdown = std::map<ggml_backend_buffer_type_t, mgs_expert_memory_data>;

// Not thread-safe: call it from the thread that decodes with the context.
MGS_API mgs_expert_memory_breakdown mgs_get_expert_memory_breakdown(const struct mgs_context *ctx);
