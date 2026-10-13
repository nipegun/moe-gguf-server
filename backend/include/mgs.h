#ifndef MGS_H
#define MGS_H

#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"
#include "ggml-opt.h"
#include "gguf.h"

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdbool.h>

#ifdef MGS_SHARED
#if defined(_WIN32) && !defined(__MINGW32__)
#ifdef MGS_BUILD
#define MGS_API __declspec(dllexport)
#else
#define MGS_API __declspec(dllimport)
#endif
#else
#define MGS_API __attribute__((visibility("default")))
#endif
#else
#define MGS_API
#endif

#define MGS_DEFAULT_SEED 0xFFFFFFFF

#define MGS_TOKEN_NULL -1

#define MGS_FILE_MAGIC_GGSN 0x6767736eu // 'ggsn'
#define MGS_FILE_MAGIC_GGSQ 0x67677371u // 'ggsq'

#define MGS_SESSION_MAGIC MGS_FILE_MAGIC_GGSN
#define MGS_SESSION_VERSION 9

#define MGS_STATE_SEQ_MAGIC MGS_FILE_MAGIC_GGSQ
#define MGS_STATE_SEQ_VERSION 2

#ifdef __cplusplus
extern "C" {
#endif

//
// C interface
//
// TODO: show sample usage
//

struct mgs_vocab;
struct mgs_model;
struct mgs_context;
struct mgs_sampler;

typedef struct mgs_memory_i *mgs_memory_t;

typedef int32_t mgs_pos;
typedef int32_t mgs_token;
typedef int32_t mgs_seq_id;

enum mgs_vocab_type {
  MGS_VOCAB_TYPE_NONE = 0,   // For models without vocab
  MGS_VOCAB_TYPE_SPM = 1,    // SentencePiece tokenizer based on byte-level BPE with byte fallback
  MGS_VOCAB_TYPE_BPE = 2,    // GPT-2 tokenizer based on byte-level BPE
  MGS_VOCAB_TYPE_WPM = 3,    // BERT tokenizer based on WordPiece
  MGS_VOCAB_TYPE_UGM = 4,    // T5 tokenizer based on Unigram
  MGS_VOCAB_TYPE_RWKV = 5,   // RWKV tokenizer based on greedy tokenization
  MGS_VOCAB_TYPE_PLAMO2 = 6, // PLaMo-2 tokenizer based on Aho-Corasick with dynamic programming
};

enum mgs_rope_type {
  MGS_ROPE_TYPE_NONE = -1,
  MGS_ROPE_TYPE_NORM = 0,
  MGS_ROPE_TYPE_NEOX = GGML_ROPE_TYPE_NEOX,
  MGS_ROPE_TYPE_MROPE = GGML_ROPE_TYPE_MROPE,
  MGS_ROPE_TYPE_IMROPE = GGML_ROPE_TYPE_IMROPE,
  MGS_ROPE_TYPE_VISION = GGML_ROPE_TYPE_VISION,
};

enum mgs_token_type { // TODO: remove, required until per token attributes are available from GGUF file
  MGS_TOKEN_TYPE_UNDEFINED = 0,
  MGS_TOKEN_TYPE_NORMAL = 1,
  MGS_TOKEN_TYPE_UNKNOWN = 2,
  MGS_TOKEN_TYPE_CONTROL = 3,
  MGS_TOKEN_TYPE_USER_DEFINED = 4,
  MGS_TOKEN_TYPE_UNUSED = 5,
  MGS_TOKEN_TYPE_BYTE = 6,
};

enum mgs_token_attr {
  MGS_TOKEN_ATTR_UNDEFINED = 0,
  MGS_TOKEN_ATTR_UNKNOWN = 1 << 0,
  MGS_TOKEN_ATTR_UNUSED = 1 << 1,
  MGS_TOKEN_ATTR_NORMAL = 1 << 2,
  MGS_TOKEN_ATTR_CONTROL = 1 << 3, // SPECIAL?
  MGS_TOKEN_ATTR_USER_DEFINED = 1 << 4,
  MGS_TOKEN_ATTR_BYTE = 1 << 5,
  MGS_TOKEN_ATTR_NORMALIZED = 1 << 6,
  MGS_TOKEN_ATTR_LSTRIP = 1 << 7,
  MGS_TOKEN_ATTR_RSTRIP = 1 << 8,
  MGS_TOKEN_ATTR_SINGLE_WORD = 1 << 9,
};

// model file types
enum mgs_ftype {
  MGS_FTYPE_ALL_F32 = 0,
  MGS_FTYPE_MOSTLY_F16 = 1,  // except 1d tensors
  MGS_FTYPE_MOSTLY_Q4_0 = 2, // except 1d tensors
  MGS_FTYPE_MOSTLY_Q4_1 = 3, // except 1d tensors
  // MGS_FTYPE_MOSTLY_Q4_1_SOME_F16 = 4,  // tok_embeddings.weight and output.weight are F16
  // MGS_FTYPE_MOSTLY_Q4_2       = 5,  // support has been removed
  // MGS_FTYPE_MOSTLY_Q4_3       = 6,  // support has been removed
  MGS_FTYPE_MOSTLY_Q8_0 = 7,     // except 1d tensors
  MGS_FTYPE_MOSTLY_Q5_0 = 8,     // except 1d tensors
  MGS_FTYPE_MOSTLY_Q5_1 = 9,     // except 1d tensors
  MGS_FTYPE_MOSTLY_Q2_K = 10,    // except 1d tensors
  MGS_FTYPE_MOSTLY_Q3_K_S = 11,  // except 1d tensors
  MGS_FTYPE_MOSTLY_Q3_K_M = 12,  // except 1d tensors
  MGS_FTYPE_MOSTLY_Q3_K_L = 13,  // except 1d tensors
  MGS_FTYPE_MOSTLY_Q4_K_S = 14,  // except 1d tensors
  MGS_FTYPE_MOSTLY_Q4_K_M = 15,  // except 1d tensors
  MGS_FTYPE_MOSTLY_Q5_K_S = 16,  // except 1d tensors
  MGS_FTYPE_MOSTLY_Q5_K_M = 17,  // except 1d tensors
  MGS_FTYPE_MOSTLY_Q6_K = 18,    // except 1d tensors
  MGS_FTYPE_MOSTLY_IQ2_XXS = 19, // except 1d tensors
  MGS_FTYPE_MOSTLY_IQ2_XS = 20,  // except 1d tensors
  MGS_FTYPE_MOSTLY_Q2_K_S = 21,  // except 1d tensors
  MGS_FTYPE_MOSTLY_IQ3_XS = 22,  // except 1d tensors
  MGS_FTYPE_MOSTLY_IQ3_XXS = 23, // except 1d tensors
  MGS_FTYPE_MOSTLY_IQ1_S = 24,   // except 1d tensors
  MGS_FTYPE_MOSTLY_IQ4_NL = 25,  // except 1d tensors
  MGS_FTYPE_MOSTLY_IQ3_S = 26,   // except 1d tensors
  MGS_FTYPE_MOSTLY_IQ3_M = 27,   // except 1d tensors
  MGS_FTYPE_MOSTLY_IQ2_S = 28,   // except 1d tensors
  MGS_FTYPE_MOSTLY_IQ2_M = 29,   // except 1d tensors
  MGS_FTYPE_MOSTLY_IQ4_XS = 30,  // except 1d tensors
  MGS_FTYPE_MOSTLY_IQ1_M = 31,   // except 1d tensors
  MGS_FTYPE_MOSTLY_BF16 = 32,    // except 1d tensors
  // MGS_FTYPE_MOSTLY_Q4_0_4_4      = 33, // removed from gguf files, use Q4_0 and runtime repack
  // MGS_FTYPE_MOSTLY_Q4_0_4_8      = 34, // removed from gguf files, use Q4_0 and runtime repack
  // MGS_FTYPE_MOSTLY_Q4_0_8_8      = 35, // removed from gguf files, use Q4_0 and runtime repack
  MGS_FTYPE_MOSTLY_TQ1_0 = 36,     // except 1d tensors
  MGS_FTYPE_MOSTLY_TQ2_0 = 37,     // except 1d tensors
  MGS_FTYPE_MOSTLY_MXFP4_MOE = 38, // except 1d tensors
  MGS_FTYPE_MOSTLY_NVFP4 = 39,     // except 1d tensors
  MGS_FTYPE_MOSTLY_Q1_0 = 40,      // except 1d tensors
  MGS_FTYPE_MOSTLY_Q2_0 = 41,      // except 1d tensors

  MGS_FTYPE_GUESSED = 1024, // not specified in the model file
};

// Get the model file type (quantization) as a string, e.g. "Q8_0" or "Q4_K - Medium"
MGS_API const char *mgs_ftype_name(enum mgs_ftype ftype);

enum mgs_rope_scaling_type {
  MGS_ROPE_SCALING_TYPE_UNSPECIFIED = -1,
  MGS_ROPE_SCALING_TYPE_NONE = 0,
  MGS_ROPE_SCALING_TYPE_LINEAR = 1,
  MGS_ROPE_SCALING_TYPE_YARN = 2,
  MGS_ROPE_SCALING_TYPE_LONGROPE = 3,
  MGS_ROPE_SCALING_TYPE_MAX_VALUE = MGS_ROPE_SCALING_TYPE_LONGROPE,
};

enum mgs_pooling_type {
  MGS_POOLING_TYPE_UNSPECIFIED = -1,
  MGS_POOLING_TYPE_NONE = 0,
  MGS_POOLING_TYPE_MEAN = 1,
  MGS_POOLING_TYPE_CLS = 2,
  MGS_POOLING_TYPE_LAST = 3,
  MGS_POOLING_TYPE_RANK = 4, // used by reranking models to attach the classification head to the graph
};

enum mgs_attention_type {
  MGS_ATTENTION_TYPE_UNSPECIFIED = -1,
  MGS_ATTENTION_TYPE_CAUSAL = 0,
  MGS_ATTENTION_TYPE_NON_CAUSAL = 1,
};

enum mgs_flash_attn_type {
  MGS_FLASH_ATTN_TYPE_AUTO = -1,
  MGS_FLASH_ATTN_TYPE_DISABLED = 0,
  MGS_FLASH_ATTN_TYPE_ENABLED = 1,
};

MGS_API const char *mgs_flash_attn_type_name(enum mgs_flash_attn_type flash_attn_type);

enum mgs_split_mode {
  MGS_SPLIT_MODE_NONE = 0,  // single GPU
  MGS_SPLIT_MODE_LAYER = 1, // split layers and KV across GPUs
  MGS_SPLIT_MODE_ROW = 2,   // split layers and KV across GPUs, use tensor parallelism if supported
  MGS_SPLIT_MODE_TENSOR = 3,
};

enum mgs_load_mode {
  MGS_LOAD_MODE_NONE = 0,       // no special loading mode
  MGS_LOAD_MODE_MMAP = 1,       // memory map the model
  MGS_LOAD_MODE_MLOCK = 2,      // force system to keep model in RAM rather than swapping or compressing
  MGS_LOAD_MODE_MMAP_MLOCK = 3, // mmap + force system to keep model in RAM rather than swapping or compressing
  MGS_LOAD_MODE_DIRECT_IO = 4,  // use direct I/O if available
};

MGS_API const char *mgs_load_mode_name(enum mgs_load_mode load_mode);
MGS_API enum mgs_load_mode mgs_load_mode_from_str(const char *str);

enum mgs_context_type {
  MGS_CONTEXT_TYPE_DEFAULT = 0,
  MGS_CONTEXT_TYPE_MTP = 1,
};

// TODO: simplify
typedef struct mgs_token_data {
  mgs_token id; // token id
  float logit;    // log-odds of the token
  float p;        // probability of the token
} mgs_token_data;

typedef struct mgs_token_data_array {
  // TODO: consider SoA
  // NOTE: this pointer can be modified by the samplers
  mgs_token_data *data;
  size_t size;
  int64_t selected; // this is the index in the data array (i.e. not the token id)
  bool sorted;      // note: do not assume the data is sorted - always check this flag
} mgs_token_data_array;

typedef bool (*mgs_progress_callback)(float progress, void *user_data);

// Input data for mgs_encode/mgs_decode
// A mgs_batch object can contain input about one or many sequences
// The provided arrays (i.e. token, embd, pos, etc.) must have size of n_tokens
//
// - token  : the token ids of the input (used when embd is NULL)
// - embd   : token embeddings (i.e. float vector of size n_embd) (used when token is NULL)
// - pos    : the positions of the respective token in the sequence
//            (if set to NULL, the token position will be tracked automatically by mgs_encode/mgs_decode)
// - seq_id : the sequence to which the respective token belongs
//            (if set to NULL, the sequence ID will be assumed to be 0)
// - logits : if zero, the logits (and/or the embeddings) for the respective token will not be output
//            (if set to NULL:
//               - if embeddings: all tokens are output
//               - if not:        only the last token is output
//            )
//
typedef struct mgs_batch {
  int32_t n_tokens;

  mgs_token *token;
  float *embd;
  mgs_pos *pos;
  int32_t *n_seq_id;
  mgs_seq_id **seq_id;
  int8_t *logits; // TODO: rename this to "output"
} mgs_batch;

enum mgs_model_kv_override_type {
  MGS_KV_OVERRIDE_TYPE_INT,
  MGS_KV_OVERRIDE_TYPE_FLOAT,
  MGS_KV_OVERRIDE_TYPE_BOOL,
  MGS_KV_OVERRIDE_TYPE_STR,
};

enum mgs_model_meta_key {
  MGS_MODEL_META_KEY_SAMPLING_SEQUENCE,
  MGS_MODEL_META_KEY_SAMPLING_TOP_K,
  MGS_MODEL_META_KEY_SAMPLING_TOP_P,
  MGS_MODEL_META_KEY_SAMPLING_MIN_P,
  MGS_MODEL_META_KEY_SAMPLING_XTC_PROBABILITY,
  MGS_MODEL_META_KEY_SAMPLING_XTC_THRESHOLD,
  MGS_MODEL_META_KEY_SAMPLING_TEMP,
  MGS_MODEL_META_KEY_SAMPLING_PENALTY_LAST_N,
  MGS_MODEL_META_KEY_SAMPLING_PENALTY_REPEAT,
  MGS_MODEL_META_KEY_SAMPLING_MIROSTAT,
  MGS_MODEL_META_KEY_SAMPLING_MIROSTAT_TAU,
  MGS_MODEL_META_KEY_SAMPLING_MIROSTAT_ETA,
};

struct mgs_model_kv_override {
  enum mgs_model_kv_override_type tag;

  char key[128];

  union {
    int64_t val_i64;
    double val_f64;
    bool val_bool;
    char val_str[128];
  };
};

struct mgs_model_tensor_buft_override {
  const char *pattern;
  ggml_backend_buffer_type_t buft;
};

struct mgs_model_params {
  // NULL-terminated list of devices to use for offloading (if NULL, all available devices are used)
  ggml_backend_dev_t *devices;

  // NULL-terminated list of buffer types to use for tensors that match a pattern
  const struct mgs_model_tensor_buft_override *tensor_buft_overrides;

  int32_t n_gpu_layers;             // number of layers to store in VRAM, a negative value means all layers
  enum mgs_split_mode split_mode; // how to split the model across multiple GPUs
  enum mgs_load_mode load_mode;   // how to load the model

  // the GPU that is used for the entire model when split_mode is MGS_SPLIT_MODE_NONE
  int32_t main_gpu;

  // proportion of the model (layers or rows) to offload to each GPU, size: mgs_max_devices()
  const float *tensor_split;

  // Called with a progress value between 0.0 and 1.0. Pass NULL to disable.
  // If the provided progress_callback returns true, model loading continues.
  // If it returns false, model loading is immediately aborted.
  mgs_progress_callback progress_callback;

  // context pointer passed to the progress callback
  void *progress_callback_user_data;

  // override key-value pairs of the model meta data
  const struct mgs_model_kv_override *kv_overrides;

  // Keep the booleans together to avoid misalignment during copy-by-value.
  bool vocab_only;      // only load the vocabulary, no weights
  bool check_tensors;   // validate model tensor data
  bool use_extra_bufts; // use extra buffer types (used for weight repacking)
  bool no_host;         // bypass host buffer allowing extra buffers to be used
  bool no_alloc;        // only load metadata and simulate memory allocations
  bool load_mtp;        // whether to load MTP layers
};

struct mgs_sampler_seq_config {
  mgs_seq_id seq_id;
  struct mgs_sampler *sampler;
};

// NOTE: changing the default values of parameters marked as [EXPERIMENTAL] may cause crashes or incorrect results in
// certain configurations
struct mgs_context_params {
  uint32_t n_ctx;          // text context, 0 = from model
  uint32_t n_batch;        // logical maximum batch size that can be submitted to mgs_decode
  uint32_t n_ubatch;       // physical maximum batch size
  uint32_t n_seq_max;      // max number of sequences (i.e. distinct states for recurrent models)
  uint32_t n_rs_seq;       // number of recurrent-state snapshots per seq for rollback (0 = no rollback) [EXPERIMENTAL]
  uint32_t n_outputs_max;  // max outputs in a ubatch (0 = n_batch)
  int32_t n_threads;       // number of threads to use for generation
  int32_t n_threads_batch; // number of threads to use for batch processing

  enum mgs_context_type ctx_type;               // set the context type (e.g. MTP)
  enum mgs_rope_scaling_type rope_scaling_type; // RoPE scaling type, from `enum mgs_rope_scaling_type`
  enum mgs_pooling_type pooling_type;           // whether to pool (sum) embedding results by sequence id
  enum mgs_attention_type attention_type;       // attention type to use for embeddings
  enum mgs_flash_attn_type flash_attn_type;     // when to enable Flash Attention

  float rope_freq_base;   // RoPE base frequency, 0 = from model
  float rope_freq_scale;  // RoPE frequency scaling factor, 0 = from model
  float yarn_ext_factor;  // YaRN extrapolation mix factor, negative = from model
  float yarn_attn_factor; // YaRN magnitude scaling factor
  float yarn_beta_fast;   // YaRN low correction dim
  float yarn_beta_slow;   // YaRN high correction dim
  uint32_t yarn_orig_ctx; // YaRN original context size

  ggml_backend_sched_eval_callback cb_eval;
  void *cb_eval_user_data;

  enum ggml_type type_k; // data type for K cache [EXPERIMENTAL]
  enum ggml_type type_v; // data type for V cache [EXPERIMENTAL]

  // Abort callback
  // if it returns true, execution of mgs_decode() will be aborted
  // currently works only with CPU execution
  ggml_abort_callback abort_callback;
  void *abort_callback_data;

  // Keep the booleans together and at the end of the struct to avoid misalignment during copy-by-value.
  bool embeddings;  // if true, extract embeddings (together with logits)
  bool offload_kqv; // offload the KQV ops (including the KV cache) to GPU
  bool no_perf;     // measure performance timings
  bool op_offload;  // offload host tensor operations to device
  bool swa_full;   // use full-size SWA cache
                   // NOTE: setting to false when n_seq_max > 1 can cause bad performance in some cases
  bool kv_unified; // use a unified buffer across the input sequences when computing the attention
                   // try to disable when n_seq_max > 1 for improved performance when the sequences do not share a large
                   // prefix

  // [EXPERIMENTAL]
  // backend sampler chain configuration (make sure the caller keeps the sampler chains alive)
  // note: the samplers must be sampler chains (i.e. use mgs_sampler_chain_init)
  struct mgs_sampler_seq_config *samplers;
  size_t n_samplers;

  // a source/target/parent context
  // can be utilized in various ways, for example by sharing results or mgs_memory between 2 contexts
  struct mgs_context *ctx_other;
};

struct mgs_model_tensor_override {
  const char *pattern;
  enum ggml_type type;
};

struct mgs_model_imatrix_data {
  const char *name;
  const float *data;
  size_t size;
};

// model quantization parameters
typedef struct mgs_model_quantize_params {
  int32_t nthread; // number of threads to use for quantizing, if <=0 will use std::thread::hardware_concurrency()
  enum mgs_ftype ftype;              // quantize to this mgs_ftype
  enum ggml_type output_tensor_type;   // output tensor type
  enum ggml_type token_embedding_type; // token embeddings tensor type
  bool allow_requantize;               // allow quantizing non-f32/f16 tensors
  bool quantize_output_tensor;         // quantize output.weight
  bool only_copy;  // only copy tensors - ftype, allow_requantize and quantize_output_tensor are ignored
  bool pure;       // quantize all tensors to the default type
  bool keep_split; // quantize to the same number of shards
  bool dry_run;    // calculate and show the final quantization size without performing quantization
  const struct mgs_model_imatrix_data *imatrix;         // pointer to importance matrix data
  const struct mgs_model_kv_override *kv_overrides;     // pointer to kv overrides
  const struct mgs_model_tensor_override *tt_overrides; // pointer to tensor overrides
  const int32_t *prune_layers;                            // pointer to layer indices to prune
} mgs_model_quantize_params;

typedef struct mgs_logit_bias {
  mgs_token token;
  float bias;
} mgs_logit_bias;

typedef struct mgs_sampler_chain_params {
  bool no_perf; // whether to measure performance timings
} mgs_sampler_chain_params;

// lora adapter
struct mgs_adapter_lora;

// Helpers for getting default parameters
// TODO: update API to start accepting pointers to params structs
MGS_API struct mgs_model_params mgs_model_default_params(void);
MGS_API struct mgs_context_params mgs_context_default_params(void);
MGS_API struct mgs_sampler_chain_params mgs_sampler_chain_default_params(void);
MGS_API struct mgs_model_quantize_params mgs_model_quantize_default_params(void);

// Initialize the mgs + ggml backend
// If numa is true, use NUMA optimizations
// Call once at the start of the program
MGS_API void mgs_backend_init(void);

// Call once at the end of the program - currently only used for MPI
MGS_API void mgs_backend_free(void);

// optional:
MGS_API void mgs_numa_init(enum ggml_numa_strategy numa);

// Optional: an auto threadpool gets created in ggml if not passed explicitly
MGS_API void mgs_attach_threadpool(struct mgs_context *ctx, ggml_threadpool_t threadpool,
                                       ggml_threadpool_t threadpool_batch);

MGS_API void mgs_detach_threadpool(struct mgs_context *ctx);

typedef void (*mgs_model_set_tensor_data_t)(struct ggml_tensor *tensor, void *userdata);

// Create a new model from GGUF metadata as well as a function to set the tensor data
//   - tensors are created as GGML_TYPE_F32 by default,
//     override by adding a tensor with the same name but a different name to the context
MGS_API struct mgs_model *
mgs_model_init_from_user(struct gguf_context *metadata,
                           mgs_model_set_tensor_data_t set_tensor_data, // function to initialize tensor data with
                           void *set_tensor_data_ud,                      // userdata for function
                           struct mgs_model_params params);

// Load a model from a file
// If the file is split into multiple parts, the file name must follow this pattern: <name>-%05d-of-%05d.gguf
// If the split file name does not follow this pattern, use mgs_model_load_from_splits
MGS_API struct mgs_model *mgs_model_load_from_file(const char *path_model, struct mgs_model_params params);

// Load a model from an open FILE pointer
MGS_API struct mgs_model *mgs_model_load_from_file_ptr(FILE *file, struct mgs_model_params params);

// Load a model from multiple splits (support custom naming scheme)
// The paths must be in the correct order
MGS_API struct mgs_model *mgs_model_load_from_splits(const char **paths, size_t n_paths,
                                                           struct mgs_model_params params);

MGS_API void mgs_model_save_to_file(const struct mgs_model *model, const char *path_model);

MGS_API void mgs_model_free(struct mgs_model *model);

MGS_API struct mgs_context *mgs_init_from_model(struct mgs_model *model, struct mgs_context_params params);

// Frees all allocated memory
MGS_API void mgs_free(struct mgs_context *ctx);

MGS_API int64_t mgs_time_us(void);

MGS_API size_t mgs_max_devices(void);
MGS_API size_t mgs_max_parallel_sequences(void);
MGS_API size_t mgs_max_tensor_buft_overrides(void);

MGS_API bool mgs_supports_mmap(void);
MGS_API bool mgs_supports_mlock(void);
MGS_API bool mgs_supports_gpu_offload(void);
MGS_API bool mgs_supports_rpc(void);

// NOTE: After creating a mgs_context, it is recommended to query the actual values using these functions
//       In some cases the requested values via mgs_context_params may differ from the actual values used by the
//       context
MGS_API uint32_t mgs_n_ctx(const struct mgs_context *ctx);
MGS_API uint32_t mgs_n_ctx_seq(const struct mgs_context *ctx);
MGS_API uint32_t mgs_n_batch(const struct mgs_context *ctx);
MGS_API uint32_t mgs_n_ubatch(const struct mgs_context *ctx);
MGS_API uint32_t mgs_n_seq_max(const struct mgs_context *ctx);
MGS_API uint32_t mgs_n_rs_seq(const struct mgs_context *ctx);

MGS_API const struct mgs_model *mgs_get_model(const struct mgs_context *ctx);
MGS_API mgs_memory_t mgs_get_memory(const struct mgs_context *ctx);
MGS_API enum mgs_pooling_type
mgs_pooling_type(const struct mgs_context *ctx); // TODO: rename to mgs_get_pooling_type

MGS_API const struct mgs_vocab *mgs_model_get_vocab(const struct mgs_model *model);
MGS_API enum mgs_rope_type mgs_model_rope_type(const struct mgs_model *model);

MGS_API int32_t mgs_model_n_ctx_train(const struct mgs_model *model);
MGS_API int32_t mgs_model_n_embd(const struct mgs_model *model);
MGS_API int32_t mgs_model_n_embd_inp(const struct mgs_model *model);
MGS_API int32_t mgs_model_n_embd_out(const struct mgs_model *model);
MGS_API int32_t mgs_model_n_layer(const struct mgs_model *model);
MGS_API int32_t mgs_model_n_layer_nextn(const struct mgs_model *model);
MGS_API int32_t mgs_model_n_head(const struct mgs_model *model);
MGS_API int32_t mgs_model_n_head_kv(const struct mgs_model *model);
MGS_API int32_t mgs_model_n_swa(const struct mgs_model *model);

// Get the model's RoPE frequency scaling factor
MGS_API float mgs_model_rope_freq_scale_train(const struct mgs_model *model);

// Returns the number of classifier outputs (only valid for classifier models)
// Undefined behavior for non-classifier models
MGS_API uint32_t mgs_model_n_cls_out(const struct mgs_model *model);

// Returns label of classifier output by index (<n_cls_out). Returns nullptr if no label provided
MGS_API const char *mgs_model_cls_label(const struct mgs_model *model, uint32_t i);

MGS_API enum mgs_vocab_type mgs_vocab_type(const struct mgs_vocab *vocab);

MGS_API int32_t mgs_vocab_n_tokens(const struct mgs_vocab *vocab);

// Functions to access the model's GGUF metadata scalar values
// - The functions return the length of the string on success, or -1 on failure
// - The output string is always null-terminated and cleared on failure
// - When retrieving a string, an extra byte must be allocated to account for the null terminator
// - GGUF array values are not supported by these functions

// Get metadata value as a string by key name
MGS_API int32_t mgs_model_meta_val_str(const struct mgs_model *model, const char *key, char *buf,
                                           size_t buf_size);

// Get the number of metadata key/value pairs
MGS_API int32_t mgs_model_meta_count(const struct mgs_model *model);

// Get sampling metadata key name. Returns nullptr if the key is invalid
MGS_API const char *mgs_model_meta_key_str(enum mgs_model_meta_key key);

// Get metadata key name by index
MGS_API int32_t mgs_model_meta_key_by_index(const struct mgs_model *model, int32_t i, char *buf, size_t buf_size);

// Get metadata value as a string by index
MGS_API int32_t mgs_model_meta_val_str_by_index(const struct mgs_model *model, int32_t i, char *buf,
                                                    size_t buf_size);

// Get a string describing the model type
MGS_API int32_t mgs_model_desc(const struct mgs_model *model, char *buf, size_t buf_size);

// Get the model file type (quantization), e.g. MGS_FTYPE_MOSTLY_Q8_0
MGS_API enum mgs_ftype mgs_model_ftype(const struct mgs_model *model);

// Returns the total size of all the tensors in the model in bytes
MGS_API uint64_t mgs_model_size(const struct mgs_model *model);

// Get the default chat template. Returns nullptr if not available
// If name is NULL, returns the default chat template
MGS_API const char *mgs_model_chat_template(const struct mgs_model *model, const char *name);

// Returns the total number of parameters in the model
MGS_API uint64_t mgs_model_n_params(const struct mgs_model *model);

// Returns true if the model contains an encoder that requires mgs_encode() call
MGS_API bool mgs_model_has_encoder(const struct mgs_model *model);

// Returns true if the model contains a decoder that requires mgs_decode() call
MGS_API bool mgs_model_has_decoder(const struct mgs_model *model);

// For encoder-decoder models, this function returns id of the token that must be provided
// to the decoder to start generating output sequence. For other models, it returns -1.
MGS_API mgs_token mgs_model_decoder_start_token(const struct mgs_model *model);

// Returns true if the model is recurrent (like Mamba, RWKV, etc.)
MGS_API bool mgs_model_is_recurrent(const struct mgs_model *model);

// Returns true if the model is hybrid (like Jamba, Granite, etc.)
MGS_API bool mgs_model_is_hybrid(const struct mgs_model *model);

// Returns true if the model is diffusion-based (like LLaDA, Dream, etc.)
MGS_API bool mgs_model_is_diffusion(const struct mgs_model *model);

// Returns 0 on success
MGS_API uint32_t mgs_model_quantize(const char *fname_inp, const char *fname_out,
                                        const mgs_model_quantize_params *params);

//
// Adapters
//

// Load a LoRA adapter from file
// The adapter is valid as long as the associated model is not freed
MGS_API struct mgs_adapter_lora *mgs_adapter_lora_init(struct mgs_model *model, const char *path_lora);

// Functions to access the adapter's GGUF metadata scalar values
// - The functions return the length of the string on success, or -1 on failure
// - The output string is always null-terminated and cleared on failure
// - When retrieving a string, an extra byte must be allocated to account for the null terminator
// - GGUF array values are not supported by these functions

// Get metadata value as a string by key name
MGS_API int32_t mgs_adapter_meta_val_str(const struct mgs_adapter_lora *adapter, const char *key, char *buf,
                                             size_t buf_size);

// Get the number of metadata key/value pairs
MGS_API int32_t mgs_adapter_meta_count(const struct mgs_adapter_lora *adapter);

// Get metadata key name by index
MGS_API int32_t mgs_adapter_meta_key_by_index(const struct mgs_adapter_lora *adapter, int32_t i, char *buf,
                                                  size_t buf_size);

// Get metadata value as a string by index
MGS_API int32_t mgs_adapter_meta_val_str_by_index(const struct mgs_adapter_lora *adapter, int32_t i, char *buf,
                                                      size_t buf_size);

// Manually free a LoRA adapter
// NOTE: loaded adapters that are not manually freed will be freed when the associated model is deleted
MGS_API void mgs_adapter_lora_free(struct mgs_adapter_lora *adapter);

// Get the invocation tokens if the current lora is an alora
MGS_API uint64_t mgs_adapter_get_alora_n_invocation_tokens(const struct mgs_adapter_lora *adapter);
MGS_API const mgs_token *mgs_adapter_get_alora_invocation_tokens(const struct mgs_adapter_lora *adapter);

// The following functions operate on a mgs_context, hence the naming: mgs_verb_...

// Set LoRa adapters on the context. Will only modify if the adapters currently in context are different.
MGS_API int32_t mgs_set_adapters_lora(struct mgs_context *ctx, struct mgs_adapter_lora **adapters,
                                          size_t n_adapters, float *scales);

// Apply a loaded control vector to a mgs_context, or if data is NULL, clear
// the currently loaded vector.
// n_embd should be the size of a single layer's control, and data should point
// to an n_embd x n_layers buffer starting from layer 1.
// il_start and il_end are the layer range the vector should apply to (both inclusive)
// See mgs_control_vector_load in common to load a control vector.
MGS_API int32_t mgs_set_adapter_cvec(struct mgs_context *ctx, const float *data, size_t len, int32_t n_embd,
                                         int32_t il_start, int32_t il_end);

//
// Memory
//

// Clear the memory contents
// If data == true, the data buffers will also be cleared together with the metadata
MGS_API void mgs_memory_clear(mgs_memory_t mem, bool data);

// Removes all tokens that belong to the specified sequence and have positions in [p0, p1)
// Returns false if a partial sequence cannot be removed. Removing a whole sequence never fails
// seq_id < 0 : match any sequence
// p0 < 0     : [0,  p1]
// p1 < 0     : [p0, inf)
MGS_API bool mgs_memory_seq_rm(mgs_memory_t mem, mgs_seq_id seq_id, mgs_pos p0, mgs_pos p1);

// Copy all tokens that belong to the specified sequence to another sequence
// p0 < 0 : [0,  p1]
// p1 < 0 : [p0, inf)
MGS_API void mgs_memory_seq_cp(mgs_memory_t mem, mgs_seq_id seq_id_src, mgs_seq_id seq_id_dst, mgs_pos p0,
                                   mgs_pos p1);

// Removes all tokens that do not belong to the specified sequence
MGS_API void mgs_memory_seq_keep(mgs_memory_t mem, mgs_seq_id seq_id);

// Adds relative position "delta" to all tokens that belong to the specified sequence and have positions in [p0, p1)
// p0 < 0 : [0,  p1]
// p1 < 0 : [p0, inf)
MGS_API void mgs_memory_seq_add(mgs_memory_t mem, mgs_seq_id seq_id, mgs_pos p0, mgs_pos p1,
                                    mgs_pos delta);

// Integer division of the positions by factor of `d > 1`
// p0 < 0 : [0,  p1]
// p1 < 0 : [p0, inf)
MGS_API void mgs_memory_seq_div(mgs_memory_t mem, mgs_seq_id seq_id, mgs_pos p0, mgs_pos p1, int d);

// Returns the smallest position present in the memory for the specified sequence
// This is typically non-zero only for SWA caches
// Note that all positions in the range [pos_min, pos_max] are guaranteed to be present in the memory
// Return -1 if the sequence is empty
MGS_API mgs_pos mgs_memory_seq_pos_min(mgs_memory_t mem, mgs_seq_id seq_id);

// Returns the largest position present in the memory for the specified sequence
// Note that all positions in the range [pos_min, pos_max] are guaranteed to be present in the memory
// Return -1 if the sequence is empty
MGS_API mgs_pos mgs_memory_seq_pos_max(mgs_memory_t mem, mgs_seq_id seq_id);

// Check if the memory supports shifting
MGS_API bool mgs_memory_can_shift(mgs_memory_t mem);

//
// State / sessions
//

// Returns the *actual* size in bytes of the state
// (logits, embedding and memory)
// Only use when saving the state, not when restoring it, otherwise the size may be too small.
MGS_API size_t mgs_state_get_size(struct mgs_context *ctx);

// Copies the state to the specified destination address.
// Destination needs to have allocated enough memory.
// Returns the number of bytes copied
MGS_API size_t mgs_state_get_data(struct mgs_context *ctx, uint8_t *dst, size_t size);

// Set the state reading from the specified address
// Returns the number of bytes read
MGS_API size_t mgs_state_set_data(struct mgs_context *ctx, const uint8_t *src, size_t size);

// Save/load session file
MGS_API bool mgs_state_load_file(struct mgs_context *ctx, const char *path_session, mgs_token *tokens_out,
                                     size_t n_token_capacity, size_t *n_token_count_out);

MGS_API bool mgs_state_save_file(struct mgs_context *ctx, const char *path_session, const mgs_token *tokens,
                                     size_t n_token_count);

// Get the exact size needed to copy the state of a single sequence
MGS_API size_t mgs_state_seq_get_size(struct mgs_context *ctx, mgs_seq_id seq_id);

// Copy the state of a single sequence into the specified buffer
MGS_API size_t mgs_state_seq_get_data(struct mgs_context *ctx, uint8_t *dst, size_t size, mgs_seq_id seq_id);

// Copy the sequence data (originally copied with `mgs_state_seq_get_data`) into the specified sequence
// Returns:
//  - Positive: Ok
//  - Zero: Failed to load
MGS_API size_t mgs_state_seq_set_data(struct mgs_context *ctx, const uint8_t *src, size_t size,
                                          mgs_seq_id dest_seq_id);

MGS_API size_t mgs_state_seq_save_file(struct mgs_context *ctx, const char *filepath, mgs_seq_id seq_id,
                                           const mgs_token *tokens, size_t n_token_count);

MGS_API size_t mgs_state_seq_load_file(struct mgs_context *ctx, const char *filepath, mgs_seq_id dest_seq_id,
                                           mgs_token *tokens_out, size_t n_token_capacity, size_t *n_token_count_out);

#define MGS_STATE_SEQ_FLAGS_NONE 0

// work only with partial states, such as SWA KV cache or recurrent cache (e.g. Mamba)
#define MGS_STATE_SEQ_FLAGS_PARTIAL_ONLY 1

// Keeps the tensor data on device buffers (i.e. not accessible in host memory, but faster save/load).
// Getting the state for a seq_id with this flag invalidates all prior states gotten for that seq_id with this flag.
#define MGS_STATE_SEQ_FLAGS_ON_DEVICE 2

typedef uint32_t mgs_state_seq_flags;

MGS_API size_t mgs_state_seq_get_size_ext(struct mgs_context *ctx, mgs_seq_id seq_id,
                                              mgs_state_seq_flags flags);

MGS_API size_t mgs_state_seq_get_data_ext(struct mgs_context *ctx, uint8_t *dst, size_t size, mgs_seq_id seq_id,
                                              mgs_state_seq_flags flags);

MGS_API size_t mgs_state_seq_set_data_ext(struct mgs_context *ctx, const uint8_t *src, size_t size,
                                              mgs_seq_id dest_seq_id, mgs_state_seq_flags flags);

//
// Decoding
//

// Return batch for single sequence of tokens
// The sequence ID will be fixed to 0
// The position of the tokens will be tracked automatically by mgs_decode
//
// NOTE: this is a helper function to facilitate transition to the new batch API - avoid using it
//
MGS_API struct mgs_batch mgs_batch_get_one(mgs_token *tokens, int32_t n_tokens);

// Allocates a batch of tokens on the heap that can hold a maximum of n_tokens
// Each token can be assigned up to n_seq_max sequence ids
// The batch has to be freed with mgs_batch_free()
// If embd != 0, mgs_batch.embd will be allocated with size of n_tokens * embd * sizeof(float)
// Otherwise, mgs_batch.token will be allocated to store n_tokens mgs_token
// The rest of the mgs_batch members are allocated with size n_tokens
// All members are left uninitialized
MGS_API struct mgs_batch mgs_batch_init(int32_t n_tokens, int32_t embd, int32_t n_seq_max);

// Frees a batch of tokens allocated with mgs_batch_init()
MGS_API void mgs_batch_free(struct mgs_batch batch);

// Process a batch of tokens.
// In contrast to mgs_decode() - this call does not use KV cache.
// For encode-decoder contexts, processes the batch using the encoder.
// Can store the encoder output internally for later use by the decoder's cross-attention layers.
//   0 - success
// < 0 - error. the memory state is restored to the state before this call
MGS_API int32_t mgs_encode(struct mgs_context *ctx, struct mgs_batch batch);

// Process a batch of tokens.
// Requires the context to have a memory.
// For encode-decoder contexts, processes the batch using the decoder.
// Positive return values does not mean a fatal error, but rather a warning.
// Upon fatal-error or abort, the ubatches that managed to be been processed will remain in the memory state of the
// context
//   To handle this correctly, query the memory state using mgs_memory_seq_pos_min() and mgs_memory_seq_pos_max()
// Upon other return values, the memory state is restored to the state before this call
//    0 - success
//    1 - could not find a KV slot for the batch (try reducing the size of the batch or increase the context)
//    2 - aborted     (processed ubatches will remain in the context's memory)
//   -1 - invalid input batch
// < -1 - fatal error (processed ubatches will remain in the context's memory)
MGS_API int32_t mgs_decode(struct mgs_context *ctx, struct mgs_batch batch);

// Set the number of threads used for decoding
// n_threads is the number of threads used for generation (single token)
// n_threads_batch is the number of threads used for prompt and batch processing (multiple tokens)
MGS_API void mgs_set_n_threads(struct mgs_context *ctx, int32_t n_threads, int32_t n_threads_batch);

// Get the number of threads used for generation of a single token.
MGS_API int32_t mgs_n_threads(struct mgs_context *ctx);

// Get the number of threads used for prompt and batch processing (multiple token).
MGS_API int32_t mgs_n_threads_batch(struct mgs_context *ctx);

// Set whether the context outputs embeddings or not
// TODO: rename to avoid confusion with mgs_get_embeddings()
MGS_API void mgs_set_embeddings(struct mgs_context *ctx, bool embeddings);

// Set whether to use causal attention or not
// If set to true, the model will only attend to the past tokens
MGS_API void mgs_set_causal_attn(struct mgs_context *ctx, bool causal_attn);

// Set abort callback
MGS_API void mgs_set_abort_callback(struct mgs_context *ctx, ggml_abort_callback abort_callback,
                                        void *abort_callback_data);

// Wait until all computations are finished
// This is automatically done when using one of the functions below to obtain the computation results
// and is not necessary to call it explicitly in most cases
MGS_API void mgs_synchronize(struct mgs_context *ctx);

// Token logits obtained from the last call to mgs_decode()
// The logits for which mgs_batch.logits[i] != 0 are stored contiguously
// in the order they have appeared in the batch.
// Rows: number of tokens for which mgs_batch.logits[i] != 0
// Cols: n_vocab
// TODO: deprecate in favor of mgs_get_logits_ith()
MGS_API float *mgs_get_logits(struct mgs_context *ctx);

// Logits for the ith token. For positive indices, Equivalent to:
// mgs_get_logits(ctx) + ctx->lOutputIds[i]*n_vocab
// Negative indices can be used to access logits in reverse order, -1 is the last logit.
// returns NULL for invalid ids.
MGS_API float *mgs_get_logits_ith(struct mgs_context *ctx, int32_t i);

// Get all output token embeddings.
// when pooling_type == MGS_POOLING_TYPE_NONE or when using a generative model,
// the embeddings for which mgs_batch.logits[i] != 0 are stored contiguously
// in the order they have appeared in the batch.
// shape: [n_outputs*n_embd]
// Otherwise, returns NULL.
// TODO: deprecate in favor of mgs_get_embeddings_ith()
MGS_API float *mgs_get_embeddings(struct mgs_context *ctx);

// Get the embeddings for the ith token. For positive indices, Equivalent to:
// mgs_get_embeddings(ctx) + ctx->lOutputIds[i]*n_embd
// Negative indices can be used to access embeddings in reverse order, -1 is the last embedding.
// shape: [n_embd] (1-dimensional)
// returns NULL for invalid ids.
MGS_API float *mgs_get_embeddings_ith(struct mgs_context *ctx, int32_t i);

// Get the embeddings for a sequence id
// Returns NULL if pooling_type is MGS_POOLING_TYPE_NONE
// when pooling_type == MGS_POOLING_TYPE_RANK, returns float[n_cls_out] with the rank(s) of the sequence
// otherwise: float[n_embd] (1-dimensional)
MGS_API float *mgs_get_embeddings_seq(struct mgs_context *ctx, mgs_seq_id seq_id);

//
// backend sampling API [EXPERIMENTAL]
// note: use only if the mgs_context was created with at least one mgs_sampler_seq_config
//

// Get the backend sampled token for the ith token.
// Returns MGS_TOKEN_NULL if no token was sampled.
MGS_API mgs_token mgs_get_sampled_token_ith(struct mgs_context *ctx, int32_t i);

// Get the backend sampled probabilities for the ith token
// The index matches mgs_get_sampled_token_ith().
// Returns NULL if no probabilities were generated.
MGS_API float *mgs_get_sampled_probs_ith(struct mgs_context *ctx, int32_t i);
MGS_API uint32_t mgs_get_sampled_probs_count_ith(struct mgs_context *ctx, int32_t i);

// Get the backend sampled logits for the ith token
// Returns NULL if no logits were sampled.
MGS_API float *mgs_get_sampled_logits_ith(struct mgs_context *ctx, int32_t i);
MGS_API uint32_t mgs_get_sampled_logits_count_ith(struct mgs_context *ctx, int32_t i);

// Get the backend sampled candidates (token ids) for the ith token
// These are needed to map probability/logit indices to vocab token ids.
// Returns NULL if no candidates were sampled.
MGS_API mgs_token *mgs_get_sampled_candidates_ith(struct mgs_context *ctx, int32_t i);
MGS_API uint32_t mgs_get_sampled_candidates_count_ith(struct mgs_context *ctx, int32_t i);

//
// Vocab
//

MGS_API const char *mgs_vocab_get_text(const struct mgs_vocab *vocab, mgs_token token);

MGS_API float mgs_vocab_get_score(const struct mgs_vocab *vocab, mgs_token token);

MGS_API enum mgs_token_attr mgs_vocab_get_attr(const struct mgs_vocab *vocab, mgs_token token);

// Check if the token is supposed to end generation (end-of-generation, eg. EOS, EOT, etc.)
MGS_API bool mgs_vocab_is_eog(const struct mgs_vocab *vocab, mgs_token token);

// Identify if Token Id is a control token or a render-able token
MGS_API bool mgs_vocab_is_control(const struct mgs_vocab *vocab, mgs_token token);

// Special tokens
MGS_API mgs_token mgs_vocab_bos(const struct mgs_vocab *vocab);  // beginning-of-sentence
MGS_API mgs_token mgs_vocab_eos(const struct mgs_vocab *vocab);  // end-of-sentence
MGS_API mgs_token mgs_vocab_eot(const struct mgs_vocab *vocab);  // end-of-turn
MGS_API mgs_token mgs_vocab_sep(const struct mgs_vocab *vocab);  // sentence separator
MGS_API mgs_token mgs_vocab_nl(const struct mgs_vocab *vocab);   // next-line
MGS_API mgs_token mgs_vocab_pad(const struct mgs_vocab *vocab);  // padding
MGS_API mgs_token mgs_vocab_mask(const struct mgs_vocab *vocab); // mask

MGS_API bool mgs_vocab_get_add_bos(const struct mgs_vocab *vocab);
MGS_API bool mgs_vocab_get_add_eos(const struct mgs_vocab *vocab);
MGS_API bool mgs_vocab_get_add_sep(const struct mgs_vocab *vocab);

// model-specific suppress tokens (gguf key: tokenizer.ggml.suppress_tokens)
MGS_API const mgs_token *mgs_vocab_get_suppress_tokens(const struct mgs_vocab *vocab,
                                                             int32_t *n_suppress_tokens);

MGS_API mgs_token mgs_vocab_fim_pre(const struct mgs_vocab *vocab);
MGS_API mgs_token mgs_vocab_fim_suf(const struct mgs_vocab *vocab);
MGS_API mgs_token mgs_vocab_fim_mid(const struct mgs_vocab *vocab);
MGS_API mgs_token mgs_vocab_fim_pad(const struct mgs_vocab *vocab);
MGS_API mgs_token mgs_vocab_fim_rep(const struct mgs_vocab *vocab);
MGS_API mgs_token mgs_vocab_fim_sep(const struct mgs_vocab *vocab);


//
// Tokenization
//
// The API is thread-safe.
//

/// @details Convert the provided text into tokens.
/// @param tokens The tokens pointer must be large enough to hold the resulting tokens.
/// @return Returns the number of tokens on success, no more than n_tokens_max
/// @return Returns a negative number on failure - the number of tokens that would have been returned
/// @return Returns INT32_MIN on overflow (e.g., tokenization result size exceeds int32_t limit)
/// @param add_special Allow to add BOS and EOS tokens if model is configured to do so.
/// @param parse_special Allow tokenizing special and/or control tokens which otherwise are not exposed and treated
///                      as plaintext. Does not insert a leading space.
MGS_API int32_t mgs_tokenize(const struct mgs_vocab *vocab, const char *text, int32_t text_len,
                                 mgs_token *tokens, int32_t n_tokens_max, bool add_special, bool parse_special);

// Token Id -> Piece.
// Uses the vocabulary in the provided context.
// Does not write null terminator to the buffer.
// User can skip up to 'lstrip' leading spaces before copying (useful when encoding/decoding multiple tokens with
// 'add_space_prefix')
// @param special If true, special tokens are rendered in the output.
MGS_API int32_t mgs_token_to_piece(const struct mgs_vocab *vocab, mgs_token token, char *buf, int32_t length,
                                       int32_t lstrip, bool special);

/// @details Convert the provided tokens into text (inverse of mgs_tokenize()).
/// @param text The char pointer must be large enough to hold the resulting text.
/// @return Returns the number of chars/bytes on success, no more than text_len_max.
/// @return Returns a negative number on failure - the number of chars/bytes that would have been returned.
/// @param remove_special Allow to remove BOS and EOS tokens if model is configured to do so.
/// @param unparse_special If true, special tokens are rendered in the output.
MGS_API int32_t mgs_detokenize(const struct mgs_vocab *vocab, const mgs_token *tokens, int32_t n_tokens,
                                   char *text, int32_t text_len_max, bool remove_special, bool unparse_special);

//
// Sampling API
//
// Sample usage:
//
//    // prepare the sampling chain at the start
//    auto vSparams = mgs_sampler_chain_default_params();
//
//    mgs_sampler * vSmpl = mgs_sampler_chain_init(vSparams);
//
//    mgs_sampler_chain_add(vSmpl, mgs_sampler_init_top_k(50));
//    mgs_sampler_chain_add(vSmpl, mgs_sampler_init_top_p(0.9, 1));
//    mgs_sampler_chain_add(vSmpl, mgs_sampler_init_temp (0.8));
//
//    // typically, the chain should end with a sampler such as "greedy", "dist" or "mirostat"
//    // this sampler will be responsible to select the actual token
//    mgs_sampler_chain_add(vSmpl, mgs_sampler_init_dist(vSeed));
//
//    ...
//
//    // decoding loop:
//    while (...) {
//        ...
//
//        mgs_decode(vCtx, vBatch);
//
//        // sample from the logits of the last token in the batch
//        const mgs_token cId = mgs_sampler_sample(vSmpl, vCtx, -1);
//
//        ...
//    }
//
//    mgs_sampler_free(vSmpl);
//

typedef void *mgs_sampler_context_t;

struct mgs_sampler_data {
  struct ggml_tensor *logits;
  struct ggml_tensor *probs;
  struct ggml_tensor *sampled;
  struct ggml_tensor *candidates;
};

// user code can implement the interface below in order to create custom mgs_sampler
struct mgs_sampler_i {
  const char *(*name)(const struct mgs_sampler *smpl);                    // can be NULL
  void (*accept)(struct mgs_sampler *smpl, mgs_token token);            // can be NULL
  void (*apply)(struct mgs_sampler *smpl, mgs_token_data_array *cur_p); // required
  void (*reset)(struct mgs_sampler *smpl);                                // can be NULL
  struct mgs_sampler *(*clone)(const struct mgs_sampler *smpl);         // can be NULL if ctx is NULL
  void (*free)(struct mgs_sampler *smpl);                                 // can be NULL if ctx is NULL

  // [EXPERIMENTAL]
  // backend sampling interface:

  // return true if the backend supports all ops needed by the sampler
  // note: call once per sampler
  bool (*backend_init)(struct mgs_sampler *smpl, ggml_backend_buffer_type_t buft);

  // call after .backend_apply()
  void (*backend_accept)(struct mgs_sampler *smpl, struct ggml_context *ctx, struct ggml_cgraph *gf,
                         struct ggml_tensor *selected_token);

  // call after .backend_init()
  void (*backend_apply)(struct mgs_sampler *smpl, struct ggml_context *ctx, struct ggml_cgraph *gf,
                        struct mgs_sampler_data *data);

  // called before graph execution to set inputs for the current ubatch
  void (*backend_set_input)(struct mgs_sampler *smpl);
};

struct mgs_sampler {
  struct mgs_sampler_i *iface;

  mgs_sampler_context_t ctx;
};

// [EXPERIMENTAL]
// attach a sampler to the context
// note: prefer initializing the context with mgs_context_params.samplers when possible
MGS_API bool mgs_set_sampler(struct mgs_context *ctx, mgs_seq_id seq_id, struct mgs_sampler *smpl);

// mirror of mgs_sampler_i:
MGS_API struct mgs_sampler *mgs_sampler_init(struct mgs_sampler_i *iface, mgs_sampler_context_t ctx);
MGS_API const char *mgs_sampler_name(const struct mgs_sampler *smpl);
MGS_API void mgs_sampler_accept(struct mgs_sampler *smpl, mgs_token token);
MGS_API void mgs_sampler_apply(struct mgs_sampler *smpl, mgs_token_data_array *cur_p);
MGS_API void mgs_sampler_reset(struct mgs_sampler *smpl);
MGS_API struct mgs_sampler *mgs_sampler_clone(const struct mgs_sampler *smpl);
// important: do not free if the sampler has been added to a mgs_sampler_chain (via mgs_sampler_chain_add)
MGS_API void mgs_sampler_free(struct mgs_sampler *smpl);

// mgs_sampler_chain
// a type of mgs_sampler that can chain multiple samplers one after another

MGS_API struct mgs_sampler *mgs_sampler_chain_init(struct mgs_sampler_chain_params params);

// important: takes ownership of the sampler object and will free it when mgs_sampler_free is called
MGS_API void mgs_sampler_chain_add(struct mgs_sampler *chain, struct mgs_sampler *smpl);

// return NULL if:
//   - the sampler is NULL
//   - the sampler is not a mgs_sampler_chain
//   - the index is out of bounds, unless i == -1
//   - if i == -1, returns the chain itself (can be used to check if the sampler is a chain)
MGS_API struct mgs_sampler *mgs_sampler_chain_get(struct mgs_sampler *chain, int32_t i);

// the total number of samplers in the chain
MGS_API int mgs_sampler_chain_n(const struct mgs_sampler *chain);

// after removing a sampler, the chain will no longer own it, and it will not be freed when the chain is freed
MGS_API struct mgs_sampler *mgs_sampler_chain_remove(struct mgs_sampler *chain, int32_t i);

// available samplers:

MGS_API struct mgs_sampler *mgs_sampler_init_greedy(void);

/// seed == MGS_DEFAULT_SEED to use a random seed.
MGS_API struct mgs_sampler *mgs_sampler_init_dist(uint32_t seed);

/// @details Top-K sampling described in academic paper "The Curious Case of Neural Text Degeneration"
/// https://arxiv.org/abs/1904.09751 Setting k <= 0 makes this a noop
MGS_API struct mgs_sampler *mgs_sampler_init_top_k(int32_t k);

/// @details Nucleus sampling described in academic paper "The Curious Case of Neural Text Degeneration"
/// https://arxiv.org/abs/1904.09751
MGS_API struct mgs_sampler *mgs_sampler_init_top_p(float p, size_t min_keep);

/// @details Minimum P sampling
MGS_API struct mgs_sampler *mgs_sampler_init_min_p(float p, size_t min_keep);

/// @details Locally Typical Sampling implementation described in the paper https://arxiv.org/abs/2202.00666.
MGS_API struct mgs_sampler *mgs_sampler_init_typical(float p, size_t min_keep);

/// #details Updates the logits l_i` = l_i/t. When t <= 0.0f, the maximum logit is kept at it's original value, the rest
/// are set to -inf
MGS_API struct mgs_sampler *mgs_sampler_init_temp(float t);

/// @details Dynamic temperature implementation (a.k.a. entropy) described in the paper
/// https://arxiv.org/abs/2309.02772.
MGS_API struct mgs_sampler *mgs_sampler_init_temp_ext(float t, float delta, float exponent);

/// @details XTC sampler as described in https://github.com/oobabooga/text-generation-webui/pull/6335
MGS_API struct mgs_sampler *mgs_sampler_init_xtc(float p, float t, size_t min_keep, uint32_t seed);

/// @details Top n sigma sampling as described in academic paper "Top-nσ: Not All Logits Are You Need"
/// https://arxiv.org/pdf/2411.07641
MGS_API struct mgs_sampler *mgs_sampler_init_top_n_sigma(float n);

/// @details Mirostat 1.0 algorithm described in the paper https://arxiv.org/abs/2007.14966. Uses tokens instead of
/// words.
/// @param candidates A vector of `mgs_token_data` containing the candidate tokens, their probabilities (p), and
/// log-odds (logit) for the current position in the generated text.
/// @param tau  The target cross-entropy (or surprise) value you want to achieve for the generated text. A higher value
/// corresponds to more surprising or less predictable text, while a lower value corresponds to less surprising or more
/// predictable text.
/// @param eta The learning rate used to update `mu` based on the error between the target and observed surprisal of the
/// sampled word. A larger learning rate will cause `mu` to be updated more quickly, while a smaller learning rate will
/// result in slower updates.
/// @param m The number of tokens considered in the estimation of `s_hat`. This is an arbitrary value that is used to
/// calculate `s_hat`, which in turn helps to calculate the value of `k`. In the paper, they use `m = 100`, but you can
/// experiment with different values to see how it affects the performance of the algorithm.
/// @param mu Maximum cross-entropy. This value is initialized to be twice the target cross-entropy (`2 * tau`) and is
/// updated in the algorithm based on the error between the target and observed surprisal.
MGS_API struct mgs_sampler *mgs_sampler_init_mirostat(int32_t n_vocab, uint32_t seed, float tau, float eta,
                                                            int32_t m);

/// @details Mirostat 2.0 algorithm described in the paper https://arxiv.org/abs/2007.14966. Uses tokens instead of
/// words.
/// @param candidates A vector of `mgs_token_data` containing the candidate tokens, their probabilities (p), and
/// log-odds (logit) for the current position in the generated text.
/// @param tau  The target cross-entropy (or surprise) value you want to achieve for the generated text. A higher value
/// corresponds to more surprising or less predictable text, while a lower value corresponds to less surprising or more
/// predictable text.
/// @param eta The learning rate used to update `mu` based on the error between the target and observed surprisal of the
/// sampled word. A larger learning rate will cause `mu` to be updated more quickly, while a smaller learning rate will
/// result in slower updates.
/// @param mu Maximum cross-entropy. This value is initialized to be twice the target cross-entropy (`2 * tau`) and is
/// updated in the algorithm based on the error between the target and observed surprisal.
MGS_API struct mgs_sampler *mgs_sampler_init_mirostat_v2(uint32_t seed, float tau, float eta);

/// @details Initializes a GBNF grammar, see grammars/README.md for details.
/// @param vocab The vocabulary that this grammar will be used with.
/// @param grammar_str The production rules for the grammar, encoded as a string. Returns an empty grammar if empty.
/// Returns NULL if parsing of grammar_str fails.
/// @param grammar_root The name of the start symbol for the grammar.
MGS_API struct mgs_sampler *mgs_sampler_init_grammar(const struct mgs_vocab *vocab, const char *grammar_str,
                                                           const char *grammar_root);

/// @details Lazy grammar sampler
/// @param trigger_patterns A list of patterns that will trigger the grammar sampler. Pattern will be matched from the
/// start of the generation output, and grammar sampler will be fed content starting from its first match group.
/// @param trigger_tokens A list of tokens that will trigger the grammar sampler. Grammar sampler will be fed content
/// starting from the trigger token included.
MGS_API struct mgs_sampler *mgs_sampler_init_grammar_lazy_patterns(
    const struct mgs_vocab *vocab, const char *grammar_str, const char *grammar_root, const char **trigger_patterns,
    size_t num_trigger_patterns, const mgs_token *trigger_tokens, size_t num_trigger_tokens);

/// NOTE: Avoid using on the full vocabulary as searching for repeated tokens can become slow. For example, apply top-k
/// or top-p sampling first.
MGS_API struct mgs_sampler *mgs_sampler_init_penalties(
    int32_t n_vocab,
    int32_t penalty_last_n, // last n tokens to penalize (0 = disable penalty, -1 = context size)
    float penalty_repeat,   // must be > 0.0, 1.0 = disabled
    float penalty_freq,     // must be finite, 0.0 = disabled
    float penalty_present); // must be finite, 0.0 = disabled

///  @details DRY sampler, designed by p-e-w, as described in:
///  https://github.com/oobabooga/text-generation-webui/pull/5677, porting Koboldcpp implementation authored by pi6am:
///  https://github.com/LostRuins/koboldcpp/pull/982
MGS_API struct mgs_sampler *mgs_sampler_init_dry(const struct mgs_vocab *vocab, int32_t n_ctx_train,
                                                       float dry_multiplier, float dry_base, int32_t dry_allowed_length,
                                                       int32_t dry_penalty_last_n, const char **seq_breakers,
                                                       size_t num_breakers);

/// adaptive-p: select tokens near a configurable target probability over time.
///
/// the adaptive-p sampler transforms the token probability distribution to favor tokens
/// that fall near a user-configurable probability target.
///
/// internally, the sampler maintains an exponential moving average of the *ORIGINAL*
/// probabilities of selected tokens at each sampling step. it uses this EMA to compute an
/// adapted target probability at each sampling step, thus maintaining the desired target
/// probability over time.
///
/// adaptive-p selects a token ID rather than just mutating candidates, so it must be last
/// in the sampler chain (like mirostat, dist, greedy).
///
/// only mild truncation before this sampler is recommended. we suggest applying min-p
/// before adaptive-p as the only other active sampler in the chain.
///
/// @param target select tokens near this probability (valid range 0.0 to 1.0; negative = disabled)
/// @param decay  EMA decay for adaptation; history ≈ 1/(1-decay) tokens (valid range 0.0 - 0.99)
/// @param seed   RNG seed
///
MGS_API struct mgs_sampler *mgs_sampler_init_adaptive_p(float target, float decay, uint32_t seed);

MGS_API struct mgs_sampler *mgs_sampler_init_logit_bias(int32_t n_vocab, int32_t n_logit_bias,
                                                              const mgs_logit_bias *logit_bias);

// this sampler is meant to be used for fill-in-the-middle infilling
// it's supposed to be used after top_k + top_p sampling
//
// 1. if the sum of the EOG probs times the number of candidates is higher than the sum of the other probs -> pick EOG
// 2. combine probs of tokens that have the same prefix
//
// example:
//
// - before:
//   "hel":   0.5
//   "hell":  0.2
//   "hello": 0.1
//   "dummy": 0.1
//
// - after:
//   "hel":   0.8
//   "dummy": 0.1
//
// 3. discard non-EOG tokens with low prob
// 4. if no tokens are left -> pick EOT
//
MGS_API struct mgs_sampler *mgs_sampler_init_infill(const struct mgs_vocab *vocab);

// Returns the seed used by the sampler if applicable, MGS_DEFAULT_SEED otherwise
MGS_API uint32_t mgs_sampler_get_seed(const struct mgs_sampler *smpl);

/// @details Sample and accept a token from the idx-th output of the last evaluation
//
// Shorthand for:
//    const auto * vLogits = mgs_get_logits_ith(ctx, idx);
//    mgs_token_data_array vCurP = { ... init from vLogits ... };
//    mgs_sampler_apply(smpl, &vCurP);
//    auto vToken = vCurP.data[vCurP.selected].id;
//    mgs_sampler_accept(smpl, vToken);
//    return vToken;
// Returns the sampled token
MGS_API mgs_token mgs_sampler_sample(struct mgs_sampler *smpl, struct mgs_context *ctx, int32_t idx);

// TODO: extend in the future
// MGS_API void mgs_decode_with_sampler(struct mgs_context * ctx, struct mgs_sampler * smpl, struct mgs_batch
// batch, ...);

//
// Model split
//

/// @details Build a split GGUF final path for this chunk.
///          mgs_split_path(split_path, sizeof(split_path), "/models/ggml-model-q4_0", 2, 4) => split_path =
///          "/models/ggml-model-q4_0-00002-of-00004.gguf"
//  Returns the split_path length.
MGS_API int32_t mgs_split_path(char *split_path, size_t maxlen, const char *path_prefix, int32_t split_no,
                                   int32_t split_count);

/// @details Extract the path prefix from the split_path if and only if the split_no and split_count match.
///          mgs_split_prefix(split_prefix, 64, "/models/ggml-model-q4_0-00002-of-00004.gguf", 2, 4) => split_prefix =
///          "/models/ggml-model-q4_0"
//  Returns the split_prefix length.
MGS_API int32_t mgs_split_prefix(char *split_prefix, size_t maxlen, const char *split_path, int32_t split_no,
                                     int32_t split_count);

// Print system information
MGS_API const char *mgs_print_system_info(void);

// Set callback for all future logging events.
// If this is not called, or NULL is supplied, everything is output on stderr.
// The logger state is global so these functions are NOT thread safe.
MGS_API void mgs_log_get(ggml_log_callback *log_callback, void **user_data);
MGS_API void mgs_log_set(ggml_log_callback log_callback, void *user_data);

//
// Performance utils
//
// NOTE: Used by mgs examples/tools, avoid using in third-party apps. Instead, do your own performance
// measurements.
//

struct mgs_perf_context_data {
  // ms == milliseconds
  double t_start_ms;  // absolute start time
  double t_load_ms;   // time needed for loading the model
  double t_p_eval_ms; // time needed for processing the prompt
  double t_eval_ms;   // time needed for generating tokens

  int32_t n_p_eval; // number of prompt tokens
  int32_t n_eval;   // number of generated tokens
  int32_t n_reused; // number of times a ggml compute graph had been reused
};

struct mgs_perf_sampler_data {
  double t_sample_ms; // time needed for sampling in ms

  int32_t n_sample; // number of sampled tokens
};

MGS_API struct mgs_perf_context_data mgs_perf_context(const struct mgs_context *ctx);
MGS_API void mgs_perf_context_print(const struct mgs_context *ctx);
MGS_API void mgs_perf_context_reset(struct mgs_context *ctx);

// NOTE: the following work only with samplers constructed via mgs_sampler_chain_init
MGS_API struct mgs_perf_sampler_data mgs_perf_sampler(const struct mgs_sampler *chain);
MGS_API void mgs_perf_sampler_print(const struct mgs_sampler *chain);
MGS_API void mgs_perf_sampler_reset(struct mgs_sampler *chain);

//
// training
//

// function that returns whether or not a given tensor contains trainable parameters
typedef bool (*mgs_opt_param_filter)(const struct ggml_tensor *tensor, void *userdata);

// always returns true
MGS_API bool mgs_opt_param_filter_all(const struct ggml_tensor *tensor, void *userdata);

struct mgs_opt_params {
  uint32_t n_ctx_train; // assumed context size post training, use context size specified in mgs_context if 0

  mgs_opt_param_filter param_filter; // callback for determining which tensors contain trainable parameters
  void *param_filter_ud;               // userdata for determining which tensors contain trainable parameters

  ggml_opt_get_optimizer_params get_opt_pars; // callback for calculating optimizer parameters
  void *get_opt_pars_ud;                      // userdata for calculating optimizer parameters

  enum ggml_opt_optimizer_type optimizer_type;
};

MGS_API void mgs_opt_init(struct mgs_context *lctx, struct mgs_model *model,
                              struct mgs_opt_params lopt_params);

MGS_API void mgs_opt_epoch(struct mgs_context *lctx, ggml_opt_dataset_t dataset, ggml_opt_result_t result_train,
                               ggml_opt_result_t result_eval, int64_t idata_split,
                               ggml_opt_epoch_callback callback_train, ggml_opt_epoch_callback callback_eval);

#ifdef __cplusplus
}
#endif

#endif // MGS_H
