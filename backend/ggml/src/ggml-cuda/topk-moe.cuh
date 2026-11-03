#include "common.cuh"
#include "ggml.h"

#include <initializer_list>

// Rows that one CUDA block handles.
#define TOPK_MOE_ROWS_PER_BLOCK 8

struct ggml_cuda_topk_moe_args {
  bool sigmoid{};
  bool sqrt_softplus{};
  bool softmax{};
  bool delayed_softmax{};
  bool prob_bias{};
  bool norm{};
  bool scale{};
};

void ggml_cuda_op_topk_moe(ggml_backend_cuda_context &pCtx, const ggml_tensor *pLogits, ggml_tensor *pWeights,
                           ggml_tensor *pIds, const ggml_tensor *pClamp, const ggml_tensor *pScale,
                           const ggml_tensor *pBias, const ggml_cuda_topk_moe_args &pArgs);

bool ggml_cuda_should_use_topk_moe(const ggml_tensor *pGatingOp, const ggml_tensor *pWeights,
                                   const ggml_tensor *pLogits, const ggml_tensor *pIds);
