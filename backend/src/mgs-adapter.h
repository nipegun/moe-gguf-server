#pragma once

#include "mgs.h"

#include "ggml-cpp.h"

#include <string>
#include <unordered_map>
#include <vector>

// TODO: pimpl

//
// mgs_adapter_cvec
//

struct mgs_adapter_cvec {
  ggml_tensor *mTensorFor(int pIl) const;

  ggml_tensor *mApplyTo(ggml_context *pCtx, ggml_tensor *pCur, int pIl) const;

  bool mApply(const mgs_model &pModel, const float *pData, size_t pLen, int32_t pNEmbd, int32_t pIlStart,
              int32_t pIlEnd);

private:
  bool mInit(const mgs_model &pModel);

  int32_t vLayerStart = -1;
  int32_t vLayerEnd = -1;

  std::vector<ggml_context_ptr> lCtxs;
  std::vector<ggml_backend_buffer_ptr> lBufs;

  std::vector<ggml_tensor *> lTensors; // per layer
};

using mgs_adapter_cvec_ptr = std::shared_ptr<mgs_adapter_cvec>;

//
// mgs_adapter_lora
//

struct mgs_adapter_lora_weight {
  ggml_tensor *vA = nullptr;
  ggml_tensor *vB = nullptr;

  // get actual scale based on rank and alpha
  float mGetScale(float pAlpha, float pAdapterScale) const {
    const float cRank = (float)vB->ne[0];
    const float cScale = pAlpha ? pAdapterScale * pAlpha / cRank : pAdapterScale;
    return cScale;
  }

  mgs_adapter_lora_weight() = default;
  mgs_adapter_lora_weight(ggml_tensor *pA, ggml_tensor *pB) : vA(pA), vB(pB) {}
};

struct mgs_adapter_lora {
  mgs_model *vModel = nullptr;

  // map tensor name to lora_a_b
  std::unordered_map<std::string, mgs_adapter_lora_weight> dAbMap;

  std::vector<ggml_context_ptr> lCtxs;
  std::vector<ggml_backend_buffer_ptr> lBufs;

  float vAlpha;

  // gguf metadata
  std::unordered_map<std::string, std::string> dGgufKv;

  // activated lora (aLoRA)
  std::vector<mgs_token> lAloraInvocationTokens;

  explicit mgs_adapter_lora(mgs_model *pModel) : vModel(pModel) {}
  ~mgs_adapter_lora() = default;

  mgs_adapter_lora_weight *mGetWeight(ggml_tensor *pW);

  uint32_t mGetNNodes() const {
    return dAbMap.size() * 6u; // a, b, scale, add, 2 x mul_mat
  }
};

using mgs_adapter_loras = std::unordered_map<mgs_adapter_lora *, float>;
using mgs_adapter_loras_ptr = std::unique_ptr<mgs_adapter_loras>;
