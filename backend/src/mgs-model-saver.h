#pragma once

#include "gguf.h"
#include "mgs.h"
#include "mgs-arch.h"

#include <vector>

// FIXME temporary function for better error messages
bool mgs_model_saver_supports_arch(llm_arch pArch);

struct mgs_model_saver {
  struct gguf_context *vGgufCtx = nullptr;
  const bool cGgufCtxOwned;
  const struct mgs_model *vModel;
  const struct LLM_KV cLlmKv;

  mgs_model_saver(const struct mgs_model *pModel);
  mgs_model_saver(enum llm_arch pArch, struct gguf_context *pGgufCtx);
  ~mgs_model_saver();

  void mAddKv(enum llm_kv pKey, uint32_t pValue);
  void mAddKv(enum llm_kv pKey, int32_t pValue);
  void mAddKv(enum llm_kv pKey, float pValue);
  void mAddKv(enum llm_kv pKey, bool pValue);
  void mAddKv(enum llm_kv pKey, const char *pValue);

  [[noreturn]]
  void mAddKv(enum llm_kv pKey, char pValue); // needed to make the template below compile

  template <typename Container> void mAddKv(enum llm_kv pKey, const Container &pValue, bool pPerLayer = false);

  void mAddKv(enum llm_kv pKey, const std::vector<std::string> &pValue);

  void mAddTensor(const struct ggml_tensor *pTensor);

  void mAddKvFromModel();

  void mAddTensorsFromModel();

  void mSave(const std::string &pPathModel);
  void mSave(FILE *pFile);
};
