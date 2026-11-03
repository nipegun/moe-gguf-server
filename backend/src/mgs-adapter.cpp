#include "mgs-adapter.h"

#include "mgs-impl.h"
#include "mgs-mmap.h"
#include "mgs-model.h"

#include <map>
#include <cassert>
#include <sstream>
#include <stdexcept>

// vec

ggml_tensor *mgs_adapter_cvec::mTensorFor(int pIl) const {
  if (pIl < 0 || pIl < vLayerStart || pIl > vLayerEnd || (size_t)pIl >= lTensors.size()) {
    return nullptr;
  }

  return lTensors[pIl];
}

ggml_tensor *mgs_adapter_cvec::mApplyTo(ggml_context *pCtx, ggml_tensor *pCur, int pIl) const {
  ggml_tensor *vLayerDir = mTensorFor(pIl);
  if (vLayerDir != nullptr) {
    pCur = ggml_add(pCtx, pCur, vLayerDir);
  }

  return pCur;
}

bool mgs_adapter_cvec::mInit(const mgs_model &pModel) {
  const auto &vHparams = pModel.vHparams;

  GGML_ASSERT(lTensors.empty());
  GGML_ASSERT(lCtxs.empty());
  GGML_ASSERT(lBufs.empty());

  // create a context for each buffer type
  std::map<ggml_backend_buffer_type_t, ggml_context *> dCtxMap;
  auto vCtxForBuft = [&](ggml_backend_buffer_type_t pBuft) -> ggml_context * {
    auto vIt = dCtxMap.find(pBuft);
    if (vIt == dCtxMap.end()) {
      ggml_init_params vParams = {
          /*.mem_size   =*/vHparams.mNLayer() * ggml_tensor_overhead(),
          /*.mem_buffer =*/NULL,
          /*.no_alloc   =*/true,
      };

      ggml_context *vCtx = ggml_init(vParams);
      if (!vCtx) {
        return nullptr;
      }

      dCtxMap[pBuft] = vCtx;
      lCtxs.emplace_back(vCtx);

      return vCtx;
    }

    return vIt->second;
  };

  // make tensors
  lTensors.reserve(vHparams.mNLayer());
  lTensors.push_back(nullptr); // there's never a tensor for layer 0
  for (size_t vIl = 1; vIl < vHparams.mNLayer(); vIl++) {
    ggml_backend_buffer_type_t vBuft = pModel.mSelectBuft(vIl);
    ggml_context *vCtx = vCtxForBuft(vBuft);
    if (!vCtx) {
      MGS_LOG_ERROR("%s: failed to allocate context for control vector\n", __func__);
      return false;
    }
    ggml_tensor *vTensor = ggml_new_tensor_1d(vCtx, GGML_TYPE_F32, vHparams.vNEmbd);
    lTensors.push_back(vTensor);
  }

  // allocate tensors / buffers and zero
  lBufs.reserve(dCtxMap.size());
  for (auto vIt : dCtxMap) {
    ggml_backend_buffer_type_t vBuft = vIt.first;
    ggml_context *vCtx = vIt.second;
    ggml_backend_buffer_t vBuf = ggml_backend_alloc_ctx_tensors_from_buft(vCtx, vBuft);
    if (!vBuf) {
      MGS_LOG_ERROR("%s: failed to allocate buffer for control vector\n", __func__);
      return false;
    }
    ggml_backend_buffer_clear(vBuf, 0);
    lBufs.emplace_back(vBuf);
  }

  return true;
}

bool mgs_adapter_cvec::mApply(const mgs_model &pModel, const float *pData, size_t pLen, int32_t pNEmbd,
                              int32_t pIlStart, int32_t pIlEnd) {
  const auto &vHparams = pModel.vHparams;

  if (pData == nullptr) {
    // disable the current control vector (but leave allocated for later)
    vLayerStart = -1;
    vLayerEnd = -1;
    return true;
  }

  if (pNEmbd != (int)vHparams.vNEmbd) {
    MGS_LOG_ERROR("%s: control vector n_embd does not match model\n", __func__);
    return false;
  }

  if (lTensors.empty()) {
    if (!mInit(pModel)) {
      return false;
    }
  }

  vLayerStart = pIlStart;
  vLayerEnd = pIlEnd;

  for (size_t vIl = 1; vIl < vHparams.mNLayer(); vIl++) {
    assert(lTensors[vIl] != nullptr);

    const size_t cOff = pNEmbd * (vIl - 1); // buffer doesn't have data for layer 0, since it's never present
    if (cOff + pNEmbd <= pLen) {
      ggml_backend_tensor_set(lTensors[vIl], pData + cOff, 0, pNEmbd * ggml_element_size(lTensors[vIl]));
    }
  }

  return true;
}

// lora

mgs_adapter_lora_weight *mgs_adapter_lora::mGetWeight(ggml_tensor *pW) {
  const std::string cName(pW->name);

  const auto cPos = dAbMap.find(cName);
  if (cPos != dAbMap.end()) {
    return &cPos->second;
  }

  return nullptr;
}

static void fMgsAdapterLoraInitImpl(mgs_model &pModel, const char *pPathLora, mgs_adapter_lora &pAdapter) {
  MGS_LOG_INFO("%s: loading lora adapter from '%s' ...\n", __func__, pPathLora);

  ggml_context *vCtxInit;
  gguf_init_params vMetaGgufParams = {
      /* .no_alloc = */ true,
      /* .ctx      = */ &vCtxInit,
  };

  gguf_context_ptr vCtxGguf{gguf_init_from_file(pPathLora, vMetaGgufParams)};
  if (!vCtxGguf) {
    throw std::runtime_error("failed to load lora adapter file from " + std::string(pPathLora));
  }

  ggml_context_ptr vCtx{vCtxInit};

  // check metadata
  {
    const gguf_context *vGgufCtx = vCtxGguf.get();

    MGS_LOG_INFO("%s: Dumping metadata keys/values.\n", __func__);

    // get metadata as string
    for (int vI = 0; vI < gguf_get_n_kv(vGgufCtx); vI++) {
      gguf_type vType = gguf_get_kv_type(vGgufCtx, vI);
      const std::string cTypeName = vType == GGUF_TYPE_ARRAY ? fFormat("%s[%s,%zu]", gguf_type_name(vType),
                                                                       gguf_type_name(gguf_get_arr_type(vGgufCtx, vI)),
                                                                       gguf_get_arr_n(vGgufCtx, vI))
                                                             : gguf_type_name(vType);
      const char *vName = gguf_get_key(vGgufCtx, vI);
      const std::string cValue = fGgufKvToStr(vGgufCtx, vI);

      if (vType != GGUF_TYPE_ARRAY) {
        pAdapter.dGgufKv.emplace(vName, cValue);
      }

      const size_t cMaxValueLen = 40;
      std::string vPrintValue =
          cValue.size() > cMaxValueLen ? fFormat("%s...", cValue.substr(0, cMaxValueLen - 3).c_str()) : cValue;
      fReplaceAll(vPrintValue, "\n", "\\n");

      MGS_LOG_INFO("%s: - kv %3d: %42s %-16s = %s\n", __func__, vI, vName, cTypeName.c_str(), vPrintValue.c_str());
    }

    auto vGetKvStr = [&](const std::string &pKey) -> std::string {
      int vId = gguf_find_key(vGgufCtx, pKey.c_str());
      return vId < 0 ? "" : std::string(gguf_get_val_str(vGgufCtx, vId));
    };
    auto vGetKvF32 = [&](const std::string &pKey) -> float {
      int vId = gguf_find_key(vGgufCtx, pKey.c_str());
      return vId < 0 ? 0.0f : gguf_get_val_f32(vGgufCtx, vId);
    };
    LLM_KV vLlmKv = LLM_KV(cLlmArchUnknown);

    auto vGeneralType = vGetKvStr(vLlmKv(cLlmKvGeneralType));
    if (vGeneralType != "adapter") {
      throw std::runtime_error("expect general.type to be 'adapter', but got: " + vGeneralType);
    }

    auto vGeneralArchStr = vGetKvStr(vLlmKv(cLlmKvGeneralArchitecture));
    auto vGeneralArch = fLlmArchFromString(vGeneralArchStr);
    if (vGeneralArch != pModel.vArch) {
      throw std::runtime_error("model arch and LoRA arch mismatch");
    }

    auto vAdapterType = vGetKvStr(vLlmKv(cLlmKvAdapterType));
    if (vAdapterType != "lora") {
      throw std::runtime_error("expect adapter.type to be 'lora', but got: " + vAdapterType);
    }

    pAdapter.vAlpha = vGetKvF32(vLlmKv(cLlmKvAdapterLoraAlpha));

    // parse alora invocation sequence vector
    const auto &vKey = vLlmKv(cLlmKvAdapterAloraInvocationTokens);
    const int cKid = gguf_find_key(vCtxGguf.get(), vKey.c_str());
    if (cKid >= 0) {
      if (gguf_get_kv_type(vCtxGguf.get(), cKid) != GGUF_TYPE_ARRAY) {
        throw std::runtime_error("invalid gguf type for " + vKey);
      }
      const auto cArrType = gguf_get_arr_type(vCtxGguf.get(), cKid);
      if (cArrType != GGUF_TYPE_UINT32) {
        throw std::runtime_error("invalid gguf element type for " + vKey);
      }
      const size_t cSeqLen = gguf_get_arr_n(vCtxGguf.get(), cKid);
      const void *vData = gguf_get_arr_data(vCtxGguf.get(), cKid);
      pAdapter.lAloraInvocationTokens.resize(cSeqLen);
      std::copy((const mgs_token *)vData, (const mgs_token *)vData + cSeqLen, pAdapter.lAloraInvocationTokens.begin());
    }
  }

  int vNTensors = gguf_get_n_tensors(vCtxGguf.get());

  // contexts for each buffer type
  std::map<ggml_backend_buffer_type_t, ggml_context *> dCtxMap;
  auto vCtxForBuft = [&](ggml_backend_buffer_type_t pBuft) -> ggml_context * {
    auto vIt = dCtxMap.find(pBuft);
    if (vIt == dCtxMap.end()) {
      // add a new context
      ggml_init_params vParams = {
          /*.mem_size   =*/vNTensors * ggml_tensor_overhead(),
          /*.mem_buffer =*/NULL,
          /*.no_alloc   =*/true,
      };
      ggml_context *vBuftCtx = ggml_init(vParams);
      if (!vBuftCtx) {
        return nullptr;
      }
      dCtxMap[pBuft] = vBuftCtx;
      pAdapter.lCtxs.emplace_back(vBuftCtx);
      return vBuftCtx;
    };
    return vIt->second;
  };

  // bundle lora_a and lora_b into pairs
  std::map<std::string, mgs_adapter_lora_weight> dAbMap;
  auto vStrEndswith = [](const std::string &pStr, const std::string &pSuffix) {
    return pStr.size() >= pSuffix.size() && pStr.compare(pStr.size() - pSuffix.size(), pSuffix.size(), pSuffix) == 0;
  };

  for (ggml_tensor *vCur = ggml_get_first_tensor(vCtx.get()); vCur; vCur = ggml_get_next_tensor(vCtx.get(), vCur)) {
    std::string vName(vCur->name);
    if (vStrEndswith(vName, ".lora_a")) {
      fReplaceAll(vName, ".lora_a", "");
      if (dAbMap.find(vName) == dAbMap.end()) {
        dAbMap[vName] = mgs_adapter_lora_weight(vCur, nullptr);
      } else {
        dAbMap[vName].vA = vCur;
      }
    } else if (vStrEndswith(vName, ".lora_b")) {
      fReplaceAll(vName, ".lora_b", "");
      if (dAbMap.find(vName) == dAbMap.end()) {
        dAbMap[vName] = mgs_adapter_lora_weight(nullptr, vCur);
      } else {
        dAbMap[vName].vB = vCur;
      }
    } else if (vStrEndswith(vName, "_norm.weight")) {
      // TODO: add support for norm vector
      // for now, we don't really care because most adapters still work fine without it
      continue;
    } else {
      throw std::runtime_error("LoRA tensor '" + vName + "' has unexpected suffix");
    }
  }

  // get extra buffer types of the CPU
  // TODO: a more general solution for non-CPU extra buft should be implemented in the future
  std::vector<ggml_backend_buffer_type_t> lBuftExtra;
  {
    auto *vCpuDev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if (!vCpuDev) {
      throw std::runtime_error(fFormat("%s: no CPU backend found", __func__));
    }
    auto *vCpuReg = ggml_backend_dev_backend_reg(vCpuDev);

    auto vGgmlBackendDevGetExtraBuftsFn = (ggml_backend_dev_get_extra_bufts_t)ggml_backend_reg_get_proc_address(
        vCpuReg, "ggml_backend_dev_get_extra_bufts");

    if (vGgmlBackendDevGetExtraBuftsFn) {
      ggml_backend_buffer_type_t *vExtraBufts = vGgmlBackendDevGetExtraBuftsFn(vCpuDev);
      while (vExtraBufts && *vExtraBufts) {
        lBuftExtra.emplace_back(*vExtraBufts);
        ++vExtraBufts;
      }
    }
  }

  // add tensors
  for (auto &vIt : dAbMap) {
    const std::string &vName = vIt.first;
    mgs_adapter_lora_weight &vW = vIt.second;
    bool vIsTokenEmbd = vStrEndswith(vName, "token_embd.weight");

    if (!vW.vA || !vW.vB) {
      throw std::runtime_error("LoRA tensor pair for '" + vName + "' is missing one component");
    }

    // device buft and device ctx
    const auto *vModelTensor = pModel.mGetTensor(vName.c_str());
    if (!vModelTensor) {
      throw std::runtime_error("LoRA tensor '" + vName +
                               "' does not exist in base model (hint: maybe wrong base model?)");
    }

    auto *vBuft = ggml_backend_buffer_get_type(vModelTensor->buffer);

    // do not load loras to extra buffer types (i.e. bufts for repacking) -> use the CPU in that case
    for (auto &vEx : lBuftExtra) {
      if (vEx == vBuft) {
        MGS_LOG_WARN("%s: lora for '%s' cannot use buft '%s', fallback to CPU\n", __func__, vModelTensor->name,
                     ggml_backend_buft_name(vBuft));

        auto *vCpuDev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        if (!vCpuDev) {
          throw std::runtime_error(fFormat("%s: no CPU backend found", __func__));
        }
        vBuft = ggml_backend_dev_buffer_type(vCpuDev);

        break;
      }
    }

    MGS_LOG_DEBUG("%s: lora for '%s' -> '%s'\n", __func__, vModelTensor->name, ggml_backend_buft_name(vBuft));

    ggml_context *vDevCtx = vCtxForBuft(vBuft);
    // validate tensor shape
    if (vIsTokenEmbd) {
      // expect B to be non-transposed, A and B are flipped; see llm_build_inp_embd()
      if (vModelTensor->ne[0] != vW.vB->ne[1] || vModelTensor->ne[1] != vW.vA->ne[1]) {
        throw std::runtime_error("tensor '" + vName + "' has incorrect shape (hint: maybe wrong base model?)");
      }
    } else {
      if (vModelTensor->ne[0] != vW.vA->ne[0] || vModelTensor->ne[1] != vW.vB->ne[1]) {
        throw std::runtime_error("tensor '" + vName + "' has incorrect shape (hint: maybe wrong base model?)");
      }
      if (vW.vA->ne[1] != vW.vB->ne[0]) {
        throw std::runtime_error(
            "lora_a tensor is not transposed (hint: adapter from \"finetune\" example is no longer supported)");
      }
    }

    // save tensor to adapter
    ggml_tensor *vTensorA = ggml_dup_tensor(vDevCtx, vW.vA);
    ggml_tensor *vTensorB = ggml_dup_tensor(vDevCtx, vW.vB);
    ggml_set_name(vTensorA, vW.vA->name);
    ggml_set_name(vTensorB, vW.vB->name);
    pAdapter.dAbMap[vName] = mgs_adapter_lora_weight(vTensorA, vTensorB);
  }

  // allocate tensors / buffers and zero
  {
    pAdapter.lCtxs.reserve(dCtxMap.size());
    pAdapter.lBufs.reserve(dCtxMap.size());
    for (auto &vIt : dCtxMap) {
      ggml_backend_buffer_type_t vBuft = vIt.first;
      ggml_context *vCtxDev = vIt.second;
      ggml_backend_buffer_ptr vBuf{ggml_backend_alloc_ctx_tensors_from_buft(vCtxDev, vBuft)};
      if (!vBuf) {
        throw std::runtime_error("failed to allocate buffer for lora adapter\n");
      }
      MGS_LOG_INFO("%s: %10s LoRA buffer size = %8.2f MiB\n", __func__, ggml_backend_buffer_name(vBuf.get()),
                   ggml_backend_buffer_get_size(vBuf.get()) / 1024.0 / 1024.0);
      pAdapter.lBufs.emplace_back(std::move(vBuf));
    }
  }

  // set tensor data
  {
    mgs_file vGgufFile(pPathLora, "rb");
    std::vector<uint8_t> lReadBuf;
    auto vSetTensor = [&](ggml_tensor *pOrig, ggml_tensor *pDev) {
      size_t vOffs = gguf_get_data_offset(vCtxGguf.get()) +
                     gguf_get_tensor_offset(vCtxGguf.get(), gguf_find_tensor(vCtxGguf.get(), pOrig->name));
      size_t vSize = ggml_nbytes(pOrig);
      lReadBuf.resize(vSize);
      vGgufFile.mSeek(vOffs, SEEK_SET);
      vGgufFile.mReadRaw(lReadBuf.data(), vSize);
      ggml_backend_tensor_set(pDev, lReadBuf.data(), 0, vSize);
    };
    for (auto &vIt : pAdapter.dAbMap) {
      auto vOrig = dAbMap[vIt.first];
      auto vDev = vIt.second;
      vSetTensor(vOrig.vA, vDev.vA);
      vSetTensor(vOrig.vB, vDev.vB);
    }
  }

  // register adapter with model
  pModel.sLoras.insert(&pAdapter);

  MGS_LOG_INFO("%s: loaded %zu tensors from lora file\n", __func__, pAdapter.dAbMap.size() * 2);
}

mgs_adapter_lora *mgs_adapter_lora_init(mgs_model *pModel, const char *pPathLora) {
  mgs_adapter_lora *vAdapter = new mgs_adapter_lora(pModel);

  try {
    fMgsAdapterLoraInitImpl(*pModel, pPathLora, *vAdapter);
    return vAdapter;
  } catch (const std::exception &vErr) {
    MGS_LOG_ERROR("%s: failed to apply lora adapter: %s\n", __func__, vErr.what());

    delete vAdapter;
  }

  return nullptr;
}

int32_t mgs_adapter_meta_val_str(const mgs_adapter_lora *pAdapter, const char *pKey, char *pBuf, size_t pBufSize) {
  const auto &vIt = pAdapter->dGgufKv.find(pKey);
  if (vIt == pAdapter->dGgufKv.end()) {
    if (pBufSize > 0) {
      pBuf[0] = '\0';
    }
    return -1;
  }
  return snprintf(pBuf, pBufSize, "%s", vIt->second.c_str());
}

int32_t mgs_adapter_meta_count(const mgs_adapter_lora *pAdapter) { return (int)pAdapter->dGgufKv.size(); }

int32_t mgs_adapter_meta_key_by_index(const mgs_adapter_lora *pAdapter, int pI, char *pBuf, size_t pBufSize) {
  if (pI < 0 || pI >= (int)pAdapter->dGgufKv.size()) {
    if (pBufSize > 0) {
      pBuf[0] = '\0';
    }
    return -1;
  }
  auto vIt = pAdapter->dGgufKv.begin();
  std::advance(vIt, pI);
  return snprintf(pBuf, pBufSize, "%s", vIt->first.c_str());
}

int32_t mgs_adapter_meta_val_str_by_index(const mgs_adapter_lora *pAdapter, int32_t pI, char *pBuf, size_t pBufSize) {
  if (pI < 0 || pI >= (int)pAdapter->dGgufKv.size()) {
    if (pBufSize > 0) {
      pBuf[0] = '\0';
    }
    return -1;
  }
  auto vIt = pAdapter->dGgufKv.begin();
  std::advance(vIt, pI);
  return snprintf(pBuf, pBufSize, "%s", vIt->second.c_str());
}

void mgs_adapter_lora_free(mgs_adapter_lora *pAdapter) {
  if (pAdapter == nullptr) {
    return;
  }

  if (pAdapter->vModel != nullptr) {
    pAdapter->vModel->sLoras.erase(pAdapter);
    pAdapter->vModel = nullptr;
  }

  delete pAdapter;
}

uint64_t mgs_adapter_get_alora_n_invocation_tokens(const struct mgs_adapter_lora *pAdapter) {
  if (!pAdapter) {
    return 0;
  }
  return pAdapter->lAloraInvocationTokens.size();
}

const mgs_token *mgs_adapter_get_alora_invocation_tokens(const mgs_adapter_lora *pAdapter) {
  GGML_ASSERT(pAdapter);
  return pAdapter->lAloraInvocationTokens.data();
}
