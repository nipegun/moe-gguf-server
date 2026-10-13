#include "mgs-context.h"

#include "ggml.h"
#include "mgs-arch.h"
#include "mgs-graph.h"
#include "mgs-impl.h"
#include "mgs-batch.h"
#include "mgs-io.h"
#include "mgs-memory.h"
#include "mgs-mmap.h"
#include "mgs-model.h"
#include "mgs-ext.h"
#include "mgs.h"

#include <cinttypes>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>

//
// mgs_context
//

static llm_graph_type fCtxTypeToGraphType(mgs_context_type pCtxType) {
  switch (pCtxType) {
  case MGS_CONTEXT_TYPE_DEFAULT:
    return cLlmGraphTypeDefault;
  case MGS_CONTEXT_TYPE_MTP:
    return cLlmGraphTypeDecoderMtp;
  }
  throw std::runtime_error("Unsupported ctx type");
}

struct llm_fused_op_probe {
  llm_fused_op vOp;
  const char *vName;
  uint32_t vNTokensPerSeq;
};

static const llm_fused_op_probe cLlmFusedOpFlashAttnProbe = {
    /*.vOp              =*/cLlmFusedOpFlashAttn,
    /*.vName            =*/"Flash Attention",
    /*.vNTokensPerSeq   =*/1,
};

static const llm_fused_op_probe cLlmFusedOpGdnArProbe = {
    /*.vOp              =*/cLlmFusedOpGdnAr,
    /*.vName            =*/"fused Gated Delta Net (autoregressive)",
    /*.vNTokensPerSeq   =*/1,
};

static const llm_fused_op_probe cLlmFusedOpGdnChProbe = {
    /*.vOp              =*/cLlmFusedOpGdnCh,
    /*.vName            =*/"fused Gated Delta Net (chunked)",
    /*.vNTokensPerSeq   =*/16,
};

static const llm_fused_op_probe cLlmFusedOpLidProbe = {
    /*.vOp              =*/cLlmFusedOpLightningIndexer,
    /*.vName            =*/"Lightning Indexer",
    /*.vNTokensPerSeq   =*/1,
};

static const llm_fused_op_probe cLlmFusedOpDsv4HcPreProbe = {
    /*.vOp              =*/cLlmFusedOpDsv4HcPre,
    /*.vName            =*/"fused DeepSeek V4 HC pre",
    /*.vNTokensPerSeq   =*/1,
};

static const llm_fused_op_probe cLlmFusedOpDsv4HcCombProbe = {
    /*.vOp              =*/cLlmFusedOpDsv4HcComb,
    /*.vName            =*/"fused DeepSeek V4 HC comb",
    /*.vNTokensPerSeq   =*/1,
};

static const llm_fused_op_probe cLlmFusedOpDsv4HcPostProbe = {
    /*.vOp              =*/cLlmFusedOpDsv4HcPost,
    /*.vName            =*/"fused DeepSeek V4 HC post",
    /*.vNTokensPerSeq   =*/1,
};

mgs_context::mgs_context(const mgs_model &pModel, mgs_context_params pParams)
    : vModel(pModel), vCvec(std::make_unique<mgs_adapter_cvec>()), vLoras(std::make_unique<mgs_adapter_loras>()),
      vBalloc(std::make_unique<mgs_batch_allocr>(pModel.vHparams.mNPosPerEmbd())) {
  // TODO warning when creating mgs_context with awkward ctx size that is not a power of 2,
  //     may need to be backend-dependent
  MGS_LOG_INFO("%s: constructing mgs_context\n", __func__);

  vTStartUs = pModel.vTStartUs;
  vTLoadUs = pModel.vTLoadUs;

  const auto &vHparams = pModel.vHparams;

  vCparams.vNSeqMax = std::max(1u, pParams.n_seq_max);
  if (vCparams.vNSeqMax > MGS_MAX_SEQ) {
    throw std::runtime_error("n_seq_max must be <= " + std::to_string(MGS_MAX_SEQ));
  }

  vCparams.vNRsSeq = pParams.n_rs_seq;
  if (vCparams.vNRsSeq > 0 && !fLlmArchSupportsRsRollback(pModel.vArch)) {
    MGS_LOG_DEBUG(
        "%s: n_rs_seq=%u requested but model arch does not support recurrent partial rollback; clamping to 0\n",
        __func__, vCparams.vNRsSeq);
    vCparams.vNRsSeq = 0;
  }

  vCparams.vNThreads = pParams.n_threads;
  vCparams.vNThreadsBatch = pParams.n_threads_batch;
  vCparams.vYarnExtFactor = pParams.yarn_ext_factor >= 0.0f ? pParams.yarn_ext_factor : vHparams.vYarnExtFactor;
  vCparams.vYarnAttnFactor = pParams.yarn_attn_factor >= 0.0f ? pParams.yarn_attn_factor : vHparams.vYarnAttnFactor;
  vCparams.vYarnBetaFast = pParams.yarn_beta_fast >= 0.0f ? pParams.yarn_beta_fast : vHparams.vYarnBetaFast;
  vCparams.vYarnBetaSlow = pParams.yarn_beta_slow >= 0.0f ? pParams.yarn_beta_slow : vHparams.vYarnBetaSlow;
  vCparams.vEmbeddings = pParams.embeddings;
  vCparams.vEmbeddingsNextn = false;
  vCparams.vEmbeddingsNextnMasked = false;
  vCparams.vOffloadKqv = pParams.offload_kqv;
  vCparams.vNoPerf = pParams.no_perf;

  // +1: id n_layer() taps the output of the last layer ("input" of the head)
  vCparams.lEmbeddingsLayerInp.resize(vHparams.mNLayer() + 1, false);
  lEmbdLayerInp.resize(vHparams.mNLayer() + 1);

  vCparams.vCtxType = pParams.ctx_type;
  vCparams.vPoolingType = pParams.pooling_type;

  vCparams.vNCtx = pParams.n_ctx == 0 ? vHparams.vNCtxTrain : pParams.n_ctx;
  vCparams.vRopeFreqBase = pParams.rope_freq_base == 0.0f ? vHparams.vRopeFreqBaseTrain : pParams.rope_freq_base;
  vCparams.vRopeFreqScale = pParams.rope_freq_scale == 0.0f ? vHparams.vRopeFreqScaleTrain : pParams.rope_freq_scale;

  vCparams.vNCtxOrigYarn = pParams.yarn_orig_ctx != 0    ? pParams.yarn_orig_ctx
                           : vHparams.vNCtxOrigYarn != 0 ? vHparams.vNCtxOrigYarn
                                                         : vHparams.vNCtxTrain;

  vCparams.vCbEval = pParams.cb_eval;
  vCparams.vCbEvalUserData = pParams.cb_eval_user_data;

  vCparams.vCtxOther = nullptr;

  if (pModel.vArch == cLlmArchEagle3 || pModel.vArch == cLlmArchDflash) {
    if (pModel.vTokEmbd == nullptr || pModel.vOutput == nullptr) {
      if (pParams.ctx_other == nullptr) {
        throw std::runtime_error(pModel.mArchName() +
                                 " requires ctx_other to be set (this warning is normal during memory fitting)");
      }
      vCparams.vCtxOther = pParams.ctx_other;
    }
  }

  // Initialize backend samplers here so they are part of the sampling graph
  // before the reserve passes run later in this function. This avoids a later
  // re-reserve when graph nodes change.
  if (pParams.samplers != nullptr && pParams.n_samplers > 0) {
    for (size_t vI = 0; vI < pParams.n_samplers; ++vI) {
      const auto &vConfig = pParams.samplers[vI];

      if (mgs_sampler_chain_get(vConfig.sampler, -1) == nullptr) {
        throw std::runtime_error("the backend samplers must be of type mgs_sampler_chain");
      }

      if (mSetSampler(vConfig.seq_id, vConfig.sampler)) {
        const int cNSamplers = mgs_sampler_chain_n(vConfig.sampler);

        MGS_LOG_INFO("%s: setting backend sampler for seq_id %d (n = %d)\n", __func__, vConfig.seq_id, cNSamplers);
      }
    }
  }

  auto vRopeScalingType = pParams.rope_scaling_type;
  if (vRopeScalingType == MGS_ROPE_SCALING_TYPE_UNSPECIFIED) {
    vRopeScalingType = vHparams.vRopeScalingTypeTrain;
  }

  if (vRopeScalingType == MGS_ROPE_SCALING_TYPE_NONE) {
    vCparams.vRopeFreqScale = 1.0f; // never scale if scaling type is none
  }

  if (vCparams.vYarnExtFactor < 0.0f) { // negative indicates 'not set'
    vCparams.vYarnExtFactor = vRopeScalingType == MGS_ROPE_SCALING_TYPE_YARN ? 1.0f : 0.0f;
  }

  if (vCparams.vYarnExtFactor != 0) {
    static auto vGetMscale = [](float pScale, float pMscale) {
      return pScale <= 1.0f ? 1.0f : (0.1f * pMscale * logf(pScale) + 1.0f);
    };

    const float cFactor = 1.0f / vCparams.vRopeFreqScale;

    // ref:
    // https://github.com/huggingface/transformers/blob/6d00f6b0a5679c36510f203e4226e36f517c3032/src/transformers/modeling_rope_utils.py#L336-L348
    // [TAG_YARN_LOG_MUL_FIX]
    if (vHparams.vRopeYarnLogMul != 0.0f) {
      // note: here we assume `mscale == 1.0f`
      // TODO: start reading the actual value of mscale and handle the case where it is not 1.0f
      const float cMscale = 1.0f;
      const float cMscaleAllDims = vHparams.vRopeYarnLogMul;

      vCparams.vYarnAttnFactor = vGetMscale(cFactor, cMscale) / vGetMscale(cFactor, cMscaleAllDims);

      MGS_LOG_WARN("%s: setting new yarn_attn_factor = %.4f (mscale == %.1f, mscale_all_dim = %.1f)\n", __func__,
                   vCparams.vYarnAttnFactor, cMscale, cMscaleAllDims);
    } else {
      vCparams.vYarnAttnFactor = vGetMscale(cFactor, 1.0f);
    }

    // when YARN is applied with yarn_ext_factor != 0.0f, we need to cancel this factor
    // (see the mscale applied by rope_yarn() in ggml/src/ggml-cpu/ops.cpp)
    vCparams.vYarnAttnFactor *= 1.0f / (1.0f + 0.1f * logf(cFactor));
  }

  vCparams.vYarnAttnFactor *= vHparams.vRopeAttnFactor;

  if (vCparams.vPoolingType == MGS_POOLING_TYPE_UNSPECIFIED) {
    if (vHparams.vPoolingType == MGS_POOLING_TYPE_UNSPECIFIED) {
      vCparams.vPoolingType = MGS_POOLING_TYPE_NONE;
    } else {
      vCparams.vPoolingType = vHparams.vPoolingType;
    }
  }

  if (pParams.attention_type == MGS_ATTENTION_TYPE_UNSPECIFIED) {
    vCparams.vCausalAttn = vHparams.vCausalAttn;
  } else {
    vCparams.vCausalAttn = pParams.attention_type == MGS_ATTENTION_TYPE_CAUSAL;
  }

  vCparams.vFlashAttn = pParams.flash_attn_type != MGS_FLASH_ATTN_TYPE_DISABLED;
  vCparams.vAutoFa = pParams.flash_attn_type == MGS_FLASH_ATTN_TYPE_AUTO;

  vCparams.vFusedGdnAr = true;
  vCparams.vFusedGdnCh = true;
  vCparams.vAutoFgdn = true;

  vCparams.vFusedLid = true;
  vCparams.vAutoFlid = true;

  vCparams.vFusedDsv4HcPre = true;
  vCparams.vFusedDsv4HcComb = true;
  vCparams.vFusedDsv4HcPost = true;
  vCparams.vAutoFhc = true;

  // with causal attention, the batch size is limited by the context size
  vCparams.vNBatch = vCparams.vCausalAttn ? std::min(vCparams.vNCtx, pParams.n_batch) : pParams.n_batch;

  vCparams.vNUbatch = std::min(vCparams.vNBatch, pParams.n_ubatch == 0 ? pParams.n_batch : pParams.n_ubatch);

  vCparams.vNOutputsMax =
      pParams.n_outputs_max == 0 || mgs_model_has_encoder(&pModel) ? vCparams.vNBatch : pParams.n_outputs_max;

  vCparams.vOpOffload = pParams.op_offload;
  vCparams.vKvUnified = pParams.kv_unified;

  // initialized later
  vCparams.vPipelineParallel = false;

  {
    const char *vMgsGraphReuseDisable = getenv("MGS_GRAPH_REUSE_DISABLE");
    vGraphReuseDisable = vMgsGraphReuseDisable ? (atoi(vMgsGraphReuseDisable) != 0) : vGraphReuseDisable;

    if (vGraphReuseDisable) {
      MGS_LOG_WARN("%s: graph reuse disabled\n", __func__);
    }
  }

  vCparams.vNCtx = GGML_PAD(vCparams.vNCtx, 256);

  if (vCparams.vKvUnified) {
    vCparams.vNCtxSeq = vCparams.vNCtx;
  } else {
    vCparams.vNCtxSeq = vCparams.vNCtx / vCparams.vNSeqMax;
    vCparams.vNCtxSeq = GGML_PAD(vCparams.vNCtxSeq, 256);

    if (vCparams.vNCtxSeq == 0) {
      throw std::runtime_error("n_ctx_seq == 0");
    }

    if (vCparams.vNCtx != vCparams.vNCtxSeq * vCparams.vNSeqMax) {
      vCparams.vNCtx = vCparams.vNCtxSeq * vCparams.vNSeqMax;
      MGS_LOG_WARN("%s: n_ctx is not divisible by n_seq_max - rounding down to %u\n", __func__, vCparams.vNCtx);
    }
  }

  MGS_LOG_INFO("%s: n_seq_max     = %u\n", __func__, vCparams.vNSeqMax);
  MGS_LOG_INFO("%s: n_ctx         = %u\n", __func__, vCparams.vNCtx);
  MGS_LOG_INFO("%s: n_ctx_seq     = %u\n", __func__, vCparams.vNCtxSeq);
  MGS_LOG_INFO("%s: n_batch       = %u\n", __func__, vCparams.vNBatch);
  MGS_LOG_INFO("%s: n_ubatch      = %u\n", __func__, vCparams.vNUbatch);
  MGS_LOG_INFO("%s: causal_attn   = %d\n", __func__, vCparams.vCausalAttn);
  MGS_LOG_INFO("%s: flash_attn    = %s\n", __func__, mgs_flash_attn_type_name(pParams.flash_attn_type));
  MGS_LOG_INFO("%s: kv_unified    = %s\n", __func__, vCparams.vKvUnified ? "true" : "false");
  MGS_LOG_INFO("%s: freq_base     = %.1f\n", __func__, vCparams.vRopeFreqBase);
  MGS_LOG_INFO("%s: freq_scale    = %g\n", __func__, vCparams.vRopeFreqScale);
  MGS_LOG_INFO("%s: n_rs_seq      = %u\n", __func__, vCparams.vNRsSeq);
  MGS_LOG_INFO("%s: n_outputs_max = %u\n", __func__, vCparams.vNOutputsMax);

  if (vCparams.vNCtxSeq < vHparams.vNCtxTrain) {
    MGS_LOG_INFO("%s: n_ctx_seq (%u) < n_ctx_train (%u) -- the full capacity of the model will not be utilized\n",
                 __func__, vCparams.vNCtxSeq, vHparams.vNCtxTrain);
  }

  if (vCparams.vNCtxSeq > vHparams.vNCtxTrain) {
    MGS_LOG_WARN("%s: n_ctx_seq (%u) > n_ctx_train (%u) -- possible training context overflow\n", __func__,
                 vCparams.vNCtxSeq, vHparams.vNCtxTrain);
  }

  if (!vHparams.vVocabOnly) {
    // GPU backends
    for (const auto &vDev : pModel.lDevices) {
      ggml_backend_t vBackend = ggml_backend_dev_init(vDev.vDev, nullptr);
      if (vBackend == nullptr) {
        throw std::runtime_error(fFormat("failed to initialize %s backend", ggml_backend_dev_name(vDev.vDev)));
      }
      lBackends.emplace_back(vBackend);
    }

    // add ACCEL backends (such as BLAS)
    for (size_t vI = 0; vI < ggml_backend_dev_count(); ++vI) {
      ggml_backend_dev_t vDev = ggml_backend_dev_get(vI);
      if (ggml_backend_dev_type(vDev) == GGML_BACKEND_DEVICE_TYPE_ACCEL) {
        ggml_backend_t vBackend = ggml_backend_dev_init(vDev, nullptr);
        if (vBackend == nullptr) {
          throw std::runtime_error(fFormat("failed to initialize %s backend", ggml_backend_dev_name(vDev)));
        }
        lBackends.emplace_back(vBackend);
      }
    }

    // add CPU backend
    vBackendCpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (vBackendCpu == nullptr) {
      throw std::runtime_error("failed to initialize CPU backend");
    }
    lBackends.emplace_back(vBackendCpu);

    // create a list of the set_n_threads functions in the backends
    for (auto &vBackend : lBackends) {
      ggml_backend_dev_t vDev = ggml_backend_get_device(vBackend.get());
      ggml_backend_reg_t vReg = vDev ? ggml_backend_dev_backend_reg(vDev) : nullptr;
      if (vReg) {
        auto vGgmlBackendSetNThreadsFn =
            (ggml_backend_set_n_threads_t)ggml_backend_reg_get_proc_address(vReg, "ggml_backend_set_n_threads");
        if (vGgmlBackendSetNThreadsFn) {
          lSetNThreadsFns.emplace_back(vBackend.get(), vGgmlBackendSetNThreadsFn);
        }
      }
    }

    mgs_set_abort_callback(this, pParams.abort_callback, pParams.abort_callback_data);

    // graph outputs buffer
    {
      if (mOutputReserve(pParams.n_seq_max) < pParams.n_seq_max) {
        throw std::runtime_error("failed to reserve initial output buffer");
      }

      MGS_LOG_INFO("%s: %10s  output buffer size = %8.2f MiB\n", __func__, ggml_backend_buffer_name(vBufOutput.get()),
                   ggml_backend_buffer_get_size(vBufOutput.get()) / 1024.0 / 1024.0);
    }
  }

  // init the memory module
  if (!vHparams.vVocabOnly) {
    mgs_memory_params vParamsMem = {
        /*.vTypeK    =*/pParams.type_k,
        /*.vTypeV    =*/pParams.type_v,
        /*.vSwaFull  =*/pParams.swa_full,
        /*.vCtxType  =*/vCparams.vCtxType,
        /*.vMemOther =*/mgs_get_memory(vCparams.vCtxOther),
    };

    vMemory.reset(pModel.mCreateMemory(vParamsMem, vCparams));
  }

  // init backends
  if (!vHparams.vVocabOnly) {
    MGS_LOG_DEBUG("%s: enumerating backends\n", __func__);

    lBackendBuft.clear();
    lBackendPtrs.clear();
    lBackendBufExpSize.clear();

    for (auto &vBackend : lBackends) {
      auto *vBuft = ggml_backend_get_default_buffer_type(vBackend.get());
      auto vBackendType = ggml_backend_dev_type(ggml_backend_get_device(vBackend.get()));

      if (vBackendType == GGML_BACKEND_DEVICE_TYPE_CPU && !pModel.lDevices.empty()) {
        // use the host buffer of the first device CPU for faster transfer of the intermediate state
        const auto &vDev = pModel.lDevices[0];
        auto *vHostBuft = ggml_backend_dev_host_buffer_type(vDev.vDev);
        if (vHostBuft) {
          vBuft = vHostBuft;
        }
      }

      lBackendBuft.push_back(vBuft);
      lBackendPtrs.push_back(vBackend.get());
      lBackendBufExpSize.push_back(0);
    }

    MGS_LOG_DEBUG("%s: backend_ptrs.size() = %zu\n", __func__, lBackendPtrs.size());

    // TODO: move these checks to ggml_backend_sched
    // enabling pipeline parallelism in the scheduler increases memory usage, so it is only done when necessary
    bool vPipelineParallel = pModel.mNDevices() > 1 && pModel.mNGpuLayers() > pModel.vHparams.vNLayerAll &&
                             pModel.mSplitMode() == MGS_SPLIT_MODE_LAYER && vCparams.vOffloadKqv &&
                             !pModel.mHasTensorOverrides();

    // pipeline parallelism requires support for async compute and events in all devices
    if (vPipelineParallel) {
      for (auto &vBackend : lBackends) {
        auto vDevType = ggml_backend_dev_type(ggml_backend_get_device(vBackend.get()));
        if (vDevType == GGML_BACKEND_DEVICE_TYPE_CPU) {
          // ignore CPU backend
          // TODO: should we ignore ACCEL types too?
          continue;
        }
        auto *vDev = ggml_backend_get_device(vBackend.get());
        ggml_backend_dev_props vProps;
        ggml_backend_dev_get_props(vDev, &vProps);
        if (!vProps.caps.async || !vProps.caps.events) {
          // device does not support async compute or events
          vPipelineParallel = false;
          break;
        }
      }
    }

    vCparams.vPipelineParallel = vPipelineParallel;

    if (vCparams.vPipelineParallel) {
      MGS_LOG_INFO("%s: pipeline parallelism enabled\n", __func__);
    }

    mSchedReserve();

    if (!vCparams.vFlashAttn) {
      if (ggml_is_quantized(pParams.type_v)) {
        throw std::runtime_error("quantized V cache was requested, but this requires Flash Attention");
      }
    }
  }

  // Initialize the full vocabulary token ids for backend samplers.
  {
    const int cNVocab = pModel.vVocab.mNTokens();

    vSampling.lTokenIdsFullVocab.resize(cNVocab);
    for (int vI = 0; vI < cNVocab; ++vI) {
      vSampling.lTokenIdsFullVocab[vI] = vI;
    }
  }
}

mgs_context::~mgs_context() {
  // wait for any pending asynchronous copies into the output buffers before they are freed
  mSynchronize();

  if (!vModel.vHparams.vNoAlloc) {
    for (size_t vI = 0; vI < lBackendPtrs.size(); ++vI) {
      ggml_backend_t vBackend = lBackendPtrs[vI];
      ggml_backend_buffer_type_t vBuft = lBackendBuft[vI];

      const size_t cSizeExp = lBackendBufExpSize[vI];
      const size_t cSizeAct = ggml_backend_sched_get_buffer_size(vSched.get(), vBackend);
      if (cSizeExp == cSizeAct) {
        MGS_LOG_DEBUG("%s: %10s compute buffer size is %8.4f MiB, matches expectation of %8.4f MiB\n", __func__,
                      ggml_backend_buft_name(vBuft), cSizeAct / (1024.0 * 1024.0), cSizeExp / (1024.0 * 1024.0));
      } else {
        MGS_LOG_WARN("%s: %10s compute buffer size of %8.4f MiB, does not match expectation of %8.4f MiB\n", __func__,
                     ggml_backend_buft_name(vBuft), cSizeAct / (1024.0 * 1024.0), cSizeExp / (1024.0 * 1024.0));
      }
    }
  }
  ggml_opt_free(vOptCtx);
}

void mgs_context::mResolveFusedOps(const mgs_memory_context_i *pMctx, uint32_t pNSeqs) {
  const char *vFunc = __func__;
  auto vResolve = [&](const llm_fused_op_probe &pProbe, bool &pEnabled) {
    if (!pEnabled) {
      return;
    }

    const uint32_t cNTokensProbe = pProbe.vNTokensPerSeq * pNSeqs;

    auto *vGf = mGraphReserve(cNTokensProbe, pNSeqs, cNTokensProbe, pMctx, true);
    if (!vGf) {
      throw std::runtime_error(std::string("failed to reserve graph for ") + pProbe.vName + " check");
    }

    bool vDeviceMismatch = false;
    for (const auto &vNode : mGetGfResReserve()->mGetFusedNodes()) {
      if (vNode.vOp != pProbe.vOp) {
        continue;
      }

      GGML_ASSERT(vNode.vIl >= 0);

      ggml_backend_t vBackendFused = ggml_backend_sched_get_tensor_backend(vSched.get(), vNode.vTensor);
      ggml_backend_dev_t vDeviceFused = vBackendFused ? ggml_backend_get_device(vBackendFused) : nullptr;

      // TODO: make this descriptor-specific; vModel.mDevLayer() preserves the current behavior,
      // but is still wrong for cases like --no-kv-offload.
      ggml_backend_dev_t vDeviceLayer = vModel.mDevLayer(vNode.vIl);

      if (vDeviceFused != vDeviceLayer) {
        MGS_LOG_WARN("%s: layer %d is assigned to device %s but %s "
                     "is assigned to device %s (usually due to missing support)\n",
                     vFunc, vNode.vIl, vDeviceLayer ? ggml_backend_dev_name(vDeviceLayer) : "none", pProbe.vName,
                     vDeviceFused ? ggml_backend_dev_name(vDeviceFused) : "none");
        vDeviceMismatch = true;
        break;
      }
    }

    if (vDeviceMismatch) {
      pEnabled = false;
      MGS_LOG_WARN("%s: %s not supported, set to disabled\n", vFunc, pProbe.vName);
    } else {
      pEnabled = true;
      MGS_LOG_INFO("%s: %s enabled\n", vFunc, pProbe.vName);
    }
  };

  if (vCparams.vAutoFa) {
    vResolve(cLlmFusedOpFlashAttnProbe, vCparams.vFlashAttn);
    vCparams.vAutoFa = false;
  }

  if (vCparams.vAutoFgdn) {
    MGS_LOG_INFO("%s: resolving fused Gated Delta Net support:\n", vFunc);
    vResolve(cLlmFusedOpGdnArProbe, vCparams.vFusedGdnAr);
    vResolve(cLlmFusedOpGdnChProbe, vCparams.vFusedGdnCh);
    vCparams.vAutoFgdn = false;
  }

  if (vCparams.vAutoFlid) {
    MGS_LOG_INFO("%s: resolving fused Lightning Indexer support:\n", vFunc);
    vResolve(cLlmFusedOpLidProbe, vCparams.vFusedLid);
    vCparams.vAutoFlid = false;
  }

  if (vCparams.vAutoFhc) {
    MGS_LOG_INFO("%s: resolving fused DeepSeek V4 HC support:\n", vFunc);
    vResolve(cLlmFusedOpDsv4HcPreProbe, vCparams.vFusedDsv4HcPre);
    vResolve(cLlmFusedOpDsv4HcCombProbe, vCparams.vFusedDsv4HcComb);
    vResolve(cLlmFusedOpDsv4HcPostProbe, vCparams.vFusedDsv4HcPost);
    vCparams.vAutoFhc = false;
  }
}

void mgs_context::mSchedReserve() {
  if (!vSchedNeedReserve) {
    return;
  }

  vSchedNeedReserve = false;

  MGS_LOG_INFO("%s: reserving ...\n", __func__);

  mSynchronize();

  const int64_t cTStartUs = ggml_time_us();

  const uint32_t cNSeqs = vCparams.vNSeqMax;
  const uint32_t cNTokens = std::min(vCparams.vNCtx, vCparams.vNUbatch);

  const size_t cMaxNodes = this->mGraphMaxNodes(cNTokens);

  MGS_LOG_DEBUG("%s: max_nodes = %zu\n", __func__, cMaxNodes);

  for (auto &vResult : aGfResPrev) {
    vResult.reset();
  }
  vGfResReserve.reset(new llm_graph_result(cMaxNodes));
  vGfResPrevActive = nullptr;

  vSched.reset(ggml_backend_sched_new(lBackendPtrs.data(), lBackendBuft.data(), lBackendPtrs.size(), cMaxNodes,
                                      vCparams.vPipelineParallel, vCparams.vOpOffload));
  // Expert prefetch uploads reuse the experts already held by the MoE expert cache.
  ggml_backend_sched_set_weight_upload(vSched.get(), vExpertCache ? mgs_expert_cache::mFUploadWeights : nullptr,
                                       vExpertCache.get());

  mgs_memory_context_ptr vMctx;
  if (vMemory) {
    MGS_LOG_DEBUG("%s: reserving full memory module\n", __func__);
    vMctx = vMemory->mInitFull();
    if (!vMctx) {
      throw std::runtime_error("failed to initialize memory module");
    }
  }

  // avoid reserving graphs with zero outputs - assume one output per sequence
  const int cNOutputs = cNSeqs;

  MGS_LOG_DEBUG("%s: worst-case: n_tokens = %d, n_seqs = %d, n_outputs = %d\n", __func__, cNTokens, cNSeqs, cNOutputs);

  mResolveFusedOps(vMctx.get(), cNSeqs);

  // reserve worst-case graph
  int vNSplitsPp = -1;
  int vNNodesPp = -1;

  int vNSplitsTg = -1;
  int vNNodesTg = -1;

  const uint32_t cNOutputsPp = std::min(cNTokens, vCparams.vNOutputsMax);

  // reserve pp (prompt processing) graph first so that buffers are only allocated once
  {
    auto *vGf = mGraphReserve(cNTokens, cNSeqs, cNOutputsPp, vMctx.get(), vModel.vHparams.vNoAlloc,
                              vModel.vHparams.vNoAlloc ? lBackendBufExpSize.data() : nullptr);
    if (!vGf) {
      if (vCparams.vPipelineParallel) {
        MGS_LOG_WARN("%s: compute buffer allocation failed, retrying without pipeline parallelism\n", __func__);
        vCparams.vPipelineParallel = false;
        vSched.reset(ggml_backend_sched_new(lBackendPtrs.data(), lBackendBuft.data(), lBackendPtrs.size(), cMaxNodes,
                                            false, vCparams.vOpOffload));
        ggml_backend_sched_set_weight_upload(vSched.get(), vExpertCache ? mgs_expert_cache::mFUploadWeights : nullptr,
                                             vExpertCache.get());
        vGf = mGraphReserve(cNTokens, cNSeqs, cNOutputsPp, vMctx.get());
      }
      if (!vGf) {
        throw std::runtime_error("failed to allocate compute pp buffers");
      }
    }

    vNSplitsPp = ggml_backend_sched_get_n_splits(vSched.get());
    vNNodesPp = ggml_graph_n_nodes(vGf);
  }

  // reserve with tg (token generation) graph to get the number of splits and nodes
  {
    auto *vGf = mGraphReserve(cNSeqs, cNSeqs, cNSeqs, vMctx.get(), vModel.vHparams.vNoAlloc);
    if (!vGf) {
      throw std::runtime_error("failed to allocate compute tg buffers");
    }

    vNSplitsTg = ggml_backend_sched_get_n_splits(vSched.get());
    vNNodesTg = ggml_graph_n_nodes(vGf);
  }

  // reserve again with pp graph to avoid ggml-alloc reallocations during inference
  {
    // TODO: not sure if the following graph would be worst case for multi-stream KV caches:
    //
    // auto * vGf = mGraphReserve(cNTokens, 1, cNTokens, vMctx.get());
    //
    auto *vGf = mGraphReserve(cNTokens, cNSeqs, cNOutputsPp, vMctx.get(), vModel.vHparams.vNoAlloc);
    if (!vGf) {
      throw std::runtime_error("failed to allocate compute pp buffers");
    }
  }

  for (size_t vI = 0; vI < lBackendPtrs.size(); ++vI) {
    ggml_backend_t vBackend = lBackendPtrs[vI];
    ggml_backend_buffer_type_t vBuft = lBackendBuft[vI];
    if (!vModel.vHparams.vNoAlloc) {
      lBackendBufExpSize[vI] = ggml_backend_sched_get_buffer_size(vSched.get(), vBackend);
    }
    if (lBackendBufExpSize[vI] > 1) {
      MGS_LOG_INFO("%s: %10s compute buffer size = %8.2f MiB\n", __func__, ggml_backend_buft_name(vBuft),
                   lBackendBufExpSize[vI] / 1024.0 / 1024.0);
    }
  }

  if (vNNodesPp == vNNodesTg) {
    MGS_LOG_INFO("%s: graph nodes  = %d\n", __func__, vNNodesPp);
  } else {
    MGS_LOG_INFO("%s: graph nodes  = %d (with bs=%d), %d (with bs=1)\n", __func__, vNNodesPp, cNTokens, vNNodesTg);
  }

  if (vNSplitsPp == vNSplitsTg) {
    MGS_LOG_INFO("%s: graph splits = %d\n", __func__, vNSplitsPp);
  } else {
    MGS_LOG_INFO("%s: graph splits = %d (with bs=%d), %d (with bs=1)\n", __func__, vNSplitsPp, cNTokens, vNSplitsTg);
  }

  const int64_t cTEndUs = ggml_time_us();

  MGS_LOG_INFO("%s: reserve took %.2f ms, sched copies = %d\n", __func__, (cTEndUs - cTStartUs) / 1000.0,
               ggml_backend_sched_get_n_copies(vSched.get()));
}

void mgs_context::mSynchronize() {
  if (!vSched) {
    return;
  }

  ggml_backend_sched_synchronize(vSched.get());

  // FIXME: if multiple single tokens are evaluated without a synchronization,
  // the stats will be added to the prompt evaluation stats
  // this should only happen when using batch size 1 to evaluate a batch

  // add the evaluation to the stats
  if (vNQueuedTokens == 1) {
    if (!vCparams.vNoPerf) {
      vTEvalUs += ggml_time_us() - vTComputeStartUs;
    }
    vNEval++;
  } else if (vNQueuedTokens > 1) {
    if (!vCparams.vNoPerf) {
      vTPEvalUs += ggml_time_us() - vTComputeStartUs;
    }
    vNPEval += vNQueuedTokens;
  }

  // get a more accurate load time, upon first eval
  if (vNQueuedTokens > 0 && !vHasEvaluatedOnce) {
    vTLoadUs = ggml_time_us() - vTStartUs;
    vHasEvaluatedOnce = true;
  }

  vNQueuedTokens = 0;
  vTComputeStartUs = 0;
}

const mgs_model &mgs_context::mGetModel() const { return vModel; }

const mgs_cparams &mgs_context::mGetCparams() const { return vCparams; }

ggml_backend_sched_t mgs_context::mGetSched() const { return vSched.get(); }

uint32_t mgs_context::mNCtx() const { return vCparams.vNCtx; }

uint32_t mgs_context::mNCtxSeq() const { return vCparams.vNCtxSeq; }

uint32_t mgs_context::mNBatch() const { return vCparams.vNBatch; }

uint32_t mgs_context::mNUbatch() const { return vCparams.vNUbatch; }

uint32_t mgs_context::mNSeqMax() const { return vCparams.vNSeqMax; }

uint32_t mgs_context::mNThreads() const { return vCparams.vNThreads; }

uint32_t mgs_context::mNThreadsBatch() const { return vCparams.vNThreadsBatch; }

mgs_memory_t mgs_context::mGetMemory() const { return vMemory.get(); }

bool mgs_context::mMemoryUpdate(bool pOptimize) {
  if (!vMemory) {
    return false;
  }

  {
    const auto cMctx = vMemory->mInitUpdate(this, pOptimize);
    switch (cMctx->mGetStatus()) {
    case cMgsMemoryStatusSuccess: {
      // noop
    } break;
    case cMgsMemoryStatusNoUpdate: {
      // no updates need to be performed
      return false;
    }
    case cMgsMemoryStatusFailedPrepare:
    case cMgsMemoryStatusFailedCompute: {
      MGS_LOG_ERROR("%s: failed to prepare memory update\n", __func__);
      return false;
    }
    }

    // reset the previous graph results to make sure that they won't be reused
    // TODO: change the mctx->apply() to return information if a graph reserve is needed
    //       reset the graph result only if the memory module did reset the scheduler
    for (auto &vResult : aGfResPrev) {
      if (vResult) {
        vResult->mReset();
      }
    }
    vGfResPrevActive = nullptr;

    if (!cMctx->mApply()) {
      MGS_LOG_ERROR("%s: failed to apply memory update\n", __func__);
    }
  }

  // if the memory module did any computation, we have to reserve a new worst-case graph
  {
    const auto cMctx = vMemory->mInitFull();
    if (!cMctx) {
      throw std::runtime_error("failed to initialize memory context");
    }

    const uint32_t cNSeqs = vCparams.vNSeqMax;
    const uint32_t cNTokens = std::min(vCparams.vNCtx, vCparams.vNUbatch);

    const uint32_t cNOutputsMax = std::min(cNTokens, vCparams.vNOutputsMax);

    auto *vGf = mGraphReserve(cNTokens, cNSeqs, cNOutputsMax, cMctx.get());
    if (!vGf) {
      MGS_LOG_ERROR("%s: failed to reserve graph after the memory update\n", __func__);
    }
  }

  return true;
}

enum mgs_pooling_type mgs_context::mPoolingType() const { return vCparams.vPoolingType; }

float *mgs_context::mGetLogits() {
  mOutputReorder();

  return vLogits.vData;
}

int64_t mgs_context::mOutputResolveRow(int32_t pI) const {
  int64_t vJ = -1;

  // support negative indices (last output row)
  if (pI < 0) {
    vJ = vNOutputs + pI;
    if (vJ < 0) {
      throw std::runtime_error(fFormat("negative index out of range [0, %d)", vNOutputs));
    }
  } else if ((size_t)pI >= lOutputIds.size()) {
    throw std::runtime_error(fFormat("out of range [0, %zu)", lOutputIds.size()));
  } else {
    // use output_ids to translate the batch token index into a row number
    // that holds this token's data.
    vJ = lOutputIds[pI];
  }

  if (vJ < 0) {
    // the batch token was not configured to output anything
    throw std::runtime_error(fFormat("batch.logits[%d] != true", pI));
  }

  if (vJ >= vNOutputs) {
    throw std::runtime_error(fFormat("corrupt output buffer (j=%" PRId64 ", n_outputs=%d)", vJ, vNOutputs));
  }

  return vJ;
}

float *mgs_context::mGetLogitsIth(int32_t pI) {
  mOutputReorder();

  try {
    if (vLogits.vData == nullptr) {
      throw std::runtime_error("no logits");
    }

    const int64_t cJ = mOutputResolveRow(pI);
    return vLogits.vData + cJ * vModel.vVocab.mNTokens();
  } catch (const std::exception &vErr) {
    MGS_LOG_ERROR("%s: invalid logits id %d, reason: %s\n", __func__, pI, vErr.what());
#ifndef NDEBUG
    GGML_ABORT("fatal error");
#else
    return nullptr;
#endif
  }
}

float *mgs_context::mGetEmbeddings() {
  mOutputReorder();

  return vEmbd.vData;
}

mgs_token *mgs_context::mGetSampledTokens() const { return vSampling.vSampled.vData; }

float *mgs_context::mGetEmbeddingsIth(int32_t pI) {
  mOutputReorder();

  try {
    if (vEmbd.vData == nullptr) {
      throw std::runtime_error("no embeddings");
    }

    const int64_t cJ = mOutputResolveRow(pI);
    const uint32_t cNEmbdOut = vModel.vHparams.mNEmbdOut();
    return vEmbd.vData + cJ * cNEmbdOut;
  } catch (const std::exception &vErr) {
    MGS_LOG_ERROR("%s: invalid embeddings id %d, reason: %s\n", __func__, pI, vErr.what());
#ifndef NDEBUG
    GGML_ABORT("fatal error");
#else
    return nullptr;
#endif
  }
}

float *mgs_context::mGetEmbeddingsSeq(mgs_seq_id pSeqId) {
  auto vIt = dEmbdSeq.find(pSeqId);
  if (vIt == dEmbdSeq.end()) {
    return nullptr;
  }

  return vIt->second.data();
}

float *mgs_context::mGetEmbeddingsNextn() {
  mOutputReorder();

  return vEmbdNextn.vData;
}

float *mgs_context::mGetEmbeddingsNextnIth(int32_t pI) {
  mOutputReorder();

  try {
    if (vEmbdNextn.vData == nullptr) {
      throw std::runtime_error("no nextn embeddings");
    }

    const uint32_t cNEmbd = vModel.vHparams.mNEmbdOut();

    if (!vCparams.vEmbeddingsNextnMasked) {
      // unmasked: nextn rows are stored densely, indexed by raw token position.
      if (pI < 0 || (size_t)(pI + 1) * cNEmbd > vEmbdNextn.vSize) {
        throw std::runtime_error(fFormat("out of range [0, %zu)", vEmbdNextn.vSize / cNEmbd));
      }
      return vEmbdNextn.vData + (size_t)pI * cNEmbd;
    }

    const int64_t cJ = mOutputResolveRow(pI);
    return vEmbdNextn.vData + cJ * cNEmbd;
  } catch (const std::exception &vErr) {
    MGS_LOG_ERROR("%s: invalid nextn embeddings id %d, reason: %s\n", __func__, pI, vErr.what());
#ifndef NDEBUG
    GGML_ABORT("fatal error");
#else
    return nullptr;
#endif
  }
}

float *mgs_context::mGetEmbeddingsLayerInp(uint32_t pLid) {
  mOutputReorder();

  GGML_ASSERT(pLid < lEmbdLayerInp.size() && lEmbdLayerInp[pLid].mHasData());

  return lEmbdLayerInp[pLid].vData;
}

mgs_token mgs_context::mGetSampledTokenIth(int32_t pIdx) {
  mOutputReorder();

  if (!vSampling.vSampled.mHasData()) {
    return MGS_TOKEN_NULL;
  }

  try {
    const int64_t cRow = mOutputResolveRow(pIdx);
    GGML_ASSERT(cRow < (int64_t)vSampling.vSampled.vSize);
    return vSampling.vSampled.vData[cRow];
  } catch (const std::exception &vErr) {
    MGS_LOG_ERROR("%s: invalid backend sampled token id %d, reason: %s\n", __func__, pIdx, vErr.what());
    return MGS_TOKEN_NULL;
  }
}

float *mgs_context::mGetSampledProbsIth(int32_t pIdx) {
  mOutputReorder();

  if (!vSampling.vProbs.mHasData()) {
    return nullptr;
  }

  try {
    const int64_t cRow = mOutputResolveRow(pIdx);
    if ((size_t)cRow >= vSampling.lProbsCount.size() || vSampling.lProbsCount[cRow] == 0) {
      return nullptr;
    }
    return vSampling.vProbs.vData + cRow * vModel.vVocab.mNTokens();
  } catch (const std::exception &vErr) {
    MGS_LOG_ERROR("%s: invalid backend sampled probs id %d, reason: %s\n", __func__, pIdx, vErr.what());
    return nullptr;
  }
}

float *mgs_context::mGetSampledLogitsIth(int32_t pIdx) {
  mOutputReorder();

  if (!vSampling.vLogits.mHasData()) {
    return nullptr;
  }

  try {
    const int64_t cRow = mOutputResolveRow(pIdx);
    if ((size_t)cRow >= vSampling.lLogitsCount.size() || vSampling.lLogitsCount[cRow] == 0) {
      return nullptr;
    }
    return vSampling.vLogits.vData + cRow * vModel.vVocab.mNTokens();
  } catch (const std::exception &vErr) {
    MGS_LOG_ERROR("%s: invalid backend sampled logits id %d, reason: %s\n", __func__, pIdx, vErr.what());
    return nullptr;
  }
}

const mgs_token *mgs_context::mGetSampledCandidatesIth(int32_t pIdx) {
  mOutputReorder();

  try {
    const int64_t cRow = mOutputResolveRow(pIdx);
    if (vSampling.vCandidates.mHasData() && (size_t)cRow < vSampling.lCandidatesCount.size() &&
        vSampling.lCandidatesCount[cRow] > 0) {
      return vSampling.vCandidates.vData + cRow * vModel.vVocab.mNTokens();
    }
  } catch (const std::exception &vErr) {
    // fallback to full vocab list
    GGML_UNUSED(vErr);
  }

  return vSampling.lTokenIdsFullVocab.data();
}

size_t mgs_context::mGetSampledCandidatesCount(int32_t pIdx) {
  mOutputReorder();

  if (!vSampling.vCandidates.mHasData()) {
    return 0;
  }

  try {
    const int64_t cRow = mOutputResolveRow(pIdx);
    if ((size_t)cRow >= vSampling.lCandidatesCount.size()) {
      return 0;
    }
    return vSampling.lCandidatesCount[cRow];
  } catch (const std::exception &vErr) {
    MGS_LOG_ERROR("%s: invalid backend sampled candidates count id %d, reason: %s\n", __func__, pIdx, vErr.what());
    return 0;
  }
}

size_t mgs_context::mGetSampledLogitsCount(int32_t pIdx) {
  mOutputReorder();

  if (!vSampling.vLogits.mHasData()) {
    return vModel.vVocab.mNTokens();
  }

  try {
    const int64_t cRow = mOutputResolveRow(pIdx);
    if ((size_t)cRow >= vSampling.lLogitsCount.size()) {
      return 0;
    }
    return vSampling.lLogitsCount[cRow];
  } catch (const std::exception &vErr) {
    MGS_LOG_ERROR("%s: invalid backend sampled logits count id %d, reason: %s\n", __func__, pIdx, vErr.what());
    return 0;
  }
}

size_t mgs_context::mGetSampledProbsCount(int32_t pIdx) {
  mOutputReorder();

  if (!vSampling.vProbs.mHasData()) {
    return 0;
  }

  try {
    const int64_t cRow = mOutputResolveRow(pIdx);
    if ((size_t)cRow >= vSampling.lProbsCount.size()) {
      return 0;
    }
    return vSampling.lProbsCount[cRow];
  } catch (const std::exception &vErr) {
    MGS_LOG_ERROR("%s: invalid backend sampled probs count id %d, reason: %s\n", __func__, pIdx, vErr.what());
    return 0;
  }
}

void mgs_context::mAttachThreadpool(ggml_threadpool_t pThreadpool, ggml_threadpool_t pThreadpoolBatch) {
  MGS_LOG_DEBUG("%s: call\n", __func__);

  this->vThreadpool = pThreadpool;
  this->vThreadpoolBatch = pThreadpoolBatch ? pThreadpoolBatch : pThreadpool;
}

void mgs_context::mDetachThreadpool() {
  MGS_LOG_DEBUG("%s: call\n", __func__);

  this->vThreadpool = nullptr;
  this->vThreadpoolBatch = nullptr;
}

void mgs_context::mSetNThreads(int32_t pNThreads, int32_t pNThreadsBatch) {
  MGS_LOG_DEBUG("%s: n_threads = %d, n_threads_batch = %d\n", __func__, pNThreads, pNThreadsBatch);

  vCparams.vNThreads = pNThreads;
  vCparams.vNThreadsBatch = pNThreadsBatch;
}

void mgs_context::mSetAbortCallback(bool (*pAbortCallback)(void *pData), void *pAbortCallbackData) {
  MGS_LOG_DEBUG("%s: call\n", __func__);

  this->vAbortCallback = pAbortCallback;
  this->vAbortCallbackData = pAbortCallbackData;

  for (auto &vBackend : lBackends) {
    auto *vReg = ggml_backend_dev_backend_reg(ggml_backend_get_device(vBackend.get()));
    if (vReg) {
      auto *vSetAbortCallbackFn =
          (ggml_backend_set_abort_callback_t)ggml_backend_reg_get_proc_address(vReg, "ggml_backend_set_abort_callback");
      if (vSetAbortCallbackFn) {
        vSetAbortCallbackFn(vBackend.get(), this->vAbortCallback, this->vAbortCallbackData);
      }
    }
  }
}

void mgs_context::mSetEmbeddings(bool pValue) {
  MGS_LOG_DEBUG("%s: value = %d\n", __func__, pValue);

  vCparams.vEmbeddings = pValue;

  // TODO: not sure yet if we want to reserve here
  // vSchedNeedReserve = true;
}

void mgs_context::mSetEmbeddingsNextn(bool pValue, bool pMasked) {
  MGS_LOG_DEBUG("%s: value = %d, masked = %d\n", __func__, pValue, pMasked);

  vCparams.vEmbeddingsNextn = pValue;
  vCparams.vEmbeddingsNextnMasked = pMasked;
}

void mgs_context::mSetEmbeddingsLayerInp(uint32_t pLid, bool pEnable) {
  MGS_LOG_DEBUG("%s: lid = %d, enable = %d\n", __func__, pLid, pEnable);

  GGML_ASSERT(pLid <= vModel.vHparams.mNLayer());

  vCparams.lEmbeddingsLayerInp[pLid] = pEnable;

  // note: without this reserve, the draft acceptance drops to zero. not sure why - this is unexpected
  vSchedNeedReserve = true;
}

void mgs_context::mSetNextnLayerOffset(int32_t pOffset) { vCparams.vNextnLayerOffset = pOffset; }

void mgs_context::mSetCausalAttn(bool pValue) {
  MGS_LOG_DEBUG("%s: value = %d\n", __func__, pValue);

  if (vCparams.vCausalAttn == pValue) {
    return;
  }

  vCparams.vCausalAttn = pValue;

  vSchedNeedReserve = true;
}

bool mgs_context::mSetSampler(mgs_seq_id pSeqId, mgs_sampler *pSampler) {
  if (!pSampler && vSampling.dSamplers.count(pSeqId) == 0) {
    return true;
  }

  MGS_LOG_DEBUG("%s: seq_id = %d, sampler = %p\n", __func__, (int)pSeqId, (void *)pSampler);

  if (pSampler && vModel.mSplitMode() == MGS_SPLIT_MODE_TENSOR) {
    static bool vWarned = false;
    if (!vWarned) {
      MGS_LOG_WARN("%s: backend sampling not supported with SPLIT_MODE_TENSOR; using CPU\n", __func__);
      vWarned = true;
    }
    if (vSampling.dSamplers.count(pSeqId) > 0) {
      vSchedNeedReserve = true;
    }
    vSampling.dSamplers.erase(pSeqId);
    return false;
  }

  const bool cCanOffload =
      pSampler && pSampler->iface->backend_init && pSampler->iface->backend_apply && mgs_sampler_chain_n(pSampler) > 0;

  if (pSampler && cCanOffload) {
    auto *vBuft = ggml_backend_dev_buffer_type(vModel.mDevOutput());

    pSampler->iface->backend_init(pSampler, vBuft);

    vSampling.dSamplers[pSeqId] = pSampler;

    vSchedNeedReserve = true;

    return true;
  }

  if (pSampler && !cCanOffload) {
    MGS_LOG_WARN("%s: sampler '%s' for seq_id = %d, cannot be offloaded to the backend\n", __func__,
                 mgs_sampler_name(pSampler), pSeqId);

    if (vSampling.dSamplers.count(pSeqId) > 0) {
      vSchedNeedReserve = true;
    }

    vSampling.dSamplers.erase(pSeqId);

    return false;
  }

  vSampling.dSamplers.erase(pSeqId);

  vSchedNeedReserve = true;

  return true;
}

void mgs_context::mSetAdaptersLora(mgs_adapter_lora **pAdapters, size_t pNAdapters, float *pScales) {
  MGS_LOG_DEBUG("%s: adapters = %p\n", __func__, (void *)pAdapters);

  if (mAdaptersLoraAreSame(pAdapters, pNAdapters, pScales)) {
    return;
  }

  vLoras.reset(new mgs_adapter_loras());

  for (size_t vI = 0; vI < pNAdapters; vI++) {
    if (pScales[vI] != 0.0f) {
      vLoras->insert({pAdapters[vI], pScales[vI]});
    }
  }

  vSchedNeedReserve = true;
}

bool mgs_context::mAdaptersLoraAreSame(mgs_adapter_lora **pAdapters, size_t pNAdapters, float *pScales) {
  MGS_LOG_DEBUG("%s: adapters = %p\n", __func__, (void *)pAdapters);

  // Adapters with a zero scale are never added to `vLoras`, so also ignore them for the comparison.
  size_t vNNonZero = 0;

  for (size_t vI = 0; vI < pNAdapters; vI++) {
    if (pScales[vI] == 0.0f) {
      continue;
    }
    vNNonZero++;

    auto vIt = vLoras->find(pAdapters[vI]);

    if (vIt == vLoras->end() || vIt->second != pScales[vI]) {
      return false;
    }
  }

  if (vNNonZero != vLoras->size()) {
    return false;
  }

  return true;
}

bool mgs_context::mSetAdapterCvec(const float *pData, size_t pLen, int32_t pNEmbd, int32_t pIlStart, int32_t pIlEnd) {
  MGS_LOG_DEBUG("%s: il_start = %d, il_end = %d\n", __func__, pIlStart, pIlEnd);

  bool vRes = vCvec->mApply(vModel, pData, pLen, pNEmbd, pIlStart, pIlEnd);

  vSchedNeedReserve = true;

  return vRes;
}

llm_graph_result *mgs_context::mProcessUbatch(const mgs_ubatch &pUbatch, llm_graph_type pGtype,
                                              mgs_memory_context_i *pMctx, ggml_status &pRet) {
  if (pMctx && !pMctx->mApply()) {
    MGS_LOG_ERROR("%s: failed to apply memory context\n", __func__);
    pRet = GGML_STATUS_FAILED;
    return nullptr;
  }

  auto *vRes = mGetGfResPrev();
  auto *vGf = vRes->mGetGf();

  // the new graph parameters
  // in order to correctly reuse a graph, it's full topology has to be uniquely determined by these parameters
  const auto cGparams = mGraphParams(vRes, pUbatch, pMctx, pGtype);

  if (!vGraphReuseDisable && vGfResPrevActive == vRes && vRes->mCanReuse(cGparams)) {
    // MGS_LOG_DEBUG("%s: reusing previous graph\n", __func__);

    // with pipeline parallelism, the previous graph_compute_async may still be running
    // on the GPU. we must synchronize before set_inputs to avoid overwriting input tensors
    // that the previous compute is still reading.
    if (vCparams.vPipelineParallel) {
      ggml_backend_sched_synchronize(vSched.get());
    }

    vNReused++;
  } else {
    vGfResPrevActive = nullptr;
    vRes->mReset();

    ggml_backend_sched_reset(vSched.get());
    ggml_backend_sched_set_eval_callback(vSched.get(), vCparams.vCbEval, vCparams.vCbEvalUserData);

    // const auto cTStartUs = ggml_time_us();

    vGf = vModel.mBuildGraph(cGparams);

    // MGS_LOG_INFO("graph build time: %.3f ms\n", (ggml_time_us() - cTStartUs)/1000.0);

    if (!vGf) {
      MGS_LOG_ERROR("%s: failed to initialize graph\n", __func__);
      pRet = GGML_STATUS_FAILED;
      return nullptr;
    }

    if (!ggml_backend_sched_alloc_graph(vSched.get(), vGf)) {
      MGS_LOG_ERROR("%s: failed to allocate graph\n", __func__);
      pRet = GGML_STATUS_ALLOC_FAILED;
      return nullptr;
    }
    vGfResPrevActive = vRes;
  }

  // set the input data for the input tensors
  {
    // const auto cTStartUs = ggml_time_us();

    // FIXME this call causes a crash if any model inputs were not used in the graph and were therefore not allocated
    vRes->mSetInputs(&pUbatch);

    // MGS_LOG_INFO("graph set inputs time: %.3f ms\n", (ggml_time_us() - cTStartUs)/1000.0);
  }

  const auto cStatus = mGraphCompute(vRes->mGetGf(), pUbatch.vNTokens > 1);
  if (cStatus != GGML_STATUS_SUCCESS) {
    MGS_LOG_ERROR("%s: failed to compute graph, compute status: %d\n", __func__, cStatus);
    pRet = cStatus;
    return nullptr;
  }

  pRet = GGML_STATUS_SUCCESS;

  return vRes;
}

int mgs_context::mEncode(const mgs_batch &pBatchInp) {
  // MTP hook batches carry both token (next-token id) and embd (h_nextn row),
  // so accept either present rather than requiring exactly one.
  GGML_ASSERT(pBatchInp.token || pBatchInp.embd);

  if (pBatchInp.n_tokens == 0) {
    MGS_LOG_ERROR("%s: n_tokens == 0\n", __func__);
    return -1;
  }

  const auto &vHparams = vModel.vHparams;

  // eagle3/DFlash: features as encoder input, and non-draft paths fall back to model's input dim
  const int64_t cNEmbd = vHparams.mNEmbdInpEnc();
  const int64_t cNVocab = vModel.vVocab.mNTokens();

  // note: during encode, we always pass the full sequence starting from pos = 0
  if (!vBalloc->mInit(pBatchInp, vModel.vVocab, nullptr, cNEmbd, vCparams.vKvUnified ? MGS_MAX_SEQ : vCparams.vNSeqMax,
                      true)) {
    MGS_LOG_ERROR("%s: failed to initialize batch\n", __func__);
    return -1;
  }

  const uint32_t cNTokens = vBalloc->mGetNTokens();

  // [TAG_NO_CACHE_PAD]
  // TODO: add new split mode where we pad the input sequences so that ubatch.mEqualSeqs() == true
  const mgs_ubatch cUbatch = vBalloc->mSplitSimple(cNTokens);

  // micro-batching is not possible for non-causal encoding, so we process the batch in a single shot
  GGML_ASSERT(vCparams.vNUbatch >= cNTokens && "encoder requires n_ubatch >= n_tokens");

  // TODO: this clear of the buffer can easily be forgotten - need something better
  // sync first so any in-flight async copies into embd_seq complete before it is freed
  if (!dEmbdSeq.empty()) {
    mSynchronize();
  }
  dEmbdSeq.clear();

  if (vTComputeStartUs == 0) {
    vTComputeStartUs = ggml_time_us();
  }

  mSchedReserve();

  vNQueuedTokens += cNTokens;

  // reserve output buffer
  if (mOutputReserve(cNTokens) < cNTokens) {
    MGS_LOG_ERROR("%s: could not reserve space for batch with %u outputs\n", __func__, cNTokens);
    return -2;
  };

  for (uint32_t vI = 0; vI < cNTokens; ++vI) {
    lOutputIds[vI] = vI;
  }

  vNOutputs = cNTokens;

  const auto cCausalAttnOrg = vCparams.vCausalAttn;

  // always use non-causal attention for encoder graphs
  // TODO: this is a tmp solution until we have a proper way to support enc-dec models
  vCparams.vCausalAttn = false;

  ggml_status vStatus;
  const auto *vRes = mProcessUbatch(cUbatch, cLlmGraphTypeEncoder, nullptr, vStatus);

  vCparams.vCausalAttn = cCausalAttnOrg;

  if (!vRes) {
    switch (vStatus) {
    case GGML_STATUS_ABORTED:
      return 2;
    case GGML_STATUS_ALLOC_FAILED:
      return -2;
    case GGML_STATUS_FAILED:
      return -3;
    case GGML_STATUS_SUCCESS:
      GGML_ABORT("should not happen");
    }
  }

  auto *vTLogits = vRes->mGetLogits();
  auto *vTEmbd = vRes->mGetEmbdPooled() ? vRes->mGetEmbdPooled() : vRes->mGetEmbd();
  auto *vTHNextn = vCparams.vEmbeddingsNextn ? vRes->mGetHNextn() : nullptr;

  // extract logits
  if (vLogits.vData && vTLogits) {
    ggml_backend_t vBackendRes = ggml_backend_sched_get_tensor_backend(vSched.get(), vTLogits);
    GGML_ASSERT(vBackendRes != nullptr);
    GGML_ASSERT(vLogits.vData != nullptr);

    ggml_backend_tensor_get_async(vBackendRes, vTLogits, vLogits.vData, 0, cNTokens * cNVocab * sizeof(float));
  }

  // extract embeddings
  if (vEmbd.vData && vTEmbd) {
    ggml_backend_t vBackendEmbd = ggml_backend_sched_get_tensor_backend(vSched.get(), vTEmbd);
    GGML_ASSERT(vBackendEmbd != nullptr);

    switch (vCparams.vPoolingType) {
    case MGS_POOLING_TYPE_NONE: {
      // extract token embeddings
      GGML_ASSERT(vEmbd.vData != nullptr);
      const uint32_t cNEmbdOut = vHparams.mNEmbdOut();

      GGML_ASSERT(cNTokens * cNEmbdOut <= (int64_t)vEmbd.vSize);
      ggml_backend_tensor_get_async(vBackendEmbd, vTEmbd, vEmbd.vData, 0, cNTokens * cNEmbdOut * sizeof(float));
    } break;
    case MGS_POOLING_TYPE_MEAN:
    case MGS_POOLING_TYPE_CLS:
    case MGS_POOLING_TYPE_LAST: {
      // extract sequence embeddings
      auto &vEmbdSeqOut = dEmbdSeq;

      for (uint32_t vS = 0; vS < cUbatch.vNSeqsUnq; ++vS) {
        const mgs_seq_id cSeqId = cUbatch.vSeqIdUnq[vS];
        const int32_t cSeqIdx = cUbatch.vSeqIdx[cSeqId];

        // use n_embd_out (not n_embd_inp) - the pooled embedding has the model's
        // output dimension, which differs from input dimension for deepstack models (e.g. qwen3vl)
        const uint32_t cNEmbdOut = vHparams.mNEmbdOut();
        vEmbdSeqOut[cSeqId].resize(cNEmbdOut);
        ggml_backend_tensor_get_async(vBackendEmbd, vTEmbd, vEmbdSeqOut[cSeqId].data(),
                                      (cNEmbdOut * cSeqIdx) * sizeof(float), cNEmbdOut * sizeof(float));
      }
    } break;
    case MGS_POOLING_TYPE_RANK: {
      // extract the rerank score - n_cls_out floats per sequence
      auto &vEmbdSeqOut = dEmbdSeq;

      const uint32_t cNClsOut = vHparams.vNClsOut;

      for (uint32_t vS = 0; vS < cUbatch.vNSeqsUnq; ++vS) {
        const mgs_seq_id cSeqId = cUbatch.vSeqIdUnq[vS];
        const int32_t cSeqIdx = cUbatch.vSeqIdx[cSeqId];

        vEmbdSeqOut[cSeqId].resize(cNClsOut);
        ggml_backend_tensor_get_async(vBackendEmbd, vTEmbd, vEmbdSeqOut[cSeqId].data(),
                                      (cNClsOut * cSeqIdx) * sizeof(float), cNClsOut * sizeof(float));
      }
    } break;
    case MGS_POOLING_TYPE_UNSPECIFIED: {
      GGML_ABORT("unknown pooling type");
    }
    }
  }

  // extract nextn embeddings (hidden state before the final output norm)
  if (vEmbdNextn.vData && vTHNextn && vCparams.vPoolingType == MGS_POOLING_TYPE_NONE) {
    ggml_backend_t vBackendH = ggml_backend_sched_get_tensor_backend(vSched.get(), vTHNextn);
    GGML_ASSERT(vBackendH != nullptr);

    const uint32_t cNEmbd = vHparams.mNEmbdOut();
    GGML_ASSERT(cNTokens * cNEmbd <= (int64_t)vEmbdNextn.vSize);
    ggml_backend_tensor_get_async(vBackendH, vTHNextn, vEmbdNextn.vData, 0, cNTokens * cNEmbd * sizeof(float));
  }

  // TODO: hacky solution
  if (vModel.vArch == cLlmArchT5 && vTEmbd) {
    // vCross.vTEmbd = vTEmbd;

    mSynchronize();

    vCross.vNEmbd = vTEmbd->ne[0];
    vCross.vNEnc = vTEmbd->ne[1];
    vCross.lVEmbd.resize(vCross.vNEmbd * vCross.vNEnc);
    memcpy(vCross.lVEmbd.data(), vEmbd.vData, ggml_nbytes(vTEmbd));

    const auto &vBatch = vBalloc->mGetBatch();

    // remember the sequence ids used during the encoding - needed for cross attention later
    vCross.sSeqIdsEnc.resize(cNTokens);
    for (uint32_t vI = 0; vI < cNTokens; vI++) {
      vCross.sSeqIdsEnc[vI].clear();

      for (int vS = 0; vS < vBatch.n_seq_id[vI]; vS++) {
        const mgs_seq_id cSeqId = vBatch.seq_id[vI][vS];

        vCross.sSeqIdsEnc[vI].insert(cSeqId);
      }
    }
  }

  return 0;
}

static std::map<mgs_seq_id, uint32_t> fBuildSeqToOutputRow(const mgs_ubatch &pUbatch, uint32_t pRowOffset) {
  std::map<mgs_seq_id, uint32_t> dSeqToRow;
  // how many output tokens we have seen so far for this ubatch.
  uint32_t vLocal = 0;
  for (uint32_t vI = 0; vI < pUbatch.vNTokens; ++vI) {
    // skip tokens that are not output.
    if (!pUbatch.vOutput[vI]) {
      continue;
    }

    const mgs_seq_id cSeqId = pUbatch.vSeqId[vI][0];
    // row_offset is the number of output tokens before this ubatch.
    dSeqToRow[cSeqId] = pRowOffset + vLocal;
    ++vLocal;
  }
  return dSeqToRow;
}

static void fCopyTensorAsyncInts(const std::map<mgs_seq_id, ggml_tensor *> &pTensorMap,
                                 const buffer_view<mgs_token> &pSampled,
                                 const std::map<mgs_seq_id, uint32_t> &pSeqToRow, ggml_backend_sched_t pSched) {
  if (!pSampled.mHasData()) {
    return;
  }

  for (const auto &[vSeqId, vTensor] : pTensorMap) {
    auto vIt = pSeqToRow.find(vSeqId);
    if (vIt == pSeqToRow.end()) {
      continue;
    }

    const uint32_t cRow = vIt->second;
    GGML_ASSERT(cRow < pSampled.vSize);

    GGML_ASSERT(ggml_is_contiguous(vTensor) && "sampled tokens tensor must be contiguous for async copy");

    ggml_backend_t vBackend = ggml_backend_sched_get_tensor_backend(pSched, vTensor);
    ggml_backend_tensor_get_async(vBackend, vTensor, pSampled.vData + cRow, 0, sizeof(pSampled.vData[cRow]));
  }
}

static void fCopyTensorAsyncFloats(const std::map<mgs_seq_id, ggml_tensor *> &pTensorMap,
                                   const buffer_view<float> &pDst, size_t pStride, std::vector<uint32_t> &pCounts,
                                   const std::map<mgs_seq_id, uint32_t> &pSeqToRow, ggml_backend_sched_t pSched) {
  if (!pDst.mHasData()) {
    return;
  }

  for (const auto &[vSeqId, vTensor] : pTensorMap) {
    auto vIt = pSeqToRow.find(vSeqId);
    if (vIt == pSeqToRow.end()) {
      continue;
    }

    const uint32_t cRow = vIt->second;
    GGML_ASSERT(cRow < pCounts.size());

    GGML_ASSERT(ggml_is_contiguous(vTensor) && "logits/probs tensor must be contiguous for async copy");

    ggml_backend_t vBackend = ggml_backend_sched_get_tensor_backend(pSched, vTensor);
    float *vRowPtr = pDst.vData + (size_t)cRow * pStride;
    ggml_backend_tensor_get_async(vBackend, vTensor, vRowPtr, 0, ggml_nbytes(vTensor));

    // Update the actual number of logits/probabilities that were written for this row.
    pCounts[cRow] = ggml_nelements(vTensor);
  }
}

static void fCopyTensorAsyncCandidates(const std::map<mgs_seq_id, ggml_tensor *> &pTensorMap,
                                       const buffer_view<mgs_token> &pDst, size_t pStride,
                                       std::vector<uint32_t> &pCounts, const std::map<mgs_seq_id, uint32_t> &pSeqToRow,
                                       ggml_backend_sched_t pSched) {
  if (!pDst.mHasData()) {
    return;
  }

  for (const auto &[vSeqId, vTensor] : pTensorMap) {
    auto vIt = pSeqToRow.find(vSeqId);
    if (vIt == pSeqToRow.end()) {
      continue;
    }

    const uint32_t cRow = vIt->second;
    GGML_ASSERT(cRow < pCounts.size());

    GGML_ASSERT(ggml_is_contiguous(vTensor) && "candidates tensor must be contiguous for async copy");

    ggml_backend_t vBackend = ggml_backend_sched_get_tensor_backend(pSched, vTensor);
    mgs_token *vRowPtr = pDst.vData + (size_t)cRow * pStride;
    ggml_backend_tensor_get_async(vBackend, vTensor, vRowPtr, 0, ggml_nbytes(vTensor));

    // Update the actual number of candidates that were written.
    pCounts[cRow] = ggml_nelements(vTensor);
  }
}

static bool fNeedsRawLogits(const mgs_ubatch &pUbatch, const std::map<mgs_seq_id, mgs_sampler *> &pSamplers) {
  for (uint32_t vI = 0; vI < pUbatch.vNTokens; vI++) {
    if (!pUbatch.vOutput[vI]) {
      continue;
    }

    // Check if the output token has at least one sequence without a backend sampler.
    for (int32_t vJ = 0; vJ < pUbatch.vNSeqId[vI]; ++vJ) {
      mgs_seq_id vSeqId = pUbatch.vSeqId[vI][vJ];
      if (pSamplers.find(vSeqId) == pSamplers.end()) {
        return true;
      }
    }
  }
  return false; // all sequences use backend sampling
}

int mgs_context::mDecode(const mgs_batch &pBatchInp) {
  // MTP hook batches carry both token (next-token id) and embd (h_nextn row),
  // so accept either present rather than requiring exactly one.
  GGML_ASSERT(pBatchInp.token || pBatchInp.embd);

  if (!vMemory) {
    MGS_LOG_DEBUG("%s: cannot decode batches with this context (calling encode() instead)\n", __func__);
    return mEncode(pBatchInp);
  }

  if (pBatchInp.n_tokens == 0) {
    MGS_LOG_ERROR("%s: n_tokens == 0\n", __func__);
    return -1;
  }

  const auto &vVocab = vModel.vVocab;
  const auto &vHparams = vModel.vHparams;

  const int64_t cNVocab = vVocab.mNTokens();
  const bool cMtpEmbd = vCparams.vCtxType == MGS_CONTEXT_TYPE_MTP && pBatchInp.embd;
  const int64_t cNEmbd = cMtpEmbd ? vHparams.mNEmbdOut() : vHparams.mNEmbdInp();

  // when computing embeddings, all tokens are output
  const bool cOutputAll = vCparams.vEmbeddings;
  const bool cHasSamplers = !vSampling.dSamplers.empty();

  const uint32_t cNSeqMax = vCparams.vKvUnified ? MGS_MAX_SEQ : vCparams.vNSeqMax;

  // TODO: avoid this workaround in the future
  if (cHasSamplers && pBatchInp.logits) {
    std::vector<int32_t> lSeqOutputCount(cNSeqMax, 0);

    for (int32_t vI = 0; vI < pBatchInp.n_tokens; ++vI) {
      if (pBatchInp.logits[vI] == 0) {
        continue;
      }

      const int cNs = pBatchInp.n_seq_id ? pBatchInp.n_seq_id[vI] : 1;

      for (int32_t vS = 0; vS < cNs; ++vS) {
        const mgs_seq_id cSeqId = pBatchInp.seq_id ? pBatchInp.seq_id[vI][vS] : 0;

        lSeqOutputCount[cSeqId]++;
        if (lSeqOutputCount[cSeqId] > 1) {
          MGS_LOG_ERROR("%s: backend sampling requires at most one output token per sequence (seq_id %d had %d)\n",
                        __func__, cSeqId, lSeqOutputCount[cSeqId]);
          return -1;
        }
      }
    }
  }

  if (!vBalloc->mInit(pBatchInp, vVocab, vMemory.get(), cNEmbd, cNSeqMax, cOutputAll)) {
    MGS_LOG_ERROR("%s: failed to initialize batch\n", __func__);
    return -1;
  }

  const uint32_t cNTokensAll = vBalloc->mGetNTokens();
  const uint32_t cNOutputsAll = vBalloc->mGetNOutputs();

  if (cOutputAll) {
    // require that all tokens are output
    if (cNOutputsAll != cNTokensAll) {
      MGS_LOG_ERROR(
          "%s: pooled embedding requires that all tokens are output (n_outputs_all = %d, n_tokens_all = %d)\n",
          __func__, cNOutputsAll, cNTokensAll);
      return -1;
    }
  }

  GGML_ASSERT(cNTokensAll <= vCparams.vNBatch);

  GGML_ASSERT((vCparams.vCausalAttn || vCparams.vNUbatch >= cNTokensAll) &&
              "non-causal attention requires n_ubatch >= n_tokens");

  // TODO: this clear of the buffer can easily be forgotten - need something better
  // sync first so any in-flight async copies into embd_seq complete before it is freed
  if (!dEmbdSeq.empty()) {
    mSynchronize();
  }
  dEmbdSeq.clear();

  if (vTComputeStartUs == 0) {
    vTComputeStartUs = ggml_time_us();
  }
  vNQueuedTokens += cNTokensAll;

  lOutputSwaps.clear();
  lEmbdBatchIdxs.clear();

  // The MoE expert cache takes the device memory that remains once every context
  // sharing the model has reserved its own, so it is created at the first regular decode.
  // A size requested at run time rebuilds it here, between two micro-batches. Pipeline
  // parallelism keeps several micro-batches in flight, which the per-layer identifiers of
  // the cache do not support.
  if (vCparams.vCtxType == MGS_CONTEXT_TYPE_DEFAULT && !vModel.vHparams.vNoAlloc && !vCparams.vPipelineParallel) {
    if (vExpertCacheRequestMib != -2) {
      mSynchronize();
      const bool vHadCache = vExpertCache != nullptr;
      vExpertCache.reset();
      vExpertCache = mgs_expert_cache::mFCreate(vModel, lBackendPtrs, vExpertCacheRequestMib);
      vExpertCacheRequestMib = -2;
      vExpertCacheChecked = true;
      vSchedNeedReserve = vSchedNeedReserve || vHadCache || vExpertCache != nullptr;
    } else if (!vExpertCacheChecked) {
      vExpertCacheChecked = true;
      mSynchronize();
      vExpertCache = mgs_expert_cache::mFCreate(vModel, lBackendPtrs);
      if (vExpertCache) {
        vSchedNeedReserve = true;
      }
    }
  }

  mSchedReserve();

  bool vDidOptimize = false;

  // handle any pending shifts/copies
  mMemoryUpdate(false);

  mgs_memory_context_ptr vMctx;

  while (true) {
    vMctx = vMemory->mInitBatch(*vBalloc, vCparams.vNUbatch, cOutputAll);
    if (!vMctx) {
      return -2;
    }

    switch (vMctx->mGetStatus()) {
    case cMgsMemoryStatusSuccess: {
    } break;
    case cMgsMemoryStatusNoUpdate: {
      MGS_LOG_ERROR("%s: unexpected memory context status: %d\n", __func__, vMctx->mGetStatus());

      return -2;
    }
    case cMgsMemoryStatusFailedPrepare: {
      if (!vDidOptimize) {
        vDidOptimize = true;

        if (mMemoryUpdate(true)) {
          MGS_LOG_DEBUG("%s: retrying batch size %d after cache optimization\n", __func__, vBalloc->mGetNTokens());

          continue;
        }
      }

      MGS_LOG_WARN("%s: failed to find a memory slot for batch of size %d\n", __func__, vBalloc->mGetNTokens());

      return 1;
    }
    case cMgsMemoryStatusFailedCompute: {
      MGS_LOG_ERROR("%s: compute failed while preparing batch of size %d\n", __func__, vBalloc->mGetNTokens());

      return -2;
    }
    }

    break;
  }

  // reserve output buffer
  if (mOutputReserve(cNOutputsAll) < cNOutputsAll) {
    MGS_LOG_ERROR("%s: could not reserve space for batch with %d outputs\n", __func__, cNOutputsAll);
    return -2;
  };

  int64_t vNOutputsPrev = 0;
  int64_t vNTokensPrev = 0;

  do {
    const auto &vUbatch = vMctx->mGetUbatch();

    // count the outputs in this ubatch
    {
      int32_t vNOutputsNew = 0;

      if (cNOutputsAll == cNTokensAll) {
        vNOutputsNew = vUbatch.vNTokens;
      } else {
        for (uint32_t vI = 0; vI < vUbatch.vNTokens; vI++) {
          vNOutputsNew += (int32_t)(vUbatch.vOutput[vI] != 0);
        }
      }

      // needs to happen before the graph is built
      vNOutputs = vNOutputsNew;
    }

    ggml_status vStatus;

    const auto *vRes = mProcessUbatch(vUbatch, fCtxTypeToGraphType(vCparams.vCtxType), vMctx.get(), vStatus);

    if (!vRes) {
      // the last ubatch failed or was aborted -> remove all positions of that ubatch from the memory module
      mgs_pos aPosMin[MGS_MAX_SEQ];
      for (int vS = 0; vS < MGS_MAX_SEQ; ++vS) {
        aPosMin[vS] = std::numeric_limits<mgs_pos>::max();
      }

      for (uint32_t vI = 0; vI < vUbatch.vNTokens; ++vI) {
        const auto &vSeqId = vUbatch.vSeqId[vI][0];

        aPosMin[vSeqId] = std::min(aPosMin[vSeqId], vUbatch.vPos[vI]);
      }

      for (int vS = 0; vS < MGS_MAX_SEQ; ++vS) {
        if (aPosMin[vS] == std::numeric_limits<mgs_pos>::max()) {
          continue;
        }

        MGS_LOG_WARN("%s: removing memory module entries for seq_id = %d, pos = [%d, +inf)\n", __func__, vS,
                     aPosMin[vS]);

        vMemory->mSeqRm(vS, aPosMin[vS], -1);
      }

      switch (vStatus) {
      case GGML_STATUS_ABORTED:
        return 2;
      case GGML_STATUS_ALLOC_FAILED:
        return -2;
      case GGML_STATUS_FAILED:
        return -3;
      case GGML_STATUS_SUCCESS:
        GGML_ABORT("should not happen");
      }
    }

    // plot the computation graph in dot format (for debugging purposes)
    // if (n_past%100 == 0) {
    //    ggml_graph_dump_dot(vGf, NULL, "mgs.dot");
    //}

    auto *vTLogits = vRes->mGetLogits();
    auto *vTEmbd = vCparams.vEmbeddings ? vRes->mGetEmbd() : nullptr;
    auto *vTHNextn = vCparams.vEmbeddingsNextn ? vRes->mGetHNextn() : nullptr;

    if (vTEmbd && vRes->mGetEmbdPooled()) {
      vTEmbd = vRes->mGetEmbdPooled();
    }

    // extract logits
    if (vLogits.vData && vTLogits && vNOutputs > 0 && fNeedsRawLogits(vUbatch, vSampling.dSamplers)) {
      ggml_backend_t vBackendRes = ggml_backend_sched_get_tensor_backend(vSched.get(), vTLogits);
      GGML_ASSERT(vBackendRes != nullptr);
      GGML_ASSERT(vLogits.vData != nullptr);

      float *vLogitsOut = vLogits.vData + vNOutputsPrev * cNVocab;

      if (vNOutputs) {
        GGML_ASSERT(vNOutputsPrev + vNOutputs <= cNOutputsAll);
        GGML_ASSERT((vNOutputsPrev + vNOutputs) * cNVocab <= (int64_t)vLogits.vSize);
        ggml_backend_tensor_get_async(vBackendRes, vTLogits, vLogitsOut, 0, vNOutputs * cNVocab * sizeof(float));
      }
    }

    // extract embeddings
    if (vEmbd.vData && vTEmbd && vNOutputs > 0) {
      ggml_backend_t vBackendEmbd = ggml_backend_sched_get_tensor_backend(vSched.get(), vTEmbd);
      GGML_ASSERT(vBackendEmbd != nullptr);

      switch (vCparams.vPoolingType) {
      case MGS_POOLING_TYPE_NONE: {
        // extract token embeddings
        GGML_ASSERT(vEmbd.vData != nullptr);
        const uint32_t cNEmbdOut = vHparams.mNEmbdOut();
        float *vEmbdOut = vEmbd.vData + vNOutputsPrev * cNEmbdOut;

        if (vNOutputs) {
          GGML_ASSERT(vNOutputsPrev + vNOutputs <= cNOutputsAll);
          GGML_ASSERT((vNOutputsPrev + vNOutputs) * cNEmbdOut <= (int64_t)vEmbd.vSize);
          ggml_backend_tensor_get_async(vBackendEmbd, vTEmbd, vEmbdOut, 0, vNOutputs * cNEmbdOut * sizeof(float));
        }
      } break;
      case MGS_POOLING_TYPE_MEAN:
      case MGS_POOLING_TYPE_CLS:
      case MGS_POOLING_TYPE_LAST: {
        // extract sequence embeddings (cleared before processing each batch)
        auto &vEmbdSeqOut = dEmbdSeq;

        // use n_embd_out (not n_embd_inp) - the pooled embedding has the model's
        // output dimension, which differs from input dimension for deepstack models (e.g. qwen3vl)
        const uint32_t cNEmbdOut = vHparams.mNEmbdOut();

        for (uint32_t vS = 0; vS < vUbatch.vNSeqsUnq; ++vS) {
          const mgs_seq_id cSeqId = vUbatch.vSeqIdUnq[vS];
          const int32_t cSeqIdx = vUbatch.vSeqIdx[cSeqId];

          vEmbdSeqOut[cSeqId].resize(cNEmbdOut);
          ggml_backend_tensor_get_async(vBackendEmbd, vTEmbd, vEmbdSeqOut[cSeqId].data(),
                                        (cNEmbdOut * cSeqIdx) * sizeof(float), cNEmbdOut * sizeof(float));
        }
      } break;
      case MGS_POOLING_TYPE_RANK: {
        // extract the rerank score - n_cls_out floats per sequence
        auto &vEmbdSeqOut = dEmbdSeq;

        const uint32_t cNClsOut = vHparams.vNClsOut;

        for (uint32_t vS = 0; vS < vUbatch.vNSeqsUnq; ++vS) {
          const mgs_seq_id cSeqId = vUbatch.vSeqIdUnq[vS];
          const int32_t cSeqIdx = vUbatch.vSeqIdx[cSeqId];

          vEmbdSeqOut[cSeqId].resize(cNClsOut);
          ggml_backend_tensor_get_async(vBackendEmbd, vTEmbd, vEmbdSeqOut[cSeqId].data(),
                                        (cNClsOut * cSeqIdx) * sizeof(float), cNClsOut * sizeof(float));
        }
      } break;
      case MGS_POOLING_TYPE_UNSPECIFIED: {
        GGML_ABORT("unknown pooling type");
      }
      }
    }

    // [TAG_EXTRACT_TARGET_EMBEDDINGS]
    bool vExtractAllIdxs = mExtractLayerInputs(vRes, vNTokensPrev, vUbatch.vNTokens);

    // extract nextn embeddings before
    // only meaningful in MGS_POOLING_TYPE_NONE (per-token); other pooling modes are ignored.
    {
      const bool cMasked = vCparams.vEmbeddingsNextnMasked;
      const int64_t cNRows = cMasked ? vNOutputs : (int64_t)vUbatch.vNTokens;
      const int64_t cOffset = cMasked ? vNOutputsPrev : vNTokensPrev;

      if (vEmbdNextn.vData && vTHNextn && cNRows > 0 && vCparams.vPoolingType == MGS_POOLING_TYPE_NONE) {
        ggml_backend_t vBackendH = ggml_backend_sched_get_tensor_backend(vSched.get(), vTHNextn);
        GGML_ASSERT(vBackendH != nullptr);

        const uint32_t cNEmbd = vHparams.mNEmbdOut();
        float *vEmbdNextnOut = vEmbdNextn.vData + cOffset * cNEmbd;

        GGML_ASSERT((cOffset + cNRows) * cNEmbd <= (int64_t)vEmbdNextn.vSize);
        ggml_backend_tensor_get_async(vBackendH, vTHNextn, vEmbdNextnOut, 0, cNRows * cNEmbd * sizeof(float));
        vExtractAllIdxs = vExtractAllIdxs || !cMasked;
      }
    }

    if (vExtractAllIdxs) {
      GGML_ASSERT(vUbatch.vData && vUbatch.vData->lBatchIdxs.size() == vUbatch.vNTokens);
      GGML_ASSERT(lEmbdBatchIdxs.size() == (size_t)vNTokensPrev);
      const auto &vBatchIdxs = vUbatch.vData->lBatchIdxs;
      lEmbdBatchIdxs.insert(lEmbdBatchIdxs.end(), vBatchIdxs.begin(), vBatchIdxs.end());
    }

    // Copy backend sampling output if this ubatch produced any sampling tensors.
    if (cHasSamplers && (!vRes->dTSampled.empty() || !vRes->dTSampledProbs.empty() || !vRes->dTSampledLogits.empty())) {
      const auto cSeqToOutputRow = fBuildSeqToOutputRow(vUbatch, vNOutputsPrev);
      const auto cStride = cNVocab;

      // async copy the sampling data from the backend to the host
      fCopyTensorAsyncInts(vRes->dTSampled, vSampling.vSampled, cSeqToOutputRow, vSched.get());

      fCopyTensorAsyncFloats(vRes->dTSampledLogits, vSampling.vLogits, cStride, vSampling.lLogitsCount, cSeqToOutputRow,
                             vSched.get());
      fCopyTensorAsyncFloats(vRes->dTSampledProbs, vSampling.vProbs, cStride, vSampling.lProbsCount, cSeqToOutputRow,
                             vSched.get());
      fCopyTensorAsyncCandidates(vRes->dTCandidates, vSampling.vCandidates, cStride, vSampling.lCandidatesCount,
                                 cSeqToOutputRow, vSched.get());
    }

    vNOutputsPrev += vNOutputs;
    vNTokensPrev += vUbatch.vNTokens;
  } while (vMctx->mNext());

  // set to total number of outputs in the batch, for use in mgs_get_logits_ith
  vNOutputs = cNOutputsAll;

  // set output mappings
  if (vNOutputs > 0) {
    bool vSortedOutput = true;

    auto &vOutIds = vBalloc->mGetOutIds();

    GGML_ASSERT(vOutIds.size() == (size_t)vNOutputs);

    for (int64_t vI = 0; vI < vNOutputs; ++vI) {
      int64_t vOutId = vOutIds[vI];
      lOutputIds[vOutId] = vI;
      if (vOutId != vI) {
        vSortedOutput = false;
      }
    }

    // make the outputs have the same order they had in the user-provided batch
    // note: this is mostly relevant for recurrent models atm
    if (!vSortedOutput && vNOutputs > 1) {
      GGML_ASSERT((size_t)vNOutputs == vOutIds.size());

      // TODO: is there something more efficient which also minimizes swaps?
      // selection sort, to minimize swaps (from https://en.wikipedia.org/wiki/Selection_sort)
      for (uint32_t vI = 0; vI < vNOutputs - 1; ++vI) {
        uint32_t vJMin = vI;
        for (uint32_t vJ = vI + 1; vJ < vNOutputs; ++vJ) {
          if (vOutIds[vJ] < vOutIds[vJMin]) {
            vJMin = vJ;
          }
        }
        if (vJMin == vI) {
          continue;
        }
        std::swap(vOutIds[vI], vOutIds[vJMin]);

        // remember the swaps and apply them lazily upon logits/embeddings access
        lOutputSwaps.push_back({vI, vJMin});
      }

      std::fill(lOutputIds.begin(), lOutputIds.end(), -1);

      for (uint32_t vI = 0; vI < vNOutputs; ++vI) {
        lOutputIds[vOutIds[vI]] = vI;
      }
    }
  }

  // wait for the computation to finish (automatically done when obtaining the model output)
  // mSynchronize();

  return 0;
}

//
// output
//

uint32_t mgs_context::mOutputReserve(int32_t pNOutputs) {
  const auto &vHparams = vModel.vHparams;
  const auto &vVocab = vModel.vVocab;

  const int64_t cNOutputsMax = std::max<int64_t>(pNOutputs, mNSeqMax());

  const auto cNBatch = vCparams.vNBatch;
  const auto cNVocab = vVocab.mNTokens();
  const auto cNEmbd = vHparams.vNEmbd;
  const auto cNEmbdOut = vHparams.mNEmbdOut();

  bool vHasLogits = true;
  bool vHasEmbd = vCparams.vEmbeddings;
  bool vHasEmbdNextn = vCparams.vEmbeddingsNextn;

  // TODO: hacky enc-dec support
  if (vModel.vArch == cLlmArchT5) {
    vHasLogits = true;
    vHasEmbd = true;
  }

  size_t vBackendFloatCount = 0;
  size_t vBackendTokenCount = 0;
  size_t vEmbdLayerInpFloatCount = 0;

  vLogits.vSize = vHasLogits ? cNVocab * cNOutputsMax : 0;
  vEmbd.vSize = vHasEmbd ? cNEmbdOut * cNOutputsMax : 0;
  vEmbdNextn.vSize = vHasEmbdNextn ? cNEmbdOut * cNOutputsMax : 0;

  if (vHasEmbdNextn && !vCparams.vEmbeddingsNextnMasked) {
    // unmasked: nextn row exists for every token in the batch, not just
    // those flagged via batch.logits[i] -> size by token count instead.
    vEmbdNextn.vSize = (size_t)cNEmbdOut * cNBatch;
  }

  for (bool vEnabled : vCparams.lEmbeddingsLayerInp) {
    if (vEnabled) {
      vEmbdLayerInpFloatCount += (size_t)cNEmbd * cNBatch;
    }
  }

  // Allocate backend sampling output buffers if there are backend samplers configured.
  const bool cHasSampling = !vSampling.dSamplers.empty();
  if (cHasSampling) {
    vBackendFloatCount = 2 * cNVocab * cNOutputsMax;   // logits + probs
    vBackendTokenCount = (1 + cNVocab) * cNOutputsMax; // sampled + candidates
  }

  if (lOutputIds.empty()) {
    // init, never resized afterwards
    lOutputIds.resize(cNBatch);
  }

  const size_t cPrevSize = vBufOutput ? ggml_backend_buffer_get_size(vBufOutput.get()) : 0;
  const size_t cNewSize =
      (vLogits.vSize + vEmbd.vSize + vEmbdNextn.vSize + vEmbdLayerInpFloatCount + vBackendFloatCount) * sizeof(float) +
      (vBackendTokenCount) * sizeof(mgs_token);

  // alloc only when more than the current capacity is required
  // TODO: also consider shrinking the buffer
  if (!vBufOutput || cPrevSize < cNewSize) {
    if (vBufOutput) {
#ifndef NDEBUG
      // This doesn't happen often, but may be annoying in some cases (like the HellaSwag benchmark)
      MGS_LOG_DEBUG("%s: reallocating output buffer from size %.02f MiB to %.02f MiB\n", __func__,
                    cPrevSize / 1024.0 / 1024.0, cNewSize / 1024.0 / 1024.0);
#endif
      mSynchronize();

      // TODO: not needed?
      vBufOutput = nullptr;
      vLogits.vData = nullptr;
      vEmbd.vData = nullptr;
      vEmbdNextn.vData = nullptr;
      for (auto &vLayerInp : lEmbdLayerInp) {
        vLayerInp = {nullptr, 0};
      }
    }

    auto *vBuft = ggml_backend_cpu_buffer_type();
    // try to use the host buffer of the device where the output tensor is allocated for faster transfer to system
    // memory
    auto *vOutputDev = vModel.mDevOutput();
    auto *vOutputDevHostBuft = vOutputDev ? ggml_backend_dev_host_buffer_type(vOutputDev) : nullptr;
    if (vOutputDevHostBuft) {
      vBuft = vOutputDevHostBuft;
    }
    vBufOutput.reset(ggml_backend_buft_alloc_buffer(vBuft, cNewSize));
    if (vBufOutput == nullptr) {
      MGS_LOG_ERROR("%s: failed to allocate output buffer of size %.2f MiB\n", __func__, cNewSize / (1024.0 * 1024.0));
      return 0;
    }
    ggml_backend_buffer_clear(vBufOutput.get(), 0);
  }

  float *vOutputBase = (float *)ggml_backend_buffer_get_base(vBufOutput.get());

  size_t vOffset = 0;
  uint8_t *vBase = (uint8_t *)vOutputBase;

  vLogits = vHasLogits ? buffer_view<float>{vOutputBase, vLogits.vSize} : buffer_view<float>{nullptr, 0};
  vOffset += vLogits.vSize * sizeof(float);

  vEmbd = vHasEmbd ? buffer_view<float>{(float *)(vBase + vOffset), vEmbd.vSize} : buffer_view<float>{nullptr, 0};
  vOffset += vEmbd.vSize * sizeof(float);

  vEmbdNextn =
      vHasEmbdNextn ? buffer_view<float>{(float *)(vBase + vOffset), vEmbdNextn.vSize} : buffer_view<float>{nullptr, 0};
  vOffset += vEmbdNextn.vSize * sizeof(float);

  for (uint32_t vIl = 0; vIl < lEmbdLayerInp.size(); ++vIl) {
    if (vCparams.lEmbeddingsLayerInp[vIl]) {
      lEmbdLayerInp[vIl] = buffer_view<float>{(float *)(vBase + vOffset), (size_t)cNEmbd * cNBatch};
      vOffset += lEmbdLayerInp[vIl].vSize * sizeof(float);
    } else {
      lEmbdLayerInp[vIl] = buffer_view<float>{nullptr, 0};
    }
  }

  if (cHasSampling) {
    vSampling.vLogits = {(float *)(vBase + vOffset), (size_t)(cNVocab * cNOutputsMax)};
    vOffset += vSampling.vLogits.vSize * sizeof(float);

    vSampling.vProbs = {(float *)(vBase + vOffset), (size_t)(cNVocab * cNOutputsMax)};
    vOffset += vSampling.vProbs.vSize * sizeof(float);

    vSampling.vSampled = {(mgs_token *)(vBase + vOffset), (size_t)cNOutputsMax};
    vOffset += vSampling.vSampled.vSize * sizeof(mgs_token);

    vSampling.vCandidates = {(mgs_token *)(vBase + vOffset), (size_t)(cNVocab * cNOutputsMax)};
    vOffset += vSampling.vCandidates.vSize * sizeof(mgs_token);

    // The count vectors keep track of the actual number of logits/probs/candidates
    // copied from the backend for each output row.

    vSampling.lLogitsCount.resize(cNOutputsMax);
    vSampling.lProbsCount.resize(cNOutputsMax);
    vSampling.lCandidatesCount.resize(cNOutputsMax);

    std::fill(vSampling.lLogitsCount.begin(), vSampling.lLogitsCount.end(), 0);
    std::fill(vSampling.lProbsCount.begin(), vSampling.lProbsCount.end(), 0);
    std::fill(vSampling.lCandidatesCount.begin(), vSampling.lCandidatesCount.end(), 0);

    std::fill_n(vSampling.vSampled.vData, vSampling.vSampled.vSize, MGS_TOKEN_NULL);
  } else {
    vSampling.vLogits = {nullptr, 0};
    vSampling.vProbs = {nullptr, 0};
    vSampling.vSampled = {nullptr, 0};
    vSampling.vCandidates = {nullptr, 0};

    vSampling.lLogitsCount.clear();
    vSampling.lProbsCount.clear();
    vSampling.lCandidatesCount.clear();
  }

  // set all ids as invalid (negative)
  std::fill(lOutputIds.begin(), lOutputIds.end(), -1);

  this->vNOutputs = 0;

  GGML_ASSERT(cNOutputsMax <= vCparams.vNOutputsMax);

  return cNOutputsMax;
}

bool mgs_context::mExtractLayerInputs(const llm_graph_result *pRes, size_t pTokenOffset, size_t pNTokens) {
  bool vExtracted = false;
  for (uint32_t vIl = 0; vIl < vCparams.lEmbeddingsLayerInp.size(); ++vIl) {
    if (!vCparams.lEmbeddingsLayerInp[vIl]) {
      continue;
    }
    if (!lEmbdLayerInp[vIl].mHasData()) {
      GGML_ABORT("output layer input buffer not allocated");
    }
    ggml_tensor *vT = pRes->mGetLayerInp((int)vIl);
    if (!vT) {
      GGML_ABORT("layer input tensor not found");
    }

    const size_t cNbytes = ggml_nbytes(vT);
    const size_t cNfloats = cNbytes / sizeof(float);
    GGML_ASSERT(pNTokens > 0);
    GGML_ASSERT(cNfloats % pNTokens == 0);

    const size_t cRowFloats = cNfloats / pNTokens;
    GGML_ASSERT(cRowFloats == vModel.vHparams.vNEmbd);
    const size_t cDstOffset = pTokenOffset * cRowFloats;
    GGML_ASSERT(cDstOffset + cNfloats <= lEmbdLayerInp[vIl].vSize);

    ggml_backend_t vBackend = ggml_backend_sched_get_tensor_backend(vSched.get(), vT);
    GGML_ASSERT(vBackend != nullptr);
    // Tensor-split backends require a zero source offset.
    ggml_backend_tensor_get_async(vBackend, vT, lEmbdLayerInp[vIl].vData + cDstOffset, 0, cNbytes);
    vExtracted = true;
  }
  return vExtracted;
}

void mgs_context::mOutputReorder() {
  const uint64_t cNVocab = vModel.vVocab.mNTokens();
  const uint64_t cNEmbd = vModel.vHparams.vNEmbd;
  const uint64_t cNEmbdOut = vModel.vHparams.mNEmbdOut();

  for (size_t vS = 0; vS < lOutputSwaps.size(); ++vS) {
    const uint64_t cI0 = lOutputSwaps[vS].vI0;
    const uint64_t cI1 = lOutputSwaps[vS].vI1;

    if (vLogits.vSize > 0) {
      for (uint64_t vK = 0; vK < cNVocab; vK++) {
        std::swap(vLogits.vData[cI0 * cNVocab + vK], vLogits.vData[cI1 * cNVocab + vK]);
      }
    }

    if (vEmbd.vSize > 0) {
      for (uint64_t vK = 0; vK < cNEmbdOut; vK++) {
        std::swap(vEmbd.vData[cI0 * cNEmbdOut + vK], vEmbd.vData[cI1 * cNEmbdOut + vK]);
      }
    }

    if (vEmbdNextn.vSize > 0 && vCparams.vEmbeddingsNextnMasked) {
      for (uint64_t vK = 0; vK < cNEmbdOut; vK++) {
        std::swap(vEmbdNextn.vData[cI0 * cNEmbdOut + vK], vEmbdNextn.vData[cI1 * cNEmbdOut + vK]);
      }
    }

    if (!vSampling.dSamplers.empty()) {
      assert(vSampling.vLogits.vSize > 0);
      assert(vSampling.vProbs.vSize > 0);
      assert(vSampling.vCandidates.vSize > 0);
      assert(vSampling.vSampled.vSize > 0);
      assert(vSampling.lLogitsCount.size() > 0);
      assert(vSampling.lProbsCount.size() > 0);
      assert(vSampling.lCandidatesCount.size() > 0);

      for (uint64_t vK = 0; vK < cNVocab; ++vK) {
        std::swap(vSampling.vLogits.vData[cI0 * cNVocab + vK], vSampling.vLogits.vData[cI1 * cNVocab + vK]);
      }

      for (uint64_t vK = 0; vK < cNVocab; ++vK) {
        std::swap(vSampling.vProbs.vData[cI0 * cNVocab + vK], vSampling.vProbs.vData[cI1 * cNVocab + vK]);
      }

      for (uint64_t vK = 0; vK < cNVocab; ++vK) {
        std::swap(vSampling.vCandidates.vData[cI0 * cNVocab + vK], vSampling.vCandidates.vData[cI1 * cNVocab + vK]);
      }

      std::swap(vSampling.vSampled.vData[cI0], vSampling.vSampled.vData[cI1]);
      std::swap(vSampling.lLogitsCount[cI0], vSampling.lLogitsCount[cI1]);
      std::swap(vSampling.lProbsCount[cI0], vSampling.lProbsCount[cI1]);
      std::swap(vSampling.lCandidatesCount[cI0], vSampling.lCandidatesCount[cI1]);
    }
  }

  lOutputSwaps.clear();

  // [TAG_EXTRACT_TARGET_EMBEDDINGS]
  // Layer inputs and unmasked NextN embeddings contain all token rows, independent of logits selection.
  for (size_t vI = 0; vI < lEmbdBatchIdxs.size(); ++vI) {
    while (lEmbdBatchIdxs[vI] != (int32_t)vI) {
      const int32_t cJ = lEmbdBatchIdxs[vI];
      GGML_ASSERT(cJ >= 0 && (size_t)cJ < lEmbdBatchIdxs.size());
      if (vEmbdNextn.mHasData() && !vCparams.vEmbeddingsNextnMasked) {
        for (size_t vK = 0; vK < cNEmbdOut; ++vK) {
          std::swap(vEmbdNextn.vData[vI * cNEmbdOut + vK], vEmbdNextn.vData[cJ * cNEmbdOut + vK]);
        }
      }
      for (auto &vLayer : lEmbdLayerInp) {
        if (vLayer.mHasData()) {
          for (size_t vK = 0; vK < cNEmbd; ++vK) {
            std::swap(vLayer.vData[vI * cNEmbd + vK], vLayer.vData[cJ * cNEmbd + vK]);
          }
        }
      }
      std::swap(lEmbdBatchIdxs[vI], lEmbdBatchIdxs[cJ]);
    }
  }
  lEmbdBatchIdxs.clear();
}

//
// graph
//

uint32_t mgs_context::mGraphMaxNodes(uint32_t pNTokens) const {
  // Hybrid MoE execution builds a device and a CPU expert chain for every cached layer.
  auto fExpertCacheNodes = [&]() -> uint32_t { return vExpertCache ? 96u * (uint32_t)vModel.lLayers.size() : 0u; };

  if (vModel.vArch == cLlmArchKimiK3 || vModel.vArch == cLlmArchGlm5Next) {
    // the n_tokens*40 budget below is exhausted at ubatch 3840
    return std::max<uint32_t>(pNTokens * 160, 64u * vModel.mNTensors()) + fExpertCacheNodes();
  }
  if (vModel.vArch == cLlmArchQwen3next || vModel.vArch == cLlmArchKimiLinear || vModel.vArch == cLlmArchQwen35 ||
      vModel.vArch == cLlmArchQwen35moe || vModel.vArch == cLlmArchQwen4exp || vModel.vArch == cLlmArchDeepseek41 ||
      vModel.vArch == cLlmArchNanbeige ||
      vModel.vArch == cLlmArchMinimaxM3) {
    return std::max<uint32_t>(pNTokens * 40, 32u * vModel.mNTensors()) + fExpertCacheNodes();
  }
  uint32_t vRes = std::max<uint32_t>(1024u, 8u * vModel.mNTensors());
  for (const auto &vLora : vModel.sLoras) {
    vRes += vLora->mGetNNodes();
  }
  return vRes + fExpertCacheNodes();
}

llm_graph_result *mgs_context::mGetGfResReserve() const { return static_cast<llm_graph_result *>(vGfResReserve.get()); }

llm_graph_result *mgs_context::mGetGfResPrev() {
  auto &vResult = aGfResPrev[vNOutputs > 0];
  if (!vResult) {
    vResult.reset(new llm_graph_result(vGfResReserve->mGetMaxNodes()));
  }
  return vResult.get();
}

ggml_cgraph *mgs_context::mGraphReserve(uint32_t pNTokens, uint32_t pNSeqs, uint32_t pNOutputs,
                                        const mgs_memory_context_i *pMctx, bool pSplitOnly, size_t *pSizes) {
  MGS_LOG_DEBUG("%s: reserving a graph for ubatch with n_tokens = %4u, n_seqs = %2u, n_outputs = %4u\n", __func__,
                pNTokens, pNSeqs, pNOutputs);
  GGML_ASSERT(pNOutputs >= 1);

  if (pNTokens % pNSeqs != 0) {
    pNTokens = ((pNTokens + (pNSeqs - 1)) / pNSeqs) * pNSeqs; // round to next multiple of n_seqs
    MGS_LOG_DEBUG("%s: making n_tokens a multiple of n_seqs - n_tokens = %u, n_seqs = %u, n_outputs = %u\n", __func__,
                  pNTokens, pNSeqs, pNOutputs);
  }

  ggml_backend_sched_reset(vSched.get());

  // A scheduler reset invalidates both cached graph allocations.
  for (auto &vResult : aGfResPrev) {
    if (vResult) {
      vResult->mReset();
    }
  }
  vGfResPrevActive = nullptr;

  // store the n_outputs as it is, and restore it afterwards
  // TODO: not sure if needed, might simplify in the future by removing this
  const auto cSaveNOutputs = this->vNOutputs;

  this->vNOutputs = pNOutputs;

  mgs_batch_allocr vBalloc(vModel.vHparams.mNPosPerEmbd());
  mgs_ubatch vUbatch = vBalloc.mUbatchReserve(pNTokens / pNSeqs, pNSeqs);

  // set one output token per sequence in order to activate all backend samplers
  std::vector<mgs_seq_id> lSeqIds(pNSeqs);
  for (uint32_t vI = 0; vI < pNSeqs; ++vI) {
    lSeqIds[vI] = vI;
    vUbatch.vNSeqId[vI] = 1;
    vUbatch.vSeqId[vI] = &lSeqIds[vI];
    vUbatch.vOutput[vI] = true;
  }

  auto *vRes = vGfResReserve.get();

  const auto cGparams = mGraphParams(vRes, vUbatch, pMctx, fCtxTypeToGraphType(vCparams.vCtxType));

  vRes->mReset();

  auto *vGf = vModel.mBuildGraph(cGparams);

  this->vNOutputs = cSaveNOutputs;

  // initialize scheduler with the specified graph
  if (pSplitOnly) {
    if (pSizes) {
      ggml_backend_sched_reserve_size(vSched.get(), vGf, pSizes);
    } else {
      ggml_backend_sched_split_graph(vSched.get(), vGf);
    }
  } else if (!ggml_backend_sched_reserve(vSched.get(), vGf)) {
    GGML_ASSERT(!pSizes);
    MGS_LOG_ERROR("%s: failed to allocate compute buffers\n", __func__);
    return nullptr;
  }

  return vGf;
}

llm_graph_params mgs_context::mGraphParams(llm_graph_result *pRes, const mgs_ubatch &pUbatch,
                                           const mgs_memory_context_i *pMctx, llm_graph_type pGtype) const {
  return {
      /*.vArch        =*/vModel.vArch,
      /*.vHparams     =*/vModel.vHparams,
      /*.vCparams     =*/vCparams,
      /*.vUbatch      =*/pUbatch,
      /*.vGtype       =*/pGtype,
      /*.vSched       =*/vSched.get(),
      /*.vBackendCpu  =*/vBackendCpu,
      /*.vCvec        =*/vCvec.get(),
      /*.vLoras       =*/vLoras.get(),
      /*.vMctx        =*/pMctx,
      /*.vCross       =*/&vCross,
      /*.dSamplers    =*/vSampling.dSamplers,
      /*.vNOutputs    =*/vNOutputs,
      /*.vCb          =*/mGraphGetCb(),
      /*.vRes         =*/pRes,
      /*.vExpertCache =*/vExpertCache.get(),
  };
}

ggml_status mgs_context::mGraphCompute(ggml_cgraph *pGf, bool pBatched) {
  int vNThreads = pBatched ? vCparams.vNThreadsBatch : vCparams.vNThreads;
  ggml_threadpool_t vTp = pBatched ? vThreadpoolBatch : vThreadpool;

  if (vBackendCpu != nullptr) {
    auto *vReg = ggml_backend_dev_backend_reg(ggml_backend_get_device(vBackendCpu));
    auto *vSetThreadpoolFn = (decltype(ggml_backend_cpu_set_threadpool) *)ggml_backend_reg_get_proc_address(
        vReg, "ggml_backend_cpu_set_threadpool");
    if (vSetThreadpoolFn) {
      vSetThreadpoolFn(vBackendCpu, vTp);
    }
  }

  // set the number of threads for all the backends
  for (const auto &vSetNThreadsFn : lSetNThreadsFns) {
    vSetNThreadsFn.second(vSetNThreadsFn.first, vNThreads);
  }

  auto vStatus = ggml_backend_sched_graph_compute_async(vSched.get(), pGf);
  if (vStatus != GGML_STATUS_SUCCESS) {
    MGS_LOG_ERROR("%s: ggml_backend_sched_graph_compute_async failed with error %d\n", __func__, vStatus);
  }

  // fprintf(stderr, "splits: %d\n", ggml_backend_sched_get_n_splits(vSched.get()));

  return vStatus;
}

llm_graph_cb mgs_context::mGraphGetCb() const {
  return [&](const mgs_ubatch &pUbatch, ggml_tensor *pCur, const char *pName, int pIl) {
    if (pIl >= 0) {
      ggml_format_name(pCur, "%s-%d", pName, pIl);
    } else {
      ggml_set_name(pCur, pName);
    }

    // - norm may be automatically assigned to the backend of the previous layer, increasing data transfer between
    // backends
    // - force the last op of the layer on the specified backend to avoid running it on the backend of the next layer
    // due to scheduling
    // FIXME: fix in ggml_backend_sched
    const bool cFullOffload = vModel.mNGpuLayers() > vModel.vHparams.vNLayerAll;
    if (pUbatch.vNTokens < 32 || cFullOffload) {
      if (pIl != -1 && (strcmp(pName, "norm") == 0 || strcmp(pName, "l_last") == 0)) {
        const auto &vDevLayer = vModel.mDevLayer(pIl);
        for (const auto &vBackend : lBackends) {
          if (ggml_backend_get_device(vBackend.get()) == vDevLayer) {
            if (ggml_backend_supports_op(vBackend.get(), pCur)) {
              ggml_backend_sched_set_tensor_backend(vSched.get(), pCur, vBackend.get());
            }
          }
        }
      }
    }
  };
}

//
// state save/load
//

class mgs_io_write_dummy : public mgs_io_write_i {
public:
  mgs_io_write_dummy(bool pSkipTensors) : cSkipTensors(pSkipTensors) {}

  void mWrite(const void * /* src */, size_t pSize) override { vSizeWritten += pSize; }

  void mWriteTensor(ggml_tensor * /* tensor */, size_t /* offset */, size_t pSize) override {
    if (cSkipTensors) {
      return;
    }

    vSizeWritten += pSize;
  }

  size_t mNBytes() override { return vSizeWritten; }

private:
  const bool cSkipTensors;

  size_t vSizeWritten = 0;
};

class mgs_io_write_host : public mgs_io_write_i {
public:
  mgs_io_write_host(uint8_t *pP, size_t pLen) : vPtr(pP), vBufSize(pLen) {}

  ~mgs_io_write_host() {
    // TODO: add backend support to batch tensor_get? or some other way to speed this up
    for (const auto &vWinfo : lWinfos) {
      ggml_backend_tensor_get(vWinfo.vTensor, vWinfo.vPtr, vWinfo.vOffset, vWinfo.vSize);
    }
  }

  void mWrite(const void *pSrc, size_t pSize) override {
    if (pSize > vBufSize) {
      throw std::runtime_error("unexpectedly reached end of buffer");
    }
    memcpy(vPtr, pSrc, pSize);
    vPtr += pSize;
    vSizeWritten += pSize;
    vBufSize -= pSize;
  }

  void mWriteTensor(ggml_tensor *pTensor, size_t pOffset, size_t pSize) override {
    if (pSize > vBufSize) {
      throw std::runtime_error("unexpectedly reached end of buffer");
    }

    // save the write for later during destruction
    lWinfos.push_back({pTensor, vPtr, pSize, pOffset});

    vPtr += pSize;
    vSizeWritten += pSize;
    vBufSize -= pSize;
  }

  size_t mNBytes() override { return vSizeWritten; }

private:
  uint8_t *vPtr;
  size_t vBufSize = 0;
  size_t vSizeWritten = 0;

  struct write_info {
    ggml_tensor *vTensor;
    uint8_t *vPtr;
    size_t vSize;
    size_t vOffset;
  };
  std::vector<write_info> lWinfos;
};

class mgs_io_read_host : public mgs_io_read_i {
public:
  mgs_io_read_host(const uint8_t *pP, size_t pLen) : vPtr(pP), vBufSize(pLen) {}

  ~mgs_io_read_host() {
    // flush the reads
    for (const auto &vRinfo : lRinfos) {
      ggml_backend_tensor_set(vRinfo.vTensor, vRinfo.vPtr, vRinfo.vOffset, vRinfo.vSize);
    }
  }

  void mRead(void *pDst, size_t pSize) override {
    if (pSize > vBufSize) {
      throw std::runtime_error("unexpectedly reached end of buffer");
    }
    memcpy(pDst, vPtr, pSize);
    vPtr += pSize;
    vSizeRead += pSize;
    vBufSize -= pSize;
  }

  void mReadTensor(ggml_tensor *pTensor, size_t pOffset, size_t pSize) override {
    if (pSize > vBufSize) {
      throw std::runtime_error("unexpectedly reached end of buffer");
    }

    // save for later during destruction
    lRinfos.push_back({pTensor, vPtr, pSize, pOffset});

    vPtr += pSize;
    vSizeRead += pSize;
    vBufSize -= pSize;
  }

  size_t mNBytes() override { return vSizeRead; }

private:
  const uint8_t *vPtr;
  size_t vBufSize = 0;
  size_t vSizeRead = 0;

  struct read_info {
    ggml_tensor *vTensor;
    const uint8_t *vPtr;
    size_t vSize;
    size_t vOffset;
  };
  std::vector<read_info> lRinfos;
};

class mgs_io_write_file : public mgs_io_write_i {
public:
  mgs_io_write_file(mgs_file *pF) : vFile(pF) {}

  void mWrite(const void *pSrc, size_t pSize) override {
    vFile->mWriteRaw(pSrc, pSize);
    vSizeWritten += pSize;
  }

  void mWriteTensor(ggml_tensor *pTensor, size_t pOffset, size_t pSize) override {
    lTempBuffer.resize(pSize);
    ggml_backend_tensor_get(pTensor, lTempBuffer.data(), pOffset, pSize);
    mWrite(lTempBuffer.data(), lTempBuffer.size());
  }

  size_t mNBytes() override { return vSizeWritten; }

private:
  mgs_file *vFile;
  size_t vSizeWritten = 0;
  std::vector<uint8_t> lTempBuffer;
};

class mgs_io_read_file : public mgs_io_read_i {
public:
  mgs_io_read_file(mgs_file *pF) : vFile(pF) {}

  void mRead(void *pDst, size_t pSize) override {
    vFile->mReadRaw(pDst, pSize);
    vSizeRead += pSize;
  }

  void mReadTensor(ggml_tensor *pTensor, size_t pOffset, size_t pSize) override {
    lTempBuffer.resize(pSize);
    mRead(lTempBuffer.data(), pSize);
    ggml_backend_tensor_set(pTensor, lTempBuffer.data(), pOffset, pSize);
  }

  size_t mNBytes() override { return vSizeRead; }

private:
  mgs_file *vFile;
  size_t vSizeRead = 0;
  std::vector<uint8_t> lTempBuffer;
};

class mgs_io_write_device : public mgs_io_write_i {
public:
  mgs_io_write_device(uint8_t *pP, size_t pLen, mgs_memory_buffers &pMbufs)
      : vPtr(pP), vBufSize(pLen), vMbufs(pMbufs) {}

  ~mgs_io_write_device() {
    mgs_memory_buffers vMbufsNew;

    for (const auto &vWinfo : lWinfos) {
      auto *vBuft = ggml_backend_buffer_get_type(vWinfo.vTensor->buffer);

      vMbufsNew[vBuft].vNTensors++;
      vMbufsNew[vBuft].vTotalSize += vWinfo.vSize;
    }

    for (auto &[vBuft, vMbuf] : vMbufsNew) {
      ggml_init_params vParams = {
          /*.mem_size   =*/2 * vMbuf.vNTensors * ggml_tensor_overhead(),
          /*.mem_buffer =*/NULL,
          /*.no_alloc   =*/true,
      };

      vMbuf.vCtx.reset(ggml_init(vParams));

      vMbuf.lOrg.reserve(vMbuf.vNTensors);
      vMbuf.lCpy.reserve(vMbuf.vNTensors);
    }

    for (const auto &vWinfo : lWinfos) {
      auto *vBuft = ggml_backend_buffer_get_type(vWinfo.vTensor->buffer);

      const int64_t cN = vWinfo.vSize / ggml_element_size(vWinfo.vTensor);

      auto &vMbuf = vMbufsNew[vBuft];

      vMbuf.lOrg.push_back(ggml_view_1d(vMbuf.vCtx.get(), vWinfo.vTensor, cN, vWinfo.vOffset));
      vMbuf.lCpy.push_back(ggml_new_tensor_1d(vMbuf.vCtx.get(), vWinfo.vTensor->type, cN));
    }

    for (auto &[vBuft, vMbuf] : vMbufsNew) {
      auto &vMbufCur = vMbufs[vBuft];

      bool vNeedAlloc = false;

      vNeedAlloc = vNeedAlloc || (!vMbufCur.vBuf);
      vNeedAlloc = vNeedAlloc || (vMbufCur.lOrg.size() != vMbuf.lOrg.size());
      vNeedAlloc = vNeedAlloc || (vMbufCur.vTotalSize != vMbuf.vTotalSize);

      if (!vNeedAlloc) {
        for (size_t vI = 0; vI < vMbufCur.lOrg.size(); ++vI) {
          auto *vOrg0 = vMbufCur.lOrg[vI];
          auto *vOrg1 = vMbuf.lOrg[vI];

          if (!ggml_are_same_shape(vOrg0, vOrg1)) {
            vNeedAlloc = true;
            break;
          }

          if (vOrg0->view_src != vOrg1->view_src || vOrg0->view_offs != vOrg1->view_offs) {
            vNeedAlloc = true;
            break;
          }
        }
      }

      if (vNeedAlloc) {
        if (!vMbufCur.vBuf || vMbufCur.vTotalSize != vMbuf.vTotalSize) {
          vMbufCur = std::move(vMbuf);

          vMbufCur.vBuf.reset(ggml_backend_alloc_ctx_tensors_from_buft(vMbufCur.vCtx.get(), vBuft));

          MGS_LOG_INFO("%s: allocated '%s' buffer %.3f MiB\n", __func__, ggml_backend_buft_name(vBuft),
                       vMbuf.vTotalSize / 1024.0 / 1024.0);
        } else {
          // MGS_LOG_INFO("%s: reallocating tensors in '%s' buffer %.3f MiB\n", __func__,
          // ggml_backend_buft_name(vBuft), vMbuf.vTotalSize/1024.0/1024.0);

          // save the old buffer and allocate the new tensors in it
          auto vBuf = std::move(vMbufCur.vBuf);

          vMbufCur = std::move(vMbuf);

          ggml_tallocr vTalloc = ggml_tallocr_new(vBuf.get());

          for (size_t vI = 0; vI < vMbufCur.lOrg.size(); ++vI) {
            ggml_backend_view_init(vMbufCur.lOrg[vI]);
            ggml_tallocr_alloc(&vTalloc, vMbufCur.lCpy[vI]);
          }

          vMbufCur.vBuf = std::move(vBuf);
        }
      }

      for (size_t vI = 0; vI < vMbufCur.lOrg.size(); ++vI) {
        ggml_backend_tensor_copy(vMbufCur.lOrg[vI], vMbufCur.lCpy[vI]);
      }
    }
  }

  void mWrite(const void *pSrc, size_t pSize) override {
    if (pSize > vBufSize) {
      throw std::runtime_error("unexpectedly reached end of buffer");
    }
    memcpy(vPtr, pSrc, pSize);
    vPtr += pSize;
    vSizeWritten += pSize;
    vBufSize -= pSize;
  }

  void mWriteTensor(ggml_tensor *pTensor, size_t pOffset, size_t pSize) override {
    // save the write for later during destruction
    lWinfos.push_back({pTensor, vPtr, pSize, pOffset});
  }

  size_t mNBytes() override { return vSizeWritten; }

private:
  uint8_t *vPtr;
  size_t vBufSize = 0;
  size_t vSizeWritten = 0;

  struct write_info {
    ggml_tensor *vTensor;
    uint8_t *vPtr;
    size_t vSize;
    size_t vOffset;
  };
  std::vector<write_info> lWinfos;

  mgs_memory_buffers &vMbufs;
};

class mgs_io_read_device : public mgs_io_read_i {
public:
  mgs_io_read_device(const uint8_t *pP, size_t pLen, const mgs_memory_buffers &pMbufs)
      : vPtr(pP), vBufSize(pLen), vMbufs(pMbufs) {}

  ~mgs_io_read_device() {
    mgs_memory_buffers vMbufsNew;

    for (const auto &vRinfo : lRinfos) {
      auto *vBuft = ggml_backend_buffer_get_type(vRinfo.vTensor->buffer);

      vMbufsNew[vBuft].vNTensors++;
      vMbufsNew[vBuft].vTotalSize += vRinfo.vSize;
    }

    for (auto &[vBuft, vMbuf] : vMbufsNew) {
      ggml_init_params vParams = {
          /*.mem_size   =*/vMbuf.vNTensors * ggml_tensor_overhead(),
          /*.mem_buffer =*/NULL,
          /*.no_alloc   =*/true,
      };

      vMbuf.vCtx.reset(ggml_init(vParams));

      vMbuf.lOrg.reserve(vMbuf.vNTensors);
    }

    for (const auto &vRinfo : lRinfos) {
      auto *vBuft = ggml_backend_buffer_get_type(vRinfo.vTensor->buffer);

      const int64_t cN = vRinfo.vSize / ggml_element_size(vRinfo.vTensor);

      auto &vMbuf = vMbufsNew[vBuft];

      vMbuf.lOrg.push_back(ggml_view_1d(vMbuf.vCtx.get(), vRinfo.vTensor, cN, vRinfo.vOffset));

      ggml_backend_view_init(vMbuf.lOrg.back());
    }

    for (auto &[vBuft, vMbuf] : vMbufsNew) {
      const auto &vMbufCur = vMbufs.at(vBuft);

      if (!vMbufCur.vBuf || vMbufCur.vNTensors != vMbuf.vNTensors || vMbufCur.vTotalSize != vMbuf.vTotalSize) {
        GGML_ABORT("%s: memory buffer mismatch\n", __func__);
      }

      for (size_t vI = 0; vI < vMbufCur.lOrg.size(); ++vI) {
        ggml_backend_tensor_copy(vMbufCur.lCpy[vI], vMbuf.lOrg[vI]);
      }
    }

    GGML_ASSERT(vBufSize == 0);
  }

  void mRead(void *pDst, size_t pSize) override {
    if (pSize > vBufSize) {
      throw std::runtime_error("unexpectedly reached end of buffer");
    }
    memcpy(pDst, vPtr, pSize);
    vPtr += pSize;
    vSizeRead += pSize;
    vBufSize -= pSize;
  }

  void mReadTensor(ggml_tensor *pTensor, size_t pOffset, size_t pSize) override {
    // save for later during destruction
    lRinfos.push_back({pTensor, vPtr, pSize, pOffset});
  }

  size_t mNBytes() override { return vSizeRead; }

private:
  const uint8_t *vPtr;
  size_t vBufSize = 0;
  size_t vSizeRead = 0;

  struct read_info {
    ggml_tensor *vTensor;
    const uint8_t *vPtr;
    size_t vSize;
    size_t vOffset;
  };
  std::vector<read_info> lRinfos;

  const mgs_memory_buffers &vMbufs;
};

size_t mgs_context::mStateGetSize() {
  mgs_io_write_dummy vIo(false);
  try {
    return mStateWriteData(vIo);
  } catch (const std::exception &vErr) {
    MGS_LOG_ERROR("%s: error getting state size: %s\n", __func__, vErr.what());
    return 0;
  }
}

size_t mgs_context::mStateGetData(uint8_t *pDst, size_t pSize) {
  mgs_io_write_host vIo(pDst, pSize);
  try {
    return mStateWriteData(vIo);
  } catch (const std::exception &vErr) {
    MGS_LOG_ERROR("%s: error saving state: %s\n", __func__, vErr.what());
    return 0;
  }
}

size_t mgs_context::mStateSetData(const uint8_t *pSrc, size_t pSize) {
  mgs_io_read_host vIo(pSrc, pSize);
  try {
    return mStateReadData(vIo);
  } catch (const std::exception &vErr) {
    MGS_LOG_ERROR("%s: error loading state: %s\n", __func__, vErr.what());
    return 0;
  }
}

static constexpr uint32_t cIoMagic = 0xaf143cd8;

size_t mgs_context::mStateSeqGetSize(mgs_seq_id pSeqId, mgs_state_seq_flags pFlags) {
  mgs_io_write_dummy vIo(pFlags & MGS_STATE_SEQ_FLAGS_ON_DEVICE);
  try {
    vIo.mWrite(&cIoMagic, sizeof(cIoMagic));
    vIo.mWrite(&pSeqId, sizeof(pSeqId));

    return mStateSeqWriteData(vIo, pSeqId, pFlags);
  } catch (const std::exception &vErr) {
    MGS_LOG_ERROR("%s: error getting state size: %s\n", __func__, vErr.what());
    return 0;
  }
}

size_t mgs_context::mStateSeqGetData(mgs_seq_id pSeqId, uint8_t *pDst, size_t pSize, mgs_state_seq_flags pFlags) {
  std::unique_ptr<mgs_io_write_i> vIo;
  if (pFlags & MGS_STATE_SEQ_FLAGS_ON_DEVICE) {
    vIo = std::make_unique<mgs_io_write_device>(pDst, pSize, dMemStorage[pSeqId]);
  } else {
    vIo = std::make_unique<mgs_io_write_host>(pDst, pSize);
  }

  try {
    vIo->mWrite(&cIoMagic, sizeof(cIoMagic));
    vIo->mWrite(&pSeqId, sizeof(pSeqId));

    return mStateSeqWriteData(*vIo, pSeqId, pFlags);
  } catch (const std::exception &vErr) {
    MGS_LOG_ERROR("%s: error saving state: %s\n", __func__, vErr.what());
    return 0;
  }
}

size_t mgs_context::mStateSeqSetData(mgs_seq_id pSeqId, const uint8_t *pSrc, size_t pSize, mgs_state_seq_flags pFlags) {
  std::unique_ptr<mgs_io_read_i> vIo;
  if (pFlags & MGS_STATE_SEQ_FLAGS_ON_DEVICE) {
    // create a temporary io to read the magic and the src seq_id
    vIo = std::make_unique<mgs_io_read_host>(pSrc, pSize);

    uint32_t vMagicRead;
    vIo->mRead(&vMagicRead, sizeof(vMagicRead));
    if (cIoMagic != vMagicRead) {
      throw std::runtime_error("wrong sequence state magic");
    }

    mgs_seq_id vSeqIdRead;
    vIo->mRead(&vSeqIdRead, sizeof(vSeqIdRead));

    GGML_ASSERT(dMemStorage.find(vSeqIdRead) != dMemStorage.end());

    vIo = std::make_unique<mgs_io_read_device>(pSrc, pSize, dMemStorage[vSeqIdRead]);
  } else {
    vIo = std::make_unique<mgs_io_read_host>(pSrc, pSize);
  }

  try {
    uint32_t vMagicRead;
    vIo->mRead(&vMagicRead, sizeof(vMagicRead));
    if (cIoMagic != vMagicRead) {
      throw std::runtime_error("wrong sequence state magic");
    }

    mgs_seq_id vSeqIdRead;
    vIo->mRead(&vSeqIdRead, sizeof(vSeqIdRead));

    return mStateSeqReadData(*vIo, pSeqId, pFlags);
  } catch (const std::exception &vErr) {
    MGS_LOG_ERROR("%s: error loading state: %s\n", __func__, vErr.what());
    return 0;
  }
}

bool mgs_context::mStateLoadFile(const char *pFilepath, mgs_token *pTokensOut, size_t pNTokenCapacity,
                                 size_t *pNTokenCountOut) {
  mgs_file vFile(pFilepath, "rb");

  // sanity checks
  {
    const uint32_t cMagic = vFile.mReadU32();
    const uint32_t cVersion = vFile.mReadU32();

    if (cMagic != MGS_SESSION_MAGIC || cVersion != MGS_SESSION_VERSION) {
      MGS_LOG_ERROR("%s: unknown (magic, version) for session file: %08x, %08x\n", __func__, cMagic, cVersion);
      return false;
    }
  }

  // load the prompt
  {
    const uint32_t cNTokenCount = vFile.mReadU32();

    if (cNTokenCount > pNTokenCapacity) {
      MGS_LOG_ERROR("%s: token count in session file exceeded capacity! %u > %zu\n", __func__, cNTokenCount,
                    pNTokenCapacity);
      return false;
    }

    vFile.mReadRaw(pTokensOut, sizeof(mgs_token) * cNTokenCount);
    *pNTokenCountOut = cNTokenCount;
  }

  // restore the context state
  {
    const size_t cNStateSizeCur = vFile.size() - vFile.mTell();

    mgs_io_read_file vIo(&vFile);
    const size_t cNRead = mStateReadData(vIo);

    if (cNRead != cNStateSizeCur) {
      MGS_LOG_ERROR("%s: did not read all of the session file data! size %zu, got %zu\n", __func__, cNStateSizeCur,
                    cNRead);
      return false;
    }
  }

  return true;
}

bool mgs_context::mStateSaveFile(const char *pFilepath, const mgs_token *pTokens, size_t pNTokenCount) {
  mgs_file vFile(pFilepath, "wb");

  vFile.mWriteU32(MGS_SESSION_MAGIC);
  vFile.mWriteU32(MGS_SESSION_VERSION);

  // save the prompt
  vFile.mWriteU32((uint32_t)pNTokenCount);
  vFile.mWriteRaw(pTokens, sizeof(mgs_token) * pNTokenCount);

  // save the context state using stream saving
  mgs_io_write_file vIo(&vFile);
  mStateWriteData(vIo);

  return true;
}

size_t mgs_context::mStateSeqLoadFile(mgs_seq_id pSeqId, const char *pFilepath, mgs_token *pTokensOut,
                                      size_t pNTokenCapacity, size_t *pNTokenCountOut) {
  mgs_file vFile(pFilepath, "rb");

  // version checks
  {
    const uint32_t cMagic = vFile.mReadU32();
    const uint32_t cVersion = vFile.mReadU32();

    if (cMagic != MGS_STATE_SEQ_MAGIC || cVersion != MGS_STATE_SEQ_VERSION) {
      MGS_LOG_ERROR("%s: unknown (magic, version) for sequence state file: %08x, %08x\n", __func__, cMagic, cVersion);
      return 0;
    }
  }

  // load the prompt
  {
    const uint32_t cNTokenCount = vFile.mReadU32();

    if (cNTokenCount > pNTokenCapacity) {
      MGS_LOG_ERROR("%s: token count in sequence state file exceeded capacity! %u > %zu\n", __func__, cNTokenCount,
                    pNTokenCapacity);
      return 0;
    }

    vFile.mReadRaw(pTokensOut, sizeof(mgs_token) * cNTokenCount);
    *pNTokenCountOut = cNTokenCount;
  }

  // restore the context state
  {
    const size_t cStateSize = vFile.size() - vFile.mTell();
    mgs_io_read_file vIo(&vFile);
    const size_t cNread = mStateSeqReadData(vIo, pSeqId, 0);
    if (!cNread) {
      MGS_LOG_ERROR("%s: failed to restore sequence state\n", __func__);
      return 0;
    }
    GGML_ASSERT(cNread <= cStateSize);
    GGML_ASSERT(cNread + sizeof(uint32_t) * 3 + sizeof(mgs_token) * *pNTokenCountOut == vFile.mTell());
  }

  return vFile.mTell();
}

size_t mgs_context::mStateSeqSaveFile(mgs_seq_id pSeqId, const char *pFilepath, const mgs_token *pTokens,
                                      size_t pNTokenCount) {
  mgs_file vFile(pFilepath, "wb");

  vFile.mWriteU32(MGS_STATE_SEQ_MAGIC);
  vFile.mWriteU32(MGS_STATE_SEQ_VERSION);

  // save the prompt
  vFile.mWriteU32((uint32_t)pNTokenCount);
  vFile.mWriteRaw(pTokens, sizeof(mgs_token) * pNTokenCount);

  // save the context state using stream saving
  mgs_io_write_file vIo(&vFile);
  mStateSeqWriteData(vIo, pSeqId, 0);

  const size_t cRes = vFile.mTell();
  GGML_ASSERT(cRes == sizeof(uint32_t) * 3 + sizeof(mgs_token) * pNTokenCount + vIo.mNBytes());

  return cRes;
}

size_t mgs_context::mStateWriteData(mgs_io_write_i &pIo) {
  MGS_LOG_DEBUG("%s: writing state\n", __func__);

  // write model info
  {
    MGS_LOG_DEBUG("%s: - writing model info\n", __func__);

    const std::string cArchStr = fLlmArchName(vModel.vArch);
    pIo.mWriteString(cArchStr);
    // TODO: add more model-specific info which should prevent loading the session file if not identical
  }

  if (vMemory != nullptr) {
    MGS_LOG_DEBUG("%s: - writing memory module\n", __func__);
    vMemory->mStateWrite(pIo);
  }

  return pIo.mNBytes();
}

size_t mgs_context::mStateReadData(mgs_io_read_i &pIo) {
  MGS_LOG_DEBUG("%s: reading state\n", __func__);

  // read model info
  {
    MGS_LOG_DEBUG("%s: - reading model info\n", __func__);

    const std::string cCurArchStr = fLlmArchName(vModel.vArch);

    std::string vArchStr;
    pIo.mReadString(vArchStr);
    if (cCurArchStr != vArchStr) {
      throw std::runtime_error(
          fFormat("wrong model arch: '%s' instead of '%s'", vArchStr.c_str(), cCurArchStr.c_str()));
    }
    // TODO: add more info which needs to be identical but which is not verified otherwise
  }

  if (vMemory) {
    MGS_LOG_DEBUG("%s: - reading memory module\n", __func__);

    vMemory->mStateRead(pIo);
  }

  return pIo.mNBytes();
}

size_t mgs_context::mStateSeqWriteData(mgs_io_write_i &pIo, mgs_seq_id pSeqId, mgs_state_seq_flags pFlags) {
  GGML_UNUSED(pSeqId);

  if (vMemory) {
    vMemory->mStateWrite(pIo, pSeqId, pFlags);
  }

  return pIo.mNBytes();
}

size_t mgs_context::mStateSeqReadData(mgs_io_read_i &pIo, mgs_seq_id pSeqId, mgs_state_seq_flags pFlags) {
  GGML_UNUSED(pSeqId);

  if (vMemory) {
    vMemory->mStateRead(pIo, pSeqId, pFlags);
  }

  return pIo.mNBytes();
}

//
// perf
//

mgs_perf_context_data mgs_context::mPerfGetData() const {
  mgs_perf_context_data vData = {};

  vData.t_start_ms = 1e-3 * vTStartUs;
  vData.t_load_ms = 1e-3 * vTLoadUs;
  vData.t_p_eval_ms = 1e-3 * vTPEvalUs;
  vData.t_eval_ms = 1e-3 * vTEvalUs;
  vData.n_p_eval = std::max(1, vNPEval);
  vData.n_eval = std::max(1, vNEval);
  vData.n_reused = std::max(0, vNReused);

  return vData;
}

void mgs_context::mPerfReset() {
  vTStartUs = ggml_time_us();
  vTEvalUs = vNEval = 0;
  vTPEvalUs = vNPEval = 0;
  vNReused = 0;
}

mgs_memory_breakdown mgs_context::mMemoryBreakdown() const {
  std::map<ggml_backend_buffer_type_t, mgs_memory_breakdown_data> dRet;
  for (const auto &[vBuft, vSize] : vModel.mMemoryBreakdown()) {
    dRet[vBuft].model += vSize;
  }
  if (vMemory) {
    for (const auto &[vBuft, vSize] : vMemory->mMemoryBreakdown()) {
      dRet[vBuft].context += vSize;
    }
  }
  if (vModel.vHparams.vNoAlloc) {
    for (size_t vI = 0; vI < lBackends.size(); ++vI) {
      ggml_backend_t vBackend = lBackends[vI].get();
      ggml_backend_buffer_type_t vBuft = ggml_backend_sched_get_buffer_type(vSched.get(), vBackend);
      dRet[vBuft].compute += lBackendBufExpSize[vI];
    }
  } else {
    for (const auto &vBackendPtr : lBackends) {
      ggml_backend_t vBackend = vBackendPtr.get();
      ggml_backend_buffer_type_t vBuft = ggml_backend_sched_get_buffer_type(vSched.get(), vBackend);
      dRet[vBuft].compute += ggml_backend_sched_get_buffer_size(vSched.get(), vBackend);
    }
  }
  return dRet;
}

//
// training
//

static void fMgsSetParam(struct ggml_tensor *pTensor, mgs_opt_param_filter pParamFilter, void *pUserdata) {
  if (!pTensor || pTensor->type != GGML_TYPE_F32) {
    return;
  }
  if (!pParamFilter(pTensor, pUserdata)) {
    return;
  }
  if (strcmp(pTensor->name, "token_embd.weight") == 0) {
    return; // FIXME
  }
  if (strcmp(pTensor->name, "rope_freqs.weight") == 0) {
    return; // FIXME
  }
  ggml_set_param(pTensor);
}

void mgs_context::mOptInit(struct mgs_model *pModel, struct mgs_opt_params pLoptParams) {
  GGML_ASSERT(!vOptCtx);
  pModel->vHparams.vNCtxTrain = pLoptParams.n_ctx_train > 0 ? pLoptParams.n_ctx_train : mNCtx();
  const uint32_t cNBatch = std::min(this->mNBatch(), pModel->vHparams.vNCtxTrain);
  const uint32_t cNUbatch = std::min(this->mNUbatch(), cNBatch);
  GGML_ASSERT(pModel->vHparams.vNCtxTrain % cNBatch == 0);
  GGML_ASSERT(cNBatch % cNUbatch == 0);

  ggml_opt_params vOptParams = ggml_opt_default_params(vSched.get(), GGML_OPT_LOSS_TYPE_CROSS_ENTROPY);
  vOptParams.opt_period = cNBatch / cNUbatch;
  vOptParams.get_opt_pars = pLoptParams.get_opt_pars;
  vOptParams.get_opt_pars_ud = pLoptParams.get_opt_pars_ud;
  vOptParams.optimizer = pLoptParams.optimizer_type;
  vOptCtx = ggml_opt_init(vOptParams);

  mgs_opt_param_filter vParamFilter = pLoptParams.param_filter;
  void *vParamFilterUd = pLoptParams.param_filter_ud;

  // fMgsSetParam(pModel->vTokEmbd, vParamFilter, vParamFilterUd); // FIXME
  fMgsSetParam(pModel->vTypeEmbd, vParamFilter, vParamFilterUd);
  fMgsSetParam(pModel->vPosEmbd, vParamFilter, vParamFilterUd);
  fMgsSetParam(pModel->vTokNorm, vParamFilter, vParamFilterUd);
  fMgsSetParam(pModel->vTokNormB, vParamFilter, vParamFilterUd);
  fMgsSetParam(pModel->vOutputNorm, vParamFilter, vParamFilterUd);
  fMgsSetParam(pModel->vOutputNormB, vParamFilter, vParamFilterUd);
  fMgsSetParam(pModel->vOutput, vParamFilter, vParamFilterUd);
  fMgsSetParam(pModel->vOutputB, vParamFilter, vParamFilterUd);
  fMgsSetParam(pModel->vOutputNormEnc, vParamFilter, vParamFilterUd);
  fMgsSetParam(pModel->vCls, vParamFilter, vParamFilterUd);
  fMgsSetParam(pModel->vClsB, vParamFilter, vParamFilterUd);
  fMgsSetParam(pModel->vClsOut, vParamFilter, vParamFilterUd);
  fMgsSetParam(pModel->vClsOutB, vParamFilter, vParamFilterUd);
  fMgsSetParam(pModel->vClsNorm, vParamFilter, vParamFilterUd);

  for (struct mgs_layer &vLayer : pModel->lLayers) {
    for (size_t vI = 0; vI < sizeof(vLayer) / sizeof(struct ggml_tensor *); ++vI) {
      fMgsSetParam(reinterpret_cast<struct ggml_tensor **>(&vLayer)[vI], vParamFilter, vParamFilterUd);
    }
  }
}

void mgs_context::mOptEpochIter(ggml_opt_dataset_t pDataset, ggml_opt_result_t pResult,
                                const std::vector<mgs_token> &pTokens, const std::vector<mgs_token> &pLabelsSparse,
                                mgs_batch &pBatch, ggml_opt_epoch_callback pCallback, bool pTrain, int64_t pIdataInLoop,
                                int64_t pNdataInLoop, int64_t pTLoopStart) {
  GGML_ASSERT(vOptCtx);
  const uint32_t cNCtx = mgs_model_n_ctx_train(&vModel);
  const uint32_t cNBatch = std::min(this->mNBatch(), cNCtx);
  const uint32_t cNUbatch = std::min(this->mNUbatch(), cNBatch);

  vMemory->mClear(true);

  for (uint32_t vPosCtx = 0; vPosCtx < cNCtx; vPosCtx += cNBatch) {
    pBatch.n_tokens = cNBatch;
    for (uint32_t vPosBatch = 0; vPosBatch < cNBatch; ++vPosBatch) {
      pBatch.token[vPosBatch] = pTokens[vPosCtx + vPosBatch];
      pBatch.pos[vPosBatch] = vPosCtx + vPosBatch;
      pBatch.n_seq_id[vPosBatch] = 1;
      pBatch.seq_id[vPosBatch][0] = 0;
      pBatch.logits[vPosBatch] = true;
    }

    if (!vBalloc->mInit(pBatch, vModel.vVocab, nullptr, vModel.vHparams.mNEmbdInp(),
                        vCparams.vKvUnified ? MGS_MAX_SEQ : vCparams.vNSeqMax, true)) {
      MGS_LOG_ERROR("%s: failed to initialize batch\n", __func__);
      return;
    }

    const uint32_t cNTokensAll = vBalloc->mGetNTokens();

    vNQueuedTokens += cNTokensAll;

    dEmbdSeq.clear();

    uint32_t vNOutputsAll = cNTokensAll;

    auto vMctx = vMemory->mInitBatch(*vBalloc, vCparams.vNUbatch, true);
    if (!vMctx || vMctx->mGetStatus() != cMgsMemoryStatusSuccess) {
      MGS_LOG_ERROR("%s: could not initialize batch\n", __func__);
      break;
    }

    // reserve output buffer
    if (mOutputReserve(vNOutputsAll) < vNOutputsAll) {
      MGS_LOG_ERROR("%s: could not reserve space for batch with %d outputs\n", __func__, vNOutputsAll);
      GGML_ABORT("TODO: handle this error");
    };

    uint32_t vPosBatch = 0;
    do {
      const auto &vUbatch = vMctx->mGetUbatch();

      vNOutputs = vUbatch.vNTokens;

      if (!vMctx->mApply()) {
        MGS_LOG_ERROR("%s: failed to update the memory context\n", __func__);
        break;
      }

      auto *vRes = mGetGfResPrev();

      const auto cGparams = mGraphParams(vRes, vUbatch, vMctx.get(), fCtxTypeToGraphType(vCparams.vCtxType));

      // The optimizer allocates outside the scheduler; the next decode must rebuild.
      vGfResPrevActive = nullptr;
      vRes->mReset();

      auto *vGf = vModel.mBuildGraph(cGparams);

      struct ggml_context *vCtxComputeOpt;
      {
        const size_t cSizeGf = ggml_graph_size(vGf);
        const size_t cSizeMeta =
            4 * cSizeGf * ggml_tensor_overhead() + 2 * ggml_graph_overhead_custom(cSizeGf, /*grads = */ true);
        struct ggml_init_params vParams = {
            /*.mem_size   =*/cSizeMeta,
            /*.mem_buffer =*/nullptr,
            /*.no_alloc   =*/true,
        };
        vCtxComputeOpt = ggml_init(vParams);
      }
      ggml_opt_prepare_alloc(vOptCtx, vCtxComputeOpt, vGf, vRes->mGetInpTokens(), vRes->mGetLogits());
      ggml_opt_alloc(vOptCtx, pTrain);

      vRes->mSetInputs(&vUbatch);
      {
        struct ggml_tensor *vLabels = ggml_opt_labels(vOptCtx);
        GGML_ASSERT(vLabels->ne[1] == cNUbatch);
        ggml_set_zero(vLabels);
        const float cOnef = 1.0f;
        for (uint32_t vPosUbatch = 0; vPosUbatch < cNUbatch; ++vPosUbatch) {
          const uint32_t cIlabel = vPosCtx + vPosBatch + vPosUbatch;
          GGML_ASSERT(pLabelsSparse[cIlabel] < vLabels->ne[0]);
          ggml_backend_tensor_set(
              vLabels, &cOnef, (vPosUbatch * vLabels->ne[0] + pLabelsSparse[cIlabel]) * sizeof(float), sizeof(float));
        }
      }
      ggml_opt_eval(vOptCtx, pResult);
      if (pCallback) {
        pCallback(pTrain, vOptCtx, pDataset, pResult, pIdataInLoop + (vPosCtx + vPosBatch) / cNUbatch + 1, pNdataInLoop,
                  pTLoopStart);
      }
      ggml_free(vCtxComputeOpt);

      vPosBatch += vUbatch.vNTokens;
    } while (vMctx->mNext());
  }
}

void mgs_context::mOptEpoch(ggml_opt_dataset_t pDataset, ggml_opt_result_t pResultTrain, ggml_opt_result_t pResultEval,
                            int64_t pIdataSplit, ggml_opt_epoch_callback pCallbackTrain,
                            ggml_opt_epoch_callback pCallbackEval) {
  const uint32_t cNCtx = this->mNCtx();
  const uint32_t cNBatch = std::min(vCparams.vNBatch, cNCtx);
  const uint32_t cNUbatch = std::min(vCparams.vNUbatch, cNBatch);
  const int64_t cNdata = ggml_opt_dataset_ndata(pDataset);

  GGML_ASSERT(pIdataSplit >= 0);
  GGML_ASSERT(pIdataSplit <= cNdata);

  const uint32_t cUbatchPerCtx = cNCtx / cNUbatch;

  struct mgs_batch vBatch = mgs_batch_init(cNBatch, 0, 1);
  std::vector<mgs_token> lTokens(cNCtx);
  std::vector<mgs_token> lLabelsSparse(cNCtx);

  int64_t vIdata = 0;

  int64_t vTLoopStart = ggml_time_us();
  int64_t vNdataInLoop = pIdataSplit * cUbatchPerCtx;
  for (; vIdata < pIdataSplit; ++vIdata) {
    constexpr bool cTrain = true;
    const int64_t cIdataInLoop = vIdata * cUbatchPerCtx;

    ggml_opt_dataset_get_batch_host(pDataset, lTokens.data(), cNCtx * sizeof(mgs_token), lLabelsSparse.data(), vIdata);
    mOptEpochIter(pDataset, pResultTrain, lTokens, lLabelsSparse, vBatch, pCallbackTrain, cTrain, cIdataInLoop,
                  vNdataInLoop, vTLoopStart);
  }

  vTLoopStart = ggml_time_us();
  vNdataInLoop = (cNdata - pIdataSplit) * cUbatchPerCtx;
  for (; vIdata < cNdata; ++vIdata) {
    constexpr bool cTrain = false;
    const int64_t cIdataInLoop = (vIdata - pIdataSplit) * cUbatchPerCtx;

    ggml_opt_dataset_get_batch_host(pDataset, lTokens.data(), cNCtx * sizeof(mgs_token), lLabelsSparse.data(), vIdata);
    mOptEpochIter(pDataset, pResultEval, lTokens, lLabelsSparse, vBatch, pCallbackEval, cTrain, cIdataInLoop,
                  vNdataInLoop, vTLoopStart);
  }

  mgs_batch_free(vBatch);
}

//
// interface implementation
//

mgs_context_params mgs_context_default_params() {
  mgs_context_params vResult = {
      /*.n_ctx                       =*/512,
      /*.n_batch                     =*/2048,
      /*.n_ubatch                    =*/512,
      /*.n_seq_max                   =*/1,
      /*.n_rs_seq                    =*/0,
      /*.n_outputs_max               =*/0,
      /*.n_threads                   =*/GGML_DEFAULT_N_THREADS, // TODO: better default
      /*.n_threads_batch             =*/GGML_DEFAULT_N_THREADS,
      /*.ctx_type                    =*/MGS_CONTEXT_TYPE_DEFAULT,
      /*.rope_scaling_type           =*/MGS_ROPE_SCALING_TYPE_UNSPECIFIED,
      /*.pooling_type                =*/MGS_POOLING_TYPE_UNSPECIFIED,
      /*.attention_type              =*/MGS_ATTENTION_TYPE_UNSPECIFIED,
      /*.flash_attn_type             =*/MGS_FLASH_ATTN_TYPE_AUTO,
      /*.rope_freq_base              =*/0.0f,
      /*.rope_freq_scale             =*/0.0f,
      /*.yarn_ext_factor             =*/-1.0f,
      /*.yarn_attn_factor            =*/-1.0f,
      /*.yarn_beta_fast              =*/-1.0f,
      /*.yarn_beta_slow              =*/-1.0f,
      /*.yarn_orig_ctx               =*/0,
      /*.cb_eval                     =*/nullptr,
      /*.cb_eval_user_data           =*/nullptr,
      /*.type_k                      =*/GGML_TYPE_F16,
      /*.type_v                      =*/GGML_TYPE_F16,
      /*.abort_callback              =*/nullptr,
      /*.abort_callback_data         =*/nullptr,
      /*.embeddings                  =*/false,
      /*.offload_kqv                 =*/true,
      /*.no_perf                     =*/true,
      /*.op_offload                  =*/true,
      /*.swa_full                    =*/true,
      /*.kv_unified                  =*/false,
      /*.samplers                    =*/nullptr,
      /*.n_samplers                  =*/0,
      /*.ctx_other                   =*/nullptr,
  };

  return vResult;
}

mgs_context *mgs_init_from_model(mgs_model *pModel, mgs_context_params pParams) {
  if (!pModel) {
    MGS_LOG_ERROR("%s: model cannot be NULL\n", __func__);
    return nullptr;
  }

  if (pParams.n_batch == 0 && pParams.n_ubatch == 0) {
    MGS_LOG_ERROR("%s: n_batch and n_ubatch cannot both be zero\n", __func__);
    return nullptr;
  }

  if (pParams.n_ctx == 0 && pModel->vHparams.vNCtxTrain == 0) {
    MGS_LOG_ERROR("%s: n_ctx and model->hparams.n_ctx_train cannot both be zero\n", __func__);
    return nullptr;
  }

  if (pParams.flash_attn_type != MGS_FLASH_ATTN_TYPE_DISABLED && pModel->vArch == cLlmArchGrok) {
    MGS_LOG_WARN("%s: flash_attn is not compatible with Grok - forcing off\n", __func__);
    pParams.flash_attn_type = MGS_FLASH_ATTN_TYPE_DISABLED;
  }

  if (pModel->mSplitMode() == MGS_SPLIT_MODE_TENSOR) {
    if (pParams.flash_attn_type == MGS_FLASH_ATTN_TYPE_AUTO) {
      MGS_LOG_INFO("%s: enabling flash_attn since it is required for SPLIT_MODE_TENSOR\n", __func__);
      pParams.flash_attn_type = MGS_FLASH_ATTN_TYPE_ENABLED;
    }
    if (pParams.flash_attn_type != MGS_FLASH_ATTN_TYPE_ENABLED) {
      MGS_LOG_ERROR("%s: SPLIT_MODE_TENSOR requires flash_attn to be enabled\n", __func__);
      return nullptr;
    }
  }

  if ((pModel->vHparams.mIsMla() || pModel->vArch == cLlmArchDeepseek41) && pParams.type_k != pParams.type_v) {
    MGS_LOG_ERROR("%s: model does not support different K (%s) and V (%s) cache types\n", __func__,
                  ggml_type_name(pParams.type_k), ggml_type_name(pParams.type_v));
    return nullptr;
  }

  if (ggml_is_quantized(pParams.type_v) && pParams.flash_attn_type != MGS_FLASH_ATTN_TYPE_ENABLED) {
    if (pParams.flash_attn_type == MGS_FLASH_ATTN_TYPE_AUTO) {
      MGS_LOG_INFO("%s: enabling flash_attn since it is required for quantized V cache\n", __func__);
      pParams.flash_attn_type = MGS_FLASH_ATTN_TYPE_ENABLED;
    }
    if (pParams.flash_attn_type == MGS_FLASH_ATTN_TYPE_DISABLED) {
      MGS_LOG_ERROR("%s: quantized V cache requires flash_attn to be enabled\n", __func__);
      return nullptr;
    }
  }

  if (pParams.flash_attn_type != MGS_FLASH_ATTN_TYPE_DISABLED && ggml_is_quantized(pParams.type_k)) {
    const uint32_t cBlckSize = ggml_blck_size(pParams.type_k);
    for (uint32_t vIl = 0; vIl < pModel->vHparams.mNLayer(); ++vIl) {
      if (pModel->vHparams.mNEmbdHeadK(vIl) % cBlckSize != 0) {
        MGS_LOG_ERROR("%s: K cache type %s with block size %u does not divide n_embd_head_k=%u\n", __func__,
                      ggml_type_name(pParams.type_k), cBlckSize, pModel->vHparams.mNEmbdHeadK(vIl));
        return nullptr;
      }
    }
  }

  if (pParams.flash_attn_type != MGS_FLASH_ATTN_TYPE_DISABLED && ggml_is_quantized(pParams.type_v)) {
    const uint32_t cBlckSize = ggml_blck_size(pParams.type_v);
    for (uint32_t vIl = 0; vIl < pModel->vHparams.mNLayer(); ++vIl) {
      if (pModel->vHparams.mNEmbdHeadV(vIl) % cBlckSize != 0) {
        MGS_LOG_ERROR("%s: V cache type %s with block size %u does not divide n_embd_head_v=%u\n", __func__,
                      ggml_type_name(pParams.type_v), cBlckSize, pModel->vHparams.mNEmbdHeadV(vIl));
        return nullptr;
      }
    }
  }

  if (pParams.pooling_type != MGS_POOLING_TYPE_UNSPECIFIED && pParams.pooling_type != pModel->vHparams.vPoolingType) {
    // user-specified pooling-type is different from the model default
    MGS_LOG_WARN("%s: model default pooling_type is [%d], but [%d] was specified\n", __func__,
                 pModel->vHparams.vPoolingType, pParams.pooling_type);
  }

  if (pParams.ctx_type == MGS_CONTEXT_TYPE_MTP && pModel->vHparams.vNLayerNextn == 0) {
    MGS_LOG_WARN("%s: context type MTP requested but model doesn't contain MTP layers\n", __func__);
    return nullptr;
  }

  try {
    auto *vCtx = new mgs_context(*pModel, pParams);
    return vCtx;
  } catch (const std::exception &vErr) {
    MGS_LOG_ERROR("%s: failed to initialize the context: %s\n", __func__, vErr.what());
  }

  return nullptr;
}

void mgs_free(mgs_context *pCtx) { delete pCtx; }

uint32_t mgs_n_ctx(const mgs_context *pCtx) { return pCtx->mNCtx(); }

uint32_t mgs_n_ctx_seq(const mgs_context *pCtx) { return pCtx->mNCtxSeq(); }

uint32_t mgs_n_batch(const mgs_context *pCtx) { return pCtx->mNBatch(); }

uint32_t mgs_n_ubatch(const mgs_context *pCtx) { return pCtx->mNUbatch(); }

uint32_t mgs_n_seq_max(const mgs_context *pCtx) { return pCtx->mNSeqMax(); }

uint32_t mgs_n_rs_seq(const mgs_context *pCtx) { return pCtx->mGetCparams().vNRsSeq; }

const mgs_model *mgs_get_model(const mgs_context *pCtx) { return &pCtx->mGetModel(); }

enum mgs_pooling_type mgs_pooling_type(const mgs_context *pCtx) { return pCtx->mPoolingType(); }

void mgs_attach_threadpool(mgs_context *pCtx, ggml_threadpool_t pThreadpool, ggml_threadpool_t pThreadpoolBatch) {
  pCtx->mAttachThreadpool(pThreadpool, pThreadpoolBatch);
}

void mgs_detach_threadpool(mgs_context *pCtx) { pCtx->mDetachThreadpool(); }

void mgs_set_n_threads(mgs_context *pCtx, int32_t pNThreads, int32_t pNThreadsBatch) {
  pCtx->mSetNThreads(pNThreads, pNThreadsBatch);
}

int32_t mgs_n_threads(mgs_context *pCtx) { return pCtx->mNThreads(); }

int32_t mgs_n_threads_batch(mgs_context *pCtx) { return pCtx->mNThreadsBatch(); }

void mgs_set_abort_callback(mgs_context *pCtx, bool (*pAbortCallback)(void *pData), void *pAbortCallbackData) {
  pCtx->mSetAbortCallback(pAbortCallback, pAbortCallbackData);
}

void mgs_set_embeddings(mgs_context *pCtx, bool pEmbeddings) { pCtx->mSetEmbeddings(pEmbeddings); }

void mgs_set_causal_attn(mgs_context *pCtx, bool pCausalAttn) { pCtx->mSetCausalAttn(pCausalAttn); }

void mgs_synchronize(mgs_context *pCtx) { pCtx->mSynchronize(); }

float *mgs_get_logits(mgs_context *pCtx) {
  pCtx->mSynchronize();

  return pCtx->mGetLogits();
}

float *mgs_get_logits_ith(mgs_context *pCtx, int32_t pI) {
  pCtx->mSynchronize();

  float *vRes = nullptr;

  vRes = pCtx->mGetSampledLogitsIth(pI);

  if (!vRes) {
    vRes = pCtx->mGetLogitsIth(pI);
  }

  return vRes;
}

float *mgs_get_embeddings(mgs_context *pCtx) {
  pCtx->mSynchronize();

  return pCtx->mGetEmbeddings();
}

float *mgs_get_embeddings_ith(mgs_context *pCtx, int32_t pI) {
  pCtx->mSynchronize();

  return pCtx->mGetEmbeddingsIth(pI);
}

float *mgs_get_embeddings_seq(mgs_context *pCtx, mgs_seq_id pSeqId) {
  pCtx->mSynchronize();

  return pCtx->mGetEmbeddingsSeq(pSeqId);
}

void mgs_set_embeddings_nextn(mgs_context *pCtx, bool pValue, bool pMasked) {
  pCtx->mSetEmbeddingsNextn(pValue, pMasked);
}

void mgs_set_embeddings_layer_inp(mgs_context *pCtx, uint32_t pLid, bool pValue) {
  pCtx->mSetEmbeddingsLayerInp(pLid, pValue);
}

void mgs_set_nextn_layer_offset(mgs_context *pCtx, int32_t pOffset) { pCtx->mSetNextnLayerOffset(pOffset); }

mgs_memory_t mgs_get_memory(const struct mgs_context *pCtx) {
  if (!pCtx) {
    return nullptr;
  }

  return pCtx->mGetMemory();
}

float *mgs_get_embeddings_nextn(mgs_context *pCtx) {
  pCtx->mSynchronize();

  return pCtx->mGetEmbeddingsNextn();
}

float *mgs_get_embeddings_nextn_ith(mgs_context *pCtx, int32_t pI) {
  pCtx->mSynchronize();

  return pCtx->mGetEmbeddingsNextnIth(pI);
}

float *mgs_get_embeddings_layer_inp(mgs_context *pCtx, uint32_t pLid) {
  pCtx->mSynchronize();

  return pCtx->mGetEmbeddingsLayerInp(pLid);
}

bool mgs_set_sampler(mgs_context *pCtx, mgs_seq_id pSeqId, mgs_sampler *pSmpl) {
  return pCtx->mSetSampler(pSeqId, pSmpl);
}

mgs_token mgs_get_sampled_token_ith(mgs_context *pCtx, int32_t pI) {
  pCtx->mSynchronize();

  return pCtx->mGetSampledTokenIth(pI);
}

float *mgs_get_sampled_probs_ith(mgs_context *pCtx, int32_t pI) {
  pCtx->mSynchronize();

  return pCtx->mGetSampledProbsIth(pI);
}

float *mgs_get_sampled_logits_ith(mgs_context *pCtx, int32_t pI) {
  pCtx->mSynchronize();

  return pCtx->mGetSampledLogitsIth(pI);
}

mgs_token *mgs_get_sampled_candidates_ith(mgs_context *pCtx, int32_t pI) {
  pCtx->mSynchronize();

  return const_cast<mgs_token *>(pCtx->mGetSampledCandidatesIth(pI));
}

uint32_t mgs_get_sampled_candidates_count_ith(mgs_context *pCtx, int32_t pI) {
  pCtx->mSynchronize();

  return static_cast<uint32_t>(pCtx->mGetSampledCandidatesCount(pI));
}

uint32_t mgs_get_sampled_logits_count_ith(mgs_context *pCtx, int32_t pI) {
  pCtx->mSynchronize();

  return static_cast<uint32_t>(pCtx->mGetSampledLogitsCount(pI));
}

uint32_t mgs_get_sampled_probs_count_ith(mgs_context *pCtx, int32_t pI) {
  pCtx->mSynchronize();

  return static_cast<uint32_t>(pCtx->mGetSampledProbsCount(pI));
}

struct ggml_cgraph *mgs_graph_reserve(struct mgs_context *pCtx, uint32_t pNTokens, uint32_t pNSeqs,
                                      uint32_t pNOutputs) {
  auto vMemory = pCtx->mGetMemory();
  mgs_memory_context_ptr vMctx;
  if (vMemory) {
    vMctx = vMemory->mInitFull();
  }
  return pCtx->mGraphReserve(pNTokens, pNSeqs, pNOutputs, vMctx.get());
}

// mgs adapter API

int32_t mgs_set_adapters_lora(mgs_context *pCtx, mgs_adapter_lora **pAdapters, size_t pNAdapters, float *pScales) {
  if (pAdapters == nullptr || pScales == nullptr) {
    GGML_ASSERT(pNAdapters == 0 && "invalid mgs_set_adapters_lora call");
  }

  pCtx->mSetAdaptersLora(pAdapters, pNAdapters, pScales);

  return 0;
}

int32_t mgs_set_adapter_cvec(mgs_context *pCtx, const float *pData, size_t pLen, int32_t pNEmbd, int32_t pIlStart,
                             int32_t pIlEnd) {
  bool vRes = pCtx->mSetAdapterCvec(pData, pLen, pNEmbd, pIlStart, pIlEnd);

  return vRes ? 0 : -1;
}

//
// memory
//

void mgs_memory_clear(mgs_memory_t pMem, bool pData) {
  if (!pMem) {
    return;
  }

  pMem->mClear(pData);
}

bool mgs_memory_seq_rm(mgs_memory_t pMem, mgs_seq_id pSeqId, mgs_pos p0, mgs_pos p1) {
  if (!pMem) {
    return true;
  }

  return pMem->mSeqRm(pSeqId, p0, p1);
}

void mgs_memory_seq_cp(mgs_memory_t pMem, mgs_seq_id pSeqIdSrc, mgs_seq_id pSeqIdDst, mgs_pos p0, mgs_pos p1) {
  if (!pMem) {
    return;
  }

  pMem->mSeqCp(pSeqIdSrc, pSeqIdDst, p0, p1);
}

void mgs_memory_seq_keep(mgs_memory_t pMem, mgs_seq_id pSeqId) {
  if (!pMem) {
    return;
  }

  pMem->mSeqKeep(pSeqId);
}

void mgs_memory_seq_add(mgs_memory_t pMem, mgs_seq_id pSeqId, mgs_pos p0, mgs_pos p1, mgs_pos pDelta) {
  if (!pMem) {
    return;
  }

  pMem->mSeqAdd(pSeqId, p0, p1, pDelta);
}

void mgs_memory_seq_div(mgs_memory_t pMem, mgs_seq_id pSeqId, mgs_pos p0, mgs_pos p1, int pD) {
  if (!pMem) {
    return;
  }

  pMem->mSeqDiv(pSeqId, p0, p1, pD);
}

mgs_pos mgs_memory_seq_pos_min(mgs_memory_t pMem, mgs_seq_id pSeqId) {
  if (!pMem) {
    return -1;
  }

  return pMem->mSeqPosMin(pSeqId);
}

mgs_pos mgs_memory_seq_pos_max(mgs_memory_t pMem, mgs_seq_id pSeqId) {
  if (!pMem) {
    return -1;
  }

  return pMem->mSeqPosMax(pSeqId);
}

bool mgs_memory_can_shift(mgs_memory_t pMem) {
  if (!pMem) {
    return false;
  }

  return pMem->mGetCanShift();
}

// mgs state API

// Returns the *actual* size of the state.
// Intended to be used when saving to state to a buffer.
size_t mgs_state_get_size(mgs_context *pCtx) { return pCtx->mStateGetSize(); }

size_t mgs_state_get_data(mgs_context *pCtx, uint8_t *pDst, size_t pSize) {
  pCtx->mSynchronize();

  return pCtx->mStateGetData(pDst, pSize);
}

// Sets the state reading from the specified source address
size_t mgs_state_set_data(mgs_context *pCtx, const uint8_t *pSrc, size_t pSize) {
  pCtx->mSynchronize();

  return pCtx->mStateSetData(pSrc, pSize);
}

bool mgs_state_load_file(mgs_context *pCtx, const char *pPathSession, mgs_token *pTokensOut, size_t pNTokenCapacity,
                         size_t *pNTokenCountOut) {
  pCtx->mSynchronize();

  try {
    return pCtx->mStateLoadFile(pPathSession, pTokensOut, pNTokenCapacity, pNTokenCountOut);
  } catch (const std::exception &vErr) {
    MGS_LOG_ERROR("%s: error loading session file: %s\n", __func__, vErr.what());
    return false;
  }
}

bool mgs_state_save_file(mgs_context *pCtx, const char *pPathSession, const mgs_token *pTokens, size_t pNTokenCount) {
  pCtx->mSynchronize();

  try {
    return pCtx->mStateSaveFile(pPathSession, pTokens, pNTokenCount);
  } catch (const std::exception &vErr) {
    MGS_LOG_ERROR("%s: error saving session file: %s\n", __func__, vErr.what());
    return false;
  }
}

size_t mgs_state_seq_get_size(mgs_context *pCtx, mgs_seq_id pSeqId) {
  return mgs_state_seq_get_size_ext(pCtx, pSeqId, 0);
}

size_t mgs_state_seq_get_data(mgs_context *pCtx, uint8_t *pDst, size_t pSize, mgs_seq_id pSeqId) {
  return mgs_state_seq_get_data_ext(pCtx, pDst, pSize, pSeqId, 0);
}

size_t mgs_state_seq_set_data(mgs_context *pCtx, const uint8_t *pSrc, size_t pSize, mgs_seq_id pSeqId) {
  return mgs_state_seq_set_data_ext(pCtx, pSrc, pSize, pSeqId, 0);
}

size_t mgs_state_seq_get_size_ext(mgs_context *pCtx, mgs_seq_id pSeqId, mgs_state_seq_flags pFlags) {
  return pCtx->mStateSeqGetSize(pSeqId, pFlags);
}

size_t mgs_state_seq_get_data_ext(mgs_context *pCtx, uint8_t *pDst, size_t pSize, mgs_seq_id pSeqId,
                                  mgs_state_seq_flags pFlags) {
  pCtx->mSynchronize();

  return pCtx->mStateSeqGetData(pSeqId, pDst, pSize, pFlags);
}
size_t mgs_state_seq_set_data_ext(mgs_context *pCtx, const uint8_t *pSrc, size_t pSize, mgs_seq_id pSeqId,
                                  mgs_state_seq_flags pFlags) {
  pCtx->mSynchronize();

  return pCtx->mStateSeqSetData(pSeqId, pSrc, pSize, pFlags);
}

size_t mgs_state_seq_save_file(mgs_context *pCtx, const char *pFilepath, mgs_seq_id pSeqId, const mgs_token *pTokens,
                               size_t pNTokenCount) {
  pCtx->mSynchronize();

  try {
    return pCtx->mStateSeqSaveFile(pSeqId, pFilepath, pTokens, pNTokenCount);
  } catch (const std::exception &vErr) {
    MGS_LOG_ERROR("%s: error saving sequence state file: %s\n", __func__, vErr.what());
    return 0;
  }
}

size_t mgs_state_seq_load_file(mgs_context *pCtx, const char *pFilepath, mgs_seq_id pDestSeqId, mgs_token *pTokensOut,
                               size_t pNTokenCapacity, size_t *pNTokenCountOut) {
  pCtx->mSynchronize();

  try {
    return pCtx->mStateSeqLoadFile(pDestSeqId, pFilepath, pTokensOut, pNTokenCapacity, pNTokenCountOut);
  } catch (const std::exception &vErr) {
    MGS_LOG_ERROR("%s: error loading sequence state file: %s\n", __func__, vErr.what());
    return 0;
  }
}

///

int32_t mgs_encode(mgs_context *pCtx, mgs_batch pBatch) {
  const int cRet = pCtx->mEncode(pBatch);
  if (cRet != 0) {
    MGS_LOG_ERROR("%s: failed to encode, ret = %d\n", __func__, cRet);
  }

  return cRet;
}

int32_t mgs_decode(mgs_context *pCtx, mgs_batch pBatch) {
  const int cRet = pCtx->mDecode(pBatch);
  if (cRet != 0 && cRet != 1) {
    MGS_LOG_ERROR("%s: failed to decode, ret = %d\n", __func__, cRet);
  }

  return cRet;
}

//
// perf
//

mgs_perf_context_data mgs_perf_context(const mgs_context *pCtx) {
  mgs_perf_context_data vData = {};

  if (pCtx == nullptr) {
    return vData;
  }

  vData = pCtx->mPerfGetData();

  return vData;
}

void mgs_perf_context_print(const mgs_context *pCtx) {
  const auto cData = mgs_perf_context(pCtx);

  const double cTEndMs = 1e-3 * ggml_time_us();

  MGS_LOG_INFO("%s:        load time = %10.2f ms\n", __func__, cData.t_load_ms);
  MGS_LOG_INFO("%s: prompt eval time = %10.2f ms / %5d tokens (%8.2f ms per token, %8.2f tokens per second)\n",
               __func__, cData.t_p_eval_ms, cData.n_p_eval, cData.t_p_eval_ms / cData.n_p_eval,
               1e3 / cData.t_p_eval_ms * cData.n_p_eval);
  MGS_LOG_INFO("%s:        eval time = %10.2f ms / %5d runs   (%8.2f ms per token, %8.2f tokens per second)\n",
               __func__, cData.t_eval_ms, cData.n_eval, cData.t_eval_ms / cData.n_eval,
               1e3 / cData.t_eval_ms * cData.n_eval);
  MGS_LOG_INFO("%s:       total time = %10.2f ms / %5d tokens\n", __func__, (cTEndMs - cData.t_start_ms),
               (cData.n_p_eval + cData.n_eval));
  MGS_LOG_INFO("%s:    graphs reused = %10d\n", __func__, cData.n_reused);
}

void mgs_perf_context_reset(mgs_context *pCtx) { pCtx->mPerfReset(); }

//
// training
//

bool mgs_opt_param_filter_all(const struct ggml_tensor *pTensor, void *pUserdata) {
  GGML_UNUSED(pTensor);
  GGML_UNUSED(pUserdata);
  return true;
}

void mgs_opt_init(struct mgs_context *pCtx, struct mgs_model *pModel, struct mgs_opt_params pLoptParams) {
  pCtx->mOptInit(pModel, pLoptParams);
}

void mgs_opt_epoch(struct mgs_context *pCtx, ggml_opt_dataset_t pDataset, ggml_opt_result_t pResultTrain,
                   ggml_opt_result_t pResultEval, int64_t pIdataSplit, ggml_opt_epoch_callback pCallbackTrain,
                   ggml_opt_epoch_callback pCallbackEval) {
  pCtx->mOptEpoch(pDataset, pResultTrain, pResultEval, pIdataSplit, pCallbackTrain, pCallbackEval);
}

//
// ext
//

mgs_memory_breakdown mgs_get_memory_breakdown(const struct mgs_context *pCtx) { return pCtx->mMemoryBreakdown(); }

mgs_expert_cache_info mgs_context::mGetExpertCacheInfo() const {
  mgs_expert_cache_info vInfo;
  if (vExpertCache) {
    vExpertCache->mFillInfo(vInfo);
  }
  return vInfo;
}

void mgs_context::mSetExpertCacheSize(int64_t pSizeMib) { vExpertCacheRequestMib = std::max<int64_t>(-1, pSizeMib); }

mgs_expert_cache_info mgs_get_expert_cache_info(const struct mgs_context *pCtx) { return pCtx->mGetExpertCacheInfo(); }

void mgs_set_expert_cache_size(struct mgs_context *pCtx, int64_t pSizeMib) { pCtx->mSetExpertCacheSize(pSizeMib); }

mgs_expert_memory_breakdown mgs_context::mGetExpertMemoryBreakdown() const {
  mgs_expert_memory_breakdown dBreakdown;
  if (vExpertCache) {
    std::map<ggml_backend_buffer_type_t, size_t> dCache;
    vExpertCache->mAddMemory(dCache);
    for (const auto &[vBufferType, vBytes] : dCache) {
      dBreakdown[vBufferType].cache += vBytes;
    }
  }
  if (vSched) {
    ggml_backend_buffer_type_t vBufferType = nullptr;
    const size_t vBytes = ggml_backend_sched_get_prefetch_size(vSched.get(), &vBufferType);
    if (vBytes > 0 && vBufferType != nullptr) {
      dBreakdown[vBufferType].prefetch += vBytes;
    }
  }
  return dBreakdown;
}

mgs_expert_memory_breakdown mgs_get_expert_memory_breakdown(const struct mgs_context *pCtx) {
  return pCtx->mGetExpertMemoryBreakdown();
}

mgs_context *mgs_get_ctx_other(struct mgs_context *pCtx) { return pCtx->mGetCparams().vCtxOther; }
