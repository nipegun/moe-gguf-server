#include "ggml-opt.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-impl.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cinttypes>
#include <map>
#include <random>
#include <vector>

struct ggml_opt_dataset {
  struct ggml_context *vCtx = nullptr;
  ggml_backend_buffer_t vBuf = nullptr;
  struct ggml_tensor *vData = nullptr;
  struct ggml_tensor *vLabels = nullptr;

  int64_t vNdata = -1;
  int64_t vNdataShard = -1;
  size_t vNbsData = -1;
  size_t vNbsLabels = -1;

  std::vector<int64_t> vPermutation;
};

struct ggml_opt_context {
  ggml_backend_sched_t vBackendSched = nullptr;
  ggml_cgraph *vAllocatedGraph = nullptr;
  ggml_cgraph *vAllocatedGraphCopy = nullptr;
  struct ggml_context *vCtxStatic = nullptr;
  struct ggml_context *vCtxCpu = nullptr;
  struct ggml_context *vCtxCompute = nullptr;
  struct ggml_context *vCtxCopy = nullptr;
  ggml_backend_buffer_t vBufStatic = nullptr;
  ggml_backend_buffer_t vBufCpu = nullptr;
  std::mt19937 vRng;
  enum ggml_opt_loss_type vLossType;
  enum ggml_opt_build_type vBuildType;
  enum ggml_opt_build_type vBuildTypeAlloc;

  struct ggml_tensor *vInputs = nullptr;
  struct ggml_tensor *vOutputs = nullptr;
  struct ggml_tensor *vLabels = nullptr;

  struct ggml_tensor *vLoss = nullptr;
  struct ggml_tensor *vPred = nullptr;
  struct ggml_tensor *vNcorrect = nullptr;

  struct ggml_cgraph *vGf = nullptr;
  struct ggml_cgraph *vGbGrad = nullptr;
  struct ggml_cgraph *vGbOpt = nullptr;
  bool vStaticGraphs = false;
  bool vEvalReady = false;
  std::vector<struct ggml_tensor *> lGradAccs;
  std::vector<struct ggml_tensor *> lGradM;
  std::vector<struct ggml_tensor *> lGradV;

  int64_t vIter = 1;
  int32_t vOptPeriod = 1;
  int32_t vOptI = 0;
  bool vLossPerDatapoint = false;

  ggml_opt_get_optimizer_params vGetOptPars = nullptr;
  void *vGetOptParsUd = nullptr;
  struct ggml_tensor *vOptStepParams = nullptr; // Stores output of get_opt_pars.

  enum ggml_opt_optimizer_type vOptimizer = GGML_OPT_OPTIMIZER_TYPE_ADAMW;
};

struct ggml_opt_result {
  int64_t vNdata = 0;
  std::vector<float> vLoss;
  std::vector<int32_t> vPred;
  int64_t vNcorrect = 0;

  int64_t vOptPeriod = -1;
  bool vLossPerDatapoint = false;
};

// ====== Dataset ======

ggml_opt_dataset_t ggml_opt_dataset_init(enum ggml_type pTypeData, enum ggml_type pTypeLabel, int64_t pNeDatapoint,
                                         int64_t pNeLabel, int64_t pNdata, int64_t pNdataShard) {
  GGML_ASSERT(pNeDatapoint > 0);
  GGML_ASSERT(pNeLabel >= 0);
  GGML_ASSERT(pNdata > 0);
  GGML_ASSERT(pNdataShard > 0);

  ggml_opt_dataset_t vResult = new ggml_opt_dataset;
  vResult->vNdata = pNdata;
  vResult->vNdataShard = pNdataShard;

  {
    struct ggml_init_params vParams = {
        /*.mem_size   =*/2 * ggml_tensor_overhead(),
        /*.mem_buffer =*/nullptr,
        /*.no_alloc   =*/true,
    };
    vResult->vCtx = ggml_init(vParams);
  }

  vResult->vData = ggml_new_tensor_2d(vResult->vCtx, pTypeData, pNeDatapoint, pNdata);
  vResult->vNbsData = ggml_nbytes(vResult->vData) * pNdataShard / pNdata;

  if (pNeLabel > 0) {
    vResult->vLabels = ggml_new_tensor_2d(vResult->vCtx, pTypeLabel, pNeLabel, pNdata);
    vResult->vNbsLabels = ggml_nbytes(vResult->vLabels) * pNdataShard / pNdata;
  } else {
    vResult->vLabels = nullptr;
    vResult->vNbsLabels = 0;
  }

  vResult->vBuf = ggml_backend_alloc_ctx_tensors_from_buft(vResult->vCtx, ggml_backend_cpu_buffer_type());

  const int64_t cNshards = pNdata / pNdataShard;
  vResult->vPermutation.resize(cNshards);
  for (int64_t vI = 0; vI < cNshards; ++vI) {
    vResult->vPermutation[vI] = vI;
  }
  return vResult;
}

void ggml_opt_dataset_free(ggml_opt_dataset_t pDataset) {
  ggml_backend_buffer_free(pDataset->vBuf);
  ggml_free(pDataset->vCtx);
  delete pDataset;
}

int64_t ggml_opt_dataset_ndata(ggml_opt_dataset_t pDataset) { return pDataset->vNdata; }

struct ggml_tensor *ggml_opt_dataset_data(ggml_opt_dataset_t pDataset) { return pDataset->vData; }

struct ggml_tensor *ggml_opt_dataset_labels(ggml_opt_dataset_t pDataset) { return pDataset->vLabels; }

void ggml_opt_dataset_shuffle(ggml_opt_context_t pOptCtx, ggml_opt_dataset_t pDataset, int64_t pIdata) {
  GGML_ASSERT(pIdata <= pDataset->vNdata);

  if (pIdata < 0) {
    std::shuffle(pDataset->vPermutation.begin(), pDataset->vPermutation.end(), pOptCtx->vRng);
    return;
  }

  GGML_ASSERT(pIdata % pDataset->vNdataShard == 0);
  const int64_t cIshardMax = pIdata / pDataset->vNdataShard;
  std::shuffle(pDataset->vPermutation.begin(), pDataset->vPermutation.begin() + cIshardMax, pOptCtx->vRng);
}

void ggml_opt_dataset_get_batch(ggml_opt_dataset_t pDataset, struct ggml_tensor *pDataBatch,
                                struct ggml_tensor *pLabelsBatch, int64_t pIbatch) {
  GGML_ASSERT(pDataBatch && ggml_is_contiguous(pDataBatch));
  GGML_ASSERT(!pLabelsBatch || ggml_is_contiguous(pLabelsBatch));
  GGML_ASSERT((pLabelsBatch == nullptr) == (pDataset->vLabels == nullptr));
  GGML_ASSERT(pDataBatch->type == pDataset->vData->type);
  GGML_ASSERT(!pLabelsBatch || pLabelsBatch->type == pDataset->vLabels->type);

  const size_t cNbDataBatch = ggml_nbytes(pDataBatch);
  GGML_ASSERT(cNbDataBatch % pDataset->vNbsData == 0);
  const int64_t cShardsPerBatch = cNbDataBatch / pDataset->vNbsData;

  if (pLabelsBatch) {
    const size_t cNbLabelsBatch = ggml_nbytes(pLabelsBatch);
    GGML_ASSERT(cNbLabelsBatch == cShardsPerBatch * pDataset->vNbsLabels);
  }

  GGML_ASSERT((pIbatch + 1) * cShardsPerBatch <= int64_t(pDataset->vPermutation.size()));

  for (int64_t vIshardBatch = 0; vIshardBatch < cShardsPerBatch; ++vIshardBatch) {
    const int64_t cIshard = pDataset->vPermutation[pIbatch * cShardsPerBatch + vIshardBatch];

    const char *vPtrData = (const char *)pDataset->vData->data + cIshard * pDataset->vNbsData;
    ggml_backend_tensor_set(pDataBatch, vPtrData, vIshardBatch * pDataset->vNbsData, pDataset->vNbsData);

    if (!pLabelsBatch) {
      continue;
    }

    const char *vPtrLabels = (const char *)pDataset->vLabels->data + cIshard * pDataset->vNbsLabels;
    ggml_backend_tensor_set(pLabelsBatch, vPtrLabels, vIshardBatch * pDataset->vNbsLabels, pDataset->vNbsLabels);
  }
}

void ggml_opt_dataset_get_batch_host(ggml_opt_dataset_t pDataset, void *pDataBatch, size_t pNbDataBatch,
                                     void *pLabelsBatch, int64_t pIbatch) {
  GGML_ASSERT((pLabelsBatch == nullptr) == (pDataset->vLabels == nullptr));
  GGML_ASSERT(pNbDataBatch % pDataset->vNbsData == 0);

  const int64_t cShardsPerBatch = pNbDataBatch / pDataset->vNbsData;

  GGML_ASSERT((pIbatch + 1) * cShardsPerBatch <= int64_t(pDataset->vPermutation.size()));

  for (int64_t vIshardBatch = 0; vIshardBatch < cShardsPerBatch; ++vIshardBatch) {
    const int64_t cIshard = pDataset->vPermutation[pIbatch * cShardsPerBatch + vIshardBatch];

    const char *vPtrData = (const char *)pDataset->vData->data + cIshard * pDataset->vNbsData;
    char *vPtrDataBatch = (char *)pDataBatch + vIshardBatch * pDataset->vNbsData;
    memcpy(vPtrDataBatch, vPtrData, pDataset->vNbsData);

    if (!pLabelsBatch) {
      continue;
    }

    const char *vPtrLabels = (const char *)pDataset->vLabels->data + cIshard * pDataset->vNbsLabels;
    char *vPtrLabelsBatch = (char *)pLabelsBatch + vIshardBatch * pDataset->vNbsLabels;
    memcpy(vPtrLabelsBatch, vPtrLabels, pDataset->vNbsLabels);
  }
}

// ====== Model / Context ======

struct ggml_opt_optimizer_params ggml_opt_get_default_optimizer_params(void *pUserdata) {
  GGML_UNUSED(pUserdata);

  ggml_opt_optimizer_params vResult;

  vResult.adamw.alpha = 0.001f;
  vResult.adamw.beta1 = 0.9f;
  vResult.adamw.beta2 = 0.999f;
  vResult.adamw.eps = 1e-8f;
  vResult.adamw.wd = 0.0f;

  vResult.sgd.alpha = 1e-3f;
  vResult.sgd.wd = 0.0f;

  return vResult;
}

struct ggml_opt_optimizer_params ggml_opt_get_constant_optimizer_params(void *pUserdata) {
  return *((struct ggml_opt_optimizer_params *)pUserdata);
}

struct ggml_opt_params ggml_opt_default_params(ggml_backend_sched_t pBackendSched, enum ggml_opt_loss_type pLossType) {
  return {
      /*backend_sched   =*/pBackendSched,
      /*ctx_compute     =*/nullptr,
      /*inputs          =*/nullptr,
      /*logits          =*/nullptr,
      /*loss_type       =*/pLossType,
      /*build_type      =*/GGML_OPT_BUILD_TYPE_OPT,
      /*opt_period      =*/1,
      /*get_opt_pars    =*/ggml_opt_get_default_optimizer_params,
      /*get_opt_pars_ud =*/nullptr,
      /*optimizer       =*/GGML_OPT_OPTIMIZER_TYPE_ADAMW,
  };
}

static ggml_tensor *fMapTensor(std::map<ggml_tensor *, ggml_tensor *> &pTensorMap, ggml_context *pCtx,
                               ggml_tensor *pTensor) {
  if (!pTensor) {
    return nullptr;
  }

  if (pTensorMap.find(pTensor) != pTensorMap.end()) {
    return pTensorMap[pTensor];
  }

  ggml_tensor *vNewTensor = ggml_dup_tensor(pCtx, pTensor);
  pTensorMap[pTensor] = vNewTensor;

  vNewTensor->op = pTensor->op;
  for (int vI = 0; vI < GGML_MAX_DIMS; vI++) {
    vNewTensor->nb[vI] = pTensor->nb[vI];
  }
  vNewTensor->flags = pTensor->flags;
  memcpy(vNewTensor->op_params, pTensor->op_params, sizeof(pTensor->op_params));
  strcpy(vNewTensor->name, pTensor->name);
  vNewTensor->data = pTensor->data;
  vNewTensor->buffer = pTensor->buffer;
  vNewTensor->extra = pTensor->extra;
  vNewTensor->view_offs = pTensor->view_offs;
  vNewTensor->view_src = fMapTensor(pTensorMap, pCtx, pTensor->view_src);
  for (int vI = 0; vI < GGML_MAX_SRC; vI++) {
    vNewTensor->src[vI] = fMapTensor(pTensorMap, pCtx, pTensor->src[vI]);
  }

  return vNewTensor;
}

static ggml_cgraph *fDupGraph(ggml_context *pCtx, ggml_cgraph *pSrc) {
  std::map<ggml_tensor *, ggml_tensor *> dTensorMap;

  ggml_cgraph *vDst = ggml_new_graph_custom(pCtx, pSrc->size, /*grads =*/true);

  for (int vI = 0; vI < pSrc->n_leafs; vI++) {
    ggml_build_forward_expand(vDst, fMapTensor(dTensorMap, pCtx, pSrc->leafs[vI]));
  }
  GGML_ASSERT(vDst->n_leafs == pSrc->n_leafs);
  for (int vI = 0; vI < pSrc->n_nodes; vI++) {
    ggml_build_forward_expand(vDst, fMapTensor(dTensorMap, pCtx, pSrc->nodes[vI]));
  }
  GGML_ASSERT(vDst->n_nodes == pSrc->n_nodes);
  for (int vI = 0; vI < pSrc->n_nodes; ++vI) {
    const size_t cIgradSrc = ggml_hash_find(&pSrc->visited_hash_set, pSrc->nodes[vI]);
    const size_t cIgradDst = ggml_hash_find(&vDst->visited_hash_set, vDst->nodes[vI]);

    GGML_ASSERT(cIgradSrc != GGML_HASHSET_FULL);
    GGML_ASSERT(ggml_bitset_get(pSrc->visited_hash_set.used, cIgradSrc));
    GGML_ASSERT(cIgradDst != GGML_HASHSET_FULL);
    GGML_ASSERT(ggml_bitset_get(vDst->visited_hash_set.used, cIgradDst));

    vDst->grads[cIgradDst] = pSrc->grads[cIgradSrc];
    vDst->grad_accs[cIgradDst] = pSrc->grad_accs[cIgradSrc];
  }

  return vDst;
}

static void fGgmlOptBuild(ggml_opt_context_t pOptCtx) {
  GGML_ASSERT(pOptCtx->vCtxCompute &&
              "no compute context set, either use static graphs or set one with ggml_opt_prepare_alloc");
  GGML_ASSERT((!pOptCtx->vStaticGraphs || pOptCtx->vInputs->data) &&
              "when using static graphs the inputs must be allocated statically");

  const enum ggml_opt_optimizer_type cOptimizer = pOptCtx->vOptimizer;

  const bool cAccumulate =
      pOptCtx->vBuildTypeAlloc >= GGML_OPT_BUILD_TYPE_GRAD &&
      !(pOptCtx->vStaticGraphs && pOptCtx->vBuildTypeAlloc == GGML_OPT_BUILD_TYPE_OPT && pOptCtx->vOptPeriod == 1);

  const bool cNeedMomenta =
      pOptCtx->vBuildTypeAlloc == GGML_OPT_BUILD_TYPE_OPT && pOptCtx->vOptimizer == GGML_OPT_OPTIMIZER_TYPE_ADAMW;

  ggml_set_input(pOptCtx->vInputs);
  ggml_set_output(pOptCtx->vOutputs);

  int vNParam = 0;
  for (int vI = 0; vI < pOptCtx->vGf->n_nodes; ++vI) {
    const struct ggml_tensor *vNode = pOptCtx->vGf->nodes[vI];
    if (vNode->flags & GGML_TENSOR_FLAG_PARAM) {
      vNParam++;
    }
    GGML_ASSERT(!(vNode->flags & GGML_TENSOR_FLAG_LOSS) && "support for extra loss terms not implemented");
  }

  if (!pOptCtx->vCtxStatic) {
    // The static context is used for:
    //   - gradients (1 per loss, 1 tensor per param if using gradient accumulation)
    //   - optimizer momenta (2 tensors per param)
    //   - labels (if using static graphs)
    //   - loss (if using static graphs, up to 5 tensors)
    //   - pred (if using static graphs)
    //   - ncorrect (if using static graphs, 2 tensors).
    constexpr size_t cNLoss = 1;
    const size_t cTensorsPerParam = (cAccumulate ? 1 : 0) + (cNeedMomenta ? 2 : 0);
    const size_t cTensorsConst = pOptCtx->vStaticGraphs ? 9 : 0;
    const size_t cSizeMeta = (cNLoss + cTensorsPerParam * vNParam + cTensorsConst) * ggml_tensor_overhead();
    struct ggml_init_params vParams = {
        /*.mem_size   =*/cSizeMeta,
        /*.mem_buffer =*/nullptr,
        /*.no_alloc   =*/true,
    };
    pOptCtx->vCtxStatic = ggml_init(vParams);
  }
  GGML_ASSERT(pOptCtx->vBuildType <= pOptCtx->vBuildTypeAlloc);

  {
    // The cpu context is allocated statically if using static graphs, dynamically otherwise.
    // It is used for:
    //   - optimizer parameters (1 shared for all optimizer invocations)
    const size_t cSizeMeta = 1 * ggml_tensor_overhead();
    struct ggml_init_params vParams = {
        /*.mem_size   =*/cSizeMeta,
        /*.mem_buffer =*/nullptr,
        /*.no_alloc   =*/true,
    };
    ggml_free(pOptCtx->vCtxCpu);
    pOptCtx->vCtxCpu = ggml_init(vParams);

    ggml_backend_buffer_free(pOptCtx->vBufCpu);
    pOptCtx->vBufCpu = nullptr;
  }

  struct ggml_context *vCtxResults = pOptCtx->vStaticGraphs ? pOptCtx->vCtxStatic : pOptCtx->vCtxCompute;

  switch (pOptCtx->vLossType) {
  case GGML_OPT_LOSS_TYPE_MEAN: {
    pOptCtx->vLoss = ggml_sum(vCtxResults, pOptCtx->vOutputs);
    ggml_set_name(pOptCtx->vLoss, "loss_sum");
    const float cScale = 1.0f / (pOptCtx->vOptPeriod * ggml_nelements(pOptCtx->vOutputs));
    pOptCtx->vLoss = ggml_scale(vCtxResults, pOptCtx->vLoss, cScale);
    ggml_set_name(pOptCtx->vLoss, "loss_mean");
    pOptCtx->vLossPerDatapoint = true;
    break;
  }
  case GGML_OPT_LOSS_TYPE_SUM: {
    pOptCtx->vLoss = ggml_sum(vCtxResults, pOptCtx->vOutputs);
    ggml_set_name(pOptCtx->vLoss, "loss_sum");
    pOptCtx->vLossPerDatapoint = false;
    break;
  }
  case GGML_OPT_LOSS_TYPE_CROSS_ENTROPY: {
    pOptCtx->vLabels = ggml_dup_tensor(vCtxResults, pOptCtx->vOutputs);
    ggml_set_input(pOptCtx->vLabels);
    ggml_set_name(pOptCtx->vLabels, "labels");
    pOptCtx->vLoss = ggml_cross_entropy_loss(vCtxResults, pOptCtx->vOutputs, pOptCtx->vLabels);
    ggml_set_name(pOptCtx->vLoss, "loss_cross_entropy");
    if (pOptCtx->vOptPeriod > 1) {
      pOptCtx->vLoss = ggml_scale(vCtxResults, pOptCtx->vLoss, 1.0f / pOptCtx->vOptPeriod);
      ggml_set_name(pOptCtx->vLoss, "loss_cross_entropy_scaled");
    }
    pOptCtx->vLossPerDatapoint = true;
    break;
  }
  case GGML_OPT_LOSS_TYPE_MEAN_SQUARED_ERROR: {
    pOptCtx->vLabels = ggml_dup_tensor(vCtxResults, pOptCtx->vOutputs);
    ggml_set_input(pOptCtx->vLabels);
    ggml_set_name(pOptCtx->vLabels, "labels");
    pOptCtx->vLoss = ggml_sub(vCtxResults, pOptCtx->vOutputs, pOptCtx->vLabels);
    ggml_set_name(pOptCtx->vLoss, "loss_error");
    pOptCtx->vLoss = ggml_sqr(vCtxResults, pOptCtx->vLoss);
    ggml_set_name(pOptCtx->vLoss, "loss_squared_error");
    pOptCtx->vLoss = ggml_sum(vCtxResults, pOptCtx->vLoss);
    ggml_set_name(pOptCtx->vLoss, "loss_sum_squared_error");
    const float cScale = 1.0f / (pOptCtx->vOptPeriod * ggml_nelements(pOptCtx->vOutputs));
    pOptCtx->vLoss = ggml_scale(vCtxResults, pOptCtx->vLoss, cScale);
    ggml_set_name(pOptCtx->vLoss, "loss_mean_squared_error");
    pOptCtx->vLossPerDatapoint = true;
    break;
  }
  }
  ggml_set_output(pOptCtx->vLoss);
  ggml_set_loss(pOptCtx->vLoss);
  ggml_build_forward_expand(pOptCtx->vGf, pOptCtx->vLoss);

  if (pOptCtx->vLossType == GGML_OPT_LOSS_TYPE_CROSS_ENTROPY) {
    pOptCtx->vPred = ggml_argmax(vCtxResults, pOptCtx->vOutputs);
    ggml_set_name(pOptCtx->vPred, "pred");
    ggml_set_output(pOptCtx->vPred);
    ggml_build_forward_expand(pOptCtx->vGf, pOptCtx->vPred);

    pOptCtx->vNcorrect = ggml_count_equal(vCtxResults, pOptCtx->vPred, ggml_argmax(vCtxResults, pOptCtx->vLabels));
    ggml_set_name(pOptCtx->vNcorrect, "ncorrect");
    ggml_set_output(pOptCtx->vNcorrect);
    ggml_build_forward_expand(pOptCtx->vGf, pOptCtx->vNcorrect);
  }

  if (pOptCtx->vBufStatic) {
    if (pOptCtx->vBuildType == GGML_OPT_BUILD_TYPE_FORWARD) {
      return;
    }
  } else if (pOptCtx->vBuildTypeAlloc == GGML_OPT_BUILD_TYPE_FORWARD) {
    pOptCtx->vBufStatic =
        ggml_backend_alloc_ctx_tensors(pOptCtx->vCtxStatic, ggml_backend_sched_get_backend(pOptCtx->vBackendSched, 0));
    return;
  }

  if (pOptCtx->lGradAccs.empty()) {
    GGML_ASSERT(pOptCtx->vBuildTypeAlloc >= GGML_OPT_BUILD_TYPE_GRAD);

    const int cNNodes = pOptCtx->vGf->n_nodes;
    pOptCtx->lGradAccs.resize(cNNodes);
    for (int vI = 0; vI < cNNodes; ++vI) {
      ggml_tensor *vNode = pOptCtx->vGf->nodes[vI];
      if ((cAccumulate && (vNode->flags & GGML_TENSOR_FLAG_PARAM)) || (vNode->flags & GGML_TENSOR_FLAG_LOSS)) {
        pOptCtx->lGradAccs[vI] = ggml_new_tensor(pOptCtx->vCtxStatic, GGML_TYPE_F32, GGML_MAX_DIMS, vNode->ne);
      } else {
        pOptCtx->lGradAccs[vI] = nullptr;
      }
    }

    if (cNeedMomenta && pOptCtx->vBuildTypeAlloc >= GGML_OPT_BUILD_TYPE_OPT) {
      pOptCtx->lGradM.resize(cNNodes);
      pOptCtx->lGradV.resize(cNNodes);
      for (int vI = 0; vI < cNNodes; ++vI) {
        ggml_tensor *vNode = pOptCtx->vGf->nodes[vI];
        if (vNode->flags & GGML_TENSOR_FLAG_PARAM) {
          pOptCtx->lGradM[vI] = ggml_new_tensor(pOptCtx->vCtxStatic, GGML_TYPE_F32, GGML_MAX_DIMS, vNode->ne);
          pOptCtx->lGradV[vI] = ggml_new_tensor(pOptCtx->vCtxStatic, GGML_TYPE_F32, GGML_MAX_DIMS, vNode->ne);
        } else {
          pOptCtx->lGradM[vI] = nullptr;
          pOptCtx->lGradV[vI] = nullptr;
        }
      }
    }
  }

  // gb_grad == graph backward gradients, forward pass, then backward pass to calculate gradients.
  pOptCtx->vGbGrad = ggml_graph_dup(pOptCtx->vCtxCompute, pOptCtx->vGf, /*force_grads =*/true);
  ggml_build_backward_expand(pOptCtx->vCtxCompute, pOptCtx->vGbGrad, pOptCtx->lGradAccs.data());

  if (pOptCtx->vBufStatic) {
    if (pOptCtx->vBuildType == GGML_OPT_BUILD_TYPE_GRAD) {
      return;
    }
  } else if (pOptCtx->vBuildTypeAlloc == GGML_OPT_BUILD_TYPE_GRAD) {
    pOptCtx->vBufStatic =
        ggml_backend_alloc_ctx_tensors(pOptCtx->vCtxStatic, ggml_backend_sched_get_backend(pOptCtx->vBackendSched, 0));
    ggml_graph_reset(pOptCtx->vGbGrad);
  }

  GGML_ASSERT(pOptCtx->vBuildTypeAlloc == GGML_OPT_BUILD_TYPE_OPT);

  // gb_opt == graph backward optimize, forward pass, then backward pass to calculate gradients, then optimizer step.
  pOptCtx->vGbOpt = ggml_graph_dup(pOptCtx->vCtxCompute, pOptCtx->vGbGrad, /*force_grads =*/true);

  pOptCtx->vOptStepParams = ggml_new_tensor_1d(pOptCtx->vCtxCpu, GGML_TYPE_F32, cNeedMomenta ? 7 : 2);
  ggml_tensor *vAdamwParams = pOptCtx->vOptStepParams;
  ggml_set_input(vAdamwParams);
  const char *vOptimizerName = ggml_opt_optimizer_name(pOptCtx->vOptimizer);
  ggml_format_name(vAdamwParams, "%s_params", vOptimizerName);
  for (int vI = pOptCtx->vGf->n_nodes - 1; vI >= 0; --vI) {
    struct ggml_tensor *vNode = pOptCtx->vGbOpt->nodes[vI];
    struct ggml_tensor *vGrad = ggml_graph_get_grad(pOptCtx->vGbOpt, vNode);

    if (vGrad && (vNode->flags & GGML_TENSOR_FLAG_PARAM)) {
      struct ggml_tensor *vM = nullptr;
      struct ggml_tensor *v_ = nullptr;
      if (cNeedMomenta) {
        vM = pOptCtx->lGradM[vI];
        v_ = pOptCtx->lGradV[vI];
        ggml_format_name(vM, "AdamW m for %s", vNode->name);
        ggml_format_name(v_, "AdamW v for %s", vNode->name);
      }
      struct ggml_tensor *vOptStep;
      switch (cOptimizer) {
      case GGML_OPT_OPTIMIZER_TYPE_ADAMW:
        vOptStep = ggml_opt_step_adamw(pOptCtx->vCtxCompute, vNode, vGrad, vM, v_, vAdamwParams);
        break;
      case GGML_OPT_OPTIMIZER_TYPE_SGD:
        vOptStep = ggml_opt_step_sgd(pOptCtx->vCtxCompute, vNode, vGrad, vAdamwParams);
        break;
      default:
        GGML_ABORT("fatal error");
      }
      ggml_format_name(vOptStep, "%s step for %s", vOptimizerName, vNode->name);
      ggml_build_forward_expand(pOptCtx->vGbOpt, vOptStep);
    }
  }

  if (!pOptCtx->vBufStatic) {
    pOptCtx->vBufStatic =
        ggml_backend_alloc_ctx_tensors(pOptCtx->vCtxStatic, ggml_backend_sched_get_backend(pOptCtx->vBackendSched, 0));
    ggml_graph_reset(pOptCtx->vGbOpt);
  }

  pOptCtx->vBufCpu = ggml_backend_alloc_ctx_tensors_from_buft(pOptCtx->vCtxCpu, ggml_backend_cpu_buffer_type());
}

ggml_opt_context_t ggml_opt_init(struct ggml_opt_params pParams) {
  ggml_opt_context_t vResult = new struct ggml_opt_context;
  vResult->vBackendSched = pParams.backend_sched;
  vResult->vCtxCompute = pParams.ctx_compute;
  vResult->vLossType = pParams.loss_type;
  vResult->vBuildType = pParams.build_type;
  vResult->vBuildTypeAlloc = pParams.build_type;
  vResult->vInputs = pParams.inputs;
  vResult->vOutputs = pParams.outputs;
  vResult->vOptPeriod = pParams.opt_period;
  vResult->vGetOptPars = pParams.get_opt_pars;
  vResult->vGetOptParsUd = pParams.get_opt_pars_ud;
  vResult->vOptimizer = pParams.optimizer;

  GGML_ASSERT(vResult->vOptPeriod >= 1);

  vResult->vStaticGraphs = vResult->vCtxCompute;

  if (!vResult->vStaticGraphs) {
    GGML_ASSERT(!vResult->vInputs);
    GGML_ASSERT(!vResult->vOutputs);
    return vResult;
  }

  GGML_ASSERT(vResult->vInputs);
  GGML_ASSERT(vResult->vOutputs);

  vResult->vGf = ggml_new_graph_custom(vResult->vCtxCompute, GGML_DEFAULT_GRAPH_SIZE, /*grads =*/true); // Forward pass.
  ggml_build_forward_expand(vResult->vGf, vResult->vOutputs);

  fGgmlOptBuild(vResult);

  return vResult;
}

void ggml_opt_free(ggml_opt_context_t pOptCtx) {
  if (pOptCtx == nullptr) {
    return;
  }
  ggml_backend_buffer_free(pOptCtx->vBufStatic);
  ggml_backend_buffer_free(pOptCtx->vBufCpu);
  ggml_free(pOptCtx->vCtxStatic);
  ggml_free(pOptCtx->vCtxCpu);
  ggml_free(pOptCtx->vCtxCopy);
  delete pOptCtx;
}

void ggml_opt_reset(ggml_opt_context_t pOptCtx, bool pOptimizer) {
  if (pOptimizer) {
    ggml_graph_reset(pOptCtx->vGbOpt);
    pOptCtx->vIter = 1;
  } else {
    ggml_graph_reset(pOptCtx->vGbGrad);
  }
}

bool ggml_opt_static_graphs(ggml_opt_context_t pOptCtx) { return pOptCtx->vStaticGraphs; }

struct ggml_tensor *ggml_opt_inputs(ggml_opt_context_t pOptCtx) { return pOptCtx->vInputs; }

struct ggml_tensor *ggml_opt_outputs(ggml_opt_context_t pOptCtx) { return pOptCtx->vOutputs; }

struct ggml_tensor *ggml_opt_labels(ggml_opt_context_t pOptCtx) { return pOptCtx->vLabels; }

struct ggml_tensor *ggml_opt_loss(ggml_opt_context_t pOptCtx) { return pOptCtx->vLoss; }

struct ggml_tensor *ggml_opt_pred(ggml_opt_context_t pOptCtx) { return pOptCtx->vPred; }

struct ggml_tensor *ggml_opt_ncorrect(ggml_opt_context_t pOptCtx) { return pOptCtx->vNcorrect; }

struct ggml_tensor *ggml_opt_grad_acc(ggml_opt_context_t pOptCtx, struct ggml_tensor *pNode) {
  return ggml_graph_get_grad_acc(pOptCtx->vGbOpt, pNode);
}

// ====== Optimization Result ======

ggml_opt_result_t ggml_opt_result_init() { return new ggml_opt_result; }

void ggml_opt_result_free(ggml_opt_result_t pResult) { delete pResult; }

void ggml_opt_result_reset(ggml_opt_result_t pResult) {
  pResult->vNdata = 0;
  pResult->vLoss.clear();
  pResult->vPred.clear();
  pResult->vNcorrect = 0;
}

void ggml_opt_result_ndata(ggml_opt_result_t pResult, int64_t *pNdata) { *pNdata = pResult->vNdata; }

void ggml_opt_result_loss(ggml_opt_result_t pResult, double *pLoss, double *pUnc) {
  const int64_t cNbatches = pResult->vLoss.size(); // Number of physical batches.

  if (cNbatches == 0) {
    *pLoss = 0.0;
    *pUnc = NAN;
    return;
  }

  double vSum = 0.0;
  double vSumSquared = 0.0;

  for (const float &vLoss : pResult->vLoss) {
    // If the loss is per datapoint it was scaled by 1.0f/opt_period for each physical batch.
    const float cLossScaled = pResult->vLossPerDatapoint ? vLoss * pResult->vOptPeriod : vLoss;
    vSum += cLossScaled;
    vSumSquared += cLossScaled * cLossScaled;
  }

  const double cMean = vSum / cNbatches;
  *pLoss = pResult->vLossPerDatapoint ? cMean : vSum;

  if (!pUnc) {
    return;
  }

  if (cNbatches < 2) {
    *pUnc = NAN;
    return;
  }

  const double cVarSum =
      vSumSquared / cNbatches - cMean * cMean; // variance without Bessel's correction, i.e. nbatches/(nbatches-1)
  *pUnc = pResult->vLossPerDatapoint ? sqrt(cVarSum / (cNbatches - 1)) : sqrt(cVarSum * cNbatches / (cNbatches - 1));
}

void ggml_opt_result_pred(ggml_opt_result_t pResult, int32_t *pPred) {
  for (size_t vI = 0; vI < pResult->vPred.size(); ++vI) {
    pPred[vI] = pResult->vPred[vI];
  }
}

void ggml_opt_result_accuracy(ggml_opt_result_t pResult, double *pAccuracy, double *pUnc) {
  *pAccuracy = pResult->vNcorrect >= 0 ? double(pResult->vNcorrect) / double(pResult->vNdata) : NAN;

  if (!pUnc) {
    return;
  }

  *pUnc = pResult->vNcorrect >= 0 && pResult->vNdata >= 2
              ? sqrt((*pAccuracy) * (1.0 - (*pAccuracy)) / double(pResult->vNdata - 1))
              : NAN;
}

// ====== Computation ======

void ggml_opt_prepare_alloc(ggml_opt_context_t pOptCtx, struct ggml_context *pCtxCompute, struct ggml_cgraph *pGf,
                            struct ggml_tensor *pInputs, struct ggml_tensor *pOutputs) {
  GGML_ASSERT(!pOptCtx->vStaticGraphs);
  pOptCtx->vCtxCompute = pCtxCompute;
  pOptCtx->vGf = pGf;
  pOptCtx->vInputs = pInputs;
  pOptCtx->vOutputs = pOutputs;
}

void ggml_opt_alloc(ggml_opt_context_t pOptCtx, bool pBackward) {
  GGML_ASSERT(!pOptCtx->vEvalReady);
  if (pOptCtx->vBuildType == GGML_OPT_BUILD_TYPE_OPT && pOptCtx->vOptPeriod > 1 && pOptCtx->vOptI == 0) {
    ggml_graph_reset(pOptCtx->vGbGrad);
  }
  if (pBackward) {
    const int32_t cOptINext = (pOptCtx->vOptI + 1) % pOptCtx->vOptPeriod;
    pOptCtx->vBuildType = cOptINext == 0 ? GGML_OPT_BUILD_TYPE_OPT : GGML_OPT_BUILD_TYPE_GRAD;
  } else {
    pOptCtx->vBuildType = GGML_OPT_BUILD_TYPE_FORWARD;
  }

  if (!pOptCtx->vStaticGraphs) {
    fGgmlOptBuild(pOptCtx);
  }

  struct ggml_cgraph *vGraph = nullptr;
  switch (pOptCtx->vBuildType) {
  case GGML_OPT_BUILD_TYPE_FORWARD: {
    vGraph = pOptCtx->vGf;
  } break;
  case GGML_OPT_BUILD_TYPE_GRAD: {
    vGraph = pOptCtx->vGbGrad;
  } break;
  case GGML_OPT_BUILD_TYPE_OPT: {
    vGraph = pOptCtx->vGbOpt;
  } break;
  }
  GGML_ASSERT(vGraph);

  if (pOptCtx->vAllocatedGraph == vGraph) {
    pOptCtx->vEvalReady = true;
    return;
  }

  ggml_backend_sched_reset(pOptCtx->vBackendSched); // clear allocation of previous graph

  if (pOptCtx->vStaticGraphs) {
    ggml_init_params vParams = {
        /*.mem_size   =*/vGraph->size * ggml_tensor_overhead() +
            ggml_graph_overhead_custom(vGraph->size, vGraph->grads),
        /*.mem_buffer =*/nullptr,
        /*.no_alloc   =*/true,
    };
    ggml_free(pOptCtx->vCtxCopy);
    pOptCtx->vCtxCopy = ggml_init(vParams);

    pOptCtx->vAllocatedGraphCopy = fDupGraph(pOptCtx->vCtxCopy, vGraph);
  } else {
    pOptCtx->vAllocatedGraphCopy = vGraph;
  }

  ggml_backend_sched_alloc_graph(pOptCtx->vBackendSched, pOptCtx->vAllocatedGraphCopy);
  pOptCtx->vAllocatedGraph = vGraph;

  pOptCtx->vEvalReady = true;
}

void ggml_opt_eval(ggml_opt_context_t pOptCtx, ggml_opt_result_t pResult) {
  GGML_ASSERT(pOptCtx->vEvalReady);
  if (pOptCtx->vAllocatedGraph == pOptCtx->vGbOpt) {
    const ggml_opt_optimizer_params &vOptPars = pOptCtx->vGetOptPars(pOptCtx->vGetOptParsUd);

    switch (pOptCtx->vOptimizer) {
    case GGML_OPT_OPTIMIZER_TYPE_ADAMW: {
      GGML_ASSERT(vOptPars.adamw.alpha > 0.0f);
      GGML_ASSERT(vOptPars.adamw.beta1 >= 0.0f);
      GGML_ASSERT(vOptPars.adamw.beta1 <= 1.0f);
      GGML_ASSERT(vOptPars.adamw.beta2 >= 0.0f);
      GGML_ASSERT(vOptPars.adamw.beta2 <= 1.0f);
      GGML_ASSERT(vOptPars.adamw.eps >= 0.0f);
      GGML_ASSERT(vOptPars.adamw.wd >= 0.0f);
      GGML_ASSERT(vOptPars.adamw.wd <= 1.0f);

      // beta1, beta2 after applying warmup
      const float cBeta1h = 1.0f / (1.0f - powf(vOptPars.adamw.beta1, pOptCtx->vIter));
      const float cBeta2h = 1.0f / (1.0f - powf(vOptPars.adamw.beta2, pOptCtx->vIter));

      float *vAdamwParData = ggml_get_data_f32(pOptCtx->vOptStepParams);
      vAdamwParData[0] = vOptPars.adamw.alpha;
      vAdamwParData[1] = vOptPars.adamw.beta1;
      vAdamwParData[2] = vOptPars.adamw.beta2;
      vAdamwParData[3] = vOptPars.adamw.eps;
      vAdamwParData[4] = vOptPars.adamw.wd;
      vAdamwParData[5] = cBeta1h;
      vAdamwParData[6] = cBeta2h;
    } break;
    case GGML_OPT_OPTIMIZER_TYPE_SGD: {
      GGML_ASSERT(vOptPars.sgd.alpha > 0.0f);
      GGML_ASSERT(vOptPars.sgd.wd >= 0.0f);
      GGML_ASSERT(vOptPars.sgd.wd <= 1.0f);
      float *vSgd = ggml_get_data_f32(pOptCtx->vOptStepParams);
      vSgd[0] = vOptPars.sgd.alpha;
      vSgd[1] = vOptPars.sgd.wd;
    } break;
    default:
      GGML_ABORT("fatal error");
    }
  }

  ggml_backend_sched_graph_compute(pOptCtx->vBackendSched, pOptCtx->vAllocatedGraphCopy);
  pOptCtx->vIter += pOptCtx->vAllocatedGraph == pOptCtx->vGbOpt;
  pOptCtx->vOptI = (pOptCtx->vOptI + 1) % pOptCtx->vOptPeriod;

  if (!pOptCtx->vStaticGraphs) {
    pOptCtx->vGf = nullptr;
    pOptCtx->vGbGrad = nullptr;
    pOptCtx->vGbOpt = nullptr;
    pOptCtx->vAllocatedGraph = nullptr;
    pOptCtx->vAllocatedGraphCopy = nullptr;
  }

  pOptCtx->vEvalReady = false;

  if (!pResult) {
    return;
  }

  if (pResult->vNdata == 0) {
    pResult->vLossPerDatapoint = pOptCtx->vLossPerDatapoint;
    pResult->vOptPeriod = pOptCtx->vOptPeriod;
  } else {
    GGML_ASSERT(pResult->vLossPerDatapoint == pOptCtx->vLossPerDatapoint);
    GGML_ASSERT(pResult->vOptPeriod == pOptCtx->vOptPeriod);
  }

  const int64_t cNdata = pOptCtx->vOutputs->ne[1];
  GGML_ASSERT(pResult->vNdata == cNdata * int64_t(pResult->vLoss.size()) && "varying batch size not supported");
  pResult->vNdata += cNdata;

  GGML_ASSERT(ggml_is_scalar(pOptCtx->vLoss));
  GGML_ASSERT(pOptCtx->vLoss->type == GGML_TYPE_F32);
  float vLoss;
  ggml_backend_tensor_get(pOptCtx->vLoss, &vLoss, 0, ggml_nbytes(pOptCtx->vLoss));
  pResult->vLoss.push_back(vLoss);

  if (pOptCtx->vPred) {
    GGML_ASSERT(pOptCtx->vPred->type == GGML_TYPE_I32);
    std::vector<int32_t> vPred(cNdata);
    ggml_backend_tensor_get(pOptCtx->vPred, vPred.data(), 0, ggml_nbytes(pOptCtx->vPred));
    pResult->vPred.insert(pResult->vPred.end(), vPred.begin(), vPred.end());
  }

  if (!pOptCtx->vNcorrect || pResult->vNcorrect < 0) {
    pResult->vNcorrect = -1;
    return;
  }

  GGML_ASSERT(ggml_is_scalar(pOptCtx->vNcorrect));
  GGML_ASSERT(pOptCtx->vNcorrect->type == GGML_TYPE_I64);
  int64_t vNcorrect;
  ggml_backend_tensor_get(pOptCtx->vNcorrect, &vNcorrect, 0, ggml_nbytes(pOptCtx->vNcorrect));
  pResult->vNcorrect += vNcorrect;
}

// ====== High-Level Functions ======

void ggml_opt_epoch(ggml_opt_context_t pOptCtx, ggml_opt_dataset_t pDataset, ggml_opt_result_t pResultTrain,
                    ggml_opt_result_t pResultEval, int64_t pIdataSplit, ggml_opt_epoch_callback pCallbackTrain,
                    ggml_opt_epoch_callback pCallbackEval) {
  GGML_ASSERT(ggml_opt_static_graphs(pOptCtx) && "ggml_opt_epoch requires static graphs");
  struct ggml_tensor *vInputs = ggml_opt_inputs(pOptCtx);
  struct ggml_tensor *vLabels = ggml_opt_labels(pOptCtx);
  struct ggml_tensor *vData = ggml_opt_dataset_data(pDataset);
  GGML_ASSERT(vData->ne[0] == vInputs->ne[0]);

  const int64_t cNdata = vData->ne[1];
  const int64_t cNdataBatch = vInputs->ne[1];

  GGML_ASSERT(vData->ne[1] % vInputs->ne[1] == 0);
  const int64_t cNbatches = cNdata / cNdataBatch;

  pIdataSplit = pIdataSplit < 0 ? cNdata : pIdataSplit;
  GGML_ASSERT(pIdataSplit % cNdataBatch == 0);
  const int64_t cIbatchSplit = pIdataSplit / cNdataBatch;

  int64_t vIbatch = 0;
  int64_t vTLoopStart = ggml_time_us();
  for (; vIbatch < cIbatchSplit; ++vIbatch) {
    ggml_opt_alloc(pOptCtx, /*backward =*/true);
    ggml_opt_dataset_get_batch(pDataset, vInputs, vLabels, vIbatch);
    ggml_opt_eval(pOptCtx, pResultTrain);
    if (pCallbackTrain) {
      pCallbackTrain(true, pOptCtx, pDataset, pResultTrain, vIbatch + 1, cIbatchSplit, vTLoopStart);
    }
  }
  vTLoopStart = ggml_time_us();
  for (; vIbatch < cNbatches; ++vIbatch) {
    ggml_opt_alloc(pOptCtx, /*backward =*/false);
    ggml_opt_dataset_get_batch(pDataset, vInputs, vLabels, vIbatch);
    ggml_opt_eval(pOptCtx, pResultEval);
    if (pCallbackEval) {
      pCallbackEval(false, pOptCtx, pDataset, pResultEval, vIbatch + 1 - cIbatchSplit, cNbatches - cIbatchSplit,
                    vTLoopStart);
    }
  }
}

void ggml_opt_epoch_callback_progress_bar(bool pTrain, ggml_opt_context_t pOptCtx, ggml_opt_dataset_t pDataset,
                                          ggml_opt_result_t pResult, int64_t pIbatch, int64_t pIbatchMax,
                                          int64_t pTStartUs) {
  fprintf(stderr, "%s[", pTrain ? "train: " : "val:   ");

  // The progress bar consists of partially filled blocks, unicode has 8 separate fill levels.
  constexpr int64_t cBarLength = 8;
  const int64_t cIbatch8 = 8 * pIbatch;
  for (int64_t vJ = 0; vJ < cBarLength; ++vJ) {
    if (pIbatchMax * (8 * vJ + 8) / cBarLength < cIbatch8) {
      fprintf(stderr, "\u2588"); // full block
    } else if (pIbatchMax * (8 * vJ + 7) / cBarLength < cIbatch8) {
      fprintf(stderr, "\u2589"); // 7/8 filled
    } else if (pIbatchMax * (8 * vJ + 6) / cBarLength < cIbatch8) {
      fprintf(stderr, "\u258A"); // 6/8 filled
    } else if (pIbatchMax * (8 * vJ + 5) / cBarLength < cIbatch8) {
      fprintf(stderr, "\u258B"); // 5/8 filled
    } else if (pIbatchMax * (8 * vJ + 4) / cBarLength < cIbatch8) {
      fprintf(stderr, "\u258C"); // 4/8 filled
    } else if (pIbatchMax * (8 * vJ + 3) / cBarLength < cIbatch8) {
      fprintf(stderr, "\u258D"); // 3/8 filled
    } else if (pIbatchMax * (8 * vJ + 2) / cBarLength < cIbatch8) {
      fprintf(stderr, "\u258E"); // 2/8 filled
    } else if (pIbatchMax * (8 * vJ + 1) / cBarLength < cIbatch8) {
      fprintf(stderr, "\u258F"); // 1/8 filled
    } else {
      fprintf(stderr, " ");
    }
  }

  const int64_t cBatchSize = ggml_opt_inputs(pOptCtx)->ne[1];
  const int64_t cIdata = pIbatch * cBatchSize;
  const int64_t cIdataMax = pIbatchMax * cBatchSize;

  double vLoss;
  double vLossUnc;
  ggml_opt_result_loss(pResult, &vLoss, &vLossUnc);

  double vAccuracy;
  double vAccuracyUnc;
  ggml_opt_result_accuracy(pResult, &vAccuracy, &vAccuracyUnc);

  const int64_t cTIbatchUs = ggml_time_us() - pTStartUs;
  int64_t vTIbatchS = cTIbatchUs / 1000000;
  const int64_t cTIbatchH = vTIbatchS / 3600;
  vTIbatchS -= cTIbatchH * 3600;
  const int64_t cTIbatchM = vTIbatchS / 60;
  vTIbatchS -= cTIbatchM * 60;

  const int64_t cTEtaUs = cTIbatchUs * (pIbatchMax - pIbatch) / pIbatch;
  int64_t vTEtaS = cTEtaUs / 1000000;
  const int64_t cTEtaH = vTEtaS / 3600;
  vTEtaS -= cTEtaH * 3600;
  const int64_t cTEtaM = vTEtaS / 60;
  vTEtaS -= cTEtaM * 60;

  fprintf(stderr,
          "] data=%07" PRId64 "/%07" PRId64 " loss=%.5lf±%.5lf acc=%.2lf±%.2lf%% "
          "t=%02" PRId64 ":%02" PRId64 ":%02" PRId64 " ETA=%02" PRId64 ":%02" PRId64 ":%02" PRId64 " \r",
          cIdata, cIdataMax, vLoss, vLossUnc, 100.0 * vAccuracy, 100.0 * vAccuracyUnc, cTIbatchH, cTIbatchM, vTIbatchS,
          cTEtaH, cTEtaM, vTEtaS);
  if (pIbatch == pIbatchMax) {
    fprintf(stderr, "\n");
  }
  fflush(stderr);

  GGML_UNUSED(pDataset);
}

void ggml_opt_fit(ggml_backend_sched_t pBackendSched, ggml_context *pCtxCompute, ggml_tensor *pInputs,
                  ggml_tensor *pOutputs, ggml_opt_dataset_t pDataset, enum ggml_opt_loss_type pLossType,
                  enum ggml_opt_optimizer_type pOptimizer, ggml_opt_get_optimizer_params pGetOptPars, int64_t pNepoch,
                  int64_t pNbatchLogical, float pValSplit, bool pSilent) {
  ggml_time_init();
  const int64_t cTStartUs = ggml_time_us();

  const int64_t cNdata = ggml_opt_dataset_data(pDataset)->ne[1];
  const int64_t cNbatchPhysical = pInputs->ne[1];
  GGML_ASSERT(cNdata % pNbatchLogical == 0);
  GGML_ASSERT(pNbatchLogical % cNbatchPhysical == 0);

  const int64_t cOptPeriod = pNbatchLogical / cNbatchPhysical;
  const int64_t cNbatchesLogical = cNdata / pNbatchLogical;

  GGML_ASSERT(pValSplit >= 0.0f);
  GGML_ASSERT(pValSplit < 1.0f);
  const int64_t cIbatchSplit =
      int64_t(((1.0f - pValSplit) * cNbatchesLogical)) * cOptPeriod; // train <-> val split index (physical)
  const int64_t cIdataSplit = cIbatchSplit * cNbatchPhysical;

  int64_t vEpoch = 1;

  ggml_opt_params vParams = ggml_opt_default_params(pBackendSched, pLossType);
  vParams.ctx_compute = pCtxCompute;
  vParams.inputs = pInputs;
  vParams.outputs = pOutputs;
  vParams.opt_period = cOptPeriod;
  vParams.get_opt_pars = pGetOptPars;
  vParams.get_opt_pars_ud = &vEpoch;
  vParams.optimizer = pOptimizer;
  ggml_opt_context_t vOptCtx = ggml_opt_init(vParams);

  // Shuffling the data is generally useful but there is only a point if not all data is used in a single batch.
  if (pNbatchLogical < cNdata) {
    ggml_opt_dataset_shuffle(vOptCtx, pDataset, -1); // Shuffle all data (train + validation).
  }

  ggml_opt_result_t vResultTrain = ggml_opt_result_init();
  ggml_opt_result_t vResultVal = ggml_opt_result_init();

  ggml_opt_epoch_callback vEpochCallback = pSilent ? nullptr : ggml_opt_epoch_callback_progress_bar;

  for (; vEpoch <= pNepoch; ++vEpoch) {
    if (pNbatchLogical < cIdataSplit) {
      ggml_opt_dataset_shuffle(vOptCtx, pDataset, cIdataSplit);
    }

    ggml_opt_result_reset(vResultTrain);
    ggml_opt_result_reset(vResultVal);

    if (!pSilent) {
      fprintf(stderr, "%s: epoch %04" PRId64 "/%04" PRId64 ":\n", __func__, vEpoch, pNepoch);
    }
    ggml_opt_epoch(vOptCtx, pDataset, vResultTrain, vResultVal, cIdataSplit, vEpochCallback, vEpochCallback);
    if (!pSilent) {
      fprintf(stderr, "\n");
    }
  }

  if (!pSilent) {
    int64_t vTTotalS = (ggml_time_us() - cTStartUs) / 1000000;
    const int64_t cTTotalH = vTTotalS / 3600;
    vTTotalS -= cTTotalH * 3600;
    const int64_t cTTotalM = vTTotalS / 60;
    vTTotalS -= cTTotalM * 60;
    fprintf(stderr, "%s: training took %02" PRId64 ":%02" PRId64 ":%02" PRId64 "\n", __func__, cTTotalH, cTTotalM,
            vTTotalS);
  }

  ggml_opt_free(vOptCtx);
  ggml_opt_result_free(vResultTrain);
  ggml_opt_result_free(vResultVal);
}

enum ggml_opt_optimizer_type ggml_opt_context_optimizer_type(ggml_opt_context_t pC) { return pC->vOptimizer; }

GGML_API const char *ggml_opt_optimizer_name(enum ggml_opt_optimizer_type pO) {
  switch (pO) {
  case GGML_OPT_OPTIMIZER_TYPE_ADAMW:
    return "adamw";
  case GGML_OPT_OPTIMIZER_TYPE_SGD:
    return "sgd";
  default:
    return "undefined";
  };
}
