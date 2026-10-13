#include "mgs-sampler.h"

#include "mgs-impl.h"
#include "mgs-vocab.h"
#include "mgs-grammar.h"

#include "ggml-cpp.h"

#include <array>
#include <algorithm>
#include <cassert>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <numeric>
#include <random>
#include <unordered_map>
#include <stdexcept>

// the ring buffer works similarly to std::deque, but with a fixed capacity
template <typename T> struct ring_buffer {
  ring_buffer(size_t pCap) : vCapacity(pCap), lData(pCap) {}

  T &mFront() {
    if (vSz == 0) {
      throw std::runtime_error("ring buffer is empty");
    }
    return lData[vFirst];
  }

  const T &mFront() const {
    if (vSz == 0) {
      throw std::runtime_error("ring buffer is empty");
    }
    return lData[vFirst];
  }

  T &mBack() {
    if (vSz == 0) {
      throw std::runtime_error("ring buffer is empty");
    }
    return lData[vPos];
  }

  const T &mBack() const {
    if (vSz == 0) {
      throw std::runtime_error("ring buffer is empty");
    }
    return lData[vPos];
  }

  void mPushBack(const T &pValue) {
    if (vCapacity == 0) {
      throw std::runtime_error("ring buffer: capacity is zero");
    }

    if (vSz == vCapacity) {
      // advance the start when buffer is full
      vFirst = (vFirst + 1) % vCapacity;
    } else {
      vSz++;
    }
    lData[vPos] = pValue;
    vPos = (vPos + 1) % vCapacity;
  }

  T mPopFront() {
    if (vSz == 0) {
      throw std::runtime_error("ring buffer is empty");
    }
    T vValue = lData[vFirst];
    vFirst = (vFirst + 1) % vCapacity;
    vSz--;
    return vValue;
  }

  // T & operator[](size_t pI) {
  //     if (pI >= vSz) {
  //         throw std::runtime_error("ring buffer: index out of bounds");
  //     }
  //     return lData[(vFirst + pI) % vCapacity];
  // }

  // const T & mAt(size_t pI) const {
  //     if (pI >= vSz) {
  //         throw std::runtime_error("ring buffer: index out of bounds");
  //     }
  //     return lData[(vFirst + pI) % vCapacity];
  // }

  const T &mRat(size_t pI) const {
    if (pI >= vSz) {
      throw std::runtime_error("ring buffer: index out of bounds");
    }
    return lData[(vFirst + vSz - pI - 1) % vCapacity];
  }

  std::vector<T> mToVector() const {
    std::vector<T> lResult;
    lResult.reserve(vSz);
    for (size_t vI = 0; vI < vSz; vI++) {
      lResult.push_back(lData[(vFirst + vI) % vCapacity]);
    }
    return lResult;
  }

  void mClear() {
    // here only reset the status of the buffer
    vSz = 0;
    vFirst = 0;
    vPos = 0;
  }

  bool empty() const { return vSz == 0; }

  size_t size() const { return vSz; }

  size_t vCapacity = 0;
  size_t vSz = 0;
  size_t vFirst = 0;
  size_t vPos = 0;

  std::vector<T> lData;
};

// writes result in res, does not mutate cur
static void fMgsTokenDataArrayPartialSort(const mgs_token_data_array &pCur, int pNpartial,
                                          std::vector<mgs_token_data> &pRes) {
  static const auto cComp = [](const mgs_token_data &pA, const mgs_token_data &pB) { return pA.logit > pB.logit; };

  constexpr int cNbuckets = 128;
  constexpr float cBucketLow = -10.0f;
  constexpr float cBucketHigh = 10.0f;
  constexpr float cBucketScale = cNbuckets / (cBucketHigh - cBucketLow);
  constexpr float cBucketInter = -cBucketLow * cBucketScale;

  std::vector<int> lBucketIdx;
  std::vector<int> lHisto(cNbuckets, 0);

  std::vector<mgs_token_data *> lBucketPtrs;

  lBucketIdx.reserve(pCur.size);

  for (int vI = 0; vI < (int)pCur.size; ++vI) {
    const float cVal = pCur.data[vI].logit;
    int vIb = int(cBucketScale * cVal + cBucketInter); // cNbuckets * (cVal - cBucketLow) / (cBucketHigh - cBucketLow);
    vIb = std::max(0, std::min(cNbuckets - 1, vIb));
    lBucketIdx.push_back(vIb);
    ++lHisto[vIb];
  }
  int vNhave = 0;
  int vIb = cNbuckets - 1;
  for (; vIb >= 0; --vIb) {
    vNhave += lHisto[vIb];
    if (vNhave >= pNpartial) {
      break;
    }
  }
  pRes.resize(vNhave);
  auto *vPtr = pRes.data();
  lBucketPtrs.reserve(cNbuckets - vIb);
  for (int vJ = cNbuckets - 1; vJ >= vIb; --vJ) {
    lBucketPtrs.push_back(vPtr);
    vPtr += lHisto[vJ];
  }
  for (int vI = 0; vI < (int)pCur.size; ++vI) {
    int vJ = lBucketIdx[vI];
    if (vJ >= vIb) {
      *lBucketPtrs[cNbuckets - 1 - vJ]++ = pCur.data[vI];
    }
  }

  vPtr = pRes.data();
  int vNdone = 0;
  for (int vJ = cNbuckets - 1; vJ > vIb; --vJ) {
    std::sort(vPtr, vPtr + lHisto[vJ], cComp);
    vPtr += lHisto[vJ];
    vNdone += lHisto[vJ];
  }
  std::partial_sort(vPtr, vPtr + pNpartial - vNdone, vPtr + lHisto[vIb], cComp);
}

// reduces the size of cur_p to npartial, keeping only the top npartial elements
static void fMgsTokenDataArrayPartialSortInplace(mgs_token_data_array *pCurP, int pNpartial) {
  static const auto cComp = [](const mgs_token_data &pA, const mgs_token_data &pB) { return pA.logit > pB.logit; };

  if (pNpartial <= 128) {
    std::partial_sort(pCurP->data, pCurP->data + pNpartial, pCurP->data + pCurP->size, cComp);

    pCurP->size = pNpartial;
    pCurP->sorted = true;

    return;
  }

  std::vector<mgs_token_data> lTmp;

  fMgsTokenDataArrayPartialSort(*pCurP, pNpartial, lTmp);

  std::copy(lTmp.data(), lTmp.data() + pNpartial, pCurP->data);

  pCurP->size = pNpartial;
  pCurP->sorted = true;
}

static int fMgsSampleDist(mgs_token_data_array *pCurP, std::mt19937 &pRng) {
  // iterator for the probabilities
#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-local-typedefs"
#endif

  struct probs_iterator {
    typedef std::input_iterator_tag iterator_category;
    typedef float value_type;
    typedef float *pointer;
    typedef float &reference;
    typedef ptrdiff_t difference_type;

    const mgs_token_data *vData;

    bool operator==(const probs_iterator &pOther) const { return vData == pOther.vData; }
    bool operator!=(const probs_iterator &pOther) const { return vData != pOther.vData; }
    const float &operator*() const { return vData->p; }
    probs_iterator &operator++() {
      ++vData;
      return *this;
    }
    probs_iterator operator++(int) {
      probs_iterator vTmp = *this;
      ++vData;
      return vTmp;
    }
  };

#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif

  std::discrete_distribution<int> vDist(probs_iterator{pCurP->data}, probs_iterator{pCurP->data + pCurP->size});

  return vDist(pRng);
}

/*
static void fMgsLogSoftmax(float * pArray, size_t pSize) {
    float vMaxL = *std::max_element(pArray, pArray + pSize);
    float vSum = 0.f;
    for (size_t vI = 0; vI < pSize; ++vI) {
        float vP = expf(pArray[vI] - vMaxL);
        vSum += vP;
        pArray[vI] = vP;
    }

    for (size_t vI = 0; vI < pSize; ++vI) {
        pArray[vI] = logf(pArray[vI] / vSum);
    }
}
*/

static void fMgsSamplerTempImpl(mgs_token_data_array *pCurP, float pTemp) {
  if (pCurP->size == 0) {
    return;
  }

  if (pTemp <= 0.0f) {
    // find the token with the highest logit and set the rest to -inf
    size_t vMaxI = 0;
    float vMaxL = pCurP->data[0].logit;

    for (size_t vI = 1; vI < pCurP->size; ++vI) {
      if (pCurP->data[vI].logit > vMaxL) {
        pCurP->data[vMaxI].logit = -INFINITY;
        vMaxI = vI;
        vMaxL = pCurP->data[vI].logit;
      } else {
        pCurP->data[vI].logit = -INFINITY;
      }
    }

    return;
  }

  for (size_t vI = 0; vI < pCurP->size; ++vI) {
    pCurP->data[vI].logit /= pTemp;
  }
}

static void fMgsSamplerSoftmaxImpl(mgs_token_data_array *pCurP, bool pDoSort) {
  GGML_ASSERT(pCurP->size > 0);

  // Sort the logits in descending order if requested
  if (pDoSort && !pCurP->sorted) {
    fMgsTokenDataArrayPartialSortInplace(pCurP, pCurP->size);
  }

  float vMaxL = pCurP->data[0].logit;
  if (!pCurP->sorted) {
    for (size_t vI = 1; vI < pCurP->size; ++vI) {
      vMaxL = std::max(vMaxL, pCurP->data[vI].logit);
    }
  }

  float vCumSum = 0.0f;

  for (size_t vI = 0; vI < pCurP->size; ++vI) {
    float vP = expf(pCurP->data[vI].logit - vMaxL);
    pCurP->data[vI].p = vP;
    vCumSum += vP;
  }

  for (size_t vI = 0; vI < pCurP->size; ++vI) {
    pCurP->data[vI].p /= vCumSum;
  }
}

static void fMgsSamplerTopKImpl(mgs_token_data_array *pCurP, int32_t pK) {
  // if (pK >= (int32_t)pCurP->size) {
  //     return;
  // }

  if (pK <= 0) {
    return;
  }

  pK = std::min(pK, (int)pCurP->size);

  // Sort scores in descending order
  if (!pCurP->sorted) {
    fMgsTokenDataArrayPartialSortInplace(pCurP, pK);
  }

  pCurP->size = pK;
}

static uint32_t fGetRngSeed(uint32_t pSeed) {
  if (pSeed == MGS_DEFAULT_SEED) {
    // use system clock if std::random_device is not a true RNG
    static bool vIsRdPrng = std::random_device().entropy() == 0;
    if (vIsRdPrng) {
      return (uint32_t)std::chrono::system_clock::now().time_since_epoch().count();
    }
    std::random_device vRd;
    return vRd();
  }
  return pSeed;
}

// mgs_sampler API

struct mgs_sampler *mgs_sampler_init(struct mgs_sampler_i *pIface, mgs_sampler_context_t pCtx) {
  return new mgs_sampler{
      /* .iface = */ pIface,
      /* .ctx   = */ pCtx,
  };
}

const char *mgs_sampler_name(const struct mgs_sampler *pSmpl) {
  if (!pSmpl->iface) {
    return "(null)";
  }

  return pSmpl->iface->name(pSmpl);
}

void mgs_sampler_accept(struct mgs_sampler *pSmpl, mgs_token pToken) {
  if (!pSmpl) {
    return;
  }

  if (pSmpl->iface->accept) {
    pSmpl->iface->accept(pSmpl, pToken);
  }
}

void mgs_sampler_apply(struct mgs_sampler *pSmpl, struct mgs_token_data_array *pCurP) {
  if (!pSmpl) {
    return;
  }

  GGML_ASSERT(pSmpl->iface->apply);
  pSmpl->iface->apply(pSmpl, pCurP);
}

void mgs_sampler_reset(struct mgs_sampler *pSmpl) {
  if (!pSmpl) {
    return;
  }

  if (pSmpl->iface->reset) {
    pSmpl->iface->reset(pSmpl);
  }
}

struct mgs_sampler *mgs_sampler_clone(const struct mgs_sampler *pSmpl) {
  if (!pSmpl) {
    return nullptr;
  }

  if (pSmpl->iface->clone) {
    return pSmpl->iface->clone(pSmpl);
  }

  if (pSmpl->ctx == nullptr) {
    return mgs_sampler_init(
        /* .iface = */ pSmpl->iface,
        /* .ctx   = */ nullptr);
  }

  GGML_ABORT("the sampler does not support cloning");
}

void mgs_sampler_free(struct mgs_sampler *pSmpl) {
  if (pSmpl == nullptr) {
    return;
  }

  if (pSmpl->iface->free) {
    pSmpl->iface->free(pSmpl);
  }

  delete pSmpl;
}

// empty sampler

struct mgs_sampler_empty {
  const char *vName;
};

static struct mgs_sampler *fMgsSamplerInitEmpty(const char *pName);

static const char *fMgsSamplerEmptyName(const struct mgs_sampler *pSmpl) {
  auto *vCtx = (mgs_sampler_empty *)pSmpl->ctx;
  return vCtx->vName;
}

static void fMgsSamplerEmptyAccept(struct mgs_sampler *pSmpl, mgs_token pToken) {
  GGML_UNUSED(pSmpl);
  GGML_UNUSED(pToken);
}

static void fMgsSamplerEmptyApply(struct mgs_sampler *pSmpl, mgs_token_data_array *pCurP) {
  GGML_UNUSED(pSmpl);
  GGML_UNUSED(pCurP);
}

static void fMgsSamplerEmptyReset(struct mgs_sampler *pSmpl) { GGML_UNUSED(pSmpl); }

static struct mgs_sampler *fMgsSamplerEmptyClone(const struct mgs_sampler *pSmpl) {
  auto *vCtx = (mgs_sampler_empty *)pSmpl->ctx;
  return fMgsSamplerInitEmpty(vCtx->vName);
}

static void fMgsSamplerEmptyFree(struct mgs_sampler *pSmpl) { delete (mgs_sampler_empty *)pSmpl->ctx; }

static bool fMgsSamplerEmptyBackendInit(struct mgs_sampler *pSmpl, ggml_backend_buffer_type_t pBuft) {
  GGML_UNUSED(pSmpl);
  GGML_UNUSED(pBuft);

  return true;
}

static void fMgsSamplerEmptyBackendAccept(struct mgs_sampler *pSmpl, ggml_context *pCtx, ggml_cgraph *pGf,
                                          struct ggml_tensor *pSelectedToken) {
  GGML_UNUSED(pSmpl);
  GGML_UNUSED(pCtx);
  GGML_UNUSED(pGf);
  GGML_UNUSED(pSelectedToken);
}

static void fMgsSamplerEmptyBackendApply(struct mgs_sampler *pSmpl, struct ggml_context *pCtx, struct ggml_cgraph *pGf,
                                         struct mgs_sampler_data *pData) {
  GGML_UNUSED(pSmpl);
  GGML_UNUSED(pCtx);
  GGML_UNUSED(pGf);
  GGML_UNUSED(pData);
}

static void fMgsSamplerEmptyBackendSetInput(struct mgs_sampler *pSmpl) { GGML_UNUSED(pSmpl); }

static struct mgs_sampler_i vMgsSamplerEmptyI = {
    /* .name              = */ fMgsSamplerEmptyName,
    /* .accept            = */ fMgsSamplerEmptyAccept,
    /* .apply             = */ fMgsSamplerEmptyApply,
    /* .reset             = */ fMgsSamplerEmptyReset,
    /* .clone             = */ fMgsSamplerEmptyClone,
    /* .free              = */ fMgsSamplerEmptyFree,
    /* .backend_init      = */ fMgsSamplerEmptyBackendInit,
    /* .backend_accept    = */ fMgsSamplerEmptyBackendAccept,
    /* .backend_apply     = */ fMgsSamplerEmptyBackendApply,
    /* .backend_set_input = */ fMgsSamplerEmptyBackendSetInput,
};

struct mgs_sampler *fMgsSamplerInitEmpty(const char *pName) {
  return mgs_sampler_init(
      /* .iface = */ &vMgsSamplerEmptyI,
      /* .ctx   = */ new mgs_sampler_empty{
          /* .vName = */ pName,
      });
}

// common backend sampler functionality
//
// +name : means that the sampler is support and will run on the backend
// -name : means that a ggml operator is not supported by the backend
//
struct mgs_sampler_backend {
  mgs_sampler_backend(const char *pName) : vName(pName), vNameExt(pName), vIsInit(false), vSupport(false) {}

  const char *mGetName() {
    if (!vIsInit) {
      return vName.c_str();
    }

    if (vSupport) {
      vNameExt = "+" + vName;
    } else {
      vNameExt = "-" + vName;
    }

    return vNameExt.c_str();
  }

  void mInit(bool pSupport) {
    GGML_ASSERT(this->vIsInit == false);

    this->vIsInit = true;
    this->vSupport = pSupport;
  }

private:
  std::string vName;
  std::string vNameExt;

  bool vIsInit;
  bool vSupport;
};

// check if all ggml ops used by the sampler are supported by the backend
static bool fMgsSamplerBackendSupport(mgs_sampler *pSmpl, ggml_backend_buffer_type_t pBuft) {
  auto *vDevice = ggml_backend_buft_get_device(pBuft);
  if (!vDevice) {
    // CPU backend always supported
    return true;
  }

  ggml_init_params vParams = {
      /*.mem_size   =*/128 * ggml_tensor_overhead() + ggml_graph_overhead(),
      /*.mem_buffer =*/NULL,
      /*.no_alloc   =*/true,
  };

  ggml_context_ptr vCtxPtr{ggml_init(vParams)};
  if (!vCtxPtr) {
    throw std::runtime_error(fFormat("failed to create ggml context"));
  }

  ggml_context *vCtx = vCtxPtr.get();

  const int64_t cN = 1024 * 1024;

  mgs_sampler_data vData = {
      /*.logits     = */ ggml_new_tensor_1d(vCtx, GGML_TYPE_F32, cN),
      /*.probs      = */ nullptr,
      /*.sampled    = */ nullptr,
      /*.candidates = */ ggml_new_tensor_1d(vCtx, GGML_TYPE_I32, cN),
  };

  ggml_cgraph *vGf = ggml_new_graph(vCtx);

  pSmpl->iface->backend_apply(pSmpl, vCtx, vGf, &vData);

  if (vData.logits) {
    ggml_build_forward_expand(vGf, vData.logits);
  }

  if (vData.probs) {
    ggml_build_forward_expand(vGf, vData.probs);
  }

  if (vData.sampled) {
    ggml_build_forward_expand(vGf, vData.sampled);
  }

  if (vData.candidates) {
    ggml_build_forward_expand(vGf, vData.candidates);
  }

  for (int vI = 0; vI < ggml_graph_n_nodes(vGf); vI++) {
    struct ggml_tensor *vOp = ggml_graph_node(vGf, vI);

    if (!ggml_backend_dev_supports_op(vDevice, vOp)) {
      MGS_LOG_WARN("%s: device '%s' does not have support for op %s needed for sampler '%s'\n", __func__,
                   ggml_backend_dev_name(vDevice), ggml_op_name(vOp->op), pSmpl->iface->name(pSmpl));

      return false;
    }
  }

  return true;
}

// sampler chain

static const char *fMgsSamplerChainName(const struct mgs_sampler * /*smpl*/) { return "chain"; }

static void fMgsSamplerChainAccept(struct mgs_sampler *pSmpl, mgs_token pToken) {
  auto *vChain = (mgs_sampler_chain *)pSmpl->ctx;

  time_meas vTm(vChain->vTSampleUs, vChain->vParams.no_perf);

  for (auto &vSmpl : vChain->lSamplers) {
    mgs_sampler_accept(vSmpl.vPtr, pToken);
  }

  vChain->vNSample++;
}

static void fMgsSamplerChainApply(struct mgs_sampler *pSmpl, mgs_token_data_array *pCurP) {
  auto *vChain = (mgs_sampler_chain *)pSmpl->ctx;

  time_meas vTm(vChain->vTSampleUs, vChain->vParams.no_perf);

  bool vIsBackend = vChain->vIsInit;

  for (auto &vSmpl : vChain->lSamplers) {
    if (vIsBackend && vSmpl.vIsBackend) {
      continue;
    }

    vIsBackend = false;

    if (vSmpl.vPtr->iface->apply == nullptr) {
      continue;
    }

    mgs_sampler_apply(vSmpl.vPtr, pCurP);
  }
}

static void fMgsSamplerChainReset(struct mgs_sampler *pSmpl) {
  auto *vChain = (mgs_sampler_chain *)pSmpl->ctx;

  for (auto &vSmpl : vChain->lSamplers) {
    mgs_sampler_reset(vSmpl.vPtr);
  }
}

static struct mgs_sampler *fMgsSamplerChainClone(const struct mgs_sampler *pSmpl) {
  const auto *vChainSrc = (const mgs_sampler_chain *)pSmpl->ctx;

  auto *vResult = mgs_sampler_chain_init(vChainSrc->vParams);

  for (const auto &vSmpl : vChainSrc->lSamplers) {
    mgs_sampler_chain_add(vResult, mgs_sampler_clone(vSmpl.vPtr));
  }

  return vResult;
}

static void fMgsSamplerChainFree(struct mgs_sampler *pSmpl) {
  auto *vChain = (mgs_sampler_chain *)pSmpl->ctx;

  for (auto &vSmpl : vChain->lSamplers) {
    mgs_sampler_free(vSmpl.vPtr);
  }

  delete vChain;
}

static bool fMgsSamplerChainBackendInit(struct mgs_sampler *pSmpl, ggml_backend_buffer_type_t pBuft) {
  auto *vChain = (mgs_sampler_chain *)pSmpl->ctx;

  GGML_ASSERT(vChain->vIsInit == false && "fMgsSamplerChainBackendInit() called twice");

  vChain->vIsInit = true;

  bool vRes = true;

  for (auto &vSmpl : vChain->lSamplers) {
    bool vResCur = true;

    // to be able to run a sampler on the backend, it has to:
    // - have the .backend_init() API implemented
    // - return true during .backend_init()
    if (vSmpl.vPtr->iface->backend_init) {
      if (!vSmpl.vPtr->iface->backend_init(vSmpl.vPtr, pBuft)) {
        vResCur = false;
      }
    } else {
      vResCur = false;
    }

    vSmpl.vIsBackend = vResCur;

    vRes = vRes && vResCur;
  }

  return vRes;
}

static void fMgsSamplerChainBackendAccept(struct mgs_sampler *pSmpl, ggml_context *pCtx, ggml_cgraph *pGf,
                                          struct ggml_tensor *pSelectedToken) {
  auto *vChain = (mgs_sampler_chain *)pSmpl->ctx;

  for (auto &vSmpl : vChain->lSamplers) {
    if (!vSmpl.vIsBackend) {
      break;
    }

    if (vSmpl.vPtr->iface->backend_accept) {
      vSmpl.vPtr->iface->backend_accept(vSmpl.vPtr, pCtx, pGf, pSelectedToken);
    }
  }
}

static void fMgsSamplerChainBackendApply(struct mgs_sampler *pSmpl, struct ggml_context *pCtx, struct ggml_cgraph *pGf,
                                         struct mgs_sampler_data *pData) {
  auto *vChain = (mgs_sampler_chain *)pSmpl->ctx;

  GGML_ASSERT(vChain->vIsInit && "fMgsSamplerChainBackendInit() not called");

  for (auto &vSmpl : vChain->lSamplers) {
    if (!vSmpl.vIsBackend) {
      break;
    }

    if (vSmpl.vPtr->iface->backend_apply) {
      vSmpl.vPtr->iface->backend_apply(vSmpl.vPtr, pCtx, pGf, pData);
    }
  }
}

static void fMgsSamplerChainBackendSetInput(struct mgs_sampler *pSmpl) {
  auto *vChain = (mgs_sampler_chain *)pSmpl->ctx;

  for (auto &vSmpl : vChain->lSamplers) {
    if (!vSmpl.vIsBackend) {
      break;
    }

    if (vSmpl.vPtr->iface->backend_set_input) {
      vSmpl.vPtr->iface->backend_set_input(vSmpl.vPtr);
    }
  }
}

static struct mgs_sampler_i vMgsSamplerChainI = {
    /* .name              = */ fMgsSamplerChainName,
    /* .accept            = */ fMgsSamplerChainAccept,
    /* .apply             = */ fMgsSamplerChainApply,
    /* .reset             = */ fMgsSamplerChainReset,
    /* .clone             = */ fMgsSamplerChainClone,
    /* .free              = */ fMgsSamplerChainFree,
    /* .backend_init      = */ fMgsSamplerChainBackendInit,
    /* .backend_accept    = */ fMgsSamplerChainBackendAccept,
    /* .backend_apply     = */ fMgsSamplerChainBackendApply,
    /* .backend_set_input = */ fMgsSamplerChainBackendSetInput,
};

struct mgs_sampler *mgs_sampler_chain_init(struct mgs_sampler_chain_params pParams) {
  return mgs_sampler_init(
      /* .iface = */ &vMgsSamplerChainI,
      /* .ctx   = */ new mgs_sampler_chain{
          /* .vParams    = */ pParams,
          /* .vIsInit    = */ false,
          /* .lSamplers  = */ {},
          /* .lCur       = */ {},
          /* .vTSampleUs = */ 0,
          /* .vNSample   = */ 0,
      });
}

mgs_token mgs_sampler_sample(struct mgs_sampler *pSmpl, struct mgs_context *pCtx, int32_t pIdx) {
  const mgs_token cSampledToken = mgs_get_sampled_token_ith(pCtx, pIdx);
  const float *vSampledProbs = mgs_get_sampled_probs_ith(pCtx, pIdx);
  const float *vSampledLogits = mgs_get_sampled_logits_ith(pCtx, pIdx);
  const mgs_token *vSampledIds = mgs_get_sampled_candidates_ith(pCtx, pIdx);

  // If a backend sampler has already sampled a token, return it.
  if (cSampledToken != MGS_TOKEN_NULL) {
    MGS_LOG_DEBUG("%s: Backend sampler selected token for idx %d. Skipping CPU samplers\n", __func__, pIdx);
    return cSampledToken;
  }

  const mgs_model *vModel = mgs_get_model(pCtx);
  const mgs_vocab *vVocab = mgs_model_get_vocab(vModel);

  const int cNVocab = mgs_vocab_n_tokens(vVocab);

  // use pre-allocated buffer from chain if available, otherwise allocate locally
  std::vector<mgs_token_data> *lCurPtr;
  std::vector<mgs_token_data> lCurLocal;

  if (pSmpl->iface == &vMgsSamplerChainI) {
    auto *vChain = (mgs_sampler_chain *)pSmpl->ctx;
    lCurPtr = &vChain->lCur;
  } else {
    lCurPtr = &lCurLocal;
  }

  auto &vCur = *lCurPtr;

  if (vSampledProbs) {
    const uint32_t cSampledProbsCount = mgs_get_sampled_probs_count_ith(pCtx, pIdx);
    vCur.resize(cSampledProbsCount);
    for (uint32_t vI = 0; vI < cSampledProbsCount; ++vI) {
      vCur[vI] = mgs_token_data{vSampledIds[vI], vSampledLogits[vI], vSampledProbs[vI]};
    }
  } else if (vSampledLogits) {
    const uint32_t cSampledLogitsCount = mgs_get_sampled_logits_count_ith(pCtx, pIdx);
    vCur.resize(cSampledLogitsCount);
    for (mgs_token vI = 0; vI < (int)cSampledLogitsCount; vI++) {
      vCur[vI] = mgs_token_data{vSampledIds[vI], vSampledLogits[vI], 0.0f};
    }
  } else {
    const auto *vLogits = mgs_get_logits_ith(pCtx, pIdx);
    GGML_ASSERT(vLogits != nullptr);
    vCur.resize(cNVocab);
    for (mgs_token vTokenId = 0; vTokenId < cNVocab; vTokenId++) {
      vCur[vTokenId] = mgs_token_data{vTokenId, vLogits[vTokenId], 0.0f};
    }
  }

  mgs_token_data_array vCurP = {
      /* .data       = */ vCur.data(),
      /* .size       = */ vCur.size(),
      /* .selected   = */ -1,
      /* .sorted     = */ false,
  };

  mgs_sampler_apply(pSmpl, &vCurP);

  GGML_ASSERT(vCurP.selected >= 0 && vCurP.selected < (int32_t)vCurP.size);

  auto vToken = vCurP.data[vCurP.selected].id;

  mgs_sampler_accept(pSmpl, vToken);

  return vToken;
}

void mgs_sampler_chain_add(struct mgs_sampler *pChain, struct mgs_sampler *pSmpl) {
  auto *vP = (mgs_sampler_chain *)pChain->ctx;
  vP->lSamplers.push_back({
      /* .vIsBackend = */ false,
      /* .vPtr       = */ pSmpl,
  });
}

struct mgs_sampler *mgs_sampler_chain_get(struct mgs_sampler *pChain, int32_t pI) {
  if (pChain == nullptr) {
    return nullptr;
  }

  if (pChain->iface != &vMgsSamplerChainI) {
    return nullptr;
  }

  if (pI == -1) {
    return pChain;
  }

  const auto *vP = (const mgs_sampler_chain *)pChain->ctx;

  if (pI < 0 || (size_t)pI >= vP->lSamplers.size()) {
    return nullptr;
  }

  return vP->lSamplers[pI].vPtr;
}

struct mgs_sampler *mgs_sampler_chain_remove(struct mgs_sampler *pChain, int32_t pI) {
  auto *vP = (mgs_sampler_chain *)pChain->ctx;

  if (pI < 0 || (size_t)pI >= vP->lSamplers.size()) {
    return nullptr;
  }

  auto *vResult = vP->lSamplers[pI].vPtr;
  vP->lSamplers.erase(vP->lSamplers.begin() + pI);

  return vResult;
}

int mgs_sampler_chain_n(const struct mgs_sampler *pChain) {
  const auto *vP = (const mgs_sampler_chain *)pChain->ctx;

  return vP->lSamplers.size();
}

//
// samplers
//

// greedy

struct mgs_sampler_greedy : public mgs_sampler_backend {};

static const char *fMgsSamplerGreedyName(const struct mgs_sampler *pSmpl) {
  auto *vSctx = (mgs_sampler_greedy *)pSmpl->ctx;
  return vSctx->mGetName();
}

static void fMgsSamplerGreedyReset(struct mgs_sampler *pSmpl) {
  auto *vCtx = (mgs_sampler_greedy *)pSmpl->ctx;
  GGML_UNUSED(vCtx);
}

static struct mgs_sampler *fMgsSamplerGreedyClone(const struct mgs_sampler *pSmpl) {
  const auto *vCtx = (const mgs_sampler_greedy *)pSmpl->ctx;
  auto *vResult = mgs_sampler_init_greedy();

  // copy the state
  {
    auto *vResultCtx = (mgs_sampler_greedy *)vResult->ctx;

    GGML_UNUSED(vCtx);
    GGML_UNUSED(vResultCtx);
  }

  return vResult;
}

static void fMgsSamplerGreedyFree(struct mgs_sampler *pSmpl) { delete (mgs_sampler_greedy *)pSmpl->ctx; }

static void fMgsSamplerGreedyApply(struct mgs_sampler * /*smpl*/, mgs_token_data_array *pCurP) {
  pCurP->selected = 0;
  for (size_t vI = 1; vI < pCurP->size; ++vI) {
    if (pCurP->data[vI].logit > pCurP->data[pCurP->selected].logit) {
      pCurP->selected = vI;
    }
  }
}

static bool fMgsSamplerGreedyBackendInit(struct mgs_sampler *pSmpl, ggml_backend_buffer_type_t pBuft) {
  auto *vSctx = (mgs_sampler_greedy *)pSmpl->ctx;

  const bool cRes = fMgsSamplerBackendSupport(pSmpl, pBuft);

  vSctx->mInit(cRes);

  return cRes;
}

static void fMgsSamplerGreedyBackendApply(struct mgs_sampler *pSmpl, struct ggml_context *pCtx, struct ggml_cgraph *pGf,
                                          struct mgs_sampler_data *pData) {
  GGML_UNUSED(pGf);
  GGML_UNUSED(pSmpl);

  struct ggml_tensor *vLogits = ggml_reshape_1d(pCtx, pData->logits, ggml_nelements(pData->logits));

  struct ggml_tensor *vCurl = ggml_argmax(pCtx, vLogits);
  ggml_set_name(vCurl, "greedy_argmax");

  pData->sampled = vCurl;
}

static struct mgs_sampler_i vMgsSamplerGreedyI = {
    /* .name              = */ fMgsSamplerGreedyName,
    /* .accept            = */ nullptr,
    /* .apply             = */ fMgsSamplerGreedyApply,
    /* .reset             = */ fMgsSamplerGreedyReset,
    /* .clone             = */ fMgsSamplerGreedyClone,
    /* .free              = */ fMgsSamplerGreedyFree,
    /* .backend_init      = */ fMgsSamplerGreedyBackendInit,
    /* .backend_accept    = */ nullptr,
    /* .backend_apply     = */ fMgsSamplerGreedyBackendApply,
    /* .backend_set_input = */ nullptr,
};

struct mgs_sampler *mgs_sampler_init_greedy() {
  return mgs_sampler_init(
      /* .iface = */ &vMgsSamplerGreedyI,
      /* .ctx   = */ new mgs_sampler_greedy{
          ("greedy"),
      });
}

// dist

struct mgs_sampler_dist : public mgs_sampler_backend {
  const uint32_t cSeed;
  uint32_t vSeedCur;

  std::mt19937 vRng;

  ggml_tensor *vInpUniform;
};

static const char *fMgsSamplerDistName(const struct mgs_sampler *pSmpl) {
  auto *vSctx = (mgs_sampler_dist *)pSmpl->ctx;
  return vSctx->mGetName();
}

static void fMgsSamplerDistApply(struct mgs_sampler *pSmpl, mgs_token_data_array *pCurP) {
  auto *vCtx = (mgs_sampler_dist *)pSmpl->ctx;

  // edge cases
  if (pCurP->size == 0) {
    pCurP->selected = -1;
    return;
  }

  pCurP->selected = 0;

  if (pCurP->size == 1) {
    pCurP->data[0].p = 1.0f;
    return;
  }

  // max logit for numerical stability
  float vMaxL = pCurP->data[0].logit;
  if (!pCurP->sorted) {
    for (size_t vI = 1; vI < pCurP->size; ++vI) {
      vMaxL = std::max(vMaxL, pCurP->data[vI].logit);
    }
  }

  // apply softmax to obtain the probabilities
  double vSumCum = 0.0f;
  for (size_t vI = 0; vI < pCurP->size; ++vI) {
    float vP = expf(pCurP->data[vI].logit - vMaxL);
    pCurP->data[vI].p = vP;
    vSumCum += vP;
  }

#if 1
  // sample from the obtained probabilities and normalize the probs in a single pass
  // this is ~3x faster on Mac with full gpt-oss vocab than the version below
  //
  std::uniform_real_distribution<double> vDist(0.0f, 1.0f);
  const double cRnd = vDist(vCtx->vRng);

  double vSumRun = 0.0f;
  const double cSumTgt = vSumCum * cRnd;

  bool vFound = false;
  for (size_t vI = 0; vI < pCurP->size; ++vI) {
    if (!vFound) {
      // accumulate probs until we reach the target sum
      vSumRun += pCurP->data[vI].p;
      if (vSumRun >= cSumTgt) {
        pCurP->selected = vI;
        vFound = true;
      }
    }

    // normalize probs
    pCurP->data[vI].p /= vSumCum;
  }

  // fallback to the last token (don't think this can happen)
  assert(vFound);
  if (!vFound) {
    pCurP->selected = pCurP->size - 1;
  }
#else
  // for clarity, this is the same as above but does one pass for normalization and one extra pass for sampling
  for (size_t vI = 0; vI < pCurP->size; ++vI) {
    pCurP->data[vI].p /= vSumCum;
  }

  pCurP->selected = fMgsSampleDist(pCurP, vCtx->vRng);
#endif
}

static void fMgsSamplerDistReset(struct mgs_sampler *pSmpl) {
  auto *vCtx = (mgs_sampler_dist *)pSmpl->ctx;
  vCtx->vSeedCur = fGetRngSeed(vCtx->cSeed);
  vCtx->vRng.seed(vCtx->vSeedCur);
}

static struct mgs_sampler *fMgsSamplerDistClone(const struct mgs_sampler *pSmpl) {
  const auto *vCtx = (const mgs_sampler_dist *)pSmpl->ctx;
  auto *vResult = mgs_sampler_init_dist(vCtx->cSeed);

  // copy the state
  {
    auto *vResultCtx = (mgs_sampler_dist *)vResult->ctx;

    vResultCtx->vRng = vCtx->vRng;
  }

  return vResult;
}

static void fMgsSamplerDistFree(struct mgs_sampler *pSmpl) { delete (mgs_sampler_dist *)pSmpl->ctx; }

static bool fMgsSamplerDistBackendInit(struct mgs_sampler *pSmpl, ggml_backend_buffer_type_t pBuft) {
  auto *vSctx = (mgs_sampler_dist *)pSmpl->ctx;

  const bool cRes = fMgsSamplerBackendSupport(pSmpl, pBuft);

  vSctx->mInit(cRes);

  return cRes;
}

static void fMgsSamplerDistBackendApply(struct mgs_sampler *pSmpl, struct ggml_context *pCtx, struct ggml_cgraph *pGf,
                                        struct mgs_sampler_data *pData) {
  GGML_UNUSED(pGf);

  auto *vSctx = (mgs_sampler_dist *)pSmpl->ctx;

  vSctx->vInpUniform = ggml_new_tensor_1d(pCtx, GGML_TYPE_F32, 1);
  ggml_set_name(vSctx->vInpUniform, "uniform");
  ggml_set_input(vSctx->vInpUniform);

  // flatten
  struct ggml_tensor *vLogits = ggml_reshape_1d(pCtx, pData->logits, ggml_nelements(pData->logits));

  struct ggml_tensor *vProbs = ggml_soft_max(pCtx, vLogits);
  ggml_set_name(vProbs, "dist_probs");

  struct ggml_tensor *vCumsum = ggml_cumsum(pCtx, vProbs);
  ggml_set_name(vCumsum, "dist_cumsum");

  // The uniform tensor has a random value and we subtract this tensor with
  // the cumsum tensor (the uniform tensor will be broadcasted by ggml_sub).
  // Recall that each entry in cumsum is the cumulative probability up to that
  // index so values stay negative while the cumulative total is below the
  // random value, and become zero/positive once the threshold is crossed.
  struct ggml_tensor *vDiff = ggml_sub(pCtx, vCumsum, vSctx->vInpUniform);
  ggml_set_name(vDiff, "dist_cumsum");

  // The ggml_step function produces a tensor where entries are 1 if the
  // corresponding entry in diff is > 0, and 0 otherwise. So all values up to
  // the index where the cumulative probability exceeds the random value are 0,
  // and all entries after that are 1.
  struct ggml_tensor *vMask = ggml_step(pCtx, vDiff);
  ggml_set_name(vMask, "dist_mask");

  // Taking the sum of the mask gives us the sum of elements after the threshold
  // we are interested in.
  struct ggml_tensor *vIdxf = ggml_sum(pCtx, vMask);
  ggml_set_name(vIdxf, "dist_index_f32");

  // Use ggml_scale_bias to scale the index value by -1 and then add the size
  // of the mask to that value so we get the correct index ((-1 * idxf) + n).
  struct ggml_tensor *vIdx = ggml_cast(pCtx, ggml_scale_bias(pCtx, vIdxf, -1.0f, vMask->ne[0]), GGML_TYPE_I32);
  ggml_set_name(vIdx, "dist_index_i32");

  // Map back to original vocab ids if a candidates tensor is available.
  struct ggml_tensor *vSampledToken = vIdx;
  if (pData->candidates != nullptr) {
    struct ggml_tensor *vCandidates = ggml_reshape_2d(pCtx, pData->candidates, 1, ggml_nelements(pData->candidates));

    vSampledToken = ggml_get_rows(pCtx, vCandidates, vIdx);
    ggml_set_name(vSampledToken, "dist_sampled_token");
  }

  pData->sampled = vSampledToken;
  pData->probs = vProbs;
}

static void fMgsSamplerDistBackendSetInput(struct mgs_sampler *pSmpl) {
  auto *vSctx = (mgs_sampler_dist *)pSmpl->ctx;

  GGML_ASSERT(vSctx->vInpUniform != nullptr);

  // We sample in double precision and cast to float to match rnd numbers of
  // mgs_dampler_dist which uses double precision (sampling from
  // std::uniform_real_distribution<double> and
  // std::uniform_real_distribution<float> with same rng will produce
  // different sequences).
  std::uniform_real_distribution<double> vDist(0.0f, 1.0f);
  const float cRnd = vDist(vSctx->vRng);

  ggml_backend_tensor_set(vSctx->vInpUniform, &cRnd, 0, sizeof(float));
}

static struct mgs_sampler_i vMgsSamplerDistI = {
    /* .name              = */ fMgsSamplerDistName,
    /* .accept            = */ nullptr,
    /* .apply             = */ fMgsSamplerDistApply,
    /* .reset             = */ fMgsSamplerDistReset,
    /* .clone             = */ fMgsSamplerDistClone,
    /* .free              = */ fMgsSamplerDistFree,
    /* .backend_init      = */ fMgsSamplerDistBackendInit,
    /* .backend_accept    = */ nullptr,
    /* .backend_apply     = */ fMgsSamplerDistBackendApply,
    /* .backend_set_input = */ fMgsSamplerDistBackendSetInput,
};

struct mgs_sampler *mgs_sampler_init_dist(uint32_t pSeed) {
  auto vSeedCur = fGetRngSeed(pSeed);
  return mgs_sampler_init(
      /* .iface = */ &vMgsSamplerDistI,
      /* .ctx   = */ new mgs_sampler_dist{
          ("dist"),
          /* .cSeed       = */ pSeed,
          /* .vSeedCur    = */ vSeedCur,
          /* .vRng        = */ std::mt19937(vSeedCur),
          /* .vInpUniform = */ nullptr,
      });
}

// top-k

struct mgs_sampler_top_k : public mgs_sampler_backend {
  const int32_t cK;
};

static const char *fMgsSamplerTopKName(const struct mgs_sampler *pSmpl) {
  auto *vSctx = (mgs_sampler_top_k *)pSmpl->ctx;
  return vSctx->mGetName();
}

static void fMgsSamplerTopKApply(struct mgs_sampler *pSmpl, mgs_token_data_array *pCurP) {
  auto *vCtx = (mgs_sampler_top_k *)pSmpl->ctx;
  fMgsSamplerTopKImpl(pCurP, vCtx->cK);
}

static struct mgs_sampler *fMgsSamplerTopKClone(const struct mgs_sampler *pSmpl) {
  const auto *vCtx = (const mgs_sampler_top_k *)pSmpl->ctx;
  return mgs_sampler_init_top_k(vCtx->cK);
}

static void fMgsSamplerTopKFree(struct mgs_sampler *pSmpl) { delete (mgs_sampler_top_k *)pSmpl->ctx; }

static bool fMgsSamplerTopKBackendInit(struct mgs_sampler *pSmpl, ggml_backend_buffer_type_t pBuft) {
  auto *vSctx = (mgs_sampler_top_k *)pSmpl->ctx;

  const bool cRes = fMgsSamplerBackendSupport(pSmpl, pBuft);

  vSctx->mInit(cRes);

  return cRes;
}

static void fMgsSamplerTopKBackendApply(struct mgs_sampler *pSmpl, struct ggml_context *pCtx, struct ggml_cgraph *pGf,
                                        struct mgs_sampler_data *pData) {
  auto *vSctx = (mgs_sampler_top_k *)pSmpl->ctx;

  struct ggml_tensor *vLogits = ggml_reshape_1d(pCtx, pData->logits, ggml_nelements(pData->logits));

  struct ggml_tensor *vTopK = ggml_top_k(pCtx, vLogits, vSctx->cK);
  ggml_set_name(vTopK, "top_k");

  if (pData->candidates) {
    struct ggml_tensor *vCandidatesRows = ggml_reshape_2d(pCtx, pData->candidates, 1, pData->candidates->ne[0]);
    pData->candidates = ggml_get_rows(pCtx, vCandidatesRows, vTopK);
    ggml_set_name(pData->candidates, "top_k_candidates");
  } else {
    pData->candidates = vTopK;
  }

  struct ggml_tensor *vLogitsRows = ggml_reshape_2d(pCtx, vLogits, 1, vLogits->ne[0]);
  pData->logits = ggml_get_rows(pCtx, vLogitsRows, vTopK);
  ggml_set_name(pData->logits, "top_k_rows");

  GGML_UNUSED(pGf);
}

static struct mgs_sampler_i vMgsSamplerTopKI = {
    /* .name              = */ fMgsSamplerTopKName,
    /* .accept            = */ nullptr,
    /* .apply             = */ fMgsSamplerTopKApply,
    /* .reset             = */ nullptr,
    /* .clone             = */ fMgsSamplerTopKClone,
    /* .free              = */ fMgsSamplerTopKFree,
    /* .backend_init      = */ fMgsSamplerTopKBackendInit,
    /* .backend_accept    = */ nullptr,
    /* .backend_apply     = */ fMgsSamplerTopKBackendApply,
    /* .backend_set_input = */ nullptr,
};

struct mgs_sampler *mgs_sampler_init_top_k(int32_t pK) {
  const bool cIsEmpty = (pK <= 0);

  if (cIsEmpty) {
    return fMgsSamplerInitEmpty("?top-k");
  }

  return mgs_sampler_init(
      /* .iface = */ &vMgsSamplerTopKI,
      /* .ctx   = */ new mgs_sampler_top_k{
          ("top-k"),
          /* .cK = */ pK,
      });
}

// top-p

struct mgs_sampler_top_p : public mgs_sampler_backend {
  const float cP;
  const size_t cMinKeep;

  std::vector<mgs_token_data> lBufSort;
};

static const char *fMgsSamplerTopPName(const struct mgs_sampler *pSmpl) {
  auto *vSctx = (mgs_sampler_top_p *)pSmpl->ctx;
  return vSctx->mGetName();
}

static void fMgsSamplerTopPApply(struct mgs_sampler *pSmpl, mgs_token_data_array *pCurP) {
  auto *vCtx = (mgs_sampler_top_p *)pSmpl->ctx;

  if (vCtx->cP >= 1.0f) {
    return;
  }

  fMgsSamplerSoftmaxImpl(pCurP, false);

  size_t vK = pCurP->size;
  auto *vPdata = pCurP->data;

  auto &vBufSort = vCtx->lBufSort;

  // if not sorted, try adaptive top-k sorting
  if (!pCurP->sorted && pCurP->size > 1024) {
    vK = std::min<size_t>(256, pCurP->size);
    fMgsTokenDataArrayPartialSort(*pCurP, vK, vBufSort);
    vPdata = vBufSort.data();
  } else if (!pCurP->sorted) {
    // small candidates -> sort inplace
    fMgsTokenDataArrayPartialSortInplace(pCurP, vK);
  }

  // Compute the cumulative probabilities
  float vCumSum = 0.0f;
  size_t vLastIdx = pCurP->size;

  for (size_t vI = 0; vI < pCurP->size; ++vI) {
    vCumSum += vPdata[vI].p;

    // Check if the running sum is at least p or if we have kept at least min_keep tokens
    // we set the last index to i+1 to indicate that the current iterate should be included in the set
    if (vCumSum >= vCtx->cP && vI + 1 >= vCtx->cMinKeep) {
      vLastIdx = vI + 1;
      break;
    }

    // we exceeded the current top-k heuristic -> increase k and continue
    if (!pCurP->sorted && vI == vK - 1) {
      vK = pCurP->size;
      fMgsTokenDataArrayPartialSort(*pCurP, vK, vBufSort);
      vPdata = vBufSort.data();
    }
  }

  // Resize the output vector to keep only the top-p tokens
  if (!pCurP->sorted) {
    std::copy(vBufSort.data(), vBufSort.data() + vLastIdx, pCurP->data);
    pCurP->sorted = true;
  }

  pCurP->size = vLastIdx;
}

static struct mgs_sampler *fMgsSamplerTopPClone(const struct mgs_sampler *pSmpl) {
  const auto *vCtx = (const mgs_sampler_top_p *)pSmpl->ctx;
  return mgs_sampler_init_top_p(vCtx->cP, vCtx->cMinKeep);
}

static void fMgsSamplerTopPFree(struct mgs_sampler *pSmpl) { delete (mgs_sampler_top_p *)pSmpl->ctx; }

static bool fMgsSamplerTopPBackendInit(struct mgs_sampler *pSmpl, ggml_backend_buffer_type_t pBuft) {
  auto *vSctx = (mgs_sampler_top_p *)pSmpl->ctx;

  const bool cRes = fMgsSamplerBackendSupport(pSmpl, pBuft);

  vSctx->mInit(cRes);

  return cRes;
}

static void fMgsSamplerTopPBackendApply(struct mgs_sampler *pSmpl, struct ggml_context *pCtx, struct ggml_cgraph *pGf,
                                        struct mgs_sampler_data *pData) {
  auto *vSctx = (mgs_sampler_top_p *)pSmpl->ctx;

  // flatten
  struct ggml_tensor *vLogits = ggml_reshape_1d(pCtx, pData->logits, ggml_nelements(pData->logits));

  auto vGgmlSort = [pCtx](struct ggml_tensor *pA, struct ggml_tensor *pB) {
    GGML_ASSERT(ggml_nrows(pA) == 1);
    struct ggml_tensor *vAReshaped = ggml_reshape_2d(pCtx, pA, 1, pA->ne[0]);
    struct ggml_tensor *vASorted = ggml_get_rows(pCtx, vAReshaped, pB);
    return vASorted;
  };

  // Get the sorted logits in descending order.
  struct ggml_tensor *vSortedIdx = ggml_argsort(pCtx, vLogits, GGML_SORT_ORDER_DESC);
  ggml_set_name(vSortedIdx, "top_p_sorted_idx");

  // Do the sorting via reshape + get_rows
  struct ggml_tensor *vSortedLogits = vGgmlSort(vLogits, vSortedIdx);
  ggml_set_name(vSortedLogits, "top_p_sorted_logits");

  vSortedLogits = ggml_reshape_1d(pCtx, vSortedLogits, ggml_nelements(vSortedLogits));
  struct ggml_tensor *vSoftmax = ggml_soft_max(pCtx, vSortedLogits);
  ggml_set_name(vSoftmax, "top_p_softmax");

  // If candidates are provided, sort them as well. Otherwise, set sorted indices as candidates.
  if (pData->candidates) {
    pData->candidates = vGgmlSort(pData->candidates, vSortedIdx);
  } else {
    pData->candidates = vSortedIdx;
  }
  ggml_set_name(pData->candidates, "top_p_candidates");

  // Compute Cumulative Distribution Function (CDF) by means of GGML_OP_CUMSUM.
  struct ggml_tensor *vCdf = ggml_cumsum(pCtx, vSoftmax);
  ggml_set_name(vCdf, "top_p_cdf");

  // Invert CDF and add top-p value so that ggml_step yields 1 for values we want to keep
  struct ggml_tensor *vCdfScaled = ggml_scale_bias(pCtx, vCdf, -1.0f, vSctx->cP);
  ggml_set_name(vCdfScaled, "top_p_cdf_scaled");

  struct ggml_tensor *vMask = ggml_step(pCtx, vCdfScaled);
  ggml_set_name(vMask, "top_p_mask");

  // Taking the sum of the mask gives us the sum of elements after the threshold
  // we are interested in.
  struct ggml_tensor *vIdxf = ggml_sum(pCtx, vMask);
  ggml_set_name(vIdxf, "top_p_index_f32");

  // prevent out-of-bounds access
  vIdxf = ggml_clamp(pCtx, vIdxf, 0.0f, vMask->ne[0] - 1);

  // construct ones tensor to set the value in the mask
  struct ggml_tensor *vOnes = ggml_scale_bias(pCtx, vIdxf, 0.0f, 1.0f);
  ggml_set_name(vOnes, "top_p_ones");

  // Make top-p inclusive (i.e. return all values such that cum_sum/cdf >= p)
  struct ggml_tensor *vMaskReshaped = ggml_reshape_2d(pCtx, vMask, 1, vMask->ne[0]);

  vMaskReshaped = ggml_set_rows(pCtx, vMaskReshaped, vOnes, ggml_cast(pCtx, vIdxf, GGML_TYPE_I32));
  vMask = ggml_reshape_1d(pCtx, vMaskReshaped, vMask->ne[0]);

  // Apply -INFINITY bias for masked-out tokens
  // log(1) = 0 (keep), log(0) = -INF (discard)
  struct ggml_tensor *vTopPBias = ggml_log(pCtx, vMask);
  ggml_set_name(vTopPBias, "top_p_bias");

  pData->logits = ggml_add(pCtx, vSortedLogits, vTopPBias);
  ggml_set_name(pData->logits, "top_p_logits");

  GGML_UNUSED(pGf);
}

static struct mgs_sampler_i vMgsSamplerTopPI = {
    /* .name              = */ fMgsSamplerTopPName,
    /* .accept            = */ nullptr,
    /* .apply             = */ fMgsSamplerTopPApply,
    /* .reset             = */ nullptr,
    /* .clone             = */ fMgsSamplerTopPClone,
    /* .free              = */ fMgsSamplerTopPFree,
    /* .backend_init      = */ fMgsSamplerTopPBackendInit,
    /* .backend_accept    = */ nullptr,
    /* .backend_apply     = */ fMgsSamplerTopPBackendApply,
    /* .backend_set_input = */ nullptr,
};

struct mgs_sampler *mgs_sampler_init_top_p(float pP, size_t pMinKeep) {
  const bool cIsEmpty = pP >= 1.0f;

  if (cIsEmpty) {
    return fMgsSamplerInitEmpty("?top-p");
  }

  return mgs_sampler_init(
      /* .iface = */ &vMgsSamplerTopPI,
      /* .ctx   = */ new mgs_sampler_top_p{
          ("top-p"),
          /* .cP       = */ pP,
          /* .cMinKeep = */ pMinKeep,
          /* .lBufSort = */ {},
      });
}

// min-p

struct mgs_sampler_min_p : public mgs_sampler_backend {
  const float cP;
  const size_t cMinKeep;
};

static const char *fMgsSamplerMinPName(const struct mgs_sampler *pSmpl) {
  auto *vSctx = (mgs_sampler_min_p *)pSmpl->ctx;
  return vSctx->mGetName();
}

static void fMgsSamplerMinPApply(struct mgs_sampler *pSmpl, mgs_token_data_array *pCurP) {
  auto *vCtx = (mgs_sampler_min_p *)pSmpl->ctx;

  if (vCtx->cP <= 0.0f || !pCurP->size) {
    return;
  }

  bool vMinPApplied = false;

  // if the cur_p aren't sorted, try the unsorted implementation first
  if (!pCurP->sorted) {
    std::vector<mgs_token_data> lFilteredTokens;

    float vMaxLogit = -FLT_MAX;
    for (size_t vI = 0; vI < pCurP->size; ++vI) {
      vMaxLogit = std::max(vMaxLogit, pCurP->data[vI].logit);
    }
    const float cMinLogit = vMaxLogit + logf(vCtx->cP); // min logit for p_i >= p * p_max

    for (size_t vI = 0; vI < pCurP->size; ++vI) {
      if (pCurP->data[vI].logit >= cMinLogit) {
        lFilteredTokens.push_back(pCurP->data[vI]);
      }
    }

    // if we have enough values the operation was a success
    if (!lFilteredTokens.empty() && lFilteredTokens.size() >= vCtx->cMinKeep) {
      std::copy(lFilteredTokens.begin(), lFilteredTokens.end(), pCurP->data);
      pCurP->size = lFilteredTokens.size();
      vMinPApplied = true;
    }
  }

  // if the cur_p are sorted or the unsorted implementation failed, use this implementation
  if (!vMinPApplied) {
    // Sort the logits in descending order
    if (!pCurP->sorted) {
      fMgsTokenDataArrayPartialSortInplace(pCurP, pCurP->size);
    }

    const float cMinLogit = pCurP->data[0].logit + logf(vCtx->cP); // min logit for p_i >= p * p_max
    size_t vI = 1;                                                 // first token always matches

    for (; vI < pCurP->size; ++vI) {
      if (pCurP->data[vI].logit < cMinLogit && vI >= vCtx->cMinKeep) {
        break; // prob too small
      }
    }

    // Resize the output vector to keep only the matching tokens
    pCurP->size = vI;
  }
}

static struct mgs_sampler *fMgsSamplerMinPClone(const struct mgs_sampler *pSmpl) {
  const auto *vCtx = (const mgs_sampler_min_p *)pSmpl->ctx;
  return mgs_sampler_init_min_p(vCtx->cP, vCtx->cMinKeep);
}

static void fMgsSamplerMinPFree(struct mgs_sampler *pSmpl) { delete (mgs_sampler_min_p *)pSmpl->ctx; }

static bool fMgsSamplerMinPBackendInit(struct mgs_sampler *pSmpl, ggml_backend_buffer_type_t pBuft) {
  auto *vSctx = (mgs_sampler_min_p *)pSmpl->ctx;

  const bool cRes = fMgsSamplerBackendSupport(pSmpl, pBuft);

  vSctx->mInit(cRes);

  return cRes;
}

static void fMgsSamplerMinPBackendApply(struct mgs_sampler *pSmpl, struct ggml_context *pCtx, struct ggml_cgraph *pGf,
                                        struct mgs_sampler_data *pData) {
  auto *vSctx = (mgs_sampler_min_p *)pSmpl->ctx;

  struct ggml_tensor *vLogits = ggml_reshape_1d(pCtx, pData->logits, ggml_nelements(pData->logits));

  struct ggml_tensor *vMaxIdx = ggml_argmax(pCtx, vLogits);
  ggml_set_name(vMaxIdx, "max_idx");

  struct ggml_tensor *vLogitsRows = ggml_reshape_2d(pCtx, vLogits, 1, vLogits->ne[0]);
  ggml_set_name(vLogitsRows, "logits_rows");

  struct ggml_tensor *vMaxLogit = ggml_get_rows(pCtx, vLogitsRows, vMaxIdx);
  ggml_set_name(vMaxLogit, "max_logit");

  // Calculate the threshold value.
  struct ggml_tensor *vThreshold = ggml_scale_bias(pCtx, vMaxLogit, 1.0f, logf(vSctx->cP));
  ggml_set_name(vThreshold, "min_p_threshold");

  // Subtract the threshold from logits.
  struct ggml_tensor *vSub = ggml_sub(pCtx, vLogits, vThreshold);

  // Create a mask where logits below the threshold are 0 (discard),
  // and others are 1 (keep).
  struct ggml_tensor *vMask = ggml_step(pCtx, vSub);
  ggml_set_name(vMask, "min_p_mask");

  // Apply -INFINITY bias for masked-out tokens
  // log(1) = 0 (keep), log(0) = -INF (discard)
  struct ggml_tensor *vMinPBias = ggml_log(pCtx, vMask);
  ggml_set_name(vMinPBias, "min_p_bias");

  pData->logits = ggml_add(pCtx, vLogits, vMinPBias);
  ggml_set_name(pData->logits, "min_p_logits");

  GGML_UNUSED(pGf);
}

static struct mgs_sampler_i vMgsSamplerMinPI = {
    /* .name              = */ fMgsSamplerMinPName,
    /* .accept            = */ nullptr,
    /* .apply             = */ fMgsSamplerMinPApply,
    /* .reset             = */ nullptr,
    /* .clone             = */ fMgsSamplerMinPClone,
    /* .free              = */ fMgsSamplerMinPFree,
    /* .backend_init      = */ fMgsSamplerMinPBackendInit,
    /* .backend_accept    = */ nullptr,
    /* .backend_apply     = */ fMgsSamplerMinPBackendApply,
    /* .backend_set_input = */ nullptr,
};

struct mgs_sampler *mgs_sampler_init_min_p(float pP, size_t pMinKeep) {
  const bool cIsEmpty = (pP <= 0.0f);

  if (cIsEmpty) {
    return fMgsSamplerInitEmpty("?min-p");
  }

  return mgs_sampler_init(
      /* .iface = */ &vMgsSamplerMinPI,
      /* .ctx   = */ new mgs_sampler_min_p{
          ("min-p"),
          /* .cP       = */ pP,
          /* .cMinKeep = */ pMinKeep,
      });
}

// typical

struct mgs_sampler_typical {
  const float cP;
  const size_t cMinKeep;
};

static const char *fMgsSamplerTypicalName(const struct mgs_sampler * /*smpl*/) { return "typical"; }

static void fMgsSamplerTypicalApply(struct mgs_sampler *pSmpl, mgs_token_data_array *pCurP) {
  auto *vCtx = (mgs_sampler_typical *)pSmpl->ctx;

  // Reference implementation:
  // https://github.com/huggingface/transformers/compare/main...cimeister:typical-sampling:typical-pr
  if (vCtx->cP >= 1.0f) {
    return;
  }

  // Compute the softmax of logits and calculate entropy
  fMgsSamplerSoftmaxImpl(pCurP, true);

  float vEntropy = 0.0f;
  for (size_t vI = 0; vI < pCurP->size; ++vI) {
    vEntropy += -pCurP->data[vI].p * logf(pCurP->data[vI].p);
  }

  // Compute the absolute difference between negative log probability and entropy for each candidate
  std::vector<float> lShiftedScores;
  for (size_t vI = 0; vI < pCurP->size; ++vI) {
    float vShiftedScore = fabsf(-logf(pCurP->data[vI].p) - vEntropy);
    lShiftedScores.push_back(vShiftedScore);
  }

  // Sort tokens based on the shifted_scores and their corresponding indices
  std::vector<size_t> lIndices(pCurP->size);
  std::iota(lIndices.begin(), lIndices.end(), 0);

  std::sort(lIndices.begin(), lIndices.end(),
            [&](size_t pA, size_t pB) { return lShiftedScores[pA] < lShiftedScores[pB]; });

  // Compute the cumulative probabilities
  float vCumSum = 0.0f;
  size_t vLastIdx = lIndices.size();

  for (size_t vI = 0; vI < lIndices.size(); ++vI) {
    size_t vIdx = lIndices[vI];
    vCumSum += pCurP->data[vIdx].p;

    // Check if the running sum is greater than typical or if we have kept at least min_keep tokens
    if (vCumSum > vCtx->cP && (vCtx->cMinKeep == 0 || vI >= vCtx->cMinKeep - 1)) {
      vLastIdx = vI + 1;
      break;
    }
  }

  // Resize the output vector to keep only the locally typical tokens
  std::vector<mgs_token_data> lCurPNew;
  for (size_t vI = 0; vI < vLastIdx; ++vI) {
    size_t vIdx = lIndices[vI];
    lCurPNew.push_back(pCurP->data[vIdx]);
  }

  // Replace the data in cur_p with the cur_p_new data
  std::copy(lCurPNew.begin(), lCurPNew.end(), pCurP->data);
  pCurP->size = lCurPNew.size();
  pCurP->sorted = false;
}

static struct mgs_sampler *fMgsSamplerTypicalClone(const struct mgs_sampler *pSmpl) {
  const auto *vCtx = (const mgs_sampler_typical *)pSmpl->ctx;
  return mgs_sampler_init_typical(vCtx->cP, vCtx->cMinKeep);
}

static void fMgsSamplerTypicalFree(struct mgs_sampler *pSmpl) { delete (mgs_sampler_typical *)pSmpl->ctx; }

static struct mgs_sampler_i vMgsSamplerTypicalI = {
    /* .name              = */ fMgsSamplerTypicalName,
    /* .accept            = */ nullptr,
    /* .apply             = */ fMgsSamplerTypicalApply,
    /* .reset             = */ nullptr,
    /* .clone             = */ fMgsSamplerTypicalClone,
    /* .free              = */ fMgsSamplerTypicalFree,
    /* .backend_init      = */ nullptr,
    /* .backend_accept    = */ nullptr,
    /* .backend_apply     = */ nullptr,
    /* .backend_set_input = */ nullptr,
};

struct mgs_sampler *mgs_sampler_init_typical(float pP, size_t pMinKeep) {
  const bool cIsEmpty = (pP >= 1.0f);

  if (cIsEmpty) {
    return fMgsSamplerInitEmpty("?typical");
  }

  return mgs_sampler_init(
      /* .iface = */ &vMgsSamplerTypicalI,
      /* .ctx   = */ new mgs_sampler_typical{
          /* .cP       = */ pP,
          /* .cMinKeep = */ pMinKeep,
      });
}

// temp

struct mgs_sampler_temp : public mgs_sampler_backend {
  const float cTemp;
};

static const char *fMgsSamplerTempName(const struct mgs_sampler *pSmpl) {
  auto *vSctx = (mgs_sampler_temp *)pSmpl->ctx;
  return vSctx->mGetName();
}

static void fMgsSamplerTempApply(struct mgs_sampler *pSmpl, mgs_token_data_array *pCurP) {
  const auto *vCtx = (mgs_sampler_temp *)pSmpl->ctx;

  fMgsSamplerTempImpl(pCurP, vCtx->cTemp);
}

static struct mgs_sampler *fMgsSamplerTempClone(const struct mgs_sampler *pSmpl) {
  const auto *vCtx = (const mgs_sampler_temp *)pSmpl->ctx;
  return mgs_sampler_init_temp(vCtx->cTemp);
}

static void fMgsSamplerTempFree(struct mgs_sampler *pSmpl) { delete (mgs_sampler_temp *)pSmpl->ctx; }

static void fMgsSamplerBackendTempSampling(struct ggml_context *pCtx, struct ggml_cgraph *pGf,
                                           struct mgs_sampler_data *pData, float pTemp) {
  if (pTemp <= 0.0f) {
    struct ggml_tensor *vLogits = ggml_reshape_1d(pCtx, pData->logits, ggml_nelements(pData->logits));

    // Find the most probable token index.
    struct ggml_tensor *vMaxIdx = ggml_argmax(pCtx, vLogits);
    ggml_set_name(vMaxIdx, "temp_max_idx");

    if (pData->candidates) {
      struct ggml_tensor *vCandidatesRows =
          ggml_reshape_2d(pCtx, pData->candidates, 1, ggml_nelements(pData->candidates));
      pData->candidates = ggml_get_rows(pCtx, vCandidatesRows, vMaxIdx);
    } else {
      pData->candidates = vMaxIdx;
    }

    struct ggml_tensor *vLogitsRows = ggml_reshape_2d(pCtx, vLogits, 1, ggml_nelements(vLogits));
    pData->logits = ggml_get_rows(pCtx, vLogitsRows, vMaxIdx);

    return;
  }

  pData->logits = ggml_scale(pCtx, pData->logits, 1.0f / pTemp);

  GGML_UNUSED(pGf);
}

static bool fMgsSamplerTempBackendInit(struct mgs_sampler *pSmpl, ggml_backend_buffer_type_t pBuft) {
  auto *vSctx = (mgs_sampler_temp *)pSmpl->ctx;

  const bool cRes = fMgsSamplerBackendSupport(pSmpl, pBuft);

  vSctx->mInit(cRes);

  return cRes;
}

static void fMgsSamplerTempBackendApply(struct mgs_sampler *pSmpl, struct ggml_context *pCtx, struct ggml_cgraph *pGf,
                                        struct mgs_sampler_data *pData) {
  auto *vSctx = (mgs_sampler_temp *)pSmpl->ctx;
  fMgsSamplerBackendTempSampling(pCtx, pGf, pData, vSctx->cTemp);
}

static struct mgs_sampler_i vMgsSamplerTempI = {
    /* .name              = */ fMgsSamplerTempName,
    /* .accept            = */ nullptr,
    /* .apply             = */ fMgsSamplerTempApply,
    /* .reset             = */ nullptr,
    /* .clone             = */ fMgsSamplerTempClone,
    /* .free              = */ fMgsSamplerTempFree,
    /* .backend_init      = */ fMgsSamplerTempBackendInit,
    /* .backend_accept    = */ nullptr,
    /* .backend_apply     = */ fMgsSamplerTempBackendApply,
    /* .backend_set_input = */ nullptr,
};

struct mgs_sampler *mgs_sampler_init_temp(float pTemp) {
  const bool cIsEmpty = pTemp == 1.0f;

  if (cIsEmpty) {
    return fMgsSamplerInitEmpty("?temp");
  }

  return mgs_sampler_init(
      /* .iface = */ &vMgsSamplerTempI,
      /* .ctx   = */ new mgs_sampler_temp{
          ("temp"),
          /*.cTemp = */ pTemp,
      });
}

// temp-ext

struct mgs_sampler_temp_ext : public mgs_sampler_backend {
  const float cTemp;
  const float cDelta;
  const float cExponent;
};

static const char *fMgsSamplerTempExtName(const struct mgs_sampler *pSmpl) {
  auto *vSctx = (mgs_sampler_temp_ext *)pSmpl->ctx;
  return vSctx->mGetName();
}

static void fMgsSamplerTempExtApply(struct mgs_sampler *pSmpl, mgs_token_data_array *pCurP) {
  auto *vCtx = (mgs_sampler_temp_ext *)pSmpl->ctx;
  if (vCtx->cDelta > 0) {
    const float cMinTemp = std::max(0.0f, vCtx->cTemp - vCtx->cDelta);
    const float cMaxTemp = vCtx->cTemp + vCtx->cDelta;

    float vExponentVal = vCtx->cExponent;

    // no need to do anything if there is only one (or zero) candidates
    if (pCurP->size <= 1) {
      return;
    }

    // Calculate maximum possible entropy
    float vMaxEntropy = -logf(1.0f / pCurP->size);

    fMgsSamplerSoftmaxImpl(pCurP, true);

    // Calculate entropy of the softmax probabilities
    float vEntropy = 0.0f;
    for (size_t vI = 0; vI < pCurP->size; ++vI) {
      float vProb = pCurP->data[vI].p;
      if (vProb > 0.0f) { // Ensure no log(0)
        vEntropy -= vProb * logf(vProb);
      }
    }

    // Normalize the entropy (max_entropy cannot be 0 here because we checked cur_p->size != 1 above)
    float vNormalizedEntropy = vEntropy / vMaxEntropy;

    // Map the normalized entropy to the desired temperature range using the power function
    float vDynTemp = cMinTemp + (cMaxTemp - cMinTemp) * powf(vNormalizedEntropy, vExponentVal);

#ifdef DEBUG
    MGS_LOG_INFO("Your text maxtemp value is: %f\n", cMaxTemp);
    MGS_LOG_INFO("Entropy: %f\n", vEntropy);
    MGS_LOG_INFO("Max Possible Entropy: %f\n", vMaxEntropy);
    MGS_LOG_INFO("Normalized Entropy: %f\n", vNormalizedEntropy);
    MGS_LOG_INFO("Exponent: %f\n", vExponentVal);
    MGS_LOG_INFO("Dynamic Temperature (dyn_temp): %f\n", vDynTemp);
#endif

    // Apply the dynamically calculated temperature scaling
    fMgsSamplerTempImpl(pCurP, vDynTemp);

    // Re-compute softmax probabilities after scaling logits with dynamic temperature
    const double cMaxLDouble = pCurP->data[0].logit;

    double vCumSumDouble = 0.0;
    for (size_t vI = 0; vI < pCurP->size; ++vI) {
      double vP = exp(pCurP->data[vI].logit - cMaxLDouble);
      pCurP->data[vI].p = vP; // Store the scaled probability
      vCumSumDouble += vP;
    }

    for (size_t vI = 0; vI < pCurP->size; ++vI) {
      pCurP->data[vI].p /= vCumSumDouble; // Re-normalize the probabilities
    }

#ifdef DEBUG
    // Print the updated top 25 probabilities after temperature scaling
    MGS_LOG_INFO("\nUpdated Top 25 Probabilities After Dynamic Temperature Scaling (in percentages):\n");
    for (size_t vI = 0; vI < 25 && vI < pCurP->size; ++vI) {
      MGS_LOG_INFO("Token %zu: %f%%\n", vI + 1, pCurP->data[vI].p * 100.0f);
    }
#endif
  } else {
    fMgsSamplerTempImpl(pCurP, vCtx->cTemp);
  }
}

static struct mgs_sampler *fMgsSamplerTempExtClone(const struct mgs_sampler *pSmpl) {
  const auto *vCtx = (const mgs_sampler_temp_ext *)pSmpl->ctx;
  return mgs_sampler_init_temp_ext(vCtx->cTemp, vCtx->cDelta, vCtx->cExponent);
}

static void fMgsSamplerTempExtFree(struct mgs_sampler *pSmpl) { delete (mgs_sampler_temp_ext *)pSmpl->ctx; }

static bool fMgsSamplerTempExtBackendInit(struct mgs_sampler *pSmpl, ggml_backend_buffer_type_t pBuft) {
  auto *vSctx = (mgs_sampler_temp_ext *)pSmpl->ctx;

  const bool cRes = fMgsSamplerBackendSupport(pSmpl, pBuft);

  vSctx->mInit(cRes);

  return cRes;
}

static void fMgsSamplerTempExtBackendApply(struct mgs_sampler *pSmpl, struct ggml_context *pCtx,
                                           struct ggml_cgraph *pGf, struct mgs_sampler_data *pData) {
  auto *vSctx = (mgs_sampler_temp_ext *)pSmpl->ctx;

  // Revert to standard temperature scaling if delta or temp are non-positive.
  if (vSctx->cDelta <= 0.0f || vSctx->cTemp <= 0.0f) {
    fMgsSamplerBackendTempSampling(pCtx, pGf, pData, vSctx->cTemp);
    return;
  }

  struct ggml_tensor *vLogits = ggml_reshape_1d(pCtx, pData->logits, ggml_nelements(pData->logits));

  // Calculate min_temp, max_temp, and max_entropy.
  const float cMinTemp = std::max(0.0f, vSctx->cTemp - vSctx->cDelta);
  const float cMaxTemp = vSctx->cTemp + vSctx->cDelta;
  const float cMaxEntropy = logf(vLogits->ne[0]);

  // Calculate the probabilities.
  struct ggml_tensor *vProbs = ggml_soft_max(pCtx, vLogits);
  ggml_set_name(vProbs, "temp_ext_softmax_probs");

  // Clamp probabilities to avoid log(0) which would give -inf
  struct ggml_tensor *vProbsClamped = ggml_clamp(pCtx, vProbs, 1e-10f, 1.0f);
  ggml_set_name(vProbsClamped, "temp_ext_probs_clamped");

  // Calculate the entropy, entropy = -Σ(p * log(p)).
  struct ggml_tensor *vLogProbs = ggml_log(pCtx, vProbsClamped);
  struct ggml_tensor *vPLogP = ggml_mul(pCtx, vProbsClamped, vLogProbs);
  struct ggml_tensor *vSumPLogP = ggml_sum(pCtx, vPLogP);
  struct ggml_tensor *vEntropy = ggml_scale(pCtx, vSumPLogP, -1.0f);
  ggml_set_name(vLogProbs, "temp_ext_log_probs");
  ggml_set_name(vPLogP, "temp_ext_p_log_p");
  ggml_set_name(vSumPLogP, "temp_ext_sum_p_log_p");
  ggml_set_name(vEntropy, "temp_ext_entropy");

  // Normalize the entropy, norm_entropy = entropy / max_entropy
  struct ggml_tensor *vNormEntropy = ggml_scale(pCtx, vEntropy, 1.0f / cMaxEntropy);
  ggml_set_name(vNormEntropy, "temp_ext_norm_entropy");

  // Calculate the dynamic temperature:
  // dyn_temp = min_temp + (max_temp - min_temp) * powf(normalized_entropy, exponent);
  //
  // Calculate powf(normalized_entropy, exponent) as
  // norm_entropy^exponent = exp(exponent * log(norm_entropy))
  struct ggml_tensor *vLogNormEntropy = ggml_log(pCtx, vNormEntropy);
  struct ggml_tensor *vScaledLog = ggml_scale(pCtx, vLogNormEntropy, vSctx->cExponent);
  struct ggml_tensor *vPowEntropy = ggml_exp(pCtx, vScaledLog);
  // With pow_entropy computed we can now compute dyn_temp, scaling by
  // (max_temp - min_temp) and then adding min_temp.
  struct ggml_tensor *vDynTemp = ggml_scale_bias(pCtx, vPowEntropy, cMaxTemp - cMinTemp, cMinTemp);
  ggml_set_name(vLogNormEntropy, "temp_ext_log_norm_entropy");
  ggml_set_name(vScaledLog, "temp_ext_scaled_log");
  ggml_set_name(vPowEntropy, "temp_ext_pow_entropy");
  ggml_set_name(vDynTemp, "temp_ext_dyn_temp");

  // Scale the logits by the dynamic temperature
  struct ggml_tensor *vScaledLogits = ggml_div(pCtx, vLogits, vDynTemp);
  ggml_set_name(vScaledLogits, "temp_ext_scaled_logits");

  pData->logits = vScaledLogits;
}

static struct mgs_sampler_i vMgsSamplerTempExtI = {
    /* .name              = */ fMgsSamplerTempExtName,
    /* .accept            = */ nullptr,
    /* .apply             = */ fMgsSamplerTempExtApply,
    /* .reset             = */ nullptr,
    /* .clone             = */ fMgsSamplerTempExtClone,
    /* .free              = */ fMgsSamplerTempExtFree,
    /* .backend_init      = */ fMgsSamplerTempExtBackendInit,
    /* .backend_accept    = */ nullptr,
    /* .backend_apply     = */ fMgsSamplerTempExtBackendApply,
    /* .backend_set_input = */ nullptr,
};

struct mgs_sampler *mgs_sampler_init_temp_ext(float pTemp, float pDelta, float pExponent) {
  const bool cIsEmpty = pTemp == 1.0f && pDelta <= 0.0f;

  if (cIsEmpty) {
    return fMgsSamplerInitEmpty("?temp-ext");
  }

  auto *vRes = mgs_sampler_init(
      /* .iface = */ &vMgsSamplerTempExtI,
      /* .ctx   = */ new mgs_sampler_temp_ext{
          ("temp-ext"),
          /* .cTemp     = */ pTemp,
          /* .cDelta    = */ pDelta,
          /* .cExponent = */ pExponent,
      });

  return vRes;
}

// xtc

struct mgs_sampler_xtc {
  const float cProbability;
  const float cThreshold;
  const size_t cMinKeep;

  const uint32_t cSeed;
  uint32_t vSeedCur;

  std::mt19937 vRng;
};

static const char *fMgsSamplerXtcName(const struct mgs_sampler * /*smpl*/) { return "xtc"; }

static void fMgsSampleXtcApply(struct mgs_sampler *pSmpl, mgs_token_data_array *pCurP) {
  auto *vCtx = (mgs_sampler_xtc *)pSmpl->ctx;

  if (vCtx->cProbability <= 0.0f || vCtx->cThreshold > 0.5f || pCurP->size < 2) {
    return;
  }

  std::uniform_real_distribution<float> vDistribution(0.0f, 1.0f);
  float vChance = vDistribution(vCtx->vRng);
  if (vChance > vCtx->cProbability) {
    return;
  }

  fMgsSamplerSoftmaxImpl(pCurP, true);

  int vPosLast = 0;

  for (size_t vI = 0; vI < pCurP->size; ++vI) {
    if (pCurP->data[vI].p >= vCtx->cThreshold) {
      vPosLast = vI;
    } else {
      break;
    }
  }

  if (pCurP->size - vPosLast >= vCtx->cMinKeep && vPosLast > 0) {
    pCurP->data += vPosLast;
    pCurP->size -= vPosLast;
  }
}

static struct mgs_sampler *fMgsSamplerXtcClone(const struct mgs_sampler *pSmpl) {
  const auto *vCtx = (const mgs_sampler_xtc *)pSmpl->ctx;
  auto *vResult = mgs_sampler_init_xtc(vCtx->cProbability, vCtx->cThreshold, vCtx->cMinKeep, vCtx->cSeed);

  // copy the state
  {
    auto *vResultCtx = (mgs_sampler_xtc *)vResult->ctx;

    vResultCtx->vRng = vCtx->vRng;
  }

  return vResult;
}

static void fMgsSamplerXtcFree(struct mgs_sampler *pSmpl) { delete (mgs_sampler_xtc *)pSmpl->ctx; }

static void fMgsSamplerXtcReset(struct mgs_sampler *pSmpl) {
  auto *vCtx = (mgs_sampler_xtc *)pSmpl->ctx;
  vCtx->vSeedCur = fGetRngSeed(vCtx->cSeed);
  vCtx->vRng.seed(vCtx->vSeedCur);
}

static struct mgs_sampler_i vMgsSamplerXtcI = {
    /* .name              = */ fMgsSamplerXtcName,
    /* .accept            = */ nullptr,
    /* .apply             = */ fMgsSampleXtcApply,
    /* .reset             = */ fMgsSamplerXtcReset,
    /* .clone             = */ fMgsSamplerXtcClone,
    /* .free              = */ fMgsSamplerXtcFree,
    /* .backend_init      = */ nullptr,
    /* .backend_accept    = */ nullptr,
    /* .backend_apply     = */ nullptr,
    /* .backend_set_input = */ nullptr,
};

struct mgs_sampler *mgs_sampler_init_xtc(float pP, float pT, size_t pMinKeep, uint32_t pSeed) {
  const bool cIsEmpty = (pP <= 0.0f || pT > 0.5f);

  if (cIsEmpty) {
    return fMgsSamplerInitEmpty("?xtc");
  }

  const auto cSeedCur = fGetRngSeed(pSeed);

  return mgs_sampler_init(
      /* .iface = */ &vMgsSamplerXtcI,
      /* .ctx   = */ new mgs_sampler_xtc{
          /* .cProbability = */ pP,
          /* .cThreshold   = */ pT,
          /* .cMinKeep     = */ pMinKeep,
          /* .cSeed        = */ pSeed,
          /* .vSeedCur     = */ cSeedCur,
          /* .vRng         = */ std::mt19937(cSeedCur),
      });
}

// mirostat

struct mgs_sampler_mirostat {
  const int32_t cNVocab;

  const uint32_t cSeed;
  uint32_t vSeedCur;

  const float cTau;
  const float cEta;

  const int32_t cM;

  float vMu;

  std::mt19937 vRng;
};

static const char *fMgsSamplerMirostatName(const struct mgs_sampler * /*smpl*/) { return "mirostat"; }

static void fMgsSamplerMirostatApply(struct mgs_sampler *pSmpl, mgs_token_data_array *pCurP) {
  auto *vCtx = (mgs_sampler_mirostat *)pSmpl->ctx;

  fMgsSamplerSoftmaxImpl(pCurP, true);

  // Estimate s_hat using the most probable m tokens
  float vSHat = 0.0;
  float vSumTiBi = 0.0;
  float vSumTiSq = 0.0;
  for (size_t vI = 0; vI < size_t(vCtx->cM - 1) && vI < pCurP->size - 1; ++vI) {
    float vTI = logf(float(vI + 2) / float(vI + 1));
    float vBI = logf(pCurP->data[vI].p / pCurP->data[vI + 1].p);
    vSumTiBi += vTI * vBI;
    vSumTiSq += vTI * vTI;
  }
  vSHat = vSumTiBi / vSumTiSq;

  // Compute k from the estimated s_hat and target surprise value
  float vEpsilonHat = vSHat - 1;
  float vK = powf((vEpsilonHat * powf(2, vCtx->vMu)) / (1 - powf(vCtx->cNVocab, -vEpsilonHat)), 1 / vSHat);

  fMgsSamplerTopKImpl(pCurP, std::max(int(vK), 1));

  fMgsSamplerSoftmaxImpl(pCurP, true);

  const int cIdx = fMgsSampleDist(pCurP, vCtx->vRng);

  pCurP->selected = cIdx;

  float vObservedSurprise = -log2f(pCurP->data[cIdx].p);
  float vE = vObservedSurprise - vCtx->cTau;

  // Update mu using the learning rate and error
  vCtx->vMu = vCtx->vMu - vCtx->cEta * vE;
}

static struct mgs_sampler *fMgsSamplerMirostatClone(const struct mgs_sampler *pSmpl) {
  const auto *vCtx = (const mgs_sampler_mirostat *)pSmpl->ctx;
  auto *vResult = mgs_sampler_init_mirostat(vCtx->cNVocab, vCtx->cSeed, vCtx->cTau, vCtx->cEta, vCtx->cM);

  // copy the state
  {
    auto *vResultCtx = (mgs_sampler_mirostat *)pSmpl->ctx;

    vResultCtx->vMu = vCtx->vMu;
    vResultCtx->vRng = vCtx->vRng;
  }

  return vResult;
}

static void fMgsSamplerMirostatReset(struct mgs_sampler *pSmpl) {
  auto *vCtx = (mgs_sampler_mirostat *)pSmpl->ctx;
  vCtx->vMu = 2.0f * vCtx->cTau;
  vCtx->vSeedCur = fGetRngSeed(vCtx->cSeed);
  vCtx->vRng.seed(vCtx->vSeedCur);
}

static void fMgsSamplerMirostatFree(struct mgs_sampler *pSmpl) { delete (mgs_sampler_mirostat *)pSmpl->ctx; }

static struct mgs_sampler_i vMgsSamplerMirostatI = {
    /* .name              = */ fMgsSamplerMirostatName,
    /* .accept            = */ nullptr,
    /* .apply             = */ fMgsSamplerMirostatApply,
    /* .reset             = */ fMgsSamplerMirostatReset,
    /* .clone             = */ fMgsSamplerMirostatClone,
    /* .free              = */ fMgsSamplerMirostatFree,
    /* .backend_init      = */ nullptr,
    /* .backend_accept    = */ nullptr,
    /* .backend_apply     = */ nullptr,
    /* .backend_set_input = */ nullptr,
};

struct mgs_sampler *mgs_sampler_init_mirostat(int32_t pNVocab, uint32_t pSeed, float pTau, float pEta, int32_t pM) {
  const auto cSeedCur = fGetRngSeed(pSeed);

  return mgs_sampler_init(
      /* .iface = */ &vMgsSamplerMirostatI,
      /* .ctx   = */ new mgs_sampler_mirostat{
          /* .cNVocab  = */ pNVocab,
          /* .cSeed    = */ pSeed,
          /* .vSeedCur = */ cSeedCur,
          /* .cTau     = */ pTau,
          /* .cEta     = */ pEta,
          /* .cM       = */ pM,
          /* .vMu      = */ 2.0f * pTau,
          /* .vRng     = */ std::mt19937(cSeedCur),
      });
}

// mirostat v2

struct mgs_sampler_mirostat_v2 {
  const uint32_t cSeed;
  uint32_t vSeedCur;

  const float cTau;
  const float cEta;

  float vMu;

  std::mt19937 vRng;
};

static const char *fMgsSamplerMirostatV2Name(const struct mgs_sampler * /*smpl*/) { return "mirostat-v2"; }

static void fMgsSamplerMirostatV2Apply(struct mgs_sampler *pSmpl, mgs_token_data_array *pCurP) {
  auto *vCtx = (mgs_sampler_mirostat_v2 *)pSmpl->ctx;

  fMgsSamplerSoftmaxImpl(pCurP, true);

  // Truncate the words with surprise values greater than mu
  pCurP->size = std::distance(
      pCurP->data, std::find_if(pCurP->data, pCurP->data + pCurP->size,
                                [&](const mgs_token_data &pCandidate) { return -log2f(pCandidate.p) > vCtx->vMu; }));

  if (pCurP->size == 0) {
    pCurP->size = 1;
  }

  // Normalize the probabilities of the remaining words
  fMgsSamplerSoftmaxImpl(pCurP, true);

  const int cIdx = fMgsSampleDist(pCurP, vCtx->vRng);

  pCurP->selected = cIdx;

  float vObservedSurprise = -log2f(pCurP->data[cIdx].p);
  float vE = vObservedSurprise - vCtx->cTau;

  // Update mu using the learning rate and error
  vCtx->vMu = vCtx->vMu - vCtx->cEta * vE;
}

static void fMgsSamplerMirostatV2Reset(struct mgs_sampler *pSmpl) {
  auto *vCtx = (mgs_sampler_mirostat_v2 *)pSmpl->ctx;
  vCtx->vMu = 2.0f * vCtx->cTau;
  vCtx->vSeedCur = fGetRngSeed(vCtx->cSeed);
  vCtx->vRng.seed(vCtx->vSeedCur);
}

static struct mgs_sampler *fMgsSamplerMirostatV2Clone(const struct mgs_sampler *pSmpl) {
  const auto *vCtx = (const mgs_sampler_mirostat_v2 *)pSmpl->ctx;

  auto *vResult = mgs_sampler_init_mirostat_v2(vCtx->cSeed, vCtx->cTau, vCtx->cEta);

  // copy the state
  {
    auto *vResultCtx = (mgs_sampler_mirostat_v2 *)vResult->ctx;

    vResultCtx->vMu = vCtx->vMu;
    vResultCtx->vRng = vCtx->vRng;
  }

  return vResult;
}

static void fMgsSamplerMirostatV2Free(struct mgs_sampler *pSmpl) { delete (mgs_sampler_mirostat_v2 *)pSmpl->ctx; }

static struct mgs_sampler_i vMgsSamplerMirostatV2I = {
    /* .name              = */ fMgsSamplerMirostatV2Name,
    /* .accept            = */ nullptr,
    /* .apply             = */ fMgsSamplerMirostatV2Apply,
    /* .reset             = */ fMgsSamplerMirostatV2Reset,
    /* .clone             = */ fMgsSamplerMirostatV2Clone,
    /* .free              = */ fMgsSamplerMirostatV2Free,
    /* .backend_init      = */ nullptr,
    /* .backend_accept    = */ nullptr,
    /* .backend_apply     = */ nullptr,
    /* .backend_set_input = */ nullptr,
};

struct mgs_sampler *mgs_sampler_init_mirostat_v2(uint32_t pSeed, float pTau, float pEta) {
  auto vSeedCur = fGetRngSeed(pSeed);
  return mgs_sampler_init(
      /* .iface = */ &vMgsSamplerMirostatV2I,
      /* .ctx   = */ new mgs_sampler_mirostat_v2{
          /* .cSeed    = */ pSeed,
          /* .vSeedCur = */ vSeedCur,
          /* .cTau     = */ pTau,
          /* .cEta     = */ pEta,
          /* .vMu      = */ 2.0f * pTau,
          /* .vRng     = */ std::mt19937(vSeedCur),
      });
}

// grammar

struct mgs_sampler_grammar {
  const struct mgs_vocab *vVocab;

  std::string vGrammarStr;
  std::string vGrammarRoot;

  struct mgs_grammar *vGrammar;
};

static const char *fMgsSamplerGrammarName(const struct mgs_sampler * /*smpl*/) { return "grammar"; }

static void fMgsSamplerGrammarAcceptImpl(struct mgs_sampler *pSmpl, mgs_token pToken) {
  auto *vCtx = (mgs_sampler_grammar *)pSmpl->ctx;
  if (vCtx->vGrammar) {
    mgs_grammar_accept_impl(*vCtx->vGrammar, pToken);
  }
}

static void fMgsSamplerGrammarApply(struct mgs_sampler *pSmpl, mgs_token_data_array *pCurP) {
  auto *vCtx = (mgs_sampler_grammar *)pSmpl->ctx;
  if (vCtx->vGrammar) {
    mgs_grammar_apply_impl(*vCtx->vGrammar, pCurP);
  }
}

// Fwd declare to break reset --> init_impl --> mgs_sampler_grammar_i --> reset cycle.
static struct mgs_sampler *fMgsSamplerInitGrammarImpl(const struct mgs_vocab *pVocab, const char *pGrammarStr,
                                                      const char *pGrammarRoot, bool pLazy,
                                                      const mgs_token *pTriggerTokens, size_t pNumTriggerTokens,
                                                      const char **pTriggerPatterns, size_t pNumTriggerPatterns);

static void fMgsSamplerGrammarReset(struct mgs_sampler *pSmpl) {
  auto *vCtx = (mgs_sampler_grammar *)pSmpl->ctx;
  if (!vCtx->vGrammar) {
    return;
  }

  std::vector<const char *> lTriggerPatternsC;
  lTriggerPatternsC.reserve(vCtx->vGrammar->vTriggerPatterns.size());
  for (auto &vTriggerPattern : vCtx->vGrammar->vTriggerPatterns) {
    lTriggerPatternsC.push_back(vTriggerPattern.vPattern.c_str());
  }

  auto *vGrammarNew =
      mgs_grammar_init_impl(vCtx->vGrammar->vVocab, vCtx->vGrammarStr.c_str(), vCtx->vGrammarRoot.c_str(),
                            vCtx->vGrammar->vLazy, lTriggerPatternsC.data(), lTriggerPatternsC.size(),
                            vCtx->vGrammar->vTriggerTokens.data(), vCtx->vGrammar->vTriggerTokens.size());

  mgs_grammar_free_impl(vCtx->vGrammar);
  vCtx->vGrammar = vGrammarNew;
}

static struct mgs_sampler *fMgsSamplerGrammarClone(const struct mgs_sampler *pSmpl) {
  const auto *vCtx = (const mgs_sampler_grammar *)pSmpl->ctx;

  auto *vResult = fMgsSamplerInitGrammarImpl(vCtx->vVocab, nullptr, nullptr, false, nullptr, 0, nullptr, 0);
  GGML_ASSERT(vResult);

  // copy the state
  {
    auto *vResultCtx = (mgs_sampler_grammar *)vResult->ctx;

    if (vCtx->vGrammar) {
      vResultCtx->vGrammarStr = vCtx->vGrammarStr;
      vResultCtx->vGrammarRoot = vCtx->vGrammarRoot;

      vResultCtx->vGrammar = mgs_grammar_clone_impl(*vCtx->vGrammar);
    }
  }

  return vResult;
}

static void fMgsSamplerGrammarFree(struct mgs_sampler *pSmpl) {
  const auto *vCtx = (mgs_sampler_grammar *)pSmpl->ctx;

  if (vCtx->vGrammar) {
    mgs_grammar_free_impl(vCtx->vGrammar);
  }

  delete vCtx;
}

static struct mgs_sampler_i vMgsSamplerGrammarI = {
    /* .name              = */ fMgsSamplerGrammarName,
    /* .accept            = */ fMgsSamplerGrammarAcceptImpl,
    /* .apply             = */ fMgsSamplerGrammarApply,
    /* .reset             = */ fMgsSamplerGrammarReset,
    /* .clone             = */ fMgsSamplerGrammarClone,
    /* .free              = */ fMgsSamplerGrammarFree,
    /* .backend_init      = */ nullptr,
    /* .backend_accept    = */ nullptr,
    /* .backend_apply     = */ nullptr,
    /* .backend_set_input = */ nullptr,
};

static struct mgs_sampler *fMgsSamplerInitGrammarImpl(const struct mgs_vocab *pVocab, const char *pGrammarStr,
                                                      const char *pGrammarRoot, bool pLazy,
                                                      const mgs_token *pTriggerTokens, size_t pNumTriggerTokens,
                                                      const char **pTriggerPatterns, size_t pNumTriggerPatterns) {
  auto *vCtx = new mgs_sampler_grammar;

  if (pGrammarStr != nullptr && pGrammarStr[0] != '\0') {
    mgs_grammar *vGrammar = mgs_grammar_init_impl(pVocab, pGrammarStr, pGrammarRoot, pLazy, pTriggerPatterns,
                                                  pNumTriggerPatterns, pTriggerTokens, pNumTriggerTokens);
    *vCtx = {
        /* .vVocab       = */ pVocab,
        /* .vGrammarStr  = */ pGrammarStr,
        /* .vGrammarRoot = */ pGrammarRoot,
        /* .vGrammar     = */ vGrammar,
    };
    if (!vCtx->vGrammar) {
      delete vCtx;
      return nullptr;
    }
  } else {
    *vCtx = {
        /* .vVocab       = */ pVocab,
        /* .vGrammarStr  = */ {},
        /* .vGrammarRoot = */ {},
        /* .vGrammar     = */ nullptr,
    };
  }

  return mgs_sampler_init(
      /* .iface = */ &vMgsSamplerGrammarI,
      /* .ctx   = */ vCtx);
}

struct mgs_sampler *mgs_sampler_init_grammar(const struct mgs_vocab *pVocab, const char *pGrammarStr,
                                             const char *pGrammarRoot) {
  return fMgsSamplerInitGrammarImpl(pVocab, pGrammarStr, pGrammarRoot, /* lazy= */ false, nullptr, 0, nullptr, 0);
}

struct mgs_sampler *mgs_sampler_init_grammar_lazy_patterns(const struct mgs_vocab *pVocab, const char *pGrammarStr,
                                                           const char *pGrammarRoot, const char **pTriggerPatterns,
                                                           size_t pNumTriggerPatterns, const mgs_token *pTriggerTokens,
                                                           size_t pNumTriggerTokens) {
  return fMgsSamplerInitGrammarImpl(pVocab, pGrammarStr, pGrammarRoot, /* lazy= */ true, pTriggerTokens,
                                    pNumTriggerTokens, pTriggerPatterns, pNumTriggerPatterns);
}

// penalties

struct mgs_sampler_penalties : public mgs_sampler_backend {
  const int32_t cNVocab;
  const int32_t cPenaltyLastN;
  const float cPenaltyRepeat;
  const float cPenaltyFreq;
  const float cPenaltyPresent;

  ring_buffer<mgs_token> vPrev;

  // a frequency map to count token occurrences
  std::unordered_map<mgs_token, int> dTokenCount;

  // backend graph inputs
  ggml_tensor *vInpTokenIds = nullptr;
  ggml_tensor *vInpCounts = nullptr;

  // backend helpers
  int32_t vNMax = 0;
  bool vHasCandidates = false;

  std::vector<int32_t> lHostTokenIds;
  std::vector<int32_t> lHostCounts;

  static bool mIsDisabled(int32_t pPenaltyLastN, float pPenaltyRepeat, float pPenaltyFreq, float pPenaltyPresent) {
    return pPenaltyLastN == 0 || (pPenaltyRepeat == 1.0f && pPenaltyFreq == 0.0f && pPenaltyPresent == 0.0f);
  }

  bool mIsDisabled() const { return mIsDisabled(cPenaltyLastN, cPenaltyRepeat, cPenaltyFreq, cPenaltyPresent); }

  mgs_sampler_penalties(int32_t pNVocab, int32_t pPenaltyLastN, float pPenaltyRepeat, float pPenaltyFreq,
                        float pPenaltyPresent)
      : mgs_sampler_backend("penalties"), cNVocab(pNVocab), cPenaltyLastN(pPenaltyLastN),
        cPenaltyRepeat(pPenaltyRepeat), cPenaltyFreq(pPenaltyFreq), cPenaltyPresent(pPenaltyPresent),
        vPrev(pPenaltyLastN) {}
};

static const char *fMgsSamplerPenaltiesName(const struct mgs_sampler *pSmpl) {
  auto *vCtx = (mgs_sampler_penalties *)pSmpl->ctx;
  return vCtx->mGetName();
}

static void fMgsSamplerPenaltiesAccept(struct mgs_sampler *pSmpl, mgs_token pToken) {
  auto *vCtx = (mgs_sampler_penalties *)pSmpl->ctx;
  if (vCtx->cPenaltyLastN == 0) {
    return;
  }

  vCtx->dTokenCount[pToken]++;

  // if the ring buffer is full, remove the oldest token
  if (vCtx->vPrev.size() >= (size_t)vCtx->cPenaltyLastN) {
    const auto cOld = vCtx->vPrev.mFront();

    vCtx->dTokenCount[cOld]--;
    if (vCtx->dTokenCount[cOld] == 0) {
      vCtx->dTokenCount.erase(cOld);
    }
  }

  vCtx->vPrev.mPushBack(pToken);

#if 0
    // sanity check
    std::unordered_map<mgs_token, int> dTmp;
    for (int vI = 0; vI < std::min<int>(vCtx->cPenaltyLastN, vCtx->vPrev.size()); ++vI) {
        dTmp[vCtx->vPrev.mRat(vI)]++;
    }

    assert(vCtx->dTokenCount == dTmp);
#endif
}

static void fMgsSamplerPenaltiesApply(struct mgs_sampler *pSmpl, mgs_token_data_array *pCurP) {
  auto *vCtx = (mgs_sampler_penalties *)pSmpl->ctx;

  if (vCtx->mIsDisabled()) {
    return;
  }

  // Apply frequency and presence penalties to the cur_p
  for (size_t vI = 0; vI < pCurP->size; ++vI) {
    const auto cTokenIter = vCtx->dTokenCount.find(pCurP->data[vI].id);
    if (cTokenIter == vCtx->dTokenCount.end()) {
      continue;
    }

    const int cCount = cTokenIter->second;

    assert(cCount > 0 && cCount <= vCtx->cPenaltyLastN);

    // The academic publication that described this technique actually just only divided, but that would cause tokens
    // with negative logits to become more likely, which is obviously wrong. This is common fix for this problem, which
    // is to multiply by the penalty instead of dividing.
    if (pCurP->data[vI].logit <= 0) {
      pCurP->data[vI].logit *= vCtx->cPenaltyRepeat;
    } else {
      pCurP->data[vI].logit /= vCtx->cPenaltyRepeat;
    }

    pCurP->data[vI].logit -= float(cCount) * vCtx->cPenaltyFreq + float(cCount > 0) * vCtx->cPenaltyPresent;
  }

  pCurP->sorted = false;
}

static void fMgsSamplerPenaltiesReset(struct mgs_sampler *pSmpl) {
  auto *vCtx = (mgs_sampler_penalties *)pSmpl->ctx;
  vCtx->vPrev.mClear();
  vCtx->dTokenCount.clear();
}

static struct mgs_sampler *fMgsSamplerPenaltiesClone(const struct mgs_sampler *pSmpl) {
  const auto *vCtx = (const mgs_sampler_penalties *)pSmpl->ctx;
  auto *vResult = mgs_sampler_init_penalties(vCtx->cNVocab, vCtx->cPenaltyLastN, vCtx->cPenaltyRepeat,
                                             vCtx->cPenaltyFreq, vCtx->cPenaltyPresent);

  // copy the state
  {
    auto *vResultCtx = (mgs_sampler_penalties *)vResult->ctx;

    vResultCtx->vPrev = vCtx->vPrev;
    vResultCtx->dTokenCount = vCtx->dTokenCount;
  }

  return vResult;
}

static void fMgsSamplerPenaltiesFree(struct mgs_sampler *pSmpl) { delete (mgs_sampler_penalties *)pSmpl->ctx; }

static bool fMgsSamplerPenaltiesBackendInit(struct mgs_sampler *pSmpl, ggml_backend_buffer_type_t pBuft) {
  auto *vSctx = (mgs_sampler_penalties *)pSmpl->ctx;

  const bool cRes = fMgsSamplerBackendSupport(pSmpl, pBuft);

  vSctx->mInit(cRes);

  return cRes;
}

static void fMgsSamplerPenaltiesBackendApply(struct mgs_sampler *pSmpl, struct ggml_context *pCtx,
                                             struct ggml_cgraph *pGf, struct mgs_sampler_data *pData) {
  GGML_UNUSED(pGf);

  auto *vSctx = (mgs_sampler_penalties *)pSmpl->ctx;

  if (vSctx->mIsDisabled()) {
    return;
  }

  GGML_ASSERT(vSctx->cNVocab > 0);

  vSctx->vHasCandidates = pData->candidates != nullptr;
  vSctx->vNMax = std::min(vSctx->cPenaltyLastN, vSctx->cNVocab);

  vSctx->vInpTokenIds = ggml_new_tensor_1d(pCtx, GGML_TYPE_I32, vSctx->vNMax);
  ggml_set_name(vSctx->vInpTokenIds, "penalties_token_ids");
  ggml_set_input(vSctx->vInpTokenIds);

  vSctx->vInpCounts = ggml_new_tensor_1d(pCtx, GGML_TYPE_I32, vSctx->vNMax);
  ggml_set_name(vSctx->vInpCounts, "penalties_counts");
  ggml_set_input(vSctx->vInpCounts);

  if ((int32_t)vSctx->lHostTokenIds.size() != vSctx->vNMax) {
    vSctx->lHostTokenIds.assign(vSctx->vNMax, 0);
    vSctx->lHostCounts.assign(vSctx->vNMax, 0);
  }

  // flatten
  ggml_tensor *vLogits = ggml_reshape_1d(pCtx, pData->logits, ggml_nelements(pData->logits));
  ggml_tensor *vGathered = vLogits;
  ggml_tensor *vCountsF32 = ggml_cast(pCtx, vSctx->vInpCounts, GGML_TYPE_F32);

  if (vSctx->vHasCandidates) {
    ggml_tensor *vCandidates = ggml_reshape_1d(pCtx, pData->candidates, ggml_nelements(pData->candidates));
    const int64_t cNCandidates = vCandidates->ne[0];
    GGML_ASSERT(cNCandidates == ggml_nelements(vLogits));

    ggml_tensor *vCountsRows = ggml_fill(pCtx, ggml_new_tensor_2d(pCtx, GGML_TYPE_F32, 1, vSctx->cNVocab), 0.0f);
    ggml_tensor *vScatterRows = ggml_reshape_2d(pCtx, vCountsF32, 1, vSctx->vNMax);
    vCountsRows = ggml_set_rows(pCtx, vCountsRows, vScatterRows, vSctx->vInpTokenIds);
    vCountsF32 = ggml_get_rows(pCtx, vCountsRows, vCandidates);
    vCountsF32 = ggml_reshape_1d(pCtx, vCountsF32, cNCandidates);
  } else {
    ggml_tensor *vLogitsRows = ggml_reshape_2d(pCtx, vLogits, 1, ggml_nelements(vLogits));
    vGathered = ggml_get_rows(pCtx, vLogitsRows, vSctx->vInpTokenIds);
    vGathered = ggml_reshape_1d(pCtx, vGathered, vSctx->vNMax);
  }

  ggml_tensor *vActiveMask = ggml_step(pCtx, vCountsF32);
  ggml_tensor *vInactiveMask = ggml_sub(pCtx, ggml_fill(pCtx, vActiveMask, 1.0f), vActiveMask);

  ggml_tensor *vPenalized = vGathered;

  if (vSctx->cPenaltyRepeat != 1.0f) {
    ggml_tensor *vPosMask = ggml_step(pCtx, vPenalized);
    ggml_tensor *vNegMask = ggml_sub(pCtx, ggml_fill(pCtx, vPosMask, 1.0f), vPosMask);

    ggml_tensor *vPosScale = ggml_scale(pCtx, vPosMask, 1.0f / vSctx->cPenaltyRepeat);
    ggml_tensor *vNegScale = ggml_scale(pCtx, vNegMask, vSctx->cPenaltyRepeat);
    ggml_tensor *vRepeatScale = ggml_add(pCtx, vPosScale, vNegScale);

    // scale inactive entries with 1 to avoid -INF * 0 = NaN for values masked by top-p
    vRepeatScale = ggml_mul(pCtx, vRepeatScale, vActiveMask);
    vRepeatScale = ggml_add(pCtx, vRepeatScale, vInactiveMask);
    vPenalized = ggml_mul(pCtx, vGathered, vRepeatScale);
  }

  if (vSctx->cPenaltyFreq != 0.0f) {
    ggml_tensor *vPenaltyFreq = ggml_scale(pCtx, vCountsF32, vSctx->cPenaltyFreq);
    vPenalized = ggml_sub(pCtx, vPenalized, vPenaltyFreq);
  }

  if (vSctx->cPenaltyPresent != 0.0f) {
    ggml_tensor *vPenaltyPresent = ggml_scale(pCtx, vActiveMask, vSctx->cPenaltyPresent);
    vPenalized = ggml_sub(pCtx, vPenalized, vPenaltyPresent);
  }

  if (vSctx->vHasCandidates) {
    pData->logits = vPenalized;
  } else {
    ggml_tensor *vLogitsRows = ggml_reshape_2d(pCtx, vLogits, 1, ggml_nelements(vLogits));
    ggml_tensor *vScatterRows = ggml_reshape_2d(pCtx, vPenalized, 1, vSctx->vNMax);
    vLogitsRows = ggml_set_rows(pCtx, vLogitsRows, vScatterRows, vSctx->vInpTokenIds);
    pData->logits = ggml_reshape_1d(pCtx, vLogitsRows, ggml_nelements(vLogits));
  }
}

static void fMgsSamplerPenaltiesBackendSetInput(struct mgs_sampler *pSmpl) {
  auto *vSctx = (mgs_sampler_penalties *)pSmpl->ctx;

  if (!vSctx->vInpTokenIds || !vSctx->vInpCounts || vSctx->vNMax <= 0 || vSctx->cNVocab <= 0) {
    return;
  }

  if (vSctx->mIsDisabled()) {
    return;
  }

  // fill active entries from the map
  int32_t vNActive = 0;

  for (const auto &vIt : vSctx->dTokenCount) {
    GGML_ASSERT(vNActive < vSctx->vNMax);
    vSctx->lHostTokenIds[vNActive] = vIt.first;
    vSctx->lHostCounts[vNActive] = vIt.second;
    ++vNActive;
  }

  // Sorting is required because backend_apply uses ggml_set_rows (a scatter-back operation)
  std::vector<std::pair<int32_t, int32_t>> lEntries;
  lEntries.reserve(vNActive);
  for (int32_t vI = 0; vI < vNActive; ++vI) {
    lEntries.emplace_back(vSctx->lHostTokenIds[vI], vSctx->lHostCounts[vI]);
  }
  std::sort(lEntries.begin(), lEntries.end(), [](const auto &pA, const auto &pB) { return pA.first < pB.first; });
  for (int32_t vI = 0; vI < vNActive; ++vI) {
    vSctx->lHostTokenIds[vI] = lEntries[vI].first;
    vSctx->lHostCounts[vI] = lEntries[vI].second;
  }

  // Padding: Finds a filler token id that is not present in token_count.
  // Use it to do padding for the arrays, it avoids resizing every time.
  // The arrays must always have exactly n_max entries (the GPU tensor is a fixed size).
  int32_t vFiller = 0;
  if (vNActive < vSctx->vNMax) {
    while (vSctx->dTokenCount.find(vFiller) != vSctx->dTokenCount.end()) {
      ++vFiller;
    }
    GGML_ASSERT(vFiller < vSctx->cNVocab);
  }

  // Fill the rest of the arrays with the filler token id and count 0.
  // Inactive slots are padded with a unique dummy token ID (count = 0).
  // The uniqueness matters because ggml_set_rows with duplicate indices can produce non-deterministic or incorrect
  // results. Using a filler token with count 0 that isn't in the active set is safe, because the active_mask step in
  // backend_apply filters them out via ggml_step(counts_f32)
  for (int32_t vI = vNActive; vI < vSctx->vNMax; ++vI) {
    vSctx->lHostTokenIds[vI] = vFiller;
    vSctx->lHostCounts[vI] = 0;
  }

  ggml_backend_tensor_set(vSctx->vInpTokenIds, vSctx->lHostTokenIds.data(), 0, vSctx->vNMax * sizeof(int32_t));
  ggml_backend_tensor_set(vSctx->vInpCounts, vSctx->lHostCounts.data(), 0, vSctx->vNMax * sizeof(int32_t));
}

static struct mgs_sampler_i vMgsSamplerPenaltiesI = {
    /* .name              = */ fMgsSamplerPenaltiesName,
    /* .accept            = */ fMgsSamplerPenaltiesAccept,
    /* .apply             = */ fMgsSamplerPenaltiesApply,
    /* .reset             = */ fMgsSamplerPenaltiesReset,
    /* .clone             = */ fMgsSamplerPenaltiesClone,
    /* .free              = */ fMgsSamplerPenaltiesFree,
    /* .backend_init      = */ fMgsSamplerPenaltiesBackendInit,
    /* .backend_accept    = */ nullptr,
    /* .backend_apply     = */ fMgsSamplerPenaltiesBackendApply,
    /* .backend_set_input = */ fMgsSamplerPenaltiesBackendSetInput,
};

struct mgs_sampler *mgs_sampler_init_penalties(int32_t pNVocab, int32_t pPenaltyLastN, float pPenaltyRepeat,
                                               float pPenaltyFreq, float pPenaltyPresent) {
  pPenaltyLastN = std::max(pPenaltyLastN, 0);

  if (mgs_sampler_penalties::mIsDisabled(pPenaltyLastN, pPenaltyRepeat, pPenaltyFreq, pPenaltyPresent)) {
    return fMgsSamplerInitEmpty("?penalties");
  }

  return mgs_sampler_init(
      /* .iface = */ &vMgsSamplerPenaltiesI,
      /* .ctx   = */ new mgs_sampler_penalties(pNVocab, pPenaltyLastN, pPenaltyRepeat, pPenaltyFreq, pPenaltyPresent));
}

// top-n-sigma

struct mgs_sampler_top_n_sigma {
  const float cN;
};

static const char *fMgsSamplerTopNSigmaName(const struct mgs_sampler * /*smpl*/) { return "top-n-sigma"; }

static void fMgsSamplerTopNSigmaApply(struct mgs_sampler *pSmpl, mgs_token_data_array *pCurP) {
  auto *vCtx = (mgs_sampler_top_n_sigma *)pSmpl->ctx;

  if (vCtx->cN <= 0.0f || pCurP->size <= 1) {
    return;
  }

  // find max logit and calculate mean
  float vMax = pCurP->data[0].logit;
  float vLogitsSum = 0;
  size_t vValidCount = 0;
  for (size_t vI = 0; vI < pCurP->size; ++vI) {
    // Only count non-negative infinity values
    if (pCurP->data[vI].logit != -INFINITY) {
      vMax = std::max(vMax, pCurP->data[vI].logit);
      vLogitsSum += pCurP->data[vI].logit;
      vValidCount++;
    }
  }
  float vMean = vValidCount > 0 ? vLogitsSum / vValidCount : 0;

  // calculate standard deviation
  float vAcc = 0;
  for (size_t vI = 0; vI < pCurP->size; ++vI) {
    // Skip -infinity in std calculation
    if (pCurP->data[vI].logit != -INFINITY) {
      vAcc += pow(pCurP->data[vI].logit - vMean, 2);
    }
  }
  float vStd = vValidCount > 0 ? sqrt(vAcc / vValidCount) : 0;

  // apply mask
  for (size_t vI = 0; vI < pCurP->size; ++vI) {
    if (pCurP->data[vI].logit < vMax - (vCtx->cN * vStd)) {
      pCurP->data[vI].logit = -INFINITY;
    }
  }
}

static struct mgs_sampler *fMgsSamplerTopNSigmaClone(const struct mgs_sampler *pSmpl) {
  const auto *vCtx = (const mgs_sampler_top_n_sigma *)pSmpl->ctx;
  return mgs_sampler_init_top_n_sigma(vCtx->cN);
}

static void fMgsSamplerTopNSigmaFree(struct mgs_sampler *pSmpl) { delete (mgs_sampler_top_n_sigma *)pSmpl->ctx; }

static struct mgs_sampler_i vMgsSamplerTopNSigmaI = {
    /* .name              = */ fMgsSamplerTopNSigmaName,
    /* .accept            = */ nullptr,
    /* .apply             = */ fMgsSamplerTopNSigmaApply,
    /* .reset             = */ nullptr,
    /* .clone             = */ fMgsSamplerTopNSigmaClone,
    /* .free              = */ fMgsSamplerTopNSigmaFree,
    /* .backend_init      = */ nullptr,
    /* .backend_accept    = */ nullptr,
    /* .backend_apply     = */ nullptr,
    /* .backend_set_input = */ nullptr,
};

struct mgs_sampler *mgs_sampler_init_top_n_sigma(float pN) {
  const bool cIsEmpty = (pN <= 0.0f);

  if (cIsEmpty) {
    return fMgsSamplerInitEmpty("?top-n-sigma");
  }

  return mgs_sampler_init(
      /* .iface = */ &vMgsSamplerTopNSigmaI,
      /* .ctx   = */ new mgs_sampler_top_n_sigma{
          /* .cN = */ pN,
      });
}

// DRY

struct mgs_sampler_dry {
  int32_t vTotalContextSize;

  const float cDryMultiplier;
  const float cDryBase;
  const int32_t cDryAllowedLength;
  const int32_t cDryPenaltyLastN;

  std::unordered_multimap<mgs_token, std::vector<mgs_token>> dDryProcessedBreakers;
  std::vector<int> lDryRepeatCount;
  std::unordered_map<mgs_token, int> dDryMaxTokenRepeat;
  ring_buffer<mgs_token> vLastTokens;
};

// Ported from Koboldcpp, original PR: https://github.com/LostRuins/koboldcpp/pull/982 (Original author: pi6am)
static void fGetOverlappingTokenSequences(const mgs_vocab &pVocab, const std::string &pStr,
                                          std::unordered_multimap<mgs_token, std::vector<mgs_token>> &pTokenSequences,
                                          int pMaxTailLen = -1) {
  for (mgs_token vTokenId = 0; vTokenId < (mgs_token)pVocab.mNTokens(); vTokenId++) {
    std::string vWord = pVocab.mDetokenize({vTokenId}, true);
    if (vWord.find(pStr) != std::string::npos) {
      pTokenSequences.emplace(vTokenId, std::vector<mgs_token>());
    } else {
      size_t vWordLen = vWord.size();
      size_t vStrLen = pStr.size();
      size_t vPos = -1;
      while ((vPos = vWord.find(pStr[0], vPos + 1)) != std::string::npos) {
        bool vMatch = true;
        size_t vI;
        for (vI = 1; vI < vStrLen && vI + vPos < vWordLen; ++vI) {
          if (vWord[vPos + vI] != pStr[vI]) {
            vMatch = false;
            break;
          }
        }
        if (vMatch) {
          std::vector<mgs_token> lTokenization = pVocab.mTokenize(pStr.substr(vI), false, false);
          if (pMaxTailLen >= 0 && lTokenization.size() > (size_t)pMaxTailLen) {
            lTokenization.resize(pMaxTailLen);
          }

          // Ensure we don't already have a duplicate matching tokenization
          auto vIts = pTokenSequences.equal_range(vTokenId);
          bool vFound = false;
          for (auto vIt = vIts.first; vIt != vIts.second; ++vIt) {
            if (lTokenization == vIt->second) {
              vFound = true;
              break;
            }
          }
          if (!vFound) {
            pTokenSequences.emplace(vTokenId, lTokenization);
          }
        }
      }
    }
  }
}

static const char *fMgsSamplerDryName(const struct mgs_sampler * /*smpl*/) { return "dry"; }

static void fMgsSamplerDryAccept(struct mgs_sampler *pSmpl, mgs_token pToken) {
  auto *vCtx = (mgs_sampler_dry *)pSmpl->ctx;
  if (vCtx->cDryMultiplier == 0.0f || vCtx->cDryBase < 1.0f || vCtx->cDryPenaltyLastN == 0) {
    return;
  }

  vCtx->vLastTokens.mPushBack(pToken);
}

// Ported from Koboldcpp, original PR: https://github.com/LostRuins/koboldcpp/pull/982 (Original author: pi6am)
static void fMgsSamplerDryApply(struct mgs_sampler *pSmpl, mgs_token_data_array *pCurP) {
  auto *vCtx = (mgs_sampler_dry *)pSmpl->ctx;

  if (vCtx->cDryMultiplier == 0.0f || vCtx->cDryBase < 1.0f || vCtx->cDryPenaltyLastN == 0) {
    return;
  }

  int32_t vEffectiveDryPenaltyLastN =
      (vCtx->cDryPenaltyLastN == -1) ? vCtx->vTotalContextSize : std::max(vCtx->cDryPenaltyLastN, 0);
  int vLastNRepeat =
      std::min(std::min((int)vCtx->vLastTokens.size(), vEffectiveDryPenaltyLastN), vCtx->vTotalContextSize);

  if (vLastNRepeat <= vCtx->cDryAllowedLength) {
    return;
  }

  vCtx->lDryRepeatCount.assign(vLastNRepeat, 0);
  vCtx->dDryMaxTokenRepeat.clear();

  // Step 1: Look for restart sequences to limit the maximum repetition length.
  // Work backwards through the context looking for any token that begins a restart sequence.
  //
  // The collection `restart_sequences` is a mapping from a "head" token to all "tail"
  // sequences that together comprise a restart sequence. This allows us to quickly check
  // whether each token is the head of a complete sequence. Most restart sequences are actually
  // a single token, and for these the "tail" is an empty vector.
  //
  // If the token is a "head", test all restart sequences that begin with this token
  // (there will often only be one sequence for each token, but if sequences like 'aaaq1' and
  // 'aaa1' are used as restart strings, both could start with 'aaa' when tokenized). The
  // longest matching sequence (if any) is used to limit the maximum repetition length.
  //
  // Note that in the case case of a short sequence contained in a longer one, this might fail to
  // find the smallest value for `vRepLimit`. For example, if 'amniotic' and 'ni' are both used as
  // restart sequences, 'ni' will be found first, and since it's shorter it will fail to suppress
  // 'otic'. This is a minor issue since fully contained restart sequences are likely to be rare.
  //
  // This is theoretically worst-case O(N^2) for arbitrary restart sequences, which is why we
  // have already clamped the maximum tail sequence length when generating `restart_sequences`.
  // With clamping, this scan is O(N) in the context length.

  int vRepLimit = vLastNRepeat;
  for (int vI = 0; vI < vLastNRepeat; ++vI) {
    mgs_token vToken = vCtx->vLastTokens.mRat(vI);
    auto vIts = vCtx->dDryProcessedBreakers.equal_range(vToken);
    if (vIts.first == vCtx->dDryProcessedBreakers.end()) {
      continue;
    }
    int vLongestMatch = -1;
    for (auto vIt = vIts.first; vIt != vIts.second; ++vIt) {
      // Note that (*it) does not contain the head character, so seq_len will be
      // the restart sequence length minus 1.
      // In the common case of a single-token restart sequence, (*it) will be empty
      // and we will trivially match.
      int vSeqLen = (int)vIt->second.size();
      if (vSeqLen > vLongestMatch && vSeqLen <= (int)vI) {
        bool vMatch = true;
        for (int vOffset = 0; vOffset < vSeqLen; ++vOffset) {
          // The -1 when indexing `vLastTokens` is because we already matched the head.
          if (vIt->second[vOffset] != vCtx->vLastTokens.mRat(vI - vOffset - 1)) {
            vMatch = false;
            break;
          }
        }
        if (vMatch) {
          vLongestMatch = vSeqLen;
        }
      }
    }
    if (vLongestMatch >= 0) {
      // We found a restart sequence starting `vI` tokens from the end and continuing for
      // `vLongestMatch` tokens.
      vRepLimit = vI - vLongestMatch;
      break;
    }
  }
  if (vRepLimit < vCtx->cDryAllowedLength) {
    return;
  }

  // Step 2: Iterate in reverse over the last N tokens of the context, using the "Z-algorithm" (in
  // the reverse direction) to efficiently compute the positions and lengths of suffixes appearing
  // elsewhere in the context. We limit the suffix length to `vRepLimit` to respect restart sequences.
  //
  // This algorithm is not currently documented on Wikipedia, but there is a clear description here:
  // https://ivanyu.me/blog/2014/10/15/z-algorithm/
  //
  // The code below is adapted from the public domain implementation by the same author here:
  // https://github.com/ivanyu/string-algorithms/blob/master/z_algorithm.py
  //
  // Example:
  // Last N tokens: a b c c b c y a b c
  // Repeat counts: 0 0 3 1 0 2 0 0 0 0
  //                    ^
  //   This `3` means that the last three tokens of the context (a b c) also appear here.
  //
  // This step is worst case O(N) since the Z-algorithm is linear, despite the appearance of nested
  // for/while loops. This can be seen by observing that the `lt` and `rt` bounds are set after each
  // repeated suffix is detected (i.e. after each while loop when n > 0). These bound variables
  // ensure that the inner while loops only examine each token in the context once as the outer
  // for loop iterates over the context.

  {
    const int cLast = vLastNRepeat - 1;

    int vRt = 0;
    int vLt = 0;

    for (int vK = 1; vK < vLastNRepeat; ++vK) {
      if (vK > vRt) {
        // If k is outside the current Z-box, do naive computation.
        int vN = 0;
        while (vN + vK < vLastNRepeat && vCtx->vLastTokens.mRat(vN) == vCtx->vLastTokens.mRat(vN + vK)) {
          ++vN;
        }
        vCtx->lDryRepeatCount[cLast - vK] = std::min(vN, vRepLimit);
        if (vN > 0) {
          vLt = vK;
          vRt = vK + vN - 1;
        }
      } else {
        // If k is inside the current Z-box, consider two cases.

        int vP = vK - vLt; // Pair index.
        int vRightPartLen = vRt - vK + 1;

        if (vCtx->lDryRepeatCount[cLast - vP] < vRightPartLen) {
          int vN = std::min(vCtx->lDryRepeatCount[cLast - vP], vRepLimit);
          vCtx->lDryRepeatCount[cLast - vK] = vN;
        } else {
          int vI = vRt + 1;
          while (vI < vLastNRepeat && vCtx->vLastTokens.mRat(vI) == vCtx->vLastTokens.mRat(vI - vK)) {
            vI += 1;
          }

          int vN = std::min(vI - vK, vRepLimit);
          vCtx->lDryRepeatCount[cLast - vK] = vN;
          vLt = vK;
          vRt = vI - 1;
        }
      }
    }
  }

  // Step 3: Iterate over dry_repeat_count and last_tokens, examining the maximum repeat length
  // that would be generated by emitting each new token that would extend a sequence.
  //
  // Following the same example as above:
  // Last N tokens: a b c c b c y a b c
  // Repeat counts: 0 0 3 1 0 2 0 0 0 0
  //
  // For each non-zero, look ahead one token. This token, if emitted, would extend the repetition.
  // c: 3 -> 4 (from `a b c` to `a b c c`)
  // b: 1 -> 2 (from `c` to `c b`)
  // y: 2 -> 3 (from `b c` to `b c y`)

  for (int vI = 0; vI < vLastNRepeat - 1; ++vI) {
    int vRepeatLen = vCtx->lDryRepeatCount[vI];
    if (vRepeatLen >= vCtx->cDryAllowedLength) {
      // This token ends a repeat, so the next token would continue one.
      // By convention, the value of `vRepeatLen` only includes the tokens currently
      // in the context, not the new token that would be added.
      mgs_token vToken = vCtx->vLastTokens.mRat(vLastNRepeat - 2 - vI);
      // Track the maximum sequence ending in this token.
      const auto &vIt = vCtx->dDryMaxTokenRepeat.find(vToken);
      if (vIt == vCtx->dDryMaxTokenRepeat.end() || vIt->second < vRepeatLen) {
        vCtx->dDryMaxTokenRepeat[vToken] = vRepeatLen;
      }
    }
  }

  // Step 4: Apply logit penalties based on the maximum repeat length for relevant tokens.

  // Prevent floating point overflow in `pow(penalty_base, exponent)` by clamping to `vMaxExponent`.
  // Compute it from `penalty_base` and the approximate log of `std::numeric_limits<float>::max()`
  const float cFloatMaxLog = 88.7228391f;
  int vMaxExponent = 0;
  if (vCtx->cDryBase > 1.000001f) {
    vMaxExponent = cFloatMaxLog / std::log(vCtx->cDryBase);
  }

  for (size_t vI = 0; vI < pCurP->size; ++vI) {
    const auto &vAfKvp = vCtx->dDryMaxTokenRepeat.find(pCurP->data[vI].id);
    if (vAfKvp != vCtx->dDryMaxTokenRepeat.end()) {
      // Check all sequence breakers starting with this token
      auto vRange = vCtx->dDryProcessedBreakers.equal_range(pCurP->data[vI].id);
      bool vIsSingleTokenBreaker = false;

      for (auto vIt = vRange.first; vIt != vRange.second; ++vIt) {
        if (vIt->second.empty()) {
          vIsSingleTokenBreaker = true;
          break;
        }
      }

      // Apply penalty only if it's not a single-token sequence breaker
      if (!vIsSingleTokenBreaker) {
        int vRepeatExp = vAfKvp->second - vCtx->cDryAllowedLength;
        if (vMaxExponent > 0 && vRepeatExp > vMaxExponent) {
          vRepeatExp = vMaxExponent;
        }
        float vPenalty = vCtx->cDryMultiplier * std::pow(vCtx->cDryBase, vRepeatExp);
        pCurP->data[vI].logit -= vPenalty;
      }
    }
  }

  pCurP->sorted = false;
}

static void fMgsSamplerDryReset(struct mgs_sampler *pSmpl) {
  auto *vCtx = (mgs_sampler_dry *)pSmpl->ctx;
  vCtx->vLastTokens.mClear();
  vCtx->lDryRepeatCount.clear();
  vCtx->dDryMaxTokenRepeat.clear();
}

static struct mgs_sampler *fMgsSamplerDryClone(const struct mgs_sampler *pSmpl) {
  const auto *vCtx = (mgs_sampler_dry *)pSmpl->ctx;

  mgs_vocab vDummyVocab;

  // dummy vocab is passed because it is only needed for raw sequence breaker processing, which we have already done and
  // will simply be copying
  auto *vResult = mgs_sampler_init_dry(&vDummyVocab, vCtx->vTotalContextSize, vCtx->cDryMultiplier, vCtx->cDryBase,
                                       vCtx->cDryAllowedLength, vCtx->cDryPenaltyLastN, NULL, 0);

  // Copy the state, including the processed breakers
  {
    auto *vResultCtx = (mgs_sampler_dry *)vResult->ctx;
    vResultCtx->dDryProcessedBreakers = vCtx->dDryProcessedBreakers;
    vResultCtx->lDryRepeatCount = vCtx->lDryRepeatCount;
    vResultCtx->dDryMaxTokenRepeat = vCtx->dDryMaxTokenRepeat;
    vResultCtx->vLastTokens = vCtx->vLastTokens;
  }

  return vResult;
}

static void fMgsSamplerDryFree(struct mgs_sampler *pSmpl) { delete (mgs_sampler_dry *)pSmpl->ctx; }

static struct mgs_sampler_i vMgsSamplerDryI = {
    /* .name              = */ fMgsSamplerDryName,
    /* .accept            = */ fMgsSamplerDryAccept,
    /* .apply             = */ fMgsSamplerDryApply,
    /* .reset             = */ fMgsSamplerDryReset,
    /* .clone             = */ fMgsSamplerDryClone,
    /* .free              = */ fMgsSamplerDryFree,
    /* .backend_init      = */ nullptr,
    /* .backend_accept    = */ nullptr,
    /* .backend_apply     = */ nullptr,
    /* .backend_set_input = */ nullptr,
};

struct mgs_sampler *mgs_sampler_init_dry(const struct mgs_vocab *pVocab, int32_t pNCtxTrain, float pDryMultiplier,
                                         float pDryBase, int32_t pDryAllowedLength, int32_t pDryPenaltyLastN,
                                         const char **pSeqBreakers, size_t pNumBreakers) {
  int32_t vEffectiveDryPenaltyLastN = (pDryPenaltyLastN == -1) ? pNCtxTrain : std::max(pDryPenaltyLastN, 0);
  std::unordered_multimap<mgs_token, std::vector<mgs_token>> dProcessedBreakers;
  const int cMaxCharLen = 40;
  const int cMaxSeqLen = 20;

  const bool cDryEnabled = (pDryMultiplier != 0.0f && pDryBase >= 1.0f && pDryPenaltyLastN != 0);

  if (!cDryEnabled) {
    return fMgsSamplerInitEmpty("?dry");
  }

  if (cDryEnabled && pSeqBreakers != nullptr && pNumBreakers > 0) {
    // Process sequence breakers
    for (size_t vI = 0; vI < pNumBreakers; ++vI) {
      if (pSeqBreakers[vI] == nullptr || std::strlen(pSeqBreakers[vI]) == 0) {
        MGS_LOG_WARN("skipping null or empty DRY sequence breaker at index %zu\n", vI);
        continue;
      }

      std::string vSequenceBreak(pSeqBreakers[vI]);
      if (vSequenceBreak.empty()) {
        MGS_LOG_WARN("skipping empty DRY sequence breaker\n");
        continue;
      }

      if (vSequenceBreak.size() > cMaxCharLen) {
        MGS_LOG_WARN("truncating DRY sequence breaker to %d characters\n", cMaxCharLen);
        vSequenceBreak.resize(cMaxCharLen);
      }

      fGetOverlappingTokenSequences(*pVocab, vSequenceBreak, dProcessedBreakers, cMaxSeqLen);
    }
  }

  return mgs_sampler_init(
      /* .iface = */ &vMgsSamplerDryI,
      /* .ctx   = */ new mgs_sampler_dry{
          /* .vTotalContextSize     = */ pNCtxTrain,
          /* .cDryMultiplier        = */ pDryMultiplier,
          /* .cDryBase              = */ pDryBase,
          /* .cDryAllowedLength     = */ pDryAllowedLength,
          /* .cDryPenaltyLastN      = */ pDryPenaltyLastN,
          /* .dDryProcessedBreakers = */ std::move(dProcessedBreakers),
          /* .lDryRepeatCount       = */
          cDryEnabled ? std::vector<int>(vEffectiveDryPenaltyLastN, 0) : std::vector<int>{},
          /* .dDryMaxTokenRepeat    = */ {},
          /* .vLastTokens           = */
          cDryEnabled ? ring_buffer<mgs_token>(vEffectiveDryPenaltyLastN) : ring_buffer<mgs_token>(0),
      });
}

// wrapper for test-sampling.cpp
struct mgs_sampler *mgs_sampler_init_dry_testing(int32_t pContextSize, float pDryMultiplier, float pDryBase,
                                                 int32_t pDryAllowedLength, int32_t pDryPenaltyLastN,
                                                 const std::vector<std::vector<mgs_token>> &pSeqBreakers) {
  mgs_vocab vDummyVocab;
  auto *vResult = mgs_sampler_init_dry(&vDummyVocab, pContextSize, pDryMultiplier, pDryBase, pDryAllowedLength,
                                       pDryPenaltyLastN, NULL, 0);
  auto *vCtx = (mgs_sampler_dry *)vResult->ctx;

  // Process the token-based sequence breakers
  vCtx->dDryProcessedBreakers.clear();
  if (pSeqBreakers.empty()) {
    MGS_LOG_WARN("empty DRY sequence breakers list in mgs_sampler_init_dry_testing\n");
  } else {
    for (const auto &vBreaker : pSeqBreakers) {
      if (vBreaker.empty()) {
        MGS_LOG_WARN("skipping DRY empty sequence breaker\n");
        continue;
      }
      mgs_token vHeadToken = vBreaker[0];
      std::vector<mgs_token> lTailTokens(vBreaker.begin() + 1, vBreaker.end());
      vCtx->dDryProcessedBreakers.emplace(vHeadToken, std::move(lTailTokens));
    }

    if (vCtx->dDryProcessedBreakers.empty()) {
      MGS_LOG_WARN("no valid DRY sequence breakers processed in mgs_sampler_init_dry_testing\n");
    }
  }

  return vResult;
}

// adaptive-p sampler state
//
// maintains an exponential moving average of the *ORIGINAL* probabilities
// of selected tokens, used to compute an adapted target at each sampling step.
//
// see mgs.h for a full description of the sampler
//
struct mgs_sampler_adaptive_p {
  const float cTarget;               // target probability (0.0 - 1.0; negative = disabled)
  const float cDecay;                // EMA decay; history ~= 1/(1-decay) tokens (0.0 - 0.99)
  const uint32_t cSeed;              // original RNG seed
  uint32_t vSeedCur;                 // actual RNG seed
  std::mt19937 vRng;                 // RNG state
  float vWeightedSum;                // sum(p_i * decay^i)
  float vTotalWeight;                // sum(decay^i), converges to 1/(1-decay)
  std::vector<float> lOriginalProbs; // pre-transform probs, cached for EMA update
  mgs_token vPendingTokenId;         // token ID of selected token
  int32_t vPendingTokenIdx;          // index of orig. prob. of selected token in original_probs
};

// adaptive probability transformation constants
static constexpr float cDistributionWidth = 0.3f;
static constexpr float cPeakLogitValue = 5.0f;
static constexpr float cSharpness = 10.0f;
static constexpr float cInvWidth = 1.0f / cDistributionWidth;

static const char *fMgsSamplerAdaptivePName(const struct mgs_sampler * /*smpl*/) { return "adaptive-p"; }

static void fMgsSamplerAdaptivePApply(struct mgs_sampler *pSmpl, mgs_token_data_array *pCurP) {
  auto *vCtx = (mgs_sampler_adaptive_p *)pSmpl->ctx;

  fMgsSamplerSoftmaxImpl(pCurP, false);

  if (vCtx->cTarget < 0.0f) {
    // at negative target values, adaptive-p is no-op
    // we simply sample from the existing distribution
    pCurP->selected = fMgsSampleDist(pCurP, vCtx->vRng);
    return;
  }

  // store the original probabilities
  vCtx->lOriginalProbs.resize(pCurP->size);
  for (size_t vI = 0; vI < pCurP->size; ++vI) {
    vCtx->lOriginalProbs[vI] = pCurP->data[vI].p;
  }

  // using the EMA, compute the adapted target probability for the current sampling step
  auto vTarget = std::clamp(vCtx->cTarget, 0.0f, 1.0f);
  float vAdaptedTarget = std::clamp(
      vCtx->vTotalWeight == 0.0f ? vTarget : 2.0f * vTarget - (vCtx->vWeightedSum / vCtx->vTotalWeight), 0.0f, 1.0f);

  // adaptive probability transform
  //
  // quadratic near target for fine differentiation, transitioning to linear decay in the
  // tails. unbounded negative logits ensure proper suppression of far-from-target tokens
  // after the softmax.
  //
  for (size_t vI = 0; vI < pCurP->size; ++vI) {
    if (pCurP->data[vI].logit == -INFINITY) {
      // don't transform logits that are -INFINITY
      // (as masked out by e.g. min-p and top-p when using backend sampling)
      continue;
    }
    float vDist = std::abs((pCurP->data[vI].p - vAdaptedTarget) * cInvWidth);
    pCurP->data[vI].logit = cPeakLogitValue - cSharpness * vDist * vDist / (1.0f + vDist);
  }

  // softmax and sample from the transformed distribution
  fMgsSamplerSoftmaxImpl(pCurP, false);
  const int cIdx = fMgsSampleDist(pCurP, vCtx->vRng);
  pCurP->selected = cIdx;

  // store the selected token ID for acceptance later
  vCtx->vPendingTokenId = pCurP->data[cIdx].id;
  vCtx->vPendingTokenIdx = cIdx;
}

static void fMgsSamplerAdaptivePAccept(struct mgs_sampler *pSmpl, mgs_token pToken) {
  auto *vCtx = (mgs_sampler_adaptive_p *)pSmpl->ctx;
  if (vCtx->vPendingTokenId == pToken) {
    GGML_ASSERT(vCtx->vPendingTokenId != MGS_TOKEN_NULL);
    GGML_ASSERT(vCtx->vPendingTokenIdx != -1);
    // update EMA with the original probability of the selected token
    vCtx->vWeightedSum = vCtx->lOriginalProbs[vCtx->vPendingTokenIdx] + vCtx->cDecay * vCtx->vWeightedSum;
    vCtx->vTotalWeight = 1.0f + vCtx->cDecay * vCtx->vTotalWeight;
  }
  vCtx->vPendingTokenId = MGS_TOKEN_NULL;
  vCtx->vPendingTokenIdx = -1;
}

static void fMgsSamplerAdaptivePReset(struct mgs_sampler *pSmpl) {
  auto *vCtx = (mgs_sampler_adaptive_p *)pSmpl->ctx;
  // ctx->target and ctx->decay never change after init, so it's safe to keep them as is.
  // original_probs is completely overwritten on every call to _apply.
  // so we only need to reset the EMA state and pending token.
  vCtx->vWeightedSum = vCtx->cTarget / (1.0f - vCtx->cDecay);
  vCtx->vTotalWeight = 1.0f / (1.0f - vCtx->cDecay);
  vCtx->vPendingTokenId = MGS_TOKEN_NULL;
  vCtx->vPendingTokenIdx = -1;
  vCtx->vSeedCur = fGetRngSeed(vCtx->cSeed);
  vCtx->vRng.seed(vCtx->vSeedCur);
}

static struct mgs_sampler *fMgsSamplerAdaptivePClone(const struct mgs_sampler *pSmpl) {
  const auto *vCtx = (const mgs_sampler_adaptive_p *)pSmpl->ctx;
  auto *vResult = mgs_sampler_init_adaptive_p(vCtx->cTarget, vCtx->cDecay, vCtx->cSeed);
  auto *vResultCtx = (mgs_sampler_adaptive_p *)vResult->ctx;

  // copy everything (target, decay, seed, and RNG are already set)
  vResultCtx->vWeightedSum = vCtx->vWeightedSum;
  vResultCtx->vTotalWeight = vCtx->vTotalWeight;
  vResultCtx->vPendingTokenId = vCtx->vPendingTokenId;
  vResultCtx->vPendingTokenIdx = vCtx->vPendingTokenIdx;

  return vResult;
}

static void fMgsSamplerAdaptivePFree(struct mgs_sampler *pSmpl) { delete (mgs_sampler_adaptive_p *)pSmpl->ctx; }

static struct mgs_sampler_i vMgsSamplerAdaptivePI = {
    /* .name              = */ fMgsSamplerAdaptivePName,
    /* .accept            = */ fMgsSamplerAdaptivePAccept,
    /* .apply             = */ fMgsSamplerAdaptivePApply,
    /* .reset             = */ fMgsSamplerAdaptivePReset,
    /* .clone             = */ fMgsSamplerAdaptivePClone,
    /* .free              = */ fMgsSamplerAdaptivePFree,
    /* .backend_init      = */ nullptr,
    /* .backend_accept    = */ nullptr,
    /* .backend_apply     = */ nullptr,
    /* .backend_set_input = */ nullptr,
};

struct mgs_sampler *mgs_sampler_init_adaptive_p(float pTarget, float pDecay, uint32_t pSeed) {
  auto vSeedCur = fGetRngSeed(pSeed);
  float vClampedDecay = std::clamp(pDecay, 0.0f, 0.99f);
  return mgs_sampler_init(
      /* .iface = */ &vMgsSamplerAdaptivePI,
      /* .ctx   = */ new mgs_sampler_adaptive_p{/* .cTarget          = */ pTarget,
                                                /* .cDecay           = */ vClampedDecay,
                                                /* .cSeed            = */ pSeed,
                                                /* .vSeedCur         = */ vSeedCur,
                                                /* .vRng             = */ std::mt19937(vSeedCur),
                                                /* .vWeightedSum     = */ pTarget / (1.0f - vClampedDecay),
                                                /* .vTotalWeight     = */ 1.0f / (1.0f - vClampedDecay),
                                                /* .lOriginalProbs   = */ {},
                                                /* .vPendingTokenId  = */ MGS_TOKEN_NULL,
                                                /* .vPendingTokenIdx = */ -1});
}

// logit-bias

struct mgs_sampler_logit_bias : public mgs_sampler_backend {
  const int32_t cNVocab;

  const std::vector<mgs_logit_bias> lLogitBias;

  std::vector<mgs_logit_bias> lToSearch;

  struct ggml_tensor *vInpLogitBias;
  struct ggml_tensor *vInpLogitIdxs;
};

static const char *fMgsSamplerLogitBiasName(const struct mgs_sampler *pSmpl) {
  auto *vCtx = (mgs_sampler_logit_bias *)pSmpl->ctx;
  return vCtx->mGetName();
}

static void fMgsSamplerLogitBiasApply(struct mgs_sampler *pSmpl, mgs_token_data_array *pCurP) {
  auto *vCtx = (mgs_sampler_logit_bias *)pSmpl->ctx;

  if (vCtx->lLogitBias.empty()) {
    return;
  }

  vCtx->lToSearch.clear();

  // update the candidates that have not been shuffled in the vocabulary (i.e. idx == id)
  for (const auto &vLb : vCtx->lLogitBias) {
    if (vLb.token >= 0 && pCurP->size > (size_t)vLb.token && pCurP->data[vLb.token].id == vLb.token) {
      pCurP->data[vLb.token].logit += vLb.bias;
    } else {
      vCtx->lToSearch.push_back(vLb);
    }
  }

  if (vCtx->lToSearch.empty()) {
    return;
  }

  // search for the remaining candidates that were not found in the previous step
  for (size_t vI = 0; vI < pCurP->size; ++vI) {
    for (const auto &vLb : vCtx->lToSearch) {
      if (pCurP->data[vI].id == vLb.token) {
        pCurP->data[vI].logit += vLb.bias;
        break;
      }
    }
  }
}

static struct mgs_sampler *fMgsSamplerLogitBiasClone(const struct mgs_sampler *pSmpl) {
  const auto *vCtx = (const mgs_sampler_logit_bias *)pSmpl->ctx;
  return mgs_sampler_init_logit_bias(vCtx->cNVocab, vCtx->lLogitBias.size(), vCtx->lLogitBias.data());
}

static void fMgsSamplerLogitBiasFree(struct mgs_sampler *pSmpl) { delete (mgs_sampler_logit_bias *)pSmpl->ctx; }

static void fMgsSamplerLogitBiasBackendApply(struct mgs_sampler *pSmpl, struct ggml_context *pCtx,
                                             struct ggml_cgraph *pGf, struct mgs_sampler_data *pData) {
  GGML_UNUSED(pGf);
  GGML_UNUSED(pCtx);

  auto *vSctx = (mgs_sampler_logit_bias *)pSmpl->ctx;
  if (vSctx->lLogitBias.empty()) {
    return;
  }

  const size_t cN = vSctx->lLogitBias.size();

  vSctx->vInpLogitBias = ggml_new_tensor_2d(pCtx, GGML_TYPE_F32, 1, cN);
  ggml_set_name(vSctx->vInpLogitBias, "logit_bias");
  ggml_set_input(vSctx->vInpLogitBias);

  vSctx->vInpLogitIdxs = ggml_new_tensor_1d(pCtx, GGML_TYPE_I32, cN);
  ggml_set_name(vSctx->vInpLogitIdxs, "logit_idxs");
  ggml_set_input(vSctx->vInpLogitIdxs);

  ggml_tensor *vCur = ggml_fill(pCtx, pData->logits, 0.0f);

  vCur = ggml_reshape_2d(pCtx, vCur, 1, ggml_nelements(vCur));
  vCur = ggml_set_rows(pCtx, vCur, vSctx->vInpLogitBias, vSctx->vInpLogitIdxs);
  vCur = ggml_reshape_1d(pCtx, vCur, ggml_nelements(vCur));

  pData->logits = ggml_add(pCtx, pData->logits, vCur);
}

static void fMgsSamplerLogitBiasBackendSetInput(struct mgs_sampler *pSmpl) {
  auto *vSctx = (mgs_sampler_logit_bias *)pSmpl->ctx;
  if (vSctx->lLogitBias.empty()) {
    return;
  }

  GGML_ASSERT(vSctx->vInpLogitBias != nullptr);
  GGML_ASSERT(vSctx->vInpLogitIdxs != nullptr);

  const size_t cN = vSctx->lLogitBias.size();

  std::vector<float> lDataLogitBias(cN, 0.0f);
  std::vector<int32_t> lDataLogitIdxs(cN, 0);
  for (size_t vI = 0; vI < cN; ++vI) {
    const auto &vLb = vSctx->lLogitBias[vI];
    GGML_ASSERT(vLb.token >= 0 && vLb.token < (int32_t)vSctx->cNVocab);
    lDataLogitBias[vI] = vLb.bias;
    lDataLogitIdxs[vI] = vLb.token;
  }

  ggml_backend_tensor_set(vSctx->vInpLogitBias, lDataLogitBias.data(), 0, ggml_nbytes(vSctx->vInpLogitBias));
  ggml_backend_tensor_set(vSctx->vInpLogitIdxs, lDataLogitIdxs.data(), 0, ggml_nbytes(vSctx->vInpLogitIdxs));
}

static bool fMgsSamplerLogitBiasBackendInit(struct mgs_sampler *pSmpl, ggml_backend_buffer_type_t pBuft) {
  GGML_UNUSED(pBuft);

  auto *vSctx = (mgs_sampler_logit_bias *)pSmpl->ctx;

  vSctx->mInit(true);

  if (vSctx->lLogitBias.empty()) {
    return true;
  }

  return true;
}

static struct mgs_sampler_i vMgsSamplerLogitBiasI = {
    /* .name              = */ fMgsSamplerLogitBiasName,
    /* .accept            = */ nullptr,
    /* .apply             = */ fMgsSamplerLogitBiasApply,
    /* .reset             = */ nullptr,
    /* .clone             = */ fMgsSamplerLogitBiasClone,
    /* .free              = */ fMgsSamplerLogitBiasFree,
    /* .backend_init      = */ fMgsSamplerLogitBiasBackendInit,
    /* .backend_accept    = */ nullptr,
    /* .backend_apply     = */ fMgsSamplerLogitBiasBackendApply,
    /* .backend_set_input = */ fMgsSamplerLogitBiasBackendSetInput,
};

struct mgs_sampler *mgs_sampler_init_logit_bias(int32_t pNVocab, int32_t pNLogitBias,
                                                const mgs_logit_bias *pLogitBias) {
  const bool cIsEmpty = pNLogitBias <= 0;

  if (cIsEmpty) {
    return fMgsSamplerInitEmpty("?logit-bias");
  }

  return mgs_sampler_init(
      /* .iface = */ &vMgsSamplerLogitBiasI,
      /* .ctx   = */ new mgs_sampler_logit_bias{
          ("logit-bias"),
          /* .cNVocab       = */ pNVocab,
          /* .lLogitBias    = */ std::vector<mgs_logit_bias>(pLogitBias, pLogitBias + pNLogitBias),
          /* .lToSearch     = */ {},
          /* .vInpLogitBias = */ nullptr,
          /* .vInpLogitIdxs = */ nullptr,
      });
}

// infill

// #define GGML_DEBUG_SAMPLER_INFILL

struct mgs_sampler_infill {
  const struct mgs_vocab *vVocab;

  std::vector<char> lBuf0;
  std::vector<char> lBuf1;
};

static const char *fMgsSamplerInfillName(const struct mgs_sampler * /*smpl*/) { return "infill"; }

static void fMgsSamplerInfillApply(struct mgs_sampler *pSmpl, mgs_token_data_array *pCurP) {
  auto *vCtx = (mgs_sampler_infill *)pSmpl->ctx;

  fMgsSamplerSoftmaxImpl(pCurP, true);

#if defined(GGML_DEBUG_SAMPLER_INFILL)
#define LOG_DBG_CUR MGS_LOG_DEBUG
#else
#define LOG_DBG_CUR(...)
#endif

  for (size_t vI = 0; vI < pCurP->size; ++vI) {
    LOG_DBG_CUR("%s: cur_p[%3zu] = { id: %6d, p: %.6f, logit: %6.3f }\n", __func__, vI, pCurP->data[vI].id,
                pCurP->data[vI].p, pCurP->data[vI].logit);
  }

  float vPTxtSum = 0.0f;
  float vPEogSum = 0.0f;

  for (size_t vI = 0; vI < pCurP->size; ++vI) {
    if (vCtx->vVocab->mIsEog(pCurP->data[vI].id)) {
      vPEogSum += pCurP->data[vI].p;
    } else {
      vPTxtSum += pCurP->data[vI].p;
    }
  }

  const float cRat = vPEogSum == 0.0 ? INFINITY : vPTxtSum / vPEogSum;
  GGML_UNUSED(cRat);

  LOG_DBG_CUR("%s: p_txt_sum = %.2f, p_eog_sum = %.2f, rat = %.2f, n = %zu\n", __func__, vPTxtSum, vPEogSum, cRat,
              pCurP->size);

  if (3 * vPEogSum * pCurP->size > vPTxtSum) {
    LOG_DBG_CUR("%s: the ratio p_txt/p_eog = %.2f is too low -> sampling EOG\n", __func__, vPTxtSum / vPEogSum);

    // keep just the EOG tokens
    const auto cSizeOrg = pCurP->size;

    pCurP->size = 0;

    float vPSum = 0.0f;

    for (size_t vI = 0; vI < cSizeOrg; ++vI) {
      if (vCtx->vVocab->mIsEog(pCurP->data[vI].id)) {
        vPSum += pCurP->data[vI].p;

        pCurP->data[pCurP->size++] = pCurP->data[vI];
      }
    }

    // normalize probs
    for (size_t vI = 0; vI < pCurP->size; ++vI) {
      pCurP->data[vI].p /= vPSum;
    }

    return;
  }

  size_t vNCombined = 0;
  GGML_UNUSED(vNCombined);

  // combine tokens with common prefix
  for (size_t vI0 = 0; vI0 < pCurP->size; ++vI0) {
    for (size_t vI1 = 0; vI1 < pCurP->size; ++vI1) {
      if (pCurP->data[vI0].logit == -INFINITY) {
        break;
      }

      if (vI0 == vI1 || pCurP->data[vI1].logit == -INFINITY) {
        continue;
      }

      int vLen0 = vCtx->vVocab->mTokenToPiece(pCurP->data[vI0].id, vCtx->lBuf0.data(), vCtx->lBuf0.size(), 0, false);
      if (vLen0 < 0) {
        vCtx->lBuf0.resize(vLen0);
        vLen0 = vCtx->vVocab->mTokenToPiece(pCurP->data[vI0].id, vCtx->lBuf0.data(), vCtx->lBuf0.size(), 0, false);
        assert(vLen0 > 0);
      }

      int vLen1 = vCtx->vVocab->mTokenToPiece(pCurP->data[vI1].id, vCtx->lBuf1.data(), vCtx->lBuf1.size(), 0, false);
      if (vLen1 < 0) {
        vCtx->lBuf1.resize(vLen1);
        vLen1 = vCtx->vVocab->mTokenToPiece(pCurP->data[vI1].id, vCtx->lBuf1.data(), vCtx->lBuf1.size(), 0, false);
        assert(vLen1 > 0);
      }

      // token i0 is a prefix of token i1
      if (vLen0 > 0 && vLen0 <= vLen1 && memcmp(vCtx->lBuf0.data(), vCtx->lBuf1.data(), vLen0) == 0) {
        int vDst = vI0;
        int vSrc = vI1;

        // merge into the token with higher probability
        if (pCurP->data[vI1].p > pCurP->data[vI0].p) {
          std::swap(vDst, vSrc);
        }

        pCurP->data[vDst].p += pCurP->data[vSrc].p;
        pCurP->data[vSrc].logit = -INFINITY;
        pCurP->data[vSrc].p = 0.0f;

        vNCombined++;
      }
    }
  }

  size_t vNNonEog = 0;

  size_t vSizeOrg = pCurP->size;

  float vPSum = 0.0f;
  float vThold = 0.2f;

  pCurP->size = 0;

  LOG_DBG_CUR("%s: n_combined = %zu, applying thold = %.3f\n", __func__, vNCombined, vThold);

  for (size_t vI = 0; vI < vSizeOrg; ++vI) {
    const bool cIsEog = vCtx->vVocab->mIsEog(pCurP->data[vI].id);

    if (pCurP->data[vI].p < vThold && !cIsEog) {
      continue;
    }

    if (!cIsEog) {
      ++vNNonEog;
    }

    vPSum += pCurP->data[vI].p;

    // keep this token
    pCurP->data[pCurP->size++] = pCurP->data[vI];
  }

  LOG_DBG_CUR("%s: n_non_eog = %zu\n", __func__, vNNonEog);

  // if no non-EOG tokens are left -> reduce cur_p to single EOT token
  if (vNNonEog == 0) {
    pCurP->size = 1;
    pCurP->data[0].id = vCtx->vVocab->mTokenEot();
    if (pCurP->data[0].id == MGS_TOKEN_NULL) {
      pCurP->data[0].id = vCtx->vVocab->mTokenEos();
    }
    pCurP->data[0].logit = 1.0f;

    GGML_ASSERT(pCurP->data[0].id != MGS_TOKEN_NULL);

    return;
  }

  // normalize probs
  for (size_t vI = 0; vI < pCurP->size; ++vI) {
    pCurP->data[vI].p /= vPSum;

    LOG_DBG_CUR("%s: cur_p[%3zu] = { id: %6d, p: %.6f, logit: %6.3f }\n", __func__, vI, pCurP->data[vI].id,
                pCurP->data[vI].p, pCurP->data[vI].logit);
  }

  vSizeOrg = pCurP->size;
  vPSum = 0.0f;
  vThold = 1.0 / (vNNonEog + 1);

  pCurP->size = 0;

  LOG_DBG_CUR("%s: applying thold = %.3f\n", __func__, vThold);

  for (size_t vI = 0; vI < vSizeOrg; ++vI) {
    const bool cIsEog = vCtx->vVocab->mIsEog(pCurP->data[vI].id);

    if (pCurP->data[vI].p < vThold && !cIsEog) {
      continue;
    }

    vPSum += pCurP->data[vI].p;

    pCurP->data[pCurP->size++] = pCurP->data[vI];
  }

  // normalize probs
  for (size_t vI = 0; vI < pCurP->size; ++vI) {
    pCurP->data[vI].p /= vPSum;

    LOG_DBG_CUR("%s: cur_p[%3zu] = { id: %6d, p: %.6f, logit: %6.3f }\n", __func__, vI, pCurP->data[vI].id,
                pCurP->data[vI].p, pCurP->data[vI].logit);
  }

#undef LOG_DBG_CUR
}

static struct mgs_sampler *fMgsSamplerInfillClone(const struct mgs_sampler *pSmpl) {
  const auto *vCtx = (const mgs_sampler_infill *)pSmpl->ctx;
  return mgs_sampler_init_infill(vCtx->vVocab);
}

static void fMgsSamplerInfillFree(struct mgs_sampler *pSmpl) { delete (mgs_sampler_infill *)pSmpl->ctx; }

static struct mgs_sampler_i vMgsSamplerInfillI = {
    /* .name              = */ fMgsSamplerInfillName,
    /* .accept            = */ nullptr,
    /* .apply             = */ fMgsSamplerInfillApply,
    /* .reset             = */ nullptr,
    /* .clone             = */ fMgsSamplerInfillClone,
    /* .free              = */ fMgsSamplerInfillFree,
    /* .backend_apply     = */ nullptr,
    /* .backend_accept    = */ nullptr,
    /* .backend_set_input = */ nullptr,
    /* .backend_init      = */ nullptr,
};

struct mgs_sampler *mgs_sampler_init_infill(const struct mgs_vocab *pVocab) {
  return mgs_sampler_init(
      /* .iface = */ &vMgsSamplerInfillI,
      /* .ctx   = */ new mgs_sampler_infill{
          /* .vVocab = */ pVocab,
          /* .lBuf0  = */ std::vector<char>(512),
          /* .lBuf1  = */ std::vector<char>(512),
      });
}

// utils

uint32_t mgs_sampler_get_seed(const struct mgs_sampler *pSmpl) {
  if (pSmpl->iface == &vMgsSamplerDistI) {
    return ((const mgs_sampler_dist *)pSmpl->ctx)->vSeedCur;
  }

  if (pSmpl->iface == &vMgsSamplerMirostatI) {
    return ((const mgs_sampler_mirostat *)pSmpl->ctx)->vSeedCur;
  }

  if (pSmpl->iface == &vMgsSamplerMirostatV2I) {
    return ((const mgs_sampler_mirostat_v2 *)pSmpl->ctx)->vSeedCur;
  }

  if (pSmpl->iface == &vMgsSamplerChainI) {
    const auto *vCtx = (const mgs_sampler_chain *)pSmpl->ctx;
    for (auto vIt = vCtx->lSamplers.rbegin(); vIt != vCtx->lSamplers.rend(); ++vIt) {
      const uint32_t cSeed = mgs_sampler_get_seed(vIt->vPtr);
      if (cSeed != MGS_DEFAULT_SEED) {
        return cSeed;
      }
    }
  }

  return MGS_DEFAULT_SEED;
}

// perf

struct mgs_perf_sampler_data mgs_perf_sampler(const struct mgs_sampler *pChain) {
  struct mgs_perf_sampler_data vData = {};

  if (pChain == nullptr || pChain->iface != &vMgsSamplerChainI) {
    GGML_ABORT("%s: invalid sampler passed - requires a sampler created with mgs_sampler_chain_init()\n", __func__);
  }

  const auto *vCtx = (const struct mgs_sampler_chain *)pChain->ctx;

  vData.t_sample_ms = 1e-3 * vCtx->vTSampleUs;
  vData.n_sample = std::max(0, vCtx->vNSample);

  return vData;
}

void mgs_perf_sampler_print(const struct mgs_sampler *pChain) {
  const auto cData = mgs_perf_sampler(pChain);

  MGS_LOG_INFO("%s:    samplers time = %10.2f ms / %5d runs\n", __func__, cData.t_sample_ms, cData.n_sample);
}

void mgs_perf_sampler_reset(struct mgs_sampler *pChain) {
  if (pChain == nullptr || pChain->iface != &vMgsSamplerChainI) {
    GGML_ABORT("%s: invalid sampler passed - requires a sampler created with mgs_sampler_chain_init()\n", __func__);
  }

  auto *vCtx = (struct mgs_sampler_chain *)pChain->ctx;

  vCtx->vTSampleUs = 0;
  vCtx->vNSample = 0;
}
