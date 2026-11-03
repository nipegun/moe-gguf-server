#pragma once

#include "server-common.h"
#include "server-task.h"

#include "sampling.h"
#include "speculative.h"

#include <climits>
#include <functional>
#include <limits>
#include <memory>
#include <string>
#include <vector>

namespace server_schema {

struct field_eval_context {
  task_params &vParams;
  const mgs_vocab *vVocab = nullptr;
  const std::vector<mgs_logit_bias> *lLogitBiasEog = nullptr;
  field_eval_context(task_params &pParams) : vParams(pParams) {}
};

using field_handler = std::function<void(field_eval_context &, const json &)>;

struct field {
  std::vector<const char *> lName;
  const char *vDesc = "";
  field_handler vCustomHandler;
  field() = default;
  field(const char *pN) : lName({pN}) {}
  virtual ~field() = default;
  field *mSetDesc(const char *pS) {
    vDesc = pS;
    return this;
  }
  // if 'name' is present, use it, otherwise look for aliases following the order they were added
  field *mAddAlias(const char *pN) {
    lName.push_back(pN);
    return this;
  }
  field *mSetHandler(field_handler pH) {
    this->vCustomHandler = pH;
    return this;
  }
  virtual void mEval(field_eval_context &pCtx, const json &pData) = 0;
};

template <typename T = int32_t> struct field_num : public field {
  T &vVal;
  T vMin = std::numeric_limits<T>::lowest();
  T vMax = std::numeric_limits<T>::max();
  bool vIsHardLimit = false; // if true, throw error if the value is invalid
  field_num(const char *pN, T &pVal) : field(pN), vVal(pVal) {}
  // limits are inclusive, min <= value <= max
  field_num *mSetLimits(T pMin, T pMax) {
    this->vMin = pMin;
    this->vMax = pMax;
    return this;
  }
  field_num *mSetHardLimits(T pMin, T pMax) {
    mSetLimits(pMin, pMax);
    vIsHardLimit = true;
    return this;
  }
  virtual void mEval(field_eval_context &pCtx, const json &pData) override;
};

struct field_str : public field {
  field_str(const char *pN) : field(pN) {}
  virtual void mEval(field_eval_context &pCtx, const json &pData) override;
};

struct field_bool : public field {
  bool &vVal;
  field_bool(const char *pN, bool &pVal) : field(pN), vVal(pVal) {}
  virtual void mEval(field_eval_context &pCtx, const json &pData) override;
};

struct field_json : public field {
  field_json(const char *pN) : field(pN) {}
  virtual void mEval(field_eval_context &pCtx, const json &pData) override;
};

struct field_nested : public field {
  std::vector<std::unique_ptr<field>> lSubfields;
  field_nested(const char *pN) : field(pN) {}
  field_nested *mAddSubfield(field *pF) {
    lSubfields.emplace_back(std::unique_ptr<field>(pF));
    return this;
  }
  virtual void mEval(field_eval_context &pCtx, const json &pData) override;
};

std::vector<std::unique_ptr<field>> fMakeMgsCmplSchema(const common_params &pParamsBase, task_params &pParams);

task_params fEvalMgsCmplSchema(const mgs_vocab *pVocab, const common_params &pParamsBase, const int pNCtxSlot,
                               const std::vector<mgs_logit_bias> &pLogitBiasEog, const json &pData);

} // namespace server_schema
