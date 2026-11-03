#include "unary-ops.h"

static inline float op_abs(float pX) { return fabsf(pX); }

static inline float op_sgn(float pX) { return (pX > 0.f) ? 1.f : ((pX < 0.f) ? -1.f : 0.f); }

static inline float op_neg(float pX) { return -pX; }

static inline float op_step(float pX) { return (pX > 0.f) ? 1.f : 0.f; }

static inline float op_tanh(float pX) { return tanhf(pX); }

static inline float op_elu(float pX) { return (pX > 0.f) ? pX : expm1f(pX); }

static inline float op_relu(float pX) { return (pX > 0.f) ? pX : 0.f; }

static inline float op_sigmoid(float pX) { return 1.f / (1.f + expf(-pX)); }

static inline float fOpHardsigmoid(float pX) { return fminf(1.0f, fmaxf(0.0f, (pX + 3.0f) / 6.0f)); }

static inline float op_exp(float pX) { return expf(pX); }

static inline float op_hardswish(float pX) { return pX * fminf(1.0f, fmaxf(0.0f, (pX + 3.0f) / 6.0f)); }

static inline float op_sqr(float pX) { return pX * pX; }

static inline float op_sqrt(float pX) { return sqrtf(pX); }

static inline float fOpXielu(float pX, float pAlphaN, float pAlphaP, float pBeta, float pEps) {
  if (pX > 0.0f) {
    return pAlphaP * pX * pX + pBeta * pX;
  } else {
    const float cMinXEps = fminf(pX, pEps);
    return (expm1f(cMinXEps) - pX) * pAlphaN + pBeta * pX;
  }
}

static inline float op_sin(float pX) { return sinf(pX); }

static inline float op_cos(float pX) { return cosf(pX); }

static inline float op_log(float pX) { return logf(pX); }

static inline float op_expm1(float pX) { return expf(pX) - 1.0f; }

static inline float op_softplus(float pX) { return (pX > 20.0f) ? pX : logf(1.0f + expf(pX)); }

static inline float op_floor(float pX) { return floorf(pX); }

static inline float op_ceil(float pX) { return ceilf(pX); }

static inline float op_round(float pX) { return roundf(pX); }

static inline float op_trunc(float pX) { return truncf(pX); }

template <float (*op)(float), typename src0_t, typename dst_t>
static inline void fVecUnaryOp(int64_t pN, dst_t *pY, const src0_t *pX) {
  constexpr auto cSrc0ToF32 = type_conversion_table<src0_t>::to_f32;
  constexpr auto cF32ToDst = type_conversion_table<dst_t>::from_f32;

  for (int vI = 0; vI < pN; vI++) {
    pY[vI] = cF32ToDst(op(cSrc0ToF32(pX[vI])));
  }
}

template <float (*op)(float), typename src0_t, typename dst_t>
static void apply_unary_op(const ggml_compute_params *pParams, ggml_tensor *dst) {
  const ggml_tensor *src0 = dst->src[0];

  GGML_ASSERT(ggml_is_contiguous_rows(src0) && ggml_is_contiguous_rows(dst) && ggml_are_same_shape(src0, dst));

  GGML_TENSOR_UNARY_OP_LOCALS

  GGML_ASSERT(nb0 == sizeof(dst_t));
  GGML_ASSERT(nb00 == sizeof(src0_t));

  const auto [ir0, ir1] = get_thread_range(pParams, src0);

  for (int64_t vIr = ir0; vIr < ir1; ++vIr) {
    const int64_t cI03 = vIr / (ne02 * ne01);
    const int64_t cI02 = (vIr - cI03 * ne02 * ne01) / ne01;
    const int64_t cI01 = (vIr - cI03 * ne02 * ne01 - cI02 * ne01);

    dst_t *vDstPtr = (dst_t *)((char *)dst->data + cI03 * nb3 + cI02 * nb2 + cI01 * nb1);
    const src0_t *vSrc0Ptr = (const src0_t *)((const char *)src0->data + cI03 * nb03 + cI02 * nb02 + cI01 * nb01);

    fVecUnaryOp<op>(ne0, vDstPtr, vSrc0Ptr);
  }
}

// TODO: Use the 'traits' lookup table (for type conversion fns), instead of a mass of 'if' conditions with long
// templates
template <float (*op)(float)> static void fUnaryOp(const ggml_compute_params *pParams, ggml_tensor *pDst) {
  const ggml_tensor *vSrc0 = pDst->src[0];

  /*  */ if (vSrc0->type == GGML_TYPE_F32 && pDst->type == GGML_TYPE_F32) { // all f32
    apply_unary_op<op, float, float>(pParams, pDst);
  } else if (vSrc0->type == GGML_TYPE_F16 && pDst->type == GGML_TYPE_F16) { // all f16
    apply_unary_op<op, ggml_fp16_t, ggml_fp16_t>(pParams, pDst);
  } else if (vSrc0->type == GGML_TYPE_BF16 && pDst->type == GGML_TYPE_BF16) { // all bf16
    apply_unary_op<op, ggml_bf16_t, ggml_bf16_t>(pParams, pDst);
  } else if (vSrc0->type == GGML_TYPE_BF16 && pDst->type == GGML_TYPE_F32) {
    apply_unary_op<op, ggml_bf16_t, float>(pParams, pDst);
  } else if (vSrc0->type == GGML_TYPE_F16 && pDst->type == GGML_TYPE_F32) {
    apply_unary_op<op, ggml_fp16_t, float>(pParams, pDst);
  } else {
    fprintf(stderr, "%s: unsupported types: dst: %s, src0: %s\n", __func__, ggml_type_name(pDst->type),
            ggml_type_name(vSrc0->type));
    GGML_ABORT("fatal error");
  }
}

template <float (*op)(float, ggml_tensor *)>
static void fUnaryOpParams(const ggml_compute_params *pParams, ggml_tensor *pDst) {
  const ggml_tensor *vSrc0 = pDst->src[0];

  /*  */ if (vSrc0->type == GGML_TYPE_F32 && pDst->type == GGML_TYPE_F32) { // all f32
    apply_unary_op<op, float, float>(pParams, pDst);
  } else if (vSrc0->type == GGML_TYPE_F16 && pDst->type == GGML_TYPE_F16) { // all f16
    apply_unary_op<op, ggml_fp16_t, ggml_fp16_t>(pParams, pDst);
  } else if (vSrc0->type == GGML_TYPE_BF16 && pDst->type == GGML_TYPE_BF16) { // all bf16
    apply_unary_op<op, ggml_bf16_t, ggml_bf16_t>(pParams, pDst);
  } else if (vSrc0->type == GGML_TYPE_BF16 && pDst->type == GGML_TYPE_F32) {
    apply_unary_op<op, ggml_bf16_t, float>(pParams, pDst);
  } else if (vSrc0->type == GGML_TYPE_F16 && pDst->type == GGML_TYPE_F32) {
    apply_unary_op<op, ggml_fp16_t, float>(pParams, pDst);
  } else {
    fprintf(stderr, "%s: unsupported types: dst: %s, src0: %s\n", __func__, ggml_type_name(pDst->type),
            ggml_type_name(vSrc0->type));
    GGML_ABORT("fatal error");
  }
}

// Extend vec_unary_op to support functors
template <typename Op, typename src0_t, typename dst_t>
static inline void fVecUnaryOpFunctor(int64_t pN, dst_t *pY, const src0_t *pX, Op pOp) {
  constexpr auto cSrc0ToF32 = type_conversion_table<src0_t>::to_f32;
  constexpr auto cF32ToDst = type_conversion_table<dst_t>::from_f32;

  for (int vI = 0; vI < pN; vI++) {
    pY[vI] = cF32ToDst(pOp(cSrc0ToF32(pX[vI])));
  }
}

// Extend apply_unary_op to support functors
template <typename Op, typename src0_t, typename dst_t>
static void fApplyUnaryOpFunctor(const ggml_compute_params *pParams, ggml_tensor *dst, Op pOp) {
  const ggml_tensor *src0 = dst->src[0];

  GGML_ASSERT(ggml_is_contiguous_1(src0) && ggml_is_contiguous_1(dst) && ggml_are_same_shape(src0, dst));

  GGML_TENSOR_UNARY_OP_LOCALS

  GGML_ASSERT(nb0 == sizeof(dst_t));
  GGML_ASSERT(nb00 == sizeof(src0_t));

  const auto [ir0, ir1] = get_thread_range(pParams, src0);

  for (int64_t vIr = ir0; vIr < ir1; ++vIr) {
    const int64_t cI03 = vIr / (ne02 * ne01);
    const int64_t cI02 = (vIr - cI03 * ne02 * ne01) / ne01;
    const int64_t cI01 = (vIr - cI03 * ne02 * ne01 - cI02 * ne01);

    dst_t *vDstPtr = (dst_t *)((char *)dst->data + cI03 * nb3 + cI02 * nb2 + cI01 * nb1);
    const src0_t *vSrc0Ptr = (const src0_t *)((const char *)src0->data + cI03 * nb03 + cI02 * nb02 + cI01 * nb01);

    fVecUnaryOpFunctor(ne0, vDstPtr, vSrc0Ptr, pOp);
  }
}

// Generic dispatcher for functors
template <typename Op> static void fUnaryOpFunctor(const ggml_compute_params *pParams, ggml_tensor *pDst, Op pOp) {
  const ggml_tensor *vSrc0 = pDst->src[0];

  /*  */ if (vSrc0->type == GGML_TYPE_F32 && pDst->type == GGML_TYPE_F32) { // all f32
    fApplyUnaryOpFunctor<Op, float, float>(pParams, pDst, pOp);
  } else if (vSrc0->type == GGML_TYPE_F16 && pDst->type == GGML_TYPE_F16) { // all f16
    fApplyUnaryOpFunctor<Op, ggml_fp16_t, ggml_fp16_t>(pParams, pDst, pOp);
  } else if (vSrc0->type == GGML_TYPE_BF16 && pDst->type == GGML_TYPE_BF16) { // all bf16
    fApplyUnaryOpFunctor<Op, ggml_bf16_t, ggml_bf16_t>(pParams, pDst, pOp);
  } else if (vSrc0->type == GGML_TYPE_BF16 && pDst->type == GGML_TYPE_F32) {
    fApplyUnaryOpFunctor<Op, ggml_bf16_t, float>(pParams, pDst, pOp);
  } else if (vSrc0->type == GGML_TYPE_F16 && pDst->type == GGML_TYPE_F32) {
    fApplyUnaryOpFunctor<Op, ggml_fp16_t, float>(pParams, pDst, pOp);
  } else {
    fprintf(stderr, "%s: unsupported types: dst: %s, src0: %s\n", __func__, ggml_type_name(pDst->type),
            ggml_type_name(vSrc0->type));
    GGML_ABORT("fatal error");
  }
}

void ggml_compute_forward_abs(const ggml_compute_params *pParams, ggml_tensor *pDst) {
  fUnaryOp<op_abs>(pParams, pDst);
}

void ggml_compute_forward_sgn(const ggml_compute_params *pParams, ggml_tensor *pDst) {
  fUnaryOp<op_sgn>(pParams, pDst);
}

void ggml_compute_forward_neg(const ggml_compute_params *pParams, ggml_tensor *pDst) {
  fUnaryOp<op_neg>(pParams, pDst);
}

void ggml_compute_forward_step(const ggml_compute_params *pParams, ggml_tensor *pDst) {
  fUnaryOp<op_step>(pParams, pDst);
}

void ggml_compute_forward_tanh(const ggml_compute_params *pParams, ggml_tensor *pDst) {
  fUnaryOp<op_tanh>(pParams, pDst);
}

void ggml_compute_forward_elu(const ggml_compute_params *pParams, ggml_tensor *pDst) {
  fUnaryOp<op_elu>(pParams, pDst);
}

void ggml_compute_forward_relu(const ggml_compute_params *pParams, ggml_tensor *pDst) {
  fUnaryOp<op_relu>(pParams, pDst);
}

void ggml_compute_forward_sigmoid(const ggml_compute_params *pParams, ggml_tensor *pDst) {
  fUnaryOp<op_sigmoid>(pParams, pDst);
}

void ggml_compute_forward_hardsigmoid(const ggml_compute_params *pParams, ggml_tensor *pDst) {
  fUnaryOp<fOpHardsigmoid>(pParams, pDst);
}

void ggml_compute_forward_exp(const ggml_compute_params *pParams, ggml_tensor *pDst) {
  fUnaryOp<op_exp>(pParams, pDst);
}

void ggml_compute_forward_hardswish(const ggml_compute_params *pParams, ggml_tensor *pDst) {
  fUnaryOp<op_hardswish>(pParams, pDst);
}

void ggml_compute_forward_sqr(const ggml_compute_params *pParams, ggml_tensor *pDst) {
  fUnaryOp<op_sqr>(pParams, pDst);
}

void ggml_compute_forward_sqrt(const ggml_compute_params *pParams, ggml_tensor *pDst) {
  fUnaryOp<op_sqrt>(pParams, pDst);
}

void ggml_compute_forward_sin(const ggml_compute_params *pParams, ggml_tensor *pDst) {
  fUnaryOp<op_sin>(pParams, pDst);
}

void ggml_compute_forward_cos(const ggml_compute_params *pParams, ggml_tensor *pDst) {
  fUnaryOp<op_cos>(pParams, pDst);
}

void ggml_compute_forward_log(const ggml_compute_params *pParams, ggml_tensor *pDst) {
  fUnaryOp<op_log>(pParams, pDst);
}

void ggml_compute_forward_expm1(const ggml_compute_params *pParams, ggml_tensor *pDst) {
  fUnaryOp<op_expm1>(pParams, pDst);
}

void ggml_compute_forward_softplus(const ggml_compute_params *pParams, ggml_tensor *pDst) {
  fUnaryOp<op_softplus>(pParams, pDst);
}

void ggml_compute_forward_floor(const ggml_compute_params *pParams, ggml_tensor *pDst) {
  fUnaryOp<op_floor>(pParams, pDst);
}

void ggml_compute_forward_ceil(const ggml_compute_params *pParams, ggml_tensor *pDst) {
  fUnaryOp<op_ceil>(pParams, pDst);
}

void ggml_compute_forward_round(const ggml_compute_params *pParams, ggml_tensor *pDst) {
  fUnaryOp<op_round>(pParams, pDst);
}

void ggml_compute_forward_trunc(const ggml_compute_params *pParams, ggml_tensor *pDst) {
  fUnaryOp<op_trunc>(pParams, pDst);
}

void ggml_compute_forward_xielu(const ggml_compute_params *pParams, ggml_tensor *pDst) {
  const float cAlphaN = ggml_get_op_params_f32(pDst, 1);
  const float cAlphaP = ggml_get_op_params_f32(pDst, 2);
  const float cBeta = ggml_get_op_params_f32(pDst, 3);
  const float cEps = ggml_get_op_params_f32(pDst, 4);

  const auto fXieluOpParams = [cAlphaN, cAlphaP, cBeta, cEps](float pF) {
    return fOpXielu(pF, cAlphaN, cAlphaP, cBeta, cEps);
  };

  fUnaryOpFunctor(pParams, pDst, fXieluOpParams);
}
