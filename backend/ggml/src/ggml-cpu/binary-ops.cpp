#include "binary-ops.h"

#if defined(GGML_USE_ACCELERATE)
#include <Accelerate/Accelerate.h>

using vDSP_fn_t = void (*)(const float *, vDSP_Stride, const float *, vDSP_Stride, float *, vDSP_Stride, vDSP_Length);
#endif

static inline float op_add(float pA, float pB) { return pA + pB; }

static inline float op_sub(float pA, float pB) { return pA - pB; }

static inline float op_mul(float pA, float pB) { return pA * pB; }

static inline float op_div(float pA, float pB) { return pA / pB; }

template <float (*op)(float, float), typename src0_t, typename src1_t, typename dst_t>
static inline void fVecBinaryOpContiguous(const int64_t pN, dst_t *pZ, const src0_t *pX, const src1_t *pY) {
  constexpr auto cSrc0ToF32 = type_conversion_table<src0_t>::to_f32;
  constexpr auto cSrc1ToF32 = type_conversion_table<src1_t>::to_f32;
  constexpr auto cF32ToDst = type_conversion_table<dst_t>::from_f32;

  for (int vI = 0; vI < pN; vI++) {
    pZ[vI] = cF32ToDst(op(cSrc0ToF32(pX[vI]), cSrc1ToF32(pY[vI])));
  }
}

template <float (*op)(float, float), typename src0_t, typename src1_t, typename dst_t>
static inline void fVecBinaryOpNonContiguous(const int64_t pN, const int64_t pNe10, const int64_t pNb10, dst_t *pZ,
                                             const src0_t *pX, const src1_t *pY) {
  constexpr auto cSrc0ToF32 = type_conversion_table<src0_t>::to_f32;
  constexpr auto cSrc1ToF32 = type_conversion_table<src1_t>::to_f32;
  constexpr auto cF32ToDst = type_conversion_table<dst_t>::from_f32;

  for (int vI = 0; vI < pN; vI++) {
    int vI10 = vI % pNe10;
    const src1_t *vYPtr = (const src1_t *)((const char *)pY + vI10 * pNb10);
    pZ[vI] = cF32ToDst(op(cSrc0ToF32(pX[vI]), cSrc1ToF32(*vYPtr)));
  }
}

template <float (*op)(float, float), typename src0_t, typename src1_t, typename dst_t>
static void fApplyBinaryOp(const ggml_compute_params *pParams, ggml_tensor *dst) {
  const ggml_tensor *src0 = dst->src[0];
  const ggml_tensor *src1 = dst->src[1];

  GGML_ASSERT(ggml_can_repeat(src1, src0) && ggml_are_same_shape(src0, dst));

  GGML_TENSOR_BINARY_OP_LOCALS

  GGML_ASSERT(nb0 == sizeof(dst_t));
  GGML_ASSERT(nb00 == sizeof(src0_t));

  const auto [ir0, ir1] = get_thread_range(pParams, src0);
  const bool cIsSrc1ContiguousRows = ggml_is_contiguous_rows(src1);

#ifdef GGML_USE_ACCELERATE
  vDSP_fn_t vDSP_op = nullptr;
  // TODO - avoid the f32-only check using type 'trait' lookup tables and row-based src-to-float conversion functions
  if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
    if (op == op_add) {
      vDSP_op = vDSP_vadd;
    } else if (op == op_sub) {
      vDSP_op = vDSP_vsub;
    } else if (op == op_mul) {
      vDSP_op = vDSP_vmul;
    } else if (op == op_div) {
      vDSP_op = vDSP_vdiv;
    }
  }
#endif

  for (int64_t vIr = ir0; vIr < ir1; ++vIr) {
    const int64_t cI03 = vIr / (ne02 * ne01);
    const int64_t cI02 = (vIr - cI03 * ne02 * ne01) / ne01;
    const int64_t cI01 = (vIr - cI03 * ne02 * ne01 - cI02 * ne01);

    const int64_t cI13 = cI03 % ne13;
    const int64_t cI12 = cI02 % ne12;
    const int64_t cI11 = cI01 % ne11;

    dst_t *vDstPtr = (dst_t *)((char *)dst->data + cI03 * nb3 + cI02 * nb2 + cI01 * nb1);
    const src0_t *vSrc0Ptr = (const src0_t *)((const char *)src0->data + cI03 * nb03 + cI02 * nb02 + cI01 * nb01);
    const src1_t *vSrc1Ptr = (const src1_t *)((const char *)src1->data + cI13 * nb13 + cI12 * nb12 + cI11 * nb11);

    if (cIsSrc1ContiguousRows) {
      // src1 is broadcastable across src0 and dst in i1, i2, i3
      const int64_t cNr0 = ne00 / ne10;

      for (int64_t vR = 0; vR < cNr0; ++vR) {
#ifdef GGML_USE_ACCELERATE
        if constexpr (std::is_same_v<src0_t, float> && std::is_same_v<src1_t, float> && std::is_same_v<dst_t, float>) {
          if (vDSP_op != nullptr) {
            vDSP_op(vSrc1Ptr, 1, vSrc0Ptr + vR * ne10, 1, vDstPtr + vR * ne10, 1, ne10);
            continue;
          }
        }
#endif
        fVecBinaryOpContiguous<op>(ne10, vDstPtr + vR * ne10, vSrc0Ptr + vR * ne10, vSrc1Ptr);
      }
    } else {
      fVecBinaryOpNonContiguous<op>(ne0, ne10, nb10, vDstPtr, vSrc0Ptr, vSrc1Ptr);
    }
  }
}

// TODO: Use the 'traits' lookup table (for type conversion fns), instead of a mass of 'if' conditions with long
// templates
template <float (*op)(float, float)> static void fBinaryOp(const ggml_compute_params *pParams, ggml_tensor *pDst) {
  const ggml_tensor *vSrc0 = pDst->src[0];
  const ggml_tensor *vSrc1 = pDst->src[1];

  /*  */ if (vSrc0->type == GGML_TYPE_F32 && vSrc1->type == GGML_TYPE_F32 && pDst->type == GGML_TYPE_F32) { // all f32
    fApplyBinaryOp<op, float, float, float>(pParams, pDst);
  } else if (vSrc0->type == GGML_TYPE_F16 && vSrc1->type == GGML_TYPE_F16 && pDst->type == GGML_TYPE_F16) { // all f16
    fApplyBinaryOp<op, ggml_fp16_t, ggml_fp16_t, ggml_fp16_t>(pParams, pDst);
  } else if (vSrc0->type == GGML_TYPE_BF16 && vSrc1->type == GGML_TYPE_BF16 &&
             pDst->type == GGML_TYPE_BF16) { // all bf16
    fApplyBinaryOp<op, ggml_bf16_t, ggml_bf16_t, ggml_bf16_t>(pParams, pDst);
  } else if (vSrc0->type == GGML_TYPE_BF16 && vSrc1->type == GGML_TYPE_F32 && pDst->type == GGML_TYPE_BF16) {
    fApplyBinaryOp<op, ggml_bf16_t, float, ggml_bf16_t>(pParams, pDst);
  } else if (vSrc0->type == GGML_TYPE_BF16 && vSrc1->type == GGML_TYPE_F32 && pDst->type == GGML_TYPE_F32) {
    fApplyBinaryOp<op, ggml_bf16_t, float, float>(pParams, pDst);
  } else if (vSrc0->type == GGML_TYPE_F16 && vSrc1->type == GGML_TYPE_F32 && pDst->type == GGML_TYPE_F16) {
    fApplyBinaryOp<op, ggml_fp16_t, float, ggml_fp16_t>(pParams, pDst);
  } else if (vSrc0->type == GGML_TYPE_F16 && vSrc1->type == GGML_TYPE_F32 && pDst->type == GGML_TYPE_F32) {
    fApplyBinaryOp<op, ggml_fp16_t, float, float>(pParams, pDst);
  } else {
    GGML_ABORT("%s: unsupported types: dst: %s, src0: %s, src1: %s\n", __func__, ggml_type_name(pDst->type),
               ggml_type_name(vSrc0->type), ggml_type_name(vSrc1->type));
  }
}

void ggml_compute_forward_add_non_quantized(const ggml_compute_params *pParams, ggml_tensor *pDst) {
  fBinaryOp<op_add>(pParams, pDst);
}

void ggml_compute_forward_sub(const ggml_compute_params *pParams, ggml_tensor *pDst) {
  fBinaryOp<op_sub>(pParams, pDst);
}

void ggml_compute_forward_mul(const ggml_compute_params *pParams, ggml_tensor *pDst) {
  fBinaryOp<op_mul>(pParams, pDst);
}

void ggml_compute_forward_div(const ggml_compute_params *pParams, ggml_tensor *pDst) {
  fBinaryOp<op_div>(pParams, pDst);
}
