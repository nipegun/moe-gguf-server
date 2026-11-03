#include "ops.h"

#include "ggml-cpu.h"
#include "ggml-impl.h"
#include "binary-ops.h"
#include "simd-gemm.h"
#include "ggml.h"
#include "unary-ops.h"
#include "vec.h"

#include <algorithm>
#include <cfloat>
#include <cmath>

// ggml_compute_forward_dup

static void fGgmlComputeForwardDupSameCont(const ggml_compute_params *pParams, ggml_tensor *pDst) {

  const ggml_tensor *vSrc0 = pDst->src[0];

  GGML_ASSERT(ggml_nelements(pDst) == ggml_nelements(vSrc0));
  GGML_ASSERT(ggml_is_contiguous(pDst) && ggml_is_contiguous(vSrc0));
  GGML_ASSERT(vSrc0->type == pDst->type);

  const size_t cNb0 = ggml_type_size(vSrc0->type);

  const int cIth = pParams->ith; // thread index
  const int cNth = pParams->nth; // number of threads

  // parallelize by blocks
  const int cNk = ggml_nelements(vSrc0) / ggml_blck_size(vSrc0->type);
  const int cDr = (cNk + cNth - 1) / cNth;
  const int cK0 = cDr * cIth;
  const int cK1 = MIN(cK0 + cDr, cNk);

  if (cK0 < cK1) {
    memcpy(((char *)pDst->data + cK0 * cNb0), ((char *)vSrc0->data + cK0 * cNb0), (cK1 - cK0) * cNb0);
  }
}

template <typename src_t, typename dst_t>
static void fGgmlComputeForwardDupFlt(const ggml_compute_params *pParams, ggml_tensor *dst) {

  const ggml_tensor *src0 = dst->src[0];

  GGML_ASSERT(ggml_nelements(dst) == ggml_nelements(src0));
  GGML_ASSERT(!ggml_is_quantized(src0->type) && !ggml_is_quantized(dst->type));

  GGML_TENSOR_UNARY_OP_LOCALS

  const int cIth = pParams->ith; // thread index
  const int cNth = pParams->nth; // number of threads

  // parallelize by rows
  const int cNr = ne01;
  // number of rows per thread
  const int cDr = (cNr + cNth - 1) / cNth;
  // row range for this thread
  const int cIr0 = cDr * cIth;
  const int cIr1 = MIN(cIr0 + cDr, cNr);

  // case: type & row size equal
  if (src0->type == dst->type && ne00 == ne0 && nb00 == ggml_type_size(src0->type) &&
      nb0 == ggml_type_size(dst->type)) {
    // copy by rows
    const size_t cRs = ne00 * nb00;
    for (int64_t vI03 = 0; vI03 < ne03; vI03++) {
      for (int64_t vI02 = 0; vI02 < ne02; vI02++) {
        for (int64_t vI01 = cIr0; vI01 < cIr1; vI01++) {
          memcpy(((char *)dst->data + vI01 * nb1 + vI02 * nb2 + vI03 * nb3),
                 ((char *)src0->data + vI01 * nb01 + vI02 * nb02 + vI03 * nb03), cRs);
        }
      }
    }
    return;
  }

  // case: dst tensor is contiguous
  if (ggml_is_contiguous(dst)) {
    if (nb00 == sizeof(src_t)) {
      if constexpr (std::is_same_v<dst_t, src_t>) {
        // same type
        size_t id = 0;
        const size_t rs = ne00 * nb00;
        char *dst_ptr = (char *)dst->data;

        for (int i03 = 0; i03 < ne03; i03++) {
          for (int i02 = 0; i02 < ne02; i02++) {
            id += rs * cIr0;
            for (int i01 = cIr0; i01 < cIr1; i01++) {
              const char *src0_ptr = (char *)src0->data + i01 * nb01 + i02 * nb02 + i03 * nb03;
              memcpy(dst_ptr + id, src0_ptr, rs);
              id += rs;
            }
            id += rs * (ne01 - cIr1);
          }
        }
      } else {
        // casting between non-quantized types
        size_t vId = 0;
        dst_t *vDstPtr = (dst_t *)dst->data;

        for (int vI03 = 0; vI03 < ne03; vI03++) {
          for (int vI02 = 0; vI02 < ne02; vI02++) {
            vId += ne00 * cIr0;
            for (int vI01 = cIr0; vI01 < cIr1; vI01++) {
              const src_t *vSrc0Ptr = (src_t *)((char *)src0->data + vI01 * nb01 + vI02 * nb02 + vI03 * nb03);
              for (int vI00 = 0; vI00 < ne00; vI00++) {
                float vTmp = type_conversion_table<src_t>::to_f32(vSrc0Ptr[vI00]);
                vDstPtr[vId] = type_conversion_table<dst_t>::from_f32(vTmp);
                vId++;
              }
            }
            vId += ne00 * (ne01 - cIr1);
          }
        }
      }
    } else {
      // printf("%s: this is not optimal - fix me\n", __func__);

      size_t vId = 0;
      dst_t *vDstPtr = (dst_t *)dst->data;

      for (int vI03 = 0; vI03 < ne03; vI03++) {
        for (int vI02 = 0; vI02 < ne02; vI02++) {
          vId += ne00 * cIr0;
          for (int vI01 = cIr0; vI01 < cIr1; vI01++) {
            for (int vI00 = 0; vI00 < ne00; vI00++) {
              const src_t *vSrc0Ptr =
                  (src_t *)((char *)src0->data + vI00 * nb00 + vI01 * nb01 + vI02 * nb02 + vI03 * nb03);

              float vTmp = type_conversion_table<src_t>::to_f32(*vSrc0Ptr);
              vDstPtr[vId] = type_conversion_table<dst_t>::from_f32(vTmp);
              vId++;
            }
          }
          vId += ne00 * (ne01 - cIr1);
        }
      }
    }
    return;
  }

  // dst counters
  int64_t vI10 = 0;
  int64_t vI11 = 0;
  int64_t vI12 = 0;
  int64_t vI13 = 0;

  if constexpr (std::is_same_v<dst_t, src_t>) {
    for (int64_t i03 = 0; i03 < ne03; i03++) {
      for (int64_t i02 = 0; i02 < ne02; i02++) {
        vI10 += ne00 * cIr0;
        while (vI10 >= ne0) {
          vI10 -= ne0;
          if (++vI11 == ne1) {
            vI11 = 0;
            if (++vI12 == ne2) {
              vI12 = 0;
              if (++vI13 == ne3) {
                vI13 = 0;
              }
            }
          }
        }
        for (int64_t i01 = cIr0; i01 < cIr1; i01++) {
          for (int64_t i00 = 0; i00 < ne00; i00++) {
            const char *src0_ptr = ((char *)src0->data + i00 * nb00 + i01 * nb01 + i02 * nb02 + i03 * nb03);
            char *dst_ptr = ((char *)dst->data + vI10 * nb0 + vI11 * nb1 + vI12 * nb2 + vI13 * nb3);

            memcpy(dst_ptr, src0_ptr, sizeof(dst_t));

            if (++vI10 == ne00) {
              vI10 = 0;
              if (++vI11 == ne01) {
                vI11 = 0;
                if (++vI12 == ne02) {
                  vI12 = 0;
                  if (++vI13 == ne03) {
                    vI13 = 0;
                  }
                }
              }
            }
          }
        }
        vI10 += ne00 * (ne01 - cIr1);
        while (vI10 >= ne0) {
          vI10 -= ne0;
          if (++vI11 == ne1) {
            vI11 = 0;
            if (++vI12 == ne2) {
              vI12 = 0;
              if (++vI13 == ne3) {
                vI13 = 0;
              }
            }
          }
        }
      }
    }

  } else {
    for (int64_t vI03 = 0; vI03 < ne03; vI03++) {
      for (int64_t vI02 = 0; vI02 < ne02; vI02++) {
        vI10 += ne00 * cIr0;
        while (vI10 >= ne0) {
          vI10 -= ne0;
          if (++vI11 == ne1) {
            vI11 = 0;
            if (++vI12 == ne2) {
              vI12 = 0;
              if (++vI13 == ne3) {
                vI13 = 0;
              }
            }
          }
        }
        for (int64_t vI01 = cIr0; vI01 < cIr1; vI01++) {
          for (int64_t vI00 = 0; vI00 < ne00; vI00++) {
            const char *vSrc0Ptr = ((char *)src0->data + vI00 * nb00 + vI01 * nb01 + vI02 * nb02 + vI03 * nb03);
            char *vDstPtr = ((char *)dst->data + vI10 * nb0 + vI11 * nb1 + vI12 * nb2 + vI13 * nb3);

            float vTmp = type_conversion_table<src_t>::to_f32(*(const src_t *)vSrc0Ptr);
            *(dst_t *)vDstPtr = type_conversion_table<dst_t>::from_f32(vTmp);

            if (++vI10 == ne0) {
              vI10 = 0;
              if (++vI11 == ne1) {
                vI11 = 0;
                if (++vI12 == ne2) {
                  vI12 = 0;
                  if (++vI13 == ne3) {
                    vI13 = 0;
                  }
                }
              }
            }
          }
        }
        vI10 += ne00 * (ne01 - cIr1);
        while (vI10 >= ne0) {
          vI10 -= ne0;
          if (++vI11 == ne1) {
            vI11 = 0;
            if (++vI12 == ne2) {
              vI12 = 0;
              if (++vI13 == ne3) {
                vI13 = 0;
              }
            }
          }
        }
      }
    }
  }
}

template <typename src_t> static void fGgmlComputeForwardDupToQ(const ggml_compute_params *pParams, ggml_tensor *dst) {

  const ggml_tensor *src0 = dst->src[0];

  GGML_ASSERT(ggml_nelements(dst) == ggml_nelements(src0));
  GGML_ASSERT(!ggml_is_quantized(src0->type));

  GGML_TENSOR_UNARY_OP_LOCALS

  const int cIth = pParams->ith; // thread index
  const int cNth = pParams->nth; // number of threads

  // parallelize by rows
  const int cNr = ne01;
  // number of rows per thread
  const int cDr = (cNr + cNth - 1) / cNth;
  // row range for this thread
  const int cIr0 = cDr * cIth;
  const int cIr1 = MIN(cIr0 + cDr, cNr);

  if (ggml_is_contiguous(dst) && nb00 == sizeof(src_t) && ggml_get_type_traits_cpu(dst->type)->from_float) {
    // casting non-quantized types --> intermediate f32 --> quantized
    ggml_from_float_t const cQuantizeRowQ = ggml_get_type_traits_cpu(dst->type)->from_float;
    float *vSrc0F32 = (float *)pParams->wdata + (ne00 + CACHE_LINE_SIZE_F32) * cIth;

    size_t vId = 0;
    size_t vRs = nb0 * (ne00 / ggml_blck_size(dst->type));
    char *vDstPtr = (char *)dst->data;

    for (int vI03 = 0; vI03 < ne03; vI03++) {
      for (int vI02 = 0; vI02 < ne02; vI02++) {
        vId += vRs * cIr0;
        for (int vI01 = cIr0; vI01 < cIr1; vI01++) {
          const src_t *vSrc0Ptr = (src_t *)((char *)src0->data + vI01 * nb01 + vI02 * nb02 + vI03 * nb03);

          for (int vI00 = 0; vI00 < ne00; vI00++) {
            vSrc0F32[vI00] = type_conversion_table<src_t>::to_f32(vSrc0Ptr[vI00]);
          }

          cQuantizeRowQ(vSrc0F32, vDstPtr + vId, ne00);
          vId += vRs;
        }
        vId += vRs * (ne01 - cIr1);
      }
    }
  } else {
    // printf("%s %s\n", ggml_type_name(src0->type), ggml_type_name(dst->type));
    GGML_ABORT("not implemented");
  }
}

// A simplified version of ggml_compute_forward_dup that doesn't do float upcasting, and just plain old memcpy.
static void fGgmlComputeForwardDupBytes(const ggml_compute_params *pParams, ggml_tensor *dst) {
  const ggml_tensor *src0 = dst->src[0];

  GGML_ASSERT(ggml_nelements(dst) == ggml_nelements(src0));
  GGML_ASSERT(src0->type == dst->type);

  GGML_TENSOR_UNARY_OP_LOCALS;

  if (ggml_is_contiguous(src0) && ggml_is_contiguous(dst)) {
    fGgmlComputeForwardDupSameCont(pParams, dst);
    return;
  }

  const size_t cTypeSize = ggml_type_size(src0->type);

  const int cIth = pParams->ith; // thread index
  const int cNth = pParams->nth; // number of threads

  // parallelize by rows
  const int cNr = ne01;
  // number of rows per thread
  const int cDr = (cNr + cNth - 1) / cNth;
  // row range for this thread
  const int cIr0 = cDr * cIth;
  const int cIr1 = MIN(cIr0 + cDr, cNr);

  if (src0->type == dst->type && ggml_are_same_shape(src0, dst) && nb00 == cTypeSize && nb0 == cTypeSize) {
    // copy by rows
    const size_t cRs = ggml_row_size(src0->type, ne00);
    for (int64_t vI03 = 0; vI03 < ne03; vI03++) {
      for (int64_t vI02 = 0; vI02 < ne02; vI02++) {
        for (int64_t vI01 = cIr0; vI01 < cIr1; vI01++) {
          memcpy(((char *)dst->data + vI01 * nb1 + vI02 * nb2 + vI03 * nb3),
                 ((char *)src0->data + vI01 * nb01 + vI02 * nb02 + vI03 * nb03), cRs);
        }
      }
    }
    return;
  }

  if (ggml_is_contiguous(dst)) {
    size_t vId = 0;
    char *vDstPtr = (char *)dst->data;
    const size_t cRs = ne00 * cTypeSize;

    if (nb00 == cTypeSize) {
      // src0 is contiguous on first dimension, copy by rows
      for (int64_t vI03 = 0; vI03 < ne03; vI03++) {
        for (int64_t vI02 = 0; vI02 < ne02; vI02++) {
          vId += cRs * cIr0;
          for (int64_t vI01 = cIr0; vI01 < cIr1; vI01++) {
            const char *vSrc0Ptr = (char *)src0->data + vI01 * nb01 + vI02 * nb02 + vI03 * nb03;
            memcpy(vDstPtr + vId, vSrc0Ptr, cRs);
            vId += cRs;
          }
          vId += cRs * (ne01 - cIr1);
        }
      }
    } else {
      // printf("%s: this is not optimal - fix me\n", __func__);

      for (int64_t vI03 = 0; vI03 < ne03; vI03++) {
        for (int64_t vI02 = 0; vI02 < ne02; vI02++) {
          vId += cRs * cIr0;
          for (int64_t vI01 = cIr0; vI01 < cIr1; vI01++) {
            for (int64_t vI00 = 0; vI00 < ne00; vI00++) {
              const char *vSrc0Ptr = (char *)src0->data + vI00 * nb00 + vI01 * nb01 + vI02 * nb02 + vI03 * nb03;
              memcpy(vDstPtr + vId, vSrc0Ptr, cTypeSize);

              vId += cTypeSize;
            }
          }
          vId += cRs * (ne01 - cIr1);
        }
      }
    }

    return;
  }

  // dst counters
  int64_t vK10 = 0;
  int64_t vI11 = 0;
  int64_t vI12 = 0;
  int64_t vI13 = 0;

  // number of blocks in a row
  const int64_t cNk00 = ne00 / ggml_blck_size(src0->type);
  const int64_t cNk0 = ne0 / ggml_blck_size(dst->type);

  for (int64_t vI03 = 0; vI03 < ne03; vI03++) {
    for (int64_t vI02 = 0; vI02 < ne02; vI02++) {
      vK10 += cNk00 * cIr0;
      while (vK10 >= cNk0) {
        vK10 -= cNk0;
        if (++vI11 == ne1) {
          vI11 = 0;
          if (++vI12 == ne2) {
            vI12 = 0;
            if (++vI13 == ne3) {
              vI13 = 0;
            }
          }
        }
      }
      for (int64_t vI01 = cIr0; vI01 < cIr1; vI01++) {
        for (int64_t vK00 = 0; vK00 < cNk00; vK00++) {
          const char *vSrc0Ptr = ((char *)src0->data + vK00 * nb00 + vI01 * nb01 + vI02 * nb02 + vI03 * nb03);
          char *vDstPtr = ((char *)dst->data + vK10 * nb0 + vI11 * nb1 + vI12 * nb2 + vI13 * nb3);

          memcpy(vDstPtr, vSrc0Ptr, cTypeSize);

          if (++vK10 == cNk0) {
            vK10 = 0;
            if (++vI11 == ne1) {
              vI11 = 0;
              if (++vI12 == ne2) {
                vI12 = 0;
                if (++vI13 == ne3) {
                  vI13 = 0;
                }
              }
            }
          }
        }
      }
      vK10 += cNk00 * (ne01 - cIr1);
      while (vK10 >= cNk0) {
        vK10 -= cNk0;
        if (++vI11 == ne1) {
          vI11 = 0;
          if (++vI12 == ne2) {
            vI12 = 0;
            if (++vI13 == ne3) {
              vI13 = 0;
            }
          }
        }
      }
    }
  }
}

static void fGgmlComputeForwardDupFromQ(const ggml_compute_params *pParams, ggml_tensor *dst) {

  const ggml_tensor *src0 = dst->src[0];
  const ggml_tensor *src1 = dst->src[1];

  GGML_TENSOR_BINARY_OP_LOCALS

  const ggml_type cType = src0->type;
  ggml_to_float_t const cDequantizeRowQ = ggml_get_type_traits(cType)->to_float;

  size_t vQk = ggml_blck_size(cType);
  const int64_t cNr = ggml_nelements(src1) / vQk;

  // destination must be contiguous in the first dimension
  GGML_ASSERT(nb10 == ggml_type_size(dst->type));
  // must either have first dimension large enough to hold a row, or fully contiguous
  GGML_ASSERT((ne10 % vQk) == 0 || ggml_is_contiguous(dst));

  const int cIth = pParams->ith;
  const int cNth = pParams->nth;

  const int cDr = (cNr + cNth - 1) / cNth;

  // row range for this thread
  const int cIr0 = cDr * cIth;
  const int cIr1 = MIN(cIr0 + cDr, cNr);

  for (int64_t vIr = cIr0; vIr < cIr1; ++vIr) {

    uint32_t vI = vIr * vQk;

    const int64_t cI03 = vI / (ne00 * ne01 * ne02);
    const int64_t cI02 = (vI - cI03 * ne00 * ne01 * ne02) / (ne00 * ne01);
    const int64_t cI01 = (vI - cI03 * ne00 * ne01 * ne02 - cI02 * ne01 * ne00) / ne00;
    const int64_t cI00 = vI - cI03 * ne00 * ne01 * ne02 - cI02 * ne01 * ne00 - cI01 * ne00;
    const int64_t cXOffset = (cI00 / vQk) * nb00 + cI01 * nb01 + cI02 * nb02 + cI03 * nb03;

    const int64_t cI13 = vI / (ne10 * ne11 * ne12);
    const int64_t cI12 = (vI - cI13 * ne10 * ne11 * ne12) / (ne10 * ne11);
    const int64_t cI11 = (vI - cI13 * ne10 * ne11 * ne12 - cI12 * ne10 * ne11) / ne10;
    const int64_t cI10 = vI - cI13 * ne10 * ne11 * ne12 - cI12 * ne10 * ne11 - cI11 * ne10;
    const int64_t cDstOffset = cI10 * nb10 + cI11 * nb11 + cI12 * nb12 + cI13 * nb13;

    cDequantizeRowQ((const void *)((char *)src0->data + cXOffset), (float *)((char *)dst->data + cDstOffset), vQk);
  }
}

void ggml_compute_forward_dup(const ggml_compute_params *pParams, ggml_tensor *pDst) {

  const ggml_tensor *vSrc0 = pDst->src[0];

  if (vSrc0->type == pDst->type) {
    fGgmlComputeForwardDupBytes(pParams, pDst);
    return;
  }

  switch (vSrc0->type) {
  case GGML_TYPE_F16: {
    /**/ if (pDst->type == GGML_TYPE_F16)
      fGgmlComputeForwardDupFlt<ggml_fp16_t, ggml_fp16_t>(pParams, pDst);
    else if (pDst->type == GGML_TYPE_BF16)
      fGgmlComputeForwardDupFlt<ggml_fp16_t, ggml_bf16_t>(pParams, pDst);
    else if (pDst->type == GGML_TYPE_F32)
      fGgmlComputeForwardDupFlt<ggml_fp16_t, float>(pParams, pDst);
    else
      fGgmlComputeForwardDupToQ<ggml_fp16_t>(pParams, pDst);
  } break;
  case GGML_TYPE_BF16: {
    /**/ if (pDst->type == GGML_TYPE_F16)
      fGgmlComputeForwardDupFlt<ggml_bf16_t, ggml_fp16_t>(pParams, pDst);
    else if (pDst->type == GGML_TYPE_BF16)
      fGgmlComputeForwardDupFlt<ggml_bf16_t, ggml_bf16_t>(pParams, pDst);
    else if (pDst->type == GGML_TYPE_F32)
      fGgmlComputeForwardDupFlt<ggml_bf16_t, float>(pParams, pDst);
    else
      fGgmlComputeForwardDupToQ<ggml_bf16_t>(pParams, pDst);
  } break;
  case GGML_TYPE_F32: {
    /**/ if (pDst->type == GGML_TYPE_F16)
      fGgmlComputeForwardDupFlt<float, ggml_fp16_t>(pParams, pDst);
    else if (pDst->type == GGML_TYPE_BF16)
      fGgmlComputeForwardDupFlt<float, ggml_bf16_t>(pParams, pDst);
    else if (pDst->type == GGML_TYPE_F32)
      fGgmlComputeForwardDupFlt<float, float>(pParams, pDst);
    else if (pDst->type == GGML_TYPE_I32)
      fGgmlComputeForwardDupFlt<float, int32_t>(pParams, pDst);
    else
      fGgmlComputeForwardDupToQ<float>(pParams, pDst);
  } break;
  case GGML_TYPE_I32: {
    if (pDst->type == GGML_TYPE_F32)
      fGgmlComputeForwardDupFlt<int32_t, float>(pParams, pDst);
    else
      GGML_ABORT("not implemented");
  } break;
  default: {
    if (ggml_is_quantized(vSrc0->type) && pDst->type == GGML_TYPE_F32) {
      fGgmlComputeForwardDupFromQ(pParams, pDst);
      break;
    }
    GGML_ABORT("fatal error");
  }
  }
}

// ggml_compute_forward_add

static void fGgmlComputeForwardAddQF32(const ggml_compute_params *pParams, ggml_tensor *dst) {

  const ggml_tensor *src0 = dst->src[0];
  const ggml_tensor *src1 = dst->src[1];

  GGML_ASSERT(ggml_are_same_shape(src0, src1) && ggml_are_same_shape(src0, dst));

  const int cNr = ggml_nrows(src0);

  GGML_TENSOR_BINARY_OP_LOCALS

  const int cIth = pParams->ith;
  const int cNth = pParams->nth;

  const ggml_type cType = src0->type;
  const ggml_type cDtype = dst->type;
  ggml_to_float_t const cDequantizeRowQ = ggml_get_type_traits(cType)->to_float;
  ggml_from_float_t const cQuantizeRowQ = ggml_get_type_traits_cpu(cDtype)->from_float;

  // we don't support permuted src0 or src1
  GGML_ASSERT(nb00 == ggml_type_size(cType));
  GGML_ASSERT(nb10 == sizeof(float));

  // dst cannot be transposed or permuted
  GGML_ASSERT(nb0 <= nb1);
  GGML_ASSERT(nb1 <= nb2);
  GGML_ASSERT(nb2 <= nb3);

  GGML_ASSERT(ggml_is_quantized(src0->type));
  GGML_ASSERT(src1->type == GGML_TYPE_F32);

  // rows per thread
  const int cDr = (cNr + cNth - 1) / cNth;

  // row range for this thread
  const int cIr0 = cDr * cIth;
  const int cIr1 = MIN(cIr0 + cDr, cNr);

  float *vWdata = (float *)pParams->wdata + (ne00 + CACHE_LINE_SIZE_F32) * cIth;

  for (int vIr = cIr0; vIr < cIr1; ++vIr) {
    // src0 indices
    const int cI03 = vIr / (ne02 * ne01);
    const int cI02 = (vIr - cI03 * ne02 * ne01) / ne01;
    const int cI01 = (vIr - cI03 * ne02 * ne01 - cI02 * ne01);

    // src1 and dst are same shape as src0 => same indices
    const int cI13 = cI03;
    const int cI12 = cI02;
    const int cI11 = cI01;

    const int cI3 = cI03;
    const int cI2 = cI02;
    const int cI1 = cI01;

    void *vSrc0Row = (void *)((char *)src0->data + (cI01 * nb01 + cI02 * nb02 + cI03 * nb03));
    float *vSrc1Row = (float *)((char *)src1->data + (cI11 * nb11 + cI12 * nb12 + cI13 * nb13));
    void *vDstRow = (void *)((char *)dst->data + (cI1 * nb1 + cI2 * nb2 + cI3 * nb3));

    assert(ne00 % 32 == 0);

    // unquantize row from src0 to temp buffer
    cDequantizeRowQ(vSrc0Row, vWdata, ne00);
    // add src1
    ggml_vec_acc_f32(ne00, vWdata, vSrc1Row);
    // quantize row to dst
    if (cQuantizeRowQ != NULL) {
      cQuantizeRowQ(vWdata, vDstRow, ne00);
    } else {
      memcpy(vDstRow, vWdata, ne0 * nb0);
    }
  }
}

void ggml_compute_forward_add(const ggml_compute_params *pParams, ggml_tensor *pDst) {

  const ggml_tensor *vSrc0 = pDst->src[0];

  switch (vSrc0->type) {
  case GGML_TYPE_F32:
  case GGML_TYPE_F16:
  case GGML_TYPE_BF16: {
    ggml_compute_forward_add_non_quantized(pParams, pDst);
  } break;
  case GGML_TYPE_Q1_0:
  case GGML_TYPE_Q2_0:
  case GGML_TYPE_Q4_0:
  case GGML_TYPE_Q4_1:
  case GGML_TYPE_Q5_0:
  case GGML_TYPE_Q5_1:
  case GGML_TYPE_Q8_0:
  case GGML_TYPE_MXFP4:
  case GGML_TYPE_NVFP4:
  case GGML_TYPE_Q2_K:
  case GGML_TYPE_Q3_K:
  case GGML_TYPE_Q4_K:
  case GGML_TYPE_Q5_K:
  case GGML_TYPE_Q6_K:
  case GGML_TYPE_TQ1_0:
  case GGML_TYPE_TQ2_0:
  case GGML_TYPE_IQ2_XXS:
  case GGML_TYPE_IQ2_XS:
  case GGML_TYPE_IQ3_XXS:
  case GGML_TYPE_IQ1_S:
  case GGML_TYPE_IQ1_M:
  case GGML_TYPE_IQ4_NL:
  case GGML_TYPE_IQ4_XS:
  case GGML_TYPE_IQ3_S:
  case GGML_TYPE_IQ2_S: {
    fGgmlComputeForwardAddQF32(pParams, pDst);
  } break;
  default: {
    GGML_ABORT("fatal error");
  }
  }
}

// ggml_compute_forward_add_id

static void fGgmlComputeForwardAddIdF32(const ggml_compute_params *pParams, ggml_tensor *dst) {

  const ggml_tensor *src0 = dst->src[0];
  const ggml_tensor *src1 = dst->src[1];
  const ggml_tensor *src2 = dst->src[2];

  GGML_ASSERT(dst->type == GGML_TYPE_F32);
  GGML_ASSERT(src0->type == GGML_TYPE_F32);
  GGML_ASSERT(src1->type == GGML_TYPE_F32);
  GGML_ASSERT(src2->type == GGML_TYPE_I32);

  GGML_ASSERT(src0->nb[0] == sizeof(float));
  GGML_ASSERT(src1->nb[0] == sizeof(float));

  const int cIth = pParams->ith;
  const int cNth = pParams->nth;

  const int cNr = ggml_nrows(src0);

  GGML_TENSOR_TERNARY_OP_LOCALS

  GGML_ASSERT(nb0 == sizeof(float));
  GGML_ASSERT(nb10 == sizeof(float));

  // rows per thread
  const int cDr = (cNr + cNth - 1) / cNth;

  // row range for this thread
  const int cIr0 = cDr * cIth;
  const int cIr1 = MIN(cIr0 + cDr, cNr);

  for (int vIr = cIr0; vIr < cIr1; ++vIr) {
    // src0 indices
    const int cI3 = vIr / (ne2 * ne1);
    const int cI2 = (vIr - cI3 * ne2 * ne1) / ne1;
    const int cI1 = (vIr - cI3 * ne2 * ne1 - cI2 * ne1);

    // src1 indices
    const int cI11 = *(int32_t *)((char *)src2->data + cI1 * nb20 + cI2 * nb21);

    // A negative identifier marks an expert slot computed by another backend
    // (hybrid MoE execution): keep the row unchanged.
    if (cI11 < 0) {
      if (dst->data != src0->data) {
        memcpy((char *)dst->data + cI3 * nb3 + cI2 * nb2 + cI1 * nb1,
               (char *)src0->data + cI3 * nb03 + cI2 * nb02 + cI1 * nb01, ne0 * sizeof(float));
      }
      continue;
    }

    GGML_ASSERT(cI11 < ne11);

    ggml_vec_add_f32(ne0, (float *)((char *)dst->data + cI3 * nb3 + cI2 * nb2 + cI1 * nb1),
                     (float *)((char *)src0->data + cI3 * nb03 + cI2 * nb02 + cI1 * nb01),
                     (float *)((char *)src1->data + cI11 * nb11));
  }
}

void ggml_compute_forward_add_id(const ggml_compute_params *pParams, ggml_tensor *pDst) {

  const ggml_tensor *vSrc0 = pDst->src[0];

  switch (vSrc0->type) {
  case GGML_TYPE_F32: {
    fGgmlComputeForwardAddIdF32(pParams, pDst);
  } break;
  default: {
    GGML_ABORT("unsupported type for ggml_compute_forward_add_id: %s", ggml_type_name(vSrc0->type));
  }
  }
}

// ggml_compute_forward_add1

static void fGgmlComputeForwardAdd1F32(const ggml_compute_params *pParams, ggml_tensor *dst) {

  const ggml_tensor *src0 = dst->src[0];
  const ggml_tensor *vSrc1 = dst->src[1];

  GGML_ASSERT(ggml_are_same_shape(src0, dst));
  GGML_ASSERT(ggml_is_scalar(vSrc1));

  const int cIth = pParams->ith;
  const int cNth = pParams->nth;

  const int cNr = ggml_nrows(src0);

  GGML_TENSOR_UNARY_OP_LOCALS

  GGML_ASSERT(nb0 == sizeof(float));
  GGML_ASSERT(nb00 == sizeof(float));

  // rows per thread
  const int cDr = (cNr + cNth - 1) / cNth;

  // row range for this thread
  const int cIr0 = cDr * cIth;
  const int cIr1 = MIN(cIr0 + cDr, cNr);

  for (int vIr = cIr0; vIr < cIr1; ++vIr) {
    // src0 and dst are same shape => same indices
    const int cI3 = vIr / (ne2 * ne1);
    const int cI2 = (vIr - cI3 * ne2 * ne1) / ne1;
    const int cI1 = (vIr - cI3 * ne2 * ne1 - cI2 * ne1);

#ifdef GGML_USE_ACCELERATE
    GGML_UNUSED(ggml_vec_add1_f32);

    vDSP_vadd((float *)((char *)src0->data + cI3 * nb03 + cI2 * nb02 + cI1 * nb01), 1, (float *)((char *)vSrc1->data),
              0, (float *)((char *)dst->data + cI3 * nb3 + cI2 * nb2 + cI1 * nb1), 1, ne0);
#else
    ggml_vec_add1_f32(ne0, (float *)((char *)dst->data + cI3 * nb3 + cI2 * nb2 + cI1 * nb1),
                      (float *)((char *)src0->data + cI3 * nb03 + cI2 * nb02 + cI1 * nb01), *(float *)vSrc1->data);
#endif
  }
}

static void fGgmlComputeForwardAdd1F16F32(const ggml_compute_params *pParams, ggml_tensor *dst) {

  const ggml_tensor *src0 = dst->src[0];
  const ggml_tensor *vSrc1 = dst->src[1];

  GGML_ASSERT(ggml_are_same_shape(src0, dst));
  GGML_ASSERT(ggml_is_scalar(vSrc1));

  // scalar to add
  const float cV = *(float *)vSrc1->data;

  const int cIth = pParams->ith;
  const int cNth = pParams->nth;

  const int cNr = ggml_nrows(src0);

  GGML_TENSOR_UNARY_OP_LOCALS

  GGML_ASSERT(src0->type == GGML_TYPE_F16);
  GGML_ASSERT(vSrc1->type == GGML_TYPE_F32);
  GGML_ASSERT(dst->type == GGML_TYPE_F16);

  GGML_ASSERT(nb0 == sizeof(ggml_fp16_t));
  GGML_ASSERT(nb00 == sizeof(ggml_fp16_t));

  // rows per thread
  const int cDr = (cNr + cNth - 1) / cNth;

  // row range for this thread
  const int cIr0 = cDr * cIth;
  const int cIr1 = MIN(cIr0 + cDr, cNr);

  for (int vIr = cIr0; vIr < cIr1; ++vIr) {
    // src0 and dst are same shape => same indices
    const int cI3 = vIr / (ne2 * ne1);
    const int cI2 = (vIr - cI3 * ne2 * ne1) / ne1;
    const int cI1 = (vIr - cI3 * ne2 * ne1 - cI2 * ne1);

    ggml_fp16_t *vDstPtr = (ggml_fp16_t *)((char *)dst->data + cI3 * nb3 + cI2 * nb2 + cI1 * nb1);
    ggml_fp16_t *vSrc0Ptr = (ggml_fp16_t *)((char *)src0->data + cI3 * nb03 + cI2 * nb02 + cI1 * nb01);
    for (int vI = 0; vI < ne0; vI++) {
      vDstPtr[vI] = GGML_CPU_FP32_TO_FP16(GGML_CPU_FP16_TO_FP32(vSrc0Ptr[vI]) + cV);
    }
  }
}

static void fGgmlComputeForwardAdd1F16F16(const ggml_compute_params *pParams, ggml_tensor *dst) {

  const ggml_tensor *src0 = dst->src[0];
  const ggml_tensor *vSrc1 = dst->src[1];

  GGML_ASSERT(ggml_are_same_shape(src0, dst));
  GGML_ASSERT(ggml_is_scalar(vSrc1));

  // scalar to add
  const float cV = GGML_CPU_FP16_TO_FP32(*(ggml_fp16_t *)vSrc1->data);

  const int cIth = pParams->ith;
  const int cNth = pParams->nth;

  const int cNr = ggml_nrows(src0);

  GGML_TENSOR_UNARY_OP_LOCALS

  GGML_ASSERT(src0->type == GGML_TYPE_F16);
  GGML_ASSERT(vSrc1->type == GGML_TYPE_F16);
  GGML_ASSERT(dst->type == GGML_TYPE_F16);

  GGML_ASSERT(nb0 == sizeof(ggml_fp16_t));
  GGML_ASSERT(nb00 == sizeof(ggml_fp16_t));

  // rows per thread
  const int cDr = (cNr + cNth - 1) / cNth;

  // row range for this thread
  const int cIr0 = cDr * cIth;
  const int cIr1 = MIN(cIr0 + cDr, cNr);

  for (int vIr = cIr0; vIr < cIr1; ++vIr) {
    // src0 and dst are same shape => same indices
    const int cI3 = vIr / (ne2 * ne1);
    const int cI2 = (vIr - cI3 * ne2 * ne1) / ne1;
    const int cI1 = (vIr - cI3 * ne2 * ne1 - cI2 * ne1);

    ggml_fp16_t *vDstPtr = (ggml_fp16_t *)((char *)dst->data + cI3 * nb3 + cI2 * nb2 + cI1 * nb1);
    ggml_fp16_t *vSrc0Ptr = (ggml_fp16_t *)((char *)src0->data + cI3 * nb03 + cI2 * nb02 + cI1 * nb01);
    for (int vI = 0; vI < ne0; vI++) {
      vDstPtr[vI] = GGML_CPU_FP32_TO_FP16(GGML_CPU_FP16_TO_FP32(vSrc0Ptr[vI]) + cV);
    }
  }
}

static void fGgmlComputeForwardAdd1QF32(const ggml_compute_params *pParams, ggml_tensor *dst) {

  const ggml_tensor *src0 = dst->src[0];
  const ggml_tensor *vSrc1 = dst->src[1];

  GGML_ASSERT(ggml_are_same_shape(src0, dst));
  GGML_ASSERT(ggml_is_scalar(vSrc1));

  // scalar to add
  const float cV = *(float *)vSrc1->data;

  const int cIth = pParams->ith;
  const int cNth = pParams->nth;

  const int cNr = ggml_nrows(src0);

  GGML_TENSOR_UNARY_OP_LOCALS

  const ggml_type cType = src0->type;
  ggml_to_float_t const cDequantizeRowQ = ggml_get_type_traits(cType)->to_float;
  ggml_from_float_t const cQuantizeRowQ = ggml_get_type_traits_cpu(cType)->from_float;

  // we don't support permuted src0
  GGML_ASSERT(nb00 == ggml_type_size(cType));

  // dst cannot be transposed or permuted
  GGML_ASSERT(nb0 <= nb1);
  GGML_ASSERT(nb1 <= nb2);
  GGML_ASSERT(nb2 <= nb3);

  GGML_ASSERT(ggml_is_quantized(src0->type));
  GGML_ASSERT(dst->type == src0->type);
  GGML_ASSERT(vSrc1->type == GGML_TYPE_F32);

  // rows per thread
  const int cDr = (cNr + cNth - 1) / cNth;

  // row range for this thread
  const int cIr0 = cDr * cIth;
  const int cIr1 = MIN(cIr0 + cDr, cNr);

  float *vWdata = (float *)pParams->wdata + (ne0 + CACHE_LINE_SIZE_F32) * cIth;

  for (int vIr = cIr0; vIr < cIr1; ++vIr) {
    // src0 and dst are same shape => same indices
    const int cI3 = vIr / (ne2 * ne1);
    const int cI2 = (vIr - cI3 * ne2 * ne1) / ne1;
    const int cI1 = (vIr - cI3 * ne2 * ne1 - cI2 * ne1);

    void *vSrc0Row = (void *)((char *)src0->data + (cI1 * nb01 + cI2 * nb02 + cI3 * nb03));
    void *vDstRow = (void *)((char *)dst->data + (cI1 * nb1 + cI2 * nb2 + cI3 * nb0));

    assert(ne0 % 32 == 0);

    // unquantize row from src0 to temp buffer
    cDequantizeRowQ(vSrc0Row, vWdata, ne0);
    // add src1
    ggml_vec_acc1_f32(ne0, vWdata, cV);
    // quantize row to dst
    cQuantizeRowQ(vWdata, vDstRow, ne0);
  }
}

static void fGgmlComputeForwardAdd1Bf16F32(const ggml_compute_params *pParams, ggml_tensor *dst) {

  const ggml_tensor *src0 = dst->src[0];
  const ggml_tensor *vSrc1 = dst->src[1];

  GGML_ASSERT(ggml_are_same_shape(src0, dst));
  GGML_ASSERT(ggml_is_scalar(vSrc1));

  // scalar to add
  const float cV = *(float *)vSrc1->data;

  const int cIth = pParams->ith;
  const int cNth = pParams->nth;

  const int cNr = ggml_nrows(src0);

  GGML_TENSOR_UNARY_OP_LOCALS

  GGML_ASSERT(src0->type == GGML_TYPE_BF16);
  GGML_ASSERT(vSrc1->type == GGML_TYPE_F32);
  GGML_ASSERT(dst->type == GGML_TYPE_BF16);

  GGML_ASSERT(nb0 == sizeof(ggml_bf16_t));
  GGML_ASSERT(nb00 == sizeof(ggml_bf16_t));

  // rows per thread
  const int cDr = (cNr + cNth - 1) / cNth;

  // row range for this thread
  const int cIr0 = cDr * cIth;
  const int cIr1 = MIN(cIr0 + cDr, cNr);

  for (int vIr = cIr0; vIr < cIr1; ++vIr) {
    // src0 and dst are same shape => same indices
    const int cI3 = vIr / (ne2 * ne1);
    const int cI2 = (vIr - cI3 * ne2 * ne1) / ne1;
    const int cI1 = (vIr - cI3 * ne2 * ne1 - cI2 * ne1);

    ggml_bf16_t *vDstPtr = (ggml_bf16_t *)((char *)dst->data + cI3 * nb3 + cI2 * nb2 + cI1 * nb1);
    ggml_bf16_t *vSrc0Ptr = (ggml_bf16_t *)((char *)src0->data + cI3 * nb03 + cI2 * nb02 + cI1 * nb01);
    for (int vI = 0; vI < ne0; vI++) {
      vDstPtr[vI] = GGML_FP32_TO_BF16(GGML_BF16_TO_FP32(vSrc0Ptr[vI]) + cV);
    }
  }
}

static void fGgmlComputeForwardAdd1Bf16Bf16(const ggml_compute_params *pParams, ggml_tensor *dst) {

  const ggml_tensor *src0 = dst->src[0];
  const ggml_tensor *vSrc1 = dst->src[1];

  GGML_ASSERT(ggml_are_same_shape(src0, dst));
  GGML_ASSERT(ggml_is_scalar(vSrc1));

  // scalar to add
  const float cV = GGML_BF16_TO_FP32(*(ggml_bf16_t *)vSrc1->data);

  const int cIth = pParams->ith;
  const int cNth = pParams->nth;

  const int cNr = ggml_nrows(src0);

  GGML_TENSOR_UNARY_OP_LOCALS

  GGML_ASSERT(src0->type == GGML_TYPE_BF16);
  GGML_ASSERT(vSrc1->type == GGML_TYPE_BF16);
  GGML_ASSERT(dst->type == GGML_TYPE_BF16);

  GGML_ASSERT(nb0 == sizeof(ggml_bf16_t));
  GGML_ASSERT(nb00 == sizeof(ggml_bf16_t));

  // rows per thread
  const int cDr = (cNr + cNth - 1) / cNth;

  // row range for this thread
  const int cIr0 = cDr * cIth;
  const int cIr1 = MIN(cIr0 + cDr, cNr);

  for (int vIr = cIr0; vIr < cIr1; ++vIr) {
    // src0 and dst are same shape => same indices
    const int cI3 = vIr / (ne2 * ne1);
    const int cI2 = (vIr - cI3 * ne2 * ne1) / ne1;
    const int cI1 = (vIr - cI3 * ne2 * ne1 - cI2 * ne1);

    ggml_bf16_t *vDstPtr = (ggml_bf16_t *)((char *)dst->data + cI3 * nb3 + cI2 * nb2 + cI1 * nb1);
    ggml_bf16_t *vSrc0Ptr = (ggml_bf16_t *)((char *)src0->data + cI3 * nb03 + cI2 * nb02 + cI1 * nb01);
    for (int vI = 0; vI < ne0; vI++) {
      vDstPtr[vI] = GGML_FP32_TO_BF16(GGML_BF16_TO_FP32(vSrc0Ptr[vI]) + cV);
    }
  }
}

void ggml_compute_forward_add1(const ggml_compute_params *pParams, ggml_tensor *pDst) {

  const ggml_tensor *vSrc0 = pDst->src[0];
  const ggml_tensor *vSrc1 = pDst->src[1];

  switch (vSrc0->type) {
  case GGML_TYPE_F32: {
    fGgmlComputeForwardAdd1F32(pParams, pDst);
  } break;
  case GGML_TYPE_F16: {
    if (vSrc1->type == GGML_TYPE_F16) {
      fGgmlComputeForwardAdd1F16F16(pParams, pDst);
    } else if (vSrc1->type == GGML_TYPE_F32) {
      fGgmlComputeForwardAdd1F16F32(pParams, pDst);
    } else {
      GGML_ABORT("fatal error");
    }
  } break;
  case GGML_TYPE_BF16: {
    if (vSrc1->type == GGML_TYPE_BF16) {
      fGgmlComputeForwardAdd1Bf16Bf16(pParams, pDst);
    } else if (vSrc1->type == GGML_TYPE_F32) {
      fGgmlComputeForwardAdd1Bf16F32(pParams, pDst);
    } else {
      GGML_ABORT("fatal error");
    }
  } break;
  case GGML_TYPE_Q1_0:
  case GGML_TYPE_Q2_0:
  case GGML_TYPE_Q4_0:
  case GGML_TYPE_Q4_1:
  case GGML_TYPE_Q5_0:
  case GGML_TYPE_Q5_1:
  case GGML_TYPE_Q8_0:
  case GGML_TYPE_Q8_1:
  case GGML_TYPE_MXFP4:
  case GGML_TYPE_NVFP4:
  case GGML_TYPE_Q2_K:
  case GGML_TYPE_Q3_K:
  case GGML_TYPE_Q4_K:
  case GGML_TYPE_Q5_K:
  case GGML_TYPE_Q6_K:
  case GGML_TYPE_TQ1_0:
  case GGML_TYPE_TQ2_0:
  case GGML_TYPE_IQ2_XXS:
  case GGML_TYPE_IQ2_XS:
  case GGML_TYPE_IQ3_XXS:
  case GGML_TYPE_IQ1_S:
  case GGML_TYPE_IQ1_M:
  case GGML_TYPE_IQ4_NL:
  case GGML_TYPE_IQ4_XS:
  case GGML_TYPE_IQ3_S:
  case GGML_TYPE_IQ2_S: {
    fGgmlComputeForwardAdd1QF32(pParams, pDst);
  } break;
  default: {
    GGML_ABORT("fatal error");
  }
  }
}

// ggml_compute_forward_acc

static void fGgmlComputeForwardAccF32(const ggml_compute_params *pParams, ggml_tensor *pDst) {

  const ggml_tensor *vSrc0 = pDst->src[0];
  const ggml_tensor *vSrc1 = pDst->src[1];

  GGML_ASSERT(ggml_are_same_shape(vSrc0, pDst));
  GGML_ASSERT(ggml_is_contiguous(pDst) && ggml_is_contiguous(vSrc0));

  // view src0 and dst with these strides and data offset inbytes during acc
  // nb0 is implicitly element_size because src0 and dst are contiguous
  size_t nb1 = ((int32_t *)pDst->op_params)[0];
  size_t vNb2 = ((int32_t *)pDst->op_params)[1];
  size_t vNb3 = ((int32_t *)pDst->op_params)[2];
  size_t vOffset = ((int32_t *)pDst->op_params)[3];
  bool vInplace = (bool)((int32_t *)pDst->op_params)[4];

  if (!vInplace) {
    if (pParams->ith == 0) {
      // memcpy needs to be synchronized across threads to avoid race conditions.
      // => do it in INIT phase
      memcpy(((char *)pDst->data), ((char *)vSrc0->data), ggml_nbytes(pDst));
    }
    ggml_barrier(pParams->threadpool);
  }

  const int cIth = pParams->ith;
  const int cNth = pParams->nth;

  const int cNr = ggml_nrows(vSrc1);
  const int cNc = vSrc1->ne[0];

  GGML_TENSOR_LOCALS(int64_t, ne1, vSrc1, ne)
  GGML_TENSOR_LOCALS(size_t, nb1, vSrc1, nb)

  // src0 and dst as viewed during acc
  const size_t cNb0 = ggml_element_size(vSrc0);

  const size_t cNb00 = cNb0;
  const size_t cNb01 = nb1;
  const size_t cNb02 = vNb2;
  const size_t cNb03 = vNb3;

  GGML_ASSERT(vOffset + (ne10 == 0 ? 0 : ne10 - 1) * cNb0 + (ne11 == 0 ? 0 : ne11 - 1) * nb1 +
                  (ne12 == 0 ? 0 : ne12 - 1) * vNb2 + (ne13 == 0 ? 0 : ne13 - 1) * vNb3 <
              ggml_nbytes(pDst));
  GGML_ASSERT(vOffset + (ne10 == 0 ? 0 : ne10 - 1) * cNb00 + (ne11 == 0 ? 0 : ne11 - 1) * cNb01 +
                  (ne12 == 0 ? 0 : ne12 - 1) * cNb02 + (ne13 == 0 ? 0 : ne13 - 1) * cNb03 <
              ggml_nbytes(vSrc0));

  GGML_ASSERT(nb10 == sizeof(float));

  // rows per thread
  const int cDr = (cNr + cNth - 1) / cNth;

  // row range for this thread
  const int cIr0 = cDr * cIth;
  const int cIr1 = MIN(cIr0 + cDr, cNr);

  for (int vIr = cIr0; vIr < cIr1; ++vIr) {
    // src0 and dst are viewed with shape of src1 and offset
    // => same indices
    const int cI3 = vIr / (ne12 * ne11);
    const int cI2 = (vIr - cI3 * ne12 * ne11) / ne11;
    const int cI1 = (vIr - cI3 * ne12 * ne11 - cI2 * ne11);

#ifdef GGML_USE_ACCELERATE
    vDSP_vadd((float *)((char *)vSrc0->data + cI3 * cNb03 + cI2 * cNb02 + cI1 * cNb01 + vOffset), 1,
              (float *)((char *)vSrc1->data + cI3 * nb13 + cI2 * nb12 + cI1 * nb11), 1,
              (float *)((char *)pDst->data + cI3 * vNb3 + cI2 * vNb2 + cI1 * nb1 + vOffset), 1, cNc);
#else
    ggml_vec_add_f32(cNc, (float *)((char *)pDst->data + cI3 * vNb3 + cI2 * vNb2 + cI1 * nb1 + vOffset),
                     (float *)((char *)vSrc0->data + cI3 * cNb03 + cI2 * cNb02 + cI1 * cNb01 + vOffset),
                     (float *)((char *)vSrc1->data + cI3 * nb13 + cI2 * nb12 + cI1 * nb11));
#endif
  }
}

void ggml_compute_forward_acc(const ggml_compute_params *pParams, ggml_tensor *pDst) {

  const ggml_tensor *vSrc0 = pDst->src[0];

  switch (vSrc0->type) {
  case GGML_TYPE_F32: {
    fGgmlComputeForwardAccF32(pParams, pDst);
  } break;
  case GGML_TYPE_F16:
  case GGML_TYPE_BF16:
  case GGML_TYPE_Q1_0:
  case GGML_TYPE_Q2_0:
  case GGML_TYPE_Q4_0:
  case GGML_TYPE_Q4_1:
  case GGML_TYPE_Q5_0:
  case GGML_TYPE_Q5_1:
  case GGML_TYPE_Q8_0:
  case GGML_TYPE_Q8_1:
  case GGML_TYPE_MXFP4:
  case GGML_TYPE_NVFP4:
  case GGML_TYPE_Q2_K:
  case GGML_TYPE_Q3_K:
  case GGML_TYPE_Q4_K:
  case GGML_TYPE_Q5_K:
  case GGML_TYPE_Q6_K:
  case GGML_TYPE_TQ1_0:
  case GGML_TYPE_TQ2_0:
  case GGML_TYPE_IQ2_XXS:
  case GGML_TYPE_IQ2_XS:
  case GGML_TYPE_IQ3_XXS:
  case GGML_TYPE_IQ1_S:
  case GGML_TYPE_IQ1_M:
  case GGML_TYPE_IQ4_NL:
  case GGML_TYPE_IQ4_XS:
  case GGML_TYPE_IQ3_S:
  case GGML_TYPE_IQ2_S:
  default: {
    GGML_ABORT("fatal error");
  }
  }
}

// ggml_compute_forward_sum

static void fGgmlComputeForwardSumF32(const ggml_compute_params *pParams, ggml_tensor *pDst) {

  const ggml_tensor *vSrc0 = pDst->src[0];

  if (pParams->ith != 0) {
    return;
  }

  assert(ggml_is_scalar(pDst));
  assert(vSrc0->nb[0] == sizeof(float));

  GGML_TENSOR_LOCALS(int64_t, ne0, vSrc0, ne)
  GGML_TENSOR_LOCALS(size_t, nb0, vSrc0, nb)

  ggml_float vSum = 0;
  ggml_float vRowSum = 0;

  for (int64_t vI03 = 0; vI03 < ne03; vI03++) {
    for (int64_t vI02 = 0; vI02 < ne02; vI02++) {
      for (int64_t vI01 = 0; vI01 < ne01; vI01++) {
        ggml_vec_sum_f32_ggf(ne00, &vRowSum, (float *)((char *)vSrc0->data + vI01 * nb01 + vI02 * nb02 + vI03 * nb03));
        vSum += vRowSum;
      }
    }
  }
  ((float *)pDst->data)[0] = vSum;
}

static void fGgmlComputeForwardSumF16(const ggml_compute_params *pParams, ggml_tensor *pDst) {

  const ggml_tensor *vSrc0 = pDst->src[0];

  if (pParams->ith != 0) {
    return;
  }

  assert(ggml_is_scalar(pDst));

  assert(vSrc0->nb[0] == sizeof(ggml_fp16_t));

  GGML_TENSOR_LOCALS(int64_t, ne0, vSrc0, ne)
  GGML_TENSOR_LOCALS(size_t, nb0, vSrc0, nb)

  float vSum = 0;
  float vRowSum = 0;

  for (int64_t vI03 = 0; vI03 < ne03; vI03++) {
    for (int64_t vI02 = 0; vI02 < ne02; vI02++) {
      for (int64_t vI01 = 0; vI01 < ne01; vI01++) {
        ggml_vec_sum_f16_ggf(ne00, &vRowSum,
                             (ggml_fp16_t *)((char *)vSrc0->data + vI01 * nb01 + vI02 * nb02 + vI03 * nb03));
        vSum += vRowSum;
      }
    }
  }
  ((ggml_fp16_t *)pDst->data)[0] = GGML_CPU_FP32_TO_FP16(vSum);
}

static void fGgmlComputeForwardSumBf16(const ggml_compute_params *pParams, ggml_tensor *pDst) {

  const ggml_tensor *vSrc0 = pDst->src[0];

  if (pParams->ith != 0) {
    return;
  }

  assert(ggml_is_scalar(pDst));

  assert(vSrc0->nb[0] == sizeof(ggml_bf16_t));

  GGML_TENSOR_LOCALS(int64_t, ne0, vSrc0, ne)
  GGML_TENSOR_LOCALS(size_t, nb0, vSrc0, nb)

  float vSum = 0;
  float vRowSum = 0;

  for (int64_t vI03 = 0; vI03 < ne03; vI03++) {
    for (int64_t vI02 = 0; vI02 < ne02; vI02++) {
      for (int64_t vI01 = 0; vI01 < ne01; vI01++) {
        ggml_vec_sum_bf16_ggf(ne00, &vRowSum,
                              (ggml_bf16_t *)((char *)vSrc0->data + vI01 * nb01 + vI02 * nb02 + vI03 * nb03));
        vSum += vRowSum;
      }
    }
  }
  ((ggml_bf16_t *)pDst->data)[0] = GGML_FP32_TO_BF16(vSum);
}

void ggml_compute_forward_sum(const ggml_compute_params *pParams, ggml_tensor *pDst) {

  const ggml_tensor *vSrc0 = pDst->src[0];

  switch (vSrc0->type) {
  case GGML_TYPE_F32: {
    fGgmlComputeForwardSumF32(pParams, pDst);
  } break;
  case GGML_TYPE_F16: {
    fGgmlComputeForwardSumF16(pParams, pDst);
  } break;
  case GGML_TYPE_BF16: {
    fGgmlComputeForwardSumBf16(pParams, pDst);
  } break;
  default: {
    GGML_ABORT("fatal error");
  }
  }
}

// ggml_compute_forward_cumsum

static void fGgmlComputeForwardCumsumF32(const ggml_compute_params *pParams, ggml_tensor *dst) {

  const ggml_tensor *src0 = dst->src[0];

  GGML_ASSERT(src0->nb[0] == sizeof(float));
  GGML_ASSERT(dst->nb[0] == sizeof(float));

  GGML_TENSOR_UNARY_OP_LOCALS

  GGML_ASSERT(ne0 == ne00);
  GGML_ASSERT(ne1 == ne01);
  GGML_ASSERT(ne2 == ne02);
  GGML_ASSERT(ne3 == ne03);

  const auto [ir0, ir1] = get_thread_range(pParams, src0);

  for (int64_t vIr = ir0; vIr < ir1; ++vIr) {
    const int64_t cI03 = vIr / (ne02 * ne01);
    const int64_t cI02 = (vIr - cI03 * ne02 * ne01) / ne01;
    const int64_t cI01 = (vIr - cI03 * ne02 * ne01 - cI02 * ne01);

    float *vSrcRow = (float *)((char *)src0->data + cI01 * nb01 + cI02 * nb02 + cI03 * nb03);
    float *vDstRow = (float *)((char *)dst->data + cI01 * nb1 + cI02 * nb2 + cI03 * nb3);

    ggml_vec_cumsum_f32(ne00, vDstRow, vSrcRow);
  }
}

void ggml_compute_forward_cumsum(const ggml_compute_params *pParams, ggml_tensor *pDst) {

  const ggml_tensor *vSrc0 = pDst->src[0];

  switch (vSrc0->type) {
  case GGML_TYPE_F32: {
    fGgmlComputeForwardCumsumF32(pParams, pDst);
  } break;
  default: {
    GGML_ABORT("fatal error");
  }
  }
}

// ggml_compute_forward_sum_rows

static void fGgmlComputeForwardSumRowsF32(const ggml_compute_params *pParams, ggml_tensor *dst) {

  const ggml_tensor *src0 = dst->src[0];

  if (pParams->ith != 0) {
    return;
  }

  GGML_ASSERT(src0->nb[0] == sizeof(float));
  GGML_ASSERT(dst->nb[0] == sizeof(float));

  GGML_TENSOR_UNARY_OP_LOCALS

  GGML_ASSERT(ne0 == 1);
  GGML_ASSERT(ne1 == ne01);
  GGML_ASSERT(ne2 == ne02);
  GGML_ASSERT(ne3 == ne03);

  for (int64_t vI3 = 0; vI3 < ne03; vI3++) {
    for (int64_t vI2 = 0; vI2 < ne02; vI2++) {
      for (int64_t vI1 = 0; vI1 < ne01; vI1++) {
        float *vSrcRow = (float *)((char *)src0->data + vI1 * nb01 + vI2 * nb02 + vI3 * nb03);
        float *vDstRow = (float *)((char *)dst->data + vI1 * nb1 + vI2 * nb2 + vI3 * nb3);
        float vRowSum = 0;
        ggml_vec_sum_f32(ne00, &vRowSum, vSrcRow);
        vDstRow[0] = vRowSum;
      }
    }
  }
}

void ggml_compute_forward_sum_rows(const ggml_compute_params *pParams, ggml_tensor *pDst) {

  const ggml_tensor *vSrc0 = pDst->src[0];

  switch (vSrc0->type) {
  case GGML_TYPE_F32: {
    fGgmlComputeForwardSumRowsF32(pParams, pDst);
  } break;
  default: {
    GGML_ABORT("fatal error");
  }
  }
}

// ggml_compute_forward_mean

static void fGgmlComputeForwardMeanF32(const ggml_compute_params *pParams, ggml_tensor *dst) {

  const ggml_tensor *src0 = dst->src[0];

  if (pParams->ith != 0) {
    return;
  }

  assert(src0->nb[0] == sizeof(float));

  GGML_TENSOR_UNARY_OP_LOCALS

  assert(ne0 == 1);
  assert(ne1 == ne01);
  assert(ne2 == ne02);
  assert(ne3 == ne03);

  GGML_UNUSED(ne0);
  GGML_UNUSED(ne1);
  GGML_UNUSED(ne2);
  GGML_UNUSED(ne3);

  for (int64_t vI03 = 0; vI03 < ne03; vI03++) {
    for (int64_t vI02 = 0; vI02 < ne02; vI02++) {
      for (int64_t vI01 = 0; vI01 < ne01; vI01++) {
        ggml_vec_sum_f32(ne00, (float *)((char *)dst->data + vI01 * nb1 + vI02 * nb2 + vI03 * nb3),
                         (float *)((char *)src0->data + vI01 * nb01 + vI02 * nb02 + vI03 * nb03));

        *(float *)((char *)dst->data + vI01 * nb1 + vI02 * nb2 + vI03 * nb3) /= (float)ne00;
      }
    }
  }
}

void ggml_compute_forward_mean(const ggml_compute_params *pParams, ggml_tensor *pDst) {

  const ggml_tensor *vSrc0 = pDst->src[0];

  switch (vSrc0->type) {
  case GGML_TYPE_F32: {
    fGgmlComputeForwardMeanF32(pParams, pDst);
  } break;
  default: {
    GGML_ABORT("fatal error");
  }
  }
}

// ggml_compute_forward_argmax

static void fGgmlComputeForwardArgmaxF32(const ggml_compute_params *pParams, ggml_tensor *pDst) {

  const ggml_tensor *vSrc0 = pDst->src[0];

  if (pParams->ith != 0) {
    return;
  }

  assert(vSrc0->nb[0] == sizeof(float));
  assert(pDst->nb[0] == sizeof(float));

  const int64_t cNe00 = vSrc0->ne[0];
  const int64_t cNe01 = vSrc0->ne[1];

  const size_t cNb01 = vSrc0->nb[1];
  const size_t cNb0 = pDst->nb[0];

  for (int64_t vI1 = 0; vI1 < cNe01; vI1++) {
    float *vSrc = (float *)((char *)vSrc0->data + vI1 * cNb01);
    int32_t *vDst = (int32_t *)((char *)pDst->data + vI1 * cNb0);
    int v_ = 0;
    ggml_vec_argmax_f32(cNe00, &v_, vSrc);
    vDst[0] = v_;
  }
}

void ggml_compute_forward_argmax(const ggml_compute_params *pParams, ggml_tensor *pDst) {

  const ggml_tensor *vSrc0 = pDst->src[0];

  switch (vSrc0->type) {
  case GGML_TYPE_F32: {
    fGgmlComputeForwardArgmaxF32(pParams, pDst);
  } break;
  default: {
    GGML_ABORT("fatal error");
  }
  }
}

// ggml_compute_forward_amax_pow2

static void fGgmlComputeForwardAmaxPow2F32(const ggml_compute_params *pParams, ggml_tensor *pDst) {
  const ggml_tensor *src0 = pDst->src[0];
  const ggml_tensor *dst = pDst;

  GGML_ASSERT(src0->nb[0] == sizeof(float));

  GGML_TENSOR_UNARY_OP_LOCALS

  const float cThreshold = ggml_get_op_params_f32(pDst, 0);

  const int64_t cNRows = ne01 * ne02 * ne03;
  const int64_t cRowsPerThread = (cNRows + pParams->nth - 1) / pParams->nth;
  const int64_t cIr0 = cRowsPerThread * pParams->ith;
  const int64_t cIr1 = MIN(cIr0 + cRowsPerThread, cNRows);

  for (int64_t vIr = cIr0; vIr < cIr1; ++vIr) {
    const int64_t cI03 = vIr / (ne02 * ne01);
    const int64_t cI02 = (vIr - cI03 * ne02 * ne01) / ne01;
    const int64_t cI01 = vIr - cI03 * ne02 * ne01 - cI02 * ne01;

    const float *vSrc = (const float *)((const char *)src0->data + cI01 * nb01 + cI02 * nb02 + cI03 * nb03);

    float vAmax = 0.0f;
    for (int64_t vI00 = 0; vI00 < ne00; ++vI00) {
      vAmax = MAX(vAmax, fabsf(vSrc[vI00]));
    }

    float *vDst = (float *)((char *)dst->data + cI01 * nb1 + cI02 * nb2 + cI03 * nb3);
    vDst[0] = fGgmlAmaxPow2F32(vAmax, cThreshold);
  }
}

void ggml_compute_forward_amax_pow2(const ggml_compute_params *pParams, ggml_tensor *pDst) {
  const ggml_tensor *vSrc0 = pDst->src[0];

  switch (vSrc0->type) {
  case GGML_TYPE_F32: {
    fGgmlComputeForwardAmaxPow2F32(pParams, pDst);
  } break;
  default: {
    GGML_ABORT("fatal error");
  }
  }
}

// ggml_compute_forward_count_equal

static void fGgmlComputeForwardCountEqualI32(const ggml_compute_params *pParams, ggml_tensor *dst) {

  const ggml_tensor *src0 = dst->src[0];
  const ggml_tensor *src1 = dst->src[1];

  GGML_TENSOR_BINARY_OP_LOCALS;

  GGML_ASSERT(src0->type == GGML_TYPE_I32);
  GGML_ASSERT(src1->type == GGML_TYPE_I32);
  GGML_ASSERT(ggml_are_same_shape(src0, src1));
  GGML_ASSERT(ggml_is_scalar(dst));
  GGML_ASSERT(dst->type == GGML_TYPE_I64);

  const int64_t cNr = ggml_nrows(src0);

  const int cIth = pParams->ith;
  const int cNth = pParams->nth;

  int64_t *vSums = (int64_t *)pParams->wdata;
  int64_t vSumThread = 0;

  // rows per thread
  const int64_t cDr = (cNr + cNth - 1) / cNth;

  // row range for this thread
  const int64_t cIr0 = cDr * cIth;
  const int64_t cIr1 = MIN(cIr0 + cDr, cNr);

  for (int64_t vIr = cIr0; vIr < cIr1; ++vIr) {
    const int64_t cI03 = vIr / (ne02 * ne01);
    const int64_t cI02 = (vIr - cI03 * ne03) / ne01;
    const int64_t cI01 = vIr - cI03 * ne03 - cI02 * ne02;

    const char *vData0 = (const char *)src0->data + cI03 * nb03 + cI02 * nb02 + cI01 * nb01;
    const char *vData1 = (const char *)src1->data + cI03 * nb13 + cI02 * nb12 + cI01 * nb11;

    for (int64_t vI00 = 0; vI00 < ne00; ++vI00) {
      const int32_t cVal0 = *((const int32_t *)(vData0 + vI00 * nb00));
      const int32_t cVal1 = *((const int32_t *)(vData1 + vI00 * nb10));

      vSumThread += cVal0 == cVal1;
    }
  }
  if (cIth != 0) {
    vSums[cIth] = vSumThread;
  }
  ggml_barrier(pParams->threadpool);

  if (cIth != 0) {
    return;
  }

  for (int vIthOther = 1; vIthOther < cNth; ++vIthOther) {
    vSumThread += vSums[vIthOther];
  }
  *((int64_t *)dst->data) = vSumThread;
}

void ggml_compute_forward_count_equal(const ggml_compute_params *pParams, ggml_tensor *pDst) {

  const ggml_tensor *vSrc0 = pDst->src[0];

  switch (vSrc0->type) {
  case GGML_TYPE_I32: {
    fGgmlComputeForwardCountEqualI32(pParams, pDst);
  } break;
  default: {
    GGML_ABORT("fatal error");
  }
  }
}

// ggml_compute_forward_repeat

static void fGgmlComputeForwardRepeatF32(const ggml_compute_params *pParams, ggml_tensor *dst) {

  const ggml_tensor *src0 = dst->src[0];

  if (pParams->ith != 0) {
    return;
  }

  GGML_ASSERT(ggml_can_repeat(src0, dst));

  GGML_TENSOR_UNARY_OP_LOCALS

  // guaranteed to be an integer due to the check in ggml_can_repeat
  const int cNr0 = (int)(ne0 / ne00);
  const int cNr1 = (int)(ne1 / ne01);
  const int cNr2 = (int)(ne2 / ne02);
  const int cNr3 = (int)(ne3 / ne03);

  // TODO: support for transposed / permuted tensors
  GGML_ASSERT(nb0 == sizeof(float));
  GGML_ASSERT(nb00 == sizeof(float));

  // TODO: maybe this is not optimal?
  for (int vI3 = 0; vI3 < cNr3; vI3++) {
    for (int vK3 = 0; vK3 < ne03; vK3++) {
      for (int vI2 = 0; vI2 < cNr2; vI2++) {
        for (int vK2 = 0; vK2 < ne02; vK2++) {
          for (int vI1 = 0; vI1 < cNr1; vI1++) {
            for (int vK1 = 0; vK1 < ne01; vK1++) {
              for (int vI0 = 0; vI0 < cNr0; vI0++) {
                ggml_vec_cpy_f32(ne00,
                                 (float *)((char *)dst->data + (vI3 * ne03 + vK3) * nb3 + (vI2 * ne02 + vK2) * nb2 +
                                           (vI1 * ne01 + vK1) * nb1 + (vI0 * ne00) * nb0),
                                 (float *)((char *)src0->data + (vK3)*nb03 + (vK2)*nb02 + (vK1)*nb01));
              }
            }
          }
        }
      }
    }
  }
}

static void fGgmlComputeForwardRepeatF16(const ggml_compute_params *pParams, ggml_tensor *dst) {

  const ggml_tensor *src0 = dst->src[0];

  if (pParams->ith != 0) {
    return;
  }

  GGML_ASSERT(ggml_can_repeat(src0, dst));

  GGML_TENSOR_UNARY_OP_LOCALS

  // guaranteed to be an integer due to the check in ggml_can_repeat
  const int cNr0 = (int)(ne0 / ne00);
  const int cNr1 = (int)(ne1 / ne01);
  const int cNr2 = (int)(ne2 / ne02);
  const int cNr3 = (int)(ne3 / ne03);

  // TODO: support for transposed / permuted tensors
  GGML_ASSERT(nb0 == sizeof(ggml_fp16_t));
  GGML_ASSERT(nb00 == sizeof(ggml_fp16_t));

  // TODO: maybe this is not optimal?
  for (int vI3 = 0; vI3 < cNr3; vI3++) {
    for (int vK3 = 0; vK3 < ne03; vK3++) {
      for (int vI2 = 0; vI2 < cNr2; vI2++) {
        for (int vK2 = 0; vK2 < ne02; vK2++) {
          for (int vI1 = 0; vI1 < cNr1; vI1++) {
            for (int vK1 = 0; vK1 < ne01; vK1++) {
              for (int vI0 = 0; vI0 < cNr0; vI0++) {
                ggml_fp16_t *vY =
                    (ggml_fp16_t *)((char *)dst->data + (vI3 * ne03 + vK3) * nb3 + (vI2 * ne02 + vK2) * nb2 +
                                    (vI1 * ne01 + vK1) * nb1 + (vI0 * ne00) * nb0);
                ggml_fp16_t *vX = (ggml_fp16_t *)((char *)src0->data + (vK3)*nb03 + (vK2)*nb02 + (vK1)*nb01);
                // ggml_vec_cpy_f16(ne00, y, x)
                for (int vI = 0; vI < ne00; ++vI) {
                  vY[vI] = vX[vI];
                }
              }
            }
          }
        }
      }
    }
  }
}

void ggml_compute_forward_repeat(const ggml_compute_params *pParams, ggml_tensor *pDst) {

  const ggml_tensor *vSrc0 = pDst->src[0];

  switch (vSrc0->type) {
  case GGML_TYPE_F16:
  case GGML_TYPE_BF16:
  case GGML_TYPE_I16: {
    fGgmlComputeForwardRepeatF16(pParams, pDst);
  } break;
  case GGML_TYPE_F32:
  case GGML_TYPE_I32: {
    fGgmlComputeForwardRepeatF32(pParams, pDst);
  } break;
  // TODO: templateify the implementation and support for I64
  // case GGML_TYPE_I64:
  //    {
  //        ggml_compute_forward_repeat_i64(params, dst);
  //    } break;
  default: {
    GGML_ABORT("fatal error");
  }
  }
}

// ggml_compute_forward_repeat_back

static void fGgmlComputeForwardRepeatBackF32(const ggml_compute_params *pParams, ggml_tensor *dst) {

  const ggml_tensor *src0 = dst->src[0];

  if (pParams->ith != 0) {
    return;
  }

  GGML_ASSERT(ggml_can_repeat(dst, src0));

  GGML_TENSOR_UNARY_OP_LOCALS

  // guaranteed to be an integer due to the check in ggml_can_repeat
  const int cNr0 = (int)(ne00 / ne0);
  const int cNr1 = (int)(ne01 / ne1);
  const int cNr2 = (int)(ne02 / ne2);
  const int cNr3 = (int)(ne03 / ne3);

  // TODO: support for transposed / permuted tensors
  GGML_ASSERT(nb0 == sizeof(float));
  GGML_ASSERT(nb00 == sizeof(float));

  if (ggml_is_contiguous(dst)) {
    ggml_vec_set_f32(ne0 * ne1 * ne2 * ne3, (float *)dst->data, 0);
  } else {
    for (int vK3 = 0; vK3 < ne3; vK3++) {
      for (int vK2 = 0; vK2 < ne2; vK2++) {
        for (int vK1 = 0; vK1 < ne1; vK1++) {
          ggml_vec_set_f32(ne0, (float *)((char *)dst->data + vK1 * nb1 + vK2 * nb2 + vK3 * nb3), 0);
        }
      }
    }
  }

  // TODO: maybe this is not optimal?
  for (int vI3 = 0; vI3 < cNr3; vI3++) {
    for (int vK3 = 0; vK3 < ne3; vK3++) {
      for (int vI2 = 0; vI2 < cNr2; vI2++) {
        for (int vK2 = 0; vK2 < ne2; vK2++) {
          for (int vI1 = 0; vI1 < cNr1; vI1++) {
            for (int vK1 = 0; vK1 < ne1; vK1++) {
              for (int vI0 = 0; vI0 < cNr0; vI0++) {
                ggml_vec_acc_f32(ne0, (float *)((char *)dst->data + (vK3)*nb3 + (vK2)*nb2 + (vK1)*nb1),
                                 (float *)((char *)src0->data + (vI3 * ne3 + vK3) * nb03 + (vI2 * ne2 + vK2) * nb02 +
                                           (vI1 * ne1 + vK1) * nb01 + (vI0 * ne0) * nb00));
              }
            }
          }
        }
      }
    }
  }
}

void ggml_compute_forward_repeat_back(const ggml_compute_params *pParams, ggml_tensor *pDst) {

  const ggml_tensor *vSrc0 = pDst->src[0];

  switch (vSrc0->type) {
  case GGML_TYPE_F32: {
    fGgmlComputeForwardRepeatBackF32(pParams, pDst);
  } break;
  default: {
    GGML_ABORT("fatal error");
  }
  }
}

// ggml_compute_forward_concat

static void fGgmlComputeForwardConcatAny(const ggml_compute_params *pParams, ggml_tensor *dst) {

  const ggml_tensor *src0 = dst->src[0];
  const ggml_tensor *src1 = dst->src[1];

  const size_t cLen = ggml_type_size(src0->type);

  const int cIth = pParams->ith;
  const int cNth = pParams->nth;

  GGML_TENSOR_BINARY_OP_LOCALS

  const int32_t cDim = ggml_get_op_params_i32(dst, 0);

  GGML_ASSERT(cDim >= 0 && cDim < 4);

  int64_t aO[4] = {0, 0, 0, 0};
  if (cDim == 0) {
    aO[cDim] = src0->ne[cDim] / ggml_blck_size(src0->type);
  } else {
    aO[cDim] = src0->ne[cDim];
  }

  const char *vX;

  // TODO: smarter multi-theading
  for (int vI3 = 0; vI3 < ne3; vI3++) {
    for (int vI2 = cIth; vI2 < ne2; vI2 += cNth) {
      for (int vI1 = 0; vI1 < ne1; vI1++) {
        for (int vI0 = 0; vI0 < ne0 / ggml_blck_size(dst->type); vI0++) {
          if (vI0 < ne00 / ggml_blck_size(src0->type) && vI1 < ne01 && vI2 < ne02 && vI3 < ne03) {
            vX = (const char *)src0->data + (vI0)*nb00 + (vI1)*nb01 + (vI2)*nb02 + (vI3)*nb03;
          } else {
            vX = (const char *)src1->data + (vI0 - aO[0]) * nb10 + (vI1 - aO[1]) * nb11 + (vI2 - aO[2]) * nb12 +
                 (vI3 - aO[3]) * nb13;
          }

          char *vY = (char *)dst->data + vI0 * nb0 + vI1 * nb1 + vI2 * nb2 + vI3 * nb3;

          memcpy(vY, vX, cLen);
        }
      }
    }
  }
}

static void fGgmlComputeForwardConcatI8(const ggml_compute_params *pParams, ggml_tensor *dst) {

  const ggml_tensor *src0 = dst->src[0];
  const ggml_tensor *src1 = dst->src[1];

  GGML_ASSERT(ggml_type_size(src0->type) == sizeof(int8_t));

  const int cIth = pParams->ith;
  const int cNth = pParams->nth;

  GGML_TENSOR_BINARY_OP_LOCALS

  const int32_t cDim = ggml_get_op_params_i32(dst, 0);

  GGML_ASSERT(cDim >= 0 && cDim < 4);

  int64_t aO[4] = {0, 0, 0, 0};
  aO[cDim] = src0->ne[cDim];

  const int8_t *vX;

  // TODO: smarter multi-theading
  for (int vI3 = 0; vI3 < ne3; vI3++) {
    for (int vI2 = cIth; vI2 < ne2; vI2 += cNth) {
      for (int vI1 = 0; vI1 < ne1; vI1++) {
        for (int vI0 = 0; vI0 < ne0; vI0++) {
          if (vI0 < ne00 && vI1 < ne01 && vI2 < ne02 && vI3 < ne03) {
            vX = (const int8_t *)((const char *)src0->data + (vI0)*nb00 + (vI1)*nb01 + (vI2)*nb02 + (vI3)*nb03);
          } else {
            vX = (const int8_t *)((const char *)src1->data + (vI0 - aO[0]) * nb10 + (vI1 - aO[1]) * nb11 +
                                  (vI2 - aO[2]) * nb12 + (vI3 - aO[3]) * nb13);
          }

          int8_t *vY = (int8_t *)((char *)dst->data + vI0 * nb0 + vI1 * nb1 + vI2 * nb2 + vI3 * nb3);

          *vY = *vX;
        }
      }
    }
  }
}

static void fGgmlComputeForwardConcatF16(const ggml_compute_params *pParams, ggml_tensor *dst) {

  const ggml_tensor *src0 = dst->src[0];
  const ggml_tensor *src1 = dst->src[1];

  GGML_ASSERT(ggml_type_size(src0->type) == sizeof(ggml_fp16_t));

  const int cIth = pParams->ith;
  const int cNth = pParams->nth;

  GGML_TENSOR_BINARY_OP_LOCALS

  const int32_t cDim = ggml_get_op_params_i32(dst, 0);

  GGML_ASSERT(cDim >= 0 && cDim < 4);

  int64_t aO[4] = {0, 0, 0, 0};
  aO[cDim] = src0->ne[cDim];

  const ggml_fp16_t *vX;

  // TODO: smarter multi-theading
  for (int vI3 = 0; vI3 < ne3; vI3++) {
    for (int vI2 = cIth; vI2 < ne2; vI2 += cNth) {
      for (int vI1 = 0; vI1 < ne1; vI1++) {
        for (int vI0 = 0; vI0 < ne0; vI0++) {
          if (vI0 < ne00 && vI1 < ne01 && vI2 < ne02 && vI3 < ne03) {
            vX = (const ggml_fp16_t *)((const char *)src0->data + (vI0)*nb00 + (vI1)*nb01 + (vI2)*nb02 + (vI3)*nb03);
          } else {
            vX = (const ggml_fp16_t *)((const char *)src1->data + (vI0 - aO[0]) * nb10 + (vI1 - aO[1]) * nb11 +
                                       (vI2 - aO[2]) * nb12 + (vI3 - aO[3]) * nb13);
          }

          ggml_fp16_t *vY = (ggml_fp16_t *)((char *)dst->data + vI0 * nb0 + vI1 * nb1 + vI2 * nb2 + vI3 * nb3);

          *vY = *vX;
        }
      }
    }
  }
}

static void fGgmlComputeForwardConcatF32(const ggml_compute_params *pParams, ggml_tensor *dst) {

  const ggml_tensor *src0 = dst->src[0];
  const ggml_tensor *src1 = dst->src[1];

  GGML_ASSERT(ggml_type_size(src0->type) == sizeof(float));

  const int cIth = pParams->ith;
  const int cNth = pParams->nth;

  GGML_TENSOR_BINARY_OP_LOCALS

  const int32_t cDim = ggml_get_op_params_i32(dst, 0);

  GGML_ASSERT(cDim >= 0 && cDim < 4);

  int64_t aO[4] = {0, 0, 0, 0};
  aO[cDim] = src0->ne[cDim];

  const float *vX;

  // TODO: smarter multi-theading
  for (int vI3 = 0; vI3 < ne3; vI3++) {
    for (int vI2 = cIth; vI2 < ne2; vI2 += cNth) {
      for (int vI1 = 0; vI1 < ne1; vI1++) {
        for (int vI0 = 0; vI0 < ne0; vI0++) {
          if (vI0 < ne00 && vI1 < ne01 && vI2 < ne02 && vI3 < ne03) {
            vX = (const float *)((const char *)src0->data + (vI0)*nb00 + (vI1)*nb01 + (vI2)*nb02 + (vI3)*nb03);
          } else {
            vX = (const float *)((const char *)src1->data + (vI0 - aO[0]) * nb10 + (vI1 - aO[1]) * nb11 +
                                 (vI2 - aO[2]) * nb12 + (vI3 - aO[3]) * nb13);
          }

          float *vY = (float *)((char *)dst->data + vI0 * nb0 + vI1 * nb1 + vI2 * nb2 + vI3 * nb3);

          *vY = *vX;
        }
      }
    }
  }
}

void ggml_compute_forward_concat(const ggml_compute_params *pParams, ggml_tensor *pDst) {

  const ggml_tensor *vSrc0 = pDst->src[0];
  const ggml_tensor *vSrc1 = pDst->src[1];

  if (ggml_is_quantized(vSrc0->type)) {
    GGML_ASSERT(ggml_is_contiguous_rows(vSrc0));
    GGML_ASSERT(ggml_is_contiguous_rows(vSrc1));
    GGML_ASSERT(vSrc0->ne[0] % ggml_blck_size(vSrc0->type) == 0);
    GGML_ASSERT(vSrc1->ne[0] % ggml_blck_size(vSrc1->type) == 0);
  }

  switch (vSrc0->type) {
  case GGML_TYPE_F16:
  case GGML_TYPE_BF16:
  case GGML_TYPE_I16: {
    fGgmlComputeForwardConcatF16(pParams, pDst);
  } break;
  case GGML_TYPE_I8: {
    fGgmlComputeForwardConcatI8(pParams, pDst);
  } break;
  case GGML_TYPE_F32:
  case GGML_TYPE_I32: {
    fGgmlComputeForwardConcatF32(pParams, pDst);
  } break;
  default: {
    fGgmlComputeForwardConcatAny(pParams, pDst);
  }
  }
}

// ggml_compute_forward_gelu

static void fGgmlComputeForwardGeluF32(const ggml_compute_params *pParams, ggml_tensor *pDst) {

  const ggml_tensor *vSrc0 = pDst->src[0];

  assert(ggml_is_contiguous_rows(vSrc0));
  assert(ggml_are_same_shape(vSrc0, pDst));

  GGML_TENSOR_LOCALS(int64_t, ne0, vSrc0, ne)
  GGML_TENSOR_LOCALS(size_t, nb0, vSrc0, nb)
  GGML_TENSOR_LOCALS(int64_t, ne, pDst, ne)
  GGML_TENSOR_LOCALS(size_t, nb, pDst, nb)

  const int cIth = pParams->ith;
  const int cNth = pParams->nth;

  const int cNc = vSrc0->ne[0];
  const int cNr = ggml_nrows(vSrc0);

  // rows per thread
  const int cDr = (cNr + cNth - 1) / cNth;

  // row range for this thread
  const int cIr0 = cDr * cIth;
  const int cIr1 = MIN(cIr0 + cDr, cNr);

  for (int vIr = cIr0; vIr < cIr1; ++vIr) {
    const int cI3 = vIr / (ne02 * ne01);
    const int cI2 = (vIr - cI3 * ne02 * ne01) / ne01;
    const int cI1 = (vIr - cI3 * ne02 * ne01 - cI2 * ne01);

    ggml_vec_gelu_f32(cNc, (float *)((char *)pDst->data + cI3 * nb3 + cI2 * nb2 + cI1 * nb1),
                      (float *)((char *)vSrc0->data + cI3 * nb03 + cI2 * nb02 + cI1 * nb01));

#ifndef NDEBUG
    for (int vK = 0; vK < cNc; vK++) {
      const float cX = ((float *)((char *)pDst->data + cI3 * nb3 + cI2 * nb2 + cI1 * (pDst->nb[1])))[vK];
      GGML_UNUSED(cX);
      assert(!isnan(cX));
      assert(!isinf(cX));
    }
#endif // NDEBUG
  }
}

static void fGgmlComputeForwardGeluF16(const ggml_compute_params *pParams, ggml_tensor *pDst) {

  const ggml_tensor *vSrc0 = pDst->src[0];

  assert(ggml_is_contiguous_rows(vSrc0));
  assert(ggml_are_same_shape(vSrc0, pDst));

  GGML_TENSOR_LOCALS(int64_t, ne0, vSrc0, ne)
  GGML_TENSOR_LOCALS(size_t, nb0, vSrc0, nb)
  GGML_TENSOR_LOCALS(int64_t, ne, pDst, ne)
  GGML_TENSOR_LOCALS(size_t, nb, pDst, nb)

  const int cIth = pParams->ith;
  const int cNth = pParams->nth;

  const int cNc = vSrc0->ne[0];
  const int cNr = ggml_nrows(vSrc0);

  // rows per thread
  const int cDr = (cNr + cNth - 1) / cNth;

  // row range for this thread
  const int cIr0 = cDr * cIth;
  const int cIr1 = MIN(cIr0 + cDr, cNr);

  for (int vIr = cIr0; vIr < cIr1; ++vIr) {
    const int cI3 = vIr / (ne02 * ne01);
    const int cI2 = (vIr - cI3 * ne02 * ne01) / ne01;
    const int cI1 = (vIr - cI3 * ne02 * ne01 - cI2 * ne01);

    ggml_vec_gelu_f16(cNc, (ggml_fp16_t *)((char *)pDst->data + cI3 * nb3 + cI2 * nb2 + cI1 * nb1),
                      (ggml_fp16_t *)((char *)vSrc0->data + cI3 * nb03 + cI2 * nb02 + cI1 * nb01));

#ifndef NDEBUG
    for (int vK = 0; vK < cNc; vK++) {
      const ggml_fp16_t cX = ((ggml_fp16_t *)((char *)pDst->data + cI3 * nb3 + cI2 * nb2 + cI1 * (pDst->nb[1])))[vK];
      const float cV = GGML_CPU_FP16_TO_FP32(cX);
      GGML_UNUSED(cV);
      assert(!isnan(cV));
      assert(!isinf(cV));
    }
#endif // NDEBUG
  }
}

static void fGgmlComputeForwardGelu(const ggml_compute_params *pParams, ggml_tensor *pDst) {

  const ggml_tensor *vSrc0 = pDst->src[0];

  switch (vSrc0->type) {
  case GGML_TYPE_F32: {
    fGgmlComputeForwardGeluF32(pParams, pDst);
  } break;
  case GGML_TYPE_F16: {
    fGgmlComputeForwardGeluF16(pParams, pDst);
  } break;
  default: {
    GGML_ABORT("fatal error");
  }
  }
}

// ggml_compute_fill

static void fGgmlComputeForwardFillF32(const ggml_compute_params *pParams, ggml_tensor *pDst) {
  const float c_ = ggml_get_op_params_f32(pDst, 0);

  GGML_TENSOR_LOCALS(int64_t, ne, pDst, ne);
  GGML_TENSOR_LOCALS(size_t, nb, pDst, nb);

  const auto [ir0, ir1] = get_thread_range(pParams, pDst);

  for (int64_t vIr = ir0; vIr < ir1; ++vIr) {
    const int64_t cI03 = vIr / (ne2 * ne1);
    const int64_t cI02 = (vIr - cI03 * ne2 * ne1) / ne1;
    const int64_t cI01 = (vIr - cI03 * ne2 * ne1 - cI02 * ne1);

    float *vDstPtr = (float *)((char *)pDst->data + cI03 * nb3 + cI02 * nb2 + cI01 * nb1);

    ggml_vec_set_f32(ne0, vDstPtr, c_);
  }
}

static void fGgmlComputeForwardFillF16(const ggml_compute_params *pParams, ggml_tensor *pDst) {
  const ggml_fp16_t c_ = GGML_CPU_FP32_TO_FP16(ggml_get_op_params_f32(pDst, 0));

  GGML_TENSOR_LOCALS(int64_t, ne, pDst, ne);
  GGML_TENSOR_LOCALS(size_t, nb, pDst, nb);

  const auto [ir0, ir1] = get_thread_range(pParams, pDst);

  for (int64_t vIr = ir0; vIr < ir1; ++vIr) {
    const int64_t cI03 = vIr / (ne2 * ne1);
    const int64_t cI02 = (vIr - cI03 * ne2 * ne1) / ne1;
    const int64_t cI01 = (vIr - cI03 * ne2 * ne1 - cI02 * ne1);

    ggml_fp16_t *vDstPtr = (ggml_fp16_t *)((char *)pDst->data + cI03 * nb3 + cI02 * nb2 + cI01 * nb1);

    ggml_vec_set_f16(ne0, vDstPtr, c_);
  }
}

void ggml_compute_forward_fill(const ggml_compute_params *pParams, ggml_tensor *pDst) {
  const ggml_tensor *vSrc0 = pDst->src[0];

  switch (vSrc0->type) {
  case GGML_TYPE_F32: {
    fGgmlComputeForwardFillF32(pParams, pDst);
  } break;
  case GGML_TYPE_F16: {
    fGgmlComputeForwardFillF16(pParams, pDst);
  } break;
  default: {
    GGML_ABORT("unsupported type for ggml_compute_forward_fill: %s", ggml_type_name(vSrc0->type));
  }
  }
}

// ggml_compute_tri

static void fGgmlComputeForwardTriF32(const ggml_compute_params *pParams, ggml_tensor *dst) {
  const ggml_tensor *src0 = dst->src[0];

  const ggml_tri_type cTtype = (ggml_tri_type)ggml_get_op_params_i32(dst, 0);

  GGML_ASSERT(ggml_is_contiguous(src0));

  GGML_TENSOR_UNARY_OP_LOCALS

  const auto [ir0, ir1] = get_thread_range(pParams, src0);

  bool (*vBipred)(int, int);

  switch (cTtype) {
  case GGML_TRI_TYPE_LOWER:
    vBipred = [](int pI, int pR) { return pI < pR; };
    break;
  case GGML_TRI_TYPE_LOWER_DIAG:
    vBipred = [](int pI, int pR) { return pI <= pR; };
    break;
  case GGML_TRI_TYPE_UPPER:
    vBipred = [](int pI, int pR) { return pI > pR; };
    break;
  case GGML_TRI_TYPE_UPPER_DIAG:
    vBipred = [](int pI, int pR) { return pI >= pR; };
    break;
  default:
    GGML_ABORT("invalid tri type");
  }

  for (int64_t vIr = ir0; vIr < ir1; ++vIr) {
    const int64_t cI03 = vIr / (ne02 * ne01);
    const int64_t cI02 = (vIr - cI03 * ne02 * ne01) / ne01;
    const int64_t cI01 = (vIr - cI03 * ne02 * ne01 - cI02 * ne01);

    const float *vSrcPtr = (const float *)((const char *)src0->data + cI03 * nb03 + cI02 * nb02 + cI01 * nb01);
    float *vDstPtr = (float *)((char *)dst->data + cI03 * nb3 + cI02 * nb2 + cI01 * nb1);

    for (int vI0 = 0; vI0 < ne0; ++vI0) {
      vDstPtr[vI0] = vBipred(vI0, cI01) ? vSrcPtr[vI0] : 0.0f;
    }
  }
}

void ggml_compute_forward_tri(const ggml_compute_params *pParams, ggml_tensor *pDst) {
  const ggml_tensor *vSrc0 = pDst->src[0];

  switch (vSrc0->type) {
  case GGML_TYPE_F32: {
    fGgmlComputeForwardTriF32(pParams, pDst);
  } break;
  default: {
    GGML_ABORT("fatal error");
  }
  }
}

// ggml_compute_forward_gelu_erf

static void fGgmlComputeForwardGeluErfF32(const ggml_compute_params *pParams, ggml_tensor *pDst) {

  const ggml_tensor *vSrc0 = pDst->src[0];

  assert(ggml_is_contiguous_rows(vSrc0));
  assert(ggml_are_same_shape(vSrc0, pDst));

  GGML_TENSOR_LOCALS(int64_t, ne0, vSrc0, ne)
  GGML_TENSOR_LOCALS(size_t, nb0, vSrc0, nb)
  GGML_TENSOR_LOCALS(int64_t, ne, pDst, ne)
  GGML_TENSOR_LOCALS(size_t, nb, pDst, nb)

  const int cIth = pParams->ith;
  const int cNth = pParams->nth;

  const int cNc = vSrc0->ne[0];
  const int cNr = ggml_nrows(vSrc0);

  // rows per thread
  const int cDr = (cNr + cNth - 1) / cNth;

  // row range for this thread
  const int cIr0 = cDr * cIth;
  const int cIr1 = MIN(cIr0 + cDr, cNr);

  for (int vIr = cIr0; vIr < cIr1; ++vIr) {
    const int cI3 = vIr / (ne02 * ne01);
    const int cI2 = (vIr - cI3 * ne02 * ne01) / ne01;
    const int cI1 = (vIr - cI3 * ne02 * ne01 - cI2 * ne01);

    ggml_vec_gelu_erf_f32(cNc, (float *)((char *)pDst->data + cI3 * nb3 + cI2 * nb2 + cI1 * nb1),
                          (float *)((char *)vSrc0->data + cI3 * nb03 + cI2 * nb02 + cI1 * nb01));

#ifndef NDEBUG
    for (int vK = 0; vK < cNc; vK++) {
      const float cX = ((float *)((char *)pDst->data + cI3 * nb3 + cI2 * nb2 + cI1 * (pDst->nb[1])))[vK];
      GGML_UNUSED(cX);
      assert(!isnan(cX));
      assert(!isinf(cX));
    }
#endif // NDEBUG
  }
}

static void fGgmlComputeForwardGeluErfF16(const ggml_compute_params *pParams, ggml_tensor *pDst) {

  const ggml_tensor *vSrc0 = pDst->src[0];

  assert(ggml_is_contiguous_rows(vSrc0));
  assert(ggml_are_same_shape(vSrc0, pDst));

  GGML_TENSOR_LOCALS(int64_t, ne0, vSrc0, ne)
  GGML_TENSOR_LOCALS(size_t, nb0, vSrc0, nb)
  GGML_TENSOR_LOCALS(int64_t, ne, pDst, ne)
  GGML_TENSOR_LOCALS(size_t, nb, pDst, nb)

  const int cIth = pParams->ith;
  const int cNth = pParams->nth;

  const int cNc = vSrc0->ne[0];
  const int cNr = ggml_nrows(vSrc0);

  // rows per thread
  const int cDr = (cNr + cNth - 1) / cNth;

  // row range for this thread
  const int cIr0 = cDr * cIth;
  const int cIr1 = MIN(cIr0 + cDr, cNr);

  for (int vIr = cIr0; vIr < cIr1; ++vIr) {
    const int cI3 = vIr / (ne02 * ne01);
    const int cI2 = (vIr - cI3 * ne02 * ne01) / ne01;
    const int cI1 = (vIr - cI3 * ne02 * ne01 - cI2 * ne01);

    ggml_vec_gelu_erf_f16(cNc, (ggml_fp16_t *)((char *)pDst->data + cI3 * nb3 + cI2 * nb2 + cI1 * nb1),
                          (ggml_fp16_t *)((char *)vSrc0->data + cI3 * nb03 + cI2 * nb02 + cI1 * nb01));

#ifndef NDEBUG
    for (int vK = 0; vK < cNc; vK++) {
      const ggml_fp16_t cX = ((ggml_fp16_t *)((char *)pDst->data + cI3 * nb3 + cI2 * nb2 + cI1 * (pDst->nb[1])))[vK];
      const float cV = GGML_CPU_FP16_TO_FP32(cX);
      GGML_UNUSED(cV);
      assert(!isnan(cV));
      assert(!isinf(cV));
    }
#endif // NDEBUG
  }
}

static void fGgmlComputeForwardGeluErf(const ggml_compute_params *pParams, ggml_tensor *pDst) {

  const ggml_tensor *vSrc0 = pDst->src[0];

  switch (vSrc0->type) {
  case GGML_TYPE_F32: {
    fGgmlComputeForwardGeluErfF32(pParams, pDst);
  } break;
  case GGML_TYPE_F16: {
    fGgmlComputeForwardGeluErfF16(pParams, pDst);
  } break;
  default: {
    GGML_ABORT("fatal error");
  }
  }
}

// ggml_compute_forward_gelu_quick

static void fGgmlComputeForwardGeluQuickF32(const ggml_compute_params *pParams, ggml_tensor *pDst) {

  const ggml_tensor *vSrc0 = pDst->src[0];

  assert(ggml_is_contiguous_rows(vSrc0));
  assert(ggml_are_same_shape(vSrc0, pDst));

  GGML_TENSOR_LOCALS(int64_t, ne0, vSrc0, ne)
  GGML_TENSOR_LOCALS(size_t, nb0, vSrc0, nb)
  GGML_TENSOR_LOCALS(int64_t, ne, pDst, ne)
  GGML_TENSOR_LOCALS(size_t, nb, pDst, nb)

  const int cIth = pParams->ith;
  const int cNth = pParams->nth;

  const int cNc = vSrc0->ne[0];
  const int cNr = ggml_nrows(vSrc0);

  // rows per thread
  const int cDr = (cNr + cNth - 1) / cNth;

  // row range for this thread
  const int cIr0 = cDr * cIth;
  const int cIr1 = MIN(cIr0 + cDr, cNr);

  for (int vIr = cIr0; vIr < cIr1; ++vIr) {
    const int cI3 = vIr / (ne02 * ne01);
    const int cI2 = (vIr - cI3 * ne02 * ne01) / ne01;
    const int cI1 = (vIr - cI3 * ne02 * ne01 - cI2 * ne01);

    ggml_vec_gelu_quick_f32(cNc, (float *)((char *)pDst->data + cI3 * nb3 + cI2 * nb2 + cI1 * nb1),
                            (float *)((char *)vSrc0->data + cI3 * nb03 + cI2 * nb02 + cI1 * nb01));

#ifndef NDEBUG
    for (int vK = 0; vK < cNc; vK++) {
      const float cX = ((float *)((char *)pDst->data + cI3 * nb3 + cI2 * nb2 + cI1 * (pDst->nb[1])))[vK];
      GGML_UNUSED(cX);
      assert(!isnan(cX));
      assert(!isinf(cX));
    }
#endif // NDEBUG
  }
}

static void fGgmlComputeForwardGeluQuickF16(const ggml_compute_params *pParams, ggml_tensor *pDst) {

  const ggml_tensor *vSrc0 = pDst->src[0];

  assert(ggml_is_contiguous_rows(vSrc0));
  assert(ggml_are_same_shape(vSrc0, pDst));

  GGML_TENSOR_LOCALS(int64_t, ne0, vSrc0, ne)
  GGML_TENSOR_LOCALS(size_t, nb0, vSrc0, nb)
  GGML_TENSOR_LOCALS(int64_t, ne, pDst, ne)
  GGML_TENSOR_LOCALS(size_t, nb, pDst, nb)

  const int cIth = pParams->ith;
  const int cNth = pParams->nth;

  const int cNc = vSrc0->ne[0];
  const int cNr = ggml_nrows(vSrc0);

  // rows per thread
  const int cDr = (cNr + cNth - 1) / cNth;

  // row range for this thread
  const int cIr0 = cDr * cIth;
  const int cIr1 = MIN(cIr0 + cDr, cNr);

  for (int vIr = cIr0; vIr < cIr1; ++vIr) {
    const int cI3 = vIr / (ne02 * ne01);
    const int cI2 = (vIr - cI3 * ne02 * ne01) / ne01;
    const int cI1 = (vIr - cI3 * ne02 * ne01 - cI2 * ne01);

    ggml_vec_gelu_quick_f16(cNc, (ggml_fp16_t *)((char *)pDst->data + cI3 * nb3 + cI2 * nb2 + cI1 * nb1),
                            (ggml_fp16_t *)((char *)vSrc0->data + cI3 * nb03 + cI2 * nb02 + cI1 * nb01));

#ifndef NDEBUG
    for (int vK = 0; vK < cNc; vK++) {
      const ggml_fp16_t cX = ((ggml_fp16_t *)((char *)pDst->data + cI3 * nb3 + cI2 * nb2 + cI1 * (pDst->nb[1])))[vK];
      const float cV = GGML_CPU_FP16_TO_FP32(cX);
      GGML_UNUSED(cV);
      assert(!isnan(cV));
      assert(!isinf(cV));
    }
#endif // NDEBUG
  }
}

static void fGgmlComputeForwardGeluQuick(const ggml_compute_params *pParams, ggml_tensor *pDst) {

  const ggml_tensor *vSrc0 = pDst->src[0];

  switch (vSrc0->type) {
  case GGML_TYPE_F32: {
    fGgmlComputeForwardGeluQuickF32(pParams, pDst);
  } break;
  case GGML_TYPE_F16: {
    fGgmlComputeForwardGeluQuickF16(pParams, pDst);
  } break;
  default: {
    GGML_ABORT("fatal error");
  }
  }
}

// ggml_compute_forward_silu

static void fGgmlComputeForwardSiluF32(const ggml_compute_params *pParams, ggml_tensor *pDst) {

  const ggml_tensor *vSrc0 = pDst->src[0];

  assert(ggml_is_contiguous_rows(vSrc0));
  assert(ggml_are_same_shape(vSrc0, pDst));

  GGML_TENSOR_LOCALS(int64_t, ne0, vSrc0, ne)
  GGML_TENSOR_LOCALS(size_t, nb0, vSrc0, nb)
  GGML_TENSOR_LOCALS(int64_t, ne, pDst, ne)
  GGML_TENSOR_LOCALS(size_t, nb, pDst, nb)

  const int cIth = pParams->ith;
  const int cNth = pParams->nth;

  const int cNc = vSrc0->ne[0];
  const int cNr = ggml_nrows(vSrc0);

  // rows per thread
  const int cDr = (cNr + cNth - 1) / cNth;

  // row range for this thread
  const int cIr0 = cDr * cIth;
  const int cIr1 = MIN(cIr0 + cDr, cNr);

  for (int vIr = cIr0; vIr < cIr1; ++vIr) {
    const int cI3 = vIr / (ne02 * ne01);
    const int cI2 = (vIr - cI3 * ne02 * ne01) / ne01;
    const int cI1 = (vIr - cI3 * ne02 * ne01 - cI2 * ne01);

    ggml_vec_silu_f32(cNc, (float *)((char *)pDst->data + cI3 * nb3 + cI2 * nb2 + cI1 * nb1),
                      (float *)((char *)vSrc0->data + cI3 * nb03 + cI2 * nb02 + cI1 * nb01));

#ifndef NDEBUG
    for (int vK = 0; vK < cNc; vK++) {
      const float cX = ((float *)((char *)pDst->data + cI3 * nb3 + cI2 * nb2 + cI1 * (pDst->nb[1])))[vK];
      GGML_UNUSED(cX);
      assert(!isnan(cX));
      assert(!isinf(cX));
    }
#endif // NDEBUG
  }
}

static void fGgmlComputeForwardSiluF16(const ggml_compute_params *pParams, ggml_tensor *pDst) {

  const ggml_tensor *vSrc0 = pDst->src[0];

  assert(ggml_is_contiguous_rows(vSrc0));
  assert(ggml_are_same_shape(vSrc0, pDst));

  GGML_TENSOR_LOCALS(int64_t, ne0, vSrc0, ne)
  GGML_TENSOR_LOCALS(size_t, nb0, vSrc0, nb)
  GGML_TENSOR_LOCALS(int64_t, ne, pDst, ne)
  GGML_TENSOR_LOCALS(size_t, nb, pDst, nb)

  const int cIth = pParams->ith;
  const int cNth = pParams->nth;

  const int cNc = vSrc0->ne[0];
  const int cNr = ggml_nrows(vSrc0);

  // rows per thread
  const int cDr = (cNr + cNth - 1) / cNth;

  // row range for this thread
  const int cIr0 = cDr * cIth;
  const int cIr1 = MIN(cIr0 + cDr, cNr);

  for (int vIr = cIr0; vIr < cIr1; ++vIr) {
    const int cI3 = vIr / (ne02 * ne01);
    const int cI2 = (vIr - cI3 * ne02 * ne01) / ne01;
    const int cI1 = (vIr - cI3 * ne02 * ne01 - cI2 * ne01);

    ggml_vec_silu_f16(cNc, (ggml_fp16_t *)((char *)pDst->data + cI3 * nb3 + cI2 * nb2 + cI1 * nb1),
                      (ggml_fp16_t *)((char *)vSrc0->data + cI3 * nb03 + cI2 * nb02 + cI1 * nb01));

#ifndef NDEBUG
    for (int vK = 0; vK < cNc; vK++) {
      const ggml_fp16_t cX = ((ggml_fp16_t *)((char *)pDst->data + cI3 * nb3 + cI2 * nb2 + cI1 * (pDst->nb[1])))[vK];
      const float cV = GGML_CPU_FP16_TO_FP32(cX);
      GGML_UNUSED(cV);
      assert(!isnan(cV));
      assert(!isinf(cV));
    }
#endif // NDEBUG
  }
}

static void fGgmlComputeForwardSilu(const ggml_compute_params *pParams, ggml_tensor *pDst) {

  const ggml_tensor *vSrc0 = pDst->src[0];

  switch (vSrc0->type) {
  case GGML_TYPE_F32: {
    fGgmlComputeForwardSiluF32(pParams, pDst);
  } break;
  case GGML_TYPE_F16: {
    fGgmlComputeForwardSiluF16(pParams, pDst);
  } break;
  default: {
    GGML_ABORT("fatal error");
  }
  }
}
// ggml_compute_forward_leaky_relu

static void fGgmlComputeForwardLeakyReluF32(const ggml_compute_params *pParams, ggml_tensor *pDst) {

  const ggml_tensor *vSrc0 = pDst->src[0];

  if (pParams->ith != 0) {
    return;
  }

  assert(ggml_is_contiguous_1(vSrc0));
  assert(ggml_is_contiguous_1(pDst));
  assert(ggml_are_same_shape(vSrc0, pDst));

  const int cN = ggml_nrows(vSrc0);
  const int cNc = vSrc0->ne[0];

  float vNegativeSlope;
  memcpy(&vNegativeSlope, pDst->op_params, sizeof(float));

  assert(pDst->nb[0] == sizeof(float));
  assert(vSrc0->nb[0] == sizeof(float));

  for (int vI = 0; vI < cN; vI++) {
    ggml_vec_leaky_relu_f32(cNc, (float *)((char *)pDst->data + vI * (pDst->nb[1])),
                            (float *)((char *)vSrc0->data + vI * (vSrc0->nb[1])), vNegativeSlope);
  }
}

static void fGgmlComputeForwardLeakyReluF16(const ggml_compute_params *pParams, ggml_tensor *pDst) {

  const ggml_tensor *vSrc0 = pDst->src[0];

  if (pParams->ith != 0) {
    return;
  }

  assert(ggml_is_contiguous_1(vSrc0));
  assert(ggml_is_contiguous_1(pDst));
  assert(ggml_are_same_shape(vSrc0, pDst));

  const int cN = ggml_nrows(vSrc0);
  const int cNc = vSrc0->ne[0];

  float vNegativeSlope;
  memcpy(&vNegativeSlope, pDst->op_params, sizeof(float));

  assert(pDst->nb[0] == sizeof(ggml_fp16_t));
  assert(vSrc0->nb[0] == sizeof(ggml_fp16_t));

  for (int vI = 0; vI < cN; vI++) {
    ggml_vec_leaky_relu_f16(cNc, (ggml_fp16_t *)((char *)pDst->data + vI * (pDst->nb[1])),
                            (ggml_fp16_t *)((char *)vSrc0->data + vI * (vSrc0->nb[1])), vNegativeSlope);
  }
}

void ggml_compute_forward_leaky_relu(const ggml_compute_params *pParams, ggml_tensor *pDst) {

  const ggml_tensor *vSrc0 = pDst->src[0];

  switch (vSrc0->type) {
  case GGML_TYPE_F32: {
    fGgmlComputeForwardLeakyReluF32(pParams, pDst);
  } break;
  case GGML_TYPE_F16: {
    fGgmlComputeForwardLeakyReluF16(pParams, pDst);
  } break;
  default: {
    GGML_ABORT("fatal error");
  }
  }
}

// ggml_compute_forward_silu_back

static void fGgmlComputeForwardSiluBackF32(const ggml_compute_params *pParams, ggml_tensor *pDst) {

  const ggml_tensor *vGrad = pDst->src[0];
  const ggml_tensor *vSrc1 = pDst->src[1];

  assert(ggml_is_contiguous_1(vGrad));
  assert(ggml_is_contiguous_1(vSrc1));
  assert(ggml_is_contiguous_1(pDst));
  assert(ggml_are_same_shape(vSrc1, pDst));
  assert(ggml_are_same_shape(vSrc1, vGrad));

  const int cIth = pParams->ith;
  const int cNth = pParams->nth;

  const int cNc = vSrc1->ne[0];
  const int cNr = ggml_nrows(vSrc1);

  // rows per thread
  const int cDr = (cNr + cNth - 1) / cNth;

  // row range for this thread
  const int cIr0 = cDr * cIth;
  const int cIr1 = MIN(cIr0 + cDr, cNr);

  for (int vI1 = cIr0; vI1 < cIr1; vI1++) {
    ggml_vec_silu_backward_f32(cNc, (float *)((char *)pDst->data + vI1 * (pDst->nb[1])),
                               (float *)((char *)vSrc1->data + vI1 * (vSrc1->nb[1])),
                               (float *)((char *)vGrad->data + vI1 * (vGrad->nb[1])));

#ifndef NDEBUG
    for (int vK = 0; vK < cNc; vK++) {
      const float cX = ((float *)((char *)pDst->data + vI1 * (pDst->nb[1])))[vK];
      GGML_UNUSED(cX);
      assert(!isnan(cX));
      assert(!isinf(cX));
    }
#endif // NDEBUG
  }
}

static void fGgmlComputeForwardSiluBackF16(const ggml_compute_params *pParams, ggml_tensor *pDst) {

  const ggml_tensor *vGrad = pDst->src[0];
  const ggml_tensor *vSrc1 = pDst->src[1];

  assert(ggml_is_contiguous_1(vGrad));
  assert(ggml_is_contiguous_1(vSrc1));
  assert(ggml_is_contiguous_1(pDst));
  assert(ggml_are_same_shape(vSrc1, pDst));
  assert(ggml_are_same_shape(vSrc1, vGrad));

  const int cIth = pParams->ith;
  const int cNth = pParams->nth;

  const int cNc = vSrc1->ne[0];
  const int cNr = ggml_nrows(vSrc1);

  // rows per thread
  const int cDr = (cNr + cNth - 1) / cNth;

  // row range for this thread
  const int cIr0 = cDr * cIth;
  const int cIr1 = MIN(cIr0 + cDr, cNr);

  for (int vI1 = cIr0; vI1 < cIr1; vI1++) {
    ggml_vec_silu_backward_f16(cNc, (ggml_fp16_t *)((char *)pDst->data + vI1 * (pDst->nb[1])),
                               (ggml_fp16_t *)((char *)vSrc1->data + vI1 * (vSrc1->nb[1])),
                               (ggml_fp16_t *)((char *)vGrad->data + vI1 * (vGrad->nb[1])));

#ifndef NDEBUG
    for (int vK = 0; vK < cNc; vK++) {
      const float cX = ((ggml_fp16_t *)((char *)pDst->data + vI1 * (pDst->nb[1])))[vK];
      const float cV = GGML_CPU_FP16_TO_FP32(cX);
      GGML_UNUSED(cV);
      assert(!isnan(cV));
      assert(!isinf(cV));
    }
#endif // NDEBUG
  }
}

void ggml_compute_forward_silu_back(const ggml_compute_params *pParams, ggml_tensor *pDst) {

  const ggml_tensor *vSrc0 = pDst->src[0];

  switch (vSrc0->type) {
  case GGML_TYPE_F32: {
    fGgmlComputeForwardSiluBackF32(pParams, pDst);
  } break;
  case GGML_TYPE_F16: {
    fGgmlComputeForwardSiluBackF16(pParams, pDst);
  } break;
  default: {
    GGML_ABORT("fatal error");
  }
  }
}

// ggml_compute_forward_reglu

static void fGgmlComputeForwardRegluF32(const ggml_compute_params *pParams, ggml_tensor *pDst) {

  const ggml_tensor *vSrc0 = pDst->src[0];
  const ggml_tensor *vSrc1 = pDst->src[1];
  char *vSrc0D = (char *)vSrc0->data;
  char *vSrc1D = (char *)(vSrc1 ? vSrc1->data : vSrc0->data);
  const size_t cSrc0O = vSrc0->nb[1];
  const size_t cSrc1O = vSrc1 ? vSrc1->nb[1] : vSrc0->nb[1];

  GGML_ASSERT(ggml_is_contiguous_1(vSrc0));
  GGML_ASSERT(ggml_is_contiguous_1(pDst));

  if (vSrc1) {
    GGML_ASSERT(ggml_is_contiguous_1(vSrc1));
    GGML_ASSERT(vSrc0->type == vSrc1->type);
  }

  const int cIth = pParams->ith;
  const int cNth = pParams->nth;

  const int cNc = vSrc1 ? vSrc0->ne[0] : vSrc0->ne[0] / 2;
  const int cNr = ggml_nrows(vSrc0);

  GGML_ASSERT(pDst->ne[0] == cNc);
  GGML_ASSERT(ggml_nrows(pDst) == cNr);

  const int32_t cSwapped = ggml_get_op_params_i32(pDst, 1);

  // rows per thread
  const int cDr = (cNr + cNth - 1) / cNth;

  // row range for this thread
  const int cIr0 = cDr * cIth;
  const int cIr1 = MIN(cIr0 + cDr, cNr);

  for (int vI1 = cIr0; vI1 < cIr1; vI1++) {
    float *vSrc0P = (float *)(vSrc0D + vI1 * cSrc0O);
    float *vSrc1P = (float *)(vSrc1D + vI1 * cSrc1O);

    if (!vSrc1) {
      vSrc0P += cSwapped ? cNc : 0;
      vSrc1P += cSwapped ? 0 : cNc;
    }

    ggml_vec_reglu_f32(cNc, (float *)((char *)pDst->data + vI1 * (pDst->nb[1])), vSrc0P, vSrc1P);

#ifndef NDEBUG
    for (int vK = 0; vK < cNc; vK++) {
      const float cX = ((float *)((char *)pDst->data + vI1 * (pDst->nb[1])))[vK];
      GGML_UNUSED(cX);
      assert(!isnan(cX));
      assert(!isinf(cX));
    }
#endif // NDEBUG
  }
}

static void fGgmlComputeForwardRegluF16(const ggml_compute_params *pParams, ggml_tensor *pDst) {

  const ggml_tensor *vSrc0 = pDst->src[0];
  const ggml_tensor *vSrc1 = pDst->src[1];
  char *vSrc0D = (char *)vSrc0->data;
  char *vSrc1D = (char *)(vSrc1 ? vSrc1->data : vSrc0->data);
  const size_t cSrc0O = vSrc0->nb[1];
  const size_t cSrc1O = vSrc1 ? vSrc1->nb[1] : vSrc0->nb[1];

  GGML_ASSERT(ggml_is_contiguous_1(vSrc0));
  GGML_ASSERT(ggml_is_contiguous_1(pDst));

  if (vSrc1) {
    GGML_ASSERT(ggml_is_contiguous_1(vSrc1));
    GGML_ASSERT(vSrc0->type == vSrc1->type);
  }

  const int cIth = pParams->ith;
  const int cNth = pParams->nth;

  const int cNc = vSrc1 ? vSrc0->ne[0] : vSrc0->ne[0] / 2;
  const int cNr = ggml_nrows(vSrc0);

  GGML_ASSERT(pDst->ne[0] == cNc);
  GGML_ASSERT(ggml_nrows(pDst) == cNr);

  const int32_t cSwapped = ggml_get_op_params_i32(pDst, 1);

  // rows per thread
  const int cDr = (cNr + cNth - 1) / cNth;

  // row range for this thread
  const int cIr0 = cDr * cIth;
  const int cIr1 = MIN(cIr0 + cDr, cNr);

  for (int vI1 = cIr0; vI1 < cIr1; vI1++) {
    ggml_fp16_t *vSrc0P = (ggml_fp16_t *)(vSrc0D + vI1 * cSrc0O);
    ggml_fp16_t *vSrc1P = (ggml_fp16_t *)(vSrc1D + vI1 * cSrc1O);

    if (!vSrc1) {
      vSrc0P += cSwapped ? cNc : 0;
      vSrc1P += cSwapped ? 0 : cNc;
    }

    ggml_vec_reglu_f16(cNc, (ggml_fp16_t *)((char *)pDst->data + vI1 * (pDst->nb[1])), vSrc0P, vSrc1P);

#ifndef NDEBUG
    for (int vK = 0; vK < cNc; vK++) {
      const ggml_fp16_t cX = ((ggml_fp16_t *)((char *)pDst->data + vI1 * (pDst->nb[1])))[vK];
      const float cV = GGML_FP16_TO_FP32(cX);
      GGML_UNUSED(cV);
      assert(!isnan(cV));
      assert(!isinf(cV));
    }
#endif // NDEBUG
  }
}

static void fGgmlComputeForwardReglu(const ggml_compute_params *pParams, ggml_tensor *pDst) {

  const ggml_tensor *vSrc0 = pDst->src[0];

  switch (vSrc0->type) {
  case GGML_TYPE_F32: {
    fGgmlComputeForwardRegluF32(pParams, pDst);
  } break;
  case GGML_TYPE_F16: {
    fGgmlComputeForwardRegluF16(pParams, pDst);
  } break;
  default: {
    GGML_ABORT("fatal error");
  }
  }
}

// ggml_compute_forward_geglu

static void fGgmlComputeForwardGegluF32(const ggml_compute_params *pParams, ggml_tensor *pDst) {

  const ggml_tensor *vSrc0 = pDst->src[0];
  const ggml_tensor *vSrc1 = pDst->src[1];
  char *vSrc0D = (char *)vSrc0->data;
  char *vSrc1D = (char *)(vSrc1 ? vSrc1->data : vSrc0->data);
  const size_t cSrc0O = vSrc0->nb[1];
  const size_t cSrc1O = vSrc1 ? vSrc1->nb[1] : vSrc0->nb[1];

  GGML_ASSERT(ggml_is_contiguous_1(vSrc0));
  GGML_ASSERT(ggml_is_contiguous_1(pDst));

  if (vSrc1) {
    GGML_ASSERT(ggml_is_contiguous_1(vSrc1));
    GGML_ASSERT(vSrc0->type == vSrc1->type);
  }

  const int cIth = pParams->ith;
  const int cNth = pParams->nth;

  const int cNc = vSrc1 ? vSrc0->ne[0] : vSrc0->ne[0] / 2;
  const int cNr = ggml_nrows(vSrc0);

  GGML_ASSERT(pDst->ne[0] == cNc);
  GGML_ASSERT(ggml_nrows(pDst) == cNr);

  const int32_t cSwapped = ggml_get_op_params_i32(pDst, 1);

  // rows per thread
  const int cDr = (cNr + cNth - 1) / cNth;

  // row range for this thread
  const int cIr0 = cDr * cIth;
  const int cIr1 = MIN(cIr0 + cDr, cNr);

  for (int vI1 = cIr0; vI1 < cIr1; vI1++) {
    float *vSrc0P = (float *)(vSrc0D + vI1 * cSrc0O);
    float *vSrc1P = (float *)(vSrc1D + vI1 * cSrc1O);

    if (!vSrc1) {
      vSrc0P += cSwapped ? cNc : 0;
      vSrc1P += cSwapped ? 0 : cNc;
    }

    ggml_vec_geglu_f32(cNc, (float *)((char *)pDst->data + vI1 * (pDst->nb[1])), vSrc0P, vSrc1P);

#ifndef NDEBUG
    for (int vK = 0; vK < cNc; vK++) {
      const float cX = ((float *)((char *)pDst->data + vI1 * (pDst->nb[1])))[vK];
      GGML_UNUSED(cX);
      assert(!isnan(cX));
      assert(!isinf(cX));
    }
#endif // NDEBUG
  }
}

static void fGgmlComputeForwardGegluF16(const ggml_compute_params *pParams, ggml_tensor *pDst) {

  const ggml_tensor *vSrc0 = pDst->src[0];
  const ggml_tensor *vSrc1 = pDst->src[1];
  char *vSrc0D = (char *)vSrc0->data;
  char *vSrc1D = (char *)(vSrc1 ? vSrc1->data : vSrc0->data);
  const size_t cSrc0O = vSrc0->nb[1];
  const size_t cSrc1O = vSrc1 ? vSrc1->nb[1] : vSrc0->nb[1];

  GGML_ASSERT(ggml_is_contiguous_1(vSrc0));
  GGML_ASSERT(ggml_is_contiguous_1(pDst));

  if (vSrc1) {
    GGML_ASSERT(ggml_is_contiguous_1(vSrc1));
    GGML_ASSERT(vSrc0->type == vSrc1->type);
  }

  const int cIth = pParams->ith;
  const int cNth = pParams->nth;

  const int cNc = vSrc1 ? vSrc0->ne[0] : vSrc0->ne[0] / 2;
  const int cNr = ggml_nrows(vSrc0);

  GGML_ASSERT(pDst->ne[0] == cNc);
  GGML_ASSERT(ggml_nrows(pDst) == cNr);

  const int32_t cSwapped = ggml_get_op_params_i32(pDst, 1);

  // rows per thread
  const int cDr = (cNr + cNth - 1) / cNth;

  // row range for this thread
  const int cIr0 = cDr * cIth;
  const int cIr1 = MIN(cIr0 + cDr, cNr);

  for (int vI1 = cIr0; vI1 < cIr1; vI1++) {
    ggml_fp16_t *vSrc0P = (ggml_fp16_t *)(vSrc0D + vI1 * cSrc0O);
    ggml_fp16_t *vSrc1P = (ggml_fp16_t *)(vSrc1D + vI1 * cSrc1O);

    if (!vSrc1) {
      vSrc0P += cSwapped ? cNc : 0;
      vSrc1P += cSwapped ? 0 : cNc;
    }

    ggml_vec_geglu_f16(cNc, (ggml_fp16_t *)((char *)pDst->data + vI1 * (pDst->nb[1])), vSrc0P, vSrc1P);

#ifndef NDEBUG
    for (int vK = 0; vK < cNc; vK++) {
      const ggml_fp16_t cX = ((ggml_fp16_t *)((char *)pDst->data + vI1 * (pDst->nb[1])))[vK];
      const float cV = GGML_FP16_TO_FP32(cX);
      GGML_UNUSED(cV);
      assert(!isnan(cV));
      assert(!isinf(cV));
    }
#endif // NDEBUG
  }
}

static void fGgmlComputeForwardGeglu(const ggml_compute_params *pParams, ggml_tensor *pDst) {

  const ggml_tensor *vSrc0 = pDst->src[0];

  switch (vSrc0->type) {
  case GGML_TYPE_F32: {
    fGgmlComputeForwardGegluF32(pParams, pDst);
  } break;
  case GGML_TYPE_F16: {
    fGgmlComputeForwardGegluF16(pParams, pDst);
  } break;
  default: {
    GGML_ABORT("fatal error");
  }
  }
}

// ggml_compute_forward_swiglu

static void fGgmlComputeForwardSwigluF32(const ggml_compute_params *pParams, ggml_tensor *pDst) {

  const ggml_tensor *vSrc0 = pDst->src[0];
  const ggml_tensor *vSrc1 = pDst->src[1];
  char *vSrc0D = (char *)vSrc0->data;
  char *vSrc1D = (char *)(vSrc1 ? vSrc1->data : vSrc0->data);
  const size_t cSrc0O = vSrc0->nb[1];
  const size_t cSrc1O = vSrc1 ? vSrc1->nb[1] : vSrc0->nb[1];

  GGML_ASSERT(ggml_is_contiguous_1(vSrc0));
  GGML_ASSERT(ggml_is_contiguous_1(pDst));

  if (vSrc1) {
    GGML_ASSERT(ggml_is_contiguous_1(vSrc1));
    GGML_ASSERT(vSrc0->type == vSrc1->type);
  }

  const int cIth = pParams->ith;
  const int cNth = pParams->nth;

  const int cNc = vSrc1 ? vSrc0->ne[0] : vSrc0->ne[0] / 2;
  const int cNr = ggml_nrows(vSrc0);

  GGML_ASSERT(pDst->ne[0] == cNc);
  GGML_ASSERT(ggml_nrows(pDst) == cNr);

  const int32_t cSwapped = ggml_get_op_params_i32(pDst, 1);

  // rows per thread
  const int cDr = (cNr + cNth - 1) / cNth;

  // row range for this thread
  const int cIr0 = cDr * cIth;
  const int cIr1 = MIN(cIr0 + cDr, cNr);

  for (int vI1 = cIr0; vI1 < cIr1; vI1++) {
    float *vSrc0P = (float *)(vSrc0D + vI1 * cSrc0O);
    float *vSrc1P = (float *)(vSrc1D + vI1 * cSrc1O);

    if (!vSrc1) {
      vSrc0P += cSwapped ? cNc : 0;
      vSrc1P += cSwapped ? 0 : cNc;
    }

    ggml_vec_swiglu_f32(cNc, (float *)((char *)pDst->data + vI1 * (pDst->nb[1])), vSrc0P, vSrc1P);

#ifndef NDEBUG
    for (int vK = 0; vK < cNc; vK++) {
      const float cX = ((float *)((char *)pDst->data + vI1 * (pDst->nb[1])))[vK];
      GGML_UNUSED(cX);
      assert(!isnan(cX));
      assert(!isinf(cX));
    }
#endif // NDEBUG
  }
}

static void fGgmlComputeForwardSwigluF16(const ggml_compute_params *pParams, ggml_tensor *pDst) {

  const ggml_tensor *vSrc0 = pDst->src[0];
  const ggml_tensor *vSrc1 = pDst->src[1];
  char *vSrc0D = (char *)vSrc0->data;
  char *vSrc1D = (char *)(vSrc1 ? vSrc1->data : vSrc0->data);
  const size_t cSrc0O = vSrc0->nb[1];
  const size_t cSrc1O = vSrc1 ? vSrc1->nb[1] : vSrc0->nb[1];

  GGML_ASSERT(ggml_is_contiguous_1(vSrc0));
  GGML_ASSERT(ggml_is_contiguous_1(pDst));

  if (vSrc1) {
    GGML_ASSERT(ggml_is_contiguous_1(vSrc1));
    GGML_ASSERT(vSrc0->type == vSrc1->type);
  }

  const int cIth = pParams->ith;
  const int cNth = pParams->nth;

  const int cNc = vSrc1 ? vSrc0->ne[0] : vSrc0->ne[0] / 2;
  const int cNr = ggml_nrows(vSrc0);

  GGML_ASSERT(pDst->ne[0] == cNc);
  GGML_ASSERT(ggml_nrows(pDst) == cNr);

  const int32_t cSwapped = ggml_get_op_params_i32(pDst, 1);

  // rows per thread
  const int cDr = (cNr + cNth - 1) / cNth;

  // row range for this thread
  const int cIr0 = cDr * cIth;
  const int cIr1 = MIN(cIr0 + cDr, cNr);

  for (int vI1 = cIr0; vI1 < cIr1; vI1++) {
    ggml_fp16_t *vSrc0P = (ggml_fp16_t *)(vSrc0D + vI1 * cSrc0O);
    ggml_fp16_t *vSrc1P = (ggml_fp16_t *)(vSrc1D + vI1 * cSrc1O);

    if (!vSrc1) {
      vSrc0P += cSwapped ? cNc : 0;
      vSrc1P += cSwapped ? 0 : cNc;
    }

    ggml_vec_swiglu_f16(cNc, (ggml_fp16_t *)((char *)pDst->data + vI1 * (pDst->nb[1])), vSrc0P, vSrc1P);

#ifndef NDEBUG
    for (int vK = 0; vK < cNc; vK++) {
      const ggml_fp16_t cX = ((ggml_fp16_t *)((char *)pDst->data + vI1 * (pDst->nb[1])))[vK];
      const float cV = GGML_FP16_TO_FP32(cX);
      GGML_UNUSED(cV);
      assert(!isnan(cV));
      assert(!isinf(cV));
    }
#endif // NDEBUG
  }
}

static void fGgmlComputeForwardSwiglu(const ggml_compute_params *pParams, ggml_tensor *pDst) {

  const ggml_tensor *vSrc0 = pDst->src[0];

  switch (vSrc0->type) {
  case GGML_TYPE_F32: {
    fGgmlComputeForwardSwigluF32(pParams, pDst);
  } break;
  case GGML_TYPE_F16: {
    fGgmlComputeForwardSwigluF16(pParams, pDst);
  } break;
  default: {
    GGML_ABORT("fatal error");
  }
  }
}

// ggml_compute_forward_swiglu_oai

static void fGgmlComputeForwardSwigluOaiF32(const ggml_compute_params *pParams, ggml_tensor *pDst) {

  const ggml_tensor *vSrc0 = pDst->src[0];
  const ggml_tensor *vSrc1 = pDst->src[1];
  char *vSrc0D = (char *)vSrc0->data;
  char *vSrc1D = (char *)(vSrc1 ? vSrc1->data : vSrc0->data);
  const size_t cSrc0O = vSrc0->nb[1];
  const size_t cSrc1O = vSrc1 ? vSrc1->nb[1] : vSrc0->nb[1];

  GGML_ASSERT(ggml_is_contiguous_1(vSrc0));
  GGML_ASSERT(ggml_is_contiguous_1(pDst));

  if (vSrc1) {
    GGML_ASSERT(ggml_is_contiguous_1(vSrc1));
    GGML_ASSERT(vSrc0->type == vSrc1->type);
  }

  const int cIth = pParams->ith;
  const int cNth = pParams->nth;

  const int cNc = vSrc1 ? vSrc0->ne[0] : vSrc0->ne[0] / 2;
  const int cNr = ggml_nrows(vSrc0);

  GGML_ASSERT(pDst->ne[0] == cNc);
  GGML_ASSERT(ggml_nrows(pDst) == cNr);

  const int32_t cSwapped = ggml_get_op_params_i32(pDst, 1);
  const float cAlpha = ggml_get_op_params_f32(pDst, 2);
  const float cLimit = ggml_get_op_params_f32(pDst, 3);

  // rows per thread
  const int cDr = (cNr + cNth - 1) / cNth;

  // row range for this thread
  const int cIr0 = cDr * cIth;
  const int cIr1 = MIN(cIr0 + cDr, cNr);

  for (int vI1 = cIr0; vI1 < cIr1; vI1++) {
    float *vSrc0P = (float *)(vSrc0D + vI1 * cSrc0O);
    float *vSrc1P = (float *)(vSrc1D + vI1 * cSrc1O);
    float *vDstP = (float *)((char *)pDst->data + vI1 * (pDst->nb[1]));

    if (!vSrc1) {
      vSrc0P += cSwapped ? cNc : 0;
      vSrc1P += cSwapped ? 0 : cNc;
    }

    for (int vK = 0; vK < cNc; vK++) {
      const float cX = std::min(vSrc0P[vK], cLimit);
      const float cY = std::clamp(vSrc1P[vK], -cLimit, cLimit);
      const float cOutGlu = cX / (1.f + expf(cAlpha * (-cX)));
      vDstP[vK] = cOutGlu * (cY + 1.f);
    }

#ifndef NDEBUG
    for (int vK = 0; vK < cNc; vK++) {
      const float cX = vDstP[vK];
      GGML_UNUSED(cX);
      assert(!isnan(cX));
      assert(!isinf(cX));
    }
#endif // NDEBUG
  }
}

static void fGgmlComputeForwardSwigluOai(const ggml_compute_params *pParams, ggml_tensor *pDst) {

  const ggml_tensor *vSrc0 = pDst->src[0];

  switch (vSrc0->type) {
  case GGML_TYPE_F32: {
    fGgmlComputeForwardSwigluOaiF32(pParams, pDst);
  } break;
  default: {
    GGML_ABORT("fatal error");
  }
  }
}

// ggml_compute_forward_swiglu_clamp

static void fGgmlComputeForwardSwigluClampF32(const ggml_compute_params *pParams, ggml_tensor *pDst) {
  const ggml_tensor *vSrc0 = pDst->src[0];
  const ggml_tensor *vSrc1 = pDst->src[1];
  char *vSrc0D = (char *)vSrc0->data;
  char *vSrc1D = (char *)(vSrc1 ? vSrc1->data : vSrc0->data);
  const size_t cSrc0O = vSrc0->nb[1];
  const size_t cSrc1O = vSrc1 ? vSrc1->nb[1] : vSrc0->nb[1];

  GGML_ASSERT(ggml_is_contiguous_1(vSrc0));
  GGML_ASSERT(ggml_is_contiguous_1(pDst));

  if (vSrc1) {
    GGML_ASSERT(ggml_is_contiguous_1(vSrc1));
    GGML_ASSERT(vSrc0->type == vSrc1->type);
  }

  const int cIth = pParams->ith;
  const int cNth = pParams->nth;

  const int cNc = vSrc1 ? vSrc0->ne[0] : vSrc0->ne[0] / 2;
  const int cNr = ggml_nrows(vSrc0);

  GGML_ASSERT(pDst->ne[0] == cNc);
  GGML_ASSERT(ggml_nrows(pDst) == cNr);

  const int32_t cSwapped = ggml_get_op_params_i32(pDst, 1);
  const float cLimit = ggml_get_op_params_f32(pDst, 3);

  const int cDr = (cNr + cNth - 1) / cNth;
  const int cIr0 = cDr * cIth;
  const int cIr1 = MIN(cIr0 + cDr, cNr);

  for (int vI1 = cIr0; vI1 < cIr1; vI1++) {
    float *vSrc0P = (float *)(vSrc0D + vI1 * cSrc0O);
    float *vSrc1P = (float *)(vSrc1D + vI1 * cSrc1O);
    float *vDstP = (float *)((char *)pDst->data + vI1 * (pDst->nb[1]));

    if (!vSrc1) {
      vSrc0P += cSwapped ? cNc : 0;
      vSrc1P += cSwapped ? 0 : cNc;
    }

    for (int vK = 0; vK < cNc; vK++) {
      const float cGate = std::min(vSrc0P[vK], cLimit);
      const float cUp = std::clamp(vSrc1P[vK], -cLimit, cLimit);
      vDstP[vK] = cGate / (1.f + expf(-cGate)) * cUp;
    }

#ifndef NDEBUG
    for (int vK = 0; vK < cNc; vK++) {
      const float cX = vDstP[vK];
      GGML_UNUSED(cX);
      assert(!isnan(cX));
      assert(!isinf(cX));
    }
#endif // NDEBUG
  }
}

static void fGgmlComputeForwardSwigluClampF16(const ggml_compute_params *pParams, ggml_tensor *pDst) {
  const ggml_tensor *vSrc0 = pDst->src[0];
  const ggml_tensor *vSrc1 = pDst->src[1];
  char *vSrc0D = (char *)vSrc0->data;
  char *vSrc1D = (char *)(vSrc1 ? vSrc1->data : vSrc0->data);
  const size_t cSrc0O = vSrc0->nb[1];
  const size_t cSrc1O = vSrc1 ? vSrc1->nb[1] : vSrc0->nb[1];

  GGML_ASSERT(ggml_is_contiguous_1(vSrc0));
  GGML_ASSERT(ggml_is_contiguous_1(pDst));

  if (vSrc1) {
    GGML_ASSERT(ggml_is_contiguous_1(vSrc1));
    GGML_ASSERT(vSrc0->type == vSrc1->type);
  }

  const int cIth = pParams->ith;
  const int cNth = pParams->nth;

  const int cNc = vSrc1 ? vSrc0->ne[0] : vSrc0->ne[0] / 2;
  const int cNr = ggml_nrows(vSrc0);

  GGML_ASSERT(pDst->ne[0] == cNc);
  GGML_ASSERT(ggml_nrows(pDst) == cNr);

  const int32_t cSwapped = ggml_get_op_params_i32(pDst, 1);
  const float cLimit = ggml_get_op_params_f32(pDst, 3);

  const int cDr = (cNr + cNth - 1) / cNth;
  const int cIr0 = cDr * cIth;
  const int cIr1 = MIN(cIr0 + cDr, cNr);

  for (int vI1 = cIr0; vI1 < cIr1; vI1++) {
    ggml_fp16_t *vSrc0P = (ggml_fp16_t *)(vSrc0D + vI1 * cSrc0O);
    ggml_fp16_t *vSrc1P = (ggml_fp16_t *)(vSrc1D + vI1 * cSrc1O);
    ggml_fp16_t *vDstP = (ggml_fp16_t *)((char *)pDst->data + vI1 * (pDst->nb[1]));

    if (!vSrc1) {
      vSrc0P += cSwapped ? cNc : 0;
      vSrc1P += cSwapped ? 0 : cNc;
    }

    for (int vK = 0; vK < cNc; vK++) {
      const float cGate = std::min(GGML_FP16_TO_FP32(vSrc0P[vK]), cLimit);
      const float cUp = std::clamp(GGML_FP16_TO_FP32(vSrc1P[vK]), -cLimit, cLimit);
      vDstP[vK] = GGML_FP32_TO_FP16(cGate / (1.f + expf(-cGate)) * cUp);
    }

#ifndef NDEBUG
    for (int vK = 0; vK < cNc; vK++) {
      const float cX = GGML_FP16_TO_FP32(vDstP[vK]);
      GGML_UNUSED(cX);
      assert(!isnan(cX));
      assert(!isinf(cX));
    }
#endif // NDEBUG
  }
}

static void fGgmlComputeForwardSwigluClamp(const ggml_compute_params *pParams, ggml_tensor *pDst) {
  switch (pDst->src[0]->type) {
  case GGML_TYPE_F32:
    fGgmlComputeForwardSwigluClampF32(pParams, pDst);
    break;
  case GGML_TYPE_F16:
    fGgmlComputeForwardSwigluClampF16(pParams, pDst);
    break;
  default:
    GGML_ABORT("fatal error");
  }
}

// ggml_compute_forward_geglu_erf

static void fGgmlComputeForwardGegluErfF32(const ggml_compute_params *pParams, ggml_tensor *pDst) {

  const ggml_tensor *vSrc0 = pDst->src[0];
  const ggml_tensor *vSrc1 = pDst->src[1];
  char *vSrc0D = (char *)vSrc0->data;
  char *vSrc1D = (char *)(vSrc1 ? vSrc1->data : vSrc0->data);
  const size_t cSrc0O = vSrc0->nb[1];
  const size_t cSrc1O = vSrc1 ? vSrc1->nb[1] : vSrc0->nb[1];

  GGML_ASSERT(ggml_is_contiguous_1(vSrc0));
  GGML_ASSERT(ggml_is_contiguous_1(pDst));

  if (vSrc1) {
    GGML_ASSERT(ggml_is_contiguous_1(vSrc1));
    GGML_ASSERT(vSrc0->type == vSrc1->type);
  }

  const int cIth = pParams->ith;
  const int cNth = pParams->nth;

  const int cNc = vSrc1 ? vSrc0->ne[0] : vSrc0->ne[0] / 2;
  const int cNr = ggml_nrows(vSrc0);

  GGML_ASSERT(pDst->ne[0] == cNc);
  GGML_ASSERT(ggml_nrows(pDst) == cNr);

  const int32_t cSwapped = ggml_get_op_params_i32(pDst, 1);

  // rows per thread
  const int cDr = (cNr + cNth - 1) / cNth;

  // row range for this thread
  const int cIr0 = cDr * cIth;
  const int cIr1 = MIN(cIr0 + cDr, cNr);

  for (int vI1 = cIr0; vI1 < cIr1; vI1++) {
    float *vSrc0P = (float *)(vSrc0D + vI1 * cSrc0O);
    float *vSrc1P = (float *)(vSrc1D + vI1 * cSrc1O);

    if (!vSrc1) {
      vSrc0P += cSwapped ? cNc : 0;
      vSrc1P += cSwapped ? 0 : cNc;
    }

    ggml_vec_geglu_erf_f32(cNc, (float *)((char *)pDst->data + vI1 * (pDst->nb[1])), vSrc0P, vSrc1P);

#ifndef NDEBUG
    for (int vK = 0; vK < cNc; vK++) {
      const float cX = ((float *)((char *)pDst->data + vI1 * (pDst->nb[1])))[vK];
      GGML_UNUSED(cX);
      assert(!isnan(cX));
      assert(!isinf(cX));
    }
#endif // NDEBUG
  }
}

static void fGgmlComputeForwardGegluErfF16(const ggml_compute_params *pParams, ggml_tensor *pDst) {

  const ggml_tensor *vSrc0 = pDst->src[0];
  const ggml_tensor *vSrc1 = pDst->src[1];
  char *vSrc0D = (char *)vSrc0->data;
  char *vSrc1D = (char *)(vSrc1 ? vSrc1->data : vSrc0->data);
  const size_t cSrc0O = vSrc0->nb[1];
  const size_t cSrc1O = vSrc1 ? vSrc1->nb[1] : vSrc0->nb[1];

  GGML_ASSERT(ggml_is_contiguous_1(vSrc0));
  GGML_ASSERT(ggml_is_contiguous_1(pDst));

  if (vSrc1) {
    GGML_ASSERT(ggml_is_contiguous_1(vSrc1));
    GGML_ASSERT(vSrc0->type == vSrc1->type);
  }

  const int cIth = pParams->ith;
  const int cNth = pParams->nth;

  const int cNc = vSrc1 ? vSrc0->ne[0] : vSrc0->ne[0] / 2;
  const int cNr = ggml_nrows(vSrc0);

  GGML_ASSERT(pDst->ne[0] == cNc);
  GGML_ASSERT(ggml_nrows(pDst) == cNr);

  const int32_t cSwapped = ggml_get_op_params_i32(pDst, 1);

  // rows per thread
  const int cDr = (cNr + cNth - 1) / cNth;

  // row range for this thread
  const int cIr0 = cDr * cIth;
  const int cIr1 = MIN(cIr0 + cDr, cNr);

  for (int vI1 = cIr0; vI1 < cIr1; vI1++) {
    ggml_fp16_t *vSrc0P = (ggml_fp16_t *)(vSrc0D + vI1 * cSrc0O);
    ggml_fp16_t *vSrc1P = (ggml_fp16_t *)(vSrc1D + vI1 * cSrc1O);

    if (!vSrc1) {
      vSrc0P += cSwapped ? cNc : 0;
      vSrc1P += cSwapped ? 0 : cNc;
    }

    ggml_vec_geglu_erf_f16(cNc, (ggml_fp16_t *)((char *)pDst->data + vI1 * (pDst->nb[1])), vSrc0P, vSrc1P);

#ifndef NDEBUG
    for (int vK = 0; vK < cNc; vK++) {
      const ggml_fp16_t cX = ((ggml_fp16_t *)((char *)pDst->data + vI1 * (pDst->nb[1])))[vK];
      const float cV = GGML_FP16_TO_FP32(cX);
      GGML_UNUSED(cV);
      assert(!isnan(cV));
      assert(!isinf(cV));
    }
#endif // NDEBUG
  }
}

static void fGgmlComputeForwardGegluErf(const ggml_compute_params *pParams, ggml_tensor *pDst) {

  const ggml_tensor *vSrc0 = pDst->src[0];

  switch (vSrc0->type) {
  case GGML_TYPE_F32: {
    fGgmlComputeForwardGegluErfF32(pParams, pDst);
  } break;
  case GGML_TYPE_F16: {
    fGgmlComputeForwardGegluErfF16(pParams, pDst);
  } break;
  default: {
    GGML_ABORT("fatal error");
  }
  }
}

// ggml_compute_forward_geglu_quick

static void fGgmlComputeForwardGegluQuickF32(const ggml_compute_params *pParams, ggml_tensor *pDst) {

  const ggml_tensor *vSrc0 = pDst->src[0];
  const ggml_tensor *vSrc1 = pDst->src[1];
  char *vSrc0D = (char *)vSrc0->data;
  char *vSrc1D = (char *)(vSrc1 ? vSrc1->data : vSrc0->data);
  const size_t cSrc0O = vSrc0->nb[1];
  const size_t cSrc1O = vSrc1 ? vSrc1->nb[1] : vSrc0->nb[1];

  GGML_ASSERT(ggml_is_contiguous_1(vSrc0));
  GGML_ASSERT(ggml_is_contiguous_1(pDst));

  if (vSrc1) {
    GGML_ASSERT(ggml_is_contiguous_1(vSrc1));
    GGML_ASSERT(vSrc0->type == vSrc1->type);
  }

  const int cIth = pParams->ith;
  const int cNth = pParams->nth;

  const int cNc = vSrc1 ? vSrc0->ne[0] : vSrc0->ne[0] / 2;
  const int cNr = ggml_nrows(vSrc0);

  GGML_ASSERT(pDst->ne[0] == cNc);
  GGML_ASSERT(ggml_nrows(pDst) == cNr);

  const int32_t cSwapped = ggml_get_op_params_i32(pDst, 1);

  // rows per thread
  const int cDr = (cNr + cNth - 1) / cNth;

  // row range for this thread
  const int cIr0 = cDr * cIth;
  const int cIr1 = MIN(cIr0 + cDr, cNr);

  for (int vI1 = cIr0; vI1 < cIr1; vI1++) {
    float *vSrc0P = (float *)(vSrc0D + vI1 * cSrc0O);
    float *vSrc1P = (float *)(vSrc1D + vI1 * cSrc1O);

    if (!vSrc1) {
      vSrc0P += cSwapped ? cNc : 0;
      vSrc1P += cSwapped ? 0 : cNc;
    }

    ggml_vec_geglu_quick_f32(cNc, (float *)((char *)pDst->data + vI1 * (pDst->nb[1])), vSrc0P, vSrc1P);

#ifndef NDEBUG
    for (int vK = 0; vK < cNc; vK++) {
      const float cX = ((float *)((char *)pDst->data + vI1 * (pDst->nb[1])))[vK];
      GGML_UNUSED(cX);
      assert(!isnan(cX));
      assert(!isinf(cX));
    }
#endif // NDEBUG
  }
}

static void fGgmlComputeForwardGegluQuickF16(const ggml_compute_params *pParams, ggml_tensor *pDst) {

  const ggml_tensor *vSrc0 = pDst->src[0];
  const ggml_tensor *vSrc1 = pDst->src[1];
  char *vSrc0D = (char *)vSrc0->data;
  char *vSrc1D = (char *)(vSrc1 ? vSrc1->data : vSrc0->data);
  const size_t cSrc0O = vSrc0->nb[1];
  const size_t cSrc1O = vSrc1 ? vSrc1->nb[1] : vSrc0->nb[1];

  GGML_ASSERT(ggml_is_contiguous_1(vSrc0));
  GGML_ASSERT(ggml_is_contiguous_1(pDst));

  if (vSrc1) {
    GGML_ASSERT(ggml_is_contiguous_1(vSrc1));
    GGML_ASSERT(vSrc0->type == vSrc1->type);
  }

  const int cIth = pParams->ith;
  const int cNth = pParams->nth;

  const int cNc = vSrc1 ? vSrc0->ne[0] : vSrc0->ne[0] / 2;
  const int cNr = ggml_nrows(vSrc0);

  GGML_ASSERT(pDst->ne[0] == cNc);
  GGML_ASSERT(ggml_nrows(pDst) == cNr);

  const int32_t cSwapped = ggml_get_op_params_i32(pDst, 1);

  // rows per thread
  const int cDr = (cNr + cNth - 1) / cNth;

  // row range for this thread
  const int cIr0 = cDr * cIth;
  const int cIr1 = MIN(cIr0 + cDr, cNr);

  for (int vI1 = cIr0; vI1 < cIr1; vI1++) {
    ggml_fp16_t *vSrc0P = (ggml_fp16_t *)(vSrc0D + vI1 * cSrc0O);
    ggml_fp16_t *vSrc1P = (ggml_fp16_t *)(vSrc1D + vI1 * cSrc1O);

    if (!vSrc1) {
      vSrc0P += cSwapped ? cNc : 0;
      vSrc1P += cSwapped ? 0 : cNc;
    }

    ggml_vec_geglu_quick_f16(cNc, (ggml_fp16_t *)((char *)pDst->data + vI1 * (pDst->nb[1])), vSrc0P, vSrc1P);

#ifndef NDEBUG
    for (int vK = 0; vK < cNc; vK++) {
      const ggml_fp16_t cX = ((ggml_fp16_t *)((char *)pDst->data + vI1 * (pDst->nb[1])))[vK];
      const float cV = GGML_FP16_TO_FP32(cX);
      GGML_UNUSED(cV);
      assert(!isnan(cV));
      assert(!isinf(cV));
    }
#endif // NDEBUG
  }
}

static void fGgmlComputeForwardGegluQuick(const ggml_compute_params *pParams, ggml_tensor *pDst) {

  const ggml_tensor *vSrc0 = pDst->src[0];

  switch (vSrc0->type) {
  case GGML_TYPE_F32: {
    fGgmlComputeForwardGegluQuickF32(pParams, pDst);
  } break;
  case GGML_TYPE_F16: {
    fGgmlComputeForwardGegluQuickF16(pParams, pDst);
  } break;
  default: {
    GGML_ABORT("fatal error");
  }
  }
}

// ggml_compute_forward_norm

static void fGgmlComputeForwardNormF32(const ggml_compute_params *pParams, ggml_tensor *dst) {

  const ggml_tensor *src0 = dst->src[0];

  GGML_ASSERT(ggml_are_same_shape(src0, dst));

  const int cIth = pParams->ith;
  const int cNth = pParams->nth;

  GGML_TENSOR_UNARY_OP_LOCALS

  float vEps;
  memcpy(&vEps, dst->op_params, sizeof(float));

  GGML_ASSERT(vEps >= 0.0f);

  for (int64_t vI03 = 0; vI03 < ne03; vI03++) {
    for (int64_t vI02 = 0; vI02 < ne02; vI02++) {
      for (int64_t vI01 = cIth; vI01 < ne01; vI01 += cNth) {
        const char *vX = (const char *)src0->data + vI01 * nb01 + vI02 * nb02 + vI03 * nb03;
        char *vY = (char *)dst->data + vI01 * nb1 + vI02 * nb2 + vI03 * nb3;

        if (nb00 == sizeof(float) && nb0 == sizeof(float)) {
          const float *vXf = (const float *)vX;

          float vSum = 0.0;
          ggml_vec_sum_f32(ne00, &vSum, vXf);
          float mean = vSum / ne00;

          float *vYf = (float *)vY;
          float vVariance = 0;

#ifdef GGML_USE_ACCELERATE
          mean = -mean;
          vDSP_vsadd(vXf, 1, &mean, vYf, 1, ne00);
          vDSP_measqv(vYf, 1, &vVariance, ne00);
#else
          vVariance = ggml_vec_cvar_f32(ne00, vYf, vXf, mean);
#endif // GGML_USE_ACCELERATE

          const float cScale = 1.0f / sqrtf(vVariance + vEps);
          ggml_vec_scale_f32(ne00, vYf, cScale);
        } else {
          float vSum = 0.0;
          for (int64_t vI00 = 0; vI00 < ne00; vI00++) {
            vSum += *(const float *)(vX + vI00 * nb00);
          }
          const float mean = vSum / ne00;

          float vVariance = 0.0f;
          for (int64_t vI00 = 0; vI00 < ne00; vI00++) {
            const float cV = *(const float *)(vX + vI00 * nb00) - mean;
            *(float *)(vY + vI00 * nb0) = cV;
            vVariance += cV * cV;
          }
          vVariance /= ne00;

          const float cScale = 1.0f / sqrtf(vVariance + vEps);
          for (int64_t vI00 = 0; vI00 < ne00; vI00++) {
            *(float *)(vY + vI00 * nb0) *= cScale;
          }
        }
      }
    }
  }
}

void ggml_compute_forward_norm(const ggml_compute_params *pParams, ggml_tensor *pDst) {

  const ggml_tensor *vSrc0 = pDst->src[0];

  switch (vSrc0->type) {
  case GGML_TYPE_F32: {
    fGgmlComputeForwardNormF32(pParams, pDst);
  } break;
  default: {
    GGML_ABORT("fatal error");
  }
  }
}

// ggml_compute_forward_group_rms_norm

// fusion kinds that can be combined with the rms_norm computation in a single pass.
// extend this enum when adding new fused variants (e.g. FUSE_ADD, FUSE_MUL_ADD, ...).
enum ggml_rms_norm_fuse_op {
  GGML_RMS_NORM_FUSE_OP_NONE,
  GGML_RMS_NORM_FUSE_OP_MUL,
};

template <ggml_rms_norm_fuse_op FUSE_OP>
static void fGgmlComputeForwardRmsNormF32(const ggml_compute_params *pParams, ggml_tensor *pDstRmsNorm,
                                          ggml_tensor *pDstFused = nullptr) {

  const ggml_tensor *src0 = pDstRmsNorm->src[0];
  const ggml_tensor *src1 = nullptr;
  ggml_tensor *dst = pDstRmsNorm;

  if constexpr (FUSE_OP == GGML_RMS_NORM_FUSE_OP_MUL) {
    src1 = (pDstFused->src[0] == pDstRmsNorm) ? pDstFused->src[1] : pDstFused->src[0];
    dst = pDstFused;
  }

  GGML_ASSERT(ggml_are_same_shape(src0, dst));

  GGML_ASSERT(src0->nb[0] == sizeof(float));

  const int cIth = pParams->ith;
  const int cNth = pParams->nth;

  GGML_TENSOR_BINARY_OP_LOCALS

  float vEps;
  memcpy(&vEps, pDstRmsNorm->op_params, sizeof(float));
  GGML_ASSERT(vEps >= 0.0f);

  // TODO: optimize
  for (int64_t vI03 = 0; vI03 < ne03; vI03++) {
    for (int64_t vI02 = 0; vI02 < ne02; vI02++) {
      for (int64_t vI01 = cIth; vI01 < ne01; vI01 += cNth) {
        const float *vX = (float *)((char *)src0->data + vI01 * nb01 + vI02 * nb02 + vI03 * nb03);

        ggml_float vSum = 0.0;
        // worth switching to explicit SIMD?
        for (int64_t vI00 = 0; vI00 < ne00; vI00++) {
          vSum += (ggml_float)(vX[vI00] * vX[vI00]);
        }

        const float cMean = vSum / ne00;
        const float cScale = 1.0f / sqrtf(cMean + vEps);

        // if you hit this, likely you got an inf somewhere earlier
        assert(cScale > 0.0f);

        float *vY = (float *)((char *)dst->data + vI01 * nb1 + vI02 * nb2 + vI03 * nb3);

        if constexpr (FUSE_OP == GGML_RMS_NORM_FUSE_OP_MUL) {
          const int64_t cI11 = vI01 % ne11;
          const int64_t cI12 = vI02 % ne12;
          const int64_t cI13 = vI03 % ne13;
          const float *vW = (float *)((char *)src1->data + cI11 * nb11 + cI12 * nb12 + cI13 * nb13);

          for (int64_t vI00 = 0; vI00 < ne00; vI00++) {
            vY[vI00] = vX[vI00] * cScale * vW[vI00];
          }
        } else {
          memcpy(vY, vX, ne00 * sizeof(float));
          ggml_vec_scale_f32(ne00, vY, cScale);
        }
      }
    }
  }
}

void ggml_compute_forward_rms_norm(const ggml_compute_params *pParams, ggml_tensor *pDst) {

  const ggml_tensor *vSrc0 = pDst->src[0];

  switch (vSrc0->type) {
  case GGML_TYPE_F32: {
    fGgmlComputeForwardRmsNormF32<GGML_RMS_NORM_FUSE_OP_NONE>(pParams, pDst);
  } break;
  default: {
    GGML_ABORT("fatal error");
  }
  }
}

// Fused RMS_NORM + MUL: computes dst = rms_norm(src0) * src1 in a single pass.
// This avoids materializing the intermediate rms_norm result in memory.
void ggml_compute_forward_rms_norm_mul_fused(const ggml_compute_params *pParams, ggml_tensor *pDstRmsNorm,
                                             ggml_tensor *pDstMul) {

  GGML_ASSERT(pDstMul != nullptr);
  GGML_ASSERT(pDstMul->src[0] == pDstRmsNorm || pDstMul->src[1] == pDstRmsNorm);

  const ggml_tensor *vSrc0 = pDstRmsNorm->src[0];

  switch (vSrc0->type) {
  case GGML_TYPE_F32: {
    fGgmlComputeForwardRmsNormF32<GGML_RMS_NORM_FUSE_OP_MUL>(pParams, pDstRmsNorm, pDstMul);
  } break;
  default: {
    GGML_ABORT("fatal error");
  }
  }
}

static void fGgmlComputeForwardRmsNormBackF32(const ggml_compute_params *pParams, ggml_tensor *dst) {

  const ggml_tensor *src0 = dst->src[0]; // gradients from forward pass output
  const ggml_tensor *src1 = dst->src[1]; // src1 from forward pass

  GGML_ASSERT(ggml_are_same_shape(src0, dst) && ggml_are_same_shape(src0, src1));

  GGML_ASSERT(src0->nb[0] == sizeof(float));
  GGML_ASSERT(src1->nb[0] == sizeof(float));

  const int cIth = pParams->ith;
  const int cNth = pParams->nth;

  GGML_TENSOR_BINARY_OP_LOCALS

  float vEps;
  memcpy(&vEps, dst->op_params, sizeof(float));

  // TODO: optimize
  for (int64_t vI03 = 0; vI03 < ne03; vI03++) {
    for (int64_t vI02 = 0; vI02 < ne02; vI02++) {
      for (int64_t vI01 = cIth; vI01 < ne01; vI01 += cNth) {
        // src1 is same shape as src0 => same indices
        const int64_t cI11 = vI01;
        const int64_t cI12 = vI02;
        const int64_t cI13 = vI03;

        const float *vDz = (float *)((char *)src0->data + vI01 * nb01 + vI02 * nb02 + vI03 * nb03);
        const float *vX = (float *)((char *)src1->data + cI11 * nb11 + cI12 * nb12 + cI13 * nb13);

        ggml_float vSumXx = 0.0;
        ggml_float vSumXdz = 0.0;

        for (int64_t vI00 = 0; vI00 < ne00; vI00++) {
          vSumXx += (ggml_float)(vX[vI00] * vX[vI00]);
          vSumXdz += (ggml_float)(vX[vI00] * vDz[vI00]);
        }

        // const float mean     = (float)(sum_xx)/ne00;
        const float cMeanEps = (float)(vSumXx) / ne00 + vEps;
        const float cSumEps = (float)(vSumXx) + vEps * ne00;
        // const float mean_xdz = (float)(sum_xdz)/ne00;
        //  we could cache rms from forward pass to improve performance.
        //  to do this implement ggml_rms and compose ggml_rms_norm using ggml_rms.
        // const float rms      = sqrtf(mean_eps);
        const float cRrms = 1.0f / sqrtf(cMeanEps);
        // const float scale    = -rrms/(ne00 * mean_eps); // -1/(n*rms**3)

        {
          // z = rms_norm(x)
          //
          // rms_norm(src1) =
          //     scale(
          //         src1,
          //         div(
          //             1,
          //             sqrt(
          //                 add(
          //                     scale(
          //                         sum(
          //                             sqr(
          //                                 src1)),
          //                         (1.0/N)),
          //                     eps))));

          // postorder:
          // ## op    args         grad
          // 00 param src1         grad[#00]
          // 01 const 1
          // 02 sqr   (#00)        grad[#02]
          // 03 sum   (#02)        grad[#03]
          // 04 const 1/N
          // 05 scale (#03, #04)   grad[#05]
          // 06 const eps
          // 07 add   (#05, #06)   grad[#07]
          // 08 sqrt  (#07)        grad[#08]
          // 09 div   (#01,#08)    grad[#09]
          // 10 scale (#00,#09)    grad[#10]
          //
          // backward pass, given grad[#10]
          // #10: scale
          // grad[#00] += scale(grad[#10],#09)
          // grad[#09] += sum(mul(grad[#10],#00))
          // #09: div
          // grad[#08] += neg(mul(grad[#09], div(#09,#08)))
          // #08: sqrt
          // grad[#07] += mul(grad[#08], div(0.5, #08))
          // #07: add
          // grad[#05] += grad[#07]
          // #05: scale
          // grad[#03] += scale(grad[#05],#04)
          // #03: sum
          // grad[#02] += repeat(grad[#03], #02)
          // #02:
          // grad[#00] += scale(mul(#00, grad[#02]), 2.0)
          //
          // substitute and simplify:
          // grad[#00] = scale(grad(#10), #09) + scale(mul(#00, grad[#02]), 2.0)
          // grad[#02] = repeat(grad[#03], #02)
          // grad[#02] = repeat(scale(grad[#05],#04), #02)
          // grad[#02] = repeat(scale(grad[#07],#04), #02)
          // grad[#02] = repeat(scale(mul(grad[#08], div(0.5, #08)),#04), #02)
          // grad[#02] = repeat(scale(mul(neg(mul(grad[#09], div(#09,#08))), div(0.5, #08)),#04), #02)
          // grad[#02] = repeat(scale(mul(neg(mul(sum(mul(grad[#10],#00)), div(#09,#08))), div(0.5, #08)),#04), #02)
          // grad[#02] = repeat(-(sum(mul(grad[#10],#00)) * div(#09,#08) * div(0.5, #08) * (1/N)), #02)
          // grad[#02] = repeat(-(sum(mul(grad[#10],#00)) * div(div(#01,#08),#08) * div(0.5, #08) * (1/N)), #02)
          // grad[#02] = repeat(-(sum(mul(grad[#10],#00)) * div(1,#08*#08) * div(0.5, #08) * (1/N)), #02)
          // grad[#02] = repeat(-(sum(mul(grad[#10],#00)) * div(1,#07) * div(0.5, #08) * (1/N)), #02)
          // grad[#00] = scale(grad(#10), #09) + scale(mul(#00, grad[#02]), 2.0)
          // grad[#00] = scale(grad(#10), #09) + scale(mul(#00, repeat(-(sum(mul(grad[#10],#00)) * div(1,#07) * div(0.5,
          // #08) * (1/N)), #02)), 2.0) grad[#00] = scale(grad(#10), #09) + scale(scale(#00, -(sum(mul(grad[#10],#00)) *
          // div(1,#07) * div(0.5, #08) * (1/N))), 2.0) grad[#00] = scale(grad(#10), #09) + scale(#00,
          // -(sum(mul(grad[#10],#00)) * div(1,#07) * div(1,#08) * (1/N))) grad[#00] = scale(grad(#10), #09) +
          // scale(#00, sum(mul(grad[#10],#00)) * div(1,#07*#08) * (-1/N)) grad[#00] = scale(grad(#10), #09) +
          // scale(#00, sum(mul(grad[#10],#00)) * div(1,#07*#08) * (-1/N)) grad[#00] = scale(grad(#10), #09) +
          // scale(#00, sum(mul(grad[#10],#00)) * div(1,mean_eps*rms) * (-1/N)) grad[#00] = scale(grad(#10), #09) +
          // scale(#00, sum(mul(grad[#10],#00)) * div(-1,rms*N*mean_eps)) grad[#00] = scale(grad(#10), #09) + scale(#00,
          // sum(mul(grad[#10],#00)) * div(-1,rms*N*(sum_xx/N+eps))) grad[#00] = scale(grad(#10), #09) + scale(#00,
          // sum(mul(grad[#10],#00)) * div(-1,rms*N*sum_xx+rms*N*eps)) grad[#00] = scale(dz, rrms) + scale(x,
          // sum(mul(dz,x)) * div(-1,rms*N*mean_eps)) grad[#00] = scale(dz, rrms) + scale(x, sum_xdz *
          // div(-1,rms*N*mean_eps)) a = b*c + d*e a = b*c*f/f + d*e*f/f a = (b*c*f + d*e*f)*(1/f) a = (b*c*(1/c) +
          // d*e*(1/c))*(1/(1/c)) a = (b + d*e/c)*c b = dz, c = rrms, d = x, e = sum_xdz * div(-1,rms*N*mean_eps) a =
          // (dz + x*sum_xdz * div(-1,rms*N*mean_eps)/rrms)*rrms a = (dz + x*sum_xdz * div(-1,rms*N*mean_eps)*rms)*rrms
          // a = (dz + x*sum_xdz * div(-rms,rms*N*mean_eps))*rrms
          // a = (dz + x*sum_xdz * div(-1,N*mean_eps))*rrms
          // a = (dz + x*div(-sum_xdz,N*mean_eps))*rrms
          // a = (dz + x*div(-mean_xdz,mean_eps))*rrms
          // grad[#00] = scale(dz + scale(x, div(-mean_xdz,mean_eps)),rrms)
          // grad[#00] = scale(dz + scale(x, -mean_xdz/mean_eps),rrms)
          // dx = scale(dz + scale(x, -mean_xdz/mean_eps),rrms)
        }
        // dx = scale(dz + scale(x, -mean_xdz/mean_eps),rrms)
        // post-order:
        // dx := x
        // dx := scale(dx,-mean_xdz/mean_eps)
        // dx := add(dx, dz)
        // dx := scale(dx, rrms)
        float *vDx = (float *)((char *)dst->data + vI01 * nb1 + vI02 * nb2 + vI03 * nb3);

        // dx[i00] = (dz + x*(-sum_xdz/sum_eps)) * rrms
        // note: https://github.com/ggml-org/ggml/issues/1491
        const float cScaleX = (float)(-vSumXdz) / cSumEps;
        for (int64_t vI00 = 0; vI00 < ne00; vI00++) {
          vDx[vI00] = (vDz[vI00] + vX[vI00] * cScaleX) * cRrms;
        }
      }
    }
  }
}

void ggml_compute_forward_rms_norm_back(const ggml_compute_params *pParams, ggml_tensor *pDst) {

  const ggml_tensor *vSrc0 = pDst->src[0];

  switch (vSrc0->type) {
  case GGML_TYPE_F32: {
    fGgmlComputeForwardRmsNormBackF32(pParams, pDst);
  } break;
  default: {
    GGML_ABORT("fatal error");
  }
  }
}

// ggml_compute_forward_group_norm

static void fGgmlComputeForwardGroupNormF32(const ggml_compute_params *pParams, ggml_tensor *dst) {

  const ggml_tensor *src0 = dst->src[0];

  GGML_ASSERT(ggml_are_same_shape(src0, dst));

  GGML_ASSERT(src0->nb[0] == sizeof(float));

  const int cIth = pParams->ith;
  const int cNth = pParams->nth;

  GGML_TENSOR_UNARY_OP_LOCALS

  // TODO: optimize

  float vEps;
  memcpy(&vEps, dst->op_params + 1, sizeof(float));

  int vNChannels = src0->ne[2];
  int vNGroups = dst->op_params[0];
  int vNChannelsPerGroup = (vNChannels + vNGroups - 1) / vNGroups;
  for (int vI = cIth; vI < vNGroups; vI += cNth) {
    int vStart = vI * vNChannelsPerGroup;
    int vEnd = vStart + vNChannelsPerGroup;
    if (vEnd > vNChannels) {
      vEnd = vNChannels;
    }
    int vStep = vEnd - vStart;

    for (int64_t vI03 = 0; vI03 < ne03; vI03++) {
      ggml_float vSum = 0.0;
      for (int64_t vI02 = vStart; vI02 < vEnd; vI02++) {
        for (int64_t vI01 = 0; vI01 < ne01; vI01++) {
          const float *vX = (float *)((char *)src0->data + vI01 * nb01 + vI02 * nb02 + vI03 * nb03);

          ggml_float vSumr = 0.0;
          for (int64_t vI00 = 0; vI00 < ne00; vI00++) {
            vSumr += (ggml_float)vX[vI00];
          }
          vSum += vSumr;
        }
      }
      const float cMean = vSum / (ne00 * ne01 * vStep);

      ggml_float vSum2 = 0.0;
      for (int64_t vI02 = vStart; vI02 < vEnd; vI02++) {
        for (int64_t vI01 = 0; vI01 < ne01; vI01++) {
          const float *vX = (float *)((char *)src0->data + vI01 * nb01 + vI02 * nb02 + vI03 * nb03);

          float *vY = (float *)((char *)dst->data + vI01 * nb1 + vI02 * nb2 + vI03 * nb3);

          ggml_float vSumr = 0.0;
          for (int64_t vI00 = 0; vI00 < ne00; vI00++) {
            float v_ = vX[vI00] - cMean;
            vY[vI00] = v_;
            vSumr += (ggml_float)(v_ * v_);
          }
          vSum2 += vSumr;
        }
      }
      const float cVariance = vSum2 / (ne00 * ne01 * vStep);
      const float cScale = 1.0f / sqrtf(cVariance + vEps);

      for (int64_t vI02 = vStart; vI02 < vEnd; vI02++) {
        for (int64_t vI01 = 0; vI01 < ne01; vI01++) {
          float *vY = (float *)((char *)dst->data + vI01 * nb1 + vI02 * nb2 + vI03 * nb3);
          ggml_vec_scale_f32(ne00, vY, cScale);
        }
      }
    }
  }
}

void ggml_compute_forward_group_norm(const ggml_compute_params *pParams, ggml_tensor *pDst) {

  const ggml_tensor *vSrc0 = pDst->src[0];

  switch (vSrc0->type) {
  case GGML_TYPE_F32: {
    fGgmlComputeForwardGroupNormF32(pParams, pDst);
  } break;
  default: {
    GGML_ABORT("fatal error");
  }
  }
}

// ggml_compute_forward_l2_norm

static void fGgmlComputeForwardL2NormF32(const ggml_compute_params *pParams, ggml_tensor *dst) {

  const ggml_tensor *src0 = dst->src[0];

  GGML_ASSERT(ggml_are_same_shape(src0, dst));

  const int cIth = pParams->ith;
  const int cNth = pParams->nth;

  GGML_TENSOR_UNARY_OP_LOCALS

  float vEps;
  memcpy(&vEps, dst->op_params, sizeof(float));

  GGML_ASSERT(vEps >= 0.0f);

  // TODO: optimize
  for (int64_t vI03 = 0; vI03 < ne03; vI03++) {
    for (int64_t vI02 = 0; vI02 < ne02; vI02++) {
      for (int64_t vI01 = cIth; vI01 < ne01; vI01 += cNth) {
        const char *vX = (const char *)src0->data + vI01 * nb01 + vI02 * nb02 + vI03 * nb03;

        ggml_float vSum = 0.0;
        for (int64_t vI00 = 0; vI00 < ne00; vI00++) {
          const float cXi = *(const float *)(vX + vI00 * nb00);
          vSum += (ggml_float)(cXi * cXi);
        }

        const float cScale = 1.0f / fmaxf(sqrtf(vSum), vEps);

        char *vY = (char *)dst->data + vI01 * nb1 + vI02 * nb2 + vI03 * nb3;

        if (nb00 == sizeof(float) && nb0 == sizeof(float)) {
          memcpy(vY, vX, ne00 * sizeof(float));
          ggml_vec_scale_f32(ne00, (float *)vY, cScale);
        } else {
          for (int64_t vI00 = 0; vI00 < ne00; vI00++) {
            const float cXi = *(const float *)(vX + vI00 * nb00);
            *(float *)(vY + vI00 * nb0) = cXi * cScale;
          }
        }
      }
    }
  }
}

void ggml_compute_forward_l2_norm(const ggml_compute_params *pParams, ggml_tensor *pDst) {

  const ggml_tensor *vSrc0 = pDst->src[0];

  switch (vSrc0->type) {
  case GGML_TYPE_F32: {
    fGgmlComputeForwardL2NormF32(pParams, pDst);
  } break;
  default: {
    GGML_ABORT("fatal error");
  }
  }
}

// ggml_compute_forward_out_prod

static void fGgmlComputeForwardOutProdF32(const ggml_compute_params *pParams, ggml_tensor *dst) {

  const ggml_tensor *src0 = dst->src[0];
  const ggml_tensor *src1 = dst->src[1];

  GGML_TENSOR_BINARY_OP_LOCALS

  GGML_ASSERT(dst->type == GGML_TYPE_F32);
  GGML_ASSERT(src0->type == GGML_TYPE_F32);
  GGML_ASSERT(src1->type == GGML_TYPE_F32);

  const int cIth = pParams->ith;
  const int cNth = pParams->nth;

  GGML_ASSERT(ne0 == ne00);
  GGML_ASSERT(ne1 == ne10);
  GGML_ASSERT(ne2 == ne12);
  GGML_ASSERT(ne3 == ne13);

  GGML_ASSERT(ne2 % ne02 == 0);
  GGML_ASSERT(ne3 % ne03 == 0);

  // we don't support permuted src0 or src1
  GGML_ASSERT(nb00 == sizeof(float));

  // dst cannot be transposed or permuted
  GGML_ASSERT(nb0 == sizeof(float));
  // GGML_ASSERT(nb0 <= nb1);
  // GGML_ASSERT(nb1 <= nb2);
  // GGML_ASSERT(nb2 <= nb3);

  // nb01 >= nb00 - src0 is not transposed
  //   compute by src0 rows

  if (cIth == 0) {
    ggml_vec_set_f32(ne0 * ne1 * ne2 * ne3, (float *)dst->data, 0);
  }
  ggml_barrier(pParams->threadpool);

  // dst[:,:,:,:] = 0
  // for i2,i3:
  //   for i1:
  //     for i01:
  //       for i0:
  //         dst[i0,i1,i2,i3] += src0[i0,i01,i2,i3] * src1[i1,i01,i2,i3]

  // parallelize by last three dimensions

  // total rows in dst
  const int64_t cNr = ne1 * ne2 * ne3;

  // rows per thread
  const int64_t cDr = (cNr + cNth - 1) / cNth;

  // row range for this thread
  const int64_t cIr0 = cDr * cIth;
  const int64_t cIr1 = MIN(cIr0 + cDr, cNr);

  // block-tiling attempt
  const int64_t cBlck0 = MAX(GGML_VEC_MAD_UNROLL, 32);
  const int64_t cBlck1 = 16;

  // dps == dst per src0, used for group query attention
  const int64_t cDps2 = ne2 / ne02;
  const int64_t cDps3 = ne3 / ne03;

  for (int64_t vBir = cIr0; vBir < cIr1; vBir += cBlck1) {
    const int64_t cBir1 = MIN(vBir + cBlck1, cIr1);
    for (int64_t vBi01 = 0; vBi01 < ne01; vBi01 += cBlck0) {
      const int64_t cBne01 = MIN(vBi01 + cBlck0, ne01);
      for (int64_t vIr = vBir; vIr < cBir1; ++vIr) {
        // dst indices
        const int64_t cI3 = vIr / (ne2 * ne1);
        const int64_t cI2 = (vIr - cI3 * ne2 * ne1) / ne1;
        const int64_t cI1 = (vIr - cI3 * ne2 * ne1 - cI2 * ne1);

        const int64_t cI02 = cI2 / cDps2;
        const int64_t cI03 = cI3 / cDps3;

        // const int64_t i10 = i1;
        const int64_t cI12 = cI2;
        const int64_t cI13 = cI3;

#if GGML_VEC_MAD_UNROLL > 2
        const int64_t cBne01Unroll = cBne01 - (cBne01 % GGML_VEC_MAD_UNROLL);
        for (int64_t vI01 = vBi01; vI01 < cBne01Unroll; vI01 += GGML_VEC_MAD_UNROLL) {
          const int64_t cI11 = vI01;

          float *vS0 = (float *)((char *)src0->data + (vI01 * nb01 + cI02 * nb02 + cI03 * nb03));
          float *vS1 = (float *)((char *)src1->data + (cI1 * nb10 + cI11 * nb11 + cI12 * nb12 + cI13 * nb13));
          float *vD = (float *)((char *)dst->data + (cI1 * nb1 + cI2 * nb2 + cI3 * nb3));

          ggml_vec_mad_f32_unroll(ne0, nb01, nb11, vD, vS0, vS1);
        }
        for (int64_t vI01 = cBne01Unroll; vI01 < cBne01; ++vI01) {
          const int64_t cI11 = vI01;

          float *vS0 = (float *)((char *)src0->data + (vI01 * nb01 + cI02 * nb02 + cI03 * nb03));
          float *vS1 = (float *)((char *)src1->data + (cI1 * nb10 + cI11 * nb11 + cI12 * nb12 + cI13 * nb13));
          float *vD = (float *)((char *)dst->data + (cI1 * nb1 + cI2 * nb2 + cI3 * nb3));

          ggml_vec_mad_f32(ne0, vD, vS0, *vS1);
        }
#else
        for (int64_t vI01 = vBi01; vI01 < cBne01; ++vI01) {
          const int64_t cI11 = vI01;

          float *vS0 = (float *)((char *)src0->data + (vI01 * nb01 + cI02 * nb02 + cI03 * nb03));
          float *vS1 = (float *)((char *)src1->data + (cI1 * nb10 + cI11 * nb11 + cI12 * nb12 + cI13 * nb13));
          float *vD = (float *)((char *)dst->data + (cI1 * nb1 + cI2 * nb2 + cI3 * nb3));

          ggml_vec_mad_f32(ne0, vD, vS0, *vS1);
        }
#endif
      }
    }
  }
}

static void fGgmlComputeForwardOutProdQF32(const ggml_compute_params *pParams, ggml_tensor *dst) {

  const ggml_tensor *src0 = dst->src[0];
  const ggml_tensor *src1 = dst->src[1];

  GGML_TENSOR_BINARY_OP_LOCALS;

  const int cIth = pParams->ith;
  const int cNth = pParams->nth;

  const ggml_type cType = src0->type;
  ggml_to_float_t const cDequantizeRowQ = ggml_get_type_traits(cType)->to_float;

  GGML_ASSERT(ne02 == ne12);
  GGML_ASSERT(ne03 == ne13);
  GGML_ASSERT(ne2 == ne12);
  GGML_ASSERT(ne3 == ne13);

  // we don't support permuted src0 dim0
  GGML_ASSERT(nb00 == ggml_type_size(cType));

  // dst dim0 cannot be transposed or permuted
  GGML_ASSERT(nb0 == sizeof(float));
  // GGML_ASSERT(nb0 <= nb1);
  // GGML_ASSERT(nb1 <= nb2);
  // GGML_ASSERT(nb2 <= nb3);

  GGML_ASSERT(ne0 == ne00);
  GGML_ASSERT(ne1 == ne10);
  GGML_ASSERT(ne2 == ne02);
  GGML_ASSERT(ne3 == ne03);

  // nb01 >= nb00 - src0 is not transposed
  //   compute by src0 rows

  if (cIth == 0) {
    ggml_vec_set_f32(ne0 * ne1 * ne2 * ne3, (float *)dst->data, 0);
  }
  ggml_barrier(pParams->threadpool);

  // parallelize by last three dimensions

  // total rows in dst
  const int64_t cNr = ne1 * ne2 * ne3;

  // rows per thread
  const int64_t cDr = (cNr + cNth - 1) / cNth;

  // row range for this thread
  const int64_t cIr0 = cDr * cIth;
  const int64_t cIr1 = MIN(cIr0 + cDr, cNr);

  // dst[:,:,:,:] = 0
  // for i2,i3:
  //   for i1:
  //     for i01:
  //       for i0:
  //         dst[i0,i1,i2,i3] += src0[i0,i01,i2,i3] * src1[i1,i01,i2,i3]

  float *vWdata = (float *)pParams->wdata + (ne0 + CACHE_LINE_SIZE_F32) * cIth;

  for (int64_t vIr = cIr0; vIr < cIr1; ++vIr) {
    // dst indices
    const int64_t cI3 = vIr / (ne2 * ne1);
    const int64_t cI2 = (vIr - cI3 * ne2 * ne1) / ne1;
    const int64_t cI1 = (vIr - cI3 * ne2 * ne1 - cI2 * ne1);

    const int64_t cI02 = cI2;
    const int64_t cI03 = cI3;

    // const int64_t i10 = i1;
    const int64_t cI12 = cI2;
    const int64_t cI13 = cI3;

    for (int64_t vI01 = 0; vI01 < ne01; ++vI01) {
      const int64_t cI11 = vI01;

      float *vS0 = (float *)((char *)src0->data + (vI01 * nb01 + cI02 * nb02 + cI03 * nb03));
      float *vS1 = (float *)((char *)src1->data + (cI1 * nb10 + cI11 * nb11 + cI12 * nb12 + cI13 * nb13));
      float *vD = (float *)((char *)dst->data + (cI1 * nb1 + cI2 * nb2 + cI3 * nb3));

      cDequantizeRowQ(vS0, vWdata, ne0);
      ggml_vec_mad_f32(ne0, vD, vWdata, *vS1);
    }
  }
}

static void fGgmlComputeForwardOutProdF16F32(const ggml_compute_params *pParams, ggml_tensor *dst) {

  const ggml_tensor *src0 = dst->src[0];
  const ggml_tensor *src1 = dst->src[1];

  GGML_TENSOR_BINARY_OP_LOCALS;

  const int cIth = pParams->ith;
  const int cNth = pParams->nth;

  GGML_ASSERT(src0->type == GGML_TYPE_F16);
  GGML_ASSERT(src1->type == GGML_TYPE_F32);
  GGML_ASSERT(dst->type == GGML_TYPE_F32);

  GGML_ASSERT(ne02 == ne12);
  GGML_ASSERT(ne03 == ne13);
  GGML_ASSERT(ne2 == ne12);
  GGML_ASSERT(ne3 == ne13);

  GGML_ASSERT(nb00 == sizeof(ggml_fp16_t));
  GGML_ASSERT(nb0 == sizeof(float));

  GGML_ASSERT(ne0 == ne00);
  GGML_ASSERT(ne1 == ne10);
  GGML_ASSERT(ne2 == ne02);
  GGML_ASSERT(ne3 == ne03);

  if (cIth == 0) {
    ggml_vec_set_f32(ne0 * ne1 * ne2 * ne3, (float *)dst->data, 0);
  }
  ggml_barrier(pParams->threadpool);

  const int64_t cNr = ne1 * ne2 * ne3;
  const int64_t cDr = (cNr + cNth - 1) / cNth;
  const int64_t cIr0 = cDr * cIth;
  const int64_t cIr1 = MIN(cIr0 + cDr, cNr);

  float *vWdata = (float *)pParams->wdata + (ne0 + CACHE_LINE_SIZE_F32) * cIth;

  for (int64_t vIr = cIr0; vIr < cIr1; ++vIr) {
    const int64_t cI3 = vIr / (ne2 * ne1);
    const int64_t cI2 = (vIr - cI3 * ne2 * ne1) / ne1;
    const int64_t cI1 = (vIr - cI3 * ne2 * ne1 - cI2 * ne1);

    const int64_t cI02 = cI2;
    const int64_t cI03 = cI3;

    const int64_t cI12 = cI2;
    const int64_t cI13 = cI3;

    float *vD = (float *)((char *)dst->data + (cI1 * nb1 + cI2 * nb2 + cI3 * nb3));

    for (int64_t vI01 = 0; vI01 < ne01; ++vI01) {
      const int64_t cI11 = vI01;
      ggml_fp16_t *vS0 = (ggml_fp16_t *)((char *)src0->data + (vI01 * nb01 + cI02 * nb02 + cI03 * nb03));
      float *vS1 = (float *)((char *)src1->data + (cI1 * nb10 + cI11 * nb11 + cI12 * nb12 + cI13 * nb13));
      ggml_fp16_to_fp32_row(vS0, vWdata, ne0);
      ggml_vec_mad_f32(ne0, vD, vWdata, *vS1);
    }
  }
}

void ggml_compute_forward_out_prod(const ggml_compute_params *pParams, ggml_tensor *pDst) {

  const ggml_tensor *vSrc0 = pDst->src[0];

  switch (vSrc0->type) {
  case GGML_TYPE_Q1_0:
  case GGML_TYPE_Q2_0:
  case GGML_TYPE_Q4_0:
  case GGML_TYPE_Q4_1:
  case GGML_TYPE_Q5_0:
  case GGML_TYPE_Q5_1:
  case GGML_TYPE_Q8_0:
  case GGML_TYPE_MXFP4:
  case GGML_TYPE_NVFP4:
  case GGML_TYPE_Q2_K:
  case GGML_TYPE_Q3_K:
  case GGML_TYPE_Q4_K:
  case GGML_TYPE_Q5_K:
  case GGML_TYPE_Q6_K:
  case GGML_TYPE_TQ1_0:
  case GGML_TYPE_TQ2_0:
  case GGML_TYPE_IQ2_XXS:
  case GGML_TYPE_IQ2_XS:
  case GGML_TYPE_IQ3_XXS:
  case GGML_TYPE_IQ1_S:
  case GGML_TYPE_IQ1_M:
  case GGML_TYPE_IQ4_NL:
  case GGML_TYPE_IQ4_XS:
  case GGML_TYPE_IQ3_S:
  case GGML_TYPE_IQ2_S: {
    fGgmlComputeForwardOutProdQF32(pParams, pDst);
  } break;
  case GGML_TYPE_F16: {
    fGgmlComputeForwardOutProdF16F32(pParams, pDst);
  } break;
  case GGML_TYPE_F32: {
    fGgmlComputeForwardOutProdF32(pParams, pDst);
  } break;
  default: {
    GGML_ABORT("fatal error");
  }
  }
}

// ggml_compute_forward_scale

static void fGgmlComputeForwardScaleF32(const ggml_compute_params *pParams, ggml_tensor *pDst) {

  const ggml_tensor *vSrc0 = pDst->src[0];

  GGML_ASSERT(ggml_is_contiguous(vSrc0));
  GGML_ASSERT(ggml_is_contiguous(pDst));
  GGML_ASSERT(ggml_are_same_shape(vSrc0, pDst));

  float vS; // scale factor
  float vB; // bias

  memcpy(&vS, (float *)pDst->op_params + 0, sizeof(float));
  memcpy(&vB, (float *)pDst->op_params + 1, sizeof(float));

  const int cIth = pParams->ith;
  const int cNth = pParams->nth;

  const int cNc = vSrc0->ne[0];
  const int cNr = ggml_nrows(vSrc0);

  // rows per thread
  const int cDr = (cNr + cNth - 1) / cNth;

  // row range for this thread
  const int cIr0 = cDr * cIth;
  const int cIr1 = MIN(cIr0 + cDr, cNr);

  const size_t cNb01 = vSrc0->nb[1];

  const size_t cNb1 = pDst->nb[1];

  if (vB == 0.0f) {
    for (int vI1 = cIr0; vI1 < cIr1; vI1++) {
      if (pDst->data != vSrc0->data) {
        // src0 is same shape as dst => same indices
        // TODO: add x parameter to ggml_vec_scale_f32 and remove this memcpy
        memcpy((char *)pDst->data + vI1 * cNb1, (char *)vSrc0->data + vI1 * cNb01, cNc * sizeof(float));
      }
      ggml_vec_scale_f32(cNc, (float *)((char *)pDst->data + vI1 * cNb1), vS);
    }
  } else {
    for (int vI1 = cIr0; vI1 < cIr1; vI1++) {
      ggml_vec_mad1_f32(cNc, (float *)((char *)pDst->data + vI1 * cNb1), (float *)((char *)vSrc0->data + vI1 * cNb1),
                        vS, vB);
    }
  }
}

void ggml_compute_forward_scale(const ggml_compute_params *pParams, ggml_tensor *pDst) {

  const ggml_tensor *vSrc0 = pDst->src[0];

  switch (vSrc0->type) {
  case GGML_TYPE_F32: {
    fGgmlComputeForwardScaleF32(pParams, pDst);
  } break;
  default: {
    GGML_ABORT("fatal error");
  }
  }
}

// ggml_compute_forward_set

static void fGgmlComputeForwardSetF32(const ggml_compute_params *pParams, ggml_tensor *pDst) {

  const ggml_tensor *vSrc0 = pDst->src[0];
  const ggml_tensor *vSrc1 = pDst->src[1];

  GGML_ASSERT(ggml_are_same_shape(vSrc0, pDst));
  GGML_ASSERT(ggml_is_contiguous(pDst) && ggml_is_contiguous(vSrc0));

  // view src0 and dst with these strides and data offset inbytes during set
  // nb0 is implicitly element_size because src0 and dst are contiguous
  size_t nb1 = ((int32_t *)pDst->op_params)[0];
  size_t vNb2 = ((int32_t *)pDst->op_params)[1];
  size_t vNb3 = ((int32_t *)pDst->op_params)[2];
  size_t vOffset = ((int32_t *)pDst->op_params)[3];
  bool vInplace = (bool)((int32_t *)pDst->op_params)[4];

  if (!vInplace) {
    if (pParams->ith == 0) {
      // memcpy needs to be synchronized across threads to avoid race conditions.
      // => do it in INIT phase
      memcpy(((char *)pDst->data), ((char *)vSrc0->data), ggml_nbytes(pDst));
    }
    ggml_barrier(pParams->threadpool);
  }

  const int cIth = pParams->ith;
  const int cNth = pParams->nth;

  const int cNr = ggml_nrows(vSrc1);
  const int cNc = vSrc1->ne[0];

  GGML_TENSOR_LOCALS(int64_t, ne1, vSrc1, ne)
  GGML_TENSOR_LOCALS(size_t, nb1, vSrc1, nb)

  // src0 and dst as viewed during set
  const size_t cNb0 = ggml_element_size(vSrc0);

  const int cIm0 = (ne10 == 0 ? 0 : ne10 - 1);
  const int cIm1 = (ne11 == 0 ? 0 : ne11 - 1);
  const int cIm2 = (ne12 == 0 ? 0 : ne12 - 1);
  const int cIm3 = (ne13 == 0 ? 0 : ne13 - 1);

  GGML_ASSERT(vOffset + cIm0 * cNb0 + cIm1 * nb1 + cIm2 * vNb2 + cIm3 * vNb3 <= ggml_nbytes(pDst));

  GGML_ASSERT(nb10 == sizeof(float));

  // rows per thread
  const int cDr = (cNr + cNth - 1) / cNth;

  // row range for this thread
  const int cIr0 = cDr * cIth;
  const int cIr1 = MIN(cIr0 + cDr, cNr);

  for (int vIr = cIr0; vIr < cIr1; ++vIr) {
    // src0 and dst are viewed with shape of src1 and offset
    // => same indices
    const int cI3 = vIr / (ne12 * ne11);
    const int cI2 = (vIr - cI3 * ne12 * ne11) / ne11;
    const int cI1 = (vIr - cI3 * ne12 * ne11 - cI2 * ne11);

    ggml_vec_cpy_f32(cNc, (float *)((char *)pDst->data + cI3 * vNb3 + cI2 * vNb2 + cI1 * nb1 + vOffset),
                     (float *)((char *)vSrc1->data + cI3 * nb13 + cI2 * nb12 + cI1 * nb11));
  }
}

static void fGgmlComputeForwardSetI32(const ggml_compute_params *pParams, ggml_tensor *pDst) {

  const ggml_tensor *vSrc0 = pDst->src[0];
  const ggml_tensor *vSrc1 = pDst->src[1];

  GGML_ASSERT(ggml_are_same_shape(vSrc0, pDst));
  GGML_ASSERT(ggml_is_contiguous(pDst) && ggml_is_contiguous(vSrc0));

  // view src0 and dst with these strides and data offset inbytes during set
  // nb0 is implicitly element_size because src0 and dst are contiguous
  size_t nb1 = ((int32_t *)pDst->op_params)[0];
  size_t vNb2 = ((int32_t *)pDst->op_params)[1];
  size_t vNb3 = ((int32_t *)pDst->op_params)[2];
  size_t vOffset = ((int32_t *)pDst->op_params)[3];
  bool vInplace = (bool)((int32_t *)pDst->op_params)[4];

  if (!vInplace) {
    if (pParams->ith == 0) {
      // memcpy needs to be synchronized across threads to avoid race conditions.
      // => do it in INIT phase
      memcpy(((char *)pDst->data), ((char *)vSrc0->data), ggml_nbytes(pDst));
    }
    ggml_barrier(pParams->threadpool);
  }

  const int cIth = pParams->ith;
  const int cNth = pParams->nth;

  const int cNr = ggml_nrows(vSrc1);
  const int cNc = vSrc1->ne[0];

  GGML_TENSOR_LOCALS(int64_t, ne1, vSrc1, ne)
  GGML_TENSOR_LOCALS(size_t, nb1, vSrc1, nb)

  // src0 and dst as viewed during set
  const size_t cNb0 = ggml_element_size(vSrc0);

  const int cIm0 = (ne10 == 0 ? 0 : ne10 - 1);
  const int cIm1 = (ne11 == 0 ? 0 : ne11 - 1);
  const int cIm2 = (ne12 == 0 ? 0 : ne12 - 1);
  const int cIm3 = (ne13 == 0 ? 0 : ne13 - 1);

  GGML_ASSERT(vOffset + cIm0 * cNb0 + cIm1 * nb1 + cIm2 * vNb2 + cIm3 * vNb3 <= ggml_nbytes(pDst));

  GGML_ASSERT(nb10 == sizeof(int32_t));

  // rows per thread
  const int cDr = (cNr + cNth - 1) / cNth;

  // row range for this thread
  const int cIr0 = cDr * cIth;
  const int cIr1 = MIN(cIr0 + cDr, cNr);

  for (int vIr = cIr0; vIr < cIr1; ++vIr) {
    // src0 and dst are viewed with shape of src1 and offset
    // => same indices
    const int cI3 = vIr / (ne12 * ne11);
    const int cI2 = (vIr - cI3 * ne12 * ne11) / ne11;
    const int cI1 = (vIr - cI3 * ne12 * ne11 - cI2 * ne11);

    ggml_vec_cpy_i32(cNc, (int32_t *)((char *)pDst->data + cI3 * vNb3 + cI2 * vNb2 + cI1 * nb1 + vOffset),
                     (int32_t *)((char *)vSrc1->data + cI3 * nb13 + cI2 * nb12 + cI1 * nb11));
  }
}

void ggml_compute_forward_set(const ggml_compute_params *pParams, ggml_tensor *pDst) {

  const ggml_tensor *vSrc0 = pDst->src[0];

  switch (vSrc0->type) {
  case GGML_TYPE_F32: {
    fGgmlComputeForwardSetF32(pParams, pDst);
  } break;
  case GGML_TYPE_I32: {
    fGgmlComputeForwardSetI32(pParams, pDst);
  } break;
  case GGML_TYPE_F16:
  case GGML_TYPE_BF16:
  case GGML_TYPE_Q1_0:
  case GGML_TYPE_Q2_0:
  case GGML_TYPE_Q4_0:
  case GGML_TYPE_Q4_1:
  case GGML_TYPE_Q5_0:
  case GGML_TYPE_Q5_1:
  case GGML_TYPE_Q8_0:
  case GGML_TYPE_Q8_1:
  case GGML_TYPE_MXFP4:
  case GGML_TYPE_NVFP4:
  case GGML_TYPE_Q2_K:
  case GGML_TYPE_Q3_K:
  case GGML_TYPE_Q4_K:
  case GGML_TYPE_Q5_K:
  case GGML_TYPE_Q6_K:
  case GGML_TYPE_TQ1_0:
  case GGML_TYPE_TQ2_0:
  case GGML_TYPE_IQ2_XXS:
  case GGML_TYPE_IQ2_XS:
  case GGML_TYPE_IQ3_XXS:
  case GGML_TYPE_IQ1_S:
  case GGML_TYPE_IQ1_M:
  case GGML_TYPE_IQ4_NL:
  case GGML_TYPE_IQ4_XS:
  case GGML_TYPE_IQ3_S:
  case GGML_TYPE_IQ2_S:
  default: {
    GGML_ABORT("fatal error");
  }
  }
}

// ggml_compute_forward_cpy

void ggml_compute_forward_cpy(const ggml_compute_params *pParams, ggml_tensor *pDst) {
  ggml_compute_forward_dup(pParams, pDst);
}

// ggml_compute_forward_cont

void ggml_compute_forward_cont(const ggml_compute_params *pParams, ggml_tensor *pDst) {
  ggml_compute_forward_dup(pParams, pDst);
}

// ggml_compute_forward_get_rows

static void fGgmlComputeForwardGetRowsQ(const ggml_compute_params *pParams, ggml_tensor *dst) {

  const ggml_tensor *src0 = dst->src[0];
  const ggml_tensor *src1 = dst->src[1];

  GGML_TENSOR_BINARY_OP_LOCALS

  const int64_t cNc = ne00;
  const int64_t cNr = ggml_nelements(src1);

  const ggml_type cType = src0->type;
  ggml_to_float_t const cDequantizeRowQ = ggml_get_type_traits(cType)->to_float;

  assert(ne0 == cNc);
  assert(ne02 == ne11);
  assert(nb00 == ggml_type_size(cType));
  assert(ggml_nrows(dst) == cNr);

  const int cIth = pParams->ith;
  const int cNth = pParams->nth;

  // rows per thread
  const int cDr = (cNr + cNth - 1) / cNth;

  // row range for this thread
  const int cIr0 = cDr * cIth;
  const int cIr1 = MIN(cIr0 + cDr, cNr);

  for (int64_t vI = cIr0; vI < cIr1; ++vI) {
    const int64_t cI12 = vI / (ne11 * ne10);
    const int64_t cI11 = (vI - cI12 * ne11 * ne10) / ne10;
    const int64_t cI10 = (vI - cI12 * ne11 * ne10 - cI11 * ne10);
    const int64_t cI01 = *(int32_t *)((char *)src1->data + cI10 * nb10 + cI11 * nb11 + cI12 * nb12);

    GGML_ASSERT(cI01 >= 0 && cI01 < ne01);

    cDequantizeRowQ((const void *)((char *)src0->data + cI01 * nb01 + cI11 * nb02 + cI12 * nb03),
                    (float *)((char *)dst->data + cI10 * nb1 + cI11 * nb2 + cI12 * nb3), cNc);
  }
}

static void fGgmlComputeForwardGetRowsF16(const ggml_compute_params *pParams, ggml_tensor *dst) {

  const ggml_tensor *src0 = dst->src[0];
  const ggml_tensor *src1 = dst->src[1];

  GGML_TENSOR_BINARY_OP_LOCALS

  const int64_t cNc = ne00;
  const int64_t cNr = ggml_nelements(src1);

  assert(ne0 == cNc);
  assert(ne02 == ne11);
  assert(nb00 == sizeof(ggml_fp16_t));
  assert(ggml_nrows(dst) == cNr);

  const int cIth = pParams->ith;
  const int cNth = pParams->nth;

  // rows per thread
  const int cDr = (cNr + cNth - 1) / cNth;

  // row range for this thread
  const int cIr0 = cDr * cIth;
  const int cIr1 = MIN(cIr0 + cDr, cNr);

  for (int64_t vI = cIr0; vI < cIr1; ++vI) {
    const int64_t cI12 = vI / (ne11 * ne10);
    const int64_t cI11 = (vI - cI12 * ne11 * ne10) / ne10;
    const int64_t cI10 = (vI - cI12 * ne11 * ne10 - cI11 * ne10);
    const int64_t cI01 = *(int32_t *)((char *)src1->data + cI10 * nb10 + cI11 * nb11 + cI12 * nb12);

    GGML_ASSERT(cI01 >= 0 && cI01 < ne01);

    ggml_cpu_fp16_to_fp32((const ggml_fp16_t *)((char *)src0->data + cI01 * nb01 + cI11 * nb02 + cI12 * nb03),
                          (float *)((char *)dst->data + cI10 * nb1 + cI11 * nb2 + cI12 * nb3), cNc);
  }
}

static void fGgmlComputeForwardGetRowsBf16(const ggml_compute_params *pParams, ggml_tensor *dst) {

  const ggml_tensor *src0 = dst->src[0];
  const ggml_tensor *src1 = dst->src[1];

  GGML_TENSOR_BINARY_OP_LOCALS

  const int64_t cNc = ne00;
  const int64_t cNr = ggml_nelements(src1);

  assert(ne0 == cNc);
  assert(ne02 == ne11);
  assert(nb00 == sizeof(ggml_bf16_t));
  assert(ggml_nrows(dst) == cNr);

  const int cIth = pParams->ith;
  const int cNth = pParams->nth;

  // rows per thread
  const int cDr = (cNr + cNth - 1) / cNth;

  // row range for this thread
  const int cIr0 = cDr * cIth;
  const int cIr1 = MIN(cIr0 + cDr, cNr);

  for (int64_t vI = cIr0; vI < cIr1; ++vI) {
    const int64_t cI12 = vI / (ne11 * ne10);
    const int64_t cI11 = (vI - cI12 * ne11 * ne10) / ne10;
    const int64_t cI10 = (vI - cI12 * ne11 * ne10 - cI11 * ne10);
    const int64_t cI01 = *(int32_t *)((char *)src1->data + cI10 * nb10 + cI11 * nb11 + cI12 * nb12);

    GGML_ASSERT(cI01 >= 0 && cI01 < ne01);

    ggml_cpu_bf16_to_fp32((const ggml_bf16_t *)((char *)src0->data + cI01 * nb01 + cI11 * nb02 + cI12 * nb03),
                          (float *)((char *)dst->data + cI10 * nb1 + cI11 * nb2 + cI12 * nb3), cNc);
  }
}

static void fGgmlComputeForwardGetRowsF32(const ggml_compute_params *pParams, ggml_tensor *dst) {

  const ggml_tensor *src0 = dst->src[0];
  const ggml_tensor *src1 = dst->src[1];

  GGML_TENSOR_BINARY_OP_LOCALS

  const int64_t cNc = ne00;
  const int64_t cNr = ggml_nelements(src1);

  assert(ne0 == cNc);
  assert(ne02 == ne11);
  assert(nb00 == sizeof(float));
  assert(ggml_nrows(dst) == cNr);

  const int cIth = pParams->ith;
  const int cNth = pParams->nth;

  // rows per thread
  const int cDr = (cNr + cNth - 1) / cNth;

  // row range for this thread
  const int cIr0 = cDr * cIth;
  const int cIr1 = MIN(cIr0 + cDr, cNr);

  for (int64_t vI = cIr0; vI < cIr1; ++vI) {
    const int64_t cI12 = vI / (ne11 * ne10);
    const int64_t cI11 = (vI - cI12 * ne11 * ne10) / ne10;
    const int64_t cI10 = (vI - cI12 * ne11 * ne10 - cI11 * ne10);
    const int64_t cI01 = *(int32_t *)((char *)src1->data + cI10 * nb10 + cI11 * nb11 + cI12 * nb12);

    GGML_ASSERT(cI01 >= 0 && cI01 < ne01);

    ggml_vec_cpy_f32(cNc, (float *)((char *)dst->data + cI10 * nb1 + cI11 * nb2 + cI12 * nb3),
                     (float *)((char *)src0->data + cI01 * nb01 + cI11 * nb02 + cI12 * nb03));
  }
}

void ggml_compute_forward_get_rows(const ggml_compute_params *pParams, ggml_tensor *pDst) {

  const ggml_tensor *vSrc0 = pDst->src[0];

  switch (vSrc0->type) {
  case GGML_TYPE_Q1_0:
  case GGML_TYPE_Q2_0:
  case GGML_TYPE_Q4_0:
  case GGML_TYPE_Q4_1:
  case GGML_TYPE_Q5_0:
  case GGML_TYPE_Q5_1:
  case GGML_TYPE_Q8_0:
  case GGML_TYPE_Q8_1:
  case GGML_TYPE_MXFP4:
  case GGML_TYPE_NVFP4:
  case GGML_TYPE_Q2_K:
  case GGML_TYPE_Q3_K:
  case GGML_TYPE_Q4_K:
  case GGML_TYPE_Q5_K:
  case GGML_TYPE_Q6_K:
  case GGML_TYPE_TQ1_0:
  case GGML_TYPE_TQ2_0:
  case GGML_TYPE_IQ2_XXS:
  case GGML_TYPE_IQ2_XS:
  case GGML_TYPE_IQ3_XXS:
  case GGML_TYPE_IQ1_S:
  case GGML_TYPE_IQ1_M:
  case GGML_TYPE_IQ4_NL:
  case GGML_TYPE_IQ4_XS:
  case GGML_TYPE_IQ3_S:
  case GGML_TYPE_IQ2_S: {
    fGgmlComputeForwardGetRowsQ(pParams, pDst);
  } break;
  case GGML_TYPE_F16: {
    fGgmlComputeForwardGetRowsF16(pParams, pDst);
  } break;
  case GGML_TYPE_BF16: {
    fGgmlComputeForwardGetRowsBf16(pParams, pDst);
  } break;
  case GGML_TYPE_F32:
  case GGML_TYPE_I32: {
    fGgmlComputeForwardGetRowsF32(pParams, pDst);
  } break;
  default: {
    GGML_ABORT("fatal error");
  }
  }

  // static bool first = true;
  // printf("ne0 = %d, ne1 = %d, ne2 = %d\n", dst->ne[0], dst->ne[1], dst->ne[2]);
  // if (first) {
  //     first = false;
  // } else {
  //     for (int k = 0; k < dst->ne[1]; ++k) {
  //         for (int j = 0; j < dst->ne[0]/16; ++j) {
  //             for (int i = 0; i < 16; ++i) {
  //                 printf("%8.4f ", ((float *) dst->data)[k*dst->ne[0] + j*16 + i]);
  //             }
  //             printf("\n");
  //         }
  //         printf("\n");
  //     }
  //     printf("\n");
  //     exit(0);
  // }
}

template <typename src_t, typename idx_t>
static void fGgmlComputeForwardSetRowsImpl(const ggml_compute_params *pParams, ggml_tensor *dst) {

  const ggml_tensor *src0 = dst->src[0];
  const ggml_tensor *src1 = dst->src[1];

  GGML_TENSOR_BINARY_OP_LOCALS

  const int64_t cNc = ne00;
  const int64_t cNr = ne01;

  assert(ne0 == cNc);
  assert(ne2 == ne02);
  assert(ne3 == ne03);
  GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);
  assert(ne02 % ne11 == 0);
  assert(ne03 % ne12 == 0);

  const int cIth = pParams->ith;
  const int cNth = pParams->nth;

  // rows per thread
  const int64_t cDr = (cNr + cNth - 1) / cNth;

  // row range for this thread
  const int64_t cIr0 = cDr * cIth;
  const int64_t cIr1 = std::min(cIr0 + cDr, cNr);

  const size_t cRs = ggml_row_size(src0->type, cNc);

  ggml_from_float_t const cFromFloat = ggml_get_type_traits_cpu(dst->type)->from_float;

  for (int64_t vI03 = 0; vI03 < ne03; ++vI03) {
    for (int64_t vI02 = 0; vI02 < ne02; ++vI02) {
      for (int64_t vI = cIr0; vI < cIr1; ++vI) {
        const int64_t cI12 = vI03 % ne12;
        const int64_t cI11 = vI02 % ne11;
        const int64_t cI10 = vI;

        const int64_t cI1 = *(idx_t *)((char *)src1->data + cI10 * nb10 + cI11 * nb11 + cI12 * nb12);

        GGML_ASSERT(cI1 >= 0 && cI1 < ne1);

        if constexpr (std::is_same_v<src_t, float>) {
          cFromFloat((const float *)((char *)src0->data + vI * nb01 + vI02 * nb02 + vI03 * nb03),
                     ((char *)dst->data + cI1 * nb1 + vI02 * nb2 + vI03 * nb3), cNc);
        } else if constexpr (std::is_same_v<src_t, ggml_fp16_t>) {
          if (dst->type == GGML_TYPE_F16) {
            memcpy(((char *)dst->data + cI1 * nb1 + vI02 * nb2 + vI03 * nb3),
                   ((char *)src0->data + vI * nb01 + vI02 * nb02 + vI03 * nb03), cRs);
          } else {
            float *vWdata = (float *)pParams->wdata + (cNc + CACHE_LINE_SIZE_F32) * cIth;
            ggml_fp16_to_fp32_row((const ggml_fp16_t *)((char *)src0->data + vI * nb01 + vI02 * nb02 + vI03 * nb03),
                                  vWdata, cNc);
            cFromFloat(vWdata, ((char *)dst->data + cI1 * nb1 + vI02 * nb2 + vI03 * nb3), cNc);
          }
        } else {
          GGML_ABORT("src0->type = %d (%s) not supported", src0->type, ggml_type_name(src0->type));
        }
      }
    }
  }
}

void ggml_compute_forward_set_rows(const ggml_compute_params *pParams, ggml_tensor *pDst) {

  const ggml_tensor *vSrc0 = pDst->src[0];
  const ggml_tensor *vSrc1 = pDst->src[1];

  switch (vSrc0->type) {
  case GGML_TYPE_F32: {
    if (vSrc1->type == GGML_TYPE_I64) {
      fGgmlComputeForwardSetRowsImpl<float, int64_t>(pParams, pDst);
    } else if (vSrc1->type == GGML_TYPE_I32) {
      fGgmlComputeForwardSetRowsImpl<float, int32_t>(pParams, pDst);
    } else {
      GGML_ABORT("src1->type = %d (%s) not supported", vSrc1->type, ggml_type_name(vSrc1->type));
    }
  } break;
  case GGML_TYPE_F16: {
    if (vSrc1->type == GGML_TYPE_I64) {
      fGgmlComputeForwardSetRowsImpl<ggml_fp16_t, int64_t>(pParams, pDst);
    } else if (vSrc1->type == GGML_TYPE_I32) {
      fGgmlComputeForwardSetRowsImpl<ggml_fp16_t, int32_t>(pParams, pDst);
    } else {
      GGML_ABORT("src1->type = %d (%s) not supported", vSrc1->type, ggml_type_name(vSrc1->type));
    }
  } break;
  default: {
    GGML_ABORT("src0->type = %d (%s) not supported", vSrc0->type, ggml_type_name(vSrc0->type));
  }
  }
}

// ggml_compute_forward_get_rows_back

static void fGgmlComputeForwardGetRowsBackF32F16(const ggml_compute_params *pParams, ggml_tensor *pDst) {

  const ggml_tensor *vSrc0 = pDst->src[0];
  const ggml_tensor *vSrc1 = pDst->src[1];

  if (pParams->ith != 0) {
    return;
  }

  GGML_ASSERT(ggml_is_contiguous(pDst));

  // ggml_compute_forward_dup_same_cont(params, opt0, dst);

  memset(pDst->data, 0, ggml_nbytes(pDst));

  const int cNc = vSrc0->ne[0];
  const int cNr = ggml_nelements(vSrc1);

  GGML_ASSERT(pDst->ne[0] == cNc);
  GGML_ASSERT(vSrc0->nb[0] == sizeof(ggml_fp16_t));

  for (int vI = 0; vI < cNr; ++vI) {
    const int cR = ((int32_t *)vSrc1->data)[vI];

    for (int vJ = 0; vJ < cNc; ++vJ) {
      ggml_fp16_t v_ = ((ggml_fp16_t *)((char *)vSrc0->data + vI * vSrc0->nb[1]))[vJ];
      ((float *)((char *)pDst->data + cR * pDst->nb[1]))[vJ] += GGML_CPU_FP16_TO_FP32(v_);
    }
  }
}

static void fGgmlComputeForwardGetRowsBackF32(const ggml_compute_params *pParams, ggml_tensor *pDst) {

  const ggml_tensor *vSrc0 = pDst->src[0];
  const ggml_tensor *vSrc1 = pDst->src[1];

  if (pParams->ith != 0) {
    return;
  }

  GGML_ASSERT(ggml_is_contiguous(pDst));

  // ggml_compute_forward_dup_same_cont(params, opt0, dst);

  memset(pDst->data, 0, ggml_nbytes(pDst));

  const int cNc = vSrc0->ne[0];
  const int cNr = ggml_nelements(vSrc1);

  GGML_ASSERT(pDst->ne[0] == cNc);
  GGML_ASSERT(vSrc0->nb[0] == sizeof(float));

  for (int vI = 0; vI < cNr; ++vI) {
    const int cR = ((int32_t *)vSrc1->data)[vI];

    ggml_vec_add_f32(cNc, (float *)((char *)pDst->data + cR * pDst->nb[1]),
                     (float *)((char *)pDst->data + cR * pDst->nb[1]),
                     (float *)((char *)vSrc0->data + vI * vSrc0->nb[1]));
  }
}

void ggml_compute_forward_get_rows_back(const ggml_compute_params *pParams, ggml_tensor *pDst) {

  const ggml_tensor *vSrc0 = pDst->src[0];

  switch (vSrc0->type) {
  case GGML_TYPE_F16: {
    fGgmlComputeForwardGetRowsBackF32F16(pParams, pDst);
  } break;
  case GGML_TYPE_F32: {
    fGgmlComputeForwardGetRowsBackF32(pParams, pDst);
  } break;
  default: {
    GGML_ABORT("fatal error");
  }
  }

  // static bool first = true;
  // printf("ne0 = %d, ne1 = %d, ne2 = %d\n", dst->ne[0], dst->ne[1], dst->ne[2]);
  // if (first) {
  //     first = false;
  // } else {
  //     for (int k = 0; k < dst->ne[1]; ++k) {
  //         for (int j = 0; j < dst->ne[0]/16; ++j) {
  //             for (int i = 0; i < 16; ++i) {
  //                 printf("%8.4f ", ((float *) dst->data)[k*dst->ne[0] + j*16 + i]);
  //             }
  //             printf("\n");
  //         }
  //         printf("\n");
  //     }
  //     printf("\n");
  //     exit(0);
  // }
}

// ggml_compute_forward_diag

static void fGgmlComputeForwardDiagF32(const ggml_compute_params *pParams, ggml_tensor *dst) {

  const ggml_tensor *src0 = dst->src[0];

  if (pParams->ith != 0) {
    return;
  }

  // TODO: handle transposed/permuted matrices

  GGML_TENSOR_UNARY_OP_LOCALS

  GGML_ASSERT(ne00 == ne0);
  GGML_ASSERT(ne00 == ne1);
  GGML_ASSERT(ne01 == 1);
  GGML_ASSERT(ne02 == ne2);
  GGML_ASSERT(ne03 == ne3);

  GGML_ASSERT(nb00 == sizeof(float));
  GGML_ASSERT(nb0 == sizeof(float));

  for (int vI3 = 0; vI3 < ne3; vI3++) {
    for (int vI2 = 0; vI2 < ne2; vI2++) {
      for (int vI1 = 0; vI1 < ne1; vI1++) {
        float *vD = (float *)((char *)dst->data + vI3 * nb3 + vI2 * nb2 + vI1 * nb1);
        float *vS = (float *)((char *)src0->data + vI3 * nb03 + vI2 * nb02);
        for (int vI0 = 0; vI0 < vI1; vI0++) {
          vD[vI0] = 0;
        }
        vD[vI1] = vS[vI1];
        for (int vI0 = vI1 + 1; vI0 < ne0; vI0++) {
          vD[vI0] = 0;
        }
      }
    }
  }
}

void ggml_compute_forward_diag(const ggml_compute_params *pParams, ggml_tensor *pDst) {

  const ggml_tensor *vSrc0 = pDst->src[0];

  switch (vSrc0->type) {
  case GGML_TYPE_F32: {
    fGgmlComputeForwardDiagF32(pParams, pDst);
  } break;
  default: {
    GGML_ABORT("fatal error");
  }
  }
}

// ggml_compute_forward_diag_mask_inf

static void fGgmlComputeForwardDiagMaskF32(const ggml_compute_params *pParams, ggml_tensor *pDst, const float pValue) {

  const ggml_tensor *vSrc0 = pDst->src[0];

  const int cIth = pParams->ith;
  const int cNth = pParams->nth;

  const int cNPast = ((int32_t *)pDst->op_params)[0];
  const bool cInplace = vSrc0->data == pDst->data;

  GGML_ASSERT(cNPast >= 0);

  if (!cInplace) {
    if (cIth == 0) {
      // memcpy needs to be synchronized across threads to avoid race conditions.
      // => do it in INIT phase
      GGML_ASSERT(ggml_nelements(pDst) == ggml_nelements(vSrc0));
      GGML_ASSERT(ggml_is_contiguous(pDst) && ggml_is_contiguous(vSrc0));
      memcpy(((char *)pDst->data), ((char *)vSrc0->data), ggml_nbytes(pDst));
    }
    ggml_barrier(pParams->threadpool);
  }

  // TODO: handle transposed/permuted matrices

  const int cN = ggml_nrows(vSrc0);
  const int cNc = vSrc0->ne[0];
  const int cNr = vSrc0->ne[1];
  const int cNz = cN / cNr;

  GGML_ASSERT(pDst->nb[0] == sizeof(float));
  GGML_ASSERT(vSrc0->nb[0] == sizeof(float));

  for (int vK = 0; vK < cNz; vK++) {
    for (int vJ = cIth; vJ < cNr; vJ += cNth) {
      for (int vI = cNPast; vI < cNc; vI++) {
        if (vI > cNPast + vJ) {
          *(float *)((char *)pDst->data + vK * pDst->nb[2] + vJ * pDst->nb[1] + vI * pDst->nb[0]) = pValue;
        }
      }
    }
  }
}

void ggml_compute_forward_diag_mask_inf(const ggml_compute_params *pParams, ggml_tensor *pDst) {

  const ggml_tensor *vSrc0 = pDst->src[0];

  switch (vSrc0->type) {
  case GGML_TYPE_F32: {
    fGgmlComputeForwardDiagMaskF32(pParams, pDst, -INFINITY);
  } break;
  default: {
    GGML_ABORT("fatal error");
  }
  }
}

void ggml_compute_forward_diag_mask_zero(const ggml_compute_params *pParams, ggml_tensor *pDst) {

  const ggml_tensor *vSrc0 = pDst->src[0];

  switch (vSrc0->type) {
  case GGML_TYPE_F32: {
    fGgmlComputeForwardDiagMaskF32(pParams, pDst, 0);
  } break;
  default: {
    GGML_ABORT("fatal error");
  }
  }
}

// ggml_compute_forward_soft_max

static void fGgmlComputeForwardSoftMaxF32(const ggml_compute_params *pParams, ggml_tensor *dst) {

  const ggml_tensor *src0 = dst->src[0];
  const ggml_tensor *vSrc1 = dst->src[1];
  const ggml_tensor *vSrc2 = dst->src[2];

  assert(ggml_is_contiguous(dst));
  assert(ggml_are_same_shape(src0, dst));

  float vScale = 1.0f;
  float vMaxBias = 0.0f;

  memcpy(&vScale, (float *)dst->op_params + 0, sizeof(float));
  memcpy(&vMaxBias, (float *)dst->op_params + 1, sizeof(float));

  const int cIth = pParams->ith;
  const int cNth = pParams->nth;

  GGML_TENSOR_UNARY_OP_LOCALS

  const int64_t cNb11 = vSrc1 ? vSrc1->nb[1] : 1;
  const int64_t cNb12 = vSrc1 ? vSrc1->nb[2] : 1;
  const int64_t cNb13 = vSrc1 ? vSrc1->nb[3] : 1;

  const int64_t cNe12 = vSrc1 ? vSrc1->ne[2] : 1;
  const int64_t cNe13 = vSrc1 ? vSrc1->ne[3] : 1;

  // TODO: is this supposed to be ceil instead of floor?
  //       https://huggingface.co/mosaicml/mpt-7b/blob/main/attention.py#L370
  const uint32_t cNHead = ne02;
  const uint32_t cNHeadLog2 = 1u << (uint32_t)floor(log2(cNHead));

  const float cM0 = powf(2.0f, -(vMaxBias) / cNHeadLog2);
  const float cM1 = powf(2.0f, -(vMaxBias / 2.0f) / cNHeadLog2);

  float *vWp = (float *)pParams->wdata + (ne00 + CACHE_LINE_SIZE_F32) * cIth;

  const bool cUseF16 = (vSrc1 && vSrc1->type == GGML_TYPE_F16);

  // sinks
  const float *vSk = vSrc2 ? (float *)((char *)vSrc2->data) : nullptr;

  for (int64_t vI03 = 0; vI03 < ne03; vI03++) {
    for (int64_t vI02 = 0; vI02 < ne02; vI02++) {
      for (int64_t vI01 = cIth; vI01 < ne01; vI01 += cNth) {
        const int64_t cI11 = vI01;
        const int64_t cI12 = vI02 % cNe12;
        const int64_t cI13 = vI03 % cNe13;

        // ALiBi
        const uint32_t cH = vI02; // head
        const float cSlope =
            (vMaxBias > 0.0f) ? cH < cNHeadLog2 ? powf(cM0, cH + 1) : powf(cM1, 2 * (cH - cNHeadLog2) + 1) : 1.0f;

        float *vSp = (float *)((char *)src0->data + vI01 * nb01 + vI02 * nb02 + vI03 * nb03);
        float *vDp = (float *)((char *)dst->data + vI01 * nb1 + vI02 * nb2 + vI03 * nb3);

        // broadcast the mask across rows
        ggml_fp16_t *vMpF16 =
            vSrc1 ? (ggml_fp16_t *)((char *)vSrc1->data + cI11 * cNb11 + cI12 * cNb12 + cI13 * cNb13) : NULL;
        float *vMpF32 = vSrc1 ? (float *)((char *)vSrc1->data + cI11 * cNb11 + cI12 * cNb12 + cI13 * cNb13) : NULL;

        ggml_vec_cpy_f32(ne00, vWp, vSp);
        ggml_vec_scale_f32(ne00, vWp, vScale);
        if (vMpF32) {
          if (cUseF16) {
            for (int vI = 0; vI < ne00; ++vI) {
              vWp[vI] += cSlope * GGML_CPU_FP16_TO_FP32(vMpF16[vI]);
            }
          } else {
            for (int vI = 0; vI < ne00; ++vI) {
              vWp[vI] += cSlope * vMpF32[vI];
            }
          }
        }

#ifndef NDEBUG
        for (int vI = 0; vI < ne00; ++vI) {
          // printf("p[%d] = %f\n", i, p[i]);
          assert(!isnan(vWp[vI]));
        }
#endif // NDEBUG

        float vMax = -INFINITY;
        ggml_vec_max_f32(ne00, &vMax, vWp);

        // if we have sinks, make a correction as if they were included in the softmax
        if (vSk) {
          vMax = MAX(vMax, vSk[vI02]);
        }

        ggml_float vSum = ggml_vec_soft_max_f32(ne00, vDp, vWp, vMax);
        assert(vSum > 0.0);

        if (vSk) {
          vSum += (ggml_float)expf(vSk[vI02] - vMax);
        }

        vSum = 1.0 / vSum;
        ggml_vec_scale_f32(ne00, vDp, vSum);

#ifndef NDEBUG
        for (int vI = 0; vI < ne00; ++vI) {
          assert(!isnan(vDp[vI]));
          assert(!isinf(vDp[vI]));
        }
#endif // NDEBUG
      }
    }
  }
}

void ggml_compute_forward_soft_max(const ggml_compute_params *pParams, ggml_tensor *pDst) {

  const ggml_tensor *vSrc0 = pDst->src[0];

  switch (vSrc0->type) {
  case GGML_TYPE_F32: {
    fGgmlComputeForwardSoftMaxF32(pParams, pDst);
  } break;
  default: {
    GGML_ABORT("fatal error");
  }
  }
}

// ggml_compute_forward_soft_max_ext_back

static void fGgmlComputeForwardSoftMaxExtBackF32(const ggml_compute_params *pParams, ggml_tensor *pDst) {

  const ggml_tensor *vSrc0 = pDst->src[0];
  const ggml_tensor *vSrc1 = pDst->src[1];

  GGML_ASSERT(ggml_is_contiguous(vSrc0));
  GGML_ASSERT(ggml_is_contiguous(vSrc1));
  GGML_ASSERT(ggml_is_contiguous(pDst));
  GGML_ASSERT(ggml_are_same_shape(vSrc0, pDst));
  GGML_ASSERT(ggml_are_same_shape(vSrc1, pDst));

  float vScale = 1.0f;
  float vMaxBias = 0.0f;

  memcpy(&vScale, (const float *)pDst->op_params + 0, sizeof(float));
  memcpy(&vMaxBias, (const float *)pDst->op_params + 1, sizeof(float));

  GGML_ASSERT(vMaxBias == 0.0f);

  // TODO: handle transposed/permuted matrices

  const int cIth = pParams->ith;
  const int cNth = pParams->nth;

  const int cNc = vSrc0->ne[0];
  const int cNr = ggml_nrows(vSrc0);

  // rows per thread
  const int cDr = (cNr + cNth - 1) / cNth;

  // row range for this thread
  const int cIr0 = cDr * cIth;
  const int cIr1 = MIN(cIr0 + cDr, cNr);

  for (int vI1 = cIr0; vI1 < cIr1; vI1++) {
    float *vDy = (float *)((char *)vSrc0->data + vI1 * vSrc0->nb[1]);
    float *vY = (float *)((char *)vSrc1->data + vI1 * vSrc1->nb[1]);
    float *vDx = (float *)((char *)pDst->data + vI1 * pDst->nb[1]);

#ifndef NDEBUG
    for (int vI = 0; vI < cNc; ++vI) {
      // printf("p[%d] = %f\n", i, p[i]);
      assert(!isnan(vDy[vI]));
      assert(!isnan(vY[vI]));
    }
#endif // NDEBUG
       // Jii = yi - yi*yi
       // Jij = -yi*yj
       // J = diag(y)-y.T*y
       // dx = J * dy
       // dxk = sum_i(Jki * dyi)
       // dxk = sum_i(-yk*yi * dyi) - (-yk*yk)*dyk + (yk - yk*yk)*dyk
       // dxk = sum_i(-yk*yi * dyi) + yk*yk*dyk + yk*dyk - yk*yk*dyk
       // dxk = sum_i(-yk*yi * dyi) + yk*dyk
       // dxk = -yk * sum_i(yi * dyi) + yk*dyk
       // dxk = -yk * dot(y, dy) + yk*dyk
       // dxk = yk * (- dot(y, dy) + dyk)
       // dxk = yk * (dyk - dot(y, dy))
       //
       // post-order:
       // dot_y_dy := dot(y, dy)
       // dx := dy
       // dx := dx - dot_y_dy
       // dx := dx * y

    // linear runtime, no additional memory
    float vDotYDy = 0;
    ggml_vec_dot_f32(cNc, &vDotYDy, 0, vY, 0, vDy, 0, 1);
    ggml_vec_cpy_f32(cNc, vDx, vDy);
    ggml_vec_acc1_f32(cNc, vDx, -vDotYDy);
    ggml_vec_mul_f32(cNc, vDx, vDx, vY);
    ggml_vec_scale_f32(cNc, vDx, vScale);

#ifndef NDEBUG
    for (int vI = 0; vI < cNc; ++vI) {
      assert(!isnan(vDx[vI]));
      assert(!isinf(vDx[vI]));
    }
#endif // NDEBUG
  }
}

void ggml_compute_forward_soft_max_ext_back(const ggml_compute_params *pParams, ggml_tensor *pDst) {

  const ggml_tensor *vSrc0 = pDst->src[0];

  switch (vSrc0->type) {
  case GGML_TYPE_F32: {
    fGgmlComputeForwardSoftMaxExtBackF32(pParams, pDst);
  } break;
  default: {
    GGML_ABORT("fatal error");
  }
  }
}

// ggml_compute_forward_clamp

static void fGgmlComputeForwardClampF32(const ggml_compute_params *pParams, ggml_tensor *pDst) {

  const ggml_tensor *vSrc0 = pDst->src[0];

  float vMin;
  float vMax;
  memcpy(&vMin, (float *)pDst->op_params + 0, sizeof(float));
  memcpy(&vMax, (float *)pDst->op_params + 1, sizeof(float));

  const int cIth = pParams->ith;
  const int cNth = pParams->nth;

  const int cN = ggml_nrows(vSrc0);
  const int cNc = vSrc0->ne[0];

  const size_t cNb00 = vSrc0->nb[0];
  const size_t cNb01 = vSrc0->nb[1];

  const size_t cNb0 = pDst->nb[0];
  const size_t cNb1 = pDst->nb[1];

  GGML_ASSERT(cNb0 == sizeof(float));
  GGML_ASSERT(cNb00 == sizeof(float));

  for (int vJ = cIth; vJ < cN; vJ += cNth) {
    float *vDstPtr = (float *)((char *)pDst->data + vJ * cNb1);
    float *vSrc0Ptr = (float *)((char *)vSrc0->data + vJ * cNb01);

    for (int vI = 0; vI < cNc; vI++) {
      vDstPtr[vI] = MAX(MIN(vSrc0Ptr[vI], vMax), vMin);
    }
  }
}

static void fGgmlComputeForwardClampF16(const ggml_compute_params *pParams, ggml_tensor *pDst) {

  const ggml_tensor *vSrc0 = pDst->src[0];

  float vMin;
  float vMax;
  memcpy(&vMin, (float *)pDst->op_params + 0, sizeof(float));
  memcpy(&vMax, (float *)pDst->op_params + 1, sizeof(float));

  const int cIth = pParams->ith;
  const int cNth = pParams->nth;

  const int cN = ggml_nrows(vSrc0);
  const int cNc = vSrc0->ne[0];

  const size_t cNb00 = vSrc0->nb[0];
  const size_t cNb01 = vSrc0->nb[1];

  const size_t cNb0 = pDst->nb[0];
  const size_t cNb1 = pDst->nb[1];

  GGML_ASSERT(cNb0 == sizeof(ggml_fp16_t));
  GGML_ASSERT(cNb00 == sizeof(ggml_fp16_t));

  for (int vJ = cIth; vJ < cN; vJ += cNth) {
    ggml_fp16_t *vDstPtr = (ggml_fp16_t *)((char *)pDst->data + vJ * cNb1);
    ggml_fp16_t *vSrc0Ptr = (ggml_fp16_t *)((char *)vSrc0->data + vJ * cNb01);

    for (int vI = 0; vI < cNc; vI++) {
      float v_ = GGML_CPU_FP16_TO_FP32(vSrc0Ptr[vI]);
      vDstPtr[vI] = GGML_CPU_FP32_TO_FP16(MAX(MIN(v_, vMax), vMin));
    }
  }
}

void ggml_compute_forward_clamp(const ggml_compute_params *pParams, ggml_tensor *pDst) {

  const ggml_tensor *vSrc0 = pDst->src[0];

  switch (vSrc0->type) {
  case GGML_TYPE_F32: {
    fGgmlComputeForwardClampF32(pParams, pDst);
  } break;
  case GGML_TYPE_F16: {
    fGgmlComputeForwardClampF16(pParams, pDst);
  } break;
  case GGML_TYPE_BF16:
  case GGML_TYPE_Q1_0:
  case GGML_TYPE_Q2_0:
  case GGML_TYPE_Q4_0:
  case GGML_TYPE_Q4_1:
  case GGML_TYPE_Q5_0:
  case GGML_TYPE_Q5_1:
  case GGML_TYPE_Q8_0:
  case GGML_TYPE_Q8_1:
  case GGML_TYPE_MXFP4:
  case GGML_TYPE_NVFP4:
  case GGML_TYPE_Q2_K:
  case GGML_TYPE_Q3_K:
  case GGML_TYPE_Q4_K:
  case GGML_TYPE_Q5_K:
  case GGML_TYPE_Q6_K:
  case GGML_TYPE_TQ1_0:
  case GGML_TYPE_TQ2_0:
  case GGML_TYPE_IQ2_XXS:
  case GGML_TYPE_IQ2_XS:
  case GGML_TYPE_IQ3_XXS:
  case GGML_TYPE_IQ1_S:
  case GGML_TYPE_IQ1_M:
  case GGML_TYPE_IQ4_NL:
  case GGML_TYPE_IQ4_XS:
  case GGML_TYPE_IQ3_S:
  case GGML_TYPE_IQ2_S:
  case GGML_TYPE_Q8_K:
  case GGML_TYPE_I8:
  case GGML_TYPE_I16:
  case GGML_TYPE_I32:
  case GGML_TYPE_I64:
  case GGML_TYPE_F64:
  case GGML_TYPE_COUNT: {
    GGML_ABORT("fatal error");
  }
  }
}

// ggml_compute_forward_rope

static float fRopeYarnRamp(const float pLow, const float pHigh, const int pI0) {
  const float cY = (pI0 / 2 - pLow) / MAX(0.001f, pHigh - pLow);
  return 1 - MIN(1, MAX(0, cY));
}

// YaRN algorithm based on LlamaYaRNScaledRotaryEmbedding.py from https://github.com/jquesnelle/yarn
// MIT licensed. Copyright (c) 2023 Jeffrey Quesnelle and Bowen Peng.
static void fRopeYarn(float pThetaExtrap, float pFreqScale, float pCorrDims[2], int64_t pI0, float pExtFactor,
                      float pMscale, float *pCosTheta, float *pSinTheta) {
  // Get n-d rotational scaling corrected for extrapolation
  float vThetaInterp = pFreqScale * pThetaExtrap;
  float vTheta = vThetaInterp;
  if (pExtFactor != 0.0f) {
    float vRampMix = fRopeYarnRamp(pCorrDims[0], pCorrDims[1], pI0) * pExtFactor;
    vTheta = vThetaInterp * (1 - vRampMix) + pThetaExtrap * vRampMix;

    // Get n-d magnitude scaling corrected for interpolation
    pMscale *= 1.0f + 0.1f * logf(1.0f / pFreqScale);
  }
  *pCosTheta = cosf(vTheta) * pMscale;
  *pSinTheta = sinf(vTheta) * pMscale;
}

static void fGgmlRopeCacheInit(float pThetaBase, float pFreqScale, const float *pFreqFactors, float pCorrDims[2],
                               int64_t pNe0, float pExtFactor, float pMscale, float *pCache, float pSinSign,
                               float pThetaScale) {
  // ref: https://github.com/jquesnelle/yarn/blob/master/scaled_rope/LlamaYaRNScaledRotaryEmbedding.py
  float vTheta = pThetaBase;
  for (int64_t vI0 = 0; vI0 < pNe0; vI0 += 2) {
    const float cFf = pFreqFactors ? pFreqFactors[vI0 / 2] : 1.0f;
    fRopeYarn(vTheta / cFf, pFreqScale, pCorrDims, vI0, pExtFactor, pMscale, &pCache[vI0 + 0], &pCache[vI0 + 1]);
    pCache[vI0 + 1] *= pSinSign;

    vTheta *= pThetaScale;
  }
}

static void fGgmlMropeCacheInit(float pThetaBaseT, float pThetaBaseH, float pThetaBaseW, float pThetaBaseE,
                                int pSections[4], bool pIsImrope, bool pIndepSects, float pFreqScale,
                                const float *pFreqFactors, float pCorrDims[2], int64_t pNe0, float pExtFactor,
                                float pMscale, float *pCache, float pSinSign, float pThetaScale) {
  // ref: https://github.com/jquesnelle/yarn/blob/master/scaled_rope/LlamaYaRNScaledRotaryEmbedding.py
  float vThetaT = pThetaBaseT;
  float vThetaH = pThetaBaseH;
  float vThetaW = pThetaBaseW;
  float vThetaE = pThetaBaseE; // extra position id for vision encoder
  int vSectDims = pSections[0] + pSections[1] + pSections[2] + pSections[3];
  int vSecW = pSections[1] + pSections[0];
  int vSecE = pSections[2] + vSecW;
  GGML_ASSERT(vSectDims <= pNe0);

  for (int64_t vI0 = 0; vI0 < pNe0; vI0 += 2) {
    const float cFf = pFreqFactors ? pFreqFactors[vI0 / 2] : 1.0f;

    int vSector = (vI0 / 2) % vSectDims;
    if (pIndepSects) {
      // compute theta independently for each dim sections
      // (i.e. reset corresponding theta when `i0` go from one section to another)
      if (vSector == 0) {
        vThetaT = pThetaBaseT;
      } else if (vSector == pSections[0]) {
        vThetaH = pThetaBaseH;
        ;
      } else if (vSector == vSecW) {
        vThetaW = pThetaBaseW;
      } else if (vSector == vSecE) {
        vThetaE = pThetaBaseE;
      }
    }

    float vTheta = vThetaT;
    if (pIsImrope) { // qwen3vl apply interleaved mrope
      if (vSector % 3 == 1 && vSector < 3 * pSections[1]) {
        vTheta = vThetaH;
      } else if (vSector % 3 == 2 && vSector < 3 * pSections[2]) {
        vTheta = vThetaW;
      } else if (vSector % 3 == 0 && vSector < 3 * pSections[0]) {
        vTheta = vThetaT;
      } else {
        vTheta = vThetaE;
      }
    } else {
      if (vSector >= pSections[0] && vSector < vSecW) {
        vTheta = vThetaH;
      } else if (vSector >= vSecW && vSector < vSecW + pSections[2]) {
        vTheta = vThetaW;
      } else if (vSector >= vSecW + pSections[2]) {
        vTheta = vThetaE;
      }
    }

    fRopeYarn(vTheta / cFf, pFreqScale, pCorrDims, vI0, pExtFactor, pMscale, &pCache[vI0 + 0], &pCache[vI0 + 1]);
    pCache[vI0 + 1] *= pSinSign;

    vThetaT *= pThetaScale;
    vThetaW *= pThetaScale;
    vThetaH *= pThetaScale;
    vThetaE *= pThetaScale;
  }
}

template <typename T>
static void fRotatePairs(const int64_t pN, const int64_t pNOffset, const float *pCache, const T *pSrcData, T *pDstData,
                         const int pScale = 2) {
  for (int64_t vI0 = 0; vI0 < pN; vI0 += 2) {
    const int64_t cIc =
        vI0 / pScale; // hack for GGML_ROPE_TYPE_NORMAL, where we need ic = i0; for all other cases, ic = i0/2

    const float cCosTheta = pCache[vI0 + 0];
    const float cSinTheta = pCache[vI0 + 1];

    const T *const cSrc = pSrcData + cIc;
    T *vDst = pDstData + cIc;

    const float cX0 = type_conversion_table<T>::to_f32(cSrc[0]);
    const float cX1 = type_conversion_table<T>::to_f32(cSrc[pNOffset]);

    vDst[0] = type_conversion_table<T>::from_f32(cX0 * cCosTheta - cX1 * cSinTheta);
    vDst[pNOffset] = type_conversion_table<T>::from_f32(cX0 * cSinTheta + cX1 * cCosTheta);
  }
}

template <typename T> // float or ggml_fp16_t
static void fGgmlComputeForwardRopeFlt(const ggml_compute_params *pParams, ggml_tensor *dst, const bool pForward) {

  const ggml_tensor *src0 = dst->src[0];
  const ggml_tensor *vSrc1 = dst->src[1];
  const ggml_tensor *vSrc2 = dst->src[2];

  GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);
  GGML_ASSERT(vSrc1->type == GGML_TYPE_I32);

  float vFreqBase, vFreqScale, vExtFactor, vAttnFactor, vBetaFast, vBetaSlow;
  int vSections[4];

  // const int n_past     = ((int32_t *) dst->op_params)[0];
  const int cNDims = ((int32_t *)dst->op_params)[1];
  const int cMode = ((int32_t *)dst->op_params)[2];
  // const int n_ctx      = ((int32_t *) dst->op_params)[3];
  const int cNCtxOrig = ((int32_t *)dst->op_params)[4];

  memcpy(&vFreqBase, (int32_t *)dst->op_params + 5, sizeof(float));
  memcpy(&vFreqScale, (int32_t *)dst->op_params + 6, sizeof(float));
  memcpy(&vExtFactor, (int32_t *)dst->op_params + 7, sizeof(float));
  memcpy(&vAttnFactor, (int32_t *)dst->op_params + 8, sizeof(float));
  memcpy(&vBetaFast, (int32_t *)dst->op_params + 9, sizeof(float));
  memcpy(&vBetaSlow, (int32_t *)dst->op_params + 10, sizeof(float));
  memcpy(&vSections, (int32_t *)dst->op_params + 11, sizeof(int) * 4);

  GGML_TENSOR_UNARY_OP_LOCALS

  // printf("ne0: %d, ne1: %d, ne2: %d, ne3: %d\n", ne0, ne1, ne2, ne3);
  // printf("n_past = %d, ne2 = %d\n", n_past, ne2);

  GGML_ASSERT(nb0 == nb00);
  GGML_ASSERT(nb0 == sizeof(T));

  const int cIth = pParams->ith;
  const int cNth = pParams->nth;

  const int cNr = ggml_nrows(dst);

  GGML_ASSERT(cNDims <= ne0);
  GGML_ASSERT(cNDims % 2 == 0);

  // rows per thread
  const int cDr = (cNr + cNth - 1) / cNth;

  // row range for this thread
  const int cIr0 = cDr * cIth;
  const int cIr1 = MIN(cIr0 + cDr, cNr);

  // row index used to determine which thread to use
  int vIr = 0;

  const float cThetaScale = powf(vFreqBase, -2.0f / cNDims);

  float aCorrDims[2];
  ggml_rope_yarn_corr_dims(cNDims, cNCtxOrig, vFreqBase, vBetaFast, vBetaSlow, aCorrDims);

  const bool cIsImrope = cMode == GGML_ROPE_TYPE_IMROPE; // qwen3vl apply interleaved mrope
  const bool cMropeUsed =
      cMode & GGML_ROPE_TYPE_MROPE; // ggml_rope_multi, note: also true for vision (24 & 8 == true) and for imrope
  const bool cIsVision = cMode == GGML_ROPE_TYPE_VISION;

  if (cMropeUsed) {
    GGML_ASSERT(vSections[0] > 0 || vSections[1] > 0 || vSections[2] > 0);
  }

  if (cIsVision) {
    GGML_ASSERT(cNDims == ne0 / 2);
  }

  const float *vFreqFactors = NULL;
  if (vSrc2 != NULL) {
    GGML_ASSERT(vSrc2->type == GGML_TYPE_F32);
    GGML_ASSERT(vSrc2->ne[0] >= cNDims / 2);
    vFreqFactors = (const float *)vSrc2->data;
  }

  // backward process uses inverse rotation by cos and sin.
  // cos and sin build a rotation matrix, where the inverse is the transpose.
  // this essentially just switches the sign of sin.
  const float cSinSign = pForward ? 1.0f : -1.0f;

  const int32_t *vPos = (const int32_t *)vSrc1->data;

  int64_t vLastI2 = -1;

  for (int64_t vI3 = 0; vI3 < ne3; vI3++) {     // batch
    for (int64_t vI2 = 0; vI2 < ne2; vI2++) {   // seq-len
      for (int64_t vI1 = 0; vI1 < ne1; vI1++) { // attn-heads
        if (vIr++ < cIr0)
          continue; // skip rows mapped to other threads
        if (vIr > cIr1)
          break;

        float *vCache = (float *)pParams->wdata + (ne0 + CACHE_LINE_SIZE_F32) * cIth;
        if (vLastI2 != vI2) {
          if (!cMropeUsed) {
            const int64_t cP = vPos[vI2];
            fGgmlRopeCacheInit(cP, vFreqScale, vFreqFactors, aCorrDims, ne0, vExtFactor, vAttnFactor, vCache, cSinSign,
                               cThetaScale);
          } else {
            const int64_t cPT = vPos[vI2];
            const int64_t cPH = vPos[vI2 + ne2];
            const int64_t cPW = vPos[vI2 + ne2 * 2];
            const int64_t cPE = vPos[vI2 + ne2 * 3];
            fGgmlMropeCacheInit(cPT, cPH, cPW, cPE, vSections, cIsImrope, cIsVision, vFreqScale, vFreqFactors,
                                aCorrDims, ne0, vExtFactor, vAttnFactor, vCache, cSinSign, cThetaScale);
          }

          vLastI2 = vI2;
        }

        T *vSrc = (T *)((char *)src0->data + vI3 * nb03 + vI2 * nb02 + vI1 * nb01);
        T *vDstData = (T *)((char *)dst->data + vI3 * nb3 + vI2 * nb2 + vI1 * nb1);

        switch (cMode) {
        case GGML_ROPE_TYPE_NORMAL:
          fRotatePairs<T>(cNDims, 1, vCache, vSrc, vDstData, 1);
          break;
        case GGML_ROPE_TYPE_NEOX:
        case GGML_ROPE_TYPE_MROPE:
        case GGML_ROPE_TYPE_IMROPE:
          fRotatePairs<T>(cNDims, cNDims / 2, vCache, vSrc, vDstData);
          break;
        case GGML_ROPE_TYPE_VISION:
          fRotatePairs<T>(ne0, cNDims, vCache, vSrc, vDstData);
          break;
        default:
          GGML_ABORT("rope type not supported");
        }

        if (!cIsVision) {
          // fill the remain channels with data from src tensor
          for (int64_t vI0 = cNDims; vI0 < ne0; vI0 += 2) {
            const T *const cSrc = (T *)((char *)src0->data + vI3 * nb03 + vI2 * nb02 + vI1 * nb01 + vI0 * nb00);
            T *vDstData = (T *)((char *)dst->data + vI3 * nb3 + vI2 * nb2 + vI1 * nb1 + vI0 * nb0);

            vDstData[0] = cSrc[0];
            vDstData[1] = cSrc[1];
          }
        }
      } // attn-heads
    }
  }
}

void ggml_compute_forward_rope(const ggml_compute_params *pParams, ggml_tensor *pDst) {

  const ggml_tensor *vSrc0 = pDst->src[0];

  switch (vSrc0->type) {
  case GGML_TYPE_F16: {
    fGgmlComputeForwardRopeFlt<ggml_fp16_t>(pParams, pDst, true);
  } break;
  case GGML_TYPE_F32: {
    fGgmlComputeForwardRopeFlt<float>(pParams, pDst, true);
  } break;
  default: {
    GGML_ABORT("fatal error");
  }
  }
}

// ggml_compute_forward_rope_back

void ggml_compute_forward_rope_back(const ggml_compute_params *pParams, ggml_tensor *pDst) {

  const ggml_tensor *vSrc0 = pDst->src[0];

  switch (vSrc0->type) {
  case GGML_TYPE_F16: {
    fGgmlComputeForwardRopeFlt<ggml_fp16_t>(pParams, pDst, false);
  } break;
  case GGML_TYPE_F32: {
    fGgmlComputeForwardRopeFlt<float>(pParams, pDst, false);
  } break;
  default: {
    GGML_ABORT("fatal error");
  }
  }
}

// ggml_compute_forward_conv_transpose_1d

static void fGgmlComputeForwardConvTranspose1dF16F32(const ggml_compute_params *pParams, ggml_tensor *dst) {

  const ggml_tensor *src0 = dst->src[0];
  const ggml_tensor *src1 = dst->src[1];

  GGML_ASSERT(src0->type == GGML_TYPE_F16);
  GGML_ASSERT(src1->type == GGML_TYPE_F32);
  GGML_ASSERT(dst->type == GGML_TYPE_F32);

  GGML_TENSOR_BINARY_OP_LOCALS

  const int cIth = pParams->ith;
  const int cNth = pParams->nth;

  const int cNk = ne00 * ne01 * ne02;

  GGML_ASSERT(nb00 == sizeof(ggml_fp16_t));
  GGML_ASSERT(nb10 == sizeof(float));

  if (cIth == 0) {
    memset(pParams->wdata, 0, pParams->wsize);

    // permute kernel data (src0) from (K x Cout x Cin) to (Cin x K x Cout)
    {
      ggml_fp16_t *const cWdata = (ggml_fp16_t *)pParams->wdata + 0;

      for (int64_t vI02 = 0; vI02 < ne02; vI02++) {
        for (int64_t vI01 = 0; vI01 < ne01; vI01++) {
          const ggml_fp16_t *const cSrc = (ggml_fp16_t *)((char *)src0->data + vI02 * nb02 + vI01 * nb01);
          ggml_fp16_t *vDstData = cWdata + vI01 * ne00 * ne02;
          for (int64_t vI00 = 0; vI00 < ne00; vI00++) {
            vDstData[vI00 * ne02 + vI02] = cSrc[vI00];
          }
        }
      }
    }

    // permute source data (src1) from (L x Cin) to (Cin x L)
    {
      ggml_fp16_t *const cWdata = (ggml_fp16_t *)pParams->wdata + cNk;
      ggml_fp16_t *vDstData = cWdata;

      for (int64_t vI11 = 0; vI11 < ne11; vI11++) {
        const float *const cSrc = (float *)((char *)src1->data + vI11 * nb11);
        for (int64_t vI10 = 0; vI10 < ne10; vI10++) {
          vDstData[vI10 * ne11 + vI11] = GGML_CPU_FP32_TO_FP16(cSrc[vI10]);
        }
      }
    }

    // need to zero dst since we are accumulating into it
    memset(dst->data, 0, ggml_nbytes(dst));
  }
  ggml_barrier(pParams->threadpool);

  const int32_t cS0 = ((const int32_t *)(dst->op_params))[0];

  // total rows in dst
  const int cNr = ne1;

  // rows per thread
  const int cDr = (cNr + cNth - 1) / cNth;

  // row range for this thread
  const int cIr0 = cDr * cIth;
  const int cIr1 = MIN(cIr0 + cDr, cNr);

  ggml_fp16_t *const cWdata = (ggml_fp16_t *)pParams->wdata + 0;
  ggml_fp16_t *const cWdataSrc = cWdata + cNk;

  for (int vI1 = cIr0; vI1 < cIr1; vI1++) {
    float *vDstData = (float *)((char *)dst->data + vI1 * nb1);
    ggml_fp16_t *vWdataKernel = cWdata + vI1 * ne02 * ne00;
    for (int vI10 = 0; vI10 < ne10; vI10++) {
      const int cI1n = vI10 * ne11;
      for (int vI00 = 0; vI00 < ne00; vI00++) {
        float v_ = 0;
        ggml_vec_dot_f16(ne02, &v_, 0, (ggml_fp16_t *)cWdataSrc + cI1n, 0, (ggml_fp16_t *)vWdataKernel + vI00 * ne02, 0,
                         1);
        vDstData[vI10 * cS0 + vI00] += v_;
      }
    }
  }
}

static void fGgmlComputeForwardConvTranspose1dF32(const ggml_compute_params *pParams, ggml_tensor *dst) {

  const ggml_tensor *src0 = dst->src[0];
  const ggml_tensor *src1 = dst->src[1];

  GGML_ASSERT(src0->type == GGML_TYPE_F32);
  GGML_ASSERT(src1->type == GGML_TYPE_F32);
  GGML_ASSERT(dst->type == GGML_TYPE_F32);

  GGML_TENSOR_BINARY_OP_LOCALS

  const int cIth = pParams->ith;
  const int cNth = pParams->nth;

  const int cNk = ne00 * ne01 * ne02;

  GGML_ASSERT(nb00 == sizeof(float));
  GGML_ASSERT(nb10 == sizeof(float));

  if (cIth == 0) {
    memset(pParams->wdata, 0, pParams->wsize);

    // prepare kernel data (src0) from (K x Cout x Cin) to (Cin x K x Cout)
    {
      float *const cWdata = (float *)pParams->wdata + 0;

      for (int64_t vI02 = 0; vI02 < ne02; vI02++) {
        for (int64_t vI01 = 0; vI01 < ne01; vI01++) {
          const float *const cSrc = (float *)((char *)src0->data + vI02 * nb02 + vI01 * nb01);
          float *vDstData = cWdata + vI01 * ne00 * ne02;
          for (int64_t vI00 = 0; vI00 < ne00; vI00++) {
            vDstData[vI00 * ne02 + vI02] = cSrc[vI00];
          }
        }
      }
    }

    // prepare source data (src1)
    {
      float *const cWdata = (float *)pParams->wdata + cNk;
      float *vDstData = cWdata;

      for (int64_t vI11 = 0; vI11 < ne11; vI11++) {
        const float *const cSrc = (float *)((char *)src1->data + vI11 * nb11);
        for (int64_t vI10 = 0; vI10 < ne10; vI10++) {
          vDstData[vI10 * ne11 + vI11] = cSrc[vI10];
        }
      }
    }

    // need to zero dst since we are accumulating into it
    memset(dst->data, 0, ggml_nbytes(dst));
  }
  ggml_barrier(pParams->threadpool);

  const int32_t cS0 = ((const int32_t *)(dst->op_params))[0];

  // total rows in dst
  const int cNr = ne1;

  // rows per thread
  const int cDr = (cNr + cNth - 1) / cNth;

  // row range for this thread
  const int cIr0 = cDr * cIth;
  const int cIr1 = MIN(cIr0 + cDr, cNr);

  float *const cWdata = (float *)pParams->wdata + 0;
  float *const cWdataSrc = cWdata + cNk;

  for (int vI1 = cIr0; vI1 < cIr1; vI1++) {
    float *vDstData = (float *)((char *)dst->data + vI1 * nb1);
    float *vWdataKernel = cWdata + vI1 * ne02 * ne00;
    for (int vI10 = 0; vI10 < ne10; vI10++) {
      const int cI1n = vI10 * ne11;
      for (int vI00 = 0; vI00 < ne00; vI00++) {
        float v_ = 0;
        ggml_vec_dot_f32(ne02, &v_, 0, cWdataSrc + cI1n, 0, vWdataKernel + vI00 * ne02, 0, 1);
        vDstData[vI10 * cS0 + vI00] += v_;
      }
    }
  }
}

void ggml_compute_forward_conv_transpose_1d(const ggml_compute_params *pParams, ggml_tensor *pDst) {

  const ggml_tensor *vSrc0 = pDst->src[0];

  switch (vSrc0->type) {
  case GGML_TYPE_F16: {
    fGgmlComputeForwardConvTranspose1dF16F32(pParams, pDst);
  } break;
  case GGML_TYPE_F32: {
    fGgmlComputeForwardConvTranspose1dF32(pParams, pDst);
  } break;
  default: {
    GGML_ABORT("fatal error");
  }
  }
}

// ggml_compute_forward_im2col_f32
// src0: kernel [OC, IC, KH, KW]
// src1: image [N, IC, IH, IW]
// dst:  result [N, OH, OW, IC*KH*KW]
static void fGgmlComputeForwardIm2colF32(const ggml_compute_params *pParams, ggml_tensor *dst) {

  const ggml_tensor *src0 = dst->src[0];
  const ggml_tensor *src1 = dst->src[1];

  GGML_ASSERT(src1->type == GGML_TYPE_F32);
  GGML_ASSERT(dst->type == GGML_TYPE_F32);

  GGML_TENSOR_BINARY_OP_LOCALS;

  const int32_t cS0 = ((const int32_t *)(dst->op_params))[0];
  const int32_t cS1 = ((const int32_t *)(dst->op_params))[1];
  const int32_t cP0 = ((const int32_t *)(dst->op_params))[2];
  const int32_t cP1 = ((const int32_t *)(dst->op_params))[3];
  const int32_t cD0 = ((const int32_t *)(dst->op_params))[4];
  const int32_t cD1 = ((const int32_t *)(dst->op_params))[5];
  const bool cIs2D = ((const int32_t *)(dst->op_params))[6] == 1;

  const int cIth = pParams->ith;
  const int cNth = pParams->nth;

  const int64_t cN = cIs2D ? ne13 : ne12;
  const int64_t cIc = cIs2D ? ne12 : ne11;
  const int64_t cIh = cIs2D ? ne11 : 1;
  const int64_t cIw = ne10;

  const int64_t cKh = cIs2D ? ne01 : 1;
  const int64_t cKw = ne00;

  const int64_t cOh = cIs2D ? ne2 : 1;
  const int64_t cOw = ne1;

  int vOfs0 = cIs2D ? nb13 : nb12;
  int vOfs1 = cIs2D ? nb12 : nb11;

  GGML_ASSERT(nb10 == sizeof(float));

  // im2col: [N, IC, IH, IW] => [N, OH, OW, IC*KH*KW]
  {
    float *const cWdata = (float *)dst->data;

    for (int64_t vIn = 0; vIn < cN; vIn++) {
      for (int64_t vIoh = 0; vIoh < cOh; vIoh++) { // 1
        for (int64_t vIow = 0; vIow < cOw; vIow++) {
          for (int64_t vIic = cIth; vIic < cIc; vIic += cNth) {

            // micro kernel
            float *vDstData = cWdata + (vIn * cOh * cOw + vIoh * cOw + vIow) * (cIc * cKh * cKw);     // [IC, KH, KW]
            const float *const cSrcData = (float *)((char *)src1->data + vIn * vOfs0 + vIic * vOfs1); // [IH, IW]

            for (int64_t vIkh = 0; vIkh < cKh; vIkh++) { // 1
              for (int64_t vIkw = 0; vIkw < cKw; vIkw++) {
                const int64_t cIiw = vIow * cS0 + vIkw * cD0 - cP0;
                const int64_t cIih = vIoh * cS1 + vIkh * cD1 - cP1;

                if (cIih < 0 || cIih >= cIh || cIiw < 0 || cIiw >= cIw) {
                  vDstData[vIic * (cKh * cKw) + vIkh * cKw + vIkw] = 0;
                } else {
                  vDstData[vIic * (cKh * cKw) + vIkh * cKw + vIkw] = (cSrcData[cIih * cIw + cIiw]);
                }
              }
            }
          }
        }
      }
    }
  }
}

// ggml_compute_forward_im2col_f16
// src0: kernel [OC, IC, KH, KW]
// src1: image [N, IC, IH, IW]
// dst:  result [N, OH, OW, IC*KH*KW]
static void fGgmlComputeForwardIm2colF16(const ggml_compute_params *pParams, ggml_tensor *dst) {

  const ggml_tensor *src0 = dst->src[0];
  const ggml_tensor *src1 = dst->src[1];

  GGML_ASSERT(src1->type == GGML_TYPE_F16 || src1->type == GGML_TYPE_F32);
  GGML_ASSERT(dst->type == GGML_TYPE_F16);

  GGML_TENSOR_BINARY_OP_LOCALS;

  const int32_t cS0 = ((const int32_t *)(dst->op_params))[0];
  const int32_t cS1 = ((const int32_t *)(dst->op_params))[1];
  const int32_t cP0 = ((const int32_t *)(dst->op_params))[2];
  const int32_t cP1 = ((const int32_t *)(dst->op_params))[3];
  const int32_t cD0 = ((const int32_t *)(dst->op_params))[4];
  const int32_t cD1 = ((const int32_t *)(dst->op_params))[5];
  const bool cIs2D = ((const int32_t *)(dst->op_params))[6] == 1;

  const int cIth = pParams->ith;
  const int cNth = pParams->nth;

  const int64_t cN = cIs2D ? ne13 : ne12;
  const int64_t cIc = cIs2D ? ne12 : ne11;
  const int64_t cIh = cIs2D ? ne11 : 1;
  const int64_t cIw = ne10;

  const int64_t cKh = cIs2D ? ne01 : 1;
  const int64_t cKw = ne00;

  const int64_t cOh = cIs2D ? ne2 : 1;
  const int64_t cOw = ne1;

  int vOfs0 = cIs2D ? nb13 : nb12;
  int vOfs1 = cIs2D ? nb12 : nb11;

  GGML_ASSERT(nb10 == ggml_type_size(src1->type));

  // im2col: [N, IC, IH, IW] => [N, OH, OW, IC*KH*KW]
  {
    ggml_fp16_t *const cWdata = (ggml_fp16_t *)dst->data;

    for (int64_t vIn = 0; vIn < cN; vIn++) {
      for (int64_t vIoh = 0; vIoh < cOh; vIoh++) { // 1
        for (int64_t vIow = 0; vIow < cOw; vIow++) {
          for (int64_t vIic = cIth; vIic < cIc; vIic += cNth) {

            // micro kernel
            ggml_fp16_t *vDstData = cWdata + (vIn * cOh * cOw + vIoh * cOw + vIow) * (cIc * cKh * cKw); // [IC, KH, KW]
            const float *const cSrcDataF32 =
                src1->type == GGML_TYPE_F32 ? (const float *)((const char *)src1->data + vIn * vOfs0 + vIic * vOfs1)
                                            : nullptr; // [IH, IW]
            const ggml_fp16_t *const cSrcDataF16 =
                src1->type == GGML_TYPE_F16
                    ? (const ggml_fp16_t *)((const char *)src1->data + vIn * vOfs0 + vIic * vOfs1)
                    : nullptr; // [IH, IW]

            for (int64_t vIkh = 0; vIkh < cKh; vIkh++) { // 1
              for (int64_t vIkw = 0; vIkw < cKw; vIkw++) {
                const int64_t cIiw = vIow * cS0 + vIkw * cD0 - cP0;
                const int64_t cIih = vIoh * cS1 + vIkh * cD1 - cP1;

                if (cIih < 0 || cIih >= cIh || cIiw < 0 || cIiw >= cIw) {
                  vDstData[vIic * (cKh * cKw) + vIkh * cKw + vIkw] = 0;
                } else {
                  if (cSrcDataF32 != nullptr) {
                    vDstData[vIic * (cKh * cKw) + vIkh * cKw + vIkw] =
                        GGML_CPU_FP32_TO_FP16(cSrcDataF32[cIih * cIw + cIiw]);
                  } else {
                    vDstData[vIic * (cKh * cKw) + vIkh * cKw + vIkw] = cSrcDataF16[cIih * cIw + cIiw];
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}

void ggml_compute_forward_im2col(const ggml_compute_params *pParams, ggml_tensor *pDst) {
  switch (pDst->type) {
  case GGML_TYPE_F16: {
    fGgmlComputeForwardIm2colF16(pParams, pDst);
  } break;
  case GGML_TYPE_F32: {
    fGgmlComputeForwardIm2colF32(pParams, pDst);
  } break;
  default: {
    GGML_ABORT("fatal error");
  }
  }
}

// ggml_compute_forward_im2col_back_f32

void ggml_compute_forward_im2col_back_f32(const ggml_compute_params *pParams, ggml_tensor *dst) {

  const ggml_tensor *src0 = dst->src[0]; // gradients of forward pass output
  const ggml_tensor *src1 = dst->src[1]; // convolution kernel

  GGML_ASSERT(src0->type == GGML_TYPE_F32);
  GGML_ASSERT(src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16);
  GGML_ASSERT(dst->type == GGML_TYPE_F32);

  GGML_TENSOR_BINARY_OP_LOCALS;

  const int32_t cS0 = ((const int32_t *)(dst->op_params))[0];
  const int32_t cS1 = ((const int32_t *)(dst->op_params))[1];
  const int32_t cP0 = ((const int32_t *)(dst->op_params))[2];
  const int32_t cP1 = ((const int32_t *)(dst->op_params))[3];
  const int32_t cD0 = ((const int32_t *)(dst->op_params))[4];
  const int32_t cD1 = ((const int32_t *)(dst->op_params))[5];
  const bool cIs2D = ((const int32_t *)(dst->op_params))[6] == 1;

  const int cIth = pParams->ith;
  const int cNth = pParams->nth;

  const int64_t cN = cIs2D ? ne3 : ne2;
  const int64_t cIc = cIs2D ? ne2 : ne1;
  const int64_t cIh = cIs2D ? ne1 : 1;
  const int64_t cIw = ne0;

  const int64_t cKh = cIs2D ? ne11 : 1;
  const int64_t cKw = ne10;

  const int64_t cOh = cIs2D ? ne02 : 1;
  const int64_t cOw = ne01;

  int vOfs0 = cIs2D ? nb3 : nb2;
  int vOfs1 = cIs2D ? nb2 : nb1;

  GGML_ASSERT(nb0 == sizeof(float));

  // im2col: [N, IC, IH, IW] => [N, OH, OW, IC*KH*KW]
  {
    float *const cWdata = (float *)dst->data;

    for (int64_t vIn = 0; vIn < cN; vIn++) {
      for (int64_t vIic = cIth; vIic < cIc; vIic += cNth) {
        for (int64_t vIih = 0; vIih < cIh; vIih++) {
          for (int64_t vIiw = 0; vIiw < cIw; vIiw++) {

            // micro kernel
            float vGrad = 0.0f;
            for (int64_t vIkh = 0; vIkh < cKh; vIkh++) {
              for (int64_t vIkw = 0; vIkw < cKw; vIkw++) {
                // For s0 > 1 some values were skipped over in the forward pass.
                // These values have tmpw % s0 != 0 and need to be skipped in the backwards pass as well.
                const int64_t cTmpw = (vIiw + cP0 - vIkw * cD0);
                if (cTmpw % cS0 != 0) {
                  continue;
                }
                const int64_t cIow = cTmpw / cS0;

                // Equivalent logic as above except for s1.
                int64_t vIoh;
                if (cIs2D) {
                  const int64_t cTmph = vIih + cP1 - vIkh * cD1;

                  if (cTmph % cS1 != 0) {
                    continue;
                  }

                  vIoh = cTmph / cS1;
                } else {
                  vIoh = 0;
                }

                if (cIow < 0 || cIow >= cOw || vIoh < 0 || vIoh >= cOh) {
                  continue;
                }

                const float *const cGradIn = (const float *)src0->data +
                                             (vIn * cOh * cOw + vIoh * cOw + cIow) * (cIc * cKh * cKw); // [IC, KH, KW]
                vGrad += cGradIn[vIic * (cKh * cKw) + vIkh * cKw + vIkw];
              }
            }
            float *vDstData = (float *)((char *)cWdata + (vIn * vOfs0 + vIic * vOfs1)); // [IH, IW]
            vDstData[vIih * cIw + vIiw] = vGrad;
          }
        }
      }
    }
  }
}

// ggml_compute_forward_im2col_3d_f16
// src0: kernel [OC*IC, KD, KH, KW]
// src1: image [N*IC, ID, IH, IW]
// dst:  result [N*OD, OH, OW, IC * KD * KH * KW]
static void fGgmlComputeForwardIm2col3dF16(const ggml_compute_params *pParams, ggml_tensor *dst) {

  const ggml_tensor *src0 = dst->src[0];
  const ggml_tensor *src1 = dst->src[1];

  GGML_ASSERT(src1->type == GGML_TYPE_F32);
  GGML_ASSERT(dst->type == GGML_TYPE_F16);

  GGML_TENSOR_BINARY_OP_LOCALS;

  const int32_t cS0 = ((const int32_t *)(dst->op_params))[0];
  const int32_t cS1 = ((const int32_t *)(dst->op_params))[1];
  const int32_t cS2 = ((const int32_t *)(dst->op_params))[2];
  const int32_t cP0 = ((const int32_t *)(dst->op_params))[3];
  const int32_t cP1 = ((const int32_t *)(dst->op_params))[4];
  const int32_t cP2 = ((const int32_t *)(dst->op_params))[5];
  const int32_t cD0 = ((const int32_t *)(dst->op_params))[6];
  const int32_t cD1 = ((const int32_t *)(dst->op_params))[7];
  const int32_t cD2 = ((const int32_t *)(dst->op_params))[8];
  const int32_t cIc = ((const int32_t *)(dst->op_params))[9];

  const int cIth = pParams->ith;
  const int cNth = pParams->nth;

  const int64_t cN = ne13 / cIc;
  const int64_t cId = ne12;
  const int64_t cIh = ne11;
  const int64_t cIw = ne10;

  const int64_t cOc = ne03 / cIc;
  GGML_UNUSED(cOc);
  const int64_t cKd = ne02;
  const int64_t cKh = ne01;
  const int64_t cKw = ne00;

  const int64_t cOd = ne3 / cN;
  const int64_t cOh = ne2;
  const int64_t cOw = ne1;
  const int64_t cOhOw = cOh * cOw;
  const int64_t cKdKhKw = cKd * cKh * cKw;
  const int64_t cKhKw = cKh * cKw;
  const int64_t cIcKdKhKw = cIc * cKd * cKh * cKw;

  GGML_ASSERT(nb10 == sizeof(float));

  // im2col: [N*IC, ID, IH, IW] => [N*OD, OH, OW, IC * KD * KH * KW]
  {
    ggml_fp16_t *const cWdata = (ggml_fp16_t *)dst->data;

    for (int64_t vIn = 0; vIn < cN; vIn++) {
      for (int64_t vIod = 0; vIod < cOd; vIod++) {
        for (int64_t vIoh = 0; vIoh < cOh; vIoh++) {
          for (int64_t vIow = 0; vIow < cOw; vIow++) {
            for (int64_t vIic = cIth; vIic < cIc; vIic += cNth) {

              // micro kernel
              ggml_fp16_t *vDstData =
                  cWdata + (vIn * cOd * cOhOw + vIod * cOhOw + vIoh * cOw + vIow) * cIcKdKhKw; // [IC, KD, KH, KW]
              const float *const cSrcData =
                  (const float *)((const char *)src1->data + (vIn * cIc + vIic) * nb13); // [ID, IH, IW]

              for (int64_t vIkd = 0; vIkd < cKd; vIkd++) {
                for (int64_t vIkh = 0; vIkh < cKh; vIkh++) {
                  for (int64_t vIkw = 0; vIkw < cKw; vIkw++) {
                    const int64_t cIiw = vIow * cS0 + vIkw * cD0 - cP0;
                    const int64_t cIih = vIoh * cS1 + vIkh * cD1 - cP1;
                    const int64_t cIid = vIod * cS2 + vIkd * cD2 - cP2;

                    if (cIid < 0 || cIid >= cId || cIih < 0 || cIih >= cIh || cIiw < 0 || cIiw >= cIw) {
                      vDstData[vIic * cKdKhKw + vIkd * cKhKw + vIkh * cKw + vIkw] = 0;
                    } else {
                      const float *const cS = (const float *)((const char *)cSrcData + cIid * nb12 + cIih * nb11 +
                                                              cIiw * nb10); // [ID, IH, IW]
                      vDstData[vIic * cKdKhKw + vIkd * cKhKw + vIkh * cKw + vIkw] = GGML_CPU_FP32_TO_FP16(*cS);
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}

// ggml_compute_forward_im2col_3d_f32
// src0: kernel [OC*IC, KD, KH, KW]
// src1: image [N*IC, ID, IH, IW]
// dst:  result [N*OD, OH, OW, IC * KD * KH * KW]
static void fGgmlComputeForwardIm2col3dF32(const ggml_compute_params *pParams, ggml_tensor *dst) {

  const ggml_tensor *src0 = dst->src[0];
  const ggml_tensor *src1 = dst->src[1];

  GGML_ASSERT(src1->type == GGML_TYPE_F32);
  GGML_ASSERT(dst->type == GGML_TYPE_F32);

  GGML_TENSOR_BINARY_OP_LOCALS;

  const int32_t cS0 = ((const int32_t *)(dst->op_params))[0];
  const int32_t cS1 = ((const int32_t *)(dst->op_params))[1];
  const int32_t cS2 = ((const int32_t *)(dst->op_params))[2];
  const int32_t cP0 = ((const int32_t *)(dst->op_params))[3];
  const int32_t cP1 = ((const int32_t *)(dst->op_params))[4];
  const int32_t cP2 = ((const int32_t *)(dst->op_params))[5];
  const int32_t cD0 = ((const int32_t *)(dst->op_params))[6];
  const int32_t cD1 = ((const int32_t *)(dst->op_params))[7];
  const int32_t cD2 = ((const int32_t *)(dst->op_params))[8];
  const int32_t cIc = ((const int32_t *)(dst->op_params))[9];

  const int cIth = pParams->ith;
  const int cNth = pParams->nth;

  const int64_t cN = ne13 / cIc;
  const int64_t cId = ne12;
  const int64_t cIh = ne11;
  const int64_t cIw = ne10;

  const int64_t cOc = ne03 / cIc;
  GGML_UNUSED(cOc);
  const int64_t cKd = ne02;
  const int64_t cKh = ne01;
  const int64_t cKw = ne00;

  const int64_t cOd = ne3 / cN;
  const int64_t cOh = ne2;
  const int64_t cOw = ne1;

  const int64_t cOhOw = cOh * cOw;
  const int64_t cKdKhKw = cKd * cKh * cKw;
  const int64_t cKhKw = cKh * cKw;
  const int64_t cIcKdKhKw = cIc * cKd * cKh * cKw;

  GGML_ASSERT(nb10 == sizeof(float));

  // im2col: [N*IC, ID, IH, IW] => [N*OD, OH, OW, IC * KD * KH * KW]
  {
    float *const cWdata = (float *)dst->data;

    for (int64_t vIn = 0; vIn < cN; vIn++) {
      for (int64_t vIod = 0; vIod < cOd; vIod++) {
        for (int64_t vIoh = 0; vIoh < cOh; vIoh++) {
          for (int64_t vIow = 0; vIow < cOw; vIow++) {
            for (int64_t vIic = cIth; vIic < cIc; vIic += cNth) {

              // micro kernel
              float *vDstData =
                  cWdata + (vIn * cOd * cOhOw + vIod * cOhOw + vIoh * cOw + vIow) * cIcKdKhKw; // [IC, KD, KH, KW]
              const float *const cSrcData =
                  (const float *)((const char *)src1->data + (vIn * cIc + vIic) * nb13); // [ID, IH, IW]

              for (int64_t vIkd = 0; vIkd < cKd; vIkd++) {
                for (int64_t vIkh = 0; vIkh < cKh; vIkh++) {
                  for (int64_t vIkw = 0; vIkw < cKw; vIkw++) {
                    const int64_t cIiw = vIow * cS0 + vIkw * cD0 - cP0;
                    const int64_t cIih = vIoh * cS1 + vIkh * cD1 - cP1;
                    const int64_t cIid = vIod * cS2 + vIkd * cD2 - cP2;

                    if (cIid < 0 || cIid >= cId || cIih < 0 || cIih >= cIh || cIiw < 0 || cIiw >= cIw || cIid < 0 ||
                        cIid >= cId) {
                      vDstData[vIic * cKdKhKw + vIkd * cKhKw + vIkh * cKw + vIkw] = 0;
                    } else {
                      const float *const cS = (const float *)((const char *)cSrcData + cIid * nb12 + cIih * nb11 +
                                                              cIiw * nb10); // [ID, IH, IW]
                      vDstData[vIic * cKdKhKw + vIkd * cKhKw + vIkh * cKw + vIkw] = *cS;
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}

void ggml_compute_forward_im2col_3d(const ggml_compute_params *pParams, ggml_tensor *pDst) {
  switch (pDst->type) {
  case GGML_TYPE_F16: {
    fGgmlComputeForwardIm2col3dF16(pParams, pDst);
  } break;
  case GGML_TYPE_F32: {
    fGgmlComputeForwardIm2col3dF32(pParams, pDst);
  } break;
  default: {
    GGML_ABORT("fatal error");
  }
  }
}

static void fGgmlCallMulMat(ggml_type pType, const ggml_compute_params *pParams, int64_t pM, int64_t pN, int64_t pK,
                            void *pA, void *pB, float *pC) {
  const ggml_type_traits *vTraits = ggml_get_type_traits(pType);
  struct ggml_tensor vSrc1 = {};
  vSrc1.type = pType;
  vSrc1.ne[0] = pK;
  vSrc1.ne[1] = pM;
  vSrc1.ne[2] = 1;
  vSrc1.ne[3] = 1;
  vSrc1.nb[0] = vTraits->type_size;
  vSrc1.nb[1] = pK * vTraits->type_size;
  vSrc1.nb[2] = vSrc1.nb[1];
  vSrc1.nb[3] = vSrc1.nb[2];
  vSrc1.data = pA;

  struct ggml_tensor vSrc0 = {};
  vSrc0.type = pType;
  vSrc0.ne[0] = pK;
  vSrc0.ne[1] = pN;
  vSrc0.ne[2] = 1;
  vSrc0.ne[3] = 1;
  vSrc0.nb[0] = vTraits->type_size;
  vSrc0.nb[1] = pK * vTraits->type_size;
  vSrc0.nb[2] = vSrc0.nb[1];
  vSrc0.nb[3] = vSrc0.nb[2];
  vSrc0.data = pB;

  struct ggml_tensor vDst = {};
  vDst.ne[0] = pN;
  vDst.ne[1] = pM;
  vDst.ne[2] = 1;
  vDst.ne[3] = 1;
  vDst.nb[0] = sizeof(float);
  vDst.nb[1] = pN * sizeof(float);
  vDst.nb[2] = vDst.nb[1];
  vDst.nb[3] = vDst.nb[2];
  vDst.data = pC;
  vDst.src[0] = &vSrc0;
  vDst.src[1] = &vSrc1;

  ggml_compute_forward_mul_mat(pParams, &vDst);
}

static inline int64_t fGgmlWrapAround(int64_t pCoord, int64_t pSize) {
  return (pCoord + pSize) % pSize; // adding size avoids negative number weirdness
}

// ggml_compute_forward_col2im_1d
//
// Scatter-add columns [K*OC, T_in] -> signal [T_out, OC]
// where T_out = (T_in - 1)*s + K - 2*p.  Gather approach: each output reads ceil(K/s) inputs.
// Parallelized over the time axis so the split stays balanced whatever OC is.
// Supports F32, F16, BF16 input/output (same type), F32 accumulator.

template <typename elem_t>
static void fGgmlComputeForwardCol2im1dImpl(const ggml_compute_params *pParams, ggml_tensor *pDst) {

  const ggml_tensor *vSrc = pDst->src[0]; // [K*OC, T_in]

  GGML_ASSERT(ggml_is_contiguous(vSrc));
  GGML_ASSERT(ggml_is_contiguous(pDst));

  const int32_t cS0 = ((const int32_t *)(pDst->op_params))[0];
  const int32_t cOc = ((const int32_t *)(pDst->op_params))[1];
  const int32_t cP0 = ((const int32_t *)(pDst->op_params))[2];

  const int64_t cKOc = vSrc->ne[0];
  const int64_t cTIn = vSrc->ne[1];
  const int64_t cK = cKOc / cOc;
  const int64_t cTOut = pDst->ne[0];

  const elem_t *vColData = (const elem_t *)vSrc->data;
  elem_t *vDstData = (elem_t *)pDst->data;

  const int cIth = pParams->ith;
  const int cNth = pParams->nth;

  // Parallelize over the time axis: the split stays balanced whatever OC is,
  // down to OC = 1 for mono audio, and threads read disjoint column bands
  const int64_t cDr = (cTOut + cNth - 1) / cNth;
  const int64_t cIt0 = cDr * cIth;
  const int64_t cIt1 = cIt0 + cDr < cTOut ? cIt0 + cDr : cTOut;

  for (int64_t vOc = 0; vOc < cOc; vOc++) {
    for (int64_t vTOut = cIt0; vTOut < cIt1; vTOut++) {
      const int64_t cTAbs = vTOut + cP0; // absolute position in uncropped signal
      // Gather: find all (t_in, k) where t_in * s + k == t_abs, 0 <= k < K
      int64_t vTInMin = (cTAbs - cK + 1 + cS0 - 1) / cS0; // ceil((t_abs-K+1)/s)
      if (vTInMin < 0)
        vTInMin = 0;
      int64_t vTInMax = cTAbs / cS0;
      if (vTInMax >= cTIn)
        vTInMax = cTIn - 1;

      float vSum = 0.0f;
      for (int64_t vTIn = vTInMin; vTIn <= vTInMax; vTIn++) {
        int64_t vK = cTAbs - vTIn * cS0;
        if (vK >= 0 && vK < cK) {
          // col layout: [K*OC, T_in], element (oc*K+k, t_in)
          vSum += type_conversion_table<elem_t>::to_f32(vColData[(vOc * cK + vK) + vTIn * cKOc]);
        }
      }
      // dst layout: [T_out, OC], element (t_out, oc)
      vDstData[vTOut + vOc * cTOut] = type_conversion_table<elem_t>::from_f32(vSum);
    }
  }
}

void ggml_compute_forward_col2im_1d(const ggml_compute_params *pParams, ggml_tensor *pDst) {
  switch (pDst->src[0]->type) {
  case GGML_TYPE_F32:
    fGgmlComputeForwardCol2im1dImpl<float>(pParams, pDst);
    break;
  case GGML_TYPE_F16:
    fGgmlComputeForwardCol2im1dImpl<ggml_fp16_t>(pParams, pDst);
    break;
  case GGML_TYPE_BF16:
    fGgmlComputeForwardCol2im1dImpl<ggml_bf16_t>(pParams, pDst);
    break;
  default:
    GGML_ABORT("col2im_1d: unsupported type %d", pDst->src[0]->type);
  }
}

// ggml_compute_forward_conv_2d

static void fGgmlComputeForwardConv2dImpl(const ggml_compute_params *pParams,
                                          const ggml_tensor *pKernel, // [KW, KH, IC, OC]
                                          const ggml_tensor *pSrc,    // [W, H, C, N]
                                          ggml_tensor *pDst,          // [OW, OH, OC, N]
                                          ggml_type pKernelType) {

  GGML_ASSERT(ggml_is_contiguous(pKernel));
  GGML_ASSERT(pKernelType == GGML_TYPE_F16 || pKernelType == GGML_TYPE_F32);
  GGML_ASSERT(pKernel->type == pKernelType);

  const ggml_type_traits *vTraits = ggml_get_type_traits(pKernelType);

  const int32_t cStrideX = pDst->op_params[0];
  const int32_t cStrideY = pDst->op_params[1];
  const int32_t cPadX = pDst->op_params[2];
  const int32_t cPadY = pDst->op_params[3];
  const int32_t cDilationX = pDst->op_params[4];
  const int32_t cDilationY = pDst->op_params[5];

  const int64_t cCIn = pSrc->ne[2];
  const int64_t cCOut = pKernel->ne[3];
  GGML_ASSERT(cCIn == pKernel->ne[2]);

  const int64_t cSrcW = pSrc->ne[0];
  const int64_t cSrcH = pSrc->ne[1];
  const int64_t cKnlW = pKernel->ne[0];
  const int64_t cKnlH = pKernel->ne[1];
  const int64_t cDstW = pDst->ne[0];
  const int64_t cDstH = pDst->ne[1];

  const float *vSrcData = (float *)pSrc->data;
  void *vKnlData = pKernel->data;
  float *vDstData = (float *)pDst->data;

  const int64_t cKnlN = cKnlW * cKnlH * cCIn;
  const int64_t cPatchTotal = pDst->ne[3] * cDstW * cDstH;

  const int64_t cSpacePerPatch = cKnlN * vTraits->type_size + cCOut * sizeof(float);
  const int64_t cBatchSize = pParams->wsize / cSpacePerPatch;
  const int64_t cPatchesPerBatch = cBatchSize > 8 ? (cBatchSize / 8) * 8 : cBatchSize;
  const int64_t cBatchN = (cPatchTotal + cPatchesPerBatch - 1) / cPatchesPerBatch;

  GGML_ASSERT(cPatchesPerBatch > 0 && cBatchSize >= 1);

  void *vTmp = pParams->wdata;

  for (int64_t vBatchI = 0; vBatchI < cBatchN; ++vBatchI) {

    const int64_t cPatchStartBatch = vBatchI * cPatchesPerBatch;
    const int64_t cPatchEndBatch = std::min(cPatchStartBatch + cPatchesPerBatch, cPatchTotal);
    const int64_t cPatchN = cPatchEndBatch - cPatchStartBatch;

    const int64_t cPatchPerThread = (cPatchN + pParams->nth - 1) / pParams->nth;
    const int64_t cPatchStart = cPatchStartBatch + pParams->ith * cPatchPerThread;
    const int64_t cPatchEnd = std::min(cPatchStart + cPatchPerThread, cPatchEndBatch);

    // im2col for a patch
    for (int64_t vP = cPatchStart; vP < cPatchEnd; ++vP) {
      const int64_t cBatchN = vP / (cDstW * cDstH);
      const int64_t cSrcX = (vP / cDstW) % cDstH;
      const int64_t cSrcY = vP % cDstW;

      const float *vSrcBase = (const float *)((const char *)vSrcData + cBatchN * pSrc->nb[3]);
      char *vDstRow = (char *)vTmp + (vP % cPatchesPerBatch) * cKnlN * vTraits->type_size;

      for (int64_t vIc = 0; vIc < cCIn; ++vIc) {
        for (int64_t vKy = 0; vKy < cKnlH; ++vKy) {
          for (int64_t vKx = 0; vKx < cKnlW; ++vKx) {
            const int64_t cSy = cSrcX * cStrideY + vKy * cDilationY - cPadY;
            const int64_t cSx = cSrcY * cStrideX + vKx * cDilationX - cPadX;

            int64_t vDstIdx = vIc * (cKnlH * cKnlW) + vKy * cKnlW + vKx;

            float vSrcVal;
            if (cSy < 0 || cSy >= cSrcH || cSx < 0 || cSx >= cSrcW) {
              vSrcVal = 0.0f;
            } else {
              const float *vSrcPtr =
                  (const float *)((const char *)vSrcBase + cSx * pSrc->nb[0] + cSy * pSrc->nb[1] + vIc * pSrc->nb[2]);
              vSrcVal = *vSrcPtr;
            }

            char *vElementPtr = vDstRow + vDstIdx * vTraits->type_size;
            if (pKernelType == GGML_TYPE_F32) {
              *(float *)vElementPtr = vSrcVal;
            } else if (pKernelType == GGML_TYPE_F16) {
              *(ggml_fp16_t *)vElementPtr = GGML_CPU_FP32_TO_FP16(vSrcVal);
            }
          }
        }
      }
    } // patches handled by this thread

    ggml_barrier(pParams->threadpool);

    float *vGemmOutput = (float *)((char *)vTmp + cPatchesPerBatch * cKnlN * vTraits->type_size);

    GGML_ASSERT(vGemmOutput + cPatchN * cCOut <= (float *)vTmp + pParams->wsize);

    // GEMM: patches[patch_n, knl_n] × kernel[knl_n, c_out] = output[patch_n, c_out]
    fGgmlCallMulMat(pKernelType, pParams, cPatchN, cCOut, cKnlN, vTmp, vKnlData, vGemmOutput);

    ggml_barrier(pParams->threadpool);

    // permute back [OC, N, OH, OW] to [N, OC, OH, OW]
    const int64_t cPermutePerThread = (cPatchN + pParams->nth - 1) / pParams->nth;
    const int64_t cPermuteStart = pParams->ith * cPermutePerThread;
    const int64_t cPermuteEnd = std::min(cPermuteStart + cPermutePerThread, cPatchN);

    for (int64_t vI = cPermuteStart; vI < cPermuteEnd; ++vI) {
      const int64_t cP = cPatchStartBatch + vI;
      const int64_t cBatchN = cP / (cDstW * cDstH);
      const int64_t cDstY = (cP / cDstW) % cDstH;
      const int64_t cDstX = cP % cDstW;

      for (int64_t vOc = 0; vOc < cCOut; ++vOc) {
        const float cValue = vGemmOutput[vI * cCOut + vOc];
        float *vDstPtr = (float *)((char *)vDstData + cDstX * pDst->nb[0] + cDstY * pDst->nb[1] + vOc * pDst->nb[2] +
                                   cBatchN * pDst->nb[3]);
        *vDstPtr = cValue;
      }
    }
  }
}

void ggml_compute_forward_conv_2d(const ggml_compute_params *pParams, ggml_tensor *pDst) {

  const ggml_tensor *vSrc0 = pDst->src[0];
  const ggml_tensor *vSrc1 = pDst->src[1];

  fGgmlComputeForwardConv2dImpl(pParams, vSrc0, vSrc1, pDst, vSrc0->type);
}

// ggml_compute_forward_conv_3d

static void fGgmlComputeForwardConv3dImpl(const ggml_compute_params *pParams, const ggml_tensor *pKernel,
                                          const ggml_tensor *pSrc, ggml_tensor *pDst, ggml_type pKernelType) {

  GGML_ASSERT(ggml_is_contiguous(pKernel));
  GGML_ASSERT(pKernelType == GGML_TYPE_F16 || pKernelType == GGML_TYPE_F32);
  GGML_ASSERT(pKernel->type == pKernelType);

  const ggml_type_traits *vTraits = ggml_get_type_traits(pKernelType);

  const int32_t cS0 = pDst->op_params[0];
  const int32_t cS1 = pDst->op_params[1];
  const int32_t cS2 = pDst->op_params[2];
  const int32_t cP0 = pDst->op_params[3];
  const int32_t cP1 = pDst->op_params[4];
  const int32_t cP2 = pDst->op_params[5];
  const int32_t cD0 = pDst->op_params[6];
  const int32_t cD1 = pDst->op_params[7];
  const int32_t cD2 = pDst->op_params[8];
  const int32_t c_ = pDst->op_params[9];
  const int32_t cN = pDst->op_params[10];
  const int32_t cOc = pDst->op_params[11];

  const int64_t cSrcW = pSrc->ne[0];
  const int64_t cSrcH = pSrc->ne[1];
  const int64_t cSrcD = pSrc->ne[2];
  const int64_t cKnlW = pKernel->ne[0];
  const int64_t cKnlH = pKernel->ne[1];
  const int64_t cKnlD = pKernel->ne[2];
  const int64_t cDstW = pDst->ne[0];
  const int64_t cDstH = pDst->ne[1];
  const int64_t cDstD = pDst->ne[2];

  const float *vSrcData = (float *)pSrc->data;
  void *vKnlData = pKernel->data;
  float *vDstData = (float *)pDst->data;

  const int64_t cKnlNPerChannel = cKnlW * cKnlH * cKnlD;
  const int64_t cKnlNTotal = cKnlNPerChannel * c_;
  const int64_t cPatchTotal = cN * cDstW * cDstH * cDstD;

  const int64_t cSpacePerPatch = cKnlNTotal * vTraits->type_size + cOc * sizeof(float);
  const int64_t cBatchSize = pParams->wsize / cSpacePerPatch;
  const int64_t cPatchesPerBatch = cBatchSize > 8 ? (cBatchSize / 8) * 8 : cBatchSize;
  const int64_t cBatchN = (cPatchTotal + cPatchesPerBatch - 1) / cPatchesPerBatch;

  GGML_ASSERT(cPatchesPerBatch > 0 && cBatchSize >= 1);

  void *vTmp = pParams->wdata;

  for (int64_t vBatchI = 0; vBatchI < cBatchN; ++vBatchI) {
    const int64_t cPatchStartBatch = vBatchI * cPatchesPerBatch;
    const int64_t cPatchEndBatch = std::min(cPatchStartBatch + cPatchesPerBatch, cPatchTotal);
    const int64_t cPatchNInBatch = cPatchEndBatch - cPatchStartBatch;

    const int64_t cPatchPerThread = (cPatchNInBatch + pParams->nth - 1) / pParams->nth;
    const int64_t cPatchStart = cPatchStartBatch + pParams->ith * cPatchPerThread;
    const int64_t cPatchEnd = std::min(cPatchStart + cPatchPerThread, cPatchEndBatch);

    for (int64_t vP = cPatchStart; vP < cPatchEnd; ++vP) {
      const int64_t cPInBatch = vP % (cDstW * cDstH * cDstD);
      const int64_t cPInDepth = cPInBatch % (cDstW * cDstH);
      const int64_t cBatchIdx = vP / (cDstW * cDstH * cDstD);
      const int64_t cDstZ = cPInBatch / (cDstW * cDstH);
      const int64_t cDstY = cPInDepth / cDstW;
      const int64_t cDstX = cPInDepth % cDstW;

      char *vDstRow = (char *)vTmp + (vP % cPatchesPerBatch) * cKnlNTotal * vTraits->type_size;

      for (int64_t vIc = 0; vIc < c_; ++vIc) {
        for (int64_t vKz = 0; vKz < cKnlD; ++vKz) {
          for (int64_t vKy = 0; vKy < cKnlH; ++vKy) {
            for (int64_t vKx = 0; vKx < cKnlW; ++vKx) {
              const int64_t cSz = cDstZ * cS2 + vKz * cD2 - cP2;
              const int64_t cSy = cDstY * cS1 + vKy * cD1 - cP1;
              const int64_t cSx = cDstX * cS0 + vKx * cD0 - cP0;

              int64_t vDstIdx = vIc * cKnlNPerChannel + vKz * (cKnlH * cKnlW) + vKy * cKnlW + vKx;

              float vSrcVal;
              if (cSz < 0 || cSz >= cSrcD || cSy < 0 || cSy >= cSrcH || cSx < 0 || cSx >= cSrcW) {
                vSrcVal = 0.0f;
              } else {
                const int64_t cCnIdx = cBatchIdx * c_ + vIc;
                const float *vSrcPtr = (const float *)((const char *)vSrcData + cSx * pSrc->nb[0] + cSy * pSrc->nb[1] +
                                                       cSz * pSrc->nb[2] + cCnIdx * pSrc->nb[3]);
                vSrcVal = *vSrcPtr;
              }

              char *vElementPtr = vDstRow + vDstIdx * vTraits->type_size;
              if (pKernelType == GGML_TYPE_F32) {
                *(float *)vElementPtr = vSrcVal;
              } else if (pKernelType == GGML_TYPE_F16) {
                *(ggml_fp16_t *)vElementPtr = GGML_CPU_FP32_TO_FP16(vSrcVal);
              }
            }
          }
        }
      }
    }

    ggml_barrier(pParams->threadpool);

    float *vGemmOutput = (float *)((char *)vTmp + cPatchesPerBatch * cKnlNTotal * vTraits->type_size);
    fGgmlCallMulMat(pKernelType, pParams, cPatchNInBatch, cOc, cKnlNTotal, vTmp, vKnlData, vGemmOutput);

    ggml_barrier(pParams->threadpool);

    const int64_t cPermutePerThread = (cPatchNInBatch + pParams->nth - 1) / pParams->nth;
    const int64_t cPermuteStart = pParams->ith * cPermutePerThread;
    const int64_t cPermuteEnd = std::min(cPermuteStart + cPermutePerThread, cPatchNInBatch);

    for (int64_t vI = cPermuteStart; vI < cPermuteEnd; ++vI) {
      const int64_t cP = cPatchStartBatch + vI;
      const int64_t cPInBatch = cP % (cDstW * cDstH * cDstD);
      const int64_t cPInDepth = cPInBatch % (cDstW * cDstH);
      const int64_t cBatchIdx = cP / (cDstW * cDstH * cDstD);
      const int64_t cDstZ = cPInBatch / (cDstW * cDstH);
      const int64_t cDstY = cPInDepth / cDstW;
      const int64_t cDstX = cPInDepth % cDstW;

      for (int64_t vIoc = 0; vIoc < cOc; ++vIoc) {
        const float cValue = vGemmOutput[vI * cOc + vIoc];
        const int64_t cOcnIdx = cBatchIdx * cOc + vIoc;
        float *vDstPtr = (float *)((char *)vDstData + cDstX * pDst->nb[0] + cDstY * pDst->nb[1] + cDstZ * pDst->nb[2] +
                                   cOcnIdx * pDst->nb[3]);
        *vDstPtr = cValue;
      }
    }
  }
}

void ggml_compute_forward_conv_3d(const ggml_compute_params *pParams, ggml_tensor *pDst) {
  const ggml_tensor *vSrc0 = pDst->src[0];
  const ggml_tensor *vSrc1 = pDst->src[1];
  fGgmlComputeForwardConv3dImpl(pParams, vSrc0, vSrc1, pDst, vSrc0->type);
}

template <typename kernel_t>
static void fGgmlComputeForwardConvTranspose2dImpl(const ggml_compute_params *pParams, ggml_tensor *dst) {

  const ggml_tensor *src0 = dst->src[0];
  const ggml_tensor *src1 = dst->src[1];

  GGML_ASSERT(src0->type == GGML_TYPE_F16 || src0->type == GGML_TYPE_F32);
  GGML_ASSERT(src1->type == GGML_TYPE_F32);
  GGML_ASSERT(dst->type == GGML_TYPE_F32);

  GGML_TENSOR_BINARY_OP_LOCALS

  const int cIth = pParams->ith;
  const int cNth = pParams->nth;

  const int cNk = ne00 * ne01 * ne02 * ne03;

  GGML_ASSERT(nb00 == ggml_type_size(src0->type));
  GGML_ASSERT(nb10 == sizeof(float));

  if (cIth == 0) {
    memset(pParams->wdata, 0, pParams->wsize);

    // permute kernel data (src0) from (Kw x Kh x Cout x Cin) to (Cin x Kw x Kh x Cout)
    {
      kernel_t *const cWdata = (kernel_t *)pParams->wdata + 0;

      for (int64_t vI03 = 0; vI03 < ne03; vI03++) {
        for (int64_t vI02 = 0; vI02 < ne02; vI02++) {
          const kernel_t *const cSrc = (kernel_t *)((char *)src0->data + vI03 * nb03 + vI02 * nb02);
          kernel_t *vDstData = cWdata + vI02 * ne01 * ne00 * ne03;
          for (int64_t vI01 = 0; vI01 < ne01; vI01++) {
            for (int64_t vI00 = 0; vI00 < ne00; vI00++) {
              vDstData[vI01 * ne00 * ne03 + vI00 * ne03 + vI03] = cSrc[vI01 * ne00 + vI00];
            }
          }
        }
      }
    }

    // permute source data (src1) from (Sw x Sh x Cin) to (Cin x Sw x Sh)
    {
      kernel_t *const cWdata = (kernel_t *)pParams->wdata + cNk;
      for (int vI12 = 0; vI12 < ne12; vI12++) {
        for (int vI11 = 0; vI11 < ne11; vI11++) {
          const float *const cSrc = (float *)((char *)src1->data + vI12 * nb12 + vI11 * nb11);
          kernel_t *vDstData = cWdata + vI11 * ne10 * ne12;
          for (int vI10 = 0; vI10 < ne10; vI10++) {
            if constexpr (std::is_same_v<kernel_t, ggml_fp16_t>) {
              vDstData[vI10 * ne12 + vI12] = GGML_CPU_FP32_TO_FP16(cSrc[vI10]);
            } else {
              vDstData[vI10 * ne12 + vI12] = cSrc[vI10];
            }
          }
        }
      }
    }

    memset(dst->data, 0, ggml_nbytes(dst));
  }
  ggml_barrier(pParams->threadpool);

  const int32_t cStride = ggml_get_op_params_i32(dst, 0);

  // total patches in dst
  const int cNp = ne2;

  // patches per thread
  const int cDp = (cNp + cNth - 1) / cNth;

  // patch range for this thread
  const int cIp0 = cDp * cIth;
  const int cIp1 = MIN(cIp0 + cDp, cNp);

  kernel_t *const cWdata = (kernel_t *)pParams->wdata + 0;
  kernel_t *const cWdataSrc = cWdata + cNk;

  for (int vI2 = cIp0; vI2 < cIp1; vI2++) { // Cout
    float *vDstData = (float *)((char *)dst->data + vI2 * nb2);
    kernel_t *vWdataKernel = cWdata + vI2 * ne01 * ne00 * ne03;
    for (int vI11 = 0; vI11 < ne11; vI11++) {
      for (int vI10 = 0; vI10 < ne10; vI10++) {
        const int cI1n = vI11 * ne10 * ne12 + vI10 * ne12;
        for (int vI01 = 0; vI01 < ne01; vI01++) {
          for (int vI00 = 0; vI00 < ne00; vI00++) {
            float v_ = 0;
            if constexpr (std::is_same_v<kernel_t, ggml_fp16_t>) {
              ggml_vec_dot_f16(ne03, &v_, 0, cWdataSrc + cI1n, 0, vWdataKernel + vI01 * ne00 * ne03 + vI00 * ne03, 0,
                               1);
            } else {
              ggml_vec_dot_f32(ne03, &v_, 0, cWdataSrc + cI1n, 0, vWdataKernel + vI01 * ne00 * ne03 + vI00 * ne03, 0,
                               1);
            }
            vDstData[(vI11 * cStride + vI01) * ne0 + vI10 * cStride + vI00] += v_;
          }
        }
      }
    }
  }
}

void ggml_compute_forward_conv_transpose_2d(const ggml_compute_params *pParams, ggml_tensor *pDst) {

  const ggml_tensor *vSrc0 = pDst->src[0];

  switch (vSrc0->type) {
  case GGML_TYPE_F16: {
    fGgmlComputeForwardConvTranspose2dImpl<ggml_fp16_t>(pParams, pDst);
  } break;
  case GGML_TYPE_F32: {
    fGgmlComputeForwardConvTranspose2dImpl<float>(pParams, pDst);
  } break;
  default: {
    GGML_ABORT("fatal error");
  }
  }
}

// ggml_compute_forward_conv_2d_dw

struct ggml_conv_2d_dw_params {
  int64_t vChannels;
  int64_t vBatch;
  int64_t vSrcW;
  int64_t vSrcH;
  int64_t vDstW;
  int64_t vDstH;
  int64_t vKnlW;
  int64_t vKnlH;
  int vStrideX;
  int vStrideY;
  int vPadX;
  int vPadY;
  int vDilationX;
  int vDilationY;
};

static inline float fGgmlConv2dDwKnlF32(const char *pData, int64_t pI, ggml_type pType) {
  if (pType == GGML_TYPE_F16) {
    return GGML_FP16_TO_FP32(((const ggml_fp16_t *)pData)[pI]);
  }
  return ((const float *)pData)[pI];
}

static void fGgmlComputeForwardConv2dDwCwhn(const ggml_compute_params *pParams, const ggml_tensor *pSrc,
                                            const ggml_tensor *pKernel, ggml_tensor *pDst,
                                            const ggml_conv_2d_dw_params &p_) {

  const int64_t c_ = p_.vChannels;
  const char *vKnlData = (const char *)pKernel->data;
  const ggml_type cKnlType = pKernel->type;

  const int64_t cRowsTotal = p_.vDstH * p_.vBatch;
  const int64_t cRowsPerThread = (cRowsTotal + pParams->nth - 1) / pParams->nth;
  const int64_t cRowStart = pParams->ith * cRowsPerThread;
  const int64_t cRowEnd = MIN(cRowStart + cRowsPerThread, cRowsTotal);

#ifdef GGML_SIMD
  int64_t vCPkgEnd = 0;
  int64_t vPkgSize = GGML_F32_EPR;
  if (cKnlType == GGML_TYPE_F32) {
#if defined(__ARM_FEATURE_SVE)
    vPkgSize = svcntw();
#else
    vPkgSize = GGML_F32_EPR;
#endif
    vCPkgEnd = (c_ / vPkgSize) * vPkgSize;
  }
#else
  const int64_t vCPkgEnd = 0;
#endif

  for (int64_t vRow = cRowStart; vRow < cRowEnd; ++vRow) {
    const int64_t cDstY = vRow % p_.vDstH;
    const float *vSrcData = (const float *)pSrc->data + (vRow / p_.vDstH) * p_.vSrcW * p_.vSrcH * c_;
    for (int64_t vDstX = 0; vDstX < p_.vDstW; ++vDstX) {
      float *vDstData = (float *)pDst->data + (vRow * p_.vDstW + vDstX) * c_;
      const int64_t cSrcYBase = cDstY * p_.vStrideY - p_.vPadY;
      const int64_t cSrcXBase = vDstX * p_.vStrideX - p_.vPadX;

#ifdef GGML_SIMD
      for (int64_t vCI = 0; vCI < vCPkgEnd; vCI += vPkgSize) {
        GGML_F32_VEC vSum = GGML_F32_VEC_ZERO;
        for (int64_t vKnlY = 0; vKnlY < p_.vKnlH; ++vKnlY) {
          const int64_t cSrcY = cSrcYBase + vKnlY * p_.vDilationY;
          if (cSrcY < 0 || cSrcY >= p_.vSrcH) {
            continue;
          }
          for (int64_t vKnlX = 0; vKnlX < p_.vKnlW; ++vKnlX) {
            const int64_t cSrcX = cSrcXBase + vKnlX * p_.vDilationX;
            if (cSrcX < 0 || cSrcX >= p_.vSrcW) {
              continue;
            }
            const float *vKp = (const float *)vKnlData + (vKnlY * p_.vKnlW + vKnlX) * c_ + vCI;
            GGML_F32_VEC vK = GGML_F32_VEC_LOAD(vKp);
            GGML_F32_VEC vS = GGML_F32_VEC_LOAD(vSrcData + (cSrcY * p_.vSrcW + cSrcX) * c_ + vCI);
            vSum = GGML_F32_VEC_FMA(vSum, vK, vS);
          }
        }
        GGML_F32_VEC_STORE(vDstData + vCI, vSum);
      }
#endif
      for (int64_t vCI = vCPkgEnd; vCI < c_; ++vCI) {
        float vSum = 0.0f;
        for (int64_t vKnlY = 0; vKnlY < p_.vKnlH; ++vKnlY) {
          const int64_t cSrcY = cSrcYBase + vKnlY * p_.vDilationY;
          if (cSrcY < 0 || cSrcY >= p_.vSrcH) {
            continue;
          }
          for (int64_t vKnlX = 0; vKnlX < p_.vKnlW; ++vKnlX) {
            const int64_t cSrcX = cSrcXBase + vKnlX * p_.vDilationX;
            if (cSrcX < 0 || cSrcX >= p_.vSrcW) {
              continue;
            }
            vSum += fGgmlConv2dDwKnlF32(vKnlData, (vKnlY * p_.vKnlW + vKnlX) * c_ + vCI, cKnlType) *
                    vSrcData[(cSrcY * p_.vSrcW + cSrcX) * c_ + vCI];
          }
        }
        vDstData[vCI] = vSum;
      }
    }
  }
}

static void fGgmlComputeForwardConv2dDwWhcn(const ggml_compute_params *pParams, const ggml_tensor *pSrc,
                                            const ggml_tensor *pKernel, ggml_tensor *pDst,
                                            const ggml_conv_2d_dw_params &p_) {

  const int64_t cN = p_.vChannels * p_.vBatch;
  const int64_t cPerThread = (cN + pParams->nth - 1) / pParams->nth;
  const int64_t cStart = pParams->ith * cPerThread;
  const int64_t cEnd = MIN(cStart + cPerThread, cN);
  const char *vKnlBase = (const char *)pKernel->data;
  const ggml_type cKnlType = pKernel->type;

  for (int64_t vI = cStart; vI < cEnd; ++vI) {
    const int64_t cKnlOffset = (vI % p_.vChannels) * p_.vKnlW * p_.vKnlH;
    const float *vSrcData = (const float *)pSrc->data + vI * p_.vSrcW * p_.vSrcH;
    float *vDstData = (float *)pDst->data + vI * p_.vDstW * p_.vDstH;

    for (int64_t vDstY = 0; vDstY < p_.vDstH; ++vDstY) {
      for (int64_t vDstX = 0; vDstX < p_.vDstW; ++vDstX) {

        float vSum = 0.0f;
        for (int64_t vKnlY = 0; vKnlY < p_.vKnlH; ++vKnlY) {
          const int64_t cSrcY = vDstY * p_.vStrideY + vKnlY * p_.vDilationY - p_.vPadY;
          if (cSrcY < 0 || cSrcY >= p_.vSrcH) {
            continue;
          }
          for (int64_t vKnlX = 0; vKnlX < p_.vKnlW; ++vKnlX) {
            const int64_t cSrcX = vDstX * p_.vStrideX + vKnlX * p_.vDilationX - p_.vPadX;
            if (cSrcX < 0 || cSrcX >= p_.vSrcW) {
              continue;
            }
            vSum += fGgmlConv2dDwKnlF32(vKnlBase, cKnlOffset + vKnlY * p_.vKnlW + vKnlX, cKnlType) *
                    vSrcData[cSrcY * p_.vSrcW + cSrcX];
          }
        }
        vDstData[vDstY * p_.vDstW + vDstX] = vSum;
      }
    }
  }
}

void ggml_compute_forward_conv_2d_dw(const ggml_compute_params *pParams, ggml_tensor *pDst) {

  const ggml_tensor *vKernel = pDst->src[0];
  const ggml_tensor *vSrc = pDst->src[1];
  ggml_conv_2d_dw_params vP;
  vP.vChannels = vSrc->ne[2];
  vP.vBatch = vSrc->ne[3];
  vP.vSrcW = vSrc->ne[0];
  vP.vSrcH = vSrc->ne[1];
  vP.vDstW = pDst->ne[0];
  vP.vDstH = pDst->ne[1];
  vP.vKnlW = vKernel->ne[0];
  vP.vKnlH = vKernel->ne[1];
  vP.vStrideX = pDst->op_params[0];
  vP.vStrideY = pDst->op_params[1];
  vP.vPadX = pDst->op_params[2];
  vP.vPadY = pDst->op_params[3];
  vP.vDilationX = pDst->op_params[4];
  vP.vDilationY = pDst->op_params[5];

  GGML_ASSERT(vKernel->type == GGML_TYPE_F32 || vKernel->type == GGML_TYPE_F16);
  GGML_ASSERT(vKernel->ne[3] == vP.vChannels);
  GGML_ASSERT(pDst->ne[3] == vP.vBatch);

  if (ggml_is_contiguous(vSrc)) {
    fGgmlComputeForwardConv2dDwWhcn(pParams, vSrc, vKernel, pDst, vP);
  } else if (ggml_is_contiguous_channels(vSrc)) {
    GGML_ASSERT(vKernel->nb[0] >= vKernel->nb[2] && vKernel->nb[1] >= vKernel->nb[0]);
    fGgmlComputeForwardConv2dDwCwhn(pParams, vSrc, vKernel, pDst, vP);
  } else {
    GGML_ABORT("non-contiguous memory layout not supported");
  }
}

// ggml_compute_forward_pool_1d_ksp
static void fGgmlComputeForwardPool1dKsp(const ggml_compute_params *pParams, const ggml_op_pool pOp, const int pK,
                                         const int pS, const int p_, ggml_tensor *pDst) {

  const ggml_tensor *vSrc = pDst->src[0];

  assert(vSrc->type == GGML_TYPE_F32 || vSrc->type == GGML_TYPE_F16);

  if (pParams->ith != 0) {
    return;
  }

  const int64_t cIw = vSrc->ne[0];
  const int64_t cOw = pDst->ne[0];

  const int64_t cNr = ggml_nrows(vSrc);

  for (int64_t vIr = 0; vIr < cNr; ++vIr) {
    const char *vSrowBytes = (const char *)vSrc->data + vIr * vSrc->nb[1];
    float *vDrow = (float *)((char *)pDst->data + vIr * pDst->nb[1]);

    for (int64_t vOw = 0; vOw < cOw; ++vOw) {
      float vRes = 0;
      switch (pOp) {
      case GGML_OP_POOL_AVG:
        vRes = 0.0f;
        break;
      case GGML_OP_POOL_MAX:
        vRes = -FLT_MAX;
        break;
      case GGML_OP_POOL_COUNT:
        GGML_ABORT("fatal error");
      }

      int vCount = 0;
      const int cBase = (int)vOw * pS - p_;

      for (int vKi = 0; vKi < pK; ++vKi) {
        const int cJ = cBase + vKi;
        if (cJ < 0 || cJ >= (int)cIw) {
          continue;
        }

        float v_;
        if (vSrc->type == GGML_TYPE_F32) {
          v_ = ((const float *)vSrowBytes)[cJ];
        } else {
          v_ = GGML_CPU_FP16_TO_FP32(((const ggml_fp16_t *)vSrowBytes)[cJ]);
        }

        switch (pOp) {
        case GGML_OP_POOL_AVG:
          vRes += v_;
          break;
        case GGML_OP_POOL_MAX:
          vRes = std::max(v_, vRes);
          break;
        case GGML_OP_POOL_COUNT:
          GGML_ABORT("fatal error");
        }

        ++vCount;
      }

      switch (pOp) {
      case GGML_OP_POOL_AVG:
        vRes = (vCount > 0) ? (vRes / vCount) : 0.0f;
        break;
      case GGML_OP_POOL_MAX:
        break;
      case GGML_OP_POOL_COUNT:
        GGML_ABORT("fatal error");
      }

      vDrow[vOw] = vRes;
    }
  }
}

// ggml_compute_forward_pool_1d

void ggml_compute_forward_pool_1d(const ggml_compute_params *pParams, ggml_tensor *pDst) {

  const int32_t *vOpts = (const int32_t *)pDst->op_params;
  ggml_op_pool vOp = static_cast<ggml_op_pool>(vOpts[0]);
  const int cK0 = vOpts[1];
  const int cS0 = vOpts[2];
  const int cP0 = vOpts[3];

  fGgmlComputeForwardPool1dKsp(pParams, vOp, cK0, cS0, cP0, pDst);
}

// ggml_compute_forward_pool_2d

void ggml_compute_forward_pool_2d(const ggml_compute_params *pParams, ggml_tensor *pDst) {

  const ggml_tensor *vSrc = pDst->src[0];

  assert(vSrc->type == GGML_TYPE_F32 || vSrc->type == GGML_TYPE_F16);

  if (pParams->ith != 0) {
    return;
  }

  const int32_t *vOpts = (const int32_t *)pDst->op_params;

  ggml_op_pool vOp = static_cast<ggml_op_pool>(vOpts[0]);
  const int cK0 = vOpts[1];
  const int cK1 = vOpts[2];
  const int cS0 = vOpts[3];
  const int cS1 = vOpts[4];
  const int cP0 = vOpts[5];
  const int cP1 = vOpts[6];
  const char *vCdata = (const char *)vSrc->data;
  const char *const cDataEnd = vCdata + ggml_nbytes(vSrc);

  const int64_t cPx = pDst->ne[0];
  const int64_t cPy = pDst->ne[1];
  const int64_t cPa = cPx * cPy;

  float *vDplane = (float *)pDst->data;

  const int cKa = cK0 * cK1;
  const int cOffset0 = -cP0;
  const int cOffset1 = -cP1;

  while (vCdata < cDataEnd) {
    for (int vOy = 0; vOy < cPy; ++vOy) {
      float *const cDrow = vDplane + vOy * cPx;
      float *const cOut = cDrow;

      for (int vOx = 0; vOx < cPx; ++vOx) {
        float vRes = 0;
        switch (vOp) {
        case GGML_OP_POOL_AVG:
          vRes = 0;
          break;
        case GGML_OP_POOL_MAX:
          vRes = -FLT_MAX;
          break;
        case GGML_OP_POOL_COUNT:
          GGML_ABORT("fatal error");
        }

        const int cIx = cOffset0 + vOx * cS0;
        const int cIy = cOffset1 + vOy * cS1;

        for (int vKy = 0; vKy < cK1; ++vKy) {
          if (cIy + vKy < 0 || cIy + vKy >= vSrc->ne[1]) {
            continue;
          }

          const void *vSrow = (const void *)(vCdata + vSrc->nb[1] * (cIy + vKy));
          for (int vKx = 0; vKx < cK0; ++vKx) {
            int vJ = cIx + vKx;
            if (vJ < 0 || vJ >= vSrc->ne[0]) {
              continue;
            }

            const float cSrowJ = (vSrc->type == GGML_TYPE_F32)
                                     ? ((const float *)vSrow)[vJ]
                                     : GGML_CPU_FP16_TO_FP32(((const ggml_fp16_t *)vSrow)[vJ]);
            switch (vOp) {
            case GGML_OP_POOL_AVG:
              vRes += cSrowJ;
              break;
            case GGML_OP_POOL_MAX:
              vRes = std::max(cSrowJ, vRes);
              break;
            case GGML_OP_POOL_COUNT:
              GGML_ABORT("fatal error");
            }
          }
        }
        switch (vOp) {
        case GGML_OP_POOL_AVG:
          vRes /= cKa;
          break;
        case GGML_OP_POOL_MAX:
          break;
        case GGML_OP_POOL_COUNT:
          GGML_ABORT("fatal error");
        }

        cOut[vOx] = vRes;
      }
    }

    vCdata += vSrc->nb[2];
    vDplane += cPa;
  }
}

// ggml_compute_forward_pool_2d_back

void ggml_compute_forward_pool_2d_back(const ggml_compute_params *pParams, ggml_tensor *pDst) {

  const ggml_tensor *vSrc = pDst->src[0];
  const ggml_tensor *vDstf = pDst->src[1]; // forward tensor of dst

  assert(pDst->type == GGML_TYPE_F32 || pDst->type == GGML_TYPE_F16);

  if (pParams->ith != 0) {
    return;
  }

  const int32_t *vOpts = (const int32_t *)pDst->op_params;
  ggml_op_pool vOp = static_cast<ggml_op_pool>(vOpts[0]);
  const int cK0 = vOpts[1];
  const int cK1 = vOpts[2];
  const int cS0 = vOpts[3];
  const int cS1 = vOpts[4];
  const int cP0 = vOpts[5];
  const int cP1 = vOpts[6];

  char *vCdata = (char *)pDst->data;
  const char *vCdataf = (const char *)vDstf->data;
  const char *const cDataEnd = vCdata + ggml_nbytes(pDst);

  GGML_ASSERT(pParams->ith == 0);
  memset(vCdata, 0, ggml_nbytes(pDst));

  const int64_t cPx = vSrc->ne[0];
  const int64_t cPy = vSrc->ne[1];
  const int64_t cPa = cPx * cPy;

  const float *vSplane = (const float *)vSrc->data;

  const int cKa = cK0 * cK1;
  const int cOffset0 = -cP0;
  const int cOffset1 = -cP1;

  while (vCdata < cDataEnd) {
    for (int vOy = 0; vOy < cPy; ++vOy) {
      const float *const cSrow = vSplane + vOy * cPx;
      for (int vOx = 0; vOx < cPx; ++vOx) {
        const float cGrad0 = cSrow[vOx];

        const int cIx = cOffset0 + vOx * cS0;
        const int cIy = cOffset1 + vOy * cS1;

        if (vOp == GGML_OP_POOL_MAX) {
          float vMaxval = -FLT_MAX;
          int vKxmax = -1;
          int vKymax = -1;

          for (int vKy = 0; vKy < cK1; ++vKy) {
            if (cIy + vKy < 0 || cIy + vKy >= pDst->ne[1]) {
              continue;
            }
            const void *vDrowf = (const void *)(vCdataf + pDst->nb[1] * (cIy + vKy));
            for (int vKx = 0; vKx < cK0; ++vKx) {
              int vJ = cIx + vKx;
              if (vJ < 0 || vJ >= pDst->ne[0]) {
                continue;
              }

              const float cVal = pDst->type == GGML_TYPE_F32 ? ((const float *)vDrowf)[vJ]
                                                             : GGML_CPU_FP16_TO_FP32(((const ggml_fp16_t *)vDrowf)[vJ]);
              if (cVal <= vMaxval) {
                continue;
              }

              vMaxval = cVal;
              vKxmax = vKx;
              vKymax = vKy;
            }
          }

          if (vKxmax == -1 || vKymax == -1) {
            continue;
          }

          void *vDrow = (void *)(vCdata + pDst->nb[1] * (cIy + vKymax));
          const int cJ = cIx + vKxmax;
          if (pDst->type == GGML_TYPE_F32) {
            ((float *)vDrow)[cJ] += cGrad0;
          } else {
            ((ggml_fp16_t *)vDrow)[cJ] =
                GGML_CPU_FP32_TO_FP16(cGrad0 + GGML_CPU_FP16_TO_FP32(((const ggml_fp16_t *)vDrow)[cJ]));
          }
        } else if (vOp == GGML_OP_POOL_AVG) {
          const float cGrad = cGrad0 / cKa;

          for (int vKy = 0; vKy < cK1; ++vKy) {
            if (cIy + vKy < 0 || cIy + vKy >= pDst->ne[1]) {
              continue;
            }
            void *vDrow = (void *)(vCdata + pDst->nb[1] * (cIy + vKy));
            for (int vKx = 0; vKx < cK0; ++vKx) {
              int vJ = cIx + vKx;
              if (vJ < 0 || vJ >= pDst->ne[0]) {
                continue;
              }

              if (pDst->type == GGML_TYPE_F32) {
                ((float *)vDrow)[vJ] += cGrad;
              } else {
                ((ggml_fp16_t *)vDrow)[vJ] += GGML_CPU_FP32_TO_FP16(cGrad);
              }
            }
          }
        } else {
          GGML_ASSERT(false);
        }
      }
    }

    vCdata += pDst->nb[2];
    vCdataf += pDst->nb[2];
    vSplane += cPa;
  }
}

// ggml_compute_forward_upscale

static void fGgmlComputeForwardUpscaleF32(const ggml_compute_params *pParams, ggml_tensor *dst) {

  const ggml_tensor *src0 = dst->src[0];

  GGML_ASSERT(src0->type == GGML_TYPE_F32);

  const int cIth = pParams->ith;
  const int cNth = pParams->nth;

  GGML_TENSOR_UNARY_OP_LOCALS

  float vSf0 = (float)ne0 / src0->ne[0];
  float vSf1 = (float)ne1 / src0->ne[1];
  float vSf2 = (float)ne2 / src0->ne[2];
  float vSf3 = (float)ne3 / src0->ne[3];
  float vPixelOffset = 0.5f;

  const int32_t cModeFlags = ggml_get_op_params_i32(dst, 0);
  const ggml_scale_mode cMode = (ggml_scale_mode)(cModeFlags & 0xFF);

  if (cModeFlags & GGML_SCALE_FLAG_ALIGN_CORNERS) {
    vPixelOffset = 0.0f;
    vSf0 = ne0 > 1 && ne00 > 1 ? (float)(ne0 - 1) / (ne00 - 1) : vSf0;
    vSf1 = ne1 > 1 && ne01 > 1 ? (float)(ne1 - 1) / (ne01 - 1) : vSf1;
  }

  if (cMode == GGML_SCALE_MODE_NEAREST) {
    for (int64_t vI3 = 0; vI3 < ne3; vI3++) {
      const int64_t cI03 = vI3 / vSf3;
      for (int64_t vI2 = cIth; vI2 < ne2; vI2 += cNth) {
        const int64_t cI02 = vI2 / vSf2;
        for (int64_t vI1 = 0; vI1 < ne1; vI1++) {
          const int64_t cI01 = vI1 / vSf1;
          for (int64_t vI0 = 0; vI0 < ne0; vI0++) {
            const int64_t cI00 = vI0 / vSf0;

            const float *vX = (float *)((char *)src0->data + cI00 * nb00 + cI01 * nb01 + cI02 * nb02 + cI03 * nb03);
            float *vY = (float *)((char *)dst->data + vI0 * nb0 + vI1 * nb1 + vI2 * nb2 + vI3 * nb3);

            *vY = *vX;
          }
        }
      }
    }
  } else if (cMode == GGML_SCALE_MODE_BILINEAR && (cModeFlags & GGML_SCALE_FLAG_ANTIALIAS)) {
    // Similar to F.interpolate(..., mode="bilinear", align_corners=False, antialias=True)
    // https://github.com/pytorch/pytorch/blob/8871ff29b743948d1225389d5b7068f37b22750b/aten/src/ATen/native/cpu/UpSampleKernel.cpp
    auto fTriangleFilter = [](float pX) -> float { return std::max(1.0f - fabsf(pX), 0.0f); };

    // support and invscale, minimum 1 pixel for bilinear
    const float cSupport1 = std::max(1.0f, 1.0f / vSf1);
    const float cInvscale1 = 1.0f / cSupport1;
    const float cSupport0 = std::max(1.0f, 1.0f / vSf0);
    const float cInvscale0 = 1.0f / cSupport0;

    for (int64_t vI3 = 0; vI3 < ne3; vI3++) {
      const int64_t cI03 = vI3 / vSf3;
      for (int64_t vI2 = cIth; vI2 < ne2; vI2 += cNth) {
        const int64_t cI02 = vI2 / vSf2;
        for (int64_t vI1 = 0; vI1 < ne1; vI1++) {
          const float cY = ((float)vI1 + vPixelOffset) / vSf1;
          for (int64_t vI0 = 0; vI0 < ne0; vI0++) {
            const float cX = ((float)vI0 + vPixelOffset) / vSf0;

            // the range of source pixels that contribute
            const int64_t cXMin = std::max<int64_t>(cX - cSupport0 + vPixelOffset, 0);
            const int64_t cXMax = std::min<int64_t>(cX + cSupport0 + vPixelOffset, ne00);
            const int64_t cYMin = std::max<int64_t>(cY - cSupport1 + vPixelOffset, 0);
            const int64_t cYMax = std::min<int64_t>(cY + cSupport1 + vPixelOffset, ne01);

            // bilinear filter with antialiasing
            float vVal = 0.0f;
            float vTotalWeight = 0.0f;

            for (int64_t vSy = cYMin; vSy < cYMax; vSy++) {
              const float cWeightY = fTriangleFilter((vSy - cY + vPixelOffset) * cInvscale1);

              for (int64_t vSx = cXMin; vSx < cXMax; vSx++) {
                const float cWeightX = fTriangleFilter((vSx - cX + vPixelOffset) * cInvscale0);
                const float cWeight = cWeightX * cWeightY;

                if (cWeight <= 0.0f) {
                  continue;
                }

                const float cPixel =
                    *(const float *)((const char *)src0->data + vSx * nb00 + vSy * nb01 + cI02 * nb02 + cI03 * nb03);
                vVal += cPixel * cWeight;
                vTotalWeight += cWeight;
              }
            }

            if (vTotalWeight > 0.0f) {
              vVal /= vTotalWeight;
            }

            float *vDstPtr = (float *)((char *)dst->data + vI0 * nb0 + vI1 * nb1 + vI2 * nb2 + vI3 * nb3);
            *vDstPtr = vVal;
          }
        }
      }
    }
  } else if (cMode == GGML_SCALE_MODE_BILINEAR) {
    for (int64_t vI3 = 0; vI3 < ne3; vI3++) {
      const int64_t cI03 = vI3 / vSf3;
      for (int64_t vI2 = cIth; vI2 < ne2; vI2 += cNth) {
        const int64_t cI02 = vI2 / vSf2;
        for (int64_t vI1 = 0; vI1 < ne1; vI1++) {
          const float cY = ((float)vI1 + vPixelOffset) / vSf1 - vPixelOffset;
          int64_t vY0 = (int64_t)floorf(cY);
          int64_t vY1 = vY0 + 1;

          vY0 = std::max(int64_t(0), std::min(vY0, ne01 - 1));
          vY1 = std::max(int64_t(0), std::min(vY1, ne01 - 1));

          float vDy = cY - (float)vY0;
          vDy = std::max(0.0f, std::min(vDy, 1.0f));

          for (int64_t vI0 = 0; vI0 < ne0; vI0++) {
            const float cX = ((float)vI0 + vPixelOffset) / vSf0 - vPixelOffset;
            int64_t vX0 = (int64_t)floorf(cX);
            int64_t vX1 = vX0 + 1;

            vX0 = std::max(int64_t(0), std::min(vX0, ne00 - 1));
            vX1 = std::max(int64_t(0), std::min(vX1, ne00 - 1));

            float vDx = cX - (float)vX0;
            vDx = std::max(0.0f, std::min(vDx, 1.0f));

            // fetch the four surrounding pixel values and interpolate
            const float cA =
                *(const float *)((const char *)src0->data + vX0 * nb00 + vY0 * nb01 + cI02 * nb02 + cI03 * nb03);
            const float cB =
                *(const float *)((const char *)src0->data + vX1 * nb00 + vY0 * nb01 + cI02 * nb02 + cI03 * nb03);
            const float c_ =
                *(const float *)((const char *)src0->data + vX0 * nb00 + vY1 * nb01 + cI02 * nb02 + cI03 * nb03);
            const float cD =
                *(const float *)((const char *)src0->data + vX1 * nb00 + vY1 * nb01 + cI02 * nb02 + cI03 * nb03);

            const float cVal =
                cA * (1 - vDx) * (1 - vDy) + cB * vDx * (1 - vDy) + c_ * (1 - vDx) * vDy + cD * vDx * vDy;

            float *vYDst = (float *)((char *)dst->data + vI0 * nb0 + vI1 * nb1 + vI2 * nb2 + vI3 * nb3);
            *vYDst = cVal;
          }
        }
      }
    }
  } else if (cMode == GGML_SCALE_MODE_BICUBIC) {
    // https://en.wikipedia.org/wiki/Bicubic_interpolation#Bicubic_convolution_algorithm
    const float cA = -0.75f; // use alpha = -0.75 (same as PyTorch)
    auto vWeight1 = [cA](float pX) { return ((cA + 2) * pX - (cA + 3)) * pX * pX + 1; };
    auto vWeight2 = [cA](float pX) { return ((cA * pX - 5 * cA) * pX + 8 * cA) * pX - 4 * cA; };
    auto fBicubic = [=](float p0, float p1, float p2, float p3, float pX) {
      const float cW0 = vWeight2(pX + 1);
      const float cW1 = vWeight1(pX + 0);
      const float cW2 = vWeight1(1 - pX);
      const float cW3 = vWeight2(2 - pX);
      return p0 * cW0 + p1 * cW1 + p2 * cW2 + p3 * cW3;
    };

    for (int64_t vI3 = 0; vI3 < ne3; vI3++) {
      const int64_t cI03 = vI3 / vSf3;
      for (int64_t vI2 = cIth; vI2 < ne2; vI2 += cNth) {
        const int64_t cI02 = vI2 / vSf2;
        for (int64_t vI1 = 0; vI1 < ne1; vI1++) {
          const float cY = ((float)vI1 + vPixelOffset) / vSf1 - vPixelOffset;
          const int64_t cY0 = (int64_t)floorf(cY);
          const float cDy = cY - (float)cY0;

          for (int64_t vI0 = 0; vI0 < ne0; vI0++) {
            const float cX = ((float)vI0 + vPixelOffset) / vSf0 - vPixelOffset;
            const int64_t cX0 = (int64_t)floorf(cX);
            const float cDx = cX - (float)cX0;

            auto vP = [=](int64_t pXOff, int64_t pYOff) -> float {
              int64_t vI00 = std::max(int64_t(0), std::min(cX0 + pXOff, ne00 - 1));
              int64_t vI01 = std::max(int64_t(0), std::min(cY0 + pYOff, ne01 - 1));
              return *(const float *)((const char *)src0->data + vI00 * nb00 + vI01 * nb01 + cI02 * nb02 + cI03 * nb03);
            };

            const float cVal = fBicubic(fBicubic(vP(-1, -1), vP(0, -1), vP(1, -1), vP(2, -1), cDx),
                                        fBicubic(vP(-1, 0), vP(0, 0), vP(1, 0), vP(2, 0), cDx),
                                        fBicubic(vP(-1, 1), vP(0, 1), vP(1, 1), vP(2, 1), cDx),
                                        fBicubic(vP(-1, 2), vP(0, 2), vP(1, 2), vP(2, 2), cDx), cDy);

            float *vYDst = (float *)((char *)dst->data + vI0 * nb0 + vI1 * nb1 + vI2 * nb2 + vI3 * nb3);
            *vYDst = cVal;
          }
        }
      }
    }
  } else {
    GGML_ABORT("unsupported upscale mode");
  }
}

void ggml_compute_forward_upscale(const ggml_compute_params *pParams, ggml_tensor *pDst) {

  const ggml_tensor *vSrc0 = pDst->src[0];

  switch (vSrc0->type) {
  case GGML_TYPE_F32: {
    fGgmlComputeForwardUpscaleF32(pParams, pDst);
  } break;
  default: {
    GGML_ABORT("fatal error");
  }
  }
}

// ggml_compute_forward_pad

template <bool circular_t> static void fGgmlComputeForwardPadF32(const ggml_compute_params *pParams, ggml_tensor *dst) {

  const ggml_tensor *src0 = dst->src[0];

  assert(dst->nb[0] == sizeof(float));

  const int cIth = pParams->ith;
  const int cNth = pParams->nth;

  GGML_TENSOR_UNARY_OP_LOCALS

  float *vDstPtr = (float *)dst->data;
  const int32_t cLp0 = ggml_get_op_params_i32(dst, 0);
  const int32_t cRp0 = ggml_get_op_params_i32(dst, 1);
  const int32_t cLp1 = ggml_get_op_params_i32(dst, 2);
  const int32_t cRp1 = ggml_get_op_params_i32(dst, 3);
  const int32_t cLp2 = ggml_get_op_params_i32(dst, 4);
  const int32_t cRp2 = ggml_get_op_params_i32(dst, 5);
  const int32_t cLp3 = ggml_get_op_params_i32(dst, 6);
  const int32_t cRp3 = ggml_get_op_params_i32(dst, 7);

  // TODO: optimize

  for (int64_t vI2 = 0; vI2 < ne2; ++vI2) {
    for (int64_t vI1 = cIth; vI1 < ne1; vI1 += cNth) {
      for (int64_t vI0 = 0; vI0 < ne0; ++vI0) {
        for (int64_t vI3 = 0; vI3 < ne3; ++vI3) {
          // circular means wrap around on a torus, so x and y loop around
          if constexpr (circular_t) {
            const int64_t dst_idx = vI3 * (ne0 * ne1 * ne2) + vI2 * (ne0 * ne1) + vI1 * ne0 + vI0;
            const int64_t src_i0 = fGgmlWrapAround(vI0 - cLp0, ne00);
            const int64_t src_i1 = fGgmlWrapAround(vI1 - cLp1, ne01);
            const int64_t src_i2 = fGgmlWrapAround(vI2 - cLp2, ne02);
            const int64_t src_i3 = fGgmlWrapAround(vI3 - cLp3, ne03);

            const int64_t src_idx = src_i3 * nb03 + src_i2 * nb02 + src_i1 * nb01 + src_i0 * nb00;

            const float *src_ptr = (const float *)((char *)src0->data + src_idx);
            vDstPtr[dst_idx] = *src_ptr;
          } else {
            const int64_t cDstIdx = vI3 * (ne0 * ne1 * ne2) + vI2 * (ne0 * ne1) + vI1 * ne0 + vI0;
            if ((vI0 >= cLp0 && vI0 < ne0 - cRp0) && (vI1 >= cLp1 && vI1 < ne1 - cRp1) &&
                (vI2 >= cLp2 && vI2 < ne2 - cRp2) && (vI3 >= cLp3 && vI3 < ne3 - cRp3)) {
              const int64_t cSrcIdx =
                  (vI3 - cLp3) * nb03 + (vI2 - cLp2) * nb02 + (vI1 - cLp1) * nb01 + (vI0 - cLp0) * nb00;
              const float *vSrcPtr = (const float *)((char *)src0->data + cSrcIdx);
              vDstPtr[cDstIdx] = *vSrcPtr;
            } else {
              vDstPtr[cDstIdx] = 0;
            }
          }
        }
      }
    }
  }
}

void ggml_compute_forward_pad(const ggml_compute_params *pParams, ggml_tensor *pDst) {
  const ggml_tensor *vSrc0 = pDst->src[0];
  const bool cCircular = (bool)ggml_get_op_params_i32(pDst, 8);
  switch (vSrc0->type) {
  case GGML_TYPE_F32: {
    if (cCircular) {
      fGgmlComputeForwardPadF32<true>(pParams, pDst);
    } else {
      fGgmlComputeForwardPadF32<false>(pParams, pDst);
    }
  } break;
  default: {
    GGML_ABORT("fatal error");
  }
  }
}

// ggml_compute_forward_pad_reflect_1d

void ggml_compute_forward_pad_reflect_1d(const ggml_compute_params *pParams, ggml_tensor *dst) {

  const ggml_tensor *src0 = dst->src[0];

  GGML_ASSERT(src0->type == GGML_TYPE_F32);
  GGML_ASSERT(dst->type == GGML_TYPE_F32);

  const int cIth = pParams->ith;
  const int cNth = pParams->nth;

  const int32_t *vOpts = (const int32_t *)dst->op_params;
  const int cP0 = vOpts[0];
  const int cP1 = vOpts[1];

  GGML_TENSOR_UNARY_OP_LOCALS

  for (int64_t vI3 = 0; vI3 < ne3; vI3++) {
    for (int64_t vI2 = 0; vI2 < ne2; vI2++) {
      for (int64_t vI1 = cIth; vI1 < ne1; vI1 += cNth) {
        float *vLeft = (float *)((char *)dst->data + vI3 * nb3 + vI2 * nb2 + vI1 * nb1 + cP0 * nb0);
        float *vRight = (float *)((char *)dst->data + vI3 * nb3 + vI2 * nb2 + vI1 * nb1 + (ne0 - cP1 - 1) * nb0);

        ggml_vec_cpy_f32(ne00, vLeft, (float *)((char *)src0->data + vI3 * nb03 + vI2 * nb02 + vI1 * nb01));

        for (int vI0 = 1; vI0 <= cP0; vI0++) {
          vLeft[-vI0] = vLeft[vI0];
        }
        for (int vI0 = 1; vI0 <= cP1; vI0++) {
          vRight[vI0] = vRight[-vI0];
        }
      }
    }
  }
}

// ggml_compute_forward_roll

static int64_t fGgmlWrapIndex(int64_t pI, int64_t pNe) {
  if (pI < 0) {
    return pI + pNe;
  } else if (pI >= pNe) {
    return pI - pNe;
  }
  return pI;
}

static void fGgmlComputeForwardRollF32(const ggml_compute_params *pParams, ggml_tensor *dst) {

  const ggml_tensor *src0 = dst->src[0];
  const float *vSrcData = (const float *)src0->data;
  float *vDstData = (float *)dst->data;

  GGML_TENSOR_UNARY_OP_LOCALS

  const int cS0 = ggml_get_op_params_i32(dst, 0);
  const int cS1 = ggml_get_op_params_i32(dst, 1);
  const int cS2 = ggml_get_op_params_i32(dst, 2);
  const int cS3 = ggml_get_op_params_i32(dst, 3);

  const int64_t cTotal = ne1 * ne2 * ne3;
  const int64_t cPerThread = (cTotal + pParams->nth) / pParams->nth;
  const int64_t cStart = pParams->ith * cPerThread;
  const int64_t cEnd = std::min(cStart + cPerThread, cTotal);

  for (int64_t vI = cStart; vI < cEnd; ++vI) {
    const int64_t cI1 = vI % ne1;
    const int64_t cI2 = (vI / ne1) % ne2;
    const int64_t cI3 = vI / (ne2 * ne1);
    float *vDstRow = vDstData + (cI3 * nb3 + cI2 * nb2 + cI1 * nb1) / sizeof(float);

    const int64_t cI01 = fGgmlWrapIndex(cI1 - cS1, ne01);
    const int64_t cI02 = fGgmlWrapIndex(cI2 - cS2, ne02);
    const int64_t cI03 = fGgmlWrapIndex(cI3 - cS3, ne03);
    const float *vSrcRow = vSrcData + (cI03 * nb03 + cI02 * nb02 + cI01 * nb01) / sizeof(float);

    const int64_t cS = fGgmlWrapIndex(-cS0, ne00);
    const int64_t cN = ne00 - cS;
    ggml_vec_cpy_f32(cN, vDstRow, vSrcRow + cS);
    ggml_vec_cpy_f32(cS, vDstRow + cN, vSrcRow);
  }
}

void ggml_compute_forward_roll(const ggml_compute_params *pParams, ggml_tensor *pDst) {

  const ggml_tensor *vSrc0 = pDst->src[0];

  switch (vSrc0->type) {
  case GGML_TYPE_F32: {
    fGgmlComputeForwardRollF32(pParams, pDst);
  } break;
  default: {
    GGML_ABORT("fatal error");
  }
  }
}

// ggml_compute_forward_arange

static void fGgmlComputeForwardArangeF32(const ggml_compute_params *pParams, ggml_tensor *pDst) {

  GGML_ASSERT(pDst->nb[0] == sizeof(float));

  const int cIth = pParams->ith;
  const int cNth = pParams->nth;

  const float cStart = ggml_get_op_params_f32(pDst, 0);
  const float cStop = ggml_get_op_params_f32(pDst, 1);
  const float cStep = ggml_get_op_params_f32(pDst, 2);

  const int64_t cSteps = (int64_t)ceilf((cStop - cStart) / cStep);

  GGML_ASSERT(ggml_nelements(pDst) == cSteps);

  for (int64_t vI = cIth; vI < cSteps; vI += cNth) {
    float vValue = cStart + cStep * vI;
    ((float *)pDst->data)[vI] = vValue;
  }
}

void ggml_compute_forward_arange(const ggml_compute_params *pParams, ggml_tensor *pDst) {
  switch (pDst->type) {
  case GGML_TYPE_F32: {
    fGgmlComputeForwardArangeF32(pParams, pDst);
  } break;
  default: {
    GGML_ABORT("fatal error");
  }
  }
}

static void fGgmlComputeForwardTimestepEmbeddingF32(const ggml_compute_params *pParams, ggml_tensor *dst) {

  const ggml_tensor *src0 = dst->src[0];

  GGML_ASSERT(src0->nb[0] == sizeof(float));

  const int cIth = pParams->ith;
  const int cNth = pParams->nth;

  GGML_TENSOR_UNARY_OP_LOCALS

  const int cDim = ggml_get_op_params_i32(dst, 0);
  const int cMaxPeriod = ggml_get_op_params_i32(dst, 1);

  int vHalf = cDim / 2;

  for (int64_t vI = 0; vI < ne00; vI++) {
    float *vEmbedData = (float *)((char *)dst->data + vI * nb1);
    for (int64_t vJ = cIth; vJ < vHalf; vJ += cNth) {
      float vTimestep = ((float *)src0->data)[vI];
      float vFreq = (float)expf(-logf(cMaxPeriod) * vJ / vHalf);
      float vArg = vTimestep * vFreq;
      vEmbedData[vJ] = cosf(vArg);
      vEmbedData[vJ + vHalf] = sinf(vArg);
    }
    if (cDim % 2 != 0 && cIth == 0) {
      vEmbedData[2 * vHalf] = 0.f;
    }
  }
}

void ggml_compute_forward_timestep_embedding(const ggml_compute_params *pParams, ggml_tensor *pDst) {

  const ggml_tensor *vSrc0 = pDst->src[0];

  switch (vSrc0->type) {
  case GGML_TYPE_F32: {
    fGgmlComputeForwardTimestepEmbeddingF32(pParams, pDst);
  } break;
  default: {
    GGML_ABORT("fatal error");
  }
  }
}

// ggml_compute_forward_argsort

template <enum ggml_sort_order order> struct cmp_argsort {
  const float *data;
  bool operator()(int32_t pA, int32_t pB) const {
    if constexpr (order == GGML_SORT_ORDER_ASC) {
      return data[pA] < data[pB];
    } else {
      return data[pA] > data[pB];
    }
  }
};

static void fGgmlComputeForwardArgsortF32(const ggml_compute_params *pParams, ggml_tensor *dst) {

  const ggml_tensor *src0 = dst->src[0];

  GGML_TENSOR_UNARY_OP_LOCALS

  GGML_ASSERT(nb0 == sizeof(float));

  const int cIth = pParams->ith;
  const int cNth = pParams->nth;

  const int64_t cNr = ggml_nrows(src0);

  ggml_sort_order vOrder = (ggml_sort_order)ggml_get_op_params_i32(dst, 0);

  for (int64_t vI = cIth; vI < cNr; vI += cNth) {
    const float *vSrcData = (float *)((char *)src0->data + vI * nb01);

    int32_t *vDstData = (int32_t *)((char *)dst->data + vI * nb1);

    for (int64_t vJ = 0; vJ < ne0; vJ++) {
      vDstData[vJ] = vJ;
    }

    switch (vOrder) {
    case GGML_SORT_ORDER_ASC:
      std::sort(vDstData, vDstData + ne0, cmp_argsort<GGML_SORT_ORDER_ASC>{vSrcData});
      break;

    case GGML_SORT_ORDER_DESC:
      std::sort(vDstData, vDstData + ne0, cmp_argsort<GGML_SORT_ORDER_DESC>{vSrcData});
      break;

    default:
      GGML_ABORT("invalid sort order");
    }
  }
}

void ggml_compute_forward_argsort(const ggml_compute_params *pParams, ggml_tensor *pDst) {

  const ggml_tensor *vSrc0 = pDst->src[0];

  switch (vSrc0->type) {
  case GGML_TYPE_F32: {
    fGgmlComputeForwardArgsortF32(pParams, pDst);
  } break;
  default: {
    GGML_ABORT("fatal error");
  }
  }
}

// ggml_compute_forward_top_k

struct cmp_top_k {
  const float *data;
  bool operator()(int32_t pA, int32_t pB) const { return data[pA] > data[pB]; }
};

static void fGgmlComputeForwardTopKF32(const ggml_compute_params *pParams, ggml_tensor *dst) {

  const ggml_tensor *src0 = dst->src[0];

  GGML_TENSOR_UNARY_OP_LOCALS

  GGML_ASSERT(nb0 == sizeof(float));

  const int cIth = pParams->ith;
  const int cNth = pParams->nth;

  const int64_t cNr = ggml_nrows(src0);

  const int cTopK = ne0;

  int32_t *vTmp = (int32_t *)pParams->wdata + (ne00 + CACHE_LINE_SIZE_F32) * cIth;

  for (int64_t vI = cIth; vI < cNr; vI += cNth) {
    const float *vSrcData = (float *)((char *)src0->data + vI * nb01);

    for (int64_t vJ = 0; vJ < ne00; vJ++) {
      vTmp[vJ] = vJ;
    }

    std::partial_sort(vTmp, vTmp + cTopK, vTmp + ne00, cmp_top_k{vSrcData});

    int32_t *vDstData = (int32_t *)((char *)dst->data + vI * nb1);

    std::copy(vTmp, vTmp + cTopK, vDstData);

    // emphasize that the order is not important
    if (cTopK > 1) {
      std::swap(vDstData[0], vDstData[1]);
    }
  }
}

void ggml_compute_forward_top_k(const ggml_compute_params *pParams, ggml_tensor *pDst) {

  const ggml_tensor *vSrc0 = pDst->src[0];

  switch (vSrc0->type) {
  case GGML_TYPE_F32: {
    fGgmlComputeForwardTopKF32(pParams, pDst);
  } break;
  default: {
    GGML_ABORT("fatal error");
  }
  }
}

static void fGgmlComputeForwardFlashAttnExtF16OneChunk(const ggml_compute_params *pParams, ggml_tensor *pDst, int pIr0,
                                                       int pIr1, int64_t pIcStart, int64_t pIcEnd, float *pPartials,
                                                       int64_t pPartialStride) {

  const bool cWritePartials = (pPartials != nullptr);
  const ggml_tensor *vQ = pDst->src[0];
  const ggml_tensor *vK = pDst->src[1];
  const ggml_tensor *v_ = pDst->src[2];
  const ggml_tensor *vMask = pDst->src[3];
  const ggml_tensor *vSinks = pDst->src[4];

  GGML_TENSOR_LOCALS(int64_t, neq, vQ, ne)
  GGML_TENSOR_LOCALS(size_t, nbq, vQ, nb)
  GGML_TENSOR_LOCALS(int64_t, nek, vK, ne)
  GGML_TENSOR_LOCALS(size_t, nbk, vK, nb)
  GGML_TENSOR_LOCALS(int64_t, nev, v_, ne)
  GGML_TENSOR_LOCALS(size_t, nbv, v_, nb)
  GGML_TENSOR_LOCALS(int64_t, ne, pDst, ne)
  GGML_TENSOR_LOCALS(size_t, nb, pDst, nb)

  const int64_t cDk = nek0;
  const int64_t cDv = nev0;
  const int64_t cN = neq1;

  GGML_ASSERT(ne0 == cDv);
  GGML_ASSERT(ne2 == cN);

  // input tensor rows must be contiguous
  GGML_ASSERT(nbq0 == ggml_type_size(vQ->type));
  GGML_ASSERT(nbk0 == ggml_type_size(vK->type));
  GGML_ASSERT(nbv0 == ggml_type_size(v_->type));

  GGML_ASSERT(neq0 == cDk);
  GGML_ASSERT(nek0 == cDk);
  GGML_ASSERT(nev0 == cDv);

  GGML_ASSERT(neq1 == cN);

  // dst cannot be transposed or permuted
  GGML_ASSERT(nb0 == sizeof(float));
  GGML_ASSERT(nb0 <= nb1);
  GGML_ASSERT(nb1 <= nb2);
  GGML_ASSERT(nb2 <= nb3);

  // broadcast factors
  const int64_t cRk2 = neq2 / nek2;
  const int64_t cRk3 = neq3 / nek3;

  const int64_t cRv2 = neq2 / nev2;
  const int64_t cRv3 = neq3 / nev3;

  // parallelize by q rows using ggml_vec_dot_f32

  float vScale = 1.0f;
  float vMaxBias = 0.0f;
  float vLogitSoftcap = 0.0f;

  memcpy(&vScale, (float *)pDst->op_params + 0, sizeof(float));
  memcpy(&vMaxBias, (float *)pDst->op_params + 1, sizeof(float));
  memcpy(&vLogitSoftcap, (float *)pDst->op_params + 2, sizeof(float));

  if (vLogitSoftcap != 0) {
    vScale /= vLogitSoftcap;
  }

  const uint32_t cNHead = neq2;
  const uint32_t cNHeadLog2 = 1u << (uint32_t)floor(log2(cNHead));

  const float cM0 = powf(2.0f, -(vMaxBias) / cNHeadLog2);
  const float cM1 = powf(2.0f, -(vMaxBias / 2.0f) / cNHeadLog2);

  ggml_type const cKVecDotType = ggml_get_type_traits_cpu(vK->type)->vec_dot_type;
  ggml_from_float_t const cQToVecDot = ggml_get_type_traits_cpu(cKVecDotType)->from_float;
  ggml_vec_dot_t const cKqVecDot = ggml_get_type_traits_cpu(vK->type)->vec_dot;
  ggml_to_float_t const cVToFloat = ggml_get_type_traits(v_->type)->to_float;

  GGML_ASSERT((cQToVecDot) && "fattn: unsupported K-type");
  GGML_ASSERT((v_->type == GGML_TYPE_F32 || cVToFloat) && "fattn: unsupported V-type");

  int vIth = pParams->ith;

  for (int vIr = pIr0; vIr < pIr1; ++vIr) {
    // q indices
    const int cIq3 = vIr / (neq2 * neq1);
    const int cIq2 = (vIr - cIq3 * neq2 * neq1) / neq1;
    const int cIq1 = (vIr - cIq3 * neq2 * neq1 - cIq2 * neq1);

    const uint32_t cH = cIq2; // head index
    const float cSlope =
        (vMaxBias > 0.0f) ? cH < cNHeadLog2 ? powf(cM0, cH + 1) : powf(cM1, 2 * (cH - cNHeadLog2) + 1) : 1.0f;

    float S = 0.0f;       // sum
    float vM = -INFINITY; // maximum KQ value

    float *vVkQ32 = (float *)pParams->wdata + vIth * (1 * cDk + 2 * cDv + CACHE_LINE_SIZE_F32); // FP32 VKQ accumulator
    float *vV32 = (vVkQ32 + 1 * cDv);                        // (temporary) FP32 V buffer
    ggml_fp16_t *vVkQ16 = (ggml_fp16_t *)(vVkQ32 + 1 * cDv); // (temporary) FP16 VKQ accumulator
    ggml_fp16_t *vQQ = (ggml_fp16_t *)(vVkQ32 + 2 * cDv);    // (temporary) buffer for Q converted to quantized/FP16

    if (v_->type == GGML_TYPE_F16) {
      memset(vVkQ16, 0, cDv * sizeof(ggml_fp16_t));
    } else {
      memset(vVkQ32, 0, cDv * sizeof(float));
    }

    const ggml_fp16_t *vMp =
        vMask ? (ggml_fp16_t *)((char *)vMask->data + cIq1 * vMask->nb[1] + (cIq2 % vMask->ne[2]) * vMask->nb[2] +
                                (cIq3 % vMask->ne[3]) * vMask->nb[3])
              : NULL;

    // k indices
    const int cIk3 = cIq3 / cRk3;
    const int cIk2 = cIq2 / cRk2;

    // v indices
    const int cIv3 = cIq3 / cRv3;
    const int cIv2 = cIq2 / cRv2;

    const float *vPq = (const float *)((char *)vQ->data + (cIq1 * nbq1 + cIq2 * nbq2 + cIq3 * nbq3));
    cQToVecDot(vPq, vQQ, cDk);

    // online softmax / attention
    // loop over n_kv and n_head_kv
    // ref: https://arxiv.org/pdf/2112.05682.pdf

    for (int64_t vIc = pIcStart; vIc < pIcEnd; ++vIc) {
      const float cMv = vMp ? cSlope * GGML_CPU_FP16_TO_FP32(vMp[vIc]) : 0.0f;
      if (cMv == -INFINITY) {
        continue;
      }

      float s; // KQ value

      const char *vKData = (const char *)vK->data + (vIc * nbk1 + cIk2 * nbk2 + cIk3 * nbk3);
      cKqVecDot(cDk, &s, 0, vKData, 0, vQQ, 0, 1);

      s = s * vScale; // scale KQ value

      if (vLogitSoftcap != 0.0f) {
        s = vLogitSoftcap * tanhf(s);
      }

      s += cMv; // apply mask

      const float cMold = vM;

      float vMs = 1.0f; // upon new higher max val, scale VKQ and KQ sum with this value
      float vVs = 1.0f; // post-softmax KQ value, expf(s - M)

      const char *vVData = ((const char *)v_->data + (vIc * nbv1 + cIv2 * nbv2 + cIv3 * nbv3));

      if (v_->type == GGML_TYPE_F16) {
        if (s > vM) {
          // s is new maximum, ms < 1.0f, vs == expf(s - s) == 1.0f
          vM = s;
          vMs = expf(cMold - vM);

          // V = V*expf(Mold - M)
          ggml_vec_scale_f16(cDv, vVkQ16, vMs);
        } else {
          // no new maximum, ms == 1.0f, vs != 1.0f
          vVs = expf(s - vM);
        }

        // V += v*expf(s - M)
        ggml_vec_mad_f16(cDv, vVkQ16, (const ggml_fp16_t *)vVData, vVs);
      } else {
        if (s > vM) {
          // s is new maximum, ms < 1.0f, vs == expf(s - s) == 1.0f
          vM = s;
          vMs = expf(cMold - vM);

          // V = V*expf(Mold - M)
          ggml_vec_scale_f32(cDv, vVkQ32, vMs);
        } else {
          // no new maximum, ms == 1.0f, vs != 1.0f
          vVs = expf(s - vM);
        }

        // V += v*expf(s - M)
        if (cVToFloat) {
          cVToFloat(vVData, vV32, cDv);
          ggml_vec_mad_f32(cDv, vVkQ32, vV32, vVs);
        } else {
          // V is F32
          ggml_vec_mad_f32(cDv, vVkQ32, (const float *)vVData, vVs);
        }
      }

      S = S * vMs + vVs; // scale and increment sum with partial sum
    }

    if (v_->type == GGML_TYPE_F16) {
      for (int64_t vD = 0; vD < cDv; ++vD) {
        vVkQ32[vD] = GGML_CPU_FP16_TO_FP32(vVkQ16[vD]);
      }
    }

    // sinks - apply only on the first kv-chunk
    if (vSinks && pIcStart == 0) {
      const float cS = ((float *)((char *)vSinks->data))[cH];

      float vMs = 1.0f;
      float vVs = 1.0f;

      if (cS > vM) {
        vMs = expf(vM - cS);
        vM = cS;
        ggml_vec_scale_f32(cDv, vVkQ32, vMs);
      } else {
        vVs = expf(cS - vM);
      }

      S = S * vMs + vVs;
    }

    if (cWritePartials) {
      // Write M, S, VKQ to partials for later reduction
      // partials layout: [M, S, VKQ[DV]] per query head
      float *vPartial = pPartials + vIr * pPartialStride;
      vPartial[0] = vM;
      vPartial[1] = S;
      memcpy(vPartial + 2, vVkQ32, cDv * sizeof(float));
    } else {
      // V /= S
      const float cSInv = S == 0.0f ? 0.0f : 1.0f / S;
      ggml_vec_scale_f32(cDv, vVkQ32, cSInv);

      // dst indices
      const int cI1 = cIq1;
      const int cI2 = cIq2;
      const int cI3 = cIq3;

      // permute(0, 2, 1, 3)
      memcpy((char *)pDst->data + (cI3 * ne2 * ne1 + cI2 + cI1 * ne1) * nb1, vVkQ32, nb1);
    }
  }
}

static void fGgmlComputeForwardFlashAttnExtTiled(const ggml_compute_params *pParams, ggml_tensor *pDst, int pIr0,
                                                 int pIr1) {
  const ggml_tensor *vQ = pDst->src[0];
  const ggml_tensor *vK = pDst->src[1];
  const ggml_tensor *v_ = pDst->src[2];
  const ggml_tensor *vMask = pDst->src[3];
  const ggml_tensor *vSinks = pDst->src[4];

  GGML_TENSOR_LOCALS(int64_t, neq, vQ, ne)
  GGML_TENSOR_LOCALS(size_t, nbq, vQ, nb)
  GGML_TENSOR_LOCALS(int64_t, nek, vK, ne)
  GGML_TENSOR_LOCALS(size_t, nbk, vK, nb)
  GGML_TENSOR_LOCALS(int64_t, nev, v_, ne)
  GGML_TENSOR_LOCALS(size_t, nbv, v_, nb)
  GGML_TENSOR_LOCALS(int64_t, ne, pDst, ne)
  GGML_TENSOR_LOCALS(size_t, nb, pDst, nb)

  const int64_t cDk = nek0;
  const int64_t cDv = nev0;
  const int64_t cN = neq1;

  GGML_ASSERT(ne0 == cDv);
  GGML_ASSERT(ne2 == cN);

  // input tensor rows must be contiguous
  GGML_ASSERT(nbq0 == ggml_type_size(vQ->type));
  GGML_ASSERT(nbk0 == ggml_type_size(vK->type));
  GGML_ASSERT(nbv0 == ggml_type_size(v_->type));

  GGML_ASSERT(neq0 == cDk);
  GGML_ASSERT(nek0 == cDk);
  GGML_ASSERT(nev0 == cDv);

  GGML_ASSERT(neq1 == cN);

  // dst cannot be transposed or permuted
  GGML_ASSERT(nb0 == sizeof(float));
  GGML_ASSERT(nb0 <= nb1);
  GGML_ASSERT(nb1 <= nb2);
  GGML_ASSERT(nb2 <= nb3);

  GGML_ASSERT(vK->type == v_->type);
  const ggml_type cKvType = vK->type;

  // broadcast factors
  const int64_t cRk2 = neq2 / nek2;
  const int64_t cRk3 = neq3 / nek3;

  const int64_t cRv2 = neq2 / nev2;
  const int64_t cRv3 = neq3 / nev3;

  float vScale = 1.0f;
  float vMaxBias = 0.0f;
  float vLogitSoftcap = 0.0f;

  memcpy(&vScale, (float *)pDst->op_params + 0, sizeof(float));
  memcpy(&vMaxBias, (float *)pDst->op_params + 1, sizeof(float));
  memcpy(&vLogitSoftcap, (float *)pDst->op_params + 2, sizeof(float));

  if (vLogitSoftcap != 0) {
    vScale /= vLogitSoftcap;
  }

  const uint32_t cNHead = neq2;
  const uint32_t cNHeadLog2 = 1u << (uint32_t)floor(log2(cNHead));

  const float cM0 = powf(2.0f, -(vMaxBias) / cNHeadLog2);
  const float cM1 = powf(2.0f, -(vMaxBias / 2.0f) / cNHeadLog2);

  int vIth = pParams->ith;

  static constexpr int cQTileSz = ggml_fa_tile_config::Q;
  static constexpr int cKvTileSz = ggml_fa_tile_config::KV;

  int vIr = pIr0;
  while (vIr < pIr1) {
    // q indices for the start of this tile
    const int cIq3 = vIr / (neq2 * neq1);
    const int cIq2 = (vIr - cIq3 * neq2 * neq1) / neq1;
    const int cIq1 = (vIr - cIq3 * neq2 * neq1 - cIq2 * neq1);

    // Number of valid rows in this tile:
    // - limited by tile size (Q_TILE_SZ)
    // - limited by chunk boundary (ir1 - ir)
    // - limited by head boundary (neq1 - iq1) to avoid crossing into next head
    const int cTileRows = MIN(cQTileSz, MIN((int)(pIr1 - vIr), (int)(neq1 - cIq1)));
    GGML_ASSERT(cTileRows > 0);

    const uint32_t cH = cIq2; // head index
    const float cSlope =
        (vMaxBias > 0.0f) ? cH < cNHeadLog2 ? powf(cM0, cH + 1) : powf(cM1, 2 * (cH - cNHeadLog2) + 1) : 1.0f;

    float vS[cQTileSz];
    float vM[cQTileSz];

    for (int vI = 0; vI < cQTileSz; ++vI) {
      vS[vI] = 0.;
      vM[vI] = -INFINITY;
    }

    // Per-thread scratch layout:
    // Q_q:    Q_TILE_SZ * DK (converted Q tile — F32 for GEMM, KV type for scalar)
    // KQ:     Q_TILE_SZ * KV_TILE_SZ (attention scores in float)
    // mask:   Q_TILE_SZ * KV_TILE_SZ (mask in float)
    // VKQ32:  Q_TILE_SZ * DV (FP32 output accumulator)
    // V32:    KV_TILE_SZ * DV (F32 buffer for V tile)
    // K_f32:  KV_TILE_SZ * DK (F32 buffer for K tile — GEMM path)
    float *vBase = (float *)pParams->wdata + vIth * (cQTileSz * cDk + 2 * cQTileSz * cKvTileSz + cQTileSz * cDv +
                                                     cKvTileSz * cDv + cKvTileSz * cDk + CACHE_LINE_SIZE_F32);

    void *vQQ = vBase;
    float *vKq = (float *)((char *)vBase + cQTileSz * cDk * sizeof(float));
    float *vMask32 = vKq + cQTileSz * cKvTileSz;
    float *vVkQ32 = vMask32 + cQTileSz * cKvTileSz;
    float *vV32 = vVkQ32 + cQTileSz * cDv;
    float *vKF32 = vV32 + cKvTileSz * cDv;

    memset(vVkQ32, 0, cQTileSz * cDv * sizeof(float));
    memset(vMask32, 0, cQTileSz * cKvTileSz * sizeof(float));

    // k indices
    const int cIk3 = cIq3 / cRk3;
    const int cIk2 = cIq2 / cRk2;

    // v indices
    const int cIv3 = cIq3 / cRv3;
    const int cIv2 = cIq2 / cRv2;

    {
      float *vQF32 = (float *)vQQ;
      for (int vTq = 0; vTq < cTileRows; vTq++) {
        const float *vPq = (const float *)((char *)vQ->data + ((cIq1 + vTq) * nbq1 + cIq2 * nbq2 + cIq3 * nbq3));
        memcpy(vQF32 + vTq * cDk, vPq, cDk * sizeof(float));
      }
      for (int vTq = cTileRows; vTq < cQTileSz; vTq++) {
        memset(vQF32 + vTq * cDk, 0, cDk * sizeof(float));
      }
    }

    memset(vKF32, 0, cDk * cKvTileSz * sizeof(float));
    memset(vV32, 0, cKvTileSz * cDv * sizeof(float));

    for (int64_t vIc = 0; vIc < nek1; vIc += cKvTileSz) {
      const int cKvTile = (int)std::min((int64_t)cKvTileSz, nek1 - vIc);

      // skip the tile entirely if all the masks are -inf
      if (vMask) {
        bool vCanSkip = true;
        for (int vTq = 0; vTq < cTileRows; vTq++) {
          const ggml_fp16_t *vMpRow =
              (const ggml_fp16_t *)((const char *)vMask->data + (cIq1 + vTq) * vMask->nb[1] +
                                    (cIq2 % vMask->ne[2]) * vMask->nb[2] + (cIq3 % vMask->ne[3]) * vMask->nb[3]);
          for (int vTk = 0; vTk < cKvTile; vTk++) {
            vMask32[vTq * cKvTileSz + vTk] = cSlope * GGML_CPU_FP16_TO_FP32(vMpRow[vIc + vTk]);
            if (vMask32[vTq * cKvTileSz + vTk] != -INFINITY) {
              vCanSkip = false;
            }
          }
          // Pad remaining mask entries with -inf
          for (int vTk = cKvTile; vTk < cKvTileSz; vTk++) {
            vMask32[vTq * cKvTileSz + vTk] = -INFINITY;
          }
        }

        if (vCanSkip) {
          continue;
        }
      }

      // Pack K tile transposed: K_f32[dk][kv] so KV_TILE is contiguous (SIMD dim)
      // Zero-pad the last tile so the GEMM always operates on KV_TILE_SZ columns
      for (int vTk = 0; vTk < cKvTile; vTk++) {
        const char *vKData = (const char *)vK->data + (vIc + vTk) * nbk1 + cIk2 * nbk2 + cIk3 * nbk3;
        if (cKvType == GGML_TYPE_F16) {
          const ggml_fp16_t *vKF16 = (const ggml_fp16_t *)vKData;
          for (int64_t vDk = 0; vDk < cDk; vDk++) {
            vKF32[vDk * cKvTileSz + vTk] = GGML_CPU_FP16_TO_FP32(vKF16[vDk]);
          }
        } else {
          const float *vKF32Src = (const float *)vKData;
          for (int64_t vDk = 0; vDk < cDk; vDk++) {
            vKF32[vDk * cKvTileSz + vTk] = vKF32Src[vDk];
          }
        }
      }
      memset(vKq, 0, cQTileSz * cKvTileSz * sizeof(float));
      simd_gemm(vKq, (const float *)vQQ, vKF32, cQTileSz, cDk, cKvTileSz);
      ggml_vec_scale_f32(cQTileSz * cKvTileSz, vKq, vScale);

      if (vLogitSoftcap != 0.0f) {
        ggml_vec_tanh_f32(cQTileSz * cKvTileSz, vKq, vKq);
        ggml_vec_scale_f32(cQTileSz * cKvTileSz, vKq, vLogitSoftcap);
      }

      // Set padded KQ entries to -inf so softmax gives them zero weight
      if (cKvTile < cKvTileSz) {
        for (int vTq = 0; vTq < cQTileSz; vTq++) {
          for (int vTk = cKvTile; vTk < cKvTileSz; vTk++) {
            vKq[vTq * cKvTileSz + vTk] = -INFINITY;
          }
        }
      }

      if (vMask) {
        ggml_vec_add_f32(cTileRows * cKvTileSz, vKq, vKq, vMask32);
      }

      bool aSkip[cQTileSz] = {};

      for (int vTq = 0; vTq < cQTileSz; vTq++) {
        float *vKqRow = vKq + vTq * cKvTileSz;

        float vTileMax;
        ggml_vec_max_f32(cKvTileSz, &vTileMax, vKqRow);

        if (vTileMax == -INFINITY) {
          aSkip[vTq] = true;
          continue;
        }

        const float cMold = vM[vTq];
        const float cMnew = fmaxf(cMold, vTileMax);

        if (cMnew > cMold) {
          const float cMs = expf(cMold - cMnew);
          ggml_vec_scale_f32(cDv, vVkQ32 + vTq * cDv, cMs);
          vS[vTq] *= cMs;
        }
        vM[vTq] = cMnew;

        vS[vTq] += ggml_vec_soft_max_f32(cKvTileSz, vKqRow, vKqRow, cMnew);
      }

      // V accumulation: VKQ32 += softmax(KQ) * V
      // Pack V tile to contiguous F32, zero-padded
      for (int vTk = 0; vTk < cKvTile; vTk++) {
        const char *vVData = (const char *)v_->data + (vIc + vTk) * nbv1 + cIv2 * nbv2 + cIv3 * nbv3;
        if (cKvType == GGML_TYPE_F16) {
          ggml_cpu_fp16_to_fp32((const ggml_fp16_t *)vVData, vV32 + vTk * cDv, cDv);
        } else {
          memcpy(vV32 + vTk * cDv, vVData, cDv * sizeof(float));
        }
      }
      for (int vTq = 0; vTq < cQTileSz; vTq++) {
        if (aSkip[vTq]) {
          memset(vKq + vTq * cKvTileSz, 0, cKvTileSz * sizeof(float));
        }
      }
      simd_gemm(vVkQ32, vKq, vV32, cQTileSz, cKvTileSz, cDv);
    }

    // sinks (apply only to valid rows in the tile)
    if (vSinks) {
      const float cS = ((float *)((char *)vSinks->data))[cH];

      for (int vTq = 0; vTq < cTileRows; vTq++) {
        float vMs = 1.0f;
        float vVs = 1.0f;

        if (cS > vM[vTq]) {
          vMs = expf(vM[vTq] - cS);
          ggml_vec_scale_f32(cDv, vVkQ32 + vTq * cDv, vMs);
        } else {
          vVs = expf(cS - vM[vTq]);
        }

        vS[vTq] = vS[vTq] * vMs + vVs;
      }
    }

    for (int vTq = 0; vTq < cTileRows; vTq++) {
      // V /= S
      const float cSInv = vS[vTq] == 0.0f ? 0.0f : 1.0f / vS[vTq];
      ggml_vec_scale_f32(cDv, vVkQ32 + vTq * cDv, cSInv);

      // dst indices
      const int cI1 = cIq1 + vTq;
      const int cI2 = cIq2;
      const int cI3 = cIq3;

      // permute(0, 2, 1, 3)
      memcpy((char *)pDst->data + (cI3 * ne2 * ne1 + cI2 + cI1 * ne1) * nb1, vVkQ32 + vTq * cDv, nb1);
    }

    vIr += cTileRows;
  }
}

// Reduction function: combines partial results across KV chunks
// Partials layout in wdata: [n_q_heads][n_chunks][2 + DV]
static void fGgmlFlashAttnExtReducePartials(const ggml_compute_params *pParams, ggml_tensor *pDst,
                                            const int64_t pNChunks, const int64_t pChunkSize) {

  const ggml_tensor *vQ = pDst->src[0];
  const ggml_tensor *vK = pDst->src[1];
  const ggml_tensor *v_ = pDst->src[2];

  const int64_t cDk = vK->ne[0];
  const int64_t cDv = v_->ne[0];
  const int64_t cNek1 = vK->ne[1];
  const int64_t cNQHeads = vQ->ne[2];

  const int cIth = pParams->ith;
  const int cNth = pParams->nth;

  const int64_t cWdataPerThread = cDk + 2 * cDv + CACHE_LINE_SIZE_F32;
  float *vThreadWdata = (float *)pParams->wdata + cIth * cWdataPerThread;

  const int64_t cPartialsOffset = cNth * (cDk + 2 * cDv + CACHE_LINE_SIZE_F32);
  const int64_t cPartialSize = 2 + cDv;
  const float *vPartialsBase = (const float *)pParams->wdata + cPartialsOffset;

  // Output layout
  const int64_t cNe1 = pDst->ne[1];
  const int64_t cNe2 = pDst->ne[2];
  const size_t cNb1 = pDst->nb[1];

  // Each thread reduces a subset of query heads
  for (int64_t vQHead = cIth; vQHead < cNQHeads; vQHead += cNth) {
    float vMFinal = -INFINITY;
    float vSFinal = 0.0f;
    float *vVkqFinal = vThreadWdata;
    memset(vVkqFinal, 0, cDv * sizeof(float));

    // Combine partials from all chunks
    for (int64_t vChunkIdx = 0; vChunkIdx < pNChunks; ++vChunkIdx) {
      const int64_t cIcStart = vChunkIdx * pChunkSize;
      if (cIcStart >= cNek1)
        continue;

      const float *vPartial = vPartialsBase + (vQHead * pNChunks + vChunkIdx) * cPartialSize;
      const float cMChunk = vPartial[0];
      const float cSChunk = vPartial[1];
      const float *vVkqChunk = vPartial + 2;

      if (cSChunk == 0.0f)
        continue;

      const float cMNew = fmaxf(vMFinal, cMChunk);
      const float cScaleOld = expf(vMFinal - cMNew);
      const float cScaleNew = expf(cMChunk - cMNew);

      for (int64_t vD = 0; vD < cDv; ++vD) {
        vVkqFinal[vD] = vVkqFinal[vD] * cScaleOld + vVkqChunk[vD] * cScaleNew;
      }
      vSFinal = vSFinal * cScaleOld + cSChunk * cScaleNew;
      vMFinal = cMNew;
    }

    // Normalize and write to output
    if (vSFinal != 0.0f) {
      const float cSInv = 1.0f / vSFinal;
      ggml_vec_scale_f32(cDv, vVkqFinal, cSInv);
    }
    // iq1=0, iq3=0 for decode
    memcpy((char *)pDst->data + (0 * cNe2 * cNe1 + vQHead + 0 * cNe1) * cNb1, vVkqFinal, cNb1);
  }
}

static void fGgmlComputeForwardFlashAttnExtF16(const ggml_compute_params *pParams, ggml_tensor *pDst) {

  const ggml_tensor *vQ = pDst->src[0];
  const ggml_tensor *vK = pDst->src[1];
  const ggml_tensor *v_ = pDst->src[2];

  GGML_TENSOR_LOCALS(int64_t, neq, vQ, ne)
  GGML_TENSOR_LOCALS(size_t, nbq, vQ, nb)
  GGML_TENSOR_LOCALS(int64_t, nek, vK, ne)
  GGML_TENSOR_LOCALS(size_t, nbk, vK, nb)
  GGML_TENSOR_LOCALS(int64_t, nev, v_, ne)
  GGML_TENSOR_LOCALS(size_t, nbv, v_, nb)
  GGML_TENSOR_LOCALS(int64_t, ne, pDst, ne)
  GGML_TENSOR_LOCALS(size_t, nb, pDst, nb)

  const int64_t cDk = nek0;
  const int64_t cDv = nev0;
  const int64_t cN = neq1;

  GGML_ASSERT(ne0 == cDv);
  GGML_ASSERT(ne2 == cN);

  // input tensor rows must be contiguous
  GGML_ASSERT(nbq0 == ggml_type_size(vQ->type));
  GGML_ASSERT(nbk0 == ggml_type_size(vK->type));
  GGML_ASSERT(nbv0 == ggml_type_size(v_->type));

  GGML_ASSERT(neq0 == cDk);
  GGML_ASSERT(nek0 == cDk);
  GGML_ASSERT(nev0 == cDv);

  GGML_ASSERT(neq1 == cN);

  // dst cannot be transposed or permuted
  GGML_ASSERT(nb0 == sizeof(float));
  GGML_ASSERT(nb0 <= nb1);
  GGML_ASSERT(nb1 <= nb2);
  GGML_ASSERT(nb2 <= nb3);

  const int cIth = pParams->ith;
  const int cNth = pParams->nth;

  // When use_ref is set, force the vec-only reference implementation (no tiling, no KV-chunking)
  const bool cUseRef = pParams->use_ref;

  const bool cKvIsF32OrF16 = (vK->type == GGML_TYPE_F32 || vK->type == GGML_TYPE_F16);
  const bool cUseSplitKvPath = !cUseRef && (neq1 == 1 && neq3 == 1) && cKvIsF32OrF16 && (vK->type == v_->type) &&
                               vQ->type == GGML_TYPE_F32 && nek1 >= 512;

  if (cUseSplitKvPath) {
    const int64_t cChunkSize = (nek1 + cNth - 1) / cNth;

    // Partials buffer layout: [q_head][kv_chunk][M, S, VKQ]
    const int64_t cPartialSize = 2 + cDv;
    float *vPartialsBase = (float *)pParams->wdata + cNth * (cDk + 2 * cDv + CACHE_LINE_SIZE_F32);

    const int64_t cIcStart = cIth * cChunkSize;
    const int64_t cIcEnd = std::min(cIcStart + cChunkSize, nek1);

    const int64_t cPartialStride = cNth * cPartialSize;
    float *vChunkPartials = vPartialsBase + cIth * cPartialSize;

    if (cIcStart < nek1) {
      for (int64_t vQHead = 0; vQHead < neq2; vQHead++) {
        fGgmlComputeForwardFlashAttnExtF16OneChunk(pParams, pDst, vQHead, vQHead + 1, cIcStart, cIcEnd, vChunkPartials,
                                                   cPartialStride);
      }
    } else {
      for (int64_t vQHead = 0; vQHead < neq2; vQHead++) {
        float *vQPartials = vChunkPartials + vQHead * cPartialStride;
        vQPartials[0] = -INFINITY; // M
        vQPartials[1] = 0.0f;      // S
      }
    }

    ggml_barrier(pParams->threadpool);
    fGgmlFlashAttnExtReducePartials(pParams, pDst, cNth, cChunkSize);
  } else {

    // total rows in q
    const int64_t cNr = neq1 * neq2 * neq3;

    // disable for NUMA
    const bool cDisableChunking = ggml_is_numa();

    // 4x chunks per thread
    int vNthScaled = cNth * 4;
    int64_t vChunkSize = (cNr + vNthScaled - 1) / vNthScaled;
    int64_t vNchunk = (cNr + vChunkSize - 1) / vChunkSize;

    if (cNth == 1 || vNchunk < cNth || cDisableChunking) {
      vNchunk = cNth;
    }

    if (cIth == 0) {
      ggml_threadpool_chunk_set(pParams->threadpool, cNth);
    }

    ggml_barrier(pParams->threadpool);

    const int64_t cDr = (cNr + vNchunk - 1) / vNchunk;

    static constexpr int64_t cQTileSz = ggml_fa_tile_config::Q;
    bool vUseTiled =
        !cUseRef && (vQ->type == GGML_TYPE_F32 && cKvIsF32OrF16 && vK->type == v_->type && neq1 >= cQTileSz);
#if defined(GGML_SIMD) && !defined(__x86_64__) && !defined(_M_X64)
#if defined(__ARM_FEATURE_SVE)
    const int64_t f32_epr = svcntw();
#else
    const int64_t f32_epr = GGML_F32_EPR;
#endif
    vUseTiled &= (cDv % f32_epr == 0);
#endif
    int vCurrentChunk = cIth;

    while (vCurrentChunk < vNchunk) {
      const int64_t cIr0 = cDr * vCurrentChunk;
      const int64_t cIr1 = MIN(cIr0 + cDr, cNr);

      if (vUseTiled) {
        fGgmlComputeForwardFlashAttnExtTiled(pParams, pDst, cIr0, cIr1);
      } else {
        fGgmlComputeForwardFlashAttnExtF16OneChunk(pParams, pDst, cIr0, cIr1, 0, nek1, nullptr, 0);
      }

      vCurrentChunk = ggml_threadpool_chunk_add(pParams->threadpool, 1);
    }
  }
}

void ggml_compute_forward_flash_attn_ext(const ggml_compute_params *pParams, ggml_tensor *pDst) {
  switch (pDst->op_params[3]) {
  case GGML_PREC_DEFAULT:
  case GGML_PREC_F32: {
    // uses F32 accumulators
    fGgmlComputeForwardFlashAttnExtF16(pParams, pDst);
  } break;
  default: {
    GGML_ABORT("fatal error");
  }
  }
}

// ggml_compute_forward_flash_attn_back

static void fGgmlComputeForwardFlashAttnBackF32(const ggml_compute_params *pParams, const bool pMasked,
                                                ggml_tensor *pDst) {

  const ggml_tensor *vQ = pDst->src[0];
  const ggml_tensor *vK = pDst->src[1];
  const ggml_tensor *v_ = pDst->src[2];
  const ggml_tensor *vD = pDst->src[3];

  GGML_TENSOR_LOCALS(int64_t, neq, vQ, ne)
  GGML_TENSOR_LOCALS(size_t, nbq, vQ, nb)
  GGML_TENSOR_LOCALS(int64_t, nek, vK, ne)
  GGML_TENSOR_LOCALS(size_t, nbk, vK, nb)
  GGML_TENSOR_LOCALS(int64_t, nev, v_, ne)
  GGML_TENSOR_LOCALS(size_t, nbv, v_, nb)
  GGML_TENSOR_LOCALS(int64_t, ned, vD, ne)
  GGML_TENSOR_LOCALS(size_t, nbd, vD, nb)
  GGML_TENSOR_LOCALS(int64_t, ne, pDst, ne)
  GGML_TENSOR_LOCALS(size_t, nb, pDst, nb)

  const int cIth = pParams->ith;
  const int cNth = pParams->nth;

  const int64_t cD = neq0;
  const int64_t cN = neq1;
  const int64_t cP = nek1 - cN;
  const int64_t cM = cP + cN;

  const int cMup = ggml_up(cM, GGML_SOFT_MAX_UNROLL);
  const int cMxDm = MAX(cD, cMup);

  // GGML_ASSERT(ne0 == D);
  // GGML_ASSERT(ne1 == N);
  GGML_ASSERT(cP >= 0);

  GGML_ASSERT(nbq0 == sizeof(float));
  GGML_ASSERT(nbk0 == sizeof(float));
  GGML_ASSERT(nbv0 == sizeof(float));

  GGML_ASSERT(neq0 == cD);
  GGML_ASSERT(nek0 == cD);
  GGML_ASSERT(nev1 == cD);
  GGML_ASSERT(ned0 == cD);

  GGML_ASSERT(neq1 == cN);
  GGML_ASSERT(nek1 == cN + cP);
  GGML_ASSERT(nev1 == cD);
  GGML_ASSERT(ned1 == cN);

  // dst cannot be transposed or permuted
  GGML_ASSERT(nb0 == sizeof(float));
  GGML_ASSERT(nb0 <= nb1);
  GGML_ASSERT(nb1 <= nb2);
  GGML_ASSERT(nb2 <= nb3);

  if (cIth == 0) {
    memset(pDst->data, 0, nb0 * ne0 * ne1 * ne2 * ne3);
  }
  ggml_barrier(pParams->threadpool);

  const int64_t cElemQ = ggml_nelements(vQ);
  const int64_t cElemK = ggml_nelements(vK);

  ggml_type vResultType = pDst->type;
  GGML_ASSERT(ggml_blck_size(vResultType) == 1);
  const size_t cTsize = ggml_type_size(vResultType);

  const size_t cOffsQ = 0;
  const size_t cOffsK = cOffsQ + GGML_PAD(cElemQ * cTsize, GGML_MEM_ALIGN);
  const size_t cOffsV = cOffsK + GGML_PAD(cElemK * cTsize, GGML_MEM_ALIGN);

  void *vGradQ = (char *)pDst->data;
  void *vGradK = (char *)pDst->data + cOffsK;
  void *vGradV = (char *)pDst->data + cOffsV;

  const size_t cNbgq1 = nb0 * neq0;
  const size_t cNbgq2 = nb0 * neq0 * neq1;
  const size_t cNbgq3 = nb0 * neq0 * neq1 * neq2;

  const size_t cNbgk1 = nb0 * nek0;
  const size_t cNbgk2 = nb0 * nek0 * nek1;
  const size_t cNbgk3 = nb0 * nek0 * nek1 * neq2;

  const size_t cNbgv1 = nb0 * nev0;
  const size_t cNbgv2 = nb0 * nev0 * nev1;
  const size_t cNbgv3 = nb0 * nev0 * nev1 * neq2;

  // parallelize by k rows using ggml_vec_dot_f32

  // total rows in k
  const int cNr = nek2 * nek3;

  // rows per thread
  const int cDr = (cNr + cNth - 1) / cNth;

  // row range for this thread
  const int cIr0 = cDr * cIth;
  const int cIr1 = MIN(cIr0 + cDr, cNr);

  const float cScale = 1.0f / sqrtf(cD);

  // printf("P=%d N=%d D=%d ir0=%d ir1=%d scale = %f\n", P, N, D, ir0, ir1, scale);

  // how often k2 (and v2) is repeated in q2
  int vNrep = neq2 / nek2;

  for (int vIr = cIr0; vIr < cIr1; ++vIr) {
    // q indices
    const int cIk3 = vIr / (nek2);
    const int cIk2 = vIr - cIk3 * nek2;

    const int cIq3 = cIk3;
    const int cId3 = cIk3;
    const int cIv3 = cIk3;
    const int cIv2 = cIk2;

    for (int vIrep = 0; vIrep < vNrep; ++vIrep) {
      const int cIq2 = cIk2 + vIrep * nek2;
      const int cId2 = cIq2;

      // (ik2 + irep*nek2) % nek2 == ik2
      for (int vIq1 = 0; vIq1 < neq1; ++vIq1) {
        const int cId1 = vIq1;

        // not sure about CACHE_LINE_SIZE_F32..
        // - maybe it must not be multiplied by 2 and excluded from .. in SM 1*(..) offset?
        float *vS =
            (float *)pParams->wdata + cIth * 2 * (cMxDm + CACHE_LINE_SIZE_F32) + 0 * (cMxDm + CACHE_LINE_SIZE_F32);
        float *vSm =
            (float *)pParams->wdata + cIth * 2 * (cMxDm + CACHE_LINE_SIZE_F32) + 1 * (cMxDm + CACHE_LINE_SIZE_F32);

        for (int vI = cM; vI < cMup; ++vI) {
          vS[vI] = -INFINITY;
        }

        const int64_t cMaskedBegin = pMasked ? (cP + vIq1 + 1) : cM;
        for (int64_t vIc = 0; vIc < cMaskedBegin; ++vIc) {
          // k indices
          const int cIk1 = vIc;

          // S indices
          const int cI1 = cIk1;

          ggml_vec_dot_f32(neq0, vS + cI1, 0, (float *)((char *)vK->data + (cIk1 * nbk1 + cIk2 * nbk2 + cIk3 * nbk3)),
                           0, (float *)((char *)vQ->data + (vIq1 * nbq1 + cIq2 * nbq2 + cIq3 * nbq3)), 0, 1);
        }

        // scale
        ggml_vec_scale_f32(cMaskedBegin, vS, cScale);

        for (int64_t vI = cMaskedBegin; vI < cM; vI++) {
          vS[vI] = -INFINITY;
        }

        // softmax
        // exclude known -INF S[..] values from max and loop
        // dont forget to set their SM values to zero
        {
          float vMax = -INFINITY;
          ggml_vec_max_f32(cMaskedBegin, &vMax, vS);

          ggml_float vSum = 0.0;
          {
#ifdef GGML_SOFT_MAX_ACCELERATE
            vMax = -vMax;
            vDSP_vsadd(vSm, 1, &vMax, vSm, 1, cMup);
            vvexpf(vSm, vSm, &cMup);
            ggml_vec_sum_f32(cMup, &vSum, vSm);
#else
            vSum = ggml_vec_soft_max_f32(cMup, vSm, vS, vMax);
#endif
          }

          assert(vSum > 0.0);

          vSum = 1.0 / vSum;
          ggml_vec_scale_f32(cMaskedBegin, vSm, vSum);
        }

        // step-by-step explanation
        {
          // forward-process                    shape      grads from backward process
          // parallel_for ik2,ik3:
          //  for irep:
          //   iq2 = ik2 + irep*nek2
          //   k[:D,:M,:,:]                     [D,M,:,:]  grad[k][:D,:M,ik2,ik3]  += grad[kcur]
          //   q[:D,:N,:,:]                     [D,N,:,:]  grad[q][:D,iq1,iq2,iq3] += grad[qcur]
          //   v[:M,:D,:,:]                     [M,D,:,:]  grad[v][:M,:D,iv2,iv3]  += grad[vcur]
          //   for iq1:
          //    kcur   = k[:D,:M,ik2,ik3]       [D,M,1,1]  grad[kcur] = grad[S1].T @ qcur
          //    qcur   = q[:D,iq1,iq2,iq3]      [D,1,1,1]  grad[qcur] = grad[S1]   @ kcur
          //    vcur   = v[:M,:D,iv2,iv3]       [M,D,1,1]  grad[vcur] = grad[S5].T @ S4
          //    S0     = -Inf                   [D,1,1,1]
          //   ~S1[i]  = dot(kcur[:D,i], qcur)
          //    S1     = qcur @ kcur.T          [M,1,1,1]  grad[S1]   = grad[S2] * scale
          //    S2     = S1 * scale             [M,1,1,1]  grad[S2]   = diag_mask_zero(grad[S3], P)
          //    S3     = diag_mask_inf(S2, P)   [M,1,1,1]  grad[S3]   = S4 * (grad[S4] - dot(S4, grad[S4]))
          //    S4     = softmax(S3)            [M,1,1,1]  grad[S4]   = grad[S5] @ vcur
          //   ~S5[i]  = dot(vcur[:,i], S4)
          //    S5     = S4 @ vcur.T            [D,1,1,1]  grad[S5]   = d[:D,id1,id2,id3]
          //   ~dst[i,iq1,iq2,iq3]  = S5[i]              ^
          //    dst[:D,iq1,iq2,iq3] = S5                 | grad[dst[:D,iq1,iq2,iq3]] = d[:D,id1,id2,id3]
          // dst                               backward-/ grad[dst]                 = d
          //
          // output gradients with their dependencies:
          //
          // grad[kcur] = grad[S1].T @ qcur
          // grad[S1]   = diag_mask_zero(grad[S3], P) * scale
          // grad[S3]   = S4 * (grad[S4] - dot(S4, grad[S4]))
          // grad[S4]   = grad[S5] @ vcur
          // grad[S4]   = d[:D,id1,id2,id3] @ vcur
          // grad[qcur] = grad[S1]   @ kcur
          // grad[vcur] = grad[S5].T @ S4
          // grad[vcur] = d[:D,id1,id2,id3].T @ S4
          //
          // in post-order:
          //
          // S1         = qcur @ kcur.T
          // S2         = S1 * scale
          // S3         = diag_mask_inf(S2, P)
          // S4         = softmax(S3)
          // grad[S4]   = d[:D,id1,id2,id3] @ vcur
          // grad[S3]   = S4 * (grad[S4] - dot(S4, grad[S4]))
          // grad[S1]   = diag_mask_zero(grad[S3], P) * scale
          // grad[qcur] = grad[S1]   @ kcur
          // grad[kcur] = grad[S1].T @ qcur
          // grad[vcur] = d[:D,id1,id2,id3].T @ S4
          //
          // using less variables (SM=S4):
          //
          // S             = diag_mask_inf(qcur @ kcur.T * scale, P)
          // SM            = softmax(S)
          // S             = d[:D,iq1,iq2,iq3] @ vcur
          // dot_SM_gradSM = dot(SM, S)
          // S             = SM * (S - dot(SM, S))
          // S             = diag_mask_zero(S, P) * scale
          //
          // grad[q][:D,iq1,iq2,iq3] += S   @ kcur
          // grad[k][:D,:M,ik2,ik3]  += S.T @ qcur
          // grad[v][:M,:D,iv2,iv3]  += d[:D,id1,id2,id3].T @ SM
        }

        // S = gradSM = d[:D,id1,id2,id3] @ vcur[:,:,iv2,iv3]
        // S = d[:D,id1,id2,id3] @ vcur[:,:,iv2,iv3]
        // for ic:
        //   S[:M] += vcur[:M,ic,iv2,iv3] * d[ic,id1,id2,id3]
        // exclude known future zero S[..] values from operation
        ggml_vec_set_f32(cMaskedBegin, vS, 0);
        for (int64_t vIc = 0; vIc < cD; ++vIc) {
          ggml_vec_mad_f32(cMaskedBegin, vS, (float *)((char *)v_->data + (vIc * nbv1 + cIv2 * nbv2 + cIv3 * nbv3)),
                           *(float *)((char *)vD->data + (vIc * nbd0 + cId1 * nbd1 + cId2 * nbd2 + cId3 * nbd3)));
        }

        // S = SM * (S - dot(SM, S))
        float vDotSmGradSm = 0;
        ggml_vec_dot_f32(cMaskedBegin, &vDotSmGradSm, 0, vSm, 0, vS, 0, 1);
        ggml_vec_acc1_f32(cM, vS, -vDotSmGradSm);
        ggml_vec_mul_f32(cMaskedBegin, vS, vS, vSm);

        // S = diag_mask_zero(S, P) * scale
        // already done by above ggml_vec_set_f32

        // exclude known zero S[..] values from operation
        ggml_vec_scale_f32(cMaskedBegin, vS, cScale);

        // S    shape [M,1]
        // SM   shape [M,1]
        // kcur shape [D,M]
        // qcur shape [D,1]
        // vcur shape [M,D]

        // grad[q][:D,iq1,iq2,iq3] += S @ kcur
        // grad[q][:D,iq1,iq2,iq3] += shape[M,1] @ shape[D,M]
        // for ic:
        //  grad[q][:D,iq1,iq2,iq3] += S[ic] * kcur[:D,ic,ik2,ik3]
        // exclude known zero S[..] values from loop
        for (int64_t vIc = 0; vIc < cMaskedBegin; ++vIc) {
          ggml_vec_mad_f32(cD, (float *)((char *)vGradQ + (vIq1 * cNbgq1 + cIq2 * cNbgq2 + cIq3 * cNbgq3)),
                           (float *)((char *)vK->data + (vIc * nbk1 + cIk2 * nbk2 + cIk3 * nbk3)), vS[vIc]);
        }

        // grad[k][:D,:M,iq2,iq3] += S.T @ qcur
        // for ic:
        //  grad[k][:D,ic,iq2,iq3] += S.T[0,ic] * qcur[:D,0]
        //  grad[k][:D,ic,iq2,iq3] += S[ic]     * qcur[:D,0]
        // exclude known zero S[..] values from loop
        for (int64_t vIc = 0; vIc < cMaskedBegin; ++vIc) {
          ggml_vec_mad_f32(cD, (float *)((char *)vGradK + (vIc * cNbgk1 + cIk2 * cNbgk2 + cIk3 * cNbgk3)),
                           (float *)((char *)vQ->data + (vIq1 * nbq1 + cIq2 * nbq2 + cIq3 * nbq3)), vS[vIc]);
        }

        // grad[v][:M,:D,iv2,iv3] += d[:D,id1,id2,id3].T       @ SM
        // for ic:
        //  grad[v][:M,ic,iv2,iv3] += d[:D,id1,id2,id3].T[0,ic] * SM[:M]
        //  grad[v][:M,ic,iv2,iv3] += d[ic,id1,id2,id3]         * SM[:M]
        // exclude known zero SM[..] values from mad
        for (int64_t vIc = 0; vIc < cD; ++vIc) {
          ggml_vec_mad_f32(cMaskedBegin, (float *)((char *)vGradV + (vIc * cNbgv1 + cIv2 * cNbgv2 + cIv3 * cNbgv3)),
                           vSm, *(float *)((char *)vD->data + (vIc * nbd0 + cId1 * nbd1 + cId2 * nbd2 + cId3 * nbd3)));
        }
      }
    }
  }
}

void ggml_compute_forward_flash_attn_back(const ggml_compute_params *pParams, const bool pMasked, ggml_tensor *pDst) {

  const ggml_tensor *vQ = pDst->src[0];

  switch (vQ->type) {
  case GGML_TYPE_F32: {
    fGgmlComputeForwardFlashAttnBackF32(pParams, pMasked, pDst);
  } break;
  default: {
    GGML_ABORT("fatal error");
  }
  }
}

// ggml_compute_forward_ssm_conv

static void fGgmlComputeForwardSsmConvF32(const ggml_compute_params *pParams, ggml_tensor *pDst) {
  const ggml_tensor *vSrc0 = pDst->src[0]; // conv_x
  const ggml_tensor *vSrc1 = pDst->src[1]; // conv1d.weight

  const int cIth = pParams->ith;
  const int cNth = pParams->nth;

  const int cNc = vSrc1->ne[0];  // d_conv
  const int cNcs = vSrc0->ne[0]; // d_conv - 1 + n_t
  const int cNr = vSrc0->ne[1];  // d_inner
  const int cNT = pDst->ne[1];   // tokens per sequence
  const int cNS = pDst->ne[2];   // number of sequences in the batch

  GGML_ASSERT(pDst->ne[0] == cNr);
  GGML_ASSERT(vSrc0->nb[0] == sizeof(float));
  GGML_ASSERT(vSrc1->nb[0] == sizeof(float));
  GGML_ASSERT(vSrc0->nb[1] == vSrc0->ne[0] * sizeof(float));

  // rows per thread
  const int cDr = (cNr + cNth - 1) / cNth;

  // row range for this thread
  const int cIr0 = cDr * cIth;
  const int cIr1 = MIN(cIr0 + cDr, cNr);
  const int cIr = cIr1 - cIr0;

  for (int vI3 = 0; vI3 < cNS; ++vI3) {
    for (int vI2 = 0; vI2 < cNT; ++vI2) {
      // {d_conv - 1 + n_t, d_inner, n_seqs}
      // sliding window
      const float *vS = (const float *)((const char *)vSrc0->data + cIr0 * (vSrc0->nb[1]) + vI2 * (vSrc0->nb[0]) +
                                        vI3 * (vSrc0->nb[2]));                              // {d_conv, d_inner, n_s}
      const float *vC = (const float *)((const char *)vSrc1->data + cIr0 * (vSrc1->nb[1])); // {d_conv, d_inner}
      float *vX = (float *)((char *)pDst->data + cIr0 * (pDst->nb[0]) + vI2 * (pDst->nb[1]) +
                            vI3 * (pDst->nb[2])); // {d_inner, n_t, n_s}

      // TODO: transpose the output for smaller strides for big batches?
      // d_inner
      for (int vI1 = 0; vI1 < cIr; ++vI1) {
        // rowwise dot product
        // NOTE: not using ggml_vec_dot_f32, because its sum is in double precision
        float vSumf = 0.0f;

        // d_conv
        for (int vI0 = 0; vI0 < cNc; ++vI0) {
          vSumf += vS[vI0 + vI1 * cNcs] * vC[vI0 + vI1 * cNc];
        }
        vX[vI1] = vSumf;
      }
    }
  }
}

void ggml_compute_forward_ssm_conv(const ggml_compute_params *pParams, ggml_tensor *pDst) {
  switch (pDst->src[0]->type) {
  case GGML_TYPE_F32: {
    fGgmlComputeForwardSsmConvF32(pParams, pDst);
  } break;
  default: {
    GGML_ABORT("fatal error");
  }
  }
}

// ggml_compute_forward_ssm_scan

static void fGgmlComputeForwardSsmScanF32(const ggml_compute_params *pParams, ggml_tensor *pDst) {
  const ggml_tensor *vSrc0 = pDst->src[0]; // s  {d_state, dim, n_head, n_seqs+}
  const ggml_tensor *vSrc1 = pDst->src[1]; // x  {dim, n_head, n_seq_tokens, n_seqs}
  const ggml_tensor *vSrc2 = pDst->src[2]; // dt {n_head, n_seq_tokens, n_seqs}
  const ggml_tensor *vSrc3 = pDst->src[3]; // A  {d_state, n_head} or {1, n_head}
  const ggml_tensor *vSrc4 = pDst->src[4]; // B  {d_state, n_group, n_seq_tokens, n_seqs}
  const ggml_tensor *vSrc5 = pDst->src[5]; // C  {d_state, n_group, n_seq_tokens, n_seqs}
  const ggml_tensor *vSrc6 = pDst->src[6]; // ids {n_seqs}

  const int cIth = pParams->ith;
  const int cNth = pParams->nth;

  const int64_t cNc = vSrc0->ne[0]; // d_state
  const int64_t cNr = vSrc0->ne[1]; // dim
  const int64_t cNh = vSrc1->ne[1]; // n_head
  const int64_t cNg = vSrc4->ne[1];
  const int64_t cNt = vSrc1->ne[2]; // number of tokens per sequence
  const int64_t cNs = vSrc1->ne[3]; // number of sequences in the batch

  // can't use ggml_nbytes because src1 is not necessarily contiguous
  const int64_t cSOff = ggml_nelements(vSrc1) * ggml_element_size(vSrc1);

  GGML_ASSERT(ggml_nelements(vSrc1) + cNc * cNr * cNh * cNs == ggml_nelements(pDst));
  GGML_ASSERT(vSrc0->nb[0] == sizeof(float));
  GGML_ASSERT(vSrc1->nb[0] == sizeof(float));
  GGML_ASSERT(vSrc2->nb[0] == sizeof(float));
  GGML_ASSERT(vSrc3->nb[0] == sizeof(float));
  GGML_ASSERT(vSrc4->nb[0] == sizeof(float));
  GGML_ASSERT(vSrc5->nb[0] == sizeof(float));
  GGML_ASSERT(vSrc6->nb[0] == sizeof(int32_t));
  GGML_ASSERT(cNh % cNg == 0);

  // heads per thread
  const int cDh = (cNh + cNth - 1) / cNth;

  // head range for this thread
  const int cIh0 = cDh * cIth;
  const int cIh1 = MIN(cIh0 + cDh, cNh);

  const int32_t *vIds = (const int32_t *)vSrc6->data;

  for (int vI3 = 0; vI3 < cNs; ++vI3) {
    const float *vS0 =
        (const float *)((const char *)vSrc0->data + vIds[vI3] * (vSrc0->nb[3])); // {d_state, dim, nh, ns}
    float *vS = (float *)((char *)pDst->data + vI3 * (vSrc0->nb[3]) + cSOff);    // {d_state, dim, nh, ns}

    for (int vI2 = 0; vI2 < cNt; ++vI2) {
      const float *vX =
          (const float *)((const char *)vSrc1->data + vI2 * (vSrc1->nb[2]) + vI3 * (vSrc1->nb[3])); // {dim, nh, nt, ns}
      const float *vDt =
          (const float *)((const char *)vSrc2->data + vI2 * (vSrc2->nb[1]) + vI3 * (vSrc2->nb[2])); // {nh, nt, ns}
      const float *A = (const float *)((const char *)vSrc3->data); // {d_state, nh} or {1, nh}
      const float *B = (const float *)((const char *)vSrc4->data + vI2 * (vSrc4->nb[2]) +
                                       vI3 * (vSrc4->nb[3])); // {d_state, ng, nt, ns}
      const float *C = (const float *)((const char *)vSrc5->data + vI2 * (vSrc5->nb[2]) +
                                       vI3 * (vSrc5->nb[3])); // {d_state, ng, nt, ns}
      float *vY = (float *)((char *)pDst->data + vI2 * (cNh * cNr * sizeof(float)) +
                            vI3 * (cNt * cNh * cNr * sizeof(float))); // {dim, nh, nt, ns}

      if (vSrc3->ne[0] == 1) {
        // Mamba-2 has a scalar decay factor per head; dA can be outside the state-wise loop

        // n_head
        for (int vH = cIh0; vH < cIh1; ++vH) {
          // ref:
          // https://github.com/state-spaces/mamba/blob/62db608da60f6fc790b8ed9f4b3225e95ca15fde/mamba_ssm/ops/triton/softplus.py#L16
          const float cDtSoftPlus = ggml_compute_softplus_f32(vDt[vH]);
          const float cDA = expf(cDtSoftPlus * A[vH]);
          const int cG = vH / (cNh / cNg); // repeat_interleave

          // dim
          for (int vI1 = 0; vI1 < cNr; ++vI1) {
            const int cIi = vI1 + vH * cNr;
            const float cXDt = vX[cIi] * cDtSoftPlus;
            float vSumf = 0.0f;
#if defined(GGML_SIMD)
#if defined(__ARM_FEATURE_SVE)
            const int ggml_f32_epr = svcntw();
            const int ggml_f32_step = 1 * ggml_f32_epr;

            const int cNp = (cNc & ~(ggml_f32_step - 1));

            GGML_F32_VEC vSum = GGML_F32_VEC_ZERO;

            GGML_F32_VEC vAdA = GGML_F32_VEC_SET1(cDA);
            GGML_F32_VEC vAxdt = GGML_F32_VEC_SET1(cXDt);

            for (int i = 0; i < cNp; i += ggml_f32_step) {
              // TODO: maybe unroll more?
              for (int vJ = 0; vJ < 1; vJ++) {
                GGML_F32_VEC t0 = GGML_F32_VEC_LOAD(vS0 + i + vJ * ggml_f32_epr + cIi * cNc);
                GGML_F32_VEC t1 = GGML_F32_VEC_LOAD(B + i + vJ * ggml_f32_epr + cG * cNc);
                GGML_F32_VEC t2 = GGML_F32_VEC_LOAD(C + i + vJ * ggml_f32_epr + cG * cNc);

                t0 = GGML_F32_VEC_MUL(t0, vAdA);
                t1 = GGML_F32_VEC_MUL(t1, vAxdt);

                t0 = GGML_F32_VEC_ADD(t0, t1);

                vSum = GGML_F32_VEC_FMA(vSum, t0, t2);

                GGML_F32_VEC_STORE(vS + i + vJ * ggml_f32_epr + cIi * cNc, t0);
              }
            }

            vSumf = GGML_F32xt_REDUCE_ONE(vSum);
#elif defined(__riscv_v_intrinsic)
            // todo: RVV implementation
            const int cNp = 0;
#else
            const int cNp = (cNc & ~(GGML_F32_STEP - 1));

            GGML_F32_VEC vSum[GGML_F32_ARR] = {GGML_F32_VEC_ZERO};

            GGML_F32_VEC vAdA = GGML_F32_VEC_SET1(cDA);
            GGML_F32_VEC vAxdt = GGML_F32_VEC_SET1(cXDt);

            GGML_F32_VEC vAx[GGML_F32_ARR];
            GGML_F32_VEC vAy[GGML_F32_ARR];
            GGML_F32_VEC vAz[GGML_F32_ARR];

            for (int i = 0; i < cNp; i += GGML_F32_STEP) {
              for (int vJ = 0; vJ < GGML_F32_ARR; vJ++) {
                vAx[vJ] = GGML_F32_VEC_LOAD(vS0 + i + vJ * GGML_F32_EPR + cIi * cNc);
                vAy[vJ] = GGML_F32_VEC_LOAD(B + i + vJ * GGML_F32_EPR + cG * cNc);
                vAz[vJ] = GGML_F32_VEC_LOAD(C + i + vJ * GGML_F32_EPR + cG * cNc);

                vAx[vJ] = GGML_F32_VEC_MUL(vAx[vJ], vAdA);
                vAy[vJ] = GGML_F32_VEC_MUL(vAy[vJ], vAxdt);

                vAx[vJ] = GGML_F32_VEC_ADD(vAx[vJ], vAy[vJ]);

                vSum[vJ] = GGML_F32_VEC_FMA(vSum[vJ], vAx[vJ], vAz[vJ]);

                GGML_F32_VEC_STORE(vS + i + vJ * GGML_F32_EPR + cIi * cNc, vAx[vJ]);
              }
            }

            // reduce sum0..sum3 to sum0
            GGML_F32_VEC_REDUCE(vSumf, vSum);
#endif
#else
            const int cNp = 0;
#endif
            // d_state
            for (int vI0 = cNp; vI0 < cNc; ++vI0) {
              const int i = vI0 + cIi * cNc;
              const int cIg = vI0 + cG * cNc;
              // state = prev_state * dA + dB * x
              const float cState = (vS0[i] * cDA) + (B[cIg] * cXDt);
              // y = rowwise_dotprod(state, C)
              vSumf += cState * C[cIg];
              vS[i] = cState;
            }
            vY[cIi] = vSumf;
          }
        }
      } else {
        // Mamba-1 has an element-wise decay factor for the states

        // n_head
        for (int vH = cIh0; vH < cIh1; ++vH) {
          // ref:
          // https://github.com/state-spaces/mamba/blob/62db608da60f6fc790b8ed9f4b3225e95ca15fde/mamba_ssm/ops/triton/softplus.py#L16
          const float cDtSoftPlus = ggml_compute_softplus_f32(vDt[vH]);
          const int cG = vH / (cNh / cNg); // repeat_interleave

          // dim
          for (int vI1 = 0; vI1 < cNr; ++vI1) {
            const int cIi = vI1 + vH * cNr;
            const float cXDt = vX[cIi] * cDtSoftPlus;
#if defined(__ARM_FEATURE_SVE)
            svfloat32_t vx_dt = GGML_F32_VEC_SET1(cXDt);
            svfloat32_t vdt_soft_plus = GGML_F32_VEC_SET1(cDtSoftPlus);
            svfloat32_t r1_vector = GGML_F32_VEC_ZERO;

            // d_state
            // TODO: what happens when (d_state % svcntw()) != 0?
            for (int64_t k = 0; k < cNc; k += svcntw()) {
              svfloat32_t vA = GGML_F32_VEC_LOAD(&A[vH * cNc + k]);
              svfloat32_t vB = GGML_F32_VEC_LOAD(&B[k + cG * cNc]);
              svfloat32_t vC = GGML_F32_VEC_LOAD(&C[k + cG * cNc]);
              svfloat32_t vs0 = GGML_F32_VEC_LOAD(&vS0[cIi * cNc + k]);

              svfloat32_t t1 = GGML_F32_VEC_MUL(vdt_soft_plus, vA);
              t1 = exp_ps_sve(svptrue_b32(), t1);
              svfloat32_t t2 = GGML_F32_VEC_MUL(vx_dt, vB);

              vs0 = GGML_F32_VEC_FMA(t2, vs0, t1);
              r1_vector = GGML_F32_VEC_ADD(GGML_F32_VEC_MUL(vs0, vC), r1_vector);

              GGML_F32_VEC_STORE(&vS[cIi * cNc + k], vs0);
            }
            vY[cIi] = GGML_F32xt_REDUCE_ONE(r1_vector);
#else
            float vSumf = 0.0f;
            // NOTE: can't really use GGML_SIMD here because d_state is usually 16
            //       and also because expf is used within the loop.
            // d_state
            for (int vI0 = 0; vI0 < cNc; ++vI0) {
              const int i = vI0 + cIi * cNc;
              const int cIg = vI0 + cG * cNc;
              // state = prev_state * dA + dB * x
              const float cState = (vS0[i] * expf(cDtSoftPlus * A[vI0 + vH * cNc])) + (B[cIg] * cXDt);
              // y = rowwise_dotprod(state, C)
              vSumf += cState * C[cIg];
              vS[i] = cState;
            }
            vY[cIi] = vSumf;
#endif
          }
        }
      }
      // use the output as the source when it's not the first token-wise iteration
      vS0 = vS;
    }
  }
}

void ggml_compute_forward_ssm_scan(const ggml_compute_params *pParams, ggml_tensor *pDst) {
  switch (pDst->src[0]->type) {
  case GGML_TYPE_F32: {
    fGgmlComputeForwardSsmScanF32(pParams, pDst);
  } break;
  default: {
    GGML_ABORT("fatal error");
  }
  }
}

// ggml_compute_forward_win_part

static void fGgmlComputeForwardWinPartF32(const ggml_compute_params *pParams, ggml_tensor *pDst) {
  GGML_UNUSED(pParams);

  const ggml_tensor *vSrc0 = pDst->src[0];

  GGML_TENSOR_LOCALS(int64_t, ne0, vSrc0, ne)
  GGML_TENSOR_LOCALS(int64_t, ne, pDst, ne)

  const int32_t cNep0 = ((const int32_t *)(pDst->op_params))[0];
  const int32_t cNep1 = ((const int32_t *)(pDst->op_params))[1];
  const int32_t cW = ((const int32_t *)(pDst->op_params))[2];

  assert(ne00 == ne0);
  assert(ne3 == cNep0 * cNep1);

  // TODO: optimize / multi-thread
  for (int vPy = 0; vPy < cNep1; ++vPy) {
    for (int vPx = 0; vPx < cNep0; ++vPx) {
      const int64_t cI3 = vPy * cNep0 + vPx;
      for (int64_t vI2 = 0; vI2 < ne2; ++vI2) {
        for (int64_t vI1 = 0; vI1 < ne1; ++vI1) {
          for (int64_t vI0 = 0; vI0 < ne0; ++vI0) {
            const int64_t cI02 = vPy * cW + vI2;
            const int64_t cI01 = vPx * cW + vI1;
            const int64_t cI00 = vI0;

            const int64_t cI = cI3 * ne2 * ne1 * ne0 + vI2 * ne1 * ne0 + vI1 * ne0 + vI0;
            const int64_t cJ = cI02 * ne01 * ne00 + cI01 * ne00 + cI00;

            if (vPy * cW + vI2 >= ne02 || vPx * cW + vI1 >= ne01) {
              ((float *)pDst->data)[cI] = 0.0f;
            } else {
              ((float *)pDst->data)[cI] = ((float *)vSrc0->data)[cJ];
            }
          }
        }
      }
    }
  }
}

void ggml_compute_forward_win_part(const ggml_compute_params *pParams, ggml_tensor *pDst) {

  const ggml_tensor *vSrc0 = pDst->src[0];

  switch (vSrc0->type) {
  case GGML_TYPE_F32: {
    fGgmlComputeForwardWinPartF32(pParams, pDst);
  } break;
  default: {
    GGML_ABORT("fatal error");
  }
  }
}

// ggml_compute_forward_win_unpart

static void fGgmlComputeForwardWinUnpartF32(const ggml_compute_params *pParams, ggml_tensor *pDst) {
  GGML_UNUSED(pParams);

  const ggml_tensor *vSrc0 = pDst->src[0];

  GGML_TENSOR_LOCALS(int64_t, ne0, vSrc0, ne)
  GGML_TENSOR_LOCALS(int64_t, ne, pDst, ne)

  const int32_t cW = ((const int32_t *)(pDst->op_params))[0];

  // padding
  const int cPx = (cW - ne1 % cW) % cW;
  // const int py = (w - ne2%w)%w;

  const int cNpx = (cPx + ne1) / cW;
  // const int npy = (py + ne2)/w;

  assert(ne0 == ne00);

  // TODO: optimize / multi-thread
  for (int64_t vI2 = 0; vI2 < ne2; ++vI2) {
    for (int64_t vI1 = 0; vI1 < ne1; ++vI1) {
      for (int64_t vI0 = 0; vI0 < ne0; ++vI0) {
        const int cIp2 = vI2 / cW;
        const int cIp1 = vI1 / cW;

        const int64_t cI02 = vI2 % cW;
        const int64_t cI01 = vI1 % cW;
        const int64_t cI00 = vI0;

        const int64_t cI = (cIp2 * cNpx + cIp1) * ne02 * ne01 * ne00 + cI02 * ne01 * ne00 + cI01 * ne00 + cI00;
        const int64_t cJ = vI2 * ne1 * ne0 + vI1 * ne0 + vI0;

        ((float *)pDst->data)[cJ] = ((float *)vSrc0->data)[cI];
      }
    }
  }
}

void ggml_compute_forward_win_unpart(const ggml_compute_params *pParams, ggml_tensor *pDst) {

  const ggml_tensor *vSrc0 = pDst->src[0];

  switch (vSrc0->type) {
  case GGML_TYPE_F32: {
    fGgmlComputeForwardWinUnpartF32(pParams, pDst);
  } break;
  default: {
    GGML_ABORT("fatal error");
  }
  }
}

// ggml_compute_forward_unary

void ggml_compute_forward_unary(const ggml_compute_params *pParams, ggml_tensor *pDst) {

  const ggml_unary_op cOp = ggml_get_unary_op(pDst);

  switch (cOp) {
  case GGML_UNARY_OP_ABS: {
    ggml_compute_forward_abs(pParams, pDst);
  } break;
  case GGML_UNARY_OP_SGN: {
    ggml_compute_forward_sgn(pParams, pDst);
  } break;
  case GGML_UNARY_OP_NEG: {
    ggml_compute_forward_neg(pParams, pDst);
  } break;
  case GGML_UNARY_OP_STEP: {
    ggml_compute_forward_step(pParams, pDst);
  } break;
  case GGML_UNARY_OP_TANH: {
    ggml_compute_forward_tanh(pParams, pDst);
  } break;
  case GGML_UNARY_OP_ELU: {
    ggml_compute_forward_elu(pParams, pDst);
  } break;
  case GGML_UNARY_OP_RELU: {
    ggml_compute_forward_relu(pParams, pDst);
  } break;
  case GGML_UNARY_OP_SIGMOID: {
    ggml_compute_forward_sigmoid(pParams, pDst);
  } break;
  case GGML_UNARY_OP_GELU: {
    fGgmlComputeForwardGelu(pParams, pDst);
  } break;
  case GGML_UNARY_OP_GELU_ERF: {
    fGgmlComputeForwardGeluErf(pParams, pDst);
  } break;
  case GGML_UNARY_OP_GELU_QUICK: {
    fGgmlComputeForwardGeluQuick(pParams, pDst);
  } break;
  case GGML_UNARY_OP_SILU: {
    fGgmlComputeForwardSilu(pParams, pDst);
  } break;
  case GGML_UNARY_OP_HARDSWISH: {
    ggml_compute_forward_hardswish(pParams, pDst);
  } break;
  case GGML_UNARY_OP_HARDSIGMOID: {
    ggml_compute_forward_hardsigmoid(pParams, pDst);
  } break;
  case GGML_UNARY_OP_EXP: {
    ggml_compute_forward_exp(pParams, pDst);
  } break;
  case GGML_UNARY_OP_FLOOR: {
    ggml_compute_forward_floor(pParams, pDst);
  } break;
  case GGML_UNARY_OP_CEIL: {
    ggml_compute_forward_ceil(pParams, pDst);
  } break;
  case GGML_UNARY_OP_ROUND: {
    ggml_compute_forward_round(pParams, pDst);
  } break;
  case GGML_UNARY_OP_TRUNC: {
    ggml_compute_forward_trunc(pParams, pDst);
  } break;
  case GGML_UNARY_OP_XIELU: {
    ggml_compute_forward_xielu(pParams, pDst);
  } break;
  case GGML_UNARY_OP_EXPM1: {
    ggml_compute_forward_expm1(pParams, pDst);
  } break;
  case GGML_UNARY_OP_SOFTPLUS: {
    ggml_compute_forward_softplus(pParams, pDst);
  } break;
  default: {
    GGML_ABORT("fatal error");
  }
  }
}

// ggml_compute_forward_glu

void ggml_compute_forward_glu(const ggml_compute_params *pParams, ggml_tensor *pDst) {

  const ggml_glu_op cOp = ggml_get_glu_op(pDst);

  switch (cOp) {
  case GGML_GLU_OP_REGLU: {
    fGgmlComputeForwardReglu(pParams, pDst);
  } break;
  case GGML_GLU_OP_GEGLU: {
    fGgmlComputeForwardGeglu(pParams, pDst);
  } break;
  case GGML_GLU_OP_SWIGLU: {
    fGgmlComputeForwardSwiglu(pParams, pDst);
  } break;
  case GGML_GLU_OP_SWIGLU_OAI: {
    fGgmlComputeForwardSwigluOai(pParams, pDst);
  } break;
  case GGML_GLU_OP_GEGLU_ERF: {
    fGgmlComputeForwardGegluErf(pParams, pDst);
  } break;
  case GGML_GLU_OP_GEGLU_QUICK: {
    fGgmlComputeForwardGegluQuick(pParams, pDst);
  } break;
  case GGML_GLU_OP_SWIGLU_CLAMP: {
    fGgmlComputeForwardSwigluClamp(pParams, pDst);
  } break;
  default: {
    GGML_ABORT("fatal error");
  }
  }
}

// ggml_compute_forward_get_rel_pos

static void fGgmlComputeForwardGetRelPosF16(const ggml_compute_params *pParams, ggml_tensor *dst) {
  GGML_UNUSED(pParams);

  const ggml_tensor *src0 = dst->src[0];

  // ref:
  // https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/image_encoder.py#L292-L322

  GGML_TENSOR_UNARY_OP_LOCALS

  const int64_t cW = ne1;

  ggml_fp16_t *vSrc0Data = (ggml_fp16_t *)src0->data;
  ggml_fp16_t *vDstData = (ggml_fp16_t *)dst->data;

  for (int64_t vI2 = 0; vI2 < ne2; ++vI2) {
    for (int64_t vI1 = 0; vI1 < ne1; ++vI1) {
      const int64_t cPos = (cW - vI1 - 1) + vI2;
      for (int64_t vI0 = 0; vI0 < ne0; ++vI0) {
        vDstData[vI2 * ne1 * ne0 + vI1 * ne0 + vI0] = vSrc0Data[cPos * ne00 + vI0];
      }
    }
  }
}

void ggml_compute_forward_get_rel_pos(const ggml_compute_params *pParams, ggml_tensor *pDst) {

  const ggml_tensor *vSrc0 = pDst->src[0];

  switch (vSrc0->type) {
  case GGML_TYPE_F16:
  case GGML_TYPE_BF16: {
    fGgmlComputeForwardGetRelPosF16(pParams, pDst);
  } break;
  default: {
    GGML_ABORT("fatal error");
  }
  }
}

// ggml_compute_forward_add_rel_pos

static void fGgmlComputeForwardAddRelPosF32(const ggml_compute_params *pParams, ggml_tensor *pDst) {

  const ggml_tensor *vSrc0 = pDst->src[0];
  const ggml_tensor *vSrc1 = pDst->src[1];
  const ggml_tensor *vSrc2 = pDst->src[2];

  const bool cInplace = (bool)((int32_t *)pDst->op_params)[0];
  if (!cInplace) {
    if (pParams->ith == 0) {
      memcpy((char *)pDst->data, (char *)vSrc0->data, ggml_nbytes(pDst));
    }
    ggml_barrier(pParams->threadpool);
  }
  // ref:
  // https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/image_encoder.py#L357-L359

  float *vSrc1Data = (float *)vSrc1->data;
  float *vSrc2Data = (float *)vSrc2->data;
  float *vDstData = (float *)pDst->data;

  const int64_t cNe10 = vSrc1->ne[0];
  const int64_t cNe11 = vSrc1->ne[1];
  const int64_t cNe12 = vSrc1->ne[2];
  const int64_t cNe13 = vSrc1->ne[3];

  const int cIth = pParams->ith;
  const int cNth = pParams->nth;

  // total patches in dst
  const int cNp = cNe13;

  // patches per thread
  const int cDp = (cNp + cNth - 1) / cNth;

  // patch range for this thread
  const int cIp0 = cDp * cIth;
  const int cIp1 = MIN(cIp0 + cDp, cNp);

  for (int64_t vI13 = cIp0; vI13 < cIp1; ++vI13) {
    for (int64_t vI12 = 0; vI12 < cNe12; ++vI12) {
      for (int64_t vI11 = 0; vI11 < cNe11; ++vI11) {
        const int64_t cJp1 = vI13 * cNe12 * cNe11 * cNe10 + vI12 * cNe11 * cNe10 + vI11 * cNe10;
        for (int64_t vI10 = 0; vI10 < cNe10; ++vI10) {
          const int64_t cJp0 = cJp1 + vI10;
          const float cSrc1E = vSrc1Data[cJp0];
          const float cSrc2E = vSrc2Data[cJp0];

          const int64_t cJdh = cJp0 * cNe10;
          const int64_t cJdw = cJdh - (cNe10 - 1) * vI10;

          for (int64_t vJ = 0; vJ < cNe10; ++vJ) {
            vDstData[cJdh + vJ] += cSrc2E;
            vDstData[cJdw + vJ * cNe10] += cSrc1E;
          }
        }
      }
    }
  }
}

void ggml_compute_forward_add_rel_pos(const ggml_compute_params *pParams, ggml_tensor *pDst) {

  const ggml_tensor *vSrc0 = pDst->src[0];

  switch (vSrc0->type) {
  case GGML_TYPE_F32: {
    fGgmlComputeForwardAddRelPosF32(pParams, pDst);
  } break;
  default: {
    GGML_ABORT("fatal error");
  }
  }
}

// ggml_compute_forward_rwkv_wkv6

static void fGgmlComputeForwardRwkvWkv6F32(const ggml_compute_params *pParams, ggml_tensor *pDst) {
  const int64_t cT = pDst->src[1]->ne[2];
  const int64_t cC = pDst->ne[0];
  const int64_t cHeads = pDst->src[1]->ne[1];
  const int64_t cNSeqs = pDst->src[5]->ne[1];
  const int64_t cHeadSize = cC / cHeads;

  float *vDstData = (float *)pDst->data;
  float *vState = ((float *)pDst->data) + cC * cT;

  const int cIth = pParams->ith;
  const int cNth = pParams->nth;

  const int cHStart = (cHeads * (cIth)) / cNth;
  const int cHEnd = ((cHeads * (cIth + 1)) / cNth < cHeads) ? (cHeads * (cIth + 1)) / cNth : cHeads;

  float *vK = (float *)pDst->src[0]->data;
  float *v_ = (float *)pDst->src[1]->data;
  float *vR = (float *)pDst->src[2]->data;
  float *vTimeFaaaa = (float *)pDst->src[3]->data;
  float *vTimeDecay = (float *)pDst->src[4]->data;

  size_t vTStride = cHeads * cHeadSize; // Same to C

  size_t vHStride = cC / cHeads;
  GGML_ASSERT(cC % cHeads == 0); // C must be divisible by HEADS
  size_t vHStride2d = cHeadSize * cHeadSize;

  if (cIth == 0) {
    memset(vDstData, 0, cT * cC * sizeof(float));
  }
  ggml_barrier(pParams->threadpool);

#if defined(__AVX__) && !defined(__AVX512F__)
#define GGML_F32X GGML_F32x8
#define GGML_F32X_SET1 GGML_F32x8_SET1
#define GGML_F32X_LOAD GGML_F32x8_LOAD
#define GGML_F32X_STORE GGML_F32x8_STORE
#define GGML_F32X_MUL GGML_F32x8_MUL
#define GGML_F32X_FMA GGML_F32x8_FMA
#define WKV_VECTOR_SIZE 8
#elif defined(__AVX512F__)
#define GGML_F32X GGML_F32x16
#define GGML_F32X_SET1 GGML_F32x16_SET1
#define GGML_F32X_LOAD GGML_F32x16_LOAD
#define GGML_F32X_STORE GGML_F32x16_STORE
#define GGML_F32X_MUL GGML_F32x16_MUL
#define GGML_F32X_FMA GGML_F32x16_FMA
#define WKV_VECTOR_SIZE 16
#elif defined(__ARM_FEATURE_SVE) && defined(__aarch64__)
#define GGML_F32X GGML_F32xt
#define GGML_F32X_SET1 GGML_F32xt_SET1
#define GGML_F32X_LOAD GGML_F32xt_LOAD
#define GGML_F32X_STORE GGML_F32xt_STORE
#define GGML_F32X_MUL GGML_F32xt_MUL
#define GGML_F32X_FMA GGML_F32xt_FMA
#define WKV_VECTOR_SIZE 8
#elif defined(__ARM_NEON) && defined(__aarch64__)
#define GGML_F32X GGML_F32x4
#define GGML_F32X_SET1 GGML_F32x4_SET1
#define GGML_F32X_LOAD GGML_F32x4_LOAD
#define GGML_F32X_STORE GGML_F32x4_STORE
#define GGML_F32X_MUL GGML_F32x4_MUL
#define GGML_F32X_FMA GGML_F32x4_FMA
#define WKV_VECTOR_SIZE 4
#endif

#ifdef WKV_VECTOR_SIZE
  int vWkvVectorSize;
#if defined(__ARM_FEATURE_SVE)
  vWkvVectorSize = svcntw();
#else
  vWkvVectorSize = WKV_VECTOR_SIZE;
#endif
  const int64_t cVecCount = cHeadSize / vWkvVectorSize;

  for (int64_t vT = 0; vT < cT; vT++) {
    size_t vTOffset = vT * vTStride;
    size_t vStateOffset = cHeadSize * cC * (vT / (cT / cNSeqs));
    float *vStateCur = vState + vStateOffset;
    float *vStatePrev = vT % (cT / cNSeqs) ? vStateCur : (float *)pDst->src[5]->data + vStateOffset;

    for (int64_t vH = cHStart; vH < cHEnd; vH++) {
      size_t vHOffset = vH * vHStride;
      size_t vTHOffset = vTOffset + vHOffset;
      size_t vH2dOffset = vH * vHStride2d;

      for (int64_t vI = 0; vI < cHeadSize; vI++) {
        size_t vTHIOffset = vTHOffset + vI;
        size_t vHIOffset = vHOffset + vI;
        size_t vH2dIOffset = vH2dOffset + vI * vHStride;

        float vKVal = vK[vTHIOffset];
        float vRVal = vR[vTHIOffset];
        float vTimeFaaaaVal = vTimeFaaaa[vHIOffset];
        float vTimeDecayVal = vTimeDecay[vTHIOffset];

        // Broadcast scalar values to vectors
        GGML_F32X vKVec = GGML_F32X_SET1(vKVal);
        GGML_F32X vRVec = GGML_F32X_SET1(vRVal);
        GGML_F32X vTimeFaaaaVec = GGML_F32X_SET1(vTimeFaaaaVal);
        GGML_F32X vTimeDecayVec = GGML_F32X_SET1(vTimeDecayVal);

        for (int64_t vJ = 0; vJ < cVecCount; vJ++) {
          size_t vBaseJ = vJ * vWkvVectorSize;
          size_t vTHJOffset = vTHOffset + vBaseJ;
          size_t vH2dIJOffset = vH2dIOffset + vBaseJ;

          // Load x elements at once
          GGML_F32X vVVec = GGML_F32X_LOAD(&v_[vTHJOffset]);
          GGML_F32X vPrevStateVec = GGML_F32X_LOAD(&vStatePrev[vH2dIJOffset]);
          GGML_F32X vDstVec = GGML_F32X_LOAD(&vDstData[vTHJOffset]);

          // Compute kv = v * k
          GGML_F32X vKvVec = GGML_F32X_MUL(vVVec, vKVec);

          // Compute temp = kv * time_faaaa + prev_state
          GGML_F32X vTempVec = GGML_F32X_FMA(vPrevStateVec, vKvVec, vTimeFaaaaVec);

          // Update dst: dst += temp * r
          vDstVec = GGML_F32X_FMA(vDstVec, vTempVec, vRVec);
          GGML_F32X_STORE(&vDstData[vTHJOffset], vDstVec);

          // Update state: state = prev_state * time_decay + kv
          GGML_F32X vNewStateVec = GGML_F32X_FMA(vKvVec, vPrevStateVec, vTimeDecayVec);
          GGML_F32X_STORE(&vStateCur[vH2dIJOffset], vNewStateVec);
        }

        // Handle remaining elements, this will not be used.
        for (int64_t vJ = cVecCount * vWkvVectorSize; vJ < cHeadSize; vJ++) {
          size_t vTHJOffset = vTHOffset + vJ;
          size_t vH2dIJOffset = vH2dIOffset + vJ;
          float vVVal = v_[vTHJOffset];
          float vKvVal = vVVal * vKVal;
          float vPrevStateVal = vStatePrev[vH2dIJOffset];
          float vTempVal = vKvVal * vTimeFaaaaVal + vPrevStateVal;
          vDstData[vTHJOffset] += vTempVal * vRVal;
          vStateCur[vH2dIJOffset] = vPrevStateVal * vTimeDecayVal + vKvVal;
        }
      }
    }
  }

#else
  // basically fused operations:
  // dst = r @ (time_faaaa * (k @ v) + state),
  // state = time_decay * state + (k @ v),
  // recursive through each token
  for (int64_t vT = 0; vT < cT; vT++) {
    size_t vTOffset = vT * vTStride;
    size_t vStateOffset = cHeadSize * cC * (vT / (cT / cNSeqs));
    float *vStateCur = vState + vStateOffset;
    float *vStatePrev = vT % (cT / cNSeqs) ? vStateCur : (float *)pDst->src[5]->data + vStateOffset;

    for (int64_t vH = cHStart; vH < cHEnd; vH++) {
      size_t vHOffset = vH * vHStride;
      size_t vTHOffset = vTOffset + vHOffset;
      size_t vH2dOffset = vH * vHStride2d;

      for (int64_t vI = 0; vI < cHeadSize; vI++) {
        size_t vTHIOffset = vTHOffset + vI;
        size_t vHIOffset = vHOffset + vI;
        size_t vH2dIOffset = vH2dOffset + vI * vHStride;

        float vKVal = vK[vTHIOffset];
        float vRVal = vR[vTHIOffset];
        float vTimeFaaaaVal = vTimeFaaaa[vHIOffset];
        // RWKV v6: different time_decay for each token.
        float vTimeDecayVal = vTimeDecay[vTHIOffset];

        for (int64_t vJ = 0; vJ < cHeadSize; vJ++) {
          size_t vTHJOffset = vTHOffset + vJ;
          size_t vH2dIJOffset = vH2dIOffset + vJ;

          float vVVal = v_[vTHJOffset];
          float vKvVal = vVVal * vKVal;
          float vPrevStateVal = vStatePrev[vH2dIJOffset];
          float vTempVal = vKvVal * vTimeFaaaaVal + vPrevStateVal;
          vDstData[vTHJOffset] += vTempVal * vRVal;
          vStateCur[vH2dIJOffset] = vPrevStateVal * vTimeDecayVal + vKvVal;
        }
      }
    }
  }
#endif
}

void ggml_compute_forward_rwkv_wkv6(const ggml_compute_params *pParams, ggml_tensor *pDst) {

  const ggml_tensor *vSrc0 = pDst->src[0];

  switch (vSrc0->type) {
  case GGML_TYPE_F32: {
    fGgmlComputeForwardRwkvWkv6F32(pParams, pDst);
  } break;
  default: {
    GGML_ABORT("fatal error");
  }
  }
}

// ggml_compute_forward_gla

static void fGgmlComputeForwardGlaF32(const ggml_compute_params *pParams, ggml_tensor *pDst) {
  const int64_t cT = pDst->src[1]->ne[2];
  const int64_t cC = pDst->ne[0];
  const int64_t cHeads = pDst->src[1]->ne[1];
  const int64_t cNSeqs = pDst->src[4]->ne[1];
  const int64_t cHeadSize = cC / cHeads;
  const float cScale = ggml_get_op_params_f32(pDst, 0);

  float *vDstData = (float *)pDst->data;
  float *vState = ((float *)pDst->data) + cC * cT;

  const int cIth = pParams->ith;
  const int cNth = pParams->nth;

  const int cHStart = (cHeads * (cIth)) / cNth;
  const int cHEnd = ((cHeads * (cIth + 1)) / cNth < cHeads) ? (cHeads * (cIth + 1)) / cNth : cHeads;

  float *vK = (float *)pDst->src[0]->data;
  float *v_ = (float *)pDst->src[1]->data;
  float *vQ = (float *)pDst->src[2]->data;
  float *vG = (float *)pDst->src[3]->data;

  size_t vTStride = cHeads * cHeadSize; // Same to C

  size_t vHStride = cC / cHeads;
  GGML_ASSERT(cC % cHeads == 0); // C must be divisible by HEADS
  size_t vHStride2d = cHeadSize * cHeadSize;

  if (cIth == 0) {
    memset(vDstData, 0, cT * cC * sizeof(float));
  }
  ggml_barrier(pParams->threadpool);

#if defined(__AVX__) && !defined(__AVX512F__)
#define GGML_F32X GGML_F32x8
#define GGML_F32X_SET1 GGML_F32x8_SET1
#define GGML_F32X_LOAD GGML_F32x8_LOAD
#define GGML_F32X_STORE GGML_F32x8_STORE
#define GGML_F32X_MUL GGML_F32x8_MUL
#define GGML_F32X_FMA GGML_F32x8_FMA
#define GLA_VECTOR_SIZE 8
#elif defined(__AVX512F__)
#define GGML_F32X GGML_F32x16
#define GGML_F32X_SET1 GGML_F32x16_SET1
#define GGML_F32X_LOAD GGML_F32x16_LOAD
#define GGML_F32X_STORE GGML_F32x16_STORE
#define GGML_F32X_MUL GGML_F32x16_MUL
#define GGML_F32X_FMA GGML_F32x16_FMA
#define GLA_VECTOR_SIZE 16
#elif defined(__ARM_FEATURE_SVE) && defined(__aarch64__)
#define GGML_F32X GGML_F32xt
#define GGML_F32X_SET1 GGML_F32xt_SET1
#define GGML_F32X_LOAD GGML_F32xt_LOAD
#define GGML_F32X_STORE GGML_F32xt_STORE
#define GGML_F32X_MUL GGML_F32xt_MUL
#define GGML_F32X_FMA GGML_F32xt_FMA
#define GLA_VECTOR_SIZE 8
#elif defined(__ARM_NEON) && defined(__aarch64__)
#define GGML_F32X GGML_F32x4
#define GGML_F32X_SET1 GGML_F32x4_SET1
#define GGML_F32X_LOAD GGML_F32x4_LOAD
#define GGML_F32X_STORE GGML_F32x4_STORE
#define GGML_F32X_MUL GGML_F32x4_MUL
#define GGML_F32X_FMA GGML_F32x4_FMA
#define GLA_VECTOR_SIZE 4
#endif

#ifdef GLA_VECTOR_SIZE
  int vGlaVectorSize;
#if defined(__ARM_FEATURE_SVE)
  vGlaVectorSize = svcntw();
#else
  vGlaVectorSize = GLA_VECTOR_SIZE;
#endif
  const int64_t cVecCount = cHeadSize / vGlaVectorSize;

  for (int64_t vT = 0; vT < cT; vT++) {
    size_t vTOffset = vT * vTStride;
    size_t vStateOffset = cHeadSize * cC * (vT / (cT / cNSeqs));
    float *vStateCur = vState + vStateOffset;
    float *vStatePrev = vT % (cT / cNSeqs) ? vStateCur : (float *)pDst->src[4]->data + vStateOffset;

    for (int64_t vH = cHStart; vH < cHEnd; vH++) {
      size_t vHOffset = vH * vHStride;
      size_t vTHOffset = vTOffset + vHOffset;
      size_t vH2dOffset = vH * vHStride2d;

      for (int64_t vI = 0; vI < cHeadSize; vI++) {
        size_t vTHIOffset = vTHOffset + vI;
        size_t vH2dIOffset = vH2dOffset + vI * vHStride;

        float vKVal = vK[vTHIOffset];
        float vQVal = vQ[vTHIOffset] * cScale;
        float vGVal = vG[vTHIOffset];

        // Broadcast scalar values to vectors
        GGML_F32X vKVec = GGML_F32X_SET1(vKVal);
        GGML_F32X vQVec = GGML_F32X_SET1(vQVal);
        GGML_F32X vGVec = GGML_F32X_SET1(vGVal);

        for (int64_t vJ = 0; vJ < cVecCount; vJ++) {
          size_t vBaseJ = vJ * vGlaVectorSize;
          size_t vTHJOffset = vTHOffset + vBaseJ;
          size_t vH2dIJOffset = vH2dIOffset + vBaseJ;

          // Load x elements at once
          GGML_F32X vVVec = GGML_F32X_LOAD(&v_[vTHJOffset]);
          GGML_F32X vPrevStateVec = GGML_F32X_LOAD(&vStatePrev[vH2dIJOffset]);
          GGML_F32X vDstVec = GGML_F32X_LOAD(&vDstData[vTHJOffset]);

          // Compute kv = v * k
          GGML_F32X vKvVec = GGML_F32X_MUL(vVVec, vKVec);

          // Compute temp = prev_state * g + kv
          GGML_F32X vTempVec = GGML_F32X_FMA(vKvVec, vPrevStateVec, vGVec);

          // Update dst: dst += temp * q
          vDstVec = GGML_F32X_FMA(vDstVec, vTempVec, vQVec);
          GGML_F32X_STORE(&vDstData[vTHJOffset], vDstVec);

          // Update state
          GGML_F32X_STORE(&vStateCur[vH2dIJOffset], vTempVec);
        }

        // Handle remaining elements, this will not be used.
        for (int64_t vJ = cVecCount * vGlaVectorSize; vJ < cHeadSize; vJ++) {
          size_t vTHJOffset = vTHOffset + vJ;
          size_t vH2dIJOffset = vH2dIOffset + vJ;
          float vVVal = v_[vTHJOffset];
          float vKvVal = vVVal * vKVal;
          float vPrevStateVal = vStatePrev[vH2dIJOffset];
          float vTempVal = vKvVal + vPrevStateVal * vGVal;
          vDstData[vTHJOffset] += vTempVal * vQVal;
          vStateCur[vH2dIJOffset] = vTempVal;
        }
      }
    }
  }

#else
  for (int64_t vT = 0; vT < cT; vT++) {
    size_t vTOffset = vT * vTStride;
    size_t vStateOffset = cHeadSize * cC * (vT / (cT / cNSeqs));
    float *vStateCur = vState + vStateOffset;
    float *vStatePrev = vT % (cT / cNSeqs) ? vStateCur : (float *)pDst->src[4]->data + vStateOffset;

    for (int64_t vH = cHStart; vH < cHEnd; vH++) {
      size_t vHOffset = vH * vHStride;
      size_t vTHOffset = vTOffset + vHOffset;
      size_t vH2dOffset = vH * vHStride2d;

      for (int64_t vI = 0; vI < cHeadSize; vI++) {
        size_t vTHIOffset = vTHOffset + vI;
        size_t vH2dIOffset = vH2dOffset + vI * vHStride;

        float vKVal = vK[vTHIOffset];
        float vQVal = vQ[vTHIOffset] * cScale;
        float vGVal = vG[vTHIOffset];

        for (int64_t vJ = 0; vJ < cHeadSize; vJ++) {
          size_t vTHJOffset = vTHOffset + vJ;
          size_t vH2dIJOffset = vH2dIOffset + vJ;

          float vVVal = v_[vTHJOffset];
          float vKvVal = vVVal * vKVal;
          float vPrevStateVal = vStatePrev[vH2dIJOffset];
          float vTempVal = vPrevStateVal * vGVal + vKvVal;
          vDstData[vTHJOffset] += vTempVal * vQVal;
          vStateCur[vH2dIJOffset] = vTempVal;
        }
      }
    }
  }
#endif
}

void ggml_compute_forward_gla(const ggml_compute_params *pParams, ggml_tensor *pDst) {

  const ggml_tensor *vSrc0 = pDst->src[0];

  switch (vSrc0->type) {
  case GGML_TYPE_F32: {
    fGgmlComputeForwardGlaF32(pParams, pDst);
  } break;
  default: {
    GGML_ABORT("fatal error");
  }
  }
}

static void fGgmlComputeForwardSolveTriF32(const struct ggml_compute_params *pParams, struct ggml_tensor *dst) {
  const struct ggml_tensor *src0 = dst->src[0]; // A (lower triangular)
  const struct ggml_tensor *src1 = dst->src[1]; // B (RHS)

  GGML_TENSOR_BINARY_OP_LOCALS;

  GGML_ASSERT(src0->type == GGML_TYPE_F32);
  GGML_ASSERT(src1->type == GGML_TYPE_F32);
  GGML_ASSERT(dst->type == GGML_TYPE_F32);

  GGML_ASSERT(ne00 == ne01); // A must be square
  GGML_ASSERT(ne0 == ne10);  // solution cols == B cols
  GGML_ASSERT(ne1 == ne11);  // solution rows == B rows

  GGML_ASSERT(ne02 == ne12 && ne12 == ne2);
  GGML_ASSERT(ne03 == ne13 && ne13 == ne3);

  const int cIth = pParams->ith;
  const int cNth = pParams->nth;

  const int64_t cK = ne10;              // number of RHS columns
  const int64_t cN = ne11;              // A is n×n
  const int64_t cNr = ne02 * ne03 * cK; // we're parallelizing on columns here, so seq x token x column will be the unit

  // chunks per thread
  const int64_t cDr = (cNr + cNth - 1) / cNth;

  // chunk range for this thread
  const int64_t cIr0 = cDr * cIth;
  const int64_t cIr1 = MIN(cIr0 + cDr, cNr);

  const float *vA = (const float *)src0->data; // [n, n, B1, B2]
  const float *vB = (const float *)src1->data; // [n, k, B1, B2]
  float *vX = (float *)dst->data;              // [n, k, B1, B2]

  for (int64_t vIr = cIr0; vIr < cIr1; ++vIr) {
    const int64_t cI03 = vIr / (ne02 * cK);
    const int64_t cI02 = (vIr - cI03 * ne02 * cK) / cK;
    const int64_t cI01 = (vIr - cI03 * ne02 * cK - cI02 * cK);

    const float *vABatch = vA + cI02 * nb02 / sizeof(float) + cI03 * nb03 / sizeof(float);
    const float *vBBatch = vB + cI02 * nb12 / sizeof(float) + cI03 * nb13 / sizeof(float);

    float *vXBatch = vX + cI02 * nb2 / sizeof(float) + cI03 * nb3 / sizeof(float);

    for (int64_t vI00 = 0; vI00 < cN; ++vI00) {
      float vSum = 0.0f;
      for (int64_t vT = 0; vT < vI00; ++vT) {
        vSum += vABatch[vI00 * cN + vT] * vXBatch[vT * cK + cI01];
      }

      const float cDiag = vABatch[vI00 * cN + vI00];
      assert(cDiag != 0.0f && "Zero diagonal in triangular matrix");

      vXBatch[vI00 * cK + cI01] = (vBBatch[vI00 * cK + cI01] - vSum) / cDiag;
    }
  }
}

void ggml_compute_forward_solve_tri(const struct ggml_compute_params *pParams, struct ggml_tensor *pDst) {
  const ggml_tensor *vSrc0 = pDst->src[0];
  const ggml_tensor *vSrc1 = pDst->src[1];

  if (vSrc0->type == GGML_TYPE_F32 && vSrc1->type == GGML_TYPE_F32) {
    fGgmlComputeForwardSolveTriF32(pParams, pDst);
  } else {
    GGML_ABORT("fatal error");
  }
}

// ggml_compute_forward_gated_delta_net
static void fGgmlComputeForwardGatedDeltaNetOneChunk(const ggml_compute_params *pParams, ggml_tensor *pDst,
                                                     int64_t pIr0, int64_t pIr1) {

  ggml_tensor *vSrcQ = pDst->src[0];
  ggml_tensor *vSrcK = pDst->src[1];
  ggml_tensor *vSrcV = pDst->src[2];
  ggml_tensor *vSrcG = pDst->src[3];
  ggml_tensor *vSrcBeta = pDst->src[4];
  ggml_tensor *vSrcState = pDst->src[5];

  const int64_t cSV = vSrcV->ne[0];
  const int64_t cH = vSrcV->ne[1];
  const int64_t cNTokens = vSrcV->ne[2];
  const int64_t cNSeqs = vSrcV->ne[3];

  GGML_ASSERT(ggml_is_contiguous_rows(vSrcQ));
  GGML_ASSERT(ggml_is_contiguous_rows(vSrcK));
  GGML_ASSERT(ggml_is_contiguous_rows(vSrcV));
  GGML_ASSERT(ggml_is_contiguous(vSrcG));
  GGML_ASSERT(ggml_is_contiguous(vSrcBeta));
  GGML_ASSERT(ggml_is_contiguous(vSrcState));

  GGML_ASSERT(vSrcG->ne[0] == 1 || vSrcG->ne[0] == cSV);
  GGML_ASSERT(vSrcBeta->ne[0] == 1);

  GGML_TENSOR_LOCALS(int64_t, neq, vSrcQ, ne);
  GGML_TENSOR_LOCALS(size_t, nbq, vSrcQ, nb);
  GGML_TENSOR_LOCALS(int64_t, nek, vSrcK, ne);
  GGML_TENSOR_LOCALS(size_t, nbk, vSrcK, nb);
  GGML_TENSOR_LOCALS(int64_t, nev, vSrcV, ne);
  GGML_TENSOR_LOCALS(size_t, nbv, vSrcV, nb);
  GGML_TENSOR_LOCALS(int64_t, neg, vSrcG, ne);
  GGML_TENSOR_LOCALS(size_t, nbg, vSrcG, nb);
  GGML_TENSOR_LOCALS(size_t, nbb, vSrcBeta, nb);

  const bool cKda = (neg0 == cSV);

  // K (snapshot slot count) is an op param; state holds s0 only [S_v, S_v, H, n_seqs].
  const int64_t cK = ggml_get_op_params_i32(pDst, 0);
  GGML_ASSERT(cK >= 1);
  // per-seq stride in floats (seq s starts at state + s * seq_stride)
  const int64_t cStateSeqStride = vSrcState->nb[3] / sizeof(float);

  const int64_t cPerThread = cSV + (cK > 1 ? cSV * cSV : 0);
  const int cIth = pParams->ith;

  float *vDelta = (float *)pParams->wdata + cIth * cPerThread + CACHE_LINE_SIZE_F32;
  float *vStateWork = cK > 1 ? (vDelta + cSV) : nullptr;

  // output layout: [attn_scores | new_states]
  // attn_scores: S_v * H * n_tokens * n_seqs    floats
  // new_states:  S_v * S_v * H * n_seqs * K     floats  (K snapshot slots; last min(n_tokens, K))
  const int64_t cAttnScoreElems = cSV * cH * cNTokens * cNSeqs;
  const int64_t cStateSizePerSnap = cSV * cSV * cH * cNSeqs;
  float *vAttnOutBase = (float *)pDst->data;
  float *vStateOutBase = (float *)pDst->data + cAttnScoreElems;

  // snapshot slot mapping: slot 0 = most recent state, slot s = s tokens back.
  // When n_tokens < K only slots 0..n_tokens-1 are written; older slots are caller-owned.

  const float *vStateInBase = (const float *)vSrcState->data;

  // const int64_t rq1 = nev1 / neq1;
  // const int64_t rk1 = nev1 / nek1;
  const int64_t cRq3 = nev3 / neq3;
  const int64_t cRk3 = nev3 / nek3;

  const float cScale = 1.0f / sqrtf((float)cSV);

  for (int64_t vIr = pIr0; vIr < pIr1; ++vIr) {
    const int64_t cIv1 = vIr % cH; // head_index
    const int64_t cIv3 = vIr / cH; // sequence

    const int64_t cIq1 = cIv1 % neq1;
    const int64_t cIk1 = cIv1 % nek1;

    const int64_t cIq3 = cIv3 / cRq3;
    const int64_t cIk3 = cIv3 / cRk3;

    // For K=1, write directly to the single output slot to avoid an extra memcpy at the end.
    // For K>1, work in scratch and copy out per-token when the slot is in range.
    float *vSOut = (cK > 1) ? vStateWork : vStateOutBase + (cIv3 * cH + cIv1) * cSV * cSV;

    // copy input state into the working buffer and operate in-place
    // state layout [S_v, S_v, H, n_seqs]: seq iv3 starts at iv3 * state_seq_stride.
    const float *vSIn = vStateInBase + cIv3 * cStateSeqStride + cIv1 * cSV * cSV;
    memcpy(vSOut, vSIn, cSV * cSV * sizeof(float));

    // attn output pointer for first token of this (head, seq)
    float *vAttnData = vAttnOutBase + (cIv3 * cNTokens * cH + cIv1) * cSV;

    for (int64_t vT = 0; vT < cNTokens; vT++) {
      const float *vQD = (const float *)((const char *)vSrcQ->data + cIq3 * nbq3 + vT * nbq2 + cIq1 * nbq1);
      const float *vKD = (const float *)((const char *)vSrcK->data + cIk3 * nbk3 + vT * nbk2 + cIk1 * nbk1);
      const float *vVD = (const float *)((const char *)vSrcV->data + cIv3 * nbv3 + vT * nbv2 + cIv1 * nbv1);

      const float cBetaVal = *(const float *)((const char *)vSrcBeta->data + cIv3 * nbb3 + vT * nbb2 + cIv1 * nbb1);
      const float *vGD = (const float *)((const char *)vSrcG->data + cIv3 * nbg3 + vT * nbg2 + cIv1 * nbg1);

      // state is stored transposed: s_out[j*S_v + i] = S[i][j]
      // so row j of s_out = column j of S (contiguous access)

      if (cKda) {
        // precompute exp(g) into delta scratch (reused below)
        for (int64_t vI = 0; vI < cSV; ++vI) {
          vDelta[vI] = expf(vGD[vI]);
        }
        // S[i][:] *= exp(g[i]) => for each row j of M: M[j][i] *= exp(g[i])
        for (int64_t vJ = 0; vJ < cSV; ++vJ) {
          ggml_vec_mul_f32(cSV, &vSOut[vJ * cSV], &vSOut[vJ * cSV], vDelta);
        }
      } else {
        ggml_vec_scale_f32(cSV * cSV, vSOut, expf(vGD[0]));
      }

      // delta[j] = sum_i S[i][j] * k[i] = dot(row j of M, k)
      for (int64_t vJ = 0; vJ < cSV; ++vJ) {
        float vSum = 0.0f;
        ggml_vec_dot_f32(cSV, &vSum, 0, &vSOut[vJ * cSV], 0, vKD, 0, 1);
        vDelta[vJ] = (vVD[vJ] - vSum) * cBetaVal;
      }

      // outer product: S[i][j] += k[i] * delta[j] => M[j][i] += delta[j] * k[i]
      for (int64_t vJ = 0; vJ < cSV; ++vJ) {
        ggml_vec_mad_f32(cSV, &vSOut[vJ * cSV], vKD, vDelta[vJ]);
      }

      // attn_out[j] = sum_i S[i][j] * q[i] = dot(row j of M, q)
      for (int64_t vJ = 0; vJ < cSV; ++vJ) {
        float vSum = 0.0f;
        ggml_vec_dot_f32(cSV, &vSum, 0, &vSOut[vJ * cSV], 0, vQD, 0, 1);
        vAttnData[vJ] = vSum * cScale;
      }

      vAttnData += cSV * cH; // advance to next token

      if (cK > 1) {
        const int64_t cTargetSlot = cNTokens - 1 - vT;
        if (cTargetSlot >= 0 && cTargetSlot < cK) {
          float *vCurrStateO = vStateOutBase + cTargetSlot * cStateSizePerSnap + (cIv3 * cH + cIv1) * cSV * cSV;
          memcpy(vCurrStateO, vSOut, cSV * cSV * sizeof(float));
        }
      }
    }
  }
}

static void fGgmlComputeForwardGatedDeltaNetF32(const ggml_compute_params *pParams, ggml_tensor *pDst) {

  ggml_tensor *vV = pDst->src[2];
  int64_t vNr = vV->ne[1] * vV->ne[3];

  // disable for NUMA
  const bool cDisableChunking = ggml_is_numa();

  int vNth = pParams->nth;
  int vIth = pParams->ith;

  // 4x chunks per thread
  int vNthScaled = vNth * 4;
  int64_t vChunkSize = (vNr + vNthScaled - 1) / vNthScaled;
  int64_t vNchunk = (vNr + vChunkSize - 1) / vChunkSize;

  if (vNth == 1 || vNchunk < vNth || cDisableChunking) {
    vNchunk = vNth;
  }

  if (vIth == 0) {
    ggml_threadpool_chunk_set(pParams->threadpool, vNth);
  }

  ggml_barrier(pParams->threadpool);

  const int64_t cDr = (vNr + vNchunk - 1) / vNchunk;

  int vCurrentChunk = vIth;

  while (vCurrentChunk < vNchunk) {
    const int64_t cIr0 = cDr * vCurrentChunk;
    const int64_t cIr1 = MIN(cIr0 + cDr, vNr);

    fGgmlComputeForwardGatedDeltaNetOneChunk(pParams, pDst, cIr0, cIr1);
    vCurrentChunk = ggml_threadpool_chunk_add(pParams->threadpool, 1);
  }
}

void ggml_compute_forward_gated_delta_net(const ggml_compute_params *pParams, ggml_tensor *pDst) {
  const ggml_tensor *vSrc0 = pDst->src[0];

  switch (vSrc0->type) {
  case GGML_TYPE_F32: {
    fGgmlComputeForwardGatedDeltaNetF32(pParams, pDst);
  } break;
  default: {
    GGML_ABORT("fatal error");
  }
  }
}

// ggml_compute_forward_dsv4_hc_comb

static void fGgmlDsv4HcCombNormCols(float *pComb, float pEps) {
  constexpr int64_t cHc = 4;

  for (int64_t vIdst = 0; vIdst < cHc; ++vIdst) {
    float vSum = pEps;
    for (int64_t vIsrc = 0; vIsrc < cHc; ++vIsrc) {
      vSum += pComb[vIdst + cHc * vIsrc];
    }

    const float cInvSum = 1.0f / vSum;
    for (int64_t vIsrc = 0; vIsrc < cHc; ++vIsrc) {
      pComb[vIdst + cHc * vIsrc] *= cInvSum;
    }
  }
}

static void fGgmlDsv4HcCombNormRows(float *pComb, float pEps) {
  constexpr int64_t cHc = 4;

  for (int64_t vIsrc = 0; vIsrc < cHc; ++vIsrc) {
    float vSum = pEps;
    for (int64_t vIdst = 0; vIdst < cHc; ++vIdst) {
      vSum += pComb[vIdst + cHc * vIsrc];
    }

    const float cInvSum = 1.0f / vSum;
    for (int64_t vIdst = 0; vIdst < cHc; ++vIdst) {
      pComb[vIdst + cHc * vIsrc] *= cInvSum;
    }
  }
}

static void fGgmlComputeForwardDsv4HcCombF32(const ggml_compute_params *pParams, ggml_tensor *pDst) {
  const ggml_tensor *vMixes = pDst->src[0];
  const ggml_tensor *vScale = pDst->src[1];
  const ggml_tensor *vBase = pDst->src[2];

  GGML_ASSERT(vMixes->type == GGML_TYPE_F32);
  GGML_ASSERT(vScale->type == GGML_TYPE_F32);
  GGML_ASSERT(vBase->type == GGML_TYPE_F32);
  GGML_ASSERT(pDst->type == GGML_TYPE_F32);

  constexpr int64_t cHc = 4;
  constexpr int64_t cCombOffset = 2 * cHc;
  constexpr int64_t cHcMixDim = (2 + cHc) * cHc;

  const int64_t cNTokens = vMixes->ne[1];

  GGML_ASSERT(vMixes->ne[0] == cHcMixDim);
  GGML_ASSERT(pDst->ne[0] == cHc);
  GGML_ASSERT(pDst->ne[1] == cHc);
  GGML_ASSERT(pDst->ne[2] == cNTokens);
  GGML_ASSERT(vScale->ne[0] >= 3);
  GGML_ASSERT(vBase->ne[0] == cHcMixDim);

  GGML_TENSOR_LOCALS(size_t, nbm, vMixes, nb);
  GGML_TENSOR_LOCALS(size_t, nbs, vScale, nb);
  GGML_TENSOR_LOCALS(size_t, nbb, vBase, nb);
  GGML_TENSOR_LOCALS(size_t, nbd, pDst, nb);

  const float cEps = ggml_get_op_params_f32(pDst, 0);
  const int32_t cNIter = ggml_get_op_params_i32(pDst, 1);
  GGML_ASSERT(cNIter > 0);

  const int cIth = pParams->ith;
  const int cNth = pParams->nth;

  const int64_t cDr = (cNTokens + cNth - 1) / cNth;
  const int64_t cIt0 = cDr * cIth;
  const int64_t cIt1 = MIN(cIt0 + cDr, cNTokens);

  const float cScaleComb = *(const float *)((const char *)vScale->data + 2 * nbs0);

  for (int64_t vIt = cIt0; vIt < cIt1; ++vIt) {
    float vComb[cHc * cHc];

    for (int64_t vIsrc = 0; vIsrc < cHc; ++vIsrc) {
      float vMax = -INFINITY;
      for (int64_t vIdst = 0; vIdst < cHc; ++vIdst) {
        const int64_t cIdx = vIdst + cHc * vIsrc;
        const float cXv = *(const float *)((const char *)vMixes->data + (cCombOffset + cIdx) * nbm0 + vIt * nbm1);
        const float cBv = *(const float *)((const char *)vBase->data + (cCombOffset + cIdx) * nbb0);
        const float cV = cXv * cScaleComb + cBv;
        vComb[cIdx] = cV;
        vMax = MAX(vMax, cV);
      }

      float vSum = 0.0f;
      for (int64_t vIdst = 0; vIdst < cHc; ++vIdst) {
        const int64_t cIdx = vIdst + cHc * vIsrc;
        const float cV = expf(vComb[cIdx] - vMax);
        vComb[cIdx] = cV;
        vSum += cV;
      }

      const float cInvSum = 1.0f / vSum;
      for (int64_t vIdst = 0; vIdst < cHc; ++vIdst) {
        const int64_t cIdx = vIdst + cHc * vIsrc;
        vComb[cIdx] = vComb[cIdx] * cInvSum + cEps;
      }
    }

    fGgmlDsv4HcCombNormCols(vComb, cEps);
    for (int32_t vI = 1; vI < cNIter; ++vI) {
      fGgmlDsv4HcCombNormRows(vComb, cEps);
      fGgmlDsv4HcCombNormCols(vComb, cEps);
    }

    for (int64_t vIsrc = 0; vIsrc < cHc; ++vIsrc) {
      for (int64_t vIdst = 0; vIdst < cHc; ++vIdst) {
        const int64_t cIdx = vIdst + cHc * vIsrc;
        *(float *)((char *)pDst->data + vIdst * nbd0 + vIsrc * nbd1 + vIt * nbd2) = vComb[cIdx];
      }
    }
  }
}

void ggml_compute_forward_dsv4_hc_comb(const ggml_compute_params *pParams, ggml_tensor *pDst) {
  const ggml_tensor *vSrc0 = pDst->src[0];

  switch (vSrc0->type) {
  case GGML_TYPE_F32: {
    fGgmlComputeForwardDsv4HcCombF32(pParams, pDst);
  } break;
  default: {
    GGML_ABORT("fatal error");
  }
  }
}

// ggml_compute_forward_dsv4_hc_pre

static void fGgmlComputeForwardDsv4HcPreF32(const ggml_compute_params *pParams, ggml_tensor *pDst) {
  const ggml_tensor *vX = pDst->src[0];
  const ggml_tensor *vWeights = pDst->src[1];

  GGML_ASSERT(vX->type == GGML_TYPE_F32);
  GGML_ASSERT(vWeights->type == GGML_TYPE_F32);
  GGML_ASSERT(pDst->type == GGML_TYPE_F32);

  const int64_t cNEmbd = vX->ne[0];
  const int64_t cHc = vX->ne[1];
  const int64_t cNTokens = vX->ne[2];

  const float cScale = ggml_get_op_params_f32(pDst, 0);
  const bool cGated = ggml_get_op_params_i32(pDst, 1) != 0;

  GGML_ASSERT(pDst->ne[0] == cNEmbd);
  GGML_ASSERT(pDst->ne[1] == cNTokens);
  if (cGated) {
    GGML_ASSERT(vWeights->ne[0] == cNEmbd);
    GGML_ASSERT(vWeights->ne[1] == cHc);
    GGML_ASSERT(vWeights->ne[2] == cNTokens);
  } else {
    GGML_ASSERT(vWeights->ne[0] == cHc);
    GGML_ASSERT(vWeights->ne[1] == cNTokens);
  }

  GGML_TENSOR_LOCALS(size_t, nbx, vX, nb);
  GGML_TENSOR_LOCALS(size_t, nbw, vWeights, nb);
  GGML_TENSOR_LOCALS(size_t, nbd, pDst, nb);

  const int cIth = pParams->ith;
  const int cNth = pParams->nth;

  const int64_t cNr = cNEmbd * cNTokens;
  const int64_t cDr = (cNr + cNth - 1) / cNth;
  const int64_t cIr0 = cDr * cIth;
  const int64_t cIr1 = MIN(cIr0 + cDr, cNr);

  for (int64_t vIr = cIr0; vIr < cIr1; ++vIr) {
    const int64_t cI0 = vIr % cNEmbd;
    const int64_t cIt = vIr / cNEmbd;

    float vSum = 0.0f;
    for (int64_t vIh = 0; vIh < cHc; ++vIh) {
      const float cXv = *(const float *)((const char *)vX->data + cI0 * nbx0 + vIh * nbx1 + cIt * nbx2);
      float vWv;
      if (cGated) {
        const float cGv = *(const float *)((const char *)vWeights->data + cI0 * nbw0 + vIh * nbw1 + cIt * nbw2);
        vWv = 1.0f / (1.0f + expf(-cGv));
      } else {
        vWv = *(const float *)((const char *)vWeights->data + vIh * nbw0 + cIt * nbw1);
      }
      vSum += cXv * vWv;
    }

    *(float *)((char *)pDst->data + cI0 * nbd0 + cIt * nbd1) = cScale * vSum;
  }
}

void ggml_compute_forward_dsv4_hc_pre(const ggml_compute_params *pParams, ggml_tensor *pDst) {
  const ggml_tensor *vSrc0 = pDst->src[0];

  switch (vSrc0->type) {
  case GGML_TYPE_F32: {
    fGgmlComputeForwardDsv4HcPreF32(pParams, pDst);
  } break;
  default: {
    GGML_ABORT("fatal error");
  }
  }
}

// ggml_compute_forward_dsv4_hc_post

static void fGgmlComputeForwardDsv4HcPostF32(const ggml_compute_params *pParams, ggml_tensor *pDst) {
  const ggml_tensor *vX = pDst->src[0];
  const ggml_tensor *vResidual = pDst->src[1];
  const ggml_tensor *vPost = pDst->src[2];
  const ggml_tensor *vComb = pDst->src[3];

  GGML_ASSERT(vX->type == GGML_TYPE_F32);
  GGML_ASSERT(vResidual->type == GGML_TYPE_F32);
  GGML_ASSERT(vPost->type == GGML_TYPE_F32);
  GGML_ASSERT(pDst->type == GGML_TYPE_F32);

  const int64_t cNEmbd = vX->ne[0];
  const int64_t cNTokens = vX->ne[1];
  const int64_t cHc = vResidual->ne[1];

  GGML_ASSERT(pDst->ne[0] == cNEmbd);
  GGML_ASSERT(pDst->ne[1] == cHc);
  GGML_ASSERT(pDst->ne[2] == cNTokens);
  GGML_ASSERT(vResidual->ne[0] == cNEmbd);
  GGML_ASSERT(vResidual->ne[2] == cNTokens);
  GGML_ASSERT(vPost->ne[0] == cHc);
  GGML_ASSERT(vPost->ne[1] == cNTokens);

  // comb == NULL: identity mixing, each stream keeps its own residual
  size_t vNbc0 = 0;
  size_t vNbc1 = 0;
  size_t vNbc2 = 0;
  if (vComb) {
    GGML_ASSERT(vComb->type == GGML_TYPE_F32);
    GGML_ASSERT(vComb->ne[0] == cHc);
    GGML_ASSERT(vComb->ne[1] == cHc);
    GGML_ASSERT(vComb->ne[2] == cNTokens);
    vNbc0 = vComb->nb[0];
    vNbc1 = vComb->nb[1];
    vNbc2 = vComb->nb[2];
  }

  GGML_TENSOR_LOCALS(size_t, nbx, vX, nb);
  GGML_TENSOR_LOCALS(size_t, nbr, vResidual, nb);
  GGML_TENSOR_LOCALS(size_t, nbp, vPost, nb);
  GGML_TENSOR_LOCALS(size_t, nbd, pDst, nb);

  const int cIth = pParams->ith;
  const int cNth = pParams->nth;

  const int64_t cNr = cNEmbd * cHc * cNTokens;
  const int64_t cDr = (cNr + cNth - 1) / cNth;
  const int64_t cIr0 = cDr * cIth;
  const int64_t cIr1 = MIN(cIr0 + cDr, cNr);

  for (int64_t vIr = cIr0; vIr < cIr1; ++vIr) {
    const int64_t cI0 = vIr % cNEmbd;
    const int64_t cIdst = (vIr / cNEmbd) % cHc;
    const int64_t cIt = vIr / (cNEmbd * cHc);

    const float cXv = *(const float *)((const char *)vX->data + cI0 * nbx0 + cIt * nbx1);
    const float cPv = *(const float *)((const char *)vPost->data + cIdst * nbp0 + cIt * nbp1);

    float vSum = cXv * cPv;
    if (vComb) {
      for (int64_t vIsrc = 0; vIsrc < cHc; ++vIsrc) {
        const float cRv = *(const float *)((const char *)vResidual->data + cI0 * nbr0 + vIsrc * nbr1 + cIt * nbr2);
        const float cCv = *(const float *)((const char *)vComb->data + cIdst * vNbc0 + vIsrc * vNbc1 + cIt * vNbc2);
        vSum += cRv * cCv;
      }
    } else {
      vSum += *(const float *)((const char *)vResidual->data + cI0 * nbr0 + cIdst * nbr1 + cIt * nbr2);
    }

    *(float *)((char *)pDst->data + cI0 * nbd0 + cIdst * nbd1 + cIt * nbd2) = vSum;
  }
}

void ggml_compute_forward_dsv4_hc_post(const ggml_compute_params *pParams, ggml_tensor *pDst) {
  const ggml_tensor *vSrc0 = pDst->src[0];

  switch (vSrc0->type) {
  case GGML_TYPE_F32: {
    fGgmlComputeForwardDsv4HcPostF32(pParams, pDst);
  } break;
  default: {
    GGML_ABORT("fatal error");
  }
  }
}

// ggml_compute_forward_rwkv_wkv7

static void fGgmlComputeForwardRwkvWkv7F32(const ggml_compute_params *pParams, ggml_tensor *pDst) {
  const int64_t cT = pDst->src[1]->ne[2];
  const int64_t cC = pDst->ne[0];
  const int64_t cHeads = pDst->src[1]->ne[1];
  const int64_t cNSeqs = pDst->src[6]->ne[1];
  const int64_t cHeadSize = cC / cHeads;

  float *vDstData = (float *)pDst->data;
  float *vState = ((float *)pDst->data) + cC * cT;

  const int cIth = pParams->ith;
  const int cNth = pParams->nth;

  const int cHStart = (cHeads * (cIth)) / cNth;
  const int cHEnd = ((cHeads * (cIth + 1)) / cNth < cHeads) ? (cHeads * (cIth + 1)) / cNth : cHeads;

  float *vR = (float *)pDst->src[0]->data;
  float *vW = (float *)pDst->src[1]->data;
  float *vK = (float *)pDst->src[2]->data;
  float *v_ = (float *)pDst->src[3]->data;
  float *vA = (float *)pDst->src[4]->data;
  float *vB = (float *)pDst->src[5]->data;

  int64_t vTStride = cHeads * cHeadSize; // Same to C

  int64_t vHStride = cC / cHeads;
  GGML_ASSERT(cC % cHeads == 0); // C must be divisible by HEADS
  int64_t vHStride2d = cHeadSize * cHeadSize;

#if defined(GGML_SIMD)
#if defined(__ARM_FEATURE_SVE) || defined(__riscv_v_intrinsic)
  // scalar Route to scalar implementation       //TODO: Write SVE code and RVV code
  for (int64_t vT = 0; vT < cT; vT++) {
    int64_t vTOffset = vT * vTStride;
    int64_t vStateOffset = cHeadSize * cC * (vT / (cT / cNSeqs));
    float *vStateCur = vState + vStateOffset;
    float *vStatePrev = vT % (cT / cNSeqs) ? vStateCur : (float *)pDst->src[6]->data + vStateOffset;

    for (int64_t vH = cHStart; vH < cHEnd; vH++) {
      int64_t vHOffset = vH * vHStride;
      int64_t vTHOffset = vTOffset + vHOffset;
      int64_t vH2dOffset = vH * vHStride2d;

      for (int64_t i = 0; i < cHeadSize; i++) {
        int64_t vTHIOffset = vTHOffset + i;
        int64_t vH2dIOffset = vH2dOffset + i * vHStride;

        float v_val = v_[vTHIOffset];

        float vSa = 0, result = 0;
        for (int64_t vJ = 0; vJ < cHeadSize; vJ++) {
          vSa += vA[vTHOffset + vJ] * vStatePrev[vH2dIOffset + vJ];
        }

        for (int64_t vJ = 0; vJ < cHeadSize; vJ++) {
          int64_t vTHJOffset = vTHOffset + vJ;
          int64_t vH2dIJOffset = vH2dIOffset + vJ;

          float vRVal = vR[vTHJOffset];
          float vWVal = vW[vTHJOffset];
          float vKVal = vK[vTHJOffset];
          float vBVal = vB[vTHJOffset];
          float vKvVal = v_val * vKVal;
          float vPrevStateVal = vStatePrev[vH2dIJOffset];
          vStateCur[vH2dIJOffset] = vPrevStateVal * vWVal + vKvVal + vSa * vBVal;
          result += vStateCur[vH2dIJOffset] * vRVal;
        }
        vDstData[vTHIOffset] = result;
      }
    }
  }
#else
  for (int64_t vT = 0; vT < cT; vT++) {
    int64_t vTOffset = vT * vTStride;
    int64_t vStateOffset = cHeadSize * cC * (vT / (cT / cNSeqs));
    float *vStateCur = vState + vStateOffset;
    float *vStatePrev = vT % (cT / cNSeqs) ? vStateCur : (float *)pDst->src[6]->data + vStateOffset;

    for (int64_t vH = cHStart; vH < cHEnd; vH++) {
      int64_t vHOffset = vH * vHStride;
      int64_t vTHOffset = vTOffset + vHOffset;
      int64_t vH2dOffset = vH * vHStride2d;

      for (int64_t vIi = 0; vIi < cHeadSize; vIi++) {
        int64_t vTHIOffset = vTHOffset + vIi;
        int64_t vH2dIOffset = vH2dOffset + vIi * vHStride;

        GGML_F32_VEC vVVec = GGML_F32_VEC_SET1(v_[vTHIOffset]);

        float vSa = 0;
        {
          GGML_F32_VEC vSum[GGML_F32_ARR] = {GGML_F32_VEC_ZERO};
          GGML_F32_VEC vAx[GGML_F32_ARR];
          GGML_F32_VEC vAy[GGML_F32_ARR];
          for (int64_t vJ = 0; vJ < cHeadSize; vJ += GGML_F32_STEP) {
            for (int64_t vKk = 0; vKk < GGML_F32_ARR; vKk++) {
              vAx[vKk] = GGML_F32_VEC_LOAD(&vA[vTHOffset + vJ + vKk * GGML_F32_EPR]);
              vAy[vKk] = GGML_F32_VEC_LOAD(&vStatePrev[vH2dIOffset + vJ + vKk * GGML_F32_EPR]);
              vSum[vKk] = GGML_F32_VEC_FMA(vSum[vKk], vAx[vKk], vAy[vKk]);
            }
          }
          GGML_F32_VEC_REDUCE(vSa, vSum);
        }

        GGML_F32_VEC vSaVec = GGML_F32_VEC_SET1(vSa);

        int64_t vJ = 0;
        GGML_F32_VEC vResultVec[GGML_F32_ARR] = {GGML_F32_VEC_ZERO};
        for (; vJ < cHeadSize; vJ += GGML_F32_STEP) {
          for (int64_t vKk = 0; vKk < GGML_F32_ARR; vKk++) {
            int64_t vTHJOffset = vTHOffset + vJ + vKk * GGML_F32_EPR;
            int64_t vH2dIJOffset = vH2dIOffset + vJ + vKk * GGML_F32_EPR;

            GGML_F32_VEC vRVec = GGML_F32_VEC_LOAD(&vR[vTHJOffset]);
            GGML_F32_VEC vWVec = GGML_F32_VEC_LOAD(&vW[vTHJOffset]);
            GGML_F32_VEC vKVec = GGML_F32_VEC_LOAD(&vK[vTHJOffset]);
            GGML_F32_VEC vBVec = GGML_F32_VEC_LOAD(&vB[vTHJOffset]);

            vKVec = GGML_F32_VEC_MUL(vVVec, vKVec);

            GGML_F32_VEC vStateVec = GGML_F32_VEC_LOAD(&vStatePrev[vH2dIJOffset]);
            // kv + s * decay + sa * b
            vStateVec = GGML_F32_VEC_FMA(vKVec, vStateVec, vWVec);
            vStateVec = GGML_F32_VEC_FMA(vStateVec, vSaVec, vBVec);
            GGML_F32_VEC_STORE(&vStateCur[vH2dIJOffset], vStateVec);

            vResultVec[vKk] = GGML_F32_VEC_FMA(vResultVec[vKk], vStateVec, vRVec);
          }
        }
        GGML_F32_VEC_REDUCE(vDstData[vTHIOffset], vResultVec);

        // There shouldn't be left-overs though.
        for (; vJ < cHeadSize; vJ++) {
          int64_t vTHJOffset = vTHOffset + vJ;
          int64_t vH2dIJOffset = vH2dIOffset + vJ;

          float vRVal = vR[vTHJOffset];
          float vWVal = vW[vTHJOffset];
          float vKVal = vK[vTHJOffset];
          float vBVal = vB[vTHJOffset];
          float vKvVal = v_[vTHIOffset] * vKVal;

          float vPrevStateVal = vStatePrev[vH2dIJOffset];
          vStateCur[vH2dIJOffset] = vPrevStateVal * vWVal + vKvVal + vSa * vBVal;
          vDstData[vTHIOffset] += vStateCur[vH2dIJOffset] * vRVal;
        }
      }
    }
  }
#endif
#else
  for (int64_t vT = 0; vT < cT; vT++) {
    int64_t vTOffset = vT * vTStride;
    int64_t vStateOffset = cHeadSize * cC * (vT / (cT / cNSeqs));
    float *vStateCur = vState + vStateOffset;
    float *vStatePrev = vT % (cT / cNSeqs) ? vStateCur : (float *)pDst->src[6]->data + vStateOffset;

    for (int64_t vH = cHStart; vH < cHEnd; vH++) {
      int64_t vHOffset = vH * vHStride;
      int64_t vTHOffset = vTOffset + vHOffset;
      int64_t vH2dOffset = vH * vHStride2d;

      for (int64_t i = 0; i < cHeadSize; i++) {
        int64_t vTHIOffset = vTHOffset + i;
        int64_t vH2dIOffset = vH2dOffset + i * vHStride;

        float v_val = v_[vTHIOffset];

        float vSa = 0, result = 0;
        for (int64_t vJ = 0; vJ < cHeadSize; vJ++) {
          vSa += vA[vTHOffset + vJ] * vStatePrev[vH2dIOffset + vJ];
        }

        for (int64_t vJ = 0; vJ < cHeadSize; vJ++) {
          int64_t vTHJOffset = vTHOffset + vJ;
          int64_t vH2dIJOffset = vH2dIOffset + vJ;

          float vRVal = vR[vTHJOffset];
          float vWVal = vW[vTHJOffset];
          float vKVal = vK[vTHJOffset];
          float vBVal = vB[vTHJOffset];
          float vKvVal = v_val * vKVal;
          float vPrevStateVal = vStatePrev[vH2dIJOffset];
          vStateCur[vH2dIJOffset] = vPrevStateVal * vWVal + vKvVal + vSa * vBVal;
          result += vStateCur[vH2dIJOffset] * vRVal;
        }
        vDstData[vTHIOffset] = result;
      }
    }
  }
#endif
}

void ggml_compute_forward_rwkv_wkv7(const ggml_compute_params *pParams, ggml_tensor *pDst) {

  const ggml_tensor *vSrc0 = pDst->src[0];

  switch (vSrc0->type) {
  case GGML_TYPE_F32: {
    fGgmlComputeForwardRwkvWkv7F32(pParams, pDst);
  } break;
  default: {
    GGML_ABORT("fatal error");
  }
  }
}

// ggml_compute_forward_map_custom1

void ggml_compute_forward_map_custom1(const ggml_compute_params *pParams, ggml_tensor *pDst) {

  const ggml_tensor *vA = pDst->src[0];

  struct ggml_map_custom1_op_params vP;
  memcpy(&vP, pDst->op_params, sizeof(vP));

  vP.fun(pDst, vA, pParams->ith, pParams->nth, vP.userdata);
}

// ggml_compute_forward_map_custom2

void ggml_compute_forward_map_custom2(const ggml_compute_params *pParams, ggml_tensor *pDst) {

  const ggml_tensor *vA = pDst->src[0];
  const ggml_tensor *vB = pDst->src[1];

  struct ggml_map_custom2_op_params vP;
  memcpy(&vP, pDst->op_params, sizeof(vP));

  vP.fun(pDst, vA, vB, pParams->ith, pParams->nth, vP.userdata);
}

// ggml_compute_forward_map_custom3

void ggml_compute_forward_map_custom3(const ggml_compute_params *pParams, ggml_tensor *pDst) {

  const ggml_tensor *vA = pDst->src[0];
  const ggml_tensor *vB = pDst->src[1];
  const ggml_tensor *vC = pDst->src[2];

  struct ggml_map_custom3_op_params vP;
  memcpy(&vP, pDst->op_params, sizeof(vP));

  vP.fun(pDst, vA, vB, vC, pParams->ith, pParams->nth, vP.userdata);
}

// ggml_compute_forward_custom

void ggml_compute_forward_custom(const struct ggml_compute_params *pParams, struct ggml_tensor *pDst) {

  struct ggml_custom_op_params vP;
  memcpy(&vP, pDst->op_params, sizeof(vP));

  vP.fun(pDst, pParams->ith, pParams->nth, vP.userdata);
}

// ggml_compute_forward_cross_entropy_loss

static void fGgmlComputeForwardCrossEntropyLossF32(const ggml_compute_params *pParams, ggml_tensor *pDst) {

  const ggml_tensor *vSrc0 = pDst->src[0];
  const ggml_tensor *vSrc1 = pDst->src[1];

  GGML_ASSERT(vSrc0->type == GGML_TYPE_F32);
  GGML_ASSERT(vSrc1->type == GGML_TYPE_F32);
  GGML_ASSERT(vSrc0->nb[0] == ggml_type_size(vSrc0->type));
  GGML_ASSERT(vSrc1->nb[0] == ggml_type_size(vSrc1->type));
  GGML_ASSERT(ggml_are_same_shape(vSrc0, vSrc1));
  GGML_ASSERT(ggml_is_scalar(pDst));
  GGML_ASSERT(pDst->type == GGML_TYPE_F32);

  // TODO: handle transposed/permuted matrices
  const int64_t cNc = vSrc0->ne[0];
  const int64_t cNr = ggml_nrows(vSrc0);

  const int cIth = pParams->ith;
  const int cNth = pParams->nth;

  float *vSums = (float *)pParams->wdata;
  float *vSt = ((float *)pParams->wdata) + cNth + cIth * cNc;
  float vSumThread = 0.0f;

  GGML_ASSERT(pParams->wsize >= sizeof(float) * (cNth + cNth * cNc));

  // rows per thread
  const int64_t cDr = (cNr + cNth - 1) / cNth;

  // row range for this thread
  const int64_t cIr0 = cDr * cIth;
  const int64_t cIr1 = MIN(cIr0 + cDr, cNr);

  for (int64_t vI1 = cIr0; vI1 < cIr1; ++vI1) {
    const float *vS0 = (const float *)((const char *)vSrc0->data + vI1 * vSrc0->nb[1]);
    const float *vS1 = (const float *)((const char *)vSrc1->data + vI1 * vSrc1->nb[1]);

#ifndef NDEBUG
    for (int64_t vI = 0; vI < cNc; ++vI) {
      // printf("p[%d] = %f\n", i, p[i]);
      assert(!isnan(vS0[vI]));
      assert(!isnan(vS1[vI]));
    }
#endif // NDEBUG

    float vMax = -INFINITY;
    ggml_vec_max_f32(cNc, &vMax, vS0);
    const ggml_float cSumSoftmax = ggml_vec_log_soft_max_f32(cNc, vSt, vS0, vMax);
    assert(cSumSoftmax >= 0.0);

    ggml_vec_add1_f32(cNc, vSt, vSt, -cSumSoftmax);
    ggml_vec_mul_f32(cNc, vSt, vSt, vS1);

    float vSumSt = 0.0f;
    ggml_vec_sum_f32(cNc, &vSumSt, vSt);
    vSumThread += vSumSt;

#ifndef NDEBUG
    for (int64_t vI = 0; vI < cNc; ++vI) {
      assert(!isnan(vSt[vI]));
      assert(!isinf(vSt[vI]));
    }
#endif // NDEBUG
  }
  vSums[cIth] = vSumThread;
  ggml_barrier(pParams->threadpool);

  if (cIth == 0) {
    float *vDp = (float *)pDst->data;
    ggml_vec_sum_f32(cNth, vDp, vSums);
    vDp[0] *= -1.0f / (float)cNr;
  }
}

void ggml_compute_forward_cross_entropy_loss(const ggml_compute_params *pParams, ggml_tensor *pDst) {

  const ggml_tensor *vSrc0 = pDst->src[0];

  switch (vSrc0->type) {
  case GGML_TYPE_F32: {
    fGgmlComputeForwardCrossEntropyLossF32(pParams, pDst);
  } break;
  default: {
    GGML_ABORT("fatal error");
  }
  }
}

// ggml_compute_forward_cross_entropy_loss_back

static void fGgmlComputeForwardCrossEntropyLossBackF32(const ggml_compute_params *pParams, ggml_tensor *pDst) {

  const ggml_tensor *vGrad = pDst->src[0];  // gradient of forward pass output
  const ggml_tensor *vSrc0f = pDst->src[1]; // src0 of forward pass
  const ggml_tensor *vSrc1f = pDst->src[2]; // src1 of forward pass

  GGML_ASSERT(ggml_is_contiguous(pDst));
  GGML_ASSERT(ggml_is_contiguous(vSrc0f));
  GGML_ASSERT(ggml_is_contiguous(vSrc1f));
  GGML_ASSERT(ggml_is_contiguous(vGrad));
  GGML_ASSERT(ggml_are_same_shape(vSrc0f, vSrc1f) && ggml_are_same_shape(vSrc0f, pDst));

  const int64_t cIth = pParams->ith;
  const int64_t cNth = pParams->nth;

  // TODO: handle transposed/permuted matrices
  const int64_t cNc = vSrc0f->ne[0];
  const int64_t cNr = ggml_nrows(vSrc0f);

  // rows per thread
  const int64_t cDr = (cNr + cNth - 1) / cNth;

  // row range for this thread
  const int64_t cIr0 = cDr * cIth;
  const int64_t cIr1 = MIN(cIr0 + cDr, cNr);

  const float cDByNr = ((const float *)vGrad->data)[0] / (float)cNr;

  for (int64_t vI1 = cIr0; vI1 < cIr1; vI1++) {
    float *vDs0 = (float *)((char *)pDst->data + vI1 * pDst->nb[1]);
    const float *vS0 = (const float *)((const char *)vSrc0f->data + vI1 * vSrc0f->nb[1]);
    const float *vS1 = (const float *)((const char *)vSrc1f->data + vI1 * vSrc1f->nb[1]);

#ifndef NDEBUG
    for (int64_t vI = 0; vI < cNc; ++vI) {
      // printf("p[%d] = %f\n", i, p[i]);
      assert(!isnan(vS0[vI]));
      assert(!isnan(vS1[vI]));
    }
#endif // NDEBUG

    // soft_max
    float vMax = -INFINITY;
    ggml_vec_max_f32(cNc, &vMax, vS0);
    const ggml_float cSum = ggml_vec_soft_max_f32(cNc, vDs0, vS0, vMax);
    assert(cSum > 0.0);
    ggml_vec_scale_f32(cNc, vDs0, 1.0 / cSum);

    // grad(src0f) = (softmax(src0f) - src1f) * grad(cross_entropy_loss(src0f, src1f)) / nr
    ggml_vec_sub_f32(cNc, vDs0, vDs0, vS1);
    ggml_vec_scale_f32(cNc, vDs0, cDByNr);

#ifndef NDEBUG
    for (int64_t vI = 0; vI < cNc; ++vI) {
      assert(!isnan(vDs0[vI]));
      assert(!isinf(vDs0[vI]));
    }
#endif // NDEBUG
  }
}

void ggml_compute_forward_cross_entropy_loss_back(const ggml_compute_params *pParams, ggml_tensor *pDst) {

  const ggml_tensor *vSrc0 = pDst->src[0];

  switch (vSrc0->type) {
  case GGML_TYPE_F32: {
    fGgmlComputeForwardCrossEntropyLossBackF32(pParams, pDst);
  } break;
  default: {
    GGML_ABORT("fatal error");
  }
  }
}

static void fGgmlComputeForwardOptStepAdamwF32(const ggml_compute_params *pParams, ggml_tensor *dst) {

  const ggml_tensor *src0 = dst->src[0];
  const ggml_tensor *vSrc0Grad = dst->src[1];
  const ggml_tensor *vSrc0GradM = dst->src[2];
  const ggml_tensor *vSrc0GradV = dst->src[3];
  const ggml_tensor *vAdamwParams = dst->src[4];

  GGML_ASSERT(ggml_are_same_shape(src0, vSrc0Grad));
  GGML_ASSERT(ggml_are_same_shape(src0, vSrc0GradM));
  GGML_ASSERT(ggml_are_same_shape(src0, vSrc0GradV));
  GGML_ASSERT(ggml_nelements(vAdamwParams) == 7);

  const int cIth = pParams->ith;
  const int cNth = pParams->nth;

  const int cNr = ggml_nrows(src0);

  GGML_TENSOR_UNARY_OP_LOCALS
  GGML_ASSERT(nb00 == sizeof(float));

  // rows per thread
  const int cDr = (cNr + cNth - 1) / cNth;

  // row range for this thread
  const int cIr0 = cDr * cIth;
  const int cIr1 = MIN(cIr0 + cDr, cNr);

  const float *vAdamwParamsPtr = ggml_get_data_f32(vAdamwParams);

  const float cAlpha = vAdamwParamsPtr[0];
  const float cBeta1 = vAdamwParamsPtr[1];
  const float cBeta2 = vAdamwParamsPtr[2];
  const float cEps = vAdamwParamsPtr[3];
  const float cWd = vAdamwParamsPtr[4];
  const float cBeta1h = vAdamwParamsPtr[5];
  const float cBeta2h = vAdamwParamsPtr[6];
  const float cKeep = 1.f - cAlpha * cWd;
  for (int vIr = cIr0; vIr < cIr1; ++vIr) {
    const int64_t cI03 = vIr / (ne02 * ne01);
    const int64_t cI02 = (vIr - cI03 * ne02 * ne01) / ne01;
    const int64_t cI01 = (vIr - cI03 * ne02 * ne01 - cI02 * ne01);

    const size_t cOffset = cI03 * nb03 + cI02 * nb02 + cI01 * nb01;

    float *vW = (float *)((char *)src0->data + cOffset);                        // weight
    const float *vG = (const float *)((const char *)vSrc0Grad->data + cOffset); // grad
    float *vM = (float *)((char *)vSrc0GradM->data + cOffset);
    float *v_ = (float *)((char *)vSrc0GradV->data + cOffset);

    for (int vI00 = 0; vI00 < ne00; ++vI00) {
      vM[vI00] = vM[vI00] * cBeta1 + vG[vI00] * (1.0f - cBeta1);
      v_[vI00] = v_[vI00] * cBeta2 + vG[vI00] * vG[vI00] * (1.0f - cBeta2);

      const float cMh = vM[vI00] * cBeta1h;
      const float cVh = sqrtf(v_[vI00] * cBeta2h) + cEps;

      // The weight decay is applied independently of the Adam momenta m and v.
      // This is NOT equivalent to l2 regularization that adds w[i00]*w[i00] to the loss.
      // See: https://arxiv.org/pdf/1711.05101v3.pdf
      vW[vI00] = vW[vI00] * cKeep - cAlpha * cMh / cVh;
    }
  }
}

void ggml_compute_forward_opt_step_adamw(const ggml_compute_params *pParams, ggml_tensor *pDst) {

  const ggml_tensor *vSrc0 = pDst->src[0];

  switch (vSrc0->type) {
  case GGML_TYPE_F32: {
    fGgmlComputeForwardOptStepAdamwF32(pParams, pDst);
  } break;
  default: {
    GGML_ABORT("fatal error");
  }
  }
}

static void fGgmlComputeForwardOptStepSgdF32(const ggml_compute_params *pParams, ggml_tensor *dst) {
  const ggml_tensor *src0 = dst->src[0];
  const ggml_tensor *vSrc0Grad = dst->src[1];
  const ggml_tensor *vSgdParams = dst->src[2];

  GGML_ASSERT(ggml_are_same_shape(src0, vSrc0Grad));
  GGML_ASSERT(ggml_nelements(vSgdParams) == 2);

  const int cIth = pParams->ith;
  const int cNth = pParams->nth;

  const int cNr = ggml_nrows(src0);

  GGML_TENSOR_UNARY_OP_LOCALS
  GGML_ASSERT(nb00 == sizeof(float));

  // rows per thread
  const int cDr = (cNr + cNth - 1) / cNth;

  // row range for this thread
  const int cIr0 = cDr * cIth;
  const int cIr1 = MIN(cIr0 + cDr, cNr);

  // using adamw param subset we care about - alpha, wd - could have a separate struct
  const float *vSgdParamsPtr = ggml_get_data_f32(vSgdParams);
  const float cAlpha = vSgdParamsPtr[0];
  const float cKeep = 1.f - cAlpha * vSgdParamsPtr[1];

  for (int vIr = cIr0; vIr < cIr1; ++vIr) {
    const int64_t cI03 = vIr / (ne02 * ne01);
    const int64_t cI02 = (vIr - cI03 * ne02 * ne01) / ne01;
    const int64_t cI01 = (vIr - cI03 * ne02 * ne01 - cI02 * ne01);

    const size_t cOffset = cI03 * nb03 + cI02 * nb02 + cI01 * nb01;

    float *vW = (float *)((char *)src0->data + cOffset);                        // weight
    const float *vG = (const float *)((const char *)vSrc0Grad->data + cOffset); // grad

    for (int vI00 = 0; vI00 < ne00; ++vI00) {
      vW[vI00] = vW[vI00] * cKeep - cAlpha * vG[vI00];
    }
  }
}

void ggml_compute_forward_opt_step_sgd(const ggml_compute_params *pParams, ggml_tensor *pDst) {
  const ggml_tensor *vSrc0 = pDst->src[0];

  switch (vSrc0->type) {
  case GGML_TYPE_F32: {
    fGgmlComputeForwardOptStepSgdF32(pParams, pDst);
  } break;
  default: {
    GGML_ABORT("fatal error - sgd is F32 only");
  }
  }
}

static void fGgmlComputeForwardFwhtF32(const ggml_compute_params *pParams, ggml_tensor *dst) {
  const ggml_tensor *src0 = dst->src[0];
  const ggml_tensor *src1 = dst->src[1];

  GGML_ASSERT(src1->type == GGML_TYPE_F32);
  GGML_ASSERT(dst->type == GGML_TYPE_F32);

  GGML_TENSOR_BINARY_OP_LOCALS

  const int cIth = pParams->ith;
  const int cNth = pParams->nth;

  const int64_t cN = ne10;
  GGML_ASSERT((cN & (cN - 1)) == 0); // must be power of 2

  const int64_t cNr = ne11 * ne12 * ne13;
  const int64_t cRowsPerThread = (cNr + cNth - 1) / cNth;
  const int64_t cStartRow = cIth * cRowsPerThread;
  const int64_t cEndRow = MIN(cStartRow + cRowsPerThread, cNr);

  const float cScale = 1.0f / sqrtf((float)cN);

#if defined(GGML_SIMD)
  const GGML_F32_VEC cVMinusOne = GGML_F32_VEC_SET1(-1.0f);
#endif

  for (int64_t vR = cStartRow; vR < cEndRow; vR++) {
    const int64_t cI13 = vR / (ne11 * ne12);
    const int64_t cI12 = (vR - cI13 * ne11 * ne12) / ne11;
    const int64_t cI11 = vR - cI13 * ne11 * ne12 - cI12 * ne11;

    const float *vSrcRow = (const float *)((const char *)src1->data + cI11 * nb11 + cI12 * nb12 + cI13 * nb13);
    float *vDstRow = (float *)((char *)dst->data + cI11 * nb1 + cI12 * nb2 + cI13 * nb3);

    for (int64_t vJ = 0; vJ < cN; vJ++) {
      vDstRow[vJ] = vSrcRow[vJ] * cScale;
    }

    // Scalar passes
#if defined(GGML_SIMD)
#if defined(__ARM_FEATURE_SVE)
    const int cStep = svcntw();
#else
    const int cStep = GGML_F32_EPR;
#endif
#else
    const int cStep = cN;
#endif
    for (int64_t vLen = 1; vLen < cStep && vLen < cN; vLen <<= 1) {
      for (int64_t vI = 0; vI < cN; vI += 2 * vLen) {
        for (int64_t vJ = 0; vJ < vLen; vJ++) {
          float vU = vDstRow[vI + vJ];
          float v_ = vDstRow[vI + vLen + vJ];
          vDstRow[vI + vJ] = vU + v_;
          vDstRow[vI + vLen + vJ] = vU - v_;
        }
      }
    }

    // SIMD passes using GGML_F32_VEC_* macros for multi-architecture support
#if defined(GGML_SIMD)
    for (int64_t vLen = cStep; vLen < cN; vLen <<= 1) {
      for (int64_t vI = 0; vI < cN; vI += 2 * vLen) {
        for (int64_t vJ = 0; vJ < vLen; vJ += cStep) {
          GGML_F32_VEC vU = GGML_F32_VEC_LOAD(vDstRow + vI + vJ);
          GGML_F32_VEC v_ = GGML_F32_VEC_LOAD(vDstRow + vI + vLen + vJ);

          GGML_F32_VEC_STORE(vDstRow + vI + vJ, GGML_F32_VEC_ADD(vU, v_));
          GGML_F32_VEC_STORE(vDstRow + vI + vLen + vJ, GGML_F32_VEC_FMA(vU, v_, cVMinusOne));
        }
      }
    }
#endif
  }
}

void ggml_compute_forward_fwht(const ggml_compute_params *pParams, ggml_tensor *pDst) {
  const ggml_tensor *vSrc1 = pDst->src[1];

  switch (vSrc1->type) {
  case GGML_TYPE_F32: {
    fGgmlComputeForwardFwhtF32(pParams, pDst);
  } break;
  default: {
    GGML_ABORT("fatal error - fwht is F32 only");
  }
  }
}

// ggml_compute_forward_lightning_indexer

void ggml_compute_forward_lightning_indexer(const ggml_compute_params *pParams, ggml_tensor *pDst) {

  const ggml_tensor *vQ = pDst->src[0];
  const ggml_tensor *vK = pDst->src[1];
  const ggml_tensor *vW = pDst->src[2]; // weights
  const ggml_tensor *vM = pDst->src[3]; // mask

  GGML_ASSERT(pDst->type == GGML_TYPE_F32);
  GGML_ASSERT(vQ->type == GGML_TYPE_F32);
  GGML_ASSERT(vW->type == GGML_TYPE_F32);
  GGML_ASSERT(vM->type == GGML_TYPE_F16);

  GGML_TENSOR_LOCALS(int64_t, neq, vQ, ne)
  GGML_TENSOR_LOCALS(size_t, nbq, vQ, nb)
  GGML_TENSOR_LOCALS(int64_t, nek, vK, ne)
  GGML_TENSOR_LOCALS(size_t, nbk, vK, nb)
  GGML_TENSOR_LOCALS(int64_t, new, vW, ne)
  GGML_TENSOR_LOCALS(size_t, nbw, vW, nb)
  GGML_TENSOR_LOCALS(int64_t, nem, vM, ne)
  GGML_TENSOR_LOCALS(size_t, nbm, vM, nb)
  GGML_TENSOR_LOCALS(int64_t, ne, pDst, ne)
  GGML_TENSOR_LOCALS(size_t, nb, pDst, nb)

  GGML_ASSERT(nb0 == ggml_type_size(pDst->type));
  GGML_ASSERT(nbq0 == ggml_type_size(vQ->type));
  GGML_ASSERT(nbk0 == ggml_type_size(vK->type));
  GGML_ASSERT(nbw0 == ggml_type_size(vW->type));
  GGML_ASSERT(nbm0 == ggml_type_size(vM->type));

  const int cNEmbd = vQ->ne[0];
  const int cNHead = vQ->ne[1];
  const int cNTokens = vQ->ne[2];
  const int cNStream = vQ->ne[3];
  const int cNKv = vK->ne[2];

  ggml_to_float_t const cKToFloat = ggml_get_type_traits(vK->type)->to_float;
  GGML_ASSERT((vK->type == GGML_TYPE_F32 || cKToFloat) && "lightning indexer: unsupported K-type");

  const int cNr = cNKv;
  const int cIth = pParams->ith;
  const int cNth = pParams->nth;

  // (temporary) buffer for K converted to float
  float *vKRowF32 = (float *)pParams->wdata + cIth * (1 * cNEmbd + CACHE_LINE_SIZE_F32);

  // rows per thread
  const int cDr = (cNr + cNth - 1) / cNth;

  // row range for this thread
  const int cIr0 = cDr * cIth;
  const int cIr1 = MIN(cIr0 + cDr, cNr);

  for (int vS = 0; vS < cNStream; ++vS) {
    for (int vT = 0; vT < cNTokens; ++vT) {
      const float *vWRow = (float *)((char *)vW->data + vT * nbw1 + vS * nbw3);
      const ggml_fp16_t *vMRow = (ggml_fp16_t *)((char *)vM->data + vT * nbm1 + (vS % nem3) * nbm3);
      float *vDstRow = (float *)((char *)pDst->data + vT * nb1 + vS * nb3);
      for (int vIk = cIr0; vIk < cIr1; ++vIk) {
        char *vKRow = (char *)vK->data + vIk * nbk2 + vS * nbk3;
        if (cKToFloat) {
          cKToFloat(vKRow, vKRowF32, cNEmbd);
        } else {
          vKRowF32 = (float *)vKRow;
        }
        float vScore = 0.0f;
        for (int vH = 0; vH < cNHead; ++vH) {
          // dot product of q and k for head h
          float vQk = 0.0f;
          const float *vQRow = (float *)((char *)vQ->data + vH * nbq1 + vT * nbq2 + vS * nbq3);
          ggml_vec_dot_f32(cNEmbd, &vQk, 0, vQRow, 0, vKRowF32, 0, 1);
          // ReLU and weights (prescaled)
          vScore += MAX(vQk, 0.0f) * vWRow[vH];
        }
        // apply mask
        vDstRow[vIk] = vScore + GGML_CPU_FP16_TO_FP32(vMRow[vIk]);
      }
    }
  }
}
