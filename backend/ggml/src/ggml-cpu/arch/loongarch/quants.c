#define GGML_COMMON_IMPL_C
#include "ggml-common.h"
#include "ggml-quants.h"
#include "ggml-impl.h"
#include "ggml-cpu.h"
#include "simd-mappings.h"

#include "../../quants.h"
#include "../../ggml-cpu-impl.h"

#include <math.h>
#include <string.h>
#include <assert.h>
#include <float.h>
#include <stdlib.h> // for qsort
#include <stdio.h>  // for GGML_ASSERT

#define GROUP_MAX_EPS 1e-15f
#define GROUP_MAX_EPS_IQ3_XXS 1e-8f
#define GROUP_MAX_EPS_IQ2_S 1e-8f
#define GROUP_MAX_EPS_IQ1_M 1e-7f
#define GROUP_MAX_EPS_IQ1_S 1e-12f

#define UNUSED GGML_UNUSED

#if defined(__loongarch_sx)

static __m128i fLsxPacksW(__m128i pA, __m128i pB) {
  __m128i vTmp, vTmp1;
  vTmp = __lsx_vsat_w(pA, 15);
  vTmp1 = __lsx_vsat_w(pB, 15);
  return __lsx_vpickev_h(vTmp1, vTmp);
}

static __m128i fLsxPacksH(__m128i pA, __m128i pB) {
  __m128i vTmp, vTmp1;
  vTmp = __lsx_vsat_h(pA, 7);
  vTmp1 = __lsx_vsat_h(pB, 7);
  return __lsx_vpickev_b(vTmp1, vTmp);
}

static __m128i fLsxPackusH(__m128i pA, __m128i pB) {
  __m128i vTmp, vTmp1;
  vTmp = __lsx_vsat_hu(pA, 7);
  vTmp1 = __lsx_vsat_hu(pB, 7);
  return __lsx_vpickev_b(vTmp1, vTmp);
}

static __m128i fLsxMaddubsH(__m128i pA, __m128i pB) {
  __m128i vTmp1, vTmp2;
  vTmp1 = __lsx_vmulwev_h_b(pA, pB);
  vTmp2 = __lsx_vmulwod_h_b(pA, pB);
  return __lsx_vsadd_h(vTmp1, vTmp2);
}

static __m128i fLsxMaddH(__m128i pA, __m128i pB) {
  __m128i vTmp1, vTmp2;
  vTmp1 = __lsx_vmulwev_w_h(pA, pB);
  vTmp2 = __lsx_vmulwod_w_h(pA, pB);
  return __lsx_vadd_w(vTmp1, vTmp2);
}

static __m128i fLsxSetW(int32_t pA, int32_t pB, int32_t pC, int32_t pD) {
  v4i32 __ret = {pD, pC, pB, pA};
  return (__m128i)__ret;
}

static __m128i fLsxShuffleB(__m128i pA, __m128i pB) {
  __m128i vMaskF, vZero, vTmp0, vTmp2, vMask;
  int vF = 0x8f;
  vMaskF = __lsx_vreplgr2vr_b(vF);
  vZero = __lsx_vldi(0);
  vTmp0 = __lsx_vand_v(pB, vMaskF);   // get mask with low 4 bit and sign bits
  vTmp0 = __lsx_vori_b(vTmp0, 0x10);  // make each mask or  with 0x10 prepare for positive
  vMask = __lsx_vsle_b(vZero, vTmp0); // if mask >= 0, set mask
  vTmp2 = __lsx_vand_v(vTmp0, vMask); // maskout the in2 < ones
  return __lsx_vshuf_b(pA, vZero, vTmp2);
}

static __m128i fLsxHaddH(__m128i pA, __m128i pB) {
  __m128i vTmp1 = __lsx_vpickev_h(pB, pA);
  __m128i vTmp2 = __lsx_vpickod_h(pB, pA);
  return __lsx_vadd_h(vTmp1, vTmp2);
}

static __m128i fLsxHaddW(__m128i pA, __m128i pB) {
  __m128i vTmp1 = __lsx_vpickev_w(pB, pA);
  __m128i vTmp2 = __lsx_vpickod_w(pB, pA);
  return __lsx_vadd_w(vTmp1, vTmp2);
}

static __m128 fLsxHaddS(__m128 pA, __m128 pB) {
  __m128 vTmp1 = (__m128)__lsx_vpickev_w((__m128i)pB, (__m128i)pA);
  __m128 vTmp2 = (__m128)__lsx_vpickod_w((__m128i)pB, (__m128i)pA);

  return __lsx_vfadd_s(vTmp1, vTmp2);
}

static inline float fHsumFloat4x4(const __m128 pA, const __m128 pB, const __m128 pC, const __m128 pD) {
  __m128 vRes0 = fLsxHaddS(pA, pB);
  __m128 vRes1 = fLsxHaddS(pC, pD);
  __m128 vRes = fLsxHaddS(vRes0, vRes1);
  vRes = fLsxHaddS(vRes, vRes);
  vRes = fLsxHaddS(vRes, vRes);

  return ((v4f32)vRes)[0];
}

// multiply int8_t, add results pairwise twice
static inline __m128i fMulSumI8Pairs(const __m128i pX, const __m128i pY) {
  // Get absolute values of x vectors
  const __m128i cAx = __lsx_vsigncov_b(pX, pX);
  // Sign the values of the y vectors
  const __m128i cSy = __lsx_vsigncov_b(pX, pY);
  // Perform multiplication and create 16-bit values
  const __m128i cDot = fLsxMaddubsH(cAx, cSy);
  const __m128i cOnes = __lsx_vreplgr2vr_h(1);
  return fLsxMaddH(cOnes, cDot);
}
#endif

#if defined(__loongarch_asx)

#ifdef __clang__
#define VREGS_PREFIX "$vr"
#define XREGS_PREFIX "$xr"
#else // GCC
#define VREGS_PREFIX "$f"
#define XREGS_PREFIX "$f"
#endif
#define __ALL_REGS "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31"
// Convert __m128i to __m256i
static inline __m256i ____m256i(__m128i in) {
  __m256i out = __lasx_xvldi(0);
  __asm__ volatile(".irp i," __ALL_REGS "\n\t"
                   " .ifc %[out], " XREGS_PREFIX "\\i    \n\t"
                   "  .irp j," __ALL_REGS "\n\t"
                   "   .ifc %[in], " VREGS_PREFIX "\\j  \n\t"
                   "    xvpermi.q $xr\\i, $xr\\j, 0x20  \n\t"
                   "   .endif                           \n\t"
                   "  .endr                             \n\t"
                   " .endif                             \n\t"
                   ".endr                               \n\t"
                   : [out] "+f"(out)
                   : [in] "f"(in));
  return out;
}
// Convert two __m128i to __m256i
static inline __m256i fLasxSetQ(__m128i pInhi, __m128i pInlo) {
  __m256i out;
  __asm__ volatile(".irp i," __ALL_REGS "\n\t"
                   " .ifc %[hi], " VREGS_PREFIX "\\i    \n\t"
                   "  .irp j," __ALL_REGS "\n\t"
                   "   .ifc %[lo], " VREGS_PREFIX "\\j  \n\t"
                   "    xvpermi.q $xr\\i, $xr\\j, 0x20  \n\t"
                   "   .endif                           \n\t"
                   "  .endr                             \n\t"
                   " .endif                             \n\t"
                   ".endr                               \n\t"
                   ".ifnc %[out], %[hi]                 \n\t"
                   ".irp i," __ALL_REGS "\n\t"
                   " .ifc %[out], " XREGS_PREFIX "\\i   \n\t"
                   "  .irp j," __ALL_REGS "\n\t"
                   "   .ifc %[hi], " VREGS_PREFIX "\\j  \n\t"
                   "    xvori.b $xr\\i, $xr\\j, 0       \n\t"
                   "   .endif                           \n\t"
                   "  .endr                             \n\t"
                   " .endif                             \n\t"
                   ".endr                               \n\t"
                   ".endif                              \n\t"
                   : [out] "=f"(out), [hi] "+f"(pInhi)
                   : [lo] "f"(pInlo));
  return out;
}
// Convert __m256i low part to __m128i
static inline __m128i fLasxExtracti128Lo(__m256i in) {
  __m128i out;
  __asm__ volatile(".ifnc %[out], %[in]                 \n\t"
                   ".irp i," __ALL_REGS "\n\t"
                   " .ifc %[out], " VREGS_PREFIX "\\i   \n\t"
                   "  .irp j," __ALL_REGS "\n\t"
                   "   .ifc %[in], " XREGS_PREFIX "\\j  \n\t"
                   "    vori.b $vr\\i, $vr\\j, 0        \n\t"
                   "   .endif                           \n\t"
                   "  .endr                             \n\t"
                   " .endif                             \n\t"
                   ".endr                               \n\t"
                   ".endif                              \n\t"
                   : [out] "=f"(out)
                   : [in] "f"(in));
  return out;
}
// Convert __m256i high part to __m128i
static inline __m128i fLasxExtracti128Hi(__m256i in) {
  __m128i out;
  __asm__ volatile(".irp i," __ALL_REGS "\n\t"
                   " .ifc %[out], " VREGS_PREFIX "\\i   \n\t"
                   "  .irp j," __ALL_REGS "\n\t"
                   "   .ifc %[in], " XREGS_PREFIX "\\j  \n\t"
                   "    xvpermi.q $xr\\i, $xr\\j, 0x11  \n\t"
                   "   .endif                           \n\t"
                   "  .endr                             \n\t"
                   " .endif                             \n\t"
                   ".endr                               \n\t"
                   : [out] "=f"(out)
                   : [in] "f"(in));
  return out;
}

static __m256i fLasxSetW(int pE7, int pE6, int pE5, int pE4, int pE3, int pE2, int pE1, int pE0) {
  v8i32 __ret = {pE0, pE1, pE2, pE3, pE4, pE5, pE6, pE7};
  return (__m256i)__ret;
}

static __m256i fLasxSetD(int64_t pA, int64_t pB, int64_t pC, int64_t pD) {
  v4i64 __ret = {pD, pC, pB, pA};
  return (__m256i)__ret;
}

static __m256i fLasxInsertf128(__m128i pX, __m128i pY) { return fLasxSetQ(pX, pY); }

static __m256i fLasxShuffleB(__m256i pA, __m256i pB) {
  __m256i vMaskF, vZero, vTmp0, vTmp2, vMask;
  int vF = 0x8f;
  vMaskF = __lasx_xvreplgr2vr_b(vF);
  vZero = __lasx_xvldi(0);
  vTmp0 = __lasx_xvand_v(pB, vMaskF);   // get mask with low 4 bit and sign bits
  vTmp0 = __lasx_xvori_b(vTmp0, 0x10);  // make each mask or  with 0x10 prepare for positive
  vMask = __lasx_xvsle_b(vZero, vTmp0); // if mask >= 0, set mask
  vTmp2 = __lasx_xvand_v(vTmp0, vMask); // maskout the in2 < ones
  return __lasx_xvshuf_b(pA, vZero, vTmp2);
}

static __m256i fLasxExtu816(__m128i pA) { return __lasx_vext2xv_hu_bu(____m256i(pA)); }

static __m256i fLasxExt816(__m128i pA) { return __lasx_vext2xv_h_b(____m256i(pA)); }

static __m256i fLasxExt1632(__m128i pA) { return __lasx_vext2xv_w_h(____m256i(pA)); }

static __m128i fLasxExtracti128(__m256i pA, int pPos) {
  __m128i vRet;
  if (pPos == 0) {
    vRet = fLasxExtracti128Lo(pA);
  } else {
    vRet = fLasxExtracti128Hi(pA);
  }
  return vRet;
}

static __m128 fLasxExtractf128(__m256 pA, int pPos) {
  __m128 vRet;
  if (pPos == 0) {
    vRet = (__m128)fLasxExtracti128Lo((__m256i)pA);
  } else {
    vRet = (__m128)fLasxExtracti128Hi((__m256i)pA);
  }
  return vRet;
}

static __m256i fLasxMaddubsH(__m256i pA, __m256i pB) {
  __m256i vTmp1, vTmp2;
  vTmp1 = __lasx_xvmulwev_h_b(pA, pB);
  vTmp2 = __lasx_xvmulwod_h_b(pA, pB);
  return __lasx_xvsadd_h(vTmp1, vTmp2);
}

static __m256i fLasxMaddH(__m256i pA, __m256i pB) {
  __m256i vTmp1, vTmp2;
  vTmp1 = __lasx_xvmulwev_w_h(pA, pB);
  vTmp2 = __lasx_xvmulwod_w_h(pA, pB);
  return __lasx_xvadd_w(vTmp1, vTmp2);
}

static __m256i fLasxPacksW(__m256i pA, __m256i pB) {
  __m256i vTmp, vTmp1;
  vTmp = __lasx_xvsat_w(pA, 15);
  vTmp1 = __lasx_xvsat_w(pB, 15);
  return __lasx_xvpickev_h(vTmp1, vTmp);
}

static __m256i fLasxPacksH(__m256i pA, __m256i pB) {
  __m256i vTmp, vTmp1;
  vTmp = __lasx_xvsat_h(pA, 7);
  vTmp1 = __lasx_xvsat_h(pB, 7);
  return __lasx_xvpickev_b(vTmp1, vTmp);
}

static inline __m256i fLasxMaddHB(__m256i pA, __m256i pB) {
  __m256i vTmp1, vTmp2;
  vTmp1 = __lasx_xvmulwev_h_b(pA, pB);
  vTmp2 = __lasx_xvmulwod_h_b(pA, pB);
  return __lasx_xvadd_h(vTmp1, vTmp2);
}

static inline __m256i fLasxXvrepl128veiH(__m256i pA, const unsigned int pB) {
  switch (pB) {
  case 0:
    return __lasx_xvrepl128vei_h(pA, 0);
  case 1:
    return __lasx_xvrepl128vei_h(pA, 1);
  case 2:
    return __lasx_xvrepl128vei_h(pA, 2);
  case 3:
    return __lasx_xvrepl128vei_h(pA, 3);
  case 4:
    return __lasx_xvrepl128vei_h(pA, 4);
  case 5:
    return __lasx_xvrepl128vei_h(pA, 5);
  case 6:
    return __lasx_xvrepl128vei_h(pA, 6);
  case 7:
    return __lasx_xvrepl128vei_h(pA, 7);
  default:
    __builtin_unreachable();
  }
}

static inline __m256i fLasxXvandiBBit(__m256i pA, const unsigned int pB) {
  switch (pB) {
  case 0:
    return __lasx_xvandi_b(pA, 1 << 0);
  case 1:
    return __lasx_xvandi_b(pA, 1 << 1);
  case 2:
    return __lasx_xvandi_b(pA, 1 << 2);
  case 3:
    return __lasx_xvandi_b(pA, 1 << 3);
  case 4:
    return __lasx_xvandi_b(pA, 1 << 4);
  case 5:
    return __lasx_xvandi_b(pA, 1 << 5);
  case 6:
    return __lasx_xvandi_b(pA, 1 << 6);
  case 7:
    return __lasx_xvandi_b(pA, 1 << 7);
  default:
    __builtin_unreachable();
  }
}

// horizontally add 8 floats
static inline float fHsumFloat8(const __m256 pX) {
  __m128 vRes = fLasxExtractf128(pX, 1);
  vRes = __lsx_vfadd_s(vRes, fLasxExtractf128(pX, 0));
  vRes = __lsx_vfadd_s(vRes, (__m128)__lsx_vpickod_d((__m128i)vRes, (__m128i)vRes));
  vRes = __lsx_vfadd_s(vRes, (__m128)__lsx_vinsgr2vr_w(__lsx_vldi(0), __lsx_vpickve2gr_w(vRes, 1), 0));
  return ((v4f32)vRes)[0];
}

// horizontally add 8 int32_t
static inline int fHsumI328(const __m256i pA) {

  __m256i vTmp1 = __lasx_xvpermi_q(pA, pA, 0x11);
  __m256i vTmp2 = __lasx_xvpermi_q(pA, pA, 0x00);

  __m128i vTmp1128 = fLasxExtracti128Lo(vTmp1);
  __m128i vTmp2128 = fLasxExtracti128Lo(vTmp2);

  __m128i vSum128 = __lsx_vadd_w(vTmp1128, vTmp2128);

  __m128i vEv = __lsx_vpickev_w(vSum128, vSum128);
  __m128i vOd = __lsx_vpickod_w(vSum128, vSum128);
  __m128i vSum64 = __lsx_vadd_w(vEv, vOd);

  int vSum641, vSum642;
  vSum641 = __lsx_vpickve2gr_w(vSum64, 0);
  vSum642 = __lsx_vpickve2gr_w(vSum64, 1);

  return vSum641 + vSum642;
}

// horizontally add 4 int32_t
static inline int fHsumI324(const __m128i pA) {
  __m128i vEv = __lsx_vpickev_w(pA, pA);
  __m128i vOd = __lsx_vpickod_w(pA, pA);
  __m128i vSum64 = __lsx_vadd_w(vEv, vOd);

  int vSum641, vSum642;
  vSum641 = __lsx_vpickve2gr_w(vSum64, 0);
  vSum642 = __lsx_vpickve2gr_w(vSum64, 1);

  return vSum641 + vSum642;
}

// spread 32 bits to 32 bytes { 0x00, 0xFF }
static inline __m256i fBytesFromBits32(const uint8_t *pX) {

  uint32_t vX32;
  memcpy(&vX32, pX, sizeof(uint32_t));
  const __m256i cShufMask = fLasxSetD(0x0303030303030303, 0x0202020202020202, 0x0101010101010101, 0x0000000000000000);

  __m256i vBytes = fLasxShuffleB(__lasx_xvreplgr2vr_w(vX32), cShufMask);
  const __m256i cBitMask = __lasx_xvreplgr2vr_d(0x7fbfdfeff7fbfdfe);
  vBytes = __lasx_xvor_v(vBytes, cBitMask);
  return __lasx_xvseq_b(vBytes, __lasx_xvreplgr2vr_d(-1));
}

// Unpack 32 4-bit fields into 32 bytes
// The output vector contains 32 bytes, each one in [ 0 .. 15 ] interval
static inline __m256i fBytesFromNibbles32(const uint8_t *pRsi) {
  const __m128i cLo = __lsx_vld((const __m128i *)pRsi, 0);
  __m128i vHi = __lsx_vsrli_h(cLo, 4);
  return __lasx_xvandi_b(fLasxInsertf128(vHi, cLo), 0xf);
}

// add int16_t pairwise and return as float vector
static inline __m256 fSumI16PairsFloat(const __m256i pX) {
  __m256i v_ = __lasx_xvpackod_h(pX, pX);
  __m256i vSummedPairs = __lasx_xvaddwev_w_h(pX, v_);
  return __lasx_xvffint_s_w(vSummedPairs);
}

static inline __m256 fMulSumUs8PairsFloat(const __m256i pAx, const __m256i pSy) {
  // Perform multiplication and create 16-bit values
  const __m256i cDot = fLasxMaddubsH(pAx, pSy);
  return fSumI16PairsFloat(cDot);
}

// multiply int8_t, add results pairwise twice and return as float vector
static inline __m256 fMulSumI8PairsFloat(const __m256i pX, const __m256i pY) {
  const __m256i cDot = fLasxMaddHB(pX, pY);
  return fSumI16PairsFloat(cDot);
}

static inline __m128i fPackNibbles(__m256i pBytes) {
  // Move bits within 16-bit lanes from 0000_abcd_0000_efgh into 0000_0000_abcd_efgh
  const __m256i cLowByte = __lasx_xvreplgr2vr_h(0xFF);
  __m256i vHigh = __lasx_xvandn_v(cLowByte, pBytes);
  __m256i vLow = __lasx_xvand_v(cLowByte, pBytes);
  vHigh = __lasx_xvsrli_h(vHigh, 4);
  pBytes = __lasx_xvor_v(vLow, vHigh);
  // Compress uint16_t lanes into bytes
  __m128i *vR0 = (__m128i *)&pBytes;
  __m256i vTmpH128 = __lasx_xvpermi_q(pBytes, pBytes, 0x11);
  __m128i *vR1 = (__m128i *)&vTmpH128;

  __m128i vZero = __lsx_vldi(0);
  __m128i vTmp, vTmp2, vTmp3;

  vTmp = __lsx_vmax_h(vZero, *vR0);
  vTmp2 = __lsx_vsat_hu(vTmp, 7);

  vTmp = __lsx_vmax_h(vZero, *vR1);
  vTmp3 = __lsx_vsat_hu(vTmp, 7);
  return __lsx_vpickev_b(vTmp3, vTmp2);
}
#endif //__loongarch_asx

void quantize_row_q8_0(const float *GGML_RESTRICT pX, void *GGML_RESTRICT pVy, int64_t pK) {
  assert(QK8_0 == 32);
  assert(pK % QK8_0 == 0);
  const int cNb = pK / QK8_0;

  block_q8_0 *GGML_RESTRICT vY = pVy;

#if defined(__loongarch_asx)
  for (int vI = 0; vI < cNb; vI++) {
    __m256 v0 = (__m256)__lasx_xvld(pX, 0);
    __m256 v1 = (__m256)__lasx_xvld(pX, 32);
    __m256 v2 = (__m256)__lasx_xvld(pX, 64);
    __m256 v3 = (__m256)__lasx_xvld(pX, 96);
    pX += 32;

    // Compute max(abs(e)) for the block
    const __m256 cSignBit = __lasx_xvreplfr2vr_s(-0.0f);
    __m256 vMaxAbs = (__m256)__lasx_xvandn_v((__m256i)cSignBit, (__m256i)v0);
    vMaxAbs = __lasx_xvfmax_s(vMaxAbs, (__m256)__lasx_xvandn_v((__m256i)cSignBit, (__m256i)v1));
    vMaxAbs = __lasx_xvfmax_s(vMaxAbs, (__m256)__lasx_xvandn_v((__m256i)cSignBit, (__m256i)v2));
    vMaxAbs = __lasx_xvfmax_s(vMaxAbs, (__m256)__lasx_xvandn_v((__m256i)cSignBit, (__m256i)v3));

    __m128 vMax4 = __lsx_vfmax_s(fLasxExtractf128(vMaxAbs, 1), fLasxExtractf128(vMaxAbs, 0));
    vMax4 = __lsx_vfmax_s(vMax4, (__m128)__lsx_vpickod_d((__m128i)vMax4, (__m128i)vMax4));
    __m128 vTmp = vMax4;
    vMax4 = __lsx_vfmax_s(vMax4, (__m128)__lsx_vinsgr2vr_w(vTmp, __lsx_vpickve2gr_w(vMax4, 1), 0));
    const float cMaxScalar = ((v4f32)vMax4)[0];

    // Quantize these floats
    const float cD = cMaxScalar / 127.f;
    vY[vI].d = GGML_CPU_FP32_TO_FP16(cD);
    const float cId = (cMaxScalar != 0.0f) ? 127.f / cMaxScalar : 0.0f;
    const __m256 cMul = (__m256)__lasx_xvreplfr2vr_s(cId);

    // Apply the multiplier
    v0 = __lasx_xvfmul_s(v0, cMul);
    v1 = __lasx_xvfmul_s(v1, cMul);
    v2 = __lasx_xvfmul_s(v2, cMul);
    v3 = __lasx_xvfmul_s(v3, cMul);

    // Round to nearest integer
    __m256i vI0 = __lasx_xvftintrne_w_s(v0);
    __m256i vI1 = __lasx_xvftintrne_w_s(v1);
    __m256i vI2 = __lasx_xvftintrne_w_s(v2);
    __m256i vI3 = __lasx_xvftintrne_w_s(v3);

    __m128i vNi0 = fLasxExtracti128(vI0, 0);
    __m128i vNi1 = fLasxExtracti128(vI0, 1);
    __m128i vNi2 = fLasxExtracti128(vI1, 0);
    __m128i vNi3 = fLasxExtracti128(vI1, 1);
    __m128i vNi4 = fLasxExtracti128(vI2, 0);
    __m128i vNi5 = fLasxExtracti128(vI2, 1);
    __m128i vNi6 = fLasxExtracti128(vI3, 0);
    __m128i vNi7 = fLasxExtracti128(vI3, 1);

    // Convert int32 to int16
    vNi0 = fLsxPacksW(vNi0, vNi1);
    vNi2 = fLsxPacksW(vNi2, vNi3);
    vNi4 = fLsxPacksW(vNi4, vNi5);
    vNi6 = fLsxPacksW(vNi6, vNi7);
    // Convert int16 to int8
    vNi0 = fLsxPacksH(vNi0, vNi2);
    vNi4 = fLsxPacksH(vNi4, vNi6);

    __lsx_vst(vNi0, (__m128i *)(vY[vI].qs + 0), 0);
    __lsx_vst(vNi4, (__m128i *)(vY[vI].qs + 16), 0);
  }
#else
  GGML_UNUSED(cNb);
  // scalar
  quantize_row_q8_0_ref(pX, vY, pK);
#endif
}

void quantize_row_q8_1(const float *GGML_RESTRICT pX, void *GGML_RESTRICT pVy, int64_t pK) {
  assert(pK % QK8_1 == 0);
  const int cNb = pK / QK8_1;

  block_q8_1 *GGML_RESTRICT vY = pVy;

#if defined(__loongarch_asx)
  for (int vI = 0; vI < cNb; vI++) {
    __m256 v0 = (__m256)__lasx_xvld(pX, 0);
    __m256 v1 = (__m256)__lasx_xvld(pX, 32);
    __m256 v2 = (__m256)__lasx_xvld(pX, 64);
    __m256 v3 = (__m256)__lasx_xvld(pX, 96);
    pX += 32;

    // Compute max(abs(e)) for the block
    const __m256 cSignBit = __lasx_xvreplfr2vr_s(-0.0f);
    __m256 vMaxAbs = (__m256)__lasx_xvandn_v((__m256i)cSignBit, (__m256i)v0);
    vMaxAbs = __lasx_xvfmax_s(vMaxAbs, (__m256)__lasx_xvandn_v((__m256i)cSignBit, (__m256i)v1));
    vMaxAbs = __lasx_xvfmax_s(vMaxAbs, (__m256)__lasx_xvandn_v((__m256i)cSignBit, (__m256i)v2));
    vMaxAbs = __lasx_xvfmax_s(vMaxAbs, (__m256)__lasx_xvandn_v((__m256i)cSignBit, (__m256i)v3));

    __m128 vMax4 = __lsx_vfmax_s(fLasxExtractf128(vMaxAbs, 1), fLasxExtractf128(vMaxAbs, 0));
    vMax4 = __lsx_vfmax_s(vMax4, (__m128)__lsx_vpickod_d((__m128i)vMax4, (__m128i)vMax4));
    __m128 vTmp = vMax4;
    vMax4 = __lsx_vfmax_s(vMax4, (__m128)__lsx_vextrins_w((__m128i)vTmp, (__m128i)vMax4, 0x1));
    const float cMaxScalar = ((v4f32)vMax4)[0];

    // Quantize these floats
    const float cD = cMaxScalar / 127.f;
    vY[vI].d = GGML_CPU_FP32_TO_FP16(cD);
    const float cId = (cMaxScalar != 0.0f) ? 127.f / cMaxScalar : 0.0f;
    const __m256 cMul = __lasx_xvreplfr2vr_s(cId);

    // Apply the multiplier
    v0 = __lasx_xvfmul_s(v0, cMul);
    v1 = __lasx_xvfmul_s(v1, cMul);
    v2 = __lasx_xvfmul_s(v2, cMul);
    v3 = __lasx_xvfmul_s(v3, cMul);

    // Round to nearest integer
    __m256i vI0 = __lasx_xvftintrne_w_s(v0);
    __m256i vI1 = __lasx_xvftintrne_w_s(v1);
    __m256i vI2 = __lasx_xvftintrne_w_s(v2);
    __m256i vI3 = __lasx_xvftintrne_w_s(v3);

    __m128i vNi0 = fLasxExtracti128(vI0, 0);
    __m128i vNi1 = fLasxExtracti128(vI0, 1);
    __m128i vNi2 = fLasxExtracti128(vI1, 0);
    __m128i vNi3 = fLasxExtracti128(vI1, 1);
    __m128i vNi4 = fLasxExtracti128(vI2, 0);
    __m128i vNi5 = fLasxExtracti128(vI2, 1);
    __m128i vNi6 = fLasxExtracti128(vI3, 0);
    __m128i vNi7 = fLasxExtracti128(vI3, 1);

    // Compute the sum of the quants and set y[i].s
    const __m128i cS0 = __lsx_vadd_w(__lsx_vadd_w(vNi0, vNi1), __lsx_vadd_w(vNi2, vNi3));
    const __m128i cS1 = __lsx_vadd_w(__lsx_vadd_w(vNi4, vNi5), __lsx_vadd_w(vNi6, vNi7));
    vY[vI].s = GGML_CPU_FP32_TO_FP16(cD * fHsumI324(__lsx_vadd_w(cS0, cS1)));

    // Convert int32 to int16
    vNi0 = fLsxPacksW(vNi0, vNi1);
    vNi2 = fLsxPacksW(vNi2, vNi3);
    vNi4 = fLsxPacksW(vNi4, vNi5);
    vNi6 = fLsxPacksW(vNi6, vNi7);
    // Convert int16 to int8
    vNi0 = fLsxPacksH(vNi0, vNi2);
    vNi4 = fLsxPacksH(vNi4, vNi6);

    __lsx_vst(vNi0, (__m128i *)(vY[vI].qs + 0), 0);
    __lsx_vst(vNi4, (__m128i *)(vY[vI].qs + 16), 0);
  }
#else
  GGML_UNUSED(cNb);
  // scalar
  quantize_row_q8_1_ref(pX, vY, pK);
#endif
}

//===================================== Dot products =================================

//
// Helper functions
//

#if defined(__loongarch_asx)
// shuffles to pick the required scales in dot products
static inline __m256i fGetScaleShuffleQ3k(int pI) {
  static const uint8_t cKShuffle[128] = {
      0,  1,  0,  1,  0,  1,  0,  1,  0,  1,  0,  1,  0,  1,  0,  1,  2,  3,  2,  3,  2,  3,  2,  3,  2,  3,
      2,  3,  2,  3,  2,  3,  4,  5,  4,  5,  4,  5,  4,  5,  4,  5,  4,  5,  4,  5,  4,  5,  6,  7,  6,  7,
      6,  7,  6,  7,  6,  7,  6,  7,  6,  7,  6,  7,  8,  9,  8,  9,  8,  9,  8,  9,  8,  9,  8,  9,  8,  9,
      8,  9,  10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 12, 13, 12, 13, 12, 13, 12, 13,
      12, 13, 12, 13, 12, 13, 12, 13, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15,
  };
  return __lasx_xvld((const __m256i *)cKShuffle + pI, 0);
}
static inline __m256i fGetScaleShuffleK4(int pI) {
  static const uint8_t cKShuffle[256] = {
      0,  1,  0,  1,  0,  1,  0,  1,  0,  1,  0,  1,  0,  1,  0,  1,  0,  1,  0,  1,  0,  1,  0,  1,  0,  1,
      0,  1,  0,  1,  0,  1,  2,  3,  2,  3,  2,  3,  2,  3,  2,  3,  2,  3,  2,  3,  2,  3,  2,  3,  2,  3,
      2,  3,  2,  3,  2,  3,  2,  3,  2,  3,  2,  3,  4,  5,  4,  5,  4,  5,  4,  5,  4,  5,  4,  5,  4,  5,
      4,  5,  4,  5,  4,  5,  4,  5,  4,  5,  4,  5,  4,  5,  4,  5,  4,  5,  6,  7,  6,  7,  6,  7,  6,  7,
      6,  7,  6,  7,  6,  7,  6,  7,  6,  7,  6,  7,  6,  7,  6,  7,  6,  7,  6,  7,  6,  7,  6,  7,  8,  9,
      8,  9,  8,  9,  8,  9,  8,  9,  8,  9,  8,  9,  8,  9,  8,  9,  8,  9,  8,  9,  8,  9,  8,  9,  8,  9,
      8,  9,  8,  9,  10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11,
      10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13,
      12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15,
      14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15};
  return __lasx_xvld((const __m256i *)cKShuffle + pI, 0);
}
static inline __m128i fGetScaleShuffle(int pI) {
  static const uint8_t cKShuffle[128] = {
      0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  1,  1,  1,  1,  1,  1,  2,  2,  2,  2,  2,  2,  2,  2,  3,  3,
      3,  3,  3,  3,  3,  3,  4,  4,  4,  4,  4,  4,  4,  4,  5,  5,  5,  5,  5,  5,  5,  5,  6,  6,  6,  6,
      6,  6,  6,  6,  7,  7,  7,  7,  7,  7,  7,  7,  8,  8,  8,  8,  8,  8,  8,  8,  9,  9,  9,  9,  9,  9,
      9,  9,  10, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 12, 12, 12,
      13, 13, 13, 13, 13, 13, 13, 13, 14, 14, 14, 14, 14, 14, 14, 14, 15, 15, 15, 15, 15, 15, 15, 15};
  return __lsx_vld((const __m128i *)cKShuffle + pI, 0);
}
#endif

void ggml_vec_dot_q4_0_q8_0(int pN, float *GGML_RESTRICT pS, size_t pBs, const void *GGML_RESTRICT pVx, size_t pBx,
                            const void *GGML_RESTRICT pVy, size_t pBy, int pNrc) {
  const int cQk = QK8_0;
  const int cNb = pN / cQk;

  assert(pN % cQk == 0);
  assert(pNrc == 1);
  UNUSED(pNrc);
  UNUSED(pBx);
  UNUSED(pBy);
  UNUSED(pBs);

  const block_q4_0 *GGML_RESTRICT vX = pVx;
  const block_q8_0 *GGML_RESTRICT vY = pVy;

  int vIb = 0;
  float vSumf = 0;

#if defined(__loongarch_asx)
  // Initialize accumulator with zeros
  __m256 vAcc = (__m256)__lasx_xvldi(0);

  // Main loop
  for (; vIb < cNb; ++vIb) {
    /* Compute combined scale for the block */
    const __m256 cD = __lasx_xvreplfr2vr_s(GGML_CPU_FP16_TO_FP32(vX[vIb].d) * GGML_CPU_FP16_TO_FP32(vY[vIb].d));

    __m256i vQx = fBytesFromNibbles32(vX[vIb].qs);

    // Now we have a vector with bytes in [ 0 .. 15 ] interval. Offset them into [ -8 .. +7 ] interval.
    const __m256i cOff = __lasx_xvreplgr2vr_b(8);
    vQx = __lasx_xvsub_b(vQx, cOff);

    __m256i vQy = __lasx_xvld((const __m256i *)vY[vIb].qs, 0);

    const __m256 cQ = fMulSumI8PairsFloat(vQx, vQy);

    /* Multiply q with scale and accumulate */
    vAcc = __lasx_xvfmadd_s(cD, cQ, vAcc);
  }

  vSumf = fHsumFloat8(vAcc);

#elif defined(__loongarch_sx)
  // set constants
  const __m128i low_mask = __lsx_vreplgr2vr_b(0xF);
  const __m128i cOff = __lsx_vreplgr2vr_b(8);

  // Initialize accumulator with zeros
  __m128 acc_0 = (__m128)__lsx_vldi(0);
  __m128 acc_1 = (__m128)__lsx_vldi(0);
  __m128 acc_2 = (__m128)__lsx_vldi(0);
  __m128 acc_3 = (__m128)__lsx_vldi(0);

  for (; vIb + 1 < cNb; vIb += 2) {

    // Compute combined scale for the block 0 and 1
    const float ft0 = GGML_CPU_FP16_TO_FP32(vX[vIb].d) * GGML_CPU_FP16_TO_FP32(vY[vIb].d);
    const __m128 d_0_1 = (__m128)(v4f32){ft0, ft0, ft0, ft0};

    const __m128i tmp_0_1 = __lsx_vld((const __m128i *)vX[vIb].qs, 0);

    __m128i bx_0 = __lsx_vand_v(low_mask, tmp_0_1);
    __m128i by_0 = __lsx_vld((const __m128i *)vY[vIb].qs, 0);
    bx_0 = __lsx_vsub_b(bx_0, cOff);
    const __m128i i32_0 = fMulSumI8Pairs(bx_0, by_0);

    __m128i bx_1 = __lsx_vand_v(low_mask, __lsx_vsrli_d(tmp_0_1, 4));
    __m128i by_1 = __lsx_vld((const __m128i *)(vY[vIb].qs + 16), 0);
    bx_1 = __lsx_vsub_b(bx_1, cOff);
    const __m128i i32_1 = fMulSumI8Pairs(bx_1, by_1);

    // Compute combined scale for the block 2 and 3
    const float ft1 = GGML_CPU_FP16_TO_FP32(vX[vIb + 1].d) * GGML_CPU_FP16_TO_FP32(vY[vIb + 1].d);
    const __m128 d_2_3 = (__m128)(v4f32){ft1, ft1, ft1, ft1};

    const __m128i tmp_2_3 = __lsx_vld((const __m128i *)vX[vIb + 1].qs, 0);

    __m128i bx_2 = __lsx_vand_v(low_mask, tmp_2_3);
    __m128i by_2 = __lsx_vld((const __m128i *)vY[vIb + 1].qs, 0);
    bx_2 = __lsx_vsub_b(bx_2, cOff);
    const __m128i i32_2 = fMulSumI8Pairs(bx_2, by_2);

    __m128i bx_3 = __lsx_vand_v(low_mask, __lsx_vsrli_d(tmp_2_3, 4));
    __m128i by_3 = __lsx_vld((const __m128i *)(vY[vIb + 1].qs + 16), 0);
    bx_3 = __lsx_vsub_b(bx_3, cOff);
    const __m128i i32_3 = fMulSumI8Pairs(bx_3, by_3);

    // Convert int32_t to float
    __m128 p0 = __lsx_vffint_s_w(i32_0);
    __m128 p1 = __lsx_vffint_s_w(i32_1);
    __m128 p2 = __lsx_vffint_s_w(i32_2);
    __m128 p3 = __lsx_vffint_s_w(i32_3);

    // Apply the scale
    __m128 p0_d = __lsx_vfmul_s(d_0_1, p0);
    __m128 p1_d = __lsx_vfmul_s(d_0_1, p1);
    __m128 p2_d = __lsx_vfmul_s(d_2_3, p2);
    __m128 p3_d = __lsx_vfmul_s(d_2_3, p3);

    // Acummulate
    acc_0 = __lsx_vfadd_s(p0_d, acc_0);
    acc_1 = __lsx_vfadd_s(p1_d, acc_1);
    acc_2 = __lsx_vfadd_s(p2_d, acc_2);
    acc_3 = __lsx_vfadd_s(p3_d, acc_3);
  }

  vSumf = fHsumFloat4x4(acc_0, acc_1, acc_2, acc_3);

#endif
  for (; vIb < cNb; ++vIb) {
    int vSumi0 = 0;
    int vSumi1 = 0;

    for (int vJ = 0; vJ < cQk / 2; ++vJ) {
      const int cV0 = (vX[vIb].qs[vJ] & 0x0F) - 8;
      const int cV1 = (vX[vIb].qs[vJ] >> 4) - 8;

      vSumi0 += (cV0 * vY[vIb].qs[vJ]);
      vSumi1 += (cV1 * vY[vIb].qs[vJ + cQk / 2]);
    }

    int vSumi = vSumi0 + vSumi1;
    vSumf += vSumi * GGML_CPU_FP16_TO_FP32(vX[vIb].d) * GGML_CPU_FP16_TO_FP32(vY[vIb].d);
  }

  *pS = vSumf;
}

void ggml_vec_dot_q4_1_q8_1(int pN, float *GGML_RESTRICT pS, size_t pBs, const void *GGML_RESTRICT pVx, size_t pBx,
                            const void *GGML_RESTRICT pVy, size_t pBy, int pNrc) {
  const int cQk = QK8_1;
  const int cNb = pN / cQk;

  assert(pN % cQk == 0);
  assert(pNrc == 1);
  UNUSED(pNrc);
  UNUSED(pBx);
  UNUSED(pBy);
  UNUSED(pBs);

  const block_q4_1 *GGML_RESTRICT vX = pVx;
  const block_q8_1 *GGML_RESTRICT vY = pVy;

  int vIb = 0;
  float vSumf = 0;

#if defined(__loongarch_asx)
  // Initialize accumulator with zeros
  __m256 vAcc = (__m256)__lasx_xvldi(0);

  float vSumms = 0;

  // Main loop
  for (; vIb < cNb; ++vIb) {
    const float cD0 = GGML_CPU_FP16_TO_FP32(vX[vIb].d);
    const float cD1 = GGML_CPU_FP16_TO_FP32(vY[vIb].d);

    vSumms += GGML_CPU_FP16_TO_FP32(vX[vIb].m) * GGML_CPU_FP16_TO_FP32(vY[vIb].s);

    const __m256 cD0v = __lasx_xvreplfr2vr_s(cD0);
    const __m256 cD1v = __lasx_xvreplfr2vr_s(cD1);

    // Compute combined scales
    const __m256 cD0d1 = __lasx_xvfmul_s(cD0v, cD1v);

    // Load 16 bytes, and unpack 4 bit fields into bytes, making 32 bytes
    const __m256i cQx = fBytesFromNibbles32(vX[vIb].qs);
    const __m256i cQy = __lasx_xvld((const __m256i *)vY[vIb].qs, 0);

    const __m256 cXy = fMulSumUs8PairsFloat(cQx, cQy);

    // Accumulate d0*d1*x*y
    vAcc = __lasx_xvfmadd_s(cD0d1, cXy, vAcc);
  }

  vSumf = fHsumFloat8(vAcc) + vSumms;

  *pS = vSumf;
#else
  UNUSED(cNb);
  UNUSED(vX);
  UNUSED(vY);
  UNUSED(vIb);
  UNUSED(vSumf);
  ggml_vec_dot_q4_1_q8_1_generic(pN, pS, pBs, pVx, pBx, pVy, pBy, pNrc);
#endif
}

void ggml_vec_dot_q5_0_q8_0(int pN, float *GGML_RESTRICT pS, size_t pBs, const void *GGML_RESTRICT pVx, size_t pBx,
                            const void *GGML_RESTRICT pVy, size_t pBy, int pNrc) {
  const int cQk = QK8_0;
  const int cNb = pN / cQk;

  int vIb = 0;
  float vSumf = 0;

  assert(pN % cQk == 0);
  assert(cQk == QK5_0);
  assert(pNrc == 1);
  UNUSED(pNrc);
  UNUSED(pBx);
  UNUSED(pBy);
  UNUSED(pBs);

  const block_q5_0 *GGML_RESTRICT vX = pVx;
  const block_q8_0 *GGML_RESTRICT vY = pVy;

#if defined(__loongarch_asx)
  // Initialize accumulator with zeros
  __m256 vAcc = (__m256)__lasx_xvldi(0);

  // Main loop
  for (; vIb < cNb; ++vIb) {
    /* Compute combined scale for the block */
    const __m256 cD = __lasx_xvreplfr2vr_s(GGML_CPU_FP16_TO_FP32(vX[vIb].d) * GGML_CPU_FP16_TO_FP32(vY[vIb].d)); // FIXME

    __m256i vQx = fBytesFromNibbles32(vX[vIb].qs);
    __m256i vBxhi = fBytesFromBits32(vX[vIb].qh);
    vBxhi = __lasx_xvandn_v(vBxhi, __lasx_xvreplgr2vr_b((char)0xF0));
    vQx = __lasx_xvor_v(vQx, vBxhi);

    __m256i vQy = __lasx_xvld((const __m256i *)vY[vIb].qs, 0);

    const __m256 cQ = fMulSumI8PairsFloat(vQx, vQy);

    /* Multiply q with scale and accumulate */
    vAcc = __lasx_xvfmadd_s(cD, cQ, vAcc);
  }

  vSumf = fHsumFloat8(vAcc);

  *pS = vSumf;
#else
  UNUSED(cNb);
  UNUSED(vIb);
  UNUSED(vSumf);
  UNUSED(vX);
  UNUSED(vY);
  ggml_vec_dot_q5_0_q8_0_generic(pN, pS, pBs, pVx, pBx, pVy, pBy, pNrc);
#endif
}

void ggml_vec_dot_q5_1_q8_1(int pN, float *GGML_RESTRICT pS, size_t pBs, const void *GGML_RESTRICT pVx, size_t pBx,
                            const void *GGML_RESTRICT pVy, size_t pBy, int pNrc) {
  const int cQk = QK8_1;
  const int cNb = pN / cQk;

  int vIb = 0;
  float vSumf = 0;

  assert(pN % cQk == 0);
  assert(cQk == QK5_1);
  assert(pNrc == 1);
  UNUSED(pNrc);
  UNUSED(pBx);
  UNUSED(pBy);
  UNUSED(pBs);

  const block_q5_1 *GGML_RESTRICT vX = pVx;
  const block_q8_1 *GGML_RESTRICT vY = pVy;

#if defined(__loongarch_asx)
  // Initialize accumulator with zeros
  __m256 vAcc = (__m256)__lasx_xvldi(0);

  float vSumms = 0.0f;

  // Main loop
  for (; vIb < cNb; ++vIb) {
    const __m256 cDx = __lasx_xvreplfr2vr_s(GGML_CPU_FP16_TO_FP32(vX[vIb].d));

    vSumms += GGML_CPU_FP16_TO_FP32(vX[vIb].m) * GGML_CPU_FP16_TO_FP32(vY[vIb].s);

    __m256i vQx = fBytesFromNibbles32(vX[vIb].qs);
    __m256i vBxhi = fBytesFromBits32(vX[vIb].qh);
    vBxhi = __lasx_xvand_v(vBxhi, __lasx_xvreplgr2vr_b(0x10));
    vQx = __lasx_xvor_v(vQx, vBxhi);

    const __m256 cDy = __lasx_xvreplfr2vr_s(GGML_CPU_FP16_TO_FP32(vY[vIb].d));
    const __m256i cQy = __lasx_xvld((const __m256i *)vY[vIb].qs, 0);

    const __m256 cQ = fMulSumUs8PairsFloat(vQx, cQy);

    vAcc = __lasx_xvfmadd_s(cQ, __lasx_xvfmul_s(cDx, cDy), vAcc);
  }

  vSumf = fHsumFloat8(vAcc) + vSumms;

  *pS = vSumf;
#else
  UNUSED(cNb);
  UNUSED(vIb);
  UNUSED(vSumf);
  UNUSED(vX);
  UNUSED(vY);
  ggml_vec_dot_q5_1_q8_1_generic(pN, pS, pBs, pVx, pBx, pVy, pBy, pNrc);
#endif
}

void ggml_vec_dot_q8_0_q8_0(int pN, float *GGML_RESTRICT pS, size_t pBs, const void *GGML_RESTRICT pVx, size_t pBx,
                            const void *GGML_RESTRICT pVy, size_t pBy, int pNrc) {
  const int cQk = QK8_0;
  const int cNb = pN / cQk;

  assert(pN % cQk == 0);
  assert(pNrc == 1);
  UNUSED(pNrc);
  UNUSED(pBx);
  UNUSED(pBy);
  UNUSED(pBs);

  const block_q8_0 *GGML_RESTRICT vX = pVx;
  const block_q8_0 *GGML_RESTRICT vY = pVy;

  int vIb = 0;
  float vSumf = 0;

#if defined(__loongarch_asx)
  // Initialize accumulator with zeros
  __m256 vAcc = (__m256)__lasx_xvldi(0);

  // Main loop
  for (; vIb < cNb; ++vIb) {
    // Compute combined scale for the block
    const __m256 cD = __lasx_xvreplfr2vr_s(GGML_CPU_FP16_TO_FP32(vX[vIb].d) * GGML_CPU_FP16_TO_FP32(vY[vIb].d));
    __m256i vQx = __lasx_xvld((const __m256i *)vX[vIb].qs, 0);
    __m256i vQy = __lasx_xvld((const __m256i *)vY[vIb].qs, 0);

    const __m256 cQ = fMulSumI8PairsFloat(vQx, vQy);

    // Multiply q with scale and accumulate
    vAcc = __lasx_xvfmadd_s(cD, cQ, vAcc);
  }

  vSumf = fHsumFloat8(vAcc);

  *pS = vSumf;

#elif defined(__loongarch_sx)

  __m128 vAcc = (__m128)__lsx_vldi(0);

  for (; vIb < cNb; ++vIb) {
    const float cD = GGML_CPU_FP16_TO_FP32(vX[vIb].d) * GGML_CPU_FP16_TO_FP32(vY[vIb].d);
    const __m128i qx_0 = __lsx_vld((const __m128i *)vX[vIb].qs, 0);
    const __m128i qx_1 = __lsx_vld((const __m128i *)vX[vIb].qs + 1, 0);
    const __m128i qy_0 = __lsx_vld((const __m128i *)vY[vIb].qs, 0);
    const __m128i qy_1 = __lsx_vld((const __m128i *)vY[vIb].qs + 1, 0);

    const __m128i p16_0 = fLsxMaddubsH(qx_0, qy_0);
    const __m128i p16_1 = fLsxMaddubsH(qx_1, qy_1);

    // Sum int16 pairs → int32
    const __m128i s_0 = __lsx_vaddwev_w_h(p16_0, p16_1);
    const __m128i s_1 = __lsx_vaddwod_w_h(p16_0, p16_1);

    const __m128 cQ = __lsx_vffint_s_w(__lsx_vadd_w(s_0, s_1));
    vAcc = __lsx_vfmadd_s(__lsx_vreplfr2vr_s(cD), cQ, vAcc);
  }

  __m128 res = fLsxHaddS(vAcc, vAcc);
  res = fLsxHaddS(res, res);
  vSumf = ((v4f32)res)[0];

  *pS = vSumf;

#else
  UNUSED(cNb);
  UNUSED(vIb);
  UNUSED(vSumf);
  UNUSED(vX);
  UNUSED(vY);
  ggml_vec_dot_q8_0_q8_0_generic(pN, pS, pBs, pVx, pBx, pVy, pBy, pNrc);
#endif
}

void ggml_vec_dot_q2_K_q8_K(int pN, float *GGML_RESTRICT pS, size_t pBs, const void *GGML_RESTRICT pVx, size_t pBx,
                            const void *GGML_RESTRICT pVy, size_t pBy, int pNrc) {
  assert(pNrc == 1);
  UNUSED(pNrc);
  UNUSED(pBx);
  UNUSED(pBy);
  UNUSED(pBs);

  const block_q2_K *GGML_RESTRICT vX = pVx;
  const block_q8_K *GGML_RESTRICT vY = pVy;

  const int cNb = pN / QK_K;

#if defined __loongarch_asx

  __m256 vAcc = (__m256)__lasx_xvldi(0);

  for (int vI = 0; vI < cNb; ++vI) {

    const float cD = vY[vI].d * GGML_CPU_FP16_TO_FP32(vX[vI].d);
    const float cDmin = -vY[vI].d * GGML_CPU_FP16_TO_FP32(vX[vI].dmin);

    const uint8_t *GGML_RESTRICT vQ2 = vX[vI].qs;
    const int8_t *GGML_RESTRICT vQ8 = vY[vI].qs;

    const __m128i cMinsAndScales128 = __lsx_vld((const __m128i *)vX[vI].scales, 0);
    const __m128i cScales128 = __lsx_vandi_b(cMinsAndScales128, 0xf);
    const __m256i cMins = fLasxExt816(__lsx_vsrli_b(cMinsAndScales128, 4));
    const __m256i cProd = fLasxMaddH(cMins, __lasx_xvld((const __m256i *)vY[vI].bsums, 0));

    vAcc = __lasx_xvfmadd_s(__lasx_xvreplfr2vr_s(cDmin), __lasx_xvffint_s_w(cProd), vAcc);

    const v16i8 cShuffleMask = {0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15};
    const __m256i cScalesShuffled = fLasxExt816(__lsx_vshuf_b(cScales128, cScales128, (__m128i)cShuffleMask));

    __m256i vSumi = __lasx_xvldi(0);

    for (int vJ = 0; vJ < QK_K / 128; ++vJ) {

      const __m256i cQ2bits = __lasx_xvld((const __m256i *)vQ2, 0);
      vQ2 += 32;

      const __m256i cQ80 = __lasx_xvld((const __m256i *)vQ8, 0);
      vQ8 += 32;
      const __m256i cQ81 = __lasx_xvld((const __m256i *)vQ8, 0);
      vQ8 += 32;
      const __m256i cQ82 = __lasx_xvld((const __m256i *)vQ8, 0);
      vQ8 += 32;
      const __m256i cQ83 = __lasx_xvld((const __m256i *)vQ8, 0);
      vQ8 += 32;

      const __m256i cQ20 = __lasx_xvandi_b(cQ2bits, 3);
      const __m256i cQ21 = __lasx_xvandi_b(__lasx_xvsrli_b(cQ2bits, 2), 3);
      const __m256i cQ22 = __lasx_xvandi_b(__lasx_xvsrli_b(cQ2bits, 4), 3);
      const __m256i cQ23 = __lasx_xvsrli_b(cQ2bits, 6);

      __m256i vP0 = fLasxMaddHB(cQ20, cQ80);
      __m256i vP1 = fLasxMaddHB(cQ21, cQ81);
      __m256i vP2 = fLasxMaddHB(cQ22, cQ82);
      __m256i vP3 = fLasxMaddHB(cQ23, cQ83);

      vP0 = fLasxMaddH(fLasxXvrepl128veiH(cScalesShuffled, 4 * vJ + 0), vP0);
      vP1 = fLasxMaddH(fLasxXvrepl128veiH(cScalesShuffled, 4 * vJ + 1), vP1);
      vP2 = fLasxMaddH(fLasxXvrepl128veiH(cScalesShuffled, 4 * vJ + 2), vP2);
      vP3 = fLasxMaddH(fLasxXvrepl128veiH(cScalesShuffled, 4 * vJ + 3), vP3);

      vP0 = __lasx_xvadd_w(vP0, vP1);
      vP2 = __lasx_xvadd_w(vP2, vP3);

      vSumi = __lasx_xvadd_w(vSumi, __lasx_xvadd_w(vP0, vP2));
    }

    vAcc = __lasx_xvfmadd_s(__lasx_xvreplfr2vr_s(cD), __lasx_xvffint_s_w(vSumi), vAcc);
  }

  *pS = fHsumFloat8(vAcc);

#else
  UNUSED(vX);
  UNUSED(vY);
  UNUSED(cNb);
  ggml_vec_dot_q2_K_q8_K_generic(pN, pS, pBs, pVx, pBx, pVy, pBy, pNrc);
#endif
}

void ggml_vec_dot_q3_K_q8_K(int pN, float *GGML_RESTRICT pS, size_t pBs, const void *GGML_RESTRICT pVx, size_t pBx,
                            const void *GGML_RESTRICT pVy, size_t pBy, int pNrc) {
  assert(pN % QK_K == 0);
  assert(pNrc == 1);
  UNUSED(pNrc);
  UNUSED(pBx);
  UNUSED(pBy);
  UNUSED(pBs);

  const uint32_t cKmask1 = 0x03030303;
  const uint32_t cKmask2 = 0x0f0f0f0f;

  const block_q3_K *GGML_RESTRICT vX = pVx;
  const block_q8_K *GGML_RESTRICT vY = pVy;

  const int cNb = pN / QK_K;

#if defined __loongarch_asx

  const __m128i cM32 = __lsx_vreplgr2vr_b(32);

  __m256 vAcc = (__m256)__lasx_xvldi(0);

  uint32_t aAux[3];

  for (int vI = 0; vI < cNb; ++vI) {

    const float cD = vY[vI].d * GGML_CPU_FP16_TO_FP32(vX[vI].d);
    const uint8_t *GGML_RESTRICT vQ3 = vX[vI].qs;
    const int8_t *GGML_RESTRICT vQ8 = vY[vI].qs;
    // Set up scales
    memcpy(aAux, vX[vI].scales, 12);
    __m128i vScales128 = fLsxSetW(((aAux[1] >> 4) & cKmask2) | (((aAux[2] >> 6) & cKmask1) << 4),
                                  ((aAux[0] >> 4) & cKmask2) | (((aAux[2] >> 4) & cKmask1) << 4),
                                  (aAux[1] & cKmask2) | (((aAux[2] >> 2) & cKmask1) << 4),
                                  (aAux[0] & cKmask2) | (((aAux[2] >> 0) & cKmask1) << 4));
    vScales128 = __lsx_vsub_b(vScales128, cM32);

    const v16i8 cShuffleMask = {0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15};
    const __m256i cScalesShuffled = fLasxExt816(__lsx_vshuf_b(vScales128, vScales128, (__m128i)cShuffleMask));

    // high bit
    const __m256i cHbits = __lasx_xvld((const __m256i *)vX[vI].hmask, 0);

    // integer accumulator
    __m256i vSumi = __lasx_xvldi(0);

    for (int vJ = 0; vJ < QK_K / 128; ++vJ) {
      // load low 2 bits
      const __m256i cQ3bits = __lasx_xvld((const __m256i *)vQ3, 0);
      vQ3 += 32;

      // prepare low and high bits
      const __m256i cQ3l0 = __lasx_xvandi_b(cQ3bits, 3);
      const __m256i cQ3l1 = __lasx_xvandi_b(__lasx_xvsrli_b(cQ3bits, 2), 3);
      const __m256i cQ3l2 = __lasx_xvandi_b(__lasx_xvsrli_b(cQ3bits, 4), 3);
      const __m256i cQ3l3 = __lasx_xvsrli_b(cQ3bits, 6);
      const __m256i cQ3h0 = __lasx_xvslli_b(__lasx_xvseqi_b(fLasxXvandiBBit(cHbits, 4 * vJ + 0), 0), 2);
      const __m256i cQ3h1 = __lasx_xvslli_b(__lasx_xvseqi_b(fLasxXvandiBBit(cHbits, 4 * vJ + 1), 0), 2);
      const __m256i cQ3h2 = __lasx_xvslli_b(__lasx_xvseqi_b(fLasxXvandiBBit(cHbits, 4 * vJ + 2), 0), 2);
      const __m256i cQ3h3 = __lasx_xvslli_b(__lasx_xvseqi_b(fLasxXvandiBBit(cHbits, 4 * vJ + 3), 0), 2);
      const __m256i cQ30 = __lasx_xvor_v(cQ3h0, cQ3l0);
      const __m256i cQ31 = __lasx_xvor_v(cQ3h1, cQ3l1);
      const __m256i cQ32 = __lasx_xvor_v(cQ3h2, cQ3l2);
      const __m256i cQ33 = __lasx_xvor_v(cQ3h3, cQ3l3);

      // load Q8 quants
      const __m256i cQ80 = __lasx_xvld((const __m256i *)vQ8, 0);
      vQ8 += 32;
      const __m256i cQ81 = __lasx_xvld((const __m256i *)vQ8, 0);
      vQ8 += 32;
      const __m256i cQ82 = __lasx_xvld((const __m256i *)vQ8, 0);
      vQ8 += 32;
      const __m256i cQ83 = __lasx_xvld((const __m256i *)vQ8, 0);
      vQ8 += 32;

      __m256i vP160 = fLasxMaddHB(cQ80, cQ30);
      __m256i vP161 = fLasxMaddHB(cQ81, cQ31);
      __m256i vP162 = fLasxMaddHB(cQ82, cQ32);
      __m256i vP163 = fLasxMaddHB(cQ83, cQ33);

      // multiply with scales
      vP160 = fLasxMaddH(fLasxXvrepl128veiH(cScalesShuffled, 4 * vJ + 0), vP160);
      vP161 = fLasxMaddH(fLasxXvrepl128veiH(cScalesShuffled, 4 * vJ + 1), vP161);
      vP162 = fLasxMaddH(fLasxXvrepl128veiH(cScalesShuffled, 4 * vJ + 2), vP162);
      vP163 = fLasxMaddH(fLasxXvrepl128veiH(cScalesShuffled, 4 * vJ + 3), vP163);

      // accumulate
      vP160 = __lasx_xvadd_w(vP160, vP161);
      vP162 = __lasx_xvadd_w(vP162, vP163);
      vSumi = __lasx_xvadd_w(vSumi, __lasx_xvadd_w(vP160, vP162));
    }
    // multiply with block scale and accumulate
    vAcc = __lasx_xvfmadd_s(__lasx_xvreplfr2vr_s(cD), __lasx_xvffint_s_w(vSumi), vAcc);
  }

  *pS = fHsumFloat8(vAcc);

#else
  UNUSED(cKmask1);
  UNUSED(cKmask2);
  UNUSED(vX);
  UNUSED(vY);
  UNUSED(cNb);
  ggml_vec_dot_q3_K_q8_K_generic(pN, pS, pBs, pVx, pBx, pVy, pBy, pNrc);
#endif
}

void ggml_vec_dot_q4_K_q8_K(int pN, float *GGML_RESTRICT pS, size_t pBs, const void *GGML_RESTRICT pVx, size_t pBx,
                            const void *GGML_RESTRICT pVy, size_t pBy, int pNrc) {
  assert(pN % QK_K == 0);
  assert(pNrc == 1);
  UNUSED(pNrc);
  UNUSED(pBx);
  UNUSED(pBy);
  UNUSED(pBs);

  const block_q4_K *GGML_RESTRICT vX = pVx;
  const block_q8_K *GGML_RESTRICT vY = pVy;

  const int cNb = pN / QK_K;

  static const uint32_t cKmask1 = 0x3f3f3f3f;
  static const uint32_t cKmask2 = 0x0f0f0f0f;
  static const uint32_t cKmask3 = 0x03030303;

  uint32_t vUtmp[4];

#if defined __loongarch_asx

  __m256 vAcc = (__m256)__lasx_xvldi(0);
  __m128 vAccM = (__m128)__lsx_vldi(0);

  for (int vI = 0; vI < cNb; ++vI) {

    const float cD = vY[vI].d * GGML_CPU_FP16_TO_FP32(vX[vI].d);
    const float cDmin = -vY[vI].d * GGML_CPU_FP16_TO_FP32(vX[vI].dmin);

    memcpy(vUtmp, vX[vI].scales, 12);
    vUtmp[3] = ((vUtmp[2] >> 4) & cKmask2) | (((vUtmp[1] >> 6) & cKmask3) << 4);
    const uint32_t cUaux = vUtmp[1] & cKmask1;
    vUtmp[1] = (vUtmp[2] & cKmask2) | (((vUtmp[0] >> 6) & cKmask3) << 4);
    vUtmp[2] = cUaux;
    vUtmp[0] &= cKmask1;

    const uint8_t *GGML_RESTRICT vQ4 = vX[vI].qs;
    const int8_t *GGML_RESTRICT vQ8 = vY[vI].qs;

    const __m128i cMinsAndScales128 = fLsxSetW(vUtmp[3], vUtmp[2], vUtmp[1], vUtmp[0]);
    const __m128i cMins128 = __lsx_vexth_h_b(cMinsAndScales128);
    const __m128i cScales128 = __lsx_vsllwil_h_b(cMinsAndScales128, 0);

    const __m256i cQ8sums = __lasx_xvld((const __m256i *)vY[vI].bsums, 0);
    const __m128i cQ8s = fLsxHaddH(fLasxExtracti128(cQ8sums, 0), fLasxExtracti128(cQ8sums, 1));
    const __m128i cProd = fLsxMaddH(cMins128, cQ8s);
    vAccM = __lsx_vfmadd_s(__lsx_vreplfr2vr_s(cDmin), __lsx_vffint_s_w(cProd), vAccM);

    const __m256i cScales = fLasxInsertf128(cScales128, cScales128);

    __m256i vSumi = __lasx_xvldi(0);

    for (int vJ = 0; vJ < QK_K / 64; ++vJ) {

      const __m256i cScaleL = fLasxXvrepl128veiH(cScales, 2 * vJ + 0);
      const __m256i cScaleH = fLasxXvrepl128veiH(cScales, 2 * vJ + 1);

      const __m256i cQ4bits = __lasx_xvld((const __m256i *)vQ4, 0);
      vQ4 += 32;
      const __m256i cQ4l = __lasx_xvandi_b(cQ4bits, 0xf);
      const __m256i cQ4h = __lasx_xvsrli_b(cQ4bits, 4);

      const __m256i cQ8l = __lasx_xvld((const __m256i *)vQ8, 0);
      vQ8 += 32;
      __m256i vP16l = fLasxMaddHB(cQ4l, cQ8l);
      vP16l = fLasxMaddH(cScaleL, vP16l);

      const __m256i cQ8h = __lasx_xvld((const __m256i *)vQ8, 0);
      vQ8 += 32;
      __m256i vP16h = fLasxMaddHB(cQ4h, cQ8h);
      vP16h = fLasxMaddH(cScaleH, vP16h);
      const __m256i cSumj = __lasx_xvadd_w(vP16l, vP16h);

      vSumi = __lasx_xvadd_w(vSumi, cSumj);
    }

    __m256 vVd = __lasx_xvreplfr2vr_s(cD);
    vAcc = __lasx_xvfmadd_s(vVd, __lasx_xvffint_s_w(vSumi), vAcc);
  }

  vAccM = __lsx_vfadd_s(vAccM, (__m128)__lsx_vpermi_w((__m128i)vAccM, (__m128i)vAccM, 0xee));
  __m128i vTmp1 = __lsx_vinsgr2vr_w(__lsx_vldi(0), __lsx_vpickve2gr_w((__m128i)vAccM, 1), 0);
  vAccM = __lsx_vfadd_s(vAccM, (__m128)vTmp1);

  *pS = fHsumFloat8(vAcc) + ((v4f32)vAccM)[0];

#else
  UNUSED(vX);
  UNUSED(vY);
  UNUSED(cNb);
  UNUSED(cKmask1);
  UNUSED(cKmask2);
  UNUSED(cKmask3);
  UNUSED(vUtmp);
  ggml_vec_dot_q4_K_q8_K_generic(pN, pS, pBs, pVx, pBx, pVy, pBy, pNrc);
#endif
}

void ggml_vec_dot_q5_K_q8_K(int pN, float *GGML_RESTRICT pS, size_t pBs, const void *GGML_RESTRICT pVx, size_t pBx,
                            const void *GGML_RESTRICT pVy, size_t pBy, int pNrc) {
  assert(pN % QK_K == 0);
  assert(pNrc == 1);
  UNUSED(pNrc);
  UNUSED(pBx);
  UNUSED(pBy);
  UNUSED(pBs);

  const block_q5_K *GGML_RESTRICT vX = pVx;
  const block_q8_K *GGML_RESTRICT vY = pVy;

  const int cNb = pN / QK_K;

  static const uint32_t cKmask1 = 0x3f3f3f3f;
  static const uint32_t cKmask2 = 0x0f0f0f0f;
  static const uint32_t cKmask3 = 0x03030303;

  uint32_t vUtmp[4];

#if defined __loongarch_asx

  __m256 vAcc = (__m256)__lasx_xvldi(0);
  __m128 vAccM = (__m128)__lsx_vldi(0);

  for (int vI = 0; vI < cNb; ++vI) {

    const uint8_t *GGML_RESTRICT vQ5 = vX[vI].qs;
    const int8_t *GGML_RESTRICT vQ8 = vY[vI].qs;

    const float cD = vY[vI].d * GGML_CPU_FP16_TO_FP32(vX[vI].d);
    const float cDmin = -vY[vI].d * GGML_CPU_FP16_TO_FP32(vX[vI].dmin);

    memcpy(vUtmp, vX[vI].scales, 12);
    vUtmp[3] = ((vUtmp[2] >> 4) & cKmask2) | (((vUtmp[1] >> 6) & cKmask3) << 4);
    const uint32_t cUaux = vUtmp[1] & cKmask1;
    vUtmp[1] = (vUtmp[2] & cKmask2) | (((vUtmp[0] >> 6) & cKmask3) << 4);
    vUtmp[2] = cUaux;
    vUtmp[0] &= cKmask1;

    const __m128i cMinsAndScales128 = fLsxSetW(vUtmp[3], vUtmp[2], vUtmp[1], vUtmp[0]);
    const __m128i cMins128 = __lsx_vexth_h_b(cMinsAndScales128);
    const __m128i cScales128 = __lsx_vsllwil_h_b(cMinsAndScales128, 0);

    const __m256i cQ8sums = __lasx_xvld((const __m256i *)vY[vI].bsums, 0);
    const __m128i cQ8s = fLsxHaddH(fLasxExtracti128(cQ8sums, 0), fLasxExtracti128(cQ8sums, 1));
    const __m128i cProd = fLsxMaddH(cMins128, cQ8s);
    vAccM = __lsx_vfmadd_s(__lsx_vreplfr2vr_s(cDmin), __lsx_vffint_s_w(cProd), vAccM);

    const __m256i cScales = fLasxInsertf128(cScales128, cScales128);

    const __m256i cHbits = __lasx_xvld((const __m256i *)vX[vI].qh, 0);

    __m256i vSumi = __lasx_xvldi(0);

    for (int vJ = 0; vJ < QK_K / 64; ++vJ) {

      const __m256i cScale0 = fLasxXvrepl128veiH(cScales, 2 * vJ + 0);
      const __m256i cScale1 = fLasxXvrepl128veiH(cScales, 2 * vJ + 1);

      const __m256i cQ5bits = __lasx_xvld((const __m256i *)vQ5, 0);
      vQ5 += 32;

      const __m256i cQ5l0 = __lasx_xvandi_b(cQ5bits, 0xf);
      const __m256i cQ5l1 = __lasx_xvsrli_b(cQ5bits, 4);
      const __m256i cQ5h0 = __lasx_xvnori_b(__lasx_xvseqi_b(fLasxXvandiBBit(cHbits, 2 * vJ + 0), 0), 0xef);
      const __m256i cQ5h1 = __lasx_xvnori_b(__lasx_xvseqi_b(fLasxXvandiBBit(cHbits, 2 * vJ + 1), 0), 0xef);
      const __m256i cQ50 = __lasx_xvor_v(cQ5l0, cQ5h0);
      const __m256i cQ51 = __lasx_xvor_v(cQ5l1, cQ5h1);

      const __m256i cQ80 = __lasx_xvld((const __m256i *)vQ8, 0);
      vQ8 += 32;
      const __m256i cQ81 = __lasx_xvld((const __m256i *)vQ8, 0);
      vQ8 += 32;

      __m256i vP160 = fLasxMaddHB(cQ50, cQ80);
      __m256i vP161 = fLasxMaddHB(cQ51, cQ81);

      vP160 = fLasxMaddH(cScale0, vP160);
      vP161 = fLasxMaddH(cScale1, vP161);

      vSumi = __lasx_xvadd_w(vSumi, __lasx_xvadd_w(vP160, vP161));
    }

    __m256 vVd = __lasx_xvreplfr2vr_s(cD);
    vAcc = __lasx_xvfmadd_s(vVd, __lasx_xvffint_s_w(vSumi), vAcc);
  }

  vAccM = __lsx_vfadd_s(vAccM, (__m128)__lsx_vbsrl_v(vAccM, 8));
  vAccM = __lsx_vfadd_s(vAccM, (__m128)__lsx_vbsrl_v(vAccM, 4));

  *pS = fHsumFloat8(vAcc) + ((v4f32)vAccM)[0];

#else
  UNUSED(vX);
  UNUSED(vY);
  UNUSED(cNb);
  UNUSED(cKmask1);
  UNUSED(cKmask2);
  UNUSED(cKmask3);
  UNUSED(vUtmp);
  ggml_vec_dot_q5_K_q8_K_generic(pN, pS, pBs, pVx, pBx, pVy, pBy, pNrc);
#endif
}

void ggml_vec_dot_q6_K_q8_K(int pN, float *GGML_RESTRICT pS, size_t pBs, const void *GGML_RESTRICT pVx, size_t pBx,
                            const void *GGML_RESTRICT pVy, size_t pBy, int pNrc) {
  assert(pN % QK_K == 0);
  assert(pNrc == 1);
  UNUSED(pNrc);
  UNUSED(pBx);
  UNUSED(pBy);
  UNUSED(pBs);

  const block_q6_K *GGML_RESTRICT vX = pVx;
  const block_q8_K *GGML_RESTRICT vY = pVy;

  const int cNb = pN / QK_K;

#if defined __loongarch_asx

  const __m256i cM32s = __lasx_xvreplgr2vr_b(32);

  __m256 vAcc = (__m256)__lasx_xvldi(0);

  for (int vI = 0; vI < cNb; ++vI) {

    const float cD = vY[vI].d * GGML_CPU_FP16_TO_FP32(vX[vI].d);

    const uint8_t *GGML_RESTRICT vQ4 = vX[vI].ql;
    const uint8_t *GGML_RESTRICT vQh = vX[vI].qh;
    const int8_t *GGML_RESTRICT vQ8 = vY[vI].qs;

    const __m128i cScales128 = __lsx_vld((const __m128i *)vX[vI].scales, 0);
    const v16i8 cShuffleMask = {0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15};
    const __m256i cScalesShuffled = fLasxExt816(__lsx_vshuf_b(cScales128, cScales128, (__m128i)cShuffleMask));

    __m256i vSumi = __lasx_xvldi(0);

    for (int vJ = 0; vJ < QK_K / 128; ++vJ) {

      const __m256i cQ4bits1 = __lasx_xvld((const __m256i *)vQ4, 0);
      vQ4 += 32;
      const __m256i cQ4bits2 = __lasx_xvld((const __m256i *)vQ4, 0);
      vQ4 += 32;
      const __m256i cQ4bitsH = __lasx_xvld((const __m256i *)vQh, 0);
      vQh += 32;

      const __m256i cQ4h0 = __lasx_xvslli_b(__lasx_xvandi_b(cQ4bitsH, 3), 4);
      const __m256i cQ4h1 = __lasx_xvslli_b(__lasx_xvandi_b(cQ4bitsH, 3 << 2), 2);
      const __m256i cQ4h2 = __lasx_xvandi_b(cQ4bitsH, 3 << 4);
      const __m256i cQ4h3 = __lasx_xvsrli_b(__lasx_xvandi_b(cQ4bitsH, 3 << 6), 2);

      const __m256i cQ40 = __lasx_xvor_v(__lasx_xvandi_b(cQ4bits1, 0xf), cQ4h0);
      const __m256i cQ41 = __lasx_xvor_v(__lasx_xvandi_b(cQ4bits2, 0xf), cQ4h1);
      const __m256i cQ42 = __lasx_xvor_v(__lasx_xvsrli_b(cQ4bits1, 4), cQ4h2);
      const __m256i cQ43 = __lasx_xvor_v(__lasx_xvsrli_b(cQ4bits2, 4), cQ4h3);

      const __m256i cQ80 = __lasx_xvld((const __m256i *)vQ8, 0);
      vQ8 += 32;
      const __m256i cQ81 = __lasx_xvld((const __m256i *)vQ8, 0);
      vQ8 += 32;
      const __m256i cQ82 = __lasx_xvld((const __m256i *)vQ8, 0);
      vQ8 += 32;
      const __m256i cQ83 = __lasx_xvld((const __m256i *)vQ8, 0);
      vQ8 += 32;

      __m256i vP160 = fLasxMaddHB(__lasx_xvsub_b(cQ40, cM32s), cQ80);
      __m256i vP161 = fLasxMaddHB(__lasx_xvsub_b(cQ41, cM32s), cQ81);
      __m256i vP162 = fLasxMaddHB(__lasx_xvsub_b(cQ42, cM32s), cQ82);
      __m256i vP163 = fLasxMaddHB(__lasx_xvsub_b(cQ43, cM32s), cQ83);

      vP160 = fLasxMaddH(fLasxXvrepl128veiH(cScalesShuffled, 4 * vJ + 0), vP160);
      vP161 = fLasxMaddH(fLasxXvrepl128veiH(cScalesShuffled, 4 * vJ + 1), vP161);
      vP162 = fLasxMaddH(fLasxXvrepl128veiH(cScalesShuffled, 4 * vJ + 2), vP162);
      vP163 = fLasxMaddH(fLasxXvrepl128veiH(cScalesShuffled, 4 * vJ + 3), vP163);

      vSumi = __lasx_xvadd_w(vSumi, __lasx_xvadd_w(vP160, vP161));
      vSumi = __lasx_xvadd_w(vSumi, __lasx_xvadd_w(vP162, vP163));
    }

    vAcc = __lasx_xvfmadd_s((__m256)__lasx_xvreplfr2vr_s(cD), __lasx_xvffint_s_w(vSumi), vAcc);
  }

  *pS = fHsumFloat8(vAcc);

#elif defined(__loongarch_sx)

  const __m128i cM32s = __lsx_vreplgr2vr_b(32);

  __m128 acc_0 = (__m128)__lsx_vldi(0);
  __m128 acc_1 = (__m128)__lsx_vldi(0);

  for (int vI = 0; vI < cNb; ++vI) {

    const float cD = vY[vI].d * GGML_CPU_FP16_TO_FP32(vX[vI].d);

    const uint8_t *GGML_RESTRICT vQ4 = vX[vI].ql;
    const uint8_t *GGML_RESTRICT vQh = vX[vI].qh;
    const int8_t *GGML_RESTRICT vQ8 = vY[vI].qs;

    const __m128i scale_i8 = __lsx_vld(vX[vI].scales, 0);
    const __m128i scales_lo = __lsx_vsllwil_h_b(scale_i8, 0);
    const __m128i scales_hi = __lsx_vsllwil_h_b(__lsx_vbsrl_v(scale_i8, 8), 0);

    __m128i sumi_0 = __lsx_vldi(0);
    __m128i sumi_1 = __lsx_vldi(0);

    for (int vJ = 0; vJ < QK_K / 128; ++vJ) {

      const __m128i q4bitsH_0 = __lsx_vld((const __m128i *)vQh, 0);
      vQh += 16;
      const __m128i q4bitsH_1 = __lsx_vld((const __m128i *)vQh, 0);
      vQh += 16;

      const __m128i cQ4h0 = __lsx_vslli_b(__lsx_vandi_b(q4bitsH_0, 3), 4);
      const __m128i cQ4h1 = __lsx_vslli_b(__lsx_vandi_b(q4bitsH_1, 3), 4);
      const __m128i cQ4h2 = __lsx_vslli_b(__lsx_vandi_b(q4bitsH_0, 3 << 2), 2);
      const __m128i cQ4h3 = __lsx_vslli_b(__lsx_vandi_b(q4bitsH_1, 3 << 2), 2);
      const __m128i q4h_4 = __lsx_vandi_b(q4bitsH_0, 3 << 4);
      const __m128i q4h_5 = __lsx_vandi_b(q4bitsH_1, 3 << 4);
      const __m128i q4h_6 = __lsx_vsrli_b(__lsx_vandi_b(q4bitsH_0, 3 << 6), 2);
      const __m128i q4h_7 = __lsx_vsrli_b(__lsx_vandi_b(q4bitsH_1, 3 << 6), 2);

      const __m128i q4bits1_0 = __lsx_vld((const __m128i *)vQ4, 0);
      vQ4 += 16;
      const __m128i q4bits1_1 = __lsx_vld((const __m128i *)vQ4, 0);
      vQ4 += 16;
      const __m128i q4bits2_0 = __lsx_vld((const __m128i *)vQ4, 0);
      vQ4 += 16;
      const __m128i q4bits2_1 = __lsx_vld((const __m128i *)vQ4, 0);
      vQ4 += 16;

      const __m128i cQ40 = __lsx_vor_v(__lsx_vandi_b(q4bits1_0, 0xf), cQ4h0);
      const __m128i cQ41 = __lsx_vor_v(__lsx_vandi_b(q4bits1_1, 0xf), cQ4h1);
      const __m128i cQ42 = __lsx_vor_v(__lsx_vandi_b(q4bits2_0, 0xf), cQ4h2);
      const __m128i cQ43 = __lsx_vor_v(__lsx_vandi_b(q4bits2_1, 0xf), cQ4h3);
      const __m128i q4_4 = __lsx_vor_v(__lsx_vsrli_b(q4bits1_0, 4), q4h_4);
      const __m128i q4_5 = __lsx_vor_v(__lsx_vsrli_b(q4bits1_1, 4), q4h_5);
      const __m128i q4_6 = __lsx_vor_v(__lsx_vsrli_b(q4bits2_0, 4), q4h_6);
      const __m128i q4_7 = __lsx_vor_v(__lsx_vsrli_b(q4bits2_1, 4), q4h_7);

      const __m128i cQ80 = __lsx_vld((const __m128i *)vQ8, 0);
      vQ8 += 16;
      const __m128i cQ81 = __lsx_vld((const __m128i *)vQ8, 0);
      vQ8 += 16;
      const __m128i cQ82 = __lsx_vld((const __m128i *)vQ8, 0);
      vQ8 += 16;
      const __m128i cQ83 = __lsx_vld((const __m128i *)vQ8, 0);
      vQ8 += 16;
      const __m128i q8_4 = __lsx_vld((const __m128i *)vQ8, 0);
      vQ8 += 16;
      const __m128i q8_5 = __lsx_vld((const __m128i *)vQ8, 0);
      vQ8 += 16;
      const __m128i q8_6 = __lsx_vld((const __m128i *)vQ8, 0);
      vQ8 += 16;
      const __m128i q8_7 = __lsx_vld((const __m128i *)vQ8, 0);
      vQ8 += 16;

      __m128i vP160 = fLsxMaddubsH(__lsx_vsub_b(cQ40, cM32s), cQ80);
      __m128i vP161 = fLsxMaddubsH(__lsx_vsub_b(cQ41, cM32s), cQ81);
      __m128i vP162 = fLsxMaddubsH(__lsx_vsub_b(cQ42, cM32s), cQ82);
      __m128i vP163 = fLsxMaddubsH(__lsx_vsub_b(cQ43, cM32s), cQ83);
      __m128i p16_4 = fLsxMaddubsH(__lsx_vsub_b(q4_4, cM32s), q8_4);
      __m128i p16_5 = fLsxMaddubsH(__lsx_vsub_b(q4_5, cM32s), q8_5);
      __m128i p16_6 = fLsxMaddubsH(__lsx_vsub_b(q4_6, cM32s), q8_6);
      __m128i p16_7 = fLsxMaddubsH(__lsx_vsub_b(q4_7, cM32s), q8_7);

      const __m128i sc_vec = vJ == 0 ? scales_lo : scales_hi;

      vP160 = fLsxMaddH(__lsx_vreplvei_h(sc_vec, 0), vP160);
      vP161 = fLsxMaddH(__lsx_vreplvei_h(sc_vec, 1), vP161);
      vP162 = fLsxMaddH(__lsx_vreplvei_h(sc_vec, 2), vP162);
      vP163 = fLsxMaddH(__lsx_vreplvei_h(sc_vec, 3), vP163);
      p16_4 = fLsxMaddH(__lsx_vreplvei_h(sc_vec, 4), p16_4);
      p16_5 = fLsxMaddH(__lsx_vreplvei_h(sc_vec, 5), p16_5);
      p16_6 = fLsxMaddH(__lsx_vreplvei_h(sc_vec, 6), p16_6);
      p16_7 = fLsxMaddH(__lsx_vreplvei_h(sc_vec, 7), p16_7);

      sumi_0 = __lsx_vadd_w(sumi_0, __lsx_vadd_w(vP160, vP162));
      sumi_1 = __lsx_vadd_w(sumi_1, __lsx_vadd_w(vP161, vP163));
      sumi_0 = __lsx_vadd_w(sumi_0, __lsx_vadd_w(p16_4, p16_6));
      sumi_1 = __lsx_vadd_w(sumi_1, __lsx_vadd_w(p16_5, p16_7));
    }

    __m128 p_0 = __lsx_vfmul_s(__lsx_vreplfr2vr_s(cD), __lsx_vffint_s_w(sumi_0));
    __m128 p_1 = __lsx_vfmul_s(__lsx_vreplfr2vr_s(cD), __lsx_vffint_s_w(sumi_1));
    acc_0 = __lsx_vfadd_s(p_0, acc_0);
    acc_1 = __lsx_vfadd_s(p_1, acc_1);
  }

  *pS = fHsumFloat4x4(acc_0, acc_1, (__m128)__lsx_vldi(0), (__m128)__lsx_vldi(0));

#else
  UNUSED(vX);
  UNUSED(vY);
  UNUSED(cNb);
  ggml_vec_dot_q6_K_q8_K_generic(pN, pS, pBs, pVx, pBx, pVy, pBy, pNrc);
#endif
}

#if defined(__loongarch_asx)
static const int8_t keven_signs_q2xs[1024] = {
    1,  1,  1,  1,  1,  1,  1,  1,  -1, 1,  1,  1,  1,  1,  1,  -1, 1,  -1, 1,  1,  1,  1,  1,  -1, -1, -1, 1,  1,  1,
    1,  1,  1,  1,  1,  -1, 1,  1,  1,  1,  -1, -1, 1,  -1, 1,  1,  1,  1,  1,  1,  -1, -1, 1,  1,  1,  1,  1,  -1, -1,
    -1, 1,  1,  1,  1,  -1, 1,  1,  1,  -1, 1,  1,  1,  -1, -1, 1,  1,  -1, 1,  1,  1,  1,  1,  -1, 1,  -1, 1,  1,  1,
    1,  -1, -1, 1,  -1, 1,  1,  1,  -1, 1,  1,  -1, -1, 1,  1,  1,  1,  -1, 1,  -1, -1, 1,  1,  1,  -1, 1,  -1, -1, -1,
    1,  1,  1,  -1, -1, -1, -1, -1, 1,  1,  1,  1,  1,  1,  1,  1,  -1, 1,  1,  -1, -1, 1,  1,  1,  -1, 1,  1,  1,  1,
    -1, 1,  1,  -1, 1,  1,  1,  -1, -1, 1,  1,  -1, 1,  1,  -1, 1,  1,  -1, 1,  -1, 1,  1,  1,  -1, 1,  -1, 1,  -1, 1,
    1,  -1, 1,  -1, -1, 1,  -1, 1,  1,  -1, -1, -1, -1, 1,  -1, 1,  1,  1,  1,  1,  1,  -1, -1, 1,  1,  1,  -1, 1,  1,
    -1, -1, 1,  1,  -1, 1,  -1, 1,  -1, -1, 1,  1,  -1, -1, -1, 1,  -1, -1, 1,  1,  1,  1,  1,  -1, -1, -1, 1,  1,  -1,
    -1, 1,  -1, -1, -1, 1,  1,  1,  1,  -1, -1, -1, -1, 1,  1,  1,  -1, -1, -1, -1, -1, 1,  1,  -1, 1,  1,  1,  1,  1,
    -1, 1,  -1, -1, 1,  1,  1,  1,  -1, 1,  1,  1,  -1, 1,  1,  1,  -1, 1,  1,  -1, -1, 1,  1,  1,  -1, 1,  -1, 1,  1,
    -1, 1,  1,  -1, 1,  1,  -1, 1,  -1, 1,  1,  -1, 1,  -1, 1,  -1, -1, 1,  1,  -1, 1,  -1, -1, -1, -1, 1,  1,  -1, 1,
    1,  1,  1,  1,  -1, 1,  -1, 1,  1,  -1, 1,  1,  -1, 1,  -1, 1,  -1, 1,  -1, 1,  -1, 1,  -1, 1,  -1, -1, -1, 1,  -1,
    1,  -1, 1,  1,  1,  1,  -1, -1, 1,  -1, 1,  -1, -1, 1,  -1, -1, 1,  -1, 1,  1,  1,  -1, -1, -1, 1,  -1, 1,  1,  -1,
    -1, -1, -1, 1,  -1, 1,  -1, 1,  1,  1,  1,  -1, -1, 1,  1,  -1, 1,  1,  1,  -1, -1, 1,  -1, 1,  -1, 1,  1,  -1, -1,
    1,  -1, -1, -1, 1,  1,  -1, -1, 1,  1,  1,  1,  -1, 1,  -1, -1, 1,  -1, -1, 1,  -1, 1,  -1, -1, 1,  1,  1,  -1, -1,
    1,  -1, -1, 1,  1,  -1, -1, -1, 1,  -1, -1, 1,  -1, 1,  1,  1,  -1, -1, -1, 1,  -1, -1, 1,  1,  -1, -1, -1, 1,  1,
    1,  -1, 1,  -1, -1, -1, 1,  1,  -1, -1, 1,  -1, -1, -1, 1,  -1, 1,  1,  -1, -1, -1, -1, 1,  1,  -1, 1,  -1, -1, -1,
    -1, 1,  -1, 1,  -1, -1, -1, -1, -1, 1,  -1, -1, -1, -1, -1, -1, -1, 1,  1,  1,  1,  1,  1,  1,  1,  -1, -1, -1, 1,
    1,  1,  1,  1,  -1, 1,  1,  -1, 1,  1,  1,  1,  -1, 1,  -1, -1, 1,  1,  1,  1,  -1, -1, 1,  1,  -1, 1,  1,  1,  -1,
    1,  -1, 1,  -1, 1,  1,  1,  -1, -1, 1,  -1, -1, 1,  1,  1,  -1, -1, -1, -1, -1, 1,  1,  1,  -1, 1,  1,  1,  1,  -1,
    1,  1,  -1, 1,  -1, 1,  1,  -1, 1,  1,  -1, -1, 1,  -1, 1,  -1, 1,  1,  -1, -1, -1, -1, 1,  -1, 1,  1,  -1, 1,  1,
    1,  -1, -1, 1,  1,  -1, -1, -1, 1,  -1, -1, 1,  1,  -1, 1,  1,  -1, -1, -1, 1,  1,  -1, 1,  -1, -1, -1, -1, 1,  1,
    -1, -1, 1,  1,  1,  1,  -1, 1,  -1, 1,  -1, 1,  1,  1,  -1, 1,  -1, -1, 1,  -1, 1,  1,  -1, 1,  -1, -1, -1, -1, 1,
    1,  -1, 1,  -1, 1,  1,  1,  -1, 1,  -1, 1,  -1, -1, -1, 1,  -1, 1,  -1, 1,  -1, 1,  1,  -1, -1, 1,  -1, 1,  -1, 1,
    -1, -1, -1, 1,  -1, 1,  -1, -1, 1,  1,  1,  -1, -1, 1,  -1, -1, -1, 1,  1,  -1, -1, 1,  -1, 1,  1,  -1, 1,  -1, -1,
    1,  -1, 1,  -1, -1, 1,  -1, -1, 1,  -1, -1, 1,  1,  -1, -1, -1, 1,  -1, 1,  -1, 1,  -1, -1, -1, 1,  -1, -1, 1,  -1,
    -1, -1, -1, 1,  -1, -1, -1, -1, -1, -1, -1, 1,  -1, 1,  1,  1,  1,  1,  1,  -1, -1, 1,  -1, 1,  1,  1,  1,  -1, -1,
    -1, 1,  -1, 1,  1,  1,  -1, -1, -1, -1, -1, 1,  1,  1,  -1, -1, 1,  1,  1,  -1, 1,  1,  -1, -1, -1, -1, 1,  -1, 1,
    1,  -1, -1, 1,  1,  -1, -1, 1,  1,  -1, -1, 1,  -1, -1, -1, 1,  1,  -1, -1, -1, 1,  1,  1,  -1, 1,  -1, -1, -1, -1,
    1,  1,  -1, 1,  -1, -1, 1,  1,  -1, 1,  -1, 1,  -1, -1, 1,  -1, -1, 1,  -1, 1,  -1, -1, -1, 1,  1,  -1, -1, 1,  -1,
    -1, 1,  -1, 1,  -1, -1, 1,  -1, -1, -1, 1,  -1, -1, -1, 1,  -1, -1, -1, -1, -1, -1, -1, 1,  -1, -1, 1,  1,  1,  1,
    1,  -1, -1, -1, -1, -1, 1,  1,  1,  -1, -1, -1, 1,  1,  -1, 1,  1,  -1, -1, -1, 1,  -1, -1, 1,  1,  -1, -1, -1, -1,
    1,  1,  -1, 1,  -1, -1, -1, 1,  -1, 1,  -1, 1,  -1, -1, -1, -1, 1,  -1, -1, 1,  -1, -1, -1, -1, -1, -1, -1, 1,  -1,
    -1, -1, 1,  1,  1,  1,  -1, -1, -1, -1, 1,  -1, 1,  1,  -1, -1, -1, -1, -1, 1,  -1, 1,  -1, -1, -1, -1, -1, -1, -1,
    1,  -1, -1, -1, -1, 1,  1,  1,  -1, -1, -1, -1, -1, -1, -1, 1,  -1, -1, -1, -1, -1, 1,  1,  -1, -1, -1, -1, -1, -1,
    1,  -1, -1, -1, -1, -1, -1, -1, -1,
};
#endif

void ggml_vec_dot_iq2_xxs_q8_K(int pN, float *GGML_RESTRICT pS, size_t pBs, const void *GGML_RESTRICT pVx, size_t pBx,
                               const void *GGML_RESTRICT pVy, size_t pBy, int pNrc) {
  assert(pN % QK_K == 0);
  assert(pNrc == 1);
  UNUSED(pNrc);
  UNUSED(pBx);
  UNUSED(pBy);
  UNUSED(pBs);

  const block_iq2_xxs *GGML_RESTRICT vX = pVx;
  const block_q8_K *GGML_RESTRICT vY = pVy;

  const int cNb = pN / QK_K;

#if defined(__loongarch_asx)

  const uint64_t *vSigns64 = (const uint64_t *)keven_signs_q2xs;

  uint32_t aAux32[4];
  const uint8_t *vAux8 = (const uint8_t *)aAux32;

  __m256 vAccumf = (__m256)__lasx_xvldi(0);
  for (int vI = 0; vI < cNb; ++vI) {
    const float cD = GGML_CPU_FP16_TO_FP32(vX[vI].d) * vY[vI].d;
    const uint16_t *GGML_RESTRICT vQ2 = vX[vI].qs;
    const int8_t *GGML_RESTRICT vQ8 = vY[vI].qs;
    __m256i vSumi1 = __lasx_xvldi(0);
    __m256i vSumi2 = __lasx_xvldi(0);
    for (int vIb32 = 0; vIb32 < QK_K / 32; vIb32 += 2) {
      const __m256i cQ81 = __lasx_xvld((const __m256i *)vQ8, 0);
      vQ8 += 32;
      const __m256i cQ82 = __lasx_xvld((const __m256i *)vQ8, 0);
      vQ8 += 32;
      memcpy(aAux32, vQ2, 4 * sizeof(uint32_t));
      vQ2 += 8;

      const __m256i cQ21 =
          fLasxSetD(iq2xxs_grid[vAux8[3]], iq2xxs_grid[vAux8[2]], iq2xxs_grid[vAux8[1]], iq2xxs_grid[vAux8[0]]);
      const __m256i cQ22 =
          fLasxSetD(iq2xxs_grid[vAux8[11]], iq2xxs_grid[vAux8[10]], iq2xxs_grid[vAux8[9]], iq2xxs_grid[vAux8[8]]);
      const __m256i cS21 = fLasxSetD(vSigns64[(aAux32[1] >> 21) & 127], vSigns64[(aAux32[1] >> 14) & 127],
                                     vSigns64[(aAux32[1] >> 7) & 127], vSigns64[(aAux32[1] >> 0) & 127]);
      const __m256i cS22 = fLasxSetD(vSigns64[(aAux32[3] >> 21) & 127], vSigns64[(aAux32[3] >> 14) & 127],
                                     vSigns64[(aAux32[3] >> 7) & 127], vSigns64[(aAux32[3] >> 0) & 127]);
      const __m256i cQ8s1 = __lasx_xvsigncov_b(cS21, cQ81);
      const __m256i cQ8s2 = __lasx_xvsigncov_b(cS22, cQ82);
      const __m256i cDot1 = fLasxMaddubsH(cQ21, cQ8s1);
      const __m256i cDot2 = fLasxMaddubsH(cQ22, cQ8s2);
      const uint16_t cLs1 = aAux32[1] >> 28;
      const uint16_t cLs2 = aAux32[3] >> 28;
      const __m256i cP1 = fLasxMaddH(cDot1, __lasx_xvreplgr2vr_h(2 * cLs1 + 1));
      const __m256i cP2 = fLasxMaddH(cDot2, __lasx_xvreplgr2vr_h(2 * cLs2 + 1));
      vSumi1 = __lasx_xvadd_w(vSumi1, cP1);
      vSumi2 = __lasx_xvadd_w(vSumi2, cP2);
    }

    vAccumf = __lasx_xvfmadd_s(__lasx_xvreplfr2vr_s(cD), __lasx_xvffint_s_w(__lasx_xvadd_w(vSumi1, vSumi2)), vAccumf);
  }

  *pS = 0.125f * fHsumFloat8(vAccumf);

#else
  UNUSED(vX);
  UNUSED(vY);
  UNUSED(cNb);
  ggml_vec_dot_iq2_xxs_q8_K_generic(pN, pS, pBs, pVx, pBx, pVy, pBy, pNrc);
#endif
}

void ggml_vec_dot_iq2_xs_q8_K(int pN, float *GGML_RESTRICT pS, size_t pBs, const void *GGML_RESTRICT pVx, size_t pBx,
                              const void *GGML_RESTRICT pVy, size_t pBy, int pNrc) {
  assert(pN % QK_K == 0);
  assert(pNrc == 1);
  UNUSED(pNrc);
  UNUSED(pBx);
  UNUSED(pBy);
  UNUSED(pBs);

  const block_iq2_xs *GGML_RESTRICT vX = pVx;
  const block_q8_K *GGML_RESTRICT vY = pVy;

  const int cNb = pN / QK_K;

#if defined(__loongarch_asx)

  const __m256i cMone = __lasx_xvreplgr2vr_b(1);
  static const char cBlockSignShuffleMask1[32] = {
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02,
      0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x06, 0x06, 0x06, 0x06, 0x06, 0x06, 0x06, 0x06,
  };
  static const char cBlockSignShuffleMask2[32] = {
      0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x0a, 0x0a, 0x0a, 0x0a, 0x0a, 0x0a, 0x0a, 0x0a,
      0x0c, 0x0c, 0x0c, 0x0c, 0x0c, 0x0c, 0x0c, 0x0c, 0x0e, 0x0e, 0x0e, 0x0e, 0x0e, 0x0e, 0x0e, 0x0e,
  };
  static const uint8_t cBitSelectorMaskBytes[32] = {
      0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,
      0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,
  };

  const __m256i cBitSelectorMask = __lasx_xvld((const __m256i *)cBitSelectorMaskBytes, 0);
  const __m256i cBlockSignShuffle1 = __lasx_xvld((const __m256i *)cBlockSignShuffleMask1, 0);
  const __m256i cBlockSignShuffle2 = __lasx_xvld((const __m256i *)cBlockSignShuffleMask2, 0);

  static const uint8_t cKBitHelper[32] = {
      0x00, 0x80, 0x80, 0x00, 0x80, 0x00, 0x00, 0x80, 0x80, 0x00, 0x00, 0x80, 0x00, 0x80, 0x80, 0x00,
      0x00, 0x80, 0x80, 0x00, 0x80, 0x00, 0x00, 0x80, 0x80, 0x00, 0x00, 0x80, 0x00, 0x80, 0x80, 0x00,
  };
  const __m256i cBitHelper = __lasx_xvld((const __m256i *)cKBitHelper, 0);
  const __m256i cM511 = __lasx_xvreplgr2vr_h(511);
  const __m128i cM4 = __lsx_vreplgr2vr_b(0xf);
  const __m128i cM1 = __lsx_vreplgr2vr_b(1);

  uint64_t vAux64;

  // somewhat hacky, but gives a significant boost in performance
  __m256i vAuxGindex;
  const uint16_t *vGindex = (const uint16_t *)&vAuxGindex;

  __m256 vAccumf = (__m256)__lasx_xvldi(0);
  for (int vI = 0; vI < cNb; ++vI) {
    const float cD = GGML_CPU_FP16_TO_FP32(vX[vI].d) * vY[vI].d;
    const uint16_t *GGML_RESTRICT vQ2 = vX[vI].qs;
    const int8_t *GGML_RESTRICT vQ8 = vY[vI].qs;

    memcpy(&vAux64, vX[vI].scales, 8);
    __m128i vStmp = __lsx_vreplgr2vr_d(vAux64);
    vStmp = __lsx_vilvl_b(__lsx_vand_v(__lsx_vsrli_h(vStmp, 4), cM4), __lsx_vand_v(vStmp, cM4));
    const __m128i cScales = __lsx_vadd_b(__lsx_vslli_h(vStmp, 1), cM1);

    __m256i vSumi1 = __lasx_xvldi(0);
    __m256i vSumi2 = __lasx_xvldi(0);
    for (int vIb32 = 0; vIb32 < QK_K / 32; vIb32 += 4) {

      const __m256i cQ2Data = __lasx_xvld((const __m256i *)vQ2, 0);
      vQ2 += 16;
      vAuxGindex = __lasx_xvand_v(cQ2Data, cM511);

      const __m256i cPartialSignBits = __lasx_xvsrli_h(cQ2Data, 9);
      const __m256i cPartialSignBitsUpper = __lasx_xvsrli_h(cQ2Data, 13);
      const __m256i cPartialSignBitsForCounting = __lasx_xvxor_v(cPartialSignBits, cPartialSignBitsUpper);

      const __m256i cOddBits = fLasxShuffleB(cBitHelper, cPartialSignBitsForCounting);
      const __m256i cFullSignBits = __lasx_xvor_v(cPartialSignBits, cOddBits);

      const __m256i cQ81 = __lasx_xvld((const __m256i *)vQ8, 0);
      vQ8 += 32;
      const __m256i cQ82 = __lasx_xvld((const __m256i *)vQ8, 0);
      vQ8 += 32;
      const __m256i cQ83 = __lasx_xvld((const __m256i *)vQ8, 0);
      vQ8 += 32;
      const __m256i cQ84 = __lasx_xvld((const __m256i *)vQ8, 0);
      vQ8 += 32;

      const __m256i cQ21 =
          fLasxSetD(iq2xs_grid[vGindex[3]], iq2xs_grid[vGindex[2]], iq2xs_grid[vGindex[1]], iq2xs_grid[vGindex[0]]);
      const __m256i cQ22 =
          fLasxSetD(iq2xs_grid[vGindex[7]], iq2xs_grid[vGindex[6]], iq2xs_grid[vGindex[5]], iq2xs_grid[vGindex[4]]);
      const __m256i cQ23 =
          fLasxSetD(iq2xs_grid[vGindex[11]], iq2xs_grid[vGindex[10]], iq2xs_grid[vGindex[9]], iq2xs_grid[vGindex[8]]);
      const __m256i cQ24 =
          fLasxSetD(iq2xs_grid[vGindex[15]], iq2xs_grid[vGindex[14]], iq2xs_grid[vGindex[13]], iq2xs_grid[vGindex[12]]);

      const __m128i cFullSignsL = fLasxExtracti128(cFullSignBits, 0);
      const __m128i cFullSignsH = fLasxExtracti128(cFullSignBits, 1);
      const __m256i cFullSigns1 = fLasxInsertf128(cFullSignsL, cFullSignsL);
      const __m256i cFullSigns2 = fLasxInsertf128(cFullSignsH, cFullSignsH);

      __m256i vSigns;
      vSigns = fLasxShuffleB(cFullSigns1, cBlockSignShuffle1);
      vSigns = __lasx_xvseq_b(__lasx_xvand_v(vSigns, cBitSelectorMask), cBitSelectorMask);
      const __m256i cQ8s1 = __lasx_xvsigncov_b(__lasx_xvor_v(vSigns, cMone), cQ81);

      vSigns = fLasxShuffleB(cFullSigns1, cBlockSignShuffle2);
      vSigns = __lasx_xvseq_b(__lasx_xvand_v(vSigns, cBitSelectorMask), cBitSelectorMask);
      const __m256i cQ8s2 = __lasx_xvsigncov_b(__lasx_xvor_v(vSigns, cMone), cQ82);

      vSigns = fLasxShuffleB(cFullSigns2, cBlockSignShuffle1);
      vSigns = __lasx_xvseq_b(__lasx_xvand_v(vSigns, cBitSelectorMask), cBitSelectorMask);
      const __m256i cQ8s3 = __lasx_xvsigncov_b(__lasx_xvor_v(vSigns, cMone), cQ83);

      vSigns = fLasxShuffleB(cFullSigns2, cBlockSignShuffle2);
      vSigns = __lasx_xvseq_b(__lasx_xvand_v(vSigns, cBitSelectorMask), cBitSelectorMask);
      const __m256i cQ8s4 = __lasx_xvsigncov_b(__lasx_xvor_v(vSigns, cMone), cQ84);

      const __m256i cDot1 = fLasxMaddubsH(cQ21, cQ8s1);
      const __m256i cDot2 = fLasxMaddubsH(cQ22, cQ8s2);
      const __m256i cDot3 = fLasxMaddubsH(cQ23, cQ8s3);
      const __m256i cDot4 = fLasxMaddubsH(cQ24, cQ8s4);

      const __m256i cSc1 = fLasxExt816(fLsxShuffleB(cScales, fGetScaleShuffle(vIb32 + 0)));
      const __m256i cSc2 = fLasxExt816(fLsxShuffleB(cScales, fGetScaleShuffle(vIb32 + 1)));
      const __m256i cSc3 = fLasxExt816(fLsxShuffleB(cScales, fGetScaleShuffle(vIb32 + 2)));
      const __m256i cSc4 = fLasxExt816(fLsxShuffleB(cScales, fGetScaleShuffle(vIb32 + 3)));

      vSumi1 = __lasx_xvadd_w(vSumi1, fLasxMaddH(cDot1, cSc1));
      vSumi2 = __lasx_xvadd_w(vSumi2, fLasxMaddH(cDot2, cSc2));
      vSumi1 = __lasx_xvadd_w(vSumi1, fLasxMaddH(cDot3, cSc3));
      vSumi2 = __lasx_xvadd_w(vSumi2, fLasxMaddH(cDot4, cSc4));
    }

    vAccumf = __lasx_xvfmadd_s(__lasx_xvreplfr2vr_s(cD), __lasx_xvffint_s_w(__lasx_xvadd_w(vSumi1, vSumi2)), vAccumf);
  }

  *pS = 0.125f * fHsumFloat8(vAccumf);

#else
  UNUSED(vX);
  UNUSED(vY);
  UNUSED(cNb);
  ggml_vec_dot_iq2_xs_q8_K_generic(pN, pS, pBs, pVx, pBx, pVy, pBy, pNrc);
#endif
}

void ggml_vec_dot_iq2_s_q8_K(int pN, float *GGML_RESTRICT pS, size_t pBs, const void *GGML_RESTRICT pVx, size_t pBx,
                             const void *GGML_RESTRICT pVy, size_t pBy, int pNrc) {
  assert(pN % QK_K == 0);
  assert(pNrc == 1);
  UNUSED(pNrc);
  UNUSED(pBx);
  UNUSED(pBy);
  UNUSED(pBs);

  const block_iq2_s *GGML_RESTRICT vX = pVx;
  const block_q8_K *GGML_RESTRICT vY = pVy;

  const int cNb = pN / QK_K;

#if defined(__loongarch_asx)

  static const uint8_t cKMask1[32] = {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x01, 0x01,
                                      0x01, 0x01, 0x01, 0x01, 0x01, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02,
                                      0x02, 0x02, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03};

  static const uint8_t cKMask2[32] = {
      0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,
      0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,
  };

  const __m128i cM4 = __lsx_vreplgr2vr_b(0xf);
  const __m128i cM1 = __lsx_vreplgr2vr_b(1);

  const __m256i cMask1 = __lasx_xvld((const __m256i *)cKMask1, 0);
  const __m256i cMask2 = __lasx_xvld((const __m256i *)cKMask2, 0);
  uint64_t vAux64;

  __m256 vAccumf = (__m256)__lasx_xvldi(0);
  for (int vI = 0; vI < cNb; ++vI) {
    const float cD = GGML_CPU_FP16_TO_FP32(vX[vI].d) * vY[vI].d;
    const uint8_t *GGML_RESTRICT vQs = vX[vI].qs;
    const uint8_t *GGML_RESTRICT vQh = vX[vI].qh;
    const uint16_t *GGML_RESTRICT vSigns = (const uint16_t *)(vX[vI].qs + QK_K / 8);
    const int8_t *GGML_RESTRICT vQ8 = vY[vI].qs;

    __m128i vTmp1;
    memcpy(&vAux64, vX[vI].scales, 8);
    vTmp1 = __lsx_vinsgr2vr_d(vTmp1, vAux64, 0);
    vTmp1 = __lsx_vinsgr2vr_d(vTmp1, vAux64 >> 4, 1);
    const __m128i cScales8 = __lsx_vadd_b(__lsx_vslli_h(__lsx_vand_v(vTmp1, cM4), 1), cM1);
    const __m256i cScales16 = fLasxExt816(cScales8); // 0 2 4 6 8 10 12 14 1 3 5 7 9 11 13 15

    __m256i vSumi1 = __lasx_xvldi(0);
    __m256i vSumi2 = __lasx_xvldi(0);
    for (int vIb32 = 0; vIb32 < QK_K / 32; vIb32 += 2) {
      const __m256i cQ81 = __lasx_xvld((const __m256i *)vQ8, 0);
      vQ8 += 32;
      const __m256i cQ82 = __lasx_xvld((const __m256i *)vQ8, 0);
      vQ8 += 32;
      const __m256i cQ21 = fLasxSetD(
          iq2s_grid[vQs[3] | ((vQh[vIb32 + 0] << 2) & 0x300)], iq2s_grid[vQs[2] | ((vQh[vIb32 + 0] << 4) & 0x300)],
          iq2s_grid[vQs[1] | ((vQh[vIb32 + 0] << 6) & 0x300)], iq2s_grid[vQs[0] | ((vQh[vIb32 + 0] << 8) & 0x300)]);
      const __m256i cQ22 = fLasxSetD(
          iq2s_grid[vQs[7] | ((vQh[vIb32 + 1] << 2) & 0x300)], iq2s_grid[vQs[6] | ((vQh[vIb32 + 1] << 4) & 0x300)],
          iq2s_grid[vQs[5] | ((vQh[vIb32 + 1] << 6) & 0x300)], iq2s_grid[vQs[4] | ((vQh[vIb32 + 1] << 8) & 0x300)]);
      vQs += 8;

      __m256i vAux256 = __lasx_xvreplgr2vr_w(vSigns[0] | ((uint32_t)vSigns[1] << 16));
      vAux256 = __lasx_xvand_v(fLasxShuffleB(vAux256, cMask1), cMask2);
      const __m256i cS21 = __lasx_xvseq_b(vAux256, cMask2);
      const __m256i cQ8s1 = __lasx_xvsub_b(__lasx_xvxor_v(cS21, cQ81), cS21);

      vAux256 = __lasx_xvreplgr2vr_w(vSigns[2] | ((uint32_t)vSigns[3] << 16));
      vAux256 = __lasx_xvand_v(fLasxShuffleB(vAux256, cMask1), cMask2);
      const __m256i cS22 = __lasx_xvseq_b(vAux256, cMask2);
      const __m256i cQ8s2 = __lasx_xvsub_b(__lasx_xvxor_v(cS22, cQ82), cS22);

      vSigns += 4;

      const __m256i cDot1 = fLasxMaddubsH(cQ21, cQ8s1); // blocks 2*ib32+0, 2*ib32+1
      const __m256i cDot2 = fLasxMaddubsH(cQ22, cQ8s2); // blocks 2*ib32+2, 2*ib32+3

      const __m256i cP1 = fLasxMaddH(cDot1, fLasxShuffleB(cScales16, fGetScaleShuffleK4(vIb32 + 0)));
      const __m256i cP2 = fLasxMaddH(cDot2, fLasxShuffleB(cScales16, fGetScaleShuffleK4(vIb32 + 1)));
      vSumi1 = __lasx_xvadd_w(vSumi1, cP1);
      vSumi2 = __lasx_xvadd_w(vSumi2, cP2);
    }

    vAccumf = __lasx_xvfmadd_s(__lasx_xvreplfr2vr_s(cD), __lasx_xvffint_s_w(__lasx_xvadd_w(vSumi1, vSumi2)), vAccumf);
  }

  *pS = 0.125f * fHsumFloat8(vAccumf);

#else
  UNUSED(vX);
  UNUSED(vY);
  UNUSED(cNb);
  ggml_vec_dot_iq2_s_q8_K_generic(pN, pS, pBs, pVx, pBx, pVy, pBy, pNrc);
#endif
}

void ggml_vec_dot_iq3_xxs_q8_K(int pN, float *GGML_RESTRICT pS, size_t pBs, const void *GGML_RESTRICT pVx, size_t pBx,
                               const void *GGML_RESTRICT pVy, size_t pBy, int pNrc) {
  assert(pN % QK_K == 0);
  assert(pNrc == 1);
  UNUSED(pNrc);
  UNUSED(pBx);
  UNUSED(pBy);
  UNUSED(pBs);

  const block_iq3_xxs *GGML_RESTRICT vX = pVx;
  const block_q8_K *GGML_RESTRICT vY = pVy;

  const int cNb = pN / QK_K;

#if defined(__loongarch_asx)

  const uint64_t *vSigns64 = (const uint64_t *)keven_signs_q2xs;

  uint32_t aAux32[2];

  __m256 vAccumf = (__m256)__lasx_xvldi(0);
  for (int vI = 0; vI < cNb; ++vI) {
    const float cD = GGML_CPU_FP16_TO_FP32(vX[vI].d) * vY[vI].d;
    const uint8_t *GGML_RESTRICT vQ3 = vX[vI].qs;
    const uint8_t *GGML_RESTRICT vGas = vX[vI].qs + QK_K / 4;
    const int8_t *GGML_RESTRICT vQ8 = vY[vI].qs;
    __m256i vSumi1 = __lasx_xvldi(0);
    __m256i vSumi2 = __lasx_xvldi(0);
    for (int vIb32 = 0; vIb32 < QK_K / 32; vIb32 += 2) {
      const __m256i cQ81 = __lasx_xvld((const __m256i *)vQ8, 0);
      vQ8 += 32;
      const __m256i cQ82 = __lasx_xvld((const __m256i *)vQ8, 0);
      vQ8 += 32;
      const __m256i cQ21 =
          fLasxSetW(iq3xxs_grid[vQ3[7]], iq3xxs_grid[vQ3[6]], iq3xxs_grid[vQ3[5]], iq3xxs_grid[vQ3[4]],
                    iq3xxs_grid[vQ3[3]], iq3xxs_grid[vQ3[2]], iq3xxs_grid[vQ3[1]], iq3xxs_grid[vQ3[0]]);
      vQ3 += 8;
      const __m256i cQ22 =
          fLasxSetW(iq3xxs_grid[vQ3[7]], iq3xxs_grid[vQ3[6]], iq3xxs_grid[vQ3[5]], iq3xxs_grid[vQ3[4]],
                    iq3xxs_grid[vQ3[3]], iq3xxs_grid[vQ3[2]], iq3xxs_grid[vQ3[1]], iq3xxs_grid[vQ3[0]]);
      vQ3 += 8;
      memcpy(aAux32, vGas, 8);
      vGas += 8;

      const __m256i cS21 = fLasxSetD(vSigns64[(aAux32[0] >> 21) & 127], vSigns64[(aAux32[0] >> 14) & 127],
                                     vSigns64[(aAux32[0] >> 7) & 127], vSigns64[(aAux32[0] >> 0) & 127]);
      const __m256i cS22 = fLasxSetD(vSigns64[(aAux32[1] >> 21) & 127], vSigns64[(aAux32[1] >> 14) & 127],
                                     vSigns64[(aAux32[1] >> 7) & 127], vSigns64[(aAux32[1] >> 0) & 127]);
      const __m256i cQ8s1 = __lasx_xvsigncov_b(cS21, cQ81);
      const __m256i cQ8s2 = __lasx_xvsigncov_b(cS22, cQ82);
      const __m256i cDot1 = fLasxMaddubsH(cQ21, cQ8s1);
      const __m256i cDot2 = fLasxMaddubsH(cQ22, cQ8s2);
      const uint16_t cLs1 = aAux32[0] >> 28;
      const uint16_t cLs2 = aAux32[1] >> 28;

      const __m256i cP1 = fLasxMaddH(cDot1, __lasx_xvreplgr2vr_h(2 * cLs1 + 1));
      const __m256i cP2 = fLasxMaddH(cDot2, __lasx_xvreplgr2vr_h(2 * cLs2 + 1));
      vSumi1 = __lasx_xvadd_w(vSumi1, cP1);
      vSumi2 = __lasx_xvadd_w(vSumi2, cP2);
    }

    vAccumf = __lasx_xvfmadd_s(__lasx_xvreplfr2vr_s(cD), __lasx_xvffint_s_w(__lasx_xvadd_w(vSumi1, vSumi2)), vAccumf);
  }

  *pS = 0.25f * fHsumFloat8(vAccumf);

#else
  UNUSED(vX);
  UNUSED(vY);
  UNUSED(cNb);
  ggml_vec_dot_iq3_xxs_q8_K_generic(pN, pS, pBs, pVx, pBx, pVy, pBy, pNrc);
#endif
}

void ggml_vec_dot_iq3_s_q8_K(int pN, float *GGML_RESTRICT pS, size_t pBs, const void *GGML_RESTRICT pVx, size_t pBx,
                             const void *GGML_RESTRICT pVy, size_t pBy, int pNrc) {
  assert(pN % QK_K == 0);
  assert(pNrc == 1);
  UNUSED(pNrc);
  UNUSED(pBx);
  UNUSED(pBy);
  UNUSED(pBs);

  const block_iq3_s *GGML_RESTRICT vX = pVx;
  const block_q8_K *GGML_RESTRICT vY = pVy;

  const int cNb = pN / QK_K;

#if defined(__loongarch_asx)

  static const uint8_t cKMask1[32] = {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x01, 0x01,
                                      0x01, 0x01, 0x01, 0x01, 0x01, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02,
                                      0x02, 0x02, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03};

  static const uint8_t cKMask2[32] = {
      0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,
      0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,
  };

  const __m256i cMask1 = __lasx_xvld((const __m256i *)cKMask1, 0);
  const __m256i cMask2 = __lasx_xvld((const __m256i *)cKMask2, 0);

  __m256i vIdxShift = fLasxSetW(1, 2, 3, 4, 5, 6, 7, 8);
  const __m256i cIdxMask = __lasx_xvreplgr2vr_w(256);

  typedef union {
    __m256i aVec[2];
    uint32_t aIndex[16];
  } index_t;

  index_t vIdx;

  __m256 vAccumf = (__m256)__lasx_xvldi(0);
  for (int vI = 0; vI < cNb; ++vI) {
    const float cD = GGML_CPU_FP16_TO_FP32(vX[vI].d) * vY[vI].d;
    const uint8_t *GGML_RESTRICT vQs = vX[vI].qs;
    const uint8_t *GGML_RESTRICT vQh = vX[vI].qh;
    const uint16_t *GGML_RESTRICT vSigns = (const uint16_t *)vX[vI].signs;
    const int8_t *GGML_RESTRICT vQ8 = vY[vI].qs;
    __m256i vSumi1 = __lasx_xvldi(0);
    __m256i vSumi2 = __lasx_xvldi(0);
    for (int vIb32 = 0; vIb32 < QK_K / 32; vIb32 += 2) {
      const __m256i cQ81 = __lasx_xvld((const __m256i *)vQ8, 0);
      vQ8 += 32;
      const __m256i cQ82 = __lasx_xvld((const __m256i *)vQ8, 0);
      vQ8 += 32;
      const __m256i cIdxL = fLasxExtu816(__lsx_vld(vQs, 0));
      vQs += 16;
      vIdx.aVec[0] = __lasx_xvreplgr2vr_w(vQh[vIb32 + 0]);
      vIdx.aVec[1] = __lasx_xvreplgr2vr_w(vQh[vIb32 + 1]);
      vIdx.aVec[0] = __lasx_xvand_v(__lasx_xvsll_w(vIdx.aVec[0], vIdxShift), cIdxMask);
      vIdx.aVec[1] = __lasx_xvand_v(__lasx_xvsll_w(vIdx.aVec[1], vIdxShift), cIdxMask);
      vIdx.aVec[0] = __lasx_xvor_v(vIdx.aVec[0], fLasxExt1632(fLasxExtracti128(cIdxL, 0)));
      vIdx.aVec[1] = __lasx_xvor_v(vIdx.aVec[1], fLasxExt1632(fLasxExtracti128(cIdxL, 1)));

      // At leat on my CPU (Ryzen 7950X), using _mm256_i32gather_epi32 is slower than _mm256_set_epi32. Strange.
      // const __m256i q2_1 = _mm256_i32gather_epi32((const int *)iq3s_grid, idx.vec[0], 4);
      // const __m256i q2_2 = _mm256_i32gather_epi32((const int *)iq3s_grid, idx.vec[1], 4);
      const __m256i cQ21 = fLasxSetW(iq3s_grid[vIdx.aIndex[7]], iq3s_grid[vIdx.aIndex[6]], iq3s_grid[vIdx.aIndex[5]],
                                     iq3s_grid[vIdx.aIndex[4]], iq3s_grid[vIdx.aIndex[3]], iq3s_grid[vIdx.aIndex[2]],
                                     iq3s_grid[vIdx.aIndex[1]], iq3s_grid[vIdx.aIndex[0]]);
      const __m256i cQ22 = fLasxSetW(iq3s_grid[vIdx.aIndex[15]], iq3s_grid[vIdx.aIndex[14]], iq3s_grid[vIdx.aIndex[13]],
                                     iq3s_grid[vIdx.aIndex[12]], iq3s_grid[vIdx.aIndex[11]], iq3s_grid[vIdx.aIndex[10]],
                                     iq3s_grid[vIdx.aIndex[9]], iq3s_grid[vIdx.aIndex[8]]);

      __m256i vAux256 = __lasx_xvreplgr2vr_w(vSigns[0] | (vSigns[1] << 16));
      vAux256 = __lasx_xvand_v(fLasxShuffleB(vAux256, cMask1), cMask2);
      const __m256i cS21 = __lasx_xvseq_b(vAux256, cMask2);
      const __m256i cQ8s1 = __lasx_xvsub_b(__lasx_xvxor_v(cS21, cQ81), cS21);

      vAux256 = __lasx_xvreplgr2vr_w(vSigns[2] | (vSigns[3] << 16));
      vAux256 = __lasx_xvand_v(fLasxShuffleB(vAux256, cMask1), cMask2);
      const __m256i cS22 = __lasx_xvseq_b(vAux256, cMask2);
      const __m256i cQ8s2 = __lasx_xvsub_b(__lasx_xvxor_v(cS22, cQ82), cS22);

      vSigns += 4;

      const __m256i cDot1 = fLasxMaddubsH(cQ21, cQ8s1);
      const __m256i cDot2 = fLasxMaddubsH(cQ22, cQ8s2);
      const uint16_t cLs1 = vX[vI].scales[vIb32 / 2] & 0xf;
      const uint16_t cLs2 = vX[vI].scales[vIb32 / 2] >> 4;
      const __m256i cP1 = fLasxMaddH(cDot1, __lasx_xvreplgr2vr_h(2 * cLs1 + 1));
      const __m256i cP2 = fLasxMaddH(cDot2, __lasx_xvreplgr2vr_h(2 * cLs2 + 1));
      vSumi1 = __lasx_xvadd_w(vSumi1, cP1);
      vSumi2 = __lasx_xvadd_w(vSumi2, cP2);
    }

    vAccumf = __lasx_xvfmadd_s(__lasx_xvreplfr2vr_s(cD), __lasx_xvffint_s_w(__lasx_xvadd_w(vSumi1, vSumi2)), vAccumf);
  }

  *pS = fHsumFloat8(vAccumf);

#else
  UNUSED(vX);
  UNUSED(vY);
  UNUSED(cNb);
  ggml_vec_dot_iq3_s_q8_K_generic(pN, pS, pBs, pVx, pBx, pVy, pBy, pNrc);
#endif
}

#if defined(__loongarch_asx)
static inline __m256i fMulAddEpi8(const __m256i pX, const __m256i pY) {
  const __m256i cA = __lasx_xvmulwev_h_b(pX, pY);
  const __m256i cB = __lasx_xvmulwod_h_b(pX, pY);
  return __lasx_xvadd_h(cA, cB);
}
#endif

void ggml_vec_dot_iq1_s_q8_K(int pN, float *GGML_RESTRICT pS, size_t pBs, const void *GGML_RESTRICT pVx, size_t pBx,
                             const void *GGML_RESTRICT pVy, size_t pBy, int pNrc) {
  assert(pN % QK_K == 0);
  assert(pNrc == 1);
  UNUSED(pNrc);
  UNUSED(pBx);
  UNUSED(pBy);
  UNUSED(pBs);

  const block_iq1_s *GGML_RESTRICT vX = pVx;
  const block_q8_K *GGML_RESTRICT vY = pVy;

  const int cNb = pN / QK_K;

#if defined(__loongarch_asx)

  __m256 vAccum = (__m256)__lasx_xvldi(0);
  float vAccum1 = 0;
  for (int vI = 0; vI < cNb; ++vI) {

    const int8_t *vQ8 = vY[vI].qs;
    const uint8_t *vQs = vX[vI].qs;
    const uint16_t *vQh = vX[vI].qh;

    __m256i vSumi = __lasx_xvldi(0);
    int vSumi1 = 0;
    for (int vIb = 0; vIb < QK_K / 32; vIb += 2) {
      __m256i vQ1b1 = __lasx_xvinsgr2vr_d(vQ1b1, iq1s_grid[vQs[0] | ((vQh[vIb + 0] << 8) & 0x700)], 0);
      vQ1b1 = __lasx_xvinsgr2vr_d(vQ1b1, iq1s_grid[vQs[1] | ((vQh[vIb + 0] << 5) & 0x700)], 1);
      vQ1b1 = __lasx_xvinsgr2vr_d(vQ1b1, iq1s_grid[vQs[2] | ((vQh[vIb + 0] << 2) & 0x700)], 2);
      vQ1b1 = __lasx_xvinsgr2vr_d(vQ1b1, iq1s_grid[vQs[3] | ((vQh[vIb + 0] >> 1) & 0x700)], 3);

      __m256i vQ1b2 = __lasx_xvinsgr2vr_d(vQ1b2, iq1s_grid[vQs[4] | ((vQh[vIb + 1] << 8) & 0x700)], 0);
      vQ1b2 = __lasx_xvinsgr2vr_d(vQ1b2, iq1s_grid[vQs[5] | ((vQh[vIb + 1] << 5) & 0x700)], 1);
      vQ1b2 = __lasx_xvinsgr2vr_d(vQ1b2, iq1s_grid[vQs[6] | ((vQh[vIb + 1] << 2) & 0x700)], 2);
      vQ1b2 = __lasx_xvinsgr2vr_d(vQ1b2, iq1s_grid[vQs[7] | ((vQh[vIb + 1] >> 1) & 0x700)], 3);

      vQs += 8;
      const __m256i cQ8b1 = __lasx_xvld((const __m256i *)vQ8, 0);
      vQ8 += 32;
      const __m256i cQ8b2 = __lasx_xvld((const __m256i *)vQ8, 0);
      vQ8 += 32;

      const __m256i cDot1 = fMulAddEpi8(vQ1b1, cQ8b1);
      const __m256i cDot2 = fMulAddEpi8(vQ1b2, cQ8b2);
      const int16_t cLs1 = 2 * ((vQh[vIb + 0] >> 12) & 7) + 1;
      const int16_t cLs2 = 2 * ((vQh[vIb + 1] >> 12) & 7) + 1;

      __m256i vTmp1, vTmp5, vTmp6;
      vTmp1 = __lasx_xvreplgr2vr_h(cLs1);
      vTmp5 = __lasx_xvmulwev_w_h(cDot1, vTmp1);
      vTmp6 = __lasx_xvmulwod_w_h(cDot1, vTmp1);
      const __m256i cP1 = __lasx_xvadd_w(vTmp5, vTmp6);

      vTmp1 = __lasx_xvreplgr2vr_h(cLs2);
      vTmp5 = __lasx_xvmulwev_w_h(cDot2, vTmp1);
      vTmp6 = __lasx_xvmulwod_w_h(cDot2, vTmp1);
      const __m256i cP2 = __lasx_xvadd_w(vTmp5, vTmp6);

      vSumi = __lasx_xvadd_w(vSumi, __lasx_xvadd_w(cP1, cP2));
      vSumi1 += (vY[vI].bsums[2 * vIb + 0] + vY[vI].bsums[2 * vIb + 1]) * (vQh[vIb + 0] & 0x8000 ? -1 : 1) * cLs1 +
                (vY[vI].bsums[2 * vIb + 2] + vY[vI].bsums[2 * vIb + 3]) * (vQh[vIb + 1] & 0x8000 ? -1 : 1) * cLs2;
    }

    const float cD = vY[vI].d * GGML_CPU_FP16_TO_FP32(vX[vI].d);
    vAccum = __lasx_xvfmadd_s(__lasx_xvreplfr2vr_s(cD), __lasx_xvffint_s_w(vSumi), vAccum);
    vAccum1 += cD * vSumi1;
  }

  *pS = fHsumFloat8(vAccum) + IQ1S_DELTA * vAccum1;

#else
  UNUSED(vX);
  UNUSED(vY);
  UNUSED(cNb);
  ggml_vec_dot_iq1_s_q8_K_generic(pN, pS, pBs, pVx, pBx, pVy, pBy, pNrc);
#endif
}

void ggml_vec_dot_iq4_nl_q8_0(int pN, float *GGML_RESTRICT pS, size_t pBs, const void *GGML_RESTRICT pVx, size_t pBx,
                              const void *GGML_RESTRICT pVy, size_t pBy, int pNrc) {
  assert(pNrc == 1);
  UNUSED(pNrc);
  UNUSED(pBx);
  UNUSED(pBy);
  UNUSED(pBs);
  assert(pN % QK4_NL == 0);
  static_assert(QK4_NL == QK8_0, "QK4_NL and QK8_0 must be the same");

  const block_iq4_nl *GGML_RESTRICT vX = pVx;
  const block_q8_0 *GGML_RESTRICT vY = pVy;

  const int cNb = pN / QK4_NL;

  int vIb = 0;
  float vSumf = 0;

#if defined(__loongarch_asx)

  const __m128i cValues128 = __lsx_vld((const __m128i *)kvalues_iq4nl, 0);
  const __m128i cM4b = __lsx_vreplgr2vr_b(0x0f);
  const __m256i cMone = __lasx_xvreplgr2vr_h(1);

  __m256 vAccum1 = (__m256)__lasx_xvldi(0);
  __m256 vAccum2 = (__m256)__lasx_xvldi(0);
  for (; vIb + 1 < cNb; vIb += 2) {
    const __m128i cQ4bits1 = __lsx_vld((const __m128i *)vX[vIb + 0].qs, 0);
    const __m128i cQ4bits2 = __lsx_vld((const __m128i *)vX[vIb + 1].qs, 0);
    const __m256i cQ8b1 = __lasx_xvld((const __m256i *)vY[vIb + 0].qs, 0);
    const __m256i cQ8b2 = __lasx_xvld((const __m256i *)vY[vIb + 1].qs, 0);
    const __m256i cQ4b1 = fLasxInsertf128(fLsxShuffleB(cValues128, __lsx_vand_v(__lsx_vsrli_h(cQ4bits1, 4), cM4b)),
                                          fLsxShuffleB(cValues128, __lsx_vand_v(cQ4bits1, cM4b)));
    const __m256i cQ4b2 = fLasxInsertf128(fLsxShuffleB(cValues128, __lsx_vand_v(__lsx_vsrli_h(cQ4bits2, 4), cM4b)),
                                          fLsxShuffleB(cValues128, __lsx_vand_v(cQ4bits2, cM4b)));
    const __m256i cP161 = fMulAddEpi8(cQ4b1, cQ8b1);
    const __m256i cP162 = fMulAddEpi8(cQ4b2, cQ8b2);
    const __m256i cP1 = fLasxMaddH(cP161, cMone);
    const __m256i cP2 = fLasxMaddH(cP162, cMone);
    vAccum1 = __lasx_xvfmadd_s(
        __lasx_xvreplfr2vr_s(GGML_CPU_FP16_TO_FP32(vY[vIb + 0].d) * GGML_CPU_FP16_TO_FP32(vX[vIb + 0].d)),
        __lasx_xvffint_s_w(cP1), vAccum1);
    vAccum2 = __lasx_xvfmadd_s(
        __lasx_xvreplfr2vr_s(GGML_CPU_FP16_TO_FP32(vY[vIb + 1].d) * GGML_CPU_FP16_TO_FP32(vX[vIb + 1].d)),
        __lasx_xvffint_s_w(cP2), vAccum2);
  }

  vSumf = fHsumFloat8(__lasx_xvfadd_s(vAccum1, vAccum2));

#endif
  for (; vIb < cNb; ++vIb) {
    const float cD = GGML_CPU_FP16_TO_FP32(vY[vIb].d) * GGML_CPU_FP16_TO_FP32(vX[vIb].d);
    int vSumi1 = 0, vSumi2 = 0;
    for (int vJ = 0; vJ < QK4_NL / 2; ++vJ) {
      vSumi1 += vY[vIb].qs[vJ + 0] * kvalues_iq4nl[vX[vIb].qs[vJ] & 0xf];
      vSumi2 += vY[vIb].qs[vJ + QK4_NL / 2] * kvalues_iq4nl[vX[vIb].qs[vJ] >> 4];
    }
    vSumf += cD * (vSumi1 + vSumi2);
  }
  *pS = vSumf;
}

void ggml_vec_dot_iq4_xs_q8_K(int pN, float *GGML_RESTRICT pS, size_t pBs, const void *GGML_RESTRICT pVx, size_t pBx,
                              const void *GGML_RESTRICT pVy, size_t pBy, int pNrc) {
  assert(pNrc == 1);
  UNUSED(pNrc);
  UNUSED(pBx);
  UNUSED(pBy);
  UNUSED(pBs);
  assert(pN % QK_K == 0);

  const block_iq4_xs *GGML_RESTRICT vX = pVx;
  const block_q8_K *GGML_RESTRICT vY = pVy;

  const int cNb = pN / QK_K;

#if defined(__loongarch_asx)

  const __m128i cValues128 = __lsx_vld((const __m128i *)kvalues_iq4nl, 0);

  __m256 vAccum = (__m256)__lasx_xvldi(0);

  for (int vIbl = 0; vIbl < cNb; ++vIbl) {
    const uint8_t *vQs = vX[vIbl].qs;
    const int8_t *vQ8 = vY[vIbl].qs;
    uint16_t vSh = vX[vIbl].scales_h;
    __m256i vSumi1 = __lasx_xvldi(0);
    __m256i vSumi2 = __lasx_xvldi(0);
    for (int vIb = 0; vIb < QK_K / 32; vIb += 2) {
      const __m128i cQ4bits1 = __lsx_vld((const __m128i *)vQs, 0);
      vQs += 16;
      const __m128i cQ4bits2 = __lsx_vld((const __m128i *)vQs, 0);
      vQs += 16;
      const __m256i cQ8b1 = __lasx_xvld((const __m256i *)vQ8, 0);
      vQ8 += 32;
      const __m256i cQ8b2 = __lasx_xvld((const __m256i *)vQ8, 0);
      vQ8 += 32;
      const __m256i cQ4b1 = fLasxInsertf128(__lsx_vshuf_b(cValues128, cValues128, __lsx_vsrli_b(cQ4bits1, 4)),
                                            __lsx_vshuf_b(cValues128, cValues128, __lsx_vandi_b(cQ4bits1, 0xf)));
      const __m256i cQ4b2 = fLasxInsertf128(__lsx_vshuf_b(cValues128, cValues128, __lsx_vsrli_b(cQ4bits2, 4)),
                                            __lsx_vshuf_b(cValues128, cValues128, __lsx_vandi_b(cQ4bits2, 0xf)));
      const __m256i cP161 = fMulAddEpi8(cQ4b1, cQ8b1);
      const __m256i cP162 = fMulAddEpi8(cQ4b2, cQ8b2);
      const int16_t cLs1 = ((vX[vIbl].scales_l[vIb / 2] & 0xf) | ((vSh << 4) & 0x30)) - 32;
      const int16_t cLs2 = ((vX[vIbl].scales_l[vIb / 2] >> 4) | ((vSh << 2) & 0x30)) - 32;
      vSh >>= 4;
      const __m256i cP1 = fLasxMaddH(cP161, __lasx_xvreplgr2vr_h(cLs1));
      const __m256i cP2 = fLasxMaddH(cP162, __lasx_xvreplgr2vr_h(cLs2));
      vSumi1 = __lasx_xvadd_w(cP1, vSumi1);
      vSumi2 = __lasx_xvadd_w(cP2, vSumi2);
    }
    vAccum = __lasx_xvfmadd_s(__lasx_xvreplfr2vr_s(GGML_CPU_FP16_TO_FP32(vX[vIbl].d) * vY[vIbl].d),
                              __lasx_xvffint_s_w(__lasx_xvadd_w(vSumi1, vSumi2)), vAccum);
  }

  *pS = fHsumFloat8(vAccum);

#elif defined(__loongarch_sx)

  const __m128i cValues128 = __lsx_vld((const __m128i *)kvalues_iq4nl, 0);

  __m128 vAccum = (__m128)__lsx_vldi(0);
  for (int vIbl = 0; vIbl < cNb; ++vIbl) {
    const uint8_t *vQs = vX[vIbl].qs;
    const int8_t *vQ8 = vY[vIbl].qs;
    uint16_t vSh = vX[vIbl].scales_h;
    __m128i sumi = __lsx_vldi(0);
    for (int vIb = 0; vIb < QK_K / 32; ++vIb) {
      const __m128i q4bits = __lsx_vld((const __m128i *)vQs, 0);
      vQs += 16;
      const __m128i q8b_0 = __lsx_vld((const __m128i *)vQ8, 0);
      vQ8 += 16;
      const __m128i cQ8b1 = __lsx_vld((const __m128i *)vQ8, 0);
      vQ8 += 16;
      const __m128i q4b_0 = __lsx_vshuf_b(cValues128, cValues128, __lsx_vandi_b(q4bits, 0xf));
      const __m128i cQ4b1 = __lsx_vshuf_b(cValues128, cValues128, __lsx_vsrli_b(q4bits, 4));
      const __m128i p16_0 = fLsxMaddubsH(q4b_0, q8b_0);
      const __m128i cP161 = fLsxMaddubsH(cQ4b1, cQ8b1);
      const int16_t ls = (((vX[vIbl].scales_l[vIb / 2] >> ((vIb & 1) * 4)) & 0xf) | ((vSh & 0x3) << 4)) - 32;
      vSh >>= 2;
      sumi = __lsx_vadd_w(fLsxMaddH(p16_0, __lsx_vreplgr2vr_h(ls)), sumi);
      sumi = __lsx_vadd_w(fLsxMaddH(cP161, __lsx_vreplgr2vr_h(ls)), sumi);
    }
    const float ds = GGML_CPU_FP16_TO_FP32(vX[vIbl].d) * vY[vIbl].d;
    vAccum = __lsx_vfadd_s(__lsx_vfmul_s(__lsx_vreplfr2vr_s(ds), __lsx_vffint_s_w(sumi)), vAccum);
  }

  *pS = ((v4f32)fLsxHaddS(fLsxHaddS(vAccum, vAccum), fLsxHaddS(vAccum, vAccum)))[0];

#else
  UNUSED(vX);
  UNUSED(vY);
  UNUSED(cNb);
  ggml_vec_dot_iq4_xs_q8_K_generic(pN, pS, pBs, pVx, pBx, pVy, pBy, pNrc);
#endif
}
