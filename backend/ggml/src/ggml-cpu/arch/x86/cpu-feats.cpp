#include "ggml-backend-impl.h"

#if defined(__x86_64__) || (defined(_MSC_VER) && defined(_M_AMD64))

#ifdef _MSC_VER
#include <intrin.h>
#endif

#include <cstring>
#include <vector>
#include <bitset>
#include <array>
#include <string>

// ref: https://cdrdv2-public.intel.com/782156/325383-sdm-vol-2abcd.pdf
struct cpuid_x86 {
  bool mSsE3(void) { return vF1Ecx[0]; }
  bool mPclmulqdq(void) { return vF1Ecx[1]; }
  bool mMonitor(void) { return vF1Ecx[3]; }
  bool mSssE3(void) { return vF1Ecx[9]; }
  bool mFma(void) { return vF1Ecx[12]; }
  bool mCmpxchG16B(void) { return vF1Ecx[13]; }
  bool mSsE41(void) { return vF1Ecx[19]; }
  bool mSsE42(void) { return vF1Ecx[20]; }
  bool mMovbe(void) { return vF1Ecx[22]; }
  bool mPopcnt(void) { return vF1Ecx[23]; }
  bool mAes(void) { return vF1Ecx[25]; }
  bool mXsave(void) { return vF1Ecx[26]; }
  bool mOsxsave(void) { return vF1Ecx[27]; }
  bool mAvx(void) { return vF1Ecx[28]; }
  bool mF16C(void) { return vF1Ecx[29]; }
  bool mRdrand(void) { return vF1Ecx[30]; }

  bool mMsr(void) { return vF1Edx[5]; }
  bool mCX8(void) { return vF1Edx[8]; }
  bool mSep(void) { return vF1Edx[11]; }
  bool mCmov(void) { return vF1Edx[15]; }
  bool mClfsh(void) { return vF1Edx[19]; }
  bool mMmx(void) { return vF1Edx[23]; }
  bool mFxsr(void) { return vF1Edx[24]; }
  bool mSse(void) { return vF1Edx[25]; }
  bool mSsE2(void) { return vF1Edx[26]; }

  bool mFsgsbase(void) { return vF7Ebx[0]; }
  bool mBmI1(void) { return vF7Ebx[3]; }
  bool mHle(void) { return vIsIntel && vF7Ebx[4]; }
  bool mAvX2(void) { return vF7Ebx[5]; }
  bool mBmI2(void) { return vF7Ebx[8]; }
  bool mErms(void) { return vF7Ebx[9]; }
  bool mInvpcid(void) { return vF7Ebx[10]; }
  bool mRtm(void) { return vIsIntel && vF7Ebx[11]; }
  bool mAvX512F(void) { return vF7Ebx[16]; }
  bool mAvX512Dq(void) { return vF7Ebx[17]; }
  bool mRdseed(void) { return vF7Ebx[18]; }
  bool mAdx(void) { return vF7Ebx[19]; }
  bool mAvX512Pf(void) { return vF7Ebx[26]; }
  bool mAvX512Er(void) { return vF7Ebx[27]; }
  bool mAvX512Cd(void) { return vF7Ebx[28]; }
  bool mAvX512Bw(void) { return vF7Ebx[30]; }
  bool mAvX512Vl(void) { return vF7Ebx[31]; }

  bool mSha(void) { return vF7Ebx[29]; }

  bool mPrefetchwT1(void) { return vF7Ecx[0]; }

  bool mLahf(void) { return vF81Ecx[0]; }
  bool mLzcnt(void) { return vIsIntel && vF81Ecx[5]; }
  bool mAbm(void) { return vIsAmd && vF81Ecx[5]; }
  bool mSsE4a(void) { return vIsAmd && vF81Ecx[6]; }
  bool mXop(void) { return vIsAmd && vF81Ecx[11]; }
  bool mTbm(void) { return vIsAmd && vF81Ecx[21]; }

  bool mSyscall(void) { return vIsIntel && vF81Edx[11]; }
  bool mMmxext(void) { return vIsAmd && vF81Edx[22]; }
  bool mRdtscp(void) { return vIsIntel && vF81Edx[27]; }
  bool m3Dnowext(void) { return vIsAmd && vF81Edx[30]; }
  bool m3Dnow(void) { return vIsAmd && vF81Edx[31]; }

  bool mAvX512Vbmi(void) { return vF7Ecx[1]; }
  bool mAvX512Vnni(void) { return vF7Ecx[11]; }
  bool mAvX512FP16(void) { return vF7Edx[23]; }
  bool mAvX512BF16(void) { return vF71Eax[5]; }
  bool mAvxVnni(void) { return vF71Eax[4]; }

  bool mAmxTile(void) { return vF7Edx[24]; }
  bool mAmxInT8(void) { return vF7Edx[25]; }
  bool mAmxFP16(void) { return vF71Eax[21]; }
  bool mAmxBF16(void) { return vF7Edx[22]; }

#ifdef _MSC_VER
  static void cpuid(int cpu_info[4], int eax) { __cpuid(cpu_info, eax); }
  static void cpuidex(int cpu_info[4], int eax, int ecx) { __cpuidex(cpu_info, eax, ecx); }
#else
  static void cpuid(int pCpuInfo[4], int pEax) {
    __asm__ __volatile__("cpuid"
                         : "=a"(pCpuInfo[0]), "=b"(pCpuInfo[1]), "=c"(pCpuInfo[2]), "=d"(pCpuInfo[3])
                         : "a"(pEax), "c"(0));
  }
  static void cpuidex(int pCpuInfo[4], int pEax, int pEcx) {
    __asm__ __volatile__("cpuid"
                         : "=a"(pCpuInfo[0]), "=b"(pCpuInfo[1]), "=c"(pCpuInfo[2]), "=d"(pCpuInfo[3])
                         : "a"(pEax), "c"(pEcx));
  }
#endif

  cpuid_x86() {
    std::array<int, 4> aCpui;
    std::vector<std::array<int, 4>> lData;

    // calling __cpuid with 0x0 as the function_id argument
    // gets the number of the highest valid function ID.
    cpuid(aCpui.data(), 0);
    int vNIds = aCpui[0];

    for (int vI = 0; vI <= vNIds; ++vI) {
      cpuidex(aCpui.data(), vI, 0);
      lData.push_back(aCpui);
    }

    // capture vendor string
    char vVendor[0x20] = {};
    *reinterpret_cast<int *>(vVendor) = lData[0][1];
    *reinterpret_cast<int *>(vVendor + 4) = lData[0][3];
    *reinterpret_cast<int *>(vVendor + 8) = lData[0][2];
    this->vVendor = vVendor;
    if (this->vVendor == "GenuineIntel") {
      vIsIntel = true;
    } else if (this->vVendor == "AuthenticAMD") {
      vIsAmd = true;
    }

    // load bitset with flags for function 0x00000001
    if (vNIds >= 1) {
      vF1Ecx = lData[1][2];
      vF1Edx = lData[1][3];
    }

    // load bitset with flags for function 0x00000007
    if (vNIds >= 7) {
      vF7Ebx = lData[7][1];
      vF7Ecx = lData[7][2];
      vF7Edx = lData[7][3];
      cpuidex(aCpui.data(), 7, 1);
      vF71Eax = aCpui[0];
    }

    // calling __cpuid with 0x80000000 as the function_id argument
    // gets the number of the highest valid extended ID.
    cpuid(aCpui.data(), 0x80000000);
    unsigned int vNExIds = aCpui[0];

    std::vector<std::array<int, 4>> lExtData;
    for (unsigned int vI = 0x80000000; vI <= vNExIds; ++vI) {
      cpuidex(aCpui.data(), vI, 0);
      lExtData.push_back(aCpui);
    }

    // load bitset with flags for function 0x80000001
    if (vNExIds >= 0x80000001) {
      vF81Ecx = lExtData[1][2];
      vF81Edx = lExtData[1][3];
    }

    // interpret CPU brand string if reported
    char vBrand[0x40] = {};
    if (vNExIds >= 0x80000004) {
      std::memcpy(vBrand, lExtData[2].data(), sizeof(aCpui));
      std::memcpy(vBrand + 16, lExtData[3].data(), sizeof(aCpui));
      std::memcpy(vBrand + 32, lExtData[4].data(), sizeof(aCpui));
      this->vBrand = vBrand;
    }
  }

  bool vIsIntel = false;
  bool vIsAmd = false;
  std::string vVendor;
  std::string vBrand;
  std::bitset<32> vF1Ecx;
  std::bitset<32> vF1Edx;
  std::bitset<32> vF7Ebx;
  std::bitset<32> vF7Ecx;
  std::bitset<32> vF7Edx;
  std::bitset<32> vF71Eax;
  std::bitset<32> vF81Ecx;
  std::bitset<32> vF81Edx;
};

#if 0
void test_x86_is() {
    cpuid_x86 is;
    printf("CPU Vendor: %s\n", is.vVendor.c_str());
    printf("Brand: %s\n", is.vBrand.c_str());
    printf("is_intel: %d\n", is.vIsIntel);
    printf("is_amd: %d\n", is.vIsAmd);
    printf("sse3: %d\n", is.mSsE3());
    printf("pclmulqdq: %d\n", is.mPclmulqdq());
    printf("ssse3: %d\n", is.mSssE3());
    printf("fma: %d\n", is.mFma());
    printf("cmpxchg16b: %d\n", is.mCmpxchG16B());
    printf("sse41: %d\n", is.mSsE41());
    printf("sse42: %d\n", is.mSsE42());
    printf("movbe: %d\n", is.mMovbe());
    printf("popcnt: %d\n", is.mPopcnt());
    printf("aes: %d\n", is.mAes());
    printf("xsave: %d\n", is.mXsave());
    printf("osxsave: %d\n", is.mOsxsave());
    printf("avx: %d\n", is.mAvx());
    printf("f16c: %d\n", is.mF16C());
    printf("rdrand: %d\n", is.mRdrand());
    printf("msr: %d\n", is.mMsr());
    printf("cx8: %d\n", is.mCX8());
    printf("sep: %d\n", is.mSep());
    printf("cmov: %d\n", is.mCmov());
    printf("clflush: %d\n", is.mClfsh());
    printf("mmx: %d\n", is.mMmx());
    printf("fxsr: %d\n", is.mFxsr());
    printf("sse: %d\n", is.mSse());
    printf("sse2: %d\n", is.mSsE2());
    printf("fsgsbase: %d\n", is.mFsgsbase());
    printf("bmi1: %d\n", is.mBmI1());
    printf("hle: %d\n", is.mHle());
    printf("avx2: %d\n", is.mAvX2());
    printf("bmi2: %d\n", is.mBmI2());
    printf("erms: %d\n", is.mErms());
    printf("invpcid: %d\n", is.mInvpcid());
    printf("rtm: %d\n", is.mRtm());
    printf("avx512f: %d\n", is.mAvX512F());
    printf("rdseed: %d\n", is.mRdseed());
    printf("adx: %d\n", is.mAdx());
    printf("avx512pf: %d\n", is.mAvX512Pf());
    printf("avx512er: %d\n", is.mAvX512Er());
    printf("avx512cd: %d\n", is.mAvX512Cd());
    printf("sha: %d\n", is.mSha());
    printf("prefetchwt1: %d\n", is.mPrefetchwT1());
    printf("lahf: %d\n", is.mLahf());
    printf("lzcnt: %d\n", is.mLzcnt());
    printf("abm: %d\n", is.mAbm());
    printf("sse4a: %d\n", is.mSsE4a());
    printf("xop: %d\n", is.mXop());
    printf("tbm: %d\n", is.mTbm());
    printf("syscall: %d\n", is.mSyscall());
    printf("mmxext: %d\n", is.mMmxext());
    printf("rdtscp: %d\n", is.mRdtscp());
    printf("3dnowext: %d\n", is.m3Dnowext());
    printf("3dnow: %d\n", is.m3Dnow());
    printf("avx512_vbmi: %d\n", is.mAvX512Vbmi());
    printf("avx512_vnni: %d\n", is.mAvX512Vnni());
    printf("avx512_fp16: %d\n", is.mAvX512FP16());
    printf("avx512_bf16: %d\n", is.mAvX512BF16());
    printf("amx_tile: %d\n", is.mAmxTile());
    printf("amx_int8: %d\n", is.mAmxInT8());
    printf("amx_fp16: %d\n", is.mAmxFP16());
    printf("amx_bf16: %d\n", is.mAmxBF16());
}
#endif

static int fGgmlBackendCpuX86Score() {
  // FIXME: this does not check for OS support

  int vScore = 1;
  cpuid_x86 vIs;

#ifdef GGML_FMA
  if (!vIs.mFma()) {
    return 0;
  }
  vScore += 1;
#endif
#ifdef GGML_F16C
  if (!vIs.mF16C()) {
    return 0;
  }
  vScore += 1 << 1;
#endif
#ifdef GGML_SSE42
  if (!vIs.mSsE42()) {
    return 0;
  }
  vScore += 1 << 2;
#endif
#ifdef GGML_BMI2
  if (!vIs.mBmI2()) {
    return 0;
  }
  vScore += 1 << 3;
#endif
#ifdef GGML_AVX
  if (!vIs.mAvx()) {
    return 0;
  }
  vScore += 1 << 4;
#endif
#ifdef GGML_AVX2
  if (!vIs.mAvX2()) {
    return 0;
  }
  vScore += 1 << 5;
#endif
#ifdef GGML_AVX_VNNI
  if (!vIs.mAvxVnni()) {
    return 0;
  }
  vScore += 1 << 6;
#endif
#ifdef GGML_AVX512
  if (!vIs.mAvX512F()) {
    return 0;
  }
  if (!vIs.mAvX512Cd()) {
    return 0;
  }
  if (!vIs.mAvX512Vl()) {
    return 0;
  }
  if (!vIs.mAvX512Dq()) {
    return 0;
  }
  if (!vIs.mAvX512Bw()) {
    return 0;
  }
  vScore += 1 << 7;
#endif
#ifdef GGML_AVX512_VBMI
  if (!vIs.mAvX512Vbmi()) {
    return 0;
  }
  vScore += 1 << 8;
#endif
#ifdef GGML_AVX512_BF16
  if (!vIs.mAvX512BF16()) {
    return 0;
  }
  vScore += 1 << 9;
#endif
#ifdef GGML_AVX512_VNNI
  if (!vIs.mAvX512Vnni()) {
    return 0;
  }
  vScore += 1 << 10;
#endif
#ifdef GGML_AMX_INT8
  if (!vIs.mAmxInT8()) {
    return 0;
  }
  vScore += 1 << 11;
#endif

  return vScore;
}

GGML_BACKEND_DL_SCORE_IMPL(fGgmlBackendCpuX86Score)

#endif // defined(__x86_64__) || (defined(_MSC_VER) && defined(_M_AMD64))
