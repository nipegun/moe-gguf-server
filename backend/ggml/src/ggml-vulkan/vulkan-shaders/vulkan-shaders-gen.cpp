#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <stdexcept>
#include <array>
#include <vector>
#include <map>
#include <thread>
#include <mutex>
#include <future>
#include <queue>
#include <condition_variable>
#include <atomic>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cassert>
#include <algorithm>
#include <sys/stat.h>
#include <sys/types.h>
#include <filesystem>

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#include <direct.h> // For _mkdir on Windows
#else
#include <unistd.h>
#include <sys/wait.h>
#include <fcntl.h>
#endif

#define ASYNCIO_CONCURRENCY 64

std::mutex lock;
std::vector<std::pair<std::string, std::string>> shader_fnames;
// Set when any shader subprocess fails (non-zero exit / stderr / launch failure) so the
// build is stopped instead of silently producing a broken libggml-vulkan. (issue #24393)
static std::atomic<bool> compile_failed{false};
std::locale c_locale("C");

std::string GLSLC = "glslc";
std::string input_filepath = "";
std::string output_dir = "/tmp";
std::string target_hpp = "";
std::string target_cpp = "";

const std::vector<std::string> type_names = {
    "f32",   "f16",     "q1_0",  "q2_0",   "q4_0",   "q4_1",  "q5_0",  "q5_1",    "q8_0",
    "q2_k",  "q3_k",    "q4_k",  "q5_k",   "q6_k",   "iq1_s", "iq1_m", "iq2_xxs", "iq2_xs",
    "iq2_s", "iq3_xxs", "iq3_s", "iq4_xs", "iq4_nl", "mxfp4", "nvfp4", "bf16",
};

enum MatMulIdType {
  NONE,
  DEFAULT,
  SUBGROUP,
};

namespace {

int fExecuteCommand(std::vector<std::string> &pCommand, std::string &pStdoutStr, std::string &pStderrStr) {
#ifdef _WIN32
  HANDLE stdout_read, stdout_write;
  HANDLE stderr_read, stderr_write;
  SECURITY_ATTRIBUTES sa = {sizeof(SECURITY_ATTRIBUTES), NULL, TRUE};

  if (!CreatePipe(&stdout_read, &stdout_write, &sa, 0) || !SetHandleInformation(stdout_read, HANDLE_FLAG_INHERIT, 0)) {
    throw std::runtime_error("Failed to create stdout pipe");
  }

  if (!CreatePipe(&stderr_read, &stderr_write, &sa, 0) || !SetHandleInformation(stderr_read, HANDLE_FLAG_INHERIT, 0)) {
    throw std::runtime_error("Failed to create stderr pipe");
  }

  PROCESS_INFORMATION pi;
  STARTUPINFOA si = {};
  si.cb = sizeof(STARTUPINFOA);
  si.dwFlags = STARTF_USESTDHANDLES;
  si.hStdOutput = stdout_write;
  si.hStdError = stderr_write;

  std::string cmd;
  for (const auto &vPart : pCommand) {
    cmd += vPart + " ";
  }

  if (!CreateProcessA(NULL, cmd.data(), NULL, NULL, TRUE, 0, NULL, NULL, &si, &pi)) {
    throw std::runtime_error("Failed to create process");
  }

  CloseHandle(stdout_write);
  CloseHandle(stderr_write);

  std::array<char, 128> vBuffer;
  DWORD vBytesRead;

  while (ReadFile(stdout_read, vBuffer.data(), (DWORD)vBuffer.size(), &vBytesRead, NULL) && vBytesRead > 0) {
    pStdoutStr.append(vBuffer.data(), vBytesRead);
  }

  while (ReadFile(stderr_read, vBuffer.data(), (DWORD)vBuffer.size(), &vBytesRead, NULL) && vBytesRead > 0) {
    pStderrStr.append(vBuffer.data(), vBytesRead);
  }

  CloseHandle(stdout_read);
  CloseHandle(stderr_read);
  WaitForSingleObject(pi.hProcess, INFINITE);
  DWORD exit_code = 1;
  GetExitCodeProcess(pi.hProcess, &exit_code);
  CloseHandle(pi.hProcess);
  CloseHandle(pi.hThread);
  return (int)exit_code;
#else
  int aStdoutPipe[2];
  int aStderrPipe[2];

  if (pipe(aStdoutPipe) != 0 || pipe(aStderrPipe) != 0) {
    throw std::runtime_error("Failed to create pipes");
  }

  pid_t vPid = fork();
  if (vPid < 0) {
    std::cerr << strerror(errno) << "\n";
    throw std::runtime_error("Failed to fork process");
  }

  std::vector<char *> lArgv;
  for (std::string &vPart : pCommand) {
    lArgv.push_back(vPart.data());
  }
  lArgv.push_back(nullptr);

  if (vPid == 0) {
    close(aStdoutPipe[0]);
    close(aStderrPipe[0]);
    dup2(aStdoutPipe[1], STDOUT_FILENO);
    dup2(aStderrPipe[1], STDERR_FILENO);
    close(aStdoutPipe[1]);
    close(aStderrPipe[1]);
    execvp(lArgv[0], lArgv.data());
    _exit(EXIT_FAILURE);
  } else {
    close(aStdoutPipe[1]);
    close(aStderrPipe[1]);

    std::array<char, 128> vBuffer;
    ssize_t vBytesRead;

    while ((vBytesRead = read(aStdoutPipe[0], vBuffer.data(), vBuffer.size())) > 0) {
      pStdoutStr.append(vBuffer.data(), vBytesRead);
    }

    while ((vBytesRead = read(aStderrPipe[0], vBuffer.data(), vBuffer.size())) > 0) {
      pStderrStr.append(vBuffer.data(), vBytesRead);
    }

    close(aStdoutPipe[0]);
    close(aStderrPipe[0]);
    int vStatus = 0;
    waitpid(vPid, &vStatus, 0);
    return WIFEXITED(vStatus) ? WEXITSTATUS(vStatus) : -1;
  }
#endif
}

bool fDirectoryExists(const std::string &pPath) {
  struct stat vInfo;
  if (stat(pPath.c_str(), &vInfo) != 0) {
    return false; // Path doesn't exist or can't be accessed
  }
  return (vInfo.st_mode & S_IFDIR) != 0; // Check if it is a directory
}

bool fCreateDirectory(const std::string &pPath) {
#ifdef _WIN32
  return _mkdir(pPath.c_str()) == 0 || errno == EEXIST; // EEXIST means the directory already exists
#else
  return mkdir(pPath.c_str(), 0755) == 0 || errno == EEXIST; // 0755 is the directory permissions
#endif
}

std::string fToUppercase(const std::string &pInput) {
  std::string vResult = pInput;
  for (char &vC : vResult) {
    vC = std::toupper(vC);
  }
  return vResult;
}

bool fStringStartsWith(const std::string &pStr, const std::string &pPrefix) {
  if (pPrefix.size() > pStr.size()) {
    return false;
  }
  return std::equal(pPrefix.begin(), pPrefix.end(), pStr.begin());
}

bool fStringEndsWith(const std::string &pStr, const std::string &pSuffix) {
  if (pSuffix.size() > pStr.size()) {
    return false;
  }
  return std::equal(pSuffix.rbegin(), pSuffix.rend(), pStr.rbegin());
}

bool fIsQuantizedType(const std::string &pTypeName) {
  return pTypeName != "f32" && pTypeName != "f16" && pTypeName != "bf16";
}

bool fIsLegacyQuant(const std::string &pTypeName) {
  return pTypeName == "q2_0" || pTypeName == "q4_0" || pTypeName == "q4_1" || pTypeName == "q5_0" ||
         pTypeName == "q5_1" || pTypeName == "q8_0";
}

bool fIsKQuant(const std::string &pTypeName) { return fStringEndsWith(pTypeName, "_k"); }

bool fIsIqQuant(const std::string &pTypeName) { return fStringStartsWith(pTypeName, "iq"); }

static const char path_separator = '/';

std::string fJoinPaths(const std::string &pPath1, const std::string &pPath2) {
  return pPath1 + path_separator + pPath2;
}

std::string fBasename(const std::string &pPath) { return pPath.substr(pPath.find_last_of("/\\") + 1); }

std::stringstream fMakeGenericStringstream() {
  std::stringstream vSs;
  vSs.imbue(c_locale);
  return vSs;
}

std::string fReadBinaryFile(const std::string &pPath, bool pMayNotExist = false) {
  FILE *vF = fopen(pPath.c_str(), "rb");
  if (!vF) {
    if (!pMayNotExist) {
      std::cerr << "Error opening file: " << pPath << " (" << strerror(errno) << ")\n";
    }
    return {};
  }

  fseek(vF, 0, SEEK_END);
  size_t vSize = ftell(vF);
  fseek(vF, 0, SEEK_SET);

  std::string vData(vSize, '\0');
  size_t vReadSize = fread(vData.data(), 1, vSize, vF);
  fclose(vF);
  if (vReadSize != vSize) {
    std::cerr << "Error reading file: " << pPath << " (" << strerror(errno) << ")\n";
    return {};
  }

  return vData;
}

void fWriteBinaryFile(const std::string &pPath, const std::string &pContent) {
  FILE *vF = fopen(pPath.c_str(), "wb");
  if (!vF) {
    std::cerr << "Error opening file for writing: " << pPath << " (" << strerror(errno) << ")\n";
    return;
  }

  size_t vWriteSize = fwrite(pContent.data(), 1, pContent.size(), vF);
  fclose(vF);
  if (vWriteSize != pContent.size()) {
    std::cerr << "Error writing file: " << pPath << " (" << strerror(errno) << ")\n";
    return;
  }
}

void fWriteFileIfChanged(const std::string &pPath, const std::string &pContent) {
  std::string vExisting = fReadBinaryFile(pPath, true);
  if (vExisting != pContent) {
    fWriteBinaryFile(pPath, pContent);
  }
}

// variables to track number of compiles in progress
static uint32_t compile_count = 0;
static std::mutex compile_count_mutex;
static std::condition_variable compile_count_cond;
static bool generate_dep_file = true;

void fDecrementCompileCount(uint32_t *pCount) {
  if (pCount) {
    std::lock_guard<std::mutex> vGuard(compile_count_mutex);
    assert(compile_count > 0);
    compile_count--;
    compile_count_cond.notify_all();
  }
}

using compile_count_guard = std::unique_ptr<uint32_t, decltype(&fDecrementCompileCount)>;

compile_count_guard fAcquireCompileSlot() {
  // wait until fewer than N compiles are in progress.
  // 16 is an arbitrary limit, the goal is to avoid "failed to create pipe" errors.
  uint32_t vN = std::max(1u, std::min(16u, std::thread::hardware_concurrency()));
  std::unique_lock<std::mutex> vGuard(compile_count_mutex);
  compile_count_cond.wait(vGuard, [vN] { return compile_count < vN; });
  compile_count++;
  return compile_count_guard(&compile_count, &fDecrementCompileCount);
}

void fStringToSpvFunc(std::string pName, std::string pInPath, std::string pOutPath,
                      std::map<std::string, std::string> pDefines, bool pCoopmat, bool pDepFile,
                      compile_count_guard pSlot) {
  std::string vTargetEnv =
      (pName.find("_cm2") != std::string::npos) ? "--target-env=vulkan1.3" : "--target-env=vulkan1.2";

#ifdef _WIN32
  std::vector<std::string> vCmd = {GLSLC, "-fshader-stage=compute", vTargetEnv, "\"" + pInPath + "\"",
                                   "-o",  "\"" + pOutPath + "\""};
#else
  std::vector<std::string> vCmd = {GLSLC, "-fshader-stage=compute", vTargetEnv, pInPath, "-o", pOutPath};
#endif

  // disable spirv-opt for coopmat shaders
  // disable spirv-opt for bf16 shaders
  // disable spirv-opt for rope shaders
  // disable spirv-opt for dot2 shaders (spirv-opt doesn't recognize SPV_VALVE_mixed_float_dot_product capability)
  if (!pCoopmat && pName.find("bf16") == std::string::npos && pName.find("rope") == std::string::npos &&
      pName.find("_dot2") == std::string::npos) {
    vCmd.push_back("-O");
  }

  if (pDepFile) {
    vCmd.push_back("-MD");
    vCmd.push_back("-MF");
#ifdef _WIN32
    vCmd.push_back("\"" + target_cpp + ".d\"");
#else
    vCmd.push_back(target_cpp + ".d");
#endif
  }

#ifdef GGML_VULKAN_SHADER_DEBUG_INFO
  vCmd.push_back("-g");
#endif

  for (const auto &vDefine : pDefines) {
    vCmd.push_back("-D" + vDefine.first + "=" + vDefine.second);
  }

  std::string vCommand;
  for (const auto &vPart : vCmd) {
    vCommand += vPart + " ";
  }

  std::string vStdoutStr, vStderrStr;
  try {
    // std::cout << "Executing command: ";
    // for (const auto& part : cmd) {
    //     std::cout << part << " ";
    // }
    // std::cout << std::endl;

    int vExitCode = fExecuteCommand(vCmd, vStdoutStr, vStderrStr);
    if (vExitCode != 0 || !vStderrStr.empty()) {
      std::cerr << "cannot compile " << pName << " (exit code " << vExitCode << ")\n\n";
      for (const auto &vPart : vCmd) {
        std::cerr << vPart << " ";
      }
      std::cerr << "\n\n" << vStderrStr << std::endl;
      compile_failed = true;
      return;
    }

    if (pDepFile) {
      // replace .spv output path with the embed .cpp path which is used as output in CMakeLists.txt
      std::string vDep = fReadBinaryFile(target_cpp + ".d", true);
      if (!vDep.empty()) {
        size_t vPos = vDep.find(pOutPath);
        if (vPos != std::string::npos) {
          vDep.replace(vPos, pOutPath.length(), target_cpp);
        }
        fWriteBinaryFile(target_cpp + ".d", vDep);
      }
    }

    std::lock_guard<std::mutex> vGuard(lock);
    shader_fnames.push_back(std::make_pair(pName, pOutPath));
  } catch (const std::exception &vE) {
    std::cerr << "Error executing command for " << pName << ": " << vE.what() << std::endl;
    compile_failed = true;
  }
}

std::map<std::string, std::string> fMergeMaps(const std::map<std::string, std::string> &pA,
                                              const std::map<std::string, std::string> &pB) {
  std::map<std::string, std::string> dResult = pA;
  dResult.insert(pB.begin(), pB.end());
  return dResult;
}

static std::deque<std::future<void>> compiles;
void fStringToSpv(std::string pName, const std::string &pSource, const std::map<std::string, std::string> &pDefines,
                  bool pFp16 = true, bool pCoopmat = false, bool pCoopmat2 = false, bool pF16acc = false,
                  const std::string &pSuffix = "") {
  pName = pName + (pF16acc ? "_f16acc" : "") + (pCoopmat ? "_cm1" : "") +
          (pCoopmat2 ? "_cm2" : (pFp16 ? "" : "_fp32")) + pSuffix;
  std::string vOutPath = fJoinPaths(output_dir, pName + ".spv");

  if (input_filepath == "") {
    // No input source to compile, only generate header for all shaders
    shader_fnames.push_back(std::pair(pName, vOutPath));
    return;
  } else if (fBasename(input_filepath) != pSource) {
    // Only compile shader variants matching the input filename
    return;
  }

  compile_count_guard vSlot = fAcquireCompileSlot();
  compiles.push_back(std::async(fStringToSpvFunc, pName, input_filepath, vOutPath, pDefines, pCoopmat,
                                generate_dep_file, std::move(vSlot)));
  // Don't write the same dep file from multiple processes
  generate_dep_file = false;

  // Clean up completed futures - don't accumulate virtual memory for completed threads' stacks.
  while (!compiles.empty() && compiles.front().wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
    compiles.pop_front();
  }
}

void fMatmulShaders(bool pFp16, MatMulIdType pMatmulIdType, bool pCoopmat, bool pCoopmat2, bool pF16acc,
                    bool pDot2 = false) {
  std::string vLoadVec = pCoopmat2 ? "1" : pFp16 ? "8" : "4";
  std::string vAlignedBTypeF32 = pCoopmat2 ? "float" : pFp16 ? "mat2x4" : "vec4";
  std::string vAlignedBTypeF16 = pCoopmat2 ? "float16_t" : pFp16 ? "f16mat2x4" : "f16vec4";
  std::string vDot2Sfx = pDot2 ? "_dot2" : "";

  std::map<std::string, std::string> vBaseDict;
  std::string vShaderName = "matmul";

  if (pMatmulIdType == MatMulIdType::DEFAULT) {
    vBaseDict["MUL_MAT_ID"] = "1";
    vShaderName = "matmul_id";
  } else if (pMatmulIdType == MatMulIdType::SUBGROUP) {
    vBaseDict["MUL_MAT_ID"] = "1";
    vBaseDict["MUL_MAT_ID_USE_SUBGROUPS"] = "1";
    vShaderName = "matmul_id_subgroup";
  }

  if (pFp16) {
    vBaseDict["FLOAT16"] = "1";
  }

  vBaseDict["ACC_TYPE"] = pF16acc ? "float16_t" : "float";
  vBaseDict["ACC_TYPEV2"] = pF16acc ? "f16vec2" : "vec2";
  if (pF16acc) {
    vBaseDict["ACC_TYPE_MAX"] = "float16_t(65504.0)";
  }

  if (pCoopmat) {
    vBaseDict["COOPMAT"] = "1";
  }
#if defined(GGML_VULKAN_COOPMAT2_DECODE_VECTOR_GLSLC_SUPPORT)
  if (pCoopmat2) {
    vBaseDict["GGML_VULKAN_COOPMAT2_DECODE_VECTOR"] = "1";
  }
#endif

  if (pDot2) {
    vBaseDict["DOT2_F16"] = "1";
  }

  const std::string cSourceName = pCoopmat2 ? "mul_mm_cm2.comp" : "mul_mm.comp";

  auto const &fFloatType = [&](int pVec, const std::string &pT) -> std::string {
    switch (pVec) {
    case 1:
      if (pT == "bf16") {
        // scalar path promotes to float
        if (!pCoopmat && !pCoopmat2) {
          return "float";
        }
        return "bfloat16_t";
      }
      if (pCoopmat2 || pFp16) {
        return "float16_t";
      }
      return "float";
    case 2:
      if (pT == "bf16") {
        // scalar path promotes to float
        if (!pCoopmat && !pCoopmat2) {
          return "vec2";
        }
        return "bf16vec2";
      }
      if (pCoopmat2 || pFp16) {
        return "f16vec2";
      }
      return "vec2";
    case 4:
      if (pT == "bf16") {
        // scalar path promotes to float
        if (!pCoopmat && !pCoopmat2) {
          return "vec4";
        }
        return "bf16vec4";
      }
      if (pCoopmat2 || pFp16) {
        return "f16vec4";
      }
      return "vec4";
    case 8:
      if (pT == "bf16") {
        // scalar path promotes to float
        if (!pCoopmat && !pCoopmat2) {
          return "mat2x4";
        }
        throw std::runtime_error("bf16 vec8 not supported");
      }
      if (pCoopmat2 || pFp16) {
        return "f16mat2x4";
      }
      return "mat2x4";
    default:
      throw std::runtime_error("invalid vector size");
    }
  };

  const std::map<std::string, std::string> dFloatTypeDictF16 = {
      {"FLOAT_TYPE", fFloatType(1, "f16")},
      {"FLOAT_TYPEV2", fFloatType(2, "f16")},
      {"FLOAT_TYPEV4", fFloatType(4, "f16")},
      {"FLOAT_TYPEV8", fFloatType(8, "f16")},
  };

  // Shaders with f16 B_TYPE
  fStringToSpv(vShaderName + "_f32_f16" + vDot2Sfx, cSourceName,
               fMergeMaps(fMergeMaps(vBaseDict, dFloatTypeDictF16), {{"DATA_A_F32", "1"},
                                                                     {"LOAD_VEC_A", vLoadVec},
                                                                     {"LOAD_VEC_B", vLoadVec},
                                                                     {"B_TYPE", vAlignedBTypeF16},
                                                                     {"B_TYPE_SCALAR", "float16_t"},
                                                                     {"B_TYPEV4", "f16vec4"},
                                                                     {"D_TYPE", "float"}}),
               pFp16, pCoopmat, pCoopmat2, pF16acc);

  fStringToSpv(vShaderName + "_f16" + vDot2Sfx, cSourceName,
               fMergeMaps(fMergeMaps(vBaseDict, dFloatTypeDictF16), {{"DATA_A_F16", "1"},
                                                                     {"LOAD_VEC_A", vLoadVec},
                                                                     {"LOAD_VEC_B", vLoadVec},
                                                                     {"B_TYPE", vAlignedBTypeF16},
                                                                     {"B_TYPE_SCALAR", "float16_t"},
                                                                     {"B_TYPEV4", "f16vec4"},
                                                                     {"D_TYPE", "float"}}),
               pFp16, pCoopmat, pCoopmat2, pF16acc);

  // bf16
  {
    // For aligned matmul loads
    std::string vLoadVecA = pCoopmat2 ? "1" : "4";

    // scalar path promotes to float
    std::string vToFloatType = (pCoopmat || pCoopmat2) ? "uintBitsToBFloat16EXT" : "bf16_to_fp32";

    const std::map<std::string, std::string> dFloatTypeDictBf16 = {
        {"FLOAT_TYPE", fFloatType(1, "bf16")},
        {"FLOAT_TYPEV2", fFloatType(2, "bf16")},
        {"FLOAT_TYPEV4", fFloatType(4, "bf16")},
    };

    // If bfloat16 is not supported, then only compile the scalar (promote to fp32) shader
#if !defined(GGML_VULKAN_BFLOAT16_GLSLC_SUPPORT)
    if (!(pCoopmat || pCoopmat2))
#endif
    {
      if (!pDot2) {
        fStringToSpv(vShaderName + "_bf16", cSourceName,
                     fMergeMaps(fMergeMaps(vBaseDict, dFloatTypeDictBf16),
                                {{"TO_FLOAT_TYPE", vToFloatType},
                                 {"DATA_A_BF16", "1"},
                                 {"LOAD_VEC_A", vLoadVecA},
                                 {"LOAD_VEC_B", "4"},
                                 {"B_TYPE", pCoopmat2 ? "bfloat16_t" : "u16vec4"},
                                 {"B_TYPE_SCALAR", pCoopmat2 ? "bfloat16_t" : "uint16_t"},
                                 {"B_TYPEV4", "bf16vec4"},
                                 {"D_TYPE", "float"},
                                 {"B_IS_FLOAT", "1"},
                                 {"DATA_B_BF16", "1"}}),
                     pFp16, pCoopmat, pCoopmat2, pF16acc);
      }
    }
  }

  for (const auto &vTname : type_names) {
    std::string vLoadVecQuant = "2";
    if ((vTname == "q1_0") || (vTname == "q4_0") || (vTname == "q4_1") || (vTname == "q5_1") || (vTname == "iq1_s") ||
        (vTname == "iq1_m") || (vTname == "iq2_xxs") || (vTname == "iq2_xs") || (vTname == "iq2_s"))
      vLoadVecQuant = "8";
    else if ((vTname == "q2_0") || (vTname == "q5_0") || (vTname == "q8_0") || (vTname == "q2_k") ||
             (vTname == "q4_k") || (vTname == "q5_k") || (vTname == "iq3_xxs") || (vTname == "iq3_s") ||
             (vTname == "iq4_xs") || (vTname == "iq4_nl") || (vTname == "mxfp4") || (vTname == "nvfp4"))
      vLoadVecQuant = "4";

    if (vTname == "bf16") {
      continue;
    }

    std::string vDataAKey = "DATA_A_" + fToUppercase(vTname);
    // For aligned matmul loads
    std::string vLoadVecA =
        (pCoopmat2 || vTname == "f32" || vTname == "f16" || vTname == "bf16") ? vLoadVec : vLoadVecQuant;

    const std::map<std::string, std::string> cFloatTypeDict = {
        {"FLOAT_TYPE", fFloatType(1, vTname)},
        {"FLOAT_TYPEV2", fFloatType(2, vTname)},
        {"FLOAT_TYPEV4", fFloatType(4, vTname)},
        {"FLOAT_TYPEV8", fFloatType(8, vTname)},
    };

    // don't generate f32 variants for coopmat2
    if (!pCoopmat2) {
      fStringToSpv(vShaderName + "_" + vTname + "_f32" + vDot2Sfx, cSourceName,
                   fMergeMaps(fMergeMaps(vBaseDict, cFloatTypeDict), {{vDataAKey, "1"},
                                                                      {"LOAD_VEC_A", vLoadVecA},
                                                                      {"LOAD_VEC_B", vLoadVec},
                                                                      {"B_TYPE", vAlignedBTypeF32},
                                                                      {"B_TYPE_SCALAR", "float"},
                                                                      {"B_TYPEV4", "vec4"},
                                                                      {"D_TYPE", "float"}}),
                   pFp16, pCoopmat, pCoopmat2, pF16acc);
    }

    if (vTname != "f16" && vTname != "f32") {
      fStringToSpv(vShaderName + "_" + vTname + "_f16" + vDot2Sfx, cSourceName,
                   fMergeMaps(fMergeMaps(vBaseDict, cFloatTypeDict), {{vDataAKey, "1"},
                                                                      {"LOAD_VEC_A", vLoadVecA},
                                                                      {"LOAD_VEC_B", vLoadVec},
                                                                      {"B_TYPE", vAlignedBTypeF16},
                                                                      {"B_TYPE_SCALAR", "float16_t"},
                                                                      {"B_TYPEV4", "f16vec4"},
                                                                      {"D_TYPE", "float"}}),
                   pFp16, pCoopmat, pCoopmat2, pF16acc);
    }

#if defined(GGML_VULKAN_FLOAT_E2M1_GLSLC_SUPPORT) && defined(GGML_VULKAN_FLOAT_E4M3_GLSLC_SUPPORT)
    if ((pCoopmat || pCoopmat2) && (vTname == "mxfp4" || vTname == "nvfp4")) {
      if (!pCoopmat2) {
        fStringToSpv(vShaderName + "_" + vTname + "_f32_ocp" + vDot2Sfx, cSourceName,
                     fMergeMaps(fMergeMaps(vBaseDict, cFloatTypeDict), {{vDataAKey, "1"},
                                                                        {"USE_OCP_FP4", "1"},
                                                                        {"LOAD_VEC_A", vLoadVecA},
                                                                        {"LOAD_VEC_B", vLoadVec},
                                                                        {"B_TYPE", vAlignedBTypeF32},
                                                                        {"B_TYPE_SCALAR", "float"},
                                                                        {"B_TYPEV4", "vec4"},
                                                                        {"D_TYPE", "float"}}),
                     pFp16, pCoopmat, pCoopmat2, pF16acc);
      }
      fStringToSpv(vShaderName + "_" + vTname + "_f16_ocp" + vDot2Sfx, cSourceName,
                   fMergeMaps(fMergeMaps(vBaseDict, cFloatTypeDict), {{vDataAKey, "1"},
                                                                      {"USE_OCP_FP4", "1"},
                                                                      {"LOAD_VEC_A", vLoadVecA},
                                                                      {"LOAD_VEC_B", vLoadVec},
                                                                      {"B_TYPE", vAlignedBTypeF16},
                                                                      {"B_TYPE_SCALAR", "float16_t"},
                                                                      {"B_TYPEV4", "f16vec4"},
                                                                      {"D_TYPE", "float"}}),
                   pFp16, pCoopmat, pCoopmat2, pF16acc);
    }
#endif

#if defined(GGML_VULKAN_INTEGER_DOT_GLSLC_SUPPORT)
    // Integer dot mmq performs better with f32 accumulators (different shader, skip for dot2)
    if (!pF16acc && !pCoopmat && !pCoopmat2 && !pDot2 &&
        (fIsLegacyQuant(vTname) || fIsKQuant(vTname) || vTname == "mxfp4")) {
      fStringToSpv(vShaderName + "_" + vTname + "_q8_1", "mul_mmq.comp",
                   fMergeMaps(fMergeMaps(vBaseDict, cFloatTypeDict),
                              {
                                  {vDataAKey, "1"},
                                  {"D_TYPE", "float"},
                              }),
                   pFp16, pCoopmat, pCoopmat2, pF16acc);
    }
#endif
  }
}

void fProcessShaders() {
  // matmul
  for (const MatMulIdType &vMatmulIdType : {MatMulIdType::NONE, MatMulIdType::DEFAULT, MatMulIdType::SUBGROUP}) {
    // No coopmats
    // fp32
    fMatmulShaders(false, vMatmulIdType, false, false, false);

    // fp16, fp32acc and fp16acc
    fMatmulShaders(true, vMatmulIdType, false, false, false);
    fMatmulShaders(true, vMatmulIdType, false, false, true);

    // dot2 variants (scalar fp16 only)
    fMatmulShaders(true, vMatmulIdType, false, false, false, true);
    fMatmulShaders(true, vMatmulIdType, false, false, true, true);

    if (vMatmulIdType != MatMulIdType::DEFAULT) {
#if defined(GGML_VULKAN_COOPMAT_GLSLC_SUPPORT)
      // Coopmat, fp32acc and fp16acc
      fMatmulShaders(true, vMatmulIdType, true, false, false);
      fMatmulShaders(true, vMatmulIdType, true, false, true);
#endif

#if defined(GGML_VULKAN_COOPMAT2_GLSLC_SUPPORT)
      // Coopmat2, fp32acc and fp16acc
      fMatmulShaders(true, vMatmulIdType, false, true, false);
      fMatmulShaders(true, vMatmulIdType, false, true, true);
#endif
    }
  }

  for (const bool &vFp16 : {false, true}) {
    std::map<std::string, std::string> vBaseDict;
    if (vFp16) {
      vBaseDict = {{"FLOAT_TYPE", "float16_t"},
                   {"FLOAT_TYPEV2", "f16vec2"},
                   {"FLOAT_TYPEV4", "f16vec4"},
                   {"FLOAT16", "1"},
                   {"FLOAT_TYPE_MAX", "float16_t(65504.0)"}};
    } else {
      vBaseDict = {{"FLOAT_TYPE", "float"}, {"FLOAT_TYPEV2", "vec2"}, {"FLOAT_TYPEV4", "vec4"}};
    }

    // flash attention
    for (const bool &vF16acc : {false, true}) {
      std::map<std::string, std::string> vFaBaseDict = vBaseDict;
      vFaBaseDict["ACC_TYPE"] = vFp16 && vF16acc ? "float16_t" : "float";
      vFaBaseDict["ACC_TYPEV2"] = vFp16 && vF16acc ? "f16vec2" : "vec2";
      vFaBaseDict["ACC_TYPEV4"] = vFp16 && vF16acc ? "f16vec4" : "vec4";
      // Compile IQ4_NL support into all FA variants so its shared LUT is available when K or V uses it.
      vFaBaseDict["DATA_A_IQ4_NL"] = "1";
      if (vFp16 && vF16acc) {
        vFaBaseDict["ACC_TYPE_MAX"] = "float16_t(65504.0)";
      }

      if (vFp16) {
#if defined(GGML_VULKAN_COOPMAT2_GLSLC_SUPPORT)
        fStringToSpv("flash_attn_f32_f16", "flash_attn_cm2.comp",
                     fMergeMaps(vFaBaseDict, {{"Q_TYPE", "float"}, {"D_TYPE", "float"}, {"D_TYPEV4", "vec4"}}), vFp16,
                     false, true, vF16acc);
#endif

#if defined(GGML_VULKAN_COOPMAT_GLSLC_SUPPORT)
        fStringToSpv(
            "flash_attn_f32_f16", "flash_attn_cm1.comp",
            fMergeMaps(vFaBaseDict, {{"Q_TYPE", "float"}, {"D_TYPE", "float"}, {"D_TYPEV4", "vec4"}, {"COOPMAT", "1"}}),
            vFp16, true, false, vF16acc);
#endif
      }

      fStringToSpv("flash_attn_f32_f16", "flash_attn.comp",
                   fMergeMaps(vFaBaseDict, {{"Q_TYPE", "float"}, {"D_TYPE", "float"}, {"D_TYPEV4", "vec4"}}), vFp16,
                   false, false, vF16acc);

      if (vFp16) {
        fStringToSpv("flash_attn_f32_f16_dot2", "flash_attn.comp",
                     fMergeMaps(vFaBaseDict,
                                {{"Q_TYPE", "float"}, {"D_TYPE", "float"}, {"D_TYPEV4", "vec4"}, {"DOT2_F16", "1"}}),
                     vFp16, false, false, vF16acc);
      }

#if defined(GGML_VULKAN_INTEGER_DOT_GLSLC_SUPPORT)
      fStringToSpv(
          "flash_attn_f32_f16", "flash_attn.comp",
          fMergeMaps(
              vFaBaseDict,
              {{"Q_TYPE", "float"}, {"D_TYPE", "float"}, {"D_TYPEV4", "vec4"}, {"MMQ", "1"}, {"FA_MMQ_MIXED", "1"}}),
          vFp16, false, false, vF16acc, "_int8");
#endif
    }
  }

  const std::map<std::string, std::string> cFaBf16Dict = {
      {"FLOAT_TYPE", "bfloat16_t"},
      {"FLOAT_TYPEV2", "bf16vec2"},
      {"FLOAT_TYPEV4", "bf16vec4"},
      {"ACC_TYPE", "float"},
      {"ACC_TYPEV2", "vec2"},
      {"ACC_TYPEV4", "vec4"},
      {"BFLOAT16", "1"},
  };

#if defined(GGML_VULKAN_BFLOAT16_GLSLC_SUPPORT) && defined(GGML_VULKAN_COOPMAT_GLSLC_SUPPORT)
  fStringToSpv(
      "flash_attn_f32_f16_bf16", "flash_attn_cm1.comp",
      fMergeMaps(cFaBf16Dict, {{"Q_TYPE", "float"}, {"D_TYPE", "float"}, {"D_TYPEV4", "vec4"}, {"COOPMAT", "1"}}), true,
      true, false, false);
#endif

#if defined(GGML_VULKAN_BFLOAT16_GLSLC_SUPPORT) && defined(GGML_VULKAN_COOPMAT2_GLSLC_SUPPORT)
  fStringToSpv("flash_attn_f32_f16_bf16", "flash_attn_cm2.comp",
               fMergeMaps(cFaBf16Dict, {{"Q_TYPE", "float"}, {"D_TYPE", "float"}, {"D_TYPEV4", "vec4"}}), true, false,
               true, false);
#endif

  std::map<std::string, std::string> vBaseDict = {{"FLOAT_TYPE", "float"}, {"FLOAT_TYPEV2", "vec2"}};

  for (const auto &vTname : type_names) {
    // mul mat vec
    std::string vDataAKey = "DATA_A_" + fToUppercase(vTname);
    std::string vShader = (fStringEndsWith(vTname, "_k") || fStringStartsWith(vTname, "iq1_") ||
                           fStringStartsWith(vTname, "iq2_") || fStringStartsWith(vTname, "iq3_"))
                              ? "mul_mat_vec_" + vTname + ".comp"
                              : "mul_mat_vec.comp";

    fStringToSpv(
        "mul_mat_vec_" + vTname + "_f32_f32", vShader,
        fMergeMaps(
            vBaseDict,
            {{vDataAKey, "1"}, {"B_TYPE", "float"}, {"B_TYPEV2", "vec2"}, {"B_TYPEV4", "vec4"}, {"D_TYPE", "float"}}));
    fStringToSpv("mul_mat_vec_" + vTname + "_f16_f32", vShader,
                 fMergeMaps(vBaseDict, {{vDataAKey, "1"},
                                        {"B_TYPE", "float16_t"},
                                        {"B_TYPEV2", "f16vec2"},
                                        {"B_TYPEV4", "f16vec4"},
                                        {"D_TYPE", "float"}}));

    fStringToSpv("mul_mat_vec_" + vTname + "_f32_f32_subgroup", vShader,
                 fMergeMaps(vBaseDict, {{vDataAKey, "1"},
                                        {"B_TYPE", "float"},
                                        {"B_TYPEV2", "vec2"},
                                        {"B_TYPEV4", "vec4"},
                                        {"D_TYPE", "float"},
                                        {"USE_SUBGROUP_ADD", "1"}}));
    fStringToSpv("mul_mat_vec_" + vTname + "_f16_f32_subgroup", vShader,
                 fMergeMaps(vBaseDict, {{vDataAKey, "1"},
                                        {"B_TYPE", "float16_t"},
                                        {"B_TYPEV2", "f16vec2"},
                                        {"B_TYPEV4", "f16vec4"},
                                        {"D_TYPE", "float"},
                                        {"USE_SUBGROUP_ADD", "1"}}));

    fStringToSpv("mul_mat_vec_" + vTname + "_f32_f32_subgroup_no_shmem", vShader,
                 fMergeMaps(vBaseDict, {{vDataAKey, "1"},
                                        {"B_TYPE", "float"},
                                        {"B_TYPEV2", "vec2"},
                                        {"B_TYPEV4", "vec4"},
                                        {"D_TYPE", "float"},
                                        {"USE_SUBGROUP_ADD_NO_SHMEM", "1"}}));
    fStringToSpv("mul_mat_vec_" + vTname + "_f16_f32_subgroup_no_shmem", vShader,
                 fMergeMaps(vBaseDict, {{vDataAKey, "1"},
                                        {"B_TYPE", "float16_t"},
                                        {"B_TYPEV2", "f16vec2"},
                                        {"B_TYPEV4", "f16vec4"},
                                        {"D_TYPE", "float"},
                                        {"USE_SUBGROUP_ADD_NO_SHMEM", "1"}}));

#if defined(GGML_VULKAN_FLOAT_E2M1_GLSLC_SUPPORT) && defined(GGML_VULKAN_FLOAT_E4M3_GLSLC_SUPPORT)
    if (vTname == "mxfp4" || vTname == "nvfp4") {
      fStringToSpv("mul_mat_vec_" + vTname + "_f32_f32_ocp", vShader,
                   fMergeMaps(vBaseDict, {{vDataAKey, "1"},
                                          {"USE_OCP_FP4", "1"},
                                          {"B_TYPE", "float"},
                                          {"B_TYPEV2", "vec2"},
                                          {"B_TYPEV4", "vec4"},
                                          {"D_TYPE", "float"}}));
      fStringToSpv("mul_mat_vec_" + vTname + "_f16_f32_ocp", vShader,
                   fMergeMaps(vBaseDict, {{vDataAKey, "1"},
                                          {"USE_OCP_FP4", "1"},
                                          {"B_TYPE", "float16_t"},
                                          {"B_TYPEV2", "f16vec2"},
                                          {"B_TYPEV4", "f16vec4"},
                                          {"D_TYPE", "float"}}));
      fStringToSpv("mul_mat_vec_" + vTname + "_f32_f32_ocp_subgroup", vShader,
                   fMergeMaps(vBaseDict, {{vDataAKey, "1"},
                                          {"USE_OCP_FP4", "1"},
                                          {"B_TYPE", "float"},
                                          {"B_TYPEV2", "vec2"},
                                          {"B_TYPEV4", "vec4"},
                                          {"D_TYPE", "float"},
                                          {"USE_SUBGROUP_ADD", "1"}}));
      fStringToSpv("mul_mat_vec_" + vTname + "_f16_f32_ocp_subgroup", vShader,
                   fMergeMaps(vBaseDict, {{vDataAKey, "1"},
                                          {"USE_OCP_FP4", "1"},
                                          {"B_TYPE", "float16_t"},
                                          {"B_TYPEV2", "f16vec2"},
                                          {"B_TYPEV4", "f16vec4"},
                                          {"D_TYPE", "float"},
                                          {"USE_SUBGROUP_ADD", "1"}}));
      fStringToSpv("mul_mat_vec_" + vTname + "_f32_f32_ocp_subgroup_no_shmem", vShader,
                   fMergeMaps(vBaseDict, {{vDataAKey, "1"},
                                          {"USE_OCP_FP4", "1"},
                                          {"B_TYPE", "float"},
                                          {"B_TYPEV2", "vec2"},
                                          {"B_TYPEV4", "vec4"},
                                          {"D_TYPE", "float"},
                                          {"USE_SUBGROUP_ADD_NO_SHMEM", "1"}}));
      fStringToSpv("mul_mat_vec_" + vTname + "_f16_f32_ocp_subgroup_no_shmem", vShader,
                   fMergeMaps(vBaseDict, {{vDataAKey, "1"},
                                          {"USE_OCP_FP4", "1"},
                                          {"B_TYPE", "float16_t"},
                                          {"B_TYPEV2", "f16vec2"},
                                          {"B_TYPEV4", "f16vec4"},
                                          {"D_TYPE", "float"},
                                          {"USE_SUBGROUP_ADD_NO_SHMEM", "1"}}));
      fStringToSpv("mul_mat_vec_id_" + vTname + "_f32_f32_ocp", vShader,
                   fMergeMaps(vBaseDict, {{"MUL_MAT_ID", "1"},
                                          {vDataAKey, "1"},
                                          {"USE_OCP_FP4", "1"},
                                          {"B_TYPE", "float"},
                                          {"B_TYPEV2", "vec2"},
                                          {"B_TYPEV4", "vec4"},
                                          {"D_TYPE", "float"}}));
      fStringToSpv("mul_mat_vec_id_" + vTname + "_f32_f32_ocp_subgroup", vShader,
                   fMergeMaps(vBaseDict, {{"MUL_MAT_ID", "1"},
                                          {vDataAKey, "1"},
                                          {"USE_OCP_FP4", "1"},
                                          {"B_TYPE", "float"},
                                          {"B_TYPEV2", "vec2"},
                                          {"B_TYPEV4", "vec4"},
                                          {"D_TYPE", "float"},
                                          {"USE_SUBGROUP_ADD", "1"}}));
      fStringToSpv("mul_mat_vec_id_" + vTname + "_f32_f32_ocp_subgroup_no_shmem", vShader,
                   fMergeMaps(vBaseDict, {{"MUL_MAT_ID", "1"},
                                          {vDataAKey, "1"},
                                          {"USE_OCP_FP4", "1"},
                                          {"B_TYPE", "float"},
                                          {"B_TYPEV2", "vec2"},
                                          {"B_TYPEV4", "vec4"},
                                          {"D_TYPE", "float"},
                                          {"USE_SUBGROUP_ADD_NO_SHMEM", "1"}}));
    }
#endif

    fStringToSpv("mul_mat_vec_id_" + vTname + "_f32_f32", vShader,
                 fMergeMaps(vBaseDict, {{"MUL_MAT_ID", "1"},
                                        {vDataAKey, "1"},
                                        {"B_TYPE", "float"},
                                        {"B_TYPEV2", "vec2"},
                                        {"B_TYPEV4", "vec4"},
                                        {"D_TYPE", "float"}}));
    fStringToSpv("mul_mat_vec_id_" + vTname + "_f32_f32_subgroup", vShader,
                 fMergeMaps(vBaseDict, {{"MUL_MAT_ID", "1"},
                                        {vDataAKey, "1"},
                                        {"B_TYPE", "float"},
                                        {"B_TYPEV2", "vec2"},
                                        {"B_TYPEV4", "vec4"},
                                        {"D_TYPE", "float"},
                                        {"USE_SUBGROUP_ADD", "1"}}));
    fStringToSpv("mul_mat_vec_id_" + vTname + "_f32_f32_subgroup_no_shmem", vShader,
                 fMergeMaps(vBaseDict, {{"MUL_MAT_ID", "1"},
                                        {vDataAKey, "1"},
                                        {"B_TYPE", "float"},
                                        {"B_TYPEV2", "vec2"},
                                        {"B_TYPEV4", "vec4"},
                                        {"D_TYPE", "float"},
                                        {"USE_SUBGROUP_ADD_NO_SHMEM", "1"}}));

    // mul mat vec with integer dot product
#if defined(GGML_VULKAN_INTEGER_DOT_GLSLC_SUPPORT)
    if (fIsLegacyQuant(vTname) || vTname == "mxfp4" || fIsKQuant(vTname) || vTname == "iq1_s" || vTname == "iq1_m") {
      fStringToSpv("mul_mat_vec_" + vTname + "_q8_1_f32", "mul_mat_vecq.comp",
                   fMergeMaps(vBaseDict, {{vDataAKey, "1"},
                                          {"D_TYPE", "float"},
                                          {"FLOAT_TYPE", "float"},
                                          {"FLOAT_TYPEV2", "vec2"},
                                          {"ACC_TYPE", "float"}}));
      fStringToSpv("mul_mat_vec_" + vTname + "_q8_1_f32_subgroup", "mul_mat_vecq.comp",
                   fMergeMaps(vBaseDict, {{vDataAKey, "1"},
                                          {"D_TYPE", "float"},
                                          {"FLOAT_TYPE", "float"},
                                          {"FLOAT_TYPEV2", "vec2"},
                                          {"ACC_TYPE", "float"},
                                          {"USE_SUBGROUP_ADD", "1"}}));
      fStringToSpv("mul_mat_vec_" + vTname + "_q8_1_f32_subgroup_no_shmem", "mul_mat_vecq.comp",
                   fMergeMaps(vBaseDict, {{vDataAKey, "1"},
                                          {"D_TYPE", "float"},
                                          {"FLOAT_TYPE", "float"},
                                          {"FLOAT_TYPEV2", "vec2"},
                                          {"ACC_TYPE", "float"},
                                          {"USE_SUBGROUP_ADD_NO_SHMEM", "1"}}));

      fStringToSpv("mul_mat_vec_id_" + vTname + "_q8_1_f32", "mul_mat_vecq.comp",
                   fMergeMaps(vBaseDict, {{"MUL_MAT_ID", "1"},
                                          {vDataAKey, "1"},
                                          {"D_TYPE", "float"},
                                          {"FLOAT_TYPE", "float"},
                                          {"FLOAT_TYPEV2", "vec2"},
                                          {"ACC_TYPE", "float"}}));
      fStringToSpv("mul_mat_vec_id_" + vTname + "_q8_1_f32_subgroup", "mul_mat_vecq.comp",
                   fMergeMaps(vBaseDict, {{"MUL_MAT_ID", "1"},
                                          {vDataAKey, "1"},
                                          {"D_TYPE", "float"},
                                          {"FLOAT_TYPE", "float"},
                                          {"FLOAT_TYPEV2", "vec2"},
                                          {"ACC_TYPE", "float"},
                                          {"USE_SUBGROUP_ADD", "1"}}));
      fStringToSpv("mul_mat_vec_id_" + vTname + "_q8_1_f32_subgroup_no_shmem", "mul_mat_vecq.comp",
                   fMergeMaps(vBaseDict, {{"MUL_MAT_ID", "1"},
                                          {vDataAKey, "1"},
                                          {"D_TYPE", "float"},
                                          {"FLOAT_TYPE", "float"},
                                          {"FLOAT_TYPEV2", "vec2"},
                                          {"ACC_TYPE", "float"},
                                          {"USE_SUBGROUP_ADD_NO_SHMEM", "1"}}));
    }
#endif

    // Dequant shaders
    if (vTname != "f16" && vTname != "bf16") {
      fStringToSpv("dequant_" + vTname, "dequant_" + vTname + ".comp",
                   fMergeMaps(vBaseDict, {{vDataAKey, "1"}, {"D_TYPE", "float16_t"}}));
    }

    vShader = (vTname == "f32" || vTname == "f16" || vTname == "bf16") ? "get_rows.comp" : "get_rows_quant.comp";

    if (vTname == "f16") {
      fStringToSpv("get_rows_" + vTname, vShader,
                   fMergeMaps(vBaseDict, {{"TEMP_TYPE", "FLOAT_TYPE"},
                                          {vDataAKey, "1"},
                                          {"B_TYPE", "int"},
                                          {"D_TYPE", "float16_t"},
                                          {"OPTIMIZATION_ERROR_WORKAROUND", "1"}}));
    } else {
      fStringToSpv(
          "get_rows_" + vTname, vShader,
          fMergeMaps(vBaseDict,
                     {{"TEMP_TYPE", "FLOAT_TYPE"}, {vDataAKey, "1"}, {"B_TYPE", "int"}, {"D_TYPE", "float16_t"}}));
    }
    fStringToSpv(
        "get_rows_" + vTname + "_f32", vShader,
        fMergeMaps(vBaseDict, {{"TEMP_TYPE", "FLOAT_TYPE"}, {vDataAKey, "1"}, {"B_TYPE", "int"}, {"D_TYPE", "float"}}));
  }

  fStringToSpv("get_rows_i32", "get_rows.comp",
               {{"TEMP_TYPE", "uint"}, {"A_TYPE", "uint"}, {"B_TYPE", "int"}, {"D_TYPE", "uint"}});

  fStringToSpv("mul_mat_vec_p021_f16_f32_subgroup_add", "mul_mat_vec_p021.comp",
               {{"A_TYPE", "float16_t"},
                {"A_TYPEV4", "f16vec4"},
                {"B_TYPE", "float"},
                {"B_TYPEV4", "vec4"},
                {"D_TYPE", "float"},
                {"USE_SUBGROUP_ADD", "1"}});
  fStringToSpv("mul_mat_vec_p021_f16_f32", "mul_mat_vec_p021.comp",
               {{"A_TYPE", "float16_t"},
                {"A_TYPEV4", "f16vec4"},
                {"B_TYPE", "float"},
                {"B_TYPEV4", "vec4"},
                {"D_TYPE", "float"}});
  fStringToSpv("mul_mat_vec_nc_f16_f32", "mul_mat_vec_nc.comp",
               {{"A_TYPE", "float16_t"},
                {"A_TYPEV4", "f16vec4"},
                {"B_TYPE", "float"},
                {"B_TYPEV4", "vec4"},
                {"D_TYPE", "float"}});

  // Norms
  fStringToSpv("norm_f32", "norm.comp", fMergeMaps(vBaseDict, {{"A_TYPE", "float"}, {"D_TYPE", "float"}}));
  fStringToSpv("group_norm_f32", "group_norm.comp", fMergeMaps(vBaseDict, {{"A_TYPE", "float"}, {"D_TYPE", "float"}}));
  // the row norms reduce through shared memory or, with the _subgroup variants, through subgroup additions
  for (bool vUseSubgroups : {false, true}) {
    const std::string cSuffix = vUseSubgroups ? "_subgroup" : "";
    const std::string cSubgroupAdd = vUseSubgroups ? "1" : "0";
    fStringToSpv("rms_norm_f32" + cSuffix, "rms_norm.comp",
                 fMergeMaps(vBaseDict, {{"A_TYPE", "float"},
                                        {"B_TYPE", "float"},
                                        {"D_TYPE", "float"},
                                        {"USE_SUBGROUP_ADD", cSubgroupAdd}}));
    fStringToSpv("rms_norm_mul_rope_f32_f32" + cSuffix, "rms_norm.comp",
                 fMergeMaps(vBaseDict, {{"A_TYPE", "float"},
                                        {"B_TYPE", "float"},
                                        {"D_TYPE", "float"},
                                        {"ROPE_D_TYPE", "float"},
                                        {"RMS_NORM_ROPE_FUSION", "1"},
                                        {"USE_SUBGROUP_ADD", cSubgroupAdd}}));
    fStringToSpv("rms_norm_mul_rope_f32_f16" + cSuffix, "rms_norm.comp",
                 fMergeMaps(vBaseDict, {{"A_TYPE", "float"},
                                        {"B_TYPE", "float"},
                                        {"D_TYPE", "float"},
                                        {"ROPE_D_TYPE", "float16_t"},
                                        {"RMS_NORM_ROPE_FUSION", "1"},
                                        {"USE_SUBGROUP_ADD", cSubgroupAdd}}));
    fStringToSpv("l2_norm_f32" + cSuffix, "l2_norm.comp",
                 fMergeMaps(vBaseDict, {{"A_TYPE", "float"}, {"D_TYPE", "float"}, {"USE_SUBGROUP_ADD", cSubgroupAdd}}));
  }
  fStringToSpv("rms_norm_partials_f32", "rms_norm_partials.comp",
               fMergeMaps(vBaseDict, {{"A_TYPE", "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}}));
  fStringToSpv("rms_norm_back_f32", "rms_norm_back.comp",
               fMergeMaps(vBaseDict, {{"A_TYPE", "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}}));

  fStringToSpv("cpy_f32_f32", "copy.comp", {{"A_TYPE", "float"}, {"D_TYPE", "float"}});
  fStringToSpv("cpy_f32_f16", "copy.comp", {{"A_TYPE", "float"}, {"D_TYPE", "float16_t"}});
  fStringToSpv("cpy_f16_f16", "copy.comp",
               {{"A_TYPE", "float16_t"}, {"D_TYPE", "float16_t"}, {"OPTIMIZATION_ERROR_WORKAROUND", "1"}});
  fStringToSpv("cpy_f16_f32", "copy.comp",
               {{"A_TYPE", "float16_t"}, {"D_TYPE", "float"}, {"OPTIMIZATION_ERROR_WORKAROUND", "1"}});
  fStringToSpv("cpy_f32_bf16", "copy.comp", {{"A_TYPE", "float"}, {"D_TYPE", "uint16_t"}, {"DATA_D_BF16", "1"}});
  fStringToSpv("cpy_bf16_f32", "copy.comp", {{"A_TYPE", "uint16_t"}, {"D_TYPE", "float"}, {"DATA_A_BF16", "1"}});
  fStringToSpv("contig_cpy_f32_f32", "contig_copy.comp", {{"A_TYPE", "float"}, {"D_TYPE", "float"}});
  fStringToSpv("contig_cpy_f32_i32", "contig_copy.comp", {{"A_TYPE", "float"}, {"D_TYPE", "int"}});
  fStringToSpv("contig_cpy_i32_f32", "contig_copy.comp", {{"A_TYPE", "int"}, {"D_TYPE", "float"}});
  fStringToSpv("contig_cpy_f32_f16", "contig_copy.comp", {{"A_TYPE", "float"}, {"D_TYPE", "float16_t"}});
  fStringToSpv("contig_cpy_f16_f16", "contig_copy.comp",
               {{"A_TYPE", "float16_t"}, {"D_TYPE", "float16_t"}, {"OPTIMIZATION_ERROR_WORKAROUND", "1"}});
  fStringToSpv("contig_cpy_f16_f32", "contig_copy.comp",
               {{"A_TYPE", "float16_t"}, {"D_TYPE", "float"}, {"OPTIMIZATION_ERROR_WORKAROUND", "1"}});
  fStringToSpv("contig_cpy_f32_bf16", "contig_copy.comp",
               {{"A_TYPE", "float"}, {"D_TYPE", "uint16_t"}, {"DATA_D_BF16", "1"}});
  fStringToSpv("contig_cpy_bf16_f32", "contig_copy.comp",
               {{"A_TYPE", "uint16_t"}, {"D_TYPE", "float"}, {"DATA_A_BF16", "1"}});
  fStringToSpv("cpy_f32_i32", "copy.comp", {{"A_TYPE", "float"}, {"D_TYPE", "int"}});
  fStringToSpv("cpy_i32_f32", "copy.comp", {{"A_TYPE", "int"}, {"D_TYPE", "float"}});

  fStringToSpv("cpy_transpose_16", "copy_transpose.comp", {{"A_TYPE", "uint16_t"}, {"D_TYPE", "uint16_t"}});
  fStringToSpv("cpy_transpose_32", "copy_transpose.comp", {{"A_TYPE", "uint"}, {"D_TYPE", "uint"}});

  for (std::string vT : {"q1_0", "q2_0", "q4_0", "q4_1", "q5_0", "q5_1", "q8_0", "iq4_nl"}) {
    fStringToSpv(
        "cpy_f32_" + vT, "copy_to_quant.comp",
        {{"DATA_A_" + fToUppercase(vT), "1"}, {"S_TYPE", "float"}, {"D_TYPE", "float"}, {"FLOAT_TYPE", "float"}});
    fStringToSpv("cpy_" + vT + "_f32", "copy_from_quant.comp",
                 {{"DATA_A_" + fToUppercase(vT), "1"}, {"D_TYPE", "float"}, {"FLOAT_TYPE", "float"}});
  }

  for (auto vSrc : {std::pair{"f32", "float"}, std::pair{"f16", "float16_t"}}) {
    for (std::string vDst : {"f32", "f16", "bf16", "q1_0", "q2_0", "q4_0", "q4_1", "q5_0", "q5_1", "q8_0", "iq4_nl"}) {
      fStringToSpv("set_rows_" + std::string(vSrc.first) + "_" + vDst + "_i32", "copy_to_quant.comp",
                   {{"SET_ROWS", "1"},
                    {"DATA_A_" + fToUppercase(vDst), "1"},
                    {"B_TYPE", "uint"},
                    {"B_SIZE", "32"},
                    {"S_TYPE", vSrc.second},
                    {"D_TYPE", "float"},
                    {"FLOAT_TYPE", "float"}});
      fStringToSpv("set_rows_" + std::string(vSrc.first) + "_" + vDst + "_i64", "copy_to_quant.comp",
                   {{"SET_ROWS", "1"},
                    {"DATA_A_" + fToUppercase(vDst), "1"},
                    {"B_TYPE", "uvec2"},
                    {"B_SIZE", "64"},
                    {"S_TYPE", vSrc.second},
                    {"D_TYPE", "float"},
                    {"FLOAT_TYPE", "float"}});
    }
  }

  auto fGetTypeStr = [](bool pF16) { return pF16 ? "float16_t" : "float"; };
  auto fGetSuffix = [](bool pSrc0F16, bool pSrc1F16, bool pDstF16) {
    std::string vS;
    vS += std::string(pSrc0F16 ? "_f16" : "_f32");
    vS += std::string(pSrc1F16 ? "_f16" : "_f32");
    vS += std::string(pDstF16 ? "_f16" : "_f32");
    return vS;
  };
  for (std::string vOp : {
           "add",
           "sub",
           "mul",
           "div",
           "add_rms",
       }) {
    for (auto vSrc0F16 : {false, true}) {
      for (auto vSrc1F16 : {false, true}) {
        for (auto vDstF16 : {false, true}) {
          auto vSource = vOp == "add_rms" ? std::string("add") : vOp;
          auto vName = vOp + fGetSuffix(vSrc0F16, vSrc1F16, vDstF16);
          auto vAddRms = vOp == "add_rms" ? "1" : "0";
          fStringToSpv(vName.c_str(), vSource + ".comp",
                       {{"A_TYPE", fGetTypeStr(vSrc0F16)},
                        {"B_TYPE", fGetTypeStr(vSrc1F16)},
                        {"D_TYPE", fGetTypeStr(vDstF16)},
                        {"FLOAT_TYPE", "float"},
                        {"ADD_RMS", vAddRms}});
        }
      }
    }
  }

  fStringToSpv("sub_f32", "sub.comp",
               {{"A_TYPE", "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}, {"FLOAT_TYPE", "float"}});

  fStringToSpv("acc_f32", "acc.comp",
               {{"A_TYPE", "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}, {"FLOAT_TYPE", "float"}});

  fStringToSpv("split_k_reduce", "mul_mat_split_k_reduce.comp", {});
  fStringToSpv("fa_split_k_reduce", "flash_attn_split_k_reduce.comp", {});

  fStringToSpv("fa_mask_opt", "flash_attn_mask_opt.comp", {});

  fStringToSpv("quantize_q8_1", "quantize_q8_1.comp", {});
  fStringToSpv("quantize_q8_1_subgroup", "quantize_q8_1.comp", {{"USE_SUBGROUPS", "1"}});

  fStringToSpv("quantize_q8_1_x4", "quantize_q8_1.comp", {{"QBLOCK_X4", "1"}});
  fStringToSpv("quantize_q8_1_x4_subgroup", "quantize_q8_1.comp", {{"QBLOCK_X4", "1"}, {"USE_SUBGROUPS", "1"}});

  fStringToSpv("mul_f32", "mul.comp",
               {{"A_TYPE", "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}, {"FLOAT_TYPE", "float"}});

  fStringToSpv("div_f32", "div.comp",
               {{"A_TYPE", "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}, {"FLOAT_TYPE", "float"}});

  fStringToSpv("repeat_i32", "repeat.comp", {{"A_TYPE", "int32_t"}, {"D_TYPE", "int32_t"}});
  fStringToSpv("repeat_back_f32", "repeat_back.comp", {{"A_TYPE", "float"}, {"D_TYPE", "float"}});
  fStringToSpv("get_rows_back_f32", "get_rows_back.comp",
               {{"A_TYPE", "float"}, {"B_TYPE", "int"}, {"D_TYPE", "float"}});

  fStringToSpv("repeat_i16", "repeat.comp", {{"A_TYPE", "int16_t"}, {"D_TYPE", "int16_t"}});

  fStringToSpv("scale_f32", "scale.comp", {{"A_TYPE", "float"}, {"D_TYPE", "float"}, {"FLOAT_TYPE", "float"}});

  fStringToSpv("pad_f32", "pad.comp", {{"A_TYPE", "float"}, {"D_TYPE", "float"}});

  fStringToSpv("concat_i8", "concat.comp", {{"A_TYPE", "uint8_t"}, {"B_TYPE", "uint8_t"}, {"D_TYPE", "uint8_t"}});
  fStringToSpv("concat_i16", "concat.comp", {{"A_TYPE", "uint16_t"}, {"B_TYPE", "uint16_t"}, {"D_TYPE", "uint16_t"}});
  fStringToSpv("concat_i32", "concat.comp", {{"A_TYPE", "uint"}, {"B_TYPE", "uint"}, {"D_TYPE", "uint"}});
  fStringToSpv("concat_i64", "concat.comp", {{"A_TYPE", "uvec2"}, {"B_TYPE", "uvec2"}, {"D_TYPE", "uvec2"}});

  fStringToSpv("upscale_f32", "upscale.comp", {{"A_TYPE", "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}});

  fStringToSpv("exp_f16", "unary.comp", {{"A_TYPE", "float16_t"}, {"D_TYPE", "float16_t"}, {"OP", "op_exp"}});
  fStringToSpv("exp_f32", "unary.comp", {{"A_TYPE", "float"}, {"D_TYPE", "float"}, {"OP", "op_exp"}});
  fStringToSpv("expm1_f16", "unary.comp", {{"A_TYPE", "float16_t"}, {"D_TYPE", "float16_t"}, {"OP", "op_expm1"}});
  fStringToSpv("expm1_f32", "unary.comp", {{"A_TYPE", "float"}, {"D_TYPE", "float"}, {"OP", "op_expm1"}});

  fStringToSpv("log_f16", "log.comp", {{"A_TYPE", "float16_t"}, {"D_TYPE", "float16_t"}});
  fStringToSpv("log_f32", "log.comp", {{"A_TYPE", "float"}, {"D_TYPE", "float"}});
  fStringToSpv("gelu_f16", "unary.comp", {{"A_TYPE", "float16_t"}, {"D_TYPE", "float16_t"}, {"OP", "op_gelu"}});
  fStringToSpv("gelu_f32", "unary.comp", {{"A_TYPE", "float"}, {"D_TYPE", "float"}, {"OP", "op_gelu"}});
  fStringToSpv("gelu_erf_f16", "unary.comp", {{"A_TYPE", "float16_t"}, {"D_TYPE", "float16_t"}, {"OP", "op_gelu_erf"}});
  fStringToSpv("gelu_erf_f32", "unary.comp", {{"A_TYPE", "float"}, {"D_TYPE", "float"}, {"OP", "op_gelu_erf"}});
  fStringToSpv("gelu_quick_f16", "unary.comp",
               {{"A_TYPE", "float16_t"}, {"D_TYPE", "float16_t"}, {"OP", "op_gelu_quick"}});
  fStringToSpv("gelu_quick_f32", "unary.comp", {{"A_TYPE", "float"}, {"D_TYPE", "float"}, {"OP", "op_gelu_quick"}});
  fStringToSpv("silu_f16", "unary.comp", {{"A_TYPE", "float16_t"}, {"D_TYPE", "float16_t"}, {"OP", "op_silu"}});
  fStringToSpv("silu_f32", "unary.comp", {{"A_TYPE", "float"}, {"D_TYPE", "float"}, {"OP", "op_silu"}});
  fStringToSpv("relu_f16", "unary.comp", {{"A_TYPE", "float16_t"}, {"D_TYPE", "float16_t"}, {"OP", "op_relu"}});
  fStringToSpv("relu_f32", "unary.comp", {{"A_TYPE", "float"}, {"D_TYPE", "float"}, {"OP", "op_relu"}});
  fStringToSpv("sqr_f16", "unary.comp", {{"A_TYPE", "float16_t"}, {"D_TYPE", "float16_t"}, {"OP", "op_sqr"}});
  fStringToSpv("sqr_f32", "unary.comp", {{"A_TYPE", "float"}, {"D_TYPE", "float"}, {"OP", "op_sqr"}});
  fStringToSpv("sqrt_f16", "unary.comp", {{"A_TYPE", "float16_t"}, {"D_TYPE", "float16_t"}, {"OP", "op_sqrt"}});
  fStringToSpv("sqrt_f32", "unary.comp", {{"A_TYPE", "float"}, {"D_TYPE", "float"}, {"OP", "op_sqrt"}});
  fStringToSpv("sin_f16", "unary.comp", {{"A_TYPE", "float16_t"}, {"D_TYPE", "float16_t"}, {"OP", "op_sin"}});
  fStringToSpv("sin_f32", "unary.comp", {{"A_TYPE", "float"}, {"D_TYPE", "float"}, {"OP", "op_sin"}});
  fStringToSpv("cos_f16", "unary.comp", {{"A_TYPE", "float16_t"}, {"D_TYPE", "float16_t"}, {"OP", "op_cos"}});
  fStringToSpv("cos_f32", "unary.comp", {{"A_TYPE", "float"}, {"D_TYPE", "float"}, {"OP", "op_cos"}});
  fStringToSpv("clamp_f16", "unary.comp", {{"A_TYPE", "float16_t"}, {"D_TYPE", "float16_t"}, {"OP", "op_clamp"}});
  fStringToSpv("clamp_f32", "unary.comp", {{"A_TYPE", "float"}, {"D_TYPE", "float"}, {"OP", "op_clamp"}});
  fStringToSpv("leaky_relu_f16", "unary.comp",
               {{"A_TYPE", "float16_t"}, {"D_TYPE", "float16_t"}, {"OP", "op_leaky_relu"}});
  fStringToSpv("leaky_relu_f32", "unary.comp", {{"A_TYPE", "float"}, {"D_TYPE", "float"}, {"OP", "op_leaky_relu"}});
  fStringToSpv("neg_f16", "unary.comp", {{"A_TYPE", "float16_t"}, {"D_TYPE", "float16_t"}, {"OP", "op_neg"}});
  fStringToSpv("neg_f32", "unary.comp", {{"A_TYPE", "float"}, {"D_TYPE", "float"}, {"OP", "op_neg"}});
  fStringToSpv("tanh_f16", "unary.comp", {{"A_TYPE", "float16_t"}, {"D_TYPE", "float16_t"}, {"OP", "op_tanh"}});
  fStringToSpv("tanh_f32", "unary.comp", {{"A_TYPE", "float"}, {"D_TYPE", "float"}, {"OP", "op_tanh"}});
  fStringToSpv("sigmoid_f16", "unary.comp", {{"A_TYPE", "float16_t"}, {"D_TYPE", "float16_t"}, {"OP", "op_sigmoid"}});
  fStringToSpv("sigmoid_f32", "unary.comp", {{"A_TYPE", "float"}, {"D_TYPE", "float"}, {"OP", "op_sigmoid"}});
  fStringToSpv("hardsigmoid_f16", "unary.comp",
               {{"A_TYPE", "float16_t"}, {"D_TYPE", "float16_t"}, {"OP", "op_hardsigmoid"}});
  fStringToSpv("hardsigmoid_f32", "unary.comp", {{"A_TYPE", "float"}, {"D_TYPE", "float"}, {"OP", "op_hardsigmoid"}});
  fStringToSpv("hardswish_f16", "unary.comp",
               {{"A_TYPE", "float16_t"}, {"D_TYPE", "float16_t"}, {"OP", "op_hardswish"}});
  fStringToSpv("hardswish_f32", "unary.comp", {{"A_TYPE", "float"}, {"D_TYPE", "float"}, {"OP", "op_hardswish"}});
  fStringToSpv("abs_f16", "unary.comp", {{"A_TYPE", "float16_t"}, {"D_TYPE", "float16_t"}, {"OP", "op_abs"}});
  fStringToSpv("abs_f32", "unary.comp", {{"A_TYPE", "float"}, {"D_TYPE", "float"}, {"OP", "op_abs"}});
  fStringToSpv("elu_f16", "unary.comp", {{"A_TYPE", "float16_t"}, {"D_TYPE", "float16_t"}, {"OP", "op_elu"}});
  fStringToSpv("elu_f32", "unary.comp", {{"A_TYPE", "float"}, {"D_TYPE", "float"}, {"OP", "op_elu"}});
  fStringToSpv("xielu_f16", "unary.comp", {{"A_TYPE", "float16_t"}, {"D_TYPE", "float16_t"}, {"OP", "op_xielu"}});
  fStringToSpv("xielu_f32", "unary.comp", {{"A_TYPE", "float"}, {"D_TYPE", "float"}, {"OP", "op_xielu"}});
  fStringToSpv("sgn_f16", "unary.comp", {{"A_TYPE", "float16_t"}, {"D_TYPE", "float16_t"}, {"OP", "op_sgn"}});
  fStringToSpv("sgn_f32", "unary.comp", {{"A_TYPE", "float"}, {"D_TYPE", "float"}, {"OP", "op_sgn"}});

  fStringToSpv("tri_f16", "tri.comp", {{"A_TYPE", "float16_t"}, {"D_TYPE", "float16_t"}});
  fStringToSpv("tri_f32", "tri.comp", {{"A_TYPE", "float"}, {"D_TYPE", "float"}});
  fStringToSpv("diag_f16", "diag.comp", {{"A_TYPE", "float16_t"}, {"D_TYPE", "float16_t"}});
  fStringToSpv("diag_f32", "diag.comp", {{"A_TYPE", "float"}, {"D_TYPE", "float"}});

  fStringToSpv("softplus_f16", "unary.comp", {{"A_TYPE", "float16_t"}, {"D_TYPE", "float16_t"}, {"OP", "op_softplus"}});
  fStringToSpv("softplus_f32", "unary.comp", {{"A_TYPE", "float"}, {"D_TYPE", "float"}, {"OP", "op_softplus"}});

  // a unary op fused with the MUL that consumes it (gates of gated delta net, shared experts and GLU-like blocks)
  for (const char *vOp : {"gelu", "sigmoid", "silu", "softplus"}) {
    const std::string cOp = vOp;
    fStringToSpv(cOp + "_mul_f32", "unary.comp",
                 {{"A_TYPE", "float"},
                  {"B_TYPE", "float"},
                  {"D_TYPE", "float"},
                  {"FLOAT_TYPE", "float"},
                  {"OP", "op_" + cOp},
                  {"UNARY_MUL_FUSION", "1"}});
    fStringToSpv(cOp + "_mul_f16", "unary.comp",
                 {{"A_TYPE", "float16_t"},
                  {"B_TYPE", "float16_t"},
                  {"D_TYPE", "float16_t"},
                  {"FLOAT_TYPE", "float"},
                  {"OP", "op_" + cOp},
                  {"UNARY_MUL_FUSION", "1"}});
  }

  fStringToSpv("add1_f16_f16", "add1.comp",
               {{"A_TYPE", "float16_t"}, {"B_TYPE", "float16_t"}, {"D_TYPE", "float16_t"}, {"FLOAT_TYPE", "float"}});
  fStringToSpv("add1_f16_f32", "add1.comp",
               {{"A_TYPE", "float16_t"}, {"B_TYPE", "float"}, {"D_TYPE", "float16_t"}, {"FLOAT_TYPE", "float"}});
  fStringToSpv("add1_f32_f32", "add1.comp",
               {{"A_TYPE", "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}, {"FLOAT_TYPE", "float"}});
  fStringToSpv("arange_f32", "arange.comp", {{"A_TYPE", "float"}, {"D_TYPE", "float"}, {"FLOAT_TYPE", "float"}});
  fStringToSpv("fill_f32", "fill.comp", {{"D_TYPE", "float"}, {"FLOAT_TYPE", "float"}});
  fStringToSpv("fill_f16", "fill.comp", {{"D_TYPE", "float16_t"}, {"FLOAT_TYPE", "float"}});
  fStringToSpv("step_f16", "unary.comp", {{"A_TYPE", "float16_t"}, {"D_TYPE", "float16_t"}, {"OP", "op_step"}});
  fStringToSpv("step_f32", "unary.comp", {{"A_TYPE", "float"}, {"D_TYPE", "float"}, {"OP", "op_step"}});
  fStringToSpv("round_f16", "unary.comp", {{"A_TYPE", "float16_t"}, {"D_TYPE", "float16_t"}, {"OP", "op_round"}});
  fStringToSpv("round_f32", "unary.comp", {{"A_TYPE", "float"}, {"D_TYPE", "float"}, {"OP", "op_round"}});
  fStringToSpv("ceil_f16", "unary.comp", {{"A_TYPE", "float16_t"}, {"D_TYPE", "float16_t"}, {"OP", "op_ceil"}});
  fStringToSpv("ceil_f32", "unary.comp", {{"A_TYPE", "float"}, {"D_TYPE", "float"}, {"OP", "op_ceil"}});
  fStringToSpv("floor_f16", "unary.comp", {{"A_TYPE", "float16_t"}, {"D_TYPE", "float16_t"}, {"OP", "op_floor"}});
  fStringToSpv("floor_f32", "unary.comp", {{"A_TYPE", "float"}, {"D_TYPE", "float"}, {"OP", "op_floor"}});
  fStringToSpv("trunc_f16", "unary.comp", {{"A_TYPE", "float16_t"}, {"D_TYPE", "float16_t"}, {"OP", "op_trunc"}});
  fStringToSpv("trunc_f32", "unary.comp", {{"A_TYPE", "float"}, {"D_TYPE", "float"}, {"OP", "op_trunc"}});

  fStringToSpv("geglu_f16", "geglu.comp", {{"A_TYPE", "float16_t"}, {"D_TYPE", "float16_t"}});
  fStringToSpv("geglu_f32", "geglu.comp", {{"A_TYPE", "float"}, {"D_TYPE", "float"}});
  fStringToSpv("reglu_f16", "reglu.comp", {{"A_TYPE", "float16_t"}, {"D_TYPE", "float16_t"}});
  fStringToSpv("reglu_f32", "reglu.comp", {{"A_TYPE", "float"}, {"D_TYPE", "float"}});
  fStringToSpv("swiglu_f16", "swiglu.comp", {{"A_TYPE", "float16_t"}, {"D_TYPE", "float16_t"}});
  fStringToSpv("swiglu_f32", "swiglu.comp", {{"A_TYPE", "float"}, {"D_TYPE", "float"}});
  fStringToSpv("swiglu_oai_f16", "swiglu_oai.comp", {{"A_TYPE", "float16_t"}, {"D_TYPE", "float16_t"}});
  fStringToSpv("swiglu_oai_f32", "swiglu_oai.comp", {{"A_TYPE", "float"}, {"D_TYPE", "float"}});
  fStringToSpv("swiglu_clamp_f16", "swiglu_clamp.comp", {{"A_TYPE", "float16_t"}, {"D_TYPE", "float16_t"}});
  fStringToSpv("swiglu_clamp_f32", "swiglu_clamp.comp", {{"A_TYPE", "float"}, {"D_TYPE", "float"}});
  fStringToSpv("geglu_erf_f16", "geglu_erf.comp", {{"A_TYPE", "float16_t"}, {"D_TYPE", "float16_t"}});
  fStringToSpv("geglu_erf_f32", "geglu_erf.comp", {{"A_TYPE", "float"}, {"D_TYPE", "float"}});
  fStringToSpv("geglu_quick_f16", "geglu_quick.comp", {{"A_TYPE", "float16_t"}, {"D_TYPE", "float16_t"}});
  fStringToSpv("geglu_quick_f32", "geglu_quick.comp", {{"A_TYPE", "float"}, {"D_TYPE", "float"}});

  fStringToSpv("silu_back_f32", "silu_back.comp", {{"A_TYPE", "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}});

  fStringToSpv("diag_mask_inf_f32", "diag_mask_inf.comp", {{"A_TYPE", "float"}, {"D_TYPE", "float"}});

  fStringToSpv("soft_max_f32", "soft_max.comp",
               fMergeMaps(vBaseDict, {{"A_TYPE", "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}}));
  fStringToSpv("soft_max_f32_f16", "soft_max.comp",
               fMergeMaps(vBaseDict, {{"A_TYPE", "float"}, {"B_TYPE", "float16_t"}, {"D_TYPE", "float"}}));
  fStringToSpv("soft_max_back_f32", "soft_max_back.comp",
               fMergeMaps(vBaseDict, {{"A_TYPE", "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}}));

  fStringToSpv("soft_max_large1_f32", "soft_max_large1.comp",
               fMergeMaps(vBaseDict, {{"A_TYPE", "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}}));
  fStringToSpv("soft_max_large2_f32", "soft_max_large2.comp",
               fMergeMaps(vBaseDict, {{"A_TYPE", "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}}));
  fStringToSpv("soft_max_large3_f32", "soft_max_large3.comp",
               fMergeMaps(vBaseDict, {{"A_TYPE", "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}}));
  fStringToSpv("soft_max_large1_f32_f16", "soft_max_large1.comp",
               fMergeMaps(vBaseDict, {{"A_TYPE", "float"}, {"B_TYPE", "float16_t"}, {"D_TYPE", "float"}}));
  fStringToSpv("soft_max_large2_f32_f16", "soft_max_large2.comp",
               fMergeMaps(vBaseDict, {{"A_TYPE", "float"}, {"B_TYPE", "float16_t"}, {"D_TYPE", "float"}}));
  fStringToSpv("soft_max_large3_f32_f16", "soft_max_large3.comp",
               fMergeMaps(vBaseDict, {{"A_TYPE", "float"}, {"B_TYPE", "float16_t"}, {"D_TYPE", "float"}}));

  fStringToSpv("rope_norm_f32", "rope_norm.comp", {{"A_TYPE", "float"}, {"ROPE_D_TYPE", "float"}});
  fStringToSpv("rope_norm_f16", "rope_norm.comp", {{"A_TYPE", "float16_t"}, {"ROPE_D_TYPE", "float16_t"}});
  fStringToSpv("rope_norm_f32_f16", "rope_norm.comp", {{"A_TYPE", "float"}, {"ROPE_D_TYPE", "float16_t"}});

  fStringToSpv("rope_neox_f32", "rope_neox.comp", {{"A_TYPE", "float"}, {"ROPE_D_TYPE", "float"}});
  fStringToSpv("rope_neox_f16", "rope_neox.comp", {{"A_TYPE", "float16_t"}, {"ROPE_D_TYPE", "float16_t"}});
  fStringToSpv("rope_neox_f32_f16", "rope_neox.comp", {{"A_TYPE", "float"}, {"ROPE_D_TYPE", "float16_t"}});

  fStringToSpv("rope_multi_f32", "rope_multi.comp", {{"A_TYPE", "float"}, {"ROPE_D_TYPE", "float"}});
  fStringToSpv("rope_multi_f16", "rope_multi.comp", {{"A_TYPE", "float16_t"}, {"ROPE_D_TYPE", "float16_t"}});
  fStringToSpv("rope_multi_f32_f16", "rope_multi.comp", {{"A_TYPE", "float"}, {"ROPE_D_TYPE", "float16_t"}});

  fStringToSpv("rope_vision_f32", "rope_vision.comp", {{"A_TYPE", "float"}, {"ROPE_D_TYPE", "float"}});
  fStringToSpv("rope_vision_f16", "rope_vision.comp", {{"A_TYPE", "float16_t"}, {"ROPE_D_TYPE", "float16_t"}});

  fStringToSpv("argsort_f32", "argsort.comp", {{"A_TYPE", "float"}});
  fStringToSpv("argsort_large_f32", "argsort_large.comp", {{"A_TYPE", "float"}});

  fStringToSpv("topk_argsort_f32", "topk_argsort.comp", {{"A_TYPE", "float"}});
  fStringToSpv("topk_nary_search_f32", "topk_nary_search.comp", {{"A_TYPE", "float"}});

  fStringToSpv("argmax_f32", "argmax.comp", fMergeMaps(vBaseDict, {{"A_TYPE", "float"}, {"D_TYPE", "int"}}));
  fStringToSpv("amax_pow2_f32", "amax_pow2.comp",
               fMergeMaps(vBaseDict, {{"A_TYPE", "float"}, {"D_TYPE", "float"}}));
  fStringToSpv("amax_pow2_fused_f32", "amax_pow2.comp",
               fMergeMaps(vBaseDict, {{"A_TYPE", "float"}, {"D_TYPE", "float"}, {"FUSED", "1"}}));
  fStringToSpv("sum_rows_f32", "sum_rows.comp", fMergeMaps(vBaseDict, {{"A_TYPE", "float"}, {"D_TYPE", "float"}}));
  fStringToSpv("fwht_f32", "fwht.comp", {});
  fStringToSpv("fwht_shmem_f32", "fwht.comp", {{"FWHT_SHMEM", "1"}});
  fStringToSpv("count_equal_i32", "count_equal.comp",
               fMergeMaps(vBaseDict, {{"A_TYPE", "int"}, {"B_TYPE", "int"}, {"D_TYPE", "int"}}));
  fStringToSpv("dsv4_hc_comb_f32", "dsv4_hc_comb.comp", {});
  fStringToSpv("dsv4_hc_pre_f32", "dsv4_hc_pre.comp", {});
  fStringToSpv("dsv4_hc_post_f32", "dsv4_hc_post.comp", {});
  fStringToSpv("cumsum_f32", "cumsum.comp", fMergeMaps(vBaseDict, {{"A_TYPE", "float"}, {"D_TYPE", "float"}}));
  fStringToSpv("cumsum_multipass1_f32", "cumsum_multipass1.comp",
               fMergeMaps(vBaseDict, {{"A_TYPE", "float"}, {"D_TYPE", "float"}}));
  fStringToSpv("cumsum_multipass2_f32", "cumsum_multipass2.comp",
               fMergeMaps(vBaseDict, {{"A_TYPE", "float"}, {"D_TYPE", "float"}}));

  fStringToSpv("count_experts", "count_experts.comp", fMergeMaps(vBaseDict, {{"A_TYPE", "uint"}, {"D_TYPE", "uint"}}));

  for (std::string vDimStr : {"", "_3d"}) {
    for (bool vBda : {false, true}) {
      std::string vBdaStr = vBda ? "_bda" : "";
      std::string vBdaDef = vBda ? "1" : "0";
      fStringToSpv(
          "im2col" + vDimStr + "_f32" + vBdaStr, "im2col" + vDimStr + ".comp",
          fMergeMaps(vBaseDict, {{"A_TYPE", "float"}, {"D_TYPE", "float"}, {"D_SIZE", "4"}, {"BDA", vBdaDef}}));
      fStringToSpv(
          "im2col" + vDimStr + "_f32_f16" + vBdaStr, "im2col" + vDimStr + ".comp",
          fMergeMaps(vBaseDict, {{"A_TYPE", "float"}, {"D_TYPE", "float16_t"}, {"D_SIZE", "2"}, {"BDA", vBdaDef}}));
    }
  }

  fStringToSpv("out_prod_f32", "out_prod.comp", {});

  fStringToSpv("timestep_embedding_f32", "timestep_embedding.comp",
               fMergeMaps(vBaseDict, {{"A_TYPE", "float"}, {"D_TYPE", "float"}}));

  fStringToSpv("conv_transpose_1d_f32", "conv_transpose_1d.comp",
               {{"A_TYPE", "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}});
  fStringToSpv("col2im_1d_f32", "col2im_1d.comp", {{"DATA_A_F32", "1"}, {"A_TYPE", "float"}, {"D_TYPE", "float"}});
  fStringToSpv("col2im_1d_f16", "col2im_1d.comp",
               {{"DATA_A_F16", "1"}, {"A_TYPE", "float16_t"}, {"D_TYPE", "float16_t"}});
  fStringToSpv("col2im_1d_bf16", "col2im_1d.comp",
               {{"DATA_A_BF16", "1"}, {"A_TYPE", "uint16_t"}, {"D_TYPE", "uint16_t"}});

  fStringToSpv("snake_f32", "snake.comp", {{"DATA_A_F32", "1"}, {"A_TYPE", "float"}, {"D_TYPE", "float"}});
  fStringToSpv("snake_f16", "snake.comp", {{"DATA_A_F16", "1"}, {"A_TYPE", "float16_t"}, {"D_TYPE", "float16_t"}});
  fStringToSpv("snake_bf16", "snake.comp",
               {{"DATA_A_BF16", "1"}, {"DATA_D_BF16", "1"}, {"A_TYPE", "uint16_t"}, {"D_TYPE", "uint16_t"}});

  fStringToSpv("pool1d_f32", "pool1d.comp", fMergeMaps(vBaseDict, {{"A_TYPE", "float"}, {"D_TYPE", "float"}}));
  fStringToSpv("pool2d_f32", "pool2d.comp", fMergeMaps(vBaseDict, {{"A_TYPE", "float"}, {"D_TYPE", "float"}}));

  fStringToSpv("rwkv_wkv6_f32", "wkv6.comp", fMergeMaps(vBaseDict, {{"A_TYPE", "float"}}));

  fStringToSpv("gated_linear_attn_f32", "gla.comp", fMergeMaps(vBaseDict, {{"A_TYPE", "float"}}));

  // IQ4_NL support is compiled in so its shared lookup table is there when K uses it; the K type is a spec constant
  std::map<std::string, std::string> dLightningIndexerDict = {
      {"FLOAT_TYPE", "float"}, {"FLOAT_TYPEV4", "vec4"}, {"DATA_A_IQ4_NL", "1"}};
  fStringToSpv("lightning_indexer_f32", "lightning_indexer.comp", dLightningIndexerDict);

  fStringToSpv("rwkv_wkv7_f32", "wkv7.comp", fMergeMaps(vBaseDict, {{"A_TYPE", "float"}}));

  fStringToSpv(
      "gated_delta_net_f32", "gated_delta_net.comp",
      fMergeMaps(vBaseDict, {{"FLOAT_TYPE", "float"}, {"USE_SUBGROUP_ADD", "1"}, {"USE_SUBGROUP_CLUSTERED", "1"}}));
  fStringToSpv(
      "gated_delta_net_f32_nocluster", "gated_delta_net.comp",
      fMergeMaps(vBaseDict, {{"FLOAT_TYPE", "float"}, {"USE_SUBGROUP_ADD", "1"}, {"USE_SUBGROUP_CLUSTERED", "0"}}));
  fStringToSpv(
      "gated_delta_net_f32_shmem", "gated_delta_net.comp",
      fMergeMaps(vBaseDict, {{"FLOAT_TYPE", "float"}, {"USE_SUBGROUP_ADD", "0"}, {"USE_SUBGROUP_CLUSTERED", "0"}}));

  fStringToSpv("opt_step_adamw_f32", "opt_step_adamw.comp", fMergeMaps(vBaseDict, {{"A_TYPE", "float"}}));
  fStringToSpv("opt_step_sgd_f32", "opt_step_sgd.comp", fMergeMaps(vBaseDict, {{"A_TYPE", "float"}}));

  fStringToSpv("solve_tri_f32", "solve_tri.comp",
               fMergeMaps(vBaseDict, {{"A_TYPE", "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}}));

  for (auto vTranspose : {false, true}) {
    for (auto vUnroll : {false, true}) {
      for (auto vAF16 : {false, true}) {
        std::map<std::string, std::string> vDefines = {
            {"A_TYPE", vAF16 ? "float16_t" : "float"},
            {"B_TYPE", "float"},
            {"D_TYPE", "float"},
            {"USE_COLLECTIVES", "1"},
            {"UNROLL", vUnroll ? "[[unroll]]" : ""},
        };
        if (vTranspose)
          vDefines["TRANSPOSE"] = "1";
        std::string vName = std::string(vTranspose ? "conv_transpose_2d" : "conv2d") + (vAF16 ? "_f16" : "") + "_f32";
        fStringToSpv(vName + (vUnroll ? "_unroll" : ""), "conv2d_mm.comp", vDefines);
#if defined(GGML_VULKAN_COOPMAT2_GLSLC_SUPPORT)
        if (vUnroll) {
          auto cm2_defines = vDefines;
          cm2_defines["COOPMAT2"] = "1";
          fStringToSpv(vName, "conv2d_mm.comp", cm2_defines, true, false, true);
        }
#endif
#if defined(GGML_VULKAN_COOPMAT_GLSLC_SUPPORT)
        if (vUnroll) {
          auto cm1_defines = vDefines;
          cm1_defines["COOPMAT"] = "1";
          fStringToSpv(vName, "conv2d_mm.comp", cm1_defines, true, true, false);
        }
#endif
      }
    }
  }

  for (auto vUnroll : {false, true}) {
    for (auto vAF16 : {false, true}) {
      std::map<std::string, std::string> vDefines = {
          {"A_TYPE", vAF16 ? "float16_t" : "float"},
          {"B_TYPE", "float"},
          {"D_TYPE", "float"},
          {"UNROLL", vUnroll ? "[[unroll]]" : ""},
      };
      std::string vName = std::string("conv3d") + (vAF16 ? "_f16" : "") + "_f32";
      fStringToSpv(vName + (vUnroll ? "_unroll" : ""), "conv3d_mm.comp", vDefines);
#if defined(GGML_VULKAN_COOPMAT2_GLSLC_SUPPORT)
      if (vUnroll) {
        auto cm2_defines = vDefines;
        cm2_defines["COOPMAT2"] = "1";
        fStringToSpv(vName, "conv3d_mm.comp", cm2_defines, true, false, true);
      }
#endif
#if defined(GGML_VULKAN_COOPMAT_GLSLC_SUPPORT)
      if (vUnroll) {
        auto cm1_defines = vDefines;
        cm1_defines["COOPMAT"] = "1";
        fStringToSpv(vName, "conv3d_mm.comp", cm1_defines, true, true, false);
      }
#endif
    }
  }

  fStringToSpv("conv2d_dw_whcn_f32", "conv2d_dw.comp",
               fMergeMaps(vBaseDict, {{"A_TYPE", "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}, {"WHCN", "1"}}));
  fStringToSpv("conv2d_dw_cwhn_f32", "conv2d_dw.comp",
               fMergeMaps(vBaseDict, {{"A_TYPE", "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}, {"CWHN", "1"}}));
  fStringToSpv(
      "conv2d_dw_whcn_f16_f32", "conv2d_dw.comp",
      fMergeMaps(vBaseDict, {{"A_TYPE", "float16_t"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}, {"WHCN", "1"}}));
  fStringToSpv(
      "conv2d_dw_cwhn_f16_f32", "conv2d_dw.comp",
      fMergeMaps(vBaseDict, {{"A_TYPE", "float16_t"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}, {"CWHN", "1"}}));

  fStringToSpv("roll_f32", "roll.comp", fMergeMaps(vBaseDict, {{"A_TYPE", "float"}, {"D_TYPE", "float"}}));

  fStringToSpv("add_id_f32", "add_id.comp",
               fMergeMaps(vBaseDict, {{"A_TYPE", "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}}));

  fStringToSpv(
      "multi_add_f32", "multi_add.comp",
      {{"A_TYPE", "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}, {"FLOAT_TYPE", "float"}, {"ADD_RMS", "0"}});
  fStringToSpv(
      "multi_add_rms_f32", "multi_add.comp",
      {{"A_TYPE", "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}, {"FLOAT_TYPE", "float"}, {"ADD_RMS", "1"}});

  fStringToSpv("ssm_scan_f32", "ssm_scan.comp", {{"A_TYPE", "float"}});
  fStringToSpv("ssm_scan_subgroup_f32", "ssm_scan.comp", {{"A_TYPE", "float"}, {"USE_SUBGROUP_ADD", "1"}});

  fStringToSpv("ssm_conv_f32", "ssm_conv.comp", {{"A_TYPE", "float"}});

  fStringToSpv("topk_moe_f32", "topk_moe.comp", {});

  for (auto &vC : compiles) {
    vC.wait();
  }
}

void fWriteOutputFiles() {
  std::stringstream vHdr = fMakeGenericStringstream();
  std::stringstream vSrc = fMakeGenericStringstream();

  vHdr << "#include <cstdint>\n\n";
  vSrc << "#include \"" << fBasename(target_hpp) << "\"\n\n";

  std::sort(shader_fnames.begin(), shader_fnames.end());
  for (const auto &vPair : shader_fnames) {
    const std::string &vName = vPair.first;
#ifdef _WIN32
    std::string vPath = vPair.second;
    std::replace(vPath.begin(), vPath.end(), '/', '\\');
#else
    const std::string &vPath = vPair.second;
#endif

    vHdr << "extern const uint64_t " << vName << "_len;\n";
    vHdr << "extern const unsigned char " << vName << "_data[];\n\n";

    if (input_filepath != "") {
      std::string vData = fReadBinaryFile(vPath);
      if (vData.empty()) {
        continue;
      }

      vSrc << "const uint64_t " << vName << "_len = " << vData.size() << ";\n";
      vSrc << "const unsigned char " << vName << "_data[" << vData.size() << "] = {\n" << std::hex;
      auto vBytes = reinterpret_cast<const uint8_t *>(vData.data());
      for (size_t vI = 0; vI < vData.size(); ++vI) {
        vSrc << "0x" << static_cast<int>(vBytes[vI]) << ",";
        if ((vI + 1) % 12 == 0)
          vSrc << "\n";
      }
      vSrc << std::dec << "\n};\n\n";
    }
  }

  std::string aSuffixes[2] = {"_f32", "_f16"};
  for (std::string vOp : {"add", "sub", "mul", "div", "add_rms"}) {
    vHdr << "extern const void * " << vOp << "_data[2][2][2];\n";
    vHdr << "extern const uint64_t " << vOp << "_len[2][2][2];\n";

    std::string vOpFile = vOp == "add_rms" ? "add.comp" : std::string(vOp) + ".comp";
    if (fBasename(input_filepath) != vOpFile) {
      continue;
    }
    std::stringstream vData = fMakeGenericStringstream();
    std::stringstream vLen = fMakeGenericStringstream();
    vData << "const void * " << vOp << "_data[2][2][2] = ";
    vLen << "const uint64_t " << vOp << "_len[2][2][2] = ";
    for (uint32_t vT0 = 0; vT0 < 2; ++vT0) {
      if (vT0 == 0) {
        vData << "{";
        vLen << "{";
      }
      for (uint32_t vT1 = 0; vT1 < 2; ++vT1) {
        if (vT1 == 0) {
          vData << "{";
          vLen << "{";
        }
        for (uint32_t vT2 = 0; vT2 < 2; ++vT2) {
          if (vT2 == 0) {
            vData << "{";
            vLen << "{";
          }
          vData << vOp << aSuffixes[vT0] << aSuffixes[vT1] << aSuffixes[vT2];
          vLen << vOp << aSuffixes[vT0] << aSuffixes[vT1] << aSuffixes[vT2];
          vData << "_data,";
          vLen << "_len,";
          if (vT2 == 1) {
            vData << "}, ";
            vLen << "}, ";
          }
        }
        if (vT1 == 1) {
          vData << "}, ";
          vLen << "}, ";
        }
      }
      if (vT0 == 1) {
        vData << "};\n";
        vLen << "};\n";
      }
    }
    vSrc << vData.str();
    vSrc << vLen.str();
  }

  std::vector<std::string> vBtypes = {"f16", "f32"};

#if defined(GGML_VULKAN_INTEGER_DOT_GLSLC_SUPPORT)
  vBtypes.push_back("q8_1");
#endif

  for (const std::string &vBtype : vBtypes) {
    for (const auto &vTname : type_names) {
      if (vBtype == "q8_1" && !fIsLegacyQuant(vTname) && vTname != "mxfp4" && !fIsKQuant(vTname) && vTname != "iq1_s" &&
          vTname != "iq1_m") {
        continue;
      }
      vHdr << "extern const void * arr_dmmv_" << vTname << "_" << vBtype << "_f32_data[3];\n";
      vHdr << "extern const uint64_t arr_dmmv_" << vTname << "_" << vBtype << "_f32_len[3];\n";
      if (fBasename(input_filepath) == "mul_mat_vec.comp") {
        vSrc << "const void * arr_dmmv_" << vTname << "_" << vBtype << "_f32_data[3] = {mul_mat_vec_" << vTname << "_"
             << vBtype << "_f32_data, mul_mat_vec_" << vTname << "_" << vBtype << "_f32_subgroup_data, mul_mat_vec_"
             << vTname << "_" << vBtype << "_f32_subgroup_no_shmem_data};\n";
        vSrc << "const uint64_t arr_dmmv_" << vTname << "_" << vBtype << "_f32_len[3] =  {mul_mat_vec_" << vTname << "_"
             << vBtype << "_f32_len,  mul_mat_vec_" << vTname << "_" << vBtype << "_f32_subgroup_len, mul_mat_vec_"
             << vTname << "_" << vBtype << "_f32_subgroup_no_shmem_len};\n";
      }

      if (vBtype == "f16") {
        continue;
      }
      vHdr << "extern const void * arr_dmmv_id_" << vTname << "_" << vBtype << "_f32_data[3];\n";
      vHdr << "extern const uint64_t arr_dmmv_id_" << vTname << "_" << vBtype << "_f32_len[3];\n";
      if (fBasename(input_filepath) == "mul_mat_vec.comp") {
        vSrc << "const void * arr_dmmv_id_" << vTname << "_" << vBtype << "_f32_data[3] = {mul_mat_vec_id_" << vTname
             << "_" << vBtype << "_f32_data, mul_mat_vec_id_" << vTname << "_" << vBtype
             << "_f32_subgroup_data, mul_mat_vec_id_" << vTname << "_" << vBtype << "_f32_subgroup_no_shmem_data};\n";
        vSrc << "const uint64_t arr_dmmv_id_" << vTname << "_" << vBtype << "_f32_len[3] =  {mul_mat_vec_id_" << vTname
             << "_" << vBtype << "_f32_len,  mul_mat_vec_id_" << vTname << "_" << vBtype
             << "_f32_subgroup_len, mul_mat_vec_id_" << vTname << "_" << vBtype << "_f32_subgroup_no_shmem_len};\n";
      }
    }
  }

#if defined(GGML_VULKAN_FLOAT_E2M1_GLSLC_SUPPORT) && defined(GGML_VULKAN_FLOAT_E4M3_GLSLC_SUPPORT)
  for (const std::string &vBtype : {"f16", "f32"}) {
    for (const std::string &vTname : {"mxfp4", "nvfp4"}) {
      vHdr << "extern const void * arr_dmmv_" << vTname << "_" << vBtype << "_f32_ocp_data[3];\n";
      vHdr << "extern const uint64_t arr_dmmv_" << vTname << "_" << vBtype << "_f32_ocp_len[3];\n";
      if (fBasename(input_filepath) == "mul_mat_vec.comp") {
        vSrc << "const void * arr_dmmv_" << vTname << "_" << vBtype << "_f32_ocp_data[3] = {mul_mat_vec_" << vTname
             << "_" << vBtype << "_f32_ocp_data, mul_mat_vec_" << vTname << "_" << vBtype
             << "_f32_ocp_subgroup_data, mul_mat_vec_" << vTname << "_" << vBtype
             << "_f32_ocp_subgroup_no_shmem_data};\n";
        vSrc << "const uint64_t arr_dmmv_" << vTname << "_" << vBtype << "_f32_ocp_len[3] =  {mul_mat_vec_" << vTname
             << "_" << vBtype << "_f32_ocp_len,  mul_mat_vec_" << vTname << "_" << vBtype
             << "_f32_ocp_subgroup_len, mul_mat_vec_" << vTname << "_" << vBtype
             << "_f32_ocp_subgroup_no_shmem_len};\n";
      }
    }
  }
  for (const std::string &vTname : {"mxfp4", "nvfp4"}) {
    vHdr << "extern const void * arr_dmmv_id_" << vTname << "_f32_f32_ocp_data[3];\n";
    vHdr << "extern const uint64_t arr_dmmv_id_" << vTname << "_f32_f32_ocp_len[3];\n";
    if (fBasename(input_filepath) == "mul_mat_vec.comp") {
      vSrc << "const void * arr_dmmv_id_" << vTname << "_f32_f32_ocp_data[3] = {mul_mat_vec_id_" << vTname
           << "_f32_f32_ocp_data, mul_mat_vec_id_" << vTname << "_f32_f32_ocp_subgroup_data, mul_mat_vec_id_" << vTname
           << "_f32_f32_ocp_subgroup_no_shmem_data};\n";
      vSrc << "const uint64_t arr_dmmv_id_" << vTname << "_f32_f32_ocp_len[3] =  {mul_mat_vec_id_" << vTname
           << "_f32_f32_ocp_len,  mul_mat_vec_id_" << vTname << "_f32_f32_ocp_subgroup_len, mul_mat_vec_id_" << vTname
           << "_f32_f32_ocp_subgroup_no_shmem_len};\n";
    }
  }
#endif

  if (input_filepath == "") {
    fWriteFileIfChanged(target_hpp, vHdr.str());
  }
  if (target_cpp != "") {
    fWriteBinaryFile(target_cpp, vSrc.str());
  }
}

} // namespace

int main(int argc, char **argv) {
  std::map<std::string, std::string> dArgs;
  for (int vI = 1; vI < argc; ++vI) {
    std::string vArg = argv[vI];
    if (vArg.rfind("--", 0) == 0) {
      if (vI + 1 < argc && argv[vI + 1][0] != '-') {
        dArgs[vArg] = argv[vI + 1];
        ++vI;
      } else {
        dArgs[vArg] = "";
      }
    }
  }

  if (dArgs.find("--glslc") != dArgs.end()) {
    GLSLC = dArgs["--glslc"]; // Path to glslc
  }
  if (dArgs.find("--source") != dArgs.end()) {
    input_filepath = dArgs["--source"]; // The shader source file to compile
  }
  if (dArgs.find("--output-dir") != dArgs.end()) {
    output_dir = dArgs["--output-dir"]; // Directory for containing SPIR-V output
  }
  if (dArgs.find("--target-hpp") != dArgs.end()) {
    target_hpp = dArgs["--target-hpp"]; // Path to generated header file
  }
  if (dArgs.find("--target-cpp") != dArgs.end()) {
    target_cpp = dArgs["--target-cpp"]; // Path to generated cpp file
  }

  if (!fDirectoryExists(output_dir)) {
    if (!fCreateDirectory(output_dir)) {
      std::cerr << "Error creating output directory: " << output_dir << "\n";
      return EXIT_FAILURE;
    }
  }

  fProcessShaders();

  if (compile_failed) {
    std::cerr << "vulkan-shaders-gen: one or more shaders failed to compile" << std::endl;
    return EXIT_FAILURE;
  }

  fWriteOutputFiles();

  return EXIT_SUCCESS;
}
