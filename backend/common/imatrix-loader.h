#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

inline constexpr const char *cLlmKvImatrixDatasets = "imatrix.datasets";
inline constexpr const char *cLlmKvImatrixChunkCount = "imatrix.chunk_count";
inline constexpr const char *cLlmKvImatrixChunkSize = "imatrix.chunk_size";

struct common_imatrix_entry {
  std::vector<float> lSums;
  std::vector<int64_t> lCounts;
};

struct common_imatrix {
  std::map<std::string, common_imatrix_entry> dEntries;
  std::vector<std::string> lDatasets;
  int32_t vChunkCount = 0;
  int32_t vChunkSize = 0;
  bool vHasMetadata = false;
};

bool fCommonImatrixLoad(const std::string &pFname, common_imatrix &pImatrix);
