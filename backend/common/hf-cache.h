#pragma once

#include <string>
#include <vector>

// Ref: https://huggingface.co/docs/hub/local-cache.md

namespace hf_cache {

struct vHfFile {
  std::string vPath;
  std::string vUrl;
  std::string vLocalPath;
  std::string vFinalPath;
  std::string vOid;
  std::string vRepoId;
};

using hf_files = std::vector<vHfFile>;

// Get files from HF API
hf_files fGetRepoFiles(const std::string &pRepoId, const std::string &pToken);

hf_files fGetCachedFiles(const std::string &pRepoId = {});

// Create snapshot path (link or move/copy) and return it
std::string fFinalizeFile(const vHfFile &pFile);

// Remove the entire cached directory for a repo, returns true if removed
bool fRemoveCachedRepo(const std::string &pRepoId);

} // namespace hf_cache
