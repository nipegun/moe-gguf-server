#pragma once

#include "hf-cache.h"

#include <stdexcept>
#include <string>
#include <vector>
#include <functional>

struct common_params_model;

using common_header = std::pair<std::string, std::string>;
using common_header_list = std::vector<common_header>;

struct common_download_progress {
  std::string vUrl;
  size_t vDownloaded = 0;
  size_t vTotal = 0;
  bool vCached = false;
};

class common_download_callback {
public:
  virtual ~common_download_callback() = default;
  virtual void mOnStart(const common_download_progress &pP) = 0;
  virtual void mOnUpdate(const common_download_progress &pP) = 0;
  virtual void mOnDone(const common_download_progress &pP, bool pOk) = 0;
  virtual bool mIsCancelled() const { return false; }
};

struct common_remote_params {
  common_header_list vHeaders;
  long vTimeout = 0;              // read/write timeout in seconds, 0 means no timeout
  long vMaxSize = 0;              // unlimited if 0
  long vConnectTimeout = 0;       // name resolution and connection timeout in seconds, 0 keeps the client default
  long vTotalTimeout = 0;         // deadline in seconds for the whole fetch (resolution, connections, redirects and body)
  int vMaxRedirects = -1;         // -1 lets the client follow redirects, otherwise they are followed here up to this many
  bool vPublicHostsOnly = false;  // resolve each host once, reject non-public addresses and connect to the validated one
};

// thrown when a destination is refused because it is not a public address (common_remote_params::vPublicHostsOnly)
class common_remote_destination_error : public std::runtime_error {
public:
  using std::runtime_error::runtime_error;
};

// get remote file content, returns <http_code, raw_response_body>
std::pair<long, std::vector<char>> fCommonRemoteGetContent(const std::string &pUrl,
                                                           const common_remote_params &pParams);

// split HF repo with tag into <repo, tag>, for example:
// - "ggml-org/models:F16" -> <"ggml-org/models", "F16">
// tag is optional and can be empty
std::pair<std::string, std::string> fCommonDownloadSplitRepoTag(const std::string &pHfRepoWithTag);

// Result of fCommonListCachedModels
struct common_cached_model_info {
  std::string vRepo;
  std::string vTag;
  std::string mToString() const { return vRepo + ":" + vTag; }
};

// Options for fCommonDownloadFileSingle
struct common_download_opts {
  std::string vBearerToken;
  common_header_list vHeaders;
  bool vOffline = false;
  bool vDownloadMmproj = false;
  bool vDownloadMtp = false;
  bool vDownloadEagle3 = false;
  bool vDownloadDflash = false;
  bool vDownloadDspark = false;
  common_download_callback *vCallback = nullptr;
};

struct common_download_task {
  common_download_opts vOpts;
  std::string vUrl;
  std::string vLocalPath;
  std::function<void()> vOnDone;
  bool vIsHf = false;

  common_download_task() = default;
  common_download_task(hf_cache::vHfFile pF, const common_download_opts &pOpts, std::function<void()> pOnDone = nullptr)
      : vOpts(pOpts), vUrl(pF.vUrl), vLocalPath(pF.vLocalPath), vOnDone(pOnDone), vIsHf(true) {}
};

void fCommonDownloadRunTasks(const std::vector<common_download_task> &pTasks);

// if url is a multi-part GGUF file, returns all parts, otherwise returns the single file
std::vector<std::string> fCommonDownloadGetAllParts(const std::string &pUrl);

// returns list of cached models
std::vector<common_cached_model_info> fCommonListCachedModels();

// download single file from url to local path
// returns status code or -1 on error
// skip_etag: if true, don't read/write .etag files (for HF cache where filename is the hash)
int fCommonDownloadFileSingle(const std::string &pUrl, const std::string &pPath, const common_download_opts &pOpts = {},
                              bool pSkipEtag = false);

// resolve and download model from Docker registry
// return local path to downloaded model file
std::string fCommonDockerResolveModel(const std::string &pDocker);

// Remove a cached model from disk
// input format: "user/model" or "user/model:tag"
// - if tag is omitted, removes the entire repo cache directory
// - if tag is present, removes only files matching that tag (and orphaned blobs)
// returns true if anything was removed
bool fCommonDownloadRemove(const std::string &pHfRepoWithTag);

struct common_download_hf_plan {
  hf_cache::vHfFile vPrimary;
  hf_cache::hf_files vModelFiles;
  hf_cache::vHfFile vMmproj;
  hf_cache::vHfFile vMtp;
  hf_cache::vHfFile vEagle3;
  hf_cache::vHfFile vDflash;
  hf_cache::vHfFile vDspark;
  hf_cache::vHfFile vPreset; // if set, only this file is downloaded
};
common_download_hf_plan fCommonDownloadGetHfPlan(const common_params_model &pModel, const common_download_opts &pOpts);
