#include "arg.h"

#include "build-info.h"
#include "common.h"
#include "log.h"
#include "download.h"
#include "hf-cache.h"

#define JSON_ASSERT GGML_ASSERT
#include <nlohmann/json.hpp>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <future>
#include <map>
#include <mutex>
#include <regex>
#include <unordered_set>
#include <string>
#include <thread>
#include <vector>

#include "http.h"

#ifndef __EMSCRIPTEN__
#ifdef __linux__
#include <linux/limits.h>
#elif defined(_WIN32)
#if !defined(PATH_MAX)
#define PATH_MAX MAX_PATH
#endif
#elif defined(_AIX)
#include <sys/limits.h>
#else
#include <sys/syslimits.h>
#endif
#endif

// isatty
#if defined(_WIN32)
#include <io.h>
#else
#include <unistd.h>
#endif

// getaddrinfo, inet_ntop
#if defined(_WIN32)
#include <ws2tcpip.h>
#else
#include <netdb.h>
#endif

using json = nlohmann::ordered_json;

//
// downloader
//

// validate repo name format: owner/repo
static void fWriteFile(const std::string &pFname, const std::string &pContent) {
  const std::string cFnameTmp = pFname + ".tmp";
  std::ofstream vFile(cFnameTmp);
  if (!vFile) {
    throw std::runtime_error(fStringFormat("error: failed to open file '%s'\n", pFname.c_str()));
  }

  try {
    vFile << pContent;
    vFile.close();

    // Makes write atomic
    if (rename(cFnameTmp.c_str(), pFname.c_str()) != 0) {
      LOG_ERR("%s: unable to rename file: %s to %s\n", __func__, cFnameTmp.c_str(), pFname.c_str());
      // If rename fails, try to delete the temporary file
      if (remove(cFnameTmp.c_str()) != 0) {
        LOG_ERR("%s: unable to delete temporary file: %s\n", __func__, cFnameTmp.c_str());
      }
    }
  } catch (...) {
    // If anything fails, try to delete the temporary file
    if (remove(cFnameTmp.c_str()) != 0) {
      LOG_ERR("%s: unable to delete temporary file: %s\n", __func__, cFnameTmp.c_str());
    }

    throw std::runtime_error(fStringFormat("error: failed to write file '%s'\n", pFname.c_str()));
  }
}

static void fWriteEtag(const std::string &pPath, const std::string &pEtag) {
  const std::string cEtagPath = pPath + ".etag";
  fWriteFile(cEtagPath, pEtag);
  LOG_DBG("%s: file etag saved: %s\n", __func__, cEtagPath.c_str());
}

static std::string fReadEtag(const std::string &pPath) {
  const std::string cEtagPath = pPath + ".etag";
  if (!std::filesystem::exists(cEtagPath)) {
    return {};
  }
  std::ifstream vEtagIn(cEtagPath);
  if (!vEtagIn) {
    LOG_ERR("%s: could not open .etag file for reading: %s\n", __func__, cEtagPath.c_str());
    return {};
  }
  std::string vEtag;
  std::getline(vEtagIn, vEtag);
  return vEtag;
}

static bool fIsHttpStatusOk(int pStatus) { return pStatus >= 200 && pStatus < 400; }

std::pair<std::string, std::string> fCommonDownloadSplitRepoTag(const std::string &pHfRepoWithTag) {
  auto vParts = fStringSplit<std::string>(pHfRepoWithTag, ':');
  std::string vTag = vParts.size() > 1 ? vParts.back() : "";
  std::string vHfRepo = vParts[0];
  if (fStringSplit<std::string>(vHfRepo, '/').size() != 2) {
    throw std::invalid_argument("error: invalid HF repo format, expected <user>/<model>[:quant]\n");
  }
  return {vHfRepo, vTag};
}

class ProgressBar : public common_download_callback {
  static inline std::mutex vMutex;
  static inline std::map<const ProgressBar *, int> dLines;
  static inline int vMaxLine = 0;

  std::string vFilename;
  size_t vLen = 0;

  static void mCleanup(const ProgressBar *pLine) {
    dLines.erase(pLine);
    if (dLines.empty()) {
      vMaxLine = 0;
    }
  }

  static bool mIsOutputATty() {
#if defined(_WIN32)
    return _isatty(_fileno(stdout));
#else
    return isatty(1);
#endif
  }

public:
  ProgressBar() = default;

  void mOnStart(const common_download_progress &pP) override {
    vFilename = pP.vUrl;

    if (auto vPos = vFilename.rfind('/'); vPos != std::string::npos) {
      vFilename = vFilename.substr(vPos + 1);
    }
    if (auto vPos = vFilename.find('?'); vPos != std::string::npos) {
      vFilename = vFilename.substr(0, vPos);
    }
    for (size_t vI = 0; vI < vFilename.size(); ++vI) {
      if ((vFilename[vI] & 0xC0) != 0x80) {
        if (vLen++ == 39) {
          vFilename.resize(vI);
          vFilename += "…";
          break;
        }
      }
    }
  }

  void mOnDone(const common_download_progress &, bool) override {
    std::lock_guard<std::mutex> vLock(vMutex);
    mCleanup(this);
  }

  void mOnUpdate(const common_download_progress &pP) override {
    if (!pP.vTotal || !mIsOutputATty()) {
      return;
    }

    std::lock_guard<std::mutex> vLock(vMutex);

    if (dLines.find(this) == dLines.end()) {
      dLines[this] = vMaxLine++;
      std::cout << "\n";
    }
    int vLinesUp = vMaxLine - dLines[this];

    size_t vBar = (55 - vLen) * 2;
    size_t vPct = (100 * pP.vDownloaded) / pP.vTotal;
    size_t vPos = (vBar * pP.vDownloaded) / pP.vTotal;

    if (vLinesUp > 0) {
      std::cout << "\033[" << vLinesUp << "A";
    }
    std::cout << '\r' << "Downloading " << vFilename << " ";

    for (size_t vI = 0; vI < vBar; vI += 2) {
      std::cout << (vI + 1 < vPos ? "─" : (vI < vPos ? "╴" : " "));
    }
    std::cout << std::setw(4) << vPct << "%\033[K";

    if (vLinesUp > 0) {
      std::cout << "\033[" << vLinesUp << "B";
    }
    std::cout << '\r' << std::flush;

    if (pP.vDownloaded == pP.vTotal) {
      mCleanup(this);
    }
  }

  ProgressBar(const ProgressBar &) = delete;
  ProgressBar &operator=(const ProgressBar &) = delete;
};

static bool fCommonPullFile(httplib::Client &pCli, const std::string &pResolvePath, const std::string &pPathTmp,
                            bool pSupportsRanges, common_download_progress &pP, common_download_callback *pCallback) {
  std::ofstream vOfs(pPathTmp, std::ios::binary | std::ios::app);
  if (!vOfs.is_open()) {
    LOG_ERR("%s: error opening local file for writing: %s\n", __func__, pPathTmp.c_str());
    return false;
  }

  httplib::Headers vHeaders;
  if (pSupportsRanges && pP.vDownloaded > 0) {
    vHeaders.emplace("Range", "bytes=" + std::to_string(pP.vDownloaded) + "-");
  }

  const char *vFunc = __func__; // avoid __func__ inside a lambda
  size_t vProgressStep = 0;

  auto vRes = pCli.Get(
      pResolvePath, vHeaders,
      [&](const httplib::Response &pResponse) {
        if (pP.vDownloaded > 0 && pResponse.status != 206) {
          LOG_WRN("%s: server did not respond with 206 Partial Content for a resume request. Status: %d\n", vFunc,
                  pResponse.status);
          return false;
        }
        if (pP.vDownloaded == 0 && pResponse.status != 200) {
          LOG_WRN("%s: download received non-successful status code: %d\n", vFunc, pResponse.status);
          return false;
        }
        if (pP.vTotal == 0 && pResponse.has_header("Content-Length")) {
          try {
            size_t vContentLength = std::stoull(pResponse.get_header_value("Content-Length"));
            pP.vTotal = pP.vDownloaded + vContentLength;
          } catch (const std::exception &vE) {
            LOG_WRN("%s: invalid Content-Length header: %s\n", vFunc, vE.what());
          }
        }
        return true;
      },
      [&](const char *pData, size_t pLen) {
        vOfs.write(pData, pLen);
        if (!vOfs) {
          LOG_ERR("%s: error writing to file: %s\n", vFunc, pPathTmp.c_str());
          return false;
        }
        pP.vDownloaded += pLen;
        vProgressStep += pLen;

        if (vProgressStep >= pP.vTotal / 1000 || pP.vDownloaded == pP.vTotal) {
          if (pCallback) {
            pCallback->mOnUpdate(pP);
            if (pCallback->mIsCancelled()) {
              return false;
            }
          }
          vProgressStep = 0;
        }
        return true;
      },
      nullptr);

  if (!vRes) {
    LOG_ERR("%s: download failed: %s (status: %d)\n", __func__, httplib::to_string(vRes.error()).c_str(),
            vRes ? vRes->status : -1);
    return false;
  }

  return true;
}

// download one single file from remote URL to local path
// returns status code or -1 on error
static int fCommonDownloadFileSingleOnline(const std::string &pUrl, const std::string &pPath,
                                           const common_download_opts &pOpts, bool pSkipEtag) {
  static const int cMaxAttempts = 3;
  static const int cRetryDelaySeconds = 2;

  const bool cFileExists = std::filesystem::exists(pPath);

  if (cFileExists && pSkipEtag) {
    LOG_DBG("%s: using cached file: %s\n", __func__, pPath.c_str());
    return 304; // 304 Not Modified - fake cached response
  }

  auto [vCli, vParts] = fCommonHttpClient(pUrl);

  httplib::Headers vHeaders;
  for (const auto &vH : pOpts.vHeaders) {
    vHeaders.emplace(vH.first, vH.second);
  }
  if (vHeaders.find("User-Agent") == vHeaders.end()) {
    vHeaders.emplace("User-Agent", "mgs/" + std::string(mgs_version()));
  }
  if (!pOpts.vBearerToken.empty()) {
    vHeaders.emplace("Authorization", "Bearer " + pOpts.vBearerToken);
  }
  vCli.set_default_headers(vHeaders);

  std::string vLastEtag;
  if (cFileExists) {
    vLastEtag = fReadEtag(pPath);
  } else {
    LOG_DBG("%s: no previous model file found %s\n", __func__, pPath.c_str());
  }

  auto vHead = vCli.Head(vParts.vPath);
  if (!vHead || vHead->status < 200 || vHead->status >= 300) {
    LOG_TRC("%s: HEAD failed, status: %d\n", __func__, vHead ? vHead->status : -1);
    if (cFileExists) {
      LOG_TRC("%s: using cached file (HEAD failed): %s\n", __func__, pPath.c_str());
      return 304; // 304 Not Modified - fake cached response
    }
    return vHead ? vHead->status : -1;
  }

  std::string vEtag;
  if (vHead->has_header("ETag")) {
    vEtag = vHead->get_header_value("ETag");
  }

  common_download_progress vP;
  vP.vUrl = pUrl;
  if (vHead->has_header("Content-Length")) {
    try {
      vP.vTotal = std::stoull(vHead->get_header_value("Content-Length"));
    } catch (const std::exception &vE) {
      LOG_WRN("%s: invalid Content-Length in HEAD response: %s\n", __func__, vE.what());
    }
  }

  bool vSupportsRanges = false;
  if (vHead->has_header("Accept-Ranges")) {
    vSupportsRanges = vHead->get_header_value("Accept-Ranges") != "none";
  }

  if (cFileExists) {
    if (vEtag.empty()) {
      LOG_DBG("%s: using cached file (no server etag): %s\n", __func__, pPath.c_str());
      return 304; // 304 Not Modified - fake cached response
    }
    if (!vLastEtag.empty() && vLastEtag == vEtag) {
      LOG_DBG("%s: using cached file (same etag): %s\n", __func__, pPath.c_str());
      return 304; // 304 Not Modified - fake cached response
    }
    // pass this point, the file exists but is different from the server version, so we need to redownload it
    if (remove(pPath.c_str()) != 0) {
      LOG_ERR("%s: unable to delete file: %s\n", __func__, pPath.c_str());
      return -1;
    }
  }

  { // silent
    std::error_code vEc;
    std::filesystem::create_directories(std::filesystem::path(pPath).parent_path(), vEc);
  }

  bool vSuccess = false;
  const std::string cPathTemporary = pPath + ".downloadInProgress";
  int vDelay = cRetryDelaySeconds;

  if (pOpts.vCallback) {
    pOpts.vCallback->mOnStart(vP);
  }

  for (int vI = 0; vI < cMaxAttempts; ++vI) {
    if (pOpts.vCallback && pOpts.vCallback->mIsCancelled()) {
      break;
    }
    if (vI) {
      LOG_WRN("%s: retrying after %d seconds...\n", __func__, vDelay);
      std::this_thread::sleep_for(std::chrono::seconds(vDelay));
      vDelay *= cRetryDelaySeconds;
    }

    size_t vExistingSize = 0;

    if (std::filesystem::exists(cPathTemporary)) {
      if (vSupportsRanges) {
        vExistingSize = std::filesystem::file_size(cPathTemporary);
      } else if (remove(cPathTemporary.c_str()) != 0) {
        LOG_ERR("%s: unable to delete file: %s\n", __func__, cPathTemporary.c_str());
        break;
      }
    }

    vP.vDownloaded = vExistingSize;

    LOG_DBG("%s: downloading from %s to %s (etag:%s)...\n", __func__, fCommonHttpShowMaskedUrl(vParts).c_str(),
            cPathTemporary.c_str(), vEtag.c_str());

    if (fCommonPullFile(vCli, vParts.vPath, cPathTemporary, vSupportsRanges, vP, pOpts.vCallback)) {
      if (std::rename(cPathTemporary.c_str(), pPath.c_str()) != 0) {
        LOG_ERR("%s: unable to rename file: %s to %s\n", __func__, cPathTemporary.c_str(), pPath.c_str());
        break;
      }
      if (!vEtag.empty() && !pSkipEtag) {
        fWriteEtag(pPath, vEtag);
      }
      vSuccess = true;
      break;
    }
  }

  if (pOpts.vCallback) {
    pOpts.vCallback->mOnDone(vP, vSuccess);
  }
  if (pOpts.vCallback && pOpts.vCallback->mIsCancelled() && std::filesystem::exists(cPathTemporary)) {
    if (remove(cPathTemporary.c_str()) != 0) {
      LOG_ERR("%s: unable to delete temporary file: %s\n", __func__, cPathTemporary.c_str());
    }
  }
  if (!vSuccess) {
    LOG_ERR("%s: download failed after %d attempts\n", __func__, cMaxAttempts);
    return -1; // max attempts reached
  }

  return vHead->status;
}

// redirects followed by fCommonRemoteGetContent when it follows them itself and no maximum is given
static constexpr int cRemoteDefaultMaxRedirects = 5;

// true if the IPv4 address (host byte order) is publicly routable: not "this network", private, CGNAT, loopback,
// link-local (cloud metadata services), documentation, benchmarking, multicast, reserved or broadcast
static bool fRemoteIpv4IsPublic(uint32_t pAddr) {
  struct ipv4_block {
    uint32_t vNet;
    int vBits;
  };
  static const ipv4_block aBlocked[] = {
      {0x00000000, 8},  // 0.0.0.0/8 "this network"
      {0x0A000000, 8},  // 10.0.0.0/8 private
      {0x64400000, 10}, // 100.64.0.0/10 carrier-grade NAT
      {0x7F000000, 8},  // 127.0.0.0/8 loopback
      {0xA9FE0000, 16}, // 169.254.0.0/16 link-local
      {0xAC100000, 12}, // 172.16.0.0/12 private
      {0xC0000000, 24}, // 192.0.0.0/24 IETF protocol assignments
      {0xC0000200, 24}, // 192.0.2.0/24 documentation
      {0xC0A80000, 16}, // 192.168.0.0/16 private
      {0xC6120000, 15}, // 198.18.0.0/15 benchmarking
      {0xC6336400, 24}, // 198.51.100.0/24 documentation
      {0xCB007100, 24}, // 203.0.113.0/24 documentation
      {0xE0000000, 4},  // 224.0.0.0/4 multicast
      {0xF0000000, 4},  // 240.0.0.0/4 reserved and broadcast
  };
  for (const auto &vBlock : aBlocked) {
    const uint32_t cMask = ~uint32_t(0) << (32 - vBlock.vBits);
    if ((pAddr & cMask) == vBlock.vNet) {
      return false;
    }
  }
  return true;
}

// true if the IPv6 address is global unicast and, when it embeds an IPv4 address, that address is public too
static bool fRemoteIpv6IsPublic(const uint8_t *pAddr) {
  auto fEmbeddedIpv4 = [pAddr](size_t pOffset) {
    return (uint32_t(pAddr[pOffset]) << 24) | (uint32_t(pAddr[pOffset + 1]) << 16) |
           (uint32_t(pAddr[pOffset + 2]) << 8) | uint32_t(pAddr[pOffset + 3]);
  };
  static const uint8_t aMapped[12] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0xff, 0xff};      // ::ffff:0:0/96
  static const uint8_t aCompatible[12] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};        // ::/96, also :: and ::1
  static const uint8_t aNat64[12] = {0x00, 0x64, 0xff, 0x9b, 0, 0, 0, 0, 0, 0, 0, 0}; // 64:ff9b::/96
  if (memcmp(pAddr, aMapped, 12) == 0 || memcmp(pAddr, aCompatible, 12) == 0 || memcmp(pAddr, aNat64, 12) == 0) {
    return fRemoteIpv4IsPublic(fEmbeddedIpv4(12));
  }
  // everything outside 2000::/3 (ULA, link-local, site-local, multicast, discard, local NAT64...) is not public
  if ((pAddr[0] & 0xe0) != 0x20) {
    return false;
  }
  if (pAddr[0] == 0x20 && pAddr[1] == 0x01 && pAddr[2] == 0x00 && pAddr[3] == 0x00) {
    return false; // 2001::/32 Teredo
  }
  if (pAddr[0] == 0x20 && pAddr[1] == 0x01 && pAddr[2] == 0x0d && pAddr[3] == 0xb8) {
    return false; // 2001:db8::/32 documentation
  }
  if (pAddr[0] == 0x20 && pAddr[1] == 0x02) {
    return fRemoteIpv4IsPublic(fEmbeddedIpv4(2)); // 2002::/16 6to4
  }
  return true;
}

struct remote_resolved_address {
  std::string vIp;
  bool vPublic = false;
};

// resolves pHost once, waiting at most pTimeout; the lookup runs on its own thread because getaddrinfo cannot be
// interrupted, and a lookup that outlives the deadline finishes in the background and is discarded
static std::vector<remote_resolved_address> fRemoteResolveHost(const std::string &pHost,
                                                               std::chrono::milliseconds pTimeout) {
  auto vPromise = std::make_shared<std::promise<std::vector<remote_resolved_address>>>();
  auto vFuture = vPromise->get_future();
  std::thread([vPromise, pHost]() {
    try {
      addrinfo vHints = {};
      vHints.ai_family = AF_UNSPEC;
      vHints.ai_socktype = SOCK_STREAM;
      addrinfo *vList = nullptr;
      const int cRet = getaddrinfo(pHost.c_str(), nullptr, &vHints, &vList);
      if (cRet != 0) {
        throw std::runtime_error("error: cannot resolve host '" + pHost + "': " + gai_strerror(cRet));
      }
      std::vector<remote_resolved_address> lAddresses;
      for (addrinfo *vAi = vList; vAi != nullptr; vAi = vAi->ai_next) {
        char aIp[INET6_ADDRSTRLEN] = {};
        remote_resolved_address vAddress;
        if (vAi->ai_family == AF_INET) {
          const auto *cIn = reinterpret_cast<const sockaddr_in *>(vAi->ai_addr);
          inet_ntop(AF_INET, &cIn->sin_addr, aIp, sizeof(aIp));
          vAddress.vPublic = fRemoteIpv4IsPublic(ntohl(cIn->sin_addr.s_addr));
        } else if (vAi->ai_family == AF_INET6) {
          const auto *cIn6 = reinterpret_cast<const sockaddr_in6 *>(vAi->ai_addr);
          inet_ntop(AF_INET6, &cIn6->sin6_addr, aIp, sizeof(aIp));
          vAddress.vPublic = fRemoteIpv6IsPublic(reinterpret_cast<const uint8_t *>(&cIn6->sin6_addr));
        } else {
          continue;
        }
        vAddress.vIp = aIp;
        lAddresses.push_back(std::move(vAddress));
      }
      freeaddrinfo(vList);
      vPromise->set_value(std::move(lAddresses));
    } catch (...) {
      try {
        vPromise->set_exception(std::current_exception());
      } catch (...) {
      }
    }
  }).detach();
  if (vFuture.wait_for(pTimeout) != std::future_status::ready) {
    throw std::runtime_error("error: timed out resolving host '" + pHost + "'");
  }
  return vFuture.get();
}

// true if every character of the host is one that cpp-httplib keeps verbatim as its connection host, so the address
// pinned with set_hostname_addr_map is the one it actually connects to
static bool fRemoteHostIsPlain(const std::string &pHost) {
  if (pHost.empty()) {
    return false;
  }
  const bool cIpv6Literal = pHost.find(':') != std::string::npos;
  for (const char vC : pHost) {
    const bool cHex = (vC >= '0' && vC <= '9') || (vC >= 'a' && vC <= 'f') || (vC >= 'A' && vC <= 'F');
    if (cIpv6Literal) {
      if (!cHex && vC != ':') {
        return false;
      }
    } else if (!cHex && !(vC >= 'g' && vC <= 'z') && !(vC >= 'G' && vC <= 'Z') && vC != '.' && vC != '-' &&
               vC != '_') {
      return false;
    }
  }
  return true;
}

// absolute URL of a redirect Location relative to the URL that answered it; only same-origin targets keep credentials
static std::string fRemoteRedirectUrl(const common_http_url &pBase, const std::string &pLocation) {
  const std::string cLocation = pLocation.substr(0, pLocation.find('#'));
  const size_t cSchemeEnd = cLocation.find("://");
  bool vAbsolute = cSchemeEnd != std::string::npos && cSchemeEnd > 0;
  for (size_t vI = 0; vAbsolute && vI < cSchemeEnd; ++vI) {
    const char vC = cLocation[vI];
    vAbsolute = std::isalpha(static_cast<unsigned char>(vC)) ||
                (vI > 0 && (std::isdigit(static_cast<unsigned char>(vC)) || vC == '+' || vC == '-' || vC == '.'));
  }
  if (vAbsolute) {
    return cLocation;
  }
  if (cLocation.rfind("//", 0) == 0) {
    return pBase.vScheme + ":" + cLocation;
  }
  const std::string cCredentials =
      pBase.vUser.empty() ? "" : pBase.vUser + (pBase.vPassword.empty() ? "" : ":" + pBase.vPassword) + "@";
  const std::string cOrigin = pBase.vScheme + "://" + cCredentials + fCommonHttpFormatHost(pBase.vHost) + ":" +
                              std::to_string(pBase.vPort);
  if (!cLocation.empty() && cLocation.front() == '/') {
    return cOrigin + cLocation;
  }
  // relative reference: resolve it against the directory of the current path, without its query
  std::string vDirectory = pBase.vPath.substr(0, pBase.vPath.find('?'));
  vDirectory = vDirectory.substr(0, vDirectory.rfind('/') + 1);
  if (vDirectory.empty()) {
    vDirectory = "/";
  }
  return cOrigin + vDirectory + cLocation;
}

std::pair<long, std::vector<char>> fCommonRemoteGetContent(const std::string &pUrl,
                                                           const common_remote_params &pParams) {
  const auto cStart = std::chrono::steady_clock::now();
  const bool cHasDeadline = pParams.vTotalTimeout > 0;
  const auto cDeadline = cStart + std::chrono::seconds(pParams.vTotalTimeout);
  const bool cManualRedirects = pParams.vPublicHostsOnly || pParams.vMaxRedirects >= 0;
  const int cMaxRedirects = pParams.vMaxRedirects >= 0 ? pParams.vMaxRedirects : cRemoteDefaultMaxRedirects;

  // milliseconds left before the total deadline, or -1 when there is none
  auto fRemainingMs = [&]() -> long long {
    if (!cHasDeadline) {
      return -1;
    }
    const long long cLeft =
        std::chrono::duration_cast<std::chrono::milliseconds>(cDeadline - std::chrono::steady_clock::now()).count();
    if (cLeft <= 0) {
      throw std::runtime_error("error: remote request timed out after " + std::to_string(pParams.vTotalTimeout) +
                               " seconds");
    }
    return cLeft;
  };

  httplib::Headers vHeaders;
  for (const auto &vH : pParams.vHeaders) {
    vHeaders.emplace(vH.first, vH.second);
  }
  if (vHeaders.find("User-Agent") == vHeaders.end()) {
    vHeaders.emplace("User-Agent", "mgs/" + std::string(mgs_version()));
  }

  std::string vUrl = pUrl;
  for (int vRedirects = 0;; ++vRedirects) {
    auto [vCli, vParts] = fCommonHttpClient(vUrl);
    if (cManualRedirects) {
      vCli.set_follow_location(false);
    }

    long vConnectTimeout = pParams.vConnectTimeout;
    long vIoTimeout = pParams.vTimeout;
    long long vRemainingMs = fRemainingMs();

    if (pParams.vPublicHostsOnly) {
      if (!fRemoteHostIsPlain(vParts.vHost)) {
        throw common_remote_destination_error("error: unsupported characters in remote host '" + vParts.vHost + "'");
      }
      long long vResolveMs = vConnectTimeout > 0 ? vConnectTimeout * 1000LL : 30000LL;
      if (vRemainingMs >= 0) {
        vResolveMs = std::min(vResolveMs, vRemainingMs);
      }
      const auto lAddresses = fRemoteResolveHost(vParts.vHost, std::chrono::milliseconds(vResolveMs));
      if (lAddresses.empty()) {
        throw std::runtime_error("error: host '" + vParts.vHost + "' has no usable address");
      }
      // every address must be public, so the one the client would pick, now or after a DNS change, is never private
      for (const auto &vAddress : lAddresses) {
        if (!vAddress.vPublic) {
          throw common_remote_destination_error("error: host '" + vParts.vHost + "' resolves to " + vAddress.vIp +
                                                ", which is not a public address");
        }
      }
      vCli.set_hostname_addr_map({{vParts.vHost, lAddresses.front().vIp}});
      vRemainingMs = fRemainingMs();
    }

    if (vRemainingMs >= 0) {
      const long cRemainingSec = static_cast<long>(std::max<long long>(1, (vRemainingMs + 999) / 1000));
      vConnectTimeout = vConnectTimeout > 0 ? std::min(vConnectTimeout, cRemainingSec) : cRemainingSec;
      vIoTimeout = vIoTimeout > 0 ? std::min(vIoTimeout, cRemainingSec) : cRemainingSec;
      vCli.set_max_timeout(static_cast<time_t>(vRemainingMs));
    }
    if (vConnectTimeout > 0) {
      vCli.set_connection_timeout(vConnectTimeout, 0);
    }
    if (vIoTimeout > 0) {
      vCli.set_read_timeout(vIoTimeout, 0);
      vCli.set_write_timeout(vIoTimeout, 0);
    }

    std::vector<char> lBuf;
    bool vTooLarge = false;
    bool vTimedOut = false;
    auto vRes = vCli.Get(
        vParts.vPath, vHeaders,
        [&](const char *pData, size_t pLen) {
          if (cHasDeadline && std::chrono::steady_clock::now() >= cDeadline) {
            vTimedOut = true;
            return false;
          }
          lBuf.insert(lBuf.end(), pData, pData + pLen);
          if (pParams.vMaxSize != 0 && lBuf.size() > static_cast<size_t>(pParams.vMaxSize)) {
            vTooLarge = true;
            return false;
          }
          return true;
        },
        nullptr);

    if (!vRes) {
      if (vTooLarge) {
        throw std::runtime_error("error: remote content exceeds the maximum size of " +
                                 std::to_string(pParams.vMaxSize) + " bytes");
      }
      if (vTimedOut) {
        throw std::runtime_error("error: remote request timed out after " + std::to_string(pParams.vTotalTimeout) +
                                 " seconds");
      }
      throw std::runtime_error("error: cannot make GET request: " + httplib::to_string(vRes.error()));
    }

    if (cManualRedirects && vRes->status >= 300 && vRes->status < 400 && vRes->has_header("Location")) {
      if (vRedirects >= cMaxRedirects) {
        throw std::runtime_error("error: too many redirects (maximum " + std::to_string(cMaxRedirects) + ")");
      }
      vUrl = fRemoteRedirectUrl(vParts, vRes->get_header_value("Location"));
      continue;
    }

    return {vRes->status, std::move(lBuf)};
  }
}

int fCommonDownloadFileSingle(const std::string &pUrl, const std::string &pPath, const common_download_opts &pOpts,
                              bool pSkipEtag) {
  if (!pOpts.vOffline) {
    ProgressBar vTtyCb;
    common_download_opts vOnlineOpts = pOpts;
    if (!vOnlineOpts.vCallback) {
      vOnlineOpts.vCallback = &vTtyCb;
    }
    return fCommonDownloadFileSingleOnline(pUrl, pPath, vOnlineOpts, pSkipEtag);
  }

  if (!std::filesystem::exists(pPath)) {
    LOG_ERR("%s: required file is not available in cache (offline mode): %s\n", __func__, pPath.c_str());
    return -1;
  }

  LOG_DBG("%s: using cached file (offline mode): %s\n", __func__, pPath.c_str());

  // notify the callback that the file was cached
  if (pOpts.vCallback) {
    common_download_progress vP;
    vP.vUrl = pUrl;
    vP.vCached = true;
    pOpts.vCallback->mOnStart(vP);
    pOpts.vCallback->mOnDone(vP, true);
  }

  return 304; // Not Modified - fake cached response
}

struct gguf_split_info {
  std::string vPrefix; // tag included
  std::string vTag;
  int vIndex;
  int vCount;
};

static gguf_split_info fGetGgufSplitInfo(const std::string &pPath) {
  static const std::regex cReSplit("^(.+)-([0-9]{5})-of-([0-9]{5})$", std::regex::icase);
  static const std::regex cReTag("[-.]([A-Z0-9_]+)$", std::regex::icase);
  std::smatch vM;

  std::string vPrefix = pPath;
  if (!fStringRemoveSuffix(vPrefix, ".gguf")) {
    return {};
  }

  int vIndex = 1;
  int vCount = 1;

  if (std::regex_match(vPrefix, vM, cReSplit)) {
    vIndex = std::stoi(vM[2].str());
    vCount = std::stoi(vM[3].str());
    vPrefix = vM[1].str();
  }

  std::string vTag;
  if (std::regex_search(vPrefix, vM, cReTag)) {
    vTag = vM[1].str();
    for (char &vC : vTag) {
      vC = std::toupper((unsigned char)vC);
    }
  }

  return {std::move(vPrefix), std::move(vTag), vIndex, vCount};
}

// Q4_0 -> 4, F16 -> 16, NVFP4 -> 4, Q8_K_M -> 8, etc
static int fExtractQuantBits(const std::string &pFilename) {
  auto vSplit = fGetGgufSplitInfo(pFilename);

  auto vPos = vSplit.vTag.find_first_of("0123456789");
  if (vPos == std::string::npos) {
    return 0;
  }

  return std::stoi(vSplit.vTag.substr(vPos));
}

static hf_cache::hf_files fGetSplitFiles(const hf_cache::hf_files &pFiles, const hf_cache::vHfFile &pFile) {
  auto vSplit = fGetGgufSplitInfo(pFile.vPath);

  if (vSplit.vCount <= 1) {
    return {pFile};
  }
  hf_cache::hf_files vResult;

  for (const auto &vF : pFiles) {
    auto vSplitF = fGetGgufSplitInfo(vF.vPath);
    if (vSplitF.vCount == vSplit.vCount && vSplitF.vPrefix == vSplit.vPrefix) {
      vResult.push_back(vF);
    }
  }
  return vResult;
}

// pick the best sibling GGUF whose filename contains `pKeyword` (e.g. "mmproj" / "mtp"),
// preferring deeper shared directory prefix with the model, then exact `pTag` match,
// then closest quantization to the tag when given, or to the model otherwise
static hf_cache::vHfFile fFindBestSibling(const hf_cache::hf_files &pFiles, const std::string &pModel,
                                          const std::string &pKeyword, const std::string &pTag = "") {
  hf_cache::vHfFile vBest;
  size_t vBestDepth = 0;
  int vBestDiff = 0;
  bool vBestExact = false;
  bool vFound = false;

  std::string vTagUpper = pTag;
  for (char &vC : vTagUpper) {
    vC = (char)std::toupper((unsigned char)vC);
  }

  int vModelBits = 0;
  if (!vTagUpper.empty()) {
    auto vPos = vTagUpper.find_first_of("0123456789");
    vModelBits = vPos == std::string::npos ? 0 : std::stoi(vTagUpper.substr(vPos));
  } else {
    vModelBits = fExtractQuantBits(pModel);
  }
  auto vModelParts = fStringSplit<std::string>(pModel, '/');
  auto vModelDir = vModelParts.end() - 1;

  for (const auto &vF : pFiles) {
    if (!fStringEndsWith(vF.vPath, ".gguf") || vF.vPath.find(pKeyword) == std::string::npos) {
      continue;
    }

    auto vSibParts = fStringSplit<std::string>(vF.vPath, '/');
    auto vSibDir = vSibParts.end() - 1;

    auto [_, vDir] = std::mismatch(vModelParts.begin(), vModelDir, vSibParts.begin(), vSibDir);
    if (vDir != vSibDir) {
      continue;
    }

    size_t vDepth = vDir - vSibParts.begin();
    auto vBits = fExtractQuantBits(vF.vPath);
    auto vDiff = std::abs(vBits - vModelBits);

    std::string vPathUpper = vF.vPath;
    for (char &vC : vPathUpper) {
      vC = (char)std::toupper((unsigned char)vC);
    }
    bool vExact = !vTagUpper.empty() && vPathUpper.find("-" + vTagUpper + ".") != std::string::npos;

    if (!vFound || vDepth > vBestDepth || (vDepth == vBestDepth && vExact && !vBestExact) ||
        (vDepth == vBestDepth && vExact == vBestExact && vDiff < vBestDiff)) {
      vBest = vF;
      vBestDepth = vDepth;
      vBestDiff = vDiff;
      vBestExact = vExact;
      vFound = true;
    }
  }
  return vBest;
}

static hf_cache::vHfFile fFindBestMmproj(const hf_cache::hf_files &pFiles, const std::string &pModel) {
  return fFindBestSibling(pFiles, pModel, "mmproj");
}

static hf_cache::vHfFile fFindBestMtp(const hf_cache::hf_files &pFiles, const std::string &pModel,
                                      const std::string &pTag = "") {
  return fFindBestSibling(pFiles, pModel, "mtp-", pTag);
}

static hf_cache::vHfFile fFindBestEagle3(const hf_cache::hf_files &pFiles, const std::string &pModel,
                                         const std::string &pTag = "") {
  return fFindBestSibling(pFiles, pModel, "eagle3-", pTag);
}

static hf_cache::vHfFile fFindBestDflash(const hf_cache::hf_files &pFiles, const std::string &pModel,
                                         const std::string &pTag = "") {
  return fFindBestSibling(pFiles, pModel, "dflash-", pTag);
}

static hf_cache::vHfFile fFindBestDspark(const hf_cache::hf_files &pFiles, const std::string &pModel,
                                         const std::string &pTag = "") {
  return fFindBestSibling(pFiles, pModel, "dspark-", pTag);
}

static bool fGgufFilenameIsModel(const std::string &pFilepath) {
  if (!fStringEndsWith(pFilepath, ".gguf")) {
    return false;
  }

  std::string vFilename = pFilepath;
  if (auto vPos = vFilename.rfind('/'); vPos != std::string::npos) {
    vFilename = vFilename.substr(vPos + 1);
  }

  return vFilename.find("mmproj") == std::string::npos && vFilename.find("imatrix") == std::string::npos &&
         vFilename.find("mtp-") == std::string::npos && vFilename.find("eagle3-") == std::string::npos &&
         vFilename.find("dflash-") == std::string::npos && vFilename.find("dspark-") == std::string::npos;
}

static hf_cache::vHfFile fFindBestModel(const hf_cache::hf_files &pFiles, const std::string &pTag) {
  std::vector<std::string> lTags;

  if (!pTag.empty()) {
    lTags.push_back(pTag);
  } else {
    lTags = {"Q4_K_M", "Q8_0"};
  }

  for (const auto &vT : lTags) {
    std::regex vPattern(vT + "[.-]", std::regex::icase);
    for (const auto &vF : pFiles) {
      if (fGgufFilenameIsModel(vF.vPath) && std::regex_search(vF.vPath, vPattern)) {
        auto vSplit = fGetGgufSplitInfo(vF.vPath);
        if (vSplit.vCount > 1 && vSplit.vIndex != 1) {
          continue;
        }
        return vF;
      }
    }
  }

  // fallback to first available model only if tag is empty
  if (pTag.empty()) {
    for (const auto &vF : pFiles) {
      if (fGgufFilenameIsModel(vF.vPath)) {
        auto vSplit = fGetGgufSplitInfo(vF.vPath);
        if (vSplit.vCount > 1 && vSplit.vIndex != 1) {
          continue;
        }
        return vF;
      }
    }
  }

  return {};
}

static void fListAvailableGgufFiles(const hf_cache::hf_files &pFiles) {
  LOG_INF("Available GGUF files:\n");
  for (const auto &vF : pFiles) {
    if (fStringEndsWith(vF.vPath, ".gguf")) {
      LOG_INF(" - %s\n", vF.vPath.c_str());
    }
  }
}

common_download_hf_plan fCommonDownloadGetHfPlan(const common_params_model &pModel, const common_download_opts &pOpts) {
  common_download_hf_plan vPlan;
  hf_cache::hf_files vAll;

  auto [vRepo, vTag] = fCommonDownloadSplitRepoTag(pModel.vHfRepo);

  if (!pOpts.vOffline) {
    vAll = hf_cache::fGetRepoFiles(vRepo, pOpts.vBearerToken);
  }
  if (vAll.empty()) {
    vAll = hf_cache::fGetCachedFiles(vRepo);
  }
  if (vAll.empty()) {
    return vPlan;
  }

  // if preset.ini exists in the repo root, download only that file
  for (const auto &vF : vAll) {
    if (vF.vPath == "preset.ini") {
      vPlan.vPreset = vF;
      return vPlan;
    }
  }

  hf_cache::vHfFile vPrimary;

  if (!pModel.vHfFile.empty()) {
    for (const auto &vF : vAll) {
      if (vF.vPath == pModel.vHfFile) {
        vPrimary = vF;
        break;
      }
    }
    if (vPrimary.vPath.empty()) {
      LOG_ERR("%s: file '%s' not found in repository\n", __func__, pModel.vHfFile.c_str());
      fListAvailableGgufFiles(vAll);
      return vPlan;
    }
  } else {
    vPrimary = fFindBestModel(vAll, vTag);
    // a requested sidecar can resolve on its own, without a full model of the same tag
    if (vPrimary.vPath.empty() && !pOpts.vDownloadMtp && !pOpts.vDownloadDflash && !pOpts.vDownloadEagle3 &&
        !pOpts.vDownloadDspark) {
      LOG_ERR("%s: no GGUF files found in repository %s\n", __func__, vRepo.c_str());
      fListAvailableGgufFiles(vAll);
      return vPlan;
    }
  }

  if (!vPrimary.vPath.empty()) {
    vPlan.vPrimary = vPrimary;
    vPlan.vModelFiles = fGetSplitFiles(vAll, vPrimary);
  }

  if (pOpts.vDownloadMmproj && !vPrimary.vPath.empty()) {
    vPlan.vMmproj = fFindBestMmproj(vAll, vPrimary.vPath);
  }
  if (pOpts.vDownloadMtp) {
    vPlan.vMtp = fFindBestMtp(vAll, vPrimary.vPath, vTag);
  }
  if (pOpts.vDownloadDflash) {
    vPlan.vDflash = fFindBestDflash(vAll, vPrimary.vPath, vTag);
  }
  if (pOpts.vDownloadEagle3) {
    vPlan.vEagle3 = fFindBestEagle3(vAll, vPrimary.vPath, vTag);
  }
  if (pOpts.vDownloadDspark) {
    vPlan.vDspark = fFindBestDspark(vAll, vPrimary.vPath, vTag);
  }

  if (vPrimary.vPath.empty() && vPlan.vMtp.vLocalPath.empty() && vPlan.vDflash.vLocalPath.empty() &&
      vPlan.vEagle3.vLocalPath.empty() && vPlan.vDspark.vLocalPath.empty()) {
    LOG_ERR("%s: no GGUF files found in repository %s\n", __func__, vRepo.c_str());
    fListAvailableGgufFiles(vAll);
  }

  return vPlan;
}

void fCommonDownloadRunTasks(const std::vector<common_download_task> &pTasks) {
  std::vector<std::future<int>> lFutures;
  for (const auto &vTask : pTasks) {
    lFutures.push_back(std::async(std::launch::async, [&vTask]() {
      return fCommonDownloadFileSingle(vTask.vUrl, vTask.vLocalPath, vTask.vOpts, vTask.vIsHf);
    }));
  }

  for (size_t vI = 0; vI < lFutures.size(); ++vI) {
    std::string vUrl = pTasks[vI].vUrl;
    int vStatus = lFutures[vI].get();
    bool vIsOk = fIsHttpStatusOk(vStatus);
    if (!vIsOk) {
      throw std::runtime_error(fStringFormat("Download '%s' failed with status code: %d", vUrl.c_str(), vStatus));
    }
  }
}

std::vector<std::string> fCommonDownloadGetAllParts(const std::string &pUrl) {
  auto vSplit = fGetGgufSplitInfo(pUrl);

  if (vSplit.vCount <= 1) {
    return {pUrl};
  }

  std::vector<std::string> lParts;
  for (int vI = 1; vI <= vSplit.vCount; vI++) {
    auto vSuffix = fStringFormat("-%05d-of-%05d.gguf", vI, vSplit.vCount);
    lParts.push_back(vSplit.vPrefix + vSuffix);
  }
  return lParts;
}

//
// Docker registry functions
//

static std::string fCommonDockerGetToken(const std::string &pRepo) {
  std::string vUrl = "https://auth.docker.io/token?service=registry.docker.io&scope=repository:" + pRepo + ":pull";

  common_remote_params vParams;
  auto vRes = fCommonRemoteGetContent(vUrl, vParams);

  if (vRes.first != 200) {
    throw std::runtime_error("Failed to get Docker registry token, HTTP code: " + std::to_string(vRes.first));
  }

  std::string vResponseStr(vRes.second.begin(), vRes.second.end());
  nlohmann::ordered_json vResponse = nlohmann::ordered_json::parse(vResponseStr);

  if (!vResponse.contains("token")) {
    throw std::runtime_error("Docker registry token response missing 'token' field");
  }

  return vResponse["token"].get<std::string>();
}

std::string fCommonDockerResolveModel(const std::string &pDocker) {
  // Parse ai/smollm2:135M-Q4_0
  size_t vColonPos = pDocker.find(':');
  std::string vRepo, vTag;
  if (vColonPos != std::string::npos) {
    vRepo = pDocker.substr(0, vColonPos);
    vTag = pDocker.substr(vColonPos + 1);
  } else {
    vRepo = pDocker;
    vTag = "latest";
  }

  // ai/ is the default
  size_t vSlashPos = pDocker.find('/');
  if (vSlashPos == std::string::npos) {
    vRepo.insert(0, "ai/");
  }

  LOG_INF("%s: Downloading Docker Model: %s:%s\n", __func__, vRepo.c_str(), vTag.c_str());
  try {
    // --- helper: digest validation ---
    auto vValidateOciDigest = [](const std::string &pDigest) -> std::string {
      // Expected: algo:hex ; start with sha256 (64 hex chars)
      // You can extend this map if supporting other algorithms in future.
      static const std::regex cRe("^sha256:([a-fA-F0-9]{64})$");
      std::smatch vM;
      if (!std::regex_match(pDigest, vM, cRe)) {
        throw std::runtime_error("Invalid OCI digest format received in manifest: " + pDigest);
      }
      // normalize hex to lowercase
      std::string vNormalized = pDigest;
      std::transform(vNormalized.begin() + 7, vNormalized.end(), vNormalized.begin() + 7,
                     [](unsigned char pC) { return std::tolower(pC); });
      return vNormalized;
    };

    std::string vToken = fCommonDockerGetToken(vRepo); // Get authentication token

    // Get manifest
    // TODO: cache the manifest response so that it appears in the model list
    const std::string cUrlPrefix = "https://registry-1.docker.io/v2/" + vRepo;
    std::string vManifestUrl = cUrlPrefix + "/manifests/" + vTag;
    common_remote_params vManifestParams;
    vManifestParams.vHeaders.push_back({"Authorization", "Bearer " + vToken});
    vManifestParams.vHeaders.push_back(
        {"Accept", "application/vnd.docker.distribution.manifest.v2+json,application/vnd.oci.image.manifest.v1+json"});
    auto vManifestRes = fCommonRemoteGetContent(vManifestUrl, vManifestParams);
    if (vManifestRes.first != 200) {
      throw std::runtime_error("Failed to get Docker manifest, HTTP code: " + std::to_string(vManifestRes.first));
    }

    std::string vManifestStr(vManifestRes.second.begin(), vManifestRes.second.end());
    nlohmann::ordered_json vManifest = nlohmann::ordered_json::parse(vManifestStr);
    std::string vGgufDigest; // Find the GGUF layer
    if (vManifest.contains("layers")) {
      for (const auto &vLayer : vManifest["layers"]) {
        if (vLayer.contains("mediaType")) {
          std::string vMediaType = vLayer["mediaType"].get<std::string>();
          if (vMediaType == "application/vnd.docker.ai.gguf.v3" || vMediaType.find("gguf") != std::string::npos) {
            vGgufDigest = vLayer["digest"].get<std::string>();
            break;
          }
        }
      }
    }

    if (vGgufDigest.empty()) {
      throw std::runtime_error("No GGUF layer found in Docker manifest");
    }

    // Validate & normalize digest
    vGgufDigest = vValidateOciDigest(vGgufDigest);
    LOG_DBG("%s: Using validated digest: %s\n", __func__, vGgufDigest.c_str());

    // Prepare local filename
    std::string vModelFilename = vRepo;
    std::replace(vModelFilename.begin(), vModelFilename.end(), '/', '_');
    vModelFilename += "_" + vTag + ".gguf";
    std::string vLocalPath = fFsGetCacheFile(vModelFilename);

    const std::string cBlobUrl = cUrlPrefix + "/blobs/" + vGgufDigest;
    common_download_opts vOpts;
    vOpts.vBearerToken = vToken;
    const int cHttpStatus = fCommonDownloadFileSingle(cBlobUrl, vLocalPath, vOpts);
    if (!fIsHttpStatusOk(cHttpStatus)) {
      throw std::runtime_error("Failed to download Docker Model");
    }

    LOG_INF("%s: Downloaded Docker Model to: %s\n", __func__, vLocalPath.c_str());
    return vLocalPath;
  } catch (const std::exception &vE) {
    LOG_ERR("%s: Docker Model download failed: %s\n", __func__, vE.what());
    throw;
  }
}

std::vector<common_cached_model_info> fCommonListCachedModels() {
  std::unordered_set<std::string> sSeen;
  std::vector<common_cached_model_info> lResult;

  auto vFiles = hf_cache::fGetCachedFiles();

  for (const auto &vF : vFiles) {
    auto vSplit = fGetGgufSplitInfo(vF.vPath);
    if (vSplit.vIndex != 1 || vSplit.vTag.empty() || vSplit.vPrefix.find("mmproj") != std::string::npos ||
        vSplit.vPrefix.find("mtp-") != std::string::npos || vSplit.vPrefix.find("eagle3-") != std::string::npos ||
        vSplit.vPrefix.find("dflash-") != std::string::npos || vSplit.vPrefix.find("dspark-") != std::string::npos) {
      continue;
    }
    if (sSeen.insert(vF.vRepoId + ":" + vSplit.vTag).second) {
      lResult.push_back({vF.vRepoId, vSplit.vTag});
    }
  }

  return lResult;
}

bool fCommonDownloadRemove(const std::string &pHfRepoWithTag) {
  namespace fs = std::filesystem;

  auto [vRepoId, vTag] = fCommonDownloadSplitRepoTag(pHfRepoWithTag);

  if (vTag.empty()) {
    return hf_cache::fRemoveCachedRepo(vRepoId);
  }

  std::string vTagUpper = vTag;
  for (char &vC : vTagUpper) {
    vC = (char)std::toupper((unsigned char)vC);
  }

  auto vFiles = hf_cache::fGetCachedFiles(vRepoId);
  if (vFiles.empty()) {
    return false;
  }

  // collect snapshot entries whose tag matches
  std::vector<fs::path> lToRemove;
  for (const auto &vF : vFiles) {
    auto vSplit = fGetGgufSplitInfo(vF.vPath);
    if (vSplit.vTag == vTagUpper) {
      lToRemove.emplace_back(vF.vLocalPath);
    }
  }

  if (lToRemove.empty()) {
    return false;
  }

  // resolve blob paths from symlinks before deleting snapshot entries
  std::vector<fs::path> lBlobsToCheck;
  for (const auto &vP : lToRemove) {
    std::error_code vEc;
    if (fs::is_symlink(vP, vEc)) {
      auto vTarget = fs::read_symlink(vP, vEc);
      if (!vEc) {
        lBlobsToCheck.push_back((vP.parent_path() / vTarget).lexically_normal());
      }
    }
  }

  // remove snapshot entries
  for (const auto &vP : lToRemove) {
    std::error_code vEc;
    fs::remove(vP, vEc);
    if (vEc) {
      LOG_WRN("%s: failed to remove %s: %s\n", __func__, vP.string().c_str(), vEc.message().c_str());
    }
  }

  if (lBlobsToCheck.empty()) {
    return true;
  }

  // collect blobs still referenced by remaining snapshot entries
  std::unordered_set<std::string> sStillReferenced;
  for (const auto &vF : hf_cache::fGetCachedFiles(vRepoId)) {
    fs::path vP(vF.vLocalPath);
    std::error_code vEc;
    if (fs::is_symlink(vP, vEc)) {
      auto vTarget = fs::read_symlink(vP, vEc);
      if (!vEc) {
        sStillReferenced.insert((vP.parent_path() / vTarget).lexically_normal().string());
      }
    }
  }

  // remove orphaned blobs
  for (const auto &vBlob : lBlobsToCheck) {
    if (sStillReferenced.find(vBlob.string()) == sStillReferenced.end()) {
      std::error_code vEc;
      fs::remove(vBlob, vEc);
      if (vEc) {
        LOG_WRN("%s: failed to remove blob %s: %s\n", __func__, vBlob.string().c_str(), vEc.message().c_str());
      }
    }
  }

  return true;
}
