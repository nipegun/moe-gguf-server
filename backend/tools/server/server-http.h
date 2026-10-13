#pragma once

#include <atomic>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <thread>
#include <vector>
#include <cstdint>
#include <unordered_map>
#include <set>

struct common_params;

// generator-like API for HTTP response generation
// this object response with one of the 2 modes:
// 1) normal response: `data` contains the full response body
// 2) streaming response: each call to next(output) generates the next chunk
//    when next(output) returns false, no more data after the current chunk
//    note: some chunks can be empty, in which case no data is sent for that chunk
struct server_http_res {
  std::string vContentType = "application/json; charset=utf-8";
  int vStatus = 200;
  std::string vData;
  std::map<std::string, std::string> dHeaders;

  std::function<bool(std::string &)> vNext = nullptr;
  bool mIsStream() const { return vNext != nullptr; }

  // fired before req and res are destroyed
  virtual void mOnComplete() {}

  virtual ~server_http_res() = default;
};

// unique pointer, used by set_chunked_content_provider
// httplib requires the stream provider to be stored in heap
using server_http_res_ptr = std::unique_ptr<server_http_res>;
using vRawBuffer = std::vector<uint8_t>;

struct uploaded_file {
  vRawBuffer vData;
  std::string vFilename;
  std::string vContentType;
};

struct server_http_req {
  std::map<std::string, std::string> dParams;  // path_params + query_params
  std::map<std::string, std::string> dHeaders; // used by MCP proxy
  std::string vPath;
  std::string vQueryString; // query parameters string (e.g. "action=save")
  std::string vBody;
  std::map<std::string, uploaded_file> dFiles; // used for file uploads (form data)
  const std::function<bool()> &vShouldStop;

  std::string mGetParam(const std::string &pKey, const std::string &pDef = "") const {
    auto vIt = dParams.find(pKey);
    if (vIt != dParams.end()) {
      return vIt->second;
    }
    return pDef;
  }
};

struct server_http_context {
  class Impl;
  std::unique_ptr<Impl> vPimpl;

  std::thread vThread; // server thread
  std::atomic<bool> vIsReady = false;

  // note: the handler should never throw exceptions
  using handler_t = std::function<server_http_res_ptr(const server_http_req &pReq)>;
  mutable std::unordered_map<std::string, handler_t> dHandlers;
  mutable std::map<std::string, std::set<std::string>> dApiMethods;

  std::string vPathPrefix;
  std::string vHostname;
  int vPort = 11443;
  bool vIsSsl = false;

  server_http_context();
  ~server_http_context();

  bool mInit(const common_params &pParams);
  bool mStart();
  void mStop() const;

  void mGet(const std::string &pPath, const handler_t &pHandler) const;
  void mPost(const std::string &pPath, const handler_t &pHandler) const;
  void mDel(const std::string &pPath, const handler_t &pHandler) const;

  // Register the Google Cloud Platform (Vertex AI) compat (AIP_PREDICT_ROUTE env var, or /predict)
  // Must be called AFTER all other API routes are registered
  void mRegisterGcpCompat() const;

  // for debugging
  std::string vListeningAddress;
};
