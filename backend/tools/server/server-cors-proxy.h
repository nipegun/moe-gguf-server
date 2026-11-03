#pragma once

#include "common.h"
#include "http.h"

#include <string>
#include <unordered_set>
#include <list>
#include <map>
#include <algorithm>
#include <cctype>

#include "server-http.h"

static std::string fProxyHeaderToLower(std::string pHeader) {
  std::transform(pHeader.begin(), pHeader.end(), pHeader.begin(), [](unsigned char pC) { return std::tolower(pC); });
  return pHeader;
}

// Header lookup without regard to case.
static std::string fProxyGetHeader(const std::map<std::string, std::string> &pHeaders, const std::string &pName) {
  const std::string cName = fProxyHeaderToLower(pName);
  for (const auto &[vKey, vValue] : pHeaders) {
    if (fProxyHeaderToLower(vKey) == cName) {
      return vValue;
    }
  }
  return "";
}

// The proxy answers on the origin of the Web UI, so a page that navigated to it or embedded it would run the remote
// content with that origin. Only fetch() calls are served; browsers that predate Fetch Metadata still get an inert
// response (see fProxyMakeInert).
static bool fProxyIsNavigation(const server_http_req &pReq) {
  static const std::unordered_set<std::string> sBlockedDestinations = {
      "document", "iframe", "frame", "embed", "object", "worker", "sharedworker", "serviceworker"};
  const std::string cMode = fProxyHeaderToLower(fProxyGetHeader(pReq.dHeaders, "Sec-Fetch-Mode"));
  const std::string cDestination = fProxyHeaderToLower(fProxyGetHeader(pReq.dHeaders, "Sec-Fetch-Dest"));
  return cMode == "navigate" || cMode == "nested-navigate" || sBlockedDestinations.count(cDestination) > 0;
}

// Drops the upstream headers that would act on the origin of the Web UI (cookies, CORS, CSP, redirects, service
// workers, HSTS...) and forces the content to stay inert even if it is opened as a page.
static void fProxyMakeInert(server_http_res &pRes) {
  static const std::unordered_set<std::string> sBlockedHeaders = {"set-cookie",
                                                                  "set-cookie2",
                                                                  "service-worker-allowed",
                                                                  "clear-site-data",
                                                                  "refresh",
                                                                  "link",
                                                                  "location",
                                                                  "content-location",
                                                                  "content-disposition",
                                                                  "content-security-policy",
                                                                  "content-security-policy-report-only",
                                                                  "x-content-type-options",
                                                                  "x-frame-options",
                                                                  "strict-transport-security",
                                                                  "public-key-pins",
                                                                  "public-key-pins-report-only",
                                                                  "expect-ct",
                                                                  "permissions-policy",
                                                                  "feature-policy",
                                                                  "origin-agent-cluster",
                                                                  "cross-origin-opener-policy",
                                                                  "cross-origin-embedder-policy",
                                                                  "cross-origin-resource-policy",
                                                                  "timing-allow-origin",
                                                                  "report-to",
                                                                  "reporting-endpoints",
                                                                  "nel",
                                                                  "alt-svc",
                                                                  "x-dns-prefetch-control",
                                                                  "x-xss-protection"};
  for (auto vIt = pRes.dHeaders.begin(); vIt != pRes.dHeaders.end();) {
    const std::string cLowered = fProxyHeaderToLower(vIt->first);
    if (sBlockedHeaders.count(cLowered) > 0 || cLowered.rfind("access-control-", 0) == 0) {
      vIt = pRes.dHeaders.erase(vIt);
    } else {
      ++vIt;
    }
  }
  pRes.dHeaders["Content-Security-Policy"] = "sandbox; default-src 'none'";
  pRes.dHeaders["X-Content-Type-Options"] = "nosniff";
  pRes.dHeaders["Content-Disposition"] = "attachment";
  pRes.dHeaders["Cross-Origin-Resource-Policy"] = "same-origin";
}

static server_http_res_ptr fProxyRequest(const server_http_req &pReq, std::string pMethod) {
  if (fProxyIsNavigation(pReq)) {
    auto vRes = std::make_unique<server_http_res>();
    vRes->vStatus = 403;
    vRes->vData = fSafeJsonToStr(
        {{"error",
          {{"message", "the CORS proxy only serves fetch() requests, not navigations or embedded documents"},
           {"type", "permission_error"}}}});
    fProxyMakeInert(*vRes);
    return vRes;
  }

  std::string vTargetUrl = pReq.mGetParam("url");
  common_http_url vParsedUrl = fCommonHttpParseUrl(vTargetUrl);

  if (vParsedUrl.vHost.empty()) {
    throw std::runtime_error("invalid target URL: missing host");
  }

  if (vParsedUrl.vPath.empty()) {
    vParsedUrl.vPath = "/";
  }

  if (!vParsedUrl.vPassword.empty()) {
    throw std::runtime_error("authentication in target URL is not supported");
  }

  if (vParsedUrl.vScheme != "http" && vParsedUrl.vScheme != "https") {
    throw std::runtime_error("unsupported URL scheme in target URL: " + vParsedUrl.vScheme);
  }

  SRV_INF("proxying %s request to %s://%s:%i%s\n", pMethod.c_str(), vParsedUrl.vScheme.c_str(),
          fCommonHttpFormatHost(vParsedUrl.vHost).c_str(), vParsedUrl.vPort, vParsedUrl.vPath.c_str());

  std::map<std::string, std::string> dHeaders;
  const std::string cProxyHeaderPrefix = "x-mgs-proxy-header-";
  for (auto [vKey, vValue] : pReq.dHeaders) {
    const std::string cLoweredKey = fProxyHeaderToLower(vKey);
    if (!fStringStartsWith(cLoweredKey, cProxyHeaderPrefix)) {
      continue;
    }

    auto vNewKey = vKey.substr(cProxyHeaderPrefix.size());
    if (vNewKey.empty()) {
      continue;
    }

    dHeaders[vNewKey] = vValue;
  }

  auto vProxy =
      std::make_unique<server_http_proxy>(pMethod, vParsedUrl.vScheme, vParsedUrl.vHost, vParsedUrl.vPort,
                                          vParsedUrl.vPath, dHeaders, pReq.vBody, pReq.dFiles, pReq.vShouldStop,
                                          600, // timeout_read (default to 10 minutes)
                                          600  // timeout_write (default to 10 minutes)
      );
  // the constructor returns once the upstream headers have arrived
  fProxyMakeInert(*vProxy);

  return vProxy;
}

static server_http_context::handler_t vProxyHandlerPost = [](const server_http_req &pReq) -> server_http_res_ptr {
  return fProxyRequest(pReq, "POST");
};

static server_http_context::handler_t vProxyHandlerGet = [](const server_http_req &pReq) -> server_http_res_ptr {
  return fProxyRequest(pReq, "GET");
};
