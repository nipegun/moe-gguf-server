#include "common.h"
#include "http.h"
#include "server-http.h"
#include "server-common.h"
#include "ui.h"
#include "server-api-doc.h"
#include "server-tls.h"
#include "server-ui-settings.h"

#include <cpp-httplib/httplib.h>

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <functional>
#include <future>
#include <memory>
#include <regex>
#include <string>
#include <thread>

// the listener reads from a client only while it receives a request (request line, headers and body); a long
// generation only writes, so a short read timeout cannot cut it, while --timeout keeps governing the writes and the
// router's proxy to its model processes. cpp-httplib applies it to every read, not to the whole header block
static constexpr int32_t cHttpRequestReadTimeoutSec = 30;
// accepted sockets that may wait for a free HTTP thread once every thread is busy
static constexpr size_t cHttpMaxQueuedConnections = 256;
// deepest array/object nesting accepted in a request body: the JSON library copies, dumps and compares documents
// recursively, so a body of nested brackets could exhaust a thread's stack in any handler
static constexpr size_t cHttpMaxRequestJsonDepth = 1024;

// scans a request body as JSON text without parsing it and returns true if its array/object nesting exceeds
// pMaxDepth; brackets inside strings are ignored
static bool fJsonNestingExceeds(const std::string &pBody, size_t pMaxDepth) {
  size_t vDepth = 0;
  bool vInString = false;
  bool vEscaped = false;
  for (const char vChar : pBody) {
    if (vInString) {
      if (vEscaped) {
        vEscaped = false;
      } else if (vChar == '\\') {
        vEscaped = true;
      } else if (vChar == '"') {
        vInString = false;
      }
      continue;
    }
    if (vChar == '"') {
      vInString = true;
    } else if (vChar == '[' || vChar == '{') {
      if (++vDepth > pMaxDepth) {
        return true;
      }
    } else if ((vChar == ']' || vChar == '}') && vDepth > 0) {
      --vDepth;
    }
  }
  return false;
}

//
// HTTP implementation using cpp-httplib
//

class server_http_context::Impl {
public:
  std::unique_ptr<httplib::Server> vSrv;
  std::unique_ptr<httplib::Server> vRedirectServer;
  std::thread vRedirectThread;
  int vHttpPort = 11080;
  bool vAutoPort = true;
};

server_http_context::server_http_context() : vPimpl(std::make_unique<Impl>()) {}

server_http_context::~server_http_context() {
  mStop();
  if (vThread.joinable()) {
    vThread.join();
  }
  if (vPimpl->vRedirectThread.joinable()) {
    vPimpl->vRedirectThread.join();
  }
}

// Bind the actual listening socket, so another process cannot claim a selected port.
static bool fBindServerPort(httplib::Server &pServer, const std::string &pHost, int &pPort, bool pAutomatic,
                            int pExcludedPort = -1) {
  if (pPort == 0) {
    pPort = pServer.bind_to_any_port(pHost);
    return pPort > 0;
  }
  if (pPort < 1 || pPort > 65535) {
    return false;
  }
  bool vBound = false;
  bool vConflict = false;
  bool vOtherError = false;
  pServer.set_error_logger([&vConflict, &vOtherError](const httplib::Error &pError, const httplib::Request *) {
#ifdef _WIN32
    const bool vAddressInUse = WSAGetLastError() == WSAEADDRINUSE;
#else
    const bool vAddressInUse = errno == EADDRINUSE;
#endif
    if (pError == httplib::Error::BindIPAddress && vAddressInUse) {
      vConflict = true;
    } else {
      vOtherError = true;
    }
  });
  while (pPort <= 65535) {
    if (pPort != pExcludedPort) {
      vConflict = false;
      vOtherError = false;
      if (pServer.bind_to_port(pHost, pPort)) {
        vBound = true;
        break;
      }
      if (!pAutomatic || !vConflict || vOtherError) {
        break;
      }
    } else if (!pAutomatic) {
      break;
    }
    ++pPort;
  }
  pServer.set_error_logger({});
  return vBound;
}

static void fLogServerRequest(const httplib::Request &pReq, const httplib::Response &pRes) {
  // skip logging requests that are regularly sent, to avoid log spam
  if (pReq.path == "/health" || pReq.path == "/v1/health" || pReq.path == "/models" || pReq.path == "/v1/models" ||
      pReq.path == "/props" || pReq.path == "/metrics") {
    return;
  }

  // reminder: this function is not covered by httplib's exception handler; if someone does more complicated stuff,
  // think about wrapping it in try-catch

  SRV_TRC("done request: %s %s %s %d\n", pReq.method.c_str(), pReq.path.c_str(), pReq.remote_addr.c_str(), pRes.status);

  SRV_DBG("request:  %s\n", pReq.body.c_str());
  SRV_DBG("response: %s\n", pRes.body.c_str());
}

// returns true if the Origin header value's host is localhost / 127.0.0.1 / ::1 (any port)
static bool fOriginIsLocalhost(const std::string &pOrigin) {
  try {
    const std::string cHost = fCommonHttpParseUrl(pOrigin).vHost;
    return cHost == "localhost" || cHost == "127.0.0.1" || cHost == "::1";
  } catch (const std::exception &) {
    return false;
  }
}

// "https://Host:443/path" -> "host"; "host:11443" -> "host:11443": default ports are dropped so an Origin and a Host
// header naming the same server compare equal
static std::string fNormalizeAuthority(const std::string &pValue, std::string pScheme) {
  std::string vRest = pValue;
  const size_t cSeparator = vRest.find("://");
  if (cSeparator != std::string::npos) {
    pScheme = vRest.substr(0, cSeparator);
    vRest = vRest.substr(cSeparator + 3);
  }
  vRest = vRest.substr(0, vRest.find_first_of("/?#,"));
  while (!vRest.empty() && std::isspace(static_cast<unsigned char>(vRest.back()))) {
    vRest.pop_back();
  }
  while (!vRest.empty() && std::isspace(static_cast<unsigned char>(vRest.front()))) {
    vRest.erase(0, 1);
  }
  std::transform(vRest.begin(), vRest.end(), vRest.begin(),
                 [](unsigned char pChar) { return static_cast<char>(std::tolower(pChar)); });
  std::transform(pScheme.begin(), pScheme.end(), pScheme.begin(),
                 [](unsigned char pChar) { return static_cast<char>(std::tolower(pChar)); });
  if ((pScheme == "https" && vRest.size() > 4 && vRest.compare(vRest.size() - 4, 4, ":443") == 0) ||
      (pScheme == "http" && vRest.size() > 3 && vRest.compare(vRest.size() - 3, 3, ":80") == 0)) {
    vRest = vRest.substr(0, vRest.rfind(':'));
  }
  return vRest;
}

// returns true if the Origin header names this same server, as the pages of the Web UI do; X-Forwarded-Host covers
// a reverse proxy, and a cross-site page cannot set it without a preflight that CORS refuses
static bool fOriginIsSameHost(const httplib::Request &pReq, const std::string &pOrigin) {
  const size_t cSeparator = pOrigin.find("://");
  if (cSeparator == std::string::npos) {
    return false;
  }
  const std::string cScheme = pOrigin.substr(0, cSeparator);
  const std::string cOriginAuthority = fNormalizeAuthority(pOrigin, cScheme);
  if (cOriginAuthority.empty()) {
    return false;
  }
  for (const char *vName : {"X-Forwarded-Host", "Host"}) {
    const std::string cHost = pReq.get_header_value(vName);
    if (!cHost.empty() && fNormalizeAuthority(cHost, cScheme) == cOriginAuthority) {
      return true;
    }
  }
  return false;
}

// For Google Cloud Platform deployment compatibility
struct gcp_params {
  bool vEnabled;
  std::string vPathHealth;
  std::string vPathPredict;
  int vPort;

  // Ref: https://docs.cloud.google.com/vertex-ai/docs/predictions/custom-container-requirements#aip-variables
  gcp_params() {
    vEnabled = mGetenv("AIP_MODE", "") == "PREDICTION";
    vPathHealth = mGetenv("AIP_HEALTH_ROUTE", "", true); // default: using the route defined in server.cpp
    vPathPredict = mGetenv("AIP_PREDICT_ROUTE", "/predict", true);
    vPort = std::stoi(mGetenv("AIP_HTTP_PORT", "8080"));
  }

  static std::string mGetenv(const char *pName, const std::string &pDefaultValue, bool pEnsureLeadingSlash = false) {
    const auto *vValue = std::getenv(pName);
    if (vValue == nullptr || vValue[0] == '\0') {
      return pDefaultValue;
    }
    std::string vVal = vValue;
    if (pEnsureLeadingSlash && !vVal.empty() && vVal[0] != '/') {
      vVal.insert(vVal.begin(), '/');
    }
    return vVal;
  }
};

bool server_http_context::mInit(const common_params &pParams) {
  const gcp_params cGcp;

  vPathPrefix = pParams.api_prefix;
  if (vPathPrefix == "/api/doc" || vPathPrefix.rfind("/api/doc/", 0) == 0) {
    SRV_ERR("%s", "/api/doc is reserved for API documentation\n");
    return false;
  }
  if ((vPathPrefix != "/api" && vPathPrefix.rfind("/api/", 0) != 0) || vPathPrefix.back() == '/' ||
      vPathPrefix.find("..") != std::string::npos ||
      vPathPrefix.find_first_not_of("/abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_") !=
          std::string::npos) {
    SRV_ERR("%s", "API prefix must be /api or a path below /api, without a trailing slash\n");
    return false;
  }
  vPort = pParams.vPort;
  vHostname = pParams.vHostname;
  const bool vUnixSocket = fStringEndsWith(vHostname, ".sock");
  vIsSsl = pParams.vUseHttps && !cGcp.vEnabled && !vUnixSocket;
  vPimpl->vAutoPort = !pParams.vPortExplicit && pParams.vPort == 11443 && !cGcp.vEnabled;
  vPimpl->vHttpPort = pParams.vHttpPort;
  if (vPimpl->vAutoPort && !vIsSsl) {
    vPort = 11080;
  }
  if (pParams.ssl_file_key.empty() != pParams.ssl_file_cert.empty()) {
    SRV_ERR("%s", "--ssl-key-file and --ssl-cert-file must be provided together\n");
    return false;
  }
  if (!vIsSsl && !pParams.ssl_file_cert.empty()) {
    SRV_ERR("%s", "SSL certificate files require HTTPS over TCP\n");
    return false;
  }
  if (vPimpl->vHttpPort < 1 || vPimpl->vHttpPort > 65535) {
    SRV_ERR("%s", "HTTP redirect port must be between 1 and 65535\n");
    return false;
  }

  if (cGcp.vEnabled) {
    SRV_TRC("Google Cloud Platform compat: health route = %s, predict route = %s, port = %d\n",
            cGcp.vPathHealth.c_str(), cGcp.vPathPredict.c_str(), cGcp.vPort);

    if (vPort != cGcp.vPort) {
      SRV_WRN("Google Cloud Platform compat: overriding server port %d with AIP_HTTP_PORT %d\n", vPort, cGcp.vPort);
    }

    vPort = cGcp.vPort;
  }

  auto &vSrv = vPimpl->vSrv;

  if (vIsSsl) {
    vSrv = fCreateHttpsServer(vHostname, pParams.ssl_file_cert, pParams.ssl_file_key);
  } else {
    SRV_TRC("%s", "running without SSL\n");
    vSrv = std::make_unique<httplib::Server>();
  }
  if (!vSrv || !vSrv->is_valid()) {
    SRV_ERR("%s", "cannot initialize the server; check HTTPS support, certificate and private key\n");
    return false;
  }

  if (vIsSsl) {
    auto &vRedirect = vPimpl->vRedirectServer;
    vRedirect = std::make_unique<httplib::Server>();
    vRedirect->set_read_timeout(std::min(pParams.vTimeoutRead, cHttpRequestReadTimeoutSec));
    vRedirect->set_write_timeout(pParams.vTimeoutWrite);
    vRedirect->new_task_queue = [] { return new httplib::ThreadPool(2, 2, cHttpMaxQueuedConnections); };
    vRedirect->set_socket_options(
        [](const socket_t pSocket) { httplib::set_socket_opt(pSocket, SOL_SOCKET, SO_REUSEADDR, 1); });
    vRedirect->set_pre_routing_handler([this](const httplib::Request &pRequest, httplib::Response &pResponse) {
      static const std::regex cHostPattern(R"(^(\[[A-Za-z0-9:.%_-]+\]|[A-Za-z0-9._-]+)(:[0-9]{1,5})?$)");
      const std::string vHost = pRequest.get_header_value("Host");
      std::smatch vMatch;
      if (!std::regex_match(vHost, vMatch, cHostPattern) || pRequest.target.empty() || pRequest.target[0] != '/' ||
          std::any_of(pRequest.target.begin(), pRequest.target.end(),
                      [](unsigned char pChar) { return pChar <= 32 || pChar == 127; })) {
        pResponse.status = 400;
      } else {
        pResponse.set_redirect("https://" + vMatch[1].str() + ":" + std::to_string(vPort) + pRequest.target, 308);
        pResponse.set_header("Cache-Control", "no-store");
      }
      return httplib::Server::HandlerResponse::Handled;
    });
  }

  vSrv->set_default_headers({{"Server", "mgs"}});
  // srv->set_logger(log_server_request); // TODO @ngxson : this is too spamy, no very useful; improve it in the future
  vSrv->set_exception_handler([](const httplib::Request &, httplib::Response &pRes, const std::exception_ptr &pEp) {
    // this is fail-safe; exceptions should already handled by `ex_wrapper`

    std::string vMessage;
    try {
      std::rethrow_exception(pEp);
    } catch (const std::exception &vE) {
      vMessage = vE.what();
    } catch (...) {
      vMessage = "Unknown Exception";
    }

    pRes.status = 500;
    pRes.set_content(vMessage, "text/plain");
    SRV_ERR("got exception: %s\n", vMessage.c_str());
  });

  vSrv->set_error_handler([](const httplib::Request &, httplib::Response &pRes) {
    if (pRes.status == 404) {
      pRes.set_content(
          fSafeJsonToStr(json{{"error", {{"message", "File Not Found"}, {"type", "not_found_error"}, {"code", 404}}}}),
          "application/json; charset=utf-8");
    }
    // for other error codes, we skip processing here because it's already done by res->error()
  });

  // set timeouts and change hostname and port
  vSrv->set_read_timeout(std::min(pParams.vTimeoutRead, cHttpRequestReadTimeoutSec));
  vSrv->set_write_timeout(pParams.vTimeoutWrite);
  vSrv->set_socket_options([vReusePort = pParams.vReusePort](const socket_t pSock) {
    httplib::set_socket_opt(pSock, SOL_SOCKET, SO_REUSEADDR, 1);
    if (vReusePort) {
#ifdef SO_REUSEPORT
      httplib::set_socket_opt(pSock, SOL_SOCKET, SO_REUSEPORT, 1);
#else
      SRV_WRN("%s", "SO_REUSEPORT is not supported\n");
#endif
    }
  });

  if (pParams.lApiKeys.size() == 1) {
    const auto cKey = pParams.lApiKeys[0];
    const std::string cSubstr = cKey.substr(std::max(static_cast<int>(cKey.length() - 4), 0));
    SRV_TRC("api_keys: ****%s\n", cSubstr.c_str());
  } else if (pParams.lApiKeys.size() > 1) {
    SRV_TRC("api_keys: %zu keys loaded\n", pParams.lApiKeys.size());
  }

  //
  // Middlewares
  //

  // Frontend paths - all embedded UI assets
  static const std::unordered_set<std::string> sFrontendPaths = []() {
    std::unordered_set<std::string> sPaths{"/"};
    for (const mgs_ui_asset &vA : mgs_ui_get_assets()) {
      sPaths.insert("/" + vA.name);
    }
    return sPaths;
  }();

  // Public endpoints include the configured API prefix and root UI assets.
  const std::unordered_set<std::string> sGetPublicEndpoints = [&pParams]() {
    std::unordered_set<std::string> sEndpoints{
        pParams.api_prefix + "/health",
        pParams.api_prefix + "/v1/health",
        pParams.api_prefix + "/models",
        pParams.api_prefix + "/v1/models",
        "/api/config.js",
        "/api/doc",
        "/api/doc/",
        "/api/doc/openapi.json",
        "/api/doc/swagger-ui.css",
        "/api/doc/swagger-ui-bundle.js",
    };
    sEndpoints.insert(sFrontendPaths.begin(), sFrontendPaths.end());
    return sEndpoints;
  }();

  const std::string vModelsPath = pParams.api_prefix + "/models";
  const std::string vModelsV1Path = pParams.api_prefix + "/v1/models";
  auto vMiddlewareValidateApiKey = [vApiKeys = pParams.lApiKeys, sGetPublicEndpoints, vModelsPath,
                                    vModelsV1Path](const httplib::Request &pReq, httplib::Response &pRes) {
    // If API key is not set, skip validation
    if (vApiKeys.empty()) {
      return true;
    }

    // If path is public or a UI asset, skip validation; only reads are public: the router also registers POST and
    // DELETE on /models, and GET /models?reload=... rescans the model sources
    const bool cIsRead = pReq.method == "GET" || pReq.method == "HEAD";
    const bool cIsModelsReload =
        (pReq.path == vModelsPath || pReq.path == vModelsV1Path) && !pReq.get_param_value("reload").empty();
    if (cIsRead && !cIsModelsReload && sGetPublicEndpoints.count(pReq.path)) {
      return true;
    }

    // Check for API key in the Authorization header
    std::string vReqApiKey = pReq.get_header_value("Authorization");
    if (vReqApiKey.empty()) {
      // retry with anthropic header
      vReqApiKey = pReq.get_header_value("X-Api-Key");
    }

    // remove the "Bearer " prefix if needed
    static std::string vPrefix = "Bearer ";
    if (vReqApiKey.substr(0, vPrefix.size()) == vPrefix) {
      vReqApiKey = vReqApiKey.substr(vPrefix.size());
    }

    // validate the API key
    if (std::find(vApiKeys.begin(), vApiKeys.end(), vReqApiKey) != vApiKeys.end()) {
      return true; // API key is valid
    }

    // API key is invalid or not provided
    pRes.status = 401;
    pRes.set_content(fSafeJsonToStr(json{
                         {"error", {{"message", "Invalid API Key"}, {"type", "authentication_error"}, {"code", 401}}}}),
                     "application/json; charset=utf-8");

    SRV_WRN("%s", "unauthorized: Invalid API Key\n");

    return false;
  };

  // CPU and GPU usage never touch the model, so the status bar keeps working while it loads;
  // neither does the shell MCP server, so the UI can connect to it before the model is ready,
  // nor the Web UI settings, which the interface reads as soon as it opens
  const std::string vMonitorUsagePath = pParams.api_prefix + "/monitor/usage";
  const std::string vShellMcpPath = pParams.api_prefix + "/mcp/shell";
  const std::string vShellAskpassPath = pParams.api_prefix + "/mcp/shell/askpass";
  const std::string vUiSettingsPath = pParams.api_prefix + "/ui/settings";
  auto vMiddlewareServerState = [this, vMonitorUsagePath, vShellMcpPath, vShellAskpassPath,
                                 vUiSettingsPath](const httplib::Request &pReq, httplib::Response &pRes) {
    if (!vIsReady.load()) {
      if (sFrontendPaths.count(pReq.path) || pReq.path == "/api/config.js" || pReq.path == "/api/doc" ||
          pReq.path.rfind("/api/doc/", 0) == 0) {
        return true; // frontend asset, allow it to load and show "loading"
      }
      if (pReq.path == vMonitorUsagePath || pReq.path == vShellMcpPath || pReq.path == vShellAskpassPath ||
          pReq.path == vUiSettingsPath) {
        return true;
      }
      // no endpoints are allowed to be accessed when the server is not ready
      // this is to prevent any data races or inconsistent states
      pRes.status = 503;
      pRes.set_content(
          fSafeJsonToStr(json{{"error", {{"message", "Loading model"}, {"type", "unavailable_error"}, {"code", 503}}}}),
          "application/json; charset=utf-8");
      return false;
    }
    return true;
  };

  // register server middlewares
  vSrv->set_pre_routing_handler([&pParams, vMiddlewareValidateApiKey,
                                 vMiddlewareServerState](const httplib::Request &pReq, httplib::Response &pRes) {
    if (pParams.vCorsCredentials && pParams.vCorsOrigins == "*") {
      // special case: echo back the Origin header to allow any origin to access the server with credentials
      pRes.set_header("Access-Control-Allow-Origin", pReq.get_header_value("Origin"));
    } else if (pParams.vCorsOrigins == "localhost") {
      // special case: only reflect the Origin header if it is a localhost origin
      std::string vOrigin = pReq.get_header_value("Origin");
      if (!vOrigin.empty() && fOriginIsLocalhost(vOrigin)) {
        pRes.set_header("Access-Control-Allow-Origin", vOrigin);
      } else if (!vOrigin.empty()) {
        SRV_WRN("(CORS) skip non-localhost origin: %s\n", vOrigin.c_str());
      }
    } else {
      pRes.set_header("Access-Control-Allow-Origin", pParams.vCorsOrigins);
    }
    // If this is OPTIONS request, skip validation because browsers don't include Authorization header
    if (pReq.method == "OPTIONS") {
      pRes.set_header("Access-Control-Allow-Credentials", pParams.vCorsCredentials ? "true" : "false");
      pRes.set_header("Access-Control-Allow-Methods", pParams.vCorsMethods);
      pRes.set_header("Access-Control-Allow-Headers", pParams.vCorsHeaders);
      pRes.set_content("", "text/html");                // blank response, no data
      return httplib::Server::HandlerResponse::Handled; // skip further processing
    }
    // CORS only decides whether a page may read the response, not whether the request runs: a cross-site page can
    // still send a simple POST. With tools or MCP servers enabled (CORS "localhost"), browsers that name a foreign
    // Origin are refused before any handler runs; clients without Origin, such as curl, are left to the API key
    if (pParams.vCorsOrigins == "localhost" && pReq.method != "GET" && pReq.method != "HEAD") {
      const std::string cOrigin = pReq.get_header_value("Origin");
      if (!cOrigin.empty() && !fOriginIsLocalhost(cOrigin) && !fOriginIsSameHost(pReq, cOrigin)) {
        SRV_WRN("(CORS) refused %s %s from foreign origin: %s\n", pReq.method.c_str(), pReq.path.c_str(),
                cOrigin.c_str());
        pRes.status = 403;
        pRes.set_content(fSafeJsonToStr(json{{"error",
                                              {{"message", "Cross-origin request refused"},
                                               {"type", "permission_error"},
                                               {"code", 403}}}}),
                         "application/json; charset=utf-8");
        return httplib::Server::HandlerResponse::Handled;
      }
    }
    if (!vMiddlewareServerState(pReq, pRes)) {
      return httplib::Server::HandlerResponse::Handled;
    }
    if (!vMiddlewareValidateApiKey(pReq, pRes)) {
      return httplib::Server::HandlerResponse::Handled;
    }
    return httplib::Server::HandlerResponse::Unhandled;
  });

  auto vNThreadsHttp = pParams.vNThreadsHttp;
  if (vNThreadsHttp < 1) {
    // +4 threads for monitoring, health and some threads reserved for MCP and other tasks in the future
    vNThreadsHttp = std::max(pParams.vNParallel + 4, static_cast<int32_t>(std::thread::hardware_concurrency() - 1));
  }
  SRV_TRC("using %d threads for HTTP server\n", vNThreadsHttp);
  vSrv->new_task_queue = [vNThreadsHttp] {
    // spawn n_threads_http fixed thread (always alive), while allow up to 1024 max possible additional threads
    // when n_threads_http is used, server will create new "dynamic" threads that will be destroyed after processing
    // each request ref: https://github.com/yhirose/cpp-httplib/pull/2368
    const auto cMaxThreads = static_cast<size_t>(vNThreadsHttp + 1024);
    // accepted sockets wait here only once every thread is busy; past the bound they are closed at once instead of
    // piling up open descriptors
    return new httplib::ThreadPool(vNThreadsHttp, cMaxThreads, cHttpMaxQueuedConnections);
  };

  //
  // Web UI setup
  //

  fRegisterApiDocumentation(*vSrv, *this);
  vSrv->Get("/api/config.js", [vPrefix = vPathPrefix](const httplib::Request &, httplib::Response &pResponse) {
    // the saved interface language is published here so the interface starts in it before its modules load
    const std::string cLanguage = fUiSettingsLanguage();
    pResponse.set_header("Cache-Control", "no-store");
    pResponse.set_content("globalThis.cApiPrefix = " + json(vPrefix).dump() + ";\nglobalThis.cUiLanguage = " +
                              (cLanguage.empty() ? std::string("null") : json(cLanguage).dump()) + ";\n",
                          "application/javascript; charset=utf-8");
  });

  if (!pParams.vUi) {
    SRV_INF("%s", "The UI is disabled\n");
    SRV_INF("%s", "Use --ui/--no-ui to enable/disable\n");
  } else {
    // register static assets routes
    if (!pParams.public_path.empty()) {
      // Set the base directory for serving static files
      if (const auto cIsFound = vSrv->set_mount_point("/", pParams.public_path); !cIsFound) {
        SRV_ERR("static assets path not found: %s\n", pParams.public_path.c_str());
        return false;
      }
    } else {
#if defined(MGS_UI_HAS_ASSETS)
      static auto vHandleGzipHeader = [](const httplib::Request &pReq, httplib::Response &pRes) {
        if (!mgs_ui_use_gzip()) {
          // no gzip build, skip
          return true;
        }
        if (pReq.get_header_value("Accept-Encoding").find("gzip") == std::string::npos) {
          pRes.status = 415; // unsupported media type
          pRes.set_content("Error: gzip is not supported by this browser", "text/plain");
          return false;
        } else {
          pRes.set_header("Content-Encoding", "gzip");
        }
        return true;
      };

      auto vServeAssetCached = [](const std::string &pName, bool pIsolation) {
        return [pName, pIsolation](const httplib::Request &pReq, httplib::Response &pRes) {
          if (!vHandleGzipHeader(pReq, pRes)) {
            return true; // returns error message
          }
          const mgs_ui_asset *vA = mgs_ui_find_asset(pName);
          if (!vA) {
            pRes.status = 404;
            return false;
          }
          pRes.set_header("ETag", vA->etag);
          if (const std::string &vInm = pReq.get_header_value("If-None-Match");
              !vInm.empty() && (vInm == vA->etag || vInm == std::string("W/") + vA->etag)) {
            pRes.status = 304;
            return false;
          }
          if (pIsolation) {
            pRes.set_header("Cross-Origin-Embedder-Policy", "require-corp");
            pRes.set_header("Cross-Origin-Opener-Policy", "same-origin");
          }
          pRes.set_header("Cache-Control", pIsolation ? "no-cache" : "public, max-age=31536000, immutable");
          pRes.set_content(reinterpret_cast<const char *>(vA->data), vA->size, vA->type.c_str());
          return false;
        };
      };

      auto vServeAssetNocache = [](const std::string &pName) {
        return [pName](const httplib::Request &pReq, httplib::Response &pRes) {
          if (!vHandleGzipHeader(pReq, pRes)) {
            return true; // returns error message
          }
          const mgs_ui_asset *vA = mgs_ui_find_asset(pName);
          if (!vA) {
            pRes.status = 404;
            return false;
          }
          pRes.set_header("Cache-Control", "no-cache");
          pRes.set_content(reinterpret_cast<const char *>(vA->data), vA->size, vA->type.c_str());
          return false;
        };
      };

      // main index file
      vSrv->Get("/", vServeAssetCached("index.html", true));
      vSrv->Get("/index.html", vServeAssetCached("index.html", true));

      // All remaining assets registered directly from the embedded asset table.
      // PWA revalidation files (sw.js, manifest, version.json) use no-cache;
      // everything else is immutable.
      static const std::unordered_set<std::string> cNoCacheNames = {"sw.js", "manifest.webmanifest",
                                                                    "_app/version.json", "build.json"};

      for (const auto &vA : mgs_ui_get_assets()) {
        if (vA.name == "index.html")
          continue; // served at "/" and "/index.html" above
        if (cNoCacheNames.count(vA.name)) {
          SRV_DBG("serve nocache for %s\n", vA.name.c_str());
          vSrv->Get("/" + vA.name, vServeAssetNocache(vA.name));
        } else {
          vSrv->Get("/" + vA.name, vServeAssetCached(vA.name, false));
        }
      }

#endif
    }
  }
  return true;
}

bool server_http_context::mStart() {
  // Bind and listen

  const auto &vSrv = vPimpl->vSrv;
  auto vWasBound = false;
  auto vIsSock = false;
  if (fStringEndsWith(std::string(vHostname), ".sock")) {
    vIsSock = true;
    SRV_TRC("%s", "setting address family to AF_UNIX\n");
    vSrv->set_address_family(AF_UNIX);
    // bind_to_port requires a second arg, any value other than 0 should
    // simply get ignored
    vWasBound = vSrv->bind_to_port(vHostname, 1);
  } else {
    SRV_TRC("%s", "binding port with default address family\n");
    vWasBound = fBindServerPort(*vSrv, vHostname, vPort, vPimpl->vAutoPort);
  }

  if (!vWasBound) {
    SRV_ERR("couldn't bind HTTP server socket, hostname: %s, port: %d\n", vHostname.c_str(), vPort);
    return false;
  }

  // Reserve HTTPS first, then select a different HTTP redirect port.
  if (vPimpl->vRedirectServer &&
      !fBindServerPort(*vPimpl->vRedirectServer, vHostname, vPimpl->vHttpPort, true, vPort)) {
    SRV_ERR("couldn't bind HTTP redirect socket, hostname: %s, port: %d\n", vHostname.c_str(), vPimpl->vHttpPort);
    mStop();
    return false;
  }
  try {
    vThread = std::thread([this] {
      vPimpl->vSrv->listen_after_bind();
      mStop();
    });
    vSrv->wait_until_ready();
    if (!vSrv->is_running()) {
      mStop();
      return false;
    }
    if (vPimpl->vRedirectServer) {
      vPimpl->vRedirectThread = std::thread([this] {
        vPimpl->vRedirectServer->listen_after_bind();
        mStop();
      });
      vPimpl->vRedirectServer->wait_until_ready();
      if (!vPimpl->vRedirectServer->is_running()) {
        mStop();
        return false;
      }
      SRV_INF("HTTP redirect listening on http://%s:%d -> https://%s:%d\n", fCommonHttpFormatHost(vHostname).c_str(),
              vPimpl->vHttpPort, fCommonHttpFormatHost(vHostname).c_str(), vPort);
    }
  } catch (const std::exception &vException) {
    SRV_ERR("cannot start server listeners: %s\n", vException.what());
    mStop();
    return false;
  }

  vListeningAddress =
      vIsSock ? fStringFormat("unix://%s", vHostname.c_str())
              : fStringFormat("%s://%s:%d", vIsSsl ? "https" : "http", fCommonHttpFormatHost(vHostname).c_str(), vPort);
  return true;
}

void server_http_context::mStop() const {
  if (vPimpl->vRedirectServer) {
    vPimpl->vRedirectServer->stop();
    vPimpl->vRedirectServer->decommission();
  }
  if (vPimpl->vSrv) {
    vPimpl->vSrv->stop();
    vPimpl->vSrv->decommission();
  }
}

static void fSetHeaders(httplib::Response &pRes, const std::map<std::string, std::string> &pHeaders) {
  for (const auto &[vKey, vValue] : pHeaders) {
    pRes.set_header(vKey, vValue);
  }
}

// percent-decode a path component (%XX). path params arrive raw from httplib, unlike query
// params, so a conv id like "conv::model" sent as "conv%3A%3Amodel" must be decoded here to
// match the value the client put in the X-Conversation-Id header
static std::string fDecodePathComponent(const std::string &pIn) {
  std::string vOut;
  vOut.reserve(pIn.size());
  for (size_t vI = 0; vI < pIn.size(); vI++) {
    if (pIn[vI] == '%' && vI + 2 < pIn.size()) {
      auto vHex = [](char pC) -> int {
        if (pC >= '0' && pC <= '9')
          return pC - '0';
        if (pC >= 'a' && pC <= 'f')
          return pC - 'a' + 10;
        if (pC >= 'A' && pC <= 'F')
          return pC - 'A' + 10;
        return -1;
      };
      int vHi = vHex(pIn[vI + 1]);
      int vLo = vHex(pIn[vI + 2]);
      if (vHi >= 0 && vLo >= 0) {
        vOut.push_back(char((vHi << 4) | vLo));
        vI += 2;
        continue;
      }
    }
    vOut.push_back(pIn[vI]);
  }
  return vOut;
}

static std::map<std::string, std::string> fGetParams(const httplib::Request &pReq) {
  std::map<std::string, std::string> dParams;
  for (const auto &[vKey, vValue] : pReq.params) {
    dParams[vKey] = vValue;
  }
  for (const auto &[vKey, vValue] : pReq.path_params) {
    dParams[vKey] = fDecodePathComponent(vValue);
  }
  return dParams;
}

static std::map<std::string, std::string> fGetHeaders(const httplib::Request &pReq) {
  std::map<std::string, std::string> dHeaders;
  for (const auto &[vKey, vValue] : pReq.headers) {
    dHeaders[vKey] = vValue;
  }
  return dHeaders;
}

static std::string fBuildQueryString(const httplib::Request &pReq) {
  std::string vQs;
  for (const auto &[vKey, vValue] : pReq.params) {
    if (!vQs.empty()) {
      vQs += '&';
    }
    vQs += httplib::encode_query_component(vKey) + "=" + httplib::encode_query_component(vValue);
  }
  return vQs;
}

// using unique_ptr for request to allow safe capturing in lambdas
using server_http_req_ptr = std::unique_ptr<server_http_req>;

// refuses a request body nested deeper than cHttpMaxRequestJsonDepth before any handler parses it
static bool fRejectDeepRequestBody(const std::string &pBody, httplib::Response &pRes) {
  if (!fJsonNestingExceeds(pBody, cHttpMaxRequestJsonDepth)) {
    return false;
  }
  pRes.status = 400;
  pRes.set_content(
      fSafeJsonToStr(json{{"error",
                           {{"message", "Request body is nested deeper than " +
                                            std::to_string(cHttpMaxRequestJsonDepth) + " levels"},
                            {"type", "invalid_request_error"},
                            {"code", 400}}}}),
      "application/json; charset=utf-8");
  return true;
}

static void fProcessHandlerResponse(server_http_req_ptr &&pRequest, server_http_res_ptr &pResponse,
                                    httplib::Response &pRes) {
  if (pResponse->mIsStream()) {
    pRes.status = pResponse->vStatus;
    // Tell Nginx to not buffer any streamed response
    pResponse->dHeaders["X-Accel-Buffering"] = "no";
    fSetHeaders(pRes, pResponse->dHeaders);
    const std::string cContentType = pResponse->vContentType;
    // convert to shared_ptr as both cChunkedContentProvider() and cOnComplete() need to use it
    std::shared_ptr<server_http_req> vQPtr = std::move(pRequest);
    std::shared_ptr<server_http_res> vRPtr = std::move(pResponse);

    const auto cChunkedContentProvider = [vResponse = vRPtr](size_t, httplib::DataSink &pSink) -> bool {
      std::string vChunk;
      const bool cHasNext = vResponse->vNext(vChunk);
      if (!vChunk.empty()) {
        if (!pSink.write(vChunk.data(), vChunk.size())) {
          return false;
        }
        SRV_DBG("http: streamed chunk: %s\n", vChunk.c_str());
      }
      if (!cHasNext) {
        pSink.done();
        SRV_DBG("%s", "http: stream ended\n");
      }
      return cHasNext;
    };
    const auto cOnComplete = [vRequest = vQPtr, vResponse = vRPtr](bool) mutable {
      vResponse->mOnComplete();
      vResponse.reset();
      vRequest.reset();
    };
    pRes.set_chunked_content_provider(cContentType, cChunkedContentProvider, cOnComplete);
  } else {
    pRes.status = pResponse->vStatus;
    fSetHeaders(pRes, pResponse->dHeaders);
    pRes.set_content(pResponse->vData, pResponse->vContentType);
    pResponse->mOnComplete();
  }
}

void server_http_context::mGet(const std::string &pPath, const server_http_context::handler_t &pHandler) const {
  dHandlers.emplace(pPath, pHandler);
  dApiMethods[pPath].insert("get");
  vPimpl->vSrv->Get(vPathPrefix + pPath, [pHandler, vPrefix = vPathPrefix](const httplib::Request &pReq,
                                                                           httplib::Response &pRes) {
    server_http_req_ptr vRequest = std::make_unique<server_http_req>(server_http_req{fGetParams(pReq),
                                                                                     fGetHeaders(pReq),
                                                                                     pReq.path.substr(vPrefix.size()),
                                                                                     fBuildQueryString(pReq),
                                                                                     pReq.body,
                                                                                     {},
                                                                                     pReq.is_connection_closed});
    server_http_res_ptr vResponse = pHandler(*vRequest);
    fProcessHandlerResponse(std::move(vRequest), vResponse, pRes);
  });
}

void server_http_context::mPost(const std::string &pPath, const server_http_context::handler_t &pHandler) const {
  dHandlers.emplace(pPath, pHandler);
  dApiMethods[pPath].insert("post");
  vPimpl->vSrv->Post(
      vPathPrefix + pPath, [pHandler, vPrefix = vPathPrefix](const httplib::Request &pReq, httplib::Response &pRes) {
        std::string vBody = pReq.body;
        std::map<std::string, uploaded_file> dFiles;

        if (pReq.is_multipart_form_data()) {
          // translate text fields to a JSON object and use it as the body
          json vFormJson = json::object();
          for (const auto &[vKey, vField] : pReq.form.fields) {
            if (vFormJson.contains(vKey)) {
              // if the key already exists, convert it to an array
              if (!vFormJson[vKey].is_array()) {
                json vExistingValue = vFormJson[vKey];
                vFormJson[vKey] = json::array({vExistingValue});
              }
              vFormJson[vKey].push_back(vField.content);
            } else {
              vFormJson[vKey] = vField.content;
            }
          }
          vBody = vFormJson.dump();

          // populate files from multipart form
          for (const auto &[vKey, vFile] : pReq.form.files) {
            dFiles[vKey] = uploaded_file{
                vRawBuffer(vFile.content.begin(), vFile.content.end()),
                vFile.filename,
                vFile.content_type,
            };
          }
        } else if (fRejectDeepRequestBody(vBody, pRes)) {
          return;
        }

        server_http_req_ptr vRequest = std::make_unique<server_http_req>(
            server_http_req{fGetParams(pReq), fGetHeaders(pReq), pReq.path.substr(vPrefix.size()),
                            fBuildQueryString(pReq), vBody, std::move(dFiles), pReq.is_connection_closed});
        server_http_res_ptr vResponse = pHandler(*vRequest);
        fProcessHandlerResponse(std::move(vRequest), vResponse, pRes);
      });
}

void server_http_context::mDel(const std::string &pPath, const server_http_context::handler_t &pHandler) const {
  dHandlers.emplace(pPath, pHandler);
  dApiMethods[pPath].insert("delete");
  vPimpl->vSrv->Delete(vPathPrefix + pPath, [pHandler, vPrefix = vPathPrefix](const httplib::Request &pReq,
                                                                              httplib::Response &pRes) {
    if (fRejectDeepRequestBody(pReq.body, pRes)) {
      return;
    }
    server_http_req_ptr vRequest = std::make_unique<server_http_req>(server_http_req{fGetParams(pReq),
                                                                                     fGetHeaders(pReq),
                                                                                     pReq.path.substr(vPrefix.size()),
                                                                                     fBuildQueryString(pReq),
                                                                                     pReq.body,
                                                                                     {},
                                                                                     pReq.is_connection_closed});
    server_http_res_ptr vResponse = pHandler(*vRequest);
    fProcessHandlerResponse(std::move(vRequest), vResponse, pRes);
  });
}

//
// Vertex AI Prediction protocol (AIP_PREDICT_ROUTE)
// https://cloud.google.com/vertex-ai/docs/predictions/custom-container-requirements
//

// Derives the camelCase @requestFormat alias for a registered path.
// e.g. "/v1/chat/completions" -> "chatCompletions", "/apply-template" -> "applyTemplate"
static std::string fPathToGcpFormat(const std::string &pPath) {
  std::string vS = pPath;
  if (vS.size() > 3 && vS[0] == '/' && vS[1] == 'v' && vS[2] == '1') {
    vS = vS.substr(3);
  }
  if (!vS.empty() && vS[0] == '/') {
    vS = vS.substr(1);
  }
  std::string vResult;
  bool vCap = false;
  for (unsigned char vC : vS) {
    if (vC == ':')
      break; // stop before path parameters
    if (vC == '/' || vC == '-' || vC == '_') {
      vCap = true;
    } else {
      vResult += static_cast<char>(vCap ? std::toupper(vC) : vC);
      vCap = false;
    }
  }
  return vResult;
}

static json fParseGcpPredictResponse(const server_http_res_ptr &pRes) {
  if (pRes == nullptr) {
    throw std::runtime_error("empty response from internal handler");
  }
  if (pRes->mIsStream()) {
    throw std::invalid_argument("predict route does not support streaming responses");
  }
  if (pRes->vData.empty()) {
    return nullptr;
  }
  try {
    return json::parse(pRes->vData);
  } catch (...) {
    return pRes->vData;
  }
}

void server_http_context::mRegisterGcpCompat() const {
  const gcp_params cGcp;

  if (!cGcp.vEnabled) {
    // do nothing
    return;
  }

  if (dHandlers.count(cGcp.vPathPredict)) {
    SRV_ERR("AIP_PREDICT_ROUTE=%s conflicts with an existing mgs route\n", cGcp.vPathPredict.c_str());
    exit(1);
  }

  // camelCase alias -> canonical path (first registration wins on collision)
  // e.g. "chatCompletions" -> "/v1/chat/completions"
  std::unordered_map<std::string, std::string> dAliasToPath;
  for (const auto &[vPath, _] : dHandlers) {
    dAliasToPath.emplace(fPathToGcpFormat(vPath), vPath);
  }

  if (!cGcp.vPathHealth.empty()) {
    const auto cHealthHandler = dHandlers.find("/health");
    GGML_ASSERT(cHealthHandler != dHandlers.end());
    mGet(cGcp.vPathHealth, cHealthHandler->second);
  }

  mPost(
      cGcp.vPathPredict,
      [this, vAliasToPath = std::move(dAliasToPath)](const server_http_req &pReq) -> server_http_res_ptr {
        static const auto cBuildError = [](const std::string &pMessage, error_type pType) -> json {
          return json{{"error", fFormatErrorResponse(pMessage, pType)}};
        };

        json vData;
        try {
          vData = json::parse(pReq.vBody);
        } catch (const std::exception &vE) {
          auto vRes = std::make_unique<server_http_res>();
          vRes->vStatus = 400;
          vRes->vData = fSafeJsonToStr({{"error", fFormatErrorResponse(vE.what(), cErrorTypeInvalidRequest)}});
          return vRes;
        }
        if (!vData.is_object()) {
          auto vRes = std::make_unique<server_http_res>();
          vRes->vStatus = 400;
          vRes->vData = fSafeJsonToStr(
              {{"error", fFormatErrorResponse("request body must be a JSON object", cErrorTypeInvalidRequest)}});
          return vRes;
        }
        if (!vData.contains("instances") || !vData.at("instances").is_array()) {
          auto vRes = std::make_unique<server_http_res>();
          vRes->vStatus = 400;
          vRes->vData =
              fSafeJsonToStr({{"error", fFormatErrorResponse("request body must include an array field named instances",
                                                             cErrorTypeInvalidRequest)}});
          return vRes;
        }

        const json &vInstances = vData.at("instances");
        static const size_t cMaxInstances = 128;
        if (vInstances.size() > cMaxInstances) {
          auto vRes = std::make_unique<server_http_res>();
          vRes->vStatus = 400;
          vRes->vData = fSafeJsonToStr({{"error", fFormatErrorResponse("instances array exceeds maximum size of " +
                                                                           std::to_string(cMaxInstances),
                                                                       cErrorTypeInvalidRequest)}});
          return vRes;
        }

        std::vector<std::future<json>> vLFutures;
        vLFutures.reserve(vInstances.size());

        for (const auto &vInstance : vInstances) {
          vLFutures.push_back(std::async(std::launch::async, [this, &pReq, &vAliasToPath, vInstance]() -> json {
            if (!vInstance.is_object()) {
              return cBuildError("each instance must be a JSON object", cErrorTypeInvalidRequest);
            }
            if (!vInstance.contains("@requestFormat") || !vInstance.at("@requestFormat").is_string()) {
              return cBuildError("each instance must include a string @requestFormat", cErrorTypeInvalidRequest);
            }

            try {
              json vPayload = vInstance;
              const std::string cFormat = vPayload.at("@requestFormat").get<std::string>();
              vPayload.erase("@requestFormat");

              if (vPayload.contains("stream")) {
                SRV_WRN(
                    "%s",
                    "ignoring client-provided stream field in instance, streaming is not supported in predict route\n");
                vPayload["stream"] = false;
              }

              // accept both camelCase aliases (e.g. "chatCompletions") and direct paths
              std::string vDispatchPath;
              auto vItAlias = vAliasToPath.find(cFormat);
              if (vItAlias != vAliasToPath.end()) {
                vDispatchPath = vItAlias->second;
              } else if (dHandlers.count(cFormat)) {
                vDispatchPath = cFormat;
              } else {
                return cBuildError("no handler registered for @requestFormat: " + cFormat, cErrorTypeInvalidRequest);
              }

              const server_http_req cInternalReq{
                  pReq.dParams, pReq.dHeaders, vDispatchPath, pReq.vQueryString, vPayload.dump(), {}, pReq.vShouldStop,
              };

              server_http_res_ptr vInternalRes = dHandlers.at(vDispatchPath)(cInternalReq);
              return fParseGcpPredictResponse(vInternalRes);
            } catch (const std::invalid_argument &vE) {
              return cBuildError(vE.what(), cErrorTypeInvalidRequest);
            } catch (const std::exception &vE) {
              return cBuildError(vE.what(), cErrorTypeServer);
            } catch (...) {
              return cBuildError("unknown error", cErrorTypeServer);
            }
          }));
        }

        json vPredictions = json::array();
        for (auto &vFuture : vLFutures) {
          vPredictions.push_back(vFuture.get());
        }

        auto vRes = std::make_unique<server_http_res>();
        vRes->vData = fSafeJsonToStr({{"predictions", vPredictions}});
        return vRes;
      });
}
