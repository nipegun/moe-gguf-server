#include "server-context.h"
#include "server-http.h"
#include "server-models.h"
#include "server-cors-proxy.h"
#include "server-shell-mcp.h"
#include "server-stream.h"
#include "server-tools.h"
#include "server-ui-settings.h"
#include "server-chats.h"

#include "arg.h"
#include "build-info.h"
#include "common.h"
#include "fit.h"
#include "mgs.h"
#include "log.h"
#include "speculative.h"

#include <atomic>
#include <cerrno>
#include <clocale>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <signal.h>
#include <thread> // for std::thread::hardware_concurrency and the shutdown thread

#if defined(_WIN32)
#include <windows.h>
#else
#include <fcntl.h>
#include <unistd.h>
#endif

static std::function<void(int)> fShutdownHandler;
static std::atomic_flag vIsTerminating = ATOMIC_FLAG_INIT;

#if defined(__unix__) || (defined(__APPLE__) && defined(__MACH__))
// The shutdown handler takes mutexes (the task queue) that the thread interrupted by the signal may hold, so the
// signal handler only writes the signal number into this pipe and a thread of its own runs the shutdown handler.
static int aShutdownPipe[2] = {-1, -1};
#endif

static inline void fSignalHandler(int pSignal) {
  if (vIsTerminating.test_and_set()) {
    // in case it hangs, we can force terminate the server by hitting Ctrl+C twice
    // this is for better developer experience, we can remove when the server is stable enough
    fprintf(stderr, "Received second interrupt, terminating immediately.\n");
    exit(1);
  }

#if defined(__unix__) || (defined(__APPLE__) && defined(__MACH__))
  if (aShutdownPipe[1] >= 0) {
    const unsigned char cSignal = (unsigned char)pSignal;
    const ssize_t vWritten = write(aShutdownPipe[1], &cSignal, 1);
    if (vWritten == 1) {
      return;
    }
  }
#endif

  fShutdownHandler(pSignal);
}

#if defined(__unix__) || (defined(__APPLE__) && defined(__MACH__))
// Opens the shutdown pipe (closed on exec, so that the commands of the shell MCP server do not inherit it) and starts
// the thread that waits on it; without the pipe the signal handler runs the shutdown handler itself, as before.
static void fStartShutdownThread() {
  if (pipe(aShutdownPipe) != 0) {
    LOG_WRN("%s: could not create the shutdown pipe (%s); the signal handler stops the server itself\n", __func__,
            std::strerror(errno));
    aShutdownPipe[0] = -1;
    aShutdownPipe[1] = -1;
    return;
  }
  fcntl(aShutdownPipe[0], F_SETFD, FD_CLOEXEC);
  fcntl(aShutdownPipe[1], F_SETFD, FD_CLOEXEC);
  std::thread([]() {
    unsigned char vSignal = 0;
    ssize_t vRead;
    do {
      vRead = read(aShutdownPipe[0], &vSignal, 1);
    } while (vRead < 0 && errno == EINTR);
    if (vRead == 1) {
      fShutdownHandler(vSignal);
    }
  }).detach();
}
#endif

// satisfies -Wmissing-declarations (used by mgs command)
int mgs_server(int pArgc, char **pArgv);

// to be used via CLI (argc / argv are used by router mode only)
int mgs_server(common_params &pParams, int pArgc, char **pArgv);
void mgs_server_terminate();
void mgs_server_terminate() {
  if (fShutdownHandler) {
    fShutdownHandler(0);
  }
}

// wrapper function that handles exceptions and logs errors
// this is to make sure handler_t never throws exceptions; instead, it returns an error response
static server_http_context::handler_t fWrapExceptions(server_http_context::handler_t pFunction) {
  return [pFunction = std::move(pFunction)](const server_http_req &pRequest) -> server_http_res_ptr {
    std::string vMessage;
    error_type vError;
    try {
      return pFunction(pRequest);
    } catch (const std::invalid_argument &vException) {
      // treat invalid_argument as invalid request (400)
      vError = cErrorTypeInvalidRequest;
      vMessage = vException.what();
    } catch (const std::exception &vException) {
      // treat other exceptions as server error (500)
      vError = cErrorTypeServer;
      vMessage = vException.what();
    } catch (...) {
      vError = cErrorTypeServer;
      vMessage = "unknown error";
    }

    auto vResponse = std::make_unique<server_http_res>();
    vResponse->vStatus = 500;
    try {
      json dErrorData = fFormatErrorResponse(vMessage, vError);
      vResponse->vStatus = fJsonValue(dErrorData, "code", 500);
      vResponse->vData = fSafeJsonToStr({{"error", dErrorData}});
      SRV_WRN("got exception: %s\n", vResponse->vData.c_str());
    } catch (const std::exception &vException) {
      SRV_ERR("got another exception: %s | while handling exception: %s\n", vException.what(), vMessage.c_str());
      vResponse->vData = "Internal Server Error";
    }
    return vResponse;
  };
}

// On an integrated GPU the CPU and the GPU share the power budget of the package, and the OpenMP threads of the CPU
// backend spin while they wait for the GPU part of every token, taking that power from the GPU: with
// Qwen3.8-Flash-Next on a Radeon 860M the GPU ran at 918 MHz with spinning threads and at 1720 MHz with sleeping
// ones, both at the 14.6 W of the package, and generation went from 5.25 to 5.5 tokens/s. libgomp reads
// OMP_WAIT_POLICY only when it is loaded, so when the only GPU is integrated and the variable is not set, the process
// starts itself again with OMP_WAIT_POLICY=passive; it returns when it does not, or when the restart fails. A
// discrete GPU has its own power budget and keeps the default.
static void fWaitPassivelyOnIntegratedGpu(char **pArgv) {
#if defined(__linux__)
  bool vAnyIntegrated = false;
  bool vAnyDiscrete = false;
  for (size_t vI = 0; vI < ggml_backend_dev_count(); vI++) {
    const enum ggml_backend_dev_type cType = ggml_backend_dev_type(ggml_backend_dev_get(vI));
    vAnyIntegrated = vAnyIntegrated || cType == GGML_BACKEND_DEVICE_TYPE_IGPU;
    vAnyDiscrete = vAnyDiscrete || cType == GGML_BACKEND_DEVICE_TYPE_GPU;
  }
  if (!vAnyIntegrated || vAnyDiscrete) {
    return;
  }
  const char *vPolicy = getenv("OMP_WAIT_POLICY");
  if (vPolicy == nullptr) {
    // In the self-contained bundle the process image is the bundled dynamic loader, which would read the arguments of
    // mgs as its own options, so the restart goes through the launcher, which publishes its path in MGS_EXECUTABLE
    const char *vLauncher = getenv("MGS_EXECUTABLE");
    const char *vExecutable = vLauncher != nullptr && vLauncher[0] != '\0' ? vLauncher : "/proc/self/exe";
    setenv("OMP_WAIT_POLICY", "passive", 1);
    execv(vExecutable, pArgv);
    unsetenv("OMP_WAIT_POLICY");
    LOG_WRN("%s: could not restart with OMP_WAIT_POLICY=passive (%s); the CPU threads keep spinning while the "
            "integrated GPU works\n",
            __func__, std::strerror(errno));
    return;
  }
  if (std::strcmp(vPolicy, "passive") == 0) {
    LOG_INF("%s: integrated GPU: the OpenMP threads of the CPU wait passively (OMP_WAIT_POLICY=passive), leaving the "
            "power of the package to the GPU\n",
            __func__);
  }
#else
  GGML_UNUSED(pArgv);
#endif
}

int mgs_server(int pArgc, char **pArgv) {
  // sudo starts this same executable as the password helper (SUDO_ASKPASS) of the local shell MCP server
  const int vAskpassStatus = fServerShellMcpAskpassMain(pArgc, pArgv);
  if (vAskpassStatus >= 0) {
    return vAskpassStatus;
  }

  std::setlocale(LC_NUMERIC, "C");

#ifndef _WIN32
  // Ignore SIGPIPE so the server does not crash if an MCP child exits while we are writing to its stdin
  signal(SIGPIPE, SIG_IGN);
#endif

  // own arguments required by this example
  common_params vParams;

  fCommonInit();

  // start the stream session manager GC right after common init, before any HTTP route can
  // touch it. lifecycle is symmetric, mStopGc() runs in vCleanUp() before backend free
  fServerStreamSessionManagerStart();

  if (!fCommonParamsParse(pArgc, pArgv, vParams)) {
    return 1;
  }

#if !defined(_WIN32)
  // the inference and web services are meant to run without root; it is allowed, but warned about
  if (geteuid() == 0) {
    LOG_WRN("%s", "mgs is running as root: run it with an unprivileged user instead (it never needs root, and the "
                  "local shell MCP server would give the model a root shell)\n");
  }
#endif

  mgs_backend_init();
  mgs_numa_init(vParams.vNuma);

  // the router server never loads a model and must not touch the GPU
  if (!vParams.vModel.vPath.empty() || !vParams.vModel.vHfRepo.empty()) {
    fWaitPassivelyOnIntegratedGpu(pArgv);
  }

  return mgs_server(vParams, pArgc, pArgv);
}

int mgs_server(common_params &pParams, int pArgc, char **pArgv) {
  bool vIsRunByCli = (pArgv == nullptr);

  common_models_handler vModelsHandler;

  // note: router mode also accepts -hf remote-preset, so we need to check that first
  if (!vIsRunByCli && !pParams.vModel.vHfRepo.empty()) {
    try {
      vModelsHandler = fCommonModelsHandlerInit(pParams);
      if (fCommonModelsHandlerIsPresetRepo(vModelsHandler)) {
        // apply the preset and start the server in router mode
        fCommonModelsHandlerApply(vModelsHandler, pParams);
      }
    } catch (const std::exception &vE) {
      SRV_ERR("failed to fetch model metadata: %s\n", vE.what());
      return 1;
    }
  }

  // router server never loads a model and must not touch the GPU
  const bool cIsRouterServer = pParams.vModel.vPath.empty() && pParams.vModel.vHfRepo.empty();

  // skip device enumeration so the CUDA primary context stays uncreated
  fCommonParamsPrintInfo(pParams, !cIsRouterServer);

  if (!cIsRouterServer) {
    // validate batch size for embeddings
    // embeddings require all tokens to be processed in a single ubatch
    if (pParams.vEmbedding && pParams.vNBatch > pParams.vNUbatch) {
      SRV_WRN("embeddings enabled with n_batch (%d) > n_ubatch (%d)\n", pParams.vNBatch, pParams.vNUbatch);
      SRV_WRN("setting n_batch = n_ubatch = %d to avoid assertion failure\n", pParams.vNUbatch);
      pParams.vNBatch = pParams.vNUbatch;
    }
  }

  // for consistency between server router mode and single-model mode, we set the same model name as alias
  auto vModelName = pParams.vModel.mGetName();
  if (pParams.sModelAlias.empty() && !vModelName.empty()) {
    pParams.sModelAlias.insert(vModelName);
  }

  // note: this is guaranteed to out-live ctx_http and tools
  server_mcp vMcpMgr;
  // local shell MCP server, must out-live ctx_http as well
  server_shell_mcp vShellMcp;

  // struct that contains mgs context and inference
  server_context vCtxServer;

  server_http_context vCtxHttp;
  if (!vCtxHttp.mInit(pParams)) {
    SRV_ERR("%s", "failed to initialize HTTP server\n");
    return 1;
  }

  //
  // Router
  //

  // register API routes
  server_child vChild; // only used in non-router mode
  server_routes vRoutes(pParams, vCtxServer);
  lServerTools vTools;

  std::optional<server_models_routes> vModelsRoutes{};
  if (cIsRouterServer) {
    // setup server instances manager
    try {
      vModelsRoutes.emplace(pParams, pArgc, pArgv);
    } catch (const std::exception &vE) {
      SRV_ERR("failed to initialize router models: %s\n", vE.what());
      return 1;
    }

    // proxy handlers
    // note: vRoutes.vGetHealth stays the same
    vRoutes.vGetMetrics = vModelsRoutes->vProxyGet;
    vRoutes.vPostProps = vModelsRoutes->vProxyPost;
    vRoutes.vPostCompletions = vModelsRoutes->vProxyPost;
    vRoutes.vPostCompletionsOai = vModelsRoutes->vProxyPost;
    vRoutes.vPostChatCompletions = vModelsRoutes->vProxyPost;
    vRoutes.vPostControl = vModelsRoutes->vProxyPost;
    vRoutes.vPostResponsesOai = vModelsRoutes->vProxyPost;
    vRoutes.vPostTranscriptionsOai = vModelsRoutes->vProxyPost;
    vRoutes.vPostAnthropicMessages = vModelsRoutes->vProxyPost;
    vRoutes.vPostAnthropicCountTokens = vModelsRoutes->vProxyPost;
    vRoutes.vPostInfill = vModelsRoutes->vProxyPost;
    vRoutes.vPostEmbeddings = vModelsRoutes->vProxyPost;
    vRoutes.vPostEmbeddingsOai = vModelsRoutes->vProxyPost;
    vRoutes.vPostRerank = vModelsRoutes->vProxyPost;
    vRoutes.vPostTokenize = vModelsRoutes->vProxyPost;
    vRoutes.vPostDetokenize = vModelsRoutes->vProxyPost;
    vRoutes.vPostApplyTemplate = vModelsRoutes->vProxyPost;
    vRoutes.vPostChatCompletionsTok = vModelsRoutes->vProxyPost;
    vRoutes.vPostResponsesTokOai = vModelsRoutes->vProxyPost;
    vRoutes.vGetLoraAdapters = vModelsRoutes->vProxyGet;
    vRoutes.vGetExpertCache = vModelsRoutes->vProxyGet;
    vRoutes.vPostExpertCache = vModelsRoutes->vProxyPost;
    vRoutes.vGetMonitor = vModelsRoutes->vGetRouterMonitor;
    vRoutes.vGetMonitorMemory = vModelsRoutes->vGetRouterMonitorMemory;
    // vRoutes.vGetMonitorUsage stays the same: CPU and GPU usage are system-wide
    vRoutes.vPostLoraAdapters = vModelsRoutes->vProxyPost;
    vRoutes.vGetSlots = vModelsRoutes->vProxyGet;
    vRoutes.vPostSlots = vModelsRoutes->vProxyPost;

    // custom routes for router
    vRoutes.vGetProps = vModelsRoutes->vGetRouterProps;
    vRoutes.vGetModels = vModelsRoutes->vGetRouterModels;

    vCtxHttp.mPost("/models", fWrapExceptions(vModelsRoutes->vPostRouterModels));
    vCtxHttp.mPost("/models/load", fWrapExceptions(vModelsRoutes->vPostRouterModelsLoad));
    vCtxHttp.mPost("/models/unload", fWrapExceptions(vModelsRoutes->vPostRouterModelsUnload));
    vCtxHttp.mGet("/models/sse", fWrapExceptions(vModelsRoutes->vGetRouterModelsSse));
    vCtxHttp.mDel("/models", fWrapExceptions(vModelsRoutes->vDelRouterModels));
    // folder of local models, chosen in Settings > Models of the Web UI and saved in the section modelsFolder of
    // ~/.mgs/config/ui-settings.json
    vCtxHttp.mGet("/models/folder", fWrapExceptions(vModelsRoutes->vGetRouterModelsFolder));
    vCtxHttp.mPost("/models/folder", fWrapExceptions(vModelsRoutes->vPostRouterModelsFolder));
    // subfolders of a folder of this computer, for the folder browser of Settings > Models
    vCtxHttp.mGet("/models/folder/browse", fWrapExceptions(vModelsRoutes->vGetRouterModelsFolderBrowse));
  }

  vCtxHttp.mGet("/health", fWrapExceptions(vRoutes.vGetHealth));    // public endpoint (no API key check)
  vCtxHttp.mGet("/metrics", fWrapExceptions(vRoutes.vGetMetrics));
  vCtxHttp.mGet("/props", fWrapExceptions(vRoutes.vGetProps));
  vCtxHttp.mPost("/props", fWrapExceptions(vRoutes.vPostProps));
  vCtxHttp.mGet("/v1/models", fWrapExceptions(vRoutes.vGetModels)); // public endpoint (no API key check)
  vCtxHttp.mPost("/completions", fWrapExceptions(vRoutes.vPostCompletions));
  vCtxHttp.mPost("/v1/completions", fWrapExceptions(vRoutes.vPostCompletionsOai));
  vCtxHttp.mPost("/v1/chat/completions", fWrapExceptions(vRoutes.vPostChatCompletions));
  vCtxHttp.mPost("/v1/chat/completions/control", fWrapExceptions(vRoutes.vPostControl));
  vCtxHttp.mPost("/v1/responses", fWrapExceptions(vRoutes.vPostResponsesOai));
  vCtxHttp.mPost("/v1/audio/transcriptions", fWrapExceptions(vRoutes.vPostTranscriptionsOai));
  vCtxHttp.mPost("/v1/messages", fWrapExceptions(vRoutes.vPostAnthropicMessages)); // anthropic messages API
  vCtxHttp.mPost("/infill", fWrapExceptions(vRoutes.vPostInfill));
  vCtxHttp.mPost("/embeddings", fWrapExceptions(vRoutes.vPostEmbeddings));
  vCtxHttp.mPost("/v1/embeddings", fWrapExceptions(vRoutes.vPostEmbeddingsOai));
  vCtxHttp.mPost("/rerank", fWrapExceptions(vRoutes.vPostRerank));
  vCtxHttp.mPost("/v1/rerank", fWrapExceptions(vRoutes.vPostRerank));
  vCtxHttp.mPost("/tokenize", fWrapExceptions(vRoutes.vPostTokenize));
  vCtxHttp.mPost("/detokenize", fWrapExceptions(vRoutes.vPostDetokenize));
  vCtxHttp.mPost("/apply-template", fWrapExceptions(vRoutes.vPostApplyTemplate));
  // token counting
  vCtxHttp.mPost("/v1/chat/completions/input_tokens", fWrapExceptions(vRoutes.vPostChatCompletionsTok));
  vCtxHttp.mPost("/v1/responses/input_tokens", fWrapExceptions(vRoutes.vPostResponsesTokOai));
  vCtxHttp.mPost("/v1/messages/count_tokens",
                 fWrapExceptions(vRoutes.vPostAnthropicCountTokens)); // anthropic token counting
  // LoRA adapters hotswap
  vCtxHttp.mGet("/lora-adapters", fWrapExceptions(vRoutes.vGetLoraAdapters));
  vCtxHttp.mPost("/lora-adapters", fWrapExceptions(vRoutes.vPostLoraAdapters));
  // MoE expert cache state and run-time size
  vCtxHttp.mGet("/expert-cache", fWrapExceptions(vRoutes.vGetExpertCache));
  vCtxHttp.mPost("/expert-cache", fWrapExceptions(vRoutes.vPostExpertCache));
  // Monitor: memory by purpose (GPUs and RAM), CPU and GPU usage
  vCtxHttp.mGet("/monitor", fWrapExceptions(vRoutes.vGetMonitor));
  vCtxHttp.mGet("/monitor/memory", fWrapExceptions(vRoutes.vGetMonitorMemory));
  vCtxHttp.mGet("/monitor/usage", fWrapExceptions(vRoutes.vGetMonitorUsage));
  // Web UI settings, kept in ~/.mgs/config/ui-settings.json; served by this process in every mode
  vCtxHttp.mGet("/ui/settings", fWrapExceptions(fUiSettingsMakeGetHandler()));
  vCtxHttp.mPost("/ui/settings", fWrapExceptions(fUiSettingsMakePostHandler()));
  // chats of the Web UI in ~/.mgs/chats/, shared by every browser (the fixed paths go before /chats/:id_chat)
  vCtxHttp.mGet("/chats", fWrapExceptions(fChatsMakeListHandler()));
  vCtxHttp.mGet("/chats/version", fWrapExceptions(fChatsMakeVersionHandler()));
  vCtxHttp.mGet("/chats/search", fWrapExceptions(fChatsMakeSearchHandler()));
  vCtxHttp.mGet("/chats/:id_chat", fWrapExceptions(fChatsMakeGetHandler()));
  vCtxHttp.mPost("/chats/:id_chat", fWrapExceptions(fChatsMakePostHandler()));
  vCtxHttp.mDel("/chats/:id_chat", fWrapExceptions(fChatsMakeDeleteHandler()));
  // Save & load slots
  vCtxHttp.mGet("/slots", fWrapExceptions(vRoutes.vGetSlots));
  vCtxHttp.mPost("/slots/:id_slot", fWrapExceptions(vRoutes.vPostSlots));

  // resumable streaming: a child binds the local session factories, the router binds
  // proxies that resolve the owning child, see server-stream.h
  server_http_context::handler_t vStreamGetH;
  server_http_context::handler_t vStreamsLookupH;
  server_http_context::handler_t vStreamDeleteH;
  if (cIsRouterServer) {
    vStreamGetH = vModelsRoutes->vRouterStreamGet;
    vStreamsLookupH = vModelsRoutes->vRouterStreamsLookup;
    vStreamDeleteH = vModelsRoutes->vRouterStreamDelete;
  } else {
    vStreamGetH = fServerStreamMakeGetHandler();
    vStreamsLookupH = fServerStreamMakeLookupHandler();
    vStreamDeleteH = fServerStreamMakeDeleteHandler();
  }
  vCtxHttp.mGet("/v1/stream", fWrapExceptions(vStreamGetH));
  vCtxHttp.mPost("/v1/streams/lookup", fWrapExceptions(vStreamsLookupH));
  vCtxHttp.mDel("/v1/stream", fWrapExceptions(vStreamDeleteH));

  // Google Cloud Platform (Vertex AI) compat
  vCtxHttp.mRegisterGcpCompat();

  // return 403 for disabled features
  server_http_context::handler_t vRes403 = [](const server_http_req &) {
    auto vRes = std::make_unique<server_http_res>();
    vRes->vStatus = 403;
    vRes->vData = fSafeJsonToStr({{"error",
                                   {
                                       {"message", "this feature is disabled"},
                                       {"type", "feature_disabled"},
                                   }}});
    return vRes;
  };

  if (pParams.vCorsOrigins == "*" && pParams.lApiKeys.empty()) {
    SRV_WRN("%s", "-----------------\n");
    SRV_WRN("%s", "CORS is set to allow all origins ('*') and no API key is set\n");
    SRV_WRN("%s", "this can be a security risk (cross-origin attacks); set --api-key or restrict --cors-origins\n");
    SRV_WRN("%s", "-----------------\n");
  }

  // CORS proxy (EXPERIMENTAL, only used by the Web UI for MCP)
  std::vector<std::string> lWarnNames;
  if (cIsRouterServer) {
    lWarnNames.push_back("router mode");
  }

  if (pParams.vUiMcpProxy) {
    vCtxHttp.mGet("/cors-proxy", fWrapExceptions(vProxyHandlerGet));
    vCtxHttp.mPost("/cors-proxy", fWrapExceptions(vProxyHandlerPost));
    lWarnNames.push_back("MCP proxy (experimental)");
  } else {
    vCtxHttp.mGet("/cors-proxy", fWrapExceptions(vRes403));
    vCtxHttp.mPost("/cors-proxy", fWrapExceptions(vRes403));
  }

  // Local shell MCP server (EXPERIMENTAL, added to the Web UI by default): turned on with --local-shell-mcp-server or
  // from the web UI (saved in ui-settings.json). Its endpoints always exist and answer 403 while it is off.
  const bool cHasApiKey = !pParams.lApiKeys.empty();
  // a router child never serves it: the router parent does, for every model
  if (pParams.vShellMcpServer || (!vChild.mIsChild() && fUiSettingsLocalShellMcpServer())) {
    vShellMcp.mSetEnabled(true, pParams.vShellMcpServer);
    lWarnNames.push_back(
        "local shell MCP server (experimental): anyone who can use the API can run commands as this user");
  }
  const auto fShellMcpGuard = [vRes403](const server_http_context::handler_t &pHandler) {
    return server_http_context::handler_t([vRes403, pHandler](const server_http_req &pRequest) {
      return fServerShellMcpEnabled() ? pHandler(pRequest) : vRes403(pRequest);
    });
  };
  const server_http_context::handler_t vShellUnsupported = [&vShellMcp](const server_http_req &pRequest) {
    return vShellMcp.mHandleUnsupported(pRequest);
  };
  vCtxHttp.mPost("/mcp/shell", fWrapExceptions(fShellMcpGuard([&vShellMcp](const server_http_req &pRequest) {
                   return vShellMcp.mHandlePost(pRequest);
                 })));
  vCtxHttp.mGet("/mcp/shell", fWrapExceptions(fShellMcpGuard(vShellUnsupported)));
  vCtxHttp.mDel("/mcp/shell", fWrapExceptions(fShellMcpGuard(vShellUnsupported)));
  vCtxHttp.mGet("/mcp/shell/askpass", fWrapExceptions(fShellMcpGuard([&vShellMcp](const server_http_req &pRequest) {
                  return vShellMcp.mHandleAskpassGet(pRequest);
                })));
  vCtxHttp.mPost("/mcp/shell/askpass", fWrapExceptions(fShellMcpGuard([&vShellMcp](const server_http_req &pRequest) {
                   return vShellMcp.mHandleAskpassPost(pRequest);
                 })));
  // POST {"enabled": true|false}: the switch of Settings > MCP servers. --local-shell-mcp-server prevails, and
  // without an API key only this computer may use it, so nobody else on the network can open a shell here
  vCtxHttp.mPost("/mcp/shell/enabled", fWrapExceptions([&vShellMcp, cHasApiKey](const server_http_req &pRequest) {
                   auto vRes = std::make_unique<server_http_res>();
                   const auto fError = [&vRes](const std::string &pMessage, enum error_type pType) {
                     const json cError = fFormatErrorResponse(pMessage, pType);
                     vRes->vStatus = cError.at("code").get<int>();
                     vRes->vData = fSafeJsonToStr({{"error", cError}});
                     return std::move(vRes);
                   };
                   const json cBody = json::parse(pRequest.vBody.empty() ? "{}" : pRequest.vBody, nullptr, false);
                   if (!cBody.is_object() || !cBody.contains("enabled") || !cBody.at("enabled").is_boolean()) {
                     return fError("expected {\"enabled\": true|false}", cErrorTypeInvalidRequest);
                   }
                   if (fServerShellMcpForced()) {
                     return fError("mgs was started with --local-shell-mcp-server, which keeps it on",
                                   cErrorTypeNotSupported);
                   }
                   if (!fServerShellMcpMayToggle(pRequest, cHasApiKey)) {
                     return fError("without an API key, only this computer can turn the local shell MCP server "
                                   "on or off",
                                   cErrorTypePermission);
                   }
                   const bool cEnabled = cBody.at("enabled").get<bool>();
                   std::string vError;
                   if (!fUiSettingsSaveLocalShellMcpServer(cEnabled, vError)) {
                     return fError("cannot save the setting: " + vError, cErrorTypeServer);
                   }
                   vShellMcp.mSetEnabled(cEnabled);
                   if (cEnabled) {
                     SRV_WRN("%s", "local shell MCP server turned on from the web UI: anyone who can use the API can "
                                   "run commands as this user\n");
                   } else {
                     SRV_INF("%s", "local shell MCP server turned off from the web UI\n");
                   }
                   vRes->vData = fSafeJsonToStr(fServerShellMcpProps(pRequest, cHasApiKey));
                   return vRes;
                 }));

  try {
    vMcpMgr.mStart(pParams);
  } catch (const std::exception &vE) {
    SRV_ERR("MCP starting failed: %s\n", vE.what());
    return 1;
  }

  if (!pParams.lServerTools.empty() || !vMcpMgr.empty()) {
    try {
      vTools.mSetup(pParams.lServerTools, vMcpMgr);
    } catch (const std::exception &vE) {
      SRV_ERR("tools setup failed: %s\n", vE.what());
      return 1;
    }
    vCtxHttp.mGet("/tools", fWrapExceptions(vTools.vHandleGet));
    vCtxHttp.mPost("/tools", fWrapExceptions(vTools.vHandlePost));
    if (!pParams.lServerTools.empty()) {
      lWarnNames.push_back("built-in tools (experimental)");
    }
    if (!vMcpMgr.empty()) {
      lWarnNames.push_back("MCP servers (experimental)");
    }
  } else {
    vCtxHttp.mGet("/tools", fWrapExceptions(vRes403));
    vCtxHttp.mPost("/tools", fWrapExceptions(vRes403));
  }

  if (lWarnNames.size() > 0) {
    SRV_WRN("%s", "-----------------\n");
    SRV_WRN("%s", "the following feature(s) are enabled:\n");
    for (const auto &vName : lWarnNames) {
      SRV_WRN("    %s\n", vName.c_str());
    }
    SRV_WRN("%s", "do not expose the server to untrusted environments\n");
    SRV_WRN("%s", "-----------------\n");
  }

  //
  // Handle downloading model
  //

  if (vChild.mIsChild() && vChild.mGetMode() == cServerChildModeDownload) {
    return vChild.mRunDownload(pParams);
  } else if (!cIsRouterServer && !vIsRunByCli) {
    // single-model mode (NOT spawned by router)
    // if this is invoked by CLI, model downloading should be already handled
    try {
      fCommonModelsHandlerApply(vModelsHandler, pParams);
    } catch (const std::exception &vE) {
      SRV_ERR("failed to download model: %s\n", vE.what());
      return 1;
    }
  }

  //
  // mgs only loads Mixture of Experts (MoE) models: a dense model stops the server before anything is loaded
  //

  if (!cIsRouterServer) {
    // MTP is enabled first, so that a separate MTP GGUF passed with --spec-draft-model is checked as an MTP head
    fCommonMaybeEnableMtp(pParams);

    // a file whose metadata cannot be read (-1) passes, so that the regular loading reports why
    auto fIsDense = [](const std::string &pPath, const mgs_model_params &pMparams) {
      return !pPath.empty() && mgs_model_file_is_moe(pPath.c_str(), pMparams) == 0;
    };
    // a head-only MTP GGUF is never a main model
    mgs_model_params vMparams = fCommonModelParamsToMgs(pParams);
    vMparams.load_mtp = false;
    std::string vDenseModel;
    if (fIsDense(pParams.vModel.vPath, vMparams)) {
      vDenseModel = "the model '" + pParams.vModel.vPath + "'";
    } else if (pParams.vSpeculative.mHasDft()) {
      common_params vParamsDft = fCommonBaseParamsToSpeculative(pParams);
      if (fIsDense(vParamsDft.vModel.vPath, fCommonModelParamsToMgs(vParamsDft))) {
        vDenseModel = "the draft model '" + vParamsDft.vModel.vPath + "'";
      }
    }
    if (!vDenseModel.empty()) {
      SRV_ERR("%s is not a Mixture of Experts (MoE) model\n", vDenseModel.c_str());
      SRV_ERR("%s", "mgs only loads MoE models; dense models are not supported\n");
      SRV_ERR("%s", "exiting due to a dense model\n");
      return 1;
    }
  }

  //
  // Start the server
  //

  std::function<void()> vCleanUp;

  if (cIsRouterServer) {
    SRV_INF("%s", "starting server in router mode. models will be automatically loaded on-demand\n");

    vCleanUp = [&vModelsRoutes, &vMcpMgr]() {
      SRV_INF("%s: cleaning up before exit...\n", __func__);
      // stop the session GC first, it finalizes live sessions and wakes pending readers
      fServerStreamSessionManagerStop();
      if (vModelsRoutes.has_value()) {
        vModelsRoutes->vStopping.store(true); // maybe redundant, but just to be safe
        vModelsRoutes->vModels.mUnloadAll();
      }
      vMcpMgr.mShutdown();
      mgs_backend_free();
    };

    if (!vCtxHttp.mStart()) {
      vCleanUp();
      SRV_ERR("%s", "exiting due to HTTP server error\n");
      return 1;
    }
    vCtxHttp.vIsReady.store(true);

    fShutdownHandler = [&](int) {
      if (vModelsRoutes.has_value()) {
        // important to disconnect any SSE clients
        vModelsRoutes->vStopping.store(true);
      }
      vMcpMgr.mShutdown();
      vCtxHttp.mStop();
    };

  } else {
    // setup clean up function, to be called before exit
    vCleanUp = [&vCtxHttp, &vCtxServer, &vMcpMgr]() {
      SRV_INF("%s: cleaning up before exit...\n", __func__);
      // stop the session GC first, it finalizes live sessions and wakes pending readers
      fServerStreamSessionManagerStop();
      vCtxHttp.mStop();
      vCtxServer.mTerminate();
      vMcpMgr.mShutdown();
      mgs_backend_free();
    };

    // start the HTTP server before loading the model to be able to serve /health requests
    if (!vCtxHttp.mStart()) {
      vCleanUp();
      SRV_ERR("%s", "exiting due to HTTP server error\n");
      return 1;
    }

    // setup communication child --> router if necessary
    if (vChild.mIsChild()) {
      vCtxServer.mSetStateCallback(
          [&](server_state pState, json pPayload) { vChild.mNotifyToRouter(fServerStateToStr(pState), pPayload); });
    }

    if (!vCtxServer.mLoadModel(pParams)) {
      vCleanUp();
      if (vCtxHttp.vThread.joinable()) {
        vCtxHttp.vThread.join();
      }
      SRV_ERR("%s", "exiting due to model loading error\n");
      return 1;
    }

    vRoutes.mUpdateMeta(vCtxServer);
    vCtxHttp.vIsReady.store(true);

    SRV_INF("%s", "model loaded\n");

    fShutdownHandler = [&](int) {
      vMcpMgr.mShutdown();
      // this will unblock mStartLoop()
      vCtxServer.mTerminate();
    };
  }

  // register signal handler if not running by CLI
  if (!vIsRunByCli) {
#if defined(__unix__) || (defined(__APPLE__) && defined(__MACH__))
    fStartShutdownThread();
    struct sigaction vSigintAction;
    vSigintAction.sa_handler = fSignalHandler;
    sigemptyset(&vSigintAction.sa_mask);
    vSigintAction.sa_flags = 0;
    sigaction(SIGINT, &vSigintAction, NULL);
    sigaction(SIGTERM, &vSigintAction, NULL);
#elif defined(_WIN32)
    auto vConsoleCtrlHandler =
        +[](DWORD pCtrlType) -> BOOL { return (pCtrlType == CTRL_C_EVENT) ? (fSignalHandler(SIGINT), true) : false; };
    SetConsoleCtrlHandler(reinterpret_cast<PHANDLER_ROUTINE>(vConsoleCtrlHandler), true);
#endif
  }

  SRV_INF("listening on %s\n", vCtxHttp.vListeningAddress.c_str());

  if (cIsRouterServer) {
    if (!pParams.vModelsPresetHf.empty()) {
      SRV_WRN("NOTE: using preset.ini from HF repo '%s'\n", pParams.vModelsPresetHf.c_str());
      SRV_WRN("%s", "      please only use presets that you can trust! Unknown presets may be unsafe\n");
    }

    if (vCtxHttp.vThread.joinable()) {
      vCtxHttp.vThread.join(); // keep the main thread alive
    }

    // when the HTTP server stops, clean up and exit
    vCleanUp();
  } else {
    // optionally, notify router server that this instance is ready
    std::thread vMonitorThread;
    if (vChild.mIsChild()) {
      vMonitorThread = vChild.mSetup(fShutdownHandler);
      vChild.mNotifyToRouter(fServerStateToStr(cServerStateReady), vRoutes.mGetModelInfo());
    }

    // this call blocks the main thread until queue_tasks.terminate() is called
    vCtxServer.mStartLoop();

    vCleanUp();
    if (vCtxHttp.vThread.joinable()) {
      vCtxHttp.vThread.join();
    }
    if (vMonitorThread.joinable()) {
      vMonitorThread.join();
    }

    auto *vLlCtx = vCtxServer.mGetMgsContext();
    if (vLlCtx != nullptr) {
      fCommonMemoryBreakdownPrint(vLlCtx);
    }
  }

  return 0;
}
