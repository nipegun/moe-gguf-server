#include "server-api-doc.h"
#include "server-http.h"
#include "server-common.h"
#include "server-schema.h"
#include "server-doc-assets.h"

#include <cpp-httplib/httplib.h>
#include <cctype>

static json fFieldSchema(const server_schema::field &pField) {
  json dSchema = {{"description", pField.vDesc}};
  if (dynamic_cast<const server_schema::field_bool *>(&pField)) {
    dSchema["type"] = "boolean";
  } else if (dynamic_cast<const server_schema::field_str *>(&pField)) {
    dSchema["type"] = "string";
  } else if (dynamic_cast<const server_schema::field_num<float> *>(&pField)) {
    dSchema["type"] = "number";
  } else if (dynamic_cast<const server_schema::field_num<int32_t> *>(&pField) ||
             dynamic_cast<const server_schema::field_num<int64_t> *>(&pField)) {
    dSchema["type"] = "integer";
  } else if (const auto *vNested = dynamic_cast<const server_schema::field_nested *>(&pField)) {
    dSchema["type"] = "object";
    dSchema["properties"] = json::object();
    for (const auto &vField : vNested->lSubfields) {
      for (const char *vName : vField->lName)
        dSchema["properties"][vName] = fFieldSchema(*vField);
    }
  }
  return dSchema;
}

static json fCompletionProperties() {
  common_params vDefaults;
  task_params vTask;
  const auto aFields = server_schema::fMakeMgsCmplSchema(vDefaults, vTask);
  json dProperties = {{"model", {{"type", "string"}, {"description", "Model identifier in router mode."}}}};
  for (const auto &vField : aFields) {
    for (const char *vName : vField->lName)
      dProperties[vName] = fFieldSchema(*vField);
  }
  return dProperties;
}

static std::string fRouteDescription(const std::string &pPath) {
  const std::map<std::string, std::string> dDescriptions = {
      {"/health", "Check whether the server has finished loading the model."},
      {"/metrics", "Prometheus metrics; requires --metrics. Includes cumulative speculative decoding counters: "
                   "mgs:spec_decode_num_draft_tokens_total, mgs:spec_decode_num_accepted_tokens_total, "
                   "mgs:spec_decode_num_drafts_total and "
                   "mgs:spec_decode_num_accepted_tokens_per_pos_total with a zero-based position label. "
                   "Draft counters update when generation finishes; the per-position series is absent before "
                   "the first completed speculative request. In router mode select a model with ?model=MODEL_ID."},
      {"/props", "Read server properties and default generation settings. Updates depend on server mode."},
      {"/models", "List, download or remove models. Mutations require router mode."},
      {"/models/load", "Load an available model in a router child process."},
      {"/models/unload", "Unload a router model without deleting its files."},
      {"/models/sse", "Stream router model state changes as server-sent events."},
      {"/completions", "Generate text completions from a prompt."},
      {"/chat/completions", "Generate an assistant response from messages; optionally stream tokens."},
      {"/chat/completions/control", "Control an active generation, for example ending reasoning."},
      {"/responses", "Generate using the OpenAI Responses-compatible format."},
      {"/messages", "Generate using the Anthropic-compatible Messages format."},
      {"/audio/transcriptions", "Transcribe an audio upload using a compatible model."},
      {"/infill", "Complete code between input_prefix and input_suffix, with optional input_extra."},
      {"/embeddings", "Create embedding vectors from input text or token arrays."},
      {"/rerank", "Rank documents by relevance to a query using a reranking model."},
      {"/tokenize", "Convert content into token identifiers and optional token pieces."},
      {"/detokenize", "Convert token identifiers back into text."},
      {"/apply-template", "Render the model's chat template for the supplied messages."},
      {"/lora-adapters", "Inspect or update the scales of loaded LoRA adapters."},
      {"/expert-cache", "Inspect the MoE expert cache (size, hit ratio, uploads, CPU experts) or request a new size "
                        "with {\"size_mib\": N} (-1 automatic, 0 disabled); the next request applies it."},
      {"/monitor", "Monitor report: CPU usage (global, per physical processor with its model and per logical CPU), "
                   "the system load average, RAM and every GPU with its usage, "
                   "temperature and power, plus the memory of the loaded models split by purpose (segments "
                   "mtp_weights, expert_weights, dense_weights, draft_weights, kv_cache, draft_kv_cache, compute, "
                   "expert_cache, prefetch, prompt_cache and other, in bytes). In router mode it adds every loaded "
                   "child; it never loads or wakes a model."},
      {"/monitor/memory", "Memory snapshots of the loaded models ({\"models\": [...]}): segments per GPU and in RAM, "
                          "resident memory and expert cache state, as published by each inference loop."},
      {"/monitor/usage", "CPU and GPU usage in percent, for status bars: the global CPU usage, every physical "
                         "processor and every GPU. Answers while the model loads and never touches the model."},
      {"/slots", "Inspect inference slots; requires slot monitoring."},
      {"/slots/:id_slot", "Save, restore or erase a slot cache using id_slot and action."},
      {"/stream", "Resume or delete a retained stream using its conversation identity."},
      {"/streams/lookup", "Locate retained streams for conversation_ids."},
      {"/cors-proxy", "Proxy HTTP to the url query parameter; requires explicit enablement."},
      {"/mcp/shell",
       "Local shell MCP server (MCP Streamable HTTP, JSON responses, no session); requires --local-shell-mcp-server. "
       "POST one JSON-RPC message: initialize, ping, tools/list or tools/call. It exposes bash.run, zsh.run, sh.run "
       "and psh.run, only for the shells found when the server started; arguments: command (required), cwd and "
       "timeout (seconds, default 60, maximum 3600). A tools/call result carries, as structuredContent and as the "
       "JSON text of its content, exit_code (null when unknown), stdout, stderr, signal, timed_out, cancelled, "
       "timeout, seconds, shell and cwd. Commands run as the server's user. Notifications return 202; GET and "
       "DELETE return 405. Requests with an Origin header must come from a page served by this server."},
      {"/mcp/shell/askpass",
       "Password prompts of sudo in commands of the local shell MCP server (Linux); requires "
       "--local-shell-mcp-server. GET ?call=<id> returns the prompt pending in the call whose tools/call carried that "
       "id in _meta[\"mgs/callId\"] ({\"pending\": {\"id\", \"prompt\"}} or {\"pending\": null}); POST {call, id, "
       "password} sends the password to sudo, and POST {call, id, cancel: true} makes sudo fail. The password only "
       "goes to sudo, never to the model or the logs. Requests with an Origin header must come from a page served "
       "by this server."},
      {"/tools", "List built-in tools or execute a tool; requires explicit enablement."},
      {"/ui/settings",
       "Settings of the Web UI, kept by the server in ~/.mgs/config/ui-settings.json of the user that runs mgs and "
       "shared by every browser. GET returns {\"settings\": DOCUMENT}, or {\"settings\": null} before the first "
       "save; POST {\"settings\": DOCUMENT} replaces the whole file (mode 0600). The document holds the sections "
       "config, userOverrides, language, favoriteModels, disabledToolKeys, alwaysAllowedTools, "
       "reasoningEffortDefault and dismissedRecommendedMcpServers; it never holds the API key. Answers while the "
       "model loads."}};
  const std::string vPath = pPath.rfind("/v1/", 0) == 0 ? pPath.substr(3) : pPath;
  const auto vEntry = dDescriptions.find(vPath);
  if (vEntry != dDescriptions.end())
    return vEntry->second;
  if (vPath.find("input_tokens") != std::string::npos || vPath.find("count_tokens") != std::string::npos)
    return "Count input tokens using the matching chat, responses or messages format.";
  return "Optional compatibility route registered according to process configuration.";
}

static json fOperation(const std::string &pPath, const std::string &pMethod) {
  std::string vIdentifier = pMethod;
  for (unsigned char vCharacter : pPath) {
    vIdentifier += std::isalnum(vCharacter) ? static_cast<char>(vCharacter) : '_';
  }
  const bool cPublic = pPath == "/health" || pPath == "/v1/health" || pPath == "/models" || pPath == "/v1/models";
  json dOperation = {
      {"operationId", vIdentifier},
      {"summary", fRouteDescription(pPath)},
      {"description",
       "Registered mgs operation. Optional capabilities return 403 when disabled. Model-dependent operations "
       "return 503 while the model loads. Router mode accepts a model selector."},
      {"security", cPublic && pMethod == "get"
                       ? json::array()
                       : json::array({{{"BearerAuth", json::array()}}, {{"ApiKeyAuth", json::array()}}})},
      {"parameters", json::array()},
      {"responses",
       {{"200",
         {{"description", "Successful response; streaming operations emit server-sent events."},
          {"content",
           {{"application/json", {{"schema", json::object()}}},
            {"text/event-stream", {{"schema", {{"type", "string"}}}}}}}}},
        {"400", {{"description", "Invalid parameters or unsupported operation."}}},
        {"401", {{"description", "Missing or invalid API key."}}},
        {"403",
         {{"description", "This optional feature is disabled, or, with tools or MCP servers enabled, a non-GET "
                          "request came from a foreign Origin (neither localhost nor this server)."}}},
        {"404", {{"description", "The model, slot or stream does not exist."}}},
        {"500", {{"description", "Server error."}}},
        {"503", {{"description", "Model loading or server unavailable."}}}}}};
  if (pPath.find("/models") != std::string::npos) {
    dOperation["tags"] = {"Models"};
  } else if (pPath.find("stream") != std::string::npos) {
    dOperation["tags"] = {"Streams"};
  } else if (pPath == "/health" || pPath == "/v1/health" || pPath == "/metrics" || pPath == "/props" ||
             pPath.rfind("/monitor", 0) == 0) {
    dOperation["tags"] = {"Monitoring"};
  } else if (pPath == "/ui/settings") {
    dOperation["tags"] = {"Web interface"};
    dOperation["responses"]["400"] = {{"description", "The body is not {\"settings\": {...}} or is too large."}};
    dOperation["responses"]["500"] = {{"description", "The settings file could not be read or written."}};
  } else {
    dOperation["tags"] = {"Inference and tools"};
  }
  if (pPath == "/mcp/shell/askpass") {
    dOperation["responses"]["403"] = {
        {"description", "The shell MCP server is disabled, or the Origin header does not match this server."}};
    dOperation["responses"]["404"] = {{"description", "No prompt with that call and id is pending."}};
    dOperation["responses"]["415"] = {{"description", "Content-Type is not application/json."}};
  }
  if (pPath == "/mcp/shell") {
    dOperation["responses"]["202"] = {{"description", "Notification or response accepted, without a body."}};
    dOperation["responses"]["403"] = {
        {"description", "The shell MCP server is disabled, or the Origin header does not match this server."}};
    dOperation["responses"]["405"] = {{"description", "Only POST is supported (no SSE stream and no sessions)."}};
    dOperation["responses"]["415"] = {{"description", "Content-Type is not application/json."}};
  }
  if (pMethod == "get" && pPath.rfind("/monitor", 0) != 0 && pPath.rfind("/mcp/shell", 0) != 0 &&
      pPath != "/ui/settings") {
    dOperation["parameters"].push_back({{"name", "model"},
                                        {"in", "query"},
                                        {"schema", {{"type", "string"}}},
                                        {"description", "Router model identifier, when applicable."}});
  }
  if (pPath == "/mcp/shell/askpass" && pMethod == "get") {
    dOperation["parameters"].push_back({{"name", "call"},
                                        {"in", "query"},
                                        {"required", true},
                                        {"schema", {{"type", "string"}, {"pattern", "^[A-Za-z0-9_-]{8,128}$"}}},
                                        {"description", "Call id sent in _meta[\"mgs/callId\"] of tools/call."}});
  }
  if (pPath == "/cors-proxy") {
    dOperation["parameters"].push_back({{"name", "url"},
                                        {"in", "query"},
                                        {"required", true},
                                        {"schema", {{"type", "string"}, {"format", "uri"}}},
                                        {"description", "HTTP or HTTPS target URL, without embedded credentials."}});
    dOperation["description"] =
        "Fetches the target for fetch() calls of the Web UI. The response is inert: it carries "
        "Content-Security-Policy sandbox, X-Content-Type-Options nosniff and Content-Disposition attachment, and the "
        "upstream cookie, CORS, CSP, redirect and service-worker headers are dropped.";
    dOperation["responses"]["403"] = {
        {"description", "The proxy is disabled, or the request is a navigation or an embedding (Sec-Fetch-Mode "
                        "navigate, Sec-Fetch-Dest document, iframe, frame, embed, object or a worker)."}};
  }
  if (pPath == "/tools" && pMethod == "post") {
    dOperation["responses"]["415"] = {{"description", "Content-Type is not application/json."}};
  }
  if ((pPath == "/models" || pPath == "/v1/models") && pMethod == "get") {
    dOperation["parameters"].push_back(
        {{"name", "reload"},
         {"in", "query"},
         {"schema", {{"type", "string"}}},
         {"description", "Router mode: any non-empty value rescans the model sources; it then requires the API key "
                         "when keys are configured."}});
  }
  if (pPath == "/models" && pMethod == "delete") {
    dOperation["parameters"].push_back(
        {{"name", "model"}, {"in", "query"}, {"required", true}, {"schema", {{"type", "string"}}}});
  }
  if (pPath == "/v1/stream") {
    dOperation["parameters"].push_back({{"name", "conv_id"},
                                        {"in", "query"},
                                        {"required", true},
                                        {"schema", {{"type", "string"}}},
                                        {"description", "Conversation identity, optionally suffixed with ::model."}});
    if (pMethod == "get") {
      dOperation["parameters"].push_back({{"name", "from"},
                                          {"in", "query"},
                                          {"schema", {{"type", "integer"}, {"minimum", 0}}},
                                          {"description", "Resume cursor returned by the stream."}});
    }
  }
  if (pPath.find(":id_slot") != std::string::npos) {
    dOperation["parameters"].push_back(
        {{"name", "id_slot"}, {"in", "path"}, {"required", true}, {"schema", {{"type", "integer"}}}});
    dOperation["parameters"].push_back({{"name", "action"},
                                        {"in", "query"},
                                        {"required", true},
                                        {"schema", {{"type", "string"}, {"enum", {"save", "restore", "erase"}}}}});
  }
  if (pMethod == "post") {
    json dSchema = {{"type", "object"}, {"additionalProperties", true}};
    json dExample = json::object();
    if (pPath.find("chat/completions") != std::string::npos && pPath.find("control") == std::string::npos) {
      dSchema["required"] = {"messages"};
      dSchema["properties"] = {{"model", {{"type", "string"}}},
                               {"messages",
                                {{"type", "array"},
                                 {"items",
                                  {{"type", "object"},
                                   {"required", {"role", "content"}},
                                   {"properties", {{"role", {{"type", "string"}}}, {"content", json::object()}}}}}}},
                               {"stream", {{"type", "boolean"}}},
                               {"temperature", {{"type", "number"}}},
                               {"max_tokens", {{"type", "integer"}}}};
      dExample = {{"messages", {{{"role", "user"}, {"content", "Hello"}}}}, {"stream", false}};
      json dProperties = fCompletionProperties();
      dProperties.update(dSchema["properties"]);
      dSchema["properties"] = std::move(dProperties);
    } else if (pPath == "/completions" || pPath == "/v1/completions") {
      dSchema["properties"] = fCompletionProperties();
      dSchema["properties"]["prompt"] = {{"description", "Input text or token IDs."}};
      dSchema["required"] = {"prompt"};
      dExample = {{"prompt", "Hello"}, {"n_predict", 64}};
    } else if (pPath == "/tokenize") {
      dSchema["properties"] = {{"content", {{"type", "string"}}},
                               {"add_special", {{"type", "boolean"}}},
                               {"with_pieces", {{"type", "boolean"}}}};
      dSchema["required"] = {"content"};
      dExample = {{"content", "Hello"}};
    } else if (pPath == "/detokenize") {
      dSchema["properties"] = {{"tokens", {{"type", "array"}, {"items", {{"type", "integer"}}}}}};
      dSchema["required"] = {"tokens"};
      dExample = {{"tokens", json::array()}};
    } else if (pPath == "/models" || pPath == "/models/load" || pPath == "/models/unload") {
      dSchema["properties"] = {{"model", {{"type", "string"}}}};
      dSchema["required"] = {"model"};
      dExample = {{"model", "model-id"}};
    } else if (pPath == "/tools") {
      dSchema["properties"] = {
          {"tool", {{"type", "string"}}}, {"params", {{"type", "object"}}}, {"stream", {{"type", "boolean"}}}};
      dSchema["required"] = {"tool"};
    } else if (pPath == "/mcp/shell") {
      dSchema["properties"] = {{"jsonrpc", {{"type", "string"}, {"enum", {"2.0"}}}},
                               {"id", {{"description", "Request identifier; omitted in notifications."}}},
                               {"method", {{"type", "string"}}},
                               {"params", {{"type", "object"}}}};
      dSchema["required"] = {"jsonrpc", "method"};
      dExample = {{"jsonrpc", "2.0"},
                  {"id", 1},
                  {"method", "tools/call"},
                  {"params", {{"name", "bash.run"}, {"arguments", {{"command", "uname -a"}, {"timeout", 30}}}}}};
    } else if (pPath == "/mcp/shell/askpass") {
      dSchema["properties"] = {{"call", {{"type", "string"}, {"description", "Call id sent in _meta of tools/call."}}},
                               {"id", {{"type", "string"}, {"description", "Id of the pending prompt."}}},
                               {"password", {{"type", "string"}, {"format", "password"}, {"maxLength", 4096}}},
                               {"cancel", {{"type", "boolean"}}}};
      dSchema["required"] = {"call", "id"};
      dExample = {{"call", "3f2b9c1e-7a4d-4e8b-9f0a-1c2d3e4f5a6b"}, {"id", "9c41e0b27d5a8f63"}, {"cancel", true}};
    } else if (pPath == "/ui/settings") {
      dSchema["properties"] = {
          {"settings",
           {{"type", "object"},
            {"additionalProperties", true},
            {"description", "Whole settings document; it replaces the saved one."}}}};
      dSchema["required"] = {"settings"};
      dExample = {{"settings",
                   {{"config", {{"theme", "system"}, {"agenticMaxTurns", 50}}},
                    {"userOverrides", json::array()},
                    {"language", "es-ES"}}}};
    } else if (pPath == "/v1/streams/lookup") {
      dSchema["properties"] = {{"conversation_ids", {{"type", "array"}, {"items", {{"type", "string"}}}}}};
      dSchema["required"] = {"conversation_ids"};
    } else if (pPath.find("embedding") != std::string::npos) {
      dSchema["properties"] = {{"model", {{"type", "string"}}},
                               {"input", {{"description", "Text, a text array or token IDs."}}}};
      dExample = {{"input", "Text to embed"}};
    } else if (pPath.find("rerank") != std::string::npos) {
      dSchema["properties"] = {{"query", {{"type", "string"}}},
                               {"documents", {{"type", "array"}, {"items", {{"type", "string"}}}}},
                               {"top_n", {{"type", "integer"}}}};
      dSchema["required"] = {"query", "documents"};
      dExample = {{"query", "Question"}, {"documents", {"First document", "Second document"}}};
    } else if (pPath == "/infill") {
      dSchema["properties"] = fCompletionProperties();
      dSchema["properties"]["input_prefix"] = {{"type", "string"}};
      dSchema["properties"]["input_suffix"] = {{"type", "string"}};
      dSchema["required"] = {"input_prefix", "input_suffix"};
    } else if (pPath == "/v1/chat/completions/control") {
      dSchema["properties"] = {{"id", {{"type", "string"}}},
                               {"action", {{"type", "string"}, {"enum", {"reasoning_end"}}}}};
      dSchema["required"] = {"id", "action"};
      dExample = {{"id", "completion-id"}, {"action", "reasoning_end"}};
    } else if (pPath.find("responses") != std::string::npos) {
      dSchema["properties"] = fCompletionProperties();
      dSchema["properties"]["input"] = {{"description", "Text or a Responses-compatible input item array."}};
      dSchema["properties"]["instructions"] = {{"type", "string"}};
      dSchema["required"] = {"input"};
      dExample = {{"input", "Hello"}};
    } else if (pPath.find("messages") != std::string::npos || pPath == "/apply-template") {
      dSchema["properties"] = fCompletionProperties();
      dSchema["properties"]["messages"] = {{"type", "array"},
                                           {"items", {{"type", "object"}, {"additionalProperties", true}}}};
      dSchema["required"] = {"messages"};
      dExample = {{"messages", {{{"role", "user"}, {"content", "Hello"}}}}, {"max_tokens", 128}};
    } else if (pPath == "/lora-adapters") {
      dSchema = {{"type", "array"},
                 {"items",
                  {{"type", "object"},
                   {"required", {"id", "scale"}},
                   {"properties", {{"id", {{"type", "integer"}}}, {"scale", {{"type", "number"}}}}}}}};
      dExample = {{{"id", 0}, {"scale", 1.0}}};
    } else if (pPath.find("/slots/") == 0) {
      dSchema["properties"] = {
          {"filename", {{"type", "string"}, {"description", "Cache filename for save/restore; not used by erase."}}}};
    } else if (pPath.find("predict") != std::string::npos) {
      dSchema["properties"] = {
          {"instances", {{"type", "array"}, {"items", {{"type", "object"}, {"additionalProperties", true}}}}}};
      dSchema["required"] = {"instances"};
    }
    dOperation["requestBody"] = {{"required", pPath != "/props"},
                                 {"content", {{"application/json", {{"schema", dSchema}, {"example", dExample}}}}}};
    if (pPath == "/cors-proxy") {
      dOperation["requestBody"] = {
          {"required", false},
          {"content", {{"application/octet-stream", {{"schema", {{"type", "string"}, {"format", "binary"}}}}}}}};
    }
    if (pPath.find("audio/transcriptions") != std::string::npos) {
      dOperation["requestBody"]["content"] = {
          {"multipart/form-data",
           {{"schema",
             {{"type", "object"},
              {"required", {"file"}},
              {"properties",
               {{"file", {{"type", "string"}, {"format", "binary"}}}, {"model", {{"type", "string"}}}}}}}}}};
    }
  }
  if (pPath == "/metrics") {
    dOperation["responses"]["200"]["content"] = {{"text/plain", {{"schema", {{"type", "string"}}}}}};
  }
  return dOperation;
}

void fRegisterApiDocumentation(httplib::Server &pServer, server_http_context &pContext) {
  pServer.Get("/api/doc",
              [](const httplib::Request &, httplib::Response &pResponse) { pResponse.set_redirect("/api/doc/", 308); });
  pServer.Get("/api/doc/", [](const httplib::Request &, httplib::Response &pResponse) {
    pResponse.set_content(cSwaggerPage, "text/html; charset=utf-8");
  });
  pServer.Get("/api/doc/swagger-ui.css", [](const httplib::Request &, httplib::Response &pResponse) {
    pResponse.set_content(cSwaggerStyle, "text/css; charset=utf-8");
  });
  pServer.Get("/api/doc/swagger-ui-bundle.js", [](const httplib::Request &, httplib::Response &pResponse) {
    pResponse.set_content(cSwaggerScript, "application/javascript; charset=utf-8");
  });
  pServer.Get("/api/doc/openapi.json", [&pContext](const httplib::Request &, httplib::Response &pResponse) {
    json dDocument = {
        {"openapi", "3.0.3"},
        {"info",
         {{"title", "moe-gguf-server"},
          {"version", "1.0.0"},
          {"description",
           "Live API inventory generated from registered routes. Inference accepts additional mgs parameters. "
           "All API operations are under /api/. Set the OpenAI client base URL to the configured API prefix followed "
           "by /v1. Authentication is required only when the server is configured with API keys."}}},
        {"servers", {{{"url", pContext.vPathPrefix}}}},
        {"paths", json::object()},
        {"components",
         {{"securitySchemes",
           {{"BearerAuth", {{"type", "http"}, {"scheme", "bearer"}}},
            {"ApiKeyAuth", {{"type", "apiKey"}, {"in", "header"}, {"name", "X-Api-Key"}}}}}}}};
    for (const auto &[vPath, sMethods] : pContext.dApiMethods) {
      std::string vOpenApiPath = vPath;
      const auto vPosition = vOpenApiPath.find(":id_slot");
      if (vPosition != std::string::npos)
        vOpenApiPath.replace(vPosition, 8, "{id_slot}");
      for (const auto &vMethod : sMethods) {
        dDocument["paths"][vOpenApiPath][vMethod] = fOperation(vPath, vMethod);
      }
    }
    for (const auto &vPath : {"/api/config.js", "/api/doc", "/api/doc/", "/api/doc/openapi.json",
                              "/api/doc/swagger-ui.css", "/api/doc/swagger-ui-bundle.js"}) {
      dDocument["paths"][vPath] = {{"servers", {{{"url", "/"}}}},
                                   {"get",
                                    {{"summary", "Public API documentation or client configuration"},
                                     {"security", json::array()},
                                     {"responses",
                                      {{"200", {{"description", "Documentation asset or API prefix configuration."}}},
                                       {"308", {{"description", "Canonical documentation URL."}}}}}}}};
    }
    pResponse.set_header("Cache-Control", "no-store");
    pResponse.set_content(dDocument.dump(2), "application/json; charset=utf-8");
  });
}
