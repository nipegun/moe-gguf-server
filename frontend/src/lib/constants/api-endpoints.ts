import { fApiUrl } from '../api-url';
export const dApiModels = {
  LIST: fApiUrl('/v1/models'),
  LOAD: fApiUrl('/models/load'),
  UNLOAD: fApiUrl('/models/unload'),
  SSE: fApiUrl('/models/sse')
};

// chat completion routes, the control route drives realtime inference (e.g. end reasoning)
export const dApiChat = {
  COMPLETIONS: fApiUrl('./v1/chat/completions'),
  CONTROL: fApiUrl('./v1/chat/completions/control')
};

// slot introspection, requires the --slots flag on the server
export const dApiSlots = {
  LIST: fApiUrl('./slots')
};

export const dApiTools = {
  LIST: fApiUrl('/tools'),
  EXECUTE: fApiUrl('/tools')
};

// resumable stream routes, the conv::model identity travels as the conv_id query param
// because model names can contain slashes that a path segment cannot carry
// resume retry cadence while the owning model is still loading (server answers 503)
export const cStreamResumeRetryMs = 2000;

export const dApiStream = {
  BASE: fApiUrl('./v1/stream'),
  LOOKUP: fApiUrl('./v1/streams/lookup')
};

/** CORS proxy endpoint path */
export const cCorsProxyEndpoint = fApiUrl('/cors-proxy');

/** Local shell MCP server (--local-shell-mcp-server), served by this same server */
export const cShellMcpEndpoint = fApiUrl('/mcp/shell');
/** sudo password prompts of its commands: GET ?call=<id> for the pending prompt, POST to answer it */
export const cShellMcpAskpassEndpoint = fApiUrl('/mcp/shell/askpass');
