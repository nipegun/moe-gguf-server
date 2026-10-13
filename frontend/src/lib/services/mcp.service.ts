import { fTranslate } from '../i18n';
import { Client } from '@modelcontextprotocol/sdk/client';
import {
  StreamableHTTPClientTransport,
  StreamableHTTPError
} from '@modelcontextprotocol/sdk/client/streamableHttp.js';
import { SSEClientTransport } from '@modelcontextprotocol/sdk/client/sse.js';
import { WebSocketClientTransport } from '@modelcontextprotocol/sdk/client/websocket.js';
import type {
  Tool,
  Prompt,
  GetPromptResult,
  ListChangedHandlers
} from '@modelcontextprotocol/sdk/types.js';
import type { Transport } from '@modelcontextprotocol/sdk/shared/transport.js';
import {
  cDefaultMcpConfig,
  cDefaultClientVersion,
  cDefaultImageMimeType,
  cCorsProxyHeaderPrefix,
  dMcpPartialRedactHeaders,
  cCorsProxyEndpoint
} from '$lib/constants';
import {
  MCPConnectionPhase,
  MCPLogLevel,
  MCPTransportType,
  MCPContentType,
  MCPRefType
} from '$lib/enums';
import type {
  MCPServerConfig,
  MCPResourceIcon,
  ToolCallParams,
  ToolExecutionResult,
  Implementation,
  ClientCapabilities,
  MCPConnection,
  MCPPhaseCallback,
  MCPConnectionLog,
  MCPServerInfo,
  MCPResource,
  MCPResourceTemplate,
  MCPResourceContent,
  MCPReadResourceResult
} from '$lib/types';
import {
  buildProxiedUrl,
  buildProxiedHeaders,
  getAuthHeaders,
  sanitizeHeaders,
  throwIfAborted,
  isAbortError,
  createBase64DataUrl,
  getRequestUrl,
  getRequestMethod,
  getRequestBody,
  summarizeRequestBody,
  formatDiagnosticErrorMessage,
  extractJsonRpcMethods,
  type RequestBodySummary
} from '$lib/utils';

interface ToolResultContentItem {
  type: string;
  text?: string;
  data?: string;
  mimeType?: string;
  resource?: { text?: string; blob?: string; uri?: string };
}

interface ToolCallResult {
  content?: ToolResultContentItem[];
  isError?: boolean;
  _meta?: Record<string, unknown>;
}

interface DiagnosticRequestDetails {
  url: string;
  method: string;
  credentials?: RequestCredentials;
  mode?: RequestMode;
  headers: Record<string, string>;
  body: RequestBodySummary;
  jsonRpcMethods?: string[];
}

export class MCPService {
  /**
   * Create a connection log entry for phase tracking.
   *
   * @param pPhase - The connection phase this log belongs to
   * @param pMessage - Human-readable log message
   * @param pLevel - Log severity level (default: INFO)
   * @param pDetails - Optional structured details for debugging
   * @returns Formatted connection log entry
   */
  private static mCreateLog(
    pPhase: MCPConnectionPhase,
    pMessage: string,
    pLevel: MCPLogLevel = MCPLogLevel.INFO,
    pDetails?: unknown
  ): MCPConnectionLog {
    return {
      timestamp: new Date(),
      phase: pPhase,
      message: pMessage,
      level: pLevel,
      details: pDetails
    };
  }

  private static mCreateDiagnosticRequestDetails(
    pInput: RequestInfo | URL,
    pInit: RequestInit | undefined,
    pBaseInit: RequestInit,
    pRequestHeaders: Headers,
    pExtraRedactedHeaders?: Iterable<string>
  ): DiagnosticRequestDetails {
    const cBody = getRequestBody(pInput, pInit);
    const dDetails: DiagnosticRequestDetails = {
      url: getRequestUrl(pInput),
      method: getRequestMethod(pInput, pInit, pBaseInit).toUpperCase(),
      credentials: pInit?.credentials ?? pBaseInit.credentials,
      mode: pInit?.mode ?? pBaseInit.mode,
      headers: sanitizeHeaders(pRequestHeaders, pExtraRedactedHeaders, dMcpPartialRedactHeaders),
      body: summarizeRequestBody(cBody)
    };
    const cJsonRpcMethods = extractJsonRpcMethods(cBody);

    if (cJsonRpcMethods) {
      dDetails.jsonRpcMethods = cJsonRpcMethods;
    }

    return dDetails;
  }

  private static mAddRequestHeaders(
    pRequestHeaders: Headers,
    pHeaders: HeadersInit,
    pUseProxy: boolean
  ) {
    for (const [cKey, cValue] of new Headers(pHeaders).entries()) {
      const cProxiedKey =
        pUseProxy && !cKey.toLowerCase().startsWith(cCorsProxyHeaderPrefix)
          ? `${cCorsProxyHeaderPrefix}${cKey}`
          : cKey;
      pRequestHeaders.set(cProxiedKey, cValue);
    }
  }

  private static mSummarizeError(pError: unknown): Record<string, unknown> {
    if (pError instanceof Error) {
      return {
        name: pError.name,
        message: pError.message,
        cause:
          pError.cause instanceof Error
            ? { name: pError.cause.name, message: pError.cause.message }
            : pError.cause,
        stack: pError.stack?.split('\n').slice(0, 6).join('\n')
      };
    }

    return { value: String(pError) };
  }

  private static mGetBrowserContext(
    pTargetUrl: URL,
    pUseProxy: boolean
  ): Record<string, unknown> | undefined {
    if (typeof window === 'undefined') {
      return undefined;
    }

    return {
      location: window.location.href,
      origin: window.location.origin,
      protocol: window.location.protocol,
      isSecureContext: window.isSecureContext,
      targetOrigin: pTargetUrl.origin,
      targetProtocol: pTargetUrl.protocol,
      sameOrigin: window.location.origin === pTargetUrl.origin,
      useProxy: pUseProxy
    };
  }

  private static mGetConnectionHints(
    pTargetUrl: URL,
    pConfig: MCPServerConfig,
    pError: unknown
  ): string[] {
    const aHints: string[] = [];
    const cMessage = pError instanceof Error ? pError.message : String(pError);
    const cHeaderNames = Object.keys(pConfig.headers ?? {});

    if (typeof window !== 'undefined') {
      if (
        window.location.protocol === 'https:' &&
        pTargetUrl.protocol === 'http:' &&
        !pConfig.useProxy
      ) {
        aHints.push(fTranslate('text_d0e6a8974b06'));
      }

      if (window.location.origin !== pTargetUrl.origin && !pConfig.useProxy) {
        aHints.push(fTranslate('text_441d6da995a7'));
      }
    }

    if (cHeaderNames.length > 0) {
      aHints.push(fTranslate('text_fedd9cfc304f', { p0: cHeaderNames.join(', ') }));
    }

    if (pConfig.credentials && pConfig.credentials !== 'omit') {
      aHints.push(fTranslate('text_bcfe5dc653f5'));
    }

    if (cMessage.includes('Failed to fetch')) {
      aHints.push(
        '"Failed to fetch" is a browser-level network failure. Common causes are CORS rejection, mixed-content blocking, certificate/TLS errors, DNS failures, or nothing listening on the target port.'
      );
    }

    return aHints;
  }

  private static mCreateDiagnosticFetch(
    pServerName: string,
    pConfig: MCPServerConfig,
    pBaseInit: RequestInit,
    pTargetUrl: URL,
    pUseProxy: boolean,
    pOnLog?: (pLog: MCPConnectionLog) => void
  ): {
    fetch: typeof fetch;
    disable: () => void;
  } {
    let vEnabled = true;
    const fLogIfEnabled = (pLog: MCPConnectionLog) => {
      if (vEnabled) {
        pOnLog?.(pLog);
      }
    };

    return {
      fetch: async (pInput, pInit) => {
        if (pUseProxy && typeof window !== 'undefined') {
          let vRequestUrlStr = '';
          if (typeof pInput === 'string') {
            vRequestUrlStr = pInput;
          } else if (pInput instanceof URL) {
            vRequestUrlStr = pInput.href;
          }

          if (vRequestUrlStr) {
            const cParsedRequestUrl = new URL(vRequestUrlStr, window.location.origin);
            if (
              cParsedRequestUrl.origin === window.location.origin &&
              !cParsedRequestUrl.pathname.includes(cCorsProxyEndpoint)
            ) {
              const cOriginalConfigUrl = new URL(pConfig.url);
              const cRealTargetUrl = new URL(
                cParsedRequestUrl.pathname + cParsedRequestUrl.search,
                cOriginalConfigUrl.origin
              );
              const cProxiedUrl = buildProxiedUrl(cRealTargetUrl.href);

              if (typeof pInput === 'string') {
                pInput = cProxiedUrl.href;
              } else if (pInput instanceof URL) {
                pInput = cProxiedUrl;
              }
            }
          }
        }

        const cStartedAt = performance.now();
        const cRequestHeaders = new Headers(pBaseInit.headers);

        if (typeof Request !== 'undefined' && pInput instanceof Request) {
          this.mAddRequestHeaders(cRequestHeaders, pInput.headers, pUseProxy);
        }

        if (pInit?.headers) {
          this.mAddRequestHeaders(cRequestHeaders, pInit.headers, pUseProxy);
        }

        const cRequest = this.mCreateDiagnosticRequestDetails(
          pInput,
          pInit,
          pBaseInit,
          cRequestHeaders,
          Object.keys(pConfig.headers ?? {})
        );
        const { method: cMethod, url: cUrl } = cRequest;

        fLogIfEnabled(
          this.mCreateLog(
            MCPConnectionPhase.INITIALIZING,
            `HTTP ${cMethod} ${cUrl}`,
            MCPLogLevel.INFO,
            {
              serverName: pServerName,
              request: cRequest
            }
          )
        );

        if (cMethod === 'DELETE' && cUrl.includes(cCorsProxyEndpoint)) {
          const cResponse = new Response(null, { status: 200, statusText: 'OK' });

          fLogIfEnabled(
            this.mCreateLog(
              MCPConnectionPhase.INITIALIZING,
              `HTTP 200 ${cMethod} ${cUrl} (fake response)`,
              MCPLogLevel.INFO,
              {
                response: {
                  url: cUrl,
                  status: cResponse.status,
                  statusText: cResponse.statusText,
                  durationMs: 0,
                  isFake: true
                }
              }
            )
          );

          // fake response, bypass real fetch()
          return cResponse;
        }

        try {
          const cResponse = await fetch(pInput, {
            ...pBaseInit,
            ...pInit,
            headers: cRequestHeaders
          });
          const cDurationMs = Math.round(performance.now() - cStartedAt);

          fLogIfEnabled(
            this.mCreateLog(
              MCPConnectionPhase.INITIALIZING,
              `HTTP ${cResponse.status} ${cMethod} ${cUrl} (${cDurationMs}ms)`,
              cResponse.ok ? MCPLogLevel.INFO : MCPLogLevel.WARN,
              {
                response: {
                  url: cUrl,
                  status: cResponse.status,
                  statusText: cResponse.statusText,
                  headers: sanitizeHeaders(cResponse.headers, undefined, dMcpPartialRedactHeaders),
                  durationMs: cDurationMs
                }
              }
            )
          );

          return cResponse;
        } catch (vError) {
          const cDurationMs = Math.round(performance.now() - cStartedAt);

          fLogIfEnabled(
            this.mCreateLog(
              MCPConnectionPhase.ERROR,
              `HTTP ${cMethod} ${cUrl} failed: ${formatDiagnosticErrorMessage(vError)}`,
              MCPLogLevel.ERROR,
              {
                serverName: pServerName,
                request: cRequest,
                error: this.mSummarizeError(vError),
                browser: this.mGetBrowserContext(pTargetUrl, pUseProxy),
                hints: this.mGetConnectionHints(pTargetUrl, pConfig, vError),
                durationMs: cDurationMs
              }
            )
          );

          throw vError;
        }
      },
      disable: () => {
        vEnabled = false;
      }
    };
  }

  /**
   * Detect if an error indicates an expired/invalidated MCP session.
   * Per MCP spec 2025-11-25: HTTP 404 means session invalidated, client MUST
   * discard its session ID and start a new session with a fresh initialize request.
   *
   * @param pError - The caught error to inspect
   * @returns true if the error is a StreamableHTTP 404 (session not found)
   */
  static mIsSessionExpiredError(pError: unknown): boolean {
    return pError instanceof StreamableHTTPError && pError.code === 404;
  }

  /**
   * Create transport based on server configuration.
   * Supports WebSocket, StreamableHTTP (modern), and SSE (legacy) transports.
   * When `useProxy` is enabled, routes HTTP requests through the mgs CORS proxy.
   *
   * **Fallback Order:**
   * 1. WebSocket — if explicitly configured (no CORS proxy support)
   * 2. StreamableHTTP — default for HTTP connections
   * 3. SSE — automatic fallback if StreamableHTTP fails
   *
   * @param pConfig - Server configuration with url, transport type, proxy, and auth settings
   * @returns Object containing the created transport and the transport type used
   * @throws {Error} If url is missing, WebSocket + proxy combination, or all transports fail
   */
  static mCreateTransport(
    pServerName: string,
    pConfig: MCPServerConfig,
    pOnLog?: (pLog: MCPConnectionLog) => void
  ): {
    transport: Transport;
    type: MCPTransportType;
    stopPhaseLogging: () => void;
  } {
    if (!pConfig.url) {
      throw new Error(fTranslate('text_44a9a196bae7'));
    }

    const cUseProxy = pConfig.useProxy ?? false;
    const dRequestInit: RequestInit = {};

    if (pConfig.headers) {
      dRequestInit.headers = pConfig.useProxy
        ? buildProxiedHeaders(pConfig.headers)
        : pConfig.headers;
    }

    if (cUseProxy) {
      dRequestInit.headers = {
        ...getAuthHeaders(),
        ...(dRequestInit.headers as Record<string, string>)
      };
    } else if (typeof window !== 'undefined' && pConfig.transport !== MCPTransportType.WEBSOCKET) {
      // MCP servers served by this same server (such as the local shell MCP server) take the UI's API key
      let vSameOrigin = false;
      try {
        vSameOrigin = new URL(pConfig.url).origin === window.location.origin;
      } catch {
        vSameOrigin = false;
      }
      if (vSameOrigin) {
        dRequestInit.headers = {
          ...getAuthHeaders(),
          ...(dRequestInit.headers as Record<string, string>)
        };
      }
    }

    if (pConfig.credentials) {
      dRequestInit.credentials = pConfig.credentials;
    }

    if (pConfig.transport === MCPTransportType.WEBSOCKET) {
      if (cUseProxy) {
        throw new Error(fTranslate('text_79c12c60af8a'));
      }

      const cUrl = new URL(pConfig.url);

      if (import.meta.env.DEV && import.meta.env.VITE_DEBUG) {
        console.log(`[MCPService] Creating WebSocket transport for ${cUrl.href}`);
      }

      return {
        transport: new WebSocketClientTransport(cUrl),
        type: MCPTransportType.WEBSOCKET,
        stopPhaseLogging: () => {}
      };
    }

    if (pConfig.transport === MCPTransportType.SSE) {
      const cUrl = cUseProxy ? buildProxiedUrl(pConfig.url) : new URL(pConfig.url);
      const { fetch: cDiagnosticFetch, disable: cStopPhaseLogging } = this.mCreateDiagnosticFetch(
        pServerName,
        pConfig,
        dRequestInit,
        cUrl,
        cUseProxy,
        pOnLog
      );

      if (import.meta.env.DEV && import.meta.env.VITE_DEBUG) {
        console.log(`[MCPService] Creating SSE transport for ${cUrl.href}`);
      }

      return {
        transport: new SSEClientTransport(cUrl, {
          requestInit: dRequestInit,
          fetch: cDiagnosticFetch,
          eventSourceInit: { fetch: cDiagnosticFetch }
        }),
        type: MCPTransportType.SSE,
        stopPhaseLogging: cStopPhaseLogging
      };
    }

    const cUrl = cUseProxy ? buildProxiedUrl(pConfig.url) : new URL(pConfig.url);
    const { fetch: cDiagnosticFetch, disable: cStopPhaseLogging } = this.mCreateDiagnosticFetch(
      pServerName,
      pConfig,
      dRequestInit,
      cUrl,
      cUseProxy,
      pOnLog
    );

    if (cUseProxy && import.meta.env.DEV && import.meta.env.VITE_DEBUG) {
      console.log(`[MCPService] Using CORS proxy for ${pConfig.url} -> ${cUrl.href}`);
    }

    try {
      if (import.meta.env.DEV && import.meta.env.VITE_DEBUG) {
        console.log(`[MCPService] Creating StreamableHTTP transport for ${cUrl.href}`);
      }

      return {
        transport: new StreamableHTTPClientTransport(cUrl, {
          requestInit: dRequestInit,
          fetch: cDiagnosticFetch
        }),
        type: MCPTransportType.STREAMABLE_HTTP,
        stopPhaseLogging: cStopPhaseLogging
      };
    } catch (vHttpError) {
      console.warn(`[MCPService] StreamableHTTP failed, trying SSE transport...`, vHttpError);

      try {
        return {
          transport: new SSEClientTransport(cUrl, {
            requestInit: dRequestInit,
            fetch: cDiagnosticFetch,
            eventSourceInit: { fetch: cDiagnosticFetch }
          }),
          type: MCPTransportType.SSE,
          stopPhaseLogging: cStopPhaseLogging
        };
      } catch (vSseError) {
        const cHttpMsg = vHttpError instanceof Error ? vHttpError.message : String(vHttpError);
        const cSseMsg = vSseError instanceof Error ? vSseError.message : String(vSseError);

        throw new Error(fTranslate('text_f72422fbcabb', { p0: cHttpMsg, p1: cSseMsg }));
      }
    }
  }

  /**
   * Extract server info from SDK Implementation type.
   * Normalizes the SDK's server version response into our MCPServerInfo type.
   *
   * @param pImpl - Raw Implementation object from MCP SDK
   * @returns Normalized server info or undefined if input is empty
   */
  private static mExtractServerInfo(pImpl: Implementation | undefined): MCPServerInfo | undefined {
    if (!pImpl) {
      return undefined;
    }

    return {
      name: pImpl.name,
      version: pImpl.version,
      title: pImpl.title,
      description: pImpl.description,
      websiteUrl: pImpl.websiteUrl,
      icons: pImpl.icons?.map((pIcon: MCPResourceIcon) => ({
        src: pIcon.src,
        mimeType: pIcon.mimeType,
        sizes: pIcon.sizes,
        theme: pIcon.theme
      }))
    };
  }

  /**
   * Connect to a single MCP server with detailed phase tracking.
   *
   * Performs the full MCP connection lifecycle:
   * 1. Transport creation (with automatic fallback)
   * 2. Client initialization and capability exchange
   * 3. Tool discovery via `listTools`
   *
   * Reports progress via `onPhase` callback at each step, enabling
   * UI progress indicators during connection.
   *
   * @param pServerName - Display name for the server (used in logging)
   * @param pServerConfig - Server URL, transport type, proxy, and auth configuration
   * @param pClientInfo - Optional client identification (defaults to app info)
   * @param pCapabilities - Optional client capability declaration
   * @param pOnPhase - Optional callback for connection phase progress updates
   * @param pListChangedHandlers - Optional handlers for server-initiated list change notifications
   * @returns Full connection object with client, transport, tools, server info, and timing
   * @throws {Error} If transport creation or connection fails
   */
  static async mConnect(
    pServerName: string,
    pServerConfig: MCPServerConfig,
    pClientInfo?: Implementation,
    pCapabilities?: ClientCapabilities,
    pOnPhase?: MCPPhaseCallback,
    pListChangedHandlers?: ListChangedHandlers
  ): Promise<MCPConnection> {
    const cStartTime = performance.now();
    const cEffectiveClientInfo = pClientInfo ?? cDefaultMcpConfig.clientInfo;
    const cEffectiveCapabilities = pCapabilities ?? cDefaultMcpConfig.capabilities;

    // Phase: Creating transport
    pOnPhase?.(
      MCPConnectionPhase.TRANSPORT_CREATING,
      this.mCreateLog(
        MCPConnectionPhase.TRANSPORT_CREATING,
        fTranslate('text_800a501b965c', { p0: pServerConfig.url })
      )
    );

    if (import.meta.env.DEV && import.meta.env.VITE_DEBUG) {
      console.log(`[MCPService][${pServerName}] Creating transport...`);
    }

    const {
      transport: cTransport,
      type: cTransportType,
      stopPhaseLogging: cStopPhaseLogging
    } = this.mCreateTransport(pServerName, pServerConfig, (pLog) => pOnPhase?.(pLog.phase, pLog));

    // Setup WebSocket reconnection handler
    if (cTransportType === MCPTransportType.WEBSOCKET) {
      cTransport.onclose = () => {
        console.log(`[MCPService][${pServerName}] WebSocket closed, notifying for reconnection`);
        pOnPhase?.(
          MCPConnectionPhase.DISCONNECTED,
          this.mCreateLog(MCPConnectionPhase.DISCONNECTED, fTranslate('text_d00ec64829f9'))
        );
      };
    }

    // Phase: Transport ready
    pOnPhase?.(
      MCPConnectionPhase.TRANSPORT_READY,
      this.mCreateLog(
        MCPConnectionPhase.TRANSPORT_READY,
        fTranslate('text_5c7940366183', { p0: cTransportType })
      ),
      { transportType: cTransportType }
    );

    const cClient = new Client(
      {
        name: cEffectiveClientInfo.name,
        version: cEffectiveClientInfo.version ?? cDefaultClientVersion
      },
      {
        capabilities: cEffectiveCapabilities,
        listChanged: pListChangedHandlers
      }
    );

    const fRuntimeErrorHandler = (pError: Error) => {
      // the SDK reports any post initialize error here, including the abort we trigger
      // ourselves on the next health check cycle, on tab unload, or on server teardown.
      // these are lifecycle aborts, not actionable errors, so we keep them out of the red console.
      // the SDK wraps the original AbortError in a generic Error like
      //   "SSE stream disconnected: AbortError: The operation was aborted."
      // which isAbortError cannot recognize by name alone, so we also pattern match on the message
      if (isAbortError(pError)) {
        return;
      }
      const cMsg = pError?.message ?? '';
      if (
        /SSE stream disconnected:.*AbortError/i.test(cMsg) ||
        /AbortError: .*aborted/i.test(cMsg) ||
        /stream locked by a reader/i.test(cMsg)
      ) {
        return;
      }
      console.error(`[MCPService][${pServerName}] Protocol error after initialize:`, pError);
    };

    cClient.onerror = (pError) => {
      pOnPhase?.(
        MCPConnectionPhase.ERROR,
        this.mCreateLog(
          MCPConnectionPhase.ERROR,
          fTranslate('text_af6452e5dcf6', { p0: pError.message }),
          MCPLogLevel.ERROR,
          {
            error: this.mSummarizeError(pError)
          }
        )
      );
    };

    // Phase: Initializing
    pOnPhase?.(
      MCPConnectionPhase.INITIALIZING,
      this.mCreateLog(MCPConnectionPhase.INITIALIZING, fTranslate('text_6999519ebdf0'))
    );

    // The SDK timeout only covers the initialize request, not transport.start(),
    // which can hang forever on an unreachable host (SSE endpoint wait, WebSocket
    // handshake, proxied fetch). This race bounds the whole handshake and closes
    // the transport on expiry so the underlying fetch or socket is aborted.
    const cHandshakeTimeoutMs =
      pServerConfig.handshakeTimeoutMs ?? cDefaultMcpConfig.connectionTimeoutMs;

    try {
      let vHandshakeTimer: ReturnType<typeof setTimeout> | undefined;
      const cHandshakeDeadline = new Promise<never>((pIgnored, pReject) => {
        vHandshakeTimer = setTimeout(() => {
          void cTransport.close().catch(() => {});
          pReject(
            new Error(
              fTranslate('text_3da5d921e88d', { p0: Math.round(cHandshakeTimeoutMs / 1000) })
            )
          );
        }, cHandshakeTimeoutMs);
      });

      try {
        await Promise.race([
          cClient.connect(cTransport, { timeout: cHandshakeTimeoutMs }),
          cHandshakeDeadline
        ]);
      } finally {
        clearTimeout(vHandshakeTimer);
      }

      // Transport diagnostics are only for the initial handshake, not long-lived traffic.
      cStopPhaseLogging();
      cClient.onerror = fRuntimeErrorHandler;
    } catch (vError) {
      cClient.onerror = fRuntimeErrorHandler;
      const cUrl =
        (pServerConfig.useProxy ?? false)
          ? buildProxiedUrl(pServerConfig.url)
          : new URL(pServerConfig.url);

      pOnPhase?.(
        MCPConnectionPhase.ERROR,
        this.mCreateLog(
          MCPConnectionPhase.ERROR,
          fTranslate('text_439c2f42472a', {
            p0: vError instanceof Error ? vError.message : String(vError)
          }),
          MCPLogLevel.ERROR,
          {
            error: this.mSummarizeError(vError),
            config: {
              serverName: pServerName,
              configuredUrl: pServerConfig.url,
              effectiveUrl: cUrl.href,
              transportType: cTransportType,
              useProxy: pServerConfig.useProxy ?? false,
              headers: sanitizeHeaders(
                pServerConfig.headers,
                Object.keys(pServerConfig.headers ?? {}),
                dMcpPartialRedactHeaders
              ),
              credentials: pServerConfig.credentials
            },
            browser: this.mGetBrowserContext(cUrl, pServerConfig.useProxy ?? false),
            hints: this.mGetConnectionHints(cUrl, pServerConfig, vError)
          }
        )
      );

      throw vError;
    }

    const cServerVersion = cClient.getServerVersion();
    const cServerCapabilities = cClient.getServerCapabilities();
    const cInstructions = cClient.getInstructions();
    const cServerInfo = this.mExtractServerInfo(cServerVersion);

    // Phase: Capabilities exchanged
    pOnPhase?.(
      MCPConnectionPhase.CAPABILITIES_EXCHANGED,
      this.mCreateLog(
        MCPConnectionPhase.CAPABILITIES_EXCHANGED,
        fTranslate('text_5c6361cca49a'),
        MCPLogLevel.INFO,
        {
          serverCapabilities: cServerCapabilities,
          serverInfo: cServerInfo
        }
      ),
      {
        serverInfo: cServerInfo,
        serverCapabilities: cServerCapabilities,
        clientCapabilities: cEffectiveCapabilities,
        instructions: cInstructions
      }
    );

    // Phase: Listing tools
    pOnPhase?.(
      MCPConnectionPhase.LISTING_TOOLS,
      this.mCreateLog(MCPConnectionPhase.LISTING_TOOLS, fTranslate('text_dabc13bcbe04'))
    );

    if (import.meta.env.DEV && import.meta.env.VITE_DEBUG) {
      console.log(`[MCPService][${pServerName}] Connected, listing tools...`);
    }

    const cTools = await this.mListTools({
      client: cClient,
      transport: cTransport,
      tools: [],
      serverName: pServerName,
      transportType: cTransportType,
      connectionTimeMs: 0,
      requestTimeoutMs:
        pServerConfig.requestTimeoutMs ?? cDefaultMcpConfig.requestTimeoutSeconds * 1000
    });

    const cConnectionTimeMs = Math.round(performance.now() - cStartTime);

    // Phase: Connected
    pOnPhase?.(
      MCPConnectionPhase.CONNECTED,
      this.mCreateLog(
        MCPConnectionPhase.CONNECTED,
        fTranslate('text_abd9a9fb4190', { p0: cTools.length, p1: cConnectionTimeMs })
      )
    );
    if (import.meta.env.DEV && import.meta.env.VITE_DEBUG) {
      console.log(
        `[MCPService][${pServerName}] Initialization complete with ${cTools.length} tools in ${cConnectionTimeMs}ms`
      );
    }

    return {
      client: cClient,
      transport: cTransport,
      tools: cTools,
      serverName: pServerName,
      transportType: cTransportType,
      serverInfo: cServerInfo,
      serverCapabilities: cServerCapabilities,
      clientCapabilities: cEffectiveCapabilities,
      protocolVersion: cDefaultMcpConfig.protocolVersion,
      instructions: cInstructions,
      connectionTimeMs: cConnectionTimeMs,
      requestTimeoutMs:
        pServerConfig.requestTimeoutMs ?? cDefaultMcpConfig.requestTimeoutSeconds * 1000
    };
  }

  /**
   * Disconnect from a server.
   * Clears the `onclose` handler to prevent reconnection attempts on voluntary disconnect.
   *
   * @param pConnection - The active MCP connection to close
   */
  static async mDisconnect(pConnection: MCPConnection): Promise<void> {
    if (import.meta.env.DEV && import.meta.env.VITE_DEBUG) {
      console.log(`[MCPService][${pConnection.serverName}] Disconnecting...`);
    }

    try {
      // Terminate the session first for streamable-http transports to cleanly
      // close streams, matching the inspector's disconnect flow.
      if (pConnection.transport instanceof StreamableHTTPClientTransport) {
        await pConnection.transport.terminateSession();
      }

      // Clear error handlers before closing to prevent noise from expected
      // abort errors during shutdown. The inspector avoids this entirely
      // by not setting onerror, but since we use it for protocol logging,
      // we must clear it before disconnect.
      pConnection.client.onerror = undefined;
      if (pConnection.transport.onclose) {
        pConnection.transport.onclose = undefined;
      }

      await pConnection.client.close();
    } catch (vError) {
      console.warn(`[MCPService][${pConnection.serverName}] Error during disconnect:`, vError);
    }
  }

  /**
   * List tools from a connection.
   * Silently returns empty array on failure (logged as warning).
   *
   * @param pConnection - The MCP connection to query
   * @returns Array of available tools, or empty array on error
   */
  static async mListTools(pConnection: MCPConnection): Promise<Tool[]> {
    try {
      const cResult = await pConnection.client.listTools();

      return cResult.tools ?? [];
    } catch (vError) {
      // Let session-expired errors propagate for reconnection handling
      if (this.mIsSessionExpiredError(vError)) {
        throw vError;
      }

      console.warn(`[MCPService][${pConnection.serverName}] Failed to list tools:`, vError);

      return [];
    }
  }

  /**
   * List prompts from a connection.
   * Silently returns empty array on failure (logged as warning).
   *
   * @param pConnection - The MCP connection to query
   * @returns Array of available prompts, or empty array on error
   */
  static async mListPrompts(pConnection: MCPConnection): Promise<Prompt[]> {
    try {
      const cResult = await pConnection.client.listPrompts();

      return cResult.prompts ?? [];
    } catch (vError) {
      // Let session-expired errors propagate for reconnection handling
      if (this.mIsSessionExpiredError(vError)) {
        throw vError;
      }

      console.warn(`[MCPService][${pConnection.serverName}] Failed to list prompts:`, vError);

      return [];
    }
  }

  /**
   * Get a specific prompt with arguments.
   * Unlike list operations, this throws on failure since the caller explicitly
   * requested a specific prompt and needs to handle the error.
   *
   * @param pConnection - The MCP connection to use
   * @param pName - The prompt name to retrieve
   * @param pArgs - Optional key-value arguments to pass to the prompt
   * @returns The prompt result with messages and metadata
   * @throws {Error} If the prompt retrieval fails
   */
  static async mGetPrompt(
    pConnection: MCPConnection,
    pName: string,
    pArgs?: Record<string, string>
  ): Promise<GetPromptResult> {
    try {
      return await pConnection.client.getPrompt({ name: pName, arguments: pArgs });
    } catch (vError) {
      console.error(`[MCPService][${pConnection.serverName}] Failed to get prompt:`, vError);

      throw vError;
    }
  }

  /**
   * Execute a tool call on a connection.
   * Supports abort signal for cancellable operations (e.g., when user stops generation).
   * Formats the raw tool result into a string representation.
   *
   * @param pConnection - The MCP connection to execute against
   * @param pParams - Tool name and arguments to execute
   * @param pSignal - Optional AbortSignal for cancellation support
   * @returns Formatted tool execution result with content string and error flag
   * @throws {Error} If tool execution fails or is aborted
   */
  static async mCallTool(
    pConnection: MCPConnection,
    pParams: ToolCallParams,
    pSignal?: AbortSignal
  ): Promise<ToolExecutionResult> {
    throwIfAborted(pSignal);

    try {
      const cResult = await pConnection.client.callTool(
        { name: pParams.name, arguments: pParams.arguments, _meta: pParams._meta },
        undefined,
        { signal: pSignal, timeout: pConnection.requestTimeoutMs }
      );

      return {
        content: this.mFormatToolResult(cResult as ToolCallResult),
        isError: (cResult as ToolCallResult).isError ?? false
      };
    } catch (vError) {
      if (isAbortError(vError)) {
        throw vError;
      }

      // Let session-expired errors propagate unwrapped for reconnection handling
      if (this.mIsSessionExpiredError(vError)) {
        throw vError;
      }

      const cMessage = vError instanceof Error ? vError.message : String(vError);

      throw new Error(
        fTranslate('text_58f056dbcf4b', {
          p0: pParams.name,
          p1: pConnection.serverName,
          p2: cMessage
        }),
        { cause: vError instanceof Error ? vError : undefined }
      );
    }
  }

  /**
   * Format tool result content items to a single string.
   * Handles text, image (base64 data URL), and embedded resource content types.
   *
   * @param pResult - Raw tool call result from MCP SDK
   * @returns Concatenated string representation of all content items
   */
  private static mFormatToolResult(pResult: ToolCallResult): string {
    const cContent = pResult.content;
    if (!Array.isArray(cContent)) return '';

    return cContent
      .map((pItem) => this.mFormatSingleContent(pItem))
      .filter(Boolean)
      .join('\n');
  }

  private static mFormatSingleContent(pContent: ToolResultContentItem): string {
    if (pContent.type === MCPContentType.TEXT && pContent.text) {
      return pContent.text;
    }

    if (pContent.type === MCPContentType.IMAGE && pContent.data) {
      return createBase64DataUrl(pContent.mimeType ?? cDefaultImageMimeType, pContent.data);
    }

    if (pContent.type === MCPContentType.RESOURCE && pContent.resource) {
      const cResource = pContent.resource;

      if (cResource.text) return cResource.text;
      if (cResource.blob) return cResource.blob;

      return JSON.stringify(cResource);
    }

    if (pContent.data && pContent.mimeType) {
      return createBase64DataUrl(pContent.mimeType, pContent.data);
    }

    return JSON.stringify(pContent);
  }

  /**
   *
   *
   * Completions Operations
   *
   *
   */

  /**
   * Request completion suggestions from a server.
   * Used for autocompleting prompt arguments or resource URI templates.
   *
   * @param pConnection - The MCP connection to use
   * @param pRef - Reference to the prompt or resource template
   * @param pArgument - The argument being completed (name and current value)
   * @returns Completion result with suggested values
   */
  static async mComplete(
    pConnection: MCPConnection,
    pRef: { type: MCPRefType.PROMPT; name: string } | { type: MCPRefType.RESOURCE; uri: string },
    pArgument: { name: string; value: string }
  ): Promise<{ values: string[]; total?: number; hasMore?: boolean } | null> {
    try {
      const cResult = await pConnection.client.complete({
        ref: pRef,
        argument: pArgument
      });

      return cResult.completion;
    } catch (vError) {
      console.error(`[MCPService] Failed to get completions:`, vError);

      return null;
    }
  }

  /**
   *
   *
   * Resources Operations
   *
   *
   */

  /**
   * List resources from a connection.
   * @param pConnection - The MCP connection to use
   * @param pCursor - Optional pagination cursor
   * @returns Array of available resources and optional next cursor
   */
  static async mListResources(
    pConnection: MCPConnection,
    pCursor?: string
  ): Promise<{ resources: MCPResource[]; nextCursor?: string }> {
    try {
      const cResult = await pConnection.client.listResources(
        pCursor ? { cursor: pCursor } : undefined
      );

      return {
        resources: (cResult.resources ?? []) as MCPResource[],
        nextCursor: cResult.nextCursor
      };
    } catch (vError) {
      if (this.mIsSessionExpiredError(vError)) {
        throw vError;
      }

      console.warn(`[MCPService][${pConnection.serverName}] Failed to list resources:`, vError);

      return { resources: [] };
    }
  }

  /**
   * List all resources from a connection (handles pagination automatically).
   * @param pConnection - The MCP connection to use
   * @returns Array of all available resources
   */
  static async mListAllResources(pConnection: MCPConnection): Promise<MCPResource[]> {
    const aAllResources: MCPResource[] = [];
    let vCursor: string | undefined;

    do {
      const cResult = await this.mListResources(pConnection, vCursor);
      aAllResources.push(...cResult.resources);
      vCursor = cResult.nextCursor;
    } while (vCursor);

    return aAllResources;
  }

  /**
   * List resource templates from a connection.
   * @param pConnection - The MCP connection to use
   * @param pCursor - Optional pagination cursor
   * @returns Array of available resource templates and optional next cursor
   */
  static async mListResourceTemplates(
    pConnection: MCPConnection,
    pCursor?: string
  ): Promise<{ resourceTemplates: MCPResourceTemplate[]; nextCursor?: string }> {
    try {
      const cResult = await pConnection.client.listResourceTemplates(
        pCursor ? { cursor: pCursor } : undefined
      );

      return {
        resourceTemplates: (cResult.resourceTemplates ?? []) as MCPResourceTemplate[],
        nextCursor: cResult.nextCursor
      };
    } catch (vError) {
      if (this.mIsSessionExpiredError(vError)) {
        throw vError;
      }

      console.warn(
        `[MCPService][${pConnection.serverName}] Failed to list resource templates:`,
        vError
      );

      return { resourceTemplates: [] };
    }
  }

  /**
   * List all resource templates from a connection (handles pagination automatically).
   * @param pConnection - The MCP connection to use
   * @returns Array of all available resource templates
   */
  static async mListAllResourceTemplates(
    pConnection: MCPConnection
  ): Promise<MCPResourceTemplate[]> {
    const aAllTemplates: MCPResourceTemplate[] = [];
    let vCursor: string | undefined;

    do {
      const cResult = await this.mListResourceTemplates(pConnection, vCursor);
      aAllTemplates.push(...cResult.resourceTemplates);
      vCursor = cResult.nextCursor;
    } while (vCursor);

    return aAllTemplates;
  }

  /**
   * Read the contents of a resource.
   * @param pConnection - The MCP connection to use
   * @param pUri - The URI of the resource to read
   * @returns The resource contents
   */
  static async mReadResource(
    pConnection: MCPConnection,
    pUri: string
  ): Promise<MCPReadResourceResult> {
    try {
      const cResult = await pConnection.client.readResource({ uri: pUri });

      return {
        contents: (cResult.contents ?? []) as MCPResourceContent[],
        _meta: cResult._meta
      };
    } catch (vError) {
      console.error(`[MCPService][${pConnection.serverName}] Failed to read resource:`, vError);

      throw vError;
    }
  }

  /**
   * Subscribe to updates for a resource.
   * The server will send notifications/resources/updated when the resource changes.
   * @param pConnection - The MCP connection to use
   * @param pUri - The URI of the resource to subscribe to
   */
  static async mSubscribeResource(pConnection: MCPConnection, pUri: string): Promise<void> {
    try {
      await pConnection.client.subscribeResource({ uri: pUri });

      console.log(`[MCPService][${pConnection.serverName}] Subscribed to resource: ${pUri}`);
    } catch (vError) {
      console.error(
        `[MCPService][${pConnection.serverName}] Failed to subscribe to resource:`,
        vError
      );

      throw vError;
    }
  }

  /**
   * Unsubscribe from updates for a resource.
   * @param pConnection - The MCP connection to use
   * @param pUri - The URI of the resource to unsubscribe from
   */
  static async mUnsubscribeResource(pConnection: MCPConnection, pUri: string): Promise<void> {
    try {
      await pConnection.client.unsubscribeResource({ uri: pUri });

      if (import.meta.env.DEV && import.meta.env.VITE_DEBUG) {
        console.log(`[MCPService][${pConnection.serverName}] Unsubscribed from resource: ${pUri}`);
      }
    } catch (vError) {
      console.error(
        `[MCPService][${pConnection.serverName}] Failed to unsubscribe from resource:`,
        vError
      );

      throw vError;
    }
  }

  /**
   * Check if a connection supports resources.
   * Per MCP spec: presence of the `resources` key (even as empty object `{}`) indicates support.
   * Empty object means resources are supported but no sub-features (subscribe, listChanged).
   *
   * @param pConnection - The MCP connection to check
   * @returns Whether the server declares the resources capability
   */
  static mSupportsResources(pConnection: MCPConnection): boolean {
    // Per MCP spec: "Servers that support resources MUST declare the resources capability"
    // The presence of the key indicates support, even if it's an empty object
    return pConnection.serverCapabilities?.resources !== undefined;
  }

  /**
   * Check if a connection supports resource subscriptions.
   * @param pConnection - The MCP connection to check
   * @returns Whether the server supports resource subscriptions
   */
  static mSupportsResourceSubscriptions(pConnection: MCPConnection): boolean {
    return !!pConnection.serverCapabilities?.resources?.subscribe;
  }
}
