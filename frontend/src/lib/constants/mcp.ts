import { Zap, Globe, Radio } from '@lucide/svelte';
import { MCPTransportType } from '$lib/enums';
import type { ClientCapabilities, Implementation } from '$lib/types';
import type { Component } from 'svelte';
import { MimeTypeImage } from '$lib/enums/files.enums';

export const cDefaultClientVersion = '1.0.0';
export const cMcpClientName = 'mgs-ui-mcp';
export const cDefaultImageMimeType = MimeTypeImage.PNG;

/** MIME types considered safe for rendering MCP server icons */
export const sMcpAllowedIconMimeTypes = new Set([
  MimeTypeImage.PNG,
  MimeTypeImage.JPEG,
  MimeTypeImage.JPG,
  MimeTypeImage.SVG,
  MimeTypeImage.WEBP,
  MimeTypeImage.ICO,
  MimeTypeImage.ICO_MICROSOFT
]);

/**
 * MCP specification version this client targets.
 * Update when the upstream MCP spec introduces a new stable version:
 * https://spec.modelcontextprotocol.io/
 */
export const cMcpProtocolVersion = '2025-06-18';

export const cDefaultMcpConfig = {
  protocolVersion: cMcpProtocolVersion,
  capabilities: { tools: { listChanged: true } } as ClientCapabilities,
  clientInfo: { name: cMcpClientName, version: cDefaultClientVersion } as Implementation,
  requestTimeoutSeconds: 300, // 5 minutes for long-running tools
  connectionTimeoutMs: 10_000 // 10 seconds for connection establishment
} as const;

export const cMcpServerIdPrefix = 'MgsUI-MCP-Server';

/** Settings entry that mirrors the server's local shell MCP server (--local-shell-mcp-server) */
export const cShellMcpServerId = 'mgs-shell';
export const cShellMcpServerName = 'local shell mcp server';
/** Tools the local shell MCP server can expose (only those of the shells it finds) */
export const aShellMcpToolNames: readonly string[] = ['bash.run', 'zsh.run', 'sh.run', 'psh.run'];
/** _meta key of tools/call that names the call, so the UI can answer the sudo password prompts of its command */
export const cShellMcpCallIdMeta = 'mgs/callId';
/** How often an executing shell call checks whether sudo is asking for a password */
export const cShellMcpAskpassPollMs = 500;

export const cMcpReconnectInitialDelay = 1000;
export const cMcpReconnectBackoffMultiplier = 2;
export const cMcpReconnectMaxDelay = 30000;
/** Per-attempt timeout for a single reconnection attempt before giving up and backing off. */
export const cMcpReconnectAttemptTimeoutMs = 15_000;

/** Maximum number of MCP server avatars to display in the chat form */
export const cMaxDisplayedMcpAvatars = 4;

/** Expected count when two theme-less icons represent a light/dark pair */
export const cExpectedThemedIconPairCount = 2;

/** CORS proxy URL query parameter name */
export const cCorsProxyUrlParam = 'url';

/** Header prefix for headers that should be forwarded by the CORS proxy */
export const cCorsProxyHeaderPrefix = 'x-mgs-proxy-header-';

/** Number of trailing characters to keep visible when partially redacting mcp-session-id */
export const cMcpSessionIdVisibleChars = 5;

/** Partial-redaction rules for MCP headers: header name -> visible trailing chars */
export const dMcpPartialRedactHeaders = new Map<string, number>([
  ['mcp-session-id', cMcpSessionIdVisibleChars]
]);

/** Bearer scheme prefix used for Authorization headers (RFC 6750) */
export const cBearerPrefix = 'Bearer ';

/** Canonical casing for the Authorization header (RFC 7235) */
export const cAuthorizationHeader = 'Authorization';

/** Content-Type HTTP header name */
export const cContentTypeHeader = 'Content-Type';

/** Header names whose values should be redacted in diagnostic logs */
export const sRedactedHeaders = new Set([
  'authorization',
  'api-key',
  'cookie',
  'mcp-session-id',
  'proxy-authorization',
  'set-cookie',
  'x-auth-token',
  'x-api-key'
]);

/** Human-readable labels for MCP transport types */
export const dMcpTransportLabels: Record<MCPTransportType, string> = {
  [MCPTransportType.WEBSOCKET]: 'WebSocket',
  [MCPTransportType.STREAMABLE_HTTP]: 'HTTP',
  [MCPTransportType.SSE]: 'SSE'
};

/** Icon components for MCP transport types */
export const dMcpTransportIcons: Record<MCPTransportType, Component> = {
  [MCPTransportType.WEBSOCKET]: Zap,
  [MCPTransportType.STREAMABLE_HTTP]: Globe,
  [MCPTransportType.SSE]: Radio
};

/** Standard SSE endpoint path indicators */
export const cMcpSseEndpoint = '/sse';
export const cMcpSseEndpointSlash = '/sse/';
export const cMcpSseEndpointQuery = '/sse?';
