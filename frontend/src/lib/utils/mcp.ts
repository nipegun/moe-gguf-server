import type { MCPServerSettingsEntry, MCPResourceContent, MCPResourceInfo } from '$lib/types';
import {
  MCPTransportType,
  MCPLogLevel,
  UrlProtocol,
  MimeTypePrefix,
  MimeTypeIncludes,
  UriPattern,
  MimeTypeText
} from '$lib/enums';
import {
  cMcpServerIdPrefix,
  cImageFileExtensionRegex,
  cCodeFileExtensionRegex,
  cTextFileExtensionRegex,
  cProtocolPrefixRegex,
  cFileExtensionRegex,
  cDisplayNameSeparatorRegex,
  cPathSeparator,
  cResourceTextContentSeparator,
  cDefaultResourceFilename,
  cMcpSseEndpoint,
  cMcpSseEndpointSlash,
  cMcpSseEndpointQuery
} from '$lib/constants';
import {
  Database,
  File,
  FileText,
  Image,
  Code,
  Info,
  AlertTriangle,
  XCircle
} from '@lucide/svelte';
import type { Component } from 'svelte';
import type { MimeTypeUnion } from '$lib/types/common';

/**
 * Detects the MCP transport type from a URL.
 * WebSocket URLs (ws:// or wss://) use 'websocket', others use 'streamable_http'.
 */
export function fDetectMcpTransportFromUrl(pUrl: string): MCPTransportType {
  const cNormalized = pUrl.trim().toLowerCase();

  if (
    cNormalized.startsWith(UrlProtocol.WEBSOCKET) ||
    cNormalized.startsWith(UrlProtocol.WEBSOCKET_SECURE)
  ) {
    return MCPTransportType.WEBSOCKET;
  }

  if (
    cNormalized.endsWith(cMcpSseEndpoint) ||
    cNormalized.endsWith(cMcpSseEndpointSlash) ||
    cNormalized.includes(cMcpSseEndpointQuery)
  ) {
    return MCPTransportType.SSE;
  }

  return MCPTransportType.STREAMABLE_HTTP;
}

/**
 * Parses MCP server settings from a JSON string or array.
 * @param pRawServers - The raw servers to parse
 * @returns An empty array if the input is invalid.
 */
export function fParseMcpServerSettings(pRawServers: unknown): MCPServerSettingsEntry[] {
  if (!pRawServers) return [];

  let vParsed: unknown;

  if (typeof pRawServers === 'string') {
    const cTrimmed = pRawServers.trim();
    if (!cTrimmed) return [];

    try {
      vParsed = JSON.parse(cTrimmed);
    } catch (vError) {
      console.warn('[MCP] Failed to parse mcpServers JSON, ignoring value:', vError);

      return [];
    }
  } else {
    vParsed = pRawServers;
  }

  if (!Array.isArray(vParsed)) return [];

  return vParsed.map((pEntry, pIndex) => {
    const cUrl = typeof pEntry?.url === 'string' ? pEntry.url.trim() : '';
    const cHeaders = typeof pEntry?.headers === 'string' ? pEntry.headers.trim() : undefined;
    const cId =
      typeof (pEntry as { id?: unknown })?.id === 'string' && (pEntry as { id?: string }).id?.trim()
        ? (pEntry as { id: string }).id.trim()
        : `${cMcpServerIdPrefix}-${pIndex + 1}`;

    return {
      id: cId,
      enabled: Boolean((pEntry as { enabled?: unknown })?.enabled),
      url: cUrl,
      name: (pEntry as { name?: string })?.name,
      displayName: (pEntry as { displayName?: string })?.displayName,
      headers: cHeaders || undefined,
      useProxy: Boolean((pEntry as { useProxy?: unknown })?.useProxy)
    } satisfies MCPServerSettingsEntry;
  });
}

/**
 * Get the appropriate icon component for a log level
 *
 * @param pLevel - MCP log level
 * @returns Lucide icon component
 */
export function fGetMcpLogLevelIcon(pLevel: MCPLogLevel): Component {
  switch (pLevel) {
    case MCPLogLevel.ERROR:
      return XCircle;
    case MCPLogLevel.WARN:
      return AlertTriangle;
    default:
      return Info;
  }
}

/**
 * Get the appropriate CSS class for a log level
 *
 * @param pLevel - MCP log level
 * @returns Tailwind CSS class string
 */
export function fGetMcpLogLevelClass(pLevel: MCPLogLevel): string {
  switch (pLevel) {
    case MCPLogLevel.ERROR:
      return 'text-destructive';
    case MCPLogLevel.WARN:
      return 'text-yellow-600 dark:text-yellow-500';
    default:
      return 'text-muted-foreground';
  }
}

/**
 * Check if a MIME type represents an image.
 *
 * @param pMimeType - The MIME type to check
 * @returns True if the MIME type starts with 'image/'
 */
export function fIsImageMimeType(pMimeType?: MimeTypeUnion): boolean {
  return pMimeType?.startsWith(MimeTypePrefix.IMAGE) ?? false;
}

/**
 * Parse a resource URI into path segments, stripping the protocol prefix.
 *
 * @param pUri - The resource URI to parse
 * @returns Array of non-empty path segments
 */
export function fParseResourcePath(pUri: string): string[] {
  try {
    const cWithoutProtocol = pUri.replace(cProtocolPrefixRegex, '');
    return cWithoutProtocol.split(cPathSeparator).filter((pP) => pP.length > 0);
  } catch {
    return [pUri];
  }
}

/**
 * Convert a path part into a human-readable display name.
 * Strips file extensions and converts kebab-case/snake_case to Title Case.
 *
 * @param pPathPart - The path segment to convert
 * @returns Human-readable display name
 */
export function fGetDisplayName(pPathPart: string): string {
  const cWithoutExt = pPathPart.replace(cFileExtensionRegex, '');
  return cWithoutExt
    .split(cDisplayNameSeparatorRegex)
    .map((pWord) => pWord.charAt(0).toUpperCase() + pWord.slice(1))
    .join(' ');
}

/**
 * Get the display name from a resource, extracting the last path segment from the URI.
 *
 * @param pResource - The MCP resource info
 * @returns Display name string
 */
export function fGetResourceDisplayName(pResource: MCPResourceInfo): string {
  try {
    const cParts = fParseResourcePath(pResource.uri);
    return cParts[cParts.length - 1] || pResource.name || pResource.uri;
  } catch {
    return pResource.name || pResource.uri;
  }
}

/**
 * Determine if a MIME type and/or URI represents code content.
 *
 * @param pMimeType - Optional MIME type string
 * @param pUri - Optional URI string
 * @returns True if the content is code
 */
export function fIsCodeResource(pMimeType?: MimeTypeUnion, pUri?: string): boolean {
  const cMime = pMimeType?.toLowerCase() || '';
  const cU = pUri?.toLowerCase() || '';
  return (
    cMime.includes(MimeTypeIncludes.JSON) ||
    cMime.includes(MimeTypeIncludes.JAVASCRIPT) ||
    cMime.includes(MimeTypeIncludes.TYPESCRIPT) ||
    cCodeFileExtensionRegex.test(cU)
  );
}

/**
 * Determine if a MIME type and/or URI represents image content.
 *
 * @param pMimeType - Optional MIME type string
 * @param pUri - Optional URI string
 * @returns True if the content is an image
 */
export function fIsImageResource(pMimeType?: MimeTypeUnion, pUri?: string): boolean {
  const cMime = pMimeType?.toLowerCase() || '';
  const cU = pUri?.toLowerCase() || '';
  return cMime.startsWith(MimeTypePrefix.IMAGE) || cImageFileExtensionRegex.test(cU);
}

/**
 * Get the appropriate Lucide icon component for an MCP resource based on its MIME type and URI.
 *
 * @param pMimeType - Optional MIME type of the resource
 * @param pUri - Optional URI of the resource
 * @returns Lucide icon component
 */
export function fGetResourceIcon(pMimeType?: MimeTypeUnion, pUri?: string): Component {
  const cMime = pMimeType?.toLowerCase() || '';
  const cU = pUri?.toLowerCase() || '';

  if (cMime.startsWith(MimeTypePrefix.IMAGE) || cImageFileExtensionRegex.test(cU)) {
    return Image;
  }

  if (
    cMime.includes(MimeTypeIncludes.JSON) ||
    cMime.includes(MimeTypeIncludes.JAVASCRIPT) ||
    cMime.includes(MimeTypeIncludes.TYPESCRIPT) ||
    cCodeFileExtensionRegex.test(cU)
  ) {
    return Code;
  }

  if (cMime.includes(MimeTypePrefix.TEXT) || cTextFileExtensionRegex.test(cU)) {
    return FileText;
  }

  if (cU.includes(UriPattern.DATABASE_KEYWORD) || cU.includes(UriPattern.DATABASE_SCHEME)) {
    return Database;
  }

  return File;
}

/**
 * Extract text content from MCP resource content array.
 *
 * @param pContent - Array of MCP resource content items
 * @returns Joined text content string
 */
export function fGetResourceTextContent(pContent: MCPResourceContent[] | null | undefined): string {
  if (!pContent) return '';
  return pContent
    .filter((pC): pC is { uri: string; mimeType?: MimeTypeUnion; text: string } => 'text' in pC)
    .map((pC) => pC.text)
    .join(cResourceTextContentSeparator);
}

/**
 * Extract blob content from MCP resource content array.
 *
 * @param pContent - Array of MCP resource content items
 * @returns Array of blob content items
 */
export function fGetResourceBlobContent(
  pContent: MCPResourceContent[] | null | undefined
): Array<{ uri: string; mimeType?: MimeTypeUnion; blob: string }> {
  if (!pContent) return [];

  return pContent.filter(
    (pC): pC is { uri: string; mimeType?: MimeTypeUnion; blob: string } => 'blob' in pC
  );
}

/**
 * Trigger a file download from text content.
 *
 * @param pText - The text content to download
 * @param pMimeType - MIME type for the blob
 * @param pFilename - Suggested filename
 */
export function fDownloadResourceContent(
  pText: string,
  pMimeType: MimeTypeUnion = MimeTypeText.PLAIN,
  pFilename: string = cDefaultResourceFilename
): void {
  const cBlob = new Blob([pText], { type: pMimeType });
  const cUrl = URL.createObjectURL(cBlob);
  const cA = document.createElement('a');
  cA.href = cUrl;
  cA.download = pFilename;
  document.body.appendChild(cA);
  cA.click();
  document.body.removeChild(cA);
  URL.revokeObjectURL(cUrl);
}
