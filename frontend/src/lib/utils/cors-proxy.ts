/**
 * CORS Proxy utility for routing requests through the mgs CORS proxy.
 */

import { base } from '$app/paths';
import { cCorsProxyEndpoint, cCorsProxyHeaderPrefix, cCorsProxyUrlParam } from '$lib/constants';

/**
 * Build a proxied URL that routes through the mgs CORS proxy.
 * @param pTargetUrl - The original URL to proxy
 * @returns URL pointing to the CORS proxy with target encoded
 */
export function fBuildProxiedUrl(pTargetUrl: string): URL {
  const cProxyPath = `${base}${cCorsProxyEndpoint}`;
  const cProxyUrl = new URL(cProxyPath, window.location.origin);

  cProxyUrl.searchParams.set(cCorsProxyUrlParam, pTargetUrl);

  return cProxyUrl;
}

/**
 * Wrap original headers for proxying through the CORS proxy. This avoids issues with duplicated mgs-specific and target headers when using the CORS proxy.
 * @param pHeaders - The original headers to be proxied to target
 * @returns List of "wrapped" headers to be sent to the CORS proxy
 */
export function fBuildProxiedHeaders(pHeaders: Record<string, string>): Record<string, string> {
  const dProxiedHeaders: Record<string, string> = {};

  for (const [cKey, cValue] of Object.entries(pHeaders)) {
    dProxiedHeaders[`${cCorsProxyHeaderPrefix}${cKey}`] = cValue;
  }

  return dProxiedHeaders;
}
