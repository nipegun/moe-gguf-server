import { fConfig } from '$lib/stores/settings.svelte';
import {
  cAuthorizationHeader,
  cBearerPrefix,
  cContentTypeHeader,
  cCorsProxyHeaderPrefix,
  sRedactedHeaders
} from '$lib/constants';
import { MimeTypeApplication } from '$lib/enums';
import { fRedactValue } from './redact';

/**
 * Get authorization headers for API requests
 * Includes Bearer token if API key is configured
 */
export function fGetAuthHeaders(): Record<string, string> {
  const cCurrentConfig = fConfig();
  const cApiKey = cCurrentConfig.apiKey?.toString().trim();

  return cApiKey ? { [cAuthorizationHeader]: `${cBearerPrefix}${cApiKey}` } : {};
}

/**
 * Get standard JSON headers with optional authorization
 */
export function fGetJsonHeaders(): Record<string, string> {
  return {
    [cContentTypeHeader]: MimeTypeApplication.JSON,
    ...fGetAuthHeaders()
  };
}

/**
 * Sanitize HTTP headers by redacting sensitive values.
 * Known sensitive headers (from REDACTED_HEADERS) and any extra headers
 * specified by the caller are fully redacted. Headers listed in
 * `partialRedactHeaders` are partially redacted, showing only the
 * specified number of trailing characters.
 *
 * @param pHeaders - Headers to sanitize
 * @param pExtraRedactedHeaders - Additional header names to fully redact
 * @param pPartialRedactHeaders - Map of header name -> number of trailing chars to keep visible
 * @returns Object with header names as keys and (possibly redacted) values
 */
export function fSanitizeHeaders(
  pHeaders?: HeadersInit,
  pExtraRedactedHeaders?: Iterable<string>,
  pPartialRedactHeaders?: Map<string, number>
): Record<string, string> {
  if (!pHeaders) {
    return {};
  }

  const cNormalized = new Headers(pHeaders);
  const dSanitized: Record<string, string> = {};
  const sExtraRedactedHeaders = new Set(
    Array.from(pExtraRedactedHeaders ?? [], (pHeader) => pHeader.toLowerCase())
  );

  for (const [cKey, cValue] of cNormalized.entries()) {
    const cNormalizedKey = cKey.toLowerCase();
    const cUnproxiedKey = cNormalizedKey.startsWith(cCorsProxyHeaderPrefix)
      ? cNormalizedKey.slice(cCorsProxyHeaderPrefix.length)
      : cNormalizedKey;
    const cPartialChars =
      pPartialRedactHeaders?.get(cNormalizedKey) ?? pPartialRedactHeaders?.get(cUnproxiedKey);

    if (cPartialChars !== undefined) {
      dSanitized[cKey] = fRedactValue(cValue, cPartialChars);
    } else if (
      sRedactedHeaders.has(cNormalizedKey) ||
      sRedactedHeaders.has(cUnproxiedKey) ||
      sExtraRedactedHeaders.has(cNormalizedKey) ||
      sExtraRedactedHeaders.has(cUnproxiedKey)
    ) {
      dSanitized[cKey] = fRedactValue(cValue);
    } else {
      dSanitized[cKey] = cValue;
    }
  }

  return dSanitized;
}
