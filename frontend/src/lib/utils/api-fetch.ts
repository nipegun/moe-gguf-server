import { base } from '$app/paths';
import { fApiUrl } from '$lib/api-url';
import { fGetJsonHeaders, fGetAuthHeaders } from './api-headers';
import { UrlProtocol } from '$lib/enums';
import { dErrorMessages, dHttpCodeToString } from '$lib/constants/error';

/**
 * API Fetch Utilities
 *
 * Provides common fetch patterns used across services:
 * - Automatic JSON headers
 * - Error handling with proper error messages
 * - Base path resolution
 */

/**
 * Error thrown when an API request fails, carrying the HTTP status code
 * so callers can distinguish e.g. a 503 "still loading" response from a
 * genuine failure.
 */
export class ApiError extends Error {
  vStatus: number;

  constructor(pMessage: string, pStatus: number) {
    super(pMessage);
    this.name = 'ApiError';
    this.vStatus = pStatus;
  }
}

export interface ApiFetchOptions extends Omit<RequestInit, 'headers'> {
  /**
   * Use auth-only headers (no Content-Type).
   * Default: false (uses JSON headers with Content-Type: application/json)
   */
  authOnly?: boolean;
  /**
   * Additional headers to merge with default headers.
   */
  headers?: Record<string, string>;
}

/**
 * Fetch JSON data from an API endpoint with standard headers and error handling.
 *
 * @param pPath - API path (will be prefixed with base path)
 * @param pOptions - Fetch options with additional authOnly flag
 * @returns Parsed JSON response
 * @throws Error with formatted message on failure
 *
 * @example
 * ```typescript
 * // GET request
 * const models = await apiFetch<ApiModelListResponse>('/v1/models');
 *
 * // POST request
 * const result = await apiFetch<ApiResponse>('/models/load', {
 *   method: 'POST',
 *   body: JSON.stringify({ model: 'gpt-4' })
 * });
 * ```
 */
export async function fApiFetch<T>(pPath: string, pOptions: ApiFetchOptions = {}): Promise<T> {
  const { authOnly: cAuthOnly = false, headers: cCustomHeaders, ...cFetchOptions } = pOptions;

  const cBaseHeaders = cAuthOnly ? fGetAuthHeaders() : fGetJsonHeaders();
  const dHeaders = { ...cBaseHeaders, ...cCustomHeaders };

  const cUrl =
    pPath.startsWith(UrlProtocol.HTTP) || pPath.startsWith(UrlProtocol.HTTPS)
      ? pPath
      : fApiUrl(pPath);

  let vResponse;
  try {
    vResponse = await fetch(cUrl, {
      ...cFetchOptions,
      headers: dHeaders
    });
  } catch (vE) {
    throw new Error(fBeautifyNetworkError(vE));
  }

  if (!vResponse.ok) {
    const cErrorMessage = await fParseErrorMessage(vResponse);
    throw new ApiError(cErrorMessage, vResponse.status);
  }

  return vResponse.json() as Promise<T>;
}

/**
 * Fetch with URL constructed from base URL and query parameters.
 *
 * @param pBasePath - Base API path
 * @param pParams - Query parameters to append
 * @param pOptions - Fetch options
 * @returns Parsed JSON response
 *
 * @example
 * ```typescript
 * const props = await apiFetchWithParams<ApiProps>('./props', {
 *   model: 'gpt-4',
 *   autoload: 'false'
 * });
 * ```
 */
export async function fApiFetchWithParams<T>(
  pBasePath: string,
  pParams: Record<string, string>,
  pOptions: ApiFetchOptions = {}
): Promise<T> {
  const cUrl = new URL(fApiUrl(pBasePath), window.location.href);

  for (const [cKey, cValue] of Object.entries(pParams)) {
    if (cValue !== undefined && cValue !== null) {
      cUrl.searchParams.set(cKey, cValue);
    }
  }

  const { authOnly: cAuthOnly = false, headers: cCustomHeaders, ...cFetchOptions } = pOptions;

  const cBaseHeaders = cAuthOnly ? fGetAuthHeaders() : fGetJsonHeaders();
  const dHeaders = { ...cBaseHeaders, ...cCustomHeaders };

  let vResponse;
  try {
    vResponse = await fetch(cUrl.toString(), {
      ...cFetchOptions,
      headers: dHeaders
    });
  } catch (vE) {
    throw new Error(fBeautifyNetworkError(vE));
  }

  if (!vResponse.ok) {
    const cErrorMessage = await fParseErrorMessage(vResponse);
    throw new ApiError(cErrorMessage, vResponse.status);
  }

  return vResponse.json() as Promise<T>;
}

/**
 * POST JSON data to an API endpoint.
 *
 * @param pPath - API path
 * @param pBody - Request body (will be JSON stringified)
 * @param pOptions - Additional fetch options
 * @returns Parsed JSON response
 */
export async function fApiPost<T, B = unknown>(
  pPath: string,
  pBody: B,
  pOptions: ApiFetchOptions = {}
): Promise<T> {
  return fApiFetch<T>(pPath, {
    method: 'POST',
    body: JSON.stringify(pBody),
    ...pOptions
  });
}

/**
 * Parse error message from a failed response.
 * Tries to extract error message from JSON body, falls back to status text.
 */
async function fParseErrorMessage(pResponse: Response): Promise<string> {
  try {
    const cErrorData = await pResponse.json();
    if (cErrorData?.error?.message) {
      return cErrorData.error.message;
    }
    if (cErrorData?.error && typeof cErrorData.error === 'string') {
      return cErrorData.error;
    }
    if (cErrorData?.message) {
      return cErrorData.message;
    }
  } catch {
    // JSON parsing failed, use status text
  }

  const cHttpErrorStr = dHttpCodeToString[pResponse.status];
  if (cHttpErrorStr) {
    return cHttpErrorStr;
  }

  return `${dErrorMessages.HTTP.GENERIC}: ${pResponse.status} ${pResponse.statusText}`;
}

/**
 * Converts a network issue into a human-readable message.
 * @param pThrowable - The throwable raised during fetch operation
 * @returns Error in an human-readable format
 */
function fBeautifyNetworkError(pThrowable: unknown): string {
  let vMessage;
  if (pThrowable instanceof Error) {
    vMessage = pThrowable.message;
    if (pThrowable.name === 'TypeError' && vMessage.includes('fetch')) {
      return dErrorMessages.NETWORK.UNREACHABLE;
    }
  } else {
    vMessage = String(pThrowable);
  }

  if (vMessage.includes('ECONNREFUSED')) {
    return dErrorMessages.NETWORK.REFUSED;
  } else if (vMessage.includes('ENOTFOUND')) {
    return dErrorMessages.NETWORK.NXDOMAIN;
  } else if (vMessage.includes('ETIMEDOUT')) {
    return dErrorMessages.NETWORK.TIMEOUT;
  }

  return `${dErrorMessages.NETWORK.GENERIC} (${vMessage})`;
}
