/**
 * HTTP request inspection utilities for diagnostic logging.
 * These helpers extract metadata from fetch-style request arguments
 * without exposing sensitive payload data.
 */

export interface RequestBodySummary {
  kind: string;
  size?: number;
}

export function fGetRequestUrl(pInput: RequestInfo | URL): string {
  if (typeof pInput === 'string') {
    return pInput;
  }

  if (pInput instanceof URL) {
    return pInput.href;
  }

  return pInput.url;
}

export function fGetRequestMethod(
  pInput: RequestInfo | URL,
  pInit?: RequestInit,
  pBaseInit?: RequestInit
): string {
  if (pInit?.method) {
    return pInit.method;
  }

  if (typeof Request !== 'undefined' && pInput instanceof Request) {
    return pInput.method;
  }

  return pBaseInit?.method ?? 'GET';
}

export function fGetRequestBody(
  pInput: RequestInfo | URL,
  pInit?: RequestInit
): BodyInit | null | undefined {
  if (pInit?.body !== undefined) {
    return pInit.body;
  }

  if (typeof Request !== 'undefined' && pInput instanceof Request) {
    return pInput.body;
  }

  return undefined;
}

export function fSummarizeRequestBody(pBody: BodyInit | null | undefined): RequestBodySummary {
  if (pBody == null) {
    return { kind: 'empty' };
  }

  if (typeof pBody === 'string') {
    return { kind: 'string', size: pBody.length };
  }

  if (pBody instanceof Blob) {
    return { kind: 'blob', size: pBody.size };
  }

  if (pBody instanceof URLSearchParams) {
    return { kind: 'urlsearchparams', size: pBody.toString().length };
  }

  if (pBody instanceof FormData) {
    return { kind: 'formdata' };
  }

  if (pBody instanceof ArrayBuffer) {
    return { kind: 'arraybuffer', size: pBody.byteLength };
  }

  if (ArrayBuffer.isView(pBody)) {
    return { kind: pBody.constructor.name, size: pBody.byteLength };
  }

  return { kind: typeof pBody };
}

export function fFormatDiagnosticErrorMessage(pError: unknown): string {
  const cMessage = pError instanceof Error ? pError.message : String(pError);

  return cMessage.includes('Failed to fetch') ? `${cMessage} (check CORS?)` : cMessage;
}

export function fExtractJsonRpcMethods(pBody: BodyInit | null | undefined): string[] | undefined {
  if (typeof pBody !== 'string') {
    return undefined;
  }

  try {
    const cParsed = JSON.parse(pBody);
    const cMessages = Array.isArray(cParsed) ? cParsed : [cParsed];
    const cMethods = cMessages
      .map((pMessage: Record<string, unknown>) =>
        typeof pMessage?.method === 'string' ? (pMessage.method as string) : undefined
      )
      .filter((pMethod: string | undefined): pMethod is string => Boolean(pMethod));

    return cMethods.length > 0 ? cMethods : undefined;
  } catch {
    return undefined;
  }
}
