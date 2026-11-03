/**
 * Header utilities for parsing and serializing HTTP headers.
 * Generic utilities not specific to MCP.
 */

/**
 * Parses a JSON string of headers into an array of key-value pairs.
 * Returns empty array if the JSON is invalid or empty.
 */
export function fParseHeadersToArray(pHeadersJson: string): { key: string; value: string }[] {
  if (!pHeadersJson?.trim()) return [];

  try {
    const cParsed = JSON.parse(pHeadersJson);
    if (typeof cParsed === 'object' && cParsed !== null && !Array.isArray(cParsed)) {
      return Object.entries(cParsed).map(([pKey, pValue]) => ({
        key: pKey,
        value: String(pValue)
      }));
    }
  } catch {
    return [];
  }

  return [];
}

/**
 * Serializes an array of header key-value pairs to a JSON string.
 * Filters out pairs with empty keys and returns empty string if no valid pairs.
 */
export function fSerializeHeaders(pPairs: { key: string; value: string }[]): string {
  const cValidPairs = pPairs.filter((pP) => pP.key.trim());

  if (cValidPairs.length === 0) return '';

  const dObj: Record<string, string> = {};

  for (const cPair of cValidPairs) {
    dObj[cPair.key.trim()] = cPair.value;
  }

  return JSON.stringify(dObj);
}
