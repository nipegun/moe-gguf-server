/**
 * Parsers for MCP web-search tool responses shaped like:
 *
 *     Title: <text>
 *     URL: <https url>
 *     Published: <iso date or N/A>
 *     Author: <name or N/A>
 *     Highlights:
 *     <multi-line excerpt>
 *     ---
 *     Title: <next result>
 *     ...
 *
 * The model is content-driven (any tool emitting `Title:` / `URL:` lines
 * separated by `---` qualifies), so it adapts to other web-search MCP
 * servers without hardcoding tool names.
 */

export type SearchResult = {
  title: string;
  url: string;
  published?: string;
  author?: string;
  highlights?: string;
};

const cSeparatorLineRe = /^\s*---\s*$/;
const cUrlSchemeRe = /^https?:\/\//i;

// Match either Unix or Windows line endings so chunking/parsing handles
// payloads written by either scheme without off-by-one mismatches.
const cLineBreakRe = /\r?\n/;

// Sentinel the search-result wire format uses when a field is absent
// (e.g. `Author: N/A`). Treated identically to a missing field so the
// rendered card hides the row either way.
const cNotAvailableValue = 'N/A';

// Section header that announces the start of the multi-line Highlights
// block. Everything from that line onward (until the next `---`
// separator or end of chunk) is captured verbatim as highlight text
// instead of being re-scanned for `Title:`/`URL:`/... field lines.
const cHighlightsSectionHeader = 'Highlights:';

// Field name conventionally used by web-search tools (Exa etc.) as the
// user-supplied query parameter. Extracted so future tool schemas that
// adopt the same convention stay grep-compatible with this parser.
const cSearchToolQueryField = 'query';

// URL schemes the favicon helper will resolve to a hosted favicon. Any
// other scheme (e.g. data:, blob:) intentionally returns null so the UI
// can fall back to a generic globe icon.
const aResolvableUrlProtocols: readonly string[] = ['https:', 'http:'];

// Conventional favicon path served by virtually every web host.
// Appended to the URL origin as a best-effort lookup target; ignore
// 404s at render time.
const cFaviconPath = '/favicon.ico';

// Wire-format field names emitted by the search-result parser. String
// values match the keys the chunk parser writes into the `fields` map
// (and that callers read off `SearchResult`), so `FieldKey.TITLE` is a
// drop-in for the literal `'title'`.
enum FieldKey {
  TITLE = 'title',
  URL = 'url',
  PUBLISHED = 'published',
  AUTHOR = 'author'
}
const aFieldPrefixes: ReadonlyArray<{ key: FieldKey; prefix: string }> = [
  { key: FieldKey.TITLE, prefix: 'Title:' },
  { key: FieldKey.URL, prefix: 'URL:' },
  { key: FieldKey.PUBLISHED, prefix: 'Published:' },
  { key: FieldKey.AUTHOR, prefix: 'Author:' }
];

/**
 * Split a tool result string into individual search-result chunks by
 * scanning line-by-line for `---` separator rows. Handles multi-line
 * safely (line-aware, not regex on the full string) so trailing /
 * leading / consecutive separators are not lost.
 */
function fSplitChunks(pText: string): string[] {
  const cLines = pText.split(cLineBreakRe);
  const aChunks: string[] = [];
  let aBuffer: string[] = [];
  for (const cLine of cLines) {
    if (cSeparatorLineRe.test(cLine)) {
      if (aBuffer.length > 0) {
        aChunks.push(aBuffer.join('\n'));
        aBuffer = [];
      }
    } else {
      aBuffer.push(cLine);
    }
  }
  if (aBuffer.length > 0) aChunks.push(aBuffer.join('\n'));
  return aChunks;
}

/**
 * Parse a single chunk into a SearchResult. Returns null when the chunk
 * has neither a title nor a URL — those are required for an entry to be
 * actionable (otherwise it is almost certainly malformed or a stray
 * separator line).
 */
function fParseChunk(pChunk: string): SearchResult | null {
  const cTrimmed = pChunk.trim();
  if (!cTrimmed) return null;

  const cLines = pChunk.split(cLineBreakRe);

  const dFields: Record<FieldKey, string | undefined> = {
    [FieldKey.TITLE]: undefined,
    [FieldKey.URL]: undefined,
    [FieldKey.PUBLISHED]: undefined,
    [FieldKey.AUTHOR]: undefined
  };
  const aHighlightLines: string[] = [];
  let vInHighlights = false;

  for (const cLine of cLines) {
    if (!vInHighlights && cLine.trim() === cHighlightsSectionHeader) {
      vInHighlights = true;
      continue;
    }

    if (vInHighlights) {
      aHighlightLines.push(cLine);
      continue;
    }

    for (const { key: cKey, prefix: cPrefix } of aFieldPrefixes) {
      if (!cLine.startsWith(cPrefix)) continue;
      const cValue = cLine.slice(cPrefix.length).trim();
      if (cValue && cValue !== cNotAvailableValue) {
        dFields[cKey] = cValue;
      }
      break;
    }
  }

  if (
    !dFields[FieldKey.TITLE] ||
    !dFields[FieldKey.URL] ||
    !cUrlSchemeRe.test(dFields[FieldKey.URL])
  )
    return null;

  const cHighlights = aHighlightLines.join('\n').trim();

  const dResult: SearchResult = {
    title: dFields[FieldKey.TITLE],
    url: dFields[FieldKey.URL]
  };
  if (dFields[FieldKey.PUBLISHED]) dResult.published = dFields[FieldKey.PUBLISHED];
  if (dFields[FieldKey.AUTHOR]) dResult.author = dFields[FieldKey.AUTHOR];
  if (cHighlights) dResult.highlights = cHighlights;
  return dResult;
}

/** Bounded cache for extractSearchResults results. */
const cSearchResultsCacheMaxSize = 32;
const dSearchResultsCache = new Map<string, SearchResult[]>();

/**
 * Extract a SearchResult[] from a tool-result string. Returns `[]` when
 * the input does not match the expected shape — useful for branching
 * between dedicated search-results rendering and the generic tool-call
 * block. Memoized: called per render during streaming on unchanged
 * tool result strings.
 */
export function fExtractSearchResults(pText: string | undefined | null): SearchResult[] {
  if (!pText) return [];

  const cCached = dSearchResultsCache.get(pText);
  if (cCached) return cCached;

  const aResults: SearchResult[] = [];
  for (const cChunk of fSplitChunks(pText)) {
    const cParsed = fParseChunk(cChunk);
    if (cParsed) aResults.push(cParsed);
  }

  if (dSearchResultsCache.size >= cSearchResultsCacheMaxSize) {
    dSearchResultsCache.delete(dSearchResultsCache.keys().next().value!);
  }
  dSearchResultsCache.set(pText, aResults);

  return aResults;
}

/** Bounded cache for extractSearchQuery results. */
const cSearchQueryCacheMaxSize = 32;
const dSearchQueryCache = new Map<string, string>();

/**
 * Best-effort extraction of the search query out of a tool call's JSON
 * argument blob. Currently looks for a `query` field (the convention
 * used by Exa and most web-search MCP servers); returns an empty string
 * if it cannot be located. Memoized: called per render during streaming
 * on unchanged tool args strings.
 */
export function fExtractSearchQuery(pToolArgs: string | undefined | null): string {
  if (!pToolArgs) return '';

  const cCached = dSearchQueryCache.get(pToolArgs);
  if (cCached !== undefined) return cCached;

  let vResult = '';
  try {
    const cParsed: unknown = JSON.parse(pToolArgs);
    if (cParsed && typeof cParsed === 'object' && !Array.isArray(cParsed)) {
      const cCandidate = (cParsed as Record<string, unknown>)[cSearchToolQueryField];
      if (typeof cCandidate === 'string') vResult = cCandidate.trim();
    }
  } catch {
    vResult = '';
  }

  if (dSearchQueryCache.size >= cSearchQueryCacheMaxSize) {
    dSearchQueryCache.delete(dSearchQueryCache.keys().next().value!);
  }
  dSearchQueryCache.set(pToolArgs, vResult);

  return vResult;
}

/**
 * Resolve a best-effort favicon URL for a search result, derived from the
 * result's origin (`https://host/favicon.ico`). Returns `null` when the
 * URL is malformed, has no recognizable host, or uses a non-http(s)
 * scheme — callers should fall back to a generic globe icon.
 */
export function fFaviconForUrl(pUrl: string): string | null {
  try {
    const cParsed = new URL(pUrl);
    if (!aResolvableUrlProtocols.includes(cParsed.protocol)) return null;
    return `${cParsed.protocol}//${cParsed.host}${cFaviconPath}`;
  } catch {
    return null;
  }
}

// Web-search MCP servers broadly follow the `web_search` token convention
// for their primary tool, but the rich pill UI makes assumptions about
// both the request shape (single `query` string) and the response shape
// (Title:/URL:/Published:/Author:/Highlights blocks). Adding a tool here
// is a deliberate signal that the renderer is known to handle its output.
// Continued maintenance note: when broadening this list, verify both the
// tool schema and the response format against the supported spec above.
export const aSupportedWebSearchToolNames: readonly string[] = ['web_search_exa'];

/**
 * True when the tool's name is in the explicit allow-list of web-search
 * tools above. Returned to the dispatcher so it can route the call's UI
 * early (before results arrive) without false-firing on non-web-search
 * tools that also happen to accept a `query` argument.
 */
export function fIsWebSearchToolName(pToolName: string | undefined | null): boolean {
  if (!pToolName) return false;
  return aSupportedWebSearchToolNames.includes(pToolName);
}
