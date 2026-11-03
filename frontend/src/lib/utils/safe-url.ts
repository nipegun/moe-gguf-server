/**
 * Helpers that keep untrusted text and URLs (model output, tool results,
 * imported conversations) from turning into active HTML or script navigation.
 */

const cHtmlEscapes: Record<string, string> = {
  '&': '&amp;',
  '<': '&lt;',
  '>': '&gt;',
  '"': '&quot;',
  "'": '&#39;'
};

const cHtmlEscapeRegex = /[&<>"']/g;

// Browsers drop ASCII control characters and whitespace while parsing a URL,
// so "java\tscript:" still runs as "javascript:"; strip them before reading the scheme
// eslint-disable-next-line no-control-regex
const cUrlIgnoredCharsRegex = /[\u0000- \u007f]/g;
const cUrlSchemeRegex = /^([a-z][a-z0-9+.-]*):/i;

// Schemes a rendered link may navigate to; anything else (javascript:, vbscript:,
// data:, blob:, file:, custom handlers) is dropped
const sSafeLinkSchemes = new Set(['http', 'https', 'mailto', 'tel']);

const sHttpSchemes = new Set(['http', 'https']);

// An inline image stored by the UI: data:image/<subtype>[;param...];base64,<payload>
const cDataImageUrlRegex = /^data:image\/[a-z0-9.+-]+(?:;[a-z0-9.+=-]+)*;base64,[a-z0-9+/=\s]*$/i;

/**
 * Escapes text so it can be inserted into HTML as inert content.
 */
export function fEscapeHtml(pText: string): string {
  return pText.replace(cHtmlEscapeRegex, (pChar) => cHtmlEscapes[pChar]);
}

/**
 * Returns the lowercase scheme of a URL, or null when it has none (relative URL).
 */
function fGetUrlScheme(pUrl: string): string | null {
  const cMatch = pUrl.replace(cUrlIgnoredCharsRegex, '').match(cUrlSchemeRegex);
  return cMatch ? cMatch[1].toLowerCase() : null;
}

/**
 * Whether a link target is safe to navigate to: a relative URL or an
 * http(s), mailto or tel URL.
 */
export function fIsSafeLinkUrl(pUrl: string): boolean {
  const cScheme = fGetUrlScheme(pUrl);
  return cScheme === null || sSafeLinkSchemes.has(cScheme);
}

/**
 * Whether a URL is an absolute http(s) URL.
 */
export function fIsHttpUrl(pUrl: string): boolean {
  const cScheme = fGetUrlScheme(pUrl);
  return cScheme !== null && sHttpSchemes.has(cScheme);
}

/**
 * Whether a value is a base64 image data URL, the only form the UI stores
 * for image attachments.
 */
export function fIsDataImageUrl(pUrl: string | null | undefined): boolean {
  // Imported values are only typed, not guaranteed, so the type is checked at runtime too
  return typeof pUrl === 'string' && cDataImageUrlRegex.test(pUrl);
}
