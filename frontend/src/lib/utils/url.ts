import {
  cTrailingSlashesRegex,
  cTwoPartPublicSuffixes,
  cWildcardPublicSuffixes
} from '$lib/constants';
import { UrlProtocol } from '$lib/enums';

/**
 * Check whether a hostname looks like an IPv4 or IPv6 address.
 */
function fIsIpAddress(pHostname: string): boolean {
  if (pHostname.includes(':')) return true;

  if (/^\d{1,3}(\.\d{1,3}){3}$/.test(pHostname)) return true;

  return false;
}

/**
 * Extract the registrable root domain from a URL.
 *
 * @example
 *   'mcp.example.com'           -> 'example.com'
 *   'www.example.co.uk'         -> 'example.co.uk'
 *   'bar.foo.nom.br'            -> 'bar.foo.nom.br'
 *   '192.168.1.1'               -> null
 *   'localhost'                 -> null
 */
export function fExtractRootDomain(pUrl: URL): string | null {
  const cHostname = pUrl.hostname.toLowerCase();
  if (!cHostname || fIsIpAddress(cHostname)) return null;

  const cParts = cHostname.split('.');

  if (cParts.length < 2) return null;

  if (cParts.length >= 3) {
    const cSuffix2 = `${cParts[cParts.length - 2]}.${cParts[cParts.length - 1]}`;

    if (cTwoPartPublicSuffixes.has(cSuffix2)) {
      return cParts.slice(-3).join('.');
    }
  }

  for (let vI = 2; vI <= cParts.length; vI++) {
    const cCandidate = cParts.slice(-vI).join('.');

    if (cWildcardPublicSuffixes.has(cCandidate)) {
      if (cParts.length === vI + 1) {
        return cHostname;
      }

      return cParts.slice(-(vI + 2)).join('.');
    }
  }

  return cParts.slice(-2).join('.');
}

/**
 * Sanitize an external URL string for safe use in an `<a href>`.
 * Only allows http: and https: schemes. Returns `null` for anything else.
 */
export function fSanitizeExternalUrl(pRaw: string): string | null {
  try {
    const cUrl = new URL(pRaw);

    if (cUrl.protocol !== UrlProtocol.HTTP && cUrl.protocol !== UrlProtocol.HTTPS) {
      return null;
    }

    return cUrl.href;
  } catch {
    return null;
  }
}

/**
 * Canonicalize a server URL for "is this the same server?" checks across
 * the user's settings and the recommended-server list. Lowercases scheme
 * and host, drops the port entirely, and strips any trailing slashes off
 * the path so a stored `https://api.example.com:8443/mcp/` matches the
 * recommended `https://api.example.com/mcp`. Falls back to a cheap
 * trim+lowercase+strip pass when the input isn't a parseable URL.
 *
 * Query strings are preserved deliberately - if the user entered one,
 * it's part of their endpoint. The port is always stripped because the
 * underlying `URL` parser is asymmetric (it auto-drops HTTPS default
 * :443 but keeps HTTP default :80), so a half-hearted "drop default
 * ports" policy never matches consistently across schemes.
 */
export function fCanonicalizeServerUrl(pRaw: string): string {
  const cTrimmed = pRaw.trim();

  try {
    const cParsed = new URL(cTrimmed);
    const cPathname = cParsed.pathname.replace(cTrailingSlashesRegex, '');

    // Aggressive: drop the port unconditionally. We only use this for
    // equality checks between user-typed URLs and a hard-coded list of
    // recommendations, where the port can never carry distinguishing
    // information we care about (a different port = a different server,
    // but two URLs that differ only in `:80` vs no-port are clearly the
    // same intent). Lowercasing the hostname matches HTTP/HTTPS
    // case-insensitivity - the URL parser does NOT lowercase it.
    const cHost = cParsed.hostname.toLowerCase();

    return `${cParsed.protocol}//${cHost}${cPathname}${cParsed.search}`;
  } catch {
    return cTrimmed.toLowerCase().replace(cTrailingSlashesRegex, '');
  }
}
