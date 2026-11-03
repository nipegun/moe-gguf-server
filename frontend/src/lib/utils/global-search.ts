/**
 * Matching and navigation helpers of the global search of the sidebar, which finds chats, the
 * content of their messages, pages, settings, models, tools and MCP servers.
 */

/** Lower case without diacritics, so "configuracion" finds "configuración". */
export function fNormalizeSearchText(pText: string): string {
  return pText
    .normalize('NFD')
    .replace(/[̀-ͯ]/g, '')
    .toLowerCase();
}

/** The words of a query, already normalized. */
export function fSearchTokens(pQuery: string): string[] {
  return fNormalizeSearchText(pQuery).split(/\s+/).filter(Boolean);
}

/**
 * Score of a text for the words of a query: 0 when some word is missing, 3 when the title holds the
 * whole query, 2 when it holds every word, 1 when only the rest of the text (help, descriptions) does.
 */
export function fSearchScore(pTokens: string[], pTitle: string, pDetails = ''): number {
  if (pTokens.length === 0) return 0;

  const cTitle = fNormalizeSearchText(pTitle);
  const cAll = `${cTitle} ${fNormalizeSearchText(pDetails)}`;
  if (!pTokens.every((pToken) => cAll.includes(pToken))) return 0;
  if (cTitle.includes(pTokens.join(' '))) return 3;
  if (pTokens.every((pToken) => cTitle.includes(pToken))) return 2;

  return 1;
}

/** Part of a long text around the first word of the query, for the results of message contents. */
export function fSearchSnippet(pText: string, pTokens: string[], pRadius = 48): string {
  const cFlat = pText.replace(/\s+/g, ' ').trim();
  // normalized character by character, to map a match back to the original text (NFD splits
  // some characters, such as Hangul syllables, into several)
  let vNormalized = '';
  const aOrigin: number[] = [];
  for (let vI = 0; vI < cFlat.length; vI++) {
    const cPart = fNormalizeSearchText(cFlat[vI]);
    vNormalized += cPart;
    for (let vJ = 0; vJ < cPart.length; vJ++) aOrigin.push(vI);
  }
  const cMatch = pTokens.length > 0 ? vNormalized.indexOf(pTokens[0]) : -1;
  if (cMatch < 0) return cFlat.slice(0, pRadius * 2);
  const cIndex = aOrigin[cMatch];

  const cStart = Math.max(0, cIndex - pRadius);
  const cEnd = Math.min(cFlat.length, cIndex + pTokens[0].length + pRadius);

  return `${cStart > 0 ? '…' : ''}${cFlat.slice(cStart, cEnd)}${cEnd < cFlat.length ? '…' : ''}`;
}

const aRevealClasses = ['ring-2', 'ring-primary', 'ring-offset-2', 'ring-offset-background'];

/**
 * Scrolls the first element of pIds that the page renders into view and highlights it for a moment:
 * the page of a search result renders after the navigation, so it is looked for during pTimeoutMs.
 * The next ids are fallbacks, such as the setting that a hidden dependent setting depends on.
 */
export function fRevealSearchTarget(pIds: string[], pTimeoutMs = 3000): void {
  if (typeof window === 'undefined' || pIds.length === 0) return;

  const cStart = performance.now();

  const fTry = () => {
    // the primary id may render a moment after its fallback, so the fallbacks wait half the time
    const cElapsed = performance.now() - cStart;
    const aCandidates = cElapsed < pTimeoutMs / 2 ? pIds.slice(0, 1) : pIds;

    for (const cId of aCandidates) {
      const cElement = document.getElementById(cId);
      if (!cElement) continue;

      cElement.scrollIntoView({ block: 'center', behavior: 'smooth' });
      cElement.classList.add(...aRevealClasses);
      setTimeout(() => cElement.classList.remove(...aRevealClasses), 2000);
      return;
    }

    if (cElapsed < pTimeoutMs) requestAnimationFrame(fTry);
  };

  requestAnimationFrame(fTry);
}
