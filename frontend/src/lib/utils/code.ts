import hljs from 'highlight.js';
import {
  cNewline,
  cDefaultLanguage,
  cLangPattern,
  cAmpersandRegex,
  cLtRegex,
  cGtRegex,
  cFencePattern,
  cTrimLeadingPaddingRegex,
  cTrimTrailingPaddingRegex
} from '$lib/constants';

export interface IncompleteCodeBlock {
  language: string;
  code: string;
  openingIndex: number;
}

/**
 * Strips empty lines (whitespace-only) from the start and end of code.
 *
 * Tool call payloads frequently arrive with surrounding whitespace from LLM
 * formatting (`"\nfunction ...\n"`). Preserving those newlines makes hljs emit
 * a leading/trailing empty line that `<pre>` then renders as a phantom row,
 * pushing real content away from the box edge. The trim keeps the body intact
 * so internal blank lines are still rendered as such.
 */
function fTrimCodePadding(pCode: string): string {
  return pCode.replace(cTrimLeadingPaddingRegex, '').replace(cTrimTrailingPaddingRegex, '');
}

function fEscapeCode(pCode: string): string {
  return pCode
    .replace(cAmpersandRegex, '&amp;')
    .replace(cLtRegex, '&lt;')
    .replace(cGtRegex, '&gt;');
}

/** Bounded cache for highlightCode results. */
const cHighlightCacheMaxSize = 64;
const dHighlightCache = new Map<string, string>();

/**
 * Highlights code using highlight.js
 * @param pCode - The code to highlight
 * @param pLanguage - The programming language
 * @param pAutoDetect - Fall back to `highlightAuto` when `language` is unknown.
 *   Callers rendering a still-streaming block should pass false: auto-detection
 *   costs ~38ms per call and re-guesses on every chunk, so the language (and
 *   therefore the whole highlight) flickers as the block grows.
 * @returns HTML string with syntax highlighting
 */
export function fHighlightCode(pCode: string, pLanguage: string, pAutoDetect = true): string {
  if (!pCode) return '';

  // Cache key includes language and autoDetect flag since results differ.
  // During streaming, the same code string may be highlighted repeatedly
  // (e.g., when text after a code block changes but the code itself doesn't).
  const cCacheKey = `${pLanguage}:${pAutoDetect}:${pCode}`;
  const cCached = dHighlightCache.get(cCacheKey);
  if (cCached) return cCached;

  const cTrimmed = fTrimCodePadding(pCode);
  let vResult: string;

  try {
    const cLang = pLanguage.toLowerCase();
    const cIsSupported = hljs.getLanguage(cLang);

    if (cIsSupported) {
      vResult = hljs.highlight(cTrimmed, { language: cLang }).value;
    } else if (pAutoDetect) {
      vResult = hljs.highlightAuto(cTrimmed).value;
    } else {
      vResult = fEscapeCode(cTrimmed);
    }
  } catch {
    vResult = fEscapeCode(cTrimmed);
  }

  if (dHighlightCache.size >= cHighlightCacheMaxSize) {
    dHighlightCache.delete(dHighlightCache.keys().next().value!);
  }
  dHighlightCache.set(cCacheKey, vResult);

  return vResult;
}

export { fTrimCodePadding as trimCodePadding };

/**
 * Detects if markdown ends with an incomplete code block (opened but not closed).
 * Returns the code block info if found, null otherwise.
 * @param pMarkdown - The raw markdown string to check
 * @returns IncompleteCodeBlock info or null
 */
export function fDetectIncompleteCodeBlock(pMarkdown: string): IncompleteCodeBlock | null {
  // Count all code fences in the markdown
  // A code block is incomplete if there's an odd number of ``` fences
  const cFenceMatcher = new RegExp(cFencePattern.source, cFencePattern.flags);
  const aFences: number[] = [];
  let vFenceMatch;

  while ((vFenceMatch = cFenceMatcher.exec(pMarkdown)) !== null) {
    // Store the position after the ```
    const cPos = vFenceMatch[0].startsWith(cNewline) ? vFenceMatch.index + 1 : vFenceMatch.index;
    aFences.push(cPos);
  }

  // If even number of fences (including 0), all code blocks are closed
  if (aFences.length % 2 === 0) {
    return null;
  }

  // Odd number means last code block is incomplete
  // The last fence is the opening of the incomplete block
  const cOpeningIndex = aFences[aFences.length - 1];
  const cAfterOpening = pMarkdown.slice(cOpeningIndex + 3);

  // Extract language and code content
  const cLangMatch = cAfterOpening.match(cLangPattern);
  const cLanguage = cLangMatch?.[1] || cDefaultLanguage;
  const cCodeStartIndex = cOpeningIndex + 3 + (cLangMatch?.[0]?.length ?? 0);
  const cCode = pMarkdown.slice(cCodeStartIndex);

  return {
    language: cLanguage,
    code: cCode,
    openingIndex: cOpeningIndex
  };
}
