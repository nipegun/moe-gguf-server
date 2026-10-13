// JSON delimiters used while scanning partial streamed JSON. Single-char
// tokens so they only need eq-comparison, but naming them keeps the
// scanner readable and keeps the literal source-of-truth in one place.
const cJsonQuote = '"';
const cJsonBackslash = '\\';
const cJsonObjectOpen = '{';
const cJsonObjectClose = '}';
const cJsonArrayOpen = '[';
const cJsonArrayClose = ']';

// Trailing punctuation to strip before re-closing a partial object/array.
// Matches an optional trailing comma plus any trailing whitespace; lets
// us re-emit a syntactically-valid JSON document without an orphaned
// comma when the model cut off mid-key.
const cTrailingJsonPunctuationRegex = /,?\s*$/;

/** Bounded cache for parsePartialJsonArgs results. */
const cPartialJsonCacheMaxSize = 32;
const dPartialJsonCache = new Map<string, Record<string, unknown> | null>();

function fCacheResult(pInput: string, pResult: Record<string, unknown> | null): void {
  if (dPartialJsonCache.size >= cPartialJsonCacheMaxSize) {
    dPartialJsonCache.delete(dPartialJsonCache.keys().next().value!);
  }
  dPartialJsonCache.set(pInput, pResult);
}

// Parse partial tool-arg JSON streamed token-by-token. Closes any
// unterminated string and dangling open containers (in reverse order),
// so parsers can still surface keys already received while the call
// is still in flight. Memoized: the char-by-char scanner runs on every
// render during streaming even when toolArgs hasn't changed.
export function fParsePartialJsonArgs(pToolArgsString: string): Record<string, unknown> | null {
  const cCached = dPartialJsonCache.get(pToolArgsString);
  if (cCached !== undefined) return cCached;

  let dResult: Record<string, unknown> | null;

  try {
    const cParsed: unknown = JSON.parse(pToolArgsString);
    dResult =
      cParsed && typeof cParsed === 'object' && !Array.isArray(cParsed)
        ? (cParsed as Record<string, unknown>)
        : null;
  } catch {
    dResult = fScanPartialJson(pToolArgsString);
  }

  fCacheResult(pToolArgsString, dResult);
  return dResult;
}

/** Char-by-char scanner for unterminated partial JSON. */
function fScanPartialJson(pToolArgsString: string): Record<string, unknown> | null {
  let vInString = false;
  let vEscape = false;
  const aStack: ('{' | '[')[] = [];

  for (let vI = 0; vI < pToolArgsString.length; vI++) {
    const cCh = pToolArgsString[vI];
    if (vEscape) {
      vEscape = false;
      continue;
    }
    if (cCh === cJsonBackslash && vInString) {
      vEscape = true;
      continue;
    }
    if (cCh === cJsonQuote) {
      vInString = !vInString;
      continue;
    }
    if (vInString) continue;
    if (cCh === cJsonObjectOpen) aStack.push(cJsonObjectOpen);
    else if (cCh === cJsonObjectClose) {
      if (aStack.length === 0 || aStack[aStack.length - 1] !== cJsonObjectOpen) return null;
      aStack.pop();
    } else if (cCh === cJsonArrayOpen) aStack.push(cJsonArrayOpen);
    else if (cCh === cJsonArrayClose) {
      if (aStack.length === 0 || aStack[aStack.length - 1] !== cJsonArrayOpen) return null;
      aStack.pop();
    }
  }

  let vCompleted = pToolArgsString;
  if (vEscape) {
    // Dangling escape at end of partial JSON: escape the trailing
    // backslash as a literal so we can close the string cleanly.
    vCompleted += cJsonBackslash;
  }
  if (vInString) vCompleted += cJsonQuote;
  if (!vInString) vCompleted = vCompleted.replace(cTrailingJsonPunctuationRegex, '');

  // Close in reverse nesting order: innermost container first.
  for (let vI = aStack.length - 1; vI >= 0; vI--) {
    vCompleted += aStack[vI] === cJsonObjectOpen ? cJsonObjectClose : cJsonArrayClose;
  }

  try {
    const cParsed: unknown = JSON.parse(vCompleted);
    return cParsed && typeof cParsed === 'object' && !Array.isArray(cParsed)
      ? (cParsed as Record<string, unknown>)
      : null;
  } catch {
    return null;
  }
}
