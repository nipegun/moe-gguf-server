import {
  cCodeBlockPlaceholderRegexp,
  cCodeBlockRegexp,
  cLatexBackslash,
  cLatexBlockquotePrefixRegexp,
  cLatexCurrencyDollarRegexp,
  cLatexCurrencyEscape,
  cLatexDigitRegexp,
  cLatexDisplayBlockRegexp,
  cLatexDisplayClose,
  cLatexDisplayConvertRegexp,
  cLatexDisplayDelimiter,
  cLatexDisplayOpen,
  cLatexInlineClose,
  cLatexInlineConvertRegexp,
  cLatexInlineDelimiter,
  cLatexInlineOpen,
  cLatexMathAndCodePattern,
  cLatexMhchemCe,
  cLatexMhchemPu,
  cLatexLinebreakRegexp,
  cLatexNeighborCharRegexp,
  cLatexNonWhitespaceRegexp,
  cLatexPlaceholderRegexp,
  cLatexProtectRegexp,
  cLatexTriggerRegexp,
  aMhchemPatternMap,
  cNewline
} from '$lib/constants';

/**
 * Replaces inline LaTeX expressions enclosed in `$...$` with placeholders, avoiding dollar signs
 * that appear to be part of monetary values or identifiers.
 *
 * This function processes the input line by line and skips `$` sequences that are likely
 * part of money amounts (e.g., `$5`, `$100.99`) or code-like tokens (e.g., `var$`, `$var`).
 * Valid LaTeX inline math is replaced with a placeholder like `<<LATEX_0>>`, and the
 * actual LaTeX content is stored in the provided `latexExpressions` array.
 *
 * @param pContent - The input text potentially containing LaTeX expressions.
 * @param pLatexExpressions - An array used to collect extracted LaTeX expressions.
 * @returns The processed string with LaTeX replaced by placeholders.
 */
export function fMaskInlineLaTeX(pContent: string, pLatexExpressions: string[]): string {
  if (!pContent.includes(cLatexInlineDelimiter)) {
    return pContent;
  }
  return pContent
    .split(cNewline)
    .map((pLine) => {
      if (pLine.indexOf(cLatexInlineDelimiter) == -1) {
        return pLine;
      }

      let vProcessedLine = '';
      let vCurrentPosition = 0;

      while (vCurrentPosition < pLine.length) {
        const cOpenDollarIndex = pLine.indexOf(cLatexInlineDelimiter, vCurrentPosition);

        if (cOpenDollarIndex == -1) {
          vProcessedLine += pLine.slice(vCurrentPosition);
          break;
        }

        // Is there a next $-sign?
        const cCloseDollarIndex = pLine.indexOf(cLatexInlineDelimiter, cOpenDollarIndex + 1);

        if (cCloseDollarIndex == -1) {
          vProcessedLine += pLine.slice(vCurrentPosition);
          break;
        }

        const cCharBeforeOpen = cOpenDollarIndex > 0 ? pLine[cOpenDollarIndex - 1] : '';
        const cCharAfterOpen = pLine[cOpenDollarIndex + 1];
        const cCharBeforeClose =
          cOpenDollarIndex + 1 < cCloseDollarIndex ? pLine[cCloseDollarIndex - 1] : '';
        const cCharAfterClose =
          cCloseDollarIndex + 1 < pLine.length ? pLine[cCloseDollarIndex + 1] : '';

        let vShouldSkipAsNonLatex = false;

        if (cCloseDollarIndex == vCurrentPosition + 1) {
          // No content
          vShouldSkipAsNonLatex = true;
        }

        if (cLatexNeighborCharRegexp.test(cCharBeforeOpen)) {
          // Character, digit, $, _ or - before first '$', no TeX.
          vShouldSkipAsNonLatex = true;
        }

        if (
          cLatexDigitRegexp.test(cCharAfterOpen) &&
          (cLatexNeighborCharRegexp.test(cCharAfterClose) || ' ' == cCharBeforeClose)
        ) {
          // First $ seems to belong to an amount.
          vShouldSkipAsNonLatex = true;
        }

        if (vShouldSkipAsNonLatex) {
          vProcessedLine += pLine.slice(vCurrentPosition, cOpenDollarIndex + 1);
          vCurrentPosition = cOpenDollarIndex + 1;

          continue;
        }

        // Treat as LaTeX
        vProcessedLine += pLine.slice(vCurrentPosition, cOpenDollarIndex);
        const cLatexContent = pLine.slice(cOpenDollarIndex, cCloseDollarIndex + 1);
        pLatexExpressions.push(cLatexContent);
        vProcessedLine += `<<LATEX_${pLatexExpressions.length - 1}>>`;
        vCurrentPosition = cCloseDollarIndex + 1;
      }

      return vProcessedLine;
    })
    .join(cNewline);
}

function fEscapeBrackets(pText: string): string {
  return pText.replace(
    cLatexMathAndCodePattern,
    (
      pMatch: string,
      pCodeBlock: string | undefined,
      pSquareBracket: string | undefined,
      pRoundBracket: string | undefined
    ): string => {
      if (pCodeBlock != null) {
        return pCodeBlock;
      } else if (pSquareBracket != null) {
        return `${cLatexDisplayDelimiter}${pSquareBracket}${cLatexDisplayDelimiter}`;
      } else if (pRoundBracket != null) {
        return `${cLatexInlineDelimiter}${pRoundBracket}${cLatexInlineDelimiter}`;
      }

      return pMatch;
    }
  );
}

// Escape $\\ce{...} → $\\ce{...} but with proper handling
function fEscapeMhchem(pText: string): string {
  return aMhchemPatternMap.reduce((pResult, [pPattern, pReplacement]) => {
    return pResult.replace(pPattern, pReplacement);
  }, pText);
}

const cDoEscapeMhchem = false;

/**
 * Preprocesses markdown content to safely handle LaTeX math expressions while protecting
 * against false positives (e.g., dollar amounts like $5.99) and ensuring proper rendering.
 *
 * This function:
 * - Protects code blocks (```) and inline code (`...`)
 * - Safeguards block and inline LaTeX: \(...\), \[...\], $$...$$, and selective $...$
 * - Escapes standalone dollar signs before numbers (e.g., $5 → \$5) to prevent misinterpretation
 * - Restores protected LaTeX and code blocks after processing
 * - Converts \(...\) → $...$ and \[...\] → $$...$$ for compatibility with math renderers
 * - Applies additional escaping for brackets and mhchem syntax if needed
 *
 * @param content - The raw text (e.g., markdown) that may contain LaTeX or code blocks.
 * @returns The preprocessed string with properly escaped and normalized LaTeX.
 *
 * @example
 * preprocessLaTeX("Price: $10. The equation is \\(x^2\\).")
 * // → "Price: $10. The equation is $x^2$."
 */
/** Bounded cache for preprocessLaTeX results. */
const cLatexCacheMaxSize = 64;
const dLatexCache = new Map<string, string>();

export function fPreprocessLaTeX(pContent: string): string {
  // See also:
  // https://github.com/danny-avila/LibreChat/blob/main/client/src/utils/latex.ts

  // Memoize on the input string. During streaming the prefix before an
  // incomplete code block stays the same across multiple tokens, so the
  // full protect/restore pipeline would re-run unnecessarily.
  const cCached = dLatexCache.get(pContent);
  if (cCached !== undefined) return cCached;

  // Save original before the function mutates `content` through steps 0-8
  const cOriginalContent = pContent;

  // Every step below keys off a `$` or a backslash escape (\[ \] \( \) \ce{ \pu{).
  // With neither present the protect/restore passes round-trip the input
  // unchanged, so skip them: the step 2 scan is O(n^2) in line length and costs
  // ~90ms on a 26KB single-line message that contains no math at all. This
  // matters during streaming, where the whole message is reprocessed per frame.
  if (!cLatexTriggerRegexp.test(pContent)) {
    if (dLatexCache.size >= cLatexCacheMaxSize) {
      dLatexCache.delete(dLatexCache.keys().next().value!);
    }
    dLatexCache.set(cOriginalContent, pContent);
    return pContent;
  }

  // Step 0: Temporarily remove blockquote markers (>) to process LaTeX correctly
  // Store the structure so we can restore it later
  const dBlockquoteMarkers: Map<number, string> = new Map();
  const cLines = pContent.split(cNewline);
  const cProcessedLines = cLines.map((pLine, pIndex) => {
    const cMatch = pLine.match(cLatexBlockquotePrefixRegexp);
    if (cMatch) {
      dBlockquoteMarkers.set(pIndex, cMatch[1]);
      return pLine.slice(cMatch[1].length);
    }
    return pLine;
  });
  pContent = cProcessedLines.join(cNewline);

  // Step 1: Protect code blocks
  const aCodeBlocks: string[] = [];

  pContent = pContent.replace(cCodeBlockRegexp, (pMatch) => {
    aCodeBlocks.push(pMatch);

    return `<<CODE_BLOCK_${aCodeBlocks.length - 1}>>`;
  });

  // Step 2: Protect existing LaTeX expressions
  const aLatexExpressions: string[] = [];

  // Match \S...\[...\] and protect them and insert a line-break.
  // Guarded: with no `\[` present this pattern still probes every start offset,
  // expanding `.*?` to the end of each line before failing - O(n^2) for nothing.
  if (pContent.includes(cLatexDisplayOpen)) {
    pContent = pContent.replace(cLatexDisplayBlockRegexp, (pMatch, pGroup1, pGroup2, pGroup3) => {
      // Check if there are characters following the formula (display-formula in a table-cell?)
      if (pGroup1.endsWith(cLatexBackslash)) {
        return pMatch; // Backslash before \[, do nothing.
      }
      const cHasSuffix = cLatexNonWhitespaceRegexp.test(pGroup3);
      let vOptBreak;

      if (cHasSuffix) {
        aLatexExpressions.push(`${cLatexInlineOpen}${pGroup2.trim()}${cLatexInlineClose}`); // Convert into inline.
        vOptBreak = '';
      } else {
        aLatexExpressions.push(`${cLatexDisplayOpen}${pGroup2}${cLatexDisplayClose}`);
        vOptBreak = cNewline;
      }

      return `${pGroup1}${vOptBreak}<<LATEX_${aLatexExpressions.length - 1}>>${vOptBreak}${pGroup3}`;
    });
  }

  // Match \(...\), \[...\], $$...$$ and protect them
  pContent = pContent.replace(cLatexProtectRegexp, (pMatch) => {
    aLatexExpressions.push(pMatch);

    return `<<LATEX_${aLatexExpressions.length - 1}>>`;
  });

  // Protect inline $...$ but NOT if it looks like money (e.g., $10, $3.99)
  pContent = fMaskInlineLaTeX(pContent, aLatexExpressions);

  // Step 3: Escape standalone $ before digits (currency like $5 → \$5)
  // (Now that inline math is protected, this will only escape dollars not already protected)
  pContent = pContent.replace(cLatexCurrencyDollarRegexp, cLatexCurrencyEscape);

  // Step 4: Restore protected LaTeX expressions (they are valid)
  pContent = pContent.replace(cLatexPlaceholderRegexp, (pIgnored, pIndex) => {
    let vExpr = aLatexExpressions[parseInt(pIndex)];
    const cMatch = vExpr.match(cLatexLinebreakRegexp);
    if (cMatch) {
      // Katex: The $$-delimiters should be in their own line
      // if there are \\-line-breaks.
      const cFormula = cMatch[1];
      const cPrefix = cFormula.startsWith(cNewline) ? '' : cNewline;
      const cSuffix = cFormula.endsWith(cNewline) ? '' : cNewline;
      vExpr = cLatexDisplayDelimiter + cPrefix + cFormula + cSuffix + cLatexDisplayDelimiter;
    }
    return vExpr;
  });

  // Step 5: Apply additional escaping functions (brackets and mhchem)
  // This must happen BEFORE restoring code blocks to avoid affecting code content
  pContent = fEscapeBrackets(pContent);

  if (cDoEscapeMhchem && (pContent.includes(cLatexMhchemCe) || pContent.includes(cLatexMhchemPu))) {
    pContent = fEscapeMhchem(pContent);
  }

  // Step 6: Convert remaining \(...\) → $...$, \[...\] → $$...$$
  // This must happen BEFORE restoring code blocks to avoid affecting code content
  pContent = pContent
    // Using the look‑behind pattern `(?<!\\)` we skip matches
    // that are preceded by a backslash, e.g.
    // `Definitions\\(also called macros)` (title of chapter 20 in The TeXbook).
    .replace(cLatexInlineConvertRegexp, (pIgnored, pFormula: string) => {
      return `${cLatexInlineDelimiter}${pFormula}${cLatexInlineDelimiter}`;
    }) // inline
    .replace(
      // Using the look‑behind pattern `(?<!\\)` we skip matches
      // that are preceded by a backslash, e.g. `\\[4pt]`.
      cLatexDisplayConvertRegexp, // display, see also PR #16599
      (pIgnored, pFormula: string) => {
        return `${cLatexDisplayDelimiter}${pFormula}${cLatexDisplayDelimiter}`;
      }
    );

  // Step 7: Restore code blocks
  // This happens AFTER all LaTeX conversions to preserve code content
  pContent = pContent.replace(cCodeBlockPlaceholderRegexp, (pIgnored, pIndex) => {
    return aCodeBlocks[parseInt(pIndex)];
  });

  // Step 8: Restore blockquote markers
  if (dBlockquoteMarkers.size > 0) {
    const cFinalLines = pContent.split(cNewline);
    const cRestoredLines = cFinalLines.map((pLine, pIndex) => {
      const cMarker = dBlockquoteMarkers.get(pIndex);
      return cMarker ? cMarker + pLine : pLine;
    });
    pContent = cRestoredLines.join(cNewline);
  }

  if (dLatexCache.size >= cLatexCacheMaxSize) {
    dLatexCache.delete(dLatexCache.keys().next().value!);
  }
  dLatexCache.set(cOriginalContent, pContent);

  return pContent;
}
