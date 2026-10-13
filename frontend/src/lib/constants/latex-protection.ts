/**
 * Matches common Markdown code blocks to exclude them from further processing (e.g. LaTeX).
 * - Fenced: ```...```
 * - Inline: `...` (does NOT support nested backticks or multi-backtick syntax)
 *
 * Note: This pattern does not handle advanced cases like:
 *       `` `code with `backticks` `` or \\``...\\``
 */
export const cCodeBlockRegexp = /(```[\s\S]*?```|`[^`\n]+`)/g;

/**
 * Matches LaTeX math delimiters \(...\) and \[...\] only when not preceded by a backslash (i.e., not escaped),
 * while also capturing code blocks (```, `...`) so they can be skipped during processing.
 *
 * Uses negative lookbehind `(?<!\\)` to avoid matching \\( or \\[.
 * Using the look‑behind pattern `(?<!\\)` we skip matches
 * that are preceded by a backslash, e.g.
 * `Definitions\\(also called macros)` (title of chapter 20 in The TeXbook)
 * or `\\[4pt]` (LaTeX line-break).
 *
 * group 1: code-block
 * group 2: square-bracket
 * group 3: round-bracket
 */
export const cLatexMathAndCodePattern =
  /(```[\S\s]*?```|`.*?`)|(?<!\\)\\\[([\S\s]*?[^\\])\\]|(?<!\\)\\\((.*?)\\\)/g;

/** Regex to capture the content of a $$...\\\\...$$ block (display-formula with line-break) */
export const cLatexLinebreakRegexp = /\$\$([\s\S]*?\\\\[\s\S]*?)\$\$/;

/**
 * Matches the unescaped `\[...\]` display-math delimiter and surrounding
 * context so callers can insert line-breaks around the placeholder or convert
 * to inline when the formula has a non-empty trailing context (e.g. a table
 * cell that opens with `\[` and closes with content after `\]`).
 *
 * group 1: prefix before `\[`
 * group 2: formula body
 * group 3: trailing context after `\]`
 */
export const cLatexDisplayBlockRegexp = /([\S].*?)\\\[([\s\S]*?)\\\](.*)/g;

/**
 * Cheap gate for `preprocessLaTeX`. Every transformation it performs is triggered
 * by a `$` (inline/display math, currency escaping) or a backslash escape
 * (`\(`, `\[`, `\ce{`, `\pu{`). Text containing neither is returned untouched, so
 * this lets the caller skip the whole protect/restore pipeline.
 */
export const cLatexTriggerRegexp = /[$\\]/;

/** Inline LaTeX math delimiter (the dollar sign). */
export const cLatexInlineDelimiter = '$';

/** Display LaTeX math delimiter (paired dollar signs). */
export const cLatexDisplayDelimiter = '$$';

/** Matches a single non-whitespace character. */
export const cLatexNonWhitespaceRegexp = /\S/;

/** Matches a character that may appear adjacent to `$`, indicating a non-TeX
 * context such as an identifier (`var$`, `$var`), currency ($5), or code. */
export const cLatexNeighborCharRegexp = /[A-Za-z0-9_$-]/;

/** Matches a single digit (used to detect currency-like `$5`). */
export const cLatexDigitRegexp = /[0-9]/;

/** Matches the leading blockquote prefix (`> ` or `>`) on a markdown line. */
export const cLatexBlockquotePrefixRegexp = /^(>\s*)/;

/** Matches the placeholder inserted by the protect/restore pipeline for a
 * protected LaTeX expression. Group 1 is the index into `latexExpressions`. */
export const cLatexPlaceholderRegexp = /<<LATEX_(\d+)>>/g;

/** Matches the placeholder inserted by the protect/restore pipeline for a
 * protected code block. Group 1 is the index into `codeBlocks`. */
export const cCodeBlockPlaceholderRegexp = /<<CODE_BLOCK_(\d+)>>/g;

/** Matches a `$` immediately followed by a digit, which is treated as a
 * currency amount (e.g. `$5`) and escaped to `\$5` so it isn't parsed as math. */
export const cLatexCurrencyDollarRegexp = /\$(?=\d)/g;

/** Captures remaining `$$...$$`, `\[...\]`, `\(...\)` (only unescaped via
 * `(?<!\\)`) after the display-block pass has run. Group 1 holds the
 * matched formula. */
export const cLatexProtectRegexp = /(\$\$[\s\S]*?\$\$|(?<!\\)\\\[[\s\S]*?\\\]|(?<!\\)\\\(.*?\\\))/g;

/** Matches unescaped inline `\(...\)` (at least one char inside) used to
 * convert `\(` → `$` after the protect pass. */
export const cLatexInlineConvertRegexp = /(?<!\\)\\\((.+?)\\\)/g;

/** Matches unescaped display `\[...\]` used to convert `\[` → `$$`
 * after the protect pass. */
export const cLatexDisplayConvertRegexp = /(?<!\\)\\\[([\s\S]*?)\\\]/g;

/** `\(` — opens an inline LaTeX math block. */
export const cLatexInlineOpen = '\\(';

/** `\)` — closes an inline LaTeX math block. */
export const cLatexInlineClose = '\\)';

/** `\[` — opens a display LaTeX math block. */
export const cLatexDisplayOpen = '\\[';

/** `\]` — closes a display LaTeX math block. */
export const cLatexDisplayClose = '\\]';

/** `\` — the LaTeX escape character. */
export const cLatexBackslash = '\\';

/** `\$` — dollar sign escaped so it isn't parsed as math (used to disambiguate
 * currency amounts like `$5`). */
export const cLatexCurrencyEscape = '\\$';

/** `\ce{` — mhchem chemistry command prefix. */
export const cLatexMhchemCe = '\\ce{';

/** `\pu{` — mhchem physics-unit command prefix. */
export const cLatexMhchemPu = '\\pu{';

/** map from mchem-regexp to replacement */
export const aMhchemPatternMap: readonly [RegExp, string][] = [
  [/(\s)\$\\ce{/g, '$1$\\\\ce{'],
  [/(\s)\$\\pu{/g, '$1$\\\\pu{']
] as const;
