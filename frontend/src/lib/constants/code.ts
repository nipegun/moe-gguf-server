export const cNewline = '\n';
export const cTab = '\t';
export const cDefaultLanguage = 'text';
export const cLangPattern = /^(\w*)\n?/;
export const cAmpersandRegex = /&/g;
export const cLtRegex = /</g;
export const cGtRegex = />/g;
export const cFencePattern = /^```|\n```/g;

// Whitespace-only empty lines (between start of string and first non-empty line).
// Used by trimCodePadding to drop leading/trailing phantom blank rows from LLM
// payload wrappers without touching internal blank lines.
export const cTrimLeadingPaddingRegex = /^(?:[ \t]*\n)+/;
export const cTrimTrailingPaddingRegex = /(?:\n[ \t]*)+$/;

// Matches either Unix or Windows path separators so `String.split(REGEX)` can
// recover the trailing file-name segment from either `/foo/bar.txt` or
// `C:\foo\bar.txt`. Used wherever a parameter accepts a user-supplied path.
export const cFilePathSeparatorRegex = /[\\/]/;

// Matches the `text:` prefix that file-type identifiers use to denote a
// plain-text language (e.g. `text:typescript`). Used by tool-call renderers
// to recover the underlying highlight.js language.
export const cTextLanguagePrefixRegex = /^text:/;
