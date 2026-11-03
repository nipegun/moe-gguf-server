// Conversation filename constants

// Length of the trimmed conversation ID in the filename
export const cExportConvIdTrimLength = 8;
// Maximum length of the sanitized conversation name snippet
export const cExportConvNameSuffixMaxLength = 20;
// Characters to keep in the ISO timestamp. 19 keeps 2026-01-01T00:00:00
export const cIsoTimestampSliceLength = 19;

// Producer marker carried by the session record of a JSONL export
export const cSessionHarness = 'mgs';

// Replacements for making the conversation title filename-friendly
export const cNonAlphanumericRegex = /[^a-z0-9]/gi;
export const cExportConvNonalnumReplacement = '_';
export const cMultipleUnderscoreRegex = /_+/g;

// Replacements to the ISO date for use in the export filename
export const cIsoDateTimeSeparator = 'T';
export const cIsoDateTimeSeparatorReplacement = '_';

export const cIsoTimeSeparator = ':';
export const cIsoTimeSeparatorReplacement = '-';
