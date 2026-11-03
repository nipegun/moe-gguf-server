import type { AgenticConfig } from '$lib/types/agentic';

export const cAttachmentSavedRegex = /\[Attachment saved: ([^\]]+)\]/;

// JSON detection: trimmed content opens with an object or array literal.
export const cToolResultJsonOpenRegex = /^[[{]/;

// Markdown structural markers used by `looksLikeMarkdown`. Inline / line-level.
export const cMarkdownCodeFenceRegex = /^(```|~~~)/m;
export const cMarkdownAtxHeadingRegex = /^#{1,6}\s+\S/;
export const cMarkdownBlockquoteRegex = /^>\s+\S/;
export const cMarkdownListBulletRegex = /^\s*[-*+]\s+\S/;
export const cMarkdownListNumberedRegex = /^\s*\d+[.)]\s+\S/;
export const cMarkdownLinkRegex = /\[[^\]\n]+\]\([^)\s]+\)/;
export const cMarkdownBoldRegex = /\*\*[^*\n]+\*\*|__[^_\n]+__/;
export const cMarkdownTableSeparatorRegex = /^\s*\|?[\s:|-]+\|?\s*$/;

// Search-summary wire format used by file-glob and grep tools:
//   <matches>
//   ---
//   Total matches: N
export const cSearchSummarySeparator = '---\n';
export const cSearchSummaryTotalRegex = /Total matches:\s*(\d+)/;

// Separator rendered between stats in the tool-result footer (e.g. between a
// result message and the byte/edit count). Plain ASCII spaces bracket a hyphen
// so the whole " - " sits on one visual line even when the surrounding text
// wraps mid-paragraph.
export const cResultStatSeparator = ' - ';

export const cDefaultAgenticConfig: AgenticConfig = {
  enabled: true,
  maxTurns: 50
} as const;

export const cReasoningTags = {
  START: '<think>',
  END: '</think>'
} as const;
