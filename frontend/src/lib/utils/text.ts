import { cNewline } from '$lib/constants';

/**
 * Returns a shortened preview of the provided content capped at the given length.
 * Appends an ellipsis when the content exceeds the maximum.
 */
export function fGetPreviewText(pContent: string, pMax = 150): string {
  return pContent.length > pMax ? pContent.slice(0, pMax) + '...' : pContent;
}

/**
 * Generates a single-line title from a potentially multi-line prompt.
 * Uses the first non-empty line if `useFirstLine` is true.
 */
export function fGenerateConversationTitle(
  pContent: string,
  pUseFirstLine: boolean = false
): string {
  if (pUseFirstLine) {
    const cFirstLine = pContent.split(cNewline).find((pLine) => pLine.trim().length > 0);
    return cFirstLine ? cFirstLine.trim() : pContent.trim();
  }

  return pContent.trim();
}
