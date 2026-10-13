/**
 * Utility functions for markdown processing in MarkdownContent component.
 */

import type { RootContent as HastRootContent } from 'hast';

/**
 * Generates a unique identifier for a HAST node based on its position.
 * Used for stable block identification during incremental rendering.
 * @param pNode - The HAST root content node
 * @param pIndexFallback - Fallback index if position is unavailable
 * @returns Unique string identifier for the node
 */
export function fGetHastNodeId(pNode: HastRootContent, pIndexFallback: number): string {
  const cPosition = pNode.position;

  if (cPosition?.start?.offset != null && cPosition?.end?.offset != null) {
    return `hast-${cPosition.start.offset}-${cPosition.end.offset}`;
  }

  return `${pNode.type}-${pIndexFallback}`;
}

/**
 * Generates a hash for MDAST node based on its position.
 * Used for cache lookup during incremental rendering.
 */
export function fGetMdastNodeHash(pNode: unknown, pIndex: number): string {
  const cN = pNode as {
    type?: string;
    position?: { start?: { offset?: number }; end?: { offset?: number } };
  };

  if (cN.position?.start?.offset != null && cN.position?.end?.offset != null) {
    return `${cN.type}-${cN.position.start.offset}-${cN.position.end.offset}`;
  }

  return `${cN.type}-idx${pIndex}`;
}

/**
 * Determines if the new content is an append (new content added to existing blocks).
 * This is used to optimize cache reuse during streaming updates.
 *
 * @param pNewContent - The new markdown content
 * @param pPreviousContent - The previous markdown content to check against
 * @returns true if the content appears to be an append operation
 */
export function fIsAppendMode(pNewContent: string, pPreviousContent: string): boolean {
  return pPreviousContent.length > 0 && pNewContent.startsWith(pPreviousContent);
}

export interface CodeInfo {
  rawCode: string;
  language: string;
}

/**
 * Extracts code information from a button click target within a code block.
 * @param pTarget - The clicked button element
 * @returns Object with rawCode and language, or null if extraction fails
 */
export function fGetCodeInfoFromTarget(pTarget: HTMLElement): CodeInfo | null {
  const cWrapper = pTarget.closest('.code-block-wrapper');

  if (!cWrapper) {
    console.error('No wrapper found');
    return null;
  }

  const cCodeElement = cWrapper.querySelector<HTMLElement>('code[data-code-id]');

  if (!cCodeElement) {
    console.error('No code element found in wrapper');
    return null;
  }

  const cRawCode = cCodeElement.textContent ?? '';

  const cLanguageLabel = cWrapper.querySelector<HTMLElement>('.code-language');
  const cLanguage = cLanguageLabel?.textContent?.trim() || 'text';

  return { rawCode: cRawCode, language: cLanguage };
}
