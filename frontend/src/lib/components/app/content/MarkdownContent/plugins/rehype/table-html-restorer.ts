/**
 * Rehype plugin to restore limited HTML elements inside Markdown table cells.
 *
 * ## Problem
 * The remark/rehype pipeline neutralizes inline HTML as literal text
 * (remarkLiteralHtml) so that XML/HTML snippets in LLM responses display
 * as-is instead of being rendered. This causes <br> and <ul> markup in
 * table cells to show as plain text.
 *
 * ## Solution
 * This plugin traverses the HAST post-conversion, parses whitelisted HTML
 * patterns from text nodes, and replaces them with actual HAST element nodes
 * that will be rendered as real HTML.
 *
 * ## Supported HTML
 * - `<br>` / `<br/>` / `<br />` - Line breaks (inline)
 * - `<ul><li>...</li></ul>` - Unordered lists (block)
 *
 * ## Key Implementation Details
 *
 * ### 1. Sibling Combination (Critical)
 * The Markdown pipeline may fragment content across multiple text nodes and `<br>`
 * elements. For example, `<ul><li>a</li></ul>` might arrive as:
 *   - Text: `"<ul>"`
 *   - Element: `<br>`
 *   - Text: `"<li>a</li></ul>"`
 *
 * We must combine consecutive text nodes and `<br>` elements into a single string
 * before attempting to parse list markup. Without this, list detection fails.
 *
 * ### 2. visitParents for Deep Traversal
 * Table cell content may be wrapped in intermediate elements (e.g., `<p>` tags).
 * Using `visitParents` instead of direct child iteration ensures we find text
 * nodes at any depth within the cell.
 *
 * ### 3. Reference Comparison for No-Op Detection
 * When checking if `<br>` expansion changed anything, we compare:
 *   `expanded.length !== 1 || expanded[0] !== textNode`
 *
 * This catches both cases:
 * - Multiple nodes created (text was split)
 * - Single NEW node created (original had only `<br>`, now it's an element)
 *
 * A simple `length > 1` check would miss the single `<br>` case.
 *
 * ### 4. Strict List Validation
 * `parseList()` rejects malformed markup by checking for garbage text between
 * `<li>` elements. This prevents creating broken DOM from partial matches like
 * `<ul>garbage<li>a</li></ul>`.
 *
 * ### 5. Newline Substitution for `<br>` in Combined String
 * When combining siblings, existing `<br>` elements become `\n` in the combined
 * string. This allows list content to span visual lines while still being parsed
 * as a single unit.
 *
 * @example
 * // Input Markdown:
 * // | Feature | Notes |
 * // |---------|-------|
 * // | Multi-line | First<br>Second |
 * // | List | <ul><li>A</li><li>B</li></ul> |
 * //
 * // Without this plugin: <br> and <ul> render as literal text
 * // With this plugin: <br> becomes line break, <ul> becomes actual list
 */

import type { Plugin } from 'unified';
import type { Element, ElementContent, Root, Text } from 'hast';
import { visit } from 'unist-util-visit';
import { visitParents } from 'unist-util-visit-parents';
import { cBrPattern, cListPattern, cLiPattern } from '$lib/constants';

/**
 * Expands text containing `<br>` tags into an array of text nodes and br elements.
 */
function fExpandBrTags(pValue: string): ElementContent[] {
  const aMatches = [...pValue.matchAll(cBrPattern)];
  if (!aMatches.length) return [{ type: 'text', value: pValue } as Text];

  const aResult: ElementContent[] = [];
  let vCursor = 0;

  for (const cM of aMatches) {
    if (cM.index! > vCursor) {
      aResult.push({ type: 'text', value: pValue.slice(vCursor, cM.index) } as Text);
    }
    aResult.push({ type: 'element', tagName: 'br', properties: {}, children: [] } as Element);
    vCursor = cM.index! + cM[0].length;
  }

  if (vCursor < pValue.length) {
    aResult.push({ type: 'text', value: pValue.slice(vCursor) } as Text);
  }

  return aResult;
}

/**
 * Parses a `<ul><li>...</li></ul>` string into a HAST element.
 * Returns null if the markup is malformed or contains unexpected content.
 */
function fParseList(pValue: string): Element | null {
  const cMatch = pValue.trim().match(cListPattern);
  if (!cMatch) return null;

  const cBody = cMatch[1];
  const aItems: ElementContent[] = [];
  let vCursor = 0;

  for (const cLiMatch of cBody.matchAll(cLiPattern)) {
    // Reject if there's non-whitespace between list items
    if (cBody.slice(vCursor, cLiMatch.index!).trim()) return null;

    aItems.push({
      type: 'element',
      tagName: 'li',
      properties: {},
      children: fExpandBrTags(cLiMatch[1] ?? '')
    } as Element);

    vCursor = cLiMatch.index! + cLiMatch[0].length;
  }

  // Reject if no items found or trailing garbage exists
  if (!aItems.length || cBody.slice(vCursor).trim()) return null;

  return { type: 'element', tagName: 'ul', properties: {}, children: aItems } as Element;
}

/**
 * Processes a single table cell, restoring HTML elements from text content.
 */
function fProcessCell(pCell: Element) {
  visitParents(pCell, 'text', (pTextNode: Text, pAncestors) => {
    const cParent = pAncestors[pAncestors.length - 1];
    if (!cParent || cParent.type !== 'element') return;

    const cParentEl = cParent as Element;
    const cSiblings = cParentEl.children as ElementContent[];
    const cStartIndex = cSiblings.indexOf(pTextNode as ElementContent);
    if (cStartIndex === -1) return;

    // Combine consecutive text nodes and <br> elements into one string
    let vCombined = '';
    let vEndIndex = cStartIndex;

    for (let vI = cStartIndex; vI < cSiblings.length; vI++) {
      const cSib = cSiblings[vI];
      if (cSib.type === 'text') {
        vCombined += (cSib as Text).value;
        vEndIndex = vI;
      } else if (cSib.type === 'element' && (cSib as Element).tagName === 'br') {
        vCombined += '\n';
        vEndIndex = vI;
      } else {
        break;
      }
    }

    // Try parsing as list first (replaces entire combined range)
    const cList = fParseList(vCombined);
    if (cList) {
      cSiblings.splice(cStartIndex, vEndIndex - cStartIndex + 1, cList);
      return;
    }

    // Otherwise, just expand <br> tags in this text node
    const cExpanded = fExpandBrTags(pTextNode.value);
    if (cExpanded.length !== 1 || cExpanded[0] !== pTextNode) {
      cSiblings.splice(cStartIndex, 1, ...cExpanded);
    }
  });
}

export const fRehypeRestoreTableHtml: Plugin<[], Root> = () => (pTree) => {
  visit(pTree, 'element', (pNode: Element) => {
    if (pNode.tagName === 'td' || pNode.tagName === 'th') {
      fProcessCell(pNode);
    }
  });
};
