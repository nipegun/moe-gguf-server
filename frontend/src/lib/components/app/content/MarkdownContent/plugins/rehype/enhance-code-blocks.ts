import { fTranslate } from '../../../../../../i18n';
/**
 * Rehype plugin to enhance code blocks with wrapper, header, and action buttons.
 *
 * Wraps <pre><code> elements with a container that includes:
 * - Language label
 * - Copy button
 * - Preview button (for HTML code blocks)
 *
 * This operates directly on the HAST tree for better performance,
 * avoiding the need to stringify and re-parse HTML.
 */

import type { Plugin } from 'unified';
import type { Root, Element, ElementContent } from 'hast';
import { visit } from 'unist-util-visit';
import { cCodeBlockScrollContainerClass, cCodeBlockWrapperClass } from '$lib/constants';
import {
  fCreateBlockHeader,
  fCreateCopyButton,
  fCreatePreviewButton,
  fCreateWrapper,
  fGenerateBlockId
} from './code-block-utils';

declare global {
  interface Window {
    idxCodeBlock?: number;
  }
}

function fExtractLanguage(pCodeElement: Element): string {
  const cClassName = pCodeElement.properties?.className;
  if (!Array.isArray(cClassName)) return 'text';

  for (const cCls of cClassName) {
    if (typeof cCls === 'string' && cCls.startsWith('language-')) {
      return cCls.replace('language-', '');
    }
  }

  return 'text';
}

/**
 * Rehype plugin to enhance code blocks with wrapper, header, and action buttons.
 * This plugin wraps <pre><code> elements with a container that includes:
 * - Language label
 * - Copy button
 * - Preview button (for HTML code blocks)
 */
export const fRehypeEnhanceCodeBlocks: Plugin<[], Root> = () => {
  return (pTree: Root) => {
    visit(pTree, 'element', (pNode: Element, pIndex, pParent) => {
      if (pNode.tagName !== 'pre' || !pParent || pIndex === undefined) return;

      const cCodeElement = pNode.children.find(
        (pChild): pChild is Element => pChild.type === 'element' && pChild.tagName === 'code'
      );

      if (!cCodeElement) return;

      const cLanguage = fExtractLanguage(cCodeElement);
      const cCodeId = fGenerateBlockId('code', 'idxCodeBlock');

      cCodeElement.properties = {
        ...cCodeElement.properties,
        'data-code-id': cCodeId
      };

      const aActions: Element[] = [
        fCreateCopyButton(cCodeId, 'data-code-id', fTranslate('text_49a0053f3b0d'))
      ];

      if (cLanguage.toLowerCase() === 'html') {
        aActions.push(
          fCreatePreviewButton(cCodeId, 'data-code-id', fTranslate('text_7e8611fc24e8'))
        );
      }

      const cHeader = fCreateBlockHeader(cLanguage, cCodeId, 'data-code-id', aActions);
      const cWrapper = fCreateWrapper(
        cHeader,
        pNode,
        cCodeBlockWrapperClass,
        cCodeBlockScrollContainerClass
      );

      // Replace pre with wrapper in parent
      (pParent.children as ElementContent[])[pIndex] = cWrapper;
    });
  };
};
