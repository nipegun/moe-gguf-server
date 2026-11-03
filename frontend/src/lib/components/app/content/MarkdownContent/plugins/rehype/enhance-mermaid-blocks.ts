import { fTranslate } from '../../../../../../i18n';
/**
 * Rehype plugin to enhance mermaid diagram blocks with wrapper, header, and action buttons.
 *
 * Wraps <pre class="mermaid"> elements with a container that includes:
 * - Language label ("mermaid")
 * - Copy button (copies mermaid syntax to clipboard)
 * - Preview button (opens fullscreen preview dialog)
 *
 * This operates directly on the HAST tree for better performance,
 * avoiding the need to stringify and re-parse HTML.
 */

import type { Plugin } from 'unified';
import type { Root, Element, ElementContent } from 'hast';
import { visit } from 'unist-util-visit';
import {
  cMermaidWrapperClass,
  cMermaidScrollContainerClass,
  cMermaidBlockClass,
  cMermaidLanguage,
  cMermaidSyntaxAttr,
  cMermaidIdAttr,
  cDiagramViewModeAttr,
  cDiagramViewRendered
} from '$lib/constants';
import type { DiagramPreData } from './pre-transform';
import {
  fCreateBlockHeader,
  fCreateCopyButton,
  fCreatePreviewButton,
  fCreateToggleSourceButton,
  fCreateSourceView,
  fCreateWrapper,
  fGenerateBlockId
} from './code-block-utils';

declare global {
  interface Window {
    idxMermaidBlock?: number;
  }
}

/**
 * Rehype plugin to enhance mermaid diagram blocks with wrapper, header, and action buttons.
 * This plugin wraps <pre class="mermaid"> elements with a container that includes:
 * - Language label ("mermaid")
 * - Copy button
 * - Preview button
 */
export const fRehypeEnhanceMermaidBlocks: Plugin<[], Root> = () => {
  return (pTree: Root) => {
    visit(pTree, 'element', (pNode: Element, pIndex, pParent) => {
      if (pNode.tagName !== 'pre' || !pParent || pIndex === undefined) return;

      const cClassName = pNode.properties?.className;
      if (!Array.isArray(cClassName)) return;

      const cIsMermaid = cClassName.some(
        (pCls) => typeof pCls === 'string' && pCls === cMermaidBlockClass
      );

      if (!cIsMermaid) return;

      const cMermaidId = fGenerateBlockId(cMermaidLanguage, 'idxMermaidBlock');

      // Extract the mermaid syntax (text content of the pre element)
      const cDiagramText = pNode.children
        .map((pChild) => {
          if (pChild.type === 'text') return pChild.value;
          return '';
        })
        .join('');

      // Store the mermaid syntax in data attribute for copy functionality
      pNode.properties = {
        ...pNode.properties,
        [cMermaidSyntaxAttr]: cDiagramText,
        [cMermaidIdAttr]: cMermaidId
      };

      const aActions = [
        fCreateCopyButton(cMermaidId, cMermaidIdAttr, fTranslate('text_5763c5ded1ff')),
        fCreateToggleSourceButton(cMermaidId, cMermaidIdAttr, fTranslate('text_10b098b44705')),
        fCreatePreviewButton(cMermaidId, cMermaidIdAttr, fTranslate('text_6260fee8ea9f'))
      ];

      const cHeader = fCreateBlockHeader(cMermaidLanguage, cMermaidId, cMermaidIdAttr, aActions);
      const cPreservedCode = (pNode.data as DiagramPreData | undefined)?.sourceCode;
      const cSourceView = fCreateSourceView(cPreservedCode, cDiagramText, cMermaidLanguage);
      const cWrapper = fCreateWrapper(
        cHeader,
        pNode,
        cMermaidWrapperClass,
        cMermaidScrollContainerClass,
        {
          [cMermaidIdAttr]: cMermaidId,
          [cDiagramViewModeAttr]: cDiagramViewRendered
        },
        [cSourceView]
      );

      // Replace pre with wrapper in parent
      (pParent.children as ElementContent[])[pIndex] = cWrapper;
    });
  };
};
