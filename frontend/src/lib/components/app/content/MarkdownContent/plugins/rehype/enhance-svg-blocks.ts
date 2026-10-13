import { fTranslate } from '../../../../../../i18n';
/**
 * Rehype plugin to enhance svg blocks with wrapper, header, and action buttons.
 *
 * Wraps <pre class="svg-block"> elements with a container that includes:
 * - Language label ("svg")
 * - Copy button (copies svg source to clipboard)
 * - Preview button (opens fullscreen preview dialog)
 *
 * Operates directly on the HAST tree and reuses the shared code-block builders.
 */

import type { Plugin } from 'unified';
import type { Root, Element, ElementContent } from 'hast';
import { visit } from 'unist-util-visit';
import {
  cSvgWrapperClass,
  cSvgScrollContainerClass,
  cSvgBlockClass,
  cSvgLanguage,
  cSvgSourceAttr,
  cSvgIdAttr,
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
    idxSvgBlock?: number;
  }
}

export const fRehypeEnhanceSvgBlocks: Plugin<[], Root> = () => {
  return (pTree: Root) => {
    visit(pTree, 'element', (pNode: Element, pIndex, pParent) => {
      if (pNode.tagName !== 'pre' || !pParent || pIndex === undefined) return;

      const cClassName = pNode.properties?.className;
      if (!Array.isArray(cClassName)) return;

      const cIsSvg = cClassName.some((pCls) => typeof pCls === 'string' && pCls === cSvgBlockClass);

      if (!cIsSvg) return;

      const cSvgId = fGenerateBlockId(cSvgLanguage, 'idxSvgBlock');

      // Extract the svg source (text content of the pre element)
      const cSvgSource = pNode.children
        .map((pChild) => {
          if (pChild.type === 'text') return pChild.value;
          return '';
        })
        .join('');

      // Store the svg source in data attribute for copy and render
      pNode.properties = {
        ...pNode.properties,
        [cSvgSourceAttr]: cSvgSource,
        [cSvgIdAttr]: cSvgId
      };

      const aActions = [
        fCreateCopyButton(cSvgId, cSvgIdAttr, fTranslate('text_d11e0e1cfb0c')),
        fCreateToggleSourceButton(cSvgId, cSvgIdAttr, fTranslate('text_6a34d2f66ee3')),
        fCreatePreviewButton(cSvgId, cSvgIdAttr, fTranslate('text_bd4d3817f9f0'))
      ];

      const cHeader = fCreateBlockHeader(cSvgLanguage, cSvgId, cSvgIdAttr, aActions);
      const cPreservedCode = (pNode.data as DiagramPreData | undefined)?.sourceCode;
      const cSourceView = fCreateSourceView(cPreservedCode, cSvgSource, cSvgLanguage);
      const cWrapper = fCreateWrapper(
        cHeader,
        pNode,
        cSvgWrapperClass,
        cSvgScrollContainerClass,
        {
          [cSvgIdAttr]: cSvgId,
          [cDiagramViewModeAttr]: cDiagramViewRendered
        },
        [cSourceView]
      );

      // Replace pre with wrapper in parent
      (pParent.children as ElementContent[])[pIndex] = cWrapper;
    });
  };
};
