import type { Plugin } from 'unified';
import type { Root, Element, ElementContent, Text } from 'hast';
import { visit } from 'unist-util-visit';

/**
 * Metadata a diagram pre carries on its unist data field. The source code holds
 * the highlighted code element captured before the pre became a render target,
 * which the enhancer reuses to build a matching source view.
 */
export interface DiagramPreData {
  sourceCode: Element;
}

/**
 * Recursively extracts all text content from a HAST node.
 * Handles nested elements (e.g., span wrappers from syntax highlighting).
 */
function fExtractText(pNode: ElementContent): string {
  if (pNode.type === 'text') return pNode.value;
  if (pNode.type === 'element') {
    return (pNode.children ?? []).map(fExtractText).join('');
  }
  return '';
}

/**
 * Builds a rehype plugin that converts <pre><code class="language-{language}">
 * blocks into <pre class="{targetClass}"> elements carrying the raw text.
 *
 * Accepts one or more source languages, and an optional contentGuard that
 * receives the trimmed text and decides whether the block qualifies. The guard
 * lets a shared fence language be claimed only when its content matches, e.g.
 * an xml block is converted to svg only when it starts with <svg.
 *
 * The result has no <code> child, so rehypeEnhanceCodeBlocks skips it. Rendering
 * happens client-side, so no markup is injected at this stage. Must run BEFORE
 * rehypeEnhanceCodeBlocks.
 */
export function fCreatePreTransform(
  pLanguages: string | string[],
  pTargetClass: string,
  pContentGuard?: (pText: string) => boolean
): Plugin<[], Root> {
  const cCodeClasses = (Array.isArray(pLanguages) ? pLanguages : [pLanguages]).map(
    (pLanguage) => `language-${pLanguage}`
  );

  return () => {
    return (pTree: Root) => {
      visit(pTree, 'element', (pNode: Element, pIndex, pParent) => {
        if (pNode.tagName !== 'pre' || !pParent || pIndex === undefined) return;

        const cCodeElement = pNode.children.find(
          (pChild): pChild is Element => pChild.type === 'element' && pChild.tagName === 'code'
        );

        if (!cCodeElement) return;

        const cClassName = cCodeElement.properties?.className;
        if (!Array.isArray(cClassName)) return;

        const cMatches = cClassName.some(
          (pCls) => typeof pCls === 'string' && cCodeClasses.includes(pCls)
        );

        if (!cMatches) return;

        // Recursively extract text to handle nested spans from syntax highlighting
        const cText = cCodeElement.children.map(fExtractText).join('').trim();

        if (!cText) return;

        if (pContentGuard && !pContentGuard(cText)) return;

        const dPre: Element = {
          type: 'element',
          tagName: 'pre',
          properties: {
            className: [pTargetClass]
          },
          children: [{ type: 'text', value: cText } as Text],
          // Keep the highlighted code element so the block can offer a source
          // view that matches the app code blocks without re highlighting.
          data: { sourceCode: cCodeElement } satisfies DiagramPreData
        };

        (pParent.children as ElementContent[])[pIndex] = dPre;
      });
    };
  };
}
