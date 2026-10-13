import type { Plugin } from 'unified';
import { visit } from 'unist-util-visit';
import type { Break, Content, Paragraph, PhrasingContent, Root, Text } from 'mdast';
import { cLineBreak, cNbsp, sPhraseParents, cTabAsSpaces } from '$lib/constants';

/**
 * remark plugin that rewrites raw HTML nodes into plain-text equivalents.
 *
 * remark parses inline HTML into `html` nodes even when we do not want to render
 * them. We turn each of those nodes into regular text (plus `<br>` break markers)
 * so the downstream rehype pipeline escapes the characters instead of executing
 * them. Leading spaces and tab characters are converted to non‑breaking spaces to
 * keep indentation identical to the original author input.
 */

function fPreserveIndent(pLine: string): string {
  let vIndex = 0;
  let vOutput = '';

  while (vIndex < pLine.length) {
    const cChar = pLine[vIndex];

    if (cChar === ' ') {
      vOutput += cNbsp;
      vIndex += 1;
      continue;
    }

    if (cChar === '\t') {
      vOutput += cTabAsSpaces;
      vIndex += 1;
      continue;
    }

    break;
  }

  return vOutput + pLine.slice(vIndex);
}

function fCreateLiteralChildren(pValue: string): PhrasingContent[] {
  const cLines = pValue.split(cLineBreak);
  const aNodes: PhrasingContent[] = [];

  for (const [cLineIndex, cRawLine] of cLines.entries()) {
    if (cLineIndex > 0) {
      aNodes.push({ type: 'break' } as Break as unknown as PhrasingContent);
    }

    aNodes.push({
      type: 'text',
      value: fPreserveIndent(cRawLine)
    } as Text as unknown as PhrasingContent);
  }

  if (!aNodes.length) {
    aNodes.push({ type: 'text', value: '' } as Text as unknown as PhrasingContent);
  }

  return aNodes;
}

export const fRemarkLiteralHtml: Plugin<[], Root> = () => {
  return (pTree) => {
    visit(pTree, 'html', (pNode, pIndex, pParent) => {
      if (!pParent || typeof pIndex !== 'number') {
        return;
      }

      const cReplacement = fCreateLiteralChildren(pNode.value);

      if (!sPhraseParents.has(pParent.type as string)) {
        const dParagraph: Paragraph = {
          type: 'paragraph',
          children: cReplacement as Paragraph['children'],
          data: { literalHtml: true }
        };

        const cSiblings = pParent.children as unknown as Content[];
        cSiblings.splice(pIndex, 1, dParagraph as unknown as Content);

        if (pIndex > 0) {
          const cPrevious = cSiblings[pIndex - 1] as Paragraph | undefined;

          if (
            cPrevious?.type === 'paragraph' &&
            (cPrevious.data as { literalHtml?: boolean } | undefined)?.literalHtml
          ) {
            const cPrevChildren = cPrevious.children as unknown as PhrasingContent[];

            if (cPrevChildren.length) {
              const cLastChild = cPrevChildren[cPrevChildren.length - 1];

              if (cLastChild.type !== 'break') {
                cPrevChildren.push({
                  type: 'break'
                } as Break as unknown as PhrasingContent);
              }
            }

            cPrevChildren.push(...(dParagraph.children as unknown as PhrasingContent[]));

            cSiblings.splice(pIndex, 1);

            return pIndex;
          }
        }

        return pIndex + 1;
      }

      (pParent.children as unknown as PhrasingContent[]).splice(
        pIndex,
        1,
        ...(cReplacement as unknown as PhrasingContent[])
      );

      return pIndex + cReplacement.length;
    });
  };
};
