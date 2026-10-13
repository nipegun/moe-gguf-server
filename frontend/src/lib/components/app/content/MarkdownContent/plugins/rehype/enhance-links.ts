/**
 * Rehype plugin to enhance links with security attributes.
 *
 * Adds target="_blank" and rel="noopener noreferrer" to all anchor elements,
 * ensuring external links open in new tabs safely.
 */

import type { Plugin } from 'unified';
import type { Root, Element } from 'hast';
import { visit } from 'unist-util-visit';
import { fIsSafeLinkUrl } from '$lib/utils/safe-url';

/**
 * Rehype plugin that adds security attributes to all links.
 * This plugin ensures external links open in new tabs safely by adding:
 * - target="_blank"
 * - rel="noopener noreferrer"
 * Links whose scheme is not http(s), mailto or tel (javascript:, data:, etc.)
 * lose their href, so model output can not create script navigations.
 */
export const fRehypeEnhanceLinks: Plugin<[], Root> = () => {
  return (pTree: Root) => {
    visit(pTree, 'element', (pNode: Element) => {
      if (pNode.tagName !== 'a') return;

      const cProps = pNode.properties ?? {};

      // Only modify if href exists
      if (!cProps.href) return;

      if (!fIsSafeLinkUrl(String(cProps.href))) {
        delete cProps.href;
        pNode.properties = cProps;
        return;
      }

      cProps.target = '_blank';
      cProps.rel = ['noopener', 'noreferrer'];
      pNode.properties = cProps;
    });
  };
};
