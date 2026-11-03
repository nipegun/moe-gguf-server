import type { Root as HastRoot } from 'hast';
import { visit } from 'unist-util-visit';
import type { DatabaseMessageExtra, DatabaseMessageExtraImageFile } from '$lib/types/database';
import { AttachmentType, UrlProtocol } from '$lib/enums';
import { fIsDataImageUrl, fIsHttpUrl, fIsSafeLinkUrl } from '$lib/utils/safe-url';

/**
 * Rehype plugin to resolve attachment image sources.
 * Converts attachment names (e.g., "mcp-attachment-xxx.png") to base64 data URLs.
 * Only base64 image data URLs are accepted from attachments (they may come from an
 * imported conversation), and image sources with any other scheme are dropped.
 */
export function fRehypeResolveAttachmentImages(pOptions: { attachments?: DatabaseMessageExtra[] }) {
  return (pTree: HastRoot) => {
    visit(pTree, 'element', (pNode) => {
      if (pNode.tagName === 'img' && pNode.properties?.src) {
        const cSrc = String(pNode.properties.src);

        // Keep image data URLs and external http(s) URLs as they are
        if (fIsDataImageUrl(cSrc) || fIsHttpUrl(cSrc)) {
          return;
        }

        // Find matching attachment
        const cAttachment = pOptions.attachments?.find(
          (pA): pA is DatabaseMessageExtraImageFile =>
            pA.type === AttachmentType.IMAGE && pA.name === cSrc
        );

        // Replace with base64 URL if found
        if (cAttachment && fIsDataImageUrl(cAttachment.base64Url)) {
          pNode.properties.src = cAttachment.base64Url;
          return;
        }

        // Neither an image nor a relative reference (javascript:, other data: types, etc.)
        if (cSrc.startsWith(UrlProtocol.DATA) || !fIsSafeLinkUrl(cSrc)) {
          delete pNode.properties.src;
        }
      }
    });
  };
}
