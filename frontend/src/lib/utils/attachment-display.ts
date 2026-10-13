import { AttachmentType, FileTypeCategory, SpecialFileType } from '$lib/enums';
import { getFileTypeCategory, getFileTypeCategoryByExtension, isImageFile } from '$lib/utils';
import type {
  AttachmentDisplayItemsOptions,
  ChatAttachmentDisplayItem,
  ChatUploadedFile
} from '$lib/types';

/**
 * Check if a display item represents an MCP prompt
 * (either from attachment type or uploaded file with mcpPrompt metadata)
 */
export function fIsMcpPrompt(pItem: ChatAttachmentDisplayItem): boolean {
  if (pItem.attachment?.type === AttachmentType.MCP_PROMPT) {
    return true;
  }
  if (pItem.uploadedFile?.type === SpecialFileType.MCP_PROMPT && pItem.uploadedFile.mcpPrompt) {
    return true;
  }
  return false;
}

/**
 * Check if a display item represents an MCP resource
 */
export function fIsMcpResource(pItem: ChatAttachmentDisplayItem): boolean {
  return pItem.attachment?.type === AttachmentType.MCP_RESOURCE;
}

/**
 * Gets the file type category from an uploaded file, checking both MIME type and extension
 */
function fGetUploadedFileCategory(pFile: ChatUploadedFile): FileTypeCategory | null {
  const cCategoryByMime = getFileTypeCategory(pFile.type);

  if (cCategoryByMime) {
    return cCategoryByMime;
  }

  return getFileTypeCategoryByExtension(pFile.name);
}

/**
 * Creates a unified list of display items from uploaded files and stored attachments.
 * Items are returned in reverse order (newest first).
 */
export function fGetAttachmentDisplayItems(
  pOptions: AttachmentDisplayItemsOptions
): ChatAttachmentDisplayItem[] {
  const { uploadedFiles: cUploadedFiles = [], attachments: cAttachments = [] } = pOptions;
  const aItems: ChatAttachmentDisplayItem[] = [];

  // Add uploaded files (ChatForm)
  for (const cFile of cUploadedFiles) {
    aItems.push({
      id: cFile.id,
      name: cFile.name,
      size: cFile.size,
      preview: cFile.preview,
      isImage: fGetUploadedFileCategory(cFile) === FileTypeCategory.IMAGE,
      isLoading: cFile.isLoading,
      loadError: cFile.loadError,
      uploadedFile: cFile,
      textContent: cFile.textContent
    });
  }

  // Add stored attachments (ChatMessage)
  for (const [cIndex, cAttachment] of cAttachments.entries()) {
    const cIsImage = isImageFile(cAttachment);

    aItems.push({
      id: `attachment-${cIndex}`,
      name: cAttachment.name,
      size: 'size' in cAttachment ? cAttachment.size : undefined,
      preview: cIsImage && 'base64Url' in cAttachment ? cAttachment.base64Url : undefined,
      isImage: cIsImage,
      attachment: cAttachment,
      attachmentIndex: cIndex,
      textContent: 'content' in cAttachment ? cAttachment.content : undefined
    });
  }

  return aItems.reverse();
}
