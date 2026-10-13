import { AttachmentType, FileTypeCategory } from '$lib/enums';
import { getFileTypeCategory, getFileTypeCategoryByExtension } from '$lib/utils';

/**
 * Gets the file type category from an uploaded file, checking both MIME type and extension
 * @param pUploadedFile - The uploaded file to check
 * @returns The file type category or null if not recognized
 */
function fGetUploadedFileCategory(pUploadedFile: ChatUploadedFile): FileTypeCategory | null {
  // First try MIME type
  const cCategoryByMime = getFileTypeCategory(pUploadedFile.type);

  if (cCategoryByMime) {
    return cCategoryByMime;
  }

  // Fallback to extension (browsers don't always provide correct MIME types)
  return getFileTypeCategoryByExtension(pUploadedFile.name);
}

/**
 * Determines if an attachment or uploaded file is a text file
 * @param pUploadedFile - Optional uploaded file
 * @param pAttachment - Optional database attachment
 * @returns true if the file is a text file
 */
export function fIsTextFile(
  pAttachment?: DatabaseMessageExtra,
  pUploadedFile?: ChatUploadedFile
): boolean {
  if (pUploadedFile) {
    return fGetUploadedFileCategory(pUploadedFile) === FileTypeCategory.TEXT;
  }

  if (pAttachment) {
    return pAttachment.type === AttachmentType.TEXT;
  }

  return false;
}

/**
 * Determines if an attachment or uploaded file is an image
 * @param pUploadedFile - Optional uploaded file
 * @param pAttachment - Optional database attachment
 * @returns true if the file is an image
 */
export function fIsImageFile(
  pAttachment?: DatabaseMessageExtra,
  pUploadedFile?: ChatUploadedFile
): boolean {
  if (pUploadedFile) {
    return fGetUploadedFileCategory(pUploadedFile) === FileTypeCategory.IMAGE;
  }

  if (pAttachment) {
    return pAttachment.type === AttachmentType.IMAGE;
  }

  return false;
}

/**
 * Determines if an attachment or uploaded file is a PDF
 * @param pUploadedFile - Optional uploaded file
 * @param pAttachment - Optional database attachment
 * @returns true if the file is a PDF
 */
export function fIsPdfFile(
  pAttachment?: DatabaseMessageExtra,
  pUploadedFile?: ChatUploadedFile
): boolean {
  if (pUploadedFile) {
    return fGetUploadedFileCategory(pUploadedFile) === FileTypeCategory.PDF;
  }

  if (pAttachment) {
    return pAttachment.type === AttachmentType.PDF;
  }

  return false;
}

/**
 * Determines if an attachment or uploaded file is an audio file
 * @param pUploadedFile - Optional uploaded file
 * @param pAttachment - Optional database attachment
 * @returns true if the file is an audio file
 */
export function fIsAudioFile(
  pAttachment?: DatabaseMessageExtra,
  pUploadedFile?: ChatUploadedFile
): boolean {
  if (pUploadedFile) {
    return fGetUploadedFileCategory(pUploadedFile) === FileTypeCategory.AUDIO;
  }

  if (pAttachment) {
    return pAttachment.type === AttachmentType.AUDIO;
  }

  return false;
}

/**
 * Determines if an attachment or uploaded file is a video file
 * @param pUploadedFile - Optional uploaded file
 * @param pAttachment - Optional database attachment
 * @returns true if the file is a video file
 */
export function fIsVideoFile(
  pAttachment?: DatabaseMessageExtra,
  pUploadedFile?: ChatUploadedFile
): boolean {
  if (pUploadedFile) {
    return fGetUploadedFileCategory(pUploadedFile) === FileTypeCategory.VIDEO;
  }

  if (pAttachment) {
    return pAttachment.type === AttachmentType.VIDEO;
  }

  return false;
}
