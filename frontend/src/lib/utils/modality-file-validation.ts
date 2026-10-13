import { fTranslate } from '../i18n';
/**
 * File validation utilities based on model modalities
 * Ensures only compatible file types are processed based on model capabilities
 */

import { getFileTypeCategory } from '$lib/utils';
import { FileTypeCategory } from '$lib/enums';
import type { ModalityCapabilities } from '$lib/types';

/**
 * Check if a file type is supported by the given modalities
 * @param pFilename - The filename to check
 * @param pMimeType - The MIME type of the file
 * @param pCapabilities - The modality capabilities to check against
 * @returns true if the file type is supported
 */
export function fIsFileTypeSupportedByModel(
  pFilename: string,
  pMimeType: string | undefined,
  pCapabilities: ModalityCapabilities
): boolean {
  const cCategory = pMimeType ? getFileTypeCategory(pMimeType) : null;

  // If we can't determine the category from MIME type, fall back to general support check
  if (!cCategory) {
    // For unknown types, only allow if they might be text files
    // This is a conservative approach for edge cases
    return true; // Let the existing isFileTypeSupported handle this
  }

  switch (cCategory) {
    case FileTypeCategory.TEXT:
      // Text files are always supported
      return true;

    case FileTypeCategory.PDF:
      // PDFs are always supported (will be processed as text for non-vision models)
      return true;

    case FileTypeCategory.IMAGE:
      // Images require vision support
      return pCapabilities.hasVision;

    case FileTypeCategory.AUDIO:
      // Audio files require audio support
      return pCapabilities.hasAudio;

    case FileTypeCategory.VIDEO:
      // Video files require video support
      return pCapabilities.hasVideo;

    default:
      // Unknown categories - be conservative and allow
      return true;
  }
}

/**
 * Filter files based on model modalities and return supported/unsupported lists
 * @param pFiles - Array of files to filter
 * @param pCapabilities - The modality capabilities to check against
 * @returns Object with supportedFiles and unsupportedFiles arrays
 */
export function fFilterFilesByModalities(
  pFiles: File[],
  pCapabilities: ModalityCapabilities
): {
  supportedFiles: File[];
  unsupportedFiles: File[];
  modalityReasons: Record<string, string>;
} {
  const aSupportedFiles: File[] = [];
  const aUnsupportedFiles: File[] = [];
  const dModalityReasons: Record<string, string> = {};

  const { hasVision: cHasVision, hasAudio: cHasAudio, hasVideo: cHasVideo } = pCapabilities;

  for (const cFile of pFiles) {
    const cCategory = getFileTypeCategory(cFile.type);
    let vIsSupported = true;
    let vReason = '';

    switch (cCategory) {
      case FileTypeCategory.IMAGE:
        if (!cHasVision) {
          vIsSupported = false;
          vReason = 'Images require a vision-capable model';
        }
        break;

      case FileTypeCategory.AUDIO:
        if (!cHasAudio) {
          vIsSupported = false;
          vReason = 'Audio files require an audio-capable model';
        }
        break;

      case FileTypeCategory.VIDEO:
        if (!cHasVideo) {
          vIsSupported = false;
          vReason = 'Video files require a video-capable model';
        }
        break;

      case FileTypeCategory.TEXT:
      case FileTypeCategory.PDF:
        // Always supported
        break;

      default:
        // Unknown types keep the support decided above
        break;
    }

    if (vIsSupported) {
      aSupportedFiles.push(cFile);
    } else {
      aUnsupportedFiles.push(cFile);
      dModalityReasons[cFile.name] = vReason;
    }
  }

  return {
    supportedFiles: aSupportedFiles,
    unsupportedFiles: aUnsupportedFiles,
    modalityReasons: dModalityReasons
  };
}

/**
 * Generate a user-friendly error message for unsupported files
 * @param pUnsupportedFiles - Array of unsupported files
 * @param pModalityReasons - Reasons why files are unsupported
 * @param pCapabilities - The modality capabilities to check against
 * @returns Formatted error message
 */
export function fGenerateModalityErrorMessage(
  pUnsupportedFiles: File[],
  pModalityReasons: Record<string, string>,
  pCapabilities: ModalityCapabilities
): string {
  if (pUnsupportedFiles.length === 0) return '';

  const { hasVision: cHasVision, hasAudio: cHasAudio, hasVideo: cHasVideo } = pCapabilities;

  let vMessage = '';

  if (pUnsupportedFiles.length === 1) {
    const cFile = pUnsupportedFiles[0];
    const cReason = pModalityReasons[cFile.name];
    vMessage = fTranslate('text_b0f1d92b85c5', { p0: cFile.name, p1: cReason });
  } else {
    const cFileNames = pUnsupportedFiles.map((pF) => pF.name).join(', ');
    vMessage = fTranslate('text_dad82b66bf1d', { p0: cFileNames });
  }

  // Add helpful information about what is supported
  const aSupportedTypes: string[] = ['text files', 'PDFs'];
  if (cHasVision) aSupportedTypes.push('images');
  if (cHasAudio) aSupportedTypes.push('audio files');
  if (cHasVideo) aSupportedTypes.push('video files');

  vMessage += ` This model supports: ${aSupportedTypes.join(', ')}.`;

  return vMessage;
}
