import { fTranslate } from '../i18n';
import { fIsSvgMimeType, fSvgBase64UrlToPngDataURL } from './svg-to-png';
import { fIsWebpMimeType, fWebpBase64UrlToPngDataURL } from './webp-to-png';
import { fHeicFileToJpegDataURL, fIsHeicMimeType } from './heic-to-jpeg';
import { FileTypeCategory } from '$lib/enums';
import { cSettingsKeys } from '$lib/constants';
import { cModelsStore } from '$lib/stores/models.svelte';
import { cSettingsStore } from '$lib/stores/settings.svelte';
import { toast } from 'svelte-sonner';
import { getFileTypeCategory } from '$lib/utils';
import { fConvertPDFToText } from './pdf-processing';

/**
 * Read a file as a data URL (base64 encoded)
 * @param pFile - The file to read
 * @returns Promise resolving to the data URL string
 */
function fReadFileAsDataURL(pFile: File): Promise<string> {
  return new Promise((pResolve, pReject) => {
    const cReader = new FileReader();
    cReader.onload = () => pResolve(cReader.result as string);
    cReader.onerror = () => pReject(cReader.error);
    cReader.readAsDataURL(pFile);
  });
}

/**
 * Read a file as UTF-8 text
 * @param pFile - The file to read
 * @returns Promise resolving to the text content
 */
function fReadFileAsUTF8(pFile: File): Promise<string> {
  return new Promise((pResolve, pReject) => {
    const cReader = new FileReader();
    cReader.onload = () => pResolve(cReader.result as string);
    cReader.onerror = () => pReject(cReader.error);
    cReader.readAsText(pFile);
  });
}

/**
 * Process uploaded files into ChatUploadedFile format with previews and content
 *
 * This function processes various file types and generates appropriate previews:
 * - Images: Base64 data URLs with format normalization (SVG/WebP → PNG)
 * - Text files: UTF-8 content extraction
 * - PDFs: Metadata only (processed later in conversion pipeline)
 * - Audio: Base64 data URLs for preview
 *
 * @param pFiles - Array of File objects to process
 * @returns Promise resolving to array of ChatUploadedFile objects
 */
export async function fProcessFilesToChatUploaded(
  pFiles: File[],
  pActiveModelId?: string
): Promise<ChatUploadedFile[]> {
  const aResults: ChatUploadedFile[] = [];

  for (const cFile of pFiles) {
    const cId = Date.now().toString() + Math.random().toString(36).substr(2, 9);
    const dBase: ChatUploadedFile = {
      id: cId,
      name: cFile.name,
      size: cFile.size,
      type: cFile.type,
      file: cFile
    };

    try {
      if (getFileTypeCategory(cFile.type) === FileTypeCategory.IMAGE) {
        let vPreview = await fReadFileAsDataURL(cFile);

        // Normalize SVG and WebP to PNG, and HEIC to compressed JPEG, in previews
        if (fIsSvgMimeType(cFile.type)) {
          try {
            vPreview = await fSvgBase64UrlToPngDataURL(vPreview);
          } catch (vErr) {
            console.error('Failed to convert SVG to PNG:', vErr);
          }
        } else if (fIsWebpMimeType(cFile.type)) {
          try {
            vPreview = await fWebpBase64UrlToPngDataURL(vPreview);
          } catch (vErr) {
            console.error('Failed to convert WebP to PNG:', vErr);
          }
        } else if (fIsHeicMimeType(cFile.type)) {
          try {
            vPreview = await fHeicFileToJpegDataURL(cFile);
          } catch (vErr) {
            console.error('Failed to convert HEIC to PNG:', vErr);
            continue;
          }
        }

        aResults.push({ ...dBase, preview: vPreview });
      } else if (getFileTypeCategory(cFile.type) === FileTypeCategory.PDF) {
        // Extract text content from PDF for preview
        try {
          const cTextContent = await fConvertPDFToText(cFile);
          aResults.push({ ...dBase, textContent: cTextContent });
        } catch (vErr) {
          console.warn('Failed to extract text from PDF, adding without content:', vErr);
          aResults.push(dBase);
        }

        // Show suggestion toast if vision model is available but PDF as image is disabled
        const cHasVisionSupport = pActiveModelId
          ? cModelsStore.mModelSupportsVision(pActiveModelId)
          : false;
        const cCurrentConfig = cSettingsStore.vConfig;
        if (cHasVisionSupport && !cCurrentConfig.pdfAsImage) {
          toast.info(`You can enable parsing PDF as images with vision models.`, {
            duration: 8000,
            action: {
              label: fTranslate('text_89b167823e87'),
              onClick: () => {
                cSettingsStore.mUpdateConfig(cSettingsKeys.PDF_AS_IMAGE, true);
                toast.success(fTranslate('text_28a5e30e1a90'), {
                  duration: 3000
                });
              }
            }
          });
        }
      } else if (getFileTypeCategory(cFile.type) === FileTypeCategory.AUDIO) {
        // Generate preview URL for audio files
        const cPreview = await fReadFileAsDataURL(cFile);
        aResults.push({ ...dBase, preview: cPreview });
      } else if (getFileTypeCategory(cFile.type) === FileTypeCategory.VIDEO) {
        // Generate preview URL for video files
        const cPreview = await fReadFileAsDataURL(cFile);
        aResults.push({ ...dBase, preview: cPreview });
      } else {
        // Fallback: treat unknown files as text
        try {
          const cTextContent = await fReadFileAsUTF8(cFile);
          aResults.push({ ...dBase, textContent: cTextContent });
        } catch (vErr) {
          console.warn('Failed to read file as text, adding without content:', vErr);
          aResults.push(dBase);
        }
      }
    } catch (vError) {
      console.error('Error processing file', cFile.name, vError);
      aResults.push(dBase);
    }
  }

  return aResults;
}
