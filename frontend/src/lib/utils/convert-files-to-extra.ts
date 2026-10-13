import { fTranslate } from '../i18n';
import { fConvertPDFToImage, fConvertPDFToText } from './pdf-processing';
import { fIsSvgMimeType, fSvgBase64UrlToPngDataURL } from './svg-to-png';
import { fIsWebpMimeType, fWebpBase64UrlToPngDataURL } from './webp-to-png';
import { FileTypeCategory, AttachmentType, SpecialFileType } from '$lib/enums';
import { cSettingsKeys } from '$lib/constants';
import { fConfig, cSettingsStore } from '$lib/stores/settings.svelte';
import { cModelsStore } from '$lib/stores/models.svelte';
import { getFileTypeCategory } from '$lib/utils';
import { fReadFileAsText, fIsLikelyTextFile } from './text-files';
import { toast } from 'svelte-sonner';
import type { FileProcessingResult, ChatUploadedFile, DatabaseMessageExtra } from '$lib/types';

function fReadFileAsBase64(pFile: File): Promise<string> {
  return new Promise((pResolve, pReject) => {
    const cReader = new FileReader();

    cReader.onload = () => {
      // Extract base64 data without the data URL prefix
      const cDataUrl = cReader.result as string;
      const cBase64 = cDataUrl.split(',')[1];
      pResolve(cBase64);
    };

    cReader.onerror = () => pReject(cReader.error);

    cReader.readAsDataURL(pFile);
  });
}

export async function fParseFilesToMessageExtras(
  pFiles: ChatUploadedFile[],
  pActiveModelId?: string
): Promise<FileProcessingResult> {
  const aExtras: DatabaseMessageExtra[] = [];
  const aEmptyFiles: string[] = [];

  for (const cFile of pFiles) {
    if (cFile.type === SpecialFileType.MCP_PROMPT && cFile.mcpPrompt) {
      aExtras.push({
        type: AttachmentType.MCP_PROMPT,
        name: cFile.name,
        size: cFile.size,
        serverName: cFile.mcpPrompt.serverName,
        promptName: cFile.mcpPrompt.promptName,
        content: cFile.textContent ?? '',
        arguments: cFile.mcpPrompt.arguments
      });

      continue;
    }

    if (getFileTypeCategory(cFile.type) === FileTypeCategory.IMAGE) {
      if (cFile.preview) {
        let vBase64Url = cFile.preview;

        if (fIsSvgMimeType(cFile.type)) {
          try {
            vBase64Url = await fSvgBase64UrlToPngDataURL(vBase64Url);
          } catch (vError) {
            console.error('Failed to convert SVG to PNG for database storage:', vError);
          }
        } else if (fIsWebpMimeType(cFile.type)) {
          try {
            vBase64Url = await fWebpBase64UrlToPngDataURL(vBase64Url);
          } catch (vError) {
            console.error('Failed to convert WebP to PNG for database storage:', vError);
          }
        }

        aExtras.push({
          type: AttachmentType.IMAGE,
          name: cFile.name,
          size: cFile.size,
          base64Url: vBase64Url
        });
      }
    } else if (getFileTypeCategory(cFile.type) === FileTypeCategory.AUDIO) {
      // Process audio files (MP3 and WAV)
      try {
        const cBase64Data = await fReadFileAsBase64(cFile.file);

        aExtras.push({
          type: AttachmentType.AUDIO,
          name: cFile.name,
          size: cFile.size,
          base64Data: cBase64Data,
          mimeType: cFile.type
        });
      } catch (vError) {
        console.error(`Failed to process audio file ${cFile.name}:`, vError);
      }
    } else if (getFileTypeCategory(cFile.type) === FileTypeCategory.VIDEO) {
      // Process video files (MP4, etc)
      try {
        const cBase64Data = await fReadFileAsBase64(cFile.file);

        aExtras.push({
          type: AttachmentType.VIDEO,
          name: cFile.name,
          size: cFile.size,
          base64Data: cBase64Data,
          mimeType: cFile.type
        });
      } catch (vError) {
        console.error(`Failed to process video file ${cFile.name}:`, vError);
      }
    } else if (getFileTypeCategory(cFile.type) === FileTypeCategory.PDF) {
      try {
        // Always get base64 data for preview functionality
        const cBase64Data = await fReadFileAsBase64(cFile.file);
        const cCurrentConfig = fConfig();
        // Use per-model vision check for router mode
        const cHasVisionSupport = pActiveModelId
          ? cModelsStore.mModelSupportsVision(pActiveModelId)
          : false;

        // Force PDF-to-text for non-vision models
        let vShouldProcessAsImages = Boolean(cCurrentConfig.pdfAsImage) && cHasVisionSupport;

        // If user had pdfAsImage enabled but model doesn't support vision, update setting and notify
        if (cCurrentConfig.pdfAsImage && !cHasVisionSupport) {
          console.log('Non-vision model detected: forcing PDF-to-text mode and updating settings');

          // Update the setting (kept in the server's settings document)
          cSettingsStore.mUpdateConfig(cSettingsKeys.PDF_AS_IMAGE, false);

          // Show toast notification to user
          toast.warning(fTranslate('text_cc2587a65a82'), {
            duration: 5000
          });

          vShouldProcessAsImages = false;
        }

        if (vShouldProcessAsImages) {
          // Process PDF as images (only for vision models)
          try {
            const cImages = await fConvertPDFToImage(cFile.file);

            // Show success toast for PDF image processing
            toast.success(
              `PDF "${cFile.name}" processed as ${cImages.length} images for vision model.`,
              {
                duration: 3000
              }
            );

            aExtras.push({
              type: AttachmentType.PDF,
              name: cFile.name,
              size: cFile.size,
              content: `PDF file with ${cImages.length} pages`,
              images: cImages,
              processedAsImages: true,
              base64Data: cBase64Data
            });
          } catch (vImageError) {
            console.warn(
              `Failed to process PDF ${cFile.name} as images, falling back to text:`,
              vImageError
            );

            // Fallback to text processing
            const cContent = await fConvertPDFToText(cFile.file);

            aExtras.push({
              type: AttachmentType.PDF,
              name: cFile.name,
              size: cFile.size,
              content: cContent,
              processedAsImages: false,
              base64Data: cBase64Data
            });
          }
        } else {
          // Process PDF as text (default or forced for non-vision models)
          const cContent = await fConvertPDFToText(cFile.file);

          // Show success toast for PDF text processing
          toast.success(`PDF "${cFile.name}" processed as text content.`, {
            duration: 3000
          });

          aExtras.push({
            type: AttachmentType.PDF,
            name: cFile.name,
            size: cFile.size,
            content: cContent,
            processedAsImages: false,
            base64Data: cBase64Data
          });
        }
      } catch (vError) {
        console.error(`Failed to process PDF file ${cFile.name}:`, vError);
      }
    } else {
      try {
        const cContent = await fReadFileAsText(cFile.file);

        // Check if file is empty
        if (cContent.trim() === '') {
          console.warn(`File ${cFile.name} is empty and will be skipped`);
          aEmptyFiles.push(cFile.name);
        } else if (fIsLikelyTextFile(cContent)) {
          aExtras.push({
            type: AttachmentType.TEXT,
            name: cFile.name,
            size: cFile.size,
            content: cContent
          });
        } else {
          console.warn(`File ${cFile.name} appears to be binary and will be skipped`);
        }
      } catch (vError) {
        console.error(`Failed to read file ${cFile.name}:`, vError);
      }
    }
  }

  return { extras: aExtras, emptyFiles: aEmptyFiles };
}
