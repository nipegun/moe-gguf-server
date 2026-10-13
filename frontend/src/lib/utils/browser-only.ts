/**
 * Browser-only utility exports
 *
 * These utilities require browser APIs (DOM, Canvas, MediaRecorder, etc.)
 * and cannot be imported during SSR. Import from '$lib/utils/browser-only'
 * only in client-side code or components that are not server-rendered.
 */

// Audio utilities (MediaRecorder API)
export {
  AudioRecorder,
  fConvertToWav as convertToWav,
  fCreateAudioFile as createAudioFile,
  fIsAudioRecordingSupported as isAudioRecordingSupported
} from './audio-recording';

// PDF processing utilities (pdfjs-dist with DOMMatrix)
export {
  fConvertPDFToText as convertPDFToText,
  fConvertPDFToImage as convertPDFToImage,
  fIsPdfFile as isPdfFileFromFile,
  fIsApplicationMimeType as isApplicationMimeType
} from './pdf-processing';

// File conversion utilities (depends on pdf-processing)
export { fParseFilesToMessageExtras as parseFilesToMessageExtras } from './convert-files-to-extra';

// File upload processing utilities (depends on pdf-processing, svg-to-png, webp-to-png)
export { fProcessFilesToChatUploaded as processFilesToChatUploaded } from './process-uploaded-files';

// SVG utilities (Canvas/Image API)
export {
  fSvgBase64UrlToPngDataURL as svgBase64UrlToPngDataURL,
  fIsSvgFile as isSvgFile,
  fIsSvgMimeType as isSvgMimeType
} from './svg-to-png';

// WebP utilities (Canvas/Image API)
export {
  fWebpBase64UrlToPngDataURL as webpBase64UrlToPngDataURL,
  fIsWebpFile as isWebpFile,
  fIsWebpMimeType as isWebpMimeType
} from './webp-to-png';
