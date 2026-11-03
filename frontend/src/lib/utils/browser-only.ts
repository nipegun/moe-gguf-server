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
  fConvertToWav,
  fCreateAudioFile,
  fIsAudioRecordingSupported
} from './audio-recording';

// PDF processing utilities (pdfjs-dist with DOMMatrix)
export {
  fConvertPDFToText,
  fConvertPDFToImage,
  fIsPdfFileFromFile,
  fIsApplicationMimeType
} from './pdf-processing';

// File conversion utilities (depends on pdf-processing)
export { fParseFilesToMessageExtras } from './convert-files-to-extra';

// File upload processing utilities (depends on pdf-processing, svg-to-png, webp-to-png)
export { fProcessFilesToChatUploaded } from './process-uploaded-files';

// SVG utilities (Canvas/Image API)
export { fSvgBase64UrlToPngDataURL, fIsSvgFile, fIsSvgMimeType } from './svg-to-png';

// WebP utilities (Canvas/Image API)
export { fWebpBase64UrlToPngDataURL, fIsWebpFile, fIsWebpMimeType } from './webp-to-png';
