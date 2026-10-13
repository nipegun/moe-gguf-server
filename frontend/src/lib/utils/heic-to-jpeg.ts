import { MimeTypeImage } from '$lib/enums';
import { cHeicJpegQuality } from '$lib/constants/image-size';

// heic requires a relatively large decoder, in order to reduce primary bundle size
// we lazily load this decoder from a CDN when needed, and cache it for future conversions
const cHeicToCdnUrl = 'https://cdn.jsdelivr.net/npm/heic-to@1.5.2/dist/heic-to.js';

interface HeicToModule {
  heicTo(pArgs: { blob: Blob; type: string; quality?: number }): Promise<Blob>;
}

let vModulePromise: Promise<HeicToModule> | null = null;

/**
 * Lazily load the heic-to decoder from the CDN and cache it
 * @returns Promise resolving to the heic-to module
 */
function fGetHeicTo(): Promise<HeicToModule> {
  if (!vModulePromise) {
    vModulePromise = import(/* @vite-ignore */ cHeicToCdnUrl) as Promise<HeicToModule>;
  }

  return vModulePromise;
}

/**
 * Convert a HEIC/HEIF file to a compressed JPEG data URL
 * @param pFile - The HEIC/HEIF file to convert
 * @returns Promise resolving to JPEG data URL
 */
export async function fHeicFileToJpegDataURL(pFile: File | Blob): Promise<string> {
  const { heicTo: cHeicTo } = await fGetHeicTo();
  const cJpegBlob = await cHeicTo({
    blob: pFile,
    type: MimeTypeImage.JPEG,
    quality: cHeicJpegQuality
  });

  return new Promise((pResolve, pReject) => {
    const cReader = new FileReader();
    cReader.onload = () => pResolve(cReader.result as string);
    cReader.onerror = () => pReject(cReader.error);
    cReader.readAsDataURL(cJpegBlob);
  });
}

/**
 * Check if a MIME type represents a HEIC/HEIF image
 * @param pMimeType - The MIME type to check
 * @returns True if the MIME type is image/heic or image/heif
 */
export function fIsHeicMimeType(pMimeType: string): boolean {
  const cNormalized = pMimeType.trim().toLowerCase();

  return cNormalized === MimeTypeImage.HEIC || cNormalized === MimeTypeImage.HEIF;
}
