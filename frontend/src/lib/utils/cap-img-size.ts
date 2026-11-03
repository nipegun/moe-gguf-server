import { fTranslate } from '../i18n';
import { cMegapixelsToPixels } from '$lib/constants/image-size';
import { cBase64ImageUriRegex } from '$lib/constants/uri-template';
import { fGetJpegOrientationFromDataURL, fIsJpegMimeType } from './jpeg-orientation';
import { MimeTypeImage } from '$lib/enums';

/**
 * Converts an Image base64 data URL to another Image data URL with capped dimensions to reduce file size.
 *
 * For JPEGs the EXIF orientation is baked into the pixels in the same canvas
 * pass, the browser applies the rotation when decoding so naturalWidth and
 * naturalHeight already describe the upright image. Backends decoding with
 * stb_image ignore EXIF. Images that need
 * neither capping nor rotation pass through untouched, so at most one
 * re-encode ever happens.
 * @param pBase64UrlImage - The Image base64 data URL to convert
 * @param pMaxMegapixels - The maximum image size in megapixels for the output Image, 0 disables capping
 * @returns Promise resolving to Image data URL
 */
export function fCapImageDataURLSize(
  pBase64UrlImage: string,
  pMaxMegapixels: number
): Promise<string> {
  return new Promise((pResolve, pReject) => {
    try {
      const cMimeMatch = pBase64UrlImage.match(cBase64ImageUriRegex);

      if (!cMimeMatch) {
        return pReject(new Error(fTranslate('text_e96d8cb99375')));
      }

      const cMimeType = cMimeMatch[1] as MimeTypeImage;

      if (!Object.values(MimeTypeImage).includes(cMimeType)) {
        return pReject(new Error(fTranslate('text_0c9e37062561', { p0: cMimeType })));
      }

      const cOrientation = fIsJpegMimeType(cMimeType)
        ? fGetJpegOrientationFromDataURL(pBase64UrlImage)
        : 1;

      const cImg = new Image();

      cImg.onload = () => {
        try {
          const cCanvas = document.createElement('canvas');
          const cCtx = cCanvas.getContext('2d');

          if (!cCtx) {
            throw new Error(fTranslate('text_1eb30178a933'));
          }

          const cTargetWidth = cImg.naturalWidth;
          const cTargetHeight = cImg.naturalHeight;
          const cTotalPixels = cTargetWidth * cTargetHeight;
          const cMaxPixels = Math.floor(pMaxMegapixels * cMegapixelsToPixels);

          if (cMaxPixels > 0 && cTotalPixels > cMaxPixels) {
            const cScaleFactor = Math.sqrt(cMaxPixels / cTotalPixels);
            cCanvas.width = Math.floor(cTargetWidth * cScaleFactor);
            cCanvas.height = Math.floor(cTargetHeight * cScaleFactor);
          } else if (cOrientation > 1) {
            // No capping needed but the pixels still need the rotation baked in
            cCanvas.width = cTargetWidth;
            cCanvas.height = cTargetHeight;
          } else {
            return pResolve(pBase64UrlImage);
          }

          cCtx.drawImage(cImg, 0, 0, cCanvas.width, cCanvas.height);
          pResolve(cCanvas.toDataURL(cMimeType));
        } catch (vErr) {
          pReject(vErr instanceof Error ? vErr : new Error(String(vErr)));
        }
      };

      cImg.onerror = () => {
        pReject(new Error(fTranslate('text_3c7a3f00cb84')));
      };

      cImg.src = pBase64UrlImage;
    } catch (vError) {
      const cMessage = vError instanceof Error ? vError.message : String(vError);
      const cErrorMessage = fTranslate('text_37d6f79ee76d', { p0: cMessage });
      console.error(cErrorMessage, vError);
      pReject(new Error(cErrorMessage));
    }
  });
}
