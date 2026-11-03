import {
  cExifScanByteLimit,
  cJpegSoiMarker,
  cApp1Marker,
  cSosMarker,
  cExifSignature,
  cTiffLittleEndian,
  cTiffMagic,
  cExifOrientationTag,
  cIfdEntrySize
} from '$lib/constants/jpeg-exif';
import { MimeTypeImage } from '$lib/enums';

/**
 * Read the EXIF orientation tag from a JPEG base64 data URL
 *
 * Only a bounded prefix of the base64 payload is decoded, the APP1 segment
 * always sits near the start of the file.
 * @param pBase64UrlJpeg - The JPEG base64 data URL to inspect
 * @returns The orientation value (1 to 8), or 1 when absent or unreadable
 */
export function fGetJpegOrientationFromDataURL(pBase64UrlJpeg: string): number {
  try {
    const cPayloadStart = pBase64UrlJpeg.indexOf(',') + 1;

    if (cPayloadStart <= 0) {
      return 1;
    }

    // Keep the slice a multiple of 4 characters so atob accepts it
    const cCharLimit = Math.ceil(cExifScanByteLimit / 3) * 4;
    const cSlice = pBase64UrlJpeg.slice(cPayloadStart, cPayloadStart + cCharLimit);
    const cBinary = atob(cSlice.slice(0, cSlice.length - (cSlice.length % 4)));
    const cBytes = new Uint8Array(cBinary.length);

    for (let vI = 0; vI < cBinary.length; vI++) {
      cBytes[vI] = cBinary.charCodeAt(vI);
    }

    return fFindExifOrientation(new DataView(cBytes.buffer));
  } catch {
    return 1;
  }
}

/**
 * Walk the JPEG segments of a header buffer looking for the APP1 EXIF block
 * @param pView - DataView over the JPEG header bytes
 * @returns The orientation value (1 to 8), or 1 when absent or malformed
 */
function fFindExifOrientation(pView: DataView): number {
  if (pView.byteLength < 4 || pView.getUint16(0) !== cJpegSoiMarker) {
    return 1;
  }

  let vOffset = 2;

  while (vOffset + 4 <= pView.byteLength) {
    if (pView.getUint8(vOffset) !== 0xff) {
      return 1;
    }

    const cMarker = pView.getUint8(vOffset + 1);

    // Compressed image data starts here: no EXIF past this point
    if (cMarker === cSosMarker) {
      return 1;
    }

    const cSegmentLength = pView.getUint16(vOffset + 2);

    if (cMarker === cApp1Marker) {
      return fParseExifOrientation(pView, vOffset + 4, cSegmentLength);
    }

    vOffset += 2 + cSegmentLength;
  }

  return 1;
}

/**
 * Parse the orientation tag from an APP1 EXIF payload
 * @param pView - DataView over the JPEG header bytes
 * @param pStart - Offset of the APP1 payload, right after the segment length
 * @param pSegmentLength - Declared APP1 segment length
 * @returns The orientation value (1 to 8), or 1 when absent or malformed
 */
function fParseExifOrientation(pView: DataView, pStart: number, pSegmentLength: number): number {
  const cEnd = Math.min(pStart + pSegmentLength, pView.byteLength);

  // The payload opens with the "Exif\0\0" signature
  if (
    pStart + 6 > cEnd ||
    pView.getUint32(pStart) !== cExifSignature ||
    pView.getUint16(pStart + 4) !== 0
  ) {
    return 1;
  }

  const cTiff = pStart + 6;

  if (cTiff + 8 > cEnd) {
    return 1;
  }

  const cLittleEndian = pView.getUint16(cTiff) === cTiffLittleEndian;

  if (pView.getUint16(cTiff + 2, cLittleEndian) !== cTiffMagic) {
    return 1;
  }

  const cIfdOffset = pView.getUint32(cTiff + 4, cLittleEndian);

  if (cTiff + cIfdOffset + 2 > cEnd) {
    return 1;
  }

  const cEntryCount = pView.getUint16(cTiff + cIfdOffset, cLittleEndian);

  // Scan IFD0 entries for the orientation tag
  for (let vI = 0; vI < cEntryCount; vI++) {
    const cEntry = cTiff + cIfdOffset + 2 + vI * cIfdEntrySize;

    if (cEntry + cIfdEntrySize > cEnd) {
      return 1;
    }

    if (pView.getUint16(cEntry, cLittleEndian) === cExifOrientationTag) {
      const cOrientation = pView.getUint16(cEntry + 8, cLittleEndian);

      return cOrientation >= 1 && cOrientation <= 8 ? cOrientation : 1;
    }
  }

  return 1;
}

/**
 * Check if a MIME type represents a JPEG
 * @param pMimeType - The MIME type to check
 * @returns True if the MIME type is a JPEG variant
 */
export function fIsJpegMimeType(pMimeType: string): boolean {
  return pMimeType === MimeTypeImage.JPEG || pMimeType === MimeTypeImage.JPG;
}
