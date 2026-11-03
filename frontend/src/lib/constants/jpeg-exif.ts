/**
 * JPEG and EXIF binary format constants for orientation parsing.
 */

/** Bytes of file prefix to scan, the APP1 EXIF segment sits near the start */
export const cExifScanByteLimit = 128 * 1024;

/** JPEG start of image marker */
export const cJpegSoiMarker = 0xffd8;

/** APP1 segment marker byte, carries the EXIF payload */
export const cApp1Marker = 0xe1;

/** Start of scan marker byte, compressed data begins and no EXIF follows */
export const cSosMarker = 0xda;

/** "Exif" signature opening the APP1 payload, big endian uint32 */
export const cExifSignature = 0x45786966;

/** TIFF byte order mark for little endian ("II") */
export const cTiffLittleEndian = 0x4949;

/** TIFF magic number following the byte order mark */
export const cTiffMagic = 42;

/** EXIF tag id holding the orientation value */
export const cExifOrientationTag = 0x0112;

/** Size in bytes of one IFD directory entry */
export const cIfdEntrySize = 12;
