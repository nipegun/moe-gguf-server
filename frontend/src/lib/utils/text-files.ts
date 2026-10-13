import { fTranslate } from '../i18n';
/**
 * Text file processing utilities
 * Handles text file detection, reading, and validation
 */

import { dDefaultBinaryDetectionOptions } from '$lib/constants';
import type { BinaryDetectionOptions } from '$lib/types';
import { FileExtensionText } from '$lib/enums';

/**
 * Check if a filename indicates a text file based on its extension
 * @param pFilename - The filename to check
 * @returns True if the filename has a recognized text file extension
 */
export function fIsTextFileByName(pFilename: string): boolean {
  const cTextExtensions = Object.values(FileExtensionText);

  return cTextExtensions.some((pExt: FileExtensionText) => pFilename.toLowerCase().endsWith(pExt));
}

/**
 * Read a file's content as text
 * @param pFile - The file to read
 * @returns Promise resolving to the file's text content
 */
export async function fReadFileAsText(pFile: File): Promise<string> {
  return new Promise((pResolve, pReject) => {
    const cReader = new FileReader();

    cReader.onload = (pEvent) => {
      if (pEvent.target?.result !== null && pEvent.target?.result !== undefined) {
        pResolve(pEvent.target.result as string);
      } else {
        pReject(new Error(fTranslate('text_ea7d976ce045')));
      }
    };

    cReader.onerror = () => pReject(new Error(fTranslate('text_9306f1ec32c9')));

    cReader.readAsText(pFile);
  });
}

/**
 * Heuristic check to determine if content is likely from a text file
 * Detects binary files by counting suspicious characters and null bytes
 * @param pContent - The file content to analyze
 * @param pOptions - Optional configuration for detection parameters
 * @returns True if the content appears to be text-based
 */
export function fIsLikelyTextFile(
  pContent: string,
  pOptions: Partial<BinaryDetectionOptions> = {}
): boolean {
  if (!pContent) return true;

  const dConfig = { ...dDefaultBinaryDetectionOptions, ...pOptions };
  const cSample = pContent.substring(0, dConfig.prefixLength);

  let vNullCount = 0;
  let vSuspiciousControlCount = 0;

  for (let vI = 0; vI < cSample.length; vI++) {
    const cCharCode = cSample.charCodeAt(vI);

    // Count null bytes - these are strong indicators of binary files
    if (cCharCode === 0) {
      vNullCount++;

      continue;
    }

    // Count suspicious control characters
    // Allow common whitespace characters: tab (9), newline (10), carriage return (13)
    if (cCharCode < 32 && cCharCode !== 9 && cCharCode !== 10 && cCharCode !== 13) {
      // Count most suspicious control characters
      if (cCharCode < 8 || (cCharCode > 13 && cCharCode < 27)) {
        vSuspiciousControlCount++;
      }
    }

    // Count replacement characters (indicates encoding issues)
    if (cCharCode === 0xfffd) {
      vSuspiciousControlCount++;
    }
  }

  // Reject if too many null bytes
  if (vNullCount > dConfig.maxAbsoluteNullBytes) return false;

  // Reject if too many suspicious characters
  if (vSuspiciousControlCount / cSample.length > dConfig.suspiciousCharThresholdRatio) return false;

  return true;
}
