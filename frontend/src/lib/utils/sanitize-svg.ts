import DOMPurify from 'dompurify';
import { cSvgMaxBytes, dSvgSanitizeConfig, cSvgTagPrefix } from '$lib/constants';

/**
 * Sanitizes a raw svg string for safe inline rendering.
 * Returns the cleaned svg markup, or an empty string when the input is not a
 * usable svg, exceeds the size ceiling, or sanitizes to nothing. An empty
 * return tells the caller to keep the raw code block instead of rendering.
 */
export function fSanitizeSvg(pSource: string): string {
  const cTrimmed = pSource.trim();

  if (!cTrimmed || cTrimmed.length > cSvgMaxBytes) return '';

  if (!cTrimmed.startsWith(cSvgTagPrefix)) return '';

  const cClean = DOMPurify.sanitize(cTrimmed, dSvgSanitizeConfig) as unknown as string;

  if (!cClean || !cClean.includes(cSvgTagPrefix)) return '';

  return cClean;
}
