/**
 * Converts a rem CSS value to pixels based on the document root font size.
 */
export function fRemToPx(pRem: string): number {
  const cVal = parseFloat(pRem);
  const cFontSize = parseFloat(getComputedStyle(document.documentElement).fontSize);

  return cVal * cFontSize;
}
