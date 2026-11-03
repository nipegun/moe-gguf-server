import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const cHere = dirname(fileURLToPath(import.meta.url));
const cProjectRoot = resolve(cHere, '..');

const cDefaultLogo = resolve(cProjectRoot, 'src/lib/assets/logo.svg');
const cDefaultOutDir = resolve(cProjectRoot, 'static');
const cDefaultOutLight = resolve(cDefaultOutDir, 'favicon.svg');
const cDefaultOutDark = resolve(cDefaultOutDir, 'favicon-dark.svg');

const cCurrentColor = 'currentColor';

export interface ColorizedFavicon {
  light: string;
  dark: string;
}

export interface WriteThemeFaviconsOptions {
  sourcePath?: string;
  lightOutPath?: string;
  darkOutPath?: string;
  /**
   * Fraction of the icon (0..1) to leave as an even margin on each side.
   * Applied by wrapping the inner content in a `<g transform="...">` so the
   * source `src/lib/assets/logo.svg` is not modified. Pass 0 to disable.
   */
  padding?: number;
}

/**
 * Replace every `currentColor` occurrence in the SVG with the given color.
 * Pure: no filesystem access, so it is straightforward to unit-test.
 */
export function fColorizeFaviconSvg(
  pSvg: string,
  pLightColor: string,
  pDarkColor: string
): ColorizedFavicon {
  return {
    light: pSvg.replaceAll(cCurrentColor, pLightColor),
    dark: pSvg.replaceAll(cCurrentColor, pDarkColor)
  };
}

/**
 * Shrink the inner SVG content uniformly and re-center it so `padding` (a
 * 0..1 fraction) is reserved as equal margin on each side. Returns the input
 * unchanged for non-positive padding, missing/invalid `viewBox`, or unexpected
 * markup so the caller always gets a renderable SVG.
 */
export function fPadFaviconSvg(pSvg: string, pPadding: number): string {
  if (!(pPadding > 0) || pPadding >= 1) return pSvg;

  const cViewBoxMatch = pSvg.match(/viewBox\s*=\s*["']([^"']+)["']/i);
  if (!cViewBoxMatch) return pSvg;

  const aParts = cViewBoxMatch[1]
    .trim()
    .split(/[\s,]+/)
    .map(Number);
  if (aParts.length !== 4 || aParts.some((pN) => !Number.isFinite(pN))) return pSvg;

  const [, , cWidth, cHeight] = aParts;
  if (cWidth <= 0 || cHeight <= 0) return pSvg;

  const cScale = 1 - pPadding;
  const cTranslateX = (pPadding * cWidth) / 2;
  const cTranslateY = (pPadding * cHeight) / 2;

  const cOpenTagStart = pSvg.search(/<svg\b/i);
  if (cOpenTagStart === -1) return pSvg;
  const cOpenTagEnd = pSvg.indexOf('>', cOpenTagStart);
  if (cOpenTagEnd === -1) return pSvg;
  const cCloseStart = pSvg.lastIndexOf('</svg');
  if (cCloseStart === -1 || cCloseStart <= cOpenTagEnd) return pSvg;

  const cOpenTag = pSvg.slice(0, cOpenTagEnd + 1);
  const cInner = pSvg.slice(cOpenTagEnd + 1, cCloseStart);
  const cCloseTag = pSvg.slice(cCloseStart);

  const cGroup = `<g transform="translate(${cTranslateX} ${cTranslateY}) scale(${cScale})">`;
  return `${cOpenTag}${cGroup}${cInner}</g>${cCloseTag}`;
}

/**
 * Read `src/lib/assets/logo.svg`, colorize it for both themes, and write
 * the results to the static directory so the PWA asset generator can consume
 * them. Paths can be overridden for tests.
 */
export function fWriteThemeFavicons(
  pLightColor: string,
  pDarkColor: string,
  {
    sourcePath: pSourcePath = cDefaultLogo,
    lightOutPath: pLightOutPath = cDefaultOutLight,
    darkOutPath: pDarkOutPath = cDefaultOutDark,
    padding: pPadding = 0
  }: WriteThemeFaviconsOptions = {}
): void {
  const cSource = readFileSync(pSourcePath, 'utf-8');
  const { light: cLight, dark: cDark } = fColorizeFaviconSvg(cSource, pLightColor, pDarkColor);
  mkdirSync(dirname(pLightOutPath), { recursive: true });
  writeFileSync(pLightOutPath, fPadFaviconSvg(cLight, pPadding));
  writeFileSync(pDarkOutPath, fPadFaviconSvg(cDark, pPadding));
}
