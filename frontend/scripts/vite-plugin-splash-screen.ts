import { readdirSync, readFileSync, writeFileSync, existsSync } from 'node:fs';
import { resolve } from 'path';
import type { Plugin } from 'vite';
import { cTab, cNewline } from '../src/lib/constants/code';
import { cAppleDevices, cBuildConfig, cRegexPatterns, cSplashLink } from '../src/lib/constants/pwa';
import type { SplashDimensions } from '../src/lib/types';
import { SplashOrientation } from '../src/lib/enums/splash.enums';

let vProcessed = false;

const cOutputDir = process.env.MGS_UI_OUT_DIR ?? cBuildConfig.OUTPUT_DIR;

/**
 * Generate iOS splash screen <link> tags from generated apple-splash-*.png files.
 * Returns an array of HTML link strings to be injected into the page head.
 */
export function fGenerateSplashScreenLinks(pOutDir: string): string[] {
  const aFiles = readdirSync(pOutDir).filter((pFile) => pFile.match(cRegexPatterns.SPLASH_FILE));
  if (aFiles.length === 0) return [];

  const dDimensions = new Map<string, SplashDimensions>();
  for (const [cDims, cSpec] of Object.entries(cAppleDevices)) {
    const [cW, cH] = cDims.split('x').map(Number);
    // logical-point dimensions
    dDimensions.set(`${cW}x${cH}`, { deviceW: cSpec.width, deviceH: cSpec.height, dpr: cSpec.dpr });
    dDimensions.set(`${cH}x${cW}`, { deviceW: cSpec.width, deviceH: cSpec.height, dpr: cSpec.dpr });
    // pixel dimensions (used by actual generated splash files)
    dDimensions.set(`${cW * cSpec.dpr}x${cH * cSpec.dpr}`, {
      deviceW: cSpec.width,
      deviceH: cSpec.height,
      dpr: cSpec.dpr
    });
    dDimensions.set(`${cH * cSpec.dpr}x${cW * cSpec.dpr}`, {
      deviceW: cSpec.width,
      deviceH: cSpec.height,
      dpr: cSpec.dpr
    });
  }

  const aLightLinks: string[] = [];
  const aDarkLinks: string[] = [];

  for (const cFile of aFiles) {
    const cMatch = cFile.match(cRegexPatterns.SPLASH_FILE);
    if (!cMatch) continue;
    const cOrientation = cMatch[1] as SplashOrientation;
    const cIsDark = !!cMatch[2];
    const cPixelW = parseInt(cMatch[3]);
    const cPixelH = parseInt(cMatch[4]);

    const cKey = `${cPixelW}x${cPixelH}`;
    const cSpec = dDimensions.get(cKey);
    if (!cSpec) {
      console.warn(`Unknown splash screen dimensions: ${cKey} (${cFile})`);
      continue;
    }

    const { deviceW: cDeviceW, deviceH: cDeviceH, dpr: cDpr } = cSpec;
    const cMedia = `screen and (device-width: ${cDeviceW}px) and (device-height: ${cDeviceH}px) and (-webkit-device-pixel-ratio: ${cDpr}) and (orientation: ${cOrientation})`;
    const cHref = `./${cFile}`;

    if (cIsDark) {
      aDarkLinks.push(
        `${cSplashLink.HTML} media="${cMedia}${cSplashLink.DARK_MEDIA_SUFFIX}" href="${cHref}">`
      );
    } else {
      aLightLinks.push(`${cSplashLink.HTML} media="${cMedia}" href="${cHref}">`);
    }
  }

  return [...aLightLinks, ...aDarkLinks];
}

export function fSplashScreenPlugin(): Plugin {
  return {
    name: 'mgs:splash-screen',
    apply: 'build',
    closeBundle() {
      setTimeout(() => {
        try {
          if (vProcessed) return;
          vProcessed = true;

          const cOutDir = resolve(cOutputDir);
          const cIndexPath = resolve(cOutDir, 'index.html');
          if (!existsSync(cIndexPath)) return;

          let vContent = readFileSync(cIndexPath, 'utf-8');

          // Inject iOS splash screen <link> tags into <head>.
          // The @vite-pwa/assets-generator generates apple-splash-*.png files;
          // this scans them and creates the <link> tags SvelteKit needs.
          const aSplashLinks = fGenerateSplashScreenLinks(cOutDir);
          if (aSplashLinks.length > 0) {
            console.log(`Generated ${aSplashLinks.length} apple-splash link tags`);
            const cSplashHtml = aSplashLinks.map((pLink) => cTab + cTab + pLink).join(cNewline);
            vContent = vContent.replace(
              cRegexPatterns.HEAD_CLOSE,
              cSplashHtml + cNewline + cTab + cTab + '</head>'
            );
          }

          // Remove trailing \r from Windows line endings
          vContent = vContent.replace(/\r/g, '');
          vContent = cBuildConfig.GUIDE_COMMENT + cNewline + vContent;

          writeFileSync(cIndexPath, vContent, 'utf-8');
          console.log('Updated index.html');
        } catch (vError) {
          console.error('Failed to process build output:', vError);
        }
      }, 100);
    }
  };
}
