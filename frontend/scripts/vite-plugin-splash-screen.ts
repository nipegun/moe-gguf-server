import { readdirSync, readFileSync, writeFileSync, existsSync } from 'node:fs';
import { resolve } from 'path';
import type { Plugin } from 'vite';
import { cTab, cNewline } from '../src/lib/constants/code';
import { cAppleDevices, cBuildConfig, cRegexPatterns, cSplashLink } from '../src/lib/constants/pwa';
import type { SplashDimensions } from '../src/lib/types';
import { SplashOrientation } from '../src/lib/enums/splash.enums';

let processed = false;

const OUTPUT_DIR = process.env.MGS_UI_OUT_DIR ?? cBuildConfig.OUTPUT_DIR;

/**
 * Generate iOS splash screen <link> tags from generated apple-splash-*.png files.
 * Returns an array of HTML link strings to be injected into the page head.
 */
export function generateSplashScreenLinks(outDir: string): string[] {
  const files = readdirSync(outDir).filter((f) => f.match(cRegexPatterns.SPLASH_FILE));
  if (files.length === 0) return [];

  const dimMap = new Map<string, SplashDimensions>();
  for (const [dims, spec] of Object.entries(cAppleDevices)) {
    const [w, h] = dims.split('x').map(Number);
    // logical-point dimensions
    dimMap.set(`${w}x${h}`, { deviceW: spec.width, deviceH: spec.height, dpr: spec.dpr });
    dimMap.set(`${h}x${w}`, { deviceW: spec.width, deviceH: spec.height, dpr: spec.dpr });
    // pixel dimensions (used by actual generated splash files)
    dimMap.set(`${w * spec.dpr}x${h * spec.dpr}`, {
      deviceW: spec.width,
      deviceH: spec.height,
      dpr: spec.dpr
    });
    dimMap.set(`${h * spec.dpr}x${w * spec.dpr}`, {
      deviceW: spec.width,
      deviceH: spec.height,
      dpr: spec.dpr
    });
  }

  const lightLinks: string[] = [];
  const darkLinks: string[] = [];

  for (const file of files) {
    const match = file.match(cRegexPatterns.SPLASH_FILE);
    if (!match) continue;
    const orientation = match[1] as SplashOrientation;
    const isDark = !!match[2];
    const pixelW = parseInt(match[3]);
    const pixelH = parseInt(match[4]);

    const key = `${pixelW}x${pixelH}`;
    const spec = dimMap.get(key);
    if (!spec) {
      console.warn(`Unknown splash screen dimensions: ${key} (${file})`);
      continue;
    }

    const { deviceW, deviceH, dpr } = spec;
    const media = `screen and (device-width: ${deviceW}px) and (device-height: ${deviceH}px) and (-webkit-device-pixel-ratio: ${dpr}) and (orientation: ${orientation})`;
    const href = `./${file}`;

    if (isDark) {
      darkLinks.push(
        `${cSplashLink.HTML} media="${media}${cSplashLink.DARK_MEDIA_SUFFIX}" href="${href}">`
      );
    } else {
      lightLinks.push(`${cSplashLink.HTML} media="${media}" href="${href}">`);
    }
  }

  return [...lightLinks, ...darkLinks];
}

export function splashScreenPlugin(): Plugin {
  return {
    name: 'mgs:splash-screen',
    apply: 'build',
    closeBundle() {
      setTimeout(() => {
        try {
          if (processed) return;
          processed = true;

          const outDir = resolve(OUTPUT_DIR);
          const indexPath = resolve(outDir, 'index.html');
          if (!existsSync(indexPath)) return;

          let content = readFileSync(indexPath, 'utf-8');

          // Inject iOS splash screen <link> tags into <head>.
          // The @vite-pwa/assets-generator generates apple-splash-*.png files;
          // this scans them and creates the <link> tags SvelteKit needs.
          const splashLinks = generateSplashScreenLinks(outDir);
          if (splashLinks.length > 0) {
            console.log(`Generated ${splashLinks.length} apple-splash link tags`);
            const splashHtml = splashLinks.map((l) => cTab + cTab + l).join(cNewline);
            content = content.replace(
              cRegexPatterns.HEAD_CLOSE,
              splashHtml + cNewline + cTab + cTab + '</head>'
            );
          }

          // Remove trailing \r from Windows line endings
          content = content.replace(/\r/g, '');
          content = cBuildConfig.GUIDE_COMMENT + cNewline + content;

          writeFileSync(indexPath, content, 'utf-8');
          console.log('Updated index.html');
        } catch (error) {
          console.error('Failed to process build output:', error);
        }
      }, 100);
    }
  };
}
