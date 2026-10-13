import {
  combinePresetAndAppleSplashScreens,
  defineConfig,
  minimal2023Preset
} from '@vite-pwa/assets-generator/config';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import {
  cThemeColors,
  cPwaGeneratorDevices,
  cPwaAssetGenerator,
  cFaviconColors
} from './src/lib/constants/pwa';
import { SplashOrientation } from './src/lib/enums/splash.enums';
import { writeThemeFavicons } from './scripts/favicon-colorize';

writeThemeFavicons(cFaviconColors.LIGHT, cFaviconColors.DARK, {
  padding: cPwaAssetGenerator.FAVICON_PADDING
});

export default defineConfig({
  headLinkOptions: {
    preset: cPwaAssetGenerator.LINK_PRESET
  },
  preset: combinePresetAndAppleSplashScreens(
    {
      ...minimal2023Preset,
      // tiny margin so favicon.ico / pwa-*.png breathe inside the canvas
      transparent: {
        ...minimal2023Preset.transparent,
        padding: cPwaAssetGenerator.FAVICON_PADDING
      }
    },
    {
      padding: cPwaAssetGenerator.SPLASH_PADDING,
      resizeOptions: {
        background: cThemeColors.BACKGROUND_LIGHT,
        fit: cPwaAssetGenerator.FIT_MODE
      },
      darkResizeOptions: {
        background: cThemeColors.BACKGROUND_DARK,
        fit: cPwaAssetGenerator.FIT_MODE
      },
      darkImageResolver: async (imageName: string) => {
        if (imageName.endsWith('favicon.svg')) {
          return readFileSync(resolve('static/favicon-dark.svg'));
        }
      },
      linkMediaOptions: {
        log: true,
        addMediaScreen: cPwaAssetGenerator.ADD_MEDIA_SCREEN,
        basePath: cPwaAssetGenerator.BASE_PATH,
        xhtml: cPwaAssetGenerator.XHTML
      },
      png: {
        compressionLevel: cPwaAssetGenerator.PNG_COMPRESSION_LEVEL,
        quality: cPwaAssetGenerator.PNG_QUALITY
      },
      name: (landscape, size, dark) => {
        const orientation = landscape ? SplashOrientation.LANDSCAPE : SplashOrientation.PORTRAIT;
        const darkPrefix = dark ? cPwaAssetGenerator.DARK_PREFIX : '';
        return `apple-splash-${orientation}-${darkPrefix}${size.width}x${size.height}.png`;
      }
    },
    cPwaGeneratorDevices
  ),
  images: ['static/favicon.svg']
});
