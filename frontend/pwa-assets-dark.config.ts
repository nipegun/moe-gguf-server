import { defineConfig } from '@vite-pwa/assets-generator/config';
import { cFaviconColors, cPwaAssetGenerator } from './src/lib/constants/pwa';
import { writeThemeFavicons } from './scripts/favicon-colorize';

writeThemeFavicons(cFaviconColors.LIGHT, cFaviconColors.DARK, {
  padding: cPwaAssetGenerator.FAVICON_PADDING
});

export default defineConfig({
  headLinkOptions: {
    preset: '2023'
  },
  preset: {
    transparent: {
      sizes: [],
      favicons: [[48, 'favicon-dark.ico']],
      padding: cPwaAssetGenerator.FAVICON_PADDING
    },
    maskable: {
      sizes: []
    },
    apple: {
      sizes: []
    }
  },
  images: ['static/favicon-dark.svg']
});
