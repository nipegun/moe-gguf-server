import tailwindcss from '@tailwindcss/vite';
import { sveltekit } from '@sveltejs/kit/vite';
import { SvelteKitPWA } from '@vite-pwa/sveltekit';
import { resolve } from 'path';

import { defineConfig } from 'vite';
import { fSplashScreenPlugin } from './scripts/vite-plugin-splash-screen';
import { fBuildInfoPlugin } from './scripts/vite-plugin-build-info';
import { fRelativizeBasePlugin } from './scripts/vite-plugin-relativize-base';
import { fNerdamerPlugin } from './scripts/vite-plugin-nerdamer';
import { dSveltekitPwaOptions } from './src/lib/constants/pwa';

export default defineConfig({
  server: {
    proxy: {
      '/api': { target: 'http://127.0.0.1:8080', changeOrigin: true }
    }
  },
  resolve: {
    alias: {
      'katex-fonts': resolve('node_modules/katex/dist/fonts')
    }
  },

  build: {
    assetsInlineLimit: 32000,
    chunkSizeWarningLimit: 3072,
    minify: true
  },

  plugins: [
    tailwindcss(),
    sveltekit(),
    SvelteKitPWA(dSveltekitPwaOptions),
    fSplashScreenPlugin(),
    fBuildInfoPlugin(),
    fNerdamerPlugin(),
    fRelativizeBasePlugin()
  ]
});
