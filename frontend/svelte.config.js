import { mdsvex } from 'mdsvex';
import adapter from '@sveltejs/adapter-static';
import { vitePreprocess } from '@sveltejs/vite-plugin-svelte';

// CMake sets MGS_UI_OUT_DIR to the staging dir under the build tree; manual
// `npm run build` runs without the env var default to ./dist.
const cOutDir = process.env.MGS_UI_OUT_DIR ?? './dist';

/** @type {import('@sveltejs/kit').Config} */
const dConfig = {
  // Consult https://svelte.dev/docs/kit/integrations
  // for more information about preprocessors
  preprocess: [vitePreprocess(), mdsvex()],

  kit: {
    paths: {
      relative: true
    },
    router: { type: 'hash' },
    adapter: adapter({
      pages: cOutDir,
      assets: cOutDir,
      fallback: 'index.html',
      precompress: false,
      strict: true
    }),
    output: {
      bundleStrategy: 'single'
    },
    alias: {
      $styles: 'src/styles'
    }
  },

  extensions: ['.svelte', '.svx']
};

export default dConfig;
