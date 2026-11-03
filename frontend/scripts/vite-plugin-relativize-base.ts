import { readFileSync, writeFileSync, existsSync } from 'node:fs';
import { resolve } from 'path';
import type { Plugin } from 'vite';
import { cBuildConfig } from '../src/lib/constants/pwa';

let vProcessed = false;

const cOutputDir = process.env.MGS_UI_OUT_DIR ?? cBuildConfig.OUTPUT_DIR;

function fRewrite(pPath: string, aPairs: [string, string][]): void {
  if (!existsSync(pPath)) {
    return;
  }
  const cText = readFileSync(pPath, 'utf-8');
  let vOut = cText;
  for (const [cFrom, cTo] of aPairs) {
    vOut = vOut.split(cFrom).join(cTo);
  }
  if (vOut !== cText) {
    writeFileSync(pPath, vOut, 'utf-8');
  }
}

/**
 * Relativize SvelteKit absolute base refs so the build is relocatable under any subpath.
 *
 * SvelteKit bakes root absolute /_app/ paths into the SPA fallback because paths.relative
 * does not apply to a depth agnostic fallback page. Rewriting to ./_app/ lets a plain
 * recursive copy of the output into /any/subdir/ resolve assets against the document URL.
 * Runs after adapter-static writes index.html and the PWA plugin writes sw.js, deferred the
 * same way as fBuildInfoPlugin so the emitted files exist.
 */
export function fRelativizeBasePlugin(): Plugin {
  return {
    name: 'mgs:relativize-base',
    apply: 'build',
    closeBundle() {
      setTimeout(() => {
        try {
          if (vProcessed) return;
          vProcessed = true;

          const cOutDir = resolve(cOutputDir);

          // index.html: modulepreload, stylesheet and bootstrap import reference "/_app/
          fRewrite(resolve(cOutDir, 'index.html'), [['"/_app/', '"./_app/']]);

          // sw.js: the only absolute entries are the navigate fallback precache key and handler
          fRewrite(resolve(cOutDir, 'sw.js'), [
            ['{url:"/"', '{url:"./"'],
            ['createHandlerBoundToURL("/"', 'createHandlerBoundToURL("./"']
          ]);

          console.log('Relativized base refs in index.html and sw.js');
        } catch (vError) {
          console.error('Failed to relativize base refs:', vError);
        }
      }, 100);
    }
  };
}
