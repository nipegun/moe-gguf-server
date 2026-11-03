import { build } from 'esbuild';
import { dirname, resolve } from 'path';
import { fileURLToPath } from 'url';
import type { Plugin } from 'vite';

const cDirname = dirname(fileURLToPath(import.meta.url));

const cVendorsDir = resolve(cDirname, '../src/lib/vendors');
const cVirtualId = 'virtual:nerdamer';
const cResolvedId = '\0' + cVirtualId;

/**
 * Bundle the vendored nerdamer-prime source into a minified IIFE string,
 * exposed as the `virtual:nerdamer` module. Flags mirror the upstream
 * build (esbuild --bundle --minify --format=iife --global-name=nerdamer),
 * so only human readable source lives in the repo and minification is a
 * build artifact. Vendored under src/lib/vendors/, upstream snapshot:
 * https://github.com/together-science/nerdamer-prime/commit/1936145f8af306ec0d883b9bfd7730aedd175c24
 */
export function fNerdamerPlugin(): Plugin {
  let vBundled: string | null = null;

  return {
    name: 'mgs:nerdamer',
    resolveId(pId) {
      return pId === cVirtualId ? cResolvedId : undefined;
    },
    async load(pId) {
      if (pId !== cResolvedId) return undefined;
      if (vBundled === null) {
        const cResult = await build({
          entryPoints: [resolve(cVendorsDir, 'nerdamer-prime/all.js')],
          bundle: true,
          minify: true,
          format: 'iife',
          globalName: 'nerdamer',
          alias: {
            'big-integer': resolve(cVendorsDir, 'big-integer/BigInteger.js'),
            'decimal.js': resolve(cVendorsDir, 'decimal.js/decimal.js')
          },
          write: false,
          logLevel: 'silent'
        });
        vBundled = cResult.outputFiles[0].text;
      }
      return `export default ${JSON.stringify(vBundled)};`;
    }
  };
}
