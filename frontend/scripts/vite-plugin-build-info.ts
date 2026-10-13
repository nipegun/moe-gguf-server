import { writeFileSync, existsSync } from 'node:fs';
import { resolve } from 'path';
import type { Plugin } from 'vite';
import { cBuildConfig } from '../src/lib/constants/pwa';

let processed = false;

const OUTPUT_DIR = process.env.MGS_UI_OUT_DIR ?? cBuildConfig.OUTPUT_DIR;

/**
 * Write build.json with the mgs version.
 *
 * MGS_UI_VERSION is passed from CMake -> npm -> vite via env var.
 * Used for display of the current mgs version (e.g. "0.1.0").
 */
export function buildInfoPlugin(): Plugin {
  return {
    name: 'mgs:build-info',
    apply: 'build',
    closeBundle() {
      setTimeout(() => {
        try {
          if (processed) return;
          processed = true;

          const cVersion = process.env.MGS_UI_VERSION ?? '';

          const outDir = resolve(OUTPUT_DIR);
          const indexPath = resolve(outDir, 'index.html');
          if (!existsSync(indexPath)) return;

          const buildJsonPath = resolve(outDir, 'build.json');
          writeFileSync(buildJsonPath, JSON.stringify({ version: cVersion }), 'utf-8');
          console.log(`Created build.json (version: ${cVersion})`);
        } catch (error) {
          console.error('Failed to write build.json:', error);
        }
      }, 100);
    }
  };
}
