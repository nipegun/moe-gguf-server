import { writeFileSync, existsSync } from 'node:fs';
import { resolve } from 'path';
import type { Plugin } from 'vite';
import { cBuildConfig } from '../src/lib/constants/pwa';

let vProcessed = false;

const cOutputDir = process.env.MGS_UI_OUT_DIR ?? cBuildConfig.OUTPUT_DIR;

/**
 * Write build.json with the mgs version.
 *
 * MGS_UI_VERSION is passed from CMake -> npm -> vite via env var.
 * Used for display of the current mgs version (e.g. "0.1.0").
 */
export function fBuildInfoPlugin(): Plugin {
  return {
    name: 'mgs:build-info',
    apply: 'build',
    closeBundle() {
      setTimeout(() => {
        try {
          if (vProcessed) return;
          vProcessed = true;

          const cVersion = process.env.MGS_UI_VERSION ?? '';

          const cOutDir = resolve(cOutputDir);
          const cIndexPath = resolve(cOutDir, 'index.html');
          if (!existsSync(cIndexPath)) return;

          const cBuildJsonPath = resolve(cOutDir, 'build.json');
          writeFileSync(cBuildJsonPath, JSON.stringify({ version: cVersion }), 'utf-8');
          console.log(`Created build.json (version: ${cVersion})`);
        } catch (vError) {
          console.error('Failed to write build.json:', vError);
        }
      }, 100);
    }
  };
}
