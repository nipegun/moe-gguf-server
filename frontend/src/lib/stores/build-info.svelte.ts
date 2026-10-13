/**
 * buildInfoStore - mgs build information
 *
 * Reads the build version from `build.json` — embedded at mgs build time
 * with the mgs version (MGS_VERSION in backend/cmake/build-info.cmake). Shown in the UI when
 * `showBuildVersion` is enabled.
 *
 * In dev mode (via `npm run dev`), falls back to `import.meta.env.DEV`'s truthy
 * value since the artifact is not produced.
 */

import { browser } from '$app/environment';
import { base } from '$app/paths';

let vBuild = $state<string>('');

async function fLoadBuild() {
  if (!browser) return;

  if (import.meta.env.DEV) {
    vBuild = 'dev';
    return;
  }

  try {
    const cRes = await fetch(`${base}/build.json`, { cache: 'no-store' });
    if (cRes.ok) {
      const cData = await cRes.json();
      vBuild = cData.version ?? '';
    }
  } catch {
    // build.json missing or unreachable - leave as empty string
  }
}

fLoadBuild();

export const dBuildInfoStore = {
  get value(): string {
    return vBuild;
  }
};
