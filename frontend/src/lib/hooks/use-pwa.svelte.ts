import { browser } from '$app/environment';
import { useRegisterSW } from 'virtual:pwa-register/svelte';
import { dVersionStore } from '$lib/stores/version.svelte';
import { cBuildVersionLocalstorageKey } from '$lib/constants/storage';
import { cSwConfig } from '$lib/constants/pwa';

/**
 * Hook for PWA service worker registration, update polling, and build version mismatch detection.
 *
 * Combines two concerns that always belong together:
 * 1. SW registration with periodic polling for updates
 * 2. localStorage-based version tracking for non-PWA users
 */
export function fUsePwa() {
  let vSwCheckInterval: ReturnType<typeof setInterval> | null = null;
  let vNeedRefreshByStorage = $state(false);

  const {
    // offlineReady, // to do - add installation banners for iOS
    needRefresh: cPwaNeedRefresh,
    updateServiceWorker: cUpdateServiceWorker
  } = useRegisterSW({
    onRegisteredSW(pSwUrl: string, pR: ServiceWorkerRegistration | undefined) {
      if (vSwCheckInterval) {
        clearInterval(vSwCheckInterval);
      }
      vSwCheckInterval = setInterval(async () => {
        if (!pR || pR.installing || !navigator?.onLine) return;

        try {
          const cResp = await fetch(pSwUrl, {
            cache: cSwConfig.UPDATE_FETCH_OPTIONS.CACHE,
            headers: {
              cache: cSwConfig.UPDATE_FETCH_OPTIONS.HEADERS.CACHE,
              'cache-control': cSwConfig.UPDATE_FETCH_OPTIONS.HEADERS.CACHE_CONTROL
            }
          });
          if (cResp?.status === 200) {
            await pR.update();
          }
        } catch (vE) {
          console.error(vE);
        }
      }, cSwConfig.CHECK_INTERVAL_MS);
    },
    onRegisterError(pError: unknown) {
      console.error('[PWA] SW registration error:', pError);
    }
  });

  // Detect version mismatch via localStorage.
  // _app/version.json is SvelteKit's native version file for PWA cache invalidation.
  // This comparison detects server upgrades for non-PWA users.
  $effect(() => {
    if (!browser) return;
    // PWA pages update via the service worker path; the storage check is the non-PWA fallback only
    if (navigator.serviceWorker?.controller) return;

    const cCurrentVersion = dVersionStore.value;
    if (!cCurrentVersion) return;

    try {
      const cStoredVersion = localStorage.getItem(cBuildVersionLocalstorageKey);
      vNeedRefreshByStorage = !!cStoredVersion && cStoredVersion !== cCurrentVersion;
      localStorage.setItem(cBuildVersionLocalstorageKey, cCurrentVersion);
    } catch {
      vNeedRefreshByStorage = false;
    }
  });

  return {
    /** Writable that is true when a PWA service worker update is available */
    get needRefresh() {
      return cPwaNeedRefresh;
    },
    updateServiceWorker: cUpdateServiceWorker,
    /** Version mismatch detected via localStorage (non-PWA users) */
    get needRefreshByStorage() {
      return vNeedRefreshByStorage;
    }
  };
}
