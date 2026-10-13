import { fApiUrl } from '$lib/api-url';
import { fTranslate } from '../i18n';
import { base } from '$app/paths';
import { error } from '@sveltejs/kit';
import { browser } from '$app/environment';
import { cAuthorizationHeader, cBearerPrefix, cContentTypeHeader } from '$lib/constants';
import { MimeTypeApplication } from '$lib/enums';
import { fConfig } from '$lib/stores/settings.svelte';

/**
 * Validates API key by making a request to the server props endpoint
 * Throws SvelteKit errors for authentication failures or server issues
 */
export async function fValidateApiKey(pFetch: typeof globalThis.fetch): Promise<void> {
  if (!browser) {
    return;
  }

  const cApiKey = fConfig().apiKey;

  try {
    const dHeaders: Record<string, string> = {
      [cContentTypeHeader]: MimeTypeApplication.JSON
    };

    // Probe /props even without a stored key: on a server started with
    // --api-key the unauthenticated request returns 401 and surfaces the
    // API key splash, which is the onboarding path for entering the key.
    if (cApiKey) {
      dHeaders[cAuthorizationHeader] = `${cBearerPrefix}${cApiKey}`;
    }

    const cResponse = await pFetch(fApiUrl('/props'), { headers: dHeaders });

    if (!cResponse.ok) {
      if (cResponse.status === 401 || cResponse.status === 403) {
        throw error(401, fTranslate('text_cc11d415d932'));
      }

      console.warn(`Server responded with status ${cResponse.status} during API key validation`);
      return;
    }
  } catch (vErr) {
    // If it's already a SvelteKit error, re-throw it
    if (vErr && typeof vErr === 'object' && 'status' in vErr) {
      throw vErr;
    }

    // Network or other errors
    console.warn('Cannot connect to server for API key validation:', vErr);
  }
}
