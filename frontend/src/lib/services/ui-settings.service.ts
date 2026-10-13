import { apiFetch } from '$lib/utils';
import type { UiSettingsDocument } from '$lib/types';

export class UiSettingsService {
  /** Saved settings document of the Web UI, or null when the server has none yet. */
  static async mLoad(): Promise<UiSettingsDocument | null> {
    const cResponse = await apiFetch<{ settings: UiSettingsDocument | null }>('/ui/settings', {
      authOnly: true,
      cache: 'no-store'
    });
    return cResponse.settings ?? null;
  }

  /**
   * Replaces the whole document kept by the server (~/.mgs/config/ui-settings.json).
   * @param pKeepalive - Let the request finish while the page unloads
   */
  static async mSave(pDocument: UiSettingsDocument, pKeepalive = false): Promise<void> {
    await apiFetch<{ success: boolean }>('/ui/settings', {
      method: 'POST',
      body: JSON.stringify({ settings: pDocument }),
      keepalive: pKeepalive
    });
  }
}
