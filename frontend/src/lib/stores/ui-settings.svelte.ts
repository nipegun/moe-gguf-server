/**
 * uiSettingsStore - Settings document of the interface, kept by the server
 *
 * The server keeps every setting of the interface in ~/.mgs/config/ui-settings.json of the user that
 * runs mgs, shared by every browser (GET and POST /api/ui/settings). The document is split in
 * sections; the store that owns a section registers a loader, which receives the saved value once
 * the document arrives, and calls mSet when the value changes. Changes are saved shortly after,
 * always as the whole document. Nothing is saved before the document arrives: the saved values
 * replace whatever changed in that short time, so a partial state never overwrites the file.
 *
 * The API key that authenticates this browser is the only setting kept in the browser, because it is
 * needed to load the rest.
 */

import { browser } from '$app/environment';
import { UiSettingsService } from '$lib/services/ui-settings.service';
import { ApiError } from '$lib/utils/api-fetch';
import type { UiSettingsDocument } from '$lib/types';

type UiSettingsSection = keyof UiSettingsDocument;

const cSaveDelayMs = 300;
const cRetryDelayMs = 5000;
const cFlushTimeoutMs = 3000;

class UiSettingsStore {
  /** True once the saved document has been read from the server */
  vIsLoaded = $state(false);

  /** True when the server had no saved document: the first visit to this installation */
  vIsFirstRun = false;

  private dDocument: UiSettingsDocument = {};
  private dLoaders = new Map<UiSettingsSection, (pValue: unknown) => void>();
  private vLoading: Promise<void> | null = null;
  private vSaveTimer: ReturnType<typeof setTimeout> | null = null;
  private vSaving = false;
  private vSaveAgain = false;

  constructor() {
    if (browser) {
      // a change still waiting for its delay is sent while the page unloads
      window.addEventListener('pagehide', () => {
        if (this.vSaveTimer) {
          clearTimeout(this.vSaveTimer);
          this.vSaveTimer = null;
          void UiSettingsService.mSave(this.dDocument, true).catch(() => {});
        }
      });
    }
  }

  /**
   * Registers the loader of a section: it runs with the saved value (undefined when the document
   * has none) when the document arrives, or at once if it has already arrived.
   */
  mRegister<K extends UiSettingsSection>(
    pSection: K,
    pLoader: (pValue: UiSettingsDocument[K] | undefined) => void
  ): void {
    this.dLoaders.set(pSection, pLoader as (pValue: unknown) => void);
    if (this.vIsLoaded) {
      pLoader(this.dDocument[pSection]);
    }
  }

  mGet<K extends UiSettingsSection>(pSection: K): UiSettingsDocument[K] | undefined {
    return this.dDocument[pSection];
  }

  /** Changes a section and saves the document shortly after; ignored until the document arrives */
  mSet<K extends UiSettingsSection>(pSection: K, pValue: UiSettingsDocument[K]): void {
    if (!this.vIsLoaded) {
      return;
    }

    this.dDocument[pSection] = $state.snapshot(pValue) as UiSettingsDocument[K];
    this.mScheduleSave();
  }

  /** Reads the saved document from the server and hands every section to its loader */
  mLoad(): Promise<void> {
    if (!browser || this.vIsLoaded) {
      return Promise.resolve();
    }

    this.vLoading ??= this.mLoadOnce();

    return this.vLoading;
  }

  /**
   * Saves pending changes at once and waits for them, for example before the page reloads.
   * @returns false when they could not be saved within the time limit
   */
  async mFlush(): Promise<boolean> {
    const cDeadline = Date.now() + cFlushTimeoutMs;

    while ((this.vSaveTimer || this.vSaving || this.vSaveAgain) && Date.now() < cDeadline) {
      if (this.vSaveTimer && !this.vSaving) {
        clearTimeout(this.vSaveTimer);
        this.vSaveTimer = null;
        await this.mSaveNow();
      } else {
        await new Promise((pResolve) => setTimeout(pResolve, 50));
      }
    }

    return !this.vSaveTimer && !this.vSaving && !this.vSaveAgain;
  }

  private async mLoadOnce(): Promise<void> {
    try {
      const cSaved = await UiSettingsService.mLoad();

      this.vIsFirstRun = cSaved === null;
      this.dDocument = { ...(cSaved ?? {}) };
      this.vIsLoaded = true;

      for (const [cSection, fLoader] of this.dLoaders) {
        fLoader(this.dDocument[cSection]);
      }
    } catch (vErr) {
      this.vLoading = null;
      console.error('Failed to load the settings of the interface from the server:', vErr);

      // without the API key the load waits for the key to be entered, which reloads the page
      if (!(vErr instanceof ApiError && (vErr.vStatus === 401 || vErr.vStatus === 403))) {
        setTimeout(() => void this.mLoad(), cRetryDelayMs);
      }
    }
  }

  private mScheduleSave(pDelayMs = cSaveDelayMs): void {
    if (this.vSaveTimer) {
      clearTimeout(this.vSaveTimer);
    }

    this.vSaveTimer = setTimeout(() => {
      this.vSaveTimer = null;
      void this.mSaveNow();
    }, pDelayMs);
  }

  private async mSaveNow(): Promise<void> {
    if (this.vSaving) {
      this.vSaveAgain = true;
      return;
    }

    this.vSaving = true;

    try {
      await UiSettingsService.mSave(this.dDocument);
    } catch (vErr) {
      console.error('Failed to save the settings of the interface on the server:', vErr);
      this.mScheduleSave(cRetryDelayMs);
    } finally {
      this.vSaving = false;

      if (this.vSaveAgain) {
        this.vSaveAgain = false;
        this.mScheduleSave();
      }
    }
  }
}

export const cUiSettingsStore = new UiSettingsStore();
