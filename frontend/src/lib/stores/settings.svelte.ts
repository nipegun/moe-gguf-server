import { fTranslate } from '../i18n';
/**
 * settingsStore - Application configuration and theme management
 *
 * This store manages all application settings including AI model parameters, UI preferences,
 * and theme configuration. They are kept by the server in ~/.mgs/config/ui-settings.json
 * (sections "config" and "userOverrides" of uiSettingsStore), with reactive state management
 * using Svelte 5 runes. Only the API key stays in this browser, because it authenticates the
 * request that loads the rest.
 *
 * **Architecture & Relationships:**
 * - **settingsStore** (this class): Configuration state management
 *   - Manages AI model parameters (temperature, max tokens, etc.)
 *   - Handles theme switching and persistence
 *   - Saves through uiSettingsStore
 *   - Offers reactive configuration access
 *
 * - **ChatService**: Reads model parameters for API requests
 * - **UI Components**: Subscribe to theme and configuration changes
 *
 * **Key Features:**
 * - **Model Parameters**: Temperature, max tokens, top-p, top-k, repeat penalty
 * - **Theme Management**: Auto, light, dark theme switching
 * - **Persistence**: Server-side settings document shared by every browser
 * - **Reactive State**: Svelte 5 runes for automatic UI updates
 * - **Default Handling**: Graceful fallback to defaults for missing settings
 * - **Batch Updates**: Efficient multi-setting updates
 * - **Reset Functionality**: Restore defaults for individual or all settings
 *
 * **Configuration Categories:**
 * - Generation parameters (temperature, tokens, sampling)
 * - UI preferences (theme, display options)
 * - System settings (model selection, prompts)
 * - Advanced options (seed, penalties, context handling)
 */

import { browser } from '$app/environment';
import { ColorMode } from '$lib/enums';
import type { SettingsConfigValue, SettingsExportType } from '$lib/types';
import { setMode } from 'mode-watcher';
import { cApiKeyLocalstorageKey, dSettingConfigDefault, cSettingsKeys } from '$lib/constants';
import { cIsMobile } from '$lib/stores/viewport.svelte';
import { ParameterSyncService } from '$lib/services/parameter-sync.service';
import { cServerStore } from '$lib/stores/server.svelte';
import { cUiSettingsStore } from '$lib/stores/ui-settings.svelte';
import {
  configToParameterRecord,
  normalizeFloatingPoint,
  getConfigValue,
  setConfigValue
} from '$lib/utils';

class SettingsStore {
  /**
   *
   *
   * State
   *
   *
   */

  vConfig = $state<SettingsConfigType>({ ...dSettingConfigDefault });
  vIsInitialized = $state(false);
  vUserOverrides = $state<Set<string>>(new Set());

  // True while the server has no saved settings document; gates the one-time
  // application of server ui_settings defaults for new installations.
  private vIsFirstVisit = false;

  /**
   *
   *
   * Utilities (private helpers)
   *
   *
   */

  /**
   * Helper method to get server defaults with null safety
   * Centralizes the pattern of getting and extracting server defaults
   */
  private mGetServerDefaults(): Record<string, string | number | boolean> {
    return ParameterSyncService.mExtractServerDefaults(cServerStore.mDefaultParams);
  }

  constructor() {
    if (browser) {
      this.mInitialize();
    }
  }

  /**
   *
   *
   * Lifecycle
   *
   *
   */

  /**
   * Initialize the settings store: the API key comes from this browser at once, the rest of the
   * settings when the server's settings document arrives (see mApplySaved)
   */
  mInitialize() {
    try {
      this.vConfig[cSettingsKeys.API_KEY] = this.mLoadApiKey();
      cUiSettingsStore.mRegister('config', (pSaved) => this.mApplySaved(pSaved));
    } catch (vError) {
      console.error('Failed to initialize settings store:', vError);
    }
  }

  private mLoadApiKey(): string {
    try {
      return localStorage.getItem(cApiKeyLocalstorageKey) ?? '';
    } catch {
      return '';
    }
  }

  /**
   * Apply the configuration saved on the server
   * Returns default values for missing keys to prevent breaking changes
   */
  private mApplySaved(pSaved: Record<string, SettingsConfigValue> | undefined) {
    // First visit: the server has no settings document yet. Server ui_settings
    // apply once in this state, then the user's config diverges freely.
    this.vIsFirstVisit = cUiSettingsStore.vIsFirstRun;

    const cSavedVal = pSaved && typeof pSaved === 'object' ? pSaved : {};
    const cApiKey = this.vConfig[cSettingsKeys.API_KEY];

    // Merge with defaults to prevent breaking changes
    this.vConfig = {
      ...dSettingConfigDefault,
      ...cSavedVal,
      [cSettingsKeys.API_KEY]: cApiKey
    };

    // Default sendOnEnter to false on mobile when the user has no saved preference
    if (!(cSettingsKeys.SEND_ON_ENTER in cSavedVal)) {
      if (cIsMobile.current) {
        this.vConfig[cSettingsKeys.SEND_ON_ENTER] = false;
      }
    }

    const cSavedOverrides = cUiSettingsStore.mGet('userOverrides');
    this.vUserOverrides = new Set(Array.isArray(cSavedOverrides) ? cSavedOverrides : []);

    // Apply the saved theme
    setMode(this.vConfig[cSettingsKeys.THEME] as ColorMode);
    this.vIsInitialized = true;
  }

  /**
   *
   *
   * Config Updates
   *
   *
   */

  /**
   * Update a specific configuration setting
   * @param pKey - The configuration key to update
   * @param pValue - The new value for the configuration key
   */
  mUpdateConfig<K extends keyof SettingsConfigType>(pKey: K, pValue: SettingsConfigType[K]): void {
    this.vConfig[pKey] = pValue;

    if (ParameterSyncService.mCanSyncParameter(pKey as string)) {
      const cPropsDefaults = this.mGetServerDefaults();
      const cPropsDefault = cPropsDefaults[pKey as string];

      if (cPropsDefault !== undefined) {
        const cNormalizedValue = normalizeFloatingPoint(pValue);
        const cNormalizedDefault = normalizeFloatingPoint(cPropsDefault);

        if (cNormalizedValue === cNormalizedDefault) {
          this.vUserOverrides.delete(pKey as string);
        } else {
          this.vUserOverrides.add(pKey as string);
        }
      }
    }

    this.mSaveConfig();
  }

  /**
   * Update multiple configuration settings at once
   * @param pUpdates - Object containing the configuration updates
   */
  mUpdateMultipleConfig(pUpdates: Partial<SettingsConfigType>) {
    Object.assign(this.vConfig, pUpdates);

    const cPropsDefaults = this.mGetServerDefaults();

    for (const [cKey, cValue] of Object.entries(pUpdates)) {
      if (ParameterSyncService.mCanSyncParameter(cKey)) {
        const cPropsDefault = cPropsDefaults[cKey];

        if (cPropsDefault !== undefined) {
          const cNormalizedValue = normalizeFloatingPoint(cValue);
          const cNormalizedDefault = normalizeFloatingPoint(cPropsDefault);

          if (cNormalizedValue === cNormalizedDefault) {
            this.vUserOverrides.delete(cKey);
          } else {
            this.vUserOverrides.add(cKey);
          }
        }
      }
    }

    this.mSaveConfig();
  }

  /**
   * Save the current configuration: the API key in this browser, everything else on the server
   */
  private mSaveConfig() {
    if (!browser) return;

    const cApiKey = String(this.vConfig[cSettingsKeys.API_KEY] ?? '').trim();

    try {
      if (cApiKey) {
        localStorage.setItem(cApiKeyLocalstorageKey, cApiKey);
      } else {
        localStorage.removeItem(cApiKeyLocalstorageKey);
      }
    } catch (vError) {
      console.error('Failed to save the API key in this browser:', vError);
    }

    const dServerConfig: Record<string, SettingsConfigValue> = { ...this.vConfig };
    delete dServerConfig[cSettingsKeys.API_KEY];

    cUiSettingsStore.mSet('config', dServerConfig);
    cUiSettingsStore.mSet('userOverrides', Array.from(this.vUserOverrides));
  }

  /**
   * Update the theme setting.
   * @param pNewTheme - The new theme value
   */
  mUpdateTheme(pNewTheme: string) {
    this.mUpdateConfig(cSettingsKeys.THEME, pNewTheme);

    setMode(pNewTheme as ColorMode);
  }

  /**
   *
   *
   * Reset
   *
   *
   */

  /**
   * Reset configuration to defaults
   */
  mResetConfig() {
    this.vConfig = { ...dSettingConfigDefault };

    this.mSaveConfig();
  }

  /**
   * Reset theme to default value.
   * Theme is now stored inside the config object.
   */
  mResetTheme() {
    this.mUpdateConfig(cSettingsKeys.THEME, dSettingConfigDefault[cSettingsKeys.THEME]);

    setMode(dSettingConfigDefault[cSettingsKeys.THEME] as ColorMode);
  }

  /**
   * Reset all settings to defaults.
   */
  mResetAll() {
    this.mResetConfig();

    this.mResetTheme();
  }

  /**
   * Reset a parameter to Server default (or UI default if no Server default)
   */
  mResetParameterToServerDefault(pKey: string): void {
    const cServerDefaults = this.mGetServerDefaults();
    const cUiSettings = cServerStore.mUiSettings;

    if (cUiSettings && pKey in cUiSettings) {
      // UI setting from admin config: write actual value
      setConfigValue(this.vConfig, pKey, cUiSettings[pKey]);
    } else if (cServerDefaults[pKey] !== undefined) {
      // sampling param known by server: clear it, let server decide
      setConfigValue(this.vConfig, pKey, '');
    } else if (pKey in dSettingConfigDefault) {
      setConfigValue(this.vConfig, pKey, getConfigValue(dSettingConfigDefault, pKey));
    }

    this.vUserOverrides.delete(pKey);
    this.mSaveConfig();
  }

  /**
   *
   *
   * Server Sync
   *
   *
   */

  /**
   * Initialize settings with props defaults when server properties are first loaded
   * This sets up the default values from /props endpoint
   */
  mSyncWithServerDefaults(): void {
    const cPropsDefaults = this.mGetServerDefaults();
    if (Object.keys(cPropsDefaults).length === 0) return;

    const cUiSettings = cServerStore.mUiSettings;
    const sUiSettingsKeys = new Set(cUiSettings ? Object.keys(cUiSettings) : []);

    for (const [cKey, cPropsValue] of Object.entries(cPropsDefaults)) {
      const cCurrentValue = getConfigValue(this.vConfig, cKey);

      const cNormalizedCurrent = normalizeFloatingPoint(cCurrentValue);
      const cNormalizedDefault = normalizeFloatingPoint(cPropsValue);

      // if user value matches server, it's not a real override
      if (cNormalizedCurrent === cNormalizedDefault) {
        this.vUserOverrides.delete(cKey);

        if (
          !sUiSettingsKeys.has(cKey) &&
          getConfigValue(dSettingConfigDefault, cKey) === undefined
        ) {
          setConfigValue(this.vConfig, cKey, undefined);
        }
      }
    }

    // UI settings are the admin's defaults for new users: applied once on
    // the first visit, never on later loads, so the user's config can
    // diverge. "Reset to Default" is the explicit way back to the baseline.
    if (cUiSettings && this.vIsFirstVisit) {
      this.vIsFirstVisit = false;

      for (const [cKey, cValue] of Object.entries(cUiSettings)) {
        if (
          !this.vUserOverrides.has(cKey) &&
          cValue !== undefined &&
          getConfigValue(this.vConfig, cKey) === getConfigValue(dSettingConfigDefault, cKey)
        ) {
          setConfigValue(this.vConfig, cKey, cValue);

          // theme lives in mode-watcher, not just in config -> propagate
          if (cKey === cSettingsKeys.THEME) {
            setMode(cValue as ColorMode);
          }
        }
      }
    }

    this.mSaveConfig();
    console.log('User overrides after sync:', Array.from(this.vUserOverrides));
  }

  /**
   * Reset all parameters to their default values (from props)
   * This is used by the "Reset to Default" functionality
   * Prioritizes Server defaults from /props, falls back to UI defaults
   */
  mForceSyncWithServerDefaults(): void {
    const cPropsDefaults = this.mGetServerDefaults();
    const cUiSettings = cServerStore.mUiSettings;

    for (const cKey of ParameterSyncService.mGetSyncableParameterKeys()) {
      if (cUiSettings && cKey in cUiSettings) {
        // UI setting from admin config: write actual value
        setConfigValue(this.vConfig, cKey, cUiSettings[cKey]);
      } else if (cPropsDefaults[cKey] !== undefined) {
        // sampling param: clear it, let server decide
        setConfigValue(this.vConfig, cKey, '');
      } else if (cKey in dSettingConfigDefault) {
        setConfigValue(this.vConfig, cKey, getConfigValue(dSettingConfigDefault, cKey));
      }

      this.vUserOverrides.delete(cKey);
    }

    // Non-syncable keys: reset is a full return to the instance state, the
    // admin baseline value when defined, the factory default otherwise.
    for (const cKey of Object.keys(dSettingConfigDefault)) {
      if (ParameterSyncService.mCanSyncParameter(cKey)) {
        continue;
      }

      const cValue =
        cUiSettings && cKey in cUiSettings && cUiSettings[cKey] !== undefined
          ? cUiSettings[cKey]
          : getConfigValue(dSettingConfigDefault, cKey);

      setConfigValue(this.vConfig, cKey, cValue);

      if (cKey === cSettingsKeys.THEME) {
        setMode(cValue as ColorMode);
      }

      this.vUserOverrides.delete(cKey);
    }

    this.mSaveConfig();
  }

  /**
   *
   *
   * Utilities
   *
   *
   */

  /**
   * Get a specific configuration value
   * @param pKey - The configuration key to get
   * @returns The configuration value
   */
  mGetConfig<K extends keyof SettingsConfigType>(pKey: K): SettingsConfigType[K] {
    return this.vConfig[pKey];
  }

  /**
   * Get the entire configuration object
   * @returns The complete configuration object
   */
  mGetAllConfig(): SettingsConfigType {
    return { ...this.vConfig };
  }

  mCanSyncParameter(pKey: string): boolean {
    return ParameterSyncService.mCanSyncParameter(pKey);
  }

  /**
   * Get parameter information including source for a specific parameter
   */
  mGetParameterInfo(pKey: string) {
    const cPropsDefaults = this.mGetServerDefaults();
    const cCurrentValue = getConfigValue(this.vConfig, pKey);

    return ParameterSyncService.mGetParameterInfo(
      pKey,
      cCurrentValue ?? '',
      cPropsDefaults,
      this.vUserOverrides
    );
  }

  /**
   * Get diff between current settings and server defaults
   */
  mGetParameterDiff() {
    const cServerDefaults = this.mGetServerDefaults();
    if (Object.keys(cServerDefaults).length === 0) return {};

    const cConfigAsRecord = configToParameterRecord(
      this.vConfig,
      ParameterSyncService.mGetSyncableParameterKeys()
    );

    return ParameterSyncService.mCreateParameterDiff(cConfigAsRecord, cServerDefaults);
  }

  /**
   * Clear all user overrides (for debugging)
   */
  mClearAllUserOverrides(): void {
    this.vUserOverrides.clear();
    this.mSaveConfig();
    console.log('Cleared all user overrides');
  }

  /**
   *
   *
   * Import / Export
   *
   *
   */

  /**
   * Export all settings as a versioned JSON-compatible object.
   * The export captures the full config (excluding sensitive values like API key)
   * and user overrides. Sensitive fields are filtered out for security by default.
   * @param pIncludeSensitiveData - If true, include sensitive fields (apiKey, MCP server headers) in export
   */
  mExportSettings(pIncludeSensitiveData: boolean = false): SettingsExportType {
    // Build config excluding sensitive data unless user opts in
    const dConfigToExport: Record<string, string | number | boolean | undefined> =
      pIncludeSensitiveData
        ? { ...this.vConfig }
        : Object.fromEntries(Object.entries(this.vConfig).filter(([pKey]) => pKey !== 'apiKey'));

    // Handle MCP servers: exclude custom headers unless user opts in
    if ('mcpServers' in dConfigToExport && !pIncludeSensitiveData) {
      try {
        const cMcpServers = JSON.parse(dConfigToExport.mcpServers as string) as Array<
          Record<string, unknown>
        >;
        const cSafeServers = cMcpServers.map((pServer) => {
          delete pServer.headers;
          return pServer;
        });
        dConfigToExport.mcpServers = JSON.stringify(cSafeServers);
      } catch {
        // If parsing fails, just exclude the entire mcpServers field
        delete (dConfigToExport as Record<string, unknown>).mcpServers;
      }
    }

    return {
      version: 1,
      timestamp: Date.now(),
      config: dConfigToExport,
      userOverrides: Array.from(this.vUserOverrides)
    };
  }

  /**
   * Import settings from a previously exported object.
   * Restores config (including theme) and user overrides.
   * @param pData - The exported settings object
   */
  mImportSettings(pData: SettingsExportType): void {
    if (!browser) return;

    if (!pData || !pData.config) {
      throw new Error(fTranslate('text_0a8d4a6abfc5'));
    }

    // Restore config (theme is included in config)
    this.vConfig = {
      ...dSettingConfigDefault,
      ...pData.config
    };

    // Restore user overrides (derived state — may be stale if server defaults differ)
    this.vUserOverrides = new Set(pData.userOverrides ?? []);

    // Save on the server (the API key in this browser)
    this.mSaveConfig();

    // Apply theme for immediate visual feedback
    setMode(this.vConfig[cSettingsKeys.THEME] as ColorMode);

    console.log('Settings imported successfully');
  }
}

export const cSettingsStore = new SettingsStore();

export const fConfig = () => cSettingsStore.vConfig;
export const fTheme = () => cSettingsStore.vConfig[cSettingsKeys.THEME];
export const fIsInitialized = () => cSettingsStore.vIsInitialized;
