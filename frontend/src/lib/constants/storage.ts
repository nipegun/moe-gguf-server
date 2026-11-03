/**
 * Storage-related constants (localStorage, IndexedDB).
 *
 * Centralized to ensure consistency across the app and simplify future
 * name changes.
 *
 * The settings of the interface are kept by the server (~/.mgs/config/ui-settings.json,
 * see uiSettingsStore); the browser only keeps the API key that authenticates it, the
 * per-conversation stream state and the build version used to detect updates.
 */

/** Name prefix for all localStorage keys */
export const cStorageAppName = 'MgsUi';

/** API key of this browser: it authenticates the request that loads the settings, so it cannot be one of them */
export const cApiKeyLocalstorageKey = `${cStorageAppName}.apiKey`;

/** Key prefix for per-conversation resumable stream state, conversationId is appended */
export const cStreamResumeLocalstorageKeyPrefix = `${cStorageAppName}.streamResume.`;

/** Build version stored in localStorage for non-PWA update detection */
export const cBuildVersionLocalstorageKey = `${cStorageAppName}.buildVersion`;
