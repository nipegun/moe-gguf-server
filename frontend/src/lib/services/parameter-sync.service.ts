import { normalizeFloatingPoint } from '$lib/utils';
import { cSettingsKeys, aSyncableParameters } from '$lib/constants';
import type { ParameterRecord, ParameterInfo, ParameterValue } from '$lib/types';
import { SyncableParameterType, ParameterSource } from '$lib/enums';

export class ParameterSyncService {
  /**
   *
   *
   * Extraction
   *
   *
   */

  /**
   * Round floating-point numbers to avoid JavaScript precision issues.
   * E.g., 0.1 + 0.2 = 0.30000000000000004 → 0.3
   *
   * @param pValue - Parameter value to normalize
   * @returns Precision-normalized value
   */
  private static mRoundFloatingPoint(pValue: ParameterValue): ParameterValue {
    return normalizeFloatingPoint(pValue) as ParameterValue;
  }

  /**
   * Extract server default parameters that can be synced from `/props` response.
   * Handles both generation settings parameters and UI-specific settings.
   * Converts samplers array to semicolon-delimited string for UI display.
   *
   * @param pServerParams - Raw generation settings from server `/props` endpoint
   * @returns Record of extracted parameter key-value pairs with normalized precision
   */
  static mExtractServerDefaults(
    pServerParams: ApiMgsServerProps['default_generation_settings']['params'] | null
  ): ParameterRecord {
    const dExtracted: ParameterRecord = {};

    if (pServerParams) {
      for (const cParam of aSyncableParameters) {
        if (cParam.canSync && cParam.serverKey in pServerParams) {
          const cValue = (pServerParams as unknown as Record<string, ParameterValue>)[
            cParam.serverKey
          ];
          if (cValue !== undefined) {
            // Apply precision rounding to avoid JavaScript floating-point issues
            dExtracted[cParam.key] = this.mRoundFloatingPoint(cValue);
          }
        }
      }

      // Handle samplers array conversion to string
      if (pServerParams.samplers && Array.isArray(pServerParams.samplers)) {
        dExtracted[cSettingsKeys.SAMPLERS] = pServerParams.samplers.join(';');
      }
    }

    return dExtracted;
  }

  /**
   *
   *
   * Merging
   *
   *
   */

  /**
   * Merge server defaults with current user settings.
   * User overrides always take priority — only parameters not in `userOverrides`
   * set will be updated from server defaults.
   *
   * @param pCurrentSettings - Current parameter values in the settings store
   * @param pServerDefaults - Default values extracted from server props
   * @param pUserOverrides - Set of parameter keys explicitly overridden by the user
   * @returns Merged parameter record with user overrides preserved
   */
  static mMergeWithServerDefaults(
    pCurrentSettings: ParameterRecord,
    pServerDefaults: ParameterRecord,
    pUserOverrides: Set<string> = new Set()
  ): ParameterRecord {
    const dMerged = { ...pCurrentSettings };

    for (const [cKey, cServerValue] of Object.entries(pServerDefaults)) {
      // Only update if user hasn't explicitly overridden this parameter
      if (!pUserOverrides.has(cKey)) {
        dMerged[cKey] = this.mRoundFloatingPoint(cServerValue);
      }
    }

    return dMerged;
  }

  /**
   *
   *
   * Info
   *
   *
   */

  /**
   * Get parameter information including source and values.
   * Used by SettingsChatParameterSourceIndicator to display the correct badge
   * (Custom vs Default) for each parameter in the settings UI.
   *
   * @param pKey - The parameter key to get info for
   * @param pCurrentValue - The current value of the parameter
   * @param pPropsDefaults - Server default values from `/props`
   * @param pUserOverrides - Set of parameter keys explicitly overridden by the user
   * @returns Parameter info with source, server default, and user override values
   */
  static mGetParameterInfo(
    pKey: string,
    pCurrentValue: ParameterValue,
    pPropsDefaults: ParameterRecord,
    pUserOverrides: Set<string>
  ): ParameterInfo {
    const cHasPropsDefault = pPropsDefaults[pKey] !== undefined;
    const cIsUserOverride = pUserOverrides.has(pKey);

    // Simple logic: either using default (from props) or custom (user override)
    const cSource = cIsUserOverride ? ParameterSource.CUSTOM : ParameterSource.DEFAULT;

    return {
      value: pCurrentValue,
      source: cSource,
      serverDefault: cHasPropsDefault ? pPropsDefaults[pKey] : undefined,
      userOverride: cIsUserOverride ? pCurrentValue : undefined
    };
  }

  /**
   * Check if a parameter can be synced from server.
   *
   * @param pKey - The parameter key to check
   * @returns True if the parameter is in the syncable parameters list
   */
  static mCanSyncParameter(pKey: string): boolean {
    return aSyncableParameters.some((pParam) => pParam.key === pKey && pParam.canSync);
  }

  /**
   * Get all syncable parameter keys.
   *
   * @returns Array of parameter keys that can be synced from server
   */
  static mGetSyncableParameterKeys(): string[] {
    return aSyncableParameters.filter((pParam) => pParam.canSync).map((pParam) => pParam.key);
  }

  /**
   * Validate a server parameter value against its expected type.
   *
   * @param pKey - The parameter key to validate
   * @param pValue - The value to validate
   * @returns True if value matches the expected type for this parameter
   */
  static mValidateServerParameter(pKey: string, pValue: ParameterValue): boolean {
    const cParam = aSyncableParameters.find((pP) => pP.key === pKey);
    if (!cParam) return false;

    switch (cParam.type) {
      case SyncableParameterType.NUMBER:
        return typeof pValue === 'number' && !isNaN(pValue);
      case SyncableParameterType.STRING:
        return typeof pValue === 'string';
      case SyncableParameterType.BOOLEAN:
        return typeof pValue === 'boolean';
      default:
        return false;
    }
  }

  /**
   *
   *
   * Diff
   *
   *
   */

  /**
   * Create a diff between current settings and server defaults.
   * Shows which parameters differ from server values, useful for debugging
   * and for the "Reset to defaults" functionality.
   *
   * @param pCurrentSettings - Current parameter values in the settings store
   * @param pServerDefaults - Default values extracted from server props
   * @returns Record of parameter diffs with current value, server value, and whether they differ
   */
  static mCreateParameterDiff(
    pCurrentSettings: ParameterRecord,
    pServerDefaults: ParameterRecord
  ): Record<string, { current: ParameterValue; server: ParameterValue; differs: boolean }> {
    const dDiff: Record<
      string,
      { current: ParameterValue; server: ParameterValue; differs: boolean }
    > = {};

    for (const cKey of this.mGetSyncableParameterKeys()) {
      const cCurrentValue = pCurrentSettings[cKey];
      const cServerValue = pServerDefaults[cKey];

      if (cServerValue !== undefined) {
        dDiff[cKey] = {
          current: cCurrentValue,
          server: cServerValue,
          differs: cCurrentValue !== cServerValue
        };
      }
    }

    return dDiff;
  }
}
