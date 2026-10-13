/**
 * Type-safe configuration helpers
 *
 * Provides utilities for safely accessing and modifying configuration objects
 * with dynamic keys while maintaining TypeScript type safety.
 */

/**
 * Type-safe helper to access config properties dynamically
 * Provides better type safety than direct casting to Record
 */
export function fSetConfigValue<T extends SettingsConfigType>(
  pConfig: T,
  pKey: string,
  pValue: unknown
): void {
  if (pKey in pConfig) {
    (pConfig as Record<string, unknown>)[pKey] = pValue;
  }
}

/**
 * Type-safe helper to get config values dynamically
 */
export function fGetConfigValue<T extends SettingsConfigType>(
  pConfig: T,
  pKey: string
): string | number | boolean | undefined {
  const cValue = (pConfig as Record<string, unknown>)[pKey];
  return cValue as string | number | boolean | undefined;
}

/**
 * Convert a SettingsConfigType to a ParameterRecord for specific keys
 * Useful for parameter synchronization operations
 */
export function fConfigToParameterRecord<T extends SettingsConfigType>(
  pConfig: T,
  pKeys: string[]
): Record<string, string | number | boolean> {
  const dRecord: Record<string, string | number | boolean> = {};

  for (const cKey of pKeys) {
    const cValue = fGetConfigValue(pConfig, cKey);
    if (cValue !== undefined) {
      dRecord[cKey] = cValue;
    }
  }

  return dRecord;
}
