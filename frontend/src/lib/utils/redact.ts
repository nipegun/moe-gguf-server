/**
 * Redacts a sensitive value, optionally showing the last N characters.
 *
 * @param pValue - The value to redact
 * @param pShowLastChars - If provided, reveals the last N characters with a leading mask
 * @returns The redacted string
 */
export function fRedactValue(pValue: string, pShowLastChars?: number): string {
  if (pShowLastChars) {
    return `....${pValue.slice(-pShowLastChars)}`;
  }

  return '[redacted]';
}
