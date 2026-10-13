/**
 * Floating-point precision utilities
 *
 * Provides functions to normalize floating-point numbers for consistent comparison
 * and display, addressing JavaScript's floating-point precision issues.
 */

import { cPrecisionMultiplier } from '$lib/constants';

/**
 * Normalize floating-point numbers for consistent comparison
 * Addresses JavaScript floating-point precision issues (e.g., 0.949999988079071 → 0.95)
 */
export function fNormalizeFloatingPoint(pValue: unknown): unknown {
  return typeof pValue === 'number'
    ? Math.round(pValue * cPrecisionMultiplier) / cPrecisionMultiplier
    : pValue;
}

/**
 * Type-safe version that only accepts numbers
 */
export function fNormalizeNumber(pValue: number): number {
  return Math.round(pValue * cPrecisionMultiplier) / cPrecisionMultiplier;
}
