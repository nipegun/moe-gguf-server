import {
  cKeyValuePairKeyMaxLength,
  cKeyValuePairValueMaxLength,
  cKeyValuePairUnsafeKeyRe,
  cKeyValuePairUnsafeValueRe
} from '$lib/constants';

/**
 * Strip control characters unsafe in identifier/header-name contexts and cap length.
 * Removes all C0 controls (including TAB) and DEL.
 */
export function fSanitizeKeyValuePairKey(pRaw: string): string {
  return pRaw.replace(cKeyValuePairUnsafeKeyRe, '').slice(0, cKeyValuePairKeyMaxLength);
}

/**
 * Strip control characters that enable header injection; allow TAB; cap length.
 * Removes null bytes, CR/LF and other C0/DEL controls while keeping TAB (\x09),
 * which is a valid header-value continuation character per RFC 7230.
 */
export function fSanitizeKeyValuePairValue(pRaw: string): string {
  return pRaw.replace(cKeyValuePairUnsafeValueRe, '').slice(0, cKeyValuePairValueMaxLength);
}
