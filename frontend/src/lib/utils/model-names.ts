import { cFilePathSeparatorRegex } from '$lib/constants';

/**
 * Normalizes a model name by extracting the filename from a path, but preserves Hugging Face repository format.
 *
 * Handles both forward slashes (/) and backslashes (\) as path separators.
 * - If the model name has exactly one slash (org/model format), preserves the full "org/model" name
 * - If the model name has no slash or multiple slashes, extracts just the filename
 * - If the model name is just a filename (no path), returns it as-is.
 *
 * @param pModelName - The model name or path to normalize
 * @returns The normalized model name
 *
 * @example
 * normalizeModelName('models/qwen3-30b-a3b') // Returns: 'qwen3-30b-a3b' (multiple slashes -> filename)
 * normalizeModelName('C:\\Models\\gpt-4') // Returns: 'gpt-4' (multiple slashes -> filename)
 * normalizeModelName('Qwen/Qwen3-30B-A3B') // Returns: 'Qwen/Qwen3-30B-A3B' (Hugging Face format)
 * normalizeModelName('simple-model') // Returns: 'simple-model' (no slash)
 * normalizeModelName('  spaced  ') // Returns: 'spaced'
 * normalizeModelName('') // Returns: ''
 */
export function fNormalizeModelName(pModelName: string): string {
  const cTrimmed = pModelName.trim();

  if (!cTrimmed) {
    return '';
  }

  const cSegments = cTrimmed.split(cFilePathSeparatorRegex);

  // If we have exactly 2 segments (one slash), treat it as Hugging Face repo format
  // and preserve the full "org/model" format
  if (cSegments.length === 2) {
    const [cOrg, cModel] = cSegments;
    const cTrimmedOrg = cOrg?.trim();
    const cTrimmedModel = cModel?.trim();

    if (cTrimmedOrg && cTrimmedModel) {
      return `${cTrimmedOrg}/${cTrimmedModel}`;
    }
  }

  // For other cases (no slash, or multiple slashes), extract just the filename
  const cCandidate = cSegments.pop();
  const cNormalized = cCandidate?.trim();

  return cNormalized && cNormalized.length > 0 ? cNormalized : cTrimmed;
}

/**
 * Validates if a model name is valid (non-empty after normalization).
 *
 * @param pModelName - The model name to validate
 * @returns true if valid, false otherwise
 */
export function fIsValidModelName(pModelName: string): boolean {
  return fNormalizeModelName(pModelName).length > 0;
}
