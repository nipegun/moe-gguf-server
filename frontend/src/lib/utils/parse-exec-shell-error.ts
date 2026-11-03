export function fParseExecShellCommandError(
  pToolResultString: string | undefined
): string | undefined {
  if (!pToolResultString) return undefined;
  try {
    const cParsed: unknown = JSON.parse(pToolResultString);
    if (
      cParsed &&
      typeof cParsed === 'object' &&
      !Array.isArray(cParsed) &&
      typeof (cParsed as Record<string, unknown>).error === 'string'
    ) {
      return (cParsed as { error: string }).error;
    }
  } catch {
    // Plain-text result = stdout/stderr, no structured error to surface.
  }
  return undefined;
}
