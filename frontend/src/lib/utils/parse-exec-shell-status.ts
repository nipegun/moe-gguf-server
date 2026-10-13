/**
 * Parsing helpers for `exec_shell_command` tool output.
 *
 * The server appends one final line to the response - an exit-code summary
 * shaped as `[exit code: N]` (and optionally followed by `[exit due to timed
 * out]`) - so the renderer can color that final line based on success/failure
 * without parsing the entire output stream.
 */

export interface ExecShellExitStatus {
  code: number;
  timedOut: boolean;
  /** Length-prefix slice for matching against the rendered lines list. */
  rawText: string;
}

// Anchor to the absolute end so intermediate "[exit code: N]" string content
// (e.g. a shell echo) doesn't false-positive.
const cExitCodeTailRegex = /\[exit code: (-?\d+)\](?: \[exit due to timed out\])?\s*$/;

export function fParseExecShellCommandExitStatus(
  pToolResultString: string | undefined
): ExecShellExitStatus | undefined {
  if (!pToolResultString) return undefined;

  const cMatch = pToolResultString.match(cExitCodeTailRegex);
  if (!cMatch) return undefined;

  return {
    code: Number.parseInt(cMatch[1], 10),
    timedOut: cMatch[0].includes('exit due to timed out'),
    rawText: cMatch[0]
  };
}

/**
 * Returns true when the supplied rendered line equals (trimmed) the
 * trailing exit-code text. Used by the renderer to drop the duplicated
 * representation (since the trailing line is replaced by a status badge).
 */
export function fIsExitCodeSummaryLine(
  pLineText: string,
  pStatus: ExecShellExitStatus | undefined
): boolean {
  if (!pStatus) return false;
  return pLineText.trim() === pStatus.rawText.trim();
}
