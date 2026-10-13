// Meta parser for `exec_shell_command` tool calls. Surfaces the
// command text from args `command` / `cmd` / `shell_command` aliases.
// The exit-status and error parsing live in their own utilities
// (`parse-exec-shell-status.ts` / `parse-exec-shell-error.ts`) - this
// file only deals with what's strictly about *calling* the tool, since
// the error / exit status elide from call-section to result-section.

import { BuiltInTool } from '$lib/enums';
import type { AgenticSection } from '$lib/utils';
import { fParseToolArgs } from './_shared';

export type ExecShellCommandMeta = {
  command: string;
};

export function fParseExecShellCommandMeta(pSection: AgenticSection): ExecShellCommandMeta | null {
  const cArgs = fParseToolArgs(BuiltInTool.EXEC_SHELL_COMMAND, pSection);
  if (!cArgs) return null;

  const cCommandRaw = cArgs.command ?? cArgs.cmd ?? cArgs.shell_command;
  if (typeof cCommandRaw !== 'string' || !cCommandRaw) return null;
  return { command: cCommandRaw };
}
