// Meta parser for the tools of the local shell MCP server
// (--local-shell-mcp-server): bash.run, zsh.run, sh.run and psh.run.
// The command comes from the streamed args, so the stdin box fills in
// while the model writes it; the result is the JSON object the server
// answers with (exit_code, stdout, stderr, timed_out, cancelled...).

import { cShellMcpServerId, aShellMcpToolNames } from '$lib/constants/mcp';
import { cMcpStore } from '$lib/stores/mcp.svelte';
import { fParsePartialJsonArgs } from '$lib/utils/parse-partial-json-args';
import type { AgenticSection } from '$lib/utils/agentic';

export type LocalShellRunResult = {
  exitCode: number | null;
  stdout: string;
  stderr: string;
  timedOut: boolean;
  cancelled: boolean;
};

/**
 * True for the tools of the local shell MCP server. Another MCP server
 * may expose a tool with the same name; a tool whose owner is unknown
 * (the server is not connected any more) is still shown as a shell call.
 */
export function fIsLocalShellTool(pToolName: string | undefined): boolean {
  if (!pToolName || !aShellMcpToolNames.includes(pToolName)) return false;
  const vServerId = cMcpStore.mFindServerForTool(pToolName);
  return vServerId === undefined || vServerId === cShellMcpServerId;
}

/** The bare command of the call, also while its args are still streaming. */
export function fParseLocalShellCommand(pSection: AgenticSection): string | null {
  if (!pSection.toolArgs) return null;
  const dArgs = fParsePartialJsonArgs(pSection.toolArgs);
  const vCommand = dArgs?.command;
  return typeof vCommand === 'string' ? vCommand : null;
}

/**
 * The JSON object answered by the server, or null when the result is
 * something else (an argument error, a denied permission, a failed
 * request), which the block then shows as plain error text.
 */
export function fParseLocalShellResult(
  pToolResult: string | undefined
): LocalShellRunResult | null {
  if (!pToolResult) return null;
  try {
    const dResult: unknown = JSON.parse(pToolResult);
    if (!dResult || typeof dResult !== 'object' || Array.isArray(dResult)) return null;
    const dFields = dResult as Record<string, unknown>;
    if (typeof dFields.stdout !== 'string' || typeof dFields.stderr !== 'string') return null;
    return {
      exitCode: typeof dFields.exit_code === 'number' ? dFields.exit_code : null,
      stdout: dFields.stdout,
      stderr: dFields.stderr,
      timedOut: dFields.timed_out === true,
      cancelled: dFields.cancelled === true
    };
  } catch {
    return null;
  }
}
