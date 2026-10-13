import { apiFetch, apiFetchWithParams } from '$lib/utils';
import { cShellMcpAskpassEndpoint } from '$lib/constants';

export interface LocalShellAskpassPrompt {
  /** id of the prompt, sent back with the answer */
  id: string;
  /** text of sudo, e.g. "[sudo] password for user:" */
  prompt: string;
}

/**
 * sudo password prompts of the commands run by the local shell MCP server. The tool call
 * names itself with the id sent in the _meta of tools/call; the password only travels to
 * this same server, which hands it to sudo and never to the model.
 */
export class LocalShellAskpassService {
  /** The prompt sudo left pending in the call, or null. */
  static async mPending(pCallId: string): Promise<LocalShellAskpassPrompt | null> {
    const dAnswer = await apiFetchWithParams<{ pending: LocalShellAskpassPrompt | null }>(
      cShellMcpAskpassEndpoint,
      { call: pCallId },
      { authOnly: true, cache: 'no-store' }
    );
    return dAnswer.pending ?? null;
  }

  /** Sends the password to sudo. */
  static async mAnswer(pCallId: string, pPromptId: string, pPassword: string): Promise<void> {
    await apiFetch(cShellMcpAskpassEndpoint, {
      method: 'POST',
      cache: 'no-store',
      body: JSON.stringify({ call: pCallId, id: pPromptId, password: pPassword })
    });
  }

  /** Makes sudo fail as if no password had been typed. */
  static async mCancel(pCallId: string, pPromptId: string): Promise<void> {
    await apiFetch(cShellMcpAskpassEndpoint, {
      method: 'POST',
      cache: 'no-store',
      body: JSON.stringify({ call: pCallId, id: pPromptId, cancel: true })
    });
  }
}
