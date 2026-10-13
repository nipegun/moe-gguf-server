import { base } from '$app/paths';
import { fGetJsonHeaders } from '$lib/utils/api-headers';
import { fParseSseJsonStream, type SseJsonEvent } from '$lib/utils/sse';
import { apiFetch } from '$lib/utils';
import { dApiTools } from '$lib/constants';
import { ToolResponseField } from '$lib/enums';
import type { ToolExecutionResult, ServerBuiltinToolInfo } from '$lib/types';

export class ToolsService {
  /**
   * Fetch the list of built-in tools from the server.
   *
   * @returns Array of tool definitions in OpenAI-compatible format
   */
  static async mList(): Promise<ServerBuiltinToolInfo[]> {
    return apiFetch<ServerBuiltinToolInfo[]>(dApiTools.LIST);
  }

  /**
   * Execute a built-in tool on the server.
   */
  static async mExecuteTool(
    pToolName: string,
    pParams: Record<string, unknown>,
    pSignal?: AbortSignal
  ): Promise<ToolExecutionResult> {
    const cResult = await apiFetch<Record<string, unknown>>(dApiTools.EXECUTE, {
      method: 'POST',
      body: JSON.stringify({ tool: pToolName, params: pParams }),
      signal: pSignal
    });

    if (ToolResponseField.ERROR in cResult) {
      return { content: String(cResult[ToolResponseField.ERROR]), isError: true };
    }

    if (ToolResponseField.PLAIN_TEXT in cResult) {
      return { content: String(cResult[ToolResponseField.PLAIN_TEXT]), isError: false };
    }

    return { content: JSON.stringify(cResult), isError: false };
  }

  /**
   * Stream a built-in tool's output chunks from the server. The server
   * `POST /tools` endpoint with `{stream: true}` emits `data: {"chunk": "..."}`
   * events followed by a terminal `data: {"done": true}` (optionally with
   * `error`). Yields the chunk string for each partial event.
   *
   * The terminal event's `error` field, if present, is yielded as a final
   * synthetic chunk prefixed with an error marker so the accumulated content
   * already carries the failure context for the caller.
   *
   * Throws synchronously if the server rejects the request (e.g. tool does
   * not support streaming, or 4xx/5xx response). The HTTP fetch goes through
   * a minimal text/event-stream reader since the chat SSE parser in
   * chat.service.ts embeds extra resume logic that is unnecessary here.
   */
  static async *mStreamTool(
    pToolName: string,
    pParams: Record<string, unknown>,
    pSignal?: AbortSignal
  ): AsyncGenerator<ToolStreamEvent> {
    const cHeaders = fGetJsonHeaders();
    const cResponse = await fetch(`${base}${dApiTools.EXECUTE}`, {
      method: 'POST',
      headers: cHeaders,
      body: JSON.stringify({ tool: pToolName, params: pParams, stream: true }),
      signal: pSignal
    });

    if (!cResponse.ok || !cResponse.body) {
      const cDetail = await fFormatNonOkResponse(cResponse);
      throw new Error(cDetail);
    }

    const cIterator = fParseSseJsonStream<ToolServerEvent>(cResponse, pSignal);

    while (true) {
      const cNext: IteratorResult<SseJsonEvent<ToolServerEvent>> = await cIterator.next();
      if (cNext.done) return;
      const cEvent = cNext.value.data;

      if (cEvent.chunk !== undefined) {
        yield { chunk: cEvent.chunk, done: false };
      }
      if (cEvent.done) {
        yield { chunk: null, done: true, error: cEvent.error };
        return;
      }
    }
  }
}

/**
 * One event from streaming a tool's output.
 *   - During execution: `chunk` is a non-empty text fragment, `done: false`.
 *   - On terminal event: `done: true`, `error` populated if the call failed,
 *     and `chunk` is null.
 */
export interface ToolStreamEvent {
  chunk: string | null;
  done: boolean;
  error?: string;
}

/** Wire shape of one SSE event from `POST /tools?stream=true`. */
interface ToolServerEvent {
  chunk?: string;
  done?: boolean;
  error?: string;
}

async function fFormatNonOkResponse(pResponse: Response): Promise<string> {
  const cStatus = `${pResponse.status} ${pResponse.statusText}`.trim();
  try {
    const cErrBody = (await pResponse.clone().json()) as { error?: string; message?: string };
    if (cErrBody?.error) return `${cStatus}: ${cErrBody.error}`;
    if (cErrBody?.message) return `${cStatus}: ${cErrBody.message}`;
  } catch (vError) {
    console.error('[tools] Non-JSON error response, falling back to raw text:', vError);
    try {
      const cText = await pResponse.text();
      if (cText.trim()) return `${cStatus}: ${cText.trim()}`;
    } catch (vError) {
      console.error('[tools] Failed to read error response as text:', vError);
    }
  }
  return cStatus || `HTTP ${pResponse.status}`;
}
