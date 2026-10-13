import {
  cSseDataPrefix,
  cSseDoneMarker,
  cSseLineSeparator,
  cSseRecordSeparator
} from '$lib/constants';

/**
 * Minimal SSE-with-JSON stream iterator.
 *
 * Yields one event per `\n\n`-separated record. Each event payload is the
 * decoded `data:` field after JSON-parsing. A `[DONE]` sentinel terminates
 * the stream early. Malformed records - any record whose `data:` payload
 * fails `JSON.parse` - are skipped silently: usually a transient mid-stream
 * fault that the caller should not have to special-case, and the noise of
 * logging every occurrence on long-running streams outweighs the diagnostic
 * value.
 *
 * Less ambitious than ChatService.handleStreamResponse (no resume, no byte
 * offset tracking) - suitable for one-shot streams like `/tools?stream=true`
 * where the consumer just reads chunks until done.
 */

export interface SseJsonEvent<T = unknown> {
  data: T;
}

export async function* fParseSseJsonStream<T = unknown>(
  pResponse: Response,
  pSignal?: AbortSignal
): AsyncGenerator<SseJsonEvent<T>> {
  const cReader = pResponse.body?.getReader();
  if (!cReader) return;

  const cDecoder = new TextDecoder();
  let vBuffer = '';

  try {
    while (true) {
      if (pSignal?.aborted) return;

      const { done: cDone, value: cValue } = await cReader.read();
      if (cDone) break;

      vBuffer += cDecoder.decode(cValue, { stream: true });
      const cRecords = vBuffer.split(cSseRecordSeparator);
      vBuffer = cRecords.pop() ?? '';

      for (const cRecord of cRecords) {
        if (!cRecord) continue;
        for (const cLine of cRecord.split(cSseLineSeparator)) {
          if (!cLine.startsWith(cSseDataPrefix)) continue;
          const cPayload = cLine.slice(cSseDataPrefix.length).trim();
          if (cPayload === cSseDoneMarker) return;
          if (!cPayload) continue;
          try {
            yield { data: JSON.parse(cPayload) as T };
          } catch {
            // Skip silently per the function contract above.
          }
        }
      }
    }
  } finally {
    try {
      cReader.releaseLock();
    } catch (vError) {
      console.error('[sse] failed to release reader lock:', vError);
    }
  }
}
