/**
 * Server-sent events wire format, shared by the chat stream and the
 * /models/sse status feed (text/event-stream).
 */

// blank line between two events
export const cSseRecordSeparator = '\n\n';

// line break inside an event
export const cSseLineSeparator = '\n';

// data field prefix, the value follows after an optional space
export const cSseDataPrefix = 'data:';

// end-of-stream marker on the chat completion stream
export const cSseDoneMarker = '[DONE]';
