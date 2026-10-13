import {
  AgenticSectionType,
  AttachmentType,
  ContinueIntentKind,
  MessageRole,
  ToolResultKind
} from '$lib/enums';
import {
  cAttachmentSavedRegex,
  cMarkdownAtxHeadingRegex,
  cMarkdownBoldRegex,
  cMarkdownBlockquoteRegex,
  cMarkdownCodeFenceRegex,
  cMarkdownLinkRegex,
  cMarkdownListBulletRegex,
  cMarkdownListNumberedRegex,
  cMarkdownTableSeparatorRegex,
  cNewline,
  cReasoningTags,
  cSearchSummarySeparator,
  cSearchSummaryTotalRegex,
  cToolResultJsonOpenRegex
} from '$lib/constants';
import type { ApiChatCompletionToolCall } from '$lib/types/api';
import type {
  DatabaseMessage,
  DatabaseMessageExtra,
  DatabaseMessageExtraImageFile
} from '$lib/types/database';

/**
 * Represents a parsed section of agentic content for display
 */
export interface AgenticSection {
  type: AgenticSectionType;
  content: string;
  toolName?: string;
  toolArgs?: string;
  toolResult?: string;
  toolResultExtras?: DatabaseMessageExtra[];
  /** ID of the model-side tool call (matches tool_calls[i].id). Lets
   *  downstream consumers correlate a section with the agentic loop's
   *  currently-executing tool, e.g. to drive live-streaming UI state
   *  by matching against agenticStore.executingToolCallId. */
  toolCallId?: string;
  wasInterrupted?: boolean;
}

/**
 * Represents a tool result line that may reference an image attachment
 */
export type ToolResultLine = {
  text: string;
  image?: DatabaseMessageExtraImageFile;
};

/**
 * Derives display sections from a single assistant message and its direct tool results.
 *
 * @param pMessage - The assistant message
 * @param pToolMessages - Tool result messages for this assistant's tool_calls
 * @param pStreamingToolCalls - Partial tool calls during streaming (not yet persisted)
 */
function fDeriveSingleTurnSections(
  pMessage: DatabaseMessage,
  pToolMessages: DatabaseMessage[] = [],
  pStreamingToolCalls: ApiChatCompletionToolCall[] = [],
  pIsStreaming: boolean = false
): AgenticSection[] {
  const aSections: AgenticSection[] = [];

  // 1. Reasoning content (from dedicated field)
  if (pMessage.reasoningContent) {
    const cToolCalls = fParseToolCalls(pMessage.toolCalls);
    const cHasContentAfterReasoning =
      !!pMessage.content?.trim() || cToolCalls.length > 0 || pStreamingToolCalls.length > 0;
    const cIsPending = pIsStreaming && !cHasContentAfterReasoning;
    aSections.push({
      type: cIsPending ? AgenticSectionType.REASONING_PENDING : AgenticSectionType.REASONING,
      content: pMessage.reasoningContent,
      wasInterrupted: !pIsStreaming && !cHasContentAfterReasoning
    });
  }

  // 2. Text content
  if (pMessage.content?.trim()) {
    aSections.push({
      type: AgenticSectionType.TEXT,
      content: pMessage.content
    });
  }

  // 3. Persisted tool calls (from message.toolCalls field)
  const cToolCalls = fParseToolCalls(pMessage.toolCalls);

  // Index tool messages by toolCallId for O(1) lookup instead of O(n) find()
  const dToolMsgById = new Map<string, DatabaseMessage>();
  for (const cTm of pToolMessages) {
    if (cTm.toolCallId && !dToolMsgById.has(cTm.toolCallId)) {
      dToolMsgById.set(cTm.toolCallId, cTm);
    }
  }

  for (const cTc of cToolCalls) {
    const cResultMsg = cTc.id ? dToolMsgById.get(cTc.id) : undefined;
    // Only show as pending/loading if we're actively streaming; otherwise it's just a tool call without result
    const cType = cResultMsg
      ? AgenticSectionType.TOOL_CALL
      : pIsStreaming
        ? AgenticSectionType.TOOL_CALL_PENDING
        : AgenticSectionType.TOOL_CALL;
    aSections.push({
      type: cType,
      content: cResultMsg?.content || '',
      toolName: cTc.function?.name,
      toolArgs: cTc.function?.arguments,
      toolResult: cResultMsg?.content,
      toolResultExtras: cResultMsg?.extra,
      toolCallId: cTc.id
    });
  }

  // 4. Streaming tool calls (not yet persisted - currently being received)
  const sPersistedIds = new Set(cToolCalls.map((pT) => pT.id).filter(Boolean));
  for (const cTc of pStreamingToolCalls) {
    // Skip if already in persisted tool calls
    if (cTc.id && sPersistedIds.has(cTc.id)) continue;
    aSections.push({
      type: AgenticSectionType.TOOL_CALL_STREAMING,
      content: '',
      toolName: cTc.function?.name,
      toolArgs: cTc.function?.arguments,
      toolCallId: cTc.id
    });
  }

  return aSections;
}

/**
 * Derives display sections from structured message data.
 *
 * Handles both single-turn (one assistant + its tool results) and multi-turn
 * agentic sessions (multiple assistant + tool messages grouped together).
 *
 * When `toolMessages` contains continuation assistant messages (from multi-turn
 * agentic flows), they are processed in order to produce sections across all turns.
 *
 * @param pMessage - The first/anchor assistant message
 * @param pToolMessages - Tool result messages and continuation assistant messages
 * @param pStreamingToolCalls - Partial tool calls during streaming (not yet persisted)
 * @param pIsStreaming - Whether the message is currently being streamed
 */
export function fDeriveAgenticSections(
  pMessage: DatabaseMessage,
  pToolMessages: DatabaseMessage[] = [],
  pStreamingToolCalls: ApiChatCompletionToolCall[] = [],
  pIsStreaming: boolean = false
): AgenticSection[] {
  const cHasAssistantContinuations = pToolMessages.some((pM) => pM.role === MessageRole.ASSISTANT);

  if (!cHasAssistantContinuations) {
    return fDeriveSingleTurnSections(pMessage, pToolMessages, pStreamingToolCalls, pIsStreaming);
  }

  const aSections: AgenticSection[] = [];

  const cFirstTurnToolMsgs = fCollectToolMessages(pToolMessages, 0);
  aSections.push(...fDeriveSingleTurnSections(pMessage, cFirstTurnToolMsgs));

  let vI = cFirstTurnToolMsgs.length;

  while (vI < pToolMessages.length) {
    const cMsg = pToolMessages[vI];

    if (cMsg.role === MessageRole.ASSISTANT) {
      const cTurnToolMsgs = fCollectToolMessages(pToolMessages, vI + 1);
      const cIsLastTurn = vI + 1 + cTurnToolMsgs.length >= pToolMessages.length;

      aSections.push(
        ...fDeriveSingleTurnSections(
          cMsg,
          cTurnToolMsgs,
          cIsLastTurn ? pStreamingToolCalls : [],
          cIsLastTurn && pIsStreaming
        )
      );

      vI += 1 + cTurnToolMsgs.length;
    } else {
      vI++;
    }
  }

  return aSections;
}

/**
 * Build the raw text representation shown in the "raw output" view of an
 * assistant message. Each section is formatted as it would appear in the
 * model-facing transcript, joined by blank lines.
 */
export function fBuildAssistantRawOutput(pSections: AgenticSection[]): string {
  const aParts: string[] = [];

  for (const cSection of pSections) {
    switch (cSection.type) {
      case AgenticSectionType.REASONING:
      case AgenticSectionType.REASONING_PENDING:
        aParts.push(`${cReasoningTags.START}${cNewline}${cSection.content}${cReasoningTags.END}`);
        break;

      case AgenticSectionType.TEXT:
        aParts.push(cSection.content);
        break;

      case AgenticSectionType.TOOL_CALL:
      case AgenticSectionType.TOOL_CALL_PENDING:
      case AgenticSectionType.TOOL_CALL_STREAMING: {
        const dCallObj: Record<string, unknown> = { name: cSection.toolName };

        if (cSection.toolArgs) {
          try {
            dCallObj.arguments = JSON.parse(cSection.toolArgs);
          } catch {
            dCallObj.arguments = cSection.toolArgs;
          }
        }

        aParts.push(JSON.stringify(dCallObj, null, 2));

        if (cSection.toolResult) {
          aParts.push(`${cNewline}${cSection.toolResult}`);
        }

        break;
      }
    }
  }

  return aParts.join(`${cNewline}${cNewline}`);
}

/**
 * Collect consecutive tool messages starting at `startIndex`.
 */
function fCollectToolMessages(
  pMessages: DatabaseMessage[],
  pStartIndex: number
): DatabaseMessage[] {
  const aResult: DatabaseMessage[] = [];

  for (let vI = pStartIndex; vI < pMessages.length; vI++) {
    if (pMessages[vI].role === MessageRole.TOOL) {
      aResult.push(pMessages[vI]);
    } else {
      break;
    }
  }

  return aResult;
}

/**
 * Split a tool-result blob into a list and an optional "Total matches: N"
 * summary. Both file-glob and grep tools emit this format on the server:
 *
 *   <matches>
 *   ---
 *   Total matches: 42
 *
 * Returns the lines and exposes a callback for capturing the total so each
 * caller can stash it on its own meta type without taking a return-tuple.
 */
export function fSplitSearchSummaryList(
  pText: string,
  pCaptureTotal: (pN: number) => void
): { lines: string[] } {
  const cSeparatorIndex = pText.indexOf(cSearchSummarySeparator);
  const cMatchesText = cSeparatorIndex === -1 ? pText : pText.slice(0, cSeparatorIndex);
  const cSummaryText =
    cSeparatorIndex === -1 ? '' : pText.slice(cSeparatorIndex + cSearchSummarySeparator.length);

  const cTotalMatch = cSummaryText.match(cSearchSummaryTotalRegex);
  if (cTotalMatch) {
    pCaptureTotal(parseInt(cTotalMatch[1], 10));
  }

  const cLines = cMatchesText
    .split(cNewline)
    .map((pLine) => pLine.trim())
    .filter((pLine) => pLine.length > 0);

  return { lines: cLines };
}

/** Bounded cache for parseToolResultWithImages results. */
const cToolResultLinesCacheMaxSize = 32;
const dToolResultLinesCache = new Map<string, ToolResultLine[]>();

/**
 * Parse tool result text into lines, matching image attachments by name.
 * Memoized: called per render during streaming on unchanged tool result
 * strings with unchanged extras.
 */
export function fParseToolResultWithImages(
  pToolResult: string,
  pExtras?: DatabaseMessageExtra[]
): ToolResultLine[] {
  // Cache key includes image attachment names so we recompute when
  // attachments change, even if the count stays the same.
  const cImageNames = (pExtras ?? [])
    .filter((pE): pE is DatabaseMessageExtraImageFile => pE.type === AttachmentType.IMAGE)
    .map((pE) => pE.name)
    .join(cNewline);
  const cCacheKey = `${cImageNames}:${pToolResult}`;
  const cCached = dToolResultLinesCache.get(cCacheKey);
  if (cCached !== undefined) return cCached;

  const cLines = pToolResult.split(cNewline);
  const cResult = cLines.map((pLine) => {
    const cMatch = pLine.match(cAttachmentSavedRegex);
    if (!cMatch || !pExtras) return { text: pLine };

    const cAttachmentName = cMatch[1];
    const cImage = pExtras.find(
      (pE): pE is DatabaseMessageExtraImageFile =>
        pE.type === AttachmentType.IMAGE && pE.name === cAttachmentName
    );

    return { text: pLine, image: cImage };
  });

  if (dToolResultLinesCache.size >= cToolResultLinesCacheMaxSize) {
    dToolResultLinesCache.delete(dToolResultLinesCache.keys().next().value!);
  }
  dToolResultLinesCache.set(cCacheKey, cResult);

  return cResult;
}

/** Bounded cache for classifyToolResult results. */
const cClassifyCacheMaxSize = 32;
const dClassifyCache = new Map<string, ToolResultKind>();

/**
 * Pick a renderer tier for a tool's result content.
 *
 *   json     - trimmed content starts with `{` or `[` and parses cleanly.
 *   markdown - content shows structural markdown markers (headers, code
 *              fences, links, lists, blockquotes, tables) and should render
 *              through MarkdownContent for proper formatting.
 *   text     - everything else, rendered as plain text lines (with image
 *              attachment resolution as a side effect).
 * Memoized: called per render during streaming on unchanged content.
 */
export function fClassifyToolResult(pContent: string | undefined): ToolResultKind {
  if (!pContent) return ToolResultKind.TEXT;

  const cCached = dClassifyCache.get(pContent);
  if (cCached !== undefined) return cCached;

  const cTrimmed = pContent.trim();
  if (!cTrimmed) return ToolResultKind.TEXT;

  let vResult: ToolResultKind = ToolResultKind.TEXT;

  // Strongest signal: JSON object/array round-trips through JSON.parse.
  if (cToolResultJsonOpenRegex.test(cTrimmed)) {
    try {
      JSON.parse(cTrimmed);
      vResult = ToolResultKind.JSON;
    } catch (vError) {
      console.error('[agentic] tool result looked like JSON but failed to parse:', vError);
    }
  }

  if (vResult === ToolResultKind.TEXT && fLooksLikeMarkdown(cTrimmed)) {
    vResult = ToolResultKind.MARKDOWN;
  }

  if (dClassifyCache.size >= cClassifyCacheMaxSize) {
    dClassifyCache.delete(dClassifyCache.keys().next().value!);
  }
  dClassifyCache.set(pContent, vResult);

  return vResult;
}

/**
 * Heuristic detector for "is this content a markdown document rather than
 * plain text?". True when at least one well-known structural marker shows
 * up - headers, code fences, links, bold, lists, blockquotes, tables.
 * Each marker is specific enough that plain tool-output prose rarely
 * trips it, but plain text starting with `# 5` will - acceptable false
 * positive for the gain in formatting for tool results like search
 * summaries that come back already-mardown.
 */
function fLooksLikeMarkdown(pContent: string): boolean {
  // Code fences are unambiguous - triple backticks or tildes at line start.
  if (cMarkdownCodeFenceRegex.test(pContent)) return true;

  const cLines = pContent.split(cNewline);

  for (const cLine of cLines) {
    if (cMarkdownAtxHeadingRegex.test(cLine)) return true;
    if (cMarkdownBlockquoteRegex.test(cLine)) return true;
    if (cMarkdownListBulletRegex.test(cLine)) return true;
    if (cMarkdownListNumberedRegex.test(cLine)) return true;
  }

  // Inline structural markers anywhere in the body.
  if (cMarkdownLinkRegex.test(pContent)) return true;
  if (cMarkdownBoldRegex.test(pContent)) return true;

  // Tables: a pipe-bearing header line followed by a separator row.
  if (cLines.length >= 2) {
    const cHead = cLines[0];
    const cSep = cLines[1];

    if (cHead.includes('|') && cMarkdownTableSeparatorRegex.test(cSep)) return true;
  }

  return false;
}

/** Bounded cache for parsed tool-call JSON blobs. */
const cToolCallsCacheMaxSize = 64;
const dToolCallsParseCache = new Map<string, ApiChatCompletionToolCall[]>();

/**
 * Safely parse the toolCalls JSON string from a DatabaseMessage.
 * Memoized: the same JSON string is re-parsed on every render during
 * streaming, which is wasted CPU since tool calls don't change mid-stream.
 */
function fParseToolCalls(pToolCallsJson?: string): ApiChatCompletionToolCall[] {
  if (!pToolCallsJson) return [];

  const cCached = dToolCallsParseCache.get(pToolCallsJson);
  if (cCached) return cCached;

  let aResult: ApiChatCompletionToolCall[];
  try {
    const cParsed = JSON.parse(pToolCallsJson);
    aResult = Array.isArray(cParsed) ? cParsed : [];
  } catch {
    aResult = [];
  }

  if (dToolCallsParseCache.size >= cToolCallsCacheMaxSize) {
    dToolCallsParseCache.delete(dToolCallsParseCache.keys().next().value!);
  }
  dToolCallsParseCache.set(pToolCallsJson, aResult);

  return aResult;
}

/**
 * Check if a message has agentic content (tool calls or is part of an agentic flow).
 */
export function fHasAgenticContent(
  pMessage: DatabaseMessage,
  pToolMessages: DatabaseMessage[] = []
): boolean {
  if (pMessage.toolCalls) {
    const cTc = fParseToolCalls(pMessage.toolCalls);

    if (cTc.length > 0) return true;
  }

  return pToolMessages.length > 0;
}

/**
 * Classification of how a Continue click on an assistant message should resume
 * generation. The caller dispatches the resume path based on this value.
 *
 *   append_text  -> the target is a plain text turn, resume with
 *                   continue_final_message and rehydrate the persisted
 *                   tool_calls and attachments through the regular DB to API
 *                   message converter.
 *   rerun_turn   -> the target carries tool_calls that were never resolved by
 *                   tool result messages. The agentic stream was cut mid turn,
 *                   so we drop the target and rerun the loop from the previous
 *                   history. truncateAfter is the last kept index, inclusive.
 *   next_turn    -> the target's tool_calls were already resolved by trailing
 *                   tool results. Hand the history up to and including the
 *                   last consecutive tool result back to the agentic loop so it
 *                   starts the next turn naturally. truncateAfter points at
 *                   that last tool result.
 */
export type ContinueIntent =
  | { kind: ContinueIntentKind.APPEND_TEXT }
  | { kind: ContinueIntentKind.RERUN_TURN; truncateAfter: number }
  | { kind: ContinueIntentKind.NEXT_TURN; truncateAfter: number };

/**
 * Decide how a Continue click on messages[idx] should resume generation.
 * Pure function over the persisted history snapshot.
 */
export function fClassifyContinueIntent(
  pMessages: DatabaseMessage[],
  pIdx: number
): ContinueIntent {
  const cTarget = pMessages[pIdx];

  // Defensive default: callers already filter by role, stay deterministic.
  if (!cTarget || cTarget.role !== MessageRole.ASSISTANT) {
    return { kind: ContinueIntentKind.APPEND_TEXT };
  }

  const cHasToolCalls = fParseToolCalls(cTarget.toolCalls).length > 0;
  if (!cHasToolCalls) {
    return { kind: ContinueIntentKind.APPEND_TEXT };
  }

  // Walk consecutive trailing tool results. The agentic loop only emits tool
  // messages directly after the assistant turn that owns them, so the first
  // non tool message marks the boundary.
  let vLastTrailingTool = pIdx;
  for (let vI = pIdx + 1; vI < pMessages.length; vI++) {
    if (pMessages[vI].role === MessageRole.TOOL) {
      vLastTrailingTool = vI;
    } else {
      break;
    }
  }

  if (vLastTrailingTool > pIdx) {
    return { kind: ContinueIntentKind.NEXT_TURN, truncateAfter: vLastTrailingTool };
  }

  return { kind: ContinueIntentKind.RERUN_TURN, truncateAfter: pIdx - 1 };
}
