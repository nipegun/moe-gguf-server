import type { ErrorDialogType } from '$lib/enums';
import type { ApiChatCompletionToolCall, ApiQueueStatus } from './api';
import type { DatabaseMessage, DatabaseMessageExtra } from './database';

export interface ChatUploadedFile {
  id: string;
  name: string;
  size: number;
  type: string;
  file: File;
  preview?: string;
  textContent?: string;
  mcpPrompt?: {
    serverName: string;
    promptName: string;
    arguments?: Record<string, string>;
  };
  isLoading?: boolean;
  loadError?: string;
}

export interface ChatAttachmentDisplayItem {
  id: string;
  name: string;
  size?: number;
  preview?: string;
  isImage: boolean;
  isLoading?: boolean;
  loadError?: string;
  uploadedFile?: ChatUploadedFile;
  attachment?: DatabaseMessageExtra;
  attachmentIndex?: number;
  textContent?: string;
}

export interface ChatMessageSiblingInfo {
  message: DatabaseMessage;
  siblingIds: string[];
  currentIndex: number;
  totalSiblings: number;
}

export interface ChatMessagePromptProgress {
  cache: number;
  processed: number;
  time_ms: number;
  total: number;
}

export interface ChatMessageTimings {
  cache_n?: number;
  predicted_ms?: number;
  predicted_n?: number;
  prompt_ms?: number;
  prompt_n?: number;
  agentic?: ChatMessageAgenticTimings;
}

export interface ChatMessageAgenticTimings {
  turns: number;
  toolCallsCount: number;
  toolsMs: number;
  toolCalls?: ChatMessageToolCallTiming[];
  perTurn?: ChatMessageAgenticTurnStats[];
  llm: {
    predicted_n: number;
    predicted_ms: number;
    prompt_n: number;
    prompt_ms: number;
  };
}

export interface ChatMessageAgenticTurnStats {
  turn: number;
  llm: {
    predicted_n: number;
    predicted_ms: number;
    prompt_n: number;
    prompt_ms: number;
  };
  toolCalls: ChatMessageToolCallTiming[];
  toolsMs: number;
}

export interface ChatMessageToolCallTiming {
  name: string;
  duration_ms: number;
  success: boolean;
}

/**
 * Callbacks for streaming chat responses (used by both agentic and non-agentic paths)
 */
export interface ChatStreamCallbacks {
  onChunk?: (pChunk: string) => void;
  onReasoningChunk?: (pChunk: string) => void;
  onToolCallsStreaming?: (pToolCalls: ApiChatCompletionToolCall[]) => void;
  onAttachments?: (pMessageId: string, pExtras: DatabaseMessageExtra[]) => void;
  onModel?: (pModel: string) => void;
  onCompletionId?: (pId: string) => void;
  onTimings?: (pTimings?: ChatMessageTimings, pPromptProgress?: ChatMessagePromptProgress) => void;
  /** Queue status while the request waits for a free slot; null when it starts or ends */
  onQueue?: (pStatus: ApiQueueStatus | null) => void;
  onAssistantTurnComplete?: (
    pContent: string,
    pReasoningContent: string | undefined,
    pTimings: ChatMessageTimings | undefined,
    pToolCalls: ApiChatCompletionToolCall[] | undefined
  ) => Promise<void>;
  createToolResultMessage?: (
    pToolCallId: string,
    pContent: string,
    pExtras?: DatabaseMessageExtra[]
  ) => Promise<DatabaseMessage>;
  updateToolResultMessage?: (
    pMessageId: string,
    pContent: string,
    pExtras?: DatabaseMessageExtra[]
  ) => Promise<void>;
  createAssistantMessage?: () => Promise<DatabaseMessage>;
  onFlowComplete?: (pTimings?: ChatMessageTimings) => void;
  onError?: (pError: Error) => void;
  onTurnComplete?: (pIntermediateTimings: ChatMessageTimings) => void;
}

/**
 * Error dialog state for displaying server/timeout errors
 */
export interface ErrorDialogState {
  type: ErrorDialogType;
  message: string;
  contextInfo?: { n_prompt_tokens: number; n_ctx: number };
}

/**
 * Router mode: the model of a chat is no longer where it was when a message is sent to it
 */
export interface ModelMissingDialogState {
  /** model of the chat */
  model: string;
  /** GGUF file it was last sent with, when the chat knows it */
  path: string | null;
  /** model loaded now, which the message can be sent with instead; null when none is */
  loadedModel: string | null;
}

/**
 * Live processing stats during prompt evaluation
 */
export interface LiveProcessingStats {
  tokensProcessed: number;
  totalTokens: number;
  timeMs: number;
  tokensPerSecond: number;
  etaSecs?: number;
}

/**
 * Live generation stats during token generation
 */
export interface LiveGenerationStats {
  tokensGenerated: number;
  timeMs: number;
  tokensPerSecond: number;
}

/**
 * Options for getting attachment display items
 */
export interface AttachmentDisplayItemsOptions {
  uploadedFiles?: ChatUploadedFile[];
  attachments?: DatabaseMessageExtra[];
}

/**
 * Result of file processing operation
 */
export interface FileProcessingResult {
  extras: DatabaseMessageExtra[];
  emptyFiles: string[];
}
