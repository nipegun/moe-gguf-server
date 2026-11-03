/**
 * Unified exports for all utility functions
 * Import utilities from '$lib/utils' for cleaner imports
 *
 * For browser-only utilities (pdf-processing, audio-recording, svg-to-png,
 * webp-to-png, process-uploaded-files, convert-files-to-extra), use:
 * import { ... } from '$lib/utils/browser-only'
 */

// API utilities
export { fGetAuthHeaders, fGetJsonHeaders, fSanitizeHeaders } from './api-headers';
export { fApiFetch, fApiFetchWithParams, fApiPost, type ApiFetchOptions } from './api-fetch';
export { fValidateApiKey } from './api-key-validation';

// Attachment utilities
export { fGetAttachmentDisplayItems, fIsMcpPrompt, fIsMcpResource } from './attachment-display';
export {
  fIsTextFile,
  fIsImageFile,
  fIsPdfFile,
  fIsAudioFile,
  fIsVideoFile
} from './attachment-type';

// Textarea utilities
export { default as fAutoResizeTextarea } from './autoresize-textarea';

// Branching utilities
export {
  fFilterByLeafNodeId,
  fFindMessageById,
  fFindLeafNode,
  fFindDescendantMessages,
  fGetMessageSiblings,
  fBuildSiblingInfoMap,
  fMessageGraphHasCycle
} from './branching';

// Code
export {
  fHighlightCode,
  fDetectIncompleteCodeBlock,
  fTrimCodePadding,
  type IncompleteCodeBlock
} from './code';

// Config helpers
export { fSetConfigValue, fGetConfigValue, fConfigToParameterRecord } from './config-helpers';

// CORS Proxy
export { fBuildProxiedUrl, fBuildProxiedHeaders } from './cors-proxy';

// URL utilities
export { fExtractRootDomain, fSanitizeExternalUrl, fCanonicalizeServerUrl } from './url';

// Progress helpers
export { fModelLoadFraction, fModelLoadProgressText, fQueueStatusText } from './progress';

// Conversation utilities
export {
  fCreateMessageCountMap,
  fGetMessageCount,
  fGroupSidebarConversations,
  type SidebarConversationGroups
} from './conversation-utils';

// Clipboard utilities
export {
  fCopyToClipboard,
  fCopyCodeToClipboard,
  fFormatMessageForClipboard,
  fParseClipboardContent,
  fHasClipboardAttachments
} from './clipboard';

// File preview utilities
export { fGetFileTypeLabel } from './file-preview';
export { fGetPreviewText, fGenerateConversationTitle } from './text';

// File type utilities
export {
  fGetFileTypeCategory,
  fGetFileTypeCategoryByExtension,
  fGetFileTypeByExtension,
  fIsFileTypeSupported
} from './file-type';

// Formatting utilities
export {
  fFormatFileSize,
  fFormatParameters,
  fFormatNumber,
  fFormatJsonPretty,
  fFormatTime,
  fFormatDisplayDate,
  fFormatPerformanceTime,
  fFormatAttachmentText
} from './formatters';

// IME utilities
export { fIsIMEComposing } from './is-ime-composing';

// LaTeX utilities
export { fMaskInlineLaTeX, fPreprocessLaTeX } from './latex-protection';

// Modality file validation utilities
export {
  fIsFileTypeSupportedByModel,
  fFilterFilesByModalities,
  fGenerateModalityErrorMessage
} from './modality-file-validation';

// Model name utilities
export { fNormalizeModelName, fIsValidModelName } from './model-names';

// Portal utilities
export { fPortalToBody } from './portal-to-body';

// Precision utilities
export { fNormalizeFloatingPoint, fNormalizeNumber } from './precision';

// Syntax highlighting utilities
export { fGetLanguageFromFilename } from './syntax-highlight-language';

// Text file utilities
export { fIsTextFileByName, fReadFileAsText, fIsLikelyTextFile } from './text-files';

// Debounce utilities
export { fDebounce } from './debounce';

// Sanitization utilities
export { fSanitizeKeyValuePairKey, fSanitizeKeyValuePairValue } from './sanitize';

// Image error fallback utilities
export { fCreateImageErrorFallback } from './image-error-fallback';

// Safe HTML and URL helpers for untrusted content
export { fEscapeHtml, fIsSafeLinkUrl, fIsHttpUrl, fIsDataImageUrl } from './safe-url';

// SSE-with-JSON stream iterator (used by built-in tool streaming, decoupled
// from chat.service.ts which embeds its own SSE parser for resume support)
export { fParseSseJsonStream, type SseJsonEvent } from './sse';

// Fingerprint of a text
export { fFingerprint } from './fingerprint';

// MCP utilities
export {
  fDetectMcpTransportFromUrl,
  fParseMcpServerSettings,
  fGetMcpLogLevelIcon,
  fGetMcpLogLevelClass,
  fIsImageMimeType,
  fParseResourcePath,
  fGetDisplayName,
  fGetResourceDisplayName,
  fIsCodeResource,
  fIsImageResource,
  fGetResourceIcon,
  fGetResourceTextContent,
  fGetResourceBlobContent,
  fDownloadResourceContent,
  fMcpResourceKey
} from './mcp';

// URI Template utilities
export {
  fExtractTemplateVariables,
  fExpandTemplate,
  fIsTemplateComplete,
  fNormalizeResourceUri,
  type UriTemplateVariable
} from './uri-template';

// Data URL utilities
export { fCreateBase64DataUrl } from './data-url';

// Header utilities
export { fParseHeadersToArray, fSerializeHeaders } from './headers';

// Agentic content utilities (structured section derivation)
export {
  fDeriveAgenticSections,
  fBuildAssistantRawOutput,
  fParseToolResultWithImages,
  fSplitSearchSummaryList,
  fHasAgenticContent,
  fClassifyToolResult,
  type AgenticSection,
  type ToolResultLine
} from './agentic';

// Line-level unified diff for tool result rendering (`edit_file` block)
export {
  fComputeLineDiff,
  fPrefixFor,
  fRenderUnifiedDiff,
  type DiffLine
} from './compute-line-diff';

// Partial-incremental JSON parser for streaming tool arguments
export { fParsePartialJsonArgs } from './parse-partial-json-args';

// `exec_shell_command` result parsing
export { fParseExecShellCommandError } from './parse-exec-shell-error';
export {
  fParseExecShellCommandExitStatus,
  fIsExitCodeSummaryLine,
  type ExecShellExitStatus
} from './parse-exec-shell-status';

// Search-result parsing (web-search / fetch MCP tools)
export {
  aSupportedWebSearchToolNames,
  fExtractSearchResults,
  fExtractSearchQuery,
  fFaviconForUrl,
  fIsWebSearchToolName,
  type SearchResult
} from './search-results';

// Cache utilities
export { TTLCache, ReactiveTTLMap, type TTLCacheOptions } from './cache-ttl';

// Redaction utilities
export { fRedactValue } from './redact';

// Request inspection utilities
export {
  fGetRequestUrl,
  fGetRequestMethod,
  fGetRequestBody,
  fSummarizeRequestBody,
  fFormatDiagnosticErrorMessage,
  fExtractJsonRpcMethods,
  type RequestBodySummary
} from './request-helpers';

// Abort signal utilities
export {
  fThrowIfAborted,
  fIsAbortError,
  fCreateLinkedController,
  fCreateTimeoutSignal,
  fWithAbortSignal
} from './abort';

// Tool-call meta utilities. Parsers for each built-in tool live next to
// their renderer family under
// `src/lib/components/app/chat/ChatMessages/ChatMessage/ChatMessageToolCall/parsers/`.
// This module only carries the helpers that genuinely cross tool
// boundaries (currently: parsing the tool-result blob into a JSON
// object).
export { fTryParseToolResultObject } from './tool-call-meta';

// Per-tool UI metadata (label + icon) used by the tool-call chrome.
// Re-exported through $lib/utils so renderer components can read the
// label without depending on $lib/constants directly.
export { fGetBuiltinToolUi, type BuiltinToolUiEntry } from '$lib/constants/built-in-tools';

// Cryptography utilities

export { fUuid } from './uuid';

// CSS utilities
export { fRemToPx } from './css';

// Global search of the sidebar
export {
  fNormalizeSearchText,
  fSearchTokens,
  fSearchScore,
  fSearchSnippet,
  fRevealSearchTarget
} from './global-search';
