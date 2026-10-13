/**
 * Unified exports for all utility functions
 * Import utilities from '$lib/utils' for cleaner imports
 *
 * For browser-only utilities (pdf-processing, audio-recording, svg-to-png,
 * webp-to-png, process-uploaded-files, convert-files-to-extra), use:
 * import { ... } from '$lib/utils/browser-only'
 */

// API utilities
export {
  fGetAuthHeaders as getAuthHeaders,
  fGetJsonHeaders as getJsonHeaders,
  fSanitizeHeaders as sanitizeHeaders
} from './api-headers';
export {
  fApiFetch as apiFetch,
  fApiFetchWithParams as apiFetchWithParams,
  fApiPost as apiPost,
  type ApiFetchOptions
} from './api-fetch';
export { fValidateApiKey as validateApiKey } from './api-key-validation';

// Attachment utilities
export {
  fGetAttachmentDisplayItems as getAttachmentDisplayItems,
  fIsMcpPrompt as isMcpPrompt,
  fIsMcpResource as isMcpResource
} from './attachment-display';
export {
  fIsTextFile as isTextFile,
  fIsImageFile as isImageFile,
  fIsPdfFile as isPdfFile,
  fIsAudioFile as isAudioFile,
  fIsVideoFile as isVideoFile
} from './attachment-type';

// Textarea utilities
export { default as autoResizeTextarea } from './autoresize-textarea';

// Branching utilities
export {
  fFilterByLeafNodeId as filterByLeafNodeId,
  fFindMessageById as findMessageById,
  fFindLeafNode as findLeafNode,
  fFindDescendantMessages as findDescendantMessages,
  fGetMessageSiblings as getMessageSiblings,
  fBuildSiblingInfoMap as buildSiblingInfoMap
} from './branching';

// Code
export {
  fHighlightCode as highlightCode,
  fDetectIncompleteCodeBlock as detectIncompleteCodeBlock,
  trimCodePadding,
  type IncompleteCodeBlock
} from './code';

// Config helpers
export {
  fSetConfigValue as setConfigValue,
  fGetConfigValue as getConfigValue,
  fConfigToParameterRecord as configToParameterRecord
} from './config-helpers';

// CORS Proxy
export {
  fBuildProxiedUrl as buildProxiedUrl,
  fBuildProxiedHeaders as buildProxiedHeaders
} from './cors-proxy';

// URL utilities
export {
  fExtractRootDomain as extractRootDomain,
  fSanitizeExternalUrl as sanitizeExternalUrl,
  fCanonicalizeServerUrl as canonicalizeServerUrl
} from './url';

// Progress helpers
export {
  fModelLoadFraction as modelLoadFraction,
  fModelLoadProgressText as modelLoadProgressText,
  fQueueStatusText as queueStatusText
} from './progress';

// Conversation utilities
export {
  fCreateMessageCountMap as createMessageCountMap,
  fGetMessageCount as getMessageCount
} from './conversation-utils';

// Clipboard utilities
export {
  fCopyToClipboard as copyToClipboard,
  fCopyCodeToClipboard as copyCodeToClipboard,
  fFormatMessageForClipboard as formatMessageForClipboard,
  fParseClipboardContent as parseClipboardContent,
  fHasClipboardAttachments as hasClipboardAttachments
} from './clipboard';

// File preview utilities
export { fGetFileTypeLabel as getFileTypeLabel } from './file-preview';
export {
  fGetPreviewText as getPreviewText,
  fGenerateConversationTitle as generateConversationTitle
} from './text';

// File type utilities
export {
  fGetFileTypeCategory as getFileTypeCategory,
  fGetFileTypeCategoryByExtension as getFileTypeCategoryByExtension,
  fGetFileTypeByExtension as getFileTypeByExtension,
  fIsFileTypeSupported as isFileTypeSupported
} from './file-type';

// Formatting utilities
export {
  fFormatFileSize as formatFileSize,
  fFormatParameters as formatParameters,
  fFormatNumber as formatNumber,
  fFormatJsonPretty as formatJsonPretty,
  fFormatTime as formatTime,
  fFormatPerformanceTime as formatPerformanceTime,
  fFormatAttachmentText as formatAttachmentText
} from './formatters';

// IME utilities
export { fIsIMEComposing as isIMEComposing } from './is-ime-composing';

// LaTeX utilities
export {
  fMaskInlineLaTeX as maskInlineLaTeX,
  fPreprocessLaTeX as preprocessLaTeX
} from './latex-protection';

// Modality file validation utilities
export {
  fIsFileTypeSupportedByModel as isFileTypeSupportedByModel,
  fFilterFilesByModalities as filterFilesByModalities,
  fGenerateModalityErrorMessage as generateModalityErrorMessage
} from './modality-file-validation';

// Model name utilities
export {
  fNormalizeModelName as normalizeModelName,
  fIsValidModelName as isValidModelName
} from './model-names';

// Portal utilities
export { fPortalToBody as portalToBody } from './portal-to-body';

// Precision utilities
export {
  fNormalizeFloatingPoint as normalizeFloatingPoint,
  fNormalizeNumber as normalizeNumber
} from './precision';

// Syntax highlighting utilities
export { fGetLanguageFromFilename as getLanguageFromFilename } from './syntax-highlight-language';

// Text file utilities
export {
  fIsTextFileByName as isTextFileByName,
  fReadFileAsText as readFileAsText,
  fIsLikelyTextFile as isLikelyTextFile
} from './text-files';

// Debounce utilities
export { fDebounce as debounce } from './debounce';

// Sanitization utilities
export {
  fSanitizeKeyValuePairKey as sanitizeKeyValuePairKey,
  fSanitizeKeyValuePairValue as sanitizeKeyValuePairValue
} from './sanitize';

// Image error fallback utilities
export { fCreateImageErrorFallback as createImageErrorFallback } from './image-error-fallback';

// Safe HTML and URL helpers for untrusted content
export {
  fEscapeHtml as escapeHtml,
  fIsSafeLinkUrl as isSafeLinkUrl,
  fIsHttpUrl as isHttpUrl,
  fIsDataImageUrl as isDataImageUrl
} from './safe-url';

// SSE-with-JSON stream iterator (used by built-in tool streaming, decoupled
// from chat.service.ts which embeds its own SSE parser for resume support)
export { fParseSseJsonStream as parseSseJsonStream, type SseJsonEvent } from './sse';

// MCP utilities
export {
  fDetectMcpTransportFromUrl as detectMcpTransportFromUrl,
  fParseMcpServerSettings as parseMcpServerSettings,
  fGetMcpLogLevelIcon as getMcpLogLevelIcon,
  fGetMcpLogLevelClass as getMcpLogLevelClass,
  fIsImageMimeType as isImageMimeType,
  fParseResourcePath as parseResourcePath,
  fGetDisplayName as getDisplayName,
  fGetResourceDisplayName as getResourceDisplayName,
  fIsCodeResource as isCodeResource,
  fIsImageResource as isImageResource,
  fGetResourceIcon as getResourceIcon,
  fGetResourceTextContent as getResourceTextContent,
  fGetResourceBlobContent as getResourceBlobContent,
  fDownloadResourceContent as downloadResourceContent
} from './mcp';

// URI Template utilities
export {
  fExtractTemplateVariables as extractTemplateVariables,
  fExpandTemplate as expandTemplate,
  fIsTemplateComplete as isTemplateComplete,
  fNormalizeResourceUri as normalizeResourceUri,
  type UriTemplateVariable
} from './uri-template';

// Data URL utilities
export { fCreateBase64DataUrl as createBase64DataUrl } from './data-url';

// Header utilities
export {
  fParseHeadersToArray as parseHeadersToArray,
  fSerializeHeaders as serializeHeaders
} from './headers';

// Agentic content utilities (structured section derivation)
export {
  fDeriveAgenticSections as deriveAgenticSections,
  fBuildAssistantRawOutput as buildAssistantRawOutput,
  fParseToolResultWithImages as parseToolResultWithImages,
  fSplitSearchSummaryList as splitSearchSummaryList,
  fHasAgenticContent as hasAgenticContent,
  fClassifyToolResult as classifyToolResult,
  type AgenticSection,
  type ToolResultLine
} from './agentic';

// Line-level unified diff for tool result rendering (`edit_file` block)
export {
  fComputeLineDiff as computeLineDiff,
  fPrefixFor as prefixFor,
  fRenderUnifiedDiff as renderUnifiedDiff,
  type DiffLine
} from './compute-line-diff';

// Partial-incremental JSON parser for streaming tool arguments
export { fParsePartialJsonArgs as parsePartialJsonArgs } from './parse-partial-json-args';

// `exec_shell_command` result parsing
export { fParseExecShellCommandError as parseExecShellCommandError } from './parse-exec-shell-error';
export {
  fParseExecShellCommandExitStatus as parseExecShellCommandExitStatus,
  fIsExitCodeSummaryLine as isExitCodeSummaryLine,
  type ExecShellExitStatus
} from './parse-exec-shell-status';

// Search-result parsing (web-search / fetch MCP tools)
export {
  aSupportedWebSearchToolNames as SUPPORTED_WEB_SEARCH_TOOL_NAMES,
  fExtractSearchResults as extractSearchResults,
  fExtractSearchQuery as extractSearchQuery,
  fFaviconForUrl as faviconForUrl,
  fIsWebSearchToolName as isWebSearchToolName,
  type SearchResult
} from './search-results';

// Cache utilities
export { TTLCache, ReactiveTTLMap, type TTLCacheOptions } from './cache-ttl';

// Redaction utilities
export { fRedactValue as redactValue } from './redact';

// Request inspection utilities
export {
  fGetRequestUrl as getRequestUrl,
  fGetRequestMethod as getRequestMethod,
  fGetRequestBody as getRequestBody,
  fSummarizeRequestBody as summarizeRequestBody,
  fFormatDiagnosticErrorMessage as formatDiagnosticErrorMessage,
  fExtractJsonRpcMethods as extractJsonRpcMethods,
  type RequestBodySummary
} from './request-helpers';

// Abort signal utilities
export {
  fThrowIfAborted as throwIfAborted,
  fIsAbortError as isAbortError,
  fCreateLinkedController as createLinkedController,
  fCreateTimeoutSignal as createTimeoutSignal,
  fWithAbortSignal as withAbortSignal
} from './abort';

// Tool-call meta utilities. Parsers for each built-in tool live next to
// their renderer family under
// `src/lib/components/app/chat/ChatMessages/ChatMessage/ChatMessageToolCall/parsers/`.
// This module only carries the helpers that genuinely cross tool
// boundaries (currently: parsing the tool-result blob into a JSON
// object).
export { fTryParseToolResultObject as tryParseToolResultObject } from './tool-call-meta';

// Per-tool UI metadata (label + icon) used by the tool-call chrome.
// Re-exported through $lib/utils so renderer components can read the
// label without depending on $lib/constants directly.
export {
  fGetBuiltinToolUi as getBuiltinToolUi,
  type BuiltinToolUiEntry
} from '$lib/constants/built-in-tools';

// Cryptography utilities

export { fUuid as uuid } from './uuid';

// CSS utilities
export { fRemToPx as remToPx } from './css';
