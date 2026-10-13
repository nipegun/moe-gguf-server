import { fTranslate } from '../i18n';
import { fGetAuthHeaders, fGetJsonHeaders } from '$lib/utils/api-headers';
import { fFormatAttachmentText } from '$lib/utils/formatters';
import { fIsAbortError } from '$lib/utils/abort';
import { fStreamIdentity } from '$lib/utils/stream-identity';
import { fApplyResponseLanguage } from '$lib/utils/response-language';
import {
  cAttachmentLabelPdfFile,
  cAttachmentLabelMcpPrompt,
  cAttachmentLabelMcpResource,
  dReasoningEffortTokens,
  cSettingsKeys,
  dApiChat,
  dApiSlots,
  cControlAction,
  cSseLineSeparator,
  cSseDataPrefix,
  cSseDoneMarker,
  cStreamVisibilityKickMs,
  cStreamResumeLocalstorageKeyPrefix,
  dApiStream
} from '$lib/constants';
import {
  AttachmentType,
  ContentPartType,
  FileTypeAudio,
  MessageRole,
  MimeTypeAudio,
  ReasoningFormat,
  StreamConnectionState
} from '$lib/enums';
import type {
  ApiChatMessageContentPart,
  ApiChatMessageData,
  ApiChatCompletionToolCall,
  ApiQueueStatus,
  ApiStreamSession
} from '$lib/types/api';
import type {
  AudioInputFormat,
  DatabaseMessageExtraMcpPrompt,
  DatabaseMessageExtraMcpResource
} from '$lib/types';
import { cModelsStore } from '$lib/stores/models.svelte';
import { cSettingsStore } from '../stores/settings.svelte';
import { fCapImageDataURLSize } from '../utils/cap-img-size';

function fGetAudioInputFormat(pMimeType: string): AudioInputFormat {
  const cNormalizedMimeType = pMimeType.trim().toLowerCase();

  if (
    cNormalizedMimeType === MimeTypeAudio.WAV ||
    cNormalizedMimeType === MimeTypeAudio.WAVE ||
    cNormalizedMimeType === MimeTypeAudio.X_WAV ||
    cNormalizedMimeType === MimeTypeAudio.X_WAVE ||
    cNormalizedMimeType === MimeTypeAudio.VND_WAVE ||
    cNormalizedMimeType === MimeTypeAudio.X_PN_WAV
  ) {
    return FileTypeAudio.WAV;
  }

  return FileTypeAudio.MP3;
}

interface ResumableStreamState {
  bytesReceived: number;
  updatedAt: number;

  // model frozen at POST time, lets a reload rebuild the exact conv::model identity the
  // server keyed the session under. null when the POST carried no explicit model
  model: string | null;
}

function fStreamStorageKey(pConversationId: string): string {
  return cStreamResumeLocalstorageKeyPrefix + pConversationId;
}

export class ChatService {
  /**
   *
   *
   * Title Generation
   *
   *
   */

  /**
   * Sends a streaming chat completion request for generating a chat title.
   * Delegates to `sendMessage` for fetch, SSE parsing, and error handling.
   *
   * @param pMessage - The single message to send (a user message containing the title generation prompt)
   * @param pModel - Optional model name to use (required in ROUTER mode)
   * @param pSignal - Optional AbortSignal to cancel the request
   * @returns {Promise<string>} The aggregated title text, or empty string if request failed
   * @static
   */
  static async mGenerateTitle(
    pMessage: ApiChatMessageData,
    pModel?: string | null,
    pSignal?: AbortSignal
  ): Promise<string> {
    let vTitleResponse = '';
    try {
      await ChatService.mSendMessage(
        [pMessage],
        {
          model: pModel || undefined,
          stream: true,
          custom: { chat_template_kwargs: { enable_thinking: false } },
          onChunk: (pChunk: string) => {
            vTitleResponse += pChunk;
          }
        },
        undefined,
        pSignal
      );
    } catch {
      return '';
    }
    return vTitleResponse;
  }

  /**
   *
   *
   * Messaging
   *
   *
   */

  /**
   * Sends a chat completion request to the mgs server.
   * Supports both streaming and non-streaming responses with comprehensive parameter configuration.
   * Automatically converts database messages with attachments to the appropriate API format.
   *
   * @param pMessages - Array of chat messages to send to the API (supports both ApiChatMessageData and DatabaseMessage with attachments)
   * @param pOptions - Configuration options for the chat completion request. See `SettingsChatServiceOptions` type for details.
   * @returns {Promise<string | void>} that resolves to the complete response string (non-streaming) or void (streaming)
   * @throws {Error} if the request fails or is aborted
   */
  static async mSendMessage(
    pMessages: ApiChatMessageData[] | (DatabaseMessage & { extra?: DatabaseMessageExtra[] })[],
    pOptions: SettingsChatServiceOptions = {},
    pConversationId?: string,
    pSignal?: AbortSignal
  ): Promise<string | void> {
    const {
      stream: cStream,
      onChunk: cOnChunk,
      onComplete: cOnComplete,
      onError: cOnError,
      onConnectionState: cOnConnectionState,
      onReasoningChunk: cOnReasoningChunk,
      onToolCallChunk: cOnToolCallChunk,
      onModel: cOnModel,
      onCompletionId: cOnCompletionId,
      onTimings: cOnTimings,
      onQueue: cOnQueue,
      // Tools for function calling
      tools: cTools,
      // Generation parameters
      temperature: cTemperature,
      max_tokens: cMaxTokens,
      // Sampling parameters
      dynatemp_range: cDynatempRange,
      dynatemp_exponent: cDynatempExponent,
      top_k: cTopK,
      top_p: cTopP,
      min_p: cMinP,
      xtc_probability: cXtcProbability,
      xtc_threshold: cXtcThreshold,
      typ_p: cTypP,
      // Penalty parameters
      repeat_last_n: cRepeatLastN,
      repeat_penalty: cRepeatPenalty,
      presence_penalty: cPresencePenalty,
      frequency_penalty: cFrequencyPenalty,
      dry_multiplier: cDryMultiplier,
      dry_base: cDryBase,
      dry_allowed_length: cDryAllowedLength,
      dry_penalty_last_n: cDryPenaltyLastN,
      // Other parameters
      samplers: cSamplers,
      backend_sampling: cBackendSampling,
      custom: cCustom,
      timings_per_token: cTimingsPerToken,
      // Config options
      disableReasoningParsing: cDisableReasoningParsing,
      excludeReasoningFromContext: cExcludeReasoningFromContext,
      enableThinking: cEnableThinking,
      reasoningEffort: cReasoningEffort,
      continueFinalMessage: cContinueFinalMessage
    } = pOptions;

    const aNormalizedMessages: ApiChatMessageData[] = (
      await Promise.all(
        pMessages.map((pMsg) => {
          if ('id' in pMsg && 'convId' in pMsg && 'timestamp' in pMsg) {
            const cDbMsg = pMsg as DatabaseMessage & { extra?: DatabaseMessageExtra[] };

            return ChatService.mConvertDbMessageToApiChatMessageData(cDbMsg);
          } else {
            return pMsg as ApiChatMessageData;
          }
        })
      )
    ).filter((pMsg: { role: ChatRole; content: string | ApiChatMessageContentPart[] }) => {
      // Filter out empty system messages
      if (pMsg.role === MessageRole.SYSTEM) {
        const cContent = typeof pMsg.content === 'string' ? pMsg.content : '';

        return cContent.trim().length > 0;
      }

      return true;
    });

    // Filter out image attachments if the model doesn't support vision
    if (pOptions.model && !cModelsStore.mModelSupportsVision(pOptions.model)) {
      aNormalizedMessages.forEach((pMsg) => {
        if (Array.isArray(pMsg.content)) {
          pMsg.content = pMsg.content.filter((pPart: ApiChatMessageContentPart) => {
            if (pPart.type === ContentPartType.IMAGE_URL) {
              console.info(
                `[ChatService] Skipping image attachment in message history (model "${pOptions.model}" does not support vision)`
              );

              return false;
            }

            return true;
          });
          // If only text remains and it's a single part, simplify to string
          if (
            pMsg.content.length === 1 &&
            pMsg.content[0].type === ContentPartType.TEXT &&
            typeof pMsg.content[0].text === 'string'
          ) {
            pMsg.content = pMsg.content[0].text;
          }
        }
      });
    }

    const dRequestBody: ApiChatCompletionRequest = {
      messages: aNormalizedMessages.map((pMsg: ApiChatMessageData) => {
        const dMapped: ApiChatCompletionRequest['messages'][0] = {
          role: pMsg.role,
          content: pMsg.content,
          tool_calls: pMsg.tool_calls,
          tool_call_id: pMsg.tool_call_id
        };
        // Include reasoning_content from the dedicated field
        if (!cExcludeReasoningFromContext && pMsg.reasoning_content) {
          dMapped.reasoning_content = pMsg.reasoning_content;
        }
        return dMapped;
      }),
      stream: cStream,
      // marks the request as sent by the interface: while it waits behind other requests (for example
      // ones sent directly to the API) the server streams its queue position
      mgs_client: 'webui',
      return_progress: cStream ? true : undefined,
      sse_ping_interval: cStream ? 1 : undefined,
      tools: cTools && cTools.length > 0 ? cTools : undefined
    };

    // Include model in request if provided (required in ROUTER mode)
    if (pOptions.model) {
      dRequestBody.model = pOptions.model;
    }

    dRequestBody.reasoning_format = cDisableReasoningParsing
      ? ReasoningFormat.NONE
      : ReasoningFormat.AUTO;

    const cReasoningBudgetTokens =
      cEnableThinking && cReasoningEffort ? (dReasoningEffortTokens[cReasoningEffort] ?? -1) : -1;

    // an explicit user choice injects the kwarg, otherwise it is omitted so
    // the server default applies (--reasoning flag or chat template)
    if (cEnableThinking !== undefined) {
      dRequestBody.chat_template_kwargs = {
        ...(dRequestBody.chat_template_kwargs ?? {}),
        enable_thinking: cEnableThinking
      };
    }

    if (cReasoningBudgetTokens >= 0) {
      dRequestBody.reasoning_budget_tokens = cReasoningBudgetTokens;
    }

    // arms the budget sampler so reasoning can be ended at runtime via the control endpoint
    dRequestBody.reasoning_control = true;

    if (cContinueFinalMessage) {
      dRequestBody.continue_final_message = true;
      dRequestBody.add_generation_prompt = false;
    }

    if (cTemperature !== undefined) dRequestBody.temperature = cTemperature;
    if (cMaxTokens !== undefined) {
      // Set max_tokens to -1 (infinite) when explicitly configured as 0 or null
      dRequestBody.max_tokens = cMaxTokens !== null && cMaxTokens !== 0 ? cMaxTokens : -1;
    }

    if (cDynatempRange !== undefined) dRequestBody.dynatemp_range = cDynatempRange;
    if (cDynatempExponent !== undefined) dRequestBody.dynatemp_exponent = cDynatempExponent;
    if (cTopK !== undefined) dRequestBody.top_k = cTopK;
    if (cTopP !== undefined) dRequestBody.top_p = cTopP;
    if (cMinP !== undefined) dRequestBody.min_p = cMinP;
    if (cXtcProbability !== undefined) dRequestBody.xtc_probability = cXtcProbability;
    if (cXtcThreshold !== undefined) dRequestBody.xtc_threshold = cXtcThreshold;
    if (cTypP !== undefined) dRequestBody.typ_p = cTypP;

    if (cRepeatLastN !== undefined) dRequestBody.repeat_last_n = cRepeatLastN;
    if (cRepeatPenalty !== undefined) dRequestBody.repeat_penalty = cRepeatPenalty;
    if (cPresencePenalty !== undefined) dRequestBody.presence_penalty = cPresencePenalty;
    if (cFrequencyPenalty !== undefined) dRequestBody.frequency_penalty = cFrequencyPenalty;
    if (cDryMultiplier !== undefined) dRequestBody.dry_multiplier = cDryMultiplier;
    if (cDryBase !== undefined) dRequestBody.dry_base = cDryBase;
    if (cDryAllowedLength !== undefined) dRequestBody.dry_allowed_length = cDryAllowedLength;
    if (cDryPenaltyLastN !== undefined) dRequestBody.dry_penalty_last_n = cDryPenaltyLastN;

    if (cSamplers !== undefined) {
      dRequestBody.samplers =
        typeof cSamplers === 'string'
          ? cSamplers.split(';').filter((pS: string) => pS.trim())
          : cSamplers;
    }

    if (cBackendSampling !== undefined) dRequestBody.backend_sampling = cBackendSampling;

    if (cTimingsPerToken !== undefined) dRequestBody.timings_per_token = cTimingsPerToken;

    if (cCustom) {
      try {
        const cCustomParams = typeof cCustom === 'string' ? JSON.parse(cCustom) : cCustom;
        Object.assign(dRequestBody, cCustomParams);
      } catch (vError) {
        console.warn('Failed to parse custom parameters:', vError);
      }
    }

    // Apply after custom JSON so a custom messages array also receives the instruction.
    dRequestBody.messages = fApplyResponseLanguage(
      dRequestBody.messages,
      cSettingsStore.mGetConfig(cSettingsKeys.RESPOND_IN_INTERFACE_LANGUAGE) === true
    );

    try {
      const dHeaders: Record<string, string> = { ...fGetJsonHeaders() };
      // tag streaming requests with the conversation id, this single header is the opt in for the
      // server side replay buffer and powers discoverActiveStream on tab reopen. with an explicit
      // model the ::model suffix keeps the per model session distinct
      if (cStream && pConversationId) {
        dHeaders['X-Conversation-Id'] = fStreamIdentity(pConversationId, pOptions.model);
        // persist the pending stream before the fetch: a reload during the model load or
        // the prompt processing must still find its way back to the session once it exists
        ChatService.mSaveStreamState(pConversationId, 0, pOptions.model ?? null);
      }

      const cResponse = await fetch(dApiChat.COMPLETIONS, {
        method: 'POST',
        headers: dHeaders,
        body: JSON.stringify(dRequestBody),
        signal: pSignal
      });

      if (!cResponse.ok) {
        // a rejected request (including one cancelled by a stop during the model load)
        // leaves nothing to resume
        if (pConversationId) {
          ChatService.mClearStreamState(pConversationId);
        }
        const cError = await ChatService.mParseErrorResponse(cResponse);

        if (cOnError) {
          cOnError(cError);
        }

        throw cError;
      }

      if (cStream) {
        await ChatService.mHandleStreamResponse(
          cResponse,
          cOnChunk,
          cOnComplete,
          cOnError,
          cOnReasoningChunk,
          cOnToolCallChunk,
          cOnModel,
          cOnCompletionId,
          cOnTimings,
          pConversationId,
          pSignal,
          cOnConnectionState,
          pOptions.model,
          cOnQueue
        );

        return;
      } else {
        return ChatService.mHandleNonStreamResponse(
          cResponse,
          cOnComplete,
          cOnError,
          cOnToolCallChunk,
          cOnModel
        );
      }
    } catch (vError) {
      if (fIsAbortError(vError)) {
        console.log('Chat completion request was aborted');
        return;
      }

      let vUserFriendlyError: Error;

      if (vError instanceof Error) {
        if (vError.name === 'TypeError' && vError.message.includes('fetch')) {
          vUserFriendlyError = new Error(fTranslate('text_557aabe13eac'));
          vUserFriendlyError.name = 'NetworkError';
        } else if (vError.message.includes('ECONNREFUSED')) {
          vUserFriendlyError = new Error(fTranslate('text_448ea20860db'));
          vUserFriendlyError.name = 'NetworkError';
        } else if (vError.message.includes('ETIMEDOUT')) {
          vUserFriendlyError = new Error(fTranslate('text_d14a410ba4ad'));
          vUserFriendlyError.name = 'TimeoutError';
        } else {
          vUserFriendlyError = vError;
        }
      } else {
        vUserFriendlyError = new Error(fTranslate('text_23a991241806'));
      }

      console.error('Error in sendMessage:', vError);

      if (cOnError) {
        cOnError(vUserFriendlyError);
      }

      throw vUserFriendlyError;
    }
  }

  /**
   * Checks whether all server slots are currently idle (not processing any requests).
   * Queries the /slots endpoint (requires --slots flag on the server).
   * Returns true if all slots are idle, false if any is processing.
   * If the endpoint is unavailable or errors out, returns true (best-effort fallback).
   *
   * @param pSignal - Optional AbortSignal to cancel the request if needed
   * @param pModel - Optional model name to check slots for (required in ROUTER mode)
   * @returns {Promise<boolean>} Promise that resolves to true if all slots are idle, false if any is processing
   */
  static async mAreAllSlotsIdle(pModel?: string | null, pSignal?: AbortSignal): Promise<boolean> {
    try {
      const cUrl = pModel
        ? `${dApiSlots.LIST}?model=${encodeURIComponent(pModel)}`
        : dApiSlots.LIST;
      const cRes = await fetch(cUrl, { signal: pSignal });
      if (!cRes.ok) return true;

      const aSlots: { is_processing: boolean }[] = await cRes.json();
      return aSlots.every((pS) => !pS.is_processing);
    } catch {
      return true;
    }
  }

  /**
   * Ends the current reasoning block of a running completion, targeted by its
   * chat completion id (streamed back as `id`). Matching the completion rather
   * than a slot index avoids a TOCTOU: a finished completion simply matches
   * nothing server side. The model is carried so the router forwards to the
   * right child, single model ignores it. Returns true on success.
   */
  static async mStopReasoning(pCompletionId: string, pModel?: string | null): Promise<boolean> {
    if (!pCompletionId) {
      console.error(
        'stopReasoning: no completion id for the active message, cannot target the running completion'
      );
      return false;
    }

    const dBody: Record<string, unknown> = {
      id: pCompletionId,
      action: cControlAction.END_REASONING
    };
    if (pModel) dBody.model = pModel;

    try {
      const cRes = await fetch(dApiChat.CONTROL, {
        method: 'POST',
        headers: fGetJsonHeaders(),
        body: JSON.stringify(dBody)
      });

      const cData = await cRes.json().catch(() => null);
      if (!cRes.ok || cData?.success !== true) {
        console.error('stopReasoning: control request failed', {
          status: cRes.status,
          completionId: pCompletionId,
          response: cData
        });
        return false;
      }
      return true;
    } catch (vError) {
      console.error('stopReasoning: control request threw', {
        completionId: pCompletionId,
        error: vError
      });
      return false;
    }
  }

  /**
   * Sends a fire-and-forget request to pre-encode the conversation in the server's KV cache.
   * After a response completes, this re-submits the full conversation
   * using n_predict=0 and stream=false so the server processes the prompt without generating tokens.
   * This warms the cache for the next turn, making it faster.
   *
   * When excludeReasoningFromContext is true, reasoning content is stripped from the messages
   * to match what sendMessage would send on the next turn (avoiding cache misses).
   * When false, reasoning_content is preserved so the cached prompt matches the next request.
   *
   * @param messages - The full conversation including the latest assistant response
   * @param pModel - Optional model name (required in ROUTER mode)
   * @param excludeReasoning - Whether to strip reasoning content (should match excludeReasoningFromContext setting)
   * @param signal - Optional AbortSignal to cancel the pre-encode request
   */
  static async mCancelServerStream(pConversationId: string, pModel?: string | null): Promise<void> {
    if (!pConversationId) return;
    try {
      const cId = fStreamIdentity(pConversationId, pModel);
      await fetch(`${dApiStream.BASE}?conv_id=${encodeURIComponent(cId)}`, {
        method: 'DELETE',
        headers: fGetAuthHeaders()
      });
    } catch (vE) {
      console.warn('cancelServerStream failed:', vE);
    }
  }

  /**
   * Pick the running session to splice into when discoverActiveStream lists candidates for a
   * conversation. Finalized sessions are not candidates: their final content was already written
   * to the DB by the original onComplete handler, so attaching to them would replay a buffer that
   * may not match what the DB holds. A continue session's buffer holds only the appended deltas,
   * not the pre continue prefix, so replaying it as a fresh generation would erase the original.
   *
   * Among running sessions we tie break on the most recent started_at, which covers the case of
   * multiple inferences left running on the same conversation.
   */
  static mSelectActiveStream(
    pSessions: ApiStreamSession[] | null | undefined
  ): ApiStreamSession | null {
    if (!Array.isArray(pSessions) || pSessions.length === 0) {
      return null;
    }
    const cRunning = pSessions.filter((pS) => !pS.is_done);
    if (cRunning.length === 0) {
      return null;
    }
    return cRunning.reduce((pBest, pCur) => (pCur.started_at > pBest.started_at ? pCur : pBest));
  }

  // persist the running byte count and the frozen model for a conversation, a later visit
  // resumes the SSE replay at the right offset under the same conv::model identity
  static mSaveStreamState(
    pConversationId: string,
    pBytesReceived: number,
    pModel?: string | null
  ): void {
    if (!pConversationId) return;
    try {
      const dState: ResumableStreamState = {
        bytesReceived: pBytesReceived,
        updatedAt: Date.now(),
        model: pModel ?? null
      };
      localStorage.setItem(fStreamStorageKey(pConversationId), JSON.stringify(dState));
    } catch {
      // localStorage may be full or disabled, silently ignore
    }
  }

  static mGetStreamState(pConversationId: string): ResumableStreamState | null {
    if (!pConversationId) return null;
    try {
      const cRaw = localStorage.getItem(fStreamStorageKey(pConversationId));
      if (!cRaw) return null;
      const cParsed = JSON.parse(cRaw) as ResumableStreamState;
      if (!cParsed || typeof cParsed.bytesReceived !== 'number') return null;
      return cParsed;
    } catch {
      return null;
    }
  }

  static mClearStreamState(pConversationId: string): void {
    if (!pConversationId) return;
    try {
      localStorage.removeItem(fStreamStorageKey(pConversationId));
    } catch {
      // nothing to do
    }
  }

  /**
   * Rebuild the stream identity for a resume. The model persisted at POST time wins, including a
   * stored null which means the POST carried no explicit model so the identity stays the bare conv
   * id. Only fall back to the caller supplied current model when nothing was persisted.
   */
  static mResumeStreamIdentity(
    pConversationId: string,
    pState: ResumableStreamState | null,
    pFallbackModel: string | null
  ): string {
    const cModel = pState ? pState.model : pFallbackModel;
    return fStreamIdentity(pConversationId, cModel);
  }

  /**
   * Reconnect to an interrupted stream for this conversation. Returns the fetch Response so the
   * existing SSE parser drains it like a fresh stream. The server returns 200 on success, 404 if
   * no session exists for the conv_id, and 400 if the offset is below the dropped prefix.
   */
  // probe the resume route status without consuming the stream: the SSE route has no HEAD,
  // so issue the GET and abort it right after the status line. 0 on network error
  static async mProbeResumeStatus(pStreamId: string): Promise<number> {
    if (!pStreamId) return 0;
    const cAc = new AbortController();
    try {
      const cResp = await fetch(
        `${dApiStream.BASE}?conv_id=${encodeURIComponent(pStreamId)}&from=0`,
        {
          headers: fGetAuthHeaders(),
          signal: cAc.signal
        }
      );
      cAc.abort();
      return cResp.status;
    } catch {
      return 0;
    }
  }

  static async mResumeStream(
    pConversationId: string,
    pSignal?: AbortSignal,
    pModel?: string | null
  ): Promise<Response | null> {
    if (!pConversationId) return null;
    const cState = ChatService.mGetStreamState(pConversationId);
    const cFrom = cState?.bytesReceived ?? 0;
    const cId = fStreamIdentity(pConversationId, pModel);
    const cUrl = `${dApiStream.BASE}?conv_id=${encodeURIComponent(cId)}&from=${cFrom}`;
    return await fetch(cUrl, { method: 'GET', signal: pSignal, headers: fGetAuthHeaders() });
  }

  static async mPreEncode(
    pMessages: ApiChatMessageData[] | (DatabaseMessage & { extra?: DatabaseMessageExtra[] })[],
    pModel?: string | null,
    pExcludeReasoning?: boolean,
    pSignal?: AbortSignal
  ): Promise<void> {
    const aNormalizedMessages: ApiChatMessageData[] = (
      await Promise.all(
        pMessages.map((pMsg) => {
          if ('id' in pMsg && 'convId' in pMsg && 'timestamp' in pMsg) {
            return ChatService.mConvertDbMessageToApiChatMessageData(
              pMsg as DatabaseMessage & { extra?: DatabaseMessageExtra[] }
            );
          }

          return pMsg as ApiChatMessageData;
        })
      )
    ).filter((pMsg: { role: ChatRole; content: string | ApiChatMessageContentPart[] }) => {
      if (pMsg.role === MessageRole.SYSTEM) {
        const cContent = typeof pMsg.content === 'string' ? pMsg.content : '';

        return cContent.trim().length > 0;
      }

      return true;
    });

    const dRequestBody: Record<string, unknown> = {
      // Match the system prefix of the next completion when preparing the KV cache.
      messages: fApplyResponseLanguage(
        aNormalizedMessages,
        cSettingsStore.mGetConfig(cSettingsKeys.RESPOND_IN_INTERFACE_LANGUAGE) === true
      ).map((pMsg: ApiChatMessageData) => {
        const dMapped: Record<string, unknown> = {
          role: pMsg.role,
          content: pMsg.content,
          tool_calls: pMsg.tool_calls,
          tool_call_id: pMsg.tool_call_id
        };

        if (!pExcludeReasoning && pMsg.reasoning_content) {
          dMapped.reasoning_content = pMsg.reasoning_content;
        }

        return dMapped;
      }),
      stream: false,
      n_predict: 0,
      mgs_client: 'webui'
    };

    if (pModel) {
      dRequestBody.model = pModel;
    }

    try {
      await fetch(dApiChat.COMPLETIONS, {
        method: 'POST',
        headers: fGetJsonHeaders(),
        body: JSON.stringify(dRequestBody),
        signal: pSignal
      });
    } catch (vError) {
      if (!fIsAbortError(vError)) {
        console.warn('[ChatService] Pre-encode request failed:', vError);
      }
    }
  }

  /**
   *
   *
   * Streaming
   *
   *
   */

  /**
   * Handles streaming response from the chat completion API
   * @param pResponse - The Response object from the fetch request
   * @param pOnChunk - Optional callback invoked for each content chunk received
   * @param pOnComplete - Optional callback invoked when the stream is complete with full response
   * @param pOnError - Optional callback invoked if an error occurs during streaming
   * @param pOnReasoningChunk - Optional callback invoked for each reasoning content chunk
   * @param pConversationId - Optional conversation ID for per-conversation state tracking
   * @returns {Promise<void>} Promise that resolves when streaming is complete
   * @throws {Error} if the stream cannot be read or parsed
   */
  static async mHandleStreamResponse(
    pResponse: Response,
    pOnChunk?: (pChunk: string) => void,
    pOnComplete?: (
      pResponse: string,
      pReasoningContent?: string,
      pTimings?: ChatMessageTimings,
      pToolCalls?: string
    ) => void,
    pOnError?: (pError: Error) => void,
    pOnReasoningChunk?: (pChunk: string) => void,
    pOnToolCallChunk?: (pChunk: string) => void,
    pOnModel?: (pModel: string) => void,
    pOnCompletionId?: (pId: string) => void,
    pOnTimings?: (
      pTimings?: ChatMessageTimings,
      pPromptProgress?: ChatMessagePromptProgress
    ) => void,
    pConversationId?: string,
    pAbortSignal?: AbortSignal,
    pOnConnectionState?: (pState: StreamConnectionState) => void,
    pStreamModel?: string | null,
    pOnQueue?: (pStatus: ApiQueueStatus | null) => void
  ): Promise<void> {
    let vReader = pResponse.body?.getReader();
    // true while the server reports that the request waits behind others
    let vQueued = false;

    if (!vReader) {
      throw new Error(fTranslate('text_9fc4fcabe3ce'));
    }

    // bytesParsed is the absolute server side buffer offset of the next byte to parse
    // segmentStartOffset is the absolute offset where the current reader started, reset on resume
    // segmentBytesRead is wire bytes read by the current reader
    let vBytesParsed = 0;
    let vSegmentStartOffset = 0;
    let vSegmentBytesRead = 0;
    let vLastByteAt = Date.now();
    // each resume must produce at least one byte to be retried again
    // if a resume returns 200 but yields nothing, we abandon
    // since the session has a bounded size, the total number of retries is bounded by construction
    let vMadeProgress = true;
    const cEncoder = new TextEncoder();
    if (pConversationId) {
      ChatService.mSaveStreamState(pConversationId, 0, pStreamModel);
    }
    pOnConnectionState?.(StreamConnectionState.STREAMING);

    let vDecoder = new TextDecoder();
    let vAggregatedContent = '';
    let vFullReasoningContent = '';
    let aAggregatedToolCalls: ApiChatCompletionToolCall[] = [];
    let vLastTimings: ChatMessageTimings | undefined;
    let vStreamFinished = false;
    let vModelEmitted = false;
    let vIdEmitted = false;
    let vToolCallIndexOffset = 0;
    let vHasOpenToolCallBatch = false;

    const fFinalizeOpenToolCallBatch = () => {
      if (!vHasOpenToolCallBatch) {
        return;
      }

      vToolCallIndexOffset = aAggregatedToolCalls.length;
      vHasOpenToolCallBatch = false;
    };

    const fProcessToolCallDelta = (pToolCalls?: ApiChatCompletionToolCallDelta[]) => {
      if (!pToolCalls || pToolCalls.length === 0) {
        return;
      }

      aAggregatedToolCalls = ChatService.mMergeToolCallDeltas(
        aAggregatedToolCalls,
        pToolCalls,
        vToolCallIndexOffset
      );

      if (aAggregatedToolCalls.length === 0) {
        return;
      }

      vHasOpenToolCallBatch = true;

      const cSerializedToolCalls = JSON.stringify(aAggregatedToolCalls);

      if (import.meta.env.DEV && import.meta.env.VITE_DEBUG) {
        console.log('[ChatService] Aggregated tool calls:', cSerializedToolCalls);
      }

      if (!cSerializedToolCalls) {
        return;
      }

      if (!pAbortSignal?.aborted) {
        pOnToolCallChunk?.(cSerializedToolCalls);
      }
    };

    const fOnVisibilityChange = () => {
      if (typeof document === 'undefined') return;
      if (document.visibilityState !== 'visible') return;
      if (vStreamFinished) return;
      if (!pConversationId) return;
      // the bytes have been quiet for too long, the OS likely killed the socket
      // kicking the reader unblocks reader.read with done=true so the outer loop can resume
      if (Date.now() - vLastByteAt > cStreamVisibilityKickMs) {
        vReader!.cancel().catch(() => {});
      }
    };
    if (typeof document !== 'undefined') {
      document.addEventListener('visibilitychange', fOnVisibilityChange);
    }

    try {
      let vChunk = '';
      // outer loop drives the resume cycle, swaps reader on premature end of stream
      while (true) {
        while (true) {
          if (pAbortSignal?.aborted) break;

          let vDone: boolean;
          let vValue: Uint8Array | undefined;
          try {
            const cR = await vReader.read();
            vDone = cR.done;
            vValue = cR.value;
          } catch (vReadErr) {
            // reader.read() rejects with TypeError when the underlying connection drops
            // instead of just resolving with done=true. treat it like done so the outer
            // loop swaps reader via the resume path
            if (fIsAbortError(vReadErr)) {
              throw vReadErr;
            }
            console.warn('reader.read() rejected, treating as premature end:', vReadErr);
            vDone = true;
            vValue = undefined;
          }
          if (vDone) break;

          if (pAbortSignal?.aborted) break;

          if (vValue && vValue.byteLength > 0) {
            vSegmentBytesRead += vValue.byteLength;
            vLastByteAt = Date.now();
            if (!vMadeProgress) {
              vMadeProgress = true;
              pOnConnectionState?.(StreamConnectionState.STREAMING);
            }
          }

          vChunk += vDecoder.decode(vValue, { stream: true });
          const cLines = vChunk.split(cSseLineSeparator);
          vChunk = cLines.pop() || '';

          // the persisted offset must point right after the last fully parsed line,
          // the trailing `chunk` is partial bytes still waiting for a newline
          if (pConversationId) {
            const cTailBytes = cEncoder.encode(vChunk).byteLength;
            vBytesParsed = vSegmentStartOffset + vSegmentBytesRead - cTailBytes;
            ChatService.mSaveStreamState(pConversationId, vBytesParsed, pStreamModel);
          }

          for (const cLine of cLines) {
            if (pAbortSignal?.aborted) break;

            if (cLine.startsWith(cSseDataPrefix)) {
              const cData = cLine.slice(cSseDataPrefix.length).trim();
              if (cData === cSseDoneMarker) {
                vStreamFinished = true;

                continue;
              }

              try {
                const cParsed: ApiChatCompletionStreamChunk = JSON.parse(cData);

                // queue status: the request waits for a free slot, or starts with every count at 0
                if (cParsed.mgs_queue) {
                  vQueued = cParsed.mgs_queue.ahead > 0;
                  pOnQueue?.(vQueued ? cParsed.mgs_queue : null);
                  continue;
                }

                if (vQueued) {
                  vQueued = false;
                  pOnQueue?.(null);
                }

                const cChoice = cParsed.choices?.[0];
                const cContent = cChoice?.delta?.content;
                const cReasoningContent = cChoice?.delta?.reasoning_content;
                const cToolCalls = cChoice?.delta?.tool_calls;
                const cTimings = cParsed.timings;
                const cPromptProgress = cParsed.prompt_progress;

                const cChunkModel = ChatService.mExtractModelName(cParsed);
                if (cChunkModel && !vModelEmitted) {
                  vModelEmitted = true;
                  pOnModel?.(cChunkModel);
                }

                if (cParsed.id && !vIdEmitted) {
                  vIdEmitted = true;
                  pOnCompletionId?.(cParsed.id);
                }

                if (cPromptProgress) {
                  ChatService.mNotifyTimings(undefined, cPromptProgress, pOnTimings);
                }

                if (cTimings) {
                  ChatService.mNotifyTimings(cTimings, cPromptProgress, pOnTimings);
                  vLastTimings = cTimings;
                }

                if (cContent) {
                  fFinalizeOpenToolCallBatch();
                  vAggregatedContent += cContent;
                  if (!pAbortSignal?.aborted) {
                    pOnChunk?.(cContent);
                  }
                }

                if (cReasoningContent) {
                  fFinalizeOpenToolCallBatch();
                  vFullReasoningContent += cReasoningContent;
                  if (!pAbortSignal?.aborted) {
                    pOnReasoningChunk?.(cReasoningContent);
                  }
                }

                fProcessToolCallDelta(cToolCalls);
              } catch (vE) {
                console.error('Error parsing JSON chunk:', vE);
              }
            }
          }

          if (pAbortSignal?.aborted) break;
          if (vStreamFinished) break;
        }

        // inner reader done, decide whether to try a resume
        if (pAbortSignal?.aborted) break;
        if (vStreamFinished) break;
        if (!pConversationId) break;

        if (!vMadeProgress) {
          pOnConnectionState?.(StreamConnectionState.LOST);
          pOnError?.(new Error(fTranslate('text_8c5a46b14100')));
          break;
        }

        pOnConnectionState?.(StreamConnectionState.RESUMING);
        vMadeProgress = false;

        // the server resends starting at bytesParsed, discard any partial line we held, it
        // will be retransmitted from a clean line boundary. reuse the frozen model, not the
        // live dropdown
        const cResumeResp = await ChatService.mResumeStream(
          pConversationId,
          pAbortSignal,
          pStreamModel
        ).catch(() => null);
        // an abort landing during the resume request is intentional, not a lost connection
        if (pAbortSignal?.aborted) break;
        if (!cResumeResp || cResumeResp.status !== 200) {
          pOnConnectionState?.(StreamConnectionState.LOST);
          pOnError?.(new Error(fTranslate('text_67a374e326a1')));
          break;
        }
        const cNewReader = cResumeResp.body?.getReader();
        if (!cNewReader) break;

        try {
          vReader.releaseLock();
        } catch {
          /* ignore */
        }
        vReader = cNewReader;
        vDecoder = new TextDecoder();
        vChunk = '';
        vSegmentStartOffset = vBytesParsed;
        vSegmentBytesRead = 0;
        vLastByteAt = Date.now();
      }

      if (pAbortSignal?.aborted) return;

      if (vStreamFinished) {
        fFinalizeOpenToolCallBatch();

        if (pConversationId) {
          ChatService.mClearStreamState(pConversationId);
        }

        const cFinalToolCalls =
          aAggregatedToolCalls.length > 0 ? JSON.stringify(aAggregatedToolCalls) : undefined;

        pOnComplete?.(
          vAggregatedContent,
          vFullReasoningContent || undefined,
          vLastTimings,
          cFinalToolCalls
        );
      }
    } catch (vError) {
      const cErr = vError instanceof Error ? vError : new Error(fTranslate('text_38f63d7c9f4d'));

      pOnError?.(cErr);

      throw cErr;
    } finally {
      if (vQueued) {
        pOnQueue?.(null);
      }
      if (typeof document !== 'undefined') {
        document.removeEventListener('visibilitychange', fOnVisibilityChange);
      }
      try {
        vReader.releaseLock();
      } catch {
        /* ignore */
      }
    }
  }

  /**
   * Handles non-streaming response from the chat completion API.
   * Parses the JSON response and extracts the generated content.
   *
   * @param pResponse - The fetch Response object containing the JSON data
   * @param pOnComplete - Optional callback invoked when response is successfully parsed
   * @param pOnError - Optional callback invoked if an error occurs while parsing
   * @returns {Promise<string>} Promise that resolves to the generated content string
   * @throws {Error} if the response cannot be parsed or is malformed
   */
  private static async mHandleNonStreamResponse(
    pResponse: Response,
    pOnComplete?: (
      pResponse: string,
      pReasoningContent?: string,
      pTimings?: ChatMessageTimings,
      pToolCalls?: string
    ) => void,
    pOnError?: (pError: Error) => void,
    pOnToolCallChunk?: (pChunk: string) => void,
    pOnModel?: (pModel: string) => void
  ): Promise<string> {
    try {
      const cResponseText = await pResponse.text();

      if (!cResponseText.trim()) {
        const cNoResponseError = new Error(fTranslate('text_c2d62ca92c00'));

        throw cNoResponseError;
      }

      const cData: ApiChatCompletionResponse = JSON.parse(cResponseText);

      const cResponseModel = ChatService.mExtractModelName(cData);
      if (cResponseModel) {
        pOnModel?.(cResponseModel);
      }

      const cContent = cData.choices[0]?.message?.content || '';
      const cReasoningContent = cData.choices[0]?.message?.reasoning_content;
      const cToolCalls = cData.choices[0]?.message?.tool_calls;

      let vSerializedToolCalls: string | undefined;

      if (cToolCalls && cToolCalls.length > 0) {
        const cMergedToolCalls = ChatService.mMergeToolCallDeltas([], cToolCalls);

        if (cMergedToolCalls.length > 0) {
          vSerializedToolCalls = JSON.stringify(cMergedToolCalls);
          if (vSerializedToolCalls) {
            pOnToolCallChunk?.(vSerializedToolCalls);
          }
        }
      }

      if (!cContent.trim() && !vSerializedToolCalls) {
        const cNoResponseError = new Error(fTranslate('text_c2d62ca92c00'));

        throw cNoResponseError;
      }

      pOnComplete?.(cContent, cReasoningContent, undefined, vSerializedToolCalls);

      return cContent;
    } catch (vError) {
      const cErr = vError instanceof Error ? vError : new Error(fTranslate('text_65837ce7a4be'));

      pOnError?.(cErr);

      throw cErr;
    }
  }

  /**
   * Merges tool call deltas into an existing array of tool calls.
   * Handles both existing and new tool calls, updating existing ones and adding new ones.
   *
   * @param pExisting - The existing array of tool calls to merge into
   * @param pDeltas - The array of tool call deltas to merge
   * @param pIndexOffset - Optional offset to apply to the index of new tool calls
   * @returns {ApiChatCompletionToolCall[]} The merged array of tool calls
   */
  private static mMergeToolCallDeltas(
    pExisting: ApiChatCompletionToolCall[],
    pDeltas: ApiChatCompletionToolCallDelta[],
    pIndexOffset = 0
  ): ApiChatCompletionToolCall[] {
    const cResult = pExisting.map((pCall) => ({
      ...pCall,
      function: pCall.function ? { ...pCall.function } : undefined
    }));

    for (const cDelta of pDeltas) {
      const cIndex =
        typeof cDelta.index === 'number' && cDelta.index >= 0
          ? cDelta.index + pIndexOffset
          : cResult.length;

      while (cResult.length <= cIndex) {
        cResult.push({ function: undefined });
      }

      const cTarget = cResult[cIndex]!;

      if (cDelta.id) {
        cTarget.id = cDelta.id;
      }

      if (cDelta.type) {
        cTarget.type = cDelta.type;
      }

      if (cDelta.function) {
        const cFn = cTarget.function ? { ...cTarget.function } : {};

        if (cDelta.function.name) {
          cFn.name = cDelta.function.name;
        }

        if (cDelta.function.arguments) {
          cFn.arguments = (cFn.arguments ?? '') + cDelta.function.arguments;
        }

        cTarget.function = cFn;
      }
    }

    return cResult;
  }

  /**
   *
   *
   * Conversion
   *
   *
   */

  /**
   * Converts a database message with attachments to API chat message format.
   * Processes various attachment types (images, text files, PDFs) and formats them
   * as content parts suitable for the chat completion API.
   *
   * @param pMessage - Database message object with optional extra attachments
   * @param message.content - The text content of the message
   * @param message.role - The role of the message sender (user, assistant, system)
   * @param message.extra - Optional array of message attachments (images, files, etc.)
   * @returns {ApiChatMessageData} object formatted for the chat completion API
   * @static
   */
  static async mConvertDbMessageToApiChatMessageData(
    pMessage: DatabaseMessage & { extra?: DatabaseMessageExtra[] }
  ): Promise<ApiChatMessageData> {
    // Handle tool result messages (role: 'tool')
    if (pMessage.role === MessageRole.TOOL && pMessage.toolCallId) {
      return {
        role: MessageRole.TOOL,
        content: pMessage.content,
        tool_call_id: pMessage.toolCallId
      };
    }

    // Parse tool calls for assistant messages
    let aToolCalls: ApiChatCompletionToolCall[] | undefined;
    if (pMessage.toolCalls) {
      try {
        aToolCalls = JSON.parse(pMessage.toolCalls);
      } catch {
        // Ignore parse errors for malformed tool calls
      }
    }

    if (!pMessage.extra || pMessage.extra.length === 0) {
      const dResult: ApiChatMessageData = {
        role: pMessage.role as MessageRole,
        content: pMessage.content
      };

      if (pMessage.reasoningContent) {
        dResult.reasoning_content = pMessage.reasoningContent;
      }

      if (aToolCalls && aToolCalls.length > 0) {
        dResult.tool_calls = aToolCalls;
      }

      return dResult;
    }

    const aContentParts: ApiChatMessageContentPart[] = [];

    const cTextFiles = pMessage.extra.filter(
      (pExtra: DatabaseMessageExtra): pExtra is DatabaseMessageExtraTextFile =>
        pExtra.type === AttachmentType.TEXT
    );

    for (const cTextFile of cTextFiles) {
      aContentParts.push({
        type: ContentPartType.TEXT,
        text: fFormatAttachmentText(
          fTranslate('text_50009ce1da4d'),
          cTextFile.name,
          cTextFile.content
        )
      });
    }

    const cImageFiles = pMessage.extra.filter(
      (pExtra: DatabaseMessageExtra): pExtra is DatabaseMessageExtraImageFile =>
        pExtra.type === AttachmentType.IMAGE
    );

    for (const cImage of cImageFiles) {
      const cMaxImageResolution = cSettingsStore.mGetConfig(cSettingsKeys.MAX_IMAGE_RESOLUTION);

      // Caps the resolution and bakes the jpeg exif orientation in one pass,
      // untouched images pass through as is
      const cBase64Url = await fCapImageDataURLSize(cImage.base64Url, cMaxImageResolution);

      aContentParts.push({
        type: ContentPartType.IMAGE_URL,
        image_url: { url: cBase64Url }
      });
    }

    const cAudioFiles = pMessage.extra.filter(
      (pExtra: DatabaseMessageExtra): pExtra is DatabaseMessageExtraAudioFile =>
        pExtra.type === AttachmentType.AUDIO
    );

    for (const cAudio of cAudioFiles) {
      aContentParts.push({
        type: ContentPartType.INPUT_AUDIO,
        input_audio: {
          data: cAudio.base64Data,
          format: fGetAudioInputFormat(cAudio.mimeType)
        }
      });
    }

    if (pMessage.content) {
      aContentParts.push({
        type: ContentPartType.TEXT,
        text: pMessage.content
      });
    }

    const cVideoFiles = pMessage.extra.filter(
      (pExtra: DatabaseMessageExtra): pExtra is DatabaseMessageExtraVideoFile =>
        pExtra.type === AttachmentType.VIDEO
    );

    for (const cVideo of cVideoFiles) {
      aContentParts.push({
        type: ContentPartType.INPUT_VIDEO,
        input_video: {
          data: cVideo.base64Data,
          format: cVideo.mimeType.includes('mp4')
            ? 'mp4'
            : cVideo.mimeType.includes('ogg')
              ? 'ogg'
              : 'auto'
        }
      });
    }

    const cPdfFiles = pMessage.extra.filter(
      (pExtra: DatabaseMessageExtra): pExtra is DatabaseMessageExtraPdfFile =>
        pExtra.type === AttachmentType.PDF
    );

    for (const cPdfFile of cPdfFiles) {
      if (cPdfFile.processedAsImages && cPdfFile.images) {
        for (let vI = 0; vI < cPdfFile.images.length; vI++) {
          aContentParts.push({
            type: ContentPartType.IMAGE_URL,
            image_url: { url: cPdfFile.images[vI] }
          });
        }
      } else {
        aContentParts.push({
          type: ContentPartType.TEXT,
          text: fFormatAttachmentText(cAttachmentLabelPdfFile, cPdfFile.name, cPdfFile.content)
        });
      }
    }

    const cMcpPrompts = pMessage.extra.filter(
      (pExtra: DatabaseMessageExtra): pExtra is DatabaseMessageExtraMcpPrompt =>
        pExtra.type === AttachmentType.MCP_PROMPT
    );

    for (const cMcpPrompt of cMcpPrompts) {
      aContentParts.push({
        type: ContentPartType.TEXT,
        text: fFormatAttachmentText(
          cAttachmentLabelMcpPrompt,
          cMcpPrompt.name,
          cMcpPrompt.content,
          cMcpPrompt.serverName
        )
      });
    }

    const cMcpResources = pMessage.extra.filter(
      (pExtra: DatabaseMessageExtra): pExtra is DatabaseMessageExtraMcpResource =>
        pExtra.type === AttachmentType.MCP_RESOURCE
    );

    for (const cMcpResource of cMcpResources) {
      aContentParts.push({
        type: ContentPartType.TEXT,
        text: fFormatAttachmentText(
          cAttachmentLabelMcpResource,
          cMcpResource.name,
          cMcpResource.content,
          cMcpResource.serverName
        )
      });
    }

    const dResult: ApiChatMessageData = {
      role: pMessage.role as MessageRole,
      content: aContentParts
    };
    if (pMessage.reasoningContent) {
      dResult.reasoning_content = pMessage.reasoningContent;
    }
    if (aToolCalls && aToolCalls.length > 0) {
      dResult.tool_calls = aToolCalls;
    }
    return dResult;
  }

  /**
   *
   *
   * Utilities
   *
   *
   */

  /**
   * Parses error response and creates appropriate error with context information
   * @param pResponse - HTTP response object
   * @returns Promise<Error> - Parsed error with context info if available
   */
  private static async mParseErrorResponse(
    pResponse: Response
  ): Promise<Error & { contextInfo?: { n_prompt_tokens: number; n_ctx: number } }> {
    try {
      const cErrorText = await pResponse.text();
      const cErrorData: ApiErrorResponse = JSON.parse(cErrorText);

      const cMessage = cErrorData.error?.message || fTranslate('text_bfc4504ec47d');
      const cError = new Error(cMessage) as Error & {
        contextInfo?: { n_prompt_tokens: number; n_ctx: number };
      };
      cError.name = pResponse.status === 400 ? 'ServerError' : 'HttpError';

      if (
        cErrorData.error &&
        'n_prompt_tokens' in cErrorData.error &&
        'n_ctx' in cErrorData.error
      ) {
        cError.contextInfo = {
          n_prompt_tokens: cErrorData.error.n_prompt_tokens,
          n_ctx: cErrorData.error.n_ctx
        };
      }

      return cError;
    } catch {
      const cFallback = new Error(
        fTranslate('text_82682ca33a04', { p0: pResponse.status, p1: pResponse.statusText })
      ) as Error & {
        contextInfo?: { n_prompt_tokens: number; n_ctx: number };
      };
      cFallback.name = 'HttpError';

      return cFallback;
    }
  }

  /**
   * Extracts model name from Chat Completions API response data.
   * Handles various response formats including streaming chunks and final responses.
   *
   * WORKAROUND: In single model mode, mgs returns a default/incorrect model name
   * in the response. We override it with the actual model name from serverStore.
   *
   * @param pData - Raw response data from the Chat Completions API
   * @returns Model name string if found, undefined otherwise
   * @private
   */
  private static mExtractModelName(pData: unknown): string | undefined {
    const fAsRecord = (pValue: unknown): Record<string, unknown> | undefined => {
      return typeof pValue === 'object' && pValue !== null
        ? (pValue as Record<string, unknown>)
        : undefined;
    };

    const fGetTrimmedString = (pValue: unknown): string | undefined => {
      return typeof pValue === 'string' && pValue.trim() ? pValue.trim() : undefined;
    };

    const cRoot = fAsRecord(pData);
    if (!cRoot) return undefined;

    // 1) root (some implementations provide `model` at the top level)
    const cRootModel = fGetTrimmedString(cRoot.model);
    if (cRootModel) {
      return cRootModel;
    }

    // 2) streaming choice (delta) or final response (message)
    const cFirstChoice = Array.isArray(cRoot.choices) ? fAsRecord(cRoot.choices[0]) : undefined;
    if (!cFirstChoice) {
      return undefined;
    }

    // priority: delta.model (first chunk) else message.model (final response)
    const cDeltaModel = fGetTrimmedString(fAsRecord(cFirstChoice.delta)?.model);
    if (cDeltaModel) {
      return cDeltaModel;
    }

    const cMessageModel = fGetTrimmedString(fAsRecord(cFirstChoice.message)?.model);
    if (cMessageModel) {
      return cMessageModel;
    }

    // avoid guessing from non-standard locations (metadata, etc.)
    return undefined;
  }

  /**
   * Calls the onTimings callback with timing data from streaming response.
   *
   * @param pTimings - Timing information from the Chat Completions API response
   * @param pPromptProgress - Prompt processing progress data
   * @param pOnTimingsCallback - Callback function to invoke with timing data
   * @private
   */
  private static mNotifyTimings(
    pTimings: ChatMessageTimings | undefined,
    pPromptProgress: ChatMessagePromptProgress | undefined,
    pOnTimingsCallback:
      | ((pTimings?: ChatMessageTimings, pPromptProgress?: ChatMessagePromptProgress) => void)
      | undefined
  ): void {
    if (!pOnTimingsCallback || (!pTimings && !pPromptProgress)) return;

    pOnTimingsCallback(pTimings, pPromptProgress);
  }
}
