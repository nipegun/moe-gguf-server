import { fTranslate } from '../i18n';
import { fApiUrl } from '../api-url';
/**
 * chatStore - Reactive State Store for Chat Operations
 *
 * Manages chat lifecycle, streaming, message operations, and processing state.
 *
 * **Architecture & Relationships:**
 * - **ChatService**: Stateless API layer (sendMessage, streaming)
 * - **chatStore** (this): Reactive state + business logic
 * - **conversationsStore**: Conversation persistence and navigation
 *
 * @see ChatService in services/chat.service.ts for API operations
 */

import { SvelteMap, SvelteSet } from 'svelte/reactivity';
import { DatabaseService } from '$lib/services/database.service';
import { ChatService } from '$lib/services/chat.service';
import { cStreamResumeRetryMs } from '$lib/constants/api-endpoints';
import { fStreamIdentity } from '$lib/utils/stream-identity';
import { fGetAuthHeaders } from '$lib/utils/api-headers';
import { cContentTypeHeader } from '$lib/constants';
import { MimeTypeApplication } from '$lib/enums';
import { cConversationsStore } from '$lib/stores/conversations.svelte';
import { fConfig } from '$lib/stores/settings.svelte';
import { cAgenticStore } from '$lib/stores/agentic.svelte';
import { cMcpStore } from '$lib/stores/mcp.svelte';
import { fContextSize, fIsRouterMode } from '$lib/stores/server.svelte';
import {
  fSelectedModelName,
  cModelsStore,
  fSelectedModelContextSize
} from '$lib/stores/models.svelte';
import {
  normalizeModelName,
  filterByLeafNodeId,
  findDescendantMessages,
  findLeafNode,
  findMessageById,
  isAbortError,
  generateConversationTitle
} from '$lib/utils';
import { fClassifyContinueIntent } from '$lib/utils/agentic';
import {
  cMaxInactiveConversationStates,
  cInactiveConversationStateMaxAgeMs,
  cSystemMessagePlaceholder,
  cTitleGeneration
} from '$lib/constants';
import type {
  ChatMessageTimings,
  ChatMessagePromptProgress,
  ChatStreamCallbacks,
  ErrorDialogState
} from '$lib/types/chat';
import type {
  ApiChatMessageData,
  ApiProcessingState,
  ApiQueueStatus,
  ApiStreamSession,
  DatabaseMessage,
  DatabaseMessageExtra
} from '$lib/types';
import {
  ContinueIntentKind,
  ErrorDialogType,
  MessageRole,
  MessageType,
  ReasoningEffort,
  StreamConnectionState
} from '$lib/enums';

interface ConversationStateEntry {
  lastAccessed: number;
}

class ChatStore {
  vActiveProcessingState = $state<ApiProcessingState | null>(null);
  vCurrentResponse = $state('');
  vErrorDialogState = $state<ErrorDialogState | null>(null);
  vIsLoading = $state(false);
  // true while the active conversation streams reasoning content but no visible content yet
  vIsReasoning = $state(false);
  // resumable stream connection state for the active conversation
  // streaming -> bytes flowing normally, resuming -> waiting on /v1/stream reconnect, lost -> unrecoverable
  vStreamConnectionState = $state<StreamConnectionState>(StreamConnectionState.STREAMING);
  vChatLoadingStates = new SvelteMap<string, boolean>();
  vChatReasoningStates = new SvelteMap<string, boolean>();
  vChatStreamingStates = new SvelteMap<
    string,
    { response: string; messageId: string; model?: string | null }
  >();
  // convs that the backend reports as having a running session, populated by the global sync
  // at app mount and on visibilitychange. it does not overlap with chatLoadingStates which
  // tracks inferences driven by this browser, both are unioned to feed the sidebar spinners
  private vRemoteRunningConvs = new SvelteSet<string>();
  // per conv attach lifecycle, used to derive the global streaming flag without flipping it
  // off when one conv finishes while another is still streaming. mirrors chatLoadingStates
  // in scope but tracks the attach + tee replay path specifically
  private vAttachingConvs = new SvelteSet<string>();
  // pending resume retry timers while an owning model loads, one per conv
  private vResumeRetryTimers = new SvelteMap<string, ReturnType<typeof setTimeout>>();
  // convs whose resume waits on a model load: their loading state belongs to the retry loop,
  // so discoverActiveStream must not treat it as a live send and bail
  private vResumePendingConvs = new SvelteSet<string>();
  // in-flight discoverActiveStream guard, keyed by conv id
  private vDiscoveringConvs = new SvelteSet<string>();
  private vAbortControllers = new SvelteMap<string, AbortController>();
  private vPreEncodeAbortController: AbortController | null = null;
  private vProcessingStates = new SvelteMap<string, ApiProcessingState | null>();
  private vQueueStates = new SvelteMap<string, ApiQueueStatus>();
  private vConversationStateTimestamps = new SvelteMap<string, ConversationStateEntry>();
  private vActiveConversationId = $state<string | null>(null);
  private vIsStreamingActive = $state(false);
  private vIsEditModeActive = $state(false);
  private aAddFilesHandler: ((pFiles: File[]) => void) | null = $state(null);
  vPendingEditMessageId = $state<string | null>(null);
  private vMessageUpdateCallback:
    | ((pMessageId: string, pUpdates: Partial<DatabaseMessage>) => void)
    | null = null;
  private vPendingDraftMessage = $state<string>('');
  private vPendingDraftFiles = $state<ChatUploadedFile[]>([]);

  /** Reactive: queued pending messages for non-agentic streaming */
  private vPendingMessages = new SvelteMap<
    string,
    { content: string; extras?: DatabaseMessageExtra[] }
  >();

  private mSetChatLoading(pConvId: string, pLoading: boolean): void {
    this.mTouchConversationState(pConvId);
    if (pLoading) {
      this.vChatLoadingStates.set(pConvId, true);
      if (pConvId === cConversationsStore.vActiveConversation?.id) this.vIsLoading = true;
    } else {
      this.vChatLoadingStates.delete(pConvId);
      if (pConvId === cConversationsStore.vActiveConversation?.id) this.vIsLoading = false;
      this.mSetChatReasoning(pConvId, false);
      // the local pipe is the authoritative observer of session end: when it finishes (clean
      // onComplete or explicit Stop), the backend session is finalized too, so we drop the
      // sidebar hint for this conv right away instead of waiting for the next visibilitychange
      // snapshot. without this the spinner ghosts until the user toggles the tab
      this.vRemoteRunningConvs.delete(pConvId);
    }
  }

  private mSetChatReasoning(pConvId: string, pReasoning: boolean): void {
    if (pReasoning) {
      this.vChatReasoningStates.set(pConvId, true);
      if (pConvId === cConversationsStore.vActiveConversation?.id) this.vIsReasoning = true;
    } else {
      this.vChatReasoningStates.delete(pConvId);
      if (pConvId === cConversationsStore.vActiveConversation?.id) this.vIsReasoning = false;
    }
  }
  private mSetChatStreaming(
    pConvId: string,
    pResponse: string,
    pMessageId: string,
    pModel?: string | null
  ): void {
    this.mTouchConversationState(pConvId);
    this.vChatStreamingStates.set(pConvId, {
      response: pResponse,
      messageId: pMessageId,
      model: pModel ?? this.vChatStreamingStates.get(pConvId)?.model
    });
    if (pConvId === cConversationsStore.vActiveConversation?.id) this.vCurrentResponse = pResponse;
  }
  private mClearChatStreaming(pConvId: string, pMessageId?: string): void {
    // session aware: a stale generation must not wipe a newer one's streaming state on the
    // same conversation, that would drop the frozen stop identity and stop the wrong session
    if (pMessageId !== undefined) {
      const cCur = this.vChatStreamingStates.get(pConvId);
      if (cCur && cCur.messageId !== pMessageId) return;
    }
    this.vChatStreamingStates.delete(pConvId);
    if (pConvId === cConversationsStore.vActiveConversation?.id) this.vCurrentResponse = '';
  }
  private mGetChatStreaming(pConvId: string): { response: string; messageId: string } | undefined {
    return this.vChatStreamingStates.get(pConvId);
  }
  mSyncLoadingStateForChat(pConvId: string): void {
    this.vIsLoading = this.vChatLoadingStates.get(pConvId) || false;
    this.vIsReasoning = this.vChatReasoningStates.get(pConvId) || false;
    const cS = this.vChatStreamingStates.get(pConvId);
    this.vCurrentResponse = cS?.response || '';
    this.vIsStreamingActive = cS !== undefined;
    this.mSetActiveProcessingConversation(pConvId);
    // Sync streaming content to activeMessages so UI displays current content
    if (cS?.response && cS?.messageId) {
      const cIdx = cConversationsStore.mFindMessageIndex(cS.messageId);
      if (cIdx !== -1) {
        cConversationsStore.mUpdateMessageAtIndex(cIdx, { content: cS.response });
      }
    }
  }
  /**
   * Server side stream discovery, split in three pieces:
   *
   * probeServerStream(convId) -> hits POST /v1/streams/lookup with the conv id, returns the session to attach
   *   to or null. Pure read, no side effect, no UI lock. Safe to fire in parallel with anything.
   *
   * attachServerStream(convId) -> flips the spinner immediately, fetches the replay stream
   *   from byte 0, finds the assistant slot to splice into (creates a placeholder if the conv has
   *   no assistant message yet, for cross device or fresh local DB cases), and pipes the SSE bytes
   *   into the message via handleStreamResponse.
   *
   * discoverActiveStream(convId) -> probe + attach in one call. Used by callers that do not need
   *   to overlap the probe with other async work.
   *
   * The mount of the chat page in +page.svelte calls probeServerStream in parallel with
   * loadConversation, then attachServerStream once both have settled. This gives the earliest
   * possible time to spinner and avoids racing against an empty activeMessages array.
   */
  async mProbeServerStream(pConvId: string): Promise<ApiStreamSession | null> {
    if (!pConvId) return null;
    let vListResp: Response;
    try {
      // POST the one conv id we are probing
      vListResp = await fetch(`./v1/streams/lookup`, {
        method: 'POST',
        headers: { ...fGetAuthHeaders(), [cContentTypeHeader]: MimeTypeApplication.JSON },
        body: JSON.stringify({ conversation_ids: [pConvId] })
      });
    } catch (vE) {
      console.warn('probeServerStream fetch failed:', vE);
      return null;
    }
    if (!vListResp.ok) {
      console.warn(`probeServerStream got HTTP ${vListResp.status} for conv ${pConvId}`);
      return null;
    }
    let aSessions: ApiStreamSession[];
    try {
      aSessions = (await vListResp.json()) as ApiStreamSession[];
    } catch (vE) {
      console.warn('probeServerStream JSON parse failed:', vE);
      return null;
    }
    return ChatService.mSelectActiveStream(aSessions);
  }

  async mAttachServerStream(pConvId: string, pStreamId?: string): Promise<void> {
    if (!pConvId) return;
    if (this.vChatStreamingStates.has(pConvId)) return;

    // flip the spinner immediately, the user sees activity as soon as the conv becomes active.
    // the global isStreamingActive flag is derived from attachingConvs.size, so adding here
    // turns it on, and removing in unlock only turns it off when this is the last attach
    this.mSetChatLoading(pConvId, true);
    this.vAttachingConvs.add(pConvId);
    this.mSetStreamingActive(true);
    // only set the active processing conv if we are looking at it, otherwise a background
    // attach would steal the indicator from the conv the user is currently viewing
    if (pConvId === cConversationsStore.vActiveConversation?.id) {
      this.mSetActiveProcessingConversation(pConvId);
    }

    const fUnlock = () => {
      this.vAttachingConvs.delete(pConvId);
      // flip the global flag off only when no other conv is still attaching
      if (this.vAttachingConvs.size === 0) {
        this.mSetStreamingActive(false);
      }
      this.mSetChatLoading(pConvId, false);
      this.mClearChatStreaming(pConvId);
    };

    // fetch the replay stream from byte 0, rebuild the assistant message from scratch.
    // resolve the server side identity, fall back to streamIdentity when the caller does not
    // pass a streamId. probeServerStream returns the full id (with ::model suffix when present)
    const cId = pStreamId || fStreamIdentity(pConvId, fSelectedModelName());
    let vResponse: Response;
    try {
      vResponse = await fetch(`./v1/stream?conv_id=${encodeURIComponent(cId)}&from=0`, {
        headers: fGetAuthHeaders()
      });
    } catch (vE) {
      console.error('attachServerStream replay fetch failed:', vE);
      fUnlock();
      return;
    }
    if (!vResponse.ok) {
      console.warn(`attachServerStream replay got HTTP ${vResponse.status} for conv ${pConvId}`);
      fUnlock();
      return;
    }

    // load the target conversation messages by id, not via the active store. when multiple
    // attaches run in parallel the active store may reflect another conv and writing through
    // its index mixes content across convs (CoT flicker, message bleed). by going through the
    // DB we stay isolated, and only mirror into the active store when the attached conv is
    // the one currently displayed
    let aMessages: DatabaseMessage[];
    try {
      aMessages = await DatabaseService.mGetConversationMessages(pConvId);
    } catch (vE) {
      console.error('attachServerStream load messages failed:', vE);
      fUnlock();
      return;
    }

    // locate the slot to splice into, create a placeholder assistant message if there is none.
    // we use the conv-scoped findLastAssistantIdx helpers, they only depend on the array
    let vTargetIdx = this.mFindLastAssistantIdx(aMessages);
    if (vTargetIdx === -1) {
      const cLastUserIdx = this.mFindLastUserIdx(aMessages);
      if (cLastUserIdx === -1) {
        console.warn(
          `attachServerStream: conv ${pConvId} has no user or assistant message, cannot splice`
        );
        fUnlock();
        return;
      }
      try {
        const cPlaceholder = await DatabaseService.mCreateMessageBranch(
          {
            convId: pConvId,
            role: MessageRole.ASSISTANT,
            content: '',
            type: MessageType.TEXT,
            timestamp: Date.now(),
            parent: aMessages[cLastUserIdx].id,
            children: [],
            toolCalls: ''
          } as Omit<DatabaseMessage, 'id'>,
          aMessages[cLastUserIdx].id
        );
        aMessages = [...aMessages, cPlaceholder];
        vTargetIdx = aMessages.length - 1;
        // only push into the active store when this conv is the one displayed right now
        if (pConvId === cConversationsStore.vActiveConversation?.id) {
          cConversationsStore.mAddMessageToActive(cPlaceholder);
        }
      } catch (vE) {
        console.error('attachServerStream placeholder creation failed:', vE);
        fUnlock();
        return;
      }
    }
    if (vTargetIdx === -1) {
      fUnlock();
      return;
    }
    const cTargetMessage = aMessages[vTargetIdx];
    const cTargetMessageId = cTargetMessage.id;
    // when the assistant slot already has content, the running session is a continue or
    // another append flow and its buffer holds only the appended deltas. preserve the prefix
    // and let the replay add to it. when the slot is empty the session buffer holds the whole
    // message so we wipe and rebuild from byte 0
    const cExistingContent = cTargetMessage.content ?? '';
    const cExistingReasoning = cTargetMessage.reasoningContent ?? '';
    const cIsAppendMode = cExistingContent.length > 0;

    // helper: write to the active store only when the attached conv is currently displayed.
    // the lookup by message id is robust to reordering of activeMessages, two parallel attaches
    // can no longer step on each other's indices
    const fWriteActive = (pUpdates: Partial<DatabaseMessage>) => {
      if (pConvId !== cConversationsStore.vActiveConversation?.id) {
        return;
      }
      const cLiveIdx = cConversationsStore.mFindMessageIndex(cTargetMessageId);
      if (cLiveIdx === -1) return;
      cConversationsStore.mUpdateMessageAtIndex(cLiveIdx, pUpdates);
    };

    if (!cIsAppendMode) {
      fWriteActive({ content: '', reasoningContent: undefined });
    }

    // extract the model suffix, the resume calls in handleStreamResponse must reuse the model
    // the session was tagged with, not the live dropdown
    const cSepIdx = cId.indexOf('::');
    const cAttachedModel: string | null = cSepIdx === -1 ? null : cId.slice(cSepIdx + 2);
    this.mSetChatStreaming(pConvId, cExistingContent, cTargetMessageId, cAttachedModel);
    const cAbortController = this.mGetOrCreateAbortController(pConvId);

    let vStreamedContent = '';
    let vStreamedReasoningContent = '';

    const fCleanup = () => {
      fUnlock();
      this.mSetProcessingState(pConvId, null);
    };

    try {
      await ChatService.mHandleStreamResponse(
        vResponse,
        (pChunk: string) => {
          vStreamedContent += pChunk;
          const cDisplayed = cIsAppendMode ? cExistingContent + vStreamedContent : vStreamedContent;
          fWriteActive({ content: cDisplayed });
          this.mSetChatStreaming(pConvId, cDisplayed, cTargetMessageId);
        },
        async (
          pFinalContent?: string,
          pReasoningContent?: string,
          pTimings?: ChatMessageTimings,
          pToolCalls?: string
        ) => {
          const cStreamed = vStreamedContent || pFinalContent || '';
          const cStreamedR = vStreamedReasoningContent || pReasoningContent || '';
          const cContent = cIsAppendMode ? cExistingContent + cStreamed : cStreamed;
          const cReasoning = cIsAppendMode ? cExistingReasoning + cStreamedR : cStreamedR;
          // the DB write is the source of truth, mirror to the active store only when
          // the conv is currently displayed
          await DatabaseService.mUpdateMessage(cTargetMessageId, {
            content: cContent,
            reasoningContent: cReasoning || undefined,
            toolCalls: pToolCalls || '',
            timings: pTimings
          });
          fWriteActive({
            content: cContent,
            reasoningContent: cReasoning || undefined,
            timings: pTimings
          });
          fCleanup();
        },
        (pErr: Error) => {
          console.error('attachServerStream pipe error:', pErr);
          fCleanup();
        },
        (pChunk: string) => {
          vStreamedReasoningContent += pChunk;
          const cDisplayed = cIsAppendMode
            ? cExistingReasoning + vStreamedReasoningContent
            : vStreamedReasoningContent;
          fWriteActive({ reasoningContent: cDisplayed });
        },
        undefined,
        undefined,
        undefined,
        undefined,
        pConvId,
        cAbortController.signal,
        (pConnState: StreamConnectionState) => {
          if (pConvId === cConversationsStore.vActiveConversation?.id) {
            this.vStreamConnectionState = pConnState;
          }
        },
        cAttachedModel,
        (pStatus: ApiQueueStatus | null) => this.mSetQueueStatus(pConvId, pStatus)
      );
    } catch (vE) {
      console.error('attachServerStream pipe crashed:', vE);
      fCleanup();
    }
  }

  /**
   * Model frozen at send time for a stream awaiting resume, from the persisted stream state.
   * The load progress indicator targets it after a reload, when the message row has no model
   * yet and the dropdown selection may not be restored.
   */
  mGetResumeModel(pConvId: string): string | null {
    return ChatService.mGetStreamState(pConvId)?.model ?? null;
  }

  async mDiscoverActiveStream(pConvId: string): Promise<void> {
    if (!pConvId) return;
    if (this.vChatStreamingStates.has(pConvId)) return;
    if (this.vChatLoadingStates.get(pConvId) && !this.vResumePendingConvs.has(pConvId)) return;
    // concurrency guard: another discover may already be running for this conv (typical race
    // between mount and visibilitychange on tab switch). a second concurrent fetch on the same
    // /v1/stream would duplicate every byte into the DB message, this guard bounces it
    if (this.vDiscoveringConvs.has(pConvId)) return;
    this.vDiscoveringConvs.add(pConvId);

    try {
      // the model is frozen at POST time, rebuild the exact conv::model identity from the
      // persisted state so the lookup key matches what the server stored. null means a single
      // model conv with no ::suffix, only guess from the dropdown with no persisted state
      const cLocalState = ChatService.mGetStreamState(pConvId);
      const cStreamId = ChatService.mResumeStreamIdentity(
        pConvId,
        cLocalState,
        fSelectedModelName()
      );

      // primary path: ask the server which sessions exist for this identity
      const cServerTarget = await this.mProbeServerStream(cStreamId);
      if (cServerTarget) {
        // pass the full server side identity (may carry a ::model suffix) so the GET routes
        // straight to the owning session, no probe or fan out
        await this.mAttachServerStream(pConvId, cServerTarget.conversation_id);
        return;
      }

      // fallback: local state remembers an interrupted byte offset for this conv, the server may
      // still have a live session matching that identity (we just lost the bytes mid stream). retry
      // with the frozen identity, the server probe inside attachServerStream tells us if it exists
      if (!cLocalState) {
        return;
      }
      // quiet status probe first: a full attach flips the loading UI on every try, probing
      // keeps the retry loop invisible while the owning model is still loading (503)
      const cStatus = await ChatService.mProbeResumeStatus(cStreamId);
      if (cStatus === 503) {
        // make the wait visible: the empty assistant row persisted at send time renders
        // the processing info, whose model load percentage flows from the models feed
        this.vResumePendingConvs.add(pConvId);
        this.mSetChatLoading(pConvId, true);
        if (!this.vResumeRetryTimers.has(pConvId)) {
          this.vResumeRetryTimers.set(
            pConvId,
            setTimeout(() => {
              this.vResumeRetryTimers.delete(pConvId);
              void this.mDiscoverActiveStream(pConvId);
            }, cStreamResumeRetryMs)
          );
        }
        return;
      }
      if (this.vResumePendingConvs.delete(pConvId) && cStatus !== 200) {
        // the wait is over without a session to attach, drop the visible loading state
        this.mSetChatLoading(pConvId, false);
      }
      if (cStatus === 0) {
        // transient network failure, the next mount or visibility change retries
        return;
      }
      if (cStatus !== 200) {
        // the session is gone (stopped, TTL expired), nothing to resume anymore
        ChatService.mClearStreamState(pConvId);
        return;
      }
      await this.mAttachServerStream(pConvId, cStreamId);
      // if attachServerStream failed (session gone, TTL expired), clear the local state to avoid retrying forever
      if (!this.vChatStreamingStates.has(pConvId) && !this.vChatLoadingStates.get(pConvId)) {
        ChatService.mClearStreamState(pConvId);
      }
    } finally {
      this.vDiscoveringConvs.delete(pConvId);
    }
  }

  private mFindLastAssistantIdx(pMessages: DatabaseMessage[]): number {
    for (let vI = pMessages.length - 1; vI >= 0; vI--) {
      if (pMessages[vI].role === MessageRole.ASSISTANT) return vI;
    }
    return -1;
  }

  private mFindLastUserIdx(pMessages: DatabaseMessage[]): number {
    for (let vI = pMessages.length - 1; vI >= 0; vI--) {
      if (pMessages[vI].role === MessageRole.USER) return vI;
    }
    return -1;
  }

  mClearUIState(): void {
    this.vIsLoading = false;
    this.vCurrentResponse = '';
    this.vIsStreamingActive = false;
  }

  mSetActiveProcessingConversation(pConversationId: string | null): void {
    this.vActiveConversationId = pConversationId;
    this.vActiveProcessingState = pConversationId
      ? this.vProcessingStates.get(pConversationId) || null
      : null;
  }

  mGetProcessingState(pConversationId: string): ApiProcessingState | null {
    return this.vProcessingStates.get(pConversationId) || null;
  }

  private mSetProcessingState(pConversationId: string, pState: ApiProcessingState | null): void {
    if (pState === null) this.vProcessingStates.delete(pConversationId);
    else this.vProcessingStates.set(pConversationId, pState);
    if (pConversationId === this.vActiveConversationId) this.vActiveProcessingState = pState;
  }

  mClearProcessingState(pConversationId: string): void {
    this.vProcessingStates.delete(pConversationId);
    if (pConversationId === this.vActiveConversationId) this.vActiveProcessingState = null;
  }

  /**
   * Queue status of a conversation whose request waits for a free slot, as streamed by the
   * server: how many requests run or wait ahead, sent directly to the API or by the interface
   */
  mGetQueueStatus(pConversationId: string): ApiQueueStatus | null {
    return this.vQueueStates.get(pConversationId) || null;
  }

  private mSetQueueStatus(pConversationId: string, pStatus: ApiQueueStatus | null): void {
    if (pStatus === null) this.vQueueStates.delete(pConversationId);
    else this.vQueueStates.set(pConversationId, pStatus);
  }

  mGetActiveProcessingState(): ApiProcessingState | null {
    return this.vActiveProcessingState;
  }

  mGetCurrentProcessingStateSync(): ApiProcessingState | null {
    return this.vActiveProcessingState;
  }

  private mSetStreamingActive(pActive: boolean): void {
    this.vIsStreamingActive = pActive;
  }

  mIsStreaming(): boolean {
    return this.vIsStreamingActive;
  }

  private mGetOrCreateAbortController(pConvId: string): AbortController {
    let vC = this.vAbortControllers.get(pConvId);
    if (!vC || vC.signal.aborted) {
      vC = new AbortController();
      this.vAbortControllers.set(pConvId, vC);
    }
    return vC;
  }

  private mAbortRequest(pConvId?: string): void {
    if (pConvId) {
      const cC = this.vAbortControllers.get(pConvId);
      if (cC) {
        cC.abort();
        this.vAbortControllers.delete(pConvId);
      }
    } else {
      for (const cC of this.vAbortControllers.values()) cC.abort();
      this.vAbortControllers.clear();
    }
  }

  /**
   * Abort the current agentic flow signal without clearing loading state.
   * Used by "Send immediately" to force the agentic loop to exit so that
   * the pending steering message can be re-sent.
   *
   * Any tool calls captured mid-stream are dropped before the abort so the
   * pending message (or a manual follow-up) does not re-send a half-received
   * tool call with invalid JSON arguments to the server. Mirrors what the
   * Stop button already does through stopGenerationForChat.
   */
  async mAbortCurrentFlow(pConvId: string): Promise<void> {
    await this.mSavePartialResponseIfNeeded(pConvId);
    const cC = this.vAbortControllers.get(pConvId);
    if (cC) {
      cC.abort();
      this.vAbortControllers.delete(pConvId);
    }
  }

  private mShowErrorDialog(pState: ErrorDialogState | null): void {
    this.vErrorDialogState = pState;
  }

  mDismissErrorDialog(): void {
    this.vErrorDialogState = null;
  }

  mClearEditMode(): void {
    this.vIsEditModeActive = false;
    this.aAddFilesHandler = null;
  }

  mIsEditing(): boolean {
    return this.vIsEditModeActive;
  }

  mSetEditModeActive(pHandler: (pFiles: File[]) => void): void {
    this.vIsEditModeActive = true;
    this.aAddFilesHandler = pHandler;
  }

  mGetAddFilesHandler(): ((pFiles: File[]) => void) | null {
    return this.aAddFilesHandler;
  }

  mClearPendingEditMessageId(): void {
    this.vPendingEditMessageId = null;
  }

  mSavePendingDraft(pMessage: string, pFiles: ChatUploadedFile[]): void {
    this.vPendingDraftMessage = pMessage;
    this.vPendingDraftFiles = [...pFiles];
  }

  mConsumePendingDraft(): { message: string; files: ChatUploadedFile[] } | null {
    if (!this.vPendingDraftMessage && this.vPendingDraftFiles.length === 0) return null;
    const dD = { message: this.vPendingDraftMessage, files: [...this.vPendingDraftFiles] };
    this.vPendingDraftMessage = '';
    this.vPendingDraftFiles = [];
    return dD;
  }

  mHasPendingDraft(): boolean {
    return Boolean(this.vPendingDraftMessage) || this.vPendingDraftFiles.length > 0;
  }

  mGetAllLoadingChats(): string[] {
    // union of local (this browser is piping) and remote (backend reports a running session
    // for this conv but no local pipe yet) sources. the sidebar shows one spinner per entry
    const cOut = new SvelteSet<string>(this.vChatLoadingStates.keys());
    for (const cId of this.vRemoteRunningConvs) {
      cOut.add(cId);
    }
    return Array.from(cOut);
  }

  mGetAllStreamingChats(): string[] {
    return Array.from(this.vChatStreamingStates.keys());
  }

  /**
   * Resync the remote running convs set from the backend. Called by the layout at mount and on
   * visibilitychange, no polling. A snapshot semantic: the set is replaced wholesale, stale entries
   * for sessions that finalized while the browser was elsewhere are dropped naturally.
   */
  async mSyncRemoteRunningStreams(): Promise<void> {
    // the conversations store loads from IndexedDB asynchronously, the +layout onMount caller
    // fires before that finishes. read ids straight from the DB so the result does not depend
    // on the store init race, and the sidebar spinners light up at first paint for every conv
    // the user owns even if it has not been hydrated into the store yet
    let aIds: string[];
    try {
      const cAll = await DatabaseService.mGetAllConversations();
      aIds = cAll.map((pC) => pC.id).filter((pId) => !!pId);
    } catch (vE) {
      console.warn('syncRemoteRunningStreams DB read failed:', vE);
      return;
    }
    // only ask about conv ids the user already owns
    if (aIds.length === 0) {
      for (const cId of Array.from(this.vRemoteRunningConvs)) {
        this.vRemoteRunningConvs.delete(cId);
      }
      return;
    }
    // rebuild the frozen conv::model identity per conv so a session started with a model still
    // matches. the server response is mapped back to the bare id below for the sidebar set
    const cLookupIds = aIds.map((pId) =>
      ChatService.mResumeStreamIdentity(pId, ChatService.mGetStreamState(pId), null)
    );
    let aSessions: ApiStreamSession[];
    try {
      const cResp = await fetch(fApiUrl('/v1/streams/lookup'), {
        method: 'POST',
        headers: { ...fGetAuthHeaders(), [cContentTypeHeader]: MimeTypeApplication.JSON },
        body: JSON.stringify({ conversation_ids: cLookupIds })
      });
      if (!cResp.ok) return;
      const cBody = (await cResp.json()) as unknown;
      if (!Array.isArray(cBody)) return;
      aSessions = cBody as ApiStreamSession[];
    } catch (vE) {
      console.warn('syncRemoteRunningStreams fetch failed:', vE);
      return;
    }
    const cRunning = new SvelteSet<string>();
    for (const cS of aSessions) {
      if (cS && !cS.is_done && typeof cS.conversation_id === 'string' && cS.conversation_id) {
        // strip the optional ::model suffix, the sidebar set is keyed by the bare conv id
        const cSepIdx = cS.conversation_id.indexOf('::');
        const cBareId = cSepIdx === -1 ? cS.conversation_id : cS.conversation_id.slice(0, cSepIdx);
        cRunning.add(cBareId);
      }
    }
    for (const cId of Array.from(this.vRemoteRunningConvs)) {
      if (!cRunning.has(cId)) {
        this.vRemoteRunningConvs.delete(cId);
      }
    }
    for (const cId of cRunning) {
      this.vRemoteRunningConvs.add(cId);
    }
  }

  mGetChatStreamingPublic(pConvId: string): { response: string; messageId: string } | undefined {
    return this.mGetChatStreaming(pConvId);
  }

  mIsChatLoadingPublic(pConvId: string): boolean {
    return this.vChatLoadingStates.get(pConvId) || false;
  }

  mIsChatReasoningPublic(pConvId: string): boolean {
    return this.vChatReasoningStates.get(pConvId) || false;
  }

  private mIsChatLoadingInternal(pConvId: string): boolean {
    return this.vChatLoadingStates.has(pConvId) || this.vChatStreamingStates.has(pConvId);
  }

  mHasPendingMessage(pConvId: string): boolean {
    return this.vPendingMessages.has(pConvId);
  }

  mPendingMessageContent(pConvId: string): string | null {
    return this.vPendingMessages.get(pConvId)?.content ?? null;
  }

  mPendingMessageExtras(pConvId: string): DatabaseMessageExtra[] | undefined {
    return this.vPendingMessages.get(pConvId)?.extras;
  }

  mInjectPendingMessage(pConvId: string, pContent: string, pExtras?: DatabaseMessageExtra[]): void {
    this.vPendingMessages.set(pConvId, { content: pContent, extras: pExtras });
  }

  mClearPendingMessage(pConvId: string): void {
    this.vPendingMessages.delete(pConvId);
  }

  mConsumePendingMessage(
    pConvId: string
  ): { content: string; extras?: DatabaseMessageExtra[] } | null {
    const cMsg = this.vPendingMessages.get(pConvId);
    if (!cMsg) return null;
    this.vPendingMessages.delete(pConvId);
    return cMsg;
  }

  private mTouchConversationState(pConvId: string): void {
    this.vConversationStateTimestamps.set(pConvId, { lastAccessed: Date.now() });
  }

  mCleanupOldConversationStates(pActiveConversationIds?: string[]): number {
    const cNow = Date.now();
    const cActiveIdsList = pActiveConversationIds ?? [];
    const cPreserveIds = this.vActiveConversationId
      ? [...cActiveIdsList, this.vActiveConversationId]
      : cActiveIdsList;
    const aAllConvIds = [
      ...new Set([
        ...this.vChatLoadingStates.keys(),
        ...this.vChatStreamingStates.keys(),
        ...this.vAbortControllers.keys(),
        ...this.vProcessingStates.keys(),
        ...this.vConversationStateTimestamps.keys()
      ])
    ];
    const aCleanupCandidates: Array<{ convId: string; lastAccessed: number }> = [];
    for (const cConvId of aAllConvIds) {
      if (cPreserveIds.includes(cConvId)) continue;
      if (this.vChatLoadingStates.get(cConvId)) continue;
      if (this.vChatStreamingStates.has(cConvId)) continue;
      const cTs = this.vConversationStateTimestamps.get(cConvId);
      aCleanupCandidates.push({ convId: cConvId, lastAccessed: cTs?.lastAccessed ?? 0 });
    }
    aCleanupCandidates.sort((pA, pB) => pA.lastAccessed - pB.lastAccessed);
    let vCleanedUp = 0;
    for (const { convId: cConvId, lastAccessed: cLastAccessed } of aCleanupCandidates) {
      if (
        aCleanupCandidates.length - vCleanedUp > cMaxInactiveConversationStates ||
        cNow - cLastAccessed > cInactiveConversationStateMaxAgeMs
      ) {
        this.mCleanupConversationState(cConvId);
        vCleanedUp++;
      }
    }
    return vCleanedUp;
  }
  private mCleanupConversationState(pConvId: string): void {
    const cC = this.vAbortControllers.get(pConvId);
    if (cC && !cC.signal.aborted) cC.abort();
    this.vChatLoadingStates.delete(pConvId);
    this.vChatStreamingStates.delete(pConvId);
    this.vAbortControllers.delete(pConvId);
    this.vProcessingStates.delete(pConvId);
    this.vConversationStateTimestamps.delete(pConvId);
  }
  mGetTrackedConversationCount(): number {
    return new Set([
      ...this.vChatLoadingStates.keys(),
      ...this.vChatStreamingStates.keys(),
      ...this.vAbortControllers.keys(),
      ...this.vProcessingStates.keys()
    ]).size;
  }

  private mGetMessageByIdWithRole(
    pMessageId: string,
    pExpectedRole?: MessageRole
  ): { message: DatabaseMessage; index: number } | null {
    const cIndex = cConversationsStore.mFindMessageIndex(pMessageId);
    if (cIndex === -1) return null;
    const cMessage = cConversationsStore.vActiveMessages[cIndex];
    if (pExpectedRole && cMessage.role !== pExpectedRole) return null;
    return { message: cMessage, index: cIndex };
  }

  async mAddMessage(
    pRole: MessageRole,
    pContent: string,
    pType: MessageType = MessageType.TEXT,
    pParent: string = '-1',
    pExtras?: DatabaseMessageExtra[]
  ): Promise<DatabaseMessage> {
    const cActiveConv = cConversationsStore.vActiveConversation;
    if (!cActiveConv) throw new Error(fTranslate('text_f9edd3337773'));
    let vParentId: string | null = null;
    if (pParent === '-1') {
      const cAm = cConversationsStore.vActiveMessages;
      if (cAm.length > 0) vParentId = cAm[cAm.length - 1].id;
      else {
        const cAll = await cConversationsStore.mGetConversationMessages(cActiveConv.id);
        const cR = cAll.find((pM) => pM.parent === null && pM.type === 'root');
        vParentId = cR ? cR.id : await DatabaseService.mCreateRootMessage(cActiveConv.id);
      }
    } else vParentId = pParent;
    const cMessage = await DatabaseService.mCreateMessageBranch(
      {
        convId: cActiveConv.id,
        role: pRole,
        content: pContent,
        type: pType,
        timestamp: Date.now(),
        toolCalls: '',
        children: [],
        extra: pExtras
      },
      vParentId
    );
    cConversationsStore.mAddMessageToActive(cMessage);
    await cConversationsStore.mUpdateCurrentNode(cMessage.id);
    cConversationsStore.mUpdateConversationTimestamp();
    return cMessage;
  }

  async mAddSystemPrompt(): Promise<void> {
    let vActiveConv = cConversationsStore.vActiveConversation;
    if (!vActiveConv) {
      await cConversationsStore.mCreateConversation();
      vActiveConv = cConversationsStore.vActiveConversation;
    }
    if (!vActiveConv) return;
    try {
      const cAllMessages = await cConversationsStore.mGetConversationMessages(vActiveConv.id);
      const cRootMessage = cAllMessages.find((pM) => pM.type === 'root' && pM.parent === null);
      const cRootId = cRootMessage
        ? cRootMessage.id
        : await DatabaseService.mCreateRootMessage(vActiveConv.id);
      const cExistingSystemMessage = cAllMessages.find(
        (pM) => pM.role === MessageRole.SYSTEM && pM.parent === cRootId
      );
      if (cExistingSystemMessage) {
        this.vPendingEditMessageId = cExistingSystemMessage.id;
        if (!cConversationsStore.vActiveMessages.some((pM) => pM.id === cExistingSystemMessage.id))
          cConversationsStore.vActiveMessages.unshift(cExistingSystemMessage);
        return;
      }
      const cAm = cConversationsStore.vActiveMessages;
      const cFirstActiveMessage = cAm.find((pM) => pM.parent === cRootId);
      const cSystemMessage = await DatabaseService.mCreateSystemMessage(
        vActiveConv.id,
        cSystemMessagePlaceholder,
        cRootId
      );
      if (cFirstActiveMessage) {
        await DatabaseService.mUpdateMessage(cFirstActiveMessage.id, {
          parent: cSystemMessage.id
        });
        await DatabaseService.mUpdateMessage(cSystemMessage.id, {
          children: [cFirstActiveMessage.id]
        });
        const cUpdatedRootChildren = cRootMessage
          ? cRootMessage.children.filter((pId: string) => pId !== cFirstActiveMessage.id)
          : [];
        await DatabaseService.mUpdateMessage(cRootId, {
          children: [
            ...cUpdatedRootChildren.filter((pId: string) => pId !== cSystemMessage.id),
            cSystemMessage.id
          ]
        });
        const cFirstMsgIndex = cConversationsStore.mFindMessageIndex(cFirstActiveMessage.id);
        if (cFirstMsgIndex !== -1)
          cConversationsStore.mUpdateMessageAtIndex(cFirstMsgIndex, {
            parent: cSystemMessage.id
          });
      }
      cConversationsStore.vActiveMessages.unshift(cSystemMessage);
      this.vPendingEditMessageId = cSystemMessage.id;
      cConversationsStore.mUpdateConversationTimestamp();
    } catch (vError) {
      console.error('Failed to add system prompt:', vError);
    }
  }

  async mRemoveSystemPromptPlaceholder(pMessageId: string): Promise<boolean> {
    const cActiveConv = cConversationsStore.vActiveConversation;
    if (!cActiveConv) return false;
    try {
      const cAllMessages = await cConversationsStore.mGetConversationMessages(cActiveConv.id);
      const cSystemMessage = findMessageById(cAllMessages, pMessageId);
      if (!cSystemMessage || cSystemMessage.role !== MessageRole.SYSTEM) return false;
      const cRootMessage = cAllMessages.find((pM) => pM.type === 'root' && pM.parent === null);
      if (!cRootMessage) return false;
      if (cAllMessages.length === 2 && cSystemMessage.children.length === 0) {
        await cConversationsStore.mDeleteConversation(cActiveConv.id);
        return true;
      }
      for (const cChildId of cSystemMessage.children) {
        await DatabaseService.mUpdateMessage(cChildId, { parent: cRootMessage.id });
        const cChildIndex = cConversationsStore.mFindMessageIndex(cChildId);
        if (cChildIndex !== -1)
          cConversationsStore.mUpdateMessageAtIndex(cChildIndex, { parent: cRootMessage.id });
      }
      await DatabaseService.mUpdateMessage(cRootMessage.id, {
        children: [
          ...cRootMessage.children.filter((pId: string) => pId !== pMessageId),
          ...cSystemMessage.children
        ]
      });
      await DatabaseService.mDeleteMessage(pMessageId);
      const cSystemIndex = cConversationsStore.mFindMessageIndex(pMessageId);
      if (cSystemIndex !== -1) cConversationsStore.vActiveMessages.splice(cSystemIndex, 1);
      cConversationsStore.mUpdateConversationTimestamp();
      return false;
    } catch (vError) {
      console.error('Failed to remove system prompt placeholder:', vError);
      return false;
    }
  }

  private async mCreateAssistantMessage(pParentId?: string): Promise<DatabaseMessage> {
    const cActiveConv = cConversationsStore.vActiveConversation;
    if (!cActiveConv) throw new Error(fTranslate('text_f9edd3337773'));
    return await DatabaseService.mCreateMessageBranch(
      {
        convId: cActiveConv.id,
        type: MessageType.TEXT,
        role: MessageRole.ASSISTANT,
        content: '',
        timestamp: Date.now(),
        toolCalls: '',
        children: [],
        model: null
      },
      pParentId || null
    );
  }

  async mSendMessage(pContent: string, pExtras?: DatabaseMessageExtra[]): Promise<void> {
    if (!pContent.trim() && (!pExtras || pExtras.length === 0)) return;
    const cActiveConv = cConversationsStore.vActiveConversation;

    // If agentic loop is running, inject as a steering message instead of starting a new flow
    if (cActiveConv && cAgenticStore.mIsRunning(cActiveConv.id)) {
      cAgenticStore.mInjectSteeringMessage(cActiveConv.id, pContent, pExtras);
      return;
    }

    // If non-agentic streaming is active, queue as a pending message to send after completion
    if (cActiveConv && this.mIsChatLoadingInternal(cActiveConv.id)) {
      this.mInjectPendingMessage(cActiveConv.id, pContent, pExtras);
      return;
    }

    // Cancel any in-flight pre-encode request
    this.mCancelPreEncode();

    // Consume MCP resource attachments - converts them to extras and clears the live store
    const cResourceExtras = cMcpStore.mConsumeResourceAttachmentsAsExtras();
    const cAllExtras =
      cResourceExtras.length > 0 ? [...(pExtras || []), ...cResourceExtras] : pExtras;

    let vIsNewConversation = false;
    if (!cActiveConv) {
      await cConversationsStore.mCreateConversation();
      vIsNewConversation = true;
    }
    const cCurrentConv = cConversationsStore.vActiveConversation;
    if (!cCurrentConv) return;
    this.mShowErrorDialog(null);
    this.mSetChatLoading(cCurrentConv.id, true);
    this.mClearChatStreaming(cCurrentConv.id);
    try {
      let vParentIdForUserMessage: string | undefined;
      if (vIsNewConversation) {
        const cRootId = await DatabaseService.mCreateRootMessage(cCurrentConv.id);
        const cCurrentConfig = fConfig();
        const cSystemPrompt = cCurrentConfig.systemMessage?.toString().trim();
        if (cSystemPrompt) {
          const cSystemMessage = await DatabaseService.mCreateSystemMessage(
            cCurrentConv.id,
            cSystemPrompt,
            cRootId
          );
          cConversationsStore.mAddMessageToActive(cSystemMessage);
          vParentIdForUserMessage = cSystemMessage.id;
        } else vParentIdForUserMessage = cRootId;
      }
      const cUserMessage = await this.mAddMessage(
        MessageRole.USER,
        pContent,
        MessageType.TEXT,
        vParentIdForUserMessage ?? '-1',
        cAllExtras
      );
      if (vIsNewConversation && pContent)
        await cConversationsStore.mUpdateConversationName(
          cCurrentConv.id,
          generateConversationTitle(pContent, Boolean(fConfig().titleGenerationUseFirstLine))
        );
      const cAssistantMessage = await this.mCreateAssistantMessage(cUserMessage.id);
      cConversationsStore.mAddMessageToActive(cAssistantMessage);
      await this.mStreamChatCompletion(
        cConversationsStore.vActiveMessages.slice(0, -1),
        cAssistantMessage,
        undefined,
        undefined,
        undefined,
        fConfig().titleGenerationUseLLM && vIsNewConversation ? pContent : undefined
      );
    } catch (vError) {
      if (isAbortError(vError)) {
        this.mSetChatLoading(cCurrentConv.id, false);
        return;
      }
      console.error('Failed to send message:', vError);
      this.mSetChatLoading(cCurrentConv.id, false);
      const cDialogType =
        vError instanceof Error && vError.name === 'TimeoutError'
          ? ErrorDialogType.TIMEOUT
          : ErrorDialogType.SERVER;
      const cContextInfo = (
        vError as Error & { contextInfo?: { n_prompt_tokens: number; n_ctx: number } }
      ).contextInfo;
      this.mShowErrorDialog({
        type: cDialogType,
        message: vError instanceof Error ? vError.message : fTranslate('text_27c2ccd962c2'),
        contextInfo: cContextInfo
      });
    }
  }

  private async mStreamChatCompletion(
    pAllMessages: DatabaseMessage[],
    pAssistantMessage: DatabaseMessage,
    pOnComplete?: (pContent: string) => Promise<void>,
    pOnError?: (pError: Error) => void,
    pModelOverride?: string | null,
    pFirstUserMessageContent?: string
  ): Promise<void> {
    // the ::model suffix in the stream identity is only for router mode, where it routes to the
    // owning child. in single-model mode the identity stays the bare conv id so that attach, stop
    // and reattach all agree, regardless of fresh send vs regenerate passing a resolved model
    let vEffectiveModel: string | null | undefined = undefined;

    if (fIsRouterMode()) {
      const cConversationModel = this.mGetConversationModel(pAllMessages);
      vEffectiveModel = pModelOverride || fSelectedModelName() || cConversationModel;
    }

    if (fIsRouterMode() && vEffectiveModel) {
      if (!cModelsStore.mGetModelProps(vEffectiveModel))
        await cModelsStore.mFetchModelProps(vEffectiveModel);
    }

    // Mutable state for the current message being streamed
    let vCurrentMessageId = pAssistantMessage.id;
    let vStreamedContent = '';
    let vStreamedReasoningContent = '';
    let vResolvedModel: string | null = null;
    let vModelPersisted = false;
    const cConvId = pAssistantMessage.convId;
    // Tracks the last message created in this flow. Used as the parent for the next
    // turn's assistant message so createAssistantMessage does not have to read
    // conversationsStore.activeMessages, which may belong to a different conversation
    // after the user navigates while the loop is still running.
    let vLastCreatedInFlow = vCurrentMessageId;
    // freeze the POST identity from t0 so a stop cancels with the exact session key,
    // never a stale or empty model resolved later
    this.mSetChatStreaming(cConvId, vStreamedContent, vCurrentMessageId, vEffectiveModel);

    const fRecordModel = (
      pModelName: string | null | undefined,
      pPersistImmediately = true
    ): void => {
      if (!pModelName) return;
      const cN = normalizeModelName(pModelName);
      if (!cN || cN === vResolvedModel) return;
      vResolvedModel = cN;
      const cIdx = cConversationsStore.mFindMessageIndex(vCurrentMessageId);
      cConversationsStore.mUpdateMessageAtIndex(cIdx, { model: cN });
      if (pPersistImmediately && !vModelPersisted) {
        vModelPersisted = true;
        DatabaseService.mUpdateMessage(vCurrentMessageId, { model: cN }).catch(() => {
          vModelPersisted = false;
          vResolvedModel = null;
        });
      }
    };

    let vCompletionIdRecorded = false;
    const fRecordCompletionId = (pId: string): void => {
      if (!pId || vCompletionIdRecorded) return;
      vCompletionIdRecorded = true;
      const cIdx = cConversationsStore.mFindMessageIndex(vCurrentMessageId);
      cConversationsStore.mUpdateMessageAtIndex(cIdx, { completionId: pId });
      DatabaseService.mUpdateMessage(vCurrentMessageId, { completionId: pId }).catch(() => {
        vCompletionIdRecorded = false;
      });
    };

    const fUpdateStreamingUI = () => {
      this.mSetChatStreaming(cConvId, vStreamedContent, vCurrentMessageId, vEffectiveModel);
      const cIdx = cConversationsStore.mFindMessageIndex(vCurrentMessageId);
      cConversationsStore.mUpdateMessageAtIndex(cIdx, { content: vStreamedContent });
    };

    const fCleanupStreamingState = () => {
      this.mSetStreamingActive(false);
      this.mSetChatLoading(cConvId, false);
      this.mClearChatStreaming(cConvId, vCurrentMessageId);
      this.mSetProcessingState(cConvId, null);
      this.mSetQueueStatus(cConvId, null);
    };

    this.mSetStreamingActive(true);
    this.mSetActiveProcessingConversation(cConvId);
    const cAbortController = this.mGetOrCreateAbortController(cConvId);

    const dStreamCallbacks: ChatStreamCallbacks = {
      onChunk: (pChunk: string) => {
        vStreamedContent += pChunk;
        fUpdateStreamingUI();
        this.mSetChatReasoning(cConvId, false);
      },
      onQueue: (pStatus: ApiQueueStatus | null) => this.mSetQueueStatus(cConvId, pStatus),
      onReasoningChunk: (pChunk: string) => {
        vStreamedReasoningContent += pChunk;
        // mark streaming state so a stop mid-thinking can persist the partial reasoning
        this.mSetChatStreaming(cConvId, vStreamedContent, vCurrentMessageId, vEffectiveModel);
        const cIdx = cConversationsStore.mFindMessageIndex(vCurrentMessageId);
        cConversationsStore.mUpdateMessageAtIndex(cIdx, {
          reasoningContent: vStreamedReasoningContent
        });
        this.mSetChatReasoning(cConvId, true);
      },
      onToolCallsStreaming: (pToolCalls) => {
        const cIdx = cConversationsStore.mFindMessageIndex(vCurrentMessageId);
        cConversationsStore.mUpdateMessageAtIndex(cIdx, {
          toolCalls: JSON.stringify(pToolCalls)
        });
      },
      onAttachments: (pMessageId: string, pExtras: DatabaseMessageExtra[]) => {
        if (!pExtras.length) return;
        const cIdx = cConversationsStore.mFindMessageIndex(pMessageId);
        if (cIdx === -1) return;
        const cMsg = cConversationsStore.vActiveMessages[cIdx];
        const aUpdatedExtras = [...(cMsg.extra || []), ...pExtras];
        cConversationsStore.mUpdateMessageAtIndex(cIdx, { extra: aUpdatedExtras });
        DatabaseService.mUpdateMessage(pMessageId, { extra: aUpdatedExtras }).catch(console.error);
      },
      onModel: (pModelName: string) => fRecordModel(pModelName),
      onCompletionId: (pId: string) => fRecordCompletionId(pId),
      onTurnComplete: (pIntermediateTimings: ChatMessageTimings) => {
        // Update the first assistant message with cumulative agentic timings
        const cIdx = cConversationsStore.mFindMessageIndex(pAssistantMessage.id);
        cConversationsStore.mUpdateMessageAtIndex(cIdx, { timings: pIntermediateTimings });
      },
      onTimings: (pTimings?: ChatMessageTimings, pPromptProgress?: ChatMessagePromptProgress) => {
        const cTokensPerSecond =
          pTimings?.predicted_ms && pTimings?.predicted_n
            ? (pTimings.predicted_n / pTimings.predicted_ms) * 1000
            : 0;
        this.mUpdateProcessingStateFromTimings(
          {
            prompt_n: pTimings?.prompt_n || 0,
            prompt_ms: pTimings?.prompt_ms,
            predicted_n: pTimings?.predicted_n || 0,
            predicted_per_second: cTokensPerSecond,
            cache_n: pTimings?.cache_n || 0,
            prompt_progress: pPromptProgress
          },
          cConvId
        );
      },
      onAssistantTurnComplete: async (
        pContent: string,
        pReasoningContent: string | undefined,
        pTimings: ChatMessageTimings | undefined,
        pToolCalls: import('$lib/types/api').ApiChatCompletionToolCall[] | undefined
      ) => {
        const dUpdateData: Record<string, unknown> = {
          content: pContent,
          reasoningContent: pReasoningContent || undefined,
          toolCalls: pToolCalls ? JSON.stringify(pToolCalls) : '',
          timings: pTimings
        };
        if (vResolvedModel && !vModelPersisted) dUpdateData.model = vResolvedModel;
        await DatabaseService.mUpdateMessage(vCurrentMessageId, dUpdateData);
        const cIdx = cConversationsStore.mFindMessageIndex(vCurrentMessageId);
        const dUiUpdate: Partial<DatabaseMessage> = {
          content: pContent,
          reasoningContent: pReasoningContent || undefined,
          toolCalls: pToolCalls ? JSON.stringify(pToolCalls) : ''
        };
        if (pTimings) dUiUpdate.timings = pTimings;
        if (vResolvedModel) dUiUpdate.model = vResolvedModel;
        // touch the active ui array and node pointer only when this conversation
        // is displayed; otherwise persist the node move straight to the db so a
        // foreign conv's currNode stays untouched
        if (cConversationsStore.vActiveConversation?.id === cConvId) {
          cConversationsStore.mUpdateMessageAtIndex(cIdx, dUiUpdate);
          await cConversationsStore.mUpdateCurrentNode(vCurrentMessageId);
        } else {
          await DatabaseService.mUpdateCurrentNode(cConvId, vCurrentMessageId);
        }
      },
      createToolResultMessage: async (
        pToolCallId: string,
        pContent: string,
        pExtras?: DatabaseMessageExtra[]
      ) => {
        const cMsg = await DatabaseService.mCreateMessageBranch(
          {
            convId: cConvId,
            type: MessageType.TEXT,
            role: MessageRole.TOOL,
            content: pContent,
            toolCallId: pToolCallId,
            timestamp: Date.now(),
            toolCalls: '',
            children: [],
            extra: pExtras
          },
          vCurrentMessageId
        );
        // mirror into the active store and move the node pointer only when this
        // conversation is displayed; otherwise persist the node move straight to
        // the db for the owning conv so a foreign conv's currNode stays untouched
        if (cConversationsStore.vActiveConversation?.id === cConvId) {
          cConversationsStore.mAddMessageToActive(cMsg);
          await cConversationsStore.mUpdateCurrentNode(cMsg.id);
        } else {
          await DatabaseService.mUpdateCurrentNode(cConvId, cMsg.id);
        }
        vLastCreatedInFlow = cMsg.id;
        return cMsg;
      },
      updateToolResultMessage: async (
        pMessageId: string,
        pContent: string,
        pExtras?: DatabaseMessageExtra[]
      ) => {
        // Persist latest content + merged extras; mirror into the active
        // store so the chat view sees live updates for streaming tools
        // (e.g. exec_shell_command). The existing tool message node
        // pointer stays put - the renderer is already scoped to it.
        const dUpdates: Partial<DatabaseMessage> = { content: pContent };
        if (pExtras) {
          const cIdx = cConversationsStore.mFindMessageIndex(pMessageId);
          const cExisting =
            cIdx >= 0 ? (cConversationsStore.vActiveMessages[cIdx]?.extra ?? []) : [];
          const aMerged = [...cExisting, ...pExtras];
          dUpdates.extra = aMerged;
        }
        if (cConversationsStore.vActiveConversation?.id === cConvId) {
          const cIdx = cConversationsStore.mFindMessageIndex(pMessageId);
          if (cIdx >= 0) cConversationsStore.mUpdateMessageAtIndex(cIdx, dUpdates);
        }
        await DatabaseService.mUpdateMessage(pMessageId, dUpdates);
      },
      createAssistantMessage: async () => {
        // Reset streaming state for new message
        vStreamedContent = '';
        vStreamedReasoningContent = '';

        const cMsg = await DatabaseService.mCreateMessageBranch(
          {
            convId: cConvId,
            type: MessageType.TEXT,
            role: MessageRole.ASSISTANT,
            content: '',
            timestamp: Date.now(),
            toolCalls: '',
            children: [],
            model: vResolvedModel
          },
          vLastCreatedInFlow
        );
        if (cConversationsStore.vActiveConversation?.id === cConvId) {
          cConversationsStore.mAddMessageToActive(cMsg);
        }
        vCurrentMessageId = cMsg.id;
        vLastCreatedInFlow = cMsg.id;
        return cMsg;
      },
      onFlowComplete: (pFinalTimings?: ChatMessageTimings) => {
        if (pFinalTimings) {
          const cIdx = cConversationsStore.mFindMessageIndex(pAssistantMessage.id);

          cConversationsStore.mUpdateMessageAtIndex(cIdx, { timings: pFinalTimings });
          DatabaseService.mUpdateMessage(pAssistantMessage.id, {
            timings: pFinalTimings
          }).catch(console.error);
        }

        fCleanupStreamingState();

        if (pOnComplete) pOnComplete(vStreamedContent);
        if (fIsRouterMode()) cModelsStore.mFetchRouterModels().catch(console.error);
        // Pre-encode conversation in KV cache for faster next turn
        if (fConfig().preEncodeConversation) {
          this.mTriggerPreEncode(
            pAllMessages,
            pAssistantMessage,
            vStreamedContent,
            vEffectiveModel,
            !!fConfig().excludeReasoningFromContext
          );
        }
      },
      onError: async (pError: Error) => {
        this.mSetStreamingActive(false);
        if (isAbortError(pError)) {
          fCleanupStreamingState();
          // If aborted with a pending message (e.g. "Send immediately"), re-send it
          const cPending = this.mConsumePendingMessage(cConvId);
          if (cPending) {
            this.mSendMessage(cPending.content, cPending.extras);
          }
          return;
        }
        console.error('Streaming error:', pError);
        // keep whatever was streamed so far, the message stays in memory and in DB
        await this.mSavePartialResponseIfNeeded(cConvId);
        fCleanupStreamingState();
        this.mClearPendingMessage(cConvId);

        const cContextInfo = (
          pError as Error & { contextInfo?: { n_prompt_tokens: number; n_ctx: number } }
        ).contextInfo;
        this.mShowErrorDialog({
          type: pError.name === 'TimeoutError' ? ErrorDialogType.TIMEOUT : ErrorDialogType.SERVER,
          message: pError.message,
          contextInfo: cContextInfo
        });
        if (pOnError) pOnError(pError);
      }
    };

    const cPerChatOverrides = cConversationsStore.mGetAllMcpServerOverrides();

    {
      const cAgenticResult = await cAgenticStore.mRunAgenticFlow({
        conversationId: cConvId,
        messages: pAllMessages,
        options: {
          ...this.mGetApiOptions(),
          ...(vEffectiveModel ? { model: vEffectiveModel } : {})
        },
        callbacks: dStreamCallbacks,
        signal: cAbortController.signal,
        perChatOverrides: cPerChatOverrides
      });
      if (cAgenticResult.handled) {
        // Generate LLM based title for new conversations after agentic flow completes
        if (pFirstUserMessageContent) {
          await this.mGenerateTitleWithLLM(pFirstUserMessageContent, vStreamedContent, cConvId);
        }
        // Check if there's a pending steering message to re-send
        const cPending = cAgenticStore.mConsumePendingSteeringMessage(cConvId);
        if (cPending) {
          await this.mSendMessage(cPending.content, cPending.extras);
        }
        return;
      }
    }

    await ChatService.mSendMessage(
      pAllMessages,
      {
        ...this.mGetApiOptions(),
        ...(vEffectiveModel ? { model: vEffectiveModel } : {}),
        stream: true,
        onChunk: dStreamCallbacks.onChunk,
        onReasoningChunk: dStreamCallbacks.onReasoningChunk,
        onModel: dStreamCallbacks.onModel,
        onCompletionId: dStreamCallbacks.onCompletionId,
        onTimings: dStreamCallbacks.onTimings,
        onQueue: dStreamCallbacks.onQueue,
        onConnectionState: (pState: StreamConnectionState) => {
          if (cConvId === cConversationsStore.vActiveConversation?.id) {
            this.vStreamConnectionState = pState;
          }
        },
        onComplete: async (
          pFinalContent?: string,
          pReasoningContent?: string,
          pTimings?: ChatMessageTimings,
          pToolCalls?: string
        ) => {
          const cContent = vStreamedContent || pFinalContent || '';
          const cReasoning = vStreamedReasoningContent || pReasoningContent;
          const dUpdateData: Record<string, unknown> = {
            content: cContent,
            reasoningContent: cReasoning || undefined,
            toolCalls: pToolCalls || '',
            timings: pTimings
          };
          if (vResolvedModel && !vModelPersisted) dUpdateData.model = vResolvedModel;
          await DatabaseService.mUpdateMessage(vCurrentMessageId, dUpdateData);
          const cIdx = cConversationsStore.mFindMessageIndex(vCurrentMessageId);
          const dUiUpdate: Partial<DatabaseMessage> = {
            content: cContent,
            reasoningContent: cReasoning || undefined,
            toolCalls: pToolCalls || ''
          };
          if (pTimings) dUiUpdate.timings = pTimings;
          if (vResolvedModel) dUiUpdate.model = vResolvedModel;
          cConversationsStore.mUpdateMessageAtIndex(cIdx, dUiUpdate);
          await cConversationsStore.mUpdateCurrentNode(vCurrentMessageId);
          fCleanupStreamingState();
          if (pOnComplete) await pOnComplete(cContent);
          if (fIsRouterMode()) cModelsStore.mFetchRouterModels().catch(console.error);

          // Generate LLM based title for new conversations (avoids stale reference
          // issue when user switches conversations while streaming)
          if (pFirstUserMessageContent) {
            await this.mGenerateTitleWithLLM(pFirstUserMessageContent, vStreamedContent, cConvId);
          }

          // Check if there's a pending message queued during streaming
          const cPending = this.mConsumePendingMessage(cConvId);
          if (cPending) {
            await this.mSendMessage(cPending.content, cPending.extras);
          }
        },
        onError: dStreamCallbacks.onError
      },
      cConvId,
      cAbortController.signal
    );
  }

  async mStopGeneration(): Promise<void> {
    const cActiveConv = cConversationsStore.vActiveConversation;
    if (!cActiveConv) return;
    await this.mStopGenerationForChat(cActiveConv.id);
  }
  async mStopGenerationForChat(pConvId: string): Promise<void> {
    await this.mSavePartialResponseIfNeeded(pConvId);
    this.mSetStreamingActive(false);
    // tell the server to stop the generation, not just drop the HTTP socket. without this the
    // detached drain keeps producing tokens until eos or max_tokens. use the frozen identity
    // captured when the session started, not the live dropdown
    const cStreamStateForStop = this.vChatStreamingStates.get(pConvId);
    const cModelForStop = cStreamStateForStop?.model ?? ChatService.mGetStreamState(pConvId)?.model;
    void ChatService.mCancelServerStream(pConvId, cModelForStop);
    // an explicit stop leaves nothing to resume and kills a pending resume retry
    ChatService.mClearStreamState(pConvId);
    const cRetryTimer = this.vResumeRetryTimers.get(pConvId);
    if (cRetryTimer !== undefined) {
      clearTimeout(cRetryTimer);
      this.vResumeRetryTimers.delete(pConvId);
    }
    this.vResumePendingConvs.delete(pConvId);
    this.mAbortRequest(pConvId);
    this.mSetChatLoading(pConvId, false);
    this.mClearChatStreaming(pConvId);
    this.mSetProcessingState(pConvId, null);
    this.mClearPendingMessage(pConvId);
  }

  private async mGenerateTitleWithLLM(
    pUserContent: string,
    pAssistantContent: string,
    pConvId: string
  ): Promise<void> {
    const cEffectiveModel =
      fIsRouterMode() && fSelectedModelName() ? fSelectedModelName() : undefined;
    const cConfigValue = fConfig();
    const cTitlePromptTemplate =
      typeof cConfigValue.titleGenerationPrompt === 'string' &&
      cConfigValue.titleGenerationPrompt.trim()
        ? cConfigValue.titleGenerationPrompt
        : cTitleGeneration.DEFAULT_PROMPT;

    const cTitlePrompt = cTitlePromptTemplate
      .replace('{{USER}}', String(pUserContent || ''))
      .replace('{{ASSISTANT}}', String(pAssistantContent || ''));

    const dTitleMessage: ApiChatMessageData = {
      role: MessageRole.USER,
      content: cTitlePrompt
    };

    const cTitleResponse = await ChatService.mGenerateTitle(dTitleMessage, cEffectiveModel);

    if (!cTitleResponse) {
      return;
    }

    let vCleanTitle = cTitleResponse.trim();
    vCleanTitle = vCleanTitle
      .replace(cTitleGeneration.PREFIX_PATTERN, '')
      .replace(cTitleGeneration.QUOTE_PATTERN, '')
      .trim();
    if (!vCleanTitle || vCleanTitle.length < cTitleGeneration.MIN_LENGTH) {
      const cFirstLine = pUserContent.split('\n').find((pL) => pL.trim().length > 0);
      vCleanTitle = cFirstLine ? cFirstLine.trim() : cTitleGeneration.FALLBACK;
    }
    if (vCleanTitle && vCleanTitle.length >= cTitleGeneration.MIN_LENGTH) {
      await cConversationsStore.mUpdateConversationName(pConvId, vCleanTitle);
    }
  }

  private async mSavePartialResponseIfNeeded(pConvId?: string): Promise<void> {
    const cConversationId = pConvId || cConversationsStore.vActiveConversation?.id;
    if (!cConversationId) return;
    const cStreamingState = this.mGetChatStreaming(cConversationId);
    if (!cStreamingState) return;
    const cMessages =
      cConversationId === cConversationsStore.vActiveConversation?.id
        ? cConversationsStore.vActiveMessages
        : await cConversationsStore.mGetConversationMessages(cConversationId);
    if (!cMessages.length) return;
    const cLastMessage = cMessages[cMessages.length - 1];
    if (cLastMessage?.role !== MessageRole.ASSISTANT) return;

    const cPartialContent = cStreamingState.response;
    const cPartialReasoning = cLastMessage.reasoningContent || '';
    // snapshot the streamed tool calls before clearing so we still know whether
    // anything was captured when deciding to skip the DB write below
    const cHadPartialToolCalls = !!cLastMessage.toolCalls?.trim();

    // nothing to persist when content, reasoning, and streamed tool calls are all empty
    // (e.g. stop before any token). otherwise drop the partial tool call and write whatever
    // was streamed: incomplete arguments (truncated JSON, missing closing quote) would
    // otherwise be re-sent to the server on the next turn and rejected.
    if (!cPartialContent.trim() && !cPartialReasoning.trim() && !cHadPartialToolCalls) return;

    try {
      const dUpdateData: {
        content?: string;
        reasoningContent?: string;
        toolCalls?: string;
        timings?: ChatMessageTimings;
      } = {
        toolCalls: ''
      };
      if (cPartialContent.trim()) dUpdateData.content = cPartialContent;
      if (cPartialReasoning.trim()) dUpdateData.reasoningContent = cPartialReasoning;
      const cLastKnownState = this.mGetProcessingState(cConversationId);
      if (cLastKnownState) {
        dUpdateData.timings = {
          prompt_n: cLastKnownState.promptTokens || 0,
          prompt_ms: cLastKnownState.promptMs,
          predicted_n: cLastKnownState.tokensDecoded || 0,
          cache_n: cLastKnownState.cacheTokens || 0,
          predicted_ms:
            cLastKnownState.tokensPerSecond && cLastKnownState.tokensDecoded
              ? (cLastKnownState.tokensDecoded / cLastKnownState.tokensPerSecond) * 1000
              : undefined
        };
      }
      await DatabaseService.mUpdateMessage(cLastMessage.id, dUpdateData);
      cLastMessage.content = cPartialContent;
      // mirror the drop into the in-memory message so the next request sent via
      // sendMessage (queued pending, Send immediately, or manual follow-up) reads
      // the cleared value, not whatever the streaming widget had been showing
      cLastMessage.toolCalls = '';
      if (dUpdateData.timings) cLastMessage.timings = dUpdateData.timings;
    } catch (vError) {
      cLastMessage.content = cPartialContent;
      cLastMessage.toolCalls = '';
      console.error('Failed to save partial response:', vError);
    }
  }

  async mUpdateMessage(pMessageId: string, pNewContent: string): Promise<void> {
    const cActiveConv = cConversationsStore.vActiveConversation;
    if (!cActiveConv) return;
    if (this.mIsChatLoadingInternal(cActiveConv.id)) await this.mStopGeneration();
    const cResult = this.mGetMessageByIdWithRole(pMessageId, MessageRole.USER);
    if (!cResult) return;
    const { message: cMessageToUpdate, index: cMessageIndex } = cResult;
    const cOriginalContent = cMessageToUpdate.content;
    try {
      const cAllMessages = await cConversationsStore.mGetConversationMessages(cActiveConv.id);
      const cRootMessage = cAllMessages.find((pM) => pM.type === 'root' && pM.parent === null);
      const cIsFirstUserMessage = cRootMessage && cMessageToUpdate.parent === cRootMessage.id;
      cConversationsStore.mUpdateMessageAtIndex(cMessageIndex, { content: pNewContent });
      await DatabaseService.mUpdateMessage(pMessageId, { content: pNewContent });
      if (cIsFirstUserMessage && pNewContent.trim())
        await cConversationsStore.mUpdateConversationName(
          cActiveConv.id,
          generateConversationTitle(pNewContent, Boolean(fConfig().titleGenerationUseFirstLine))
        );
      const cMessagesToRemove = cConversationsStore.vActiveMessages.slice(cMessageIndex + 1);
      if (cMessagesToRemove.length > 0)
        await DatabaseService.mDeleteMessageCascading(cActiveConv.id, cMessagesToRemove[0].id);
      cConversationsStore.mSliceActiveMessages(cMessageIndex + 1);
      cConversationsStore.mUpdateConversationTimestamp();
      this.mSetChatLoading(cActiveConv.id, true);
      this.mClearChatStreaming(cActiveConv.id);
      const cAssistantMessage = await this.mCreateAssistantMessage();
      cConversationsStore.mAddMessageToActive(cAssistantMessage);
      await cConversationsStore.mUpdateCurrentNode(cAssistantMessage.id);
      await this.mStreamChatCompletion(
        cConversationsStore.vActiveMessages.slice(0, -1),
        cAssistantMessage,
        undefined,
        () => {
          cConversationsStore.mUpdateMessageAtIndex(
            cConversationsStore.mFindMessageIndex(pMessageId),
            {
              content: cOriginalContent
            }
          );
        }
      );
    } catch (vError) {
      if (!isAbortError(vError)) console.error('Failed to update message:', vError);
    }
  }

  async mRegenerateMessage(pMessageId: string): Promise<void> {
    const cActiveConv = cConversationsStore.vActiveConversation;
    if (!cActiveConv || this.mIsChatLoadingInternal(cActiveConv.id)) return;
    this.mCancelPreEncode();
    const cResult = this.mGetMessageByIdWithRole(pMessageId, MessageRole.ASSISTANT);
    if (!cResult) return;
    const { index: cMessageIndex } = cResult;
    try {
      const cMessagesToRemove = cConversationsStore.vActiveMessages.slice(cMessageIndex);
      await DatabaseService.mDeleteMessageCascading(cActiveConv.id, cMessagesToRemove[0].id);
      cConversationsStore.mSliceActiveMessages(cMessageIndex);
      cConversationsStore.mUpdateConversationTimestamp();
      this.mSetChatLoading(cActiveConv.id, true);
      this.mClearChatStreaming(cActiveConv.id);
      const cParentMessageId =
        cConversationsStore.vActiveMessages.length > 0
          ? cConversationsStore.vActiveMessages[cConversationsStore.vActiveMessages.length - 1].id
          : undefined;
      const cAssistantMessage = await this.mCreateAssistantMessage(cParentMessageId);
      cConversationsStore.mAddMessageToActive(cAssistantMessage);
      await this.mStreamChatCompletion(
        cConversationsStore.vActiveMessages.slice(0, -1),
        cAssistantMessage
      );
    } catch (vError) {
      if (!isAbortError(vError)) console.error('Failed to regenerate message:', vError);
      this.mSetChatLoading(cActiveConv?.id || '', false);
    }
  }

  async mRegenerateMessageWithBranching(
    pMessageId: string,
    pModelOverride?: string
  ): Promise<void> {
    const cActiveConv = cConversationsStore.vActiveConversation;
    if (!cActiveConv || this.mIsChatLoadingInternal(cActiveConv.id)) return;
    this.mCancelPreEncode();
    try {
      const cIdx = cConversationsStore.mFindMessageIndex(pMessageId);
      if (cIdx === -1) return;
      const cMsg = cConversationsStore.vActiveMessages[cIdx];
      if (cMsg.role !== MessageRole.ASSISTANT) return;
      const cAllMessages = await cConversationsStore.mGetConversationMessages(cActiveConv.id);
      const cParentMessage = findMessageById(cAllMessages, cMsg.parent);
      if (!cParentMessage) return;
      this.mSetChatLoading(cActiveConv.id, true);
      this.mClearChatStreaming(cActiveConv.id);
      const cNewAssistantMessage = await DatabaseService.mCreateMessageBranch(
        {
          convId: cMsg.convId,
          type: cMsg.type,
          timestamp: Date.now(),
          role: cMsg.role,
          content: '',
          toolCalls: '',
          children: [],
          model: null
        },
        cParentMessage.id
      );
      await cConversationsStore.mUpdateCurrentNode(cNewAssistantMessage.id);
      cConversationsStore.mUpdateConversationTimestamp();
      await cConversationsStore.mRefreshActiveMessages();
      const cConversationPath = filterByLeafNodeId(
        cAllMessages,
        cParentMessage.id,
        false
      ) as DatabaseMessage[];
      const cModelToUse = pModelOverride || cMsg.model || undefined;
      await this.mStreamChatCompletion(
        cConversationPath,
        cNewAssistantMessage,
        undefined,
        undefined,
        cModelToUse
      );
    } catch (vError) {
      if (!isAbortError(vError))
        console.error('Failed to regenerate message with branching:', vError);
      this.mSetChatLoading(cActiveConv?.id || '', false);
    }
  }

  async mGetDeletionInfo(pMessageId: string): Promise<{
    totalCount: number;
    userMessages: number;
    assistantMessages: number;
    messageTypes: string[];
  }> {
    const cActiveConv = cConversationsStore.vActiveConversation;
    if (!cActiveConv)
      return { totalCount: 0, userMessages: 0, assistantMessages: 0, messageTypes: [] };
    const cAllMessages = await cConversationsStore.mGetConversationMessages(cActiveConv.id);
    const cMessageToDelete = findMessageById(cAllMessages, pMessageId);

    // For system messages, don't count descendants as they will be preserved (reparented to root)
    if (cMessageToDelete?.role === MessageRole.SYSTEM) {
      const cMessagesToDelete = cAllMessages.filter((pM) => pM.id === pMessageId);
      let vUserMessages = 0,
        vAssistantMessages = 0;
      const aMessageTypes: string[] = [];

      for (const cMsg of cMessagesToDelete) {
        if (cMsg.role === MessageRole.USER) {
          vUserMessages++;
          if (!aMessageTypes.includes('user message')) aMessageTypes.push('user message');
        } else if (cMsg.role === MessageRole.ASSISTANT) {
          vAssistantMessages++;
          if (!aMessageTypes.includes('assistant response'))
            aMessageTypes.push('assistant response');
        }
      }

      return {
        totalCount: 1,
        userMessages: vUserMessages,
        assistantMessages: vAssistantMessages,
        messageTypes: aMessageTypes
      };
    }

    const cDescendants = findDescendantMessages(cAllMessages, pMessageId);
    const aAllToDelete = [pMessageId, ...cDescendants];
    const cMessagesToDelete = cAllMessages.filter((pM) => aAllToDelete.includes(pM.id));
    let vUserMessages = 0,
      vAssistantMessages = 0;
    const aMessageTypes: string[] = [];

    for (const cMsg of cMessagesToDelete) {
      if (cMsg.role === MessageRole.USER) {
        vUserMessages++;
        if (!aMessageTypes.includes('user message')) aMessageTypes.push('user message');
      } else if (cMsg.role === MessageRole.ASSISTANT) {
        vAssistantMessages++;
        if (!aMessageTypes.includes('assistant response')) aMessageTypes.push('assistant response');
      }
    }

    return {
      totalCount: aAllToDelete.length,
      userMessages: vUserMessages,
      assistantMessages: vAssistantMessages,
      messageTypes: aMessageTypes
    };
  }

  async mDeleteMessage(pMessageId: string): Promise<void> {
    const cActiveConv = cConversationsStore.vActiveConversation;
    if (!cActiveConv) return;
    try {
      const cAllMessages = await cConversationsStore.mGetConversationMessages(cActiveConv.id);
      const cMessageToDelete = findMessageById(cAllMessages, pMessageId);

      if (!cMessageToDelete) return;

      const cCurrentPath = filterByLeafNodeId(cAllMessages, cActiveConv.currNode || '', false);
      const cIsInCurrentPath = cCurrentPath.some((pM) => pM.id === pMessageId);

      if (cIsInCurrentPath && cMessageToDelete.parent) {
        const cSiblings = cAllMessages.filter(
          (pM) => pM.parent === cMessageToDelete.parent && pM.id !== pMessageId
        );

        if (cSiblings.length > 0) {
          const cLatestSibling = cSiblings.reduce((pLatest, pSibling) =>
            pSibling.timestamp > pLatest.timestamp ? pSibling : pLatest
          );

          await cConversationsStore.mUpdateCurrentNode(
            findLeafNode(cAllMessages, cLatestSibling.id)
          );
        } else if (cMessageToDelete.parent) {
          await cConversationsStore.mUpdateCurrentNode(
            findLeafNode(cAllMessages, cMessageToDelete.parent)
          );
        }
      }

      await DatabaseService.mDeleteMessageCascading(cActiveConv.id, pMessageId);
      await cConversationsStore.mRefreshActiveMessages();

      cConversationsStore.mUpdateConversationTimestamp();
    } catch (vError) {
      console.error('Failed to delete message:', vError);
    }
  }

  /**
   * Open a fresh assistant turn anchored at the last tool result of a resolved
   * agentic round and let streamChatCompletion route through runAgenticFlow.
   * Used by continueAssistantMessage when classifyContinueIntent returns
   * next_turn, meaning the target assistant already has its tool_calls paired
   * with trailing tool results and the next thing to generate is a brand new
   * turn rather than a token level continuation.
   */
  private async mContinueAsNextAgenticTurn(pAnchorIndex: number): Promise<void> {
    const cActiveConv = cConversationsStore.vActiveConversation;
    if (!cActiveConv) return;
    const cAnchor = cConversationsStore.vActiveMessages[pAnchorIndex];
    if (!cAnchor) return;
    this.mCancelPreEncode();
    this.mSetChatLoading(cActiveConv.id, true);
    this.mClearChatStreaming(cActiveConv.id);
    try {
      const cAllMessages = await cConversationsStore.mGetConversationMessages(cActiveConv.id);
      const cAnchorMessage = findMessageById(cAllMessages, cAnchor.id);
      if (!cAnchorMessage) {
        this.mSetChatLoading(cActiveConv.id, false);
        return;
      }
      const cNewAssistantMessage = await DatabaseService.mCreateMessageBranch(
        {
          convId: cActiveConv.id,
          type: MessageType.TEXT,
          timestamp: Date.now(),
          role: MessageRole.ASSISTANT,
          content: '',
          toolCalls: '',
          children: [],
          model: null
        },
        cAnchorMessage.id
      );
      await cConversationsStore.mUpdateCurrentNode(cNewAssistantMessage.id);
      cConversationsStore.mUpdateConversationTimestamp();
      await cConversationsStore.mRefreshActiveMessages();
      const cConversationPath = filterByLeafNodeId(
        cAllMessages,
        cAnchorMessage.id,
        false
      ) as DatabaseMessage[];
      await this.mStreamChatCompletion(cConversationPath, cNewAssistantMessage);
    } catch (vError) {
      if (!isAbortError(vError)) console.error('Failed to continue agentic turn:', vError);
      this.mSetChatLoading(cActiveConv.id, false);
    }
  }

  async mContinueAssistantMessage(pMessageId: string): Promise<void> {
    const cActiveConv = cConversationsStore.vActiveConversation;
    if (!cActiveConv || this.mIsChatLoadingInternal(cActiveConv.id)) return;
    const cResult = this.mGetMessageByIdWithRole(pMessageId, MessageRole.ASSISTANT);

    if (!cResult) return;

    const { message: cMsg, index: cIdx } = cResult;

    // Decide which resume path applies. tool_calls without tool results can
    // not be resumed mid sequence by continue_final_message, branch instead.
    // tool_calls already paired with tool results need a fresh next turn,
    // not a token level continuation of the target assistant.
    const cIntent = fClassifyContinueIntent(cConversationsStore.vActiveMessages, cIdx);
    if (cIntent.kind === ContinueIntentKind.RERUN_TURN) {
      return this.mRegenerateMessageWithBranching(pMessageId);
    }
    if (cIntent.kind === ContinueIntentKind.NEXT_TURN) {
      return this.mContinueAsNextAgenticTurn(cIntent.truncateAfter);
    }

    try {
      this.mShowErrorDialog(null);
      this.mSetChatLoading(cActiveConv.id, true);
      this.mClearChatStreaming(cActiveConv.id);

      const cAllMessages = await cConversationsStore.mGetConversationMessages(cActiveConv.id);
      const cDbMessage = findMessageById(cAllMessages, pMessageId);

      if (!cDbMessage) {
        this.mSetChatLoading(cActiveConv.id, false);
        return;
      }

      const cOriginalContent = cDbMessage.content;
      const cOriginalReasoning = cDbMessage.reasoningContent || '';
      // Hand the persisted DatabaseMessage straight to sendMessage so its
      // internal converter preserves tool_calls and extras when present.
      // Reconstructing a bare {role, content} here would drop those fields
      // and break continue_final_message for messages with tool calls.
      const cContextWithContinue = cConversationsStore.vActiveMessages.slice(0, cIdx + 1);

      let vAppendedContent = '';
      let vAppendedReasoning = '';
      let vHasReceivedContent = false;

      const fUpdateStreamingContent = (pFullContent: string) => {
        this.mSetChatStreaming(cMsg.convId, pFullContent, cMsg.id);
        // resolve the row by id on every write, switching to another conv mid continue makes
        // this a no op instead of writing positionally into the now displayed conversation
        cConversationsStore.mUpdateMessageAtIndex(cConversationsStore.mFindMessageIndex(cMsg.id), {
          content: pFullContent
        });
      };

      const cAbortController = this.mGetOrCreateAbortController(cMsg.convId);

      await ChatService.mSendMessage(
        cContextWithContinue,
        {
          ...this.mGetApiOptions(),
          continueFinalMessage: true,
          onConnectionState: (pState: StreamConnectionState) => {
            if (cMsg.convId === cConversationsStore.vActiveConversation?.id) {
              this.vStreamConnectionState = pState;
            }
          },
          onChunk: (pChunk: string) => {
            vAppendedContent += pChunk;
            vHasReceivedContent = true;
            fUpdateStreamingContent(cOriginalContent + vAppendedContent);
            this.mSetChatReasoning(cMsg.convId, false);
          },
          onCompletionId: (pId: string) => {
            if (!pId) return;
            // refresh the message id so a later skip targets the live slot after a continue
            cConversationsStore.mUpdateMessageAtIndex(
              cConversationsStore.mFindMessageIndex(cMsg.id),
              {
                completionId: pId
              }
            );
            DatabaseService.mUpdateMessage(cMsg.id, { completionId: pId }).catch(() => {});
          },
          onReasoningChunk: (pChunk: string) => {
            vAppendedReasoning += pChunk;
            vHasReceivedContent = true;
            // mark streaming state so a stop mid-thinking can persist the partial reasoning
            this.mSetChatStreaming(cMsg.convId, cOriginalContent + vAppendedContent, cMsg.id);
            cConversationsStore.mUpdateMessageAtIndex(
              cConversationsStore.mFindMessageIndex(cMsg.id),
              {
                reasoningContent: cOriginalReasoning + vAppendedReasoning
              }
            );
            this.mSetChatReasoning(cMsg.convId, true);
          },
          onQueue: (pStatus: ApiQueueStatus | null) => this.mSetQueueStatus(cMsg.convId, pStatus),
          onTimings: (
            pTimings?: ChatMessageTimings,
            pPromptProgress?: ChatMessagePromptProgress
          ) => {
            const cTokensPerSecond =
              pTimings?.predicted_ms && pTimings?.predicted_n
                ? (pTimings.predicted_n / pTimings.predicted_ms) * 1000
                : 0;
            this.mUpdateProcessingStateFromTimings(
              {
                prompt_n: pTimings?.prompt_n || 0,
                prompt_ms: pTimings?.prompt_ms,
                predicted_n: pTimings?.predicted_n || 0,
                predicted_per_second: cTokensPerSecond,
                cache_n: pTimings?.cache_n || 0,
                prompt_progress: pPromptProgress
              },
              cMsg.convId
            );
          },
          onComplete: async (
            pFinalContent?: string,
            pReasoningContent?: string,
            pTimings?: ChatMessageTimings
          ) => {
            const cFinalAppendedContent = vHasReceivedContent
              ? vAppendedContent
              : pFinalContent || '';
            const cFinalAppendedReasoning = vHasReceivedContent
              ? vAppendedReasoning
              : pReasoningContent || '';
            const cFullContent = cOriginalContent + cFinalAppendedContent;
            const cFullReasoning = cOriginalReasoning + cFinalAppendedReasoning || undefined;

            await DatabaseService.mUpdateMessage(cMsg.id, {
              content: cFullContent,
              reasoningContent: cFullReasoning,
              timestamp: Date.now(),
              timings: pTimings
            });

            cConversationsStore.mUpdateMessageAtIndex(
              cConversationsStore.mFindMessageIndex(cMsg.id),
              {
                content: cFullContent,
                reasoningContent: cFullReasoning,
                timestamp: Date.now(),
                timings: pTimings
              }
            );

            cConversationsStore.mUpdateConversationTimestamp(cMsg.convId);

            this.mSetChatLoading(cMsg.convId, false);
            this.mClearChatStreaming(cMsg.convId);
            this.mSetProcessingState(cMsg.convId, null);
          },
          onError: async (pError: Error) => {
            if (isAbortError(pError)) {
              if (vHasReceivedContent && vAppendedContent) {
                await DatabaseService.mUpdateMessage(cMsg.id, {
                  content: cOriginalContent + vAppendedContent,
                  reasoningContent: cOriginalReasoning + vAppendedReasoning || undefined,
                  timestamp: Date.now()
                });

                cConversationsStore.mUpdateMessageAtIndex(
                  cConversationsStore.mFindMessageIndex(cMsg.id),
                  {
                    content: cOriginalContent + vAppendedContent,
                    reasoningContent: cOriginalReasoning + vAppendedReasoning || undefined,
                    timestamp: Date.now()
                  }
                );
              }

              this.mSetChatLoading(cMsg.convId, false);
              this.mClearChatStreaming(cMsg.convId);
              this.mSetProcessingState(cMsg.convId, null);

              return;
            }

            console.error('Continue generation error:', pError);
            // keep whatever was appended so far, the message stays in memory and in DB
            await DatabaseService.mUpdateMessage(cMsg.id, {
              content: cOriginalContent + vAppendedContent,
              reasoningContent: cOriginalReasoning + vAppendedReasoning || undefined,
              timestamp: Date.now()
            });
            cConversationsStore.mUpdateMessageAtIndex(
              cConversationsStore.mFindMessageIndex(cMsg.id),
              {
                content: cOriginalContent + vAppendedContent,
                reasoningContent: cOriginalReasoning + vAppendedReasoning || undefined,
                timestamp: Date.now()
              }
            );

            this.mSetChatLoading(cMsg.convId, false);
            this.mClearChatStreaming(cMsg.convId);
            this.mSetProcessingState(cMsg.convId, null);
            this.mShowErrorDialog({
              type:
                pError.name === 'TimeoutError' ? ErrorDialogType.TIMEOUT : ErrorDialogType.SERVER,
              message: pError.message
            });
          }
        },

        cMsg.convId,
        cAbortController.signal
      );
    } catch (vError) {
      if (!isAbortError(vError)) console.error('Failed to continue message:', vError);
      if (cActiveConv) this.mSetChatLoading(cActiveConv.id, false);
    }
  }

  async mEditAssistantMessage(
    pMessageId: string,
    pNewContent: string,
    pShouldBranch: boolean
  ): Promise<void> {
    const cActiveConv = cConversationsStore.vActiveConversation;
    if (!cActiveConv || this.mIsChatLoadingInternal(cActiveConv.id)) return;

    const cResult = this.mGetMessageByIdWithRole(pMessageId, MessageRole.ASSISTANT);
    if (!cResult) return;

    const { message: cMsg, index: cIdx } = cResult;

    try {
      if (pShouldBranch) {
        const cNewMessage = await DatabaseService.mCreateMessageBranch(
          {
            convId: cMsg.convId,
            type: cMsg.type,
            timestamp: Date.now(),
            role: cMsg.role,
            content: pNewContent,
            toolCalls: cMsg.toolCalls || '',
            children: [],
            model: cMsg.model
          },
          cMsg.parent!
        );

        await cConversationsStore.mUpdateCurrentNode(cNewMessage.id);
      } else {
        await DatabaseService.mUpdateMessage(cMsg.id, { content: pNewContent });
        cConversationsStore.mUpdateMessageAtIndex(cIdx, { content: pNewContent });
      }

      cConversationsStore.mUpdateConversationTimestamp();

      await cConversationsStore.mRefreshActiveMessages();
    } catch (vError) {
      console.error('Failed to edit assistant message:', vError);
    }
  }

  async mEditUserMessagePreserveResponses(
    pMessageId: string,
    pNewContent: string,
    pNewExtras?: DatabaseMessageExtra[]
  ): Promise<void> {
    const cActiveConv = cConversationsStore.vActiveConversation;
    if (!cActiveConv) return;

    const cResult = this.mGetMessageByIdWithRole(pMessageId, MessageRole.USER);
    if (!cResult) return;

    const { message: cMsg, index: cIdx } = cResult;
    try {
      const dUpdateData: Partial<DatabaseMessage> = { content: pNewContent };

      if (pNewExtras !== undefined) dUpdateData.extra = JSON.parse(JSON.stringify(pNewExtras));

      await DatabaseService.mUpdateMessage(pMessageId, dUpdateData);

      cConversationsStore.mUpdateMessageAtIndex(cIdx, dUpdateData);

      const cAllMessages = await cConversationsStore.mGetConversationMessages(cActiveConv.id);
      const cRootMessage = cAllMessages.find((pM) => pM.type === 'root' && pM.parent === null);

      if (cRootMessage && cMsg.parent === cRootMessage.id && pNewContent.trim()) {
        await cConversationsStore.mUpdateConversationName(
          cActiveConv.id,
          generateConversationTitle(pNewContent, Boolean(fConfig().titleGenerationUseFirstLine))
        );
      }

      cConversationsStore.mUpdateConversationTimestamp();
    } catch (vError) {
      console.error('Failed to edit user message:', vError);
    }
  }

  async mEditMessageWithBranching(
    pMessageId: string,
    pNewContent: string,
    pNewExtras?: DatabaseMessageExtra[]
  ): Promise<void> {
    const cActiveConv = cConversationsStore.vActiveConversation;
    if (!cActiveConv || this.mIsChatLoadingInternal(cActiveConv.id)) return;
    let vResult = this.mGetMessageByIdWithRole(pMessageId, MessageRole.USER);
    if (!vResult) vResult = this.mGetMessageByIdWithRole(pMessageId, MessageRole.SYSTEM);
    if (!vResult) return;
    const { message: cMsg, index: cIdx } = vResult;
    try {
      const cAllMessages = await cConversationsStore.mGetConversationMessages(cActiveConv.id);
      const cRootMessage = cAllMessages.find((pM) => pM.type === 'root' && pM.parent === null);
      const cIsFirstUserMessage =
        cMsg.role === MessageRole.USER && cRootMessage && cMsg.parent === cRootMessage.id;
      const cExtrasToUse =
        pNewExtras !== undefined
          ? JSON.parse(JSON.stringify(pNewExtras))
          : cMsg.extra
            ? JSON.parse(JSON.stringify(cMsg.extra))
            : undefined;

      let vMessageIdForResponse: string;

      const cDbMsg = findMessageById(cAllMessages, cMsg.id);
      const cHasChildren = cDbMsg ? cDbMsg.children.length > 0 : cMsg.children.length > 0;

      if (!cHasChildren) {
        // No responses after this message — update in place instead of branching
        const dUpdates: Partial<DatabaseMessage> = {
          content: pNewContent,
          timestamp: Date.now(),
          extra: cExtrasToUse
        };
        await DatabaseService.mUpdateMessage(cMsg.id, dUpdates);
        cConversationsStore.mUpdateMessageAtIndex(cIdx, dUpdates);
        vMessageIdForResponse = cMsg.id;
      } else {
        // Has children — create a new branch as sibling
        const cParentId = cMsg.parent || cRootMessage?.id;
        if (!cParentId) return;
        const cNewMessage = await DatabaseService.mCreateMessageBranch(
          {
            convId: cMsg.convId,
            type: cMsg.type,
            timestamp: Date.now(),
            role: cMsg.role,
            content: pNewContent,
            toolCalls: cMsg.toolCalls || '',
            children: [],
            extra: cExtrasToUse,
            model: cMsg.model
          },
          cParentId
        );
        await cConversationsStore.mUpdateCurrentNode(cNewMessage.id);
        vMessageIdForResponse = cNewMessage.id;
      }

      cConversationsStore.mUpdateConversationTimestamp();
      if (cIsFirstUserMessage && pNewContent.trim())
        await cConversationsStore.mUpdateConversationName(
          cActiveConv.id,
          generateConversationTitle(pNewContent, Boolean(fConfig().titleGenerationUseFirstLine))
        );
      await cConversationsStore.mRefreshActiveMessages();
      if (cMsg.role === MessageRole.USER)
        await this.mGenerateResponseForMessage(vMessageIdForResponse);
    } catch (vError) {
      console.error('Failed to edit message with branching:', vError);
    }
  }

  private async mGenerateResponseForMessage(pUserMessageId: string): Promise<void> {
    const cActiveConv = cConversationsStore.vActiveConversation;
    if (!cActiveConv) return;

    this.mShowErrorDialog(null);
    this.mSetChatLoading(cActiveConv.id, true);
    this.mClearChatStreaming(cActiveConv.id);

    try {
      const cAllMessages = await cConversationsStore.mGetConversationMessages(cActiveConv.id);
      const cConversationPath = filterByLeafNodeId(
        cAllMessages,
        pUserMessageId,
        false
      ) as DatabaseMessage[];
      const cAssistantMessage = await DatabaseService.mCreateMessageBranch(
        {
          convId: cActiveConv.id,
          type: MessageType.TEXT,
          timestamp: Date.now(),
          role: MessageRole.ASSISTANT,
          content: '',
          toolCalls: '',
          children: [],
          model: null
        },
        pUserMessageId
      );

      cConversationsStore.mAddMessageToActive(cAssistantMessage);

      await this.mStreamChatCompletion(cConversationPath, cAssistantMessage);
    } catch (vError) {
      console.error('Failed to generate response:', vError);
      this.mSetChatLoading(cActiveConv.id, false);
    }
  }

  private mGetContextTotal(): number | null {
    const cActiveConvId = this.vActiveConversationId;
    const cActiveState = cActiveConvId ? this.mGetProcessingState(cActiveConvId) : null;

    if (
      cActiveState &&
      typeof cActiveState.contextTotal === 'number' &&
      cActiveState.contextTotal > 0
    )
      return cActiveState.contextTotal;

    if (fIsRouterMode()) {
      const cModelContextSize = fSelectedModelContextSize();

      if (typeof cModelContextSize === 'number' && cModelContextSize > 0) {
        return cModelContextSize;
      }
    } else {
      const cPropsContextSize = fContextSize();

      if (typeof cPropsContextSize === 'number' && cPropsContextSize > 0) {
        return cPropsContextSize;
      }
    }

    return null;
  }

  mUpdateProcessingStateFromTimings(
    pTimingData: {
      prompt_n: number;
      prompt_ms?: number;
      predicted_n: number;
      predicted_per_second: number;
      cache_n: number;
      prompt_progress?: ChatMessagePromptProgress;
    },
    pConversationId?: string
  ): void {
    const cProcessingState = this.mParseTimingData(pTimingData);

    if (cProcessingState === null) {
      console.warn('Failed to parse timing data - skipping update');
      return;
    }

    const cTargetId = pConversationId || this.vActiveConversationId;
    if (cTargetId) {
      this.mSetProcessingState(cTargetId, cProcessingState);
    }
  }

  private mParseTimingData(pTimingData: Record<string, unknown>): ApiProcessingState | null {
    const cPromptTokens = (pTimingData.prompt_n as number) || 0,
      cPromptMs = (pTimingData.prompt_ms as number) || undefined,
      cPredictedTokens = (pTimingData.predicted_n as number) || 0,
      cTokensPerSecond = (pTimingData.predicted_per_second as number) || 0,
      cCacheTokens = (pTimingData.cache_n as number) || 0;
    const cPromptProgress = pTimingData.prompt_progress as
      | { total: number; cache: number; processed: number; time_ms: number }
      | undefined;
    const cContextTotal = this.mGetContextTotal();
    const cCurrentConfig = fConfig();
    const cOutputTokensMax = cCurrentConfig.max_tokens || -1;
    const cContextUsed = cPromptTokens + cCacheTokens + cPredictedTokens,
      cOutputTokensUsed = cPredictedTokens;
    const cProgressCache = cPromptProgress?.cache || 0,
      cProgressActualDone = (cPromptProgress?.processed ?? 0) - cProgressCache,
      cProgressActualTotal = (cPromptProgress?.total ?? 0) - cProgressCache;
    const cProgressPercent = cPromptProgress
      ? Math.round((cProgressActualDone / cProgressActualTotal) * 100)
      : undefined;
    return {
      status: cPredictedTokens > 0 ? 'generating' : cPromptProgress ? 'preparing' : 'idle',
      tokensDecoded: cPredictedTokens,
      tokensRemaining: cOutputTokensMax - cPredictedTokens,
      contextUsed: cContextUsed,
      contextTotal: cContextTotal,
      outputTokensUsed: cOutputTokensUsed,
      outputTokensMax: cOutputTokensMax,
      hasNextToken: cPredictedTokens > 0,
      tokensPerSecond: cTokensPerSecond,
      temperature: cCurrentConfig.temperature ?? 0.8,
      topP: cCurrentConfig.top_p ?? 0.95,
      speculative: false,
      progressPercent: cProgressPercent,
      promptProgress: cPromptProgress,
      promptTokens: cPromptTokens,
      promptMs: cPromptMs,
      cacheTokens: cCacheTokens
    };
  }

  mRestoreProcessingStateFromMessages(pMessages: DatabaseMessage[], pConversationId: string): void {
    for (let vI = pMessages.length - 1; vI >= 0; vI--) {
      const cMessage = pMessages[vI];
      if (cMessage.role === MessageRole.ASSISTANT && cMessage.timings) {
        const cRestoredState = this.mParseTimingData({
          prompt_n: cMessage.timings.prompt_n || 0,
          prompt_ms: cMessage.timings.prompt_ms,
          predicted_n: cMessage.timings.predicted_n || 0,
          predicted_per_second:
            cMessage.timings.predicted_n && cMessage.timings.predicted_ms
              ? (cMessage.timings.predicted_n / cMessage.timings.predicted_ms) * 1000
              : 0,
          cache_n: cMessage.timings.cache_n || 0
        });
        if (cRestoredState) {
          this.mSetProcessingState(pConversationId, cRestoredState);
          return;
        }
      }
    }
  }

  mGetConversationModel(pMessages: DatabaseMessage[]): string | null {
    for (let vI = pMessages.length - 1; vI >= 0; vI--) {
      const cMessage = pMessages[vI];
      if (cMessage.role === MessageRole.ASSISTANT && cMessage.model) return cMessage.model;
    }
    return null;
  }

  private mGetApiOptions(): Record<string, unknown> {
    const cCurrentConfig = fConfig();
    const fHasValue = (pValue: unknown): boolean =>
      pValue !== undefined && pValue !== null && pValue !== '';
    const dApiOptions: Record<string, unknown> = { stream: true, timings_per_token: true };

    if (fIsRouterMode()) {
      const cModelName = fSelectedModelName();
      if (cModelName) dApiOptions.model = cModelName;
    }

    if (cCurrentConfig.systemMessage) dApiOptions.systemMessage = cCurrentConfig.systemMessage;

    if (cCurrentConfig.disableReasoningParsing) dApiOptions.disableReasoningParsing = true;

    if (cCurrentConfig.excludeReasoningFromContext) dApiOptions.excludeReasoningFromContext = true;

    // an explicit reasoning choice overrides the server default, DEFAULT sends nothing
    const cEffort = cConversationsStore.mGetReasoningEffort();
    if (cEffort !== ReasoningEffort.DEFAULT) {
      dApiOptions.enableThinking = cEffort !== ReasoningEffort.OFF;
      if (cEffort !== ReasoningEffort.OFF) dApiOptions.reasoningEffort = cEffort;
    }

    if (fHasValue(cCurrentConfig.temperature))
      dApiOptions.temperature = Number(cCurrentConfig.temperature);

    if (fHasValue(cCurrentConfig.max_tokens))
      dApiOptions.max_tokens = Number(cCurrentConfig.max_tokens);

    if (fHasValue(cCurrentConfig.dynatemp_range))
      dApiOptions.dynatemp_range = Number(cCurrentConfig.dynatemp_range);

    if (fHasValue(cCurrentConfig.dynatemp_exponent))
      dApiOptions.dynatemp_exponent = Number(cCurrentConfig.dynatemp_exponent);

    if (fHasValue(cCurrentConfig.top_k)) dApiOptions.top_k = Number(cCurrentConfig.top_k);

    if (fHasValue(cCurrentConfig.top_p)) dApiOptions.top_p = Number(cCurrentConfig.top_p);

    if (fHasValue(cCurrentConfig.min_p)) dApiOptions.min_p = Number(cCurrentConfig.min_p);

    if (fHasValue(cCurrentConfig.xtc_probability))
      dApiOptions.xtc_probability = Number(cCurrentConfig.xtc_probability);

    if (fHasValue(cCurrentConfig.xtc_threshold))
      dApiOptions.xtc_threshold = Number(cCurrentConfig.xtc_threshold);

    if (fHasValue(cCurrentConfig.typ_p)) dApiOptions.typ_p = Number(cCurrentConfig.typ_p);

    if (fHasValue(cCurrentConfig.repeat_last_n))
      dApiOptions.repeat_last_n = Number(cCurrentConfig.repeat_last_n);

    if (fHasValue(cCurrentConfig.repeat_penalty))
      dApiOptions.repeat_penalty = Number(cCurrentConfig.repeat_penalty);

    if (fHasValue(cCurrentConfig.presence_penalty))
      dApiOptions.presence_penalty = Number(cCurrentConfig.presence_penalty);

    if (fHasValue(cCurrentConfig.frequency_penalty))
      dApiOptions.frequency_penalty = Number(cCurrentConfig.frequency_penalty);

    if (fHasValue(cCurrentConfig.dry_multiplier))
      dApiOptions.dry_multiplier = Number(cCurrentConfig.dry_multiplier);

    if (fHasValue(cCurrentConfig.dry_base)) dApiOptions.dry_base = Number(cCurrentConfig.dry_base);

    if (fHasValue(cCurrentConfig.dry_allowed_length))
      dApiOptions.dry_allowed_length = Number(cCurrentConfig.dry_allowed_length);

    if (fHasValue(cCurrentConfig.dry_penalty_last_n))
      dApiOptions.dry_penalty_last_n = Number(cCurrentConfig.dry_penalty_last_n);

    if (cCurrentConfig.samplers) dApiOptions.samplers = cCurrentConfig.samplers;

    if (fHasValue(cCurrentConfig.backend_sampling))
      dApiOptions.backend_sampling = cCurrentConfig.backend_sampling;

    if (cCurrentConfig.customJson) dApiOptions.custom = cCurrentConfig.customJson;

    return dApiOptions;
  }

  private mCancelPreEncode(): void {
    if (this.vPreEncodeAbortController) {
      this.vPreEncodeAbortController.abort();
      this.vPreEncodeAbortController = null;
    }
  }

  private async mTriggerPreEncode(
    pAllMessages: DatabaseMessage[],
    pAssistantMessage: DatabaseMessage,
    pAssistantContent: string,
    pModel?: string | null,
    pExcludeReasoning?: boolean
  ): Promise<void> {
    this.mCancelPreEncode();
    this.vPreEncodeAbortController = new AbortController();

    const cSignal = this.vPreEncodeAbortController.signal;

    try {
      const cAllIdle = await ChatService.mAreAllSlotsIdle(pModel, cSignal);
      if (!cAllIdle || cSignal.aborted) return;

      const aMessagesWithAssistant: DatabaseMessage[] = [
        ...pAllMessages,
        { ...pAssistantMessage, content: pAssistantContent }
      ];

      await ChatService.mPreEncode(aMessagesWithAssistant, pModel, pExcludeReasoning, cSignal);
    } catch (vErr) {
      if (!isAbortError(vErr)) {
        console.warn('[ChatStore] Pre-encode failed:', vErr);
      }
    }
  }
}

export const cChatStore = new ChatStore();

export const fActiveProcessingState = () => cChatStore.vActiveProcessingState;
export const fCurrentResponse = () => cChatStore.vCurrentResponse;
export const fErrorDialog = () => cChatStore.vErrorDialogState;
export const fGetAddFilesHandler = () => cChatStore.mGetAddFilesHandler();
export const fGetAllLoadingChats = () => cChatStore.mGetAllLoadingChats();
export const fGetAllStreamingChats = () => cChatStore.mGetAllStreamingChats();
export const fGetChatStreaming = (pConvId: string) => cChatStore.mGetChatStreamingPublic(pConvId);
export const fIsChatLoading = (pConvId: string) => cChatStore.mIsChatLoadingPublic(pConvId);
export const fIsChatStreaming = () => cChatStore.mIsStreaming();
export const fIsEditing = () => cChatStore.mIsEditing();
export const fIsLoading = () => cChatStore.vIsLoading;
export const fIsReasoning = () => cChatStore.vIsReasoning;
export const fPendingEditMessageId = () => cChatStore.vPendingEditMessageId;
export const fChatHasPendingMessage = (pConvId: string) => cChatStore.mHasPendingMessage(pConvId);
export const fChatPendingMessageContent = (pConvId: string) =>
  cChatStore.mPendingMessageContent(pConvId);
export const fChatPendingMessageExtras = (pConvId: string) =>
  cChatStore.mPendingMessageExtras(pConvId);
export const fChatClearPendingMessage = (pConvId: string) =>
  cChatStore.mClearPendingMessage(pConvId);
export const fChatInjectPendingMessage = (
  pConvId: string,
  pContent: string,
  pExtras?: DatabaseMessageExtra[]
) => cChatStore.mInjectPendingMessage(pConvId, pContent, pExtras);
