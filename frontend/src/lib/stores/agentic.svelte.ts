import { fTranslate } from '../i18n';
/**
 * agenticStore - Reactive State Store for Agentic Loop Orchestration
 *
 * Manages multi-turn agentic loop with MCP tools:
 * - LLM streaming with tool call detection
 * - Tool execution via mcpStore
 * - Session state management
 * - Turn limit enforcement
 *
 * Each agentic turn produces separate DB messages:
 * - One assistant message per LLM turn (with tool_calls if any)
 * - One tool result message per tool call execution
 *
 * **Architecture & Relationships:**
 * - **ChatService**: Stateless API layer (sendMessage, streaming)
 * - **mcpStore**: MCP connection management and tool execution
 * - **agenticStore** (this): Reactive state + business logic
 *
 * @see ChatService in services/chat.service.ts for API operations
 * @see cMcpStore in stores/mcp.svelte.ts for MCP operations
 */

import { ChatService } from '$lib/services';
import { fConfig } from '$lib/stores/settings.svelte';
import { cMcpStore } from '$lib/stores/mcp.svelte';
import { cModelsStore } from '$lib/stores/models.svelte';
import { cToolsStore } from '$lib/stores/tools.svelte';
import { cPermissionsStore } from '$lib/stores/permissions.svelte';
import { BuiltInTool, ToolSource, ToolPermissionDecision } from '$lib/enums';
import { SvelteMap } from 'svelte/reactivity';
import { toast } from 'svelte-sonner';
import { ToolsService } from '$lib/services/tools.service';
import { SandboxService } from '$lib/services/sandbox.service';
import { isAbortError } from '$lib/utils';
import { cDefaultAgenticConfig, cNewline } from '$lib/constants';
import {
  dImageMimeToExtension,
  cDataUriBase64Regex,
  cMcpAttachmentNamePrefix,
  cDefaultImageExtension
} from '$lib/constants';
import {
  AttachmentType,
  ContentPartType,
  MessageRole,
  MimeTypePrefix,
  ToolCallType
} from '$lib/enums';
import type {
  AgenticFlowParams,
  AgenticFlowResult,
  AgenticSession,
  AgenticConfig,
  SettingsConfigType,
  McpServerOverride,
  MCPToolCall,
  ToolEntry
} from '$lib/types';
import type {
  AgenticMessage,
  AgenticToolCallList,
  AgenticFlowCallbacks,
  AgenticFlowOptions,
  SteeringMessage
} from '$lib/types/agentic';
import type {
  ApiChatCompletionToolCall,
  ApiChatMessageData,
  ApiChatMessageContentPart
} from '$lib/types/api';
import type {
  ChatMessagePromptProgress,
  ChatMessageTimings,
  ChatMessageAgenticTimings,
  ChatMessageToolCallTiming,
  ChatMessageAgenticTurnStats
} from '$lib/types/chat';
import type {
  DatabaseMessage,
  DatabaseMessageExtra,
  DatabaseMessageExtraImageFile
} from '$lib/types/database';

function fCreateDefaultSession(): AgenticSession {
  return {
    isRunning: false,
    currentTurn: 0,
    totalToolCalls: 0,
    lastError: null,
    streamingToolCall: null,
    pendingPermissionRequest: null,
    executingToolCallId: null
  };
}

function fToAgenticMessages(pMessages: ApiChatMessageData[]): AgenticMessage[] {
  return pMessages.map((pMessage) => {
    if (
      pMessage.role === MessageRole.ASSISTANT &&
      pMessage.tool_calls &&
      pMessage.tool_calls.length > 0
    ) {
      return {
        role: MessageRole.ASSISTANT,
        content: pMessage.content,
        reasoning_content: pMessage.reasoning_content,
        tool_calls: pMessage.tool_calls.map((pCall, pIndex) => ({
          id: pCall.id ?? `call_${pIndex}`,
          type: (pCall.type as ToolCallType.FUNCTION) ?? ToolCallType.FUNCTION,
          function: {
            name: pCall.function?.name ?? '',
            arguments: pCall.function?.arguments ?? ''
          }
        }))
      } satisfies AgenticMessage;
    }
    if (pMessage.role === MessageRole.ASSISTANT) {
      return {
        role: MessageRole.ASSISTANT,
        content: pMessage.content,
        reasoning_content: pMessage.reasoning_content
      } satisfies AgenticMessage;
    }
    if (pMessage.role === MessageRole.TOOL && pMessage.tool_call_id) {
      return {
        role: MessageRole.TOOL,
        tool_call_id: pMessage.tool_call_id,
        content: typeof pMessage.content === 'string' ? pMessage.content : ''
      } satisfies AgenticMessage;
    }
    return {
      role: pMessage.role as MessageRole.SYSTEM | MessageRole.USER,
      content: pMessage.content
    } satisfies AgenticMessage;
  });
}

class AgenticStore {
  private vSessions = new SvelteMap<string, AgenticSession>();
  // Tool names last reported as withheld because several MCP servers list them
  private vLastToolConflictWarning = '';
  /** Dedicated reactive state for pending permission requests (ensures immediate UI updates) */
  private vPendingPermissions = new SvelteMap<
    string,
    { toolName: string; serverLabel: string } | null
  >();
  /** Non-reactive: stores resolve functions for pending permission Promises */
  private dPermissionResolvers = new Map<string, (pDecision: ToolPermissionDecision) => void>();

  /** Dedicated reactive state for pending continue requests (turn limit reached) */
  private vPendingContinueRequests = new SvelteMap<string, boolean>();
  /** Non-reactive: stores resolve functions for pending continue Promises */
  private dContinueResolvers = new Map<string, (pShouldContinue: boolean) => void>();

  /** Reactive: queued steering messages to inject between turns */
  private vSteeringMessages = new SvelteMap<string, SteeringMessage>();

  get mIsReady(): boolean {
    return true;
  }
  get mIsAnyRunning(): boolean {
    for (const cSession of this.vSessions.values()) {
      if (cSession.isRunning) return true;
    }
    return false;
  }

  mGetSession(pConversationId: string): AgenticSession {
    let vSession = this.vSessions.get(pConversationId);
    if (!vSession) {
      vSession = fCreateDefaultSession();
      this.vSessions.set(pConversationId, vSession);
    }
    return vSession;
  }

  private mUpdateSession(pConversationId: string, pUpdate: Partial<AgenticSession>): void {
    const cSession = this.mGetSession(pConversationId);
    this.vSessions.set(pConversationId, { ...cSession, ...pUpdate });
  }

  mClearSession(pConversationId: string): void {
    this.vSessions.delete(pConversationId);
  }

  mGetActiveSessions(): Array<{ conversationId: string; session: AgenticSession }> {
    const aActive: Array<{ conversationId: string; session: AgenticSession }> = [];
    for (const [cConversationId, cSession] of this.vSessions.entries()) {
      if (cSession.isRunning) aActive.push({ conversationId: cConversationId, session: cSession });
    }
    return aActive;
  }

  mIsRunning(pConversationId: string): boolean {
    return this.vSessions.get(pConversationId)?.isRunning ?? false;
  }

  mCurrentTurn(pConversationId: string): number {
    return this.vSessions.get(pConversationId)?.currentTurn ?? 0;
  }

  mTotalToolCalls(pConversationId: string): number {
    return this.vSessions.get(pConversationId)?.totalToolCalls ?? 0;
  }

  mLastError(pConversationId: string): Error | null {
    return this.vSessions.get(pConversationId)?.lastError ?? null;
  }

  mStreamingToolCall(pConversationId: string): { name: string; arguments: string } | null {
    return this.vSessions.get(pConversationId)?.streamingToolCall ?? null;
  }

  mExecutingToolCallId(pConversationId: string): string | null {
    return this.vSessions.get(pConversationId)?.executingToolCallId ?? null;
  }

  mPendingPermissionRequest(
    pConversationId: string
  ): { toolName: string; serverLabel: string } | null {
    return this.vPendingPermissions.get(pConversationId) ?? null;
  }

  mPendingContinueRequest(pConversationId: string): boolean {
    return this.vPendingContinueRequests.get(pConversationId) ?? false;
  }

  mResolveContinue(pConversationId: string, pShouldContinue: boolean): void {
    const cResolver = this.dContinueResolvers.get(pConversationId);
    if (cResolver) {
      this.dContinueResolvers.delete(pConversationId);
      cResolver(pShouldContinue);
    }
  }

  mResolvePermission(pConversationId: string, pDecision: ToolPermissionDecision): void {
    const cResolver = this.dPermissionResolvers.get(pConversationId);
    if (cResolver) {
      this.dPermissionResolvers.delete(pConversationId);
      cResolver(pDecision);
    }
  }

  mClearError(pConversationId: string): void {
    this.mUpdateSession(pConversationId, { lastError: null });
  }

  mHasPendingSteeringMessage(pConversationId: string): boolean {
    return this.vSteeringMessages.has(pConversationId);
  }

  mPendingSteeringMessageContent(pConversationId: string): string | null {
    return this.vSteeringMessages.get(pConversationId)?.content ?? null;
  }

  mPendingSteeringMessageExtras(pConversationId: string): DatabaseMessageExtra[] | undefined {
    return this.vSteeringMessages.get(pConversationId)?.extras;
  }

  /**
   * Queue a steering message. When the current agentic turn completes,
   * the flow exits and the caller re-sends the message as a normal chat message.
   */
  mInjectSteeringMessage(
    pConversationId: string,
    pContent: string,
    pExtras?: DatabaseMessageExtra[]
  ): void {
    this.vSteeringMessages.set(pConversationId, { content: pContent, extras: pExtras });
  }

  /**
   * Clear the pending steering message without consuming it.
   */
  mClearSteeringMessage(pConversationId: string): void {
    this.vSteeringMessages.delete(pConversationId);
  }

  /**
   * Consume and return the pending steering message for re-sending.
   * Called by chatStore after the agentic flow exits.
   */
  mConsumePendingSteeringMessage(pConversationId: string): SteeringMessage | null {
    const cMsg = this.vSteeringMessages.get(pConversationId);
    if (!cMsg) return null;
    this.vSteeringMessages.delete(pConversationId);
    return cMsg;
  }

  mGetConfig(
    pSettings: SettingsConfigType,
    pPerChatOverrides?: McpServerOverride[]
  ): AgenticConfig {
    const cMaxTurns = Number(pSettings.agenticMaxTurns) || cDefaultAgenticConfig.maxTurns;
    const cHasTools =
      cMcpStore.mHasEnabledServers(pPerChatOverrides) ||
      cToolsStore.mBuiltinTools.length > 0 ||
      cToolsStore.mFrontendTools.length > 0 ||
      cToolsStore.mCustomTools.length > 0;
    return {
      enabled: cHasTools && cDefaultAgenticConfig.enabled,
      maxTurns: cMaxTurns
    };
  }

  private mParseToolArguments(pArgs: string | Record<string, unknown>): Record<string, unknown> {
    if (typeof pArgs === 'object') return pArgs;
    const cTrimmed = pArgs.trim();
    if (cTrimmed === '') return {};
    return JSON.parse(cTrimmed) as Record<string, unknown>;
  }

  /**
   * Asks for permission to run one resolved tool entry; the approval, the
   * "always" grants and the call that follows all refer to that entry.
   */
  private async mRequestPermission(
    pConversationId: string,
    pEntry: ToolEntry,
    pServerLabel: string,
    pSignal?: AbortSignal
  ): Promise<ToolPermissionDecision> {
    const cPermissionKey = pEntry.key;
    if (cPermissionsStore.mHasTool(cPermissionKey)) {
      return ToolPermissionDecision.ONCE;
    }

    this.vPendingPermissions.set(pConversationId, {
      toolName: pEntry.definition.function.name,
      serverLabel: pServerLabel
    });

    return new Promise<ToolPermissionDecision>((pResolve) => {
      if (pSignal?.aborted) {
        this.vPendingPermissions.set(pConversationId, null);
        pResolve(ToolPermissionDecision.DENY);
        return;
      }

      this.dPermissionResolvers.set(pConversationId, (pDecision) => {
        this.vPendingPermissions.set(pConversationId, null);
        if (pDecision === ToolPermissionDecision.ALWAYS) {
          cPermissionsStore.mAllowTool(cPermissionKey);
        } else if (pDecision === ToolPermissionDecision.ALWAYS_SERVER) {
          // MCP tools are granted per server id (labels may repeat), the other sources per source
          const cServerToolKeys = cToolsStore.mAllTools
            .filter((pT) =>
              pEntry.source === ToolSource.MCP
                ? pT.source === ToolSource.MCP && pT.serverId === pEntry.serverId
                : pT.source === pEntry.source
            )
            .map((pT) => pT.key);
          cPermissionsStore.mAllowTools(cServerToolKeys);
        }
        pResolve(pDecision);
      });

      pSignal?.addEventListener(
        'abort',
        () => {
          const cResolver = this.dPermissionResolvers.get(pConversationId);
          if (cResolver) {
            this.dPermissionResolvers.delete(pConversationId);
            this.vPendingPermissions.set(pConversationId, null);
            pResolve(ToolPermissionDecision.DENY);
          }
        },
        { once: true }
      );
    });
  }

  private async mRequestContinue(pConversationId: string, pSignal?: AbortSignal): Promise<boolean> {
    this.vPendingContinueRequests.set(pConversationId, true);

    return new Promise<boolean>((pResolve) => {
      if (pSignal?.aborted) {
        this.vPendingContinueRequests.set(pConversationId, false);
        pResolve(false);
        return;
      }

      this.dContinueResolvers.set(pConversationId, (pShouldContinue) => {
        this.vPendingContinueRequests.set(pConversationId, false);
        pResolve(pShouldContinue);
      });

      pSignal?.addEventListener(
        'abort',
        () => {
          const cResolver = this.dContinueResolvers.get(pConversationId);
          if (cResolver) {
            this.dContinueResolvers.delete(pConversationId);
            this.vPendingContinueRequests.set(pConversationId, false);
            pResolve(false);
          }
        },
        { once: true }
      );
    });
  }

  async mRunAgenticFlow(pParams: AgenticFlowParams): Promise<AgenticFlowResult> {
    const {
      conversationId: cConversationId,
      messages: cMessages,
      options: cOptions = {},
      callbacks: cCallbacks,
      signal: cSignal,
      perChatOverrides: cPerChatOverrides
    } = pParams;

    // Clear any pending permissions/continue requests for this conversation when starting a new flow
    this.vPendingPermissions.set(cConversationId, null);
    this.dPermissionResolvers.delete(cConversationId);
    this.vPendingContinueRequests.set(cConversationId, false);
    this.dContinueResolvers.delete(cConversationId);
    this.vSteeringMessages.delete(cConversationId);

    // Retry disabled tools only when the user opens the tools panel.
    if (
      cToolsStore.mBuiltinTools.length === 0 &&
      !cToolsStore.mLoading &&
      !cToolsStore.mIsToolsEndpointUnreachable
    ) {
      await cToolsStore.mFetchBuiltinTools();
    }

    const cAgenticConfig = this.mGetConfig(fConfig(), cPerChatOverrides);
    if (!cAgenticConfig.enabled) return { handled: false };

    const cHasMcpServers = cMcpStore.mHasEnabledServers(cPerChatOverrides);
    if (cHasMcpServers) {
      const cInitialized = await cMcpStore.mEnsureInitialized(cPerChatOverrides);

      if (!cInitialized) {
        console.log('[AgenticStore] MCP not initialized');
      }
    }

    const cTools = cToolsStore.mGetEnabledToolsForLLM();

    // Warn once per distinct set of withheld names rather than on every message
    const cConflictingTools = cToolsStore.mConflictingMcpToolNames.join(', ');
    if (cConflictingTools && cConflictingTools !== this.vLastToolConflictWarning) {
      toast.warning(fTranslate('text_521171accad0', { p0: cConflictingTools }));
    }
    this.vLastToolConflictWarning = cConflictingTools;

    if (cTools.length === 0) {
      return { handled: false };
    }

    console.log(`[AgenticStore] Starting agentic flow with ${cTools.length} tools`);

    const aNormalizedMessages: ApiChatMessageData[] = (
      await Promise.all(
        cMessages.map((pMsg) => {
          if ('id' in pMsg && 'convId' in pMsg && 'timestamp' in pMsg)
            return ChatService.mConvertDbMessageToApiChatMessageData(
              pMsg as DatabaseMessage & { extra?: DatabaseMessageExtra[] }
            );
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

    this.mUpdateSession(cConversationId, {
      isRunning: true,
      currentTurn: 0,
      totalToolCalls: 0,
      lastError: null
    });

    if (cHasMcpServers) cMcpStore.mAcquireConnection();

    try {
      await this.mExecuteAgenticLoop({
        conversationId: cConversationId,
        messages: aNormalizedMessages,
        options: cOptions,
        tools: cTools,
        agenticConfig: cAgenticConfig,
        callbacks: cCallbacks,
        signal: cSignal
      });
      return { handled: true };
    } catch (vError) {
      const cNormalizedError = vError instanceof Error ? vError : new Error(String(vError));
      this.mUpdateSession(cConversationId, { lastError: cNormalizedError });
      cCallbacks.onError?.(cNormalizedError);
      return { handled: true, error: cNormalizedError };
    } finally {
      this.mUpdateSession(cConversationId, { isRunning: false });

      if (cHasMcpServers) {
        await cMcpStore
          .mReleaseConnection()
          .catch((pErr: unknown) =>
            console.warn('[AgenticStore] Failed to release MCP connection:', pErr)
          );
      }
    }
  }

  private async mExecuteAgenticLoop(pParams: {
    conversationId: string;
    messages: ApiChatMessageData[];
    options: AgenticFlowOptions;
    tools: ReturnType<typeof cToolsStore.mGetEnabledToolsForLLM>;
    agenticConfig: AgenticConfig;
    callbacks: AgenticFlowCallbacks;
    signal?: AbortSignal;
  }): Promise<void> {
    const {
      conversationId: cConversationId,
      messages: cMessages,
      options: cOptions,
      tools: cTools,
      agenticConfig: cAgenticConfig,
      callbacks: cCallbacks,
      signal: cSignal
    } = pParams;
    const {
      onChunk: cOnChunk,
      onReasoningChunk: cOnReasoningChunk,
      onToolCallsStreaming: cOnToolCallsStreaming,
      onAttachments: cOnAttachments,
      onModel: cOnModel,
      onCompletionId: cOnCompletionId,
      onAssistantTurnComplete: cOnAssistantTurnComplete,
      createToolResultMessage: cCreateToolResultMessage,
      updateToolResultMessage: cUpdateToolResultMessage,
      createAssistantMessage: cCreateAssistantMessage,
      onFlowComplete: cOnFlowComplete,
      onTimings: cOnTimings,
      onQueue: cOnQueue,
      onTurnComplete: cOnTurnComplete
    } = cCallbacks;

    const aSessionMessages: AgenticMessage[] = fToAgenticMessages(cMessages);
    let vCapturedTimings: ChatMessageTimings | undefined;
    let vTotalToolCallCount = 0;

    const dAgenticTimings: ChatMessageAgenticTimings = {
      turns: 0,
      toolCallsCount: 0,
      toolsMs: 0,
      toolCalls: [],
      perTurn: [],
      llm: { predicted_n: 0, predicted_ms: 0, prompt_n: 0, prompt_ms: 0 }
    };
    const cMaxTurns = cAgenticConfig.maxTurns;

    const cEffectiveModel = cOptions.model || cModelsStore.vModels[0]?.model || '';

    let vTurn = 0;
    while (true) {
      if (vTurn >= cMaxTurns) {
        // Turn limit reached - ask user whether to continue
        const cShouldContinue = await this.mRequestContinue(cConversationId, cSignal);

        // Yield to allow Svelte to flush the UI update
        await new Promise((pR) => setTimeout(pR, 0));

        if (!cShouldContinue || cSignal?.aborted) {
          cOnFlowComplete?.(this.mBuildFinalTimings(vCapturedTimings, dAgenticTimings));
          return;
        }

        // User chose to continue - extend the limit
        vTurn = 0;
      }

      this.mUpdateSession(cConversationId, { currentTurn: vTurn + 1 });
      dAgenticTimings.turns = vTurn + 1;

      if (cSignal?.aborted) {
        cOnFlowComplete?.(this.mBuildFinalTimings(vCapturedTimings, dAgenticTimings));
        return;
      }

      // For turns > 0, create a new assistant message via callback
      if (vTurn > 0 && cCreateAssistantMessage) {
        await cCreateAssistantMessage();
      }

      let vTurnContent = '';
      let vTurnReasoningContent = '';
      let aTurnToolCalls: ApiChatCompletionToolCall[] = [];
      let vLastStreamingToolCallName = '';
      let vLastStreamingToolCallArgsLength = 0;
      let vTurnTimings: ChatMessageTimings | undefined;

      const dTurnStats: ChatMessageAgenticTurnStats = {
        turn: vTurn + 1,
        llm: { predicted_n: 0, predicted_ms: 0, prompt_n: 0, prompt_ms: 0 },
        toolCalls: [],
        toolsMs: 0
      };

      try {
        await ChatService.mSendMessage(
          aSessionMessages as ApiChatMessageData[],
          {
            ...cOptions,
            stream: true,
            tools: cTools.length > 0 ? cTools : undefined,
            onChunk: (pChunk: string) => {
              vTurnContent += pChunk;
              cOnChunk?.(pChunk);
            },
            onReasoningChunk: (pChunk: string) => {
              vTurnReasoningContent += pChunk;
              cOnReasoningChunk?.(pChunk);
            },
            onToolCallChunk: (pSerialized: string) => {
              try {
                aTurnToolCalls = JSON.parse(pSerialized) as ApiChatCompletionToolCall[];

                cOnToolCallsStreaming?.(aTurnToolCalls);

                if (aTurnToolCalls.length > 0 && aTurnToolCalls[0]?.function) {
                  const cName = aTurnToolCalls[0].function.name || '';
                  const cArgs = aTurnToolCalls[0].function.arguments || '';
                  const cArgsLengthBucket = Math.floor(cArgs.length / 100);
                  if (
                    cName !== vLastStreamingToolCallName ||
                    cArgsLengthBucket !== vLastStreamingToolCallArgsLength
                  ) {
                    vLastStreamingToolCallName = cName;
                    vLastStreamingToolCallArgsLength = cArgsLengthBucket;
                    this.mUpdateSession(cConversationId, {
                      streamingToolCall: { name: cName, arguments: cArgs }
                    });
                  }
                }
              } catch {
                /* Ignore parse errors during streaming */
              }
            },
            onModel: cOnModel,
            onCompletionId: cOnCompletionId,
            onQueue: cOnQueue,
            onTimings: (pTimings?: ChatMessageTimings, pProgress?: ChatMessagePromptProgress) => {
              cOnTimings?.(pTimings, pProgress);
              if (pTimings) {
                vCapturedTimings = pTimings;
                vTurnTimings = pTimings;
              }
            },
            onComplete: () => {
              /* Completion handled after sendMessage resolves */
            },
            onError: (pError: Error) => {
              throw pError;
            }
          },
          cConversationId,
          cSignal
        );

        this.mUpdateSession(cConversationId, { streamingToolCall: null });

        if (vTurnTimings) {
          dAgenticTimings.llm.predicted_n += vTurnTimings.predicted_n || 0;
          dAgenticTimings.llm.predicted_ms += vTurnTimings.predicted_ms || 0;
          dAgenticTimings.llm.prompt_n += vTurnTimings.prompt_n || 0;
          dAgenticTimings.llm.prompt_ms += vTurnTimings.prompt_ms || 0;
          dTurnStats.llm.predicted_n = vTurnTimings.predicted_n || 0;
          dTurnStats.llm.predicted_ms = vTurnTimings.predicted_ms || 0;
          dTurnStats.llm.prompt_n = vTurnTimings.prompt_n || 0;
          dTurnStats.llm.prompt_ms = vTurnTimings.prompt_ms || 0;
        }
      } catch (vError) {
        if (cSignal?.aborted) {
          // Save whatever we have for this turn before exiting
          await cOnAssistantTurnComplete?.(
            vTurnContent,
            vTurnReasoningContent || undefined,
            this.mBuildFinalTimings(vCapturedTimings, dAgenticTimings),
            undefined
          );
          cOnFlowComplete?.(this.mBuildFinalTimings(vCapturedTimings, dAgenticTimings));
          return;
        }
        const cNormalizedError =
          vError instanceof Error ? vError : new Error(fTranslate('text_80c39c3fcd37'));
        // preserve partial output as is, the outer error dialog informs the user separately
        await cOnAssistantTurnComplete?.(
          vTurnContent,
          vTurnReasoningContent || undefined,
          this.mBuildFinalTimings(vCapturedTimings, dAgenticTimings),
          undefined
        );
        cOnFlowComplete?.(this.mBuildFinalTimings(vCapturedTimings, dAgenticTimings));
        throw cNormalizedError;
      }

      // If the abort landed while ChatService.sendMessage was still resolving, the
      // outer catch above never fires because ChatService swallows the AbortError
      // and returns normally. Bail out here so a half-received tool_call (truncated
      // arguments JSON) is not persisted as if it were complete.
      if (cSignal?.aborted) {
        await cOnAssistantTurnComplete?.(
          vTurnContent,
          vTurnReasoningContent || undefined,
          this.mBuildFinalTimings(vCapturedTimings, dAgenticTimings),
          undefined
        );
        cOnFlowComplete?.(this.mBuildFinalTimings(vCapturedTimings, dAgenticTimings));
        return;
      }

      // === Steering check: if a user message was queued during this turn, exit the flow.
      // The caller (chatStore) will consume the pending message and re-send it normally.
      if (this.vSteeringMessages.has(cConversationId)) {
        console.log('[AgenticStore] Steering message detected after turn, exiting agentic flow');
        await cOnAssistantTurnComplete?.(
          vTurnContent,
          vTurnReasoningContent || undefined,
          this.mBuildFinalTimings(vCapturedTimings, dAgenticTimings),
          aTurnToolCalls.length > 0 ? this.mNormalizeToolCalls(aTurnToolCalls) : undefined
        );
        cOnFlowComplete?.(this.mBuildFinalTimings(vCapturedTimings, dAgenticTimings));
        return;
      }

      // No tool calls = final turn, save and complete
      if (aTurnToolCalls.length === 0) {
        dAgenticTimings.perTurn!.push(dTurnStats);

        const cFinalTimings = this.mBuildFinalTimings(vCapturedTimings, dAgenticTimings);

        await cOnAssistantTurnComplete?.(
          vTurnContent,
          vTurnReasoningContent || undefined,
          cFinalTimings,
          undefined
        );

        if (cFinalTimings) cOnTurnComplete?.(cFinalTimings);

        cOnFlowComplete?.(cFinalTimings);

        return;
      }

      // Normalize and save assistant turn with tool calls
      const cNormalizedCalls = this.mNormalizeToolCalls(aTurnToolCalls);
      if (cNormalizedCalls.length === 0) {
        await cOnAssistantTurnComplete?.(
          vTurnContent,
          vTurnReasoningContent || undefined,
          this.mBuildFinalTimings(vCapturedTimings, dAgenticTimings),
          undefined
        );
        cOnFlowComplete?.(this.mBuildFinalTimings(vCapturedTimings, dAgenticTimings));
        return;
      }

      vTotalToolCallCount += cNormalizedCalls.length;
      this.mUpdateSession(cConversationId, { totalToolCalls: vTotalToolCallCount });

      // Save the assistant message with its tool calls
      await cOnAssistantTurnComplete?.(
        vTurnContent,
        vTurnReasoningContent || undefined,
        vTurnTimings,
        cNormalizedCalls
      );

      // Add assistant message to session history
      aSessionMessages.push({
        role: MessageRole.ASSISTANT,
        content: vTurnContent || undefined,
        reasoning_content: vTurnReasoningContent || undefined,
        tool_calls: cNormalizedCalls
      });

      // Execute each tool call and create result messages
      for (let vI = 0; vI < cNormalizedCalls.length; vI++) {
        const cToolCall = cNormalizedCalls[vI];

        if (cSignal?.aborted) {
          cOnFlowComplete?.(this.mBuildFinalTimings(vCapturedTimings, dAgenticTimings));
          return;
        }

        // Check for pending steering message - skip remaining tool calls
        if (this.vSteeringMessages.has(cConversationId)) {
          console.log(
            `[AgenticStore] Steering message detected, skipping ${cNormalizedCalls.length - vI} remaining tool call(s)`
          );
          for (let vJ = vI; vJ < cNormalizedCalls.length; vJ++) {
            const cRemainingCall = cNormalizedCalls[vJ];
            const cInterruptedContent = fTranslate('text_4cda0e9b51ac');
            if (cCreateToolResultMessage) {
              await cCreateToolResultMessage(cRemainingCall.id, cInterruptedContent);
            }
            aSessionMessages.push({
              role: MessageRole.TOOL,
              tool_call_id: cRemainingCall.id,
              content: cInterruptedContent
            });
          }
          break;
        }

        const cToolName = cToolCall.function.name;
        // The single enabled tool this name refers to; unknown, disabled and ambiguous
        // names resolve to nothing and are never offered for approval or run
        const cToolEntry = cToolsStore.mResolveEnabledEntry(cToolName);
        const cServerLabel = cToolEntry ? cToolsStore.mGetEntryServerLabel(cToolEntry) : '';

        // Ask for permission before executing the tool
        const cPermission = cToolEntry
          ? await this.mRequestPermission(cConversationId, cToolEntry, cServerLabel, cSignal)
          : ToolPermissionDecision.ONCE;

        // Yield to allow Svelte to flush the UI update (hide permission dialog)
        await new Promise((pR) => setTimeout(pR, 0));

        if (cSignal?.aborted) {
          cOnFlowComplete?.(this.mBuildFinalTimings(vCapturedTimings, dAgenticTimings));
          return;
        }

        const cToolStartTime = performance.now();
        const cToolSource = cToolEntry?.source ?? null;

        let vResult = '';
        let vToolSuccess = true;
        let vCreatedToolResultMessageId: string | null = null;

        // Streaming tools (currently only exec_shell_command): mark
        // the session so the matching renderer can switch to live mode.
        // Cleared unconditionally below.
        this.mUpdateSession(cConversationId, { executingToolCallId: cToolCall.id });

        if (cPermission === ToolPermissionDecision.DENY) {
          vResult = fTranslate('text_d25c4113c2ef');
          vToolSuccess = false;
        } else {
          try {
            if (
              cToolSource === ToolSource.BUILTIN &&
              cToolName === BuiltInTool.EXEC_SHELL_COMMAND &&
              cCreateToolResultMessage &&
              cUpdateToolResultMessage
            ) {
              const cArgs = this.mParseToolArguments(cToolCall.function.arguments);
              const cMsg = await cCreateToolResultMessage(cToolCall.id, '');
              vCreatedToolResultMessageId = cMsg.id;

              let vAccumulated = '';
              for await (const cEv of ToolsService.mStreamTool(cToolName, cArgs, cSignal)) {
                if (cEv.chunk !== null) {
                  vAccumulated += cEv.chunk;
                  await cUpdateToolResultMessage(cMsg.id, vAccumulated);
                }
                if (cEv.done) {
                  if (cEv.error) {
                    vAccumulated = vAccumulated
                      ? `${vAccumulated}\nError: ${cEv.error}`
                      : fTranslate('text_8904a29c5dcf', { p0: cEv.error });
                    await cUpdateToolResultMessage(cMsg.id, vAccumulated);
                    vToolSuccess = false;
                  }
                  break;
                }
              }
              vResult = vAccumulated;
            } else if (cToolSource === ToolSource.BUILTIN) {
              const cArgs = this.mParseToolArguments(cToolCall.function.arguments);
              const cExecutionResult = await ToolsService.mExecuteTool(cToolName, cArgs, cSignal);

              vResult = cExecutionResult.content;

              if (cExecutionResult.isError) vToolSuccess = false;
            } else if (cToolSource === ToolSource.FRONTEND) {
              const cArgs = this.mParseToolArguments(cToolCall.function.arguments);
              const cExecutionResult = await SandboxService.mExecuteTool(cToolName, cArgs, cSignal);

              vResult = cExecutionResult.content;

              if (cExecutionResult.isError) vToolSuccess = false;
            } else if (cToolSource === ToolSource.MCP && cToolEntry?.serverId) {
              const dMcpCall: MCPToolCall = {
                id: cToolCall.id,
                function: { name: cToolName, arguments: cToolCall.function.arguments }
              };
              const cExecutionResult = await cMcpStore.mExecuteTool(
                dMcpCall,
                cToolEntry.serverId,
                cSignal
              );

              vResult = cExecutionResult.content;
            } else {
              // Unknown, disabled or ambiguous names, and custom JSON definitions, have no executor
              throw new Error(fTranslate('text_7a0a26cba180', { p0: cToolName }));
            }
          } catch (vError) {
            if (isAbortError(vError)) {
              this.mUpdateSession(cConversationId, { executingToolCallId: null });
              cOnFlowComplete?.(this.mBuildFinalTimings(vCapturedTimings, dAgenticTimings));
              return;
            }
            // Carry the partial stream contents already mirrored to the UI -
            // they show up as live output even if the stream broke off mid-run.
            vResult = vResult
              ? `${vResult}\nError: ${vError instanceof Error ? vError.message : String(vError)}`
              : fTranslate('text_8904a29c5dcf', {
                  p0: vError instanceof Error ? vError.message : String(vError)
                });
            vToolSuccess = false;
            if (vCreatedToolResultMessageId && cUpdateToolResultMessage) {
              await cUpdateToolResultMessage(vCreatedToolResultMessageId, vResult);
            }
          }
        }

        this.mUpdateSession(cConversationId, { executingToolCallId: null });

        const cToolDurationMs = performance.now() - cToolStartTime;
        const dToolTiming: ChatMessageToolCallTiming = {
          name: cToolCall.function.name,
          duration_ms: Math.round(cToolDurationMs),
          success: vToolSuccess
        };

        dAgenticTimings.toolCalls!.push(dToolTiming);
        dAgenticTimings.toolCallsCount++;
        dAgenticTimings.toolsMs += Math.round(cToolDurationMs);
        dTurnStats.toolCalls.push(dToolTiming);
        dTurnStats.toolsMs += Math.round(cToolDurationMs);

        if (cSignal?.aborted) {
          cOnFlowComplete?.(this.mBuildFinalTimings(vCapturedTimings, dAgenticTimings));
          return;
        }

        const { cleanedResult: cCleanedResult, attachments: cAttachments } =
          this.mExtractBase64Attachments(vResult);

        // For streaming tools the result message was created empty
        // at the start of execution and updated in place as chunks
        // arrived via updateToolResultMessage. Skip the second
        // create call - just attach any base64 attachments found in
        // the final accumulator (rare, since chunks usually don't
        // carry image data URIs) and emit the attachments callback.
        let vToolResultMessage: DatabaseMessage | undefined;
        if (vCreatedToolResultMessageId) {
          vToolResultMessage = { id: vCreatedToolResultMessageId } as DatabaseMessage;
          if (cAttachments.length > 0 && cUpdateToolResultMessage) {
            await cUpdateToolResultMessage(
              vCreatedToolResultMessageId,
              cCleanedResult,
              cAttachments
            );
          }
        } else if (cCreateToolResultMessage) {
          vToolResultMessage = await cCreateToolResultMessage(
            cToolCall.id,
            cCleanedResult,
            cAttachments.length > 0 ? cAttachments : undefined
          );
        }

        if (cAttachments.length > 0 && vToolResultMessage) {
          cOnAttachments?.(vToolResultMessage.id, cAttachments);
        }

        // Build content parts for session history (including images for vision models)
        const aContentParts: ApiChatMessageContentPart[] = [
          { type: ContentPartType.TEXT, text: cCleanedResult }
        ];
        for (const cAttachment of cAttachments) {
          if (cAttachment.type === AttachmentType.IMAGE) {
            if (cModelsStore.mModelSupportsVision(cEffectiveModel)) {
              aContentParts.push({
                type: ContentPartType.IMAGE_URL,
                image_url: {
                  url: (cAttachment as DatabaseMessageExtraImageFile).base64Url
                }
              });
            } else {
              console.info(
                `[AgenticStore] Skipping image attachment (model "${cEffectiveModel}" does not support vision)`
              );
            }
          }
        }

        aSessionMessages.push({
          role: MessageRole.TOOL,
          tool_call_id: cToolCall.id,
          content: aContentParts.length === 1 ? cCleanedResult : aContentParts
        });
      }

      if (dTurnStats.toolCalls.length > 0) {
        dAgenticTimings.perTurn!.push(dTurnStats);

        const cIntermediateTimings = this.mBuildFinalTimings(vCapturedTimings, dAgenticTimings);
        if (cIntermediateTimings) cOnTurnComplete?.(cIntermediateTimings);
      }

      // If tools were interrupted by a steering message, exit now instead of starting another LLM turn
      if (this.vSteeringMessages.has(cConversationId)) {
        console.log(
          '[AgenticStore] Steering message detected after tool execution, exiting agentic flow'
        );
        cOnFlowComplete?.(this.mBuildFinalTimings(vCapturedTimings, dAgenticTimings));
        return;
      }

      vTurn++;
    }
  }

  private mBuildFinalTimings(
    pCapturedTimings: ChatMessageTimings | undefined,
    pAgenticTimings: ChatMessageAgenticTimings
  ): ChatMessageTimings | undefined {
    if (pAgenticTimings.toolCallsCount === 0) return pCapturedTimings;
    return {
      predicted_n: pCapturedTimings?.predicted_n,
      predicted_ms: pCapturedTimings?.predicted_ms,
      prompt_n: pCapturedTimings?.prompt_n,
      prompt_ms: pCapturedTimings?.prompt_ms,
      cache_n: pCapturedTimings?.cache_n,
      agentic: pAgenticTimings
    };
  }

  private mNormalizeToolCalls(pToolCalls: ApiChatCompletionToolCall[]): AgenticToolCallList {
    if (!pToolCalls) return [];
    return pToolCalls.map((pCall, pIndex) => ({
      id: pCall?.id ?? `tool_${pIndex}`,
      type: (pCall?.type as ToolCallType.FUNCTION) ?? ToolCallType.FUNCTION,
      function: {
        name: pCall?.function?.name ?? '',
        arguments: pCall?.function?.arguments ?? ''
      }
    }));
  }

  private mExtractBase64Attachments(pResult: string): {
    cleanedResult: string;
    attachments: DatabaseMessageExtra[];
  } {
    if (!pResult.trim()) {
      return { cleanedResult: pResult, attachments: [] };
    }

    const cLines = pResult.split(cNewline);
    const aAttachments: DatabaseMessageExtra[] = [];
    let vAttachmentIndex = 0;

    const cCleanedLines = cLines.map((pLine) => {
      const cTrimmedLine = pLine.trim();

      const cMatch = cTrimmedLine.match(cDataUriBase64Regex);
      if (!cMatch) {
        return pLine;
      }

      const cMimeType = cMatch[1].toLowerCase();
      const cBase64Data = cMatch[2];

      if (!cBase64Data) {
        return pLine;
      }

      vAttachmentIndex += 1;
      const cName = this.mBuildAttachmentName(cMimeType, vAttachmentIndex);

      if (cMimeType.startsWith(MimeTypePrefix.IMAGE)) {
        aAttachments.push({ type: AttachmentType.IMAGE, name: cName, base64Url: cTrimmedLine });

        return `[Attachment saved: ${cName}]`;
      }

      return pLine;
    });

    return { cleanedResult: cCleanedLines.join(cNewline), attachments: aAttachments };
  }

  private mBuildAttachmentName(pMimeType: string, pIndex: number): string {
    const cExtension = dImageMimeToExtension[pMimeType] ?? cDefaultImageExtension;

    return `${cMcpAttachmentNamePrefix}-${Date.now()}-${pIndex}.${cExtension}`;
  }
}

export const cAgenticStore = new AgenticStore();

export function fAgenticIsRunning(pConversationId: string) {
  return cAgenticStore.mIsRunning(pConversationId);
}

export function fAgenticCurrentTurn(pConversationId: string) {
  return cAgenticStore.mCurrentTurn(pConversationId);
}

export function fAgenticTotalToolCalls(pConversationId: string) {
  return cAgenticStore.mTotalToolCalls(pConversationId);
}

export function fAgenticLastError(pConversationId: string) {
  return cAgenticStore.mLastError(pConversationId);
}

export function fAgenticStreamingToolCall(pConversationId: string) {
  return cAgenticStore.mStreamingToolCall(pConversationId);
}

export function fAgenticPendingPermissionRequest(pConversationId: string) {
  return cAgenticStore.mPendingPermissionRequest(pConversationId);
}

export function fAgenticResolvePermission(
  pConversationId: string,
  pDecision: ToolPermissionDecision
) {
  cAgenticStore.mResolvePermission(pConversationId, pDecision);
}

export function fAgenticPendingContinueRequest(pConversationId: string) {
  return cAgenticStore.mPendingContinueRequest(pConversationId);
}

export function fAgenticResolveContinue(pConversationId: string, pShouldContinue: boolean) {
  cAgenticStore.mResolveContinue(pConversationId, pShouldContinue);
}

export function fAgenticHasPendingSteeringMessage(pConversationId: string) {
  return cAgenticStore.mHasPendingSteeringMessage(pConversationId);
}

export function fAgenticInjectSteeringMessage(
  pConversationId: string,
  pContent: string,
  pExtras?: DatabaseMessageExtra[]
) {
  cAgenticStore.mInjectSteeringMessage(pConversationId, pContent, pExtras);
}

export function fAgenticPendingSteeringMessageContent(pConversationId: string) {
  return cAgenticStore.mPendingSteeringMessageContent(pConversationId);
}

export function fAgenticPendingSteeringMessageExtras(pConversationId: string) {
  return cAgenticStore.mPendingSteeringMessageExtras(pConversationId);
}

export function fAgenticClearSteeringMessage(pConversationId: string) {
  cAgenticStore.mClearSteeringMessage(pConversationId);
}

export function fAgenticIsAnyRunning() {
  return cAgenticStore.mIsAnyRunning;
}

export function fAgenticExecutingToolCallId(pConversationId: string) {
  return cAgenticStore.mExecutingToolCallId(pConversationId);
}
