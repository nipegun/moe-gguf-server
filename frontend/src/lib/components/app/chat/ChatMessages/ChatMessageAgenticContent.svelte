<script lang="ts">
  import {
    ChatMessageStatistics,
    MarkdownContent,
    ChatMessageActionCardPermissionRequest,
    ChatMessageActionCardContinueRequest
  } from '$lib/components/app';

  import { AgenticSectionType, ChatMessageStatsView, ToolPermissionDecision } from '$lib/enums';
  import type {
    ChatMessageAgenticTimings,
    ChatMessageAgenticTurnStats,
    DatabaseMessage
  } from '$lib/types';
  import { fDeriveAgenticSections, type AgenticSection } from '$lib/utils';
  import {
    fAgenticPendingPermissionRequest,
    fAgenticResolvePermission,
    fAgenticPendingContinueRequest,
    fAgenticResolveContinue,
    fAgenticLastError,
    fAgenticExecutingToolCallId
  } from '$lib/stores/agentic.svelte';
  import { fConfig } from '$lib/stores/settings.svelte';
  import ChatMessageReasoningBlock from './ChatMessageReasoningBlock.svelte';
  import ChatMessageToolCallBlock from './ChatMessage/ChatMessageToolCall/ChatMessageToolCallBlock.svelte';
  import { fIsLocalShellTool } from './ChatMessage/ChatMessageToolCall/parsers/local-shell-run';

  interface Props {
    message: DatabaseMessage;
    toolMessages?: DatabaseMessage[];
    isStreaming?: boolean;
    isLastAssistantMessage?: boolean;
  }

  let {
    message: vMessage,
    toolMessages: vToolMessages = [],
    isStreaming: vIsStreaming = false,
    isLastAssistantMessage: vIsLastAssistantMessage = false
  }: Props = $props();

  let dExpandedStates: Record<number, boolean> = $state({});

  const cRenderThinkingAsMarkdown = $derived(fConfig().renderThinkingAsMarkdown as boolean);
  const cShowThoughtInProgress = $derived(Boolean(fConfig().showThoughtInProgress));
  const cAlwaysShowToolCallContent = $derived(Boolean(fConfig().alwaysShowToolCallContent));
  const vExpandLocalShellCommands = $derived(Boolean(fConfig().expandLocalShellCommands));
  const cShowMessageStats = $derived(Boolean(fConfig().showMessageStats));
  const cShowAgenticTurnStats = $derived(
    cShowMessageStats && Boolean(fConfig().showAgenticTurnStats)
  );

  const cHasReasoningError = $derived(
    vIsLastAssistantMessage ? !!fAgenticLastError(vMessage.convId) : false
  );

  let vPermissionDismissed = $state(false);

  const cPendingPermission = $derived(
    vIsStreaming && vIsLastAssistantMessage
      ? fAgenticPendingPermissionRequest(vMessage.convId)
      : null
  );

  let vPrevPendingRef: typeof cPendingPermission = null;
  $effect(() => {
    if (cPendingPermission !== vPrevPendingRef) {
      vPrevPendingRef = cPendingPermission;
      if (cPendingPermission) {
        vPermissionDismissed = false;
      }
    }
  });

  function fHandlePermission(pDecision: ToolPermissionDecision) {
    vPermissionDismissed = true;
    fAgenticResolvePermission(vMessage.convId, pDecision);
  }

  let vContinueDismissed = $state(false);

  const cPendingContinue = $derived(
    vIsStreaming && vIsLastAssistantMessage
      ? fAgenticPendingContinueRequest(vMessage.convId)
      : false
  );

  let vPrevContinueRef = false;
  $effect(() => {
    if (cPendingContinue !== vPrevContinueRef) {
      vPrevContinueRef = cPendingContinue;
      if (cPendingContinue) {
        vContinueDismissed = false;
      }
    }
  });

  function fHandleContinue(pShouldContinue: boolean) {
    vContinueDismissed = true;
    fAgenticResolveContinue(vMessage.convId, pShouldContinue);
  }

  const cSections = $derived(fDeriveAgenticSections(vMessage, vToolMessages, [], vIsStreaming));

  const cCurrentlyExecutingToolCallId = $derived(
    vIsStreaming ? fAgenticExecutingToolCallId(vMessage.convId) : null
  );

  type TurnGroup = {
    sections: AgenticSection[];
    flatIndices: number[];
  };

  const aTurnGroups: TurnGroup[] = $derived.by(() => {
    const aGroups: TurnGroup[] = [];
    let aCurrentTurn: AgenticSection[] = [];
    let aCurrentIndices: number[] = [];
    let vPrevWasTool = false;

    for (let vI = 0; vI < cSections.length; vI++) {
      const cSection = cSections[vI];
      const cIsTool =
        cSection.type === AgenticSectionType.TOOL_CALL ||
        cSection.type === AgenticSectionType.TOOL_CALL_PENDING ||
        cSection.type === AgenticSectionType.TOOL_CALL_STREAMING;

      if (!cIsTool && vPrevWasTool && aCurrentTurn.length > 0) {
        aGroups.push({ sections: aCurrentTurn, flatIndices: aCurrentIndices });
        aCurrentTurn = [];
        aCurrentIndices = [];
      }

      aCurrentTurn.push(cSection);
      aCurrentIndices.push(vI);
      vPrevWasTool = cIsTool;
    }

    if (aCurrentTurn.length > 0) {
      aGroups.push({ sections: aCurrentTurn, flatIndices: aCurrentIndices });
    }

    return aGroups;
  });

  function fGetDefaultExpanded(pSection: AgenticSection): boolean {
    if (
      pSection.type === AgenticSectionType.TOOL_CALL ||
      pSection.type === AgenticSectionType.TOOL_CALL_PENDING ||
      pSection.type === AgenticSectionType.TOOL_CALL_STREAMING
    ) {
      return (
        cAlwaysShowToolCallContent ||
        (vExpandLocalShellCommands && fIsLocalShellTool(pSection.toolName))
      );
    }

    if (pSection.type === AgenticSectionType.REASONING_PENDING) {
      return cShowThoughtInProgress;
    }

    return false;
  }

  function fIsExpanded(pIndex: number, pSection: AgenticSection): boolean {
    if (dExpandedStates[pIndex] !== undefined) {
      return dExpandedStates[pIndex];
    }

    return fGetDefaultExpanded(pSection);
  }

  function fToggleExpanded(pIndex: number, pSection: AgenticSection) {
    const cCurrentState = fIsExpanded(pIndex, pSection);

    dExpandedStates[pIndex] = !cCurrentState;
  }

  function fBuildTurnAgenticTimings(
    pStats: ChatMessageAgenticTurnStats
  ): ChatMessageAgenticTimings {
    return {
      turns: 1,
      toolCallsCount: pStats.toolCalls.length,
      toolsMs: pStats.toolsMs,
      toolCalls: pStats.toolCalls,
      llm: pStats.llm
    };
  }
</script>

{#snippet renderSection(pSection: AgenticSection, pIndex: number)}
  {#if pSection.type === AgenticSectionType.TEXT}
    <div class="agentic-text">
      <MarkdownContent content={pSection.content} attachments={vMessage?.extra} />
    </div>
  {:else if pSection.type === AgenticSectionType.REASONING || pSection.type === AgenticSectionType.REASONING_PENDING}
    <ChatMessageReasoningBlock
      section={pSection}
      open={fIsExpanded(pIndex, pSection)}
      isStreaming={vIsStreaming}
      renderThinkingAsMarkdown={cRenderThinkingAsMarkdown}
      hasReasoningError={cHasReasoningError}
      attachments={vMessage?.extra}
      onToggle={() => fToggleExpanded(pIndex, pSection)}
    />
  {:else if pSection.type === AgenticSectionType.TOOL_CALL || pSection.type === AgenticSectionType.TOOL_CALL_PENDING || pSection.type === AgenticSectionType.TOOL_CALL_STREAMING}
    <ChatMessageToolCallBlock
      section={pSection}
      open={fIsExpanded(pIndex, pSection)}
      isStreaming={vIsStreaming}
      isExecuting={pSection.toolCallId !== undefined &&
        pSection.toolCallId === cCurrentlyExecutingToolCallId}
      attachments={vMessage?.extra}
      onToggle={() => fToggleExpanded(pIndex, pSection)}
    />
  {/if}
{/snippet}

<div class="agentic-content gap-2">
  {#if aTurnGroups.length > 1}
    {#each aTurnGroups as vTurn, vTurnIndex (vTurnIndex)}
      {@const cTurnStats = vMessage?.timings?.agentic?.perTurn?.[vTurnIndex]}

      <div class="agentic-turn group/turn grid gap-2">
        {#each vTurn.sections as vSection, vSectionIndex (vTurn.flatIndices[vSectionIndex])}
          {@render renderSection(vSection, vTurn.flatIndices[vSectionIndex])}
        {/each}

        {#if cTurnStats && cShowAgenticTurnStats}
          <div class="turn-stats transition-opacity duration-150 mt-1 mb-4">
            <ChatMessageStatistics
              promptTokens={cTurnStats.llm.prompt_n}
              promptMs={cTurnStats.llm.prompt_ms}
              predictedTokens={cTurnStats.llm.predicted_n}
              predictedMs={cTurnStats.llm.predicted_ms}
              agenticTimings={cTurnStats.toolCalls.length > 0
                ? fBuildTurnAgenticTimings(cTurnStats)
                : undefined}
              initialView={ChatMessageStatsView.GENERATION}
              hideSummary
            />
          </div>
        {/if}
      </div>
    {/each}
  {:else}
    {#each cSections as vSection, vIndex (vIndex)}
      {@render renderSection(vSection, vIndex)}
    {/each}
  {/if}

  {#if cPendingPermission && !vPermissionDismissed}
    <ChatMessageActionCardPermissionRequest
      toolName={cPendingPermission.toolName}
      serverLabel={cPendingPermission.serverLabel}
      onDecision={fHandlePermission}
    />
  {/if}

  {#if cPendingContinue && !vContinueDismissed}
    <ChatMessageActionCardContinueRequest onDecision={fHandleContinue} />
  {/if}
</div>

<style>
  .agentic-content {
    display: flex;
    flex-direction: column;
    width: 100%;
    max-width: 48rem;
  }

  .agentic-content > :global(*),
  .agentic-turn > :global(*) {
    min-width: 0;
  }

  .agentic-text {
    width: 100%;
  }

  .turn-stats {
    border-top: 1px solid hsl(var(--muted) / 0.5);
  }
</style>
