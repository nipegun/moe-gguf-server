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
  import { deriveAgenticSections, type AgenticSection } from '$lib/utils';
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

  const cSections = $derived(deriveAgenticSections(vMessage, vToolMessages, [], vIsStreaming));

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

{#snippet renderSection(section: AgenticSection, index: number)}
  {#if section.type === AgenticSectionType.TEXT}
    <div class="agentic-text">
      <MarkdownContent content={section.content} attachments={vMessage?.extra} />
    </div>
  {:else if section.type === AgenticSectionType.REASONING || section.type === AgenticSectionType.REASONING_PENDING}
    <ChatMessageReasoningBlock
      {section}
      open={fIsExpanded(index, section)}
      isStreaming={vIsStreaming}
      renderThinkingAsMarkdown={cRenderThinkingAsMarkdown}
      hasReasoningError={cHasReasoningError}
      attachments={vMessage?.extra}
      onToggle={() => fToggleExpanded(index, section)}
    />
  {:else if section.type === AgenticSectionType.TOOL_CALL || section.type === AgenticSectionType.TOOL_CALL_PENDING || section.type === AgenticSectionType.TOOL_CALL_STREAMING}
    <ChatMessageToolCallBlock
      {section}
      open={fIsExpanded(index, section)}
      isStreaming={vIsStreaming}
      isExecuting={section.toolCallId !== undefined &&
        section.toolCallId === cCurrentlyExecutingToolCallId}
      attachments={vMessage?.extra}
      onToggle={() => fToggleExpanded(index, section)}
    />
  {/if}
{/snippet}

<div class="agentic-content gap-2">
  {#if aTurnGroups.length > 1}
    {#each aTurnGroups as turn, turnIndex (turnIndex)}
      {@const turnStats = vMessage?.timings?.agentic?.perTurn?.[turnIndex]}

      <div class="agentic-turn group/turn grid gap-2">
        {#each turn.sections as section, sIdx (turn.flatIndices[sIdx])}
          {@render renderSection(section, turn.flatIndices[sIdx])}
        {/each}

        {#if turnStats && cShowAgenticTurnStats}
          <div class="turn-stats transition-opacity duration-150 mt-1 mb-4">
            <ChatMessageStatistics
              promptTokens={turnStats.llm.prompt_n}
              promptMs={turnStats.llm.prompt_ms}
              predictedTokens={turnStats.llm.predicted_n}
              predictedMs={turnStats.llm.predicted_ms}
              agenticTimings={turnStats.toolCalls.length > 0
                ? fBuildTurnAgenticTimings(turnStats)
                : undefined}
              initialView={ChatMessageStatsView.GENERATION}
              hideSummary
            />
          </div>
        {/if}
      </div>
    {/each}
  {:else}
    {#each cSections as section, index (index)}
      {@render renderSection(section, index)}
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
