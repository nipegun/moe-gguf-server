<script lang="ts">
  import { fTranslate } from '../../../../../i18n';

  import { Clock, Gauge, WholeWord, BookOpenText, Sparkles, Wrench, Layers } from '@lucide/svelte';
  import { ChatMessageStatisticsBadge } from '$lib/components/app';
  import * as Tooltip from '$lib/components/ui/tooltip';
  import { ChatMessageStatsView, ChatMessageStatisticsMode } from '$lib/enums';
  import type { ChatMessageAgenticTimings } from '$lib/types/chat';
  import { formatPerformanceTime } from '$lib/utils';
  import { cMsPerSecond, cDefaultPerformanceTime } from '$lib/constants';
  import type { Component } from 'svelte';

  interface Props {
    predictedTokens?: number;
    predictedMs?: number;
    promptTokens?: number;
    promptMs?: number;
    isLive?: boolean;
    isProcessingPrompt?: boolean;
    initialView?: ChatMessageStatsView;
    agenticTimings?: ChatMessageAgenticTimings;
    onActiveViewChange?: (pView: ChatMessageStatsView) => void;
    hideSummary?: boolean;
    mode?: ChatMessageStatisticsMode;
  }

  let {
    predictedTokens: vPredictedTokens,
    predictedMs: vPredictedMs,
    promptTokens: vPromptTokens,
    promptMs: vPromptMs,
    isLive: vIsLive = false,
    isProcessingPrompt: vIsProcessingPrompt = false,
    initialView: vInitialView = ChatMessageStatsView.GENERATION,
    agenticTimings: vAgenticTimings,
    onActiveViewChange: vOnActiveViewChange,
    hideSummary: vHideSummary = false,
    mode: vMode = ChatMessageStatisticsMode.SWITCHABLE
  }: Props = $props();

  let vIsSwitchable = $derived(vMode === ChatMessageStatisticsMode.SWITCHABLE);

  let vActiveView: ChatMessageStatsView = $derived(
    vMode === ChatMessageStatisticsMode.READING
      ? ChatMessageStatsView.READING
      : vMode === ChatMessageStatisticsMode.GENERATION
        ? ChatMessageStatsView.GENERATION
        : vInitialView
  );
  let vHasAutoSwitchedToGeneration = $state(false);

  $effect(() => {
    if (vIsSwitchable) {
      vOnActiveViewChange?.(vActiveView);
    }
  });

  // In live mode: auto-switch to GENERATION tab when prompt processing completes
  $effect(() => {
    if (vIsLive && vIsSwitchable) {
      // Auto-switch to generation tab only when prompt processing is done (once)
      if (
        !vHasAutoSwitchedToGeneration &&
        !vIsProcessingPrompt &&
        vPredictedTokens &&
        vPredictedTokens > 0
      ) {
        vActiveView = ChatMessageStatsView.GENERATION;
        vHasAutoSwitchedToGeneration = true;
      } else if (!vHasAutoSwitchedToGeneration) {
        // Stay on READING while prompt is still being processed
        vActiveView = ChatMessageStatsView.READING;
      }
    }
  });

  let vHasGenerationStats = $derived(
    vPredictedTokens !== undefined &&
      vPredictedTokens > 0 &&
      vPredictedMs !== undefined &&
      vPredictedMs > 0
  );

  let vTokensPerSecond = $derived(
    vHasGenerationStats ? (vPredictedTokens! / vPredictedMs!) * cMsPerSecond : 0
  );
  let vFormattedTime = $derived(
    vPredictedMs !== undefined ? formatPerformanceTime(vPredictedMs) : cDefaultPerformanceTime
  );

  let vPromptTokensPerSecond = $derived(
    vPromptTokens !== undefined && vPromptMs !== undefined && vPromptMs > 0
      ? (vPromptTokens / vPromptMs) * cMsPerSecond
      : undefined
  );

  let vFormattedPromptTime = $derived(
    vPromptMs !== undefined ? formatPerformanceTime(vPromptMs) : undefined
  );

  let vHasPromptStats = $derived(
    vPromptTokens !== undefined &&
      vPromptMs !== undefined &&
      vPromptTokensPerSecond !== undefined &&
      vFormattedPromptTime !== undefined
  );

  let vIsGenerationDisabled = $derived(vIsLive && vIsSwitchable && !vHasGenerationStats);

  let vHasAgenticStats = $derived(
    vAgenticTimings !== undefined && vAgenticTimings.toolCallsCount > 0
  );

  let vAgenticToolsPerSecond = $derived(
    vHasAgenticStats && vAgenticTimings!.toolsMs > 0
      ? (vAgenticTimings!.toolCallsCount / vAgenticTimings!.toolsMs) * cMsPerSecond
      : 0
  );

  let vFormattedAgenticToolsTime = $derived(
    vHasAgenticStats ? formatPerformanceTime(vAgenticTimings!.toolsMs) : cDefaultPerformanceTime
  );

  let vAgenticTotalTimeMs = $derived(
    vHasAgenticStats
      ? vAgenticTimings!.toolsMs +
          vAgenticTimings!.llm.predicted_ms +
          vAgenticTimings!.llm.prompt_ms
      : 0
  );

  let vFormattedAgenticTotalTime = $derived(formatPerformanceTime(vAgenticTotalTimeMs));
</script>

{#snippet viewButton(opts: {
  view: ChatMessageStatsView;
  icon: Component;
  label: string;
  tooltipText: string;
  disabled?: boolean;
})}
  {@const IconComponent = opts.icon}
  <Tooltip.Root>
    <Tooltip.Trigger>
      <!-- prevent another nested button element -->
      {#snippet child({ props })}
        <button
          {...props}
          type="button"
          class="inline-flex h-5 w-5 items-center justify-center rounded-sm transition-colors {vActiveView ===
          opts.view
            ? 'bg-background text-foreground shadow-sm'
            : opts.disabled
              ? 'cursor-not-allowed opacity-40'
              : 'hover:text-foreground'}"
          onclick={() => !opts.disabled && (vActiveView = opts.view)}
          disabled={opts.disabled}
        >
          <IconComponent class="h-3 w-3" />

          <span class="sr-only">{opts.label}</span>
        </button>
      {/snippet}
    </Tooltip.Trigger>

    <Tooltip.Content>
      <p>{opts.tooltipText}</p>
    </Tooltip.Content>
  </Tooltip.Root>
{/snippet}

<div class="inline-flex items-center text-xs text-muted-foreground">
  {#if vIsSwitchable}
    <div class="inline-flex items-center rounded-sm bg-muted-foreground/15 p-0.5">
      {#if vHasPromptStats || vIsLive}
        {@render viewButton({
          view: ChatMessageStatsView.READING,
          icon: BookOpenText,
          label: fTranslate('text_463816d07097'),
          tooltipText: 'Processing'
        })}
      {/if}

      {@render viewButton({
        view: ChatMessageStatsView.GENERATION,
        icon: Sparkles,
        label: fTranslate('text_40536dd8a0e2'),
        tooltipText: vIsGenerationDisabled
          ? fTranslate('text_17a548800288')
          : fTranslate('text_40536dd8a0e2'),
        disabled: vIsGenerationDisabled
      })}

      {#if vHasAgenticStats}
        {@render viewButton({
          view: ChatMessageStatsView.TOOLS,
          icon: Wrench,
          label: fTranslate('text_ea93d6a262ec'),
          tooltipText: fTranslate('text_da5122dc0f97')
        })}

        {#if !vHideSummary}
          {@render viewButton({
            view: ChatMessageStatsView.SUMMARY,
            icon: Layers,
            label: fTranslate('text_8e76a94ac832'),
            tooltipText: fTranslate('text_ce93a62f83db')
          })}
        {/if}
      {/if}
    </div>
  {/if}

  <div class="flex items-center gap-1 px-2">
    {#if vActiveView === ChatMessageStatsView.GENERATION && vHasGenerationStats}
      <ChatMessageStatisticsBadge
        class="bg-transparent"
        icon={WholeWord}
        value="{vPredictedTokens?.toLocaleString()} tokens"
        tooltipLabel="Generated tokens"
      />

      <ChatMessageStatisticsBadge
        class="bg-transparent"
        icon={Clock}
        value={vFormattedTime}
        tooltipLabel="Generation time"
      />

      <ChatMessageStatisticsBadge
        class="bg-transparent"
        icon={Gauge}
        value="{vTokensPerSecond.toFixed(2)} t/s"
        tooltipLabel="Generation speed"
      />
    {:else if vActiveView === ChatMessageStatsView.TOOLS && vHasAgenticStats}
      <ChatMessageStatisticsBadge
        class="bg-transparent"
        icon={Wrench}
        value="{vAgenticTimings!.toolCallsCount} calls"
        tooltipLabel="Tool calls executed"
      />

      <ChatMessageStatisticsBadge
        class="bg-transparent"
        icon={Clock}
        value={vFormattedAgenticToolsTime}
        tooltipLabel="Tool execution time"
      />

      <ChatMessageStatisticsBadge
        class="bg-transparent"
        icon={Gauge}
        value="{vAgenticToolsPerSecond.toFixed(2)} calls/s"
        tooltipLabel="Tool execution rate"
      />
    {:else if vActiveView === ChatMessageStatsView.SUMMARY && vHasAgenticStats}
      <ChatMessageStatisticsBadge
        class="bg-transparent"
        icon={Layers}
        value="{vAgenticTimings!.turns} turns"
        tooltipLabel="Agentic turns (LLM calls)"
      />

      <ChatMessageStatisticsBadge
        class="bg-transparent"
        icon={WholeWord}
        value="{vAgenticTimings!.llm.predicted_n.toLocaleString()} tokens"
        tooltipLabel="Total tokens generated"
      />

      <ChatMessageStatisticsBadge
        class="bg-transparent"
        icon={Clock}
        value={vFormattedAgenticTotalTime}
        tooltipLabel="Total time (LLM + tools)"
      />
    {:else if vHasPromptStats && (vMode === ChatMessageStatisticsMode.READING || vIsSwitchable)}
      <ChatMessageStatisticsBadge
        class="bg-transparent"
        icon={WholeWord}
        value="{vPromptTokens} tokens"
        tooltipLabel="Prompt tokens"
      />

      <ChatMessageStatisticsBadge
        class="bg-transparent"
        icon={Clock}
        value={vFormattedPromptTime ?? '0s'}
        tooltipLabel="Prompt processing time"
      />

      <ChatMessageStatisticsBadge
        class="bg-transparent"
        icon={Gauge}
        value="{vPromptTokensPerSecond!.toFixed(2)} tokens/s"
        tooltipLabel="Prompt processing speed"
      />
    {/if}
  </div>
</div>
