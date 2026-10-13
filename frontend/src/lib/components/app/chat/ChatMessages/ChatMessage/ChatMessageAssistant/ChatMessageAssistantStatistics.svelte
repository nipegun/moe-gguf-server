<script lang="ts">
  import { ChatMessageStatistics } from '$lib/components/app';
  import { ChatMessageStatisticsMode } from '$lib/enums';
  import type { UseProcessingStateReturn } from '$lib/hooks/use-processing-state.svelte';

  interface Props {
    message: DatabaseMessage;
    isLoading: boolean;
    processingState: UseProcessingStateReturn;
    showMessageStats: boolean;
  }

  let {
    message: vMessage,
    isLoading: vIsLoading,
    processingState: vProcessingState,
    showMessageStats: vShowMessageStats
  }: Props = $props();
</script>

{#if vShowMessageStats && vMessage.timings && vMessage.timings.predicted_n && vMessage.timings.predicted_ms}
  {@const agentic = vMessage.timings.agentic}
  <ChatMessageStatistics
    mode={ChatMessageStatisticsMode.GENERATION}
    promptTokens={agentic ? agentic.llm.prompt_n : vMessage.timings.prompt_n}
    promptMs={agentic ? agentic.llm.prompt_ms : vMessage.timings.prompt_ms}
    predictedTokens={agentic ? agentic.llm.predicted_n : vMessage.timings.predicted_n}
    predictedMs={agentic ? agentic.llm.predicted_ms : vMessage.timings.predicted_ms}
    agenticTimings={agentic}
  />
{:else if vIsLoading && vShowMessageStats}
  {@const liveStats = vProcessingState.getLiveProcessingStats()}
  {@const genStats = vProcessingState.getLiveGenerationStats()}

  {#if genStats}
    <ChatMessageStatistics
      mode={ChatMessageStatisticsMode.GENERATION}
      isLive
      promptTokens={liveStats?.tokensProcessed}
      promptMs={liveStats?.timeMs}
      predictedTokens={genStats.tokensGenerated}
      predictedMs={genStats.timeMs}
    />
  {/if}
{/if}
