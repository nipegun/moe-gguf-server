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
  {@const cAgentic = vMessage.timings.agentic}
  <ChatMessageStatistics
    mode={ChatMessageStatisticsMode.GENERATION}
    promptTokens={cAgentic ? cAgentic.llm.prompt_n : vMessage.timings.prompt_n}
    promptMs={cAgentic ? cAgentic.llm.prompt_ms : vMessage.timings.prompt_ms}
    predictedTokens={cAgentic ? cAgentic.llm.predicted_n : vMessage.timings.predicted_n}
    predictedMs={cAgentic ? cAgentic.llm.predicted_ms : vMessage.timings.predicted_ms}
    agenticTimings={cAgentic}
  />
{:else if vIsLoading && vShowMessageStats}
  {@const cLiveStats = vProcessingState.mGetLiveProcessingStats()}
  {@const cGenStats = vProcessingState.mGetLiveGenerationStats()}

  {#if cGenStats}
    <ChatMessageStatistics
      mode={ChatMessageStatisticsMode.GENERATION}
      isLive
      promptTokens={cLiveStats?.tokensProcessed}
      promptMs={cLiveStats?.timeMs}
      predictedTokens={cGenStats.tokensGenerated}
      predictedMs={cGenStats.timeMs}
    />
  {/if}
{/if}
