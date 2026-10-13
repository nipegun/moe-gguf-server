<script lang="ts">
  import { fade } from 'svelte/transition';
  import type { UseProcessingStateReturn } from '$lib/hooks/use-processing-state.svelte';

  interface Props {
    modelLoadingText: string | null;
    /** Queue notice while the request waits behind others; it goes before every other text */
    queueText?: string | null;
    processingState: UseProcessingStateReturn;
    position: 'top' | 'bottom';
  }

  let {
    modelLoadingText: vModelLoadingText,
    queueText: vQueueText = null,
    processingState: vProcessingState,
    position: vPosition
  }: Props = $props();

  const cMarginClass = $derived(vPosition === 'top' ? 'mt-6' : 'mt-4');
</script>

<div class="{cMarginClass} w-full max-w-3xl" in:fade>
  <div class="flex min-w-0 flex-col items-start gap-2">
    <span class="shimmer-text text-sm [overflow-wrap:anywhere]">
      {vQueueText ??
        vModelLoadingText ??
        vProcessingState.getPromptProgressText() ??
        vProcessingState.getProcessingMessage() ??
        'Processing...'}
    </span>
  </div>
</div>
