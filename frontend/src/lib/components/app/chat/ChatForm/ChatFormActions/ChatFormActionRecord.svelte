<script lang="ts">
  import { fTranslate } from '../../../../../i18n';

  import { cIconClassDefault } from '$lib/constants/css-classes';
  import { Mic, Square } from '@lucide/svelte';
  import { Button } from '$lib/components/ui/button';
  import * as Tooltip from '$lib/components/ui/tooltip';

  interface Props {
    class?: string;
    disabled?: boolean;
    hasAudioModality?: boolean;
    isLoading?: boolean;
    isRecording?: boolean;
    onMicClick?: () => void;
  }

  let {
    class: vClassName = '',
    disabled: vDisabled = false,
    hasAudioModality: vHasAudioModality = false,
    isLoading: vIsLoading = false,
    isRecording: vIsRecording = false,
    onMicClick: vOnMicClick
  }: Props = $props();
</script>

<div class="flex items-center gap-1 {vClassName}">
  <Tooltip.Root>
    <Tooltip.Trigger>
      <Button
        class="h-8 w-8 rounded-full p-0 {vIsRecording
          ? 'animate-pulse bg-red-500 text-white hover:bg-red-600'
          : ''}"
        disabled={vDisabled || vIsLoading || !vHasAudioModality}
        onclick={vOnMicClick}
        type="button"
      >
        <span class="sr-only"
          >{vIsRecording ? fTranslate('text_3d591e826d54') : fTranslate('text_30756af0134e')}</span
        >

        {#if vIsRecording}
          <Square class="{cIconClassDefault} animate-pulse fill-white" />
        {:else}
          <Mic class={cIconClassDefault} />
        {/if}
      </Button>
    </Tooltip.Trigger>

    {#if !vHasAudioModality}
      <Tooltip.Content>
        <p>{fTranslate('text_5408a261204f')}</p>
      </Tooltip.Content>
    {/if}
  </Tooltip.Root>
</div>
