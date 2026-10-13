<script lang="ts">
  import { fTranslate } from '../../../i18n';

  import { Eye, Mic, Video } from '@lucide/svelte';
  import { ModelModality } from '$lib/enums';

  interface Props {
    modalities: ModelModality[];
    class?: string;
  }

  let { modalities: vModalities, class: vClassName = '' }: Props = $props();
</script>

{#each vModalities as modality (modality)}
  {#if modality === ModelModality.VISION || modality === ModelModality.AUDIO || modality === ModelModality.VIDEO}
    <span
      class={[
        'inline-flex items-center gap-1 rounded-md bg-muted px-2 py-1 text-xs font-medium',
        vClassName
      ]}
    >
      {#if modality === ModelModality.VISION}
        <Eye class="h-3 w-3" /> {fTranslate('text_9c50f47085be')}
      {:else if modality === ModelModality.VIDEO}
        <Video class="h-3 w-3" /> {fTranslate('text_f21477c59cc5')}
      {:else}
        <Mic class="h-3 w-3" /> {fTranslate('text_bc1b88907d3b')}
      {/if}
    </span>
  {/if}
{/each}
