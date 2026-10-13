<script lang="ts">
  import { fTranslate } from '../../../../../../i18n';

  import { Music } from '@lucide/svelte';

  interface Props {
    currentItem: { name?: string } | null;
    audioSrc: string | null;
  }

  let { currentItem: vCurrentItem, audioSrc: vAudioSrc }: Props = $props();
</script>

<div class="flex flex-1 items-center justify-center p-8">
  <div class="w-full max-w-md text-center">
    <Music class="mx-auto mb-4 h-16 w-16 text-white/50" />

    {#if vAudioSrc}
      <audio controls class="mb-4 w-full" src={vAudioSrc}>
        {fTranslate('text_a21b58ba4c2f')}
      </audio>
    {:else}
      <p class="mb-4 text-white/70">{fTranslate('text_b4fa97f78aa1')}</p>
    {/if}

    <p class="text-sm text-white/50">{vCurrentItem?.name || 'Audio'}</p>
  </div>
</div>
