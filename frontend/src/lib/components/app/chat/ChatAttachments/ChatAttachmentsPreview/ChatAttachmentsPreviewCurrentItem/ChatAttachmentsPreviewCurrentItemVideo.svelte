<script lang="ts">
  import { fTranslate } from '../../../../../../i18n';

  import { Video } from '@lucide/svelte';

  interface Props {
    currentItem: { name?: string } | null;
    videoSrc: string | null;
  }

  let { currentItem: vCurrentItem, videoSrc: vVideoSrc }: Props = $props();
</script>

<div class="flex flex-1 items-center justify-center p-8">
  <div class="w-full max-w-md text-center">
    <Video class="mx-auto mb-4 h-16 w-16 text-white/50" />

    {#if vVideoSrc}
      <video controls class="mb-4 w-full" src={vVideoSrc}>
        <track kind="captions" src="" />
        {fTranslate('text_0a63af83e9f4')}
      </video>
    {:else}
      <p class="mb-4 text-white/70">{fTranslate('text_f70e8aef44b2')}</p>
    {/if}

    <p class="text-sm text-white/50">{vCurrentItem?.name || 'Video'}</p>
  </div>
</div>
