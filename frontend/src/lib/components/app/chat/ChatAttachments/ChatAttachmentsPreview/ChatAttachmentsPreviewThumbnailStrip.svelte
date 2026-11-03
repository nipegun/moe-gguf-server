<script lang="ts">
  import { fTranslate } from '../../../../../i18n';

  import { cIconClassDefault } from '$lib/constants/css-classes';
  import { Music, Video, FileText } from '@lucide/svelte';
  import { HorizontalScrollCarousel } from '$lib/components/app/misc';

  interface PreviewItem {
    id: string;
    name: string;
    isImage: boolean;
    isAudio: boolean;
    isVideo: boolean;
    preview?: string;
  }

  interface Props {
    items: PreviewItem[];
    currentIndex: number;
    onNavigate: (pIndex: number) => void;
  }

  let { items: vItems, currentIndex: vCurrentIndex, onNavigate: vOnNavigate }: Props = $props();

  function fGetFileExtension(pName: string): string {
    const cParts = pName.split('.');
    if (cParts.length > 1) {
      return cParts.pop()?.toUpperCase() ?? '';
    }
    return '';
  }
</script>

{#if vItems.length > 1}
  <div class="sticky bottom-0 z-10 mt-4 flex-shrink-0">
    <HorizontalScrollCarousel class="max-w-full">
      {#each vItems as vItem, vIndex (vItem.id)}
        <button
          data-thumbnail-index={vIndex}
          class={[
            'relative flex-shrink-0 cursor-pointer overflow-hidden rounded border-2 bg-black/80 backdrop-blur-sm transition-all hover:opacity-90',
            vIndex === vCurrentIndex ? 'border-white' : 'border-transparent opacity-60',
            '[&:not(:first-child)]:last:mr-4 [&:not(:last-child)]:first:ml-4'
          ]}
          onclick={() => vOnNavigate(vIndex)}
          aria-label={fTranslate('text_cdab1f66db51', { p0: vItem.name })}
        >
          {#if vItem.isImage && vItem.preview}
            <img src={vItem.preview} alt={vItem.name} class="h-12 w-12 object-cover" />
          {:else}
            <div
              class="bg-foreground-muted/50 flex h-12 w-12 flex-col items-center justify-center gap-0.5 py-1"
            >
              {#if vItem.isAudio}
                <Music class="{cIconClassDefault} text-white/70" />
              {:else if vItem.isVideo}
                <Video class="{cIconClassDefault} text-white/70" />
              {:else}
                <FileText class="{cIconClassDefault} text-white/70" />
              {/if}

              <span class="font-mono text-[9px] text-white/60">{fGetFileExtension(vItem.name)}</span
              >
            </div>
          {/if}
        </button>
      {/each}
    </HorizontalScrollCarousel>
  </div>
{/if}
