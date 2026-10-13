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
      {#each vItems as item, index (item.id)}
        <button
          data-thumbnail-index={index}
          class={[
            'relative flex-shrink-0 cursor-pointer overflow-hidden rounded border-2 bg-black/80 backdrop-blur-sm transition-all hover:opacity-90',
            index === vCurrentIndex ? 'border-white' : 'border-transparent opacity-60',
            '[&:not(:first-child)]:last:mr-4 [&:not(:last-child)]:first:ml-4'
          ]}
          onclick={() => vOnNavigate(index)}
          aria-label={fTranslate('text_cdab1f66db51', { p0: item.name })}
        >
          {#if item.isImage && item.preview}
            <img src={item.preview} alt={item.name} class="h-12 w-12 object-cover" />
          {:else}
            <div
              class="bg-foreground-muted/50 flex h-12 w-12 flex-col items-center justify-center gap-0.5 py-1"
            >
              {#if item.isAudio}
                <Music class="{cIconClassDefault} text-white/70" />
              {:else if item.isVideo}
                <Video class="{cIconClassDefault} text-white/70" />
              {:else}
                <FileText class="{cIconClassDefault} text-white/70" />
              {/if}

              <span class="font-mono text-[9px] text-white/60">{fGetFileExtension(item.name)}</span>
            </div>
          {/if}
        </button>
      {/each}
    </HorizontalScrollCarousel>
  </div>
{/if}
