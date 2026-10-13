<script lang="ts" generics="T">
  import { fTranslate } from '../../../../../../i18n';

  import type { Snippet } from 'svelte';
  import { SearchInput } from '$lib/components/app';
  import ScrollArea from '$lib/components/ui/scroll-area/scroll-area.svelte';
  import { cChatFormPopoverMaxHeight } from '$lib/constants';

  interface Props {
    items: T[];
    isLoading: boolean;
    selectedIndex: number;
    searchQuery: string;
    showSearchInput: boolean;
    searchPlaceholder?: string;
    emptyMessage?: string;
    itemKey: (pItem: T, pIndex: number) => string;
    item: Snippet<[T, number, boolean]>;
    skeleton?: Snippet;
    footer?: Snippet;
  }

  let {
    items: vItems,
    isLoading: vIsLoading,
    selectedIndex: vSelectedIndex,
    searchQuery: vSearchQuery = $bindable(),
    showSearchInput: vShowSearchInput,
    searchPlaceholder: vSearchPlaceholder = 'Search...',
    emptyMessage: vEmptyMessage = fTranslate('text_0c9ecb6faaf9'),
    itemKey: vItemKey,
    item: vItem,
    skeleton: vSkeleton,
    footer: vFooter
  }: Props = $props();

  let vListContainer = $state<HTMLDivElement | null>(null);

  $effect(() => {
    if (vListContainer && vSelectedIndex >= 0 && vSelectedIndex < vItems.length) {
      const cSelectedElement = vListContainer.querySelector(
        `[data-picker-index="${vSelectedIndex}"]`
      ) as HTMLElement;

      if (cSelectedElement) {
        cSelectedElement.scrollIntoView({
          behavior: 'smooth',
          block: 'center',
          inline: 'nearest'
        });
      }
    }
  });
</script>

<ScrollArea>
  {#if vShowSearchInput}
    <div class="absolute top-0 right-0 left-0 z-10 p-2 pb-0">
      <SearchInput placeholder={vSearchPlaceholder} bind:value={vSearchQuery} />
    </div>
  {/if}

  <div
    bind:this={vListContainer}
    class={[`${cChatFormPopoverMaxHeight} p-2`, vShowSearchInput && 'pt-13']}
  >
    {#if vIsLoading}
      {#if vSkeleton}
        {@render vSkeleton()}
      {/if}
    {:else if vItems.length === 0}
      <div class="py-6 text-center text-sm text-muted-foreground">{vEmptyMessage}</div>
    {:else}
      {#each vItems as itemData, index (vItemKey(itemData, index))}
        {@render vItem(itemData, index, index === vSelectedIndex)}
      {/each}
    {/if}
  </div>

  {#if vFooter}
    {@render vFooter()}
  {/if}
</ScrollArea>
