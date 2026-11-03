<script lang="ts">
  import { fTranslate } from '../../../i18n';

  import type { Snippet } from 'svelte';
  import * as DropdownMenu from '$lib/components/ui/dropdown-menu';
  import { SearchInput } from '$lib/components/app';

  interface Props {
    placeholder?: string;
    searchValue?: string;
    onSearchChange?: (pValue: string) => void;
    onSearchKeyDown?: (pEvent: KeyboardEvent) => void;
    emptyMessage?: string;
    isEmpty?: boolean;
    children: Snippet;
    footer?: Snippet;
  }

  let {
    placeholder: vPlaceholder = fTranslate('text_7f55382219f0'),
    searchValue: vSearchValue = $bindable(''),
    onSearchChange: vOnSearchChange,
    onSearchKeyDown: vOnSearchKeyDown,
    emptyMessage: vEmptyMessage = fTranslate('text_949da6a3278e'),
    isEmpty: vIsEmpty = false,
    children: vChildren,
    footer: vFooter
  }: Props = $props();
</script>

<div class="sticky top-0 z-10 mb-2 bg-popover p-1 pt-2">
  <SearchInput
    placeholder={vPlaceholder}
    bind:value={vSearchValue}
    onInput={vOnSearchChange}
    onKeyDown={vOnSearchKeyDown}
  />
</div>

<div class="overflow-y-auto">
  {@render vChildren()}

  {#if vIsEmpty}
    <div class="px-2 py-3 text-center text-sm text-muted-foreground">{vEmptyMessage}</div>
  {/if}
</div>

{#if vFooter}
  <DropdownMenu.Separator />

  {@render vFooter()}
{/if}
