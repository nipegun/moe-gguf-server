<script lang="ts">
  import { fTranslate } from '../../../../i18n';

  import { cIconClassDefault } from '$lib/constants/css-classes';
  import { RefreshCw, Loader2 } from '@lucide/svelte';
  import { Button } from '$lib/components/ui/button';
  import { SearchInput } from '$lib/components/app/forms';

  interface Props {
    isLoading: boolean;
    onRefresh: () => void;
    onSearch?: (pQuery: string) => void;
    searchQuery?: string;
  }

  let {
    isLoading: vIsLoading,
    onRefresh: vOnRefresh,
    onSearch: vOnSearch,
    searchQuery: vSearchQuery = ''
  }: Props = $props();
</script>

<div class="flex flex-col gap-2">
  <div class="mb-2 flex items-center gap-4">
    <SearchInput
      placeholder={fTranslate('text_76625c6090df')}
      value={vSearchQuery}
      onInput={(value) => vOnSearch?.(value)}
    />

    <Button
      variant="ghost"
      size="sm"
      class="h-8 w-8 p-0"
      onclick={vOnRefresh}
      disabled={vIsLoading}
      title={fTranslate('text_28a1a41378e7')}
    >
      {#if vIsLoading}
        <Loader2 class="{cIconClassDefault} animate-spin" />
      {:else}
        <RefreshCw class={cIconClassDefault} />
      {/if}
    </Button>
  </div>

  <h3 class="text-sm font-medium">{fTranslate('text_036bcef84f47')}</h3>
</div>
