<script lang="ts">
  import { fTranslate } from '../../../i18n';

  import { cIconClassDefault } from '$lib/constants/css-classes';
  import ChevronDown from '@lucide/svelte/icons/chevron-down';
  import * as Collapsible from '$lib/components/ui/collapsible/index.js';
  import { cn } from '$lib/components/ui/utils';
  import type { Snippet } from 'svelte';
  import type { Component } from 'svelte';

  interface Props {
    open?: boolean;
    class?: string;
    icon?: Component;
    iconClass?: string;
    iconUrl?: string | null;
    title?: string;
    titleSnippet?: Snippet;
    subtitle?: string;
    shimmerTitle?: boolean;
    onToggle?: () => void;
    children: Snippet;
  }

  let {
    open: vOpen = $bindable(false),
    class: vClassName = '',
    icon: IconComponent,
    iconClass: vIconClass = cIconClassDefault,
    iconUrl: vIconUrl = null,
    title: vTitle = '',
    titleSnippet: vTitleSnippet,
    subtitle: vSubtitle,
    shimmerTitle: vShimmerTitle = false,
    onToggle: vOnToggle,
    children: vChildren
  }: Props = $props();

  function fHideBrokenIcon(pEvent: Event) {
    (pEvent.currentTarget as HTMLImageElement).style.display = 'none';
  }
</script>

<Collapsible.Root
  open={vOpen}
  onOpenChange={(value) => {
    vOpen = value;
    vOnToggle?.();
  }}
  class={cn('group/collapsible', 'my-0!', vClassName)}
>
  <Collapsible.Trigger
    class={cn(
      'flex w-full cursor-pointer items-start justify-between gap-2 text-left',
      'py-1.5 pr-1'
    )}
  >
    <div class="flex min-w-0 items-start gap-2 text-muted-foreground">
      {#if vIconUrl}
        <img
          src={vIconUrl}
          alt=""
          class={cn('shrink-0 rounded-sm  mt-0.75', vIconClass)}
          onerror={fHideBrokenIcon}
        />
      {:else if IconComponent}
        <IconComponent class={cn('shrink-0 text-muted-foreground/60 mt-0.75', vIconClass)} />
      {/if}

      <span
        class={cn('text-sm font-medium', vShimmerTitle ? 'shimmer-text' : 'text-foreground/80')}
      >
        {#if vTitleSnippet}
          {@render vTitleSnippet()}
        {:else}
          {vTitle}
        {/if}
      </span>

      {#if vSubtitle}
        <span class="text-xs italic text-muted-foreground/70">{vSubtitle}</span>
      {/if}
    </div>

    <ChevronDown
      class={cn(
        'size-4 shrink-0 text-muted-foreground/60 transition-all duration-150 ease-out opacity-0 group-hover/collapsible:opacity-100 mt-0.75',
        vOpen && 'rotate-180'
      )}
    />

    <span class="sr-only">{fTranslate('text_fc7a07580f42')}</span>
  </Collapsible.Trigger>

  <Collapsible.Content>
    <!-- Collapsible.Content renders its children unconditionally and only sets
         `hidden`, so a closed block would keep re-rendering its whole body on
         every streamed token. Gate on `open` so collapsed content costs nothing. -->
    {#if vOpen}
      <div class="pl-1.5 grid min-w-0" style="min-height: var(--min-message-height);">
        <div class="min-w-0 border-l border-muted-foreground/20 pl-4 pb-2 my-2">
          {@render vChildren()}
        </div>
      </div>
    {/if}
  </Collapsible.Content>
</Collapsible.Root>
