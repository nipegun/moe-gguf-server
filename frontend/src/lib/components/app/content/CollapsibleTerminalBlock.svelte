<script lang="ts">
  import { fTranslate } from '../../../i18n';

  import ChevronDown from '@lucide/svelte/icons/chevron-down';
  import * as Collapsible from '$lib/components/ui/collapsible/index.js';
  import { cn } from '$lib/components/ui/utils';
  import { cIconClassDefault } from '$lib/constants';
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
  class={cn('group/collapsible', 'overflow-hidden rounded-md', vClassName)}
  style="background: var(--code-background); border: 1px solid color-mix(in oklch, var(--border) 30%, transparent);"
>
  <Collapsible.Trigger
    class={cn(
      'flex w-full cursor-pointer items-start justify-between gap-2 text-left',
      'px-3 py-2'
    )}
  >
    <div class="flex min-w-0 items-start gap-2 text-muted-foreground">
      {#if vIconUrl}
        <img
          src={vIconUrl}
          alt=""
          class={cn('shrink-0 rounded-sm mt-0.5', vIconClass)}
          onerror={fHideBrokenIcon}
        />
      {:else if IconComponent}
        <IconComponent class={cn('shrink-0 text-muted-foreground/60 mt-0.5', vIconClass)} />
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
        'size-4 shrink-0 text-muted-foreground/60 transition-all duration-150 ease-out opacity-0 group-hover/collapsible:opacity-100 mt-0.5',
        vOpen && 'rotate-180'
      )}
    />

    <span class="sr-only">{fTranslate('text_fc7a07580f42')}</span>
  </Collapsible.Trigger>

  <Collapsible.Content>
    <!-- See CollapsibleContentBlock: bits-ui keeps closed content mounted, which
         makes a collapsed tool result re-render on every streamed token. -->
    {#if vOpen}
      <div class="p-3 pt-1">
        {@render vChildren()}
      </div>
    {/if}
  </Collapsible.Content>
</Collapsible.Root>
