<script lang="ts">
  import { fTranslate } from '../../../../i18n';

  import { cIconClassDefault } from '$lib/constants/css-classes';
  import type { Component } from 'svelte';
  import { Button, type ButtonVariant } from '$lib/components/ui/button';

  let {
    title: aTitle,
    description: aDescription,
    IconComponent,
    buttonText: aButtonText,
    onclick: aOnclick,
    titleClass: aTitleClass,
    buttonVariant: aButtonVariant,
    buttonClass: aButtonClass,
    wrapperClass: aWrapperClass,
    summary: aSummary
  }: {
    title: string;
    description: string;
    IconComponent: Component;
    buttonText: string;
    onclick: () => void;
    titleClass?: string;
    buttonVariant?: ButtonVariant;
    buttonClass?: string;
    wrapperClass?: string;
    summary?: { show: boolean; verb: string; items: DatabaseConversation[] };
  } = $props();

  let vSectionButtonClass = $derived(aButtonClass ?? 'justify-start justify-self-start md:w-auto');
  let vSectionButtonVariant = $derived(aButtonVariant ?? 'outline');
</script>

<div class="grid gap-1 {aWrapperClass ?? ''}">
  <h4 class="mt-0 mb-2 text-sm font-medium {aTitleClass ?? ''}">{aTitle}</h4>

  <p class="mb-4 text-sm text-muted-foreground">{aDescription}</p>

  <Button class={vSectionButtonClass} onclick={aOnclick} variant={vSectionButtonVariant}>
    <IconComponent class="mr-2 {cIconClassDefault}" />

    {aButtonText}
  </Button>

  {#if aSummary && aSummary.show && aSummary.items.length > 0}
    <div class="mt-4 grid overflow-x-auto rounded-lg border border-border/50 bg-muted/30 p-4">
      <h5 class="mb-2 text-sm font-medium">
        {aSummary.verb}
        {aSummary.items.length}
        {fTranslate('text_8b34dbc2c05e')}{aSummary.items.length === 1 ? '' : 's'}
      </h5>

      <ul class="space-y-1 text-sm text-muted-foreground">
        {#each aSummary.items.slice(0, 10) as conv (conv.id)}
          <li class="truncate">• {conv.name || fTranslate('text_31d248c44579')}</li>
        {/each}

        {#if aSummary.items.length > 10}
          <li class="italic">
            {fTranslate('text_4044e985a090')}
            {aSummary.items.length - 10}
            {fTranslate('text_187897ce0afc')}
          </li>
        {/if}
      </ul>
    </div>
  {/if}
</div>
