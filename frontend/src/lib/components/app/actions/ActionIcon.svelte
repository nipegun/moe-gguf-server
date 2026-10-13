<script lang="ts">
  import { Button, type ButtonVariant, type ButtonSize } from '$lib/components/ui/button';
  import * as Tooltip from '$lib/components/ui/tooltip';
  import type { Component } from 'svelte';
  import { TooltipSide } from '$lib/enums';

  interface Props {
    ariaLabel?: string;
    class?: string;
    disabled?: boolean;
    href?: string;
    icon: Component;
    iconSize?: string;
    onclick?: (pE?: MouseEvent) => void;
    size?: ButtonSize;
    stopPropagationOnClick?: boolean;
    tooltip?: string;
    variant?: ButtonVariant;
    tooltipSide?: TooltipSide;
  }

  let {
    icon: vIcon,
    tooltip: vTooltip,
    variant: vVariant = 'ghost',
    href: vHref = '',
    size: vSize = 'sm',
    class: vClassName = '',
    disabled: vDisabled = false,
    iconSize: vIconSize = 'h-3 w-3',
    tooltipSide: vTooltipSide = TooltipSide.TOP,
    stopPropagationOnClick: vStopPropagationOnClick = false,
    onclick: vOnclick,
    ariaLabel: vAriaLabel
  }: Props = $props();

  let vInnerWidth = $state(0);
  const cShowTooltip = $derived(!!vTooltip && vInnerWidth > 768);
</script>

{#snippet button(props = {})}
  <Button
    {...props}
    href={vHref}
    variant={vVariant}
    size={vSize}
    disabled={vDisabled}
    onclick={(e: MouseEvent) => {
      if (vStopPropagationOnClick) e.stopPropagation();

      vOnclick?.(e);
    }}
    class="h-6 w-6 p-0 {vClassName} flex hover:bg-transparent data-[state=open]:bg-transparent!"
    aria-label={vAriaLabel || vTooltip}
  >
    {#if vIcon}
      {@const IconComponent = vIcon}

      <IconComponent class={vIconSize} />
    {/if}
  </Button>
{/snippet}

{#if cShowTooltip}
  <Tooltip.Root>
    <Tooltip.Trigger>
      <!-- prevent another nested button element -->
      {#snippet child({ props })}
        {#if vDisabled}
          <!-- disabled buttons have pointer-events:none; wrap in a span so the tooltip hover surface stays alive -->
          <span {...props}>
            {@render button({})}
          </span>
        {:else}
          {@render button(props)}
        {/if}
      {/snippet}
    </Tooltip.Trigger>

    <Tooltip.Content side={vTooltipSide}>
      <p>{vTooltip}</p>
    </Tooltip.Content>
  </Tooltip.Root>
{:else}
  {@render button({ href: vHref })}
{/if}

<svelte:window bind:innerWidth={vInnerWidth} />
