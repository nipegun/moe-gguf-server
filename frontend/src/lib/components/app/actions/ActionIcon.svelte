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

{#snippet button(pProps = {})}
  <Button
    {...pProps}
    href={vHref}
    variant={vVariant}
    size={vSize}
    disabled={vDisabled}
    onclick={(pE: MouseEvent) => {
      if (vStopPropagationOnClick) pE.stopPropagation();

      vOnclick?.(pE);
    }}
    class="h-6 w-6 p-0 has-[>svg]:p-0 {vClassName} flex hover:bg-transparent data-[state=open]:bg-transparent!"
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
      {#snippet child({ props: pProps })}
        {#if vDisabled}
          <!-- disabled buttons have pointer-events:none; wrap in a span so the tooltip hover surface stays alive -->
          <span {...pProps}>
            {@render button({})}
          </span>
        {:else}
          {@render button(pProps)}
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
