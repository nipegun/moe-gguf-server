<script lang="ts">
  import { BadgeInfo } from '$lib/components/app';
  import * as Tooltip from '$lib/components/ui/tooltip';
  import { copyToClipboard } from '$lib/utils';
  import type { Component } from 'svelte';

  interface Props {
    class?: string;
    icon: Component;
    value: string | number;
    tooltipLabel?: string;
  }

  let {
    class: vClassName = '',
    icon: IconComponent,
    value: vValue,
    tooltipLabel: vTooltipLabel
  }: Props = $props();

  function fHandleClick() {
    void copyToClipboard(String(vValue));
  }
</script>

{#if vTooltipLabel}
  <Tooltip.Root>
    <Tooltip.Trigger>
      <!-- prevent another nested button element -->
      {#snippet child({ props })}
        <BadgeInfo {...props} class={vClassName} onclick={fHandleClick}>
          {#snippet icon()}
            <IconComponent class="h-3 w-3" />
          {/snippet}

          {vValue}
        </BadgeInfo>
      {/snippet}
    </Tooltip.Trigger>
    <Tooltip.Content>
      <p>{vTooltipLabel}</p>
    </Tooltip.Content>
  </Tooltip.Root>
{:else}
  <BadgeInfo class={vClassName} onclick={fHandleClick}>
    {#snippet icon()}
      <IconComponent class="h-3 w-3" />
    {/snippet}

    {vValue}
  </BadgeInfo>
{/if}
