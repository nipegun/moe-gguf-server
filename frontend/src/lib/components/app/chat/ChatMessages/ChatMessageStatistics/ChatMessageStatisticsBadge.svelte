<script lang="ts">
  import { BadgeInfo } from '$lib/components/app';
  import * as Tooltip from '$lib/components/ui/tooltip';
  import { fCopyToClipboard } from '$lib/utils';
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
    void fCopyToClipboard(String(vValue));
  }
</script>

{#if vTooltipLabel}
  <Tooltip.Root>
    <Tooltip.Trigger>
      <!-- prevent another nested button element -->
      {#snippet child({ props: pProps })}
        <BadgeInfo {...pProps} class={vClassName} onclick={fHandleClick}>
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
