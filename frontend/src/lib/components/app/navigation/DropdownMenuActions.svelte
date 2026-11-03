<script lang="ts">
  import * as DropdownMenu from '$lib/components/ui/dropdown-menu';
  import * as Tooltip from '$lib/components/ui/tooltip';
  import { KeyboardShortcutInfo } from '$lib/components/app';
  import type { Component } from 'svelte';

  interface ActionItem {
    icon: Component;
    label: string;
    onclick: (pEvent: Event) => void;
    variant?: 'default' | 'destructive';
    disabled?: boolean;
    shortcut?: string[];
    separator?: boolean;
  }

  interface Props {
    triggerIcon: Component;
    triggerTooltip?: string;
    triggerClass?: string;
    actions: ActionItem[];
    align?: 'start' | 'center' | 'end';
    open?: boolean;
  }

  let {
    triggerIcon: vTriggerIcon,
    triggerTooltip: vTriggerTooltip,
    triggerClass: vTriggerClass = '',
    actions: vActions,
    align: vAlign = 'end',
    open: vOpen = $bindable(false)
  }: Props = $props();
</script>

<DropdownMenu.Root bind:open={vOpen}>
  <Tooltip.Root>
    <Tooltip.Trigger>
      <!-- prevent another nested button element -->
      {#snippet child({ props: pProps })}
        <DropdownMenu.Trigger
          {...pProps}
          class="flex h-6 w-6 cursor-pointer items-center justify-center rounded-md p-0 text-sm font-medium transition-colors hover:bg-accent hover:text-accent-foreground focus:bg-accent focus:text-accent-foreground focus:outline-none disabled:pointer-events-none disabled:opacity-50 data-[state=open]:bg-accent data-[state=open]:text-accent-foreground {vTriggerClass}"
          onclick={(pE) => pE.stopPropagation()}
        >
          {@render iconComponent(vTriggerIcon, 'h-3 w-3')}
          {#if vTriggerTooltip}
            <span class="sr-only">{vTriggerTooltip}</span>
          {/if}
        </DropdownMenu.Trigger>
      {/snippet}
    </Tooltip.Trigger>
    {#if vTriggerTooltip}
      <Tooltip.Content>
        <p>{vTriggerTooltip}</p>
      </Tooltip.Content>
    {/if}
  </Tooltip.Root>

  <DropdownMenu.Content align={vAlign} class="z-[999999] w-48">
    {#each vActions as vAction, vIndex (vAction.label)}
      {#if vAction.separator && vIndex > 0}
        <DropdownMenu.Separator />
      {/if}

      <DropdownMenu.Item
        onclick={vAction.onclick}
        variant={vAction.variant}
        disabled={vAction.disabled}
        class="flex items-center justify-between hover:[&>kbd]:opacity-100"
      >
        <div class="flex items-center gap-2">
          {@render iconComponent(
            vAction.icon,
            `h-4 w-4 ${vAction.variant === 'destructive' ? 'text-destructive' : ''}`
          )}
          {vAction.label}
        </div>

        {#if vAction.shortcut}
          <KeyboardShortcutInfo keys={vAction.shortcut} variant={vAction.variant} />
        {/if}
      </DropdownMenu.Item>
    {/each}
  </DropdownMenu.Content>
</DropdownMenu.Root>

{#snippet iconComponent(IconComponent: Component, pClassName: string)}
  <IconComponent class={pClassName} />
{/snippet}
