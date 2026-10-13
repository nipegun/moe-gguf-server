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
      {#snippet child({ props })}
        <DropdownMenu.Trigger
          {...props}
          class="flex h-6 w-6 cursor-pointer items-center justify-center rounded-md p-0 text-sm font-medium transition-colors hover:bg-accent hover:text-accent-foreground focus:bg-accent focus:text-accent-foreground focus:outline-none disabled:pointer-events-none disabled:opacity-50 data-[state=open]:bg-accent data-[state=open]:text-accent-foreground {vTriggerClass}"
          onclick={(e) => e.stopPropagation()}
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
    {#each vActions as action, index (action.label)}
      {#if action.separator && index > 0}
        <DropdownMenu.Separator />
      {/if}

      <DropdownMenu.Item
        onclick={action.onclick}
        variant={action.variant}
        disabled={action.disabled}
        class="flex items-center justify-between hover:[&>kbd]:opacity-100"
      >
        <div class="flex items-center gap-2">
          {@render iconComponent(
            action.icon,
            `h-4 w-4 ${action.variant === 'destructive' ? 'text-destructive' : ''}`
          )}
          {action.label}
        </div>

        {#if action.shortcut}
          <KeyboardShortcutInfo keys={action.shortcut} variant={action.variant} />
        {/if}
      </DropdownMenu.Item>
    {/each}
  </DropdownMenu.Content>
</DropdownMenu.Root>

{#snippet iconComponent(IconComponent: Component, className: string)}
  <IconComponent class={className} />
{/snippet}
