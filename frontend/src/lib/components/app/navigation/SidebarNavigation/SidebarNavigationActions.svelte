<script lang="ts">
  import { cSidebarIconClass } from '$lib/constants/css-classes';
  import { goto } from '$app/navigation';
  import { page } from '$app/state';
  import { ActionIcon, KeyboardShortcutInfo } from '$lib/components/app';
  import { Button } from '$lib/components/ui/button';
  import {
    cIconStripTransitionDuration,
    cIconStripTransitionDelayMultiplier,
    aSidebarActionsItems
  } from '$lib/constants';
  import { cIsMobile } from '$lib/stores/viewport.svelte';
  import { TooltipSide } from '$lib/enums';
  import { fade } from 'svelte/transition';
  import { circIn } from 'svelte/easing';
  import { onMount } from 'svelte';
  import type { Component } from 'svelte';

  interface Props {
    class: string;
    isExpandedMode: boolean;
    /** before an item navigates, so the sidebar can collapse on mobile */
    onNavigate?: () => void;
  }

  let {
    class: vClassName,
    isExpandedMode: vIsExpandedMode = false,
    onNavigate: vOnNavigate
  }: Props = $props();

  let vInitialized = $state(false);
  let vShowIcons = $state(false);

  const cIsOnMobile = $derived(cIsMobile.current);

  onMount(() => {
    vShowIcons = true;

    setTimeout(() => {
      vInitialized = true;
    }, cIconStripTransitionDelayMultiplier * aSidebarActionsItems.length);
  });

  function fOpen(pRoute: string) {
    vOnNavigate?.();
    goto(pRoute);
  }

  function fIsItemActive(pItem: {
    activeRouteId?: string;
    activeRoutePrefix?: string;
    activeUrlIncludes?: string;
  }): boolean {
    if (pItem.activeRouteId) {
      return page.route.id === pItem.activeRouteId;
    }

    if (pItem.activeRoutePrefix) {
      return !!page.route.id?.startsWith(pItem.activeRoutePrefix);
    }

    if (pItem.activeUrlIncludes) {
      return page.url?.hash?.includes(pItem.activeUrlIncludes) ?? false;
    }

    return false;
  }
</script>

{#snippet itemIcon(IconComponent: Component)}
  <IconComponent class="{cSidebarIconClass} shrink-0" />
{/snippet}

{#if vIsExpandedMode || cIsOnMobile}
  <div
    class="{vClassName} flex flex-col gap-5 md:gap-1 mt-2 md:mt-0 {!vIsExpandedMode && cIsOnMobile
      ? 'hidden pointer-events-none'
      : ''}"
  >
    {#each aSidebarActionsItems as vItem, vI (vItem.tooltip)}
      {@const cIsActive = fIsItemActive(vItem)}
      {@const cItemTransition = {
        duration: cIconStripTransitionDuration,
        delay: !vInitialized ? vI * cIconStripTransitionDelayMultiplier : 0,
        easing: circIn
      }}

      {#if vShowIcons}
        <div transition:fade={cItemTransition}>
          <Button
            class="w-full min-w-9 justify-between px-2 backdrop-blur-none! hover:[&>kbd]:opacity-100 {cIsActive
              ? 'bg-accent text-accent-foreground'
              : ''}"
            href={vItem.route}
            onclick={() => fOpen(vItem.route)}
            variant="ghost"
            size="default"
          >
            <span class="flex min-w-0 items-center px-0.5 gap-2">
              {@render itemIcon(vItem.icon)}

              {#if vShowIcons}
                <span in:fade={cItemTransition} out:fade={cItemTransition} class="min-w-0 truncate"
                  >{vItem.tooltip}</span
                >
              {/if}
            </span>

            {#if vItem.keys}
              <KeyboardShortcutInfo keys={vItem.keys} />
            {/if}
          </Button>
        </div>
      {/if}
    {/each}
  </div>
{:else}
  <div class="{vClassName} flex-col gap-1 hidden md:flex">
    {#each aSidebarActionsItems as vItem, vI (vItem.tooltip)}
      {@const cIsActive = fIsItemActive(vItem)}
      {@const cItemTransition = {
        duration: cIconStripTransitionDuration,
        delay: !vInitialized ? vI * cIconStripTransitionDelayMultiplier : 0,
        easing: circIn
      }}

      {#if vShowIcons}
        <div transition:fade={cItemTransition}>
          <ActionIcon
            icon={vItem.icon}
            tooltip={vItem.tooltip}
            tooltipSide={TooltipSide.RIGHT}
            size="lg"
            iconSize={cSidebarIconClass}
            class="h-9 w-9 rounded-full hover:bg-accent! {cIsActive
              ? 'bg-accent text-accent-foreground'
              : ''}"
            onclick={() => fOpen(vItem.route)}
          />
        </div>
      {/if}
    {/each}
  </div>
{/if}
