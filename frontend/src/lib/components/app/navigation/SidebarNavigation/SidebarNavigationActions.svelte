<script lang="ts">
  import { fTranslate } from '../../../../i18n';

  import { cIconClassDefault } from '$lib/constants/css-classes';
  import { goto } from '$app/navigation';
  import { page } from '$app/state';
  import { Search } from '@lucide/svelte';
  import { ActionIcon, KeyboardShortcutInfo, SearchInput } from '$lib/components/app';
  import { Button } from '$lib/components/ui/button';
  import {
    cIconStripTransitionDuration,
    cIconStripTransitionDelayMultiplier,
    cRoutes,
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
    isSearchModeActive: boolean;
    searchQuery: string;
    onSearchDeactivated?: () => void;
    onSearchClick?: () => void;
    onNewChat?: () => void;
  }

  let {
    class: vClassName,
    isExpandedMode: vIsExpandedMode = false,
    isSearchModeActive: vIsSearchModeActive = $bindable(false),
    searchQuery: vSearchQuery = $bindable(''),
    onSearchDeactivated: vOnSearchDeactivated,
    onSearchClick: vOnSearchClick,
    onNewChat: vOnNewChat
  }: Props = $props();

  let vInitialized = $state(false);
  let vShowIcons = $state(false);
  let vSearchInputRef = $state<HTMLInputElement | null>(null);

  const cIsOnMobile = $derived(cIsMobile.current);

  $effect(() => {
    if (vIsSearchModeActive && vSearchInputRef) {
      vSearchInputRef.focus();
    }
  });

  onMount(() => {
    vShowIcons = true;

    setTimeout(() => {
      vInitialized = true;
    }, cIconStripTransitionDelayMultiplier * aSidebarActionsItems.length);
  });

  function fHandleSearchModeDeactivate() {
    vIsSearchModeActive = false;
    vSearchQuery = '';
    vOnSearchDeactivated?.();
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
  <IconComponent class={cIconClassDefault} />
{/snippet}

{#if vIsSearchModeActive}
  <div class="px-4 my-2">
    <SearchInput
      bind:value={vSearchQuery}
      bind:ref={vSearchInputRef}
      onClose={fHandleSearchModeDeactivate}
      onKeyDown={(e) => e.key === 'Escape' && fHandleSearchModeDeactivate()}
      placeholder={fTranslate('text_e90dd0173105')}
    />
  </div>
{:else if vIsExpandedMode || cIsOnMobile}
  <div
    class="{vClassName} flex flex-col gap-5 md:gap-1 mt-2 md:mt-0 {!vIsExpandedMode && cIsOnMobile
      ? 'hidden pointer-events-none'
      : ''}"
  >
    {#each aSidebarActionsItems as item, i (item.tooltip)}
      {@const isActive = fIsItemActive(item)}
      {@const isSearchOnMobile = item.icon === Search && cIsMobile.current}
      {@const itemHref = isSearchOnMobile ? cRoutes.SEARCH : item.route}
      {@const itemOnClick = item.route
        ? () => {
            vOnNewChat?.();
            goto(item.route!);
          }
        : isSearchOnMobile
          ? undefined
          : vOnSearchClick}
      {@const itemTransition = {
        duration: cIconStripTransitionDuration,
        delay: !vInitialized ? i * cIconStripTransitionDelayMultiplier : 0,
        easing: circIn
      }}

      {#if vShowIcons}
        <div transition:fade={itemTransition}>
          <Button
            class="w-full min-w-9 justify-between px-2 backdrop-blur-none! hover:[&>kbd]:opacity-100 {isActive
              ? 'bg-accent text-accent-foreground'
              : ''}"
            href={itemHref}
            onclick={itemOnClick}
            variant="ghost"
            size="default"
          >
            <span class="flex min-w-0 items-center px-0.5 gap-2">
              {@render itemIcon(item.icon)}

              {#if vShowIcons}
                <span in:fade={itemTransition} out:fade={itemTransition} class="min-w-0 truncate"
                  >{item.tooltip}</span
                >
              {/if}
            </span>

            {#if item.keys}
              <KeyboardShortcutInfo keys={item.keys} />
            {/if}
          </Button>
        </div>
      {/if}
    {/each}
  </div>
{:else}
  <div class="{vClassName} flex-col gap-1 hidden md:flex">
    {#each aSidebarActionsItems as item, i (item.tooltip)}
      {@const isActive = fIsItemActive(item)}
      {@const isSearchOnMobile = item.icon === Search && cIsMobile.current}
      {@const itemOnClick = item.route
        ? () => {
            vOnNewChat?.();
            goto(item.route!);
          }
        : isSearchOnMobile
          ? undefined
          : vOnSearchClick}
      {@const itemTransition = {
        duration: cIconStripTransitionDuration,
        delay: !vInitialized ? i * cIconStripTransitionDelayMultiplier : 0,
        easing: circIn
      }}

      {#if vShowIcons}
        <div transition:fade={itemTransition}>
          <ActionIcon
            icon={item.icon}
            tooltip={item.tooltip}
            tooltipSide={TooltipSide.RIGHT}
            size="lg"
            iconSize={cIconClassDefault}
            class="h-9 w-9 rounded-full hover:bg-accent! {isActive
              ? 'bg-accent text-accent-foreground'
              : ''}"
            onclick={itemOnClick}
          />
        </div>
      {/if}
    {/each}
  </div>
{/if}
