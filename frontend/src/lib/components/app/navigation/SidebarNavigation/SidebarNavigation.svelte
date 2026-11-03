<script lang="ts">
  import { fTranslate } from '../../../../i18n';

  import { goto } from '$app/navigation';
  import { page } from '$app/state';
  import { tick } from 'svelte';
  import { Menu } from '@lucide/svelte';
  import {
    ActionIcon,
    DialogConversationRename,
    SearchInput,
    SidebarNavigationConversationList,
    SidebarNavigationActions
  } from '$lib/components/app';
  import SidebarNavigationGlobalSearch from './SidebarNavigationGlobalSearch.svelte';
  import { cRoutes } from '$lib/constants';
  import { cSidebarIconClass } from '$lib/constants/css-classes';
  import { fGroupSidebarConversations } from '$lib/utils';
  import { fade } from 'svelte/transition';
  import { SvelteSet } from 'svelte/reactivity';
  import { fUseMarqueeSelection } from '$lib/hooks/use-marquee-selection.svelte';

  import { fUseKeyboardShortcuts } from '$lib/hooks/use-keyboard-shortcuts.svelte';
  import { cConversationsStore, fConversations } from '$lib/stores/conversations.svelte';
  import { cChatStore } from '$lib/stores/chat.svelte';
  import { fConfig } from '$lib/stores/settings.svelte';
  import { RouterService } from '$lib/services/router.service';
  import { cIsMobile } from '$lib/stores/viewport.svelte';
  import { TooltipSide } from '$lib/enums';
  import { cDevice } from '$lib/stores/device.svelte';

  const { mHandleKeydown: cHandleKeydown } = fUseKeyboardShortcuts({
    activateSearchMode: () => void fFocusSearch(),
    toggleSidebar: () => fToggleExpandedMode()
  });

  let vIsExpandedMode = $state(false);
  let vHoveredTooltip = $state<string | null>(null);

  const cIsStripExpanded = $derived(vIsExpandedMode || vHoveredTooltip !== null);
  const cIsOnMobile = $derived(cIsMobile.current);
  const cAlwaysShowOnDesktop = $derived(fConfig().alwaysShowSidebarOnDesktop as boolean);

  $effect(() => {
    if (cAlwaysShowOnDesktop && !cIsOnMobile) {
      vIsExpandedMode = true;
    }
  });

  function fToggleExpandedMode() {
    vIsExpandedMode = !vIsExpandedMode;
    if (!vIsExpandedMode) {
      vHoveredTooltip = null;
    }
  }

  $effect(() => {
    if (!vIsExpandedMode) {
      vSearchQuery = '';
      if (vIsSelectionMode) fExitSelectionMode();
      fCancelMobileCollapse();
    }
  });

  let vCurrentChatId = $derived(page.params.id);

  // the global search: chats and their messages, pages, settings, models, tools and MCP servers
  let vSearchQuery = $state('');
  let vSearchInputRef = $state<HTMLInputElement | null>(null);
  let vGlobalSearch = $state<{ mOpenFirst: () => void } | undefined>();
  let vIsSearching = $derived(vSearchQuery.trim().length > 0);

  async function fFocusSearch() {
    vIsExpandedMode = true;
    await tick();
    vSearchInputRef?.focus();
  }

  function fHandleSearchKeydown(pEvent: KeyboardEvent) {
    if (pEvent.key === 'Enter') {
      pEvent.preventDefault();
      vGlobalSearch?.mOpenFirst();
    } else if (pEvent.key === 'Escape') {
      if (vSearchQuery) {
        vSearchQuery = '';
      } else {
        vSearchInputRef?.blur();
      }
    }
  }

  function fHandleSearchNavigated() {
    vSearchQuery = '';
    vSearchInputRef?.blur();
    if (cIsMobile.current) {
      fScheduleMobileCollapse();
    }
  }

  let vFilteredConversations = $derived(fConversations());

  // pinned chats, "Recent chats" (the five worked with last) and "Chats" (the rest, newest first)
  let vConversationGroups = $derived(fGroupSidebarConversations(vFilteredConversations));

  let vIsSelectionMode = $state(false);
  let vSelectedIds = new SvelteSet<string>();

  let vRenameDialogOpen = $state(false);
  let vRenameTargetConversationId = $state<string | null>(null);
  let vRenameDraft = $state('');
  let vRenameOriginalTitle = $state('');

  const cRenderedOrderIds = $derived(
    [...vConversationGroups.pinned, ...vConversationGroups.recent, ...vConversationGroups.others].map(
      (pC) => pC.id
    )
  );

  const cAllSelectedArePinned = $derived.by(() => {
    if (vSelectedIds.size === 0) return false;
    const cConvs = fConversations();
    for (const cId of vSelectedIds) {
      const cC = cConvs.find((pConv) => pConv.id === cId);
      if (cC && !cC.pinned) return false;
    }
    return true;
  });

  const cPinStateIsMixed = $derived.by(() => {
    if (vSelectedIds.size === 0) return false;
    const cConvs = fConversations();
    let vAnyPinned = false;
    let vAnyUnpinned = false;
    for (const cId of vSelectedIds) {
      const cC = cConvs.find((pConv) => pConv.id === cId);
      if (!cC) continue;
      if (cC.pinned) vAnyPinned = true;
      else vAnyUnpinned = true;
      if (vAnyPinned && vAnyUnpinned) return true;
    }
    return false;
  });

  const cVisibleSelectionStats = $derived.by(() => {
    const cVisibleIds = vFilteredConversations.map((pC) => pC.id);
    let vSelectedVisible = 0;
    for (const cId of cVisibleIds) {
      if (vSelectedIds.has(cId)) vSelectedVisible++;
    }
    return {
      visibleCount: cVisibleIds.length,
      selectedVisibleCount: vSelectedVisible
    };
  });

  function fEnterSelectionMode(pId?: string) {
    vIsSelectionMode = true;
    if (pId !== undefined) {
      vSelectedIds.add(pId);
    }
  }

  function fExitSelectionMode() {
    vIsSelectionMode = false;
    vSelectedIds.clear();
  }

  function fToggleSelected(pId: string) {
    if (vSelectedIds.has(pId)) {
      vSelectedIds.delete(pId);
    } else {
      vSelectedIds.add(pId);
    }
  }

  function fToggleSelectAllVisible() {
    const cVisibleIds = vFilteredConversations.map((pC) => pC.id);
    const cAllSelected =
      cVisibleIds.length > 0 && cVisibleIds.every((pId) => vSelectedIds.has(pId));

    if (cAllSelected) {
      for (const cId of cVisibleIds) vSelectedIds.delete(cId);
    } else {
      for (const cId of cVisibleIds) vSelectedIds.add(cId);
    }
  }

  async function fHandleBulkDelete() {
    const cIds = Array.from(vSelectedIds);
    if (cIds.length === 0) return;
    await cConversationsStore.mBulkDeleteConversations(cIds);
    fExitSelectionMode();
  }

  async function fHandleBulkPinToggle() {
    const cIds = Array.from(vSelectedIds);
    if (cIds.length === 0) return;
    await cConversationsStore.mBulkToggleConversationPin(cIds);
  }

  async function fHandleBulkExport() {
    const cIds = Array.from(vSelectedIds);
    if (cIds.length === 0) return;
    await cConversationsStore.mBulkExportConversations(cIds);
  }

  const cMarquee = fUseMarqueeSelection({
    selectedIds: () => vSelectedIds,
    orderedIds: () => cRenderedOrderIds,
    enabled: () => vIsSelectionMode
  });

  function fHandleRowMouseDown(pId: string, pEvent: MouseEvent) {
    if (!vIsSelectionMode) return;
    cMarquee.mRowMouseDown(pId, pEvent);
  }

  function fHandleSelectionClick(pId: string, pOptions: { shiftKey: boolean }): void {
    if (!vIsSelectionMode) return;
    cMarquee.mRowClick(pId, pOptions.shiftKey);
  }

  async function fSelectConversation(pId: string) {
    if (cIsMobile.current) {
      fScheduleMobileCollapse();
    }
    await goto(RouterService.mChat(pId));
  }

  async function fHandleEditConversation(pId: string) {
    const cConversation = fConversations().find((pConv) => pConv.id === pId);
    if (!cConversation) return;

    vRenameTargetConversationId = pId;
    vRenameOriginalTitle = cConversation.name;
    vRenameDraft = cConversation.name;
    vRenameDialogOpen = true;
  }

  async function fHandleRenameConfirm() {
    const cId = vRenameTargetConversationId;
    if (!cId) return;

    const cNextName = vRenameDraft.trim();
    if (!cNextName || cNextName === vRenameOriginalTitle.trim()) return;

    await cConversationsStore.mUpdateConversationName(cId, cNextName);

    vRenameDialogOpen = false;
    vRenameTargetConversationId = null;
  }

  function fHandleRenameCancel() {
    vRenameDialogOpen = false;
    vRenameTargetConversationId = null;
    vRenameDraft = '';
    vRenameOriginalTitle = '';
  }

  async function fHandleDeleteConversation(pId: string) {
    const cConversation = fConversations().find((pConv) => pConv.id === pId);
    if (!cConversation) return;

    const cConfirmed = window.confirm(fTranslate('text_749f8d33a6ef', { p0: cConversation.name }));
    if (!cConfirmed) return;

    await cConversationsStore.mDeleteConversation(pId, { deleteWithForks: false });
  }

  function fHandleStopGeneration(pId: string) {
    cChatStore.mStopGenerationForChat(pId);
  }

  let vInnerWidth = $state(0);
  let vPendingCollapse = $state<ReturnType<typeof setTimeout> | null>(null);

  function fScheduleMobileCollapse() {
    if (vPendingCollapse) {
      clearTimeout(vPendingCollapse);
    }
    vPendingCollapse = setTimeout(() => {
      vIsExpandedMode = false;
      vPendingCollapse = null;
    }, 100);
  }

  function fCancelMobileCollapse() {
    if (vPendingCollapse) {
      clearTimeout(vPendingCollapse);
      vPendingCollapse = null;
    }
  }
</script>

<svelte:window onkeydown={cHandleKeydown} bind:innerWidth={vInnerWidth} />

{#if vInnerWidth > 768 || (!page.url.hash.includes(cRoutes.SETTINGS) && !page.url.hash.includes(cRoutes.MONITOR) && !page.url.hash.includes(cRoutes.MODEL))}
  <aside
    class={[
      'fixed md:sticky top-2 start-2 md:start-0 md:ms-2 md:mt-2 pt-2 z-10 w-[calc(100dvw-1rem)]',
      'md:h-[calc(100dvh-1.125rem-var(--status-bar-height))]',
      vIsExpandedMode &&
        (cDevice.isStandalone
          ? 'h-[calc(100dvh-2rem-var(--status-bar-height))]'
          : cDevice.isIOSDevice
            ? 'h-[calc(100dvh-0.5rem-var(--status-bar-height))]'
            : 'h-[calc(100dvh-1rem-var(--status-bar-height))]'),
      'rounded-3xl md:rounded-2xl',
      'flex flex-col justify-between',
      'md:transition-[width,padding] duration-200 ease-out',
      cIsStripExpanded && 'md:w-72 md:bg-muted/60 md:backdrop-blur-xl border-border shadow-md',
      !cIsStripExpanded && 'md:w-12',
      vIsExpandedMode && 'is-expanded'
    ]}
  >
    <div class="px-2 flex items-center gap-2">
      <!-- the three lines expand and collapse the menu; on mobile it is the only part shown while collapsed -->
      <ActionIcon
        icon={Menu}
        size="lg"
        iconSize={cSidebarIconClass}
        class="{vIsExpandedMode
          ? 'bg-muted! md:bg-foreground/5!'
          : 'bg-transparent!'} md:h-9 md:w-9 h-10 w-10 shrink-0 rounded-full md:hover:bg-foreground/10! pointer-events-auto"
        onclick={fToggleExpandedMode}
        tooltip={vIsExpandedMode ? fTranslate('text_8896c14d09e7') : fTranslate('text_ef84dd64b90f')}
        tooltipSide={TooltipSide.RIGHT}
        ariaLabel={vIsExpandedMode
          ? fTranslate('text_8896c14d09e7')
          : fTranslate('text_4a984a4ea6d0')}
      />

      {#if vIsExpandedMode}
        <SearchInput
          class="min-w-0 flex-1"
          bind:value={vSearchQuery}
          bind:ref={vSearchInputRef}
          onKeyDown={fHandleSearchKeydown}
          placeholder={fTranslate('text_828338d56fd0')}
        />
      {/if}
    </div>

    <div
      class="mt-2 flex min-h-0 flex-1 flex-col gap-4 md:gap-1 {cIsMobile.current
        ? 'transition-[opacity,height] duration-200 ease-out'
        : ''} {cIsMobile.current && !vIsExpandedMode ? 'opacity-0 !h-0' : ''}"
      in:fade={{ duration: 200 }}
      out:fade={{ duration: 200 }}
    >
      {#if vIsSearching && vIsExpandedMode}
        <SidebarNavigationGlobalSearch
          bind:this={vGlobalSearch}
          query={vSearchQuery}
          currentChatId={vCurrentChatId}
          onNavigated={fHandleSearchNavigated}
          onEdit={fHandleEditConversation}
          onDelete={fHandleDeleteConversation}
          onStop={fHandleStopGeneration}
        />
      {:else}
        <SidebarNavigationActions
          isExpandedMode={vInnerWidth > 768 ? vIsExpandedMode : true}
          class="px-2"
          onNavigate={() => {
            if (cIsMobile.current) {
              fScheduleMobileCollapse();
            }
          }}
        />
      {/if}

      {#if !vIsSearching && !vIsExpandedMode && !cIsOnMobile}
        <!-- collapsed on desktop: the numbered bubble of every chat under the menu icons -->
        <div class="flex min-h-0 flex-1 flex-col overflow-y-auto pb-2">
          <SidebarNavigationConversationList
            class="px-1.5"
            compact
            groups={vConversationGroups}
            currentChatId={vCurrentChatId}
            onSelect={fSelectConversation}
            onEdit={fHandleEditConversation}
            onDelete={fHandleDeleteConversation}
            onStop={fHandleStopGeneration}
            visibleCount={0}
            allVisibleSelected={false}
            someVisibleSelected={false}
            allSelectedArePinned={false}
            pinStateIsMixed={false}
            onSelectAllToggle={fToggleSelectAllVisible}
            onBulkPinToggle={fHandleBulkPinToggle}
            onBulkExport={fHandleBulkExport}
            onBulkDelete={fHandleBulkDelete}
            onCloseSelection={fExitSelectionMode}
          />
        </div>
      {:else if !vIsSearching}
        <div class="flex min-h-0 flex-1 flex-col overflow-y-auto">
          <SidebarNavigationConversationList
            class="px-2"
            groups={vConversationGroups}
            currentChatId={vCurrentChatId}
            isSelectionMode={vIsSelectionMode}
            selectedIds={vSelectedIds}
            onSelect={fSelectConversation}
            onEdit={fHandleEditConversation}
            onDelete={fHandleDeleteConversation}
            onStop={fHandleStopGeneration}
            onToggleSelect={fToggleSelected}
            onEnterSelectionMode={fEnterSelectionMode}
            onSelectionClick={fHandleSelectionClick}
            onRowMouseDown={fHandleRowMouseDown}
            visibleCount={cVisibleSelectionStats.visibleCount}
            allVisibleSelected={cVisibleSelectionStats.visibleCount > 0 &&
              cVisibleSelectionStats.selectedVisibleCount === cVisibleSelectionStats.visibleCount}
            someVisibleSelected={cVisibleSelectionStats.selectedVisibleCount > 0 &&
              cVisibleSelectionStats.selectedVisibleCount < cVisibleSelectionStats.visibleCount}
            allSelectedArePinned={cAllSelectedArePinned}
            pinStateIsMixed={cPinStateIsMixed}
            onSelectAllToggle={fToggleSelectAllVisible}
            onBulkPinToggle={fHandleBulkPinToggle}
            onBulkExport={fHandleBulkExport}
            onBulkDelete={fHandleBulkDelete}
            onCloseSelection={fExitSelectionMode}
          />
        </div>
      {/if}
    </div>
  </aside>
{/if}

<DialogConversationRename
  bind:open={vRenameDialogOpen}
  currentTitle={vRenameOriginalTitle}
  bind:value={vRenameDraft}
  onConfirm={fHandleRenameConfirm}
  onCancel={fHandleRenameCancel}
/>

<style>
  aside {
    @media (max-width: 768px) {
      --size: 1.125rem;
    }
  }

  @media (max-width: 768px) {
    aside {
      &:not(.is-expanded) {
        pointer-events: none;
      }
    }

    aside.is-expanded::before {
      content: '';
      position: fixed;
      top: -0.5rem;
      bottom: -0.25rem;
      left: -0.5rem;
      right: -0.5rem;
      z-index: -1;
      background: var(--background);
      backdrop-filter: blur(1rem);
      pointer-events: none;
    }
  }
</style>
