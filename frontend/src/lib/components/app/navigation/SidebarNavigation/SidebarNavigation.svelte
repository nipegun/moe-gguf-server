<script lang="ts">
  import { fTranslate } from '../../../../i18n';

  import { goto } from '$app/navigation';
  import { page } from '$app/state';
  import { PanelLeftClose, PanelLeftOpen, X } from '@lucide/svelte';
  import {
    ActionIcon,
    DialogConversationRename,
    Logo,
    SidebarNavigationConversationList,
    SidebarNavigationActions
  } from '$lib/components/app';
  import { cRoutes } from '$lib/constants';
  import { fade } from 'svelte/transition';
  import { SvelteSet } from 'svelte/reactivity';
  import { fUseMarqueeSelection } from '$lib/hooks/use-marquee-selection.svelte';

  import { fUseKeyboardShortcuts } from '$lib/hooks/use-keyboard-shortcuts.svelte';
  import {
    fBuildConversationTree,
    cConversationsStore,
    fConversations
  } from '$lib/stores/conversations.svelte';
  import { cChatStore } from '$lib/stores/chat.svelte';
  import { fConfig } from '$lib/stores/settings.svelte';
  import { RouterService } from '$lib/services/router.service';
  import { cIsMobile } from '$lib/stores/viewport.svelte';
  import { TooltipSide } from '$lib/enums';
  import { cDevice } from '$lib/stores/device.svelte';
  import { circIn } from 'svelte/easing';

  interface Props {
    onSearchClick?: () => void;
  }

  let { onSearchClick: vOnSearchClick = () => {} }: Props = $props();

  const { handleKeydown: cHandleKeydown } = fUseKeyboardShortcuts({
    activateSearchMode: () => vOnSearchClick(),
    toggleSidebar: () => fToggleExpandedMode()
  });

  let vIsExpandedMode = $state(false);
  let vHoveredTooltip = $state<string | null>(null);
  let vLogoHovered = $state(false);

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
      vIsSearchModeActive = false;
      vSearchQuery = '';
      if (vIsSelectionMode) fExitSelectionMode();
      fCancelMobileCollapse();
    }
  });

  $effect(() => {
    if (cIsMobile.current && page.url.hash.includes(cRoutes.SEARCH)) {
      vIsExpandedMode = false;
    }
  });

  let vCurrentChatId = $derived(page.params.id);
  let vIsSearchModeActive = $state(false);
  let vSearchQuery = $state('');

  let vFilteredConversations = $derived.by(() => {
    if (vIsSearchModeActive) {
      if (vSearchQuery.trim().length > 0) {
        return fConversations().filter((pConversation: { name: string }) =>
          pConversation.name.toLowerCase().includes(vSearchQuery.toLowerCase())
        );
      }

      return [];
    }

    return fConversations();
  });

  let vIsSelectionMode = $state(false);
  let vSelectedIds = new SvelteSet<string>();

  let vRenameDialogOpen = $state(false);
  let vRenameTargetConversationId = $state<string | null>(null);
  let vRenameDraft = $state('');
  let vRenameOriginalTitle = $state('');

  const cRenderedOrderIds = $derived(
    fBuildConversationTree(vFilteredConversations).map((pT) => pT.conversation.id)
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
    cMarquee.rowMouseDown(pId, pEvent);
  }

  function fHandleSelectionClick(pId: string, pOptions: { shiftKey: boolean }): void {
    if (!vIsSelectionMode) return;
    cMarquee.rowClick(pId, pOptions.shiftKey);
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

{#if vInnerWidth > 768 || (!page.url.hash.includes(cRoutes.SETTINGS) && !page.url.hash.includes(cRoutes.MCP_SERVERS) && !page.url.hash.includes(cRoutes.MONITOR) && !page.url.hash.includes(cRoutes.SEARCH))}
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
    <div class="px-2 flex items-center justify-between">
      <div
        role="button"
        tabindex="0"
        class="relative"
        onmouseenter={() => (vLogoHovered = true)}
        onmouseleave={() => (vLogoHovered = false)}
      >
        <ActionIcon
          icon={!vIsExpandedMode && vLogoHovered && vInnerWidth > 768 ? PanelLeftOpen : Logo}
          size="lg"
          iconSize="h-4.5 w-4.5 md:h-4 md:w-4"
          class="{vIsExpandedMode
            ? 'bg-muted! md:bg-foreground/5!'
            : 'bg-transparent!'} md:h-9 md:w-9 h-10 w-10 rounded-full md:hover:bg-foreground/10! pointer-events-auto"
          href={vIsExpandedMode ? cRoutes.START : undefined}
          onclick={vIsExpandedMode ? undefined : fToggleExpandedMode}
          tooltip={vIsExpandedMode ? undefined : fTranslate('text_ef84dd64b90f')}
          tooltipSide={TooltipSide.RIGHT}
          ariaLabel={vIsExpandedMode
            ? fTranslate('text_f88c4c5b5ef6')
            : fTranslate('text_4a984a4ea6d0')}
        />
      </div>

      {#if cIsOnMobile || (vIsExpandedMode && !cAlwaysShowOnDesktop)}
        <div
          class="flex items-center transition-all duration-150 ease-out {cIsMobile.current &&
          !vIsExpandedMode
            ? 'opacity-0 h-0!'
            : ''}"
          in:fade={{ duration: 150, easing: circIn, delay: 50 }}
          out:fade={{ duration: 100 }}
        >
          <ActionIcon
            icon={cIsMobile.current ? X : PanelLeftClose}
            size="lg"
            iconSize="h-4.5 w-4.5 md:h-4 md:w-4"
            class="backdrop-blur-none md:h-9 md:w-9 h-10 w-10 rounded-full mr-1 hover:bg-accent!"
            onclick={fToggleExpandedMode}
            tooltip={fTranslate('text_8896c14d09e7')}
            tooltipSide={TooltipSide.LEFT}
            ariaLabel="Collapse navigation"
          />
        </div>
      {/if}
    </div>

    <div
      class="mt-2 flex min-h-0 flex-1 flex-col gap-4 md:gap-1 {cIsMobile.current
        ? 'transition-[opacity,height] duration-200 ease-out'
        : ''} {cIsMobile.current && !vIsExpandedMode ? 'opacity-0 !h-0' : ''}"
      in:fade={{ duration: 200 }}
      out:fade={{ duration: 200 }}
    >
      <SidebarNavigationActions
        isExpandedMode={vInnerWidth > 768 ? vIsExpandedMode : true}
        class="px-2"
        bind:isSearchModeActive={vIsSearchModeActive}
        bind:searchQuery={vSearchQuery}
        onSearchDeactivated={() => {
          vIsSearchModeActive = false;
          vSearchQuery = '';
        }}
        onSearchClick={() => {
          vIsExpandedMode = true;
          vIsSearchModeActive = true;
        }}
        onNewChat={() => {
          if (cIsMobile.current) {
            fScheduleMobileCollapse();
          }
        }}
      />

      {#if vIsExpandedMode || cIsOnMobile}
        <div class="flex min-h-0 flex-1 flex-col overflow-y-auto">
          <SidebarNavigationConversationList
            class="px-2"
            filteredConversations={vFilteredConversations}
            currentChatId={vCurrentChatId}
            isSearchModeActive={vIsSearchModeActive}
            searchQuery={vSearchQuery}
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
