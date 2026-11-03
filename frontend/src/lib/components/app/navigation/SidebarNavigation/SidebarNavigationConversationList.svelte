<script lang="ts">
  import { fTranslate } from '../../../../i18n';

  import { Pin } from '@lucide/svelte';
  import type { SidebarConversationGroups } from '$lib/utils';
  import SidebarNavigationConversationItem from './SidebarNavigationConversationItem.svelte';
  import SidebarNavigationSelectionBar from './SidebarNavigationSelectionBar.svelte';

  interface Props {
    class: string;
    /** pinned chats, "Recent chats" and "Chats", in the order they are shown */
    groups: SidebarConversationGroups;
    /** collapsed sidebar: one numbered bubble per chat, without titles */
    compact?: boolean;
    currentChatId: string | undefined;
    isSelectionMode?: boolean;
    selectedIds?: Set<string>;
    onSelect: (pId: string) => void;
    onEdit: (pId: string) => void;
    onDelete: (pId: string) => void;
    onStop: (pId: string) => void;
    onToggleSelect?: (pId: string) => void;
    onEnterSelectionMode?: (pId: string) => void;
    onSelectionClick?: (pId: string, pOptions: { shiftKey: boolean }) => void;
    onRowMouseDown?: (pId: string, pEvent: MouseEvent) => void;
    visibleCount: number;
    allVisibleSelected: boolean;
    someVisibleSelected: boolean;
    allSelectedArePinned: boolean;
    pinStateIsMixed: boolean;
    onSelectAllToggle: () => void;
    onBulkPinToggle: () => void;
    onBulkExport: () => void;
    onBulkDelete: () => void;
    onCloseSelection: () => void;
  }

  let {
    class: vClassName,
    groups: vGroups,
    compact: vCompact = false,
    currentChatId: vCurrentChatId,
    isSelectionMode: vIsSelectionMode = false,
    selectedIds: vSelectedIds = new Set<string>(),
    onSelect: vOnSelect,
    onEdit: vOnEdit,
    onDelete: vOnDelete,
    onStop: vOnStop,
    onToggleSelect: vOnToggleSelect,
    onEnterSelectionMode: vOnEnterSelectionMode,
    onSelectionClick: vOnSelectionClick,
    onRowMouseDown: vOnRowMouseDown,
    visibleCount: vVisibleCount,
    allVisibleSelected: vAllVisibleSelected,
    someVisibleSelected: vSomeVisibleSelected,
    allSelectedArePinned: vAllSelectedArePinned,
    pinStateIsMixed: vPinStateIsMixed,
    onSelectAllToggle: vOnSelectAllToggle,
    onBulkPinToggle: vOnBulkPinToggle,
    onBulkExport: vOnBulkExport,
    onBulkDelete: vOnBulkDelete,
    onCloseSelection: vOnCloseSelection
  }: Props = $props();

  let vSections = $derived([
    { key: 'pinned', title: fTranslate('text_f20c87946555'), items: vGroups.pinned },
    { key: 'recent', title: fTranslate('text_2ccfecbca101'), items: vGroups.recent },
    { key: 'others', title: fTranslate('text_ef5b4049bb36'), items: vGroups.others }
  ]);

  let vIsEmpty = $derived(
    vGroups.pinned.length + vGroups.recent.length + vGroups.others.length === 0
  );
</script>

{#if vCompact}
  <!-- collapsed sidebar: the numbered bubbles, one group after the other -->
  <div class="flex flex-col items-center gap-1 {vClassName}">
    {#each vSections as vSection (vSection.key)}
      {#if vSection.items.length > 0}
        <div class="my-1 h-px w-6 shrink-0 bg-border" role="separator"></div>

        {#each vSection.items as vConversation (vConversation.id)}
          <SidebarNavigationConversationItem
            compact
            conversation={vConversation}
            isActive={vCurrentChatId === vConversation.id}
            onSelect={vOnSelect}
          />
        {/each}
      {/if}
    {/each}
  </div>
{:else}
  <div class="flex min-h-0 flex-1 flex-col">
    {#each vSections as vSection (vSection.key)}
      {#if vSection.items.length > 0}
        <div class="flex py-2 whitespace-nowrap {vClassName}">
          <div
            class="text-muted-foreground inline-flex h-8 shrink-0 items-center gap-1 rounded-md px-2 text-xs font-medium"
          >
            {#if vSection.key === 'pinned'}
              <Pin class="h-3.5 w-3.5" />
            {/if}

            <span>{vSection.title}</span>
          </div>
        </div>

        <ul class="flex w-full min-w-0 flex-col gap-4 md:gap-1 {vClassName}">
          {#each vSection.items as vConversation (vConversation.id)}
            <li class="group/item relative mb-1 p-0">
              <SidebarNavigationConversationItem
                conversation={vConversation}
                isActive={vCurrentChatId === vConversation.id}
                isSelectionMode={vIsSelectionMode}
                isSelected={vSelectedIds.has(vConversation.id)}
                onSelect={vOnSelect}
                onEdit={vOnEdit}
                onDelete={vOnDelete}
                onStop={vOnStop}
                onToggleSelect={vOnToggleSelect}
                onEnterSelectionMode={vOnEnterSelectionMode}
                onSelectionClick={vOnSelectionClick}
                onRowMouseDown={vOnRowMouseDown}
              />
            </li>
          {/each}
        </ul>
      {/if}
    {/each}

    {#if vIsEmpty}
      <p class="mb-4 p-4 text-center text-sm text-muted-foreground {vClassName}">
        {fTranslate('text_0d60084f056e')}
      </p>
    {/if}

    {#if vIsSelectionMode}
      <SidebarNavigationSelectionBar
        class="sticky top-0 z-10 m-2 mt-0"
        selectedCount={vSelectedIds.size}
        visibleCount={vVisibleCount}
        allVisibleSelected={vAllVisibleSelected}
        someVisibleSelected={vSomeVisibleSelected}
        someSelectedPinned={vAllSelectedArePinned}
        pinStateIsMixed={vPinStateIsMixed}
        onSelectAllToggle={vOnSelectAllToggle}
        onBulkPinToggle={vOnBulkPinToggle}
        onBulkExport={vOnBulkExport}
        onBulkDelete={vOnBulkDelete}
        onClose={vOnCloseSelection}
      />
    {/if}
  </div>
{/if}
