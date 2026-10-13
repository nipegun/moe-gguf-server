<script lang="ts">
  import { fTranslate } from '../../../../i18n';

  import { Pin } from '@lucide/svelte';
  import { fBuildConversationTree } from '$lib/stores/conversations.svelte';
  import SidebarNavigationConversationItem from './SidebarNavigationConversationItem.svelte';
  import SidebarNavigationSearchResults from './SidebarNavigationSearchResults.svelte';
  import SidebarNavigationSelectionBar from './SidebarNavigationSelectionBar.svelte';

  interface Props {
    class: string;
    filteredConversations: DatabaseConversation[];
    currentChatId: string | undefined;
    isSearchModeActive: boolean;
    searchQuery: string;
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
    filteredConversations: vFilteredConversations,
    currentChatId: vCurrentChatId,
    isSearchModeActive: vIsSearchModeActive,
    searchQuery: vSearchQuery,
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

  let vConversationTree = $derived(fBuildConversationTree(vFilteredConversations));

  let vPinnedConversations = $derived(
    vConversationTree.filter(({ conversation: pConversation }) => pConversation.pinned)
  );

  let vUnpinnedConversations = $derived(
    vConversationTree.filter(({ conversation: pConversation }) => !pConversation.pinned)
  );

  const cRecentEmptyMessage = $derived(
    vSearchQuery.length > 0 ? fTranslate('text_7d7e36058609') : fTranslate('text_0d60084f056e')
  );
</script>

<div class="flex min-h-0 flex-1 flex-col">
  {#if vIsSearchModeActive}
    <SidebarNavigationSearchResults
      class={vClassName}
      searchQuery={vSearchQuery}
      filteredConversations={vFilteredConversations}
      currentChatId={vCurrentChatId}
      onSelect={vOnSelect}
      onEdit={vOnEdit}
      onDelete={vOnDelete}
      onStop={vOnStop}
      isSelectionMode={vIsSelectionMode}
      selectedIds={vSelectedIds}
      onToggleSelect={vOnToggleSelect}
      onEnterSelectionMode={vOnEnterSelectionMode}
      onSelectionClick={vOnSelectionClick}
      onRowMouseDown={vOnRowMouseDown}
    />
  {:else}
    {#if vPinnedConversations.length > 0}
      <div class="py-2 flex whitespace-nowrap {vClassName}">
        <div
          class="text-muted-foreground inline-flex h-8 shrink-0 items-center rounded-md px-2 text-xs font-medium gap-1"
        >
          <Pin class="h-3.5 w-3.5" />

          <span>{fTranslate('text_f20c87946555')}</span>
        </div>
      </div>

      <ul class="flex w-full min-w-0 flex-col gap-4 md:gap-1 {vClassName}">
        {#each vPinnedConversations as { conversation, depth } (conversation.id)}
          <li class="group/item relative mb-1 p-0">
            <SidebarNavigationConversationItem
              conversation={{
                id: conversation.id,
                name: conversation.name,
                lastModified: conversation.lastModified,
                currNode: conversation.currNode,
                forkedFromConversationId: conversation.forkedFromConversationId,
                pinned: conversation.pinned
              }}
              {depth}
              isActive={vCurrentChatId === conversation.id}
              isSelectionMode={vIsSelectionMode}
              isSelected={vSelectedIds.has(conversation.id)}
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

    <div class="mt-2 flex min-h-0 flex-1 flex-col gap-4 md:gap-0 whitespace-nowrap {vClassName}">
      {#if vFilteredConversations.length > 0}
        <div
          class="text-muted-foreground flex h-8 shrink-0 items-center rounded-md px-2 text-xs font-medium"
        >
          {fTranslate('text_529ef4c48988')}
        </div>
      {/if}

      <div class="min-h-0 flex-1 md:overflow-y-auto">
        <ul class="flex w-full min-w-0 flex-col gap-4 md:gap-0">
          {#each vUnpinnedConversations as { conversation, depth } (conversation.id)}
            <li class="group/item relative mb-1 p-0">
              <SidebarNavigationConversationItem
                conversation={{
                  id: conversation.id,
                  name: conversation.name,
                  lastModified: conversation.lastModified,
                  currNode: conversation.currNode,
                  forkedFromConversationId: conversation.forkedFromConversationId,
                  pinned: conversation.pinned
                }}
                {depth}
                isActive={vCurrentChatId === conversation.id}
                isSelectionMode={vIsSelectionMode}
                isSelected={vSelectedIds.has(conversation.id)}
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

          {#if vUnpinnedConversations.length === 0}
            <li class="px-2 py-4 text-center">
              <p class="mb-4 p-4 text-sm text-muted-foreground">
                {cRecentEmptyMessage}
              </p>
            </li>
          {/if}
        </ul>
      </div>
    </div>

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
  {/if}
</div>
