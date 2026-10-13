<script lang="ts">
  import { fTranslate } from '../../../../i18n';

  import { fBuildConversationTree } from '$lib/stores/conversations.svelte';
  import SidebarNavigationConversationItem from './SidebarNavigationConversationItem.svelte';

  interface Props {
    class?: string;
    searchQuery: string;
    filteredConversations: DatabaseConversation[];
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
  }

  let {
    class: vClassName = '',
    searchQuery: vSearchQuery,
    filteredConversations: vFilteredConversations,
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
    onRowMouseDown: vOnRowMouseDown
  }: Props = $props();

  let vTree = $derived(fBuildConversationTree(vFilteredConversations));

  const cHasQuery = $derived(vSearchQuery.trim().length > 0);
  const cShowHeader = $derived(cHasQuery && vFilteredConversations.length > 0);

  const cEmptyMessage = $derived(
    cHasQuery ? fTranslate('text_7d7e36058609') : fTranslate('text_87bbaa5db6b2')
  );
</script>

<div class="flex min-h-0 flex-1 flex-col gap-2 whitespace-nowrap {vClassName}">
  {#if cShowHeader}
    <div
      class="text-muted-foreground flex h-8 shrink-0 items-center rounded-md px-2 text-xs font-medium"
    >
      {fTranslate('text_e978b00de465')}
    </div>
  {/if}

  <div class="min-h-0 flex-1 overflow-y-auto">
    <ul class="flex w-full min-w-0 flex-col gap-1">
      {#each vTree as { conversation, depth } (conversation.id)}
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

      {#if vTree.length === 0}
        <li class="px-2 py-4 text-center">
          <p class="mb-4 p-4 text-sm text-muted-foreground">
            {cEmptyMessage}
          </p>
        </li>
      {/if}
    </ul>
  </div>
</div>
