<script lang="ts">
  import { fTranslate } from '../../../../../i18n';

  import { cConversationsStore } from '$lib/stores/conversations.svelte';
  import { cMcpStore } from '$lib/stores/mcp.svelte';
  import { cMcpResourceStore } from '$lib/stores/mcp-resources.svelte';
  import { KeyboardKey } from '$lib/enums';
  import type { MCPResourceInfo, MCPServerSettingsEntry } from '$lib/types';
  import { SvelteMap } from 'svelte/reactivity';
  import { FolderOpen } from '@lucide/svelte';
  import { Button } from '$lib/components/ui/button';
  import {
    ChatFormPickerPopover,
    ChatFormPickerList,
    ChatFormPickerListItem,
    ChatFormPickerItemHeader,
    ChatFormPickerListItemSkeleton
  } from '$lib/components/app/chat';

  interface Props {
    class?: string;
    isOpen?: boolean;
    searchQuery?: string;
    onClose?: () => void;
    onResourceSelect?: (pResource: MCPResourceInfo) => void;
    onBrowse?: () => void;
  }

  let {
    class: vClassName = '',
    isOpen: vIsOpen = false,
    searchQuery: vSearchQuery = '',
    onClose: vOnClose,
    onResourceSelect: vOnResourceSelect,
    onBrowse: vOnBrowse
  }: Props = $props();

  let vResources = $state<MCPResourceInfo[]>([]);
  let vIsLoading = $state(false);
  let vSelectedIndex = $state(0);
  let vInternalSearchQuery = $state('');

  let vServerSettingsMap = $derived.by(() => {
    const cServers = cMcpStore.mGetServers();
    const cMap = new SvelteMap<string, MCPServerSettingsEntry>();

    for (const cServer of cServers) {
      cMap.set(cServer.id, cServer);
    }

    return cMap;
  });

  $effect(() => {
    if (vIsOpen) {
      fLoadResources();
      vSelectedIndex = 0;
    }
  });

  $effect(() => {
    if (vFilteredResources.length > 0 && vSelectedIndex >= vFilteredResources.length) {
      vSelectedIndex = 0;
    }
  });

  async function fLoadResources() {
    vIsLoading = true;

    try {
      const cPerChatOverrides = cConversationsStore.mGetAllMcpServerOverrides();
      const cInitialized = await cMcpStore.mEnsureInitialized(cPerChatOverrides);

      if (!cInitialized) {
        vResources = [];

        return;
      }

      await cMcpStore.mFetchAllResources();
      vResources = cMcpResourceStore.mGetAllResourceInfos();
    } catch (vError) {
      console.error('[ChatFormPickerMcpResources] Failed to load resources:', vError);
      vResources = [];
    } finally {
      vIsLoading = false;
    }
  }

  function fHandleResourceClick(pResource: MCPResourceInfo) {
    cMcpStore.mAttachResource(pResource.uri);

    vOnResourceSelect?.(pResource);
    vOnClose?.();
  }

  function fIsResourceAttached(pUri: string): boolean {
    return cMcpResourceStore.mIsAttached(pUri);
  }

  export function fHandleKeydown(pEvent: KeyboardEvent): boolean {
    if (!vIsOpen) return false;

    if (pEvent.key === KeyboardKey.ESCAPE) {
      pEvent.preventDefault();
      vOnClose?.();

      return true;
    }

    if (pEvent.key === KeyboardKey.ARROW_DOWN) {
      pEvent.preventDefault();

      if (vFilteredResources.length > 0) {
        vSelectedIndex = (vSelectedIndex + 1) % vFilteredResources.length;
      }

      return true;
    }

    if (pEvent.key === KeyboardKey.ARROW_UP) {
      pEvent.preventDefault();
      if (vFilteredResources.length > 0) {
        vSelectedIndex = vSelectedIndex === 0 ? vFilteredResources.length - 1 : vSelectedIndex - 1;
      }

      return true;
    }

    if (pEvent.key === KeyboardKey.ENTER) {
      pEvent.preventDefault();
      if (vFilteredResources[vSelectedIndex]) {
        fHandleResourceClick(vFilteredResources[vSelectedIndex]);
      }

      return true;
    }

    return false;
  }

  let vFilteredResources = $derived.by(() => {
    const cSortedServers = cMcpStore.mGetServers();
    const dServerOrderMap = new Map(cSortedServers.map((pServer, pIndex) => [pServer.id, pIndex]));

    const cSortedResources = [...vResources].sort((pA, pB) => {
      const cOrderA = dServerOrderMap.get(pA.serverName) ?? Number.MAX_SAFE_INTEGER;
      const cOrderB = dServerOrderMap.get(pB.serverName) ?? Number.MAX_SAFE_INTEGER;

      return cOrderA - cOrderB;
    });

    const cQuery = (vSearchQuery || vInternalSearchQuery).toLowerCase();
    if (!cQuery) return cSortedResources;

    return cSortedResources.filter(
      (pResource) =>
        pResource.name.toLowerCase().includes(cQuery) ||
        pResource.title?.toLowerCase().includes(cQuery) ||
        pResource.description?.toLowerCase().includes(cQuery) ||
        pResource.uri.toLowerCase().includes(cQuery)
    );
  });

  let vShowSearchInput = $derived(vResources.length > 3);
</script>

<ChatFormPickerPopover
  bind:isOpen={vIsOpen}
  class={vClassName}
  srLabel="Open resource picker"
  onClose={vOnClose}
  onKeydown={fHandleKeydown}
>
  <ChatFormPickerList
    items={vFilteredResources}
    isLoading={vIsLoading}
    selectedIndex={vSelectedIndex}
    bind:searchQuery={vInternalSearchQuery}
    showSearchInput={vShowSearchInput}
    searchPlaceholder="Search resources..."
    emptyMessage={fTranslate('text_25fb17b35f9c')}
    itemKey={(resource) => resource.serverName + ':' + resource.uri}
  >
    {#snippet item(resource, index, isSelected)}
      {@const server = vServerSettingsMap.get(resource.serverName)}
      {@const serverLabel = server ? cMcpStore.mGetServerLabel(server) : resource.serverName}

      <ChatFormPickerListItem
        dataIndex={index}
        {isSelected}
        onclick={() => fHandleResourceClick(resource)}
      >
        <ChatFormPickerItemHeader
          {server}
          {serverLabel}
          title={resource.title || resource.name}
          description={resource.description}
        >
          {#snippet titleExtra()}
            {#if fIsResourceAttached(resource.uri)}
              <span
                class="inline-flex items-center rounded-full bg-primary/10 px-1.5 py-0.5 text-[10px] font-medium text-primary"
              >
                {fTranslate('text_55d9067ef2bf')}
              </span>
            {/if}
          {/snippet}

          {#snippet subtitle()}
            <p class="mt-0.5 truncate text-xs text-muted-foreground/60">
              {resource.uri}
            </p>
          {/snippet}
        </ChatFormPickerItemHeader>
      </ChatFormPickerListItem>
    {/snippet}

    {#snippet skeleton()}
      <ChatFormPickerListItemSkeleton />
    {/snippet}

    {#snippet footer()}
      {#if vOnBrowse && vResources.length > 3}
        <Button
          class="fixed right-3 bottom-[calc(0.75rem+var(--status-bar-height))]"
          type="button"
          onclick={vOnBrowse}
          variant="secondary"
          size="sm"
        >
          <FolderOpen class="h-3 w-3" />
          {fTranslate('text_723735c4aab6')}
        </Button>
      {/if}
    {/snippet}
  </ChatFormPickerList>
</ChatFormPickerPopover>
