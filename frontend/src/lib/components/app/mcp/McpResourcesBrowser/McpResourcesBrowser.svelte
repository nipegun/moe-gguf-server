<script lang="ts">
  import { cMcpStore } from '$lib/stores/mcp.svelte';
  import { fMcpResources, fMcpResourcesLoading } from '$lib/stores/mcp-resources.svelte';
  import type { MCPServerResources, MCPResourceInfo, MCPResourceTemplateInfo } from '$lib/types';
  import { SvelteMap, SvelteSet } from 'svelte/reactivity';
  import { parseResourcePath } from '$lib/utils';
  import McpResourcesBrowserHeader from './McpResourcesBrowserHeader.svelte';
  import McpResourcesBrowserEmptyState from './McpResourcesBrowserEmptyState.svelte';
  import McpResourcesBrowserServerItem from './McpResourcesBrowserServerItem.svelte';

  interface Props {
    onSelect?: (pResource: MCPResourceInfo, pShiftKey?: boolean) => void;
    onToggle?: (pResource: MCPResourceInfo, pChecked: boolean) => void;
    onTemplateSelect?: (pTemplate: MCPResourceTemplateInfo) => void;
    selectedUris?: Set<string>;
    selectedTemplateUri?: string | null;
    expandToUri?: string;
    class?: string;
  }

  let {
    onSelect: vOnSelect,
    onToggle: vOnToggle,
    onTemplateSelect: vOnTemplateSelect,
    selectedUris: vSelectedUris = new Set(),
    selectedTemplateUri: vSelectedTemplateUri,
    expandToUri: vExpandToUri,
    class: vClassName
  }: Props = $props();

  let vExpandedServers = new SvelteSet<string>();
  let vExpandedFolders = new SvelteSet<string>();
  let vSearchQuery = $state('');

  const cResources = $derived(fMcpResources());
  const cIsLoading = $derived(fMcpResourcesLoading());

  const cFilteredResources = $derived.by(() => {
    if (!vSearchQuery.trim()) {
      return cResources;
    }

    const cQuery = vSearchQuery.toLowerCase();
    const cFiltered = new SvelteMap();

    for (const [cServerName, cServerRes] of cResources.entries()) {
      const cFilteredResources = cServerRes.resources.filter((pR) => {
        return (
          pR.title?.toLowerCase().includes(cQuery) ||
          pR.uri.toLowerCase().includes(cQuery) ||
          cServerName.toLowerCase().includes(cQuery)
        );
      });

      const cFilteredTemplates = cServerRes.templates.filter((pT) => {
        return (
          pT.name?.toLowerCase().includes(cQuery) ||
          pT.title?.toLowerCase().includes(cQuery) ||
          pT.uriTemplate.toLowerCase().includes(cQuery) ||
          cServerName.toLowerCase().includes(cQuery)
        );
      });

      if (cFilteredResources.length > 0 || cFilteredTemplates.length > 0 || cQuery.trim()) {
        cFiltered.set(cServerName, {
          ...cServerRes,
          resources: cFilteredResources,
          templates: cFilteredTemplates
        });
      }
    }

    return cFiltered;
  });

  $effect(() => {
    if (vExpandToUri && cResources.size > 0) {
      fAutoExpandToResource(vExpandToUri);
    }
  });

  function fAutoExpandToResource(pUri: string) {
    for (const [cServerName, cServerRes] of cResources.entries()) {
      const cResource = cServerRes.resources.find((pR) => pR.uri === pUri);
      if (cResource) {
        vExpandedServers.add(cServerName);

        const cPathParts = parseResourcePath(pUri);
        if (cPathParts.length > 1) {
          let vCurrentPath = '';
          for (let vI = 0; vI < cPathParts.length - 1; vI++) {
            vCurrentPath = `${vCurrentPath}/${cPathParts[vI]}`;
            const cFolderId = `${cServerName}:${vCurrentPath}`;
            vExpandedFolders.add(cFolderId);
          }
        }
        break;
      }
    }
  }

  function fToggleServer(pServerName: string) {
    if (vExpandedServers.has(pServerName)) {
      vExpandedServers.delete(pServerName);
    } else {
      vExpandedServers.add(pServerName);
    }
  }

  function fToggleFolder(pFolderId: string) {
    if (vExpandedFolders.has(pFolderId)) {
      vExpandedFolders.delete(pFolderId);
    } else {
      vExpandedFolders.add(pFolderId);
    }
  }

  function fHandleRefresh() {
    cMcpStore.mFetchAllResources();
  }
</script>

<div class={['flex flex-col gap-2', vClassName]}>
  <McpResourcesBrowserHeader
    isLoading={cIsLoading}
    onRefresh={fHandleRefresh}
    onSearch={(q) => (vSearchQuery = q)}
    searchQuery={vSearchQuery}
  />

  <div class="flex flex-col gap-1">
    {#if cFilteredResources.size === 0}
      <McpResourcesBrowserEmptyState isLoading={cIsLoading} />
    {:else}
      {#each [...cFilteredResources.entries()] as [serverName, serverRes] (serverName)}
        <McpResourcesBrowserServerItem
          serverName={serverName as string}
          serverRes={serverRes as MCPServerResources}
          isExpanded={vExpandedServers.has(serverName as string)}
          selectedUris={vSelectedUris}
          selectedTemplateUri={vSelectedTemplateUri}
          expandedFolders={vExpandedFolders}
          onToggleServer={() => fToggleServer(serverName as string)}
          onToggleFolder={fToggleFolder}
          onSelect={vOnSelect}
          onToggle={vOnToggle}
          onTemplateSelect={vOnTemplateSelect}
          searchQuery={vSearchQuery}
        />
      {/each}
    {/if}
  </div>
</div>
