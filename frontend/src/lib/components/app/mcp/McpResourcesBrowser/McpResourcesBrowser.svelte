<script lang="ts">
  import { cMcpStore } from '$lib/stores/mcp.svelte';
  import { fMcpResources, fMcpResourcesLoading } from '$lib/stores/mcp-resources.svelte';
  import type { MCPServerResources, MCPResourceInfo, MCPResourceTemplateInfo } from '$lib/types';
  import { SvelteMap, SvelteSet } from 'svelte/reactivity';
  import { fParseResourcePath } from '$lib/utils';
  import McpResourcesBrowserHeader from './McpResourcesBrowserHeader.svelte';
  import McpResourcesBrowserEmptyState from './McpResourcesBrowserEmptyState.svelte';
  import McpResourcesBrowserServerItem from './McpResourcesBrowserServerItem.svelte';

  interface Props {
    onSelect?: (pResource: MCPResourceInfo, pShiftKey?: boolean) => void;
    onToggle?: (pResource: MCPResourceInfo, pChecked: boolean) => void;
    onTemplateSelect?: (pTemplate: MCPResourceTemplateInfo) => void;
    /** Keys (fMcpResourceKey) of the selected resources */
    selectedKeys?: Set<string>;
    selectedTemplateUri?: string | null;
    expandToResource?: MCPResourceInfo;
    class?: string;
  }

  let {
    onSelect: vOnSelect,
    onToggle: vOnToggle,
    onTemplateSelect: vOnTemplateSelect,
    selectedKeys: vSelectedKeys = new Set(),
    selectedTemplateUri: vSelectedTemplateUri,
    expandToResource: vExpandToResource,
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
    if (vExpandToResource && cResources.size > 0) {
      fAutoExpandToResource(vExpandToResource);
    }
  });

  // only the server that lists the resource is opened: another server may list the same URI
  function fAutoExpandToResource(pResource: MCPResourceInfo) {
    const cServerName = pResource.serverName;
    const cServerRes = cResources.get(cServerName);
    if (!cServerRes?.resources.some((pR) => pR.uri === pResource.uri)) return;

    vExpandedServers.add(cServerName);

    const cPathParts = fParseResourcePath(pResource.uri);
    if (cPathParts.length > 1) {
      let vCurrentPath = '';
      for (let vI = 0; vI < cPathParts.length - 1; vI++) {
        vCurrentPath = `${vCurrentPath}/${cPathParts[vI]}`;
        const cFolderId = `${cServerName}:${vCurrentPath}`;
        vExpandedFolders.add(cFolderId);
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
    onSearch={(pQ) => (vSearchQuery = pQ)}
    searchQuery={vSearchQuery}
  />

  <div class="flex flex-col gap-1">
    {#if cFilteredResources.size === 0}
      <McpResourcesBrowserEmptyState isLoading={cIsLoading} />
    {:else}
      {#each [...cFilteredResources.entries()] as [vServerName, vServerRes] (vServerName)}
        <McpResourcesBrowserServerItem
          serverName={vServerName as string}
          serverRes={vServerRes as MCPServerResources}
          isExpanded={vExpandedServers.has(vServerName as string)}
          selectedKeys={vSelectedKeys}
          selectedTemplateUri={vSelectedTemplateUri}
          expandedFolders={vExpandedFolders}
          onToggleServer={() => fToggleServer(vServerName as string)}
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
