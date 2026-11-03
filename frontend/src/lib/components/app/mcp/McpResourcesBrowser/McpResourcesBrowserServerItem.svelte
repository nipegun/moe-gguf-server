<script lang="ts">
  import { fTranslate } from '../../../../i18n';

  import { cIconClassDefault } from '$lib/constants/css-classes';
  import { FolderOpen, ChevronDown, ChevronRight, Loader2, Braces } from '@lucide/svelte';
  import { Checkbox } from '$lib/components/ui/checkbox';
  import * as Collapsible from '$lib/components/ui/collapsible';
  import { cMcpStore } from '$lib/stores/mcp.svelte';
  import type { MCPResourceInfo, MCPResourceTemplateInfo, MCPServerResources } from '$lib/types';
  import { SvelteSet } from 'svelte/reactivity';
  import {
    type ResourceTreeNode,
    fBuildResourceTree,
    fCountTreeResources,
    fSortTreeChildren
  } from './mcp-resources-browser';
  import { fGetDisplayName, fGetResourceIcon, fMcpResourceKey } from '$lib/utils';
  import { McpServerIdentity } from '$lib/components/app/mcp';

  interface Props {
    serverName: string;
    serverRes: MCPServerResources;
    isExpanded: boolean;
    /** Keys (fMcpResourceKey) of the selected resources */
    selectedKeys: Set<string>;
    selectedTemplateUri?: string | null;
    expandedFolders: SvelteSet<string>;
    onToggleServer: () => void;
    onToggleFolder: (pFolderId: string) => void;
    onSelect?: (pResource: MCPResourceInfo, pShiftKey?: boolean) => void;
    onToggle?: (pResource: MCPResourceInfo, pChecked: boolean) => void;
    onTemplateSelect?: (pTemplate: MCPResourceTemplateInfo) => void;
    searchQuery?: string;
  }

  let {
    serverName: vServerName,
    serverRes: vServerRes,
    isExpanded: vIsExpanded,
    selectedKeys: vSelectedKeys,
    selectedTemplateUri: vSelectedTemplateUri,
    expandedFolders: vExpandedFolders,
    onToggleServer: vOnToggleServer,
    onToggleFolder: vOnToggleFolder,
    onSelect: vOnSelect,
    onToggle: vOnToggle,
    onTemplateSelect: vOnTemplateSelect,
    searchQuery: vSearchQuery = ''
  }: Props = $props();

  let vServerDisplayName = $derived(cMcpStore.mGetServerDisplayName(vServerName));
  let vServerFaviconUrl = $derived(cMcpStore.mGetServerFavicon(vServerName));

  const cHasResources = $derived(vServerRes.resources.length > 0);
  const cHasTemplates = $derived(vServerRes.templates.length > 0);
  const cHasContent = $derived(cHasResources || cHasTemplates);
  const cResourceTree = $derived(
    fBuildResourceTree(vServerRes.resources, vServerName, vSearchQuery)
  );

  const cTemplateInfos = $derived<MCPResourceTemplateInfo[]>(
    vServerRes.templates.map((pT) => ({
      uriTemplate: pT.uriTemplate,
      name: pT.name,
      title: pT.title,
      description: pT.description,
      mimeType: pT.mimeType,
      serverName: vServerName,
      annotations: pT.annotations,
      icons: pT.icons
    }))
  );

  function fHandleResourceClick(pResource: MCPResourceInfo, pEvent: MouseEvent) {
    vOnSelect?.(pResource, pEvent.shiftKey);
  }

  function fHandleCheckboxChange(pResource: MCPResourceInfo, pChecked: boolean) {
    vOnToggle?.(pResource, pChecked);
  }

  function fIsResourceSelected(pResource: MCPResourceInfo): boolean {
    return vSelectedKeys.has(fMcpResourceKey(pResource.serverName, pResource.uri));
  }
</script>

{#snippet renderTreeNode(pNode: ResourceTreeNode, pDepth: number, pParentPath: string)}
  {@const cIsFolder = !pNode.resource && pNode.children.size > 0}
  {@const cFolderId = `${vServerName}:${pParentPath}/${pNode.name}`}
  {@const cIsFolderExpanded = vExpandedFolders.has(cFolderId)}

  {#if cIsFolder}
    {@const cFolderCount = fCountTreeResources(pNode)}
    <Collapsible.Root open={cIsFolderExpanded} onOpenChange={() => vOnToggleFolder(cFolderId)}>
      <Collapsible.Trigger
        class="flex w-full items-center gap-2 rounded px-2 py-1 text-sm hover:bg-muted/50"
      >
        {#if cIsFolderExpanded}
          <ChevronDown class="h-3 w-3" />
        {:else}
          <ChevronRight class="h-3 w-3" />
        {/if}

        <FolderOpen class="h-3.5 w-3.5 text-muted-foreground" />

        <span class="font-medium">{pNode.name}</span>

        <span class="text-xs text-muted-foreground">({cFolderCount})</span>
      </Collapsible.Trigger>

      <Collapsible.Content>
        <div class="ml-4 flex flex-col gap-0.5 border-l border-border/50 pl-2">
          {#each fSortTreeChildren( [...pNode.children.values()] ) as vChild (vChild.resource?.uri || `${vServerName}:${pParentPath}/${pNode.name}/${vChild.name}`)}
            {@render renderTreeNode(vChild, pDepth + 1, `${pParentPath}/${pNode.name}`)}
          {/each}
        </div>
      </Collapsible.Content>
    </Collapsible.Root>
  {:else if pNode.resource}
    {@const cResource = pNode.resource}
    {@const ResourceIcon = fGetResourceIcon(cResource.mimeType, cResource.uri)}
    {@const cIsSelected = fIsResourceSelected(cResource)}
    {@const cResourceDisplayName = cResource.title || fGetDisplayName(pNode.name)}

    <div class="group flex w-full items-center gap-2">
      {#if vOnToggle}
        <Checkbox
          checked={cIsSelected}
          onCheckedChange={(pChecked: boolean | 'indeterminate') =>
            fHandleCheckboxChange(cResource, pChecked === true)}
          class={cIconClassDefault}
        />
      {/if}

      <button
        class={[
          'flex flex-1 items-center gap-2 rounded px-2 py-1 text-left text-sm transition-colors',
          'hover:bg-muted/50',
          cIsSelected && 'bg-muted'
        ]}
        onclick={(pE: MouseEvent) => fHandleResourceClick(cResource, pE)}
        title={cResourceDisplayName}
      >
        <ResourceIcon class="h-3.5 w-3.5 shrink-0 text-muted-foreground" />

        <span class="min-w-0 flex-1 truncate text-left">
          {cResourceDisplayName}
        </span>
      </button>
    </div>
  {/if}
{/snippet}

<Collapsible.Root open={vIsExpanded} onOpenChange={vOnToggleServer}>
  <Collapsible.Trigger
    class="flex w-full items-center gap-2 rounded px-2 py-1.5 text-sm hover:bg-muted/50"
  >
    {#if vIsExpanded}
      <ChevronDown class="h-3.5 w-3.5" />
    {:else}
      <ChevronRight class="h-3.5 w-3.5" />
    {/if}

    <span class="inline-flex flex-col items-start gap-1 text-left">
      <div class="inline-flex min-w-0 items-center gap-1.5">
        <McpServerIdentity
          displayName={vServerDisplayName}
          faviconUrl={vServerFaviconUrl}
          iconClass={cIconClassDefault}
          showVersion={false}
        />
      </div>

      <span class="text-xs text-muted-foreground">
        ({vServerRes.resources.length}
        {fTranslate('text_5de95319f174')}{vServerRes.resources.length !== 1
          ? 's'
          : ''}{#if cHasTemplates}, {vServerRes.templates.length}
          {fTranslate('text_5cde0f1298f4')}{vServerRes.templates.length !== 1 ? 's' : ''}{/if})
      </span>
    </span>

    {#if vServerRes.loading}
      <Loader2 class="ml-auto h-3 w-3 animate-spin text-muted-foreground" />
    {/if}
  </Collapsible.Trigger>

  <Collapsible.Content>
    <div class="ml-4 flex flex-col gap-0.5 border-l border-border/50 pl-2">
      {#if vServerRes.error}
        <div class="py-1 text-xs text-red-500">
          {fTranslate('text_617062906764')}
          {vServerRes.error}
        </div>
      {:else if !cHasContent}
        <div class="py-1 text-xs text-muted-foreground">{fTranslate('text_76010760c629')}</div>
      {:else}
        {#if cHasResources}
          {#each fSortTreeChildren( [...cResourceTree.children.values()] ) as vChild (vChild.resource?.uri || `${vServerName}:${vChild.name}`)}
            {@render renderTreeNode(vChild, 1, '')}
          {/each}
        {/if}

        {#if cHasTemplates && vOnTemplateSelect}
          {#if cHasResources}
            <div class="my-1 border-t border-border/30"></div>
          {/if}

          <div
            class="py-0.5 text-[11px] font-medium tracking-wide text-muted-foreground/70 uppercase"
          >
            {fTranslate('text_56b564b75c7f')}
          </div>

          {#each cTemplateInfos as vTemplate (vTemplate.uriTemplate)}
            <button
              class={[
                'flex w-full items-center gap-2 rounded px-2 py-1 text-left text-sm transition-colors',
                'hover:bg-muted/50',
                vSelectedTemplateUri === vTemplate.uriTemplate && 'bg-muted'
              ]}
              onclick={() => vOnTemplateSelect(vTemplate)}
              title={vTemplate.uriTemplate}
            >
              <Braces class="h-3.5 w-3.5 shrink-0 text-muted-foreground" />

              <span class="min-w-0 flex-1 truncate text-left">
                {vTemplate.title || vTemplate.name}
              </span>
            </button>
          {/each}
        {/if}
      {/if}
    </div>
  </Collapsible.Content>
</Collapsible.Root>
