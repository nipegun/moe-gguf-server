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
  import { getDisplayName, getResourceIcon } from '$lib/utils';
  import { McpServerIdentity } from '$lib/components/app/mcp';

  interface Props {
    serverName: string;
    serverRes: MCPServerResources;
    isExpanded: boolean;
    selectedUris: Set<string>;
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
    selectedUris: vSelectedUris,
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
    return vSelectedUris.has(pResource.uri);
  }
</script>

{#snippet renderTreeNode(node: ResourceTreeNode, depth: number, parentPath: string)}
  {@const isFolder = !node.resource && node.children.size > 0}
  {@const folderId = `${vServerName}:${parentPath}/${node.name}`}
  {@const isFolderExpanded = vExpandedFolders.has(folderId)}

  {#if isFolder}
    {@const folderCount = fCountTreeResources(node)}
    <Collapsible.Root open={isFolderExpanded} onOpenChange={() => vOnToggleFolder(folderId)}>
      <Collapsible.Trigger
        class="flex w-full items-center gap-2 rounded px-2 py-1 text-sm hover:bg-muted/50"
      >
        {#if isFolderExpanded}
          <ChevronDown class="h-3 w-3" />
        {:else}
          <ChevronRight class="h-3 w-3" />
        {/if}

        <FolderOpen class="h-3.5 w-3.5 text-muted-foreground" />

        <span class="font-medium">{node.name}</span>

        <span class="text-xs text-muted-foreground">({folderCount})</span>
      </Collapsible.Trigger>

      <Collapsible.Content>
        <div class="ml-4 flex flex-col gap-0.5 border-l border-border/50 pl-2">
          {#each fSortTreeChildren( [...node.children.values()] ) as child (child.resource?.uri || `${vServerName}:${parentPath}/${node.name}/${child.name}`)}
            {@render renderTreeNode(child, depth + 1, `${parentPath}/${node.name}`)}
          {/each}
        </div>
      </Collapsible.Content>
    </Collapsible.Root>
  {:else if node.resource}
    {@const resource = node.resource}
    {@const ResourceIcon = getResourceIcon(resource.mimeType, resource.uri)}
    {@const isSelected = fIsResourceSelected(resource)}
    {@const resourceDisplayName = resource.title || getDisplayName(node.name)}

    <div class="group flex w-full items-center gap-2">
      {#if vOnToggle}
        <Checkbox
          checked={isSelected}
          onCheckedChange={(checked: boolean | 'indeterminate') =>
            fHandleCheckboxChange(resource, checked === true)}
          class={cIconClassDefault}
        />
      {/if}

      <button
        class={[
          'flex flex-1 items-center gap-2 rounded px-2 py-1 text-left text-sm transition-colors',
          'hover:bg-muted/50',
          isSelected && 'bg-muted'
        ]}
        onclick={(e: MouseEvent) => fHandleResourceClick(resource, e)}
        title={resourceDisplayName}
      >
        <ResourceIcon class="h-3.5 w-3.5 shrink-0 text-muted-foreground" />

        <span class="min-w-0 flex-1 truncate text-left">
          {resourceDisplayName}
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
          {#each fSortTreeChildren( [...cResourceTree.children.values()] ) as child (child.resource?.uri || `${vServerName}:${child.name}`)}
            {@render renderTreeNode(child, 1, '')}
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

          {#each cTemplateInfos as template (template.uriTemplate)}
            <button
              class={[
                'flex w-full items-center gap-2 rounded px-2 py-1 text-left text-sm transition-colors',
                'hover:bg-muted/50',
                vSelectedTemplateUri === template.uriTemplate && 'bg-muted'
              ]}
              onclick={() => vOnTemplateSelect(template)}
              title={template.uriTemplate}
            >
              <Braces class="h-3.5 w-3.5 shrink-0 text-muted-foreground" />

              <span class="min-w-0 flex-1 truncate text-left">
                {template.title || template.name}
              </span>
            </button>
          {/each}
        {/if}
      {/if}
    </div>
  </Collapsible.Content>
</Collapsible.Root>
