<script lang="ts">
  import { fTranslate } from '../../../i18n';

  import { cIconClassDefault } from '$lib/constants/css-classes';
  import { FolderOpen, Plus, Loader2, Braces } from '@lucide/svelte';
  import { toast } from 'svelte-sonner';
  import * as Dialog from '$lib/components/ui/dialog';
  import { Button } from '$lib/components/ui/button';
  import { cMcpStore } from '$lib/stores/mcp.svelte';
  import { cConversationsStore } from '$lib/stores/conversations.svelte';
  import {
    fMcpResources,
    fMcpTotalResourceCount,
    cMcpResourceStore
  } from '$lib/stores/mcp-resources.svelte';
  import {
    McpResourcesBrowser,
    McpResourcePreview,
    McpResourceTemplateForm
  } from '$lib/components/app';
  import { getResourceDisplayName } from '$lib/utils';
  import type { MCPResourceInfo, MCPResourceContent, MCPResourceTemplateInfo } from '$lib/types';
  import { SvelteSet } from 'svelte/reactivity';

  interface Props {
    open?: boolean;
    onOpenChange?: (pOpen: boolean) => void;
    onAttach?: (pResource: MCPResourceInfo) => void;
    preSelectedUri?: string;
  }

  let {
    open: vOpen = $bindable(false),
    onOpenChange: vOnOpenChange,
    onAttach: vOnAttach,
    preSelectedUri: vPreSelectedUri
  }: Props = $props();

  let vSelectedResources = new SvelteSet<string>();
  let vLastSelectedUri = $state<string | null>(null);
  let vIsAttaching = $state(false);

  let vSelectedTemplate = $state<MCPResourceTemplateInfo | null>(null);
  let vTemplatePreviewUri = $state<string | null>(null);
  let vTemplatePreviewContent = $state<MCPResourceContent[] | null>(null);
  let vTemplatePreviewLoading = $state(false);
  let vTemplatePreviewError = $state<string | null>(null);

  const cTotalCount = $derived(fMcpTotalResourceCount());

  $effect(() => {
    if (vOpen) {
      fLoadResources();

      if (vPreSelectedUri) {
        vSelectedResources.clear();
        vSelectedResources.add(vPreSelectedUri);
        vLastSelectedUri = vPreSelectedUri;
      }
    }
  });

  async function fLoadResources() {
    const cPerChatOverrides = cConversationsStore.mGetAllMcpServerOverrides();
    const cInitialized = await cMcpStore.mEnsureInitialized(cPerChatOverrides);

    if (cInitialized) {
      await cMcpStore.mFetchAllResources();
    }
  }

  function fHandleOpenChange(pNewOpen: boolean) {
    vOpen = pNewOpen;
    vOnOpenChange?.(pNewOpen);

    if (!pNewOpen) {
      vSelectedResources.clear();
      vLastSelectedUri = null;
      fClearTemplateState();
    }
  }

  function fClearTemplateState() {
    vSelectedTemplate = null;
    vTemplatePreviewUri = null;
    vTemplatePreviewContent = null;
    vTemplatePreviewLoading = false;
    vTemplatePreviewError = null;
  }

  function fHandleTemplateSelect(pTemplate: MCPResourceTemplateInfo) {
    vSelectedResources.clear();
    vLastSelectedUri = null;

    if (
      vSelectedTemplate?.uriTemplate === pTemplate.uriTemplate &&
      vSelectedTemplate?.serverName === pTemplate.serverName
    ) {
      fClearTemplateState();

      return;
    }

    vSelectedTemplate = pTemplate;
    vTemplatePreviewUri = null;
    vTemplatePreviewContent = null;
    vTemplatePreviewLoading = false;
    vTemplatePreviewError = null;
  }

  async function fHandleTemplateResolve(pUri: string, pServerName: string) {
    vTemplatePreviewUri = pUri;
    vTemplatePreviewContent = null;
    vTemplatePreviewLoading = true;
    vTemplatePreviewError = null;

    try {
      const cContent = await cMcpStore.mReadResourceByUri(pServerName, pUri);

      if (cContent) {
        vTemplatePreviewContent = cContent;
      } else {
        vTemplatePreviewError = fTranslate('text_61847ecebbd0');
      }
    } catch (vError) {
      vTemplatePreviewError =
        vError instanceof Error ? vError.message : fTranslate('text_27c2ccd962c2');
    } finally {
      vTemplatePreviewLoading = false;
    }
  }

  function fHandleTemplateCancelForm() {
    fClearTemplateState();
  }

  async function fHandleAttachTemplateResource() {
    if (!vTemplatePreviewUri || !vSelectedTemplate || !vTemplatePreviewContent) return;

    vIsAttaching = true;

    try {
      const cKnownResource = cMcpResourceStore.mFindResourceByUri(vTemplatePreviewUri);

      if (cKnownResource) {
        if (!cMcpResourceStore.mIsAttached(cKnownResource.uri)) {
          await cMcpStore.mAttachResource(cKnownResource.uri);
        }

        toast.success(
          fTranslate('text_8490588970e0', { p0: cKnownResource.title || cKnownResource.name })
        );
      } else {
        if (cMcpResourceStore.mIsAttached(vTemplatePreviewUri)) {
          toast.info(fTranslate('text_1aa9ca580c64'));
          fHandleOpenChange(false);
          return;
        }

        const dResourceInfo: MCPResourceInfo = {
          uri: vTemplatePreviewUri,
          name: vTemplatePreviewUri.split('/').pop() || vTemplatePreviewUri,
          serverName: vSelectedTemplate.serverName
        };

        const cAttachment = cMcpResourceStore.mAddAttachment(dResourceInfo);
        cMcpResourceStore.mUpdateAttachmentContent(cAttachment.id, vTemplatePreviewContent);

        toast.success(fTranslate('text_8490588970e0', { p0: dResourceInfo.name }));
      }

      fHandleOpenChange(false);
    } catch (vError) {
      console.error('Failed to attach template resource:', vError);
    } finally {
      vIsAttaching = false;
    }
  }

  function fHandleResourceSelect(pResource: MCPResourceInfo, pShiftKey: boolean = false) {
    fClearTemplateState();

    if (pShiftKey && vLastSelectedUri) {
      const cAllResources = fGetAllResourcesFlatInTreeOrder();
      const cLastIndex = cAllResources.findIndex((pR) => pR.uri === vLastSelectedUri);
      const cCurrentIndex = cAllResources.findIndex((pR) => pR.uri === pResource.uri);

      if (cLastIndex !== -1 && cCurrentIndex !== -1) {
        const cStart = Math.min(cLastIndex, cCurrentIndex);
        const cEnd = Math.max(cLastIndex, cCurrentIndex);

        for (let vI = cStart; vI <= cEnd; vI++) {
          vSelectedResources.add(cAllResources[vI].uri);
        }
      }
    } else {
      vSelectedResources.clear();
      vSelectedResources.add(pResource.uri);
      vLastSelectedUri = pResource.uri;
    }
  }

  function fHandleResourceToggle(pResource: MCPResourceInfo, pChecked: boolean) {
    fClearTemplateState();

    if (pChecked) {
      vSelectedResources.add(pResource.uri);
    } else {
      vSelectedResources.delete(pResource.uri);
    }

    vLastSelectedUri = pResource.uri;
  }

  function fGetAllResourcesFlatInTreeOrder(): MCPResourceInfo[] {
    const aAllResources: MCPResourceInfo[] = [];
    const cResourcesMap = fMcpResources();

    for (const [cServerName, cServerRes] of cResourcesMap.entries()) {
      for (const cResource of cServerRes.resources) {
        aAllResources.push({ ...cResource, serverName: cServerName });
      }
    }

    return aAllResources.sort((pA, pB) => {
      const aName = getResourceDisplayName(pA);
      const cBName = getResourceDisplayName(pB);
      return aName.localeCompare(cBName);
    });
  }

  async function fHandleAttach() {
    if (vSelectedResources.size === 0) return;

    vIsAttaching = true;

    try {
      const cAllResources = fGetAllResourcesFlatInTreeOrder();
      const cResourcesToAttach = cAllResources.filter((pR) => vSelectedResources.has(pR.uri));

      for (const cResource of cResourcesToAttach) {
        await cMcpStore.mAttachResource(cResource.uri);
        vOnAttach?.(cResource);
      }

      const cCount = cResourcesToAttach.length;

      toast.success(
        cCount === 1
          ? fTranslate('text_8490588970e0', { p0: cResourcesToAttach[0].name })
          : `${cCount} resources attached`
      );

      fHandleOpenChange(false);
    } catch (vError) {
      console.error('Failed to attach resources:', vError);
    } finally {
      vIsAttaching = false;
    }
  }

  const cSelectedTemplateUri = $derived(vSelectedTemplate?.uriTemplate ?? null);

  const cHasTemplateResult = $derived(
    !!vSelectedTemplate && !!vTemplatePreviewContent && !!vTemplatePreviewUri
  );
</script>

<Dialog.Root open={vOpen} onOpenChange={fHandleOpenChange}>
  <Dialog.Content class="max-h-[80vh] !max-w-4xl overflow-hidden p-0">
    <Dialog.Header class="border-b border-border/30 px-6 py-4">
      <Dialog.Title class="flex items-center gap-2">
        <FolderOpen class="h-5 w-5" />

        <span>{fTranslate('text_77a814db2847')}</span>

        {#if cTotalCount > 0}
          <span class="text-sm font-normal text-muted-foreground">({cTotalCount})</span>
        {/if}
      </Dialog.Title>

      <Dialog.Description>{fTranslate('text_e49707e6f350')}</Dialog.Description>
    </Dialog.Header>

    <div class="flex h-[500px] min-w-0">
      <div class="w-72 shrink-0 overflow-y-auto border-r border-border/30 p-4">
        <McpResourcesBrowser
          onSelect={fHandleResourceSelect}
          onToggle={fHandleResourceToggle}
          onTemplateSelect={fHandleTemplateSelect}
          selectedUris={vSelectedResources}
          selectedTemplateUri={cSelectedTemplateUri}
          expandToUri={vPreSelectedUri}
        />
      </div>

      <div class="min-w-0 flex-1 overflow-auto p-4">
        {#if vSelectedTemplate && !vTemplatePreviewContent}
          <div class="flex h-full flex-col">
            <div class="mb-3 flex items-center gap-2">
              <Braces class="{cIconClassDefault} text-muted-foreground" />

              <span class="text-sm font-medium">
                {vSelectedTemplate.title || vSelectedTemplate.name}
              </span>
            </div>

            {#if vSelectedTemplate.description}
              <p class="mb-4 text-xs text-muted-foreground">
                {vSelectedTemplate.description}
              </p>
            {/if}

            <div class="mb-4 rounded-md border border-border/50 bg-muted/30 px-3 py-2">
              <p class="font-mono text-xs break-all text-muted-foreground">
                {vSelectedTemplate.uriTemplate}
              </p>
            </div>

            {#if vTemplatePreviewLoading}
              <div class="flex flex-1 items-center justify-center">
                <Loader2 class="h-6 w-6 animate-spin text-muted-foreground" />
              </div>
            {:else if vTemplatePreviewError}
              <div class="flex flex-1 flex-col items-center justify-center gap-2 text-red-500">
                <span class="text-sm">{vTemplatePreviewError}</span>

                <Button
                  size="sm"
                  variant="outline"
                  onclick={() => {
                    vTemplatePreviewError = null;
                  }}
                >
                  {fTranslate('text_d8b8392e2c54')}
                </Button>
              </div>
            {:else}
              <McpResourceTemplateForm
                template={vSelectedTemplate}
                onResolve={fHandleTemplateResolve}
                onCancel={fHandleTemplateCancelForm}
              />
            {/if}
          </div>
        {:else if cHasTemplateResult}
          <!-- Template resolved: show preview -->
          <McpResourcePreview
            resource={{
              uri: vTemplatePreviewUri ?? '',
              name: vTemplatePreviewUri?.split('/').pop() || (vTemplatePreviewUri ?? ''),
              serverName: vSelectedTemplate?.serverName || ''
            }}
            preloadedContent={vTemplatePreviewContent}
          />
        {:else if vSelectedResources.size === 1}
          {@const allResources = fGetAllResourcesFlatInTreeOrder()}
          {@const selectedResource = allResources.find((r) => vSelectedResources.has(r.uri))}

          <McpResourcePreview resource={selectedResource ?? null} />
        {:else if vSelectedResources.size > 1}
          <div class="flex flex-col gap-10">
            {#each fGetAllResourcesFlatInTreeOrder() as resource (resource.uri)}
              {#if vSelectedResources.has(resource.uri)}
                <McpResourcePreview {resource} />
              {/if}
            {/each}
          </div>
        {:else}
          <div class="flex h-full items-center justify-center text-sm text-muted-foreground">
            {fTranslate('text_a09ca1a10c95')}
          </div>
        {/if}
      </div>
    </div>

    <Dialog.Footer class="border-t border-border/30 px-6 py-4">
      <Button variant="outline" onclick={() => fHandleOpenChange(false)}
        >{fTranslate('text_19766ed6ccb2')}</Button
      >

      {#if cHasTemplateResult}
        <Button onclick={fHandleAttachTemplateResource} disabled={vIsAttaching}>
          {#if vIsAttaching}
            <Loader2 class="mr-2 {cIconClassDefault} animate-spin" />
          {:else}
            <Plus class="mr-2 {cIconClassDefault}" />
          {/if}
          {fTranslate('text_1a3db5f016d3')}
        </Button>
      {:else}
        <Button onclick={fHandleAttach} disabled={vSelectedResources.size === 0 || vIsAttaching}>
          {#if vIsAttaching}
            <Loader2 class="mr-2 {cIconClassDefault} animate-spin" />
          {:else}
            <Plus class="mr-2 {cIconClassDefault}" />
          {/if}
          {fTranslate('text_d406ade2958c')}
          {vSelectedResources.size > 0
            ? `(${vSelectedResources.size})`
            : fTranslate('text_eb7a842ff958')}
        </Button>
      {/if}
    </Dialog.Footer>
  </Dialog.Content>
</Dialog.Root>
