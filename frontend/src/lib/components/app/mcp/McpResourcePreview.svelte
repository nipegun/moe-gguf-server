<script lang="ts">
  import { fTranslate } from '../../../i18n';

  import { cIconClassDefault } from '$lib/constants/css-classes';
  import { FileText, Loader2, AlertCircle, Download } from '@lucide/svelte';
  import { Button } from '$lib/components/ui/button';
  import { cMcpStore } from '$lib/stores/mcp.svelte';
  import {
    isImageMimeType,
    createBase64DataUrl,
    getResourceTextContent,
    getResourceBlobContent,
    downloadResourceContent
  } from '$lib/utils';
  import { MimeTypeApplication, MimeTypeText } from '$lib/enums';
  import { ActionIconCopyToClipboard } from '$lib/components/app';
  import type { MCPResourceInfo, MCPResourceContent } from '$lib/types';

  interface Props {
    resource: MCPResourceInfo | null;
    /** Pre-loaded content (e.g., from template resolution). Skips store fetch when provided. */
    preloadedContent?: MCPResourceContent[] | null;
    class?: string;
  }

  let {
    resource: vResource,
    preloadedContent: vPreloadedContent,
    class: vClassName
  }: Props = $props();

  let vContent = $state<MCPResourceContent[] | null>(null);
  let vIsLoading = $state(false);
  let vError = $state<string | null>(null);

  $effect(() => {
    if (vResource) {
      if (vPreloadedContent) {
        vContent = vPreloadedContent;
        vIsLoading = false;
        vError = null;
      } else {
        fLoadContent(vResource.uri);
      }
    } else {
      vContent = null;
      vError = null;
    }
  });

  async function fLoadContent(pUri: string) {
    vIsLoading = true;
    vError = null;

    try {
      const cResult = await cMcpStore.mReadResource(pUri);
      if (cResult) {
        vContent = cResult;
      } else {
        vError = fTranslate('text_10768df2d448');
      }
    } catch (vE) {
      vError = vE instanceof Error ? vE.message : fTranslate('text_27c2ccd962c2');
    } finally {
      vIsLoading = false;
    }
  }

  function fHandleDownload() {
    const cText = getResourceTextContent(vContent);
    if (!cText || !vResource) return;
    downloadResourceContent(
      cText,
      vResource.mimeType || MimeTypeText.PLAIN,
      vResource.name || 'resource.txt'
    );
  }
</script>

<div class={['flex flex-col gap-3', vClassName]}>
  {#if !vResource}
    <div class="flex flex-col items-center justify-center gap-2 py-8 text-muted-foreground">
      <FileText class="h-8 w-8 opacity-50" />

      <span class="text-sm">{fTranslate('text_a09ca1a10c95')}</span>
    </div>
  {:else}
    <div class="flex items-start justify-between gap-2">
      <div class="min-w-0 flex-1">
        <h3 class="truncate font-medium">{vResource.title || vResource.name}</h3>

        <p class="truncate text-xs text-muted-foreground">{vResource.uri}</p>

        {#if vResource.description}
          <p class="mt-1 text-sm text-muted-foreground">{vResource.description}</p>
        {/if}
      </div>

      <div class="flex items-center gap-1">
        <ActionIconCopyToClipboard
          text={getResourceTextContent(vContent)}
          canCopy={!vIsLoading && !!getResourceTextContent(vContent)}
          ariaLabel="Copy content"
        />

        <Button
          variant="ghost"
          size="sm"
          class="h-7 w-7 p-0"
          onclick={fHandleDownload}
          disabled={vIsLoading || !getResourceTextContent(vContent)}
          title={fTranslate('text_6eca010826ea')}
        >
          <Download class="h-3.5 w-3.5" />
        </Button>
      </div>
    </div>

    <div class="min-h-[200px] overflow-auto rounded-md border bg-muted/30 p-3 break-all">
      {#if vIsLoading}
        <div class="flex items-center justify-center py-8">
          <Loader2 class="h-6 w-6 animate-spin text-muted-foreground" />
        </div>
      {:else if vError}
        <div class="flex flex-col items-center justify-center gap-2 py-8 text-red-500">
          <AlertCircle class="h-6 w-6" />

          <span class="text-sm">{vError}</span>
        </div>
      {:else if vContent}
        {@const textContent = getResourceTextContent(vContent)}
        {@const blobContent = getResourceBlobContent(vContent)}

        {#if textContent}
          <pre class="font-mono text-xs wrap-anywhere whitespace-pre-wrap">{textContent}</pre>
        {/if}

        {#each blobContent as blob (blob.uri)}
          {#if isImageMimeType(blob.mimeType ?? MimeTypeApplication.OCTET_STREAM)}
            <img
              src={createBase64DataUrl(
                blob.mimeType ?? MimeTypeApplication.OCTET_STREAM,
                blob.blob
              )}
              alt={fTranslate('text_5c80d1708bf1')}
              class="max-w-full rounded"
            />
          {:else}
            <div class="flex items-center gap-2 rounded bg-muted p-2 text-sm text-muted-foreground">
              <FileText class={cIconClassDefault} />

              <span>{fTranslate('text_78aee1d526dd')}{blob.mimeType || 'unknown type'})</span>
            </div>
          {/if}
        {/each}

        {#if !textContent && blobContent.length === 0}
          <div class="py-4 text-center text-sm text-muted-foreground">
            {fTranslate('text_a7c49ff5b9e2')}
          </div>
        {/if}
      {/if}
    </div>

    {#if vResource.mimeType || vResource.annotations}
      <div class="flex flex-wrap gap-2 text-xs text-muted-foreground">
        {#if vResource.mimeType}
          <span class="rounded bg-muted px-1.5 py-0.5">{vResource.mimeType}</span>
        {/if}

        {#if vResource.annotations?.priority !== undefined}
          <span class="rounded bg-muted px-1.5 py-0.5">
            {fTranslate('text_34574af1e694')}
            {vResource.annotations.priority}
          </span>
        {/if}

        <span class="rounded bg-muted px-1.5 py-0.5">
          {fTranslate('text_087fbe073393')}
          {vResource.serverName}
        </span>
      </div>
    {/if}
  {/if}
</div>
