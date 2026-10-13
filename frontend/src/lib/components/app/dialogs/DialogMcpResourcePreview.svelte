<script lang="ts">
  import { fTranslate } from '../../../i18n';

  import * as Dialog from '$lib/components/ui/dialog';
  import { Download } from '@lucide/svelte';
  import { Button } from '$lib/components/ui/button';
  import { cMcpStore } from '$lib/stores/mcp.svelte';
  import { SyntaxHighlightedCode, ActionIconCopyToClipboard } from '$lib/components/app';
  import {
    getLanguageFromFilename,
    isCodeResource,
    isImageResource,
    downloadResourceContent
  } from '$lib/utils';
  import { MimeTypeIncludes, MimeTypeText } from '$lib/enums';
  import { cDefaultResourceFilename } from '$lib/constants';
  import type { DatabaseMessageExtraMcpResource } from '$lib/types';

  interface Props {
    open: boolean;
    onOpenChange?: (pOpen: boolean) => void;
    extra: DatabaseMessageExtraMcpResource;
  }

  let { open: vOpen = $bindable(), onOpenChange: vOnOpenChange, extra: vExtra }: Props = $props();

  const cServerName = $derived(cMcpStore.mGetServerDisplayName(vExtra.serverName));
  const cFavicon = $derived(cMcpStore.mGetServerFavicon(vExtra.serverName));

  function fGetLanguage(): string {
    if (vExtra.mimeType?.includes(MimeTypeIncludes.JSON)) return MimeTypeIncludes.JSON;
    if (vExtra.mimeType?.includes(MimeTypeIncludes.JAVASCRIPT)) return MimeTypeIncludes.JAVASCRIPT;
    if (vExtra.mimeType?.includes(MimeTypeIncludes.TYPESCRIPT)) return MimeTypeIncludes.TYPESCRIPT;

    const cName = vExtra.name || vExtra.uri || '';

    return getLanguageFromFilename(cName) || 'plaintext';
  }

  function fHandleDownload() {
    if (!vExtra.content) return;

    downloadResourceContent(
      vExtra.content,
      vExtra.mimeType || MimeTypeText.PLAIN,
      vExtra.name || cDefaultResourceFilename
    );
  }
</script>

<Dialog.Root bind:open={vOpen} onOpenChange={vOnOpenChange}>
  <Dialog.Content class="grid max-h-[90vh] max-w-5xl overflow-hidden sm:w-auto sm:max-w-6xl">
    <Dialog.Header>
      <Dialog.Title class="pr-8">{vExtra.name}</Dialog.Title>

      <Dialog.Description>
        <div class="flex items-center gap-2">
          <span class="text-xs text-muted-foreground">{vExtra.uri}</span>

          {#if cServerName}
            <span class="flex items-center gap-1 text-xs text-muted-foreground">
              ·
              {#if cFavicon}
                <img
                  src={cFavicon}
                  alt=""
                  class="h-3 w-3 shrink-0 rounded-sm"
                  onerror={(e) => {
                    (e.currentTarget as HTMLImageElement).style.display = 'none';
                  }}
                />
              {/if}
              {cServerName}
            </span>
          {/if}

          {#if vExtra.mimeType}
            <span class="rounded bg-muted px-1.5 py-0.5 text-xs">{vExtra.mimeType}</span>
          {/if}
        </div>
      </Dialog.Description>
    </Dialog.Header>

    <div class="flex items-center justify-end gap-1">
      <ActionIconCopyToClipboard
        text={vExtra.content}
        canCopy={!!vExtra.content}
        ariaLabel="Copy content"
      />

      <Button
        variant="ghost"
        size="sm"
        class="h-7 w-7 p-0"
        onclick={fHandleDownload}
        disabled={!vExtra.content}
        title={fTranslate('text_6eca010826ea')}
      >
        <Download class="h-3.5 w-3.5" />
      </Button>
    </div>

    <div class="overflow-auto">
      {#if isImageResource(vExtra.mimeType, vExtra.uri) && vExtra.content}
        <div class="flex items-center justify-center">
          <img
            src={vExtra.content.startsWith('data:')
              ? vExtra.content
              : `data:${vExtra.mimeType || 'image/png'};base64,${vExtra.content}`}
            alt={vExtra.name}
            class="max-h-[70vh] max-w-full rounded object-contain"
          />
        </div>
      {:else if isCodeResource(vExtra.mimeType, vExtra.uri) && vExtra.content}
        <SyntaxHighlightedCode code={vExtra.content} language={fGetLanguage()} maxHeight="70vh" />
      {:else if vExtra.content}
        <pre
          class="max-h-[70vh] overflow-auto rounded-md border bg-muted/30 p-4 font-mono text-sm wrap-anywhere whitespace-pre-wrap">{vExtra.content}</pre>
      {:else}
        <div class="py-8 text-center text-sm text-muted-foreground">
          {fTranslate('text_a7c49ff5b9e2')}
        </div>
      {/if}
    </div>
  </Dialog.Content>
</Dialog.Root>
