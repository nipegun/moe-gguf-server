<script lang="ts">
  import { fTranslate } from '../../../i18n';

  import { untrack } from 'svelte';
  import { ArrowUp, Folder, FolderOpen, House, Loader2 } from '@lucide/svelte';
  import * as Dialog from '$lib/components/ui/dialog';
  import { Button } from '$lib/components/ui/button';
  import { Checkbox } from '$lib/components/ui/checkbox';
  import { Input } from '$lib/components/ui/input';
  import { Label } from '$lib/components/ui/label';
  import { ModelsService } from '$lib/services/models.service';
  import type { ApiModelsFolderBrowse } from '$lib/types';

  interface Props {
    open?: boolean;
    /**
     * Receives the absolute path of the chosen folder. The dialog closes when it resolves and
     * stays open, showing the error, when it throws.
     */
    onChoose: (pPath: string) => Promise<void>;
  }

  let { open: vOpen = $bindable(false), onChoose: vOnChoose }: Props = $props();

  // folder shown, as the server lists it
  let vListing = $state<ApiModelsFolderBrowse | null>(null);
  let vPathInput = $state('');
  let vShowHidden = $state(false);
  let vLoading = $state(false);
  let vChoosing = $state(false);
  let vBrowseError = $state<string | null>(null);
  let vChooseError = $state<string | null>(null);

  // the answer of a request that a newer one has replaced is dropped
  let vRequest = 0;

  let vFolders = $derived(
    (vListing?.folders ?? []).filter((pFolder) => vShowHidden || !pFolder.hidden)
  );

  // every time it opens, the browser starts at the models folder (else at the home folder)
  $effect(() => {
    if (!vOpen) return;

    untrack(() => {
      vListing = null;
      vChooseError = null;
      void fBrowse('');
    });
  });

  async function fBrowse(pPath: string): Promise<boolean> {
    const cRequest = ++vRequest;
    vLoading = true;
    vBrowseError = null;

    try {
      const cListing = await ModelsService.mBrowseFolder(pPath);
      if (cRequest !== vRequest) return false;

      vListing = cListing;
      vPathInput = cListing.path;
      return true;
    } catch (vError) {
      // the folder shown stays, and so does a typed path, so that it can be corrected
      if (cRequest === vRequest) {
        vBrowseError = vError instanceof Error ? vError.message : String(vError);
      }
      return false;
    } finally {
      if (cRequest === vRequest) vLoading = false;
    }
  }

  async function fChoose() {
    if (vLoading || vChoosing) return;

    // a path typed without pressing Enter is opened first, so that the folder chosen is the one shown
    const cTyped = vPathInput.trim();
    if (cTyped && cTyped !== vListing?.path && !(await fBrowse(cTyped))) return;
    if (!vListing) return;

    vChoosing = true;
    vChooseError = null;

    try {
      await vOnChoose(vListing.path);
      vOpen = false;
    } catch (vError) {
      vChooseError = vError instanceof Error ? vError.message : String(vError);
    } finally {
      vChoosing = false;
    }
  }
</script>

<Dialog.Root bind:open={vOpen}>
  <Dialog.Content class="flex max-h-[90dvh] flex-col gap-3 sm:max-w-xl md:max-h-[90dvh]">
    <Dialog.Header>
      <Dialog.Title class="flex items-center gap-2">
        <FolderOpen class="h-5 w-5 shrink-0" />
        {fTranslate('text_21075c107103')}
      </Dialog.Title>

      <Dialog.Description>{fTranslate('text_ecba9f3a8d4c')}</Dialog.Description>
    </Dialog.Header>

    <form
      class="flex min-w-0 items-center gap-2"
      onsubmit={(pEvent) => {
        pEvent.preventDefault();
        if (vPathInput.trim()) void fBrowse(vPathInput.trim());
      }}
    >
      <Button
        type="button"
        size="icon"
        variant="outline"
        class="shrink-0"
        disabled={!vListing?.parent || vLoading || vChoosing}
        title={fTranslate('text_158f5a01ef8c')}
        aria-label={fTranslate('text_158f5a01ef8c')}
        onclick={() => vListing && fBrowse(vListing.parent)}
      >
        <ArrowUp class="h-4 w-4" />
      </Button>

      <Button
        type="button"
        size="icon"
        variant="outline"
        class="shrink-0"
        disabled={!vListing?.home || vLoading || vChoosing}
        title={fTranslate('text_6772da558a23')}
        aria-label={fTranslate('text_6772da558a23')}
        onclick={() => vListing && fBrowse(vListing.home)}
      >
        <House class="h-4 w-4" />
      </Button>

      <Input
        class="min-w-0 flex-1 font-mono text-xs"
        dir="ltr"
        bind:value={vPathInput}
        aria-label={fTranslate('text_98bca2fab915')}
        disabled={vChoosing}
        autocomplete="off"
        spellcheck="false"
      />
    </form>

    <div class="h-72 max-h-[40dvh] min-h-24 overflow-y-auto rounded-md border border-border/50">
      {#if !vListing}
        <div class="flex h-full items-center justify-center">
          {#if vLoading}
            <Loader2 class="h-5 w-5 animate-spin text-muted-foreground" />
          {/if}
        </div>
      {:else if vFolders.length === 0}
        <p class="px-3 py-2 text-sm text-muted-foreground">{fTranslate('text_7b6d8d675da1')}</p>
      {:else}
        <ul class="py-1">
          {#each vFolders as vFolder (vFolder.path)}
            <li>
              <button
                type="button"
                class="flex w-full min-w-0 items-center gap-2 px-3 py-1.5 text-start text-sm hover:bg-muted focus-visible:bg-muted focus-visible:outline-none disabled:opacity-50"
                disabled={vLoading || vChoosing}
                onclick={() => fBrowse(vFolder.path)}
              >
                <Folder class="h-4 w-4 shrink-0 text-muted-foreground" />
                <span class="min-w-0 truncate">{vFolder.name}</span>
              </button>
            </li>
          {/each}
        </ul>
      {/if}
    </div>

    <div class="flex flex-wrap items-center justify-between gap-x-4 gap-y-2 text-xs">
      <div class="flex items-center gap-2">
        <Checkbox id="models-folder-show-hidden" bind:checked={vShowHidden} />
        <Label for="models-folder-show-hidden" class="text-xs leading-none">
          {fTranslate('text_95dd573975e2')}
        </Label>
      </div>

      {#if vListing}
        <span class="text-muted-foreground">
          {fTranslate('text_14bdf0dcb487', { p0: vListing.gguf_files })}
        </span>
      {/if}
    </div>

    {#if vBrowseError || vChooseError}
      <div class="space-y-1 text-xs text-destructive [overflow-wrap:anywhere]">
        {#if vBrowseError}
          <p>{fTranslate('text_61d0e838bde3', { p0: vBrowseError })}</p>
        {/if}

        {#if vChooseError}
          <p>{fTranslate('text_0326680ed421', { p0: vChooseError })}</p>
        {/if}
      </div>
    {/if}

    <Dialog.Footer>
      <Button type="button" variant="outline" disabled={vChoosing} onclick={() => (vOpen = false)}>
        {fTranslate('text_19766ed6ccb2')}
      </Button>

      <Button type="button" disabled={!vListing || vLoading || vChoosing} onclick={fChoose}>
        {#if vChoosing}
          <Loader2 class="h-3.5 w-3.5 animate-spin" />
        {/if}
        {fTranslate('text_c7f937836f5d')}
      </Button>
    </Dialog.Footer>
  </Dialog.Content>
</Dialog.Root>
