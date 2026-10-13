<script lang="ts">
  import { fTranslate } from '../../../../i18n';

  import { X, AlertTriangle } from '@lucide/svelte';
  import { Button } from '$lib/components/ui/button';
  import { Switch } from '$lib/components/ui/switch';
  import { ChatForm, DialogConfirmation } from '$lib/components/app';
  import { getMessageEditContext } from '$lib/contexts';
  import { KeyboardKey, MessageRole } from '$lib/enums';
  import { cChatStore } from '$lib/stores/chat.svelte';
  import { processFilesToChatUploaded } from '$lib/utils/browser-only';

  const cEditCtx = getMessageEditContext();

  let vSaveWithoutRegenerate = $state(false);
  let vShowDiscardDialog = $state(false);
  let vBranchAfterEdit = $state(false);

  let vIsUserMessage = $derived(cEditCtx.messageRole === MessageRole.USER);
  let vIsAssistantMessage = $derived(cEditCtx.messageRole === MessageRole.ASSISTANT);

  let vHasUnsavedChanges = $derived.by(() => {
    if (cEditCtx.editedContent !== cEditCtx.originalContent) return true;
    if (cEditCtx.editedUploadedFiles.length > 0) return true;

    const cExtrasChanged =
      cEditCtx.editedExtras.length !== cEditCtx.originalExtras.length ||
      cEditCtx.editedExtras.some((pExtra, pI) => pExtra !== cEditCtx.originalExtras[pI]);

    if (cExtrasChanged) return true;

    return false;
  });

  let vHasAttachments = $derived(
    (cEditCtx.editedExtras && cEditCtx.editedExtras.length > 0) ||
      (cEditCtx.editedUploadedFiles && cEditCtx.editedUploadedFiles.length > 0)
  );

  let vCanSubmit = $derived(cEditCtx.editedContent.trim().length > 0 || vHasAttachments);

  function fHandleGlobalKeydown(pEvent: KeyboardEvent) {
    if (pEvent.key === KeyboardKey.ESCAPE) {
      pEvent.preventDefault();
      fAttemptCancel();
    }
  }

  function fAttemptCancel() {
    if (vHasUnsavedChanges) {
      vShowDiscardDialog = true;
    } else {
      cEditCtx.cancel();
    }
  }

  function fHandleSubmit() {
    if (!vCanSubmit) return;

    if (vIsUserMessage && vSaveWithoutRegenerate && cEditCtx.showSaveOnlyOption) {
      cEditCtx.saveOnly();
    } else {
      if (vIsAssistantMessage && cEditCtx.setShouldBranchAfterEdit) {
        cEditCtx.setShouldBranchAfterEdit(vBranchAfterEdit);
      }

      cEditCtx.save();
    }

    vSaveWithoutRegenerate = false;
    vBranchAfterEdit = false;
  }

  function fHandleAttachmentRemove(pIndex: number) {
    const aNewExtras = [...cEditCtx.editedExtras];
    aNewExtras.splice(pIndex, 1);
    cEditCtx.setExtras(aNewExtras);
  }

  function fHandleUploadedFileRemove(pFileId: string) {
    const cNewFiles = cEditCtx.editedUploadedFiles.filter((pF) => pF.id !== pFileId);
    cEditCtx.setUploadedFiles(cNewFiles);
  }

  async function fHandleFilesAdd(pFiles: File[]) {
    const cProcessed = await processFilesToChatUploaded(pFiles);
    cEditCtx.setUploadedFiles([...cEditCtx.editedUploadedFiles, ...cProcessed]);
  }

  $effect(() => {
    cChatStore.mSetEditModeActive(fHandleFilesAdd);

    return () => {
      cChatStore.mClearEditMode();
    };
  });
</script>

<svelte:window onkeydown={fHandleGlobalKeydown} />

<div class="relative w-full max-w-[80%]">
  <ChatForm
    value={cEditCtx.editedContent}
    attachments={cEditCtx.editedExtras}
    bind:uploadedFiles={cEditCtx.editedUploadedFiles}
    placeholder={fTranslate('text_0bc9ade7fa33')}
    showMcpPromptButton
    showAddButton={cEditCtx.messageRole === MessageRole.USER}
    showModelSelector={cEditCtx.messageRole === MessageRole.USER}
    onValueChange={cEditCtx.setContent}
    onAttachmentRemove={fHandleAttachmentRemove}
    onUploadedFileRemove={fHandleUploadedFileRemove}
    onFilesAdd={fHandleFilesAdd}
    onSubmit={fHandleSubmit}
  />
</div>

<div class="mt-2 flex w-full max-w-[80%] items-center justify-between">
  {#if vIsUserMessage && cEditCtx.showSaveOnlyOption}
    <div class="flex items-center gap-2">
      <Switch id="save-only-switch" bind:checked={vSaveWithoutRegenerate} class="scale-75" />

      <label for="save-only-switch" class="cursor-pointer text-xs text-muted-foreground">
        {fTranslate('text_d6ba6ecbbf7a')}
      </label>
    </div>
  {:else if vIsAssistantMessage}
    <div class="flex items-center gap-2">
      <Switch id="branch-after-edit" bind:checked={vBranchAfterEdit} class="scale-75" />

      <label for="branch-after-edit" class="cursor-pointer text-xs text-muted-foreground">
        {fTranslate('text_cc2c478ba8cd')}
      </label>
    </div>
  {:else}
    <div></div>
  {/if}

  <Button class="h-7 px-3 text-xs" onclick={fAttemptCancel} size="sm" variant="ghost">
    <X class="mr-1 h-3 w-3" />
    {fTranslate('text_19766ed6ccb2')}
  </Button>
</div>

<DialogConfirmation
  bind:open={vShowDiscardDialog}
  title={fTranslate('text_85bcf41666a5')}
  description={fTranslate('text_e8515d324597')}
  confirmText={fTranslate('text_eb1a70e39274')}
  cancelText={fTranslate('text_e76fd2add010')}
  variant="destructive"
  icon={AlertTriangle}
  onConfirm={cEditCtx.cancel}
  onCancel={() => (vShowDiscardDialog = false)}
/>
