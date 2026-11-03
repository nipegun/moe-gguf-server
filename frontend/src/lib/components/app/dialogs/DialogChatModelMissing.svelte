<script lang="ts">
  import { fTranslate } from '../../../i18n';

  import { AlertTriangle } from '@lucide/svelte';
  import * as AlertDialog from '$lib/components/ui/alert-dialog';
  import { cChatStore, fModelMissingDialog } from '$lib/stores/chat.svelte';

  /**
   * Router mode: a message is sent to a chat whose model is no longer where it was. The user
   * cancels it (the form keeps its text) or sends it with the model loaded now, when there is one.
   */
  let vState = $derived(fModelMissingDialog());

  function fHandleOpenChange(pOpen: boolean) {
    if (!pOpen && vState) {
      cChatStore.mAnswerModelMissingDialog(false);
    }
  }
</script>

<AlertDialog.Root open={Boolean(vState)} onOpenChange={fHandleOpenChange}>
  <AlertDialog.Content>
    <AlertDialog.Header>
      <AlertDialog.Title class="flex items-center gap-2">
        <AlertTriangle class="h-5 w-5 text-destructive" />
        {fTranslate('text_ad048e33705c')}
      </AlertDialog.Title>

      <AlertDialog.Description class="[overflow-wrap:anywhere]">
        {#if vState?.path}
          {fTranslate('text_69aa5ab70dd6', { p0: vState.model, p1: vState.path })}
        {:else if vState}
          {fTranslate('text_107878a07647', { p0: vState.model })}
        {/if}
      </AlertDialog.Description>
    </AlertDialog.Header>

    <AlertDialog.Footer>
      <AlertDialog.Cancel onclick={() => cChatStore.mAnswerModelMissingDialog(false)}>
        {fTranslate('text_851a8dfc6ad3')}
      </AlertDialog.Cancel>

      {#if vState?.loadedModel}
        <AlertDialog.Action
          class="[overflow-wrap:anywhere]"
          onclick={() => cChatStore.mAnswerModelMissingDialog(true)}
        >
          {fTranslate('text_33e11e390e40', { p0: vState.loadedModel })}
        </AlertDialog.Action>
      {/if}
    </AlertDialog.Footer>
  </AlertDialog.Content>
</AlertDialog.Root>
