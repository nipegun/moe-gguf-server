<script lang="ts">
  import { fTranslate } from '../../../i18n';

  import { Button } from '$lib/components/ui/button';
  import * as AlertDialog from '$lib/components/ui/alert-dialog';
  import { cSettingsStore } from '$lib/stores/settings.svelte';
  import { RotateCcw } from '@lucide/svelte';

  interface Props {
    onReset?: () => void;
    onSave?: () => void;
  }

  let { onReset: vOnReset, onSave: vOnSave }: Props = $props();

  let vShowResetDialog = $state(false);

  function fHandleResetClick() {
    vShowResetDialog = true;
  }

  function fHandleConfirmReset() {
    cSettingsStore.mForceSyncWithServerDefaults();
    vOnReset?.();

    vShowResetDialog = false;
  }

  function fHandleSave() {
    vOnSave?.();
  }
</script>

<!-- the buttons wrap on narrow screens: their labels never break, and the Spanish ones need ~430px in a row -->
<div
  class="sticky bottom-(--status-bar-height) mx-auto mt-4 flex w-full flex-wrap justify-between gap-2 p-4 md:p-6"
>
  <div class="flex min-w-0 gap-2">
    <Button variant="outline" class="h-auto min-h-9 whitespace-normal" onclick={fHandleResetClick}>
      <RotateCcw class="h-3 w-3" />
      {fTranslate('text_bc5b45ae7b60')}
    </Button>
  </div>

  <Button onclick={fHandleSave}>{fTranslate('text_7f3a3b142814')}</Button>
</div>

<AlertDialog.Root bind:open={vShowResetDialog}>
  <AlertDialog.Content>
    <AlertDialog.Header>
      <AlertDialog.Title>{fTranslate('text_7047e493541d')}</AlertDialog.Title>
      <AlertDialog.Description>{fTranslate('text_395d23b1d04b')}</AlertDialog.Description>
    </AlertDialog.Header>
    <AlertDialog.Footer>
      <AlertDialog.Cancel>{fTranslate('text_19766ed6ccb2')}</AlertDialog.Cancel>
      <AlertDialog.Action onclick={fHandleConfirmReset}
        >{fTranslate('text_c0334bae1053')}</AlertDialog.Action
      >
    </AlertDialog.Footer>
  </AlertDialog.Content>
</AlertDialog.Root>
