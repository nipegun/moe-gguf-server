<script lang="ts">
  import { fTranslate } from '../../../i18n';

  import * as AlertDialog from '$lib/components/ui/alert-dialog';
  import { Button } from '$lib/components/ui/button';
  import { Input } from '$lib/components/ui/input';
  import { Pencil } from '@lucide/svelte';

  interface Props {
    open: boolean;
    currentTitle: string;
    value: string;
    onConfirm: () => void;
    onCancel: () => void;
  }

  let {
    open: vOpen = $bindable(),
    currentTitle: vCurrentTitle,
    value: vValue = $bindable(''),
    onConfirm: vOnConfirm,
    onCancel: vOnCancel
  }: Props = $props();

  let vInputRef = $state<HTMLInputElement | null>(null);

  const cCanSubmit = $derived(vValue.trim().length > 0 && vValue.trim() !== vCurrentTitle.trim());

  $effect(() => {
    if (vOpen) {
      vValue = vCurrentTitle;
      queueMicrotask(() => {
        vInputRef?.focus();
        vInputRef?.select();
      });
    }
  });

  function fHandleOpenChange(pNewOpen: boolean) {
    if (!pNewOpen) {
      vOnCancel();
    }
  }

  function fHandleSubmit(pEvent: Event) {
    pEvent.preventDefault();
    if (!cCanSubmit) return;
    vValue = vValue.trim();
    vOnConfirm();
  }
</script>

<AlertDialog.Root bind:open={vOpen} onOpenChange={fHandleOpenChange}>
  <AlertDialog.Content>
    <AlertDialog.Header>
      <AlertDialog.Title class="flex items-center gap-2">
        <Pencil class="h-5 w-5" />
        {fTranslate('text_4f10b6dc1d44')}
      </AlertDialog.Title>

      <AlertDialog.Description>{fTranslate('text_715430ab8c99')}</AlertDialog.Description>
    </AlertDialog.Header>

    <form onsubmit={fHandleSubmit} class="space-y-2 pt-2 pb-4">
      <label for="conversation-rename-input" class="text-sm font-medium text-muted-foreground">
        {fTranslate('text_5b5cd3e853ae')}
      </label>

      <Input
        id="conversation-rename-input"
        bind:ref={vInputRef}
        bind:value={vValue}
        placeholder={fTranslate('text_5b5cd3e853ae')}
        maxlength={200}
        autocomplete="off"
        autocorrect="off"
        spellcheck={false}
      />
    </form>

    <AlertDialog.Footer>
      <AlertDialog.Cancel>{fTranslate('text_19766ed6ccb2')}</AlertDialog.Cancel>

      <Button type="button" onclick={fHandleSubmit} disabled={!cCanSubmit}
        >{fTranslate('text_1509f561f241')}</Button
      >
    </AlertDialog.Footer>
  </AlertDialog.Content>
</AlertDialog.Root>
