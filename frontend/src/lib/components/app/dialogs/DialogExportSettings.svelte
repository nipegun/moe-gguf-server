<script lang="ts">
  import { fTranslate } from '../../../i18n';

  import * as AlertDialog from '$lib/components/ui/alert-dialog';
  import { Checkbox } from '$lib/components/ui/checkbox';
  import Label from '$lib/components/ui/label/label.svelte';
  import { Shield, ShieldOff } from '@lucide/svelte';

  let {
    open: vOpen = $bindable(),
    includeSensitiveData: vIncludeSensitiveData = $bindable(false),
    onCancel: vOnCancel,
    onConfirm: vOnConfirm
  }: {
    open: boolean;
    includeSensitiveData: boolean;
    onCancel: () => void;
    onConfirm: () => void;
  } = $props();

  function fHandleOpenChange(pNewOpen: boolean) {
    if (!pNewOpen) {
      vOnCancel();
    }
  }
</script>

<AlertDialog.Root open={vOpen} onOpenChange={fHandleOpenChange}>
  <AlertDialog.Content>
    <AlertDialog.Header>
      <AlertDialog.Title class="flex items-center gap-2">
        {#if vIncludeSensitiveData}
          <ShieldOff class="h-5 w-5 text-destructive" />
        {:else}
          <Shield class="h-5 w-5 text-destructive" />
        {/if}
        {fTranslate('text_653fc2d4487b')}
      </AlertDialog.Title>

      <AlertDialog.Description>
        {#if vIncludeSensitiveData}
          <p class="text-amber-500">{fTranslate('text_2128979a60fe')}</p>
        {:else}
          <p>{fTranslate('text_2a84957e75b8')}</p>
        {/if}
      </AlertDialog.Description>
    </AlertDialog.Header>

    <div class="flex items-center gap-2 py-2">
      <Checkbox id="include-sensitive" bind:checked={vIncludeSensitiveData} />

      <Label
        for="include-sensitive"
        class="text-sm leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70"
      >
        {#if vIncludeSensitiveData}
          <span class="text-destructive">{fTranslate('text_e62045a726cd')}</span>
        {:else}
          <span>{fTranslate('text_56edd55de767')}</span>
        {/if}
      </Label>
    </div>

    <AlertDialog.Footer>
      <AlertDialog.Cancel onclick={vOnCancel}>{fTranslate('text_19766ed6ccb2')}</AlertDialog.Cancel>

      <AlertDialog.Action
        onclick={vOnConfirm}
        class="bg-destructive text-white hover:bg-destructive/80"
      >
        {#if vIncludeSensitiveData}
          {fTranslate('text_2e0e2b8be2ff')}
        {:else}
          {fTranslate('text_41c9da198e89')}
        {/if}
      </AlertDialog.Action>
    </AlertDialog.Footer>
  </AlertDialog.Content>
</AlertDialog.Root>
