<script lang="ts">
  import { fTranslate } from '../../../i18n';

  import * as AlertDialog from '$lib/components/ui/alert-dialog';
  import type { Component, Snippet } from 'svelte';
  import { KeyboardKey } from '$lib/enums';

  interface Props {
    open: boolean;
    title: string;
    description: string;
    confirmText?: string;
    cancelText?: string;
    variant?: 'default' | 'destructive';
    icon?: Component;
    onConfirm: () => void;
    onCancel: () => void;
    onKeydown?: (pEvent: KeyboardEvent) => void;
    children?: Snippet;
  }

  let {
    open: vOpen = $bindable(),
    title: vTitle,
    description: vDescription,
    confirmText: vConfirmText = fTranslate('text_eebdd24a77d9'),
    cancelText: vCancelText = fTranslate('text_19766ed6ccb2'),
    variant: vVariant = 'default',
    icon: vIcon,
    onConfirm: vOnConfirm,
    onCancel: vOnCancel,
    onKeydown: vOnKeydown,
    children: vChildren
  }: Props = $props();

  function fHandleKeydown(pEvent: KeyboardEvent) {
    if (pEvent.key === KeyboardKey.ENTER) {
      pEvent.preventDefault();

      vOnConfirm();
    }
    vOnKeydown?.(pEvent);
  }

  function fHandleOpenChange(pNewOpen: boolean) {
    if (!pNewOpen) {
      vOnCancel();
    }
  }
</script>

<AlertDialog.Root open={vOpen} onOpenChange={fHandleOpenChange}>
  <AlertDialog.Content onkeydown={fHandleKeydown}>
    <AlertDialog.Header>
      <AlertDialog.Title class="flex items-center gap-2">
        {#if vIcon}
          {@const IconComponent = vIcon}

          <IconComponent class="h-5 w-5 {vVariant === 'destructive' ? 'text-destructive' : ''}" />
        {/if}
        {vTitle}
      </AlertDialog.Title>

      <AlertDialog.Description>
        {vDescription}
      </AlertDialog.Description>
    </AlertDialog.Header>

    {#if vChildren}
      {@render vChildren()}
    {/if}

    <AlertDialog.Footer>
      <AlertDialog.Cancel onclick={vOnCancel}>{vCancelText}</AlertDialog.Cancel>
      <AlertDialog.Action
        onclick={vOnConfirm}
        class={vVariant === 'destructive'
          ? 'bg-destructive text-white hover:bg-destructive/80'
          : ''}
      >
        {vConfirmText}
      </AlertDialog.Action>
    </AlertDialog.Footer>
  </AlertDialog.Content>
</AlertDialog.Root>
