<script lang="ts">
  import { fTranslate } from '../../../i18n';

  import { Dialog } from 'bits-ui';
  import { X } from '@lucide/svelte';
  import * as DialogUI from '$lib/components/ui/dialog';
  import { ChatAttachmentsPreview } from '$lib/components/app';
  import { KeyboardKey } from '$lib/enums';

  interface Props {
    open: boolean;
    uploadedFiles?: ChatUploadedFile[];
    attachments?: DatabaseMessageExtra[];
    activeModelId?: string;
    previewFocusIndex?: number;
  }

  let {
    open: vOpen = $bindable(false),
    uploadedFiles: vUploadedFiles = [],
    attachments: vAttachments = [],
    activeModelId: vActiveModelId,
    previewFocusIndex: vPreviewFocusIndex = 0
  }: Props = $props();

  function fHandleClose() {
    vOpen = false;
  }

  $effect(() => {
    if (!vOpen) return;

    function fHandleKeyDown(pEvent: KeyboardEvent) {
      const cTarget = pEvent.target as HTMLElement;

      if (cTarget.tagName === 'INPUT' || cTarget.tagName === 'TEXTAREA') return;

      switch (pEvent.key) {
        case KeyboardKey.ARROW_LEFT:
          pEvent.preventDefault();
          pEvent.stopPropagation();

          document.dispatchEvent(new CustomEvent('chat-attachments-nav', { detail: -1 }));

          break;
        case KeyboardKey.ARROW_RIGHT:
          pEvent.preventDefault();
          pEvent.stopPropagation();

          document.dispatchEvent(new CustomEvent('chat-attachments-nav', { detail: 1 }));

          break;
        case KeyboardKey.SPACE:
          pEvent.preventDefault();
          pEvent.stopPropagation();

          document.dispatchEvent(new CustomEvent('chat-attachments-nav', { detail: 1 }));

          break;
      }
    }

    document.addEventListener('keydown', fHandleKeyDown);
    return () => document.removeEventListener('keydown', fHandleKeyDown);
  });
</script>

<Dialog.Root bind:open={vOpen}>
  <Dialog.Portal>
    <DialogUI.Overlay class="bg-black/85" style="z-index: 1000" />

    <Dialog.Content class="fixed inset-0 z-[1000] flex flex-col bg-transparent outline-none">
      <Dialog.Close
        class="absolute top-4 right-4 z-10 cursor-pointer text-white hover:text-gray-400"
        onclick={fHandleClose}
        aria-label={fTranslate('text_7d9eb7acb13e')}
      >
        <X class="size-4" />
      </Dialog.Close>

      <ChatAttachmentsPreview
        uploadedFiles={vUploadedFiles}
        attachments={vAttachments}
        activeModelId={vActiveModelId}
        previewFocusIndex={vPreviewFocusIndex}
        class="min-h-0 flex-1"
      />
    </Dialog.Content>
  </Dialog.Portal>
</Dialog.Root>
