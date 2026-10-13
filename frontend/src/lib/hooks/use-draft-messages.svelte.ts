import { onMount } from 'svelte';
import { afterNavigate, beforeNavigate } from '$app/navigation';
import { cDraftMessagesStore } from '$lib/stores/draft-messages.svelte';

interface UseDraftMessagesOptions {
  getChatId: () => string | undefined;
  getMessage: () => string;
  getFiles: () => ChatUploadedFile[];
  setMessage: (pMessage: string) => void;
  setFiles: (pFiles: ChatUploadedFile[]) => void;
  getInitialMessage: () => string;
}

export function fUseDraftMessages(pOptions: UseDraftMessagesOptions) {
  onMount(() => {
    const cChatId = pOptions.getChatId();
    const cDraft = cDraftMessagesStore.mGetDraftMessage(cChatId);

    if ((cDraft.message || cDraft.files.length > 0) && !pOptions.getInitialMessage()) {
      pOptions.setMessage(cDraft.message);
      pOptions.setFiles(cDraft.files);
    }
  });

  beforeNavigate(() => {
    const cChatId = pOptions.getChatId();
    cDraftMessagesStore.mSaveDraftMessage(cChatId, pOptions.getMessage(), pOptions.getFiles());
  });

  afterNavigate((pNavigation) => {
    if (pNavigation?.from != null) {
      const cChatId = pOptions.getChatId();
      const cDraft = cDraftMessagesStore.mGetDraftMessage(cChatId);
      pOptions.setMessage(cDraft.message);
      pOptions.setFiles(cDraft.files);
    }
  });

  function fClearDraft() {
    const cChatId = pOptions.getChatId();
    cDraftMessagesStore.mClearDraftMessage(cChatId);
  }

  return { clearDraft: fClearDraft };
}
