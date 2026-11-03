import { cNewChatDraftKey } from '$lib/constants';

interface DraftMessage {
  message: string;
  files: ChatUploadedFile[];
}

class DraftMessagesStore {
  private dDrafts = new Map<string, DraftMessage>();

  mGetDraftMessage(pChatId: string | undefined): DraftMessage {
    const cKey = pChatId ?? cNewChatDraftKey;
    return this.dDrafts.get(cKey) ?? { message: '', files: [] };
  }

  mSaveDraftMessage(
    pChatId: string | undefined,
    pMessage: string,
    pFiles: ChatUploadedFile[]
  ): void {
    const cKey = pChatId ?? cNewChatDraftKey;
    if (pMessage || pFiles.length > 0) {
      this.dDrafts.set(cKey, { message: pMessage, files: [...pFiles] });
    } else {
      this.dDrafts.delete(cKey);
    }
  }

  mClearDraftMessage(pChatId: string | undefined): void {
    const cKey = pChatId ?? cNewChatDraftKey;
    this.dDrafts.delete(cKey);
  }
}

export const cDraftMessagesStore = new DraftMessagesStore();
