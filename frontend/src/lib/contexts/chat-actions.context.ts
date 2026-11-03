import { getContext, setContext } from 'svelte';
import { cContextKeyChatActions } from '$lib/constants';

export interface ChatActionsContext {
  copy: (pMessage: DatabaseMessage) => void;
  delete: (pMessage: DatabaseMessage) => void;
  navigateToSibling: (pSiblingId: string) => void;
  editWithBranching: (
    pMessage: DatabaseMessage,
    pNewContent: string,
    pNewExtras?: DatabaseMessageExtra[]
  ) => void;
  editWithReplacement: (
    pMessage: DatabaseMessage,
    pNewContent: string,
    pShouldBranch: boolean
  ) => void;
  editUserMessagePreserveResponses: (
    pMessage: DatabaseMessage,
    pNewContent: string,
    pNewExtras?: DatabaseMessageExtra[]
  ) => void;
  regenerateWithBranching: (pMessage: DatabaseMessage, pModelOverride?: string) => void;
  continueAssistantMessage: (pMessage: DatabaseMessage) => void;
  forkConversation: (
    pMessage: DatabaseMessage,
    pOptions: { name: string; includeAttachments: boolean }
  ) => void;
}

const cChatActionsKey = Symbol.for(cContextKeyChatActions);

export function fSetChatActionsContext(pCtx: ChatActionsContext): ChatActionsContext {
  return setContext(cChatActionsKey, pCtx);
}

export function fGetChatActionsContext(): ChatActionsContext {
  return getContext(cChatActionsKey);
}
