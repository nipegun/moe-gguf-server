import { getContext, setContext } from 'svelte';
import { cContextKeyMessageEdit } from '$lib/constants';
import { MessageRole } from '$lib/enums';

export interface MessageEditState {
  readonly isEditing: boolean;
  readonly editedContent: string;
  readonly editedExtras: DatabaseMessageExtra[];
  readonly editedUploadedFiles: ChatUploadedFile[];
  readonly originalContent: string;
  readonly originalExtras: DatabaseMessageExtra[];
  readonly showSaveOnlyOption: boolean;
  readonly showBranchAfterEditOption: boolean;
  readonly shouldBranchAfterEdit: boolean;
  readonly messageRole: MessageRole;
  readonly rawEditContent?: string;
}

export interface MessageEditActions {
  setContent: (pContent: string) => void;
  setExtras: (pExtras: DatabaseMessageExtra[]) => void;
  setUploadedFiles: (pFiles: ChatUploadedFile[]) => void;
  save: () => void;
  saveOnly: () => void;
  cancel: () => void;
  startEdit: () => void;
}

export interface AssistantEditActions {
  setShouldBranchAfterEdit: (pValue: boolean) => void;
}

export type MessageEditContext = MessageEditState &
  MessageEditActions &
  Partial<AssistantEditActions>;

const cMessageEditKey = Symbol.for(cContextKeyMessageEdit);

/**
 * Sets the message edit context. Call this in the parent component (ChatMessage.svelte).
 */
export function fSetMessageEditContext(pCtx: MessageEditContext): MessageEditContext {
  return setContext(cMessageEditKey, pCtx);
}

/**
 * Gets the message edit context. Call this in child components.
 */
export function fGetMessageEditContext(): MessageEditContext {
  return getContext(cMessageEditKey);
}
