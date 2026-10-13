<script lang="ts">
  import { goto } from '$app/navigation';
  import { getChatActionsContext, setMessageEditContext } from '$lib/contexts';
  import { cChatStore, fPendingEditMessageId } from '$lib/stores/chat.svelte';
  import { cIsMobile } from '$lib/stores/viewport.svelte';
  import { cConversationsStore } from '$lib/stores/conversations.svelte';
  import { DatabaseService } from '$lib/services/database.service';
  import { cSystemMessagePlaceholder } from '$lib/constants';
  import { cReasoningTags } from '$lib/constants/agentic';
  import { MessageRole, AttachmentType, AgenticSectionType } from '$lib/enums';
  import {
    ChatMessageAssistant,
    ChatMessageUser,
    ChatMessageSystem,
    ChatMessageMcpPrompt
  } from '$lib/components/app/chat';
  import { parseFilesToMessageExtras } from '$lib/utils/browser-only';
  import { deriveAgenticSections } from '$lib/utils';
  import type { DatabaseMessageExtraMcpPrompt } from '$lib/types';
  import { cRoutes } from '$lib/constants/routes';

  interface Props {
    class?: string;
    message: DatabaseMessage;
    toolMessages?: DatabaseMessage[];
    isLastAssistantMessage?: boolean;
    isLastUserMessage?: boolean;
    nextAssistantMessage?: DatabaseMessage | null;
    siblingInfo?: ChatMessageSiblingInfo | null;
  }

  let {
    class: vClassName = '',
    message: vMessage,
    toolMessages: vToolMessages = [],
    isLastAssistantMessage: vIsLastAssistantMessage = false,
    isLastUserMessage: vIsLastUserMessage = false,
    nextAssistantMessage: vNextAssistantMessage = null,
    siblingInfo: vSiblingInfo = null
  }: Props = $props();

  const cChatActions = getChatActionsContext();

  let vDeletionInfo = $state<{
    totalCount: number;
    userMessages: number;
    assistantMessages: number;
    messageTypes: string[];
  } | null>(null);
  // The system message placeholder must never surface as editable content; keeping
  // it in the derived (not just in handleEdit) guards against prop invalidation
  // reverting the override while editing
  let vEditedContent = $derived(
    vMessage.role === MessageRole.SYSTEM && vMessage.content === cSystemMessagePlaceholder
      ? ''
      : vMessage.content
  );

  let vRawEditContent = $derived.by(() => {
    if (vMessage.role !== MessageRole.ASSISTANT) return undefined;

    const cSections = deriveAgenticSections(vMessage, vToolMessages, [], false);
    const aParts: string[] = [];

    for (const cSection of cSections) {
      switch (cSection.type) {
        case AgenticSectionType.REASONING:
        case AgenticSectionType.REASONING_PENDING:
          aParts.push(`${cReasoningTags.START}\n${cSection.content}\n${cReasoningTags.END}`);
          break;

        case AgenticSectionType.TEXT:
          aParts.push(cSection.content);
          break;

        case AgenticSectionType.TOOL_CALL:
        case AgenticSectionType.TOOL_CALL_PENDING:
        case AgenticSectionType.TOOL_CALL_STREAMING: {
          const dCallObj: Record<string, unknown> = { name: cSection.toolName };

          if (cSection.toolArgs) {
            try {
              dCallObj.arguments = JSON.parse(cSection.toolArgs);
            } catch {
              dCallObj.arguments = cSection.toolArgs;
            }
          }

          aParts.push(JSON.stringify(dCallObj, null, 2));

          if (cSection.toolResult) {
            aParts.push(`[Tool Result]\n${cSection.toolResult}`);
          }

          break;
        }
      }
    }

    return aParts.join('\n\n\n');
  });
  let vEditedExtras = $derived<DatabaseMessageExtra[]>(vMessage.extra ? [...vMessage.extra] : []);
  let vEditedUploadedFiles = $state<ChatUploadedFile[]>([]);
  let vIsEditing = $state(false);
  let vShowDeleteDialog = $state(false);
  let vShouldBranchAfterEdit = $state(false);
  let vTextareaElement: HTMLTextAreaElement | undefined = $state();

  let vShowSaveOnlyOption = $derived(vMessage.role === MessageRole.USER);
  let vShowBranchAfterEditOption = $derived(vMessage.role === MessageRole.ASSISTANT);

  setMessageEditContext({
    get isEditing() {
      return vIsEditing;
    },
    get editedContent() {
      return vEditedContent;
    },
    get editedExtras() {
      return vEditedExtras;
    },
    get editedUploadedFiles() {
      return vEditedUploadedFiles;
    },
    get originalContent() {
      return vMessage.role === MessageRole.ASSISTANT
        ? (vRawEditContent ?? vMessage.content)
        : vMessage.content;
    },
    get originalExtras() {
      return vMessage.extra || [];
    },
    get showSaveOnlyOption() {
      return vShowSaveOnlyOption;
    },
    get showBranchAfterEditOption() {
      return vShowBranchAfterEditOption;
    },
    get shouldBranchAfterEdit() {
      return vShouldBranchAfterEdit;
    },
    get messageRole() {
      return vMessage.role;
    },
    get rawEditContent() {
      return vRawEditContent;
    },
    setContent: (pContent: string) => {
      vEditedContent = pContent;
    },
    setExtras: (pExtras: DatabaseMessageExtra[]) => {
      vEditedExtras = pExtras;
    },
    setUploadedFiles: (pFiles: ChatUploadedFile[]) => {
      vEditedUploadedFiles = pFiles;
    },
    setShouldBranchAfterEdit: (pValue: boolean) => {
      vShouldBranchAfterEdit = pValue;
    },
    save: fHandleSaveEdit,
    saveOnly: fHandleSaveEditOnly,
    cancel: fHandleCancelEdit,
    startEdit: fHandleEdit
  });

  let vMcpPromptExtra = $derived.by(() => {
    if (vMessage.role !== MessageRole.USER) return null;
    if (vMessage.content.trim()) return null;
    if (!vMessage.extra || vMessage.extra.length !== 1) return null;

    const cExtra = vMessage.extra[0];

    if (cExtra.type === AttachmentType.MCP_PROMPT) {
      return cExtra as DatabaseMessageExtraMcpPrompt;
    }

    return null;
  });

  $effect(() => {
    const cPendingId = fPendingEditMessageId();

    if (cPendingId && cPendingId === vMessage.id && !vIsEditing) {
      fHandleEdit();
      cChatStore.mClearPendingEditMessageId();
    }
  });

  async function fHandleCancelEdit() {
    vIsEditing = false;

    // If canceling a new system message with placeholder content, remove it without deleting children
    if (vMessage.role === MessageRole.SYSTEM && vMessage.content === cSystemMessagePlaceholder) {
      const cConversationDeleted = await cChatStore.mRemoveSystemPromptPlaceholder(vMessage.id);

      if (cConversationDeleted) {
        goto(cRoutes.START);
      }

      return;
    }

    vEditedContent =
      vMessage.role === MessageRole.ASSISTANT
        ? vRawEditContent || vMessage.content || ''
        : vMessage.content;
    vEditedExtras = vMessage.extra ? [...vMessage.extra] : [];
    vEditedUploadedFiles = [];
  }

  function fHandleCopy() {
    cChatActions.copy(vMessage);
  }

  async function fHandleConfirmDelete() {
    if (vMessage.role === MessageRole.SYSTEM) {
      const cConversationDeleted = await cChatStore.mRemoveSystemPromptPlaceholder(vMessage.id);

      if (cConversationDeleted) {
        goto(cRoutes.START);
      }
    } else {
      cChatActions.delete(vMessage);
    }

    vShowDeleteDialog = false;
  }

  async function fHandleDelete() {
    vDeletionInfo = await cChatStore.mGetDeletionInfo(vMessage.id);
    vShowDeleteDialog = true;
  }

  function fHandleEdit() {
    vIsEditing = true;
    // Clear temporary placeholder content for system messages
    if (vMessage.role === MessageRole.SYSTEM && vMessage.content === cSystemMessagePlaceholder) {
      vEditedContent = '';
    } else if (vMessage.role === MessageRole.ASSISTANT) {
      vEditedContent = vRawEditContent || vMessage.content || '';
    } else {
      vEditedContent = vMessage.content;
    }

    vTextareaElement?.focus({ preventScroll: true });
    vEditedExtras = vMessage.extra ? [...vMessage.extra] : [];
    vEditedUploadedFiles = [];

    setTimeout(() => {
      if (vTextareaElement) {
        vTextareaElement.focus();
        vTextareaElement.setSelectionRange(
          vTextareaElement.value.length,
          vTextareaElement.value.length
        );
      }
    }, 0);
  }

  function fHandleRegenerate(pModelOverride?: string) {
    cChatActions.regenerateWithBranching(vMessage, pModelOverride);
  }

  function fHandleContinue() {
    cChatActions.continueAssistantMessage(vMessage);
  }

  function fHandleForkConversation(pOptions: { name: string; includeAttachments: boolean }) {
    cChatActions.forkConversation(vMessage, pOptions);
  }

  function fHandleNavigateToSibling(pSiblingId: string) {
    cChatActions.navigateToSibling(pSiblingId);
  }

  // After the system message flow ends, hand focus to the main chat form
  function fFocusMainChatForm() {
    if (cIsMobile.current) return;
    document.querySelector<HTMLTextAreaElement>('.chat-screen-form-wrapper textarea')?.focus();
  }

  async function fHandleSaveEdit() {
    if (vMessage.role === MessageRole.SYSTEM) {
      // System messages: update in place without branching
      const cNewContent = vEditedContent.trim();

      // If content is empty, remove without deleting children
      if (!cNewContent) {
        const cConversationDeleted = await cChatStore.mRemoveSystemPromptPlaceholder(vMessage.id);
        vIsEditing = false;
        if (cConversationDeleted) {
          goto(cRoutes.START);
        } else {
          fFocusMainChatForm();
        }
        return;
      }

      await DatabaseService.mUpdateMessage(vMessage.id, { content: cNewContent });
      const cIndex = cConversationsStore.mFindMessageIndex(vMessage.id);
      if (cIndex !== -1) {
        cConversationsStore.mUpdateMessageAtIndex(cIndex, { content: cNewContent });
      }
      fFocusMainChatForm();
    } else if (vMessage.role === MessageRole.USER) {
      const cFinalExtras = await fGetMergedExtras();
      cChatActions.editWithBranching(vMessage, vEditedContent.trim(), cFinalExtras);
    } else {
      // For assistant messages, preserve exact content including trailing whitespace
      // This is important for the Continue feature to work properly
      cChatActions.editWithReplacement(vMessage, vEditedContent, vShouldBranchAfterEdit);
    }

    vIsEditing = false;
    vShouldBranchAfterEdit = false;
    vEditedUploadedFiles = [];
  }

  async function fHandleSaveEditOnly() {
    if (vMessage.role === MessageRole.USER) {
      // For user messages, trim to avoid accidental whitespace
      const cFinalExtras = await fGetMergedExtras();
      cChatActions.editUserMessagePreserveResponses(vMessage, vEditedContent.trim(), cFinalExtras);
    }

    vIsEditing = false;
    vEditedUploadedFiles = [];
  }

  async function fGetMergedExtras(): Promise<DatabaseMessageExtra[]> {
    if (vEditedUploadedFiles.length === 0) {
      return vEditedExtras;
    }

    const cPlainFiles = $state.snapshot(vEditedUploadedFiles);
    const cResult = await parseFilesToMessageExtras(cPlainFiles);
    const cNewExtras = cResult?.extras || [];

    return [...vEditedExtras, ...cNewExtras];
  }

  function fHandleShowDeleteDialogChange(pShow: boolean) {
    vShowDeleteDialog = pShow;
  }
</script>

<div class="chat-message">
  {#if vMessage.role === MessageRole.SYSTEM}
    <ChatMessageSystem
      bind:textareaElement={vTextareaElement}
      class={vClassName}
      deletionInfo={vDeletionInfo}
      message={vMessage}
      onConfirmDelete={fHandleConfirmDelete}
      onCopy={fHandleCopy}
      onDelete={fHandleDelete}
      onEdit={fHandleEdit}
      onNavigateToSibling={fHandleNavigateToSibling}
      onShowDeleteDialogChange={fHandleShowDeleteDialogChange}
      showDeleteDialog={vShowDeleteDialog}
      siblingInfo={vSiblingInfo}
    />
  {:else if vMcpPromptExtra}
    <ChatMessageMcpPrompt
      class={vClassName}
      deletionInfo={vDeletionInfo}
      message={vMessage}
      mcpPrompt={vMcpPromptExtra}
      onConfirmDelete={fHandleConfirmDelete}
      onCopy={fHandleCopy}
      onDelete={fHandleDelete}
      onEdit={fHandleEdit}
      onNavigateToSibling={fHandleNavigateToSibling}
      onShowDeleteDialogChange={fHandleShowDeleteDialogChange}
      showDeleteDialog={vShowDeleteDialog}
      siblingInfo={vSiblingInfo}
    />
  {:else if vMessage.role === MessageRole.USER}
    <ChatMessageUser
      class={vClassName}
      deletionInfo={vDeletionInfo}
      isLastUserMessage={vIsLastUserMessage}
      message={vMessage}
      nextAssistantMessage={vNextAssistantMessage}
      onConfirmDelete={fHandleConfirmDelete}
      onCopy={fHandleCopy}
      onDelete={fHandleDelete}
      onEdit={fHandleEdit}
      onForkConversation={fHandleForkConversation}
      onNavigateToSibling={fHandleNavigateToSibling}
      onShowDeleteDialogChange={fHandleShowDeleteDialogChange}
      showDeleteDialog={vShowDeleteDialog}
      siblingInfo={vSiblingInfo}
    />
  {:else}
    <ChatMessageAssistant
      bind:textareaElement={vTextareaElement}
      class={vClassName}
      deletionInfo={vDeletionInfo}
      isLastAssistantMessage={vIsLastAssistantMessage}
      message={vMessage}
      toolMessages={vToolMessages}
      onConfirmDelete={fHandleConfirmDelete}
      onContinue={fHandleContinue}
      onCopy={fHandleCopy}
      onDelete={fHandleDelete}
      onEdit={fHandleEdit}
      onForkConversation={fHandleForkConversation}
      onNavigateToSibling={fHandleNavigateToSibling}
      onRegenerate={fHandleRegenerate}
      onShowDeleteDialogChange={fHandleShowDeleteDialogChange}
      showDeleteDialog={vShowDeleteDialog}
      siblingInfo={vSiblingInfo}
    />
  {/if}
</div>

<style>
  /*
   * The browser skips layout and paint for messages outside the
   * viewport. contain-intrinsic-size reuses the last rendered size
   * once known; 500px sizes messages that have never been rendered.
   */
  .chat-message {
    content-visibility: auto;
    contain-intrinsic-size: auto 500px;
  }
</style>
