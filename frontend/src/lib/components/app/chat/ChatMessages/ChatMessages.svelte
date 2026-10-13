<script lang="ts">
  import { fTranslate } from '../../../../i18n';

  import { ChatMessage, ChatMessageUserPending } from '$lib/components/app';
  import { setChatActionsContext } from '$lib/contexts';
  import { MessageRole } from '$lib/enums';
  import { cChatStore } from '$lib/stores/chat.svelte';
  import {
    fChatPendingMessageContent,
    fChatPendingMessageExtras,
    fChatClearPendingMessage,
    fChatInjectPendingMessage
  } from '$lib/stores/chat.svelte';
  import { cConversationsStore, fActiveConversation } from '$lib/stores/conversations.svelte';
  import { fConfig } from '$lib/stores/settings.svelte';
  import {
    fAgenticPendingSteeringMessageContent,
    fAgenticPendingSteeringMessageExtras,
    fAgenticClearSteeringMessage,
    fAgenticInjectSteeringMessage
  } from '$lib/stores/agentic.svelte';
  import {
    buildSiblingInfoMap,
    copyToClipboard,
    formatMessageForClipboard,
    hasAgenticContent
  } from '$lib/utils';

  interface Props {
    messages?: DatabaseMessage[];
    onUserAction?: () => void;
    onMessagesReady?: (pMessageCount: number) => void;
  }

  let {
    messages: vMessages = [],
    onUserAction: vOnUserAction,
    onMessagesReady: vOnMessagesReady
  }: Props = $props();

  let vAllConversationMessages = $state<DatabaseMessage[]>([]);

  const cCurrentConfig = fConfig();

  setChatActionsContext({
    copy: async (pMessage: DatabaseMessage) => {
      const cAsPlainText = Boolean(cCurrentConfig.copyTextAttachmentsAsPlainText);
      const cClipboardContent = formatMessageForClipboard(
        pMessage.content,
        pMessage.extra,
        cAsPlainText
      );
      await copyToClipboard(cClipboardContent, fTranslate('text_d56d654f01ce'));
    },

    delete: async (pMessage: DatabaseMessage) => {
      await cChatStore.mDeleteMessage(pMessage.id);
      fRefreshAllMessages();
    },

    navigateToSibling: async (pSiblingId: string) => {
      await cConversationsStore.mNavigateToSibling(pSiblingId);
    },

    editWithBranching: async (
      pMessage: DatabaseMessage,
      pNewContent: string,
      pNewExtras?: DatabaseMessageExtra[]
    ) => {
      vOnUserAction?.();
      await cChatStore.mEditMessageWithBranching(pMessage.id, pNewContent, pNewExtras);
      fRefreshAllMessages();
    },

    editWithReplacement: async (
      pMessage: DatabaseMessage,
      pNewContent: string,
      pShouldBranch: boolean
    ) => {
      vOnUserAction?.();
      await cChatStore.mEditAssistantMessage(pMessage.id, pNewContent, pShouldBranch);
      fRefreshAllMessages();
    },

    editUserMessagePreserveResponses: async (
      pMessage: DatabaseMessage,
      pNewContent: string,
      pNewExtras?: DatabaseMessageExtra[]
    ) => {
      vOnUserAction?.();
      await cChatStore.mEditUserMessagePreserveResponses(pMessage.id, pNewContent, pNewExtras);
      fRefreshAllMessages();
    },

    regenerateWithBranching: async (pMessage: DatabaseMessage, pModelOverride?: string) => {
      vOnUserAction?.();
      await cChatStore.mRegenerateMessageWithBranching(pMessage.id, pModelOverride);
      fRefreshAllMessages();
    },

    continueAssistantMessage: async (pMessage: DatabaseMessage) => {
      vOnUserAction?.();
      await cChatStore.mContinueAssistantMessage(pMessage.id);
      fRefreshAllMessages();
    },

    forkConversation: async (
      pMessage: DatabaseMessage,
      pOptions: { name: string; includeAttachments: boolean }
    ) => {
      await cConversationsStore.mForkConversation(pMessage.id, pOptions);
    }
  });

  function fRefreshAllMessages() {
    const cConversation = fActiveConversation();

    if (cConversation) {
      cConversationsStore.mGetConversationMessages(cConversation.id).then((pMessages) => {
        vAllConversationMessages = pMessages;
      });
    } else {
      vAllConversationMessages = [];
    }
  }

  // Refresh messages whenever the active conversation changes
  $effect(() => {
    if (fActiveConversation()) {
      fRefreshAllMessages();
    }
  });

  $effect(() => {
    void vAllConversationMessages;

    vOnMessagesReady?.(vDisplayMessages.length);
  });

  let vSiblingInfoByMessageId = $derived(buildSiblingInfoMap(vAllConversationMessages));

  let vDisplayMessages = $derived.by(() => {
    if (!vMessages.length) {
      return [];
    }

    const cFilteredMessages = cCurrentConfig.showSystemMessage
      ? vMessages
      : vMessages.filter((pMsg) => pMsg.type !== MessageRole.SYSTEM);

    // Build display entries, grouping agentic sessions into single entries.
    // An agentic session = assistant(with tool_calls) → tool → assistant → tool → ... → assistant(final)
    const aResult: Array<{
      message: DatabaseMessage;
      toolMessages: DatabaseMessage[];
      isLastAssistantMessage: boolean;
      isLastUserMessage: boolean;
      nextAssistantMessage: DatabaseMessage | null;
      siblingInfo: ChatMessageSiblingInfo;
    }> = [];

    for (let vI = 0; vI < cFilteredMessages.length; vI++) {
      const cMsg = cFilteredMessages[vI];

      // Skip tool messages - they're grouped with preceding assistant
      if (cMsg.role === MessageRole.TOOL) continue;

      const aToolMessages: DatabaseMessage[] = [];
      if (cMsg.role === MessageRole.ASSISTANT && hasAgenticContent(cMsg)) {
        let vJ = vI + 1;

        while (vJ < cFilteredMessages.length) {
          const cNext = cFilteredMessages[vJ];

          if (cNext.role === MessageRole.TOOL) {
            aToolMessages.push(cNext);

            vJ++;
          } else if (cNext.role === MessageRole.ASSISTANT) {
            aToolMessages.push(cNext);

            vJ++;
          } else {
            break;
          }
        }

        vI = vJ - 1;
      } else if (cMsg.role === MessageRole.ASSISTANT) {
        let vJ = vI + 1;

        while (vJ < cFilteredMessages.length && cFilteredMessages[vJ].role === MessageRole.TOOL) {
          aToolMessages.push(cFilteredMessages[vJ]);
          vJ++;
        }
      }

      const cSiblingInfo = vSiblingInfoByMessageId.get(cMsg.id) ?? {
        message: cMsg,
        siblingIds: [cMsg.id],
        currentIndex: 0,
        totalSiblings: 1
      };

      aResult.push({
        message: cMsg,
        toolMessages: aToolMessages,
        isLastAssistantMessage: false,
        isLastUserMessage: false,
        nextAssistantMessage: null,
        siblingInfo: cSiblingInfo
      });
    }

    let vLastAssistantIdx = -1;
    for (let vI = aResult.length - 1; vI >= 0; vI--) {
      if (aResult[vI].message.role === MessageRole.ASSISTANT) {
        aResult[vI].isLastAssistantMessage = true;
        vLastAssistantIdx = vI;
        break;
      }
    }

    if (vLastAssistantIdx > 0 && aResult[vLastAssistantIdx - 1].message.role === MessageRole.USER) {
      aResult[vLastAssistantIdx - 1].isLastUserMessage = true;
    }

    for (let vI = 0; vI < aResult.length; vI++) {
      if (aResult[vI].message.role !== MessageRole.USER) continue;

      for (let vJ = vI + 1; vJ < aResult.length; vJ++) {
        if (aResult[vJ].message.role === MessageRole.ASSISTANT) {
          aResult[vI].nextAssistantMessage = aResult[vJ].message;
          break;
        }
      }
    }

    return aResult;
  });
</script>

<div>
  {#each vDisplayMessages as { message, toolMessages, isLastAssistantMessage, isLastUserMessage, nextAssistantMessage, siblingInfo } (message.id)}
    <ChatMessage
      class="mx-auto mt-12 w-full max-w-3xl"
      {message}
      {toolMessages}
      {isLastAssistantMessage}
      {isLastUserMessage}
      {nextAssistantMessage}
      {siblingInfo}
    />
  {/each}

  {#if fActiveConversation() && fAgenticPendingSteeringMessageContent(fActiveConversation()!.id)}
    {@const convId = fActiveConversation()!.id}
    {@const pendingContent = fAgenticPendingSteeringMessageContent(convId)}

    {#if pendingContent}
      <ChatMessageUserPending
        class="mx-auto mt-12 w-full max-w-[48rem]"
        content={pendingContent}
        extras={fAgenticPendingSteeringMessageExtras(convId)}
        onSendImmediately={() => cChatStore.mAbortCurrentFlow(convId)}
        onEdit={(newContent, extras) => fAgenticInjectSteeringMessage(convId, newContent, extras)}
        onDelete={() => fAgenticClearSteeringMessage(convId)}
      />
    {/if}
  {:else if fActiveConversation() && fChatPendingMessageContent(fActiveConversation()!.id)}
    {@const convId = fActiveConversation()!.id}
    {@const pendingContent = fChatPendingMessageContent(convId)}

    {#if pendingContent}
      <ChatMessageUserPending
        class="mx-auto mt-12 w-full max-w-[48rem]"
        content={pendingContent}
        extras={fChatPendingMessageExtras(convId)}
        onSendImmediately={() => cChatStore.mAbortCurrentFlow(convId)}
        onEdit={(newContent, extras) => fChatInjectPendingMessage(convId, newContent, extras)}
        onDelete={() => fChatClearPendingMessage(convId)}
      />
    {/if}
  {/if}
</div>
