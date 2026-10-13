<script lang="ts">
  import { fTranslate } from '../../../../../../i18n';

  import {
    ChatMessageAgenticContent,
    ChatMessageActionIcons,
    ChatMessageAssistantModel,
    ChatMessageAssistantProcessingInfo,
    ChatMessageAssistantRawOutput,
    ChatMessageAssistantStatistics,
    ChatMessageEditForm
  } from '$lib/components/app';
  import { getMessageEditContext } from '$lib/contexts';
  import { fUseProcessingState } from '$lib/hooks/use-processing-state.svelte';
  import { cChatStore, fIsLoading, fIsChatStreaming } from '$lib/stores/chat.svelte';
  import { modelLoadProgressText, queueStatusText } from '$lib/utils';
  import { MessageRole } from '$lib/enums';
  import { fConfig } from '$lib/stores/settings.svelte';
  import { fIsRouterMode } from '$lib/stores/server.svelte';
  import { cModelsStore } from '$lib/stores/models.svelte';

  import { hasAgenticContent } from '$lib/utils';

  interface Props {
    class?: string;
    deletionInfo: {
      totalCount: number;
      userMessages: number;
      assistantMessages: number;
      messageTypes: string[];
    } | null;
    isLastAssistantMessage?: boolean;
    message: DatabaseMessage;
    toolMessages?: DatabaseMessage[];
    onCopy: () => void;
    onConfirmDelete: () => void;
    onContinue?: () => void;
    onDelete: () => void;
    onEdit?: () => void;
    onForkConversation?: (pOptions: { name: string; includeAttachments: boolean }) => void;
    onNavigateToSibling?: (pSiblingId: string) => void;
    onRegenerate: (pModelOverride?: string) => void;
    onShowDeleteDialogChange: (pShow: boolean) => void;
    showDeleteDialog: boolean;
    siblingInfo?: ChatMessageSiblingInfo | null;
    textareaElement?: HTMLTextAreaElement;
  }

  let {
    class: vClassName = '',
    deletionInfo: vDeletionInfo,
    isLastAssistantMessage: vIsLastAssistantMessage = false,
    message: vMessage,
    toolMessages: vToolMessages = [],
    onConfirmDelete: vOnConfirmDelete,
    onContinue: vOnContinue,
    onCopy: vOnCopy,
    onDelete: vOnDelete,
    onEdit: vOnEdit,
    onForkConversation: vOnForkConversation,
    onNavigateToSibling: vOnNavigateToSibling,
    onRegenerate: vOnRegenerate,
    onShowDeleteDialogChange: vOnShowDeleteDialogChange,
    showDeleteDialog: vShowDeleteDialog,
    siblingInfo: vSiblingInfo = null,
    textareaElement: vTextareaElement = $bindable()
  }: Props = $props();

  // Get edit context
  const cEditCtx = getMessageEditContext();

  const cIsAgentic = $derived(hasAgenticContent(vMessage, vToolMessages));
  const cProcessingState = fUseProcessingState();

  let vCurrentConfig = $derived(fConfig());
  let vIsRouter = $derived(fIsRouterMode());

  let vShowRawOutput = $state(false);

  let vDisplayedModel = $derived(vMessage.model ?? null);

  let vIsCurrentlyLoading = $derived(fIsLoading());
  let vIsStreaming = $derived(fIsChatStreaming());
  let vHasNoContent = $derived(!vMessage?.content?.trim());
  let vIsActivelyProcessing = $derived(vIsCurrentlyLoading || vIsStreaming);

  // during a router auto-load the message has no model yet: target the model frozen in the
  // persisted stream state (survives a reload), then fall back to the dropdown selection
  let vLoadTargetModel = $derived(
    vMessage.model ?? cChatStore.mGetResumeModel(vMessage.convId) ?? cModelsStore.vSelectedModelName
  );
  let vModelLoadProgress = $derived(
    vIsRouter && vLoadTargetModel ? cModelsStore.mGetLoadProgress(vLoadTargetModel) : null
  );
  let vModelLoadingText = $derived(modelLoadProgressText(vModelLoadProgress));
  // the request waits behind others, for example ones sent directly to the API
  let vQueueText = $derived(queueStatusText(cChatStore.mGetQueueStatus(vMessage.convId)));

  let vShowProcessingInfoTop = $derived(
    vMessage?.role === MessageRole.ASSISTANT &&
      vIsActivelyProcessing &&
      vHasNoContent &&
      !cIsAgentic &&
      vIsLastAssistantMessage
  );

  let vShowProcessingInfoBottom = $derived(
    vMessage?.role === MessageRole.ASSISTANT &&
      vIsActivelyProcessing &&
      (!vHasNoContent || cIsAgentic) &&
      vIsLastAssistantMessage
  );

  let vAssistantEl: HTMLDivElement | undefined = $state();
  let vLastUserMessageHeight = $state(0);
  let vAssistantMarginTop = $state(0);

  $effect(() => {
    if (!vAssistantEl) return;

    vAssistantMarginTop = Math.round(parseFloat(getComputedStyle(vAssistantEl).marginTop));

    const cChatMessageEl = vAssistantEl.closest('.chat-message');
    const cPreviousChatMessage = cChatMessageEl?.previousElementSibling;
    const cUserMessageEl = cPreviousChatMessage?.querySelector(
      '.chat-message-user'
    ) as HTMLElement | null;

    if (!cUserMessageEl) {
      vLastUserMessageHeight = 0;
      return;
    }

    const fUpdateHeight = () => {
      const cRect = cUserMessageEl.getBoundingClientRect();
      const cMarginTop = Math.round(parseFloat(getComputedStyle(cUserMessageEl).marginTop));
      vLastUserMessageHeight = Math.round(cRect.height + cMarginTop);
    };

    fUpdateHeight();

    const cResizeObserver = new ResizeObserver(fUpdateHeight);
    cResizeObserver.observe(cUserMessageEl);

    return () => {
      cResizeObserver.disconnect();
    };
  });

  $effect(() => {
    if (vShowProcessingInfoTop || vShowProcessingInfoBottom) {
      cProcessingState.startMonitoring();
    }
  });
</script>

<div
  bind:this={vAssistantEl}
  class="chat-message-assistant text-md group w-full leading-7.5 {vClassName}"
  style:--last-user-message-height={vLastUserMessageHeight > 0
    ? `${vLastUserMessageHeight}px`
    : undefined}
  style:--assistant-margin-top={vAssistantMarginTop > 0 ? `${vAssistantMarginTop}px` : undefined}
  role="group"
  aria-label={fTranslate('text_d9495b112c58')}
>
  {#if vShowProcessingInfoTop}
    <ChatMessageAssistantProcessingInfo
      modelLoadingText={vModelLoadingText}
      queueText={vQueueText}
      processingState={cProcessingState}
      position="top"
    />
  {/if}

  {#if cEditCtx.isEditing}
    <ChatMessageEditForm />
  {:else if vShowRawOutput}
    <ChatMessageAssistantRawOutput message={vMessage} toolMessages={vToolMessages} />
  {:else}
    <ChatMessageAgenticContent
      message={vMessage}
      toolMessages={vToolMessages}
      isStreaming={fIsChatStreaming()}
      isLastAssistantMessage={vIsLastAssistantMessage}
    />
  {/if}

  {#if vShowProcessingInfoBottom}
    <ChatMessageAssistantProcessingInfo
      modelLoadingText={vModelLoadingText}
      queueText={vQueueText}
      processingState={cProcessingState}
      position="bottom"
    />
  {/if}

  <div class="info my-6 grid gap-4 tabular-nums">
    {#if vDisplayedModel}
      <div class="inline-flex flex-wrap items-start gap-2 text-xs text-muted-foreground">
        <ChatMessageAssistantModel
          displayedModel={vDisplayedModel}
          isLoading={fIsLoading()}
          isRouter={vIsRouter}
          onRegenerate={vOnRegenerate}
        />

        <ChatMessageAssistantStatistics
          message={vMessage}
          isLoading={fIsLoading()}
          processingState={cProcessingState}
          showMessageStats={vCurrentConfig.showMessageStats}
        />
      </div>
    {/if}
  </div>

  {#if vMessage.timestamp && !cEditCtx.isEditing}
    <ChatMessageActionIcons
      role={MessageRole.ASSISTANT}
      justify="start"
      actionsPosition="left"
      siblingInfo={vSiblingInfo}
      showDeleteDialog={vShowDeleteDialog}
      deletionInfo={vDeletionInfo}
      onCopy={vOnCopy}
      onEdit={vOnEdit}
      onRegenerate={vOnRegenerate}
      onContinue={vCurrentConfig.enableContinueGeneration ? vOnContinue : undefined}
      onForkConversation={vOnForkConversation}
      onDelete={vOnDelete}
      onConfirmDelete={vOnConfirmDelete}
      onNavigateToSibling={vOnNavigateToSibling}
      onShowDeleteDialogChange={vOnShowDeleteDialogChange}
      showRawOutputSwitch={vCurrentConfig.showRawOutputSwitch}
      rawOutputEnabled={vShowRawOutput}
      onRawOutputToggle={(enabled) => (vShowRawOutput = enabled)}
    />
  {/if}
</div>

<style>
  :global(.chat-message):last-child .chat-message-assistant {
    --assistant-min-height-offset: calc(
      var(--last-user-message-height, 19rem) + var(--chat-form-height, 6rem) +
        var(--chat-form-bottom-position, 0.5rem) + var(--chat-form-padding-top, 6rem) +
        var(--assistant-margin-top, 3rem)
    );
    min-height: calc(100dvh - var(--assistant-min-height-offset));

    @media (width > 768px) {
      --assistant-min-height-offset: calc(
        var(--last-user-message-height, 18rem) + var(--chat-form-height, 6rem) +
          var(--chat-form-bottom-position, 1rem) + var(--chat-form-padding-top, 6rem) +
          var(--assistant-margin-top, 3rem)
      );
    }
  }
</style>
