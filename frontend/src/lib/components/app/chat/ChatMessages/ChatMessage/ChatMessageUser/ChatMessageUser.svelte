<script lang="ts">
  import { fTranslate } from '../../../../../../i18n';

  import {
    ChatMessageActionIcons,
    ChatMessageEditForm,
    ChatMessageStatistics,
    ChatMessageUserBubble
  } from '$lib/components/app/chat';
  import { getMessageEditContext } from '$lib/contexts';
  import { fUseProcessingState } from '$lib/hooks/use-processing-state.svelte';
  import { fIsLoading } from '$lib/stores/chat.svelte';
  import { MessageRole, ChatMessageStatisticsMode } from '$lib/enums';
  import { fConfig } from '$lib/stores/settings.svelte';

  interface Props {
    class?: string;
    message: DatabaseMessage;
    siblingInfo?: ChatMessageSiblingInfo | null;
    deletionInfo: {
      totalCount: number;
      userMessages: number;
      assistantMessages: number;
      messageTypes: string[];
    } | null;
    isLastUserMessage?: boolean;
    nextAssistantMessage?: DatabaseMessage | null;
    showDeleteDialog: boolean;
    onEdit: () => void;
    onDelete: () => void;
    onConfirmDelete: () => void;
    onForkConversation?: (pOptions: { name: string; includeAttachments: boolean }) => void;
    onShowDeleteDialogChange: (pShow: boolean) => void;
    onNavigateToSibling?: (pSiblingId: string) => void;
    onCopy: () => void;
  }

  let {
    class: vClassName = '',
    message: vMessage,
    siblingInfo: vSiblingInfo = null,
    deletionInfo: vDeletionInfo,
    isLastUserMessage: vIsLastUserMessage = false,
    nextAssistantMessage: vNextAssistantMessage = null,
    showDeleteDialog: vShowDeleteDialog,
    onEdit: vOnEdit,
    onDelete: vOnDelete,
    onConfirmDelete: vOnConfirmDelete,
    onForkConversation: vOnForkConversation,
    onShowDeleteDialogChange: vOnShowDeleteDialogChange,
    onNavigateToSibling: vOnNavigateToSibling,
    onCopy: vOnCopy
  }: Props = $props();

  // Get contexts
  const cEditCtx = getMessageEditContext();
  const cProcessingState = fUseProcessingState();

  const cCurrentConfig = $derived(fConfig());
  const cIsActivelyProcessing = $derived(vIsLastUserMessage && fIsLoading());

  // For agentic turns, prefer the cumulative agentic.llm totals over per-call timings.
  let vStoredReadingStats = $derived.by(() => {
    const cTimings = vNextAssistantMessage?.timings;
    if (!cTimings?.prompt_n || !cTimings?.prompt_ms) return null;

    const cAgentic = cTimings.agentic;

    return {
      promptTokens: cAgentic ? cAgentic.llm.prompt_n : cTimings.prompt_n,
      promptMs: cAgentic ? cAgentic.llm.prompt_ms : cTimings.prompt_ms
    };
  });

  let vShowStoredReadingStats = $derived(
    Boolean(cCurrentConfig.showMessageStats) && vStoredReadingStats !== null
  );

  let vShowLiveReadingStats = $derived(
    Boolean(cCurrentConfig.showMessageStats) &&
      cIsActivelyProcessing &&
      vStoredReadingStats === null
  );

  $effect(() => {
    if (vShowLiveReadingStats) {
      cProcessingState.startMonitoring();
    }
  });
</script>

<div
  aria-label={fTranslate('text_4c435e0762de')}
  class="chat-message-user group flex flex-col items-end gap-3 md:gap-2 {vClassName}"
  role="group"
>
  {#if cEditCtx.isEditing}
    <ChatMessageEditForm />
  {:else}
    <ChatMessageUserBubble
      content={vMessage.content}
      attachments={vMessage.extra}
      renderMarkdown={true}
    />

    {#if vShowStoredReadingStats}
      <!-- Reading stats sourced from the assistant message that followed this turn -->
      <div class="info my-2 grid w-full justify-items-end gap-4 tabular-nums">
        <div
          class="inline-flex flex-wrap items-start justify-end gap-2 text-xs text-muted-foreground"
        >
          <ChatMessageStatistics
            mode={ChatMessageStatisticsMode.READING}
            promptTokens={vStoredReadingStats!.promptTokens}
            promptMs={vStoredReadingStats!.promptMs}
          />
        </div>
      </div>
    {:else if vShowLiveReadingStats}
      {@const liveStats = cProcessingState.getLiveProcessingStats()}
      {#if liveStats}
        <div class="info my-2 grid w-full justify-items-end gap-4 tabular-nums">
          <div
            class="inline-flex flex-wrap items-start justify-end gap-2 text-xs text-muted-foreground"
          >
            <ChatMessageStatistics
              mode={ChatMessageStatisticsMode.READING}
              isLive
              promptTokens={liveStats.tokensProcessed}
              promptMs={liveStats.timeMs}
            />
          </div>
        </div>
      {/if}
    {/if}

    {#if vMessage.timestamp}
      <div class="max-w-[80%]">
        <ChatMessageActionIcons
          actionsPosition="right"
          deletionInfo={vDeletionInfo}
          justify="end"
          onConfirmDelete={vOnConfirmDelete}
          onCopy={vOnCopy}
          onDelete={vOnDelete}
          onEdit={vOnEdit}
          onForkConversation={vOnForkConversation}
          onNavigateToSibling={vOnNavigateToSibling}
          onShowDeleteDialogChange={vOnShowDeleteDialogChange}
          siblingInfo={vSiblingInfo}
          showDeleteDialog={vShowDeleteDialog}
          role={MessageRole.USER}
        />
      </div>
    {/if}
  {/if}
</div>
