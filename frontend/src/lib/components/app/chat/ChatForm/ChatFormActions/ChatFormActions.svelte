<script lang="ts">
  import { fTranslate } from '../../../../../i18n';

  import { cIconClassDefault } from '$lib/constants/css-classes';
  import { Square, SkipForward } from '@lucide/svelte';
  import { Button } from '$lib/components/ui/button';
  import { ChatService } from '$lib/services';
  import {
    ChatFormActionsAdd,
    ChatFormActionModels,
    ChatFormActionRecord,
    ChatFormActionSubmit,
    ChatFormContextGauge
  } from '$lib/components/app';
  import { FileTypeCategory, MessageRole } from '$lib/enums';
  import { cMcpStore } from '$lib/stores/mcp.svelte';
  import { fConfig } from '$lib/stores/settings.svelte';
  import { fActiveMessages, cConversationsStore } from '$lib/stores/conversations.svelte';
  import {
    fActiveProcessingState,
    fIsChatStreaming,
    fIsLoading as chatIsLoading
  } from '$lib/stores/chat.svelte';
  import { getFileTypeCategory } from '$lib/utils';
  import { goto } from '$app/navigation';
  import { page } from '$app/state';
  import { cRoutes } from '$lib/constants/routes';

  interface Props {
    canSend?: boolean;
    canSubmit?: boolean;
    class?: string;
    disabled?: boolean;
    isLoading?: boolean;
    isReasoning?: boolean;
    isRecording?: boolean;
    showAddButton?: boolean;
    showModelSelector?: boolean;
    uploadedFiles?: ChatUploadedFile[];
    onFileUpload?: () => void;
    onMicClick?: () => void;
    onStop?: () => void;
    onSystemPromptClick?: () => void;
    onMcpPromptClick?: () => void;
    onMcpResourcesClick?: () => void;
  }

  let {
    canSend: vCanSend = false,
    canSubmit: vCanSubmit = false,
    class: vClassName = '',
    disabled: vDisabled = false,
    isLoading: vIsLoading = false,
    isReasoning: vIsReasoning = false,
    isRecording: vIsRecording = false,
    showAddButton: vShowAddButton = true,
    showModelSelector: vShowModelSelector = true,
    uploadedFiles: vUploadedFiles = [],
    onFileUpload: vOnFileUpload,
    onMicClick: vOnMicClick,
    onStop: vOnStop,
    onSystemPromptClick: vOnSystemPromptClick,
    onMcpPromptClick: vOnMcpPromptClick,
    onMcpResourcesClick: vOnMcpResourcesClick
  }: Props = $props();

  let vCurrentConfig = $derived(fConfig());

  let vHasMcpPromptsSupport = $derived.by(() => {
    const cPerChatOverrides = cConversationsStore.mGetAllMcpServerOverrides();

    return cMcpStore.mHasPromptsCapability(cPerChatOverrides);
  });

  let vHasMcpResourcesSupport = $derived.by(() => {
    const cPerChatOverrides = cConversationsStore.mGetAllMcpServerOverrides();

    return cMcpStore.mHasResourcesCapability(cPerChatOverrides);
  });

  let vHasAudioModality = $state(false);
  let vHasVideoModality = $state(false);
  let vHasVisionModality = $state(false);
  let vHasModelSelected = $state(false);
  let vIsSelectedModelInCache = $state(true);
  let vSubmitTooltip = $state('');

  let vHasAudioAttachments = $derived(
    vUploadedFiles.some((pFile) => getFileTypeCategory(pFile.type) === FileTypeCategory.AUDIO)
  );
  let vShouldShowRecordButton = $derived(
    vHasAudioModality && !vCanSubmit && !vHasAudioAttachments && vCurrentConfig.autoMicOnEmpty
  );

  let vSelectorModelRef: ChatFormActionModels | undefined = $state(undefined);

  export function fOpenModelSelector() {
    vSelectorModelRef?.fOpen();
  }
  // the streaming assistant message carries both the completion id and the model that
  // produced it, targeting reasoning control from the same source keeps them consistent
  let vActiveMessage = $derived(
    cConversationsStore.vActiveMessages[cConversationsStore.vActiveMessages.length - 1]
  );

  let vHasProcessedTokens = $derived.by(() => {
    if (!page.params.id) return false;

    const cMessages = fActiveMessages() as DatabaseMessage[];
    let vTotalHistoricalTokens = 0;
    for (const cM of cMessages) {
      if (cM.role !== MessageRole.ASSISTANT) continue;
      const cTimings = cM.timings;
      if (!cTimings) continue;
      const cAgenticLlm = cTimings.agentic?.llm;
      if (cAgenticLlm?.prompt_n != null || cAgenticLlm?.predicted_n != null) {
        vTotalHistoricalTokens += (cAgenticLlm?.prompt_n ?? 0) + (cAgenticLlm?.predicted_n ?? 0);
      } else {
        vTotalHistoricalTokens += (cTimings.prompt_n ?? 0) + (cTimings.predicted_n ?? 0);
      }
    }
    if (vTotalHistoricalTokens > 0) return true;

    if (!chatIsLoading() && !fIsChatStreaming()) return false;

    const cProcessingState = fActiveProcessingState();
    if (!cProcessingState) return false;
    const cLivePromptTokens = Math.max(
      cProcessingState.promptTokens ?? 0,
      cProcessingState.promptProgress?.processed ?? 0
    );
    const cLiveOutputTokens = cProcessingState.outputTokensUsed ?? 0;
    return cLivePromptTokens > 0 || cLiveOutputTokens > 0;
  });
</script>

<div
  class="flex w-full items-center gap-3 {vClassName} {vShowAddButton ? '' : 'justify-end'}"
  style="container-type: inline-size"
>
  {#if vShowAddButton}
    <div class="mr-auto flex items-center gap-2">
      <ChatFormActionsAdd
        disabled={vDisabled}
        hasAudioModality={vHasAudioModality}
        hasVideoModality={vHasVideoModality}
        hasVisionModality={vHasVisionModality}
        hasMcpPromptsSupport={vHasMcpPromptsSupport}
        hasMcpResourcesSupport={vHasMcpResourcesSupport}
        onFileUpload={vOnFileUpload}
        onSystemPromptClick={vOnSystemPromptClick}
        onMcpPromptClick={vOnMcpPromptClick}
        onMcpResourcesClick={vOnMcpResourcesClick}
        onMcpSettingsClick={() => goto(cRoutes.MCP_SERVERS)}
      />
    </div>
  {/if}

  <div class="flex items-center gap-1.5">
    {#if vHasProcessedTokens}
      <ChatFormContextGauge />
    {/if}

    {#if vShowModelSelector}
      <ChatFormActionModels
        disabled={vDisabled}
        bind:this={vSelectorModelRef}
        bind:hasAudioModality={vHasAudioModality}
        bind:hasVideoModality={vHasVideoModality}
        bind:hasVisionModality={vHasVisionModality}
        bind:hasModelSelected={vHasModelSelected}
        bind:isSelectedModelInCache={vIsSelectedModelInCache}
        bind:submitTooltip={vSubmitTooltip}
        forceForegroundText
        useGlobalSelection
      />
    {/if}
  </div>

  {#if vIsReasoning}
    <Button
      type="button"
      variant="secondary"
      onclick={() =>
        ChatService.mStopReasoning(vActiveMessage?.completionId ?? '', vActiveMessage?.model)}
      class="group h-8 w-8 rounded-full p-0"
      title={fTranslate('text_43ec555ebd2b')}
    >
      <span class="sr-only">{fTranslate('text_43ec555ebd2b')}</span>

      <SkipForward
        class="{cIconClassDefault} stroke-muted-foreground group-hover:stroke-foreground"
      />
    </Button>
  {/if}

  {#if vIsLoading && !vCanSubmit}
    <Button
      type="button"
      variant="secondary"
      onclick={vOnStop}
      class="group h-8 w-8 rounded-full p-0 hover:bg-destructive/10!"
    >
      <span class="sr-only">{fTranslate('text_cae7d57bc067')}</span>

      <Square
        class="h-8 w-8 fill-muted-foreground stroke-muted-foreground group-hover:fill-destructive group-hover:stroke-destructive hover:fill-destructive hover:stroke-destructive"
      />
    </Button>
  {:else if vShouldShowRecordButton}
    <ChatFormActionRecord
      disabled={vDisabled}
      hasAudioModality={vHasAudioModality}
      isLoading={vIsLoading}
      isRecording={vIsRecording}
      onMicClick={vOnMicClick}
    />
  {:else}
    <ChatFormActionSubmit
      canSend={vCanSend &&
        (vShowModelSelector ? vHasModelSelected && vIsSelectedModelInCache : true)}
      disabled={vDisabled}
      tooltipLabel={vSubmitTooltip}
      showErrorState={vShowModelSelector && vHasModelSelected && !vIsSelectedModelInCache}
    />
  {/if}
</div>
