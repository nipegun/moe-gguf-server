<script lang="ts">
  import { fTranslate } from '../../../../i18n';

  import {
    ChatAttachmentsList,
    ChatFormActions,
    ChatFormFileInputInvisible,
    ChatFormMcpResourcesList,
    ChatFormPickers,
    ChatFormTextarea,
    DialogMcpResourcesBrowser
  } from '$lib/components/app';
  import {
    cClipboardContentQuotePrefix,
    cInputClasses,
    dSettingConfigDefault,
    cInitialFileSize,
    cPromptContentSeparator,
    cPromptTriggerPrefix,
    cResourceTriggerPrefix
  } from '$lib/constants';
  import {
    ContentPartType,
    FileExtensionText,
    KeyboardKey,
    MimeTypeText,
    SpecialFileType
  } from '$lib/enums';
  import { fConfig } from '$lib/stores/settings.svelte';
  import ContextGaugePopup from './ChatFormContextGauge/ContextGaugePopup.svelte';
  import { fModelOptions, fSelectedModelId } from '$lib/stores/models.svelte';
  import { fIsRouterMode } from '$lib/stores/server.svelte';
  import { cChatStore } from '$lib/stores/chat.svelte';
  import { cMcpStore } from '$lib/stores/mcp.svelte';
  import { fMcpHasResourceAttachments } from '$lib/stores/mcp-resources.svelte';
  import { cConversationsStore, fActiveMessages } from '$lib/stores/conversations.svelte';
  import type { GetPromptResult, MCPPromptInfo, MCPResourceInfo, PromptMessage } from '$lib/types';
  import { isIMEComposing, parseClipboardContent, uuid } from '$lib/utils';
  import {
    AudioRecorder,
    convertToWav,
    createAudioFile,
    isAudioRecordingSupported
  } from '$lib/utils/browser-only';
  import { onMount } from 'svelte';

  interface Props {
    // Data
    attachments?: DatabaseMessageExtra[];
    uploadedFiles?: ChatUploadedFile[];
    value?: string;

    // UI State
    class?: string;
    disabled?: boolean;
    isLoading?: boolean;
    placeholder?: string;
    showMcpPromptButton?: boolean;
    showAddButton?: boolean;
    showModelSelector?: boolean;

    // Event Handlers
    onAttachmentRemove?: (pIndex: number) => void;
    onFilesAdd?: (pFiles: File[]) => void;
    onStop?: () => void;
    onSubmit?: () => void;
    onSystemPromptClick?: (pDraft: { message: string; files: ChatUploadedFile[] }) => void;
    onUploadedFileRemove?: (pFileId: string) => void;
    onUploadedFilesChange?: (pFiles: ChatUploadedFile[]) => void;
    onValueChange?: (pValue: string) => void;
  }

  let {
    attachments: vAttachments = [],
    class: vClassName = '',
    disabled: vDisabled = false,
    isLoading: vIsLoading = false,
    placeholder: vPlaceholder = fTranslate('text_69518e684f06'),
    showMcpPromptButton: vShowMcpPromptButton = false,
    showAddButton: vShowAddButton = true,
    showModelSelector: vShowModelSelector = true,
    uploadedFiles: vUploadedFiles = $bindable([]),
    value: vValue = $bindable(''),
    onAttachmentRemove: vOnAttachmentRemove,
    onFilesAdd: vOnFilesAdd,
    onStop: vOnStop,
    onSubmit: vOnSubmit,
    onSystemPromptClick: vOnSystemPromptClick,
    onUploadedFileRemove: vOnUploadedFileRemove,
    onUploadedFilesChange: vOnUploadedFilesChange,
    onValueChange: vOnValueChange
  }: Props = $props();

  // Component References
  let vAudioRecorder: AudioRecorder | undefined;
  let vChatFormActionsRef: ChatFormActions | undefined = $state(undefined);
  let vFileInputRef: ChatFormFileInputInvisible | undefined = $state(undefined);
  let vPickersRef: { fHandleKeydown: (pEvent: KeyboardEvent) => boolean } | undefined =
    $state(undefined);
  let vTextareaRef: ChatFormTextarea | undefined = $state(undefined);

  // Audio Recording State
  let vIsRecording = $state(false);
  let vRecordingSupported = $state(false);

  // Picker State
  let vIsPromptPickerOpen = $state(false);
  let vPromptSearchQuery = $state('');
  let vIsInlineResourcePickerOpen = $state(false);
  let vResourceSearchQuery = $state('');

  // Resource Dialog State
  let vIsResourceDialogOpen = $state(false);
  let vPreSelectedResourceUri = $state<string | undefined>(undefined);

  let vCurrentConfig = $derived(fConfig());

  let vPasteLongTextToFileLength = $derived.by(() => {
    const cN = Number(vCurrentConfig.pasteLongTextToFileLen);
    return Number.isNaN(cN) ? Number(dSettingConfigDefault.pasteLongTextToFileLen) : cN;
  });

  let vIsRouter = $derived(fIsRouterMode());
  let vConversationModel = $derived(
    cChatStore.mGetConversationModel(fActiveMessages() as DatabaseMessage[])
  );
  let vActiveModelId = $derived.by(() => {
    const cOptions = fModelOptions();

    if (!vIsRouter) {
      return cOptions.length > 0 ? cOptions[0].model : null;
    }

    const cSelectedId = fSelectedModelId();
    if (cSelectedId) {
      const cModel = cOptions.find((pM) => pM.id === cSelectedId);
      if (cModel) return cModel.model;
    }

    if (vConversationModel) {
      const cModel = cOptions.find((pM) => pM.model === vConversationModel);
      if (cModel) return cModel.model;
    }

    return null;
  });

  let vHasModelSelected = $derived(!vIsRouter || !!vConversationModel || !!fSelectedModelId());
  let vHasLoadingAttachments = $derived(vUploadedFiles.some((pF) => pF.isLoading));
  let vHasAttachments = $derived(
    (vAttachments && vAttachments.length > 0) || (vUploadedFiles && vUploadedFiles.length > 0)
  );
  let vCanSubmit = $derived(vValue.trim().length > 0 || vHasAttachments);

  onMount(() => {
    vRecordingSupported = isAudioRecordingSupported();
    vAudioRecorder = new AudioRecorder();
  });

  export function fFocus() {
    vTextareaRef?.fFocus();
  }

  export function fResetTextareaHeight() {
    vTextareaRef?.fResetHeight();
  }

  export function fOpenModelSelector() {
    vChatFormActionsRef?.fOpenModelSelector();
  }

  export function fCheckModelSelected(): boolean {
    if (!vHasModelSelected) {
      vChatFormActionsRef?.fOpenModelSelector();
      return false;
    }
    return true;
  }

  function fHandleFileSelect(pFiles: File[]) {
    vOnFilesAdd?.(pFiles);
  }

  function fHandleFileUpload() {
    vFileInputRef?.fClick();
  }

  function fHandleFileRemove(pFileId: string) {
    if (pFileId.startsWith('attachment-')) {
      const cIndex = parseInt(pFileId.replace('attachment-', ''), 10);
      if (!isNaN(cIndex) && cIndex >= 0 && cIndex < vAttachments.length) {
        vOnAttachmentRemove?.(cIndex);
      }
    } else {
      vOnUploadedFileRemove?.(pFileId);
    }
  }

  function fHandleInput() {
    const cPerChatOverrides = cConversationsStore.mGetAllMcpServerOverrides();
    const cHasServers = cMcpStore.mHasEnabledServers(cPerChatOverrides);

    if (vValue.startsWith(cPromptTriggerPrefix) && cHasServers) {
      vIsPromptPickerOpen = true;
      vPromptSearchQuery = vValue.slice(1);
      vIsInlineResourcePickerOpen = false;
      vResourceSearchQuery = '';
    } else if (
      vValue.startsWith(cResourceTriggerPrefix) &&
      cHasServers &&
      cMcpStore.mHasResourcesCapability(cPerChatOverrides)
    ) {
      vIsInlineResourcePickerOpen = true;
      vResourceSearchQuery = vValue.slice(1);
      vIsPromptPickerOpen = false;
      vPromptSearchQuery = '';
    } else {
      vIsPromptPickerOpen = false;
      vPromptSearchQuery = '';
      vIsInlineResourcePickerOpen = false;
      vResourceSearchQuery = '';
    }
  }

  function fHandleKeydown(pEvent: KeyboardEvent) {
    if (vPickersRef?.fHandleKeydown(pEvent)) {
      return;
    }

    if (pEvent.key === KeyboardKey.ESCAPE && vIsPromptPickerOpen) {
      vIsPromptPickerOpen = false;
      vPromptSearchQuery = '';
      return;
    }

    if (pEvent.key === KeyboardKey.ESCAPE && vIsInlineResourcePickerOpen) {
      vIsInlineResourcePickerOpen = false;
      vResourceSearchQuery = '';
      return;
    }

    if (pEvent.key === KeyboardKey.ENTER && !pEvent.shiftKey && !isIMEComposing(pEvent)) {
      const cIsModifier = pEvent.ctrlKey || pEvent.metaKey;
      const cSendOnEnter = vCurrentConfig.sendOnEnter !== false;

      if (cSendOnEnter || cIsModifier) {
        pEvent.preventDefault();

        if (!vCanSubmit || vDisabled || vHasLoadingAttachments) return;

        vOnSubmit?.();
      }
    }
  }

  function fHandlePaste(pEvent: ClipboardEvent) {
    if (!pEvent.clipboardData) return;

    const cFiles = Array.from(pEvent.clipboardData.items)
      .filter((pItem) => pItem.kind === 'file')
      .map((pItem) => pItem.getAsFile())
      .filter((pFile): pFile is File => pFile !== null);

    if (cFiles.length > 0) {
      pEvent.preventDefault();
      vOnFilesAdd?.(cFiles);
      return;
    }

    const cText = pEvent.clipboardData.getData(MimeTypeText.PLAIN);

    if (cText.startsWith(cClipboardContentQuotePrefix)) {
      const cParsed = parseClipboardContent(cText);

      if (cParsed.textAttachments.length > 0 || cParsed.mcpPromptAttachments.length > 0) {
        pEvent.preventDefault();
        vValue = cParsed.message;
        vOnValueChange?.(cParsed.message);

        // Handle text attachments as files
        if (cParsed.textAttachments.length > 0) {
          const cAttachmentFiles = cParsed.textAttachments.map(
            (pAtt) =>
              new File([pAtt.content], pAtt.name, {
                type: MimeTypeText.PLAIN
              })
          );
          vOnFilesAdd?.(cAttachmentFiles);
        }

        // Handle MCP prompt attachments as ChatUploadedFile with mcpPrompt data
        if (cParsed.mcpPromptAttachments.length > 0) {
          const aMcpPromptFiles: ChatUploadedFile[] = cParsed.mcpPromptAttachments.map((pAtt) => ({
            id: uuid(),
            name: pAtt.name,
            size: pAtt.content.length,
            type: SpecialFileType.MCP_PROMPT,
            file: new File([pAtt.content], `${pAtt.name}${FileExtensionText.TXT}`, {
              type: MimeTypeText.PLAIN
            }),
            isLoading: false,
            textContent: pAtt.content,
            mcpPrompt: {
              serverName: pAtt.serverName,
              promptName: pAtt.promptName,
              arguments: pAtt.arguments
            }
          }));

          vUploadedFiles = [...vUploadedFiles, ...aMcpPromptFiles];
          vOnUploadedFilesChange?.(vUploadedFiles);
        }

        setTimeout(() => {
          vTextareaRef?.fFocus();
        }, 10);

        return;
      }
    }

    if (
      cText.length > 0 &&
      vPasteLongTextToFileLength > 0 &&
      cText.length > vPasteLongTextToFileLength
    ) {
      pEvent.preventDefault();

      const cTextFile = new File([cText], 'Pasted', {
        type: MimeTypeText.PLAIN
      });

      vOnFilesAdd?.([cTextFile]);
    }
  }

  function fHandlePromptLoadStart(
    pPlaceholderId: string,
    pPromptInfo: MCPPromptInfo,
    pArgs?: Record<string, string>
  ) {
    // Only clear the value if the prompt was triggered by typing '/'
    if (vValue.startsWith(cPromptTriggerPrefix)) {
      vValue = '';
      vOnValueChange?.('');
    }
    vIsPromptPickerOpen = false;
    vPromptSearchQuery = '';

    const cPromptName = pPromptInfo.title || pPromptInfo.name;
    const dPlaceholder: ChatUploadedFile = {
      id: pPlaceholderId,
      name: cPromptName,
      size: cInitialFileSize,
      type: SpecialFileType.MCP_PROMPT,
      file: new File([], 'loading'),
      isLoading: true,
      mcpPrompt: {
        serverName: pPromptInfo.serverName,
        promptName: pPromptInfo.name,
        arguments: pArgs ? { ...pArgs } : undefined
      }
    };

    vUploadedFiles = [...vUploadedFiles, dPlaceholder];
    vOnUploadedFilesChange?.(vUploadedFiles);
    vTextareaRef?.fFocus();
  }

  function fHandlePromptLoadComplete(pPlaceholderId: string, pResult: GetPromptResult) {
    const cPromptText = pResult.messages
      ?.map((pMsg: PromptMessage) => {
        if (typeof pMsg.content === 'string') {
          return pMsg.content;
        }

        if (pMsg.content.type === ContentPartType.TEXT) {
          return pMsg.content.text;
        }

        return '';
      })
      .filter(Boolean)
      .join(cPromptContentSeparator);

    vUploadedFiles = vUploadedFiles.map((pF) =>
      pF.id === pPlaceholderId
        ? {
            ...pF,
            isLoading: false,
            textContent: cPromptText,
            size: cPromptText.length,
            file: new File([cPromptText], `${pF.name}${FileExtensionText.TXT}`, {
              type: MimeTypeText.PLAIN
            })
          }
        : pF
    );
    vOnUploadedFilesChange?.(vUploadedFiles);
  }

  function fHandlePromptLoadError(pPlaceholderId: string, pError: string) {
    vUploadedFiles = vUploadedFiles.map((pF) =>
      pF.id === pPlaceholderId ? { ...pF, isLoading: false, loadError: pError } : pF
    );
    vOnUploadedFilesChange?.(vUploadedFiles);
  }

  function fHandlePromptPickerClose() {
    vIsPromptPickerOpen = false;
    vPromptSearchQuery = '';
    vTextareaRef?.fFocus();
  }

  function fHandleInlineResourcePickerClose() {
    vIsInlineResourcePickerOpen = false;
    vResourceSearchQuery = '';
    vTextareaRef?.fFocus();
  }

  function fHandleInlineResourceSelect() {
    if (vValue.startsWith(cResourceTriggerPrefix)) {
      vValue = '';
      vOnValueChange?.('');
    }

    vIsInlineResourcePickerOpen = false;
    vResourceSearchQuery = '';
    vTextareaRef?.fFocus();
  }

  function fHandleBrowseResources() {
    vIsInlineResourcePickerOpen = false;
    vResourceSearchQuery = '';

    if (vValue.startsWith(cResourceTriggerPrefix)) {
      vValue = '';
      vOnValueChange?.('');
    }

    vIsResourceDialogOpen = true;
  }

  async function fHandleMicClick() {
    if (!vAudioRecorder || !vRecordingSupported) {
      console.warn('Audio recording not supported');
      return;
    }

    if (vIsRecording) {
      vIsRecording = false;
      try {
        const cAudioBlob = await vAudioRecorder.mStopRecording();
        const cWavBlob = await convertToWav(cAudioBlob);
        const cAudioFile = createAudioFile(cWavBlob);

        vOnFilesAdd?.([cAudioFile]);
      } catch (vError) {
        console.error('Failed to stop recording:', vError);
      }
    } else {
      try {
        await vAudioRecorder.mStartRecording();
        vIsRecording = true;
      } catch (vError) {
        console.error('Failed to start recording:', vError);
      }
    }
  }
</script>

<ChatFormFileInputInvisible bind:this={vFileInputRef} onFileSelect={fHandleFileSelect} />

<form
  class="relative {vClassName}"
  onsubmit={(event) => {
    event.preventDefault();

    if (!vCanSubmit || vDisabled || vHasLoadingAttachments) return;

    vOnSubmit?.();
  }}
>
  <ChatFormPickers
    bind:this={vPickersRef}
    isPromptPickerOpen={vIsPromptPickerOpen}
    promptSearchQuery={vPromptSearchQuery}
    isInlineResourcePickerOpen={vIsInlineResourcePickerOpen}
    resourceSearchQuery={vResourceSearchQuery}
    onPromptPickerClose={fHandlePromptPickerClose}
    onInlineResourcePickerClose={fHandleInlineResourcePickerClose}
    onInlineResourceSelect={fHandleInlineResourceSelect}
    onPromptLoadStart={fHandlePromptLoadStart}
    onPromptLoadComplete={fHandlePromptLoadComplete}
    onPromptLoadError={fHandlePromptLoadError}
    onInlineResourceBrowse={fHandleBrowseResources}
  />

  <div
    class="{cInputClasses} overflow-hidden rounded-4xl md:rounded-3xl backdrop-blur-md {vDisabled
      ? 'cursor-not-allowed opacity-60'
      : ''}"
    data-slot="input-area"
  >
    <ChatAttachmentsList
      attachments={vAttachments}
      bind:uploadedFiles={vUploadedFiles}
      onFileRemove={fHandleFileRemove}
      limitToSingleRow
      class="py-5"
      style="scroll-padding: 1rem;"
      activeModelId={vActiveModelId ?? undefined}
    />

    <div
      class="flex-column relative min-h-12 items-center rounded-4xl md:rounded-3xl py-2 pb-2.25 shadow-sm transition-all focus-within:shadow-md md:py-3!"
      onpaste={fHandlePaste}
    >
      <ChatFormTextarea
        class="px-5 py-1.5 md:pt-0"
        bind:this={vTextareaRef}
        bind:value={vValue}
        onKeydown={fHandleKeydown}
        onInput={() => {
          fHandleInput();
          vOnValueChange?.(vValue);
        }}
        disabled={vDisabled}
        placeholder={vPlaceholder}
      />

      {#if fMcpHasResourceAttachments()}
        <ChatFormMcpResourcesList
          class="mb-3"
          onResourceClick={(uri) => {
            vPreSelectedResourceUri = uri;
            vIsResourceDialogOpen = true;
          }}
        />
      {/if}

      <ChatFormActions
        class="px-3"
        bind:this={vChatFormActionsRef}
        canSend={vCanSubmit}
        disabled={vDisabled}
        isLoading={vIsLoading}
        isReasoning={cChatStore.vIsReasoning}
        isRecording={vIsRecording}
        showAddButton={vShowAddButton}
        showModelSelector={vShowModelSelector}
        uploadedFiles={vUploadedFiles}
        onFileUpload={fHandleFileUpload}
        onMicClick={fHandleMicClick}
        onStop={vOnStop}
        onSystemPromptClick={() =>
          vOnSystemPromptClick?.({ message: vValue, files: vUploadedFiles })}
        onMcpPromptClick={vShowMcpPromptButton ? () => (vIsPromptPickerOpen = true) : undefined}
        onMcpResourcesClick={() => (vIsResourceDialogOpen = true)}
      />
    </div>
  </div>

  <ContextGaugePopup />
</form>

<DialogMcpResourcesBrowser
  bind:open={vIsResourceDialogOpen}
  preSelectedUri={vPreSelectedResourceUri}
  onAttach={(resource: MCPResourceInfo) => {
    cMcpStore.mAttachResource(resource.uri);
  }}
  onOpenChange={(newOpen: boolean) => {
    if (!newOpen) {
      vPreSelectedResourceUri = undefined;
    }
  }}
/>
