<script lang="ts">
  import { page } from '$app/state';
  import {
    ChatScreenForm,
    ChatMessages,
    ChatScreenDragOverlay,
    ChatScreenStreamResumeStatus,
    ServerLoadingSplash,
    ChatScreenServerError
  } from '$lib/components/app';
  import { fCreateAutoScrollController } from '$lib/hooks/use-auto-scroll.svelte';
  import { fUseChatScreenActiveModel } from '$lib/hooks/use-chat-screen-active-model.svelte';
  import { fUseChatScreenDragAndDrop } from '$lib/hooks/use-chat-screen-drag-and-drop.svelte';
  import { fUseChatScreenFileUpload } from '$lib/hooks/use-chat-screen-file-upload.svelte';
  import { fUseChatScreenScroll } from '$lib/hooks/use-chat-screen-scroll.svelte';
  import { fUseKeyboardShortcuts } from '$lib/hooks/use-keyboard-shortcuts.svelte';
  import { cDevice } from '$lib/stores/device.svelte';
  import { cIsMobile } from '$lib/stores/viewport.svelte';
  import {
    cChatStore,
    fErrorDialog,
    fIsLoading,
    fIsChatStreaming,
    fIsEditing
  } from '$lib/stores/chat.svelte';
  import {
    cConversationsStore,
    fActiveMessages,
    fActiveConversation
  } from '$lib/stores/conversations.svelte';
  import { fConfig } from '$lib/stores/settings.svelte';
  import { fServerLoading, fServerError } from '$lib/stores/server.svelte';
  import { parseFilesToMessageExtras } from '$lib/utils/browser-only';
  import { onDestroy, onMount, tick } from 'svelte';
  import ChatScreenGreeting from './ChatScreenGreeting.svelte';
  import ChatScreenActionScrollDown from './ChatScreenActionScrollDown.svelte';
  import ChatScreenDialogsAndAlerts from './ChatScreenDialogsAndAlerts.svelte';
  import { cLandingSettleMaxMs, cLandingStableFrames, cRoutes } from '$lib/constants';

  let { showCenteredEmpty: vShowCenteredEmpty = false } = $props();

  let vDisableAutoScroll = $derived(Boolean(fConfig().disableAutoScroll) || cIsMobile.current);
  let vIsMobileUserScrolledUp = $state(false);
  let vMobileScrollDownHint = $state(false);
  let vMobileScrollDownHintLockedUntil = $state(0);
  let vEmptyFileNames = $state<string[]>([]);
  let vInitialMessage = $state('');
  let vShowDeleteDialog = $state(false);
  let vShowEmptyFileDialog = $state(false);
  let vIsEmpty = $derived(
    vShowCenteredEmpty && !fActiveConversation() && fActiveMessages().length === 0 && !fIsLoading()
  );
  let vActiveErrorDialog = $derived(fErrorDialog());
  let vIsServerLoading = $derived(fServerLoading());
  let vHasPropsError = $derived(!!fServerError());
  let vIsCurrentConversationLoading = $derived(fIsLoading() || fIsChatStreaming());
  // every offset sits above the status bar (--status-bar-height is 0 when it is hidden)
  let vChatFormBottomPosition = $derived.by(() => {
    if (!cIsMobile.current) return 'calc(1rem + var(--status-bar-height))';
    if (cDevice.isStandalone) return 'calc(1.5rem + var(--status-bar-height))';
    if (cDevice.isIOSSafari) return 'calc(0.25rem + var(--status-bar-height))';
    return 'calc(0.5rem + var(--status-bar-height))';
  });

  const cAutoScroll = fCreateAutoScrollController();
  const cScroll = fUseChatScreenScroll(cAutoScroll);
  const cActiveModel = fUseChatScreenActiveModel();
  const cFileUpload = fUseChatScreenFileUpload({
    capabilities: () => ({
      hasVision: cActiveModel.hasVisionModality,
      hasAudio: cActiveModel.hasAudioModality,
      hasVideo: cActiveModel.hasVideoModality
    }),
    activeModelId: () => cActiveModel.activeModelId
  });
  const cDragAndDrop = fUseChatScreenDragAndDrop({
    onDrop: cFileUpload.handleFileUpload
  });
  const { handleKeydown: cHandleKeydown } = fUseKeyboardShortcuts({
    deleteActiveConversation: () => {
      if (fActiveConversation()) {
        vShowDeleteDialog = true;
      }
    }
  });

  function fHandleMobileScroll() {
    if (!cIsMobile.current) return;

    const cContainer = cScroll.chatScrollContainer;
    if (!cContainer) return;

    const cDistanceFromBottom =
      cContainer.scrollHeight - cContainer.clientHeight - cContainer.scrollTop;
    vIsMobileUserScrolledUp = cDistanceFromBottom > 300;
  }

  async function fHandleDeleteConfirm() {
    const cConversation = fActiveConversation();

    if (cConversation) {
      await cConversationsStore.mDeleteConversation(cConversation.id);
    }

    vShowDeleteDialog = false;
  }

  async function fHandleSendMessage(
    pMessage: string,
    pFiles?: ChatUploadedFile[]
  ): Promise<boolean> {
    const cPlainFiles = pFiles ? $state.snapshot(pFiles) : undefined;
    const cResult = cPlainFiles
      ? await parseFilesToMessageExtras(cPlainFiles, cActiveModel.activeModelId ?? undefined)
      : undefined;

    if (cResult?.emptyFiles && cResult.emptyFiles.length > 0) {
      vEmptyFileNames = cResult.emptyFiles;
      vShowEmptyFileDialog = true;
      if (pFiles) {
        const sEmptyFileNamesSet = new Set(cResult.emptyFiles);
        cFileUpload.uploadedFiles = cFileUpload.uploadedFiles.filter(
          (pFile) => !sEmptyFileNamesSet.has(pFile.name)
        );
      }
      return false;
    }

    fHandleSendLikeScroll();

    await cChatStore.mSendMessage(pMessage, cResult?.extras);
    return true;
  }

  let vLastScrolledConversationId: string | null = null;

  // Lands at the bottom of a conversation the first time its messages
  // render, whether the route comes from another conversation or from a
  // non-conversation route. The page keeps growing after the first pin
  // without DOM mutations (content-visibility size realizations, syntax
  // highlight passes), so the instant pin repeats every frame until the
  // height settles, bailing out on user scroll or conversation change.
  async function fHandleMessagesReady(pMessageCount: number) {
    if (pMessageCount === 0) return;
    const cId = fActiveConversation()?.id ?? null;
    if (!cId || cId === vLastScrolledConversationId) return;
    vLastScrolledConversationId = cId;
    await tick();
    cAutoScroll.mScrollToBottom();

    const cContainer = cScroll.chatScrollContainer;
    if (!cContainer) return;
    const cStarted = performance.now();
    let vStableFrames = 0;
    let vLastHeight = cContainer.scrollHeight;
    const fSettle = () => {
      if (cAutoScroll.mUserScrolledUp) return;
      if (fActiveConversation()?.id !== cId) return;
      cAutoScroll.mScrollToBottom();
      const cHeight = cContainer.scrollHeight;
      vStableFrames = cHeight === vLastHeight ? vStableFrames + 1 : 0;
      vLastHeight = cHeight;
      if (vStableFrames >= cLandingStableFrames) return;
      if (performance.now() - cStarted > cLandingSettleMaxMs) return;
      requestAnimationFrame(fSettle);
    };
    requestAnimationFrame(fSettle);
  }

  function fHandleSendLikeScroll() {
    if (!cIsMobile.current) {
      cAutoScroll.mEnable();
    }

    setTimeout(() => {
      const cContainer = cScroll.chatScrollContainer;
      if (!cContainer) return;

      const cLastUserBubble = cContainer.querySelector(
        '.chat-message:nth-last-child(2) .chat-message-user .chat-message-user-bubble'
      ) as HTMLElement | null;

      if (cIsMobile.current) {
        // Keep the last user message bubble just above the input on mobile
        const cBubbleHeight = cLastUserBubble?.scrollHeight ?? 0;
        const cBaseHeight = cContainer.scrollHeight - innerHeight;

        cContainer.scrollTo({
          top: cBubbleHeight > 0 ? cBaseHeight - cBubbleHeight : cBaseHeight,
          behavior: 'smooth'
        });
      } else if (cLastUserBubble) {
        // On desktop, place the last user message near the top of the viewport
        const cTopPadding = 24;
        const cBubbleRect = cLastUserBubble.getBoundingClientRect();
        cContainer.scrollTo({
          top: Math.max(0, cContainer.scrollTop + cBubbleRect.top - cTopPadding),
          behavior: 'smooth'
        });
      } else {
        cAutoScroll.mScrollToBottom();
      }
    }, 100);

    if (cIsMobile.current) {
      cAutoScroll.mSetDisabled(vDisableAutoScroll);
      vMobileScrollDownHint = true;
      vMobileScrollDownHintLockedUntil = Date.now() + 500;
    }
  }

  function fHandleErrorDialogOpenChange(pOpen: boolean) {
    if (!pOpen) {
      cChatStore.mDismissErrorDialog();
    }
  }

  async function fHandleSystemPromptAdd(pDraft: { message: string; files: ChatUploadedFile[] }) {
    if (pDraft.message || pDraft.files.length > 0) {
      cChatStore.mSavePendingDraft(pDraft.message, pDraft.files);
    }
    await cChatStore.mAddSystemPrompt();
  }

  $effect(() => {
    const cShouldDisableAutoScroll =
      fConfig().disableAutoScroll || (cIsMobile.current && vIsCurrentConversationLoading);
    cAutoScroll.mSetDisabled(cShouldDisableAutoScroll);
    if (!cShouldDisableAutoScroll) {
      cAutoScroll.mEnable();
    }
  });

  onMount(() => {
    const cPendingDraft = cChatStore.mConsumePendingDraft();
    if (cPendingDraft) {
      vInitialMessage = cPendingDraft.message;
      cFileUpload.uploadedFiles = cPendingDraft.files;
    }

    cAutoScroll.mStartObserving();

    if (!vDisableAutoScroll) {
      cAutoScroll.mEnable();
    }

    if (cIsMobile.current && vIsCurrentConversationLoading) {
      vMobileScrollDownHint = true;
      vMobileScrollDownHintLockedUntil = Date.now() + 500;
    }

    fHandleMobileScroll();
  });

  onDestroy(() => cAutoScroll.mDestroy());
</script>

{#if cDragAndDrop.isDragOver}
  <ChatScreenDragOverlay />
{/if}

<svelte:window
  onkeydown={cHandleKeydown}
  onscroll={(e) => {
    cScroll.handleScroll(e);
    fHandleMobileScroll();
    if (e.isTrusted && Date.now() > vMobileScrollDownHintLockedUntil) {
      vMobileScrollDownHint = false;
    }
  }}
/>

{#if vIsServerLoading}
  <ServerLoadingSplash />
{:else}
  <div
    class="chat-screen flex grow flex-col min-h-[calc(100dvh-1rem-var(--status-bar-height))] md:min-h-full px-4 md:py-0 pt-12 pb-[calc(12rem+var(--status-bar-height))] md:pb-4"
    style:--chat-form-bottom-position={vChatFormBottomPosition}
    ondragenter={cDragAndDrop.dragHandlers.dragenter}
    ondragleave={cDragAndDrop.dragHandlers.dragleave}
    ondragover={cDragAndDrop.dragHandlers.dragover}
    ondrop={cDragAndDrop.dragHandlers.drop}
    role="main"
  >
    {#if !vIsEmpty}
      <ChatMessages
        messages={fActiveMessages()}
        onMessagesReady={fHandleMessagesReady}
        onUserAction={() => {
          fHandleSendLikeScroll();
        }}
      />
    {/if}

    <div
      class={[
        'pointer-events-none md:sticky fixed  mt-auto transition-all duration-200',
        cDevice.isStandalone
          ? 'bottom-[calc(1.5rem+var(--status-bar-height))] right-4 left-4'
          : cDevice.isIOSSafari
            ? 'bottom-[calc(0.25rem+var(--status-bar-height))] left-2 right-2'
            : 'bottom-[calc(0.5rem+var(--status-bar-height))] right-2 left-2',
        vIsEmpty
          ? 'md:bottom-[calc(50dvh-7rem)] 2xl:bottom-[calc(50dvh-4rem)]'
          : 'md:bottom-[calc(1rem+var(--status-bar-height))]'
      ]}
      style:padding-top={!vIsEmpty ? 'var(--chat-form-padding-top)' : undefined}
    >
      <ChatScreenGreeting isEmpty={vIsEmpty} />

      <ChatScreenServerError />

      {#if page.params.id}
        <ChatScreenStreamResumeStatus />
      {/if}

      <div class="pointer-events-none flex flex-col gap-6 items-center w-full">
        {#if (cIsMobile.current ? vMobileScrollDownHint || vIsMobileUserScrolledUp : cAutoScroll.mUserScrolledUp) && page.url.hash.includes(cRoutes.CHAT) && page.params.id}
          <ChatScreenActionScrollDown
            onclick={() => {
              vMobileScrollDownHint = false;
              cScroll.chatScrollContainer?.scrollTo({
                top: cScroll.chatScrollContainer.scrollHeight,
                behavior: 'smooth'
              });
            }}
          />
        {/if}
      </div>

      <ChatScreenForm
        class="pointer-events-auto conversation-chat-form"
        disabled={vHasPropsError || fIsEditing()}
        initialMessage={vInitialMessage}
        isLoading={vIsCurrentConversationLoading}
        onFileRemove={cFileUpload.handleFileRemove}
        onFileUpload={cFileUpload.handleFileUpload}
        onSend={fHandleSendMessage}
        onStop={() => cChatStore.mStopGeneration()}
        onSystemPromptAdd={fHandleSystemPromptAdd}
        bind:uploadedFiles={cFileUpload.uploadedFiles}
      />
    </div>
  </div>
{/if}

<ChatScreenDialogsAndAlerts
  showDeleteDialog={vShowDeleteDialog}
  handleDeleteConfirm={fHandleDeleteConfirm}
  showEmptyFileDialog={vShowEmptyFileDialog}
  emptyFileNames={vEmptyFileNames}
  activeErrorDialog={vActiveErrorDialog}
  handleErrorDialogOpenChange={fHandleErrorDialogOpenChange}
  fileUpload={cFileUpload}
/>
