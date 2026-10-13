<script lang="ts">
  import { afterNavigate } from '$app/navigation';
  import { page } from '$app/state';
  import { ChatForm } from '$lib/components/app';
  import { cIsMobile } from '$lib/stores/viewport.svelte';
  import { onMount } from 'svelte';
  import { fUseDraftMessages } from '$lib/hooks/use-draft-messages.svelte';

  interface Props {
    class?: string;
    disabled?: boolean;
    initialMessage?: string;
    isLoading?: boolean;
    onFileRemove?: (pFileId: string) => void;
    onFileUpload?: (pFiles: File[]) => void;
    onSend?: (pMessage: string, pFiles?: ChatUploadedFile[]) => Promise<boolean>;
    onStop?: () => void;
    onSystemPromptAdd?: (pDraft: { message: string; files: ChatUploadedFile[] }) => void;
    uploadedFiles?: ChatUploadedFile[];
  }

  let {
    class: vClassName,
    disabled: vDisabled = false,
    initialMessage: vInitialMessage = '',
    isLoading: vIsLoading = false,
    onFileRemove: vOnFileRemove,
    onFileUpload: vOnFileUpload,
    onSend: vOnSend,
    onStop: vOnStop,
    onSystemPromptAdd: vOnSystemPromptAdd,
    uploadedFiles: vUploadedFiles = $bindable([])
  }: Props = $props();

  let vChatFormRef: ChatForm | undefined = $state(undefined);
  let vFormWrapperEl: HTMLDivElement | undefined = $state();
  let vChatId = $derived(page.params.id as string | undefined);

  $effect(() => {
    if (!vFormWrapperEl) return;

    const cFormEl = vFormWrapperEl.querySelector('form') as HTMLElement | null;
    if (!cFormEl) return;

    const fUpdateHeight = () => {
      const cHeight = Math.round(cFormEl.getBoundingClientRect().height);
      document.documentElement.style.setProperty('--chat-form-height', `${cHeight}px`);
    };

    fUpdateHeight();

    const cResizeObserver = new ResizeObserver(fUpdateHeight);
    cResizeObserver.observe(cFormEl);

    return () => {
      cResizeObserver.disconnect();
      document.documentElement.style.removeProperty('--chat-form-height');
    };
  });
  let vHasLoadingAttachments = $derived(vUploadedFiles.some((pF) => pF.isLoading));
  let vMessage = $derived(vInitialMessage);
  let vPreviousIsLoading = $derived(vIsLoading);
  let vPreviousInitialMessage = $derived(vInitialMessage);

  const { clearDraft: cClearDraft } = fUseDraftMessages({
    getChatId: () => vChatId,
    getMessage: () => vMessage,
    getFiles: () => vUploadedFiles,
    setMessage: (pM) => (vMessage = pM),
    setFiles: (pF) => (vUploadedFiles = pF),
    getInitialMessage: () => vInitialMessage
  });

  function fHandleFilesAdd(pFiles: File[]) {
    vOnFileUpload?.(pFiles);
  }

  async function fHandleSubmit() {
    if ((!vMessage.trim() && vUploadedFiles.length === 0) || vDisabled || vHasLoadingAttachments)
      return;

    if (!vChatFormRef?.fCheckModelSelected()) return;

    const cMessageToSend = vMessage.trim();
    const aFilesToSend = [...vUploadedFiles];

    vMessage = '';
    vUploadedFiles = [];
    cClearDraft();

    vChatFormRef?.fResetTextareaHeight();

    const cSuccess = await vOnSend?.(cMessageToSend, aFilesToSend);

    if (!cSuccess) {
      vMessage = cMessageToSend;
      vUploadedFiles = aFilesToSend;
    }
  }

  function fHandleSystemPromptClick() {
    vOnSystemPromptAdd?.({ message: vMessage, files: vUploadedFiles });
  }

  function fHandleUploadedFileRemove(pFileId: string) {
    vOnFileRemove?.(pFileId);
  }

  // Auto-focus must not steal focus already claimed elsewhere (e.g. the system
  // message editor opened just before a navigation)
  function fFocusFormUnlessCaptured() {
    const cActive = document.activeElement;
    if (cActive instanceof HTMLTextAreaElement || cActive instanceof HTMLInputElement) return;
    vChatFormRef?.fFocus();
  }

  onMount(() => {
    if (!cIsMobile.current) {
      setTimeout(fFocusFormUnlessCaptured, 100);
    }
  });

  afterNavigate((pNavigation) => {
    if (pNavigation?.from != null && !cIsMobile.current) {
      setTimeout(fFocusFormUnlessCaptured, 100);
    }
  });

  $effect(() => {
    if (vInitialMessage !== vPreviousInitialMessage) {
      vMessage = vInitialMessage;
      vPreviousInitialMessage = vInitialMessage;
    }
  });

  $effect(() => {
    if (vPreviousIsLoading && !vIsLoading) {
      setTimeout(fFocusFormUnlessCaptured, 10);
    }

    vPreviousIsLoading = vIsLoading;
  });
</script>

<div class="chat-screen-form-wrapper" bind:this={vFormWrapperEl}>
  <ChatForm
    class="mx-auto max-w-3xl {vClassName}"
    bind:this={vChatFormRef}
    bind:value={vMessage}
    bind:uploadedFiles={vUploadedFiles}
    disabled={vDisabled}
    isLoading={vIsLoading}
    showMcpPromptButton
    onFilesAdd={fHandleFilesAdd}
    onStop={vOnStop}
    onSubmit={fHandleSubmit}
    onSystemPromptClick={fHandleSystemPromptClick}
    onUploadedFileRemove={fHandleUploadedFileRemove}
  />
</div>
