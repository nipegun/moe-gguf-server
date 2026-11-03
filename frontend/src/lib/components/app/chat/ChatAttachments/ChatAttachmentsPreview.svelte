<script lang="ts">
  import { fTranslate } from '../../../../i18n';

  import {
    ChatAttachmentsPreviewCurrentItem,
    ChatAttachmentsPreviewFileInfo,
    ChatAttachmentsPreviewNavButtons,
    ChatAttachmentsPreviewThumbnailStrip
  } from '$lib/components/app';
  import { cModelsStore } from '$lib/stores/models.svelte';
  import {
    fCreateBase64DataUrl,
    fFormatFileSize,
    fGetAttachmentDisplayItems,
    fGetLanguageFromFilename,
    fIsAudioFile,
    fIsVideoFile,
    fIsImageFile,
    fIsMcpPrompt,
    fIsMcpResource,
    fIsPdfFile,
    fIsTextFile
  } from '$lib/utils';

  interface PreviewItem {
    id: string;
    name: string;
    size?: number;
    preview?: string;
    uploadedFile?: ChatUploadedFile;
    attachment?: DatabaseMessageExtra;
    textContent?: string;
    isImage: boolean;
    isAudio: boolean;
    isVideo: boolean;
  }

  interface Props {
    uploadedFiles?: ChatUploadedFile[];
    attachments?: DatabaseMessageExtra[];
    activeModelId?: string;
    class?: string;
    previewFocusIndex?: number;
  }

  let {
    uploadedFiles: vUploadedFiles = [],
    attachments: vAttachments = [],
    activeModelId: vActiveModelId,
    class: vClassName = '',
    previewFocusIndex: vPreviewFocusIndex = 0
  }: Props = $props();

  let vAllItems = $derived(
    fGetAttachmentDisplayItems({ uploadedFiles: vUploadedFiles, attachments: vAttachments })
      .filter((pItem) => !fIsMcpPrompt(pItem) && !fIsMcpResource(pItem))
      .map(
        (pItem): PreviewItem => ({
          ...pItem,
          isImage: fIsImageFile(pItem.attachment, pItem.uploadedFile),
          isAudio: fIsAudioFile(pItem.attachment, pItem.uploadedFile),
          isVideo: fIsVideoFile(pItem.attachment, pItem.uploadedFile)
        })
      )
  );

  let vCurrentIndex = $state(0);

  $effect(() => {
    if (vPreviewFocusIndex >= 0 && vPreviewFocusIndex < vAllItems.length) {
      vCurrentIndex = vPreviewFocusIndex;
    }
  });

  $effect(() => {
    const fHandler = (pE: Event) => {
      const cDelta = (pE as CustomEvent).detail;

      if (cDelta < 0) {
        vCurrentIndex = vCurrentIndex > 0 ? vCurrentIndex - 1 : vAllItems.length - 1;
      } else {
        vCurrentIndex = vCurrentIndex < vAllItems.length - 1 ? vCurrentIndex + 1 : 0;
      }
    };

    document.addEventListener('chat-attachments-nav', fHandler);

    return () => document.removeEventListener('chat-attachments-nav', fHandler);
  });

  $effect(() => {
    const cIndex = vCurrentIndex;
    setTimeout(() => {
      const cThumbnail = document.querySelector(`[data-thumbnail-index="${cIndex}"]`);

      cThumbnail?.scrollIntoView({ behavior: 'smooth', inline: 'center', block: 'nearest' });
    }, 0);
  });

  let vCurrentItem = $derived(vAllItems[vCurrentIndex] ?? null);
  let vDisplayName = $derived(
    vCurrentItem?.name ||
      vCurrentItem?.uploadedFile?.name ||
      vCurrentItem?.attachment?.name ||
      fTranslate('text_5f19db9fcb9b')
  );
  let vIsAudio = $derived(
    vCurrentItem ? fIsAudioFile(vCurrentItem.attachment, vCurrentItem.uploadedFile) : false
  );
  let vIsVideo = $derived(
    vCurrentItem ? fIsVideoFile(vCurrentItem.attachment, vCurrentItem.uploadedFile) : false
  );
  let vIsImage = $derived(
    vCurrentItem ? fIsImageFile(vCurrentItem.attachment, vCurrentItem.uploadedFile) : false
  );
  let vIsPdf = $derived(
    vCurrentItem ? fIsPdfFile(vCurrentItem.attachment, vCurrentItem.uploadedFile) : false
  );
  let vIsText = $derived(
    vCurrentItem ? fIsTextFile(vCurrentItem.attachment, vCurrentItem.uploadedFile) : false
  );

  let vDisplayPreview = $derived(
    vCurrentItem?.uploadedFile?.preview ||
      (vIsImage && vCurrentItem?.attachment && 'base64Url' in vCurrentItem.attachment
        ? vCurrentItem.attachment.base64Url
        : vCurrentItem?.preview)
  );

  let vDisplayTextContent = $derived(
    vCurrentItem?.uploadedFile?.textContent ||
      (vCurrentItem?.attachment && 'content' in vCurrentItem.attachment
        ? vCurrentItem.attachment.content
        : vCurrentItem?.textContent)
  );

  let vLanguage = $derived(fGetLanguageFromFilename(vDisplayName));

  let vFileSize = $derived(vCurrentItem?.size ? fFormatFileSize(vCurrentItem.size) : '');

  let vHasVisionModality = $derived(
    vCurrentItem && vActiveModelId ? cModelsStore.mModelSupportsVision(vActiveModelId) : false
  );

  let vAudioSrc = $derived(
    vIsAudio && vCurrentItem
      ? (vCurrentItem.uploadedFile?.preview ??
          (vCurrentItem.attachment &&
          'mimeType' in vCurrentItem.attachment &&
          'base64Data' in vCurrentItem.attachment
            ? fCreateBase64DataUrl(
                vCurrentItem.attachment.mimeType,
                vCurrentItem.attachment.base64Data
              )
            : null))
      : null
  );

  let vVideoSrc = $derived(
    vIsVideo && vCurrentItem
      ? (vCurrentItem.uploadedFile?.preview ??
          (vCurrentItem.attachment &&
          'mimeType' in vCurrentItem.attachment &&
          'base64Data' in vCurrentItem.attachment
            ? fCreateBase64DataUrl(
                vCurrentItem.attachment.mimeType,
                vCurrentItem.attachment.base64Data
              )
            : null))
      : null
  );

  export function fPrev() {
    vCurrentIndex = vCurrentIndex > 0 ? vCurrentIndex - 1 : vAllItems.length - 1;
  }

  export function fNext() {
    vCurrentIndex = vCurrentIndex < vAllItems.length - 1 ? vCurrentIndex + 1 : 0;
  }

  function fOnNavigate(pIndex: number) {
    vCurrentIndex = pIndex;
  }
</script>

<div class="{vClassName} flex flex-col text-white">
  <div class="relative flex min-h-0 flex-1 items-center justify-center overflow-hidden">
    <ChatAttachmentsPreviewNavButtons onPrev={fPrev} onNext={fNext} show={vAllItems.length > 1} />

    <div class="flex h-full w-full flex-col items-center justify-start overflow-auto py-4">
      {#if vCurrentItem}
        <ChatAttachmentsPreviewFileInfo displayName={vDisplayName} fileSize={vFileSize} />

        <ChatAttachmentsPreviewCurrentItem
          currentItem={vCurrentItem}
          isImage={vIsImage}
          isAudio={vIsAudio}
          isVideo={vIsVideo}
          isPdf={vIsPdf}
          isText={vIsText}
          displayPreview={vDisplayPreview}
          displayTextContent={vDisplayTextContent}
          audioSrc={vAudioSrc}
          videoSrc={vVideoSrc}
          language={vLanguage}
          hasVisionModality={vHasVisionModality}
          activeModelId={vActiveModelId}
        />
      {/if}

      <ChatAttachmentsPreviewThumbnailStrip
        items={vAllItems}
        currentIndex={vCurrentIndex}
        onNavigate={fOnNavigate}
      />
    </div>
  </div>
</div>
