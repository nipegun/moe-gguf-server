<script lang="ts">
  import {
    ChatAttachmentsListItem,
    DialogChatAttachmentsPreview,
    DialogMcpResourcePreview,
    HorizontalScrollCarousel
  } from '$lib/components/app';
  import type { DatabaseMessageExtraMcpResource } from '$lib/types';
  import { getAttachmentDisplayItems, isMcpPrompt, isMcpResource } from '$lib/utils';

  interface Props {
    class?: string;
    style?: string;
    // For ChatMessage - stored attachments
    attachments?: DatabaseMessageExtra[];
    readonly?: boolean;
    // For ChatForm - pending uploads
    onFileRemove?: (pFileId: string) => void;
    uploadedFiles?: ChatUploadedFile[];
    // Image size customization
    imageClass?: string;
    imageHeight?: string;
    imageWidth?: string;
    // Limit display to single row with "+ X more" button
    limitToSingleRow?: boolean;
    // For vision modality check
    activeModelId?: string;
  }

  let {
    class: vClassName = '',
    style: vStyle = '',
    attachments: vAttachments = [],
    readonly: vReadonly = false,
    onFileRemove: vOnFileRemove,
    uploadedFiles: vUploadedFiles = $bindable([]),
    // Default to small size for form previews
    imageClass: vImageClass = '',
    imageHeight: vImageHeight = 'h-24',
    imageWidth: vImageWidth = 'w-auto',
    limitToSingleRow: vLimitToSingleRow = false,
    activeModelId: vActiveModelId
  }: Props = $props();

  let vCarouselRef: HorizontalScrollCarousel | undefined = $state();
  let vMcpResourcePreviewOpen = $state(false);
  let vMcpResourcePreviewExtra = $state<DatabaseMessageExtraMcpResource | null>(null);
  let vPreviewFocusIndex = $state(0);
  let vViewAllDialogOpen = $state(false);

  let vDisplayItems = $derived(
    getAttachmentDisplayItems({ uploadedFiles: vUploadedFiles, attachments: vAttachments })
  );

  function fOpenPreview(pItem: ChatAttachmentDisplayItem, pEvent?: MouseEvent) {
    pEvent?.stopPropagation();
    pEvent?.preventDefault();

    // Find the index of the clicked item among non-MCP attachments
    const cNonMcpItems = vDisplayItems.filter((pI) => !isMcpPrompt(pI) && !isMcpResource(pI));
    const cIndex = cNonMcpItems.findIndex((pI) => pI.id === pItem.id);

    vPreviewFocusIndex = cIndex >= 0 ? cIndex : 0;
    vViewAllDialogOpen = true;
  }

  function fOpenMcpResourcePreview(pExtra: DatabaseMessageExtraMcpResource) {
    vMcpResourcePreviewExtra = pExtra;
    vMcpResourcePreviewOpen = true;
  }

  $effect(() => {
    if (vCarouselRef && vDisplayItems.length) {
      vCarouselRef.fResetScroll();
    }
  });
</script>

{#snippet attachmentitem(item: ChatAttachmentDisplayItem)}
  <ChatAttachmentsListItem
    imageClass={vImageClass}
    imageHeight={vImageHeight}
    imageWidth={vImageWidth}
    {item}
    limitToSingleRow={vLimitToSingleRow}
    onFileRemove={vOnFileRemove}
    onMcpResourcePreview={fOpenMcpResourcePreview}
    onPreview={(i: ChatAttachmentDisplayItem, event?: MouseEvent) => fOpenPreview(i, event)}
    readonly={vReadonly}
  />
{/snippet}

{#if vDisplayItems.length > 0}
  <div class={vClassName} style={vStyle}>
    {#if vLimitToSingleRow}
      <HorizontalScrollCarousel bind:this={vCarouselRef}>
        {#each vDisplayItems as item (item.id)}
          {@render attachmentitem(item)}
        {/each}
      </HorizontalScrollCarousel>
    {:else}
      <div class="flex flex-wrap items-start justify-end gap-3">
        {#each vDisplayItems as item (item.id)}
          {@render attachmentitem(item)}
        {/each}
      </div>
    {/if}
  </div>
{/if}

<DialogChatAttachmentsPreview
  activeModelId={vActiveModelId}
  attachments={vAttachments}
  bind:open={vViewAllDialogOpen}
  previewFocusIndex={vPreviewFocusIndex}
  uploadedFiles={vUploadedFiles}
/>

{#if vMcpResourcePreviewExtra}
  <DialogMcpResourcePreview extra={vMcpResourcePreviewExtra} bind:open={vMcpResourcePreviewOpen} />
{/if}
