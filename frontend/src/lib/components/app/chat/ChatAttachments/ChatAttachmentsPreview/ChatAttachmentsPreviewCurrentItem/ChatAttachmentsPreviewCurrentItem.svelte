<script lang="ts">
  import type { ChatAttachmentDisplayItem } from '$lib/types';
  import { Image, Music, Video, FileText, FileIcon } from '@lucide/svelte';
  import ChatAttachmentsPreviewCurrentItemPdf from './ChatAttachmentsPreviewCurrentItemPdf.svelte';
  import ChatAttachmentsPreviewCurrentItemImage from './ChatAttachmentsPreviewCurrentItemImage.svelte';
  import ChatAttachmentsPreviewCurrentItemAudio from './ChatAttachmentsPreviewCurrentItemAudio.svelte';
  import ChatAttachmentsPreviewCurrentItemVideo from './ChatAttachmentsPreviewCurrentItemVideo.svelte';
  import ChatAttachmentsPreviewCurrentItemText from './ChatAttachmentsPreviewCurrentItemText.svelte';
  import ChatAttachmentsPreviewCurrentItemUnavailable from './ChatAttachmentsPreviewCurrentItemUnavailable.svelte';

  interface Props {
    currentItem: ChatAttachmentDisplayItem | null;
    isImage: boolean;
    isAudio: boolean;
    isVideo: boolean;
    isPdf: boolean;
    isText: boolean;
    displayPreview: string | undefined;
    displayTextContent: string | undefined;
    audioSrc: string | null;
    videoSrc: string | null;
    language: string;
    hasVisionModality: boolean;
    activeModelId?: string;
  }

  let {
    currentItem: vCurrentItem,
    isImage: vIsImage,
    isAudio: vIsAudio,
    isVideo: vIsVideo,
    isPdf: vIsPdf,
    isText: vIsText,
    displayPreview: vDisplayPreview,
    displayTextContent: vDisplayTextContent,
    audioSrc: vAudioSrc,
    videoSrc: vVideoSrc,
    language: vLanguage,
    hasVisionModality: vHasVisionModality,
    activeModelId: vActiveModelId
  }: Props = $props();

  let vIconComponent = $derived(
    vIsImage ? Image : vIsText || vIsPdf ? FileText : vIsAudio ? Music : vIsVideo ? Video : FileIcon
  );

  let vIsUnavailable = $derived(
    !vIsPdf && !vIsImage && !(vIsText && vDisplayTextContent) && !vIsAudio && !vIsVideo
  );
</script>

{#if vCurrentItem}
  {#key vCurrentItem.id}
    {#if vIsPdf}
      <ChatAttachmentsPreviewCurrentItemPdf
        currentItem={vCurrentItem}
        displayName={vCurrentItem.name}
        displayTextContent={vDisplayTextContent}
        hasVisionModality={vHasVisionModality}
        activeModelId={vActiveModelId}
      />
    {:else if vIsImage}
      <ChatAttachmentsPreviewCurrentItemImage
        currentItem={vCurrentItem}
        displayPreview={vDisplayPreview}
      />
    {:else if vIsText && vDisplayTextContent}
      <ChatAttachmentsPreviewCurrentItemText
        displayTextContent={vDisplayTextContent}
        language={vLanguage}
      />
    {:else if vIsAudio}
      <ChatAttachmentsPreviewCurrentItemAudio currentItem={vCurrentItem} audioSrc={vAudioSrc} />
    {:else if vIsVideo}
      <ChatAttachmentsPreviewCurrentItemVideo currentItem={vCurrentItem} videoSrc={vVideoSrc} />
    {:else if vIsUnavailable}
      <ChatAttachmentsPreviewCurrentItemUnavailable IconComponent={vIconComponent} />
    {/if}
  {/key}
{/if}
