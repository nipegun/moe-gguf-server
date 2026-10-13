<script lang="ts">
  import { fTranslate } from '../../../../../../i18n';

  import { cIconClassDefault } from '$lib/constants/css-classes';
  import { X, Music, Video } from '@lucide/svelte';
  import {
    formatFileSize,
    getFileTypeLabel,
    getPreviewText,
    isPdfFile,
    isAudioFile,
    isVideoFile,
    isTextFile
  } from '$lib/utils';
  import { ActionIcon } from '$lib/components/app';
  import { AttachmentType } from '$lib/enums';

  interface Props {
    attachment?: DatabaseMessageExtra;
    class?: string;
    id: string;
    onclick?: (pEvent: MouseEvent) => void;
    onRemove?: (pId: string) => void;
    name: string;
    readonly?: boolean;
    size?: number;
    textContent?: string;
    // Either uploaded file or stored attachment
    uploadedFile?: ChatUploadedFile;
  }

  let {
    attachment: vAttachment,
    class: vClassName = '',
    id: vId,
    onclick: vOnclick,
    onRemove: vOnRemove,
    name: vName,
    readonly: vReadonly = false,
    size: vSize,
    textContent: vTextContent,
    uploadedFile: vUploadedFile
  }: Props = $props();

  let vIsPdf = $derived(isPdfFile(vAttachment, vUploadedFile));
  let vIsAudio = $derived(isAudioFile(vAttachment, vUploadedFile));
  let vIsVideo = $derived(isVideoFile(vAttachment, vUploadedFile));
  let vIsPdfWithContent = $derived(vIsPdf && !!vTextContent);

  let vIsText = $derived(isTextFile(vAttachment, vUploadedFile));
  let vIsTextWithContent = $derived(vIsText && !!vTextContent);

  let vFileTypeLabel = $derived.by(() => {
    if (vUploadedFile?.type) {
      return getFileTypeLabel(vUploadedFile.type);
    }

    if (vAttachment) {
      if ('mimeType' in vAttachment && vAttachment.mimeType) {
        return getFileTypeLabel(vAttachment.mimeType);
      }

      if (vAttachment.type) {
        return getFileTypeLabel(vAttachment.type);
      }
    }

    return getFileTypeLabel(vName);
  });

  let vPdfProcessingMode = $derived.by(() => {
    if (vAttachment?.type === AttachmentType.PDF) {
      const cPdfAttachment = vAttachment as DatabaseMessageExtraPdfFile;

      return cPdfAttachment.processedAsImages
        ? fTranslate('text_45ed77775701')
        : fTranslate('text_8c0835a02aa5');
    }

    return null;
  });
</script>

{#snippet textPreview(content: string)}
  <div class="relative">
    <div
      class="font-mono text-xs leading-relaxed wrap-anywhere whitespace-pre-wrap text-muted-foreground {!vReadonly
        ? 'max-h-3rem line-height-1.2'
        : ''}"
    >
      {getPreviewText(content)}
    </div>

    {#if content.length > 150}
      <div
        class="pointer-events-none absolute right-0 bottom-0 left-0 h-4 bg-gradient-to-t from-muted to-transparent {vReadonly
          ? 'h-6'
          : ''}"
      ></div>
    {/if}
  </div>
{/snippet}

{#snippet removeButton()}
  <div
    class="absolute top-2 right-2 opacity-0 transition-opacity group-focus-within:opacity-100 group-hover:opacity-100"
  >
    <ActionIcon
      icon={X}
      tooltip={fTranslate('text_c3812fc4acb8')}
      stopPropagationOnClick
      onclick={() => vOnRemove?.(vId)}
    />
  </div>
{/snippet}

{#snippet fileIcon()}
  <div
    class="flex h-8 w-8 items-center justify-center rounded bg-primary/10 text-xs font-medium text-primary"
  >
    {#if vIsAudio}
      <Music class="{cIconClassDefault} text-white/70" />
    {:else if vIsVideo}
      <Video class="{cIconClassDefault} text-white/70" />
    {:else}
      {vFileTypeLabel}
    {/if}
  </div>
{/snippet}

{#snippet info(text: string | undefined)}
  {#if text}
    <span class="text-xs text-muted-foreground">{text}</span>
  {/if}
{/snippet}

{#if vIsTextWithContent || vIsPdfWithContent}
  <button
    aria-label={vReadonly ? fTranslate('text_e100b47318c5', { p0: vName }) : undefined}
    class="rounded-lg border border-border bg-muted p-3 {vClassName} cursor-pointer {vReadonly
      ? 'w-full max-w-2xl transition-shadow hover:shadow-md'
      : `group relative text-left ${vTextContent ? 'max-h-24 max-w-72' : 'max-w-36'}`} overflow-hidden"
    onclick={vOnclick}
    type="button"
  >
    {#if !vReadonly}
      {@render removeButton()}
    {/if}

    <div class={[!vReadonly && 'pr-8', 'overflow-hidden']}>
      {#if vReadonly}
        <div class="flex items-start gap-3">
          <div class="flex min-w-0 flex-1 flex-col items-start text-left">
            <span class="w-full truncate text-sm font-medium text-foreground">{vName}</span>

            {@render info(vPdfProcessingMode || (vSize ? formatFileSize(vSize) : undefined))}

            {#if vTextContent}
              {@render textPreview(vTextContent)}
            {/if}
          </div>
        </div>
      {:else}
        <span class="mb-3 block truncate text-sm font-medium text-foreground">{vName}</span>

        {#if vTextContent}
          {@render textPreview(vTextContent)}
        {/if}
      {/if}
    </div>
  </button>
{:else}
  <button
    class="group flex items-center gap-3 rounded-lg border border-border bg-muted p-3 {vClassName} relative"
    onclick={vOnclick}
    type="button"
  >
    {@render fileIcon()}

    <div class="flex flex-col items-start gap-0.5">
      <span
        class="max-w-24 truncate text-sm font-medium text-foreground {vReadonly
          ? ''
          : 'group-hover:pr-6'} md:max-w-32"
      >
        {vName}
      </span>

      {@render info(vPdfProcessingMode || (vSize ? formatFileSize(vSize) : undefined))}
    </div>

    {#if !vReadonly}
      {@render removeButton()}
    {/if}
  </button>
{/if}
