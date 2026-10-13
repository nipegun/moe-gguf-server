<script lang="ts">
  import {
    ChatAttachmentsListItemMcpPrompt,
    ChatAttachmentsListItemMcpResource,
    ChatAttachmentsListItemThumbnailImage,
    ChatAttachmentsListItemThumbnailFile
  } from '$lib/components/app';
  import { AttachmentType } from '$lib/enums';
  import type {
    ChatAttachmentDisplayItem,
    DatabaseMessageExtraMcpPrompt,
    DatabaseMessageExtraMcpResource,
    MCPResourceAttachment
  } from '$lib/types';
  import { isMcpPrompt, isMcpResource, isPdfFile } from '$lib/utils';

  interface Props {
    class?: string;
    imageClass?: string;
    imageHeight?: string;
    imageWidth?: string;
    item: ChatAttachmentDisplayItem;
    limitToSingleRow?: boolean;
    onFileRemove?: (pFileId: string) => void;
    onMcpResourcePreview?: (pExtra: DatabaseMessageExtraMcpResource) => void;
    onPreview?: (pItem: ChatAttachmentDisplayItem) => void;
    readonly?: boolean;
  }

  let {
    class: vClassName = '',
    imageClass: vImageClass = '',
    imageHeight: vImageHeight = 'h-24',
    imageWidth: vImageWidth = 'w-auto',
    item: vItem,
    limitToSingleRow: vLimitToSingleRow = false,
    onFileRemove: vOnFileRemove,
    onMcpResourcePreview: vOnMcpResourcePreview,
    onPreview: vOnPreview,
    readonly: vReadonly = false
  }: Props = $props();

  const cScrollClasses = $derived(vLimitToSingleRow ? 'first:ml-4 last:mr-4' : '');

  function fToMcpResourceAttachment(
    pExtra: DatabaseMessageExtraMcpResource,
    pId: string
  ): MCPResourceAttachment {
    return {
      id: pId,
      resource: {
        uri: pExtra.uri,
        name: pExtra.name,
        title: pExtra.name,
        serverName: pExtra.serverName
      }
    };
  }
</script>

{#if isMcpPrompt(vItem)}
  {@const mcpPrompt =
    vItem.attachment?.type === AttachmentType.MCP_PROMPT
      ? (vItem.attachment as DatabaseMessageExtraMcpPrompt)
      : vItem.uploadedFile?.mcpPrompt
        ? {
            type: AttachmentType.MCP_PROMPT as const,
            name: vItem.name,
            serverName: vItem.uploadedFile.mcpPrompt.serverName,
            promptName: vItem.uploadedFile.mcpPrompt.promptName,
            content: vItem.textContent ?? '',
            arguments: vItem.uploadedFile.mcpPrompt.arguments
          }
        : null}
  {#if mcpPrompt}
    <ChatAttachmentsListItemMcpPrompt
      class="max-w-[300px] min-w-[200px] flex-shrink-0 {vClassName} {cScrollClasses}"
      prompt={mcpPrompt}
      readonly={vReadonly}
      isLoading={vItem.isLoading}
      loadError={vItem.loadError}
      onRemove={vOnFileRemove ? () => vOnFileRemove(vItem.id) : undefined}
    />
  {/if}
{:else if isMcpResource(vItem)}
  {@const mcpResource = vItem.attachment as DatabaseMessageExtraMcpResource}

  <ChatAttachmentsListItemMcpResource
    class="flex-shrink-0 {vClassName} {cScrollClasses}"
    attachment={fToMcpResourceAttachment(mcpResource, vItem.id)}
    onclick={() => vOnMcpResourcePreview?.(mcpResource)}
  />
{:else if vItem.isImage && vItem.preview}
  <ChatAttachmentsListItemThumbnailImage
    class="flex-shrink-0 cursor-pointer {vClassName} {cScrollClasses}"
    id={vItem.id}
    name={vItem.name}
    preview={vItem.preview}
    readonly={vReadonly}
    onRemove={vOnFileRemove}
    height={vImageHeight}
    width={vImageWidth}
    imageClass={vImageClass}
    onclick={() => vOnPreview?.(vItem)}
  />
{:else if isPdfFile(vItem.attachment, vItem.uploadedFile)}
  <ChatAttachmentsListItemThumbnailFile
    class="flex-shrink-0 cursor-pointer {vClassName} {cScrollClasses}"
    id={vItem.id}
    name={vItem.name}
    size={vItem.size}
    readonly={vReadonly}
    onRemove={vOnFileRemove}
    textContent={vItem.textContent}
    attachment={vItem.attachment}
    uploadedFile={vItem.uploadedFile}
    onclick={() => vOnPreview?.(vItem)}
  />
{:else}
  <ChatAttachmentsListItemThumbnailFile
    class="flex-shrink-0 cursor-pointer {vClassName} {cScrollClasses}"
    id={vItem.id}
    name={vItem.name}
    size={vItem.size}
    readonly={vReadonly}
    onRemove={vOnFileRemove}
    textContent={vItem.textContent}
    attachment={vItem.attachment}
    uploadedFile={vItem.uploadedFile}
    onclick={() => vOnPreview?.(vItem)}
  />
{/if}
