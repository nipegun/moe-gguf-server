<script lang="ts">
  import { cMcpStore } from '$lib/stores/mcp.svelte';
  import {
    fMcpResourceAttachments,
    fMcpHasResourceAttachments
  } from '$lib/stores/mcp-resources.svelte';
  import {
    ChatAttachmentsListItemMcpResource,
    HorizontalScrollCarousel
  } from '$lib/components/app';
  import type { MCPResourceInfo } from '$lib/types';

  interface Props {
    class?: string;
    onResourceClick?: (pResource: MCPResourceInfo) => void;
  }

  let { class: vClassName, onResourceClick: vOnResourceClick }: Props = $props();

  const cAttachments = $derived(fMcpResourceAttachments());
  const cHasAttachments = $derived(fMcpHasResourceAttachments());

  function fHandleRemove(pAttachmentId: string) {
    cMcpStore.mRemoveResourceAttachment(pAttachmentId);
  }

  function fHandleResourceClick(pResource: MCPResourceInfo) {
    vOnResourceClick?.(pResource);
  }
</script>

{#if cHasAttachments}
  <div class={vClassName}>
    <HorizontalScrollCarousel gapSize="2">
      {#each cAttachments as vAttachment, vI (vAttachment.id)}
        <ChatAttachmentsListItemMcpResource
          class={vI === 0 ? 'ml-3' : ''}
          attachment={vAttachment}
          onRemove={fHandleRemove}
          onclick={() => fHandleResourceClick(vAttachment.resource)}
        />
      {/each}
    </HorizontalScrollCarousel>
  </div>
{/if}
