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

  interface Props {
    class?: string;
    onResourceClick?: (pUri: string) => void;
  }

  let { class: vClassName, onResourceClick: vOnResourceClick }: Props = $props();

  const cAttachments = $derived(fMcpResourceAttachments());
  const cHasAttachments = $derived(fMcpHasResourceAttachments());

  function fHandleRemove(pAttachmentId: string) {
    cMcpStore.mRemoveResourceAttachment(pAttachmentId);
  }

  function fHandleResourceClick(pUri: string) {
    vOnResourceClick?.(pUri);
  }
</script>

{#if cHasAttachments}
  <div class={vClassName}>
    <HorizontalScrollCarousel gapSize="2">
      {#each cAttachments as attachment, i (attachment.id)}
        <ChatAttachmentsListItemMcpResource
          class={i === 0 ? 'ml-3' : ''}
          {attachment}
          onRemove={fHandleRemove}
          onclick={() => fHandleResourceClick(attachment.resource.uri)}
        />
      {/each}
    </HorizontalScrollCarousel>
  </div>
{/if}
