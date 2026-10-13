<script lang="ts">
  import { Card } from '$lib/components/ui/card';
  import { ChatAttachmentsList, MarkdownContent } from '$lib/components/app';
  import { fConfig } from '$lib/stores/settings.svelte';
  import type { DatabaseMessageExtra } from '$lib/types/database';

  interface Props {
    content: string;
    attachments?: DatabaseMessageExtra[];
    renderMarkdown?: boolean;
    textColorClass?: string;
    cardBgClass?: string;
    maxHeightStyle?: string;
  }

  let {
    content: vContent,
    attachments: vAttachments = [],
    renderMarkdown: vRenderMarkdown = false,
    textColorClass: vTextColorClass = 'text-foreground',
    cardBgClass: vCardBgClass = 'dark:bg-primary/15',
    maxHeightStyle: vMaxHeightStyle = ''
  }: Props = $props();

  let vIsMultiline = $state(false);
  let vMessageElement: HTMLElement | undefined = $state();
  const cCurrentConfig = fConfig();

  $effect(() => {
    if (!vMessageElement || !vContent.trim()) return;

    if (vContent.includes('\n')) {
      vIsMultiline = true;
      return;
    }

    const cResizeObserver = new ResizeObserver((pEntries) => {
      for (const cEntry of pEntries) {
        const cElement = cEntry.target as HTMLElement;
        const cEstimatedSingleLineHeight = 24; // Typical line height for text-md

        vIsMultiline = cElement.offsetHeight > cEstimatedSingleLineHeight * 1.5;
      }
    });

    cResizeObserver.observe(vMessageElement);

    return () => {
      cResizeObserver.disconnect();
    };
  });
</script>

{#if vAttachments && vAttachments.length > 0}
  <div class="mb-2 max-w-[80%]">
    <ChatAttachmentsList attachments={vAttachments} readonly imageHeight="h-40" />
  </div>
{/if}

{#if vContent.trim()}
  <Card
    class="chat-message-user-bubble max-w-[80%] overflow-y-auto rounded-[1.125rem] border-none bg-primary/5 px-3.75 py-1.5 {vTextColorClass} backdrop-blur-md data-multiline:py-2.5 {vCardBgClass}"
    data-multiline={vIsMultiline ? '' : undefined}
    style="{vMaxHeightStyle} overflow-wrap: anywhere; word-break: break-word;"
  >
    {#if vRenderMarkdown && cCurrentConfig.renderUserContentAsMarkdown}
      <div bind:this={vMessageElement}>
        <MarkdownContent class="markdown-user-content" content={vContent} />
      </div>
    {:else}
      <span dir="auto" bind:this={vMessageElement} class="text-md whitespace-pre-wrap">
        {vContent}
      </span>
    {/if}
  </Card>
{/if}
