<script lang="ts">
  import { Lightbulb } from '@lucide/svelte';
  import { CollapsibleContentBlock, MarkdownContent } from '$lib/components/app';
  import { AgenticSectionType } from '$lib/enums';
  import { cReasoningScrollAtBottomThresholdPx } from '$lib/constants/auto-scroll';
  import type { DatabaseMessageExtra } from '$lib/types';
  import type { AgenticSection } from '$lib/utils';

  interface Props {
    section: AgenticSection;
    open: boolean;
    isStreaming: boolean;
    renderThinkingAsMarkdown: boolean;
    hasReasoningError?: boolean;
    attachments?: DatabaseMessageExtra[];
    onToggle?: () => void;
  }

  let {
    section: vSection,
    open: vOpen,
    isStreaming: vIsStreaming,
    renderThinkingAsMarkdown: vRenderThinkingAsMarkdown,
    hasReasoningError: vHasReasoningError = false,
    attachments: vAttachments,
    onToggle: vOnToggle
  }: Props = $props();

  const cReasoningHeader = 'Reasoning';
  const cReasoningHeaderPending = 'Reasoning...';
  const cReasoningSubtitleError = 'Error';
  const cReasoningSubtitleCancelled = 'Cancelled';

  const cIsPending = $derived(vSection.type === AgenticSectionType.REASONING_PENDING);
  const cTitle = $derived(cIsPending && vIsStreaming ? cReasoningHeaderPending : cReasoningHeader);
  const cSubtitle = $derived.by(() => {
    if (cIsPending && !vIsStreaming) {
      return vHasReasoningError ? cReasoningSubtitleError : cReasoningSubtitleCancelled;
    }
    if (vSection.wasInterrupted) {
      return vHasReasoningError ? cReasoningSubtitleError : cReasoningSubtitleCancelled;
    }
    return vIsStreaming ? '' : undefined;
  });
  const cShimmerTitle = $derived(cIsPending && vIsStreaming);

  let vScrollEl: HTMLDivElement | undefined = $state();

  const cScrollBottomThresholdPx = cReasoningScrollAtBottomThresholdPx;

  let vUserScrolledUp = $state(false);
  let vLastScrollTop = 0;
  let vPendingFrame: number | null = null;

  function fIsAtBottom(): boolean {
    if (!vScrollEl) return false;
    return (
      vScrollEl.scrollHeight - vScrollEl.clientHeight - vScrollEl.scrollTop <=
      cScrollBottomThresholdPx
    );
  }

  function fScrollToBottomOnFrame() {
    if (vPendingFrame !== null || !vScrollEl || vUserScrolledUp) return;
    vPendingFrame = requestAnimationFrame(() => {
      vPendingFrame = null;
      // User may scroll between scheduling and paint.
      if (vScrollEl && !vUserScrolledUp) {
        vScrollEl.scrollTop = vScrollEl.scrollHeight;
      }
    });
  }

  function fHandleScrollEvent() {
    if (!vScrollEl) return;
    const cIsScrollingUp = vScrollEl.scrollTop < vLastScrollTop;
    if (cIsScrollingUp && !fIsAtBottom()) {
      vUserScrolledUp = true;
    } else if (fIsAtBottom()) {
      vUserScrolledUp = false;
    }
    vLastScrollTop = vScrollEl.scrollTop;
  }

  $effect(() => {
    void vSection.content;
    if (!vScrollEl || !cIsPending || !vIsStreaming) return;
    fScrollToBottomOnFrame();
  });

  $effect(() => {
    // Layout shifts that don't change section.content (markdown re-parse,
    // syntax-highlight settle, image loads).
    if (!vScrollEl || !cIsPending || !vIsStreaming) return;

    const cObserver = new MutationObserver(() => fScrollToBottomOnFrame());
    cObserver.observe(vScrollEl, {
      childList: true,
      subtree: true,
      characterData: true
    });

    return () => cObserver.disconnect();
  });

  $effect(() => {
    // Pin to bottom at the start of each round.
    if (!cIsPending) {
      vUserScrolledUp = false;
      vLastScrollTop = 0;
    }
  });
</script>

<CollapsibleContentBlock
  open={vOpen}
  class="my-2"
  icon={Lightbulb}
  iconClass="h-3.5 w-3.5"
  title={cTitle}
  subtitle={cSubtitle}
  shimmerTitle={cShimmerTitle}
  onToggle={vOnToggle}
>
  <div
    bind:this={vScrollEl}
    class="reasoning-content"
    class:is-streaming={cIsPending}
    onscroll={fHandleScrollEvent}
  >
    {#if vRenderThinkingAsMarkdown}
      <MarkdownContent
        content={vSection.content}
        class="text-muted-foreground"
        attachments={vAttachments}
      />
    {:else}
      <div
        class="text-[13px] leading-relaxed wrap-anywhere whitespace-pre-wrap text-muted-foreground"
      >
        {vSection.content}
      </div>
    {/if}
  </div>
</CollapsibleContentBlock>

<style>
  .reasoning-content.is-streaming {
    max-height: 28rem;
    overflow-y: auto;
    overscroll-behavior: contain;
    scrollbar-gutter: stable;
    padding-right: 0.25rem;
  }
</style>
