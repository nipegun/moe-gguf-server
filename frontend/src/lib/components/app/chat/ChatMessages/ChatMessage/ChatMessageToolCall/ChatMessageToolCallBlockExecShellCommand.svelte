<script lang="ts">
  import { fTranslate } from '../../../../../../i18n';

  // Block for `exec_shell_command`. Unlike the other tools, this
  // renderer uses CollapsibleTerminalBlock (terminal-style frame)
  // and treats "live" output chunks as active even after the call
  // resolved, so the spinner stays on while stdout is still flowing.
  // The scroll-to-bottom auto-scroll logic mirrors what was here
  // before extraction.

  import { Check, Loader2, XCircle, AlertTriangle } from '@lucide/svelte';
  import { CollapsibleTerminalBlock } from '$lib/components/app';
  import { cSettingsKeys } from '$lib/constants';
  import { fConfig } from '$lib/stores/settings.svelte';
  import { cToolRuntimeScrollAtBottomThresholdPx } from '$lib/constants/auto-scroll';
  import {
    highlightCode,
    isExitCodeSummaryLine,
    parseExecShellCommandError,
    parseExecShellCommandExitStatus,
    parseToolResultWithImages,
    type AgenticSection,
    type ExecShellExitStatus,
    type ToolResultLine
  } from '$lib/utils';
  import { fParseExecShellCommandMeta } from './parsers/exec-shell-command';
  import type { DatabaseMessageExtra } from '$lib/types';
  import ToolCallBlock from './ToolCallBlock.svelte';

  interface Props {
    section: AgenticSection;
    open: boolean;
    isStreaming: boolean;
    /** True while the agentic loop is streaming output chunks for THIS
     *  tool call. Drives max-height + auto-scroll while true; releases
     *  them when the loop reports this call as done. */
    isExecuting?: boolean;
    attachments?: DatabaseMessageExtra[];
    onToggle?: () => void;
  }

  let {
    section: vSection,
    open: vOpen,
    isStreaming: vIsStreaming,
    isExecuting: vIsExecuting = false,
    attachments: vAttachments,
    onToggle: vOnToggle
  }: Props = $props();

  // `isLive` covers all in-flight phases: pre-chunk spinner and
  // streaming itself. Frozen output (tool done while agent continues)
  // is not live.
  const cIsLive = $derived(vIsExecuting);

  const cExecShellMeta = $derived(fParseExecShellCommandMeta(vSection));
  const cExecShellError = $derived(parseExecShellCommandError(vSection.toolResult));
  const cExecShellExitStatus: ExecShellExitStatus | undefined = $derived(
    parseExecShellCommandExitStatus(vSection.toolResult)
  );

  const aParsedLines: ToolResultLine[] = $derived(
    vSection.toolResult ? parseToolResultWithImages(vSection.toolResult, vAttachments) : []
  );

  // Drop the trailing "[exit code: N]" line - rendered as a colored
  // badge below. During streaming we keep it so a partial stream still
  // shows the status once the final chunk lands.
  const aOutputLines: ToolResultLine[] = $derived(
    cExecShellExitStatus && aParsedLines.length > 0
      ? aParsedLines.slice(0, aParsedLines.length - 1)
      : aParsedLines
  );

  const cIsExitCodeFinalLine = $derived(
    cExecShellExitStatus !== undefined &&
      aParsedLines.length > 0 &&
      isExitCodeSummaryLine(aParsedLines[aParsedLines.length - 1].text, cExecShellExitStatus)
  );

  // Highlight just the command for the title; the (typically large)
  // output blob uses bare monospace to skip hljs per-line highlighting.
  const cHighlightedCommandHtml = $derived(
    cExecShellMeta ? highlightCode(cExecShellMeta.command, 'bash') : ''
  );

  const cExitBadgeClass = $derived(
    cExecShellExitStatus?.timedOut
      ? 'exit-badge warning'
      : cExecShellExitStatus?.code === 0
        ? 'exit-badge success'
        : 'exit-badge failure'
  );

  const cUseFullHeightCodeBlocks = $derived(
    Boolean(fConfig()[cSettingsKeys.FULL_HEIGHT_CODE_BLOCKS])
  );

  const cAutoScroll = $derived(cIsLive && !cUseFullHeightCodeBlocks);

  const cScrollBottomThresholdPx = cToolRuntimeScrollAtBottomThresholdPx;

  let vScrollEl: HTMLDivElement | undefined = $state();
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

      // Re-check on rAF - user may scroll between scheduling and paint.
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
    void vSection.toolResult;
    if (!vScrollEl || !cAutoScroll) return;
    fScrollToBottomOnFrame();
  });

  $effect(() => {
    // Catch layout changes that don't touch toolResult (line-wrap
    // reflow, image attaches, hljs settle).
    if (!vScrollEl || !cAutoScroll) return;

    const cObserver = new MutationObserver(() => fScrollToBottomOnFrame());
    cObserver.observe(vScrollEl, {
      childList: true,
      subtree: true,
      characterData: true
    });

    return () => cObserver.disconnect();
  });

  $effect(() => {
    // Reset on stream end so the next render (full-height) starts
    // pinned.
    if (!cIsLive) {
      vUserScrolledUp = false;
      vLastScrollTop = 0;
    }
  });
</script>

{#snippet execShellTitle()}
  {#if cHighlightedCommandHtml}
    <span class="font-mono">{@html cHighlightedCommandHtml}</span>
  {:else}
    <span class="font-mono">{cExecShellMeta?.command}</span>
  {/if}
{/snippet}

<ToolCallBlock
  section={vSection}
  open={vOpen}
  isStreaming={vIsStreaming}
  meta={cExecShellMeta ? { errorMessage: cExecShellError } : null}
  wrapper={CollapsibleTerminalBlock}
  extraLiveStreaming={cIsLive}
  spinIconWhenActive={true}
  onToggle={vOnToggle}
>
  {#snippet titleSnippet()}
    {@render execShellTitle()}
  {/snippet}

  {#snippet children(_meta, ctx)}
    {#if ctx.isPending}
      <div class="flex items-start gap-2 text-xs text-muted-foreground/70">
        <Loader2 class="h-3 w-3 animate-spin" />
        {fTranslate('text_4977a7e5bb36')}
      </div>
    {:else if cExecShellError}
      <div class="flex items-start gap-2 text-xs text-red-600 italic dark:text-red-400">
        <XCircle class="mt-0.5 h-3 w-3 shrink-0" />
        <span>{cExecShellError}</span>
      </div>
    {:else if vSection.toolResult}
      <div
        bind:this={vScrollEl}
        class="terminal-output"
        class:is-clamped={!cUseFullHeightCodeBlocks}
        onscroll={fHandleScrollEvent}
      >
        {#each aOutputLines as line, i (i)}
          <div class="font-mono text-[11px] leading-relaxed whitespace-pre-wrap">{line.text}</div>
          {#if line.image}
            <img
              src={line.image.base64Url}
              alt={line.image.name}
              class="mt-2 mb-2 h-auto max-w-full rounded-lg"
              loading="lazy"
            />
          {/if}
        {/each}

        {#if cIsExitCodeFinalLine && cExecShellExitStatus}
          <div class={cExitBadgeClass}>
            {#if cExecShellExitStatus.timedOut}
              <AlertTriangle class="h-3 w-3" />
              <span>{fTranslate('text_3dcd80f1b15f')}</span>
              <span class="exit-sep">&middot;</span>
              <span>{fTranslate('text_e596899f114b')} {cExecShellExitStatus.code}</span>
            {:else if cExecShellExitStatus.code === 0}
              <Check class="h-3 w-3" />
              <span>{fTranslate('text_c22995adc297')}</span>
            {:else}
              <XCircle class="h-3 w-3" />
              <span>{fTranslate('text_e596899f114b')} {cExecShellExitStatus.code}</span>
            {/if}
          </div>
        {/if}
      </div>
    {/if}
  {/snippet}
</ToolCallBlock>

<style>
  .terminal-output {
    overscroll-behavior: contain;
  }

  .terminal-output.is-clamped {
    max-height: 28rem;
    overflow-y: auto;
    scrollbar-gutter: stable;
    padding-right: 0.25rem;
  }

  .exit-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    margin-top: 0.5rem;
    padding: 0.2rem 0.55rem;
    border-radius: 0.375rem;
    font-family: var(--font-mono);
    font-size: 11px;
    font-weight: 500;
    letter-spacing: 0.01em;
    line-height: 1;
  }

  .exit-badge.success {
    background: color-mix(in oklch, var(--color-green-500, #22c55e) 14%, transparent);
    color: var(--color-green-700, #15803d);
  }

  :global(.dark) .exit-badge.success {
    background: color-mix(in oklch, var(--color-green-400, #4ade80) 18%, transparent);
    color: var(--color-green-300, #86efac);
  }

  .exit-badge.failure {
    background: color-mix(in oklch, var(--color-red-500, #ef4444) 14%, transparent);
    color: var(--color-red-700, #b91c1c);
  }

  :global(.dark) .exit-badge.failure {
    background: color-mix(in oklch, var(--color-red-400, #f87171) 18%, transparent);
    color: var(--color-red-300, #fca5a5);
  }

  .exit-badge.warning {
    background: color-mix(in oklch, var(--color-amber-500, #f59e0b) 14%, transparent);
    color: var(--color-amber-700, #b45309);
  }

  :global(.dark) .exit-badge.warning {
    background: color-mix(in oklch, var(--color-amber-400, #fbbf24) 18%, transparent);
    color: var(--color-amber-300, #fcd34d);
  }

  .exit-sep {
    opacity: 0.45;
  }
</style>
