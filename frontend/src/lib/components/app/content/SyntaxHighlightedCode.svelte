<script lang="ts">
  import { browser } from '$app/environment';
  import { mode } from 'mode-watcher';

  import githubDarkCss from 'highlight.js/styles/github-dark.css?inline';
  import githubLightCss from 'highlight.js/styles/github.css?inline';
  import { ColorMode } from '$lib/enums';
  import { cSyntaxCodeScrollAtBottomThresholdPx } from '$lib/constants/auto-scroll';
  import { highlightCode } from '$lib/utils';

  interface Props {
    code: string;
    language?: string;
    class?: string;
    maxHeight?: string;
    maxWidth?: string;
    /** Border colour (any CSS colour, such as a custom property); a faint border by default. */
    borderColor?: string;
    /** Auto-scrolls to the bottom of new chunks; pauses on user scroll-up
     *  until the user returns to the bottom. */
    streaming?: boolean;
  }

  let {
    code: vCode,
    language: vLanguage = 'text',
    class: vClassName = '',
    maxHeight: vMaxHeight = '60vh',
    maxWidth: vMaxWidth = '',
    borderColor: vBorderColor = '',
    streaming: vStreaming = false
  }: Props = $props();

  const cHighlightedHtml = $derived(highlightCode(vCode, vLanguage));

  let vScrollEl = $state<HTMLDivElement>();
  let vUserScrolledUp = $state(false);
  let vLastScrollTop = 0;
  const cScrollBottomThresholdPx = cSyntaxCodeScrollAtBottomThresholdPx;
  let vPendingFrame: number | null = null;

  function fLoadHighlightTheme(pIsDark: boolean) {
    if (!browser) return;

    const cExistingThemes = document.querySelectorAll('style[data-highlight-theme-preview]');
    cExistingThemes.forEach((pStyle) => pStyle.remove());

    const cStyle = document.createElement('style');
    cStyle.setAttribute('data-highlight-theme-preview', 'true');
    cStyle.textContent = pIsDark ? githubDarkCss : githubLightCss;

    document.head.appendChild(cStyle);
  }

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
    const cCurrentMode = mode.current;
    const cIsDark = cCurrentMode === ColorMode.DARK;

    fLoadHighlightTheme(cIsDark);
  });

  // Pin to bottom at the start of each streaming episode.
  $effect(() => {
    if (vStreaming) {
      vUserScrolledUp = false;
      vLastScrollTop = 0;
    }
  });

  $effect(() => {
    void vCode;
    if (!vStreaming || vUserScrolledUp) return;
    fScrollToBottomOnFrame();
  });

  // Layout shifts that don't change `code` (highlight.js re-tokenize, line-wrap reflow).
  $effect(() => {
    if (!vStreaming || !vScrollEl) return;

    const cObserver = new MutationObserver(() => fScrollToBottomOnFrame());
    cObserver.observe(vScrollEl, {
      childList: true,
      subtree: true,
      characterData: true
    });

    return () => cObserver.disconnect();
  });
</script>

<div
  bind:this={vScrollEl}
  onscroll={fHandleScrollEvent}
  class="code-preview-wrapper min-w-0 max-w-full overflow-auto rounded-xl border shadow-[0_1px_2px_0_rgb(0_0_0_/_0.05)] {vClassName}"
  style="border-color: {vBorderColor ||
    'color-mix(in oklch, var(--border) 30%, transparent)'}; background: var(--code-background); max-height: {vMaxHeight}; {vMaxWidth
    ? `max-width: ${vMaxWidth};`
    : ''}"
>
  <!-- Single line: hljs injection depends on a contiguous source string. -->
  <pre class="m-0"><code class="hljs text-sm leading-relaxed">{@html cHighlightedHtml}</code></pre>
</div>

<style>
  .code-preview-wrapper {
    overscroll-behavior: contain;
  }

  .code-preview-wrapper pre {
    background: transparent;
    padding: 0;
  }

  .code-preview-wrapper code {
    background: transparent;
    display: block;
    padding: 0.5rem;
  }

  :global(.dark) .code-preview-wrapper {
    border-color: color-mix(in oklch, var(--border) 20%, transparent);
  }
</style>
