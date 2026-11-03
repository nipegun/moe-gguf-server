<script lang="ts">
  import { fTranslate } from '../../../i18n';

  import { cIconClassDefault } from '$lib/constants/css-classes';
  import { Download } from '@lucide/svelte';
  import ZoomInIcon from '@lucide/svelte/icons/zoom-in';
  import ZoomOutIcon from '@lucide/svelte/icons/zoom-out';
  import RotateCcwIcon from '@lucide/svelte/icons/rotate-ccw';

  interface Props {
    scale: number;
    svgHtml: string;
    onZoomIn: () => void;
    onZoomOut: () => void;
    onResetView: () => void;
  }

  let {
    scale: vScale,
    svgHtml: vSvgHtml,
    onZoomIn: vOnZoomIn,
    onZoomOut: vOnZoomOut,
    onResetView: vOnResetView
  }: Props = $props();

  function fDownloadSvg() {
    if (!vSvgHtml) return;
    const cBlob = new Blob([vSvgHtml], { type: 'image/svg+xml' });
    const cUrl = URL.createObjectURL(cBlob);
    const cA = document.createElement('a');
    cA.href = cUrl;
    cA.download = 'diagram.svg';
    cA.click();
    URL.revokeObjectURL(cUrl);
  }
</script>

<div
  class="mermaid-preview-controls absolute bottom-8 flex shrink-0 items-center justify-center p-3"
>
  <div class="mermaid-preview-controls-inner flex items-center gap-1 rounded-lg bg-muted p-1">
    <button
      class="mermaid-preview-btn flex h-8 w-8 cursor-pointer items-center justify-center rounded-md border-0 bg-transparent text-foreground transition-colors hover:bg-muted-foreground/15 active:bg-muted-foreground/25"
      onclick={vOnZoomOut}
      title={fTranslate('text_bc7b631a689b')}
      aria-label={fTranslate('text_bc7b631a689b')}
    >
      <ZoomOutIcon class="mermaid-preview-btn-icon {cIconClassDefault}" />
    </button>
    <span
      class="mermaid-preview-zoom-label min-w-[3.5rem] px-0.5 text-center text-xs font-medium text-muted-foreground tabular-nums select-none"
      >{Math.round(vScale * 100)}%</span
    >
    <button
      class="mermaid-preview-btn flex h-8 w-8 cursor-pointer items-center justify-center rounded-md border-0 bg-transparent text-foreground transition-colors hover:bg-muted-foreground/15 active:bg-muted-foreground/25"
      onclick={vOnZoomIn}
      title={fTranslate('text_0e47f09a748f')}
      aria-label={fTranslate('text_0e47f09a748f')}
    >
      <ZoomInIcon class="mermaid-preview-btn-icon {cIconClassDefault}" />
    </button>
    <div class="mermaid-preview-controls-separator mx-1 h-5 w-px bg-border/50"></div>

    <button
      class="mermaid-preview-btn flex h-8 w-8 cursor-pointer items-center justify-center rounded-md border-0 bg-transparent text-foreground transition-colors hover:bg-muted-foreground/15 active:bg-muted-foreground/25"
      onclick={vOnResetView}
      title={fTranslate('text_ce89ae822e8b')}
      aria-label={fTranslate('text_ce89ae822e8b')}
    >
      <RotateCcwIcon class="mermaid-preview-btn-icon {cIconClassDefault}" />
    </button>
    <div class="mermaid-preview-controls-separator mx-1 h-5 w-px bg-border/50"></div>

    <button
      class="mermaid-preview-btn flex h-8 w-8 cursor-pointer items-center justify-center rounded-md border-0 bg-transparent text-foreground transition-colors hover:bg-muted-foreground/15 active:bg-muted-foreground/25"
      onclick={fDownloadSvg}
      title={fTranslate('text_91aec62280a3')}
      aria-label={fTranslate('text_91aec62280a3')}
    >
      <Download class="mermaid-preview-btn-icon {cIconClassDefault}" />
    </button>
  </div>
</div>
