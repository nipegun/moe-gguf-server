<script lang="ts">
  import MermaidPreviewControls from './MermaidPreviewControls.svelte';
  import { fMountSvgShadow } from '$lib/utils/svg-shadow';
  import { cSvgDialogShadowStyle } from '$lib/constants';

  interface Props {
    svgHtml: string;
  }

  let { svgHtml: vSvgHtml }: Props = $props();

  let vSvgHost = $state<HTMLDivElement | null>(null);

  // Re-mount on every svgHtml change so a live streaming svg keeps rendering while zoomed
  $effect(() => {
    if (vSvgHost) fMountSvgShadow(vSvgHost, vSvgHtml, cSvgDialogShadowStyle);
  });

  // Zoom and pan state
  let vScale = $state(1);
  let vTranslateX = $state(0);
  let vTranslateY = $state(0);
  let vIsDragging = $state(false);
  const dContainerRef = { current: null as HTMLDivElement | null };

  // Drag start position
  let vDragStartX = 0;
  let vDragStartY = 0;
  let vDragStartTranslateX = 0;
  let vDragStartTranslateY = 0;

  const cMinScale = 0.1;
  const cMaxScale = 10;
  const cZoomStep = 0.15;

  function fResetView() {
    vScale = 1;
    vTranslateX = 0;
    vTranslateY = 0;
  }

  function fZoomIn() {
    vScale = Math.min(vScale + cZoomStep, cMaxScale);
  }

  function fZoomOut() {
    vScale = Math.max(vScale - cZoomStep, cMinScale);
  }

  function fHandleWheel(pEvent: WheelEvent) {
    pEvent.preventDefault();

    const cDelta = pEvent.deltaY > 0 ? -cZoomStep : cZoomStep;
    vScale = Math.min(Math.max(vScale + cDelta, cMinScale), cMaxScale);
  }

  // Imperatively attach a non-passive wheel listener so preventDefault() actually works
  // (Svelte 5 wheel listeners are passive by default, making preventDefault() a no-op)
  $effect(() => {
    const cEl = dContainerRef.current;
    if (!cEl) return;

    function fOnWheel(pE: WheelEvent) {
      fHandleWheel(pE);
    }

    cEl.addEventListener('wheel', fOnWheel, { passive: false });
    return () => cEl.removeEventListener('wheel', fOnWheel);
  });

  function fHandlePointerDown(pEvent: PointerEvent) {
    if (pEvent.button !== 0 && pEvent.pointerType === 'mouse') return;

    vIsDragging = true;
    vDragStartX = pEvent.clientX;
    vDragStartY = pEvent.clientY;
    vDragStartTranslateX = vTranslateX;
    vDragStartTranslateY = vTranslateY;

    (pEvent.currentTarget as HTMLElement).setPointerCapture(pEvent.pointerId);
  }

  function fHandlePointerMove(pEvent: PointerEvent) {
    if (!vIsDragging) return;

    vTranslateX = vDragStartTranslateX + (pEvent.clientX - vDragStartX);
    vTranslateY = vDragStartTranslateY + (pEvent.clientY - vDragStartY);
  }

  function fHandlePointerUp() {
    vIsDragging = false;
  }
</script>

<div
  bind:this={dContainerRef.current}
  class="mermaid-preview relative flex items-center justify-center overflow-hidden bg-muted/20"
>
  <!-- svelte-ignore a11y_no_static_element_interactions -->
  <div
    class="mermaid-preview-diagram transform-origin-center inline-block min-h-fit min-w-fit will-change-transform {vIsDragging &&
      'select-none'}"
    style="transform: translate({vTranslateX}px, {vTranslateY}px) scale({vScale}); cursor: {vIsDragging
      ? 'grabbing'
      : 'grab'};"
    onpointerdown={fHandlePointerDown}
    onpointermove={fHandlePointerMove}
    onpointerup={fHandlePointerUp}
    onpointerleave={fHandlePointerUp}
  >
    <div bind:this={vSvgHost}></div>
  </div>

  <MermaidPreviewControls
    scale={vScale}
    svgHtml={vSvgHtml}
    onZoomIn={fZoomIn}
    onZoomOut={fZoomOut}
    onResetView={fResetView}
  />
</div>
