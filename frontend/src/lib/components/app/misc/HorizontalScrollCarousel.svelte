<script lang="ts">
  import { fTranslate } from '../../../i18n';

  import { cIconClassDefault } from '$lib/constants/css-classes';
  import { ChevronLeft, ChevronRight } from '@lucide/svelte';
  import type { Snippet } from 'svelte';

  interface Props {
    class?: string;
    children?: Snippet;
    gapSize?: string;
    onScrollableChange?: (pIsScrollable: boolean) => void;
  }

  let {
    class: vClassName = '',
    children: vChildren,
    gapSize: vGapSize = '3',
    onScrollableChange: vOnScrollableChange
  }: Props = $props();

  let vCanScrollLeft = $state(false);
  let vCanScrollRight = $state(false);
  let vScrollContainer: HTMLDivElement | undefined = $state();

  function fScrollLeft(pEvent?: MouseEvent) {
    pEvent?.stopPropagation();
    pEvent?.preventDefault();

    if (!vScrollContainer) return;

    vScrollContainer.scrollBy({ left: vScrollContainer.clientWidth * -0.67, behavior: 'smooth' });
  }

  function fScrollRight(pEvent?: MouseEvent) {
    pEvent?.stopPropagation();
    pEvent?.preventDefault();

    if (!vScrollContainer) return;

    vScrollContainer.scrollBy({ left: vScrollContainer.clientWidth * 0.67, behavior: 'smooth' });
  }

  function fUpdateScrollButtons() {
    if (!vScrollContainer) return;

    const {
      scrollLeft: cScrollLeft,
      scrollWidth: cScrollWidth,
      clientWidth: cClientWidth
    } = vScrollContainer;

    vCanScrollLeft = cScrollLeft > 0;
    vCanScrollRight = cScrollLeft < cScrollWidth - cClientWidth - 1;

    const cIsScrollable = cScrollWidth > cClientWidth;
    vOnScrollableChange?.(cIsScrollable);
  }

  export function fResetScroll() {
    if (vScrollContainer) {
      vScrollContainer.scrollLeft = 0;
      setTimeout(() => {
        fUpdateScrollButtons();
      }, 0);
    }
  }

  $effect(() => {
    if (!vScrollContainer) return;

    const cObserver = new ResizeObserver(() => fUpdateScrollButtons());
    cObserver.observe(vScrollContainer);

    return () => cObserver.disconnect();
  });
</script>

<div class="relative {vClassName}">
  <button
    class="absolute top-1/2 left-4 z-10 flex h-6 w-6 -translate-y-1/2 items-center justify-center rounded-full bg-background/25 shadow-md backdrop-blur-xs transition-opacity hover:bg-background/45 disabled:pointer-events-none disabled:opacity-0"
    onclick={fScrollLeft}
    disabled={!vCanScrollLeft}
    aria-label={fTranslate('text_35dfa1045cb2')}
  >
    <ChevronLeft class={cIconClassDefault} />
  </button>

  <div
    class="scrollbar-hide flex items-start gap-{vGapSize} overflow-x-auto"
    bind:this={vScrollContainer}
    onscroll={fUpdateScrollButtons}
  >
    {@render vChildren?.()}
  </div>

  <button
    class="absolute top-1/2 right-4 z-10 flex h-6 w-6 -translate-y-1/2 items-center justify-center rounded-full bg-background/25 shadow-md backdrop-blur-xs transition-opacity hover:bg-background/45 disabled:pointer-events-none disabled:opacity-0"
    onclick={fScrollRight}
    disabled={!vCanScrollRight}
    aria-label={fTranslate('text_366bd578350a')}
  >
    <ChevronRight class={cIconClassDefault} />
  </button>
</div>
