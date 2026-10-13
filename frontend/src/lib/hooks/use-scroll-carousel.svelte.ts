export function fUseScrollCarousel() {
  let vCanScrollLeft = $state(false);
  let vCanScrollRight = $state(false);
  let vScrollContainer = $state<HTMLDivElement | undefined>();

  function fScrollToCenter(pElement: HTMLElement) {
    if (!vScrollContainer) return;

    const cContainerRect = vScrollContainer.getBoundingClientRect();
    const cElementRect = pElement.getBoundingClientRect();

    const cElementCenter = cElementRect.left + cElementRect.width / 2;
    const cContainerCenter = cContainerRect.left + cContainerRect.width / 2;
    const cScrollOffset = cElementCenter - cContainerCenter;

    vScrollContainer.scrollBy({ left: cScrollOffset, behavior: 'smooth' });
  }

  function fScrollLeft() {
    if (!vScrollContainer) return;
    vScrollContainer.scrollBy({ left: -250, behavior: 'smooth' });
  }

  function fScrollRight() {
    if (!vScrollContainer) return;
    vScrollContainer.scrollBy({ left: 250, behavior: 'smooth' });
  }

  function fUpdateScrollButtons() {
    if (!vScrollContainer) return;

    const {
      scrollLeft: cSl,
      scrollWidth: cScrollWidth,
      clientWidth: cClientWidth
    } = vScrollContainer;
    vCanScrollLeft = cSl > 0;
    vCanScrollRight = cSl < cScrollWidth - cClientWidth - 1;
  }

  $effect(() => {
    if (vScrollContainer) {
      fUpdateScrollButtons();
    }
  });

  return {
    get canScrollLeft() {
      return vCanScrollLeft;
    },
    get canScrollRight() {
      return vCanScrollRight;
    },
    get scrollContainer() {
      return vScrollContainer;
    },
    set scrollContainer(pEl: HTMLDivElement | undefined) {
      vScrollContainer = pEl;
    },
    scrollToCenter: fScrollToCenter,
    scrollLeft: fScrollLeft,
    scrollRight: fScrollRight,
    updateScrollButtons: fUpdateScrollButtons
  };
}
