import { cAutoScrollAtBottomThreshold, cAutoScrollInterval } from '$lib/constants';

export interface AutoScrollOptions {
  disabled?: boolean;
}

/**
 * Creates an auto-scroll controller for a scrollable container.
 *
 * Features:
 * - Auto-scrolls to bottom during streaming/loading
 * - Stops auto-scroll when user manually scrolls up
 * - Resumes auto-scroll when user scrolls back to bottom
 */
export class AutoScrollController {
  private vAutoScrollEnabled = $state(true);
  private vUserScrolledUp = $state(false);
  private vLastScrollTop = $state(0);
  private vScrollInterval: ReturnType<typeof setInterval> | undefined;
  private vContainer: HTMLElement | undefined;
  private vDisabled: boolean;
  private vMutationObserver: MutationObserver | null = null;
  private vRafPending = false;
  private vObserverEnabled = false;
  constructor(pOptions: AutoScrollOptions = {}) {
    this.vDisabled = pOptions.disabled ?? false;
  }

  get mAutoScrollEnabled(): boolean {
    return this.vAutoScrollEnabled;
  }

  get mUserScrolledUp(): boolean {
    return this.vUserScrolledUp;
  }

  /**
   * Binds the controller to a scrollable container element.
   */
  mSetContainer(pContainer: HTMLElement | undefined): void {
    this.mDoStopObserving();
    this.vContainer = pContainer;

    if (this.vObserverEnabled && pContainer && !this.vDisabled) {
      this.mDoStartObserving();
    }
  }

  /**
   * Updates the disabled state.
   */
  mSetDisabled(pDisabled: boolean): void {
    if (this.vDisabled === pDisabled) return;
    this.vDisabled = pDisabled;
    if (pDisabled) {
      this.vAutoScrollEnabled = false;
      this.mStopInterval();
      this.mDoStopObserving();
    } else if (this.vObserverEnabled && this.vContainer && !this.vMutationObserver) {
      this.mDoStartObserving();
    }
  }

  /**
   * Handles scroll events to detect user scroll direction and toggle auto-scroll.
   */
  mHandleScroll(): void {
    if (this.vDisabled || !this.vContainer) return;

    const {
      scrollTop: cScrollTop,
      scrollHeight: cScrollHeight,
      clientHeight: cClientHeight
    } = this.vContainer;
    const cDistanceFromBottom = cScrollHeight - cClientHeight - cScrollTop;
    const cIsScrollingUp = cScrollTop < this.vLastScrollTop;
    const cIsAtBottom = cDistanceFromBottom < cAutoScrollAtBottomThreshold;

    if (cIsScrollingUp && !cIsAtBottom) {
      this.vUserScrolledUp = true;
      this.vAutoScrollEnabled = false;
    } else if (cIsAtBottom && this.vUserScrolledUp) {
      this.vUserScrolledUp = false;
      this.vAutoScrollEnabled = true;
    }

    this.vLastScrollTop = cScrollTop;
  }

  /**
   * Scrolls the container to the bottom instantly.
   */
  mScrollToBottom(): void {
    if (this.vDisabled || !this.vContainer) return;
    this.vContainer.scrollTop = this.vContainer.scrollHeight;
  }

  /**
   * Enables auto-scroll (e.g., when user sends a message).
   */
  mEnable(): void {
    if (this.vDisabled) return;
    this.vUserScrolledUp = false;
    this.vAutoScrollEnabled = true;
  }

  /**
   * Resets scroll state when switching conversations.
   */
  mResetScrollState(): void {
    this.vUserScrolledUp = false;
    this.vAutoScrollEnabled = !this.vDisabled;
    if (this.vContainer) {
      this.vLastScrollTop = this.vContainer.scrollTop;
    }
  }

  /**
   * Starts the auto-scroll interval for continuous scrolling during streaming.
   */
  mStartInterval(): void {
    if (this.vDisabled || this.vScrollInterval) return;

    this.vScrollInterval = setInterval(() => {
      this.mScrollToBottom();
    }, cAutoScrollInterval);
  }

  /**
   * Stops the auto-scroll interval.
   */
  mStopInterval(): void {
    if (this.vScrollInterval) {
      clearInterval(this.vScrollInterval);
      this.vScrollInterval = undefined;
    }
  }

  /**
   * Updates the auto-scroll interval based on streaming state.
   * Call this in a $effect to automatically manage the interval.
   */
  mUpdateInterval(pIsStreaming: boolean): void {
    if (this.vDisabled) {
      this.mStopInterval();
      return;
    }

    if (pIsStreaming && this.vAutoScrollEnabled) {
      if (!this.vScrollInterval) {
        this.mStartInterval();
      }
    } else {
      this.mStopInterval();
    }
  }

  /**
   * Cleans up resources. Call this in onDestroy or when the component unmounts.
   */
  mDestroy(): void {
    this.mStopInterval();
    this.mDoStopObserving();
  }

  /**
   * Starts a MutationObserver on the container that auto-scrolls to bottom
   * on content changes. More responsive than interval-based polling.
   */
  mStartObserving(): void {
    this.vObserverEnabled = true;

    if (this.vContainer && !this.vDisabled && !this.vMutationObserver) {
      this.mDoStartObserving();
    }
  }

  /**
   * Stops the MutationObserver.
   */
  mStopObserving(): void {
    this.vObserverEnabled = false;
    this.mDoStopObserving();
  }

  private mDoStartObserving(): void {
    if (!this.vContainer || this.vMutationObserver) return;

    this.vMutationObserver = new MutationObserver(() => {
      if (!this.vAutoScrollEnabled || this.vRafPending) return;
      this.vRafPending = true;
      requestAnimationFrame(() => {
        this.vRafPending = false;
        if (this.vAutoScrollEnabled && this.vContainer) {
          this.vContainer.scrollTop = this.vContainer.scrollHeight;
        }
      });
    });

    this.vMutationObserver.observe(this.vContainer, {
      childList: true,
      subtree: true,
      characterData: true
    });
  }

  private mDoStopObserving(): void {
    if (this.vMutationObserver) {
      this.vMutationObserver.disconnect();
      this.vMutationObserver = null;
    }
    this.vRafPending = false;
  }
}

/**
 * Creates a new AutoScrollController instance.
 */
export function fCreateAutoScrollController(
  pOptions: AutoScrollOptions = {}
): AutoScrollController {
  return new AutoScrollController(pOptions);
}
